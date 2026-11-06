#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 XP5.ORG
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""meshdialer.py -- wire N audio-modem nodes into a full mesh of SIP calls (one sipbridge per leg).

Each node n has two PulseAudio sinks:
    PREFIX_n       the modem's TX; its .monitor is what the dialer sends into every call
    PREFIX_n_rx    the modem's RX; every call leg plays into it, and PulseAudio sums the streams

(--server --conference N replaces all of this with ONE call per node to a hosted conference bridge.)

For every pair (i < j) there is ONE call, which carries both directions: node j answers, node i dials.
A node with k peers therefore hears the sum of k calls on its RX sink and its TX goes to all k --
the shared "air" of a mesh.  No mixer module is needed.

    meshdialer.py --nodes A,B,C --hosts A=10.0.0.1,B=10.0.0.2,C=10.0.0.3 --local A,B --out DIR

With --server HOST:PORT every call goes through a SIP server that rings REGISTERed users (sipserver.py
"audiobridge" lines, see sipserver_mesh_external.json): pair k (in A-B, A-C, B-C order) is number 2001+k, the
answerer REGISTERs as user 1001+k and the caller dials 2001+k.  --hosts then only gives each node's public
address for RTP (SDP).

Run it on each machine with --local naming the nodes that live there (default: all, on 127.0.0.1).
The answering node's host:port is what the caller dials, so --hosts must be reachable from the callers.
Ports: SIP = --sip-base + 10*j + i on the answerer j, RTP = --rtp-base + 4*(10*j + i) (+2 for the caller).

stdout (stable lines for test_filecontains):
    LEG pair=A-B role=answer|call READY      sipbridge bound
    ANSWERERS READY / GO                     only with --go-file: answerers bound / go file seen, callers start now
    MESH UP calls=N/N                        every call this process owns is up
    MESH DOWN reason=...                     a leg died or a signal arrived
"""
from __future__ import annotations

import argparse
import itertools
import os
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BRIDGE = os.path.join(HERE, "..", "..", "audiomodem_linux", "sipbridge.py")


def log(msg: str) -> None:
    print(msg, flush=True)


class Leg:
    def __init__(self, pair, role, argv, logfile):
        self.pair, self.role = pair, role
        self.ready, self.up, self.ended = threading.Event(), threading.Event(), threading.Event()
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.logf = open(logfile, "w")
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self.logf.write(line)
            self.logf.flush()
            if line.startswith("READY sip="):
                self.ready.set()
                log(f"LEG pair={self.pair} role={self.role} READY")
            elif line.startswith("CALL UP"):
                self.up.set()
            elif line.startswith("CALL ENDED"):
                self.ended.set()
        self.ended.set()

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--nodes", default="A,B,C", help="comma-separated node names, order = index")
    p.add_argument("--hosts", default="", help="NAME=IP,... (default 127.0.0.1 for every node)")
    p.add_argument("--local", default="", help="nodes started on THIS machine (default: all)")
    p.add_argument("--server", help="route every call through this SIP server HOST:PORT instead of node-to-node")
    p.add_argument("--prefix", default="mesh", help="sink prefix: PREFIX_<node>, PREFIX_<node>_rx")
    p.add_argument("--sip-base", type=int, default=15100)
    p.add_argument("--rtp-base", type=int, default=41000)
    p.add_argument("--bridge", default=DEFAULT_BRIDGE)
    p.add_argument("--python", default=sys.executable, help="interpreter with numpy for sipbridge")
    p.add_argument("--out", required=True, help="directory for per-leg logs")
    p.add_argument("--max-s", type=float, default=3000)
    p.add_argument("--up-timeout", type=float, default=60)
    p.add_argument("--ready-timeout", type=float, default=45,
                   help="how long to wait for each local answerer leg's own READY. Keep this comfortably "
                        "above sipbridge's --reg-timeout (ready-timeout - 10, passed through below) so a "
                        "merely-slow external sipserver registration shows up in that leg's log as "
                        "'register failed' instead of this process giving up first and reporting the "
                        "less useful 'not ready'.")
    p.add_argument("--go-file", help="after this process's answerers are READY print ANSWERERS READY and wait for this file to "
                                     "exist before placing the callers (distributed test group: the go signal)")
    p.add_argument("--go-timeout", type=float, default=300)
    p.add_argument("--conference", metavar="NUMBER",
                   help="with --server: ONE call per local node to this conference number on the SIP server (a hosted "
                        "conference bridge, sipserver.py line type `conference`): 64 kbit/s up + 64 down per node, the "
                        "server mixes everyone else's audio into each node's return. No answerers, no pair calls.")
    p.add_argument("--stats-dir", help="have every leg write live counters to DIR/<pair>_<role>.json (read by meshgui.py)")
    a = p.parse_args(argv)

    nodes = a.nodes.split(",")
    hosts = {n: "127.0.0.1" for n in nodes}
    hosts.update(dict(kv.split("=", 1) for kv in a.hosts.split(",") if kv))
    local = set(a.local.split(",")) if a.local else set(nodes)
    os.makedirs(a.out, exist_ok=True)
    if a.stats_dir:
        os.makedirs(a.stats_dir, exist_ok=True)

    legs: list[Leg] = []
    quit_ = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: quit_.set())
    signal.signal(signal.SIGINT, lambda *_: quit_.set())

    def port(j, i):                       # i < j; keyed by the answerer j
        return 10 * j + i

    pair_no = {pr: k for k, pr in enumerate(itertools.combinations(range(len(nodes)), 2))}

    def bridge_argv(mode, node, peer_idx_pair, extra):
        st = ["--stats-file", f"{a.stats_dir}/{nodes[peer_idx_pair[0]]}-{nodes[peer_idx_pair[1]]}_{mode}.json"] if a.stats_dir else []
        return [a.python, a.bridge, mode, "--source", f"{a.prefix}_{node}.monitor",
                "--sink", f"{a.prefix}_{node}_rx", "--max-s", str(a.max_s), *st, *extra]

    pairs = [(i, j) for i, j in itertools.combinations(range(len(nodes)), 2)]
    expected = 0
    if a.conference and not a.server:
        p.error("--conference needs --server")
    try:
        if a.conference:
            # one SIP line per client: each local node dials the conference number, the server bridges the room
            log("ANSWERERS READY")                       # nothing to answer; keeps the group go-signal protocol
            if a.go_file:
                t_go = time.time() + a.go_timeout
                while not os.path.exists(a.go_file):
                    if quit_.is_set() or time.time() > t_go:
                        log("MESH DOWN reason=no go signal")
                        return 1
                    time.sleep(0.1)
                log("GO")
            for i, n in enumerate(nodes):
                if n in local:
                    h = hosts[n]
                    bind = "127.0.0.1" if h == "127.0.0.1" else "0.0.0.0"
                    st = ["--stats-file", f"{a.stats_dir}/{n}-conference_call.json"] if a.stats_dir else []
                    legs.append(Leg(f"{n}-conference", "call", [
                        a.python, a.bridge, "call", "--source", f"{a.prefix}_{n}.monitor",
                        "--sink", f"{a.prefix}_{n}_rx", "--max-s", str(a.max_s), *st,
                        "--to", a.server, "--number", a.conference, "--user", f"{a.prefix}{n}",
                        "--bind", bind, "--advertise", h, "--rtp-port", str(a.rtp_base + 4 * i + 2)],
                        f"{a.out}/leg_{n}-conference_call.log"))
                    expected += 1
            deadline = time.time() + a.up_timeout
            while time.time() < deadline and not quit_.is_set() and not all(l.up.is_set() for l in legs):
                time.sleep(0.25)
            n_up = sum(l.up.is_set() for l in legs)
            log(f"MESH UP calls={n_up}/{expected}" if n_up == expected else f"MESH PARTIAL calls={n_up}/{expected}")
            while not quit_.is_set() and not any(l.ended.is_set() for l in legs):
                time.sleep(0.5)
            log("MESH DOWN reason=" + ("signal" if quit_.is_set() else "leg ended"))
            return 0 if n_up == expected else 1
        # answerers first (higher index of each pair)
        for i, j in pairs:
            ni, nj = nodes[i], nodes[j]
            if nj in local:
                k = port(j, i)
                h = hosts[nj]
                bind = "127.0.0.1" if h == "127.0.0.1" else "0.0.0.0"
                legs.append(Leg(f"{ni}-{nj}", "answer", bridge_argv(
                    "answer", nj, (i, j),
                    ["--listen", f"{bind}:{a.sip_base + k}", "--bind", bind, "--advertise", h,
                     "--rtp-port", str(a.rtp_base + 4 * k),
                     *(["--register", a.server, "--user", str(1001 + pair_no[(i, j)]),
                        "--reg-timeout", str(max(10, a.ready_timeout - 10))] if a.server else [])]),
                    f"{a.out}/leg_{ni}-{nj}_answer.log"))
                expected += 1
        for leg in legs:
            if not leg.ready.wait(a.ready_timeout):
                log(f"MESH DOWN reason=answerer {leg.pair} not ready")
                return 1
        if a.go_file:
            log("ANSWERERS READY")
            t_go = time.time() + a.go_timeout
            while not os.path.exists(a.go_file):
                # A sibling leg ending here is THAT leg's problem alone -- it must not pull down every
                # other leg on this node (and with it, their SIP registrations) before the go signal
                # arrives. The other legs may still be exactly what a peer host's node needs to complete
                # its own calls, so only a real quit signal or the go-timeout itself aborts the wait.
                if quit_.is_set() or time.time() > t_go:
                    log("MESH DOWN reason=no go signal")
                    return 1
                time.sleep(0.1)
            log("GO")
        # then callers (lower index)
        time.sleep(0.5)
        for i, j in pairs:
            ni, nj = nodes[i], nodes[j]
            if ni in local:
                k = port(j, i)
                h = hosts[ni]
                bind = "127.0.0.1" if h == "127.0.0.1" else "0.0.0.0"
                legs.append(Leg(f"{ni}-{nj}", "call", bridge_argv(
                    "call", ni, (i, j),
                    ["--to", a.server or f"{hosts[nj]}:{a.sip_base + k}",
                     "--number", str(2001 + pair_no[(i, j)]) if a.server else str(1000 + j),
                     "--user", f"{a.prefix}{ni}", "--bind", bind, "--advertise", h,
                     "--rtp-port", str(a.rtp_base + 4 * k + 2)]), f"{a.out}/leg_{ni}-{nj}_call.log"))
                expected += 1
        deadline = time.time() + a.up_timeout
        while time.time() < deadline and not quit_.is_set():
            if all(l.up.is_set() for l in legs):
                break
            # As above: one leg (even a local caller) ending here must not abort the wait and tear
            # down every leg on this node -- the answerer legs among them may still be what a peer
            # host needs. Just let the deadline run out; n_up below reports what actually came up.
            time.sleep(0.25)
        n_up = sum(l.up.is_set() for l in legs)
        # callers whose peer lives on another host come up when that host's dialer starts, so a
        # partial result is reported, not hidden
        log(f"MESH UP calls={n_up}/{expected}" if n_up == expected else f"MESH PARTIAL calls={n_up}/{expected}")
        while not quit_.is_set() and not any(l.ended.is_set() for l in legs):
            time.sleep(0.5)
        log("MESH DOWN reason=" + ("signal" if quit_.is_set() else "leg ended"))
        return 0 if n_up == expected else 1
    finally:
        for l in legs:
            l.stop()


if __name__ == "__main__":
    sys.exit(main())
