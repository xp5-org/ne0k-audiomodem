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

"""Self-test for the sipserver.py `conference` line: three G.711 callers dial ONE number and each sends a different
tone. Every caller must hear the OTHER two tones (the server mixes them) and not its own, at 50 packets/s of 160
bytes, a caller that leaves must drop out of the mix without disturbing the rest, and a 2-member room (max_members)
must answer a third caller busy. Exit 0 = pass.
"""
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sipcore as sc  # noqa: E402
import sipserver as ss  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (("  -- " + str(detail)) if detail and not cond else ""))


def tone_frame(freq, n, amp=6000):
    enc = ss.G711_ENC["ulaw"]
    return bytes(enc[(int(amp * math.sin(2 * math.pi * freq * (n * 160 + i) / 8000)) >> 2) + 8192] for i in range(160))


def level(payloads, freq):
    """Amplitude of `freq` in the decoded ulaw payloads (single-bin DFT)."""
    dec = ss.G711_DEC["ulaw"]
    x = [dec[b] for p in payloads for b in p]
    re = sum(v * math.cos(2 * math.pi * freq * i / 8000) for i, v in enumerate(x))
    im = sum(v * math.sin(2 * math.pi * freq * i / 8000) for i, v in enumerate(x))
    return 2 * math.hypot(re, im) / max(1, len(x))


class Caller:
    def __init__(self, user, srv, number, freq):
        self.user, self.srv, self.freq, self.number = user, srv, freq, number
        self.sip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sip.bind(("127.0.0.1", 0))
        self.rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rtp.bind(("127.0.0.1", 0))
        self.rtp.setblocking(False)
        self.hp = "127.0.0.1:%d" % self.sip.getsockname()[1]
        self.cid = "conf-%s" % user
        self.frm, self.to = "<sip:%s@127.0.0.1>;tag=%s" % (user, user), "<sip:%s@127.0.0.1>" % number
        self.uri = "sip:%s@%s:%d" % (number, srv[0], srv[1])
        self.seq, self.n, self.peer, self.final, self.heard = 0, 0, None, None, []

    def invite(self):
        sdp = sc.build_sdp_g711("127.0.0.1", self.rtp.getsockname()[1], (0,), 101)
        self.sip.sendto(sc.build_request("INVITE", self.uri, self.hp, self.frm, self.to, self.cid, 1,
                                         contact="<sip:%s@%s>" % (self.user, self.hp), body=sdp), self.srv)
        self.sip.settimeout(5)
        while True:
            m = sc.parse(self.sip.recvfrom(4096)[0])
            if not m.is_request and m.status >= 200:
                self.final = m
                break
        if m.status == 200:
            info = sc.parse_sdp(m.body)
            self.peer = (info["addr"], info["port"])
            self.sip.sendto(sc.build_request("ACK", self.uri, self.hp, self.frm, m.get("to"), self.cid, 1), self.srv)
        return m.status

    def pump(self):
        """Send one 20 ms frame of this caller's tone, drain whatever the server sent."""
        pkt = ss.rtp_packet(0, self.seq, self.n * 160, 0x1000 + hash(self.user) % 999, tone_frame(self.freq, self.n))
        self.rtp.sendto(pkt, self.peer)
        self.seq += 1
        self.n += 1
        try:
            while True:
                self.heard.append(self.rtp.recvfrom(4096)[0])
        except BlockingIOError:
            pass

    def bye(self):
        self.sip.sendto(sc.build_request("BYE", self.uri, self.hp, self.frm, self.final.get("to"), self.cid, 2), self.srv)


def run(callers, seconds):
    t_end, nxt = time.monotonic() + seconds, time.monotonic()
    while time.monotonic() < t_end:
        for c in callers:
            c.pump()
        nxt += 0.02
        time.sleep(max(0, nxt - time.monotonic()))


def main():
    tmp = tempfile.mkdtemp(prefix="conference_selftest_")
    cfg = {"listen": {"host": "127.0.0.1", "port": 0}, "rtp_ports": [31700, 31900],
           "ready_file": os.path.join(tmp, "ready"),
           "lines": {"room": {"type": "conference"}, "small": {"type": "conference", "max_members": 2}},
           "numbers": {"3000": {"lines": ["room"]}, "3001": {"lines": ["small"]}}}
    cfg_path, log_path = os.path.join(tmp, "cfg.json"), os.path.join(tmp, "server.log")
    json.dump(cfg, open(cfg_path, "w"))
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "sipserver.py"), "--config", cfg_path],
                            stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(50):
            if os.path.exists(cfg["ready_file"]):
                break
            time.sleep(0.1)
        else:
            print("server never became ready")
            return 1
        host, port = open(cfg["ready_file"]).read().strip().rsplit(":", 1)
        srv = (host, int(port))
        A, B, C = (Caller(u, srv, "3000", f) for u, f in (("1001", 500), ("1002", 1000), ("1003", 1700)))
        check("A, B, C all joined the SAME number (200 OK)", [c.invite() for c in (A, B, C)] == [200, 200, 200])
        run([A, B, C], 1.0)                              # everyone sending: playout buffers prime
        for c in (A, B, C):
            c.heard.clear()
        run([A, B, C], 2.5)                              # the measured window
        for c, others, own in ((A, (1000, 1700), 500), (B, (500, 1700), 1000), (C, (500, 1000), 1700)):
            pl = [p[12:] for p in c.heard]
            n = len(c.heard)
            check(f"{c.user}: ~50 RTP packets/s of 160 B (got {n} in 2.5 s)", 115 <= n <= 135 and all(len(p) == 160 for p in pl))
            lv = {f: level(pl[:100], f) for f in (500, 1000, 1700)}
            check(f"{c.user}: hears both other callers", all(lv[f] > 3000 for f in others), lv)
            check(f"{c.user}: does NOT hear itself", lv[own] < 0.1 * min(lv[f] for f in others), lv)
        # C leaves: A and B keep a 2-way room, C's tone disappears from the mix
        C.bye()
        run([A, B], 0.6)
        A.heard.clear()
        B.heard.clear()
        run([A, B], 1.5)
        la = {f: level([p[12:] for p in A.heard[10:60]], f) for f in (1000, 1700)}
        check("after C left: A still hears B, no longer C", la[1000] > 3000 and la[1700] < 0.1 * la[1000], la)
        # room capacity: max_members=2 -> the third caller gets busy
        D, E, F = (Caller(u, srv, "3001", 800) for u in ("1011", "1012", "1013"))
        st = [D.invite(), E.invite(), F.invite()]
        check("full room answers busy (200, 200, 486)", st == [200, 200, 486], st)
        for c in (A, B, D, E):
            c.bye()
        time.sleep(0.5)
        log = open(log_path).read()
        check("log: conference joined (members=3)", "conference room: joined (members=3)" in log, log)
        check("log: conference left", "conference room: left" in log, log)
    finally:
        proc.terminate()
        proc.wait(5)
    jitter_no_slip()
    print("%d/%d checks passed" % (sum(results), len(results)))
    return 0 if all(results) else 1


class _StubCall:
    def __init__(self, label):
        self.label, self.offer, self.stats, self.out = label, {"law": "ulaw", "data_pt": 0}, {"tx": 0}, []

    def event(self, *a, **k):
        pass

    def _emit(self, kind, pkt):
        self.out.append(pkt[12:])


def jitter_no_slip():
    """A delay spike in the middle of a burst must not re-time the rest of it (the old byte FIFO dropped the oldest
    frames on the catch-up burst: a -20 ms slip that kills an OFDM frame on the mesh).  Sender A sends a numbered
    ramp, 1 frame = its index; listener B must hear every frame that arrived in time at ITS OWN position."""
    ss.log = lambda *a: None
    line = ss.ConferenceLine("slip", {})
    A, B = _StubCall("A"), _StubCall("B")
    for c in (A, B):
        line.members[c] = ss.ConfMember(c)
    codes = [c for c in range(256) if c & 0x7F not in (0x7F,)]       # every u-law code but the two zeros
    rt = {c for c in codes if ss.G711_ENC["ulaw"][(ss.G711_DEC["ulaw"][c] >> 2) + 8192] == c}
    codes = sorted(rt)                                       # codes that survive decode -> mix -> encode
    tag = lambda k: bytes([codes[k // len(codes)], codes[k % len(codes)]]) * 80   # frame k, readable after the mix
    evs = []
    for k in range(400):
        delay = 0.04 + (0.13 if 200 <= k < 206 else 0.0)    # one 130 ms spike mid-stream
        evs.append((k * 0.02 + delay, A, ss.rtp_packet(0, k, k * 160, 1, tag(k))))
        evs.append((k * 0.02 + 0.04, B, ss.rtp_packet(0, k, k * 160, 2, bytes([0xFF]) * 160)))
    evs.sort(key=lambda e: e[0])
    i = 0
    for t in range(420):
        while i < len(evs) and evs[i][0] <= t * 0.02 + 0.005:
            line.on_audio(evs[i][1], evs[i][2])
            i += 1
        line._tick()
    idx = {tag(k)[:2]: k for k in range(400)}
    offsets = {j - idx[p[:2]] for j, p in enumerate(B.out) if p[:2] in idx and p[2:4] == p[:2]}
    check("jitter spike: no time slip (every frame heard at one fixed offset)", len(offsets) == 1, sorted(offsets))
    st = line.members[A].stats
    check("jitter spike: no overrun drop", st["overruns"] == 0 and st["skips"] == 0, st)


if __name__ == "__main__":
    sys.exit(main())
