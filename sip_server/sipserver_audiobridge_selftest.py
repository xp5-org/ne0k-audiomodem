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

"""Self-test for the sipserver.py `audiobridge` line: a G.711 caller and a G.711 callee on
loopback, the server between them. RTP must arrive byte-for-byte, both ways, and both log
lines the NE0K fax testlist waits for must appear. Exit 0 = pass.
"""
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sipcore as sc  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (("  -- " + str(detail)) if detail and not cond else ""))


def pkt(seq, payload, pt=0):
    return struct.pack("!BBHII", 0x80, pt, seq, seq * 160, 0xABCD) + payload


def main():
    tmp = tempfile.mkdtemp(prefix="audiobridge_selftest_")
    callee_sip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    callee_sip.bind(("127.0.0.1", 0))
    callee_rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    callee_rtp.bind(("127.0.0.1", 0))
    cfg = {"listen": {"host": "127.0.0.1", "port": 0}, "rtp_ports": [31400, 31600],
           "ready_file": os.path.join(tmp, "ready"),
           "lines": {"to_b": {"type": "audiobridge", "target": "127.0.0.1:%d" % callee_sip.getsockname()[1],
                              "ring_timeout_s": 5}},
           "numbers": {"2002": {"lines": ["to_b"]}}}
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

        csip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        csip.bind(("127.0.0.1", 0))
        crtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        crtp.bind(("127.0.0.1", 0))
        hp = "127.0.0.1:%d" % csip.getsockname()[1]
        frm, to, cid = "<sip:1001@127.0.0.1>;tag=abc", "<sip:2002@127.0.0.1>", "cid-audiobridge"
        uri = "sip:2002@%s:%d" % srv
        sdp = sc.build_sdp_g711("127.0.0.1", crtp.getsockname()[1], (0,), 101)
        csip.sendto(sc.build_request("INVITE", uri, hp, frm, to, cid, 1, contact="<sip:1001@%s>" % hp, body=sdp), srv)

        # callee: answer the server's INVITE with PCMU
        callee_sip.settimeout(5)
        data, saddr = callee_sip.recvfrom(4096)
        inv = sc.parse(data)
        check("callee INVITE offers PCMU, not x-modem-bytes",
              inv.is_request and inv.method == "INVITE" and "PCMU/8000" in inv.body and sc.DATA_CODEC not in inv.body.lower(),
              inv.body)
        info = sc.parse_sdp(inv.body)
        callee_peer = (info["addr"], info["port"])
        csdp = sc.build_sdp_g711("127.0.0.1", callee_rtp.getsockname()[1], (0,), 101)
        callee_sip.sendto(sc.build_response(inv, 180, "Ringing", to_tag="ct1"), saddr)
        callee_sip.sendto(sc.build_response(inv, 200, "OK", to_tag="ct1", extra=["Contact: <sip:2002@127.0.0.1:%d>" % callee_sip.getsockname()[1]], body=csdp), saddr)

        final, rtp_to = None, None
        csip.settimeout(5)
        while final is None:
            m = sc.parse(csip.recvfrom(4096)[0])
            if not m.is_request and m.status >= 200:
                final = m
        check("caller got 200", final.status == 200, final.status)
        crtp_peer = (sc.parse_sdp(final.body)["addr"], sc.parse_sdp(final.body)["port"])
        csip.sendto(sc.build_request("ACK", uri, hp, frm, final.get("to"), cid, 1), srv)

        # audio both ways, untouched
        payloads = [bytes([(i * 7 + j) & 0xFF for j in range(160)]) for i in range(10)]
        callee_rtp.settimeout(2)
        crtp.settimeout(2)
        got_b, got_a = [], []
        for i, p in enumerate(payloads):
            crtp.sendto(pkt(i, p), crtp_peer)
            callee_rtp.sendto(pkt(i, p[::-1]), callee_peer)
        for _ in payloads:
            try:
                got_b.append(callee_rtp.recvfrom(4096)[0][12:])
                got_a.append(crtp.recvfrom(4096)[0][12:])
            except socket.timeout:
                break
        check("caller -> callee payloads identical", got_b == payloads, len(got_b))
        check("callee -> caller payloads identical", got_a == [p[::-1] for p in payloads], len(got_a))

        csip.sendto(sc.build_request("BYE", uri, hp, frm, final.get("to"), cid, 2), srv)
        callee_sip.settimeout(5)
        bye = None
        while bye is None:
            m = sc.parse(callee_sip.recvfrom(4096)[0])
            if m.is_request and m.method == "BYE":
                bye = m
                callee_sip.sendto(sc.build_response(m, 200, "OK"), saddr)
        check("callee got BYE when the caller hung up", bye is not None)
        time.sleep(0.5)
        log = open(log_path).read()
        check("log: audiobridge: target answered", "audiobridge: target answered" in log, log)
        check("log: audiobridge: call ended", "audiobridge: call ended" in log, log)
    finally:
        proc.terminate()
        proc.wait(5)
    print("%d/%d checks passed" % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
