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

"""SIP answering server + RTP relay for the modem-over-SIP test rig.

Run in the FOREGROUND under test_start_hostservice, which hands it a JSON
config (the testlist's own params) as `--config PATH` / $HOSTSERVICE_CONFIG.
Everything a test wants to vary is in that file -- which numbers exist, which
lines answer them, how many rings, what each line does, what latency the
media path gets -- so a testlist is a pure declaration:

  {
    "listen": {"host": "0.0.0.0", "port": 5060, "advertise_host": null},
    "rtp_ports": [20000, 20999],
    "seed": 1,                        # makes jitter/loss reproducible
    "log_data": true,                 # echo payload bytes into the log
    "events_file": "/path/events.jsonl",   # optional: one JSON event per line
    "ready_file": "/path/ready",           # optional: touched once bound
    "numbers": {
      "1234": {"lines": ["line1", "line2", "line3"],
               "hunt": "first_free",        # or "round_robin"
               "rings": 2, "ring_ms": 2000, # per-line values override these
               "busy_code": 486}
    },
    "lines": {
      "line1": {"type": "echo",
                "behavior": "answer",       # answer|busy|reject|no_answer
                "rings": 3,
                "banner": "CONNECT 9600\\r\\n", "echo": true,
                "dtmf_echo": true, "rate_cps": 960,
                "hangup_after_s": 0}
    },
    "delay_rules": [
      {"number": "1234", "line": "line2", "caller": "1001",
       "direction": "to_caller",            # to_line | to_caller | both
       "delay_ms": 250, "jitter_ms": 40, "loss_pct": 0,
       "start_s": 0, "end_s": 30}           # window, relative to answer
    ]
  }

Rules are matched first-hit-wins per packet and direction; every match field
is optional. Jitter larger than the 20 ms packet spacing reorders packets.

MEDIA. SDP negotiates one audio line whose payload is the private
"x-modem-bytes/8000" (each RTP payload is raw bytes of the modem's data
stream) plus RFC 4733 telephone-event for DTMF, both directions. That keeps
latency / jitter / loss insertion and DTMF in one place -- this process --
without any audio codec in the way.

LINES. `lines.<name>.type` picks a behaviour class from LINE_TYPES:

  "echo"   banner on connect, optional echo / DTMF echo, optional hangup timer.
  "modem"  in-process scriptable answering modem (Python). Options:
             banner, echo (default false) + transform (none|upper|lower|reverse),
             rules: [{"expect": regex, "send": text, "digits": "123",
                      "hangup": bool, "once": bool}]   # matched against what
                      the caller sent; a match consumes the buffer up to it
             dtmf: {"1": {"send": text, "digits": "..", "hangup": bool}},
             dtmf_echo, connect_digits, idle_hangup_s, hangup_after_s
  "fax"    the far fax machine, in-process: answers, offers its id and capabilities,
           receives pages from a Class 2 modem mock (modemmock.py with a "fax" config) and saves
           them. Options:
             id ("FAX LINE"), caps ("vr,br,wd,ln,df,ec,bf,st" = "1,3,0,2,0,0,0,0"),
             answer_delay_ms (1000: the pause before id + caps, like CED/DIS),
             response_delay_ms (200), save_dir (page files + a TIFF per call),
             page_responses: ["2", "1"]   # +FPTS value per page, default always "1" (good);
                                          # "2" rejects the page (it is not kept)
           Wire protocol and frame types: see faxcodec.py.
  "spanfax" the far fax machine as a real T.30 receiver (spandsp) over G.711 audio: the caller
           must offer PCMU/PCMA (modemmock `media: "g711"`), and CED/DIS/V.21/V.29/V.17 and the
           page data are really modulated. Options: id, save_dir (a TIFF per call), ecm (0|1),
           modems ("v27,v29,v17"), hangup_delay_ms (2500, line held after DCN), debug. Log: "fax: caller id", "fax: page N received",
           "fax: session complete pages=N tiff=PATH", ended (fax complete).
  "bridge" a call to ANOTHER modem mock: the server rings the target over its own SIP leg,
           answers the caller when the target answers, then relays data and DTMF both ways (the
           delay rules still act on the caller's leg). Options:
             target "127.0.0.1:15172"   the callee mock's SIP address (its "listen_port")
             target_user "1002"         instead of target: ring the address that user REGISTERed
                                        from (as seen on the wire, so NAT needs no forward; the
                                        client must refresh within the Expires we grant, <=120 s)
             ring_timeout_s (30)        give up (480) if the target never answers
             rate_cps (960)             pacing of data towards the callee, like any line
             watcher {...}              a Python watcher in the middle: see faxwatch.py
           A busy or refusing target is relayed to the caller (486 etc.): the caller's modem
           prints BUSY. Hunt several hosts by listing several bridge lines under one number.
  "audiobridge" like bridge, but for real audio: the caller must offer G.711 PCMU, the server
           offers PCMU to the target over its own SIP leg, and RTP is relayed UNTOUCHED both ways
           (no spandsp, no byte pipe; delay rules act on the caller's leg as for bridge).
           Options: target, ring_timeout_s. Log: "call c1 audiobridge: ringing HOST:PORT",
           "audiobridge: target answered", "audiobridge: call ended".
  "conference" a hosted telephone conference bridge: any number of callers dial the SAME number (one call
           each, one 64 kbit/s PCMU/PCMA stream up and down), and each hears the mix of everyone else,
           never itself. Real mixing (decode, sum, re-encode per member and per law, 20 ms ticks, a small
           playout buffer per member), no relaying of other people's packets. Options: max_members (10),
           prime_frames (2), max_buffer_frames (6), mix_gain (1.0). A full room answers busy (486).
           Log: "call cN conference NAME: joined (members=K)", "... left (members=K) in=.. out=..".
  Any line type but fax/bridge also takes a `spandsp` block to answer REAL modem audio:
             "spandsp": {"mods": ["v22","v21"], "bps": 2400, "v8": true, "connect_timeout_s": 45}
           A caller that offers G.711 (modemmock `media: "g711"`) is then served by the
           spanmodem helper (V.8 -> V.22bis/V.21); the line sees the demodulated bytes exactly as
           on the byte pipe, and its on_connect (banner...) runs when the modems have trained.
           Log: "call c1 spandsp: connect v22bis 2400". A byte-pipe caller still works, and a
           G.711 caller on a line without `spandsp` is refused (488). V.34 ("v34" in mods) is
           spandsp's unfinished code and does not train: see build_spanmodem.sh.
  "exec"   a child process per call, for simulations written in anything
           (linemodem.c, linemodem.py are the references). Options:
             command: [argv...], env: {..}, cwd

EXEC PROTOCOL. The child's stdin/stdout carry frames, both directions:
    [type: 1 byte][length: 2 bytes big-endian][payload]
  server -> child   'C' call start, payload "number\\ncaller\\nline\\n" (first frame)
                    'D' bytes the caller sent      'T' one DTMF digit the caller sent
                    'E' call over -- exit now
  child -> server   'D' bytes for the caller (paced at the line's rate_cps)
                    'T' one DTMF digit to send     'H' hang up (payload = reason);
                    queued data is flushed first
  stderr lines are logged as "call cN line: <text>". The child exiting ends the
  call; the call ending closes its stdin and, a second later, kills its group.

Stdout is the test log. Stable phrases to gate on with test_filecontains:
  call c1 from 1001 dialed 1234 -> line1     call c1 ring 1/2
  call c1 answered                           call c1 dtmf 5 (to line)
  call c1 data to line 12 bytes: b'ATZ\\r'   call c1 ended (bye from caller)
  busy: no free line for 1234                READY sip=0.0.0.0:5060
  call c1 fax: caller id "ACME"              call c1 fax: DCS 1,3,0,2,0,0,0,0
  call c1 fax: page 1 received 4812 bytes (EOP)
  call c1 fax: session complete pages=1 ...  call c1 ended (fax complete)
  call c1 bridge: ringing 127.0.0.1:15172    call c1 bridge: target answered
  call c1 bridge: target hung up             call c1 bridge: target refused 486 Busy Here
  (and the watcher's "call c1 watch: ..." lines, see faxwatch.py)
"""
import argparse
import asyncio
import json
import os
import random
import re
import signal
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sipcore as sc  # noqa: E402
import faxcodec as fc  # noqa: E402
import faxwatch  # noqa: E402
import spanhelper as sph  # noqa: E402

DTMF_DIGITS = "0123456789*#ABCD"
TICK = 0.02                     # RTP packet spacing, seconds
CLOCK = 8000                    # RTP clock rate for both payloads


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class ConfigError(Exception):
    pass


# ---------------------------------------------------------------------------
# line behaviours
# ---------------------------------------------------------------------------

class LineBase:
    """What sits on the far side of a call. Subclasses override the hooks;
    `call.send(bytes)`, `call.send_dtmf(d)` and `call.hangup()` are the API
    back into the call."""
    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg

    def on_connect(self, call):
        pass

    def on_data(self, call, data):
        pass

    def on_dtmf(self, call, digit):
        pass

    def on_end(self, call):
        pass

    async def shutdown(self):
        """Await anything this line still has running after its calls ended."""
        pass


class EchoLine(LineBase):
    def on_connect(self, call):
        banner = self.cfg.get("banner", "")
        if banner:
            call.send(banner.encode("latin-1", "replace"))
        after = float(self.cfg.get("hangup_after_s", 0) or 0)
        if after > 0:
            call.spawn(self._hangup_later(call, after))

    async def _hangup_later(self, call, after):
        await asyncio.sleep(after)
        call.hangup("line hangup timer")

    def on_data(self, call, data):
        if self.cfg.get("echo", True):
            call.send(data)

    def on_dtmf(self, call, digit):
        if self.cfg.get("dtmf_echo", False):
            call.send_dtmf(digit)


def _transform(text, mode):
    if mode == "upper":
        return text.upper()
    if mode == "lower":
        return text.lower()
    if mode == "reverse":
        return text[::-1]
    return text


class ModemLine(LineBase):
    """Scriptable answering modem, in-process. See the module docstring."""
    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.rules = [(re.compile(r.get("expect", "")), r) for r in cfg.get("rules", [])]

    def _act(self, call, act):
        if act.get("send"):
            call.send(act["send"].encode("latin-1", "replace"))
        if act.get("digits"):
            call.spawn(self._digits(call, act["digits"]))
        if act.get("hangup"):
            call.hangup_when_flushed("script hangup")

    async def _digits(self, call, digits):
        for d in digits:
            call.send_dtmf(d)
            await asyncio.sleep(0.2)

    def on_connect(self, call):
        st = call.line_state
        st.update(buf="", fired=set(), last=time.monotonic())
        if self.cfg.get("banner"):
            call.send(self.cfg["banner"].encode("latin-1", "replace"))
        if self.cfg.get("connect_digits"):
            call.spawn(self._digits(call, self.cfg["connect_digits"]))
        after = float(self.cfg.get("hangup_after_s", 0) or 0)
        if after > 0:
            call.spawn(self._later(call, after))
        idle = float(self.cfg.get("idle_hangup_s", 0) or 0)
        if idle > 0:
            call.spawn(self._idle(call, idle))

    async def _later(self, call, after):
        await asyncio.sleep(after)
        call.hangup_when_flushed("line hangup timer")

    async def _idle(self, call, idle):
        st = call.line_state
        while True:
            await asyncio.sleep(min(0.25, idle))
            if time.monotonic() - st["last"] >= idle:
                call.hangup_when_flushed("idle timeout")
                return

    def on_data(self, call, data):
        st = call.line_state
        st["last"] = time.monotonic()
        text = data.decode("latin-1")
        if self.cfg.get("echo", False):
            call.send(_transform(text, self.cfg.get("transform", "none")).encode("latin-1", "replace"))
        st["buf"] = (st["buf"] + text)[-4096:]
        progress = True
        while progress:
            progress = False
            for i, (rx, rule) in enumerate(self.rules):
                if rule.get("once") and i in st["fired"]:
                    continue
                m = rx.search(st["buf"])
                if m:
                    st["buf"] = st["buf"][m.end():]
                    st["fired"].add(i)
                    log(f"{call.label} line {self.name} rule {i} matched")
                    self._act(call, rule)
                    progress = True
                    break

    def on_dtmf(self, call, digit):
        call.line_state["last"] = time.monotonic()
        act = self.cfg.get("dtmf", {}).get(digit)
        if act:
            self._act(call, act)
        if self.cfg.get("dtmf_echo", False):
            call.send_dtmf(digit)


class ExecLine(LineBase):
    """One child process per call, framed over stdin/stdout. Protocol in the
    module docstring."""
    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.reapers = set()

    @staticmethod
    def _frame(t, payload=b""):
        return t + len(payload).to_bytes(2, "big") + payload

    def _write(self, call, frame):
        st = call.line_state
        proc = st.get("proc")
        if proc is None:
            st["pending"].append(frame)
            return
        try:
            proc.stdin.write(frame)
        except (BrokenPipeError, ConnectionError, RuntimeError):
            pass

    def on_connect(self, call):
        call.line_state.update(proc=None, pending=[], closed=False)
        self._write(call, self._frame(b"C", f"{call.number}\n{call.caller}\n{self.name}\n".encode()))
        call.spawn(self._run(call))

    def on_data(self, call, data):
        for i in range(0, len(data), 65535):
            self._write(call, self._frame(b"D", data[i:i + 65535]))

    def on_dtmf(self, call, digit):
        self._write(call, self._frame(b"T", digit.encode("latin-1")))

    async def _run(self, call):
        st = call.line_state
        env = dict(os.environ, LINE_NAME=self.name,
                   **{k: str(v) for k, v in self.cfg.get("env", {}).items()})
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.cfg["command"], stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=env, cwd=self.cfg.get("cwd") or None, start_new_session=True)
        except OSError as e:
            log(f"{call.label} line {self.name} cannot start {self.cfg['command'][0]}: {e}")
            call.hangup("cannot start line process")
            return
        st["proc"] = proc
        if st["closed"]:
            self._kill(proc)
            await proc.wait()
            return
        for f in st["pending"]:
            proc.stdin.write(f)
        st["pending"] = []
        call.spawn(self._stderr(call, proc))
        try:
            while True:
                hdr = await proc.stdout.readexactly(3)
                n = int.from_bytes(hdr[1:], "big")
                payload = await proc.stdout.readexactly(n) if n else b""
                t = hdr[:1]
                if t == b"D":
                    call.send(payload)
                elif t == b"T" and payload:
                    call.send_dtmf(payload[:1].decode("latin-1"))
                elif t == b"H":
                    call.hangup_when_flushed(payload.decode("latin-1", "replace") or "line hung up")
                else:
                    log(f"{call.label} line {self.name}: unknown frame {t!r}")
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        if call.state == "answered":
            call.hangup("line process exited")

    async def _stderr(self, call, proc):
        async for raw in proc.stderr:
            log(f"{call.label} line: {raw.decode('latin-1', 'replace').rstrip()}")

    @staticmethod
    def _kill(proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    async def _reap(self, proc):
        """Give the child a second to exit on EOF / 'E', then kill its group,
        and always wait() so the transport is closed before the loop is."""
        try:
            await asyncio.wait_for(proc.wait(), 1.0)
        except asyncio.TimeoutError:
            self._kill(proc)
            await proc.wait()

    async def shutdown(self):
        if self.reapers:
            await asyncio.gather(*list(self.reapers), return_exceptions=True)

    def on_end(self, call):
        st = call.line_state
        st["closed"] = True
        proc = st.get("proc")
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.stdin.write(self._frame(b"E"))
            proc.stdin.close()
        except (BrokenPipeError, ConnectionError, RuntimeError, OSError):
            pass
        t = asyncio.ensure_future(self._reap(proc))
        self.reapers.add(t)
        t.add_done_callback(self.reapers.discard)


class FaxLine(LineBase):
    """The answering fax machine. Talks the framed protocol of faxcodec.py with the caller's
    modem mock. See the module docstring for the options."""

    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.script = [str(x) for x in cfg.get("page_responses", [])]

    def on_connect(self, call):
        call.line_state.update(deframer=fc.Deframer(), pages=[], chunks=[], dcs=None, page_no=0,
                               done=False, caller_id="", responses=0)
        call.spawn(self._greet(call))

    async def _greet(self, call):
        await asyncio.sleep(float(self.cfg.get("answer_delay_ms", 1000)) / 1000.0)
        if call.state != "answered":
            return
        caps = str(self.cfg.get("caps", "1,3,0,2,0,0,0,0"))
        call.send(fc.pack("I", str(self.cfg.get("id", "FAX LINE"))))
        call.send(fc.pack("A", caps))
        log(f"{call.label} fax: offered id and DIS {caps}")

    def on_data(self, call, data):
        st = call.line_state
        for ftype, payload in st["deframer"].feed(data):
            self._frame(call, ftype, payload)

    def _frame(self, call, ftype, payload):
        st = call.line_state
        if ftype == "I":
            st["caller_id"] = payload.decode("latin-1", "replace")
            log(f'{call.label} fax: caller id "{st["caller_id"]}"')
        elif ftype == "S":
            st["dcs"] = payload.decode("latin-1", "replace")
            log(f"{call.label} fax: DCS {st['dcs']}")
            call.send(fc.pack("C"))         # ready for the page
        elif ftype == "P":
            st["chunks"].append(payload)
        elif ftype == "N":
            self._end_of_page(call, payload[:1].decode("latin-1") or "2")
        elif ftype == "D":
            log(f"{call.label} fax: caller disconnected (code {payload[:1].decode('latin-1') or '0'})")
            self._finish(call)
        else:
            log(f"{call.label} fax: unexpected frame {ftype!r}")

    def _end_of_page(self, call, post):
        st = call.line_state
        data = b"".join(st["chunks"])
        st["chunks"] = []
        n = st["responses"]
        ppr = self.script[n] if n < len(self.script) else "1"
        st["responses"] = n + 1
        word = {"0": "MPS", "1": "EOM", "2": "EOP"}.get(post, post)
        if ppr in ("1", "3"):
            st["pages"].append(data)
            k = len(st["pages"])
            log(f"{call.label} fax: page {k} received {len(data)} bytes ({word})")
            self._save(call, f"p{k}.t4", data)
        else:
            log(f"{call.label} fax: page rejected, {len(data)} bytes ({word}), ppr {ppr}")
        call.spawn(self._respond(call, ppr, post))

    async def _respond(self, call, ppr, post):
        await asyncio.sleep(float(self.cfg.get("response_delay_ms", 200)) / 1000.0)
        if call.state != "answered":
            return
        call.send(fc.pack("R", ppr))
        if post == "2" and ppr in ("1", "3"):
            await asyncio.sleep(float(self.cfg.get("response_delay_ms", 200)) / 1000.0)
            call.send(fc.pack("D", "0"))
            self._finish(call)
            call.hangup_when_flushed("fax complete")

    def _save(self, call, suffix, data):
        d = self.cfg.get("save_dir")
        if not d:
            return None
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"call_{call.label.split()[-1]}_{suffix}")
        with open(path, "wb") as f:
            f.write(data)
        return path

    def _finish(self, call):
        st = call.line_state
        if st["done"]:
            return
        st["done"] = True
        pages = st["pages"] + ([b"".join(st["chunks"])] if st["chunks"] else [])
        if not pages:
            return
        tif = None
        d = self.cfg.get("save_dir")
        if d:
            os.makedirs(d, exist_ok=True)
            tif = os.path.join(d, f"call_{call.label.split()[-1]}.tif")
            try:
                info = fc.write_tiff(tif, pages, st["dcs"] or self.cfg.get("caps", "1,3,0,2,0,0,0,0"))
                extra = f" lines={info['lines']} width={info['width']}"
            except OSError as e:
                extra, tif = f" (tiff failed: {e})", None
        else:
            extra = ""
        log(f"{call.label} fax: session complete pages={len(st['pages'])}"
            f"{' (+1 partial)' if st['chunks'] else ''} tiff={tif}{extra}")

    def on_end(self, call):
        st = call.line_state
        if not st.get("done") and (st.get("pages") or st.get("chunks")):
            log(f"{call.label} fax: call ended before the last page was confirmed")
            self._finish(call)


class Leg:
    """The callee side of a bridged call: a SIP UAC from the server to a modem mock, with its
    own RTP endpoint. Mirrors what Call does for the caller side."""

    def __init__(self, line, call, target):
        self.line, self.call, self.server = line, call, call.server
        self.target = target
        self.cid = sc.new_token(14)
        self.ltag = sc.new_token(6)
        self.state = "new"              # new|inviting|answered|ended
        self.fut = None
        self.rtp = None
        self.rtp_peer = None
        self.data_pt, self.dtmf_pt = 96, 101
        self.seq, self.ssrc = random.getrandbits(16), random.getrandbits(32)
        self.t0 = time.monotonic()
        self.out = bytearray()
        self.out_marker = True
        self.tasks = set()
        self.bye_done = asyncio.Event()
        self.to_hdr = None
        self.contact_uri = None
        self._dtmf_seen = set()
        self.stats = {"to_callee": 0, "from_callee": 0}

    kind = "bridge"                     # the log prefix: "call c1 bridge: target answered"

    def offer_sdp(self, adv, port):
        return sc.build_sdp(adv, port, 96, 101)

    def spawn(self, coro):
        t = asyncio.ensure_future(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    # ---- signalling ------------------------------------------------------------

    async def invite(self, timeout):
        srv, call = self.server, self.call
        self.rtp = await srv.open_rtp(self)
        adv = srv.advertise_for(self.target[0])
        port = self.rtp.transport.get_extra_info("sockname")[1]
        sdp = self.offer_sdp(adv, port)
        callee = f"{self.target[0]}:{self.target[1]}"
        self.uri = f"sip:{call.number}@{callee}"
        self.via = f"{adv}:{srv.port}"
        self.from_hdr = f"<sip:{call.caller}@{adv}>;tag={self.ltag}"
        self.base_to = f"<sip:{call.number}@{callee}>"
        msg = sc.build_request("INVITE", self.uri, self.via, self.from_hdr, self.base_to, self.cid, 1,
                               contact=f"<sip:{call.number}@{adv}:{srv.port}>", body=sdp)
        self.state = "inviting"
        self.fut = asyncio.get_running_loop().create_future()
        log(f"{call.label} {self.kind}: ringing {callee}")
        self.spawn(self._retransmit(msg))
        try:
            return await asyncio.wait_for(asyncio.shield(self.fut), timeout)
        except asyncio.TimeoutError:
            log(f"{call.label} {self.kind}: target did not answer within {timeout:g}s")
            await self.cancel()
            return (480, "Temporarily Unavailable")
        except asyncio.CancelledError:
            await self.cancel()
            raise

    async def _retransmit(self, msg):
        delay = 0.5
        for _ in range(6):
            self.server.sip_send(msg, self.target)
            await asyncio.sleep(delay)
            if self.state != "inviting" or self.ringing:
                return
            delay = min(delay * 2, 4.0)
        if self.fut and not self.fut.done():
            self.fut.set_result((408, "Request Timeout"))

    ringing = False

    def on_response(self, m):
        method = m.cseq()[1]
        if method == "BYE" and m.status >= 200:
            self.bye_done.set()
            return
        if method != "INVITE" or self.state not in ("inviting", "answered"):
            return
        if m.status < 200:
            self.ringing = True
            self.call.respond(180, "Ringing")
            return
        if m.status == 200:
            if self.state == "inviting":
                info = sc.parse_sdp(m.body)
                if info is None:
                    self._ack(m)
                    if self.fut and not self.fut.done():
                        self.fut.set_result((488, "Not Acceptable Here"))
                    return
                self.rtp_peer = (info["addr"], info["port"])
                self.data_pt, self.dtmf_pt = info["data_pt"], info["dtmf_pt"]
                self.to_hdr = m.get("to", self.base_to)
                self.contact_uri = sc.uri_of(m.get("contact", "")) or self.uri
                self.state = "answered"
                log(f"{self.call.label} {self.kind}: target answered")
                if self.fut and not self.fut.done():
                    self.fut.set_result(None)
            self._ack(m)
            return
        self._ack(m, non2xx=True)
        status = m.status if 400 <= m.status < 700 else 502
        log(f"{self.call.label} {self.kind}: target refused {m.status} {m.reason}")
        self.state = "ended"
        if self.fut and not self.fut.done():
            self.fut.set_result((status, m.reason or "Refused"))

    def _ack(self, m, non2xx=False):
        uri = self.uri if non2xx else (sc.uri_of(m.get("contact", "")) or self.uri)
        self.server.sip_send(sc.build_request("ACK", uri, self.via, self.from_hdr,
                                              m.get("to", self.base_to), self.cid, 1), self.target)

    async def cancel(self):
        if self.state != "inviting":
            return
        self.state = "ended"
        msg = sc.build_request("CANCEL", self.uri, self.via, self.from_hdr, self.base_to, self.cid, 1)
        self.server.sip_send(msg, self.target)
        self.close()

    def on_remote_bye(self):
        if self.state == "answered":
            self.state = "ended"
            log(f"{self.call.label} {self.kind}: target hung up")
            self.call.hangup_when_flushed("callee hung up")

    async def hangup(self):
        """BYE the callee (the caller hung up or the server is stopping)."""
        if self.state == "answered":
            self.state = "ended"
            bye = sc.build_request("BYE", self.contact_uri, self.via, self.from_hdr,
                                   self.to_hdr or self.base_to, self.cid, 2)
            for _ in range(4):
                self.server.sip_send(bye, self.target)
                try:
                    await asyncio.wait_for(self.bye_done.wait(), 0.5)
                    break
                except asyncio.TimeoutError:
                    continue
        self.close()

    def close(self):
        self.state = "ended"
        for t in list(self.tasks):
            if t is not asyncio.current_task():
                t.cancel()
        if self.rtp:
            self.rtp.close()
        self.server.legs.pop(self.cid, None)

    # ---- media ------------------------------------------------------------------

    def start(self):
        self.t0 = time.monotonic()
        self.spawn(self._sender())

    def _ts(self):
        return int((time.monotonic() - self.t0) * CLOCK)

    def _put(self, pkt):
        if self.state == "answered" and self.rtp and self.rtp.transport and not self.rtp.transport.is_closing():
            self.rtp.transport.sendto(pkt, self.rtp_peer)

    def send_data(self, data):
        self.out.extend(data)

    async def _sender(self):
        rate = float(self.line.cfg.get("rate_cps", 960))
        per_tick = max(1, int(round(rate * TICK)))
        while True:
            await asyncio.sleep(TICK)
            if not self.out:
                self.out_marker = True
                continue
            chunk = bytes(self.out[:per_tick])
            del self.out[:per_tick]
            self.seq = (self.seq + 1) & 0xFFFF
            self.stats["to_callee"] += len(chunk)
            self._put(rtp_packet(self.data_pt, self.seq, self._ts(), self.ssrc, chunk, self.out_marker))
            self.out_marker = False

    def send_dtmf(self, digit):
        if self.dtmf_pt is not None and digit in DTMF_DIGITS:
            self.spawn(self._dtmf_burst(digit))

    async def _dtmf_burst(self, digit):
        ev, ts, pt = DTMF_DIGITS.index(digit), self._ts(), self.dtmf_pt
        for i in range(5):
            self.seq = (self.seq + 1) & 0xFFFF
            self._put(rtp_packet(pt, self.seq, ts, self.ssrc,
                                 bytes([ev, 10, ((i + 1) * 160) >> 8, ((i + 1) * 160) & 0xFF]), i == 0))
            await asyncio.sleep(TICK)
        for _ in range(3):
            self.seq = (self.seq + 1) & 0xFFFF
            self._put(rtp_packet(pt, self.seq, ts, self.ssrc, bytes([ev, 0x80 | 10, 800 >> 8, 800 & 0xFF])))

    def on_rtp(self, data, addr):
        if len(data) < 12 or data[0] >> 6 != 2 or self.state != "answered":
            return
        cc = data[0] & 0x0F
        pt = data[1] & 0x7F
        _, ts, ssrc = struct.unpack("!HII", data[2:12])
        payload = data[12 + 4 * cc:]
        self.rtp_peer = addr
        if pt == self.data_pt and payload:
            self.stats["from_callee"] += len(payload)
            self.line.behavior.from_callee(self.call, bytes(payload))
        elif self.dtmf_pt is not None and pt == self.dtmf_pt and len(payload) >= 4:
            if (ssrc, ts) in self._dtmf_seen:
                return
            self._dtmf_seen.add((ssrc, ts))
            if payload[0] < len(DTMF_DIGITS):
                self.call.send_dtmf(DTMF_DIGITS[payload[0]])


class BridgeLine(LineBase):
    """A line that is another modem mock. See the module docstring."""

    leg_class = Leg

    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.target_user = cfg.get("target_user")   # else the callee is wherever it REGISTERed from
        if self.target_user is None:
            host, _, port = str(cfg["target"]).rpartition(":")
            self.target = (host, int(port))
        else:
            self.target = None
        self.line = None            # set by Server: the (name, cfg, behavior) record Leg reads rate_cps from
        self.pending = set()

    async def connect_leg(self, call):
        """Ring the callee; None when it answered, else (status, reason) to reject the caller with."""
        target = self.target
        if self.target_user is not None:
            ent = call.server.registry.get(str(self.target_user))
            if not ent or ent[1] < time.monotonic():
                log(f"{call.label} bridge: {self.target_user} is not registered")
                return (480, "Temporarily Unavailable")
            target = ent[0]
        leg = self.leg_class(self.line, call, target)
        call.line_state["leg"] = leg
        call.server.legs[leg.cid] = leg
        res = await leg.invite(float(self.cfg.get("ring_timeout_s", 30)))
        if res is not None:
            leg.close()
            call.line_state.pop("leg", None)
        return res

    def on_connect(self, call):
        st = call.line_state
        leg = st.get("leg")
        if leg is None:
            return
        wcfg = self.cfg.get("watcher")
        st["watcher"] = faxwatch.load_watcher(wcfg, call.label, log) if wcfg is not None else None
        leg.start()

    def _run(self, call, direction, data, sink):
        w = call.line_state.get("watcher")
        for chunk in (w.process(direction, data) if w else [data]):
            sink(chunk)

    def on_data(self, call, data):
        leg = call.line_state.get("leg")
        if leg:
            self._run(call, "a2b", data, leg.send_data)

    def from_callee(self, call, data):
        self._run(call, "b2a", data, call.send)

    def on_dtmf(self, call, digit):
        leg = call.line_state.get("leg")
        if leg:
            leg.send_dtmf(digit)

    def on_end(self, call):
        st = call.line_state
        leg, w = st.get("leg"), st.get("watcher")
        if leg:
            if leg.state == "answered":
                t = asyncio.ensure_future(leg.hangup())
                self.pending.add(t)
                t.add_done_callback(self.pending.discard)
            else:
                leg.close()
        if w:
            try:
                w.close()
            except Exception as e:
                log(f"{call.label} watcher close error: {e!r}")

    async def shutdown(self):
        if self.pending:
            await asyncio.gather(*list(self.pending), return_exceptions=True)


class AudioLeg(Leg):
    """Callee leg of an audiobridge call: offers G.711 PCMU and hands the callee's RTP to the
    caller untouched (only the payload type byte is rewritten when the two ends differ)."""

    kind = "audiobridge"

    def offer_sdp(self, adv, port):
        return sc.build_sdp_g711(adv, port, (self.call.offer["data_pt"],), self.call.offer["dtmf_pt"])

    def start(self):
        self.t0 = time.monotonic()

    def send_audio(self, pkt):
        """Caller -> callee: the caller's RTP packet as received."""
        if len(pkt) < 12:
            return
        if (pkt[1] & 0x7F) == self.call.offer["data_pt"] and self.data_pt != self.call.offer["data_pt"]:
            pkt = pkt[:1] + bytes([(pkt[1] & 0x80) | self.data_pt]) + pkt[2:]
        self.stats["to_callee"] += len(pkt) - 12
        self._put(pkt)

    def on_rtp(self, data, addr):
        if len(data) < 12 or data[0] >> 6 != 2 or self.state != "answered":
            return
        self.rtp_peer = addr
        pt = data[1] & 0x7F
        if pt != self.data_pt:
            return
        if pt != self.call.offer["data_pt"]:
            data = data[:1] + bytes([(data[1] & 0x80) | self.call.offer["data_pt"]]) + data[2:]
        self.stats["from_callee"] += len(data) - 12
        self.call.stats["tx"] += len(data) - 12
        self.call._emit("to_caller", data)


class AudioBridgeLine(BridgeLine):
    """A line that is another G.711 endpoint (an NE0K modem): like `bridge`, but the call is
    real audio and RTP is relayed untouched between the two legs. See the module docstring."""

    audio_relay = True
    leg_class = AudioLeg

    def on_connect(self, call):
        leg = call.line_state.get("leg")
        if leg is None:
            return
        leg.start()

    def on_audio(self, call, pkt):
        leg = call.line_state.get("leg")
        if leg:
            leg.send_audio(pkt)

    def on_end(self, call):
        log(f"{call.label} audiobridge: call ended")
        super().on_end(call)


class SpanFaxLine(LineBase):
    """The answering fax machine as a real T.30 receiver (spandsp) over G.711 audio: CED,
    DIS, V.21 / V.27ter / V.29 / V.17 training and page data, ECM, all on the wire. The caller
    must offer G.711 (modemmock with `media: "g711"`, or any SIP fax sender). Options: id,
    save_dir (a TIFF per call), ecm (0|1), modems ("v27,v29,v17"), debug."""

    def span_args(self, call):
        d = self.cfg.get("save_dir") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(d, exist_ok=True)
        call.line_state["tiff"] = os.path.join(d, f"call_{call.label.split()[-1]}.tif")
        call.line_state["pages"] = 0
        args = ["--mode", "faxrx", "--law", call.offer["law"], "--tiff", call.line_state["tiff"],
                "--ident", str(self.cfg.get("id", "FAX LINE")), "--ecm", str(int(self.cfg.get("ecm", 0)))]
        if self.cfg.get("modems"):
            args += ["--fax-modems", str(self.cfg["modems"])]
        if self.cfg.get("debug"):
            args.append("--debug")
        return args, "fax"

    def on_span_status(self, call, text):
        st = call.line_state
        m = re.match(r"fax phase-b remote=(.*)$", text)
        if m:
            log(f'{call.label} fax: caller id "{m.group(1)}"')
        m = re.match(r"fax page (\d+) done", text)
        if m:
            log(f"{call.label} fax: page {m.group(1)} received")
        m = re.match(r"fax result (\d+) .* pages_tx=(\d+) pages_rx=(\d+)", text)
        if m:
            st["pages"] = int(m.group(3))

    def on_span_done(self, call, result):
        st = call.line_state
        ok = result == "ok"
        if ok:
            log(f"{call.label} fax: session complete pages={st.get('pages', 0)} tiff={st.get('tiff')}")
        else:
            log(f"{call.label} fax: session failed ({result})")
        # hold the line briefly after DCN like a real machine: the sender's own T.30 engine
        # reports completion a moment after ours, and a BYE before that reads as a lost call
        call.spawn(self._hold_then_hangup(call, "fax complete" if ok else "fax failed"))

    async def _hold_then_hangup(self, call, reason):
        await asyncio.sleep(float(self.cfg.get("hangup_delay_ms", 2500)) / 1000.0)
        call.hangup_when_flushed(reason)


# ---------------------------------------------------------------------------
# conference bridge
# ---------------------------------------------------------------------------

def _build_g711():
    """Stdlib G.711 tables (audioop is gone in Python 3.13): decode[256] -> int16, and encode over the
    top 14 bits of a clipped 16-bit sample, both laws."""
    def ulaw_dec(u):
        u = ~u & 0xFF
        t = (((u & 0x0F) << 3) + 0x84) << ((u & 0x70) >> 4)
        return (0x84 - t) if u & 0x80 else (t - 0x84)

    def alaw_dec(a):
        a ^= 0x55
        t, seg = (a & 0x0F) << 4, (a & 0x70) >> 4
        t = t + 8 if seg == 0 else (t + 0x108) << (seg - 1)
        return t if a & 0x80 else -t

    def ulaw_enc(pcm):
        pcm = max(-32635, min(32635, pcm))
        sign = 0x80 if pcm < 0 else 0
        pcm = abs(pcm) + 0x84
        exp = max(0, pcm.bit_length() - 8)
        return ~(sign | (exp << 4) | ((pcm >> (exp + 3)) & 0x0F)) & 0xFF

    def alaw_enc(pcm):
        sign = 0x80 if pcm >= 0 else 0
        pcm = min(abs(pcm) if pcm >= 0 else -pcm - 1, 32767)
        if pcm < 256:
            v = pcm >> 4
        else:
            exp = pcm.bit_length() - 8
            v = (exp << 4) | ((pcm >> (exp + 3)) & 0x0F)
        return (v | sign) ^ 0x55

    dec = {"ulaw": [ulaw_dec(i) for i in range(256)], "alaw": [alaw_dec(i) for i in range(256)]}
    # 16-bit sample -> code, indexed by (sample >> 2) + 8192 (14 bit resolution, enough for 8 bit companding)
    enc = {"ulaw": bytes(ulaw_enc((i - 8192) << 2) for i in range(16384)),
           "alaw": bytes(alaw_enc((i - 8192) << 2) for i in range(16384))}
    return dec, enc


G711_DEC, G711_ENC = _build_g711()
FRAME = 160                      # samples (= PCM bytes) per 20 ms packet at 8 kHz


def _ts_diff(a, b):
    """a - b for 32-bit RTP timestamps, wrap-safe."""
    return ((a - b + 0x80000000) & 0xFFFFFFFF) - 0x80000000


class ConfMember:
    """One caller in a conference: an RTP-timestamp-indexed playout buffer in, a paced RTP stream out.

    Audio is placed by its RTP timestamp, not by arrival order, so network jitter never re-times a stream: a packet
    that is late or lost becomes silence in ITS OWN 20 ms slot and everything after it stays where the sender put
    it.  The old byte FIFO dropped the oldest audio on a burst and re-primed (inserted silence) after a gap, so a
    single delay spike shifted the rest of a talkspurt by +-20 ms -- fatal to a modem frame (OFDM loses sync)."""

    def __init__(self, call):
        self.call = call
        self.law = call.offer.get("law", "ulaw")
        self.pt = call.offer["data_pt"]
        self.jb = {}                # rtp ts -> G.711 payload, not yet mixed
        self.play = None            # rtp ts of the next frame to mix (None until the first packet)
        self.hi = None              # rtp ts just past the newest audio received
        self.primed = False         # playout starts once `prime` frames are buffered
        self.hold = False           # drift correction: skip advancing for one tick
        self.idle = 0               # consecutive ticks with nothing buffered
        self.seq = random.getrandbits(16)
        self.ts = random.getrandbits(32)
        self.ssrc = random.getrandbits(32)
        self.stats = {"frames_in": 0, "underruns": 0, "overruns": 0, "frames_out": 0,
                      "late": 0, "lost": 0, "skips": 0, "inserts": 0, "resyncs": 0}

    def depth(self):
        """Buffered audio ahead of the playout point, in samples."""
        return 0 if self.play is None else max(0, _ts_diff(self.hi, self.play))


class ConferenceLine(LineBase):
    """A hosted telephone conference. Any number of callers dial the same number; each hears the mix of
    everyone ELSE (never itself), like a real bridge. Per 20 ms tick the server takes one frame from every
    member's playout buffer (silence when it has none), sums the frames, and sends each member `total - own`
    re-encoded in that member's own law. Members join and leave at any time; the room stays up while anyone is in
    it.

    The playout buffer is indexed by RTP timestamp (see ConfMember): jitter costs at most a lost packet, never a
    time slip.  Clock drift between a sender and this server's tick is absorbed by dropping or repeating-as-silence
    ONE frame, and only while that member's audio is silent, so a modem burst is never cut or stretched.

    Options: max_members (10), prime_frames (5: playout delay, 100 ms of jitter absorbed), max_buffer_frames (50:
    hard bound; beyond it a frame is dropped even mid-burst, counted in `overruns`), drift_slack_frames (4: depth
    above prime + this is trimmed in silence), silence_level (64: |sample| at or below it counts as silent), idle_frames (25: this long with nothing
    buffered = the sender stopped; playout waits and re-primes),
    mix_gain (1.0). Log: "conference NAME: cN joined (members=K)", "conference NAME: cN left (members=K) ..." with
    the buffer counters."""

    audio_relay = True       # the caller's RTP arrives whole via on_audio, no byte pipe
    shared = True            # many calls on one line (hunt() must not treat it as busy)

    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.members = {}
        self.max_members = int(cfg.get("max_members", 10))
        self.prime = int(cfg.get("prime_frames", 5)) * FRAME
        self.max_buf = int(cfg.get("max_buffer_frames", 50)) * FRAME
        self.slack = int(cfg.get("drift_slack_frames", 4)) * FRAME
        self.quiet = int(cfg.get("silence_level", 64))
        self.idle_ticks = int(cfg.get("idle_frames", 25))
        self.gain = float(cfg.get("mix_gain", 1.0))
        self.task = None

    def admits(self, call):
        return len(self.members) < self.max_members

    def on_connect(self, call):
        m = ConfMember(call)
        self.members[call] = m
        log(f"{call.label} conference {self.name}: joined (members={len(self.members)})")
        call.event("conf_join", members=len(self.members))
        if self.task is None or self.task.done():
            self.task = asyncio.ensure_future(self._mixer())

    def on_audio(self, call, pkt):
        m = self.members.get(call)
        if m is None or len(pkt) < 12:
            return
        if pkt[0] & 0x10:                                   # header extension
            return
        payload = bytes(pkt[12 + 4 * (pkt[0] & 0x0F):])
        if not payload:
            return
        ts = struct.unpack("!I", pkt[4:8])[0]
        m.stats["frames_in"] += 1
        if m.play is None:
            m.play, m.hi = ts, ts
        elif _ts_diff(ts, m.hi) > self.max_buf or _ts_diff(ts, m.play) < -self.max_buf:
            # a real discontinuity (new SSRC/talkspurt after DTX, sender restart): re-time once, here
            m.jb.clear()
            m.play, m.hi, m.primed = ts, ts, False
            m.stats["resyncs"] += 1
        if _ts_diff(ts, m.play) < 0:                        # its slot has already been played
            m.stats["late"] += 1
            return
        m.jb[ts] = payload
        end = (ts + len(payload)) & 0xFFFFFFFF
        if _ts_diff(end, m.hi) > 0:
            m.hi = end

    def on_end(self, call):
        m = self.members.pop(call, None)
        if m is not None:
            st = " ".join(f"{k}={v}" for k, v in m.stats.items())
            log(f"{call.label} conference {self.name}: left (members={len(self.members)}) {st}")
            call.event("conf_leave", members=len(self.members), **m.stats)

    async def _mixer(self):
        loop = asyncio.get_running_loop()
        nxt = loop.time()
        while self.members:
            nxt += TICK
            self._tick()
            await asyncio.sleep(max(0.0, nxt - loop.time()))
            if loop.time() - nxt > 0.5:                     # the process stalled: resynchronise, don't burst
                nxt = loop.time()

    def _take(self, m):
        """The member's next 20 ms frame as decoded samples, or None (not playing / silence).

        Once primed, the playout point advances by one frame EVERY tick, whatever has arrived: a packet that
        misses its slot is silence there and is discarded when it turns up, so nothing after it moves."""
        if m.play is None:
            return None
        if not m.primed:
            if m.depth() < self.prime:
                return None
            m.primed, m.idle = True, 0
        if m.hold:                                          # drift: one extra silent tick, playout point unmoved
            m.hold = False
            return None
        dec = G711_DEC[m.law]
        if m.depth() <= 0:
            m.idle += 1
            m.stats["underruns"] += 1
            if m.idle >= self.idle_ticks:                   # the sender has stopped: wait for it, re-prime on return
                m.primed = False
                return None
            m.play = (m.play + FRAME) & 0xFFFFFFFF
            return None
        m.idle = 0
        raw = m.jb.pop(m.play, None)
        m.play = (m.play + (len(raw) if raw else FRAME)) & 0xFFFFFFFF
        if raw is None:
            m.stats["lost"] += 1                            # lost, or late: silence in its own slot
            return None
        fr = [dec[b] for b in raw]
        silent = max(map(abs, fr)) <= self.quiet
        depth = m.depth()
        if depth > self.max_buf:                            # hard bound: drop regardless (should stay 0)
            m.jb.pop(m.play, None)
            m.play = (m.play + FRAME) & 0xFFFFFFFF
            m.stats["overruns"] += 1
        elif silent and depth > self.prime + self.slack:    # sender clock fast: drop one SILENT frame
            nxt = m.jb.get(m.play)
            if nxt is None or max(abs(dec[b]) for b in nxt) <= self.quiet:
                m.jb.pop(m.play, None)
                m.play = (m.play + FRAME) & 0xFFFFFFFF
                m.stats["skips"] += 1
        elif silent and depth < self.prime - self.slack // 2 and depth < self.prime:
            m.hold = True                                   # sender clock slow: one extra silent tick
            m.stats["inserts"] += 1
        return fr

    def _tick(self):
        frames = {call: self._take(m) for call, m in list(self.members.items())}
        total = [0] * FRAME
        for fr in frames.values():
            if fr:
                total = [a + b for a, b in zip(total, fr)]
        g = self.gain
        for call, m in list(self.members.items()):
            own = frames.get(call)
            mixed = total if not own else [a - b for a, b in zip(total, own)]
            enc = G711_ENC[m.law]
            out = bytes(enc[max(-8192, min(8191, int(v * g) >> 2)) + 8192] for v in mixed)
            m.seq = (m.seq + 1) & 0xFFFF
            m.ts = (m.ts + FRAME) & 0xFFFFFFFF
            m.stats["frames_out"] += 1
            call.stats["tx"] += len(out)
            call._emit("to_caller", rtp_packet(m.pt, m.seq, m.ts, m.ssrc, out, marker=(m.stats["frames_out"] == 1)))


LINE_TYPES = {"echo": EchoLine, "modem": ModemLine, "exec": ExecLine, "fax": FaxLine,
              "bridge": BridgeLine, "audiobridge": AudioBridgeLine, "conference": ConferenceLine,
              "spanfax": SpanFaxLine}


# ---------------------------------------------------------------------------
# latency / jitter / loss
# ---------------------------------------------------------------------------

class Impairment:
    def __init__(self, rules, rng):
        self.rules = rules
        self.rng = rng

    def _match(self, call, direction):
        elapsed = time.monotonic() - call.t_answer if call.t_answer else 0.0
        for r in self.rules:
            if r.get("direction", "both") not in ("both", direction):
                continue
            if "number" in r and r["number"] != call.number:
                continue
            if "line" in r and r["line"] != call.line.name:
                continue
            if "caller" in r and r["caller"] not in call.caller:
                continue
            if elapsed < float(r.get("start_s", 0)):
                continue
            if "end_s" in r and elapsed >= float(r["end_s"]):
                continue
            return r
        return None

    def hold_time(self, call, direction):
        """Longest a packet can currently be held (seconds): what a BYE that
        follows media must wait so it cannot overtake it."""
        r = self._match(call, direction)
        if r is None:
            return 0.0
        return max(0.0, (float(r.get("delay_ms", 0)) + float(r.get("jitter_ms", 0))) / 1000.0)

    def send(self, call, direction, fn):
        r = self._match(call, direction)
        if r is None:
            fn()
            return
        if self.rng.random() * 100 < float(r.get("loss_pct", 0)):
            call.stats["dropped"] += 1
            return
        jitter = float(r.get("jitter_ms", 0))
        delay = float(r.get("delay_ms", 0)) + (self.rng.uniform(-jitter, jitter) if jitter else 0)
        if delay <= 0:
            fn()
        else:
            asyncio.get_running_loop().call_later(delay / 1000.0, fn)


# ---------------------------------------------------------------------------
# RTP
# ---------------------------------------------------------------------------

class RtpEndpoint(asyncio.DatagramProtocol):
    def __init__(self, call):
        self.call = call
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        self.call.on_rtp(data, addr)

    def close(self):
        if self.transport:
            self.transport.close()


def rtp_packet(pt, seq, ts, ssrc, payload, marker=False):
    return struct.pack("!BBHII", 0x80, (0x80 if marker else 0) | pt,
                       seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc) + payload


# ---------------------------------------------------------------------------
# one call
# ---------------------------------------------------------------------------

class Call:
    def __init__(self, server, cid, invite, peer, number, line, sdp):
        self.server = server
        self.cid = cid
        self.n = server.next_call_no()
        self.tag = sc.new_token(8)
        self.invite = invite
        self.peer = peer                    # SIP source (host, port)
        self.number = number
        self.caller = sc.user_of(sc.uri_of(invite.get("from", ""))) or invite.get("from", "")
        self.line = line
        self.offer = sdp
        self.state = "ringing"              # ringing|answered|ended|rejected
        self.last_resp = None
        self.acked = False
        self.t_answer = None
        self.rtp = None
        self.rtp_peer = (sdp["addr"], sdp["port"])
        self.ssrc = random.getrandbits(32)
        self.seq = random.getrandbits(16)
        self.t0 = None
        self.out = bytearray()
        self.out_marker = True
        self.stats = {"rx": 0, "tx": 0, "dropped": 0, "dtmf_rx": 0, "dtmf_tx": 0}
        self.tasks = set()
        self.line_state = {}                # scratch space for the line behaviour
        self.bye_done = asyncio.Event()
        self._dtmf_seen = set()
        self.end_reason = None
        self.bye_cseq = 100
        self.audio_relay = bool(getattr(line.behavior, "audio_relay", False))
        self.g711 = bool(sdp.get("g711"))   # real audio (spandsp) instead of the byte pipe
        self.span = None                    # the spanmodem helper, g711 calls only
        self.span_connected = False
        self.span_ts = random.getrandbits(32)

    @property
    def label(self):
        return f"call c{self.n}"

    def spawn(self, coro):
        t = asyncio.ensure_future(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    def event(self, kind, **kw):
        self.server.event(kind, call=f"c{self.n}", number=self.number,
                          line=self.line.name, **kw)

    # ---- signalling -------------------------------------------------------

    def respond(self, status, reason, extra=(), body=None, final=False):
        data = sc.build_response(self.invite, status, reason, to_tag=self.tag,
                                 extra=extra, body=body)
        if final:
            self.last_resp = data
        self.server.sip_send(data, self.peer)

    async def run_ringing(self):
        if hasattr(self.line.behavior, "connect_leg"):
            self.respond(100, "Trying")
            res = await self.line.behavior.connect_leg(self)
            if res is None:
                await self.answer()
            else:
                self.reject(*res)
            return
        cfg = self.line.cfg
        num = self.server.cfg["numbers"][self.number]
        rings = int(cfg.get("rings", num.get("rings", 1)))
        ring_ms = float(cfg.get("ring_ms", num.get("ring_ms", 2000)))
        behavior = cfg.get("behavior", "answer")
        self.respond(100, "Trying")
        for i in range(1, rings + 1):
            log(f"{self.label} ring {i}/{rings}")
            self.respond(180, "Ringing")
            await asyncio.sleep(ring_ms / 1000.0)
        if behavior == "answer":
            await self.answer()
        elif behavior == "busy":
            self.reject(486, "Busy Here")
        elif behavior == "reject":
            self.reject(603, "Decline")
        elif behavior == "no_answer":
            if rings == 0:
                self.respond(180, "Ringing")
            await asyncio.sleep(float(cfg.get("no_answer_timeout_s", 30)))
            self.reject(480, "Temporarily Unavailable")
        else:
            self.reject(500, "Server Error")

    def reject(self, status, reason):
        if self.state != "ringing":
            return
        self.respond(status, reason, final=True)
        self.state = "rejected"
        log(f"{self.label} rejected {status} {reason}")
        self.event("rejected", status=status)
        self.server.release(self)
        self.server.forget_later(self.cid, 5)

    async def answer(self):
        try:
            self.rtp = await self.server.open_rtp(self)
        except OSError as e:
            self.reject(500, "No RTP port")
            log(f"{self.label} cannot open RTP: {e}")
            return
        adv = self.server.advertise_for(self.peer[0])
        port = self.rtp.transport.get_extra_info("sockname")[1]
        if self.g711:
            sdp = sc.build_sdp_g711(adv, port, (self.offer["data_pt"],), self.offer["dtmf_pt"])
        else:
            sdp = sc.build_sdp(adv, port, self.offer["data_pt"], self.offer["dtmf_pt"])
        contact = f"Contact: <sip:{self.number}@{adv}:{self.server.port}>"
        self.respond(200, "OK", extra=[contact], body=sdp, final=True)
        self.state = "answered"
        self.t_answer = time.monotonic()
        self.t0 = self.t_answer
        log(f"{self.label} answered (rtp {adv}:{port} <-> {self.rtp_peer[0]}:{self.rtp_peer[1]})")
        self.event("answered", rtp_port=port)
        self.spawn(self._retransmit_200())
        if self.g711 and not self.audio_relay:
            await self.start_span()
            return
        if not self.g711:
            self.spawn(self._sender())
        self.line.behavior.on_connect(self)

    # ---- spandsp media (g711 calls) ---------------------------------------

    def span_cfg(self):
        """-> (args, mode) for the helper, from the line's `spandsp` block (data) or the line type."""
        beh = self.line.behavior
        if hasattr(beh, "span_args"):
            return beh.span_args(self)
        c = self.line.cfg.get("spandsp") or {}
        args = ["--mode", "data", "--role", "answerer", "--law", self.offer["law"]]
        mods = c.get("mods", ["v22", "v21"])
        args += ["--mods", ",".join(mods), "--bps", str(c.get("bps", 2400))]
        if c.get("v8", True) is False:
            args.append("--no-v8")
        if "guard_ms" in c:
            args += ["--guard-ms", str(c["guard_ms"])]
        if c.get("debug"):
            args.append("--debug")
        return args, "data"

    async def start_span(self):
        args, mode = self.span_cfg()
        self.span_mode = mode
        pt = self.offer["data_pt"]
        self.span = sph.PacedModem(args, on_audio=self._span_audio, on_data=self._span_data,
                                   on_status=self._span_status, on_done=self._span_done,
                                   on_stderr=lambda t: log(f"{self.label} spandsp: {t}"))
        try:
            await self.span.start()
        except OSError as e:
            log(f"{self.label} cannot start spanmodem: {e}")
            self.hangup("spandsp helper failed")
            return
        log(f"{self.label} spandsp {mode}: {' '.join(args)} (pt {pt})")
        if mode == "data":
            # the line's on_connect (banner, etc.) waits for the modem to train
            self.spawn(self._span_connect_timeout())
        else:
            self.line.behavior.on_connect(self)

    async def _span_connect_timeout(self):
        await asyncio.sleep(float(self.line.cfg.get("spandsp", {}).get("connect_timeout_s", 45)))
        if self.state == "answered" and not self.span_connected:
            log(f"{self.label} spandsp: modem never connected")
            self.hangup("modem training timeout")

    def _span_audio(self, payload):
        if self.state not in ("answered", "ended") or not self.rtp:
            return
        self.seq = (self.seq + 1) & 0xFFFF
        pkt = rtp_packet(self.offer["data_pt"], self.seq, self.span_ts, self.ssrc, payload)
        self.span_ts = (self.span_ts + len(payload)) & 0xFFFFFFFF
        self._emit("to_caller", pkt)

    def _span_data(self, data):
        self._deliver_data(data)

    def _span_status(self, text):
        log(f"{self.label} spandsp: {text}")
        self.event("spandsp", text=text)
        hook = getattr(self.line.behavior, "on_span_status", None)
        if hook:
            hook(self, text)
        if text.startswith("connect ") and not self.span_connected:
            self.span_connected = True
            if self.span_mode == "data":
                self.line.behavior.on_connect(self)
        elif text == "carrier down" and self.span_connected and self.state == "answered":
            self.hangup("carrier lost")

    def _span_done(self, result):
        hook = getattr(self.line.behavior, "on_span_done", None)
        if hook:
            hook(self, result)
        elif self.state == "answered" and result != "ok":
            log(f"{self.label} spandsp: helper ended ({result})")
            self.hangup("spandsp helper ended")

    async def _retransmit_200(self):
        delay = 0.5
        for _ in range(6):
            await asyncio.sleep(delay)
            if self.acked or self.state != "answered":
                return
            self.server.sip_send(self.last_resp, self.peer)
            delay = min(delay * 2, 4.0)
        if not self.acked and self.state == "answered":
            self.end("no ACK for 200 OK")

    def on_ack(self):
        self.acked = True

    def hangup_when_flushed(self, reason="line hangup"):
        """BYE once everything the line queued has been sent (data is paced,
        so a plain hangup() right after send() would cut it off) AND any delay
        rule has released it: signalling is not delayed, so without the extra
        wait the BYE would overtake the media it follows."""
        async def wait():
            while self.out:
                await asyncio.sleep(TICK)
            await asyncio.sleep(TICK * 2 + self.server.impair.hold_time(self, "to_caller"))
            self.hangup(reason)
        self.spawn(wait())

    def hangup(self, reason="line hangup"):
        """Server-initiated BYE."""
        if self.state != "answered":
            return
        self.state = "ended"
        self.end_reason = reason
        adv = self.server.advertise_for(self.peer[0])
        to_hdr = self.invite.get("from", "")
        from_hdr = self.invite.get("to", "")
        if not sc.get_tag(from_hdr):
            from_hdr += f";tag={self.tag}"
        uri = sc.uri_of(self.invite.get("contact", "")) or sc.uri_of(to_hdr)
        self.bye_cseq += 1
        data = sc.build_request("BYE", uri, f"{adv}:{self.server.port}", from_hdr,
                                to_hdr, self.cid, self.bye_cseq)
        log(f"{self.label} hangup ({reason}) -> BYE")
        self.spawn(self._send_bye(data))

    async def _send_bye(self, data):
        for _ in range(4):
            self.server.sip_send(data, self.peer)
            try:
                await asyncio.wait_for(self.bye_done.wait(), 0.5)
                break
            except asyncio.TimeoutError:
                continue
        self.finish()

    def on_bye_response(self):
        self.bye_done.set()

    def end(self, reason):
        """Terminate on the far side's request (or a local failure)."""
        if self.state == "ended":
            return
        self.state = "ended"
        self.end_reason = reason
        self.finish()

    def finish(self):
        if getattr(self, "_finished", False):
            return
        self._finished = True
        self.state = "ended"
        for t in list(self.tasks):
            if t is not asyncio.current_task():
                t.cancel()
        if self.span:
            asyncio.ensure_future(self.span.stop())
            self.span = None
        try:
            self.line.behavior.on_end(self)
        except Exception as e:      # a line bug must not wedge teardown
            log(f"{self.label} line on_end error: {e!r}")
        if self.rtp:
            self.rtp.close()
        log(f"{self.label} ended ({self.end_reason}) rx={self.stats['rx']}B "
            f"tx={self.stats['tx']}B dropped={self.stats['dropped']}")
        self.event("ended", reason=self.end_reason, **self.stats)
        self.server.release(self)
        self.server.forget_later(self.cid, 5)

    # ---- media -----------------------------------------------------------

    def on_rtp(self, data, addr):
        if len(data) < 12 or data[0] >> 6 != 2 or self.state != "answered":
            return
        cc = data[0] & 0x0F
        pt = data[1] & 0x7F
        seq, ts, ssrc = struct.unpack("!HII", data[2:12])
        payload = data[12 + 4 * cc:]
        self.rtp_peer = addr            # symmetric RTP: answer where it came from
        if pt == self.offer["data_pt"] and self.g711 and self.audio_relay:
            self.stats["rx"] += len(payload)
            self.server.impair.send(self, "to_line", lambda p=bytes(data): self.line.behavior.on_audio(self, p))
        elif pt == self.offer["data_pt"] and self.g711:
            if self.span:
                self.server.impair.send(self, "to_line", lambda p=payload: self.span.audio(p))
        elif pt == self.offer["data_pt"]:
            self.server.impair.send(self, "to_line", lambda p=payload: self._deliver_data(p))
        elif self.offer["dtmf_pt"] is not None and pt == self.offer["dtmf_pt"] and len(payload) >= 4:
            key = (ssrc, ts)
            if key in self._dtmf_seen:
                return
            self._dtmf_seen.add(key)
            ev = payload[0]
            if ev < len(DTMF_DIGITS):
                digit = DTMF_DIGITS[ev]
                self.server.impair.send(self, "to_line", lambda d=digit: self._deliver_dtmf(d))

    def _deliver_data(self, payload):
        if self.state != "answered" or not payload:
            return
        self.stats["rx"] += len(payload)
        if self.server.cfg.get("log_data", True):
            log(f"{self.label} data to line {len(payload)} bytes: {bytes(payload[:60])!r}")
        self.line.behavior.on_data(self, bytes(payload))

    def _deliver_dtmf(self, digit):
        if self.state != "answered":
            return
        self.stats["dtmf_rx"] += 1
        log(f"{self.label} dtmf {digit} (to line)")
        self.event("dtmf", digit=digit, direction="to_line")
        self.line.behavior.on_dtmf(self, digit)

    def send(self, data):
        """Queue bytes for the caller, paced at the line's rate_cps (or, on a spandsp call,
        modulated by the helper at the modem's own rate)."""
        if self.g711:
            if self.span:
                self.stats["tx"] += len(data)
                self.span.dte(bytes(data))
            return
        self.out.extend(data)

    def send_dtmf(self, digit):
        if self.offer["dtmf_pt"] is None or digit not in DTMF_DIGITS:
            return
        self.spawn(self._dtmf_burst(digit))

    def _emit(self, direction, pkt):
        def go():
            # "ended" still sends: a line that hangs up right after its last
            # bytes must not lose them to a delay rule still holding them
            if (self.state in ("answered", "ended") and self.rtp and self.rtp.transport
                    and not self.rtp.transport.is_closing()):
                self.rtp.transport.sendto(pkt, self.rtp_peer)
        self.server.impair.send(self, direction, go)

    def _ts(self):
        return int((time.monotonic() - self.t0) * CLOCK)

    async def _sender(self):
        rate = float(self.line.cfg.get("rate_cps", 960))
        per_tick = max(1, int(round(rate * TICK)))
        while True:
            await asyncio.sleep(TICK)
            if not self.out:
                self.out_marker = True
                continue
            chunk = bytes(self.out[:per_tick])
            del self.out[:per_tick]
            self.seq = (self.seq + 1) & 0xFFFF
            pkt = rtp_packet(self.offer["data_pt"], self.seq, self._ts(),
                             self.ssrc, chunk, marker=self.out_marker)
            self.out_marker = False
            self.stats["tx"] += len(chunk)
            self._emit("to_caller", pkt)

    async def _dtmf_burst(self, digit):
        ev = DTMF_DIGITS.index(digit)
        ts = self._ts()
        self.stats["dtmf_tx"] += 1
        log(f"{self.label} dtmf {digit} (to caller)")
        self.event("dtmf", digit=digit, direction="to_caller")
        pt = self.offer["dtmf_pt"]
        for i in range(5):
            self.seq = (self.seq + 1) & 0xFFFF
            payload = bytes([ev, 10, ((i + 1) * 160) >> 8, ((i + 1) * 160) & 0xFF])
            self._emit("to_caller", rtp_packet(pt, self.seq, ts, self.ssrc, payload, marker=(i == 0)))
            await asyncio.sleep(TICK)
        for _ in range(3):
            self.seq = (self.seq + 1) & 0xFFFF
            payload = bytes([ev, 0x80 | 10, (800 >> 8) & 0xFF, 800 & 0xFF])
            self._emit("to_caller", rtp_packet(pt, self.seq, ts, self.ssrc, payload))


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

class SipProtocol(asyncio.DatagramProtocol):
    def __init__(self, server):
        self.server = server

    def connection_made(self, transport):
        self.server.sip_transport = transport

    def datagram_received(self, data, addr):
        try:
            self.server.handle(data, addr)
        except Exception as e:      # one bad datagram must not kill the server
            log(f"error handling datagram from {addr}: {e!r}")


class Server:
    def __init__(self, cfg):
        self.cfg = cfg
        self.calls = {}
        self.line_call = {}         # line name -> Call
        self.legs = {}              # callee-leg Call-ID -> Leg (bridged calls)
        self.registry = {}          # REGISTERed user -> (addr as seen on the wire, expiry time)
        self.rr = {}                # number -> next index for round_robin
        self._n = 0
        self.sip_transport = None
        self.port = None
        self.host = None
        self.impair = Impairment(cfg.get("delay_rules", []),
                                 random.Random(cfg.get("seed")))
        self.stopping = False
        self.lines = {}
        for name, lc in cfg["lines"].items():
            cls = LINE_TYPES[lc.get("type", "echo")]
            line = type("Line", (), {})()
            line.name, line.cfg = name, lc
            line.behavior = cls(name, lc)
            if isinstance(line.behavior, BridgeLine):
                line.behavior.line = line
            self.lines[name] = line

    def next_call_no(self):
        self._n += 1
        return self._n

    def event(self, kind, **kw):
        path = self.cfg.get("events_file")
        if path:
            try:
                with open(path, "a") as f:
                    f.write(json.dumps({"t": round(time.time(), 3), "event": kind, **kw}) + "\n")
            except OSError:
                pass

    def sip_send(self, data, addr):
        if self.sip_transport:
            self.sip_transport.sendto(data, addr)

    def advertise_for(self, peer_host):
        adv = self.cfg["listen"].get("advertise_host")
        if adv:
            return adv
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((peer_host, 9))
            return s.getsockname()[0]
        except OSError:
            return self.host if self.host != "0.0.0.0" else "127.0.0.1"
        finally:
            s.close()

    async def open_rtp(self, call):
        lo, hi = self.cfg.get("rtp_ports", [20000, 20999])
        loop = asyncio.get_running_loop()
        host = self.cfg["listen"].get("host", "0.0.0.0")
        first = lo + (lo % 2)
        ports = list(range(first, hi, 2))
        random.shuffle(ports)
        last = None
        for p in ports:
            ep = RtpEndpoint(call)
            try:
                await loop.create_datagram_endpoint(lambda: ep, local_addr=(host, p))
                return ep
            except OSError as e:
                last = e
        raise OSError(f"no free RTP port in {lo}-{hi}: {last}")

    def release(self, call):
        if self.line_call.get(call.line.name) is call:
            del self.line_call[call.line.name]

    def forget_later(self, cid, secs):
        asyncio.get_running_loop().call_later(secs, self.calls.pop, cid, None)

    # ---- SIP dispatch -----------------------------------------------------

    def handle(self, data, addr):
        msg = sc.parse(data)
        if msg is None:
            return
        if not msg.is_request:
            leg = self.legs.get(msg.call_id)
            if leg:
                leg.on_response(msg)
                return
            call = self.calls.get(msg.call_id)
            if call and msg.cseq()[1] == "BYE" and msg.status >= 200:
                call.on_bye_response()
            return
        m = msg.method
        if m == "OPTIONS":
            self.sip_send(sc.build_response(msg, 200, "OK", to_tag=sc.new_token(6),
                          extra=["Allow: INVITE, ACK, BYE, CANCEL, OPTIONS, REGISTER"]), addr)
        elif m == "REGISTER":
            contact = msg.get("contact")
            extra = [f"Contact: {contact}"] if contact else []
            user = sc.user_of(sc.uri_of(msg.get('from', '')))
            try:
                exp = int(msg.get("expires") or 60)
            except ValueError:
                exp = 60
            exp = max(0, min(exp, 120))   # short: the refresh keeps a NAT pinhole to the client open
            if exp:
                self.registry[user] = (addr, time.monotonic() + exp)
            else:
                self.registry.pop(user, None)
            self.sip_send(sc.build_response(msg, 200, "OK", to_tag=sc.new_token(6),
                          extra=extra + [f"Expires: {exp}"]), addr)
            log(f"register {user} from {addr[0]}:{addr[1]}" + ("" if exp else " (removed)"))
        elif m == "INVITE":
            self.on_invite(msg, addr)
        elif m == "ACK":
            call = self.calls.get(msg.call_id)
            if call:
                call.on_ack()
        elif m == "BYE":
            self.on_bye(msg, addr)
        elif m == "CANCEL":
            self.on_cancel(msg, addr)
        else:
            self.sip_send(sc.build_response(msg, 501, "Not Implemented"), addr)

    def on_invite(self, msg, addr):
        cid = msg.call_id
        call = self.calls.get(cid)
        if call:
            # retransmission (or re-INVITE on an answered call): repeat the last final answer
            if call.last_resp:
                self.sip_send(call.last_resp, call.peer)
            return
        if self.stopping:
            self.sip_send(sc.build_response(msg, 503, "Service Unavailable"), addr)
            return
        number = sc.user_of(sc.uri_of(msg.uri) or msg.uri)
        ncfg = self.cfg["numbers"].get(number)
        if ncfg is None:
            log(f"invite from {addr[0]} to unknown number {number!r} -> 404")
            self.event("not_found", number=number)
            self.sip_send(sc.build_response(msg, 404, "Not Found", to_tag=sc.new_token(6)), addr)
            return
        sdp = sc.parse_sdp(msg.body)
        if sdp is None:
            log(f"invite for {number} has no usable {sc.DATA_CODEC} offer -> 488")
            self.sip_send(sc.build_response(msg, 488, "Not Acceptable Here", to_tag=sc.new_token(6)), addr)
            return
        line = self.hunt(number, ncfg)
        if line is None:
            code = int(ncfg.get("busy_code", 486))
            log(f"busy: no free line for {number}")
            self.event("busy", number=number)
            self.sip_send(sc.build_response(msg, code, "Busy Here", to_tag=sc.new_token(6)), addr)
            return
        relay = getattr(line.behavior, "audio_relay", False)
        if sdp.get("g711") and not relay and not (line.cfg.get("spandsp") or hasattr(line.behavior, "span_args")):
            log(f"invite for {number} offers audio but line {line.name} has no spandsp -> 488")
            self.sip_send(sc.build_response(msg, 488, "Not Acceptable Here", to_tag=sc.new_token(6)), addr)
            return
        if sdp.get("g711") and not relay and not sph.available():
            log("invite offers audio but spanmodem is not built (build_spanmodem.sh) -> 488")
            self.sip_send(sc.build_response(msg, 488, "Not Acceptable Here", to_tag=sc.new_token(6)), addr)
            return
        call = Call(self, cid, msg, addr, number, line, sdp)
        self.calls[cid] = call
        self.line_call[line.name] = call
        log(f"{call.label} from {call.caller} dialed {number} -> {line.name}")
        call.event("invite", caller=call.caller)
        call.spawn(call.run_ringing())

    def _line_free(self, name):
        """A line takes one call at a time, except a shared one (a conference), which takes callers until full."""
        beh = self.lines[name].behavior
        if getattr(beh, "shared", False):
            return beh.admits(None)
        return name not in self.line_call

    def hunt(self, number, ncfg):
        names = ncfg["lines"]
        free = [n for n in names if self._line_free(n)]
        if not free:
            return None
        if ncfg.get("hunt", "first_free") == "round_robin":
            start = self.rr.get(number, 0)
            for i in range(len(names)):
                n = names[(start + i) % len(names)]
                if n in free:
                    self.rr[number] = (names.index(n) + 1) % len(names)
                    return self.lines[n]
        return self.lines[free[0]]

    def on_bye(self, msg, addr):
        leg = self.legs.get(msg.call_id)
        if leg is not None:
            self.sip_send(sc.build_response(msg, 200, "OK"), addr)
            leg.on_remote_bye()
            return
        call = self.calls.get(msg.call_id)
        if call is None:
            self.sip_send(sc.build_response(msg, 481, "Call/Transaction Does Not Exist"), addr)
            return
        self.sip_send(sc.build_response(msg, 200, "OK", to_tag=call.tag), addr)
        if call.state == "answered":
            call.end("bye from caller")

    def on_cancel(self, msg, addr):
        call = self.calls.get(msg.call_id)
        if call is None:
            self.sip_send(sc.build_response(msg, 481, "Call/Transaction Does Not Exist"), addr)
            return
        self.sip_send(sc.build_response(msg, 200, "OK", to_tag=call.tag), addr)
        if call.state == "ringing":
            for t in list(call.tasks):
                t.cancel()
            call.reject(487, "Request Terminated")
            log(f"{call.label} cancelled by caller")

    # ---- lifecycle --------------------------------------------------------

    async def shutdown(self):
        self.stopping = True
        active = [c for c in self.calls.values() if c.state == "answered"]
        for c in active:
            c.hangup("server shutting down")
        if active:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(c.bye_done.wait() for c in active)), 2.0)
            except asyncio.TimeoutError:
                pass
        for c in list(self.calls.values()):
            if c.state == "ringing":
                for t in list(c.tasks):
                    t.cancel()
                c.reject(503, "Service Unavailable")
            elif c.state in ("answered", "ended"):
                c.finish()
        await asyncio.gather(*(l.behavior.shutdown() for l in self.lines.values()))
        log("shutdown complete")


# ---------------------------------------------------------------------------
# config + main
# ---------------------------------------------------------------------------

def load_config(path):
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        raise ConfigError(f"cannot read config {path}: {e}")
    cfg.setdefault("listen", {})
    cfg["listen"].setdefault("host", "0.0.0.0")
    cfg["listen"].setdefault("port", 5060)
    cfg.setdefault("lines", {})
    cfg.setdefault("numbers", {})
    if not cfg["numbers"]:
        raise ConfigError("config has no 'numbers'")
    for num, nc in cfg["numbers"].items():
        if not nc.get("lines"):
            raise ConfigError(f"number {num}: 'lines' is empty")
        for ln in nc["lines"]:
            if ln not in cfg["lines"]:
                raise ConfigError(f"number {num}: line {ln!r} is not defined under 'lines'")
    for ln, lc in cfg["lines"].items():
        t = lc.get("type", "echo")
        if t not in LINE_TYPES:
            raise ConfigError(f"line {ln}: unknown type {t!r} (have: {', '.join(sorted(LINE_TYPES))})")
        if lc.get("behavior", "answer") not in ("answer", "busy", "reject", "no_answer"):
            raise ConfigError(f"line {ln}: bad behavior {lc['behavior']!r}")
        if t in ("bridge", "audiobridge"):
            tgt = str(lc.get("target", ""))
            host, _, port = tgt.rpartition(":")
            if lc.get("target_user") is None and not (host and port.isdigit()):
                raise ConfigError(f"line {ln}: bridge needs 'target' as host:port (the callee mock's listen_port)")
        if t == "exec":
            cmd = lc.get("command")
            if not (isinstance(cmd, list) and cmd and all(isinstance(c, str) for c in cmd)):
                raise ConfigError(f"line {ln}: exec needs 'command' as a non-empty list of strings")
        if t == "modem":
            for j, r in enumerate(lc.get("rules", [])):
                try:
                    re.compile(r.get("expect", ""))
                except re.error as e:
                    raise ConfigError(f"line {ln}: rules[{j}].expect is not a valid regex: {e}")
            if lc.get("transform", "none") not in ("none", "upper", "lower", "reverse"):
                raise ConfigError(f"line {ln}: bad transform {lc['transform']!r}")
    for i, r in enumerate(cfg.get("delay_rules", [])):
        if r.get("direction", "both") not in ("both", "to_line", "to_caller"):
            raise ConfigError(f"delay_rules[{i}]: bad direction {r['direction']!r}")
    return cfg


async def amain(cfg):
    server = Server(cfg)
    loop = asyncio.get_running_loop()
    host, port = cfg["listen"]["host"], int(cfg["listen"]["port"])
    transport, _ = await loop.create_datagram_endpoint(
        lambda: SipProtocol(server), local_addr=(host, port))
    server.host = host
    server.port = transport.get_extra_info("sockname")[1]
    stop = asyncio.Event()

    def on_signal():
        if not stop.is_set():
            log("signal received, shutting down")
            stop.set()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, on_signal)
        except NotImplementedError:
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(on_signal))

    for num, nc in cfg["numbers"].items():
        log(f"number {num} -> {', '.join(nc['lines'])} ({nc.get('hunt', 'first_free')})")
    log(f"READY sip={host}:{server.port}")
    if cfg.get("ready_file"):
        with open(cfg["ready_file"], "w") as f:
            f.write(f"{host}:{server.port}\n")
    await stop.wait()
    await server.shutdown()
    transport.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=os.environ.get("HOSTSERVICE_CONFIG"),
                    help="JSON config path (default: $HOSTSERVICE_CONFIG)")
    args = ap.parse_args()
    if not args.config:
        print("sipserver: no --config and no $HOSTSERVICE_CONFIG", file=sys.stderr)
        return 2
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"sipserver: config error: {e}", file=sys.stderr)
        return 2
    try:
        asyncio.run(amain(cfg))
    except OSError as e:
        print(f"sipserver: cannot bind {cfg['listen']['host']}:{cfg['listen']['port']}: {e}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
