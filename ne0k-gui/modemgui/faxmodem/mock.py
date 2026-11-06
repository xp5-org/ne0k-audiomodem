#!/usr/bin/env python3
"""Hayes modem mock: an 86Box COM port on one side, a SIP call to sipserver.py
on the other. One process serves every serial port of ONE 86Box instance.

Run in the FOREGROUND under test_start_hostservice, AFTER the test_start_86box
step for that VM (the PTYs only exist once 86Box is running):

    {"action": "test_start_hostservice", "param": {
        "name": "mock_hostC",
        "command": ["python3", "{projbasedir}{projdir}/modemmock.py",
                    "--config", "{hostservice_config}"],
        "json_config": {...see below...},
        "ready_regex": "READY modems=2"}}

CONFIG (JSON)
  {
    "vm_path": "/testsrc/sourcedir/86box_sipmodem/sx_box_c",  # 86Box `-P` path
    "find_timeout": 60,
    "sip": {"server": "127.0.0.1:5060", "local_host": null},
    "modems": [
      {"port": 1, "name": "hostC-com1", "user": "3001",   # caller id (From user)
       "baud": 9600,                # CONNECT <baud>; data paced to baud/10 cps
       "connect_delay_ms": 0,       # "training" pause between 200 OK and CONNECT
       "escape_guard_s": 1.0,       # +++ guard time (S12 * 20ms if unset)
       "init": "ATE0",              # run silently at startup (no OK on the wire)
       "result_delay_ms": 0,        # pause before BUSY / NO CARRIER / NO ANSWER (dial
                                    # failures AND a server hangup), so a DOS guest has
                                    # reached its read before it lands
       "ok_delay_ms": 0,            # the same for every OK (ATH answers within
                                    # milliseconds, before `copy com1` has started)
       "eof_after_result": false,   # append Ctrl-Z (0x1A) after every result except
                                    # CONNECT: DOS `COPY COM1 file` only returns on it
       "dtmf_report": "log",        # log | unsolicited (prints DTMF=5 in command mode)
       "s7": 60,                    # dial timeout, seconds
       "pty": null}                 # explicit slave path; skips 86Box discovery
    ],
    "events_file": null, "ready_file": null
  }

WHICH PTY IS WHICH COM PORT. 86Box's stdio ("Virtual Console (COM)") devices
each hold one /dev/ptmx master, and nothing labels them. Ports are matched to
masters by ascending fd number: the lowest fd is the lowest configured "port".
That relies on 86Box creating its serial devices in COM order -- verify it on
your build; if it ever doesn't hold, give each modem an explicit "pty". The
number of masters found must equal the number of modems without a "pty", or
the mock refuses to guess.

LISTEN (ATM / ATL). Add "listen": true (or an object, per modem or top level) and each modem plays what
crosses its SIP call through PulseAudio, one stream per modem; see modemaudio.py for the cues and options.
ATM0..3 picks when it is audible (off / until carrier / always / after dialling until carrier), ATL0..3 the
volume. Without "listen" the commands are stored and silent.

HAYES SUBSET. AT, E V Q X M L Z &F &C &D &K &W &Y H I O S<n>[=v|?], D<number>
(T/P/W/, ignored), +VTS=<digits> (send DTMF during a call), and the +++ escape
with guard time. `ATD` places a SIP INVITE to the number; 180 is silent, 200
prints CONNECT <baud>, 486/600 BUSY, 408/480 NO ANSWER, anything else NO
CARRIER. A byte received while dialling aborts the call (CANCEL).

INCOMING CALLS. The mock also answers a SIP INVITE (from a sipserver.py `bridge` line, or any UAC) on its
SIP port: set "listen_port" so the server can be told where (the server's bridge `target`). The modem
prints RING every "ring_interval_ms" (3000), answers with ATA or automatically once S0 rings have been
counted (ATS0=1), and gives up with 480 after "ring_timeout_s" (30). A modem that is off-hook (ATH1, a
call up, or dialling) answers 486, so the caller prints BUSY; ATH while it rings declines (603).
ATA in class 0 -> CONNECT <baud> and online mode like a dial; in class 2 -> the fax answer below.

LINE PICKUP AND DIAL TONE. ATH1 picks the line up (off-hook, "modem X off-hook (pickup, dial tone)"),
ATH0 / ATH puts it down. A dial seizes the line itself. With ATX2 or higher (the default is 4), or a W in
the dial string, a dial needs dial tone: a modem configured "dialtone": false (after "dialtone_wait_s",
1.0) answers NO DIALTONE and goes back on-hook; ATX0 skips the check. "dialtone_delay_ms" delays the
dial the way a slow tone does; each comma in a dial string pauses "comma_pause_s" (2.0).

FAX (opt-in: a modem needs a "fax" object in its config, {} is enough). Identification: AT+FCLASS=? / ? / =n
(classes "0,1,2" by default, reset to 0 by ATZ and AT&F), +FMFR? +FMDL? +FREV?, and in class 1 +FTM=?
+FRM=? +FTH=? +FRH=? +FTS=? +FRS=?; every value can be overridden in the "fax" object (classes, mfr,
model, rev, speeds, hdlc_speeds, silence, dcc), "ati": {"0": "...", "3": "..."} sets the ATI<n> replies.
CLASS 2 (the pre-2.0 TR-29.2 / Rockwell dialect DOS-era software such as DosFax LITE speaks; the mock
IS the modem and talks to the far fax end in the faxcodec.py frame protocol, see that file):
  settings   +FDCC +FDIS +FLID +FBOR +FCR +FCQ +FPHCTO +FBUG +FCT +FNR +FAA (set, ?, =?)
  sending    ATD -> +FCON, +FCSI, +FDIS, OK;  +FDT -> +FDCS, CONNECT, XON (DC1), DLE-shielded page
             data, DLE ETX -> OK;  +FET=0|1|2 (more pages / new document / last) -> +FPTS: n and, on
             the last page, +FHNG: 0 then OK. The far end's go-ahead (CFR) gates +FDT's CONNECT.
  receiving  RING, ATA or S0 -> +FCON, then +FTSI and +FDCS and OK once the caller's parameters
             arrive;  +FDR -> +FDCS, CONNECT, DC2, page data (DLE-shielded) ... DLE ETX, +FPTS: n,
             +FET: n, OK;  the NEXT +FDR sends the page response (set it first with +FPTS=n) and, after
             the last page, ends the session (+FHNG: 0, OK).
  failures   +FHNG: 20 (far end silent / no confirmation), +FHNG: 10 (session lost).
fax options ("fax" object): connect_timeout_s (8), page_timeout_s (15), train_timeout_s (12),
dcn_timeout_s (3), train_ms (500), answer_delay_ms (500). Class 1 actions (+FTH, +FRM ...) are not
implemented and answer ERROR.

Stdout is the test log. Stable phrases for test_filecontains:
  modem hostC-com1 COM1 -> /dev/pts/7        modem hostC-com1 dialing 1234
  modem hostC-com1 CONNECT 9600              modem hostC-com1 BUSY
  modem hostC-com1 NO CARRIER (...)          modem hostC-com1 dtmf 5 (from line)
  modem hostC-com1 data from line 12 bytes: b'...'
  modem hostB incoming call from 1001        modem hostB RING 1
  modem hostB auto-answer (S0=1)             modem hostB answering (ATA)
  modem hostB off-hook (pickup, dial tone)   modem hostB NO DIALTONE
  modem hostA fax: connected (+FCON)         modem hostA fax: session ended (+FHNG: 0) ...
  READY modems=2
POSIX only (Linux, macOS). Windows has no PTY to attach to.
"""
import argparse
import asyncio
import json
import os
import re
import signal
import socket
import struct
import sys
import time
import random

import types

HERE = os.path.dirname(os.path.abspath(__file__))
from . import sipcore as sc  # noqa: E402
from . import ptyhelpers  # noqa: E402
from . import faxcodec as fc  # noqa: E402


class _SilentSpeaker:
    """No local loudspeaker in NE0K: every audio cue is a no-op."""

    def __getattr__(self, name):
        return lambda *a, **k: None

    async def close(self):
        pass


ma = types.SimpleNamespace(DTMF_KEYS="0123456789*#ABCD", Speaker=lambda *a, **k: _SilentSpeaker())
sph = types.SimpleNamespace(available=lambda: False)

DTMF_DIGITS = "0123456789*#ABCD"
TICK = 0.02
REG_EXPIRES = 120           # asked for; the server grants <= this and the refresh runs at half the grant (max 25 s)
CLOCK = 8000
RESULT_CODES = {"OK": 0, "CONNECT": 1, "RING": 2, "NO CARRIER": 3, "ERROR": 4,
                "NO DIALTONE": 6, "BUSY": 7, "NO ANSWER": 8}
FAIL_WORDS = {486: "BUSY", 600: "BUSY", 408: "NO ANSWER", 480: "NO ANSWER"}
HANDLED = "handled"        # cmd_plus: the command already produced its own final response
NOT_MINE = object()        # fax_command: not an +F command


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------------------
# one modem
# ---------------------------------------------------------------------------

class SipUa(asyncio.DatagramProtocol):
    def __init__(self, modem):
        self.modem = modem

    def connection_made(self, transport):
        self.modem.sip = transport

    def datagram_received(self, data, addr):
        try:
            self.modem.on_sip(data, addr)
        except Exception as e:
            log(f"modem {self.modem.name} sip error: {e!r}")


class RtpUa(asyncio.DatagramProtocol):
    def __init__(self, modem):
        self.modem = modem
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        self.modem.on_rtp(data, addr)


def rtp_packet(pt, seq, ts, ssrc, payload, marker=False):
    return struct.pack("!BBHII", 0x80, (0x80 if marker else 0) | pt,
                       seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc) + payload


class Modem:
    def __init__(self, app, mcfg, pty_path):
        self.app = app
        self.cfg = mcfg
        self.port = int(mcfg["port"])
        self.name = mcfg.get("name", f"com{self.port}")
        self.user = str(mcfg.get("user", self.name))
        self.baud = int(mcfg.get("baud", 9600))
        self.base_baud = self.baud  # a spandsp call sets self.baud to the trained rate
        self.pty_path = pty_path
        self.fd = None
        self.sip = None
        self.rtp = None
        self.echo, self.verbose, self.quiet = True, True, False
        self.fclass = "0"           # +FCLASS service class; only used when the modem has a "fax" config
        self.offhook = False        # line seized (ATH1 pickup, or a dial in progress / a call up)
        self.xlevel = 4             # ATX: >= 2 makes a dial wait for / require dial tone
        self.f2 = self.f2_defaults()
        self.fax_q = None           # frames from the far fax machine (faxcodec wire protocol)
        self.fax_df = fc.Deframer()
        self.fax_dle = fc.DleDecoder()
        self.fax_bytes = 0
        self.pbuf = bytearray()     # page data waiting to be framed (feed_fax_data)
        self.pbuf_timer = None
        self.ring_task = None
        self.sregs = {0: 0, 7: int(mcfg.get("s7", 60)), 12: 50}
        self.state = "command"      # command | dialing | online | online_cmd
        self.cmdbuf = bytearray()
        self.lf_until = 0.0         # an LF before this time belongs to the CR just seen
        self.cmdq = None
        self.outbuf = bytearray()
        self.writing = False
        self.tx = bytearray()
        self.rx_pending = bytearray()
        self.call = None
        self.dial_fut = None
        self.tasks = set()
        self.tx_task = None
        # +++ escape tracking
        self.last_byte_t = 0.0
        self.plus_n = 0
        self.held = b""
        self.esc_timer = None
        # rtp send side
        self.seq = random.getrandbits(16)
        self.ssrc = random.getrandbits(32)
        self.t0 = time.monotonic()
        self._dtmf_seen = set()
        self.stats = {"rx": 0, "tx": 0}
        self.audio = ma.Speaker(self.name, mcfg.get("listen", app.listen), log)
        self.ne0k = mcfg.get("media") == "ne0k"     # the line is OFDM data inside G.711 audio
        self.pipe = None
        self.fclass = str(mcfg.get("fclass", "0"))

    # ---- plumbing --------------------------------------------------------

    @property
    def tag(self):
        return f"modem {self.name}"

    def spawn(self, coro):
        t = asyncio.ensure_future(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    def event(self, kind, **kw):
        self.app.event(kind, modem=self.name, port=self.port, **kw)

    async def start(self):
        self.cmdq = asyncio.Queue()
        self.fd = ptyhelpers.open_raw(self.pty_path)
        loop = asyncio.get_running_loop()
        loop.add_reader(self.fd, self.on_serial_readable)
        bind = self.app.local_host or "0.0.0.0"
        await loop.create_datagram_endpoint(lambda: SipUa(self),
                                            local_addr=(bind, int(self.cfg.get("listen_port", 0))))
        self.reg_ok = asyncio.Event()
        if self.app.register:
            self.reg_cid, self.reg_tag, self.reg_cseq = sc.new_token(14), sc.new_token(6), 0
            self.reg_resp, self.reg_status, self.reg_expires = asyncio.Event(), None, 0
            self.spawn(self._register_loop())
        init = str(self.cfg.get("init", "")).upper()
        if init:
            saved = self.quiet
            self.quiet = True
            await self.run_command(init[init.find("AT") + 2:] if "AT" in init else init)
            if "Q" not in init:
                self.quiet = saved
        self.spawn(self._command_worker())
        log(f"{self.tag} COM{self.port} -> {self.pty_path}")

    # ---- registration: the server finds this modem by user, at the address its REGISTER arrived from. It is
    # sent from the SIP socket that places and takes calls, so the NAT pinhole is the one the INVITE comes back to.

    def _reg_request(self, expires):
        srv, adv = self.app.server, self.app.advertise
        sip_port = self.sip.get_extra_info("sockname")[1]
        who = f"<sip:{self.user}@{srv[0]}>"
        self.reg_cseq += 1
        return sc.build_request("REGISTER", f"sip:{srv[0]}:{srv[1]}", f"{adv}:{sip_port}",
                                f"{who};tag={self.reg_tag}", who, self.reg_cid, self.reg_cseq,
                                contact=f"<sip:{self.user}@{adv}:{sip_port}>", extra=[f"Expires: {expires}"])

    def _on_register_response(self, m):
        if m.cseq()[0] != self.reg_cseq or m.status < 200:
            return                                  # 1xx, or a late answer to an earlier try
        granted = m.get("expires") or ""
        if not granted.strip().isdigit():
            hit = re.search(r"expires=(\d+)", m.get("contact", "") or "", re.I)
            granted = hit.group(1) if hit else str(REG_EXPIRES)
        self.reg_status, self.reg_expires = m.status, int(granted)
        self.reg_resp.set()

    async def _register_loop(self):
        fails = 0
        while True:
            self.reg_resp.clear()
            self.reg_status = None
            self.sip.sendto(self._reg_request(REG_EXPIRES), self.app.server)
            try:
                await asyncio.wait_for(self.reg_resp.wait(), 2.0)
            except asyncio.TimeoutError:
                fails += 1
                if fails in (1, 10):
                    log(f"{self.tag} register: no answer from {self.app.server[0]}:{self.app.server[1]} (try {fails})")
                continue
            if 200 <= self.reg_status < 300:
                every = max(5.0, min(25.0, self.reg_expires / 2))
                if not self.reg_ok.is_set() or fails:
                    log(f"{self.tag} register: ok user={self.user} expires={self.reg_expires}s refresh every {every:.0f}s")
                    self.event("registered", user=self.user)
                self.reg_ok.set()
                fails = 0
                await asyncio.sleep(every)
            else:
                fails += 1
                if fails in (1, 10):
                    log(f"{self.tag} register: refused {self.reg_status} (try {fails})")
                await asyncio.sleep(5.0)

    def unregister(self):
        if self.app.register and self.sip and self.reg_ok.is_set():
            self.sip.sendto(self._reg_request(0), self.app.server)

    def guard(self):
        g = self.cfg.get("escape_guard_s")
        return float(g) if g is not None else self.sregs[12] * 0.02

    # ---- serial side -------------------------------------------------------

    def write(self, data):
        self.outbuf.extend(data)
        self._flush()

    def _flush(self):
        loop = asyncio.get_running_loop()
        while self.outbuf and self.fd is not None:
            try:
                n = os.write(self.fd, bytes(self.outbuf[:1024]))
            except BlockingIOError:
                if not self.writing:
                    loop.add_writer(self.fd, self._flush)
                    self.writing = True
                return
            except OSError:
                self.outbuf.clear()
                return
            del self.outbuf[:n]
        if self.writing and self.fd is not None:
            loop.remove_writer(self.fd)
            self.writing = False

    def respond(self, word, extra=""):
        if self.quiet:
            return
        if self.verbose:
            self.write(b"\r\n" + (word + extra).encode() + b"\r\n")
        else:
            self.write(f"{RESULT_CODES.get(word, 4)}\r".encode())
        if self.cfg.get("eof_after_result") and word != "CONNECT":
            self.write(b"\x1a")

    def on_serial_readable(self):
        try:
            data = os.read(self.fd, 4096)
        except BlockingIOError:
            return
        except OSError:
            data = b""
        if not data:
            log(f"{self.tag} serial closed (86Box gone?)")
            self.event("serial_closed")
            self.close_serial()
            self.spawn(self.hangup_call())
            self.app.serial_closed(self)
            return
        self.feed(data)

    def close_serial(self):
        if self.fd is not None:
            asyncio.get_running_loop().remove_reader(self.fd)
            if self.writing:
                asyncio.get_running_loop().remove_writer(self.fd)
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def feed(self, data):
        # DOS `echo X>com1` sends X, CR, LF -- and the LF can land in a SEPARATE read,
        # after the CR has already started the command (found on the first real run of
        # the variant list: a late LF aborted the dial that its own CR had begun, or
        # would have been forwarded to the line as data after ATO). The one LF that
        # directly follows a command's CR is part of the line ending, so drop it.
        lf_ok = time.monotonic() < self.lf_until
        self.lf_until = 0.0
        if lf_ok and data[:1] == b"\n":
            data = data[1:]
            if not data:
                return
        if self.state == "dialing":
            if self.call and self.call["state"] in ("inviting", "connected"):
                log(f"{self.tag} dial aborted by serial input")
                self.spawn(self.hangup_call())
            return
        if self.state == "online":
            self.feed_online(data)
        elif self.state == "fax_data":
            self.feed_fax_data(data)
        elif self.state == "fax_rx_data":
            return                      # the DTE is listening; nothing it says matters here
        else:
            self.feed_command(data)

    # ---- command mode ------------------------------------------------------

    def feed_command(self, data):
        for b in data:
            if self.echo:
                self.write(bytes([b]))
            if b == 13:
                line = self.cmdbuf.decode("latin-1")
                self.cmdbuf.clear()
                s = line.strip().upper()
                i = s.find("AT")
                if i >= 0:
                    self.cmdq.put_nowait(s[i + 2:])
            elif b in (8, 127):
                del self.cmdbuf[-1:]
            elif b != 10 and len(self.cmdbuf) < 255:
                self.cmdbuf.append(b)
        if data.endswith(b"\r"):
            self.lf_until = time.monotonic() + 1.0

    async def _command_worker(self):
        while True:
            line = await self.cmdq.get()
            log(f"{self.tag} cmd 'AT{line}'")
            try:
                await self.run_command(line)
            except Exception as e:
                log(f"{self.tag} command error: {e!r}")
                self.respond("ERROR")

    async def run_command(self, s):
        i = 0
        while i < len(s):
            c = s[i]
            if c == " ":
                i += 1
            elif c == "D":
                await self.cmd_dial(s[i + 1:])
                return
            elif c == "+":
                # "+FBOR=0;+FCLASS=2": a semicolon ends an extended command (DosFax chains them),
                # unless it sits inside a quoted string (+FLID="A;B")
                j, quoted = i, False
                while j < len(s) and (quoted or s[j] != ";"):
                    quoted ^= s[j] == '"'
                    j += 1
                r = await self.cmd_plus(s[i:j])
                if r == HANDLED:
                    return
                if not r:
                    self.respond("ERROR")
                    return
                if j >= len(s):
                    break
                i = j + 1
            elif c == "&":
                m = re.match(r"&([A-Z])(\d*)", s[i:])
                if not m:
                    return self.respond("ERROR")
                if m.group(1) == "F":
                    self.reset_settings()
                i += m.end()
            elif c == "S":
                m = re.match(r"S(\d+)(?:=(\d+)|(\?))?", s[i:])
                if not m:
                    return self.respond("ERROR")
                n = int(m.group(1))
                if m.group(3):
                    self.write(f"\r\n{self.sregs.get(n, 0):03d}".encode())
                elif m.group(2) is not None:
                    self.sregs[n] = int(m.group(2))
                i += m.end()
            else:
                m = re.match(r"([A-Z])(\d*)", s[i:])
                if not m:
                    return self.respond("ERROR")
                cmd, arg = m.group(1), m.group(2)
                i += m.end()
                if cmd == "E":
                    self.echo = arg != "0"
                elif cmd == "V":
                    self.verbose = arg != "0"
                elif cmd == "Q":
                    self.quiet = arg == "1"
                elif cmd == "Z":
                    self.reset_settings()
                elif cmd == "H":
                    if arg == "1":
                        self.pickup()
                    else:
                        await self.hangup_call()
                        self.putdown()
                elif cmd == "X":
                    self.xlevel = int(arg or 0)
                elif cmd == "O":
                    return await self.cmd_online()
                elif cmd == "A":
                    return await self.cmd_answer()
                elif cmd == "I":
                    ati = self.cfg.get("ati") or {}
                    self.write(("\r\n" + str(ati.get(arg or "0", "sipmodem mock 1.0"))).encode("latin-1"))
                elif cmd in "ML":
                    v = int(arg or 0)
                    if v > 3:
                        return self.respond("ERROR")
                    (self.audio.set_mode if cmd == "M" else self.audio.set_volume)(v)
                elif cmd in "TP":
                    pass
                else:
                    return self.respond("ERROR")
        await self.ok()

    async def ok(self):
        delay = float(self.cfg.get("ok_delay_ms", 0)) / 1000.0
        if delay:
            await asyncio.sleep(delay)
        self.respond("OK")

    def reset_settings(self):
        self.echo, self.verbose, self.quiet = True, True, False
        self.fclass = "0"
        self.xlevel = 4
        self.f2 = self.f2_defaults()
        self.sregs = {0: 0, 7: int(self.cfg.get("s7", 60)), 12: 50}
        self.audio.reset()

    # ---- line pickup / dial tone ------------------------------------------

    def pickup(self):
        if not self.offhook:
            self.offhook = True
            log(f"{self.tag} off-hook (pickup, dial tone)")
            self.event("offhook")
            if self.cfg.get("dialtone", True):
                self.audio.dial_tone()

    def putdown(self):
        if self.offhook:
            self.offhook = False
            log(f"{self.tag} on-hook")
            self.event("onhook")

    async def seize_line(self, dial):
        """Take the line for a dial. Returns False (and answers NO DIALTONE) when the
        line has no dial tone and the modem is set to look for it (ATX2 and up, or a W in the
        dial string). cfg: dialtone (default true), dialtone_delay_ms, dialtone_wait_s."""
        if not self.offhook:
            self.offhook = True
            log(f"{self.tag} off-hook (line seized for dial)")
            self.event("offhook")
            if self.cfg.get("dialtone", True):
                self.audio.dial_tone()
        if self.xlevel >= 2 or "W" in dial:
            if not self.cfg.get("dialtone", True):
                await asyncio.sleep(float(self.cfg.get("dialtone_wait_s", 1.0)))
                log(f"{self.tag} NO DIALTONE")
                self.event("result", word="NO DIALTONE")
                self.putdown()
                self.respond("NO DIALTONE")
                return False
            delay = float(self.cfg.get("dialtone_delay_ms", 0)) / 1000.0
            if delay:
                await asyncio.sleep(delay)
        return True

    # ---- incoming calls: RING, ATA, S0 auto-answer ----------------------------------

    def on_incoming_invite(self, m, addr):
        sip_reply = self.app.sip_reply
        if self.call is not None or self.offhook or self.state != "command":
            log(f"{self.tag} incoming INVITE refused (line busy)")
            self.event("incoming_busy")
            sip_reply(self, sc.build_response(m, 486, "Busy Here", to_tag=sc.new_token(6)), addr)
            return
        info = sc.parse_sdp(m.body)
        if info is None:
            log(f"{self.tag} incoming INVITE without a usable offer")
            sip_reply(self, sc.build_response(m, 488, "Not Acceptable Here", to_tag=sc.new_token(6)), addr)
            return
        adv = self.app.advertise
        sip_port = self.sip.get_extra_info("sockname")[1]
        tag = sc.new_token(6)
        to_hdr = m.get("to", "")
        caller = sc.user_of(sc.uri_of(m.get("from", ""))) or m.get("from", "")
        self.call = {"cid": m.call_id, "incoming": True, "state": "ringing", "ltag": tag, "invite": m,
                     "peer": addr, "number": caller,
                     "uri": sc.uri_of(m.get("contact", "")) or sc.uri_of(m.get("from", "")),
                     "from": to_hdr if sc.get_tag(to_hdr) else f"{to_hdr};tag={tag}",
                     "to": m.get("from", ""), "via": f"{adv}:{sip_port}",
                     "contact": f"<sip:{self.user}@{adv}:{sip_port}>",
                     "remote_rtp": (info["addr"], info["port"]),
                     "data_pt": info["data_pt"], "dtmf_pt": info["dtmf_pt"],
                     "g711": bool(info.get("g711")) and not self.ne0k, "law": info.get("law"),
                     "ne0k": self.ne0k and bool(info.get("g711")),
                     "bye_evt": asyncio.Event(), "ringing": True, "acked": False,
                     "fax": self.fax_active()}
        self.state = "ringing"
        sip_reply(self, sc.build_response(m, 180, "Ringing", to_tag=tag), addr)
        log(f"{self.tag} incoming call from {caller}")
        self.event("incoming", caller=caller)
        self.ring_task = self.spawn(self._ring_loop())

    async def _ring_loop(self):
        call = self.call
        interval = float(self.cfg.get("ring_interval_ms", 3000)) / 1000.0
        deadline = time.monotonic() + float(self.cfg.get("ring_timeout_s", 30))
        n = 0
        while self.call is call and call["state"] == "ringing":
            n += 1
            log(f"{self.tag} RING {n}")
            self.event("ring", n=n)
            self.respond("RING")
            s0 = self.sregs.get(0, 0)
            if s0 and n >= s0:
                log(f"{self.tag} auto-answer (S0={s0})")
                await self.answer_incoming()
                return
            await asyncio.sleep(interval)
            if time.monotonic() >= deadline and self.call is call and call["state"] == "ringing":
                log(f"{self.tag} nobody answered: 480")
                self.app.sip_reply(self, sc.build_response(call["invite"], 480, "Temporarily Unavailable",
                                                           to_tag=call["ltag"]), call["peer"])
                self.cleanup_call()
                return

    async def cmd_answer(self):
        call = self.call
        if not call or not call.get("incoming") or call["state"] != "ringing":
            return self.respond("ERROR")
        log(f"{self.tag} answering (ATA)")
        await self.answer_incoming()

    async def answer_incoming(self):
        call = self.call
        if not call or call["state"] != "ringing":
            return
        if self.ring_task and self.ring_task is not asyncio.current_task():
            self.ring_task.cancel()
        self.ring_task = None
        loop = asyncio.get_running_loop()
        try:
            rtp = RtpUa(self)
            await loop.create_datagram_endpoint(lambda: rtp, local_addr=(self.app.local_host or "0.0.0.0", 0))
        except OSError as e:
            log(f"{self.tag} cannot open RTP: {e}")
            self.app.sip_reply(self, sc.build_response(call["invite"], 500, "No RTP port",
                                                       to_tag=call["ltag"]), call["peer"])
            self.cleanup_call()
            return
        self.rtp = rtp
        rtp_port = rtp.transport.get_extra_info("sockname")[1]
        adv = self.app.advertise
        # fax or data is decided NOW: a fax program sends +FCLASS=2 after it has seen RING
        call["fax"] = self.fax_active()
        if call["fax"]:
            self.fax_q, self.fax_df = asyncio.Queue(), fc.Deframer()
        if call.get("g711") or call.get("ne0k"):
            sdp = sc.build_sdp_g711(adv, rtp_port, (call["data_pt"],), call["dtmf_pt"])
        else:
            sdp = sc.build_sdp(adv, rtp_port, call["data_pt"], call["dtmf_pt"])
        resp = sc.build_response(call["invite"], 200, "OK", to_tag=call["ltag"],
                                 extra=[f"Contact: {call['contact']}"], body=sdp)
        call["state"] = "connected"
        self.offhook = True
        self.event("offhook")
        self.audio.answered(fax=bool(call.get("fax")))
        self.app.sip_reply(self, resp, call["peer"])
        self.spawn(self._retransmit_200(call, resp))
        if call.get("ne0k") and not await self.ne0k_up(call, caller=False):
            return self.respond("NO CARRIER")
        if call.get("g711") and not call["fax"]:
            return await self.span_begin("answerer")
        if call.get("g711"):
            log(f"{self.tag} fax: receiving over G.711 (spandsp) is not built; hanging up")
            await self.hangup_call()
            return self.respond("NO CARRIER")
        if call["fax"]:
            return await self.fax_answer_begin()
        delay = float(self.cfg.get("connect_delay_ms", 0)) / 1000.0
        if delay:
            await asyncio.sleep(delay)
        if self.call is not call or call["state"] != "connected":
            return self.respond("NO CARRIER")
        self.tx_task = self.spawn(self._tx_loop())
        log(f"{self.tag} CONNECT {self.baud} (answered)")
        self.event("connect")
        self.respond("CONNECT", f" {self.baud}")
        self.go_online()

    async def _retransmit_200(self, call, resp):
        delay = 0.5
        for _ in range(5):
            await asyncio.sleep(delay)
            if call.get("acked") or self.call is not call or call["state"] != "connected":
                return
            self.app.sip_reply(self, resp, call["peer"])
            delay = min(delay * 2, 4.0)

    # ---- fax: receiving ---------------------------------------------------------

    def fax_on_frame(self, frame):
        """A frame from the far fax end. While page data is being received it is written straight to
        the DTE; everything else goes to the queue the command handlers wait on."""
        t, payload = frame
        call = self.call
        if t == "D":
            call["fax_ending"] = True
            if self.state == "fax_rx_data":
                self.state = "command"
                self.write(b"\x10\x03")
                self.spawn(self.fax_hangup(int(payload[:1] or b"0") if payload[:1].isdigit() else 0,
                                           "sender disconnected mid-page", call=call))
                return
        elif self.state == "fax_rx_data" and t == "P":
            self.fax_bytes += len(payload)
            call.setdefault("rx_page", bytearray()).extend(payload)
            self.write(fc.dle_encode(payload))
            return
        elif self.state == "fax_rx_data" and t == "N":
            n = payload.decode("latin-1", "replace") or "2"
            self.state = "command"
            self.write(b"\x10\x03")
            call["fax_resp_due"], call["fax_last"] = True, n == "2"
            call["fax_pages"] = call.get("fax_pages", 0) + 1
            log(f"{self.tag} fax: page {call['fax_pages']} received, {self.fax_bytes} bytes (post-page {n})")
            self._save_page(call)
            self.spawn(self._fax_rx_page_done(n))
            return
        if self.fax_q is not None:
            self.fax_q.put_nowait(frame)

    def _save_page(self, call):
        sd = (self.cfg.get("fax") or {}).get("save_dir")
        data = bytes(call.pop("rx_page", b""))
        if not sd:
            return
        os.makedirs(sd, exist_ok=True)
        path = os.path.join(sd, f"{self.name}_rx_{int(time.time())}_p{call['fax_pages']}.bin")
        with open(path, "wb") as f:
            f.write(data)
        log(f"{self.tag} fax: page {call['fax_pages']} saved to {path} ({len(data)} bytes)")

    async def _fax_rx_page_done(self, n):
        self.write(f"\r\n+FPTS: 1\r\n+FET: {n}\r\n".encode())
        await self.ok()

    async def fax_answer_begin(self):
        """Answering in class 2: offer id + capabilities, report +FCON, then the caller's id and
        parameters (+FTSI, +FDCS) and OK -- or +FHNG when the caller never sends them."""
        cap = self.fax_cap()
        call = self.call
        call["fax_role"] = "rx"
        self.tx_task = self.spawn(self._tx_loop())
        await asyncio.sleep(float(cap["answer_delay_ms"]) / 1000.0)
        if self.call is not call or call["state"] != "connected":
            return
        self.fax_send("I", self.f2["LID"] or "")
        self.fax_send("A", self.f2["DIS"])
        self.state = "command"
        log(f"{self.tag} fax: answering (+FCON)")
        self.event("connect", fax=True)
        self.write(b"\r\n+FCON\r\n")
        tsi, dcs = "", None
        end = asyncio.get_running_loop().time() + float(cap["connect_timeout_s"]) + 4.0
        while dcs is None:
            r = await self.fax_wait("IS", max(0.0, end - asyncio.get_running_loop().time()), call)
            if r is None or r[0] == "D":
                break
            if r[0] == "I":
                tsi = r[1]
            else:
                dcs = r[1]
        if dcs is None:
            return await self.fax_hangup(20, "the caller never sent its parameters", call)
        call["fax_dcs"], call["fax_s_ready"] = dcs, True
        log(f'{self.tag} fax: caller id "{tsi}" DCS {dcs}')
        self.write(f'\r\n+FTSI: "{tsi}"\r\n+FDCS: {dcs}\r\n'.encode("latin-1", "replace"))
        await self.ok()

    async def fax_fdr(self):
        call = self.call
        if not call or call.get("fax_role") != "rx" or self.state != "command":
            return False
        cap = self.fax_cap()
        if call.get("fax_resp_due"):
            ppr = self.f2.pop("PTS", None) or "1"
            call["fax_resp_due"] = False
            self.fax_send("R", ppr)
            log(f"{self.tag} fax: page response {ppr} sent")
            if call.get("fax_last") and ppr in ("1", "3"):
                self.fax_send("D", "0")
                while self.tx and self.call is call:
                    await asyncio.sleep(TICK)
                await asyncio.sleep(0.2)
                await self.fax_hangup(0, "last page confirmed", call, final=True)
                return HANDLED
        if not call.get("fax_s_ready"):
            r = await self.fax_wait("S", float(cap["page_timeout_s"]), call)
            if r is None or r[0] == "D":
                await self.fax_hangup(20 if r is None else int(r[1] or 0) or 20, "no further page", call)
                return HANDLED
            call["fax_dcs"] = r[1]
        call["fax_s_ready"] = False
        self.fax_bytes = 0
        self.state = "fax_rx_data"
        self.fax_send("C")
        log(f"{self.tag} fax: receiving page, DCS {call['fax_dcs']}")
        self.write(f"\r\n+FDCS: {call['fax_dcs']}\r\n".encode())
        self.respond("CONNECT")
        self.write(bytes([fc.DC2]))
        return HANDLED

    # ---- fax identification and Class 2 -------------------------------------
    # Opt-in per modem: a "fax" object in its config. Class 2 here is the pre-2.0 TR-29.2 /
    # Rockwell dialect that DOS-era software (DosFax LITE) speaks. Sender side only for now:
    # ATD in class 2 -> +FCON, +FCSI, +FDIS, OK; +FDT -> +FDCS, CONNECT, XON, DLE-shielded
    # page data, DLE ETX, OK; +FET=n -> +FPTS, (+FHNG on the last page), OK. Receiving
    # (+FDR, ATA, RING) is not implemented: those answer ERROR.

    FAX_DEFAULTS = {"classes": "0,1,2", "mfr": "sipmodem", "model": "fax mock", "rev": "1.0",
                    "speeds": "24,48,72,96", "hdlc_speeds": "3", "silence": "(0-255)",
                    "dcc": "1,3,0,2,0,0,0,0", "connect_timeout_s": 8.0, "page_timeout_s": 15.0,
                    "dcn_timeout_s": 3.0, "train_ms": 500, "train_timeout_s": 12.0,
                    "answer_delay_ms": 500}
    F2_RANGES = {"DCC": "(0,1),(0-5),(0-2),(0-2),(0-3),(0-2),(0-1),(0-7)",
                 "DIS": "(0,1),(0-5),(0-2),(0-2),(0-3),(0-2),(0-1),(0-7)",
                 "LID": "(20),(20-7E)", "BOR": "(0,1)", "CR": "(0,1)", "CQ": "(0-2)",
                 "PHCTO": "(0-255)", "BUG": "(0,1)", "CT": "(0-FF)", "NR": "(0-3)", "AA": "(0,1)"}

    def fax_cap(self):
        fax = self.cfg.get("fax")
        return dict(self.FAX_DEFAULTS, **(fax if isinstance(fax, dict) else {}))

    def f2_defaults(self):
        c = self.fax_cap() if self.cfg.get("fax") is not None else self.FAX_DEFAULTS
        return {"DCC": c["dcc"], "DIS": c["dcc"], "LID": "", "BOR": "0", "CR": "0", "CQ": "0",
                "PHCTO": "30", "BUG": "0", "CT": "1E", "NR": "0", "AA": "0"}

    def fax_send(self, ftype, payload=""):
        if ftype != "P":
            self._flush_pbuf()                  # page data queued so far goes out BEFORE this frame
        if ftype == "D" and self.call:
            self.call["fax_ending"] = True      # we are ending the session: the BYE that follows is normal
        self.tx.extend(fc.pack(ftype, payload))

    async def fax_wait(self, want, timeout, call=None):
        """Next frame whose type is in `want` from the far fax machine, or a disconnect
        ("D", code); None on timeout or when THIS call went away. Bound to the call (and its
        queue) it was started for, so a late task can never read the next call's frames."""
        loop = asyncio.get_running_loop()
        call = call or self.call
        q = self.fax_q
        end = loop.time() + timeout
        while True:
            if q is None or call is None or self.call is not call:
                return None
            left = end - loop.time()
            if left <= 0:
                return None
            try:
                ftype, payload = await asyncio.wait_for(q.get(), left)
            except asyncio.TimeoutError:
                return None
            if ftype == "X":            # the call went away while we waited (cleanup_call)
                return None
            if ftype in want or ftype == "D":
                return ftype, payload.decode("latin-1", "replace")
            log(f"{self.tag} fax: ignoring unexpected frame {ftype!r}")

    async def fax_begin(self):
        """The SIP call is up and the modem is in class 2: +FCON, then the answerer's id and
        capabilities (or +FHNG when they never come), then OK."""
        cap = self.fax_cap()
        call = self.call
        self.tx_task = self.spawn(self._tx_loop())
        delay = float(self.cfg.get("connect_delay_ms", 0)) / 1000.0
        if delay:
            await asyncio.sleep(delay)
        if not self.call or self.call["state"] != "connected":
            return self.respond("NO CARRIER")
        self.state = "command"
        log(f"{self.tag} fax: connected (+FCON)")
        self.event("connect", fax=True)
        self.write(b"\r\n+FCON\r\n")
        csi, dis = "", None
        end = asyncio.get_running_loop().time() + float(cap["connect_timeout_s"])
        while dis is None:
            r = await self.fax_wait("IA", max(0.0, end - asyncio.get_running_loop().time()), call)
            if r is None or r[0] == "D":
                break
            if r[0] == "I":
                csi = r[1]
            else:
                dis = r[1]
        if dis is None:
            return await self.fax_hangup(20, "no answer tone / DIS from the far end", call)
        self.call["fax_dis"], self.call["fax_csi"] = dis, csi
        log(f'{self.tag} fax: remote id "{csi}" DIS {dis}')
        self.write(f'\r\n+FCSI: "{csi}"\r\n+FDIS: {dis}\r\n'.encode("latin-1", "replace"))
        await self.ok()

    async def fax_hangup(self, code, why, call=None, final=False):
        """End a fax session with +FHNG. Bound to the call it was started for: if that call is already
        gone (the far end hung up first) this is silent for an error, because the DTE was already told,
        and for a normal end (final=True) it only reports to the DTE and never touches whatever call
        is current now."""
        call = call or self.call
        if call is None:
            return
        gone = self.call is not call
        if gone and not final:
            return
        log(f"{self.tag} fax: session ended (+FHNG: {code}) {why}")
        if final:
            log(f"{self.tag} fax: session complete pages={call.get('fax_pages', 0)}")
        call["fax_ending"] = True
        self.write(f"\r\n+FHNG: {code}\r\n".encode())
        if not gone:
            await self.hangup_call()
        await self.ok()

    def _flush_pbuf(self):
        """Send the batched page data as P frames (<= 512 bytes each)."""
        if self.pbuf_timer:
            self.pbuf_timer.cancel()
            self.pbuf_timer = None
        data, self.pbuf = bytes(self.pbuf), bytearray()
        for i in range(0, len(data), 512):
            self.tx.extend(fc.pack("P", data[i:i + 512]))

    def feed_fax_data(self, data):
        # The guest trickles page data in 1-2 byte serial reads; a frame each costs a 5 byte header
        # per byte (3x on the wire, and the baud pacing turned that into a late page). Batch them: one
        # frame per TICK at most, flushed at once ahead of any other frame.
        call = self.call
        if call and call.get("g711"):
            for ev in self.fax_dle.feed(data):
                if ev[0] == "data":
                    self.fax_bytes += len(ev[1])
                    call["span_cur"].extend(ev[1])
                else:
                    call["span_pages"].append(bytes(call["span_cur"]))
                    call["span_cur"] = bytearray()
                    self.state = "command"
                    log(f"{self.tag} fax: page {len(call['span_pages'])} data complete, "
                        f"{self.fax_bytes} bytes (DLE ETX)")
                    self.spawn(self.ok())
            return
        for ev in self.fax_dle.feed(data):
            if ev[0] == "data":
                self.fax_bytes += len(ev[1])
                self.pbuf.extend(ev[1])
                if len(self.pbuf) >= 512:
                    self._flush_pbuf()
                elif self.pbuf_timer is None:
                    self.pbuf_timer = asyncio.get_running_loop().call_later(TICK, self._flush_pbuf)
            else:
                self._flush_pbuf()
                self.state = "command"
                log(f"{self.tag} fax: page data complete, {self.fax_bytes} bytes (DLE ETX)")
                self.spawn(self.ok())

    async def fax_command(self, s):
        m = re.match(r"\+F([A-Z]+)(=\?|\?|=(.*))?$", s)
        fax = self.cfg.get("fax")
        if not m or fax is None:
            return NOT_MINE
        cap = self.fax_cap()
        name, form, val = m.group(1), m.group(2) or "", m.group(3)
        reply = lambda t: (self.write(("\r\n" + t).encode("latin-1")), True)[1]
        if name == "CLASS":
            if form == "=?":
                return reply(cap["classes"])
            if form == "?":
                return reply(self.fclass)
            if val is not None and val.strip() in cap["classes"].split(","):
                self.fclass = val.strip()
                return True
            return False
        if name in ("MFR", "MDL", "REV") and form in ("?", "=?"):
            return reply(cap[{"MFR": "mfr", "MDL": "model", "REV": "rev"}[name]])
        if self.fclass == "1" and form == "=?":
            if name in ("TM", "RM"):
                return reply(cap["speeds"])
            if name in ("TH", "RH"):
                return reply(cap["hdlc_speeds"])
            if name in ("TS", "RS"):
                return reply(cap["silence"])
            return False
        if self.fclass in ("0", "2"):
            if name in self.f2:
                if form == "=?":
                    return reply(self.F2_RANGES.get(name, "(0-255)"))
                if form == "?":
                    cur = self.f2[name]
                    return reply(f'"{cur}"' if name == "LID" else cur)
                if val is not None:
                    v = val.strip()
                    if name in ("DCC", "DIS"):
                        # DosFax sends "+FDCC=1": the fields it leaves out keep their defaults
                        have = [x.strip() for x in v.split(",") if x.strip() != ""]
                        base = self.fax_cap()["dcc"].split(",")
                        v = ",".join(have + base[len(have):])
                        if name == "DCC":
                            self.f2["DIS"] = v
                    self.f2[name] = v.strip('"') if name == "LID" else v
                    return True
                return False
        if self.fclass == "2":
            if name == "DT":
                return await self.fax_dt()
            if name == "ET" and val is not None:
                return await self.fax_et(val.strip())
            if name == "K":
                if self.call and self.call.get("fax"):
                    self.fax_send("D", "0")
                    await asyncio.sleep(0.2)
                    await self.hangup_call()
                    return True
                return False
            if name == "PTS":
                if val is not None:
                    self.f2["PTS"] = val.strip()
                return True
            if name == "DR":
                return await self.fax_fdr()
        log(f"{self.tag} fax: unsupported command +F{name}{form or ''}")
        return False

    async def fax_dt(self):
        call = self.call
        if not call or not call.get("fax") or not call.get("fax_dis") or self.state != "command":
            return False
        cap = self.fax_cap()
        dcs = fc.caps_text(fc.intersect(self.f2["DIS"], call["fax_dis"]))
        if call.get("g711"):
            return await self.span_fax_dt(call, dcs)
        self.fax_send("I", self.f2["LID"] or "")
        self.fax_send("S", dcs)
        r = await self.fax_wait("C", float(cap["train_timeout_s"]), call)
        if r is None or r[0] == "D":
            await self.fax_hangup(20 if r is None else int(r[1] or 0) or 20, "far end never confirmed the page", call)
            return HANDLED
        await asyncio.sleep(float(cap["train_ms"]) / 1000.0)
        if not self.call:
            return False
        self.fax_bytes = 0
        self.fax_dle = fc.DleDecoder()
        log(f"{self.tag} fax: DCS {dcs}, sending page")
        self.write(f"\r\n+FDCS: {dcs}\r\n".encode())
        self.respond("CONNECT")
        self.write(bytes([fc.XON]))
        self.state = "fax_data"
        return HANDLED

    async def fax_et(self, n):
        call = self.call
        if n not in ("0", "1", "2") or not call or not call.get("fax"):
            return False
        cap = self.fax_cap()
        if call.get("g711"):
            return await self.span_fax_et(call, n)
        self.fax_send("N", n)
        r = await self.fax_wait("R", float(cap["page_timeout_s"]), call)
        if r is None or r[0] == "D":
            await self.fax_hangup(20 if r is None else int(r[1] or 0) or 20, "no page response", call)
            return HANDLED
        ppr = r[1]
        log(f"{self.tag} fax: page response {ppr}")
        if ppr in ("1", "3"):
            call["fax_pages"] = call.get("fax_pages", 0) + 1
            log(f"{self.tag} fax: page {call['fax_pages']} sent")
        self.write(f"\r\n+FPTS: {ppr}\r\n".encode())
        if n == "2" and ppr in ("1", "3"):
            d = await self.fax_wait("", float(cap["dcn_timeout_s"]), call)
            await self.fax_hangup(int(d[1] or 0) if d and d[1].isdigit() else 0, "last page confirmed", call,
                                  final=True)
            return HANDLED
        return True

    async def cmd_plus(self, s):
        r = await self.fax_command(s)
        if r is not NOT_MINE:
            return r
        m = re.match(r"\+VTS=(.*)$", s)
        if not m:
            return False
        digits = [d for d in m.group(1) if d in DTMF_DIGITS]
        if not self.call or self.call["state"] != "connected" or not digits:
            return False
        for d in digits:
            await self.send_dtmf(d)
            await asyncio.sleep(0.06)
        return True

    async def cmd_online(self):
        if not self.call or self.call["state"] != "connected":
            return self.respond("ERROR")
        # Same pause as a dial: the guest's `copy comN` is not reading yet, and CONNECT
        # plus any held line data arriving first overruns the UART ("General failure
        # reading device COM1").
        delay = max(float(self.cfg.get("connect_delay_ms", 0)),
                    float(self.cfg.get("result_delay_ms", 0))) / 1000.0
        if delay:
            await asyncio.sleep(delay)
        if not self.call or self.call["state"] != "connected":
            return self.respond("NO CARRIER")
        self.respond("CONNECT", f" {self.baud}")
        self.go_online()

    def go_online(self):
        self.state = "online"
        self.last_byte_t = 0.0
        self.plus_n, self.held = 0, b""
        if self.rx_pending:
            self.write(bytes(self.rx_pending))
            self.rx_pending.clear()

    # ---- online mode + escape ---------------------------------------------

    def feed_online(self, data):
        guard = self.guard()
        for b in data:
            now = time.monotonic()
            if b == 0x2B:
                if self.plus_n == 0:
                    if now - self.last_byte_t >= guard:
                        self.plus_n, self.held = 1, b"+"
                    else:
                        self.tx.extend(b"+")
                elif self.plus_n < 3 and now - self.last_byte_t < guard:
                    self.plus_n += 1
                    self.held += b"+"
                else:
                    self._escape_abort()
                    self.tx.extend(b"+")
                if self.plus_n == 3 and self.esc_timer is None:
                    self.esc_timer = asyncio.get_running_loop().call_later(guard, self._escape)
            else:
                self._escape_abort()
                self.tx.append(b)
            self.last_byte_t = now

    def _escape_abort(self):
        if self.esc_timer:
            self.esc_timer.cancel()
            self.esc_timer = None
        if self.held:
            self.tx.extend(self.held)
        self.held, self.plus_n = b"", 0

    def _escape(self):
        self.esc_timer = None
        self.held, self.plus_n = b"", 0
        if self.state == "online":
            self.state = "online_cmd"
            log(f"{self.tag} escape (+++) -> command mode")
            self.event("escape")
            self.respond("OK")

    # ---- dialling ----------------------------------------------------------

    async def cmd_dial(self, dial):
        number = re.sub(r"[^0-9*#A-D]", "", dial.replace("T", "").replace("P", ""))
        if self.call:
            return self.respond("ERROR")
        if not number:
            return self.respond("ERROR")
        if not await self.seize_line(dial.upper()):
            return
        self.audio.dial_digits("".join(c for c in dial.upper() if c in ma.DTMF_KEYS or c == ","))
        pauses = dial.count(",")
        if pauses:
            await asyncio.sleep(pauses * float(self.cfg.get("comma_pause_s", 2.0)))
        loop = asyncio.get_running_loop()
        try:
            rtp = RtpUa(self)
            await loop.create_datagram_endpoint(lambda: rtp, local_addr=(self.app.local_host or "0.0.0.0", 0))
        except OSError as e:
            log(f"{self.tag} cannot open RTP: {e}")
            return self.respond("NO CARRIER")
        self.rtp = rtp
        srv = self.app.server
        adv = self.app.advertise
        sip_port = self.sip.get_extra_info("sockname")[1]
        rtp_port = rtp.transport.get_extra_info("sockname")[1]
        cid = sc.new_token(14)
        ltag = sc.new_token(6)
        self.call = {"cid": cid, "ltag": ltag, "number": number, "state": "inviting",
                     "uri": f"sip:{number}@{srv[0]}:{srv[1]}",
                     "from": f"<sip:{self.user}@{adv}>;tag={ltag}",
                     "to": f"<sip:{number}@{srv[0]}:{srv[1]}>",
                     "via": f"{adv}:{sip_port}", "contact": f"<sip:{self.user}@{adv}:{sip_port}>",
                     "remote_rtp": None, "data_pt": 96, "dtmf_pt": 101,
                     "bye_evt": asyncio.Event(), "ringing": False,
                     "fax": self.fax_active()}
        if self.call["fax"]:
            self.fax_q, self.fax_df = asyncio.Queue(), fc.Deframer()
        self.dial_fut = loop.create_future()
        self.state = "dialing"
        if self.ne0k:
            sdp = sc.build_sdp_g711(adv, rtp_port, (0,), 101)
        elif self.span_media():
            sdp = sc.build_sdp_g711(adv, rtp_port, (0, 8), 101)
        else:
            sdp = sc.build_sdp(adv, rtp_port, 96, 101)
        invite = sc.build_request("INVITE", self.call["uri"], self.call["via"],
                                  self.call["from"], self.call["to"], cid, 1,
                                  contact=self.call["contact"], body=sdp)
        log(f"{self.tag} dialing {number}")
        log(f"{self.tag} dial: {number}")
        self.event("dial", number=number)
        self.spawn(self._retransmit(invite))
        try:
            word = await asyncio.wait_for(asyncio.shield(self.dial_fut), self.sregs[7])
        except asyncio.TimeoutError:
            log(f"{self.tag} dial timeout after {self.sregs[7]}s")
            await self.hangup_call()
            word = "NO CARRIER"
        except asyncio.CancelledError:
            raise
        if word == "CONNECT" and self.call and self.call.get("ne0k"):
            if not await self.ne0k_up(self.call, caller=True):
                return self.respond("NO CARRIER")
        if word == "CONNECT" and self.call and self.call.get("g711") and not self.call.get("fax"):
            return await self.span_begin("caller")
        if word == "CONNECT" and self.call and self.call.get("g711"):
            return await self.span_fax_begin()
        if word == "CONNECT" and self.call and self.call.get("fax"):
            return await self.fax_begin()
        if word == "CONNECT":
            delay = float(self.cfg.get("connect_delay_ms", 0)) / 1000.0
            if delay:
                await asyncio.sleep(delay)
            if not self.call or self.call["state"] != "connected":
                return self.respond("NO CARRIER")
            self.tx_task = self.spawn(self._tx_loop())
            log(f"{self.tag} CONNECT {self.baud}")
            self.event("connect")
            self.respond("CONNECT", f" {self.baud}")
            self.go_online()        # after CONNECT, so a line banner never precedes it
        else:
            delay = float(self.cfg.get("result_delay_ms", 0)) / 1000.0
            if delay:
                await asyncio.sleep(delay)
            log(f"{self.tag} {word}")
            self.event("result", word=word)
            self.respond(word)

    def fax_active(self):
        return self.cfg.get("fax") is not None and self.fclass == "2"

    async def _retransmit(self, invite):
        call = self.call
        delay = 0.5
        for _ in range(6):
            self.sip.sendto(invite, self.app.server)
            await asyncio.sleep(delay)
            if not self.call or self.call is not call or call["ringing"] or call["state"] != "inviting":
                return
            delay = min(delay * 2, 4.0)
        if self.dial_fut and not self.dial_fut.done():
            self.dial_fut.set_result("NO CARRIER")

    def _resolve(self, word):
        if self.dial_fut and not self.dial_fut.done():
            self.dial_fut.set_result(word)

    # ---- SIP ---------------------------------------------------------------

    def on_sip(self, data, addr):
        m = sc.parse(data)
        if m is None:
            return
        call = self.call
        if not m.is_request:
            if self.app.register and m.call_id == self.reg_cid:
                return self._on_register_response(m)
            if not call or m.call_id != call["cid"]:
                return
            method = m.cseq()[1]
            if method == "INVITE":
                self._on_invite_response(m)
            elif method == "BYE" and m.status >= 200:
                call["bye_evt"].set()
            return
        if m.method == "BYE":
            if call and m.call_id == call["cid"]:
                self.app.sip_reply(self, sc.build_response(m, 200, "OK"), addr)
                if call["state"] in ("connected", "inviting"):
                    self.remote_hangup()
            else:
                self.app.sip_reply(self, sc.build_response(m, 481, "Call/Transaction Does Not Exist"), addr)
        elif m.method == "OPTIONS":
            self.app.sip_reply(self, sc.build_response(m, 200, "OK", to_tag=sc.new_token(6)), addr)
        elif m.method == "INVITE":
            self.on_incoming_invite(m, addr)
        elif m.method == "ACK":
            if call and m.call_id == call["cid"]:
                call["acked"] = True
        elif m.method == "CANCEL":
            if call and m.call_id == call["cid"] and call.get("incoming") and call["state"] == "ringing":
                self.app.sip_reply(self, sc.build_response(m, 200, "OK", to_tag=call["ltag"]), addr)
                self.app.sip_reply(self, sc.build_response(call["invite"], 487, "Request Terminated",
                                                           to_tag=call["ltag"]), addr)
                log(f"{self.tag} caller gave up before answer")
                self.event("no_answer")
                self.cleanup_call()
            else:
                self.app.sip_reply(self, sc.build_response(m, 481, "Call/Transaction Does Not Exist"), addr)

    def _on_invite_response(self, m):
        call = self.call
        if m.status < 200:
            if not call["ringing"]:
                call["ringing"] = True
                log(f"{self.tag} ringing ({m.status})")
                self.audio.ringback()
            return
        if m.status == 200:
            if call["state"] == "inviting":
                info = sc.parse_sdp(m.body)
                if info is None:
                    log(f"{self.tag} 200 OK without usable SDP")
                    self.send_ack(m)
                    self.spawn(self.hangup_call())
                    self._resolve("NO CARRIER")
                    return
                call["remote_rtp"] = (info["addr"], info["port"])
                call["data_pt"], call["dtmf_pt"] = info["data_pt"], info["dtmf_pt"]
                call["g711"], call["law"] = bool(info.get("g711")) and not self.ne0k, info.get("law")
                call["ne0k"] = self.ne0k and bool(info.get("g711"))
                call["to"] = m.get("to", call["to"])
                call["state"] = "connected"
                self.send_ack(m)
                self.audio.answered(fax=bool(call.get("fax")))
                self._resolve("CONNECT")
            else:
                self.send_ack(m)            # 200 retransmit
            return
        # final failure
        self.send_ack(m, non2xx=True)
        if call["state"] in ("inviting", "cancelling"):
            call["state"] = "failed"
            word = FAIL_WORDS.get(m.status, "NO CARRIER")
            log(f"{self.tag} call failed {m.status} {m.reason}")
            self._resolve(word)
            call["bye_evt"].set()
            self.cleanup_call()
            if m.status in (486, 600):
                self.audio.busy()

    def send_ack(self, m, non2xx=False):
        call = self.call
        uri = call["uri"] if non2xx else (sc.uri_of(m.get("contact", "")) or call["uri"])
        self.sip.sendto(sc.build_request("ACK", uri, call["via"], call["from"],
                                         m.get("to", call["to"]), call["cid"], 1),
                        self.app.server)

    async def hangup_call(self):
        call = self.call
        if not call:
            self.state = "command"
            return
        st = call["state"]
        if call.get("incoming") and st == "ringing":
            self.app.sip_reply(self, sc.build_response(call["invite"], 603, "Decline", to_tag=call["ltag"]),
                               call["peer"])
            log(f"{self.tag} incoming call declined")
            self.cleanup_call()
            return
        if st == "inviting":
            call["state"] = "cancelling"
            cancel = sc.build_request("CANCEL", call["uri"], call["via"], call["from"],
                                      call["to"], call["cid"], 1)
            for _ in range(3):
                self.sip.sendto(cancel, self.app.server)
                try:
                    await asyncio.wait_for(call["bye_evt"].wait(), 0.5)
                    break
                except asyncio.TimeoutError:
                    continue
            self._resolve("NO CARRIER")
        elif st == "connected":
            call["state"] = "ending"
            bye = sc.build_request("BYE", call["uri"], call["via"], call["from"],
                                   call["to"], call["cid"], 2)
            for _ in range(4):
                self.sip.sendto(bye, self.app.server)
                try:
                    await asyncio.wait_for(call["bye_evt"].wait(), 0.5)
                    break
                except asyncio.TimeoutError:
                    continue
            log(f"{self.tag} hung up ({call['number']}) rx={self.stats['rx']}B tx={self.stats['tx']}B")
            self.event("hangup", **self.stats)
        self.cleanup_call()

    def remote_hangup(self):
        call = self.call
        if call["state"] == "inviting":
            self._resolve("NO CARRIER")
        log(f"{self.tag} NO CARRIER (bye from server) rx={self.stats['rx']}B tx={self.stats['tx']}B")
        self.event("no_carrier", **self.stats)
        was_online = self.state in ("online", "online_cmd")
        fax_lost = call.get("fax") and call["state"] == "connected" and not call.get("fax_ending")
        self.cleanup_call()
        if fax_lost:
            log(f"{self.tag} fax: session lost (+FHNG: 10)")
            self.write(b"\r\n+FHNG: 10\r\n")
            self.respond("OK")
        if was_online:
            delay = float(self.cfg.get("result_delay_ms", 0)) / 1000.0
            if delay:
                self.spawn(self._late("NO CARRIER", delay))
            else:
                self.respond("NO CARRIER")

    async def _late(self, word, delay):
        await asyncio.sleep(delay)
        self.respond(word)

    def cleanup_call(self):
        if self.pipe:
            pipe, self.pipe = self.pipe, None
            asyncio.get_running_loop().run_in_executor(None, pipe.stop)
        span = self.call.get("span") if self.call else None
        if span:
            self.call["span"] = None
            asyncio.ensure_future(span.stop())
        self.baud = self.base_baud
        if self.ring_task and self.ring_task is not asyncio.current_task():
            self.ring_task.cancel()
        self.ring_task = None
        if self.fax_q is not None:
            self.fax_q.put_nowait(("X", b""))       # wake anything waiting on the far end
        self.call = None
        self.fax_q = None
        if self.pbuf_timer:
            self.pbuf_timer.cancel()
            self.pbuf_timer = None
        self.pbuf.clear()
        self.state = "command"
        self.audio.call_ended()
        self.putdown()
        self._escape_abort()
        self.tx.clear()
        self.rx_pending.clear()
        if self.tx_task:
            self.tx_task.cancel()
            self.tx_task = None
        if self.rtp and self.rtp.transport:
            self.rtp.transport.close()
        self.rtp = None

    # ---- RTP ---------------------------------------------------------------

    def on_rtp(self, data, addr):
        call = self.call
        if not call or call["state"] != "connected" or len(data) < 12 or data[0] >> 6 != 2:
            return
        cc = data[0] & 0x0F
        pt = data[1] & 0x7F
        _, ts, ssrc = struct.unpack("!HII", data[2:12])
        payload = data[12 + 4 * cc:]
        call["remote_rtp"] = addr
        if call.get("ne0k"):
            if self.pipe:
                self.pipe.feed(data)
            return
        if call.get("g711") and pt == call["data_pt"]:
            if call.get("span"):
                call["span"].audio(payload)
            return
        if pt == call["data_pt"] and payload:
            self.audio.line_data(payload, self.baud, tx=False)
        if pt == call["data_pt"] and payload and call.get("fax"):
            self.stats["rx"] += len(payload)
            for frame in self.fax_df.feed(payload):
                self.fax_on_frame(frame)
        elif pt == call["data_pt"] and payload:
            self.stats["rx"] += len(payload)
            log(f"{self.tag} data from line {len(payload)} bytes: {bytes(payload[:60])!r}")
            if self.state == "online":
                self.write(payload)
            else:
                self.rx_pending.extend(payload[:65536])
        elif call["dtmf_pt"] is not None and pt == call["dtmf_pt"] and len(payload) >= 4:
            if (ssrc, ts) in self._dtmf_seen:
                return
            self._dtmf_seen.add((ssrc, ts))
            if payload[0] < len(DTMF_DIGITS):
                d = DTMF_DIGITS[payload[0]]
                log(f"{self.tag} dtmf {d} (from line)")
                self.event("dtmf", digit=d, direction="from_line")
                self.audio.dtmf(d, int(struct.unpack("!H", payload[2:4])[0] / CLOCK * 1000) or 100)
                if self.cfg.get("dtmf_report", "log") == "unsolicited" and self.state != "online":
                    self.write(f"\r\nDTMF={d}\r\n".encode())

    # ---- spandsp media: a real modem over G.711 audio ----------------------------------

    async def span_fax_begin(self):
        """Class 2 fax over G.711. The T.30 engine (spandsp) cannot be driven page by page from a
        Class 2 DTE, so: the call is up and the helper waits silently (--hold); the DTE gets +FCON and
        the configured remote caps and delivers its pages as usual; at the last +FET the pages become
        a TIFF and the helper runs the real T.30 session (CED/DIS/V.21/V.29/V.17...) with the far
        fax, and only then does the DTE get +FPTS / +FHNG. Cost: per-page confirmations are not
        real (+FPTS: 1 is returned at once for every page but the last)."""
        call = self.call
        cap = self.fax_cap()
        c = self.cfg.get("spandsp") or {}
        sd = c.get("save_dir") or os.path.join(HERE, "output")
        os.makedirs(sd, exist_ok=True)
        call["span_tiff"] = os.path.join(sd, f"{self.name}_tx_{int(time.time())}.tif")
        call["span_pages"], call["span_cur"] = [], bytearray()
        call["span_ts"], call["span_on"] = random.getrandbits(32), False
        call["span_fax_done"] = asyncio.get_running_loop().create_future()
        args = ["--mode", "faxtx", "--hold", "--law", call["law"], "--ident", self.f2["LID"] or "SIPMODEM",
                "--ecm", str(int(c.get("ecm", 0)))]
        if c.get("fax_modems"):
            args += ["--fax-modems", str(c["fax_modems"])]
        if c.get("debug"):
            args.append("--debug")
        span = sph.PacedModem(args,
                              on_audio=lambda p, k=call: self._span_audio(k, p),
                              on_status=lambda t, k=call: self._span_fax_status(k, t),
                              on_done=lambda r, k=call: self._span_fax_done(k, r),
                              on_stderr=lambda t: log(f"{self.tag} spandsp: {t}"))
        try:
            await span.start()
        except OSError as e:
            log(f"{self.tag} cannot start spanmodem: {e}")
            await self.hangup_call()
            return self.respond("NO CARRIER")
        call["span"] = span
        delay = float(self.cfg.get("connect_delay_ms", 0)) / 1000.0
        if delay:
            await asyncio.sleep(delay)
        if self.call is not call or call["state"] != "connected":
            return self.respond("NO CARRIER")
        self.state = "command"
        dis = cap["dcc"]
        call["fax_dis"], call["fax_csi"] = dis, str(cap.get("remote_id", ""))
        log(f"{self.tag} fax: connected (+FCON), spandsp T.30 sender held until the last page")
        self.event("connect", fax=True)
        self.write(b"\r\n+FCON\r\n")
        self.write(f'\r\n+FCSI: "{call["fax_csi"]}"\r\n+FDIS: {dis}\r\n'.encode("latin-1", "replace"))
        await self.ok()

    async def span_fax_dt(self, call, dcs):
        cap = self.fax_cap()
        await asyncio.sleep(float(cap["train_ms"]) / 1000.0)
        if self.call is not call:
            return False
        call["span_dcs"] = dcs
        self.fax_bytes = 0
        self.fax_dle = fc.DleDecoder()
        call["span_cur"] = bytearray()
        log(f"{self.tag} fax: DCS {dcs}, taking page {len(call['span_pages']) + 1} from the DTE")
        self.write(f"\r\n+FDCS: {dcs}\r\n".encode())
        self.respond("CONNECT")
        self.write(bytes([fc.XON]))
        self.state = "fax_data"
        return HANDLED

    async def span_fax_et(self, call, n):
        if not call.get("span_pages"):
            return False
        if n == "0":                          # more pages follow: accept this one
            self.write(b"\r\n+FPTS: 1\r\n")
            return True
        c = self.cfg.get("spandsp") or {}
        info = fc.write_tiff(call["span_tiff"], call["span_pages"], call.get("span_dcs"))
        log(f"{self.tag} fax: sending {info['pages']} page(s) via spandsp T.30 ({call['span_tiff']})")
        call["span"].go(call["span_tiff"])
        try:
            res = await asyncio.wait_for(asyncio.shield(call["span_fax_done"]),
                                         float(c.get("timeout_s", 240)))
        except asyncio.TimeoutError:
            res = "timeout"
        if self.call is not call:
            return HANDLED
        if res == "ok":
            self.write(b"\r\n+FPTS: 1\r\n")
            await self.fax_hangup(0, "last page confirmed (T.30 complete)", call, final=True)
        else:
            self.write(b"\r\n+FPTS: 2\r\n")
            await self.fax_hangup(20, f"T.30 failed: {res}", call)
        return HANDLED

    def _span_fax_status(self, call, text):
        log(f"{self.tag} spandsp: {text}")
        self.event("spandsp", text=text)

    def _span_fax_done(self, call, result):
        f = call.get("span_fax_done")
        if f and not f.done():
            f.set_result(result)

    def span_media(self):
        """True when this modem offers real audio (cfg media: "g711") and spanmodem is built."""
        return self.cfg.get("media") == "g711" and sph.available()

    async def span_begin(self, role):
        """The call is up and carries G.711: run V.8 + V.22bis/V.21 through spandsp, print CONNECT
        only when the modems have trained (that is what a real modem does), else NO CARRIER."""
        call = self.call
        c = self.cfg.get("spandsp") or {}
        args = ["--mode", "data", "--role", role, "--law", call["law"],
                "--mods", ",".join(c.get("mods", ["v22", "v21"])), "--bps", str(c.get("bps", 2400))]
        if c.get("v8", True) is False:
            args.append("--no-v8")
        if "guard_ms" in c:
            args += ["--guard-ms", str(c["guard_ms"])]
        if c.get("debug"):
            args.append("--debug")
        up = asyncio.get_running_loop().create_future()
        call["span_up"], call["span_ts"], call["span_on"] = up, random.getrandbits(32), False
        span = sph.PacedModem(args,
                              on_audio=lambda p, k=call: self._span_audio(k, p),
                              on_data=lambda d, k=call: self._span_data(k, d),
                              on_status=lambda t, k=call: self._span_status(k, t),
                              on_done=lambda r, k=call: self._span_done(k, r),
                              on_stderr=lambda t: log(f"{self.tag} spandsp: {t}"))
        try:
            await span.start()
        except OSError as e:
            log(f"{self.tag} cannot start spanmodem: {e}")
            await self.hangup_call()
            return self.respond("NO CARRIER")
        call["span"] = span
        log(f"{self.tag} spandsp {role}: {' '.join(args)}")
        try:
            bps = await asyncio.wait_for(asyncio.shield(up), float(self.sregs[7]))
        except asyncio.TimeoutError:
            log(f"{self.tag} spandsp: modem never connected (S7={self.sregs[7]}s)")
            await self.hangup_call()
            return self.respond("NO CARRIER")
        except asyncio.CancelledError:
            raise
        if self.call is not call or call["state"] != "connected":
            return self.respond("NO CARRIER")
        call["span_on"] = True
        self.baud = int(bps)
        delay = float(self.cfg.get("connect_delay_ms", 0)) / 1000.0
        if delay:
            await asyncio.sleep(delay)
        if self.call is not call or call["state"] != "connected":
            return self.respond("NO CARRIER")
        self.tx_task = self.spawn(self._tx_loop())
        log(f"{self.tag} CONNECT {self.baud}{' (answered)' if role == 'answerer' else ''}")
        self.event("connect")
        self.respond("CONNECT", f" {self.baud}")
        self.go_online()

    def _span_audio(self, call, payload):
        if self.call is not call or call["state"] != "connected":
            return
        self.seq = (self.seq + 1) & 0xFFFF
        pkt = rtp_packet(call["data_pt"], self.seq, call["span_ts"], self.ssrc, payload)
        call["span_ts"] = (call["span_ts"] + len(payload)) & 0xFFFFFFFF
        self._send_rtp(pkt)

    def _span_data(self, call, data):
        if self.call is not call or not data:
            return
        self.audio.line_data(data, self.baud, tx=False)
        self.stats["rx"] += len(data)
        log(f"{self.tag} data from line {len(data)} bytes: {bytes(data[:60])!r}")
        if self.state == "online":
            self.write(data)
        else:
            self.rx_pending.extend(data[:65536])

    def _span_status(self, call, text):
        log(f"{self.tag} spandsp: {text}")
        self.event("spandsp", text=text)
        up = call.get("span_up")
        m = re.match(r"connect (\S+) (\d+)", text)
        if m and up and not up.done():
            up.set_result(int(m.group(2)))
        elif text == "carrier down" and call.get("span_on") and self.call is call:
            call["span_on"] = False
            self.spawn(self._span_carrier_lost(call))

    async def _span_carrier_lost(self, call):
        was_online = self.state in ("online", "online_cmd")
        if self.call is call:
            await self.hangup_call()
        if was_online:
            self.respond("NO CARRIER")

    def _span_done(self, call, result):
        if result not in ("ok", "exited") or self.call is not call:
            log(f"{self.tag} spandsp: helper ended ({result})")

    def _ts(self):
        return int((time.monotonic() - self.t0) * CLOCK)

    def _send_rtp(self, pkt):
        call = self.call
        if call and call["remote_rtp"] and self.rtp and self.rtp.transport:
            self.rtp.transport.sendto(pkt, call["remote_rtp"])

    async def ne0k_up(self, call, caller):
        """Start the OFDM link on this call's RTP leg and wait for the BAT exchange (carrier)."""
        from .ne0kpipe import Ne0kPipe
        loop = asyncio.get_running_loop()
        o = self.cfg.get("ofdm") or {}
        pipe = Ne0kPipe(self.name, caller, log=lambda m: log(f"{self.tag} {m}"), **o)
        pipe.on_data = lambda d: loop.call_soon_threadsafe(self._line_rx, call, d)
        pipe.attach(lambda pkt: loop.call_soon_threadsafe(self._send_rtp, pkt), call["remote_rtp"])
        self.pipe = pipe
        await loop.run_in_executor(None, pipe.start)
        t0 = time.monotonic()
        up = await loop.run_in_executor(None, pipe.wait_up, float(self.cfg.get("carrier_timeout_s", 90)))
        if not up or self.call is not call:
            log(f"{self.tag} no carrier: the OFDM handshake did not finish")
            await self.hangup_call()
            return False
        log(f"{self.tag} carrier up (OFDM BAT exchanged in {time.monotonic() - t0:.1f}s)")
        self.event("carrier")
        return True

    def _line_rx(self, call, payload):
        if self.call is not call or not payload:
            return
        self.stats["rx"] += len(payload)
        self.audio.line_data(payload, self.baud, tx=False)
        if call.get("fax"):
            for frame in self.fax_df.feed(payload):
                self.fax_on_frame(frame)
        elif self.state == "online":
            self.write(payload)
        else:
            self.rx_pending.extend(payload[:65536])

    async def _ne0k_tx_loop(self):
        """Hand self.tx to the OFDM pipe in bursts.  Bytes stay in self.tx until the burst has aired,
        so callers that wait for `not self.tx` really wait for the line."""
        call, loop = self.call, asyncio.get_running_loop()
        while self.call is call and self.pipe:
            await asyncio.sleep(0.05)
            n = len(self.tx)
            if not n:
                continue
            while len(self.tx) != n and len(self.tx) < 4096:      # let the DTE finish its write
                n = len(self.tx)
                await asyncio.sleep(0.15)
            data = bytes(self.tx)
            ok = await loop.run_in_executor(None, self.pipe.send, data) if self.pipe else False
            del self.tx[:len(data)]
            self.stats["tx"] += len(data)
            if not ok:
                log(f"{self.tag} line: burst of {len(data)} bytes was not sent")

    async def _tx_loop(self):
        if self.call and self.call.get("ne0k"):
            return await self._ne0k_tx_loop()
        marker = True
        per_tick = max(1, int(round(self.baud / 10.0 * TICK)))
        while True:
            await asyncio.sleep(TICK)
            if not self.tx:
                marker = True
                continue
            chunk = bytes(self.tx[:per_tick])
            del self.tx[:per_tick]
            self.seq = (self.seq + 1) & 0xFFFF
            self.stats["tx"] += len(chunk)
            self.audio.line_data(chunk, self.baud, tx=True)
            call = self.call
            if not call:
                return
            if call.get("g711"):
                if call.get("span"):
                    call["span"].dte(chunk)         # the helper modulates it
                continue
            self._send_rtp(rtp_packet(call["data_pt"], self.seq, self._ts(),
                                      self.ssrc, chunk, marker))
            marker = False

    async def send_dtmf(self, digit):
        call = self.call
        if not call or call["dtmf_pt"] is None:
            return
        ev, ts, pt = DTMF_DIGITS.index(digit), self._ts(), call["dtmf_pt"]
        log(f"{self.tag} dtmf {digit} (to line)")
        self.event("dtmf", digit=digit, direction="to_line")
        self.audio.dtmf(digit, 100)
        for i in range(5):
            self.seq = (self.seq + 1) & 0xFFFF
            self._send_rtp(rtp_packet(pt, self.seq, ts, self.ssrc,
                                      bytes([ev, 10]) + struct.pack("!H", 160 * (i + 1)), i == 0))
            await asyncio.sleep(TICK)
        for _ in range(3):
            self.seq = (self.seq + 1) & 0xFFFF
            self._send_rtp(rtp_packet(pt, self.seq, ts, self.ssrc,
                                      bytes([ev, 0x80 | 10]) + struct.pack("!H", 800)))


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

class App:
    def __init__(self, cfg):
        self.cfg = cfg
        host, _, port = cfg["sip"]["server"].rpartition(":")
        self.server = (host or cfg["sip"]["server"], int(port) if host else 5060)
        self.local_host = cfg["sip"].get("local_host")
        self.register = bool(cfg["sip"].get("register"))     # REGISTER each modem's user, refreshed while running
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((self.server[0], 9))
            self.advertise = self.local_host or s.getsockname()[0]
        except OSError:
            self.advertise = self.local_host or "127.0.0.1"
        finally:
            s.close()
        self.listen = cfg.get("listen")      # default "listen" (speaker) config for every modem
        self.modems = []
        self.open_modems = 0
        self.stop = None

    def event(self, kind, **kw):
        path = self.cfg.get("events_file")
        if path:
            try:
                with open(path, "a") as f:
                    f.write(json.dumps({"t": round(time.time(), 3), "event": kind, **kw}) + "\n")
            except OSError:
                pass

    def sip_reply(self, modem, data, addr):
        modem.sip.sendto(data, addr)

    def serial_closed(self, modem):
        self.open_modems -= 1
        if self.open_modems <= 0:
            log("all serial ports closed, exiting")
            self.stop.set()


def load_config(path):
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(f"modemmock: cannot read config {path}: {e}")
    if "server" not in cfg.get("sip", {}):
        raise SystemExit("modemmock: config needs sip.server (host:port)")
    if not cfg.get("modems"):
        raise SystemExit("modemmock: config has no 'modems'")
    ports = [int(m["port"]) for m in cfg["modems"]]
    if len(set(ports)) != len(ports):
        raise SystemExit("modemmock: duplicate 'port' in modems")
    cfg["modems"].sort(key=lambda m: int(m["port"]))
    need = [m for m in cfg["modems"] if not m.get("pty")]
    if need and not cfg.get("vm_path"):
        raise SystemExit("modemmock: 'vm_path' is required unless every modem has an explicit 'pty'")
    return cfg


async def amain(cfg, ptys):
    app = App(cfg)
    app.stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for mcfg, path in zip(cfg["modems"], ptys):
        m = Modem(app, mcfg, path)
        await m.start()
        app.modems.append(m)
    app.open_modems = len(app.modems)

    def on_signal():
        if not app.stop.is_set():
            log("signal received, shutting down")
            app.stop.set()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, on_signal)

    if app.register:
        wait = float(cfg["sip"].get("register_timeout_s", 60))
        try:
            await asyncio.wait_for(asyncio.gather(*(m.reg_ok.wait() for m in app.modems)), wait)
        except asyncio.TimeoutError:
            log(f"register: no 200 OK from {app.server[0]}:{app.server[1]} within {wait:.0f}s")
            for m in app.modems:
                m.close_serial()
            raise SystemExit(1)
    log(f"READY modems={len(app.modems)}")
    if cfg.get("ready_file"):
        with open(cfg["ready_file"], "w") as f:
            f.write("ready\n")
    await app.stop.wait()
    await asyncio.gather(*(m.hangup_call() for m in app.modems if m.call), return_exceptions=True)
    for m in app.modems:
        m.unregister()
    for m in app.modems:
        m.close_serial()
        await m.audio.close()
    log("shutdown complete")


def main():
    if sys.platform.startswith("win"):
        print("modemmock: POSIX only (no PTY to attach to on Windows)", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=os.environ.get("HOSTSERVICE_CONFIG"))
    args = ap.parse_args()
    if not args.config:
        print("modemmock: no --config and no $HOSTSERVICE_CONFIG", file=sys.stderr)
        return 2
    return run(load_config(args.config))


def run(cfg):
    need = [m for m in cfg["modems"] if not m.get("pty")]
    found = []
    if need:
        try:
            found = ptyhelpers.discover_ptys(cfg["vm_path"], len(need), float(cfg.get("find_timeout", 60)))
        except RuntimeError as e:
            print(f"modemmock: {e}", file=sys.stderr)
            return 1
    it = iter(found)
    ptys = [m.get("pty") or next(it) for m in cfg["modems"]]
    asyncio.run(amain(cfg, ptys))
    return 0


if __name__ == "__main__":
    sys.exit(main())
