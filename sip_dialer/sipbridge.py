#!/usr/bin/env python3
"""sipbridge.py -- carry modem audio over a SIP call.  External to modemgui.

A SIP G.711 u-law RTP, 8 kHz call is wired to two PulseAudio streams. The GUI or
`modemgui serial` talk to a phone line exactly as they talk to any other audio device:

    modemgui TX  -> sink S_TX  -> [S_TX.monitor]  --parec-->  RTP  --> far end
    modemgui RX  <- [R.monitor] <-- sink R  <--pacat--  RTP  <-- far end

    sipbridge.py call   --to HOST:PORT --number 1234 --source S_TX.monitor --sink R
    sipbridge.py answer --listen 0.0.0.0:5060         --source S_TX.monitor --sink R

`call` places the call, `answer` waits for one.  Either side can be anything that speaks SIP + PCMU
`answer --register HOST:PORT` REGISTERs so a server that rings registered users can reach it.  

Out of scope: digest auth, TLS, SRTP, NAT traversal, codecs other than PCMU, DTMF.

The line is a telephone channel: ~300-3400 Hz at 8 kHz.  PulseAudio resamples, so the modem keeps
its 48 kHz rate, but it must stay inside the band . OFDM: freq_lo 400, freq_hi 3300; CSS: f_max 3300.

stdout carries stable lines for test_filecontains:
    READY sip=HOST:PORT rtp=PORT          bound, waiting / about to dial
    RINGING                               180 seen call or sent/answer
    CALL UP peer=HOST:PORT                media flowing
    CALL ENDED reason=...                 BYE either way, timeout, or signal
    STATS tx=N rx=N lost=N late=N jumps=N once, at the end
    RTP LOSS t=UNIX seq=N gap=N           each receive gap
    RTP LATE t=UNIX seq=N behind=N        a packet older than one already played: dropped
    RTP JUMP t=UNIX seq=N gap=N           a sequence jump too big to fill
    RTP TX STALL t=UNIX seq=N behind=MS   our own send fell MS behind its 20 ms schedule ; past the
                                          far end's playout delay that frame is silence in the room

`listen` is a receive-only `call` for people monitoring a test from their own desktop,
without logging into the test hosts.  It dials the same conference the test nodes use and plays what it
hears on the local speakers:

    pip install sounddevice

    sipbridge.py listen --to your-server-ip:15064 --number 3000          # mesh_grp_* conference

It never opens a capture device: it sends RTP-encoded digital silence (u-law 0xFF = sample 0) every 20 ms,
which keeps the NAT mapping open so the server's symmetric RTP reaches it, and which a conference bridge
adds to the other members' mix as exactly zero.  The SDP says a=recvonly.  It binds 0.0.0.0, advertises the
address of the interface that routes to --to, and stays in the call until Ctrl-C, so it keeps hearing every
later run that joins the same conference.
Playback (--player): `pulse` (pacat, Linux/PipeWire), `sounddevice` .
`ffplay`, or `wav:FILE` .  `auto` takes the first of these

Needs numpy and the pulseaudio-utils parec/pacat.  Exit code 0 when a call was established.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import select
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

import numpy as np

RATE, PTIME, SAMPLES = 8000, 0.02, 160
STALL_S = 0.06           # a send this far behind the 20 ms schedule is logged (RTP TX STALL)
PCMU = 0


def log(msg: str) -> None:
    print(msg, flush=True)


# ---- G.711 u-law -------------------------------------------------------------------------------
_EXP = np.array([max(i.bit_length() - 1, 0) for i in range(256)], dtype=np.int32)


def lin2ulaw(pcm: np.ndarray) -> bytes:
    s = pcm.astype(np.int32)
    sign = (s >> 8) & 0x80
    s = np.minimum(np.abs(s), 32635) + 0x84
    e = _EXP[(s >> 7) & 0xFF]
    m = (s >> (e + 3)) & 0x0F
    return (~(sign | (e << 4) | m) & 0xFF).astype(np.uint8).tobytes()


def _ulaw_table() -> np.ndarray:
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    e, m = (u >> 4) & 7, u & 0x0F
    v = (((m << 3) + 0x84) << e) - 0x84
    return np.where(u & 0x80, -v, v).astype(np.int16)


_ULAW = _ulaw_table()


def ulaw2lin(data: bytes) -> np.ndarray:
    return _ULAW[np.frombuffer(data, dtype=np.uint8)]


# ---- SIP (just enough of RFC 3261 for one call over UDP) ---------------------------------------
def parse(data: bytes):
    head, _, body = data.decode("utf-8", "replace").partition("\r\n\r\n")
    lines = head.split("\r\n")
    hdr: dict[str, str] = {}
    for ln in lines[1:]:
        k, _, v = ln.partition(":")
        hdr.setdefault(k.strip().lower(), v.strip())
    return lines[0], hdr, body


def sdp(ip: str, port: int, direction: str = "sendrecv") -> str:
    sid = random.randint(10 ** 9, 10 ** 10 - 1)
    return "\r\n".join(["v=0", f"o=- {sid} {sid} IN IP4 {ip}", "s=sipbridge", f"c=IN IP4 {ip}", "t=0 0",
                        f"m=audio {port} RTP/AVP 0", "a=rtpmap:0 PCMU/8000", "a=ptime:20",
                        f"a={direction}", ""])


def sdp_peer(body: str):
    ip = port = None
    for ln in body.replace("\r\n", "\n").split("\n"):
        if ln.startswith("c=") and ip is None:
            ip = ln.split()[-1].split("/")[0]
        elif ln.startswith("m=audio") and port is None:
            pts = ln.split()[3:]
            if "0" not in pts:
                return None
            port = int(ln.split()[1])
    return (ip, port) if ip and port else None


PULSE_FMT = ["--rate=%d" % RATE, "--channels=1", "--format=s16le", "--raw", "--latency-msec=20"]


# ---- playback backends (8 kHz mono s16le in) ---------------------------------------------------
class PacatPlayer:
    """The test framework's path: pacat into a PulseAudio sink (the modem's RX sink, or the default one)."""

    def __init__(self, sink=None):
        self.proc = subprocess.Popen(["pacat", "-p", *(["-d", sink] if sink else []), *PULSE_FMT,
                                      "--client-name=sipbridge-rx"], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write(self, pcm: bytes):
        self.proc.stdin.write(pcm)
        self.proc.stdin.flush()

    def close(self):
        try:
            self.proc.terminate()
        except OSError:
            pass


class FfplayPlayer(PacatPlayer):
    def __init__(self):
        self.proc = subprocess.Popen(["ffplay", "-nodisp", "-loglevel", "error", "-fflags", "nobuffer",
                                      "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "-"],
                                     stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


class WavPlayer:
    def __init__(self, path):
        import wave
        self.w = wave.open(path, "wb")
        self.w.setnchannels(1)
        self.w.setsampwidth(2)
        self.w.setframerate(RATE)

    def write(self, pcm: bytes):
        self.w.writeframes(pcm)

    def close(self):
        self.w.close()


class StreamResampler:
    """8 kHz -> device rate, one 20 ms packet at a time, continuous across packets.

    The old per-packet np.interp restarted at every packet (a step at each join: a 50 Hz click train) and linear
    interpolation left the 4-8 kHz images only ~16 dB down (a harsh, clipped sound).  This keeps the input history
    and the fractional read position between calls and interpolates with a Kaiser-windowed sinc (cutoff 3.7 kHz,
    images ~-55 dB).  `step` (input samples per output sample) can be trimmed a little per call, which is how the
    player follows clock drift without dropping or repeating audio."""

    HALF, PHASES = 16, 512

    def __init__(self, in_rate: int, out_rate: int, cutoff_hz: float = 3700.0, beta: float = 8.0):
        self.nominal = self.step = in_rate / out_rate
        h, fc = self.HALF, cutoff_hz / in_rate               # fc in cycles per input sample
        k = np.arange(-h + 1, h + 1)                         # taps at i-h+1 .. i+h around the read position i+frac
        frac = np.arange(self.PHASES)[:, None] / self.PHASES
        d = k[None, :] - frac
        win = np.i0(beta * np.sqrt(np.clip(1 - (d / h) ** 2, 0, None))) / np.i0(beta)   # Kaiser, continuous in d
        tab = 2 * fc * np.sinc(2 * fc * d) * win
        self.tab = (tab / tab.sum(axis=1, keepdims=True)).astype(np.float32)   # unity gain at DC on every phase
        self.hist = np.zeros(2 * h, dtype=np.float32)
        self.pos = float(h)                                  # read position within hist+new

    def process(self, x: np.ndarray) -> np.ndarray:
        h = self.HALF
        buf = np.concatenate((self.hist, x.astype(np.float32)))
        n = int(np.floor((len(buf) - h - self.pos) / self.step))
        if n <= 0:
            self.hist = buf
            return np.zeros(0, dtype=np.float32)
        t = self.pos + np.arange(n) * self.step
        i = np.floor(t).astype(np.int64)
        ph = np.minimum(((t - i) * self.PHASES).astype(np.int64), self.PHASES - 1)
        idx = i[:, None] + np.arange(-h + 1, h + 1)[None, :]
        y = np.einsum("ij,ij->i", buf[idx], self.tab[ph])
        self.pos += n * self.step
        keep = int(self.pos) - h                             # drop input no longer under the kernel
        self.hist, self.pos = buf[keep:], self.pos - keep
        return y.astype(np.float32)


class SoundDevicePlayer:
    """PortAudio (pip install sounddevice) -- the path on Windows and macOS.  Resampled to the output device's
    own rate (StreamResampler) and played from a jitter buffer that holds about `prime_ms`: playout starts once
    that much is queued, and the resampler's rate is trimmed by up to TRIM to keep the fill there, so the two
    clocks (the sender's 8 kHz, this sound card) drifting apart never needs a drop.  An underrun fades out and
    re-primes, a far overrun (a stall) drops the oldest audio with a short crossfade -- never a hard edge."""

    TRIM = 0.004             # most the playout rate is nudged to follow the fill (0.4%: inaudible)
    FADE_S = 0.004           # fade at an underrun / re-prime / overrun splice

    def __init__(self, prime_ms=200, max_ms=1000):
        import sounddevice as sd
        self.rate = int(sd.query_devices(kind="output")["default_samplerate"])
        self.rs = StreamResampler(RATE, self.rate)
        self.prime, self.max = int(self.rate * prime_ms / 1000), int(self.rate * max_ms / 1000)
        self.fade = max(8, int(self.rate * self.FADE_S))
        self.buf, self.lock, self.primed, self.fill = np.zeros(0, dtype=np.float32), threading.Lock(), False, 0.0
        self.fade_in = False
        self.underruns = self.overruns = 0
        self.stream = sd.OutputStream(samplerate=self.rate, channels=1, dtype="float32", callback=self._cb,
                                      latency="high")
        self.stream.start()

    def _cb(self, out, frames, _time, _status):
        with self.lock:
            if not self.primed and len(self.buf) >= self.prime:
                self.primed, self.fade_in = True, True
            if not self.primed:
                out.fill(0)
                return
            n = min(frames, len(self.buf))
            y, self.buf = self.buf[:n].copy(), self.buf[n:]
            if self.fade_in:                                 # re-primed after silence: ramp up, not a step
                m = min(self.fade, n)
                y[:m] *= np.linspace(0, 1, m, dtype=np.float32)
                self.fade_in = False
            if n < frames:                                   # underrun: fade the tail out and wait to re-prime
                m = min(self.fade, n)
                if m:
                    y[n - m:] *= np.linspace(1, 0, m, dtype=np.float32)
                self.primed = False
                self.underruns += 1
            out[:n, 0] = y
            out[n:, 0] = 0

    def write(self, pcm: bytes):
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        with self.lock:
            fill = len(self.buf)
        self.fill += 0.05 * (fill - self.fill)               # ~1 s smoothing at 50 packets/s: jitter averages out
        err = (self.fill - self.prime) / self.prime          # > 0: too much queued -> read the input faster
        self.rs.step = self.rs.nominal * (1 + float(np.clip(0.02 * err, -self.TRIM, self.TRIM)))
        y = self.rs.process(x)
        with self.lock:
            self.buf = np.concatenate((self.buf, y))
            if len(self.buf) > self.max:                    # far behind (a stall): back to the prime level,
                cut = len(self.buf) - self.prime            # crossfading the splice so it is not a click
                m = min(self.fade, self.prime)
                head, tail = self.buf[:m], self.buf[cut:cut + m]
                r = np.linspace(0, 1, m, dtype=np.float32)
                self.buf = self.buf[cut:].copy()
                self.buf[:m] = head * (1 - r) + tail * r
                self.fill = float(self.prime)
                self.overruns += 1

    def close(self):
        self.stream.stop()
        self.stream.close()


class MuteGate:
    """listen --mute-file: while PATH exists, the player gets silence instead of the call audio.  The call and the
    output device stay open, so a GUI can mute/unmute without dropping out of the conference."""

    def __init__(self, player, path):
        self.player, self.path, self.muted, self.t_check = player, path, True, 0.0

    def write(self, pcm: bytes):
        now = time.monotonic()
        if now - self.t_check >= 0.1:
            self.muted, self.t_check = os.path.exists(self.path), now
        self.player.write(bytes(len(pcm)) if self.muted else pcm)

    def close(self):
        self.player.close()


def make_player(spec: str):
    """--player: auto | pulse | sounddevice | ffplay | wav:FILE.  auto = the first that opens."""
    import shutil
    if spec.startswith("wav:"):
        return WavPlayer(spec[4:])
    order = ["pulse", "sounddevice", "ffplay"] if spec == "auto" else [spec]
    errors = []
    for kind in order:
        try:
            if kind == "pulse":
                if not shutil.which("pacat") or subprocess.run(["pactl", "info"], capture_output=True).returncode:
                    raise RuntimeError("no PulseAudio/PipeWire server (pacat/pactl)")
                return PacatPlayer()
            if kind == "sounddevice":
                return SoundDevicePlayer()
            if kind == "ffplay":
                if not shutil.which("ffplay"):
                    raise RuntimeError("ffplay not on PATH")
                return FfplayPlayer()
            raise RuntimeError("unknown player")
        except Exception as e:                            # ImportError, PortAudioError, ...
            errors.append(f"{kind}: {e}")
    raise SystemExit("no audio output available (" + "; ".join(errors) + ")\n"
                     "macOS: pip3 install numpy sounddevice   Linux: pulseaudio-utils / pipewire-pulse")


class Audio:
    """parec -> RTP out, RTP in -> pacat.  One instance per call.

    listen mode: `player` replaces pacat and `silent_tx` replaces parec with paced digital silence."""

    def __init__(self, source, sink, rtp: socket.socket, peer, player=None, silent_tx=False):
        self.rtp, self.peer, self.stop = rtp, peer, threading.Event()
        self.tx = self.rx = self.lost = self.late = self.jumps = 0
        self.seq, self.ts, self.ssrc = random.randint(0, 65535), random.randint(0, 2 ** 32 - 1), random.getrandbits(32)
        self.silent_tx = silent_tx
        self.rec = None if silent_tx else subprocess.Popen(
            ["parec", *(["-d", source] if source else []), *PULSE_FMT, "--client-name=sipbridge-tx"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.play = player or PacatPlayer(sink)
        self.threads = [threading.Thread(target=self._send, daemon=True),
                        threading.Thread(target=self._recv, daemon=True)]
        for t in self.threads:
            t.start()

    def _frames(self):
        """20 ms of PCM per item: from parec, or listen digital silence paced by the clock."""
        need = SAMPLES * 2
        if not self.silent_tx:
            while True:
                buf = self.rec.stdout.read(need)
                if len(buf) < need:
                    return
                yield buf
        silence, nxt = bytes(need), time.monotonic()
        while not self.stop.is_set():
            yield silence
            nxt += PTIME
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -1.0:                             # suspended laptop etc.: don't burst to catch up
                nxt = time.monotonic()

    def _send(self):
        # A packet that leaves more than the far end's playout delay (the conference primes 100 ms) behind its 20 ms
        # schedule is played as silence there.  Log every send stall past STALL_S so a hole in the room's audio can be
        # told apart: stalled here (parec / this host's load) or dropped downstream (network / bridge).
        t_first, stalled = None, False
        for buf in self._frames():
            if self.stop.is_set():
                break
            now = time.monotonic()
            if t_first is None:
                t_first = now - self.tx * PTIME
            behind = now - (t_first + self.tx * PTIME)
            if behind < 0:
                t_first += behind                          # early: re-anchor, only lateness counts
            if behind > STALL_S and not stalled:           # one line per stall, at its first late packet
                log(f"RTP TX STALL t={time.time():.3f} seq={self.seq & 0xFFFF} behind={behind * 1000:.0f}ms")
            stalled = behind > STALL_S / 2
            pkt = struct.pack("!BBHII", 0x80, PCMU, self.seq & 0xFFFF, self.ts & 0xFFFFFFFF, self.ssrc)
            try:
                self.rtp.sendto(pkt + lin2ulaw(np.frombuffer(buf, dtype="<i2")), self.peer)
            except OSError:
                if not self.silent_tx:
                    raise
            self.seq += 1
            self.ts += SAMPLES
            self.tx += 1

    def _recv(self):
        last = None
        silence = bytes(SAMPLES * 2)
        while not self.stop.is_set():
            r, _, _ = select.select([self.rtp], [], [], 0.2)
            if not r:
                continue
            pkt, _ = self.rtp.recvfrom(2048)
            if len(pkt) < 12 or pkt[1] & 0x7F != PCMU:
                continue                                   # telephone-event, comfort noise, runt
            seq = struct.unpack("!H", pkt[2:4])[0]
            d = 1 if last is None else (seq - last) & 0xFFFF
            if d == 0 or d >= 0x8000:                      # duplicate, or older than a frame already played:
                self.late += 1                             # its slot went out as silence -- playing it now would
                log(f"RTP LATE t={time.time():.3f} seq={seq} behind={(last - seq) & 0xFFFF}")   # shift all later audio
                continue
            gap = d - 1
            if 0 < gap < 50:                               # a lost packet becomes silence, keeping the timing
                self.lost += gap
                log(f"RTP LOSS t={time.time():.3f} seq={seq} gap={gap}")
                try:
                    self.play.write(silence * gap)
                except OSError as e:
                    log(f"AUDIO OUT FAILED {e!r}")
                    return
            elif gap:
                self.jumps += 1
                log(f"RTP JUMP t={time.time():.3f} seq={seq} gap={gap}")
            last = seq
            self.rx += 1
            try:
                self.play.write(ulaw2lin(pkt[12:]).astype("<i2").tobytes())
            except OSError as e:
                log(f"AUDIO OUT FAILED {e!r}")
                return

    def close(self):
        self.stop.set()
        if self.rec:
            try:
                self.rec.terminate()
            except OSError:
                pass
        try:
            self.play.close()
        except Exception:
            pass
        log(f"STATS tx={self.tx} rx={self.rx} lost={self.lost} late={self.late} jumps={self.jumps}")


class Call:
    def __init__(self, a):
        self.a = a
        self.ip = a.advertise
        self.sip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host, port = (a.listen or f"{a.bind}:0").rsplit(":", 1)
        self.sip.bind((host, int(port)))
        self.rtp.bind((a.bind, a.rtp_port))
        self.sip_port = self.sip.getsockname()[1]
        self.rtp_port = self.rtp.getsockname()[1]
        self.audio: Audio | None = None
        self.call_id = f"{random.getrandbits(48):x}@{self.ip}"
        self.tag = f"{random.getrandbits(32):x}"
        self.quit = threading.Event()
        self.reason = "signal"
        self.up = False
        self.sent_bye = False
        self.reg_dest = None
        self.reg_call_id = f"reg{random.getrandbits(48):x}@{self.ip}"
        self.reg_cseq = 0

    def _msg(self, first, hdrs, body=""):
        h = [first, *hdrs, f"Content-Length: {len(body.encode())}"]
        if body:
            h.insert(len(h) - 1, "Content-Type: application/sdp")
        return ("\r\n".join(h) + "\r\n\r\n" + body).encode()

    def _recv(self, timeout):
        r, _, _ = select.select([self.sip], [], [], timeout)
        if not r:
            return None
        data, addr = self.sip.recvfrom(8192)
        return (*parse(data), addr)

    def _media(self, body, addr):
        peer = sdp_peer(body)
        if not peer:
            return False
        listen = self.a.mode == "listen"
        self.audio = Audio(self.a.source, self.a.sink, self.rtp, peer,
                           player=getattr(self.a, "player_obj", None), silent_tx=listen)
        self.up = True
        if getattr(self.a, "stats_file", None):
            threading.Thread(target=self._stats_writer, daemon=True).start()
        log(f"CALL UP peer={peer[0]}:{peer[1]}")
        return True

    def _bye(self, to, hdrs_extra=()):
        if self.sent_bye:
            return
        self.sent_bye = True
        self.sip.sendto(self._msg(f"BYE {self.remote_uri} SIP/2.0", [
            f"Via: SIP/2.0/UDP {self.ip}:{self.sip_port};branch=z9hG4bK{random.getrandbits(32):x}",
            "Max-Forwards: 70", self.from_h, self.to_h, f"Call-ID: {self.call_id}",
            f"CSeq: {self.cseq + 1} BYE"]), to)

    # -- caller ---------------------------------------------------------------------------------
    def run_call(self):
        a = self.a
        host, port = a.to.rsplit(":", 1)
        dest = (host, int(port))
        self.remote_uri = f"sip:{a.number}@{a.to}"
        self.from_h = f"From: <sip:{a.user}@{self.ip}>;tag={self.tag}"
        self.to_h = f"To: <{self.remote_uri}>"
        self.cseq = 1
        branch = f"z9hG4bK{random.getrandbits(32):x}"
        invite = self._msg(f"INVITE {self.remote_uri} SIP/2.0", [
            f"Via: SIP/2.0/UDP {self.ip}:{self.sip_port};branch={branch};rport", "Max-Forwards: 70",
            self.from_h, self.to_h, f"Call-ID: {self.call_id}", "CSeq: 1 INVITE",
            f"Contact: <sip:{a.user}@{self.ip}:{self.sip_port}>"],
            sdp(self.ip, self.rtp_port, "recvonly" if a.mode == "listen" else "sendrecv"))
        log(f"READY sip={self.ip}:{self.sip_port} rtp={self.rtp_port}")
        deadline, nxt, rtx = time.time() + a.ring_timeout, 0.0, 0.5
        self.sip.sendto(invite, dest)
        while not self.quit.is_set() and not self.up:
            if time.time() > deadline:
                self.reason = "no answer"
                return 1
            m = self._recv(0.2)
            if m is None:
                if time.time() > nxt + rtx and not self.up:
                    nxt = time.time()
                    rtx = min(rtx * 2, 4)
                    self.sip.sendto(invite, dest)
                continue
            first, h, body, addr = m
            code = int(first.split()[1]) if first.startswith("SIP/2.0") else 0
            if code == 180 or code == 183:
                log("RINGING")
            elif code >= 300:
                self.reason = f"refused {first[8:]}"
                self.sip.sendto(self._msg(f"ACK {self.remote_uri} SIP/2.0", [
                    f"Via: SIP/2.0/UDP {self.ip}:{self.sip_port};branch={branch}", self.from_h,
                    f"To: {h.get('to', '')}", f"Call-ID: {self.call_id}", "CSeq: 1 ACK"]), dest)
                return 1
            elif code == 200:
                self.to_h = f"To: {h.get('to', '')}"
                if h.get("contact"):
                    mm = re.search(r"<([^>]+)>", h["contact"])
                    self.remote_uri = mm.group(1) if mm else self.remote_uri
                self.sip.sendto(self._msg(f"ACK {self.remote_uri} SIP/2.0", [
                    f"Via: SIP/2.0/UDP {self.ip}:{self.sip_port};branch=z9hG4bK{random.getrandbits(32):x}",
                    "Max-Forwards: 70", self.from_h, self.to_h, f"Call-ID: {self.call_id}", "CSeq: 1 ACK"]), dest)
                if not self._media(body, addr):
                    self.reason = "no PCMU in answer"
                    self._bye(dest)
                    return 1
        return self._in_call(dest)

    # -- registration (answer --register) ---------------------------------------------------------
    def _send_register(self, expires):
        """REGISTER from the SIP socket that also receives the INVITE: the server rings the address it saw."""
        a = self.a
        self.reg_cseq += 1
        uri = f"<sip:{a.user}@{a.register}>"
        self.sip.sendto(self._msg(f"REGISTER sip:{a.register} SIP/2.0", [
            f"Via: SIP/2.0/UDP {self.ip}:{self.sip_port};branch=z9hG4bK{random.getrandbits(32):x};rport",
            "Max-Forwards: 70", f"From: {uri};tag={self.tag}", f"To: {uri}", f"Call-ID: {self.reg_call_id}",
            f"CSeq: {self.reg_cseq} REGISTER", f"Contact: <sip:{a.user}@{self.ip}:{self.sip_port}>",
            f"Expires: {expires}"]), self.reg_dest)

    def _register(self):
        host, port = self.a.register.rsplit(":", 1)
        self.reg_dest = (host, int(port))
        deadline, nxt = time.time() + self.a.reg_timeout, 0.0
        while not self.quit.is_set() and time.time() < deadline:
            if time.time() >= nxt:
                self._send_register(self.a.reg_expires)
                nxt = time.time() + 1.0
            m = self._recv(0.2)
            if m and m[0].startswith("SIP/2.0 200") and "REGISTER" in m[1].get("cseq", ""):
                log(f"REGISTERED user={self.a.user} server={self.a.register}")
                self.reg_next = time.time() + self.a.reg_expires / 2
                return True
        self.reason = "register failed"
        return False

    # -- callee ---------------------------------------------------------------------------------
    def run_answer(self):
        if self.a.register and not self._register():
            return 1
        log(f"READY sip={self.ip}:{self.sip_port} rtp={self.rtp_port}")
        inv = None
        while not self.quit.is_set():
            if self.reg_dest and time.time() >= self.reg_next:      # refresh while waiting for the call
                self._send_register(self.a.reg_expires)
                self.reg_next = time.time() + self.a.reg_expires / 2
            m = self._recv(0.5)
            if m and m[0].startswith("INVITE"):
                inv = m
                break
        if inv is None:
            return 1
        first, h, body, addr = inv
        peer = sdp_peer(body)
        via, frm, to, cid, cseq = (h.get(k, "") for k in ("via", "from", "to", "call-id", "cseq"))
        tag = f";tag={self.tag}"

        def reply(code, text, extra=(), body=""):
            self.sip.sendto(self._msg(f"SIP/2.0 {code} {text}", [
                f"Via: {via}", f"From: {frm}", f"To: {to}{tag}", f"Call-ID: {cid}", f"CSeq: {cseq}", *extra], body), addr)

        if not peer:
            reply(488, "Not Acceptable Here")
            self.reason = "no PCMU in offer"
            return 1
        reply(180, "Ringing")
        log("RINGING")
        time.sleep(self.a.answer_delay)
        contact = f"Contact: <sip:{self.a.user}@{self.ip}:{self.sip_port}>"
        ok_body = sdp(self.ip, self.rtp_port)
        reply(200, "OK", [contact], ok_body)
        self.remote_uri = (re.search(r"<([^>]+)>", h.get("contact", "")) or [None, f"sip:x@{addr[0]}:{addr[1]}"])[1]
        self.from_h, self.to_h, self.cseq = f"From: {to}{tag}", f"To: {frm}", 1
        deadline, tries = time.time() + 0.5, 0
        while not self.quit.is_set():                      # wait for the ACK, resend the 200 on silence
            m = self._recv(0.1)
            if m and m[0].startswith("ACK"):
                break
            if time.time() > deadline:
                tries += 1
                if tries > 6:
                    self.reason = "no ACK"
                    return 1
                reply(200, "OK", [contact], ok_body)
                deadline = time.time() + 0.5
        self._media(body, addr)
        return self._in_call(addr)

    def _in_call(self, dest):
        t_end = time.time() + self.a.max_s if self.a.max_s else None
        while not self.quit.is_set():
            if t_end and time.time() > t_end:
                self.reason = "time limit"
                break
            m = self._recv(0.2)
            if m and m[0].startswith("BYE"):
                _, h, _, addr = m
                self.sip.sendto(self._msg("SIP/2.0 200 OK", [
                    f"Via: {h.get('via', '')}", f"From: {h.get('from', '')}", f"To: {h.get('to', '')}",
                    f"Call-ID: {h.get('call-id', '')}", f"CSeq: {h.get('cseq', '')}"]), addr)
                self.reason = "bye from far end"
                self.sent_bye = True
                return 0
        if self.quit.is_set() and self.reason == "signal":
            pass
        self._bye(dest)
        return 0

    def _stats_writer(self):
        """Opt-in --stats-file: live tx/rx/lost counters as JSON, rewritten atomically ~2x/s, for external GUIs."""
        import json
        path = self.a.stats_file
        while True:
            au = self.audio
            doc = {"t": time.time(), "up": bool(self.up and not self.quit.is_set() and au and not au.stop.is_set()),
                   "mode": self.a.mode, "tx": au.tx if au else 0, "rx": au.rx if au else 0,
                   "lost": au.lost if au else 0, "late": au.late if au else 0,
                   "jumps": au.jumps if au else 0, "ptime_ms": int(PTIME * 1000), "payload_bytes": SAMPLES}
            try:
                with open(path + ".tmp", "w") as f:
                    json.dump(doc, f)
                os.replace(path + ".tmp", path)
            except OSError:
                pass
            if au is None or au.stop.is_set():
                return
            time.sleep(0.5)

    def finish(self):
        if self.reg_dest:
            try:
                self._send_register(0)
            except OSError:
                pass
        if self.audio:
            self.audio.close()
        log(f"CALL ENDED reason={self.reason}")


def route_ip(host: str) -> str:
    """The local address the OS would use to reach host (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((host, 9))
        return s.getsockname()[0]
    except OSError:
        return "0.0.0.0"
    finally:
        s.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=("call", "answer", "listen"),
                   help="listen = receive-only call for remote monitoring (see the module docstring)")
    p.add_argument("--to", help="call: SIP server/peer HOST:PORT")
    p.add_argument("--number", default="1234", help="call: user part of the request URI")
    p.add_argument("--listen", help="answer: HOST:PORT for SIP (call: local SIP port is random)")
    p.add_argument("--user", default="modem")
    p.add_argument("--bind", default="127.0.0.1", help="local address for RTP (and SIP when --listen is absent)")
    p.add_argument("--advertise", help="address put in SDP/Contact (default: --bind)")
    p.add_argument("--rtp-port", type=int, default=0)
    p.add_argument("--source", help="pulse source whose audio is sent (e.g. SINK.monitor); default source")
    p.add_argument("--sink", help="pulse sink that plays the far end's audio; default sink")
    p.add_argument("--register", help="answer: REGISTER --user with this SIP server HOST:PORT first, so it can ring us "
                                      "(unauthenticated; refreshed while waiting, removed on exit)")
    p.add_argument("--reg-expires", type=int, default=60, help="REGISTER Expires seconds (sipserver caps at 120)")
    p.add_argument("--reg-timeout", type=float, default=35.0,
                   help="answer: give up on the initial REGISTER after this long. Must stay well under "
                        "whatever the caller (meshdialer's leg.ready.wait) will itself wait for "
                        "READY, so a slow-but-alive server shows up as 'register failed' instead of "
                        "the caller giving up first and reporting the less useful 'not ready'.")
    p.add_argument("--ring-timeout", type=float, default=30.0)
    p.add_argument("--answer-delay", type=float, default=0.5)
    p.add_argument("--max-s", type=float, default=0.0, help="hang up after this long (0 = until BYE/signal)")
    p.add_argument("--player", default="auto",
                   help="listen: auto | pulse | sounddevice | ffplay | wav:FILE (default auto)")
    p.add_argument("--mute-file", help="listen: play silence while this file exists; toggle it to mute without hanging up")
    p.add_argument("--stats-file", help="write live tx/rx/lost counters here as JSON (for meshgui.py); off by default")
    a = p.parse_args(argv)
    if a.mode == "listen":
        if not a.to:
            p.error("listen needs --to HOST:PORT (and --number, e.g. 3000 for the mesh conference)")
        if a.bind == "127.0.0.1" and not a.to.startswith("127."):
            a.bind = "0.0.0.0"
        if not a.advertise:
            a.advertise = route_ip(a.to.rsplit(":", 1)[0])
        if a.user == "modem":
            a.user = "listen-" + re.sub(r"[^A-Za-z0-9_.-]", "", socket.gethostname().split(".")[0])[:24]
        a.player_obj = make_player(a.player)       # fail before joining the room, not after
        if a.mute_file:
            a.player_obj = MuteGate(a.player_obj, a.mute_file)
    a.advertise = a.advertise or a.bind
    if a.mode == "call" and not a.to:
        p.error("call needs --to")
    if a.register and a.mode != "answer":
        p.error("--register is for answer mode")
    c = Call(a)
    signal.signal(signal.SIGTERM, lambda *_: c.quit.set())
    signal.signal(signal.SIGINT, lambda *_: c.quit.set())
    try:
        rc = c.run_call() if a.mode in ("call", "listen") else c.run_answer()
    finally:
        c.finish()
    return 0 if c.up and rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
