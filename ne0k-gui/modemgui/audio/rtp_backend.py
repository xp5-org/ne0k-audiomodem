# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""RTP connector: G.711 PCMU over a SIP call as an audio endpoint.

The modems work on mono float32 at 48 kHz (the OFDM handshake only runs at 48000/44100); a phone
line carries 8 kHz u-law.  This connector owns that adaptation, so the OFDM/CSS code above it
cannot tell a SIP call from a sound card:

    RX  RTP PCMU packets -> u-law decode -> x6 interpolate -> blocks of float32 at ``spec.rate``
    TX  pull(n) at ``spec.rate`` -> /6 decimate -> u-law encode -> one 20 ms RTP packet per tick

The SIP side (asyncio, owned by the fax modem) hands packets in with ``RtpPort.feed`` and gets
outgoing packets through the ``send`` callable it installs with ``RtpPort.attach``.  Everything
here runs on the audio threads, never on the event loop.

Address: ``rtp:<port name>``.  The TX side sends silence-padded audio continuously once the
remote end is known, so the far receiver always sees a gap-free sample clock.
"""
from __future__ import annotations

import queue
import random
import struct
import threading
import time
from typing import Callable

import numpy as np
from scipy.signal import firwin, lfilter

from .base import AudioError, Connector, DeviceInfo, Sink, SINK, SOURCE, Source, StreamSpec

PTIME = 0.020                 # one RTP packet per 20 ms
PCM_RATE = 8000
PKT_SAMPLES = 160
PT_PCMU = 0
_BIAS, _CLIP = 0x84, 32635


def _ulaw_decode_table() -> np.ndarray:
    t = np.zeros(256, np.float32)
    for u in range(256):
        v = ~u & 0xFF
        s = ((((v & 0x0F) << 3) + _BIAS) << ((v >> 4) & 7)) - _BIAS
        t[u] = (-s if v & 0x80 else s) / 32768.0
    return t


_DEC = _ulaw_decode_table()


def ulaw_decode(b: bytes) -> np.ndarray:
    return _DEC[np.frombuffer(b, np.uint8)]


def ulaw_encode(x: np.ndarray) -> bytes:
    """float32 (-1..1) -> PCMU bytes."""
    s = np.clip(np.round(np.asarray(x, np.float64) * 32768.0), -32768, 32767).astype(np.int32)
    sign = np.where(s < 0, 0x80, 0)
    m = np.minimum(np.abs(s), _CLIP) + _BIAS
    exp = np.floor(np.log2(m)).astype(np.int32) - 7
    mant = (m >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant) & 0xFF).astype(np.uint8).tobytes()


class _Resampler:
    """Streaming x``k`` interpolator or /``k`` decimator.  Blocks must be whole multiples of ``k``
    so the decimation phase never slips."""

    def __init__(self, k: int, up: bool) -> None:
        self.k, self.up = k, up
        fs = PCM_RATE * k
        # flat through the 3.3 kHz fax/modem band, >60 dB down by 4 kHz
        self.b = firwin(241, 3650.0, window=("kaiser", 8.0), fs=fs).astype(np.float64)
        self.zi = np.zeros(len(self.b) - 1)

    def run(self, x: np.ndarray) -> np.ndarray:
        if self.up:
            z = np.zeros(len(x) * self.k)
            z[::self.k] = x * self.k
            y, self.zi = lfilter(self.b, 1.0, z, zi=self.zi)
            return y.astype(np.float32)
        y, self.zi = lfilter(self.b, 1.0, np.asarray(x, np.float64), zi=self.zi)
        return y[::self.k].astype(np.float32)


class RtpPort:
    """One RTP media leg, shared between the SIP code (event loop) and the audio threads."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.rx: "queue.Queue[bytes | None]" = queue.Queue(maxsize=2000)
        self._send: Callable[[bytes], None] | None = None
        self.remote: tuple[str, int] | None = None
        self.closed = threading.Event()
        self.rx_packets = self.tx_packets = self.rx_gaps = 0

    # -- SIP side -------------------------------------------------------------

    def attach(self, send: Callable[[bytes], None]) -> None:
        """``send(packet)`` must be safe to call from any thread."""
        self._send = send

    def set_remote(self, addr: tuple[str, int]) -> None:
        self.remote = addr

    def feed(self, packet: bytes) -> None:
        if len(packet) < 12 or packet[0] >> 6 != 2 or (packet[1] & 0x7F) != PT_PCMU:
            return
        try:
            self.rx.put_nowait(packet)
        except queue.Full:
            pass

    def close(self) -> None:
        self.closed.set()
        try:
            self.rx.put_nowait(None)
        except queue.Full:
            pass

    # -- audio side -------------------------------------------------------------

    def send_packet(self, packet: bytes) -> bool:
        if self._send is None or self.remote is None or self.closed.is_set():
            return False
        self._send(packet)
        self.tx_packets += 1
        return True


class RtpSource(Source):
    def __init__(self, port: RtpPort, spec: StreamSpec) -> None:
        self.port, self.spec = port, spec
        self.k = spec.rate // PCM_RATE
        self._up = _Resampler(self.k, up=True)
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self.error = None
        self.finished = False

    def start(self, on_block) -> None:
        self._t = threading.Thread(target=self._run, args=(on_block,), name="rtp-rx", daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        self.port.close()
        if self._t:
            self._t.join(2.0)

    def _run(self, on_block) -> None:
        expect = None
        try:
            while not self._stop.is_set():
                try:
                    pkt = self.port.rx.get(timeout=0.2)
                except queue.Empty:
                    continue
                if pkt is None:
                    break
                seq = struct.unpack("!H", pkt[2:4])[0]
                cc = pkt[0] & 0x0F
                pcm = ulaw_decode(pkt[12 + 4 * cc:])
                self.port.rx_packets += 1
                if expect is not None:
                    gap = (seq - expect) & 0xFFFF
                    if 0 < gap < 50:             # lost packets: keep the sample clock honest
                        self.port.rx_gaps += gap
                        for _ in range(gap):
                            on_block(self._up.run(np.zeros(PKT_SAMPLES, np.float32)))
                    elif gap >= 50:              # a jump or a reorder: resync, insert nothing
                        pass
                expect = (seq + 1) & 0xFFFF
                if len(pcm) % 1:
                    continue
                on_block(self._up.run(pcm))
        except Exception as e:                    # surface it through the pipe's rx_error
            self.error = f"rtp rx: {type(e).__name__}: {e}"
        self.finished = True


class RtpSink(Sink):
    def __init__(self, port: RtpPort, spec: StreamSpec) -> None:
        self.port, self.spec = port, spec
        self.k = spec.rate // PCM_RATE
        self._down = _Resampler(self.k, up=False)
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self.error = None
        self.seq = random.getrandbits(16)
        self.ts = random.getrandbits(32)
        self.ssrc = random.getrandbits(32)

    def start(self, pull) -> None:
        self._t = threading.Thread(target=self._run, args=(pull,), name="rtp-tx", daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(2.0)

    def _run(self, pull) -> None:
        n = PKT_SAMPLES * self.k
        nxt = time.monotonic()
        first = True
        try:
            while not self._stop.is_set():
                nxt += PTIME
                x = np.asarray(pull(n), np.float32)
                payload = ulaw_encode(self._down.run(x))
                self.seq = (self.seq + 1) & 0xFFFF
                self.ts = (self.ts + PKT_SAMPLES) & 0xFFFFFFFF
                hdr = struct.pack("!BBHII", 0x80, (0x80 if first else 0) | PT_PCMU,
                                  self.seq, self.ts, self.ssrc)
                if self.port.send_packet(hdr + payload):
                    first = False
                delay = nxt - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                elif delay < -1.0:               # stalled (a GC, a suspended VM): do not burst to catch up
                    nxt = time.monotonic()
        except Exception as e:
            self.error = f"rtp tx: {type(e).__name__}: {e}"


class RtpConnector(Connector):
    name = "rtp"
    description = "G.711 PCMU over RTP (a SIP call), resampled to the modem rate"

    def __init__(self) -> None:
        self._ports: dict[str, RtpPort] = {}
        self._lock = threading.Lock()

    def port(self, name: str) -> RtpPort:
        with self._lock:
            return self._ports.setdefault(name, RtpPort(name))

    def drop(self, name: str) -> None:
        with self._lock:
            p = self._ports.pop(name, None)
        if p:
            p.close()

    def list_devices(self) -> list[DeviceInfo]:
        with self._lock:
            names = list(self._ports)
        return [DeviceInfo("rtp", n, f"RTP call {n}", k) for n in names for k in (SOURCE, SINK)]

    def _check(self, spec: StreamSpec) -> None:
        if spec.rate % PCM_RATE:
            raise AudioError(f"rtp: sample rate {spec.rate} is not a multiple of {PCM_RATE}")

    def open_source(self, dev_id: str, spec: StreamSpec) -> Source:
        self._check(spec)
        return RtpSource(self.port(dev_id), spec)

    def open_sink(self, dev_id: str, spec: StreamSpec) -> Sink:
        self._check(spec)
        return RtpSink(self.port(dev_id), spec)
