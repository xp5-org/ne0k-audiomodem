# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Hardware-free connectors: WAV files and in-memory loopback.

``file:<path>[#opts]``  source reads a WAV, sink writes one.
``loop:<name>``         a named in-memory bus; whatever a ``loop:x`` SINK plays is
                        what a ``loop:x`` SOURCE hears.  Two pipes in one process
                        (or a test) can talk through it with no audio stack.
``null:``               a source of silence / a sink that discards.

Options after ``#`` on a file address, comma separated:
  fast     do not pace to real time (offline decode of a capture)
  repeat   source loops the file

Everything is paced to wall-clock by default so a spectrograph or the TX
"is it still playing" logic behaves as it would on a real device.
"""
from __future__ import annotations

import threading
import time
import wave
from collections import deque
from typing import Callable

import numpy as np

from .base import (SINK, SOURCE, AudioError, Connector, DeviceInfo, Sink,
                   Source, StreamSpec, as_mono_f32)


class _Pacer:
    """Sleeps so that block k completes at start + k*block/rate."""

    def __init__(self, rate: int, enabled: bool) -> None:
        self.rate, self.enabled = rate, enabled
        self.t0, self.done = time.monotonic(), 0

    def tick(self, n: int, stop: threading.Event) -> None:
        self.done += n
        if self.enabled:
            wait = self.t0 + self.done / self.rate - time.monotonic()
            if wait > 0:
                stop.wait(wait)


def _parse_opts(dev_id: str) -> tuple[str, set[str]]:
    path, _, opts = dev_id.partition("#")
    return path, {o for o in opts.split(",") if o}


def read_wav(path: str, rate: int) -> np.ndarray:
    """Whole file -> mono float32 at `rate`."""
    try:
        with wave.open(path, "rb") as w:
            fs, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except (OSError, wave.Error, EOFError) as e:
        raise AudioError(f"cannot read {path}: {e}") from e
    if sw == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise AudioError(f"{path}: unsupported sample width {sw}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if fs != rate:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(fs, rate)
        x = resample_poly(x, rate // g, fs // g).astype(np.float32)
    return as_mono_f32(x)


class FileSource(Source):
    def __init__(self, dev_id: str, spec: StreamSpec) -> None:
        self.spec = spec
        self._path, self._opts = _parse_opts(dev_id)
        self._data = read_wav(self._path, spec.rate)     # fail at open, not at start
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def start(self, on_block: Callable[[np.ndarray], None]) -> None:
        self.finished = False
        self._stop.clear()
        self._th = threading.Thread(target=self._run, args=(on_block,),
                                    name="file-src", daemon=True)
        self._th.start()

    def _run(self, on_block) -> None:
        n, pace = self.spec.block, _Pacer(self.spec.rate, "fast" not in self._opts)
        while not self._stop.is_set():
            for i in range(0, len(self._data), n):
                if self._stop.is_set():
                    return
                on_block(self._data[i:i + n])
                pace.tick(n, self._stop)
            if "repeat" not in self._opts:
                self.finished = True
                return

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=2.0)


class FileSink(Sink):
    def __init__(self, dev_id: str, spec: StreamSpec) -> None:
        self.spec = spec
        self._path, self._opts = _parse_opts(dev_id)
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self._w: wave.Wave_write | None = None
        self.clipped = 0

    def start(self, pull: Callable[[int], np.ndarray]) -> None:
        try:
            self._w = wave.open(self._path, "wb")
        except OSError as e:
            raise AudioError(f"cannot write {self._path}: {e}") from e
        self._w.setnchannels(1)
        self._w.setsampwidth(2)
        self._w.setframerate(self.spec.rate)
        self._stop.clear()
        self._th = threading.Thread(target=self._run, args=(pull,),
                                    name="file-sink", daemon=True)
        self._th.start()

    def _run(self, pull) -> None:
        n, pace = self.spec.block, _Pacer(self.spec.rate, "fast" not in self._opts)
        while not self._stop.is_set():
            x = as_mono_f32(pull(n))
            self.clipped += int(np.count_nonzero(np.abs(x) > 1.0))
            self._w.writeframes((np.clip(x, -1, 1) * 32767.0).astype("<i2").tobytes())
            pace.tick(n, self._stop)

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=2.0)
        if self._w:
            self._w.close()
            self._w = None


class _Bus:
    """Bounded FIFO of samples shared by a loop: sink and a loop: source."""

    def __init__(self, cap_s: float = 30.0, rate: int = 48000) -> None:
        self.q: deque[np.ndarray] = deque()
        self.n = 0
        self.cap = int(cap_s * rate)
        self.lock = threading.Condition()
        self.last_put = 0.0

    def put(self, x: np.ndarray) -> None:
        with self.lock:
            self.q.append(x)
            self.n += len(x)
            self.last_put = time.monotonic()
            while self.n > self.cap:            # reader gone: shed the oldest
                self.n -= len(self.q.popleft())
            self.lock.notify_all()

    def get(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        got = 0
        with self.lock:
            # The sink and source are paced by separate timers, so on a host with coarse timers
            # (Windows) the source can ask a hair before the sink delivers.  Zero-padding then would
            # splice a silent gap into a live burst; wait briefly while the writer is streaming.
            if self.n < n and time.monotonic() - self.last_put < 0.25:
                self.lock.wait_for(lambda: self.n >= n, timeout=0.1)
            while got < n and self.q:
                head = self.q[0]
                take = min(n - got, len(head))
                out[got:got + take] = head[:take]
                got += take
                if take == len(head):
                    self.q.popleft()
                else:
                    self.q[0] = head[take:]
                self.n -= take
        return out


_BUSES: dict[str, _Bus] = {}
_BUSES_LOCK = threading.Lock()


def _bus(name: str, rate: int) -> _Bus:
    with _BUSES_LOCK:
        return _BUSES.setdefault(name, _Bus(rate=rate))


class _PumpSource(Source):
    """Source that calls `produce(n)` once per paced block."""

    def __init__(self, spec: StreamSpec, produce: Callable[[int], np.ndarray]) -> None:
        self.spec, self._produce = spec, produce
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def start(self, on_block) -> None:
        self._stop.clear()

        def run():
            n, pace = self.spec.block, _Pacer(self.spec.rate, True)
            while not self._stop.is_set():
                on_block(self._produce(n))
                pace.tick(n, self._stop)
        self._th = threading.Thread(target=run, name="pump-src", daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=2.0)


class _PumpSink(Sink):
    """Sink that hands every paced block to `consume(x)`."""

    def __init__(self, spec: StreamSpec, consume: Callable[[np.ndarray], None]) -> None:
        self.spec, self._consume = spec, consume
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def start(self, pull) -> None:
        self._stop.clear()

        def run():
            n, pace = self.spec.block, _Pacer(self.spec.rate, True)
            while not self._stop.is_set():
                self._consume(as_mono_f32(pull(n)))
                pace.tick(n, self._stop)
        self._th = threading.Thread(target=run, name="pump-sink", daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=2.0)


class FileConnector(Connector):
    name = "file"
    description = "WAV file playback / recording (path after 'file:')"

    def list_devices(self) -> list[DeviceInfo]:
        return []          # free-form path; nothing to enumerate

    def open_source(self, dev_id, spec): return FileSource(dev_id, spec)
    def open_sink(self, dev_id, spec): return FileSink(dev_id, spec)


class LoopConnector(Connector):
    name = "loop"
    description = "In-memory named loopback bus: sink 'loop:x' feeds source 'loop:x'"

    def list_devices(self) -> list[DeviceInfo]:
        with _BUSES_LOCK:
            names = sorted(_BUSES) or ["a"]
        return [DeviceInfo(self.name, n, f"loopback {n}", k)
                for n in names for k in (SOURCE, SINK)]

    def open_source(self, dev_id, spec):
        return _PumpSource(spec, _bus(dev_id or "a", spec.rate).get)

    def open_sink(self, dev_id, spec):
        return _PumpSink(spec, _bus(dev_id or "a", spec.rate).put)


class _Mix:
    """A shared medium: every node hears the SUM of what all the others transmit.

    One accumulation buffer per listener.  Each transmitter has a write position in it; a
    block is ADDED at that position (so overlapping transmissions really interfere), and a
    transmitter that was idle restarts at the listener's read position.  Sinks and sources
    are paced in real time, so write positions stay just ahead of the reads."""

    def __init__(self, rate: int) -> None:
        self.rate = rate
        self.lock = threading.Lock()
        self.nodes: set[str] = set()
        self.buf: dict[str, np.ndarray] = {}       # listener -> pending samples
        self.read: dict[str, int] = {}             # listener -> absolute read position
        self.wpos: dict[tuple[str, str], int] = {} # (tx, listener) -> absolute write position

    def join(self, node: str) -> None:
        with self.lock:
            self.nodes.add(node)
            self.buf.setdefault(node, np.zeros(0, np.float32))
            self.read.setdefault(node, 0)

    def put(self, tx: str, x: np.ndarray) -> None:
        with self.lock:
            for ln in self.nodes - {tx}:
                r = self.read[ln]
                w = max(self.wpos.get((tx, ln), r), r)
                off = w - r
                b = self.buf[ln]
                if len(b) < off + len(x):
                    b = np.concatenate([b, np.zeros(off + len(x) - len(b), np.float32)])
                b[off:off + len(x)] += x
                self.buf[ln] = b
                self.wpos[(tx, ln)] = w + len(x)

    def get(self, ln: str, n: int) -> np.ndarray:
        with self.lock:
            b = self.buf[ln]
            out = np.zeros(n, np.float32)
            k = min(n, len(b))
            out[:k] = b[:k]
            self.buf[ln] = b[k:]
            self.read[ln] += n
            return out


_MIXES: dict[str, _Mix] = {}


def _mix(dev_id: str, rate: int) -> tuple[_Mix, str]:
    net, _, node = (dev_id or "net/a").partition("/")
    with _BUSES_LOCK:
        m = _MIXES.setdefault(net, _Mix(rate))
    m.join(node or "a")
    return m, node or "a"


class MixConnector(Connector):
    name = "mix"
    description = ("Shared in-memory medium 'mix:<net>/<node>': the source hears the sum of every "
                   "OTHER node's sink on the same <net> (a many-node loopback)")

    def list_devices(self) -> list[DeviceInfo]:
        with _BUSES_LOCK:
            names = sorted(_MIXES) or ["net"]
        return [DeviceInfo(self.name, f"{n}/a", f"mix {n}", k) for n in names for k in (SOURCE, SINK)]

    def open_source(self, dev_id, spec):
        m, node = _mix(dev_id, spec.rate)
        return _PumpSource(spec, lambda n: m.get(node, n))

    def open_sink(self, dev_id, spec):
        m, node = _mix(dev_id, spec.rate)
        return _PumpSink(spec, lambda x: m.put(node, x))


class NullConnector(Connector):
    name = "null"
    description = "Silence in, discard out"

    def list_devices(self) -> list[DeviceInfo]:
        return [DeviceInfo(self.name, "", "null", k) for k in (SOURCE, SINK)]

    def open_source(self, dev_id, spec):
        return _PumpSource(spec, lambda n: np.zeros(n, np.float32))

    def open_sink(self, dev_id, spec):
        return _PumpSink(spec, lambda x: None)


CONNECTORS = [FileConnector, LoopConnector, MixConnector, NullConnector]
