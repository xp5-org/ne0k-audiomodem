# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""AudioPipe: one RX source + one TX sink, with fan-out to any number of taps.

What it adds on top of a bare connector:

  * **Taps.**  Each consumer (spectrograph, decoder, recorder) subscribes and gets
    its own bounded queue.  A slow consumer loses ITS oldest blocks (counted in
    ``Tap.dropped``); it can never stall the device callback or another consumer.
  * **History.**  The last ``history_s`` seconds of RX, addressable by absolute
    sample index, for scroll-back and for a decoder that wants to look behind a
    detected sync.
  * **TX queue.**  ``play()`` appends; the sink pulls; ``stop_tx()`` empties the
    queue first so silence is immediate (what is already inside the OS buffer is
    the connector's business -- see Sink.stop()).
  * **Own-TX flag.**  Every RX block says whether we were transmitting when it
    was captured (plus a guard for speaker-to-mic flight and latency).  Own
    transmissions arrive ~20 dB above anything from a peer, and a measurement of
    them is not a measurement of the channel, so consumers can drop or mute.
  * **Level meters.**  Per-block peak and clipped-sample count.  Clipping has
    several causes (own TX, ambient impulse, a hot input) that need opposite
    responses; the pipe reports the fact, and the ``tx_active`` flag lets the
    caller attribute it.
"""
from __future__ import annotations

import dataclasses
import queue
import threading
import time
from collections import deque

import numpy as np

from .base import AudioError, Sink, Source, StreamSpec, as_mono_f32
from .registry import Registry

CLIP_LEVEL = 0.999


@dataclasses.dataclass(frozen=True)
class Block:
    samples: np.ndarray      # float32 mono
    index: int               # absolute RX sample index of samples[0]
    tx_active: bool          # we were transmitting (or within the guard) at capture
    peak: float
    clipped: int             # samples at/over full scale


@dataclasses.dataclass
class Stats:
    rx_blocks: int = 0
    rx_samples: int = 0
    tx_samples: int = 0       # real (non-silence) samples handed to the sink
    clipped: int = 0
    peak: float = 0.0         # since last reset_peak()
    rx_error: str | None = None
    tx_error: str | None = None


class Tap:
    def __init__(self, maxblocks: int) -> None:
        self._q: queue.Queue[Block] = queue.Queue(maxsize=maxblocks)
        self.dropped = 0

    def _offer(self, b: Block) -> None:
        while True:
            try:
                self._q.put_nowait(b)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def get(self, timeout: float | None = None) -> Block | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list[Block]:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out


class AudioPipe:
    def __init__(self, registry: Registry | None = None, spec: StreamSpec | None = None,
                 history_s: float = 30.0, tx_guard_s: float = 0.3) -> None:
        self.registry = registry or Registry()
        self.spec = spec or StreamSpec()
        self.stats = Stats()
        self._src: Source | None = None
        self._snk: Sink | None = None
        self._taps: list[Tap] = []
        self._taps_lock = threading.Lock()
        self._hist: deque[Block] = deque()
        self._hist_n, self._hist_cap = 0, int(history_s * self.spec.rate)
        self._rx_index = 0
        self._tx: deque[np.ndarray] = deque()
        self._tx_lock = threading.Lock()
        self._tx_midflight = False
        self._tx_last = -1e9             # monotonic time the sink last got real audio
        self._tx_guard = tx_guard_s
        self._tx_idle = threading.Event()
        self._tx_idle.set()

    # -- lifecycle ----------------------------------------------------------

    def open(self, rx: str | None, tx: str | None) -> "AudioPipe":
        """Open both ends (either may be None) but do not start streaming."""
        if self._src or self._snk:
            raise AudioError("pipe already open")
        self._src = self.registry.open_source(rx, self.spec) if rx else None
        try:
            self._snk = self.registry.open_sink(tx, self.spec) if tx else None
        except Exception:
            self._src = None
            raise
        return self

    def start(self) -> None:
        try:
            if self._snk:
                self._snk.start(self._pull)
            if self._src:
                self._src.start(self._on_rx)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self.stop_tx()                    # silence first, then tear the stream down
        for end in (self._snk, self._src):
            if end:
                try:
                    end.stop()
                except Exception as e:      # keep closing the other end
                    self.stats.rx_error = self.stats.rx_error or f"close: {e}"
        self._src = self._snk = None

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    @property
    def rx_finished(self) -> bool:
        return bool(self._src and self._src.finished)

    # -- RX -----------------------------------------------------------------

    def subscribe(self, maxblocks: int = 256) -> Tap:
        t = Tap(maxblocks)
        with self._taps_lock:
            self._taps.append(t)
        return t

    def unsubscribe(self, tap: Tap) -> None:
        with self._taps_lock:
            if tap in self._taps:
                self._taps.remove(tap)

    def _tx_now(self) -> bool:
        return bool(self._tx) or self._tx_midflight or \
            (time.monotonic() - self._tx_last) < self._tx_guard

    def _on_rx(self, x: np.ndarray) -> None:
        x = as_mono_f32(x)
        a = np.abs(x)
        peak = float(a.max()) if len(a) else 0.0
        clipped = int(np.count_nonzero(a >= CLIP_LEVEL))
        b = Block(x, self._rx_index, self._tx_now(), peak, clipped)
        self._rx_index += len(x)
        s = self.stats
        s.rx_blocks += 1
        s.rx_samples += len(x)
        s.clipped += clipped
        s.peak = max(s.peak, peak)
        if self._src and self._src.error:
            s.rx_error = self._src.error
        self._hist.append(b)
        self._hist_n += len(x)
        while self._hist_n > self._hist_cap and len(self._hist) > 1:
            self._hist_n -= len(self._hist.popleft().samples)
        with self._taps_lock:
            taps = list(self._taps)
        for t in taps:
            t._offer(b)

    def history(self, n: int | None = None) -> tuple[np.ndarray, int]:
        """Most recent `n` RX samples (all kept if None) and the absolute index of
        the first one.  Contiguous: every block arrives in order."""
        blocks = list(self._hist)
        if not blocks:
            return np.zeros(0, np.float32), self._rx_index
        x = np.concatenate([b.samples for b in blocks])
        i0 = blocks[0].index
        if n is not None and n < len(x):
            i0 += len(x) - n
            x = x[-n:]
        return x, i0

    def reset_peak(self) -> float:
        p, self.stats.peak = self.stats.peak, 0.0
        return p

    # -- TX -----------------------------------------------------------------

    def play(self, samples: np.ndarray) -> None:
        """Queue audio for transmission.  Non-blocking; see wait_tx()."""
        if not self._snk:
            raise AudioError("no TX sink open")
        x = as_mono_f32(samples)
        if not len(x):
            return
        with self._tx_lock:
            self._tx.append(x)
            self._tx_midflight = True
            self._tx_idle.clear()

    def wait_tx(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been handed to the sink."""
        return self._tx_idle.wait(timeout)

    @property
    def tx_pending(self) -> int:
        with self._tx_lock:
            return sum(len(a) for a in self._tx)

    def stop_tx(self) -> None:
        with self._tx_lock:
            self._tx.clear()
            self._tx_midflight = False
            self._tx_idle.set()

    def _pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        got = 0
        with self._tx_lock:
            while got < n and self._tx:
                head = self._tx[0]
                take = min(n - got, len(head))
                out[got:got + take] = head[:take]
                got += take
                if take == len(head):
                    self._tx.popleft()
                else:
                    self._tx[0] = head[take:]
            if got:
                self._tx_last = time.monotonic()
                self.stats.tx_samples += got
            if got < n and self._tx_midflight and not self._tx:
                self._tx_midflight = False   # drained; the guard now covers the tail
                self._tx_idle.set()
        if self._snk and self._snk.error:
            self.stats.tx_error = self._snk.error
        return out
