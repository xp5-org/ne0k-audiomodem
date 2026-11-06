# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Session: the engine's single entry point.

Owns the AudioPipe and one worker thread that drains a tap and turns audio into
events (levels, spectrum).  Thread-safe to call from the UI thread; the UI never
touches the pipe directly.  Modem sessions (OFDM/CSS decode, TX) attach here in
later phases as additional consumers of the same tap.
"""
from __future__ import annotations

import math
import queue
import threading
import time

import numpy as np

from ..audio import AudioError, AudioPipe, DeviceInfo, Registry, StreamSpec
from .ident import Identifier
from .events import Event, Levels, Spectrum, Status
from .selftest import LoopbackResult, loopback
from .spectro import Stft

LEVEL_HZ = 20.0


def _db(x: float) -> float:
    return 20 * math.log10(x) if x > 1e-9 else -180.0


class Session:
    def __init__(self, registry: Registry | None = None, rate: int = 48000,
                 block: int = 1024, nfft: int = 2048, hop: int = 512) -> None:
        self.registry = registry or Registry()
        self.spec = StreamSpec(rate, block)
        self.events: "queue.Queue[Event]" = queue.Queue(maxsize=512)
        self._stft_args = (nfft, hop)
        self._pipe: AudioPipe | None = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.ident = Identifier(self)       # ham-mode station ID; inert until HamConfig.enabled
        self.tx_hooks: list = []            # f(samples, rx_sample_index): called for every play()

    # -- events -------------------------------------------------------------

    def _emit(self, ev: Event) -> None:
        while True:
            try:
                self.events.put_nowait(ev)
                return
            except queue.Full:                  # UI stalled: drop oldest, never block audio
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    pass

    def _emit_raw_error(self, msg: str) -> None:
        self._emit(Status("error", msg))

    def drain(self) -> list[Event]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    # -- devices ------------------------------------------------------------

    def devices(self, kind: str | None = None) -> list[DeviceInfo]:
        return self.registry.list_devices(kind)

    def connector_status(self) -> dict[str, tuple[bool, str]]:
        return self.registry.status()

    # -- lifecycle ----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._pipe is not None

    @property
    def pipe(self) -> AudioPipe | None:
        return self._pipe

    def open(self, rx: str | None, tx: str | None) -> None:
        with self._lock:
            if self._pipe:
                self._close_locked()
            pipe = AudioPipe(self.registry, self.spec)
            try:
                pipe.open(rx, tx).start()
            except AudioError as e:
                self._emit(Status("error", str(e), rx, tx))
                raise
            self._pipe = pipe
            self._stop.clear()
            self._worker = threading.Thread(target=self._run, args=(pipe, pipe.subscribe(256)),
                                            name="modemgui-session", daemon=True)
            self._worker.start()
            self.ident.reset(); self.ident.start()
            self._emit(Status("open", "", rx, tx))

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if not self._pipe:
            return
        self._stop.set()
        self.ident.stop()
        if self._worker:
            self._worker.join(2.0)
        self._pipe.close()
        self._pipe = self._worker = None
        self._emit(Status("closed"))

    # -- TX -----------------------------------------------------------------

    def play(self, samples: np.ndarray, ident: bool = True) -> None:
        """Queue a modem frame for TX, with the station ID around it when one is due.  ``ident=False``
        for timed protocol slots that cannot take extra audio (the caller sends the ID itself)."""
        if not self._pipe:
            raise AudioError("not open")
        self.play_raw(self.ident.wrap(samples, inline=ident))

    def play_raw(self, samples: np.ndarray) -> None:
        if not self._pipe:
            raise AudioError("not open")
        for h in list(self.tx_hooks):
            try:
                h(samples, self._pipe.stats.rx_samples)
            except Exception:               # a diagnostics hook must never break TX
                pass
        self._pipe.play(samples)

    def stop_tx(self) -> None:
        if self._pipe:
            self._pipe.stop_tx()

    # -- checks -------------------------------------------------------------

    def loopback_test(self, tx: str, rx: str) -> LoopbackResult:
        """Blocking (~3 s); call from a worker thread.  Needs its own pipe, so it
        fails on exclusive-access devices that are already open in this session."""
        return loopback(self.registry, tx, rx, self.spec.rate, self.spec.block)

    # -- worker -------------------------------------------------------------

    def _run(self, pipe: AudioPipe, tap) -> None:
        stft = Stft(self.spec.rate, *self._stft_args)
        acc = {"pk": 0.0, "ss": 0.0, "n": 0}
        pend_cols: list[np.ndarray] = []
        pend_tx: list[np.ndarray] = []
        pend_index: list[int] = []
        tx_now = False

        def flush() -> None:
            # Called on the LEVEL_HZ cadence AND whenever the tap goes idle, so the
            # tail of a burst (or a short file) is never stranded in the buffers.
            if pend_cols:
                self._emit(Spectrum(pend_index[0], np.concatenate(pend_cols),
                                    np.concatenate(pend_tx), stft.freqs))
                pend_cols.clear(); pend_tx.clear(); pend_index.clear()
            if acc["n"]:
                s = pipe.stats
                self._emit(Levels(_db(acc["pk"]), _db(math.sqrt(acc["ss"] / acc["n"])),
                                  s.clipped, tx_now, s.rx_samples, tap.dropped))
                acc.update(pk=0.0, ss=0.0, n=0)

        last = time.monotonic()
        while not self._stop.is_set():
            b = tap.get(timeout=0.1)
            if b is None:
                flush()
                if pipe.stats.rx_error:
                    self._emit(Status("error", pipe.stats.rx_error))
                    return
                continue
            tx_now = b.tx_active
            acc["pk"] = max(acc["pk"], b.peak)
            acc["ss"] += float(np.dot(b.samples, b.samples))
            acc["n"] += len(b.samples)
            idx, cols = stft.feed(b.samples, b.index)
            if len(cols):
                if not pend_index:
                    pend_index.append(idx)
                pend_cols.append(cols)
                pend_tx.append(np.full(len(cols), b.tx_active))
            now = time.monotonic()
            if now - last >= 1.0 / LEVEL_HZ:
                last = now
                flush()
