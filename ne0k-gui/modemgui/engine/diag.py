# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Troubleshooting capture for sending back to the developers.  Qt-free.

``Diagnostics`` keeps an in-memory log ring always, and -- when asked -- writes a debug
log file and WAV recordings of the audio we received and transmitted.  The two WAVs share a
timeline (the TX file is padded with silence to the RX sample index at the time of each
play), so they can be laid on top of each other in an editor.  ``bundle`` zips the lot with the
settings and the C-library inventory.
"""
from __future__ import annotations

import collections
import datetime
import json
import platform
import sys
import threading
import time
import wave
import zipfile
from pathlib import Path

import numpy as np

from .modemcfg import DiagConfig


class _WavOut:
    def __init__(self, path: Path, rate: int) -> None:
        self.path, self.n = path, 0
        self._w = wave.open(str(path), "wb")
        self._w.setnchannels(1); self._w.setsampwidth(2); self._w.setframerate(rate)

    def write(self, x: np.ndarray) -> None:
        pcm = (np.clip(np.asarray(x, np.float32), -1, 1) * 32767).astype("<i2")
        self._w.writeframes(pcm.tobytes())
        self.n += len(pcm)

    def pad_to(self, n: int) -> None:
        while self.n < n:
            k = min(n - self.n, 1 << 16)
            self.write(np.zeros(k, np.float32))

    def close(self) -> None:
        self._w.close()


class Diagnostics:
    def __init__(self, cfg: DiagConfig, header: dict | None = None, ring: int = 4000) -> None:
        self.cfg, self.header = cfg, header or {}
        self.ring: collections.deque = collections.deque(maxlen=ring)
        self.dir: Path | None = None
        self._log_f = None
        self._rx: _WavOut | None = None
        self._tx: _WavOut | None = None
        self._tap = self._thread = None
        self._session = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.on_line = None              # f(str): live log line (the UI mirrors it)

    # -- log ---------------------------------------------------------------

    def log(self, msg: str) -> None:
        line = f"{datetime.datetime.now():%H:%M:%S.%f}"[:-3] + " " + msg
        with self._lock:
            self.ring.append(line)
            if self._log_f:
                self._log_f.write(line + "\n"); self._log_f.flush()
        if self.on_line:
            self.on_line(line)

    def _ensure_dir(self) -> Path:
        if self.dir is None:
            self.dir = Path(self.cfg.out_dir).expanduser() / datetime.datetime.now().strftime(
                "session_%Y%m%d_%H%M%S")
            self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir

    def start_log(self) -> Path | None:
        """Open the debug log file (idempotent) and dump what a developer needs first."""
        if not self.cfg.debug_log or self._log_f:
            return None
        path = self._ensure_dir() / "debug.log"
        with self._lock:
            self._log_f = open(path, "a", encoding="utf-8")
            for line in self.ring:                      # lines logged before it was switched on
                self._log_f.write(line + "\n")
        self.log(f"debug log -> {path}")
        self.log(f"python {sys.version.split()[0]} on {platform.platform()}")
        for k, v in self.header.items():
            self.log(f"{k}: {v}")
        return path

    def stop_log(self) -> None:
        with self._lock:
            if self._log_f:
                self._log_f.close(); self._log_f = None

    # -- audio -------------------------------------------------------------

    def attach(self, session) -> None:
        """Start the recordings that are switched on; call after the session opened."""
        self.detach()
        self._session = session
        pipe = session.pipe
        if not pipe:
            return
        rate = session.spec.rate
        if self.cfg.wav_rx or self.cfg.wav_tx:
            d = self._ensure_dir()
            stamp = time.strftime("%H%M%S")
            if self.cfg.wav_rx:
                self._rx = _WavOut(d / f"rx_{stamp}.wav", rate)
                self._stop.clear()
                self._tap = pipe.subscribe(4096)
                self._thread = threading.Thread(target=self._pump, name="diag-rx", daemon=True)
                self._thread.start()
                self.log(f"recording RX audio -> {self._rx.path} ({rate} Hz mono)")
            if self.cfg.wav_tx:
                self._tx = _WavOut(d / f"tx_{stamp}.wav", rate)
                session.tx_hooks.append(self._on_play)
                self.log(f"recording TX audio -> {self._tx.path}")

    def _pump(self) -> None:
        while not self._stop.is_set():
            b = self._tap.get(timeout=0.2)
            if b is not None and self._rx:
                self._rx.write(b.samples)

    def _on_play(self, samples, rx_index: int) -> None:
        if self._tx:
            self._tx.pad_to(max(rx_index, self._tx.n))
            self._tx.write(samples)

    def detach(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2.0)
        s = self._session
        if s is not None:
            if self._on_play in s.tx_hooks:
                s.tx_hooks.remove(self._on_play)
            if self._tap and s.pipe:
                s.pipe.unsubscribe(self._tap)
        for w in (self._rx, self._tx):
            if w:
                w.close()
                self.log(f"wrote {w.path} ({w.n / max(1, s.spec.rate if s else 48000):.1f} s)")
        self._rx = self._tx = self._tap = self._thread = self._session = None

    def close(self) -> None:
        self.detach()
        with self._lock:
            if self._log_f:
                self._log_f.close(); self._log_f = None

    # -- bundle ------------------------------------------------------------

    def bundle(self, path: str | Path, extra: dict[str, str] | None = None) -> Path:
        """Zip the session folder (logs, WAVs), the in-memory log ring and ``extra`` text files
        (settings, library inventory).  Close audio first for complete WAV headers: a WAV still
        being recorded has no length field yet, though most editors still open it."""
        path = Path(path)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            if self.dir and self.dir.exists():
                for f in sorted(self.dir.iterdir()):
                    if f.is_file():
                        z.write(f, f.name)
            if not (self.dir and (self.dir / "debug.log").exists()):
                with self._lock:                     # file log was off: ship the in-memory ring
                    z.writestr("debug.log", "\n".join(self.ring) + "\n")
            z.writestr("header.json", json.dumps(self.header, indent=2, default=str))
            for name, text in (extra or {}).items():
                z.writestr(name, text)
        return path
