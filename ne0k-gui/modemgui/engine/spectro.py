# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Streaming STFT.  Pure numpy; runs on the engine worker, never in a paint path."""
from __future__ import annotations

import numpy as np


class Stft:
    def __init__(self, rate: int, nfft: int = 2048, hop: int = 512) -> None:
        if hop > nfft:
            raise ValueError("hop > nfft")
        self.rate, self.nfft, self.hop = rate, nfft, hop
        self.window = np.hanning(nfft).astype(np.float32)
        self.freqs = np.fft.rfftfreq(nfft, 1.0 / rate)
        self._buf = np.zeros(0, np.float32)
        self._next = 0                 # absolute index of _buf[0]
        self._scale = 1.0 / (self.window.sum() / 2)

    def reset(self) -> None:
        self._buf = np.zeros(0, np.float32)
        self._next = 0

    def feed(self, x: np.ndarray, index: int | None = None
             ) -> tuple[int, np.ndarray]:
        """Add samples; return (index of first new column's window, (n, bins) dB).
        ``index`` is the absolute index of x[0]; a gap resets the stream."""
        if index is not None and len(self._buf) and index != self._next + len(self._buf):
            self.reset()
        if index is not None and not len(self._buf):
            self._next = index
        self._buf = np.concatenate([self._buf, x]) if len(self._buf) else x.astype(np.float32)
        n = (len(self._buf) - self.nfft) // self.hop + 1 if len(self._buf) >= self.nfft else 0
        first = self._next
        if n <= 0:
            return first, np.zeros((0, len(self.freqs)), np.float32)
        idx = np.arange(self.nfft)[None, :] + self.hop * np.arange(n)[:, None]
        spec = np.fft.rfft(self._buf[idx] * self.window, axis=1)
        db = 20 * np.log10(np.abs(spec) * self._scale + 1e-9)
        used = n * self.hop
        self._buf, self._next = self._buf[used:], self._next + used
        return first, db.astype(np.float32)
