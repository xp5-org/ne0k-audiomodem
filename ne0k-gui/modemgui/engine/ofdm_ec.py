# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Echo canceller for same-band full duplex (OfdmConfig.duplex = "ec"), V.34 style.

V.34 does not split the line into two bands: both modems use the whole band at once and each subtracts an
adaptive estimate of its OWN signal (the echo off the 2-wire/4-wire hybrid) from what it receives.  The canceller
is trained while the far end is quiet (V.34 phase 3); here that is our own BAT probe, sent while the peer listens.

Timebase: everything is indexed by the audio pipe's RX sample count.  ``AudioPipe.tx_onset_hooks`` hands every
played buffer over with the RX index at which its first sample reached the sink (``add_ref``); every captured
block carries its RX index (``Block.index``, ``add_raw``).  The echo of reference sample ``i`` therefore sits at RX
index ``i + delay`` -- one fixed bulk delay (sink -> line -> source) plus the hybrid's short impulse response.

    echo[n] = sum_k h[k] * ref[n - delay - k],   k = 0 .. taps-1

``train`` finds ``delay`` by cross-correlation and ``h`` by least squares (Toeplitz normal equations) over our own
probe; with no echo path above ``min_corr`` (a clean 4-wire line: a 2-party SIP call never returns our own audio)
the canceller stays OFF and passes audio through.  The taps and delay are FROZEN after training.  Not done yet:
tracking through data.  A plain re-fit under double talk is a trap here -- both ends' chunks carry the same
preambles and pilots, so the far signal correlates with our reference and a re-fit (or even a delay-only search)
jumped to false lags and wrecked the link (2026-11-13).  V.34 adapts decision-directed (the decoded far-end signal
subtracted first); that is the way to add it.
"""
from __future__ import annotations

import threading

import numpy as np


class _Ring:
    """Samples by absolute index, the last ``n`` kept; never-written / expired samples read as zeros."""

    def __init__(self, n: int) -> None:
        self.buf = np.zeros(n, np.float32)
        self.n = n
        self.hi = 0                                   # one past the highest index written

    def put(self, idx: int, x: np.ndarray) -> None:
        x = np.asarray(x, np.float32)
        if len(x) >= self.n:
            idx, x = idx + len(x) - self.n, x[-self.n:]
        lo = self.hi - self.n
        if idx + len(x) <= lo:
            return
        if idx < lo:
            x, idx = x[lo - idx:], lo
        if idx > self.hi:                             # a gap: clear what we skip (stale samples of a lap ago)
            self._write(self.hi, np.zeros(min(idx - self.hi, self.n), np.float32))
        self._write(idx, x)
        self.hi = max(self.hi, idx + len(x))

    def _write(self, idx: int, x: np.ndarray) -> None:
        p = idx % self.n
        k = min(len(x), self.n - p)
        self.buf[p:p + k] = x[:k]
        if k < len(x):
            self.buf[:len(x) - k] = x[k:]

    def get(self, i0: int, i1: int) -> np.ndarray:
        out = np.zeros(max(0, i1 - i0), np.float32)
        a, b = max(i0, self.hi - self.n), min(i1, self.hi)
        if b > a:
            p = a % self.n
            k = min(b - a, self.n - p)
            out[a - i0:a - i0 + k] = self.buf[p:p + k]
            if k < b - a:
                out[a - i0 + k:b - i0] = self.buf[:b - a - k]
        return out


class EchoCanceller:
    def __init__(self, fs: int, taps: int = 256, max_delay_s: float = 0.25, keep_s: float = 90.0,
                 min_corr: float = 0.1) -> None:
        self.fs, self.taps, self.min_corr = fs, int(taps), min_corr
        self.max_delay = int(max_delay_s * fs)
        self.ref, self.raw = _Ring(int(keep_s * fs)), _Ring(int(keep_s * fs))
        self.onsets: list[tuple[int, int]] = []       # (rx index, length) of every buffer we played
        self.h = None
        self.delay = 0
        self.on = False
        self.stats: dict = {}
        self._lock = threading.Lock()

    # -- feeds ---------------------------------------------------------------

    def add_ref(self, idx: int, samples) -> None:
        """AudioPipe.tx_onset_hooks: our own audio, on the RX timebase (sink thread: keep it cheap)."""
        x = np.asarray(samples, np.float32).reshape(-1)
        with self._lock:
            self.ref.put(int(idx), x)
            self.onsets.append((int(idx), len(x)))
            del self.onsets[:-64]

    def add_raw(self, idx: int, x: np.ndarray) -> None:
        with self._lock:
            self.raw.put(int(idx), x)

    # -- training --------------------------------------------------------------

    def train(self, i0: int | None = None, i1: int | None = None) -> dict:
        """Fit delay + taps over [i0, i1) of our own transmission (default: the longest buffer we played, i.e.
        the BAT probe).  Returns the stats dict (also kept in ``self.stats``)."""
        with self._lock:
            if i0 is None:
                if not self.onsets:
                    self.stats = {"on": False, "why": "nothing transmitted to train on"}
                    return self.stats
                i0, n = max(self.onsets, key=lambda o: o[1])
                i1 = i0 + n
            r = self.ref.get(i0 - self.taps, i1)                       # leading taps of context for the FIR
            y = self.raw.get(i0, i1 + self.max_delay)
        st = self._fit(r, y)
        self.stats = st
        return st

    def _solve(self, r: np.ndarray, y: np.ndarray, need_corr: bool = True):
        """(delay, h, corr, erle_db) for r[j] = ref at absolute i0 - taps + j, y[m] = raw at absolute i0 + m, under
        the model y[m] = sum_k h[k] * ref[i0 + m - delay - k]; (None, None, corr, why) when there is no path."""
        L, D = self.taps, self.max_delay
        r, y = r.astype(np.float64), y.astype(np.float64)
        er, ey = float(np.dot(r, r)), float(np.dot(y, y))
        if er <= 0 or ey <= 0:                                         # nothing sent / nothing came back at all
            return None, None, 0.0, "nothing transmitted" if er <= 0 else "no echo path"
        # bulk delay: c[s] = sum_m y[m] r[m + s]; absolute lag d pairs y[m] with ref[i0 + m - d] = r[m + L - d]
        nfft = 1 << int(np.ceil(np.log2(len(r) + len(y))))
        c = np.fft.irfft(np.conj(np.fft.rfft(y, nfft)) * np.fft.rfft(r, nfft), nfft)
        cc = c[(L - np.arange(D + 1)) % nfft]
        d = int(np.argmax(np.abs(cc)))
        corr = float(np.abs(cc[d]) / np.sqrt(er * ey))
        if need_corr and corr < self.min_corr:
            return None, None, corr, "no echo path"
        delay = d - min(L // 8, d)                                     # room for energy just before the peak
        # regressor u[t] = ref[i0 + t - (L - 1) - delay] = r[t + 1 - delay], so est = convolve(u, h, "valid")
        t = np.arange(len(y) + L - 1) + 1 - delay
        u = np.zeros(len(t))
        ok = (t >= 0) & (t < len(r))
        u[ok] = r[t[ok]]
        n2 = 1 << int(np.ceil(np.log2(len(u) + len(y))))
        U = np.fft.rfft(u, n2)
        acf = np.fft.irfft(U * np.conj(U), n2)[:L]                     # Toeplitz normal equations
        p = np.fft.irfft(np.conj(np.fft.rfft(y, n2)) * U, n2)[L - 1 - np.arange(L)]
        acf[0] = acf[0] * (1.0 + 1e-6) + 1e-12                         # light diagonal loading
        from scipy.linalg import solve_toeplitz
        h = solve_toeplitz(acf, p)
        res = y - np.convolve(u, h, mode="valid")[:len(y)]
        erle = 10 * np.log10(ey / max(float(np.dot(res, res)), 1e-30))
        return delay, h.astype(np.float32), corr, float(erle)

    def _fit(self, r: np.ndarray, y: np.ndarray) -> dict:
        delay, h, corr, erle = self._solve(r, y)
        if h is None:
            with self._lock:
                self.on, self.h = False, None
            return {"on": False, "why": erle, "corr": round(corr, 3)}
        with self._lock:
            self.delay, self.h = delay, h
            self.on = bool(erle > 3.0)
        st = {"on": self.on, "corr": round(corr, 3), "delay_ms": round(1000 * delay / self.fs, 2), "taps": self.taps,
              "erle_db": round(erle, 1)}
        if not self.on:
            st["why"] = f"training ERLE only {erle:.1f} dB"
        return st

    # -- running -----------------------------------------------------------------

    def cancel(self, idx: int, x: np.ndarray) -> np.ndarray:
        """``x`` (captured at RX index ``idx``) minus our own echo; ``x`` unchanged while off."""
        if not self.on or self.h is None:
            return x
        with self._lock:
            h, delay = self.h, self.delay
            r = self.ref.get(idx - delay - (len(h) - 1), idx - delay + len(x))
        if not np.any(r):
            return x
        return (np.asarray(x, np.float32) - np.convolve(r, h, mode="valid")[:len(x)]).astype(np.float32)
