# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_syncdet.py

Soft preamble / Barker-code sync detector, split out of css_decoder.py.

SoftPreambleDetector keeps a sliding-window confidence history and
correlates it against the expected preamble pattern (e.g. Barker-13)
to declare sync acquisition.
"""

import numpy as np


class SoftPreambleDetector:
    """Sliding-window preamble correlator with a persistent confidence history."""

    def __init__(self, preamble, threshold=0.65, lookback=None,
                 conf_floor=0.0, cov_knee=0.0):
        self.n         = len(preamble)
        self.threshold = threshold
        self.expected  = np.array([1.0 if b else -1.0 for b in preamble],
                                  dtype=np.float32)
        self.lookback  = lookback if lookback is not None else self.n * 3
        self._history  = np.zeros(self.lookback, dtype=np.float32)
        self._pos      = 0
        self._filled   = 0
        self.best_lag  = 0
        self.best_window = None
        # Energy-aware scoring (0 disables either term; both 0 = plain cosine).
        #
        # The cosine score is scale-free: it judges the SHAPE of the window
        # and ignores how much signal is in it.  Two windows fool it:
        #   * a window of near-zero noise confidences, whose shape matches the
        #     Barker pattern by chance (0.7+ is common across a 100-lag ring);
        #   * signal onset: the first 5 training chirps land on Barker-13's
        #     alternating tail (1,0,1,0,1) with 8 empty slots before them, and
        #     score sqrt(5/13) = 0.62 -- above the 0.45 threshold floor.  Any
        #     lead-in longer than the positional guard (training+preamble
        #     symbols from the last resync) leaves only this score between
        #     the stream and a lock 11+ symbols early.
        # A real preamble fills every slot with a confident symbol, so the
        # score is multiplied by two terms that are 1.0 for it:
        #   energy   min(1, rms(window) / conf_floor)
        #   coverage min(1, PR / cov_knee), PR = (sum|c|)^2 / (n*sum c^2),
        #            1.0 for equal magnitudes, k/n when only k slots carry
        #            signal, 2/pi for Gaussian noise.
        self.conf_floor = float(conf_floor)
        self.cov_knee   = float(cov_knee)
        # Unmasked best window of the last push (diagnostics for the reject
        # logs): its score, lag, and why it was not eligible ("" = it was).
        self.raw_best_score = 0.0
        self.raw_best_lag   = 0
        self.raw_reason     = ""

    def alternation(self):
        """Fraction of adjacent pairs in the winning window that change sign.

        The training run-in alternates up/down chirps, so its confidences
        alternate sign every step and this reads ~1.0. Barker-13 contains
        runs (11111 00 11 0 1 0 1) and reads ~0.58. This distinguishes a
        genuine preamble match from a training-pattern sidelobe without
        relying on stream position, which drifts under clock offset.

        Returns None before any window has been scored.
        """
        w = self.best_window
        if w is None or len(w) < 2:
            return None
        sign = np.signbit(w)
        return float(np.count_nonzero(sign[1:] != sign[:-1]) / (len(w) - 1))

    def push(self, confidence, lag_max=None, early_lag=None, soft=False,
             trust_thr=None, alt_thr=None):
        """Push one confidence; return (acquired, score) of the best ELIGIBLE window.

        lag_max    windows more than lag_max symbols old are stale.
        early_lag  windows older than this end before the full preamble can
                   have arrived since the last resync; they stay eligible only
                   when `soft` and (score >= trust_thr or alternation < alt_thr).
        Ineligible windows are excluded from the argmax rather than rejected
        after it: a stale or early window that can never be acquired would
        otherwise hide every newer window below its score until it aged out
        of the ring (observed: one 0.75 noise window blocked a real preamble
        for 14 symbols).  With every argument None this is the original
        unmasked argmax.
        """
        self._history[self._pos] = float(confidence)
        self._pos    = (self._pos + 1) % self.lookback
        self._filled = min(self._filled + 1, self.lookback)
        if self._filled < self.n:
            return False, 0.0
        if self._filled < self.lookback:
            flat = self._history[:self._filled]          # already ordered
        else:
            flat = np.roll(self._history, -self._pos)    # rotate oldest to [0]
        windows   = np.lib.stride_tricks.sliding_window_view(flat, self.n)
        dots      = windows @ self.expected                        # (W,)
        win_norms = np.sqrt((windows * windows).sum(axis=1)) + 1e-9
        exp_norm  = float(np.sqrt(np.dot(self.expected, self.expected))) + 1e-9
        scores    = dots / (win_norms * exp_norm)                  # cosine in [-1,1]
        if self.conf_floor > 0.0 or self.cov_knee > 0.0:
            sq   = (windows * windows).astype(np.float64).sum(axis=1)
            gain = np.ones(len(scores))
            if self.conf_floor > 0.0:
                gain *= np.minimum(1.0, np.sqrt(sq / self.n) / self.conf_floor)
            if self.cov_knee > 0.0:
                asum = np.abs(windows).astype(np.float64).sum(axis=1)
                pr   = asum * asum / (self.n * sq + 1e-30)
                gain *= np.minimum(1.0, pr / self.cov_knee)
            scores = scores.astype(np.float64) * gain
        W    = len(scores)
        lags = W - 1 - np.arange(W)
        raw_idx = int(np.argmax(scores))
        self.raw_best_score = float(scores[raw_idx])
        self.raw_best_lag   = W - 1 - raw_idx
        ok = np.ones(W, dtype=bool)
        stale = np.zeros(W, dtype=bool)
        early = np.zeros(W, dtype=bool)
        if lag_max is not None:
            stale = lags > lag_max
            ok &= ~stale
        if early_lag is not None:
            early = lags > early_lag
            if early.any():
                trusted = np.zeros(W, dtype=bool)
                if soft:
                    sg  = np.signbit(windows)
                    alt = (sg[:, 1:] != sg[:, :-1]).sum(axis=1) / (self.n - 1)
                    trusted = (scores >= trust_thr) | ~(alt >= alt_thr)
                ok &= ~early | trusted
        self.raw_reason = ("" if ok[raw_idx] else
                           "stale" if stale[raw_idx] else "early")
        if not ok.any():
            self.best_lag    = self.raw_best_lag
            self.best_window = np.array(windows[raw_idx], dtype=np.float32)
            return False, 0.0
        masked     = np.where(ok, scores, -np.inf)
        best_idx   = int(np.argmax(masked))
        best_score = float(scores[best_idx])
        self.best_lag = W - 1 - best_idx
        self.best_window = np.array(windows[best_idx], dtype=np.float32)
        return best_score >= self.threshold, best_score

    def reset(self):
        pass  # soft reset — history persists

    def clear_history(self):
        self._history[:] = 0.0
        self._pos        = 0
        self._filled     = 0
        self.best_lag    = 0
        self.best_window = None
