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

    def __init__(self, preamble, threshold=0.65, lookback=None):
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

    def push(self, confidence):
        self._history[self._pos] = float(confidence)
        self._pos    = (self._pos + 1) % self.lookback
        self._filled = min(self._filled + 1, self.lookback)

        if self._filled < self.n:
            return False, 0.0

        # Vectorised lag search: O(filled) numpy instead of O(lookback x n)
        # Python. sliding_window_view creates a (W, n) view with zero copies;
        # the dot + norm are one BLAS call each.
        #
        # Linearise the ring oldest->newest so flat[-1] is the most recent
        # push, flat[-n:] is the lag-0 window.
        if self._filled < self.lookback:
            flat = self._history[:self._filled]          # already ordered
        else:
            flat = np.roll(self._history, -self._pos)    # rotate oldest to [0]

        windows   = np.lib.stride_tricks.sliding_window_view(flat, self.n)
        # (W, n): windows[0] = oldest window (highest lag),
        #         windows[-1] = most recent window (lag 0)

        # Cosine similarity (L2-normalised dot product): robust to outlier
        # magnitudes that inflate mean_abs under L1 normalization.
        # Score is in [-1, 1].
        dots      = windows @ self.expected                        # (W,)
        win_norms = np.sqrt((windows * windows).sum(axis=1)) + 1e-9
        exp_norm  = float(np.sqrt(np.dot(self.expected, self.expected))) + 1e-9
        scores    = dots / (win_norms * exp_norm)                  # cosine in [-1,1]

        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        # Convert array index to lag: index 0 is oldest (lag = W-1), index W-1 is lag 0
        self.best_lag = len(scores) - 1 - best_idx
        # Caller needs the winning window itself, not just its score, to
        # distinguish a Barker-shaped match from a training-pattern sidelobe.
        # Copied because `flat` may be a view onto the ring buffer, which the
        # next push overwrites.
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
