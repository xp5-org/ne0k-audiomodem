"""Adaptive level carrier threshold for ``carrier_detect level``.

A fixed dBFS threshold sits on the room floor of a real speaker/mic pair (-55 dBFS fixed vs a -57 dBFS floor with
-51 dBFS RMS on the laptops, 2026-11-17): every 50 ms window read busy, so CSMA never saw the 2.5 s of idle DATA
needs and sends waited minutes for a quiet moment.  The threshold here is ``max(fixed, floor + margin)``.

The floor is a low percentile of the recent window levels.  It follows the room down at once and up slowly, and
much more slowly while a burst is open, so a long frame (minutes at 0.5 s CSS symbols) cannot lift the threshold
to its own level and cut itself off.  Weak signals below the margin are not lost to carrier sense: CSS
``channel_busy`` also asks the chirp tracker, which hears under the noise.
"""
from __future__ import annotations

import collections
import math

import numpy as np


class LevelFloor:
    def __init__(self, win_s: float, fixed_db: float, margin_db: float, hist_s: float = 5.0, pct: float = 10.0,
                 rise_db_s: float = 0.5, rise_busy_db_s: float = 0.02):
        self.win_s = win_s
        self.fixed_db = fixed_db
        self.margin_db = margin_db
        self.pct = pct
        self.rise = rise_db_s * win_s
        self.rise_busy = rise_busy_db_s * win_s
        self._hist: collections.deque = collections.deque(maxlen=max(4, int(round(hist_s / win_s))))
        self._warm = max(4, int(round(1.0 / win_s)))     # 1 s of room before the floor counts
        self.floor_db: float | None = None

    @staticmethod
    def db(rms: float) -> float:
        return 20.0 * math.log10(max(rms, 1e-10))

    def threshold_db(self) -> float:
        if self.margin_db <= 0 or self.floor_db is None:
            return self.fixed_db
        return max(self.fixed_db, self.floor_db + self.margin_db)

    def update(self, rms: float, in_burst: bool) -> float:
        """Feed one window's RMS (linear); returns the threshold (linear RMS) to judge THIS window by.  The
        threshold is the one from before the window, so a loud onset is judged against the room, not itself."""
        thr = 10 ** (self.threshold_db() / 20)
        if self.margin_db > 0:
            self._hist.append(self.db(rms))
            if len(self._hist) >= self._warm:
                target = float(np.percentile(self._hist, self.pct))
                if self.floor_db is None or target <= self.floor_db:
                    self.floor_db = target
                else:
                    self.floor_db = min(target, self.floor_db + (self.rise_busy if in_burst else self.rise))
        return thr
