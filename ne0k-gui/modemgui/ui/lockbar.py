# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Receiver lock / timing strip shared by the CSS and OFDM panes: a lock indicator, the time
the receiver took to acquire the last burst, and a bar for how well the preamble aligned."""
from __future__ import annotations

import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

LOCK_HOLD_S = 8.0                          # "locked" fades to "idle" this long after the last lock


class LockBar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.state = QLabel(); self.timing = QLabel()
        self.align = QProgressBar(); self.align.setRange(0, 100); self.align.setFormat("align %p%")
        self.align.setMaximumWidth(220)
        self.align.setToolTip("Preamble correlation score at the last lock (100% = perfectly aligned).")
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        for w in (self.state, self.timing, self.align):
            lay.addWidget(w)
        lay.addStretch(1)
        self._open = False
        self._t_lock: float | None = None
        self._acq: list[float] = []
        self._timer = QTimer(self); self._timer.setInterval(250)
        self._timer.timeout.connect(self._paint)
        self.reset()

    def reset(self) -> None:
        self._t_lock = None; self._acq.clear()
        self.align.setValue(0)
        self._paint()

    def set_open(self, on: bool) -> None:
        self._open = on
        self._timer.start() if on else self._timer.stop()
        self._paint()

    def locked(self, acq_s: float | None = None, score: float | None = None) -> None:
        self._t_lock = time.monotonic()
        if acq_s is not None:
            self._acq = (self._acq + [acq_s])[-20:]
        if score is not None:
            self.align.setValue(int(max(0.0, min(1.0, score)) * 100))
        self._paint()

    def _paint(self) -> None:
        if not self._open:
            self.state.setText("PLL: audio closed"); self.state.setStyleSheet("color:gray")
        elif self._t_lock is not None and time.monotonic() - self._t_lock < LOCK_HOLD_S:
            self.state.setText("PLL: LOCKED"); self.state.setStyleSheet("color:#3fb950;font-weight:bold")
        else:
            self.state.setText("PLL: searching"); self.state.setStyleSheet("color:#e3b341;font-weight:bold")
        if self._acq:
            avg = sum(self._acq) / len(self._acq)
            self.timing.setText(f"time to acquire: last {self._acq[-1]:.2f} s, avg {avg:.2f} s "
                                f"({len(self._acq)})")
        else:
            self.timing.setText("time to acquire: -")
