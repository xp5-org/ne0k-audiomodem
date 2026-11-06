# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Top strip of the front page (css_modem_UI.xls): process CPU %, the PLL / decode / encode
share, and buffer / memory. A value the engine does not report shows as "-"."""
from __future__ import annotations

import os
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def rss_mb() -> float | None:
    """Resident memory of this process in MB (Linux /proc; None elsewhere)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * _PAGE / 1048576
    except (OSError, ValueError, IndexError):
        return None


class StatusStrip(QWidget):
    def __init__(self, session=None, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.cpu = QLabel(); self.stages = QLabel(); self.buf = QLabel()
        self.stages.setToolTip("Share of CPU spent in PLL / decode / encode. The engine does not "
                               "time its stages yet, so this stays '-'.")
        lay = QHBoxLayout(self); lay.setContentsMargins(4, 0, 4, 0)
        for w in (self.cpu, self.stages, self.buf):
            lay.addWidget(w)
            lay.addSpacing(18)
        lay.addStretch(1)
        self._t = time.monotonic(); self._c = time.process_time()
        self.dropped = 0
        self._timer = QTimer(self); self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self.refresh()

    def on_levels(self, lv) -> None:
        self.dropped = getattr(lv, "dropped", self.dropped)

    def refresh(self) -> None:
        now, cpu = time.monotonic(), time.process_time()
        pct = 100.0 * (cpu - self._c) / max(now - self._t, 1e-6)
        self._t, self._c = now, cpu
        self.cpu.setText(f"CPU {pct:.0f}%")
        self.stages.setText("PLL / Decode / Encode: - / - / -")
        mem = rss_mb()
        q = getattr(self.session, "events", None)
        depth = f"{q.qsize()}/{q.maxsize}" if q is not None else "-"
        self.buf.setText(f"Buffer: events {depth}, dropped {self.dropped}  |  "
                         f"Memory: {'-' if mem is None else f'{mem:.0f} MB'}")

    def start(self) -> None:
        self._timer.start()
