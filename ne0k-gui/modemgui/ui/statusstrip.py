# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Top strip of the front page (css_modem_UI.xls): process CPU %, where it goes (decode / encode / CSS PLL
stages, then the UI, link/MAC, RX capture and audio I/O threads -- engine/cpustats.py), and buffer / memory.
The tooltip lists the busiest threads."""
from __future__ import annotations

import os
import time

from ..engine.cpustats import CORE, CpuMeter, LABELS, summary

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
        self.meter = CpuMeter()
        self._shown = set(CORE)          # groups on the strip; one that has used CPU joins and stays (no reflow)
        lay = QHBoxLayout(self); lay.setContentsMargins(4, 0, 4, 0)
        for w in (self.cpu, self.stages, self.buf):
            lay.addWidget(w)
            lay.addSpacing(18)
        lay.addStretch(1)
        self.dropped = 0
        self._timer = QTimer(self); self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self.refresh()

    def on_levels(self, lv) -> None:
        self.dropped = getattr(lv, "dropped", self.dropped)

    def refresh(self) -> None:
        smp = self.meter.sample()
        self._shown |= {g for g, v in smp["groups"].items() if v >= 0.5}
        total = f"{smp['total']:.0f}".rjust(3, "\u2007")       # figure space: digit-wide padding
        self.cpu.setText(f"CPU {total}%")
        self.stages.setText(summary(smp, self._shown))
        for lab in (self.cpu, self.stages):      # grow only: a narrower text must not pull the row left
            lab.setMinimumWidth(max(lab.minimumWidth(), lab.sizeHint().width()))
        self.stages.setToolTip(
            "CPU per stage and thread over the last second (100% = one core).  Decode / Encode / CSS PLL are the "
            "DSP calls, charged to whichever thread ran them; the other groups are what the rest of each thread "
            "did.\n\n" + "\n".join(f"{n}  [{LABELS[g]}]  {p:.1f}%" for n, g, p in smp["threads"][:15]))
        mem = rss_mb()
        q = getattr(self.session, "events", None)
        depth = f"{q.qsize()}/{q.maxsize}" if q is not None else "-"
        self.buf.setText(f"Buffer: events {depth}, dropped {self.dropped}  |  "
                         f"Memory: {'-' if mem is None else f'{mem:.0f} MB'}")

    def start(self) -> None:
        self._timer.start()
