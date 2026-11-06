# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Per-carrier bar graphs of the OFDM channel probe (the BAT exchange).

One chart per ProbeReport source/node, newest data replacing the old: the table THIS end
measured from the peer's probe appears as soon as that probe is decoded, and the table the
peer measured from OUR probe appears when its BAT report arrives.  Bar height is the SNR the
probe saw on that carrier and its colour the bits the allocator gave it (the original bat_chart
palette); a report with no SNR (the wire BAT carries bits only) is drawn as bits per carrier."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from ..engine import ProbeReport

BAT_COLORS = {-1: (240, 196, 25), 0: (201, 206, 214), 1: (109, 168, 224), 2: (47, 127, 208),
              4: (31, 157, 85), 6: (142, 68, 173)}
BAT_NAMES = {-1: "pilot", 0: "off", 1: "BPSK", 2: "QPSK", 4: "16-QAM", 6: "64-QAM"}
MAX_CHARTS = 8


def legend_html() -> str:
    return "&nbsp;&nbsp;".join(
        f'<span style="color:rgb{BAT_COLORS[b]}">&#9632;</span> {BAT_NAMES[b]}' for b in BAT_COLORS)


def title_of(r: ProbeReport) -> str:
    if r.source == "measured":
        return f"Probe from node {r.node}, measured by node {r.measured_by} (this end)"
    return f"Probe from node {r.node} (this end), as measured by node {r.measured_by}"


class ProbeChart(pg.PlotWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.setLabel("bottom", "carrier frequency", units="Hz")
        self.showGrid(x=False, y=True, alpha=0.25)
        self.setMouseEnabled(x=True, y=False)
        self.bars: pg.BarGraphItem | None = None
        self.has_snr = False

    def show_report(self, r: ProbeReport) -> None:
        freq, bits = np.asarray(r.freq_hz, float), np.asarray(r.bits, int)
        snr = np.asarray(r.snr_db, float)
        self.has_snr = bool(np.isfinite(snr).any())
        height = np.where(np.isfinite(snr), snr, 0.0) if self.has_snr else np.maximum(bits, 0).astype(float)
        self.setLabel("left", "SNR (dB)" if self.has_snr else "bits per carrier")
        if self.bars:
            self.removeItem(self.bars)
        width = (float(np.median(np.diff(freq))) if len(freq) > 1 else 10.0) * 0.9
        brushes = [pg.mkBrush(*BAT_COLORS.get(int(b), BAT_COLORS[0])) for b in bits]
        # a pilot has no bit count: give it a short bar so its position is still visible
        if not self.has_snr:
            height = np.where(bits < 0, 0.4, height)
        self.bars = pg.BarGraphItem(x=freq, height=height, width=width, brushes=brushes,
                                    pen=pg.mkPen(None))
        self.addItem(self.bars)
        if len(freq):
            self.setXRange(float(freq.min()), float(freq.max()), padding=0.02)


class ProbePanel(QWidget):
    """Scrollable stack of charts, one per (source, node)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.charts: dict[tuple, tuple[QLabel, ProbeChart]] = {}
        self.hint = QLabel("No probe yet.  The charts fill in as the BAT exchange proceeds: first "
                           "what this end measured, then what the peer measured of us.")
        self.hint.setWordWrap(True)
        self.legend = QLabel(legend_html())
        self.box = QVBoxLayout()
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.addWidget(self.legend); lay.addWidget(self.hint); lay.addLayout(self.box); lay.addStretch(1)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(inner)
        QVBoxLayout(self).addWidget(sc)

    def clear(self) -> None:
        for lbl, ch in self.charts.values():
            lbl.deleteLater(); ch.deleteLater()
        self.charts.clear()
        self.hint.show()

    def on_probe(self, r: ProbeReport) -> None:
        self.hint.hide()
        key = (r.source, r.node, r.measured_by)
        if key not in self.charts:
            if len(self.charts) >= MAX_CHARTS:
                old = next(iter(self.charts))
                lbl, ch = self.charts.pop(old); lbl.deleteLater(); ch.deleteLater()
            lbl = QLabel(); lbl.setStyleSheet("font-weight:bold"); lbl.setWordWrap(True)
            ch = ProbeChart()
            self.box.addWidget(lbl); self.box.addWidget(ch)
            self.charts[key] = (lbl, ch)
        lbl, ch = self.charts[key]
        used = int((np.asarray(r.bits) > 0).sum())
        lbl.setText(f"{title_of(r)}  -  {used} carriers in use" + (f"  -  {r.summary}" if r.summary else ""))
        ch.show_report(r)
