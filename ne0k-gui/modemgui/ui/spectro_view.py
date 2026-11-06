# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Scrolling spectrogram (pyqtgraph ImageItem).  The UI only draws columns the
engine already computed."""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSpinBox,
                               QVBoxLayout, QWidget)

from ..engine import Spectrum
from .pianoroll import BARS, KINDS, LinkLog, dots_and_bars


class SpectroView(QWidget):
    def __init__(self, columns: int = 600, log: LinkLog | None = None, parent=None) -> None:
        super().__init__(parent)
        self.columns = columns
        self.log = log or LinkLog()
        self.col_dt = 512 / 48000          # seconds per column (Session hop / rate)
        self._t_last = time.monotonic()    # arrival time of the newest column
        self._freqs: np.ndarray | None = None
        self._buf: np.ndarray | None = None        # (columns, bins), newest at the end
        self.floor = QDoubleSpinBox(); self.floor.setRange(-200, 0); self.floor.setValue(-110)
        self.ceil = QDoubleSpinBox(); self.ceil.setRange(-100, 20); self.ceil.setValue(-20)
        self.maxhz = QSpinBox(); self.maxhz.setRange(500, 24000); self.maxhz.setValue(6000)
        self.maxhz.setSingleStep(500)
        bar = QHBoxLayout()
        for lbl, w in (("floor dB", self.floor), ("ceil dB", self.ceil), ("max Hz", self.maxhz)):
            bar.addWidget(QLabel(lbl)); bar.addWidget(w)
        self.overlay = QCheckBox("Event overlay")
        self.overlay.setToolTip("Draw node TX/RX/ERR/PLL-lock/ACK events over the spectrogram")
        bar.addWidget(self.overlay)
        bar.addStretch(1)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "Hz"); self.plot.setLabel("bottom", "columns (newest right)")
        self.img = pg.ImageItem(axisOrder="col-major")
        self.img.setColorMap(pg.colormap.get("viridis"))
        self.plot.addItem(self.img)
        self.ov_dots = pg.ScatterPlotItem(); self.ov_dots.setZValue(10)
        self.plot.addItem(self.ov_dots)
        self.ov_bars = {}
        for k in BARS:
            c = pg.PlotCurveItem(connect="pairs", pen=pg.mkPen(KINDS[k][0], width=4))
            c.setZValue(9); self.plot.addItem(c); self.ov_bars[k] = c
        lay = QVBoxLayout(self); lay.addLayout(bar); lay.addWidget(self.plot, 1)
        for w in (self.floor, self.ceil, self.maxhz):
            w.valueChanged.connect(self._redraw)
        self.overlay.toggled.connect(lambda _=None: self._draw_overlay())

    def clear(self) -> None:
        self._buf = None
        self.img.clear()

    def on_spectrum(self, sp: Spectrum) -> None:
        if self._buf is None or self._freqs is None or len(self._freqs) != len(sp.freqs):
            self._freqs = sp.freqs
            self._buf = np.full((self.columns, len(sp.freqs)), self.floor.value(), np.float32)
        cols = sp.cols[-self.columns:]
        self._t_last = time.monotonic()
        self._buf = np.roll(self._buf, -len(cols), axis=0)
        self._buf[-len(cols):] = cols
        self._redraw()

    def _redraw(self) -> None:
        if self._buf is None or self._freqs is None:
            return
        nbin = int(np.searchsorted(self._freqs, self.maxhz.value())) or len(self._freqs)
        self.img.setImage(self._buf[:, :nbin], autoLevels=False,
                          levels=(self.floor.value(), self.ceil.value()))
        df = self._freqs[1] - self._freqs[0]
        self.img.setRect(0, 0, self.columns, nbin * df)
        self._top = nbin * df
        self._draw_overlay()

    # -- event overlay ------------------------------------------------------

    _top = 6000.0

    def _draw_overlay(self) -> None:
        on = self.overlay.isChecked()
        self.ov_dots.setVisible(on)
        for c in self.ov_bars.values():
            c.setVisible(on)
        if not on:
            return
        span = self.columns * self.col_dt
        t_max, t_min = self._t_last, self._t_last - span
        lanes = {n: i for i, n in enumerate(self.log.nodes())}
        top = self._top
        step = min(0.06 * top, top / (len(lanes) + 1)) if lanes else 0
        x_of = lambda t: self.columns - 1 - (self._t_last - t) / self.col_dt
        y_of = lambda lane: top - (lane + 0.7) * step
        spots, bars = dots_and_bars(self.log, lanes, t_min, t_max, y_of, x_of)
        self.ov_dots.setData(spots)
        for k, c in self.ov_bars.items():
            xs, ys = bars[k]
            c.setData(np.array(xs, float), np.array(ys, float)) if xs else c.setData([], [])

    def on_link(self, ev=None) -> None:
        self._draw_overlay()
