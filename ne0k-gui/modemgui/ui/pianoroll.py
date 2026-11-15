# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Pianoroll of link activity: one lane per node, airtime as bars, events as dots.

``LinkLog`` is the shared model (the Events tab and the spectrogram overlay both draw
from it); ``PianoRoll`` is the horizontally scrollable view.  Times are
time.monotonic() seconds as stamped by the engine."""
from __future__ import annotations

import collections
import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QScrollBar,
                               QVBoxLayout, QWidget)

from ..engine import LinkEvent

# kind -> (colour, label).  "lock" = decoder acquired the preamble (CSS has no separate PLL).
KINDS = {
    "tx":    ("#4c9be8", "TX"),
    "rx":    ("#3fb950", "RX"),
    "err":   ("#f85149", "ERR"),
    "lock":  ("#e3b341", "PLL lock"),
    "ack":   ("#39c5cf", "ACK"),
    "retry": ("#d070d0", "retry"),
    "fail":  ("#ff7b00", "FAIL"),
    "phase": ("#a371f7", "phase step"),
    "collision": ("#ff4fa3", "collision"),
    "echo":  ("#8b949e", "echo"),
}
BARS = ("tx", "rx")                       # kinds that have airtime
# Events arrive in time order give or take a burst: a tx is logged at its start with its end ahead, an rx at its end.
# Scanning back from the newest stops this far before the window (so a long burst that began earlier still shows).
SCAN_SLACK_S = 60.0
REDRAW_S = 0.25                           # views redraw at most this often, however fast events come


def legend_html() -> str:
    return "&nbsp;&nbsp;".join(f'<span style="color:{c}">&#9679;</span> {n}' for c, n in KINDS.values())


class LinkLog:
    def __init__(self, maxlen: int = 4000) -> None:
        self.events: "collections.deque[LinkEvent]" = collections.deque(maxlen=maxlen)

    def add(self, ev: LinkEvent) -> None:
        self.events.append(ev)

    def clear(self) -> None:
        self.events.clear()

    def nodes(self) -> list[int]:
        ids = {e.node for e in self.events} | {e.peer for e in self.events if e.peer >= 0}
        return sorted(ids)

    @staticmethod
    def sender(ev: LinkEvent) -> int:
        """Lane that owns a burst's airtime: we sent a tx, the peer sent what we received."""
        return ev.peer if ev.kind == "rx" and ev.peer >= 0 else ev.node

    def window(self, t_min: float, t_max: float) -> list:
        """Events overlapping [t_min, t_max], oldest first, found by scanning back from the newest: a redraw costs
        what is on screen, not the whole log.  A full duplex bulk stream logs ~15 events/s, and a whole-log redraw
        per event pegged the GUI thread, which starved the modem's threads of the GIL (2026-11-13)."""
        out = []
        for e in reversed(self.events):
            if e.t0 < t_min - SCAN_SLACK_S:
                break
            if t_min <= e.t1 and e.t0 <= t_max:
                out.append(e)
        out.reverse()
        return out

    def points(self, kinds, lanes: dict[int, int], t_min: float, t_max: float, evs=None):
        """(x_time, lane_index, event) for dot events inside [t_min, t_max]."""
        return [(e.t0 if e.kind != "rx" else e.t1, lanes[e.node], e)
                for e in (self.window(t_min, t_max) if evs is None else evs)
                if e.kind in kinds and e.node in lanes]

    def bars(self, lanes: dict[int, int], t_min: float, t_max: float, evs=None):
        return [(e, lanes[self.sender(e)]) for e in (self.window(t_min, t_max) if evs is None else evs)
                if e.kind in BARS and self.sender(e) in lanes]


_BRUSH: dict = {}
_PEN = None


def dots_and_bars(log: LinkLog, lanes, t_min, t_max, y_of, x_of):
    """Shared by both views.  Returns (spots for ScatterPlotItem, {kind: x/y arrays for bars})."""
    global _PEN
    if _PEN is None:
        _PEN = pg.mkPen("#000", width=0.5)
        _BRUSH.update({k: pg.mkBrush(c) for k, (c, _) in KINDS.items()})
    evs = log.window(t_min, t_max)
    spots = [dict(pos=(x_of(t), y_of(lane)), brush=_BRUSH[e.kind], pen=_PEN, size=9, data=e)
             for t, lane, e in log.points(KINDS, lanes, t_min, t_max, evs)]
    bars = {k: ([], []) for k in BARS}
    for e, lane in log.bars(lanes, t_min, t_max, evs):
        xs, ys = bars[e.kind]
        xs += [x_of(e.t0), x_of(e.t1)]; ys += [y_of(lane)] * 2
    return spots, bars


class PianoRoll(QWidget):
    def __init__(self, log: LinkLog | None = None, parent=None) -> None:
        super().__init__(parent)
        self.log = log or LinkLog()
        self.t_origin = time.monotonic()
        self._lanes: dict[int, int] = {}
        self._syncing = False
        self._active = False                   # audio open: only then does the live edge move
        self._t_stop = self.t_origin

        self.window = QDoubleSpinBox(); self.window.setRange(5, 3600); self.window.setValue(60)
        self.window.setSuffix(" s")
        self.follow = QCheckBox("Follow live"); self.follow.setChecked(True)
        legend = QLabel(legend_html()); legend.setTextFormat(Qt.RichText)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("window")); bar.addWidget(self.window); bar.addWidget(self.follow)
        bar.addStretch(1); bar.addWidget(legend)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "seconds since start")
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setMenuEnabled(False)
        self.plot.showAxis("left"); self.plot.getAxis("left").setWidth(60)
        self.dots = pg.ScatterPlotItem()
        self.plot.addItem(self.dots)
        self.curves = {}
        for k in BARS:
            c = pg.PlotCurveItem(connect="pairs", pen=pg.mkPen(KINDS[k][0], width=7))
            c.setOpacity(0.55); self.plot.addItem(c); self.curves[k] = c
        self.scroll = QScrollBar(Qt.Horizontal)
        self.summary = QLabel()
        lay = QVBoxLayout(self)
        lay.addLayout(bar); lay.addWidget(self.plot, 1); lay.addWidget(self.scroll)
        lay.addWidget(self.summary)

        self.plot.getViewBox().sigXRangeChanged.connect(self._on_range)
        self.scroll.valueChanged.connect(self._on_scroll)
        self.window.valueChanged.connect(lambda _=None: self.refresh())
        self.follow.toggled.connect(lambda _=None: self.refresh())
        self._dirty = False
        self._timer = QTimer(self); self._timer.setInterval(int(REDRAW_S * 1000))
        self._timer.timeout.connect(self._tick); self._timer.start()
        self.refresh()

    # -- model --------------------------------------------------------------

    def on_link(self, ev: LinkEvent) -> None:
        self.log.add(ev)
        self._dirty = True                    # drawn by the next tick, not per event

    def clear(self) -> None:
        self.log.clear(); self.t_origin = self._t_stop = time.monotonic(); self.refresh()

    # -- drawing ------------------------------------------------------------

    def _now(self) -> float:
        return (time.monotonic() if self._active else self._t_stop) - self.t_origin

    def set_active(self, on: bool) -> None:
        """Audio opened/closed.  While closed the live edge is frozen at the close time."""
        if on == self._active:
            return
        if not on:
            self._t_stop = time.monotonic()
        self._active = on
        self.refresh()

    def _tick(self) -> None:
        if self._dirty or (self._active and self.follow.isChecked() and self.isVisible()):
            self.refresh()

    def refresh(self) -> None:
        ids = self.log.nodes()
        self._lanes = {n: i for i, n in enumerate(ids)}
        ax = self.plot.getAxis("left")
        ax.setTicks([[(i, f"node {n}") for n, i in self._lanes.items()]])
        n = max(1, len(ids))
        self.plot.setYRange(-0.6, n - 0.4, padding=0)
        self._dirty = False
        o = self.t_origin
        w = self.window.value()
        if self.follow.isChecked():
            t_min, t_max = o + self._now() - 1.1 * w, o + self._now() + 0.2 * w
        else:                                 # what is in view, plus a screen either side for panning
            lo, hi = self.plot.getViewBox().viewRange()[0]
            t_min, t_max = o + lo - (hi - lo), o + hi + (hi - lo)
        spots, bars = dots_and_bars(self.log, self._lanes, t_min, t_max, float, lambda t: t - o)
        self.dots.setData(spots)
        for k, c in self.curves.items():
            xs, ys = bars[k]
            c.setData(np.array(xs, float), np.array(ys, float)) if xs else c.setData([], [])
        now = self._now()
        if self.follow.isChecked():
            self._syncing = True
            self.plot.setXRange(now - w + 0.1 * w, now + 0.1 * w, padding=0)
            self._syncing = False
        self._sync_scrollbar(now, w)
        counts = collections.Counter(e.kind for e in self.log.events)
        self.summary.setText("   ".join(f"{KINDS[k][1]}: {counts[k]}" for k in KINDS) +
                             f"   nodes: {', '.join(map(str, ids)) or '-'}")

    def _sync_scrollbar(self, now: float, w: float) -> None:
        lo, hi = self.plot.getViewBox().viewRange()[0]
        total = max(now + 0.1 * w, hi)
        self._syncing = True
        self.scroll.setRange(0, max(0, int((total - (hi - lo)) * 10)))
        self.scroll.setPageStep(int((hi - lo) * 10))
        self.scroll.setValue(int(max(0, lo) * 10))
        self._syncing = False

    def _on_range(self, *_a) -> None:
        if self._syncing:
            return
        self.follow.setChecked(False)         # user panned/zoomed: stop chasing the live edge
        self._dirty = True                    # draw what panned into view
        self._sync_scrollbar(self._now(), self.window.value())

    def _on_scroll(self, v: int) -> None:
        if self._syncing:
            return
        self.follow.setChecked(False)
        lo, hi = self.plot.getViewBox().viewRange()[0]
        self._syncing = True
        self.plot.setXRange(v / 10, v / 10 + (hi - lo), padding=0)
        self._syncing = False
