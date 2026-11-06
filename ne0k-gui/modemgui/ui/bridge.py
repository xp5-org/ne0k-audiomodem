# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Engine -> Qt adapter: a timer on the GUI thread drains the session's event
queue and re-emits typed signals.  Polling (not cross-thread signals from the
audio worker) keeps the engine Qt-free and bounds GUI work per tick."""
from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal

from ..engine import Decoded, Levels, LinkEvent, OfdmData, ProbeReport, Session, Spectrum, Status
from ..engine.events import ConstellationFrame


class EngineBridge(QObject):
    status = Signal(object)
    levels = Signal(object)
    spectrum = Signal(object)
    constellation = Signal(object)
    decoded = Signal(object)
    link = Signal(object)
    probe = Signal(object)
    ofdm_data = Signal(object)

    def __init__(self, session: Session, interval_ms: int = 33, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.poll)
        self._timer.start()

    def poll(self) -> None:
        batch = self.session.drain()
        # coalesce: only the latest Levels matters; Spectrum batches must all be kept
        last_levels = None
        for ev in batch:
            if isinstance(ev, Levels):
                last_levels = ev
            elif isinstance(ev, Spectrum):
                self.spectrum.emit(ev)
            elif isinstance(ev, Status):
                self.status.emit(ev)
            elif isinstance(ev, Decoded):
                self.decoded.emit(ev)
            elif isinstance(ev, LinkEvent):
                self.link.emit(ev)
            elif isinstance(ev, ProbeReport):
                self.probe.emit(ev)
            elif isinstance(ev, OfdmData):
                self.ofdm_data.emit(ev)
            elif isinstance(ev, ConstellationFrame):
                self.constellation.emit(ev)
        if last_levels is not None:
            self.levels.emit(last_levels)
