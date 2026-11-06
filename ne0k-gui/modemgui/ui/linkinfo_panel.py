# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Link Info group of the front page: measured payload rates for both directions.

The engine does not report the PHY link rate or the on-air overhead yet, so those rows show
"-" until it does; the bitrates are measured here from the payload bytes that crossed the link
in the last WINDOW_S seconds."""
from __future__ import annotations

import time
from collections import deque

from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

WINDOW_S = 10.0


class LinkInfoPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rx: deque[tuple[float, int]] = deque()
        self._tx: deque[tuple[float, int]] = deque()
        self.rx_total = self.tx_total = 0
        self.rx_rate, self.rx_bitrate = QLabel("-"), QLabel("-")
        self.tx_rate, self.tx_bitrate = QLabel("-"), QLabel("-")
        self.overhead, self.detail = QLabel("-"), QLabel("")
        self.detail.setWordWrap(True)
        for w in (self.rx_rate, self.tx_rate, self.overhead):
            w.setToolTip("Not reported by the engine yet.")
        form = QFormLayout()
        form.addRow("RX Linkrate", self.rx_rate)
        form.addRow("RX Bitrate", self.rx_bitrate)
        form.addRow("TX Linkrate", self.tx_rate)
        form.addRow("TX Bitrate", self.tx_bitrate)
        form.addRow("% Overhead", self.overhead)
        box = QGroupBox("Link Info")
        lay = QVBoxLayout(box); lay.addLayout(form); lay.addWidget(self.detail)
        QVBoxLayout(self).addWidget(box)
        self.setContentsMargins(0, 0, 0, 0)

    def record_rx(self, nbytes: int) -> None:
        self.rx_total += nbytes
        self._rx.append((time.monotonic(), nbytes))

    def record_tx(self, nbytes: int) -> None:
        self.tx_total += nbytes
        self._tx.append((time.monotonic(), nbytes))

    def clear(self) -> None:
        self._rx.clear(); self._tx.clear()
        self.rx_total = self.tx_total = 0
        self.refresh("")

    @staticmethod
    def _rate(q: deque, now: float) -> float:
        while q and now - q[0][0] > WINDOW_S:
            q.popleft()
        return sum(n for _, n in q) * 8 / WINDOW_S

    def refresh(self, detail: str) -> None:
        now = time.monotonic()
        self.rx_bitrate.setText(f"{self._rate(self._rx, now):.0f} bit/s  ({self.rx_total} B)")
        self.tx_bitrate.setText(f"{self._rate(self._tx, now):.0f} bit/s  ({self.tx_total} B)")
        self.detail.setText(detail)
