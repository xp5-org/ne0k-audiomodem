# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Serial uart info group of the front page: the virtual serial port's mode, buffer and throughput."""
from __future__ import annotations

import time

from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QToolButton, QVBoxLayout, QWidget


class UartPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.state, self.mode, self.buf = QLabel("closed"), QLabel("-"), QLabel("-")
        self.rx, self.tx, self.retries = QLabel("-"), QLabel("-"), QLabel("-")
        form = QFormLayout()
        for name, w in (("Port", self.state), ("Mode", self.mode), ("Buffer", self.buf),
                        ("RX", self.rx), ("TX", self.tx), ("Retries / fails", self.retries)):
            form.addRow(name, w)
        box = QGroupBox("Serial uart info")
        self.port_btn = QToolButton(); self.port_btn.setText("Open virtual serial port")
        lay = QVBoxLayout(box); lay.addLayout(form); lay.addWidget(self.port_btn)
        QVBoxLayout(self).addWidget(box)
        self._last: tuple[float, int, int] | None = None

    def set_action(self, action) -> None:
        """The checkable open/close action that owns the port (keeps its shortcut and tooltip)."""
        self.port_btn.setDefaultAction(action)

    def refresh(self, br) -> None:
        """``br`` is the engine SerialBridge, or None when the port is closed."""
        if br is None:
            self._last = None
            self.state.setText("closed")
            for w in (self.mode, self.buf, self.rx, self.tx, self.retries):
                w.setText("-")
            return
        now = time.monotonic()
        rx_bps = tx_bps = 0.0
        if self._last:
            dt = max(now - self._last[0], 1e-3)
            rx_bps, tx_bps = (br.rx_count - self._last[1]) / dt, (br.tx_count - self._last[2]) / dt
        self._last = (now, br.rx_count, br.tx_count)
        self.state.setText(br.link_path)
        self.mode.setText(str(br.mode))
        self.buf.setText(f"{br.buf_size} B")
        self.rx.setText(f"{br.rx_count} B  ({rx_bps:.0f} B/s)")
        self.tx.setText(f"{br.tx_count} B  ({tx_bps:.0f} B/s)")
        self.retries.setText(f"{br.retry_count} / {br.fail_count}")
