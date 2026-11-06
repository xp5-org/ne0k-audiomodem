# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Input/output selection, open/close, loopback test and level meters."""
from __future__ import annotations

import threading

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QProgressBar, QPushButton, QVBoxLayout,
                               QWidget)

from ..audio import SINK, SOURCE
from ..engine import Levels, Session, Status

METER_FLOOR_DB = -80


class IoPanel(QWidget):
    message = Signal(str)
    _test_done = Signal(str)            # worker thread -> GUI thread

    def __init__(self, session: Session, settings: QSettings | None = None, parent=None):
        super().__init__(parent)
        self.session = session
        self.settings = settings or QSettings()
        self.rx = QComboBox(); self.tx = QComboBox()
        for c in (self.rx, self.tx):
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(24)
        self.refresh_btn = QPushButton("Refresh")
        self.open_btn = QPushButton("Open")
        self.test_btn = QPushButton("Test loopback")
        self.unavail = QLabel(); self.unavail.setWordWrap(True)
        self.unavail.setStyleSheet("color: gray")

        self.peak = QProgressBar(); self.peak.setRange(METER_FLOOR_DB, 0)
        self.peak.setTextVisible(False)
        self.peak_lbl = QLabel("-- dBFS")
        self.clip_lbl = QLabel("clip 0")
        self.tx_lbl = QLabel("TX"); self.tx_lbl.setAlignment(Qt.AlignCenter)
        self._tx_style(False)

        dev = QFormLayout()
        dev.addRow("Input (RX)", self.rx)
        dev.addRow("Output (TX)", self.tx)
        row = QHBoxLayout()
        for b in (self.refresh_btn, self.open_btn, self.test_btn):
            row.addWidget(b)
        meters = QHBoxLayout()
        meters.addWidget(self.peak, 1); meters.addWidget(self.peak_lbl)
        meters.addWidget(self.clip_lbl); meters.addWidget(self.tx_lbl)
        box = QGroupBox("Audio I/O")
        lay = QVBoxLayout(box)
        lay.addLayout(dev); lay.addWidget(self.unavail); lay.addLayout(row); lay.addLayout(meters)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.addWidget(box)

        self.refresh_btn.clicked.connect(self.refresh)
        self.open_btn.clicked.connect(self.toggle_open)
        self.test_btn.clicked.connect(self.run_test)
        self._test_done.connect(self._on_test_done)
        self.refresh()

    # -- devices ------------------------------------------------------------

    def refresh(self) -> None:
        """Repopulate from the registry.  Selection is restored by address, then by
        saved display name (an index like sd:3 moves when a USB device is replugged)."""
        saved = {"rx": self._saved("rx"), "tx": self._saved("tx")}
        for combo, kind, key in ((self.rx, SOURCE, "rx"), (self.tx, SINK, "tx")):
            want_addr = combo.currentData() or saved[key][0]
            want_name = saved[key][1]
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(none)", None)
            for d in self.session.devices(kind):
                label = f"{d.name}  [{d.address}]" + ("  (monitor)" if d.monitor else "")
                combo.addItem(label, d.address)
                combo.setItemData(combo.count() - 1, d.name, Qt.UserRole + 1)
            i = combo.findData(want_addr) if want_addr else -1
            if i < 0 and want_name:
                i = next((j for j in range(combo.count())
                          if combo.itemData(j, Qt.UserRole + 1) == want_name), -1)
            combo.setCurrentIndex(max(i, 0))
            combo.blockSignals(False)
        bad = [f"{n}: {why}" for n, (ok, why) in self.session.connector_status().items() if not ok]
        self.unavail.setText("Unavailable - " + "; ".join(bad) if bad else "")

    def select(self, rx: str | None, tx: str | None) -> None:
        """Show addresses opened from the command line; unknown ones get an entry."""
        for combo, addr in ((self.rx, rx), (self.tx, tx)):
            if not addr:
                continue
            i = combo.findData(addr)
            if i < 0:
                combo.addItem(f"{addr}  [{addr}]", addr)
                i = combo.count() - 1
            combo.setCurrentIndex(i)

    def describe(self, addr: str | None, kind: str) -> str:
        """``sd:2 [Device name]`` for the log; the bare address when the device is gone."""
        if not addr:
            return "none"
        for d in self.session.devices(kind):
            if d.address == addr:
                return f"{addr} [{d.name}]"
        return addr

    def _saved(self, key: str) -> tuple[str | None, str | None]:
        return (self.settings.value(f"io/{key}_addr"), self.settings.value(f"io/{key}_name"))

    def _save(self) -> None:
        for key, combo in (("rx", self.rx), ("tx", self.tx)):
            self.settings.setValue(f"io/{key}_addr", combo.currentData() or "")
            self.settings.setValue(f"io/{key}_name",
                                   combo.itemData(combo.currentIndex(), Qt.UserRole + 1) or "")

    # -- open / test --------------------------------------------------------

    def toggle_open(self) -> None:
        if self.session.is_open:
            self.session.close()
            return
        self._save()
        try:
            self.session.open(self.rx.currentData(), self.tx.currentData())
        except Exception as e:           # engine already emitted Status(error)
            self.message.emit(f"open failed: {e}")

    def run_test(self) -> None:
        rx, tx = self.rx.currentData(), self.tx.currentData()
        if not rx or not tx:
            self.message.emit("Pick both an input and an output to test.")
            return
        if self.session.is_open:
            self.session.close()
        self.test_btn.setEnabled(False)
        self.message.emit("Testing loopback...")

        def work():
            try:
                self._test_done.emit(self.session.loopback_test(tx, rx).summary())
            except Exception as e:
                self._test_done.emit(f"FAIL: {e}")
        threading.Thread(target=work, daemon=True, name="loopback-test").start()

    def _on_test_done(self, text: str) -> None:
        self.test_btn.setEnabled(True)
        self.message.emit(text)

    # -- engine events ------------------------------------------------------

    def on_status(self, st: Status) -> None:
        self.open_btn.setText("Close" if st.state == "open" else "Open")
        for w in (self.rx, self.tx, self.refresh_btn, self.test_btn):
            w.setEnabled(st.state != "open")
        if st.state != "open":
            self.peak.setValue(METER_FLOOR_DB); self.peak_lbl.setText("-- dBFS")
            self._tx_style(False)
        if st.state == "error":
            self.message.emit(f"audio error: {st.message}")

    def on_levels(self, lv: Levels) -> None:
        self.peak.setValue(int(max(METER_FLOOR_DB, lv.peak_db)))
        self.peak_lbl.setText(f"{lv.peak_db:6.1f} dBFS")
        self.clip_lbl.setText(f"clip {lv.clipped}")
        self.clip_lbl.setStyleSheet("color: red" if lv.clipped else "")
        self._tx_style(lv.tx_active)

    def _tx_style(self, on: bool) -> None:
        self.tx_lbl.setStyleSheet(
            "background:#c33;color:white;padding:2px 6px;border-radius:3px" if on
            else "background:#444;color:#888;padding:2px 6px;border-radius:3px")
