# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Left column of the front page (css_modem_UI.xls): I/O type and modem, I/O settings, link
info and serial uart info, always visible beside the dockable displays."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QFormLayout, QGroupBox, QLabel, QScrollArea, QVBoxLayout, QWidget

from .io_panel import IoPanel
from .linkinfo_panel import LinkInfoPanel
from .uart_panel import UartPanel

IO_TYPES = (("audio", "Audio", True), ("gnuradio", "GNURadio", False), ("libsdr", "libsdr", False),
            ("local", "Local", False), ("network", "Network", False))


class ControlColumn(QScrollArea):

    def __init__(self, io: IoPanel, parent=None) -> None:
        super().__init__(parent)
        self.io = io
        self.link_info = LinkInfoPanel()
        self.uart = UartPanel()
        self.io_type = QComboBox()
        for key, label, ok in IO_TYPES:
            self.io_type.addItem(label if ok else f"{label} (not available yet)", key)
            if not ok:
                item = self.io_type.model().item(self.io_type.count() - 1)
                item.setEnabled(False)
        self.mode = QLabel()
        self.mode.setToolTip("Set under Settings > Modem Mode Settings.")
        form = QFormLayout()
        form.addRow("I/O Type", self.io_type)
        form.addRow("Mode", self.mode)
        top = QGroupBox("I/O Type")
        top.setLayout(form)
        inner = QWidget(); lay = QVBoxLayout(inner)
        for w in (top, io, self.link_info, self.uart):
            lay.addWidget(w)
        lay.addStretch(1)
        self.setWidget(inner); self.setWidgetResizable(True)

    def set_mode(self, text: str) -> None:
        self.mode.setText(text)

    def set_locked(self, locked: bool) -> None:
        """The I/O type can't change while audio is open."""
        self.io_type.setEnabled(not locked)
