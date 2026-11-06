# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class Placeholder(QWidget):
    """Marks a view that is planned but not built; see PLAN.md for its phase."""

    def __init__(self, title: str, note: str, parent=None) -> None:
        super().__init__(parent)
        lbl = QLabel(f"<h2>{title}</h2><p>{note}</p>")
        lbl.setAlignment(Qt.AlignCenter); lbl.setWordWrap(True)
        QVBoxLayout(self).addWidget(lbl)
