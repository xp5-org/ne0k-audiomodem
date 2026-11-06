# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Log tab: the engine log, live.  ``add`` is safe to call from any thread."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget

#: Any log line containing this is a CSS transmission that phy_mode=ofdm_only must never allow
#: (see mesh_node.py's _ofdm_only_violation) -- rare, and the whole point is that it must not be
#: missed, so it gets its own bold red highlighting instead of blending into the scrollback.
VIOLATION_MARKER = "OFDM_ONLY VIOLATION"


class _ViolationHighlighter(QSyntaxHighlighter):
    def __init__(self, document) -> None:
        super().__init__(document)
        self._fmt = QTextCharFormat()
        self._fmt.setForeground(QColor("red"))
        self._fmt.setFontWeight(700)                  # bold

    def highlightBlock(self, text: str) -> None:
        if VIOLATION_MARKER in text:
            self.setFormat(0, len(text), self._fmt)


class LogView(QWidget):
    _line = Signal(str)

    def __init__(self, backlog=(), parent=None) -> None:
        super().__init__(parent)
        self.text = QPlainTextEdit(); self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(5000)
        self.text.setPlaceholderText("Engine log.  Turn on Diagnostics > Save debug log to keep it.")
        self._highlighter = _ViolationHighlighter(self.text.document())
        QVBoxLayout(self).addWidget(self.text)
        self._line.connect(self.text.appendPlainText)
        for l in backlog:
            self.text.appendPlainText(l)

    def add(self, line: str) -> None:
        self._line.emit(line)
