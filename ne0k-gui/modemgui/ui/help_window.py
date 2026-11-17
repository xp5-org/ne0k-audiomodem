# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Built-in help viewer: the text comes from ``engine.helptext`` -- every element of the window (``ui.*`` /
``menu.*`` keys) and every setting (``<section>.<field>``: what, impact, relation to OFDM)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QTextBrowser, QVBoxLayout

from ..engine import helptext


def page(key: str | None = None) -> str:
    """The clicked element's topic on top, then the whole help: the window's elements and every setting."""
    return (helptext.focus_html(key) if key else "") + helptext.ui_html() + helptext.html()


class HelpWindow(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.setWindowFlag(Qt.Window, True)
        self.setModal(False)
        self.resize(600, 560)
        self.view = QTextBrowser(self)
        self.view.setOpenExternalLinks(False)
        self.view.setHtml(page())
        self.key = ""
        lay = QVBoxLayout(self); lay.addWidget(self.view)

    def show_key(self, key: str | None = None) -> None:
        self.show(); self.raise_(); self.activateWindow()
        if key:
            self.key = key
            self.view.setHtml(page(key))
            self.view.verticalScrollBar().setValue(0)
            self.setWindowTitle(f"Help - {helptext.title(key)}")


_windows: dict[int, HelpWindow] = {}


def open_help(key: str, owner=None) -> HelpWindow:
    """Show the help at ``key``, one window per owner.  Over a modal dialog (the settings dialog) the window is
    parented to that dialog, or it could not be clicked while the dialog is up."""
    if isinstance(owner, HelpWindow):                  # right-click inside the help itself
        owner.show_key(key)
        return owner
    parent = QApplication.activeModalWidget() or owner
    win = _windows.get(id(parent))
    if win is None:
        win = HelpWindow(parent)
        _windows[id(parent)] = win
        win.destroyed.connect(lambda *_: _windows.pop(id(parent), None))
    win.show_key(key)
    return win
