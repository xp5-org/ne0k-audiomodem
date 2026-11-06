# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""UI inspector: hover any widget or menu entry to see a stable dotted path, click to copy it.

``name_widgets`` gives every widget/action the name of the Python attribute that holds it, so the
path shown here is the path in the code: ``MainWindow.bar-pll.pll.align`` is
``self.pll.align``.  Toggle with Ctrl+Shift+I."""
from __future__ import annotations

import re

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import QApplication, QMenu, QMenuBar, QToolTip, QWidget


def name_widgets(root: QObject) -> None:
    """Set objectName from the attribute name for every unnamed QWidget/QAction under ``root``."""
    seen: set[int] = set()

    def walk(o: QObject) -> None:
        if id(o) in seen:
            return
        seen.add(id(o))
        for attr, v in list(vars(o).items()):
            vals = v if isinstance(v, (list, tuple)) else [v]
            for x in vals:
                if isinstance(x, (QWidget, QAction)):
                    if not x.objectName() and not attr.startswith("_"):
                        x.setObjectName(attr)
                    walk(x)
    walk(root)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("&", "")).strip()


def widget_path(w: QWidget) -> str:
    """Named ancestors joined by dots; an unnamed leaf shows as <ClassName>."""
    parts = []
    o: QObject | None = w
    while o is not None:
        n = o.objectName()
        if n:
            parts.append(n)
        elif o is w:
            parts.append(f"<{type(o).__name__}>")
        o = o.parent()
    if not parts or parts[-1] != type(w.window()).__name__:
        parts.append(type(w.window()).__name__)
    return ".".join(reversed(parts))


def menu_path(menu: QMenu, action: QAction | None) -> str:
    bar = []
    m: QObject | None = menu
    while isinstance(m, QMenu):
        bar.append(_clean(m.title()))
        m = m.parent()
        if isinstance(m, QMenu):
            continue
        break
    path = "menu." + ".".join(reversed([b for b in bar if b]))
    if action is not None and not action.isSeparator():
        path += "." + _clean(action.text())
        if action.objectName():
            path += f"  (self.{action.objectName()})"
    return path


class Inspector(QObject):
    def __init__(self, window, parent=None) -> None:
        super().__init__(parent or window)
        self.window = window
        self.on = False
        self.last = ""

    def toggle(self, on: bool | None = None) -> None:
        self.on = (not self.on) if on is None else on
        app = QApplication.instance()
        (app.installEventFilter if self.on else app.removeEventFilter)(self)
        self.window.statusBar().showMessage(
            "Inspect: hover a control for its path, click to copy (Ctrl+Shift+I to leave)"
            if self.on else "Inspect off")

    def describe(self, w: QWidget, gpos) -> str:
        if isinstance(w, QMenu):
            return menu_path(w, w.actionAt(w.mapFromGlobal(gpos)))
        if isinstance(w, QMenuBar):
            a = w.actionAt(w.mapFromGlobal(gpos))
            return "menu." + _clean(a.text()) if a else widget_path(w)
        return f"{widget_path(w)}   [{type(w).__name__}]"

    def eventFilter(self, obj, ev) -> bool:
        t = ev.type()
        if t not in (QEvent.MouseMove, QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
            return False
        gpos = ev.globalPosition().toPoint()
        w = QApplication.widgetAt(gpos)
        if w is None:
            return False
        if t == QEvent.MouseMove:
            self.last = self.describe(w, gpos)
            QToolTip.showText(gpos, self.last, w)
            self.window.statusBar().showMessage(self.last)
            return False
        if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
            self.last = self.describe(w, gpos)
            QGuiApplication.clipboard().setText(self.last.split("   [")[0].split("  (")[0])
            self.window.statusBar().showMessage("Copied: " + self.last)
            return True                      # swallow: inspecting must not press the button
        return t == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton
