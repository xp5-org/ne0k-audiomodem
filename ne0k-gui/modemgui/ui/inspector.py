# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""UI inspector: hover any widget or menu entry to see a stable dotted path, click to copy it.

``name_widgets`` gives every widget/action the name of the Python attribute that holds it, so the
path shown here is the path in the code: ``MainWindow.bar-pll.pll.align`` is
``self.pll.align``.  Toggle with Ctrl+Shift+I.

Help mode (Ctrl+Shift+H) is the same hook with a different click action: a setting widget carries the
``helpKey`` property (``<section>.<field>``, set where the settings page is built), and clicking it opens
the built-in help at that setting.  Names and help keys share one scheme."""
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
    top = type(QWidget.window(w)).__name__          # not w.window(): a panel may keep a widget in self.window
    if not parts or parts[-1] != top:
        parts.append(top)
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


def help_key(w: QWidget | None) -> str:
    """The ``<section>.<field>`` of the setting ``w`` belongs to, or "" if it is not part of one."""
    while w is not None:
        k = w.property("helpKey")
        if k:
            return str(k)
        w = w.parentWidget()
    return ""


class Inspector(QObject):
    NAMES, HELP = "names", "help"

    def __init__(self, window, parent=None) -> None:
        super().__init__(parent or window)
        self.window = window
        self.on = False
        self.mode = self.NAMES
        self.last = ""
        self.help_window = None
        self.mode_changed = None          # optional callback(mode | None) so menu checkmarks follow

    def toggle(self, on: bool | None = None, mode: str | None = None) -> None:
        mode = mode or self.mode
        if on is None:
            on = not (self.on and mode == self.mode)
        if on and self.on and mode != self.mode:         # switch mode without dropping the filter
            self.mode = mode
        else:
            self.on, self.mode = on, mode
            app = QApplication.instance()
            (app.installEventFilter if self.on else app.removeEventFilter)(self)
        if self.mode_changed:
            self.mode_changed(self.mode if self.on else None)
        self.window.statusBar().showMessage(
            ("Help: click any setting for its description (Ctrl+Shift+H to leave)" if self.mode == self.HELP else
             "Inspect: hover a control for its path, click to copy (Ctrl+Shift+I to leave)")
            if self.on else "Inspect off")

    def show_help(self, key: str) -> None:
        from .help_window import open_help
        self.help_window = open_help(key, self.window)

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
        if self.mode == self.HELP:
            if isinstance(w, QMenu | QMenuBar):
                return False                                    # menus keep working in help mode
            key = help_key(w)
            if t == QEvent.MouseMove:
                tip = f"{key}: click for help" if key else "No help for this control"
                QToolTip.showText(gpos, tip, w)
                self.window.statusBar().showMessage(tip)
                return False
            if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                if key:
                    self.show_help(key)
                return True                                     # swallow: help must not press the button
            return t == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton
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
