# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Right-click any element of any window for its help.

One application-wide event filter (``install``), so a widget added later needs no code here -- only a help entry
(``tests/test_context_help.py`` fails until it has one):

* a plain widget gets a one-entry menu, "Help: <element>";
* a text field keeps its usual Cut / Copy / Paste menu, with the help entry added;
* a pyqtgraph plot keeps its own menu (view all, export...), with the help entry added; a plot with that menu
  turned off gets the plain one;
* a menu entry opens its help instead of running (right-click inside an open menu), and so does a menu bar title.

Which entry: a settings field carries ``helpKey`` (``<section>.<field>``, the settings help); anything else is
looked up by object name -- the attribute that holds the widget, as ``inspector.name_widgets`` sets it -- as
``ui.<parent>.<name>`` then ``ui.<name>``, climbing to the nearest ancestor that has one.  A label beside a field
(form rows, toolbars) stands for that field.  ``uiHelp`` on a widget names its ``ui.*`` entry explicitly.  The window
class (``ui.MainWindow``, ``ui.SettingsDialog``...) catches the rest.
"""
from __future__ import annotations

import re

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtWidgets import (QApplication, QBoxLayout, QFormLayout, QGraphicsView, QLabel, QLineEdit, QMenu,
                               QMenuBar, QPlainTextEdit, QTextEdit, QWidget)

from ..engine import helptext

FALLBACK = "ui.MainWindow"


def clean(text: str) -> str:
    """Menu text as the help keys spell it: "&&" -> "&", other "&" mnemonics dropped, no shortcut tab."""
    return re.sub(r"\s+", " ", re.sub(r"&(&?)", r"\1", text.split("\t")[0])).strip()


def menu_key(menu: QMenu | QMenuBar, action=None) -> str:
    titles = []
    m = menu
    while isinstance(m, QMenu):
        titles.append(clean(m.title()))
        m = m.parent()
    key = ".".join(["menu"] + list(reversed(titles)))
    if action is not None and not action.isSeparator():
        key += "." + clean(action.menu().title() if action.menu() else action.text())
    return helptext.lookup_menu(key) or FALLBACK


def _find_in(lay, w):
    for i in range(lay.count()):
        it = lay.itemAt(i)
        if it.widget() is w:
            return lay, i
        if it.layout() is not None:
            got = _find_in(it.layout(), w)
            if got:
                return got
    return None


def partner(label: QLabel) -> QWidget | None:
    """The field an unnamed label names: its form row's field, or the widget right after it in a row of controls."""
    p = label.parentWidget()
    got = _find_in(p.layout(), label) if p is not None and p.layout() is not None else None
    if not got:
        return None
    lay, i = got
    if isinstance(lay, QFormLayout):
        row, role = lay.getWidgetPosition(label)
        if role == QFormLayout.LabelRole:
            it = lay.itemAt(row, QFormLayout.FieldRole)
            return it.widget() if it is not None else None
        return None
    if isinstance(lay, QBoxLayout) and i + 1 < lay.count():
        nxt = lay.itemAt(i + 1).widget()
        if nxt is not None and not isinstance(nxt, QLabel):
            return nxt
    return None


def _named_parent(o: QWidget) -> str:
    q = o.parentWidget()
    while q is not None and (not q.objectName() or q.objectName().startswith("qt_")):
        q = q.parentWidget()
    return q.objectName() if q is not None else ""


def topic_for(w: QWidget) -> str:
    """The help key for widget ``w`` (never empty)."""
    if isinstance(w, QLabel) and not w.objectName():
        w = partner(w) or w
    o = w
    while o is not None:
        k = o.property("helpKey")
        if k:
            return str(k)
        k = o.property("uiHelp")
        if k:
            return "ui." + str(k)
        n = o.objectName()
        if n:
            p = _named_parent(o)
            for cand in ([f"ui.{p}.{n}"] if p else []) + [f"ui.{n}"]:
                if cand in helptext.UI:
                    return cand
        o = o.parentWidget()
    top = f"ui.{type(QWidget.window(w)).__name__}"
    return top if top in helptext.UI else FALLBACK


def _plot_of(w: QWidget | None):
    """The pyqtgraph PlotWidget ``w`` belongs to (the view or its viewport), else None."""
    while w is not None:
        if isinstance(w, QGraphicsView) and hasattr(w, "getViewBox"):
            return w
        w = w.parentWidget()
    return None


def _editor(w: QWidget):
    for x in (w, w.parentWidget()):
        if isinstance(x, (QLineEdit, QPlainTextEdit, QTextEdit)):
            return x
    return None


def show(key: str, near: QWidget | None) -> None:
    from .help_window import open_help
    while (p := QApplication.activePopupWidget()) is not None:     # an open menu would sit over the help
        p.close()
    while isinstance(near, QMenu):                 # a menu is its own popup window: use the window it belongs to
        near = near.parentWidget()
    open_help(key, QWidget.window(near) if near is not None else None)


class ContextHelp(QObject):
    _declined = None            # where a plot was left to pop its own menu: its parents get the same event next
    def menu_for(self, w: QWidget) -> QMenu:
        """What a right-click on ``w`` pops up: its standard menu (text fields) plus the help entry."""
        key = topic_for(w)
        ed = _editor(w)
        menu = ed.createStandardContextMenu() if ed is not None else QMenu()
        if not menu.isEmpty():
            menu.addSeparator()
        act = menu.addAction(f"Help: {helptext.title(key)}")
        act.setData(key)
        act.triggered.connect(lambda _=False, k=key, n=w: show(k, n))
        return menu

    def _add_to_plot_menu(self, plot) -> None:
        vb = plot.getViewBox()
        if getattr(vb, "_help_added", False) or getattr(vb, "menu", None) is None:
            return
        vb._help_added = True
        key = topic_for(plot)
        vb.menu.addSeparator()
        vb.menu.addAction(f"Help: {helptext.title(key)}").triggered.connect(lambda _=False: show(key, plot))

    def eventFilter(self, obj, ev) -> bool:
        t = ev.type()
        if t == QEvent.MouseButtonPress:
            self._declined = None
        if t in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease) and isinstance(obj, QWidget) \
                and ev.button() == Qt.RightButton:
            if isinstance(obj, QMenu):                     # an entry in an open menu: its help, not the action
                if t == QEvent.MouseButtonRelease:
                    a = obj.actionAt(ev.position().toPoint())
                    if a is not None and not a.isSeparator():
                        show(menu_key(obj, a), obj)
                return True
            if t == QEvent.MouseButtonPress:
                plot = _plot_of(obj)
                if plot is not None and plot.getViewBox().menuEnabled():
                    self._add_to_plot_menu(plot)          # before pyqtgraph pops its menu on this click
            return False
        if t != QEvent.ContextMenu or not isinstance(obj, QWidget) or isinstance(obj, QMenu):
            return False
        gpos = ev.globalPos()
        if self._declined == (gpos.x(), gpos.y()):
            return False                                    # the same click, passed up from a plot that has a menu
        w = obj
        if ev.reason() == QContextMenuEvent.Mouse:
            w = QApplication.widgetAt(gpos) or obj          # the widget under the mouse, even a disabled one
        else:                                               # the Menu key: the focused widget, even after it declined
            f = QApplication.focusWidget()
            if f is not None and (f is obj or obj.isAncestorOf(f)):
                w = f
        if isinstance(w, QMenuBar):
            a = w.actionAt(w.mapFromGlobal(gpos))
            show(menu_key(a.menu()) if a is not None and a.menu() else topic_for(w), w)
            return True
        plot = _plot_of(w)
        if plot is not None and plot.getViewBox().menuEnabled():
            self._declined = (gpos.x(), gpos.y())
            return False                                    # pyqtgraph's own menu, with the help entry added
        if w.contextMenuPolicy() in (Qt.CustomContextMenu, Qt.ActionsContextMenu) and w is not obj:
            return False                                    # a widget with a menu of its own keeps it
        if obj.contextMenuPolicy() in (Qt.CustomContextMenu, Qt.ActionsContextMenu):
            return False
        menu = self.menu_for(w)
        self._popup(menu, gpos)
        menu.deleteLater()
        return True

    @staticmethod
    def _popup(menu: QMenu, gpos) -> None:
        menu.exec(gpos)


_installed: ContextHelp | None = None


def install(app: QApplication | None = None) -> ContextHelp:
    """Install the filter once per process (every window shares it)."""
    global _installed
    if _installed is None:
        app = app or QApplication.instance()
        _installed = ContextHelp(app)
        app.installEventFilter(_installed)
    return _installed
