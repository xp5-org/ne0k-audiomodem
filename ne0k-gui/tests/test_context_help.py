# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Right-click help (ui/context_help.py): every element of the window and every menu entry has an entry of its own,
and the right-click paths open it."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QEvent, QPointF, QSettings, Qt                       # noqa: E402
from PySide6.QtGui import QMouseEvent                                           # noqa: E402
from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QApplication, QComboBox, QLabel,  # noqa: E402
                               QLineEdit, QMenu, QPlainTextEdit, QProgressBar, QScrollBar, QWidget)

from modemgui.audio import Registry                                             # noqa: E402
from modemgui.engine import helptext                                            # noqa: E402
from modemgui.engine.session import Session                                     # noqa: E402
from modemgui.ui import context_help as ch                                      # noqa: E402
from modemgui.ui import help_window                                             # noqa: E402

CONTROLS = (QAbstractButton, QComboBox, QAbstractSpinBox, QProgressBar, QLabel, QScrollBar, QPlainTextEdit)


@pytest.fixture(scope="module")
def win(tmp_path_factory):
    from modemgui.ui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    w = MainWindow(Session(Registry()), QSettings(str(tmp_path_factory.mktemp("ch") / "s.ini"), QSettings.IniFormat))
    w.show(); app.processEvents()
    yield w
    w.close()


def _widgets(w):
    return [x for x in w.findChildren(QWidget) if x.isVisibleTo(w) and not isinstance(x, QMenu)]


def _menu_actions(menu, out):
    for a in menu.actions():
        if a.isSeparator():
            continue
        out.append((menu, a))
        if a.menu():
            _menu_actions(a.menu(), out)
    return out


def test_every_named_control_has_its_own_entry(win):
    """A named control (button, field, meter, label...) must resolve to an entry named after IT, not fall back to
    its panel: adding a widget without a help entry fails here."""
    missing = []
    for x in _widgets(win):
        n = x.objectName()
        if not n or n.startswith("qt_") or not isinstance(x, CONTROLS + (type(win.spectro.plot),)):
            continue
        k = ch.topic_for(x)
        if not (k.endswith("." + n) and k in helptext.UI):
            missing.append(f"{n} ({type(x).__name__}) -> {k}")
    assert not missing, missing


def test_every_element_resolves_and_labels_stand_for_their_field(win):
    for x in _widgets(win):
        k = ch.topic_for(x)
        assert k in helptext.UI or helptext.topic(k), (x, k)
    io = win.column.io
    form_label = io.layout().itemAt(0).widget().layout().itemAt(0).layout().labelForField(io.rx)
    assert ch.topic_for(form_label) == "ui.io.rx"
    assert ch.topic_for(io.rx) == "ui.io.rx" and ch.topic_for(win.column.uart.rx) == "ui.uart.rx"
    assert ch.topic_for(win.spectro.floor.findChild(QLineEdit)) == "ui.floor"     # the spin box's inner editor
    toolbar_label = [lb for lb in win.pianoroll.findChildren(QLabel) if lb.text() == "window"][0]
    assert ch.topic_for(toolbar_label) == "ui.pianoroll.window"
    assert ch.topic_for(win.io.power_btn) == "ui.power_btn"          # disabled until audio opens: still answers


def test_every_menu_entry_has_its_own_entry(win):
    missing = []
    for top in win.menuBar().actions():
        for menu, a in _menu_actions(top.menu(), []):
            full = ".".join(["menu"] + _titles(menu) + [ch.clean(a.menu().title() if a.menu() else a.text())])
            if ch.menu_key(menu, a) != full and not full.startswith("menu.View.Layout."):
                missing.append(full)
    assert not missing, missing
    assert ch.menu_key(_find_menu(win, "Layout"), _find_menu(win, "Layout").actions()[0]) == "menu.View.Layout"


def _titles(menu):
    out = []
    while isinstance(menu, QMenu):
        out.append(ch.clean(menu.title()))
        menu = menu.parent()
    return list(reversed(out))


def _find_menu(win, title):
    for top in win.menuBar().actions():
        for menu, a in _menu_actions(top.menu(), []):
            if a.menu() and ch.clean(a.menu().title()) == title:
                return a.menu()
    raise KeyError(title)


def test_no_stale_entries(win):
    """Every ui.* / menu.* entry still points at something in the window."""
    used = {ch.topic_for(x) for x in _widgets(win)}
    for top in win.menuBar().actions():
        used.add(ch.menu_key(top.menu()))
        used |= {ch.menu_key(m, a) for m, a in _menu_actions(top.menu(), [])}
    windows = {"ui.MainWindow", "ui.SettingsDialog", "ui.HelpWindow"}
    assert set(helptext.UI) - used - windows == set()


def test_menu_offers_help_and_keeps_the_text_menu(win):
    m = ch.ContextHelp().menu_for(win.io.refresh_btn)
    acts = [a for a in m.actions() if not a.isSeparator()]
    assert [a.text() for a in acts] == ["Help: Refresh"] and acts[0].data() == "ui.refresh_btn"
    m2 = ch.ContextHelp().menu_for(win.logview.text.viewport())
    texts = [a.text() for a in m2.actions()]
    assert any("Copy" in t for t in texts) and texts[-1] == "Help: Log text"
    acts[0].trigger()
    QApplication.processEvents()
    hw = help_window._windows.get(id(win))
    assert hw is not None and hw.isVisible() and hw.windowTitle() == "Help - Refresh"
    assert "List the audio devices again" in hw.view.toPlainText().split("The window")[0]   # the topic is on top
    hw.close()


def test_right_click_in_an_open_menu_opens_help_not_the_action(win):
    fired = []
    m = QMenu("Edit", win)
    a = m.addAction("Copy status", lambda: fired.append(1))
    m.popup(win.mapToGlobal(win.rect().center())); QApplication.processEvents()
    pos = QPointF(m.actionGeometry(a).center())
    for t in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
        ev = QMouseEvent(t, pos, m.mapToGlobal(pos), Qt.RightButton, Qt.RightButton, Qt.NoModifier)
        QApplication.sendEvent(m, ev)
    QApplication.processEvents()
    assert fired == []
    hw = help_window._windows.get(id(win))
    assert hw is not None and hw.windowTitle() == "Help - Copy status"
    hw.close(); m.close()


def test_plot_menu_gets_the_help_entry(win):
    plot = win.spectro.plot
    ch.ContextHelp()._add_to_plot_menu(plot)
    ch.ContextHelp()._add_to_plot_menu(plot)                          # once only
    texts = [a.text() for a in plot.getViewBox().menu.actions()]
    assert texts.count("Help: Spectrogram") == 1


def test_settings_dialog_fields_and_labels_open_the_setting(tmp_path):
    from modemgui.ui.appsettings import AppSettings
    from modemgui.ui.settings_dialog import SettingsDialog
    QApplication.instance() or QApplication([])
    dlg = SettingsDialog(AppSettings(QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)))
    for sec, pg in dlg.pages.items():
        for name, w in pg.w.items():
            assert ch.topic_for(w) == f"{sec}.{name}"
            for inner in w.findChildren(QLineEdit):
                assert ch.topic_for(inner) == f"{sec}.{name}"
    assert ch.topic_for(dlg) == "ui.SettingsDialog"
    assert "Impact:" in helptext.focus_html("ofdm.n_fft")


def test_right_click_event_pops_the_help_menu(win, monkeypatch):
    """The real path: a context-menu event on any widget, through the app-wide filter."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QContextMenuEvent
    shown = []
    monkeypatch.setattr(ch.ContextHelp, "_popup", staticmethod(lambda m, g: shown.append([x.text() for x in m.actions()])))
    btn = win.io.test_btn
    ev = QContextMenuEvent(QContextMenuEvent.Keyboard, QPoint(5, 5), btn.mapToGlobal(QPoint(5, 5)))
    QApplication.sendEvent(btn, ev)
    assert shown == [["Help: Test loopback"]]
    shown.clear()
    vp = win.spectro.plot.viewport()                     # pyqtgraph's own menu handles plots that have one
    c = vp.rect().center()
    QApplication.sendEvent(vp, QContextMenuEvent(QContextMenuEvent.Mouse, c, vp.mapToGlobal(c)))
    assert shown == []
    vp = win.pianoroll.plot.viewport()                   # ... and one with that menu turned off gets the help menu
    c = vp.rect().center()
    QApplication.sendEvent(vp, QContextMenuEvent(QContextMenuEvent.Mouse, c, vp.mapToGlobal(c)))
    assert shown == [["Help: Timeline"]]
