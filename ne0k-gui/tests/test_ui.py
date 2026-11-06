"""Offscreen smoke tests (pytest-qt).  Skipped when PySide6 is not installed."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QDockWidget  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402

from modemgui.audio import Registry  # noqa: E402
from modemgui.engine import Levels, Session, Spectrum  # noqa: E402
from modemgui.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture
def win(qtbot, tmp_path):
    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    w = MainWindow(Session(Registry()), st)
    qtbot.addWidget(w)
    w.show()
    return w


def test_window_builds_and_lists_devices(win):
    assert len(win.panes) == 4
    assert win.io.rx.count() >= 1 and win.io.tx.count() >= 1       # at least "(none)"
    assert win.statusBar().findChild(object) is not None


def test_levels_and_spectrum_reach_widgets(win, qtbot):
    win.session._emit(Levels(-6.0, -9.0, 3, True, 1000, 0))
    freqs = np.fft.rfftfreq(2048, 1 / 48000)
    win.session._emit(Spectrum(0, np.full((4, len(freqs)), -60, np.float32),
                               np.zeros(4, bool), freqs))
    qtbot.waitUntil(lambda: win.spectro._buf is not None, timeout=2000)
    assert win.io.clip_lbl.text() == "clip 3"
    assert win.io.peak.value() == -6


def test_open_file_source_and_close(win, qtbot, tmp_path):
    import wave
    x = (np.sin(np.arange(48000) * 0.3) * 8000).astype("<i2")
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000); w.writeframes(x.tobytes())
    win.session.open(f"file:{p}#repeat", None)
    qtbot.waitUntil(lambda: win.spectro._buf is not None, timeout=5000)
    assert win.io.open_btn.text() == "Close"
    win.io.toggle_open()
    qtbot.waitUntil(lambda: win.io.open_btn.text() == "Open", timeout=3000)
    assert win.spectro._buf is None


def test_selection_persists_by_name(win, tmp_path, qtbot):
    win.io.settings.setValue("io/rx_addr", "sd:99"); win.io.settings.setValue("io/rx_name", "")
    win.io.refresh()                      # stale address that no longer exists -> falls back
    assert win.io.rx.currentIndex() == 0


def test_decoded_goes_to_log_and_settings(win, qtbot):
    from modemgui.engine import Decoded
    from modemgui.ui.settings_dialog import SettingsDialog
    win.session._emit(Decoded("hi", 1, 15, 0, "DATA", True, 1, 0, 30.0, 0))
    qtbot.waitUntil(lambda: win.feed.count == 1, timeout=2000)
    assert "hi" in win.logview.text.toPlainText()
    dlg = SettingsDialog(win.app)
    qtbot.addWidget(dlg)
    dlg.pages["css"].w["symbol_dur"].setValue(0.1)
    assert dlg.pages["css"].config().symbol_dur == 0.1


def _link_events(t=100.0):
    from modemgui.engine import LinkEvent
    return [LinkEvent("tx", 1, 2, t, t + 7, "DATA seq=0 4B"),
            LinkEvent("lock", 2, 1, t + 0.1, t + 0.1, "1 frames"),
            LinkEvent("rx", 2, 1, t + 0.1, t + 7, "DATA seq=0"),
            LinkEvent("tx", 2, 1, t + 9, t + 16, "ACK seq=0"),
            LinkEvent("err", 1, -1, t + 9, t + 16, "no frames decoded"),
            LinkEvent("retry", 1, 2, t + 20, t + 20, "attempt 1"),
            LinkEvent("ack", 1, 2, t + 30, t + 30, "seq=0")]


def test_pianoroll_lanes_dots_and_scroll(win, qtbot):
    for ev in _link_events():
        win.session._emit(ev)
    qtbot.waitUntil(lambda: len(win.linklog.events) == 7, timeout=2000)
    pr = win.pianoroll
    assert pr._lanes == {1: 0, 2: 1}                       # one lane per participant
    assert len(pr.dots.points()) == 7
    assert len(pr.curves["tx"].xData) == 4 and len(pr.curves["rx"].xData) == 2
    # horizontal scroll: moving the bar leaves live-follow and shifts the x range
    pr.t_origin = 100.0
    pr.refresh()
    pr.scroll.setValue(0)
    assert not pr.follow.isChecked()
    assert pr.plot.getViewBox().viewRange()[0][0] == pytest.approx(0, abs=0.2)
    assert "ERR: 1" in pr.summary.text() and "retry: 1" in pr.summary.text()


def test_spectrogram_event_overlay(win, qtbot):
    import time
    from modemgui.engine import LinkEvent
    freqs = np.fft.rfftfreq(2048, 1 / 48000)
    win.session._emit(Spectrum(0, np.full((4, len(freqs)), -60, np.float32),
                               np.zeros(4, bool), freqs))
    qtbot.waitUntil(lambda: win.spectro._buf is not None, timeout=2000)
    now = time.monotonic()
    win.session._emit(LinkEvent("tx", 1, 2, now - 2, now - 1, "x"))
    win.session._emit(LinkEvent("err", 2, -1, now - 0.5, now - 0.5, "x"))
    qtbot.waitUntil(lambda: len(win.linklog.events) == 2, timeout=2000)
    assert not win.spectro.ov_dots.isVisible()             # off by default
    win.spectro.overlay.setChecked(True)
    assert len(win.spectro.ov_dots.points()) == 2
    xs = sorted(p.pos().x() for p in win.spectro.ov_dots.points())
    assert 0 <= xs[0] < xs[1] <= win.spectro.columns       # older event is further left
    win.spectro.overlay.setChecked(False)
    assert not win.spectro.ov_dots.isVisible()


def test_events_chart_frozen_while_audio_closed(win, qtbot):
    pr = win.pianoroll
    assert not pr._active
    t0 = pr._now()
    qtbot.wait(400)
    assert pr._now() == t0                      # closed: live edge does not advance
    pr.set_active(True)
    qtbot.wait(400)
    assert pr._now() > t0
    pr.set_active(False)
    t1 = pr._now()
    qtbot.wait(300)
    assert pr._now() == t1


def test_panes_are_docks_and_lockbar_reports_acquisition(win):
    assert all(d.features() & QDockWidget.DockWidgetMovable for d in win.panes)
    win.feed.set_open(True)
    win.pll.locked(0.42, 0.8)
    assert "LOCKED" in win.pll.state.text()
    assert "0.42" in win.pll.timing.text()
    assert win.pll.align.value() == 80
    assert [d.windowTitle() for d in win.panes] == ["Spectrograph", "Event Log", "BAT", "Log"]


def test_widgets_named_from_attributes_and_paths(win):
    from modemgui.ui.inspector import menu_path, widget_path
    assert win.pll.align.objectName() == "align"
    assert widget_path(win.pll.align) == \
        "MainWindow.bar-pll.pll.align"
    m = win.menuBar().findChildren(type(win.menuBar().addMenu("x")))[0]
    assert menu_path(m, None).startswith("menu.")


def test_layout_presets_keep_all_panes(win):
    for mode in ("vertical", "horizontal", "grid", "tabbed"):
        win.arrange(mode)
        assert all(not d.isFloating() for d in win.panes)
    win.inspector.toggle(True); win.inspector.toggle(False)


def test_one_shared_pll_bar_and_control_column(win):
    assert win.feed.lockbar is win.pll
    col = win.column
    assert win.io_dock.widget() is col and col.io is win.io
    col.link_info.record_rx(100); col.link_info.record_tx(50)
    col.link_info.refresh("x")
    assert "80 bit/s" in col.link_info.rx_bitrate.text() and "40 bit/s" in col.link_info.tx_bitrate.text()
    col.uart.refresh(None)
    assert col.uart.state.text() == "closed"


def test_modem_mode_lives_in_settings_and_front_panel_shows_it(win):
    from modemgui.ui.settings_dialog import SettingsDialog, _set
    dlg = SettingsDialog(win.app, win, "mode", io=win.io)
    assert dlg.tabs.tabText(dlg.tabs.currentIndex()) == "Modem Mode"
    g = dlg.pages["general"].w
    assert {"modem", "topology", "node_id", "peer_id", "beacon", "beacon_min_s", "beacon_max_s"} <= set(g)
    _set(g["modem"], "ofdm"); _set(g["topology"], "mesh")
    dlg.apply(); win._show_mode()
    assert win.app.general.modem == "ofdm" and "OFDM, mesh" in win.column.mode.text()
    assert not win.column.io_type.model().item(1).isEnabled()      # GNURadio not available yet


def test_mesh_topology_beacons_and_learns_neighbors(win):
    from modemgui.engine import Decoded
    win.app.general.topology = "mesh"; win.app.general.beacon = True
    assert win._make_mesh().mesh.beacon is True
    win.app.general.topology = "p2p"
    m = win._make_mesh()
    assert m.mesh.beacon is False
    m.handle(Decoded("", 7, 15, 0, "BEACON", True, 1, 0, 21.0, 0))
    assert m.neighbors[7][0] == 21.0


def test_status_strip_and_dock_layout_round_trip(tmp_path):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtWidgets import QApplication
    from modemgui.audio import Registry
    from modemgui.engine import Session
    from modemgui.ui.main_window import MainWindow
    QApplication.instance() or QApplication([])
    ini = str(tmp_path / "s.ini")
    w = MainWindow(Session(Registry()), QSettings(ini, QSettings.IniFormat))
    w.strip.refresh()
    assert w.strip.cpu.text().startswith("CPU ") and "Memory" in w.strip.buf.text()
    w.arrange("tabbed")                                   # tab, then pull one back out and float it
    assert len(w.tabifiedDockWidgets(w.panes[0])) == 3
    w.panes[2].setFloating(True)
    w.show(); QApplication.processEvents()
    w.close()
    w2 = MainWindow(Session(Registry()), QSettings(ini, QSettings.IniFormat))
    assert w2.panes[2].isFloating()
    w2.close()


def test_gui_runs_the_mesh_node_and_a_self_addressed_send_is_acked(tmp_path):
    import os, time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from modemgui.audio import Registry
    from modemgui.engine import Session
    from modemgui.engine.mesh_node import MeshNode
    from modemgui.ui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    w = MainWindow(Session(Registry()), QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    w.app.css.hear_own = True
    w.app.general.modem = "css"; w.app.general.node_id = 3; w.app.general.peer_id = 3
    w.session.open("loop:ab", "loop:ab")
    end = time.monotonic() + 10
    while w.link is None and time.monotonic() < end:          # the bridge delivers the "open" status
        app.processEvents(); time.sleep(0.05)
    assert isinstance(w.link, MeshNode) and w.link.me == 3
    w.send_text("hi mesh")
    end = time.monotonic() + 90
    while w.link.tx_count == 0 and time.monotonic() < end:
        app.processEvents(); time.sleep(0.05)
    assert w.link.tx_count == 1 and w.link.rx_count >= 1           # DATA was ACKed through the control plane
    w.close()


def test_mesh_has_no_configured_peer_and_sends_to_the_strongest_neighbour(win):
    from modemgui.engine import Decoded
    from modemgui.ui.settings_dialog import SettingsDialog, _set
    g = SettingsDialog(win.app, win, "mode", io=win.io).pages["general"].w
    dlg = SettingsDialog(win.app, win, "mode", io=win.io)
    g = dlg.pages["general"].w
    _set(g["topology"], "mesh")
    assert not g["peer_id"].isEnabled() and g["beacon"].isEnabled()
    _set(g["topology"], "p2p")
    assert g["peer_id"].isEnabled() and not g["beacon"].isEnabled()
    win.app.general.topology = "mesh"
    m = win._make_mesh()
    assert win._mesh_dst(m) is None                       # nobody heard yet: nothing is sent
    m.handle(Decoded("", 4, 15, 0, "BEACON", True, 1, 0, 12.0, 0))
    m.handle(Decoded("", 9, 15, 0, "BEACON", True, 1, 0, 25.0, 0))
    assert win._mesh_dst(m) == 9
