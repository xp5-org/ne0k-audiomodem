# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Settings persistence and the troubleshooting capture."""
import os
import wave
import zipfile

import numpy as np
import pytest

from modemgui.engine.diag import Diagnostics
from modemgui.engine.modemcfg import CssConfig, DiagConfig, HamConfig, OfdmConfig, describe


def test_every_field_is_described_for_the_dialog():
    for cls in (CssConfig, OfdmConfig, DiagConfig):
        for d in describe(cls):
            assert d["label"] and d["group"], (cls.__name__, d["name"])


def test_defaults_match_the_reference_handshake_convention():
    pytest.importorskip("scipy")
    from modemgui.engine.ofdm_link import _imports
    H, _ = _imports()
    c = OfdmConfig()
    for k in ("n_fft", "cp_ratio", "pilot_spacing", "probe_duration_s", "probe_meas_syms",
              "snr_margin_db", "max_bits_per_bin", "snr_estimator"):
        assert getattr(c, k) == H.HS_BAND[k], k


def test_diag_log_wav_and_bundle(tmp_path):
    from modemgui.audio import Registry
    from modemgui.engine import Session
    cfg = DiagConfig(debug_log=True, wav_rx=True, wav_tx=True, out_dir=str(tmp_path))
    d = Diagnostics(cfg, header={"modemgui": "test"})
    d.log("before the file was open")
    d.start_log()
    s = Session(Registry(), 48000, 1024)
    s.open("loop:ab", "loop:ab")
    d.attach(s)
    s.play(np.sin(np.arange(24000) * 0.2).astype(np.float32) * 0.5)
    s.pipe.wait_tx(5)
    import time
    time.sleep(0.5)
    d.detach(); s.close()
    d.log("done")
    z = d.bundle(tmp_path / "b.zip", extra={"settings.json": "{}"})
    d.close()
    names = set(zipfile.ZipFile(z).namelist())
    assert {"debug.log", "header.json", "settings.json"} <= names
    assert any(n.startswith("rx_") and n.endswith(".wav") for n in names)
    assert any(n.startswith("tx_") and n.endswith(".wav") for n in names)
    log = zipfile.ZipFile(z).read("debug.log").decode()
    assert "before the file was open" in log and "recording RX audio" in log
    tx = next(p for p in d.dir.iterdir() if p.name.startswith("tx_"))
    with wave.open(str(tx)) as w:
        assert w.getnframes() >= 24000 and w.getframerate() == 48000
    rx = next(p for p in d.dir.iterdir() if p.name.startswith("rx_"))
    with wave.open(str(rx)) as w:
        assert w.getnframes() >= 24000                        # the loop returned what we played


def test_ui_settings_survive_restart_and_tab_switch(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from modemgui.audio import Registry
    from modemgui.engine import Session
    from modemgui.ui.main_window import MainWindow
    from modemgui.ui.settings_dialog import SettingsDialog
    QApplication.instance() or QApplication([])
    ini = str(tmp_path / "s.ini")

    def make():
        return MainWindow(Session(Registry()), QSettings(ini, QSettings.IniFormat))

    w = make()
    dlg = SettingsDialog(w.app)
    dlg.pages["css"].w["symbol_dur"].setValue(0.25)
    dlg.pages["css"].w["conv_code"].setCurrentIndex(1)
    dlg.pages["ofdm"].w["n_fft"].setCurrentIndex(2)
    dlg.pages["general"].w["modem"].setCurrentIndex(dlg.pages["general"].w["modem"].findData("ofdm") if dlg.pages["general"].w["modem"].findData("ofdm") >= 0 else 1)
    dlg.pages["diag"].w["wav_rx"].setChecked(True)
    dlg.pages["diag"].w["out_dir"].setText(str(tmp_path / "diag"))
    dlg.apply(); w.app.save(); w._after_settings()
    for i in range(len(w.panes)):                          # switching panes must not reset anything
        w.show_pane(i)
    assert (w.app.css.symbol_dur, w.app.ofdm.n_fft) == (0.25, 2048)
    w.show_pane(2)
    w.close()
    w2 = make()
    assert w2.app.css.symbol_dur == 0.25 and w2.app.css.conv_code == "k7r12"
    assert w2.app.ofdm.n_fft == 2048 and w2.app.general.modem == "ofdm"
    assert w2.app.diag.wav_rx is True and w2.app.diag.out_dir == str(tmp_path / "diag")
    assert w2.current_pane() == 2
    w2.close()


def test_ham_settings_persist_and_reach_the_session(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from modemgui.ui.appsettings import AppSettings
    qs = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    a = AppSettings(qs)
    a.ham = HamConfig(enabled=True, callsign="K1ABC", id_mode="both", id_interval_s=300.0,
                      id_end_delay_s=3.0, cw_wpm=20.0)
    a.save()
    b = AppSettings(QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    assert b.ham == a.ham and b.ham.enabled is True


def test_help_about_names_ne0k_modem(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QMessageBox
    from modemgui.audio import Registry
    from modemgui.engine import Session
    from modemgui.ui.main_window import MainWindow
    QApplication.instance() or QApplication([])
    w = MainWindow(Session(Registry()), QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    assert w.windowTitle().startswith("NE0K Modem")
    help_menu = [a.menu() for a in w.menuBar().actions() if a.text() == "&Help"][0]
    assert any("About" in a.text() for a in help_menu.actions())
    shown = []
    monkeypatch.setattr(QMessageBox, "about", lambda parent, title, text: shown.append((title, text)))
    w._about()
    assert shown[0][1].startswith("NE0K Modem\nCopyright 2026 XP5.ORG\nMIT License")


def test_serial_port_settings_menu_and_bridge(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from modemgui.audio import Registry
    from modemgui.engine import Session
    from modemgui.ui.main_window import MainWindow
    from modemgui.ui.settings_dialog import SettingsDialog
    QApplication.instance() or QApplication([])
    ini = str(tmp_path / "s.ini")
    w = MainWindow(Session(Registry()), QSettings(ini, QSettings.IniFormat))
    assert w.serial_action is w.column.uart.port_btn.defaultAction()
    dlg = SettingsDialog(w.app)
    link = str(tmp_path / "modem0")
    dlg.pages["serial"].w["link_path"].setText(link)
    dlg.pages["serial"].w["buffer"].setValue(64)
    dlg.apply(); w.app.save()
    w.session.open("loop:ab", "loop:ab")
    w._on_status(type("S", (), {"state": "open", "rx": "loop:ab", "tx": "loop:ab", "message": ""})())
    w._toggle_serial()
    assert w.serial is not None and os.path.islink(link) and w.serial.buf_size == 64
    w._toggle_serial()
    assert w.serial is None and not os.path.exists(link)
    w.close()
    w2 = MainWindow(Session(Registry()), QSettings(ini, QSettings.IniFormat))
    assert w2.app.serial.buffer == 64 and w2.app.serial.link_path == link
    w2.close()


def _window(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from modemgui.audio import Registry
    from modemgui.engine import Session
    from modemgui.ui.main_window import MainWindow
    QApplication.instance() or QApplication([])
    return MainWindow(Session(Registry()), QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))


def test_menus_follow_the_sheet(tmp_path):
    w = _window(tmp_path)
    assert [a.text() for a in w.menuBar().actions()] == ["&File", "&Edit", "&Settings", "&View", "&Help"]
    menu = lambda t: [a.menu() for a in w.menuBar().actions() if a.text() == t][0]
    assert [a.text() for a in menu("&File").actions() if a.text()] == [
        "&Open Settings...", "&Save Settings...", "E&xit"]
    assert [a.text() for a in menu("&Settings").actions()] == [
        "&Output Device Settings...", "&Modem Mode Settings...", "&CSS Modem Settings...", "O&FDM Modem Settings...",
        "&Ident Settings...", "&Debug && Logging..."]
    w.close()


def test_dialog_has_the_six_pages_and_opens_on_the_requested_one(tmp_path):
    from modemgui.ui.settings_dialog import SettingsDialog
    w = _window(tmp_path)
    dlg = SettingsDialog(w.app, w, "diag", io=w.io)
    assert [dlg.tabs.tabText(i) for i in range(dlg.tabs.count())] == [
        "Output Device", "Modem Mode", "CSS Modem", "OFDM Modem", "Ident", "Debug && Logging"]
    assert dlg.tabs.tabText(dlg.tabs.currentIndex()) == "Debug && Logging"
    assert {"wav_rx", "wav_tx", "detail", "log_serial"} <= set(dlg.pages["diag"].w)
    assert "link_path" in dlg.pages["serial"].w            # serial lives on the Debug & Logging page
    assert "ident_mode" in dlg.pages["ham"].w
    w.close()


def test_output_page_responding_test_goes_green_or_red(tmp_path):
    import socket
    from modemgui.ui.settings_dialog import SettingsDialog, _set
    w = _window(tmp_path)
    dlg = SettingsDialog(w.app, w, "output", io=w.io)
    out = dlg.pages["output"].w
    _set(out["io_type"], "gnuradio"); _set(out["location"], "remote")
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    _set(out["host"], "127.0.0.1"); _set(out["port"], srv.getsockname()[1])
    assert dlg.probe_output()[0] is True
    dlg.test_responding()
    assert "1a7f37" in dlg.resp.styleSheet()
    srv.close()
    assert dlg.probe_output()[0] is False
    _set(out["location"], "local")
    assert dlg.probe_output() == (False, "gnuradio connector not available yet")
    w.close()


def test_save_and_open_settings_file_round_trip(tmp_path, monkeypatch):
    import json
    from PySide6.QtWidgets import QFileDialog
    w = _window(tmp_path)
    f = str(tmp_path / "cfg.json")
    w.app.css.symbol_dur = 0.4
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (f, ""))
    w._save_settings_file()
    assert json.load(open(f))["css"]["symbol_dur"] == 0.4
    w.app.css.symbol_dur = 0.1
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (f, ""))
    w._open_settings_file()
    assert w.app.css.symbol_dur == 0.4
    w.close()


def test_numeric_ident_mode_silences_the_callsign_and_basic_detail_shortens_log(tmp_path):
    from modemgui.engine import Decoded
    w = _window(tmp_path)
    w.app.ham.enabled = True; w.app.ham.callsign = "K1ABC"
    w._apply_view_settings()
    assert w.session.ident.cfg.enabled is True
    w.app.ham.ident_mode = "numeric"; w.app.diag.detail = "basic"
    w._apply_view_settings()
    assert w.session.ident.cfg.enabled is False and w.feed.detailed is False
    w.feed.on_decoded(Decoded("hi", 1, 15, 0, "DATA", True, 1, 0, 30.0, 0))
    w.close()
