"""Every setting has built-in help, and the settings dialog tags its widgets with the same keys."""
import os

from modemgui.engine import helptext


def test_every_setting_has_impact_and_ofdm_text():
    missing = [k for k in helptext.keys() if not all(helptext.topic(k)[f] for f in ("impact", "ofdm"))]
    assert not missing


def test_no_stale_help_keys():
    assert set(helptext.HELP) <= set(helptext.keys())


def test_html_has_anchor_per_setting():
    page = helptext.html()
    assert all(f"name='{k}'" in page for k in helptext.keys())


def test_dialog_widgets_carry_help_keys():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from modemgui.ui.appsettings import AppSettings
    from modemgui.ui.inspector import help_key
    from modemgui.ui.settings_dialog import SettingsDialog
    QApplication.instance() or QApplication([])
    dlg = SettingsDialog(AppSettings(QSettings("/tmp/helptest.ini", QSettings.IniFormat)))
    for sec, pg in dlg.pages.items():
        for name, w in pg.w.items():
            assert help_key(w) == f"{sec}.{name}"
    assert set(helptext.keys()) == {f"{s}.{n}" for s, p in dlg.pages.items() for n in p.w}
