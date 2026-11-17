# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
from __future__ import annotations

import sys


def main(rx: str | None = None, tx: str | None = None, autosend: str | None = None,
         hear_own: bool = False, settings_file: str | None = None,
         control_port: int | None = None, control_bind: str = "127.0.0.1") -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"error: PySide6 is required for the GUI ({e}); pip install PySide6 pyqtgraph",
              file=sys.stderr)
        return 2
    from .. import APP_NAME
    from .main_window import MainWindow
    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName("XP5-ORG"); app.setApplicationName("modemgui")   # settings key, kept so saved settings survive the rename
    app.setApplicationDisplayName(APP_NAME)
    w = MainWindow()
    if settings_file:                    # settings JSON (the Save settings format) applied for this run only, not stored
        import json
        from pathlib import Path
        w.app.load_dict(json.loads(Path(settings_file).read_text()))
        w._after_settings()
    w.css_cfg.hear_own = hear_own
    if hear_own:                         # self-loop (demo): address ourselves so the ACK comes back
        w.app.general.peer_id = w.app.general.node_id
        w._show_mode()
    if rx or tx:                         # command-line override: open immediately
        w.io.select(rx, tx)
        try:
            w.session.open(rx, tx)
            w._power_on()                # a scripted launch (tests, demos) is a powered-on modem
        except Exception as e:
            w.statusBar().showMessage(f"open failed: {e}")
    w.show()
    if control_port is not None:         # --control-port: on for this run only, not stored
        w.app.remote.enabled, w.app.remote.port, w.app.remote.bind = True, control_port, control_bind
    err = w._apply_remote()              # the control port, from the flag or from Settings > Remote Control
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if autosend:                         # demo hook: send once the window is up
        from PySide6.QtCore import QTimer

        def go():
            w.send_text(autosend)
        QTimer.singleShot(1500, go)
    return app.exec()
