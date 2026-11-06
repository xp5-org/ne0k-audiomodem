# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
from __future__ import annotations

import dataclasses
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QSettings, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QDockWidget, QFileDialog, QMainWindow, QMessageBox, QStatusBar, QTabWidget, QToolBar)

from ..audio import SINK, SOURCE
from .. import APP_NAME, __version__, native
from ..engine import OfdmData, Session
from ..engine.diag import Diagnostics
from ..engine.mesh_node import MeshConfig, MeshNode
from .appsettings import AppSettings
from .bridge import EngineBridge
from .control_column import ControlColumn
from .inspector import Inspector, name_widgets
from .io_panel import IoPanel
from .lockbar import LockBar
from .log_view import LogView
from .modem_feed import ModemFeed
from .pianoroll import LinkLog, PianoRoll
from .probe_view import ProbePanel
from .settings_dialog import SettingsDialog
from .spectro_view import SpectroView
from .statusstrip import StatusStrip


STATE_VERSION = 5                          # bump when the dock set changes (tabs -> docks 1 -> 2, BAT dock 2 -> 3, status strip 4 -> 5)


def _echo(line: str) -> None:
    """Also write a line to the process stdout (the run's modemgui.log), timestamped like the CLI path does."""
    try:
        print(f"{time.strftime('%H:%M:%S')} {line}", flush=True, file=sys.__stdout__)
    except Exception:
        pass


class _LoggedStatusBar(QStatusBar):
    """Every status-bar message (errors such as 'Mesh link: OFDM_ONLY VIOLATION ...' included) is also logged:
    to the diagnostics log and stdout, so nothing is shown on screen without leaving a trace in the log files."""

    def __init__(self, parent, log) -> None:
        super().__init__(parent)
        self._log = log

    def showMessage(self, text: str, timeout: int = 0) -> None:
        if text:
            self._log("status: " + text)
        super().showMessage(text, timeout)


class MainWindow(QMainWindow):

    def __init__(self, session: Session | None = None, settings: QSettings | None = None):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1200, 760)
        self.settings = settings or QSettings("XP5-ORG", "modemgui")
        self.app = AppSettings(self.settings)
        self.link: MeshNode | None = None     # css = control plane only, ofdm = CSS + on-demand OFDM links
        self.serial = None                    # engine.serialbridge.SerialBridge (POSIX only)
        self.session = session or Session()
        self.bridge = EngineBridge(self.session, parent=self)
        self.diag = Diagnostics(self.app.diag, header={"modemgui": __version__,
                                                       "platform": platform.platform()})
        self.logview = LogView()
        self.diag.on_line = self.logview.add
        try:
            for v in native.inventory():
                self.diag.header[f"c lib {v['lib']}"] = "ok" if v["ok"] else "ERR"
        except Exception as e:
            self.diag.header["c libs"] = f"unavailable: {e}"
        self.diag.start_log()
        self.setStatusBar(_LoggedStatusBar(self, lambda m: (self.diag.log(m), _echo(m))))
        self.session.ident.cfg, self.session.ident.log = self.app.ham, self.diag.log

        self.io = IoPanel(self.session, self.settings)
        dock = QDockWidget("I/O", self)
        dock.setObjectName("dock-io")
        self.io_dock = dock
        self.column = ControlColumn(self.io)
        dock.setWidget(self.column)
        self.column.setMinimumWidth(300)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self.linklog = LinkLog()
        self.spectro = SpectroView(log=self.linklog)
        self.spectro.col_dt = self.session._stft_args[1] / self.session.spec.rate
        self.pianoroll = PianoRoll(self.linklog)
        self.pll = LockBar()               # one PLL bar for whichever modem is active
        pll_bar = QToolBar("PLL", self)
        pll_bar.setObjectName("bar-pll"); pll_bar.setMovable(False)
        pll_bar.addWidget(self.pll)
        self.addToolBar(Qt.BottomToolBarArea, pll_bar)
        self.strip = StatusStrip(self.session)
        top_bar = QToolBar("Status", self)
        top_bar.setObjectName("bar-status"); top_bar.setMovable(False)
        top_bar.addWidget(self.strip)
        self.addToolBar(Qt.TopToolBarArea, top_bar)
        self.strip.start()
        self.bat = ProbePanel()
        self.feed = ModemFeed(self.pll, self.logview.add, self)
        # Every pane is a dock: drag a title bar to tile it beside/above/below another, onto
        # another pane to tab with it, or out of the window to float it.
        self.setDockNestingEnabled(True)
        self.setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.North)
        self.setDocumentMode(True)
        self.panes: list[QDockWidget] = []
        self._front = 0                    # last pane raised to the top of its tab group
        for title, w in (("Spectrograph", self.spectro), ("Event Log", self.pianoroll),
                         ("BAT", self.bat), ("Log", self.logview)):
            d = QDockWidget(title, self)
            d.setObjectName("dock-" + title.lower())
            d.setWidget(w)
            d.visibilityChanged.connect(lambda vis, i=len(self.panes): vis and setattr(self, "_front", i))
            self.panes.append(d)
        self._reset_layout()

        self.bridge.status.connect(self.io.on_status)
        self.bridge.levels.connect(self.io.on_levels)
        self.bridge.levels.connect(self.strip.on_levels)
        self.bridge.spectrum.connect(self.spectro.on_spectrum)
        self.bridge.status.connect(lambda st: st.state == "closed" and self.spectro.clear())
        self.bridge.decoded.connect(self._mesh_event)       # the mesh control plane reads CSS packets
        self.bridge.link.connect(self._mesh_event)
        self.bridge.decoded.connect(self.feed.on_decoded)
        self.bridge.probe.connect(self.bat.on_probe)
        self.bridge.ofdm_data.connect(self.feed.on_data)
        self.bridge.decoded.connect(lambda d: self.column.link_info.record_rx(len(d.payload) or len(d.text)))
        self.bridge.ofdm_data.connect(lambda d: self.column.link_info.record_rx(d.nbytes))
        self._info_timer = QTimer(self); self._info_timer.setInterval(500)
        self._info_timer.timeout.connect(self._refresh_info); self._info_timer.start()
        self.bridge.link.connect(self.pianoroll.on_link)
        self.bridge.link.connect(self.spectro.on_link)
        self.bridge.link.connect(self.feed.on_link)
        self.bridge.status.connect(self._on_status)
        self.io.message.connect(self.statusBar().showMessage)
        self.statusBar().addPermanentWidget(self._backend_label())

        self._build_menus(dock)
        name_widgets(self)
        self.inspector = Inspector(self)
        self.inspect_action.setShortcut("Ctrl+Shift+I")
        self.inspect_action.toggled.connect(self.inspector.toggle)
        self.feed.set_link(None)
        self._apply_view_settings()
        self._show_mode()

        # What the user had last time: window, docks, tab, draft message, overlay.
        geo = self.settings.value("window/geometry")
        if geo:
            self.restoreGeometry(geo)
        state = self.settings.value("window/state")
        if state:
            self.restoreState(state, STATE_VERSION)
        try:
            self.show_pane(int(self.settings.value("ui/tab", 0)))
        except (TypeError, ValueError):
            pass
        self.spectro.overlay.setChecked(str(self.settings.value("ui/overlay", "false")).lower() == "true")
        self.spectro.overlay.toggled.connect(lambda on: self.settings.setValue("ui/overlay", on))

    def _show_mode(self) -> None:
        g = self.app.general
        self.column.set_mode(f"{g.modem.upper()}, {'point to point' if g.topology == 'p2p' else 'mesh'}, "
                             f"node {g.node_id}" + (f" -> {g.peer_id}" if g.topology == "p2p" else ", beacons " + ("on" if g.beacon else "off")))

    def _refresh_info(self) -> None:
        m = self.link
        detail = ""
        if m is not None:
            links = ", ".join(f"{k} {v}" for k, v in m.link_states().items()) or "none"
            near = ", ".join(f"{n} ({snr:.0f} dB)" for n, (snr, _) in sorted(m.neighbors.items())) or "none"
            phy_label = {"css_only": "CSS only", "ofdm_only": "OFDM only"}.get(m.mesh.phy_mode, "CSS + OFDM")
            detail = (f"node {m.me}, {phy_label}; "
                      f"heard: {near}; beacons sent {m.beacons_tx}; "
                      f"OFDM links: {links}; tx {m.tx_count} / rx {m.rx_count}, "
                      f"CSS fallbacks {m.css_fallbacks}, requests {m.reqs}")
            if m.ofdm_only_violations:
                detail = (f"⚠ OFDM_ONLY VIOLATION ×{len(m.ofdm_only_violations)}: "
                          f"{m.ofdm_only_violations[-1]} ⚠\n") + detail
                self.column.link_info.detail.setStyleSheet("color: red; font-weight: bold")
            else:
                self.column.link_info.detail.setStyleSheet("")
        self.column.link_info.refresh(detail)
        self.column.uart.refresh(self.serial)

    def _reset_layout(self) -> None:
        """Default arrangement (css_modem_UI.xls, second mock): I/O on the left, Spectrograph /
        Event Log / BAT / Log stacked on the right."""
        for d in self.panes:
            d.setFloating(False)
            self.addDockWidget(Qt.RightDockWidgetArea, d)
            d.show()
        for a, b in zip(self.panes, self.panes[1:]):
            self.splitDockWidget(a, b, Qt.Vertical)
        self.resizeDocks([self.io_dock], [320], Qt.Horizontal)     # control column is narrow

    def _tab_all(self) -> None:
        for d in self.panes:
            d.setFloating(False)
            self.addDockWidget(Qt.RightDockWidgetArea, d)
            d.show()
        for d in self.panes[1:]:
            self.tabifyDockWidget(self.panes[0], d)
        self.panes[0].raise_()

    def reset_layout(self) -> None:
        self.io_dock.show(); self.io_dock.setFloating(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.io_dock)
        self._reset_layout()

    def arrange(self, mode: str) -> None:
        """Preset arrangements of the six panes; drag title bars from there for anything else."""
        if mode == "tabbed":
            return self._tab_all()
        if mode == "stacked":
            return self._reset_layout()
        for d in self.panes:
            d.setFloating(False)
            self.addDockWidget(Qt.RightDockWidgetArea, d)
            d.show()
        first = self.panes[0]
        if mode in ("vertical", "horizontal"):
            o = Qt.Vertical if mode == "vertical" else Qt.Horizontal
            for a, b in zip(self.panes, self.panes[1:]):
                self.splitDockWidget(a, b, o)
        else:                                         # two columns, three rows
            left, right = self.panes[0::2], self.panes[1::2]
            self.splitDockWidget(left[0], right[0], Qt.Horizontal)
            for col in (left, right):
                for a, b in zip(col, col[1:]):
                    self.splitDockWidget(a, b, Qt.Vertical)

    def show_pane(self, i: int) -> None:
        d = self.panes[i]
        d.show(); d.raise_()
        self._front = i

    def current_pane(self) -> int:
        return self._front

    # compat with the CSS-only window: the live CSS settings
    @property
    def css_cfg(self):
        return self.app.css

    # -- menus -------------------------------------------------------------

    def _build_menus(self, io_dock) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addAction("&Open Settings...", self._open_settings_file, "Ctrl+O")
        m.addAction("&Save Settings...", self._save_settings_file, "Ctrl+S")
        m.addSeparator()
        m.addAction("E&xit", self.close, "Ctrl+Q")

        m = mb.addMenu("&Edit")
        m.addAction("&Copy status", lambda: QGuiApplication.clipboard().setText(
            self.statusBar().currentMessage()), "Ctrl+C")
        m.addAction("Send &ID now", self._send_id_now, "Ctrl+I")

        m = mb.addMenu("&Settings")
        self.settings_actions = {}
        for key, text in (("output", "&Output Device Settings..."), ("mode", "&Modem Mode Settings..."), ("css", "&CSS Modem Settings..."),
                          ("ofdm", "O&FDM Modem Settings..."), ("ham", "&Ident Settings..."),
                          ("diag", "&Debug && Logging...")):
            self.settings_actions[key] = m.addAction(text, lambda k=key: self._open_settings(k))

        # The virtual serial port is opened from the Serial uart info group on the front page.
        self.serial_action = QAction("Open virtual serial port", self)
        self.serial_action.setCheckable(True); self.serial_action.setShortcut("Ctrl+P")
        self.serial_action.triggered.connect(lambda _=False: self._toggle_serial())
        if not hasattr(os, "openpty"):                  # Windows has no pty
            self.serial_action.setEnabled(False)
            self.serial_action.setToolTip("Virtual serial port needs a POSIX pty (Linux, macOS, FreeBSD).")
        self.addAction(self.serial_action)
        self.column.uart.set_action(self.serial_action)

        m = mb.addMenu("&View")
        m.addAction(io_dock.toggleViewAction())
        ov = m.addAction("Event &overlay on spectrograph")
        ov.setCheckable(True)
        ov.toggled.connect(self.spectro.overlay.setChecked)
        self.spectro.overlay.toggled.connect(ov.setChecked)
        m.addAction("&Clear events", self.pianoroll.clear)
        m.addSeparator()
        for d in self.panes:
            m.addAction(d.toggleViewAction())
        lm = m.addMenu("&Layout")
        for label, mode in (("&Default (stacked + tabs)", "stacked"), ("&All tabbed", "tabbed"),
                            ("Stacked &vertically", "vertical"),
                            ("Side by &side", "horizontal"), ("&Grid (2 columns)", "grid")):
            lm.addAction(label, lambda mode=mode: self.arrange(mode))
        m.addAction("&Reset layout", self.reset_layout)
        m.addSeparator()
        self.inspect_action = m.addAction("&Inspect UI names")
        self.inspect_action.setCheckable(True)

        m = mb.addMenu("&Help")
        m.addAction("&C library inventory", self._show_inventory)
        m.addAction("Open &capture folder", self._open_capture_dir)
        m.addAction("Save &diagnostic bundle...", self._save_bundle)
        m.addAction(f"&About {APP_NAME}", self._about)

    # -- settings ----------------------------------------------------------

    def _open_settings(self, page: str | None = None) -> None:
        dlg = SettingsDialog(self.app, self, page if isinstance(page, str) else None, io=self.io)
        if dlg.exec():
            dlg.apply()
            self.app.save()
            self._after_settings()
            self.statusBar().showMessage("Settings applied.")

    def _after_settings(self) -> None:
        self._apply_diag()
        self._show_mode()
        if self.serial:
            self._stop_serial()
        self._restart_link()

    def _send_id_now(self) -> None:
        if not self.session.is_open:
            self.statusBar().showMessage("Open audio first.")
            return
        try:
            air = self.session.ident.send_now("manual")
            self.statusBar().showMessage(f"ID sent, {air:.1f} s on air.")
        except Exception as e:
            self.statusBar().showMessage(f"ID not sent: {e} (Settings > Ident Settings)")

    def _open_settings_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open settings", str(Path.home()), "Settings (*.json)")
        if not path:
            return
        try:
            self.app.load_dict(json.loads(Path(path).read_text()))
        except (OSError, ValueError) as e:
            self.statusBar().showMessage(f"Could not open settings: {e}")
            return
        self.app.save()
        self._after_settings()
        self.statusBar().showMessage(f"Settings loaded from {path}")

    def _save_settings_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", str(Path.home() / "modemgui-settings.json"),
                                              "Settings (*.json)")
        if path:
            Path(path).write_text(json.dumps(self.app.dump(), indent=2))
            self.statusBar().showMessage(f"Settings saved to {path}")

    def _apply_view_settings(self) -> None:
        self.feed.detailed = self.app.diag.detail == "detailed"
        # numeric mode: only the 0-15 node ids identify stations, nothing is announced
        self.session.ident.cfg = dataclasses.replace(
            self.app.ham, enabled=self.app.ham.enabled and self.app.ham.ident_mode == "ham")

    def _apply_diag(self) -> None:
        self.diag.cfg = self.app.diag
        self._apply_view_settings()
        if self.app.diag.debug_log:
            self.diag.start_log()
        else:
            self.diag.stop_log()
        if self.session.is_open:
            self.diag.attach(self.session)

    def _open_capture_dir(self) -> None:
        d = Path(self.app.diag.out_dir).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    def _save_bundle(self) -> None:
        name = time.strftime("modemgui-diag-%Y%m%d_%H%M%S.zip")
        path, _ = QFileDialog.getSaveFileName(self, "Save diagnostic bundle",
                                              str(Path(self.app.diag.out_dir).expanduser() / name),
                                              "Zip (*.zip)")
        if not path:
            return
        out = self.diag.bundle(path, extra=self._bundle_extra())
        self.statusBar().showMessage(f"Diagnostic bundle saved: {out}")

    def _bundle_extra(self) -> dict:
        try:
            inv = json.dumps(native.inventory(), indent=2, default=str)
        except Exception as e:
            inv = f"unavailable: {e}"
        return {"settings.json": json.dumps(self.app.dump(), indent=2, default=str),
                "c_libraries.json": inv}

    # -- virtual serial port ---------------------------------------------------

    def _toggle_serial(self) -> None:
        if self.serial:
            self._stop_serial()
            self._restart_link()
        else:
            self._start_serial()

    def _start_serial(self) -> None:
        if not self.session.is_open:
            self.serial_action.setChecked(False)
            self.statusBar().showMessage("Open audio first.")
            return
        from ..engine.serialbridge import SerialBridge
        sc = self.app.serial
        if self.link:                                  # the bridge starts and stops its own mesh node
            self.link.stop(); self.link = None
            self.feed.set_link(None)
        try:
            mesh = self._make_mesh()
            br = SerialBridge(self.session, mesh, str(Path(sc.link_path).expanduser()), sc.buffer,
                              sc.mode, sc.idle_s, log=lambda m: self.app.diag.log_serial and (self.diag.log("serial: " + m), _echo("serial: " + m)),
                              ack=sc.ack, retries=sc.retries, dst=lambda: self._mesh_dst(mesh))
            to_host = mesh.on_data                     # the bridge's receive queue; the Log hears it too
            mesh.on_data = lambda src, d: (self._mesh_rx(src, d), to_host(src, d))
            if sc.rx_host:
                br.rx_filter = lambda src, h=sc.rx_host: src == h
            br.start()
            self.link = mesh
            self._listen_rx_host(mesh)
        except Exception as e:
            self.diag.log(f"serial port failed: {type(e).__name__}: {e}")
            self.serial_action.setChecked(False)
            self.statusBar().showMessage(f"Serial port: {e}")
            self._restart_link()
            return
        self.serial = br
        self.serial_action.setChecked(True)
        self.statusBar().showMessage(f"Serial port open: {br.link_path} (RTS/CTS on {br.ctl_path})")

    def _stop_serial(self) -> None:
        br, self.serial = self.serial, None
        self.serial_action.setChecked(False)
        if br:
            br.stop()
            self.link = None
            self.statusBar().showMessage(f"Serial port closed (tx {br.tx_count} B, rx {br.rx_count} B).")

    # -- links -------------------------------------------------------------

    def _make_mesh(self) -> MeshNode:
        g = self.app.general
        phy = g.modem if g.modem in ("ofdm_only", "css_only") else \
            ("css_then_ofdm" if g.modem == "ofdm" else "css_only")
        mc = MeshConfig(node_id=g.node_id, phy_mode=phy, beacon=g.topology == "mesh" and g.beacon,
                        beacon_min_s=g.beacon_min_s, beacon_max_s=g.beacon_max_s, first_beacon_s=g.first_beacon_s,
                        cw_s=g.cw_s, ack_stagger_s=g.ack_stagger_s, ofdm_detect=g.ofdm_detect,
                        airtime_windows=g.airtime_windows, hs_window_s=g.hs_window_s, hs_slots=g.hs_slots,
                        hs_guard_s=g.hs_guard_s, announce_round=g.announce_round,
                        announce_bat_turn_s=g.announce_bat_turn_s, announce_cycle_s=g.announce_cycle_s,
                        announce_bat_mode=g.announce_bat_mode)
        mesh = MeshNode(self.session, mc, self.app.css,
                        self.app.ofdm, log=lambda m: (self.diag.log("mesh: " + m), _echo("mesh: " + m)), own_events=False)
        mesh.on_data = self._mesh_rx
        mesh.ring_dst_fn = lambda m=mesh: self._mesh_dst(m)
        mesh.rx_peer = self.app.serial.rx_host
        return mesh

    def _mesh_dst(self, mesh: MeshNode) -> int | None:
        """p2p: the configured peer.  Mesh: a pinned peer if set, else the strongest neighbour heard."""
        g = self.app.general
        if g.topology == "p2p":
            return g.peer_id
        return g.mesh_dst or mesh.best_neighbor()

    def _mesh_rx(self, src: int, data: bytes) -> None:
        self.session._emit(OfdmData(len(data), data.decode("utf-8", "replace"), bytes(data)))

    def _mesh_event(self, ev) -> None:
        if self.link is not None:
            self.link.handle(ev)

    def _restart_link(self) -> None:
        if self.link:
            self.link.stop(); self.link = None
            self.feed.set_link(None)
        if not self.session.is_open or self.serial:
            return
        try:
            link = self._make_mesh()
            link.start()
        except Exception as e:
            self.diag.log(f"mesh failed to start: {type(e).__name__}: {e}")
            self.statusBar().showMessage(f"Mesh link: {e}")
            return
        self.link = link
        self.feed.set_link(link)
        self.diag.log(f"mesh node {link.me} started ({self.app.general.modem})")
        self._listen_rx_host(link)
        self._schedule_blockout(link)

    def _listen_rx_host(self, mesh: MeshNode) -> None:
        """ofdm_only, outside a blockout window: there is no broadcast to trigger a responder link
        the way _on_blockout_req does, and no CSS REQ either (that is the whole point of
        ofdm_only) -- so for the one peer this node's own ring position (serial.rx_host) already
        says it expects to hear from, listen from the start instead of waiting for a signal that
        is never coming.  Every path that (re)creates the mesh node must call this: the serial
        bridge builds a fresh node, and without it that node is an initiator with no responder
        anywhere on the ring, so no handshake can ever complete."""
        if mesh.mesh.phy_mode == "ofdm_only" and self.app.serial.rx_host:
            mesh._ensure_rx_link(self.app.serial.rx_host)

    def _schedule_blockout(self, mesh: MeshNode) -> None:
        """settings-driven blockout trial (general.blockout_role), same idea as the ham ident
        timer: the running GUI performs the real protocol action on its own, no manual click
        needed, so an unattended distributed run exercises it exactly as a user would -- see
        MeshNode.send_blockout / mesh_mac.Blockout."""
        g = self.app.general
        if g.topology != "mesh" or g.blockout_role == "off":
            return
        payload = g.blockout_payload.encode()
        if g.blockout_role == "announce":
            def fire() -> None:
                if self.link is mesh:          # the link may have been torn down/restarted by then
                    # announce_end=False: the peer may reply ('reply' role, the two-way case).
                    # Ending the window right after our own leg both cut that reply off and
                    # put our END on the air on top of the peer's first reply probe, so
                    # bystanders never decoded it.  The replier ends the window after its leg.
                    # If no reply comes, MeshNode._await_reply ends it for us.
                    threading.Thread(target=mesh.send_blockout, args=(g.blockout_peer, payload),
                                     kwargs={"duration_s": g.blockout_duration_s,
                                             "announce_end": False}, daemon=True).start()
            if g.blockout_go_file:
                # A distributed test group already knows (via its own barrier) when every member
                # is actually up and listening -- wait for ITS signal, not a guessed elapsed time
                # since this node's own mesh link started (see mesh_group_template.py's
                # "modems_up" barrier + the go-file step that follows it).
                def wait_for_go() -> None:
                    deadline = time.monotonic() + 300.0
                    while self.link is mesh and not os.path.exists(g.blockout_go_file):
                        if time.monotonic() > deadline:
                            self.diag.log(f"mesh: blockout go-file {g.blockout_go_file} never "
                                          "appeared -- giving up")
                            return
                        time.sleep(0.2)
                    if self.link is mesh:
                        fire()
                threading.Thread(target=wait_for_go, daemon=True).start()
            else:
                QTimer.singleShot(max(0, int(g.blockout_at_s * 1000)), fire)
        elif g.blockout_role == "reply":
            def wait_and_reply() -> None:
                deadline = time.monotonic() + g.blockout_duration_s + g.blockout_at_s + 60.0
                while self.link is mesh and not mesh.blockout.participant:
                    if time.monotonic() > deadline:
                        self.diag.log("mesh: blockout reply gave up -- never saw the announce")
                        return
                    time.sleep(0.5)
                if self.link is mesh and mesh.blockout.participant:
                    if mesh.mesh.phy_mode == "ofdm_only":
                        # ofdm_only: no CSS REQ/ACK is going to gate our own send against the
                        # announcer's leg still being in flight (see mesh_node.py's _send_ofdm/
                        # _on_blockout_req) -- wait for the announcer's actual payload to land
                        # on our own listener (opened the moment we became a participant), not
                        # just its handshake ("link up" fires before the data frame does), before
                        # taking the channel for our own reply -- single radio, one transmission
                        # at a time.
                        rx0 = mesh.rx_count
                        while self.link is mesh and time.monotonic() < deadline:
                            if mesh.rx_count > rx0:
                                break
                            time.sleep(0.5)
                    # drop the link we just received the announcer's leg on: single radio, one
                    # link at a time -- see MeshNode.release_link / _on_req.
                    mesh.release_link(g.blockout_peer)
                    mesh.send(g.blockout_peer, payload)
                    # Both legs are now done: resume the mesh for every bystander.
                    mesh.release_link(g.blockout_peer)
                    if mesh.mesh.phy_mode == "ofdm_only" and self.link is mesh:
                        mesh.end_blockout()
            threading.Thread(target=wait_and_reply, daemon=True).start()

    def _on_status(self, st) -> None:
        if st.state == "open":
            self.diag.log(f"audio open rx={self.io.describe(st.rx, SOURCE)} "
                          f"tx={self.io.describe(st.tx, SINK)} rate={self.session.spec.rate}")
            self.diag.log("settings: " + json.dumps(self.app.dump(), default=str))
            self.diag.attach(self.session)
            self._restart_link()
            if self.app.serial.autostart and not self.serial:
                self._start_serial()
        elif st.state == "closed":
            if self.serial:
                self._stop_serial()
            if self.link:
                self.link.stop(); self.link = None
                self.feed.set_link(None)
            self.diag.log("audio closed")
            self.diag.detach()
        elif st.state == "error":
            self.diag.log(f"engine error: {st.message}")
        self.column.set_locked(st.state == "open")
        self.feed.set_open(st.state == "open")
        self.pianoroll.set_active(st.state == "open")

    def send_text(self, text: str) -> None:
        """Transmit one message on the open link (the --autosend demo hook; the GUI has no send box)."""
        if self.serial:
            self.statusBar().showMessage("The serial port owns the link.")
            return
        if not self.link:
            self.statusBar().showMessage("Open audio first.")
            return
        data = text.encode("utf-8")
        link = self.link
        dst = self._mesh_dst(link)
        if dst is None:
            self.statusBar().showMessage("No mesh neighbour heard yet.")
            return

        def work() -> None:
            try:
                if link.send(dst, data):
                    self.column.link_info.record_tx(len(data))
                else:
                    self.diag.log(f"send to {dst} not delivered")
            except Exception as e:
                self.diag.log(f"send failed: {e}")
        threading.Thread(target=work, name="mesh-send", daemon=True).start()

    # -- help --------------------------------------------------------------

    def _show_inventory(self) -> None:
        try:
            rows = [f"{'OK ' if v['ok'] else 'ERR'} {v['lib']}" for v in native.inventory()]
        except Exception as e:
            rows = [f"unavailable: {e}"]
        QMessageBox.information(self, "C libraries", "\n".join(rows))

    def _about(self) -> None:
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"{APP_NAME}\nCopyright 2026 XP5.ORG\nMIT License\n\n"
                          f"Version {__version__}\n{self._backend_label().text()}")

    def _backend_label(self):
        from PySide6.QtWidgets import QLabel
        try:
            libs = native.inventory()
            ok = [v["lib"] for v in libs if v["ok"]]
            text = f"C libs: {len(ok)}/{len(libs)} loaded" + ("" if ok else " - Python fallback")
        except Exception as e:               # status bar must never block startup
            text = f"C libs: unavailable ({e})"
        return QLabel(text)

    def closeEvent(self, ev) -> None:
        if self.serial:
            self._stop_serial()
        if self.link:
            self.link.stop()
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState(STATE_VERSION))
        self.settings.setValue("ui/tab", self.current_pane())
        self.app.save()
        self.session.close()
        self.diag.close()
        super().closeEvent(ev)
