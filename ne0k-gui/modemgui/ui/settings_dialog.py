# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Settings dialog, generated from the config dataclasses' own field descriptions
(``modemcfg.describe``): one page per menu entry of css_modem_UI.xls, one group box per ``group``."""
from __future__ import annotations

import dataclasses
import socket

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from ..engine.modemcfg import OFDM_PRESETS, PRESETS, CssConfig, describe
from .appsettings import AppSettings

# Settings-menu pages -> the config sections each one edits.
TABS = (("output", "Output Device"), ("mode", "Modem Mode"), ("css", "CSS Modem"), ("ofdm", "OFDM Modem"),
        ("ham", "Ident"), ("diag", "Debug && Logging"))
SECTIONS_OF = {"diag": ("diag", "serial"), "mode": ("general",)}
KEEP_ON_PRESET = ("node_id", "peer_id", "role", "mode", "sender", "hear_own", "backend", "fs")


def _make(d: dict):
    """Widget for one described field."""
    kind, val = d["kind"], d["default"]
    if d.get("choices"):
        w = QComboBox()
        for c in d["choices"]:
            w.addItem(str(c), c)
        return w
    if kind is bool:
        return QCheckBox()
    if kind is int:
        w = QSpinBox(); w.setRange(int(d.get("lo") if d.get("lo") is not None else 0),
                                   int(d.get("hi") if d.get("hi") is not None else 1_000_000))
        return w
    if kind is float:
        w = QDoubleSpinBox(); w.setDecimals(3)
        w.setRange(d.get("lo") if d.get("lo") is not None else -1e6,
                   d.get("hi") if d.get("hi") is not None else 1e6)
        w.setSingleStep(d.get("step") or 0.1)
        return w
    return QLineEdit()


def _set(w, v) -> None:
    if isinstance(w, QComboBox):
        i = w.findData(v)
        if i < 0:
            i = w.findText(str(v))
        w.setCurrentIndex(max(i, 0))
    elif isinstance(w, QCheckBox):
        w.setChecked(bool(v))
    elif isinstance(w, QLineEdit):
        w.setText(str(v))
    else:
        w.setValue(v)


def _get(w):
    if isinstance(w, QComboBox):
        return w.currentData()
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QLineEdit):
        return w.text()
    return w.value()


class _Page(QWidget):
    """One section: preset row (optional) + groups of fields."""

    def __init__(self, cls, cfg, presets=None) -> None:
        super().__init__()
        self.cls, self.w = cls, {}
        outer = QVBoxLayout(self)
        self.preset = None
        if presets is not None:
            self.preset = QComboBox(); self.preset.addItem("(custom)"); self.preset.addItems(presets)
            row = QHBoxLayout(); row.addWidget(QLabel("Preset")); row.addWidget(self.preset, 1)
            outer.addLayout(row)
        inner = QWidget(); col = QVBoxLayout(inner)
        forms: dict[str, QFormLayout] = {}
        for d in describe(cls):
            form = forms.get(d["group"])
            if form is None:
                box = QGroupBox(d["group"]); form = QFormLayout(box); col.addWidget(box)
                forms[d["group"]] = form
            w = _make(d)
            if d.get("tip"):
                w.setToolTip(d["tip"])
            self.w[d["name"]] = w
            label = QLabel(d["label"]); label.setToolTip(d.get("tip") or "")
            if d["name"] == "out_dir":
                b = QPushButton("Browse..."); row = QHBoxLayout()
                row.addWidget(w, 1); row.addWidget(b)
                b.clicked.connect(lambda _=0, w=w: w.setText(
                    QFileDialog.getExistingDirectory(self, "Capture folder", w.text()) or w.text()))
                form.addRow(label, row)
            else:
                form.addRow(label, w)
        col.addStretch(1)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(inner)
        outer.addWidget(sc, 1)
        self.load(cfg)

    def load(self, cfg) -> None:
        for k, v in cfg.to_dict().items():
            _set(self.w[k], v)

    def config(self):
        return self.cls.from_dict({k: _get(w) for k, w in self.w.items()})


class SettingsDialog(QDialog):
    def __init__(self, app: AppSettings, parent=None, page: str | None = None, io=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(560, 640)
        self.app, self.io = app, io
        self.pages: dict[str, _Page] = {}
        tabs = self.tabs = QTabWidget()
        for key, title in TABS:
            holder = QWidget(); lay = QVBoxLayout(holder); lay.setContentsMargins(0, 0, 0, 0)
            for sec in SECTIONS_OF.get(key, (key,)):
                pg = _Page(type(getattr(app, sec)), getattr(app, sec),
                           PRESETS if sec == "css" else OFDM_PRESETS if sec == "ofdm" else None)
                self.pages[sec] = pg
                lay.addWidget(pg, 1)
            if key == "output":
                lay.addWidget(self._audio_mirror()); lay.addLayout(self._responding_row())
            tabs.addTab(holder, title)
        g = self.pages["general"].w
        g["topology"].currentIndexChanged.connect(self._topology_changed)
        self._topology_changed()
        self.pages["css"].preset.currentTextChanged.connect(self._css_preset)
        self.pages["ofdm"].preset.currentTextChanged.connect(self._ofdm_preset)
        if page in self.pages or page in dict(TABS):
            page = next((k for k, _ in TABS if page == k or page in SECTIONS_OF.get(k, (k,))), None)
            tabs.setCurrentIndex([k for k, _ in TABS].index(page))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._defaults)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Both ends of a link must use the same band, timing and PHY settings."))
        lay.addWidget(tabs, 1); lay.addWidget(bb)

    def _topology_changed(self) -> None:
        """A mesh has no configured peer and beacons; point to point has a peer and no beacons."""
        g = self.pages["general"].w
        mesh = _get(g["topology"]) == "mesh"
        g["peer_id"].setEnabled(not mesh)
        for k in ("beacon", "beacon_min_s", "beacon_max_s", "first_beacon_s",
                  "airtime_windows", "hs_window_s", "hs_slots"):
            g[k].setEnabled(mesh)

    def _audio_mirror(self) -> QGroupBox:
        """Read-only copy of the front-page audio choice; change it there."""
        box = QGroupBox("Audio (set on the front page)")
        form = QFormLayout(box)
        rx = self.io.rx.currentText() if self.io else "-"
        tx = self.io.tx.currentText() if self.io else "-"
        for name, text in (("In (RX)", rx), ("Out (TX)", tx),
                           ("Sample rate", f"{self.app.css.fs:g} Hz")):
            form.addRow(name, QLabel(text or "-"))
        return box

    def _responding_row(self) -> QHBoxLayout:
        self.resp_btn = QPushButton("Test connection")
        self.resp = QLabel("not tested")
        self.resp_btn.clicked.connect(self.test_responding)
        row = QHBoxLayout(); row.addWidget(self.resp_btn); row.addWidget(self.resp, 1)
        return row

    def probe_output(self) -> tuple[bool, str]:
        """Is the chosen output answering?  Audio: a device is selected.  GNURadio / libsdr
        remote: the TCP port accepts.  Local GNURadio / libsdr have no connector yet."""
        w = self.pages["output"].w
        kind, loc = _get(w["io_type"]), _get(w["location"])
        if kind == "audio":
            ok = bool(self.io) and all(c.currentText() not in ("", "(none)") for c in (self.io.rx, self.io.tx))
            return ok, "audio devices selected" if ok else "no audio device selected"
        if loc == "remote":
            host, port = _get(w["host"]), _get(w["port"])
            try:
                socket.create_connection((host, port), timeout=1.5).close()
                return True, f"{host}:{port} answers"
            except OSError as e:
                return False, f"{host}:{port}: {e}"
        return False, f"{kind} connector not available yet"

    def test_responding(self) -> None:
        ok, msg = self.probe_output()
        self.resp.setText(msg)
        self.resp.setStyleSheet(f"color:{'#1a7f37' if ok else '#c62828'};font-weight:bold")

    def _keep(self, pg: _Page) -> dict:
        cur = pg.config().to_dict()
        return {k: cur[k] for k in KEEP_ON_PRESET if k in cur}

    def _css_preset(self, name: str) -> None:
        if name in PRESETS:
            pg = self.pages["css"]
            pg.load(dataclasses.replace(PRESETS[name], **{k: v for k, v in self._keep(pg).items()
                                                          if k in CssConfig.__dataclass_fields__}))

    def _ofdm_preset(self, name: str) -> None:
        if name in OFDM_PRESETS:
            pg = self.pages["ofdm"]
            base = pg.cls().to_dict(); base.update(self._keep(pg)); base.update(OFDM_PRESETS[name])
            pg.load(pg.cls.from_dict(base))

    def _defaults(self) -> None:
        key = TABS[self.tabs.currentIndex()][0]
        for sec in SECTIONS_OF.get(key, (key,)):
            self.pages[sec].load(self.pages[sec].cls())

    def apply(self) -> None:
        """Copy the edited values into the AppSettings the dialog was built from."""
        for key, pg in self.pages.items():
            setattr(self.app, key, pg.config())
