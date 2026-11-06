# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""All user settings in one place, persisted in QSettings.

Every dataclass field is stored under ``<section>/<field>`` -- the dialog, the engine and the
disk all follow the same field list, so adding a setting to ``modemcfg`` is the whole job.
Switching tabs never touches them (they live here, not in the widgets); they are written when
the dialog is accepted and again on close."""
from __future__ import annotations

import dataclasses

from PySide6.QtCore import QSettings

from ..engine.modemcfg import CssConfig, DiagConfig, GeneralConfig, HamConfig, OfdmConfig, OutputConfig, SerialConfig

SECTIONS = {"general": GeneralConfig, "output": OutputConfig, "css": CssConfig, "ofdm": OfdmConfig, "ham": HamConfig, "serial": SerialConfig, "diag": DiagConfig}


class AppSettings:
    def __init__(self, qs: QSettings) -> None:
        self.qs = qs
        self.general: GeneralConfig
        self.output: OutputConfig
        self.css: CssConfig
        self.ofdm: OfdmConfig
        self.ham: HamConfig
        self.serial: SerialConfig
        self.diag: DiagConfig
        for name, cls in SECTIONS.items():
            setattr(self, name, self._load(name, cls))

    def _load(self, section: str, cls):
        stored = {}
        for f in dataclasses.fields(cls):
            v = self.qs.value(f"{section}/{f.name}")
            if v is not None:
                stored[f.name] = v
        return cls.from_dict(stored)

    def save(self) -> None:
        for name in SECTIONS:
            for k, v in getattr(self, name).to_dict().items():
                self.qs.setValue(f"{name}/{k}", v)
        self.qs.sync()

    def dump(self) -> dict:
        return {name: getattr(self, name).to_dict() for name in SECTIONS}

    def load_dict(self, d: dict) -> None:
        """Replace the sections present in ``d`` (a file written by ``dump``); unknown keys are ignored."""
        for name, cls in SECTIONS.items():
            if isinstance(d.get(name), dict):
                setattr(self, name, cls.from_dict(d[name]))
