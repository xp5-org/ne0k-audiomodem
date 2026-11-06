# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Finds connectors: built-ins, installed packages, and drop-in plugin files.

Plugin discovery, in order (later cannot override an earlier name):
  1. built-ins in this package;
  2. setuptools entry points in the group ``modemgui.connectors`` -- each
     resolves to a Connector subclass, or a callable returning a list of them;
  3. every ``*.py`` in the directories of ``MODEMGUI_PLUGIN_PATH`` (os.pathsep
     separated), each exposing ``CONNECTORS = [Cls, ...]`` or ``get_connectors()``.

A plugin that fails to import is recorded in ``Registry.errors`` and skipped; a
broken SDR driver must not take the sound card down with it.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from importlib import metadata
from pathlib import Path

from .base import AudioError, Connector, DeviceInfo, Sink, Source, StreamSpec

ENTRY_GROUP = "modemgui.connectors"
PLUGIN_ENV = "MODEMGUI_PLUGIN_PATH"

_BUILTIN_MODULES = ("file_backend", "pulse_backend", "sounddevice_backend")

# import_module() with a computed name is invisible to PyInstaller, so a frozen build would ship
# only the backends something else imports (file) and lose the system devices.  Naming them here
# as real imports makes the freezer collect them; each module tolerates a missing PortAudio/pactl.
from . import file_backend as _fb, pulse_backend as _pb, sounddevice_backend as _sb  # noqa: E402,F401


def _instances(obj) -> list[Connector]:
    if isinstance(obj, type) and issubclass(obj, Connector):
        return [obj()]
    if isinstance(obj, (list, tuple)):
        return [c for o in obj for c in _instances(o)]
    if isinstance(obj, Connector):
        return [obj]
    if callable(obj):
        return _instances(obj())
    raise TypeError(f"not a Connector or factory: {obj!r}")


class Registry:
    def __init__(self, builtins: bool = True, plugins: bool = True) -> None:
        self.connectors: dict[str, Connector] = {}
        self.errors: dict[str, str] = {}
        if builtins:
            for mod in _BUILTIN_MODULES:
                self._load_module(f"{__package__}.{mod}", mod)
        if plugins:
            self._load_entry_points()
            self._load_plugin_dirs()

    # -- loading ------------------------------------------------------------

    def add(self, c: Connector) -> None:
        if not c.name or ":" in c.name:
            raise ValueError(f"connector name {c.name!r} must be non-empty, no ':'")
        if c.name in self.connectors:
            raise ValueError(f"duplicate connector {c.name!r}")
        self.connectors[c.name] = c

    def _add_all(self, objs, label: str) -> None:
        try:
            for c in _instances(objs):
                self.add(c)
        except Exception as e:
            self.errors[label] = f"{type(e).__name__}: {e}"

    def _load_module(self, qualname: str, label: str) -> None:
        try:
            mod = importlib.import_module(qualname)
            self._add_all(getattr(mod, "get_connectors", None) or mod.CONNECTORS, label)
        except Exception as e:
            self.errors[label] = f"{type(e).__name__}: {e}"

    def _load_entry_points(self) -> None:
        try:
            eps = metadata.entry_points(group=ENTRY_GROUP)
        except Exception as e:                       # pragma: no cover
            self.errors["entry_points"] = str(e)
            return
        for ep in eps:
            try:
                self._add_all(ep.load(), f"ep:{ep.name}")
            except Exception as e:
                self.errors[f"ep:{ep.name}"] = f"{type(e).__name__}: {e}"

    def _load_plugin_dirs(self) -> None:
        for d in filter(None, os.environ.get(PLUGIN_ENV, "").split(os.pathsep)):
            for f in sorted(Path(d).glob("*.py")):
                label = f"plugin:{f.name}"
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"modemgui_plugin_{f.stem}", f)
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = mod
                    spec.loader.exec_module(mod)
                    self._add_all(getattr(mod, "get_connectors", None)
                                  or mod.CONNECTORS, label)
                except Exception as e:
                    self.errors[label] = f"{type(e).__name__}: {e}"

    # -- queries ------------------------------------------------------------

    def status(self) -> dict[str, tuple[bool, str]]:
        return {n: c.available() for n, c in self.connectors.items()}

    def list_devices(self, kind: str | None = None) -> list[DeviceInfo]:
        out: list[DeviceInfo] = []
        for c in self.connectors.values():
            ok, _ = c.available()
            if not ok:
                continue
            try:
                out += [d for d in c.list_devices() if kind in (None, d.kind)]
            except Exception as e:
                self.errors[f"list:{c.name}"] = f"{type(e).__name__}: {e}"
        return out

    def _resolve(self, address: str) -> tuple[Connector, str]:
        name, sep, dev = address.partition(":")
        if not sep:
            raise AudioError(f"bad device address {address!r}; want <connector>:<device>")
        c = self.connectors.get(name)
        if c is None:
            raise AudioError(f"unknown connector {name!r}; have {sorted(self.connectors)}")
        ok, why = c.available()
        if not ok:
            raise AudioError(f"connector {name!r} unavailable: {why}")
        return c, dev

    def open_source(self, address: str, spec: StreamSpec) -> Source:
        c, dev = self._resolve(address)
        return c.open_source(dev, spec)

    def open_sink(self, address: str, spec: StreamSpec) -> Sink:
        c, dev = self._resolve(address)
        return c.open_sink(dev, spec)
