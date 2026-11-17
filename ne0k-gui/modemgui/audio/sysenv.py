# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""The environment for host tools we run (pactl / parec / pacat / amixer).

A frozen (PyInstaller) release points LD_LIBRARY_PATH (DYLD_* on macOS) at its own ``_internal`` folder, which
holds a bundled ``libpulse.so.0``.  A system ``pactl`` started with that inherited path loads the bundled libpulse
and then fails to find the matching ``libpulsecommon-<ver>.so`` -- the Pulse connector and the host mixer both died
with "error while loading shared libraries" on the release built 2026-11-16.  The bootloader keeps the user's
original value in ``<VAR>_ORIG``; children get that back (or the variable removed when there was none).
"""
from __future__ import annotations

import os
import sys

_VARS = ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH")


def host_env(base: dict | None = None, frozen: bool | None = None) -> dict:
    env = dict(os.environ if base is None else base)
    if not (getattr(sys, "frozen", False) if frozen is None else frozen):
        return env
    for var in _VARS:
        orig = env.pop(var + "_ORIG", None)
        if orig is not None:
            env[var] = orig
        else:
            env.pop(var, None)
    return env
