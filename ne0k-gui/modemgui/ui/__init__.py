# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""PySide6 front end.  Talks to the engine only through ``engine.Session`` and its
event queue (see bridge.py); never reaches into ``audio`` internals."""
