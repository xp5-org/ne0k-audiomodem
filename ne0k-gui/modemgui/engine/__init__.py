# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Headless engine.  MUST NOT import Qt (enforced by tests/test_layering.py).

The engine owns threads and publishes plain-data events (``events.py``) on a
queue; whatever sits on top -- the Qt UI, a CLI, a future websocket server --
drains that queue.
"""
from .events import Decoded, Event, Levels, LinkEvent, OfdmData, ProbeReport, Spectrum, Status
from .session import Session

__all__ = ["Decoded", "Event", "Levels", "LinkEvent", "OfdmData", "ProbeReport", "Session", "Spectrum", "Status"]
