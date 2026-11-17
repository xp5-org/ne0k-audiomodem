# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""The PLL bar says LOCKED only for something actually received from a peer.

2026-11-17 Mac, CSS 0.15 s, idle room: the lock bar was relit by "preamble sync score 0.50" (the decoder acquiring
on room noise), by tracker grid locks on our own transmissions' echoes, and by CW-ident guesses on noise bursts
(an "rx" with peer -1).  Each relight holds LOCKED 8 s, so the bar read LOCKED for minutes with nobody sending."""
import types

import pytest

from modemgui.engine.css_stream import CssStreamRx
from modemgui.engine.events import LinkEvent


def _fake_stream():
    got = []
    st = types.SimpleNamespace(quiet=False, log=lambda *a: None, fs=48000, counts={}, busy_changes=0,
                               emit_link=lambda kind, peer, t0, t1, d: got.append((kind, peer, d)))
    return st, got


def test_preamble_sync_and_grid_lock_are_sync_events_not_lock():
    st, got = _fake_stream()
    CssStreamRx._on_event(st, {"event": "SYNC_ACQUIRED", "x": 0.5, "a": 0, "b": 0, "sym": 0})
    CssStreamRx._on_trk_event(st, {"event": "LOCK", "a": 1, "b": 0, "x": -292.0, "y": 0.0})
    assert [k for k, *_ in got] == ["sync", "sync"]


def test_lockbar_ignores_sync_and_ident_guesses_but_takes_real_frames():
    pytest.importorskip("PySide6")
    from modemgui.ui.modem_feed import ModemFeed

    calls = []
    bar = types.SimpleNamespace(locked=lambda *a, **k: calls.append(k.get("why")), unlocked=lambda: None)
    feed = types.SimpleNamespace(lockbar=bar)
    for ev in (LinkEvent("sync", 2, -1, 0, 0, "preamble sync score 0.50"),
               LinkEvent("sync", 2, -1, 0, 0, "grid 1 locked -292 ppm"),
               LinkEvent("rx", 2, -1, 0, 1, "ident heard (cw)")):
        ModemFeed.on_link(feed, ev)
    assert calls == []
    ModemFeed.on_link(feed, LinkEvent("rx", 2, 1, 0, 1, "DATA seq=3 snr=20.0dB"))
    ModemFeed.on_link(feed, LinkEvent("lock", 2, 1, 0, 0, "4 frames"))
    assert len(calls) == 2
