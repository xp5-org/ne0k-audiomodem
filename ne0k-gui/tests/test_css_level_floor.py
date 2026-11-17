# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Level carrier sense follows the room floor.

2026-11-17 laptops, CSS 0.15 s, carrier_detect level: css.carrier_db -55 dBFS fixed against a room floor of
-57/-56 dBFS with -51 dBFS RMS.  Every 50 ms window read busy (INFO: in_burst=1, last_busy_age=0.0 s from power-on),
CSMA never saw the 2.5 s of idle DATA needs, and the first send waited 2-3.5 minutes for a quiet moment."""
import queue
import threading
import time
import types

import numpy as np
import pytest

from modemgui.engine.levelfloor import LevelFloor

FS = 48000
WIN = int(0.05 * FS)


def _room(rng, n_win, rms_db=-51.0):
    """Fluctuating room noise: 50 ms windows whose RMS wanders +-4 dB around ``rms_db``."""
    lv = 10 ** ((rms_db + rng.uniform(-4, 4, n_win)) / 20)
    return [rng.standard_normal(WIN).astype(np.float32) * l for l in lv]


def _rms(x):
    return float(np.sqrt(np.mean(x * x)))


def test_fixed_threshold_reads_this_room_as_busy_and_floor_does_not():
    rng = np.random.default_rng(1)
    wins = _room(rng, 200)
    fixed = LevelFloor(0.05, -55.0, 0.0)
    adapt = LevelFloor(0.05, -55.0, 10.0)
    busy_fixed = [_rms(w) > fixed.update(_rms(w), False) for w in wins]
    busy_adapt = [_rms(w) > adapt.update(_rms(w), False) for w in wins]
    assert np.mean(busy_fixed) > 0.95                      # the bug: always busy
    assert not any(busy_adapt[40:])                        # after 1 s warm-up + 1 s: the room is idle
    assert -58 < adapt.floor_db < -50 and adapt.threshold_db() > -50


def test_signal_20db_over_room_is_busy_and_a_long_burst_does_not_cut_itself_off():
    rng = np.random.default_rng(2)
    lf = LevelFloor(0.05, -55.0, 10.0)
    for w in _room(rng, 60):
        lf.update(_rms(w), False)
    floor0 = lf.floor_db
    sig = 10 ** (-31 / 20)                                 # ~20 dB over the room RMS (laptops measured 25-44)
    busy = []
    for i in range(int(450 / 0.05)):                       # 450 s: the longest 0.5 s-symbol frame seen
        x = sig * (1 + 0.05 * rng.standard_normal())
        busy.append(x > lf.update(x, i > 0))
    assert all(busy)
    assert lf.floor_db - floor0 < 10                       # rose by at most 0.02 dB/s while in the burst


def test_floor_falls_at_once_when_the_room_goes_quiet():
    lf = LevelFloor(0.05, -70.0, 10.0)
    for _ in range(100):
        lf.update(10 ** (-40 / 20), False)
    assert lf.floor_db == pytest.approx(-40, abs=0.1)
    for _ in range(100):
        lf.update(10 ** (-60 / 20), False)
    assert lf.floor_db == pytest.approx(-60, abs=0.1)
    assert lf.threshold_db() == pytest.approx(-50, abs=0.1)


def test_margin_zero_keeps_the_fixed_threshold():
    lf = LevelFloor(0.05, -55.0, 0.0)
    for _ in range(200):
        lf.update(10 ** (-30 / 20), False)
    assert lf.threshold_db() == -55.0


def test_css_link_rx_goes_idle_in_a_room_over_the_fixed_threshold():
    pytest.importorskip("scipy")
    from modemgui.audio import Registry
    from modemgui.engine import Session
    from modemgui.engine import css_link as cl
    from modemgui.engine.modemcfg import CssConfig

    s = Session(Registry(), FS, 1024)
    s.open("loop:floor_rx", "loop:floor_tx")
    try:
        lk = cl.CssLink(s, CssConfig(symbol_dur=0.15, amplitude=0.2, carrier_db=-55.0))
        lk._decode = lambda *a: None
        tap: queue.Queue = queue.Queue()
        rng = np.random.default_rng(3)
        for w in _room(rng, 160):                          # 8 s of room at -51 dBFS RMS
            tap.put(types.SimpleNamespace(samples=w, tx_active=False))
        th = threading.Thread(target=lk._rx, args=(types.SimpleNamespace(
            get=lambda timeout: tap.get(timeout=timeout) if not tap.empty() else (time.sleep(0.01) or None)),),
            daemon=True)
        th.start()
        t_end = time.monotonic() + 10
        while not tap.empty() and time.monotonic() < t_end:
            time.sleep(0.02)
        time.sleep(0.6)                                    # past channel_busy's 0.4 s settle
        assert not lk._in_burst
        assert not lk.channel_busy()
        assert "lvl_thr=" in lk.cs_state()
    finally:
        lk._stop.set()
        s.close()
