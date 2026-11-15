# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""engine/slotimpair: rule parsing, deterministic picking, the filter's depth / SNR, class gating and arming."""
import types

import numpy as np
import pytest

from modemgui.engine import slotimpair as si

BAT = [2, 4, 6, 6, 0, 4, 2, 6, 0, 1]
HZ = [100.0 * i for i in range(10)]
PIL = [50.0, 850.0, 1650.0]


def test_parse_rules_and_defaults():
    r = si.parse("hot:4:block:data, pilot:all:fuzz@3:any")
    assert [str(x) for x in r] == ["hot:4:block@60:data", "pilot:all:fuzz@3:any"]
    assert si.parse("none") == [] and si.parse("") == []
    for bad in ("hot:4:block", "warm:4:block:data", "hot:0:block:data", "hot:4:zap:data", "hot:4:block:both"):
        with pytest.raises(ValueError):
            si.parse(bad)


@pytest.mark.parametrize("role,want", [
    ("hot", [200.0, 300.0, 700.0]),          # the 6-bit carriers, lowest index first on ties
    ("cold", [0.0, 600.0, 900.0]),           # 1 and 2 bits, never a 0-bit one
    ("run", [100.0, 200.0, 300.0]),          # the adjacent window with the most bits
    ("spread", [100.0, 500.0, 700.0]),
    ("pilot", [50.0, 850.0, 1650.0]),
    ("idle", [400.0, 800.0]),                # only two 0-bit carriers exist: fewer than asked, never others
])
def test_pick_is_by_role_and_deterministic(role, want):
    assert si.pick(role, 3, BAT, HZ, PIL) == want == si.pick(role, 3, BAT, HZ, PIL)


def test_pick_all_and_no_idle_carrier():
    assert len(si.pick("hot", None, BAT, HZ, PIL)) == 8
    assert si.pick("idle", None, [2] * 10, HZ, PIL) == []


def _tone(fs, f, n):
    return np.exp(2j * np.pi * f * np.arange(n) / fs)


def test_block_removes_only_the_slot_complex():
    fs, n = 200000, 100000
    x = (_tone(fs, 7812.5, n) + _tone(fs, -15625.0, n)).astype(np.complex64)
    y = si.apply(x, fs, [7812.5], 781.25, "block", 60, np.random.default_rng(0))
    p = lambda s, f: abs(np.vdot(_tone(fs, f, n), s) / n) ** 2
    assert y.dtype == x.dtype and len(y) == n
    assert p(y, -15625.0) > 0.99 and p(y, 7812.5) < 1e-5


def test_fuzz_sets_the_snr_inside_the_slot_and_stays_there():
    fs, n = 48000, 96000
    x = np.random.default_rng(1).standard_normal(n).astype(np.float32)
    y = si.apply(x, fs, [1000.0], 46.875, "fuzz", 6, np.random.default_rng(2))
    f = np.fft.rfftfreq(n, 1 / fs)
    X, D = np.fft.rfft(x), np.fft.rfft(y - x)
    m = np.abs(f - 1000.0) <= 20
    snr = 10 * np.log10(np.sum(np.abs(X[m]) ** 2) / np.sum(np.abs(D[m]) ** 2))
    assert abs(snr - 6) < 1.5
    slot = np.abs(f - 1000.0) <= 46.875 / 2                 # the whole slot: nothing may land outside it
    assert np.sum(np.abs(D[~slot]) ** 2) / np.sum(np.abs(D) ** 2) < 1e-6


def test_gate_filters_only_its_class_and_only_once_armed():
    up = {"v": False}
    g = si.Gate("hot:1:block:ctrl", lambda: (BAT, HZ, PIL, 4000.0, 100.0), armed=lambda: up["v"])
    x = np.cos(2 * np.pi * 200 * np.arange(4000) / 4000).astype(np.float32)
    g.cls.v = "ctrl"
    assert g.filter(x) is x                              # not armed yet: before the table is agreed
    up["v"] = True
    assert np.abs(g.filter(x)).max() < 0.01              # ctrl burst: the 200 Hz (hot) carrier is gone
    g.cls.v = "data"
    assert g.filter(x) is x                              # data burst: not this rule's class
    rep = g.report()
    assert rep["bursts"]["ctrl"] == {"filtered": 1, "passed": 1} and rep["bursts"]["data"]["filtered"] == 0
    assert g.hits("ctrl") == 1 and g.hits("data") == 0
    assert rep["picked"][0]["rules"][0]["slots_hz"] == [200.0] and rep["picked"][0]["rules"][0]["bits_hit"] == 6


def test_attach_ofdm_classes_from_what_play_says():
    seen = []
    sess = types.SimpleNamespace(play_raw=lambda s: seen.append(s))
    link = types.SimpleNamespace(up=True, _tx_bat=np.array(BAT), session=sess, cfg=None)
    link._play = lambda audio, what, ident=False: sess.play_raw(audio)
    g = si.Gate("hot:1:block:data", lambda: (BAT, HZ, PIL, 4000.0, 100.0), armed=lambda: True)
    # wire the same way attach_ofdm does, minus the handshake grid (needs the OFDM config)
    play = link._play

    def _play(audio, what, ident=False):
        w = what.split(" ", 1)[0]
        g.cls.v = "data" if w in ("DATA", "STREAM") else "ctrl" if w in ("ACK", "GO") else "other"
        try:
            return play(audio, what, ident=ident)
        finally:
            g.cls.v = "data"
    link._play = _play
    si._shadow_play_raw(sess, g)
    x = np.ones(400, np.float32)
    link._play(x, "ACK ack=1"); link._play(x, "probe"); link._play(x, "DATA seq=0 ack=0 4B")
    sess.play_raw(x)                                     # played outside _play (EC chunk): data
    assert g.counts["ctrl"] == [0, 1] and g.counts["other"] == [0, 1] and g.counts["data"] == [2, 0]


def test_pin_keeps_the_first_slots_when_the_table_changes():
    from modemgui.engine.slotimpair import Gate, parse
    r = parse("hot:2:block:data:pin,hot:2:block:data")
    assert [x.pin for x in r] == [True, False] and str(r[0]).endswith(":pin")
    tab = {"bat": [2, 6, 6, 2, 4]}
    hz = [100.0, 200.0, 300.0, 400.0, 500.0]
    g = Gate("hot:2:block:data:pin,hot:2:block:data", lambda: (tab["bat"], hz, [], 8000.0, 50.0))
    g.armed = True
    x = np.ones(800)
    g.filter(x)
    tab["bat"] = [2, 0, 0, 2, 4]                # the receiver dropped the dead carriers
    g.filter(x)
    first, last = g.picked[0]["rules"], g.picked[-1]["rules"]
    assert first[0]["slots_hz"] == last[0]["slots_hz"] == [200.0, 300.0]   # pinned: same place
    assert last[0]["bits_hit"] == 0
    assert last[1]["slots_hz"] == [400.0, 500.0] or last[1]["slots_hz"] == [100.0, 500.0]  # tracking: moved
