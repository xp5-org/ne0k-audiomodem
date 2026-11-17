# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""CSS receive: bursts too short for any frame are not decoded.  CSS transmit: chirp edges are tapered.

2026-11-17 laptops, 0.5 s symbols, level sense: every room-noise blip (0.2-11 s) went through four matched-filter
decode passes on the css-rx thread -- 23 s here, ~40 s on m6600 -- with the channel counted busy, so the node's
ACK went out 40 s late and its DATA 170 s after the send, and the RX CPU meter cycled 99 % / 0 %.  The shortest
frame (a header-only ACK) is 43 s long at 0.5 s symbols; nothing under half of that can hold one.

The same day: speakers clicked at chirp ends -- each chirp stopped at an arbitrary phase of the top frequency."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from modemgui.audio import Registry
from modemgui.engine import Session
from modemgui.engine import css_link as cl
from modemgui.engine.modemcfg import CssConfig


@pytest.fixture
def link():
    reg = Registry()
    s = Session(reg, 48000, 1024)
    s.open("loop:gate_rx", "loop:gate_tx")
    lk = cl.CssLink(s, CssConfig(symbol_dur=0.05, amplitude=0.3))
    try:
        yield lk
    finally:
        s.close()


def test_burst_shorter_than_any_frame_is_not_decoded(link, monkeypatch):
    calls = []
    monkeypatch.setattr(cl, "decode_samples", lambda c, x: calls.append(len(x)) or [])
    ack_s = link.min_frame_s()
    assert ack_s > 1.0
    noise = (np.random.default_rng(0).standard_normal(int(0.3 * 48000)) * 0.01).astype(np.float32)
    link._decode(noise, 10.0, 10.3)                      # a 0.3 s blip
    assert calls == [] and link.short_bursts == 1
    long_s = ack_s * cl.MIN_BURST_FRAC + 0.5             # long enough to be a frame: decoded (4 shifts on a miss)
    link._decode(np.zeros(int(long_s * 48000), np.float32), 10.0, 10.0 + long_s)
    assert len(calls) == 4 and link.short_bursts == 1


def test_a_real_ack_still_decodes_through_the_gate(link):
    got = []
    link.session._emit = lambda ev: got.append(ev)
    ack = cl.encode_ack(link.cfg, 2, 5)
    burst = np.concatenate([np.zeros(4800, np.float32), ack, np.zeros(48000, np.float32)])
    link._decode(burst, 0.0, len(ack) / 48000 + 0.1)
    dec = [e for e in got if type(e).__name__ == "Decoded"]
    assert dec and dec[0].ok and dec[0].type_name == "ACK" and dec[0].seq == 5
    assert link.short_bursts == 0


def _edges(x, min_gap=20):
    """Samples right before and right after every symbol gap: a run of >= ``min_gap`` exact zeros (a sine crossing
    inside a chirp also quantises to a lone 0 sample, which is not a gap)."""
    z = np.concatenate([[False], x == 0, [False]]).astype(np.int8)
    starts = np.flatnonzero(np.diff(z) == 1)          # first zero of a run (index into x)
    ends = np.flatnonzero(np.diff(z) == -1)           # one past its last zero
    out = []
    for a, b in zip(starts, ends):
        if b - a >= min_gap:
            if a > 0:
                out.append(x[a - 1])
            if b < len(x):
                out.append(x[b])
    return np.abs(np.array(out))


def test_chirp_edges_are_tapered_and_frames_still_decode():
    hard = cl.encode_bytes(CssConfig(symbol_dur=0.05, amplitude=0.5, edge_taper_cycles=0), b"taper", 3, 2)
    soft = cl.encode_bytes(CssConfig(symbol_dur=0.05, amplitude=0.5, edge_taper_cycles=3), b"taper", 3, 2)
    assert len(hard) == len(soft)
    assert np.percentile(_edges(hard), 90) > 0.15          # hard edges: steps of a good part of full scale
    assert _edges(soft).max() < 0.02                        # tapered: every edge starts/ends near zero
    out = cl.decode_samples(CssConfig(symbol_dur=0.05, amplitude=0.5, edge_taper_cycles=3), soft)
    assert out and out[0].ok and out[0].payload.rstrip(b"\x00") == b"taper"
