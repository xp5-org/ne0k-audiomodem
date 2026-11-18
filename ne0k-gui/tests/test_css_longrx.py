# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Live long-symbol CSS receive (engine/css_longrx.py + css_decoder mf hooks): the clock is tracked while the
burst arrives and keeps its focus through the decode tiers (1-2 live grid, 3 chunk re-scan, 4 burst re-scan,
5 loss).  The channel is the BT/FM link as measured: 44.1 kHz TX resampled to 48 kHz with a -529.6 ppm clock."""
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

from modemgui.engine import css_link as cl
from modemgui.engine.css_longrx import CssLongRx
from modemgui.engine.css_stream import chirp_dirs
from modemgui.engine.events import Decoded
from modemgui.engine.modemcfg import CssConfig

PPM = -529.6
FS = 48000
MSG = b"NE0K-05\n"
TX = CssConfig(symbol_dur=0.5, chunk_size=1, fs=44100, node_id=2, peer_id=1)
RX = CssConfig(symbol_dur=0.5, chunk_size=1, fs=FS, node_id=1, peer_id=2)
BLOCK = 4800


def _geom():
    info = json.loads(Path(cl._info(RX)).read_text())
    return info, int(info.get("training_symbols", 0)) + len(info["preamble"]) + int(info["frame_symbols"])


def _tx_pcm(seq=1):
    return cl.encode_bytes(TX, MSG, seq, 1).astype(np.float64)


def _chirp_grid(pcm):
    sps = int(TX.fs * TX.symbol_dur)
    eff = sps + int(round(sps * TX.symbol_gap_frac))
    first, dirs = chirp_dirs(pcm, TX.fs, TX.f_min, TX.f_max, TX.symbol_dur, sps, eff)
    return first, sps, eff, dirs                  # first chirp's START sample


def _channel(pcm, snr_db=25.0, lead=3.0, tail=3.0, seed=0):
    y = resample_poly(pcm, 160, 147)
    n = int(round(len(y) * (1 + PPM * 1e-6)))
    y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
    s = np.sqrt(np.mean(y ** 2))
    x = np.concatenate([np.zeros(int(lead * FS)), y, np.zeros(int(tail * FS))])
    x += np.random.default_rng(seed).standard_normal(len(x)) * s * 10 ** (-snr_db / 20)
    return x.astype(np.float32), int(lead * FS), int(lead * FS) + len(y)


class _Harness:
    def __init__(self):
        self.decoded, self.links, self.lines = [], [], []
        _, cm, _ = cl._imports()
        import css_decoder
        self.stop = threading.Event()
        self.chunk_fed = []                      # samples fed when each live chunk came out
        self.rx = CssLongRx(RX, cl._info(RX), None, node_id=1, hear_own=False, log=self._log,
                            emit=self.decoded.append, emit_link=lambda *a: self.links.append(a),
                            decoded_cls=Decoded, unpack=cm.unpack_packet, decoder_mod=lambda: css_decoder,
                            work=cl._work, stop_event=self.stop)

    def _log(self, m):
        self.lines.append(m)
        if m.startswith("css rx chunk"):
            self.chunk_fed.append(self.rx.fed)

    def run(self, x, bursts, pace=True):
        """Feed x in 100 ms blocks; open / close the burst gate at the given (start, end) sample spans
        (the gate opens a block late and closes 1 s after the signal, like CssLink._rx).  pace: wait for
        the decoder to drain each block, i.e. real time."""
        opened = closed = 0
        sessions = []
        for i in range(0, len(x), BLOCK):
            self.rx.feed(x[i:i + BLOCK], False)
            if opened < len(bursts) and i >= bursts[opened][0] + BLOCK:
                self.rx.burst_start(); sessions.append(self.rx.session); opened += 1
            if closed < opened and i >= bursts[closed][1] + FS:
                self.rx.burst_end(); closed += 1
            s = self.rx.session
            if pace and s is not None:
                t = time.monotonic()
                time.sleep(0.001)
                while not s.q.waiting and s.is_alive() and time.monotonic() - t < 30:
                    time.sleep(0.001)
        self.rx.burst_end()
        for s in sessions:
            s.join(600)
        return sessions


def _ok(h, n=1):
    good = [d for d in h.decoded if d.ok and d.payload.rstrip(b"\x00") == MSG]
    assert len(good) >= n, ([(d.ok, d.payload, d.frames, d.frames_errored) for d in h.decoded], h.lines[-15:])
    return good


def test_clean_link_decodes_live_and_reports_the_clock():
    x, a, b = _channel(_tx_pcm())
    h = _Harness()
    h.run(x, [(a, b)])
    d = _ok(h)[0]
    # the true symbol period at the receiver: the 44.1 kHz encoder rounds its gap to whole samples
    true_sps = _chirp_grid(_tx_pcm())[2] * FS / TX.fs * (1 + PPM * 1e-6)
    assert abs(d.grid_sps / true_sps - 1) * 1e6 < 15, (d.grid_sps, true_sps)
    assert h.chunk_fed and h.chunk_fed[0] < b - 5 * FS, "first chunk must come out while the burst is on air"
    assert abs(h.rx.clock["grid_sps"] / true_sps - 1) * 1e6 < 15
    assert not any("tier4" in l or "tier5" in l for l in h.lines), h.lines


def test_weak_flipped_symbol_is_rescued_by_the_chunk_rescan():
    pcm = _tx_pcm()
    info, period = _geom()
    s0, sps, eff, dirs = _chirp_grid(pcm)
    k = 4 * period + int(info.get("training_symbols", 0)) + len(info["preamble"]) + 5
    i = s0 + k * eff
    seg = pcm[i:i + sps].copy()
    pcm[i:i + sps] = 0.42 * seg - 0.58 * seg[::-1]        # mostly the other direction, low confidence
    x, a, b = _channel(pcm)
    h = _Harness()
    h.run(x, [(a, b)])
    _ok(h)
    assert any("MF_RESCAN" in l and "flip" in l for l in h.lines), [l for l in h.lines if "RESCAN" in l or "ERROR" in l]


@pytest.mark.parametrize("where", ["payload", "training"])
def test_sample_slip_relocks_and_later_chunks_decode(where):
    """0.3 symbol of samples lost on the RX path (a BT stall): far outside ±track_search."""
    pcm = _tx_pcm()
    info, period = _geom()
    s0, sps, eff, dirs = _chirp_grid(pcm)
    tr = int(info.get("training_symbols", 0))
    k = 3 * period + (tr + len(info["preamble"]) + 6 if where == "payload" else max(1, tr // 2))
    x, a, b = _channel(pcm)
    cut = a + int(round((s0 + k * eff + sps // 3) * 160 / 147))
    x = np.concatenate([x[:cut], x[cut + int(0.3 * FS * 0.5):]])
    b -= int(0.3 * FS * 0.5)
    h = _Harness()
    h.run(x, [(a, b)])
    assert any("MF_RELOCK" in l and "lost" in l for l in h.lines) or any(
        "MF_RESCAN" in l and "refit" in l for l in h.lines), h.lines[-20:]
    chunks = [l for l in h.lines if l.startswith("css rx chunk ")]
    got = {int(l.split()[3]) for l in chunks}
    assert {4, 5, 6, 7, 8, 9} <= got, chunks
    if where == "training":
        _ok(h)


def test_two_bursts_both_decode_and_the_second_is_seeded():
    p1, p2 = _tx_pcm(1), _tx_pcm(2)
    x1, a1, b1 = _channel(p1, tail=4.0, seed=1)
    x2, a2, b2 = _channel(p2, lead=0.37, tail=3.0, seed=2)           # a new transmission at a new phase
    x = np.concatenate([x1, x2])
    h = _Harness()
    h.run(x, [(a1, b1), (len(x1) + a2, len(x1) + b2)])
    good = _ok(h, 2)
    assert {d.seq for d in good} == {1, 2}
    assert h.rx.sessions == 2


def test_noise_only_burst_delivers_nothing():
    x = (0.02 * np.random.default_rng(5).standard_normal(20 * FS)).astype(np.float32)
    h = _Harness()
    h.run(x, [(2 * FS, 15 * FS)], pace=False)
    assert not h.decoded


def test_unrecoverable_chunk_goes_through_tier4_to_tier5_and_the_clock_holds():
    """An 8-symbol dead fade inside chunk 6's payload: no chase or refit can rebuild it (4 dead symbols the
    chase does recover: the zeroed decisions are coin flips and the least-confident ones get flipped).  Tier 4 re-scans the
    burst on the session thread, tier 5 declares the loss, and chunks 7-9 still decode live on the grid."""
    pcm = _tx_pcm()
    info, period = _geom()
    s0, sps, eff, dirs = _chirp_grid(pcm)
    k = 6 * period + int(info.get("training_symbols", 0)) + len(info["preamble"]) + 8
    pcm[s0 + k * eff:s0 + (k + 8) * eff] = 0.0
    x, a, b = _channel(pcm)
    h = _Harness()
    h.run(x, [(a, b)])
    got = {int(l.split()[3]) for l in h.lines if l.startswith("css rx chunk ")}
    assert {7, 8, 9} <= got and 6 not in got, got
    assert any("tier4 burst re-scan" in l for l in h.lines), h.lines[-10:]
    assert any("tier5 packet lost: chunks [6]" in l for l in h.lines), h.lines[-10:]
    assert not [d for d in h.decoded if d.ok]


@pytest.fixture
def node_geometry(monkeypatch):
    """The release bundles ship without audiotestserver/shared.py, so their encoder falls back to the legacy
    16-symbol training run-in (61-symbol frames at 0.5 s instead of 51).  Over the air (2026-11-18) the
    decoder then took 11/13 Barker matches inside that run-in as preambles (payload read 24 symbols early)
    and, after a lattice reset, inside a payload (13 late).  Reproduce that geometry here."""
    _, _, _ = cl._imports()
    import css_encoder
    monkeypatch.setattr(css_encoder, "_default_training_symbols", None)
    yield


def test_long_runin_false_preamble_is_rescued_and_the_lattice_follows(node_geometry):
    info, period = _geom()
    assert int(info["training_symbols"]) == 16 and period == 61
    x, a, b = _channel(_tx_pcm())
    h = _Harness()
    h.run(x, [(a, b)])
    _ok(h)
    chunks = [l for l in h.lines if l.startswith("css rx chunk ")]
    assert len({int(l.split()[3]) for l in chunks}) == 10, chunks
    assert not any("tier4" in l or "tier5" in l for l in h.lines), [l for l in h.lines if "MF_" in l][:20]
