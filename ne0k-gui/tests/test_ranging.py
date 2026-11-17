# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Two-way CSS ranging (engine/ranging.py, from audiotestserver tests/CSS/css_ranging_*)."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from modemgui.engine import ranging
from modemgui.engine.ranging import BurstParams, DETECT_DB, SPEED_OF_SOUND_M_S as C

FS = 48000


def test_distance_math():
    """css_ranging_selftest C: a known (rtt, turnaround) gives the one-way metres; a bad timing goes negative."""
    turn = 0.5
    rtt = turn + 2 * 5.0 / C
    assert ranging.distance_m(rtt, turn) == pytest.approx(5.0, abs=1e-9)
    assert ranging.distance_m(turn - 0.001, turn) < 0
    # the responder's clock runs 100 ppm fast: its turnaround reads long, ppm takes it back out
    assert ranging.distance_m(rtt, turn * (1 + 100e-6), ppm=100) == pytest.approx(5.0, abs=1e-6)
    assert ranging.distance_m(rtt, turn, bias_m=0.2) == pytest.approx(5.2)


def test_burst_matches_the_reference_script():
    """A GUI node and the audiotestserver script send the same ranging waveform, so either can be either side."""
    tests = Path(__file__).resolve().parents[2] / "audiotestserver" / "tests"
    if not (tests / "CSS" / "css_ranging_analyze.py").exists():
        pytest.skip("audiotestserver not checked out beside ne0k-gui")
    for p in (tests, tests / "CSS"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    ref = pytest.importorskip("css_ranging_analyze")
    ep = dict(f_min=800.0, f_max=4000.0, symbol_dur=0.3, amplitude=0.3)
    mine = ranging.burst(FS, BurstParams(**ep))
    theirs = ref.ranging_burst(FS, ep)
    assert len(mine) == len(theirs)
    assert np.max(np.abs(mine - theirs)) < 1e-5


def _place(cap, x, at: float):
    """Add x into cap starting at fractional sample ``at`` (FFT delay, so sub-sample positions are exact)."""
    k = int(at)
    n = 1 << int(np.ceil(np.log2(len(x) + 64)))
    f = np.fft.rfftfreq(n)
    y = np.fft.irfft(np.fft.rfft(x, n) * np.exp(-2j * np.pi * f * (at - k)), n)[:len(x) + 32]
    cap[k:k + len(y)] += y[:max(0, min(len(y), len(cap) - k))]


def test_detection_needs_a_whole_burst_and_ignores_noise():
    p = BurstParams(symbol_dur=0.1)
    tmpl = ranging.preamble(FS, p)
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, 4 * FS)
    assert all(f.score_db < DETECT_DB for f in ranging.fiducials(noise, tmpl, FS, 1))
    cap = noise.copy()
    _place(cap, ranging.burst(FS, p), 1000.3)
    got = ranging.fiducials(cap, tmpl, FS, 1)
    assert got[0].score_db > DETECT_DB + 10
    assert got[0].pos == pytest.approx(1000.3 + ranging.LEAD_S * FS, abs=0.5)
    cut = cap[:int(1000 + ranging.LEAD_S * FS + 0.7 * len(tmpl))]          # 70 % of the burst has arrived
    assert all(f.score_db < DETECT_DB for f in ranging.fiducials(cut, tmpl, FS, 1))


def _room(src, delays_m, gains, fs=FS, clock_ppm=0.0):
    """What one mic records of one burst: a sum of (path metres, gain) arrivals, sampled by a clock ``clock_ppm``
    fast.  Returns (signal, extra samples before the first arrival)."""
    out = np.zeros(len(src) + int(0.2 * fs))
    for d, g in zip(delays_m, gains):
        _place(out, src * g, d / C * fs)
    if clock_ppm:
        t = np.arange(int(len(out) * (1 + clock_ppm * 1e-6))) / (1 + clock_ppm * 1e-6)
        out = np.interp(t, np.arange(len(out)), out)
    return out


@pytest.mark.parametrize("dist,ppm", [(2.0, 0.0), (6.5, 0.0), (3.0, 120.0)])
def test_offline_two_captures_with_reflections_and_clock_skew(dist, ppm):
    """css_ranging_analyze over two synthetic captures: playback latencies differ, a wall reflection 4 ms behind
    the direct path is 1.6x LOUDER than it (the leading edge must still take the direct path), and the
    responder's sound card runs ``ppm`` fast."""
    p = BurstParams(symbol_dur=0.12)
    sig = ranging.burst(FS, p).astype(np.float64)
    spk_mic = (0.12, 0.18)                                                # each laptop's own speaker-to-mic metres
    rng = np.random.default_rng(int(dist * 10))
    n = 9 * FS
    ini, res = rng.normal(0, 0.003, n), rng.normal(0, 0.003, int(n * (1 + ppm * 1e-6)))
    t_req, turn_true = 0.6 * FS + 1777.25, 3.2                           # A plays at t_req (initiator samples)
    k = 1 + ppm * 1e-6                                                     # responder samples per initiator sample
    near = lambda s: ([s], [1.0])                                          # self-echo: one strong short path
    far = ([dist, dist + 4e-3 * C], [0.25, 0.4])                         # direct + a louder reflection
    # initiator capture: own request, then the reply
    a = _room(sig, *near(spk_mic[0])); _place(ini, a, t_req)
    t_reply = t_req + (dist / C) * FS + turn_true * FS                    # B plays turn_true after hearing A
    b = _room(sig, *far, clock_ppm=-ppm); _place(ini, b, t_reply)         # (_room adds each path's flight)
    # responder capture, on its own clock
    _place(res, _room(sig, *far, clock_ppm=ppm), t_req * k)
    _place(res, _room(sig, *near(spk_mic[1])), t_reply * k)
    r = ranging.analyse(ini, res, FS, p, ppm=ppm)
    bias = sum(spk_mic) / 2
    d = r.distance_m(bias_m=bias, ppm=ppm)
    assert d == pytest.approx(dist, abs=0.01), (d, r)
    if ppm:                                                                 # the skew, not corrected: > 4 cm
        assert abs(ranging.analyse(ini, res, FS, p).distance_m(bias_m=bias) - dist) > 0.04
    assert min(r.scores_db) > DETECT_DB


# -- live: two nodes over a simulated room, in real time ---------------------------------------------------------

class _Spec:
    rate = FS
    iq = False


class _Air:
    """Real-time medium: every block a node plays reaches every listener (itself included) along that pair's
    (delay, gain) taps after the node's playback latency; listeners also hear noise."""

    BLOCK = 480

    def __init__(self, taps: dict, latency: dict, noise: float = 0.002):
        self.taps, self.latency, self.noise = taps, latency, noise
        self.pipes = {n: _Pipe(self, n) for n in latency}
        self.t = 0
        self.fut = {n: np.zeros(4 * FS) for n in latency}               # listener -> samples from self.t on
        self.rng = np.random.default_rng(3)
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        t0 = time.monotonic()
        while not self.stop.is_set():
            B = self.BLOCK
            for tx, pipe in self.pipes.items():
                x = pipe._take(B)
                if not x.any():
                    continue
                for ln in self.pipes:
                    for d, g in self.taps[(tx, ln)]:
                        _place(self.fut[ln], x * g, self.latency[tx] + d / C * FS)
            for ln, pipe in self.pipes.items():
                f = self.fut[ln]
                pipe._hear(f[:B] + self.rng.normal(0, self.noise, B))
                self.fut[ln] = np.concatenate([f[B:], np.zeros(B)])
            self.t += B
            wait = t0 + self.t / FS - time.monotonic()
            if wait > 0:
                time.sleep(wait)


class _Pipe:
    def __init__(self, air, name):
        self.spec, self.name = _Spec(), name
        self.lock = threading.Lock()
        self.hist, self.base = np.zeros(0), 0
        self.txq = np.zeros(0)
        self.idle = threading.Event(); self.idle.set()

    def _take(self, n):
        with self.lock:
            x, self.txq = self.txq[:n], self.txq[n:]
            if not len(self.txq):
                self.idle.set()
        return np.concatenate([x, np.zeros(n - len(x))])

    def _hear(self, x):
        with self.lock:
            self.hist = np.concatenate([self.hist, x])
            if len(self.hist) > 30 * FS:
                cut = len(self.hist) - 30 * FS
                self.hist, self.base = self.hist[cut:], self.base + cut

    def history(self, n=None):
        with self.lock:
            h = self.hist if n is None else self.hist[-n:]
            return h.astype(np.float32), self.base + len(self.hist) - len(h)

    def play(self, x):
        with self.lock:
            self.txq = np.concatenate([self.txq, np.asarray(x, np.float64)])
            self.idle.clear()

    def wait_tx(self, timeout=None):
        return self.idle.wait(timeout)


def test_live_respond_and_initiate_over_a_room():
    """The control port's RANGE RESPOND / RANGE INIT over two simulated laptops 3 m apart: different playback
    latencies (37 / 81 ms, which must cancel), a reflection louder than the direct path, no coordination server."""
    dist, spk_mic = 3.0, {"A": 0.12, "B": 0.18}
    taps = {("A", "A"): [(spk_mic["A"], 0.8)], ("B", "B"): [(spk_mic["B"], 0.8)],
            ("A", "B"): [(dist, 0.2), (dist + 1.5, 0.35)], ("B", "A"): [(dist, 0.2), (dist + 1.5, 0.35)]}
    air = _Air(taps, {"A": 0.037 * FS, "B": 0.081 * FS})
    p = BurstParams(symbol_dur=0.08, amplitude=0.3)
    out = {}

    def resp():
        try:
            out["B"] = ranging.respond(air.pipes["B"], p, timeout=10, margin=0.4, tail=0.3)
        except Exception as e:                                             # surfaced by the assert below
            out["B"] = e

    try:
        time.sleep(0.3)
        th = threading.Thread(target=resp); th.start()
        time.sleep(0.5)
        a = ranging.initiate(air.pipes["A"], p, wait=10, tail=0.3)
        th.join(15)
    finally:
        air.stop.set(); air.th.join(2)
    b = out.get("B")
    assert isinstance(b, ranging.LegResult), b
    d = ranging.distance_m(a.interval_s, b.interval_s, bias_m=(spk_mic["A"] + spk_mic["B"]) / 2)
    assert d == pytest.approx(dist, abs=0.03), (d, a.rows(), b.rows())
    rows = dict(a.rows())
    assert rows["range.role"] == "initiator" and "range.rtt_s" in rows and float(rows["range.peer_db"]) > DETECT_DB
    assert "range.turnaround_s" in dict(b.rows())


def test_initiate_without_a_responder_says_so():
    air = _Air({("A", "A"): [(0.1, 0.8)]}, {"A": 0.02 * FS})
    try:
        with pytest.raises(ranging.RangingError, match="no reply within"):
            ranging.initiate(air.pipes["A"], BurstParams(symbol_dur=0.05), wait=1.5)
    finally:
        air.stop.set(); air.th.join(2)
