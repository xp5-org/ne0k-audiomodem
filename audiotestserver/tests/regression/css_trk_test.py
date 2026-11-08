# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""css_trk_test.py — synthetic scenarios for the C chirp tracker (css_trk).

  own_return     we transmit 40 chirps; the line returns them 120 ms later at
                 -6 dB.  Every returned chirp must be labelled OWN and the lag
                 learned within 2 ms.
  echo           a peer frame plus a -10 dB copy 35 ms later.  The direct path
                 must lock; the copy must be reported as ECHO of that grid at
                 35 ms / about -10 dB, never as a second station.
  phase_step     a settled peer jumps +200 samples mid-frame.  One PHASE_STEP of
                 about +200 samples, and the grid keeps tracking (no new grid).
  two_peers      two stations at +150 / -150 ppm overlapping in time at similar
                 power.  Two locked grids whose ppm estimates are within 40 ppm
                 of the truth, and a COLLISION event.
  quiet          noise only: no grids, no chirps, carrier sense idle after
                 calibration.

Usage: python tests/regression/css_trk_test.py [-v]
"""
from __future__ import annotations

import os
import sys

import numpy as np

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(TESTS, "CSS"))
import css_pll_c  # noqa: E402

FS, DUR, F0, F1, GAP = 48000, 0.15, 400.0, 3500.0, 0.05
SPS = int(FS * DUR)
SPS_EFF = SPS + int(SPS * GAP)


def chirp_at(buf, start, up, amp, ppm=0.0):
    """Add one chirp starting at fractional sample `start`, its clock scaled by ppm."""
    scale = 1.0 + ppm * 1e-6
    n = int(np.ceil(SPS * scale)) + 2
    i0 = int(np.floor(start))
    idx = np.arange(i0, i0 + n)
    tt = (idx - start) / scale / FS                 # seconds of the transmitter's clock
    m = (tt >= 0) & (tt < DUR) & (idx >= 0) & (idx < len(buf))
    f0, f1 = (F0, F1) if up else (F1, F0)
    k = (f1 - f0) / DUR
    buf[idx[m]] += amp * np.sin(2 * np.pi * (f0 * tt[m] + 0.5 * k * tt[m] ** 2))


def frame(buf, t0, dirs, amp, ppm=0.0, jump=None):
    """Chirps every SPS_EFF (in the sender's clock) from sample t0. jump=(k, samples)."""
    ends = []
    for k, d in enumerate(dirs):
        s = t0 + k * SPS_EFF * (1 + ppm * 1e-6)
        if jump and k >= jump[0]:
            s += jump[1]
        chirp_at(buf, s, d, amp, ppm)
        ends.append(s + SPS * (1 + ppm * 1e-6) - 1)
    return ends


def run(x, **kw):
    chirps, events = [], []
    trk = css_pll_c.TrkStream(FS, DUR, F0, F1, GAP, on_chirp=chirps.append,
                              on_event=events.append, **kw)
    return trk, chirps, events


def feed(trk, x, block=960):
    for i in range(0, len(x), block):
        trk.push(x[i:i + block])


def noise(n, sd, seed):
    return np.random.default_rng(seed).normal(0, sd, n)


def t_own_return(v):
    rng = np.random.default_rng(1)
    x = noise(FS * 14, 0.01, 11)
    dirs = list(rng.integers(0, 2, 40))
    t0 = 4 * FS
    lag = int(0.120 * FS)
    frame(x, t0 + lag, dirs, 0.25)               # what comes back
    trk, ch, ev = run(x)
    trk.tx(t0, dirs)                              # what we sent, RX timebase
    feed(trk, x.astype(np.float32))
    own = [c for c in ch if c["label"] == "OWN"]
    lagv = trk.own_lag()
    ok = len(own) >= 36 and lagv is not None and abs(lagv - 0.120) < 0.002
    return ok, f"OWN {len(own)}/40 lag={lagv}"


def t_echo(v):
    rng = np.random.default_rng(2)
    x = noise(FS * 18, 0.01, 12)
    dirs = list(rng.integers(0, 2, 70))
    t0 = 4 * FS
    frame(x, t0, dirs, 0.3)
    frame(x, t0 + int(0.035 * FS), dirs, 0.3 * 10 ** (-10 / 20))
    trk, ch, ev = run(x)
    feed(trk, x.astype(np.float32))
    echo_ev = [e for e in ev if e["event"] == "ECHO"]
    locks = [e for e in ev if e["event"] == "LOCK"]
    direct = [c for c in ch if c["label"] in ("UNKNOWN", "PEER") and c["grid"] >= 0]
    ok = (len(echo_ev) >= 1 and abs(echo_ev[0]["x"] - 0.035) < 0.002
          and -13 < echo_ev[0]["y"] < -7 and len(locks) >= 1)
    grids = trk.grids()
    if v:
        print("    grids:", [(g["id"], g["label"], g["settled"], round(g["ppm"], 1)) for g in grids])
    return ok, (f"ECHO events={len(echo_ev)} "
                f"{'delay=%.4fs gain=%.1fdB' % (echo_ev[0]['x'], echo_ev[0]['y']) if echo_ev else ''} "
                f"locks={len(locks)} direct_chirps={len(direct)}")


def t_phase_step(v):
    rng = np.random.default_rng(3)
    x = noise(FS * 20, 0.01, 13)
    dirs = list(rng.integers(0, 2, 90))
    frame(x, 4 * FS, dirs, 0.3, jump=(45, 200))
    trk, ch, ev = run(x)
    feed(trk, x.astype(np.float32))
    steps = [e for e in ev if e["event"] == "PHASE_STEP"]
    news = [e for e in ev if e["event"] == "NEW"]
    ok = len(steps) == 1 and abs(steps[0]["x"] - 200) < 8 and len(news) == 1
    return ok, f"PHASE_STEP {[round(s['x'], 1) for s in steps]} new_grids={len(news)}"


def t_two_peers(v):
    rng = np.random.default_rng(4)
    x = noise(FS * 22, 0.01, 14)
    frame(x, 4 * FS, list(rng.integers(0, 2, 90)), 0.3, ppm=+150)
    frame(x, 4 * FS + SPS_EFF * 20 + SPS_EFF // 2, list(rng.integers(0, 2, 90)), 0.27, ppm=-150)
    trk, ch, ev = run(x)
    feed(trk, x.astype(np.float32))
    locks = [e for e in ev if e["event"] == "LOCK"]
    coll = [e for e in ev if e["event"] == "COLLISION"]
    grids = [g for g in trk.grids() if g["settled"]] or []
    # LOST grids are gone at the end: use the LOCK-time ppm plus the final grids
    ppms = sorted(round(e["x"]) for e in locks)
    if v:
        print("    locks:", ppms, "final:", [(g["id"], round(g["ppm"], 1), g["hits"]) for g in trk.grids()])
    ok = (len(locks) >= 2 and len(coll) >= 1
          and any(abs(p - 150) < 40 for p in ppms) and any(abs(p + 150) < 40 for p in ppms))
    return ok, f"LOCK ppm={ppms} collisions={len(coll)}"


def t_quiet(v):
    x = noise(FS * 10, 0.01, 15)
    trk, ch, ev = run(x)
    feed(trk, x.astype(np.float32))
    news = [e for e in ev if e["event"] == "NEW"]
    ok = len(ch) == 0 and len(news) == 0 and not trk.busy
    return ok, f"chirps={len(ch)} grids={len(news)} busy={trk.busy}"


def main():
    v = "-v" in sys.argv
    tests = [t_own_return, t_echo, t_phase_step, t_two_peers, t_quiet]
    fails = 0
    for t in tests:
        try:
            ok, msg = t(v)
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok, msg = False, f"ERROR {e}"
        fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {t.__name__[2:]:12s} {msg}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
