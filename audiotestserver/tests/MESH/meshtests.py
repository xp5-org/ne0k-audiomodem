# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
meshtests.py — the 15-node mesh scenario testlist.

Each scenario builds a MacSim (Tier 1), optionally drives some traffic on a
schedule, runs to completion, and checks a pass criterion.  The same scenario
configs are the ones a DSP-in-loop run (Tier 2, meshsim_dsp.py) will replay for
real-audio truth, so keep them declarative.

Two flavours of scenario:
  * ACCEPTANCE — asserts healthy behaviour we expect to hold (discovery,
    degradation ordering, ranging, drift adaptation).
  * CHARACTERIZATION — pins a KNOWN pathology the 15-node scale surfaces
    (ACK-storm implosion, hidden terminals).  These document a real effect; if a
    future MAC change fixes it, the test flips and should be re-baselined.

Healthy baseline (validated: full 15-node mutual discovery, 210/210):
  first_beacon_stagger_s=0.5   — de-syncs first beacons; without it node id 0
                                 never gets a clean beacon out (its zero backoff
                                 offset makes it collide forever);
  ack_stagger_s=12             — spreads discovery ACKs so they don't implode the
                                 channel (the dominant 15-node congestion source);
  cw_s=8, rechirp 22–32, airtime 1.0.

Run:  python tests/meshtests.py                 (Tier 1, all scenarios)
      python tests/meshtests.py --only discovery15_clean
      python tests/meshtests.py --emit-dir out/ (also write per-scenario events.jsonl)
      python tests/meshtests.py --tier dsp       (route configs through Tier 2 — TODO)
"""
from __future__ import annotations

import argparse
import math
import os
import sys


# --- category-aware sys.path bootstrap (auto-inserted) ---
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_TESTS_ROOT = _HERE.parent if _HERE.name in ("CSS", "MESH", "OFDM", "QAM") else _HERE
_REPO_ROOT = _TESTS_ROOT.parent
for _p in (_TESTS_ROOT, _REPO_ROOT, _TESTS_ROOT / "CSS", _TESTS_ROOT / "MESH",
           _TESTS_ROOT / "OFDM", _TESTS_ROOT / "QAM", _TESTS_ROOT / "regression"):
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# --- end bootstrap ---

import meshevents as ev                     # noqa: E402
from meshsim_mac import MacSim              # noqa: E402

# ── geometry helpers ─────────────────────────────────────────────────────────
def ring(nid, n=15, r=1.8):
    a = 2 * math.pi * nid / n
    return (r * math.cos(a), r * math.sin(a))


def line(nid, spacing=6.0):
    return (nid * spacing, 0.0)


HEALTHY = dict(cw_s=8.0, id_slot_s=0.1, rechirp_min_s=22.0, rechirp_max_s=32.0,
               ack_stagger_s=12.0, first_beacon_stagger_s=0.5)

BUCKETS_15 = (["clean"] * 5 + ["noisy"] * 5 + ["very_noisy"] * 5)


# ── scenario definitions ─────────────────────────────────────────────────────
class Scenario:
    def __init__(self, name, kind, build, run_s, check, schedule=None, notes=""):
        self.name = name
        self.kind = kind            # "acceptance" | "characterization"
        self.build = build          # (sim) -> None  (adds nodes)
        self.run_s = run_s
        self.check = check          # (sim) -> True | "FAIL: ..."
        self.schedule = schedule or []   # [(t, fn(sim))]
        self.notes = notes


def _add_ring(sim, n=15, profiles=None, drifts=None, **kw):
    for nid in range(n):
        sim.add(nid, pos=ring(nid, n),
                profile=(profiles or ["clean"] * n)[nid],
                drift_ppm=(drifts or [0.0] * n)[nid], **kw)


# 1. discovery15_clean — ACCEPTANCE
def build_discovery15(sim):
    _add_ring(sim, 15, **HEALTHY)


def check_discovery15(sim):
    bad = {nid: sorted(set(range(15)) - {nid} - sim.discovered_by(nid))
           for nid in range(15)}
    bad = {k: v for k, v in bad.items() if v}
    return True if not bad else f"FAIL: incomplete discovery, missing={bad}"


# 2. noisy_buckets — ACCEPTANCE (degradation ordering)
def build_noisy_buckets(sim):
    _add_ring(sim, 15, profiles=BUCKETS_15, **HEALTHY)


def _mean_disc(sim, ids):
    return sum(len(sim.discovered_by(i)) for i in ids) / len(ids)


def check_noisy_buckets(sim):
    clean = [i for i in range(15) if BUCKETS_15[i] == "clean"]
    noisy = [i for i in range(15) if BUCKETS_15[i] == "noisy"]
    vnoisy = [i for i in range(15) if BUCKETS_15[i] == "very_noisy"]
    mc, mn, mv = _mean_disc(sim, clean), _mean_disc(sim, noisy), _mean_disc(sim, vnoisy)
    # clean must stay well-connected; discovery must degrade monotonically by bucket
    if mc < 12.0:
        return f"FAIL: clean nodes poorly connected (mean {mc:.1f}/14)"
    if not (mc >= mn >= mv):
        return f"FAIL: not monotonic by bucket (clean {mc:.1f} noisy {mn:.1f} vnoisy {mv:.1f})"
    if mc - mv < 0.5:
        return f"FAIL: buckets not distinguishable (clean {mc:.1f} vs vnoisy {mv:.1f})"
    return True


# 3. ack_storm — CHARACTERIZATION (synchronized ACKs congest but discovery is
#    RESILIENT).  A synchronized start (no first-beacon stagger, tight ACK hold)
#    makes a burst of ACKs collide — the channel is measurably congested.  It used
#    to also collapse discovery; now that discovery fires on ANY decoded packet
#    (not beacons alone), the very ACKs that congest also CARRY discovery, so the
#    mesh still syncs.  The test now asserts: elevated collisions AND resilience.
def build_ack_storm(sim):
    _add_ring(sim, 15, cw_s=8.0, id_slot_s=0.1, rechirp_min_s=22.0,
              rechirp_max_s=32.0, ack_stagger_s=1.0, first_beacon_stagger_s=0.0)


def check_ack_storm(sim):
    full = sum(1 for nid in range(15) if len(sim.discovered_by(nid)) == 14)
    total_rx = sim.stats["rx_ok"] + sim.stats["rx_fail"]
    coll_rate = sim.stats["collision"] / total_rx if total_rx else 0.0
    if coll_rate < 0.05:
        return f"FAIL: expected elevated collisions from the ACK storm, got {coll_rate*100:.1f}%"
    if full < 12:
        return f"FAIL: ACK storm collapsed discovery to {full}/15 (expected resilient)"
    return True


# 4. ranging_contention — ACCEPTANCE (ranging EXCHANGE survives contention)
#    NOTE: Tier 1 asserts the two-way exchange COMPLETES under background
#    beacons, not the distance value.  Airtime here is abstract and MAC backoff
#    dominates the turnaround, so the distance is meaningless — sample-accurate
#    ranging is validated by css_modem_sim.py and, over real audio, by Tier 2.
RANGE_PAIRS = [(0, 4), (1, 6), (3, 7)]


def build_ranging(sim):
    _add_ring(sim, 8, cw_s=6.0, id_slot_s=0.1, rechirp_min_s=20.0,
              rechirp_max_s=30.0, ack_stagger_s=6.0, first_beacon_stagger_s=0.5)


def _drive_ranging(sim):
    for a, b in RANGE_PAIRS:
        sim.initiate_range(a, b)


def check_ranging(sim):
    got = {(r["node"], r["peer"]) for r in sim.ranges}
    missing = [p for p in RANGE_PAIRS if p not in got]
    if missing:
        return f"FAIL: ranging exchange did not complete for {missing} (got {sorted(got)})"
    return True


# 5. load_sweep — ACCEPTANCE (offered load drives collisions up)
def run_load_sweep():
    """Standalone: sweeps re-chirp interval; returns pass/fail on the ordering
    that shorter re-chirp (higher load) yields a higher collision rate."""
    points = []
    for rmin, rmax in [(60, 80), (30, 45), (15, 22), (8, 12)]:
        sim = MacSim(airtime_s=1.0)
        _add_ring(sim, 15, cw_s=8.0, id_slot_s=0.1, rechirp_min_s=rmin,
                  rechirp_max_s=rmax, ack_stagger_s=12.0, first_beacon_stagger_s=0.5)
        sim.run(300.0)
        total_rx = sim.stats["rx_ok"] + sim.stats["rx_fail"]
        cr = sim.stats["collision"] / total_rx if total_rx else 0.0
        load = 15 * 1.0 / ((rmin + rmax) / 2)
        points.append((load, cr, sim.stats["tx"]))
    # collision rate should rise from the lowest-load to the highest-load point
    lo_cr, hi_cr = points[0][1], points[-1][1]
    ok = hi_cr >= lo_cr
    return ok, points


# 6. drift_spread — ACCEPTANCE (per-peer re-chirp adapts to drift at scale)
def build_drift_spread(sim):
    drifts = [(-200 + 400 * (nid / 14.0)) for nid in range(15)]  # -200..+200 ppm
    _add_ring(sim, 15, drifts=drifts, cw_s=8.0, id_slot_s=0.1,
              rechirp_min_s=10.0, rechirp_max_s=3000.0, ack_stagger_s=12.0,
              first_beacon_stagger_s=0.5)


def check_drift_spread(sim):
    # a node that has heard a fast-drifting peer must have pulled its re-chirp
    # interval below the ceiling; a node hearing only near-zero-drift peers holds
    # near the ceiling.  Compare the extreme-drift node to a central one.
    iv = {nid: sim.members[nid]["daemon"].rechirp_interval_s for nid in range(15)}
    fast = iv[0]        # -200 ppm node
    if not any(sim.members[nid]["daemon"].peers for nid in range(15)):
        return "FAIL: no peer drift observed"
    if fast >= 3000.0 - 1e-6:
        return f"FAIL: fast-drift node held the ceiling ({fast:.1f}s)"
    return True


# 7. hidden_terminal — CHARACTERIZATION (spread line → far pairs can't hear)
def build_hidden_terminal(sim):
    # 20 m spacing: adjacent links ≈14 dB (audible), but the 100 m endpoints sit
    # at ≈0 dB — below the 6 dB capture margin — so the chain has genuine hidden
    # terminals.  (Path loss is 40 − 20·log10(d) dB at ref 1 m / exp 2.)
    for nid in range(6):
        sim.add(nid, pos=line(nid, spacing=20.0), profile="clean",
                cw_s=6.0, id_slot_s=0.1, rechirp_min_s=18.0, rechirp_max_s=28.0,
                ack_stagger_s=6.0, first_beacon_stagger_s=0.5)


def check_hidden_terminal(sim):
    # endpoints (0 and 5) are 100 m apart: far beyond audibility, so they must
    # NOT discover each other, while adjacent neighbours do.
    if 5 in sim.discovered_by(0) or 0 in sim.discovered_by(5):
        return "FAIL: distant endpoints should not hear each other"
    if 1 not in sim.discovered_by(0):
        return "FAIL: adjacent neighbours 0–1 failed to discover"
    return True


# 8. intermittent_sync — ACCEPTANCE (all 15 sync despite a hard, time-varying
#    subset).  A small set of nodes is deliberately made difficult with a ROLLING
#    noise schedule so they miss beacons early and must rely on peers'
#    RE-BEACONING to catch up.  The pass bar is binary: every node discovers all
#    14 others.  The exact timing is meant to be swept in a later suite.
#
#    Bucket decodability at ring distance (≈3.6 m worst-case, capture margin 6 dB):
#      clean ≈29 dB, half_noisy ≈19 dB, noisy ≈11 dB — all DECODE; only
#      very_noisy ≈1 dB is genuinely DEAF.  So a real "difficult" window must be
#      very_noisy: the node hears nothing for 30 s, then partially recovers to
#      half_noisy and has to re-acquire every peer from their next beacon.
INTERMITTENT_NODES = [4, 8, 12]
INTERMITTENT_SCHEDULE = [(0.0, "very_noisy"), (30.0, "half_noisy")]


def build_intermittent_sync(sim):
    for nid in range(15):
        if nid in INTERMITTENT_NODES:
            sim.add(nid, pos=ring(nid, 15), profile=INTERMITTENT_SCHEDULE[0][1],
                    profile_schedule=INTERMITTENT_SCHEDULE, **HEALTHY)
        else:
            sim.add(nid, pos=ring(nid, 15), profile="clean", **HEALTHY)


def check_intermittent_sync(sim):
    bad = {nid: sorted(set(range(15)) - {nid} - sim.discovered_by(nid))
           for nid in range(15)}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        return f"FAIL: not fully synced — missing={bad}"
    return True


SCENARIOS = [
    Scenario("discovery15_clean", "acceptance", build_discovery15, 600.0,
             check_discovery15, notes="full 15-node mutual discovery"),
    Scenario("intermittent_sync", "acceptance", build_intermittent_sync, 600.0,
             check_intermittent_sync,
             notes="all 15 sync despite a rolling-noise difficult subset"),
    Scenario("noisy_buckets", "acceptance", build_noisy_buckets, 600.0,
             check_noisy_buckets, notes="discovery degrades by noise bucket"),
    Scenario("ack_storm", "characterization", build_ack_storm, 600.0,
             check_ack_storm, notes="synchronized ACKs implode the channel"),
    Scenario("ranging_contention", "acceptance", build_ranging, 300.0,
             check_ranging, schedule=[(30.0, _drive_ranging)],
             notes="two-way ranging under background beacons"),
    Scenario("drift_spread", "acceptance", build_drift_spread, 60.0,
             check_drift_spread, notes="per-peer re-chirp adapts to ±200 ppm"),
    Scenario("hidden_terminal", "characterization", build_hidden_terminal, 240.0,
             check_hidden_terminal, notes="6 nodes on a 30 m line; far pairs deaf"),
]


# ── runner ───────────────────────────────────────────────────────────────────
def run_scenario(sc: Scenario, emit_dir=None, airtime_s=1.0):
    log = None
    if emit_dir:
        os.makedirs(emit_dir, exist_ok=True)
        log = ev.EventLog(os.path.join(emit_dir, f"{sc.name}.jsonl"),
                          {"run_id": sc.name, "tier": "mac", "kind_of": sc.kind,
                           "notes": sc.notes})
    sim = MacSim(airtime_s=airtime_s, eventlog=log)
    sc.build(sim)
    for t, fn in sorted(sc.schedule):
        sim.run(t)
        fn(sim)
    sim.run(sc.run_s)
    if log:
        log.close()
    return sc.check(sim), sim


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the 15-node mesh testlist.")
    ap.add_argument("--only", help="run only this scenario by name")
    ap.add_argument("--emit-dir", help="also write per-scenario events.jsonl here")
    ap.add_argument("--tier", default="mac", choices=["mac", "dsp"],
                    help="mac = Tier 1 (default); dsp = Tier 2 (not yet implemented)")
    args = ap.parse_args()

    if args.tier == "dsp":
        print("Tier 2 (DSP-in-loop) not yet implemented — see meshsim_dsp.py (Phase 4).")
        return 2

    scenarios = SCENARIOS
    if args.only:
        scenarios = [s for s in SCENARIOS if s.name == args.only]
        if not scenarios:
            print(f"no scenario named {args.only!r}")
            return 2

    print("MESH TESTLIST — TIER 1 (MAC-level)\n")
    failures = 0
    for sc in scenarios:
        try:
            result, sim = run_scenario(sc, emit_dir=args.emit_dir)
            extra = (f"[tx {sim.stats['tx']} rx_ok {sim.stats['rx_ok']} "
                     f"coll {sim.stats['collision']} disc {sim.stats['discovered']}]")
        except Exception as e:
            import traceback
            traceback.print_exc()
            result, extra = f"EXCEPTION {type(e).__name__}: {e}", ""
        tag = "PASS" if result is True else "FAIL"
        if result is not True:
            failures += 1
        print(f"  {tag}  {sc.name:20} ({sc.kind[:4]})  {extra}")
        if result is not True:
            print(f"        {result}")

    # load_sweep is its own shape (multi-run), reported separately
    if not args.only or args.only == "load_sweep":
        ok, pts = run_load_sweep()
        print(f"\n  {'PASS' if ok else 'FAIL'}  load_sweep           (acce)  "
              f"collision rate by load:")
        for load, cr, tx in pts:
            print(f"        load~{load:.2f}  collisions {cr*100:5.1f}%  tx {tx}")
        if not ok:
            failures += 1

    print(f"\n*** {'ALL PASS' if not failures else str(failures)+' FAILED'} ***")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
