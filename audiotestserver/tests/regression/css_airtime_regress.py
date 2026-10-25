# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""CSS per-frame run-in: airtime bought, and what it costs. Pass/fail suite.

    python tests/regression/css_airtime_regress.py            # full suite
    python tests/regression/css_airtime_regress.py --only wake
    python tests/regression/css_airtime_regress.py -v         # per-cell detail
    python tests/regression/css_airtime_regress.py --update   # re-record budgets

css_wake_sweep.py sweeps a grid and prints a report for a human to read. This
pins the resulting budget numbers so a decoder change that erodes them fails
a test instead of being found on the air. Same split as css_sm_sweep (search)
and css_sm_regress (contract).

Each candidate is a per-frame run-in -- a training-symbol count, or a wake
word replacing the Barker run-in -- scored on three numbers:

  airtime    seconds per frame.  Pure geometry, no decode, exact.  Asserted
             exactly because any change means the frame layout moved.
  delivery   worst-case fraction of resumed chunks decoded, across every
             impairment cell in the grid.
  false/hr   worst-case false acquisitions per hour of idle listening, across
             every cell.  Each false acquisition burns a frame window in
             COLLECT, so the receiver is deaf when the next real chunk
             arrives.

All three are asserted together: airtime alone is a spec, not a test;
delivery alone lets a candidate pass by acquiring on everything it hears;
false/hr alone lets one pass by acquiring on nothing.

Budgets use the worst cell in the grid, not the mean: a candidate that fails
on one condition is not represented by its average.  Failures name the cell
that broke.

Budgets are measured, not aspirational: --update re-records them from a run
of this grid, with headroom applied so ordinary numerical drift does not
fail the suite.  budgets.json records when they were last recorded.

The grid is deliberately small -- one geometry, four candidates, three
impairment cells, two trials -- so this stays runnable next to the other
regressions (~5 min).  css_wake_sweep covers the full surface.
"""
import argparse
import json
import os
import sys
import time

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGRESS = os.path.join(TESTS, "regression")
BUDGETS = os.path.join(REGRESS, "fixtures", "airtime_budgets.json")

sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(TESTS, "CSS"))
sys.path.insert(0, REGRESS)

import css_wake_sweep as WS                        # noqa: E402


# ── the grid ─────────────────────────────────────────────────────────────────
#
# One geometry: symbol_dur=0.5 s with the PHY's chunk_size -- the shortest
# symbol duration where the MF decoder (and therefore the wake gate) runs at
# all.  The production 0.12 s profile is on the PLL path and has no wake path
# to test; it is covered by the airtime assertions only.
#
# Impairment cells are chosen to separate the candidates, not to look most
# impressive.  Pink at 12 dB and drift at 2000 ppm are comfortable for
# everything and catch a gross regression; clicks at 8/s spreads the wake
# words from 67% delivery to 14%, so it carries the actual discrimination.
CELLS = [
    ("pink", 12.0),
    ("drift", 2000.0),
    ("click", 8.0),
]

CANDIDATES = ["per_frame_full", "per_frame_t4", "wake_w4", "wake_w2"]

DUR = 0.5
CHUNK = 8
FRAMES = 6
GAP_S = 10.0
IDLE_S = 600.0
TRIALS = 2
SEED = 0

# Headroom on the recorded budgets.  Delivery and false-rate are measured over
# a few dozen frames, so they carry real sampling noise; airtime is geometry
# and carries none.
DELIVER_SLACK = 0.15      # fraction of resumed chunks
FALSE_SLACK = 2.0         # per hour, added to the recorded worst case
FALSE_REL = 0.5           # ...or +50%, whichever is larger
AIRTIME_TOL = 0.05        # seconds


def _specs(candidate, leg):
    out = []
    for fam, lv in CELLS:
        for t in range(TRIALS):
            key = f"{leg}|{candidate}|{fam}|{lv}|{DUR}"
            out.append({
                "leg": leg, "candidate": candidate, "family": fam,
                "level": lv, "dur": DUR, "trial": t, "gap_s": GAP_S,
                "idle_s": IDLE_S, "message": WS.make_message(FRAMES, CHUNK),
                "chunk": CHUNK, "dec_overrides": {},
                "seed": WS._seed(key, t, SEED),
            })
    return out


def measure(candidates, jobs, verbose=False):
    """Run the grid; return {candidate: {...worst-case metrics...}}."""
    import concurrent.futures as cf
    specs = []
    for c in candidates:
        specs += _specs(c, "wake") + _specs(c, "idle")
    jobs = max(1, min(jobs, len(specs)))
    print(f"  {len(specs)} decodes on {jobs} workers "
          f"({len(candidates)} candidates x {len(CELLS)} cells x "
          f"{TRIALS} trials x 2 legs)", flush=True)

    results = []
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(WS.run_variant, s): s for s in specs}
        done = 0
        for fut in cf.as_completed(futs, timeout=1800 * len(specs)):
            r = fut.result()
            results.append(r)
            done += 1
            if not r.get("ok"):
                print(f"    ERR {WS.spec_label(r['spec'])}: {r['error']}",
                      flush=True)
            elif verbose:
                s = r["spec"]
                d = (f"cold={int(r['cold'])} deliver={r['deliver']*100:3.0f}%"
                     if s["leg"] == "wake"
                     else f"false={r['false_acq']} ({r['false_per_hr']:.1f}/hr)")
                print(f"    {WS.spec_label(s):<44} t{s['trial']} {d}",
                      flush=True)
            elif done % 10 == 0:
                print(f"    ... {done}/{len(specs)} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    agg = WS.aggregate(results)
    out = {}
    for c in candidates:
        air = WS.airtime(c, DUR, GAP_S, WS.make_message(FRAMES, CHUNK), CHUNK)
        cells = {}
        for fam, lv in CELLS:
            w = agg.get(("wake", c, fam, lv), {})
            i = agg.get(("idle", c, fam, lv), {})
            cells[f"{fam}@{lv:g}"] = {
                "deliver": w.get("deliver"), "cold": w.get("cold"),
                "false_per_hr": i.get("false_per_hr"),
            }
        dl = [v["deliver"] for v in cells.values() if v["deliver"] is not None]
        cd = [v["cold"] for v in cells.values() if v["cold"] is not None]
        fa = [v["false_per_hr"] for v in cells.values()
              if v["false_per_hr"] is not None]
        out[c] = {
            "frame_s": air["frame_s"], "resume_syms": air["resume_syms"],
            "worst_deliver": min(dl) if dl else None,
            "worst_deliver_cell": (min(cells, key=lambda k: cells[k]["deliver"])
                                   if dl else None),
            "worst_cold": min(cd) if cd else None,
            "worst_false_per_hr": max(fa) if fa else None,
            "worst_false_cell": (max(cells, key=lambda k: cells[k]["false_per_hr"])
                                 if fa else None),
            "cells": cells,
        }
    return out


def check(got, want):
    """[] if `got` is within budget, else a list of human-readable failures."""
    bad = []
    if want is None:
        return ["no recorded budget -- run --update"]
    if abs(got["frame_s"] - want["frame_s"]) > AIRTIME_TOL:
        bad.append(f"airtime {got['frame_s']:.2f}s != {want['frame_s']:.2f}s "
                   f"(tol {AIRTIME_TOL})  -- the frame layout moved")
    if got["worst_cold"] is not None and got["worst_cold"] < 1.0:
        bad.append(f"cold acquisition {got['worst_cold']*100:.0f}% < 100% "
                   f"-- frame 0 is not landing")
    wd, bd = got["worst_deliver"], want["worst_deliver"]
    if wd is not None and bd is not None and wd < bd - DELIVER_SLACK:
        bad.append(f"worst delivery {wd*100:.0f}% < {bd*100:.0f}%"
                   f"-{DELIVER_SLACK*100:.0f}% at {got['worst_deliver_cell']}")
    wf, bf = got["worst_false_per_hr"], want["worst_false_per_hr"]
    if wf is not None and bf is not None:
        cap = max(bf + FALSE_SLACK, bf * (1 + FALSE_REL))
        if wf > cap:
            bad.append(f"worst false rate {wf:.1f}/hr > {cap:.1f}/hr "
                       f"(recorded {bf:.1f}) at {got['worst_false_cell']}")
    return bad


def render(got, want, cands):
    """The breakdown table -- the thing this suite exists to show."""
    lines = ["",
             "  candidate        run-in    frame   vs base   worst      worst      worst",
             "                     syms        s             cold    deliver   false/hr"]
    base = got.get("per_frame_full", {}).get("frame_s")
    for c in cands:
        g = got[c]
        rel = (f"{(1 - g['frame_s'] / base) * 100:+6.1f}%"
               if base else "     --")
        lines.append(
            f"  {c:<15} {g['resume_syms']:>7d} {g['frame_s']:>8.2f} {rel:>9}  "
            f"{_pct(g['worst_cold']):>7} {_pct(g['worst_deliver']):>10} "
            f"{_num(g['worst_false_per_hr']):>10}")
    lines.append("")
    lines.append("  worst = the single worst cell in the grid, not the mean: a "
                 "candidate with one")
    lines.append("  condition it cannot survive is not an average candidate.  "
                 "Cells: "
                 + ", ".join(f"{f}@{l:g}" for f, l in CELLS))
    return "\n".join(lines)


def _pct(v):
    return "-" if v is None else f"{v * 100:.0f}%"


def _num(v):
    return "-" if v is None else f"{v:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="substring filter on candidate names")
    ap.add_argument("--update", action="store_true",
                    help="re-record the budgets from this run")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes (default: cores, capped at 16)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every decode as it finishes")
    args = ap.parse_args()

    cands = [c for c in CANDIDATES if not args.only or args.only in c]
    if not cands:
        print(f"no candidate matches {args.only!r}; known: "
              f"{', '.join(CANDIDATES)}")
        return 1
    jobs = args.jobs or min(16, len(os.sched_getaffinity(0)))

    print(f"CSS AIRTIME / DELIVERY BREAKDOWN -- {len(cands)} candidates"
          + (" (RECORDING)" if args.update else ""))
    print(f"  geometry: symbol_dur={DUR}s chunk={CHUNK}B frames={FRAMES} "
          f"gap={GAP_S}s idle={IDLE_S}s trials={TRIALS}")
    t0 = time.time()
    got = measure(cands, jobs, args.verbose)

    want = {}
    if os.path.exists(BUDGETS):
        want = json.load(open(BUDGETS)).get("candidates", {})

    print(render(got, want, cands))

    if args.update:
        os.makedirs(os.path.dirname(BUDGETS), exist_ok=True)
        # Merge, so --update --only re-records one candidate without
        # discarding the rest (same rationale as css_sm_regress's golden).
        merged = dict(want)
        merged.update(got)
        with open(BUDGETS, "w") as fh:
            json.dump({
                "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "geometry": {"dur": DUR, "chunk": CHUNK, "frames": FRAMES,
                             "gap_s": GAP_S, "idle_s": IDLE_S,
                             "trials": TRIALS, "seed": SEED,
                             "cells": CELLS},
                "candidates": merged,
            }, fh, indent=1)
        print(f"\nwrote {BUDGETS} ({len(merged)} candidates)")
        return 0

    print()
    failed = 0
    for c in cands:
        bad = check(got[c], want.get(c))
        if bad:
            failed += 1
            print(f"  FAIL  {c}")
            for b in bad:
                print(f"          {b}")
        else:
            print(f"  PASS  {c}")

    print(f"\n{time.time() - t0:.0f}s wall")
    if failed:
        print(f"*** {failed} FAILED ***")
        return 1
    print("*** ALL PASS ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
