# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""CSS state-machine perturbation sweep — a SEARCH tool, not a pass/fail suite.

    python tests/regression/css_sm_sweep.py                    # full sweep
    python tests/regression/css_sm_sweep.py --family echo -v
    python tests/regression/css_sm_sweep.py --target MF_RELOCK
    python tests/regression/css_sm_sweep.py --jobs 40 --limit 100
    python tests/regression/css_sm_sweep.py --promote

css_sm_regress.py pins the state machine's behaviour on a handful of
hand-built inputs; two of its events -- MF_RELOCK and MF_LOCK_SUSPECT, the
false-lock watchdog -- are not reached by any of them.  A false lock needs
the decoder to anchor on the wrong grid phase, which depends on a continuous
offset (where a decoy or echo sits within the symbol period) interacting
with gates whose thresholds are fractions of a symbol, so there is no reason
to expect the interesting offsets to be round numbers and hand-picking one
is guesswork.

This sweeps them instead.  Each variant is a small spec; workers rebuild the
base encode themselves and apply the perturbation, and the grid runs across
every core.  The output is a coverage table plus the exact specs that reached
each event, ready to promote into css_sm_regress.CASES.

The watchdog is gated on `mf_lock_verify_min_dur`, which defaults to 1.0 s;
below that it never evaluates, and every case in css_sm_regress uses 0.5 s
symbols.  `dur` is therefore a swept axis rather than a constant, and 0.5 s
is kept in the grid as a control: a family that fires at 1.0 s and not at
0.5 s is the gate working, not a flaky test.

The echo family models the field failure the watchdog is tuned around: an
acoustic echo landing near the frame boundary, phase-stable and strong
enough to pass both the ratio gate and the two-window confirmation,
re-phasing the grid onto the echo and killing an otherwise decodable run.
The sweep builds it across a range of delays and gains rather than one fixed
pair.
"""
import argparse
import collections
import json
import os
import sys
import traceback

import numpy as np

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGRESS = os.path.join(TESTS, "regression")

sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(TESTS, "CSS"))
sys.path.insert(0, REGRESS)


# Events the sweep is hunting for.  Anything else it finds is reported too,
# but these are the ones css_sm_regress cannot currently reach.
TARGETS = ("MF_RELOCK", "MF_LOCK_SUSPECT")


# ── worker side ──────────────────────────────────────────────────────────────
#
# Imports happen lazily inside the worker rather than at module scope: the
# parent only needs to build specs and print a table, and importing the whole
# decoder stack there would be paid again in every forked child for nothing.

_BASE = {}          # dur -> (pcm, fs, info_path), per worker process


def _base(dur):
    """The unperturbed encode for this symbol duration, cached per process.

    Encoded in the WORKER, not passed in from the parent.  A 1.0 s two-frame
    encode is ~7 million float32 samples; pickling that to each of several
    hundred tasks would cost more than the decodes.  The filename carries the
    pid because css_regress.encode_synth writes into one shared work dir and
    two processes encoding the same geometry would otherwise race on it.
    """
    if dur in _BASE:
        return _BASE[dur]
    import css_regress
    wav, info = css_regress.encode_synth(
        f"sweep_{dur}_{os.getpid()}", symbol_dur=dur)
    pcm, fs = css_regress.load_wav(wav)
    _BASE[dur] = (pcm, fs, info)
    return _BASE[dur]


def _sps_eff(fs, dur, gap_frac=0.05):
    """Samples per symbol SLOT — chirp plus the inter-symbol gap.

    The gap is part of the slot, so perturbations expressed as a fraction of a
    symbol have to be measured against this and not against int(fs*dur).  The
    two differ by 5% at the default gap, which is a third of track_search --
    enough to move a variant from inside the tracking window to outside it,
    i.e. to decide the very thing the sweep is measuring.
    """
    sps = int(fs * dur)
    return sps + int(sps * gap_frac)


def perturb(spec, pcm, fs):
    """Apply one perturbation family.  Pure: returns a new array."""
    fam = spec["family"]
    se = _sps_eff(fs, spec["dur"])
    n = len(pcm)
    # `at` is a fraction of the stream; snapped to a slot boundary so the
    # perturbation lands between symbols rather than slicing one in half,
    # unless the family is specifically about slicing one in half.
    at = int(n * spec.get("at", 0.5))

    if fam == "shift":
        # Insert `frac` of a symbol slot of silence: everything after sits
        # off the grid the decoder locked to.
        k = int(round(spec["frac"] * se))
        return np.concatenate([pcm[:at], np.zeros(k, np.float32), pcm[at:]])

    if fam == "excise":
        k = int(round(spec["frac"] * se))
        return np.concatenate([pcm[:at], pcm[at + k:]])

    if fam == "echo":
        # Delayed copy at `frac` of a slot.  gain >= 1 makes the echo the
        # GLOBAL MF peak, which is what the watchdog keys on -- it takes the
        # buffer's argmax, so a quieter echo is invisible to it no matter how
        # badly it corrupts the symbol decisions.
        d = int(round(spec["frac"] * se))
        out = np.zeros(n + d, np.float32)
        out[:n] += pcm
        out[d:d + n] += pcm * spec.get("gain", 1.0)
        return np.clip(out, -1.0, 1.0)

    if fam == "decoy":
        # A train of off-grid chirps laid over the signal: a competing symbol
        # grid, phase-stable, at an offset the tracker cannot absorb.  The
        # closest synthetic analogue of a second transmitter.
        import css_fec
        d = int(round(spec["frac"] * se))
        ch = css_fec.generate_chirp(800, 4000, spec["dur"], fs)
        out = pcm.copy()
        pos = at + d
        for _ in range(spec.get("reps", 6)):
            if pos + len(ch) > n:
                break
            out[pos:pos + len(ch)] += ch * spec.get("gain", 1.0)
            pos += se
        return np.clip(out, -1.0, 1.0)

    if fam == "prelude":
        # A decoy train BEFORE the real signal, so ACQUISITION anchors to it
        # and the real train arrives on a different phase.
        import css_fec
        ch = css_fec.generate_chirp(800, 4000, spec["dur"], fs)
        lead = np.zeros(int(se * spec.get("reps", 6)) + len(ch), np.float32)
        pos = int(round(spec["frac"] * se))
        for _ in range(spec.get("reps", 6)):
            if pos + len(ch) > len(lead):
                break
            lead[pos:pos + len(ch)] += ch * spec.get("gain", 1.0)
            pos += se
        return np.clip(np.concatenate([lead, pcm]), -1.0, 1.0)

    raise ValueError(f"unknown family {fam!r}")


def run_variant(spec):
    """Decode one perturbed input.  Returns a plain dict (picklable)."""
    try:
        import css_regress
        import css_sm_regress as SM
        pcm, fs, info = _base(spec["dur"])
        p = perturb(spec, pcm, fs)
        res, log = css_regress.decode_offline(p, fs, info,
                                              overrides=spec.get("overrides"))
        trace = SM.parse_trace(log)
        return {"spec": spec, "ok": True,
                "events": dict(collections.Counter(e["ev"] for e in trace)),
                "status": res.get("status"),
                "text": res.get("decoded_text")}
    except Exception as e:      # noqa: BLE001
        return {"spec": spec, "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc()[-800:]}


# ── the grid ─────────────────────────────────────────────────────────────────

def build_specs(families=None, durs=None, steps=16):
    """The sweep grid.

    Offsets are swept as FRACTIONS OF A SYMBOL SLOT, over the open interval
    (0, 1) with the endpoints excluded -- 0 and 1 are both "no perturbation"
    as far as grid phase is concerned, so they only add runtime.  `steps`
    controls the resolution; the default 16 puts a sample every 6% of a
    symbol, which is well inside track_search (15%) and so cannot step over
    the boundary the gates sit on.
    """
    fracs = [round((i + 1) / (steps + 1), 4) for i in range(steps)]
    durs = durs or [1.0, 0.5]
    out = []
    for dur in durs:
        for f in fracs:
            for at in (0.35, 0.55):
                out.append({"family": "shift", "dur": dur, "frac": f, "at": at})
                out.append({"family": "excise", "dur": dur, "frac": f, "at": at})
            for gain in (1.0, 1.6):
                out.append({"family": "echo", "dur": dur, "frac": f,
                            "gain": gain})
                out.append({"family": "decoy", "dur": dur, "frac": f,
                            "gain": gain, "at": 0.45})
                out.append({"family": "prelude", "dur": dur, "frac": f,
                            "gain": gain})
    if families:
        out = [s for s in out if s["family"] in families]
    return out


def spec_label(s):
    bits = [s["family"], f"dur={s['dur']}", f"frac={s['frac']}"]
    if "gain" in s:
        bits.append(f"gain={s['gain']}")
    if "at" in s and s["family"] in ("shift", "excise", "decoy"):
        bits.append(f"at={s['at']}")
    return " ".join(bits)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes (default: all available cores)")
    ap.add_argument("--family", action="append",
                    help="restrict to a family (repeatable): shift, excise, "
                         "echo, decoy, prelude")
    ap.add_argument("--dur", action="append", type=float,
                    help="restrict to a symbol duration (repeatable)")
    ap.add_argument("--steps", type=int, default=16,
                    help="offsets swept per family per duration")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many variants (smoke test)")
    ap.add_argument("--target", action="append",
                    help=f"event to hunt for (default: {', '.join(TARGETS)})")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="seconds before a single variant is abandoned")
    ap.add_argument("--promote", metavar="PATH", nargs="?",
                    const=os.path.join(REGRESS, "sm_sweep_hits.json"),
                    help="write the winning specs to PATH for promotion into "
                         "css_sm_regress.CASES")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every variant, not just the hits")
    args = ap.parse_args()

    targets = tuple(args.target or TARGETS)
    specs = build_specs(args.family, args.dur, args.steps)
    if args.limit:
        specs = specs[:args.limit]
    jobs = args.jobs or len(os.sched_getaffinity(0))
    jobs = max(1, min(jobs, len(specs)))

    print(f"CSS STATE-MACHINE SWEEP -- {len(specs)} variants on {jobs} workers")
    print(f"hunting: {', '.join(targets)}\n")

    import concurrent.futures as cf
    import time

    t0 = time.time()
    results = []
    # max_tasks_per_child is deliberately NOT set: on this Python (3.13, fork
    # start method) it hangs the pool -- every worker dies at the first
    # replacement wave and the parent waits forever on futures that can never
    # complete.  _BASE already caps what a worker retains (one encode per
    # symbol duration), so nothing needs to replace it.
    #
    # The `with` block still guarantees the pool is torn down -- an orphaned
    # worker keeps the inherited stdout pipe open and leaves the run looking
    # hung long after the work is finished.
    try:
        with cf.ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(run_variant, s): s for s in specs}
            done = 0
            for fut in cf.as_completed(futs, timeout=args.timeout * len(specs)):
                done += 1
                try:
                    r = fut.result(timeout=args.timeout)
                except Exception as e:      # noqa: BLE001
                    r = {"spec": futs[fut], "ok": False,
                         "error": f"{type(e).__name__}: {e}"}
                results.append(r)
                hit = r.get("ok") and any(t in r["events"] for t in targets)
                if hit or args.verbose or not r.get("ok"):
                    label = spec_label(r["spec"])
                    if not r.get("ok"):
                        print(f"  ERR  {label:<44} {r['error']}")
                    else:
                        mark = "HIT " if hit else "    "
                        ev = " ".join(f"{k.replace('MF_', '')}={v}"
                                      for k, v in sorted(r["events"].items()))
                        print(f"  {mark} {label:<44} "
                              f"{r['status']:<7} {r['text']!r:6} {ev}")
                if done % 50 == 0 and not args.verbose:
                    print(f"  ... {done}/{len(specs)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted -- reporting what finished")

    # ── coverage report ──────────────────────────────────────────────────────
    print(f"\n{'-' * 70}")
    ok = [r for r in results if r.get("ok")]
    errs = [r for r in results if not r.get("ok")]
    seen = collections.Counter()
    for r in ok:
        seen.update(r["events"].keys())
    print(f"{len(ok)} decoded, {len(errs)} errored, "
          f"{time.time() - t0:.0f}s wall")
    print("\nevent coverage across the sweep:")
    for ev, n in sorted(seen.items()):
        print(f"  {ev:<20} {n:4d} variants")
    for t in targets:
        if t not in seen:
            print(f"  {t:<20}    0 variants   *** NOT REACHED ***")

    hits = {}
    for t in targets:
        # Rank by how much of the target event a variant produced, then prefer
        # the SIMPLEST spec that got there: a promoted case should pin the
        # behaviour, not showcase the most violent input that reaches it.
        cands = [r for r in ok if t in r["events"]]
        cands.sort(key=lambda r: (-r["events"][t], r["spec"].get("gain", 0),
                                  r["spec"]["frac"]))
        if cands:
            hits[t] = [c["spec"] for c in cands[:5]]
            print(f"\nbest inputs for {t}:")
            for c in cands[:5]:
                ev = " ".join(f"{k.replace('MF_', '')}={v}"
                              for k, v in sorted(c["events"].items()))
                print(f"  {spec_label(c['spec']):<44} "
                      f"{c['status']:<7} {c['text']!r:6} {ev}")

    if errs and args.verbose:
        print("\nerrors:")
        for r in errs[:5]:
            print(f"  {spec_label(r['spec'])}: {r['error']}")

    if args.promote:
        with open(args.promote, "w") as fh:
            json.dump({"targets": list(targets), "hits": hits}, fh, indent=1)
        print(f"\nwrote {args.promote}")

    # Exit non-zero when a hunted event was never reached: that is the
    # actionable outcome, and it is what makes this usable from a script.
    missing = [t for t in targets if t not in seen]
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
