# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""CSS acquisition/tracking state-machine trace harness.

    python tests/regression/css_sm_regress.py           # compare to golden
    python tests/regression/css_sm_regress.py --update  # re-record golden
    python tests/regression/css_sm_regress.py --only wake -v
    python tests/regression/css_sm_regress.py --show wake_15s

css_enc_regress and css_dec_regress hand the same bytes to C and Python and
compare the answers.  The acquisition/tracking loop inside decode_longsym_mf
has no such pure-function shape: it is a state machine over a streaming
buffer (acquire on the first strong chirp, confirm the grid phase, track
drift, reject a false lock, freewheel across a suspend gap, wake on the
resumed frame), and css_regress.py only asserts the final decoded string. A
port that acquires late, relocks where it should not, or reaches the right
text down a different path still passes that test. This harness grades the
state machine itself.

Each case runs the Python decoder offline over a fixed input and records the
sequence of state-machine events it logged.  That trace is compared against a
stored golden trace, field by field, with per-field tolerances (FIELD_TOL).
A change in WHAT the decoder decided, or WHEN, fails; a change in the last
digit of an SNR does not.

The contract is the decoder's own MF_* log lines, not a separate trace hook:
those lines are printed at the points where the state machine commits to a
decision and are what an operator reads when a run goes wrong, so the C port
has to emit the same log and the two implementations stay diffable in the
field.

MF_PRE_REJECT, the preamble confidence gate, is not reached by any case here;
see the COVERAGE note by CASES.  Timing-loop internals below the event level
(per-symbol timing error, grid least-squares fit, median filter) are visible
only through their effect on grid_t0/grid_sps at the events that report them,
so pinning every intermediate does not turn the trace into an implementation
lock.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGRESS = os.path.join(TESTS, "regression")
GOLDEN = os.path.join(REGRESS, "fixtures", "sm_traces.json")

sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(TESTS, "CSS"))
sys.path.insert(0, REGRESS)

import css_regress                     # noqa: E402  (fake-sounddevice harness)
import css_sm_sweep                    # noqa: E402  (perturbation families)


# ── which events are the contract, and how tightly ───────────────────────────
#
# An event's fields split three ways:
#
#   exact      integers and identities the state machine DECIDED -- the symbol
#              index it acted on, the sequence number it accepted, the sample
#              it anchored the grid to.  A port that differs here made a
#              different decision and must fail.
#   tolerant   floats that describe the same decision but are computed through
#              the whole DSP chain, so they carry float32 round-off and, for a
#              C port, a different libm.  Compared to a tolerance chosen to be
#              far below the smallest difference that would change a decision.
#   ignored    everything else: derived restatements (a time in seconds next to
#              the sample index it came from), thresholds echoed for the human
#              reader, and the decoded text, which css_regress already asserts.
#
# Anything not listed is ignored, so adding a field to a log line does not
# break the trace -- only adding it HERE makes it part of the contract.

EVENTS = ("MF_ACQ_CANDIDATE", "MF_RETRO", "MF_ACQUIRE", "MF_PREAMBLE",
          "MF_PRE_REJECT", "MF_PRE_CADENCE_REJECT", "MF_PRE_WEAK_REJECT",
          "MF_WAKE", "MF_RELOCK",
          "MF_LOCK_SUSPECT", "MF_FRAME", "MF_FRAME_ERROR",
          # PLL path.  The suite is otherwise all MF_* because that is the
          # path the C port targets, but the early-arrival guard lives only in
          # the streaming decoder and had no contract at all.
          "SYNC_TOO_EARLY")

EXACT_FIELDS = ("sym", "seq", "grid_anchor", "sps_eff", "delta", "reason",
                "confirms", "smp", "verdict")

# field -> (absolute tolerance, relative tolerance); a value passes if it is
# within EITHER.
#
# The relative term is not decoration.  MF_ACQ_CANDIDATE's `ratio` is a
# peak-over-noise-floor figure that reads ~50 on a marginal capture and ~5e7 on
# a clean synthetic one, because the "noise" underneath a noise-free signal is
# arithmetic round-off (see corr_noise_floor).  An absolute tolerance is
# meaningless across seven orders of magnitude, and the DECISION that ratio
# feeds is captured separately and exactly as `verdict`.
FIELD_TOL = {
    # Sample positions.  One sample at 48 kHz is 21 us; the decisions these
    # feed (which symbol slot a peak falls in) move on scales of hundreds of
    # samples, so +-2 is generous for round-off and nowhere near a decision.
    "grid_t0": (2.0, 0.0),
    "new_grid_t0": (2.0, 0.0),
    "at_sample": (2.0, 0.0),
    "buf_start": (2.0, 0.0),
    # Grid rate in samples/symbol.  Drift correction acts on tenths of a
    # sample per symbol, so half a sample is round-off, not a tracking change.
    "grid_sps": (0.5, 0.0),
    # Detection statistics.  Gates on these sit at 0.5-0.85, so 0.05 cannot
    # flip an accept/reject that was not already on the knife edge -- and one
    # that WAS is a real behavioural difference worth failing on.
    "match": (0.05, 0.0),
    "cstr": (0.05, 0.0),
    # Unbounded ratio statistics: relative, for the reason above.
    "contrast": (0.10, 0.10),
    "ratio": (0.50, 0.10),
    # SNR is a report, not a gate, at every event that carries it.
    "snr": (1.5, 0.0),
    "t": (0.03, 0.0),
}

_KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(-?[\d.]+(?:/\d+)?|\[[^\]]*\]|[A-Za-z_]+)")


def parse_trace(log: str):
    """stdout -> [{"ev": name, ...fields}] for the events in EVENTS.

    Deliberately forgiving about the prose around the fields: the log lines
    are written for humans and carry parenthetical restatements, arrows and
    em-dashes.  Only `key=value` pairs are lifted, which is why a field has to
    be spelled that way in the log to be part of the contract.
    """
    out = []
    for line in log.splitlines():
        line = line.strip()
        name = line.split(" ", 1)[0] if line else ""
        if name not in EVENTS:
            continue
        ev = {"ev": name}
        for k, val in _KV.findall(line):
            if k in ("need", "gate", "conf_bg", "word", "text", "f0", "f1"):
                continue
            ev[k] = val
        # "-> ACCEPT" / "-> REJECT" is the DECISION the candidate statistics
        # were gathered to make, and unlike them it is exact.  It is not a
        # key=value pair in the log, so it is lifted separately rather than
        # left out of the contract.
        m = re.search(r"->\s*(ACCEPT|REJECT)\b", line)
        if m:
            ev["verdict"] = m.group(1)
        # "MF_RETRO merged 1234 smp": the count is positional, not key=value,
        # and it is the one number that event exists to report.
        if name == "MF_RETRO":
            m = re.search(r"merged (\d+) smp", line)
            if m:
                ev["smp"] = m.group(1)
        out.append(ev)
    return out


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def diff_traces(got, want):
    """[] if equivalent, else a list of human-readable differences."""
    problems = []
    if len(got) != len(want):
        gs = [e["ev"] for e in got]
        ws = [e["ev"] for e in want]
        # Report the first place the sequences diverge, not just the lengths:
        # "13 events vs 12" is nearly useless, "an extra MF_RELOCK at index 7"
        # says what changed.
        i = next((i for i, (a, b) in enumerate(zip(gs, ws)) if a != b),
                 min(len(gs), len(ws)))
        problems.append(f"event count {len(got)} != {len(want)}; "
                        f"diverges at #{i}: got {gs[i:i + 3]} "
                        f"want {ws[i:i + 3]}")
        return problems

    for i, (g, w) in enumerate(zip(got, want)):
        if g["ev"] != w["ev"]:
            problems.append(f"#{i}: event {g['ev']} != {w['ev']}")
            continue
        for k, wv in w.items():
            if k == "ev":
                continue
            if k not in g:
                problems.append(f"#{i} {g['ev']}: field {k} missing")
                continue
            gv = g[k]
            if k in EXACT_FIELDS:
                if gv != wv:
                    problems.append(f"#{i} {g['ev']}: {k}={gv} != {wv}")
            elif k in FIELD_TOL:
                atol, rtol = FIELD_TOL[k]
                a, b = _num(gv), _num(wv)
                if a is None or b is None:
                    if gv != wv:
                        problems.append(f"#{i} {g['ev']}: {k}={gv} != {wv}")
                elif abs(a - b) > atol + rtol * abs(b):
                    problems.append(f"#{i} {g['ev']}: {k}={a} vs {b} "
                                    f"(tol {atol}+{rtol}x)")
    return problems


# ── cases ────────────────────────────────────────────────────────────────────
#
# Every case is SYNTHETIC and generated from a seed, not a recorded WAV.  A
# recording would pin the trace to a file that a fresh clone does not have
# (css_regress skips those), and this suite's whole value is being runnable
# anywhere the port is being worked on.  Noise is drawn from a fixed seed so
# the trace is reproducible; that same determinism is what lets the golden be
# a checked-in file rather than a re-measurement.

def _noisy(pcm, snr_db, seed):
    """Additive white noise at a given SNR relative to the signal's RMS."""
    rng = np.random.default_rng(seed)
    sig_rms = float(np.sqrt(np.mean(pcm ** 2))) or 1e-9
    n_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    return (pcm + rng.normal(0.0, n_rms, pcm.shape)).astype(np.float32)


def case_clean(dur=0.5):
    """Cold acquisition on a clean signal: the reference path."""
    wav, info = css_regress.encode_synth(f"sm_clean_{dur}", symbol_dur=dur)
    pcm, fs = css_regress.load_wav(wav)
    return pcm, fs, info


def case_noisy(dur=0.5, snr_db=6.0, seed=7):
    """The same encode with noise: exercises the CFAR acquisition gate and
    the preamble confidence gate, which a clean signal walks straight past."""
    wav, info = css_regress.encode_synth(f"sm_noisy_{dur}", symbol_dur=dur)
    pcm, fs = css_regress.load_wav(wav)
    return _noisy(pcm, snr_db, seed), fs, info


def case_wake(silence_s=15.0):
    """Freewheel across a suspend gap and wake on the resumed frame.

    The single most intricate path in the state machine: the grid has to coast
    with no signal for the whole gap and still land the resumed frame inside
    the per-symbol track window.  MF_WAKE firing at the right SYMBOL INDEX is
    the assertion that matters, and it is invisible to a decoded-text check.
    """
    wav, info = css_regress.encode_synth(
        "sm_wake", symbol_dur=0.5, preamble_mode="wake", wake_word=[1, 0],
        inter_frame_silence_s=silence_s)
    pcm, fs = css_regress.load_wav(wav)
    return pcm, fs, info


def case_dropped():
    """Excised samples mid-stream (the ADC-overflow emulation).

    Frame 1's training arrives early, so the decoder has to re-acquire from
    the next preamble rather than ride its freewheeling grid.  Note this does
    NOT reach MF_RELOCK -- it is handled entirely by per-frame preamble
    re-sync; see the COVERAGE note below.
    """
    wav, info = css_regress.make_dropped("sm_drop")
    pcm, fs = css_regress.load_wav(wav)
    return pcm, fs, info


def case_marginal(dur=0.5, snr_db=-28.0, seed=5):
    """Deep noise: sync holds but a frame fails its CRC.

    The chirp gives ~44 dB of processing gain at these symbol lengths, so this
    has to be driven a long way below 0 dB input SNR before anything breaks --
    at -20 dB the decoder is still clean.  -28 dB is the band where the state
    machine acquires, rejects one false candidate, finds a preamble, and then
    loses the frame, which is the only synthetic input found that reaches
    MF_FRAME_ERROR at all.

    It is marginal BY CONSTRUCTION and that is the point: it pins the
    behaviour at the edge, where a port is most likely to differ.  If it ever
    fails, check whether the trace changed shape (a real difference) or merely
    slid one event either side of the cliff (re-record with --update).
    """
    wav, info = css_regress.encode_synth(f"sm_marg_{dur}", symbol_dur=dur)
    pcm, fs = css_regress.load_wav(wav)
    return _noisy(pcm, snr_db, seed), fs, info


def case_halfsym_shift(dur=0.5, gap_frac=0.05):
    """Half a symbol of silence spliced in mid-stream.

    Everything after the splice sits half a slot off the grid the decoder
    locked to.  The recorded behaviour is that frame 0 decodes and frame 1 is
    LOST -- the decoder does not recover the second frame.  Pinning a known
    partial failure is as much the job here as pinning a success: a port that
    silently starts recovering frame 1 has changed the tracking loop, and a
    port that loses frame 0 too has broken it.
    """
    wav, info = css_regress.encode_synth(f"sm_shift_{dur}", symbol_dur=dur)
    pcm, fs = css_regress.load_wav(wav)
    sps_eff = int(fs * dur) + int(fs * dur * gap_frac)
    mid = len(pcm) // 2
    pad = np.zeros(sps_eff // 2, dtype=np.float32)
    return np.concatenate([pcm[:mid], pad, pcm[mid:]]), fs, info


def case_lead_noise(dur=0.5, seed=11):
    """Two seconds of noise BEFORE the signal.

    Acquisition has to reject noise-only candidates before the real chirp
    arrives.  The failure this catches is a port that locks onto the first
    thing above its threshold and then spends the run relocking -- which can
    still decode the text, just later and by luck.
    """
    wav, info = css_regress.encode_synth(f"sm_lead_{dur}", symbol_dur=dur)
    pcm, fs = css_regress.load_wav(wav)
    rng = np.random.default_rng(seed)
    lead = rng.normal(0.0, float(np.max(np.abs(pcm))) * 0.05,
                      int(fs * 2.0)).astype(np.float32)
    return np.concatenate([lead, pcm]), fs, info


def case_silence_only(dur=0.5, seed=13):
    """Noise, no signal at all.

    The negative case, and the reason corr_noise_floor exists: a decoder that
    reports a confident lock on pure noise is broken in a way that no
    positive test can see.  The assertion is that the trace contains no
    MF_ACQUIRE at all.
    """
    wav, info = css_regress.encode_synth(f"sm_sil_{dur}", symbol_dur=dur)
    pcm, fs = css_regress.load_wav(wav)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(np.max(np.abs(pcm))) * 0.05,
                       len(pcm)).astype(np.float32)
    return noise, fs, info


def _perturbed(name, dur, spec):
    """Base encode at `dur` with one css_sm_sweep perturbation applied.

    The perturbation code is IMPORTED from the sweep rather than copied here.
    The sweep is what found these inputs; if its idea of "an echo at 0.29 of a
    symbol" ever drifts from this file's, the promoted case stops pinning the
    thing that was actually discovered, and nothing would say so.
    """
    wav, info = css_regress.encode_synth(name, symbol_dur=dur)
    pcm, fs = css_regress.load_wav(wav)
    full = dict(spec, dur=dur)
    return css_sm_sweep.perturb(full, pcm, fs), fs, info


def case_relock_echo(dur=1.0):
    """Loud echo at 0.29 of a symbol -> the false-lock watchdog fires.

    Found by css_sm_sweep: an echo phase-stable and strong enough to become
    the buffer's global MF peak re-phases the grid onto it.  gain=1.0 makes
    the echo the argmax -- the watchdog takes the buffer maximum, so a
    quieter echo is invisible to it however badly it corrupts the symbol
    decisions.

    Recorded behaviour is a partial decode: frame 0 survives, the grid
    re-phases repeatedly, frame 1 is lost.  A port that recovers both frames
    has changed the watchdog; one that loses both has broken it.
    """
    return _perturbed(f"sm_relock_echo_{dur}", dur,
                      {"family": "echo", "frac": 0.2941, "gain": 1.0})


def case_relock_shift(dur=1.0):
    """A 0.24-symbol timing discontinuity mid-stream -> relock.

    The same watchdog reached by a different mechanism: no competing signal,
    just everything after the splice sitting off the locked grid by more than
    track_search (0.15 of a symbol).  Kept alongside the echo case because the
    two exercise the watchdog from opposite sides -- one where the WRONG peak
    is strongest, one where the RIGHT peak has moved.
    """
    return _perturbed(f"sm_relock_shift_{dur}", dur,
                      {"family": "shift", "frac": 0.2353, "at": 0.55})


def case_watchdog_gated(dur=0.5):
    """The relock echo at 0.5 s symbols, where the watchdog is GATED OFF.

    mf_lock_verify_min_dur defaults to 1.0 s, so below it the watchdog never
    evaluates.  This case asserts that gate directly: the identical
    perturbation that fires MF_RELOCK five times at 1.0 s must produce no
    watchdog event at all here.  Measured across the sweep, not assumed --
    64 perturbations at 0.5 s produced zero watchdog events between them,
    while the same families at 1.0 s produced 98.

    Without this, a port that simply forgot the duration gate would light up
    the watchdog at every symbol length and still pass every other case.
    """
    return _perturbed(f"sm_wdog_off_{dur}", dur,
                      {"family": "echo", "frac": 0.2941, "gain": 1.0})


def case_pll_training_falselock(dur=0.2, train=20, thr=0.30):
    """PLL path: a Barker sidelobe off the alternating training run-in.

    The condition the early-arrival guard exists for, forced rather than hoped
    for: a long training run-in and a LOW sync threshold, so a window sitting
    entirely inside the training pattern can score a pass.  The guard must
    reject those, and it must reject them for the right REASON -- the trace
    records `alt`, which reads ~1.0 on training and ~0.58 on a real Barker.

    The guard is a data test (on the `alt` statistic), not a positional one:
    a positional version silently rejects genuine preambles under a wobbling
    clock.  A future change that disables the guard entirely leaves this case
    decoding FEWER frames, not more -- with the guard off, every false lock
    is accepted and all four frames are lost.
    """
    wav, info = css_regress.encode_synth(
        f"sm_pll_falselock_{dur}", symbol_dur=dur,
        message="".join(chr(65 + i) * 8 for i in range(4)),
        chunk_size=8, training_symbols=train)
    pcm, fs = css_regress.load_wav(wav)
    # A clean encode does NOT reach the guard: with no noise the training
    # confidences are near-perfect and uniform, and the sidelobe that trips it
    # needs the small confidence variation noise supplies.  Recorded at 0
    # events before this line was added, which is a pin on nothing.
    return _noisy(pcm, 30.0, 4242), fs, info


CASES = [
    ("clean_0.5s", lambda: case_clean(0.5)),
    ("clean_1.0s", lambda: case_clean(1.0)),
    ("noisy_6dB", lambda: case_noisy(0.5, 6.0)),
    ("lead_noise", lambda: case_lead_noise(0.5)),
    ("wake_15s", lambda: case_wake(15.0)),
    ("dropped_4sym", case_dropped),
    ("halfsym_shift", lambda: case_halfsym_shift(0.5)),
    ("marginal_-28dB", lambda: case_marginal(0.5, -28.0)),
    ("silence_only", case_silence_only),
    # Drop the Barker bit-match gate far enough that noise patterns match the
    # preamble on bits, so the frame-cadence lattice becomes the thing that
    # refuses them.  Driven by a config override rather than a crafted signal
    # because the gate, not the input, is what decides, and mf_preamble_thr
    # is a real knob an operator can set.  All rejections in this case are
    # the cadence guard, not the confidence gate (MF_PRE_REJECT); see COVERAGE.
    ("cadence_reject", lambda: case_noisy(0.5, 6.0), {"mf_preamble_thr": 0.40}),
    # False-lock watchdog, found by css_sm_sweep (see the COVERAGE note).
    ("pll_training_falselock",
     lambda: case_pll_training_falselock(0.2, 20, 0.30),
     {"sync_threshold": 0.30}),
    ("relock_echo", lambda: case_relock_echo(1.0)),
    ("relock_shift", lambda: case_relock_shift(1.0)),
    ("watchdog_gated_0.5s", lambda: case_watchdog_gated(0.5)),
]

# Events that MUST NOT appear in a case's trace.  The mirror of
# MUST_NOT_ACQUIRE: "the golden happens to contain no MF_RELOCK" and "firing
# the watchdog here would be a bug" are different statements, and only the
# second survives someone re-recording the golden on a broken build.
MUST_NOT_EMIT = {
    "watchdog_gated_0.5s": ("MF_RELOCK", "MF_LOCK_SUSPECT"),
    "silence_only": ("MF_ACQUIRE", "MF_PREAMBLE", "MF_FRAME"),
}

# COVERAGE, as measured from the recorded traces rather than asserted here.
#
# One event is NOT reached: MF_PRE_REJECT, the noise-referenced confidence
# gate on an otherwise-matching preamble.  Reaching it needs a preamble that
# matches on bits, sits ON the cadence lattice, and still reads low
# confidence -- a three-way coincidence in the same family as MF_RELOCK, so
# it wants the sweep rather than a hand-built input:
#
#     python tests/regression/css_sm_sweep.py --target MF_PRE_REJECT
#
# The other two -- MF_RELOCK and MF_LOCK_SUSPECT, the false-lock watchdog --
# were found by tests/regression/css_sm_sweep.py, a sweep over 320
# perturbations. Two gates limit what reaches them:
#
#   1. The watchdog is gated on mf_lock_verify_min_dur, which defaults to
#      1.0 s.  `dur` is a swept axis in the sweep for this reason, and
#      watchdog_gated_0.5s pins the gate itself.
#   2. It keys on the buffer's GLOBAL MF peak, so a competing signal has to
#      be at least as loud as the direct path (gain >= 1.0) to be seen at
#      all; a quieter echo corrupts symbol decisions without reaching the
#      watchdog.
#
# The sweep measured 35 of 320 variants reaching MF_RELOCK and 98 reaching
# MF_LOCK_SUSPECT at 1.0 s, against 0 of 64 at 0.5 s.  Re-run it after any
# change to the acquisition or tracking gates:
#
#     python tests/regression/css_sm_sweep.py --jobs 40

# Cases where the decoder must NOT acquire at all.  Kept as an explicit list
# rather than inferred from an empty golden trace, for the same reason as
# MUST_NOT_EMIT above: "the golden happens to be empty" and "acquiring here
# would be a bug" are different statements.
MUST_NOT_ACQUIRE = {"silence_only"}


def run_case(name, make, overrides=None):
    pcm, fs, info = make()
    res, log = css_regress.decode_offline(pcm, fs, info, overrides=overrides)
    trace = parse_trace(log)
    return trace, res, log


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="run only cases whose name contains this")
    ap.add_argument("--update", action="store_true",
                    help="re-record the golden traces from this run")
    ap.add_argument("--show", metavar="CASE",
                    help="print one case's trace and exit")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the decoder log for failing cases")
    args = ap.parse_args()

    cases = [c for c in CASES if not args.only or args.only in c[0]]

    if args.show:
        for case in CASES:
            name = case[0]
            if args.show in name:
                trace, res, log = run_case(*case)
                print(f"--- {name}: status={res.get('status')} "
                      f"text={res.get('decoded_text')!r}")
                for i, ev in enumerate(trace):
                    print(f"  {i:3d} {ev['ev']:<18} "
                          + " ".join(f"{k}={v}" for k, v in ev.items()
                                     if k != "ev"))
                return 0
        print(f"no case matching {args.show!r}")
        return 1

    golden = {}
    if os.path.exists(GOLDEN):
        with open(GOLDEN) as fh:
            golden = json.load(fh)

    if not args.update and not golden:
        print(f"no golden traces at {GOLDEN}")
        print("  record them with: python tests/regression/css_sm_regress.py "
              "--update")
        return 1

    print(f"CSS STATE-MACHINE TRACE -- {len(cases)} cases"
          + (" (RECORDING)" if args.update else "") + "\n")
    failed = 0
    recorded = dict(golden)
    for case in cases:
        name = case[0]
        try:
            trace, res, log = run_case(*case)
        except Exception as e:      # noqa: BLE001
            failed += 1
            print(f"  ERROR {name:<16} {type(e).__name__}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue

        acquired = any(e["ev"] == "MF_ACQUIRE" for e in trace)
        summary = (f"{len(trace)} events, "
                   f"{sum(1 for e in trace if e['ev'] == 'MF_FRAME')} frames, "
                   f"status={res.get('status')}")

        if name in MUST_NOT_ACQUIRE and acquired:
            failed += 1
            print(f"  FAIL  {name:<16} acquired sync on a signal-free input "
                  f"({summary})")
            continue

        banned = [e for e in MUST_NOT_EMIT.get(name, ())
                  if any(x["ev"] == e for x in trace)]
        if banned:
            failed += 1
            print(f"  FAIL  {name:<16} emitted {', '.join(banned)}, which this "
                  f"case exists to rule out ({summary})")
            continue

        if args.update:
            recorded[name] = trace
            print(f"  REC   {name:<16} {summary}")
            continue

        if name not in golden:
            failed += 1
            print(f"  FAIL  {name:<16} no golden trace -- run --update")
            continue

        problems = diff_traces(trace, golden[name])
        if problems:
            failed += 1
            print(f"  FAIL  {name:<16} {summary}")
            for p in problems[:6]:
                print(f"          {p}")
            if len(problems) > 6:
                print(f"          ... and {len(problems) - 6} more")
            if args.verbose:
                print(log)
        else:
            print(f"  PASS  {name:<16} {summary}")

    if args.update:
        # `recorded` starts from the existing golden so that --update --only
        # re-records one case without deleting the rest.  The cost is that a
        # renamed case leaves its old trace behind forever, checked by
        # nothing and quietly implying coverage the suite no longer has.  On
        # a full run every case is accounted for, so anything left over is
        # an orphan and can go; on a narrowed run it cannot be told from a
        # case that simply did not run.
        if not args.only:
            orphans = [k for k in recorded if k not in {c[0] for c in CASES}]
            for k in orphans:
                del recorded[k]
            if orphans:
                print(f"\ndropped {len(orphans)} orphaned trace(s): "
                      f"{', '.join(sorted(orphans))}")
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as fh:
            json.dump(recorded, fh, indent=1, sort_keys=True)
        print(f"\nwrote {GOLDEN} ({len(recorded)} cases)")
        return 1 if failed else 0

    print()
    if failed:
        print(f"*** {failed} FAILED ***")
        return 1
    print("*** ALL PASS ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
