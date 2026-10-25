# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""benchlib.py — the impulse-obscuration benchmark, shared by the CSS and OFDM
suites.

What it measures
----------------
Two independent axes, swept separately so the tipping point is attributable:

  * COVERAGE — how many symbols get hit, as "1 in every N".  N=10 is 10% of
    symbols, N=1 is 100%.  This is the score axis.
  * WIDTH — how much of a symbol a single burst destroys, from a short
    absolute click (100 ms) up to 3/4 of a symbol.
  * ALIGNMENT — whether the burst sits at the start, middle or end of the
    symbol it hits.  Swept, never mixed.

Scoring: each (width, alignment) cell scores 0-100 = the highest coverage
percentage that still decoded.  Surviving 1-in-1 scores 100; surviving only
1-in-10 scores 10; failing even 1-in-10 scores 0.  The headline number is the
mean over cells.

Properties of the sweep
------------------------
1. Coverage is deterministic: symbols are chosen by `s % every_n == phase`,
   not by `rng.choice` over a random subset, so "10%" is exact and
   repeatable rather than a statistical statement. This also models
   periodic interference (a fan, a pump, a rotor).

2. Alignment is isolated per cell rather than blended, because start/middle/
   end are not equivalent to a decoder and the measured asymmetry is large:

     CSS  — `start` is the weakest column (0-33 at 60x), `middle` the
            strongest (100 at every width below 3/4 symbol).
     OFDM — the reverse: `end` is strongest (100 at 1/2 symbol), `start` and
            `middle` collapse to ~10 at the same width.

   A start-aligned burst does not survive by landing on the cyclic prefix
   alone: at 1/2 symbol the burst is 640 samples against a 256-sample CP, so
   most of it lands in the FFT window regardless.

3. Width is a swept axis in real units: absolute milliseconds or a fraction
   of a symbol, because the two matter at different scales — a 100 ms click
   is most of a CSS chirp but spans several OFDM symbols, and a "3/4 symbol"
   burst is the meaningful unit for the partial-obscuration recovery path.

Monotonicity
------------
Damage is assumed to rise with coverage, so a cell walks the ladder from
easiest to hardest and stops at the first failing level. `strict=True` walks
the whole ladder and reports any non-monotonicity instead of stopping early.
"""
from __future__ import annotations

import math

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# The axes
# ──────────────────────────────────────────────────────────────────────────────

# "1 in every N symbols" -> coverage percent.  Ordered easiest-first, which is
# the order a cell sweeps and the order the early exit depends on.
COVERAGE_LADDER: tuple[int, ...] = (10, 8, 6, 5, 4, 3, 2, 1)


def coverage_pct(every_n: int) -> float:
    """Percent of symbols hit by "1 in every N"."""
    return 100.0 / int(every_n)


class Width:
    """A burst duration, in whichever unit is meaningful for the modem.

    `ms` is absolute; `frac` is a fraction of a symbol period. Kept as one
    type with two constructors so a width ladder can mix them.
    """

    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value: float):
        if kind not in ("ms", "frac"):
            raise ValueError(f"width kind must be 'ms' or 'frac', got {kind!r}")
        if value <= 0:
            raise ValueError(f"width must be positive, got {value}")
        self.kind, self.value = kind, float(value)

    @classmethod
    def ms(cls, v: float) -> "Width":
        return cls("ms", v)

    @classmethod
    def frac(cls, v: float) -> "Width":
        return cls("frac", v)

    def samples(self, sps: int, fs: int) -> int:
        """Burst length in samples, clamped to the symbol.

        A burst longer than the symbol would erase it entirely plus damage
        the neighbour, a different test than partial obscuration. `label`
        reports when a clamp happened.
        """
        n = (self.value * 1e-3 * fs) if self.kind == "ms" else (self.value * sps)
        return max(1, min(int(round(n)), int(sps)))

    def label(self, sps: int | None = None, fs: int | None = None) -> str:
        base = (f"{self.value:g}ms" if self.kind == "ms"
                else f"{self.value:.2f}sym")
        if sps and fs:
            got = self.samples(sps, fs)
            if got >= sps and (self.kind == "frac" and self.value < 1.0
                               or self.kind == "ms"):
                if abs(got - sps) < 2 and not (self.kind == "frac"
                                               and abs(self.value - 1.0) < 1e-9):
                    base += "(=sym)"
        return base

    def __repr__(self):
        return f"Width({self.kind}, {self.value:g})"


ALIGNMENTS: tuple[str, ...] = ("start", "middle", "end")

# Burst amplitude, as a multiple of the signal's RMS. Per-modem calibration,
# not a universal constant: CSS and OFDM scores are comparable across runs
# but not to each other.
#
# Measured on this repo (fast profile, 0.5-0.75 symbol widths):
#   OFDM data path — dies at 6x. One obscured symbol puts ~16 byte errors
#                    into a chunk whose RS(272,256) corrects 8, with no
#                    interleaver.
#   CSS chirps     — survives 25x and still decodes at 100% coverage via the
#                    matched filter's spreading gain. Breaks up around
#                    50-80x.
#
# Pick the value that puts the coverage ladder in range for the modem under
# test: too low and every cell scores 100, too high and every cell scores 0.
DEFAULT_AMP_RATIO = 6.0        # OFDM-calibrated value; CSS overrides it

# The default width ladder: an absolute click that is short for CSS and long
# for OFDM, then fractions up to the 3/4-symbol case.  `frac` entries are
# scale-free, so this ladder means the same thing to both modems.
DEFAULT_WIDTHS: tuple[Width, ...] = (
    Width.ms(100), Width.frac(0.25), Width.frac(0.50), Width.frac(0.75),
)


# ──────────────────────────────────────────────────────────────────────────────
# Building the impairment
# ──────────────────────────────────────────────────────────────────────────────

def target_symbols(n_syms: int, every_n: int, phase: int = 0) -> np.ndarray:
    """Indices of the symbols a level obscures: every Nth, starting at `phase`.

    `phase` lets a caller check that a result is a property of the coverage
    and not of which particular symbols happened to be hit; not part of the
    default benchmark.
    """
    if every_n < 1:
        raise ValueError(f"every_n must be >= 1, got {every_n}")
    return np.arange(int(phase) % every_n, max(0, int(n_syms)), int(every_n))


def burst_offset(align: str, sps: int, width: int) -> int:
    """Where inside a symbol a burst of `width` samples begins.

    start  -> flush with the symbol boundary
    middle -> centred
    end    -> flush with the symbol's end

    A width equal to the symbol makes all three identical, which is correct and
    is why the width axis is clamped there.
    """
    if align == "start":
        return 0
    if align == "middle":
        return max(0, (sps - width) // 2)
    if align == "end":
        return max(0, sps - width)
    raise ValueError(f"align must be one of {ALIGNMENTS}, got {align!r}")


def apply_obscuration(audio, fs, sps, every_n, width, align, rng, *,
                      offset=0, active=None, amp_ratio=6.0, phase=0, clip=True):
    """Inject one burst per targeted symbol.  Returns (audio, n_bursts).

    `offset` is where symbol 0 starts in `audio` — the caller's job, since
    only the caller knows whether its buffer begins at a symbol boundary
    (CSS encodes from 0; an OFDM capture has a preamble in front).

    `sps` is the symbol stride (how far apart symbols start). `active` is
    how much of that stride actually carries signal, and defaults to all of
    it; they differ whenever a modem leaves dead time inside the symbol
    period (CSS puts a 5% silent gap after each chirp). Width and alignment
    are measured against `active` so a burst placed in the gap is not
    counted as damaging the symbol.

    Bursts are broadband noise at `amp_ratio` x the signal's RMS, loud
    enough to swamp the symbol in that window without being a full-scale
    clip. The level is referenced to overall RMS rather than the local
    window so one burst is the same insult wherever it lands.
    """
    out = np.array(audio, dtype=np.float32, copy=True)
    n = out.size
    sps = int(sps)
    if sps <= 0:
        raise ValueError(f"sps must be positive, got {sps}")
    act = sps if active is None else max(1, min(int(active), sps))

    w = width.samples(act, fs) if isinstance(width, Width) else int(width)
    w = max(1, min(w, act))
    n_syms = max(0, (n - int(offset)) // sps)
    targets = target_symbols(n_syms, every_n, phase)
    if targets.size == 0:
        return out, 0

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) or 1e-9
    amp = rms * float(amp_ratio)
    off_in_sym = burst_offset(align, act, w)

    for s in targets:
        pos = int(offset) + int(s) * sps + off_in_sym
        end = min(pos + w, n)
        if end <= pos:
            continue
        out[pos:end] += (rng.standard_normal(end - pos).astype(np.float32)
                         * amp)
    if clip:
        np.clip(out, -1.0, 1.0, out=out)
    return out, int(targets.size)


# ──────────────────────────────────────────────────────────────────────────────
# Sweeping a cell, scoring the grid
# ──────────────────────────────────────────────────────────────────────────────

class Cell(dict):
    """One (width, alignment) pair and its 0-100 score."""

    @property
    def score(self) -> float:
        return self["score"]


# Coverages below the 10% scoring floor, used only for diagnosis. A modem
# that fails the easiest scored level is a 0 either way; these levels
# distinguish a near miss from a total collapse. Not part of COVERAGE_LADDER
# since the score is defined as the coverage percentage survived.
FLOOR_LADDER: tuple[int, ...] = (20, 50, 100, 400)


def sweep_cell(run_level, width, align, *, ladder=COVERAGE_LADDER,
               pass_threshold=0.5, strict=False, label_fn=None,
               floor_ladder=None, geom=None):
    """Walk the coverage ladder for one (width, alignment) cell.

    `run_level(every_n) -> pass_rate in [0,1]` is the caller's decode, however
    it wants to parallelise or repeat it.  Returns a Cell holding the score,
    the pass/fail boundary, and every level actually evaluated.

    A level counts as passed at `pass_threshold` (default: more than half
    the trials decoded), matching the convention the existing CSS sweeps use.

    `floor_ladder` (try `FLOOR_LADDER`) is consulted only when the cell
    scores zero, to characterise how it failed; it does not affect the
    score.

    `geom` is `(active_samples, fs)`, and resolves absolute widths into
    symbol fractions for the report. At 44.1 kHz with a 250 ms chirp,
    "100 ms" is 0.40 of a symbol, wider than the `0.25sym` row printed below
    it, so the resolved value keeps the width axis ordering legible.
    """
    levels, score, last_pass, first_fail = [], 0.0, None, None
    anomalies = []

    for every_n in ladder:
        rate = float(run_level(every_n))
        pct  = coverage_pct(every_n)
        ok   = rate >= pass_threshold
        levels.append({"every_n": every_n, "coverage_pct": round(pct, 1),
                       "pass_rate": rate, "ok": ok})
        if ok:
            # A pass above a level that already failed means damage did not
            # rise monotonically with coverage; usually means the trial
            # count is too low for the noise.
            if first_fail is not None:
                anomalies.append(
                    f"passed {pct:.0f}% after failing "
                    f"{coverage_pct(first_fail):.0f}%")
            score = max(score, pct)
            last_pass = every_n
        elif first_fail is None:
            first_fail = every_n
            if not strict:
                break

    sub_floor = None
    if score == 0.0 and floor_ladder:
        # Nothing in the scored range survived; walk downward to find the
        # coverage that does.
        sub_floor = {"survived_every_n": None, "survived_pct": None,
                     "levels": []}
        for every_n in floor_ladder:
            rate = float(run_level(every_n))
            pct = coverage_pct(every_n)
            sub_floor["levels"].append(
                {"every_n": every_n, "coverage_pct": round(pct, 2),
                 "pass_rate": rate, "ok": rate >= pass_threshold})
            if rate >= pass_threshold:
                sub_floor["survived_every_n"] = every_n
                sub_floor["survived_pct"] = round(pct, 2)
                break

    label = width.label() if isinstance(width, Width) else str(width)
    resolved = {}
    if geom and isinstance(width, Width):
        act, fs = int(geom[0]), int(geom[1])
        ws = width.samples(act, fs)
        frac = round(ws / act, 3)
        resolved = {"width_samples": ws, "width_frac_of_symbol": frac,
                    "width_ms": round(ws / fs * 1e3, 2)}
        # Absolute widths get the resolved fraction appended; fractional
        # ones already state it.
        if width.kind == "ms":
            label = f"{label}={frac:.2f}sym"

    cell = Cell({
        "width": width.label() if isinstance(width, Width) else str(width),
        "_row": label,          # width as shown in the report's row heading
        "width_kind": width.kind if isinstance(width, Width) else "samples",
        "width_value": width.value if isinstance(width, Width) else float(width),
        **resolved,
        "align": align,
        "score": round(score, 1),
        "max_pass_every_n": last_pass,
        "first_fail_every_n": first_fail,
        # The tipping point, stated as the bracket it was localised to; an
        # open upper end means the cell was never broken at all.
        "tipping_point": {
            "survived_pct": round(coverage_pct(last_pass), 1) if last_pass else 0.0,
            "failed_pct": (round(coverage_pct(first_fail), 1)
                           if first_fail else None),
        },
        "levels": levels,
    })
    if sub_floor is not None:
        cell["sub_floor"] = sub_floor
    if anomalies:
        cell["anomalies"] = anomalies
    if label_fn:
        cell["label"] = label_fn(width, align)
    return cell


def score_grid(cells) -> dict:
    """Collapse a list of Cells into the benchmark report.

    `score` (the headline) is the mean over cells. `worst` is carried
    alongside because a mean can hold up while one alignment collapses to
    zero.
    """
    cells = list(cells)
    if not cells:
        return {"score": 0.0, "worst": 0.0, "best": 0.0, "n_cells": 0,
                "cells": [], "by_align": {}, "by_width": {}}
    scores = [c["score"] for c in cells]

    def _group(key):
        out = {}
        for c in cells:
            out.setdefault(c[key], []).append(c["score"])
        return {k: {"score": round(sum(v) / len(v), 1),
                    "worst": round(min(v), 1), "n": len(v)}
                for k, v in out.items()}

    return {
        "score":   round(sum(scores) / len(scores), 1),
        "worst":   round(min(scores), 1),
        "best":    round(max(scores), 1),
        "n_cells": len(cells),
        "by_align": _group("align"),
        "by_width": _group("width"),
        "cells":   cells,
        "anomalies": [f"{c['width']}/{c['align']}: {a}"
                      for c in cells for a in c.get("anomalies", [])],
    }


def format_grid(report, title="impulse benchmark") -> str:
    """The score surface as a fixed-width table: widths down, alignments across.

    Printed by both suites so CSS and OFDM scores are read the same way.
    """
    cells = report.get("cells", [])
    if not cells:
        return f"{title}: no cells"
    # `_row` is the width as sweep_cell resolved it: absolute widths carry
    # their symbol fraction.
    for c in cells:
        c.setdefault("_row", c["width"])
    widths = list(dict.fromkeys(c["_row"] for c in cells))
    aligns = list(dict.fromkeys(c["align"] for c in cells))
    by = {(c["_row"], c["align"]): c for c in cells}

    wcol = max(9, max(len(w) for w in widths) + 1)
    head = "  " + "width".ljust(wcol) + "".join(a.rjust(9) for a in aligns)
    lines = [f"{title}: {report['score']:.1f}/100  "
             f"(worst cell {report['worst']:.1f}, best {report['best']:.1f})",
             head, "  " + "-" * (wcol + 9 * len(aligns))]
    for w in widths:
        row = "  " + w.ljust(wcol)
        for a in aligns:
            c = by.get((w, a))
            row += ("-" if c is None else f"{c['score']:.0f}").rjust(9)
        lines.append(row)

    lines.append("")
    lines.append("  tipping points (survived% -> failed%)")
    for c in cells:
        tp = c["tipping_point"]
        died = f"{tp['failed_pct']:.0f}%" if tp["failed_pct"] else "never failed"
        line = (f"    {c['_row']:>14s} {c['align']:<7s} "
                f"survived {tp['survived_pct']:5.1f}%  ->  died at {died}")
        sf = c.get("sub_floor")
        if sf:
            # A zero score says only "failed at 10%"; this says how far
            # below the floor the real limit sits.
            line += ("  [below floor: survives "
                     f"{sf['survived_pct']:.2f}%]" if sf["survived_pct"]
                     else "  [below floor: fails even at "
                          f"{coverage_pct(sf['levels'][-1]['every_n']):.2f}%"
                          f" — a single burst is fatal]")
        lines.append(line)
    for a in report.get("anomalies", []):
        lines.append(f"  ANOMALY: {a}")
    return "\n".join(lines)


def grid_axes(widths=None, aligns=None):
    """The (width, align) cells of a benchmark, in report order."""
    widths = DEFAULT_WIDTHS if widths is None else tuple(widths)
    aligns = ALIGNMENTS if aligns is None else tuple(aligns)
    return [(w, a) for w in widths for a in aligns]


def _selftest() -> int:
    """`python tests/benchlib.py` — checks the impairment geometry itself,
    on a synthetic signal where the answer is known by construction.
    """
    fails = []

    def check(name, cond):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            fails.append(name)

    fs, sps, n_syms = 1000, 100, 20
    audio = np.ones(sps * n_syms, dtype=np.float32)
    rng = np.random.default_rng(0)

    # coverage is exact, not statistical
    for every_n in COVERAGE_LADDER:
        t = target_symbols(n_syms, every_n)
        check(f"1-in-{every_n} hits {math.ceil(n_syms / every_n)} of {n_syms}",
              t.size == math.ceil(n_syms / every_n))
    check("1-in-1 is 100%", coverage_pct(1) == 100.0)
    check("1-in-10 is 10%", coverage_pct(10) == 10.0)

    # alignment puts the burst where it claims
    w = Width.frac(0.25)
    ws = w.samples(sps, fs)
    check("0.25 symbol at sps=100 is 25 samples", ws == 25)
    for align, want in (("start", 0), ("middle", 37), ("end", 75)):
        check(f"{align} burst starts at +{want}",
              burst_offset(align, sps, ws) == want)

    # and the damage really is confined to that window
    for align in ALIGNMENTS:
        got, n = apply_obscuration(audio, fs, sps, 1, w, align, rng, clip=False)
        diff = np.nonzero(np.abs(got - audio) > 1e-6)[0]
        o = burst_offset(align, sps, ws)
        in_window = ((diff % sps) >= o) & ((diff % sps) < o + ws)
        check(f"{align}: all damage inside the declared window",
              bool(in_window.all()) and n == n_syms)

    # 100 ms at fs=1000 is 100 samples = the whole symbol -> clamped
    check("width clamps to one symbol", Width.ms(100).samples(sps, fs) == sps)
    check("width never exceeds the symbol",
          Width.frac(3.0).samples(sps, fs) == sps)

    # scoring: a cell that survives everything is 100, one that survives
    # nothing is 0, and the boundary is reported both ways
    all_pass = sweep_cell(lambda n: 1.0, w, "start")
    check("survives 1-in-1 -> 100", all_pass["score"] == 100.0)
    check("never-failed has no upper bracket",
          all_pass["tipping_point"]["failed_pct"] is None)
    all_fail = sweep_cell(lambda n: 0.0, w, "end")
    check("fails 1-in-10 -> 0", all_fail["score"] == 0.0)
    check("first fail recorded", all_fail["first_fail_every_n"] == 10)
    # passes at <=20% coverage, fails above
    mid = sweep_cell(lambda n: 1.0 if coverage_pct(n) <= 20.0 else 0.0,
                     w, "middle")
    check("tipping point localises to 20 -> 25",
          mid["score"] == 20.0
          and mid["tipping_point"]["failed_pct"] == 25.0)

    # non-monotonic decoders are reported, not smoothed away
    odd = sweep_cell(lambda n: 0.0 if n == 8 else 1.0, w, "end", strict=True)
    check("non-monotonic run raises an anomaly", bool(odd.get("anomalies")))

    rep = score_grid([all_pass, all_fail, mid])
    check("grid mean is the headline", rep["score"] == round(120 / 3, 1))
    check("worst cell carried separately", rep["worst"] == 0.0)

    # An absolute width resolves against the symbol: 100 ms at 44.1 kHz
    # against a 250 ms chirp is 0.40 of a symbol.
    css_act, css_fs = 11025, 44100
    g = sweep_cell(lambda n: 0.0, Width.ms(100), "start", geom=(css_act, css_fs))
    check("100ms resolves to 4410 samples", g["width_samples"] == 4410)
    check("100ms is 0.40 of a 250ms chirp",
          g["width_frac_of_symbol"] == 0.4)
    check("row label carries the resolved fraction",
          "0.40sym" in format_grid(score_grid([g])))
    # A fractional width needs no resolution and must not grow a suffix.
    gf = sweep_cell(lambda n: 0.0, Width.frac(0.25), "start",
                    geom=(css_act, css_fs))
    check("fractional widths label unchanged",
          gf["_row"] == "0.25sym")
    # Sub-floor diagnosis: distinguishes a near miss from a total collapse.
    sf_none = sweep_cell(lambda n: 0.0, w, "start", floor_ladder=FLOOR_LADDER)
    check("total collapse records no sub-floor survivor",
          sf_none["sub_floor"]["survived_pct"] is None)
    sf_some = sweep_cell(lambda n: 1.0 if n >= 50 else 0.0, w, "start",
                         floor_ladder=FLOOR_LADDER)
    check("near miss records where it does survive",
          sf_some["sub_floor"]["survived_pct"] == 2.0)
    check("sub-floor never changes the score", sf_some["score"] == 0.0)
    print()
    print(format_grid(rep, "selftest"))
    print()
    print("BENCHLIB SELFTEST", "PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
