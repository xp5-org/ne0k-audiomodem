# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Two-tier parameter matrix over the CSS decode chain, as one tracked report.

    python tests/regression/css_matrix.py \\
        --outer sync_threshold=0.10:0.70:0.10 \\
        --inner symbol_dur=0.15:0.45:0.10 \\
        --jobs 24 --trials 3

    python tests/regression/css_matrix.py --list-params
    python tests/regression/css_matrix.py --outer mf_min_snr_margin_db=0:10:2 \\
        --inner wobble=400:2000:400 --impair wobble --metric deliver

css_wake_sweep answers "which of these candidates is best"; this answers
"where does a continuous parameter stop working, and does that boundary move
when a second parameter changes".  Spot checks cannot answer the second
question: a knife-edge that every training value sits on is only visible once
the surface either side of it is drawn.

A run sweeps one OUTER parameter against an INNER parameter, every cell
repeated `--trials` times.  Output is built to be read as a SHAPE: curves
that show where behaviour rises, peaks and falls, backed by the numbers that
drew them.

OUTPUT.  Everything lands in one run directory under results/, in the same
shape the hardware tests write, so the Reports tab lists it without special
casing:

    matrix.json    every cell, every metric, plus the axes and the spec that
                   produced them.  The machine-readable copy.
    matrix.svg     the same data drawn.  SVG rather than PNG: axis values,
                   legend and point labels are real <text> nodes, readable by
                   something that cannot see the image.
    manifest.json  run_id / test / params / started_at, matching the existing
                   runs so shared.list_results() and run_report() work unchanged.

WHAT A CELL IS SCORED ON.  The same three metrics as css_airtime_regress, from
the same worker (css_wake_sweep.run_variant): `cold` (frame 0 decoded),
`deliver` (fraction of resumed frames) and `false_per_hr` (false acquisitions
during idle listening).  --metric picks which one the curves are drawn from;
all three are in the JSON regardless.

COST.  cells = outer_steps x inner_steps x trials x legs.  The default example
above is 7 x 4 x 3 x 1 = 84 decodes, a few minutes on 24 workers.  Check the
printed estimate before a wide grid; --dry-run shows the shape without
running it.
"""
import argparse
import json
import math
import os
import sys
import time

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGRESS = os.path.join(TESTS, "regression")
REPO = os.path.dirname(TESTS)

sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(TESTS, "CSS"))
sys.path.insert(0, REGRESS)
sys.path.insert(0, REPO)

import css_wake_sweep as WS                       # noqa: E402


# ── what can be swept, and where each parameter goes ─────────────────────────
#
# A swept name has to be routed to one of three places, and getting it wrong is
# silent: an encoder parameter passed as a decoder override changes nothing and
# the sweep reports a beautifully flat surface.  So the routing is a table, not
# a guess, and --list-params prints it.
#
#   "enc"      encoder kwarg (css_encoder.encode) -- changes what goes on the
#              wire, so the cache key and the decode info sidecar both move.
#   "dec"      css_decoder config override -- changes only how it is read.
#   "impair"   the impairment level for the family named by --impair.
#
# Adding a row is the whole cost of sweeping something new.
PARAMS = {
    # decoder gates
    "sync_threshold":      ("dec", float, "PLL preamble correlation threshold"),
    "pll_weight":          ("dec", float, "PLL timing correction weight"),
    "syndet_margin":       ("dec", float, "sync detector margin over background"),
    "timing_search_frac":  ("dec", float, "timing search window, fraction of a symbol"),
    "mf_min_snr_db":       ("dec", float, "MF strong/weak gate (absolute; prefer the margin)"),
    "mf_min_snr_margin_db": ("dec", float, "MF strong gate above the measured noise floor"),
    "mf_preamble_thr":     ("dec", float, "Barker bit-match gate"),
    "mf_pre_strong_frac":  ("dec", float, "fraction of preamble symbols that must be chirps"),
    "mf_cadence_reset_weak": ("dec", int, "weak symbols that drop the cadence lattice"),
    # The MF/PLL dispatch threshold.  Sweeping it to its extremes compares the
    # two decoders on identical audio: 0.0 forces decode_longsym_mf at every
    # symbol_dur, 99.0 forces the streaming PLL at every symbol_dur.  See the
    # note in css_decoder's dispatch block for the shipped default.
    "mf_longsym_min_dur":  ("dec", float, "MF/PLL dispatch threshold in seconds "
                                          "(0 = always MF, 99 = always PLL)"),
    "mf_longsym":          ("dec", int,   "0 disables the MF decoder entirely"),
    "retro_rate_adopt_inlier": ("dec", float, "inlier frac before the retro drift "
                                              "fit becomes the next frame's rate "
                                              "baseline (>1.0 disables adoption)"),
    # encoder geometry
    "symbol_dur":          ("enc", float, "chirp length in seconds"),
    "training_symbols":    ("enc", int,   "training chirps before the preamble"),
    "chunk_size":          ("enc", int,   "payload bytes per frame"),
    "symbol_gap_frac":     ("enc", float, "inter-symbol gap, fraction of a symbol"),
    "amplitude":           ("enc", float, "transmit amplitude"),
    # impairment level (family chosen by --impair)
    "level":               ("impair", float, "impairment level for --impair family"),
    "snr":                 ("impair", float, "alias for level, with --impair pink"),
    "ppm":                 ("impair", float, "alias for level, with --impair drift"),
    "wobble":              ("impair", float, "alias for level, with --impair wobble"),
    "rt60":                ("impair", float, "alias for level, with --impair reverb"),
    # The impairment's OWN frequency, so a feature found in the response can be
    # tested for whether it moves with the test signal (an artifact of the
    # stimulus) or stays put (a property of the decoder).  Without this axis a
    # narrow notch is indistinguishable from a resonance with the sweep's own
    # wobble period.
    "wobble_hz":           ("spec", float, "wobble frequency in Hz (--impair wobble)"),
    # The wobble's start phase, in radians.  Sweep it whenever a wobble result
    # ranks geometries against each other (training_symbols, symbol_dur,
    # chunk_size -- anything that moves where a frame's acquisition lands in
    # the cycle).  Left at its default the wobble is a single clock
    # realisation: every trial redraws the noise but not the clock, so a
    # candidate that happens to acquire on the wobble's steepest slope loses
    # the same frames in every trial and reads as reproducible without being
    # resampled.
    "wobble_phase":        ("spec", float, "wobble start phase in radians "
                                           "(--impair wobble); sweep 0:6.28 "
                                           "to separate clock phase from "
                                           "geometry"),
}

METRICS = {
    "deliver":      ("wake", "resumed chunks delivered", "%"),
    "cold":         ("wake", "frame 0 acquired", "%"),
    "false_per_hr": ("idle", "false acquisitions per hour idle", "/hr"),
}

IMPAIR_ALIAS = {"snr": "pink", "ppm": "drift", "wobble": "wobble", "rt60": "reverb"}


def parse_axis(spec):
    """'name=start:stop:step' -> (name, [values]).

    Inclusive of `stop` when it lands on the grid, with a small epsilon so
    0.1:0.7:0.1 yields 0.7 rather than stopping at 0.6 on binary round-off.
    """
    if "=" not in spec:
        raise ValueError(f"axis needs name=start:stop:step, got {spec!r}")
    name, rng = spec.split("=", 1)
    name = name.strip()
    if name not in PARAMS:
        raise ValueError(f"unknown parameter {name!r}; --list-params to see them")
    parts = rng.split(":")
    if len(parts) == 1:
        vals = [float(x) for x in parts[0].split(",")]
    elif len(parts) == 3:
        start, stop, step = (float(x) for x in parts)
        if step <= 0:
            raise ValueError(f"step must be > 0 in {spec!r}")
        n = int(math.floor((stop - start) / step + 1e-9)) + 1
        vals = [start + i * step for i in range(max(1, n))]
    else:
        raise ValueError(f"axis needs name=start:stop:step or name=a,b,c, "
                         f"got {spec!r}")
    kind, cast, _ = PARAMS[name]
    vals = [cast(round(v, 6)) for v in vals]
    return name, vals


def apply_param(name, value, enc, dec, impair):
    kind = PARAMS[name][0]
    if kind == "enc":
        enc[name] = value
    elif kind == "dec":
        dec[name] = value
    elif kind == "spec":
        impair[name] = value           # passed straight through on the spec
    else:
        impair["level"] = value
    return kind


# ── where to put the samples ─────────────────────────────────────────────────
#
# Every cell is an independent decode with its own seed and no state carries
# between cells, so traversal order does not affect the result.  The real
# limitation of a grid is that it only samples on its own lines.  Two things
# fix that, both about WHERE the points are rather than the order they are
# visited:
#
#   * put points off the lattice, so combinations between grid lines are
#     reachable at all (`diag`, `random`);
#   * put more points where the surface is actually changing, which on these
#     cliff-shaped surfaces is the only region whose location is in doubt
#     (`refine`).

def slice_z(grid, zv):
    """The 2-D sub-grid at one third-axis value, keyed (outer, inner).

    Three axes are presented as a STACK OF PLANES rather than a single fused
    view: a curve can show two axes honestly and a third only by pretending
    depth on a flat page.  Every plane is then rendered and summarised by the
    same 2-D code, so the third axis costs no new reporting logic and a 3-axis
    result reads as "the same chart, once per z".
    """
    return {(o, i): c for (o, i, z), c in grid.items() if z == zv}


def with_third(points, zvals):
    if not zvals:
        return [(o, i, None) for (o, i) in points]
    return [(o, i, z) for (o, i) in points for z in zvals]


def _lin(a, b, n):
    if n <= 1:
        return [a]
    return [a + (b - a) * k / (n - 1) for k in range(n)]


def sample_grid(ovals, ivals):
    return [(ov, iv) for ov in ovals for iv in ivals]


def sample_diag(ovals, ivals, passes=2):
    """Diagonal traversals, offset half a step on each pass.

    This is the "XX->YY, then shift and repeat" idea, implemented so that it
    actually adds information: on a plain lattice the diagonals are just grid
    points again, so each successive pass is shifted by a fraction of a step and
    lands BETWEEN the grid lines.  Both diagonal directions are walked, because
    a surface whose edge runs parallel to one of them would be invisible to that
    one alone.

    Cheaper than the full product for the same reach: n_lines x len, not
    len(o) x len(i).
    """
    if len(ovals) < 2 or len(ivals) < 2:
        return sample_grid(ovals, ivals)
    ostep = (ovals[-1] - ovals[0]) / (len(ovals) - 1)
    istep = (ivals[-1] - ivals[0]) / (len(ivals) - 1)
    n = max(len(ovals), len(ivals))
    pts = []
    for p in range(passes):
        frac = p / passes                       # 0, 1/2, ... of a step
        for start in range(n):
            for sign in (+1, -1):
                for k in range(n):
                    ov = ovals[0] + ((start + k) % n) * ostep + frac * ostep
                    iv = (ivals[0] + ((start + sign * k) % n) * istep
                          + frac * istep)
                    if (ovals[0] <= ov <= ovals[-1]
                            and ivals[0] <= iv <= ivals[-1]):
                        pts.append((round(ov, 6), round(iv, 6)))
    return sorted(set(pts))


def sample_random(ovals, ivals, n, seed):
    """Stratified (Latin-hypercube) sample over the continuous box.

    Deliberately NOT a plain uniform draw: LHS guarantees every row and column
    band of the box is hit once, so a thin feature running parallel to either
    axis cannot be missed by luck.  And deliberately not a lattice: if the
    system has any structure that happens to align with a regular step -- a
    frame period, a wobble period -- a grid can sit inside it and report a flat
    surface.  This is the sampler to reach for when a grid result looks
    suspiciously featureless.
    """
    import random
    rng = random.Random(seed)
    olo, ohi = ovals[0], ovals[-1]
    ilo, ihi = ivals[0], ivals[-1]
    orows = list(range(n))
    irows = list(range(n))
    rng.shuffle(irows)
    pts = []
    for k, j in zip(orows, irows):
        ov = olo + (ohi - olo) * (k + rng.random()) / n
        iv = ilo + (ihi - ilo) * (j + rng.random()) / n
        pts.append((round(ov, 6), round(iv, 6)))
    return sorted(set(pts))


def refine_points_plane(grid2d, metric, k, seen, zv):
    """Refinement inside ONE plane; see refine_points."""
    out = []
    for (o, i) in _steepest_midpoints(grid2d, metric):
        p = (o, i, zv)
        if p in seen:
            continue
        out.append(p)
        if len(out) >= k:
            break
    return out


def _steepest_midpoints(grid, metric):
    """Midpoints of adjacent pairs, steepest first (both axes)."""
    pts = sorted(grid)
    byo, byi = {}, {}
    for (o, i) in pts:
        byo.setdefault(o, []).append(i)
        byi.setdefault(i, []).append(o)
    cands = []
    for o, ivs in byo.items():
        ivs = sorted(ivs)
        for a, b in zip(ivs, ivs[1:]):
            va = grid.get((o, a), {}).get(metric)
            vb = grid.get((o, b), {}).get(metric)
            if va is None or vb is None:
                continue
            cands.append((abs(vb - va), (o, round((a + b) / 2, 6))))
    for i, ovs in byi.items():
        ovs = sorted(ovs)
        for a, b in zip(ovs, ovs[1:]):
            va = grid.get((a, i), {}).get(metric)
            vb = grid.get((b, i), {}).get(metric)
            if va is None or vb is None:
                continue
            cands.append((abs(vb - va), (round((a + b) / 2, 6), i)))
    cands.sort(key=lambda t: -t[0])
    return [p for d, p in cands if d > 0]


def refine_points(grid, metric, k, ovals, ivals, seen):
    """Midpoints of the steepest adjacent pairs -- adaptive edge finding.

    These surfaces are cliffs, not hills (measured: delivery goes 0% -> 100%
    between two adjacent sync_threshold steps).  On a cliff the interesting
    quantity is WHERE the edge is, and a uniform grid spends most of its budget
    confirming the two plateaus it already established.  So: rank every adjacent
    pair of sampled points by how much the metric changes across them, and put
    the next round's samples at the midpoints of the steepest ones.  Each round
    halves the uncertainty in the edge location for a handful of decodes.

    Pairs are taken along both axes.  `seen` stops a midpoint being re-run when
    a previous round already landed there.
    """
    pts = sorted({(o, i) for (o, i) in grid})
    byo, byi = {}, {}
    for (o, i) in pts:
        byo.setdefault(o, []).append(i)
        byi.setdefault(i, []).append(o)
    cands = []
    for o, ivs in byo.items():
        ivs = sorted(ivs)
        for a, b in zip(ivs, ivs[1:]):
            va = grid.get((o, a), {}).get(metric)
            vb = grid.get((o, b), {}).get(metric)
            if va is None or vb is None:
                continue
            cands.append((abs(vb - va), (o, round((a + b) / 2, 6))))
    for i, ovs in byi.items():
        ovs = sorted(ovs)
        for a, b in zip(ovs, ovs[1:]):
            va = grid.get((a, i), {}).get(metric)
            vb = grid.get((b, i), {}).get(metric)
            if va is None or vb is None:
                continue
            cands.append((abs(vb - va), (round((a + b) / 2, 6), i)))
    cands.sort(key=lambda t: -t[0])
    out = []
    for delta, p in cands:
        if delta <= 0 or p in seen:
            continue
        out.append(p)
        if len(out) >= k:
            break
    return out


def build_cells(args, outer, inner, points, third=None):
    """Specs for an explicit list of (outer, inner[, third]) points.

    A third axis is a plain extra coordinate on every point rather than a
    separate concept: the surface is still scored the same way, there is just
    one more thing varying.  It is kept OPTIONAL because two axes is what a
    curve can show, and three is what you reach for when a two-axis result is
    suspected of depending on something held constant.
    """
    oname, _ = outer
    iname, _ = inner
    zname = third[0] if third else None
    legs = sorted({METRICS[m][0] for m in args.metric})
    cells = []
    for pt in points:
        ov, iv = pt[0], pt[1]
        zv = pt[2] if (zname and len(pt) > 2) else None
        enc, dec, impair = {}, {}, {"level": args.level}
        apply_param(oname, ov, enc, dec, impair)
        apply_param(iname, iv, enc, dec, impair)
        if zname is not None:
            apply_param(zname, zv, enc, dec, impair)
        dur = enc.pop("symbol_dur", args.dur)
        for leg in legs:
            for t in range(args.trials):
                key = (f"{leg}|{oname}={ov}|{iname}={iv}"
                       + (f"|{zname}={zv}" if zname else ""))
                cells.append({
                    "leg": leg, "candidate": args.candidate,
                    "family": args.impair, "level": impair["level"],
                    "wobble_hz": impair.get("wobble_hz"),
                    "wobble_phase": impair.get("wobble_phase", 0.0),
                    "dur": dur, "trial": t, "gap_s": args.gap,
                    "idle_s": args.idle,
                    "message": WS.make_message(args.frames, args.chunk_size),
                    "chunk": args.chunk_size,
                    "enc_overrides": tuple(sorted(enc.items())),
                    "dec_overrides": dec,
                    "seed": WS._seed(key, t, args.seed),
                    "_o": ov, "_i": iv, "_z": zv,
                })
    return cells, legs


# ── aggregation ──────────────────────────────────────────────────────────────

def collapse(results, args):
    """[(outer, inner) -> {metric: mean}] over trials, plus per-cell detail."""
    import collections
    by = collections.defaultdict(list)
    for r in results:
        if not r.get("ok"):
            continue
        s = r["spec"]
        by[(s["_o"], s["_i"], s.get("_z"), s["leg"])].append(r)
    grid = {}
    for (ov, iv, zv, leg), rs in by.items():
        cell = grid.setdefault((ov, iv, zv), {"trials": 0})
        cell["trials"] = max(cell["trials"], len(rs))
        for m, (mleg, _, _) in METRICS.items():
            if mleg != leg or m not in args.metric:
                continue
            vals = [float(r[m]) for r in rs if r.get(m) is not None]
            if vals:
                cell[m] = sum(vals) / len(vals)
                cell[m + "_min"] = min(vals)
                cell[m + "_max"] = max(vals)
    return grid


# ── SVG ──────────────────────────────────────────────────────────────────────
#
# Hand-built rather than matplotlib, which is not installed here anyway.  The
# point is not the drawing: it is that every number on the chart exists as a
# <text> node and every series carries its parameter value as an attribute, so
# the file answers "where does this peak" to a reader that cannot see it.

def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def svg_curves(grid, outer, inner, metric, args):
    oname, ovals = outer
    iname, ivals = inner
    unit = METRICS[metric][2]
    scale = 100.0 if unit == "%" else 1.0
    W, H = 900, 470
    L, R, T, B = 78, 210, 46, 58
    pw, ph = W - L - R, H - T - B

    # Points come from the GRID KEYS, not from the requested axis values: the
    # diag/random/refine samplers place points between the grid lines, and
    # indexing into `ivals` would silently drop every one of them.
    present = sorted({k[0] for k in grid})
    ilo = min([k[1] for k in grid] + list(ivals))
    ihi = max([k[1] for k in grid] + list(ivals))
    ispan = (ihi - ilo) or 1.0
    pts = [(ov, iv, cell.get(metric)) for (ov, iv), cell in grid.items()]
    ys = [p[2] * scale for p in pts if p[2] is not None]
    ymax = max(ys) if ys else 1.0
    ymin = min(ys) if ys else 0.0
    if unit == "%":
        ymin, ymax = 0.0, 100.0
    else:
        pad = max(1e-9, (ymax - ymin) * 0.15)
        ymin, ymax = max(0.0, ymin - pad), ymax + pad
    span = (ymax - ymin) or 1.0

    def X(iv):
        return L + pw * (iv - ilo) / ispan

    def Y(v):
        return T + ph - ph * (v - ymin) / span

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" '
           f'data-metric="{_esc(metric)}" data-outer="{_esc(oname)}" '
           f'data-inner="{_esc(iname)}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           f'<text x="{L}" y="24" font-size="15" font-weight="600">'
           f'{_esc(METRICS[metric][1])} — {_esc(iname)} swept per '
           f'{_esc(oname)}</text>',
           f'<text x="{L}" y="40" font-size="11" fill="#666">'
           f'candidate={_esc(args.candidate)} impair={_esc(args.impair)}'
           f'@{args.level:g} trials={args.trials}</text>']

    # grid + y axis
    for k in range(6):
        v = ymin + span * k / 5
        y = Y(v)
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" '
                   f'stroke="#e8e8e8"/>')
        out.append(f'<text x="{L-8}" y="{y+4:.1f}" font-size="11" fill="#555" '
                   f'text-anchor="end">{v:.4g}{_esc(unit)}</text>')
    # x axis: ticks on the requested values, but the scale is continuous
    for iv in ivals:
        x = X(iv)
        out.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}" '
                   f'stroke="#f2f2f2"/>')
        out.append(f'<text x="{x:.1f}" y="{T+ph+18}" font-size="11" '
                   f'fill="#555" text-anchor="middle">{iv:g}</text>')
    out.append(f'<text x="{L+pw/2:.0f}" y="{H-16}" font-size="12" '
               f'text-anchor="middle">{_esc(iname)}</text>')

    hues = [(210, 70, 45), (14, 78, 48), (150, 55, 38), (280, 55, 50),
            (40, 80, 45), (0, 0, 45), (190, 60, 40), (330, 60, 48)]
    for oi, ov in enumerate(present):
        h, s_, l_ = hues[oi % len(hues)]
        col = f"hsl({h},{s_}%,{l_}%)"
        seg, d = [], []
        for iv in sorted(k[1] for k in grid if k[0] == ov):
            v = grid.get((ov, iv), {}).get(metric)
            if v is None:
                if d:
                    seg.append(d); d = []
                continue
            d.append((X(iv), Y(v * scale), v * scale, iv))
        if d:
            seg.append(d)
        for run in seg:
            path = " ".join(f"{'M' if k == 0 else 'L'}{x:.1f},{y:.1f}"
                            for k, (x, y, _, _) in enumerate(run))
            out.append(f'<path d="{path}" fill="none" stroke="{col}" '
                       f'stroke-width="2" data-{_esc(oname)}="{ov:g}"/>')
            for x, y, val, iv in run:
                # Each point carries its two coordinates and its value as
                # attributes: this is the machine-readable half of the picture.
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" '
                           f'fill="{col}" data-{_esc(iname)}="{iv:g}" '
                           f'data-value="{val:.4g}"><title>{_esc(oname)}={ov:g} '
                           f'{_esc(iname)}={iv:g} → {val:.4g}{_esc(unit)}'
                           f'</title></circle>')
        ly = T + 14 + oi * 17
        out.append(f'<line x1="{L+pw+16}" y1="{ly-4}" x2="{L+pw+38}" '
                   f'y2="{ly-4}" stroke="{col}" stroke-width="2"/>')
        _own = [(grid[k].get(metric), k[1]) for k in grid if k[0] == ov]
        peak = max(_own or [(None, 0)],
                   key=lambda t: (-1e18 if t[0] is None else t[0]))
        lbl = (f"{oname}={ov:g}" if peak[0] is None else
               f"{oname}={ov:g}  peak {peak[0]*scale:.4g}{unit}@{peak[1]:g}")
        out.append(f'<text x="{L+pw+44}" y="{ly}" font-size="11" '
                   f'fill="#333">{_esc(lbl)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg_ridge(grid, outer, inner, metric, args):
    """Metric against the OUTER parameter — the rise-and-fall curve.

    The per-inner curves show the surface; this is its profile along the axis
    the operator is actually choosing a value on.  Best and mean over the inner
    axis are both drawn because they answer different questions: best is "this
    setting can work", mean is "this setting usually works", and a parameter
    whose best is flat while its mean collapses is one to stay away from.
    """
    oname, ovals = outer
    iname, ivals = inner
    unit = METRICS[metric][2]
    scale = 100.0 if unit == "%" else 1.0
    W, H = 900, 330
    L, R, T, B = 78, 150, 46, 58
    pw, ph = W - L - R, H - T - B

    ovals = sorted({k[0] for k in grid}) or list(ovals)
    series = {"best": [], "mean": []}
    for ov in ovals:
        vals = [grid[k].get(metric) for k in grid if k[0] == ov]
        vals = [v for v in vals if v is not None]
        series["best"].append(max(vals) * scale if vals else None)
        series["mean"].append(sum(vals) / len(vals) * scale if vals else None)
    allv = [v for s in series.values() for v in s if v is not None]
    ymax = 100.0 if unit == "%" else (max(allv) * 1.15 if allv else 1.0)
    ymin = 0.0
    span = (ymax - ymin) or 1.0

    def X(k):
        return L + (pw * k / max(1, len(ovals) - 1))

    def Y(v):
        return T + ph - ph * (v - ymin) / span

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" '
           f'data-metric="{_esc(metric)}" data-axis="{_esc(oname)}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           f'<text x="{L}" y="24" font-size="15" font-weight="600">'
           f'{_esc(METRICS[metric][1])} vs {_esc(oname)} '
           f'(over {_esc(iname)})</text>']
    for k in range(6):
        v = ymin + span * k / 5
        y = Y(v)
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" '
                   f'stroke="#e8e8e8"/>')
        out.append(f'<text x="{L-8}" y="{y+4:.1f}" font-size="11" fill="#555" '
                   f'text-anchor="end">{v:.4g}{_esc(unit)}</text>')
    for k, ov in enumerate(ovals):
        out.append(f'<text x="{X(k):.1f}" y="{T+ph+18}" font-size="11" '
                   f'fill="#555" text-anchor="middle">{ov:g}</text>')
    out.append(f'<text x="{L+pw/2:.0f}" y="{H-16}" font-size="12" '
               f'text-anchor="middle">{_esc(oname)}</text>')
    for name, col in (("best", "hsl(210,70%,45%)"), ("mean", "hsl(14,78%,48%)")):
        pts = [(X(k), Y(v), v, ovals[k])
               for k, v in enumerate(series[name]) if v is not None]
        if not pts:
            continue
        path = " ".join(f"{'M' if k == 0 else 'L'}{x:.1f},{y:.1f}"
                        for k, (x, y, _, _) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{col}" '
                   f'stroke-width="2" data-series="{name}"/>')
        for x, y, v, ov in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{col}" '
                       f'data-{_esc(oname)}="{ov:g}" data-value="{v:.4g}">'
                       f'<title>{name} {_esc(oname)}={ov:g} → {v:.4g}{_esc(unit)}'
                       f'</title></circle>')
        pk = max(pts, key=lambda t: t[2])
        out.append(f'<text x="{L+pw+12}" y="{Y(pts[-1][2])+4:.1f}" '
                   f'font-size="11" fill="{col}">{name} '
                   f'(peak {pk[2]:.4g}@{pk[3]:g})</text>')
    out.append("</svg>")
    return "\n".join(out)


def find_edges(grid, metric, args):
    """Where the metric crosses the midpoint of its range, and how tightly.

    On a cliff the number worth reporting is not the peak -- both plateaus are
    obvious -- it is the LOCATION of the transition and how well it is pinned.
    For each outer value this brackets the crossing between adjacent sampled
    inner values and reports the bracket width, which is exactly what an extra
    refine round shrinks.  A row with no crossing is flat and says so.
    """
    vals = [c.get(metric) for c in grid.values() if c.get(metric) is not None]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo <= 1e-9:
        return []
    mid = (lo + hi) / 2.0
    byo = {}
    for (ov, iv), cell in grid.items():
        if cell.get(metric) is not None:
            byo.setdefault(ov, []).append((iv, cell[metric]))
    out = []
    for ov in sorted(byo):
        pts = sorted(byo[ov])
        cross = None
        for (a, va), (b, vb) in zip(pts, pts[1:]):
            if (va - mid) * (vb - mid) <= 0 and va != vb:
                # linear interpolation inside the bracket; the bracket WIDTH is
                # reported alongside because the interpolation is only as good
                # as the spacing that produced it.
                t = (mid - va) / (vb - va) if vb != va else 0.5
                cross = {"at": round(a + t * (b - a), 6),
                         "bracket": [a, b], "width": round(b - a, 6)}
                break
        out.append({"outer": ov, "crossing": cross,
                    "flat": cross is None,
                    "range": [round(min(v for _, v in pts), 6),
                              round(max(v for _, v in pts), 6)]})
    return {"threshold": round(mid, 6), "rows": out}


def summarise(grid, outer, inner, metric):
    """The shape, in words -- what a reader should take away without the chart."""
    oname, ovals = outer
    iname, ivals = inner
    unit = METRICS[metric][2]
    scale = 100.0 if unit == "%" else 1.0
    rows = []
    for ov in sorted({k[0] for k in grid}):
        vals = [(grid[k].get(metric), k[1]) for k in grid if k[0] == ov]
        vals = [(v, iv) for v, iv in vals if v is not None]
        if not vals:
            continue
        best, at = max(vals, key=lambda t: t[0])
        mean = sum(v for v, _ in vals) / len(vals)
        rows.append({oname: ov, "best": best * scale, "best_at": at,
                     "mean": mean * scale,
                     "worst": min(v for v, _ in vals) * scale})
    if not rows:
        return {"rows": [], "peak": None, "note": "no cells scored"}
    peak = max(rows, key=lambda r: r["mean"])
    return {"rows": rows, "peak": peak,
            "note": (f"mean {metric} peaks at {oname}={peak[oname]:g} "
                     f"({peak['mean']:.4g}{unit}); "
                     f"falls to {min(r['mean'] for r in rows):.4g}{unit} "
                     f"at the worst {oname} swept")}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outer", help="name=start:stop:step (or name=a,b,c)")
    ap.add_argument("--inner", help="name=start:stop:step (or name=a,b,c)")
    ap.add_argument("--metric", action="append", choices=sorted(METRICS),
                    help="metric to draw (repeatable; default deliver)")
    ap.add_argument("--impair", default="pink",
                    choices=("pink", "click", "drift", "wobble", "reverb", "none"),
                    help="impairment family held constant unless swept")
    ap.add_argument("--level", type=float, default=12.0,
                    help="impairment level when not the swept axis")
    ap.add_argument("--candidate", default="per_frame_full",
                    help=f"run-in candidate ({', '.join(WS.CANDIDATES)})")
    ap.add_argument("--dur", type=float, default=0.5,
                    help="symbol_dur when not the swept axis")
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--gap", type=float, default=0.0,
                    help="inter-frame gap (0 = contiguous, as a node sends)")
    ap.add_argument("--idle", type=float, default=600.0,
                    help="idle window, only used by the false_per_hr metric")
    ap.add_argument("--third", help="optional 3rd axis, name=start:stop:step. "
                                    "Rendered as one plane per value -- reach "
                                    "for it when a 2-axis result is suspected "
                                    "of depending on something held constant")
    ap.add_argument("--pattern", default="grid",
                    choices=("grid", "diag", "random", "refine"),
                    help="where to place samples: grid = full product; "
                         "diag = both diagonals, offset half a step per pass, "
                         "so points land BETWEEN the grid lines; random = "
                         "Latin-hypercube over the continuous box, which no "
                         "structure in the system can align with; refine = "
                         "coarse grid then extra rounds at the steepest edges")
    ap.add_argument("--rounds", type=int, default=2,
                    help="refine: adaptive rounds after the coarse grid")
    ap.add_argument("--refine-k", type=int, default=6,
                    help="refine: midpoints added per round")
    ap.add_argument("--diag-passes", type=int, default=2,
                    help="diag: passes, each offset a further 1/passes step")
    ap.add_argument("--samples", type=int, default=0,
                    help="random: point count (default: as many as the grid)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel decodes (default: cores, capped at 16)")
    ap.add_argument("--label", help="short name for this batch, used in run_id")
    ap.add_argument("--out", help="write the run directory here "
                                  "(default: results/<id>_cssmatrix)")
    ap.add_argument("--list-params", action="store_true",
                    help="print the sweepable parameters and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid and cost, run nothing")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_params:
        print("sweepable parameters (name -> where it is applied):\n")
        for n, (kind, cast, doc) in sorted(PARAMS.items(),
                                           key=lambda kv: (kv[1][0], kv[0])):
            print(f"  {n:<24} {kind:<7} {cast.__name__:<6} {doc}")
        print("\nmetrics:")
        for m, (leg, doc, unit) in sorted(METRICS.items()):
            print(f"  {m:<24} leg={leg:<5} {doc} ({unit})")
        return 0

    if not args.outer or not args.inner:
        ap.error("--outer and --inner are both required "
                 "(use --list-params to see what can be swept)")
    args.metric = args.metric or ["deliver"]
    try:
        outer = parse_axis(args.outer)
        inner = parse_axis(args.inner)
    except ValueError as e:
        ap.error(str(e))
    # An impairment alias on either axis implies its family, so
    # `--inner wobble=400:2000:400` does not silently sweep a pink level.
    for nm, _ in (outer, inner):
        if nm in IMPAIR_ALIAS:
            args.impair = IMPAIR_ALIAS[nm]
    third = None
    if args.third:
        try:
            third = parse_axis(args.third)
        except ValueError as e:
            ap.error(str(e))
        if third[0] in IMPAIR_ALIAS:
            args.impair = IMPAIR_ALIAS[third[0]]
    names = [outer[0], inner[0]] + ([third[0]] if third else [])
    if len(set(names)) != len(names):
        ap.error("--outer, --inner and --third must sweep different parameters")

    if args.pattern == "grid":
        points = sample_grid(outer[1], inner[1])
    elif args.pattern == "diag":
        points = sample_diag(outer[1], inner[1], passes=args.diag_passes)
    elif args.pattern == "random":
        points = sample_random(outer[1], inner[1],
                               args.samples or len(outer[1]) * len(inner[1]),
                               args.seed)
    else:                                          # refine: coarse grid first
        points = sample_grid(outer[1], inner[1])
    zvals = third[1] if third else None
    points = with_third(points, zvals)

    legs = sorted({METRICS[m][0] for m in args.metric})
    per_point = args.trials * len(legs)
    jobs = args.jobs or min(16, len(os.sched_getaffinity(0)))
    jobs = max(1, min(jobs, max(1, len(points) * per_point)))

    print(f"CSS MATRIX -- {outer[0]} x {inner[0]}"
          + (f" x {third[0]}" if third else "") + f"  [{args.pattern}]")
    print(f"  outer  {outer[0]:<22} {[f'{v:g}' for v in outer[1]]}")
    print(f"  inner  {inner[0]:<22} {[f'{v:g}' for v in inner[1]]}")
    if third:
        print(f"  third  {third[0]:<22} {[f'{v:g}' for v in third[1]]}")
    print(f"  metric {', '.join(args.metric)} (legs: {', '.join(legs)})")
    est = len(points) * per_point
    if args.pattern == "refine":
        est += args.rounds * args.refine_k * per_point
    print(f"  points {len(points)} x {args.trials} trials x {len(legs)} leg(s)"
          + (f" + {args.rounds} refine round(s) x {args.refine_k}"
             if args.pattern == "refine" else "")
          + f" = ~{est} decodes on {jobs} workers")
    if args.dry_run:
        print("\n  first points: "
              + ", ".join("(" + ",".join(f"{c:g}" for c in p if c is not None)
                            + ")" for p in points[:8])
              + (" ..." if len(points) > 8 else ""))
        print("\n(dry run -- nothing executed)")
        return 0

    import concurrent.futures as cf
    t0 = time.time()
    results = []
    seen = set(points)
    rounds_log = []

    def run_points(pts, tag):
        cells, _ = build_cells(args, outer, inner, pts, third)
        got = []
        with cf.ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(WS.run_variant, c): c for c in cells}
            done = 0
            for fut in cf.as_completed(futs, timeout=1800 * max(1, len(cells))):
                try:
                    r = fut.result()
                except Exception as e:             # noqa: BLE001
                    r = {"spec": futs[fut], "ok": False,
                         "error": f"{type(e).__name__}: {e}"}
                got.append(r)
                done += 1
                if not r.get("ok"):
                    s = r["spec"]
                    print(f"  ERR  {outer[0]}={s['_o']:g} "
                          f"{inner[0]}={s['_i']:g} {r.get('error')}",
                          flush=True)
                elif args.verbose:
                    s = r["spec"]
                    print(f"  {outer[0]}={s['_o']:<8g} "
                          f"{inner[0]}={s['_i']:<8g} t{s['trial']} "
                          + " ".join(f"{m}={r[m]}" for m in args.metric
                                     if m in r), flush=True)
                elif done % 20 == 0:
                    print(f"  ... {tag} {done}/{len(cells)} "
                          f"({time.time()-t0:.0f}s)", flush=True)
        return got

    results += run_points(points, "round0")
    rounds_log.append({"round": 0, "points": len(points),
                       "kind": args.pattern})

    # Adaptive rounds: each one puts its samples at the midpoints of the
    # steepest adjacent pairs found so far, so the budget follows the edge
    # instead of re-confirming the plateaus either side of it.
    if args.pattern == "refine":
        for rnd in range(1, args.rounds + 1):
            grid_so_far = collapse(results, args)
            new_pts = []
            for zv in (zvals or [None]):
                new_pts += refine_points_plane(
                    slice_z(grid_so_far, zv), args.metric[0],
                    args.refine_k, seen, zv)
            if not new_pts:
                print(f"  round {rnd}: surface is flat in "
                      f"{args.metric[0]} -- nothing left to refine")
                break
            print(f"  round {rnd}: {len(new_pts)} midpoint(s) at the "
                  f"steepest edges", flush=True)
            seen.update(new_pts)
            results += run_points(new_pts, f"round{rnd}")
            rounds_log.append({"round": rnd, "points": len(new_pts),
                               "kind": "refine"})

    grid = collapse(results, args)
    errs = [r for r in results if not r.get("ok")]
    planes = sorted({k[2] for k in grid},
                    key=lambda z: (-1e18 if z is None else z))

    def _zk(zv):
        return "all" if zv is None else f"{third[0]}={zv:g}"

    # ── write the run directory ──────────────────────────────────────────────
    label = (args.label or f"{outer[0]}_x_{inner[0]}")[:48]
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_cssmatrix"
    out_dir = args.out or os.path.join(REPO, "results", run_id)
    os.makedirs(out_dir, exist_ok=True)

    doc = {
        "run_id": run_id, "label": label,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "axes": {"outer": {"param": outer[0], "values": outer[1],
                           "role": PARAMS[outer[0]][0]},
                 "inner": {"param": inner[0], "values": inner[1],
                           "role": PARAMS[inner[0]][0]}},
        "fixed": {"candidate": args.candidate, "impair": args.impair,
                  "level": args.level, "dur": args.dur,
                  "chunk_size": args.chunk_size, "frames": args.frames,
                  "gap_s": args.gap, "idle_s": args.idle,
                  "trials": args.trials, "seed": args.seed},
        "metrics": args.metric,
        "sampling": {"pattern": args.pattern, "rounds": rounds_log,
                     "points": len(grid)},
        "summary": {m: {_zk(zv): summarise(slice_z(grid, zv), outer, inner, m)
                        for zv in planes} for m in args.metric},
        "edges": {m: {_zk(zv): find_edges(slice_z(grid, zv), m, args)
                      for zv in planes} for m in args.metric},
        "cells": [{"outer": ov, "inner": iv,
                   **({third[0]: zv} if third else {}), **cell}
                  for (ov, iv, zv), cell in sorted(
                      grid.items(), key=lambda kv: tuple(
                          -1e18 if c is None else c for c in kv[0]))],
        "errors": [{"outer": r["spec"]["_o"], "inner": r["spec"]["_i"],
                    "third": r["spec"].get("_z"),
                    "error": r.get("error")} for r in errs],
        "decodes": len(results), "errored": len(errs),
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "matrix.json"), "w") as fh:
        json.dump(doc, fh, indent=1)

    svgs = []
    for m in args.metric:
        for zv in planes:
            sub = slice_z(grid, zv)
            tag = "" if zv is None else f"_{third[0]}{zv:g}".replace(".", "p")
            for kind, fn in (("matrix", svg_curves), ("ridge", svg_ridge)):
                name = f"{kind}_{m}{tag}.svg"
                with open(os.path.join(out_dir, name), "w") as fh:
                    fh.write(fn(sub, outer, inner, m, args))
                svgs.append(name)

    # Manifest in the same shape as a hardware run, so the Reports tab lists
    # this batch alongside them without needing to know what a matrix is.
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump({
            "run_id": run_id, "test": "css_matrix",
            "started_at": doc["generated"],
            "params": {"label": label,
                       "outer": f"{outer[0]}={outer[1]}",
                       "inner": f"{inner[0]}={inner[1]}",
                       "metrics": args.metric, **doc["fixed"]},
            "roles_map": {},
        }, fh, indent=1)

    # ── the shape, printed ───────────────────────────────────────────────────
    for m in args.metric:
      for zv in planes:
        sub = slice_z(grid, zv)
        is_product = set(sub) == set(sample_grid(outer[1], inner[1]))
        unit = METRICS[m][2]
        scale = 100.0 if unit == "%" else 1.0
        print(f"\n{METRICS[m][1]} ({unit})"
              + ("" if zv is None else f"   [{third[0]}={zv:g}]"))
        if is_product:
            print("  " + f"{outer[0]:<20}" +
                  "".join(f"{v:>9g}" for v in inner[1]) + "     mean")
            for ov in outer[1]:
                row = [f"  {ov:<20g}"]
                vals = []
                for iv in inner[1]:
                    v = sub.get((ov, iv), {}).get(m)
                    if v is None:
                        row.append(f"{'-':>9}")
                    else:
                        vals.append(v * scale)
                        row.append(f"{v*scale:>9.4g}")
                row.append(f"{sum(vals)/len(vals):>9.4g}"
                           if vals else f"{'-':>9}")
                print("".join(row))
        else:
            # Off-lattice sampling has no rectangular table to print; list the
            # points by outer value instead, which is what the SVG draws.
            print(f"  ({len(sub)} points, not a rectangular grid -- "
                  f"pattern={args.pattern})")
            print("  " + f"{outer[0]:<12}" + f"{inner[0]:>12}" + f"{'value':>12}")
            for (ov, iv) in sorted(sub):
                v = sub[(ov, iv)].get(m)
                if v is not None:
                    print(f"  {ov:<12g}{iv:>12g}{v*scale:>12.4g}")
        print("  " + doc["summary"][m][_zk(zv)]["note"])

        # The edge, which is the thing a cliff-shaped surface is hiding.
        ed = doc["edges"][m][_zk(zv)]
        if ed and ed.get("rows"):
            print(f"  edge ({METRICS[m][1]} crossing "
                  f"{ed['threshold']*scale:.4g}{unit}):")
            for row in ed["rows"]:
                if row["flat"]:
                    print(f"    {outer[0]}={row['outer']:<10g} flat "
                          f"({row['range'][0]*scale:.4g}"
                          f"-{row['range'][1]*scale:.4g}{unit})")
                else:
                    c = row["crossing"]
                    print(f"    {outer[0]}={row['outer']:<10g} "
                          f"{inner[0]} ~= {c['at']:<10g} "
                          f"(bracketed {c['bracket'][0]:g}..{c['bracket'][1]:g},"
                          f" width {c['width']:g})")

    print(f"\n{len(results)-len(errs)} decoded, {len(errs)} errored, "
          f"{doc['wall_s']:.0f}s wall")
    print(f"wrote {out_dir}/")
    print(f"  matrix.json          every cell (machine-readable)")
    for name in svgs:
        print(f"  {name}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
