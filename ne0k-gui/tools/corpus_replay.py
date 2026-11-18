#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Replay real recorded RX audio through the GUI's CSS streaming receiver (CssStreamRx, decoder pll_c) offline, in
parallel, under two or more decoder settings, and compare what each decodes.

The audiotestserver idea applied to the GUI: real recordings (laptop diag sessions, group-test per-node rx.wav,
CAPTURE grabs) are the yardstick for receiver changes, not synthetic noise.

Corpus: a directory tree; each recording is ``<name>.wav`` (mono 16-bit) with its CSS settings beside it as
``<name>.css.json`` (CssConfig fields) -- or a diag session folder (rx_*.wav + settings_*.json snapshots written at
every modem start; segments follow the snapshots, CSS segments only).

Units: each recording is cut into bursts at quiet gaps (level under the room floor + 6 dB for >= --gap s); a unit is
one burst with --pre s of the room before it (so the decoder's detector state looks like a live one) and 3 s after.
Units are independent, so they spread over a process pool (one C decoder per process).

    corpus_replay.py CORPUS --variant old='{"onset_rel_db":0}' --variant new='{}' [-j 76] [--json out.json]

Prints per-variant totals (packets decoded, frames, FRAME_ERRORs, weak syncs) and every unit where the variants
disagree.  Exit 1 if any unit decodes under the first variant but not under another (a regression).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import os
import re
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GUI = HERE.parent
sys.path.insert(0, str(GUI))

FS_DEFAULT = 48000


def _load_wav(p: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(p)) as w:
        fs, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, {2: np.int16, 4: np.int32}[sw]).astype(np.float32)
    x = x.reshape(-1, ch)[:, 0] / (32768.0 if sw == 2 else 2.0 ** 31)
    return x, fs


def _css_fields() -> set[str]:
    from modemgui.engine.modemcfg import CssConfig
    return {f.name for f in dataclasses.fields(CssConfig)}


# -- corpus discovery -----------------------------------------------------------------------------------------

def discover(root: Path) -> list[dict]:
    """Recordings with their CSS settings: [{wav, css, start, end}] (sample ranges into the wav)."""
    recs = []
    for wavp in sorted(root.rglob("*.wav")):
        side = wavp.with_suffix(".css.json")
        if side.exists():
            recs.append({"wav": str(wavp), "css": json.loads(side.read_text()), "start": 0, "end": None})
            continue
        member = wavp.parent.parent / "settings.json"       # group test member: <member>/audio/rx.wav
        if wavp.name == "rx.wav" and wavp.parent.name == "audio" and member.exists():
            try:
                d = json.loads(member.read_text())
            except Exception:
                continue
            g = d.get("general", {})
            if g.get("modem") == "css" and d.get("css"):
                css = dict(d["css"])
                css.setdefault("node_id", g.get("node_id", 1))
                recs.append({"wav": str(wavp), "css": css, "start": 0, "end": None})
            continue
        stamp = wavp.stem.split("_", 1)[-1]
        evdb = wavp.parent / f"events_{stamp}.sqlite"
        base = next((p / "base.css.json" for p in wavp.parents if (p / "base.css.json").exists()), None)
        if evdb.exists() and base is not None and not list(wavp.parent.glob("settings_*.json")):
            recs += _segments_from_eventdb(wavp, evdb, json.loads(base.read_text()))
            continue
        snaps = sorted(wavp.parent.glob("settings_*.json"))
        segs = []
        for s in snaps:
            try:
                d = json.loads(s.read_text())
            except Exception:
                continue
            if d.get("rx_wav") != wavp.name:
                continue
            st = d.get("settings", {})
            mode = str(st.get("general", {}).get("modem", ""))
            segs.append((int(d.get("rx_wav_index", 0)), mode, st.get("css", {})))
        segs.sort(key=lambda t: t[0])
        for i, (a, mode, css) in enumerate(segs):
            b = segs[i + 1][0] if i + 1 < len(segs) else None
            if "css" in mode and css:
                recs.append({"wav": str(wavp), "css": css, "start": a, "end": b})
    return recs


def _segments_from_eventdb(wavp: Path, evdb: Path, base_css: dict) -> list[dict]:
    """Sessions recorded before settings_*.json snapshots: the event db logs every line with its RX sample index;
    each 'css dsp backend: ... symbol_dur=Xs' line opens a CSS segment (to the next such line, an OFDM start or a
    power-off).  Settings the log does not carry come from a base.css.json in a parent folder."""
    import sqlite3
    try:
        db = sqlite3.connect(f"file:{evdb}?mode=ro", uri=True)
        start = int(dict(db.execute("select key, value from meta")).get("wav_rx_start", 0))
        rows = db.execute("select rx_index, line from log where line like '%css dsp backend:%' "
                          "or line like '%modem powered off%' or line like '%ofdm%started%' order by id").fetchall()
        db.close()
    except Exception:
        return []
    marks = []
    for idx, line in rows:
        m = re.search(r"symbol_dur=([\d.]+)s", line)
        sd = float(m.group(1)) if m and "css dsp backend" in line else None
        if sd is not None and marks and marks[-1][1] == sd:
            continue                    # logged once per decoder / restart with the same settings: one segment
        marks.append((int(idx) - start, sd))
    out = []
    for i, (a, sd) in enumerate(marks):
        if sd is None:
            continue
        b = next((m[0] for m in marks[i + 1:] if m[0] > a), None)
        out.append({"wav": str(wavp), "css": dict(base_css, symbol_dur=sd), "start": max(0, a), "end": b})
    return out


def units_of(rec: dict, pre_s: float, gap_s: float, min_burst_s: float) -> list[dict]:
    x, fs = _load_wav(Path(rec["wav"]))
    a, b = rec["start"], rec["end"] if rec["end"] is not None else len(x)
    x = x[a:b]
    win = int(0.05 * fs)
    n = len(x) // win
    if n < 10:
        return []
    lv = 20 * np.log10(np.sqrt(np.mean(x[:n * win].reshape(n, win) ** 2, axis=1)) + 1e-12)
    floor = np.percentile(lv, 10)
    loud = lv > floor + 6.0
    # close gaps shorter than gap_s (chirp gaps, short dropouts)
    g = int(gap_s / 0.05)
    idx = np.flatnonzero(loud)
    bursts = []
    if len(idx):
        s0 = p = idx[0]
        for i in idx[1:]:
            if i - p > g:
                bursts.append((s0, p + 1)); s0 = i
            p = i
        bursts.append((s0, p + 1))
    out = []
    for s, e in bursts:
        if (e - s) * 0.05 < min_burst_s:
            continue
        u0 = max(0, s * win - int(pre_s * fs))
        u1 = min(len(x), e * win + 3 * fs)
        out.append({"wav": rec["wav"], "css": rec["css"], "a": a + u0, "b": a + u1, "burst_s": (e - s) * 0.05,
                    "t": (a + s * win) / fs})
    return out


# -- worker -----------------------------------------------------------------------------------------------------

_W: dict = {}


def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from modemgui.engine import css_link as cl
    cl.set_backend("c")
    import css_pll_c
    css_pll_c.load(False)
    _W["cl"] = cl
    _W["cm"] = cl._imports()[1]
    _W["fields"] = _css_fields()


def decode_unit(unit: dict, variants: dict[str, dict]) -> dict:
    from modemgui.engine.modemcfg import CssConfig
    from modemgui.engine.css_stream import CssStreamRx
    from modemgui.engine.events import Decoded
    cl, cm = _W["cl"], _W["cm"]
    css = {k: v for k, v in unit["css"].items() if k in _W["fields"]}
    if float(css.get("symbol_dur", 0.15)) > 0.25:
        return {"unit": unit, "skip": "symbol_dur > 0.25 s: MF burst path, not the streaming PLL"}
    x, fs = _load_wav(Path(unit["wav"]))
    x = x[unit["a"]:unit["b"]]

    def run(cfg, dec):
        info = json.loads(Path(cl._info(cfg)).read_text())
        out, logs = [], []
        rx = CssStreamRx(cfg, info, node_id=cfg.node_id, hear_own=False, log=logs.append, emit=out.append,
                         emit_link=lambda *a: None, decoded_cls=Decoded, unpack=cm.unpack_packet, dec=dict(dec))
        for i in range(0, len(x), 1024):
            rx.feed(x[i:i + 1024], False)
        rx.feed(np.zeros(3 * fs, np.float32), False)
        rx.burst_end()
        syncs = [float(m.group(1)) for l in logs for m in [re.search(r"sync acquired score=([\d.]+)", l)] if m]
        return {"packets": [(d.type_name, d.src, d.seq, d.frames, d.text[:40]) for d in out],
                "frame_errors": sum("FRAME_ERROR" in l for l in logs),
                "syncs": syncs, "weak_syncs": sum(s < 0.9 for s in syncs)}

    names = list(variants)
    res = {}
    if "crc_bytes" in css:
        cfg = CssConfig(**css)
    else:                       # older recordings: CRC-32 then, CRC-16 now -- keep whichever the baseline decodes
        best = None
        for crc in (2, 4):
            c = CssConfig(**dict(css, crc_bytes=crc))
            r = run(c, variants[names[0]])
            if best is None or len(r["packets"]) > len(best[1]["packets"]):
                best = (c, r)
        cfg, res[names[0]] = best
    for name in names:
        if name not in res:
            res[name] = run(cfg, variants[name])
    return {"unit": {k: unit[k] for k in ("wav", "a", "b", "t", "burst_s")}, "crc_bytes": cfg.crc_bytes, "res": res}


# -- main -------------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--variant", action="append", default=[],
                    help="name=JSON decode() overrides; the first is the baseline (default old/new onset guard)")
    ap.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--pre", type=float, default=20.0, help="room before each burst, s")
    ap.add_argument("--gap", type=float, default=2.0, help="quiet that splits two bursts, s")
    ap.add_argument("--min-burst", type=float, default=3.0, help="shorter bursts are skipped, s")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    variants = {}
    for v in a.variant or ['old={"onset_rel_db": 0}', "new={}"]:
        k, _, j = v.partition("=")
        variants[k] = json.loads(j or "{}")
    recs = discover(a.corpus)
    units = [u for r in recs for u in units_of(r, a.pre, a.gap, a.min_burst)]
    print(f"{len(recs)} recordings, {len(units)} burst units, {len(variants)} variants, {a.jobs} jobs", flush=True)
    results = []
    with cf.ProcessPoolExecutor(a.jobs, initializer=_worker_init) as ex:
        futs = [ex.submit(decode_unit, u, variants) for u in units]
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"error": f"{type(e).__name__}: {e}"})
            if i % 25 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} units", flush=True)
    names = list(variants)
    tot = {n: {"packets": 0, "frames": 0, "frame_errors": 0, "weak_syncs": 0} for n in names}
    regress, diffs, errors, skipped = [], [], 0, 0
    for r in results:
        if "error" in r:
            errors += 1; continue
        if "skip" in r:
            skipped += 1; continue
        for n in names:
            v = r["res"][n]
            tot[n]["packets"] += len(v["packets"])
            tot[n]["frames"] += sum(p[3] for p in v["packets"])
            tot[n]["frame_errors"] += v["frame_errors"]
            tot[n]["weak_syncs"] += v["weak_syncs"]
        keys = {n: {tuple(p[:3]) for p in r["res"][n]["packets"]} for n in names}
        if any(keys[n] != keys[names[0]] for n in names[1:]):
            u = r["unit"]
            line = (f"{Path(u['wav']).parent.name}/{Path(u['wav']).name} t={u['t']:.1f}s burst={u['burst_s']:.0f}s: "
                    + "  ".join(f"{n}={len(r['res'][n]['packets'])}pk/{r['res'][n]['frame_errors']}fe"
                                f"/syncs{[round(s, 2) for s in r['res'][n]['syncs']][:4]}" for n in names))
            diffs.append(line)
            if any(keys[names[0]] - keys[n] for n in names[1:]):
                regress.append(line)
    print(f"\nunits: {len(results)}  errors: {errors}  skipped (long symbols): {skipped}")
    for n in names:
        print(f"  {n:>8}: " + "  ".join(f"{k}={v}" for k, v in tot[n].items()))
    print(f"\nunits that differ: {len(diffs)}")
    for l in diffs:
        print("  " + l)
    print(f"\nregressions (decoded by {names[0]} only): {len(regress)}")
    for l in regress:
        print("  " + l)
    if a.json:
        a.json.write_text(json.dumps({"variants": variants, "totals": tot, "results": results}, indent=1, default=str))
    return 1 if regress else 0


if __name__ == "__main__":
    sys.exit(main())
