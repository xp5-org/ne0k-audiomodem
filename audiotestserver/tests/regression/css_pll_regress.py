# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""css_pll_regress.py — C streaming PLL decoder (libcsspll) vs css_decoder.decode().

Every case is encoded with css_encoder, put through a channel (noise, clock
drift, impulses, coloured noise / hum, gain), and decoded four ways, each in its
own subprocess because SIG is fixed at import time:

  py64  css_decoder.decode()        CSS_SIG_DTYPE=float64
  c64   libcsspll64 via css_pll_c    (grading build: must match py64)
  py32  css_decoder.decode()        CSS_SIG_DTYPE=float32 (the shipping Python)
  c32   libcsspll via css_pll_c      (the shipping C)

Both Python runs use CSS_ENC_BACKEND=c so the frame pipeline is the same
css_dec.c object code the C port links; RS uses reedsolo (the C port carries a
reedsolo-exact erasure decoder, rs255_decode_era).

Grading:
  c64 vs py64  per-symbol trace: integers exact (nr, found_peak, bit,
               timing_err, peak_est, n_read, acq/settled/sync flags, gates),
               floats to 1e-6 relative; sync/resync/frame events and decoded
               text identical.  A mismatch here is a PORT BUG.
  c32 vs py32  decoded text identical; the first diverging symbol is reported
               as a precision note, not a failure.

Usage:
  python tests/regression/css_pll_regress.py [-v] [--only SUBSTR] [--list]
         [--keep] [--no32]
The interpreter must have reedsolo (RS cases) and scipy-free numpy.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import types
import wave

import numpy as np

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(tempfile.gettempdir(), "css_pll_regress_work")
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(TESTS, "CSS"))

MSG = "PLL port parity: hello mesh 0123456789"

INT_KEYS = ("nr", "found_peak", "bit", "timing_err", "peak_est", "n_read",
            "acq_left", "settled", "sync_found", "outlier", "wpk_strong",
            "signal_present", "abs_start")
FLOAT_KEYS = ("conf", "snr_db", "winner_pk", "pll_conf", "pll_r2", "rate_est",
              "timing_frac", "final_conf", "agc_gain")
EVENT_KEYS = ("SYNC_ACQUIRED", "RESYNC", "FRAME", "RETRO_SYNC", "RETRO_GATE")


# ──────────────────────────────────────────────────────────────────────────────
# cases
# ──────────────────────────────────────────────────────────────────────────────

def _cases():
    base = dict(fs=48000, f_min=800, f_max=4000, amplitude=0.5,
                symbol_gap_frac=0.05, diff_encode=True)
    C = []

    def add(name, enc, chan=None, dec=None, msg=MSG):
        e = dict(base); e.update(enc)
        C.append(dict(name=name, enc=e, chan=chan or {}, dec=dec or {}, msg=msg))

    add("pll_025_clean",      dict(symbol_dur=0.25, chunk_size=2))
    add("pll_025_noise12",    dict(symbol_dur=0.25, chunk_size=2), dict(snr_db=12, seed=1))
    add("pll_020_drift_p300", dict(symbol_dur=0.20, chunk_size=2), dict(ppm=300, snr_db=18, seed=2))
    add("pll_020_drift_m300", dict(symbol_dur=0.20, chunk_size=2), dict(ppm=-300, snr_db=18, seed=3))
    add("pll_015_mesh",       dict(symbol_dur=0.15, f_min=400, f_max=3500, chunk_size=8,
                                   amplitude=0.6), dict(snr_db=15, seed=4))
    add("pll_025_impulses",   dict(symbol_dur=0.25, chunk_size=2), dict(snr_db=20, impulses=25, seed=5))
    # 2-byte frame check (crc_bytes=2, CRC-16): the GUI mesh default since 2026-11-08
    add("pll_015_mesh_crc16", dict(symbol_dur=0.15, f_min=400, f_max=3500, chunk_size=8,
                                   amplitude=0.6, crc_bytes=2), dict(snr_db=15, seed=21))
    add("pll_015_short_crc16", dict(symbol_dur=0.15, f_min=400, f_max=3500, chunk_size=2,
                                    amplitude=0.6, crc_bytes=2), dict(snr_db=12, impulses=25, seed=22))
    add("pll_025_hum",        dict(symbol_dur=0.25, chunk_size=2), dict(snr_db=14, hum=0.3, colored=True, seed=6))
    add("pll_025_rs2",        dict(symbol_dur=0.25, chunk_size=4, rs_t=2), dict(snr_db=6, seed=7))
    add("pll_025_k7_il4",     dict(symbol_dur=0.25, chunk_size=2, conv_code="k7r12",
                                   interleave_rows=4), dict(snr_db=8, seed=8))
    add("pll_025_perframe",   dict(symbol_dur=0.25, chunk_size=2, preamble_mode="per_frame",
                                   inter_frame_silence_s=1.0), dict(snr_db=15, seed=9))
    add("pll_05_forced",      dict(symbol_dur=0.5, chunk_size=4), dict(snr_db=10, ppm=150, seed=10),
        dec=dict(mf_longsym=False), msg="forced PLL 0.5s")
    add("pll_10_forced",      dict(symbol_dur=1.0, chunk_size=4), dict(snr_db=10, seed=11),
        dec=dict(mf_longsym=False), msg="1.0s")
    add("pll_020_lowsnr",     dict(symbol_dur=0.20, chunk_size=2), dict(snr_db=3, ppm=-120, seed=12))
    add("pll_025_gain_step",  dict(symbol_dur=0.25, chunk_size=2), dict(snr_db=16, gain_step=0.2, seed=13))
    # mesh conference (2026-11-08): G.711 leg, onset half a symbol off the read
    # grid (parity of the pre-sync full-period search + onset guard; synthetic
    # chirps decode either way, the captured blind-spot frame is in ne0k-gui
    # tests/test_css_stream.py), and a jitter-buffer insert / drop mid-frame
    # (phase-step recovery, settled and in acquisition: 4/5 frames without it)
    mesh = dict(symbol_dur=0.15, f_min=400, f_max=3500, chunk_size=8, amplitude=0.6)
    add("pll_015_g711_onset", mesh, dict(g711=True, lead_s=1.3 + 0.0787, snr_db=30, seed=14))
    add("pll_015_g711_slip",  mesh, dict(g711=True, slips=[(9.0, 1026)], snr_db=30, seed=15))
    add("pll_015_g711_drop",  mesh, dict(g711=True, slips=[(7.6, -1026)], snr_db=24, seed=16))
    # retro anchor slot count (2026-11-08): the anchor grid dropped the frame's last symbol, so the aligned frame
    # was never a candidate and a frame-1 chunk starting 0x69-0x6B CRC-16 false-locked at shift -25 (seq 255,
    # wrong data).  "j" (0x6A) opens frame 1; the broadcast port's TIME payload hit it every second.
    add("pll_015_anchor_frame1_j", dict(symbol_dur=0.15, f_min=400, f_max=3500, chunk_size=2, amplitude=0.6,
                                        crc_bytes=2), dict(snr_db=30, seed=23), msg="ABjkLo")
    # room-relative onset guard (2026-11-17): 8 s of room noise far over silence_threshold before the burst, so the
    # absolute guard never anchors and the floor tracker + sustained relative onset do
    add("pll_015_room_lead", dict(symbol_dur=0.15, f_min=400, f_max=6000, chunk_size=8, amplitude=0.2,
                                  crc_bytes=2), dict(lead_s=8.0, snr_db=12, seed=24))
    return C


def _g711(x, fs):
    """fs -> 8 kHz mu-law (G.711) -> fs, numpy only: the SIP conference leg."""
    k = fs // 8000
    t = np.arange(191) - 95
    h = np.sinc(2 * 3600.0 / fs * t) * np.hamming(191)
    h /= h.sum()
    y = np.convolve(x, h, mode="same")[::k]
    mu = 255.0
    c = np.round(np.sign(y) * np.log1p(mu * np.minimum(np.abs(y), 1.0)) / np.log1p(mu) * 127) / 127
    y = np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu
    up = np.zeros(len(y) * k)
    up[::k] = y * k
    return np.convolve(up, h, mode="same")


def _channel(pcm, fs, ch):
    rng = np.random.default_rng(ch.get("seed", 0))
    x = pcm.astype(np.float64)
    lead = np.zeros(int(fs * ch.get("lead_s", 1.3))); tail = np.zeros(int(fs * 2.0))
    if ch.get("g711"):
        x = _g711(x, fs)
    for at_s, n in ch.get("slips", ()):         # (seconds into the frame, +insert/-drop)
        i = int(at_s * fs)
        x = (np.concatenate([x[:i], np.zeros(n), x[i:]]) if n > 0
             else np.concatenate([x[:i], x[i - n:]]))
    x = np.concatenate([lead, x, tail])
    ppm = ch.get("ppm", 0)
    if ppm:
        n = len(x)
        t_new = np.arange(int(n / (1 + ppm * 1e-6))) * (1 + ppm * 1e-6)
        x = np.interp(t_new, np.arange(n), x)
    if ch.get("gain_step"):
        x[len(x) // 2:] *= ch["gain_step"]
    sig_p = float(np.mean(pcm.astype(np.float64) ** 2)) or 1e-9
    if "snr_db" in ch:
        npow = sig_p / (10 ** (ch["snr_db"] / 10))
        noise = rng.normal(0, math.sqrt(npow), len(x))
        if ch.get("colored"):
            noise = np.convolve(noise, np.ones(8) / math.sqrt(8), mode="same")
        x = x + noise
    if ch.get("hum"):
        t = np.arange(len(x)) / fs
        x = x + ch["hum"] * (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t))
    for _ in range(ch.get("impulses", 0)):
        i = int(rng.integers(0, len(x) - 40))
        x[i:i + int(rng.integers(3, 30))] += rng.choice([-1, 1]) * rng.uniform(0.6, 0.95)
    return np.clip(x, -1, 1).astype(np.float32)


def _load_wav(path):
    with wave.open(path, "rb") as w:
        fs = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, fs


def prepare(case):
    import css_encoder
    os.makedirs(WORK, exist_ok=True)
    name = case["name"]
    wav = os.path.join(WORK, name + "_tx.wav")
    info = os.path.join(WORK, name + "_info.json")
    enc = dict(case["enc"], message=case["msg"], output=wav, info_out=info,
               result_out=os.path.join(WORK, name + "_enc.json"))
    with contextlib.redirect_stdout(io.StringIO()):
        css_encoder.encode(enc)
    pcm, fs = _load_wav(wav)
    x = _channel(pcm, fs, case["chan"])
    rx = os.path.join(WORK, name + "_rx.npy")
    np.save(rx, x)
    return rx, info, fs


# ──────────────────────────────────────────────────────────────────────────────
# worker: one decode in this (fresh) interpreter
# ──────────────────────────────────────────────────────────────────────────────

def worker(rx, info, fs, dec_json, out):
    import threading
    pcm = np.load(rx)
    pos = [0]
    stop = threading.Event()

    class FakeStream:
        def __init__(s, **k): pass
        def __enter__(s): return s
        def __exit__(s, *a): return False

        def read(s, n):
            c = pcm[pos[0]:pos[0] + n]
            pos[0] += len(c)
            if len(c) < n:
                stop.set()
                c = np.concatenate([c, np.zeros(n - len(c), dtype=np.float32)])
            return c.astype(np.float32), False

    fake = types.ModuleType("sounddevice")
    fake.InputStream = lambda **k: FakeStream()
    sys.modules["sounddevice"] = fake
    import css_decoder
    trace = []
    cfg = {"info": info, "result_out": out + ".result.json", "timeout_sec": 1e9,
           "fs": int(fs), "trace": trace, "pll_nr_max": 0}
    cfg.update(json.loads(dec_json))
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = css_decoder.decode(cfg, stop_event=stop)
    el = time.time() - t0
    json.dump({"result": {k: res.get(k) for k in ("status", "decoded_text", "frames_decoded",
                                                  "frames_errored", "failure_reason")},
               "frames": res.get("frames"), "trace": trace, "elapsed": el,
               "log_tail": buf.getvalue()[-4000:]}, open(out, "w"), default=float)


ENV = {
    "py64": {"CSS_SIG_DTYPE": "float64", "CSS_ENC_BACKEND": "c", "CSS_PLL_BACKEND": ""},
    "c64":  {"CSS_SIG_DTYPE": "float64", "CSS_ENC_BACKEND": "c", "CSS_PLL_BACKEND": "c64"},
    "py32": {"CSS_SIG_DTYPE": "float32", "CSS_ENC_BACKEND": "c", "CSS_PLL_BACKEND": ""},
    "c32":  {"CSS_SIG_DTYPE": "float32", "CSS_ENC_BACKEND": "c", "CSS_PLL_BACKEND": "c"},
}


def run_backend(case, rx, info, fs, be):
    out = os.path.join(WORK, f"{case['name']}_{be}.json")
    env = dict(os.environ)
    env.update(ENV[be])
    env.pop("RS_BACKEND", None)
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", rx, info, str(fs),
           json.dumps(case["dec"]), out]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{be} worker failed:\n{p.stderr[-3000:]}")
    return json.load(open(out))


# ──────────────────────────────────────────────────────────────────────────────
# comparison
# ──────────────────────────────────────────────────────────────────────────────

def _split(trace):
    syms = [t for t in trace if "event" not in t]
    evs = [(t["event"], t["sym"], t.get("hex", ""), t.get("a", None) if t["event"] in ("RESYNC", "SYNC_ACQUIRED") else None)
           for t in trace if t.get("event") in EVENT_KEYS]
    # C emits RETRO_GATE/RETRO_SYNC with different payloads; compare kind+sym only
    evs = [(e[0], e[1], e[2], e[3]) if e[0] not in ("RETRO_SYNC", "RETRO_GATE") else (e[0], e[1], "", None)
           for e in evs]
    return syms, evs


def first_divergence(a_syms, b_syms, rtol, atol):
    n = min(len(a_syms), len(b_syms))
    for i in range(n):
        a, b = a_syms[i], b_syms[i]
        for k in INT_KEYS:
            if k in a and k in b and int(a[k]) != int(b[k]):
                return i, k, a[k], b[k]
        for k in FLOAT_KEYS:
            if k in a and k in b:
                x, y = float(a[k]), float(b[k])
                # snr_db above ~150 dB is the ratio of a peak to FFT round-off
                # (digital silence): ill-conditioned by construction.
                if k == "snr_db" and x > 150 and y > 150:
                    continue
                if not (abs(x - y) <= atol + rtol * max(abs(x), abs(y))):
                    return i, k, x, y
    if len(a_syms) != len(b_syms):
        return n, "length", len(a_syms), len(b_syms)
    return None


def grade(case, runs, verbose):
    lines, ok = [], True
    ref, got = runs["py64"], runs["c64"]
    rs, rev = _split(ref["trace"]); gs, gev = _split(got["trace"])
    div = first_divergence(rs, gs, 1e-6, 1e-9)
    t_ok = ref["result"]["decoded_text"] == got["result"]["decoded_text"]
    e_ok = rev == gev
    if div or not t_ok or not e_ok:
        ok = False
    lines.append(f"  c64 vs py64: symbols={len(gs)}/{len(rs)} "
                 f"{'trace OK' if not div else 'DIVERGE sym#%d %s py=%r c=%r' % (div[0] + 1, div[1], div[2], div[3])} "
                 f"events={'OK' if e_ok else 'DIFF'} text={'OK' if t_ok else 'DIFF'} "
                 f"[{ref['result']['status']}/{got['result']['status']}] "
                 f"t_py={ref['elapsed']:.1f}s t_c={got['elapsed']:.2f}s")
    if not e_ok and verbose:
        for i, (x, y) in enumerate(zip(rev, gev)):
            if x != y:
                lines.append(f"    first event diff #{i}: py={x} c={y}")
                break
        lines.append(f"    py events={rev[:12]}")
        lines.append(f"    c  events={gev[:12]}")
    if not t_ok:
        lines.append(f"    py text={ref['result']['decoded_text']!r}")
        lines.append(f"    c  text={got['result']['decoded_text']!r}")
    if "py32" in runs:
        r32, c32 = runs["py32"], runs["c32"]
        s32, _ = _split(r32["trace"]); g32, _ = _split(c32["trace"])
        d32 = first_divergence(s32, g32, 1e-3, 1e-6)
        t32 = r32["result"]["decoded_text"] == c32["result"]["decoded_text"]
        if not t32:
            ok = False
        lines.append(f"  c32 vs py32: text={'OK' if t32 else 'DIFF'} "
                     f"[{r32['result']['status']}/{c32['result']['status']}] "
                     f"{'trace within 1e-3' if not d32 else 'precision: first divergence sym#%d %s py=%r c=%r' % (d32[0] + 1, d32[1], d32[2], d32[3])} "
                     f"t_py={r32['elapsed']:.1f}s t_c={c32['elapsed']:.2f}s")
        if not t32:
            lines.append(f"    py32 text={r32['result']['decoded_text']!r}")
            lines.append(f"    c32  text={c32['result']['decoded_text']!r}")
    return ok, lines


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:7])
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no32", action="store_true")
    args = ap.parse_args()
    cases = [c for c in _cases() if args.only in c["name"]]
    if args.list:
        for c in cases:
            print(c["name"], c["enc"].get("symbol_dur"), c["chan"], c["dec"])
        return 0
    fails = 0
    for c in cases:
        rx, info, fs = prepare(c)
        bes = ["py64", "c64"] + ([] if args.no32 else ["py32", "c32"])
        runs = {}
        try:
            for be in bes:
                runs[be] = run_backend(c, rx, info, fs, be)
            ok, lines = grade(c, runs, args.verbose)
        except Exception as e:
            ok, lines = False, [f"  ERROR {e}"]
        fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {c['name']}")
        for ln in lines:
            print(ln)
    print(f"\n{len(cases) - fails}/{len(cases)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
