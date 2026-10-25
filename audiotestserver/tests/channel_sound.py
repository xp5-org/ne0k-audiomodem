# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""
channel_sound.py — Acoustic channel sounder & QAM parameter selector
=====================================================================
Replaces the brute-force `qam_probe_*` sweep (encode + transmit many frames at
every carrier x symbol_rate combo) with the way real modems probe a channel:
send **one** wideband known signal, measure the channel once, then **compute**
the operating point.

Excitation is a Farina exponential sine sweep (ESS).  From a single recording of
`[silence | sweep | silence]` we recover:

  * the linear impulse response  h(t)          (deconvolution with the inverse sweep)
  * the channel magnitude        |H(f)|        (FFT of h)
  * the RMS delay spread         tau_rms       (reverb / multipath → caps baud)
  * the noise floor              N(f)          (from the leading silence)
  * the received SNR             SNR(f)        (sweep-segment PSD vs noise PSD)

`select_qam_params()` then argmaxes throughput over every *legal* (carrier,
symbol_rate) pair — legality/bandwidth taken from `Modulator.auto_beta` so we
never recommend a config the modem would reject — subject to an SNR margin and a
delay-spread (ISI) guard.  No per-combo transmission; adding candidates is free.

This module is standalone and hardware-free: `selftest` builds a synthetic
channel (known multipath + band-limit + noise) and asserts the analyzer recovers
it, so the whole pipeline is validatable without a mic or speaker.

CLI
---
    python channel_sound.py selftest
    python channel_sound.py gen     --out sweep_tx.wav
    python channel_sound.py analyze --rec recording.wav --sweep sweep_tx.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, firwin, welch
from scipy.io import wavfile


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
from qam_writer import Modulator   # noqa: E402  (for auto_beta legality/bandwidth)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEF_FS       = 48000
DEF_F0       = 200.0          # sweep start (Hz); laptop speakers roll off below
DEF_F1_FRAC  = 0.45          # sweep end as fraction of fs (anti-alias margin)
DEF_DUR      = 3.0           # sweep duration (s)
DEF_AMP      = 0.7           # sweep peak amplitude
DEF_SILENCE  = 0.5           # leading/trailing silence (s) — leading = noise probe

SEL_MARGIN_DB  = 12.0        # min in-band SNR to accept a config (16-QAM-ish)
SEL_EQ_SYMBOLS = 4.0         # max delay spread the demod equalizer tolerates,
                             # in symbol periods (Demodulator._eq_taps is 9 taps
                             # at symbol rate ≈ ±4 symbols).  tau_rms must stay
                             # within EQ_SYMBOLS * symbol_period → sr ≤ EQ/tau_rms.
SEL_CF_STEP    = 100.0       # carrier grid step (Hz)


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. Sweep generation (Farina ESS)
# ---------------------------------------------------------------------------

def make_ess(fs: int = DEF_FS, f0: float = DEF_F0, f1: "float | None" = None,
             dur: float = DEF_DUR, amp: float = DEF_AMP,
             fade: float = 0.01) -> "tuple[np.ndarray, np.ndarray]":
    """
    Exponential sine sweep + matched inverse filter.

    x(t)   = sin( (w0*T / R) * (exp(t*R/T) - 1) ),   R = ln(w1/w0)
    inv(t) = time-reversed x(t) divided by the instantaneous envelope k(t),
             so that fftconvolve(x, inv) ≈ a band-limited delta.  The reversal
             is what pushes harmonic-distortion products to *negative* time in
             the deconvolution, keeping the linear IR clean.

    Returns (sweep, inverse), both float32 in ~[-1, 1] (sweep scaled by `amp`).
    """
    if f1 is None:
        f1 = DEF_F1_FRAC * fs
    n  = int(round(dur * fs))
    t  = np.arange(n) / fs
    w0 = 2.0 * np.pi * f0
    w1 = 2.0 * np.pi * f1
    R  = np.log(w1 / w0)

    k     = np.exp(t * R / dur)                 # instantaneous freq envelope, 1 → w1/w0
    sweep = np.sin((w0 * dur / R) * (k - 1.0))

    # Short raised-cosine fades to suppress start/stop clicks.
    nf = int(round(fade * fs))
    if nf > 0 and 2 * nf < n:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(nf) / nf))
        sweep[:nf]  *= ramp
        sweep[-nf:] *= ramp[::-1]

    inverse = sweep[::-1] / k                    # amplitude-compensated inverse

    sweep *= amp
    return sweep.astype(np.float32), inverse.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Deconvolution → impulse response
# ---------------------------------------------------------------------------

def measure_ir(recorded: np.ndarray, inverse: np.ndarray, fs: int,
               ir_pre_ms: float = 5.0,
               ir_post_ms: float = 80.0) -> "tuple[np.ndarray, int]":
    """
    Deconvolve `recorded` with the inverse sweep → linear impulse response.

    The linear IR is the dominant peak of the deconvolution; harmonic
    distortion appears *before* it (negative time) and is excluded by the
    causal window [peak - ir_pre_ms, peak + ir_post_ms].

    Returns (h_lin, peak_index_in_full_deconv).
    """
    h_full = fftconvolve(recorded, inverse, mode="full")
    peak   = int(np.argmax(np.abs(h_full)))
    pre    = int(round(ir_pre_ms  * fs / 1000.0))
    post   = int(round(ir_post_ms * fs / 1000.0))
    lo     = max(0, peak - pre)
    hi     = min(len(h_full), peak + post)
    return h_full[lo:hi].astype(np.float64), peak


def delay_spread(h_lin: np.ndarray, fs: int, floor_db: float = 30.0) -> "tuple[float, float]":
    """
    RMS delay spread and mean delay from the power-delay profile |h|².

    Samples more than `floor_db` below the peak power are gated out so the
    reverb-tail estimate isn't dominated by the noise floor.

    Returns (tau_rms_s, tau_mean_s).
    """
    p = h_lin.astype(np.float64) ** 2
    if p.max() <= 0:
        return 0.0, 0.0
    mask = p >= (p.max() * 10.0 ** (-floor_db / 10.0))
    idx  = np.nonzero(mask)[0]
    t    = idx / fs
    pw   = p[idx]
    tau_mean = float(np.sum(t * pw) / np.sum(pw))
    tau_rms  = float(np.sqrt(np.sum(((t - tau_mean) ** 2) * pw) / np.sum(pw)))
    return tau_rms, tau_mean


# ---------------------------------------------------------------------------
# 3. Channel analysis (H(f), noise, SNR(f))
# ---------------------------------------------------------------------------

def _find_sweep_start(recorded: np.ndarray, sweep: np.ndarray) -> int:
    """Matched-filter the recording against the sweep to find its start sample."""
    corr = fftconvolve(recorded, sweep[::-1], mode="valid")
    return int(np.argmax(np.abs(corr)))


def analyze_channel(recorded: np.ndarray, fs: int, sweep: np.ndarray,
                    inverse: np.ndarray, f0: float, f1: float,
                    noise_dur: float = DEF_SILENCE) -> dict:
    """
    Full channel measurement from a `[silence | sweep | silence]` recording.

    * noise floor  N(f)  ← leading `noise_dur` seconds (pure noise by design)
    * received PSD S_y(f) ← the recorded sweep segment (signal + noise)
    * SNR(f) = (S_y - N) / N   in dB, meaningful only inside the swept band
    * |H(f)|, group delay, tau_rms ← deconvolved linear impulse response

    Returns a JSON-serialisable dict of arrays + scalars.
    """
    recorded = np.asarray(recorded, dtype=np.float64)

    # --- impulse response + delay spread -------------------------------------
    h_lin, _peak = measure_ir(recorded, inverse, fs)
    tau_rms, tau_mean = delay_spread(h_lin, fs)

    H     = np.fft.rfft(h_lin)
    hf    = np.fft.rfftfreq(len(h_lin), 1.0 / fs)
    mag   = np.abs(H)
    mag_db = 20.0 * np.log10(mag / (mag.max() + 1e-12) + 1e-12)
    gdelay = -np.gradient(np.unwrap(np.angle(H + 1e-12))) / (2 * np.pi * (hf[1] - hf[0] + 1e-12))

    # --- noise vs signal PSD → SNR(f) ---------------------------------------
    n_noise = int(round(noise_dur * fs))
    noise_seg = recorded[:n_noise] if n_noise > 16 else recorded[: len(recorded) // 10]

    start   = _find_sweep_start(recorded, sweep)
    sig_seg = recorded[start: start + len(sweep)]

    nper = int(min(4096, max(256, len(sig_seg) // 8)))
    fN, Pn = welch(noise_seg, fs=fs, nperseg=min(nper, len(noise_seg)))
    fS, Ps = welch(sig_seg,   fs=fs, nperseg=min(nper, len(sig_seg)))

    # Common grid (welch grids match when nperseg matches; interp to be safe).
    freqs = fS
    Pn_i  = np.interp(freqs, fN, Pn)
    snr   = np.clip(Ps - Pn_i, 1e-20, None) / (Pn_i + 1e-20)
    snr_db = 10.0 * np.log10(snr)

    # Only the swept band carries a real measurement.
    band_mask = (freqs >= f0) & (freqs <= f1)
    snr_db_masked = np.where(band_mask, snr_db, np.nan)

    noise_floor_db = float(10.0 * np.log10(np.mean(Pn_i[band_mask]) + 1e-20))

    return {
        "fs":             fs,
        "f0":             float(f0),
        "f1":             float(f1),
        "freqs":          freqs.tolist(),
        "snr_db":         snr_db_masked.tolist(),
        "mag_freqs":      hf.tolist(),
        "mag_db":         mag_db.tolist(),
        "group_delay_s":  gdelay.tolist(),
        "tau_rms_s":      tau_rms,
        "tau_mean_s":     tau_mean,
        "noise_floor_db": noise_floor_db,
        "ir":             h_lin.tolist(),
    }


# ---------------------------------------------------------------------------
# 4. Parameter selection (compute, don't sweep)
# ---------------------------------------------------------------------------

def _divisor_symbol_rates(fs: int, sr_min: int = 200, sr_max: "int | None" = None) -> list:
    if sr_max is None:
        sr_max = fs // 4            # samples/symbol >= 4 (modem hard requirement)
    return [d for d in range(sr_min, sr_max + 1) if fs % d == 0]


def _min_snr_over_band(freqs: np.ndarray, snr_db: np.ndarray,
                       lo: float, hi: float) -> "float | None":
    m = (freqs >= lo) & (freqs <= hi)
    vals = snr_db[m]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None
    return float(np.min(vals))


def select_qam_params(freqs, snr_db, tau_rms: float, fs: int,
                      f0: float = DEF_F0, f1: "float | None" = None,
                      margin_db: float = SEL_MARGIN_DB,
                      eq_symbols: float = SEL_EQ_SYMBOLS,
                      cf_step: float = SEL_CF_STEP) -> dict:
    """
    Argmax throughput over every legal (carrier, symbol_rate), given SNR(f).

    A candidate survives iff:
      * `Modulator.auto_beta` deems it legal (returns a beta),
      * its occupied band [cf-BW/2, cf+BW/2] lies within [f0, f1],
      * the *worst* SNR across that band >= margin_db,
      * its delay spread stays within the equalizer's reach:
        tau_rms <= eq_symbols / sr   →   sr <= eq_symbols / tau_rms.

    Higher symbol rates are *more* ISI-tolerant here because the equalizer works
    in symbol periods, not seconds — a faster clock fits more symbol slots inside
    the same reverb tail.  Throughput is symbol_rate (16-QAM → 4 bits/sym fixed),
    tie-broken on mean band SNR.  Returns the winner + a diagnostic shortlist.
    """
    freqs  = np.asarray(freqs, dtype=np.float64)
    snr_db = np.asarray(snr_db, dtype=np.float64)
    if f1 is None:
        f1 = DEF_F1_FRAC * fs

    isi_max_sr = (eq_symbols / tau_rms) if tau_rms > 0 else float("inf")

    scored = []          # surviving candidates
    reasons = {}         # reason -> count
    def _rej(r):
        reasons[r] = reasons.get(r, 0) + 1

    for sr in _divisor_symbol_rates(fs):
        if sr > isi_max_sr:
            _rej("delay_spread")
            continue
        for cf in np.arange(f0, f1, cf_step):
            beta = Modulator.auto_beta(int(cf), int(sr), fs)
            if beta is None:
                _rej("illegal")
                continue
            bw = sr * (1.0 + beta)
            lo, hi = cf - bw / 2.0, cf + bw / 2.0
            if lo < f0 or hi > f1:
                _rej("band_out_of_range")
                continue
            smin = _min_snr_over_band(freqs, snr_db, lo, hi)
            if smin is None:
                _rej("no_snr_data")
                continue
            if smin < margin_db:
                _rej("snr_below_margin")
                continue
            smean = float(np.nanmean(np.where((freqs >= lo) & (freqs <= hi),
                                              snr_db, np.nan)))
            scored.append({
                "carrier_freq": int(round(cf)),
                "symbol_rate":  int(sr),
                "beta":         round(beta, 4),
                "band_lo":      round(lo, 1),
                "band_hi":      round(hi, 1),
                "min_snr_db":   round(smin, 1),
                "mean_snr_db":  round(smean, 1),
                "gross_bps":    int(sr * 4),
            })

    scored.sort(key=lambda c: (c["symbol_rate"], c["mean_snr_db"]), reverse=True)

    best = scored[0] if scored else None
    return {
        "best":           {"carrier_freq": best["carrier_freq"],
                           "symbol_rate":  best["symbol_rate"]} if best else None,
        "best_detail":    best,
        "n_surviving":    len(scored),
        "reject_reasons": reasons,
        "shortlist":      scored[:8],
        "tau_rms_s":      tau_rms,
        "margin_db":      margin_db,
        "eq_symbols":     eq_symbols,
        "isi_max_sr":     round(isi_max_sr, 1) if np.isfinite(isi_max_sr) else None,
    }


# ---------------------------------------------------------------------------
# 5. Synthetic channel (hardware-free test harness)
# ---------------------------------------------------------------------------

def apply_synthetic_channel(sweep: np.ndarray, fs: int,
                            taps: "list[tuple[float, float]]",
                            band: "tuple[float, float] | None",
                            snr_db: float,
                            silence: float = DEF_SILENCE,
                            seed: int = 0) -> np.ndarray:
    """
    Push `sweep` through a known channel and wrap it in silence + noise, so the
    analyzer can be validated with no hardware.

    taps   : [(delay_ms, amplitude), ...]  multipath impulse response.
    band   : (lo, hi) Hz bandpass, or (0, hi) lowpass, or None for flat.
    snr_db : broadband received SNR of the sweep segment.

    Returns float64 recording [silence | channel(sweep)+noise | silence+noise].
    """
    rng = np.random.default_rng(seed)

    max_ms = max(t for t, _ in taps)
    ir = np.zeros(int(round(max_ms * fs / 1000.0)) + 1)
    for t_ms, amp in taps:
        ir[int(round(t_ms * fs / 1000.0))] += amp
    y = fftconvolve(sweep, ir, mode="full")

    if band is not None:
        lo, hi = band
        if lo <= 0:
            fir = firwin(255, hi, fs=fs)
        else:
            fir = firwin(255, [lo, hi], pass_zero=False, fs=fs)
        y = fftconvolve(y, fir, mode="same")

    pad = int(round(silence * fs))
    clean = np.concatenate([np.zeros(pad), y, np.zeros(pad)])

    sig_rms = np.sqrt(np.mean(y ** 2))
    noise_sigma = sig_rms / (10.0 ** (snr_db / 20.0))
    return clean + rng.normal(0.0, noise_sigma, len(clean))


# ---------------------------------------------------------------------------
# 6. Optional plot
# ---------------------------------------------------------------------------

def make_channel_snr_png(report: dict, selection: dict,
                         output_path: "str | Path",
                         title: "str | None" = None) -> bool:
    """3-panel channel summary: |H(f)|, SNR(f) w/ chosen band, impulse response."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _log("  channel_sound: matplotlib not installed — skipping PNG")
        return False

    try:
        fs     = report["fs"]
        freqs  = np.array(report["freqs"])
        snr    = np.array(report["snr_db"])
        mf     = np.array(report["mag_freqs"])
        mag    = np.array(report["mag_db"])
        ir     = np.array(report["ir"])
        best   = (selection or {}).get("best_detail")

        fig, ax = plt.subplots(3, 1, figsize=(9, 9))
        ax[0].plot(mf, mag, lw=0.8)
        ax[0].set(title="Channel magnitude |H(f)|", xlabel="Hz", ylabel="dB",
                  xlim=(0, fs / 2))
        ax[0].grid(alpha=0.3)

        ax[1].plot(freqs, snr, lw=0.9, color="tab:green")
        if best:
            ax[1].axvspan(best["band_lo"], best["band_hi"], color="tab:orange",
                          alpha=0.25, label=f"chosen cf={best['carrier_freq']} "
                                            f"sr={best['symbol_rate']}")
            ax[1].legend(loc="upper right", fontsize=8)
        ax[1].axhline(selection.get("margin_db", SEL_MARGIN_DB), ls="--",
                      color="gray", lw=0.8)
        ax[1].set(title="Received SNR(f)", xlabel="Hz", ylabel="dB",
                  xlim=(0, fs / 2))
        ax[1].grid(alpha=0.3)

        t_ms = np.arange(len(ir)) / fs * 1000.0
        ax[2].plot(t_ms, ir / (np.abs(ir).max() + 1e-12), lw=0.8)
        ax[2].set(title=f"Impulse response  (tau_rms = "
                        f"{report['tau_rms_s'] * 1e3:.2f} ms)",
                  xlabel="ms", ylabel="norm")
        ax[2].grid(alpha=0.3)

        fig.suptitle(title or "Channel sounding")
        fig.tight_layout()
        fig.savefig(output_path, dpi=110)
        plt.close(fig)
        _log(f"  channel PNG -> {output_path}")
        return True
    except Exception as exc:
        _log(f"  channel_sound: PNG failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------

def _write_wav(path, fs, x):
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(path, fs, pcm)


def _read_wav(path):
    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    return data.astype(np.float64), fs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_gen(args) -> int:
    sweep, inverse = make_ess(args.fs, args.f0, args.f1, args.dur, args.amp)
    silence = np.zeros(int(round(args.silence * args.fs)))
    tx = np.concatenate([silence, sweep, silence])
    _write_wav(args.out, args.fs, tx)

    meta_path = Path(args.out).with_suffix(".json")
    np.save(Path(args.out).with_suffix(".inv.npy"), inverse)
    f1 = args.f1 if args.f1 else DEF_F1_FRAC * args.fs
    meta_path.write_text(json.dumps({
        "fs": args.fs, "f0": args.f0, "f1": f1, "dur": args.dur,
        "amp": args.amp, "silence": args.silence,
        "wav": str(args.out), "inverse": str(Path(args.out).with_suffix(".inv.npy")),
    }, indent=2))
    _log(f"Wrote sweep -> {args.out}  (+ {meta_path.name}, {meta_path.stem}.inv.npy)")
    _log(f"  {args.dur:.1f}s ESS  {args.f0:.0f}-{f1:.0f} Hz @ {args.fs} Hz")
    return 0


def cmd_analyze(args) -> int:
    meta = json.loads(Path(args.sweep).read_text())
    fs = meta["fs"]
    sweep, inverse = make_ess(fs, meta["f0"], meta["f1"], meta["dur"], meta["amp"])

    rec, rec_fs = _read_wav(args.rec)
    if rec_fs != fs:
        _log(f"WARNING: recording fs {rec_fs} != sweep fs {fs}; analyzing at {rec_fs}")
        fs = rec_fs

    report = analyze_channel(rec, fs, sweep, inverse, meta["f0"], meta["f1"],
                             noise_dur=meta.get("silence", DEF_SILENCE))
    selection = select_qam_params(report["freqs"], report["snr_db"],
                                  report["tau_rms_s"], fs,
                                  f0=meta["f0"], f1=meta["f1"],
                                  margin_db=args.margin_db, eq_symbols=args.eq_symbols)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "channel_report.json").write_text(json.dumps(
        {k: v for k, v in report.items()
         if k not in ("freqs", "snr_db", "mag_freqs", "mag_db", "group_delay_s", "ir")},
        indent=2))
    (out_dir / "channel_full.json").write_text(json.dumps(report))
    (out_dir / "channel_selection.json").write_text(json.dumps(selection, indent=2))
    if selection["best"]:
        (out_dir / "probe_best.json").write_text(json.dumps(selection["best"], indent=2))
    make_channel_snr_png(report, selection, out_dir / "channel_snr.png",
                         title="Channel sounding")

    _log(f"tau_rms = {report['tau_rms_s']*1e3:.2f} ms   "
         f"noise_floor = {report['noise_floor_db']:.1f} dB")
    if selection["best"]:
        b = selection["best_detail"]
        _log(f"BEST: carrier_freq={b['carrier_freq']} symbol_rate={b['symbol_rate']} "
             f"(band {b['band_lo']:.0f}-{b['band_hi']:.0f} Hz, "
             f"min SNR {b['min_snr_db']} dB, gross {b['gross_bps']} bps)")
    else:
        _log(f"No legal config met the margin. reasons={selection['reject_reasons']}")
    return 0


def cmd_selftest(args) -> int:
    fs = DEF_FS
    sweep, inverse = make_ess(fs)
    f1 = DEF_F1_FRAC * fs
    taps = [(0.0, 1.0), (2.0, 0.4), (5.0, 0.2)]   # multipath → tau_rms ≈ 1.09 ms
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        _log(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
        ok = ok and cond

    # --- Test A: flat, high-SNR channel ------------------------------------
    recA = apply_synthetic_channel(sweep, fs, taps, band=None, snr_db=35, seed=1)
    repA = analyze_channel(recA, fs, sweep, inverse, DEF_F0, f1)
    selA = select_qam_params(repA["freqs"], repA["snr_db"], repA["tau_rms_s"],
                             fs, f0=DEF_F0, f1=f1)
    tau_ms = repA["tau_rms_s"] * 1e3
    check("A delay spread recovered (0.3–3 ms)", 0.3 < tau_ms < 3.0,
          f"tau_rms={tau_ms:.2f} ms")
    check("A found a legal config", selA["best"] is not None,
          f"best={selA['best']}")
    if selA["best"]:
        b = selA["best"]
        legal = Modulator.auto_beta(b["carrier_freq"], b["symbol_rate"], fs) is not None
        check("A selection is modem-legal", legal, f"{b}")
        check("A picked a high symbol rate (flat channel)",
              b["symbol_rate"] >= 2000, f"sr={b['symbol_rate']}")

    # --- Test B: 4 kHz low-pass channel ------------------------------------
    recB = apply_synthetic_channel(sweep, fs, taps, band=(0, 4000), snr_db=35, seed=2)
    repB = analyze_channel(recB, fs, sweep, inverse, DEF_F0, f1)
    selB = select_qam_params(repB["freqs"], repB["snr_db"], repB["tau_rms_s"],
                             fs, f0=DEF_F0, f1=f1)
    freqs = np.array(repB["freqs"]); snr = np.array(repB["snr_db"])
    pass_snr = np.nanmean(snr[(freqs > 1000) & (freqs < 3000)])
    stop_snr = np.nanmean(snr[(freqs > 6000) & (freqs < 10000)])
    check("B passband SNR >> stopband SNR", (pass_snr - stop_snr) > 15,
          f"pass={pass_snr:.1f} dB  stop={stop_snr:.1f} dB")
    if selB["best_detail"]:
        hi = selB["best_detail"]["band_hi"]
        check("B selector avoids the rolled-off band (band_hi <= 4.5 kHz)",
              hi <= 4500, f"band_hi={hi:.0f} Hz")
    else:
        check("B found a config in the passband", False, "no config selected")

    _log("")
    _log(f"SELFTEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Acoustic channel sounder / QAM param selector")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate a sweep WAV + inverse filter")
    g.add_argument("--out", default="sweep_tx.wav")
    g.add_argument("--fs", type=int, default=DEF_FS)
    g.add_argument("--f0", type=float, default=DEF_F0)
    g.add_argument("--f1", type=float, default=None)
    g.add_argument("--dur", type=float, default=DEF_DUR)
    g.add_argument("--amp", type=float, default=DEF_AMP)
    g.add_argument("--silence", type=float, default=DEF_SILENCE)
    g.set_defaults(func=cmd_gen)

    a = sub.add_parser("analyze", help="analyze a recording against a sweep")
    a.add_argument("--rec", required=True, help="recorded WAV [silence|sweep|silence]")
    a.add_argument("--sweep", required=True, help="sweep metadata JSON from `gen`")
    a.add_argument("--out-dir", default=".")
    a.add_argument("--margin-db", type=float, default=SEL_MARGIN_DB)
    a.add_argument("--eq-symbols", type=float, default=SEL_EQ_SYMBOLS)
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("selftest", help="hardware-free end-to-end validation")
    s.set_defaults(func=cmd_selftest)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
