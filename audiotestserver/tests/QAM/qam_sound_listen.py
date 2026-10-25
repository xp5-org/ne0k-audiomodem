# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
tests/qam_sound_listen.py - role: listen (pair with qam_sound_play.py)

Channel-sounding probe receiver.  Records the player's single Farina sweep once
(at THIS device's native input rate - no resampling), measures the channel, and
COMPUTES the best QAM (carrier, symbol_rate) via channel_sound.select_qam_params.
No per-combo decoding.

Outputs (in RUN_DIR)
--------------------
    <client>_sound_capture.wav   - raw single-shot mic capture
    channel_report.json          - tau_rms, noise floor, selection summary
    channel_selection.json       - full candidate shortlist + reject reasons
    channel_full.json            - freqs / SNR(f) / |H(f)| / IR arrays
    channel_snr.png              - |H(f)|, SNR(f) w/ chosen band, impulse response
    probe_best.json              - {carrier_freq, symbol_rate}  (pipeline contract)
    result_<client>.json         - framework result

Phase-C confirmation (if player set confirm=true): after publishing probe_best,
record the player's single confirmation frame and decode it with Demodulator to
verify the computed config actually works.

Params:
{
    "record_pad_secs":     3.0,   // slack beyond expected sweep window
    "noise_secs":          0.4,   // leading silence used for the noise floor
    "margin_db":           12.0,  // min in-band SNR to accept a config
    "eq_symbols":          4.0,   // delay-spread tolerance (equalizer reach)
    "confirm_secs":        20,    // wait for player's confirm burst
    "confirm_record_secs": 6.0,   // confirmation recording window
    "zc_threshold":        0.45
}
"""

import json
import time
from pathlib import Path

import numpy as np

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

from channel_sound import (

    make_ess, analyze_channel, select_qam_params, make_channel_snr_png,
    _write_wav, DEF_F1_FRAC,
)

from testlib import (
    CLIENT_ID, INPUT_DEV, RUN_DIR, PARAMS, RUN_ID,
    log, to_int_dev, coord_signal, coord_check, coord_wait_any,
)

TESTS_DIR = Path(__file__).parent

RECORD_PAD_SECS     = float(PARAMS.get("record_pad_secs",      3.0))
NOISE_SECS          = float(PARAMS.get("noise_secs",           0.4))
MARGIN_DB           = float(PARAMS.get("margin_db",           12.0))
EQ_SYMBOLS          = float(PARAMS.get("eq_symbols",           4.0))
CONFIRM_SECS        = float(PARAMS.get("confirm_secs",          20))
CONFIRM_RECORD_SECS = float(PARAMS.get("confirm_record_secs",  6.0))
ZC_THRESH           = float(PARAMS.get("zc_threshold",        0.45))

def _native_rate(dev) -> int:
    import sounddevice as sd
    try:
        info = sd.query_devices(dev if dev is not None else None, kind="input")
        return int(round(info["default_samplerate"]))
    except Exception:
        return 48000

def coord_wait_manifest(timeout=90):
    path = RUN_DIR / "sound_manifest.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return path
        time.sleep(0.3)
    return None

def confirm_decode(in_dev, fs: int) -> dict:
    """
    Phase C: record the player's single confirmation frame and decode it.
    Best-effort; never raises.
    """
    import sounddevice as sd
    from qam_reader import Demodulator

    log(f"  [confirm] waiting for player's confirm burst (confirm_ready) ...")
    if not coord_wait_any("confirm_ready", timeout=CONFIRM_SECS):
        log("  [confirm] player never signalled confirm_ready - skipping")
        return {"confirm": "skipped_no_burst"}

    cfg_path = RUN_DIR / "confirm_config.json"
    if not cfg_path.exists():
        log("  [confirm] confirm_config.json missing - skipping")
        return {"confirm": "skipped_no_config"}

    n = int(CONFIRM_RECORD_SECS * fs)
    rec = sd.rec(n, samplerate=fs, channels=1, dtype="float32", device=in_dev)
    coord_signal("confirm_armed")
    sd.wait()
    capture = rec[:, 0]

    wav_path = RUN_DIR / f"{CLIENT_ID}_confirm_capture.wav"
    _write_wav(str(wav_path), fs, capture / (np.abs(capture).max() + 1e-9))

    try:
        demod = Demodulator(config_path=str(cfg_path))
        demod.ZC_THRESHOLD = ZC_THRESH
        stats_path = RUN_DIR / "confirm_stats.json"
        demod._stats_out = str(stats_path)
        text = demod.process_wav(str(wav_path))
        n_dec  = len(demod.received)
        total  = demod.total_frames or 0
        errs   = sum(f["err_count"] for f in demod._frame_stats)
        snr    = (sum(f.get("snr_evm_db", 0) for f in demod._frame_stats) / n_dec
                  if n_dec else 0)
        status = "ok" if (total and n_dec >= total) else ("partial" if n_dec else "fail")
        log(f"  [confirm] decoded {n_dec}/{total} frames  errs={errs}  "
            f"SNR~{snr:.1f}dB  -> {status}")
        return {
            "confirm":            status,
            "confirm_frames":     f"{n_dec}/{total}",
            "confirm_rs_errors":  errs,
            "confirm_snr_evm_db": round(snr, 1),
            "confirm_text":       text[:60],
        }
    except Exception as exc:
        log(f"  [confirm] decode failed: {exc}")
        return {"confirm": "decode_failed", "error": str(exc)}

def _save_error(reason):
    result = {"client_id": CLIENT_ID, "role": "listen", "run_id": RUN_ID,
              "status": "error", "failure_reason": reason}
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

def main():
    import sounddevice as sd

    in_dev = to_int_dev(INPUT_DEV)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    fs = _native_rate(in_dev)
    log(f"[{CLIENT_ID}] qam_sound_listen | input_device={in_dev} native_fs={fs}")

    log("  waiting for sound_manifest.json ...")
    mpath = coord_wait_manifest(timeout=90)
    if not mpath:
        log("  ERROR: manifest not found")
        _save_error("no_manifest"); return
    meta = json.loads(mpath.read_text())

    f0  = float(meta["f0"])
    f1  = float(meta.get("f1", meta.get("f1_frac", DEF_F1_FRAC) * fs))
    dur = float(meta["dur"])
    sil = float(meta["silence_secs"])

    # Regenerate the reference sweep at OUR native rate from physical params.
    sweep, inverse = make_ess(fs, f0, f1, dur, meta.get("amp", 0.7))

    record_secs = sil * 2 + dur + RECORD_PAD_SECS
    n = int(record_secs * fs)
    log(f"  recording {record_secs:.1f}s @ {fs}Hz (expected sweep window "
        f"{sil*2+dur:.1f}s + {RECORD_PAD_SECS:.1f}s pad)")

    rec = sd.rec(n, samplerate=fs, channels=1, dtype="float32", device=in_dev)
    coord_signal("sound_armed")
    sd.wait()
    capture = rec[:, 0].astype(np.float64)

    if not coord_check("sound_done"):
        log("  WARNING: sound_done not seen - sweep may have been clipped")

    cap_path = RUN_DIR / f"{CLIENT_ID}_sound_capture.wav"
    _write_wav(str(cap_path), fs, capture / (np.abs(capture).max() + 1e-9))
    log(f"  capture -> {cap_path.name}")

    # --- Analyse + select --------------------------------------------------
    report = analyze_channel(capture, fs, sweep, inverse, f0, f1,
                             noise_dur=min(NOISE_SECS, sil))
    selection = select_qam_params(report["freqs"], report["snr_db"],
                                  report["tau_rms_s"], fs, f0=f0, f1=f1,
                                  margin_db=MARGIN_DB, eq_symbols=EQ_SYMBOLS)

    (RUN_DIR / "channel_report.json").write_text(json.dumps(
        {k: v for k, v in report.items()
         if k not in ("freqs", "snr_db", "mag_freqs", "mag_db", "group_delay_s", "ir")}
        | {"selection": {k: selection[k] for k in
                         ("best", "best_detail", "n_surviving", "reject_reasons",
                          "isi_max_sr")}},
        indent=2))
    (RUN_DIR / "channel_full.json").write_text(json.dumps(report))
    (RUN_DIR / "channel_selection.json").write_text(json.dumps(selection, indent=2))
    make_channel_snr_png(report, selection, RUN_DIR / "channel_snr.png",
                         title=f"Channel sounding — {CLIENT_ID}")

    log(f"  tau_rms={report['tau_rms_s']*1e3:.2f}ms  "
        f"noise_floor={report['noise_floor_db']:.1f}dB  "
        f"isi_max_sr={selection['isi_max_sr']}")

    if selection["best"]:
        b = selection["best_detail"]
        (RUN_DIR / "probe_best.json").write_text(json.dumps(selection["best"], indent=2))
        log(f"  BEST: carrier_freq={b['carrier_freq']} symbol_rate={b['symbol_rate']}  "
            f"band {b['band_lo']:.0f}-{b['band_hi']:.0f}Hz  "
            f"minSNR={b['min_snr_db']}dB  gross={b['gross_bps']}bps")
    else:
        log(f"  No config met the margin. reasons={selection['reject_reasons']}")

    coord_signal("sound_selected")

    # --- Phase C confirmation ---------------------------------------------
    confirm_result = {"confirm": "disabled"}
    if meta.get("confirm") and selection["best"]:
        confirm_result = confirm_decode(in_dev, fs)
    elif meta.get("confirm"):
        log("  [confirm] no selection to confirm - skipping")
        confirm_result = {"confirm": "skipped_no_selection"}

    status = "ok" if selection["best"] else "fail"
    final = {
        "client_id": CLIENT_ID, "role": "listen", "run_id": RUN_ID,
        "status": status, "native_fs": fs,
        "best": selection["best"],
        "tau_rms_ms": round(report["tau_rms_s"] * 1e3, 2),
        "noise_floor_db": round(report["noise_floor_db"], 1),
        "n_surviving": selection["n_surviving"],
        **confirm_result,
    }
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final), flush=True)

if __name__ == "__main__":
    main()
