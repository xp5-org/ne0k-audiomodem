"""
ofdm_play.py  --  role: play  (pair with ofdm_decode.py)

Signal sequence:
  1. Write ofdm_config.json        (config available for decode side)
  2. Signal  'ofdm_ready'
  3. Wait    'ofdm_armed'          (decoder is recording)
  4. Play    probe                 (probe_duration_s seconds)
  5. Signal  'ofdm_probe_played'
  6. Wait    'ofdm_bat_ready'      (decoder computed BAT)
  7. Encode + play data
  8. Signal  'ofdm_played'

Params (JSON field presented to user):
{
    "input_str":        "Hello world",
    "amplitude":        0.3,
    "n_fft":            1024,
    "cp_ratio":         0.25,
    "freq_lo":          1000,
    "freq_hi":          16000,
    "pilot_spacing":    8,
    "sample_rate":      48000,
    "ecc_bytes":        16,
    "chunk_bytes":      32,
    "seq_bytes":        2,
    "taper_ms":         2.0,
    "frame_gap_ms":     20.0,
    "probe_duration_s": 4.0,
    "snr_margin_db":    6.0,
    "max_bits_per_bin": 6,
    "sync_period":      32
}
"""
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

import json
import sys
from pathlib import Path

import numpy as np
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

from testlib import (
    CLIENT_ID, OUTPUT_DEV, RUN_DIR, PARAMS, RUN_ID,
    log, to_int_dev, coord_signal, coord_wait_any,
)

from ofdm_modem import build_config, Modem

def _play(audio, sr, device, label):
    try:
        import sounddevice as sd
        log(f"  playing {label} ({len(audio)/sr:.2f}s) on device {device}...")
        sd.play(audio, samplerate=sr, device=device)
        sd.wait()
        log(f"  {label} complete")
        return "ok"
    except Exception as exc:
        log(f"  playback error ({label}): {exc}")
        return str(exc)

def main():
    out_dev = to_int_dev(OUTPUT_DEV)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[{CLIENT_ID}] ofdm_play | output_device={out_dev}")

    cfg = build_config(
        n_fft            = int(PARAMS.get("n_fft",            1024)),
        cp_ratio         = float(PARAMS.get("cp_ratio",        0.25)),
        freq_lo          = float(PARAMS.get("freq_lo",         1000)),
        freq_hi          = float(PARAMS.get("freq_hi",         16000)),
        pilot_spacing    = int(PARAMS.get("pilot_spacing",     8)),
        sample_rate      = int(PARAMS.get("sample_rate",       48000)),
        ecc_bytes        = int(PARAMS.get("ecc_bytes",         16)),
        chunk_bytes      = int(PARAMS.get("chunk_bytes",       256)),
        seq_bytes        = int(PARAMS.get("seq_bytes",         2)),
        taper_ms         = float(PARAMS.get("taper_ms",        2.0)),
        frame_gap_ms     = float(PARAMS.get("frame_gap_ms",    20.0)),
        probe_duration_s = float(PARAMS.get("probe_duration_s", 4.0)),
        snr_margin_db    = float(PARAMS.get("snr_margin_db",   6.0)),
        max_bits_per_bin = int(PARAMS.get("max_bits_per_bin",  6)),
        sync_period      = int(PARAMS.get("sync_period",       32)),
    )
    modem = Modem(cfg)
    sr    = cfg["sample_rate"]

    # Peak amplitude as a fraction of full scale, applied to BOTH probe and data.
    # The BAT is measured at the probe's level, so transmitting data at a
    # different level would invalidate it.
    amplitude = float(PARAMS.get("amplitude", 0.3))

    log(f"  n_data_carriers={cfg['n_data_carriers']}  "
        f"band={cfg['freq_lo_hz']:.0f}-{cfg['freq_hi_hz']:.0f} Hz  "
        f"probe={cfg['actual_probe_s']}s ({cfg['n_probe_syms']} syms)  "
        f"amp={amplitude}")

    # Write config so decode side can reconstruct identical parameters
    cfg_path = RUN_DIR / "ofdm_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    log(f"  config written: {cfg_path.name}")

    # Build probe
    probe_sig   = modem.build_probe()
    probe_scale = amplitude * 32700.0 / max(float(np.max(np.abs(probe_sig))), 1e-6)
    probe_float = (probe_sig * probe_scale).astype(np.int16).astype("float32") / 32768.0

    # Coordinate: wait for decoder to arm before playing
    coord_signal("ofdm_ready")
    log("  signalled ofdm_ready -- waiting for decoder to arm...")
    armed = coord_wait_any("ofdm_armed", timeout=30)
    if not armed:
        log("  WARNING: no decoder armed in 30s -- playing probe anyway")

    probe_status = _play(probe_float, sr, out_dev, "probe")
    coord_signal("ofdm_probe_played")

    # Wait for BAT (decoder needs probe_duration_s to analyse + margin)
    bat_timeout = cfg["actual_probe_s"] + 30
    bat_path    = RUN_DIR / "ofdm_bat.json"
    log(f"  waiting for BAT (timeout={bat_timeout:.0f}s)...")
    bat_signal = coord_wait_any("ofdm_bat_ready", timeout=bat_timeout)
    if not bat_signal or not bat_path.exists():
        log("  ERROR: BAT not received -- aborting")
        print(json.dumps({"client_id": CLIENT_ID, "status": "error",
                          "error": "no_bat", "run_id": RUN_ID}), flush=True)
        return

    bat = Modem.load_bat(str(bat_path))
    log(f"  BAT received: {modem.bat_summary(bat)}")

    # Encode data
    input_str = PARAMS.get("input_str")
    if not input_str:
        log("  ERROR: no input_str in params")
        print(json.dumps({"client_id": CLIENT_ID, "status": "error",
                          "error": "no_input_str", "run_id": RUN_ID}), flush=True)
        return

    log(f"  encoding {len(input_str)} chars...")
    try:
        data_sig, total_frames = modem.encode_data(input_str, bat)
    except Exception as exc:
        log(f"  encode error: {exc}")
        print(json.dumps({"client_id": CLIENT_ID, "status": "error",
                          "error": str(exc), "run_id": RUN_ID}), flush=True)
        return

    # Update config with frame count so decoder knows what to expect
    cfg["total_frames"] = total_frames
    cfg["bits_per_bin"] = bat.tolist()
    cfg_path.write_text(json.dumps(cfg, indent=2))

    data_scale = amplitude * 32700.0 / max(float(np.max(np.abs(data_sig))), 1e-6)
    data_pcm   = (data_sig * data_scale).astype(np.int16)
    data_float = data_pcm.astype("float32") / 32768.0

    wav_path = RUN_DIR / "ofdm_data.wav"
    wavfile.write(str(wav_path), sr, data_pcm)
    log(f"  data WAV: {wav_path.name}  ({len(data_pcm)/sr:.2f}s, {total_frames} frames)")

    data_armed = coord_wait_any("ofdm_data_armed", timeout=30)
    if not data_armed:
        log("  WARNING: decoder not armed for data -- playing anyway")

    data_status = _play(data_float, sr, out_dev, "data")
    coord_signal("ofdm_played")

    result = {
        "client_id":       CLIENT_ID,
        "role":            "play",
        "run_id":          RUN_ID,
        "status":          "ok" if data_status == "ok" else "error",
        "probe_status":    probe_status,
        "data_status":     data_status,
        "total_frames":    total_frames,
        "duration_s":      round(len(data_pcm) / sr, 2),
        "probe_duration_s": cfg["actual_probe_s"],
        "n_probe_syms":    cfg["n_probe_syms"],
        "active_carriers": int(np.sum(bat > 0)),
        "avg_bits_per_bin": (round(float(np.mean(bat[bat > 0])), 2)
                             if np.any(bat > 0) else 0),
        "bat_summary":     modem.bat_summary(bat),
    }
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
