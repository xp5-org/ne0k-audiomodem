# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
tests/qam_sound_play.py - role: play (pair with qam_sound_listen.py)

Channel-sounding probe transmitter.  Plays one Farina exponential sine
sweep bracketed by silence.  The listener measures the channel from that
single shot and computes the best QAM (carrier, symbol_rate) - see
channel_sound.py / qam_channel_sounding_probe.md.

Native-rate discipline
----------------------
The sweep is generated at the OUTPUT device's native sample rate and played
without resampling.  The listener regenerates its reference sweep at its own
native input rate from the shared (f0, f1, dur): frequencies and duration are
physical, so no side resamples.  48k<->44.1k resampling destroys sweep phase.

Coordination (4 signals, no per-combo handshakes)
-------------------------------------------------
    sound_ready   -> manifest written, sweep ready
    sound_armed   <- listener has started recording
    sound_start   -> about to play
    sound_done    -> playback finished

Optional Phase-C confirmation (confirm=true): after the listener writes
probe_best.json, transmit a single short frame at the chosen config so the
listener can verify the prediction actually decodes.

Params:
{
    "f0":            200,     // sweep start (Hz)
    "f1_frac":       0.45,    // sweep end as fraction of native fs
    "sweep_secs":    3.0,
    "amplitude":     0.7,
    "silence_secs":  0.5,
    "confirm":       true,    // Phase-C single confirmation frame
    "confirm_text":  "channel sounding confirm frame",
    "confirm_secs":  20        // max seconds to wait for listener's params
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

from channel_sound import make_ess

from testlib import (

    CLIENT_ID, OUTPUT_DEV, RUN_DIR, PARAMS, RUN_ID,
    log, to_int_dev, coord_signal, coord_wait_any, coord_wait,
)

TESTS_DIR = Path(__file__).parent

F0           = float(PARAMS.get("f0",            200))
F1_FRAC      = float(PARAMS.get("f1_frac",      0.45))
SWEEP_SECS   = float(PARAMS.get("sweep_secs",    3.0))
AMPLITUDE    = float(PARAMS.get("amplitude",     0.7))
SILENCE_SECS = float(PARAMS.get("silence_secs",  0.5))
CONFIRM      = bool(PARAMS.get("confirm",        True))
CONFIRM_TEXT = str(PARAMS.get("confirm_text",    "channel sounding confirm frame"))
CONFIRM_SECS = float(PARAMS.get("confirm_secs",   20))

def _native_rate(dev) -> int:
    import sounddevice as sd
    try:
        info = sd.query_devices(dev if dev is not None else None, kind="output")
        return int(round(info["default_samplerate"]))
    except Exception:
        return 48000

def confirmation_frame(out_dev, fs: int) -> dict:
    """
    Phase C: read the listener's chosen (carrier, symbol_rate) and transmit a
    single short frame so the listener can confirm the prediction decodes.
    Returns a small status dict; never raises (best-effort).
    """
    import sounddevice as sd
    from qam_writer import Modulator

    log("  [confirm] waiting for listener's probe_best.json ...")
    if not coord_wait("sound_selected", timeout=CONFIRM_SECS):
        log("  [confirm] listener never signalled sound_selected - skipping")
        return {"confirm": "skipped_no_selection"}

    best_path = RUN_DIR / "probe_best.json"
    if not best_path.exists():
        log("  [confirm] probe_best.json missing - skipping")
        return {"confirm": "skipped_no_file"}

    best = json.loads(best_path.read_text())
    cf, sr = int(best["carrier_freq"]), int(best["symbol_rate"])
    log(f"  [confirm] encoding one frame at cf={cf} sr={sr} fs={fs}")

    try:
        mod = Modulator(CONFIRM_TEXT, chunk_bytes=8, ecc_bytes=8, seq_bytes=2,
                        sample_rate=fs, carrier_freq=cf, symbol_rate=sr)
        sig = mod.encode()
        pcm = (sig * mod.scale_factor / 32700.0).astype(np.float32)
        # snapped values (Modulator may adjust an illegal/indivisible sr)
        used_sr = mod.symbol_rate
        cfg_path = RUN_DIR / "confirm_config.json"
        mod.write_config(str(cfg_path))
    except Exception as exc:
        log(f"  [confirm] encode failed: {exc}")
        return {"confirm": "encode_failed", "error": str(exc)}

    silence = np.zeros(int(SILENCE_SECS * fs), dtype=np.float32)
    tx = np.concatenate([silence, pcm, silence])

    coord_signal("confirm_ready")
    log("  [confirm] waiting for listener to arm (confirm_armed) ...")
    coord_wait_any("confirm_armed", timeout=15)

    coord_signal("confirm_start")
    sd.play(tx, samplerate=fs, device=out_dev)
    sd.wait()
    coord_signal("confirm_done")
    log("  [confirm] confirmation frame played")
    return {"confirm": "played", "carrier_freq": cf, "symbol_rate": used_sr,
            "config": "confirm_config.json"}

def main():
    import sounddevice as sd

    out_dev = to_int_dev(OUTPUT_DEV)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    fs = _native_rate(out_dev)
    f1 = F1_FRAC * fs
    log(f"[{CLIENT_ID}] qam_sound_play | output_device={out_dev} native_fs={fs}")
    log(f"  ESS {SWEEP_SECS:.1f}s  {F0:.0f}-{f1:.0f} Hz  amp={AMPLITUDE}")

    sweep, _inv = make_ess(fs, F0, f1, SWEEP_SECS, AMPLITUDE)
    silence = np.zeros(int(SILENCE_SECS * fs), dtype=np.float32)
    tx = np.concatenate([silence, sweep, silence])

    # Manifest carries physical sweep params; listener uses its OWN native fs.
    manifest = {
        "run_id":       RUN_ID,
        "player":       CLIENT_ID,
        "f0":           F0,
        "f1":           f1,
        "f1_frac":      F1_FRAC,
        "dur":          SWEEP_SECS,
        "amp":          AMPLITUDE,
        "silence_secs": SILENCE_SECS,
        "tx_fs":        fs,
        "confirm":      CONFIRM,
    }
    (RUN_DIR / "sound_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Optional: dump the exact transmitted signal for offline replay/debug.
    try:
        from channel_sound import _write_wav
        _write_wav(str(RUN_DIR / "probe_sweep_tx.wav"), fs, tx)
    except Exception:
        pass

    coord_signal("sound_ready")
    log("  manifest written; waiting for listener to arm (sound_armed) ...")
    if not coord_wait_any("sound_armed", timeout=30):
        log("  no listener armed within 30s - playing anyway")

    coord_signal("sound_start")
    log(f"  playing sweep ({len(tx)/fs:.1f}s) ...")
    sd.play(tx, samplerate=fs, device=out_dev)
    sd.wait()
    coord_signal("sound_done")
    log("  sweep played")

    confirm_result = {"confirm": "disabled"}
    if CONFIRM:
        confirm_result = confirmation_frame(out_dev, fs)

    result = {
        "client_id": CLIENT_ID, "role": "play", "run_id": RUN_ID,
        "status": "ok", "native_fs": fs,
        "sweep_secs": SWEEP_SECS, **confirm_result,
    }
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
