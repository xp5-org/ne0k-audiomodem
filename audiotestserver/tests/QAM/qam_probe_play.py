# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
tests/qam_probe_play.py - role: play (pair with qam_probe_listen.py)

Sweeps over carrier_freq × symbol_rate combinations, encoding and playing
a short test sequence for each. The listener decodes each and records
per-step SNR / ECC metrics to find the best-performing settings.

Params:
{
    "carrier_start":        800,
    "carrier_step":         400,
    "carrier_num_steps":      3,
    "symbol_rate_start":     40,
    "symbol_rate_step":      40,
    "symbol_rate_num_steps":  6,
    "sample_rate":         48000,
    "chunk_bytes":             8,   // 0 = auto-scale
    "ecc_bytes":               8,
    "seq_bytes":               2,
    "amplitude":             0.8
}
"""

import json
import struct
import subprocess
import sys
import time
import wave
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

from testlib import (

    CLIENT_ID, OUTPUT_DEV, RUN_DIR, PARAMS, RUN_ID, SERVER_URL,
    log, to_int_dev, coord_signal, coord_check, coord_wait_any,
)

TESTS_DIR = Path(__file__).parent

CARRIER_START     = int(PARAMS.get("carrier_start",       800))
CARRIER_STEP      = int(PARAMS.get("carrier_step",        400))
CARRIER_NUM_STEPS = int(PARAMS.get("carrier_num_steps",     3))
SR_START          = int(PARAMS.get("symbol_rate_start",    40))
SR_STEP           = int(PARAMS.get("symbol_rate_step",     40))
SR_NUM_STEPS      = int(PARAMS.get("symbol_rate_num_steps", 6))
SAMPLE_RATE       = int(PARAMS.get("sample_rate",       48000))
CHUNK_BYTES       = int(PARAMS.get("chunk_bytes",            8))   # 0 = auto-scale
ECC_BYTES         = int(PARAMS.get("ecc_bytes",              8))
SEQ_BYTES         = int(PARAMS.get("seq_bytes",              2))
AMPLITUDE         = float(PARAMS.get("amplitude",          0.8))
HEAD_SILENCE_SECS = float(PARAMS.get("head_silence_secs",  2.0))
TARGET_BLOCK_SECS = float(PARAMS.get("target_block_secs",  5.0))

PROBE_MESSAGE = (
    "The quick brown fox jumps over the lazy dog " * 12
)

def auto_chunk_bytes(sample_rate, symbol_rate, ecc_bytes, seq_bytes,
                     target_block_secs, pilot_interval=8, zc_length=63):
    import math
    sps = max(1, sample_rate // symbol_rate)
    max_frame_syms = target_block_secs * sample_rate / sps / 4
    headroom = max_frame_syms - zc_length
    if headroom <= 0:
        return 1
    max_raw   = headroom / (1.0 + 1.0 / pilot_interval)
    max_chunk = int(max_raw / 2) - seq_bytes - ecc_bytes
    return max(1, min(max_chunk, 512))

def snap_symbol_rate(sample_rate, symbol_rate):
    if sample_rate % symbol_rate == 0:
        return symbol_rate
    best = 1
    for d in range(1, symbol_rate + 1):
        if sample_rate % d == 0:
            best = d
    return best

_MIN_SR = 50

def build_combos():
    carriers     = [CARRIER_START + i * CARRIER_STEP for i in range(CARRIER_NUM_STEPS)]
    symbol_rates = [SR_START      + i * SR_STEP      for i in range(SR_NUM_STEPS)]
    combos = []
    for sr in symbol_rates:
        snapped = snap_symbol_rate(SAMPLE_RATE, sr)
        if snapped < _MIN_SR:
            log(f"  skipping symbol_rate={snapped} (< {_MIN_SR} min)")
            continue
        for cf in carriers:
            combos.append({"carrier_freq": cf, "symbol_rate": snapped,
                           "symbol_rate_requested": sr})
    return combos

def encode_step(step_idx, carrier_freq, symbol_rate, wav_path, config_path):
    chunk = (CHUNK_BYTES if CHUNK_BYTES > 0
             else auto_chunk_bytes(SAMPLE_RATE, symbol_rate, ECC_BYTES, SEQ_BYTES,
                                   TARGET_BLOCK_SECS))
    encode_cfg = {
        "input_str":    PROBE_MESSAGE,
        "output":       str(wav_path),
        "config_out":   str(config_path),
        "chunk_bytes":  chunk,
        "ecc_bytes":    ECC_BYTES,
        "seq_bytes":    SEQ_BYTES,
        "sample_rate":  SAMPLE_RATE,
        "carrier_freq": carrier_freq,
        "symbol_rate":  symbol_rate,
        "amplitude":    AMPLITUDE,
    }
    cfg_file = RUN_DIR / f"probe_encode_cfg_{step_idx}.json"
    cfg_file.write_text(json.dumps(encode_cfg, indent=2))

    proc = subprocess.Popen(
        [sys.executable, str(TESTS_DIR / "qam_writer.py"), str(cfg_file)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    result_json = None
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("{"):
            result_json = line
    proc.wait()

    if proc.returncode != 0:
        return None
    try:
        result = json.loads(result_json) if result_json else None
    except Exception:
        return None

    # Prepend silence so the decoder has time to stabilise before the ZC
    if result and HEAD_SILENCE_SECS > 0:
        try:
            with wave.open(str(wav_path), "rb") as wf:
                fs      = wf.getframerate()
                sw      = wf.getsampwidth()
                ch      = wf.getnchannels()
                pcm_raw = wf.readframes(wf.getnframes())
            n_silence = int(HEAD_SILENCE_SECS * fs) * ch
            silence   = struct.pack(f"<{n_silence}h", *([0] * n_silence))
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(ch)
                wf.setsampwidth(sw)
                wf.setframerate(fs)
                wf.writeframes(silence + pcm_raw)
            result["duration_s"] = round(result.get("duration_s", 0) + HEAD_SILENCE_SECS, 2)
        except Exception:
            pass

    return result

def main():
    import sounddevice as sd
    import soundfile as sf

    out_dev = to_int_dev(OUTPUT_DEV)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    combos = build_combos()
    log(f"[{CLIENT_ID}] qam_probe_play | {len(combos)} steps  output_device={out_dev}")
    log(f"  carrier: {CARRIER_START}Hz + {CARRIER_NUM_STEPS} x {CARRIER_STEP}Hz")
    log(f"  symbol_rate: {SR_START} + {SR_NUM_STEPS} x {SR_STEP} baud")

    log(f"  pre-encoding {len(combos)} WAVs...")
    steps = []
    for i, combo in enumerate(combos):
        cf  = combo["carrier_freq"]
        sr  = combo["symbol_rate"]
        wav = RUN_DIR / f"probe_{i:03d}_cf{cf}_sr{sr}.wav"
        cfg = RUN_DIR / f"probe_{i:03d}_config.json"
        result = encode_step(i, cf, sr, wav, cfg)
        if result:
            dur = result.get("duration_s", 0)
            used_chunk = (auto_chunk_bytes(SAMPLE_RATE, sr, ECC_BYTES, SEQ_BYTES,
                                           TARGET_BLOCK_SECS)
                          if CHUNK_BYTES == 0 else CHUNK_BYTES)
            log(f"  [{i+1}/{len(combos)}] cf={cf}Hz sr={sr}baud {dur:.1f}s chunk={used_chunk}B")
            steps.append({**combo, "step": i,
                          "wav":    wav.name,
                          "config": cfg.name,
                          "duration_s": dur,
                          "frames": result.get("total_frames", 0)})
        else:
            log(f"  [{i+1}/{len(combos)}] cf={cf}Hz sr={sr}baud ENCODE FAILED - skipping")

    if not steps:
        log("  ERROR: all steps failed to encode")
        print(json.dumps({"client_id": CLIENT_ID, "status": "error",
                          "run_id": RUN_ID, "error": "all encode failed"}), flush=True)
        return

    manifest = {"run_id": RUN_ID, "player": CLIENT_ID, "steps": steps}
    (RUN_DIR / "probe_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"  manifest written ({len(steps)} steps)")

    coord_signal("probe_ready")
    log("  waiting for listener to arm...")
    armed = coord_wait_any("probe_armed", timeout=30)
    if armed:
        log(f"  listener armed: {armed}")
    else:
        log("  no listener armed within 30s - playing anyway")

    for s in steps:
        i   = s["step"]
        cf  = s["carrier_freq"]
        sr  = s["symbol_rate"]
        dur = s["duration_s"]

        log(f"  step {i+1}/{len(steps)}: cf={cf}Hz sr={sr}baud - waiting for listener mic...")
        deadline = time.time() + 30
        while time.time() < deadline:
            if coord_check(f"probe_step_{i}_listen_ready"):
                break
            time.sleep(0.2)
        else:
            log(f"  step {i+1}: no listen_ready within 30s - playing anyway")

        coord_signal(f"probe_step_{i}_play")
        log(f"  step {i+1}/{len(steps)}: cf={cf}Hz sr={sr}baud {dur:.1f}s - playing")

        audio, fs_wav = sf.read(str(RUN_DIR / s["wav"]), dtype="float32")
        sd.play(audio, samplerate=fs_wav, device=out_dev)
        sd.wait()

        coord_signal(f"probe_step_{i}_played")
        log(f"  step {i+1}: done")

        time.sleep(0.3)

    coord_signal("probe_all_played")
    log("  all steps complete")

    result = {
        "client_id":    CLIENT_ID, "role": "play",
        "run_id":       RUN_ID,    "status": "ok",
        "steps_played": len(steps),
    }
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
