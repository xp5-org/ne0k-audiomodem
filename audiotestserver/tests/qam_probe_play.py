"""
tests/qam_probe_play.py  -  role: play  (pair with qam_probe_listen.py)

Sweeps over carrier_freq × symbol_rate combinations, encoding and playing
a short test sequence for each. The listener decodes each and records
per-step SNR / ECC metrics to find the best-performing settings.

Params:
{
    "carrier_start":        3000,
    "carrier_step":          500,
    "carrier_num_steps":       8,
    "symbol_rate_start":     100,
    "symbol_rate_step":       50,
    "symbol_rate_num_steps":   1,
    "sample_rate":          48000,
    "chunk_bytes":            128,
    "ecc_bytes":               8,
    "seq_bytes":               2,
    "amplitude":             0.8
}

Setting symbol_rate_num_steps=1 sweeps only carrier (8 tests).
Setting both >1 does a full grid (carrier_num_steps x symbol_rate_num_steps tests).

The test message is kept short (~200 bytes = 2 frames) so each step
takes only a few seconds even at low baud rates.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

CLIENT_ID  = os.environ.get("AUDIOTOOL_CLIENT_ID", "unknown")
OUTPUT_DEV = os.environ.get("AUDIOTOOL_OUTPUT_DEVICE", "")
RUN_DIR    = Path(os.environ.get("AUDIOTOOL_RUN_DIR", "."))
SERVER_URL = os.environ.get("AUDIOTOOL_SERVER_URL", "http://127.0.0.1:8000")
PARAMS     = json.loads(os.environ.get("AUDIOTOOL_PARAMS", "{}"))
RUN_ID     = os.environ.get("AUDIOTOOL_RUN_ID", "unknown")
TESTS_DIR  = Path(__file__).parent

# Sweep config
CARRIER_START        = int(PARAMS.get("carrier_start",        800))
CARRIER_STEP         = int(PARAMS.get("carrier_step",          400))
CARRIER_NUM_STEPS    = int(PARAMS.get("carrier_num_steps",       3))
SR_START             = int(PARAMS.get("symbol_rate_start",     40))
SR_STEP              = int(PARAMS.get("symbol_rate_step",       40))
SR_NUM_STEPS         = int(PARAMS.get("symbol_rate_num_steps",   6))
SAMPLE_RATE          = int(PARAMS.get("sample_rate",         48000))
CHUNK_BYTES          = int(PARAMS.get("chunk_bytes",            8))  # 0 = auto-scale
ECC_BYTES            = int(PARAMS.get("ecc_bytes",               8))
SEQ_BYTES            = int(PARAMS.get("seq_bytes",               2))
AMPLITUDE            = float(PARAMS.get("amplitude",           0.8))
HEAD_SILENCE_SECS    = float(PARAMS.get("head_silence_secs",   2.0))
TARGET_BLOCK_SECS    = float(PARAMS.get("target_block_secs",   5.0))  # keep reader block <= this

# Short message — enough for 2-3 frames without taking too long
PROBE_MESSAGE = (
    "The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog "
)


def log(msg):
    print(msg, flush=True)


def to_int_dev(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def coord_signal(signal):
    import urllib.request
    url  = f"{SERVER_URL}/api/coord/{RUN_ID}/{signal}"
    body = json.dumps({"client_id": CLIENT_ID}).encode()
    req  = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        log(f"  [coord] {signal} failed: {e}")


def coord_check(signal):
    import urllib.request
    try:
        url = f"{SERVER_URL}/api/coord/{RUN_ID}/{signal}"
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read()).get("clients", [])
    except Exception:
        return []


def coord_wait_any(signal, timeout=30, poll=0.3):
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            url = f"{SERVER_URL}/api/coord/{RUN_ID}/{signal}"
            with urllib.request.urlopen(url, timeout=5) as r:
                clients = json.loads(r.read()).get("clients", [])
            if clients:
                return clients
        except Exception:
            pass
        time.sleep(poll)
    return []


def auto_chunk_bytes(sample_rate: int, symbol_rate: int,
                     ecc_bytes: int, seq_bytes: int,
                     target_block_secs: float,
                     pilot_interval: int = 8,
                     zc_length: int = 63) -> int:
    """
    Calculate largest chunk_bytes such that the reader BLOCK_SAMPLES
    (4 × frame_samples) stays under target_block_secs.
    Reader BLOCK = frame_total_syms * sps * 4.
    """
    import math
    sps = max(1, sample_rate // symbol_rate)
    # frame_total_syms = zc_length + raw_data_syms + n_pilots
    # raw_data_syms = (seq+chunk+ecc)*2,  n_pilots = ceil(raw/pilot_interval)
    # Solve for max raw_data_syms:
    # (zc + raw*(1 + 1/pi)) * sps * 4 <= target * sample_rate
    max_frame_syms = target_block_secs * sample_rate / sps / 4
    # raw*(1 + 1/pi) <= max_frame_syms - zc
    headroom = max_frame_syms - zc_length
    if headroom <= 0:
        return 1
    max_raw = headroom / (1.0 + 1.0 / pilot_interval)
    # raw_data_syms = (seq+chunk+ecc)*2
    max_chunk = int(max_raw / 2) - seq_bytes - ecc_bytes
    return max(1, min(max_chunk, 512))


def snap_symbol_rate(sample_rate: int, symbol_rate: int) -> int:
    """Return largest divisor of sample_rate that is <= symbol_rate."""
    if sample_rate % symbol_rate == 0:
        return symbol_rate
    best = 1
    for d in range(1, symbol_rate + 1):
        if sample_rate % d == 0:
            best = d
    return best


# Minimum symbol rate below which a single frame overflows the reader block
# even at chunk_bytes=1.  At sr=50 block≈7s, sr=100 block≈5s.
_MIN_SR = 50

def build_combos():
    """Build list of (carrier_freq, symbol_rate) tuples to probe."""
    carriers     = [CARRIER_START + i * CARRIER_STEP for i in range(CARRIER_NUM_STEPS)]
    symbol_rates = [SR_START      + i * SR_STEP      for i in range(SR_NUM_STEPS)]
    combos = []
    for sr in symbol_rates:
        snapped = snap_symbol_rate(SAMPLE_RATE, sr)
        if snapped < _MIN_SR:
            log(f"  skipping symbol_rate={snapped} (< {_MIN_SR} min — frame too large for live decode)")
            continue
        for cf in carriers:
            combos.append({"carrier_freq": cf, "symbol_rate": snapped,
                           "symbol_rate_requested": sr})
    return combos


def encode_step(step_idx: int, carrier_freq: int, symbol_rate: int,
                wav_path: Path, config_path: Path) -> dict | None:
    """Encode one probe step. Returns encode result dict or None on failure."""
    # Auto-scale chunk_bytes so reader block fits in TARGET_BLOCK_SECS
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

    if result and HEAD_SILENCE_SECS > 0:
        # Prepend silence so the decoder subprocess has time to open the mic
        # and stabilise before the ZC preamble arrives.
        try:
            import wave, struct
            import numpy as np
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
        except Exception as e:
            pass  # silence prepend failed, continue without it

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

    # Pre-encode all steps so listener doesn't wait during encoding
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
            used_chunk = auto_chunk_bytes(SAMPLE_RATE, sr, ECC_BYTES, SEQ_BYTES,
                                          TARGET_BLOCK_SECS) if CHUNK_BYTES == 0 else CHUNK_BYTES
            log(f"  [{i+1}/{len(combos)}] cf={cf}Hz sr={sr}baud {dur:.1f}s chunk={used_chunk}B")
            steps.append({**combo, "step": i,
                          "wav": wav.name,       # filename only - each machine uses own RUN_DIR
                          "config": cfg.name,    # filename only
                          "duration_s": dur,
                          "frames": result.get("total_frames", 0)})
        else:
            log(f"  [{i+1}/{len(combos)}] cf={cf}Hz sr={sr}baud ENCODE FAILED - skipping")

    if not steps:
        log("  ERROR: all steps failed to encode")
        print(json.dumps({"client_id": CLIENT_ID, "status": "error",
                          "run_id": RUN_ID, "error": "all encode failed"}), flush=True)
        return

    # Write manifest for listener
    manifest = {"run_id": RUN_ID, "player": CLIENT_ID, "steps": steps}
    (RUN_DIR / "probe_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"  manifest written ({len(steps)} steps)")

    # Signal ready and wait for listener to arm
    coord_signal("probe_ready")
    log("  waiting for listener to arm...")
    armed = coord_wait_any("probe_armed", timeout=30)
    if armed:
        log(f"  listener armed: {armed}")
    else:
        log("  no listener armed within 30s - playing anyway")

    # Play each step in sequence
    for s in steps:
        i   = s["step"]
        cf  = s["carrier_freq"]
        sr  = s["symbol_rate"]
        dur = s["duration_s"]

        # Wait for listener to open mic before playing
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

        # Short gap between steps
        time.sleep(0.3)

    coord_signal("probe_all_played")
    log("  all steps complete")

    result = {
        "client_id":  CLIENT_ID, "role": "play",
        "run_id":     RUN_ID,    "status": "ok",
        "steps_played": len(steps),
    }
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()