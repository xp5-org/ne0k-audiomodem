# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
tests/qam_play.py - role: play (pair with qam_decode.py)

Framework wrapper for qam_writer.Modulator.
Encodes the message to a WAV, writes qam_config.json to the run dir,
coordinates with the decoder via the server API, then plays the WAV.

Params (from server UI JSON field):
{
    "input_str":    "Hello world",
    "chunk_bytes":  512,
    "ecc_bytes":    8,
    "seq_bytes":    2,
    "sample_rate":  48000,
    "carrier_freq": 12000,
    "symbol_rate":  6000
}
"""

import json
import subprocess
import sys
from pathlib import Path

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
    log, to_int_dev, coord_signal, coord_wait_any,
)

TESTS_DIR = Path(__file__).parent

def main():
    import subprocess as _subprocess

    out_dev = to_int_dev(OUTPUT_DEV)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[{CLIENT_ID}] qam_play | output_device={out_dev}")

    wav_path    = RUN_DIR / "qam_output.wav"
    config_path = RUN_DIR / "qam_config.json"

    encode_cfg = {
        "input_str":    PARAMS.get("input_str"),
        "output":       str(wav_path),
        "config_out":   str(config_path),
        "chunk_bytes":  PARAMS.get("chunk_bytes",  16),
        "ecc_bytes":    PARAMS.get("ecc_bytes",    8),
        "seq_bytes":    PARAMS.get("seq_bytes",    2),
        "sample_rate":  PARAMS.get("sample_rate",  48000),
        "carrier_freq": PARAMS.get("carrier_freq", 1400),
        "symbol_rate":  PARAMS.get("symbol_rate",  40),
    }
    encode_cfg = {k: v for k, v in encode_cfg.items() if v is not None}

    cfg_file = RUN_DIR / "qam_encode_config.json"
    cfg_file.write_text(json.dumps(encode_cfg, indent=2))

    writer_script = TESTS_DIR / "qam_writer.py"
    log("  encoding QAM audio...")
    proc = subprocess.Popen(
        [sys.executable, str(writer_script), str(cfg_file)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    encode_result_json = None
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("{"):
            encode_result_json = line
        else:
            log(f"  [encode] {line}")
    proc.wait()

    if proc.returncode != 0:
        log(f"  qam_writer.py failed (rc={proc.returncode})")
        print(json.dumps({"client_id": CLIENT_ID, "status": "error",
                          "error": "encode failed", "run_id": RUN_ID}), flush=True)
        return

    try:
        enc_result = json.loads(encode_result_json) if encode_result_json else {}
    except Exception:
        enc_result = {}

    log(f"  encoded: {enc_result.get('total_frames','?')} frames, "
        f"{enc_result.get('duration_s','?')}s")

    coord_signal("qam_ready")
    log("  signalled qam_ready - waiting for decoder(s) to arm...")
    armed = coord_wait_any("qam_armed", timeout=30)
    if armed:
        log(f"  decoder(s) armed: {armed}")
    else:
        log("  no decoders armed in 30s - playing anyway")

    result = {
        "client_id": CLIENT_ID, "role": "play", "run_id": RUN_ID,
        **enc_result,
    }
    try:
        import sounddevice as sd
        import soundfile as sf

        audio, sr = sf.read(str(wav_path), dtype="float32")
        log(f"  playing {len(audio)/sr:.1f}s on device {out_dev}...")
        sd.play(audio, samplerate=sr, device=out_dev)
        sd.wait()
        log("  playback complete")
        result["playback"] = "ok"
        result["status"]   = "ok"

    except Exception as exc:
        log(f"  playback error: {exc}")
        result["playback"] = str(exc)
        result["status"]   = "error"

    coord_signal("qam_played")

    out = RUN_DIR / f"result_{CLIENT_ID}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
