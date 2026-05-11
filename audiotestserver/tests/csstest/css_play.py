"""
css_play.py  -  Framework wrapper for css_encoder.py (role: play).
"""

import json
import subprocess
import sys
from pathlib import Path

BARKER_13 = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]

from testlib import (
    CLIENT_ID, OUTPUT_DEV, RUN_DIR, PARAMS, RUN_ID, SERVER_URL,
    log, to_int_dev, coord_signal, coord_wait_any,
)

TESTS_DIR = Path(__file__).parent


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Build config for css_encoder.py
    encode_cfg = {
        "message":    PARAMS.get("message"),
        "output":     str(RUN_DIR / "css_output.wav"),
        "info_out":   str(RUN_DIR / "css_info.json"),
        "result_out": str(RUN_DIR / f"encode_result_{CLIENT_ID}.json"),
        "fs":         PARAMS.get("fs",          44100),
        "f_min":      PARAMS.get("f_min",        400),
        "f_max":      PARAMS.get("f_max",       6000),
        "symbol_dur": PARAMS.get("symbol_dur",   0.25),
        "chunk_size": PARAMS.get("chunk_size",    12),
        "amplitude":  PARAMS.get("amplitude",    0.8),
        "preamble":   PARAMS.get("preamble", BARKER_13),
    }
    # Remove None values so css_encoder.py uses its own defaults
    encode_cfg = {k: v for k, v in encode_cfg.items() if v is not None}

    cfg_path = RUN_DIR / "css_encode_config.json"
    cfg_path.write_text(json.dumps(encode_cfg, indent=2))

    log(f"[{CLIENT_ID}] css_play: running css_encoder.py...")
    encode_script = TESTS_DIR / "css_encoder.py"
    proc = subprocess.Popen(
        [sys.executable, str(encode_script), str(cfg_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    encode_output = []
    for line in proc.stdout:
        line = line.rstrip()
        encode_output.append(line)
        log(f"  [encode] {line}")
    proc.wait()

    if proc.returncode != 0:
        log(f"  css_encoder.py failed (rc={proc.returncode})")
        print(json.dumps({"client_id": CLIENT_ID, "status": "error",
                          "error": "encode failed", "run_id": RUN_ID}), flush=True)
        return

    # Signal ready, wait for decoders
    coord_signal("css_ready")
    log(f"  signalled css_ready - waiting for decoders to arm...")
    armed = coord_wait_any("css_armed", timeout=30)
    if armed:
        log(f"  decoder(s) armed: {armed}")
    else:
        log(f"  no decoders armed in 30 s - playing anyway")

    # Play the generated wav
    try:
        import sounddevice as sd
        import soundfile as sf

        wav_path = RUN_DIR / "css_output.wav"
        audio, sr = sf.read(str(wav_path), dtype="float32")
        out_dev   = to_int_dev(OUTPUT_DEV)

        log(f"  playing {len(audio)/sr:.1f}s on device {out_dev}...")
        sd.play(audio, samplerate=sr, device=out_dev)
        sd.wait()
        log(f"  playback complete")
        play_status = "ok"
    except Exception as exc:
        log(f"  playback error: {exc}")
        play_status = str(exc)

    coord_signal("css_played")

    result_path = RUN_DIR / f"encode_result_{CLIENT_ID}.json"
    try:
        result = json.loads(result_path.read_text())
    except Exception:
        result = {}
    result.update({"client_id": CLIENT_ID, "role": "play",
                   "run_id": RUN_ID, "playback": play_status})
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()