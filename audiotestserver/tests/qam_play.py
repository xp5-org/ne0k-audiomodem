"""
tests/qam_play.py  -  role: play  (pair with qam_decode.py once available)

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

carrier_freq and symbol_rate should be chosen based on specprobe results.
Default 12000/6000 works well for typical laptop speakers/mics in a quiet room.
"""

import json
import os
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


def main():
    import subprocess

    out_dev = to_int_dev(OUTPUT_DEV)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[{CLIENT_ID}] qam_play | output_device={out_dev}")

    wav_path    = RUN_DIR / "qam_output.wav"
    config_path = RUN_DIR / "qam_config.json"

    # Build encode config from params
    encode_cfg = {
        "input_str":    PARAMS.get("input_str"),   # None = use writer's default
        "output":       str(wav_path),
        "config_out":   str(config_path),
        "chunk_bytes":  PARAMS.get("chunk_bytes",  64),
        "ecc_bytes":    PARAMS.get("ecc_bytes",    8),
        "seq_bytes":    PARAMS.get("seq_bytes",    2),
        "sample_rate":  PARAMS.get("sample_rate",  48000),
        "carrier_freq": PARAMS.get("carrier_freq", 4000),
        "symbol_rate":  PARAMS.get("symbol_rate",  80),
    }
    # Remove None so writer uses its own default message
    encode_cfg = {k: v for k, v in encode_cfg.items() if v is not None}

    cfg_file = RUN_DIR / "qam_encode_config.json"
    cfg_file.write_text(json.dumps(encode_cfg, indent=2))

    # Run qam_writer.py as a subprocess (same pattern as css_play -> css_encoder)
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

    # Signal ready (config + wav are now on disk)
    coord_signal("qam_ready")
    log("  signalled qam_ready - waiting for decoder(s) to arm...")
    armed = coord_wait_any("qam_armed", timeout=30)
    if armed:
        log(f"  decoder(s) armed: {armed}")
    else:
        log("  no decoders armed in 30s - playing anyway")

    # Play
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