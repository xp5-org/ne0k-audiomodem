"""
css_play.py  -  Framework wrapper for css_encoder.py (role: play).

All encoder parameters are exposed via AUDIOTOOL_PARAMS so they appear
in the test UI.  The param scraper discovers them via PARAMS.get() calls.

Params
------
message        : str   - text to encode (required)
fs             : int   - sample rate Hz           (default 44100)
f_min          : float - chirp low  frequency Hz  (default 400)
f_max          : float - chirp high frequency Hz  (default 6000)
symbol_dur     : float - seconds per symbol       (default 0.25)
chunk_size     : int   - payload bytes per frame  (default 12)
amplitude      : float - output amplitude 0-1     (default 0.8)
rs_t           : int   - Reed-Solomon error-correction symbols (default 0)
conv_code      : str   - convolutional code: None or "k7r12"   (default None)
interleave_rows: int   - block interleaver rows; 0 = off       (default 0)
diff_encode    : bool  - differential encoding                  (default False)
preamble       : list  - sync preamble bits (default Barker-13)
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
PARAMS     = json.loads(os.environ.get("AUDIOTOOL_PARAMS", "{}"))
RUN_ID     = os.environ.get("AUDIOTOOL_RUN_ID", "unknown")
SERVER_URL = os.environ.get("AUDIOTOOL_SERVER_URL", "http://127.0.0.1:8000")
TESTS_DIR  = Path(__file__).parent

BARKER_13 = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]


def log(msg):
    print(msg, flush=True)


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
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect all encoder parameters from the UI ────────────────────────────
    # Every PARAMS.get() here is discovered by the param scraper and shown in
    # the test UI.  Add new encoder knobs here to make them configurable.

    message         = PARAMS.get("message",         "")
    fs              = PARAMS.get("fs",              44100)
    f_min           = PARAMS.get("f_min",             400)
    f_max           = PARAMS.get("f_max",            6000)
    symbol_dur      = PARAMS.get("symbol_dur",       0.25)
    chunk_size      = PARAMS.get("chunk_size",          12)
    amplitude       = PARAMS.get("amplitude",          0.8)
    rs_t            = PARAMS.get("rs_t",                 0)
    conv_code       = PARAMS.get("conv_code",           "")   # "" = off, "k7r12" to enable
    interleave_rows = PARAMS.get("interleave_rows",      0)
    diff_encode     = PARAMS.get("diff_encode",      False)
    preamble        = PARAMS.get("preamble", [1,1,1,1,1,0,0,1,1,0,1,0,1])

    encode_cfg = {
        "output":          str(RUN_DIR / "css_output.wav"),
        "info_out":        str(RUN_DIR / "css_info.json"),
        "result_out":      str(RUN_DIR / f"encode_result_{CLIENT_ID}.json"),
        "fs":              fs,
        "f_min":           f_min,
        "f_max":           f_max,
        "symbol_dur":      symbol_dur,
        "chunk_size":      chunk_size,
        "amplitude":       amplitude,
        "rs_t":            rs_t,
        "interleave_rows": interleave_rows,
        "diff_encode":     diff_encode,
        "preamble":        preamble,
    }
    # Only pass message if the user set one
    if message:
        encode_cfg["message"] = message
    # Map empty / "none" / "off" to omitted so encoder uses its None default
    if conv_code and conv_code.strip().lower() not in ("", "none", "off"):
        encode_cfg["conv_code"] = conv_code

    cfg_path = RUN_DIR / "css_encode_config.json"
    cfg_path.write_text(json.dumps(encode_cfg, indent=2))

    log(f"[{CLIENT_ID}] css_play: running css_encoder.py...")
    log(f"  chunk_size={chunk_size}  symbol_dur={symbol_dur}s  "
        f"rs_t={rs_t}  conv_code={conv_code}  "
        f"interleave_rows={interleave_rows}  diff_encode={diff_encode}")

    encode_script = TESTS_DIR / "css_encoder.py"
    proc = subprocess.Popen(
        [sys.executable, str(encode_script), str(cfg_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in proc.stdout:
        log(f"  [encode] {line.rstrip()}")
    proc.wait()

    if proc.returncode != 0:
        log(f"  css_encoder.py failed (rc={proc.returncode})")
        print(json.dumps({"client_id": CLIENT_ID, "status": "error",
                          "error": "encode failed", "run_id": RUN_ID}), flush=True)
        return

    # Signal ready, wait for decoder(s)
    coord_signal("css_ready")
    log("  signalled css_ready - waiting for decoders to arm...")
    armed = coord_wait_any("css_armed", timeout=30)
    log(f"  decoder(s) armed: {armed}" if armed else "  no decoders armed in 30s - playing anyway")

    # Play
    try:
        import sounddevice as sd
        import soundfile as sf

        wav_path = RUN_DIR / "css_output.wav"
        audio, sr = sf.read(str(wav_path), dtype="float32")
        try:
            out_dev = int(OUTPUT_DEV)
        except (TypeError, ValueError):
            out_dev = None

        log(f"  playing {len(audio)/sr:.1f}s on device {out_dev}...")
        sd.play(audio, samplerate=sr, device=out_dev)
        sd.wait()
        log("  playback complete")
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