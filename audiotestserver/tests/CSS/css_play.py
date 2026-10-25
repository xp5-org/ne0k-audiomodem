# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_play.py  -  Framework wrapper for css_encoder.py (role: play).

All encoder parameters are exposed via AUDIOTOOL_PARAMS so they appear
in the test UI.  The param scraper discovers them via PARAMS.get() calls.

Params
------
message         : str   - text to encode (required)
                  (sample rate is auto-detected from the selected output
                   device's native rate — it is NOT a user param.)
f_min           : float - chirp low  frequency Hz               (default 400)
f_max           : float - chirp high frequency Hz               (default 6000)
symbol_dur      : float - seconds per symbol                    (default 0.25)
chunk_size      : int   - payload bytes per frame               (default 12)
amplitude       : float - output amplitude 0-1                  (default 0.8)
rs_t            : int   - Reed-Solomon error-correction symbols (default 0)
conv_code       : str   - convolutional code: None or "k7r12"   (default None)
interleave_rows : int   - block interleaver rows; 0 = off       (default 0)
diff_encode     : bool  - differential encoding                 (default False)
preamble        : list  - sync preamble bits                    (default Barker-13)
training_symbols: int   - alternating chirps before preamble for PLL lock (default 16)
max_frame_s     : float - max frame duration guard; auto-sizes chunk_size (default 30.0)
symbol_gap_frac : float - inter-symbol silence as fraction of symbol_dur (default 0.05)
                          Eliminates DAC click/pop between chirps.
                          gap_dur_ms = symbol_dur * symbol_gap_frac * 1000
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
# Reach repo-root shared.py (audio-interface probing) from tests/.

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


def _load_play_schedule(run_dir, sr, audio):
    """Best-effort (first_chirp_frame, sps_eff, n_symbols) for per-chirp emit
    logging.  first_chirp_frame is the first non-silent sample (chirp onset);
    sps_eff = chirp + inter-symbol gap from css_info.json.  (0, None, 0) on error."""
    try:
        import numpy as np
        info = json.loads((run_dir / "css_info.json").read_text())
        fs   = int(info.get("fs", sr))
        sps  = int(round(float(info["symbol_dur"]) * fs))
        gap  = int(sps * float(info.get("symbol_gap_frac", 0.0)))
        a    = audio if audio.ndim == 1 else audio[:, 0]
        peak = float(np.max(np.abs(a))) or 1.0
        nz   = np.where(np.abs(a) > 0.02 * peak)[0]
        start = int(nz[0]) if len(nz) else 0
        return start, sps + gap, int(info.get("symbols_total", 0))
    except Exception:
        return 0, None, 0


def play_with_chirp_logging(audio, sr, out_dev, run_dir):
    """Play `audio` through a callback OutputStream, logging the real output
    interface, each chirp's emit frame + wall-clock time (for cross-host
    alignment via app.py's clock offset), and any output underruns (which slip
    emit timing).  Raises on any failure so the caller can fall back to sd.play."""
    import sounddevice as sd
    import threading, queue

    a2d = audio.reshape(-1, 1) if audio.ndim == 1 else audio
    nframes, nch = a2d.shape
    start, sps_eff, _ = _load_play_schedule(run_dir, sr, audio)
    bounds = list(range(start, nframes, sps_eff)) if sps_eff else []
    stride = max(1, len(bounds) // 300)         # cap log volume on short symbols

    evq   = queue.Queue()
    state = {"pos": 0, "bidx": 0, "xruns": 0}

    def callback(outdata, frames, time_info, status):
        # Audio thread: keep cheap — only array copies and queue puts.
        if status:
            state["xruns"] += 1
            evq.put(("xrun", time.time(), state["pos"], str(status)))
        pos = state["pos"]; end = pos + frames
        chunk = a2d[pos:end]; n = len(chunk)
        outdata[:n] = chunk
        if n < frames:
            outdata[n:] = 0
        wall = time.time()
        dac0 = float(getattr(time_info, "outputBufferDacTime", 0.0))
        bidx = state["bidx"]
        while bidx < len(bounds) and bounds[bidx] < end:
            bf = bounds[bidx]
            if bf >= pos:
                evq.put(("sym", wall, bf, dac0 + (bf - pos) / sr, bidx))
            bidx += 1
        state["bidx"] = bidx
        state["pos"]  = end
        if n < frames:
            evq.put(("done", time.time(), end, 0.0, -1))
            raise sd.CallbackStop

    done = threading.Event()

    def monitor():
        while True:
            try:
                ev = evq.get(timeout=0.5)
            except queue.Empty:
                if done.is_set():
                    break
                continue
            kind = ev[0]
            if kind == "xrun":
                log(f"  [play] OUTPUT_XRUN #{state['xruns']} frame={ev[2]} "
                    f"({ev[2]/sr:.2f}s) status={ev[3]} — DAC starved, emit timing slipped")
            elif kind == "sym":
                _, wall, bf, dac, k = ev
                if k % stride == 0:
                    log(f"  [play] CHIRP_TX sym={k} frame={bf} t_emit={bf/sr:.3f}s "
                        f"wall={wall:.3f} dac={dac:.3f}")
            elif kind == "done":
                done.set()
                break

    # Probe + log the actual output interface (resample detection).
    try:
        from shared import log_audio_interface
        log_audio_interface("output", out_dev, sr, channels=nch, log=log)
    except Exception as exc:
        log(f"  [AUDIO_IFACE] output probe unavailable: {exc}")

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()
    stream = sd.OutputStream(samplerate=sr, channels=nch, dtype="float32",
                             device=out_dev, callback=callback)
    log(f"  [play] OutputStream OPEN samplerate={float(stream.samplerate):.0f}Hz "
        f"blocksize={stream.blocksize} latency={float(stream.latency)*1000:.1f}ms "
        f"device={stream.device}  ({nframes/sr:.1f}s, {len(bounds)} chirps, "
        f"first onset {start/sr:.3f}s)")
    with stream:
        done.wait(timeout=nframes / sr + 10)        # callback signals completion
        time.sleep(float(stream.latency) + 0.2)     # let buffered tail flush
    done.set()
    mon.join(timeout=1.0)
    log(f"  [play] playback complete: {len(bounds)} chirps emitted, "
        f"{state['xruns']} output underruns")
    return "ok"


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect all encoder parameters from the UI ────────────────────────────
    # Every PARAMS.get() here is discovered by the param scraper and shown in
    # the test UI.  Add new encoder knobs here to make them configurable.

    message         = PARAMS.get("message",         "")
    # Sample rate is NOT a user param: it follows the selected OUTPUT device's
    # native (shared-mode) rate so the host never resamples the playback stream.
    # A non-integer 48k->44.1k resample smears chirp-to-chirp phase and, under
    # load, drops blocks — fatal to long-symbol matched filtering.  Falls back
    # to 44100 only if the device can't be probed.
    _out_dev_probe = None
    try:
        _out_dev_probe = int(OUTPUT_DEV)
    except (TypeError, ValueError):
        _out_dev_probe = None
    try:
        from shared import native_samplerate
        fs = native_samplerate("output", _out_dev_probe) or 44100
    except Exception as _exc:
        log(f"  [AUDIO_IFACE] output rate probe failed ({_exc}); fs=44100")
        fs = 44100
    log(f"  fs={fs}Hz (from output device native rate; not a user param)")
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
    # None => let css_encoder auto-scale training_symbols by symbol_dur
    # (fewer chirps at long durations).  An explicit value still wins.
    training_symbols = PARAMS.get("training_symbols",   None)
    max_frame_s      = PARAMS.get("max_frame_s",       30.0)
    symbol_gap_frac  = PARAMS.get("symbol_gap_frac",  0.05)

    encode_cfg = {
        "output":            str(RUN_DIR / "css_output.wav"),
        "info_out":          str(RUN_DIR / "css_info.json"),
        "result_out":        str(RUN_DIR / f"encode_result_{CLIENT_ID}.json"),
        "fs":                fs,
        "f_min":             f_min,
        "f_max":             f_max,
        "symbol_dur":        symbol_dur,
        "chunk_size":        chunk_size,
        "amplitude":         amplitude,
        "rs_t":              rs_t,
        "interleave_rows":   interleave_rows,
        "diff_encode":       diff_encode,
        "preamble":          preamble,
        "max_frame_s":       max_frame_s,
        "symbol_gap_frac":   symbol_gap_frac,
    }
    # Only pin training_symbols if the user set one; otherwise the encoder
    # auto-scales it by symbol_dur and records the value in css_info.json.
    if training_symbols is not None:
        encode_cfg["training_symbols"] = training_symbols
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
        f"interleave_rows={interleave_rows}  diff_encode={diff_encode}  "
        f"training_symbols={training_symbols if training_symbols is not None else 'auto'}  "
        f"max_frame_s={max_frame_s}  "
        f"symbol_gap_frac={symbol_gap_frac}")

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
        try:
            play_status = play_with_chirp_logging(audio, sr, out_dev, RUN_DIR)
        except Exception as cb_exc:
            log(f"  callback player failed ({cb_exc}); falling back to sd.play")
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