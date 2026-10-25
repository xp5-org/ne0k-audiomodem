# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_listen.py  -  Framework wrapper for css_decoder.py (role: listen).

Handles audiotool coordination (waits for css_ready, signals css_armed),
spawns css_decoder.py as a subprocess, streams its structured stdout to the
server log, and collects the result.

WAV captures
------------
Two WAV files are produced:

1. ``<RUN_DIR>/<CLIENT_ID>_input.wav``   (decoder-internal capture)
   Written by css_decoder.py via the ``wav_out`` config key.  Recording
   begins when the decoder subprocess starts, which is ~0.5–2 s after
   css_armed fires (Python import overhead).  Used by the existing
   spectrograph generator and offline replay.

2. ``<RUN_DIR>/<CLIENT_ID>_listen_capture.wav``   (wrapper-level capture)
   Written by THIS file using a sounddevice.InputStream that is opened
   BEFORE css_armed is signalled, so it captures audio from the very first
   training chirp.  Used for cross-client comparison and as the primary
   input for the annotated spectrograph tool.

Timing sidecar
--------------
``<RUN_DIR>/<CLIENT_ID>_listen_timing.json`` accompanies the listen capture.
It contains per-symbol markers (type, frame, time offset from recording
start) derived from css_info.json and the decode result.  The spectrograph
tool reads this to draw coloured region overlays without re-parsing audio.

This file is NOT meant to be run standalone - use css_decoder.py directly.
"""

import json
import subprocess
import sys
import threading
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

from testlib import (

    CLIENT_ID, INPUT_DEV, RUN_DIR, PARAMS, RUN_ID, SERVER_URL,
    log, to_int_dev, coord_signal, coord_check, coord_wait_any,
)

TESTS_DIR = Path(__file__).parent

# ── Wrapper-level listen capture ──────────────────────────────────────────────
#
# NOTE: we intentionally do NOT open a second sounddevice.InputStream here.
# Opening the same input device in both the parent (css_listen.py) and the
# child (css_decoder.py) causes PortAudio to send SIGKILL to one of the two
# openers on many systems, silently killing the decoder before it prints a
# single byte.
#
# Instead, css_decoder.py records via its wav_out config key and writes
# <CLIENT_ID>_input.wav.  css_listen.py post-processes that file to build
# the timing sidecar JSON (_build_timing_sidecar below), which contains all
# the per-symbol annotations the spectrograph tool needs.
#
# If a true parallel capture is needed (to start before the decoder's Python
# import overhead), use a non-PortAudio recorder such as arecord (Linux) or
# sox (macOS/Linux) in a separate subprocess so PortAudio is not involved.

def _build_timing_sidecar(
    path: Path,
    info: dict,
    result: dict,
    wav_path: Path,
    fs: int,
) -> bool:
    """
    Write a JSON sidecar alongside the listen-capture WAV.

    The sidecar contains per-symbol markers derived from css_info.json and
    the decode result.  The spectrograph tool reads this to draw coloured
    region overlays (training / preamble / payload) without re-parsing audio.

    wav_path  – path to the captured WAV (decoder's wav_out file).
    fs        – sample rate of that WAV.

    Structure per encoded frame
    ---------------------------
    [training_symbols × chirp] [n_preamble × chirp] [payload_symbols × chirp]

    Marker types
    ------------
    "training"  – alternating up/down chirps, AGC and timing lock region
    "preamble"  – Barker sync sequence; marker includes "bit" field
    "payload"   – data chirps; marker includes "frame" index

    Time reference
    --------------
    t = 0 in every marker is the nominal start of the encoded audio.
    The decoder's WAV recording starts slightly after the player's first
    chirp (Python import delay, typically 0.5-2 s).  The sidecar stores
    this as playback_delta_s = 0.0 (unknown); the spectrograph tool can
    refine it using sync_marker.t to pin the grid to the actual audio.

    Sync-adjusted timing
    --------------------
    When the decoder acquires sync it reports sync_at_symbol: the symbol
    index at which the preamble END was detected (after lag correction).
    The spectrograph tool can use this to pin the marker grid to the
    actual audio by computing:

        t_recorded = (sync_at_symbol - (training_symbols + n_preamble - 1))
                     × sym_eff  +  playback_delta_s

    and shifting all markers by the residual against the observed peak.
    """
    sym_dur  = float(info.get("symbol_dur",      0.25))
    gap_frac = float(info.get("symbol_gap_frac", 0.0))
    sym_eff  = sym_dur * (1.0 + gap_frac)

    train_n  = int(info.get("training_symbols", 16))
    preamble = info.get("preamble", [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    n_pre    = len(preamble)

    dur_s    = float(info.get("duration_s", 0.0))
    total_syms = int(round(dur_s / sym_eff)) if sym_eff > 0 else 0

    # Number of frames: prefer info["n_frames"], fall back to decode result
    # old # n_frames = int(info.get("n_frames", 0) or result.get("frames_decoded") or 0)
    n_frames = int(info.get("frames", 0) or result.get("frames_decoded") or 0)
    # Payload symbols per frame (derived from total, or left as -1 if unknown)
    overhead   = train_n + n_pre          # symbols per frame that are not payload
    if n_frames > 0:
        payload_per_frame = max(0, (total_syms - n_frames * overhead) // n_frames)
    else:
        payload_per_frame = -1           # unknown

    # Build markers
    markers = []
    t       = 0.0
    sym_idx = 0
    frames_to_annotate = n_frames if n_frames > 0 else max(1, total_syms // max(1, overhead + 1))

    for fi in range(frames_to_annotate):
        # Training
        for _ in range(train_n):
            markers.append({"t": round(t, 6), "sym": sym_idx,
                            "frame": fi, "type": "training"})
            t += sym_eff;  sym_idx += 1

        # Preamble
        for bit in preamble:
            markers.append({"t": round(t, 6), "sym": sym_idx,
                            "frame": fi, "type": "preamble", "bit": int(bit)})
            t += sym_eff;  sym_idx += 1

        # Payload
        n_pay = payload_per_frame if payload_per_frame >= 0 else 0
        for _ in range(n_pay):
            markers.append({"t": round(t, 6), "sym": sym_idx,
                            "frame": fi, "type": "payload"})
            t += sym_eff;  sym_idx += 1

    # Sync acquisition marker (from decode result)
    sync_marker = None
    sat = result.get("sync_at_symbol")
    if sat is not None:
        sync_marker = {
            "sym":   int(sat),
            "t":     round(int(sat) * sym_eff, 6),
            "score": result.get("sync_score"),
            "lag":   result.get("sync_lag", 0),
        }

    sidecar = {
        # Recording metadata
        "wav_path":              str(wav_path),
        "record_duration_s":     round(wav_path.stat().st_size / (fs * 2), 3)
                                 if wav_path.exists() else 0.0,
        "fs":                    fs,
        # Waveform parameters (from css_info.json)
        "sym_dur":               sym_dur,
        "sym_eff":               sym_eff,
        "gap_frac":              gap_frac,
        "f_min":                 info.get("f_min", 400),
        "f_max":                 info.get("f_max", 6000),
        "training_symbols":      train_n,
        "n_preamble":            n_pre,
        "preamble":              preamble,
        "n_frames":              n_frames,
        "payload_syms_per_frame": payload_per_frame,
        "total_symbols_nominal": total_syms,
        # Timing reference
        # t=0 in markers = nominal start of encoded audio.
        # The listen capture began cap.start_epoch seconds before the player
        # received css_armed; set playback_delta_s to the measured or estimated
        # offset (positive = player started after our recording started).
        "playback_delta_s":      0.0,    # spectrograph may refine from sync_marker
        # Decode outcome
        "sync_marker":           sync_marker,
        "frames_decoded":        result.get("frames_decoded"),
        "decode_status":         result.get("status"),
        # Per-symbol annotation list
        "markers":               markers,
    }

    try:
        path.write_text(json.dumps(sidecar, indent=2))
        log(f"  timing sidecar: {path.name}  "
            f"({len(markers)} markers, {n_frames} frames)")
        return True
    except Exception as exc:
        log(f"  [capture] sidecar write failed: {exc}")
        return False

def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    log(f"[{CLIENT_ID}] css_listen: waiting for player css_ready signal...")
    ready = coord_wait_any("css_ready", timeout=30)
    if not ready:
        log(f"  ERROR: no css_ready signal within 30 s")
        print(json.dumps({"client_id": CLIENT_ID, "status": "error",
                          "failure_reason": "no_ready_signal", "run_id": RUN_ID}), flush=True)
        return

    log(f"  player ready: {ready}")

    # Wait for css_info.json to appear (written by the player)
    info_path = RUN_DIR / "css_info.json"
    deadline  = time.time() + 10
    while time.time() < deadline:
        if info_path.exists():
            break
        time.sleep(0.2)
    else:
        log(f"  ERROR: css_info.json not found after ready signal")
        print(json.dumps({"client_id": CLIENT_ID, "status": "error",
                          "failure_reason": "no_info_file", "run_id": RUN_ID}), flush=True)
        return

    in_dev = to_int_dev(INPUT_DEV)

    # Sample rate is NOT a user param. The decoder captures and builds its chirp
    # templates at the INPUT device's own native (shared-mode) rate so PortAudio
    # never resamples the mic stream: a 48k->44.1k resample smears intra-symbol
    # phase and collapses the matched filter. The chirp is defined physically
    # (f_min/f_max Hz over symbol_dur s), so the decoder does NOT need to match
    # the encoder's fs in css_info.json; each side runs natively. Falls back to
    # css_info.json fs only if the device can't be probed.
    try:
        # shared.py lives at the repo root, one level up from tests/; add it to
        # sys.path as css_decoder.py:83 does. Without this, the probe falls
        # back to css_info.json's fs, the ENCODER's rate. On two machines with
        # different native rates that opens the mic at the wrong rate,
        # PortAudio resamples, and intra-symbol phase smears.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from shared import native_samplerate
        listen_fs = native_samplerate("input", in_dev)
    except Exception as _exc:
        log(f"  [AUDIO_IFACE] input rate probe failed ({_exc})")
        listen_fs = None
    if listen_fs:
        log(f"  capture fs={listen_fs}Hz (input device native rate; not a user param)")
    else:
        log("  capture fs: falling back to css_info.json rate (device probe unavailable)")

    # Derive timeout from the encoded audio duration
    try:
        info_data    = json.loads(info_path.read_text())
        audio_dur    = float(info_data.get("duration_s", 0))
        auto_timeout = audio_dur + 60 if audio_dur > 0 else 360
    except Exception:
        audio_dur    = 0
        auto_timeout = 360

    try:
        timeout_sec = float(PARAMS.get("timeout_sec", auto_timeout))
    except (TypeError, ValueError):
        timeout_sec = float(auto_timeout)
    log(f"  audio duration: {audio_dur:.1f}s  decode timeout: {timeout_sec:.0f}s")

    # WAV capture path — css_decoder.py writes the raw mic input here
    wav_capture_path = RUN_DIR / f"{CLIENT_ID}_input.wav"

    # Cooperative stop sentinel.  Creating this file asks the decoder to drain
    # its buffers and write the WAV before exiting — the only way to get a clean
    # flush on Windows, where proc.terminate() (TerminateProcess) hard-kills the
    # subprocess before its end-of-run WAV write can execute.  See the long-run
    # "WAV capture: not written" symptom this replaces.
    stop_file_path = RUN_DIR / f"{CLIENT_ID}_decoder_stop"
    if stop_file_path.exists():
        stop_file_path.unlink()

    decode_cfg = {
        "info":              str(info_path),
        "result_out":        str(RUN_DIR / f"decode_result_{CLIENT_ID}.json"),
        "device":            in_dev,
        # Capture rate = input device native rate (overrides css_info.json fs).
        # Omit the key if probing failed so the decoder falls back to conf["fs"].
        **({"fs": listen_fs} if listen_fs else {}),
        "timeout_sec":       timeout_sec,
        "stop_file":         str(stop_file_path),
        "silence_threshold": PARAMS.get("silence_threshold", 0.002),
        # sync_threshold: ceiling for the adaptive effective threshold. The
        #   syndet scores with cosine similarity, range:
        #     background (random payload / Barker sidelobe): ~0.10-0.20
        #     confirmed preamble ceiling:                     ~0.85-0.95
        #   0.55 sits between both; the adaptive EMA (syndet_margin)
        #   self-calibrates the operating point below this in practice. Raise
        #   toward 0.70 only in extremely noisy environments where false
        #   positives persist.
        "sync_threshold":    PARAMS.get("sync_threshold",    0.55),
        # pll_weight: blend factor for IF-slope confidence vs MF confidence.
        #   Governs narrowband-interference rejection during preamble
        #   acquisition as well as post-sync bit decisions. 0 = MF only;
        #   1.0 = full IF-slope weight when fit is strong. Default 0.30 is a
        #   gentle blend; raise toward 0.6 in environments with persistent
        #   tonal interference (HVAC, projector fans, birds).
        "pll_weight":        PARAMS.get("pll_weight",        0.30),
        # agc_attack: lowered from 0.8 to 0.5 to prevent gain overshoot on the
        # silence→preamble transition (which reversed correlator polarity on the
        # first 2-3 preamble symbols and depressed the sync score to ~0.54).
        "agc_attack":        PARAMS.get("agc_attack",        0.5),
        # syndet_lookback: preamble detector ring size.  Auto-sized in the decoder
        # to max(n*3, frame_symbols + 2*n_preamble) when not set here; that default
        # keeps the preamble visible for the entire inter-frame gap so a single
        # missed preamble no longer forces waiting a full frame in the dark.
        # Override here only if you need a specific value (e.g. memory-constrained).
        **( {"syndet_lookback": int(PARAMS["syndet_lookback"])}
            if "syndet_lookback" in PARAMS else {} ),
        # Raw mic input WAV capture (for offline replay / spectrograph)
        "wav_out":           str(wav_capture_path),
        # -- Syndet noise-immunity knobs -------------------------------------
        # SNR gate: symbols below syndet_min_snr_db contribute zero weight to
        # the preamble cross-correlation (deep-fade erasure).
        "syndet_min_snr_db": PARAMS.get("syndet_min_snr_db",  0.0),
        "syndet_snr_ref_db": PARAMS.get("syndet_snr_ref_db", 12.0),
        # Lag guard: boolean — when True (default), reject sync acquisitions
        # where the preamble match is more than syndet_max_lag symbols in the
        # past.  Prevents stale false positives from the noise history ring.
        "syndet_lag_guard":  PARAMS.get("syndet_lag_guard",   True),
        # Adaptive background threshold: effective sync threshold =
        # min(sync_threshold, bg_ema + syndet_margin).
        "syndet_bg_alpha":   PARAMS.get("syndet_bg_alpha",    0.02),
        "syndet_margin":     PARAMS.get("syndet_margin",      0.15),
        # timing_search_frac: ±fraction of sps used as the post-acquisition
        # symbol-search window.  0.25 (default) gives a ±25% timing tolerance
        # in settled tracking mode.
        "timing_search_frac": PARAMS.get("timing_search_frac", 0.25),
        # DSP overrides – forward any explicitly-set PARAMS to the decoder so
        # that it can verify against (or supplement) the css_info.json values.
        # Keep this list in sync with the PARAMS.get() block in css_play.py.
        # NOTE: the decoder reads rs_t, interleave_rows, diff_encode, conv_code
        # from conf (css_info.json), not cfg — they are "conf-authoritative".
        # Forwarding them here is a no-op for the decoder but lets the two config
        # files be diffed for sanity-checking.
        # NOTE: symbol_gap_frac intentionally omitted — the decoder reads it
        # from conf (css_info.json), NOT cfg.  Passing it here was Bug 3.
        # NOTE: max_frame_s is an encoder-side guard only; css_decoder.py does
        # not read it.  It is NOT forwarded here to avoid confusion.
        **({k: PARAMS[k] for k in (
                "f_min", "f_max", "symbol_dur",
                "chunk_size",      # auto-sized by encoder if absent; decoder reads from conf
                "preamble",
                "conv_code",
                "diff_encode",
                "interleave_rows",
                "rs_t",            # was "rs_nsym" — renamed to match encoder's key name
                "training_symbols",
            ) if k in PARAMS}),
    }
    cfg_path = RUN_DIR / f"css_decode_config_{CLIENT_ID}.json"
    cfg_path.write_text(json.dumps(decode_cfg, indent=2))

    # Signal armed
    coord_signal("css_armed")
    log(f"  signalled css_armed - starting css_decoder.py...")

    decode_script = TESTS_DIR / "css_decoder.py"
    proc = subprocess.Popen(
        [sys.executable, str(decode_script), str(cfg_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    last_result_json = None
    output_lines = []

    def _read_stdout():
        nonlocal last_result_json
        for line in proc.stdout:
            line = line.rstrip()
            output_lines.append(line)
            if line.startswith("{"):
                last_result_json = line
            else:
                log(f"  [decode] {line}")

    reader = threading.Thread(target=_read_stdout, daemon=True)
    reader.start()

    # Main thread: poll for css_played then stop decoder after tail
    tail_secs  = float(PARAMS.get("tail_secs", 5))
    stop_after = None
    poll_start = time.time()

    # Grace window for a cooperative stop: after we drop the stop_file, the
    # decoder must finish its current blocking read (up to one symbol period),
    # set eos, drain its buffer, and write the WAV.  Size it from the symbol
    # duration so long-symbol runs (read_n = sps_eff samples per read) get
    # enough time to flush instead of being hard-killed mid-loop.
    try:
        sym_dur = float(info_data.get("symbol_dur", PARAMS.get("symbol_dur", 0.25)))
    except Exception:
        sym_dur = 0.25
    stop_grace = max(tail_secs, sym_dur * 2.0 + 8.0)

    def _stop_decoder_gracefully():
        """Ask the decoder to flush+exit; hard-kill only if it ignores us."""
        log(f"  requesting graceful decoder stop (grace {stop_grace:.0f}s)")
        try:
            stop_file_path.touch()
        except Exception as exc:
            log(f"  [stop] could not create stop_file: {exc}")
        try:
            proc.wait(timeout=stop_grace)
            log(f"  decoder exited cleanly after stop request")
            return
        except subprocess.TimeoutExpired:
            log(f"  decoder did not stop within grace - terminating")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    while proc.poll() is None:
        if stop_after is None:
            played = coord_check("css_played")
            if played:
                stop_after = time.time() + tail_secs
                log(f"  css_played received - stopping decoder after {tail_secs}s tail")
        elif time.time() >= stop_after:
            log(f"  tail elapsed - stopping decoder")
            _stop_decoder_gracefully()
            break
        if time.time() - poll_start > timeout_sec + 30:
            log(f"  hard timeout - stopping decoder")
            _stop_decoder_gracefully()
            break
        time.sleep(0.5)

    reader.join(timeout=5)
    proc.wait()

    # Log decoder-internal WAV capture outcome
    if wav_capture_path.exists():
        size_kb = wav_capture_path.stat().st_size // 1024
        log(f"  WAV capture: {wav_capture_path.name}  ({size_kb} KB)")
    else:
        log(f"  WAV capture: not written (decoder may have exited early)")

    # Load result file
    result_path = RUN_DIR / f"decode_result_{CLIENT_ID}.json"
    try:
        result = json.loads(result_path.read_text())
    except Exception:
        try:
            result = json.loads(last_result_json) if last_result_json else {}
        except Exception:
            result = {"status": "error", "error": "could not read result"}

    # CSS spectrograph is rendered SERVER-SIDE, not here: for a multi-minute
    # capture the STFT/plot took 10-17 minutes (results/20260903_071919_css,
    # _074930_css), which made this role script look hung for the rest of
    # the run even though decode had already finished.  BASE_DIR is a shared
    # SMB mount, so the server renders straight from wav_capture_path once
    # this node's result lands — see app.py's _maybe_render_css_spectrograph.

    # Timing sidecar for the decoder's WAV capture — build even if the
    # listen-capture WAV itself came from the decoder subprocess so the
    # spectrograph tool always gets an annotated companion JSON.
    listen_timing_path = RUN_DIR / f"{CLIENT_ID}_listen_timing.json"
    if wav_capture_path.exists():
        # The capture WAV is written at the decoder's actual capture rate (the
        # input device's native rate).  Read it straight from the header so the
        # timing sidecar's sample math matches the file exactly; fall back to the
        # detected rate, then the encoder rate, if the header can't be read.
        try:
            import wave as _wave
            with _wave.open(str(wav_capture_path), "rb") as _wf:
                cap_fs = _wf.getframerate()
        except Exception:
            cap_fs = listen_fs or int(info_data.get("fs", 44100))
        _build_timing_sidecar(
            listen_timing_path, info_data, result,
            wav_capture_path, cap_fs,
        )

    result.update({"client_id": CLIENT_ID, "role": "listen", "run_id": RUN_ID})
    if wav_capture_path.exists():
        result["wav_capture"] = str(wav_capture_path)
        result["listen_wav"]  = str(wav_capture_path)   # alias for spectrograph tool
    if listen_timing_path.exists():
        result["listen_timing"] = str(listen_timing_path)
    # spectrograph_png is added later by the server's background render
    # (app.py's _maybe_render_css_spectrograph patches this same JSON file).
    result_path.write_text(json.dumps(result, indent=2))

    log(f"  status={result.get('status')} frames={result.get('frames_decoded', '?')}")
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()