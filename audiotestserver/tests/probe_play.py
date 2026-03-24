"""
tests/specprobe_play.py  -  role: play  (pair with specprobe_listen.py)

Spectrum probe: plays a sequence of test tones across 300-8000Hz to let
the listener measure SNR, distortion and echo per frequency band.

Sequence per frequency:
  [silence] [tone] [silence]
  ^noise    ^signal ^echo decay

The manifest (what was played and when) is written to the run dir so
the listener can align its recording to the right windows.

Params (from server UI JSON field):
{
  "freq_start":    300,
  "freq_end":      8000,
  "num_freqs":     32,
  "tone_dur":      0.5,
  "silence_dur":   0.3,
  "amplitude":     0.8,
  "fs":            44100,
  "spacing":       "log"    // "log" or "linear"
}
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CLIENT_ID  = os.environ.get("AUDIOTOOL_CLIENT_ID", "unknown")
OUTPUT_DEV = os.environ.get("AUDIOTOOL_OUTPUT_DEVICE", "")
RUN_DIR    = Path(os.environ.get("AUDIOTOOL_RUN_DIR", "."))
SERVER_URL = os.environ.get("AUDIOTOOL_SERVER_URL", "http://127.0.0.1:8000")
PARAMS     = json.loads(os.environ.get("AUDIOTOOL_PARAMS", "{}"))
RUN_ID     = os.environ.get("AUDIOTOOL_RUN_ID", "unknown")

FS          = int(PARAMS.get("fs",          44100))
FREQ_START  = float(PARAMS.get("freq_start",  300))
FREQ_END    = float(PARAMS.get("freq_end",   8000))
NUM_FREQS   = int(PARAMS.get("num_freqs",      32))
TONE_DUR    = float(PARAMS.get("tone_dur",     0.5))
SILENCE_DUR = float(PARAMS.get("silence_dur",  0.3))
AMPLITUDE   = float(PARAMS.get("amplitude",    0.8))
SPACING     = PARAMS.get("spacing", "log")


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


def make_tone(freq, duration, fs, amplitude):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Apply short cosine fade-in/out to avoid clicks
    fade = int(fs * 0.01)
    tone = amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)
    if len(tone) > 2 * fade:
        window = np.ones(len(tone))
        window[:fade]  = np.linspace(0, 1, fade)
        window[-fade:] = np.linspace(1, 0, fade)
        tone *= window
    return tone


def main():
    import sounddevice as sd

    out_dev = to_int_dev(OUTPUT_DEV)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[{CLIENT_ID}] specprobe_play | output_device={out_dev}")

    # Build frequency list
    if SPACING == "log":
        freqs = np.logspace(np.log10(FREQ_START), np.log10(FREQ_END),
                            NUM_FREQS).tolist()
    else:
        freqs = np.linspace(FREQ_START, FREQ_END, NUM_FREQS).tolist()
    freqs = [round(f, 2) for f in freqs]

    log(f"  {NUM_FREQS} freqs: {freqs[0]:.0f}Hz - {freqs[-1]:.0f}Hz ({SPACING} spacing)")
    log(f"  tone={TONE_DUR}s  silence={SILENCE_DUR}s  amplitude={AMPLITUDE}")

    silence     = np.zeros(int(FS * SILENCE_DUR), dtype=np.float32)
    total_dur   = NUM_FREQS * (SILENCE_DUR + TONE_DUR + SILENCE_DUR)
    log(f"  total duration: {total_dur:.1f}s")

    # Build manifest and audio buffer first
    manifest = {
        "run_id":      RUN_ID,
        "player":      CLIENT_ID,
        "fs":          FS,
        "freqs":       freqs,
        "tone_dur":    TONE_DUR,
        "silence_dur": SILENCE_DUR,
        "amplitude":   AMPLITUDE,
        "sequence":    []
    }

    audio_buf     = []
    sample_offset = 0

    for freq in freqs:
        tone = make_tone(freq, TONE_DUR, FS, AMPLITUDE)

        noise_start = sample_offset
        tone_start  = sample_offset + len(silence)
        tone_end    = tone_start + len(tone)
        decay_start = tone_end
        decay_end   = decay_start + len(silence)

        manifest["sequence"].append({
            "freq":         freq,
            "noise_start":  noise_start,
            "tone_start":   tone_start,
            "tone_end":     tone_end,
            "decay_start":  decay_start,
            "decay_end":    decay_end,
        })

        audio_buf.extend(silence)
        audio_buf.extend(tone)
        audio_buf.extend(silence)

        sample_offset = decay_end

    audio = np.array(audio_buf, dtype=np.float32)

    # Write manifest BEFORE signalling ready so listener finds it immediately
    manifest_path = RUN_DIR / "specprobe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"  manifest written -> {manifest_path.name}")

    # Signal ready and wait for listener to arm
    coord_signal("specprobe_ready")
    log("  signalled specprobe_ready - waiting for listener to arm...")
    armed = coord_wait_any("specprobe_armed", timeout=30)
    if not armed:
        log("  no listener armed within 30s - playing anyway")
    else:
        log(f"  listener(s) armed: {armed}")

    # Play
    log(f"  playing {len(audio)/FS:.1f}s probe sequence...")
    sd.play(audio, samplerate=FS, device=out_dev)
    sd.wait()
    log("  playback complete")

    coord_signal("specprobe_played")

    result = {
        "client_id":  CLIENT_ID,
        "role":       "play",
        "run_id":     RUN_ID,
        "status":     "ok",
        "freqs":      freqs,
        "duration_s": round(len(audio) / FS, 2),
    }
    out = RUN_DIR / f"result_{CLIENT_ID}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()