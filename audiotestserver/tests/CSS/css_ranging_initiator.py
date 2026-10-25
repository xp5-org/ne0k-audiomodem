# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_ranging_initiator.py — two-way acoustic ranging, INITIATOR role.

Plays a short decode-free ranging BURST (Barker preamble chirps — no training /
payload / CRC), capturing continuously so its own mic records both its own burst
(self-echo, ~0 delay) and the responder's reply burst (after round-trip flight).
Distance is recovered post-hoc by css_ranging_analyze from both nodes' capture
WAVs (matched filter), which sidesteps commodity-soundcard playback-latency
jitter and the decoder's single-stream limits.

The burst is short on purpose: the responder turnaround it induces is a few
seconds instead of ~40 s, which collapses the two-soundcard clock-rate-mismatch
error (that error scales with turnaround — see css-modem-layer notes).

Coordination (server /api/coord signals): rng_ready / req_played / reply_played.
"""
import json
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).parent

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

from testlib import (                                        # noqa: E402
    CLIENT_ID, INPUT_DEV, OUTPUT_DEV, RUN_DIR, PARAMS, RUN_ID, SERVER_URL,
    log, to_int_dev, coord_signal, coord_check, coord_wait,
    Recorder, play_blocking,
)
from css_ranging_analyze import ranging_burst                # noqa: E402

INIT_ID, RESP_ID = 1, 2


def _enc_params():
    return dict(
        f_min=float(PARAMS.get("f_min", 800)),
        f_max=float(PARAMS.get("f_max", 4000)),
        symbol_dur=float(PARAMS.get("symbol_dur", 0.3)),
        amplitude=float(PARAMS.get("amplitude", 0.3)),
    )


def _wait_both_ready(timeout=30):
    coord_signal("rng_ready")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(coord_check("rng_ready")) >= 2:
            return True
        time.sleep(0.3)
    return False


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ep = _enc_params()
    in_dev  = to_int_dev(INPUT_DEV)
    out_dev = to_int_dev(OUTPUT_DEV)
    try:
        from shared import native_samplerate
        cap_fs = int(native_samplerate("input",  in_dev)  or 48000)
        out_fs = int(native_samplerate("output", out_dev) or 48000)
    except Exception:
        cap_fs = out_fs = 48000
    log(f"[{CLIENT_ID}] ranging INITIATOR  cap_fs={cap_fs} out_fs={out_fs} "
        f"in={in_dev} out={out_dev}  params={ep}")

    audio = ranging_burst(out_fs, ep)

    rec = Recorder(in_dev, cap_fs)
    rec.start()
    time.sleep(0.3)

    if not _wait_both_ready():
        log("  ERROR: responder never signalled rng_ready")
        rec.stop()
        print(json.dumps({"status": "error", "failure_reason": "no_peer_ready"}))
        return

    time.sleep(0.5)
    req_frame = rec.frames_captured
    log(f"  playing REQ burst ({len(audio)/out_fs:.1f}s) at ~frame {req_frame}")
    play_blocking(audio, out_fs, out_dev)
    coord_signal("req_played")

    if not coord_wait("reply_played", timeout=float(PARAMS.get("reply_wait_s", 30))):
        log("  WARNING: no reply_played signal — saving capture anyway")
    time.sleep(float(PARAMS.get("tail_s", 1.5)))
    rec.stop()

    cap_path = RUN_DIR / f"{CLIENT_ID}_ranging.wav"
    rec.save(cap_path)
    meta = {"role": "initiator", "client_id": CLIENT_ID,
            "init_id": INIT_ID, "resp_id": RESP_ID,
            "cap_fs": cap_fs, "out_fs": out_fs,
            "req_frame_approx": req_frame, "enc_params": ep, "wav": cap_path.name}
    (RUN_DIR / f"ranging_{CLIENT_ID}.json").write_text(json.dumps(meta, indent=2))
    log(f"  saved {cap_path.name} ({rec.frames_captured} frames)")
    print(json.dumps({"status": "ok", **meta}))


if __name__ == "__main__":
    main()
