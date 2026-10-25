# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_announce_announcer.py — over-the-air announce/ack demo, ANNOUNCER role.

Boots a ModemFirmware (id 1), plays a self-announce BEACON (full-preamble frame,
which is exactly the training+Barker a cold neighbour needs to wake+sync), then
listens for the neighbour's ACK.  Captures continuously so its mic records both
its own beacon (self-echo) and the incoming ACK; the ACK is recovered by
decoding the post-beacon region of the capture (skipping the self-echo beacon,
which shares PHY-seq 0 with the ACK and would otherwise be deduped).

Coordination signals: ann_ready (both) / beacon_played / ack_played (neighbour).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

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
    CLIENT_ID, INPUT_DEV, OUTPUT_DEV, RUN_DIR, PARAMS,
    log, to_int_dev, coord_signal, coord_check, coord_wait,
    Recorder, play_blocking, decode_array, save_wav,
)
from announcelib import (                                    # noqa: E402
    ANNOUNCER_ID, NEIGHBOUR_ID, make_firmware, bake_decode_info,
)


def _wait_both_ready(timeout=30):
    coord_signal("ann_ready")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(coord_check("ann_ready")) >= 2:
            return True
        time.sleep(0.3)
    return False


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    in_dev  = to_int_dev(INPUT_DEV)
    out_dev = to_int_dev(OUTPUT_DEV)
    try:
        from shared import native_samplerate
        cap_fs = int(native_samplerate("input",  in_dev)  or 48000)
        out_fs = int(native_samplerate("output", out_dev) or 48000)
    except Exception:
        cap_fs = out_fs = 48000
    log(f"[{CLIENT_ID}] ANNOUNCER id={ANNOUNCER_ID} cap_fs={cap_fs} "
        f"out_fs={out_fs} in={in_dev} out={out_dev}")

    fw = make_firmware(ANNOUNCER_ID, out_fs, str(RUN_DIR))
    beacon_pcm = fw.announce()                     # BEACON, encoded at out_fs
    beacon_dur = len(beacon_pcm) / out_fs
    dec_info   = bake_decode_info(cap_fs, str(RUN_DIR))
    log(f"  beacon {beacon_dur:.1f}s ({len(beacon_pcm)} smp @ {out_fs}Hz)")

    rec = Recorder(in_dev, cap_fs)
    rec.start()
    time.sleep(0.3)

    if not _wait_both_ready():
        rec.stop()
        print(json.dumps({"status": "error", "failure_reason": "no_peer_ready"}))
        return

    time.sleep(0.5)
    beacon_frame = rec.frames_captured
    log(f"  playing BEACON at ~frame {beacon_frame}")
    play_blocking(beacon_pcm, out_fs, out_dev)
    coord_signal("beacon_played")

    if not coord_wait("ack_played", timeout=float(PARAMS.get("ack_wait_s", 150))):
        log("  WARNING: no ack_played signal from neighbour")
    time.sleep(float(PARAMS.get("tail_s", 2.0)))
    rec.stop()

    cap = rec.data()
    cap_path = RUN_DIR / f"{CLIENT_ID}_announce.wav"
    save_wav(cap, cap_fs, cap_path)

    # Decode ONLY the region after our own beacon finished, so the self-echo
    # beacon (PHY-seq 0) doesn't dedup-swallow the neighbour's ACK (also seq 0).
    guard = float(PARAMS.get("post_beacon_guard_s", 1.0))
    start = int(beacon_frame + (beacon_dur + guard) * cap_fs)
    seg = cap[start:] if 0 <= start < len(cap) else cap[:0]
    log(f"  decoding ACK region [{start}:] ({len(seg)/cap_fs:.1f}s of {len(cap)/cap_fs:.1f}s)")

    ack_received = False
    ack_seq = None
    dec_status = "no_signal"
    if len(seg) > cap_fs:
        res = decode_array(seg, cap_fs, dec_info)
        dec_status = res.get("status")
        for f in res.get("frames", []):
            if "bytes_hex" not in f:
                continue
            for a in fw.node.on_rx(bytes.fromhex(f["bytes_hex"]),
                                   int(f.get("at_sample", 0))):
                if a["kind"] == "ack" and a["peer"] == NEIGHBOUR_ID:
                    ack_received = True
                    ack_seq = a["seq"]
        log(f"  ACK-region decode status={dec_status} "
            f"frames={len(res.get('frames', []))}")

    result = {"status": "ok", "role": "announcer", "client_id": CLIENT_ID,
              "ack_received": ack_received, "ack_seq": ack_seq,
              "decode_status": dec_status,
              "beacon_dur_s": round(beacon_dur, 2),
              "cap_fs": cap_fs, "out_fs": out_fs, "wav": cap_path.name}
    (RUN_DIR / f"announce_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
    log(f"  RESULT ack_received={ack_received} ack_seq={ack_seq}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
