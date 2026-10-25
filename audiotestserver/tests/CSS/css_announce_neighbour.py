# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_announce_neighbour.py — over-the-air announce/ack demo, NEIGHBOUR role.

Boots a ModemFirmware (id 2), listens for the announcer's BEACON, decodes it from
its own capture, and — via the firmware's announce->neighbour-ACK policy — plays
a unicast ACK back so the announcer learns it was heard.  This closes the loop:
the ACK is a genuine RESPONSE to decoding the announce, not a scripted reply.

Coordination signals: ann_ready (both) / beacon_played (announcer) / ack_played.
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
    log(f"[{CLIENT_ID}] NEIGHBOUR id={NEIGHBOUR_ID} cap_fs={cap_fs} "
        f"out_fs={out_fs} in={in_dev} out={out_dev}")

    fw       = make_firmware(NEIGHBOUR_ID, out_fs, str(RUN_DIR))
    dec_info = bake_decode_info(cap_fs, str(RUN_DIR))

    rec = Recorder(in_dev, cap_fs)
    rec.start()
    time.sleep(0.3)

    if not _wait_both_ready():
        rec.stop()
        print(json.dumps({"status": "error", "failure_reason": "no_peer_ready"}))
        return

    if not coord_wait("beacon_played", timeout=float(PARAMS.get("beacon_wait_s", 150))):
        rec.stop()
        print(json.dumps({"status": "error", "failure_reason": "no_beacon_signal"}))
        return

    # Small margin so the beacon's tail is fully captured, then stop the mic
    # before decoding (keeps the real sounddevice stream clear of the decode's
    # mocked module, and we don't need to record our own ACK here).
    time.sleep(float(PARAMS.get("decode_margin_s", 1.5)))
    rec.stop()

    cap = rec.data()
    cap_path = RUN_DIR / f"{CLIENT_ID}_announce.wav"
    save_wav(cap, cap_fs, cap_path)

    res = decode_array(cap, cap_fs, dec_info)
    log(f"  beacon decode status={res.get('status')} "
        f"frames={len(res.get('frames', []))} "
        f"hex={[f.get('bytes_hex') for f in res.get('frames', [])]}")

    ack_pcms = fw.handle_frames(res.get("frames", []))
    discovered = sorted({e["peer"] for e in fw.drain_events()
                         if e["kind"] == "discovered"})

    ack_played = False
    if ack_pcms:
        log(f"  discovered peers {discovered} -> playing {len(ack_pcms)} ACK(s) "
            f"@ {out_fs}Hz")
        for pcm in ack_pcms:
            play_blocking(pcm, out_fs, out_dev)
        coord_signal("ack_played")
        ack_played = True
    else:
        log(f"  no beacon decoded (discovered={discovered}) -> no ACK")
        coord_signal("ack_played")     # unblock the announcer's wait either way

    result = {"status": "ok", "role": "neighbour", "client_id": CLIENT_ID,
              "discovered": discovered, "ack_played": ack_played,
              "decode_status": res.get("status"),
              "cap_fs": cap_fs, "out_fs": out_fs, "wav": cap_path.name}
    (RUN_DIR / f"announce_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
    log(f"  RESULT discovered={discovered} ack_played={ack_played}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
