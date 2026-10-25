# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_ranging_selftest.py — offline validation of the modem/ranging plumbing.

No hardware.  Exercises the full encode -> WAV -> decode path (via the
regression harness's mocked sounddevice) to prove three things the two-way
ranging protocol depends on:

  A. BINARY INTEGRITY — a modem packet's raw bytes survive the PHY intact
     (the decoder no longer drops non-UTF-8 frames; bytes_hex round-trips).

  B. TIMESTAMP LINEARITY — the decoder's per-frame `at_sample` is a faithful
     capture-sample clock: two frames emitted a known number of symbols apart
     are reported that same distance apart (within sub-symbol tolerance).
     This is the measurement two-way ranging is built on.

  C. DISTANCE MATH — css_modem.rtt_to_distance_m turns a (round-trip,
     turnaround) sample pair into the correct one-way metres.

Run:  python tests/css_ranging_selftest.py        (exit 0 = pass)
"""
import contextlib
import io
import json
import os
import sys

TESTS = os.path.dirname(os.path.abspath(__file__))

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

import css_encoder                       # noqa: E402
import css_modem as cm                    # noqa: E402
from css_regress import decode_offline, load_wav, WORK  # noqa: E402


def _encode(name, *, message_bytes=None, message=None, **over):
    os.makedirs(WORK, exist_ok=True)
    wav = os.path.join(WORK, name + ".wav")
    info = os.path.join(WORK, name + "_info.json")
    enc = dict(output=wav, info_out=info,
               result_out=os.path.join(WORK, name + "_enc.json"),
               fs=48000, f_min=800, f_max=4000,
               amplitude=0.3, diff_encode=False, symbol_gap_frac=0.05,
               symbol_dur=0.5)
    if message_bytes is not None:
        enc["message_bytes"] = message_bytes
    if message is not None:
        enc["message"] = message
    enc.update(over)
    with contextlib.redirect_stdout(io.StringIO()):
        css_encoder.encode(enc)
    return wav, info


def test_binary_integrity():
    """A: a modem packet with high/zero bytes round-trips through the PHY."""
    pkt = cm.pack_packet(src_id=0x3, dst_id=cm.BROADCAST_ID,
                         mtype=cm.TYPE_DATA, seq=0x5,
                         payload=b"\x00\xff\x80\x01")
    # chunk_size >= packet length -> the whole packet is ONE PHY frame.
    wav, info = _encode("rng_bin", message_bytes=pkt, chunk_size=len(pkt))
    pcm, fs = load_wav(wav)
    res, log = decode_offline(pcm, fs, info)

    frames = [f for f in res.get("frames", []) if "bytes_hex" in f]
    if not frames:
        return f"FAIL: no frames with bytes_hex (status={res.get('status')})"
    got = bytes.fromhex(frames[0]["bytes_hex"])
    # The PHY pads the final short chunk with zero bytes; the packet is a prefix.
    if got[:len(pkt)] != pkt:
        return f"FAIL: bytes mismatch got={got.hex()} want(prefix)={pkt.hex()}"
    p = cm.unpack_packet(got)
    if not (p["src"] == 0x3 and p["dst"] == cm.BROADCAST_ID
            and p["type"] == cm.TYPE_DATA and p["seq"] == 0x5
            and p["is_broadcast"]):
        return f"FAIL: unpacked header wrong: {p}"
    return True


def test_timestamp_linearity():
    """B: two frames a known symbol-count apart report that same at_sample gap."""
    # 2-byte message, chunk_size=1 -> exactly 2 PHY frames, each with its own
    # full training+preamble (per_frame default) -- the same robust geometry the
    # regression suite decodes reliably at 0.5 s.  Anchor-to-anchor spacing is
    # deterministic from the info file.
    wav, info = _encode("rng_lin", message_bytes=b"hi", chunk_size=1)
    conf = json.load(open(info))
    fs = conf["fs"]
    sps = int(fs * conf["symbol_dur"])
    gap = int(sps * conf["symbol_gap_frac"])
    sps_eff = sps + gap
    frame_period_syms = (conf["training_symbols"] + len(conf["preamble"])
                         + conf["frame_symbols"])
    expected_delta = frame_period_syms * sps_eff

    pcm, fs2 = load_wav(wav)
    res, log = decode_offline(pcm, fs2, info)
    frames = [f for f in res.get("frames", [])
              if "at_sample" in f and "error" not in f]
    if len(frames) < 2:
        return (f"FAIL: need 2 frames with at_sample, got {len(frames)} "
                f"(status={res.get('status')})")
    frames.sort(key=lambda f: f["at_sample"])
    measured_delta = frames[1]["at_sample"] - frames[0]["at_sample"]

    # Tolerance: one symbol period.  The anchor is the correlation peak; grid
    # re-fits can nudge it by a fraction of a symbol, never a whole one.
    err = abs(measured_delta - expected_delta)
    if err > sps_eff:
        return (f"FAIL: at_sample delta {measured_delta} vs expected "
                f"{expected_delta} (err {err} > {sps_eff} = 1 symbol)")
    return True


def test_distance_math():
    """C: rtt_to_distance_m converts a known (rtt, turnaround) to metres."""
    fs = 48000
    turnaround = int(0.5 * fs)          # responder waits 0.5 s
    true_dist = 5.0                      # metres, one way
    tof_one_way = true_dist / cm.SPEED_OF_SOUND_M_S
    net = int(round(tof_one_way * 2 * fs))
    rtt = turnaround + net
    d = cm.rtt_to_distance_m(rtt, turnaround, fs)
    if abs(d - true_dist) > 0.05:
        return f"FAIL: distance {d:.3f} m != {true_dist} m"
    # a mis-detection (turnaround > rtt) must yield a negative, rejectable value
    if cm.rtt_to_distance_m(turnaround - 10, turnaround, fs) >= 0:
        return "FAIL: bad detection did not produce negative distance"
    return True


def main():
    tests = [
        ("binary_integrity",   test_binary_integrity),
        ("timestamp_linearity", test_timestamp_linearity),
        ("distance_math",      test_distance_math),
    ]
    print("CSS RANGING / MODEM SELFTEST\n")
    failures = 0
    for name, fn in tests:
        try:
            r = fn()
        except Exception as e:
            import traceback
            r = f"EXCEPTION {type(e).__name__}: {e}"
            traceback.print_exc()
        if r is True:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name}: {r}")
    print(f"\n*** {'ALL PASS' if not failures else str(failures)+' FAILED'} ***")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
