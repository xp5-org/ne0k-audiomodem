# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_modem_loopback.py — bind ModemNode to the REAL physical layer (DSP in loop).

css_modem_sim.py validates the protocol + ranging math at the PACKET level (no
audio).  css_ranging_selftest.py validates that the decoder's per-frame
`at_sample` tracks position through real chirps.  This module covers the piece
between them: that a ModemNode, driven by frames recovered from ACTUAL
css_encoder/css_decoder output, produces the right Actions — the exact glue the
hardware transport will use.

Why single-packet decodes (not a mixed multi-node capture)
----------------------------------------------------------
css_decoder decodes ONE coherent transmission: it stops after the info's frame
count (css_decoder.py ~L1159) and de-dups on the PHY sequence number (~L1109).
Independently-encoded modem packets each carry PHY-seq 0 and info.frames=1, so
concatenating several into one capture would cap/de-dup all but the first.  A
real half-duplex device decodes each heard transmission on its own anyway, so we
model reception as: decode one packet's waveform, route every recovered frame
through the node.  Placement offset is added back to keep `at_sample` on the
shared global timeline.

Each packet is ~30 s of audio at these PHY settings (per-frame training +
preamble + 4-byte CRC), so this runs in minutes, not the seconds of the
packet-level sim.  That airtime cost is itself a finding for hardware bring-up.

Run:  python tests/css_modem_loopback.py        (exit 0 = pass; ~minutes)
"""
import contextlib
import io
import os
import sys

import numpy as np

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

import css_encoder                                   # noqa: E402
import css_modem as cm                                # noqa: E402
from css_modem_node import ModemNode                   # noqa: E402
from css_regress import decode_offline, load_wav, WORK  # noqa: E402

FS    = 48000
C     = cm.SPEED_OF_SOUND_M_S
SYM   = 0.3                       # > 0.25 -> MF path (carries at_sample/bytes_hex)
CHUNK = 5                         # >= max packet (RANGE_REPLY = 2 hdr + 3 turn)

_ENC = dict(fs=FS, f_min=800.0, f_max=4000.0, amplitude=0.3,
            diff_encode=False, symbol_gap_frac=0.05, symbol_dur=SYM,
            chunk_size=CHUNK)

INFO = None                       # shared info path (identical framing for all)
ANCHOR_OFS = 0                    # samples from waveform start to decode anchor

_render_cache: dict[bytes, np.ndarray] = {}


# ── PHY render / recover ─────────────────────────────────────────────────────

def render(raw: bytes) -> np.ndarray:
    """Encode a modem packet to float32 PCM through the real css_encoder."""
    raw = bytes(raw)
    if raw in _render_cache:
        return _render_cache[raw]
    name = "lb_" + raw.hex()
    wav  = os.path.join(WORK, name + ".wav")
    info = os.path.join(WORK, name + "_info.json")
    enc  = dict(_ENC, message_bytes=raw, output=wav, info_out=info,
                result_out=os.path.join(WORK, name + "_enc.json"))
    os.makedirs(WORK, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        css_encoder.encode(enc)
    pcm, _ = load_wav(wav)
    global INFO
    if INFO is None:
        INFO = info
    _render_cache[raw] = pcm
    return pcm


def decode_frames(pcm: np.ndarray, base_offset: int = 0) -> list[dict]:
    """Decode one transmission's waveform through the real css_decoder and
    unpack every recovered frame.  `base_offset` is added to each at_sample so
    the caller can express it on a shared global timeline.  Returns dicts
    {packet, at_sample, raw}, sorted by at_sample."""
    res, _ = decode_offline(pcm.astype(np.float32), FS, INFO)
    out = []
    for f in res.get("frames", []):
        if "bytes_hex" not in f or "at_sample" not in f:
            continue
        try:
            p = cm.unpack_packet(bytes.fromhex(f["bytes_hex"]))
        except cm.ModemError:
            continue
        out.append({"packet": p, "raw": bytes.fromhex(f["bytes_hex"]),
                    "at_sample": int(f["at_sample"]) + base_offset})
    out.sort(key=lambda d: d["at_sample"])
    return out


def _calibrate():
    """Anchor offset of an isolated packet (needed to map placements->global)."""
    global ANCHOR_OFS
    frames = decode_frames(render(cm.pack_packet(1, 2, cm.TYPE_BEACON, 0)))
    if not frames:
        raise RuntimeError("calibration probe failed to decode")
    ANCHOR_OFS = frames[0]["at_sample"]


# ── tests ────────────────────────────────────────────────────────────────────

def test_data_binding():
    """A DATA packet, rendered to real chirps and decoded, drives a ModemNode:
    it delivers the correct payload and emits an ACK.  Proves the RX glue."""
    b = ModemNode(2, fs=FS, turnaround_samples=FS)
    raw, _tag = ModemNode(1, fs=FS, turnaround_samples=FS).send_data(2, b"hi")

    actions = []
    for fr in decode_frames(render(raw)):
        actions += b.on_rx(fr["raw"], fr["at_sample"])

    delivered = [a for a in actions if a["kind"] == "deliver"]
    if not delivered:
        return "FAIL: node did not deliver the DATA payload"
    if delivered[0]["packet"]["payload"][:2] != b"hi":
        return f"FAIL: payload {delivered[0]['packet']['payload']!r}"
    if not any(a["kind"] == "send" for a in actions):
        return "FAIL: node produced no ACK for unicast DATA"
    return True


def test_beacon_discovery_binding():
    """A BEACON drives a first-time 'discovered' action through the real PHY."""
    b = ModemNode(2, fs=FS, turnaround_samples=FS)
    raw, _tag = ModemNode(7, fs=FS, turnaround_samples=FS).make_beacon()
    actions = []
    for fr in decode_frames(render(raw)):
        actions += b.on_rx(fr["raw"], fr["at_sample"])
    disc = [a for a in actions if a["kind"] == "discovered"]
    if not disc or disc[0]["peer"] != 7:
        return f"FAIL: beacon did not surface discovery of peer 7 ({actions})"
    return True


def test_at_sample_precision(offsets=(0, 7000, 61_111, 200_000)):
    """Ranging rides on at_sample being SAMPLE-precise, not symbol-quantized.
    Place the identical packet at several capture offsets (incl. sub-symbol,
    non-round ones), decode each in isolation, and confirm at_sample tracks the
    offset to within a fraction of a symbol.  Reports the residual in samples
    and the equivalent one-way ranging error, since that is the real deployment
    number.

    A clean channel can't show acoustic/multipath jitter (a hardware question),
    but it exposes any symbol-grid quantization bug in the anchor arithmetic —
    which would make acoustic ranging unusable regardless of SNR.  (Very long
    lead-in silence can defeat acquisition; such an offset is skipped, not
    failed — that is an acquisition-patience issue, not an anchor-precision
    one.)"""
    pcm = render(cm.pack_packet(1, 2, cm.TYPE_RANGE_REQ, 3))
    residuals, skipped = [], []
    for off in offsets:
        buf = np.concatenate([np.zeros(off, dtype=np.float32), pcm])
        frames = decode_frames(buf)
        if not frames:
            skipped.append(off)
            continue
        residuals.append((off, frames[0]["at_sample"] - (off + ANCHOR_OFS)))

    print("    at_sample residual vs placement (samples -> 1-way cm):")
    worst = 0
    for off, r in residuals:
        print(f"      off={off:>8}  residual={r:+5d} smp  "
              f"({abs(r) / FS * C * 100:5.1f} cm)")
        worst = max(worst, abs(r))
    if skipped:
        print(f"    skipped (no acquisition at long lead-in): {skipped}")
    if len(residuals) < 2:
        return f"FAIL: fewer than 2 offsets decoded ({len(residuals)})"

    sps = int(FS * SYM)
    if worst >= sps:
        return (f"FAIL: at_sample off by {worst} smp >= 1 symbol ({sps}) — "
                f"anchor is symbol-quantized, ranging via at_sample unusable")
    print(f"    worst residual {worst} smp = "
          f"{worst / FS * C * 100:.1f} cm one-way (symbol = {sps} smp)")
    return True


def main():
    print("CSS MODEM DSP-IN-THE-LOOP LOOPBACK\n")
    print("  calibrating PHY (render + decode a probe)...", flush=True)
    _calibrate()
    print(f"  packet = {len(_render_cache[next(iter(_render_cache))])/FS:.1f}s "
          f"audio  ANCHOR_OFS={ANCHOR_OFS}\n", flush=True)

    tests = [
        ("data_binding",             test_data_binding),
        ("beacon_discovery_binding", test_beacon_discovery_binding),
        ("at_sample_precision",      test_at_sample_precision),
    ]
    failures = 0
    for name, fn in tests:
        print(f"  running {name} ...", flush=True)
        try:
            r = fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = f"EXCEPTION {type(e).__name__}: {e}"
        if r is True:
            print(f"  PASS  {name}\n")
        else:
            failures += 1
            print(f"  FAIL  {name}: {r}\n")
    print(f"*** {'ALL PASS' if not failures else str(failures)+' FAILED'} ***")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
