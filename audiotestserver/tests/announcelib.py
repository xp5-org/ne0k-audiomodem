# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""
announcelib.py — shared helpers for the over-the-air announce/ack demo.

Named without an underscore so the rig's `<test>_<role>.py` discovery does
not register it as a bogus role. Both css_announce role scripts import from
here.

The PHY recipe defaults to the hardware-validated settings (symbol_dur 0.5,
400-6000 Hz, diff_encode), with chunk_size 2 so a 2-byte modem packet is
exactly one PHY frame. Every param is overridable via the run's PARAMS.

TX amplitude is fixed at 0.8: through built-in laptop speakers a low-amplitude
chirp is inaudible to other nodes' carrier sense, causing talk-over
collisions. Do not lower it per-device via amp_map.

Each node encodes at its speaker's native rate (out_fs) and decodes its
capture at its mic's native rate (cap_fs); the chirp frequencies are
absolute, so the two rates need not match.
"""
import contextlib
import io
import json
import os

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

import css_encoder
import css_modem as cm
from css_modem_fw import ModemFirmware
from testlib import PARAMS

ANNOUNCER_ID, NEIGHBOUR_ID = 1, 2

def recipe(fs) -> dict:
    """PHY profile at sample rate `fs` (Hz), from PARAMS with validated defaults."""
    return dict(
        fs=int(fs),
        f_min=float(PARAMS.get("f_min", 400)),
        f_max=float(PARAMS.get("f_max", 6000)),
        symbol_dur=float(PARAMS.get("symbol_dur", 0.5)),
        amplitude=float(PARAMS.get("amplitude", 0.8)),
        symbol_gap_frac=float(PARAMS.get("symbol_gap_frac", 0.05)),
        chunk_size=int(PARAMS.get("chunk_size", 2)),
        diff_encode=bool(PARAMS.get("diff_encode", True)),
        rs_t=int(PARAMS.get("rs_t", 0)),
        conv_code=None,
    )

def make_firmware(node_id: int, out_fs: int, workdir: str) -> ModemFirmware:
    """Boot a ModemFirmware whose PHY encodes at the device output rate."""
    cfg_path = os.path.join(workdir, f"node{node_id}.json")
    return ModemFirmware(cfg_path, node_id=node_id, workdir=workdir,
                         phy_override=recipe(out_fs), verbose=True).boot()

def bake_decode_info(cap_fs: int, workdir: str, expected_frames: int = 6) -> str:
    """Produce an info.json (frame geometry) to decode captures at `cap_fs`.

    Encodes a throwaway 2-byte packet at cap_fs with the recipe, then bumps the
    'frames' field so css_decoder does not stop after a single frame (a capture
    can hold more than one transmission).
    """
    r = recipe(cap_fs)
    base = os.path.join(workdir, f"decinfo_{int(cap_fs)}")
    cfg = dict(r,
               message_bytes=cm.pack_packet(0, 0, 0, 0),   # 2 bytes -> 1 frame
               output=base + ".wav", info_out=base + "_info.json",
               result_out=base + "_e.json")
    with contextlib.redirect_stdout(io.StringIO()):
        css_encoder.encode(cfg)
    info = json.load(open(base + "_info.json"))
    info["frames"] = int(expected_frames)
    json.dump(info, open(base + "_info.json", "w"), indent=2)
    return base + "_info.json"
