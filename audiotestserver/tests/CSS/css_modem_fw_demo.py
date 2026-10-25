# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_modem_fw_demo.py — announce -> neighbour-ACK demo / test for the firmware.

Boots two ModemFirmware instances and runs a real half-duplex exchange through
the ACTUAL css_encoder (TX) and css_decoder (RX) — no hardware.  A simulated
acoustic channel just adds propagation delay (and optional noise) to the PCM
between the two nodes.  This exercises the whole stack end to end:

    A.announce()  --encode-->  [channel]  --decode-->  B.handle_frames()
        B discovers A, ACKs it
    B ack pcm     --encode-->  [channel]  --decode-->  A.handle_frames()
        A logs "neighbour heard my announce"

Also checks self-init + parameter/seq persistence across a reboot.

Run:  python tests/css_modem_fw_demo.py [-v]      (exit 0 = pass)
"""
import argparse
import os
import sys
import tempfile

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

import css_modem as cm                       # noqa: E402
from css_modem_fw import ModemFirmware        # noqa: E402
from css_regress import decode_offline        # noqa: E402  (real decoder driver)


def channel(pcm: np.ndarray, dist_m: float, fs: int,
            noise_std: float = 0.0) -> np.ndarray:
    """Simulated acoustic hop: prepend propagation delay, add optional noise."""
    delay = int(round(dist_m / cm.SPEED_OF_SOUND_M_S * fs))
    out = np.concatenate([np.zeros(delay, dtype=np.float32), pcm.astype(np.float32)])
    if noise_std > 0:
        out = out + np.random.normal(0, noise_std, out.shape).astype(np.float32)
    return out


def _decode(fw: ModemFirmware, pcm: np.ndarray):
    fs = fw.cfg["phy"]["fs"]
    res, _log = decode_offline(pcm, fs, fw.info_path)
    return res.get("frames", [])


def test_announce_ack(verbose=False, dist_m=3.0, noise_std=0.0):
    work = tempfile.mkdtemp(prefix="fwdemo_")
    cfgA = os.path.join(work, "nodeA.json")
    cfgB = os.path.join(work, "nodeB.json")

    A = ModemFirmware(cfgA, node_id=1, verbose=verbose).boot()
    B = ModemFirmware(cfgB, node_id=2, verbose=verbose).boot()
    fs = A.cfg["phy"]["fs"]

    # 1) A announces itself; B hears it through the channel.
    pcmA = A.announce()
    framesB = _decode(B, channel(pcmA, dist_m, fs, noise_std))
    acks_from_B = B.handle_frames(framesB)   # B discovers A and returns ACK pcm

    if not any(e["kind"] == "discovered" and e["peer"] == 1
               for e in B.drain_events()):
        return "FAIL: node B did not discover node A from the announce"
    if not acks_from_B:
        return "FAIL: node B produced no ACK for A's announce"

    # 2) B's ACK travels back; A must register the neighbour ack.
    for ack_pcm in acks_from_B:
        framesA = _decode(A, channel(ack_pcm, dist_m, fs, noise_std))
        A.handle_frames(framesA)

    a_events = A.drain_events()
    if not any(e["kind"] == "ack" and e["peer"] == 2 for e in a_events):
        return f"FAIL: node A never got B's ACK (events={a_events})"
    return True


def test_persistence():
    """seq and params survive a reboot from the on-disk config."""
    work = tempfile.mkdtemp(prefix="fwpersist_")
    cfg = os.path.join(work, "node.json")

    A = ModemFirmware(cfg, node_id=7).boot()
    A.announce(); A.announce(); A.announce()      # advance + persist seq
    seq_before = A.node._seq
    if seq_before == 0:
        return "FAIL: seq did not advance"

    # Reboot from the SAME config file (no node_id override needed).
    B = ModemFirmware(cfg).boot()
    if B.cfg["id"] != 7:
        return f"FAIL: id not restored ({B.cfg['id']})"
    if B.node._seq != seq_before:
        return (f"FAIL: seq not restored across reboot "
                f"({B.node._seq} != {seq_before})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="channel gaussian noise std (default 0 = clean)")
    args = ap.parse_args()

    tests = [
        ("persistence",   lambda: test_persistence()),
        ("announce_ack",  lambda: test_announce_ack(args.verbose,
                                                    noise_std=args.noise)),
    ]
    print("CSS MODEM FIRMWARE DEMO\n")
    failures = 0
    for name, fn in tests:
        try:
            r = fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = f"EXCEPTION {type(e).__name__}: {e}"
        if r is True:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name}: {r}")
    print(f"\n*** {'ALL PASS' if not failures else str(failures)+' FAILED'} ***")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
