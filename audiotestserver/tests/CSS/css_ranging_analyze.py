# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_ranging_analyze.py — post-hoc two-way ranging from a run's capture WAVs.

Reads the initiator + responder capture WAVs (and their ranging_*.json metadata)
produced by css_ranging_initiator/responder, and recovers the distance by
matched-filtering each capture against the packet template.  Working entirely
post-hoc from each node's own continuous capture sidesteps commodity-soundcard
playback-latency jitter and the decoder's single-stream limits:

  initiator capture -> req_tx  (its own REQ, self-echo)   and reply_rx (peer)
  responder capture -> req_rx  (peer)  and reply_tx (its own REPLY, self-echo)

Both transmissions share identical framing, so the matched-filter fiducial
(template-start alignment) carries the same constant offset K in every
measurement; K cancels in the differences below, as do the two nodes' clock
OFFSETS (each rtt/turnaround is a difference within ONE node's own capture):

  rtt        = reply_rx - req_tx        (initiator clock)
  turnaround = reply_tx - req_rx        (responder clock)
  distance   = ((rtt - turnaround) / 2) / fs * speed_of_sound

Usage:  python tests/css_ranging_analyze.py <run_dir> [--true-dist M]
"""
import argparse
import json
import sys
import wave
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

import css_encoder                                            # noqa: E402
import css_modem as cm                                         # noqa: E402


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        fs = w.getframerate()
        x = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
             .astype(np.float64) / 32768.0)
    return x, fs


def preamble_template(fs, ep):
    """Waveform of just the Barker preamble chirps (+ inter-symbol gaps).

    The FIDUCIAL template AND (via ranging_burst) the actual transmitted ranging
    waveform.  Barker-13 has an aperiodic, near-sidelobe-free autocorrelation, so
    its matched-filter peak is unambiguous — unlike a periodic training block,
    whose near-equal peaks one symbol apart let a low-SNR peer reception lock a
    symbol-slipped ~300 ms off (fatal to ranging).  13 symbols is ~50 dB of
    processing gain — plenty — so ranging needs nothing else in the waveform.
    """
    preamble = ep.get("preamble") or css_encoder.BARKER_13
    up = css_encoder.generate_chirp(ep["f_min"], ep["f_max"],
                                    ep["symbol_dur"], fs).astype(np.float64)
    dn = css_encoder.generate_chirp(ep["f_max"], ep["f_min"],
                                    ep["symbol_dur"], fs).astype(np.float64)
    sps = int(fs * ep["symbol_dur"])
    gap = np.zeros(int(sps * 0.05))
    return np.concatenate([np.concatenate([(up if b == 1 else dn), gap])
                           for b in preamble])


def ranging_burst(fs, ep, lead_s=0.5):
    """The transmitted ranging waveform: a short lead silence + the Barker
    preamble chirp burst, scaled by amplitude.  Decode-free (no training /
    payload / CRC), so it is ~4 s instead of a ~38 s framed packet — which
    collapses the responder turnaround and, with it, the two-soundcard
    clock-rate-mismatch error that a long turnaround amplifies."""
    pre  = preamble_template(fs, ep)
    lead = np.zeros(int(lead_s * fs))
    return (np.concatenate([lead, pre]) * float(ep.get("amplitude", 0.3))
            ).astype(np.float32)


def matched_filter(cap, tmpl):
    """|cross-correlation| of cap against tmpl; index = sample in cap where the
    template's start aligns.  (Peak = onset fiducial for that transmission.)"""
    n = len(cap) + len(tmpl) - 1
    nfft = 1 << (n - 1).bit_length()
    C = np.fft.rfft(cap, nfft) * np.conj(np.fft.rfft(tmpl, nfft))
    return np.abs(np.fft.irfft(C, nfft)[:len(cap)])


def _leading_edge(env, p, fs, back_ms=35.0, fwd_ms=6.0, thr=0.4):
    """Refine a detected correlation peak `p` to the DIRECT-PATH arrival.

    Indoors the strongest matched-filter peak is often a REFLECTION arriving a
    few ms after the (weaker) direct path — ranging off the max then overstates
    the distance.  Search a window that reaches back before p, find the first
    sample rising above thr*local-max (the leading edge), then take the local
    peak just after it.  For a clean self-echo (single peak) this returns the
    peak itself, so all four fiducials sit on the direct-path preamble peak."""
    a = max(0, p - int(back_ms / 1000 * fs))
    b = min(len(env), p + int(fwd_ms / 1000 * fs))
    seg = env[a:b]
    mx = float(seg.max()) or 1.0
    edge = int(np.argmax(seg >= thr * mx))          # first crossing
    win = seg[edge:edge + int(0.003 * fs)]          # peak within 3 ms of edge
    return a + edge + int(np.argmax(win)) if len(win) else a + edge


def two_fiducials(cap, pre_tmpl, excl, fs):
    """Direct-path preamble fiducials (sorted) of the two transmissions.

    Matched-filter against the Barker preamble (processing gain + aperiodic,
    symbol-slip-free autocorrelation), pick the two strongest >=excl-separated
    peaks (one per transmission), then refine each to its leading edge so a
    later reflection can't inflate the range.
    Returns ((pos_a, pos_b) sorted, (strength_a, strength_b))."""
    env = matched_filter(cap, pre_tmpl)
    p1 = int(np.argmax(env)); s1 = float(env[p1])
    guard = env.copy()
    guard[max(0, p1 - excl):min(len(env), p1 + excl)] = 0.0
    p2 = int(np.argmax(guard)); s2 = float(guard[p2])
    le1, le2 = _leading_edge(env, p1, fs), _leading_edge(env, p2, fs)
    order = sorted([(le1, s1), (le2, s2)])
    return (order[0][0], order[1][0]), (order[0][1], order[1][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--true-dist", type=float, default=None,
                    help="known ground-truth distance (m) for error reporting")
    ap.add_argument("--speed", type=float, default=cm.SPEED_OF_SOUND_M_S)
    args = ap.parse_args()
    run = Path(args.run_dir)

    metas = {}
    for mp in run.glob("ranging_*.json"):
        m = json.loads(mp.read_text())
        if "role" in m:                       # skip our own ranging_result.json
            metas[m["role"]] = m
    if "initiator" not in metas or "responder" not in metas:
        print(f"ERROR: need both ranging_*.json in {run} (have {list(metas)})")
        return 1

    ini, res = metas["initiator"], metas["responder"]
    fs = int(ini["cap_fs"])
    if int(res["cap_fs"]) != fs:
        print(f"WARNING: capture rates differ ({fs} vs {res['cap_fs']}); "
              f"ranging assumes a common fs")
    ep = {k: (int(v) if k == "chunk_size" else float(v))
          for k, v in ini["enc_params"].items()}

    pre  = preamble_template(fs, ep)                        # fiducial = the burst
    # Exclude ~one burst length around a peak so the 2nd peak is the OTHER
    # transmission (not a sidelobe).  The two bursts are >=turnaround apart, and
    # turnaround > burst by design, so both survive.
    excl = len(pre)
    ini_cap, _ = load_wav(run / ini["wav"])
    res_cap, _ = load_wav(run / res["wav"])

    (req_tx, reply_rx), s_ini = two_fiducials(ini_cap, pre, excl, fs)
    (req_rx, reply_tx), s_res = two_fiducials(res_cap, pre, excl, fs)

    rtt  = reply_rx - req_tx
    turn = reply_tx - req_rx
    net  = rtt - turn
    dist = (net / 2.0) / fs * args.speed

    print(f"CSS TWO-WAY RANGING  ({run.name})")
    print(f"  fs={fs}  ranging_burst/fiducial={len(pre)} smp ({len(pre)/fs:.1f}s)")
    print(f"  initiator: req_tx={req_tx}  reply_rx={reply_rx}  "
          f"(peaks {s_ini[0]:.0f}/{s_ini[1]:.0f})")
    print(f"  responder: req_rx={req_rx}  reply_tx={reply_tx}  "
          f"(peaks {s_res[0]:.0f}/{s_res[1]:.0f})")
    print(f"  rtt={rtt} smp ({rtt/fs*1000:.1f} ms)   "
          f"turnaround={turn} smp ({turn/fs*1000:.1f} ms)")
    print(f"  net flight={net} smp ({net/fs*1000:.2f} ms)")
    print(f"  => distance = {dist:.3f} m")
    if args.true_dist is not None:
        print(f"     true     = {args.true_dist:.3f} m   "
              f"error = {abs(dist-args.true_dist)*100:.1f} cm")

    out = {k: (float(v) if k == "distance_m" else int(v)) for k, v in {
        "distance_m": dist, "rtt": rtt, "turnaround": turn, "net": net,
        "fs": fs, "req_tx": req_tx, "reply_rx": reply_rx,
        "req_rx": req_rx, "reply_tx": reply_tx}.items()}
    (run / "ranging_result.json").write_text(json.dumps(out, indent=2))
    if net <= 0:
        print("  WARNING: non-positive flight time — likely a fiducial "
              "mis-detection (check peak strengths / margins)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
