# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
meshmodem_node.py -- role: node (assign to ALL hosts) -- N-node mesh modem
with OFDM as a selectable PHY, over the air.

The CSS layer is the control plane (validated cmodem loop: chirp-matched
carrier sense, CSMA/CA daemon, burst decode); OFDM is the negotiated data
plane.  One css_modem packet format rides both PHYs -- the PHY choice is a
single decision at the transmit point, PER PEER (meshphy.PhyPolicy).

N-NODE: every participating client runs this same role.  `id_map` assigns
each client a 4-bit node id (a client absent from the map REFUSES to run --
the old default-to-2 was an id-collision bomb with 3+ devices).  `data_map`
scripts the traffic: {"<src_id>": <dst_id>} -- each listed src sends its
payload to dst, upgrading that one link to OFDM when its phy_mode allows.
All other packets (ACKs to third parties, control) ride CSS.  The MAC layer
(CSMA/CA + id-staggered backoff) already arbitrates N transmitters.

Scenario per sender src->dst:
  1. src enqueues OFDM_HS_REQ to dst over CSS (phy_mode css_then_ofdm).
  2. dst auto-ACKs over CSS and, once its ACK has aired, drops into the OFDM
     handshake as RESPONDER (meshlib.run_handshake); src, on that ACK, runs
     it as INITIATOR.  One handshake at a time per node.
  3. The V.22-style handshake negotiates per-direction BATs fully in-band.
  4. src sends DATA -> PHY selector picks OFDM at the negotiated BAT; dst's
     node-layer auto-ACK rides back over OFDM the same way.

CSS FALLBACK (meshphy.PhyPolicy, per peer): CSS is home base.  If OFDM DATA
goes unacked for a full retry set, or the handshake fails, or the REQ is
never answered, the node drops back to CSS and resends the DATA there; a
node that hears ANY CSS packet from an OFDM-linked peer drops that link too
(so its ACK rides CSS -- both ends converge with no extra control traffic).
There is NO timed re-upgrade: OFDM comes back only when commanded by a peer
over CSS or when the duty-cycled passive detector
(meshlib.detect_ofdm_activity) hears OFDM on the air.

Params (JSON):
{
    "phy_mode":         "css_then_ofdm" | "css_only",
    "phy_mode_map":     {"<client_id>": "css_only", ...},   per-client override
    "id_map":           {"macdesktop": 1, "laptop": 2, "delllaptop": 3},
    "data_map":         {"1": 2, "2": 1},  src node id -> dst node id
                        (default is MUTUAL: both directions carry data, so a
                         run exercises the reverse link instead of only the
                         forward one plus its ACK.  {"1": 2} for one-way.)
    "data_payload":     "hello-over-ofdm",
    "data_payload_map": {"<client_id>": "hi", ...},         per-client override
    "run_s":            600,
    "hs_req_delay_s":   10,
    "ofdm_amplitude":   0.3,
    "ofdm_poll_s":      4.0,
    "ofdm_detect_period_s": 5.0,
    "ofdm_detect_thr":  0.5,
    "hs_timeout":       300,
    "wait_probe_s":     120,
    "wait_msg_s":       75,
    ... plus the CSS PHY params (f_min, f_max, symbol_dur, amplitude,
        symbol_gap_frac, chunk_size, diff_encode, rs_t) and the OFDM PHY /
        negotiation params (snr_margin_db, max_bits_per_bin, n_fft, cp_ratio,
        hs_freq_lo/hi, ...) -- both show up in the web UI under their own
        section headers even though phy_mode may only use one of them.
}
"""
import json
import sys
import threading
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

from testlib import (                                           # noqa: E402
    CLIENT_ID, INPUT_DEV, OUTPUT_DEV, RUN_DIR, PARAMS,
    log, to_int_dev, Recorder, play_blocking, decode_array,
    coord_signal, coord_check, coord_check_ts,
    install_shutdown_handler, on_shutdown,
)
import css_modem as cm                                          # noqa: E402
from announcelib import recipe, bake_decode_info                # noqa: E402
from css_modem_fw import ModemFirmware                          # noqa: E402
from css_modem_daemon import ModemDaemon                        # noqa: E402
from meshphy import PhyPolicy, PHY_MODES, OFDM_DATA_MODES       # noqa: E402

_MESHLIB = None


def _ml():
    """Import meshlib (and the OFDM DSP stack behind it) LAZILY.

    A css_only node never touches OFDM, but meshlib pulls in ofdm_modem ->
    reedsolo (+ scipy), which a minimal fleet device may not have installed.
    Importing at first use keeps such a node a full CSS participant on a
    CSS-only install that lacks `reedsolo`.
    """
    global _MESHLIB
    if _MESHLIB is None:
        import meshlib
        _MESHLIB = meshlib
    return _MESHLIB

DEFAULT_ID_MAP = {"macdesktop": 1, "laptop": 2, "delllaptop": 3}


CLIP_LEVEL = 0.99          # |sample| at/above this counts as clipped
CLIP_PCT_WARN = 0.01       # % of clipped samples in a burst that is a problem


def level_stats(x: np.ndarray, fs: int, win_s: float = 1.5) -> dict:
    """Peak / RMS / clipped-sample share of the LOUDEST `win_s` of `x`.

    Measured on the loudest window, not the whole buffer: a 1.1s frame inside
    a 30s poll window would otherwise have its clipping diluted by 29s of
    silence into a number nobody would look twice at.
    """
    if not len(x):
        return {"rms": 0.0, "peak": 0.0, "clip_pct": 0.0,
                "crest_db": 0.0, "hot_ms": 0.0}
    h = max(1, int(win_s * fs))
    if len(x) >= 2 * h:
        m = len(x) // h
        blocks = x[:m * h].reshape(m, h)
        x = blocks[int(np.argmax(np.mean(blocks ** 2, axis=1)))]
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    # Shape of the loudest thing in the window, so a clipped sample can be
    # attributed.  An environmental impulse (a door, a keystroke, a knock on the
    # desk) is a few ms wide with an enormous crest factor; a genuinely hot
    # OFDM burst is sustained across the window at ~11-18 dB crest.  Reporting
    # only peak/clip_pct made the two indistinguishable, so the warning blamed
    # gain for what was ambient noise.
    return {"rms": round(rms, 4),
            "peak": round(peak, 4),
            "clip_pct": round(100.0 * float(np.mean(np.abs(x) >= CLIP_LEVEL)), 3),
            # np, not math: this module never imports math, and level_stats runs
            # only when a burst arrives -- so a NameError here would surface as a
            # dead run the first time the room made a loud noise.
            "crest_db": round(float(20.0 * np.log10(peak / max(rms, 1e-9))), 1),
            "hot_ms": round(1000.0 * float(np.count_nonzero(
                np.abs(x) >= 0.5 * peak)) / max(fs, 1), 1)}


def _make_chirp_detector(f_min, f_max, symbol_dur, fs):
    """Chirp-matched carrier sense (lifted from css_cmodem_node)."""
    from css_encoder import generate_chirp
    up = generate_chirp(f_min, f_max, symbol_dur, fs).astype(np.float32)
    dn = generate_chirp(f_max, f_min, symbol_dur, fs).astype(np.float32)
    win = int(2 * symbol_dur * fs)
    nfft = 1
    while nfft < win + len(up):
        nfft <<= 1
    R_up = np.conj(np.fft.rfft(up, nfft))
    R_dn = np.conj(np.fft.rfft(dn, nfft))
    eref = float(np.sqrt(np.sum(up * up)))

    def score(w):
        if len(w) < len(up):
            return 0.0
        W = np.fft.rfft(w, nfft)
        su = float(np.max(np.abs(np.fft.irfft(W * R_up, nfft)[:len(w)])))
        sd = float(np.max(np.abs(np.fft.irfft(W * R_dn, nfft)[:len(w)])))
        return max(su, sd) / (float(np.sqrt(np.sum(w * w))) * eref + 1e-12)

    return score, win


def _full_duplex_refusal(phy_mode, id_map):   # -> str or None (py3.9: no X | Y)
    """Why ofdm_full_duplex must not start here, or None if it may.

    Full duplex splits ONE band into a downlink and an uplink around a guard
    gap.  There are exactly two halves, so there can be exactly two endpoints:
    a third node has no band to transmit in, and -- worse -- would hear the
    duplex pair as permanently busy and defer forever, or key up on top of
    them because its carrier sense never clears.

    Refusing LOUDLY at boot rather than degrading: a silent fall back to half
    duplex would look like the mode working badly instead of not running, and
    the whole point of measuring FDD is to compare it against half duplex
    honestly.  On real hardware this becomes a manually selected mode and the
    operator owns the constraint; in the test rig the framework enforces it.
    """
    if phy_mode != "ofdm_full_duplex":
        return None
    n = len(id_map)
    if n != 2:
        return (f"phy_mode=ofdm_full_duplex needs exactly 2 participants, "
                f"id_map has {n} ({sorted(id_map)}). Full duplex splits one "
                f"band into a DL and a UL half -- there is no third half, and "
                f"a third node's carrier sense would read the duplex pair as "
                f"permanently busy. Use ofdm_only (half duplex) for more than "
                f"two nodes, or cut id_map down to the two that should run.")
    return None


def _scenario_exit(data_dst, expected_srcs, got_data, *, acked: bool,
                   gave_up: bool, dst_gone: bool, done_seen: bool,
                   timed_out: bool, outbox_empty: bool) -> tuple[bool, str]:
    """May this node stop early, and why?  (False, "") to keep running.

    Pure decision, extracted so it can be regression-tested without audio --
    two separate bugs here have each cost minutes of real runtime per run, and
    both were invisible offline because the condition was buried in the main
    loop.

    The subtle term is `dst_gone`: a data_dst that has ANNOUNCED it is finished
    will never ACK, so retransmitting at it -- and the re-handshake spiral that
    follows the retries running out -- is dead air.  That is CONCLUDED, not
    delivered: `acked` stays False and the caller still reports data_ack=false.

    Why stopping there cannot lose a delivery: a node only signals DONE after
    its OWN `recv_done` held, i.e. after it received everything it was scripted
    to receive.  So a peer announcing DONE has already got my payload -- what
    went missing is its ACK, not my DATA.  Giving up on the ACK costs a
    truthful data_ack=false in my report while the peer's own `data_delivered`
    records the payload arriving; retrying instead cannot improve on that,
    because there is nobody left listening.
    """
    send_done = (data_dst is None or acked or gave_up or dst_gone)
    recv_done = all(s in got_data for s in expected_srcs)
    quiet     = (not expected_srcs or timed_out or done_seen)
    if not (send_done and recv_done and quiet and outbox_empty):
        return False, ""
    why = ("peer left before acking" if dst_gone and not acked
           else "peers reported done" if done_seen and not timed_out
           else "retry window elapsed" if timed_out
           else "nothing expected")
    return True, why


def _link_initiator(data_map: dict[int, int], node_id: int, peer: int) -> bool:
    """Should `node_id` lead the OFDM handshake with `peer`?

    data_map is {src node id -> dst node id}: who sends DATA to whom.
    Entering unconditionally as "initiator" whenever a node has a data_dst is
    correct for a ONE-WAY link (only one side has data for the other), but
    data_map can also be MUTUAL -- {"1": 2, "2": 1}, both sides sending -- and
    then BOTH endpoints reach that decision with data_dst == the other, so
    both would enter_handshake as "initiator" on the SAME shared acoustic
    link at once.  HandshakeSM has genuinely distinct initiator/responder
    state machines (i_wait_probe vs r_wait_probe), so that is a live protocol
    collision, not a harmless redundancy.

    Tie-broken by node id, which is the SAME default PhyPolicy itself falls
    back to when initiator=None (node_id == 1).  A one-way link (peer does
    not target `node_id` back) is a pure pass-through: `mutual` is False and
    this returns True exactly as the unconditional version did, so a run that
    already worked keeps working byte-for-byte.
    """
    mutual = data_map.get(peer) == node_id
    return (node_id < peer) if mutual else True


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # The rig stops a run with SIGTERM, which kills CPython outright -- so
    # anything worth keeping has to be flushed from a handler, not from the
    # bottom of this function.  Hooks are registered as their subjects come
    # into existence (capture below, result later).
    install_shutdown_handler()
    in_dev  = to_int_dev(INPUT_DEV)
    out_dev = to_int_dev(OUTPUT_DEV)
    try:
        from shared import native_samplerate
        cap_fs = int(native_samplerate("input",  in_dev)  or 48000)
        out_fs = int(native_samplerate("output", out_dev) or 48000)
    except Exception:
        cap_fs = out_fs = 48000

    id_map = PARAMS.get("id_map", DEFAULT_ID_MAP)
    if CLIENT_ID not in id_map:
        # id collisions silently corrupt an N-node run -- refuse, loudly
        msg = (f"client {CLIENT_ID!r} not in id_map {id_map} -- every "
               f"participating client needs an explicit node id")
        log(f"[{CLIENT_ID}] FATAL: {msg}")
        print(json.dumps({"client_id": CLIENT_ID, "role": "node",
                          "status": "fail", "error": msg}), flush=True)
        return
    node_id = int(id_map[CLIENT_ID])
    # node id -> client id, to match a DONE_SIG (which carries client ids)
    # against expected_srcs (which are node ids).
    id_to_client = {int(v): k for k, v in id_map.items()}

    run_s    = float(PARAMS.get("run_s", 600.0))
    tick_s   = float(PARAMS.get("tick_s", 0.4))
    min_burst_s = float(PARAMS.get("min_burst_s", 8.0))
    decode_lookback_s = float(PARAMS.get("decode_lookback_s", 600.0))
    carrier_k   = float(PARAMS.get("carrier_k", 2.2))
    hs_req_delay_s = float(PARAMS.get("hs_req_delay_s", 10.0))
    hs_timeout  = float(PARAMS.get("hs_timeout", 300.0))
    wait_probe  = float(PARAMS.get("wait_probe_s", 60.0))
    wait_msg    = float(PARAMS.get("wait_msg_s", 0.0))   # 0 = derive from budget
    hs_skip_fin = bool(float(PARAMS.get("hs_skip_fin", 1)))
    hs_poll_s   = float(PARAMS.get("hs_poll_s", 2.0))
    # Seconds of already-captured audio the handshake may search on entry: the
    # two nodes do not enter together, and a probe that finished moments before
    # was being discarded, stalling the exchange for a whole wait_probe_s.
    hs_lookback = float(PARAMS.get("hs_lookback_s", 15.0))
    # ofdm_only is OFDM HALF duplex (turn-taking, carrier-sense arbitrated);
    # ofdm_full_duplex splits the band by direction and has both ends
    # transmitting at once, which only means anything between exactly two
    # endpoints -- see the guardrail below and meshphy.PHY_MODES.
    phy_mode = str(PARAMS.get("phy_mode", "css_then_ofdm"))  # choices: css_only, css_then_ofdm, ofdm_only, ofdm_full_duplex  # group: Mode
    phy_mode = str(PARAMS.get("phy_mode_map", {}).get(CLIENT_ID, phy_mode))
    if phy_mode not in PHY_MODES:
        msg = f"phy_mode {phy_mode!r} not in {PHY_MODES}"
        log(f"[{CLIENT_ID}] FATAL: {msg}")
        print(json.dumps({"client_id": CLIENT_ID, "role": "node",
                          "node_id": node_id, "status": "fail",
                          "error": msg}), flush=True)
        return
    _fd_refusal = _full_duplex_refusal(phy_mode, id_map)
    if _fd_refusal:
        log(f"[{CLIENT_ID}] FATAL: {_fd_refusal}")
        print(json.dumps({"client_id": CLIENT_ID, "role": "node",
                          "node_id": node_id, "status": "fail",
                          "error": _fd_refusal}), flush=True)
        return
    # asymmetric acoustic links strand the return ACK on the weak direction;
    # 'symmetric' collapses both directions onto min(tx,rx) per carrier so the
    # ACK is as robust as the data (both ends compute the identical table).
    link_rate_policy = str(PARAMS.get("link_rate_policy", "symmetric"))  # choices: symmetric, independent  # group: Mode
    link_symmetric = (link_rate_policy == "symmetric")
    ofdm_amp    = float(PARAMS.get("ofdm_amplitude", 0.3))  # group: OFDM PHY
    ofdm_poll_s = float(PARAMS.get("ofdm_poll_s", 4.0))  # group: OFDM PHY
    ofdm_detect_period_s = float(PARAMS.get("ofdm_detect_period_s", 5.0))  # group: OFDM PHY
    ofdm_detect_thr = float(PARAMS.get("ofdm_detect_thr", 0.5))  # group: OFDM PHY

    # OFDM channel-negotiation knobs (override the fixed handshake convention
    # per run; both ends get identical PARAMS so they stay in agreement).
    # snr_estimator=percentile grades each carrier on its worst probe
    # sub-block so the BAT tracks data-time robustness (see
    # ofdm_modem._probe_snr_lin); raise snr_margin_db for a cleaner lower rate
    # at distance, lift max_bits_per_bin (past QPSK) to go faster when close.
    import ofdm_handshake as _ofdm_hs
    # ISI de-rating and the SNR margin are two answers to the SAME question --
    # "how much worse is data than the probe claims" -- so the margin's default
    # follows the switch.  With de-rating off, 9 dB is a blind toll standing in
    # for reverberation; with it on, the measured ceiling does that job and the
    # margin is left to cover only what it can actually cover, fade migration
    # between probe and data.  The allocator spends margin BEFORE applying the
    # ceiling so the two cannot annihilate each other, but a large margin still
    # costs carriers (11 dB vs 2 dB was 105 vs 115 active on the measured run).
    # An explicit snr_margin_db in PARAMS always wins -- this only moves the
    # DEFAULT.
    _isi_derate = float(PARAMS.get("isi_derate", 0.0))  # choices: 0, 1  # group: OFDM PHY
    # No automatic pairing: the web UI submits EVERY param with its schema
    # default, so "the run didn't ask for a margin" is not a state this script
    # can observe -- a `"snr_margin_db" not in PARAMS` test is dead code here.
    # The margin therefore always means what it says, and the note below is how
    # a mismatched pair gets flagged.
    _hs_ov = {"isi_derate": _isi_derate,
              "snr_margin_db": float(PARAMS.get("snr_margin_db", 9.0)),  # group: OFDM PHY
              "max_bits_per_bin": int(PARAMS.get("max_bits_per_bin", 2)),  # choices: 1, 2, 4, 6  # group: OFDM PHY
              "snr_estimator": str(PARAMS.get("snr_estimator", "mean")),  # choices: mean, percentile  # group: OFDM PHY
              "snr_pctile": float(PARAMS.get("snr_pctile", 15.0)),  # group: OFDM PHY
              "snr_blocks": int(PARAMS.get("snr_blocks", 8)),  # group: OFDM PHY
              # probe/handshake band edges: overridable per run to hunt for
              # usable bandwidth (e.g. hs_freq_hi=8000 probes 400-8k; the
              # allocator simply zeroes carriers that grade under margin, so
              # a dead top octave costs nothing but probe airtime)
              "freq_lo": float(PARAMS.get("hs_freq_lo", 400.0)),  # group: OFDM PHY
              "freq_hi": float(PARAMS.get("hs_freq_hi", 4000.0)),  # group: OFDM PHY
              # FFT geometry.  The cyclic prefix is cp_ratio*n_fft samples, and
              # any room reverberation arriving LATER than that lands on the
              # next symbol as ISI no equaliser can remove.  At the defaults
              # that guard is only 5.8 ms (~2 m of excess path); a reverberant
              # room can put a large fraction of received energy beyond it,
              # imposing a SINR ceiling regardless of level or distance.  Buy guard
              # time with n_fft (2048 doubles the CP to 11.6 ms for ~20% symbol
              # overhead and twice the carriers) in preference to cp_ratio (0.5
              # buys the same 11.6 ms but spends 33%).
              # Non-periodic probe symbols: see ofdm_handshake.HS_BAND.  0 goes
              # back to the periodic-only probe that over-promised.
              "probe_meas_syms": int(PARAMS.get("probe_meas_syms", 32)),  # group: OFDM PHY
              "n_fft": int(PARAMS.get("n_fft", 1024)),  # choices: 512, 1024, 2048, 4096  # group: OFDM PHY
              "cp_ratio": float(PARAMS.get("cp_ratio", 0.25))}  # choices: 0.25, 0.375, 0.5  # group: OFDM PHY
    _ofdm_hs.HS_OVERRIDES = _hs_ov
    _ofdm_hs.HS_BAT_RLE = bool(float(PARAMS.get("hs_bat_rle", 1)))  # group: OFDM PHY
    # RobustLink rate ladder (ofdm_handshake.HS_RL_STEPS).  Step k means G=2**k
    # bits per OFDM symbol -- the band splits into G interleaved groups, one bit
    # each.  Every step doubles handshake message rate for exactly 3dB of
    # per-bit spreading gain: 0=~35bps 1=~70 2=~140 3=~300 4=~600 5=~1200 (48k).
    # A CEILING; fast_groups_for still drops below it on a sparse table.
    _ofdm_hs.HS_RL_MAX_STEP = int(PARAMS.get("rl_max_step", 3))  # choices: 0, 1, 2, 3, 4, 5  # group: OFDM PHY
    # Live carriers a group must have before it may be split off.  Lower reaches
    # a higher step on fragmented tables; raise it first if step-3 messages fail
    # to decode OTA, since a thin group is the likeliest cause.
    _ofdm_hs.HS_FAST_MIN_PER_GROUP = int(PARAMS.get("rl_min_per_group", 6))  # group: OFDM PHY
    # Step for a message whose sender does not know its own tx table yet --
    # BAT_REPORT.  The receiver trials the whole ladder, so this is an SNR bet,
    # not an agreement problem; a lost one costs a full turn to retransmit.
    _ofdm_hs.HS_RL_BOOTSTRAP_STEP = int(PARAMS.get("rl_bootstrap_step", 2))  # choices: 0, 1, 2, 3  # group: OFDM PHY
    # Link capacity that must survive BAT coarsening.  Wire bytes are airtime,
    # and near the top of a table they are cheap: 0.90 typically cuts the BAT
    # ~3x for ~6% of capacity.  Raise toward 1.0 to prefer faithful over fast.
    _ofdm_hs.HS_BAT_MIN_KEEP = float(PARAMS.get("bat_min_keep", 0.90))  # group: OFDM PHY
    log(f"  ofdm negotiation: margin={_hs_ov['snr_margin_db']}dB "
        f"max_bits={_hs_ov['max_bits_per_bin']} "
        f"est={_hs_ov['snr_estimator']} p{_hs_ov['snr_pctile']:.0f} "
        f"blocks={_hs_ov['snr_blocks']} "
        f"isi_derate={'on' if _isi_derate > 0 else 'off'}")
    if _isi_derate > 0 and _hs_ov["snr_margin_db"] > 4.0:
        log(f"  note: isi_derate is on AND snr_margin_db="
            f"{_hs_ov['snr_margin_db']}dB -- these cover the same effect, so "
            f"the margin is now mostly costing carriers. Try 0-2dB.")
    # Geometry is logged with the CP in MILLISECONDS, not samples: the number
    # that matters is how much reverberation the guard interval covers, and
    # samples hide it behind the sample rate.
    #
    # Read from hs_cfg, NOT meshlib.mesh_cfg: the two now agree by construction
    # (mesh_cfg derives from the same effective band), and _ofdm_hs is already
    # imported here whereas meshlib is deliberately lazy -- a css_only node on a
    # minimal install has no reedsolo/scipy, and importing meshlib just to log a
    # line would kill it before it could do its CSS job (cf. _ml()).
    _mc = _ofdm_hs.hs_cfg(48000)
    log(f"  ofdm geometry: n_fft={_hs_ov['n_fft']} cp_ratio={_hs_ov['cp_ratio']} "
        f"-> CP {1000 * _mc['cp_len'] / 48000:.1f}ms "
        f"({343 * _mc['cp_len'] / 48000:.2f}m of excess path), "
        f"symbol {1000 * _mc['symbol_samples'] / 48000:.1f}ms, "
        f"{_mc['n_data_carriers']} data carriers "
        f"@ {_hs_ov['freq_lo']:.0f}-{_hs_ov['freq_hi']:.0f}Hz")
    log(f"  robustlink: step<={_ofdm_hs.HS_RL_MAX_STEP} "
        f"(G<={_ofdm_hs.HS_RL_STEPS[_ofdm_hs.HS_RL_MAX_STEP]}) "
        f"min/group={_ofdm_hs.HS_FAST_MIN_PER_GROUP} "
        f"bat_keep={_ofdm_hs.HS_BAT_MIN_KEEP:.2f}")

    # Frame body size.  One packet is always ONE frame (no fragmentation), so
    # this is what bounds an application payload -- not the link rate.  Bigger
    # chunks amortise the fixed preamble/CE/SYNC overhead: at QPSK/141 carriers,
    # 48B delivers 336bps of goodput and 512B delivers 2253bps.
    mesh_chunk = int(PARAMS.get("mesh_chunk", 48))  # choices: 48, 96, 237, 512, 1024
    if phy_mode != "css_only":
        # Only touch meshlib for a node that will actually use OFDM: importing
        # it pulls in the DSP stack a css_only node may not have (see _ml()).
        _ml().MESH_CHUNK = mesh_chunk

    data_payload = str(PARAMS.get("data_payload", "hello-over-ofdm"))
    data_payload = str(PARAMS.get("data_payload_map", {})
                       .get(CLIENT_ID, data_payload)).encode()
    # Check the payload NOW, not when the first DATA packet is built: an
    # oversized payload would otherwise only fail deep into a run, costing an
    # OTA slot to find out.  The cap is the frame body minus the 2B length
    # prefix and the 2B modem header.
    _ofdm_cap = mesh_chunk - 2 - 2
    if phy_mode != "css_only" and len(data_payload) > _ofdm_cap:
        _need = len(data_payload) + 4
        msg = (f"data_payload is {len(data_payload)}B but one OFDM frame "
               f"carries {_ofdm_cap}B at mesh_chunk={mesh_chunk} (frame body "
               f"{mesh_chunk}B - 2B length prefix - 2B modem header). Set "
               f"mesh_chunk>={_need} (one packet is always one frame; there is "
               f"no fragmentation layer), or shorten the payload.")
        log(f"[{CLIENT_ID}] FATAL: {msg}")
        print(json.dumps({"client_id": CLIENT_ID, "role": "node",
                          "node_id": node_id, "status": "fail",
                          "error": msg}), flush=True)
        return
    # traffic script: {"<src_id>": <dst_id>} -- who sends DATA to whom.
    #
    # Default is MUTUAL ({"1": 2, "2": 1}): both directions carry real payload.
    # A one-way default ({"1": 2}) only ever proved the FORWARD link plus its
    # ACK, so the reverse data path went untested on every OTA run to date --
    # and the reverse direction is not the same channel (a speaker->mic link is
    # not reciprocal, which is the whole reason the handshake measures both
    # directions separately).  Both endpoints get a data_dst here, which is the
    # case _link_initiator exists to disambiguate.
    #
    # This is one shared link either way -- `all_links` collapses {1:2, 2:1} to
    # the single unordered pair (1,2) -- so it costs one handshake, not two.
    # It does put two DATA senders in that link's window, which the one-way
    # default never did; CSMA arbitrates, but this is the knob to turn back to
    # {"1": 2} if a run shows the two directions colliding.
    data_map = {int(k): int(v)
                for k, v in PARAMS.get("data_map", {"1": 2, "2": 1}).items()}
    data_dst = data_map.get(node_id)                 # my DATA target (or None)
    expected_srcs = [s for s, d in data_map.items() if d == node_id]

    log(f"[{CLIENT_ID}] MESH NODE id={node_id} data_dst={data_dst} "
        f"expect_from={expected_srcs} phy_mode={phy_mode} "
        f"cap_fs={cap_fs} out_fs={out_fs} run={run_s}s")

    node_work = RUN_DIR / f"work_{CLIENT_ID}"
    node_work.mkdir(parents=True, exist_ok=True)
    # CSS is the control plane for every phy_mode (data plane too, under
    # css_only) -- these mirror announcelib.recipe()'s own PARAMS.get() calls
    # so the same keys work either way, but declared here explicitly because
    # the web UI's param schema is scanned per-file: a PARAMS.get() buried in
    # an imported module never surfaces as a UI field (see recipe() in
    # announcelib.py, which every OFDM-only test that also boots CSS relies on
    # without ever seeing these knobs in the UI).
    phy = dict(
        fs=int(out_fs),
        f_min=float(PARAMS.get("f_min", 400)),  # group: CSS PHY
        f_max=float(PARAMS.get("f_max", 6000)),  # group: CSS PHY
        symbol_dur=float(PARAMS.get("symbol_dur", 0.5)),  # group: CSS PHY
        amplitude=float(PARAMS.get("amplitude", 0.8)),  # group: CSS PHY
        symbol_gap_frac=float(PARAMS.get("symbol_gap_frac", 0.05)),  # group: CSS PHY
        chunk_size=int(PARAMS.get("chunk_size", 2)),  # group: CSS PHY
        diff_encode=bool(PARAMS.get("diff_encode", True)),  # group: CSS PHY
        rs_t=int(PARAMS.get("rs_t", 0)),  # group: CSS PHY
        conv_code=None,
    )
    amp_map = PARAMS.get("amp_map", {})
    if CLIENT_ID in amp_map:
        phy["amplitude"] = float(amp_map[CLIENT_ID])
    fw = ModemFirmware(str(node_work / "cfg.json"),
                       node_id=node_id, workdir=str(node_work),
                       phy_override=phy, verbose=True).boot()
    # only css_then_ofdm arranges handshakes over the CSS control plane
    fw.node.accept_ofdm_hs = (phy_mode == "css_then_ofdm")
    dec_info = bake_decode_info(cap_fs, str(node_work))
    daemon = ModemDaemon(
        fw,
        # Backoff is kept SMALL: collision avoidance is handled by the
        # wall-clock TX slots below, not by contention draws.  Every
        # arm-time-relative scheme tried OTA (cw 8/slot 0.1, 1.5, disjoint
        # cw 2/slot 3) collided, because nodes ARM at different times: two
        # releases can always land within the ~1-2s carrier-sense reaction
        # latency (4 collisions across 4 runs, 20260720).
        cw_s=float(PARAMS.get("cw_s", 0.8)),
        id_slot_s=float(PARAMS.get("id_slot_s", 0.0)),
        rechirp_min_s=1e6, rechirp_max_s=1e6,      # beacons OFF for this test
        first_beacon_stagger_s=1e6,
        ack_stagger_s=float(PARAMS.get("ack_stagger_s", 2.0)))

    # WALL-CLOCK TX SLOTS -- the actual collision-avoidance mechanism.
    # Every node may only BEGIN a transmission while `time.time() mod cycle`
    # is inside the first `tx_slot_open_frac` of its own id's slot; the rest
    # of the slot is a guard band.  Anchoring to the shared wall clock (LAN
    # NTP, skew << guard) makes any two nodes' TX starts >= guard seconds
    # apart BY CONSTRUCTION -- carrier sense then has time to see the first
    # transmission before the second node's window even opens.  This is what
    # local backoff cannot do: it is measured from each node's own arm time.
    # Cost: up to one cycle (n_ids * tx_slot_s) of latency per transmission,
    # trivial against 46-86s frames.  Carrier sense + the pre-TX freshness
    # check stay active on top (a slot grants the RIGHT to start, only into
    # confirmed-quiet air).
    tx_slot_s = float(PARAMS.get("tx_slot_s", 8.0))
    tx_slot_open_frac = float(PARAMS.get("tx_slot_open_frac", 0.5))
    slot_ids = sorted(int(v) for v in set(id_map.values()))
    slot_cycle = tx_slot_s * len(slot_ids)
    my_slot_t0 = slot_ids.index(node_id) * tx_slot_s
    log(f"  tx slots: cycle={slot_cycle:.0f}s mine=[{my_slot_t0:.0f},"
        f"{my_slot_t0 + tx_slot_s * tx_slot_open_frac:.0f})s of {slot_ids}")

    # HANDSHAKE MAC (ofdm_only) -- the OFDM probe/BAT exchange is a blocking
    # audio op that bypasses the data TX slots, so with 2+ links the nodes can
    # handshake over each other and negotiate garbage BATs.  Give each LINK a
    # dedicated wall-clock window long enough
    # for a full handshake; only its two endpoints handshake during it, so
    # links serialize and a double-duty node never runs two at once.  Both
    # endpoints derive the SAME schedule from data_map -> they enter together
    # with no signalling.  Links are the UNORDERED pairs from data_map.
    hs_window_s = float(PARAMS.get("hs_window_s", 150.0))
    # DYNAMIC handshake airtime budget: encode the actual probe / RobustLink
    # segments at OUR tx rate under the active negotiation overrides and
    # measure them, instead of assuming durations.  clean_s = one full 5-turn
    # transaction; max_tx_s = the longest single burst.  hs_gap_s is the
    # safety buffer added on top everywhere the budget is used.
    hs_gap_s = float(PARAMS.get("hs_gap_s", 8.0))
    hs_budget = _ml().hs_airtime_budget(out_fs)
    log(f"  hs airtime budget @{out_fs}Hz: clean={hs_budget['clean_s']}s "
        f"max_tx={hs_budget['max_tx_s']}s "
        f"(probe {hs_budget['probe_s']}s report {hs_budget['report_s']}s "
        f"ack {hs_budget['ack_s']}s go {hs_budget['go_s']}s) "
        f"gap={hs_gap_s:.0f}s")
    # retry waits sized from the budget instead of a flat 75s: a wait need
    # only outlast the peer's longest turn (probe+report, raw worst-case)
    # plus decode/cs margins -- the flat figure was most of the dead air on
    # every lost-message recovery.  (Budget is computed with RAW bodies, so
    # this stays conservative when hs_bat_rle shrinks the real messages.)
    if wait_msg <= 0:
        wait_msg = max(30.0, hs_budget["probe_s"] + hs_budget["report_s"]
                       + 18.0)
        log(f"  wait_msg derived from budget: {wait_msg:.0f}s")
    # guard band at the END of each window: a handshake STARTED here must
    # finish before the next link's window opens, else it runs past the
    # boundary and collides with the next link.  So refuse to ENTER in the
    # last hs_guard_s of the window.  Sized from the
    # measured budget (one clean transaction + gap); explicit hs_guard_s
    # param overrides.
    hs_guard_s = (float(PARAMS.get("hs_guard_s", 0.0))
                  or (hs_budget["clean_s"] + hs_gap_s))
    # the RESPONDER must be listening (in run_handshake) BEFORE the initiator's
    # first probe airs, or the probe is missed and both sit in a full
    # wait_probe timeout before retrying (poll jitter alone can delay window
    # entry enough to miss it).  So: responder enters at window start;
    # initiator delays hs_init_offset_s in, guaranteeing the responder is
    # already blocked-listening.
    # 6s (was 12): the responder is blocked-listening within ~2s of window
    # start (0.4s tick + handshake entry), so 6s still leaves 3x margin
    # while returning 6s/window to useful airtime.
    hs_init_offset_s = float(PARAMS.get("hs_init_offset_s", 6.0))
    all_links = sorted({tuple(sorted((int(s), int(d))))
                        for s, d in data_map.items()})
    hs_cycle = max(1.0, hs_window_s * len(all_links))

    # EPOCH ANCHOR: riding raw wall-clock (time.time() % cycle) starts the
    # run at a RANDOM schedule phase, which can leave a link's window with no
    # time for its DATA/ACK round trip.  The server stamps dispatch time into
    # PARAMS (sched_epoch, unix time so timezone-free; NTP skew across the
    # fleet is ~0.3s, negligible against 150s windows) and every node counts
    # its schedule from it: window 0 opens AT run start, links get their
    # windows in deterministic order.  Fallback for a missing stamp is local
    # boot time -- correct-ish phase, though nodes then skew by their boot
    # deltas.
    sched_epoch = float(PARAMS.get("sched_epoch", 0.0) or time.time())
    _sched_adv = [0.0]     # accumulated early-advance seconds (see below)

    def sched_phase(cycle: float) -> float:
        return (time.time() - sched_epoch + _sched_adv[0]) % cycle

    def link_window_open(peer: int) -> bool:
        """Per-link EXCLUSIVE AIRTIME: all of a link's traffic (handshake AND
        data) is confined to its own wall-clock window, so no two links are
        ever on the air together -- the only way to stop data on one link
        stepping on a handshake/data on another over one half-duplex channel."""
        key = tuple(sorted((node_id, peer)))
        if key not in all_links:
            return True                       # unscheduled link: no gating
        idx = all_links.index(key)
        phase = sched_phase(hs_cycle)
        start = idx * hs_window_s
        return start <= phase < start + hs_window_s

    # A short ACK may still air just after its link's window closes: if the
    # ACK clears MAC backoff+tx-slot gating a few seconds too late, holding
    # it until the window comes back around is a full cycle of dead air for
    # one short message.  The first hs_init_offset_s of the NEXT window is
    # listen-only (its initiator holds back so its responder can settle), so
    # a ~1.2s OFDM ACK there costs nothing.  Grace is capped accordingly.
    ack_grace_s = min(float(PARAMS.get("ack_grace_s", 10.0)),
                      max(0.0, hs_init_offset_s - 2.0))

    def link_window_grace(peer: int) -> bool:
        """True within ack_grace_s after this link's window closed."""
        key = tuple(sorted((node_id, peer)))
        if key not in all_links:
            return True
        end = all_links.index(key) * hs_window_s + hs_window_s
        closed_ago = (sched_phase(hs_cycle) - end) % hs_cycle
        return closed_ago < ack_grace_s

    def hs_window_open(peer: int, initiator: bool) -> bool:
        # handshake may only START in the first part of the window (guard band
        # at the end) so a ~60s handshake finishes before the next link's
        # window opens.  Data uses the full window (link_window_open).  The
        # initiator additionally waits hs_init_offset_s into the window so the
        # responder is listening first.
        key = tuple(sorted((node_id, peer)))
        if key not in all_links:
            return True
        idx = all_links.index(key)
        into = sched_phase(hs_cycle) - idx * hs_window_s
        if not (0 <= into < max(1.0, hs_window_s - hs_guard_s)):
            return False
        return into >= hs_init_offset_s if initiator else True

    if phy_mode in OFDM_DATA_MODES and all_links:
        mine = [k for k in all_links if node_id in k]
        log(f"  link airtime: cycle={hs_cycle:.0f}s window={hs_window_s:.0f}s "
            f"(hs first {hs_window_s - hs_guard_s:.0f}s) "
            f"links={all_links} mine={mine}")

    # HS-OVERRUN COLLISION GUARD.  hs_window_open only gates when a handshake
    # may START; run_handshake then blocks for up to hs_timeout (300s) -- twice
    # the 150s window -- so a struggling handshake keeps transmitting probes /
    # BAT retries deep into the NEXT link's window and collides with that
    # link's handshake.  A collided run only produces garbage, so: arm a timer at
    # handshake entry for the window boundary; if it fires while the handshake
    # is still running, the overrun IS the collision -- abort locally (the
    # abort_cb stops further TX) and broadcast on the coord bus so every node
    # aborts the run.
    hs_overrun_abort   = bool(float(PARAMS.get("hs_overrun_abort", 1)))
    hs_overrun_grace_s = float(PARAMS.get("hs_overrun_grace_s", 5.0))
    ABORT_SIG  = "abort_hs_overrun"
    # Broadcast when a node's scripted traffic has fully played out.  A
    # RECEIVER cannot tell "the sender is retransmitting" from "the sender has
    # gone home" by listening alone, so without this signal it sits out the
    # full retry window (ack_wait_s + 25s) after the last frame it heard, even
    # though the sender (whose own quiet test is vacuous, it expects nothing)
    # already exited.  This is the missing "I'm finished" edge.
    DONE_SIG   = "scenario_done"
    abort_evt  = threading.Event()
    abort_reason: list = [None]

    def flag_abort(msg: str):
        if abort_reason[0] is None:
            abort_reason[0] = msg
        abort_evt.set()

    def link_window_left(peer: int) -> float:
        """Wall-clock seconds until this link's exclusive window closes."""
        key = tuple(sorted((node_id, peer)))
        if key not in all_links:
            return float("inf")
        into = sched_phase(hs_cycle) - all_links.index(key) * hs_window_s
        return (hs_window_s - into) if 0 <= into < hs_window_s else 0.0

    def _abort_watch():
        # remote half of the guard: any node that detects an overrun signals
        # the coord bus; everyone else picks it up here and stops too
        while not abort_evt.is_set():
            who = [c for c in coord_check(ABORT_SIG) if c != CLIENT_ID]
            if who:
                flag_abort(f"peer node(s) {who} reported a handshake airtime "
                           f"overrun (cross-link collision) -- run aborted "
                           f"mesh-wide")
                return
            time.sleep(4.0)

    if hs_overrun_abort and phy_mode in OFDM_DATA_MODES and all_links:
        threading.Thread(target=_abort_watch, daemon=True).start()

    # EARLY WINDOW ADVANCE.  The wall-clock schedule strands the unused tail
    # of every window once a link finishes its handshake, DATA and ACK early.
    # When BOTH endpoints of the current
    # window's link have posted their business complete on the coord bus,
    # every node jumps the schedule to the next window boundary.  All nodes
    # compute the SAME jump from the server-stamped completion times (the
    # sched arithmetic uses only shared numbers), so slot/window alignment
    # is preserved to NTP accuracy.  Failed windows post nothing and run
    # out the wall clock exactly as before.
    sched_advance = str(PARAMS.get("sched_advance", "coord"))  # choices: coord, wallclock
    _adv_posted: set = set()
    _adv_applied: set = set()
    _adv_last_poll = [-1e9]

    def _win_business_done(key) -> bool:
        """My scripted transfers on this link are finished (sender: acked or
        gave up; receiver: payload delivered)."""
        involved = False
        for s, d in data_map.items():
            s, d = int(s), int(d)
            if tuple(sorted((s, d))) != key:
                continue
            involved = True
            if s == node_id:
                pol = policies.get(d)
                if not (pol and (pol.got_data_ack or pol.gave_up)):
                    return False
            elif d == node_id:
                if s not in got_data:
                    return False
        return involved

    def sched_advance_tick(now_t: float):
        # DEDUP IS PER LINK KEY, NOT PER (cycle, idx) SIGNAL.  Signal-keyed
        # dedup can feed back on itself: each applied advance pushes the
        # derived cycle number forward, minting a fresh signal name, so the
        # completed link's endpoints re-post "done" under every new name
        # (their business stays done forever) and every node advances again,
        # repeatedly slewing the schedule and skipping remaining links'
        # windows entirely.  The scripted scenario gives each link ONE
        # transfer, so one advance per link per run is the correct invariant.
        if sched_advance != "coord" or not all_links:
            return
        if now_t - _adv_last_poll[0] < 2.5:
            return
        _adv_last_poll[0] = now_t
        phase = sched_phase(hs_cycle)
        idx = min(int(phase // hs_window_s), len(all_links) - 1)
        key = all_links[idx]
        if key in _adv_applied:
            return
        sig = f"windone_w{idx}"
        if (node_id in key and key not in _adv_posted
                and _win_business_done(key)):
            coord_signal(sig)
            _adv_posted.add(key)
        st = coord_check_ts(sig)
        ep_ts = [t for cid, t in st.get("ts", {}).items()
                 if cid in id_map and int(id_map[cid]) in key]
        if len(ep_ts) >= 2:
            t_done = max(ep_ts)          # server clock, same for all pollers
            ph_done = (t_done - sched_epoch + _sched_adv[0]) % hs_cycle
            rem = (idx + 1) * hs_window_s - ph_done
            if 0.0 < rem < hs_window_s:
                _sched_adv[0] += rem
                _adv_applied.add(key)
                log(f"  t={now_s():6.1f} window {key} complete -- advancing "
                    f"schedule {rem:.1f}s early")

    rec = Recorder(in_dev, cap_fs)
    rec.start()
    # Stream the capture to disk as it arrives instead of saving it once at
    # the bottom of the run, so a stopped / killed / crashed node still
    # leaves a capture to diagnose from rather than nothing.
    rec.spill_to(RUN_DIR / f"{CLIENT_ID}_mesh.wav",
                 interval_s=float(PARAMS.get("capture_spill_s", 15.0)))
    on_shutdown(rec.close_spill)
    time.sleep(0.3)

    # carrier-sense calibration (validated cmodem approach, condensed)
    r = recipe(cap_fs)
    chirp_score, mf_win = _make_chirp_detector(r["f_min"], r["f_max"],
                                               r["symbol_dur"], cap_fs)
    smooth_n = int(PARAMS.get("smooth_n", 4))
    time.sleep(3.0)
    cal = rec.data()[-int(3.0 * cap_fs):]
    step = int(0.4 * cap_fs)
    cal_scores = [chirp_score(cal[i:i + mf_win])
                  for i in range(0, max(1, len(cal) - mf_win), step)]
    amb_mf = float(np.median(cal_scores)) if cal_scores else 0.035
    on_thr = min(float(PARAMS.get("carrier_thr_hi", 0.12)),
                 max(float(PARAMS.get("carrier_thr_lo", 0.05)),
                     carrier_k * amb_mf))
    off_thr = min(0.7 * on_thr, max(0.045, 1.3 * amb_mf))
    amb_rms = float(np.sqrt(np.mean(cal ** 2)))
    hold_s = float(PARAMS.get("busy_hold_s", 2.0))
    difs_s = float(PARAMS.get("difs_s", 3.0))
    log(f"  amb_mf={amb_mf:.4f} on_thr={on_thr:.4f} amb_rms={amb_rms:.5f}")

    # DEAF-NODE GUARD.  Carrier sense is the ONLY thing stopping this node
    # from transmitting over its neighbours.  A capture of pure silence
    # (wrong/muted input device) makes `busy` permanently False, so the node
    # defers to nobody and chirps straight through everyone else's frames --
    # with N devices sharing one channel a single deaf node poisons the whole
    # run.
    #
    # BUT: near-zero ambient is NOT proof of deafness.  Mics with firmware
    # noise blanking / AGC (RP2350 MicNode) gate a quiet room to digital
    # zero by design while passing real signals perfectly.  So on
    # silent ambient, arbitrate with a SELF-ECHO test: play a short tone
    # through our own speaker -- a node that can hear its own transmission
    # has working carrier sense regardless of its idle-floor reading; only
    # a node that can't is truly deaf and must not talk.
    if float(PARAMS.get("require_capture", 1)) and amb_rms < 1e-6:
        log(f"  ambient reads silent (amb_rms={amb_rms:.8f}) -- running "
            f"self-echo aliveness test (blanking/AGC mics idle at zero)")
        pre_n = len(rec.data())
        t_test = np.sin(2 * np.pi * 1500.0
                        * np.arange(int(0.5 * out_fs)) / out_fs)
        t_test *= 0.5 * np.hanning(len(t_test))
        play_blocking(t_test.astype(np.float32), out_fs, out_dev)
        time.sleep(0.5)
        echo = rec.data()[pre_n:]
        echo_pk = float(np.max(np.abs(echo))) if len(echo) else 0.0
        if echo_pk < 1e-4:
            msg = (f"capture is deaf (ambient silent AND self-echo test "
                   f"peak={echo_pk:.6f}) on input device {in_dev!r} -- "
                   f"carrier sense is non-functional, refusing to transmit. "
                   f"Select a working input device (or set require_capture=0 "
                   f"to force a deliberately deaf node).")
            log(f"[{CLIENT_ID}] FATAL: {msg}")
            rec.stop()
            print(json.dumps({"client_id": CLIENT_ID, "role": "node",
                              "node_id": node_id, "status": "fail",
                              "error": msg, "amb_rms": amb_rms}), flush=True)
            return
        log(f"  self-echo OK (peak={echo_pk:.4f}) -- capture alive, "
            f"ambient floor is blanked; using epsilon floor for TX guard")
        amb_rms = max(amb_rms, 1e-4)
    if amb_rms < 1e-4:
        log(f"  WARNING: very quiet capture (amb_rms={amb_rms:.6f}) -- "
            f"carrier sense may be insensitive to distant nodes")

    # ...and the OTHER end of the same failure.  A saturated / over-gained
    # input is deaf in a subtler way than a silent one: chirp_score is
    # energy-normalised, so a huge ambient floor divides real chirps toward
    # zero AND the tail_rms TX guard (4*amb_rms) moves out of reach of any
    # signal <= 1.0.  Such a node also transmits blind.  Observed OTA
    # 20260720: delllaptop came back from a mixer unmute at amb_rms=0.50
    # against 0.0006 / 0.0027 on its peers.
    amb_rms_max = float(PARAMS.get("amb_rms_max", 0.2))
    if float(PARAMS.get("require_capture", 1)) and amb_rms > amb_rms_max:
        msg = (f"capture is saturated (amb_rms={amb_rms:.5f} > {amb_rms_max}) "
               f"on input device {in_dev!r} -- ambient this loud means the "
               f"capture gain is clipping, which blinds carrier sense and "
               f"the TX guard.  Lower the input level (alsamixer capture, or "
               f"disable mic auto-gain), or raise amb_rms_max to override.")
        log(f"[{CLIENT_ID}] FATAL: {msg}")
        rec.stop()
        print(json.dumps({"client_id": CLIENT_ID, "role": "node",
                          "node_id": node_id, "status": "fail",
                          "error": msg, "amb_rms": amb_rms}), flush=True)
        return

    # -- state --------------------------------------------------------------
    t0 = time.time()
    score_hist: list = []
    busy = False
    below_since = None
    busy_since_frame = None
    idle_since = 0.0
    last_status = -1e9
    tx_log, rx_log, events_log = [], [], []
    ack_wait_s = float(PARAMS.get("ack_wait_s", 35.0))
    pol_kw = dict(
        phy_mode=phy_mode,
        hs_req_delay_s=hs_req_delay_s,
        req_retries=int(PARAMS.get("req_retries", 3)),
        req_ack_wait_s=float(PARAMS.get("req_ack_wait_s", 120.0)),
        data_retries=int(PARAMS.get("data_retries", 2)),
        ack_wait_s=ack_wait_s,
        link_silent_s=(float(PARAMS["link_silent_s"])
                       if PARAMS.get("link_silent_s") is not None else None),
        hs_retry_wait_s=float(PARAMS.get("hs_retry_wait_s", 20.0)),
        max_hs_attempts=int(PARAMS.get("max_hs_attempts", 4)),
        evm_high=float(PARAMS.get("evm_high", 70.0)),
        evm_low=float(PARAMS.get("evm_low", 18.0)),
        evm_margin_step=float(PARAMS.get("evm_margin_step", 3.0)),
        evm_margin_max=float(PARAMS.get("evm_margin_max", 18.0)),
        evm_climb=bool(PARAMS.get("evm_climb", 1)),
        evm_feedback=bool(PARAMS.get("evm_feedback", 0)))   # paused by default
    base_margin = float(_hs_ov["snr_margin_db"])
    # a RE-negotiation must fail fast: the peer may have finished and left, so
    # spending the full first-contact handshake timeout (~600s) per doomed
    # cycle is what turns a coordination miss into minutes of dead air.
    reneg_hs_timeout = float(PARAMS.get("reneg_hs_timeout", 90.0))
    reneg_wait_probe = float(PARAMS.get("reneg_wait_probe_s", 40.0))
    # minimum usable link rate: below this a mesh packet becomes a MANY-second
    # OFDM frame that overruns the TX slots/windows, and the slow one-frame
    # round trip lets the two ends DESYNC (receiver drops+re-handshakes while
    # the sender is still mid-frame), so they talk over each other.  A link
    # that can't clear the floor is unusable, so reject it and let the mode
    # give up rather than collapse into that regime.
    min_link_bps = float(PARAMS.get("min_link_bps", 250.0))

    policies: dict[int, PhyPolicy] = {}      # peer id -> policy

    def get_policy(peer: int) -> PhyPolicy:
        if peer not in policies:
            policies[peer] = PhyPolicy(node_id, peer, initiator=False,
                                       **pol_kw)
        return policies[peer]

    if data_dst is not None:                 # my one upgrade-eligible link
        # See _link_initiator: a MUTUAL data_map (both sides sending) needs a
        # tie-break, or both endpoints enter the shared handshake as
        # "initiator" at once.
        policies[data_dst] = PhyPolicy(
            node_id, data_dst,
            initiator=_link_initiator(data_map, node_id, data_dst), **pol_kw)
        policies[data_dst].data_pending = True
    if phy_mode in OFDM_DATA_MODES:
        # roles are STATIC here (no CSS negotiation to create them lazily):
        # whoever is scripted to receive must stand up its responder policy
        # at boot so it enters the handshake on its own clock
        for s in expected_srcs:
            get_policy(s)

    hs_pending: list[int] = []       # peers we owe a RESPONDER handshake
    req_raws: dict[int, tuple] = {}  # peer -> (raw, tag): SAME REQ on retry
    ofdm_listen_start = 0
    last_ofdm_poll = -1e9
    last_ofdm_rx_t = None
    last_css_rx_t = None
    last_detect_t = -1e9
    got_data: dict[int, bytes] = {}  # src node id -> delivered payload

    def now_s():
        return time.time() - t0

    def reset_carrier():
        nonlocal busy, below_since, busy_since_frame, idle_since, score_hist
        busy = False
        below_since = None
        busy_since_frame = None
        idle_since = time.time() - t0
        score_hist = []

    def enter_handshake(role, peer, reneg=False):
        nonlocal ofdm_listen_start
        pol = get_policy(peer)
        # closed-loop: apply this peer's learned margin_adj to the allocation
        # for THIS (re)negotiation.  The receiver's margin raises the peer's
        # TX BAT robustness, so the direction that showed high EVM is exactly
        # the one made more conservative.  Serialized handshakes make the
        # global override safe.
        _hs_ov["snr_margin_db"] = base_margin + pol.margin_adj
        _ofdm_hs.HS_OVERRIDES = _hs_ov
        hs_to = reneg_hs_timeout if reneg else hs_timeout
        # TRUE overrun fix: the retry machinery may not carry TX past the
        # window boundary, so the timeout is clamped to the wall-clock time
        # left in this link's window minus the longest single burst (an
        # already-playing segment must still END inside the window) and the
        # gap.  With this, the overrun timer below is a pure backstop.
        if (phy_mode in OFDM_DATA_MODES
                and tuple(sorted((node_id, peer))) in all_links):
            left = link_window_left(peer)
            clamp = max(0.0, left - hs_budget["max_tx_s"] - hs_gap_s)
            if clamp < hs_to:
                log(f"  t={now_s():6.1f} hs timeout clamped "
                    f"{hs_to:.0f}s -> {clamp:.0f}s "
                    f"(window_left={left:.0f}s, max_tx="
                    f"{hs_budget['max_tx_s']:.1f}s, gap={hs_gap_s:.0f}s)")
                hs_to = clamp
        wp = reneg_wait_probe if reneg else wait_probe
        log(f"  t={now_s():6.1f} === OFDM HANDSHAKE ({role}"
            f"{',reneg' if reneg else ''}) peer={peer} "
            f"margin={base_margin + pol.margin_adj:.0f}dB "
            f"(base {base_margin:.0f}+{pol.margin_adj:.0f}) ===")
        overrun_timer = None
        if (hs_overrun_abort and phy_mode in OFDM_DATA_MODES
                and tuple(sorted((node_id, peer))) in all_links):
            left = link_window_left(peer)

            def _overrun(p=peer, w=left):
                log(f"  t={now_s():6.1f} *** HS OVERRUN peer={p}: handshake "
                    f"still running {hs_overrun_grace_s:.0f}s past its "
                    f"window boundary (had {w:.0f}s left at entry) -- "
                    f"cross-link collision, aborting run mesh-wide ***")
                flag_abort(f"handshake with peer {p} overran its "
                           f"{hs_window_s:.0f}s airtime window -- it was "
                           f"transmitting in another link's window "
                           f"(cross-link collision)")
                coord_signal(ABORT_SIG)

            overrun_timer = threading.Timer(left + hs_overrun_grace_s,
                                            _overrun)
            overrun_timer.daemon = True
            overrun_timer.start()
        try:
            res = _ml().run_handshake(role, rec, cap_fs, out_fs, out_dev,
                                      amplitude=ofdm_amp, timeout=hs_to,
                                      wait_probe_s=wp,
                                      wait_msg_s=wait_msg,
                                      poll_s=hs_poll_s,
                                      skip_fin=hs_skip_fin,
                                      amb_rms=amb_rms, log=log,
                                      lookback_s=hs_lookback,
                                      abort_cb=abort_evt.is_set)
        finally:
            if overrun_timer is not None:
                overrun_timer.cancel()
        # Own-TX spans, in RUN time, so the post-hoc capture analysis can tell
        # self-coupling apart from a hot input: a node hears its own speaker at
        # full scale, which reads as sustained clipping per handshake window
        # and looks identical to an over-driven mic.
        for w in (res.get("tx_windows") or []):
            try:
                tx_windows.append({"t": round(float(w["wall"]) - t0, 1),
                                   "dur_s": float(w["dur_s"]),
                                   "what": w.get("what"), "peer": peer})
            except (TypeError, ValueError, KeyError):
                pass
        lk = (_ml().make_link(res, out_fs, symmetric=link_symmetric)
              if (res["ok"] and res["peer_rate"]) else None)
        if lk is not None:
            tx_act = int(np.sum(lk["tx_bat"] > 0))
            rx_act = int(np.sum(lk["rx_bat"] > 0))
            gross = min(int(lk["tx_gross_bps"]), int(lk["rx_gross_bps"]))
            if tx_act == 0 or rx_act == 0:
                # A one-way link is useless: even a pure receiver must send
                # ACKs, so a transfer can never complete, and decoding an
                # empty-BAT ACK can crash the peer.  Reject it so the mode's
                # own recovery (re-handshake / CSS fallback) runs.
                log(f"  t={now_s():6.1f} === HANDSHAKE degenerate peer={peer}: "
                    f"tx_active={tx_act} rx_active={rx_act} (one-way channel) "
                    f"-- rejecting link ===")
                res = {**res, "ok": False,
                       "state": f"one_way_tx{tx_act}_rx{rx_act}"}
                lk = None
            elif gross < min_link_bps:
                # too slow to be usable -- see min_link_bps
                log(f"  t={now_s():6.1f} === HANDSHAKE below floor peer={peer}: "
                    f"{gross}bps < {min_link_bps:.0f} -- rejecting link ===")
                res = {**res, "ok": False, "state": f"below_floor_{gross}bps"}
                lk = None
        if lk is not None:
            log(f"  t={now_s():6.1f} === OFDM LINK UP peer={peer}"
                f"{' [sym]' if lk.get('symmetric') else ''}: "
                f"tx {lk['tx_gross_bps']}bps @{lk['tx_rate']}  "
                f"rx {lk['rx_gross_bps']}bps @{lk['rx_rate']} ===")
            # A queued OFDM_HS_REQ for THIS peer is now obsolete -- it asks for
            # the handshake that just succeeded.  Airing it costs a full CSS
            # packet (~33s at these params) of dead air, and it sits AHEAD of
            # the data the new link exists to carry: observed 20260813_040605,
            # link up at t=337.7 and the stale REQ still went out at t=353.6.
            # Worse, the peer answers a REQ by re-entering the handshake, so a
            # late one can tear down the link that just came up.
            stale = daemon.drop(f"ofdmhs:{peer}:")
            if stale:
                log(f"  t={now_s():6.1f} dropped {len(stale)} obsolete "
                    f"OFDM_HS_REQ for peer={peer} (link is up): {stale}")
            # Frame airtime is only knowable NOW: it needs the negotiated rate
            # AND mesh_chunk.  A frame longer than a TX slot cannot be sent
            # inside one, and a big mesh_chunk on a slow link gets there easily
            # (1508B at 2756bps is ~8s), so say it while there is still time to
            # lower mesh_chunk or raise tx_slot_s.
            try:
                _full_s = (len(_ml().encode_packet_ofdm(lk, bytes(mesh_chunk - 2)))
                           / max(int(lk["tx_rate"]), 1))
                log(f"  data frame: mesh_chunk={mesh_chunk}B "
                    f"({mesh_chunk - 4}B payload) -> {_full_s:.1f}s on air, "
                    f"goodput ~{(mesh_chunk - 4) * 8 / max(_full_s, 1e-6):.0f}bps")
                if _full_s > tx_slot_s:
                    log(f"  WARNING: a full data frame ({_full_s:.1f}s) is "
                        f"longer than this node's TX slot "
                        f"({tx_slot_s:.1f}s) -- it cannot be sent inside one "
                        f"slot. Lower mesh_chunk or raise tx_slot_s.")
            except Exception as _e:                       # noqa: BLE001
                log(f"  data frame: airtime estimate unavailable ({_e})")
        else:
            log(f"  t={now_s():6.1f} === HANDSHAKE FAILED peer={peer}: "
                f"{res['state']} ===")
        handle_policy(pol, pol.on_hs_result(res, lk, now_s()))
        ofdm_listen_start = len(rec.data())
        reset_carrier()

    def handle_policy(pol, acts):
        for a in acts:
            peer = pol.peer_id
            if a["kind"] == "send_req":
                if peer not in req_raws:
                    req_raws[peer] = fw.node.request_ofdm_hs(peer)
                raw, tag = req_raws[peer]
                daemon.enqueue(raw, tag)
                pol.on_req_queued(int(tag.split(":")[2]))
                log(f"  t={now_s():6.1f} queued OFDM_HS_REQ -> {peer} "
                    f"(seq {pol.hs_req_seq}, try {pol.req_tries}"
                    f"/{1 + pol.req_retries})")
            elif a["kind"] == "send_data":
                raw, tag = fw.node.send_data(peer, data_payload)
                daemon.enqueue(raw, tag)
                pol.on_data_queued(int(tag.split(":")[2]))
                log(f"  t={now_s():6.1f} queued DATA({data_payload!r}) "
                    f"-> {peer} (try {pol.data_tries}"
                    f"/{1 + pol.data_retries}, phy={pol.final_phy})")
            elif a["kind"] == "enter_hs":
                enter_handshake(a["role"], peer, reneg=a.get("reneg", False))
            elif a["kind"] == "gave_up":
                events_log.append({"t": round(now_s(), 1), "kind": "gave_up",
                                   "peer": peer,
                                   "hs_attempts": pol.hs_attempts})
                log(f"  t={now_s():6.1f} === GAVE UP on OFDM peer={peer} "
                    f"after {pol.hs_attempts} handshake attempts "
                    f"(payload likely delivered; reverse ACK channel failed) ===")
            elif a["kind"] == "arm_responder":
                if peer not in hs_pending:   # arm; enter after our ACK airs
                    hs_pending.append(peer)
            elif a["kind"] == "link_down":
                req_raws.pop(peer, None)     # fresh REQ if re-armed
                events_log.append({"t": round(now_s(), 1), "kind": "link_down",
                                   "peer": peer, "reason": a["reason"],
                                   "margin_adj": pol.margin_adj})
                nxt = ("re-negotiate" if (pol.phy_mode in OFDM_DATA_MODES
                       or pol.upgrade_wanted) else "CSS")
                log(f"  t={now_s():6.1f} === OFDM LINK DOWN peer={peer} "
                    f"({a['reason']}) -> {nxt} "
                    f"[margin_adj={pol.margin_adj:.0f}dB] ===")

    def handle_events():
        for e in daemon.drain_events():
            events_log.append({"t": round(now_s(), 1), **{k: v for k, v in
                               e.items() if k != "packet"}})
            log(f"  t={now_s():6.1f} EVENT {e['kind']} peer={e.get('peer')}")
            if e["kind"] == "ofdm_hs_req":
                pol = get_policy(e["peer"])
                handle_policy(pol, pol.on_hs_req(now_s()))
            elif e["kind"] == "ack" and e["peer"] in policies:
                pol = policies[e["peer"]]
                handle_policy(pol, pol.on_ack(e["seq"], now_s()))
            elif e["kind"] == "deliver":
                p = e.get("packet") or {}
                if "src" in p:
                    got_data[p["src"]] = p.get("payload")

    valid_ids = set(int(v) for v in id_map.values())

    def decode_css_burst(seg):
        nonlocal last_css_rx_t
        try:
            res = decode_array(seg, cap_fs, dec_info)
        except Exception as exc:
            log(f"  decode error: {exc}")
            return
        frames = [f for f in res.get("frames", []) if "bytes_hex" in f]

        # REASSEMBLE PHY chunks into modem packets.  chunk_size (2) is
        # smaller than a DATA packet, so one packet spans several PHY frames;
        # each frame carries the encoder's chunk seq (0,1,2,...).  Treating
        # every frame as a whole packet parses PAYLOAD bytes as headers,
        # producing a phantom packet whose neighbour-discovery auto-ACK can
        # saturate the channel for the entire run.
        msgs = []                      # (joined_bytes, first_frame_dict)
        cur = prev = None
        for f in frames:
            s = int(f.get("seq", 0))
            if s == 0:                             # chunk 0 starts a packet
                if cur is not None:
                    msgs.append(cur)
                cur, prev = [bytearray(bytes.fromhex(f["bytes_hex"])), f], 0
            elif cur is not None and s == prev + 1:
                cur[0] += bytes.fromhex(f["bytes_hex"])
                prev = s
            else:                                  # gap: packet is corrupt
                if cur is not None:
                    log(f"  css chunk gap (got seq {s}) -- partial dropped")
                cur = prev = None
        if cur is not None:
            msgs.append(cur)

        for joined, f0 in msgs:
            # strip the encoder's zero padding of the final chunk (modem
            # packets carry no length field; test payloads are ASCII so
            # trailing NULs are never meaningful here)
            raw = bytes(joined).rstrip(b"\x00")
            if len(raw) < 2:
                continue
            try:
                p = cm.unpack_packet(raw)
            except cm.ModemError:
                continue
            if p["src"] not in valid_ids:
                # belt-and-braces vs any other mis-parse: never let a packet
                # from an id outside this run's roster into the MAC, where
                # discovery would auto-ACK it on air
                log(f"  t={now_s():6.1f} RX css DROP src={p['src']} "
                    f"(not in roster {sorted(valid_ids)})")
                continue
            rx_log.append({"t": round(now_s(), 1), "phy": "css",
                           "src": p["src"], "type": p["type_name"],
                           "seq": p["seq"]})
            log(f"  t={now_s():6.1f} RX css {p['type_name']} "
                f"src={p['src']} seq={p['seq']}")
            if p["src"] != node_id:
                last_css_rx_t = now_s()
                pol = policies.get(p["src"])
                if pol is not None:
                    # implicit downgrade FIRST, so any auto-ACK this
                    # packet triggers rides CSS, not the dead OFDM link.
                    # Only traffic addressed to US counts -- the peer
                    # talking CSS to a css_only third party says nothing
                    # about our link.
                    handle_policy(pol, pol.on_css_rx_from_peer(
                        now_s(), to_me=(p["dst"] == node_id)))
            daemon.handle_frame({**f0, "bytes_hex": raw.hex()}, now_s())
        handle_events()

    def linked_policies():
        return [p for p in policies.values() if p.link is not None]

    # INPUT-CLIPPING GUARD.  The boot guards catch a capture that is silent or
    # one whose AMBIENT is saturated; neither catches ambient perfectly
    # healthy but the peer's DATA frames arriving at full scale and clipping.
    # Clipping is broadband distortion -- a 5s robust probe shrugs it off
    # (coh=1.000) while a dense short frame locks and fails RS every time, so
    # the run looks like a handshake problem and is really a level problem.
    clip_events: list = []
    rx_level_worst = {"peak": 0.0, "clip_pct": 0.0, "t": None}
    clip_warned = [False]
    # Spans where THIS node was transmitting (run time).  Filled by
    # handle_hs_result from run_handshake's report and by the data TX path; the
    # report uses them to attribute self-coupling rather than blame input gain.
    tx_windows: list = []

    def note_rx_level(win, tag: str, t: float) -> dict:
        lv = level_stats(win, cap_fs)
        if lv["peak"] > rx_level_worst["peak"]:
            rx_level_worst.update(peak=lv["peak"], clip_pct=lv["clip_pct"],
                                  t=round(t, 1))
        if lv["clip_pct"] >= CLIP_PCT_WARN:
            # Attribute it before advising anything: environmental impulse
            # noise and sustained input overdrive need opposite responses,
            # and look identical without this classification.
            impulsive = (lv["crest_db"] >= 20.0 or lv["hot_ms"] <= 20.0)
            clip_events.append({"t": round(t, 1), "tag": tag,
                                "kind": "impulse" if impulsive else "level",
                                **lv})
            if impulsive:
                log(f"  t={t:6.1f} clipped samples in the window "
                    f"({lv['clip_pct']:.2f}% at full scale, peak "
                    f"{lv['peak']:.3f}) but the shape is IMPULSIVE "
                    f"(crest {lv['crest_db']:.0f}dB, {lv['hot_ms']:.0f}ms "
                    f"loud) -- ambient impulse noise, not a level problem. "
                    f"Costs the frames it lands on; gain changes will not "
                    f"help.")
            elif not clip_warned[0]:
                clip_warned[0] = True
                log(f"  t={t:6.1f} *** INPUT CLIPPING: received burst peaks "
                    f"at {lv['peak']:.3f} with {lv['clip_pct']:.2f}% of "
                    f"samples at full scale (rms={lv['rms']:.3f}, crest "
                    f"{lv['crest_db']:.0f}dB SUSTAINED) -- the ADC "
                    f"is saturating on the peer's frames.  Dense OFDM will "
                    f"lock and fail RS while robust probes still decode. "
                    f"Lower this node's INPUT gain or the peer's "
                    f"ofdm_amplitude/amp_map, or move them apart ***")
        return lv

    def peer_summary(pol):
        lk = pol.link
        # `link` is the CURRENT link and becomes None on teardown, which loses
        # the negotiated table -- so a run that ended disconnected could not be
        # analysed afterwards at all (the constellation/EVM tooling needs the
        # table the frames were sent with).  Keep the last one that existed.
        last = lk if lk is not None else getattr(pol, "_last_link_snapshot", None)
        if lk is not None:
            pol._last_link_snapshot = lk
        return {
            "final_phy": pol.final_phy,
            "data_ack": pol.got_data_ack,
            "gave_up": pol.gave_up,
            "margin_adj": pol.margin_adj,
            "evm_events": pol.evm_events,
            "hs_attempts": pol.hs_attempts,
            "fallback_events": pol.fallback_events,
            "hs": (None if pol.hs_result is None else {
                k: v for k, v in pol.hs_result.items()
                if k in ("ok", "state", "elapsed_s", "probe_rx", "peer_rate")}),
            "last_link": (None if last is None else {
                "tx_rate": last["tx_rate"], "rx_rate": last["rx_rate"],
                "tx_bat": last["tx_bat"].tolist(),
                "rx_bat": last["rx_bat"].tolist(),
            }),
            "link": (None if lk is None else {
                "tx_gross_bps": lk["tx_gross_bps"],
                "rx_gross_bps": lk["rx_gross_bps"],
                "tx_rate": lk["tx_rate"], "rx_rate": lk["rx_rate"],
                "tx_active": int(np.sum(lk["tx_bat"] > 0)),
                "rx_active": int(np.sum(lk["rx_bat"] > 0)),
                "tx_bat": lk["tx_bat"].tolist(),    # persisted for diagnosis
                "rx_bat": lk["rx_bat"].tolist(),
            }),
        }

    def build_result(interrupted: bool = False) -> dict:
        """Snapshot the run as a result dict.

        Defined BEFORE the loop so the SIGTERM hook can call it: a node that
        is stopped mid-run now reports what it had instead of vanishing from
        the run directory entirely (20260728).
        """
        send_ok = data_dst is None or policies[data_dst].got_data_ack
        recv_ok = all(s in got_data for s in expected_srcs)
        ok = send_ok and recv_ok and (data_dst is not None or expected_srcs)
        any_link = bool(linked_policies())
        aborted = abort_evt.is_set()
        status = ("interrupted" if interrupted
                  else "fail" if aborted
                  else "ok" if ok
                  else "partial" if any_link
                  else "fail")
        return {
            "client_id": CLIENT_ID, "role": "node", "node_id": node_id,
            "status": status,
            **({"interrupted": True, "elapsed_s": round(now_s(), 1)}
               if interrupted else {}),
            **({"aborted": True, "error": abort_reason[0]} if aborted else {}),
            "phy_mode": phy_mode,
            "capture": {"frames": int(rec.frames_captured),
                        "seconds": round(rec.frames_captured / max(cap_fs, 1), 1),
                        "overflows": int(rec.overflows),
                        "alive": bool(rec.alive),
                        "amb_rms": round(amb_rms, 6),
                        "rx_peak_worst": rx_level_worst,
                        "clipping": bool(clip_events),
                        "clip_events": clip_events[:20],
                        "tx_windows": tx_windows[:60]},
            "ofdm_negotiation": {"snr_margin_db_base": base_margin,
                                 "max_bits_per_bin": _hs_ov["max_bits_per_bin"],
                                 "snr_estimator": _hs_ov["snr_estimator"],
                                 "link_rate_policy": link_rate_policy,
                                 "evm_feedback": bool(PARAMS.get("evm_feedback", 0))},
            "data_dst": data_dst,
            "expected_srcs": expected_srcs,
            "peers": {str(p): peer_summary(pol) for p, pol in policies.items()},
            "data_delivered": {
                str(src): (pl.decode("utf-8", "replace")
                           if isinstance(pl, (bytes, bytearray)) else None)
                for src, pl in got_data.items()},
            "tx": tx_log, "rx": rx_log, "events": events_log,
        }

    def emit_result(result: dict) -> None:
        (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(
            json.dumps(result, indent=2))
        # log(), not print(): identical on stdout, but signal-safe when this
        # runs from the shutdown hook (see testlib.log)
        log(json.dumps(result))

    def _shutdown_result():
        log(f"  t={now_s():6.1f} *** TERMINATED mid-run -- writing partial "
            f"result ***")
        emit_result(build_result(interrupted=True))

    on_shutdown(_shutdown_result)

    # capture-liveness guard: the self-echo deaf test above runs ONCE at boot
    # and only when ambient reads silent, so a capture that dies at t=200
    # (worker crash, device yanked, host audio reset) was invisible -- the
    # node transmitted on schedule into a frozen buffer for the rest of the
    # run and blamed the peer for never ACKing (20260728).  Watch the frame
    # counter: capture delivers every ~0.1s, so no growth for this long means
    # it is gone.  A blocking handshake does not trip it -- the counter jumps
    # forward when the loop resumes.
    capture_stall_s = float(PARAMS.get("capture_stall_s", 12.0))
    cap_last_n = int(rec.frames_captured)
    cap_last_t = time.time()
    # DONE_SIG poll state: the check is an HTTP round trip, so it only runs
    # once our own business is done and then only every couple of seconds.
    last_done_poll = [-1e9]
    done_seen = [False]
    dst_gone  = [False]   # my data_dst announced DONE -- no ACK is ever coming

    # -- main loop ------------------------------------------------------------
    while time.time() - t0 < run_s:
        if abort_evt.is_set():
            log(f"  t={now_s():6.1f} *** RUN ABORTED: {abort_reason[0]} ***")
            break
        now = now_s()
        buf = rec.data()
        n = len(buf)
        sched_advance_tick(now)

        # capture liveness (see capture_stall_s above)
        fc = int(rec.frames_captured)
        if fc > cap_last_n:
            cap_last_n, cap_last_t = fc, time.time()
        elif not rec.alive or (capture_stall_s > 0
                               and time.time() - cap_last_t > capture_stall_s):
            why = ("capture worker died" if not rec.alive
                   else f"no new samples for "
                        f"{time.time() - cap_last_t:.0f}s")
            log(f"  t={now:6.1f} *** CAPTURE STALLED: {why} "
                f"({fc} frames = {fc / max(cap_fs, 1):.1f}s captured) -- "
                f"this node is deaf; carrier sense and every decode are "
                f"dead, aborting instead of transmitting blind ***")
            flag_abort(f"input capture stalled at t={now:.0f}s ({why}) -- "
                       f"node went deaf mid-run")
            coord_signal(ABORT_SIG)
            continue

        # early finish once MY part of the scenario has fully played out:
        # my DATA (if any) acked, everything I was scripted to receive
        # delivered, nothing queued, and the air quiet for a retry window.
        # Pure bystanders (no send, no expected rx) stay up the whole run.
        if data_dst is not None or expected_srcs:
            last_rx_t = max((t for t in (last_ofdm_rx_t, last_css_rx_t)
                             if t is not None), default=None)
            timed_out = (last_rx_t is not None
                         and now - last_rx_t > ack_wait_s + 25)
            # Poll the done-bus whenever ANYTHING is still outstanding -- not
            # only once my own send has concluded. Gating this behind
            # `send_done and recv_done` makes it unreachable in the one case
            # it exists for: a peer that left while I was still retrying.
            #
            # This only matters once data_map is MUTUAL. One-way traffic hides
            # it: the pure receiver has no data_dst, so send_done is always
            # True and it always polls; the pure sender has no expected_srcs,
            # so `quiet` is always True and it never needs to. With both ends
            # sending, whichever node's ACK lags gets stranded against a peer
            # that has already gone.
            if now - last_done_poll[0] > 2.0:
                last_done_poll[0] = now
                done_clients = set(coord_check(DONE_SIG))
                if expected_srcs:
                    done_seen[0] = all(id_to_client.get(int(s)) in done_clients
                                       for s in expected_srcs)
                dst_gone[0] = (data_dst is not None
                               and id_to_client.get(int(data_dst))
                               in done_clients)
            _dpol = policies[data_dst] if data_dst is not None else None
            stop, why = _scenario_exit(
                data_dst, expected_srcs, got_data,
                acked=bool(_dpol and _dpol.got_data_ack),
                gave_up=bool(_dpol and _dpol.gave_up),
                dst_gone=dst_gone[0], done_seen=done_seen[0],
                timed_out=timed_out, outbox_empty=not daemon.outbox)
            if stop:
                log(f"  t={now:6.1f} scenario complete "
                    f"(recv={sorted(got_data)}; {why})")
                # Tell the other end before leaving, so a peer still sitting in
                # its retry window can stop too.
                coord_signal(DONE_SIG)
                break

        # CSS carrier sense (chirp-matched, smoothed)
        score_hist.append(chirp_score(buf[max(0, n - mf_win):n]))
        if len(score_hist) > smooth_n:
            score_hist.pop(0)
        sc = float(np.median(score_hist))
        if not busy:
            if sc > on_thr:
                busy = True
                busy_since_frame = max(0, n - mf_win)
                below_since = None
        else:
            if sc < off_thr:
                if below_since is None:
                    below_since = now
                elif now - below_since >= hold_s:
                    busy = False
                    burst_len = (n - (busy_since_frame or n)) / cap_fs
                    if burst_len >= min_burst_s:
                        # decode from the BURST'S OWN START (carrier sense
                        # already tracks it), not a fixed lookback: a fixed
                        # window beheads any burst longer than itself, losing
                        # chunk 0 on every try and dropping the whole packet.
                        # decode_lookback_s remains only as a cost cap.
                        lo = max(0,
                                 (busy_since_frame or 0) - int(3.0 * cap_fs),
                                 n - int(decode_lookback_s * cap_fs))
                        log(f"  t={now:6.1f} css burst {burst_len:.1f}s -> decode")
                        decode_css_burst(buf[lo:n])
                    busy_since_frame = None
                    below_since = None
                    idle_since = time.time() - t0
            else:
                below_since = None

        # responder: our CSS ACK for the HS_REQ has aired once the outbox no
        # longer holds it AND we're not mid-TX (TX below is blocking).
        # One handshake at a time; stale entries (link already up) drop out.
        if hs_pending:
            peer = hs_pending[0]
            pol = get_policy(peer)
            if pol.link is not None or pol.hs_result is not None:
                hs_pending.pop(0)
            else:
                holds_ack = any(t.startswith("ack:")
                                for _, t, _ in daemon.outbox)
                if not holds_ack and not busy:
                    hs_pending.pop(0)
                    enter_handshake("responder", peer)
                    continue

        # OFDM RX poll (any link up): scan recent capture for mesh frames.
        # ONLY try the link whose airtime window is open: per-link airtime
        # means just that link's peer can be transmitting now, so its BAT is
        # the right one and any EVM is attributable to IT.  Trying every link's
        # BAT mis-decodes -- ALL mesh frames share the preamble seed, so a
        # frame LOCKS on any BAT and RS-fails, feeding a spurious evm=100 to a
        # BYSTANDER link, which can trigger a bogus back-off/re-negotiation.
        # scheduled nodes: decode ONLY the link whose window is open right now
        # (empty when it's another link's window -> we stay silent AND deaf, so
        # we don't mis-attribute a neighbour link's frame to ours and back off
        # spuriously).  Unscheduled (2-node / no data_map): all.
        if all_links:
            rx_pols = [lp for lp in linked_policies()
                       if link_window_open(lp.peer_id)]
        else:
            rx_pols = linked_policies()
        if rx_pols and now - last_ofdm_poll >= ofdm_poll_s:
            last_ofdm_poll = now
            cap = buf[ofdm_listen_start:]
            max_n = int(30.0 * cap_fs)
            if len(cap) > max_n:
                ofdm_listen_start += len(cap) - max_n
                cap = cap[-max_n:]
            if len(cap) > int(3.0 * cap_fs):
                for lp in rx_pols:
                    pkt_bytes, st = _ml().try_decode_packet_ofdm(
                        lp.link, cap, cap_fs)
                    # A frame whose preamble is here but whose body has not
                    # finished arriving is NOT a failure: keep the capture and
                    # let the next poll see the whole thing.  Treating it as an
                    # RS failure permanently loses every frame longer than the
                    # poll interval -- with mesh_chunk=512 that is most of them.
                    if pkt_bytes is None and st and st.get("incomplete"):
                        log(f"  t={now:6.1f} ofdm frame still ARRIVING "
                            f"(coh={st.get('acq_coh')}, have {st.get('have_s')}s "
                            f"of {st.get('need_s')}s) -- waiting for the rest")
                        last_ofdm_rx_t = now      # the peer IS transmitting
                        break
                    if pkt_bytes is None and st and st.get("rs_failed"):
                        lv = note_rx_level(cap, "rs_failed", now)
                        log(f"  t={now:6.1f} ofdm frame LOCKED "
                            f"(coh={st.get('acq_coh')}) but RS failed "
                            f"[rx peak={lv['peak']:.3f} rms={lv['rms']:.3f} "
                            f"clip={lv['clip_pct']:.2f}%]")
                        # Skip past just this frame.  Advancing to the live edge
                        # also discarded whatever had begun arriving during the
                        # poll, turning one bad frame into two lost ones.
                        fe, fr = st.get("frame_end"), st.get("rate")
                        if fe and fr:
                            ofdm_listen_start += int(fe * cap_fs / fr)
                            ofdm_listen_start = min(ofdm_listen_start,
                                                    len(rec.data()))
                        else:
                            ofdm_listen_start = len(rec.data())
                        last_ofdm_rx_t = now                 # peer still trying
                        # locked-but-undecodable = worst-case EVM: strongest
                        # signal that this direction's BAT is too hot
                        handle_policy(lp, lp.on_rx_evm(100.0, now))
                        break
                    if pkt_bytes is not None:
                        ofdm_listen_start = len(rec.data())
                        last_ofdm_rx_t = now
                        lv = note_rx_level(cap, "decoded", now)
                        try:
                            p = cm.unpack_packet(pkt_bytes)
                            evm = st.get("evm_pct")
                            lp.on_good_rx(now)   # link-silence clock
                            # evm_pct_1pass is present only when the two-pass
                            # data-aided estimate was kept; the pair says how
                            # much of the error was channel-ESTIMATE noise
                            # rather than the channel.
                            rx_log.append({"t": round(now, 1), "phy": "ofdm",
                                           "src": p["src"],
                                           "type": p["type_name"],
                                           "seq": p["seq"],
                                           "evm_pct": evm,
                                           "evm_pct_1pass": st.get("evm_pct_1pass"),
                                           "rs_errata": st.get("rs_errata"),
                                           "level": lv})
                            log(f"  t={now:6.1f} RX ofdm {p['type_name']} "
                                f"src={p['src']} seq={p['seq']} evm={evm}%"
                                + (f" (1-pass {st['evm_pct_1pass']}%)"
                                   if st.get("evm_pct_1pass") else "")
                                + f" [rx peak={lv['peak']:.3f} "
                                f"clip={lv['clip_pct']:.2f}%]")
                            daemon.handle_frame(
                                {"bytes_hex": pkt_bytes.hex(), "at_sample": n},
                                now)
                            handle_events()
                            # closed-loop feedback on measured EVM (may drop the
                            # link + re-negotiate at a higher margin)
                            if evm is not None and lp.link is not None:
                                handle_policy(lp, lp.on_rx_evm(float(evm), now))
                        except cm.ModemError:
                            pass
                        break

        now = now_s()          # refresh: enter_handshake may have blocked minutes

        # duty-cycled passive OFDM sensing: only while parked on CSS with no
        # upgrade in flight.  A hit re-arms the COMMANDED upgrade path (REQ
        # over CSS) toward my data peer -- we never jump straight onto OFDM.
        dpol = policies.get(data_dst) if data_dst is not None else None
        if (phy_mode == "css_then_ofdm" and dpol is not None
                and dpol.link is None and not dpol.upgrade_wanted
                and not busy and now - last_detect_t >= ofdm_detect_period_s):
            last_detect_t = now
            win = buf[max(0, n - int((ofdm_detect_period_s + 1.0) * cap_fs)):n]
            hit, dscore = _ml().detect_ofdm_activity(win, cap_fs,
                                                     thr=ofdm_detect_thr,
                                                     amb_rms=amb_rms)
            if hit:
                log(f"  t={now:6.1f} OFDM activity on the air (coh={dscore}) "
                    f"-> re-arming upgrade over CSS")
                handle_policy(dpol, dpol.on_detect(now, dscore))

        # control-plane REQ retransmits (SAME seq: the peer acts on whichever
        # copy lands, we match that ACK) and DATA (re)transmits (fresh seq
        # each try; the node layer only auto-ACKs unseen seqs).  Each policy
        # also decides here when its OFDM link has gone quiet long enough to
        # fall back.
        # carrier-sense backstop for handshake entry: even in our window,
        # defer probing if the air is loud (another node's probe / clock skew)
        tail = buf[max(0, n - int(0.9 * cap_fs)):n]
        air_quiet = (len(tail) == 0
                     or float(np.sqrt(np.mean(tail ** 2))) < 4.0 * amb_rms)
        for pol in list(policies.values()):
            hs_allowed = (hs_window_open(pol.peer_id, pol.initiator)
                          and air_quiet and not busy)
            handle_policy(pol, pol.tick(
                now,
                hs_allowed=hs_allowed,
                req_in_outbox=any(t.startswith(f"ofdmhs:{pol.peer_id}:")
                                  for _, t, _ in daemon.outbox),
                data_in_outbox=any(t.startswith(f"data:{pol.peer_id}:")
                                   for _, t, _ in daemon.outbox)))

        # DIFS + MAC
        if busy:
            idle_since = None
        elif idle_since is None:
            idle_since = now
        sustained_idle = (idle_since is not None) and (now - idle_since >= difs_s)
        sense_busy = busy or (not sustained_idle)

        if now - last_status >= 20.0:
            last_status = now
            links = sorted(p.peer_id for p in linked_policies())
            log(f"  t={now:6.1f} mf={sc:.4f} busy={busy} "
                f"links={links or '--'} "
                f"outbox={[t for _, t, _ in daemon.outbox]}"
                + (f" OVERFLOWS={rec.overflows}" if rec.overflows else ""))

        # out-of-slot is treated as a busy channel: the daemon drops any
        # countdown and re-arms fresh next time our slot opens, so a release
        # can only ever happen inside our own slot's open window
        phase = sched_phase(slot_cycle)
        in_slot = (my_slot_t0 <= phase
                   < my_slot_t0 + tx_slot_s * tx_slot_open_frac)
        rt = daemon.poll_tx(now, sense_busy or not in_slot)
        if rt is not None:
            raw, tag = rt
            pkt = cm.unpack_packet(raw)
            tx_pol = policies.get(pkt["dst"])
            use_ofdm = (tx_pol is not None
                        and tx_pol.use_ofdm_for(pkt["dst"], pkt["type"]))
            if phy_mode in OFDM_DATA_MODES and not use_ofdm:
                # the link went down between queueing and transmitting --
                # ofdm_only has no CSS to leak onto, so hold the packet
                # until the re-handshake brings the link back
                daemon.outbox.insert(0, (raw, tag, now + 5.0))
                daemon._backoff_remaining = None
                log(f"  t={now:6.1f} HOLD TX {tag} (no ofdm link)")
                time.sleep(tick_s)
                continue
            if (phy_mode in OFDM_DATA_MODES and use_ofdm
                    and not link_window_open(pkt["dst"])):
                # per-link airtime: this link's window is not open -- hold the
                # data/ACK so it can't step on another link that IS active.
                # Exception: an ACK inside the post-window grace still airs
                # (see ack_grace_s) instead of waiting a whole cycle.
                if tag.startswith("ack:") and link_window_grace(pkt["dst"]):
                    log(f"  t={now:6.1f} TX {tag} in post-window grace")
                else:
                    daemon.outbox.insert(0, (raw, tag, now + 2.0))
                    daemon._backoff_remaining = None
                    log(f"  t={now:6.1f} HOLD TX {tag} "
                        f"(link {pkt['dst']} outside airtime window)")
                    time.sleep(tick_s)
                    continue
            # ENCODE FIRST, sense LAST.  Encoding a long CSS frame takes
            # seconds; if the freshness check runs before the encode, two
            # nodes whose backoffs land close together can both see quiet air,
            # both encode, and both play into each other.  Checking AFTER the
            # encode shrinks the vulnerable window from seconds to the device
            # latency.
            if use_ofdm:
                # RMS-matched to the probe: the BAT is only valid at the
                # probe's power density (peak-matching under-drove data
                # frames by ~4.7 dB OTA and RS never survived it)
                pcm = _ml().ofdm_tx_pcm(tx_pol.link, raw, ofdm_amp)
                fs_play = tx_pol.link["tx_rate"]
                phy_name = "ofdm"
            else:
                pcm = fw.encode_packet(raw, tag)
                fs_play = out_fs
                phy_name = "css"
            # final freshness check, immediately before the speaker opens
            # (someone may have started).  The RMS guard covers EVERY
            # transmission: OFDM traffic (probes, frames) is invisible to
            # the chirp-matched sense, and a long CSS chirp launched over a
            # neighbour's handshake would wreck it.
            fresh = rec.data()
            quick_mf = chirp_score(fresh[max(0, len(fresh) - int(0.9 * cap_fs)):])
            tail_rms = float(np.sqrt(np.mean(
                fresh[max(0, len(fresh) - int(0.9 * cap_fs)):] ** 2)))
            if quick_mf > on_thr or tail_rms > 4.0 * amb_rms:
                daemon.outbox.insert(0, (raw, tag, 0.0))
                daemon._backoff_remaining = None
                busy = True
                idle_since = None
                log(f"  t={now:6.1f} ABORT TX {tag} (air active)")
            else:
                log(f"  t={now:6.1f} TX {phy_name} {pkt['type_name']} "
                    f"dst={pkt['dst']} seq={pkt['seq']} "
                    f"({len(pcm)/fs_play:.1f}s)")
                tx_log.append({"t": round(now, 1), "phy": phy_name,
                               "type": pkt["type_name"], "dst": pkt["dst"],
                               "seq": pkt["seq"]})
                tx_windows.append({"t": round(now, 1),
                                   "dur_s": round(len(pcm) / fs_play, 2),
                                   "what": f"{phy_name} {pkt['type_name']}",
                                   "peer": pkt["dst"]})
                play_blocking(pcm, fs_play, out_dev)
                daemon.on_tx(tag, len(rec.data()))
                if tx_pol is not None:
                    tx_pol.on_tx(tag, now_s())
                ofdm_listen_start = len(rec.data())     # skip our own echo
                reset_carrier()
        else:
            time.sleep(tick_s)

    # rec.stop() performs the final spill flush and closes the WAV; the file
    # has been valid on disk since the first flush (see Recorder.spill_to).
    rec.stop()

    # DATA delivered on ANY phy counts: CSS after a fallback is a success,
    # that is the whole point of the fallback
    emit_result(build_result())


if __name__ == "__main__":
    main()
