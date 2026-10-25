"""
ofdmhs_node.py -- role: node (assign to BOTH hosts) -- V.22-style OFDM
handshake, fully in-band, over the air.

Thin hardware binding around ofdm_handshake.HandshakeSM (which the meshmodem
will later import directly).  Coord is NOT used at all -- the framework
starting both scripts is the only synchronisation, exactly like a phone line
going off-hook.  No shared files are read during the handshake; each node
writes its measured/learned tables afterwards purely for post-hoc verification.

id_map decides the initiator (node 1, default macdesktop).  Each node
transmits at its speaker's native rate; the peer detects that rate by trial
acquisition (see ofdm_handshake.try_decode_probe / try_decode_msg).

Half-duplex discipline: a node only attempts to decode once the tail of its
listen window is quiet (the peer's transmission has ended), so it never talks
over the peer.  During the handshake the listen window always starts after our
own TX ends, so we never decode our own transmission -- with one bounded
exception: at ENTRY the window reaches hs_lookback_s into audio the recorder
already captured, because the two nodes do not enter together and a peer probe
that finished moments earlier was otherwise discarded (costing a full
wait_probe_s while the peer re-probed).  That lookback must stay well inside one
handshake cycle so it can never reach this node's own previous probe.

Params (JSON):
{
    "amplitude":     0.3,
    "poll_s":        3.0,
    "run_timeout":   420,
    "wait_probe_s":  75,
    "wait_msg_s":    75,
    "max_retries":   2,
    "freq_lo":       400,     # STARTING band edges, Hz (aka hs_freq_lo)
    "freq_hi":       4000,    # aka hs_freq_hi
    "band_fallback": 1,       # allow MSG_BAND renegotiation
    "band_shrink":   0.8,     # propose only if the band shrinks below this
    "max_retrains":  1,
    "id_map":        {"macdesktop": 1, "laptop": 2}
}

freq_lo/freq_hi set where the probe STARTS, overriding the fixed convention
(HS_BAND) for the run -- e.g. freq_hi=14000 probes 400-14k.  Every node gets
identical PARAMS, so both ends open on the same band.

From there the band is negotiated per direction: each node grades the probe it
receives and, if the edges are dead, sends MSG_BAND telling the peer to narrow
its TRANSMIT band, and both ends retrain on the new geometry (see
ofdm_handshake).  So the two directions can end on different bands, chosen by
whichever end has to decode them.  Probing wide is therefore cheap to try: a
dead top octave costs one retrain instead of inflating every BAT message for
the rest of the run.  band_fallback=0 pins the run to the starting band.
"""
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
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

from testlib import (                                           # noqa: E402
    CLIENT_ID, INPUT_DEV, OUTPUT_DEV, RUN_DIR, PARAMS, RUN_ID,
    log, to_int_dev, Recorder,
)
import ofdm_handshake as _ofdm_hs                               # noqa: E402
from ofdm_handshake import hs_cfg, HS_ECC as HS_ECC_B          # noqa: E402
from meshlib import run_handshake                              # noqa: E402
from ofdm_modem import bytes_to_bat                             # noqa: E402

AMP         = float(PARAMS.get("amplitude",    0.3))
POLL_S      = float(PARAMS.get("poll_s",       3.0))
RUN_TIMEOUT = float(PARAMS.get("run_timeout",  420.0))
MAX_RETRIES = int(PARAMS.get("max_retries",    2))

# A wait shorter than one message's airtime can never be satisfied: the
# deadline is armed when we decide to transmit, so a too-small value expires
# before the peer has finished replying and every state degenerates into a
# retransmit storm.  wait_msg_s=0 in particular arms at now+0 and fires on the
# same timestamp that armed it.  Clamp rather than fail: the operator gets a
# usable run plus a loud line in the log.
WAIT_FLOOR_S = 20.0


def _wait_floor(key: str, raw: float) -> float:
    # Takes the already-fetched value, not the key, on purpose: the web UI's
    # param form is built by regex-scraping literal PARAMS lookups out of this
    # file (app.py:_parse_test_params). A lookup wrapped inside a helper does
    # not match that pattern, so wait_probe_s/wait_msg_s would vanish from the
    # form. Keeping the lookup flat at the call site and clamping on its
    # result here keeps both the discoverable default and the floor.
    v = float(raw)
    if v < WAIT_FLOOR_S:
        log(f"  WARNING: {key}={v:g}s is below the {WAIT_FLOOR_S:g}s floor "
            f"(shorter than one message's airtime) -- clamping")
        return WAIT_FLOOR_S
    return v


WAIT_PROBE  = _wait_floor("wait_probe_s", PARAMS.get("wait_probe_s", 75.0))
WAIT_MSG    = _wait_floor("wait_msg_s",   PARAMS.get("wait_msg_s",   75.0))

# Handshake band edges for this run. freq_lo/freq_hi are kept as flat calls
# on PARAMS, same scraper constraint as _wait_floor above. A nested lookup
# as the default expression breaks the scraper's non-greedy regex (it stops
# at the first closing paren and fails to literal_eval the fragment),
# leaving freq_lo/freq_hi as empty fields with no default and dropping the
# legacy alias keys entirely. Do not nest a PARAMS.get() inside this default.
FREQ_LO = float(PARAMS.get("freq_lo", 400.0))
FREQ_HI = float(PARAMS.get("freq_hi", 4000.0))
# Legacy meshmodem spelling, consulted only when freq_lo/freq_hi were not set
# at all (freq_lo/freq_hi always win when present, same as before).
if "freq_lo" not in PARAMS and "hs_freq_lo" in PARAMS:
    FREQ_LO = float(PARAMS["hs_freq_lo"])
if "freq_hi" not in PARAMS and "hs_freq_hi" in PARAMS:
    FREQ_HI = float(PARAMS["hs_freq_hi"])
# Rebind on the module (not a local copy): hs_cfg() reads HS_OVERRIDES at call
# time, so every probe/message built from here on uses the requested band.
#
# Allocation + symbol geometry, same spellings meshmodem uses.  These matter
# most on a MARGINAL link:
#   cp_ratio         the cyclic prefix is the echo budget -- it must outlast
#                    the room's reflections or they land as inter-symbol
#                    interference that no SNR margin can recover.  0.25 of a
#                    1024-pt FFT at 48k is 5.3ms = ~1.8m of excess path; a
#                    10ft room with hard surfaces needs more.
#   max_bits_per_bin 1 pins every surviving carrier to BPSK, which is what a
#                    link down at tens of baud can actually hold.
#   snr_margin_db    headroom over the modulation floor; buys robustness
#                    against the probe->data fade migration at the cost of
#                    dropping marginal carriers.
#   isi_derate       measure how much of the room's energy arrives AFTER the
#                    cyclic prefix and cap each carrier's SNR accordingly.
#                    The probe cannot see this by itself: it is 150 identical
#                    symbols, so reverberation settles into a steady state and
#                    is absorbed into H as if it were gain. Turning this on
#                    makes snr_margin_db largely redundant (both cover the
#                    same effect), so its default drops to 2 dB here. They
#                    compose safely (margin is spent before the ceiling is
#                    applied), but a big margin on top just costs carriers.
_ISI_DERATE = float(PARAMS.get("isi_derate", 0.0))  # choices: 0, 1
# The margin is never adjusted automatically for isi_derate: the web UI submits
# every param with its schema default, so this script cannot tell "left alone"
# from "deliberately set to 9".  Choose the pair yourself -- with isi_derate on,
# 0-2 dB is the useful range.
_MARGIN = float(PARAMS.get("snr_margin_db", 9.0))
_HS_GEOM = {
    "n_fft":            int(PARAMS.get("n_fft", 1024)),  # choices: 512, 1024, 2048, 4096
    "cp_ratio":         float(PARAMS.get("cp_ratio", 0.25)),  # choices: 0.25, 0.375, 0.5
    "probe_duration_s": float(PARAMS.get("probe_duration_s", 5.0)),
    "probe_meas_syms":  int(PARAMS.get("probe_meas_syms", 32)),
    "isi_derate":       _ISI_DERATE,
    "snr_margin_db":    _MARGIN,
    "max_bits_per_bin": int(PARAMS.get("max_bits_per_bin", 2)),
    "snr_estimator":    str(PARAMS.get("snr_estimator", "mean")),
}
_ofdm_hs.HS_OVERRIDES = {"freq_lo": FREQ_LO, "freq_hi": FREQ_HI, **_HS_GEOM}

# Band fallback (MSG_BAND retrain).  Probe wide, keep only what survives: the
# far end of each direction grades the path and, when the edges are dead, tells
# this node to narrow its TX band.  band_shrink is how much of the span must
# fall away before a retrain pays for itself.
# Seconds of ALREADY-CAPTURED audio to search when the handshake starts.  The
# recorder runs continuously, so a peer probe that finished just before this
# node entered the handshake is on disk but was previously discarded, costing a
# full wait_probe_s while the peer re-probed.  Keep it well inside one handshake
# cycle so it can never reach back to this node's OWN previous probe.
HS_LOOKBACK = float(PARAMS.get("hs_lookback_s", 15.0))
BAND_FALLBACK = bool(float(PARAMS.get("band_fallback", 1)))
BAND_SHRINK   = float(PARAMS.get("band_shrink", 0.8))
_ofdm_hs.HS_MAX_RETRAINS = int(PARAMS.get("max_retrains", 1))

# RobustLink rate ladder (ofdm_handshake.HS_RL_STEPS).  Step k means G = 2**k
# bits per OFDM symbol -- the band splits into G interleaved groups, one bit
# each.  Every step doubles handshake message rate for exactly 3dB of per-bit
# spreading gain:  0=~35bps  1=~70  2=~140  3=~300  4=~600  5=~1200 (at 48k).
# This is a CEILING; fast_groups_for still drops below it on a sparse table.
_ofdm_hs.HS_RL_MAX_STEP = int(PARAMS.get("rl_max_step", 3))  # choices: 0, 1, 2, 3, 4, 5
# Live carriers a group must have before it may be split off.  Lower reaches a
# higher step on fragmented tables; raise it first if step-3 messages fail to
# decode OTA, since a thin group is the likeliest cause.
_ofdm_hs.HS_FAST_MIN_PER_GROUP = int(PARAMS.get("rl_min_per_group", 6))
# Step for a message whose sender does not know its own tx table yet --
# BAT_REPORT.  The receiver trials the whole ladder, so this is an SNR bet,
# not an agreement problem; a lost one costs a full turn to retransmit.
_ofdm_hs.HS_RL_BOOTSTRAP_STEP = int(PARAMS.get("rl_bootstrap_step", 2))  # choices: 0, 1, 2, 3
# Link capacity that must survive BAT coarsening.  Wire bytes are airtime, and
# near the top of a table they are cheap: 0.90 typically cuts the BAT ~3x for
# ~6% of capacity.  Raise toward 1.0 to prefer a faithful table over a fast one.
_ofdm_hs.HS_BAT_MIN_KEEP = float(PARAMS.get("bat_min_keep", 0.90))


def _native_rates(in_dev, out_dev):
    try:
        from shared import native_samplerate
        mic = int(native_samplerate("input",  in_dev)  or 48000)
        spk = int(native_samplerate("output", out_dev) or 48000)
    except Exception as exc:
        log(f"  native-rate probe failed ({exc}); assuming 48000")
        mic = spk = 48000
    return mic, spk


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0



def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    in_dev  = to_int_dev(INPUT_DEV)
    out_dev = to_int_dev(OUTPUT_DEV)
    # Literal default rather than a module constant: the UI's param scraper prefills the
    # form by literal_eval'ing this expression, and a bare name evaluates to
    # nothing, leaving the field blank.
    id_map  = PARAMS.get("id_map", {"macdesktop": 1, "laptop": 2})
    node_id = int(id_map.get(CLIENT_ID, 2))
    role    = "initiator" if node_id == 1 else "responder"
    mic_fs, spk_fs = _native_rates(in_dev, out_dev)

    band = hs_cfg(spk_fs)
    log(f"[{CLIENT_ID}] ofdmhs {role}  in={in_dev} out={out_dev}  "
        f"mic_fs={mic_fs} spk_fs={spk_fs}")
    log(f"  band: requested {FREQ_LO:.0f}-{FREQ_HI:.0f} Hz -> "
        f"{band['freq_lo_hz']:.0f}-{band['freq_hi_hz']:.0f} Hz  "
        f"({band['n_data_carriers']} data carriers)")
    log(f"  geometry: n_fft={band['n_fft']} cp={band['cp_len']} "
        f"({1000.0 * band['cp_len'] / spk_fs:.1f}ms echo budget = "
        f"{343.0 * band['cp_len'] / spk_fs:.1f}m excess path)  "
        f"margin={band['snr_margin_db']}dB "
        f"max_bits={band['max_bits_per_bin']} "
        f"probe={band['actual_probe_s']}s")
    # The CP is measured against the room, not the FFT, so a short one cannot
    # be compensated anywhere downstream: ISI scales with the signal instead
    # of sitting under it like noise, so neither SNR margin nor RobustLink
    # spreading recovers it.
    _cp_m = 343.0 * band["cp_len"] / spk_fs
    if _cp_m < 2.0:
        log(f"  WARNING: cp={1000.0 * band['cp_len'] / spk_fs:.1f}ms covers only "
            f"{_cp_m:.1f}m of excess path -- shorter than a typical room's "
            f"reverb tail.  Expect fragmentary probe acquisition and low "
            f"carrier counts; raise cp_ratio or n_fft.")
    # RobustLink's blind rate is DERIVED from this geometry, not configured:
    # narrowing the band or shortening the CP lowers it automatically so the
    # step stays as safe as it was on the a-priori convention.
    _mcfg = _ofdm_hs.hs_msg_cfg(spk_fs)
    _bg   = _ofdm_hs.bootstrap_groups(_mcfg)
    log(f"  robustlink: blind G={_bg} "
        f"(~{_ofdm_hs.rl_rate_bps(_bg, _mcfg):.0f}bps), "
        f"step<={_ofdm_hs.HS_RL_MAX_STEP}, "
        f"geometry cap=step{_ofdm_hs.geometry_step_cap(_mcfg)}, "
        f"min/group={_ofdm_hs.HS_FAST_MIN_PER_GROUP}")

    # Widening the band inflates the BAT, and the BAT *is* the message: a
    # RobustLink body costs ~1 bit per symbol, so 400-4000 Hz ships a 26B
    # table in ~7s while 400-14000 Hz ships ~95B in ~25s. A wait tuned for the
    # narrow band expires mid-reply at the wide one and the run collapses into
    # retransmissions. Derive the floor from the band actually in use instead
    # of trusting a fixed default.
    wait_probe, wait_msg = WAIT_PROBE, WAIT_MSG
    raw_bat_b  = -(-band["n_data_carriers"] * 3 // 8)      # 3 bits/carrier
    msg_bits   = (raw_bat_b + 5 + HS_ECC_B) * 8            # + header + RS
    msg_air_s  = msg_bits * band["symbol_samples"] / spk_fs
    need_msg   = round(2.5 * msg_air_s + 10.0)             # their turn + ours
    log(f"  est. worst-case message airtime {msg_air_s:.0f}s "
        f"({raw_bat_b}B raw BAT) -> wait_msg_s needs >= {need_msg}s")
    if wait_msg < need_msg:
        log(f"  WARNING: wait_msg_s={wait_msg:g}s too short for this band "
            f"-- raising to {need_msg}s")
        wait_msg = float(need_msg)
    if wait_probe < msg_air_s:
        wait_probe = float(round(msg_air_s + 10.0))
        log(f"  WARNING: wait_probe_s raised to {wait_probe:g}s for this band")

    rec = Recorder(in_dev, mic_fs)
    rec.start()
    time.sleep(0.6)                          # settle + sample ambient floor
    ambient = _rms(rec.data()[-int(0.4 * mic_fs):])
    log(f"  ambient floor: {ambient:.6f} rms")

    hs = run_handshake(
        role, rec, mic_fs, spk_fs, out_dev,
        amplitude=AMP, poll_s=POLL_S, timeout=RUN_TIMEOUT,
        wait_probe_s=wait_probe, wait_msg_s=wait_msg,
        max_retries=MAX_RETRIES, amb_rms=ambient, log=log,
        band_negotiation=BAND_FALLBACK, band_shrink=BAND_SHRINK,
        lookback_s=HS_LOOKBACK,
        # no mesh data frames follow a standalone handshake, so the q-keyed
        # preamble check would only burn decode CPU every poll
        mesh_fin=False,
        # ofdmhs historically ran symmetric waits; the mesh driver offsets the
        # responder to de-lockstep retries, which is strictly better on a weak
        # link -- keep it.
    )
    sm          = hs["sm"]
    probe_info  = hs["probe_rx"]
    fail_reason = hs["fail_reason"]
    tx_log      = hs["tx_log"]
    now         = lambda: hs["elapsed_s"]

    # -- report (files are post-hoc verification only, never protocol) --
    my_cfg = hs_cfg(spk_fs, sm.my_tx_band)
    n_my   = int(my_cfg["n_data_carriers"])
    learned = (bytes_to_bat(sm.my_tx_bat, n_my).tolist()
               if sm.my_tx_bat else None)
    status = "ok" if sm.done else ("fail" if sm.failed else "timeout")

    result = {
        "client_id":   CLIENT_ID,
        "role":        role,
        "run_id":      RUN_ID,
        "status":      status,
        "fail_reason": fail_reason,
        "state":       sm.state,
        "elapsed_s":   round(now(), 1),
        "mic_fs":      mic_fs,
        "spk_fs":      spk_fs,
        "freq_lo_req": FREQ_LO,
        "freq_hi_req": FREQ_HI,
        "band_hz":     [round(my_cfg["freq_lo_hz"], 1),
                        round(my_cfg["freq_hi_hz"], 1)],
        "n_carriers":  n_my,
        "probe_rx":    probe_info,
        "my_tx_bat_active": (int(np.sum(np.array(learned) > 0))
                             if learned else None),
        "my_tx_gross_bps": (int(sum(learned) * spk_fs
                                // my_cfg["symbol_samples"])
                            if learned else None),
        "tx_log":      tx_log,
        "sm_log":      sm.log().splitlines(),
    }
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(
        json.dumps(result, indent=2))
    # oracle files for post-hoc cross-checking (protocol never reads these)
    if sm.peer_tx_bat:
        (RUN_DIR / f"hs_measured_{CLIENT_ID}.json").write_text(
            json.dumps({"peer_tx_bat_hex": sm.peer_tx_bat.hex()}))
    if sm.my_tx_bat:
        (RUN_DIR / f"hs_learned_{CLIENT_ID}.json").write_text(
            json.dumps({"my_tx_bat_hex": sm.my_tx_bat.hex()}))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
