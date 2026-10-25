"""
ofdm2way_node.py -- symmetric TWO-WAY OFDM channel probe + IN-BAND back-channel.

Both hosts run this same script under one role ("node"); assign it to exactly
two clients.  It measures BOTH acoustic directions (a speaker->mic link is not
reciprocal) and then returns each measured bit-allocation table (BAT) to the far
end ACOUSTICALLY over OFDM -- no shared-file hand-off for the negotiation.

Phase 1 -- probe (node 1 = macdesktop goes first):
    Leg 1  node1 --probe--> node2 :  node2 measures, computes BAT_1to2
    Leg 2  node2 --probe--> node1 :  node1 measures, computes BAT_2to1

Phase 2 -- in-band back-channel (RobustLink: 1 bit/symbol, BPSK on ALL carriers,
receiver does maximal-ratio combining with the H it just measured -- robust
without any prior rate agreement, and works even though the two directions have
near-disjoint usable bands):
    Leg 3  node2 --BAT_1to2--> node1 :  node1 LEARNS its TX table to node2
    Leg 4  node1 --BAT_2to1--> node2 :  node2 LEARNS its TX table to node1

Each transmitter builds at ITS speaker's native rate (no host resample); each
receiver records at its mic rate and resamples to the transmitter's rate.  The
file BATs (bat_1to2.json / bat_2to1.json) written in phase 1 are kept only as an
ORACLE to verify the in-band transfer; the acoustic path is the real channel.

Params: id_map, n_fft, cp_ratio, freq_lo, freq_hi, pilot_spacing,
        probe_duration_s, snr_margin_db, max_bits_per_bin, amplitude,
        sc_threshold, tail_s, leg_timeout, bc_ecc_bytes.
"""
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
import json
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

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
    log, to_int_dev, coord_signal, coord_wait_any,
    Recorder, play_blocking,
)
from ofdm_modem import (                                        # noqa: E402
    build_config, Modem, RobustLink, bat_to_bytes, bytes_to_bat,
    carrier_spectrum,
)
import ofdm_handshake as _ofdm_hs                               # noqa: E402

DEFAULT_ID_MAP = {"macdesktop": 1, "laptop": 2}
AMP        = float(PARAMS.get("amplitude",     0.3))
SC_THRESH  = float(PARAMS.get("sc_threshold",  0.25))
TAIL_S     = float(PARAMS.get("tail_s",        1.5))
LEG_TMO    = float(PARAMS.get("leg_timeout",   90.0))
BC_ECC     = int(PARAMS.get("bc_ecc_bytes",    10))

# RobustLink rate ladder, shared with the handshake (ofdm_handshake.HS_RL_STEPS).
# Step k means G = 2**k bits per OFDM symbol: the band splits into G interleaved
# groups, one bit each.  Every step doubles back-channel rate for exactly 3dB of
# per-bit spreading gain: 0=~35bps 1=~70 2=~140 3=~300 4=~600 5=~1200 (at 48k).
# A CEILING -- leg 4 still steps down if the table it learned is sparse.
_ofdm_hs.HS_RL_MAX_STEP = int(PARAMS.get("rl_max_step", 3))  # choices: 0, 1, 2, 3, 4, 5
# Step for a leg whose sender does not know its own tx table yet -- leg 3, which
# is blind by construction (nobody has told node2 its table).  The receiver
# trials the whole ladder, so this is an SNR bet, not an agreement problem.
_ofdm_hs.HS_RL_BOOTSTRAP_STEP = int(PARAMS.get("rl_bootstrap_step", 2))  # choices: 0, 1, 2, 3
# Live carriers a group must have before it may be split off.
_ofdm_hs.HS_FAST_MIN_PER_GROUP = int(PARAMS.get("rl_min_per_group", 6))


def _node_id() -> int:
    id_map = PARAMS.get("id_map", DEFAULT_ID_MAP)
    return int(id_map.get(CLIENT_ID, 1))


def _probe_cfg(sample_rate: int) -> dict:
    return build_config(
        n_fft            = int(PARAMS.get("n_fft",            1024)),
        cp_ratio         = float(PARAMS.get("cp_ratio",        0.25)),
        freq_lo          = float(PARAMS.get("freq_lo",         400)),
        freq_hi          = float(PARAMS.get("freq_hi",         4000)),
        pilot_spacing    = int(PARAMS.get("pilot_spacing",     8)),
        sample_rate      = int(sample_rate),
        probe_duration_s = float(PARAMS.get("probe_duration_s", 5.0)),
        snr_margin_db    = float(PARAMS.get("snr_margin_db",   6.0)),
        max_bits_per_bin = int(PARAMS.get("max_bits_per_bin",  6)),
    )


def _native_rates(in_dev, out_dev):
    try:
        from shared import native_samplerate
        mic = int(native_samplerate("input",  in_dev)  or 48000)
        spk = int(native_samplerate("output", out_dev) or 48000)
    except Exception as exc:
        log(f"  native-rate probe failed ({exc}); assuming 48000")
        mic = spk = 48000
    return mic, spk


def _resample_to(cap: np.ndarray, mic_fs: int, tx_fs: int) -> np.ndarray:
    if tx_fs == mic_fs or len(cap) == 0:
        return cap.astype(np.float32)
    g = gcd(tx_fs, mic_fs)
    return resample_poly(cap.astype(np.float64), tx_fs // g, mic_fs // g).astype(np.float32)


def _play(sig, fs, out_dev, amp=AMP):
    peak = float(np.max(np.abs(sig))) or 1e-9
    play_blocking((amp * sig / peak).astype(np.float32), fs, out_dev)


# --------------------------------------------------------------------------- #
# Phase 1: probe legs
# --------------------------------------------------------------------------- #

def transmit_probe(leg, out_dev, spk_fs, st):
    cfg   = _probe_cfg(spk_fs)
    modem = Modem(cfg)
    probe = modem.build_probe()
    (RUN_DIR / f"leg{leg}_cfg.json").write_text(json.dumps(cfg, indent=2))
    st["tx_cfg"] = cfg                       # remember my speaker config
    log(f"  [leg{leg}] TX probe  spk_fs={spk_fs}  "
        f"band={cfg['freq_lo_hz']:.0f}-{cfg['freq_hi_hz']:.0f}Hz")
    coord_signal(f"leg{leg}_ready")
    coord_wait_any(f"leg{leg}_armed", timeout=LEG_TMO)
    _play(probe, spk_fs, out_dev)
    coord_signal(f"leg{leg}_played")
    coord_wait_any(f"leg{leg}_done", timeout=LEG_TMO)


def receive_probe(leg, direction, rec, mic_fs, st):
    if not coord_wait_any(f"leg{leg}_ready", timeout=LEG_TMO):
        return {"direction": direction, "valid": False, "error": "no_tx_ready"}
    cfg_path = RUN_DIR / f"leg{leg}_cfg.json"
    deadline = time.time() + 20
    while not cfg_path.exists() and time.time() < deadline:
        time.sleep(0.2)
    if not cfg_path.exists():
        return {"direction": direction, "valid": False, "error": "no_cfg"}
    cfg   = json.loads(cfg_path.read_text())
    tx_fs = int(cfg["sample_rate"])

    start = rec.frames_captured
    coord_signal(f"leg{leg}_armed")
    coord_wait_any(f"leg{leg}_played", timeout=cfg["actual_probe_s"] + LEG_TMO)
    time.sleep(TAIL_S)
    cap = _resample_to(rec.data()[start:], mic_fs, tx_fs)

    modem  = Modem(cfg)
    result = modem.analyze_probe(cap, threshold=SC_THRESH)
    out = {"direction": direction, "tx_fs": tx_fs, "valid": bool(result.get("valid"))}
    if not result.get("valid"):
        out["error"] = result.get("error", "probe_invalid")
        log(f"  [leg{leg}] probe INVALID: {out['error']}")
        coord_signal(f"leg{leg}_done")
        return out

    bat = modem.allocate(result)
    Modem.save_bat(bat, str(RUN_DIR / f"bat_{direction}.json"))     # oracle only
    snr = result["snr_db"]
    # remember what I measured as receiver over THIS channel (for the back-channel)
    st["rx_cfg"]   = cfg
    st["rx_H"]     = result["H"]
    st["rx_bat"]   = bat                     # this is the FAR end's TX table
    st["rx_dir"]   = direction
    # Per-carrier probe shape: frequency vs measured SNR vs allocated bits.
    # Determines whether the RobustLink interleave has live carriers in every group.
    spectrum = carrier_spectrum(cfg, snr, bat)
    (RUN_DIR / f"spectrum_{direction}.json").write_text(json.dumps(spectrum))
    n_probe = int(result.get("n_syms") or 0)
    out.update({
        "coherence":       result.get("coherence"),
        "snr_median_db":   round(float(np.median(snr)), 1),
        "snr_max_db":      round(float(snr.max()), 1),
        "snr_min_db":      round(float(snr.min()), 1),
        "snr_p25_db":      round(float(np.percentile(snr, 25)), 1),
        "n_probe_syms":    n_probe,
        "n_occupied":      int(cfg["n_occupied"]),
        "active_carriers": int(np.sum(bat > 0)),
        "gross_bps":       int(np.sum(bat) * tx_fs // cfg["symbol_samples"]),
        "bat_summary":     modem.bat_summary(bat),
        "band_hz":         [round(cfg["freq_lo_hz"], 1),
                            round(cfg["freq_hi_hz"], 1)],
        "spectrum":        spectrum,
    })
    log(f"  [leg{leg}] measured {direction}: {out['bat_summary']}")
    log(f"  [leg{leg}] probe {direction}: coh={out['coherence']:.3f} "
        f"{n_probe}syms  snr min/p25/med/max="
        f"{out['snr_min_db']}/{out['snr_p25_db']}/"
        f"{out['snr_median_db']}/{out['snr_max_db']}dB")
    coord_signal(f"leg{leg}_done")
    return out


# --------------------------------------------------------------------------- #
# Phase 2: in-band back-channel (RobustLink)
# --------------------------------------------------------------------------- #

def bc_transmit(leg, out_dev, st):
    """Send the BAT I computed as receiver back to the far end, over OFDM."""
    cfg     = st["tx_cfg"]                    # my speaker's channel config
    bat     = st["rx_bat"]                    # far end's TX table (I measured it)
    payload = bat_to_bytes(bat)
    # Ladder step. Leg 3's sender does not know its own tx table yet, so it
    # uses HS_RL_BOOTSTRAP_STEP. Leg 4's sender learned its table from leg 3
    # (st["my_tx_bat"]), so it sizes the step to the real channel.
    my_tx   = st.get("my_tx_bat")
    G       = (_ofdm_hs.fast_groups_for(my_tx, cfg) if my_tx
               else _ofdm_hs.bootstrap_groups(cfg))
    rl      = RobustLink(cfg, ecc_bytes=BC_ECC, n_groups=G)
    sig     = rl.encode(payload)
    log(f"  [leg{leg}] BC-TX  {len(payload)}B ({st['rx_dir']}) over "
        f"{cfg['sample_rate']}Hz  G={G} "
        f"(~{_ofdm_hs.rl_rate_bps(G, cfg):.0f}bps, "
        f"{'measured' if my_tx else 'bootstrap'})")
    coord_signal(f"leg{leg}_ready")
    coord_wait_any(f"leg{leg}_armed", timeout=LEG_TMO)
    _play(sig, cfg["sample_rate"], out_dev)
    coord_signal(f"leg{leg}_played")
    coord_wait_any(f"leg{leg}_done", timeout=LEG_TMO)
    return {"leg": leg, "role": "bc_tx", "sent_bytes": len(payload),
            "n_groups": G, "bps": round(_ofdm_hs.rl_rate_bps(G, cfg), 1),
            "airtime_s": round(len(sig) / cfg["sample_rate"], 2),
            "step_source": "measured" if my_tx else "bootstrap",
            "tx_band_hz": [round(cfg["freq_lo_hz"], 1),
                           round(cfg["freq_hi_hz"], 1)]}


def bc_receive(leg, rec, mic_fs, st):
    """Receive my own TX table from the far end, in-band."""
    cfg      = st["rx_cfg"]                   # far speaker channel (== transport cfg)
    tx_fs    = int(cfg["sample_rate"])
    my_tx    = st["tx_cfg"]                   # my transmit config sizes the BAT
    n_carr   = int(my_tx["n_data_carriers"])

    if not coord_wait_any(f"leg{leg}_ready", timeout=LEG_TMO):
        return {"leg": leg, "role": "bc_rx", "ok": False, "error": "no_tx_ready"}
    start = rec.frames_captured
    coord_signal(f"leg{leg}_armed")
    coord_wait_any(f"leg{leg}_played", timeout=LEG_TMO)
    time.sleep(TAIL_S)
    cap = _resample_to(rec.data()[start:], mic_fs, tx_fs)

    rl      = RobustLink(cfg, ecc_bytes=BC_ECC)
    # Trial the whole ladder: the receiver cannot infer the sender's step,
    # and acquisition is G-independent so all steps share one coherence scan.
    payload = rl.decode(cap, groups=_ofdm_hs.rl_ladder())
    out = {"leg": leg, "role": "bc_rx", "direction": st["rx_dir"]}
    # Attach decode telemetry whether or not the payload survived.
    if rl.last_stats:
        out["rl"] = rl.last_stats
        s = rl.last_stats
        log(f"  [leg{leg}] BC-RX rl: G={s['n_groups']} {s['n_bits']}bits "
            f"coh={s.get('coherence')} evm={s['evm_pct']:.1f}% "
            f"margin_min={s['margin_min']}")
        log(f"  [leg{leg}] BC-RX per-group evm%={s['evm_pct_per_group']} "
            f"power_db={s['group_power_db']}")
    if payload is None:
        out.update({"ok": False, "error": "bc_decode_fail"})
        log(f"  [leg{leg}] BC-RX decode FAILED"
            + ("" if rl.last_stats else " (no acquisition lock)"))
        coord_signal(f"leg{leg}_done")
        return out

    my_bat = bytes_to_bat(payload, n_carr)
    # My OWN tx table, now known -- lets a later leg pick its step from the real
    # channel instead of betting.  Stored packed, the form fast_groups_for reads.
    st["my_tx_bat"] = payload
    Modem.save_bat(my_bat, str(RUN_DIR / f"bat_inband_recv_{CLIENT_ID}.json"))

    # Verify against the oracle: the far end measured MY transmit path in
    # phase 1 and wrote it as bat_<my tx direction>.json on the shared dir.
    oracle_path = RUN_DIR / f"bat_{_my_tx_direction()}.json"
    matched = None
    if oracle_path.exists():
        oracle = Modem.load_bat(str(oracle_path))
        matched = (len(oracle) == len(my_bat) and bool(np.array_equal(oracle, my_bat)))
        n_diff  = int(np.sum(oracle != my_bat)) if len(oracle) == len(my_bat) else -1
        out["carriers_mismatched"] = n_diff
    out.update({"ok": True, "recv_carriers": int(len(my_bat)),
                "active_carriers": int(np.sum(my_bat > 0)),
                "matches_oracle": matched})
    log(f"  [leg{leg}] BC-RX ok  learned {int(np.sum(my_bat>0))} active carriers  "
        f"matches_oracle={matched}")
    coord_signal(f"leg{leg}_done")
    return out


def _my_tx_direction() -> str:
    """The direction label of MY transmit path (the BAT I learn in-band)."""
    return "1to2" if _node_id() == 1 else "2to1"


# --------------------------------------------------------------------------- #

def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    in_dev  = to_int_dev(INPUT_DEV)
    out_dev = to_int_dev(OUTPUT_DEV)
    node_id = _node_id()
    mic_fs, spk_fs = _native_rates(in_dev, out_dev)
    log(f"[{CLIENT_ID}] ofdm2way node_id={node_id}  in={in_dev} out={out_dev}  "
        f"mic_fs={mic_fs} spk_fs={spk_fs}")

    rec = Recorder(in_dev, mic_fs)
    rec.start()
    time.sleep(0.3)

    st: dict = {}
    measured, bc = [], []

    # Phase 1: probe both directions
    for leg, tx_node, direction in ((1, 1, "1to2"), (2, 2, "2to1")):
        if node_id == tx_node:
            transmit_probe(leg, out_dev, spk_fs, st)
        else:
            measured.append(receive_probe(leg, direction, rec, mic_fs, st))

    # Phase 2: in-band back-channel.  Leg 3: node2->node1.  Leg 4: node1->node2.
    for leg, tx_node in ((3, 2), (4, 1)):
        if node_id == tx_node:
            bc.append(bc_transmit(leg, out_dev, st))
        else:
            bc.append(bc_receive(leg, rec, mic_fs, st))

    rec.stop()
    try:
        rec.save(RUN_DIR / f"{CLIENT_ID}_2way.wav")
    except Exception:
        pass

    ok = any(b.get("ok") for b in bc)
    result = {
        "client_id": CLIENT_ID, "role": "node", "run_id": RUN_ID,
        "node_id": node_id, "mic_fs": mic_fs, "spk_fs": spk_fs,
        "measured": measured, "back_channel": bc,
        "status": "ok" if ok else "fail",
    }
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
