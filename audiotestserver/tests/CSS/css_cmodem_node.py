# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_cmodem_node.py — continuous half-duplex modem node, over-the-air (Stage 3).

SYMMETRIC: every host runs this same script (assign it to all hosts under one
role).  Each boots a ModemDaemon (CSMA/CA + drift-driven re-chirp, validated
offline in css_modem_daemon_sim) and binds it to real audio:

  * RX  — a continuous Recorder captures the mic.  Mic RMS over a short trailing
          window is the CARRIER SENSE.  When a transmission (an energy burst)
          ENDS, that burst segment is decoded with the real css_decoder
          (decode_array) and each frame is handed to the daemon.  The daemon
          ignores its own src, so hearing our own self-echo is harmless.
  * TX  — when the daemon's MAC releases a packet (channel idle + backoff
          elapsed) we encode it (at the speaker's native rate) and play it,
          blocking for the ~46 s airtime; the Recorder keeps capturing throughout.

Node IDs are settable via the `id_map` param (CLIENT_ID -> id); defaults map
macdesktop->1, laptop->2.  Carrier-sense threshold auto-calibrates from the
ambient floor measured at startup.

Params: run_s, symbol_dur, f_min, f_max, amplitude, chunk_size, diff_encode,
        rechirp_min_s, rechirp_max_s, cw_s, id_map, carrier_k, min_burst_s.
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
    log, to_int_dev, Recorder, play_blocking, decode_array, save_wav,
)
import css_modem as cm                                       # noqa: E402
from announcelib import recipe, bake_decode_info             # noqa: E402
from css_modem_fw import ModemFirmware                        # noqa: E402
from css_modem_daemon import ModemDaemon                      # noqa: E402

DEFAULT_ID_MAP = {"macdesktop": 1, "laptop": 2}


def _node_id() -> int:
    id_map = PARAMS.get("id_map", DEFAULT_ID_MAP)
    if CLIENT_ID in id_map:
        return int(id_map[CLIENT_ID])
    return (abs(hash(CLIENT_ID)) % 15) + 1        # fallback: stable-ish 1..15


def _rms(x) -> float:
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def _make_chirp_detector(f_min, f_max, symbol_dur, fs):
    """Chirp-matched carrier sense.  Broadband RMS can't tell signal from a
    constant-noise floor (the decoder can, because it matched-filters — so must
    carrier sense).  Returns (score_fn, window_len): score_fn(window) is the
    normalised matched-filter peak against an up- OR down-chirp template (~4x
    higher when a chirp is present than for noise, measured on real captures)."""
    from css_encoder import generate_chirp
    up = generate_chirp(f_min, f_max, symbol_dur, fs).astype(np.float32)
    dn = generate_chirp(f_max, f_min, symbol_dur, fs).astype(np.float32)
    win = int(2 * symbol_dur * fs)            # >= one symbol at any alignment
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
        denom = float(np.sqrt(np.sum(w * w))) * eref + 1e-12
        return max(su, sd) / denom

    return score, win


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

    node_id = _node_id()
    run_s        = float(PARAMS.get("run_s", 240.0))
    tick_s       = float(PARAMS.get("tick_s", 0.4))
    sense_win_s  = float(PARAMS.get("sense_win_s", 0.4))
    min_burst_s  = float(PARAMS.get("min_burst_s", 8.0))
    decode_lookback_s = float(PARAMS.get("decode_lookback_s", 50.0))  # ~one beacon
    carrier_k    = float(PARAMS.get("carrier_k", 2.2))   # x median ambient MF (clamped)
    guard_s      = float(PARAMS.get("guard_s", 1.0))

    log(f"[{CLIENT_ID}] CONTINUOUS NODE id={node_id} cap_fs={cap_fs} "
        f"out_fs={out_fs} in={in_dev} out={out_dev} run={run_s}s")

    # PER-NODE workdir: both hosts share RUN_DIR on the SMB results share, so all
    # scratch files (encoded TX wavs, the baked decode info) MUST be namespaced by
    # client, or the two nodes write identically-named files and race on a
    # half-written decinfo_*.json.
    node_work = RUN_DIR / f"work_{CLIENT_ID}"
    node_work.mkdir(parents=True, exist_ok=True)
    # PER-NODE transmit amplitude: the two links are asymmetric — a node with a
    # weak speaker / on the far end of an insensitive mic must TX louder so the
    # peer can carrier-sense it, while a node whose signal already clips the
    # peer's mic must TX quieter.  amp_map (CLIENT_ID -> amplitude) sets each
    # node's OWN transmit level; amplitude only scales what THIS node emits.
    phy = recipe(out_fs)
    amp_map = PARAMS.get("amp_map", {})
    if CLIENT_ID in amp_map:
        phy["amplitude"] = float(amp_map[CLIENT_ID])
    log(f"  TX amplitude={phy['amplitude']}")
    # Firmware encodes at the speaker rate; decode info baked at the mic rate.
    fw = ModemFirmware(str(node_work / "cfg.json"),
                       node_id=node_id, workdir=str(node_work),
                       phy_override=phy, verbose=True).boot()
    dec_info = bake_decode_info(cap_fs, str(node_work))
    daemon = ModemDaemon(
        fw,
        cw_s=float(PARAMS.get("cw_s", 8.0)),
        rechirp_min_s=float(PARAMS.get("rechirp_min_s", 70.0)),
        rechirp_max_s=float(PARAMS.get("rechirp_max_s", 110.0)),
        first_beacon_stagger_s=float(PARAMS.get("first_beacon_stagger_s", 20.0)),
        ack_stagger_s=float(PARAMS.get("ack_stagger_s", 6.0)))

    rec = Recorder(in_dev, cap_fs)
    rec.start()
    time.sleep(0.3)

    # Auto-calibrate carrier-sense threshold from the ambient floor.  NO high
    # fixed floor: a cross-node beacon can land only just above ambient on a weak
    # mic (seen at ~0.0015 vs 0.0004 ambient), and a 0.004 floor would make that
    # peer invisible to carrier sense.  Schmitt-trigger: rise on on_thr, fall on
    # a lower off_thr held for hold_s, so a weak/noisy burst detects without
    # flicker (which would chop a beacon into premature partial decodes).
    # Chirp-matched carrier sense (robust to the constant background noise the
    # decoder already copes with).  Calibrate the busy threshold from the ambient
    # MF-score floor: on_thr = k * ambient (k~2.5 sits ~halfway between the noise
    # floor and an in-beacon score), clamped so a bad calibration can't wedge it.
    r = recipe(cap_fs)
    chirp_score, mf_win = _make_chirp_detector(r["f_min"], r["f_max"],
                                               r["symbol_dur"], cap_fs)
    # The busy decision runs on a MEDIAN-SMOOTHED score (~3s): a single-window MF
    # score has too little processing gain, so this noisy mic's floor fluctuates
    # up INTO the raw threshold and the Schmitt trigger sticks "busy" on noise
    # (measured: raw noise 0.04-0.085 straddling on_thr; smoothed it settles ~0.04
    # while a real beacon holds ~0.15).  Calibrate the threshold from the 85th
    # PERCENTILE of ambient (captures the noise SPREAD, not a lucky-quiet median)
    # so on_thr lands in the clean gap ABOVE noise, BELOW signal.
    smooth_n = int(PARAMS.get("smooth_n", 4))
    time.sleep(3.0)
    cal = rec.data()[-int(3.0 * cap_fs):]
    step = int(0.4 * cap_fs)
    cal_scores = [chirp_score(cal[i:i + mf_win])
                  for i in range(0, max(1, len(cal) - mf_win), step)]
    # Calibrate from the ambient MEDIAN (matches the runtime SMOOTHED score's
    # scale — an 85th-percentile of RAW scores overshoots once smoothed) and hold
    # on_thr inside the validated noise/signal gap: across many hardware runs the
    # smoothed noise floor is ~0.04 and a real beacon holds ~0.10-0.16, so a tight
    # [thr_lo, thr_hi] clamp keeps the gate correct whether this node's mic is
    # quiet or noisy this run.
    amb = float(np.median(cal_scores)) if cal_scores else 0.035
    thr_lo = float(PARAMS.get("carrier_thr_lo", 0.05))
    thr_hi = float(PARAMS.get("carrier_thr_hi", 0.12))
    on_thr = min(thr_hi, max(thr_lo, carrier_k * amb))
    off_thr = min(0.7 * on_thr, max(0.045, 1.3 * amb))
    hold_s = float(PARAMS.get("busy_hold_s", 2.0))
    difs_s = float(PARAMS.get("difs_s", 3.0))     # sustained-idle guard before TX
    log(f"  amb={amb:.4f} -> on_thr={on_thr:.4f} off_thr={off_thr:.4f} "
        f"smooth_n={smooth_n} mf_win={mf_win} difs={difs_s}s")

    t0 = time.time()
    win = mf_win
    score_hist: list = []       # recent raw scores; median = smoothed carrier
    busy = False
    busy_since_frame = None
    below_since = None
    idle_since = 0.0            # channel idle since this time (DIFS accrual)
    last_status = -1e9
    n_rechirp = 0
    tx_log, rx_log = [], []

    def _decode_burst(seg, now):
        try:
            res = decode_array(seg, cap_fs, dec_info)
            for f in res.get("frames", []):
                if "bytes_hex" not in f:
                    continue
                p = cm.unpack_packet(bytes.fromhex(f["bytes_hex"]))
                rx_log.append({"t": round(now, 1), "src": p["src"],
                               "type": p["type_name"], "seq": p["seq"]})
                daemon.handle_frame(f, now)
            for e in daemon.drain_events():
                log(f"  t={now:6.1f} EVENT {e['kind']} peer={e.get('peer')}")
        except Exception as exc:
            log(f"  decode error: {exc}")

    while time.time() - t0 < run_s:
        now = time.time() - t0
        buf = rec.data()
        n = len(buf)
        score_hist.append(chirp_score(buf[max(0, n - win):n]))
        if len(score_hist) > smooth_n:
            score_hist.pop(0)
        sc = float(np.median(score_hist))    # smoothed carrier score (busy decision)

        if not busy:
            if sc > on_thr:
                busy = True
                busy_since_frame = max(0, n - win)
                below_since = None
        else:
            if sc < off_thr:
                if below_since is None:
                    below_since = now
                elif now - below_since >= hold_s:      # sustained silence: burst ended
                    busy = False
                    burst_len = (n - (busy_since_frame or n)) / cap_fs
                    if burst_len >= min_burst_s:
                        # Decode a FIXED lookback (~one beacon) ending at the
                        # burst end, NOT from busy_since: carrier detection can
                        # fire tens of seconds late on a marginal link, and
                        # decoding only from there would miss the preamble and
                        # fail.  A full-beacon lookback always spans the frame.
                        lo = max(0, n - int(decode_lookback_s * cap_fs))
                        seg = buf[lo:n]
                        log(f"  t={now:6.1f} burst {burst_len:.1f}s -> decoding "
                            f"last {len(seg)/cap_fs:.1f}s")
                        _decode_burst(seg, now)     # BLOCKS seconds; carrier frozen
                    busy_since_frame = None
                    below_since = None
                    idle_since = time.time() - t0   # restart DIFS after the (blocking) decode
            else:
                below_since = None

        # DIFS: only treat the channel as available once it has been idle for a
        # sustained guard.  This keeps a node from starting at the ragged edge of
        # a peer's burst, and — because decode blocks the loop — from firing the
        # instant a decode returns (which would re-sync both nodes into a clash).
        if busy:
            idle_since = None
        elif idle_since is None:
            idle_since = now
        sustained_idle = (idle_since is not None) and (now - idle_since >= difs_s)
        sense_busy = busy or (not sustained_idle)

        if now - last_status >= 15.0:
            last_status = now
            log(f"  t={now:6.1f} mf={sc:.4f} busy={busy} idle={sustained_idle} "
                f"next_beacon_in={max(0, daemon._next_rechirp_s - now):.0f}s "
                f"discovered={sorted(daemon.peers)}")

        # MAC: poll_tx releases a packet only when the channel is (sustained) idle
        # and the backoff has elapsed; it also queues the drift-driven re-chirp.
        rt = daemon.poll_tx(now, sense_busy)
        if rt is not None:
            raw, tag = rt
            # FINAL carrier check on the freshest ~0.9s: the 2-symbol busy window
            # lags ~1s, so a peer that began transmitting during our backoff is
            # invisible to `busy` but IS in the last 0.9s.  Catching it here turns
            # a ~1.5s collision window into ~0.9s (sub-0.9s simultaneous starts
            # remain unavoidable without collision detection — bigger cw_s makes
            # those rare).
            fresh = rec.data()
            quick = chirp_score(fresh[max(0, len(fresh) - int(0.9 * cap_fs)):])
            if quick > on_thr:
                daemon.outbox.insert(0, (raw, tag, 0.0))  # re-queue ready, retry after peer
                daemon._backoff_remaining = None
                busy = True
                idle_since = None
                log(f"  t={now:6.1f} ABORT TX {tag} — peer detected "
                    f"(final mf={quick:.4f} > {on_thr:.4f})")
            else:
                pcm = fw.encode_packet(raw, tag)
                pkt = cm.unpack_packet(raw)
                log(f"  t={now:6.1f} TX {pkt['type_name']} dst={pkt['dst']} "
                    f"seq={pkt['seq']} ({len(pcm)/out_fs:.1f}s)")
                tx_log.append({"t": round(now, 1), "type": pkt["type_name"],
                               "dst": pkt["dst"], "seq": pkt["seq"]})
                if tag.startswith("beacon:"):
                    n_rechirp += 1
                play_blocking(pcm, out_fs, out_dev)
                daemon.on_tx(tag, len(rec.data()))
                # Our own self-echo just filled the mic; reset carrier state so its
                # decaying tail isn't mistaken for a peer burst, and re-arm the
                # DIFS guard so we listen a full guard before the next transmit.
                busy = False
                busy_since_frame = None
                below_since = None
                idle_since = time.time() - t0
        else:
            time.sleep(tick_s)

    rec.stop()
    cap_path = RUN_DIR / f"{CLIENT_ID}_cmodem.wav"
    save_wav(rec.data(), cap_fs, cap_path)

    discovered = sorted({e for p, r in daemon.peers.items() for e in [p]})
    drift = {p: round(r["drift_ppm"], 1) for p, r in daemon.peers.items()}
    result = {"status": "ok", "role": "node", "client_id": CLIENT_ID,
              "node_id": node_id, "discovered": discovered,
              "n_beacons_tx": n_rechirp, "drift_ppm": drift,
              "rechirp_interval_s": round(daemon.rechirp_interval_s, 1),
              "n_rx": len(rx_log), "rx": rx_log, "tx": tx_log,
              "on_thr": round(on_thr, 5), "off_thr": round(off_thr, 5),
              "cap_fs": cap_fs, "out_fs": out_fs, "wav": cap_path.name}
    (RUN_DIR / f"cmodem_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
    log(f"  RESULT id={node_id} discovered={discovered} beacons_tx={n_rechirp} "
        f"drift={drift}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
