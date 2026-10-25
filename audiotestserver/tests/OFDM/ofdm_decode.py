"""
ofdm_decode.py  --  role: decode  (pair with ofdm_play.py)

Single continuous recording session covering probe + gap + data.

Signal sequence:
  1. Wait    ofdm_config.json
  2. Signal  'ofdm_armed'           (recording starts NOW)
  3. Record  live audio
  4. Wait    'ofdm_probe_played'    (probe finished transmitting)
  5. Wait    extra margin           (audio in flight + buffer drain)
  6. Analyse probe buffer -> SNR -> BAT
  7. Write   ofdm_bat.json
  8. Signal  'ofdm_bat_ready'
  9. Signal  'ofdm_data_armed'
 10. Continue recording
 11. Wait    'ofdm_played' + tail_secs
 12. Stop recording, decode data frames, report

Params (JSON field presented to user):
{
    "timeout":          120,
    "mic_gain_db":      0,
    "sc_threshold":     0.25,
    "tail_secs":        5,
    "snr_margin_db":    6.0,
    "max_bits_per_bin": 6,
    "n_fft":            1024,
    "cp_ratio":         0.25,
    "freq_lo":          1000,
    "freq_hi":          16000,
    "pilot_spacing":    8,
    "sample_rate":      48000,
    "ecc_bytes":        16,
    "chunk_bytes":      32,
    "seq_bytes":        2,
    "taper_ms":         2.0,
    "frame_gap_ms":     20.0,
    "probe_duration_s": 4.0,
    "sync_period":      32
}
"""
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

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

from testlib import (
    CLIENT_ID, INPUT_DEV, RUN_DIR, PARAMS, RUN_ID,
    log, to_int_dev, coord_signal, coord_check, coord_wait_any,
)

from ofdm_modem import (build_config, Modem, AdaptiveDemodulator,
                        bpsk_all_bat, carrier_spectrum)

TIMEOUT     = float(PARAMS.get("timeout",          120))
MIC_GAIN_DB = float(PARAMS.get("mic_gain_db",        0))
SC_THRESH   = float(PARAMS.get("sc_threshold",      0.25))
TAIL_SECS   = float(PARAMS.get("tail_secs",           5))
SNR_MARGIN  = float(PARAMS.get("snr_margin_db",      6.0))
MAX_BITS    = int(PARAMS.get("max_bits_per_bin",      6))
# Extra recording margin after probe_played signal:
# allows for audio-in-flight, buffer drain, and processing startup latency.
PROBE_TAIL_S = 1.5

def _wait_config(timeout=40, poll=0.5):
    cfg_path = RUN_DIR / "ofdm_config.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cfg_path.exists():
            return json.loads(cfg_path.read_text())
        time.sleep(poll)
    return None

def _build_cfg_from_params(raw_cfg):
    """
    Rebuild config from our own PARAMS so a stale ofdm_config.json with
    different freq_lo/freq_hi cannot corrupt this run.  Fields absent from
    PARAMS fall back to whatever the player wrote (raw_cfg).
    """
    def _p(key, default):
        return PARAMS.get(key, raw_cfg.get(key, default))

    cfg = build_config(
        n_fft            = int(_p("n_fft",            1024)),
        cp_ratio         = float(_p("cp_ratio",        0.25)),
        freq_lo          = float(_p("freq_lo",         raw_cfg.get("freq_lo_hz", 1000))),
        freq_hi          = float(_p("freq_hi",         raw_cfg.get("freq_hi_hz", 16000))),
        pilot_spacing    = int(_p("pilot_spacing",     8)),
        sample_rate      = int(_p("sample_rate",       48000)),
        ecc_bytes        = int(_p("ecc_bytes",         16)),
        chunk_bytes      = int(_p("chunk_bytes",       256)),
        seq_bytes        = int(_p("seq_bytes",         2)),
        taper_ms         = float(_p("taper_ms",        2.0)),
        frame_gap_ms     = float(_p("frame_gap_ms",    20.0)),
        probe_duration_s = float(_p("probe_duration_s", 4.0)),
        snr_margin_db    = SNR_MARGIN,
        max_bits_per_bin = MAX_BITS,
        sync_period      = int(_p("sync_period",       32)),
    )
    # Carry over total_frames if player already wrote it
    if "total_frames" in raw_cfg:
        cfg["total_frames"] = raw_cfg["total_frames"]
    return cfg

def main():
    in_dev = to_int_dev(INPUT_DEV)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[{CLIENT_ID}] ofdm_decode | input_device={in_dev}")

    log("  waiting for ofdm_config.json...")
    raw_cfg = _wait_config(timeout=40)
    if raw_cfg is None:
        log("  ERROR: ofdm_config.json not found within 40s")
        _save_error("no_config"); return

    cfg   = _build_cfg_from_params(raw_cfg)
    modem = Modem(cfg)
    sr    = cfg["sample_rate"]
    log(f"  config: n_fft={cfg['n_fft']}  "
        f"band={cfg['freq_lo_hz']:.0f}-{cfg['freq_hi_hz']:.0f} Hz  "
        f"carriers={cfg['n_data_carriers']}  "
        f"probe={cfg['actual_probe_s']}s ({cfg['n_probe_syms']} syms)")

    # -- Start continuous live recording immediately ---------------------------
    buf:  list[np.ndarray] = []
    lock = threading.Lock()
    stop = threading.Event()

    def _record_thread():
        try:
            import sounddevice as sd
            kw = dict(samplerate=sr, channels=1, dtype="float32")
            if in_dev is not None:
                kw["device"] = in_dev
            def cb(indata, frames, t, status):
                with lock:
                    buf.append(indata[:, 0].copy())
            with sd.InputStream(callback=cb, **kw):
                while not stop.is_set():
                    time.sleep(0.1)
        except Exception as exc:
            log(f"  record error: {exc}")

    rec = threading.Thread(target=_record_thread, daemon=True)
    rec.start()
    # Give the stream a moment to actually open before signalling
    time.sleep(0.3)

    coord_signal("ofdm_armed")
    log("  signalled ofdm_armed -- recording (waiting for probe)...")

    # -- Wait for probe to finish, then grab probe buffer ---------------------
    probe_timeout = cfg["actual_probe_s"] + 60
    coord_wait_any("ofdm_probe_played", timeout=probe_timeout)

    # Wait for audio in flight + buffer drain before snapshotting
    time.sleep(PROBE_TAIL_S)

    with lock:
        probe_samples = (np.concatenate(buf).copy()
                         if buf else np.zeros(0, dtype=np.float32))

    if MIC_GAIN_DB:
        probe_samples = probe_samples * (10 ** (MIC_GAIN_DB / 20))

    expected_probe_s = cfg["actual_probe_s"]
    captured_s       = len(probe_samples) / sr
    log(f"  probe capture: {captured_s:.2f}s "
        f"(expected probe ~{expected_probe_s:.2f}s + startup latency)")
    if captured_s < expected_probe_s * 0.5:
        log(f"  WARNING: capture is only {captured_s:.2f}s -- "
            f"less than half the {expected_probe_s:.2f}s probe duration. "
            "Check that recording started before probe played.")

    # -- Analyse probe -> SNR -> BAT -------------------------------------------
    result = modem.analyze_probe(probe_samples, threshold=SC_THRESH)

    if not result["valid"]:
        log(f"  WARNING: probe sync failed ({result.get('error')}) -- "
            "using BPSK fallback on all carriers")
        bat = np.ones(cfg["n_data_carriers"], dtype=np.int32)
    else:
        snr = result["snr_db"]
        log(f"  probe: {result['n_syms']}/{cfg['n_probe_syms']} syms averaged  "
            f"SNR min={snr.min():.1f}  median={float(np.median(snr)):.1f}  "
            f"max={snr.max():.1f} dB")

        # Per-octave band breakdown
        sc_hz     = cfg["sc_spacing_hz"]
        pilot_set = set(range(0, cfg["n_occupied"], cfg["pilot_spacing"]))
        data_snr  = np.array([snr[i] for i in range(cfg["n_occupied"])
                               if i not in pilot_set])
        bands = [(500, 2000), (2000, 5000), (5000, 10000), (10000, 20000)]
        parts = []
        for flo, fhi in bands:
            b0 = max(0, int((flo - cfg["freq_lo_hz"]) / sc_hz))
            b1 = min(len(data_snr), int((fhi - cfg["freq_lo_hz"]) / sc_hz))
            if 0 < b0 < b1 or (b0 == 0 and b1 > 0):
                med = float(np.median(data_snr[max(0, b0):b1]))
                parts.append(f"{flo//1000}-{fhi//1000}kHz:{med:.0f}dB")
        if parts:
            log(f"  SNR bands: {' | '.join(parts)}")

        bat = modem.allocate(result, snr_margin_db=SNR_MARGIN, max_bits=MAX_BITS)
        active = int(np.sum(bat > 0))
        if active == 0:
            needed = 4.0 + SNR_MARGIN
            log(f"  WARNING: all {cfg['n_data_carriers']} carriers below threshold "
                f"(need >={needed:.0f}dB for BPSK, got max {snr.max():.1f}dB) -- "
                "falling back to BPSK; expect RS errors")
            # Substitute the table, not just the log line: ofdm_bat.json is
            # the channel by which ofdm_play learns what to transmit, so a
            # 0-bit table here would size frames the sender never airs.
            bat = bpsk_all_bat(cfg['n_data_carriers'])

    log(f"  BAT: {modem.bat_summary(bat)}")

    bat_path = RUN_DIR / "ofdm_bat.json"
    Modem.save_bat(bat, str(bat_path))

    if result.get("valid"):
        (RUN_DIR / "ofdm_snr.json").write_text(json.dumps({
            "snr_db": result["snr_db"].tolist(),
            "n_syms": result["n_syms"],
            "bits_per_bin": bat.tolist(),
        }, indent=2))

    coord_signal("ofdm_bat_ready")
    coord_signal("ofdm_data_armed")
    log("  signalled ofdm_bat_ready + ofdm_data_armed -- recording data...")

    # -- Continue recording until data played + tail ---------------------------
    played_time = None
    deadline    = time.time() + TIMEOUT
    total_frames = cfg.get("total_frames", "-")

    while time.time() < deadline:
        if played_time is None and coord_check("ofdm_played"):
            played_time = time.time()
            log(f"  ofdm_played -- waiting {TAIL_SECS}s tail")
        if played_time and time.time() - played_time > TAIL_SECS:
            break
        # Reload config in case player updated total_frames after encoding
        if total_frames == "-" and (RUN_DIR / "ofdm_config.json").exists():
            try:
                updated = json.loads((RUN_DIR / "ofdm_config.json").read_text())
                if "total_frames" in updated:
                    total_frames = updated["total_frames"]
                    cfg["total_frames"] = total_frames
            except Exception:
                pass
        time.sleep(0.5)

    stop.set()
    rec.join(timeout=5)

    with lock:
        all_samples = (np.concatenate(buf) if buf
                       else np.zeros(0, dtype=np.float32))
    if MIC_GAIN_DB:
        all_samples = all_samples * (10 ** (MIC_GAIN_DB / 20))

    wav_path = RUN_DIR / f"{CLIENT_ID}_input.wav"
    if len(all_samples) > 0:
        wavfile.write(str(wav_path), sr,
                      (all_samples * 32767).clip(-32768, 32767).astype(np.int16))
        log(f"  full capture: {len(all_samples)/sr:.2f}s -> {wav_path.name}")

    # -- Decode data frames ----------------------------------------------------
    frame_stats: list[dict] = []

    def _on_frame(st):
        frame_stats.append(st)
        log(f"  frame {st['seq']+1}/{total_frames}  "
            f"errs={st['err_count']}/{st['ecc_capacity']}  "
            f"headroom={st['ecc_headroom_pct']}%  "
            f"snr={st.get('snr_evm_db',0):.1f}dB  "
            f"evm={st.get('evm_pct',100):.1f}%  "
            f"avg_bits={st.get('avg_bits_per_bin',0):.1f}  "
            f"net={st['net_bps']}bps")

    # The full recording contains: [probe audio] [gap] [data audio].
    # The probe has its own SC preamble; passing the full buffer to
    # decode_data lets the SC metric latch onto the probe SC first, and
    # min_gap (= frame length, potentially 10-15s) then blocks the data
    # frame that sits only a few seconds later. Skip everything up to the
    # probe snapshot point (taken just after ofdm_probe_played + PROBE_TAIL_S)
    # so the data frame starts after that boundary in all_samples.
    probe_end_idx = len(probe_samples)
    if len(all_samples) > probe_end_idx:
        data_input = all_samples[probe_end_idx:]
        log(f"  data decode window: {len(data_input)/sr:.2f}s  "
            f"(trimmed {probe_end_idx/sr:.2f}s probe from front)")
    else:
        data_input = all_samples
        log("  WARNING: recording too short to trim probe -- using full buffer")

    stats, chunks = modem.decode_data(data_input, bat, SC_THRESH, _on_frame)
    text = AdaptiveDemodulator.reassemble(chunks)

    n_decoded = len([s for s in frame_stats if not s.get("duplicate")])
    status = ("ok"      if total_frames != "-" and n_decoded >= int(total_frames)
              else "partial" if n_decoded > 0
              else "fail")

    def _avg(k, d=0):
        vals = [s.get(k, d) for s in frame_stats if not s.get("duplicate")]
        return round(sum(vals) / len(vals), 2) if vals else d

    result_out = {
        "client_id":        CLIENT_ID,
        "role":             "decode",
        "run_id":           RUN_ID,
        "status":           status,
        "frames_decoded":   n_decoded,
        "frames_expected":  total_frames,
        "total_rs_errors":  sum(s["err_count"] for s in frame_stats
                                if not s.get("duplicate")),
        "avg_snr_evm_db":   _avg("snr_evm_db"),
        "avg_evm_pct":      _avg("evm_pct", 100),
        "avg_ecc_headroom": _avg("ecc_headroom_pct"),
        "avg_bits_per_bin": _avg("avg_bits_per_bin"),
        "net_bps":          frame_stats[0]["net_bps"] if frame_stats else 0,
        "active_carriers":  int(np.sum(bat > 0)),
        "bat_summary":      modem.bat_summary(bat),
        "probe_valid":      result.get("valid", False),
        "probe_n_syms":     result.get("n_syms", 0),
        "probe_snr_min":    round(float(result["snr_db"].min()), 1)
                            if result.get("valid") else None,
        "probe_snr_max":    round(float(result["snr_db"].max()), 1)
                            if result.get("valid") else None,
        # per-subcarrier frequency/SNR/bits for the report's BAT chart
        "spectrum":         (carrier_spectrum(cfg, result["snr_db"], bat)
                             if result.get("valid") else None),
        "decoded_text":     text,
        "wav_capture":      str(wav_path) if wav_path.exists() else None,
    }

    log(f"\n  === OFDM Decode Summary ===")
    log(f"  Frames        : {n_decoded}/{total_frames}")
    log(f"  RS errors     : {result_out['total_rs_errors']}")
    log(f"  Avg SNR (EVM) : {result_out['avg_snr_evm_db']} dB")
    log(f"  Avg bits/bin  : {result_out['avg_bits_per_bin']}")
    log(f"  Net bitrate   : {result_out['net_bps']} bps")
    log(f"  BAT           : {result_out['bat_summary']}")
    if result.get("valid"):
        log(f"  Probe SNR     : {result_out['probe_snr_min']}.."
            f"{result_out['probe_snr_max']} dB  "
            f"({result_out['probe_n_syms']} syms averaged)")

    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(
        json.dumps(result_out, indent=2))
    print(json.dumps(result_out), flush=True)

def _save_error(reason):
    r = {"client_id": CLIENT_ID, "role": "decode",
         "run_id": RUN_ID, "status": "error", "failure_reason": reason}
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r), flush=True)

if __name__ == "__main__":
    main()
