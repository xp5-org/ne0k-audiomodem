# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""
tests/probe_listen.py  -  role: listen  (pair with probe_play.py)

Records the probe sequence and analyses each frequency window: single FFT
per window (not a per-frequency Goertzel loop), timing alignment against
expected energy, ASCII SNR chart, per-frequency verdict with reason.

The full recording is saved to <RUN_DIR>/<CLIENT_ID>_input.wav before
analysis, for regenerating the spectrograph, comparing captures across
runs, or offline analysis.

Outputs:
  <CLIENT_ID>_input.wav      - raw mic recording
  specprobe_report.json      - full per-frequency measurements
  result_<client>.json       - summary + recommended QAM subcarrier set
"""

import json
import threading
import time
from pathlib import Path

import numpy as np

from testlib import (
    CLIENT_ID, INPUT_DEV, RUN_DIR, PARAMS, RUN_ID, SERVER_URL,
    log, to_int_dev, coord_signal, coord_check, coord_wait_any, save_wav,
)

TAIL_SECS   = float(PARAMS.get("tail_secs",    2.0))
SNR_64QAM   = float(PARAMS.get("snr_64qam",   30.0))
SNR_16QAM   = float(PARAMS.get("snr_16qam",   20.0))
SNR_4QAM    = float(PARAMS.get("snr_4qam",    12.0))
MAX_THD_DB  = float(PARAMS.get("max_thd_db",  -15.0))
MAX_ECHO_DB = float(PARAMS.get("max_echo_db", -10.0))
ALIGN_SEARCH_SECS = float(PARAMS.get("align_search_secs", 2.0))


# ---------------------------------------------------------------------------
# DSP
# ---------------------------------------------------------------------------

def fft_power_at(window, target_freq, fs):
    n   = len(window)
    win = np.hanning(n)
    X   = np.fft.rfft(window * win)
    f   = np.fft.rfftfreq(n, 1.0 / fs)
    idx = int(np.argmin(np.abs(f - target_freq)))
    win_energy = float(np.sum(win ** 2))
    return float(2 * np.abs(X[idx]) ** 2 / (win_energy * n))


def fft_powers_at(window, freqs, fs):
    """Compute power at multiple frequencies in one FFT pass."""
    n   = len(window)
    win = np.hanning(n)
    X   = np.fft.rfft(window * win)
    f   = np.fft.rfftfreq(n, 1.0 / fs)
    win_energy = float(np.sum(win ** 2))
    norm = win_energy * n
    powers = {}
    for freq in freqs:
        idx = int(np.argmin(np.abs(f - freq)))
        powers[freq] = float(2 * np.abs(X[idx]) ** 2 / norm)
    return powers


def rms_power(samples):
    return float(np.mean(np.array(samples, dtype=np.float64) ** 2))


def db(power, ref=1.0):
    if power <= 0 or ref <= 0:
        return -120.0
    return float(10 * np.log10(power / ref))


def find_alignment_offset(recording, manifest, fs):
    first_tone_start = manifest["sequence"][0]["tone_start"]
    first_freq       = manifest["sequence"][0]["freq"]
    tone_samples     = int(manifest["tone_dur"] * fs)
    search_samples   = int(ALIGN_SEARCH_SECS * fs)

    best_offset = 0
    best_power  = -1.0

    for offset in range(0, search_samples, tone_samples // 4):
        start = offset
        end   = start + tone_samples
        if end > len(recording):
            break
        window = recording[start:end]
        p = fft_power_at(window, first_freq, fs)
        if p > best_power:
            best_power  = p
            best_offset = offset

    correction = best_offset - first_tone_start
    log(f"  timing alignment: first tone found at sample {best_offset} "
        f"(manifest says {first_tone_start}), correction={correction:+d} smp "
        f"({correction/fs*1000:+.0f}ms)")
    return correction


def analyse_recording(recording, manifest, fs):
    correction = find_alignment_offset(recording, manifest, fs)
    results    = []

    for entry in manifest["sequence"]:
        freq = entry["freq"]

        ns = max(0, entry["noise_start"] + correction)
        ts = max(0, entry["tone_start"]  + correction)
        te = min(len(recording), entry["tone_end"]   + correction)
        ds = max(0, entry["decay_start"] + correction)
        de = min(len(recording), entry["decay_end"]  + correction)

        noise_win = recording[ns:ts] if ns < len(recording) else np.array([])
        tone_win  = recording[ts:te] if ts < len(recording) else np.array([])
        decay_win = recording[ds:de] if ds < len(recording) else np.array([])

        if len(tone_win) < 256:
            results.append({"freq": freq, "status": "missing",
                             "snr_db": -120, "thd_db": 0, "echo_db": 0})
            continue

        measure_freqs = [freq]
        if freq * 2 < fs / 2:
            measure_freqs.append(freq * 2)
        if freq * 3 < fs / 2:
            measure_freqs.append(freq * 3)

        tone_powers = fft_powers_at(tone_win, measure_freqs, fs)

        sig_power = tone_powers[freq]
        sig_db    = db(sig_power)

        h2_power  = tone_powers.get(freq * 2, 0.0)
        h3_power  = tone_powers.get(freq * 3, 0.0)
        thd_power = h2_power + h3_power
        thd_db    = db(thd_power, ref=sig_power)

        noise_power = rms_power(noise_win) if len(noise_win) >= 64 else 1e-12
        noise_db    = db(noise_power)
        snr_db      = sig_db - noise_db

        if len(decay_win) >= 64:
            decay_power = fft_power_at(decay_win, freq, fs)
            echo_db     = db(decay_power, ref=sig_power)
        else:
            echo_db = -120.0

        results.append({
            "freq":     round(freq, 1),
            "sig_db":   round(sig_db, 1),
            "noise_db": round(noise_db, 1),
            "snr_db":   round(snr_db, 1),
            "thd_db":   round(thd_db, 1),
            "echo_db":  round(echo_db, 1),
            "status":   "ok",
        })

    return results


def ascii_snr_chart(freq_results, width=50):
    ok = [r for r in freq_results if r.get("status") == "ok"]
    if not ok:
        return
    max_snr = max(r["snr_db"] for r in ok)
    scale   = max(max_snr, 40.0)

    log("\n  SNR across spectrum:")
    log(f"  {'Freq':>8}  {'SNR bar':<{width}}  dB")
    log(f"  {'-'*8}  {'-'*width}  --")

    for r in ok:
        freq  = r["freq"]
        snr   = r["snr_db"]
        bar_w = max(0, int(snr / scale * width))
        if snr >= SNR_64QAM:
            ch = "#"
        elif snr >= SNR_16QAM:
            ch = "="
        elif snr >= SNR_4QAM:
            ch = "-"
        else:
            ch = "."
        bar = ch * bar_w
        log(f"  {freq:>8.0f}  {bar:<{width}}  {snr:.1f}")

    log(f"\n  Legend: {'#'} = 64-QAM (>{SNR_64QAM}dB)  "
        f"{'='} = 16-QAM (>{SNR_16QAM}dB)  "
        f"{'-'} = 4-QAM (>{SNR_4QAM}dB)  "
        f"{'.'} = unusable")


def recommend_subcarriers(freq_results):
    subcarriers, skipped = [], []

    for r in freq_results:
        if r.get("status") != "ok":
            continue
        freq, snr, thd, echo = r["freq"], r["snr_db"], r["thd_db"], r["echo_db"]

        if thd > MAX_THD_DB:
            skipped.append({"freq": freq,
                             "reason": f"distortion {thd:.1f}dB (max {MAX_THD_DB}dB)"})
            continue
        if echo > MAX_ECHO_DB:
            skipped.append({"freq": freq,
                             "reason": f"echo {echo:.1f}dB (max {MAX_ECHO_DB}dB)"})
            continue
        if snr < SNR_4QAM:
            skipped.append({"freq": freq,
                             "reason": f"SNR {snr:.1f}dB too low (min {SNR_4QAM}dB)"})
            continue

        if snr >= SNR_64QAM:
            qam, bps = 64, 6
        elif snr >= SNR_16QAM:
            qam, bps = 16, 4
        else:
            qam, bps = 4, 2

        subcarriers.append({"freq": freq, "qam": qam, "bits_per_symbol": bps,
                             "snr_db": snr, "thd_db": thd, "echo_db": echo})

    return subcarriers, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import sounddevice as sd

    in_dev = to_int_dev(INPUT_DEV)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[{CLIENT_ID}] specprobe_listen | input_device={in_dev}")

    # Wait for manifest to appear
    manifest_path = RUN_DIR / "specprobe_manifest.json"
    log("  waiting for specprobe_manifest.json...")
    deadline = time.time() + 40
    while time.time() < deadline:
        if manifest_path.exists():
            break
        time.sleep(0.3)
    else:
        log("  ERROR: manifest not found within 40s")
        _save_error("no_manifest")
        return

    ready = coord_check("specprobe_ready")
    log(f"  manifest found  player_ready={bool(ready)}")

    manifest    = json.loads(manifest_path.read_text())
    fs          = manifest["fs"]
    n_freqs     = len(manifest["sequence"])
    total_secs  = n_freqs * (manifest["silence_dur"] + manifest["tone_dur"]
                             + manifest["silence_dur"])
    record_secs = total_secs + ALIGN_SEARCH_SECS + TAIL_SECS + 2.0

    log(f"  {n_freqs} freqs, {total_secs:.1f}s probe, recording up to {record_secs:.1f}s")

    coord_signal("specprobe_armed")
    log("  armed - recording...")

    recording_holder = [None]

    def _record():
        recording_holder[0] = sd.rec(int(record_secs * fs), samplerate=fs,
                                     channels=1, dtype="float32", device=in_dev)
        sd.wait()

    rec_thread = threading.Thread(target=_record, daemon=True)
    rec_thread.start()

    played_time = None
    while rec_thread.is_alive():
        if played_time is None and coord_check("specprobe_played"):
            played_time = time.time()
            log(f"  player done - waiting {TAIL_SECS}s tail...")
        if played_time and time.time() - played_time > TAIL_SECS:
            sd.stop()
            break
        time.sleep(0.3)

    rec_thread.join(timeout=5)
    recording = recording_holder[0]
    if recording is None:
        log("  ERROR: no recording"); _save_error("no_recording"); return

    recording = recording.flatten()
    peak = float(np.max(np.abs(recording)))
    log(f"  recorded {len(recording)/fs:.1f}s  peak={peak:.3f}  "
        f"rms={np.sqrt(np.mean(recording**2)):.4f}")

    if peak < 0.001:
        log("  WARNING: recording is very quiet - check mic input level")

    # ── WAV capture ──────────────────────────────────────────────────────────
    wav_path = RUN_DIR / f"{CLIENT_ID}_input.wav"
    save_wav(recording, fs, wav_path)

    # Analyse
    log("\n  === Frequency Analysis ===")
    log(f"  {'Freq':>8}  {'Sig':>6}  {'Noise':>6}  {'SNR':>6}  "
        f"{'THD':>6}  {'Echo':>6}  Verdict")
    log(f"  {'----':>8}  {'---':>6}  {'-----':>6}  {'---':>6}  "
        f"{'---':>6}  {'----':>6}  -------")

    freq_results = analyse_recording(recording, manifest, fs)

    for r in freq_results:
        if r.get("status") == "missing":
            log(f"  {r['freq']:>8.0f}  [missing - recording too short]")
            continue
        freq = r["freq"]
        snr  = r["snr_db"]
        thd  = r["thd_db"]
        echo = r["echo_db"]

        if thd > MAX_THD_DB:       verdict = f"SKIP distortion {thd:.0f}dB"
        elif echo > MAX_ECHO_DB:   verdict = f"SKIP echo {echo:.0f}dB"
        elif snr < SNR_4QAM:       verdict = f"SKIP low SNR {snr:.0f}dB"
        elif snr >= SNR_64QAM:     verdict = "64-QAM"
        elif snr >= SNR_16QAM:     verdict = "16-QAM"
        else:                      verdict =  "4-QAM"

        log(f"  {freq:>8.0f}  {r['sig_db']:>+6.1f}  {r['noise_db']:>+6.1f}  "
            f"{snr:>6.1f}  {thd:>+6.1f}  {echo:>+6.1f}  {verdict}")

    ascii_snr_chart(freq_results)

    subcarriers, skipped = recommend_subcarriers(freq_results)
    total_bps = sum(s["bits_per_symbol"] for s in subcarriers)

    log(f"\n  === Summary ===")
    log(f"  Usable subcarriers : {len(subcarriers)} / {n_freqs}")
    log(f"  Skipped            : {len(skipped)}")
    log(f"  Bits per symbol    : {total_bps} (theoretical max throughput)")

    qam_counts = {}
    for s in subcarriers:
        qam_counts[s["qam"]] = qam_counts.get(s["qam"], 0) + 1
    for qam in sorted(qam_counts, reverse=True):
        log(f"    {qam}-QAM: {qam_counts[qam]} subcarriers")

    if skipped:
        log(f"\n  Skipped frequencies:")
        for s in skipped:
            log(f"    {s['freq']:>7.0f}Hz  {s['reason']}")

    report = {
        "run_id": RUN_ID, "client_id": CLIENT_ID, "fs": fs,
        "freq_results": freq_results,
        "subcarriers": subcarriers, "skipped": skipped,
        "total_bits_per_symbol": total_bps,
        "wav_capture": str(wav_path),
        "thresholds": {
            "snr_64qam": SNR_64QAM, "snr_16qam": SNR_16QAM,
            "snr_4qam": SNR_4QAM, "max_thd_db": MAX_THD_DB,
            "max_echo_db": MAX_ECHO_DB,
        },
    }
    (RUN_DIR / "specprobe_report.json").write_text(json.dumps(report, indent=2))

    result = {
        "client_id": CLIENT_ID, "role": "listen", "run_id": RUN_ID,
        "status": "ok",
        "freqs_measured": len(freq_results),
        "subcarriers_usable": len(subcarriers),
        "subcarriers_skipped": len(skipped),
        "total_bits_per_symbol": total_bps,
        "subcarriers": subcarriers,
        "wav_capture": str(wav_path),
    }
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def _save_error(reason):
    result = {"client_id": CLIENT_ID, "role": "listen",
              "run_id": RUN_ID, "status": "error", "failure_reason": reason}
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
