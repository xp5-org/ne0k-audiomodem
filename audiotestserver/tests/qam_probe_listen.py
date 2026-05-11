"""
tests/qam_probe_listen.py  -  role: listen  (pair with qam_probe_play.py)

Listens for each probe step, decodes it using qam_reader.py live mode,
and records per-step performance metrics. At the end ranks settings by:
  1. Most frames decoded (primary)
  2. Lowest RS error count (secondary)
  3. Highest ECC headroom (tertiary)

WAV capture
-----------
Each step's raw mic input is saved to:
  <RUN_DIR>/<CLIENT_ID>_input_step<NNN>.wav
These can be replayed with qam_reader.py for offline re-decoding or later
constellation analysis without needing to re-run the full probe sweep.

Outputs:
  <CLIENT_ID>_input_step<NNN>.wav  - per-step raw mic recordings (NEW)
  probe_results.json               - all per-step metrics
  probe_best.json                  - top-ranked settings
  result_<client>.json             - framework result

Params:
{
    "timeout_per_step":  30,
    "step_deadline_pad": 30,
    "mic_gain_db":        0,
    "zc_threshold":    0.45
}
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from analysis import make_qam_constellation_png, make_qam_probe_heatmap_png
from testlib import (
    CLIENT_ID, INPUT_DEV, RUN_DIR, PARAMS, RUN_ID, SERVER_URL,
    log, to_int_dev, coord_signal, coord_check, coord_wait, coord_wait_any,
)

TESTS_DIR = Path(__file__).parent

TIMEOUT_PER_STEP  = float(PARAMS.get("timeout_per_step",   30))
STEP_DEADLINE_PAD = float(PARAMS.get("step_deadline_pad",  30))
MIC_GAIN_DB       = float(PARAMS.get("mic_gain_db",         0))
ZC_THRESH         = float(PARAMS.get("zc_threshold",     0.45))
TAIL_SECS         = float(PARAMS.get("tail_secs",           3))


def coord_wait_manifest(timeout=60):
    manifest_path = RUN_DIR / "probe_manifest.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if manifest_path.exists():
            return manifest_path
        time.sleep(0.3)
    return None


def _kill_proc(proc: subprocess.Popen, label: str = ""):
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            pass
        log(f"  [proc] {label} didn't respond to SIGTERM, sending SIGKILL")
        proc.kill()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            log(f"  [proc] {label} didn't respond to SIGKILL - giving up")
    except OSError:
        pass


def decode_step(step: dict, in_dev) -> dict:
    """
    Decode one probe step using qam_reader.py live mode.

    The raw mic input for this step is saved to
    <RUN_DIR>/<CLIENT_ID>_input_step<NNN>.wav via --wav-out.
    """
    i           = step["step"]
    cf          = step["carrier_freq"]
    sr          = step["symbol_rate"]
    n_frames    = step.get("frames", "?")
    config_path = RUN_DIR / Path(step["config"]).name
    stats_path  = RUN_DIR / f"probe_{i:03d}_stats.json"
    wav_path    = RUN_DIR / f"{CLIENT_ID}_input_step{i:03d}.wav"
    label       = f"reader step {i+1} cf={cf} sr={sr}"

    step_duration_s = step.get("duration_s", TIMEOUT_PER_STEP)
    step_deadline   = time.time() + step_duration_s + STEP_DEADLINE_PAD
    reader_timeout  = step_duration_s + STEP_DEADLINE_PAD

    log(f"  step {i+1}: deadline = audio {step_duration_s:.1f}s + pad {STEP_DEADLINE_PAD:.0f}s "
        f"= {step_duration_s + STEP_DEADLINE_PAD:.0f}s total")

    args = [
        sys.executable, str(TESTS_DIR / "qam_reader.py"),
        "live",
        "--config",    str(config_path),
        "--timeout",   str(reader_timeout),
        "--stats-out", str(stats_path),
        "--threshold", str(ZC_THRESH),
        "--wav-out",   str(wav_path),       # per-step raw mic capture
    ]
    if in_dev is not None:
        args += ["--device", str(in_dev)]
    if MIC_GAIN_DB != 0:
        args += ["--mic-gain-db", str(MIC_GAIN_DB)]

    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    frame_stats  = []
    reader_ready = threading.Event()

    def _read():
        for line in proc.stdout:
            line = line.rstrip()
            if "Listening" in line:
                reader_ready.set()
            if line.startswith("FRAME_STAT:"):
                try:
                    frame_stats.append(json.loads(line[11:]))
                except Exception:
                    pass
            elif line.strip():
                log(f"    [reader] {line}")

    reader_thread = threading.Thread(target=_read, daemon=True)
    reader_thread.start()

    if not reader_ready.wait(timeout=5):
        log(f"  step {i+1}: reader slow to start - signalling anyway")
    coord_signal(f"probe_step_{i}_listen_ready")
    log(f"  step {i+1}: cf={cf}Hz sr={sr}baud - mic open, awaiting transmission "
        f"(~{step_duration_s:.0f}s audio)")

    played_time = None
    while time.time() < step_deadline:
        if proc.poll() is not None:
            log(f"  step {i+1}: reader exited naturally (rc={proc.returncode})")
            break
        if played_time is None and coord_check(f"probe_step_{i}_played"):
            played_time = time.time()
            log(f"  step {i+1}: played signal received, waiting {TAIL_SECS}s tail")
        if played_time and time.time() - played_time > TAIL_SECS:
            log(f"  step {i+1}: tail complete, stopping reader")
            break
        time.sleep(0.3)
    else:
        log(f"  step {i+1}: hard deadline reached "
            f"(played_received={played_time is not None}) — forcing stop")

    _kill_proc(proc, label)
    reader_thread.join(timeout=3)
    proc.wait()

    # Sync barrier: wait for played signal before moving to next step
    if played_time is None:
        log(f"  step {i+1}: waiting for played signal before advancing...")
        remaining = max(10.0, step_deadline - time.time() + STEP_DEADLINE_PAD)
        if coord_wait(f"probe_step_{i}_played", timeout=remaining):
            log(f"  step {i+1}: played signal received (post-decode)")
        else:
            log(f"  step {i+1}: WARNING: played signal never received")

    # Log WAV outcome
    if wav_path.exists():
        size_kb = wav_path.stat().st_size // 1024
        log(f"  step {i+1}: WAV capture: {wav_path.name}  ({size_kb} KB)")

    # Summarise
    n_decoded    = len(frame_stats)
    total_errs   = sum(f["err_count"]            for f in frame_stats)
    avg_headroom = (sum(f["ecc_headroom_pct"]    for f in frame_stats) / n_decoded
                    if n_decoded else 0)
    avg_snr      = (sum(f.get("snr_evm_db",    0) for f in frame_stats) / n_decoded
                    if n_decoded else 0)
    avg_agc      = (sum(f.get("agc_gain_db",   0) for f in frame_stats) / n_decoded
                    if n_decoded else 0)
    avg_input    = (sum(f.get("input_rms_db", -60) for f in frame_stats) / n_decoded
                    if n_decoded else -60)
    avg_zc       = (sum(f.get("zc_quality",    0) for f in frame_stats) / n_decoded
                    if n_decoded else 0)
    net_bps      = frame_stats[0]["net_bps"]   if frame_stats else 0
    coded_bps    = frame_stats[0]["coded_bps"] if frame_stats else 0

    if n_decoded == 0:
        status = "fail"
    elif isinstance(n_frames, int) and n_decoded < n_frames:
        status = "partial"
    else:
        status = "ok"

    result = {
        "step":                 i,
        "carrier_freq":         cf,
        "symbol_rate":          sr,
        "symbol_rate_req":      step.get("symbol_rate_requested", sr),
        "status":               status,
        "frames_decoded":       n_decoded,
        "frames_expected":      n_frames,
        "total_rs_errors":      total_errs,
        "avg_ecc_headroom_pct": round(avg_headroom, 1),
        "avg_snr_evm_db":       round(avg_snr, 1),
        "avg_agc_gain_db":      round(avg_agc, 1),
        "avg_input_rms_db":     round(avg_input, 1),
        "avg_zc_quality":       round(avg_zc, 3),
        "net_bps":              net_bps,
        "coded_bps":            coded_bps,
        "wav_capture":          str(wav_path) if wav_path.exists() else None,
    }

    # Per-step constellation PNG (built after result dict so we can update it)
    png_path = RUN_DIR / f"probe_{i:03d}_constellation.png"
    if stats_path.exists():
        cfg_path = RUN_DIR / Path(step["config"]).name
        try:
            step_cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        except Exception:
            step_cfg = {}
        step_cfg["carrier_freq"] = cf
        step_cfg["symbol_rate"]  = sr
        step_cfg["total_frames"] = n_frames
        if make_qam_constellation_png(
            stats_path, png_path, step_cfg,
            title=f"QAM Probe Step {i+1} — cf={cf}Hz sr={sr}baud"
        ):
            result["constellation_png"] = str(png_path)

    indicator = "OK " if status == "ok" else ("---" if status == "partial" else "FAIL")
    log(f"  [{indicator}] cf={cf}Hz sr={sr}baud  "
        f"frames={n_decoded}/{n_frames}  "
        f"errs={total_errs}  headroom={avg_headroom:.0f}%  "
        f"net={net_bps}bps")

    return result


def rank_results(step_results: list) -> list:
    return sorted(
        step_results,
        key=lambda r: (
            -r["frames_decoded"],
             r["total_rs_errors"],
            -r.get("avg_snr_evm_db", 0),
            -r["avg_ecc_headroom_pct"],
        )
    )


def print_report(ranked: list, total_steps: int):
    log("\n  === Probe Results (best first) ===")
    log(f"  {'Rank':>4}  {'Carrier':>8}  {'Sym/s':>6}  {'Frames':>8}  "
        f"{'Errs':>5}  {'SNR':>7}  {'ECC':>6}  {'AGC':>7}  "
        f"{'Input':>7}  {'ZC':>5}  {'Net bps':>8}  Status")
    log(f"  {'----':>4}  {'-------':>8}  {'-----':>6}  {'------':>8}  "
        f"{'----':>5}  {'---':>7}  {'---':>6}  {'------':>7}  "
        f"{'-----':>7}  {'--':>5}  {'-------':>8}  ------")
    for rank, r in enumerate(ranked, 1):
        log(f"  {rank:>4}  {r['carrier_freq']:>7}Hz  {r['symbol_rate']:>6}  "
            f"  {r['frames_decoded']}/{r['frames_expected']:>3}  "
            f"  {r['total_rs_errors']:>4}  "
            f"  {r.get('avg_snr_evm_db',0):>6.1f}dB  "
            f"  {r['avg_ecc_headroom_pct']:>4.0f}%  "
            f"  {r.get('avg_agc_gain_db',0):>+6.1f}dB  "
            f"  {r.get('avg_input_rms_db',-60):>6.1f}dB  "
            f"  {r.get('avg_zc_quality',0):>5.3f}  "
            f"  {r['net_bps']:>7}  {r['status']}")

    best = next((r for r in ranked if r["status"] == "ok"), None)
    if best:
        log(f"\n  Best settings:")
        log(f"    carrier_freq = {best['carrier_freq']}")
        log(f"    symbol_rate  = {best['symbol_rate']}")
        log(f"    net_bps      = {best['net_bps']}")
        log(f'\n  Paste into qam params: {{"carrier_freq": {best["carrier_freq"]}, '
            f'"symbol_rate": {best["symbol_rate"]}}}')
    else:
        log("\n  No steps fully decoded - try reducing symbol_rate or checking mic position")


def main():
    in_dev = to_int_dev(INPUT_DEV)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[{CLIENT_ID}] qam_probe_listen | input_device={in_dev}")

    log("  waiting for probe_manifest.json...")
    manifest_path = coord_wait_manifest(timeout=90)
    if not manifest_path:
        log("  ERROR: manifest not found")
        _save_error("no_manifest"); return

    manifest = json.loads(manifest_path.read_text())
    steps    = manifest["steps"]
    total_audio = sum(s.get("duration_s", 0) for s in steps)
    log(f"  manifest: {len(steps)} steps, ~{total_audio:.0f}s total audio")

    coord_signal("probe_armed")
    log("  armed")

    step_results = []
    for step in steps:
        result = decode_step(step, in_dev)
        step_results.append(result)

    # Wait for all_played in case last step timing was tight
    deadline = time.time() + 10
    while time.time() < deadline:
        if coord_check("probe_all_played"):
            break
        time.sleep(0.5)

    ranked = rank_results(step_results)

    # Probe sweep heatmap
    heatmap_path = RUN_DIR / "probe_heatmap.png"
    if make_qam_probe_heatmap_png(step_results, heatmap_path):
        log(f"  probe heatmap -> {heatmap_path.name}")

    print_report(ranked, len(steps))

    report = {
        "run_id":       RUN_ID,
        "client_id":    CLIENT_ID,
        "steps_probed": len(step_results),
        "results":      ranked,
    }
    (RUN_DIR / "probe_results.json").write_text(json.dumps(report, indent=2))

    best = next((r for r in ranked if r["status"] == "ok"), ranked[0] if ranked else None)
    best_params = None
    if best:
        best_params = {
            "carrier_freq": best["carrier_freq"],
            "symbol_rate":  best["symbol_rate"],
        }
        (RUN_DIR / "probe_best.json").write_text(json.dumps(best_params, indent=2))
        log(f"\n  probe_best.json written -> {best_params}")

    n_ok      = sum(1 for r in step_results if r["status"] == "ok")
    n_partial = sum(1 for r in step_results if r["status"] == "partial")
    status    = "ok" if n_ok > 0 else ("partial" if n_partial > 0 else "fail")

    final = {
        "client_id":     CLIENT_ID, "role": "listen",
        "run_id":        RUN_ID,    "status": status,
        "steps_probed":  len(step_results),
        "steps_ok":      n_ok,
        "steps_partial": n_partial,
        "best":          best_params,
        "ranked":        ranked,
    }
    heatmap_str = str(heatmap_path) if heatmap_path.exists() else None
    final["probe_heatmap"] = heatmap_str
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final), flush=True)


def _save_error(reason):
    result = {"client_id": CLIENT_ID, "role": "listen",
              "run_id": RUN_ID, "status": "error", "failure_reason": reason}
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
