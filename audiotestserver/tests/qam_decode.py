"""
tests/qam_decode.py  -  role: decode  (pair with qam_play.py)

Framework wrapper for qam_reader.Demodulator.
Records from the mic while the player transmits, then decodes.

WAV capture
-----------
The raw mic input is saved to  <RUN_DIR>/<CLIENT_ID>_input.wav  via
qam_reader.py's ``--wav-out`` flag.  This file can be replayed with
``qam_reader.py wav --file <wav>`` for offline analysis or constellation
re-generation without re-running the full test.

Additional outputs written to run dir:
  <CLIENT_ID>_input.wav      - raw mic recording (NEW)
  qam_constellation.png      - ECC headroom bar chart per frame
  qam_frame_stats.json       - per-frame detail
  result_<client>.json       - summary result

Params (from server UI JSON field):
{
    "timeout":      60,
    "mic_gain_db":  0,
    "zc_threshold": 0.35
}
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from analysis import make_qam_constellation_png
from testlib import (
    CLIENT_ID, INPUT_DEV, RUN_DIR, PARAMS, RUN_ID, SERVER_URL,
    log, to_int_dev, coord_signal, coord_check,
)

TESTS_DIR = Path(__file__).parent

TIMEOUT     = float(PARAMS.get("timeout",      60))
MIC_GAIN_DB = float(PARAMS.get("mic_gain_db",   0))
ZC_THRESH   = float(PARAMS.get("zc_threshold", 0.45))
TAIL_SECS   = float(PARAMS.get("tail_secs",     5))


def coord_wait_manifest(timeout=40, poll=0.3):
    """Wait for qam_config.json to appear (written by player before signalling)."""
    config_path = RUN_DIR / "qam_config.json"
    deadline    = time.time() + timeout
    while time.time() < deadline:
        if config_path.exists():
            return config_path
        time.sleep(poll)
    return None


def main():
    in_dev = to_int_dev(INPUT_DEV)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[{CLIENT_ID}] qam_decode | input_device={in_dev}")

    log("  waiting for qam_config.json...")
    config_path = coord_wait_manifest(timeout=40)
    if config_path is None:
        log("  ERROR: qam_config.json not found within 40s")
        _save_error("no_config"); return

    config = json.loads(config_path.read_text())
    total_frames = config.get("total_frames", "?")
    log(f"  config loaded: {total_frames} frames  "
        f"carrier={config.get('carrier_freq')}Hz  "
        f"symbol_rate={config.get('symbol_rate')} baud")

    coord_signal("qam_armed")
    log("  signalled qam_armed - starting decoder...")

    stats_path  = RUN_DIR / "qam_frame_stats.json"
    wav_path    = RUN_DIR / f"{CLIENT_ID}_input.wav"
    reader_script = TESTS_DIR / "qam_reader.py"

    args = [
        sys.executable, str(reader_script),
        "live",
        "--config",    str(config_path),
        "--timeout",   str(TIMEOUT),
        "--stats-out", str(stats_path),
        "--threshold", str(ZC_THRESH),
        "--wav-out",   str(wav_path),       # raw mic capture
    ]
    if in_dev is not None:
        args += ["--device", str(in_dev)]
    if MIC_GAIN_DB != 0:
        args += ["--mic-gain-db", str(MIC_GAIN_DB)]

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    frame_stats = []

    def _read_stdout():
        for line in proc.stdout:
            line = line.rstrip()
            if line.startswith("FRAME_STAT:"):
                try:
                    stat = json.loads(line[11:])
                    frame_stats.append(stat)
                    log(f"  frame {stat['seq']+1}/{total_frames}  "
                        f"errs={stat['err_count']}/{stat['ecc_capacity']}  "
                        f"headroom={stat['ecc_headroom_pct']}%  "
                        f"snr={stat.get('snr_evm_db',0):.1f}dB  "
                        f"evm={stat.get('evm_pct',100):.1f}%  "
                        f"agc={stat.get('agc_gain_db',0):+.1f}dB  "
                        f"input={stat.get('input_rms_db',-60):.1f}dBFS  "
                        f"zc={stat.get('zc_quality',0):.2f}  "
                        f"net={stat['net_bps']}bps")
                except Exception:
                    log(f"  [reader] {line}")
            elif line.startswith("FRAME_DUP:"):
                try:
                    d = json.loads(line[10:])
                    log(f"  [dup] seq={d['seq']} decoded again (already received)")
                except Exception:
                    log(f"  [reader] {line}")
            else:
                log(f"  [reader] {line}")

    reader_thread = threading.Thread(target=_read_stdout, daemon=True)
    reader_thread.start()

    # Watch for qam_played then give tail time
    played_time = None
    while proc.poll() is None:
        if played_time is None and coord_check("qam_played"):
            played_time = time.time()
            log(f"  qam_played received - waiting {TAIL_SECS}s tail then stopping")
        if played_time and time.time() - played_time > TAIL_SECS:
            log("  tail elapsed - terminating decoder")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        time.sleep(0.5)

    reader_thread.join(timeout=5)
    proc.wait()

    # Log WAV capture outcome
    if wav_path.exists():
        size_kb = wav_path.stat().st_size // 1024
        log(f"  WAV capture: {wav_path.name}  ({size_kb} KB)")
    else:
        log("  WAV capture: not written")

    # Generate constellation/stats PNG
    png_path = RUN_DIR / "qam_constellation.png"
    has_png  = False
    if stats_path.exists():
        has_png = make_qam_constellation_png(stats_path, png_path, config)

    # Build result
    n_decoded    = len(frame_stats)
    avg_headroom = (sum(f["ecc_headroom_pct"] for f in frame_stats) / n_decoded
                    if n_decoded else 0)
    total_errs   = sum(f["err_count"] for f in frame_stats)
    avg_snr      = (sum(f.get("snr_evm_db", 0)   for f in frame_stats) / n_decoded
                    if n_decoded else 0)
    avg_evm      = (sum(f.get("evm_pct", 100)     for f in frame_stats) / n_decoded
                    if n_decoded else 100)
    avg_agc      = (sum(f.get("agc_gain_db", 0)   for f in frame_stats) / n_decoded
                    if n_decoded else 0)
    avg_input    = (sum(f.get("input_rms_db", -60) for f in frame_stats) / n_decoded
                    if n_decoded else -60)
    avg_zc       = (sum(f.get("zc_quality", 0)    for f in frame_stats) / n_decoded
                    if n_decoded else 0)

    if n_decoded == 0:
        status, reason = "fail", "no frames decoded"
    elif isinstance(total_frames, int) and n_decoded < total_frames:
        status, reason = "partial", f"{n_decoded}/{total_frames} frames"
    else:
        status, reason = "ok", None

    result = {
        "client_id":            CLIENT_ID,
        "role":                 "decode",
        "run_id":               RUN_ID,
        "status":               status,
        "failure_reason":       reason,
        "frames_decoded":       n_decoded,
        "frames_expected":      total_frames,
        "total_rs_errors":      total_errs,
        "avg_ecc_headroom_pct": round(avg_headroom, 1),
        "avg_snr_evm_db":       round(avg_snr, 1),
        "avg_evm_pct":          round(avg_evm, 1),
        "avg_agc_gain_db":      round(avg_agc, 1),
        "avg_input_rms_db":     round(avg_input, 1),
        "avg_zc_quality":       round(avg_zc, 3),
        "net_bps":              frame_stats[0]["net_bps"]   if frame_stats else 0,
        "coded_bps":            frame_stats[0]["coded_bps"] if frame_stats else 0,
        "gross_bps":            frame_stats[0]["gross_bps"] if frame_stats else 0,
        "constellation_png":    str(png_path) if has_png else None,
        "wav_capture":          str(wav_path) if wav_path.exists() else None,
        "frame_stats":          frame_stats,
    }

    log(f"\n  === QAM Decode Summary ===")
    log(f"  Frames decoded   : {n_decoded}/{total_frames}")
    log(f"  RS errors total  : {total_errs}")
    log(f"  Avg ECC headroom : {avg_headroom:.1f}%")
    log(f"  Avg SNR (EVM)    : {avg_snr:.1f}dB")
    log(f"  Avg EVM          : {avg_evm:.1f}%")
    log(f"  Avg AGC gain     : {avg_agc:+.1f}dB")
    log(f"  Avg pre-AGC level: {avg_input:.1f}dBFS")
    log(f"  Avg ZC quality   : {avg_zc:.3f}  (sync confidence, 1.0=perfect)")
    if frame_stats:
        log(f"  Net bitrate      : {frame_stats[0]['net_bps']} bps")
        log(f"  Coded bitrate    : {frame_stats[0]['coded_bps']} bps")
        log(f"  Gross bitrate    : {frame_stats[0]['gross_bps']} bps")
    if has_png:
        log(f"  Constellation    : {png_path.name}")

    out = RUN_DIR / f"result_{CLIENT_ID}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def _save_error(reason):
    result = {"client_id": CLIENT_ID, "role": "decode",
              "run_id": RUN_ID, "status": "error", "failure_reason": reason}
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
