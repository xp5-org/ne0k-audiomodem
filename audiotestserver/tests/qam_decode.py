"""
tests/qam_decode.py  -  role: decode  (pair with qam_play.py)

Framework wrapper for qam_reader.Demodulator.
Records from the mic while the player transmits, then decodes.

Per-frame stats captured from qam_reader stdout (FRAME_STAT: lines):
  - seq number, RS error count, ECC headroom %
  - net / coded / gross bitrates

Additional outputs written to run dir:
  qam_constellation.png   - scatter plot of received constellation points
  qam_frame_stats.json    - per-frame detail
  result_<client>.json    - summary result

Params (from server UI JSON field):
{
    "timeout":      60,
    "mic_gain_db":  0,
    "zc_threshold": 0.45
}
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

CLIENT_ID  = os.environ.get("AUDIOTOOL_CLIENT_ID", "unknown")
INPUT_DEV  = os.environ.get("AUDIOTOOL_INPUT_DEVICE", "")
RUN_DIR    = Path(os.environ.get("AUDIOTOOL_RUN_DIR", "."))
SERVER_URL = os.environ.get("AUDIOTOOL_SERVER_URL", "http://127.0.0.1:8000")
PARAMS     = json.loads(os.environ.get("AUDIOTOOL_PARAMS", "{}"))
RUN_ID     = os.environ.get("AUDIOTOOL_RUN_ID", "unknown")
TESTS_DIR  = Path(__file__).parent

TIMEOUT     = float(PARAMS.get("timeout",      60))
MIC_GAIN_DB = float(PARAMS.get("mic_gain_db",   0))
ZC_THRESH   = float(PARAMS.get("zc_threshold", 0.45))
TAIL_SECS   = float(PARAMS.get("tail_secs",     5))


def log(msg):
    print(msg, flush=True)


def to_int_dev(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def coord_signal(signal):
    import urllib.request
    url  = f"{SERVER_URL}/api/coord/{RUN_ID}/{signal}"
    body = json.dumps({"client_id": CLIENT_ID}).encode()
    req  = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        log(f"  [coord] {signal} failed: {e}")


def coord_wait_manifest(timeout=40, poll=0.3):
    """Wait for qam_config.json to appear (written by player before signalling)."""
    config_path = RUN_DIR / "qam_config.json"
    deadline    = time.time() + timeout
    while time.time() < deadline:
        if config_path.exists():
            return config_path
        time.sleep(poll)
    return None


def coord_check(signal):
    import urllib.request
    try:
        url = f"{SERVER_URL}/api/coord/{RUN_ID}/{signal}"
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read()).get("clients", [])
    except Exception:
        return []


def make_constellation_png(frame_stats_path: Path, output_path: Path, config: dict):
    """
    Generate a scatter plot of received constellation points.
    Uses the frame stats to colour points by ECC headroom (green=clean, red=stressed).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        log("  matplotlib not installed - skipping constellation PNG")
        return False

    try:
        stats = json.loads(frame_stats_path.read_text())
        frames = stats.get("frames", [])
        if not frames:
            log("  no frames to plot")
            return False

        # Ideal 16-QAM constellation points
        ideal = [
            -3-3j, -3-1j, -3+1j, -3+3j,
            -1-3j, -1-1j, -1+1j, -1+3j,
             1-3j,  1-1j,  1+1j,  1+3j,
             3-3j,  3-1j,  3+1j,  3+3j,
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: ideal constellation
        ax = axes[0]
        for p in ideal:
            ax.plot(p.real, p.imag, "b.", markersize=8)
        ax.set_title("Ideal 16-QAM Constellation")
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Right: per-frame summary bars (ECC headroom)
        ax2 = axes[1]
        seqs     = [f["seq"]              for f in frames]
        headroom = [f["ecc_headroom_pct"] for f in frames]
        errs     = [f["err_count"]        for f in frames]

        colors = ["green" if h >= 50 else "orange" if h > 0 else "red"
                  for h in headroom]
        bars = ax2.bar(seqs, headroom, color=colors, edgecolor="black", linewidth=0.5)
        ax2.set_xlabel("Frame sequence number")
        ax2.set_ylabel("ECC headroom (%)")
        ax2.set_title(f"ECC Headroom per Frame\n"
                      f"net={stats.get('net_bps','?')} bps  "
                      f"coded={stats.get('coded_bps','?')} bps  "
                      f"gross={stats.get('gross_bps','?')} bps")
        ax2.set_ylim(0, 110)
        ax2.axhline(50, color="green", linestyle="--", alpha=0.4, label="50% headroom")
        ax2.legend(fontsize=8)

        # Annotate bars with error count
        for bar, err in zip(bars, errs):
            if err > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"+{err}", ha="center", va="bottom", fontsize=7, color="red")

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
        log(f"  constellation PNG -> {output_path.name}")
        return True

    except Exception as e:
        log(f"  constellation PNG failed: {e}")
        return False


def main():
    in_dev = to_int_dev(INPUT_DEV)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[{CLIENT_ID}] qam_decode | input_device={in_dev}")

    # Wait for player to write qam_config.json (player writes before signalling)
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

    # Signal armed
    coord_signal("qam_armed")
    log("  signalled qam_armed - starting decoder...")

    stats_path = RUN_DIR / "qam_frame_stats.json"
    reader_script = TESTS_DIR / "qam_reader.py"

    args = [
        sys.executable, str(reader_script),
        "live",
        "--config",    str(config_path),
        "--timeout",   str(TIMEOUT),
        "--stats-out", str(stats_path),
        "--threshold", str(ZC_THRESH),
    ]
    if in_dev is not None:
        args += ["--device", str(in_dev)]
    if MIC_GAIN_DB != 0:
        args += ["--mic-gain-db", str(MIC_GAIN_DB)]

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    # Read stdout in background; main thread watches for qam_played
    frame_stats   = []
    decoded_text  = []
    last_stat_json = None

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

    # Generate constellation/stats PNG
    png_path = RUN_DIR / "qam_constellation.png"
    has_png  = False
    if stats_path.exists():
        has_png = make_constellation_png(stats_path, png_path, config)
        if not has_png:
            # Try writing a simple text summary instead
            log("  writing text summary instead of PNG")

    # Build result
    n_decoded  = len(frame_stats)
    avg_headroom = (sum(f["ecc_headroom_pct"] for f in frame_stats) / n_decoded
                    if n_decoded else 0)
    total_errs   = sum(f["err_count"] for f in frame_stats)
    avg_snr      = (sum(f.get("snr_evm_db", 0) for f in frame_stats) / n_decoded
                    if n_decoded else 0)
    avg_evm      = (sum(f.get("evm_pct", 100) for f in frame_stats) / n_decoded
                    if n_decoded else 100)
    avg_agc      = (sum(f.get("agc_gain_db", 0) for f in frame_stats) / n_decoded
                    if n_decoded else 0)
    avg_input    = (sum(f.get("input_rms_db", -60) for f in frame_stats) / n_decoded
                    if n_decoded else -60)
    avg_zc       = (sum(f.get("zc_quality", 0) for f in frame_stats) / n_decoded
                    if n_decoded else 0)

    if n_decoded == 0:
        status, reason = "fail", "no frames decoded"
    elif isinstance(total_frames, int) and n_decoded < total_frames:
        status, reason = "partial", f"{n_decoded}/{total_frames} frames"
    else:
        status, reason = "ok", None

    result = {
        "client_id":         CLIENT_ID,
        "role":              "decode",
        "run_id":            RUN_ID,
        "status":            status,
        "failure_reason":    reason,
        "frames_decoded":    n_decoded,
        "frames_expected":   total_frames,
        "total_rs_errors":   total_errs,
        "avg_ecc_headroom_pct": round(avg_headroom, 1),
        "avg_snr_evm_db":    round(avg_snr, 1),
        "avg_evm_pct":       round(avg_evm, 1),
        "avg_agc_gain_db":   round(avg_agc, 1),
        "avg_input_rms_db":  round(avg_input, 1),
        "avg_zc_quality":    round(avg_zc, 3),
        "net_bps":           frame_stats[0]["net_bps"]   if frame_stats else 0,
        "coded_bps":         frame_stats[0]["coded_bps"] if frame_stats else 0,
        "gross_bps":         frame_stats[0]["gross_bps"] if frame_stats else 0,
        "constellation_png": str(png_path) if has_png else None,
        "frame_stats":       frame_stats,
    }

    log(f"\n  === QAM Decode Summary ===")
    log(f"  Frames decoded   : {n_decoded}/{total_frames}")
    log(f"  RS errors total  : {total_errs}")
    log(f"  Avg ECC headroom : {avg_headroom:.1f}%")
    log(f"  Avg SNR (EVM)    : {avg_snr:.1f}dB")
    log(f"  Avg EVM          : {avg_evm:.1f}%  (lower=cleaner, <10%=excellent)")
    log(f"  Avg AGC gain     : {avg_agc:+.1f}dB  (how much mic boost was applied)")
    log(f"  Avg input level  : {avg_input:.1f}dBFS  (pre-AGC mic level)")
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