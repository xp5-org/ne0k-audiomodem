"""
css_decoder.py

Encoding is ABSOLUTE: up-chirp = bit 1, down-chirp = bit 0.
No differential decoding.

Design: read exactly sps samples per symbol period.
  The sounddevice stream.read(sps) reads exactly that many frames.
  No hop/ssls machinery. No fractional drift. No timing sign confusion.

  Clock drift correction: after each eval, timing_err = found_peak - expected_peak.
  Fractional accumulator drives a ±1 sample correction on the next read:
    terr < 0 (chirp left)  → read sps-1 next → ring advances less → chirp right ✓
    terr > 0 (chirp right) → read sps+1 next → ring advances more → chirp left  ✓

  Why the previous hop-based approach failed:
    hop = sps//4 = 2756. 4 hops = 11024 ≠ 11025 = sps. The 1-sample remainder
    causes the ring to advance by sps+2755 every 2756 symbols (a "leap" hop),
    jumping peak_est by ~2755 samples. The timing loop was called 4× per symbol
    with the same timing_err, making effective gain 4× intended and corrections
    interact with when the threshold fired unpredictably. The -287 terr jump at
    sym=43 was this leap event.
"""

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

BARKER_13 = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
CRC_BYTES  = 4

EXAMPLE_CONFIG = {
    "info":               "css_info.json",
    "result_out":         "css_decode_result.json",
    "device":             None,
    "timeout_sec":        600,
    "silence_threshold":  0.002,
    "sync_threshold":     0.65,
    "timing_loop_gain":   0.10,
    "timing_search_frac": 0.25,
    "agc_target_peak":    0.4,
    "agc_attack":         0.8,
    "agc_release":        0.02,
    "agc_max_gain":       200.0,
    "snr_log_interval":   10,
}


# ---------------------------------------------------------------------------
# Reed-Solomon (optional)
# ---------------------------------------------------------------------------

def _make_rs_codec(rs_t):
    if rs_t <= 0:
        return None
    try:
        import reedsolo
        return reedsolo.RSCodec(rs_t * 2)
    except ImportError:
        print("WARNING: reedsolo not installed -- RS FEC disabled.", flush=True)
        return None


def rs_decode_frame(frame_bytes, codec):
    if codec is None:
        return frame_bytes, 0
    import reedsolo
    decoded, _, errata = codec.decode(bytearray(frame_bytes))
    return bytes(decoded), len(errata)


# ---------------------------------------------------------------------------
# Chirp generation
# ---------------------------------------------------------------------------

def generate_chirp(f0, f1, duration, fs):
    t   = np.linspace(0, duration, int(fs * duration), endpoint=False)
    k   = (f1 - f0) / duration
    phi = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
    return np.sin(phi).astype(np.float32)


# ---------------------------------------------------------------------------
# CRC
# ---------------------------------------------------------------------------

def get_bytes_from_bits(bit_list):
    out = []
    for i in range(0, len(bit_list) - 7, 8):
        val = 0
        for b in bit_list[i : i + 8]:
            val = (val << 1) | b
        out.append(val)
    return bytes(out)


def check_crc(raw_bytes, chunk_size):
    need = 1 + chunk_size + CRC_BYTES
    if len(raw_bytes) < need:
        return False, 0, b""
    seq      = raw_bytes[0]
    data     = raw_bytes[1 : 1 + chunk_size]
    received = raw_bytes[1 + chunk_size : need]
    expected = (zlib.crc32(bytes([seq]) + data) & 0xFFFFFFFF).to_bytes(4, "big")
    return received == expected, seq, data


# ---------------------------------------------------------------------------
# Matched-filter correlator
# ---------------------------------------------------------------------------

class ChirpCorrelator:
    """
    Full-symbol matched-filter correlator.

    symbol_decision(buf, centre) searches ±search_range around centre.
    Returns (bit, confidence, found_peak, snr_db).
    """

    def __init__(self, up_ref, down_ref, sps, search_frac=0.25, buf_size=None):
        self.sps  = sps
        buf       = buf_size if buf_size else 3 * sps
        nfft      = 1 << (buf + sps - 1).bit_length()
        self.nfft = nfft
        self.F_up = np.fft.rfft(up_ref[::-1].astype(np.float64),   nfft)
        self.F_dn = np.fft.rfft(down_ref[::-1].astype(np.float64), nfft)
        self.search_range = max(16, int(sps * search_frac))

    def correlate(self, buf):
        B      = np.fft.rfft(buf.astype(np.float64), self.nfft)
        up_env = np.abs(np.fft.irfft(B * self.F_up, self.nfft)[:len(buf)])
        dn_env = np.abs(np.fft.irfft(B * self.F_dn, self.nfft)[:len(buf)])
        return up_env, dn_env

    def symbol_decision(self, buf, centre):
        up_env, dn_env = self.correlate(buf)
        sr = self.search_range
        lo = max(0,            centre - sr)
        hi = min(len(up_env), centre + sr)
        if hi <= lo:
            return 0, 0.0, centre, -60.0

        up_pk = float(np.max(up_env[lo:hi]))
        dn_pk = float(np.max(dn_env[lo:hi]))
        total = up_pk + dn_pk
        if total < 1e-12:
            return 0, 0.0, centre, -60.0

        confidence = (up_pk - dn_pk) / total
        bit        = 1 if confidence > 0 else 0
        winner_env = up_env if up_pk >= dn_pk else dn_env
        winner_pk  = max(up_pk, dn_pk)
        found_peak = int(np.argmax(winner_env[lo:hi])) + lo

        mask = np.ones(len(winner_env), dtype=bool)
        mask[max(0, found_peak - self.sps // 4):
             min(len(winner_env), found_peak + self.sps // 4)] = False
        noise_rms = float(np.sqrt(np.mean(winner_env[mask] ** 2))) if mask.any() else 1e-9
        snr_db    = float(20 * np.log10(winner_pk / (noise_rms + 1e-12) + 1e-12))

        return bit, confidence, found_peak, snr_db


# ---------------------------------------------------------------------------
# AGC
# ---------------------------------------------------------------------------

class PeakAGC:
    def __init__(self, target_peak=0.4, attack=0.8, release=0.02, max_gain=200.0):
        self.target  = target_peak
        self.attack  = attack
        self.release = release
        self.max_g   = max_gain
        self.gain    = 1.0
        self._rms    = []

    def update_from_peak(self, peak_level):
        if peak_level < 1e-9:
            return
        desired = self.target / peak_level
        alpha   = self.attack if desired < self.gain else self.release
        self.gain += alpha * (desired - self.gain)
        self.gain  = float(np.clip(self.gain, 1.0 / self.max_g, self.max_g))

    def apply(self, buf):
        return np.clip(buf * self.gain, -1.0, 1.0).astype(np.float32)

    def track_rms(self, buf):
        self._rms.append(float(np.sqrt(np.mean(buf.astype(np.float64) ** 2))))
        if len(self._rms) > 100:
            self._rms.pop(0)

    @property
    def gain_db(self):
        return float(20 * np.log10(max(self.gain, 1e-12)))

    @property
    def level_dbfs(self):
        if not self._rms:
            return -120.0
        return float(20 * np.log10(float(np.mean(self._rms[-10:])) + 1e-12))


# ---------------------------------------------------------------------------
# Soft preamble / Barker detector
# ---------------------------------------------------------------------------

class SoftPreambleDetector:
    def __init__(self, preamble, threshold=0.65):
        self.n         = len(preamble)
        self.threshold = threshold
        self.expected  = np.array([1.0 if b else -1.0 for b in preamble])
        self.scores    = []

    def push(self, confidence):
        self.scores.append(float(confidence))
        if len(self.scores) < self.n:
            return False, 0.0
        window = np.array(self.scores[-self.n:])
        score  = float(np.mean(window * self.expected))
        return score >= self.threshold, score

    def reset(self):
        self.scores = []


# ---------------------------------------------------------------------------
# Main decode loop
# ---------------------------------------------------------------------------

def decode(cfg, stop_event=None):
    import sounddevice as sd

    info_path      = Path(cfg.get("info",               "css_info.json"))
    result_path    = Path(cfg.get("result_out",          "css_decode_result.json"))
    timeout_sec    = float(cfg.get("timeout_sec",        600))
    silence_thr    = float(cfg.get("silence_threshold",  0.002))
    sync_threshold = float(cfg.get("sync_threshold",     0.65))
    alpha          = float(cfg.get("timing_loop_gain",   0.10))
    search_frac    = float(cfg.get("timing_search_frac", 0.25))
    agc_target     = float(cfg.get("agc_target_peak",    0.4))
    agc_attack     = float(cfg.get("agc_attack",         0.8))
    agc_release    = float(cfg.get("agc_release",        0.02))
    agc_max        = float(cfg.get("agc_max_gain",       200.0))
    snr_log_ivl    = float(cfg.get("snr_log_interval",   10))
    resync_after   = int(  cfg.get("resync_after",        3))
    device         = cfg.get("device")
    if device is not None:
        try:
            device = int(device)
        except (TypeError, ValueError):
            device = None

    result = {
        "status": "error", "failure_reason": None,
        "frames_decoded": 0, "frames_errored": 0,
        "decoded_text": "", "diagnostics": {}, "frames": [],
    }

    if not info_path.exists():
        print(f"ERROR: info file not found: {info_path}", flush=True)
        result["failure_reason"] = "no_info_file"
        result_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return result

    conf          = json.loads(info_path.read_text())
    fs            = int(  cfg.get("fs",         conf["fs"]))
    dur           = float(cfg.get("symbol_dur", conf["symbol_dur"]))
    f_min         = float(cfg.get("f_min",      conf["f_min"]))
    f_max         = float(cfg.get("f_max",      conf["f_max"]))
    full_preamble = list( cfg.get("preamble",   conf.get("preamble", BARKER_13)))
    # Always use the full preamble for sync detection.
    # Older info.json files may contain a truncated "sync_word" key (e.g. 8
    # symbols) left over from a previous encoder version.  An 8-symbol
    # subsequence has much worse autocorrelation than Barker-13 and can fire
    # one symbol early at modest SNR, shifting the entire frame boundary and
    # causing permanent CRC failure.  Ignore the stored key entirely.
    sync_word     = full_preamble
    chunk_size    = int(  cfg.get("chunk_size", conf["chunk_size"]))
    crc_bytes     = int(  conf.get("crc_bytes", CRC_BYTES))
    rs_t          = int(  conf.get("rs_t",      0))
    rs_parity     = rs_t * 2

    rs_codec = _make_rs_codec(rs_t)
    if rs_t > 0 and rs_codec is None:
        rs_parity = 0

    sps      = int(fs * dur)
    buf_size = sps * 3
    # Read exactly sps samples per symbol. No hop/ssls complexity.
    # Fractional timing correction adjusts n_read by ±1 sample as needed.

    frame_bytes    = 1 + chunk_size + crc_bytes + rs_parity
    bits_per_frame = frame_bytes * 8

    up_ref   = generate_chirp(f_min, f_max, dur, fs)
    down_ref = generate_chirp(f_max, f_min, dur, fs)

    correlator = ChirpCorrelator(up_ref, down_ref, sps,
                                 search_frac=search_frac, buf_size=buf_size)
    agc        = PeakAGC(target_peak=agc_target, attack=agc_attack,
                         release=agc_release, max_gain=agc_max)
    syndet     = SoftPreambleDetector(sync_word, threshold=sync_threshold)

    expected_peak = buf_size - 1
    peak_est      = expected_peak
    timing_frac   = 0.0   # sub-sample fractional accumulator

    print(f"LISTENING fs={fs} sps={sps} dur={dur}s "
          f"f={f_min:.0f}-{f_max:.0f}Hz "
          f"chunk={chunk_size}B crc={crc_bytes}B rs_t={rs_t} "
          f"bits/frame={bits_per_frame} "
          f"sync_word={len(sync_word)} threshold={sync_threshold} "
          f"search_range=±{correlator.search_range}",
          flush=True)

    ring           = np.zeros(buf_size, dtype=np.float32)
    bit_stream     = []
    sync_found     = False
    frames_decoded = 0
    frames_errored = 0
    consec_errors  = 0
    rms_history    = []
    snr_history    = []
    decoded_frames = []
    started_at     = time.time()
    last_status_t  = time.time()
    sym_count      = 0
    timing_err     = 0
    n_read         = sps  # samples to read for next symbol

    diag = {
        "sync_acquired": False, "sync_at_symbol": None, "sync_score": 0.0,
        "symbols_received": 0, "peak_rms": 0.0, "mean_rms": 0.0,
        "silence": False, "avg_snr_db": 0.0,
    }

    def do_resync(reason=""):
        nonlocal sync_found, bit_stream, consec_errors
        print(f"RESYNC{': ' + reason if reason else ''}", flush=True)
        sync_found    = False
        bit_stream    = []
        consec_errors = 0
        syndet.reset()

    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype="float32",
                            device=device) as stream:
            while True:
                if time.time() - started_at > timeout_sec:
                    break
                if stop_event is not None and stop_event.is_set():
                    break

                raw, _ = stream.read(n_read)
                raw    = raw.flatten()
                rms_history.append(float(np.sqrt(np.mean(raw ** 2))))
                agc.track_rms(raw)

                ring = np.roll(ring, -n_read)
                ring[-n_read:] = agc.apply(raw)
                sym_count += 1

                bit, confidence, found_peak, snr_db = (
                    correlator.symbol_decision(ring, peak_est))

                peak_est   = found_peak
                timing_err = found_peak - expected_peak

                snr_history.append(snr_db)

                # AGC from MF peak
                up_env, dn_env = correlator.correlate(ring)
                sr = correlator.search_range
                lo = max(0,            peak_est - sr)
                hi = min(len(up_env), peak_est + sr)
                agc.update_from_peak(float(max(np.max(up_env[lo:hi]),
                                               np.max(dn_env[lo:hi]))))

                # Timing correction for next read.
                # When terr < 0 (chirp left): read fewer → ring advances less → chirp right.
                # When terr > 0 (chirp right): read more → ring advances more → chirp left.
                timing_frac += alpha * timing_err
                timing_frac  = float(np.clip(timing_frac, -sps // 4, sps // 4))
                int_corr     = int(timing_frac)
                timing_frac -= int_corr
                n_read = max(sps // 2, sps + int_corr)

                if time.time() - last_status_t > snr_log_ivl:
                    avg_snr = float(np.mean(snr_history[-32:])) if snr_history else 0
                    print(f"STATUS "
                          f"gain={agc.gain_db:+.1f}dB "
                          f"level={agc.level_dbfs:.1f}dBFS "
                          f"snr={avg_snr:.1f}dB "
                          f"terr={timing_err:+d}smp "
                          f"n_read={n_read} "
                          f"sync={'YES' if sync_found else 'no'}",
                          flush=True)
                    last_status_t = time.time()

                if not sync_found:
                    acquired, score = syndet.push(confidence)
                    if acquired:
                        print(f"SYNC_ACQUIRED sym={sym_count} score={score:.3f} "
                              f"snr={snr_db:.1f}dB terr={timing_err:+d}", flush=True)
                        sync_found            = True
                        bit_stream            = []
                        diag["sync_acquired"] = True
                        diag["sync_at_symbol"]= sym_count
                        diag["sync_score"]    = round(score, 3)
                    elif score > 0.3:
                        print(f"SYNC_CANDIDATE sym={sym_count} score={score:.3f} "
                              f"snr={snr_db:.1f}dB", flush=True)
                else:
                    bit_stream.append(bit)

                    if len(bit_stream) >= bits_per_frame:
                        frame_bits = bit_stream[:bits_per_frame]
                        bit_stream = bit_stream[bits_per_frame:]
                        raw_bytes  = get_bytes_from_bits(frame_bits)

                        frame_snr = (float(np.mean(snr_history[-bits_per_frame:]))
                                     if len(snr_history) >= bits_per_frame else snr_db)

                        rs_errors = 0
                        if rs_codec is not None:
                            try:
                                raw_bytes, rs_errors = rs_decode_frame(raw_bytes, rs_codec)
                                if rs_errors:
                                    print(f"RS_CORRECTED errors={rs_errors} "
                                          f"snr={frame_snr:.1f}dB", flush=True)
                            except Exception as e:
                                print(f"RS_FAIL snr={frame_snr:.1f}dB err={e}", flush=True)
                                frames_errored += 1
                                consec_errors  += 1
                                if consec_errors >= resync_after:
                                    do_resync(f"{consec_errors} RS failures")
                                continue

                        ok, seq_num, data = check_crc(raw_bytes, chunk_size)
                        if ok:
                            try:
                                msg = data.decode("utf-8", errors="strict")
                                print(f"FRAME seq={seq_num} snr={frame_snr:.1f}dB "
                                      f"rs={rs_errors} text={msg}", flush=True)
                                decoded_frames.append({"seq": seq_num, "text": msg,
                                                       "snr_db": round(frame_snr, 1),
                                                       "rs_errors": rs_errors})
                                frames_decoded += 1
                                consec_errors  = 0
                            except UnicodeDecodeError:
                                msg = data.decode("utf-8", errors="replace")
                                print(f"FRAME_ERROR seq={seq_num} reason=utf8 "
                                      f"snr={frame_snr:.1f}dB", flush=True)
                                decoded_frames.append({"seq": seq_num, "text": msg,
                                                       "error": "utf8",
                                                       "snr_db": round(frame_snr, 1)})
                                frames_errored += 1
                                consec_errors  += 1
                        else:
                            raw_hex = raw_bytes[:6].hex() if raw_bytes else "empty"
                            print(f"FRAME_ERROR seq={raw_bytes[0] if raw_bytes else -1} "
                                  f"reason=crc_fail snr={frame_snr:.1f}dB "
                                  f"raw={raw_hex}", flush=True)
                            frames_errored += 1
                            consec_errors  += 1

                        if consec_errors >= resync_after:
                            do_resync(f"{consec_errors} CRC failures")

            diag["symbols_received"] = sym_count
            diag["avg_snr_db"]       = round(float(np.mean(snr_history)), 1) if snr_history else 0

    except KeyboardInterrupt:
        print("INTERRUPTED", flush=True)
        diag["symbols_received"] = sym_count
    except Exception as exc:
        import traceback
        print(f"ERROR: {exc}", flush=True)
        traceback.print_exc()
        result["failure_reason"] = f"stream_error: {exc}"
        result["diagnostics"]    = diag
        result_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return result

    if rms_history:
        diag["peak_rms"] = round(float(np.max(rms_history)),  6)
        diag["mean_rms"] = round(float(np.mean(rms_history)), 6)
        diag["silence"]  = diag["peak_rms"] < silence_thr

    if diag.get("silence"):
        status, reason = "fail", f"no_audio (peak_rms={diag['peak_rms']:.6f})"
    elif not diag["sync_acquired"]:
        status, reason = "fail", f"no_sync (peak_rms={diag.get('peak_rms', 0):.4f})"
    elif frames_decoded == 0 and frames_errored > 0:
        status, reason = "fail", f"crc_fail ({frames_errored} frames)"
    elif frames_decoded == 0:
        status, reason = "fail", "sync acquired but no frames"
    else:
        expected_n = conf.get("frames")
        if expected_n and frames_decoded < expected_n:
            status, reason = "partial", f"{frames_decoded}/{expected_n} frames"
        else:
            status, reason = "ok", None

    result.update({
        "status": status, "failure_reason": reason,
        "frames_decoded": frames_decoded, "frames_errored": frames_errored,
        "diagnostics": diag, "frames": decoded_frames,
    })
    if decoded_frames:
        result["decoded_text"] = "".join(
            f["text"] for f in decoded_frames if "error" not in f)

    print(f"DONE status={status} frames={frames_decoded} "
          f"avg_snr={diag['avg_snr_db']}dB", flush=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    if "--print-schema" in sys.argv:
        print(json.dumps(EXAMPLE_CONFIG, indent=2))
        sys.exit(0)
    config_path = sys.argv[1] if len(sys.argv) > 1 else "css_decode_config.json"
    cfg = json.loads(Path(config_path).read_text()) if Path(config_path).exists() else {}
    result = decode(cfg)
    sys.exit(0 if result["status"] in ("ok", "partial") else 1)