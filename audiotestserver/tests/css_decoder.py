"""
css_decoder.py  -  Standalone CSS decoder.

Frame structure: [seq(1)] [data(chunk_size)] [crc32(4)]
CRC32 is stdlib zlib — identical on all platforms, no external dependencies.

A frame passes if crc32(seq+data) matches the received CRC.
No error correction — detection only. Failed frames are flagged
and excluded from the decoded text.

Usage:
    python css_decoder.py                    # uses css_decode_config.json
    python css_decoder.py my_config.json
    python css_decoder.py --print-schema

Config fields (all optional):
{
    "info":               "css_info.json",
    "result_out":         "css_decode_result.json",
    "device":             null,
    "timeout_sec":        600,
    "tail_secs":          5,
    "silence_threshold":  0.002,
    "sync_threshold":     0.85,
    "agc_target_dbfs":   -3,
    "agc_release_coef":   0.05
}

Stdout events:
    LISTENING fs=44100 symbol_dur=0.25s sync_threshold=0.85
    AGC gain=24.1dB level=-3.9dBFS
    SYNC_CANDIDATE sym=9 score=0.62
    SYNC_ACQUIRED sym=10 score=1.00
    FRAME seq=0 text=Lorem ipsum...
    FRAME_ERROR seq=0 reason=crc_fail raw=004c6f7201
    DONE status=ok frames=20
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
    "info":              "css_info.json",
    "result_out":        "css_decode_result.json",
    "device":            None,
    "timeout_sec":       600,
    "tail_secs":         5,
    "silence_threshold": 0.002,
    "sync_threshold":    0.85,
    "agc_target_dbfs":  -3,
    "agc_release_coef":  0.05,
}


# ---------------------------------------------------------------------------
# DSP
# ---------------------------------------------------------------------------

def generate_chirp(f0, f1, duration, fs):
    t   = np.linspace(0, duration, int(fs * duration), endpoint=False)
    k   = (f1 - f0) / duration
    phi = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
    return np.sin(phi).astype(np.float32)


def get_bytes_from_bits(bit_list):
    out = []
    for i in range(0, len(bit_list), 8):
        b = bit_list[i : i + 8]
        if len(b) < 8:
            break
        val = 0
        for bit in b:
            val = (val << 1) | bit
        out.append(val)
    return bytes(out)


def check_crc(raw_bytes, chunk_size):
    """
    raw_bytes = [seq(1)] [data(chunk_size)] [crc(4)]
    Returns (ok, seq, data) or (False, raw_bytes[0], raw_bytes[1:chunk_size+1])
    """
    if len(raw_bytes) < 1 + chunk_size + CRC_BYTES:
        return False, 0, b""
    seq      = raw_bytes[0]
    data     = raw_bytes[1 : 1 + chunk_size]
    received = raw_bytes[1 + chunk_size : 1 + chunk_size + CRC_BYTES]
    expected = (zlib.crc32(bytes([seq]) + data) & 0xFFFFFFFF).to_bytes(4, "big")
    return received == expected, seq, data


class AGC:
    def __init__(self, target_dbfs=-3, release_coef=0.05,
                 attack_coef=0.5, max_gain=100.0):
        self.target_rms = 10 ** (target_dbfs / 20)
        self.release    = release_coef
        self.attack     = attack_coef
        self.max_gain   = max_gain
        self.gain       = 1.0

    def process(self, chunk):
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < 1e-9:
            return chunk * self.gain, self.gain, -120.0
        desired = self.target_rms / rms
        if desired < self.gain:
            self.gain += self.attack  * (desired - self.gain)
        else:
            self.gain += self.release * (desired - self.gain)
        self.gain = float(np.clip(self.gain, 1.0, self.max_gain))
        out      = np.clip(chunk * self.gain, -1.0, 1.0).astype(np.float32)
        out_rms  = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
        out_dbfs = 20 * np.log10(out_rms) if out_rms > 1e-9 else -120.0
        return out, self.gain, out_dbfs


class FFTCorrelator:
    def __init__(self, up_ref, down_ref, buf_size):
        n         = buf_size + len(up_ref) - 1
        self.nfft = 1 << (n - 1).bit_length()
        self.f_up = np.fft.rfft(up_ref[::-1],   self.nfft)
        self.f_dn = np.fft.rfft(down_ref[::-1], self.nfft)
        self.out_len = buf_size - len(up_ref) + 1

    def score(self, buf):
        B     = np.fft.rfft(buf, self.nfft)
        up_pk = float(np.max(np.abs(np.fft.irfft(B * self.f_up,  self.nfft)[:self.out_len])))
        dn_pk = float(np.max(np.abs(np.fft.irfft(B * self.f_dn, self.nfft)[:self.out_len])))
        total = up_pk + dn_pk
        norm  = (up_pk - dn_pk) / total if total > 1e-9 else 0.0
        return (1 if norm > 0 else 0), norm, 0


class SoftPreambleDetector:
    """
    Threshold=0.85 requires 11/13 Barker symbols to agree in sign.
    High enough that accidental early firing is near-impossible.
    """
    def __init__(self, preamble, threshold=0.85):
        self.preamble  = preamble
        self.n         = len(preamble)
        self.threshold = threshold
        self.expected  = np.array([1.0 if b else -1.0 for b in preamble])
        self.scores    = []

    def push(self, norm_score):
        self.scores.append(norm_score)
        if len(self.scores) < self.n:
            return False, 0.0
        window = np.array(self.scores[-self.n:])
        agrees = (window * self.expected) > 0
        frac   = float(np.mean(agrees))
        return frac >= self.threshold, frac


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode(cfg, stop_event=None):
    import sounddevice as sd

    info_path      = Path(cfg.get("info",               "css_info.json"))
    result_path    = Path(cfg.get("result_out",          "css_decode_result.json"))
    timeout_sec    = float(cfg.get("timeout_sec",        600))
    tail_secs      = float(cfg.get("tail_secs",          5))
    silence_thr    = float(cfg.get("silence_threshold",  0.002))
    sync_threshold = float(cfg.get("sync_threshold",     0.85))
    agc_target     = float(cfg.get("agc_target_dbfs",   -3))
    agc_release    = float(cfg.get("agc_release_coef",   0.05))
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

    conf       = json.loads(info_path.read_text())
    fs         = int(  cfg.get("fs",         conf["fs"]))
    dur        = float(cfg.get("symbol_dur", conf["symbol_dur"]))
    f_min      = float(cfg.get("f_min",      conf["f_min"]))
    f_max      = float(cfg.get("f_max",      conf["f_max"]))
    preamble   = list( cfg.get("preamble",   conf.get("preamble", BARKER_13)))
    chunk_size = int(  cfg.get("chunk_size", conf["chunk_size"]))
    crc_bytes  = int(  conf.get("crc_bytes", CRC_BYTES))

    sps            = int(fs * dur)
    hop            = max(sps // 2, 256)
    buf_size       = sps * 2
    bits_per_frame = (1 + chunk_size + crc_bytes) * 8

    up_ref   = generate_chirp(f_min, f_max, dur, fs)
    down_ref = generate_chirp(f_max, f_min, dur, fs)
    corr     = FFTCorrelator(up_ref, down_ref, buf_size)
    agc      = AGC(target_dbfs=agc_target, release_coef=agc_release)
    syndet   = SoftPreambleDetector(preamble, threshold=sync_threshold)

    print(f"LISTENING fs={fs} f={f_min}-{f_max}Hz symbol_dur={dur}s "
          f"chunk={chunk_size}B crc=4B bits/frame={bits_per_frame} "
          f"sync_threshold={sync_threshold}", flush=True)

    ring                   = np.zeros(buf_size, dtype=np.float32)
    samples_since_last_sym = 0
    bit_stream             = []
    sync_found             = False
    frames_decoded         = 0
    frames_errored         = 0
    consec_errors          = 0
    resync_after           = 3
    raw_rms                = []
    agc_gains              = []
    decoded_frames         = []
    started_at             = time.time()
    agc_log_timer          = time.time()
    sym_count              = 0

    diag = {
        "sync_acquired": False, "sync_at_symbol": None, "sync_score": 0.0,
        "symbols_received": 0, "peak_rms": 0.0, "mean_rms": 0.0,
        "silence": False, "agc_peak_gain_db": 0.0,
    }

    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype="float32",
                            device=device) as stream:
            while True:
                if time.time() - started_at > timeout_sec:
                    break
                if stop_event is not None and stop_event.is_set():
                    break

                raw, _ = stream.read(hop)
                raw    = raw.flatten()
                raw_rms.append(float(np.sqrt(np.mean(raw ** 2))))

                chunk, gain, out_dbfs = agc.process(raw)
                agc_gains.append(20 * np.log10(gain) if gain > 0 else 0.0)

                if time.time() - agc_log_timer > 10:
                    print(f"AGC gain={agc_gains[-1]:.1f}dB level={out_dbfs:.1f}dBFS",
                          flush=True)
                    agc_log_timer = time.time()

                ring = np.roll(ring, -hop)
                ring[-hop:] = chunk
                samples_since_last_sym += hop

                if samples_since_last_sym < sps:
                    continue
                samples_since_last_sym -= sps
                sym_count += 1

                bit, norm, _ = corr.score(ring)

                if not sync_found:
                    acquired, score = syndet.push(norm)
                    if acquired:
                        print(f"SYNC_ACQUIRED sym={sym_count} score={score:.2f}",
                              flush=True)
                        sync_found             = True
                        diag["sync_acquired"]  = True
                        diag["sync_at_symbol"] = sym_count
                        diag["sync_score"]     = round(score, 3)
                        bit_stream             = []
                    elif score > 0.4:
                        print(f"SYNC_CANDIDATE sym={sym_count} score={score:.2f}",
                              flush=True)
                else:
                    bit_stream.append(bit)
                    if len(bit_stream) >= bits_per_frame:
                        frame_bits = bit_stream[:bits_per_frame]
                        bit_stream = bit_stream[bits_per_frame:]
                        raw_bytes  = get_bytes_from_bits(frame_bits)

                        ok, seq_num, data = check_crc(raw_bytes, chunk_size)
                        if ok:
                            try:
                                msg = data.decode("utf-8", errors="strict")
                                print(f"FRAME seq={seq_num} text={msg}", flush=True)
                                decoded_frames.append({"seq": seq_num, "text": msg})
                                frames_decoded += 1
                                consec_errors  = 0
                            except UnicodeDecodeError:
                                msg_lossy = data.decode("utf-8", errors="replace")
                                print(f"FRAME_ERROR seq={seq_num} reason=utf8",
                                      flush=True)
                                decoded_frames.append({"seq": seq_num,
                                                       "text": msg_lossy,
                                                       "error": "utf8"})
                                frames_errored += 1
                                consec_errors  += 1
                        else:
                            raw_hex = raw_bytes[:6].hex() if raw_bytes else "empty"
                            print(f"FRAME_ERROR seq={raw_bytes[0] if raw_bytes else -1} "
                                  f"reason=crc_fail raw={raw_hex}", flush=True)
                            frames_errored += 1
                            consec_errors  += 1

                        if consec_errors >= resync_after:
                            print(f"RESYNC: {consec_errors} consecutive failures - "
                                  f"re-scanning for preamble...", flush=True)
                            sync_found    = False
                            bit_stream    = []
                            consec_errors = 0
                            syndet        = SoftPreambleDetector(preamble,
                                                                 threshold=sync_threshold)

            diag["symbols_received"] = sym_count
            diag["agc_peak_gain_db"] = round(max(agc_gains), 1) if agc_gains else 0.0

    except KeyboardInterrupt:
        print("INTERRUPTED", flush=True)
        diag["symbols_received"] = sym_count
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        result["failure_reason"] = f"stream_error: {exc}"
        result["diagnostics"]    = diag
        result_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return result

    if raw_rms:
        diag["peak_rms"] = round(float(np.max(raw_rms)),  6)
        diag["mean_rms"] = round(float(np.mean(raw_rms)), 6)
        diag["silence"]  = diag["peak_rms"] < silence_thr

    if diag["silence"]:
        status, reason = "fail", f"no_audio (peak_rms={diag['peak_rms']:.6f})"
    elif not diag["sync_acquired"]:
        status, reason = "fail", f"no_sync (peak_rms={diag['peak_rms']:.4f})"
    elif frames_decoded == 0 and frames_errored > 0:
        status, reason = "fail", f"crc_fail ({frames_errored} frames failed CRC)"
    elif frames_decoded == 0:
        status, reason = "fail", "partial (sync acquired but no complete frames)"
    else:
        expected = conf.get("frames")
        if expected and frames_decoded < expected:
            status, reason = "partial", f"partial ({frames_decoded}/{expected} frames)"
        else:
            status, reason = "ok", None

    result.update({
        "status":         status,
        "failure_reason": reason,
        "frames_decoded": frames_decoded,
        "frames_errored": frames_errored,
        "diagnostics":    diag,
        "frames":         decoded_frames,
    })
    if decoded_frames:
        result["decoded_text"] = "".join(f["text"] for f in decoded_frames
                                         if "error" not in f)

    print(f"DONE status={status} frames={frames_decoded}", flush=True)
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