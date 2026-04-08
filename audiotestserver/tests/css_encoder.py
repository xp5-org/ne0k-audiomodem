"""
css_encoder.py  -  Standalone CSS encoder.

Frame structure (no FEC):  [seq(1)] [data(chunk_size)] [crc32(4)]
Frame structure (RS FEC):  [seq(1)] [data(chunk_size)] [crc32(4)] [rs_parity(2*t)]

Encoding is ABSOLUTE: bit 1 → up-chirp, bit 0 → down-chirp.
No differential encoding is applied. The decoder must match.

CRC32 -- stdlib zlib, identical on all platforms, no external dependencies.
Reed-Solomon -- optional; requires `pip install reedsolo`.  Set rs_t > 0 to enable.
  rs_t is the number of byte errors correctable per frame (parity = 2*rs_t bytes).

Usage:
    python css_encoder.py                     # uses css_encode_config.json
    python css_encoder.py my_config.json
    python css_encoder.py --print-schema

Config JSON (all optional):
{
    "message":    "Hello world",
    "output":     "css_output.wav",
    "info_out":   "css_info.json",
    "result_out": "css_encode_result.json",
    "fs":         44100,
    "f_min":      400,
    "f_max":      6000,
    "symbol_dur": 0.25,
    "chunk_size": 8,
    "amplitude":  0.8,
    "preamble":   [1,1,1,1,1,0,0,1,1,0,1,0,1],
    "rs_t":       0
}

chunk_size is auto-scaled to keep frame_duration under 30s.
rs_t=0 disables Reed-Solomon (default, backward-compatible).
rs_t=2 corrects up to 2 byte errors per frame (adds 4 parity bytes).
rs_t=4 corrects up to 4 byte errors per frame (adds 8 parity bytes).
"""

import json
import sys
import wave
import zlib
from pathlib import Path

import numpy as np

BARKER_13  = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
CRC_BYTES  = 4   # zlib.crc32 is 32-bit = 4 bytes

EXAMPLE_CONFIG = {
    "message":    "Hello world",
    "output":     "css_output.wav",
    "info_out":   "css_info.json",
    "result_out": "css_encode_result.json",
    "fs":         44100,
    "f_min":      400,
    "f_max":      6000,
    "symbol_dur": 0.5,
    "chunk_size": 12,
    "amplitude":  0.8,
    "preamble":   BARKER_13,
    "rs_t":       0,
}

DEFAULT_MSG = "Lorem ipsum dolor sit amet"


# ---------------------------------------------------------------------------
# Reed-Solomon helpers (optional)
# ---------------------------------------------------------------------------

def _make_rs_codec(rs_t):
    """Return an RSCodec instance, or None if reedsolo not available / rs_t==0."""
    if rs_t <= 0:
        return None
    try:
        import reedsolo
        return reedsolo.RSCodec(rs_t * 2)   # nsym = 2*t parity bytes
    except ImportError:
        print("WARNING: reedsolo not installed -- RS FEC disabled. "
              "Install with: pip install reedsolo", flush=True)
        return None


def rs_encode_frame(payload: bytes, codec) -> bytes:
    """Append RS parity to payload. Returns payload + parity."""
    if codec is None:
        return payload
    encoded = codec.encode(payload)
    # reedsolo.encode returns bytearray of (payload + parity)
    return bytes(encoded)


# ---------------------------------------------------------------------------
# Chirp / bit helpers
# ---------------------------------------------------------------------------

def generate_chirp(f0, f1, duration, fs):
    t   = np.linspace(0, duration, int(fs * duration), endpoint=False)
    k   = (f1 - f0) / duration
    phi = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
    return np.sin(phi)


def get_bits(byte_val):
    """MSB-first bit list for one byte."""
    return [(byte_val >> i) & 1 for i in range(7, -1, -1)]


def frame_crc(seq, data):
    """4-byte CRC32 over [seq_byte + data_bytes]."""
    payload = bytes([seq]) + data
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return crc.to_bytes(4, "big")


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def encode(cfg):
    fs         = int(  cfg.get("fs",          44100))
    f_min      = float(cfg.get("f_min",        400))
    f_max      = float(cfg.get("f_max",       6000))
    symbol_dur = float(cfg.get("symbol_dur",   0.25))
    amplitude  = float(cfg.get("amplitude",    0.8))
    preamble   = list( cfg.get("preamble",     BARKER_13))
    message    = cfg.get("message",            DEFAULT_MSG)
    out_path   = Path( cfg.get("output",       "css_output.wav"))
    info_path  = Path( cfg.get("info_out",     "css_info.json"))
    result_path = Path(cfg.get("result_out",   "css_encode_result.json"))
    rs_t       = int(  cfg.get("rs_t",         0))

    rs_parity_bytes = rs_t * 2   # 0 when disabled

    # Auto-scale chunk_size so frame_duration <= 30 s
    # frame = (1 + chunk_size + CRC_BYTES + rs_parity_bytes) * 8 symbols
    overhead = 1 + CRC_BYTES + rs_parity_bytes
    max_cs   = max(1, int(30.0 / (symbol_dur * 8) - overhead))
    chunk_size = int(cfg.get("chunk_size", min(8, max_cs)))

    frame_symbols = (1 + chunk_size + CRC_BYTES + rs_parity_bytes) * 8
    frame_dur     = frame_symbols * symbol_dur
    print(f"symbol_dur={symbol_dur}s  chunk_size={chunk_size}  crc=4B  "
          f"rs_t={rs_t} (parity={rs_parity_bytes}B)  "
          f"frame_duration={frame_dur:.1f}s", flush=True)

    rs_codec = _make_rs_codec(rs_t)
    if rs_t > 0 and rs_codec is None:
        rs_parity_bytes = 0   # disabled due to missing lib

    result = {
        "status": "error", "error": None,
        "wav_path": str(out_path), "info_path": str(info_path),
        "frames": 0, "duration_s": 0, "bytes_encoded": 0, "symbols_total": 0,
    }

    try:
        up_chirp   = generate_chirp(f_min, f_max, symbol_dur, fs)
        down_chirp = generate_chirp(f_max, f_min, symbol_dur, fs)
        raw_data   = message.encode("utf-8", "ignore")
        buf        = []

        # --- Preamble (absolute: 1→up, 0→down) ---
        for b in preamble:
            buf.extend(up_chirp if b == 1 else down_chirp)

        # --- Data frames ---
        num_frames = 0
        for i in range(0, len(raw_data), chunk_size):
            chunk   = raw_data[i : i + chunk_size]
            seq_num = (i // chunk_size) % 256
            crc     = frame_crc(seq_num, chunk)

            # Inner payload = [seq] + [data] + [crc]
            inner = bytes([seq_num]) + chunk + crc

            # Outer payload = inner + RS parity (if enabled)
            frame = rs_encode_frame(inner, rs_codec)

            # Absolute encoding: bit 1 → up-chirp, bit 0 → down-chirp
            for byte in frame:
                for bit in get_bits(byte):
                    buf.extend(up_chirp if bit == 1 else down_chirp)

            num_frames += 1
            print(f"encoded frame {num_frames} seq={seq_num} "
                  f"({i + len(chunk)}/{len(raw_data)} bytes)", flush=True)

        # Tail silence so decoder ring buffer flushes the last frame
        sps  = int(fs * symbol_dur)
        tail = np.zeros(sps * 10, dtype=np.float32)
        audio = np.clip(
            np.concatenate([np.array(buf, dtype=np.float32), tail]) * amplitude,
            -1.0, 1.0
        )
        pcm = (audio * 32767).astype(np.int16)

        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(pcm.tobytes())

        duration_s    = round(len(audio) / fs, 2)
        symbols_total = len(preamble) + num_frames * (1 + chunk_size + CRC_BYTES + rs_parity_bytes) * 8

        info = {
            "fs": fs, "f_min": f_min, "f_max": f_max,
            "symbol_dur": symbol_dur, "chunk_size": chunk_size,
            "crc_bytes": CRC_BYTES, "rs_t": rs_t,
            "rs_parity_bytes": rs_parity_bytes,
            "preamble": preamble,
            "amplitude": amplitude, "message_len": len(raw_data),
            "frames": num_frames, "duration_s": duration_s,
            "symbols_total": symbols_total,
        }
        info_path.write_text(json.dumps(info, indent=2))

        result.update({
            "status": "ok", "frames": num_frames,
            "duration_s": duration_s,
            "bytes_encoded": len(raw_data),
            "symbols_total": symbols_total,
        })
        print(f"done: {num_frames} frames, {duration_s}s, {len(raw_data)} bytes",
              flush=True)

    except Exception as exc:
        import traceback
        result["error"] = str(exc)
        print(f"ERROR: {exc}", flush=True)
        traceback.print_exc()

    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    if "--print-schema" in sys.argv:
        print(json.dumps(EXAMPLE_CONFIG, indent=2))
        sys.exit(0)

    config_path = sys.argv[1] if len(sys.argv) > 1 else "css_encode_config.json"
    cfg = json.loads(Path(config_path).read_text()) if Path(config_path).exists() else {}
    result = encode(cfg)
    sys.exit(0 if result["status"] == "ok" else 1)
    