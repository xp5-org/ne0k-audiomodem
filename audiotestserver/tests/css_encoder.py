"""
css_encoder.py  -  Standalone CSS encoder.

Frame structure: [seq(1)] [data(chunk_size)] [crc32(4)]
CRC32 is stdlib zlib — identical on all platforms, no external dependencies.

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
    "preamble":   [1,1,1,1,1,0,0,1,1,0,1,0,1]
}

chunk_size is auto-scaled to keep frame_duration under 30s.
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
    "symbol_dur": 0.25,
    "chunk_size": 8,
    "amplitude":  0.8,
    "preamble":   BARKER_13,
}

DEFAULT_MSG = (
    "Lorem ipsum dolor sit amet"
)


def generate_chirp(f0, f1, duration, fs):
    t   = np.linspace(0, duration, int(fs * duration), endpoint=False)
    k   = (f1 - f0) / duration
    phi = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
    return np.sin(phi)


def get_bits(byte_val):
    return [(byte_val >> i) & 1 for i in range(7, -1, -1)]


def frame_crc(seq, data):
    """4-byte CRC32 over [seq_byte + data_bytes]."""
    payload = bytes([seq]) + data
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return crc.to_bytes(4, "big")


def encode(cfg):
    fs         = int(cfg.get("fs",          44100))
    f_min      = float(cfg.get("f_min",      400))
    f_max      = float(cfg.get("f_max",     6000))
    symbol_dur = float(cfg.get("symbol_dur", 0.25))
    amplitude  = float(cfg.get("amplitude",  0.8))
    preamble   = list(cfg.get("preamble",    BARKER_13))
    message    = cfg.get("message",          DEFAULT_MSG)
    out_path   = Path(cfg.get("output",      "css_output.wav"))
    info_path  = Path(cfg.get("info_out",    "css_info.json"))
    result_path = Path(cfg.get("result_out", "css_encode_result.json"))

    # Auto-scale chunk_size: frame = (1 + chunk_size + CRC_BYTES) * 8 bits
    # keep frame_dur <= 30s
    max_cs     = max(1, int(30.0 / (symbol_dur * 8) - 1 - CRC_BYTES))
    chunk_size = int(cfg.get("chunk_size", min(8, max_cs)))

    frame_dur  = (1 + chunk_size + CRC_BYTES) * 8 * symbol_dur
    print(f"symbol_dur={symbol_dur}s  chunk_size={chunk_size}  crc=4B  "
          f"frame_duration={frame_dur:.1f}s", flush=True)

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

        for b in preamble:
            buf.extend(up_chirp if b == 1 else down_chirp)

        num_frames = 0
        for i in range(0, len(raw_data), chunk_size):
            chunk   = raw_data[i : i + chunk_size]
            seq_num = (i // chunk_size) % 256
            crc     = frame_crc(seq_num, chunk)
            frame   = bytes([seq_num]) + chunk + crc

            for byte in frame:
                for bit in get_bits(byte):
                    buf.extend(up_chirp if bit == 1 else down_chirp)

            num_frames += 1
            print(f"encoded frame {num_frames} seq={seq_num} "
                  f"({i+len(chunk)}/{len(raw_data)} bytes)", flush=True)

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
        symbols_total = len(preamble) + num_frames * (1 + chunk_size + CRC_BYTES) * 8

        info = {
            "fs": fs, "f_min": f_min, "f_max": f_max,
            "symbol_dur": symbol_dur, "chunk_size": chunk_size,
            "crc_bytes": CRC_BYTES, "preamble": preamble,
            "amplitude": amplitude, "message_len": len(raw_data),
            "frames": num_frames, "duration_s": duration_s,
            "symbols_total": symbols_total,
        }
        info_path.write_text(json.dumps(info, indent=2))

        result.update({
            "status": "ok", "frames": num_frames,
            "duration_s": duration_s, "bytes_encoded": len(raw_data),
            "symbols_total": symbols_total,
        })
        print(f"done: {num_frames} frames, {duration_s}s, {len(raw_data)} bytes",
              flush=True)

    except Exception as exc:
        result["error"] = str(exc)
        print(f"ERROR: {exc}", flush=True)

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