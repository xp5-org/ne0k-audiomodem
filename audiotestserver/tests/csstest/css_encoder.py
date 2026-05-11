"""
css_encoder.py  -  Standalone CSS encoder.

Frame structure (no FEC):   [seq(1)] [data(chunk_size)] [crc32(4)]
Frame structure (RS only):  [seq(1)] [data(chunk_size)] [crc32(4)] [rs_parity(2*t)]
Frame structure (RS+IL):    interleave( [seq(1)] [data(N)] [crc32(4)] [rs_parity(2*t)] )

Encoding is ABSOLUTE: bit 1 → up-chirp, bit 0 → down-chirp.

Block interleaver (interleave_rows > 0):
  Applied AFTER RS encoding, BEFORE chirp modulation.
  Converts a contiguous burst of B symbol errors into at most ceil(B/rows)
  errors per row, allowing RS to recover from fades far longer than it
  could otherwise handle.

  Matrix layout (rows x cols, read column-by-column):
    col 0:  bit 0, bit cols, bit 2*cols, ...   ← symbol stream
    col 1:  bit 1, bit cols+1, ...
    ...

  A fade lasting B symbols destroys B consecutive columns.
  Each row loses at most ceil(B/rows) bits.
  RS(t) corrects up to t byte errors per frame -- with interleaving the
  frame can survive a fade of up to rows * t * 8 symbols.

  Example: rows=8, rs_t=4, symbol_dur=0.15s
    Protected fade length = 8 * 4 * 8 * 0.15s = 38.4 s  (theoretical max)
    Practical target: protect ~1-2 s multipath/echo fades.

Config JSON (all optional):
{
    "message":         "Hello world",
    "output":          "css_output.wav",
    "info_out":        "css_info.json",
    "result_out":      "css_encode_result.json",
    "fs":              44100,
    "f_min":           400,
    "f_max":           6000,
    "symbol_dur":      0.25,
    "chunk_size":      8,
    "amplitude":       0.8,
    "preamble":        [1,1,1,1,1,0,0,1,1,0,1,0,1],
    "rs_t":            0,
    "interleave_rows": 0
}

interleave_rows=0 disables interleaving (default, backward-compatible).
Recommended: interleave_rows=8 with rs_t=4 for multipath environments.
"""

import json
import sys
import wave
import zlib
from pathlib import Path

import numpy as np

BARKER_13  = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
CRC_BYTES  = 4

EXAMPLE_CONFIG = {
    "message":         "Hello world",
    "output":          "css_output.wav",
    "info_out":        "css_info.json",
    "result_out":      "css_encode_result.json",
    "fs":              44100,
    "f_min":           400,
    "f_max":           6000,
    "symbol_dur":      0.25,
    "chunk_size":      8,
    "amplitude":       0.8,
    "preamble":        BARKER_13,
    "rs_t":            0,
    "interleave_rows": 0,
}

DEFAULT_MSG = "Lorem ipsum dolor sit amet"


# ---------------------------------------------------------------------------
# Reed-Solomon helpers (optional)
# ---------------------------------------------------------------------------

def _make_rs_codec(rs_t):
    if rs_t <= 0:
        return None
    try:
        import reedsolo
        return reedsolo.RSCodec(rs_t * 2)
    except ImportError:
        print("WARNING: reedsolo not installed -- RS FEC disabled. "
              "Install with: pip install reedsolo", flush=True)
        return None


def rs_encode_frame(payload: bytes, codec) -> bytes:
    if codec is None:
        return payload
    return bytes(codec.encode(payload))


# ---------------------------------------------------------------------------
# Block interleaver
# ---------------------------------------------------------------------------

def interleave_bits(bits, rows):
    """
    Block interleaver: write bits into a (rows × cols) matrix row-by-row,
    then read out column-by-column.

    A burst of B consecutive errors in the output (transmitted) stream
    hits at most ceil(B / rows) bits in any single row of the original
    data, allowing RS to correct fades far longer than the RS block size.

    Pads with zeros to fill the matrix; decoder trims to original length.
    Returns list of ints (0/1), length = rows * cols >= len(bits).
    """
    if rows <= 1:
        return list(bits)
    cols   = -(-len(bits) // rows)          # ceil(len/rows)
    padded = list(bits) + [0] * (rows * cols - len(bits))
    # reshape row-by-row, then read column-by-column (= transpose + flatten)
    return np.array(padded, dtype=np.int8).reshape(rows, cols).T.flatten().tolist()


def deinterleave_bits(bits, rows, orig_len):
    """
    Inverse of interleave_bits.  bits must have length rows * cols.
    Returns first orig_len bits of the de-interleaved sequence.
    """
    if rows <= 1:
        return list(bits[:orig_len])
    cols = len(bits) // rows
    # received is column-major; reshape as (cols × rows), transpose = (rows × cols)
    return np.array(bits, dtype=np.int8).reshape(cols, rows).T.flatten().tolist()[:orig_len]


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
    payload = bytes([seq]) + data
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return crc.to_bytes(4, "big")


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def encode(cfg):
    fs             = int(  cfg.get("fs",             44100))
    f_min          = float(cfg.get("f_min",           400))
    f_max          = float(cfg.get("f_max",          6000))
    symbol_dur     = float(cfg.get("symbol_dur",      0.25))
    amplitude      = float(cfg.get("amplitude",       0.8))
    preamble       = list( cfg.get("preamble",        BARKER_13))
    message        = cfg.get("message",               DEFAULT_MSG)
    out_path       = Path( cfg.get("output",          "css_output.wav"))
    info_path      = Path( cfg.get("info_out",        "css_info.json"))
    result_path    = Path( cfg.get("result_out",      "css_encode_result.json"))
    rs_t           = int(  cfg.get("rs_t",            0))
    interleave_rows = int( cfg.get("interleave_rows", 0))

    rs_parity_bytes = rs_t * 2

    # Auto-scale chunk_size so frame_duration <= 30 s
    overhead = 1 + CRC_BYTES + rs_parity_bytes
    max_cs   = max(1, int(30.0 / (symbol_dur * 8) - overhead))
    chunk_size = int(cfg.get("chunk_size", min(8, max_cs)))

    # Base bits per frame (before interleave padding)
    base_bits = (1 + chunk_size + CRC_BYTES + rs_parity_bytes) * 8

    # Interleaved bits per frame (>= base_bits, padded to rows*cols)
    if interleave_rows > 1:
        il_cols      = -(-base_bits // interleave_rows)
        frame_symbols = interleave_rows * il_cols
    else:
        interleave_rows = 0
        frame_symbols   = base_bits

    frame_dur = frame_symbols * symbol_dur
    print(f"symbol_dur={symbol_dur}s  chunk_size={chunk_size}  crc=4B  "
          f"rs_t={rs_t} (parity={rs_parity_bytes}B)  "
          f"interleave_rows={interleave_rows}  "
          f"base_bits={base_bits}  frame_symbols={frame_symbols}  "
          f"frame_duration={frame_dur:.1f}s", flush=True)

    rs_codec = _make_rs_codec(rs_t)
    if rs_t > 0 and rs_codec is None:
        rs_parity_bytes = 0

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

        sps = int(fs * symbol_dur)

        # Leading silence so the decoder's AGC can settle before the first
        # preamble arrives.  4 symbols of silence ≈ 1 s at 0.25 s/symbol.
        lead_silence_syms = 4
        buf.extend([0.0] * (sps * lead_silence_syms))

        # Preamble is emitted before EVERY frame, not just once at the start.
        # This gives the decoder multiple sync opportunities and ensures the
        # frame boundary is always self-evident regardless of startup latency
        # or how many leading symbols the decoder missed.
        num_frames = 0
        for i in range(0, len(raw_data), chunk_size):
            # Preamble before this frame
            for b in preamble:
                buf.extend(up_chirp if b == 1 else down_chirp)

            chunk   = raw_data[i : i + chunk_size]
            seq_num = (i // chunk_size) % 256
            crc     = frame_crc(seq_num, chunk)

            # Build RS-encoded frame bytes
            inner = bytes([seq_num]) + chunk + crc
            frame = rs_encode_frame(inner, rs_codec)

            # Convert to bits
            bits = []
            for byte in frame:
                bits.extend(get_bits(byte))

            # Apply block interleaver (after RS, before modulation)
            if interleave_rows > 1:
                bits = interleave_bits(bits, interleave_rows)

            # Modulate: bit 1 → up-chirp, bit 0 → down-chirp
            for bit in bits:
                buf.extend(up_chirp if bit == 1 else down_chirp)

            num_frames += 1
            print(f"encoded frame {num_frames} seq={seq_num} "
                  f"({i + len(chunk)}/{len(raw_data)} bytes)", flush=True)

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
        symbols_total = (lead_silence_syms
                        + num_frames * (len(preamble) + frame_symbols))

        info = {
            "fs": fs, "f_min": f_min, "f_max": f_max,
            "symbol_dur": symbol_dur, "chunk_size": chunk_size,
            "crc_bytes": CRC_BYTES, "rs_t": rs_t,
            "rs_parity_bytes": rs_parity_bytes,
            "interleave_rows": interleave_rows,
            "base_bits": base_bits,
            "frame_symbols": frame_symbols,
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