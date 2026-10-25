# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_encoder.py  -  Standalone CSS encoder.

Frame structure (no FEC):   [seq(1)] [data(chunk_size)] [crc32(4)]
Frame structure (RS only):  [seq(1)] [data(chunk_size)] [crc32(4)] [rs_parity(2*t)]
Frame structure (RS+IL):    interleave( conv( diff( [seq...crc...rs_parity] ) ) )

Encoding pipeline (in order):
  1. RS encode
  2. Differential encode  (diff_encode=true: each bit XOR'd with previous output)
  3. Convolutional encode (conv_code="k7r12": NASA K=7 rate-1/2, adds 6 tail bits)
  4. Block interleave
  5. Chirp modulate (bit 1 Ã¢â€ â€™ up-chirp, bit 0 Ã¢â€ â€™ down-chirp)

Chirp waveform:
  Linear FM sweep, f_minÃ¢â€ â€™f_max for up-chirps, f_maxÃ¢â€ â€™f_min for down-chirps.
  Instantaneous frequency is strictly linear in time: f(t) = f0 + (f1-f0)*t/T.
  The matched filter in the decoder uses the same unwindowed linear template,
  giving maximum correlation SNR.

Differential encoding protects against phase reversals in the acoustic channel.
Convolutional coding gives ~5dB coding gain via soft-decision Viterbi at the decoder.

Block interleaver (interleave_rows > 0):
  Applied AFTER conv encoding (coded bits are interleaved, not raw bits).
  Converts a contiguous burst of B symbol errors into at most ceil(B/rows)
  errors per row, allowing RS to recover from fades far longer than it
  could otherwise handle.

  Matrix layout (rows x cols, read column-by-column):
    col 0:  bit 0, bit cols, bit 2*cols, ...   Ã¢â€ Â symbol stream
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
    "message":           "Hello world",
    "output":            "css_output.wav",
    "info_out":          "css_info.json",
    "result_out":        "css_encode_result.json",
    "fs":                44100,
    "f_min":             400,
    "f_max":             6000,
    "symbol_dur":        0.25,
    "chunk_size":        12,
    "amplitude":         0.8,
    "preamble":          [1,1,1,1,1,0,0,1,1,0,1,0,1],
    "rs_t":              0,
    "interleave_rows":   0,
    "diff_encode":       true,
    "conv_code":         "k7r12"
}

interleave_rows=0 disables interleaving (default, backward-compatible).
diff_encode=false and conv_code=null disable those layers (backward-compatible).
Recommended: interleave_rows=8, rs_t=4, diff_encode=true, conv_code="k7r12".
"""

import json
import sys
import wave
import zlib
from pathlib import Path

import numpy as np

# css_encoder runs as a standalone subprocess from tests/CSS/, so reach
# shared.py (at the repo root) by adding THAT to sys.path -- two levels up,
# not one.  parent.parent lands on tests/, which has no shared.py, so this
# import silently failed every time and _default_training_symbols was always
# None -- every run fell back to the fixed legacy default (16) regardless of
# symbol_dur, and the auto-scale ("fewer chirps at long durations") documented
# below never actually fired.  Degrade gracefully to that same legacy default
# if shared.py is still unavailable for some other reason.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from shared import default_training_symbols as _default_training_symbols
except Exception:
    _default_training_symbols = None

BARKER_13  = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
CRC_BYTES  = 4

EXAMPLE_CONFIG = {
    "message":          "Hello world",
    "output":           "css_output.wav",
    "info_out":         "css_info.json",
    "result_out":       "css_encode_result.json",
    "fs":               44100,
    "f_min":            400,
    "f_max":            6000,
    "symbol_dur":       0.25,
    "chunk_size":       12,
    "amplitude":        0.8,
    "symbol_gap_frac":  0.05,
    "preamble":         BARKER_13,
    "rs_t":             0,
    "interleave_rows":  0,
    "training_symbols": 16,
    "diff_encode":      False,
    "conv_code":        None,

}

DEFAULT_MSG = "Lorem ipsum dolor sit amet"


# ---------------------------------------------------------------------------
# Reed-Solomon helpers (optional)
# ---------------------------------------------------------------------------

def _make_rs_codec(rs_t):
    """RS codec for `rs_t` correctable bytes, or None when RS is disabled.

    RS_BACKEND=c swaps the pure-Python reedsolo for the C core in tests/c/
    (rs255_c.RSCodec is a drop-in and defaults to reedsolo), which is why this
    imports the shim rather than the package -- the same wiring OFDM and QAM
    have had.  CSS was the last modem still calling reedsolo directly, so
    RS_BACKEND=c silently did nothing here while claiming to switch "the whole
    FEC layer"; a CSS run on the C backend was grading pure Python RS.
    """
    if rs_t <= 0:
        return None
    try:
        from rs255_c import RSCodec
        return RSCodec(rs_t * 2)
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
    Block interleaver: write bits into a (rows Ãƒâ€” cols) matrix row-by-row,
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
    # received is column-major; reshape as (cols Ãƒâ€” rows), transpose = (rows Ãƒâ€” cols)
    return np.array(bits, dtype=np.int8).reshape(cols, rows).T.flatten().tolist()[:orig_len]


# ---------------------------------------------------------------------------
# Chirp generation Ã¢â‚¬â€ linear FM sweep
# ---------------------------------------------------------------------------

def generate_chirp(f0, f1, duration, fs):
    """
    Linear FM chirp from f0 to f1 over `duration` seconds.

    Instantaneous frequency:  f(t) = f0 + (f1 - f0) * t / duration   [strictly linear]
    Phase:                    phi(t) = 2*pi * (f0*t + 0.5*k*t**2),  k = (f1-f0)/duration

    Rectangular envelope -- no edge taper.  The matched filter in the decoder
    uses the same unwindowed linear template, giving maximum correlation SNR
    and the sharpest possible MF peak (best timing precision).
    """
    n   = int(fs * duration)
    t   = np.linspace(0, duration, n, endpoint=False)
    k   = (f1 - f0) / duration
    phi = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
    sig = np.sin(phi)
    return sig.astype(np.float32)


# ---------------------------------------------------------------------------
# Convolutional coding  (K=7, rate 1/2, NASA standard)
# ---------------------------------------------------------------------------

_CONV_G1 = 0b1111001   # 0o171 = 121
_CONV_G2 = 0b1011011   # 0o133 =  91
_CONV_K  = 7           # constraint length
_CONV_TAIL = _CONV_K - 1  # tail bits to flush shift register (6)


def conv_encode_k7(bits):
    """
    Rate-1/2 K=7 convolutional encoder (NASA standard generators 0o171, 0o133).

    Appends K-1=6 zero tail bits to flush the shift register back to state 0,
    allowing the Viterbi decoder to terminate at a known state.

    Input:  list/array of bits (0/1), length N
    Output: list of coded bits, length (N + 6) * 2
    """
    reg = 0  # 7-bit shift register; new bits enter at MSB (bit 6)
    out = []
    for b in list(bits) + [0] * _CONV_TAIL:
        reg = ((reg >> 1) | (int(b) << (_CONV_K - 1))) & ((1 << _CONV_K) - 1)
        c1  = bin(reg & _CONV_G1).count('1') & 1
        c2  = bin(reg & _CONV_G2).count('1') & 1
        out.extend([c1, c2])
    return out


# ---------------------------------------------------------------------------
# Differential encoding
# ---------------------------------------------------------------------------

def diff_encode_bits(bits, ref_bit=0):
    """
    Differential encoder: out[i] = in[i] XOR out[i-1], out[-1] = ref_bit.

    Protects against polarity inversions in the acoustic channel: a sustained
    environment reversal flips every absolute chirp direction, but doesn't change
    any consecutive-pair transition.  Since the preamble is absolute, the decoder
    recovers polarity from sync and initialises diff_decode with the last preamble
    bit Ã¢â‚¬â€ no extra reference symbol required.

    Input:  bits (list), ref_bit (last preamble bit, default 0)
    Output: list of same length as bits
    """
    out  = []
    prev = int(ref_bit)
    for b in bits:
        d    = int(b) ^ prev
        out.append(d)
        prev = d
    return out


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
    fs              = int(  cfg.get("fs",             44100))
    f_min           = float(cfg.get("f_min",           400))
    f_max           = float(cfg.get("f_max",          6000))
    symbol_dur      = float(cfg.get("symbol_dur",      0.25))
    amplitude       = float(cfg.get("amplitude",       0.8))
    preamble        = list( cfg.get("preamble",        BARKER_13))
    message         = cfg.get("message",               DEFAULT_MSG)
    out_path        = Path( cfg.get("output",          "css_output.wav"))
    info_path       = Path( cfg.get("info_out",        "css_info.json"))
    result_path     = Path( cfg.get("result_out",      "css_encode_result.json"))
    rs_t            = int(  cfg.get("rs_t",            0))
    interleave_rows = int(  cfg.get("interleave_rows", 0))
    diff_enc        = bool( cfg.get("diff_encode",     False))
    conv_code       = cfg.get("conv_code", None)
    if isinstance(conv_code, str) and conv_code.lower() in ("none", "null", ""):
        conv_code = None

    # Inter-symbol silence gap.
    # Eliminates the DAC click/pop at chirp boundaries by inserting a short
    # silence between every pair of chirp symbols.  This silence also gives
    # the decoder's matched-filter search window room on the "chirp arrived
    # later than expected" side: expected_peak sits gap_samps before the
    # ring buffer's end, so timing_err can only read positive up to
    # +gap_samps before the search window clips at the ring boundary.
    # Value is a FRACTION of symbol_dur (not a percentage).
    #   0.0   -> no gap, no click/pop reduction, and the decoder's settled-
    #           mode search window (post_acq_search_frac, default 0.04) is
    #           ONE-SIDED -- it can detect the chirp arriving EARLY but not
    #           LATE, which can let drift run unchecked in that direction.
    #   0.05  -> 5 % of symbol_dur (e.g. 10 ms at symbol_dur=0.2 s) -- DEFAULT.
    #           gap_samps > post_acq_search_smp (0.04*sps), so the settled
    #           window is fully symmetric: drift in EITHER direction is
    #           detectable and correctable.
    #   0.50  -> 50 % gap (equal chirp and silence) -- extreme, but valid.
    # Values > 0.5 are clamped: a gap longer than the chirp itself is almost
    # certainly a unit error (e.g. the user typed "50" meaning 50 %).
    symbol_gap_frac = float(cfg.get("symbol_gap_frac", 0.05))
    if symbol_gap_frac > 0.5:
        corrected = symbol_gap_frac / 100.0
        print(f"WARNING: symbol_gap_frac={symbol_gap_frac} looks like a percentage "
              f"(expected 0â€“0.5 fraction). Using {corrected:.4f} instead. "
              f"Pass e.g. 0.05 for 5%, not 5.", flush=True)
        symbol_gap_frac = corrected
    symbol_gap_frac = max(0.0, min(0.5, symbol_gap_frac))

    lead_silence_syms = int(cfg.get("lead_silence_syms", 4))

    # ── Wake-word / suspend-resume mode ──────────────────────────────────────
    # preamble_mode:
    #   "per_frame" (default) — full training+preamble before EVERY frame
    #                           (robust re-acquisition; expensive).
    #   "wake"                — full training+preamble ONCE (frame 0), then a
    #                           short wake_word before each subsequent frame.
    #                           For a modem that suspends TX for seconds and
    #                           resumes: the RX keeps its symbol grid alive
    #                           (freewheels) across the gap, so a 2-symbol wake
    #                           word is enough to re-mark the frame boundary —
    #                           no need to re-send training+Barker (which at long
    #                           symbols can be longer than the pause itself).
    # inter_frame_silence_s: seconds of silence emitted BEFORE each non-first
    #   frame — the simulated suspend gap the decoder must freewheel across.
    # wake_word: the short marker (list of bits) that prefixes a resumed frame.
    #   Detected at a silence→signal edge only (see the decoder), so a 2-symbol
    #   word is unambiguous despite having only 4 possible patterns.
    preamble_mode         = str(cfg.get("preamble_mode", "per_frame")).lower()
    wake_word             = list(cfg.get("wake_word", [1, 0]))
    inter_frame_silence_s = float(cfg.get("inter_frame_silence_s", 0.0))

    # Training sequence: alternating up/down chirps emitted before each preamble.
    # Gives the decoder's timing PLL real, high-SNR signal to lock on before the
    # Barker window opens.  Critical at long symbol durations (Ã¢â€°Â¥0.5 s) where the
    # 4-symbol lead silence is not long enough for timing to converge from the
    # initial noise-driven drift.
    #
    # Rule of thumb: PLL time-constant Ãâ€ž Ã¢â€°Ë† 4 symbols at alpha_acq=0.25.
    # Convergence to < 1% residual requires Ã¢â€°Ë† 5Ãâ€ž = 20 symbols regardless of
    # symbol duration.  With training_symbols=20 the timing is settled before
    # the first Barker bit arrives.
    #
    # The alternating [1,0,1,0,...] pattern cannot trigger the Barker detector,
    # so the decoder sees it as unrecognised pre-preamble chirps and uses them
    # purely for timing Ã¢â‚¬â€ no code changes on the decoder side are needed.
    #
    # Default: auto-scale the run-in (silence + training) by symbol duration —
    # ~20 symbols for fast symbols, tapering to ~8 for long ones, so a 5 s
    # symbol link settles in ~40 s instead of ~100 s.  See
    # shared.default_training_symbols.  Override with training_symbols=0 to
    # disable (backward-compatible).
    if _default_training_symbols is not None:
        _default_training = _default_training_symbols(symbol_dur, lead_silence_syms)
    else:
        _default_training = max(0, 20 - lead_silence_syms)   # legacy fallback
    training_symbols = int(cfg.get("training_symbols", _default_training))

    rs_parity_bytes = rs_t * 2

    # Overhead bits that appear in every frame regardless of chunk_size
    overhead_bits = (1 + CRC_BYTES + rs_parity_bytes) * 8

    # conv rate denominator: 2 for k7r12 (rate 1/2), 1 for none
    conv_rate = 2 if conv_code == "k7r12" else 1

    max_frame_s  = float(cfg.get("max_frame_s", 30.0))
    conv_tail    = (_CONV_TAIL if conv_code == "k7r12" else 0)
    max_cs_syms  = max_frame_s * conv_rate / symbol_dur
    max_cs       = max(1, int((max_cs_syms - conv_tail - overhead_bits) / 8))

    if "chunk_size" in cfg:
        chunk_size = int(cfg["chunk_size"])
        effective_frame_s = ((overhead_bits + chunk_size * 8 + conv_tail) *
                              conv_rate * symbol_dur)
        if effective_frame_s > max_frame_s * 1.5:
            print(f"WARNING: chunk_size={chunk_size} with conv_code={conv_code!r} "
                  f"gives {effective_frame_s:.1f}s/frame (target {max_frame_s}s). "
                  f"Consider chunk_sizeÃ¢â€°Â¤{max_cs} or disabling conv_code.", flush=True)
    else:
        chunk_size = min(12, max_cs)

    base_bits = (1 + chunk_size + CRC_BYTES + rs_parity_bytes) * 8

    if conv_code == "k7r12":
        conv_bits = (base_bits + _CONV_TAIL) * 2
    else:
        conv_bits = base_bits

    if interleave_rows > 1:
        il_cols       = -(-conv_bits // interleave_rows)
        frame_symbols = interleave_rows * il_cols
    else:
        interleave_rows = 0
        frame_symbols   = conv_bits

    frame_dur = frame_symbols * symbol_dur
    print(f"symbol_dur={symbol_dur}s  chunk_size={chunk_size}  crc=4B  "
          f"rs_t={rs_t} (parity={rs_parity_bytes}B)  "
          f"interleave_rows={interleave_rows}  "
          f"training_symbols={training_symbols}  "
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
        # Linear FM chirps -- unwindowed rectangular envelope (matched to decoder MF).
        up_chirp   = generate_chirp(f_min, f_max, symbol_dur, fs)
        down_chirp = generate_chirp(f_max, f_min, symbol_dur, fs)
        # Payload source: a raw-bytes path for the modem layer (binary packets
        # whose bytes must survive intact), falling back to the UTF-8 text path
        # for plain-text messages.  message_bytes accepts bytes/bytearray or a
        # hex string ("ff01..") so callers can pass it through JSON configs.
        _mb = cfg.get("message_bytes", None)
        if _mb is not None:
            if isinstance(_mb, str):
                raw_data = bytes.fromhex(_mb)
            else:
                raw_data = bytes(_mb)
        else:
            raw_data = message.encode("utf-8", "ignore")
        chunks     = []

        sps = int(fs * symbol_dur)

        # Inter-symbol gap silence (may be zero-length if symbol_gap_frac=0).
        gap_samps = int(sps * symbol_gap_frac)
        sym_gap   = np.zeros(gap_samps, dtype=np.float32) if gap_samps > 0 else None
        if gap_samps > 0:
            print(f"symbol_gap_frac={symbol_gap_frac:.4f}  "
                  f"gap={gap_samps} samps ({gap_samps/fs*1000:.1f} ms)", flush=True)

        def _append_sym(chirp):
            """Append one chirp symbol followed by the inter-symbol gap."""
            chunks.append(chirp)
            if sym_gap is not None:
                chunks.append(sym_gap)

        # Leading silence so the decoder's AGC can settle before training begins.
        chunks.append(np.zeros(sps * lead_silence_syms, dtype=np.float32))

        # Preamble is emitted before EVERY frame, not just once at the start.
        num_frames = 0
        for i in range(0, len(raw_data), chunk_size):
            # Training sequence: alternating up/down chirps before the Barker
            # preamble.  The decoder's timing PLL converges on these; the Barker
            # correlator ignores them (alternating pattern Ã¢â€°Â  Barker-13).
            _is_first = (i == 0)
            # Suspend gap before every non-first frame (wake mode): the silence
            # the decoder must freewheel its symbol grid across.  QUANTIZED to a
            # whole number of symbol slots (sps_eff) so the resumed frame lands
            # back ON the decoder's freewheeling grid — this models a modem that
            # keeps its symbol clock running through the mute (the realistic
            # case).  A sub-symbol gap would put the resume beyond the decoder's
            # per-symbol track_search window; that needs an explicit phase
            # re-snap on resume, a separate feature from this freewheel path.
            if not _is_first and inter_frame_silence_s > 0.0:
                _sps_eff_enc = sps + gap_samps
                _gap_slots = max(1, int(round(inter_frame_silence_s * fs
                                              / _sps_eff_enc)))
                _gap_n = _gap_slots * _sps_eff_enc
                chunks.append(np.zeros(_gap_n, dtype=np.float32))
                print(f"SUSPEND_GAP before frame {num_frames} "
                      f"req={inter_frame_silence_s:.1f}s -> "
                      f"{_gap_n / fs:.2f}s ({_gap_slots} symbol slots, "
                      f"{_gap_n} smp)", flush=True)

            if _is_first or preamble_mode != "wake":
                # Full run-in: training (alternating; PLL/grid warm-up) then the
                # Barker preamble.  The decoder acquires timing here.
                for t in range(training_symbols):
                    _append_sym(up_chirp if t % 2 == 0 else down_chirp)
                for b in preamble:
                    _append_sym(up_chirp if b == 1 else down_chirp)
            else:
                # Resumed frame in wake mode: no training, no Barker — just the
                # short wake word.  The grid is already alive (freewheeled), so
                # this only re-marks the frame boundary.
                for b in wake_word:
                    _append_sym(up_chirp if b == 1 else down_chirp)
                print(f"WAKE_WORD before frame {num_frames} bits={wake_word}",
                      flush=True)

            chunk          = raw_data[i : i + chunk_size]
            actual_chunk_len = len(chunk)
            # Pad the last (short) chunk to chunk_size with zero bytes so that
            # every frame has the same structure: seq(1) + data(chunk_size) + crc(4).
            # Without this the last frame transmits fewer chirps than frame_symbols
            # and the decoder fills the shortfall with silence bits, shifting the CRC
            # window and causing a guaranteed CRC failure on the last frame.
            if actual_chunk_len < chunk_size:
                chunk = chunk + b'\x00' * (chunk_size - actual_chunk_len)
            seq_num = (i // chunk_size) % 256
            crc     = frame_crc(seq_num, chunk)

            inner = bytes([seq_num]) + chunk + crc
            frame = rs_encode_frame(inner, rs_codec)

            bits = []
            for byte in frame:
                bits.extend(get_bits(byte))

            # Pipeline: diff Ã¢â€ â€™ conv Ã¢â€ â€™ interleave Ã¢â€ â€™ chirp
            if diff_enc:
                bits = diff_encode_bits(bits, ref_bit=preamble[-1])

            if conv_code == "k7r12":
                bits = conv_encode_k7(bits)

            if interleave_rows > 1:
                bits = interleave_bits(bits, interleave_rows)

            for bit in bits:
                _append_sym(up_chirp if bit == 1 else down_chirp)

            num_frames += 1
            print(f"encoded frame {num_frames} seq={seq_num} "
                  f"({i + actual_chunk_len}/{len(raw_data)} bytes)", flush=True)

        tail = np.zeros(sps * 10, dtype=np.float32)
        chunks.append(tail)
        audio = np.clip(np.concatenate(chunks) * amplitude, -1.0, 1.0)
        pcm = (audio * 32767).astype(np.int16)

        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(pcm.tobytes())

        duration_s    = round(len(audio) / fs, 2)
        symbols_total = (lead_silence_syms
                        + num_frames * (training_symbols + len(preamble) + frame_symbols))

        # ── Chirp schedule summary ────────────────────────────────────────────
        # Single-line emit schedule so decoder/play-side per-chirp logs can be
        # aligned: every symbol occupies sps_eff = chirp(sps) + gap audio samples,
        # and the first chirp onset sits after lead_silence_syms*sps of silence.
        sps_eff_enc = sps + gap_samps
        first_onset = sps * lead_silence_syms
        print(f"CHIRP_SCHEDULE fs={fs} sps={sps} gap={gap_samps} sps_eff={sps_eff_enc} "
              f"({sps_eff_enc/fs*1000:.1f}ms/sym) symbols_total={symbols_total} "
              f"first_chirp_onset={first_onset}smp ({first_onset/fs:.3f}s) "
              f"frame_layout=[train{training_symbols}+pre{len(preamble)}+pay{frame_symbols}] "
              f"frames={num_frames} duration={duration_s}s", flush=True)

        info = {
            "fs": fs, "f_min": f_min, "f_max": f_max,
            "symbol_dur": symbol_dur, "chunk_size": chunk_size,
            "crc_bytes": CRC_BYTES, "rs_t": rs_t,
            "rs_parity_bytes": rs_parity_bytes,
            "interleave_rows": interleave_rows,
            "training_symbols": training_symbols,
            "symbol_gap_frac":  symbol_gap_frac,
            "base_bits": base_bits,
            "frame_symbols": frame_symbols,
            "preamble": preamble,
            "preamble_per_frame": (preamble_mode != "wake"),
            "preamble_mode": preamble_mode,
            "wake_word": wake_word,
            "inter_frame_silence_s": inter_frame_silence_s,
            "diff_encode": diff_enc,
            "conv_code":   conv_code,
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
