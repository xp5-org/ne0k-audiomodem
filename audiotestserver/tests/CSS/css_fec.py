# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_fec.py

Forward-error-correction and coding primitives shared by css_decoder.py
(and usable standalone for testing/analysis).  Split out of css_decoder.py
because these are self-contained — pure functions of their explicit
arguments, no dependency on decoder live-loop state.

Contents:
  Convolutional code (rate-1/2, K=7, NASA 0o171/0o133):
    viterbi_k7(), viterbi_k7_with_reliability()
  Differential encoding:
    diff_decode_bits()
  IF-slope confidence (second-order PLL robustness layer):
    if_slope_confidence()
  Reed-Solomon wrapper (via the `reedsolo` package):
    _make_rs_codec(), rs_decode_frame()
  Block de-interleaver:
    deinterleave_bits()
  Chirp generation:
    generate_chirp()
  Byte/CRC utilities:
    get_bytes_from_bits(), check_crc()
"""

import zlib

import numpy as np

from css_correlator import SIG

CRC_BYTES = 4

# ---------------------------------------------------------------------------
# Convolutional code (rate-1/2, K=7, NASA 0o171/0o133) — soft-decision Viterbi
# ---------------------------------------------------------------------------

_CONV_G1   = 0b1111001   # 0o171
_CONV_G2   = 0b1011011   # 0o133
_CONV_K    = 7
_CONV_TAIL = _CONV_K - 1  # 6 tail bits appended by encoder

# Precompute transition table at import time (128 entries)
_TRELLIS = {}
for _s in range(64):
    for _b in (0, 1):
        _reg = (_b << (_CONV_K - 1)) | _s
        _c1  = bin(_reg & _CONV_G1).count('1') & 1
        _c2  = bin(_reg & _CONV_G2).count('1') & 1
        _ns  = ((_b << (_CONV_K - 2)) | (_s >> 1)) & 0x3F
        _TRELLIS[(_s, _b)] = (_c1, _c2, _ns)


def viterbi_k7(soft_pairs):
    """
    Soft-decision Viterbi decoder for rate-1/2 K=7 code (NASA 0o171/0o133).

    soft_pairs: list of (conf0, conf1) — one pair per received coded symbol pair.
                conf in [-1, +1]: positive = received '1' (up-chirp), negative = '0'.

    The encoder appended 6 zero tail bits so Viterbi terminates at state 0.
    Returns decoded bits, length = len(soft_pairs) - 6.
    """
    INF = float('inf')
    pm  = [-INF] * 64
    pm[0] = 0.0
    surv = []

    for conf0, conf1 in soft_pairs:
        npm   = [-INF] * 64
        nsurv = [None]  * 64
        for s in range(64):
            if pm[s] == -INF:
                continue
            for b in (0, 1):
                c1, c2, ns = _TRELLIS[(s, b)]
                bm = (conf0 if c1 else -conf0) + (conf1 if c2 else -conf1)
                m  = pm[s] + bm
                if m > npm[ns]:
                    npm[ns]  = m
                    nsurv[ns] = (s, b)
        pm   = npm
        surv.append(nsurv)

    terminal = 0 if pm[0] != -INF else max(range(64), key=lambda s: pm[s])
    bits = []
    s    = terminal
    for step in range(len(surv) - 1, -1, -1):
        entry = surv[step][s]
        if entry is None:
            bits.append(0)
            continue
        prev_s, bit = entry
        bits.append(bit)
        s = prev_s
    bits.reverse()
    return bits[:-_CONV_TAIL]


def viterbi_k7_with_reliability(soft_pairs):
    """
    Soft-decision Viterbi decoder that also returns per-bit soft energy.

    Returns: (decoded_bits, pair_energies)
      decoded_bits  — list of int (0/1), length = len(soft_pairs) - _CONV_TAIL
      pair_energies — list of float, same length.
                      Energy = (|conf0| + |conf1|) / 2.0 for the coded pair
                      that generated each decoded bit.
                      Low energy → weak received symbol → erasure candidate for RS.

    Preserving the original soft values (not path metrics) is intentional:
    path metric spread is dominated by accumulated history and needs per-trellis
    normalisation that is expensive to compute correctly.  Raw pair energy is a
    direct, calibration-free measure of symbol-level SNR and maps cleanly onto
    the RS erasure threshold in conf-space.
    """
    INF = float('inf')
    pm  = [-INF] * 64
    pm[0] = 0.0
    surv = []
    # Store the soft energy for each (state, bit) transition winner at each step.
    step_energy = []

    for conf0, conf1 in soft_pairs:
        npm      = [-INF] * 64
        nsurv    = [None]  * 64
        nenergy  = [0.0]   * 64   # winning transition energy per next-state
        pair_e   = (abs(conf0) + abs(conf1)) * 0.5
        for s in range(64):
            if pm[s] == -INF:
                continue
            for b in (0, 1):
                c1, c2, ns = _TRELLIS[(s, b)]
                bm = (conf0 if c1 else -conf0) + (conf1 if c2 else -conf1)
                m  = pm[s] + bm
                if m > npm[ns]:
                    npm[ns]   = m
                    nsurv[ns] = (s, b)
                    nenergy[ns] = pair_e   # same for all paths at this step
        pm   = npm
        surv.append(nsurv)
        step_energy.append(nenergy)

    terminal = 0 if pm[0] != -INF else max(range(64), key=lambda s: pm[s])
    bits    = []
    energies = []
    s = terminal
    for step in range(len(surv) - 1, -1, -1):
        entry = surv[step][s]
        if entry is None:
            bits.append(0)
            energies.append(0.0)
            continue
        prev_s, bit = entry
        bits.append(bit)
        energies.append(step_energy[step][s])
        s = prev_s
    bits.reverse()
    energies.reverse()

    n = len(bits) - _CONV_TAIL
    return bits[:n], energies[:n]


# ---------------------------------------------------------------------------
# Differential decoding
# ---------------------------------------------------------------------------

def diff_decode_bits(bits, ref_bit=0):
    """
    Differential decoder: recovered[i] = received[i] XOR received[i-1].
    ref_bit = last preamble bit (absolute reference, no extra symbol needed).
    """
    out  = []
    prev = int(ref_bit)
    for b in bits:
        out.append(int(b) ^ prev)
        prev = int(b)
    return out


# ---------------------------------------------------------------------------
# IF-slope PLL confidence (second-order robustness layer)
# ---------------------------------------------------------------------------

def if_slope_confidence(seg, f_min, f_max, sps, fs):
    """
    Returns (confidence, r2) based on instantaneous frequency slope.

    A linear FM up-chirp has IF linearly increasing f_min → f_max.
    A narrowband interferer (bird) has ~constant IF (zero slope).
    This discriminates partial chirps even when MF amplitude is weak.
    """
    N = len(seg)
    if N < 16:
        return 0.0, 0.0

    S     = np.fft.rfft(seg.astype(SIG))
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    S[(freqs < f_min * 0.9) | (freqs > f_max * 1.1)] = 0.0
    bp    = np.fft.irfft(S, N)

    X = np.fft.fft(bp)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1.0;  h[1:N // 2] = 2.0
    else:
        h[0] = 1.0;               h[1:(N + 1) // 2] = 2.0
    # The two transforms above run in single precision (SIG) — they are the
    # expensive part and port to an embedded float32 FFT.  The phase unwrap is
    # done in float64 on purpose: unwrap ACCUMULATES absolute phase over the
    # whole segment (thousands of radians for a full symbol), so its resolution
    # degrades with length in a way the per-sample transforms do not.  It is one
    # cheap 1-D pass, so double costs nothing here.
    _analytic = np.fft.ifft(X * h)
    inst_freq = (np.diff(np.unwrap(np.angle(_analytic).astype(np.float64)))
                 * fs / (2 * np.pi))

    trim = max(1, N // 20)
    inst_freq = inst_freq[trim:-trim]
    if len(inst_freq) < 4:
        return 0.0, 0.0

    t     = np.arange(len(inst_freq), dtype=np.float64)
    t_m   = t.mean();  f_m = inst_freq.mean()
    t_var = np.sum((t - t_m) ** 2)
    if t_var < 1e-12:
        return 0.0, 0.0
    slope  = np.sum((t - t_m) * (inst_freq - f_m)) / t_var * fs
    expect = (f_max - f_min) / (sps / fs)

    pred   = f_m + (slope / fs) * (t - t_m)
    ss_res = float(np.sum((inst_freq - pred) ** 2))
    ss_tot = float(np.sum((inst_freq - f_m)  ** 2))
    r2     = max(0.0, 1.0 - ss_res / (ss_tot + 1e-12))

    return float(np.clip(slope / (expect + 1e-12), -1.0, 1.0)), r2


def _make_rs_codec(rs_t):
    """RS codec for `rs_t` correctable bytes, or None when RS is disabled.

    RS_BACKEND=c swaps the pure-Python reedsolo for the C core in tests/c/
    (rs255_c.RSCodec is a drop-in and defaults to reedsolo), which is why this
    imports the shim rather than the package.  See css_encoder._make_rs_codec.

    ERASURES.  The C core has no erasure decoder, so under RS_BACKEND=c a call
    that actually marks erasures raises NotImplementedError from rs255_c rather
    than quietly decoding without them.  Only the legacy streaming decode()
    marks any (erasure_conf_thr, default 0.25); the production MF decoder
    decode_longsym_mf leaves era_thr at 0.0 and never reaches that branch, so
    the C backend covers the shipping path end to end.
    """
    if rs_t <= 0:
        return None
    try:
        from rs255_c import RSCodec
        return RSCodec(rs_t * 2)
    except ImportError:
        print("WARNING: reedsolo not installed -- RS FEC disabled.", flush=True)
        return None


def rs_decode_frame(frame_bytes, codec, erase_pos=None):
    if codec is None:
        return frame_bytes, 0
    # An EMPTY erasure list is passed through as an empty bytearray rather than
    # omitted: reedsolo treats the two identically, and rs255_c only rejects a
    # NON-empty one, so the no-erasure path works on either backend unchanged.
    ep = bytearray(erase_pos) if erase_pos else bytearray()
    decoded, _, errata = codec.decode(bytearray(frame_bytes), erase_pos=ep)
    return bytes(decoded), len(errata)


# ---------------------------------------------------------------------------
# Block de-interleaver
# ---------------------------------------------------------------------------

def deinterleave_bits(bits, rows, orig_len):
    """
    Inverse of the encoder's block interleaver.
    bits     : received bit list, length = rows * cols
    rows     : interleave_rows from info.json
    orig_len : base_bits from info.json (trims interleave padding zeros)
    """
    if rows <= 1:
        return list(bits[:orig_len])
    cols = len(bits) // rows
    return np.array(bits, dtype=np.int8).reshape(cols, rows).T.flatten().tolist()[:orig_len]


# ---------------------------------------------------------------------------
# Chirp generation — linear FM sweep
# ---------------------------------------------------------------------------

def generate_chirp(f0, f1, duration, fs):
    """Linear FM chirp from f0 to f1 over `duration` seconds.

    f(t) = f0 + (f1-f0)*t/duration  (strictly linear instantaneous frequency)

    Rectangular envelope -- no edge taper.  This MUST match the encoder's
    generate_chirp() exactly: this MF template is correlated against the
    actually-transmitted (untapered) chirp, giving the sharpest possible
    correlation peak and the best timing precision.
    """
    t   = np.linspace(0, duration, int(fs * duration), endpoint=False)
    k   = (f1 - f0) / duration
    phi = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
    sig = np.sin(phi).astype(np.float32)
    return sig


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
