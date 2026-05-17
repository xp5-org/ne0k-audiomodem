"""
css_decoder.py

Encoding is ABSOLUTE: up-chirp = bit 1, down-chirp = bit 0.

Design: read exactly sps samples per symbol period.
  The sounddevice stream.read(sps) reads exactly that many frames.
  No hop/ssls machinery. No fractional drift. No timing sign confusion.

  Clock drift correction: after each eval, timing_err = found_peak - expected_peak.
  Fractional accumulator drives a ±1 sample correction on the next read:
    terr < 0 (chirp left)  → read sps-1 next → ring advances less → chirp right ✓
    terr > 0 (chirp right) → read sps+1 next → ring advances more → chirp left  ✓

Matched-filter design — linear chirp sweep (default):
  The transmitted chirps are unwindowed linear FM sweeps, so the optimal MF
  template is also an unwindowed linear chirp.  The legacy Hann-windowed
  template (mf_window="hann") introduced a ~3 dB sensitivity loss because
  the template no longer matched the transmitted waveform.

  Default: mf_window="none" (unwindowed, properly matched).
  Set mf_window="hann" only if backward-compatibility with an encoder that
  transmits Hann-windowed chirps is required.

ECC erasure marking with convolutional coding:
  When conv_code="k7r12" is active, the previous code reset frame_confs to
  [1.0, ...] after Viterbi, which discarded all soft information and disabled
  RS erasure marking.  The revised decoder preserves per-bit soft energy from
  the Viterbi input pairs as a reliability signal after decoding.  Low
  soft-energy bits (weak signal / high noise during that symbol) are marked
  as erasures for RS.  RS corrects up to 2*t erasures vs t errors, so
  erasure marking roughly doubles the effective RS correction capacity.
"""

import json
import sys
import time
import zlib
from collections import deque
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
    "sync_threshold":     0.50,
    "timing_loop_gain":   0.10,
    "timing_search_frac": 0.55,
    "agc_target_peak":    0.4,
    "agc_attack":         0.8,
    "agc_release":        0.02,
    "agc_max_gain":       200.0,
    "snr_log_interval":   10,
    "mf_window":          "none",     # "none" = linear chirp (default), "hann" = legacy
    "erasure_conf_thr":   0.25,
    "timing_min_snr_db":  3.0,        # floor to reject empty buffers; median filter handles noise rejection
}


# Convolutional decoding — soft-decision Viterbi (K=7, rate 1/2)
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

    S     = np.fft.rfft(seg.astype(np.float64))
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    S[(freqs < f_min * 0.9) | (freqs > f_max * 1.1)] = 0.0
    bp    = np.fft.irfft(S, N)

    X = np.fft.fft(bp)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1.0;  h[1:N // 2] = 2.0
    else:
        h[0] = 1.0;               h[1:(N + 1) // 2] = 2.0
    inst_freq = np.diff(np.unwrap(np.angle(np.fft.ifft(X * h)))) * fs / (2 * np.pi)

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
    if rs_t <= 0:
        return None
    try:
        import reedsolo
        return reedsolo.RSCodec(rs_t * 2)
    except ImportError:
        print("WARNING: reedsolo not installed -- RS FEC disabled.", flush=True)
        return None


def rs_decode_frame(frame_bytes, codec, erase_pos=None):
    if codec is None:
        return frame_bytes, 0
    import reedsolo
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
    """Linear FM chirp — unwindowed rectangular envelope."""
    t   = np.linspace(0, duration, int(fs * duration), endpoint=False)
    k   = (f1 - f0) / duration
    phi = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
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

    mf_window controls the amplitude envelope of the MF reference chirps:
      "none"  — unwindowed linear chirp (default).  The transmitted chirps are
                also unwindowed, so this template is properly matched and gives
                the highest possible correlation SNR (~3 dB better than "hann").
      "hann"  — Hann-windowed template (legacy).  Use only if the encoder was
                configured to transmit Hann-windowed chirps.

    symbol_decision(buf, centre) searches ±search_range around centre.
    Returns (bit, confidence, found_peak, snr_db, winner_pk).
    """

    def __init__(self, up_ref, down_ref, sps, search_frac=0.25, buf_size=None,
                 mf_window="none"):
        self.sps  = sps
        buf       = buf_size if buf_size else 3 * sps
        nfft      = 1 << (buf + sps - 1).bit_length()
        self.nfft = nfft

        if mf_window == "hann":
            # Legacy: Hann-windowed template (mismatched to rectangular-envelope transmit)
            win     = np.hanning(sps).astype(np.float64)
            up_win  = (up_ref[:sps]   * win)[::-1]
            dn_win  = (down_ref[:sps] * win)[::-1]
        else:
            # Linear chirp sweep: unwindowed template — properly matched to transmit.
            # Higher peak correlation, ~3 dB SNR gain vs Hann.
            up_win  = up_ref[:sps].astype(np.float64)[::-1]
            dn_win  = down_ref[:sps].astype(np.float64)[::-1]

        self.F_up = np.fft.rfft(up_win, nfft)
        self.F_dn = np.fft.rfft(dn_win, nfft)
        self.search_range = max(16, int(sps * search_frac))
        self._mf_window   = mf_window

        # Sub-symbol MF slices for partial-chirp recovery.
        self._slices = []
        sps_f = float(sps)
        for frac, s0_frac in ((0.33, 0.00),
                               (0.33, 0.33),
                               (0.33, 0.67),
                               (0.50, 0.00),
                               (0.50, 0.50)):
            sl = max(8, int(sps_f * frac))
            s0 = min(sps - sl, int(sps_f * s0_frac))
            s1 = s0 + sl
            peak_offset = sps - s0 - sl

            if mf_window == "hann":
                wsl   = np.hanning(sl).astype(np.float64)
                up_sl_data = (up_ref[s0:s1]   * wsl)[::-1]
                dn_sl_data = (down_ref[s0:s1] * wsl)[::-1]
            else:
                # Unwindowed slices: match the transmitted waveform faithfully
                up_sl_data = up_ref[s0:s1].astype(np.float64)[::-1]
                dn_sl_data = down_ref[s0:s1].astype(np.float64)[::-1]

            up_sl = np.zeros(nfft, dtype=np.float64)
            dn_sl = np.zeros(nfft, dtype=np.float64)
            up_sl[:sl] = up_sl_data
            dn_sl[:sl] = dn_sl_data
            self._slices.append((frac, s0, sl, peak_offset,
                                  np.fft.rfft(up_sl, nfft),
                                  np.fft.rfft(dn_sl, nfft)))

    def correlate(self, buf):
        B      = np.fft.rfft(buf.astype(np.float64), self.nfft)
        up_env = np.abs(np.fft.irfft(B * self.F_up, self.nfft)[:len(buf)])
        dn_env = np.abs(np.fft.irfft(B * self.F_dn, self.nfft)[:len(buf)])
        return up_env, dn_env

    def symbol_decision(self, buf, centre):
        buf_f = buf.astype(np.float64)
        B_raw = np.fft.rfft(buf_f, self.nfft)
        B = B_raw

        up_env = np.abs(np.fft.irfft(B * self.F_up, self.nfft)[:len(buf)])
        dn_env = np.abs(np.fft.irfft(B * self.F_dn, self.nfft)[:len(buf)])

        sr = self.search_range
        lo = max(0,            centre - sr)
        hi = min(len(up_env), centre + sr)
        if hi <= lo:
            return 0, 0.0, centre, -60.0, 0.0

        up_pk = float(np.max(up_env[lo:hi]))
        dn_pk = float(np.max(dn_env[lo:hi]))
        total = up_pk + dn_pk
        if total < 1e-12:
            return 0, 0.0, centre, -60.0, 0.0

        full_conf  = (up_pk - dn_pk) / total
        full_peak  = int(np.argmax(
                         (up_env if up_pk >= dn_pk else dn_env)[lo:hi])) + lo
        full_wpk   = max(up_pk, dn_pk)

        best_conf  = full_conf
        best_peak  = full_peak
        best_wpk   = full_wpk
        used_frac  = 1.0

        if abs(full_conf) < 0.50 and self._slices:
            env_len = len(up_env)
            for frac, s0, sl, peak_offset, F_up_sl, F_dn_sl in self._slices:
                sl_centre = centre - peak_offset
                sl_lo = max(0,        sl_centre - sr)
                # Cap sl_hi so that raw_peak + peak_offset <= env_len - 1.
                # Without this, start-of-chirp slices (large peak_offset ≈ 0.67×sps)
                # can produce converted peaks up to env_len - 1 + search_range,
                # giving a physically impossible positive terr that drives the
                # timing PLL in the wrong direction and causes the oscillation
                # seen as terr alternating between +6000 and -7000.
                sl_hi = min(env_len - peak_offset, sl_centre + sr)
                if sl_hi <= sl_lo:
                    continue
                u_sl = np.abs(np.fft.irfft(B * F_up_sl, self.nfft)[:env_len])
                d_sl = np.abs(np.fft.irfft(B * F_dn_sl, self.nfft)[:env_len])
                u_pk = float(np.max(u_sl[sl_lo:sl_hi]))
                d_pk = float(np.max(d_sl[sl_lo:sl_hi]))
                tot  = u_pk + d_pk
                if tot < 1e-12:
                    continue
                sl_conf  = (u_pk - d_pk) / tot
                s0_frac  = s0 / float(sl + s0) if (sl + s0) > 0 else 0.0
                if   s0_frac >= 0.60: freq_bias = 1.35
                elif s0_frac >= 0.40: freq_bias = 1.15
                elif s0_frac >= 0.25: freq_bias = 1.00
                else:                 freq_bias = 0.75
                weighted = abs(sl_conf) * (frac ** 0.5) * freq_bias
                if weighted > abs(best_conf) * (used_frac ** 0.5):
                    w_env     = u_sl if u_pk >= d_pk else d_sl
                    raw_peak  = int(np.argmax(w_env[sl_lo:sl_hi])) + sl_lo
                    converted = raw_peak + peak_offset
                    # Sanity-check: converted peak must be a valid ring position.
                    # This should always hold after the sl_hi correction above,
                    # but guard explicitly against edge-case rounding.
                    if not (0 <= converted < env_len):
                        continue
                    best_conf = sl_conf
                    best_peak = converted
                    best_wpk  = max(u_pk, d_pk)
                    used_frac = frac

        confidence = best_conf
        bit        = 1 if confidence > 0 else 0
        found_peak = best_peak
        winner_pk  = best_wpk

        winner_env = up_env if full_conf >= 0 else dn_env
        mask = np.ones(len(winner_env), dtype=bool)
        mask[max(0, found_peak - self.sps // 4):
             min(len(winner_env), found_peak + self.sps // 4)] = False
        noise_rms = float(np.sqrt(np.mean(winner_env[mask] ** 2))) if mask.any() else 1e-9
        snr_db    = float(20 * np.log10(winner_pk / (noise_rms + 1e-12) + 1e-12))

        return bit, confidence, found_peak, snr_db, winner_pk


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
    """Sliding-window preamble correlator with a persistent confidence history."""

    def __init__(self, preamble, threshold=0.65, lookback=None):
        self.n         = len(preamble)
        self.threshold = threshold
        self.expected  = np.array([1.0 if b else -1.0 for b in preamble],
                                  dtype=np.float32)
        self.lookback  = lookback if lookback is not None else self.n * 3
        self._history  = np.zeros(self.lookback, dtype=np.float32)
        self._pos      = 0
        self._filled   = 0
        self.best_lag  = 0

    def push(self, confidence):
        self._history[self._pos] = float(confidence)
        self._pos    = (self._pos + 1) % self.lookback
        self._filled = min(self._filled + 1, self.lookback)

        if self._filled < self.n:
            return False, 0.0

        max_lag = min(self.lookback - self.n, self._filled - self.n)
        best_score = -999.0
        best_lag   = 0

        for lag in range(max_lag + 1):
            end_excl = (self._pos - lag) % self.lookback
            idxs   = [(end_excl - self.n + i) % self.lookback for i in range(self.n)]
            window = self._history[idxs]
            mean_abs = float(np.mean(np.abs(window))) + 1e-9
            raw      = float(np.dot(window, self.expected) / self.n)
            score    = raw / mean_abs
            if score > best_score:
                best_score = score
                best_lag   = lag

        self.best_lag = best_lag
        return best_score >= self.threshold, best_score

    def reset(self):
        pass  # soft reset — history persists

    def clear_history(self):
        self._history[:] = 0.0
        self._pos        = 0
        self._filled     = 0
        self.best_lag    = 0


# ---------------------------------------------------------------------------
# Retrospective two-pass timing recovery
# ---------------------------------------------------------------------------

def retro_timing_decode(full_audio, up_ref, down_ref, sps, frame_symbols):
    """
    Two-pass frame decoder: compute the matched filter over the ENTIRE
    collected frame audio in one shot, then fit a timing line to all N
    peak positions simultaneously.

    The streaming PLL refines timing symbol-by-symbol with a noisy per-step
    measurement.  Its timing error is bounded by the loop noise bandwidth
    (~alpha × terr per step).  By contrast, fitting N peaks jointly gives
    precision that scales as 1/√N — roughly 10× better for a 100-symbol
    frame — because every peak constrains the same two-parameter model
    (t0, sps_effective).

    A single bad peak (echo, transient) is rejected by the IRLS outlier
    filter rather than corrupting the integrator as it would in the PLL.

    Returns (bits, confs) with refined timing, or None if insufficient data.

    Algorithm:
      1. Full-buffer FFT → MF envelope (one rfft + two irfft calls).
      2. Fold envelope at period sps to find coarse t0 (O(N) sum).
      3. Refine each of the N peaks within ±sps/6 of the coarse prediction.
      4. IRLS robust line fit: peak_pos[k] = t0_fit + k × sps_fit.
         Outliers (|residual| > sps/6) are down-weighted each iteration.
      5. Re-decide all bits at the fitted peak positions using the global
         MF envelopes (not re-running the MF — it's already computed).
    """
    N = len(full_audio)
    min_expected = sps * (frame_symbols // 2)
    if N < min_expected:
        print(f"RETRO skip: audio too short ({N} < {min_expected})", flush=True)
        return None

    audio = full_audio.astype(np.float64)

    # Step 1 — full-buffer MF (one FFT covers the entire frame)
    nfft = 1 << (N + sps - 1).bit_length()
    F_buf = np.fft.rfft(audio, nfft)

    up_tpl = np.zeros(nfft, dtype=np.float64)
    dn_tpl = np.zeros(nfft, dtype=np.float64)
    up_tpl[:sps] = up_ref[:sps].astype(np.float64)[::-1]
    dn_tpl[:sps] = down_ref[:sps].astype(np.float64)[::-1]

    up_env = np.abs(np.fft.irfft(F_buf * np.fft.rfft(up_tpl, nfft), nfft)[:N])
    dn_env = np.abs(np.fft.irfft(F_buf * np.fft.rfft(dn_tpl, nfft), nfft)[:N])
    combined = np.maximum(up_env, dn_env)

    # Step 2 — coarse timing: fold envelope at period sps.
    # Summing combined[t0 + k*sps] over all k is maximised when t0 aligns
    # with the peaks.  Reshaping and summing gives all t0 simultaneously.
    usable = (N // sps) * sps
    if usable < sps * 2:
        return None
    folded   = combined[:usable].reshape(-1, sps).sum(axis=0)
    coarse_t0 = int(np.argmax(folded))

    # Step 3 — refine each peak within ±search_r of the coarse prediction
    search_r = sps // 6
    raw_peaks  = []
    bits_out   = []
    confs_out  = []

    for k in range(frame_symbols):
        exp = coarse_t0 + k * sps
        lo  = max(0, exp - search_r)
        hi  = min(N, exp + search_r)
        if hi <= lo:
            raw_peaks.append(exp)
            bits_out.append(0)
            confs_out.append(0.0)
            continue

        u_local = int(np.argmax(up_env[lo:hi]))
        d_local = int(np.argmax(dn_env[lo:hi]))
        u_val   = up_env[lo + u_local]
        d_val   = dn_env[lo + d_local]

        # Timing anchored to whichever template wins
        peak_pos = (lo + u_local) if u_val >= d_val else (lo + d_local)
        raw_peaks.append(peak_pos)

        total = u_val + d_val
        conf  = (u_val - d_val) / (total + 1e-12)
        bits_out.append(1 if conf > 0 else 0)
        confs_out.append(conf)

    # Step 4 — IRLS robust line fit: peak_pos[k] = t0_fit + k × sps_fit
    # Iteratively re-fit excluding peaks that deviate > search_r from the line.
    k_all   = np.arange(frame_symbols, dtype=np.float64)
    p_all   = np.array(raw_peaks,     dtype=np.float64)
    mask    = np.ones(frame_symbols,   dtype=bool)
    sps_fit = float(sps)
    t0_fit  = float(coarse_t0)

    for _iter in range(6):
        k_in = k_all[mask];  p_in = p_all[mask]
        if len(k_in) < 4:
            break
        A = np.vstack([k_in, np.ones(len(k_in))]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, p_in, rcond=None)
        sps_fit, t0_fit = float(coeffs[0]), float(coeffs[1])
        residuals = np.abs(p_all - (t0_fit + k_all * sps_fit))
        new_mask  = residuals < search_r
        if new_mask.sum() == mask.sum():
            break   # converged
        mask = new_mask

    n_outliers = int((~mask).sum())
    print(f"RETRO t0={t0_fit:.1f} sps_fit={sps_fit:.4f} "
          f"(nominal {sps}) outliers={n_outliers}/{frame_symbols}", flush=True)

    # Step 5 — re-decide bits using refined (fitted) peak positions.
    #
    # Do NOT evaluate at a single integer point.  The up_env and dn_env
    # envelopes have their individual maxima at slightly different sample
    # positions.  A single-point sample at the fitted centre can land between
    # the two maxima so that dn_env[p] > up_env[p] even for a genuine
    # up-chirp — flipping the bit decision despite correct timing.
    #
    # Instead, take max(up) and max(dn) in a narrow window centred on the
    # fitted position.  The window only needs to cover ±1-2 samples of
    # inter-template peak offset plus any rounding in the fitted position;
    # sps//24 (≈ 4% of a symbol) is ample while remaining well clear of
    # adjacent symbol peaks.
    half_w  = max(4, search_r // 4)
    fitted  = np.round(t0_fit + k_all * sps_fit).astype(int)
    for k in range(frame_symbols):
        p   = fitted[k]
        lo  = max(0, p - half_w)
        hi  = min(N, p + half_w)
        if hi <= lo:
            continue
        u     = float(np.max(up_env[lo:hi]))
        d     = float(np.max(dn_env[lo:hi]))
        total = u + d
        if total < 1e-12:
            continue
        conf         = (u - d) / (total + 1e-12)
        bits_out[k]  = 1 if conf > 0 else 0
        confs_out[k] = conf

    return bits_out, confs_out


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
    alpha_acq      = float(cfg.get("timing_acq_gain",    0.25))
    nread_revert   = float(cfg.get("nread_revert_alpha", 0.05))
    search_frac    = float(cfg.get("timing_search_frac", 0.25))
    agc_target     = float(cfg.get("agc_target_peak",    0.4))
    agc_attack     = float(cfg.get("agc_attack",         0.8))
    agc_release    = float(cfg.get("agc_release",        0.02))
    agc_max        = float(cfg.get("agc_max_gain",       200.0))
    snr_log_ivl    = float(cfg.get("snr_log_interval",   10))
    resync_after   = int(  cfg.get("resync_after",        3))
    mf_window      = str(  cfg.get("mf_window",          "none"))   # "none" or "hann"
    device         = cfg.get("device")
    if device is not None:
        try:
            device = int(device)
        except (TypeError, ValueError):
            device = None

    wav_out   = cfg.get("wav_out")
    _capture  = [] if wav_out else None

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
    info_preamble = conf.get("preamble")
    cfg_preamble  = cfg.get("preamble")

    if cfg_preamble is not None:
        full_preamble = list(cfg_preamble)
        if info_preamble and list(cfg_preamble) != list(info_preamble):
            print(f"WARNING: cfg preamble ({len(cfg_preamble)}-sym) overrides "
                  f"info preamble ({len(info_preamble)}-sym) — confirm this is intentional",
                  flush=True)
    elif info_preamble is not None:
        full_preamble = list(info_preamble)
    else:
        full_preamble = list(BARKER_13)
        print("WARNING: no preamble in css_info.json or cfg — defaulting to Barker-13",
              flush=True)

    sync_word = full_preamble
    chunk_size    = int(  cfg.get("chunk_size", conf["chunk_size"]))
    crc_bytes     = int(  conf.get("crc_bytes", CRC_BYTES))
    rs_t          = int(  conf.get("rs_t",      0))
    rs_parity     = rs_t * 2
    interleave_rows = int(conf.get("interleave_rows", 0))

    rs_codec = _make_rs_codec(rs_t)
    if rs_t > 0 and rs_codec is None:
        rs_parity = 0

    sps      = int(fs * dur)
    buf_size = sps * 3

    min_tim_conf    = float(cfg.get("min_timing_conf",    0.10))
    max_terr_smp    = int(  cfg.get("max_timing_err_smp", sps // 2))
    presync_max_terr_smp = int(cfg.get("presync_max_terr_smp", max(1, int(sps * 0.02))))
    tim_track_alpha = float(cfg.get("timing_track_alpha", 0.050))

    # Minimum SNR (dB) before the timing PLL accepts a measurement.
    # This should only reject truly empty buffers (MF output is pure noise,
    # SNR typically negative).  DO NOT raise this above ~5 dB: for an
    # unwindowed linear FM chirp the sidelobes set a ceiling on the MF
    # peak-to-noise-floor ratio of ~13–16 dB regardless of acoustic
    # conditions.  Setting the gate above that ceiling makes it permanently
    # closed in any real environment, freezing the PLL completely.
    # Silence-phase drift is handled by the training sequence + median
    # filter — not by this gate.
    timing_min_snr_db = float(cfg.get("timing_min_snr_db", 3.0))

    # After sync fires, use alpha_acq for this many symbols before dropping to
    # the slow tracking alpha.  Ensures rapid convergence of whatever timing
    # error existed at sync acquisition, so frame symbols are collected with
    # good MF alignment instead of a slowly-decaying offset.
    # At alpha_acq=0.25, tau ≈ 4 symbols → effectively converged in ~12.
    timing_acq_symbols = int(cfg.get("timing_acq_symbols", 20))

    # Median filter window for timing_err before it reaches the PLL integrator.
    # In reverberant / echo environments the MF occasionally selects an echo
    # peak instead of the direct-path peak, producing a large outlier in terr
    # (e.g., terr jumping from -2000 to -8449 in one symbol).  A 3-sample
    # median turns that outlier into a non-event: the middle value of
    # [-2059, -8449, -1530] is -2059, discarding the echo spike entirely.
    # Set to 1 to disable (equivalent to the previous behaviour).
    timing_median_n = max(1, int(cfg.get("timing_median_n", 3)))

    # Maximum per-step contribution to timing_frac, expressed as a fraction
    # of sps.  Prevents a single large terr measurement from driving n_read
    # so far from sps that the audio buffer becomes significantly shorter than
    # frame_symbols × sps.  Since the retro decode uses the buffer for bit
    # decisions, a short buffer means missing end-of-frame symbols and RS
    # erasures that exceed the correction capacity.
    #
    # With max_step_frac=0.10: n_read stays within [0.9 × sps, 1.1 × sps].
    # The timing still converges — it just takes a few more symbols for large
    # initial errors.  The retro decode handles those extra symbols correctly.
    timing_max_step_frac = float(cfg.get("timing_max_step_frac", 0.10))

    base_bits = int(conf.get("base_bits",
                             (1 + chunk_size + crc_bytes + rs_parity) * 8))
    frame_symbols     = int(conf.get("frame_symbols", base_bits))
    preamble_per_frame = bool(conf.get("preamble_per_frame", False))
    diff_enc  = bool(conf.get("diff_encode", False))
    conv_code = conf.get("conv_code", None)
    pll_weight = float(cfg.get("pll_weight", 0.30))

    # Erasure threshold in soft-confidence space.
    # For mf_window="none" (unwindowed MF), the conf scale is the same as before
    # so the default 0.25 remains appropriate.
    era_thr = float(cfg.get("erasure_conf_thr", 0.25))

    up_ref   = generate_chirp(f_min, f_max, dur, fs)
    down_ref = generate_chirp(f_max, f_min, dur, fs)

    correlator = ChirpCorrelator(up_ref, down_ref, sps,
                                 search_frac=search_frac, buf_size=buf_size,
                                 mf_window=mf_window)
    agc        = PeakAGC(target_peak=agc_target, attack=agc_attack,
                         release=agc_release, max_gain=agc_max)
    syndet     = SoftPreambleDetector(sync_word, threshold=sync_threshold)

    print(f"LISTENING fs={fs} sps={sps} dur={dur}s "
          f"f={f_min:.0f}-{f_max:.0f}Hz "
          f"chunk={chunk_size}B crc={crc_bytes}B rs_t={rs_t} "
          f"interleave_rows={interleave_rows} "
          f"mf_window={mf_window!r} "
          f"base_bits={base_bits} frame_symbols={frame_symbols} "
          f"sync_word={len(sync_word)} threshold={sync_threshold} "
          f"search_range=±{correlator.search_range}",
          flush=True)

    expected_peak = buf_size - 1
    peak_est      = expected_peak
    timing_frac   = 0.0
    timing_settled = False

    ring           = np.zeros(buf_size, dtype=np.float32)
    bit_stream     = []
    pre_sync_buf   = deque(maxlen=syndet.lookback)
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
    n_read         = sps
    _acq_syms_left = 0   # counts down post-sync fast-acquisition symbols
    _terr_buf      = deque(maxlen=timing_median_n)  # median filter for echo rejection
    _frame_audio   = []  # AGC-processed audio chunks collected during frame

    diag = {
        "sync_acquired": False, "sync_at_symbol": None, "sync_score": 0.0,
        "symbols_received": 0, "peak_rms": 0.0, "mean_rms": 0.0,
        "silence": False, "avg_snr_db": 0.0,
    }

    def do_resync(reason="", reset_timing=False):
        nonlocal sync_found, bit_stream, consec_errors, timing_settled
        nonlocal peak_est, timing_frac, n_read, _acq_syms_left
        print(f"RESYNC{': ' + reason if reason else ''}"
              f"{' (timing reset)' if reset_timing else ' (soft)'}",
              flush=True)
        sync_found     = False
        bit_stream     = []
        consec_errors  = 0
        timing_settled = not reset_timing
        _acq_syms_left = 0
        _terr_buf.clear()
        _frame_audio.clear()
        if reset_timing:
            peak_est    = expected_peak
            timing_frac = 0.0
            n_read      = sps
        syndet.clear_history()
        pre_sync_buf.clear()

    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype="float32",
                            device=device) as stream:
            signal_present = True
            while True:
                if time.time() - started_at > timeout_sec:
                    break
                if stop_event is not None and stop_event.is_set():
                    break

                raw, _ = stream.read(n_read)
                raw    = raw.flatten()
                if _capture is not None:
                    _capture.append(raw.copy())
                rms_history.append(float(np.sqrt(np.mean(raw ** 2))))
                agc.track_rms(raw)
                signal_present = float(np.sqrt(np.mean(raw.astype(np.float64) ** 2))) > silence_thr

                ring = np.roll(ring, -n_read)
                agc_audio = agc.apply(raw)
                ring[-n_read:] = agc_audio
                sym_count += 1

                # Buffer audio for retrospective two-pass decode.
                # Only accumulate while collecting frame bits (sync_found=True).
                # Cleared on every resync so stale audio never bleeds into
                # the next frame.
                if sync_found:
                    _frame_audio.append(agc_audio)

                bit, confidence, found_peak, snr_db, winner_pk = (
                    correlator.symbol_decision(ring, peak_est))

                abs_conf = abs(confidence)

                # ── Timing PLL ───────────────────────────────────────────────
                # Gate timing updates on both conf AND SNR.  During silence or
                # AGC-settling, noise correlations can reach high conf at low SNR
                # and drive spurious timing drift.  Without the SNR gate, that
                # drift accumulates during the lead-silence period and is revealed
                # as a large terr (-5000+ smp) when the real signal arrives —
                # degrading MF SNR for the first ~80 frame symbols at long sps.
                tim_snr_ok = snr_db >= timing_min_snr_db
                if signal_present and abs_conf >= min_tim_conf and tim_snr_ok:
                    raw_err = found_peak - expected_peak
                    eff_max_terr = max_terr_smp if timing_settled else presync_max_terr_smp
                    clamped_err = int(np.clip(raw_err, -eff_max_terr, eff_max_terr))
                    peak_est   = int(round(
                        (1.0 - tim_track_alpha) * peak_est +
                         tim_track_alpha         * found_peak
                    ))
                    timing_err = clamped_err
                else:
                    timing_err = found_peak - expected_peak

                snr_history.append(snr_db)
                agc.update_from_peak(winner_pk)

                # ── Acq countdown (always, independent of SNR gate) ──────────
                # Decrement every symbol so the fast-acquisition window has a
                # fixed wall-clock duration.  Previously the decrement was inside
                # the SNR gate, causing _acq_syms_left to freeze during noisy
                # symbols and leaving the aggressive alpha_acq active long after
                # timing had nominally settled.
                if sync_found and _acq_syms_left > 0:
                    _acq_syms_left -= 1

                if signal_present and abs_conf >= min_tim_conf and tim_snr_ok:
                    # Post-sync fast acquisition: use alpha_acq for the first
                    # timing_acq_symbols symbols after sync fires, then drop to
                    # the slow tracking alpha.  This corrects the initial timing
                    # error (often -4000–6000 smp at long symbol durations) within
                    # ~12 symbols (3 tau at alpha_acq=0.25) instead of ~50 symbols
                    # (5 tau at alpha=0.10), keeping MF SNR high across the frame.
                    if sync_found and _acq_syms_left > 0:
                        cur_alpha = alpha_acq
                    elif sync_found:
                        cur_alpha = alpha
                    else:
                        cur_alpha = alpha_acq

                    # Median filter on timing_err before integration.
                    # In reverberant environments the MF occasionally selects an
                    # echo peak, producing a large outlier (e.g. terr jumps from
                    # -2000 to -8449 in one step).  A 3-sample median discards the
                    # spike: median([-2059, -8449, -1530]) = -2059.  The filtered
                    # value is used for the PLL update; the raw terr is still
                    # logged so the display reflects what the MF actually measured.
                    _terr_buf.append(timing_err)
                    filtered_terr = sorted(_terr_buf)[len(_terr_buf) // 2]

                    # Clamp the per-step PLL contribution to ±timing_max_step_frac
                    # of sps.  This keeps n_read within [0.9×sps, 1.1×sps] so
                    # the audio buffer stays ≥ 90% full even when terr is large.
                    # The timing still converges — it just takes proportionally
                    # more symbols.  The retro decode handles those extra symbols
                    # correctly without relying on PLL accuracy.
                    max_step  = sps * timing_max_step_frac
                    step_corr = float(np.clip(cur_alpha * filtered_terr,
                                              -max_step, max_step))
                    timing_frac += step_corr
                    timing_frac  = float(np.clip(timing_frac, -sps // 4, sps // 4))
                    int_corr     = int(timing_frac)
                    timing_frac -= int_corr
                    n_read = max(sps // 2, sps + int_corr)
                else:
                    n_read = int(round(n_read + nread_revert * (sps - n_read)))

                if time.time() - last_status_t > snr_log_ivl:
                    avg_snr = float(np.mean(snr_history[-32:])) if snr_history else 0
                    fw = abs_conf < min_tim_conf
                    print(f"STATUS "
                          f"gain={agc.gain_db:+.1f}dB "
                          f"level={agc.level_dbfs:.1f}dBFS "
                          f"snr={avg_snr:.1f}dB "
                          f"conf={confidence:+.2f} "
                          f"terr={timing_err:+d}smp "
                          f"n_read={n_read} "
                          f"fw={'yes' if fw else 'no'} "
                          f"sync={'YES' if sync_found else 'no'}"
                          f"{f' acq={_acq_syms_left}' if _acq_syms_left > 0 else ''}",
                          flush=True)
                    last_status_t = time.time()

                if not sync_found:
                    tim_flush_thr = int(cfg.get("syndet_flush_terr", sps // 2))
                    if abs(timing_err) <= tim_flush_thr:
                        timing_settled = True
                    pre_sync_buf.append((1 if confidence > 0 else 0, confidence))
                    acquired, score = syndet.push(confidence)
                    if acquired:
                        lag = syndet.best_lag
                        print(f"SYNC_ACQUIRED sym={sym_count} score={score:.3f} "
                              f"snr={snr_db:.1f}dB terr={timing_err:+d}"
                              f"{f' lag={lag}' if lag else ''}", flush=True)
                        sync_found            = True
                        _acq_syms_left        = timing_acq_symbols
                        if lag > 0:
                            avail      = len(pre_sync_buf)
                            take       = min(lag, avail)
                            bit_stream = list(pre_sync_buf)[-take:] if take else []
                            if take < lag:
                                print(f"  [sync] lag={lag} but only {take} "
                                      f"pre-sync symbols available", flush=True)
                        else:
                            bit_stream = []
                        diag["sync_acquired"] = True
                        diag["sync_at_symbol"]= sym_count - lag
                        diag["sync_score"]    = round(score, 3)
                        diag["sync_lag"]      = lag
                    elif score > 0.45:
                        # Only log candidates well above the spurious-data-correlation
                        # floor (~0.40 for high-confidence payload hitting a Barker
                        # sidelobe).  Printing every score > 0.20 generates ~40 lines
                        # of noise before the real preamble arrives.
                        print(f"SYNC_CANDIDATE sym={sym_count} score={score:.3f} "
                              f"snr={snr_db:.1f}dB conf={confidence:+.2f} "
                              f"need={sync_threshold:.2f}", flush=True)
                else:
                    # ── PLL (IF-slope) confidence blend ─────────────────────────
                    final_conf = confidence
                    if pll_weight > 0:
                        seg_start  = max(0, peak_est - sps + 1)
                        seg_end    = min(len(ring), peak_est + 1)
                        pll_conf, pll_r2 = if_slope_confidence(
                            ring[seg_start:seg_end], f_min, f_max, sps, fs)
                        if pll_r2 > 0.10:
                            w = pll_weight * pll_r2
                            final_conf = (1 - w) * confidence + w * pll_conf

                    final_bit  = 1 if final_conf > 0 else 0
                    bit_stream.append((final_bit, final_conf))

                    if len(bit_stream) >= frame_symbols:
                        frame_items = bit_stream[:frame_symbols]
                        bit_stream  = bit_stream[frame_symbols:]

                        stream_bits  = [b for b, _ in frame_items]
                        stream_confs = [c for _, c in frame_items]

                        frame_snr = (float(np.mean(snr_history[-frame_symbols:]))
                                     if len(snr_history) >= frame_symbols else snr_db)

                        # ── Primary path: retrospective two-pass timing decode ─────
                        #
                        # The streaming PLL is a running estimate updated one symbol
                        # at a time.  Any timing error at sync, any noise spike, any
                        # echo-induced peak switch corrupts every subsequent bit
                        # decision because the error is never revisited.
                        #
                        # The retrospective pass treats the frame like a QR code: the
                        # chirp shape is known in advance, so we run the MF over the
                        # entire buffered audio at once, find all N peaks jointly, fit
                        # a two-parameter timing line (t0, sps_effective) with outlier
                        # rejection, and read each bit at the globally-fitted position.
                        # A single bad symbol (noise, echo) is one outlier among N —
                        # the IRLS fit discards it rather than propagating the error.
                        #
                        # The streaming bits are kept as a fallback for the rare case
                        # where insufficient audio was buffered (e.g. lag > 0 at sync).
                        frame_bits  = stream_bits
                        frame_confs = stream_confs
                        used_retro  = False

                        if _frame_audio:
                            full_audio = np.concatenate(_frame_audio)
                            retro = retro_timing_decode(
                                full_audio, up_ref, down_ref, sps, frame_symbols)
                            if retro is not None:
                                frame_bits, frame_confs = retro
                                used_retro = True
                            else:
                                print("RETRO insufficient audio — using streaming bits",
                                      flush=True)
                        else:
                            print("RETRO no audio buffered — using streaming bits",
                                  flush=True)

                        def _run_pipeline(fb, fc):
                            """Run deinterleave → Viterbi → diff → RS → CRC."""
                            fb = list(fb);  fc = list(fc)

                            if interleave_rows > 1:
                                n_coded = len(fb)
                                cols    = n_coded // interleave_rows
                                idxs    = (np.arange(n_coded)
                                           .reshape(cols, interleave_rows)
                                           .T.flatten().tolist())
                                fb = [fb[i] for i in idxs]
                                fc = [fc[i] for i in idxs]

                            if conv_code == "k7r12":
                                sp = [(fc[i], fc[i + 1])
                                      for i in range(0, len(fc) - 1, 2)]
                                fb, pe = viterbi_k7_with_reliability(sp)
                                me = max(pe) if pe else 1.0
                                fc = [e / (me + 1e-12) for e in pe]

                            if diff_enc:
                                fb = diff_decode_bits(fb, ref_bit=sync_word[-1])

                            raw = get_bytes_from_bits(fb[:base_bits])

                            era = []
                            if rs_codec is not None and era_thr > 0:
                                for bi in range(len(raw)):
                                    b0 = bi * 8
                                    b1 = min(b0 + 8, len(fc))
                                    if b1 > b0 and min(abs(c) for c in fc[b0:b1]) < era_thr:
                                        era.append(bi)
                                if era:
                                    print(f"ERASURES n={len(era)} bytes={era[:8]}",
                                          flush=True)

                            rs_err = 0;  rs_fail = None
                            if rs_codec is not None:
                                try:
                                    raw, rs_err = rs_decode_frame(raw, rs_codec, era or None)
                                    if rs_err:
                                        print(f"RS_CORRECTED errors={rs_err} "
                                              f"snr={frame_snr:.1f}dB", flush=True)
                                except Exception as e:
                                    rs_fail = str(e)
                                    print(f"RS_FAIL snr={frame_snr:.1f}dB err={e}",
                                          flush=True)
                                    raw = b""

                            ok, seq, dat = check_crc(raw, chunk_size)
                            return ok, seq, dat, rs_err, rs_fail

                        ok, seq_num, data, rs_errors, rs_fail_str = _run_pipeline(
                            frame_bits, frame_confs)

                        # ── Fallback: retry with streaming bits if retro failed ────
                        if not ok and used_retro:
                            print("RETRO CRC failed — retrying with streaming bits",
                                  flush=True)
                            ok2, seq2, dat2, rs2, rf2 = _run_pipeline(
                                stream_bits, stream_confs)
                            if ok2:
                                print("STREAM fallback succeeded", flush=True)
                                ok, seq_num, data, rs_errors, rs_fail_str = (
                                    ok2, seq2, dat2, rs2, rf2)

                        # ── Outcome handling ──────────────────────────────────────
                        if rs_fail_str is not None and not ok:
                            frames_errored += 1
                            consec_errors  += 1
                            if preamble_per_frame:
                                do_resync("preamble_per_frame: RS fail, re-arming",
                                          reset_timing=False)
                            elif consec_errors >= resync_after:
                                do_resync(f"{consec_errors} RS failures",
                                          reset_timing=True)
                            continue

                        if ok:
                            try:
                                msg = data.decode("utf-8", errors="strict")
                                src = "retro" if used_retro else "stream"
                                print(f"FRAME seq={seq_num} snr={frame_snr:.1f}dB "
                                      f"rs={rs_errors} src={src} text={msg}", flush=True)
                                decoded_frames.append({"seq": seq_num, "text": msg,
                                                       "snr_db": round(frame_snr, 1),
                                                       "rs_errors": rs_errors,
                                                       "source": src})
                                frames_decoded += 1
                                consec_errors  = 0
                                if preamble_per_frame:
                                    do_resync("preamble_per_frame: re-arming for next frame",
                                              reset_timing=False)
                            except UnicodeDecodeError:
                                msg = data.decode("utf-8", errors="replace")
                                print(f"FRAME_ERROR seq={seq_num} reason=utf8 "
                                      f"snr={frame_snr:.1f}dB", flush=True)
                                decoded_frames.append({"seq": seq_num, "text": msg,
                                                       "error": "utf8",
                                                       "snr_db": round(frame_snr, 1)})
                                frames_errored += 1
                                consec_errors  += 1
                                if preamble_per_frame:
                                    do_resync("preamble_per_frame: utf8 fail, re-arming",
                                              reset_timing=False)
                        else:
                            raw_hex = data[:6].hex() if data else "empty"
                            print(f"FRAME_ERROR seq={seq_num} "
                                  f"reason=crc_fail snr={frame_snr:.1f}dB "
                                  f"raw={raw_hex}", flush=True)
                            frames_errored += 1
                            consec_errors  += 1
                            if preamble_per_frame:
                                do_resync("preamble_per_frame: CRC fail, re-arming",
                                          reset_timing=False)

                        if not preamble_per_frame and consec_errors >= resync_after:
                            do_resync(f"{consec_errors} CRC failures",
                                      reset_timing=True)

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

    if wav_out and _capture:
        import wave as _wave_mod
        try:
            _audio = np.concatenate(_capture).astype(np.float32)
            _pcm   = (np.clip(_audio, -1.0, 1.0) * 32767).astype(np.int16)
            with _wave_mod.open(str(wav_out), "wb") as _wf:
                _wf.setnchannels(1)
                _wf.setsampwidth(2)
                _wf.setframerate(fs)
                _wf.writeframes(_pcm.tobytes())
            print(f"WAV capture -> {wav_out}  ({len(_audio)/fs:.1f}s @ {fs}Hz)", flush=True)
            result["wav_capture"] = str(wav_out)
        except Exception as _e:
            print(f"WAV capture failed: {_e}", flush=True)

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
