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

# ---------------------------------------------------------------------------
# Convolutional decoding — soft-decision Viterbi (K=7, rate 1/2)
# ---------------------------------------------------------------------------

_CONV_G1   = 0b1111001   # 0o171
_CONV_G2   = 0b1011011   # 0o133
_CONV_K    = 7
_CONV_TAIL = _CONV_K - 1  # 6 tail bits appended by encoder

# Precompute transition table at import time (128 entries)
# _TRELLIS[(state, bit)] = (c1, c2, next_state)
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

    Coding gain vs hard-decision RS alone: ~4-5 dB at BER 1e-3.
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
    This discriminates partial chirps even when MF amplitude is weak:
    any un-obscured portion of the chirp still shows the correct slope direction.

    Returns:
      confidence in [-1, +1]: +1 = up-chirp slope, -1 = down-chirp
      r2        in [0,  1]:   linearity quality (1 = perfect chirp)
    """
    N = len(seg)
    if N < 16:
        return 0.0, 0.0

    # Band-pass to chirp band (FFT zeroing — suppresses birds outside band)
    S     = np.fft.rfft(seg.astype(np.float64))
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    S[(freqs < f_min * 0.9) | (freqs > f_max * 1.1)] = 0.0
    bp    = np.fft.irfft(S, N)

    # Analytic signal via one-sided FFT (Hilbert transform)
    X = np.fft.fft(bp)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1.0;  h[1:N // 2] = 2.0
    else:
        h[0] = 1.0;               h[1:(N + 1) // 2] = 2.0
    inst_freq = np.diff(np.unwrap(np.angle(np.fft.ifft(X * h)))) * fs / (2 * np.pi)

    # Trim 5% at each edge — adjacent-symbol contamination
    trim = max(1, N // 20)
    inst_freq = inst_freq[trim:-trim]
    if len(inst_freq) < 4:
        return 0.0, 0.0

    # Linear regression: measured slope vs expected chirp slope
    t     = np.arange(len(inst_freq), dtype=np.float64)
    t_m   = t.mean();  f_m = inst_freq.mean()
    t_var = np.sum((t - t_m) ** 2)
    if t_var < 1e-12:
        return 0.0, 0.0
    slope  = np.sum((t - t_m) * (inst_freq - f_m)) / t_var * fs   # Hz/s
    expect = (f_max - f_min) / (sps / fs)                           # Hz/s for one symbol

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
# Block de-interleaver (mirrors css_encoder.py interleave_bits exactly)
# ---------------------------------------------------------------------------

def deinterleave_bits(bits, rows, orig_len):
    """
    Inverse of the encoder's block interleaver.

    Encoder wrote bits row-by-row into (rows × cols) and transmitted
    column-by-column.  We received in column-major order, so reshape as
    (cols × rows), transpose to (rows × cols), read row-by-row.

    bits     : received bit list, length = rows * cols
    rows     : interleave_rows from info.json
    orig_len : base_bits from info.json (trims interleave padding zeros)
    """
    if rows <= 1:
        return list(bits[:orig_len])
    cols = len(bits) // rows
    return np.array(bits, dtype=np.int8).reshape(cols, rows).T.flatten().tolist()[:orig_len]


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
        # Apply a Hann window to the reference chirps before computing the MF.
        # Without windowing, a linear FM chirp's autocorrelation has sidelobes
        # whose integrated RMS sits ~30dB below the mainlobe — which becomes the
        # hard ceiling for the SNR estimate regardless of actual signal quality.
        # Hann windowing pushes that floor to ~43dB; Blackman-Harris to ~67dB.
        # The window is applied to the *reference* only (not the incoming audio)
        # so it shapes the MF impulse response without affecting AGC or timing.
        win = np.hanning(sps).astype(np.float64)
        up_win  = (up_ref[:sps]  * win)[::-1]
        dn_win  = (down_ref[:sps] * win)[::-1]
        self.F_up = np.fft.rfft(up_win,  nfft)
        self.F_dn = np.fft.rfft(dn_win,  nfft)
        self.search_range = max(16, int(sps * search_frac))

        # Sub-symbol MF slices for partial-chirp recovery.
        #
        # KEY INSIGHT: a linear FM chirp maps FREQUENCY to TIME linearly.
        # f(t) = f_min + (f_max - f_min) * t / sps
        # So "birds obscuring the upper 2/3 of the frequency band" =
        # "birds obscuring the LAST 2/3 of the chirp in time."
        # The only audible segment is the FIRST 1/3 (low-frequency tail).
        #
        # Previous version always took centre slices, which sit in the bird zone.
        # We now pre-compute slices at LEADING, CENTRE, and TRAILING positions.
        #
        # TIMING OFFSET: a slice from s0..s0+sl peaks in the correlation output
        # at position (expected_peak - (sps - s0 - sl)) relative to the ring.
        # Without correcting for this offset the wrong search window is used and
        # the slice peak lands outside it entirely.
        #
        # Each entry: (frac, s0, sl, peak_offset, F_up_sl, F_dn_sl)
        #   peak_offset = sps - s0 - sl
        #   adjusted_peak = slice_found_peak + peak_offset
        self._slices = []
        sps_f = float(sps)
        for frac, s0_frac in ((0.33, 0.00),   # leading  1/3 — low-freq tail (audible in bird env)
                               (0.33, 0.33),   # centre   1/3
                               (0.33, 0.67),   # trailing 1/3 — high-freq tail
                               (0.50, 0.00),   # leading  1/2
                               (0.50, 0.50)):  # trailing 1/2
            sl = max(8, int(sps_f * frac))
            s0 = min(sps - sl, int(sps_f * s0_frac))
            s1 = s0 + sl
            peak_offset = sps - s0 - sl   # correction to convert slice peak → full-chirp position
            wsl   = np.hanning(sl).astype(np.float64)
            up_sl = np.zeros(nfft, dtype=np.float64)
            dn_sl = np.zeros(nfft, dtype=np.float64)
            up_sl[:sl] = (up_ref[s0:s1] * wsl)[::-1]
            dn_sl[:sl] = (down_ref[s0:s1] * wsl)[::-1]
            self._slices.append((frac, s0, sl, peak_offset,
                                  np.fft.rfft(up_sl, nfft),
                                  np.fft.rfft(dn_sl, nfft)))

    def correlate(self, buf):
        B      = np.fft.rfft(buf.astype(np.float64), self.nfft)
        up_env = np.abs(np.fft.irfft(B * self.F_up, self.nfft)[:len(buf)])
        dn_env = np.abs(np.fft.irfft(B * self.F_dn, self.nfft)[:len(buf)])
        return up_env, dn_env

    def symbol_decision(self, buf, centre):
        # ── Spectral whitening ──────────────────────────────────────────────
        # Birds and tonal interference are narrowband; the chirp is wideband.
        # Dividing each FFT bin by its magnitude flattens the spectrum so
        # narrowband energy no longer dominates the correlation.  The chirp,
        # whose energy is spread across all bins, survives whitening with only
        # a slight SNR penalty; a bird tone loses its amplitude advantage.
        buf_f = buf.astype(np.float64)
        B_raw = np.fft.rfft(buf_f, self.nfft)
        eps   = float(np.max(np.abs(B_raw))) * 1e-4 + 1e-12
        B     = B_raw / (np.abs(B_raw) + eps)   # whitened spectrum

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

        # ── Sub-symbol slice MFs ────────────────────────────────────────────
        # Each slice covers a different time/frequency segment of the chirp.
        # CRITICAL: a slice from s0..s0+sl peaks in the correlation at
        #   (centre - peak_offset)  where  peak_offset = sps - s0 - sl
        # so we shift the search window per-slice, then add peak_offset back
        # to return found_peak in full-chirp coordinates for the timing loop.
        best_conf  = full_conf
        best_peak  = full_peak
        best_wpk   = full_wpk
        used_frac  = 1.0

        if abs(full_conf) < 0.50 and self._slices:
            for frac, s0, sl, peak_offset, F_up_sl, F_dn_sl in self._slices:
                sl_centre = centre - peak_offset        # where this slice peaks in ring
                sl_lo = max(0,            sl_centre - sr)
                sl_hi = min(len(up_env), sl_centre + sr)
                if sl_hi <= sl_lo:
                    continue
                u_sl = np.abs(np.fft.irfft(B * F_up_sl, self.nfft)[:len(buf)])
                d_sl = np.abs(np.fft.irfft(B * F_dn_sl, self.nfft)[:len(buf)])
                u_pk = float(np.max(u_sl[sl_lo:sl_hi]))
                d_pk = float(np.max(d_sl[sl_lo:sl_hi]))
                tot  = u_pk + d_pk
                if tot < 1e-12:
                    continue
                sl_conf  = (u_pk - d_pk) / tot
                # Bias toward trailing (high-freq) slices: parakeets chirp upward
                # from ~400-500Hz so they always cover the low/mid portion of the
                # chirp in time.  The trailing 1/3 (highest frequencies) is the
                # most protected segment.  s0_frac=0.67 → trailing, 0.50 → upper
                # half, 0.33 → centre, 0.00 → leading (least reliable with birds).
                s0_frac  = s0 / float(sl + s0) if (sl + s0) > 0 else 0.0
                if   s0_frac >= 0.60: freq_bias = 1.35   # trailing — most reliable
                elif s0_frac >= 0.40: freq_bias = 1.15   # upper-half / centre-high
                elif s0_frac >= 0.25: freq_bias = 1.00   # centre
                else:                 freq_bias = 0.75   # leading — parakeet zone
                weighted = abs(sl_conf) * (frac ** 0.5) * freq_bias
                if weighted > abs(best_conf) * (used_frac ** 0.5):
                    w_env     = u_sl if u_pk >= d_pk else d_sl
                    raw_peak  = int(np.argmax(w_env[sl_lo:sl_hi])) + sl_lo
                    best_conf = sl_conf
                    best_peak = raw_peak + peak_offset  # normalise to full-chirp position
                    best_wpk  = max(u_pk, d_pk)
                    used_frac = frac

        confidence = best_conf
        bit        = 1 if confidence > 0 else 0
        found_peak = best_peak
        winner_pk  = best_wpk

        # SNR from whitened full-symbol envelope (comparable across symbols)
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
    """
    Sliding-window preamble correlator with a persistent confidence history.

    Improvement over the original single-window design
    ---------------------------------------------------
    The original kept only a rolling list that was wiped on every resync.
    This meant:
      · Startup-latency misses: the first preamble was usually past before
        the decoder had collected a full 13-symbol window.
      · One-symbol-off misses: timing jitter of ±1 sample de-correlated the
        window and the opportunity was lost for 37 seconds.
      · Post-resync blindness: resync() cleared all history, so the detector
        started empty into a stream that was mid-preamble.

    The new design maintains a *persistent* circular buffer of the last
    `lookback` confidence values that is NEVER cleared by reset().  On every
    new push(), the buffer is scanned at all offsets from 0 to
    (lookback - n_preamble), computing a normalised correlation score at each
    position.  The best score across all offsets is returned.

    This means:
      · A preamble that arrived 1–5 symbols before the current push is still
        found — timing jitter and startup latency become recoverable.
      · After a resync the history already contains the confidences that were
        computed before the resync; if a real preamble was present, it will
        be found on the very next push() call.
      · The syndet never needs to be "warmed up" after a resync — the
        `timing_settled` flush-guard is no longer needed for correctness
        (though it is still used to gate the timing loop separately).

    Normalisation
    -------------
    Each window is normalised by its mean absolute confidence so the score
    sits in [−1, 1] independent of SNR.  At low SNR (avg |conf| ≈ 0.15) the
    raw dot-product is bounded by ~0.15 — far below any useful threshold.
    After normalisation, 0.5 means "50 % more matched energy than mismatched"
    regardless of signal amplitude.

    Parameters
    ----------
    preamble  : list of 0/1 bits
    threshold : minimum normalised score to declare sync (default 0.65)
    lookback  : how many symbols of history to keep and scan (default 3×N)
    """

    def __init__(self, preamble, threshold=0.65, lookback=None):
        self.n         = len(preamble)
        self.threshold = threshold
        self.expected  = np.array([1.0 if b else -1.0 for b in preamble],
                                  dtype=np.float32)
        self.lookback  = lookback if lookback is not None else self.n * 3
        # Circular buffer — never cleared on reset(), survives resyncs
        self._history  = np.zeros(self.lookback, dtype=np.float32)
        self._pos      = 0          # next write position
        self._filled   = 0          # how many entries have been written so far
        # The last best-scoring offset (symbol index ago, 0 = current)
        self.best_lag  = 0

    def push(self, confidence):
        """
        Add one symbol confidence and scan history for the best preamble match.

        Returns
        -------
        acquired : bool   — True if best score >= threshold
        score    : float  — best normalised correlation score found
        """
        self._history[self._pos] = float(confidence)
        self._pos    = (self._pos + 1) % self.lookback
        self._filled = min(self._filled + 1, self.lookback)

        if self._filled < self.n:
            return False, 0.0

        # Number of complete windows we can evaluate:
        # from lag=0 (trailing window ending at current symbol) up to
        # lag = min(lookback - n, filled - n) symbols back.
        max_lag = min(self.lookback - self.n, self._filled - self.n)

        best_score = -999.0
        best_lag   = 0

        for lag in range(max_lag + 1):
            # Extract window ending `lag` symbols before the current write pos.
            # _pos points to the slot AFTER the most recently written value.
            end_excl = (self._pos - lag) % self.lookback   # exclusive end
            # Build contiguous window via index arithmetic
            idxs   = [(end_excl - self.n + i) % self.lookback
                      for i in range(self.n)]
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
        """
        Soft reset: do NOT clear the history buffer.
        The timing loop has been re-armed; history is still valid and the
        next push() will immediately scan it for a past preamble hit.
        Call clear_history() only when the stream is fundamentally
        discontinuous (e.g. test restart, seek, hard error recovery).
        """
        pass   # intentionally empty — history persists

    def clear_history(self):
        """Hard reset: wipe history for a completely fresh start."""
        self._history[:] = 0.0
        self._pos        = 0
        self._filled     = 0
        self.best_lag    = 0


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
    alpha_acq      = float(cfg.get("timing_acq_gain",    0.25))   # was 0.50 — too aggressive at low SNR
    # Mean-reversion rate: how fast n_read decays back toward sps when freewheeling.
    # Without this, a single bad correction sends n_read far from sps and it stays
    # there indefinitely — every subsequent symbol compounds the drift.
    nread_revert   = float(cfg.get("nread_revert_alpha", 0.05))
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

    # Optional raw-input WAV capture (set cfg["wav_out"] to a file path).
    wav_out   = cfg.get("wav_out")
    _capture  = [] if wav_out else None   # accumulates raw float32 chunks

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
    # Preamble resolution: css_info.json is the authoritative source because it
    # was written by the encoder that produced the audio.  The run-level cfg
    # (css_decode_config.json) may override only when explicitly set.
    info_preamble = conf.get("preamble")    # what the encoder actually used
    cfg_preamble  = cfg.get("preamble")     # explicit override from run cfg

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
        print(f"WARNING: no preamble in css_info.json or cfg — defaulting to Barker-13",
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

    # IIR timing tracker parameters (used in the decode loop).
    # min_tim_conf : confidence below which we freewheel (don't update peak_est).
    # max_terr_smp : hard clamp on timing error fed to the IIR, prevents a single
    #                noise spike from derailing the tracker.
    # tim_track_alpha : IIR weight for blending found_peak into peak_est; faster
    #                   than timing_frac (alpha) so peak_est tracks coarsely while
    #                   timing_frac handles fine ±1-sample corrections.
    min_tim_conf    = float(cfg.get("min_timing_conf",    0.10))   # was 0.30 — too high for low SNR
    max_terr_smp    = int(  cfg.get("max_timing_err_smp", sps // 2))  # was sps//4 — too tight
    # Pre-settled clamp: 10× tighter than max_terr_smp.  Used before timing_settled
    # latches so that background noise during lead silence cannot accumulate enough
    # ring-position drift to push a long-duration (≥ 0.5 s) preamble chirp outside
    # the search window.  Configurable; default = 2 % of sps.
    presync_max_terr_smp = int(cfg.get("presync_max_terr_smp", max(1, int(sps * 0.02))))
    tim_track_alpha = float(cfg.get("timing_track_alpha", 0.50))

    # base_bits: the unpadded frame size (what RS operates on)
    base_bits = int(conf.get("base_bits",
                             (1 + chunk_size + crc_bytes + rs_parity) * 8))

    # frame_symbols: what the decoder actually collects per frame.
    # With interleaving this may be > base_bits due to padding.
    frame_symbols = int(conf.get("frame_symbols", base_bits))
    preamble_per_frame = bool(conf.get("preamble_per_frame", False))
    diff_enc  = bool(conf.get("diff_encode", False))
    conv_code = conf.get("conv_code", None)
    pll_weight = float(cfg.get("pll_weight", 0.30))   # blend weight for IF-slope confidence

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
    timing_settled = False  # latches True once |terr| first drops within threshold

    print(f"LISTENING fs={fs} sps={sps} dur={dur}s "
          f"f={f_min:.0f}-{f_max:.0f}Hz "
          f"chunk={chunk_size}B crc={crc_bytes}B rs_t={rs_t} "
          f"interleave_rows={interleave_rows} "
          f"base_bits={base_bits} frame_symbols={frame_symbols} "
          f"sync_word={len(sync_word)} threshold={sync_threshold} "
          f"search_range=±{correlator.search_range}",
          flush=True)

    ring           = np.zeros(buf_size, dtype=np.float32)
    bit_stream     = []
    # Shadow buffer of (bit, confidence) for every pre-sync symbol.
    # Survives soft resyncs so that when SYNC_ACQUIRED fires at lag>0
    # we can seed bit_stream with the payload bits already decoded.
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
    n_read         = sps  # samples to read for next symbol

    diag = {
        "sync_acquired": False, "sync_at_symbol": None, "sync_score": 0.0,
        "symbols_received": 0, "peak_rms": 0.0, "mean_rms": 0.0,
        "silence": False, "avg_snr_db": 0.0,
    }

    def do_resync(reason="", reset_timing=False):
        nonlocal sync_found, bit_stream, consec_errors, timing_settled
        nonlocal peak_est, timing_frac, n_read
        print(f"RESYNC{': ' + reason if reason else ''}"
              f"{' (timing reset)' if reset_timing else ' (soft)'}",
              flush=True)
        sync_found     = False
        bit_stream     = []
        consec_errors  = 0
        timing_settled = not reset_timing   # keep timing if soft resync
        if reset_timing:
            peak_est    = expected_peak
            timing_frac = 0.0
            n_read      = sps
        # Always clear both history buffers regardless of soft/hard.
        # The ring buffer persistence was designed to catch startup-latency
        # missed preambles on the FIRST sync acquisition.  For all subsequent
        # frames the timing is locked and the previous frame's confidence
        # history is actively harmful: the syndet scans all 27 windows on
        # the very first push after resync, and with |conf|≈0.75 data symbols
        # P(false sync in any window) ≈ 4.6% → P(hit in 27 windows) ≈ 71%.
        # Clearing here means the syndet starts fresh after each frame; the
        # startup-latency case is unaffected because the first sync acquisition
        # never goes through do_resync at all.
        syndet.clear_history()
        pre_sync_buf.clear()

    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype="float32",
                            device=device) as stream:
            signal_present = True   # updated each symbol; True keeps timing active if assignment is skipped
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
                # Gate timing updates on actual signal energy so background noise
                # during the lead-silence period cannot drift the search window.
                signal_present = float(np.sqrt(np.mean(raw.astype(np.float64) ** 2))) > silence_thr

                ring = np.roll(ring, -n_read)
                ring[-n_read:] = agc.apply(raw)
                sym_count += 1

                bit, confidence, found_peak, snr_db, winner_pk = (
                    correlator.symbol_decision(ring, peak_est))

                # ── IIR-smoothed peak tracker ────────────────────────────────────
                # Only update peak_est when the correlator is confident enough.
                # Below min_tim_conf the chirp energy is too ambiguous (e.g. the
                # chirp band is partially blocked) so we freewheel: let the ring
                # advance at the current n_read and trust the last good estimate.
                abs_conf = abs(confidence)
                if signal_present and abs_conf >= min_tim_conf:
                    raw_err = found_peak - expected_peak
                    # Two-stage error clamp:
                    #   Before timing_settled latches, background noise can
                    #   trigger timing corrections during the lead silence.
                    #   With long symbol durations (1 s) even a few noise-driven
                    #   corrections accumulate enough ring-position error to push
                    #   the first real preamble chirp outside the search window,
                    #   so the decoder never acquires sync on the true preamble.
                    #   Use a 10× tighter clamp pre-settled so the total possible
                    #   drift across N silence symbols stays well inside ±search_range.
                    #   Once timing_settled latches (first real chirp seen with
                    #   low timing_err), switch to the full clamp for fast tracking.
                    eff_max_terr = max_terr_smp if timing_settled else presync_max_terr_smp
                    clamped_err = int(np.clip(raw_err, -eff_max_terr, eff_max_terr))
                    # Blend new measurement into peak_est (IIR, not direct assign).
                    peak_est   = int(round(
                        (1.0 - tim_track_alpha) * peak_est +
                         tim_track_alpha         * found_peak
                    ))
                    timing_err = clamped_err
                else:
                    # Freewheel: no correction, just record the raw error for STATUS log
                    timing_err = found_peak - expected_peak

                snr_history.append(snr_db)

                # AGC from MF peak — reuse winner_pk already computed in symbol_decision
                # to avoid a redundant second correlate() call on the same ring buffer.
                agc.update_from_peak(winner_pk)

                # Timing correction for next read.
                # Only apply when confident; freewheel otherwise (n_read unchanged).
                # When terr < 0 (chirp left): read fewer → ring advances less → chirp right.
                # When terr > 0 (chirp right): read more → ring advances more → chirp left.
                if signal_present and abs_conf >= min_tim_conf:
                    cur_alpha = alpha if sync_found else alpha_acq
                    timing_frac += cur_alpha * timing_err
                    timing_frac  = float(np.clip(timing_frac, -sps // 4, sps // 4))
                    int_corr     = int(timing_frac)
                    timing_frac -= int_corr
                    n_read = max(sps // 2, sps + int_corr)
                else:
                    # Freewheel: gently pull n_read back toward sps so drift
                    # accumulated during a noise burst doesn't persist forever.
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
                          f"sync={'YES' if sync_found else 'no'}",
                          flush=True)
                    last_status_t = time.time()

                if not sync_found:
                    # Flush the syndet window only while timing is actively
                    # unsettled.  Once |terr| has dropped below the threshold
                    # we leave the accumulator alone so the preamble window
                    # can fill cleanly.  Using a latch (timing_settled) means
                    # we stop flushing as soon as tracking converges — even if
                    # a single later symbol has a noisy terr estimate.
                    # Flush the syndet window only when timing is truly lost
                    # (|terr| > half a symbol period).  The old sps//8 threshold
                    # was too tight at low SNR: timing_err is noisy so terr
                    # frequently exceeded sps//8 even when timing was basically
                    # correct, preventing timing_settled from ever latching and
                    # keeping the syndet permanently empty.
                    tim_flush_thr = int(cfg.get("syndet_flush_terr", sps // 2))
                    if abs(timing_err) <= tim_flush_thr:
                        timing_settled = True
                    # syndet.reset() was previously called here while timing was
                    # unsettled to prevent noise from filling the window.  Now that
                    # reset() is a no-op and the ring buffer persists, this is
                    # unnecessary — history is always valid regardless of timing state.
                    # Record the raw bit decision for lag-recovery on sync.
                    pre_sync_buf.append((1 if confidence > 0 else 0, confidence))
                    acquired, score = syndet.push(confidence)
                    if acquired:
                        lag = syndet.best_lag
                        print(f"SYNC_ACQUIRED sym={sym_count} score={score:.3f} "
                              f"snr={snr_db:.1f}dB terr={timing_err:+d}"
                              f"{f' lag={lag}' if lag else ''}", flush=True)
                        sync_found            = True
                        # Seed bit_stream with payload symbols decoded BEFORE
                        # sync fired.  pre_sync_buf holds the last lookback
                        # (bit, confidence) pairs from the pre-sync branch.
                        # lag=0 → preamble just ended, no payload yet.
                        # lag=N → preamble ended N symbols ago; take the last
                        #         N entries from pre_sync_buf as initial payload.
                        # This is the fix for long symbol durations where lag>0
                        # is common and bit_stream was always empty pre-sync.
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
                    elif score > 0.20:
                        print(f"SYNC_CANDIDATE sym={sym_count} score={score:.3f} "
                              f"snr={snr_db:.1f}dB conf={confidence:+.2f} "
                              f"need={sync_threshold:.2f}", flush=True)
                else:
                    # ── PLL (IF-slope) confidence blend ─────────────────────────
                    # Post-sync only: compute instantaneous frequency slope over
                    # the symbol segment and blend with MF confidence.
                    # The IF slope is orthogonal to amplitude: a chirp buried under
                    # a bird call that decimates MF amplitude still has a detectable
                    # slope direction in the surviving frequency fragment.
                    final_conf = confidence
                    if pll_weight > 0:
                        seg_start  = max(0, peak_est - sps + 1)
                        seg_end    = min(len(ring), peak_est + 1)
                        pll_conf, pll_r2 = if_slope_confidence(
                            ring[seg_start:seg_end], f_min, f_max, sps, fs)
                        # Only blend when the linear fit is reasonable (r² > 0.1)
                        # — very noisy segments give meaningless slope estimates
                        if pll_r2 > 0.10:
                            w = pll_weight * pll_r2   # scale by fit quality
                            final_conf = (1 - w) * confidence + w * pll_conf

                    final_bit  = 1 if final_conf > 0 else 0
                    bit_stream.append((final_bit, final_conf))   # signed conf for Viterbi

                    if len(bit_stream) >= frame_symbols:
                        frame_items = bit_stream[:frame_symbols]
                        bit_stream  = bit_stream[frame_symbols:]

                        # frame_items: list of (bit, signed_conf)
                        # signed_conf in [-1,+1]: used by Viterbi; abs used by erasure marking
                        frame_bits  = [b for b, _ in frame_items]
                        frame_confs = [c for _, c in frame_items]

                        # ── Step 1: De-interleave ─────────────────────────────────
                        # Must happen before Viterbi: interleaver acts on coded bits
                        # so we must undo it on coded soft values first.
                        if interleave_rows > 1:
                            # De-interleave both hard bits and soft confs together
                            n_coded = len(frame_bits)
                            cols    = n_coded // interleave_rows
                            # Reshape as (cols × rows) then transpose → (rows × cols)
                            idxs = (np.arange(n_coded)
                                      .reshape(cols, interleave_rows)
                                      .T.flatten().tolist())
                            frame_bits  = [frame_bits[i]  for i in idxs]
                            frame_confs = [frame_confs[i] for i in idxs]

                        # ── Step 2: Viterbi soft-decision decoding ────────────────
                        if conv_code == "k7r12":
                            # Pair up (odd, even) soft values as the two coded outputs
                            soft_pairs = [(frame_confs[i], frame_confs[i + 1])
                                          for i in range(0, len(frame_confs) - 1, 2)]
                            frame_bits = viterbi_k7(soft_pairs)
                            # After Viterbi, hard bits — rebuild abs confs from bit certainty
                            # (Viterbi path metric gives no per-bit reliability; use abs_conf=1
                            # since Viterbi already made the ML decision)
                            frame_confs = [1.0] * len(frame_bits)

                        # ── Step 3: Differential decoding ────────────────────────
                        if diff_enc:
                            ref = sync_word[-1]   # last preamble bit = absolute reference
                            frame_bits = diff_decode_bits(frame_bits, ref_bit=ref)

                        raw_bytes  = get_bytes_from_bits(frame_bits[:base_bits])

                        frame_snr = (float(np.mean(snr_history[-frame_symbols:]))
                                     if len(snr_history) >= frame_symbols else snr_db)

                        # ── Step 4: RS decode with erasure marking ───────────────
                        # Only meaningful without conv (with Viterbi the BER is low
                        # enough that erasures are rare; still run RS for error detection).
                        era_thr   = float(cfg.get("erasure_conf_thr", 0.25))
                        erase_pos = []
                        if rs_codec is not None and era_thr > 0 and conv_code is None:
                            for byte_i in range(len(raw_bytes)):
                                b0 = byte_i * 8
                                b1 = min(b0 + 8, len(frame_confs))
                                if b1 > b0 and min(abs(c) for c in frame_confs[b0:b1]) < era_thr:
                                    erase_pos.append(byte_i)
                            if erase_pos:
                                print(f"ERASURES n={len(erase_pos)} "
                                      f"bytes={erase_pos[:8]}", flush=True)

                        rs_errors = 0
                        if rs_codec is not None:
                            try:
                                raw_bytes, rs_errors = rs_decode_frame(
                                    raw_bytes, rs_codec, erase_pos or None)
                                if rs_errors:
                                    print(f"RS_CORRECTED errors={rs_errors} "
                                          f"snr={frame_snr:.1f}dB", flush=True)
                            except Exception as e:
                                print(f"RS_FAIL snr={frame_snr:.1f}dB err={e}", flush=True)
                                frames_errored += 1
                                consec_errors  += 1
                                if consec_errors >= resync_after:
                                    do_resync(f"{consec_errors} RS failures",
                                              reset_timing=True)
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
                        else:
                            raw_hex = raw_bytes[:6].hex() if raw_bytes else "empty"
                            print(f"FRAME_ERROR seq={raw_bytes[0] if raw_bytes else -1} "
                                  f"reason=crc_fail snr={frame_snr:.1f}dB "
                                  f"raw={raw_hex}", flush=True)
                            frames_errored += 1
                            consec_errors  += 1
                            # If this is the very first frame after acquiring sync and
                            # it already fails CRC, the sync position is almost certainly
                            # wrong (false sync on payload data).  Resync immediately
                            # rather than waiting for resync_after consecutive failures.
                        if consec_errors >= resync_after:
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


    # Save raw mic input to WAV if requested
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