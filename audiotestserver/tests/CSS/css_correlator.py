# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_correlator.py

Matched-filter correlator and AGC, split out of css_decoder.py.

ChirpCorrelator: full-symbol FFT-domain matched filter against the
up-chirp / down-chirp templates, with sub-symbol slice fallback for
partial-chirp recovery (used when part of a symbol is obscured).

PeakAGC: simple peak-seeking automatic gain control, driving the
matched-filter's winner_pk toward a target level via separate
attack/release time constants.

Segment decomposition (mf_window="none", the default)
-----------------------------------------------------
Matched filtering is linear, so the correlation against a sub-symbol slice is
the sum of the correlations against any disjoint segments that tile that
slice.  Cutting the template at the UNION of every slice boundary yields SIX
segments from which all eight slices AND the full-symbol envelope are exact
sums: 12 inverse transforms per symbol instead of 18, results identical to
floating-point round-off.

The slice geometry is specified in TWELFTHS of a symbol (see _SLICE_SPEC_12)
because that is the coarsest grid on which every boundary the slice table
needs — 0, 1/4, 1/3, 1/2, 2/3, 3/4, 1 — lands exactly.  The previous
`int(sps * 0.33)` / `int(sps * 0.67)` truncation missed exact thirds by a few
samples, which fragmented the partition into 8 segments (two of them 23 and 1
samples long) and gave back most of the saving.

Every segment template is a masked copy of the REVERSED full template left at
its own position within the length-sps template (not compacted to offset 0),
so all segment correlations share the same "template ends here" lag and can be
summed sample-for-sample.  A slice's peak therefore already lands at the
full-symbol lag, which is the coordinate the caller wants; the per-slice
`peak_offset` survives only to reproduce the historical search-window clipping
exactly (see symbol_decision).

mf_window="hann" (legacy) applies a separate Hann envelope to each slice, so
its slices are not segments of one common template and cannot be decomposed.
That path keeps the original per-slice transforms.
"""

import os

import numpy as np


# ---------------------------------------------------------------------------
# Signal-path precision
# ---------------------------------------------------------------------------
# SIG is the dtype for everything on the sample path: audio buffers, chirp
# templates, FFT inputs and correlation envelopes.  float32 because the target
# for this code is an embedded C port on a limited-CPU part, whose FFT library
# (CMSIS-DSP arm_rfft_fast_f32 and equivalents) and FPU are single-precision;
# validating the algorithm at float32 HERE means the port is not the first time
# the precision is exercised.  numpy >= 2.0 honours float32 through
# rfft/irfft natively (float32 in -> complex64 out, computed in single
# precision), so this genuinely tests single-precision correlation rather than
# silently upcasting.
#
# What deliberately stays float64: the timing line fits (sample POSITIONS, not
# samples) and scalar statistics accumulators.  Those are tiny — a few dozen
# points — so they cost nothing to keep in double, and on the embedded side they
# would be a hand-rolled closed-form fit over relative coordinates rather than
# this general lstsq.  Making them float32 would buy no speed and would put the
# grid fit's conditioning at risk for nothing.
# CSS_SIG_DTYPE=float64 restores the pre-port double-precision signal path.
# Kept as a runtime switch (not a source edit) so the two can be A/B'd on the
# same capture — the regression suite is the only way to tell a real precision
# regression from an unrelated change, and that comparison needs to stay cheap.
SIG = np.float32 if os.environ.get("CSS_SIG_DTYPE", "float32") == "float32" else np.float64


def corr_noise_floor(win, sps):
    """Round-off ceiling of a length-`sps` matched-filter correlation over `win`.

    Any correlation output at or below this is arithmetic noise, not signal.

    Must be computed, not a fixed constant: the decoder's "is there a chirp
    here" test is a RATIO (peak / median of the envelope), so a ratio between
    two round-off-magnitude numbers reads as large as one between two strong
    ones (same extreme-value trap the slice-selection gate in
    symbol_decision() documents). A fixed epsilon encodes an assumption about
    dtype: float64 round-off is ~1e-15, float32 is ~1e-7, so an epsilon tuned
    for one precision over- or under-suppresses at the other.

    Scaling with BOTH the window's amplitude and the template length keeps the
    test precision-agnostic: correlation output grows with scale x sps, so its
    round-off grows the same way, and dividing that out leaves a threshold
    that means the same thing at float32, float64, or fixed-point. A silent
    window reads 0 dB at any precision; a real chirp is orders of magnitude
    above this and untouched.
    """
    if len(win) == 0:
        return 0.0
    scale = float(np.max(np.abs(win)))
    if scale <= 0.0:
        return 0.0
    return float(np.finfo(win.dtype).eps) * scale * float(sps)


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

    # Sub-symbol MF slices for partial-chirp recovery, as (start, end) in
    # TWELFTHS of a symbol.  Coverage:
    #   1/3 start / mid / end   — fine-grained; catches small obscurations.
    #   1/2 start / centre / end — medium; the centre slice (1/4..3/4) fills
    #     the gap when BOTH edges are obscured simultaneously.
    #   2/3 start / end          — stronger SNR than 1/2 when only one third
    #     is blocked.
    _SLICE_SPEC_12 = ((0, 4), (4, 8), (8, 12),
                      (0, 6), (3, 9), (6, 12),
                      (0, 8), (4, 12))

    # Below this many samples per symbol the twelfth grid degenerates (segments
    # of a few samples), so fall back to the direct per-slice transforms.
    _MIN_SPS_FOR_SEGMENTS = 96

    def __init__(self, up_ref, down_ref, sps, search_frac=0.25, buf_size=None,
                 mf_window="none"):
        self.sps  = sps
        buf       = buf_size if buf_size else 3 * sps
        nfft      = 1 << (buf + sps - 1).bit_length()
        self.nfft = nfft

        if mf_window == "hann":
            # Legacy: Hann-windowed template (mismatched to rectangular-envelope transmit)
            win     = np.hanning(sps).astype(SIG)
            up_win  = (up_ref[:sps]   * win)[::-1]
            dn_win  = (down_ref[:sps] * win)[::-1]
        else:
            # Linear chirp sweep: unwindowed template — properly matched to transmit.
            # Higher peak correlation, ~3 dB SNR gain vs Hann.
            up_win  = up_ref[:sps].astype(SIG)[::-1]
            dn_win  = down_ref[:sps].astype(SIG)[::-1]

        self.F_up = np.fft.rfft(up_win, nfft)
        self.F_dn = np.fft.rfft(dn_win, nfft)
        self.search_range = max(16, int(sps * search_frac))
        self._mf_window   = mf_window

        # Slice geometry, shared by both paths.  Entries are
        # (frac, s0, sl, peak_offset, seg_lo, seg_hi); seg_* index into the
        # segment tables below and are None on the legacy path.
        self._slices   = []
        self._seg_Fup  = None
        self._seg_Fdn  = None

        use_segments = (mf_window != "hann" and sps >= self._MIN_SPS_FOR_SEGMENTS)

        if use_segments:
            edges12 = sorted({0, 12}
                             | {a for a, _ in self._SLICE_SPEC_12}
                             | {b for _, b in self._SLICE_SPEC_12})
            edge_smp = [(k * sps) // 12 for k in edges12]

            # Segment templates: the reversed full template masked to one
            # segment and left in place, so every segment correlation shares
            # the full template's lag reference and they sum directly.
            self._seg_Fup = []
            self._seg_Fdn = []
            for a, b in zip(edge_smp[:-1], edge_smp[1:]):
                # forward samples [a, b) are reversed-template [sps-b, sps-a)
                mu = np.zeros(sps, dtype=SIG)
                md = np.zeros(sps, dtype=SIG)
                mu[sps - b:sps - a] = up_win[sps - b:sps - a]
                md[sps - b:sps - a] = dn_win[sps - b:sps - a]
                self._seg_Fup.append(np.fft.rfft(mu, nfft))
                self._seg_Fdn.append(np.fft.rfft(md, nfft))

            for a12, b12 in self._SLICE_SPEC_12:
                s0 = edge_smp[edges12.index(a12)]
                s1 = edge_smp[edges12.index(b12)]
                self._slices.append(((b12 - a12) / 12.0, s0, s1 - s0, sps - s1,
                                     edges12.index(a12), edges12.index(b12)))
        else:
            sps_f = float(sps)
            for a12, b12 in self._SLICE_SPEC_12:
                s0 = (a12 * sps) // 12
                s1 = (b12 * sps) // 12
                sl = s1 - s0
                if sl < 8:
                    continue
                if mf_window == "hann":
                    wsl        = np.hanning(sl).astype(SIG)
                    up_sl_data = (up_ref[s0:s1]   * wsl)[::-1]
                    dn_sl_data = (down_ref[s0:s1] * wsl)[::-1]
                else:
                    # Unwindowed slices: match the transmitted waveform faithfully
                    up_sl_data = up_ref[s0:s1].astype(SIG)[::-1]
                    dn_sl_data = down_ref[s0:s1].astype(SIG)[::-1]

                up_sl = np.zeros(nfft, dtype=SIG)
                dn_sl = np.zeros(nfft, dtype=SIG)
                up_sl[:sl] = up_sl_data
                dn_sl[:sl] = dn_sl_data
                self._slices.append(((b12 - a12) / 12.0, s0, sl, sps - s1,
                                     np.fft.rfft(up_sl, nfft),
                                     np.fft.rfft(dn_sl, nfft)))

    def correlate(self, buf):
        B      = np.fft.rfft(buf.astype(SIG), self.nfft)
        up_env = np.abs(np.fft.irfft(B * self.F_up, self.nfft)[:len(buf)])
        dn_env = np.abs(np.fft.irfft(B * self.F_dn, self.nfft)[:len(buf)])
        return up_env, dn_env

    def symbol_decision(self, buf, centre, search_range_override=None):
        nfft  = self.nfft
        buf_f = buf.astype(SIG)
        B     = np.fft.rfft(buf_f, nfft)
        n     = len(buf)

        if self._seg_Fup is not None:
            # Prefix sums over the segment axis: any slice is one subtraction,
            # and the full-symbol envelope is the last prefix.  12 irffts total.
            S  = len(self._seg_Fup)
            Pu = np.zeros((S + 1, n), dtype=SIG)
            Pd = np.zeros((S + 1, n), dtype=SIG)
            for j in range(S):
                Pu[j + 1] = Pu[j] + np.fft.irfft(B * self._seg_Fup[j], nfft)[:n]
                Pd[j + 1] = Pd[j] + np.fft.irfft(B * self._seg_Fdn[j], nfft)[:n]
            up_env = np.abs(Pu[S])
            dn_env = np.abs(Pd[S])
        else:
            Pu = Pd = None
            up_env = np.abs(np.fft.irfft(B * self.F_up, nfft)[:n])
            dn_env = np.abs(np.fft.irfft(B * self.F_dn, nfft)[:n])

        sr = search_range_override if search_range_override is not None else self.search_range
        lo = max(0, centre - sr)
        hi = min(n,  centre + sr)
        if hi <= lo:
            return 0, 0.0, centre, -60.0, 0.0

        # Arithmetic-noise floor, same reasoning as corr_noise_floor()'s
        # docstring: `1e-12` alone is a float64-era constant, and once the
        # signal path runs at float32 (or fixed-point on the target) round-off
        # can sit ABOVE it, at which point up_pk/dn_pk are round-off and
        # full_conf is a decisive-looking ratio between two meaningless
        # numbers.  Raising the guard to the measured round-off ceiling can
        # only reject peaks that are, by construction, not signal.
        _floor = corr_noise_floor(buf_f, self.sps)
        up_pk = float(np.max(up_env[lo:hi]))
        dn_pk = float(np.max(dn_env[lo:hi]))
        total = up_pk + dn_pk
        if total < 1e-12 or max(up_pk, dn_pk) <= _floor:
            return 0, 0.0, centre, -60.0, 0.0

        full_conf  = (up_pk - dn_pk) / total
        full_peak  = int(np.argmax(
                         (up_env if up_pk >= dn_pk else dn_env)[lo:hi])) + lo
        full_wpk   = max(up_pk, dn_pk)

        best_conf  = full_conf
        best_peak  = full_peak
        best_wpk   = full_wpk
        used_frac  = 1.0
        _winner    = 'full'   # for logging

        # Slice candidates are ALWAYS evaluated — there is deliberately no
        # cheap gate on abs(full_conf) short-circuiting them.  full_conf is a
        # RATIO, (up_pk - dn_pk)/(up_pk + dn_pk), and a ratio between two
        # small, noise-floor-magnitude numbers looks just as "decisive" as one
        # between two strong ones.  Confirmed against a real degraded capture:
        # 33 of 44 affected symbols had |full_conf| >= 0.50 (so the old gate
        # skipped the fallback) despite winner_pk sitting at 1-3.5 — noise-floor
        # level, vs 5-17 during confirmed genuine lock.  The gate was blind to
        # exactly the case this mechanism exists for: a partially-obscured
        # chirp, where the full-window correlation is corrupted by the obscured
        # portion but a clean sub-interval still carries an unambiguous signal.
        #
        # snr_db / confidence-ratio triggers were considered and rejected: both
        # are LOCAL/RELATIVE measures that extreme-value statistics can inflate
        # even for a purely noise-driven peak when searching many candidate
        # positions.  Selection is left entirely to the coverage-weighted
        # comparison plus the absolute-level gate below.
        if self._slices:
            _slice_log = []   # (idx, frac, sl_conf, weighted, u_pk, d_pk, wins)
            for _si, (frac, s0, sl, peak_offset, a, b) in enumerate(self._slices):
                # Search window in CONVERTED (full-symbol lag) coordinates.
                # The historical code searched in slice-local coordinates over
                # [max(0, centre - peak_offset - sr), min(n - peak_offset, ...))
                # and then added peak_offset, which is identically the window
                # below — including its clipping of the low end at peak_offset.
                sl_lo = max(peak_offset, centre - sr)
                sl_hi = min(n,           centre + sr)
                if sl_hi <= sl_lo:
                    _slice_log.append((_si, frac, 0.0, 0.0, -1, -1, False))
                    continue

                if Pu is not None:
                    u_win = np.abs(Pu[b, sl_lo:sl_hi] - Pu[a, sl_lo:sl_hi])
                    d_win = np.abs(Pd[b, sl_lo:sl_hi] - Pd[a, sl_lo:sl_hi])
                else:
                    # legacy path: a and b are the per-slice template spectra,
                    # whose correlations live in slice-local coordinates.
                    _l0 = sl_lo - peak_offset
                    _l1 = sl_hi - peak_offset
                    u_win = np.abs(np.fft.irfft(B * a, nfft)[:n][_l0:_l1])
                    d_win = np.abs(np.fft.irfft(B * b, nfft)[:n][_l0:_l1])

                u_pk = float(np.max(u_win))
                d_pk = float(np.max(d_win))
                tot  = u_pk + d_pk
                if tot < 1e-12 or max(u_pk, d_pk) <= _floor:
                    _slice_log.append((_si, frac, 0.0, 0.0, -1, -1, False))
                    continue
                sl_conf  = (u_pk - d_pk) / tot
                s0_frac  = s0 / float(sl + s0) if (sl + s0) > 0 else 0.0
                if   s0_frac >= 0.60: freq_bias = 1.35
                elif s0_frac >= 0.40: freq_bias = 1.15
                elif s0_frac >= 0.25: freq_bias = 1.00
                else:                 freq_bias = 0.75
                weighted = abs(sl_conf) * (frac ** 0.5) * freq_bias
                sl_wins = (
                    weighted > abs(best_conf) * (used_frac ** 0.5)
                    # ── Absolute-level gate ───────────────────────────────
                    # A genuine partially-obscured chirp concentrates its
                    # energy in the clean slice portion: the slice covering
                    # that portion has a HIGHER absolute MF peak than the
                    # full window (which is diluted by the corrupted part).
                    # Noise does the opposite: short-window noise has HIGH
                    # variance in its up/down RATIO (producing extreme
                    # sl_conf by chance) but its absolute peak is always
                    # <= full_wpk (noise doesn't concentrate in one slice).
                    # Requiring sl_wpk > full_wpk enforces this physical
                    # constraint: a slice only wins if it found a stronger
                    # absolute signal than the full window, not just a more
                    # lopsided noise distribution.
                    and max(u_pk, d_pk) > full_wpk
                )
                _slice_log.append((_si, frac, sl_conf, weighted, u_pk, d_pk, sl_wins))
                if sl_wins:
                    w_win     = u_win if u_pk >= d_pk else d_win
                    converted = int(np.argmax(w_win)) + sl_lo
                    if not (0 <= converted < n):
                        continue
                    best_conf = sl_conf
                    best_peak = converted
                    best_wpk  = max(u_pk, d_pk)
                    used_frac = frac
                    _winner   = f'slice{_si}'

            # ── Per-symbol slice verbose log ──────────────────────────────────
            # Enabled when self.verbose is True (set from decoder when needed).
            # Logs full-window result, every slice candidate, and the winner.
            # Output goes to stdout so it appears in the decoder's normal log.
            if getattr(self, 'verbose', False):
                import sys
                print(f"  MF_DEC centre={centre} sr={sr} "
                      f"full: up={up_pk:.4f} dn={dn_pk:.4f} conf={full_conf:+.3f} "
                      f"peak={full_peak} wpk={full_wpk:.4f}",
                      file=sys.stdout, flush=True)
                for (_si, frac, sl_conf, weighted, u_pk, d_pk, sl_wins) in _slice_log:
                    print(f"    slice{_si} frac={frac:.2f} "
                          f"up={u_pk:.4f} dn={d_pk:.4f} "
                          f"conf={sl_conf:+.3f} weighted={weighted:.4f} "
                          f"{'WIN' if sl_wins else '   '}",
                          file=sys.stdout, flush=True)
                print(f"  MF_DEC winner={_winner} "
                      f"found_peak={best_peak} wpk={best_wpk:.4f} "
                      f"used_frac={used_frac:.2f}",
                      file=sys.stdout, flush=True)

        confidence = best_conf
        bit        = 1 if confidence > 0 else 0
        found_peak = best_peak
        winner_pk  = best_wpk

        # Use final confidence direction (not full_conf) to pick the
        # noise-floor envelope.  If a slice won with the opposite direction
        # to the full-window winner, full_conf has the wrong sign and
        # winner_env would measure noise from the wrong envelope, making
        # snr_db meaningless.
        winner_env = up_env if confidence >= 0 else dn_env
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
        self._rms.append(float(np.sqrt(np.mean(np.square(buf, dtype=SIG),
                                               dtype=np.float64))))
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
