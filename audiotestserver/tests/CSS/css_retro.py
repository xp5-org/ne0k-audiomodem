# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_retro.py

Retrospective two-pass timing recovery, split out of css_decoder.py.

retro_timing_decode() re-derives symbol timing for an entire already-
collected frame in one shot (rather than the streaming per-symbol PLL),
by jointly fitting a timing line (t0, sps_effective) to every symbol's
matched-filter peak position via IRLS outlier-robust least squares.
This gives ~10x better timing precision than the streaming PLL for a
100-symbol frame, since every peak constrains the same two-parameter
model instead of being absorbed one at a time by a noisy loop filter.

Fully self-contained: builds its own matched filter inline (does not
depend on ChirpCorrelator), so it has no dependency on css_correlator.py.
"""

import numpy as np

# Signal-path dtype — see the SIG rationale in css_correlator.py.  Audio,
# templates and MF envelopes are float32; the IRLS timing line fits below
# deliberately stay float64 (they carry sample POSITIONS, are only a few dozen
# points, and cost nothing to keep in double).
from css_correlator import SIG


def retro_timing_decode(full_audio, up_ref, down_ref, sps, frame_symbols,
                        sps_eff=None, rate_est_smp=0.0, verbose=True):
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

    sps_eff: effective symbol period in samples (sps + gap_samps).  The fold
    in step 2 and the coarse peak grid in step 3 must use this period, not
    sps, otherwise adjacent symbol peaks mis-align in the fold and coarse_t0
    is wrong.  Defaults to sps (no gap) for backward compatibility.

    rate_est_smp: persistent per-symbol clock-drift estimate (samples/
    symbol) from the streaming PLL's rate-aided loop.  Consecutive chirps
    in full_audio are spaced sps_eff + rate_est_smp apart (that's what the
    streaming n_read has been producing), not bare sps_eff.  Folding at
    the wrong period smears each chirp's contribution across roughly
    frame_symbols * |rate_est_smp| samples, weakening coarse_t0 for long
    frames.  See sps_eff_d below.  Defaults to 0.0 (no drift correction,
    matches prior behaviour).

    Algorithm:
      1. Full-buffer FFT → MF envelope (one rfft + two irfft calls).
      2. Fold envelope at period sps_eff to find coarse t0 (O(N) sum).
      3. Refine each of the N peaks within ±sps/6 of the coarse prediction.
      4. IRLS robust line fit: peak_pos[k] = t0_fit + k × sps_eff_fit.
         Outliers (|residual| > sps/6) are down-weighted each iteration.
      5. Re-decide all bits at the fitted peak positions using the global
         MF envelopes (not re-running the MF — it's already computed).
    """
    if sps_eff is None:
        sps_eff = sps     # no gap: symbol period equals chirp length
    N = len(full_audio)
    min_expected = sps * (frame_symbols // 2)
    if N < min_expected:
        print(f"RETRO skip: audio too short ({N} < {min_expected})", flush=True)
        return None

    audio = full_audio.astype(SIG)

    # Step 1 — full-buffer MF (one FFT covers the entire frame)
    nfft = 1 << (N + sps - 1).bit_length()
    F_buf = np.fft.rfft(audio, nfft)

    up_tpl = np.zeros(nfft, dtype=SIG)
    dn_tpl = np.zeros(nfft, dtype=SIG)
    up_tpl[:sps] = up_ref[:sps].astype(SIG)[::-1]
    dn_tpl[:sps] = down_ref[:sps].astype(SIG)[::-1]

    up_env = np.abs(np.fft.irfft(F_buf * np.fft.rfft(up_tpl, nfft), nfft)[:N])
    dn_env = np.abs(np.fft.irfft(F_buf * np.fft.rfft(dn_tpl, nfft), nfft)[:N])
    combined = np.maximum(up_env, dn_env)

    # ── Rate-aided fold period ───────────────────────────────
    # The streaming PLL applies rate_est_smp to every n_read, so
    # consecutive chirps within full_audio are spaced sps_eff +
    # rate_est_smp apart, not bare sps_eff.  For D = rate_est_smp != 0,
    # folding at sps_eff smears each chirp's contribution across roughly
    # frame_symbols * |D| samples in `folded` (chirp k's true position is
    # offset by k*D from the fold's k*sps_eff assumption) — for a
    # 74-symbol frame with D=200 that's a ~15000-sample smear, materially
    # weakening coarse_t0's peak.  Folding at sps_eff_d = sps_eff +
    # round(rate_est_smp) collapses the smear back to zero, restoring a
    # sharp single-column peak regardless of frame length.
    sps_eff_d = max(1, sps_eff + int(round(rate_est_smp)))

    # Step 2 — coarse timing: fold envelope at period sps_eff_d.
    # Each encoded symbol occupies sps_eff_d samples (chirp + gap + drift),
    # so peaks repeat at sps_eff_d.  Folding at the wrong period smears
    # adjacent peaks into each other, producing a wrong coarse_t0.
    usable = (N // sps_eff_d) * sps_eff_d
    if usable < sps_eff_d * 2:
        return None
    folded    = combined[:usable].reshape(-1, sps_eff_d).sum(axis=0)
    coarse_t0 = int(np.argmax(folded))

    # Step 3 — refine each peak within ±search_r of the coarse prediction
    search_r = sps // 6
    raw_peaks  = []
    bits_out   = []
    confs_out  = []

    for k in range(frame_symbols):
        exp = coarse_t0 + k * sps_eff_d   # grid spacing includes drift
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
    sps_fit = float(sps_eff_d)  # init from rate-aided period; IRLS refines
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
    if verbose:
        # sps_fit is the fitted symbol PERIOD, so compare it against sps_eff
        # (chirp + gap), not bare sps — otherwise a perfect fit looks like a
        # huge "drift" (e.g. 231525 vs sps=220500 is exactly the 5% gap).
        print(f"RETRO t0={t0_fit:.1f} sps_fit={sps_fit:.4f} "
              f"(nominal sps_eff={sps_eff}) outliers={n_outliers}/{frame_symbols}",
              flush=True)

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
    #
    # SNR-weighted confidence: scale each symbol's conf by the ratio of its
    # peak total to the median peak total across all symbols.  Symbols at or
    # below the noise floor (total << median) get attenuated toward zero,
    # automatically becoming RS erasure candidates without any hard threshold.
    # Normal-signal symbols are unchanged (scale ≈ 1.0).
    #
    # Sub-symbol slice fallback for weak symbols: when abs(conf) < 0.35 after
    # the full-template re-decision, try each sub-symbol slice template on a
    # local window around the fitted peak.  The streaming path already does
    # this (ChirpCorrelator._slices); the retro pass now matches that fallback
    # so a half-obscured symbol is recovered rather than left as a low-conf
    # erasure candidate whenever the slice gives a clearer read.
    half_w  = max(4, search_r // 4)
    fitted  = np.round(t0_fit + k_all * sps_fit).astype(int)

    # Pass 5a: collect all peak totals so we can compute the median scale ref.
    pass5_totals = []
    pass5_data   = []   # (u, d, total) per symbol — reused in 5b
    for k in range(frame_symbols):
        p  = fitted[k]
        lo = max(0, p - half_w)
        hi = min(N, p + half_w)
        if hi <= lo:
            pass5_data.append((0.0, 0.0, 0.0))
            pass5_totals.append(0.0)
            continue
        u     = float(np.max(up_env[lo:hi]))
        d     = float(np.max(dn_env[lo:hi]))
        total = u + d
        pass5_data.append((u, d, total))
        pass5_totals.append(total)

    # Robust scale reference: median of non-zero totals.
    _nonzero = [t for t in pass5_totals if t > 1e-12]
    peak_ref = float(np.median(_nonzero)) if _nonzero else 1.0

    # Precompute slice templates at this retro nfft for the local-slice fallback.
    # Each slice covers a contiguous sub-range of the chirp sweep.  We only
    # need the nfft-sized filter spectra; the actual irfft is done on a local
    # buffer around each weak symbol, so the cost is proportional to the
    # number of weak symbols rather than the full retro-frame size.
    _retro_slices = []
    for _sl_frac, _s0_frac in ((0.33, 0.00), (0.33, 0.33), (0.33, 0.67),
                                (0.50, 0.00), (0.50, 0.25), (0.50, 0.50),
                                (0.67, 0.00), (0.67, 0.33)):
        _sl = max(8, int(sps * _sl_frac))
        _s0 = min(sps - _sl, int(sps * _s0_frac))
        _peak_off = sps - _s0 - _sl
        _retro_slices.append((_sl, _s0, _peak_off,
                              float(_sl) / sps,
                              up_ref[_s0:_s0+_sl].astype(SIG)[::-1],
                              down_ref[_s0:_s0+_sl].astype(SIG)[::-1]))

    # Pass 5b: re-decide bits with SNR weighting + slice fallback.
    _SLICE_THR = 0.35   # full-template conf below this triggers slice fallback
    for k in range(frame_symbols):
        p  = fitted[k]
        u, d, total = pass5_data[k]
        if total < 1e-12:
            continue
        conf = (u - d) / (total + 1e-12)

        # Sub-symbol slice fallback for weak symbols.
        if abs(conf) < _SLICE_THR:
            # Extract a local buffer large enough for the longest slice (sps)
            # centred on the fitted peak.  Running a small local FFT rather
            # than the full retro irfft keeps the extra cost proportional to
            # the number of weak symbols (zero overhead for clean frames).
            loc_lo  = max(0, p - sps)
            loc_hi  = min(N, p + 1)
            loc_seg = audio[loc_lo:loc_hi]
            loc_n   = len(loc_seg)
            if loc_n >= 8:
                loc_nfft = 1 << (loc_n + sps - 1).bit_length()
                F_loc    = np.fft.rfft(loc_seg, loc_nfft)
                best_sl_conf = conf   # start with full-template result
                for _sl, _s0, _peak_off, _frac, _up_sl, _dn_sl in _retro_slices:
                    if _sl > loc_n:
                        continue
                    # Slice peak in local coords falls near p - loc_lo - _peak_off
                    _sl_centre = (p - loc_lo) - _peak_off
                    _sl_lo = max(0,       _sl_centre - half_w)
                    _sl_hi = min(loc_n,   _sl_centre + half_w)
                    if _sl_hi <= _sl_lo:
                        continue
                    _up_tpl = np.zeros(loc_nfft, dtype=SIG)
                    _dn_tpl = np.zeros(loc_nfft, dtype=SIG)
                    _up_tpl[:_sl] = _up_sl
                    _dn_tpl[:_sl] = _dn_sl
                    _u_sl = np.abs(np.fft.irfft(
                        F_loc * np.fft.rfft(_up_tpl, loc_nfft), loc_nfft)[:loc_n])
                    _d_sl = np.abs(np.fft.irfft(
                        F_loc * np.fft.rfft(_dn_tpl, loc_nfft), loc_nfft)[:loc_n])
                    _u_pk = float(np.max(_u_sl[_sl_lo:_sl_hi]))
                    _d_pk = float(np.max(_d_sl[_sl_lo:_sl_hi]))
                    _tot  = _u_pk + _d_pk
                    if _tot < 1e-12:
                        continue
                    _sl_conf = (_u_pk - _d_pk) / (_tot + 1e-12)
                    # Weight by sqrt(slice_fraction) as in the streaming path;
                    # use the slice total as the SNR proxy to prefer slices
                    # that have actual signal in them over those that do not.
                    _weighted = abs(_sl_conf) * (_frac ** 0.5) * (_tot / (peak_ref + 1e-12))
                    _best_w   = abs(best_sl_conf) * 1.0
                    if _weighted > _best_w:
                        best_sl_conf = _sl_conf
                        # Update total for SNR weighting below
                        total = _tot
                if abs(best_sl_conf) > abs(conf):
                    conf = best_sl_conf

        # SNR-weighted confidence: attenuate symbols below the median peak level.
        snr_w        = float(np.clip(total / (peak_ref + 1e-12), 0.0, 1.0))
        bits_out[k]  = 1 if conf > 0 else 0
        confs_out[k] = conf * snr_w

    return bits_out, confs_out, t0_fit, sps_fit, n_outliers


def retro_offset_candidates(anchor_audio, up_ref, down_ref, sps, sps_eff,
                            frame_symbols, max_shift, rate_est_smp=0.0):
    """
    Yield candidate frame decodes from `anchor_audio`, one per whole-symbol
    shift of the frame window, so the caller can pick the correctly-aligned one
    using its CRC as the oracle.

    Why this exists: retro_timing_decode() assumes its buffer begins at the
    frame's first payload symbol — true only when the streaming sync acquired
    on the exact symbol boundary.  At cold start (first frame, AGC/background
    still settling) acquisition lands on the wrong boundary, so retro reads a
    SHIFTED set of chirps: a clean line fit (0 outliers) over the WRONG symbols
    → CRC fail.  The cold-start preamble is itself too weak to re-anchor on, but
    the payload chirps are strong and the CRC is ground truth — so we try a few
    whole-symbol shifts and let the caller's CRC decide.

    Each candidate slices a frame_symbols-long window out of `anchor_audio` and
    runs the full (proven) retro_timing_decode on it.  Decoding each window
    independently — rather than one global fold over the whole buffer — is what
    makes this correct: a single global coarse_t0 over a buffer that also holds
    lead-in silence, training, preamble and clock drift does NOT align to the
    payload's local symbol phase, whereas the per-window fold does.

    Shift convention: shift 0 aligns the frame to the LAST frame_symbols symbols
    in the buffer (where a correctly-timed frame ends); negative shifts move the
    window earlier (the case for a LATE acquisition, where the buffer extends
    past the true frame end).  Yields (bits, confs, t0_fit, sps_fit,
    n_outliers, shift), the first five matching retro_timing_decode's contract.
    Ordered 0, -1, +1, -2, +2, … so aligned frames are accepted first (and the
    caller can stop at the first CRC pass, bounding the cost).

    Implementation note: the matched-filter envelope and the symbol-phase fold
    are computed ONCE over the WHOLE buffer, then each shift only slices the
    precomputed per-symbol peaks, re-fits the timing line and re-reads the bits.
    Decoding each window via a *global* fold rather than a local one (a slice +
    retro_timing_decode) is what makes this robust: a local fold over just the
    48 low-SNR cold-start payload symbols can lock onto the wrong phase, whereas
    the global fold (over training + preamble + payload, many strong chirps)
    pins the correct phase.  It is also cheaper — one FFT total, not one per
    shift.
    """
    n_audio = len(anchor_audio)
    audio   = anchor_audio.astype(SIG)

    # Global matched filter over the whole buffer (computed once).
    nfft = 1 << (n_audio + sps - 1).bit_length()
    F_buf = np.fft.rfft(audio, nfft)
    up_tpl = np.zeros(nfft, dtype=SIG)
    dn_tpl = np.zeros(nfft, dtype=SIG)
    up_tpl[:sps] = up_ref[:sps].astype(SIG)[::-1]
    dn_tpl[:sps] = down_ref[:sps].astype(SIG)[::-1]
    up_env = np.abs(np.fft.irfft(F_buf * np.fft.rfft(up_tpl, nfft), nfft)[:n_audio])
    dn_env = np.abs(np.fft.irfft(F_buf * np.fft.rfft(dn_tpl, nfft), nfft)[:n_audio])
    combined = np.maximum(up_env, dn_env)

    sps_eff_d = max(1, sps_eff + int(round(rate_est_smp)))
    usable = (n_audio // sps_eff_d) * sps_eff_d
    if usable < sps_eff_d * 2:
        return
    folded    = combined[:usable].reshape(-1, sps_eff_d).sum(axis=0)
    coarse_t0 = int(np.argmax(folded))

    M = (n_audio - coarse_t0) // sps_eff_d   # total symbol slots in buffer
    if M < frame_symbols:
        return

    search_r = sps // 6
    half_w   = max(4, search_r // 4)
    k_all    = np.arange(frame_symbols, dtype=np.float64)

    # Refined peak position for every symbol slot (computed once).
    grid_peak = np.zeros(M, dtype=np.float64)
    for m in range(M):
        exp = coarse_t0 + m * sps_eff_d
        lo  = max(0, exp - search_r)
        hi  = min(n_audio, exp + search_r)
        if hi <= lo:
            grid_peak[m] = exp
            continue
        u_local = int(np.argmax(up_env[lo:hi]))
        d_local = int(np.argmax(dn_env[lo:hi]))
        grid_peak[m] = ((lo + u_local)
                        if up_env[lo + u_local] >= dn_env[lo + d_local]
                        else (lo + d_local))

    def _decode_window(m0):
        # IRLS line fit over this window's symbol peaks.
        raw     = grid_peak[m0:m0 + frame_symbols].astype(np.float64)
        mask    = np.ones(frame_symbols, dtype=bool)
        sps_fit = float(sps_eff_d)
        t0_fit  = float(raw[0])
        for _iter in range(6):
            k_in = k_all[mask]; p_in = raw[mask]
            if len(k_in) < 4:
                break
            A = np.vstack([k_in, np.ones(len(k_in))]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, p_in, rcond=None)
            sps_fit, t0_fit = float(coeffs[0]), float(coeffs[1])
            new_mask = np.abs(raw - (t0_fit + k_all * sps_fit)) < search_r
            if new_mask.sum() == mask.sum():
                break
            mask = new_mask
        n_out  = int((~mask).sum())
        fitted = np.round(t0_fit + k_all * sps_fit).astype(int)

        data, totals = [], []
        for k in range(frame_symbols):
            p  = int(fitted[k])
            lo = max(0, p - half_w)
            hi = min(n_audio, p + half_w)
            if hi <= lo:
                data.append((0.0, 0.0, 0.0)); totals.append(0.0); continue
            u = float(np.max(up_env[lo:hi]))
            d = float(np.max(dn_env[lo:hi]))
            data.append((u, d, u + d)); totals.append(u + d)
        _nz      = [t for t in totals if t > 1e-12]
        peak_ref = float(np.median(_nz)) if _nz else 1.0
        bits, confs = [], []
        for k in range(frame_symbols):
            u, d, total = data[k]
            if total < 1e-12:
                bits.append(0); confs.append(0.0); continue
            conf  = (u - d) / (total + 1e-12)
            snr_w = float(np.clip(total / (peak_ref + 1e-12), 0.0, 1.0))
            bits.append(1 if conf > 0 else 0)
            confs.append(conf * snr_w)
        return bits, confs, t0_fit, sps_fit, n_out

    m_base = M - frame_symbols          # shift 0 = aligned to buffer end
    shifts = [0]
    for j in range(1, max_shift + 1):
        shifts += [-j, +j]
    for j in shifts:
        m0 = m_base + j
        if m0 < 0 or m0 + frame_symbols > M:
            continue
        bits, confs, t0_fit, sps_fit, n_out = _decode_window(m0)
        yield bits, confs, t0_fit, sps_fit, n_out, j
