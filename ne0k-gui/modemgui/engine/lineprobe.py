# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Line probe, stage 3: a periodic multi-tone comb and its analyzer (plan: audiotestserver/tests/OFDM/LINE_PROBE.md).

Modelled on the V.34 line probing signals L1/L2 ("T-REC-V.34-199802-I!!PDF-E-2.pdf page 31", section 10.1.2.4 and
Table 17): cosines on a 150 Hz grid, repeating at the grid spacing, with some grid slots left empty, sent at two
levels (+6 dB and nominal).  V.34 is the guideline only: here the band, the spacing and the empty slots follow the
UI parameters (a very narrow band or a 20 kHz one), because an audio + radio path (Bluetooth codec, FM limiter,
SDR) has impairments a phone line does not.

The comb is built so that FFT bins cannot bleed into each other
-------------------------------------------------------------
* Period P = fs / spacing samples, and P must be an INTEGER at every sample rate the path touches (44100 at the
  BT transmitter, 48000 at the receiver, ...).  That holds exactly when spacing = g / n for an integer n, where
  g = gcd(rates): gcd(48000, 44100) = 300, so 150, 100, 75, 60, ... Hz.  ``plan_comb`` refuses anything else and
  says which spacings are valid.
* Tone k sits at k * spacing.  A window of exactly M*P samples holds a whole number of cycles of every tone, so
  the DFT puts tone k on bin k*M with ZERO leakage in exact arithmetic, whatever sample the window starts on (a
  start offset only rotates each tone's phase).  ``tests/test_lineprobe.py`` proves this to the float64 floor
  over a parameter matrix.
* With M > 1 the M-1 bins between neighbouring tones carry no signal: they measure noise and the phase-noise
  skirt beside every tone.  Empty grid slots (omitted tones) measure noise + harmonics + intermodulation UNDER
  LOAD -- with every tone on one grid, every distortion product of the comb lands on a grid slot, so the power in
  the empty ones is the signal-dependent impairment (the reason V.34 leaves 900/1200/1800/2400 Hz out).

What does leak on a real path, and how the analyzer handles it
--------------------------------------------------------------
A clock offset eps (TX and RX sample clocks differ) moves tone k to k*M*(1+eps) bins, off its bin, so it leaks
into its neighbours (a Dirichlet kernel).  ``analyze`` estimates eps from each tone's phase drift between
windows, reports the leakage it predicts, and measures the empty bins on a RESIDUAL: a least-squares fit of every
tone at its corrected frequency is subtracted window by window, so the residual holds noise and distortion, not
leaked tone.

The per-window fit also answers "is the SNR lower at high tones because of phase noise?": each tone's phase
wander from window to window is fitted to one common timing error tau_w (phase = 2*pi*f*tau_w).  A high
``timing_fraction`` means the wander is shared timing jitter, which grows with f and is fixed by pilot tracking,
not EQ.  A low one means independent noise.
"""
from __future__ import annotations

import dataclasses
import math
from functools import reduce

import numpy as np

# V.34 Table 17 ("T-REC-V.34-199802-I!!PDF-E-2.pdf page 31"): L1/L2 tone -> initial phase in degrees.  The tones
# at 900, 1200, 1800 and 2400 Hz are omitted.  A reference case for the tests, not a mode: pass it as ``phases_deg``.
V34_L1_PHASES_DEG = {150: 0, 300: 180, 450: 0, 600: 0, 750: 0, 1050: 0, 1350: 0, 1500: 0, 1650: 180, 1950: 0,
                     2100: 0, 2250: 180, 2550: 0, 2700: 180, 2850: 0, 3000: 180, 3150: 180, 3300: 180, 3450: 180,
                     3600: 0, 3750: 0}
V34_L1_OMIT_HZ = (900, 1200, 1800, 2400)
V34_L1_BOOST_DB = 6.0          # L1 is sent 6 dB above nominal; L2 at nominal (same page)

_CREST_OVERSAMPLE = 16         # the crest factor is measured on a 16x finer grid than the sample rate


@dataclasses.dataclass(frozen=True)
class CombPlan:
    spacing_hz: float
    rates: tuple[int, ...]
    periods: dict              # sample rate -> period P in samples (an integer by construction)
    k_lo: int                  # first and last grid slot in the band (slot k = k * spacing_hz)
    k_hi: int
    tones: np.ndarray          # slots that carry a tone
    omitted: np.ndarray        # slots left empty
    catches: dict              # omitted slot -> what lands there ("h2:k", "h3:k" for present tones k)
    phases: np.ndarray         # radians, one per tone
    crest: float               # peak / rms of the comb (continuous time)
    window_periods: int        # M: analysis window = M * P samples

    @property
    def tone_hz(self) -> np.ndarray:
        return self.tones * self.spacing_hz

    @property
    def omitted_hz(self) -> np.ndarray:
        return self.omitted * self.spacing_hz

    def tone_amplitude(self, peak: float) -> float:
        """Per-tone amplitude that puts the comb's peak at ``peak`` (sum of K cosines: rms = a*sqrt(K/2))."""
        return peak / (self.crest * math.sqrt(len(self.tones) / 2.0))

    def describe(self) -> str:
        return (f"comb {self.spacing_hz:g} Hz x {len(self.tones)} tones "
                f"{self.k_lo * self.spacing_hz:g}-{self.k_hi * self.spacing_hz:g} Hz, "
                f"{len(self.omitted)} empty slots, crest {20 * math.log10(self.crest):.1f} dB, "
                f"period " + ", ".join(f"{p} @ {fs}" for fs, p in self.periods.items()) +
                f", window {self.window_periods} period(s)")


def _gcd_rate(rates) -> int:
    return reduce(math.gcd, [int(r) for r in rates])


def valid_spacings(rates, near: float | None = None, count: int = 4) -> list[float]:
    """Grid spacings whose period is an integer at every rate in ``rates``: g/n for g = gcd(rates).
    With ``near``, the ``count`` closest to it; otherwise the largest ``count``."""
    g = _gcd_rate(rates)
    if near is None or near <= 0:
        return [g / n for n in range(1, count + 1)]
    n0 = max(1, round(g / near))
    cands = {g / n for n in range(max(1, n0 - count), n0 + count + 1)}
    return sorted(cands, key=lambda s: (abs(s - near), -s))[:count]


def _exact_n(rates, spacing_hz: float) -> int:
    """n with spacing = gcd(rates)/n, or raise naming the valid ones."""
    g = _gcd_rate(rates)
    n = g / float(spacing_hz)
    if spacing_hz <= 0 or abs(n - round(n)) > 1e-9 * max(1.0, n):
        raise ValueError(f"spacing {spacing_hz:g} Hz gives a non-integer period at "
                         f"{', '.join(str(int(r)) for r in rates)} Hz (FFT bins would bleed); valid nearby: "
                         + ", ".join(f"{s:g}" for s in valid_spacings(rates, spacing_hz)) + " Hz")
    return int(round(n))


def _newman(K: int) -> np.ndarray:
    """Newman phases pi*i^2/K: a low-crest set for K equal tones."""
    i = np.arange(K, dtype=np.float64)
    return np.pi * i * i / K


def _polish(k: np.ndarray, phases: np.ndarray, iters: int = 60) -> np.ndarray:
    """Lower the crest factor by iterative clipping (Van der Ouderaa et al.): clip the waveform's peaks, keep the
    tone phases of the clipped wave, repeat.  Deterministic, amplitudes untouched; keeps the best set seen.
    Newman phases assume a gap-free comb, and the omitted slots push them past 8 dB on a wide band."""
    n = max(64, _CREST_OVERSAMPLE * 2 * int(k.max()))
    idx = k.astype(np.int64)
    best, best_c = phases.copy(), _crest(k, phases)
    ph = phases.copy()
    for _ in range(iters):
        spec = np.zeros(n // 2 + 1, dtype=np.complex128)
        spec[idx] = np.exp(1j * ph)
        x = np.fft.irfft(spec, n)
        c = 0.8 * np.max(np.abs(x))
        y = np.clip(x, -c, c)
        ph = np.angle(np.fft.rfft(y)[idx])
        cr = _crest(k, ph)
        if cr < best_c:
            best, best_c = ph.copy(), cr
    return best


def _crest(k: np.ndarray, phases: np.ndarray) -> float:
    """Peak/rms of sum cos(2*pi*k*t + phi) over one period, sampled finely enough to catch the true peak."""
    n = max(64, _CREST_OVERSAMPLE * 2 * int(k.max()))
    t = np.arange(n) / n
    x = np.cos(2 * np.pi * np.outer(t, k) + phases).sum(axis=1)
    return float(np.max(np.abs(x)) / math.sqrt(len(k) / 2.0))


def plan_comb(freq_lo: float, freq_hi: float, rates=(48000,), spacing_hz: float = 150.0,
              window_periods: int = 1, omit_every: int = 8, harmonic_refs: int = 2,
              omit_hz=None, phases_deg: dict | None = None) -> CombPlan:
    """Lay the comb out from the UI's band and the path's sample rates.

    ``omit_hz`` (explicit empty slots) replaces the automatic choice; otherwise the slots holding the 2nd and 3rd
    harmonics of the ``harmonic_refs`` lowest tones are left empty, plus one slot in every ``omit_every`` so the
    whole band has empty bins.  ``phases_deg`` ({Hz: degrees}, e.g. V34_L1_PHASES_DEG) replaces Newman phases.
    """
    rates = tuple(int(r) for r in rates)
    if not rates or min(rates) <= 0:
        raise ValueError("need at least one positive sample rate")
    if int(window_periods) < 1:
        raise ValueError("window_periods must be >= 1")
    n = _exact_n(rates, spacing_hz)
    g = _gcd_rate(rates)
    spacing = g / n
    periods = {fs: fs // g * n for fs in rates}            # fs/spacing, exact integer
    # DC and Nyquist excluded (a tone there is not a cosine with a free phase); the lowest rate binds.
    k_max = min(periods.values()) // 2 - 1
    k_lo = max(1, math.ceil(freq_lo / spacing - 1e-9))
    k_hi = min(k_max, math.floor(freq_hi / spacing + 1e-9))
    if k_hi < k_lo:
        raise ValueError(f"no {spacing:g} Hz grid slot in {freq_lo:g}-{freq_hi:g} Hz (below Nyquist "
                         f"{min(rates) / 2:g} Hz); widen the band or use a finer spacing")
    slots = np.arange(k_lo, k_hi + 1)

    if omit_hz is not None:
        omit = set()
        for f in omit_hz:
            k = f / spacing
            if abs(k - round(k)) > 1e-9 or not (k_lo <= round(k) <= k_hi):
                raise ValueError(f"omitted tone {f:g} Hz is not a grid slot in the band")
            omit.add(int(round(k)))
    else:
        omit = set()
        refs = [int(k) for k in slots[:max(0, int(harmonic_refs))]]
        for r in refs:
            for h in (2, 3):
                if k_lo <= h * r <= k_hi and h * r not in refs:
                    omit.add(h * r)
        if omit_every and omit_every > 1 and len(slots) > 2:
            run = 0
            for k in slots[1:-1]:                          # keep the band edges as tones
                if int(k) in omit:
                    run = 0
                elif run >= omit_every - 1 and int(k) not in refs:
                    omit.add(int(k)); run = 0
                else:
                    run += 1
    tones = np.array([k for k in slots if int(k) not in omit], dtype=np.int64)
    omitted = np.array(sorted(omit), dtype=np.int64)
    if len(tones) == 0:
        raise ValueError("every slot in the band is omitted: no tone left")
    if len(omitted) == 0 and int(window_periods) < 2:
        raise ValueError("no empty bin to measure noise in: widen the band (so a slot can be omitted) or set "
                         "window_periods >= 2 (bins between the tones)")

    present = set(int(k) for k in tones)
    catches = {}
    for j in omitted:
        who = [f"h{h}:{j // h}" for h in (2, 3) if j % h == 0 and j // h in present]
        catches[int(j)] = who

    if phases_deg is not None:
        try:
            phases = np.array([math.radians(phases_deg[k * spacing]) for k in tones])
        except KeyError as ex:
            raise ValueError(f"phases_deg has no entry for tone {ex.args[0]:g} Hz") from None
    else:
        phases = _polish(tones.astype(np.float64), _newman(len(tones)))
    crest = _crest(tones.astype(np.float64), phases)
    return CombPlan(spacing, rates, periods, int(k_lo), int(k_hi), tones, omitted, catches, phases, crest,
                    int(window_periods))


def plan_for_grid(g, every: int = 1, window_periods: int | None = None, **kw) -> CombPlan:
    """The comb laid ON the OFDM carriers of grid ``g`` (engine/ofdmgrid.py), so the probe measures exactly the
    channels the data uses.

    Comb slot k is OFDM bin k*every: spacing = fs*every/n_fft, period P = n_fft/every samples (``every`` a power of
    two dividing n_fft, so P is an integer by construction).  The default window is ``every`` periods = n_fft
    samples: the modem's own FFT, in which each tone lands on its carrier's bin.  The band is the grid's snapped
    band.  Other keywords go to ``plan_comb`` (omit_every, harmonic_refs, omit_hz, phases_deg)."""
    every = int(every)
    if every < 1 or every & (every - 1) or g.n_fft % every or g.n_fft // every < 4:
        raise ValueError(f"every={every}: must be a power of two dividing n_fft={g.n_fft} (period >= 4 samples)")
    lo = math.ceil(g.bin_lo / every) * every * g.spacing_hz          # first comb slot inside the band
    return plan_comb(lo, g.freq_hi, rates=(g.fs,), spacing_hz=g.spacing_hz * every,
                     window_periods=every if window_periods is None else int(window_periods), **kw)


def synth(plan: CombPlan, fs: int, n_periods: int, peak: float = 0.5, freq_scale: float = 1.0,
          jitter_s: np.ndarray | None = None) -> np.ndarray:
    """``n_periods`` periods of the comb at ``fs``, peak at ``peak``.

    ``freq_scale`` = 1 + eps models a clock offset; ``jitter_s`` (one value per sample) a timing error, both
    for tests.  With neither, the phase is computed with integer arithmetic (k*n mod P), so the waveform is
    periodic to the last bit.
    """
    P = plan.periods[int(fs)]
    N = int(n_periods) * P
    a = plan.tone_amplitude(peak)
    nn = np.arange(N, dtype=np.int64)
    x = np.zeros(N)
    for k, ph in zip(plan.tones, plan.phases):
        if freq_scale == 1.0 and jitter_s is None:
            arg = 2 * np.pi * ((int(k) * nn) % P) / P
        else:
            t = nn / fs if jitter_s is None else nn / fs + jitter_s
            arg = 2 * np.pi * (k * plan.spacing_hz * freq_scale) * t
        x += a * np.cos(arg + ph)
    return x


def _dirichlet(d: np.ndarray, L: int) -> np.ndarray:
    """|response| of an L-point DFT bin to a unit tone ``d`` bins away (d need not be an integer)."""
    d = np.asarray(d, dtype=np.float64)
    out = np.ones_like(d)
    nz = np.abs(np.sin(np.pi * d / L)) > 1e-15
    out[nz] = np.abs(np.sin(np.pi * d[nz]) / (L * np.sin(np.pi * d[nz] / L)))
    return out


def _db(p) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(p, dtype=np.float64), 1e-300))


def analyze(plan: CombPlan, x: np.ndarray, fs: int, start: int = 0) -> dict:
    """Measure a received comb.  ``x[start:]`` must be steady-state comb (stage segmentation is the caller's).

    Powers are amplitude^2 of a cosine (a tone of amplitude a reads a^2); ``*_dbc`` are relative to the mean
    tone power.  Returns per tone: amplitude, phase, SNR from the spread between windows; per empty slot and
    between-tone gap: power on the residual; clock offset; predicted leakage; timing-jitter verdict.
    """
    P = plan.periods[int(fs)]
    M = plan.window_periods
    L = M * P
    x = np.asarray(x, dtype=np.float64)[int(start):]
    W = len(x) // L
    if W < 3:
        raise ValueError(f"need at least 3 windows of {L} samples, got {len(x)} samples")
    xw = x[:W * L].reshape(W, L)
    tb = plan.tones * M                                    # tone bins in an L-point DFT
    eb = plan.omitted * M                                  # empty-slot bins
    lo_b, hi_b = plan.k_lo * M, plan.k_hi * M
    gb = np.array([b for b in range(lo_b, hi_b + 1) if b % M], dtype=np.int64)   # between-tone bins

    X = np.fft.rfft(xw, axis=1) * (2.0 / L)                # a cosine of amplitude a reads |X| = a
    A = X[:, tb]                                           # (W, K) raw tone phasors per window

    # -- clock offset: tone k advances 2*pi*k*M*eps per window --
    inc = np.sum(A[1:] * np.conj(A[:-1]), axis=0)
    theta = np.angle(inc)
    wgt = np.abs(inc)
    kf = plan.tones.astype(np.float64)
    eps = float(np.sum(wgt * kf * theta) / (2 * np.pi * M * np.sum(wgt * kf * kf) + 1e-300))

    # -- per-window least-squares fit at the corrected frequencies, absolute time base --
    f_corr = plan.tone_hz * (1.0 + eps)
    n_abs = np.arange(W * L, dtype=np.float64).reshape(W, L)
    C = np.zeros((W, len(plan.tones)), dtype=np.complex128)
    R = np.zeros_like(xw)
    for w in range(W):
        arg = 2 * np.pi * np.outer(n_abs[w] / fs, f_corr)
        B = np.concatenate([np.cos(arg), -np.sin(arg)], axis=1)
        coef, *_ = np.linalg.lstsq(B, xw[w], rcond=None)
        K = len(plan.tones)
        C[w] = coef[:K] + 1j * coef[K:]
        R[w] = xw[w] - B @ coef
    Rf = np.fft.rfft(R, axis=1) * (2.0 / L)

    Cm = C.mean(axis=0)
    tone_pow = np.abs(Cm) ** 2
    ref = float(np.mean(tone_pow))
    spread = np.var(C.real, axis=0, ddof=1) + np.var(C.imag, axis=0, ddof=1)
    tone_snr = tone_pow / np.maximum(spread, 1e-300)

    # -- timing jitter: per-window phase wander d[w,k] fitted to 2*pi*f_k*tau_w --
    d = np.angle(C * np.conj(Cm))                          # (W, K)
    fk = plan.tone_hz
    wk = tone_pow / np.maximum(spread, 1e-300)             # trust strong, steady tones more
    tau = (d * (wk * fk)).sum(axis=1) / (2 * np.pi * np.sum(wk * fk * fk) + 1e-300)
    # A clock that DRIFTS (eps changing over the capture: BT resampler, crystal warming) puts a smooth curve in
    # tau that a constant eps cannot remove.  Pilot / timing tracking follows it, so it is reported apart
    # (timing_drift_us) and the jitter verdict is taken on what is left.
    if W >= 4:
        iw = np.arange(W, dtype=np.float64)
        trend = np.polyval(np.polyfit(iw, tau, 2), iw)
        tau, d = tau - trend, d - 2 * np.pi * np.outer(trend, fk)
        d = np.angle(np.exp(1j * d))
    else:
        trend = np.zeros_like(tau)
    model = 2 * np.pi * np.outer(tau, fk)
    tot = float(np.sum(wk * d * d))
    timing_fraction = (1.0 - float(np.sum(wk * (d - model) ** 2)) / tot) if tot > 0 else 0.0

    # -- predicted leakage from the uncorrected clock offset (coherent upper bound per empty bin) --
    a_k = np.sqrt(tone_pow)
    def _leak(bins):
        if len(bins) == 0:
            return np.zeros(0)
        pos = tb * (1.0 + eps)
        return np.array([np.sum(a_k * _dirichlet(b - pos, L)) for b in bins]) ** 2

    def _pw(F, bins):
        return np.mean(np.abs(F[:, bins]) ** 2, axis=0) if len(bins) else np.zeros(0)

    empty_pow = _pw(Rf, eb)
    gap_pow = _pw(Rf, gb)
    return {
        "fs": int(fs), "window_samples": L, "windows": W, "bin_hz": fs / L,
        "tone_hz": fk, "tone_db": _db(tone_pow), "tone_phase_deg": np.degrees(np.angle(Cm)),
        "tone_snr_db": _db(tone_snr),
        "empty_hz": plan.omitted_hz, "empty_dbc": _db(empty_pow / ref),
        "empty_raw_dbc": _db(_pw(X, eb) / ref),            # before the fit: includes clock-offset leakage
        "empty_leak_bound_dbc": _db(_leak(eb) / ref),
        "catches": plan.catches,
        "gap_hz": gb * fs / L, "gap_dbc": _db(gap_pow / ref),
        "floor_dbc": float(_db(np.median(np.concatenate([empty_pow, gap_pow])) / ref))
        if len(empty_pow) + len(gap_pow) else None,
        "clock_ppm": eps * 1e6,
        "timing_jitter_us": float(np.std(tau) * 1e6), "timing_fraction": timing_fraction,
        "timing_drift_us": float(np.ptp(trend) * 1e6),
    }


def steady_span(plan: CombPlan, x: np.ndarray, fs: int, trim_s: float = 0.25) -> tuple[int, int]:
    """(start, stop) sample indices of the received comb inside a longer recording (silence, comb, silence).

    The comb is the longest run of analysis windows that are within 10 dB of the loudest windows AND repeat: the
    comb is periodic in the window, so a window correlates with the next one near 1, where noise, speech or a
    different comb do not (> 0.5 required).  ``trim_s`` comes off each end so the AGC / codec / limiter settle and
    the tail is not cut.  Raises ValueError if fewer than 3 windows of steady comb remain.
    """
    L = plan.window_periods * plan.periods[int(fs)]
    x = np.asarray(x, dtype=np.float64)
    W = len(x) // L
    if W < 3:
        raise ValueError(f"recording too short: {len(x)} samples, window {L}")
    xw = x[:W * L].reshape(W, L)
    e = _db(np.mean(xw ** 2, axis=1))
    xc = xw - xw.mean(axis=1, keepdims=True)
    nrm = np.sqrt(np.sum(xc ** 2, axis=1)) + 1e-300
    rho = np.sum(xc[1:] * xc[:-1], axis=1) / (nrm[1:] * nrm[:-1])
    rep = np.zeros(W, bool)
    rep[:-1] |= rho > 0.5
    rep[1:] |= rho > 0.5
    on = (e > np.percentile(e, 95) - 10.0) & rep
    best, run_start, best_span = 0, None, (0, 0)
    for i, v in enumerate(np.append(on, False)):
        if v and run_start is None:
            run_start = i
        elif not v and run_start is not None:
            if i - run_start > best:
                best, best_span = i - run_start, (run_start, i)
            run_start = None
    trim = int(math.ceil(trim_s * fs / L))
    a, b = best_span[0] + trim, best_span[1] - trim
    if b - a < 3:
        raise ValueError(f"no steady comb: loudest run is {best} windows of {L} samples ({best * L / fs:.2f} s)")
    return a * L, b * L


def report_rows(plan: CombPlan, res: dict, prefix: str = "lp") -> list[tuple[str, str]]:
    """``analyze`` output as control-port key/value rows: a summary, then one row per tone (level relative to the
    loudest tone, SNR, phase, and the noise in the gap bins just above it), then one per empty slot."""
    tdb = res["tone_db"]
    top = float(np.max(tdb))
    rows = [(f"{prefix}.plan", plan.describe()),
            (f"{prefix}.windows", f"{res['windows']} x {res['window_samples']} samples ({res['bin_hz']:g} Hz bins)"),
            (f"{prefix}.clock_ppm", f"{res['clock_ppm']:.2f}"),
            (f"{prefix}.timing_jitter_us", f"{res['timing_jitter_us']:.2f}"),
            (f"{prefix}.timing_fraction", f"{res['timing_fraction']:.2f}"),
            (f"{prefix}.timing_drift_us", f"{res['timing_drift_us']:.1f}"),
            (f"{prefix}.floor_dbc", "-" if res["floor_dbc"] is None else f"{res['floor_dbc']:.1f}"),
            (f"{prefix}.tone_level_dbfs_max", f"{top - 3.01:.1f}"),       # amplitude^2 -> RMS dBFS of a sine
            (f"{prefix}.snr_db_median", f"{float(np.median(res['tone_snr_db'])):.1f}")]
    if len(res["empty_dbc"]):
        rows.append((f"{prefix}.empty_dbc_median", f"{float(np.median(res['empty_dbc'])):.1f}"))
    gap_hz, gap_dbc = np.asarray(res["gap_hz"]), np.asarray(res["gap_dbc"])
    for f, d, s, ph in zip(res["tone_hz"], tdb, res["tone_snr_db"], res["tone_phase_deg"]):
        near = (gap_hz > f) & (gap_hz < f + plan.spacing_hz)
        gap = f" gap={float(np.mean(gap_dbc[near])):.1f}" if near.any() else ""
        rows.append((f"{prefix}.tone.{f:g}", f"rel={d - top:.1f} snr={s:.1f} ph={ph:.0f}{gap}"))
    for f, d, raw in zip(res["empty_hz"], res["empty_dbc"], res["empty_raw_dbc"]):
        rows.append((f"{prefix}.empty.{f:g}", f"dbc={d:.1f} raw={raw:.1f}"))
    return rows
