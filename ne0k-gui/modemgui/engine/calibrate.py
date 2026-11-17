# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Acoustic calibration: measure the speaker -> air -> mic channel and say what TX level, input gain and OFDM
echo budget (cp_ratio / n_fft) it wants.

One probe signal does everything (``make_probe``)::

    [sync chirp] [level steps: multitone at -42 .. -6 dBFS peak] [AGC test: quiet / +30 dB / quiet]
    [log sweeps at -20 / -10 / -2 dBFS] [sync chirp]

It is played either into the same host's mic (``SELF``: own speaker, mic beside it) or by a peer while this host
listens (``LISTEN`` / ``SEND``).  ``analyse`` takes the recording plus a stretch of noise recorded before it:

* **Alignment** -- the first sync chirp by cross-correlation; the second one's spacing gives the two sound cards'
  sample-clock offset in ppm (the source of OFDM ICI between two machines).
* **Levels** -- per step: tone SNR against the noise floor at the same bins, SINAD against everything between the
  tones (noise + distortion + intermodulation), RX peak / clipping, gain.  Gain falling while SINAD stays high is
  an AGC; SINAD falling is speaker / mic distortion.  The best TX level is the highest step before SINAD stops
  improving by >= 1 dB or the mic clips.
* **AGC** -- the level of a steady tone just after a +30 dB jump vs at its end, and how long the quiet tone after
  the loud part takes to come back (an AGC that pumped down on our own TX deafens us to the peer's reply).
* **Compression vs frequency** -- a speaker (and an FM radio's pre-emphasis / deviation limiter) does not compress
  evenly: lows hit cone excursion first, highs the limiter.  Per third-octave band:

  - from the level steps (OFDM-like signal, power spread over all carriers): each tone's gain against its own
    small-signal gain, so a band whose gain sags as the level rises is compressing.  ``safe`` = highest step within
    1 dB; ``mask`` = how far below the best TX level that band should run (a TX power mask, not an equaliser:
    compression in one band costs every band, since its intermodulation lands across the whole spectrum).
  - from the sweeps (constant envelope, ALL the power on one frequency at a time -- what a CSS chirp does): the
    linear transfer at each level against the quietest, and the harmonic distortion the log sweep separates out
    ahead of the linear response.  CSS is limited by the worst band inside its own band, so ``css_amplitude`` is the
    lower of the equal-RMS rule and the highest sweep level that stays within 1 dB and below -20 dB THD there.
* **Echo** -- the impulse response from the -10 dBFS sweep (regularised spectral division), its energy decay, and for each
  (n_fft, cp_ratio) the ISI ceiling ``10 log(E inside CP / E after)`` combined with the ICI ceiling from the clock
  offset; the recommendation is the most efficient pair whose ceiling clears ``target_sir_db``.

The multitone has random phases, so its crest factor is OFDM-like (~10 dB) and a step's peak maps straight to
``ofdm.amplitude``.  CSS chirps are constant-envelope (crest 3 dB), so the matching CSS level is the one with the
same RMS -- about a third of the OFDM peak (the 0.2 vs 0.6 that works on the test laptops).

No Qt and no scipy: numpy only, testable on synthetic rooms (tests/test_calibrate.py).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

N_FFT_CHOICES = (512, 1024, 2048, 4096)
CP_CHOICES = (0.25, 0.375, 0.5)

STEP_LEVELS_DB = (-42.0, -36.0, -30.0, -24.0, -18.0, -12.0, -6.0)   # multitone peak, dBFS
STEP_S, STEP_GAP_S = 0.6, 0.15
AGC_QUIET_DB, AGC_LOUD_DB = -40.0, -10.0
AGC_PART_S = 1.0
SYNC_S, GAP_S = 0.3, 0.3
SWEEP_S = 2.5
SWEEP_GAP_S = 0.5                             # > the 0.4 s of ring-out each sweep's analysis window keeps
SWEEP_LEVELS_DB = (-20.0, -10.0, -2.0)        # peak = a CSS chirp's peak at that amplitude
ECHO_SWEEP = 1                                # which sweep the echo / ISI analysis uses
COMP_DB = 1.0                                 # "compressing": gain this far below the small-signal gain
THD_MAX_DB = -20.0
SYNC_AMP_DB = -15.0
TAIL_S = 0.6


def _db(x: float) -> float:
    return 10 * math.log10(x) if x > 1e-20 else -200.0


def _a(db: float) -> float:
    return 10 ** (db / 20)


def _fade(x: np.ndarray, fs: int, s: float = 0.01) -> np.ndarray:
    n = min(int(s * fs), len(x) // 2)
    if n:
        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        x[:n] *= w
        x[-n:] *= w[::-1]
    return x


# -- signals ----------------------------------------------------------------------------------------------------------

def sync_chirp(fs: int, f0: float, f1: float, amp: float) -> np.ndarray:
    t = np.arange(int(SYNC_S * fs)) / fs
    x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * SYNC_S) * t * t))
    return _fade(amp * x, fs)


def tone_grid(fs: int, f0: float, f1: float, n_seg: int, fill: float = 0.4,
              seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """(tone bins, empty bins) of an ``n_seg`` FFT: a grid of every other bin across the band, ``fill`` of it carrying
    tones at random slots, the rest left empty (a noise-power-ratio layout).  The empty slots must be ON the grid:
    odd-order intermodulation of tones on an arithmetic grid lands back on that grid, so empty bins off it (between
    evenly spaced tones) would never see the speaker's distortion."""
    df = fs / n_seg
    lo, hi = int(math.ceil(f0 / df)) + 1, int(f1 / df) - 1
    grid = np.arange(lo, hi + 1, 2)
    rng = np.random.default_rng(seed)
    pick = np.zeros(len(grid), bool)
    pick[rng.choice(len(grid), max(2, int(fill * len(grid))), replace=False)] = True
    return grid[pick], grid[~pick]


def multitone(fs: int, bins: np.ndarray, n_seg: int, secs: float, peak: float, seed: int = 7) -> np.ndarray:
    """Tones exactly on ``bins`` of an ``n_seg``-point FFT (periodic in n_seg samples), random phases, scaled so the
    peak is ``peak``."""
    rng = np.random.default_rng(seed)
    spec = np.zeros(n_seg // 2 + 1, complex)
    spec[bins] = np.exp(2j * np.pi * rng.random(len(bins)))
    one = np.fft.irfft(spec, n_seg)
    one /= np.max(np.abs(one))
    reps = int(math.ceil(secs * fs / n_seg))
    return _fade(peak * np.tile(one, reps)[:int(secs * fs)].copy(), fs, 0.005)


def log_sweep(fs: int, f0: float, f1: float, secs: float, amp: float) -> np.ndarray:
    t = np.arange(int(secs * fs)) / fs
    r = math.log(f1 / f0)
    x = np.sin(2 * np.pi * f0 * secs / r * (np.exp(t * r / secs) - 1))
    return _fade(amp * x, fs, 0.02)


@dataclasses.dataclass
class Probe:
    signal: np.ndarray
    fs: int
    band: tuple[float, float]
    n_seg: int
    bins: np.ndarray
    empty: np.ndarray                            # grid slots with no tone: noise + distortion
    sync: np.ndarray
    steps: list[tuple[int, int, float]]          # (start, end, peak dBFS)
    agc: list[tuple[int, int, float]]            # quiet, loud, quiet
    sweeps: list[tuple[int, int, float]]         # (start, end, peak dBFS)
    sweep_sig: np.ndarray                        # unit-peak sweep
    sweep_range: tuple[float, float]
    sync2: int                                   # start of the closing sync chirp
    css_band: tuple[float, float] | None = None


def make_probe(fs: int = 48000, f0: float = 400.0, f1: float = 4000.0,
               css_band: tuple[float, float] | None = None) -> Probe:
    """``f0``-``f1``: the OFDM band (level steps, AGC, mask).  The sweeps cover it and ``css_band`` with margin."""
    n_seg = 4096 if fs <= 48000 else 8192
    bins, empty = tone_grid(fs, f0, f1, n_seg)
    sync = sync_chirp(fs, f0, f1, _a(SYNC_AMP_DB))
    parts, steps, agc = [sync, np.zeros(int(GAP_S * fs))], [], []
    pos = lambda: sum(len(p) for p in parts)                                # noqa: E731
    for lv in STEP_LEVELS_DB:
        s = pos()
        parts.append(multitone(fs, bins, n_seg, STEP_S, _a(lv)))
        steps.append((s, pos(), lv))
        parts.append(np.zeros(int(STEP_GAP_S * fs)))
    for lv in (AGC_QUIET_DB, AGC_LOUD_DB, AGC_QUIET_DB):
        s = pos()
        parts.append(multitone(fs, bins, n_seg, AGC_PART_S, _a(lv), seed=11))
        agc.append((s, pos(), lv))
    parts.append(np.zeros(int(GAP_S * fs)))
    lo = min(f0, css_band[0]) if css_band else f0
    hi = max(f1, css_band[1]) if css_band else f1
    rng = (max(lo * 0.8, 50.0), min(hi * 1.15, fs / 2 * 0.95))
    sw = log_sweep(fs, rng[0], rng[1], SWEEP_S, 1.0)
    sweeps = []
    for lv in SWEEP_LEVELS_DB:
        s = pos()
        parts.append(sw * _a(lv))
        sweeps.append((s, pos(), lv))
        parts.append(np.zeros(int(SWEEP_GAP_S * fs)))
    sync2 = pos()
    parts += [sync, np.zeros(int(TAIL_S * fs))]
    return Probe(np.concatenate(parts).astype(np.float32), fs, (f0, f1), n_seg, bins, empty, sync, steps, agc, sweeps,
                 sw, rng, sync2, css_band)


# -- analysis helpers -----------------------------------------------------------------------------------------------

def xcorr(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """|cross-correlation| of ``ref`` against every lag of ``x`` that holds all of it (FFT)."""
    n = 1 << int(math.ceil(math.log2(len(x) + len(ref))))
    return np.abs(np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(ref, n)), n)[:len(x) - len(ref) + 1])


def xcorr_peak(x: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """(sub-sample lag of ``ref`` in ``x``, peak / median ratio in dB)."""
    a = xcorr(x, ref)
    k = int(np.argmax(a))
    return _frac_peak(a, k), 20 * math.log10(a[k] / (np.median(a) + 1e-12) + 1e-12)


def _frac_peak(a: np.ndarray, k: int) -> float:
    """Parabolic sub-sample peak position around index k."""
    if 0 < k < len(a) - 1:
        y0, y1, y2 = a[k - 1], a[k], a[k + 1]
        d = y0 - 2 * y1 + y2
        if d != 0:
            return k + 0.5 * (y0 - y2) / d
    return float(k)


def seg_spectrum(x: np.ndarray, n_seg: int) -> np.ndarray:
    """Mean power spectrum over whole n_seg frames (rectangular: the tones are bin-exact and periodic)."""
    m = len(x) // n_seg
    if m == 0:
        return np.zeros(n_seg // 2 + 1)
    f = np.fft.rfft(x[:m * n_seg].reshape(m, n_seg), axis=1)
    return np.mean(np.abs(f) ** 2, axis=0) / n_seg ** 2


def impulse_response(rec: np.ndarray, sweep: np.ndarray, fs: int, band: tuple[float, float],
                     full: bool = False) -> np.ndarray:
    """IR by regularised spectral division Y X* / (|X|^2 + eps), limited to the band.  ``rec`` starts a little
    before the sweep arrived; the IR's delay is the arrival offset."""
    n = 1 << int(math.ceil(math.log2(len(rec) + len(sweep))))
    X, Y = np.fft.rfft(sweep, n), np.fft.rfft(rec, n)
    p = np.abs(X) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    inb = (f >= band[0]) & (f <= band[1])
    eps = 1e-3 * np.max(p[inb]) if inb.any() else 1e-9
    H = np.where(inb, Y * np.conj(X) / (p + eps), 0)
    # a Tukey edge on the band so the IR does not ring at the band edges
    w = np.zeros_like(f)
    w[inb] = 1.0
    k = int(0.05 * inb.sum())
    if k > 1:
        idx = np.flatnonzero(inb)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
        w[idx[:k]] = ramp
        w[idx[-k:]] = ramp[::-1]
    ir = np.fft.irfft(H * w, n)
    return ir if full else ir[:len(rec)]          # full: negative times (log-sweep harmonics) wrap to the end


def third_octaves(lo: float, hi: float) -> list[tuple[float, float, float]]:
    """(centre, low edge, high edge) of the standard third-octave bands whose centre lies in lo..hi."""
    out = []
    for k in range(-20, 20):
        fc = 1000.0 * 2 ** (k / 3)
        if lo <= fc <= hi:
            out.append((fc, fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6)))
    return out


def _fc(fc: float) -> str:
    return f"{fc:.0f}" if fc < 1000 else f"{fc / 1000:.1f}k".replace(".0k", "k")


@dataclasses.dataclass
class SweepStats:
    gain_db: dict                  # band centre -> {level: linear transfer dB}
    thd_db: dict                   # level -> harmonic (2nd + 3rd) / linear energy, dB


def sweep_stats(rec: np.ndarray, probe: "Probe", o: int, bands: list) -> SweepStats:
    fs = probe.fs
    f1, f2 = probe.sweep_range
    r = math.log(f2 / f1)
    gain: dict = {b[0]: {} for b in bands}
    thd = {}
    for (s, e, lv) in probe.sweeps:
        a0 = max(0, o + s - int(0.05 * fs))
        seg = rec[a0:o + e + int(0.4 * fs)]
        ir = impulse_response(seg, probe.sweep_sig * _a(lv), fs, probe.sweep_range, full=True)
        n = len(ir)
        pk = int(np.argmax(np.abs(ir[:len(seg)])))
        lin = ir[max(0, pk - int(0.002 * fs)):pk + int(0.1 * fs)]
        nf = 1 << int(math.ceil(math.log2(max(len(lin), fs // 10))))
        H = np.abs(np.fft.rfft(lin, nf)) ** 2
        f = np.fft.rfftfreq(nf, 1 / fs)
        for fc, a, b in bands:
            m = (f >= a) & (f < b)
            if m.any():
                gain[fc][lv] = _db(float(H[m].mean()))
        e1 = float((lin ** 2).sum()) + 1e-30
        dts = [SWEEP_S * math.log(k) / r for k in (2, 3, 4)]
        eh = 0.0
        for k in (0, 1):
            w = int(min(0.02, (dts[k + 1] - dts[k]) / 2) * fs)
            c = (pk - int(round(dts[k] * fs))) % n
            idx = np.arange(c - w // 4, c + w) % n
            eh += float((ir[idx] ** 2).sum())
        thd[lv] = _db(eh / e1)
    return SweepStats(gain, thd)


@dataclasses.dataclass
class EchoStats:
    delay_s: float                 # arrival of the direct sound within the IR window
    rms_spread_ms: float
    t20_ms: float                  # after the onset, until the remaining energy is 20 dB below the total
    t30_ms: float
    rt60_s: float | None           # Schroeder T20 extrapolated
    pre_echo_db: float             # energy before the onset vs total (a quality check: high = bad measurement)
    ceilings: dict                 # (n_fft, cp_ratio) -> (isi_db, ici_db, total_db)


def echo_stats(ir: np.ndarray, fs: int, ppm: float, f_hi: float) -> EchoStats:
    e = ir.astype(np.float64) ** 2
    pk = int(np.argmax(e))
    thr = e[pk] * 0.01                                       # onset: first tap within 20 dB of the peak
    on = pk
    look = max(0, pk - int(0.003 * fs))
    above = np.flatnonzero(e[look:pk + 1] >= thr)
    if len(above):
        on = look + int(above[0])
    on = max(0, on - int(0.0005 * fs))                       # half a millisecond of timing advance, as a modem does
    tail = e[on:on + int(1.0 * fs)]
    tot = float(tail.sum()) + 1e-30
    pre = float(e[max(0, on - int(0.05 * fs)):on].sum())
    sch = np.cumsum(tail[::-1])[::-1] / tot                  # remaining energy from t on
    t = np.arange(len(tail)) / fs
    t20 = float(t[np.argmax(sch < 0.01)] * 1000) if (sch < 0.01).any() else float(t[-1] * 1000)
    t30 = float(t[np.argmax(sch < 0.001)] * 1000) if (sch < 0.001).any() else float(t[-1] * 1000)
    mean = float((t * tail).sum() / tot)
    spread = float(math.sqrt(max(0.0, (((t - mean) ** 2) * tail).sum() / tot)) * 1000)
    rt60 = None
    s_db = 10 * np.log10(sch + 1e-30)
    i5, i25 = np.argmax(s_db < -5), np.argmax(s_db < -25)
    if i25 > i5 > 0:
        slope = (s_db[i25] - s_db[i5]) / (t[i25] - t[i5])
        if slope < 0:
            rt60 = float(-60 / slope)
    ceilings = {}
    for n in N_FFT_CHOICES:
        df = fs / n
        eps = abs(f_hi * ppm * 1e-6) / df                    # worst carrier's offset in carrier spacings
        ici = _db(3 / (math.pi * eps) ** 2) if eps > 0 else 200.0
        for cp in CP_CHOICES:
            L = int(cp * n)
            inside, outside = float(tail[:L].sum()), float(tail[L:].sum())
            isi = _db(inside / outside) if outside > 0 else 200.0
            total = -_db(10 ** (-isi / 10) + 10 ** (-ici / 10))
            ceilings[(n, cp)] = (round(isi, 1), round(min(ici, 200.0), 1), round(total, 1))
    return EchoStats(on / fs, spread, t20, t30, rt60, _db(pre / tot) if pre > 0 else -200.0, ceilings)


def recommend_ofdm(ceilings: dict, target_sir_db: float) -> tuple[int, float, float, str]:
    """(n_fft, cp_ratio, ceiling dB, why): the least CP overhead whose ceiling clears the target, then the
    smaller FFT (less ICI, shorter frames).  None clears it: the pair with the highest ceiling."""
    ok = [(cp, n) for (n, cp), (_, _, tot) in ceilings.items() if tot >= target_sir_db]
    if ok:
        cp, n = min(ok)
        return n, cp, ceilings[(n, cp)][2], f"least overhead with ceiling >= {target_sir_db:g} dB"
    (n, cp), v = max(ceilings.items(), key=lambda kv: kv[1][2])
    return n, cp, v[2], f"no pair reaches {target_sir_db:g} dB; best available"


# -- the analysis ----------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class StepResult:
    level_db: float
    snr_db: float                 # tone power vs the noise floor at the same bins (median over tones)
    sinad_db: float               # tone power vs everything between the tones
    gain_db: float                # RX tone level - TX tone level
    rx_peak_db: float
    clipped: int
    settle_db: float              # level change from the first 100 ms to the last 300 ms (AGC acting)


@dataclasses.dataclass
class CalResult:
    ok: bool
    error: str = ""
    floor_rms_db: float = -200.0
    floor_band_db: float = -200.0
    sync_db: float = 0.0
    ppm: float = 0.0
    steps: list[StepResult] = dataclasses.field(default_factory=list)
    best_level_db: float | None = None
    best_reason: str = ""
    agc_detected: bool = False
    agc_attack_db: float = 0.0
    agc_recovery_s: float | None = None
    echo: EchoStats | None = None
    rec_n_fft: int | None = None
    rec_cp: float | None = None
    rec_ceiling_db: float | None = None
    rec_reason: str = ""
    cur_ceiling: tuple | None = None
    in_gain_change_db: float | None = None
    verdict: str = ""
    bands: list = dataclasses.field(default_factory=list)    # per OFDM third-octave: dict (see analyse)
    sweep: SweepStats | None = None
    rx_gate_db: float | None = None                            # extra noise in the empty bins while the probe plays
    css_limit_db: float | None = None                         # highest clean sweep level in the CSS band
    css_reason: str = ""

    def rows(self, prefix: str = "cal", cur: tuple[int, float] | None = None) -> list[tuple[str, str]]:
        r = [(f"{prefix}.ok", "yes" if self.ok else "no")]
        if self.error:
            r.append((f"{prefix}.error", self.error))
        r += [(f"{prefix}.floor_rms_dbfs", f"{self.floor_rms_db:.1f}"),
              (f"{prefix}.floor_band_dbfs", f"{self.floor_band_db:.1f}")]
        if not self.ok:
            return r
        r += [(f"{prefix}.sync_db", f"{self.sync_db:.1f}"), (f"{prefix}.clock_ppm", f"{self.ppm:+.1f}")]
        for s in self.steps:
            r.append((f"{prefix}.step{int(s.level_db)}",
                      f"snr={s.snr_db:.1f} sinad={s.sinad_db:.1f} gain={s.gain_db:.1f} rx_peak={s.rx_peak_db:.1f} "
                      f"clipped={s.clipped} settle={s.settle_db:+.1f}"))
        if self.best_level_db is not None:
            pk = _a(self.best_level_db)
            rule = min(1.0, pk * self._crest_ratio())       # equal RMS: the multitone's crest vs a chirp's sqrt(2)
            css = rule if self.css_limit_db is None else min(rule, _a(self.css_limit_db))
            r += [(f"{prefix}.best_tx_peak_dbfs", f"{self.best_level_db:.0f}"),
                  (f"{prefix}.best_tx_reason", self.best_reason),
                  (f"{prefix}.ofdm_amplitude", f"{pk:.2f}"),
                  (f"{prefix}.css_amplitude", f"{css:.2f}"),
                  (f"{prefix}.css_amplitude_rms_rule", f"{rule:.2f}"),
                  (f"{prefix}.css_reason", self.css_reason)]
        for b in self.bands:
            comp = " ".join(f"{int(lv)}:{c:+.1f}" for lv, c in b["comp"].items())
            sw = " ".join(f"{int(lv)}:{c:+.1f}" for lv, c in b.get("sweep", {}).items())
            r.append((f"{prefix}.band.{_fc(b['fc'])}",
                      f"snr={b['snr']:.1f} safe={b['safe']} mask={b['mask']} steps[{comp}] sweep[{sw}]"))
        if self.bands:
            m = [f"{_fc(b['fc'])}:{b['mask']}" for b in self.bands if isinstance(b["mask"], float) and b["mask"] < 0]
            r.append((f"{prefix}.ofdm_tx_mask_db", " ".join(m) or "flat"))
        if self.sweep:
            r.append((f"{prefix}.sweep_thd_db", " ".join(f"{int(lv)}:{v:.1f}" for lv, v in self.sweep.thd_db.items())))
        if self.rx_gate_db is not None:
            r.append((f"{prefix}.rx_gate_db", f"{self.rx_gate_db:+.1f}"))
        if self.in_gain_change_db is not None:
            r.append((f"{prefix}.in_gain_change_db", f"{self.in_gain_change_db:+.1f}"))
        r += [(f"{prefix}.agc", "detected" if self.agc_detected else "none"),
              (f"{prefix}.agc_attack_db", f"{self.agc_attack_db:+.1f}"),
              (f"{prefix}.agc_recovery_s", "n/a" if self.agc_recovery_s is None else f"{self.agc_recovery_s:.2f}")]
        if self.echo:
            e = self.echo
            r += [(f"{prefix}.echo_delay_ms", f"{e.delay_s * 1000:.1f}"),
                  (f"{prefix}.echo_rms_spread_ms", f"{e.rms_spread_ms:.2f}"),
                  (f"{prefix}.echo_t20_ms", f"{e.t20_ms:.1f}"), (f"{prefix}.echo_t30_ms", f"{e.t30_ms:.1f}"),
                  (f"{prefix}.echo_rt60_s", "n/a" if e.rt60_s is None else f"{e.rt60_s:.2f}"),
                  (f"{prefix}.echo_pre_db", f"{e.pre_echo_db:.1f}")]
            for (n, cp), (isi, ici, tot) in sorted(e.ceilings.items()):
                r.append((f"{prefix}.ceiling.{n}.{cp:g}", f"isi={isi:.1f} ici={ici:.1f} total={tot:.1f}"))
            if cur and cur in e.ceilings:
                r.append((f"{prefix}.current_ceiling_db", f"{e.ceilings[cur][2]:.1f} (n_fft={cur[0]} "
                                                          f"cp_ratio={cur[1]:g})"))
            r += [(f"{prefix}.rec_n_fft", str(self.rec_n_fft)), (f"{prefix}.rec_cp_ratio", f"{self.rec_cp:g}"),
                  (f"{prefix}.rec_ceiling_db", f"{self.rec_ceiling_db:.1f}"), (f"{prefix}.rec_reason", self.rec_reason)]
        r.append((f"{prefix}.verdict", self.verdict))
        return r

    crest_db: float = 10.0          # the probe multitone's peak / RMS

    def _crest_ratio(self) -> float:
        """CSS peak for the same RMS as a multitone of unit peak: (1 / crest) * sqrt(2)."""
        return math.sqrt(2) / _a(self.crest_db)


def analyse(probe: Probe, rec: np.ndarray, floor: np.ndarray, *, target_sir_db: float = 20.0,
            rx_peak_target_db: float = -12.0) -> CalResult:
    fs, n_seg, bins = probe.fs, probe.n_seg, probe.bins
    rec = np.asarray(rec, np.float64)
    floor = np.asarray(floor, np.float64)
    res = CalResult(ok=False)
    res.floor_rms_db = _db(float(np.mean(floor ** 2))) if len(floor) else -200.0
    fl = seg_spectrum(floor, n_seg)
    f = np.fft.rfftfreq(n_seg, 1 / fs)
    inb = (f >= probe.band[0]) & (f <= probe.band[1])
    res.floor_band_db = _db(float(fl[inb].sum())) if fl.any() else -200.0
    one = multitone(fs, bins, n_seg, 2 * n_seg / fs, 1.0)[n_seg // 2:n_seg // 2 + n_seg]   # one clean period
    res.crest_db = 20 * math.log10(float(np.max(np.abs(one))) / math.sqrt(float(np.mean(one ** 2))))

    if len(rec) < len(probe.signal):
        res.error = f"recording too short ({len(rec) / fs:.1f}s < probe {len(probe.signal) / fs:.1f}s)"
        return res
    k1, res.sync_db = xcorr_peak(rec[:len(rec) - len(probe.signal) + len(probe.sync)], probe.sync)
    o = int(round(k1))
    if res.sync_db < 25:                     # noise alone reaches ~16 dB; a heard probe 40-70 dB
        res.error = f"probe not found in the recording (sync {res.sync_db:.1f} dB)"
        return res
    if o + len(probe.signal) > len(rec):
        res.error = "probe runs past the end of the recording"
        return res
    # closing chirp: spacing vs nominal = sample-clock offset between the two sound cards
    w0 = o + probe.sync2 - int(0.05 * fs)
    seg = rec[max(0, w0):w0 + len(probe.sync) + int(0.1 * fs)]
    if len(seg) > len(probe.sync):
        k2 = xcorr_peak(seg, probe.sync)[0] + max(0, w0)
        res.ppm = ((k2 - k1) - probe.sync2) / probe.sync2 * 1e6

    nb = probe.empty
    gain0 = None
    tx_tone = None
    tx_each = seg_spectrum(one, n_seg)[bins]
    tone_gain, tone_snr = [], []
    for (s, e, lv) in probe.steps:
        x = rec[o + s + int(0.1 * fs):o + e - int(0.02 * fs)]
        sp = seg_spectrum(x, n_seg)
        tone_p = sp[bins]
        tone_gain.append(10 * np.log10((tone_p + 1e-30) / (tx_each * _a(lv) ** 2)))
        tone_snr.append(10 * np.log10((tone_p + 1e-30) / (fl[bins] + 1e-30)))
        snr = float(np.median(tone_snr[-1]))
        sinad = _db(float(tone_p.mean()) / (float(sp[nb].mean()) + 1e-30))
        if tx_tone is None:
            ref = seg_spectrum(one, n_seg)[bins].mean()
            tx_tone = ref
        gain = _db(float(tone_p.mean()) / (tx_tone * _a(lv) ** 2))
        whole = rec[o + s:o + e]
        early = rec[o + s + int(0.02 * fs):o + s + int(0.12 * fs)]
        late = rec[o + e - int(0.32 * fs):o + e - int(0.02 * fs)]
        settle = _db(float(np.mean(late ** 2))) - _db(float(np.mean(early ** 2)))
        st = StepResult(lv, snr, sinad, gain, 20 * math.log10(float(np.max(np.abs(whole))) + 1e-12),
                        int(np.count_nonzero(np.abs(whole) >= 0.999)), settle)
        res.steps.append(st)
        if gain0 is None:
            gain0 = gain

    # best TX level: climb while SINAD improves by >= 1 dB per 6 dB step and nothing clips.  The climb starts at the
    # first step heard 10 dB over the floor: below that SINAD sits flat under any steady noise the sender adds while
    # playing (amp hiss), and "not improving" there would stop the climb at the quietest step.
    best, why = None, ""
    start = next((i for i, st in enumerate(res.steps) if st.snr_db >= 10.0), 0)
    for i, st in enumerate(res.steps):
        if i < start and not st.clipped:
            continue
        if st.clipped:
            why = f"mic clips at {st.level_db:.0f} dBFS"
            break
        if best is not None and st.sinad_db < res.steps[best].sinad_db + 1.0:
            why = f"SINAD stops improving above {res.steps[best].level_db:.0f} dBFS"
            break
        best = i
    else:
        why = "loudest step still improving (more TX level may help)"
    if best is not None:
        res.best_level_db = res.steps[best].level_db
        res.best_reason = why
        res.in_gain_change_db = rx_peak_target_db - res.steps[best].rx_peak_db

    # per-band compression from the level steps: each tone against its own gain at the first step it was heard
    # clearly (15 dB over the floor), so a band whose gain sags with level is compressing
    G, S = np.array(tone_gain), np.array(tone_snr)              # [step, tone]
    levels = [st.level_db for st in res.steps]
    ref_step = np.argmax(S >= 15.0, axis=0)
    heard = (S >= 15.0).any(axis=0)
    ref = G[ref_step, np.arange(G.shape[1])]
    comp = G - ref                                              # [step, tone]
    ftone = bins * fs / n_seg
    ob = third_octaves(*probe.band)
    sweep_bands = third_octaves(*probe.sweep_range)
    res.sweep = sweep_stats(rec, probe, o, sweep_bands)
    for fc, a, b in ob:
        m = (ftone >= a) & (ftone < b)
        if not m.any():
            continue
        ok_tone = m & heard
        row = {"fc": fc, "snr": float(np.median(S[best if best is not None else -1, m])), "comp": {}}
        if ok_tone.any():
            first = int(np.min(ref_step[ok_tone]))
            safe = None
            for k in range(first, len(levels)):
                c = float(np.median(comp[k, ok_tone]))
                row["comp"][levels[k]] = c
                if c >= -COMP_DB:
                    safe = levels[k]
                else:
                    break
            row["safe"] = "weak" if safe is None else safe
            if safe is None or res.best_level_db is None:
                row["mask"] = "n/a"
            else:
                row["mask"] = float(min(0.0, safe - res.best_level_db))
        else:
            row["safe"], row["mask"] = "weak", "n/a"
        ref_lv = SWEEP_LEVELS_DB[0]
        g = res.sweep.gain_db.get(min(res.sweep.gain_db, key=lambda c: abs(c - fc)), {})
        row["sweep"] = {lv: v - g[ref_lv] for lv, v in g.items() if ref_lv in g}
        res.bands.append(row)

    # CSS: a chirp's peak = a sweep's peak.  Highest sweep level whose transfer stays within COMP_DB of the quietest
    # in every band of the CSS band and whose harmonics stay below THD_MAX_DB.
    cb = probe.css_band or probe.band
    in_css = [fc for fc, _, _ in sweep_bands if cb[0] * 2 ** (-1 / 6) <= fc <= cb[1] * 2 ** (1 / 6)]
    lim, why = None, ""
    for lv in SWEEP_LEVELS_DB:
        worst = min((res.sweep.gain_db[fc].get(lv, 0.0) - res.sweep.gain_db[fc].get(SWEEP_LEVELS_DB[0], 0.0)
                     for fc in in_css), default=0.0)
        worst_fc = min(in_css, key=lambda fc: res.sweep.gain_db[fc].get(lv, 0.0)
                       - res.sweep.gain_db[fc].get(SWEEP_LEVELS_DB[0], 0.0)) if in_css else None
        thd = res.sweep.thd_db.get(lv, -200.0)
        if worst < -COMP_DB:
            why = f"chirp compresses {worst:.1f} dB at {_fc(worst_fc)} Hz by {lv:.0f} dBFS"
            break
        if thd > THD_MAX_DB:
            why = f"chirp harmonics {thd:.1f} dB at {lv:.0f} dBFS"
            break
        lim = lv
    else:
        why = f"clean up to {SWEEP_LEVELS_DB[-1]:.0f} dBFS: the equal-RMS rule sets the CSS level"
    res.css_limit_db = lim if lim != SWEEP_LEVELS_DB[-1] else None
    if lim is None:
        why += f" (already at the quietest sweep, {SWEEP_LEVELS_DB[0]:.0f} dBFS: keep CSS below it)"
        res.css_limit_db = SWEEP_LEVELS_DB[0] - 6.0
    res.css_reason = why

    # AGC: the loud part's level just after the jump vs its end, and the quiet part after it
    (q1s, q1e, _), (ls, le, _), (q2s, q2e, _) = probe.agc
    lvl = lambda a, b: _db(float(np.mean(rec[o + a:o + b] ** 2)))                    # noqa: E731
    attack = lvl(le - int(0.3 * fs), le - int(0.02 * fs)) - lvl(ls + int(0.02 * fs), ls + int(0.12 * fs))
    q_ref = lvl(q1e - int(0.4 * fs), q1e - int(0.02 * fs))
    res.agc_attack_db = attack
    rec_t = None
    hop = int(0.05 * fs)
    for k in range(q2s + int(0.02 * fs), q2e - hop, hop):
        if abs(lvl(k, k + hop) - q_ref) <= 2.0:
            rec_t = (k - q2s) / fs
            break
    res.agc_recovery_s = rec_t
    floor_q = res.floor_rms_db
    quiet_heard = q_ref > floor_q + 6
    res.agc_detected = attack < -3.0 or (quiet_heard and (rec_t is None or rec_t > 0.3))

    # echo
    s0, s1, slv = probe.sweeps[ECHO_SWEEP]
    a0 = max(0, o + s0 - int(0.05 * fs))
    seg = rec[a0:o + s1 + int(0.4 * fs)]
    lo, hi = probe.band
    ir = impulse_response(seg, probe.sweep_sig * _a(slv), fs, (lo, min(hi * 1.1, fs / 2 * 0.95)))
    res.echo = echo_stats(ir, fs, res.ppm, hi)
    res.rec_n_fft, res.rec_cp, res.rec_ceiling_db, res.rec_reason = recommend_ofdm(res.echo.ceilings, target_sir_db)

    # noise gate / downward expander on the input (MacBook built-in mic): the floor recorded in silence is quieter
    # than the noise that comes through once a signal opens the gate.  Linear steps (no compression) where the tones
    # are well heard should have SINAD ~= SNR; a steady shortfall is that extra noise.  It makes SNR against the
    # silent floor optimistic: SINAD is then the real figure for the modem.
    lin = [st.snr_db - st.sinad_db for st in res.steps[:(best + 1) if best is not None else None]
           if 15.0 <= st.snr_db and not st.clipped]
    if len(lin) >= 2:
        res.rx_gate_db = float(np.median(lin))

    best_snr = res.steps[best].snr_db if best is not None else -99.0
    best_sinad = res.steps[best].sinad_db if best is not None else -99.0
    v = []
    if best is None or best_snr < 6:
        v.append("no-signal")
    elif best_sinad < 15:
        v.append("environment-limited" if best_snr - best_sinad < 3 else "distortion-limited")
    if res.agc_detected:
        v.append("agc-present")
    if res.rx_gate_db is not None and res.rx_gate_db > 6.0:
        v.append("rx-noise-gate")
    if res.rec_ceiling_db is not None and res.rec_ceiling_db < target_sir_db:
        v.append("echo-limited")
    if any(isinstance(b["mask"], float) and b["mask"] <= -6 for b in res.bands):
        v.append("uneven-compression")
    res.verdict = ",".join(v) or "ok"
    res.ok = True
    return res


def split_floor(probe: Probe, rec: np.ndarray, min_s: float = 0.8) -> tuple[np.ndarray, str]:
    """(noise-only stretch of ``rec``, "" or why not) for a LISTEN recording, where nobody told us when the peer
    would start: the audio before the sync chirp, else the audio after the probe ended."""
    fs = probe.fs
    rec = np.asarray(rec, np.float64)
    if len(rec) < len(probe.signal):
        return rec[:0], f"recording too short ({len(rec) / fs:.1f}s < probe {len(probe.signal) / fs:.1f}s)"
    k, q = xcorr_peak(rec[:len(rec) - len(probe.signal) + len(probe.sync)], probe.sync)
    if q < 25:
        return rec[:0], f"probe not found in the recording (sync {q:.1f} dB)"
    o, need, guard = int(round(k)), int(min_s * fs), int(0.1 * fs)
    if o - guard >= need:
        return rec[:o - guard], ""
    end = o + len(probe.signal) + int(0.3 * fs)              # past the probe and the room's ring-out
    if len(rec) - end >= need:
        return rec[end:], ""
    return rec[:0], (f"no {min_s:g}s of quiet before or after the probe: start SEND at least {min_s + 0.2:g}s after "
                     f"LISTEN, or LISTEN longer")
