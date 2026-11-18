# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Line probe stages 0-2: quiet floor, single tones, level steps, tone pairs -- and a classifier that turns the
measurements into hardware findings (plan: audiotestserver/tests/OFDM/LINE_PROBE.md, "Why" table).

One deterministic burst, built from the same arguments on both hosts:

    wake  chirp-up  quiet  sweep tones...  level steps...  close pairs...  far pair  chirp-down

* The two chirps locate the burst in the recording and give the TX/RX clock ratio r (distance between them in RX
  samples / nominal).  Every segment is then found at its scheduled time x r, and every tone at f / r.
* All tones sit on a grid of ``GRID_HZ`` (150 Hz): a whole number of cycles per 1/150 s at 44.1 and 48 kHz.  Every
  duration is a multiple of 1/300 s, a whole number of samples at both rates.
* Tones are demodulated per block of 1/300 s by least squares (a fixed local basis, rotated to absolute time), so
  the image of the negative frequency never biases a low tone.  The block series gives amplitude (AM) and phase
  (PM) wander; the segment-wide fit gives gain, harmonics, IMD3 and the residual noise.

What each stage separates (the classifier's rules, ``classify``):
  quiet         additive floor, hum / spurs                            -> floor, spurs
  single tones  |H(f)|, cut-off, PM rms vs f, THD, noise beside a tone  -> tilt (EQ), band edge, jitter shape
  level steps   SNR, THD and gain vs level                              -> additive vs signal-proportional vs
                                                                           nonlinear; compression point
  close pairs   IMD3 vs level                                           -> limiter / clipping
  far pair      correlation of the two tones' phase wander              -> common timing jitter (pilots fix it)
  all tones     amplitude dips                                          -> Bluetooth dropouts
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

GRID_HZ = 150.0                 # tone grid: integer period at 44.1 and 48 kHz (gcd 300 / 2)
BLOCK_HZ = 300.0                # demod block rate: 160 samples @ 48 kHz, 147 @ 44.1 kHz
SETTLE_S = 0.08                 # skipped at the start of each segment (codec / AGC / limiter settle)
TAIL_S = 0.02                   # skipped at the end (fade-out)
FADE_S = 0.01
ZERO_RUN_MIN = 16               # samples of exact 0.0 that count as a digital zero-fill gap


@dataclasses.dataclass(frozen=True)
class Segment:
    kind: str                   # wake | chirp_up | chirp_down | quiet | tone | level | pair | far
    start_s: float
    dur_s: float
    tones: tuple = ()           # ((freq_hz, amplitude), ...)
    label: str = ""


@dataclasses.dataclass(frozen=True)
class Sequence:
    lo: float
    hi: float
    peak: float
    segments: tuple
    chirp_band: tuple

    @property
    def dur_s(self) -> float:
        s = self.segments[-1]
        return s.start_s + s.dur_s

    def find(self, kind: str) -> list:
        return [s for s in self.segments if s.kind == kind]

    def describe(self) -> str:
        n = {k: len(self.find(k)) for k in ("tone", "level", "pair", "far")}
        return (f"sweep {self.lo:g}-{self.hi:g} Hz peak {self.peak:g}: {n['tone']} tones, {n['level']} level steps, "
                f"{n['pair']} close pairs, {n['far']} far pair, {self.dur_s:.2f} s")


def _q(t: float) -> float:
    """Round a duration to a whole number of 1/300 s (an integer sample count at 44.1 and 48 kHz)."""
    return round(t * BLOCK_HZ) / BLOCK_HZ


def _grid(f: float) -> float:
    return max(1, round(f / GRID_HZ)) * GRID_HZ


def plan_sweep(lo: float, hi: float, peak: float = 0.5, n_tones: int = 12, tone_s: float = 0.4,
               quiet_s: float = 0.6, far_s: float = 2.0, rates=(44100, 48000)) -> Sequence:
    """The stage 0-2 burst for a band.  ``peak`` is the loudest sample; the sweep tones run at peak/2 so the level
    steps (peak/4, peak/2, peak) bracket them, and each tone of a pair at peak/2 (the pair peaks at ``peak``)."""
    nyq = 0.5 * min(int(r) for r in rates)
    hi = min(float(hi), nyq - 2 * GRID_HZ)
    lo = max(float(lo), GRID_HZ)
    if hi - lo < 4 * GRID_HZ:
        raise ValueError(f"band {lo:g}-{hi:g} Hz too narrow for a sweep (need {4 * GRID_HZ:g} Hz)")
    if not (0.01 <= peak <= 1.0) or n_tones < 3:
        raise ValueError("peak 0.01..1, n_tones >= 3")
    f_lo, f_hi = _grid(math.ceil(lo / GRID_HZ) * GRID_HZ), _grid(math.floor(hi / GRID_HZ) * GRID_HZ)
    sweep = sorted({_grid(f) for f in np.geomspace(f_lo, f_hi, n_tones)})
    sweep = [f for f in sweep if f_lo <= f <= f_hi]
    picks = [sweep[0], sweep[len(sweep) // 2], sweep[-1]]          # low / mid / high for steps and pairs
    d = 2 * GRID_HZ                                                 # pair spacing: IMD3 lands on f1-d, f2+d

    segs: list[Segment] = []
    t = 0.0

    def add(kind, dur, tones=(), label=""):
        nonlocal t
        dur = _q(dur)
        segs.append(Segment(kind, t, dur, tuple(tones), label))
        t += dur

    add("wake", 0.6, label="wake the BT stream / AGC")
    add("chirp_up", 0.3, label="sync")
    add("quiet", quiet_s, label="floor")
    for f in sweep:
        add("tone", tone_s, [(f, peak / 2)], f"{f:g}")
    for f in picks:
        for g in (0.25, 0.5, 1.0):
            add("level", tone_s, [(f, peak * g)], f"{f:g}@{20 * math.log10(g):+.0f}dB")
    for f in picks:
        f1 = max(f - GRID_HZ, f_lo + d) if f != picks[-1] else f_hi - d - GRID_HZ
        f1 = _grid(f1)
        for g in (0.25, 1.0):
            add("pair", tone_s, [(f1, peak * g / 2), (f1 + d, peak * g / 2)], f"{f1:g}+{f1 + d:g}@{20 * math.log10(g):+.0f}dB")
    add("far", far_s, [(sweep[0], peak / 2), (sweep[-1], peak / 2)], f"{sweep[0]:g}+{sweep[-1]:g}")
    add("chirp_down", 0.3, label="sync")
    return Sequence(lo, hi, float(peak), tuple(segs), (f_lo, f_hi))


# -- TX ------------------------------------------------------------------------------------------------------

def _chirp(fs: int, dur: float, f0: float, f1: float, amp: float, off=None) -> np.ndarray:
    n = int(round(dur * fs))
    t = np.arange(n) / fs + (0.0 if off is None else off)
    x = amp * np.cos(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / dur * t * t))
    w = np.ones(n)
    r = max(1, int(0.05 * n))
    w[:r] = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
    w[-r:] = w[:r][::-1]
    return x * w


def synth(seq: Sequence, fs: int, t_offset_s: np.ndarray | None = None) -> np.ndarray:
    """The burst at ``fs``.  Tones use absolute time (continuous phase across the burst) and 10 ms fades.
    ``t_offset_s`` (one value per sample, tests) evaluates tones and chirps at t + offset: exact timing jitter."""
    fs = int(fs)
    x = np.zeros(int(round(seq.dur_s * fs)))
    off = np.zeros(len(x)) if t_offset_s is None else np.asarray(t_offset_s, dtype=np.float64)[:len(x)]
    fade = int(FADE_S * fs)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    rng = np.random.default_rng(1234)
    for s in seq.segments:
        a, n = int(round(s.start_s * fs)), int(round(s.dur_s * fs))
        if s.kind == "chirp_up":
            y = _chirp(fs, s.dur_s, *seq.chirp_band, 0.7 * seq.peak, off[a:a + n])
        elif s.kind == "chirp_down":
            y = _chirp(fs, s.dur_s, seq.chirp_band[1], seq.chirp_band[0], 0.7 * seq.peak, off[a:a + n])
        elif s.kind == "wake":
            y = rng.standard_normal(n)
            spec = np.fft.rfft(y)
            fr = np.fft.rfftfreq(n, 1 / fs)
            spec[(fr < seq.chirp_band[0]) | (fr > seq.chirp_band[1])] = 0
            y = np.fft.irfft(spec, n)
            y *= 0.1 * seq.peak / (np.abs(y).max() + 1e-12)
        elif s.kind == "quiet":
            continue
        else:
            tt = (a + np.arange(n)) / fs + off[a:a + n]
            y = sum(amp * np.cos(2 * np.pi * f * tt) for f, amp in s.tones)
        y = np.asarray(y, dtype=np.float64).copy()
        y[:fade] *= ramp
        y[n - fade:] *= ramp[::-1]
        x[a:a + n] = y[:n]
    return x


# -- RX: sync ------------------------------------------------------------------------------------------------

def _xcorr_peak(x: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    n = 1 << int(math.ceil(math.log2(len(x) + len(ref))))
    c = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(ref, n)), n)[:len(x) - len(ref) + 1]
    c = np.abs(c)
    i = int(np.argmax(c))
    frac = 0.0
    if 0 < i < len(c) - 1:
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den else 0.0
    q = c[i] / (np.median(c) + 1e-30)
    return i + frac, float(q)


def locate(seq: Sequence, x: np.ndarray, fs: int) -> tuple[float, float]:
    """(t0 in RX samples of the burst's time 0, clock ratio r = RX samples per scheduled TX sample)."""
    up, dn = seq.find("chirp_up")[0], seq.find("chirp_down")[0]
    lo, hi = seq.chirp_band
    nominal = (dn.start_s - up.start_s) * fs
    r = 1.0
    # A linear chirp's matched filter turns a frequency offset into a time offset (range-Doppler coupling), with
    # opposite signs for the up and down chirps: -529 ppm reads ~12 ppm wrong.  So correlate again with
    # references stretched by the first estimate (TX time x r, frequencies / r).
    for _ in range(3):
        ref_up = _chirp(fs, up.dur_s * r, lo / r, hi / r, 1.0)
        ref_dn = _chirp(fs, dn.dur_s * r, hi / r, lo / r, 1.0)
        iu, qu = _xcorr_peak(x, ref_up)
        idn, qd = _xcorr_peak(x, ref_dn)
        if min(qu, qd) < 8:
            raise ValueError(f"sync chirps not found (peak/median {qu:.1f}, {qd:.1f}): same args on both hosts? "
                             f"SEND inside the LISTEN window?")
        r_new = (idn - iu) / nominal
        if not (0.99 < r_new < 1.01):
            raise ValueError(f"sync chirps {idn - iu:.0f} samples apart, expected {nominal:.0f}: wrong pairing")
        done = abs(r_new - r) < 1e-7
        r = r_new
        if done:
            break
    return iu - up.start_s * fs * r, float(r)


# -- RX: tone measurement ------------------------------------------------------------------------------------

def _demod(x: np.ndarray, a0: float, fs: int, freqs, B: int) -> np.ndarray:
    """Per-block complex amplitudes (W, len(freqs)) of x (starting at absolute RX sample a0) for tones ``freqs``
    (RX Hz): x ~ sum Re(c_k e^{j w_k t}) per block, by least squares on a local basis rotated to absolute time."""
    W = len(x) // B
    n = np.arange(B)
    w = 2 * np.pi * np.asarray(freqs, dtype=np.float64) / fs
    basis = np.concatenate([np.cos(np.outer(n, w)), -np.sin(np.outer(n, w))], axis=1)    # B x 2K
    pinv = np.linalg.pinv(basis)
    coef = (pinv @ x[:W * B].reshape(W, B).T).T                                          # W x 2K
    K = len(w)
    c_loc = coef[:, :K] + 1j * coef[:, K:]
    t0 = a0 + np.arange(W)[:, None] * B
    return c_loc * np.exp(-1j * w[None, :] * t0)             # x = Re(c_abs e^{jw(t0+n)}) -> c_loc = c_abs e^{jw t0}


def _fit(x: np.ndarray, a0: float, fs: int, freqs) -> tuple[np.ndarray, np.ndarray]:
    """Segment-wide least-squares fit at fixed frequencies: (complex amplitudes, residual)."""
    t = a0 + np.arange(len(x))
    w = 2 * np.pi * np.asarray(freqs, dtype=np.float64) / fs
    A = np.concatenate([np.cos(np.outer(t, w)), -np.sin(np.outer(t, w))], axis=1)
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    K = len(w)
    return coef[:K] + 1j * coef[K:], x - A @ coef


def _psd(x: np.ndarray, fs: int, nper: int):
    nper = min(nper, len(x))
    win = np.hanning(nper)
    step = nper // 2
    segs = [x[i:i + nper] * win for i in range(0, len(x) - nper + 1, step)] or [np.resize(x, nper) * win]
    P = np.mean([np.abs(np.fft.rfft(s)) ** 2 for s in segs], axis=0) / (fs * np.sum(win ** 2))
    return np.fft.rfftfreq(nper, 1 / fs), 2 * P                                 # one-sided density, units^2/Hz


def _db(p) -> float:
    return float(10 * np.log10(max(float(p), 1e-300)))


def _detrend_phase(ph: np.ndarray) -> np.ndarray:
    i = np.arange(len(ph), dtype=np.float64)
    return ph - np.polyval(np.polyfit(i, ph, 2), i) if len(ph) >= 4 else ph - ph.mean()


def _measure_tones(seg: Segment, x: np.ndarray, fs: int, t0: float, r: float, band, quiet_psd) -> dict:
    a = t0 + (seg.start_s + SETTLE_S) * fs * r
    b = t0 + (seg.start_s + seg.dur_s - TAIL_S) * fs * r
    ia, ib = int(math.ceil(a)), int(b)
    y = x[ia:ib].astype(np.float64)
    y = y - y.mean()
    f_tx = [f for f, _ in seg.tones]
    amp_tx = [amp for _, amp in seg.tones]
    f_rx = [f / r for f in f_tx]
    B = int(round(fs / BLOCK_HZ))
    C = _demod(y, ia, fs, f_rx, B)                                             # W x K
    amp = np.abs(C)
    ph = np.unwrap(np.angle(C), axis=0)
    out = {"label": seg.label, "kind": seg.kind, "f_hz": f_tx, "tx_amp": amp_tx}
    # fine frequency per tone from the phase slope (blocks are B samples apart)
    slope = np.array([np.polyfit(np.arange(len(p)), p, 1)[0] for p in ph.T]) if len(ph) >= 4 else np.zeros(len(f_rx))
    f_fine = [f + s * fs / (2 * np.pi * B) for f, s in zip(f_rx, slope)]
    out["ppm"] = [(ff / f - 1) * 1e6 for ff, f in zip(f_fine, f_tx)]
    pm = np.array([_detrend_phase(p) for p in ph.T]).T
    out["pm_rms_rad"] = list(np.sqrt(np.mean(pm ** 2, axis=0)))
    out["am_rms"] = list(np.std(amp, axis=0) / (np.mean(amp, axis=0) + 1e-30))
    med = np.median(amp, axis=0)
    dips = (amp < 0.5 * med[None, :]).any(axis=1)
    out["dip_at"] = [int(ia + i * B) for i in np.where(dips)[0]]               # absolute RX sample of each dip
    out["dropout_blocks"] = int(dips.sum())
    out["dropout_ms"] = 1000.0 * dips.sum() / BLOCK_HZ
    out["_pm_series"] = pm

    # segment-wide fit: fundamentals + 2nd/3rd harmonics (+ IMD3 for pairs), residual noise
    extra = []
    if len(f_fine) == 1:
        extra = [2 * f_fine[0], 3 * f_fine[0]]
    elif seg.kind == "pair":
        f1, f2 = f_fine
        extra = [2 * f1 - f2, 2 * f2 - f1]
    extra = [f for f in extra if 0 < f < 0.5 * fs * 0.98]
    coef, res = _fit(y, ia, fs, list(f_fine) + extra)
    tone_pow = np.abs(coef[:len(f_fine)]) ** 2 / 2
    out["gain_db"] = [_db(p / (t ** 2 / 2)) for p, t in zip(tone_pow, amp_tx)]
    out["level_dbfs"] = [_db(2 * p) for p in tone_pow]                          # re a full-scale sine
    ref = float(np.mean(tone_pow))
    xp = np.abs(coef[len(f_fine):]) ** 2 / 2
    if len(f_fine) == 1:
        out["h2_dbc"] = _db(xp[0] / ref) if len(xp) > 0 else None
        out["h3_dbc"] = _db(xp[1] / ref) if len(xp) > 1 else None
        out["thd_dbc"] = _db(xp.sum() / ref) if len(xp) else None
    elif seg.kind == "pair":
        out["imd3_dbc"] = _db(xp.mean() / ref) if len(xp) else None
    # residual: in-band power (= SNR), near-skirt and far noise vs the quiet floor
    fr, P = _psd(res, fs, int(fs / 37.5))
    df = fr[1] - fr[0]
    inb = (fr >= band[0]) & (fr <= band[1])
    out["snr_db"] = _db(ref / (np.sum(P[inb]) * df + 1e-30))
    near = np.zeros_like(inb)
    keep = inb.copy()
    for f in list(f_fine) + extra:
        near |= (np.abs(fr - f) > 20) & (np.abs(fr - f) <= 200)
        keep &= np.abs(fr - f) > 200
    out["skirt_dbc_hz"] = _db(np.mean(P[near & inb]) / ref) if (near & inb).any() else None   # density re tone
    # discrete residual peaks (power in a 37.5 Hz bin re the tone): images / mixing products / spurs
    pk = []
    Pb = P * df / ref
    for i in np.where(inb | (fr > band[1]))[0]:
        if 0 < i < len(Pb) - 1 and Pb[i] > 10 ** -5.0 and Pb[i] >= Pb[i - 1] and Pb[i] >= Pb[i + 1] \
                and all(abs(fr[i] - g) > 60 for g in list(f_fine) + extra):
            pk.append((float(fr[i]), _db(Pb[i] + Pb[i - 1] + Pb[i + 1])))
    out["peaks"] = sorted(pk, key=lambda q: -q[1])[:10]
    if keep.any():
        qf = quiet_psd(fr)
        out["noise_rise_db"] = _db(np.median(P[keep])) - _db(np.median(qf[keep]))
    else:
        out["noise_rise_db"] = None
    return out


def analyze(seq: Sequence, x: np.ndarray, fs: int) -> dict:
    """Measure a received burst (``x`` = the whole recording; the chirps find it)."""
    fs = int(fs)
    x = np.asarray(x, dtype=np.float64)
    t0, r = locate(seq, x, fs)
    end = t0 + seq.dur_s * fs * r
    if t0 < 0 or end > len(x):
        raise ValueError(f"burst runs off the recording ({t0 / fs:.2f}-{end / fs:.2f} s of {len(x) / fs:.2f} s)")
    band = seq.chirp_band
    q = seq.find("quiet")[0]
    qa = int(t0 + (q.start_s + SETTLE_S) * fs * r)
    qb = int(t0 + (q.start_s + q.dur_s - TAIL_S) * fs * r)
    qx = x[qa:qb] - x[qa:qb].mean()
    qfr, qP = _psd(qx, fs, int(fs / 37.5))
    inb = (qfr >= band[0]) & (qfr <= band[1])
    floor = float(np.median(qP[inb]))
    spurs = []
    for i in np.where(inb)[0]:
        if 0 < i < len(qP) - 1 and qP[i] > 30 * floor and qP[i] >= qP[i - 1] and qP[i] >= qP[i + 1]:
            spurs.append((float(qfr[i]), _db(qP[i] / floor)))
    spurs = sorted(spurs, key=lambda s: -s[1])[:6]

    def quiet_psd(fr):
        return np.interp(fr, qfr, qP)

    res = {"fs": fs, "t0_s": t0 / fs, "clock_ppm": (1 / r - 1) * 1e6,
           "quiet": {"floor_dbfs_hz": _db(floor), "floor_band_dbfs": _db(np.sum(qP[inb]) * (qfr[1] - qfr[0])),
                     "rms_dbfs": _db(2 * np.mean(qx ** 2)), "spurs": spurs}}
    for kind in ("tone", "level", "pair", "far"):
        res[kind] = [_measure_tones(s, x, fs, t0, r, band, quiet_psd) for s in seq.find(kind)]
    # digital zero-fill: runs of exact 0.0 inside the burst.  A radio / FM demod never outputs exact zeros, so these
    # are RX buffer underruns (audio bridge, USB).  A bridge that underruns usually also drops samples to catch up,
    # so the samples are neither purely inserted nor lost and the clock estimate cannot be corrected for them.
    ia, ib = int(max(t0, 0)), int(min(end, len(x)))
    z = np.concatenate([[False], x[ia:ib] == 0.0, [False]]).astype(np.int8)
    d = np.diff(z)
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    runs = [(ia + int(a), int(b - a)) for a, b in zip(st, en) if b - a >= ZERO_RUN_MIN]
    res["zero_runs"] = runs
    res["zero_samples"] = int(sum(n for _, n in runs))
    B = int(round(fs / BLOCK_HZ))
    for kind in ("tone", "level", "pair", "far"):
        for m in res[kind]:
            dig = [p for p in m["dip_at"] if any(a - B <= p <= a + n for a, n in runs)]
            m["digital_dips"] = len(dig)
    far = res["far"][0]
    pm = far.pop("_pm_series")
    lo, hi = pm[:, 0], pm[:, 1]
    f_lo, f_hi = far["f_hz"]
    rho = float(np.corrcoef(lo, hi)[0, 1]) if np.std(lo) > 0 and np.std(hi) > 0 else 0.0
    far["pm_correlation"] = rho
    far["pm_ratio"] = float(np.std(hi) / (np.std(lo) + 1e-30))
    far["pm_ratio_if_timing"] = f_hi / f_lo
    for kind in ("tone", "level", "pair"):
        for m in res[kind]:
            m.pop("_pm_series", None)
    return res


# -- classifier ----------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Finding:
    key: str
    severity: str               # ok | info | warn | bad
    summary: str
    fix: str = ""


def _slope(xs, ys) -> float:
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 and np.ptp(xs) > 0 else 0.0


# SDR / soundcard rates a stage might resample from; an image fold F = |rate - k*fs| names the culprit
_COMMON_RATES = (250000, 240000, 256000, 200000, 192000, 170000, 171000, 1024000, 1200000, 1920000, 2048000,
                 2400000, 2560000, 96000, 88200, 44100, 32000, 22050, 16000, 8000)


def _mixing(tones: list, fs: int):
    """(F, tones supporting, median image level dBc, alias explanation) or None."""
    votes: dict = {}
    for m in tones:
        f = m["f_hz"][0]
        for p, lvl in m.get("peaks", []):
            for F in (p + f, abs(p - f)):
                if F > 300:
                    votes.setdefault(round(F / 75.0) * 75.0, []).append((f, lvl, F))
    best = None
    for F, v in votes.items():
        near = [x for G, vv in votes.items() if abs(G - F) <= 75 for x in vv]
        tones_ok = {f for f, _, _ in near}
        if len(tones_ok) >= 3 and (best is None or len(tones_ok) > best[1]):
            best = (float(np.median([G for _, _, G in near])), len(tones_ok), float(np.median([l for _, l, _ in near])))
    if not best or best[1] < max(3, len(tones) // 2):
        return None
    F = best[0]
    alias = ""
    for r in _COMMON_RATES:
        k = round(r / fs)
        if k >= 1 and r != fs and abs(abs(r - k * fs) - F) <= 75:
            alias = f"{r} -> {fs} Hz resampling folds at {r} - {k}x{fs} = {abs(r - k * fs)} Hz"
            if r == 250000:
                alias += (" (rtl_fm -s 250k: captures at 1.25 MS/s, not an RTL-SDR streaming rate -- those are "
                          "250k, 900k, 1.2M, 2.4M; use -s 240k = 1.2 MS/s capture, an integer 5 x 48 kHz)")
            break
    return F, best[1], best[2], alias


def classify(res: dict) -> list[Finding]:
    """Map the measurements to the causes in LINE_PROBE.md's table.  Thresholds are first guesses meant to be
    tuned on real links; each finding carries its evidence in the summary."""
    out: list[Finding] = []
    tones = res["tone"]
    f = np.array([m["f_hz"][0] for m in tones])
    gain = np.array([m["gain_db"][0] for m in tones])
    top = float(gain.max())

    # clock
    ppm = res["clock_ppm"]
    tone_ppm = np.array([m["ppm"][0] for m in tones])
    sev = "warn" if abs(ppm) > 200 else "info" if abs(ppm) > 20 else "ok"
    out.append(Finding("clock", sev, f"TX/RX clock {ppm:+.1f} ppm (per-tone spread {np.ptp(tone_ppm):.1f} ppm)",
                       "receiver must track the offset (pilots / CSS PLL)" if sev != "ok" else ""))

    # quiet floor and spurs
    q = res["quiet"]
    sp = ", ".join(f"{fz:g} Hz +{d:.0f} dB" for fz, d in q["spurs"])
    out.append(Finding("floor", "warn" if q["spurs"] else "ok",
                       f"quiet floor {q['floor_band_dbfs']:.1f} dBFS in band" + (f"; spurs: {sp}" if sp else ""),
                       "tonal spurs in the idle channel (hum, pilot, whine): notch or avoid those carriers"
                       if sp else ""))

    # band edge (a steep fall: > 12 dB/octave between neighbouring tones, to > 6 dB down -- a tilt is a few dB per
    # octave at most), then the tilt over the usable part
    usable = gain >= top - 30
    i_top = int(np.argmax(gain))
    for i in range(i_top + 1, len(f)):
        if (gain[i - 1] - gain[i]) / np.log2(f[i] / f[i - 1]) > 12 and gain[i] < top - 6:
            usable[i:] = False
            break
    for i in range(i_top - 1, -1, -1):
        if (gain[i + 1] - gain[i]) / np.log2(f[i + 1] / f[i]) > 12 and gain[i] < top - 6:
            usable[:i + 1] = False
            break
    dead = f[~usable]
    if len(dead):
        out.append(Finding("band_edge", "bad", f"tones {', '.join(f'{v:g}' for v in dead)} Hz are past a steep "
                           f"fall (cut-off, {top - gain[~usable].max():.0f}+ dB down)", f"keep the band below {f[usable].max():g} Hz / above {f[usable].min():g} Hz"))
    span = float(np.ptp(gain[usable]))
    per_dec = _slope(np.log10(f[usable]), gain[usable])
    sev = "warn" if span > 6 else "info" if span > 3 else "ok"
    eq = ", ".join(f"{fz:g}:{top - g:+.1f}" for fz, g in zip(f[usable], gain[usable]))
    out.append(Finding("tilt", sev, f"response spans {span:.1f} dB over the usable band ({per_dec:+.1f} dB/decade)",
                       f"TX EQ / pre-emphasis (Hz:dB) {eq}" if sev != "ok" else ""))

    # images / mixing: a lone tone f that also shows up at |F - f| and F + f for one F common to many tones is
    # multiplied by (or folded around) F -- a resampler alias or a mixing product, level-proportional like codec
    # noise but discrete, and it smears a multi-tone signal across the band.
    mix = _mixing(res["tone"], res["fs"])
    if mix:
        F, sup, lvl, alias = mix
        out.append(Finding("mixing", "bad" if lvl > -40 else "warn",
                           f"each tone reappears mirrored around {F:.0f} Hz (at {F:.0f}-f and {F:.0f}+f, ~{lvl:.0f} dBc, "
                           f"{sup} tones agree)" + (f": {alias}" if alias else ""),
                           "an image / mixing product, not noise: it caps every carrier's SNR near "
                           f"{-lvl - 3:.0f} dB at any level. " + ("Use an integer resampling ratio (or a proper "
                           "anti-alias filter) on that stage" if alias else "Find the F Hz source (resampler "
                           "ratio, switching supply, pilot) and remove it")))

    # noise mechanism from the level steps: d(SNR)/d(level) in dB per dB, THD vs level, gain compression
    lv = res["level"]
    groups: dict = {}
    for m in lv:
        groups.setdefault(m["f_hz"][0], []).append(m)
    snr_slopes, thd_top, comp = [], [], []
    for fz, ms in groups.items():
        lvl = [20 * math.log10(m["tx_amp"][0]) for m in ms]
        snr_slopes.append(_slope(lvl, [m["snr_db"] for m in ms]))
        if ms[-1]["thd_dbc"] is not None:                    # no harmonic below Nyquist: no THD
            thd_top.append((fz, ms[-1]["thd_dbc"], _slope(lvl, [m["thd_dbc"] for m in ms])))
        comp.append((fz, ms[-1]["gain_db"][0] - ms[0]["gain_db"][0]))
    s = float(np.median(snr_slopes))
    if s <= 0.3 and mix:
        out.append(Finding("noise", "warn", f"SNR does not follow the level ({s:.2f} dB/dB), explained by the "
                           f"{mix[0]:.0f} Hz images (finding 'mixing')", "fix the images first, then re-probe"))
    elif s > 0.7:
        out.append(Finding("noise", "info", f"SNR follows the level ({s:.2f} dB/dB): additive noise",
                           "more TX level helps until distortion starts"))
    elif s > 0.3:
        out.append(Finding("noise", "warn", f"SNR rises only {s:.2f} dB per dB of level: part of the noise scales "
                           f"with the signal", "level helps a little; budget bits on the measured SNR"))
    else:
        out.append(Finding("noise", "bad", f"SNR does not follow the level ({s:.2f} dB/dB): the noise is "
                           f"signal-proportional (codec quantisation / multiplicative), or distortion",
                           "more level does not help; give high-noise carriers fewer bits"))

    # nonlinearity, per frequency (low / mid / high pick): gain compression and SNR collapse at the top level
    # step, THD of the lone tone, IMD3 of the close pair at full level.  Saturation only at the top of the band
    # with a clean low end is the FM pre-emphasis driving the deviation limiter.
    pairs = res["pair"]
    sat, worst = [], (0.0, -200.0, -200.0)
    for i, (fz, ms) in enumerate(groups.items()):
        comp_f = ms[-1]["gain_db"][0] - ms[0]["gain_db"][0]
        snr_drop = ms[-2]["snr_db"] - ms[-1]["snr_db"]                 # SNR should rise 6 dB from -6 to 0 dB
        thd_f = ms[-1]["thd_dbc"] if ms[-1]["thd_dbc"] is not None else -200.0
        pr = pairs[2 * i:2 * i + 2]
        imd_f = pr[1]["imd3_dbc"] if len(pr) == 2 and pr[1]["imd3_dbc"] is not None else -200.0
        worst = (min(worst[0], comp_f), max(worst[1], thd_f), max(worst[2], imd_f))
        if comp_f < -1.0 or snr_drop > 3.0 or thd_f > -35 or imd_f > -35:
            sat.append((fz, comp_f, thd_f, imd_f, snr_drop))
    worst_thd = worst[1]
    if not sat:
        out.append(Finding("nonlinearity", "ok", f"no saturation over 12 dB of level (worst: gain {worst[0]:+.1f} dB, "
                           f"THD {worst[1]:.1f} dBc, IMD3 {worst[2]:.1f} dBc)"))
    else:
        picks = list(groups)
        where = "; ".join(f"{fz:g} Hz: gain {c:+.1f} dB, SNR {-d:+.1f} dB at the top step"
                          + (f", THD {t:.0f} dBc" if t > -200 else "") + (f", IMD3 {m:.0f} dBc" if m > -200 else "")
                          for fz, c, t, m, d in sat)
        hf_only = [x[0] for x in sat] == [picks[-1]]
        lf_only = [x[0] for x in sat] == [picks[0]]
        fix = ("only the top of the band saturates: FM pre-emphasis drives the limiter there -- lower the level "
               "of the high carriers (TX de-emphasis / BAT power tilt), back off ~6 dB, or keep freq_hi lower"
               if hf_only else "only the low end saturates (transformer / speaker excursion): lower the low carriers"
               if lf_only else "back the TX level off (limiter / clipping); keep the crest factor down")
        out.append(Finding("nonlinearity", "bad", f"saturates at full level: {where}", fix))

    # phase noise / timing jitter: PM rms vs f, and the far pair's correlation
    pm = np.array([m["pm_rms_rad"][0] for m in tones])
    pm_slope = _slope(np.log10(f[usable]), np.log10(pm[usable] + 1e-12))
    far = res["far"][0]
    rho, ratio, want = far["pm_correlation"], far["pm_ratio"], far["pm_ratio_if_timing"]
    timing = rho > 0.5 and pm_slope > 0.6
    jitter_us = float(np.median(pm[usable] / (2 * np.pi * f[usable]))) * 1e6
    if timing:
        out.append(Finding("phase", "warn", f"phase wander grows with f (slope {pm_slope:.2f}) and is shared "
                           f"(far-pair correlation {rho:.2f}, ratio {ratio:.1f} vs {want:.1f} for pure timing): "
                           f"timing jitter ~{jitter_us:.2f} us", "pilot common-phase / timing tracking, shorter "
                           "symbols; EQ cannot fix it"))
    else:
        pm_hi = float(pm[usable][-1])
        sev = "warn" if pm_hi > 0.1 else "ok"
        out.append(Finding("phase", sev, f"phase wander not timing-like (slope {pm_slope:.2f}, far-pair correlation "
                           f"{rho:.2f}); PM rms {pm.min():.3f}-{pm.max():.3f} rad",
                           "independent per-tone noise: not fixed by pilot tracking" if sev != "ok" else ""))

    # noise beside a lone tone vs the quiet floor: signal-dependent broadband noise
    rise = [m["noise_rise_db"] for m in tones if m["noise_rise_db"] is not None]
    if rise:
        rmed = float(np.median(rise))
        harmonic_clean = worst_thd < -35
        sev = "bad" if rmed > 15 and harmonic_clean else "warn" if rmed > 6 else "ok"
        out.append(Finding("codec", sev, f"a lone tone raises the in-band noise {rmed:.1f} dB above the quiet floor"
                           + (" with clean harmonics: codec quantisation noise" if sev != "ok" and harmonic_clean
                              else ""),
                           "codec-limited: SNR is set by the codec at every level; fewer bits per carrier, or a "
                           "better codec / wired path" if sev == "bad" else ""))

    # RX zero-fill gaps (digital) vs analog dropouts (BT packet loss, fades)
    runs = res.get("zero_runs", [])
    if runs:
        lens = [n for _, n in runs]
        gaps = np.diff([a for a, _ in runs]) / res["fs"] if len(runs) > 1 else np.array([])
        every = f", every ~{np.median(gaps):.2f} s" if len(gaps) else ""
        out.append(Finding("rx_gaps", "bad",
                           f"{len(runs)} zero-filled gaps in the received audio ({min(lens)}-{max(lens)} samples, "
                           f"{1000 * sum(lens) / res['fs']:.0f} ms in all{every}): an RX buffer underrun, not the "
                           f"radio. The stream gains / loses samples around them, so the clock reading "
                           f"({res['clock_ppm']:+.0f} ppm) and per-tone ppm are not trustworthy in this run",
                           "raise the RX audio bridge buffer (e.g. pacat --latency-msec=200) and run the SDR at a "
                           "rate it streams cleanly (RTL-SDR: 250k, 900k, 1.2M, 2.4M S/s; rtl_fm -s 240k captures "
                           "at 1.2 MS/s = 25 x 48 kHz)"))
    else:
        out.append(Finding("rx_gaps", "ok", "no zero-filled gaps in the received audio"))
    drop = sum(m["dropout_blocks"] - m.get("digital_dips", 0) for k in ("tone", "level", "pair", "far") for m in res[k])
    ms = 1000.0 * drop / BLOCK_HZ
    out.append(Finding("dropouts", "bad" if drop else "ok",
                       f"{drop} blocks ({ms:.0f} ms) with a tone >6 dB below its median, not explained by RX "
                       f"zero-fill" if drop else "no analog tone dropouts",
                       "analog gaps (BT packet loss, fades, limiter pumping): longer lead, ARQ, interleaving"
                       if drop else ""))
    return out


def _f1(v) -> str:
    return "-" if v is None else f"{v:.1f}"


def report_rows(seq: Sequence, res: dict, findings: list[Finding], prefix: str = "ls") -> list[tuple[str, str]]:
    rows = [(f"{prefix}.plan", seq.describe()), (f"{prefix}.found_at_s", f"{res['t0_s']:.3f}"),
            (f"{prefix}.clock_ppm", f"{res['clock_ppm']:.2f}"),
            (f"{prefix}.zero_runs", f"{len(res.get('zero_runs', []))} ({res.get('zero_samples', 0)} samples)"),
            (f"{prefix}.quiet", f"floor {res['quiet']['floor_band_dbfs']:.1f} dBFS rms {res['quiet']['rms_dbfs']:.1f} "
                                f"dBFS spurs {res['quiet']['spurs']}")]
    for m in res["tone"]:
        rows.append((f"{prefix}.tone.{m['f_hz'][0]:g}",
                     f"gain={m['gain_db'][0]:.1f} snr={m['snr_db']:.1f} pm={m['pm_rms_rad'][0]:.3f} "
                     f"am={m['am_rms'][0]:.3f} thd={_f1(m['thd_dbc'])} rise={_f1(m['noise_rise_db'])} "
                     f"skirt={_f1(m['skirt_dbc_hz'])} drop_ms={m['dropout_ms']:.0f}"))
    for m in res["level"]:
        rows.append((f"{prefix}.level.{m['label']}", f"gain={m['gain_db'][0]:.1f} snr={m['snr_db']:.1f} "
                                                      f"thd={_f1(m['thd_dbc'])}"))
    for m in res["pair"]:
        rows.append((f"{prefix}.pair.{m['label']}", f"imd3={_f1(m['imd3_dbc'])} snr={m['snr_db']:.1f}"))
    far = res["far"][0]
    rows.append((f"{prefix}.far.{far['label']}", f"pm_corr={far['pm_correlation']:.2f} pm_ratio={far['pm_ratio']:.2f} "
                                                f"timing_would_give={far['pm_ratio_if_timing']:.1f}"))
    for fd in findings:
        rows.append((f"{prefix}.finding.{fd.key}", f"[{fd.severity}] {fd.summary}" + (f"  FIX: {fd.fix}" if fd.fix else "")))
    return rows
