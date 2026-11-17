# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""OS output-volume sweep: which "system volume" a node's speaker wants, and whether the modem can set it at all.

Gain staging.  Two knobs set how loud the modem is: the modem's own digital ``amplitude`` (float32 all the way to the
OS, linear, exact) and the OS output volume (an analog codec gain, a digital scaler, a speaker-protection limiter --
the curve is per device and unknown).  The modem samples lose nothing to software scaling: even a 16-bit DAC at
amplitude 0.2 keeps a quantisation floor ~80 dB down, far below the 25-40 dB an acoustic link has.  What matters is
WHERE it distorts: a volume so high that the speaker (or its limiter) saturates before the digital signal is near
full scale wastes the clean digital range; a volume so low that the modem must run at 1.0 leaves no headroom for
OFDM peaks.  So the OS volume is the calibrated value and the digital level stays in a known window.

The sweep holds the digital level fixed at the top of that window (``level_db``, default -6 dBFS peak: the
operating level then sits a few dB below it) and steps the OS volume from quiet to loud.  Each step plays a burst::

    [sync chirp] [multitone, OFDM-like crest, peak = level] [pure tone, equal RMS = a CSS chirp's crest]

and records a stretch of silence first (the idle floor at that volume).  Per step:

* gain      -- received tone power vs sent; the curve against the slider proves the volume control works (a device
               with fixed volume, or a mixer that takes the value and ignores it, shows a flat curve).
* SINAD     -- multitone power vs the empty grid bins between its tones (noise + distortion + intermodulation),
               the same noise-power-ratio layout as calibrate.py.  Rises with volume while noise-limited, peaks,
               then falls as the speaker distorts: the bell curve.
* THD       -- the pure tone's harmonics 2..5: the constant-envelope (CSS) view of the same speaker.
* compression -- measured gain step vs the dB step the OS reports for the slider (when it reports one).
* settle    -- multitone level at its end vs its start (an AGC pumping on our own signal).
* rx peak / clipped samples -- mic overload, which looks like speaker distortion but is the input side.

Recommendation: the highest volume whose SINAD is within ``SINAD_TOL_DB`` of the best, with THD <= ``THD_MAX_DB``,
compression <= ``COMP_DB`` and no mic clipping.  At that volume the digital level ``level_db`` is the loudest clean
one, so the modem's amplitudes come from ``CALIBRATE SELF`` run there (or, roughly, ``level_db`` - 3 dB for OFDM).

No Qt, numpy only (tests/test_volcal.py drives it with synthetic speakers).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from .calibrate import _a, _db, multitone, seg_spectrum, sync_chirp, tone_grid, xcorr_peak

LEVEL_DB = -6.0                    # multitone peak, dBFS, held for the whole sweep
SYNC_DB = -15.0
MT_S, TONE_S = 0.8, 0.6
GAP_S, TAIL_S = 0.15, 0.35
IDLE_S = 0.6
SINAD_TOL_DB = 1.0
THD_MAX_DB = -20.0
COMP_DB = 1.5
THD_RISE_DB = 3.0                  # THD+N this far above its minimum = distortion onset (the noise-limited low end
                                   # has high THD+N too, so only steps after the minimum count)
REF_SNR_DB = 20.0                  # compression is measured from the first step this far above the floor
MIN_CONTROL_DB = 6.0               # the loudest step must be at least this much louder than the quietest
SYNC_MIN_DB = 25.0


@dataclasses.dataclass
class Burst:
    signal: np.ndarray
    fs: int
    band: tuple[float, float]
    n_seg: int
    bins: np.ndarray
    empty: np.ndarray
    sync: np.ndarray
    mt: tuple[int, int]
    tone: tuple[int, int]
    tone_bin: int
    level_db: float
    tone_peak: float
    tx_bin_p: float                  # sent power per multitone bin (seg_spectrum units)
    tx_tone_p: float                 # sent power in the pure tone's bin


def make_burst(fs: int = 48000, f0: float = 400.0, f1: float = 4000.0, level_db: float = LEVEL_DB) -> Burst:
    n_seg = 4096 if fs <= 48000 else 8192
    bins, empty = tone_grid(fs, f0, f1, n_seg)
    pk = _a(level_db)
    one = multitone(fs, bins, n_seg, 2 * n_seg / fs, 1.0)[n_seg // 2:n_seg // 2 + n_seg]
    rms = math.sqrt(float(np.mean(one ** 2)))
    tone_peak = min(1.0, pk * rms * math.sqrt(2))      # same RMS as the multitone: a chirp at the CSS level
    # pure tone low in the band (a speaker runs out of excursion at the bottom first), bin-exact, harmonics < fs/2
    tb = int(round(max(f0 * 1.5, 600.0) * n_seg / fs))
    while tb * 5 >= n_seg // 2 and tb > 1:
        tb //= 2
    sync = sync_chirp(fs, f0, f1, _a(SYNC_DB))
    gap = np.zeros(int(GAP_S * fs))
    mt = multitone(fs, bins, n_seg, MT_S, pk)
    t = np.arange(int(TONE_S * fs))
    tone = tone_peak * np.sin(2 * np.pi * tb * t / n_seg)
    w = int(0.01 * fs)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, w))
    tone[:w] *= ramp
    tone[-w:] *= ramp[::-1]
    parts = [sync, gap]
    s_mt = sum(len(p) for p in parts)
    parts += [mt, gap]
    s_tone = sum(len(p) for p in parts)
    parts += [tone, np.zeros(int(TAIL_S * fs))]
    tx_bin_p = float(seg_spectrum(one * pk, n_seg)[bins].mean())
    tx_tone_p = _tone_powers(tone_peak * np.sin(2 * np.pi * tb * np.arange(2 * n_seg) / n_seg), n_seg, tb)[0]
    return Burst(np.concatenate(parts).astype(np.float32), fs, (f0, f1), n_seg, bins, empty, sync,
                 (s_mt, s_mt + len(mt)), (s_tone, s_tone + len(tone)), tb, level_db, tone_peak, tx_bin_p, tx_tone_p)


def _tone_powers(x: np.ndarray, n_seg: int, tb: int) -> tuple[float, float]:
    """(fundamental, harmonics 2..5) power of a pure tone on bin ``tb``: Hann frames, each summed over its lobe
    (a tone through a room does not stay phase-continuous across rectangular frames)."""
    m = len(x) // n_seg
    if not m:
        return 0.0, 0.0
    win = np.hanning(n_seg)
    P = np.mean(np.abs(np.fft.rfft(x[:m * n_seg].reshape(m, n_seg) * win, axis=1)) ** 2, axis=0)
    band = lambda b: float(P[max(0, b - 3):b + 4].sum())                   # noqa: E731
    return band(tb), sum(band(h * tb) for h in range(2, 6) if h * tb + 4 < len(P))


@dataclasses.dataclass
class VolStep:
    volume: float                    # OS slider 0..1 as read back after setting it
    volume_db: float | None          # what the OS says that slider is in dB (None: it does not say)
    heard: bool = False
    sync_db: float = 0.0
    gain_db: float = -200.0          # multitone bins received vs sent
    snr_db: float = -200.0           # multitone bins vs the idle floor at the same bins
    sinad_db: float = -200.0         # multitone bins vs the empty grid bins
    thd_db: float = 0.0              # pure tone: harmonics 2..5 vs fundamental
    tone_gain_db: float = -200.0
    rx_peak_db: float = -200.0
    clipped: int = 0
    settle_db: float = 0.0
    idle_rms_db: float = -200.0
    error: str = ""


def measure_step(burst: Burst, rec: np.ndarray, idle: np.ndarray, volume: float,
                 volume_db: float | None = None) -> VolStep:
    """One volume step: ``rec`` holds the burst somewhere (it is found by its sync chirp), ``idle`` is silence
    recorded at the same volume just before."""
    fs, n_seg, bins = burst.fs, burst.n_seg, burst.bins
    rec = np.asarray(rec, np.float64)
    idle = np.asarray(idle, np.float64)
    st = VolStep(volume, volume_db)
    st.idle_rms_db = _db(float(np.mean(idle ** 2))) if len(idle) else -200.0
    if len(rec) < len(burst.signal):
        st.error = "recording shorter than the burst"
        return st
    k, st.sync_db = xcorr_peak(rec[:len(rec) - len(burst.signal) + len(burst.sync)], burst.sync)
    if st.sync_db < SYNC_MIN_DB:
        st.error = f"burst not heard (sync {st.sync_db:.1f} dB)"
        return st
    o = int(round(k))
    if o + len(burst.signal) - int(TAIL_S * fs) > len(rec):
        st.error = "burst ran past the end of the recording"
        return st
    st.heard = True
    edge = int(0.1 * fs)
    mt = rec[o + burst.mt[0] + edge:o + burst.mt[1] - int(0.02 * fs)]
    sp = seg_spectrum(mt, n_seg)
    fl = seg_spectrum(idle, n_seg) if len(idle) >= n_seg else np.full_like(sp, 1e-30)
    tone_p = float(sp[bins].mean())
    st.gain_db = _db(tone_p / burst.tx_bin_p)
    st.snr_db = _db(tone_p / (float(fl[bins].mean()) + 1e-30))
    st.sinad_db = _db(tone_p / (float(sp[burst.empty].mean()) + 1e-30))
    whole = rec[o + burst.mt[0]:o + burst.tone[1]]
    st.rx_peak_db = 20 * math.log10(float(np.max(np.abs(whole))) + 1e-12)
    st.clipped = int(np.count_nonzero(np.abs(whole) >= 0.999))
    early = rec[o + burst.mt[0] + int(0.02 * fs):o + burst.mt[0] + int(0.12 * fs)]
    late = rec[o + burst.mt[1] - int(0.32 * fs):o + burst.mt[1] - int(0.02 * fs)]
    st.settle_db = _db(float(np.mean(late ** 2))) - _db(float(np.mean(early ** 2)))
    tn = rec[o + burst.tone[0] + edge:o + burst.tone[1] - int(0.02 * fs)]
    fund, harm = _tone_powers(tn, n_seg, burst.tone_bin)
    if fund > 0:
        st.thd_db = _db(harm / fund)
        st.tone_gain_db = _db(fund / burst.tx_tone_p)
    return st


@dataclasses.dataclass
class VolResult:
    ok: bool
    steps: list[VolStep]
    level_db: float = LEVEL_DB
    control_db: float = 0.0              # gain span between the quietest and loudest heard steps
    control_ok: bool = False
    db_per_step: list[float] = dataclasses.field(default_factory=list)
    comp_db: list[float | None] = dataclasses.field(default_factory=list)   # gain vs the fitted slider line, per step
    slider_slope: float | None = None    # measured dB per OS-reported dB
    idle_change_db: float = 0.0
    best_sinad_db: float = -200.0
    peak_volume: float | None = None     # volume of the SINAD peak
    rec_volume: float | None = None
    rec_reason: str = ""
    verdict: str = ""
    error: str = ""

    def rows(self, prefix: str = "vol") -> list[tuple[str, str]]:
        r = [(f"{prefix}.ok", "yes" if self.ok else "no")]
        if self.error:
            r.append((f"{prefix}.error", self.error))
        r.append((f"{prefix}.level_dbfs", f"{self.level_db:.0f}"))
        for i, s in enumerate(self.steps):
            vdb = "?" if s.volume_db is None else f"{s.volume_db:.1f}"
            if not s.heard:
                r.append((f"{prefix}.step{int(round(s.volume * 100)):03d}",
                          f"vol={s.volume:.2f} os_db={vdb} heard=no idle={s.idle_rms_db:.1f} {s.error}"))
                continue
            c = self.comp_db[i] if i < len(self.comp_db) else None
            r.append((f"{prefix}.step{int(round(s.volume * 100)):03d}",
                      f"vol={s.volume:.2f} os_db={vdb} gain={s.gain_db:.1f} snr={s.snr_db:.1f} sinad={s.sinad_db:.1f} "
                      f"thdn={s.thd_db:.1f} comp={'n/a' if c is None else f'{c:+.1f}'} rx_peak={s.rx_peak_db:.1f} "
                      f"clipped={s.clipped} settle={s.settle_db:+.1f} idle={s.idle_rms_db:.1f}"))
        r += [(f"{prefix}.control_db", f"{self.control_db:.1f}"),
              (f"{prefix}.control", "yes" if self.control_ok else "no"),
              (f"{prefix}.slider_slope", "n/a" if self.slider_slope is None else f"{self.slider_slope:.2f}"),
              (f"{prefix}.idle_change_db", f"{self.idle_change_db:+.1f}"),
              (f"{prefix}.best_sinad_db", f"{self.best_sinad_db:.1f}"),
              (f"{prefix}.peak_volume", "n/a" if self.peak_volume is None else f"{self.peak_volume:.2f}")]
        if self.rec_volume is not None:
            r += [(f"{prefix}.rec_volume", f"{self.rec_volume:.2f}"),
                  (f"{prefix}.rec_volume_pct", f"{round(self.rec_volume * 100)}"),
                  (f"{prefix}.ofdm_amplitude_hint", f"{_a(self.level_db - 3.0):.2f}")]
        r += [(f"{prefix}.rec_reason", self.rec_reason), (f"{prefix}.verdict", self.verdict)]
        return r


def analyse_volume(steps: list[VolStep], level_db: float = LEVEL_DB, *, sinad_tol_db: float = SINAD_TOL_DB,
                   thd_max_db: float = THD_MAX_DB, comp_db: float = COMP_DB) -> VolResult:
    """Steps in rising volume order."""
    res = VolResult(ok=False, steps=list(steps), level_db=level_db)
    heard = [s for s in steps if s.heard]
    if not heard:
        res.error = "the burst was not heard at any volume (speaker muted / wrong device / mic dead?)"
        res.verdict = "not-heard"
        return res
    res.control_db = heard[-1].gain_db - heard[0].gain_db if len(heard) > 1 else 0.0
    res.control_ok = res.control_db >= MIN_CONTROL_DB
    prev = None
    for s in steps:
        if s.heard and prev is not None and prev.heard:
            res.db_per_step.append(s.gain_db - prev.gain_db)
        if s.heard:
            prev = s
    # compression: gain against a line fitted (gain vs the OS-reported slider dB) through the first well-heard steps.
    # The OS dB scale is often off by a constant factor (CoreAudio on a MacBook: 0.75 dB per reported dB), which is
    # not compression; only falling BELOW that line as the volume rises is.
    fit = [s for s in steps if s.heard and s.snr_db >= REF_SNR_DB and s.volume_db is not None][:3]
    line = None
    if len(fit) >= 2:
        x = np.array([s.volume_db for s in fit])
        y = np.array([s.gain_db for s in fit])
        if np.ptp(x) > 1.0:
            line = np.polyfit(x, y, 1)
    res.comp_db = [float(s.gain_db - np.polyval(line, s.volume_db)) if (line is not None and s.heard
                   and s.volume_db is not None) else None for s in steps]
    res.slider_slope = None if line is None else float(line[0])
    idles = [s.idle_rms_db for s in steps if s.idle_rms_db > -200]
    res.idle_change_db = (idles[-1] - idles[0]) if len(idles) > 1 else 0.0

    best = max(heard, key=lambda s: s.sinad_db)
    res.best_sinad_db = best.sinad_db
    res.peak_volume = best.volume
    comp_at = {id(s): c for s, c in zip(steps, res.comp_db)}
    good = [s for s in heard if s.snr_db >= REF_SNR_DB] or heard
    thd_min = min(good, key=lambda s: s.thd_db)
    cands, why_stop, stop_v = [], "", None
    for s in heard:
        stop_v = s.volume
        if s.clipped:
            why_stop = f"mic clips at volume {s.volume:.2f}"
            break
        if s.volume > thd_min.volume and s.thd_db > thd_min.thd_db + THD_RISE_DB:
            why_stop = (f"tone THD+N up {s.thd_db - thd_min.thd_db:.1f} dB at volume {s.volume:.2f} "
                        f"(speaker distortion onset; best {thd_min.thd_db:.1f} dB at {thd_min.volume:.2f})")
            break
        if s.thd_db > thd_max_db and s.volume > best.volume:
            why_stop = f"tone THD+N {s.thd_db:.1f} dB at volume {s.volume:.2f}"
            break
        c = comp_at.get(id(s))
        if c is not None and c < -comp_db:
            why_stop = f"gain {c:+.1f} dB below the slider's line at volume {s.volume:.2f} (compression)"
            break
        if s.sinad_db >= best.sinad_db - sinad_tol_db and s.thd_db <= thd_max_db:
            cands.append(s)
    else:
        stop_v = None
    if not cands and why_stop:
        # stopped before the SINAD plateau: the last clean step before the stop is still the answer
        clean = [s for s in heard if s.volume < stop_v and s.thd_db <= thd_max_db and not s.clipped]
        cands = clean[-1:] if clean else []
    v = []
    if not res.control_ok:
        v.append("no-volume-control")
    if cands:
        pick = cands[-1]
        res.rec_volume = pick.volume
        top = heard[-1]
        if pick is top and not why_stop:
            res.rec_reason = (f"still clean at the loudest volume tried ({top.volume:.2f}): SINAD {pick.sinad_db:.1f} dB"
                              f"{'' if top.volume >= 0.99 else ', try a higher top'}")
            v.append("clean-to-top")
        else:
            res.rec_reason = (f"SINAD {pick.sinad_db:.1f} dB (best {best.sinad_db:.1f} dB at {best.volume:.2f})"
                              + (f"; stopped: {why_stop}" if why_stop else ""))
    else:
        res.rec_reason = why_stop or f"no step reached THD <= {thd_max_db:g} dB"
        v.append("distorts-everywhere")
    if any(s.clipped for s in heard):
        v.append("mic-clips")
    if any(s.settle_db < -3.0 for s in heard):
        v.append("agc-suspect")
    if best.sinad_db < 15:
        v.append("low-sinad")
    if len(heard) < len(steps):
        v.append("quiet-steps-unheard")
    res.verdict = ",".join(v) or "ok"
    res.ok = True
    return res


def volume_grid(lo: float, hi: float, n: int) -> list[float]:
    n = max(2, int(n))
    return [round(lo + (hi - lo) * i / (n - 1), 3) for i in range(n)]
