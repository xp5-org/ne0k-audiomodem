# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Line-probe comb (engine/lineprobe.py): the math that keeps FFT bins from bleeding into each other, the layout
that follows the UI's band and the path's sample rates, and the analyzer's discriminators (clock offset, noise
vs distortion vs timing jitter).  V.34 L1/L2 ("T-REC-V.34-199802-I!!PDF-E-2.pdf page 31") is a reference case."""
import itertools
import math

import numpy as np
import pytest

from modemgui.engine import lineprobe as lp

# (rates, spacing, band) -- narrow to 20 kHz wide, single and mixed sample rates
LAYOUTS = [
    ((48000,), 150.0, (400.0, 9000.0)),
    ((44100,), 150.0, (150.0, 20000.0)),
    ((44100, 48000), 150.0, (150.0, 20000.0)),
    ((44100, 48000), 75.0, (1000.0, 1600.0)),            # narrow
    ((8000,), 100.0, (200.0, 2700.0)),                    # ham band at 8 kHz
    ((8000, 44100, 48000), 100.0, (300.0, 3400.0)),       # G.711 leg in the path
]


def _plan(rates, spacing, band, M):
    return lp.plan_comb(band[0], band[1], rates=rates, spacing_hz=spacing, window_periods=M)


@pytest.mark.parametrize("layout,M", list(itertools.product(LAYOUTS, (1, 2, 4))))
def test_no_bin_bleeds_anywhere(layout, M):
    """Every DFT bin that is not a tone bin reads at the float64 floor, at every rate, for any window start."""
    rates, spacing, band = layout
    try:
        plan = _plan(rates, spacing, band, M)
    except ValueError as ex:                              # narrow band with M=1 has no empty bin: refused
        assert M == 1 and "window_periods" in str(ex)
        return
    rng = np.random.default_rng(7)
    for fs in rates:
        P = plan.periods[fs]
        assert P * plan.spacing_hz == fs                   # integer period, exactly
        L = M * P
        x = lp.synth(plan, fs, n_periods=M * 3 + 2)
        a2 = plan.tone_amplitude(0.5) ** 2
        for start in rng.integers(0, len(x) - L, size=4):
            X = np.fft.rfft(x[start:start + L]) * (2.0 / L)
            p = np.abs(X) ** 2
            tone = np.zeros(len(p), bool)
            tone[plan.tones * M] = True
            assert np.allclose(p[tone], a2, rtol=1e-9)
            assert 10 * np.log10(p[~tone].max() / a2 + 1e-300) < -240, (fs, start)


@pytest.mark.parametrize("layout", LAYOUTS[:3])
def test_float32_audio_floor(layout):
    """The same comb as float32 audio (what the audio pipe carries): leakage stays below -120 dBc."""
    plan = _plan(*layout, 2)
    for fs in layout[0]:
        L = 2 * plan.periods[fs]
        x = lp.synth(plan, fs, 6).astype(np.float32).astype(np.float64)
        p = np.abs(np.fft.rfft(x[:L]) * (2.0 / L)) ** 2
        tone = np.zeros(len(p), bool)
        tone[plan.tones * 2] = True
        assert 10 * np.log10(p[~tone].max() / p[tone].mean()) < -120


def test_spacing_must_give_integer_period_at_every_rate():
    with pytest.raises(ValueError) as ex:
        lp.plan_comb(400, 9000, rates=(44100, 48000), spacing_hz=160)
    assert "150" in str(ex.value)                         # names the valid spacing nearby
    assert 150.0 in lp.valid_spacings((44100, 48000), 160)
    with pytest.raises(ValueError):
        lp.plan_comb(200, 2700, rates=(8000,), spacing_hz=150)      # 8000/150 is not an integer
    assert lp.plan_comb(200, 2700, rates=(8000,), spacing_hz=100).periods == {8000: 80}
    assert all((300 / s).is_integer() for s in lp.valid_spacings((44100, 48000)))


def test_band_follows_ui_and_clamps_at_nyquist():
    p = lp.plan_comb(150, 30000, rates=(44100,), spacing_hz=150)
    assert p.k_hi * 150 < 22050 and p.k_hi == 294 // 2 - 1
    p = lp.plan_comb(1000, 1300, rates=(48000,), spacing_hz=150, window_periods=2)
    assert list(p.tone_hz) == [1050, 1200] and len(p.omitted) == 0
    with pytest.raises(ValueError):                       # narrow, nothing omitted, no gap bins
        lp.plan_comb(1000, 1300, rates=(48000,), spacing_hz=150, window_periods=1)
    with pytest.raises(ValueError):
        lp.plan_comb(1000, 1040, rates=(48000,), spacing_hz=150, window_periods=2)   # no slot in band


def test_v34_l1_reference_case():
    """Table 17 reproduced exactly as a parameter set, not a mode."""
    p = lp.plan_comb(150, 3750, rates=(48000,), spacing_hz=150, omit_hz=lp.V34_L1_OMIT_HZ,
                     phases_deg=lp.V34_L1_PHASES_DEG)
    assert sorted(p.tone_hz) == sorted(lp.V34_L1_PHASES_DEG)
    assert list(p.omitted_hz) == list(lp.V34_L1_OMIT_HZ)
    assert p.periods[48000] == 320                         # 150 Hz repetition, integer at 48 kHz
    assert "h2:3" in p.catches[6] and "h2:4" in p.catches[8]   # 900 = 2*450, 1200 = 2*600
    fs = 48000
    x = lp.synth(p, fs, 1, peak=1.0)
    t = np.arange(320) / fs
    a = p.tone_amplitude(1.0)
    ref = sum(a * np.cos(2 * np.pi * f * t + math.radians(d)) for f, d in lp.V34_L1_PHASES_DEG.items())
    assert np.max(np.abs(x - ref)) < 1e-12


def test_newman_phases_keep_crest_low():
    p = lp.plan_comb(150, 20000, rates=(44100, 48000), spacing_hz=150)
    flat = lp._crest(p.tones.astype(float), np.zeros(len(p.tones)))
    assert p.crest < 1.9 < flat                           # ~4.5 dB polished (Table 17 itself: 3.96 dB)
    x = lp.synth(p, 48000, 2, peak=0.5)
    assert np.max(np.abs(x)) <= 0.5 + 1e-9


def test_auto_omit_catches_harmonics_and_spreads():
    p = lp.plan_comb(300, 9000, rates=(48000,), spacing_hz=150, omit_every=8)
    assert {4, 6, 9} <= set(p.omitted)                    # 2nd/3rd harmonics of 300 and 450 Hz
    assert "h2:2" in p.catches[4] and "h3:2" in p.catches[6] and "h3:3" in p.catches[9]
    gaps = np.diff(np.concatenate([[p.k_lo - 1], p.omitted, [p.k_hi + 1]]))
    assert gaps.max() <= 9                                 # an empty slot at least every 8 tones


def test_clock_offset_leaks_predictably_and_the_fit_removes_it():
    p = lp.plan_comb(400, 9000, rates=(48000,), spacing_hz=150, window_periods=2)
    x = lp.synth(p, 48000, 40, freq_scale=1 + 100e-6)
    r = lp.analyze(p, x, 48000)
    assert abs(r["clock_ppm"] - 100) < 0.5
    assert r["empty_raw_dbc"].max() > -100                # the leakage is real...
    assert np.all(r["empty_raw_dbc"] <= r["empty_leak_bound_dbc"] + 0.1)   # ...and bounded as predicted
    assert r["empty_dbc"].max() < -100 and r["gap_dbc"].max() < -100       # the residual is clean


def test_noise_floor_reads_what_was_added():
    p = lp.plan_comb(400, 9000, rates=(48000,), spacing_hz=150, window_periods=4)
    fs, sigma = 48000, 1e-3
    x = lp.synth(p, fs, 4 * 40) + np.random.default_rng(3).normal(0, sigma, 4 * 40 * 320)
    r = lp.analyze(p, x, fs)
    ref = np.mean(10 ** (r["tone_db"] / 10))
    want = 10 * np.log10(4 * sigma ** 2 / r["window_samples"] / ref)
    assert abs(r["floor_dbc"] - want) < 1.0
    assert abs(r["clock_ppm"]) < 1


def _empty_mean_dbc(p, x, fs):
    r = lp.analyze(p, x, fs)
    return float(10 * np.log10(np.mean(10 ** (r["empty_dbc"] / 10)))), r


def test_level_step_tells_distortion_from_noise():
    """L1 (+6 dB) vs L2 (nominal): cubic distortion in the empty slots rises ~12 dB relative to the tones,
    additive noise falls 6 dB."""
    p = lp.plan_comb(400, 9000, rates=(48000,), spacing_hz=150, window_periods=2)
    fs = 48000
    nl = lambda v: v - 0.3 * v ** 3
    lo, _ = _empty_mean_dbc(p, nl(lp.synth(p, fs, 20, peak=0.25)), fs)
    hi, _ = _empty_mean_dbc(p, nl(lp.synth(p, fs, 20, peak=0.5)), fs)
    assert 10.0 < hi - lo < 14.0
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 1e-4, 20 * 320)
    lo, _ = _empty_mean_dbc(p, lp.synth(p, fs, 20, peak=0.25) + noise, fs)
    hi, _ = _empty_mean_dbc(p, lp.synth(p, fs, 20, peak=0.5) + noise, fs)
    assert -7.0 < hi - lo < -5.0


def _snr_slope(r):
    """dB per decade of tone SNR against frequency."""
    return float(np.polyfit(np.log10(r["tone_hz"]), r["tone_snr_db"], 1)[0])


def test_timing_jitter_is_told_apart_from_noise():
    """The core question: is the SNR lower at high tones because of timing/phase jitter?"""
    p = lp.plan_comb(400, 9000, rates=(48000,), spacing_hz=150, window_periods=2)
    fs, W = 48000, 60
    L = 2 * 320
    rng = np.random.default_rng(11)
    tau = np.repeat(rng.normal(0, 0.5e-6, W), L)          # 0.5 us rms, constant within a window
    r = lp.analyze(p, lp.synth(p, fs, 2 * W, jitter_s=tau), fs)
    assert r["timing_fraction"] > 0.9
    assert abs(r["timing_jitter_us"] - 0.5) < 0.15
    assert -23 < _snr_slope(r) < -17                       # phase error ~ f: SNR falls 20 dB/decade

    x = lp.synth(p, fs, 2 * W) + rng.normal(0, 3e-4, 2 * W * 320)
    r = lp.analyze(p, x, fs)
    assert r["timing_fraction"] < 0.2
    assert abs(_snr_slope(r)) < 3                          # white noise: flat SNR


def test_steady_span_finds_the_comb_in_a_recording_and_reports_it():
    """LINEPROBE LISTEN's path: silence + comb sent at 44.1 kHz (BT) resampled to 48 kHz with noise + silence; the span is found, trimmed, and the report has one row per tone and empty slot."""
    from scipy.signal import resample_poly
    p = lp.plan_comb(150, 15000, rates=(44100, 48000), spacing_hz=150, window_periods=2)
    tx = lp.synth(p, 44100, 2 * 300, peak=0.5)                       # 4 s at 44.1 kHz
    rx = resample_poly(tx, 160, 147)                                  # -> 48 kHz
    rng = np.random.default_rng(3)
    x = np.concatenate([np.zeros(48000), rx, np.zeros(48000)]) + 1e-4 * rng.standard_normal(len(rx) + 96000)
    s0, s1 = lp.steady_span(p, x, 48000)
    assert 48000 <= s0 < 48000 + 0.4 * 48000 and 48000 + len(rx) - 0.4 * 48000 < s1 <= 48000 + len(rx)
    res = lp.analyze(p, x[s0:s1], 48000)
    assert abs(res["clock_ppm"]) < 0.5                                # (clock offset: test_clock_offset_*)
    rows = dict(lp.report_rows(p, res))
    assert sum(k.startswith("lp.tone.") for k in rows) == len(p.tones)
    assert sum(k.startswith("lp.empty.") for k in rows) == len(p.omitted)
    assert float(rows["lp.snr_db_median"]) > 40 and float(rows["lp.floor_dbc"]) < -50
    assert "gap=" in rows[f"lp.tone.{p.tone_hz[0]:g}"]


def test_steady_span_refuses_silence():
    p = lp.plan_comb(300, 3000, rates=(48000,), window_periods=2)
    with pytest.raises(ValueError, match="no steady comb|too short"):
        lp.steady_span(p, 1e-5 * np.random.default_rng(0).standard_normal(48000), 48000)


def test_drifting_clock_is_reported_as_drift_not_jitter():
    """A clock whose offset changes over the capture (seen on the BT/FM link: -529 ppm in one run, -13 in the next)
    bends the timing curve; tracking follows it, so it goes to timing_drift_us and the jitter verdict stays clean."""
    p = lp.plan_comb(150, 15000, rates=(48000,), window_periods=2)
    n = 300 * p.periods[48000] * 2
    t = np.arange(n) / 48000
    x = lp.synth(p, 48000, 600, jitter_s=2e-6 * (t / t[-1]) ** 2)        # 2 us of curved drift over the capture
    x += 1e-5 * np.random.default_rng(5).standard_normal(n)
    r = lp.analyze(p, x, 48000)
    # the straight part of the curve is a constant offset (0.5 ppm here); the bend left over is the drift
    assert abs(r["clock_ppm"] - 0.5) < 0.05 and 0.3 < r["timing_drift_us"] < 0.7 and r["timing_jitter_us"] < 0.05 and r["timing_fraction"] < 0.5
