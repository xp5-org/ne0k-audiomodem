# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Line probe stages 0-2 (engine/linesweep.py): each synthetic hardware fault must be found by its own finding and
must not trip the others.  The faults are the rows of LINE_PROBE.md's "Why" table plus what the BT/FM link showed
(drifting clock, signal-proportional codec noise, dropouts, idle-channel spurs)."""
import numpy as np
import pytest
from scipy.signal import butter, lfilter, resample

from modemgui.engine import linesweep as ls

FS = 48000
SEQ = ls.plan_sweep(150, 15000, peak=0.5)
NOISE = 1e-3                     # ~47 dB in-band SNR for the sweep tones: additive noise dominates unless a fault adds


def _rx(y, noise=NOISE, seed=0, pre=0.6, post=0.8):
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.zeros(int(pre * FS)), y, np.zeros(int(post * FS))])
    return x + noise * rng.standard_normal(len(x))


def _run(x, fs=FS):
    res = ls.analyze(SEQ, x, fs)
    return res, {f.key: f for f in ls.classify(res)}


def _sev(fd, *keys):
    return {k: fd[k].severity for k in keys}


ALL = ("clock", "floor", "tilt", "noise", "nonlinearity", "phase", "codec", "dropouts", "rx_gaps")


def test_plan_is_sample_exact_at_both_rates():
    for s in SEQ.segments:
        for fs in (44100, 48000):
            assert abs(s.start_s * fs - round(s.start_s * fs)) < 1e-6 and abs(s.dur_s * fs - round(s.dur_s * fs)) < 1e-6
        for f, _ in s.tones:
            assert f % ls.GRID_HZ == 0
    assert SEQ.dur_s < 20                                   # fits a LISTEN (29 s) with room for the SEND delay
    with pytest.raises(ValueError):
        ls.plan_sweep(1000, 1200)


def test_clean_channel_flags_nothing():
    res, fd = _run(_rx(ls.synth(SEQ, FS)))
    assert all(fd[k].severity in ("ok", "info") for k in ALL), {k: (fd[k].severity, fd[k].summary) for k in ALL}
    assert fd["noise"].summary.startswith("SNR follows the level")
    assert abs(res["clock_ppm"]) < 1


def test_bt_rates_and_a_large_clock_offset_are_measured():
    """TX at 44.1 kHz, RX at 48 kHz, with the -529 ppm the BT/FM link showed (tones read low)."""
    tx = ls.synth(SEQ, 44100)
    rx = resample(tx, int(round(len(tx) * 48000 / 44100 / (1 - 529e-6))))
    res, fd = _run(_rx(rx))
    assert abs(res["clock_ppm"] + 529) < 2
    assert all(abs(m["ppm"][0] + 529) < 3 for m in res["tone"] + res["level"])   # every tone sees the same clock
    assert fd["clock"].severity == "warn"
    assert all(fd[k].severity in ("ok", "info") for k in ALL if k != "clock"), \
        {k: (fd[k].severity, fd[k].summary) for k in ALL}


def test_tilt_is_found_and_an_eq_curve_offered():
    b, a = butter(1, 2500 / (FS / 2))                       # first-order roll-off: -15 dB at 15 kHz
    res, fd = _run(_rx(lfilter(b, a, ls.synth(SEQ, FS)), noise=2e-4))
    assert fd["tilt"].severity == "warn" and "EQ" in fd["tilt"].fix
    assert fd["noise"].summary.startswith("SNR follows the level")          # additive, not signal-proportional
    assert fd["nonlinearity"].severity == "ok" and fd["codec"].severity == "ok"


def test_band_edge_is_found():
    b, a = butter(8, 8000 / (FS / 2))
    res, fd = _run(_rx(lfilter(b, a, ls.synth(SEQ, FS)), noise=2e-4))
    assert "band_edge" in fd and "15000" in fd["band_edge"].summary and "9900" in fd["band_edge"].summary


def test_timing_jitter_is_found_and_attributed_to_timing():
    rng = np.random.default_rng(7)
    n = int(round(SEQ.dur_s * FS))
    w = np.cumsum(rng.standard_normal(n // 160 + 2))                         # wander, then 20-100 Hz band
    from scipy.signal import sosfiltfilt
    sos = butter(2, [20, 100], btype="band", fs=300, output="sos")
    j = sosfiltfilt(sos, rng.standard_normal(n // 160 + 2))
    j = j / j.std() * 3e-6                                                    # 3 us rms
    tau = np.interp(np.arange(n), np.arange(len(j)) * 160, j)
    res, fd = _run(_rx(ls.synth(SEQ, FS, t_offset_s=tau), noise=1e-4))
    assert fd["phase"].severity == "warn" and "timing jitter" in fd["phase"].summary, fd["phase"].summary
    assert res["far"][0]["pm_correlation"] > 0.8
    assert fd["nonlinearity"].severity == "ok"


def test_soft_clipping_is_nonlinearity_and_rises_with_level():
    y = ls.synth(SEQ, FS)
    k = 3.0
    res, fd = _run(_rx(np.tanh(k * y) / k))
    assert fd["nonlinearity"].severity == "bad", fd["nonlinearity"].summary
    assert fd["tilt"].severity == "ok"


def test_codec_like_signal_proportional_noise():
    """Noise whose level follows the short-term signal level (a crude transform-codec model), no harmonics."""
    y = ls.synth(SEQ, FS)
    rng = np.random.default_rng(3)
    blk = 480
    env = np.sqrt(np.convolve(y ** 2, np.ones(blk) / blk, mode="same"))
    hp = butter(2, 2000 / (FS / 2), btype="high", output="ba")               # louder at high frequency
    q = lfilter(*hp, rng.standard_normal(len(y)))
    res, fd = _run(_rx(y + 0.01 * env * q / q.std(), noise=2e-5))
    assert fd["noise"].severity == "bad", fd["noise"].summary
    assert fd["codec"].severity == "bad", fd["codec"].summary
    assert fd["nonlinearity"].severity == "ok" and fd["phase"].severity in ("ok", "warn")
    assert fd["tilt"].severity == "ok"


def test_dropouts_are_counted():
    y = ls.synth(SEQ, FS)
    for t in (4.0, 7.3, 10.1):
        i = int(t * FS)
        y[i:i + int(0.03 * FS)] = 0
    res, fd = _run(_rx(y))
    assert fd["dropouts"].severity == "bad"
    assert 60 <= sum(m["dropout_ms"] for k in ("tone", "level", "pair", "far") for m in res[k]) <= 140


def test_idle_spur_is_found():
    x = _rx(ls.synth(SEQ, FS))
    x += 0.003 * np.cos(2 * np.pi * 1000 * np.arange(len(x)) / FS)          # a 1 kHz whine, always there
    res, fd = _run(x)
    assert fd["floor"].severity == "warn" and any(abs(f - 1000) < 40 for f, _ in res["quiet"]["spurs"])


def test_missing_burst_is_reported():
    with pytest.raises(ValueError, match="sync chirps not found"):
        ls.analyze(SEQ, 1e-3 * np.random.default_rng(0).standard_normal(20 * FS), FS)


def test_report_rows_cover_every_measurement():
    res, _ = _run(_rx(ls.synth(SEQ, FS)))
    rows = dict(ls.report_rows(SEQ, res, ls.classify(res)))
    assert sum(k.startswith("ls.tone.") for k in rows) == len(SEQ.find("tone"))
    assert sum(k.startswith("ls.finding.") for k in rows) >= len(ALL)


def test_resampler_alias_is_named():
    """What the BT/FM link showed: rtl_fm demodulates at 250 kS/s and resamples to 48 kHz by a non-integer ratio
    with a weak filter, so every tone also appears at 10000 -/+ f (250000 - 5 x 48000).  A crude interpolating
    resampler reproduces it; the classifier must name the fold, not call it codec noise."""
    fs_in = 250000
    y = ls.synth(SEQ, fs_in)
    y = y + 1e-4 * np.random.default_rng(1).standard_normal(len(y))
    t_out = np.arange(int(len(y) * FS / fs_in)) / FS
    x = np.interp(t_out, np.arange(len(y)) / fs_in, y)                     # no anti-alias filter: images fold
    # (linear interpolation's images are weak; add the fold explicitly at the level the link showed, -30 dBc)
    from scipy.signal import hilbert
    a = hilbert(x)
    x = x + 2 * 0.03 * np.real(a * np.cos(2 * np.pi * 10000 * t_out))      # f -> 10000 +/- f, -30 dBc each
    res, fd = _run(_rx(x, noise=1e-4))
    assert "mixing" in fd and fd["mixing"].severity == "bad", fd.get("mixing")
    assert "250000 -> 48000" in fd["mixing"].summary and abs(float(fd["mixing"].summary.split("around ")[1].split()[0]) - 10000) < 40
    assert "explained by" in fd["noise"].summary


def test_no_mixing_finding_on_clean_or_clipped_paths():
    for y in (ls.synth(SEQ, FS), np.tanh(3 * ls.synth(SEQ, FS)) / 3):
        _, fd = _run(_rx(y))
        assert "mixing" not in fd, fd.get("mixing")


def test_preemphasis_into_a_limiter_saturates_only_the_top():
    """What the BT/FM link showed with the alias fixed: 75 us pre-emphasis at the FM transmitter, a deviation
    limiter, de-emphasis at the receiver.  High tones at full level clip; the low end stays clean."""
    tau = 75e-6
    from scipy.signal import bilinear
    bp, ap = bilinear([tau, 1], [tau / 20, 1], FS)                         # pre-emphasis shelf (+~20 dB up top)
    pre = lfilter(bp, ap, ls.synth(SEQ, FS))
    lim = np.clip(pre, -1.2, 1.2)                                           # deviation limiter
    y = lfilter(ap, bp, lim)                                                # de-emphasis (exact inverse)
    res, fd = _run(_rx(y))
    nl = fd["nonlinearity"]
    assert nl.severity == "bad" and "pre-emphasis" in nl.fix, (nl.summary, nl.fix)
    assert "15000 Hz" in nl.summary and "150 Hz" not in nl.summary.replace("15000 Hz", "")


def test_rx_zero_fill_is_told_apart_from_analog_dropouts():
    """What the m6600 bridge did at pacat --latency-msec=40: exact-zero runs INSERTED into the RX stream (an
    underrun), not BT loss.  They must land in rx_gaps (which warns the clock reading is off), not in dropouts."""
    x = _rx(ls.synth(SEQ, FS))
    for t in (3.0, 5.5, 8.2, 11.0):                                  # insert 120..220 zeros (underrun)
        i = int(t * FS)
        x = np.concatenate([x[:i], np.zeros(120 + int(t * 9)), x[i:]])
    res, fd = _run(x)
    assert fd["rx_gaps"].severity == "bad" and "4 zero-filled gaps" in fd["rx_gaps"].summary
    assert fd["dropouts"].severity == "ok", fd["dropouts"].summary
    assert "not trustworthy" in fd["rx_gaps"].summary and "pacat --latency-msec" in fd["rx_gaps"].fix


def test_analog_dropouts_stay_dropouts():
    y = ls.synth(SEQ, FS)
    i = int(6.0 * FS)
    y[i:i + int(0.03 * FS)] *= 0.05                                   # a deep fade, not digital zeros
    res, fd = _run(_rx(y))
    assert fd["dropouts"].severity == "bad" and fd["rx_gaps"].severity == "ok"
