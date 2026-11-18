# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""The carrier grid the engine enforces (engine/ofdmgrid.py): the same geometry as the Python and C PHY builders,
the band snapped onto the carriers exactly, unusable combinations refused, and the line-probe comb laid on the
OFDM carriers with no bin bleed in the modem's own n_fft window."""
import ctypes
import itertools
import sys

import numpy as np
import pytest

from modemgui.engine import lineprobe as lp
from modemgui.engine import ofdmgrid as og
from modemgui.engine.modemcfg import OfdmConfig
from modemgui.engine.ofdm_link import OfdmLink, _imports

RATES = (44100, 48000)
FFTS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
BANDS = ((400.0, 9000.0), (300.0, 3400.0), (150.0, 20000.0), (1000.0, 1600.0), (400.0, 4000.0), (50.0, 24000.0))


def _cases():
    for fs, n, (lo, hi) in itertools.product(RATES, FFTS, BANDS):
        try:
            yield fs, n, lo, hi, og.grid(fs, n, lo, hi)
        except og.GridError:
            yield fs, n, lo, hi, None


def test_same_geometry_as_python_phy():
    _imports()
    from ofdm_modem import build_config
    seen = 0
    for fs, n, lo, hi, g in _cases():
        if g is None:
            continue
        c = build_config(n_fft=n, cp_ratio=0.25, freq_lo=lo, freq_hi=hi, pilot_spacing=8, sample_rate=fs)
        assert (c["bin_lo"], c["bin_hi"], c["n_pilots"], c["n_data_carriers"], c["cp_len"]) == \
               (g.bin_lo, g.bin_hi, g.n_pilots, g.n_data, g.cp_len), (fs, n, lo, hi)
        assert (c["freq_lo_hz"], c["freq_hi_hz"], c["sc_spacing_hz"]) == (g.freq_lo, g.freq_hi, g.spacing_hz)
        seen += 1
    assert seen > 60


def test_same_geometry_as_c_phy():
    from modemgui import native
    try:
        lib = native.load("libofdm")
    except native.NativeError as ex:
        pytest.skip(f"libofdm not built: {ex}")
    D, I = ctypes.c_double, ctypes.c_int
    lib.ofdm_cfg_build.argtypes = [ctypes.POINTER(native.OfdmCfg), I, D, D, D, I, I, I, I, I, D, D, I]
    for fs, n, lo, hi, g in _cases():
        c = native.OfdmCfg()
        rc = lib.ofdm_cfg_build(ctypes.byref(c), n, 0.25, lo, hi, 8, fs, 16, 256, 2, 2.0, 20.0, 32)
        if g is None:
            continue
        assert rc == 0
        assert (c.bin_lo, c.bin_hi, c.n_pilots, c.n_data_carriers, c.cp_len) == \
               (g.bin_lo, g.bin_hi, g.n_pilots, g.n_data, g.cp_len), (fs, n, lo, hi)
        assert (c.freq_lo_hz, c.freq_hi_hz) == (g.freq_lo, g.freq_hi)


def test_snapped_band_is_exact_and_idempotent():
    for fs, n, lo, hi, g in _cases():
        if g is None:
            continue
        assert g.freq_lo >= lo and g.freq_hi <= min(hi, fs / 2)
        again = og.grid(fs, n, g.freq_lo, g.freq_hi)
        assert (again.bin_lo, again.bin_hi) == (g.bin_lo, g.bin_hi) and not again.snapped
        assert g.freq_lo == g.bin_lo * fs / n                  # exact binary fraction, no drift


def test_bt_fm_link_band_is_reported_as_it_runs():
    g = og.grid(48000, 256, 400, 9000)
    assert (g.freq_lo, g.freq_hi, g.n_occupied, g.n_data) == (562.5, 9000.0, 46, 40)   # 9000 = carrier 48 exactly; 400 moves up to 3 x 187.5
    assert g.snapped and "asked 400-9000Hz" in g.describe()


@pytest.mark.parametrize("args,why", [
    ((48000, 16, 400, 9000), "outside 32-4096"),
    ((48000, 8, 400, 20000), "outside 32-4096"),
    ((48000, 100, 400, 9000), "not a power of two"),
    ((48000, 32, 400, 4000), "need 2 data carriers"),      # two 1.5 kHz carriers, one is the pilot
    ((48000, 1024, 5000, 4000), "not below"),
    ((44100, 256, 22000, 24000), "need 2 data carriers"),   # all above Nyquist
])
def test_unusable_combinations_are_refused(args, why):
    with pytest.raises(og.GridError, match=why):
        og.grid(*args)


def test_validate_set_refuses_and_passes():
    cfg = OfdmConfig(n_fft=256, freq_lo=400.0, freq_hi=4000.0)
    with pytest.raises(og.GridError):
        og.validate_set(cfg, "n_fft", 32)                     # 400-4000 at 32 has one data carrier
    assert og.validate_set(cfg, "n_fft", 512).n_fft == 512
    assert og.validate_set(cfg, "amplitude", 0.9).amplitude == 0.9   # not a grid key: no check


def test_link_runs_on_the_snapped_band_and_refuses_a_dead_grid():
    lines = []
    link = OfdmLink(None, OfdmConfig(n_fft=256, freq_lo=400.0, freq_hi=9000.0), log=lines.append)
    assert (link.cfg.freq_lo, link.cfg.freq_hi) == (562.5, 9000.0)
    assert link._band == (562.5, 9000.0)
    assert any(l.startswith("ofdm grid: fs=48000 n_fft=256") for l in lines)
    with pytest.raises(ValueError, match="ofdm grid"):
        OfdmLink(None, OfdmConfig(n_fft=32, freq_lo=400.0, freq_hi=4000.0))


@pytest.mark.parametrize("fs,n,band,every", [
    (48000, 256, (400, 9000), 1), (48000, 1024, (400, 9000), 4), (44100, 512, (150, 20000), 2),
    (44100, 64, (400, 20000), 1), (48000, 32, (400, 20000), 1), (48000, 4096, (300, 3400), 8),
])
def test_comb_sits_on_ofdm_carriers_without_bleed(fs, n, band, every):
    g = og.grid(fs, n, *band)
    p = lp.plan_for_grid(g, every=every)
    bins = p.tones * every                                    # OFDM bins the tones occupy
    assert set(bins) <= set(range(g.bin_lo, g.bin_hi + 1))
    assert np.array_equal(p.tone_hz, bins * g.spacing_hz)     # tone frequency == carrier frequency, exactly
    assert p.window_periods * p.periods[fs] == n              # analysis window == the modem's own FFT
    x = lp.synth(p, fs, 3 * p.window_periods)
    start = n // 3
    X = np.fft.rfft(x[start:start + n]) * (2.0 / n)
    pw = np.abs(X) ** 2
    on = np.zeros(len(pw), bool)
    on[bins] = True
    assert np.allclose(pw[on], p.tone_amplitude(0.5) ** 2, rtol=1e-9)
    assert 10 * np.log10(pw[~on].max() / pw[on].mean() + 1e-300) < -240


def test_comb_every_must_divide_n_fft():
    g = og.grid(48000, 256, 400, 9000)
    for bad in (3, 0, 128):
        with pytest.raises(ValueError):
            lp.plan_for_grid(g, every=bad)
