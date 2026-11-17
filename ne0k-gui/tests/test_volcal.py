"""engine/volcal.py on synthetic speakers: a volume slider in dB, a speaker that saturates past a drive level,
a fixed-volume device and a clipping mic must each come back out of the sweep analysis."""
import numpy as np
import pytest

from modemgui.engine import volcal

FS = 48000


def slider_db(v):
    """A typical OS curve: ~-40 dB at 0.2 rising to 0 dB at 1.0."""
    return 50.0 * np.log10(max(v, 1e-3))


def speaker_room(x, v, *, sat=None, fixed=False, mic_gain=1.0, noise_db=-60.0, seed=0, lead_s=0.5):
    """OS volume v -> speaker (tanh saturation at ``sat`` of full drive) -> 0.3 path loss -> mic (clips at 1)."""
    g = 1.0 if fixed else 10 ** (slider_db(v) / 20)
    y = np.asarray(x, float) * g
    if sat:
        y = sat * np.tanh(y / sat)
    y = 0.3 * y * mic_gain
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(int(lead_s * FS)), y, np.zeros(int(0.4 * FS))])
    y += rng.standard_normal(len(y)) * 10 ** (noise_db / 20)
    idle = rng.standard_normal(int(volcal.IDLE_S * FS)) * 10 ** (noise_db / 20)
    return np.clip(y, -1, 1), idle


def sweep(burst, **kw):
    steps = []
    for v in volcal.volume_grid(0.2, 1.0, 9):
        rec, idle = speaker_room(burst.signal, v, **kw)
        steps.append(volcal.measure_step(burst, rec, idle, v, None if kw.get("fixed") else slider_db(v)))
    return volcal.analyse_volume(steps, burst.level_db)


@pytest.fixture(scope="module")
def burst():
    return volcal.make_burst(FS, 400.0, 4000.0)


def test_clean_speaker_recommends_top(burst):
    r = sweep(burst)
    assert r.ok and r.control_ok
    assert r.control_db == pytest.approx(slider_db(1.0) - slider_db(0.2), abs=1.5)
    assert r.rec_volume == pytest.approx(1.0)
    assert "clean-to-top" in r.verdict
    s = r.steps[-1]
    assert s.gain_db == pytest.approx(20 * np.log10(0.3), abs=0.5)
    assert s.tone_gain_db == pytest.approx(20 * np.log10(0.3), abs=0.5)
    assert s.thd_db < -40


def test_saturating_speaker_finds_knee(burst):
    # saturates at 0.1 of full drive: the -6 dBFS multitone (peak 0.5) clips from about -14 dB of slider (v ~ 0.5)
    r = sweep(burst, sat=0.1)
    assert r.ok and r.control_ok
    assert r.rec_volume is not None and 0.3 <= r.rec_volume <= 0.6, r.rows()
    loud = r.steps[-1]
    assert loud.thd_db > -20 and loud.sinad_db < r.best_sinad_db - 6
    assert "clean-to-top" not in r.verdict


def test_fixed_volume_device_flags_no_control(burst):
    r = sweep(burst, fixed=True)
    assert r.ok
    assert not r.control_ok and "no-volume-control" in r.verdict


def test_mic_clipping_caps_volume(burst):
    r = sweep(burst, mic_gain=8.0)
    assert "mic-clips" in r.verdict
    clip_v = min(s.volume for s in r.steps if s.clipped)
    assert r.rec_volume is not None and r.rec_volume < clip_v


def test_not_heard(burst):
    steps = []
    for v in volcal.volume_grid(0.2, 1.0, 3):
        rng = np.random.default_rng(1)
        rec = rng.standard_normal(len(burst.signal) + FS) * 1e-3
        steps.append(volcal.measure_step(burst, rec, rec[:FS // 2], v))
    r = volcal.analyse_volume(steps)
    assert not r.ok and r.verdict == "not-heard"


def test_rows_render(burst):
    rows = dict(sweep(burst, sat=0.1).rows())
    assert rows["vol.ok"] == "yes" and "vol.rec_volume" in rows and "vol.step100" in rows


def test_real_m6600_sweep():
    """2026-11-17 sweep on the m6600 laptop (Pulse, built-in speaker + mic): SINAD plateaus ~38 dB from 0.7 while the
    tone's THD+N climbs from -26.9 to -20.2 dB: speaker distortion onset, so 0.7, not the top."""
    real = [(0.2, -41.9, -48.5, 5.3, 5.9, 0.6), (0.3, -31.4, -39.2, 15.1, 16.1, -11.9),
            (0.4, -23.9, -31.6, 21.9, 21.6, -17.3), (0.5, -18.1, -25.6, 27.6, 28.7, -23.8),
            (0.6, -13.3, -21.0, 32.8, 33.4, -26.8), (0.7, -9.3, -16.8, 37.1, 37.3, -26.9),
            (0.8, -5.8, -13.7, 39.8, 34.5, -23.7), (0.9, -2.8, -10.9, 39.9, 37.9, -22.5),
            (1.0, 0.0, -7.9, 43.7, 37.9, -20.2)]
    st = [volcal.VolStep(v, d, heard=True, gain_db=g, snr_db=sn, sinad_db=si, thd_db=th) for v, d, g, sn, si, th in real]
    r = volcal.analyse_volume(st)
    assert r.control_ok and r.rec_volume == pytest.approx(0.7)
    assert "distortion onset" in r.rec_reason and r.verdict == "ok"


def test_real_mac_sweep():
    """2026-11-17 MacBook Air (CoreAudio, built-in speaker + mic): gain rises 0.75 dB per reported slider dB (a scale
    mismatch, not compression); multitone SINAD peaks 42 dB at 0.5-0.6 and THD+N climbs from 0.8."""
    real = [(0.2, -35.1, -22.7, 31.4, 31.5, -31.2), (0.3, -28.7, -17.9, 37.7, 36.1, -36.6),
            (0.4, -23.3, -13.9, 40.5, 39.2, -39.2), (0.5, -18.6, -10.4, 44.4, 42.0, -41.9),
            (0.6, -14.3, -7.4, 47.5, 42.0, -45.4), (0.7, -10.4, -4.3, 51.1, 40.0, -45.6),
            (0.8, -6.7, -1.0, 53.6, 36.4, -43.8), (0.9, -3.3, 1.8, 56.5, 33.1, -39.5),
            (1.0, 0.0, 5.4, 60.8, 30.1, -34.5)]
    st = [volcal.VolStep(v, d, heard=True, gain_db=g, snr_db=sn, sinad_db=si, thd_db=th) for v, d, g, sn, si, th in real]
    r = volcal.analyse_volume(st)
    assert r.slider_slope == pytest.approx(0.75, abs=0.1)
    assert r.rec_volume == pytest.approx(0.6)
    assert "compression" not in r.rec_reason
