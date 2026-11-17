"""engine/calibrate.py on synthetic rooms: a known echo, noise floor, clock offset, speaker distortion and AGC
must come back out of the analysis."""
import numpy as np
import pytest

from modemgui.engine import calibrate as cal

FS = 48000


def room_ir(rt60: float, direct_ms: float = 2.0, reverb_db: float = -12.0, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(max(0.05, rt60 * 1.5) * FS)
    t = np.arange(n) / FS
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail[: int(0.001 * FS)] = 0
    tail *= 10 ** (reverb_db / 20) / np.sqrt(np.sum(tail ** 2))
    h = np.zeros(n + int(direct_ms * FS / 1000))
    d = int(direct_ms * FS / 1000)
    h[d] = 1.0
    h[d:d + n] += tail
    return 0.3 * h


def play_room(x, ir, noise_db=-55.0, ppm=0.0, drive=None, agc=None, lead_s=0.7, seed=5, speaker=None):
    """Speaker (optional tanh ``drive``), room ``ir``, clock offset ``ppm``, mic AGC ``agc``=(target, tau_s), noise."""
    y = np.asarray(x, float)
    if speaker:
        y = speaker(y)
    if drive:
        y = np.tanh(drive * y) / drive
    y = np.convolve(y, ir)
    if ppm:
        n = len(y)
        y = np.interp(np.arange(int(n / (1 + ppm * 1e-6))) * (1 + ppm * 1e-6), np.arange(n), y)
    y = np.concatenate([np.zeros(int(lead_s * FS)), y, np.zeros(int(0.5 * FS))])
    rng = np.random.default_rng(seed)
    y = y + rng.standard_normal(len(y)) * 10 ** (noise_db / 20)
    if agc:
        target, tau = agc
        g, env, out = 1.0, 1e-4, np.empty_like(y)
        a = 1 - np.exp(-1 / (tau * FS))
        for i, v in enumerate(y):
            env += a * (v * v - env)
            g += a * (target / np.sqrt(env + 1e-12) - g)
            g = min(g, 4.0)                      # a real AGC's range: ~12 dB of boost
            out[i] = v * g
        y = out
    floor = rng.standard_normal(int(1.5 * FS)) * 10 ** (noise_db / 20)
    return np.clip(y, -1, 1), floor


@pytest.fixture(scope="module")
def probe():
    return cal.make_probe(FS, 400.0, 4000.0, (400.0, 6000.0))


def woofer(x, fc=800.0, drive=6.0):
    """A small speaker's excursion limit: the content below ``fc`` goes through a tanh, the rest stays linear."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    low = np.fft.irfft(np.where(f < fc, X, 0), len(x))
    return np.tanh(drive * low) / drive + (x - low)


def test_clean_room_echo_clock_and_level(probe):
    rec, floor = play_room(probe.signal, room_ir(0.12), ppm=40.0)
    r = cal.analyse(probe, rec, floor)
    assert r.ok, r.error
    assert r.sync_db > 20
    assert abs(r.ppm + 40.0) < 5, r.ppm      # the room resampler slows the listener's clock: -40 ppm
    assert r.best_level_db == -6.0 and "still improving" in r.best_reason
    assert not r.agc_detected, (r.agc_attack_db, r.agc_recovery_s)
    snrs = [s.snr_db for s in r.steps]
    assert all(b > a for a, b in zip(snrs, snrs[1:]))          # 6 dB more TX, more SNR
    assert abs((snrs[-1] - snrs[0]) - 36) < 4
    e = r.echo
    assert 10 < e.t20_ms < 200 and e.pre_echo_db < -20, e
    assert r.rec_ceiling_db >= 20 and r.verdict == "ok", r.rows()
    rows = dict(r.rows(cur=(1024, 0.25)))
    assert float(rows["cal.css_amplitude"]) < float(rows["cal.ofdm_amplitude"])
    assert "cal.current_ceiling_db" in rows and "\n" not in "".join(rows.values())


def test_speaker_distortion_caps_the_tx_level(probe):
    rec, floor = play_room(probe.signal, room_ir(0.08), drive=12.0)
    r = cal.analyse(probe, rec, floor)
    assert r.ok, r.error
    assert r.best_level_db <= -18 and "SINAD stops improving" in r.best_reason, [
        (s.level_db, round(s.sinad_db, 1)) for s in r.steps]


def test_agc_is_detected(probe):
    rec, floor = play_room(probe.signal, room_ir(0.05), agc=(0.02, 0.25))
    r = cal.analyse(probe, rec, floor)
    assert r.ok, r.error
    assert r.agc_detected and "agc-present" in r.verdict, (r.agc_attack_db, r.agc_recovery_s)


def test_long_reverb_needs_a_longer_echo_budget(probe):
    rec, floor = play_room(probe.signal, room_ir(0.6, reverb_db=-3.0), noise_db=-70)
    r = cal.analyse(probe, rec, floor)
    assert r.ok, r.error
    c = r.echo.ceilings
    assert c[(512, 0.25)][0] < c[(4096, 0.5)][0]              # longer CP in seconds = less ISI
    assert (r.rec_n_fft, r.rec_cp) != (512, 0.25)


def test_no_probe_in_the_recording(probe):
    noise = np.random.default_rng(1).standard_normal(len(probe.signal) + FS) * 0.01
    r = cal.analyse(probe, noise, noise[:FS])
    assert not r.ok and "not found" in r.error
    r = cal.analyse(probe, noise[:FS], noise[:FS])
    assert not r.ok and "too short" in r.error


def test_listen_finds_its_own_noise_floor(probe):
    rec, floor = play_room(probe.signal, room_ir(0.1), lead_s=1.5)
    fl, why = cal.split_floor(probe, rec)
    assert not why and len(fl) >= 1.3 * FS
    r = cal.analyse(probe, rec, fl)
    assert r.ok and abs(r.floor_rms_db - (-55)) < 1.5
    rec2, _ = play_room(probe.signal, room_ir(0.1), lead_s=0.2)
    fl2, why2 = cal.split_floor(probe, np.concatenate([rec2, np.zeros(FS)]))
    assert not why2                                          # the quiet after the probe
    assert cal.split_floor(probe, rec2)[1].startswith("no 0.8s of quiet")


def test_chirp_compresses_where_spread_ofdm_does_not(probe):
    """The same woofer limit: the multitone spreads its power over every carrier and barely sags, the chirp puts all
    of it on one frequency at a time and compresses hard in the low bands -- so CSS is capped below the equal-RMS
    rule while the OFDM mask stays flat."""
    rec, floor = play_room(probe.signal, room_ir(0.08), speaker=woofer)
    r = cal.analyse(probe, rec, floor)
    assert r.ok, r.error
    bands = {round(b["fc"]): b for b in r.bands}
    lows = [b for fc, b in bands.items() if fc < 700]
    highs = [b for fc, b in bands.items() if fc > 1500]
    assert all(b["mask"] == 0.0 for b in bands.values())
    assert any(b["sweep"][-10.0] < -2.0 for b in lows) and all(abs(b["sweep"][-2.0]) < 1.0 for b in highs)
    rows = dict(r.rows())
    assert r.css_limit_db is not None and r.css_limit_db <= -20 and "compresses" in r.css_reason, r.css_reason
    assert float(rows["cal.css_amplitude"]) < float(rows["cal.css_amplitude_rms_rule"])


def test_hard_low_band_compression_gives_an_ofdm_mask(probe):
    rec, floor = play_room(probe.signal, room_ir(0.08), speaker=lambda x: woofer(x, drive=40.0))
    r = cal.analyse(probe, rec, floor)
    assert r.ok, r.error
    bands = {round(b["fc"]): b for b in r.bands}
    lows = [b for fc, b in bands.items() if fc < 700]
    highs = [b for fc, b in bands.items() if fc > 1500]
    assert any(isinstance(b["mask"], float) and b["mask"] < 0 for b in lows), [(b["fc"], b["mask"], b["comp"])
                                                                                 for b in lows]
    assert all(b["mask"] == 0.0 for b in highs), [(b["fc"], b["mask"], b["comp"]) for b in highs]
    assert dict(r.rows())["cal.ofdm_tx_mask_db"] != "flat"


def test_hot_chirp_harmonics_limit_css(probe):
    rec, floor = play_room(probe.signal, room_ir(0.08), drive=2.5)
    r = cal.analyse(probe, rec, floor)
    assert r.ok, r.error
    assert r.sweep.thd_db[-2.0] > r.sweep.thd_db[-20.0] + 10, r.sweep.thd_db
    assert r.css_limit_db is not None and r.css_limit_db < -2, (r.css_limit_db, r.css_reason)


def test_noise_only_while_playing_does_not_stop_the_climb(probe):
    """m6600 -> Mac 2026-11-17: 11-15 dB more noise while the probe plays than in the silent floor (the MacBook mic's
    noise gate opening; a sender's amp hiss looks the same).  SINAD then sits flat at the bottom
    steps; the climb must start where the probe is heard, not stop at -42 dBFS."""
    rng = np.random.default_rng(9)
    rec, floor = play_room(probe.signal, room_ir(0.08), noise_db=-62.0)
    lead = int(0.7 * FS)
    hiss = np.zeros(len(rec))
    hiss[lead:lead + len(probe.signal)] = rng.standard_normal(len(probe.signal)) * 10 ** (-48 / 20)
    r = cal.analyse(probe, np.clip(rec + hiss, -1, 1), floor)
    assert r.ok, r.error
    assert r.best_level_db == -6.0, [(s.level_db, round(s.snr_db, 1), round(s.sinad_db, 1)) for s in r.steps]
    assert r.steps[-1].snr_db - r.steps[-1].sinad_db > 6       # the extra noise shows as SINAD well below SNR
    assert r.rx_gate_db > 6 and "rx-noise-gate" in r.verdict


def test_clean_room_has_no_gate(probe):
    rec, floor = play_room(probe.signal, room_ir(0.12))
    r = cal.analyse(probe, rec, floor)
    assert r.rx_gate_db is not None and abs(r.rx_gate_db) < 3 and "rx-noise-gate" not in r.verdict
