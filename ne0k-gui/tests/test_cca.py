# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""engine/cca.py: carrier detect by recognising the modem's own signals, on its real waveforms under pink noise
that is louder than any level threshold (a microphone in a room), and on the sounds that must NOT read as busy."""
import types

import numpy as np
import pytest

from modemgui.engine import cca, css_link, ofdm_link
from modemgui.engine.mesh_rx import MeshReceiver
from modemgui.engine.modemcfg import CssConfig, GeneralConfig, OfdmConfig

FS = 48000
BAND = (400.0, 4000.0)


def pink(n, seed=0):
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    y = np.fft.irfft(X / np.sqrt(f), n)
    return y / np.std(y)


def band_rms(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    return np.sqrt(2 * np.sum(np.abs(X[(f >= lo) & (f <= hi)]) ** 2)) / len(x)


def scene(sig, snr_db, band, lead_s=3.0, tail_s=3.0, noise_rms=0.01, seed=0):
    """``sig`` at ``snr_db`` (in ``band``) inside pink noise of ``noise_rms``.  Returns (audio, start, end)."""
    a = int(lead_s * FS)
    nz = pink(a + len(sig) + int(tail_s * FS), seed)
    nz /= band_rms(nz, *band)
    x = nz.copy()
    x[a:a + len(sig)] += np.asarray(sig, float) / band_rms(sig, *band) * 10 ** (snr_db / 20)
    return (x / np.std(nz) * noise_rms).astype(np.float32), a, a + len(sig)


def run(spec, x, block=1024):
    p = cca.Presence(spec)
    for i in range(0, len(x), block):
        p.push(x[i:i + block])
    return p


def split(p, a, b, bit):
    fl = p.flags(0, p.n) & bit
    blk = p.blk
    inside = fl[a // blk + 4:b // blk]
    outside = np.concatenate([fl[:a // blk], fl[b // blk + 12:]])     # the 0.2 s window and the FIR trail a loud burst
    return float(np.mean(inside > 0)), int(np.count_nonzero(outside))


@pytest.fixture(scope="module")
def H():
    h, _ = ofdm_link._imports()
    old = dict(h.HS_OVERRIDES)
    h.HS_OVERRIDES = dict(n_fft=1024, cp_ratio=0.5)
    yield h
    h.HS_OVERRIDES = old


@pytest.fixture(scope="module")
def ospec():
    return cca.spec_from_config(OfdmConfig(n_fft=1024, cp_ratio=0.5, freq_lo=BAND[0], freq_hi=BAND[1]), None)


def test_ofdm_message_and_probe_found_at_and_below_the_noise(H, ospec):
    for sig in (H.make_msg_audio(FS, H.MSG_GO, 1, b"", BAND), H.make_probe_audio(FS, BAND)):
        for snr, want in ((20, 0.95), (6, 0.9), (0, 0.7)):
            x, a, b = scene(sig, snr, BAND)
            hit, false = split(run(ospec, x), a, b, cca.OFDM)
            assert hit >= want and false == 0, (len(sig) / FS, snr, hit, false)


def test_noise_tones_and_clicks_are_not_ofdm(ospec):
    n = 60 * FS
    t = np.arange(n) / FS
    rng = np.random.default_rng(3)
    quiet = pink(n, 5) * 0.001
    for name, x in (("pink", pink(n, 4) * 0.05),
                    ("tone on a carrier bin", 0.05 * np.sin(2 * np.pi * 937.5 * t) + quiet),   # correlates 1.0 at lag 1024
                    ("key-up tone", 0.05 * np.sin(2 * np.pi * 1000.0 * t) + quiet),
                    ("clicks", quiet * 3 + (rng.random(n) < 2e-5) * 0.8),
                    ("digital silence", np.zeros(n))):
        p = run(ospec, x.astype(np.float32))
        # a full-scale click rings the whitening filter: a lone 50 ms block may flag (shorter than any frame, dropped)
        assert p.counts["ofdm"] <= (2 if name == "clicks" else 0), (name, p.stats())


def test_listed_tones_read_busy_and_others_do_not(ospec):
    import dataclasses
    spec = dataclasses.replace(ospec, tones=(1000.0, 800.0))
    t = np.arange(2 * FS) / FS                               # a 2 s key-up lead after 10 s of room
    quiet = pink(12 * FS, 6) * 0.001
    for hz, want in ((1000.0, True), (800.0, True), (1250.0, False)):
        x = quiet.copy()
        x[10 * FS:] += 0.05 * np.sin(2 * np.pi * hz * t)
        p = run(spec, x.astype(np.float32))
        assert (p.counts["tone"] >= 36) == want and (want or p.counts["tone"] == 0), (hz, p.stats())
    assert run(spec, (pink(len(t), 7) * 0.05).astype(np.float32)).counts["tone"] == 0


@pytest.mark.parametrize("sym,snr", [(0.05, 0), (0.05, -6), (0.15, -6), (0.15, -10)])
def test_css_chirps_found_below_the_noise(sym, snr):
    c = CssConfig(symbol_dur=sym)
    x, a, b = scene(css_link.encode_bytes(c, b"hello"), snr, (c.f_min, c.f_max))
    hit, false = split(run(cca.spec_from_config(None, c), x), a, b, cca.CHIRP)
    assert hit >= 0.85 and false == 0, (hit, false)


def test_noise_is_not_a_chirp():
    spec = cca.spec_from_config(None, CssConfig(symbol_dur=0.15))
    assert run(spec, (pink(120 * FS, 8) * 0.05).astype(np.float32)).counts["chirp"] == 0
    t = np.arange(20 * FS) / FS
    assert run(spec, (0.05 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)).counts["chirp"] == 0


def test_spec_follows_the_settings():
    o, c = OfdmConfig(n_fft=4096, cp_ratio=0.375, freq_lo=200, freq_hi=2700), CssConfig(f_min=400, f_max=6000)
    s = cca.spec_from_config(o, c, tones=(1000.0,))
    assert s.ofdm == ((1024, 384), (4096, 1536))          # handshake messages always ride n_fft 1024
    assert s.band == (200, 6000) and s.chirp == (400.0, 6000.0, c.symbol_dur) and s.tones == (1000.0,)
    assert cca.spec_from_config(o, None).chirp is None and cca.spec_from_config(None, c).ofdm == ()
    g = GeneralConfig()
    assert g.carrier_detect == "level"                    # wire-line behaviour is the default, unchanged
    ses = types.SimpleNamespace(sense_mode="level", sense_spec=s)
    assert cca.for_session(ses) is None
    ses.sense_mode = "signal"
    assert isinstance(cca.for_session(ses), cca.Presence)


def test_capture_level_sense_calls_the_room_busy_and_signal_sense_does_not(H, ospec):
    """The acoustic failure of 2026-11-17: room floor above carrier_db, so the level gate never closes."""
    sig = H.make_msg_audio(FS, H.MSG_GO, 1, b"", BAND)
    x, a, b = scene(sig, 10, BAND, noise_rms=10 ** (-45 / 20))          # floor -45 dBFS, threshold -55
    thr, blk = 10 ** (-55 / 20), int(0.05 * FS)
    level = ofdm_link._Capture(FS)
    signal = ofdm_link._Capture(FS, sense=cca.Presence(ospec))
    for cap in (level, signal):
        for i in range(0, len(x), 2048):
            cap.add(x[i:i + 2048])
    assert level.busy_blocks(0, None, thr, blk)[1].all() and level.busy(0.4, thr)
    busy = np.flatnonzero(signal.busy_blocks(0, None, thr, blk)[1]) * blk
    assert len(busy) and a - 2 * blk <= busy[0] <= a + 5 * blk and b - 2 * blk <= busy[-1] <= b + 5 * blk
    assert not signal.busy(0.4, thr)                                    # the tail is only room noise


def test_mesh_receiver_decodes_a_message_out_of_a_noisy_room(H, ospec):
    cfg = OfdmConfig(node_id=1, n_fft=1024, cp_ratio=0.5, freq_lo=BAND[0], freq_hi=BAND[1])
    sig = H.make_msg_audio(FS, H.MSG_GO, 7, b"", BAND)
    x, a, b = scene(sig, 12, BAND, noise_rms=10 ** (-45 / 20))
    got = []
    for mode, want in (("level", 0), ("signal", 1)):
        ses = types.SimpleNamespace(pipe=None, sense_mode=mode, sense_spec=ospec)
        r = MeshReceiver(ses, cfg, 1, log=lambda m: None)
        r.on_msg = got.append
        r.data_first = lambda: False
        r._cap.add(x)
        busy = r._busy_blocks(0, r._cap.n)
        if mode == "level":
            assert busy.all()                            # one endless burst: nothing is ever handed to a decoder
            continue
        idx = np.flatnonzero(busy)
        blk = int(0.05 * FS)
        assert 0.6 * len(sig) <= (idx[-1] - idx[0] + 1) * blk <= len(sig) + FS // 2
        r._closed(int(idx[0]) * blk, int(idx[-1] + 1) * blk)
        assert len(got) == want and got[0][1] == H.MSG_GO


# -- real rooms: 12 s of each laptop's microphone, 2026-11-17 (fan lines 30-49 dB above the rest, 300-800 Hz) ------

def _roomwav(name, repeat=3):
    import wave
    from pathlib import Path
    with wave.open(str(Path(__file__).with_name("data") / name), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return np.tile(x, repeat)


LAPTOP = dict(fs=FS, band=(400.0, 9000.0), ofdm=((1024, 512),), chirp=(400.0, 6000.0, 0.05), tones=(1000.0,))


@pytest.mark.parametrize("room", ["room_mac_48k.wav", "room_m6600_48k.wav"])
def test_a_real_room_is_not_a_signal_once_whitened(room):
    """Before whitening the fan lines read as chirps 32 % of the time on the Mac and as OFDM ~20 % on both."""
    x = _roomwav(room)
    p = run(cca.SenseSpec(**LAPTOP), x, block=1024)
    settled = p.flags(4 * FS, p.n)
    assert p.whitened and not settled.any(), p.stats()


@pytest.mark.parametrize("room", ["room_mac_48k.wav", "room_m6600_48k.wav"])
def test_signals_are_found_in_a_real_room(H, room):
    nz = _roomwav(room)
    a = 14 * FS
    for sig, bit, snr in ((H.make_msg_audio(FS, H.MSG_GO, 1, b"", (400.0, 9000.0)), cca.OFDM, 0.0),
                          (css_link.encode_bytes(CssConfig(symbol_dur=0.05), b"hi")[:5 * FS], cca.CHIRP, -6.0)):
        sig = np.asarray(sig, float)
        x = nz.copy()
        x[a:a + len(sig)] += (sig / band_rms(sig, 400, 9000) * band_rms(nz[:a], 400, 9000) * 10 ** (snr / 20)).astype(np.float32)
        p = run(cca.SenseSpec(**LAPTOP), x, block=1024)
        hit, _ = split(p, a, a + len(sig), bit)
        before = p.flags(4 * FS, a - FS) & bit
        assert hit >= 0.9 and not before.any(), (bit, hit, p.stats())
