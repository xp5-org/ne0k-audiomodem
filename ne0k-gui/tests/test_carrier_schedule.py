# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Carrier-schedule OFDM: carrier scores, masked frames that still decode, the receiver's step match,
and the arq send path that walks the score."""
import contextlib
import io
import json
import threading
import types

import numpy as np
import pytest

from modemgui.engine import carrier_schedule, ofdm_link
from modemgui.engine.mesh_rx import MeshReceiver
from modemgui.engine.modemcfg import OfdmConfig
from modemgui.engine.ofdm_link import OfdmLink, pack_frame

BAND = (200.0, 2700.0)
SCORE = {"version": 1, "name": "t", "steps": [{"notes": [n], "beats": 1} for n in (76, 74, 72, 79, 67, 84)]}
SCORE_8 = {"version": 1, "name": "t8", "steps": [{"notes": [n], "beats": 1} for n in (86, 88, 84, 81, 83, 79, 74, 76)]}


@pytest.fixture(scope="module")
def H():
    return ofdm_link._imports()[0]


def _score(H, fs=48000, note_carriers=4, score=SCORE):
    return carrier_schedule.CarrierScore(score, H.hs_data_cfg(fs, BAND, 32), note_carriers)


def test_load_score_rejects_empty_and_bad_notes(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"steps": []}))
    with pytest.raises(ValueError):
        carrier_schedule.load_score(str(p))
    p.write_text(json.dumps({"steps": [{"notes": [200]}]}))
    with pytest.raises(ValueError):
        carrier_schedule.load_score(str(p))
    p.write_text(json.dumps(SCORE))
    assert len(carrier_schedule.load_score(str(p))["steps"]) == 6


def test_every_step_has_two_or_more_carriers_and_distinct_steps_differ(H):
    sc = _score(H)
    assert all(sc.mask(s).sum() >= 2 for s in range(len(sc)))
    assert not np.array_equal(sc.mask(0), sc.mask(1))
    assert np.array_equal(sc.mask(0), sc.mask(len(sc)))           # the score loops


def test_masked_bat_keeps_the_channel_bits_and_falls_back_when_nothing_is_usable(H):
    sc = _score(H)
    n = len(sc.data_bins)
    bat = np.full(n, 2, np.int32)
    out = sc.bat(bat, 0)
    assert set(np.unique(out)) == {0, 2} and (out > 0).sum() == sc.mask(0).sum()
    dead = np.zeros(n, np.int32)
    assert np.array_equal(sc.bat(dead, 0), dead)                    # nothing usable: the full (dead) table
    holes = bat.copy()
    holes[sc.mask(0)] = 0                                           # note carriers unusable: neighbours step in
    assert (sc.bat(holes, 0) > 0).any()


def test_masked_frame_decodes_and_the_spectrum_picks_its_step(H):
    from ofdm_modem import AdaptiveDemodulator
    fs, sc = 48000, _score(H)
    full = np.full(len(sc.data_bins), 2, np.int32)
    rng = np.random.default_rng(1)
    for s in (2, 3):
        bat = sc.bat(full, s)
        pay = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
        a = H.make_data_audio(fs, pay, bat, BAND, 32)
        x = np.concatenate([np.zeros(fs // 2), a, np.zeros(fs // 2)]) + rng.normal(0, 0.002, len(a) + fs)
        assert sc.rank(x, full, [s - 1, s, s + 1])[0] == s
        with contextlib.redirect_stdout(io.StringIO()):
            _, chunks = AdaptiveDemodulator(H.hs_data_cfg(fs, BAND, 32), bat).decode_samples(x)
        assert b"".join(chunks[k] for k in sorted(chunks))[:32] == pay


def _bare_link(H, score=None, mode="arq"):
    lk = OfdmLink.__new__(OfdmLink)
    lk.cfg = OfdmConfig(mode=mode, retries=1, ack_wait_s=1.0, hs_timeout_s=1.0)
    lk._lock, lk._stop, lk._up, lk._ack_evt = threading.Lock(), threading.Event(), threading.Event(), threading.Event()
    lk._up.set()
    lk._tx_seq = lk._rx_next = 0
    lk._inflight, lk._pending, lk._owe_ack, lk.on_data, lk.on_retry = None, None, False, None, None
    lk._cdx = None
    lk.capacity = 26
    lk.retry_count = lk.fail_count = 0
    lk._mel_tx = lk._mel_rx = score
    lk._mel_tx_n = lk._mel_rx_n = 0
    lk._mel_dems = {}
    lk._ev = lambda *a, **k: None
    lk.log = lambda m: None
    lk._rx_bat = np.full(46, 2, np.int32)
    lk._rx_band = BAND
    lk._dem = None
    lk._mel_rx_prev = 0
    lk._rsv_id, lk._rsv_tx, lk._rsv_rx, lk._rsv_evt = 0, None, None, threading.Event()
    return lk


def test_arq_send_walks_the_score_one_step_per_new_frame_and_resends_repeat_it(H):
    lk = _bare_link(H, _score(H))
    sent = []

    def tx(seq, ack, data, step=None):
        sent.append((seq, step))
        if len(sent) != 2:                       # the 2nd airing (frame 1's first try) gets no ACK: it is resent
            threading.Timer(0.01, lambda: lk._on_frame(0, (seq + 1) & 0xFF, b"")).start()

    lk._tx_frame_full = tx
    assert lk._send_full(b"x" * 26 * 3)          # three frames
    assert sent == [(0, 0), (1, 1), (1, 1), (2, 2)]


def test_receiver_counts_new_frames_only_and_offers_step_demods_best_first(H):
    sc = _score(H)
    lk = _bare_link(H, sc)
    lk._on_frame(0, 0, b"a")
    lk._on_frame(0, 0, b"a")                     # duplicate: not a new step
    lk._on_frame(1, 0, b"b")
    assert lk._mel_rx_n == 2
    fs = 48000
    a = H.make_data_audio(fs, b"z" * 32, sc.bat(lk._rx_bat, 2), BAND, 32)
    dems = lk.rx_demods_for(np.concatenate([np.zeros(fs // 4), a, np.zeros(fs // 4)]))
    assert [d.start for d in dems[:2]] == [2, 1]  # the next new frame, a resend of the last one
    assert dems[2] is lk._dem or len(dems) == 3   # then the full table


def test_mesh_receiver_tries_every_offered_demod_of_a_scheduled_link():
    class _Dem:
        def __init__(self, raw):
            self.raw, self.calls = raw, 0

        def decode_samples(self, x):
            self.calls += 1
            return ([{"evm_pct": 1.0}], {0: self.raw}) if self.raw else ([], {})

    miss, hit = _Dem(None), _Dem(pack_frame(0, 0, b"note", 2, 1))
    lk = types.SimpleNamespace(cfg=OfdmConfig(node_id=1, peer_id=2, mode="arq", sender="initiator"),
                               role="responder", up=True, shared_rx=True, schedule_rx=object(), got=[])
    lk.rx_demod = lambda: (miss, np.full(8, 2), BAND)
    lk.rx_demods_for = lambda x: [miss, hit]
    lk.wants_ack = lambda: False
    lk.deliver = lambda seq, ack, data, *a: lk.got.append(data)
    r = MeshReceiver(types.SimpleNamespace(pipe=None), OfdmConfig(node_id=1), 1, log=lambda m: None)
    r.links, r.expected = (lambda: [lk]), (lambda: [2])
    r._cap.add(np.zeros(48000 * 4, np.float32))
    assert r._data(48000, 96000)
    assert miss.calls == 1 and hit.calls == 1 and lk.got == [b"note"]


def test_boosted_scheduled_frame_puts_the_notes_over_the_pilots_and_still_decodes(H):
    from ofdm_modem import AdaptiveDemodulator
    fs, sc = 48000, _score(H)
    cfg = H.hs_data_cfg(fs, BAND, 32)
    bat = sc.bat(np.full(len(sc.data_bins), 2, np.int32), 1)
    pay = bytes(range(32))
    pilots = [cfg["bin_lo"] + i for i in range(0, cfg["n_occupied"], cfg["pilot_spacing"])]

    def level(a):
        n = len(a) // 1024 * 1024
        p = np.mean(np.abs(np.fft.rfft(a[:n].reshape(-1, 1024) * np.hanning(1024), axis=1)) ** 2, axis=0)
        return 10 * np.log10(p[sc.data_bins[bat > 0]].mean() / p[pilots].mean())

    with contextlib.redirect_stdout(io.StringIO()):
        flat = sc.audio(H, fs, pay, bat, BAND, 32, 0.0)
        loud = sc.audio(H, fs, pay, bat, BAND, 32, 12.0)
        assert len(flat) == len(loud)
        assert level(loud) - level(flat) > 9.0
        x = np.concatenate([np.zeros(fs // 2), loud, np.zeros(fs // 2)])
        _, chunks = AdaptiveDemodulator(cfg, bat).decode_samples(x)
    assert b"".join(chunks[k] for k in sorted(chunks))[:32] == pay


def test_a_frame_of_several_chunks_plays_one_step_per_chunk(H):
    lk = _bare_link(H, _score(H))
    lk.capacity = 249                            # arq: 255 B frames, 8 chunks of 32 B
    sent = []

    def tx(seq, ack, data, step=None):
        sent.append((seq, step, lk._chunks_of(data)))
        threading.Timer(0.01, lambda: lk._on_frame(0, (seq + 1) & 0xFF, b"")).start()

    lk._tx_frame_full = tx
    assert lk._send_full(b"x" * 600)              # 249 + 249 + 102 B
    assert sent == [(0, 0, 8), (1, 8, 8), (2, 16, 4)] and lk._mel_tx_n == 20


def _phrase_link(H, sc, **kw):
    lk = _bare_link(H, sc, **kw)
    lk.cfg = OfdmConfig(node_id=2, peer_id=1, mode="arq", sender="initiator", schedule_boost_db=12.0, gap_s=0.0)
    lk.role, lk.state, lk.shared_rx, lk.rx_filter, lk.rx_frames = "responder", "up", True, None, 0
    lk._tx_bat, lk._tx_band, lk._txlock, lk.tx_frames = np.full(46, 2, np.int32), BAND, threading.Lock(), 0
    return lk


def test_phrase_frame_loops_back_through_the_mesh_receiver_and_a_resend_is_recognised(H):
    """A 3-chunk scheduled frame (3 notes back to back) from the real TX path into the real MeshReceiver: decoded
    chunk by chunk on its steps, delivered, one ACK.  The same frame again (our ACK lost) decodes on the previous
    start, is dropped as a duplicate and ACKed again."""
    fs, sc = 48000, _score(H)
    tx = _phrase_link(H, sc)
    tx._pack = lambda seq, ack, data: pack_frame(seq, ack, data, 1, 2)
    aired = []
    tx._play = lambda audio, what, ident=False: aired.append((np.asarray(audio, np.float64), what))
    pay = bytes(range(65, 65 + 80))               # 80 + 6 B = 3 chunks
    with contextlib.redirect_stdout(io.StringIO()):
        tx._tx_frame_full(0, 0, pay, step=0)
    audio, what = aired[0]
    assert "schedule steps 0-2/6" in what

    rx = _phrase_link(H, sc)
    got, acks = [], []
    rx.on_data = got.append
    rx._tx_frame_full = lambda seq, ack, data, step=None: acks.append(ack)
    r = MeshReceiver(types.SimpleNamespace(pipe=None), OfdmConfig(node_id=2), 2, log=lambda m: None)
    r.links, r.expected = (lambda: [rx]), (lambda: [1])
    noise = np.random.default_rng(5)
    for _ in range(2):
        a = r._cap.n + fs
        r._cap.add(np.zeros(fs, np.float32))
        r._cap.add((audio + noise.normal(0, 1e-4, len(audio))).astype(np.float32))
        r._cap.add(np.zeros(fs, np.float32))
        with contextlib.redirect_stdout(io.StringIO()):
            assert r._data(a, a + len(audio))
    assert got == [pay] and acks == [1, 1] and rx._mel_rx_n == 3 and rx._mel_rx_prev == 0


@pytest.mark.parametrize("bits", [4, 6])
def test_boosted_qam_phrase_decodes_because_ce_carries_the_same_boost(H, bits):
    """16/64-QAM decide on amplitude: the note boost must reach the channel estimate too, or every symbol is 12 dB
    off the grid.  An 8-note phrase of mixed chunk lengths on the G.711 band, decoded chunk by chunk."""
    fs, band = 48000, (300.0, 4000.0)
    cfg = H.hs_data_cfg(fs, band, 32)
    sc = carrier_schedule.CarrierScore(SCORE_8, cfg, 4)
    bat = np.full(len(sc.data_bins), bits, np.int32)
    lk = OfdmLink.__new__(OfdmLink)
    lk.cfg, lk._mel_rx, lk._mel_dems = OfdmConfig(), sc, {}
    body = pack_frame(3, 0, bytes(range(249)), 1, 2)
    with contextlib.redirect_stdout(io.StringIO()):
        a = sc.audio(H, fs, body, [sc.bat(bat, k) for k in range(8)], band, 32, 12.0)
        x = np.concatenate([np.zeros(fs // 2), a, np.zeros(fs // 2)])
        x = x + np.random.default_rng(2).normal(0, np.sqrt(np.mean(a ** 2)) / 10 ** 1.6, len(x))   # 32 dB SNR
        _, chunks = ofdm_link._PhraseDem(lk, 0, bat, band).decode_samples(x)
    assert sorted(chunks) == list(range(8))
    assert b"".join(chunks[k] for k in range(8))[:len(body)] == body


def test_every_note_carries_the_same_bits_so_notes_last_the_same(H):
    """Pitch and channel decide which carriers a note gets, not how many bits: on a mixed 2/4/6-bit table every step
    of a real tune lands on the target (4 x the median carrier) to within one carrier's overshoot."""
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "audiotestserver", "tests", "OFDM", "joplin_score.json")
    cfg = H.hs_data_cfg(48000, (300.0, 4000.0), 32)
    sc = carrier_schedule.CarrierScore(carrier_schedule.load_score(path), cfg, 4)
    bat = np.resize(np.array([6, 6, 4, 6, 2, 6, 4, 6], np.int32), len(sc.data_bins))
    bat[:2] = 0
    target = 4 * int(np.median(bat[bat > 0]))
    bits = [int(sc.bat(bat, s).sum()) for s in range(len(sc))]
    assert min(bits) >= target and max(bits) < target + 6, (target, sorted(set(bits)))


# -- P2P TX window reservation ---------------------------------------------------------------------------------

RSV_BAND = (300.0, 4000.0)


def _rsv_pair(H, sc_tx, sc_rx, window=64):
    """A sender and a receiver link, the receiver's status frames captured."""
    fs = 48000
    n = len(carrier_schedule.CarrierScore(SCORE, H.hs_data_cfg(fs, RSV_BAND, 32), 4).data_bins)
    tx = _bare_link(H, sc_tx)
    rx = _bare_link(H, sc_rx)
    for lk in (tx, rx):
        lk.cfg = OfdmConfig(mode="arq", tx_window_chunks=window, schedule_boost_db=12.0, gap_s=0.0, retries=2,
                            ack_wait_s=1.0, hs_timeout_s=1.0)
        lk._tx_bat = lk._rx_bat = np.full(n, 6, np.int32)
        lk._tx_band = lk._rx_band = RSV_BAND
        lk._txlock, lk.tx_frames, lk.rx_frames, lk.last_activity = threading.Lock(), 0, 0, 0.0
        lk._rsv_id, lk._rsv_tx, lk._rsv_rx, lk._rsv_evt = 0, None, None, threading.Event()
    aired, ctrl, got = [], [], []
    tx._play = lambda audio, what, ident=False: aired.append((np.asarray(audio, np.float64), what)) or 0.0
    rx._rsv_send_ctrl = ctrl.append
    rx.on_data = got.append
    return tx, rx, aired, ctrl, got


def _reserve(rx, wid, win, n, step0):
    from ofdm_handshake import crc16
    from modemgui.engine.ofdm_link import RSV_MAGIC, _RSV
    rx._on_rsv_ctrl(RSV_MAGIC + _RSV.pack(wid, n, step0, len(win), crc16(win)))


def _status(payload):
    from modemgui.engine.ofdm_link import _STA
    wid, k = _STA.unpack(payload[4:4 + _STA.size])
    return wid, list(np.frombuffer(payload[4 + _STA.size:], "<u2")[:k])


def _burst(audio, seed=1):
    fs = 48000
    x = np.concatenate([np.zeros(fs // 5), audio, np.zeros(fs // 3)])
    return x + np.random.default_rng(seed).normal(0, np.sqrt(np.mean(audio ** 2)) / 10 ** 1.6, len(x))


@pytest.mark.parametrize("scheduled", [True, False])
def test_reserved_window_streams_every_chunk_and_the_receiver_reads_it_slot_by_slot(H, scheduled):
    fs = 48000
    sc = carrier_schedule.CarrierScore(SCORE_8, H.hs_data_cfg(fs, RSV_BAND, 32), 4) if scheduled else None
    tx, rx, aired, ctrl, got = _rsv_pair(H, sc, sc)
    win = bytes(np.random.default_rng(4).integers(0, 256, 32 * 12 - 7, dtype=np.uint8))
    _reserve(rx, 0, win, 12, 3)
    assert _status(ctrl[-1]) == (0, list(range(12)))             # armed: stream them all
    with contextlib.redirect_stdout(io.StringIO()):
        tx._rsv_stream(0, win, list(range(12)), 3, first=True)
        assert rx.rx_window(_burst(aired[0][0]))
    assert got == [win] and _status(ctrl[-1]) == (0, [])
    assert ("schedule steps 3+12/8" in aired[0][1]) == scheduled


def test_a_lost_slot_is_named_in_the_status_and_streamed_again_alone(H):
    fs = 48000
    sc = carrier_schedule.CarrierScore(SCORE_8, H.hs_data_cfg(fs, RSV_BAND, 32), 4)
    tx, rx, aired, ctrl, got = _rsv_pair(H, sc, sc)
    win = bytes(range(256)) * 1 + bytes(64)                       # 10 chunks
    _reserve(rx, 7, win, 10, 0)
    with contextlib.redirect_stdout(io.StringIO()):
        tx._rsv_stream(7, win, list(range(10)), 0, first=True)
        audio = aired[0][0].copy()
        # wipe slot 4: its start is the sum of the earlier slots' lengths
        start = sum(rx._rsv_air(k) for k in range(4)) + int(cfg_gap := H.hs_data_cfg(fs, RSV_BAND, 32)["frame_gap_samples"])
        audio[start:start + rx._rsv_air(4) - cfg_gap] = 0
        assert rx.rx_window(_burst(audio))
        assert got == [] and _status(ctrl[-1]) == (7, [4])
        tx._rsv_stream(7, win, [4], 0, first=False)
        assert rx.rx_window(_burst(aired[1][0], seed=2))
    assert got == [win] and _status(ctrl[-1]) == (7, []) and "again" in aired[1][1]


def test_sender_reserves_again_when_a_status_is_lost_and_finishes_on_progress(H):
    tx, rx, aired, ctrl, got = _rsv_pair(H, None, None, window=8)
    from modemgui.engine.ofdm_link import STA_MAGIC, _STA
    log = []

    def status(wid, miss):
        tx._on_rsv_ctrl(STA_MAGIC + _STA.pack(wid, len(miss)) + np.array(miss, "<u2").tobytes())

    replies = iter([None, list(range(8)), [2, 5], []])           # RESERVE lost, armed, 2 missing, all in

    def ctrl_tx(payload):
        log.append(payload[:4])
        r = next(replies)
        if r is not None:
            threading.Timer(0.01, status, (0, r)).start()

    tx._rsv_send_ctrl = ctrl_tx

    def stream(wid, win, todo, step0, first):
        log.append(tuple(todo))
        threading.Timer(0.01, status, (wid, next(replies))).start()
        return 0.0

    tx._rsv_stream = stream
    assert tx._send_full(bytes(200))                              # 7 chunks of 32 B
    assert log == [b"RSV1", b"RSV1", tuple(range(8)), (2, 5)]
