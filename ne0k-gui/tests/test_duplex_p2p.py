# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Point-to-point full duplex (ofdm mode full) as the GUI runs it: the band split, no carrier sense, MeshNode p2p."""
import threading
import types

import numpy as np
import pytest

pytest.importorskip("scipy")

from modemgui.audio import Registry
from modemgui.engine import Session
from modemgui.engine.mesh_node import MeshConfig, MeshNode
from modemgui.engine.modemcfg import CssConfig, OfdmConfig
from modemgui.engine.ofdm_link import OfdmLink, _imports


def _split_link(role, dl_frac, cutoff_hz):
    """A link as it stands after the BAT exchange: both tables 2 bits/carrier up to cutoff_hz, 0 above (G.711)."""
    H, _ = _imports()
    c = OfdmConfig(node_id=1 if role == "initiator" else 2, peer_id=2 if role == "initiator" else 1,
                   mode="full", dl_frac=dl_frac)
    full = H.hs_data_cfg(c.fs, (c.freq_lo, c.freq_hi), c.chunk_bytes)
    lk = OfdmLink.__new__(OfdmLink)
    lk.cfg, lk.role, lk.log = c, role, (lambda m: None)
    hz = np.array(OfdmLink._data_bins(full)) * c.fs / full["n_fft"]
    bat = np.where(hz < cutoff_hz, 2, 0).astype(np.int32)
    lk._tx_bat, lk._rx_bat, lk._band = bat.copy(), bat.copy(), (c.freq_lo, c.freq_hi)
    return H, lk


@pytest.mark.parametrize("dl_frac", [0.5, 0.3])
def test_bits_split_gives_the_responder_its_share_of_capacity_not_of_hertz(dl_frac):
    """400-4000 Hz band on a line dead above 3.4 kHz: a Hz split hands the responder (high side) a half that is
    partly dead.  duplex_split=bits moves the split down until the responder carries dl_frac of the bits, and
    both ends land on the same bands."""
    H, a = _split_link("initiator", dl_frac, 3400.0)
    _, b = _split_link("responder", dl_frac, 3400.0)
    a._setup_duplex(H); b._setup_duplex(H)
    assert a._tx_band == b._rx_band and a._rx_band == b._tx_band and a._tx_band[1] < b._tx_band[0]
    ub, db = int(a._tx_bat.sum()), int(b._tx_bat.sum())
    assert abs(db / (ub + db) - dl_frac) < 0.06, (ub, db)
    assert a.plan["dl_frac_actual"] > dl_frac + 0.05            # more Hz than the share: the top is dead
    a.cfg.duplex_split = "hz"
    a._tx_bat = a._rx_bat = _split_link("initiator", dl_frac, 3400.0)[1]._tx_bat
    a._setup_duplex(H)
    assert abs(a.plan["dl_frac_actual"] - dl_frac) < 0.02        # the old split, kept as an option


def test_full_duplex_data_does_not_wait_for_the_peers_carrier(monkeypatch):
    """The peer is on the air all the time in its own sub-band: carrier sense before a DATA frame (the half-duplex
    rule) turned full duplex back into turn-taking."""
    import modemgui.engine.ofdm_link as ol
    monkeypatch.setattr(ol, "keyup", lambda s, a, fs: a)
    for mode, deferred in (("full", False), ("arq", True)):
        lk = OfdmLink.__new__(OfdmLink)
        lk.cfg, lk.log = OfdmConfig(mode=mode), (lambda m: None)
        lk._ev = lambda *a, **k: None
        lk.session = types.SimpleNamespace(play=lambda a, ident=False: None, pipe=None)
        calls = []
        lk._defer_for_air = lambda: calls.append(1)
        monkeypatch.setattr(ol.time, "sleep", lambda s: None)
        lk._play(np.zeros(480, np.float32), "DATA seq=0 ack=0 4B")
        assert bool(calls) == deferred, mode


def test_full_mode_refuses_the_mesh_announce_round():
    with pytest.raises(ValueError):
        MeshNode(Session(Registry(), 48000, 1024), MeshConfig(node_id=1, phy_mode="ofdm_only", announce_round=True),
                 CssConfig(), OfdmConfig(mode="full"))


def test_p2p_mesh_node_full_duplex_both_send_at_once():
    """GUI p2p topology, ofdm_only, ofdm mode full: each node keeps ONE full-duplex link to its peer and both
    send at the same time."""
    reg, nodes = Registry(), {}
    try:
        for nid, peer, rx, tx in ((1, 2, "loop:p2pb", "loop:p2pa"), (2, 1, "loop:p2pa", "loop:p2pb")):
            s = Session(reg, 48000, 1024); s.open(rx, tx)
            s.keyup_tone = False                                   # a SIP line: no PL lead
            n = MeshNode(s, MeshConfig(node_id=nid, phy_mode="ofdm_only"), CssConfig(),
                         OfdmConfig(mode="full", channel="line"), log=lambda m, i=nid: print(f"[{i}] {m}", flush=True))
            n.ring_dst_fn = lambda p=peer: p
            n.got = []
            n.on_data = lambda src, d, n=n: n.got.append((src, d))
            n.session_ = s
            nodes[nid] = n
        for n in nodes.values():
            n.start()
        assert nodes[1].rx is None and nodes[1].announce is None   # no shared half-duplex receiver
        res = {}
        pa, pb = b"A->B " * 8, b"B->A " * 5
        ta = threading.Thread(target=lambda: res.setdefault(1, nodes[1].send(2, pa)))
        tb = threading.Thread(target=lambda: res.setdefault(2, nodes[2].send(1, pb)))
        ta.start(); tb.start(); ta.join(300); tb.join(300)
        assert res == {1: True, 2: True}
        # the duplex data plane is a byte stream (one chunk per frame, ec and fdd alike): compare the bytes
        for n, src, want in ((nodes[2], 1, pa), (nodes[1], 2, pb)):
            assert {s for s, _ in n.got} == {src} and b"".join(d for _, d in n.got) == want
        assert list(nodes[1].links) == [(2, "dx")] and list(nodes[2].links) == [(1, "dx")]
        assert nodes[1].links[(2, "dx")].role == "initiator" and nodes[2].links[(1, "dx")].role == "responder"
    finally:
        for n in nodes.values():
            n.stop(); n.session_.close()


# -- duplex ec: same band both ways, echo canceller (V.34 style) ------------------------------------------------------

def test_echo_canceller_finds_the_path_and_removes_it_under_double_talk():
    from modemgui.engine.ofdm_ec import EchoCanceller
    rng, fs = np.random.default_rng(1), 48000
    h = np.zeros(40); h[[0, 3, 10, 30]] = [0.8, -0.3, 0.2, 0.05]
    delay, i0 = 1100, 10000
    ec = EchoCanceller(fs, taps=128)
    ref = (0.3 * rng.standard_normal(5 * fs)).astype(np.float32)           # "our probe", the far end quiet
    raw = np.zeros(len(ref) + fs, np.float32)
    raw[delay:delay + len(ref)] = np.convolve(ref, h)[:len(ref)]
    ec.add_ref(i0, ref)
    for i in range(0, len(raw), 1024):
        ec.add_raw(i0 + i, raw[i:i + 1024])
    st = ec.train()
    assert st["on"] and st["erle_db"] > 40 and abs(st["delay_ms"] - 1000 * delay / fs) < 1.0, st
    ref2, i2 = (0.3 * rng.standard_normal(fs)).astype(np.float32), i0 + 7 * fs  # data: both talking
    ec.add_ref(i2, ref2)
    echo = np.zeros(fs + 2048, np.float32); echo[delay:delay + fs] = np.convolve(ref2, h)[:fs]
    far = (0.05 * rng.standard_normal(len(echo))).astype(np.float32)
    y = np.concatenate([ec.cancel(i2 + i, (echo + far)[i:i + 1024]) for i in range(0, len(echo), 1024)])
    assert np.sqrt(np.mean((y - far) ** 2)) < 0.01 * np.sqrt(np.mean(echo ** 2))   # > 40 dB under the far end
    clean = EchoCanceller(fs)                                                 # a 4-wire line: nothing comes back
    clean.add_ref(0, ref)
    for i in range(0, len(raw), 1024):
        clean.add_raw(i, (0.1 * rng.standard_normal(1024)).astype(np.float32))
    st = clean.train()
    assert not st["on"] and st["why"] == "no echo path"
    x = np.ones(1024, np.float32)
    assert clean.cancel(0, x) is x


@pytest.mark.parametrize("rates,share,strict,want", [
    ((100.0, 100.0), 0.5, False, (1.0, 1.0)), ((120.0, 60.0), 0.5, False, (1.0, 1.0)),
    ((100.0, 100.0), 0.3, False, (1.0, 3 / 7)), ((100.0, 100.0), 0.7, False, (3 / 7, 1.0)),
    ((100.0, 100.0), 0.1, True, (1.0, 1 / 9)), ((100.0, 100.0), 0.9, True, (1 / 9, 1.0)),
    ((100.0, 100.0), 0.5, True, (1.0, 1.0)), ((120.0, 60.0), 0.5, True, (0.5, 1.0)),
    ((60.0, 120.0), 0.3, False, (1.0, 0.5 * 0.3 / 0.7))])
def test_ec_duty_holds_each_end_to_its_share(rates, share, strict, want):
    """dl_frac 0.5 (not strict): each direction runs what its line measured (V.34).  Otherwise the end over its share
    carries data in only a fraction of its chunks, so share = duty*mine / (other + duty*mine)."""
    d_i, d_r = OfdmLink._ec_duty(*rates, share, strict)
    assert (d_i, d_r) == pytest.approx(want)
    ri, rr = rates
    got = rr * d_r / (ri * d_i + rr * d_r)
    if strict or abs(share - 0.5) > 1e-9:
        assert got == pytest.approx(share)


@pytest.mark.parametrize("ratio,want", [("50/50", (0.5, True)), ("90/10", (0.1, True)), ("10/90", (0.9, True)),
                                        ("3:1", (0.25, True)), ("", (0.3, False))])
def test_ratio_setting_is_initiator_over_responder(ratio, want):
    assert OfdmConfig(ratio=ratio, dl_frac=0.3).responder_share() == pytest.approx(want)
    with pytest.raises(ValueError):
        OfdmConfig(ratio="90").responder_share()


def _win_link(window=4, retries=3):
    lk = OfdmLink.__new__(OfdmLink)
    lk.cfg = OfdmConfig(mode="full", duplex="ec", ec_window=window, retries=retries, ack_wait_s=5.0)
    lk.log, lk._ev, lk.on_data = (lambda m: None), (lambda *a, **k: None), None
    lk._lock = threading.Lock()
    lk._ec_q, lk._ec_win, lk._ec_rxbuf = __import__("collections").deque(), {}, {}
    lk._ec_srtt = lk._ec_rttvar = None
    lk.ec_acked_bytes = lk.ec_rx_bytes = lk.fail_count = 0
    lk._tx_seq = lk._rx_next = 0
    return lk


def test_ec_window_keeps_k_frames_in_flight_and_resends_only_the_hole():
    lk = _win_link(window=4)
    box = dict(evt=threading.Event(), left=6, ok=True)
    lk._ec_q.extend(dict(data=bytes([i]), box=box, sends=0, sent=None) for i in range(6))
    sent = [lk._ec_pick_locked(t, t + 0.25)["seq"] for t in (0.0, 0.25, 0.5, 0.75)]
    assert sent == [0, 1, 2, 3] and len(lk._ec_win) == 4 and len(lk._ec_q) == 2
    assert lk._ec_pick_locked(1.0, 1.25) is None                  # window full, nothing timed out: filler
    lk._ec_acked_locked(0, 2.0)                                   # the peer still wants 0: 0 was lost
    assert lk._ec_pick_locked(2.0, 2.25) is None                  # its timer (ack_wait_s before any RTT sample) runs
    f = lk._ec_pick_locked(5.3, 5.55)
    assert f["seq"] == 0 and f["sends"] == 2                      # only the oldest is resent
    assert lk._ec_pick_locked(5.6, 5.85) is None
    lk._ec_acked_locked(4, 6.0)                                   # the peer had 1..3 already: all four clear at once
    assert not lk._ec_win and box["left"] == 2 and lk.ec_acked_bytes == 4
    assert [lk._ec_pick_locked(t, t + .25)["seq"] for t in (6.0, 6.25)] == [4, 5]
    lk._ec_acked_locked(5, 7.0)                                   # 4 acked on its first send: an RTT sample
    assert lk._ec_srtt == pytest.approx(0.75) and lk._ec_rto() == pytest.approx(2.25)
    assert lk._ec_pick_locked(8.0, 8.25) is None                  # 5 went out at 6.5: its timer runs to 8.75
    f = lk._ec_pick_locked(8.8, 9.05)                             # the timer now follows the measured RTT
    assert f["seq"] == 5 and f["sends"] == 2


def test_ec_window_fails_a_frame_after_its_retries():
    lk = _win_link(window=2, retries=1)
    box = dict(evt=threading.Event(), left=1, ok=True)
    lk._ec_q.append(dict(data=b"x", box=box, sends=0, sent=None))
    t = 0.0
    for _ in range(2):
        assert lk._ec_pick_locked(t, t + 0.25) is not None
        t += 6.0
    assert lk._ec_pick_locked(t, t + 0.25) is None and box["evt"].is_set() and not box["ok"] and lk.fail_count == 1


def test_ec_receiver_keeps_frames_past_a_gap_and_delivers_in_order():
    lk = _win_link()
    got = []
    lk.on_data = got.append
    lk._on_frame_ec(0, 0, b"a")
    lk._on_frame_ec(2, 0, b"c")                                   # 1 lost on the line
    lk._on_frame_ec(3, 0, b"d")
    assert got == [b"a"] and lk._rx_next == 1 and set(lk._ec_rxbuf) == {2, 3}
    lk._on_frame_ec(0, 0, b"a")                                   # a duplicate: dropped
    lk._on_frame_ec(1, 0, b"b")                                   # the resend fills the hole
    assert got == [b"a", b"b", b"c", b"d"] and lk._rx_next == 4 and not lk._ec_rxbuf
    lk._on_frame_ec(6, 0, b"g")                                   # 4 and 5 failed at the sender: skipped in time
    import time as _t
    lk._ec_rxbuf[6] = (b"g", _t.monotonic() - 100.0)
    lk._on_frame_ec(4, 0, b"")                                    # any chunk (a filler) runs the gap check
    assert got[-1] == b"g" and lk._rx_next == 7


def _ec_pair(rx_tx, keyup=False, **cfg):
    reg, ends = Registry(), []
    for (node, peer), (rx, tx) in zip(((1, 2), (2, 1)), rx_tx):
        s = Session(reg, 48000, 1024); s.open(rx, tx)
        s.keyup_tone = keyup
        lk = OfdmLink(s, OfdmConfig(node_id=node, peer_id=peer, mode="full", duplex="ec", channel="line", **cfg))
        lk.got, lk.sess = [], s
        lk.on_data = lk.got.append
        ends.append(lk)
    for lk in ends:
        lk.start()
    return ends


def _carrier_stays_up(a, b):
    """V.34 style: once up the carrier never drops -- chunks (filler when idle) keep coming with nothing to send."""
    import time
    n0, m0 = a.tx_frames, b.tx_frames
    time.sleep(4.0)
    for lk, k0 in ((a, n0), (b, m0)):
        assert lk.tx_frames - k0 >= 4, (lk.role, lk.tx_frames - k0)
        assert lk.session.pipe.tx_pending > 0                                 # a chunk is always queued ahead


def _both_send(a, b):
    pa, pb = b"A->B same band " * 3, b"B->A same band " * 2
    res = {}
    ta = threading.Thread(target=lambda: res.setdefault("a", a.send(pa)))
    tb = threading.Thread(target=lambda: res.setdefault("b", b.send(pb)))
    ta.start(); tb.start(); ta.join(180); tb.join(180)
    assert res == {"a": True, "b": True}
    assert b"".join(b.got) == pa and b"".join(a.got) == pb     # the stream carries chunk_bytes - HDR per frame


def test_ec_full_duplex_over_an_echoing_hybrid():
    """A 2-wire line that returns each end's own signal at -3 dB: the canceller trains on its probe's echo and
    both ends then send at once on the WHOLE band."""
    a, b = _ec_pair([("hyb:ecl/1", "hyb:ecl/1"), ("hyb:ecl/2", "hyb:ecl/2")])
    try:
        assert a.wait_up(150) and b.wait_up(40), (a.fail_reason, b.fail_reason)
        for lk in (a, b):
            assert lk._tx_band == lk._rx_band == (lk.cfg.freq_lo, lk.cfg.freq_hi)
            st = lk.plan["echo"]
            assert st["on"] and st["erle_db"] > 20, st
        _carrier_stays_up(a, b)
        _both_send(a, b)
        # the stream decoder drops ~1 % of chunks on separate hosts (SIP group runs); here both modems share one
        # process and its DSP lock, which loses more.  ARQ covers it: only a broken stream fails this bound.
        for lk in (a, b):
            assert lk.lost_frames <= max(3, 0.25 * lk.tx_frames), (lk.role, lk.lost_frames, lk.tx_frames)
        assert a.retry_count <= 2 and b.retry_count <= 2       # a lost chunk that carried data is resent
    finally:
        for lk in (a, b):
            lk.stop(); lk.sess.close()


def test_ec_on_a_clean_four_wire_line_turns_the_canceller_off():
    a, b = _ec_pair([("loop:ec4b", "loop:ec4a"), ("loop:ec4a", "loop:ec4b")])
    try:
        assert a.wait_up(150) and b.wait_up(40), (a.fail_reason, b.fail_reason)
        for lk in (a, b):
            assert not lk.plan["echo"]["on"] and lk.plan["echo"]["why"] == "no echo path"
        _carrier_stays_up(a, b)
        _both_send(a, b)
    finally:
        for lk in (a, b):
            lk.stop(); lk.sess.close()


@pytest.mark.parametrize("ratio", ["90/10", "10/90"])
def test_ec_ratio_shows_in_throughput_when_both_ends_saturate(ratio):
    """Both ends keep their send queue full: the sliding window lets the favoured end use every chunk, the other
    only its duty, so acked bytes split by the ratio (loose bound: integer chunks, one process, lost chunks)."""
    import time
    a, b = _ec_pair([("loop:ecrb", "loop:ecra"), ("loop:ecra", "loop:ecrb")], ratio=ratio)
    stop = threading.Event()
    def pump(lk):                       # 40 frames a send, two senders an end: the queue never runs dry
        while not stop.is_set():
            lk.send(b"A" * (40 * 26))
    try:
        assert a.wait_up(150) and b.wait_up(40), (a.fail_reason, b.fail_reason)
        ts = [threading.Thread(target=pump, args=(lk,), daemon=True) for lk in (a, a, b, b)]
        for t in ts:
            t.start()
        time.sleep(8.0)
        a0, b0 = a.ec_acked_bytes, b.ec_acked_bytes
        time.sleep(40.0)
        da, db = a.ec_acked_bytes - a0, b.ec_acked_bytes - b0
        want = float(ratio.split("/")[0]) / 100.0
        assert da + db > 0 and abs(da / (da + db) - want) < 0.1, (ratio, da, db)
        assert max(da, db) >= 0.6 * 40 * a.phy_bps / 8, (da, db, a.phy_bps)   # the window keeps the carrier busy
        assert set(b"".join(b.got)) == {65} and set(b"".join(a.got)) == {65}
    finally:
        stop.set()
        for lk in (a, b):
            lk.stop(); lk.sess.close()


def test_fifo_serial_port_on_a_duplex_mesh_node_has_cts_without_a_link_up_flag():
    """The GUI puts the serial bridge on the MeshNode (no .up): a FIFO port there is ready while it has room."""
    from modemgui.engine.serialbridge import SerialBridge
    br = SerialBridge.__new__(SerialBridge)
    br.fifo, br.mesh, br.ofdm, br.busy = True, True, False, True
    br.link, br.buf_size, br.txbuf = object(), 8, bytearray(b"1234")
    assert br.cts
    br.txbuf += b"5678"
    assert not br.cts


def test_ec_listener_recovers_a_preamble_hidden_by_the_chunk_before_it():
    """Back to back, the previous chunk's tail can hold the Schmidl-Cox metric above threshold into the next preamble:
    no rising edge, so the frame search never tries it (this exact stream lost chunk 116 in group run 4ade).  The
    stream's chunk length is fixed, so the listener demodulates a chunk missing between two decoded ones in place."""
    H, _ = _imports()
    c = OfdmConfig(mode="full", duplex="ec")
    band = (c.freq_lo, c.freq_hi)
    full = H.hs_data_cfg(c.fs, band, c.chunk_bytes)
    nb = len(OfdmLink._data_bins(full))
    bat = np.full(nb, 2, np.int32); bat[60:] = 0
    lk = OfdmLink.__new__(OfdmLink)
    lk.cfg, lk.log, lk._filt = c, (lambda m: None), None
    lk._tx_bat = lk._rx_bat = bat
    lk._tx_band = lk._rx_band = band
    from modemgui.engine.ofdm_link import pack_frame
    pat = bytes.fromhex("ffffff00") * 10
    bodies = [pack_frame(37, 11, b"", 2, 1)] + [pack_frame(sq, 11, pat[o:o + 26], 2, 1)
                                                 for sq, o in ((9, 2), (38, 0), (39, 2), (40, 0))]
    x = np.concatenate([np.zeros(4800, np.float32)] + [lk._ec_chunk_audio(H, b, 113 + i) for i, b in enumerate(bodies)]
                       + [np.zeros(4800, np.float32)])
    lk._rx_get = lambda i0, i1=None, raw=False: x[max(0, i0):i1]
    lk._cap = types.SimpleNamespace(get=lambda i0, i1=None: x[max(0, i0):i1])
    from modemgui.engine.ofdm_link import H_dem, _quiet
    with _quiet():
        stats, chunks = H_dem(H, c, bat, band).decode_samples(x)
    got = sorted((int(s["frame_start"]), s["seq"]) for s in stats if s.get("seq") in chunks)
    hidden = {113, 114, 115, 116, 117} - set(chunks)
    assert hidden and 113 in chunks and 117 in chunks             # the frame search alone misses one (116 here)
    got = lk._ec_fill(H, got, chunks, {})
    assert [cs for _, cs in got] == [113, 114, 115, 116, 117]
    for cs in hidden:
        assert bytes(chunks[cs])[:len(bodies[cs - 113])] == bodies[cs - 113]
