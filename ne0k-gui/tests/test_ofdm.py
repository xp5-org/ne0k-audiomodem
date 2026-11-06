# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
import threading
import time

import pytest

pytest.importorskip("scipy")             # the reference OFDM modem needs it

from modemgui.audio import Registry
from modemgui.engine import Session
from modemgui.engine.modemcfg import OfdmConfig
from modemgui.engine.ofdm_link import BROADCAST_ID, HDR, OfdmLink, pack_frame, unpack_frame
from modemgui.engine.serialbridge import HandshakePort, SerialBridge


def test_frame_roundtrip_and_crc():
    raw = pack_frame(7, 9, b"abc\x00\xff", 1, 2)
    assert len(raw) == HDR + 5 and raw[0] == 0x12                           # [src:4 | dst:4]
    assert unpack_frame(raw + b"\x00" * 11) == (1, 2, 7, 9, b"abc\x00\xff")  # modulator pads the tail
    assert unpack_frame(pack_frame(1, 2, b"")) == (0, BROADCAST_ID, 1, 2, b"")
    bad = bytearray(raw); bad[-1] ^= 1
    assert unpack_frame(bytes(bad)) is None
    bad = bytearray(raw); bad[0] = 0x13                                      # the CRC covers the address
    assert unpack_frame(bytes(bad)) is None
    assert unpack_frame(raw[:-2]) is None and unpack_frame(b"\x01") is None
    with pytest.raises(ValueError):
        pack_frame(0, 0, b"", 16, 1)


def test_link_takes_only_frames_from_its_peer_to_us():
    """Shared channel: node 3's link to node 1 hears 1->2 traffic and must neither deliver nor ACK it."""
    lk = OfdmLink.__new__(OfdmLink)
    lk.cfg, lk.foreign_frames, lk.log = OfdmConfig(node_id=3, peer_id=1, mode="arq"), 0, lambda m: None
    chunks = lambda raw: {0: raw}
    assert lk._frame_from(chunks(pack_frame(4, 5, b"hi", 1, 3))) == (4, 5, b"hi")
    assert lk._frame_from(chunks(pack_frame(4, 5, b"hi", 1, BROADCAST_ID))) == (4, 5, b"hi")
    assert lk._frame_from(chunks(pack_frame(4, 5, b"hi", 1, 2))) is False        # 1 -> 2: not ours
    assert lk._frame_from(chunks(pack_frame(4, 5, b"hi", 2, 3))) is False        # from 2: another link's
    assert lk.foreign_frames == 2
    assert lk._frame_from({}) is None
    lk.cfg = OfdmConfig(node_id=1, peer_id=1)                                   # unaddressed setup: as before
    assert lk._frame_from(chunks(pack_frame(4, 5, b"", 9, 9))) == (4, 5, b"")


def test_config_roles_and_roundtrip():
    c = OfdmConfig.from_dict({"window_s": "6", "chunk_bytes": "16", "hear_own": "yes"})
    assert (c.window_s, c.chunk_bytes, c.hear_own) == (6.0, 16, True)
    assert OfdmConfig(node_id=1, peer_id=2).to_dict()["role"] == "auto"


@pytest.fixture(scope="module")
def pair():
    reg, made = Registry(), []

    def mk(node, peer, rx, tx):
        s = Session(reg, 48000, 1024); s.open(rx, tx)
        link = OfdmLink(s, OfdmConfig(node_id=node, peer_id=peer))
        made.append((link, s)); return link, s

    a, sa = mk(1, 2, "loop:ba", "loop:ab")
    b, sb = mk(2, 1, "loop:ab", "loop:ba")
    a.got, b.got = [], []
    a.on_data, b.on_data = a.got.append, b.got.append
    stop = threading.Event()
    log = {0: [], 1: []}

    def collect():                                       # the event queue is bounded and spectra flood it
        while not stop.is_set():
            for i, sess in enumerate((sa, sb)):
                log[i] += sess.drain()
            time.sleep(0.05)
    t = threading.Thread(target=collect, daemon=True); t.start()
    a.events_log = log
    a.start(); b.start()
    assert a.wait_up(120) and b.wait_up(30), (a.fail_reason, b.fail_reason)
    yield a, b, sa, sb
    stop.set()
    for link, s in made:
        link.stop(); s.close()


def test_bat_exchange_roles_epoch_and_tables(pair):
    a, b, _, _ = pair
    assert (a.role, b.role) == ("initiator", "responder")
    # each end holds the table the peer measured for it, and the peer holds the same bytes
    assert a.sm.my_tx_bat == b.sm.peer_tx_bat and b.sm.my_tx_bat == a.sm.peer_tx_bat
    assert a.sm.done and b.capacity == a.capacity > 100
    assert abs(a.epoch - b.epoch) < 1.0                  # skew must stay inside the guard
    assert (a._tx_bat > 0).sum() == (b._rx_bat > 0).sum() > 0


def test_payload_both_ways_binary_safe(pair):
    a, b, _, sb = pair
    assert a.send(b"hello \x00\xff end")
    assert b.got[-1] == b"hello \x00\xff end"
    assert b.send(b"y" * 100)
    assert a.got[-1] == b"y" * 100
    assert a.retry_count == 0 and b.retry_count == 0


def test_ham_id_takes_a_window_slot_and_the_peer_hears_it(pair):
    from modemgui.engine.modemcfg import HamConfig
    a, b, sa, sb = pair
    sa.ident.cfg = HamConfig(enabled=True, callsign="N0CALL", id_mode="ascii", id_interval_s=3600,
                             id_end_delay_s=0)
    sa.ident.reset()
    try:
        assert a.send(b"after the id")                   # the ID goes first, the frame next window
        assert b.got[-1] == b"after the id"
        assert [r for _, _, r in sa.ident.sent] == ["window slot"]
        t0 = time.monotonic()
        while not sb.ident.heard and time.monotonic() - t0 < 10:
            time.sleep(0.2)
        assert sb.ident.heard and sb.ident.heard[0][1:] == ("ascii", "DE N0CALL")
    finally:
        sa.ident.cfg = HamConfig()


def test_events_cover_the_whole_exchange(pair):
    a, b, sa, sb = pair
    assert a.send(b"ping")                               # an ACK needs a data frame
    time.sleep(0.5)
    evs = [(i, e) for i in (0, 1) for e in a.events_log[i]]
    assert {"tx", "rx", "lock", "ack"} <= {e.kind for _, e in evs if hasattr(e, "kind")}
    # BAT exchange charts: each end gets what it measured of the peer AND what the peer measured of it
    from modemgui.engine import ProbeReport
    import numpy as np
    for i, (me, peer) in enumerate(((1, 2), (2, 1))):
        pr = {e.source: e for j, e in evs if j == i and isinstance(e, ProbeReport)}
        assert set(pr) == {"measured", "peer_report"}, pr.keys()
        m, p = pr["measured"], pr["peer_report"]
        assert (m.node, m.measured_by) == (peer, me) and (p.node, p.measured_by) == (me, peer)
        assert len(m.freq_hz) == len(m.snr_db) == len(m.bits) > 50
        assert np.isfinite(m.snr_db[m.bits >= 0]).all() and np.isnan(p.snr_db).all()
        assert (m.bits > 0).sum() > 0 and (p.bits > 0).sum() > 0
    # what A measured of B is the table B holds for itself, carrier for carrier
    a_meas = next(e for j, e in evs if j == 0 and isinstance(e, ProbeReport) and e.source == "measured")
    b_peer = next(e for j, e in evs if j == 1 and isinstance(e, ProbeReport) and e.source == "peer_report")
    assert (a_meas.bits[a_meas.bits >= 0] == b_peer.bits[b_peer.bits >= 0]).all()


def test_lost_data_retransmitted_delivered_once(pair):
    a, b, _, _ = pair
    n, seen = len(b.got), []
    b.rx_filter = lambda f: (seen.append(f["seq"]) or False) if not seen and f["len"] else True
    assert a.send(b"once only")
    b.rx_filter = None
    assert b.got[n:] == [b"once only"] and a.retry_count == 1 and len(seen) == 1


def test_lost_ack_retransmits_receiver_dedups(pair):
    a, b, _, _ = pair
    n, r0, lost = len(b.got), a.retry_count, []
    a.rx_filter = lambda f: not (f["len"] == 0 and not lost and not lost.append(1))
    assert a.send(b"dedup me")
    a.rx_filter = None
    assert b.got[n:] == [b"dedup me"]                     # the retransmission was not delivered twice
    assert a.retry_count == r0 + 1


def test_gives_up_when_nothing_is_acked(pair):
    a, b, _, _ = pair
    n, f0 = len(b.got), a.fail_count
    a.cfg.retries = 1
    b.rx_filter = lambda f: False
    try:
        assert not a.send(b"lost")
    finally:
        b.rx_filter = None
        a.cfg.retries = 3
    assert a.fail_count == f0 + 1 and b.got[n:] == []
    assert a.send(b"after")                              # the link recovered
    assert b.got[-1] == b"after"


def test_serial_bridge_over_ofdm(tmp_path):
    reg, made = Registry(), []
    try:
        for name, node, peer, rx, tx in (("A", 1, 2, "loop:dc", "loop:cd"),     # own buses: the pair
                                         ("B", 2, 1, "loop:cd", "loop:dc")):    # fixture is still live
            s = Session(reg, 48000, 1024); s.open(rx, tx)
            br = SerialBridge(s, OfdmConfig(node_id=node, peer_id=peer), str(tmp_path / name),
                              buf_size=64)
            br.start(); made.append((br, s))
        pa, pb = HandshakePort(str(tmp_path / "A")), HandshakePort(str(tmp_path / "B"))
        assert pa.wait_cd(True, 120) and pb.wait_cd(True, 30)        # BAT exchange finished
        assert pa.wait_cts(True, 5)
        assert pa.send(b"serial over ofdm\r\n", timeout=60)
        assert pb.read(100, 15) == b"serial over ofdm\r\n"
        assert pb.send(b"reply" * 8, timeout=60)
        assert pa.read(100, 15) == b"reply" * 8
        assert made[0][0].tx_count == 18 and made[1][0].rx_count == 18
    finally:
        for br, s in made:
            br.stop(); s.close()


def _mk_pair(reg, made, bus, **kw):
    """Two links on their own loop buses (node 1 initiator, node 2 responder)."""
    ends = []
    for node, peer, rx, tx in ((1, 2, f"loop:{bus}b", f"loop:{bus}a"),
                               (2, 1, f"loop:{bus}a", f"loop:{bus}b")):
        s = Session(reg, 48000, 1024); s.open(rx, tx)
        link = OfdmLink(s, OfdmConfig(node_id=node, peer_id=peer, **kw))
        link.got, link.ends = [], []
        link.on_data = link.got.append
        link.on_stream = lambda ok, n, l=link: l.ends.append((ok, n))
        made.append((link, s)); ends.append(link)
    for link in ends:
        link.start()
    return ends


@pytest.fixture(scope="module")
def made():
    out = []
    yield out
    for link, s in out:
        link.stop(); s.close()


def test_oneway_only_the_sender_transmits(made):
    a, b = _mk_pair(Registry(), made, "ow", mode="oneway")
    assert a.wait_up(120) and b.wait_up(30), (a.fail_reason, b.fail_reason)
    assert a.capacity == 255 - HDR == b.capacity           # len is a u8; the header holds addr/seq/ack/len/crc
    with pytest.raises(RuntimeError):
        b.send(b"i only listen")
    tx0 = b.tx_frames
    assert a.send(b"one way \x00\xff end")
    big = bytes(range(256)) * 2 + b"tail"              # 516 B -> three frames
    assert a.send(big)
    deadline = time.time() + 40
    while len(b.got) < 4 and time.time() < deadline:
        time.sleep(0.5)
    assert b.got[0] == b"one way \x00\xff end"
    assert b"".join(b.got[1:]) == big and len(b.got[1:]) == 3
    assert b.tx_frames == tx0 == 0 and b.lost_frames == 0 and a.retry_count == 0
    # a frame lost on the air shows up as a seq gap, never as a retransmission
    seen = []
    b.rx_filter = lambda f: bool(seen.append(f["seq"]) or len(seen) != 1)
    n = len(b.got)
    assert a.send(b"a" * 40) and a.send(b"b" * 40)
    deadline = time.time() + 30
    while len(b.got) < n + 1 and time.time() < deadline:
        time.sleep(0.5)
    b.rx_filter = None
    assert b.got[n:] == [b"b" * 40]
    assert a.send(b"c" * 40)
    deadline = time.time() + 30
    while len(b.got) < n + 2 and time.time() < deadline:
        time.sleep(0.5)
    assert b.got[-1] == b"c" * 40 and b.lost_frames == 1


def test_stream_delivers_progressively_from_the_responder(made):
    a, b = _mk_pair(Registry(), made, "st", mode="stream", sender="responder",
                    stream_poll_s=2.0)
    assert a.wait_up(120) and b.wait_up(30), (a.fail_reason, b.fail_reason)
    assert a.capacity > 1000
    payload = bytes((i * 7) & 0xFF for i in range(700))          # binary, many chunks
    with pytest.raises(RuntimeError):
        a.send(b"x")                                             # the initiator only listens
    assert b.send(payload)
    deadline = time.time() + 60
    while not a.ends and time.time() < deadline:
        time.sleep(0.5)
    assert a.ends == [(True, 700)]
    assert b"".join(a.got) == payload
    assert len(a.got) >= 2                                       # arrived in pieces, not at the end
    assert a.rx_errors == 0 and b.tx_frames == 1


def test_serial_bridge_over_oneway(tmp_path):
    reg, made = Registry(), []
    try:
        for name, node, peer, rx, tx in (("A", 1, 2, "loop:owdc", "loop:owcd"),
                                         ("B", 2, 1, "loop:owcd", "loop:owdc")):
            s = Session(reg, 48000, 1024); s.open(rx, tx)
            br = SerialBridge(s, OfdmConfig(node_id=node, peer_id=peer, mode="oneway"),
                              str(tmp_path / name), buf_size=64)
            br.start(); made.append((br, s))
        pa, pb = HandshakePort(str(tmp_path / "A")), HandshakePort(str(tmp_path / "B"))
        assert pa.wait_cd(True, 120) and pb.wait_cd(True, 30)
        assert pa.send(b"one way serial\r\n", timeout=60)
        assert pb.read(100, 20) == b"one way serial\r\n"
        assert not pb.send(b"no talking back", timeout=20)       # TXFAIL: this end only listens
        assert made[1][0].fail_count == 1
    finally:
        for br, s in made:
            br.stop(); s.close()


def test_full_duplex_both_ends_talk_at_once(made):
    a, b = _mk_pair(Registry(), made, "fd", mode="full")
    assert a.wait_up(120) and b.wait_up(30), (a.fail_reason, b.fail_reason)
    # disjoint halves of the band, initiator low
    assert a._tx_band == b._rx_band and b._tx_band == a._rx_band
    assert a._tx_band[1] < b._tx_band[0]
    pa, pb = b"from A \x00\xff" * 4, b"from B " * 9
    res = {}
    ta = threading.Thread(target=lambda: res.setdefault("a", a.send(pa)))
    tb = threading.Thread(target=lambda: res.setdefault("b", b.send(pb)))
    ta.start(); tb.start(); ta.join(120); tb.join(120)
    assert res == {"a": True, "b": True}
    assert b.got == [pa] and a.got == [pb]
    assert a.retry_count == 0 and b.retry_count == 0
    # lost data frame: resent after ack_wait_s, delivered once
    seen = []
    b.rx_filter = lambda f: bool(seen.append(f["seq"]) or len(seen) != 1) if f["len"] else True
    a.cfg.ack_wait_s = 6.0
    assert a.send(b"again")
    b.rx_filter = None
    assert b.got[-1] == b"again" and b.got.count(b"again") == 1 and a.retry_count == 1


def test_full_duplex_c_engine_both_ends_talk_at_once(made):
    """Same scenario as above, but the data plane (codec, filter, gate, ARQ) is the C engine."""
    a, b = _mk_pair(Registry(), made, "cx", mode="full", backend="cdx")
    assert a.wait_up(120) and b.wait_up(30), (a.fail_reason, b.fail_reason)
    assert a._cdx is not None and b._cdx is not None
    assert "c engine" in a.backend
    assert a._tx_band == b._rx_band and b._tx_band == a._rx_band
    assert a._tx_band[1] < b._tx_band[0]
    pa, pb = b"from A \x00\xff" * 4, b"from B " * 9
    res = {}
    ta = threading.Thread(target=lambda: res.setdefault("a", a.send(pa)))
    tb = threading.Thread(target=lambda: res.setdefault("b", b.send(pb)))
    ta.start(); tb.start(); ta.join(120); tb.join(120)
    assert res == {"a": True, "b": True}
    assert b.got == [pa] and a.got == [pb]
    assert a.retry_count == 0 and b.retry_count == 0
    # lost data frame: resent after ack_wait_s, delivered once
    seen = []
    b.rx_filter = lambda f: bool(seen.append(f["seq"]) or len(seen) != 1) if f["len"] else True
    a.cfg.ack_wait_s = 6.0
    assert a.send(b"again")
    b.rx_filter = None
    assert b.got[-1] == b"again" and b.got.count(b"again") == 1 and a.retry_count == 1


def test_phy_settings_reach_the_handshake_config():
    from modemgui.engine.ofdm_link import _imports, apply_phy
    H, _ = _imports()
    before = dict(H.HS_OVERRIDES)
    try:
        apply_phy(OfdmConfig(n_fft=2048, snr_margin_db=6.0, max_bits_per_bin=4))
        cfg = H.hs_cfg(48000)
        assert cfg["n_fft"] == 2048 and cfg["snr_margin_db"] == 6.0 and cfg["max_bits_per_bin"] == 4
    finally:
        H.HS_OVERRIDES = before
