# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""MeshReceiver: one decode per burst on a shared channel, routed by the frame's [src:4|dst:4] header."""
import types

import numpy as np

from modemgui.engine.mesh_rx import MeshReceiver
from modemgui.engine.modemcfg import OfdmConfig
from modemgui.engine.ofdm_link import BROADCAST_ID, pack_frame


class _Dem:
    def __init__(self, raw):
        self.raw, self.calls = raw, 0

    def decode_samples(self, x):
        self.calls += 1
        return ([{"evm_pct": 3.0}], {0: self.raw}) if self.raw is not None else ([], {})


class _Link:
    def __init__(self, me, peer, role, bat, dem, inflight=False):
        self.cfg = OfdmConfig(node_id=me, peer_id=peer, mode="arq", sender="initiator")
        self.role, self.up, self.shared_rx = role, True, True
        self._bat, self._dem, self._inflight = np.asarray(bat), dem, inflight
        self.got = []

    def rx_demod(self):
        return self._dem, self._bat, (400.0, 4000.0)

    def wants_ack(self):
        return self._inflight

    def deliver(self, seq, ack, data, stats, n, t0, t1):
        self.got.append((seq, ack, data))


def _rx(me, links, expected=()):
    r = MeshReceiver(types.SimpleNamespace(pipe=None), OfdmConfig(node_id=me), me, log=lambda m: None)
    r.links, r.expected = (lambda: links), (lambda: list(expected))
    r.data_first = lambda: True
    r._cap.add(np.zeros(48000 * 4, np.float32))
    return r


def test_identical_tables_decode_once_and_route_data_to_the_listening_link():
    raw = pack_frame(0, 0, b"tele", 2, 1)                          # node 2 -> node 1 (us)
    dem = _Dem(raw)
    to2 = _Link(1, 2, "responder", [2] * 8, dem)                     # listens to 2 (2 is the sender there)
    to3 = _Link(1, 3, "initiator", [2] * 8, dem, inflight=True)      # same table: one decode covers both
    r = _rx(1, [to3, to2], expected=[2])
    assert [lk.cfg.peer_id for lk in r._candidates()] == [2]
    assert r._data(48000, 96000)
    assert dem.calls == 1 and to2.got == [(0, 0, b"tele")] and to3.got == []


def test_frame_for_another_station_is_dropped_once():
    dem = _Dem(pack_frame(5, 1, b"x", 2, 3))                        # 2 -> 3, we are 1
    a, b = _Link(1, 2, "responder", [2] * 8, dem), _Link(1, 3, "initiator", [2] * 8, dem)
    r = _rx(1, [a, b])
    assert r._data(48000, 96000)
    assert r.foreign == 1 and a.got == b.got == [] and dem.calls == 1


def test_ack_goes_to_the_link_waiting_for_it_and_broadcast_is_ours():
    dem = _Dem(pack_frame(255, 1, b"", 2, 1))
    listen = _Link(1, 2, "responder", [2] * 8, dem)
    sender = _Link(1, 2, "initiator", [2] * 8, dem, inflight=True)
    r = _rx(1, [listen, sender])
    assert r._data(48000, 96000) and sender.got == [(255, 1, b"")] and listen.got == []
    dem.raw = pack_frame(3, 0, b"all", 2, BROADCAST_ID)
    assert r._data(48000, 96000) and listen.got == [(3, 0, b"all")]


def test_a_different_table_is_tried_only_when_the_first_finds_nothing():
    bad, good = _Dem(None), _Dem(pack_frame(0, 0, b"hi", 3, 1))
    l2 = _Link(1, 2, "responder", [2] * 8, bad)
    l3 = _Link(1, 3, "responder", [1] * 8, good)
    r = _rx(1, [l2, l3], expected=[2])
    assert [lk.cfg.peer_id for lk in r._candidates()] == [2, 3]
    assert r._data(48000, 96000)
    assert bad.calls == 1 and good.calls == 1 and l3.got == [(0, 0, b"hi")]
    bad.raw = good.raw = None
    assert not r._data(48000, 96000) and r.undecodable == 1      # nothing decoded: the caller tries a message


def test_a_merged_burst_is_split_at_inner_quiet():
    """BAT_ACK, 1.0 s of quiet, GO: the gate needs 0.8 s measured on samples, polls made it ~1.05 s, and the two
    merged -- the message decode then returned only the first.  Inner quiet >= 0.7 s splits them again."""
    fs = 48000
    r = _rx(1, [])
    r._cap = type(r._cap)(fs, keep_s=60.0)
    tone = lambda s: 0.1 * np.sin(2 * np.pi * 1000 * np.arange(int(s * fs)) / fs).astype(np.float32)
    quiet = lambda s: np.zeros(int(s * fs), np.float32)
    r._cap.add(np.concatenate([quiet(1.0), tone(3.4), quiet(1.0), tone(3.3), quiet(0.6), tone(0.2)]))
    parts = r._parts(int(1.0 * fs), r._cap.n)
    assert len(parts) == 2                                    # the 0.6 s gap (a CW word gap is 0.56 s) stays inside
    assert abs(parts[0][0] / fs - 1.0) < 0.06 and abs(parts[1][0] / fs - 5.4) < 0.06


def test_round_slot_waits_for_a_lower_announce_decoded_while_waiting():
    """The real AnnounceRound._round for id 3: its no-lower target has passed while a burst is still pending; the
    burst turns out to be id 1's announce.  It must then wait guard + one slot after it (B's slot), not key at once
    with the stale target (C keyed in B's slot on 2026-11-06)."""
    import threading
    import time as _t
    from modemgui.engine.ofdm_announce import AnnounceRound
    an = AnnounceRound(types.SimpleNamespace(), OfdmConfig(), 3, log=lambda m: None, listen_s=0.0, guard_s=0.5)
    an._clear = lambda quiet_s=0.8: True
    sent = []
    an._announce = lambda flags=0: sent.append(_t.monotonic())
    an.rx_open()                                           # a burst is on the air
    th = threading.Thread(target=an._round, args=(0.3,), daemon=True)
    th.start()
    _t.sleep(1.0)                                          # own target (2 slots = 0.6 s) has passed meanwhile
    assert not sent
    t1 = _t.monotonic()
    with an._lock:
        an.heard_end[1], an.masks[1] = t1, 0b10            # rx_msg: id 1's announce
    an.rx_done(None, 0.0)
    th.join(5.0)
    an._stop.set()
    assert sent and sent[0] - t1 >= 0.5 + 0.3 - 0.05       # guard + one slot after id 1


def test_own_bursts_keep_a_gap_even_from_two_threads():
    """An ACK (receiver thread) and a DATA (serial thread) of one station: the second must start OWN_GAP_S after
    the first ends, or the receivers' gate merges them into one burst and loses the second frame."""
    import threading
    import time as _t
    from modemgui.engine.ofdm_link import OWN_GAP_S, keyup
    s = types.SimpleNamespace(_keyup_end=_t.monotonic() - 10)
    fs, ends = 48000, []
    def go():
        a = keyup(s, np.zeros(int(0.3 * fs), np.float32), fs, lead_s=0.0)
        ends.append((s._keyup_end - len(a) / fs, s._keyup_end))
    ths = [threading.Thread(target=go) for _ in range(2)]
    [t.start() for t in ths]; [t.join(5) for t in ths]
    (a0, a1), (b0, b1) = sorted(ends)
    assert b0 - a1 >= OWN_GAP_S - 0.02


def test_burst_heard_during_a_decode_keeps_its_real_start():
    """A burst that begins -- or begins and ends -- while the previous one is being decoded must be closed with
    the start it really had, not the moment the gate looked again (a GO's frame was lost so)."""
    import threading
    import time

    fs = 48000
    r = _rx(1, [])
    r._cap = type(r._cap)(fs, keep_s=60.0)
    tone = lambda s: (0.3 * np.sin(2 * np.pi * 1500 * np.arange(int(s * fs)) / fs)).astype(np.float32)
    quiet = lambda s: np.zeros(int(s * fs), np.float32)
    closed = []

    def slow_closed(a, b):
        closed.append((a, b))
        if len(closed) == 1:                  # "decoding" the first burst: the second one plays meanwhile
            r._cap.add(np.concatenate([quiet(0.5), tone(0.6), quiet(1.2)]))
            time.sleep(0.1)
    r._closed = slow_closed
    th = threading.Thread(target=r._run, daemon=True)
    th.start()
    time.sleep(0.3)
    base = r._cap.n
    r._cap.add(np.concatenate([tone(1.0), quiet(1.0)]))
    end = time.monotonic() + 5
    while len(closed) < 2 and time.monotonic() < end:
        time.sleep(0.05)
    r._stop.set(); th.join(2)
    assert len(closed) == 2, closed
    blk = int(0.05 * fs)
    (a1, b1), (a2, b2) = closed
    assert abs(a1 - base) <= blk and abs(b1 - (base + fs)) <= blk
    s2 = base + 2 * fs + int(0.5 * fs)
    assert abs(a2 - s2) <= blk and abs(b2 - (s2 + int(0.6 * fs))) <= blk


def test_two_messages_closer_than_the_split_gap_both_decode():
    """A BAT_ACK and the GO answering it 0.4 s later (with its keyup tone) land in one part: both must come out."""
    from modemgui.engine.ofdm_link import _imports, _quiet

    H, _ = _imports()
    fs, band = 48000, (400.0, 4000.0)
    with _quiet():
        ack = H.make_msg_audio(fs, H.MSG_BAT_ACK, 0x1234, b"", band)
        go = H.make_msg_audio(fs, H.MSG_GO, 0x5678, b"", band)
    t = np.arange(int(2.0 * fs)) / fs
    keyup_tone = (0.2 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    z = lambda s: np.zeros(int(s * fs), np.float32)
    part = np.concatenate([z(0.5), 0.3 * ack, z(0.4), keyup_tone, 0.3 * go, z(0.5)]).astype(np.float32)
    r = _rx(1, [])
    got = []
    r.on_msg = got.append
    assert r._msgs(part, 0.0) == 2
    assert [g[1] for g in got] == [H.MSG_BAT_ACK, H.MSG_GO]
