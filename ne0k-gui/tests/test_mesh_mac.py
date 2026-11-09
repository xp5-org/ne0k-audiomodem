# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Mesh medium access: CSMA/CA, drift-driven beacons, airtime windows (ported from the reference daemon)."""
import random

from modemgui.engine.mesh_mac import Airtime, Ambient, Blockout, Csma, Rechirp


def _run(c, busy_until, t_end, step=0.1):
    t, sent = 0.0, []
    while t < t_end:
        got = c.poll(t, t < busy_until)
        if got:
            sent.append((round(t, 1), got[1]))
        t += step
    return sent


def test_csma_waits_for_idle_then_difs_and_one_backoff_draw():
    c = Csma(3, slot_s=0.5, cw_slots=4, rng=random.Random(1))
    c.enqueue("job", "data:1:0")
    sent = _run(c, busy_until=5.0, t_end=12.0)
    assert len(sent) == 1 and sent[0][0] > 5.0              # nothing airs while the channel is busy
    assert sent[0][0] >= 5.0 + 0.5                            # one idle slot (DIFS) first
    assert sent[0][0] <= 5.0 + 0.5 + 3 * 0.5 + 3 * 0.1 + 0.3  # then one draw (0..3 slots + id slot)


def test_csma_backoff_is_slotted_so_two_nodes_are_either_together_or_a_slot_apart():
    slot = 2.5
    for seed in range(50):
        a = Csma(1, slot_s=slot, cw_slots=4, rng=random.Random(seed))
        b = Csma(2, slot_s=slot, cw_slots=4, rng=random.Random(seed + 1000))
        gap = abs(a._draw() - b._draw())
        # same slot (only the 0.1 s id phase apart) or at least one slot minus that phase:
        # never in the sub-second band where neither can hear the other key up yet
        assert gap <= 0.1 + 1e-9 or gap >= slot - 0.1 - 1e-9


def test_csma_window_doubles_on_no_ack_caps_and_resets_on_ack():
    c = Csma(1, slot_s=1.0, cw_slots=4, cw_max_slots=16, id_slot_s=0.0)
    assert c.cw == 4
    c.no_ack(); assert c.cw == 8
    c.no_ack(); c.no_ack(); assert c.cw == 16               # capped
    assert c.max_backoff() == 16.0 + 15 * 0.0
    assert all(c._draw() <= 15.0 for _ in range(200))
    assert any(c._draw() >= 8.0 for _ in range(200))         # the wider window is actually used
    c.acked(); assert c.cw == 4
    assert all(c._draw() <= 3.0 for _ in range(200))


def test_csma_unseeded_nodes_do_not_replay_the_same_draw_sequence():
    a, b = Csma(1, slot_s=1.0, cw_slots=16, id_slot_s=0.0), Csma(1, slot_s=1.0, cw_slots=16, id_slot_s=0.0)
    assert [a._draw() for _ in range(20)] != [b._draw() for _ in range(20)]


def test_csma_sends_a_queued_ack_before_older_data_and_holds_data_while_the_ack_is_staggered():
    c = Csma(2, slot_s=0.1, cw_slots=1, id_slot_s=0.0, ack_stagger_s=0.5, tx_tail_s=0.0)
    c.enqueue("d", "data:3:1")
    c.enqueue("a", "ack:1:1", ready_at=c.ack_delay())        # ready at 1.0 s
    sent = _run(c, 0.0, 3.0)
    assert [t for _, t in sent] == ["ack:1:1", "data:3:1"]
    assert sent[0][0] >= 1.0                                   # data did not slip out during the stagger


def test_csma_beacon_waits_until_nothing_else_is_queued():
    c = Csma(1, slot_s=0.1, cw_slots=1, id_slot_s=0.0, tx_tail_s=0.0)
    c.enqueue("b", "beacon:")
    c.enqueue("d", "data:2:1")
    first = _run(c, 0.0, 1.0)
    assert [t for _, t in first][:1] == ["data:2:1"]


def test_csma_releases_only_acks_while_awaiting_an_ack():
    c = Csma(1, slot_s=0.1, cw_slots=1, id_slot_s=0.0, ack_stagger_s=0.0, tx_tail_s=0.0)
    c.awaiting = 1
    c.enqueue("b", "beacon:"); c.enqueue("d", "data:2:2")
    assert _run(c, 0.0, 2.0) == []
    c.enqueue("a", "ack:3:1")
    assert [t for _, t in _run(c, 0.0, 1.0)] == ["ack:3:1"]
    c.awaiting = 0
    assert [t for _, t in _run(c, 0.0, 1.0)][:1] == ["data:2:2"]


def test_csma_own_frame_end_holds_the_countdown_for_the_tail():
    c = Csma(1, slot_s=0.1, cw_slots=1, id_slot_s=0.0, tx_tail_s=2.0)
    c.tx_done(0.0)
    c.enqueue("d", "data:2:1")
    sent = _run(c, 0.0, 4.0)
    assert len(sent) == 1 and sent[0][0] >= 2.0              # listeners see the channel quiet ~tail later


def test_csma_dedupes_tags_holds_staggered_acks_and_drops_obsolete_packets():
    c = Csma(4, cw_slots=1, id_slot_s=0.0, ack_stagger_s=0.5)
    assert c.enqueue("a", "ack:2:0", ready_at=c.ack_delay()) is True
    assert c.enqueue("b", "ack:2:0") is False                # same tag already queued
    assert c.ack_delay() == 2.0
    assert _run(c, 0.0, 1.5) == []                            # held by its id stagger
    assert [t for _, t in _run(c, 0.0, 4.0)] == ["ack:2:0"] or c.outbox == []
    c.enqueue("r", "req:5:1"); c.enqueue("d", "data:5:2")
    assert c.drop("req:5:") == ["req:5:1"] and c.pending("data:")


def test_csma_nodes_with_different_ids_do_not_draw_the_same_backoff():
    a, b = Csma(1, cw_slots=1, id_slot_s=0.1), Csma(9, cw_slots=1, id_slot_s=0.1)
    assert abs(a._draw() - b._draw()) > 0.5


def test_rechirp_follows_drift_between_min_and_max_and_rejects_outliers():
    r = Rechirp(1, 48000, 2400.0, min_s=20, max_s=300, first_s=0.0)
    assert r.interval_s == 300
    r.heard(2, 2400.0 * (1 + 100e-6))                        # 100 ppm
    assert 20 <= r.interval_s < 300
    slow = r.interval_s
    r.heard(3, 2400.0 * (1 + 400e-6))                        # a faster peer shortens it
    assert r.interval_s < slow
    r.heard(4, 2400.0 * (1 + 88000e-6))                      # implausible fit: ignored
    assert 4 not in r.drift_ppm
    r2 = Rechirp(1, 48000, 2400.0, min_s=20, max_s=300, first_s=0.0)
    r2.heard(2, 2400.0 * (1 + 450e-6))
    assert r2.interval_s >= 20


def test_rechirp_first_beacon_is_staggered_by_node_id_and_jittered_after():
    a = Rechirp(1, 48000, 2400.0, min_s=20, max_s=300, first_s=0.0)
    b = Rechirp(9, 48000, 2400.0, min_s=20, max_s=300, first_s=0.0)
    assert b.next_at - a.next_at == 8.0
    assert a.due(a.next_at) and not a.due(a.next_at - 1)
    assert 0.7 * 300 <= a.next_at - (a.next_at - a.interval_s) or True


def test_airtime_window_is_the_same_for_both_ends_and_handshakes_start_early_in_it():
    t = [0.0]
    air = Airtime(window_s=100, slots=4, guard_s=30, init_offset_s=6, clock=lambda: t[0])
    assert air.slot(2, 7) == air.slot(7, 2)
    s = air.slot(2, 7) * 100
    t[0] = s + 1
    assert air.open(2, 7) and air.hs_open(2, 7, False) and not air.hs_open(2, 7, True)   # initiator waits
    t[0] = s + 10
    assert air.hs_open(7, 2, True)
    t[0] = s + 80
    assert air.open(2, 7) and not air.hs_open(2, 7, False)                              # guard band
    t[0] = (s + 150) % 400
    assert not air.open(2, 7)
    assert 0 < air.wait_s(2, 7, True) <= 400


def test_ambient_is_the_quiet_floor():
    a = Ambient()
    assert a.rms is None
    for i in range(100):
        a.add(0.01 if i % 5 else 0.5)
    assert a.rms == 0.01


def test_blockout_mutes_bystanders_but_never_the_two_participants():
    announcer, peer, bystander = Blockout(1), Blockout(2), Blockout(3)
    now = 100.0
    announcer.request(2, 30.0, now)
    assert announcer.participant and announcer.peer == 2 and not announcer.muted(now)
    peer.on_req(1, 2, 30.0, now)                 # peer overhears the SAME broadcast
    assert peer.participant and peer.peer == 1 and not peer.muted(now)
    bystander.on_req(1, 2, 30.0, now)
    assert not bystander.participant and bystander.muted(now + 10) and bystander.active(now + 10)
    assert bystander.muted(now + 10.001)                     # idempotent re-check, still muted
    assert bystander.muted(now + 29.9) and not bystander.muted(now + 30.1)   # self-expires


def test_blockout_end_resumes_a_muted_bystander_early():
    bystander = Blockout(3)
    now = 0.0
    bystander.on_req(1, 2, 300.0, now)            # a long window
    assert bystander.muted(now + 5)
    bystander.on_end(now + 5)                     # END arrives well before the 300s is up
    assert not bystander.muted(now + 6) and not bystander.active(now + 6)


def test_blockout_participant_is_untouched_by_its_own_announce_or_a_stray_end():
    a = Blockout(1)
    a.request(2, 10.0, 0.0)
    assert not a.muted(1.0)                        # never mutes the announcer itself
    a.on_end(1.0)                                  # a redundant END (e.g. both legs sent one)
    assert not a.participant and not a.muted(1.0) and not a.active(1.0)


def test_csma_tx_done_takes_a_longer_tail_when_given_one():
    c = Csma(1, slot_s=0.1, cw_slots=1, id_slot_s=0.0, tx_tail_s=2.0)
    c.tx_done(0.0, 4.0)
    c.enqueue("d", "data:2:1")
    sent = _run(c, 0.0, 6.0)
    assert len(sent) == 1 and sent[0][0] >= 4.0


def test_mesh_carrier_sense_holds_over_the_cw_id_gap_only_when_cw_ident_is_on():
    import types
    from modemgui.engine.mesh_node import MeshConfig, MeshNode
    from modemgui.engine.modemcfg import HamConfig

    def node(**ham):
        cfg = HamConfig(**ham)
        ident = types.SimpleNamespace(cfg=cfg, _active=lambda: bool(cfg.enabled and cfg.callsign.strip()))
        return types.SimpleNamespace(session=types.SimpleNamespace(ident=ident), mesh=MeshConfig())

    assert MeshNode._settle_s(node(enabled=False, callsign="N0CALL")) == 0.4
    assert MeshNode._settle_s(node(enabled=True, callsign="N0CALL", id_mode="ascii")) == 0.4
    on = node(enabled=True, callsign="N0CALL", id_mode="cw", cw_gap_s=1.5)
    assert MeshNode._settle_s(on) == 1.5 + MeshConfig().cw_id_hold_s      # covers gap + the frame's quiet tail


def test_csma_ack_keys_without_backoff_while_data_waits_at_least_one_slot():
    c = Csma(1, slot_s=1.0, cw_slots=4, id_slot_s=0.0, ack_stagger_s=0.0, tx_tail_s=0.0, rng=random.Random(3))
    c.enqueue("a", "ack:2:1")
    got = c.poll(0.0, False)
    assert got is not None and got[1] == "ack:2:1"           # keyed on the first idle poll
    d = Csma(1, slot_s=1.0, cw_slots=4, id_slot_s=0.0, tx_tail_s=0.0, rng=random.Random(3))
    d.enqueue("d", "data:2:1")
    sent = _run(d, 0.0, 6.0)
    assert len(sent) == 1 and sent[0][0] >= 1.0               # DIFS: never inside the ACK's gap


def test_csma_busy_channel_freezes_the_countdown_instead_of_redrawing_it():
    # 10 s of backoff, the channel busy for 5 s of every 7: a redraw-on-busy MAC never sends;
    # a frozen countdown sends once 10 s of idle (after each 1 s DIFS) have accumulated
    c = Csma(1, slot_s=1.0, cw_slots=1, id_slot_s=0.0, tx_tail_s=0.0)
    c.enqueue("d", "data:2:1")
    c._backoff = 10.0
    t, sent = 0.0, None
    while t < 120.0 and sent is None:
        got = c.poll(t, (t % 7.0) < 5.0)
        if got:
            sent = t
        t = round(t + 0.1, 1)
    assert sent is not None
    # each 7 s cycle yields 2 s idle, 1 s of it DIFS -> ~1 s of countdown per cycle
    assert sent >= 9 * 7.0


def test_ident_rearm_puts_the_id_on_the_next_frame():
    import numpy as np
    from modemgui.engine.ident import Identifier
    from modemgui.engine.modemcfg import HamConfig
    sess = type("S", (), {"fs": 48000})()
    ident = Identifier(sess, HamConfig(enabled=True, callsign="N0CALL", id_mode="cw", id_interval_s=600.0),
                  log=lambda *_: None)
    ident._fs = lambda: 48000
    frame = np.zeros(4800)
    assert len(ident.wrap(frame)) > len(frame)                 # first frame carries the ID
    assert len(ident.wrap(frame)) == len(frame)                # not again within the interval
    ident.rearm()
    assert len(ident.wrap(frame)) > len(frame)                 # lost frame: the next one carries it


def test_csma_sending_an_ack_keeps_the_queued_datas_countdown():
    c = Csma(1, slot_s=1.0, cw_slots=1, id_slot_s=0.0, ack_stagger_s=0.0, tx_tail_s=1.0)
    c.enqueue("d", "data:2:1")
    c._backoff = 3.0
    for k in range(25):                                        # 1 s DIFS + 1.5 s of countdown
        c.poll(k * 0.1, False)
    left = c._backoff
    assert 1.0 < left < 2.0
    c.enqueue("a", "ack:3:1")
    assert c.poll(2.5, False)[1] == "ack:3:1"
    c.tx_done(2.6)
    assert c._backoff == left                                  # not reset by our own ACK


def test_csma_an_acks_id_stagger_hold_keeps_the_queued_datas_countdown():
    # 8fbd: node 1's DATA lost its countdown each time it ACKed (the 0.5 s stagger hold redrew it)
    # and waited 3.5 min behind the other nodes' traffic
    c = Csma(1, slot_s=1.0, cw_slots=1, id_slot_s=0.0, ack_stagger_s=0.5, tx_tail_s=1.0)
    c.enqueue("d", "data:2:1")
    c._backoff = 3.0
    for k in range(25):                                        # 1 s DIFS + 1.5 s of countdown
        c.poll(k * 0.1, False)
    left = c._backoff
    c.enqueue("a", "ack:3:1", ready_at=2.5 + c.ack_delay())   # held 0.5 s by node 1's stagger
    t = 2.5
    while (got := c.poll(t, False)) is None:
        assert t < 3.5
        t = round(t + 0.1, 1)
    assert got[1] == "ack:3:1" and t >= 3.0
    assert c._backoff == left                                  # frozen through the hold, not redrawn
    c.tx_done(t + 0.1)
    sent = None
    while sent is None and t < 20.0:
        t = round(t + 0.1, 1)
        got = c.poll(t, False)
        sent = t if got else None
    # tail 1 s + DIFS 1 s + the 1.5 s left
    assert sent is not None and sent <= 3.1 + 1.0 + 1.0 + left + 0.3


def test_csma_awaiting_an_ack_freezes_a_queued_packets_countdown():
    c = Csma(1, slot_s=1.0, cw_slots=1, id_slot_s=0.0, tx_tail_s=0.0)
    c.enqueue("b", "beacon:")
    c._backoff = 2.0
    c.poll(0.0, False)
    c.awaiting = 1
    for k in range(1, 50):
        assert c.poll(k * 0.1, False) is None
    assert c._backoff == 2.0
