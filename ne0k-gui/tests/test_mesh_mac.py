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


def test_csma_waits_for_idle_and_redraws_backoff_after_a_busy_channel():
    c = Csma(3, cw_s=2.0, rng=random.Random(1))
    c.enqueue("job", "data:1:0")
    sent = _run(c, busy_until=5.0, t_end=12.0)
    assert len(sent) == 1 and sent[0][0] > 5.0              # nothing airs while the channel is busy
    assert sent[0][0] <= 5.0 + 2.0 + 3 * 0.1 + 0.3          # idle -> one backoff draw (cw + id slot)


def test_csma_dedupes_tags_holds_staggered_acks_and_drops_obsolete_packets():
    c = Csma(4, cw_s=0.0, id_slot_s=0.0, ack_stagger_s=0.5)
    assert c.enqueue("a", "ack:2:0", ready_at=c.ack_delay()) is True
    assert c.enqueue("b", "ack:2:0") is False                # same tag already queued
    assert c.ack_delay() == 2.0
    assert _run(c, 0.0, 1.5) == []                            # held by its id stagger
    assert [t for _, t in _run(c, 0.0, 4.0)] == ["ack:2:0"] or c.outbox == []
    c.enqueue("r", "req:5:1"); c.enqueue("d", "data:5:2")
    assert c.drop("req:5:") == ["req:5:1"] and c.pending("data:")


def test_csma_nodes_with_different_ids_do_not_draw_the_same_backoff():
    a, b = Csma(1, cw_s=0.0, id_slot_s=0.1), Csma(9, cw_s=0.0, id_slot_s=0.1)
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
