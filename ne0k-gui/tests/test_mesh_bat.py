# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Broadcast BAT round (ofdm_announce, bat_mode="broadcast"): one probe per node, reports, a coordinator confirm."""
import time

import numpy as np
import pytest

pytest.importorskip("scipy")

from modemgui.engine.modemcfg import OfdmConfig
from modemgui.engine.ofdm_announce import (TAB_CONFIRM, TAB_REPORT, AnnounceRound, pack_tables, unpack_tables)
from modemgui.engine.ofdm_link import _imports


def _bat(bits):
    H, _ = _imports()
    return H.bat_to_bytes(np.asarray(bits))


def test_tables_roundtrip():
    tabs = {(1, 2): _bat([0, 2, 2, 1] * 16 + [2, 2, 0]), (3, 2): _bat([2] * 67), (15, 0): _bat([0] * 67)}
    kind, frm, complete, got = unpack_tables(pack_tables(TAB_CONFIRM, 1, tabs, True))
    assert (kind, frm, complete) == (TAB_CONFIRM, 1, True)
    assert got == tabs
    assert unpack_tables(pack_tables(TAB_REPORT, 2, {}))[:3] == (TAB_REPORT, 2, False)
    assert unpack_tables(pack_tables(TAB_REPORT, 2, tabs)[:-1]) is None        # truncated
    H, _ = _imports()
    assert H.unpack_msg(H.pack_msg(H.MSG_BAT_TABLES, 0, b"x"))[0] == H.MSG_BAT_TABLES


def test_probe_slot_attribution():
    """A probe belongs to the next expected slot; one heard a whole slot (or more) late skips the silent ones; our
    own slot is never credited to anyone."""
    an = AnnounceRound(None, OfdmConfig(), 2, log=lambda m: None, shared_rx=True, bat_mode="broadcast")
    an._seq_start("probe", [1, 2, 3, 4], 10.0)
    t = an._seq["last_end"] + an.guard_s
    assert an._seq_attribute("probe", t + 0.5, t + 6.0) == 1
    an._seq_heard("probe", 2, t + 14.0)                    # our own probe
    assert an._seq_attribute("probe", t + 15.5, t + 21.0) == 3
    an._seq_start("probe", [1, 2, 3, 4], 10.0)
    t = an._seq["last_end"] + an.guard_s
    assert an._seq_attribute("probe", t + 21.0, t + 27.0) == 4    # 1 silent, 2 is us, 3 silent: two slots late
    assert an._seq_attribute("probe", t + 30.0, t + 35.0) is None  # nobody left


def test_nobody_may_answer_a_probe_of_the_broadcast_round():
    """Probes carry no node id: during the round no responder link may take one for its pair's, and nobody may
    start a pairwise handshake (a waiting responder answered B's probe right over C's)."""
    an = AnnounceRound(None, OfdmConfig(), 1, log=lambda m: None, shared_rx=True, bat_mode="broadcast")
    an.known = {2, 3}
    an.phase, an.idx = "bcast", 0
    assert not any(an.responder_may_listen(i) for i in (1, 2, 3))
    assert not an.initiator_may_start()


def test_broadcast_needs_shared_receiver():
    with pytest.raises(ValueError):
        AnnounceRound(None, OfdmConfig(), 1, log=lambda m: None, bat_mode="broadcast")


def test_three_nodes_train_in_one_broadcast_round():
    """Real audio on one shared medium: announce round, then the broadcast round brings every ring link up from the
    confirm's tables (no pairwise handshake on the air), and a payload goes round on them."""
    from modemgui.audio import Registry
    from modemgui.engine import Session
    from modemgui.engine.mesh_node import MeshConfig, MeshNode
    from modemgui.engine.modemcfg import CssConfig

    reg, nodes, logs = Registry(), {}, {1: [], 2: [], 3: []}
    try:
        for nid in (1, 2, 3):
            s = Session(reg, 48000, 1024)
            s.open(f"mix:bbat/{nid}", f"mix:bbat/{nid}")
            n = MeshNode(s, MeshConfig(node_id=nid, phy_mode="ofdm_only", announce_round=True,
                                       announce_bat_mode="broadcast"),
                         CssConfig(), OfdmConfig(),
                         log=lambda m, i=nid: (logs[i].append(m), print(f"[{i}] {m}", flush=True)))
            n.ring_dst_fn = lambda i=nid: i % 3 + 1
            n.rx_peer = (nid - 2) % 3 + 1
            n.got = []
            n.on_data = lambda src, d, n=n: n.got.append((src, d))
            nodes[nid] = (n, s)
        for n, _ in nodes.values():
            n.start()
            n._ensure_rx_link(n.rx_peer)          # as the GUI does (_listen_rx_host): it must stay out of the round
        end = time.monotonic() + 240
        while time.monotonic() < end:
            if all(any("broadcast BAT done" in m for m in logs[i]) for i in logs):
                break
            time.sleep(1.0)
        for i, (n, _) in nodes.items():
            done = [m for m in logs[i] if "broadcast BAT done" in m]
            assert done and "complete=True" in done[0] and "my links up" in done[0], (i, done)
            assert not any("hs rx probe" in m or "tx probe" in m for m in logs[i]), i     # no pairwise handshake
            assert n.links[(i % 3 + 1, "tx")].up and n.links[((i - 2) % 3 + 1, "rx")].up
        n1 = nodes[1][0]
        assert n1.send(2, b"over the broadcast tables")
        assert nodes[2][0].got == [(1, b"over the broadcast tables")]
        assert nodes[3][0].got == []
    finally:
        for n, s in nodes.values():
            n.stop(); s.close()


def test_incomplete_confirm_falls_back_to_pairwise_turns():
    """The coordinator misses node 3's report: its confirm is incomplete, so the links that needed node 3's
    measurements are trained pairwise in the BAT turns (the rest stay on the broadcast tables)."""
    from modemgui.audio import Registry
    from modemgui.engine import Session
    from modemgui.engine.mesh_node import MeshConfig, MeshNode
    from modemgui.engine.modemcfg import CssConfig

    reg, nodes, logs = Registry(), {}, {1: [], 2: [], 3: []}
    try:
        for nid in (1, 2, 3):
            s = Session(reg, 48000, 1024)
            s.open(f"mix:bbat2/{nid}", f"mix:bbat2/{nid}")
            n = MeshNode(s, MeshConfig(node_id=nid, phy_mode="ofdm_only", announce_round=True,
                                       announce_bat_mode="broadcast"),
                         CssConfig(), OfdmConfig(),
                         log=lambda m, i=nid: (logs[i].append(m), print(f"[{i}] {m}", flush=True)))
            n.ring_dst_fn = lambda i=nid: i % 3 + 1
            n.rx_peer = (nid - 2) % 3 + 1
            nodes[nid] = (n, s)
        for n, _ in nodes.values():
            n.start()
        an1 = nodes[1][0].announce
        real = an1._on_msg

        def deaf_to_3(got, end_t, real=real):
            H, _ = _imports()
            if got[1] == H.MSG_BAT_TABLES and got[3][1] == 3:
                return
            real(got, end_t)
        an1._on_msg = deaf_to_3
        end = time.monotonic() + 420
        while time.monotonic() < end and not all(any("BAT turns done" in m for m in logs[i]) for i in logs):
            time.sleep(1.0)
        for i in logs:
            done = [m for m in logs[i] if "broadcast BAT done" in m]
            assert done and "complete=False" in done[0], (i, done)
        for i, (n, _) in nodes.items():
            assert n.links[(i % 3 + 1, "tx")].up and n.links[((i - 2) % 3 + 1, "rx")].up, i
        assert any("bat exchanged (broadcast round)" in m for m in logs[1] if m.startswith("ofdm[2]"))   # 1->2
        assert any(m.startswith("ofdm[3] bat exchanged tx=") for m in logs[2])                            # 2->3
        assert any(m.startswith("ofdm[1] bat exchanged tx=") for m in logs[3])                            # 3->1
    finally:
        for n, s in nodes.values():
            n.stop(); s.close()


def test_data_turns_carry_on_across_cycles():
    """A new cycle's data turns continue the rotation; restarting at the lowest id left the last id waiting a whole
    extra rotation (96 s on 3 nodes) and its link idle past the reaper."""
    import types

    an = AnnounceRound(None, OfdmConfig(), 1, log=lambda m: None)
    an.known = {2, 3}
    seen = []

    def run(n):
        calls = []

        def wait(_s):
            seen.append(an._data_who)
            calls.append(1)
            return len(calls) >= n                   # "stopped" after n turns
        an._stop = types.SimpleNamespace(is_set=lambda: False, wait=wait)
        an._data_turns(time.monotonic() + 1e6)
    run(2)
    run(3)
    assert seen == [1, 2, 3, 1, 2]
    assert an.max_turn_gap_s() >= 2 * 3 * an.data_turn_s
