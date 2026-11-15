# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Hub & spoke MAC (engine/hub_mac.py): page framing, and three nodes on one shared medium."""
import threading
import time

import pytest

from modemgui.engine.hub_mac import (BROADCAST_ID, K_POLL, PAGE_HDR, Reassembly, pack_hub, split_page, unpack_hub)


def test_page_split_and_reassembly():
    data = bytes(range(256)) * 2
    pieces = split_page(2, 3, 7, data, 100)
    assert all(len(p) <= 100 for p in pieces) and len(pieces) == -(-len(data) // (100 - PAGE_HDR))
    r = Reassembly()
    got = [r.add(1, p) for p in pieces]
    assert got[:-1] == [None] * (len(pieces) - 1)
    assert got[-1] == (2, 3, 7, data)
    # a missing fragment drops the page instead of splicing two together
    assert [r.add(1, p) for p in (pieces[0], pieces[2])] == [None, None]
    assert split_page(1, BROADCAST_ID, 0, b"", 249) == [bytes([0x1F, 0, 0x80])]


def test_hub_message_roundtrip():
    assert unpack_hub(pack_hub(K_POLL, 1, 3, b"\x02\x58")) == (K_POLL, 1, 3, b"\x02\x58")
    assert unpack_hub(b"\x7f\x01\x02") is None and unpack_hub(b"\x02") is None


def test_client_refuses_to_address_anyone_but_the_hub():
    from types import SimpleNamespace
    from modemgui.engine.hub_mac import HubMac, K_JOIN
    from modemgui.engine.modemcfg import OfdmConfig
    node = SimpleNamespace(me=2, ofdm_cfg=OfdmConfig(), rx=None, session=None)
    h = HubMac(node, hub_id=1, n_nodes=3, relay=False)
    with pytest.raises(RuntimeError):
        h.send_msg(K_JOIN, 3)
    assert h.send_page(3, b"x") is False          # no relay: a client's page goes to the hub only


def test_three_nodes_hub_relay():
    """Hub 1, clients 2 and 3, relay on: join, training, a page to the hub, a broadcast to every client, and a
    station -> station page stored and forwarded by the hub."""
    pytest.importorskip("scipy")
    from modemgui.audio import Registry
    from modemgui.engine import Session
    from modemgui.engine.mesh_node import MeshConfig, MeshNode
    from modemgui.engine.modemcfg import CssConfig, OfdmConfig

    reg, nodes, logs = Registry(), {}, {1: [], 2: [], 3: []}
    try:
        for nid in (1, 2, 3):
            s = Session(reg, 48000, 1024)
            s.open(f"mix:hub/{nid}", f"mix:hub/{nid}")
            n = MeshNode(s, MeshConfig(node_id=nid, phy_mode="ofdm_only", topology="hub_relay", hub_id=1,
                                       hub_nodes=3, hub_cycle_s=5.0),
                         CssConfig(), OfdmConfig(),
                         log=lambda m, i=nid: (logs[i].append(m), print(f"[{i}] {m}", flush=True)))
            n.got = []
            n.on_data = lambda src, d, n=n: n.got.append((src, d))
            nodes[nid] = (n, s)
        for n, _ in nodes.values():
            n.start()
        end = time.monotonic() + 300
        while time.monotonic() < end:
            if all(any("trained with the hub" in m for m in logs[i]) for i in (2, 3)):
                break
            time.sleep(1.0)
        for i in (2, 3):
            assert any("trained with the hub" in m for m in logs[i]), i
        hub, c2, c3 = nodes[1][0], nodes[2][0], nodes[3][0]
        res = {}
        threads = [threading.Thread(target=lambda: res.__setitem__("up", c2.send(1, b"station 2 to the hub"))),
                   threading.Thread(target=lambda: res.__setitem__("all", hub.send(BROADCAST_ID, b"hub to all"))),
                   threading.Thread(target=lambda: res.__setitem__("relay", c3.send(2, b"3 to 2 via the hub")))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(400)
        assert res == {"up": True, "all": True, "relay": True}, res
        assert (2, b"station 2 to the hub") in hub.got
        assert (1, b"hub to all") in c2.got and (1, b"hub to all") in c3.got
        assert (3, b"3 to 2 via the hub") in c2.got
        assert (3, b"3 to 2 via the hub") not in hub.got          # the hub stores and forwards, it does not keep it
        for i in (2, 3):                                          # clients only ever address the hub
            tx = [m for m in logs[i] if m.startswith("hub: tx ")]
            assert tx and all(" to=1" in m for m in tx), (i, [m for m in tx if " to=1" not in m])
    finally:
        for n, s in nodes.values():
            n.stop(); s.close()
