# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""GUI p2p topology, ofdm_only, half-duplex arq links, auto_link on: ONE link per pair carries data both ways.

2026-11-17 laptops (Mac <-> m6600, 1 ft): the lower id's keeper trained its tx link and its data got through, but
the other end, to answer, started a handshake of its own that nobody listened for (timeout_i_wait_probe, 0/3).
Arming a second responder made it worse: two links to one peer take each other's handshake messages."""
import threading
import time

import pytest

pytest.importorskip("scipy")

from modemgui.audio import Registry
from modemgui.engine import Session
from modemgui.engine.mesh_node import MeshConfig, MeshNode
from modemgui.engine.modemcfg import CssConfig, OfdmConfig


@pytest.fixture
def pair():
    reg, nodes = Registry(), {}
    try:
        for nid, peer, rx, tx in ((1, 2, "loop:arqb", "loop:arqa"), (2, 1, "loop:arqa", "loop:arqb")):
            s = Session(reg, 48000, 1024)
            s.open(rx, tx)
            s.keyup_tone = False                                   # a line: no PL lead
            n = MeshNode(s, MeshConfig(node_id=nid, phy_mode="ofdm_only", auto_link=True, beacon=False),
                         CssConfig(), OfdmConfig(channel="line"),
                         log=lambda m, i=nid: print(f"[{i}] {m}", flush=True))
            n.ring_dst_fn = lambda p=peer: p
            n.got = []
            n.on_data = lambda src, d, n=n: n.got.append((src, d))
            n.session_ = s
            nodes[nid] = n
        for n in nodes.values():
            n.start()
        yield nodes
    finally:
        for n in nodes.values():
            n.stop()
            n.session_.close()


def _wait(pred, s):
    end = time.monotonic() + s
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.25)
    return False


def test_one_link_carries_data_both_ways(pair):
    n1, n2 = pair[1], pair[2]
    # the responder answers first, before the keeper's link is even up: it waits for it, it does not handshake
    pb = b"2->1 reply " + bytes(range(40))
    assert n2.send(1, pb)
    assert n1.got == [(2, pb)]
    pa = b"1->2 data \x00\xff"
    assert n1.send(2, pa)
    assert n2.got == [(1, pa)]
    assert list(n1.links) == [(2, "tx")] and list(n2.links) == [(1, "rx")]     # one link, opened by the lower id
    assert n1.links[(2, "tx")].cfg.sender == "both"


def test_both_send_at_once_on_the_one_link(pair):
    n1, n2 = pair[1], pair[2]
    assert _wait(lambda: (2, "tx") in n1.links and n1.links[(2, "tx")].up
                 and (1, "rx") in n2.links and n2.links[(1, "rx")].up, 120)
    res = {}
    pa, pb = b"A->B at once", b"B->A at once"
    ta = threading.Thread(target=lambda: res.setdefault(1, n1.send(2, pa)))
    tb = threading.Thread(target=lambda: res.setdefault(2, n2.send(1, pb)))
    ta.start(); tb.start(); ta.join(300); tb.join(300)
    assert res == {1: True, 2: True}
    assert n2.got == [(1, pa)] and n1.got == [(2, pb)]
    assert list(n1.links) == [(2, "tx")] and list(n2.links) == [(1, "rx")]
