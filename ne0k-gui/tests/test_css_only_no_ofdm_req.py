# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Plain CSS (phy_mode css_only) never asks for an OFDM link.

2026-11-17 laptops (CSS 0.15 s / 0.5 s symbols, auto_link left on from an OFDM run): the lower id's link keeper
opened an OFDM tx link and put a CSS OFDM_HS_REQ (13 s at 0.15 s, 43 s at 0.5 s) on the air before every send.
CSS -> OFDM promotion is a planned mode, not built: under css_only no REQ may go out, auto_link or not."""
import threading
import time

import pytest

pytest.importorskip("scipy")

from modemgui.audio import Registry
from modemgui.engine import Session
from modemgui.engine.mesh_node import MeshConfig, MeshNode
from modemgui.engine.modemcfg import CssConfig, OfdmConfig


def test_css_only_keeper_and_tx_link_send_no_ofdm_req():
    reg = Registry()
    s = Session(reg, 48000, 1024)
    s.open("loop:cssreq_rx", "loop:cssreq_tx")
    lines = []
    n = MeshNode(s, MeshConfig(node_id=1, phy_mode="css_only", auto_link=True, beacon=False),
                 CssConfig(), OfdmConfig(channel="line"), log=lines.append)
    n.ring_dst_fn = lambda: 2                      # lower id: the keeper would initiate
    try:
        n.start()
        time.sleep(5.0)                            # two keeper passes (every 2 s)
        res = []                                   # a REQ would block waiting for a CSS ACK nobody sends
        t = threading.Thread(target=lambda: res.append(n._open_tx_link(2)), daemon=True)
        t.start(); t.join(3.0)
        assert res == [None]
        assert n.reqs == 0 and not n.links
        assert not [m for m in lines if "OFDM_HS_REQ" in m], lines
    finally:
        n.stop()
        s.close()


def test_ack_wait_stretches_while_a_burst_is_being_received_or_decoded():
    """2026-11-17, 0.5 s symbols: the ACK wait ran out 1 s before the long-symbol decoder finished the 43 s ACK,
    and the 191 s DATA went out again.  The deadline now stretches while the CSS receiver is busy (capped)."""
    reg = Registry()
    s = Session(reg, 48000, 1024)
    s.open("loop:ackw_rx", "loop:ackw_tx")
    n = MeshNode(s, MeshConfig(node_id=1, phy_mode="css_only", beacon=False, ack_wait_s=0.5),
                 CssConfig(), OfdmConfig(channel="line"), log=lambda m: None)
    try:
        busy_until = [time.monotonic() + 1.5]
        n.css.channel_busy = lambda *a, **k: time.monotonic() < busy_until[0]
        threading.Timer(0.8, lambda: (n._acks.add((2, 7)), n._ack_evt.set())).start()
        assert n._wait_ack(2, 7)                   # ACK decoded at 0.8 s, past the 0.5 s wait: still counted

        t = time.monotonic()                       # quiet channel: gives up at the plain deadline
        busy_until[0] = 0
        assert not n._wait_ack(2, 8)
        assert time.monotonic() - t < 0.9

        t = time.monotonic()                       # busy for ever (someone else's long frame): capped at 2x
        busy_until[0] = 1e18
        assert not n._wait_ack(2, 9)
        assert 0.9 < time.monotonic() - t < 1.6
    finally:
        s.close()
