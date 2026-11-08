# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
import time

import pytest

pytest.importorskip("scipy")

from modemgui.audio import Registry
from modemgui.engine import Session
from modemgui.engine.mesh_node import MeshConfig, MeshNode, _Job
from modemgui.engine.modemcfg import CssConfig, OfdmConfig

# Retired 2026-11-07: css_then_ofdm/css_only CSS fallback is gone from the ofdm_only mesh.
# Not collected (outside tests/); delete when confirmed unneeded.


def test_unanswered_req_falls_back_to_css():
    reg, made = Registry(), []
    try:
        for nid, mode in ((1, "css_then_ofdm"), (2, "css_only")):
            s = Session(reg, 48000, 1024)
            s.open(f"mix:m2/{nid}", f"mix:m2/{nid}")
            n = MeshNode(s, MeshConfig(node_id=nid, phy_mode=mode, req_retries=0), CssConfig(),
                         OfdmConfig())
            n.got = []
            n.on_data = lambda src, d, n=n: n.got.append((src, d))
            n.start(); made.append((n, s))
        n1, n2 = made[0][0], made[1][0]
        assert n1.send(2, b"css only")
        assert n2.got == [(1, b"css only")] and n1.css_fallbacks == 1 and not n1.links
        assert n1.want_ofdm[2] is False                  # given up: no timed re-upgrade
    finally:
        for n, s in made:
            n.stop(); s.close()
