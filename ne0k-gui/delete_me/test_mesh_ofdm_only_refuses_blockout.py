# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
import time

import pytest

pytest.importorskip("scipy")

from modemgui.audio import Registry
from modemgui.engine import Session
from modemgui.engine.mesh_node import MeshConfig, MeshNode, _Job
from modemgui.engine.modemcfg import CssConfig, OfdmConfig

# Retired 2026-11-07: blockout now runs over OFDM in ofdm_only (mesh-grp-ham-blockout), so it is no longer refused.
# Not collected (outside tests/); delete when confirmed unneeded.


def test_ofdm_only_refuses_blockout():
    """send_blockout/end_blockout broadcast over CSS and must be refused outright, not run."""
    reg = Registry()
    s = Session(reg, 48000, 1024)
    s.open("mix:mo_bo/1", "mix:mo_bo/1")
    n = MeshNode(s, MeshConfig(node_id=1, phy_mode="ofdm_only"), CssConfig(), OfdmConfig())
    try:
        n.start()
        with pytest.raises(RuntimeError, match="OFDM_ONLY VIOLATION"):
            n.send_blockout(2, b"x")
        with pytest.raises(RuntimeError, match="OFDM_ONLY VIOLATION"):
            n.end_blockout()
        assert len(n.ofdm_only_violations) == 2
    finally:
        n.stop(); s.close()
