# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
import time

import pytest

pytest.importorskip("scipy")

from modemgui.audio import Registry
from modemgui.engine import Session
from modemgui.engine.mesh_node import MeshConfig, MeshNode, _Job
from modemgui.engine.modemcfg import CssConfig, OfdmConfig


@pytest.fixture(scope="module")
def mesh():
    """Three nodes on one shared medium (each hears the sum of the other two)."""
    reg, nodes = Registry(), {}
    for nid in (1, 2, 3):
        s = Session(reg, 48000, 1024)
        s.open(f"mix:m3/{nid}", f"mix:m3/{nid}")
        n = MeshNode(s, MeshConfig(node_id=nid, hold_s=60.0), CssConfig(), OfdmConfig(),
                     log=lambda m, i=nid: print(f"[{i}] {m}", flush=True))
        n.got = []
        n.on_data = lambda src, d, n=n: n.got.append((src, d))
        n.session_ = s
        nodes[nid] = n
    for n in nodes.values():
        n.start()
    yield nodes
    for n in nodes.values():
        n.stop(); n.session_.close()


def test_one_way_over_ofdm_with_a_bystander(mesh):
    n1, n2, n3 = mesh[1], mesh[2], mesh[3]
    payload = b"mesh one-way \x00\xff " + bytes(range(60))
    assert n1.send(2, payload)
    assert n2.got == [(1, payload)]
    assert n3.got == [] and n1.got == []
    assert n1.css_fallbacks == 0 and n1.reqs == 1
    lk = n1.links[(2, "tx")]
    assert lk.up and lk.cfg.mode == "arq" and n2.links[(1, "rx")].up
    assert time.monotonic() < n3._hold_until              # the bystander kept off the air
    assert (3, "tx") not in n1.links and not n3.links


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


def test_ofdm_only_never_beacons():
    """phy_mode=ofdm_only must refuse to key the CSS radio even when mesh.beacon=True."""
    reg, made = Registry(), []
    try:
        for nid, mode in ((1, "ofdm_only"), (2, "css_then_ofdm")):
            s = Session(reg, 48000, 1024)
            s.open(f"mix:mo_bcn/{nid}", f"mix:mo_bcn/{nid}")
            n = MeshNode(s, MeshConfig(node_id=nid, phy_mode=mode, beacon=True,
                                        first_beacon_s=0.01, beacon_min_s=0.01, beacon_max_s=0.02,
                                        cw_s=0.05, id_slot_s=0.01),
                         CssConfig(), OfdmConfig())
            n.start(); made.append((n, s))
        n1, n2 = made[0][0], made[1][0]
        # node_id-proportional first-beacon stagger (Rechirp.first_stagger_s=1.0) means node 2's
        # first beacon isn't due for ~2s, and a BEACON's actual CSS airtime is ~8s -- give it room.
        time.sleep(20.0)
        assert n1.beacons_tx == 0
        assert n1.ofdm_only_violations == []     # the enqueue-time gate kept it from ever queuing
        assert n2.beacons_tx > 0                  # control: beaconing works normally for everyone else
    finally:
        for n, s in made:
            n.stop(); s.close()


def test_ofdm_only_ignores_css_handshake_and_data():
    """An ofdm_only node takes no part in CSS: it runs no CSS listener at all, so a peer's CSS REQ and DATA are
    never heard, never ACKed (acking IS a CSS transmission) and never land."""
    reg, made = Registry(), []
    try:
        for nid, mode in ((1, "css_then_ofdm"), (2, "ofdm_only")):
            s = Session(reg, 48000, 1024)
            s.open(f"mix:mo_ign/{nid}", f"mix:mo_ign/{nid}")
            n = MeshNode(s, MeshConfig(node_id=nid, phy_mode=mode, req_retries=0, data_retries=0),
                         CssConfig(), OfdmConfig())
            n.got = []
            n.on_data = lambda src, d, n=n: n.got.append((src, d))
            n.start(); made.append((n, s))
        n1, n2 = made[0][0], made[1][0]
        # n1 (css_then_ofdm) tries to reach n2 (ofdm_only): n2 must ignore the REQ (no OFDM link),
        # then ignore the CSS DATA fallback too -- the transfer must simply fail, not land quietly.
        assert n2.css._thread is None                 # no CSS listener under ofdm_only
        assert n1.send(2, b"should never land") is False
        assert n2.got == []
        assert n2.ofdm_only_violations == []          # nothing CSS ever reached it to refuse
    finally:
        for n, s in made:
            n.stop(); s.close()


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


def test_ofdm_only_mac_backstop():
    """Even a job that slipped past every upstream gate must never actually key the radio."""
    reg = Registry()
    s = Session(reg, 48000, 1024)
    s.open("mix:mo_bs/1", "mix:mo_bs/1")
    n = MeshNode(s, MeshConfig(node_id=1, phy_mode="ofdm_only", cw_s=0.05, id_slot_s=0.01),
                 CssConfig(), OfdmConfig())
    try:
        n.start()
        called = []
        job = _Job(lambda *a: called.append(a) or 0.0, (1, 2, 3))
        with n._lock:
            n.csma.enqueue(job, "bypass:", 0.0)
        assert job.done.wait(3.0)
        assert called == []                       # _mac's backstop refused to call job.fn
        assert job.err is not None
        assert any("OFDM_ONLY VIOLATION" in v for v in n.ofdm_only_violations)
    finally:
        n.stop(); s.close()


def test_serial_bridge_over_mesh(tmp_path):
    from modemgui.engine.serialbridge import HandshakePort, SerialBridge
    reg, made = Registry(), []
    try:
        for nid, dst in ((1, 2), (2, 1)):
            s = Session(reg, 48000, 1024)
            s.open(f"mix:mb/{nid}", f"mix:mb/{nid}")
            node = MeshNode(s, MeshConfig(node_id=nid), CssConfig(), OfdmConfig())
            br = SerialBridge(s, node, str(tmp_path / str(nid)), buf_size=64, dst=dst)
            br.start(); made.append((br, s))
        pa = HandshakePort(str(tmp_path / "1"))
        pb = HandshakePort(str(tmp_path / "2"))
        assert pa.wait_cts(True, 5)
        assert pa.send(b"serial over mesh\r\n", timeout=300)
        assert pb.read(100, 20) == b"serial over mesh\r\n"
    finally:
        for br, s in made:
            br.stop(); s.close()


def test_announce_mask_older_than_our_announce_does_not_flag_us():
    """Node 1 announces first knowing nobody (mask 0x0002).  Node 2, announcing after it, must not read that as
    'node 1 has not heard me' for the rest of the session; a mask from a LATER announce that lacks us still does."""
    from modemgui.engine.modemcfg import OfdmConfig
    from modemgui.engine.ofdm_announce import AnnounceRound
    an = AnnounceRound(None, OfdmConfig(), 2, log=lambda m: None)
    an.masks[1], an.heard_end[1] = 0b0010, 100.0
    assert an._flagged()                       # we never announced: an announce lacking us does count
    an._own_ann_start = 101.0
    assert not an._flagged()                   # node 1's announce ended before ours began
    an.masks[3], an.heard_end[3] = 0b1010, 110.0
    assert an._flagged()                       # node 3 announced after us and did not hear us
    an.masks[3] = 0b1110
    assert not an._flagged()
