# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Programmable BAT over the control port (engine/batstore.py): a receiver's probe measurement comes out as a
saved-BAT JSON line (BAT GET), a one-way sender takes it in (BAT PUT -> udp.bat_file), and a UDP broadcast
stream comes up from it on both ends with no handshake on the air -- the one-way RF case (FM transmitter into an
RTL-SDR) where the table cannot go back over the air."""
import dataclasses
import json
import socket
import time

import numpy as np
import pytest

from modemgui.audio import Registry
from modemgui.engine import Session, batstore
from modemgui.engine.control import ControlServer, EngineHost
from modemgui.engine.modemcfg import OfdmConfig, UdpConfig
from modemgui.engine.ofdm_link import _imports, apply_phy
from modemgui.engine.udpbridge import UdpBridge, link_config, load_bat


def _measure(cfg: OfdmConfig) -> tuple[bytes, dict]:
    """The BAT a receiver measures from a clean probe (what OfdmLink._attempt_rx records)."""
    H, _ = _imports()
    apply_phy(cfg)
    band = (cfg.freq_lo, cfg.freq_hi)
    audio = H.make_probe_audio(cfg.fs, band)
    audio, _ = H.scale_hs_tx(audio, 0.5, cfg.fs, is_probe=True, band=band)
    x = np.concatenate([np.zeros(cfg.fs), audio, np.zeros(2 * cfg.fs)]).astype(np.float32)
    x += np.random.default_rng(1).normal(0, 1e-3, len(x)).astype(np.float32)
    got = H.try_decode_probe(x, cfg.fs, rates=(cfg.fs,), min_syms=12, band=band)
    assert got, "clean probe not decoded"
    return bytes(got[1]), got[2]


@pytest.fixture(autouse=True)
def _fresh():
    batstore.clear()
    yield
    batstore.clear()


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]; s.close()
    return p


def test_saved_bat_round_trip_and_checks(tmp_path):
    cfg = OfdmConfig(n_fft=1024, freq_hi=6000.0, node_id=2, peer_id=1, amplitude=0.3)
    bat, info = _measure(cfg)
    e = batstore.record(1, 2, bat, cfg, info)
    saved = batstore.to_saved(e)
    assert saved["tx_bat"] == bat.hex() and saved["config"]["n_fft"] == 1024
    for k in ("node_id", "peer_id", "amplitude", "backend"):        # the sender's own settings stay its own
        assert k not in saved["config"]
    d = batstore.parse_saved(json.dumps(saved))
    path = batstore.write(d, str(tmp_path), "fm link")
    assert path.endswith("fm_link.json")
    lb = load_bat(path)
    assert lb["tx_bat_b"] == bat
    base = OfdmConfig(amplitude=0.9, n_fft=512)
    send = link_config(base, lb, UdpConfig(role="send"))
    assert send.n_fft == 1024 and send.amplitude == 0.9 and send.mode == "bcast"

    for bad, why in (("{", "not JSON"), ('{"config":{},"tx_bat":"zz","rx_bat":"00"}', "tx_bat"),
                     ('{"config":{"warp":1},"tx_bat":"00","rx_bat":"00"}', "unknown config fields: warp")):
        with pytest.raises(ValueError, match=why):
            batstore.parse_saved(bad)


class _C:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.f = self.s.makefile("r", encoding="utf-8", newline="\n")
        self.f.readline()

    def cmd(self, text):
        self.s.sendall((text + "\n").encode())
        out = []
        while True:
            ln = self.f.readline().rstrip("\n")
            if ln == "OK" or ln.startswith(("OK ", "ERR ")):
                return out, ln
            out.append(ln)


def test_engine_host_bat_get_list_and_refused_put():
    srv = ControlServer(EngineHost(None, {}), 0); srv.start()
    try:
        c = _C(srv.port)
        assert c.cmd("BAT GET")[1].startswith("ERR no probe measured")
        assert c.cmd("BAT LIST") == ([], "OK")
        cfg = OfdmConfig(n_fft=512, node_id=2, peer_id=1)
        batstore.record(1, 2, bytes([2] * 40), cfg, {"bat_summary": "active=40", "snr_median_db": 30.0})
        lines, ok = c.cmd("BAT LIST")
        assert ok == "OK" and lines[0].startswith("bat.measured peer=1 by=2 ") and "n_fft=512" in lines[0]
        assert c.cmd("BAT GET 7")[1] == "ERR no probe measured from node 7 yet: the peer must send a probe (a handshake) first"
        lines, ok = c.cmd("BAT GET 1")
        assert ok == "OK" and lines[0].startswith("bat.json=")
        d = batstore.parse_saved(lines[0].split("=", 1)[1])
        assert d["tx_bat"] == "02" * 40 and d["measured_by"] == 2
        assert c.cmd("BAT PUT " + lines[0].split("=", 1)[1])[1].startswith("ERR this host cannot take a BAT")
        assert c.cmd("BAT FROB")[1].startswith("ERR BAT LIST")
    finally:
        srv.stop()


def test_one_way_stream_comes_up_from_a_measured_bat(tmp_path):
    """Receiver measures -> BAT GET -> sender's bat_file -> both UDP ports up with no air handshake; data crosses."""
    cfg = OfdmConfig(n_fft=1024, freq_hi=6000.0, max_bits_per_bin=2)
    bat, info = _measure(cfg)
    batstore.record(1, 2, bat, cfg, info)
    path = batstore.write(batstore.parse_saved(json.dumps(batstore.to_saved(batstore.latest(1)))), str(tmp_path))

    reg = Registry()
    tx_s, rx_s = Session(reg, 48000, 1024), Session(reg, 48000, 1024)
    tx_s.open("loop:batrx", "loop:battx"); rx_s.open("loop:battx", "loop:batrx")
    lport, dport = _free_port(), _free_port()
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); out.bind(("127.0.0.1", dport)); out.settimeout(60)
    base = dataclasses.replace(OfdmConfig(), backend="python")
    snd = UdpBridge(tx_s, dataclasses.replace(base, node_id=1),
                    UdpConfig(role="send", listen_port=lport, chunk_bytes=64, bat_file=path, sender_id=1))
    rcv = UdpBridge(rx_s, dataclasses.replace(base, node_id=2),
                    UdpConfig(role="receive", dest_port=dport, chunk_bytes=64, bat_file=path, sender_id=1))
    try:
        rcv.start(); snd.start()
        assert snd.link.cfg.n_fft == 1024 and rcv.link.cfg.n_fft == 1024
        msg = b"one-way stream from a BAT measured at the receiver"
        inp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        inp.sendto(msg, ("127.0.0.1", lport))
        got = b""
        end = time.monotonic() + 60
        while msg not in got and time.monotonic() < end:
            try:
                got += out.recv(4096)
            except socket.timeout:
                break
        assert msg in got, got
    finally:
        snd.stop(); rcv.stop(); tx_s.close(); rx_s.close(); out.close()
