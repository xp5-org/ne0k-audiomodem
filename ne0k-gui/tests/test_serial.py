# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
import dataclasses
import time

import pytest

from modemgui.audio import Registry
from modemgui.engine import Session
from modemgui.engine.modemcfg import PRESETS
from modemgui.engine.serialbridge import (HandshakePort, SerialBridge, frame,
                                          unframe)


def test_framing_keeps_trailing_nuls():
    assert unframe(frame(b"a\x00\x00")) == b"a\x00\x00"
    assert unframe(b"") == b""
    with pytest.raises(ValueError):
        frame(b"x" * 251)


@pytest.fixture
def pair(tmp_path):
    reg = Registry()
    made = []

    def mk(name, node, peer, rx, tx, **kw):
        s = Session(reg, 48000, 1024); s.open(rx, tx)
        cfg = dataclasses.replace(PRESETS["Fast (POC)"], node_id=node, peer_id=peer)
        b = SerialBridge(s, cfg, str(tmp_path / name), **kw); b.start()
        made.append((b, s)); return b
    yield mk, tmp_path
    for b, s in made:
        b.stop(); s.close()


def test_handshake_rules_and_delivery(pair):
    mk, d = pair
    a = mk("A", 1, 2, "loop:ba", "loop:ab", buf_size=8, ack=False)
    b = mk("B", 2, 1, "loop:ab", "loop:ba", buf_size=8, ack=False)
    pa, pb = HandshakePort(str(d / "A")), HandshakePort(str(d / "B"))
    assert pa.wait_cts(True, 2)

    pa.set_rts(True); time.sleep(0.3); pa.set_rts(False)        # RTS with nothing loaded
    assert pa.errors == ["empty"]

    pa.write(b"x" * 20)                                          # more than N: only N accepted
    assert pa.wait_buffered(8, 3)
    time.sleep(0.3)
    assert pa.buf == (8, 8)

    pa.set_rts(True)                                             # send 8 bytes
    assert pa.wait_cts(False, 3)                                 # TX-busy shown
    pa.set_rts(False); pa.set_rts(True)                          # RTS edge while busy
    assert pa.wait_cts(True, 60)                                 # busy cleared
    assert "busy" in pa.errors
    assert pb.read(100, 45) == b"x" * 8                          # peer got exactly the buffer
    assert a.tx_count == 8 and b.rx_count == 8


def _ack_pair(mk, **kw):
    a = mk("A", 1, 2, "loop:ba", "loop:ab", buf_size=4, retries=2, **kw)
    b = mk("B", 2, 1, "loop:ab", "loop:ba", buf_size=4)
    return a, b


def test_ack_completes_and_nothing_retried(pair):
    mk, d = pair
    a, b = _ack_pair(mk)
    pa, pb = HandshakePort(str(d / "A")), HandshakePort(str(d / "B"))
    assert pa.send(b"hi\x00!", timeout=120)
    assert pb.read(10, 5) == b"hi\x00!"
    assert a.retry_count == 0 and a.fail_count == 0


def test_lost_data_is_retransmitted_once_delivered_once(pair):
    mk, d = pair
    a, b = _ack_pair(mk)
    drops = []
    b.rx_filter = lambda ev: (drops.append(ev.seq) or False) if not drops else True
    pa, pb = HandshakePort(str(d / "A")), HandshakePort(str(d / "B"))
    assert pa.send(b"abcd", timeout=180)
    assert pb.read(10, 5) == b"abcd" and pb.read(10, 1) == b""
    assert a.retry_count == 1 and len(drops) == 1


def test_lost_ack_retransmits_but_receiver_dedups(pair):
    mk, d = pair
    a, b = _ack_pair(mk)
    lost = []
    a.rx_filter = lambda ev: not (ev.type_name == "ACK" and not lost and not lost.append(1))
    pa, pb = HandshakePort(str(d / "A")), HandshakePort(str(d / "B"))
    assert pa.send(b"wxyz", timeout=180)
    assert pb.read(10, 5) == b"wxyz" and pb.read(10, 1) == b""     # not delivered twice
    assert a.retry_count == 1


def test_gives_up_with_txfail(pair):
    mk, d = pair
    a, b = _ack_pair(mk)
    b.rx_filter = lambda ev: False
    a.ack_timeout = 25
    pa = HandshakePort(str(d / "A"))
    assert not pa.send(b"zz", timeout=240)
    assert a.fail_count == 1 and a.retry_count == 2 and pa.retries == 2
    assert pa.wait_cts(True, 5)


def test_full_buffer_sent_at_once_still_counts_as_loaded():
    """A buffer filled in one write goes out immediately: "BUF 4/4" and "BUF 0/4" can land in one recv, so the
    host never sees the level at 4.  wait_buffered must still succeed (it used to time out -> a false SENDFAIL)."""
    import socket
    import threading
    from modemgui.engine.serialbridge import HandshakePort
    hp = HandshakePort.__new__(HandshakePort)
    hp.cts, hp.cd, hp.buf, hp._buf_hi = True, None, (0, 4), 0
    hp.rx_events, hp.errors, hp._cv = None, [], threading.Condition()
    hp._txdone = hp._txfail = hp.retries = 0
    hp.sock, far = socket.socketpair()
    threading.Thread(target=hp._reader, daemon=True).start()
    with hp._cv:
        hp._buf_hi = hp.buf[0]                     # what write() does before handing the bytes over
    far.sendall(b"BUF 4/4\nBUF 0/4\nCTS 0\n")
    assert hp.wait_buffered(4, 2.0)
    assert hp.buf == (0, 4) and hp.cts is False
    far.close()
