# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Byte pipe over one SIP call: OFDM stream bursts, both directions, as G.711 audio.

The SIP side (mock.py) owns the RTP socket; it hands PCMU packets to ``feed`` and gets packets to
send through ``attach``.  Underneath is the ordinary OFDM link (BAT handshake, then ``stream`` bursts
with sender="both"), so the pipe is what a serial line is to the fax engine: ``send`` blocks for the
airtime of one burst, ``on_data`` delivers what the far end sent."""
from __future__ import annotations

import threading
import time

from ..audio import Registry
from ..audio.rtp_backend import RtpConnector
from ..engine import Session
from ..engine.modemcfg import OfdmConfig
from ..engine.ofdm_link import OfdmLink

# What the fax work found fastest on a clean G.711 call (see serial_perf.py): 64-QAM allowed, short
# quiet between bursts, 128-byte chunks.
DEFAULTS = dict(mode="stream", sender="both", chunk_bytes=128, gap_s=0.2, hang_s=0.2,
                max_bits_per_bin=6, freq_lo=400.0, freq_hi=3300.0)


class Ne0kPipe:
    def __init__(self, name: str, caller: bool, log=print, **over) -> None:
        self.name, self.log = name, log
        self.conn = RtpConnector()
        self.port = self.conn.port(name)
        reg = Registry(plugins=False)
        reg.add(self.conn)
        self.session = Session(reg, 48000, 1024)
        cfg = dict(DEFAULTS, node_id=1 if caller else 2, peer_id=2 if caller else 1,
                   role="initiator" if caller else "responder")
        cfg.update({k: v for k, v in over.items() if v is not None})
        self.cfg = OfdmConfig(**cfg)
        self.link = OfdmLink(self.session, self.cfg, log=lambda m: self.log(f"ofdm: {m}"))
        self.on_data = None                 # f(bytes), called from a link thread
        self.link.on_data = lambda d: self.on_data and self.on_data(d)
        self.link.on_stream = lambda ok, n: None if ok else self.log(f"ofdm: a burst was damaged ({n} bytes delivered)")
        self.link.on_state = lambda st: self.log(f"ofdm: state {st} {self.link.fail_reason}".rstrip())
        self.tx_bytes = self.rx_bytes = 0

    # -- SIP side -------------------------------------------------------------

    def attach(self, send, remote) -> None:
        self.port.attach(send)
        self.port.set_remote(remote)

    def feed(self, packet: bytes) -> None:
        self.port.feed(packet)

    # -- lifecycle --------------------------------------------------------------

    def start(self) -> None:
        self.session.open(f"rtp:{self.name}", f"rtp:{self.name}")
        self.link.start()

    def wait_up(self, timeout: float) -> bool:
        return self.link.wait_up(timeout)

    @property
    def up(self) -> bool:
        return self.link.up

    def send(self, data: bytes) -> bool:
        """Blocks until the burst has aired."""
        ok = self.link.send(data)
        if ok:
            self.tx_bytes += len(data)
        return ok

    def errors(self) -> list[str]:
        """Messages the link/session raised on their threads (they would otherwise vanish)."""
        return [getattr(e, "message", "") for e in self.session.drain() if type(e).__name__ == "Status"
                and getattr(e, "state", "") == "error"]

    def stop(self) -> None:
        try:
            self.link.stop()
        finally:
            self.session.close()
            self.conn.drop(self.name)
