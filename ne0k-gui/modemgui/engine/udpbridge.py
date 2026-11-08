# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""UDP data port over a continuous OFDM broadcast carrier: the data-stream sibling of the virtual serial port.

send     every datagram arriving on ``listen`` is queued straight onto the carrier (OfdmLink mode "bcast",
         addressed to the broadcast id 0xF).  The carrier stays up while datagrams keep coming; nothing is
         acknowledged.  The link logs what it aired, the filler it sent and any backlog it had to drop.
receive  every chunk the link delivers goes to ``dest`` as one UDP datagram, the moment it decodes.  The link logs
         the chunks it delivered and the ones it lost (seq gaps).

The link comes up from a saved BAT (``preset``), so there is no handshake on the air: a receiver started late locks
on the next chunk it hears.  An H.264 stream with repeated SPS/PPS (or MPEG-TS) starts playing at the next IDR."""
from __future__ import annotations

import dataclasses
import json
import socket
import threading

from .modemcfg import OfdmConfig
from .ofdm_link import OfdmLink

BROADCAST = 15


def load_bat(path: str) -> dict:
    with open(path) as f:
        saved = json.load(f)
    saved["tx_bat_b"], saved["rx_bat_b"] = bytes.fromhex(saved["tx_bat"]), bytes.fromhex(saved["rx_bat"])
    return saved


def link_config(base: OfdmConfig, saved: dict | None, cfg) -> OfdmConfig:
    """The engine config both ends must share (from the saved BAT when there is one), as a broadcast link."""
    fields = dataclasses.asdict(base)
    if saved:
        fields.update({k: v for k, v in saved["config"].items() if k not in ("node_id", "peer_id", "backend")})   # the backend is this node's own setting
    fields["mode"], fields["chunk_bytes"], fields["sender"] = "bcast", cfg.chunk_bytes, "initiator"
    if cfg.role == "send":
        fields["peer_id"], fields["role"] = BROADCAST, "initiator"
    else:
        fields["peer_id"], fields["role"] = cfg.sender_id, "responder"
    return OfdmConfig(**fields)


class UdpBridge:
    def __init__(self, session, base: OfdmConfig, cfg, log=lambda m: None) -> None:
        self.session, self.cfg, self.log = session, cfg, log
        self.saved = load_bat(cfg.bat_file) if cfg.bat_file else None
        self.link = OfdmLink(session, link_config(base, self.saved, cfg), log=log)
        if self.saved:
            tx, rx = self.saved["tx_bat_b"], self.saved["rx_bat_b"]
            self.link.preset = (tx, rx) if cfg.role == "send" else (rx, tx)
        self.tx_count = self.rx_count = self.datagrams = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._out: socket.socket | None = None

    def start(self) -> str:
        c = self.cfg
        self._stop.clear()
        if c.role == "send":
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind((c.listen_host, c.listen_port))
            self._sock.settimeout(0.25)
            what = f"udp://{c.listen_host}:{c.listen_port} -> OFDM broadcast"
        else:
            self._out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.link.on_data = self._rx
            what = f"OFDM broadcast -> udp://{c.dest_host}:{c.dest_port}"
        self.link.start()
        if not self.link.wait_up(30):
            self.stop()
            raise RuntimeError(f"broadcast link did not come up: {self.link.fail_reason}")
        if self._sock:
            self._thread = threading.Thread(target=self._send_loop, name="udp-send", daemon=True)
            self._thread.start()
        self.log(f"udp port open: {what}")
        return what

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        L = self.link
        if self.cfg.role == "send":
            self.log(f"udp port closed: {self.datagrams} datagrams {self.tx_count} B in, chunks aired={L.bcast_chunks} "
                     f"dropped={L.bcast_dropped} B")
        else:
            self.log(f"udp port closed: {self.rx_count} B out, chunks delivered={L.bcast_chunks} lost={L.lost_frames} "
                     f"bad={L.rx_errors}")
        L.stop()
        for s in (self._sock, self._out):
            if s:
                s.close()
        self._sock = self._out = None

    def _send_loop(self) -> None:
        while not self._stop.is_set():
            try:
                d, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            self.datagrams += 1
            self.tx_count += len(d)
            self.link.feed(d)

    def _rx(self, data: bytes) -> None:
        if self._out:
            self._out.sendto(data, (self.cfg.dest_host, self.cfg.dest_port))
        self.rx_count += len(data)
