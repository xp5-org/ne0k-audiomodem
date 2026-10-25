# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_modem_node.py — the modem-layer state machine (transport-agnostic).

A ModemNode is the "brain" of a device: it decides WHAT to transmit and reacts
to WHAT it hears, but it never touches audio or sockets.  All time is expressed
in the node's own continuous capture-sample counter ("local samples") — the
same counter css_decoder reports as a frame's `at_sample`.  Because every event
is timestamped in that one local counter, two-way ranging never has to compare
clocks between devices (see css_modem.rtt_to_distance_m).

The node is driven by three entry points:

  * a MAKE_* method (make_beacon / send_data / start_range) builds an outgoing
    packet and returns (raw_bytes, tag).  The transport transmits it, then must
    report back via on_tx(tag, local_sample) so the node can record exactly when
    its own chirp went out (the t0 of a ranging exchange).

  * on_rx(raw_bytes, local_sample) feeds a decoded frame in and returns a list
    of Actions the transport must carry out (send a reply after a delay, deliver
    an app payload, or surface a discovered/ranged/ack event).

This split is what makes the whole modem layer testable with no hardware:
css_modem_sim.py drives many nodes over a simulated acoustic channel, and the
same node code later binds to a real encode()/decode() transport unchanged.

Action kinds returned by on_rx (each a dict):
  {"kind":"send",      "delay":int, "tag":str, "build":callable(tx_local)->bytes}
        transmit build(tx_local) after `delay` local samples; the builder is
        given the ACTUAL transmit sample so a RANGE_REPLY can echo its true
        turnaround (delay may drift on real hardware).
  {"kind":"deliver",   "packet":dict}     an application payload addressed to us
  {"kind":"discovered","peer":int}         first BEACON heard from a peer
  {"kind":"ranged",    "peer":int, "distance_m":float, "rtt":int, "turnaround":int}
  {"kind":"ack",       "peer":int, "seq":int}
  {"kind":"ofdm_hs_req","peer":int, "seq":int}   peer asks us to run the OFDM
        handshake as responder (see ofdm_handshake.py); we auto-ACK first
"""
from __future__ import annotations

import css_modem as cm


class ModemNode:
    def __init__(self, node_id: int, *, fs: int,
                 turnaround_samples: int,
                 speed_of_sound: float = cm.SPEED_OF_SOUND_M_S):
        cm._check_nibble("node_id", node_id, cm.ID_MIN, cm.ID_MAX)
        self.id = int(node_id)
        self.fs = int(fs)
        self.turnaround = int(turnaround_samples)
        self.c = float(speed_of_sound)

        self.accept_ofdm_hs = True         # False (css_only): ignore incoming
                                           # OFDM_HS_REQ entirely -- no ACK, so
                                           # the requester's retries exhaust
                                           # and it falls back to CSS DATA
        self._seq = 0                      # our outgoing app-sequence (4-bit)
        self._seen: set[tuple[int, int]] = set()   # (src,seq) dedup for DATA
        self.peers: dict[int, dict] = {}   # peer_id -> {last_seen, distance_m}
        self._pending_range: dict[tuple[int, int], int] = {}  # (peer,seq)->t0

    # ── outgoing packet builders ─────────────────────────────────────────────
    def _take_seq(self) -> int:
        q = self._seq
        self._seq = cm.next_seq(self._seq)
        return q

    def make_beacon(self):
        """Presence/announce broadcast.  Returns (raw, tag)."""
        q = self._take_seq()
        raw = cm.pack_packet(self.id, cm.BROADCAST_ID, cm.TYPE_BEACON, q)
        return raw, f"beacon:{q}"

    def send_data(self, dst: int, payload: bytes):
        """Unicast (or dst=0xF broadcast) application data.  Returns (raw, tag)."""
        q = self._take_seq()
        raw = cm.pack_packet(self.id, dst, cm.TYPE_DATA, q, payload)
        return raw, f"data:{dst}:{q}"

    def make_ack(self, dst: int, ack_seq: int):
        """Unicast ACK acknowledging a received packet's seq.  Returns (raw, tag).
        ACKs echo the acknowledged seq rather than consuming a new one, so the
        originator can match the ACK to what it sent."""
        raw = cm.pack_packet(self.id, dst, cm.TYPE_ACK, ack_seq)
        return raw, f"ack:{dst}:{ack_seq}"

    def start_range(self, dst: int):
        """Begin a two-way ranging exchange with `dst`.  Returns (raw, tag).
        The tag encodes (peer,seq); on_tx records t0 under that key so the
        matching RANGE_REPLY can be correlated back."""
        q = self._take_seq()
        raw = cm.pack_packet(self.id, dst, cm.TYPE_RANGE_REQ, q)
        return raw, f"range:{dst}:{q}"

    def request_ofdm_hs(self, dst: int):
        """Ask `dst` to become the responder of an in-band OFDM handshake.
        Returns (raw, tag).  The peer auto-ACKs (so we know it armed) and
        surfaces an 'ofdm_hs_req' action to its driver, which then services
        the handshake proper (ofdm_handshake.HandshakeSM) outside this layer."""
        q = self._take_seq()
        raw = cm.pack_packet(self.id, dst, cm.TYPE_OFDM_HS_REQ, q)
        return raw, f"ofdmhs:{dst}:{q}"

    # ── transport must call this after actually transmitting ─────────────────
    def on_tx(self, tag: str, local_sample: int):
        """Report the local capture-sample at which a MAKE_* packet was emitted.
        Only ranging needs it (to fix t0); other tags are ignored."""
        if tag.startswith("range:"):
            _, dst, q = tag.split(":")
            self._pending_range[(int(dst), int(q))] = int(local_sample)

    # ── incoming frame ───────────────────────────────────────────────────────
    def on_rx(self, raw: bytes, local_sample: int) -> list[dict]:
        try:
            p = cm.unpack_packet(raw)
        except cm.ModemError:
            return []                      # runt/garbage frame — drop
        if p["src"] == self.id:
            return []                      # our own echo

        # Track liveness of whoever we just heard.
        peer = p["src"]
        rec = self.peers.setdefault(peer, {"last_seen": None, "distance_m": None,
                                           "known": False})
        rec["last_seen"] = int(local_sample)

        for_us = (p["dst"] == self.id) or p["is_broadcast"]
        actions: list[dict] = []
        typ = p["type"]

        # css_only: a declined OFDM_HS_REQ must produce NO response at all --
        # not even the neighbour-discovery ACK, which echoes this packet's seq
        # and would read to the requester as "request accepted" (it would then
        # enter a handshake nobody answers).  Total deafness makes the
        # requester's REQ retries exhaust and it falls back to CSS DATA.
        if typ == cm.TYPE_OFDM_HS_REQ and not self.accept_ofdm_hs:
            return []

        # Discovery = the first time we decode ANY packet from a peer (a decoded
        # frame proves the peer is in range).  It is NOT beacon-only: the whole
        # point of the neighbour-ACK is that a peer which hears OUR beacon ACKs so
        # WE discover IT — so hearing that ACK (or any packet) first must count.
        # Keying discovery on beacons alone strands a node that, after a deaf
        # spell, catches a peer's ACK before ever catching its beacon.  This is a
        # SEPARATE decision from the per-type handling below.
        if not rec["known"]:
            rec["known"] = True
            actions.append({"kind": "discovered", "peer": peer, "seq": p["seq"]})

        if typ == cm.TYPE_DATA and for_us:
            key = (peer, p["seq"])
            if key not in self._seen:          # de-dup repeated hearings
                self._seen.add(key)
                actions.append({"kind": "deliver", "packet": p})
                if p["dst"] == self.id:        # ACK unicast, not broadcast
                    aq = self._take_seq()
                    ack = cm.pack_packet(self.id, peer, cm.TYPE_ACK, p["seq"])
                    actions.append({"kind": "send", "delay": 0,
                                    "tag": f"ack:{peer}:{p['seq']}",
                                    "build": (lambda _ts, _a=ack: _a)})

        elif typ == cm.TYPE_ACK and p["dst"] == self.id:
            actions.append({"kind": "ack", "peer": peer, "seq": p["seq"]})

        elif typ == cm.TYPE_OFDM_HS_REQ and p["dst"] == self.id:
            # Unicast only (a broadcast handshake request would make every
            # listener respond at once).  ACK so the requester knows we armed,
            # then let the driver run the OFDM handshake as responder.
            key = (peer, p["seq"])
            if key not in self._seen:
                self._seen.add(key)
                ack = cm.pack_packet(self.id, peer, cm.TYPE_ACK, p["seq"])
                actions.append({"kind": "send", "delay": 0,
                                "tag": f"ack:{peer}:{p['seq']}",
                                "build": (lambda _ts, _a=ack: _a)})
                actions.append({"kind": "ofdm_hs_req", "peer": peer,
                                "seq": p["seq"]})

        elif typ == cm.TYPE_RANGE_REQ and for_us:
            # Respond after a fixed turnaround.  The builder is handed the ACTUAL
            # transmit sample so the echoed turnaround is the real one, not the
            # nominal — matters on hardware where the send can't be sample-exact.
            rx_at = int(local_sample)
            req_seq = p["seq"]

            def _build_reply(tx_local, _rx=rx_at, _peer=peer, _seq=req_seq):
                payload = cm.pack_range_reply_payload(int(tx_local) - _rx)
                return cm.pack_packet(self.id, _peer, cm.TYPE_RANGE_REPLY,
                                      _seq, payload)

            actions.append({"kind": "send", "delay": self.turnaround,
                            "tag": f"rangereply:{peer}:{req_seq}",
                            "build": _build_reply})

        elif typ == cm.TYPE_RANGE_REPLY and p["dst"] == self.id:
            key = (peer, p["seq"])
            t0 = self._pending_range.pop(key, None)
            if t0 is not None:
                try:
                    turn = cm.unpack_range_reply_payload(p["payload"])
                except cm.ModemError:
                    return actions
                rtt = int(local_sample) - t0
                dist = cm.rtt_to_distance_m(rtt, turn, self.fs, self.c)
                rec["distance_m"] = dist
                actions.append({"kind": "ranged", "peer": peer,
                                "distance_m": dist, "rtt": rtt,
                                "turnaround": turn})

        return actions
