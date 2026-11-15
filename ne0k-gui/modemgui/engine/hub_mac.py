# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Hub & spoke MAC for an OFDM-only network: one leader (the hub) runs the channel, clients only ever answer it.

Two topologies:

* ``hub``        -- station <-> hub only.  A client's page goes to the hub; the hub's page goes to one client or to
  every client at once.
* ``hub_relay``  -- the same, plus station -> station THROUGH the hub: a client's page names another station, the hub stores it and forwards it in its own next turn, then tells the source how it went.

There is no failover and no handover: if the hub drops, the network is down.  A client is configured with the hub's
id and the number of participants (``hub_nodes``, hub included), nothing else; it is never sent a list of station
ids.  Whom a client's host may address in ``hub_relay`` is the host's business, not the modem's.

Every client transmission is an answer to the hub: a JOIN to its beacon, a DONE / data frames in the turn its POLL
granted, an ACK to a frame it sent, a broadcast ACK.  A client never keys up on its own initiative and never talks to
another client.  The schedule, all on the one shared half-duplex channel:

1. **Beacon / join.**  While fewer than ``hub_nodes - 1`` clients are registered the hub sends a BEACON
   each cycle.  A client that is not registered answers with a JOIN in a slot ranked by its own id among the
   ``hub_nodes - 1`` client slots.  Being polled or trained by the hub is what registers a client: there is no separate welcome.
2. **Training.**  For each registered client without links: TRAIN, then the ordinary pairwise V.22
   handshake, hub as initiator (the hub -> client arq link).  The handshake measured both directions, so both ends
   bring the reverse link (client -> hub) up on those tables with no second handshake.  Then the hub's BTAB:
   the broadcast table, per carrier the least bits any client's hub -> client table allows, so one frame decodes at
   every client.
3. **Hub turn.**  The hub sends what it holds: its host's pages (to a client, or broadcast) and, under
   ``hub_relay``, the pages it stored for forwarding.  A broadcast frame goes out once on the broadcast table; every
   client answers it with a BACK in its own-id slot; the hub resends until every registered client has ACKed or the
   retries run out.  After a relayed page the hub sends DLVR to its
   source (delivered or not).
4. **Polls.**  For each trained client: POLL, the client sends its page on its arq link and then DONE.  A client that misses ``MISS_LIMIT`` polls in a row is dropped
   (links gone, join reopens).

The cycle is padded to at least ``hub_cycle_s`` so an idle network is not wall-to-wall polls.

Pages
carry a 3-byte header inside the frame payload, every frame:

    [route u8 = origin << 4 | destination][page id u8][frag u8: bit7 last, bits 0-6 index] + data

so a relayed page keeps its origin and a broadcast says "0xF".  ``on_data(origin, page)`` delivers a whole page.

MSG_HUB bodies (ofdm_handshake.MSG_HUB): [kind u8][from u8][to u8] + args, see the K_* constants.
"""
from __future__ import annotations

import collections
import random
import struct
import threading
import time
from typing import Callable

import numpy as np

from .ofdm_link import BROADCAST_ID, H_dem, _imports, _quiet, keyup, pack_frame

K_BEACON, K_JOIN, K_POLL, K_DONE, K_BACK, K_BTAB, K_TRAIN, K_DLVR = range(8)
K_NAMES = {K_BEACON: "BEACON", K_JOIN: "JOIN", K_POLL: "POLL", K_DONE: "DONE", K_BACK: "BACK", K_BTAB: "BTAB",
           K_TRAIN: "TRAIN", K_DLVR: "DLVR"}
F_JOIN_OPEN, F_RELAY = 1, 2           # BEACON flags
D_NEED_TRAIN, D_MORE = 1, 2           # DONE flags: no links here (retrain me) / a page is still waiting
PAGE_HDR = 3
LAST = 0x80
MISS_LIMIT = 3                        # polls a client may miss in a row before the hub drops it
BCAST_RETRIES = 2                     # broadcast frame resends before the unicast fallback
MIN_BCAST_BITS = 8                    # a broadcast table with fewer bits than this is useless: unicast instead
GUARD_S = 1.0
DECODE_S = 2.5                        # a station's decode after a burst ends, before it can answer


def pack_hub(kind: int, frm: int, to: int, args: bytes = b"") -> bytes:
    return bytes([kind, frm & 0xFF, to & 0xFF]) + args


def unpack_hub(body: bytes):
    """(kind, from, to, args) or None."""
    if len(body) < 3 or body[0] not in K_NAMES:
        return None
    return body[0], body[1], body[2], body[3:]


def split_page(origin: int, dst: int, pid: int, data: bytes, room: int) -> list[bytes]:
    """The frame payloads of one page: header + up to ``room - PAGE_HDR`` data bytes each."""
    step = max(1, room - PAGE_HDR)
    pieces = [data[i:i + step] for i in range(0, len(data), step)] or [b""]
    route = ((origin & 0xF) << 4) | (dst & 0xF)
    return [bytes([route, pid & 0xFF, (LAST if k == len(pieces) - 1 else 0) | (k & 0x7F)]) + p
            for k, p in enumerate(pieces)]


class Reassembly:
    """Pages out of frame payloads, per sending link.  Frames of one page arrive in order, so a page is its fragments 0..last; anything out of order restarts it."""

    def __init__(self) -> None:
        self._open: dict = {}

    def add(self, link_src: int, payload: bytes):
        """(origin, dst, pid, page) once a page is whole, else None."""
        if len(payload) < PAGE_HDR:
            return None
        route, pid, frag = payload[0], payload[1], payload[2]
        key = (link_src, route, pid)
        idx = frag & 0x7F
        buf = self._open.get(key)
        if idx == 0:
            buf = self._open[key] = []
        if buf is None or len(buf) != idx:
            self._open.pop(key, None)
            return None
        buf.append(payload[PAGE_HDR:])
        if not frag & LAST:
            return None
        del self._open[key]
        return route >> 4, route & 0xF, pid, b"".join(buf)


class _Page:
    def __init__(self, origin: int, dst: int, pid: int, data: bytes, local: bool) -> None:
        self.origin, self.dst, self.pid, self.data = origin, dst, pid, data
        self.local = local                       # from this node's own host
        self.done = threading.Event()
        self.ok = False
        self.t = time.monotonic()


class _BcastRx:
    """What the shared receiver needs to try the broadcast table on a burst: a stand-in link to the hub."""

    shared_rx = True
    up = True
    schedule_rx = None
    role = "responder"

    def __init__(self, cfg, hub: int, bat, band) -> None:
        self.cfg = cfg                           # OfdmConfig with peer_id = the hub
        self._bat, self._band = bat, band
        self._dem = None

    def rx_demod(self):
        if self._dem is None:
            H, _ = _imports()
            self._dem = H_dem(H, self.cfg, self._bat, self._band)
        return self._dem, self._bat, self._band

    def wants_ack(self) -> bool:
        return False


class HubMac:
    """The hub & spoke schedule, both roles.  MeshNode installs it where the mesh's AnnounceRound would go and the node's MeshReceiver feeds
    it messages (``rx_msg``) and broadcast frames (``on_bcast``)."""

    def __init__(self, node, *, hub_id: int, n_nodes: int, relay: bool, turn_s: float = 60.0,
                 cycle_s: float = 15.0, page_timeout_s: float = 300.0, bat_turn_s: float = 90.0,
                 log: Callable[[str], None] = lambda m: None) -> None:
        if not 0 <= hub_id < BROADCAST_ID:
            raise ValueError("hub_id must be 0..14")
        if not 2 <= n_nodes <= 15:
            raise ValueError("hub_nodes must be 2..15 (hub included)")
        self.node, self.log = node, log
        self.me = node.me
        self.hub_id, self.n_nodes, self.relay = hub_id, n_nodes, relay
        self.is_hub = self.me == hub_id
        self.turn_s, self.cycle_s, self.page_timeout_s, self.bat_turn_s = turn_s, cycle_s, page_timeout_s, bat_turn_s
        cfg = node.ofdm_cfg
        self.cfg, self.fs, self.band = cfg, cfg.fs, (cfg.freq_lo, cfg.freq_hi)
        self.phase = "listen"                    # listen | join | bat | data | idle  
        self.data_since = None                   # links are never idle-reaped: the hub's polls are the liveness check
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._txq: "collections.deque" = collections.deque()
        self._tx_evt = threading.Event()
        self._reasm = Reassembly()
        self._pid = random.randrange(256)
        self._queue: "collections.deque[_Page]" = collections.deque()     # pages waiting for this node's turn
        self._waiting: dict[int, _Page] = {}     # client: relayed page id -> page, until the hub's DLVR
        self._pending = False
        self._last_busy = 0.0
        self.msg_air_s = 0.0
        self.counts = collections.Counter()
        # hub state
        self.clients: dict[int, dict] = {}       # id -> {"trained": bool, "missed": int, "fails": int}
        self._joins: set[int] = set()
        self._done_evt = threading.Event()
        self._done_from: tuple[int, int] | None = None   # (client, flags) of the last DONE
        self._backs: set[int] = set()
        self._back_seq = -1
        self._bseq = random.randrange(256)
        self._btab_wire: bytes | None = None
        self._bcast_bat = None
        self._polled: int | None = None          # the client whose turn it is
        self._turn_end = 0.0
        # client state
        self.registered = False
        self._join_tries = 0
        self._train_until = 0.0                  # a TRAIN for us was heard: our responder may answer probes until then
        self._beacon_seen = 0.0
        self._last_poll = 0.0
        self._contact = False                    # polled / trained since the last beacon
        self._bcast_rx: _BcastRx | None = None
        self._bseen: "collections.deque[int]" = collections.deque(maxlen=64)
        self.on_data: Callable[[int, bytes], None] | None = None          # (origin, page)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        H, _ = _imports()
        with _quiet():
            a = H.make_msg_audio(self.fs, H.MSG_HUB, 0, pack_hub(K_DLVR, 0, 0, b"\0\0\0\0"), self.band)
        self.msg_air_s = len(a) / self.fs
        self.log(f"hub: {'HUB' if self.is_hub else 'client'} id={self.me} hub={self.hub_id} nodes={self.n_nodes} "
                 f"relay={self.relay} msg={self.msg_air_s:.1f}s slot={self._slot_s():.1f}s")
        self._stop.clear()
        for fn, name in ((self._tx_worker, "hub-tx"), (self._run_hub if self.is_hub else self._run_client, "hub-run")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        self._tx_evt.set()
        for t in self._threads:
            t.join(5.0)
        self._threads.clear()
        with self._lock:
            pages = list(self._queue) + list(self._waiting.values())
            self._queue.clear()
            self._waiting.clear()
        for p in pages:
            p.done.set()

    # -- what MeshNode / MeshReceiver ask -----------

    def order(self) -> list[int]:
        return sorted({self.me, self.hub_id, *self.clients})

    def expected_sender(self) -> int | None:
        if not self.is_hub:
            return self.hub_id
        with self._lock:
            return self._polled

    def initiator_may_start(self) -> bool:
        return self.is_hub and self.phase == "bat"

    def responder_may_listen(self, initiator_id: int) -> bool:
        if self.is_hub:
            return True                          # the hub's rx links come up preset: nothing to answer
        return initiator_id == self.hub_id and time.monotonic() < self._train_until

    def wait_data_window(self, timeout: float, need_s: float = 12.0) -> bool:
        return False                             # pages go through send_page, never MeshNode._send_ofdm

    def max_turn_gap_s(self) -> float:
        return 1e9

    def note_own_go(self) -> None:
        pass

    def extra_links(self) -> list:
        b = self._bcast_rx
        return [b] if b is not None else []

    # MeshReceiver hooks
    def rx_air(self, busy: bool) -> None:
        if busy:
            self._last_busy = time.monotonic()

    def rx_open(self) -> None:
        self._pending = True

    def rx_done(self, got, end_t: float) -> None:
        if got:
            self.rx_msg(got, end_t)
        self._pending = False

    def rx_probe(self, cap, start_t: float, end_t: float) -> bool:
        return False

    def rx_msg(self, got, end_t: float) -> None:
        H, _ = _imports()
        _, mtype, _, body = got
        if mtype != getattr(H, "MSG_HUB", -1):
            return
        m = unpack_hub(body)
        if m is None:
            self.log("hub: malformed HUB message -- ignored")
            return
        kind, frm, to, args = m
        if frm == self.me or to not in (self.me, BROADCAST_ID):
            if kind in (K_POLL, K_TRAIN) and not self.is_hub:
                self.counts["overheard"] += 1
            return
        self.counts["rx_" + K_NAMES[kind]] += 1
        detail = f" {args.hex()}" if args and kind != K_BTAB else ""
        self.log(f"hub: rx {K_NAMES[kind]} from={frm} to={to}{detail}")
        try:
            (self._hub_msg if self.is_hub else self._client_msg)(kind, frm, args, end_t)
        except Exception as e:
            self.log(f"hub: {K_NAMES[kind]} handling failed: {type(e).__name__}: {e}")

    # -- transmit -------------------------------------------------------------

    def _slot_s(self) -> float:
        return self.msg_air_s + GUARD_S + DECODE_S

    def _air_busy(self) -> bool:
        rx = self.node.rx
        return (rx is not None and rx.channel_busy()) or self._pending

    def _key(self, audio: np.ndarray, what: str) -> None:
        """Key one burst on a clear channel (carrier sense; ``ignore_busy`` skips it) and wait until it is off the air."""
        deadline = time.monotonic() + 20.0
        while not getattr(self.node.session, "ignore_busy", False) and self._air_busy() \
                and time.monotonic() < deadline and not self._stop.is_set():
            self._stop.wait(0.1)
        audio = keyup(self.node.session, audio, self.fs)
        self.node.session.play(audio, ident=False)
        pipe = self.node.session.pipe
        if pipe:
            pipe.wait_tx(len(audio) / self.fs + 10)
        time.sleep(0.3)

    def _msg_audio(self, kind: int, to: int, args: bytes) -> np.ndarray:
        H, _ = _imports()
        with _quiet():
            audio = H.make_msg_audio(self.fs, H.MSG_HUB, 0, pack_hub(kind, self.me, to, args), self.band)
            return H.scale_hs_tx(audio, self.cfg.amplitude, self.fs, is_probe=False, band=self.band)[0]

    def send_msg(self, kind: int, to: int, args: bytes = b"") -> None:
        """Key one HUB message now, on this thread."""
        if not self.is_hub and to != self.hub_id:
            raise RuntimeError(f"hub client {self.me}: refusing to address {to} (clients only answer the hub)")
        self.counts["tx_" + K_NAMES[kind]] += 1
        self.log(f"hub: tx {K_NAMES[kind]} to={to}" + (f" {args.hex()}" if args and kind != K_BTAB else ""))
        self._key(self._msg_audio(kind, to, args), K_NAMES[kind])

    def _later(self, at: float, fn, *a) -> None:
        """Run ``fn(*a)`` on the tx worker at monotonic time ``at``."""
        with self._lock:
            self._txq.append((at, fn, a))
        self._tx_evt.set()

    def _tx_worker(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                job = min(self._txq, key=lambda j: j[0]) if self._txq else None
            if job is None:
                self._tx_evt.wait(0.5)
                self._tx_evt.clear()
                continue
            wait = job[0] - time.monotonic()
            if wait > 0:
                self._tx_evt.wait(min(wait, 0.5))
                self._tx_evt.clear()
                continue
            with self._lock:
                try:
                    self._txq.remove(job)
                except ValueError:
                    continue
            try:
                job[1](*job[2])
            except Exception as e:
                self.log(f"hub: tx job failed: {type(e).__name__}: {e}")

    def _client_rank(self) -> int:
        """This client's slot among the hub_nodes - 1 client slots, from its own id alone."""
        r = self.me - (1 if self.me > self.hub_id else 0)
        return r % max(1, self.n_nodes - 1)

    # -- pages -------------------------------------------------------------------

    def _next_pid(self) -> int:
        with self._lock:
            self._pid = (self._pid + 1) & 0xFF
            return self._pid

    def send_page(self, dst: int, data: bytes) -> bool:
        """One page from this node's host to ``dst`` (a station id, or BROADCAST_ID = every station); blocks until it
        was delivered (True) or failed.  A client's page leaves in its next polled turn; with relay off it may only go
        to the hub."""
        dst = BROADCAST_ID if dst is None else dst
        if not self.is_hub:
            if dst != self.hub_id and not self.relay:
                self.log(f"hub: page to {dst} refused -- topology hub (no relay): a client sends to the hub only")
                return False
        elif dst == self.me:
            self.log("hub: page to ourselves refused")
            return False
        p = _Page(self.me, dst, self._next_pid(), bytes(data), local=True)
        with self._lock:
            self._queue.append(p)
        self.log(f"hub: page queued id={p.pid} to={dst} {len(data)}B")
        if not p.done.wait(self.page_timeout_s):
            with self._lock:
                if p in self._queue:
                    self._queue.remove(p)
                self._waiting.pop(p.pid, None)
            self.log(f"hub: page id={p.pid} to={dst} timed out after {self.page_timeout_s:.0f}s")
            return False
        return p.ok

    def _finish(self, p: _Page, ok: bool, why: str = "") -> None:
        p.ok = ok
        self.counts["page_ok" if ok else "page_fail"] += 1
        self.log(f"hub: page id={p.pid} from={p.origin} to={p.dst} {len(p.data)}B "
                 + ("DELIVERED" if ok else "FAILED") + (f" ({why})" if why else ""))
        p.done.set()

    def _deliver_up(self, origin: int, data: bytes, how: str) -> None:
        self.counts["page_rx"] += 1
        self.log(f"hub: page rx from={origin} {len(data)}B ({how})")
        if self.on_data:
            self.on_data(origin, data)

    def on_link_data(self, src: int, payload: bytes) -> None:
        """A frame payload from the arq link listening to ``src``."""
        got = self._reasm.add(src, payload)
        if got is None:
            return
        origin, dst, pid, page = got
        if self.is_hub:
            self._hub_page_in(src, origin, dst, pid, page)
        elif dst in (self.me, BROADCAST_ID):
            self._deliver_up(origin, page, f"unicast via {src}" if origin != src else "unicast")

    # -- broadcast frames (client side) -------------------------------------------

    def on_bcast(self, src: int, seq: int, data: bytes) -> None:
        """A broadcast data frame (dst 0xF) decoded by the shared receiver."""
        if self.is_hub or src != self.hub_id:
            return
        if not self.registered:
            self.log(f"hub: broadcast seq={seq} heard before registering -- not acked")
            return
        self._later(time.monotonic() + GUARD_S + self._client_rank() * self._slot_s(),
                    self.send_msg, K_BACK, self.hub_id, bytes([seq]))
        if seq in self._bseen:
            self.log(f"hub: broadcast seq={seq} duplicate (our BACK was lost) -- acked again")
            return
        self._bseen.append(seq)
        got = self._reasm.add(BROADCAST_ID, data)
        if got is not None:
            origin, _, _, page = got
            if origin == self.me:
                self.log(f"hub: broadcast of our own relayed page ({len(page)}B) -- not handed up")
                return
            self._deliver_up(origin, page, "broadcast" if origin == self.hub_id else f"broadcast relayed from {origin}")

    # -- client ---------------------------------------------------------------

    def _client_msg(self, kind: int, frm: int, args: bytes, end_t: float) -> None:
        if frm != self.hub_id:
            return                               # only the hub talks to a client
        now = time.monotonic()
        if kind == K_BEACON:
            self._beacon_seen = now
            contact, self._contact = self._contact, False
            if self.registered and not contact and args[2:3] and args[2] & F_JOIN_OPEN:
                # a trained client is polled every cycle: a whole cycle without a poll or a TRAIN means the hub dropped
                # us (it reopened the join for exactly that)
                self.log("hub: no poll since the last beacon -- the hub dropped us, joining again")
                self.registered = False
                self._join_tries = 0
                self.node.drop_links(self.hub_id)
            n, flags = (args[0], args[2]) if len(args) >= 3 else (0, 0)
            if n and n != self.n_nodes:
                self.log(f"hub: beacon says {n} nodes, we are set up for {self.n_nodes}")
            if bool(flags & F_RELAY) != self.relay:
                self.log(f"hub: beacon relay={bool(flags & F_RELAY)}, ours relay={self.relay}")
            if flags & F_JOIN_OPEN and not self.registered:
                rank = self._client_rank() if self._join_tries == 0 else random.randrange(max(1, self.n_nodes - 1))
                self._join_tries += 1
                self._later(now + GUARD_S + rank * self._slot_s(), self.send_msg, K_JOIN, self.hub_id)
        elif kind == K_TRAIN:
            self._register("train")
            self._train_until = now + self.bat_turn_s
            self._later(now, self._client_train)
        elif kind == K_POLL:
            self._register("poll")
            self._last_poll = now
            turn = (int.from_bytes(args[:2], "big") / 10.0) if len(args) >= 2 else self.turn_s
            self._later(now, self._client_turn, now + turn)
        elif kind == K_BTAB:
            self._client_btab(args)
        elif kind == K_DLVR and len(args) >= 3:
            pid, dst, ok = args[0], args[1], bool(args[2])
            with self._lock:
                p = self._waiting.pop(pid, None)
            if p is not None:
                self._finish(p, ok, f"relayed to {dst}" if ok else f"hub could not deliver to {dst}")

    def _register(self, why: str) -> None:
        self._contact = True
        if not self.registered:
            self.registered = True
            self.log(f"hub: registered with hub {self.hub_id} ({why})")

    def _client_train(self) -> None:
        """TRAIN for us: a fresh responder for the hub's probe; once it is up, the reverse link on the same tables."""
        node = self.node
        node.drop_links(self.hub_id)
        node._ensure_rx_link(self.hub_id)
        with node._lock:
            rx = node.links.get((self.hub_id, "rx"))
        if rx is None or not rx.wait_settled(self.bat_turn_s):
            self.log(f"hub: training with the hub failed: {rx.fail_reason if rx else 'no responder'}")
            return
        sm = rx.sm
        if not (sm and sm.my_tx_bat and sm.peer_tx_bat):
            self.log("hub: training: no tables on the trained link")
            return
        node.preset_link(self.hub_id, True, sm.my_tx_bat, sm.peer_tx_bat)
        self._train_until = 0.0
        self.log("hub: trained with the hub (both directions)")

    def _client_btab(self, args: bytes) -> None:
        H, _ = _imports()
        try:
            wire = H.bat_rle_unpack(args)
            bat = H.data_bat(wire, self.fs, self.band, self.cfg.chunk_bytes)
        except Exception as e:
            self.log(f"hub: bad broadcast table: {e}")
            return
        import dataclasses
        cfg = dataclasses.replace(self.cfg, node_id=self.me, peer_id=self.hub_id)
        self._bcast_rx = _BcastRx(cfg, self.hub_id, bat, self.band)
        self.log(f"hub: broadcast table {int((bat > 0).sum())} carriers {int(bat.sum())} bits")

    def _client_turn(self, end: float) -> None:
        """Our polled turn: the page we hold on the link to the hub, then DONE."""
        node, flags, sent = self.node, 0, 0
        with node._lock:
            tx = node.links.get((self.hub_id, "tx"))
        if tx is None or not tx.up:
            self.send_msg(K_DONE, self.hub_id, bytes([D_NEED_TRAIN, 0]))
            return
        while not self._stop.is_set():
            with self._lock:
                p = self._queue[0] if self._queue else None
            if p is None:
                break
            if end - time.monotonic() < 15.0:
                flags |= D_MORE
                break
            ok = True
            for piece in split_page(self.me, p.dst, p.pid, p.data, tx.capacity or 249):
                self.log(f"hub: page id={p.pid} to={p.dst} frame {len(piece)}B to the hub")
                if not tx.send(piece):
                    ok = False
                    break
                sent += 1
            with self._lock:
                if self._queue and self._queue[0] is p:
                    self._queue.popleft()
            if not ok:
                self._finish(p, False, "no ACK from the hub")
                if not tx.up or tx.state in ("failed", "closed"):
                    flags |= D_NEED_TRAIN
                break
            if p.dst == self.hub_id:
                self._finish(p, True)
            else:
                with self._lock:
                    self._waiting[p.pid] = p
                self.log(f"hub: page id={p.pid} with the hub for relay to {p.dst}: waiting for its delivery report")
        self.send_msg(K_DONE, self.hub_id, bytes([flags, sent & 0xFF]))

    def _run_client(self) -> None:
        """A client has no schedule of its own: it answers.  This only watches for a silent hub."""
        self.phase = "data"
        lost_s = max(4 * self.cycle_s, 3 * self.turn_s, 180.0)
        while not self._stop.wait(2.0):
            heard = max(self._last_poll, self._beacon_seen)
            if self.registered and heard and time.monotonic() - heard > lost_s:
                self.log(f"hub: nothing from the hub for {lost_s:.0f}s -- unregistered, waiting for its beacon")
                self.registered = False
                self._join_tries = 0
                self.node.drop_links(self.hub_id)

    # -- hub ------------------------------------------------------------------

    def _hub_msg(self, kind: int, frm: int, args: bytes, end_t: float) -> None:
        if kind == K_JOIN:
            with self._lock:
                self._joins.add(frm)
        elif kind == K_DONE:
            flags = args[0] if args else 0
            with self._lock:
                self._done_from = (frm, flags)
            self._done_evt.set()
        elif kind == K_BACK and args:
            with self._lock:
                if args[0] == self._back_seq:
                    self._backs.add(frm)

    def _hub_page_in(self, src: int, origin: int, dst: int, pid: int, page: bytes) -> None:
        if origin != src:
            self.log(f"hub: page from link {src} claims origin {origin} -- taken as {src}")
            origin = src
        if dst == self.me:
            self._deliver_up(origin, page, "to the hub")
            return
        if not self.relay:
            self.log(f"hub: page from {origin} for {dst} dropped: topology hub (no relay)")
            return
        p = _Page(origin, dst, pid, page, local=False)
        with self._lock:
            self._queue.append(p)
        self.counts["relay_in"] += 1
        self.log(f"hub: page id={pid} from={origin} for {dst} {len(page)}B stored for relay")

    def _run_hub(self) -> None:
        try:
            self._stop.wait(1.5)
            cycle = 0
            while not self._stop.is_set():
                cycle += 1
                t0 = time.monotonic()
                with self._lock:
                    joined = len(self.clients)
                if joined < self.n_nodes - 1:
                    self._beacon(cycle)
                self._train_all()
                self._hub_turn()
                self._polls()
                with self._lock:
                    self.phase = "idle"
                rest = self.cycle_s - (time.monotonic() - t0)
                if rest > 0:
                    self._stop.wait(rest)
        except Exception as e:                   # never die silently
            self.log(f"hub: schedule stopped on {type(e).__name__}: {e}")

    def _beacon(self, cycle: int) -> None:
        with self._lock:
            self.phase = "join"
            self._joins.clear()
        flags = F_JOIN_OPEN | (F_RELAY if self.relay else 0)
        self.send_msg(K_BEACON, BROADCAST_ID, bytes([self.n_nodes, cycle & 0xFF, flags]))
        end = time.monotonic() + GUARD_S + (self.n_nodes - 1) * self._slot_s() + DECODE_S
        while time.monotonic() < end or self._pending or self._air_busy():
            if self._stop.wait(0.2) or time.monotonic() > end + 20.0:
                break
        with self._lock:
            new = sorted(j for j in self._joins if j not in self.clients and j != self.me)
            for j in new:
                if len(self.clients) < self.n_nodes - 1:
                    self.clients[j] = {"trained": False, "missed": 0, "fails": 0}
            joined = sorted(self.clients)
        self.log(f"hub: join window closed: new={new} clients={joined}")

    def _train_all(self) -> None:
        with self._lock:
            todo = [c for c, st in sorted(self.clients.items()) if not st["trained"]]
        if not todo:
            return
        changed = False
        for c in todo:
            if self._stop.is_set():
                return
            with self._lock:
                self.phase = "bat"
            ok = self._train(c)
            with self._lock:
                st = self.clients.get(c)
                if st is None:
                    continue
                st["trained"] = ok
                st["fails"] = 0 if ok else st["fails"] + 1
                if not ok and st["fails"] >= MISS_LIMIT:
                    self.log(f"hub: training with {c} failed {st['fails']} times -- dropped, join reopens")
                    del self.clients[c]
            changed |= ok
        if changed:
            self._send_btab()

    def _train(self, c: int) -> bool:
        node = self.node
        node.drop_links(c)
        self.send_msg(K_TRAIN, c)
        lk = node._open_tx_link(c)
        if lk is None or not (lk.sm and lk.sm.my_tx_bat and lk.sm.peer_tx_bat):
            self.log(f"hub: training with {c} failed ({node.last_hs_fail or 'no link'})")
            return False
        node.preset_link(c, False, lk.sm.my_tx_bat, lk.sm.peer_tx_bat)
        self.log(f"hub: trained with {c} (both directions)")
        return True

    def _send_btab(self) -> None:
        """The broadcast table: per carrier the least bits any trained client's hub -> client table allows."""
        H, _ = _imports()
        node = self.node
        n = int(H.hs_cfg(self.fs, self.band)["n_data_carriers"])
        bats = []
        with node._lock:
            for (peer, d), lk in node.links.items():
                if d == "tx" and lk.up and lk.sm is not None and lk.sm.my_tx_bat and peer in self.clients:
                    bats.append(np.asarray(H.bytes_to_bat(lk.sm.my_tx_bat, n), int))
        if not bats:
            return
        wire = H.bat_to_bytes(np.min(np.stack(bats), axis=0))
        self._btab_wire = wire
        self._bcast_bat = H.data_bat(wire, self.fs, self.band, self.cfg.chunk_bytes)
        bits = int(self._bcast_bat.sum())
        self.log(f"hub: broadcast table over {len(bats)} clients: {int((self._bcast_bat > 0).sum())} carriers "
                 f"{bits} bits" + ("" if bits >= MIN_BCAST_BITS else " -- too few, broadcasts go unicast"))
        with self._lock:
            self.phase = "data"
        self.send_msg(K_BTAB, BROADCAST_ID, H.bat_rle_pack(wire))

    def _trained(self) -> list[int]:
        with self._lock:
            return [c for c, st in sorted(self.clients.items()) if st["trained"]]

    def _hub_turn(self) -> None:
        """The hub's own turn: every page it holds, its host's and (relay) the stored ones."""
        with self._lock:
            self.phase = "data"
            self._polled = None
            pages = list(self._queue)
            self._queue.clear()
        short = len(self._trained()) < self.n_nodes - 1
        held = []
        for p in pages:
            if self._stop.is_set():
                break
            if p.dst == BROADCAST_ID and short and time.monotonic() - p.t < self.page_timeout_s / 2:
                held.append(p)                   # "every station" = all hub_nodes - 1: wait for the rest to join
                continue
            if p.dst == BROADCAST_ID:
                ok, why = self._bcast_page(p)
            else:
                ok, why = self._unicast_page(p.dst, p)
            if p.local:
                self._finish(p, ok, why)
            else:
                self.counts["relay_ok" if ok else "relay_fail"] += 1
                self.log(f"hub: relay id={p.pid} {p.origin}->{p.dst} " + ("DELIVERED" if ok else f"FAILED ({why})"))
                if p.origin in self.clients:
                    self.send_msg(K_DLVR, p.origin, bytes([p.pid, p.dst & 0xFF, int(ok), 0]))
        if held:
            self.log(f"hub: {len(held)} broadcast page(s) held: {len(self._trained())} of {self.n_nodes - 1} "
                     f"stations trained")
            with self._lock:
                self._queue.extendleft(reversed(held))

    def _unicast_page(self, c: int, p: _Page) -> tuple[bool, str]:
        node = self.node
        if c not in self._trained():
            return False, f"{c} is not a registered, trained client"
        with node._lock:
            tx = node.links.get((c, "tx"))
        if tx is None or not tx.up:
            return False, f"no link to {c}"
        for piece in split_page(p.origin, p.dst, p.pid, p.data, tx.capacity or 249):
            self.log(f"hub: page id={p.pid} frame {len(piece)}B to {c}")
            if not tx.send(piece):
                with self._lock:
                    if c in self.clients:
                        self.clients[c]["trained"] = False
                return False, f"no ACK from {c}"
        return True, ""

    def _bcast_page(self, p: _Page) -> tuple[bool, str]:
        """Every frame once on the broadcast table, every client's BACK; resend for the ones missing, unicast as the
        last resort."""
        H, _ = _imports()
        clients = self._trained()
        if not clients:
            return False, "no trained clients"
        bat = self._bcast_bat
        if bat is None or int(bat.sum()) < MIN_BCAST_BITS:
            bad = [c for c in clients if not self._unicast_page(c, p)[0]]
            if bad:
                return False, f"unicast fallback missed {bad}"
            if len(clients) < self.n_nodes - 1:
                return False, f"reached {len(clients)} of {self.n_nodes - 1} stations"
            return True, ""
        missing_all: set[int] = set()
        for piece in split_page(p.origin, BROADCAST_ID, p.pid, p.data, 249):
            with self._lock:
                self._bseq = (self._bseq + 1) & 0xFF
                seq = self._bseq
                self._back_seq, self._backs = seq, set()
            body = pack_frame(seq, 0, piece, self.me, BROADCAST_ID)
            with _quiet():
                audio = H.make_data_audio(self.fs, body, bat, self.band, self.cfg.chunk_bytes)
                audio = H.scale_hs_tx(audio, self.cfg.amplitude, self.fs, is_probe=False, band=self.band)[0]
            want = set(clients)
            for attempt in range(BCAST_RETRIES + 1):
                if attempt == BCAST_RETRIES and self._btab_wire:
                    self.send_msg(K_BTAB, BROADCAST_ID, H.bat_rle_pack(self._btab_wire))   # a client may lack it
                self.counts["bcast_tx"] += 1
                self.log(f"hub: broadcast id={p.pid} seq={seq} {len(piece)}B attempt {attempt + 1} "
                         f"waiting for {sorted(want)}")
                self._key(audio, "BCAST")
                end = time.monotonic() + DECODE_S + GUARD_S + (self.n_nodes - 1) * self._slot_s() + 3.0
                while not self._stop.is_set():
                    with self._lock:
                        got = set(self._backs)
                    if want <= got and not self._pending:
                        break
                    if time.monotonic() > end and not self._pending and not self._air_busy():
                        break
                    if time.monotonic() > end + 30.0:
                        break
                    self._stop.wait(0.2)
                with self._lock:
                    got = set(self._backs)
                want -= got
                if not want:
                    break
            self.log(f"hub: broadcast id={p.pid} seq={seq} acked by {sorted(set(clients) - want)}"
                     + (f", missing {sorted(want)}" if want else ""))
            missing_all |= want
        if missing_all:
            bad = [c for c in sorted(missing_all) if not self._unicast_page(c, p)[0]]
            if bad:
                return False, f"no broadcast ACK and no unicast to {bad}"
            self.log(f"hub: broadcast id={p.pid} completed by unicast to {sorted(missing_all)}")
        if len(clients) < self.n_nodes - 1:          # sent short-handed after the hold: not "every station"
            return False, f"reached {len(clients)} of {self.n_nodes - 1} stations"
        return True, ""

    def _polls(self) -> None:
        for c in self._trained():
            if self._stop.is_set():
                return
            with self._lock:
                self.phase = "data"
                self._polled = c
                self._done_from = None
                self._done_evt.clear()
            turn = self.turn_s
            self.send_msg(K_POLL, c, int(turn * 10).to_bytes(2, "big"))
            node = self.node
            with node._lock:
                rx = node.links.get((c, "rx"))
            a0 = rx.last_activity if rx is not None else 0.0
            end = time.monotonic() + turn + DECODE_S + GUARD_S + 5.0
            while not self._stop.is_set() and time.monotonic() < end:
                if self._done_evt.wait(0.25):
                    with self._lock:
                        d = self._done_from
                    if d and d[0] == c:
                        break
                    self._done_evt.clear()
            with self._lock:
                d = self._done_from if self._done_from and self._done_from[0] == c else None
                self._polled = None
                st = self.clients.get(c)
                if st is None:
                    continue
                active = rx is not None and rx.last_activity > a0
                if d is None and not active:
                    st["missed"] += 1
                    self.log(f"hub: no answer from {c} to its poll ({st['missed']}/{MISS_LIMIT})")
                    if st["missed"] >= MISS_LIMIT:
                        self.log(f"hub: {c} dropped after {MISS_LIMIT} silent polls -- join reopens")
                        del self.clients[c]
                        drop = True
                    else:
                        drop = False
                else:
                    st["missed"] = 0
                    drop = False
                    if d and d[1] & D_NEED_TRAIN:
                        st["trained"] = False
                        self.log(f"hub: {c} asks to be trained again")
            if drop:
                node.drop_links(c)
