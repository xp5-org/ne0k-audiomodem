# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Mesh node, one-way transfers: CSS control plane, OFDM data plane brought up per link.

Every node runs the same code on a shared medium.  A transfer ``src -> dst`` is:

1. ``src`` sends ``OFDM_HS_REQ`` to ``dst`` over CSS (a robust, slow, always-available PHY) and
   waits for the CSS ACK.  Any other node that overhears the REQ stays off the air for
   ``hold_s`` so it cannot trample the handshake that follows.
2. ``dst`` ACKs over CSS and, once the ACK has aired, becomes the responder of an
   ``OfdmLink``; ``src`` becomes its initiator (V.22-style BAT exchange, same as the
   point-to-point modes).  Both links run ``mode="arq"``: only ``src`` sends data, ``dst``
   answers each frame with a pure-ACK frame.
3. ``src`` sends the payload over that link.  If the REQ is never answered, the handshake
   fails or the data goes unACKed, the node falls back to CSS DATA (stop-and-wait with a CSS
   ACK), as the reference ``PhyPolicy`` does.  There is no timed re-upgrade: the next
   ``send`` simply asks again.

Links that see no traffic for ``link_idle_s`` are torn down on both ends, so a quiet mesh
leaves the band to the CSS control plane.  ``phy_mode="css_only"`` skips OFDM altogether.

BLOCKOUT (``send_blockout``/``end_blockout``, ``self.blockout``): an on-demand exclusive window
for two nodes, requiring OFDM -- CSS's baud rate cannot carry it, so it is only available under
``phy_mode="ofdm_only"`` (refused outright under ``css_only``/``css_then_ofdm`` for now; a
CSS-control-plane variant of this feature, for when a ``css_then_ofdm`` link is already up, is
future work, not this).  One end broadcasts ``BLOCKOUT_REQ`` (the peer + a duration) as a short
OFDM frame on the shared band, same as ``MSG_ANNOUNCE``; every OTHER node mutes -- TX-silent,
still listening -- for that long instead of running its own schedule, while the two named nodes
transfer (one-way, or two-way if the peer answers back during the same window).  Either end
broadcasts ``BLOCKOUT_END`` the moment it is actually done, so muted nodes resume early and
simply carry on with whatever was already queued -- nothing is discarded while muted, so there
is no backlog to replay, just normal CSMA/CA picking up again.  See ``mesh_mac.Blockout``.

Medium access follows the reference daemon (``mesh_mac``): every CSS packet goes through a
CSMA/CA outbox (carrier sense, ``U(0, cw_s) + id * id_slot_s`` backoff, ID-staggered ACKs), beacons
are paced by the measured clock drift of the peers, handshakes wait for a quiet channel (and for the
link's exclusive wall-clock window when ``airtime_windows`` is on), and a duty-cycled passive detector
re-arms an OFDM upgrade that was given up on when it hears OFDM on the air.

Not ported from ``meshmodem_node``: the mesh-specific OFDM frame wrapper (``MeshModulator``,
conv-coded 48-byte frames) -- data rides the same frames as the point-to-point modes -- and
band renegotiation.
"""
from __future__ import annotations

import collections
import dataclasses
import struct
import threading
import time

import numpy as np

from .css_link import CssLink
from .css_link import _imports as _css_imports
from .events import Decoded, LinkEvent
from .mesh_mac import Airtime, Ambient, Blockout, Csma, Rechirp
from .modemcfg import CssConfig, OfdmConfig, _Config
from .mesh_rx import MeshReceiver
from .ofdm_link import OfdmLink, _Capture, _quiet, keyup
from .ofdm_link import _imports as _ofdm_imports

BROADCAST = 0xF
# Shared channel round trip for one arq frame, measured on the 3-node SIP conference (2026-11-06): the burst's
# 0.6 s closing quiet, ~1.5 s decode per link (every link of a node decodes every burst, one DSP call at a time),
# the ACK's 2 s keyup + frame, the same again on the sender = 11-13 s.  The point-to-point 8 s default resent every
# frame once while its ACK was still being decoded.
MESH_ACK_WAIT_S = 15.0

# BLOCKOUT_REQ/END wire bodies -- OFDM broadcast frames (MSG_BLOCKOUT_REQ/END in
# ofdm_handshake.py), same scale as the old CSS encoding (css_modem.pack_blockout_req_payload)
# so nothing about the announced window itself changes, only which PHY carries it.
_BO_REQ_BODY = "<BBH"          # announcer id, peer id, duration (BLOCKOUT_DUR_SCALE units)
_BO_END_BODY = "<B"            # announcer id
BLOCKOUT_DUR_SCALE = 0.1
_BLOCKOUT_DUR_MAX = 0xFFFF


def _frame(data: bytes) -> bytes:           # CSS pads to its chunk size: carry the length
    return bytes([len(data)]) + data


def _unframe(payload: bytes) -> bytes:
    return payload[1:1 + payload[0]] if payload else b""


class _Job:
    """One CSS packet waiting for the radio."""

    def __init__(self, fn, args, beacon: bool = False) -> None:
        self.fn, self.args, self.beacon = fn, args, beacon
        self.done = threading.Event()
        self.air = 0.0
        self.err: Exception | None = None


@dataclasses.dataclass
class MeshConfig(_Config):
    node_id: int = 1
    phy_mode: str = "css_then_ofdm"      # css_only | css_then_ofdm | ofdm_only
    req_retries: int = 2                 # REQ resends when the CSS ACK does not come back
    data_retries: int = 2                # CSS DATA resends
    ack_wait_s: float = 0.0              # CSS ACK wait; 0 = derive from the ACK airtime
    hold_s: float = 150.0                # stay quiet this long after overhearing a REQ
    link_idle_s: float = 90.0            # tear an OFDM link down after this much silence
    beacon: bool = False                 # announce this node on the CSS control plane (mesh topology)
    beacon_min_s: float = 20.0           # the beacon interval follows the peers' clock drift, clamped to [min, max]
    beacon_max_s: float = 300.0
    first_beacon_s: float = 20.0         # listen first: the first beacon comes a random time up to this after start
    cw_s: float = 4.0                    # CSMA/CA contention window
    id_slot_s: float = 0.1               # per-id backoff phase (id % 16 slots)
    ack_stagger_s: float = 0.5           # per-id hold before answering (id % 16 steps)
    airtime_windows: bool = False        # confine each link's handshake to its own wall-clock window
    hs_window_s: float = 150.0
    hs_slots: int = 4
    hs_guard_s: float = 45.0
    hs_init_offset_s: float = 6.0
    announce_round: bool = False         # ofdm_only: id-ordered OFDM presence round + BAT turn order (ofdm_announce)
    announce_bat_turn_s: float = 60.0
    announce_bat_mode: str = "pairwise"  # pairwise (acoustic) | broadcast (one probe each, see ofdm_announce)
    announce_cycle_s: float = 180.0
    ofdm_detect: bool = True             # passively listen for OFDM to re-arm an upgrade that was given up on
    ofdm_detect_period_s: float = 5.0
    ofdm_detect_thr: float = 0.5


class MeshNode:
    def __init__(self, session, mesh: MeshConfig, css: CssConfig, ofdm: OfdmConfig,
                 log=lambda m: None, own_events: bool = True) -> None:
        if mesh.phy_mode not in ("css_only", "css_then_ofdm", "ofdm_only"):
            raise ValueError("phy_mode must be css_only, css_then_ofdm or ofdm_only")
        self.session, self.mesh, self.log = session, mesh, log
        self.own_events = own_events              # False: the host drains the session and calls handle()
        self.css_cfg = dataclasses.replace(css, node_id=mesh.node_id)
        self.ofdm_cfg = ofdm
        self.css = CssLink(session, self.css_cfg)
        self.on_data = None                       # f(src, bytes)
        self.links: dict[tuple[int, str], OfdmLink] = {}      # (peer, "tx" | "rx") -> link
        self.events: collections.deque = collections.deque(maxlen=4000)
        self.rx_count = self.tx_count = self.css_fallbacks = self.reqs = 0
        self._seq = 0
        self._acks: set[tuple[int, int]] = set()
        self._ack_evt = threading.Event()
        self._last_rx: dict[int, int] = {}
        self._hold_until = 0.0
        self.neighbors: dict[int, tuple[float, float]] = {}    # node -> (snr_db, last heard)
        self.beacons_tx = 0
        self._csslock = threading.Lock()
        c = self.css_cfg
        sps = int(c.fs * c.symbol_dur)
        sps += int(round(sps * c.symbol_gap_frac))
        self.csma = Csma(mesh.node_id, cw_s=mesh.cw_s, id_slot_s=mesh.id_slot_s, ack_stagger_s=mesh.ack_stagger_s)
        self.rechirp = Rechirp(mesh.node_id, c.fs, sps, min_s=mesh.beacon_min_s, max_s=mesh.beacon_max_s,
                               first_s=mesh.first_beacon_s)
        self.air = (Airtime(window_s=mesh.hs_window_s, slots=mesh.hs_slots, guard_s=mesh.hs_guard_s,
                            init_offset_s=mesh.hs_init_offset_s) if mesh.airtime_windows else None)
        self.ambient = Ambient()
        self.css.on_level = self.ambient.add
        self.want_ofdm: dict[int, bool] = {}      # peer -> OFDM upgrade authorised (default yes)
        self.blockout = Blockout(mesh.node_id)
        self.detector_hits = 0
        self.ofdm_only_violations: list[str] = []   # every CSS tx refused/caught under phy_mode=ofdm_only
        self._mac_q = collections.deque()
        self.announce = None                  # AnnounceRound, ofdm_only + mesh.announce_round
        self.ring_dst_fn = lambda: None       # who this node trains its tx link to (set by the host app)
        self.rx_peer = 0                      # who this node expects to hear from (serial.rx_host)
        self._tx_open = threading.Lock()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.rx: MeshReceiver | None = None    # ofdm_only: the node's one ear (mesh_rx)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        ofdm_only = self.mesh.phy_mode == "ofdm_only"
        if not ofdm_only:                         # ofdm_only: CSS takes no part at all, not even listening
            self.css.start()
        self.session.ident.channel_busy = self._channel_busy
        detect = self.mesh.phy_mode == "css_then_ofdm" and self.mesh.ofdm_detect   # re-arms a CSS->OFDM upgrade
        for fn, name in ((((self._ctl, "mesh-ctl"),) if self.own_events else ())
                          + ((self._reap, "mesh-reap"), (self._mac, "mesh-mac"))
                          + (((self._detect, "mesh-detect"),) if detect else ())):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        if not ofdm_only:
            return                                # css_only / css_then_ofdm: unchanged (each OFDM link listens itself)
        # ofdm_only -- one ear for the one channel: data frames for every link, announce/GO, blockout REQ/END, IDs
        self.rx = rx = MeshReceiver(self.session, self.ofdm_cfg, self.me, lambda m: self.log(m))
        rx.links = self._all_links
        rx.expected = self._expected_senders
        rx.data_first = self._data_on_air
        rx.on_msg = self._on_ofdm_msg               # hear BLOCKOUT_REQ/END whether we ever send one or not
        rx.on_level = self.ambient.add              # carrier sense and ambient come from the OFDM ear
        if self.mesh.announce_round:
            from .ofdm_announce import AnnounceRound
            self.announce = a = AnnounceRound(self.session, self.ofdm_cfg, self.me, lambda m: self.log(m),
                                              bat_turn_s=self.mesh.announce_bat_turn_s,
                                              cycle_s=self.mesh.announce_cycle_s, shared_rx=True,
                                              bat_mode=self.mesh.announce_bat_mode)
            a.link_needed = self._tx_link_needed
            a.on_turn = self._turn_open_tx
            a.on_peer_turn = self._peer_turn
            a.on_tables = self._links_from_tables
            a.on_peer_handshake = self._peer_handshake
            rx.announce = a
            a.start()
        rx.start()

    def _all_links(self) -> list[OfdmLink]:
        with self._lock:
            return list(self.links.values())

    def _expected_senders(self) -> list[int]:
        """Who is most likely transmitting: the data turn's owner, then peers we await an ACK from, then the
        peer whose data we listen for."""
        out = []
        who = self.announce.expected_sender() if self.announce else None
        if who is not None and who != self.me:
            out.append(who)
        out += [lk.cfg.peer_id for lk in self._all_links() if lk.up and lk.wants_ack()]
        if self.rx_peer:
            out.append(self.rx_peer)
        return list(dict.fromkeys(out))

    def _data_on_air(self) -> bool:
        """Payload frames are most likely in the data turns (or while one of ours awaits its ACK); the round and BAT
        turns carry messages.  Only orders the receiver's two decodes: it never skips one."""
        if self.announce is None:
            return True
        if self.announce.phase in ("data", "idle"):
            return True
        return any(lk.up and lk.wants_ack() for lk in self._all_links())

    def _tx_link_needed(self) -> bool:
        dst = self.ring_dst_fn()
        if not dst:
            return False
        with self._lock:
            lk = self.links.get((dst, "tx"))
        return lk is None or not lk.up

    def _turn_open_tx(self) -> bool:
        dst = self.ring_dst_fn()
        return bool(dst) and self._open_tx_link(dst) is not None

    def _peer_turn(self, initiator: int) -> None:
        """It is ``initiator``'s BAT turn: make sure this node has a fresh responder waiting for it."""
        if initiator != self.rx_peer:
            return
        with self._lock:
            lk = self.links.get((initiator, "rx"))
            stale = lk if lk is not None and lk.state in ("failed", "closed") else None
            if stale is not None:
                self.links.pop((initiator, "rx"), None)
        if stale is not None:
            stale.stop()
        self._ensure_rx_link(initiator)       # no-op while a live responder is already waiting

    def _peer_handshake(self, initiator: int) -> None:
        """``initiator`` announced a pairwise handshake (bat_mode broadcast): whatever rx link we hold for it -- up
        from the broadcast round, or stale -- goes, and a fresh responder waits for its probe."""
        if initiator != self.rx_peer:
            return
        with self._lock:
            old = self.links.get((initiator, "rx"))
            if old is not None and old.passive_listen():
                return                            # a fresh responder is already waiting for this probe
            self.links.pop((initiator, "rx"), None)
        if old is not None:
            old.stop()
        self._ensure_rx_link(initiator)

    def _links_from_tables(self, tables: dict) -> bool:
        """The broadcast BAT round's confirmed tables: bring up the links this node needs (its tx link to the ring
        destination, its rx link from ``rx_peer``) on them, with no handshake.  True when every needed link is up."""
        want = []
        dst = self.ring_dst_fn()
        if dst:
            want.append((dst, True))
        if self.rx_peer:
            want.append((self.rx_peer, False))
        ok = True
        for peer, tx in want:
            mine, theirs = tables.get((self.me, peer)), tables.get((peer, self.me))
            if not (mine and theirs):
                self.log(f"broadcast BAT: no table pair with {peer} -- pairwise handshake needed")
                ok = False
                continue
            key = (peer, "tx" if tx else "rx")
            lk = self._new_link(peer, requester=tx)
            lk.preset = (mine, theirs)
            with self._lock:
                old = self.links.pop(key, None)
                self.links[key] = lk
            if old is not None:
                old.stop()
            lk.start()
            if not lk.wait_settled(10.0):
                self.log(f"broadcast BAT: link {key} did not come up: {lk.fail_reason}")
                ok = False
        return ok

    def stop(self) -> None:
        if self.rx:
            self.log(f"mesh rx: {self.rx.stats()}")
            self.rx.stop()
            self.rx = None
        if self.announce:
            self.announce.stop()
            self.announce = None
        self._stop.set()
        for t in self._threads:
            t.join(5.0)
        self._threads.clear()
        with self._lock:
            links, self.links = list(self.links.values()), {}
        for lk in links:
            lk.stop()
        self.css.stop()
        if self.session.ident.channel_busy == self._channel_busy:
            self.session.ident.channel_busy = lambda: False

    @property
    def me(self) -> int:
        return self.mesh.node_id

    # -- CSS ----------------------------------------------------------------

    def _ofdm_only_violation(self, msg: str) -> str:
        """Record and loudly log a CSS transmission that phy_mode=ofdm_only must never allow."""
        msg = f"OFDM_ONLY VIOLATION: {msg}"
        self.ofdm_only_violations.append(msg)
        self.log(msg)
        return msg

    def _css_tx(self, fn, *a, tag: str = "", delay: float = 0.0) -> float:
        """Queue one CSS packet on the CSMA/CA outbox and block until it has aired.  Returns its
        airtime, or 0.0 when an identical packet was already waiting (``tag``)."""
        job = _Job(fn, a)
        tag = tag or f"x:{id(job)}"
        with self._lock:
            if not self.csma.enqueue(job, tag, time.monotonic() + delay):
                return 0.0
        while not job.done.wait(0.5):
            if self._stop.is_set():
                return 0.0
        if job.err:
            raise job.err
        return job.air

    def _channel_busy(self) -> bool:
        """Carrier sense: CSS on the air, one of our own OFDM handshakes in progress (a responder
        still waiting for its first probe is only listening, so it does not count), or (for a
        bystander only -- see Blockout.muted) someone else's exclusive blockout window."""
        if self.blockout.muted(time.monotonic()):
            return True
        if self.mesh.phy_mode == "ofdm_only":
            if self.rx is not None and self.rx.channel_busy():
                return True
        elif self.css.channel_busy():
            return True
        with self._lock:
            return any(not lk.up and lk.state not in ("failed", "closed") and not lk.passive_listen()
                       for lk in self.links.values())

    def _mac(self) -> None:
        """The single radio: release one outbox packet at a time under CSMA/CA, and queue the beacon
        when the drift-driven cadence says it is due."""
        t0 = time.monotonic()
        self.rechirp.next_at += t0                     # first beacon is relative to start
        while not self._stop.wait(0.1):
            now = time.monotonic()
            busy = self._channel_busy()
            with self._lock:
                if (self.mesh.beacon and self.mesh.phy_mode != "ofdm_only"
                        and now >= self.rechirp.next_at and now >= self._hold_until
                        and not self.blockout.muted(now)
                        and self.rechirp.due(now) and not self.csma.pending("beacon:")):
                    self.log("queueing CSS beacon")
                    self.csma.enqueue(_Job(self.css.send_type, (BROADCAST, 0, 0, b"", "BEACON"), beacon=True),
                                      "beacon:", 0.0)
                got = self.csma.poll(now, busy)
            if not got:
                continue
            job = got[0]
            if self.mesh.phy_mode == "ofdm_only":
                # Final backstop: nothing should ever reach the outbox under ofdm_only (every
                # enqueue point above is gated too), but if one slips through, refuse to key the
                # radio rather than silently transmitting CSS.
                job.err = RuntimeError(self._ofdm_only_violation(
                    f"refused to transmit a queued CSS job ({job.fn.__qualname__})"))
                job.done.set()
                continue
            try:
                with self._csslock:
                    job.air = job.fn(*job.args)
                pipe = self.session.pipe
                if pipe:
                    pipe.wait_tx(job.air + 10)
                time.sleep(0.3)
                if job.beacon:
                    self.beacons_tx += 1
            except Exception as e:
                job.err = e
                self.log(f"css tx failed: {e}")
            job.done.set()
        with self._lock:
            for job, _, _ in self.csma.outbox:
                job.done.set()

    def _ack_wait(self) -> float:
        if self.mesh.ack_wait_s:
            return self.mesh.ack_wait_s
        # the peer holds its ACK by its id stagger and backs off before sending it
        return (self.css.ack_airtime() + 2 * self.css_cfg.hang_s + 10 + self.mesh.cw_s
                + 16 * self.mesh.id_slot_s + Csma.max_ack_delay(self.mesh.ack_stagger_s))

    def _wait_ack(self, peer: int, seq: int) -> bool:
        end = time.monotonic() + self._ack_wait()
        while not self._stop.is_set():
            if (peer, seq) in self._acks:
                self._acks.discard((peer, seq))
                return True
            self._ack_evt.clear()
            if time.monotonic() >= end:
                return False
            self._ack_evt.wait(min(0.5, end - time.monotonic()))
        return False

    def _next_seq(self) -> int:
        with self._lock:
            self._seq = (self._seq + 1) % 16
            return self._seq

    def _ctl(self) -> None:
        while not self._stop.wait(0.2):
            for ev in self.session.drain():
                self.handle(ev)

    def handle(self, ev) -> None:
        """One session event: the control plane reacts to CSS packets addressed to this node."""
        if isinstance(ev, LinkEvent):
            self.events.append(ev)
        elif isinstance(ev, Decoded) and ev.ok:
            self._on_css(ev)

    def link_states(self) -> dict[str, str]:
        with self._lock:
            return {f"{peer}/{d}": lk.state for (peer, d), lk in self.links.items()}

    def best_neighbor(self) -> int | None:
        """The node to send to when the user has not named one: the strongest one heard recently."""
        now = time.monotonic()
        live = {n: snr for n, (snr, t) in self.neighbors.items() if now - t < 3 * self.mesh.beacon_max_s}
        return max(live, key=live.get) if live else None

    def _on_css(self, ev: Decoded) -> None:
        me = self.me
        if ev.src != me:
            self.neighbors[ev.src] = (ev.snr_db, time.monotonic())
            self.rechirp.heard(ev.src, ev.grid_sps)
            if ev.type_name in ("DATA", "ACK") and ev.dst in (me, BROADCAST) and \
                    self.mesh.phy_mode in ("css_then_ofdm", "ofdm_only"):
                self._peer_on_css(ev.src)
        if ev.dst not in (me, BROADCAST):
            if ev.type_name == "OFDM_HS_REQ":      # two other nodes are about to handshake
                self._hold_until = time.monotonic() + self.mesh.hold_s
                self.log(f"overheard REQ {ev.src}->{ev.dst}: holding off")
            return
        if ev.type_name == "ACK":
            self._acks.add((ev.src, ev.seq))
            self._ack_evt.set()
            now = time.monotonic()
            self.session._emit(LinkEvent("ack", me, ev.src, now, now, f"css seq={ev.seq}"))
        elif ev.type_name == "OFDM_HS_REQ" and ev.dst == me:
            if self.mesh.phy_mode == "css_only":
                return
            if self.mesh.phy_mode == "ofdm_only":
                # ofdm_only takes no part in the CSS REQ/ACK handshake, in either direction --
                # acking this would itself be a CSS transmission.
                self._ofdm_only_violation(f"ignored OFDM_HS_REQ from {ev.src} (refusing to ACK over CSS)")
                return
            threading.Thread(target=self._on_req, args=(ev.src, ev.seq), name="mesh-req",
                             daemon=True).start()
        elif ev.type_name == "DATA" and ev.dst == me:
            if self.mesh.phy_mode == "ofdm_only":
                self._ofdm_only_violation(f"ignored CSS DATA from {ev.src} (refusing to ACK over CSS)")
                return
            threading.Thread(target=self._css_ack, args=(ev.src, ev.seq), daemon=True).start()
            if self._last_rx.get(ev.src) == ev.seq:
                self.log(f"css rx duplicate seq={ev.seq} from {ev.src}")
                return
            self._last_rx[ev.src] = ev.seq
            self.rx_count += 1
            data = _unframe(ev.payload)
            self.log(f"css rx {len(data)}B from {ev.src}")
            if self.on_data:
                self.on_data(ev.src, data)
        elif ev.type_name == "BLOCKOUT_REQ":
            self._on_blockout_req(ev)
        elif ev.type_name == "BLOCKOUT_END":
            self._on_blockout_end(ev)

    def _on_blockout_req(self, ev: Decoded) -> None:
        """CSS ``BLOCKOUT_REQ`` -- never reached under ``phy_mode="ofdm_only"``: that mode's
        BLOCKOUT_REQ is an OFDM broadcast decoded by ``_on_ofdm_msg`` (via the MeshReceiver), not a CSS event (see
        ``send_blockout``)."""
        try:
            _, cm, _ = _css_imports()
            peer_b, duration_s = cm.unpack_blockout_req_payload(ev.payload)
        except Exception as e:
            self.log(f"bad BLOCKOUT_REQ from {ev.src}: {e}")
            return
        self._apply_blockout_req(ev.src, peer_b, duration_s)

    def _apply_blockout_req(self, src: int, peer_b: int, duration_s: float) -> None:
        """``src`` and ``peer_b`` are about to trade exclusively for the announced window.
        Everyone else (us, unless we are that named peer) mutes for its duration -- see
        ``Blockout.on_req``/``_channel_busy``.  Shared by the CSS path (``_on_blockout_req``,
        css_only/css_then_ofdm) and the OFDM broadcast path (``_on_ofdm_msg``, ofdm_only)."""
        now = time.monotonic()
        self.blockout.on_req(src, peer_b, duration_s, now)
        if self.blockout.participant:
            self.log(f"blockout with {self.blockout.peer}: window {duration_s:.0f}s")
            self.session._emit(LinkEvent("blockout", self.me, self.blockout.peer, now,
                                         now + duration_s, f"window {duration_s:.0f}s"))
            if self.mesh.phy_mode == "ofdm_only":
                # ofdm_only: no CSS OFDM_HS_REQ/ACK bootstrap -- the broadcast we just heard
                # already named both of us, so there is nothing left to negotiate over CSS.
                # Start listening for src's OFDM send right now instead of waiting for a
                # CSS REQ that is never going to arrive (see _on_req, send_blockout).
                self._ensure_rx_link(src)
        else:
            self.log(f"overheard blockout {src}<->{peer_b}: muting {duration_s:.0f}s")
            self.session._emit(LinkEvent("blockout", self.me, -1, now, now + duration_s,
                                         f"muted for {src}<->{peer_b}"))

    def _on_blockout_end(self, ev: Decoded) -> None:
        """CSS ``BLOCKOUT_END`` -- never reached under ``phy_mode="ofdm_only"``, see
        ``_on_blockout_req``."""
        self._apply_blockout_end(ev.src)

    def _apply_blockout_end(self, src: int) -> None:
        """Resume now instead of waiting out the full window -- whichever participant finished
        the transfer sends this the moment it is done.  Shared by the CSS and OFDM paths, same
        as ``_apply_blockout_req``."""
        now = time.monotonic()
        was_active = self.blockout.active(now)
        self.blockout.on_end(now)
        if was_active:
            self.log(f"blockout end (from {src}): resuming")
            self.session._emit(LinkEvent("blockout", self.me, -1, now, now, "resume"))

    def _ofdm_band(self) -> tuple[float, float]:
        return (self.ofdm_cfg.freq_lo, self.ofdm_cfg.freq_hi)

    def _on_ofdm_msg(self, got) -> None:
        """ofdm_only: BLOCKOUT_REQ/BLOCKOUT_END broadcasts, handed over by the node's MeshReceiver.  The broadcast
        IS the mechanism (see ``send_blockout``/``end_blockout``), so this is the only way any node -- muted or
        not -- ever learns a window opened or closed."""
        H, _ = _ofdm_imports()
        _, mtype, _, body = got
        req_len, end_len = struct.calcsize(_BO_REQ_BODY), struct.calcsize(_BO_END_BODY)
        if mtype == H.MSG_BLOCKOUT_REQ and len(body) >= req_len:
            announcer, peer_b, ds = struct.unpack(_BO_REQ_BODY, body[:req_len])
            if announcer != self.me:
                self._apply_blockout_req(announcer, peer_b, ds * BLOCKOUT_DUR_SCALE)
        elif mtype == H.MSG_BLOCKOUT_END and len(body) >= end_len:
            (announcer,) = struct.unpack(_BO_END_BODY, body[:end_len])
            if announcer != self.me:
                self._apply_blockout_end(announcer)

    def _peer_on_css(self, src: int) -> None:
        """Any CSS data/ACK from an OFDM-linked peer is proof it is back on CSS: drop our end of
        the link so our own ACK rides CSS too and both ends converge with no control traffic."""
        with self._lock:
            gone = [self.links.pop(k) for k in ((src, "tx"), (src, "rx")) if k in self.links]
        for lk in gone:
            self.log(f"CSS packet from {src} while OFDM linked: dropping the OFDM link")
            threading.Thread(target=lk.stop, daemon=True).start()

    def _detect(self) -> None:
        """Duty-cycled passive OFDM sensing while parked on CSS: a hit re-arms the upgrade (a REQ over
        CSS, never a jump straight onto OFDM) toward every peer we gave up on."""
        try:
            from . import ofdm_link
            ofdm_link._imports()
            import meshlib
        except Exception as e:
            self.log(f"OFDM detector unavailable: {e}")
            return
        pipe = self.session.pipe
        if not pipe:
            return
        fs, per = self.css_cfg.fs, self.mesh.ofdm_detect_period_s
        tap = pipe.subscribe(256)
        buf, n, last = collections.deque(), 0, time.monotonic()
        while not self._stop.is_set():
            b = tap.get(timeout=0.5)
            if b is not None:
                if b.tx_active:
                    buf.clear(); n = 0
                else:
                    buf.append(b.samples); n += len(b.samples)
                    while n - len(buf[0]) > (per + 1.0) * fs:
                        n -= len(buf.popleft())
            now = time.monotonic()
            if now - last < per:
                continue
            last = now
            peers = list(self.neighbors)
            if all(self.want_ofdm.get(p, True) for p in peers) or self._channel_busy() or not buf:
                continue                                # nothing given up on, or the channel is in use
            with self._lock:
                if self.links:
                    continue
            hit, coh = meshlib.detect_ofdm_activity(np.concatenate(buf), fs, thr=self.mesh.ofdm_detect_thr,
                                                    amb_rms=self.ambient.rms)
            if hit:
                self.detector_hits += 1
                self.log(f"OFDM activity on the air (coherence {coh}): re-arming the upgrade")
                for p in peers:
                    self.want_ofdm[p] = True

    def _css_ack(self, dst: int, seq: int) -> None:
        # Defense in depth: _on_css already refuses to spawn this thread under ofdm_only, but it
        # runs detached (daemon thread, no caller to see an exception), so guard again here.
        if self.mesh.phy_mode == "ofdm_only":
            self._ofdm_only_violation(f"_css_ack({dst}) called while phy_mode=ofdm_only -- refusing to ACK over CSS")
            return
        # held by this node's id stagger so two peers never answer at the same instant
        self._css_tx(self.css.send_ack, dst, seq, tag=f"ack:{dst}:{seq}", delay=self.csma.ack_delay())

    def _hs_gate(self, peer: int, initiator: bool, ready=None) -> bool:
        """Wait until a handshake with ``peer`` may start: inside the link's airtime window (when
        enabled) and, for the initiator, on a quiet channel."""
        wait = (self.air.cycle if self.air else 0.0) + 60.0
        if self.announce:
            wait = self.mesh.announce_cycle_s + 120.0
        end = time.monotonic() + wait
        while not self._stop.is_set():
            if ready is not None and ready():          # the link came up meanwhile (our BAT turn opened it)
                return True
            slot_ok = (self.announce.initiator_may_start() if initiator else True) if self.announce else \
                (self.air is None or self.air.hs_open(self.me, peer, initiator))
            if slot_ok and (not initiator or not self._channel_busy()):
                return True
            if time.monotonic() >= end:
                self.log(f"no handshake slot with {peer} within {wait:.0f}s")
                return False
            self._stop.wait(0.5)
        return False

    # -- OFDM links ---------------------------------------------------------

    def _new_link(self, peer: int, requester: bool) -> OfdmLink:
        role = "initiator" if requester else "responder"
        cfg = dataclasses.replace(self.ofdm_cfg, node_id=self.me, peer_id=peer, role=role,
                                  mode="arq", sender="initiator",
                                  ack_wait_s=max(self.ofdm_cfg.ack_wait_s, MESH_ACK_WAIT_S))
        lk = OfdmLink(self.session, cfg, log=lambda m, p=peer: self.log(f"ofdm[{p}] {m}"))
        lk.shared_rx = self.rx is not None                     # ofdm_only: data frames come from self.rx, decoded once
        if self.announce:                                      # a link coming up ends the current BAT turn, once
            lk.on_state = lambda st, an=self.announce: an.note_own_go() if st == "up" else None
        if not requester:
            lk.on_data = lambda d, p=peer: self._ofdm_rx(p, d)
            lk.rx_gate = lambda p=peer: (self.announce.responder_may_listen(p) if self.announce
                                         else (self.air is None or self.air.open(self.me, p)))
        return lk

    def _ofdm_rx(self, src: int, data: bytes) -> None:
        self.rx_count += 1
        self.log(f"ofdm rx {len(data)}B from {src}")   # symmetry with the CSS DATA path below,
                                                        # which already logs its own delivery
        if self.on_data:
            self.on_data(src, data)

    def _on_req(self, src: int, seq: int) -> None:
        # Defense in depth: _on_css already refuses to spawn this thread under ofdm_only, but
        # this runs detached (daemon thread, no caller to see an exception), so guard again here
        # rather than ever let an unhandled exception silently disappear to stderr.
        if self.mesh.phy_mode == "ofdm_only":
            self._ofdm_only_violation(f"_on_req({src}) called while phy_mode=ofdm_only -- refusing to ACK over CSS")
            return
        self.log(f"REQ from {src}")
        self.want_ofdm[src] = True                 # commanded by the peer over CSS
        with self._lock:
            busy = bool(self.links)
        if busy:
            # single radio: a link (ours or one we're already running to someone else) is live,
            # so a second concurrent OfdmLink would collide on the shared audio channel. Don't
            # even ACK -- src's own _request() retries with backoff, same as an unheard REQ.
            self.log(f"REQ from {src}: channel busy with an existing link -- not acking yet")
            return
        self._css_tx(self.css.send_ack, src, seq, tag=f"ack:{src}:{seq}", delay=self.csma.ack_delay())
        if not self._hs_gate(src, False):
            return
        with self._lock:
            old = self.links.pop((src, "rx"), None)
        if old:
            old.stop()
        lk = self._new_link(src, requester=False)
        with self._lock:
            self.links[(src, "rx")] = lk
        lk.start()

    def _request(self, dst: int) -> bool:
        for attempt in range(self.mesh.req_retries + 1):
            if self._stop.is_set():
                return False
            seq = self._next_seq()
            self.reqs += 1
            self._css_tx(self.css.send_type, dst, 6, seq, b"", f"OFDM_HS_REQ seq={seq}", tag=f"req:{dst}:{seq}")
            if self._wait_ack(dst, seq):
                return True
            self.log(f"REQ to {dst} unanswered (attempt {attempt + 1})")
        return False

    def _reap(self) -> None:
        while not self._stop.wait(5.0):
            now = time.monotonic()
            # with the announce round, a link trained in the BAT turns has nothing to carry until the data turns
            # begin: its idle clock starts there (a long BAT phase reaped links before their first frame)
            since = None
            idle_s = self.mesh.link_idle_s
            if self.announce is not None:
                since = self.announce.data_since
                if since is None:
                    since = now
                # a link only carries traffic in its node's data turn: idle until the next one is not idle at all
                idle_s = max(idle_s, self.announce.max_turn_gap_s())
            with self._lock:
                dead = [k for k, lk in self.links.items()
                        if lk.state in ("failed", "closed")
                        or (lk.up and now - max(lk.last_activity, since or 0.0) > idle_s)]
                gone = [self.links.pop(k) for k in dead]
            for lk in gone:
                self.log(f"closing idle/failed link to {lk.cfg.peer_id}")
                lk.stop()

    # -- blockout -------------------------------------------------------------

    def _blockout_budget_s(self, payload: bytes) -> float:
        """Default window length when a caller does not pass ``duration_s``: generous for an
        OFDM handshake plus a small-to-medium payload at a conservative acoustic rate.  A
        transfer that genuinely needs longer should pass ``duration_s`` explicitly -- this is a
        safety margin, not a measured budget (cf. ``meshlib.hs_airtime_budget`` for the real
        measured version used by the OTA reference driver)."""
        return min(0xFFFF * 0.1, max(90.0, 90.0 + 8.0 * len(payload) / 20.0))

    def send_blockout(self, peer: int, payload: bytes, *, duration_s: float | None = None,
                      announce_end: bool = True) -> bool:
        """One-way transfer to ``peer`` inside an exclusive blockout window: broadcast
        ``BLOCKOUT_REQ`` naming ``peer`` and a duration, so every OTHER node on the mesh mutes
        (TX-silent, still listening) for that long instead of running its usual CSMA/handshake
        schedule -- see ``Blockout``.  Then send exactly as ``send()`` would (OFDM if phy_mode
        allows, CSS fallback otherwise), since being the announced pair already exempts both of
        us from the mute (``_channel_busy``).

        Two-way is simply BOTH ends doing this: ``peer`` is marked a participant by overhearing
        our REQ (``_on_blockout_req``), so it is never muted either, and may call its own
        ``send(my_id, reply)`` -- or ``send_blockout`` again with ``announce_end=False`` -- any
        time before the window (or an explicit ``end_blockout``) ends, without re-announcing.

        ``announce_end=True`` (the default) broadcasts ``BLOCKOUT_END`` as soon as THIS leg is
        done -- "ack backlog or continue": every muted node's own outbox was left untouched, so
        it resumes its ordinary CSMA/CA the instant it hears END rather than waiting out the
        full window.  Pass False for the first leg of a two-way exchange and call
        ``end_blockout()`` (from either end) once BOTH legs have actually finished.

        Only available under ``phy_mode="ofdm_only"``: the BLOCKOUT_REQ/END broadcast needs
        OFDM's speed -- CSS cannot carry it -- so it is refused outright under
        ``css_only``/``css_then_ofdm`` rather than silently keying the (too-slow) CSS radio. A
        CSS-control-plane variant for an up ``css_then_ofdm`` link is future work, not this.
        """
        if self.mesh.phy_mode != "ofdm_only":
            msg = (f"send_blockout requires OFDM (CSS cannot carry it) -- refused under "
                  f"phy_mode={self.mesh.phy_mode!r}, not ofdm_only")
            self.log(msg)
            raise RuntimeError(msg)
        while time.monotonic() < self._hold_until and not self._stop.wait(1.0):
            pass                                          # someone else's handshake is on the air
        dur = float(duration_s) if duration_s is not None else self._blockout_budget_s(payload)
        if not (0 <= dur / BLOCKOUT_DUR_SCALE <= _BLOCKOUT_DUR_MAX):
            raise ValueError(f"blockout duration {dur}s out of range "
                             f"[0..{_BLOCKOUT_DUR_MAX * BLOCKOUT_DUR_SCALE}]")
        now = time.monotonic()
        self.blockout.request(peer, dur, now)
        H, _ = _ofdm_imports()
        band = self._ofdm_band()
        body = struct.pack(_BO_REQ_BODY, self.me, peer, int(round(dur / BLOCKOUT_DUR_SCALE)))
        audio = H.make_msg_audio(self.ofdm_cfg.fs, H.MSG_BLOCKOUT_REQ, 0, body, band)
        audio, _ = H.scale_hs_tx(audio, self.ofdm_cfg.amplitude, self.ofdm_cfg.fs, is_probe=False, band=band)
        audio = keyup(self.session, audio, self.ofdm_cfg.fs)
        self.log(f"blockout window with {peer}: {dur:.0f}s")
        self.session.play(audio, ident=False)
        pipe = self.session.pipe
        if pipe:
            pipe.wait_tx(len(audio) / self.ofdm_cfg.fs + 10)
        rx0 = self.rx_count
        ok = self.send(peer, payload)
        self.release_link(peer)        # free the channel now for peer's own reply, don't idle-hold it
        if announce_end:
            self.end_blockout()
        elif self.mesh.phy_mode == "ofdm_only":
            # Two-way: peer's reply is a fresh OFDM handshake with peer as initiator.  Under
            # ofdm_only the only thing that ever opens a responder is _apply_blockout_req
            # overhearing a REQ, and the reply leg sends none.  So we arm our listener here, or
            # peer's probes go unanswered (group run 20261106_034511 _b44a: timeout_i_wait_probe).
            self._ensure_rx_link(peer)
            threading.Thread(target=self._await_reply, args=(peer, rx0), daemon=True).start()
        return ok

    def _await_reply(self, peer: int, rx0: int) -> None:
        """Two-way window, announcer side: the replier ends the window once its leg is done
        (``end_blockout``).  If no reply ever arrives (our responder for ``peer`` fails or closes
        with nothing received), end it ourselves.  Otherwise every bystander would stay muted
        for the whole announced duration on behalf of a transfer that is already over."""
        while not self._stop.wait(1.0):
            if not self.blockout.active(time.monotonic()) or self.rx_count > rx0:
                return                    # peer's END cleared it, it expired, or the reply landed
            with self._lock:
                lk = self.links.get((peer, "rx"))
            if lk is None or lk.state in ("failed", "closed"):
                break
        if self.blockout.active(time.monotonic()) and self.rx_count == rx0:
            self.log(f"blockout: no reply from {peer} -- ending the window")
            self.end_blockout()

    def release_link(self, peer: int) -> None:
        """Tear down and drop any live link (either direction) to ``peer`` right now, instead of
        waiting out ``link_idle_s``. Single radio, one link at a time (see ``_on_req``): a
        one-shot transfer that keeps its link warm blocks the peer's own reply from ever being
        accepted, so both legs of a two-way ``send_blockout`` release their link as soon as they
        are done with it."""
        with self._lock:
            gone = [self.links.pop(k) for k in list(self.links) if k[0] == peer]
        for lk in gone:
            lk.stop()

    def _ensure_rx_link(self, peer: int) -> None:
        """ofdm_only: open a responder link for ``peer`` right now, instead of waiting for a CSS
        OFDM_HS_REQ to trigger one (``_on_req``) -- there is no CSS REQ either end is ever going
        to send in this mode.  The only caller is ``_apply_blockout_req``, reacting to a
        BLOCKOUT_REQ overheard (via ``_blockout_rx``) from the peer that announced the window."""
        key = (peer, "rx")
        with self._lock:
            if key in self.links:
                return
            lk = self._new_link(peer, requester=False)
            self.links[key] = lk
        lk.start()

    def end_blockout(self) -> None:
        """Broadcast ``BLOCKOUT_END`` and clear our own blockout state now, instead of letting
        the window run out -- the explicit "resume mesh" half of the feature.  Safe to call from
        either participant (or redundantly from both, for a two-way transfer); a node that is
        not currently in a window just sends a harmless, ignorable broadcast.

        Only available under ``phy_mode="ofdm_only"`` -- same reasoning as ``send_blockout``."""
        if self.mesh.phy_mode != "ofdm_only":
            msg = (f"end_blockout requires OFDM (CSS cannot carry it) -- refused under "
                  f"phy_mode={self.mesh.phy_mode!r}, not ofdm_only")
            self.log(msg)
            raise RuntimeError(msg)
        H, _ = _ofdm_imports()
        band = self._ofdm_band()
        body = struct.pack(_BO_END_BODY, self.me)
        audio = H.make_msg_audio(self.ofdm_cfg.fs, H.MSG_BLOCKOUT_END, 0, body, band)
        audio, _ = H.scale_hs_tx(audio, self.ofdm_cfg.amplitude, self.ofdm_cfg.fs, is_probe=False, band=band)
        audio = keyup(self.session, audio, self.ofdm_cfg.fs)
        self.session.play(audio, ident=False)
        pipe = self.session.pipe
        if pipe:
            pipe.wait_tx(len(audio) / self.ofdm_cfg.fs + 10)
        self.blockout.on_end(time.monotonic())

    # -- send ---------------------------------------------------------------

    def send(self, dst: int, payload: bytes) -> bool:
        """One-way transfer to ``dst``; True once the peer ACKed it (OFDM, or CSS when
        ``phy_mode="css_then_ofdm"`` allows the fallback -- see ``mesh_fallback_regress.py``
        for that state machine; ``phy_mode="ofdm_only"`` never takes it: a failed OFDM transfer
        is just a failed ``send``, not a silent CSS delivery."""
        while time.monotonic() < self._hold_until and not self._stop.wait(1.0):
            pass                                         # someone else's handshake is on the air
        if len(payload) > 250:
            raise ValueError("mesh payload over 250 bytes")
        ofdm_only = self.mesh.phy_mode == "ofdm_only"
        # ofdm_only is unconditional: it must attempt OFDM (regardless of want_ofdm/links state,
        # which exist only to steer the css_then_ofdm upgrade/downgrade dance) and must NEVER reach
        # _send_css below on any exit from this branch -- a data payload going out over CSS while
        # ofdm_only is set is exactly the silent-fallback bug this mode exists to rule out.
        if ofdm_only or (self.mesh.phy_mode == "css_then_ofdm" and
                         (self.want_ofdm.get(dst, True) or (dst, "tx") in self.links)):
            if self._send_ofdm(dst, payload):
                self.tx_count += 1
                return True
            if ofdm_only:
                self.log(f"OFDM to {dst} failed -- ofdm_only, not falling back to CSS")
                return False
            self.css_fallbacks += 1
            self.want_ofdm[dst] = False                # no timed re-upgrade: a peer REQ or the detector re-arms it
            self.log(f"OFDM to {dst} failed -- falling back to CSS")
        ok = self._send_css(dst, payload)
        if ok:
            self.tx_count += 1
        return ok

    def _open_tx_link(self, dst: int):
        """The up tx link to ``dst``, handshaking it first if needed (None on failure).  Called from a send and
        from this node's BAT turn; the gate is passed BEFORE the lock so a send waiting for its turn cannot
        block the announce thread that opens that turn."""
        key = (dst, "tx")
        with self._lock:
            lk = self.links.get(key)
        if lk is not None and lk.up:
            return lk
        def _up() -> bool:
            with self._lock:
                k = self.links.get(key)
            return k is not None and k.up

        if not self._hs_gate(dst, True, ready=_up):
            return None
        with self._tx_open:
            with self._lock:
                lk = self.links.get(key)
            if lk is not None and lk.up:
                return lk
            if lk is not None:
                lk.stop()
                with self._lock:
                    self.links.pop(key, None)
            if self.mesh.phy_mode != "ofdm_only":
                # css_only/css_then_ofdm: negotiate the upgrade over CSS first -- dst does not
                # otherwise know to bring up a responder link for us.
                if not self._request(dst):
                    return None
                self.csma.drop(f"req:{dst}:")          # a still-queued REQ is pointless now
            lk = self._new_link(dst, requester=True)
            with self._lock:
                self.links[key] = lk
            lk.start()
            if not lk.wait_settled(lk.cfg.hs_timeout_s):
                self.log(f"handshake with {dst} failed: {lk.fail_reason}")
                lk.stop()
                with self._lock:
                    self.links.pop(key, None)
                return None
            if self.announce:
                self.announce.note_own_go()
            return lk

    def _send_ofdm(self, dst: int, payload: bytes) -> bool:
        key = (dst, "tx")
        if self.announce and not self.announce.wait_data_window(self.mesh.announce_cycle_s + 120.0):
            self.log("ofdm send: no data window (round / BAT turns still running)")
            return False
        lk = self._open_tx_link(dst)
        if lk is None:
            return False
        try:
            ok = lk.send(payload)
        except Exception as e:
            self.log(f"ofdm send: {e}")
            ok = False
        if not ok:
            lk.stop()
            with self._lock:
                self.links.pop(key, None)
        return ok

    def _send_css(self, dst: int, payload: bytes) -> bool:
        # Hard gate, independent of every caller above: a data payload must never reach the CSS
        # encoder while ofdm_only is set, no matter how a future call site gets here.
        if self.mesh.phy_mode == "ofdm_only":
            raise RuntimeError(self._ofdm_only_violation(
                "_send_css called for a data payload while phy_mode=ofdm_only"))
        seq = self._next_seq()
        for attempt in range(self.mesh.data_retries + 1):
            if self._stop.is_set():
                return False
            if attempt:
                now = time.monotonic()
                self.session._emit(LinkEvent("retry", self.me, dst, now, now, f"css attempt {attempt}"))
            self._css_tx(self.css.send_bytes, _frame(payload), seq, dst, tag=f"data:{dst}:{seq}:{attempt}")
            if self._wait_ack(dst, seq):
                return True
        now = time.monotonic()
        self.session._emit(LinkEvent("fail", self.me, dst, now, now, "css: no ACK"))
        return False
