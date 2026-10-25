# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_modem_daemon.py — continuous half-duplex modem daemon (transport-agnostic).

Turns the one-shot announce/ack firmware into a PERSISTENT node: it always
listens, periodically re-announces on a drift-driven cadence, and arbitrates the
channel with CSMA/CA + ID-staggered backoff.  Like ModemNode it touches no audio
or sockets — a driver (the offline daemon sim, or a live mic+speaker role script)
feeds it events and performs the transmissions it releases.

Driver contract
---------------
  daemon.handle_frame(frame, now)   # a decoded frame arrived (from decode's
                                    # on_frame hook): updates drift, runs policy,
                                    # may enqueue ACK/reply packets.
  raw_tag = daemon.poll_tx(now, channel_busy)
                                    # CSMA/CA: returns (raw, tag) to transmit NOW
                                    # or None.  Only releases a packet when the
                                    # channel is idle AND the ID-staggered backoff
                                    # has elapsed.  Also enqueues the periodic
                                    # re-chirp beacon when due.
  daemon.on_tx(tag, at_sample)      # driver reports it actually transmitted (fixes
                                    # ranging t0 / lets seq persist).

Medium access (CSMA/CA): carrier-sense first (a 46 s chirp is trivially audible,
so collisions are rare); when idle and something is queued, wait
backoff = id*slot_stagger + U(0,jitter), re-sense, then transmit.  The ID term
staggers nodes deterministically; the jitter breaks ties and lets unknown nodes
join without a schedule.

Re-chirp cadence: derived from measured relative clock drift.  css_decoder fits a
per-reception symbol rate (grid_sps); (grid_sps/nominal - 1) is the peer's clock
error.  A peer's freewheeled grid of US drifts out of the cheap wake-word
re-acquisition window after ~ window / (|error|*fs) seconds, so we re-chirp before
then.  Faster drift -> chirp more often; near-zero drift -> back off to the max.
"""
from __future__ import annotations

import random

import css_modem as cm


class ModemDaemon:
    def __init__(self, fw, *,
                 cw_s: float = 4.0,
                 id_slot_s: float = 0.1,
                 stagger_mod: int = 16,
                 rechirp_min_s: float = 20.0,
                 rechirp_max_s: float = 300.0,
                 freewheel_window_syms: float = 0.5,
                 first_beacon_stagger_s: float = 0.0,
                 max_plausible_ppm: float = 500.0,
                 drift_hist_len: int = 5,
                 ack_stagger_s: float = 4.0,
                 rng: random.Random | None = None):
        self.fw = fw
        self.node = fw.node
        self.id = fw.cfg["id"]
        phy = fw.cfg["phy"]
        self.fs = int(phy["fs"])
        gap = int(round(int(self.fs * phy["symbol_dur"]) * phy.get("symbol_gap_frac", 0.0)))
        self.sps_eff = int(self.fs * phy["symbol_dur"]) + gap

        # CSMA/CA backoff: a RANDOM contention window (cw_s) dominates so no node
        # is systematically starved, plus a small deterministic per-ID phase
        # (id_slot_s, mod stagger_mod) that staggers otherwise-simultaneous
        # starts and breaks exact ties.  The countdown FREEZES while the channel
        # is busy (802.11-style) instead of restarting, so a node that has waited
        # keeps its priority.
        self.cw_s = cw_s
        self.id_slot_s = id_slot_s
        self.stagger_mod = stagger_mod
        self.rechirp_min_s = rechirp_min_s
        self.rechirp_max_s = rechirp_max_s
        self.freewheel_window = freewheel_window_syms * self.sps_eff
        self.max_plausible_ppm = max_plausible_ppm
        self.drift_hist_len = drift_hist_len
        self._drift_hist: dict[int, list] = {}
        self.ack_stagger_s = ack_stagger_s
        self._rng = rng or random.Random(0xC0FFEE ^ self.id)

        # outbox entries are (raw, tag, ready_at): a packet is not eligible to
        # transmit before ready_at.  Used to ID-stagger ACK responses so two
        # nodes that discover each other at the same moment don't fire their ACKs
        # simultaneously (a sub-carrier-sense-latency collision).
        self.outbox: list[tuple[bytes, str, float]] = []
        self._backoff_remaining: float | None = None
        self._last_now: float | None = None
        self.rechirp_interval_s = rechirp_max_s      # start conservative
        # First announce: an ID-proportional stagger (so nodes don't start their
        # long first beacons simultaneously and collide before carrier sense can
        # catch them) plus a small random offset.  Wraps mod stagger_mod so high
        # IDs aren't pushed absurdly late.
        self._next_rechirp_s = (self.id % self.stagger_mod) * first_beacon_stagger_s \
            + self._rng.uniform(0.0, cw_s)
        self.peers: dict[int, dict] = {}
        self.events: list[dict] = []                  # discovered/ack/deliver/ranged

    # ── queueing ─────────────────────────────────────────────────────────────
    def enqueue(self, raw: bytes, tag: str, ready_at: float = 0.0) -> bool:
        """Queue a packet.  False (and a no-op) if an identical one is waiting.

        Tags are ``<kind>:<dst>:<seq>``, so an equal tag IS the same packet.
        Dedupe is against the QUEUE ONLY, never history: re-ACKing a peer's
        retransmitted DATA is legitimate and still works, because by then the
        first ACK has aired and left the queue.  Two identical ACKs sitting in
        the queue TOGETHER are pure duplication, and one CSS packet costs ~33s
        of air at these params -- observed 20260813_040605 with
        ``outbox=['ack:2:0', 'ack:2:0', 'data:2:2']``, where an OFDM data frame
        that needed 1.7s on air was head-of-line blocked behind ~67s of
        duplicate CSS control traffic.
        """
        if any(t == tag for _, t, _ in self.outbox):
            return False
        self.outbox.append((bytes(raw), tag, float(ready_at)))
        return True

    def drop(self, prefix: str) -> list[str]:
        """Remove queued packets whose tag starts with `prefix`; return them.

        For packets that a state change has made OBSOLETE while they sat in the
        queue.  The motivating case is an OFDM_HS_REQ still queued when the
        handshake it asks for has already succeeded: airing it is definitionally
        pointless, and at CSS rates it is ~33s of dead air ahead of the data the
        new link exists to carry.
        """
        dropped = [t for _, t, _ in self.outbox if t.startswith(prefix)]
        if dropped:
            self.outbox = [e for e in self.outbox
                           if not e[1].startswith(prefix)]
        return dropped

    def _ack_delay(self) -> float:
        """Per-node ACK hold so two peers never respond at the same instant."""
        return (self.id % self.stagger_mod) * self.ack_stagger_s

    def make_beacon(self):
        raw, tag = self.node.make_beacon()
        return raw, tag

    # ── receive ──────────────────────────────────────────────────────────────
    def handle_frame(self, frame: dict, now: float):
        if "bytes_hex" not in frame:
            return
        raw = bytes.fromhex(frame["bytes_hex"])
        at = int(frame.get("at_sample", 0))
        try:
            p = cm.unpack_packet(raw)
        except cm.ModemError:
            return
        if p["src"] != self.id:
            self._update_drift(p["src"], frame.get("grid_sps"), now)
        # Protocol actions from the node, mapped to RAW packets we enqueue for TX
        # (the driver renders them to chirps when the FSM releases them).
        for a in self.node.on_rx(raw, at):
            k = a["kind"]
            if k == "send":                       # node auto-ACK / range reply
                self.enqueue(a["build"](at), a["tag"],
                             ready_at=now + self._ack_delay())
            elif k == "discovered":               # announce -> neighbour ACK policy
                self.events.append(a)
                r, tag = self.node.make_ack(a["peer"], a.get("seq", 0))
                self.enqueue(r, tag, ready_at=now + self._ack_delay())
            else:                                 # ack / deliver / ranged
                self.events.append(a)

    def _update_drift(self, peer: int, grid_sps, now: float):
        rec = self.peers.setdefault(peer, {"drift_ppm": 0.0, "last_rx_s": None})
        rec["last_rx_s"] = now
        if grid_sps:
            ppm = (float(grid_sps) / self.sps_eff - 1.0) * 1e6
            # A marginal decode can mis-fit grid_sps and report a wildly
            # implausible drift (seen: 88000 ppm on hardware).  Real audio clocks
            # are < ~200 ppm apart; reject outliers and MEDIAN-smooth the rest so
            # one bad reading can't collapse the re-chirp interval to its floor
            # (which saturates the channel and causes the very collisions this
            # cadence is meant to avoid).
            if abs(ppm) <= self.max_plausible_ppm:
                hist = self._drift_hist.setdefault(peer, [])
                hist.append(ppm)
                if len(hist) > self.drift_hist_len:
                    hist.pop(0)
                s = sorted(hist)
                rec["drift_ppm"] = s[len(s) // 2]
        self._recompute_rechirp()

    def _recompute_rechirp(self):
        # Worst-case peer drift sets the cadence: chirp before the fastest-drifting
        # peer's freewheeled grid slides out of the re-acquisition window.
        worst = max((abs(r["drift_ppm"]) for r in self.peers.values()), default=0.0)
        if worst <= 1e-6:
            self.rechirp_interval_s = self.rechirp_max_s
        else:
            secs = self.freewheel_window / (worst * 1e-6 * self.fs)
            self.rechirp_interval_s = max(self.rechirp_min_s,
                                          min(self.rechirp_max_s, secs))

    def _draw_backoff(self) -> float:
        return (self._rng.uniform(0.0, self.cw_s)
                + (self.id % self.stagger_mod) * self.id_slot_s)

    # ── transmit arbitration (CSMA/CA, freeze-countdown) ─────────────────────
    def poll_tx(self, now: float, channel_busy: bool):
        dt = (now - self._last_now) if self._last_now is not None else 0.0
        self._last_now = now

        # Periodic re-chirp: queue a beacon when due (only if none already
        # queued).  The next deadline is jittered (0.7-1.0x) so nodes' beacon
        # bursts don't phase-align into repeated collisions.
        if now >= self._next_rechirp_s:
            self._next_rechirp_s = now + self.rechirp_interval_s \
                * self._rng.uniform(0.7, 1.0)
            if not any(t.startswith("beacon:") for _, t, _ in self.outbox):
                self.enqueue(*self.make_beacon())

        # First outbox entry whose ID-stagger hold has elapsed.
        ready_i = next((i for i, (_, _, ra) in enumerate(self.outbox)
                        if ra <= now), None)
        if ready_i is None:
            self._backoff_remaining = None
            return None
        # Non-persistent CSMA: while the channel is busy, DROP any pending
        # countdown and re-draw a fresh backoff once it clears.  This decorrelates
        # nodes that were all waiting on the same transmission (a frozen-countdown
        # resume would race them straight into another collision).
        if channel_busy:
            self._backoff_remaining = None
            return None
        if self._backoff_remaining is None:       # channel idle: arm fresh backoff
            self._backoff_remaining = self._draw_backoff()
            return None
        self._backoff_remaining -= dt
        if self._backoff_remaining <= 0.0:
            self._backoff_remaining = None
            raw, tag, _ = self.outbox.pop(ready_i)
            return (raw, tag)
        return None

    def on_tx(self, tag: str, at_sample: int):
        self.node.on_tx(tag, at_sample)
        try:
            self.fw.save()                        # persist advanced seq
        except Exception:
            pass

    def drain_events(self) -> list[dict]:
        e, self.events = self.events, []
        return e
