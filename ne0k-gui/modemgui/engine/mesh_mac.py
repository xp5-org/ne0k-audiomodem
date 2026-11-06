# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Medium access for the mesh, ported from the reference ``css_modem_daemon.ModemDaemon`` and the
airtime scheduling in ``meshmodem_node``.  Pure logic, no audio and no threads, so it is testable
with a fake clock.

* ``Csma``       outbox with per-entry ``ready_at`` (ID-staggered ACK hold), non-persistent CSMA/CA
                 (a busy channel drops the countdown; on idle a fresh backoff
                 ``U(0, cw_s) + (id % 16) * id_slot_s`` is drawn), tag de-duplication and
                 ``drop(prefix)`` for packets a state change has made obsolete.
* ``Rechirp``    beacon cadence from the measured clock drift of the peers: chirp before the fastest
                 drifting peer's freewheeled symbol grid slides out of the re-acquisition window,
                 clamped to [min, max], next deadline jittered 0.7-1.0x so beacons do not phase-align.
* ``Airtime``    per-link exclusive wall-clock windows so two handshakes never share the channel;
                 both ends derive the same window from the node pair.
* ``Ambient``    running estimate of the quiet-channel RMS for the passive OFDM detector.
"""
from __future__ import annotations

import collections
import random
import statistics
import time

STAGGER_MOD = 16


class Csma:
    def __init__(self, node_id: int, *, cw_s: float = 4.0, id_slot_s: float = 0.1,
                 ack_stagger_s: float = 0.5, rng: random.Random | None = None) -> None:
        self.id, self.cw_s, self.id_slot_s, self.ack_stagger_s = node_id, cw_s, id_slot_s, ack_stagger_s
        self.rng = rng or random.Random(0xC0FFEE ^ node_id)
        self.outbox: list[tuple[object, str, float]] = []          # (job, tag, ready_at)
        self._backoff: float | None = None
        self._last: float | None = None

    def ack_delay(self) -> float:
        """Per-node ACK hold so two peers never answer at the same instant."""
        return (self.id % STAGGER_MOD) * self.ack_stagger_s

    @staticmethod
    def max_ack_delay(ack_stagger_s: float) -> float:
        return (STAGGER_MOD - 1) * ack_stagger_s

    def enqueue(self, job, tag: str, ready_at: float = 0.0) -> bool:
        """False (and a no-op) when an identical tag is already waiting: one CSS packet costs many
        seconds of air, a queued duplicate is pure head-of-line blocking."""
        if any(t == tag for _, t, _ in self.outbox):
            return False
        self.outbox.append((job, tag, float(ready_at)))
        return True

    def drop(self, prefix: str) -> list[str]:
        dropped = [t for _, t, _ in self.outbox if t.startswith(prefix)]
        if dropped:
            self.outbox = [e for e in self.outbox if not e[1].startswith(prefix)]
        return dropped

    def pending(self, prefix: str = "") -> bool:
        return any(t.startswith(prefix) for _, t, _ in self.outbox)

    def _draw(self) -> float:
        return self.rng.uniform(0.0, self.cw_s) + (self.id % STAGGER_MOD) * self.id_slot_s

    def poll(self, now: float, channel_busy: bool):
        """The (job, tag) to transmit now, or None."""
        dt = (now - self._last) if self._last is not None else 0.0
        self._last = now
        i = next((k for k, (_, _, ra) in enumerate(self.outbox) if ra <= now), None)
        if i is None:
            self._backoff = None
            return None
        if channel_busy:             # non-persistent: nodes that waited on the same burst re-draw apart
            self._backoff = None
            return None
        if self._backoff is None:
            self._backoff = self._draw()
            return None
        self._backoff -= dt
        if self._backoff <= 0.0:
            self._backoff = None
            job, tag, _ = self.outbox.pop(i)
            return job, tag
        return None


class Rechirp:
    def __init__(self, node_id: int, fs: int, sps_eff: float, *, min_s: float, max_s: float,
                 first_s: float, first_stagger_s: float = 1.0, window_syms: float = 0.5,
                 max_ppm: float = 500.0, hist_len: int = 5, rng: random.Random | None = None) -> None:
        self.fs, self.sps_eff = fs, sps_eff
        self.min_s, self.max_s = sorted((min_s, max_s))
        self.window = window_syms * sps_eff
        self.max_ppm, self.hist_len = max_ppm, hist_len
        self.rng = rng or random.Random(0xBEAC04 ^ node_id)
        self.interval_s = self.max_s                       # start conservative
        self.drift_ppm: dict[int, float] = {}
        self._hist: dict[int, collections.deque] = {}
        # ID-proportional first beacon + a random listen window, so nodes started together do not
        # begin their long first beacons simultaneously
        self.next_at = (node_id % STAGGER_MOD) * first_stagger_s + self.rng.uniform(0.0, first_s)

    def heard(self, peer: int, grid_sps: float | None) -> None:
        if not grid_sps:
            return
        ppm = (float(grid_sps) / self.sps_eff - 1.0) * 1e6
        if abs(ppm) > self.max_ppm:            # a marginal decode mis-fits the grid: reject the outlier
            return
        h = self._hist.setdefault(peer, collections.deque(maxlen=self.hist_len))
        h.append(ppm)
        self.drift_ppm[peer] = statistics.median(h)    # median-smooth so one bad read cannot floor the interval
        worst = max(abs(v) for v in self.drift_ppm.values())
        self.interval_s = (self.max_s if worst <= 1e-6 else
                           max(self.min_s, min(self.max_s, self.window / (worst * 1e-6 * self.fs))))

    def due(self, now: float) -> bool:
        if now < self.next_at:
            return False
        self.next_at = now + self.interval_s * self.rng.uniform(0.7, 1.0)
        return True


class Airtime:
    """Exclusive wall-clock window per link.  The window is a function of the node pair, so both
    ends agree with no signalling (they only need roughly the same wall clock, NTP is plenty)."""

    def __init__(self, *, window_s: float = 150.0, slots: int = 4, guard_s: float = 45.0,
                 init_offset_s: float = 6.0, clock=time.time) -> None:
        self.window_s, self.slots, self.guard_s, self.init_offset_s = window_s, max(1, slots), guard_s, init_offset_s
        self.cycle = self.window_s * self.slots
        self.clock = clock

    def slot(self, a: int, b: int) -> int:
        lo, hi = sorted((a, b))
        return (lo * STAGGER_MOD + hi) % self.slots

    def _into(self, a: int, b: int) -> float:
        return (self.clock() % self.cycle) - self.slot(a, b) * self.window_s

    def open(self, a: int, b: int) -> bool:
        """Data and handshake traffic of the a-b link may use the air."""
        return 0 <= self._into(a, b) < self.window_s

    def hs_open(self, a: int, b: int, initiator: bool) -> bool:
        """A handshake may START only early in the window (it must finish before the next link's
        window opens); the initiator waits so the responder is already listening."""
        into = self._into(a, b)
        if not 0 <= into < max(1.0, self.window_s - self.guard_s):
            return False
        return into >= self.init_offset_s if initiator else True

    def wait_s(self, a: int, b: int, initiator: bool) -> float:
        """Seconds until hs_open() next turns true (0 when it already is)."""
        if self.hs_open(a, b, initiator):
            return 0.0
        start = self.slot(a, b) * self.window_s + (self.init_offset_s if initiator else 0.0)
        return (start - self.clock()) % self.cycle


class Blockout:
    """Mesh-wide exclusive quiet window (``BLOCKOUT_REQ``/``BLOCKOUT_END``, see
    ``css_modem``).  Two nodes get the whole channel to themselves for one
    one-way or two-way transfer; everyone else goes TX-silent (listen-only)
    for the announced duration instead of running its usual CSMA/handshake
    schedule, so the pair cannot be trampled.  Resuming never drops anything:
    a bystander's own outbox/backlog just sat still, it was not touched, so
    normal CSMA/CA picks it straight back up once the window closes.

    Pure logic, no audio and no threads -- same shape as ``Airtime``/``Csma``,
    testable with a fake clock.  One instance per node.
    """

    def __init__(self, node_id: int) -> None:
        self.node_id = int(node_id)
        self.until: float | None = None     # wall-clock deadline (participant or mute)
        self.participant = False            # True: I am one of the two transfer endpoints
        self.peer: int | None = None        # the OTHER endpoint, when participant

    def request(self, peer: int, duration_s: float, now: float) -> None:
        """I am announcing a window with ``peer`` -- mark myself a participant
        (never muted by my own announce) and arm the deadline that lets a late
        ``BLOCKOUT_END`` still self-expire if it is somehow lost."""
        self.participant = True
        self.peer = int(peer)
        self.until = now + float(duration_s)

    def on_req(self, announcer: int, peer_b: int, duration_s: float, now: float) -> None:
        """A ``BLOCKOUT_REQ`` was heard from ``announcer`` (naming ``peer_b`` as
        the other participant).  Join quietly if it is about me; otherwise mute
        for the announced duration."""
        mine = self.node_id in (announcer, peer_b)
        if mine:
            self.participant = True
            self.peer = peer_b if announcer == self.node_id else announcer
        self.until = now + float(duration_s)

    def on_end(self, now: float) -> None:
        """``BLOCKOUT_END`` heard (from either participant): resume now, whether
        I was muted or the one running the transfer."""
        self.until = None
        self.participant = False
        self.peer = None

    def muted(self, now: float) -> bool:
        """True while a BYSTANDER must stay TX-silent.  Self-expires (the safety
        net for a lost ``BLOCKOUT_END``) and never mutes a participant -- the
        two endpoints transmit throughout their own window."""
        if self.participant or self.until is None:
            return False
        if now >= self.until:
            self.until = None
            return False
        return True

    def active(self, now: float) -> bool:
        """A window is in force right now (participant or muted bystander)."""
        return self.participant or self.muted(now)


class Ambient:
    """Quiet-channel RMS: the low percentile of recent 50 ms block levels."""

    def __init__(self, n: int = 400, pct: float = 0.2) -> None:
        self.levels: collections.deque = collections.deque(maxlen=n)
        self.pct = pct

    def add(self, rms: float) -> None:
        self.levels.append(rms)

    @property
    def rms(self) -> float | None:
        if len(self.levels) < 20:
            return None
        s = sorted(self.levels)
        return s[int(self.pct * (len(s) - 1))]
