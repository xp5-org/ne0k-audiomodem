# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
meshsim_mac.py — Tier 1: MAC-level 15-node mesh simulation.

A refactor of css_modem_daemon_sim.DaemonSim that swaps the old BOOLEAN channel
for the physical Channel2D (meshchannel.py) and writes the shared event stream
(meshevents.py).  Still single-core, faster-than-real-time, and DSP-free: a
packet's on-air time is an abstract airtime_s (the MAC is airtime-agnostic), and
delivery is decided by an SINR/capture rule instead of all-or-nothing collision.

What changed vs DaemonSim
  * positions are 2-D (x, y) and every node has a noise PROFILE;
  * a reception decodes iff its SINR against the concurrently-audible
    interferers clears the capture margin — so a strong near signal can capture
    over a weak far one (real behaviour), and equal-power overlaps still collide;
  * carrier sense is power-gated (a node only senses a tx it can actually hear),
    which is what makes HIDDEN TERMINALS possible when nodes are spread out;
  * every tx / rx-ok / rx-fail / carrier-defer / discovered / ack / ranged event
    is emitted to an EventLog for the piano-roll renderer.

The 3-call daemon contract (handle_frame / poll_tx / on_tx) and the CSMA/CA +
drift-driven re-chirp logic are reused UNCHANGED from ModemDaemon.

Run:  python tests/meshsim_mac.py        (exit 0 = all scenario tests passed)
"""
from __future__ import annotations

import os
import random
import sys
import tempfile


# --- category-aware sys.path bootstrap (auto-inserted) ---
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_TESTS_ROOT = _HERE.parent if _HERE.name in ("CSS", "MESH", "OFDM", "QAM") else _HERE
_REPO_ROOT = _TESTS_ROOT.parent
for _p in (_TESTS_ROOT, _REPO_ROOT, _TESTS_ROOT / "CSS", _TESTS_ROOT / "MESH",
           _TESTS_ROOT / "OFDM", _TESTS_ROOT / "QAM", _TESTS_ROOT / "regression"):
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# --- end bootstrap ---

import css_modem as cm                                    # noqa: E402
from css_modem_fw import ModemFirmware, DEFAULT_PHY       # noqa: E402
from css_modem_daemon import ModemDaemon                  # noqa: E402
from meshchannel import Channel2D, DEFAULT_CAPTURE_MARGIN_DB  # noqa: E402
import meshevents as ev                                   # noqa: E402

DT = 0.1                    # sim tick (s)
DEFAULT_AIRTIME_S = 2.0     # abstract per-packet on-air time (MAC is agnostic)


class MacSim:
    def __init__(self, *, fs=48000, airtime_s=DEFAULT_AIRTIME_S, dt=DT,
                 capture_margin_db=DEFAULT_CAPTURE_MARGIN_DB,
                 carrier_sense_db=0.0, pathloss_exp=2.0, ref_dist_m=1.0,
                 eventlog: ev.EventLog | None = None, verbose=False):
        self.fs = int(fs)
        self.airtime_s = float(airtime_s)
        self.dt = float(dt)
        self.capture_margin_db = float(capture_margin_db)
        self.carrier_sense_db = float(carrier_sense_db)
        self.pathloss_exp = float(pathloss_exp)
        self.ref_dist_m = float(ref_dist_m)
        self.log = eventlog
        self.verbose = verbose

        self.members: dict[int, dict] = {}   # id -> {daemon,pos,profile,drift_ppm,tx_end}
        self.txs: list[dict] = []            # {id,src,raw,tag,start,end}
        self._delivered: set = set()         # (tx_id, rx_id)
        self._tx_seq = 0
        self._discovered: dict[int, set] = {}
        self._deferring: dict[int, bool] = {}
        # run-wide counters + ranging results, tracked whether or not an
        # EventLog is attached, so scenario checks don't depend on logging.
        self.stats = {"tx": 0, "rx_ok": 0, "rx_fail": 0, "collision": 0,
                      "low_sinr": 0, "deaf": 0, "discovered": 0}
        self.ranges: list[dict] = []
        self.t = 0.0
        self._chan: Channel2D | None = None
        self._has_schedules = False          # any node has a time-varying profile
        self._work = tempfile.mkdtemp(prefix="meshsim_mac_")

    # ── construction ─────────────────────────────────────────────────────────
    def add(self, node_id, pos=(0.0, 0.0), profile="clean", drift_ppm=0.0,
            profile_schedule=None, **daemon_kw):
        """profile_schedule: optional [(t_start_s, bucket), …] (sorted) that
        switches a node's RECEIVER noise bucket over sim time — e.g.
        [(0,"noisy"),(30,"half_noisy")] is 'noisy for 30 s, then half-noisy'.
        Models intermittent interference so re-beaconing / retransmit recovery
        can be exercised."""
        phy = dict(DEFAULT_PHY, fs=self.fs, symbol_dur=0.5, chunk_size=2)
        fw = ModemFirmware(os.path.join(self._work, f"n{node_id}.json"),
                           node_id=node_id, workdir=self._work,
                           phy_override=phy, bake_info=False).boot()
        d = ModemDaemon(fw, rng=random.Random(1000 + node_id), **daemon_kw)
        self.members[node_id] = {"daemon": d, "pos": (float(pos[0]), float(pos[1])),
                                 "profile": str(profile),
                                 "profile_schedule": list(profile_schedule or []),
                                 "drift_ppm": float(drift_ppm), "tx_end": -1.0}
        if profile_schedule:
            self._has_schedules = True
        self._discovered.setdefault(node_id, set())
        self._deferring.setdefault(node_id, False)
        self._rebuild_channel()
        if self.log:
            self.log.emit(ev.KIND_BOOT, t=self.t, node=node_id,
                          t_local=self._t_local(node_id, self.t),
                          detail={"profile": str(profile),
                                  "pos": [float(pos[0]), float(pos[1])],
                                  "drift_ppm": float(drift_ppm)})
        return d

    def _rebuild_channel(self):
        pos = {nid: m["pos"] for nid, m in self.members.items()}
        prof = {nid: m["profile"] for nid, m in self.members.items()}
        self._chan = Channel2D(pos, prof, ref_dist_m=self.ref_dist_m,
                               pathloss_exp=self.pathloss_exp,
                               capture_margin_db=self.capture_margin_db,
                               fs=self.fs)

    # ── clocks ───────────────────────────────────────────────────────────────
    def _t_local(self, node_id, t):
        """Node's own clock: the abstract sim has no separate counter, so model
        drift directly so the divergence is visible in the piano roll."""
        return t * (1.0 + self.members[node_id]["drift_ppm"] * 1e-6)

    # ── channel helpers ──────────────────────────────────────────────────────
    def _prop(self, a, b):
        return self._chan.prop_delay(a, b)

    def _rx_window(self, tx, rx_id):
        p = self._prop(tx["src"], rx_id)
        return tx["start"] + p, tx["end"] + p

    def _overlaps(self, a0, a1, b0, b1):
        return a0 < b1 and b0 < a1

    def _busy(self, rx_id, t):
        """Power-gated carrier sense: rx senses a tx only if it's audible above
        the sense threshold (enables hidden terminals for spread geometries)."""
        for tx in self.txs:
            if tx["src"] == rx_id:
                continue
            s, e = self._rx_window(tx, rx_id)
            if s <= t < e and self._chan.link_snr_db(tx["src"], rx_id) >= self.carrier_sense_db:
                return True
        return False

    def _grid_sps(self, src, rx):
        sps_eff = self.members[rx]["daemon"].sps_eff
        rel = (self.members[rx]["drift_ppm"] - self.members[src]["drift_ppm"]) * 1e-6
        return sps_eff * (1.0 + rel)

    # ── main loop ────────────────────────────────────────────────────────────
    def step(self):
        t = self.t
        if self._has_schedules:
            self._apply_profile_schedules()
        self._deliver(t)
        self._transmit(t)
        self._prune(t)
        self.t += self.dt

    def _apply_profile_schedules(self):
        """Switch each scheduled node's channel noise bucket to whatever its
        schedule says is active at the current sim time (e.g. noisy for 30 s,
        then half-noisy)."""
        for nid, m in self.members.items():
            sched = m.get("profile_schedule")
            if not sched:
                continue
            prof = m["profile"]
            for t0, p in sched:
                if self.t >= t0:
                    prof = p
                else:
                    break
            if self._chan.profiles[nid] != prof:
                self._chan.profiles[nid] = prof

    def _deliver(self, t):
        for tx in self.txs:
            for rx_id, m in self.members.items():
                if rx_id == tx["src"] or (tx["id"], rx_id) in self._delivered:
                    continue
                rx_start, rx_end = self._rx_window(tx, rx_id)
                if rx_end > t:
                    continue                      # reception not finished yet
                self._delivered.add((tx["id"], rx_id))

                # never even audible → not "heard", emit nothing
                if self._chan.link_snr_db(tx["src"], rx_id) < self.carrier_sense_db:
                    continue

                # half-duplex: receiver was transmitting during any of this window
                deaf = any(o["src"] == rx_id
                           and self._overlaps(rx_start, rx_end, o["start"], o["end"])
                           for o in self.txs)
                # interferers = other txs whose reception window at rx overlaps
                interferers = [o["src"] for o in self.txs
                               if o["id"] != tx["id"] and o["src"] != rx_id
                               and self._overlaps(rx_start, rx_end,
                                                  *self._rx_window(o, rx_id))]
                ptype = ev.ptype_of(tx["raw"])
                pseq = self._pseq(tx["raw"])
                tl = self._t_local(rx_id, rx_end)

                if deaf:
                    self.stats["rx_fail"] += 1
                    self.stats["deaf"] += 1
                    self._emit(ev.KIND_RX_FAIL, t=rx_end, node=rx_id, t_local=tl,
                               peer=tx["src"], ptype=ptype, pseq=pseq,
                               reason=ev.REASON_DEAF)
                    continue
                sinr = self._chan.sinr_db(tx["src"], rx_id, interferers)
                if sinr >= self.capture_margin_db:
                    self.stats["rx_ok"] += 1
                    m["daemon"].handle_frame(
                        {"bytes_hex": tx["raw"].hex(),
                         "at_sample": int(rx_end * self.fs),
                         "grid_sps": self._grid_sps(tx["src"], rx_id)}, t)
                    self._emit(ev.KIND_RX_OK, t=rx_end, node=rx_id, t_local=tl,
                               peer=tx["src"], ptype=ptype, pseq=pseq,
                               snr_db=round(sinr, 2))
                    self._drain(rx_id, rx_end)
                else:
                    reason = ev.REASON_COLLISION if interferers else ev.REASON_LOW_SINR
                    self.stats["rx_fail"] += 1
                    self.stats[reason] += 1
                    self._emit(ev.KIND_RX_FAIL, t=rx_end, node=rx_id, t_local=tl,
                               peer=tx["src"], ptype=ptype, pseq=pseq,
                               snr_db=round(sinr, 2), reason=reason)
                    if self.verbose and interferers:
                        print(f"  t={t:5.1f} COLLISION at {rx_id} from {tx['src']}")

    def _transmit(self, t):
        for node_id, m in self.members.items():
            if t < m["tx_end"]:                   # still transmitting (half-duplex)
                continue
            busy = self._busy(node_id, t)
            d = m["daemon"]
            # edge-triggered carrier-defer event: something ready but channel busy
            ready = any(ra <= t for _, _, ra in d.outbox)
            if busy and ready and not self._deferring[node_id]:
                self._emit(ev.KIND_CARRIER_DEFER, t=t, node=node_id,
                           t_local=self._t_local(node_id, t))
                self._deferring[node_id] = True
            rt = d.poll_tx(t, busy)
            if rt is None:
                continue
            self._deferring[node_id] = False
            raw, tag = rt
            d.on_tx(tag, int((t + self.airtime_s) * self.fs))
            m["tx_end"] = t + self.airtime_s
            self.txs.append({"id": self._tx_seq, "src": node_id,
                             "raw": bytes(raw), "tag": tag,
                             "start": t, "end": t + self.airtime_s})
            self.stats["tx"] += 1
            self._emit(ev.KIND_TX_START, t=t, node=node_id,
                       t_local=self._t_local(node_id, t),
                       peer=self._dst(raw), ptype=ev.ptype_of(raw),
                       pseq=self._pseq(raw), dur=self.airtime_s)
            self._tx_seq += 1
            if self.verbose:
                p = cm.unpack_packet(raw)
                print(f"  t={t:5.1f} TX  n{node_id} {p['type_name']} dst={p['dst']}")

    def _prune(self, t):
        horizon = t - self.airtime_s - self._max_prop() - self.dt
        if len(self.txs) > 64:
            mp = self._max_prop()
            self.txs = [tx for tx in self.txs if tx["end"] + mp >= horizon]

    def _max_prop(self):
        ids = list(self.members)
        return max((self._prop(a, b) for a in ids for b in ids if a != b),
                   default=0.0)

    # ── event / packet helpers ───────────────────────────────────────────────
    def _emit(self, kind, **kw):
        if self.log:
            self.log.emit(kind, **kw)

    def _drain(self, node_id, t):
        d = self.members[node_id]["daemon"]
        for a in d.drain_events():
            k = a.get("kind")
            if k == "discovered":
                self._discovered[node_id].add(a.get("peer"))
                self.stats["discovered"] += 1
            elif k == "ranged":
                self.ranges.append({"node": node_id, "peer": a.get("peer"),
                                    "distance_m": a.get("distance_m")})
            if self.log:
                ev.from_daemon_event(self.log, a, t=t, node=node_id,
                                     t_local=self._t_local(node_id, t))

    @staticmethod
    def _dst(raw):
        try:
            return cm.unpack_packet(raw)["dst"]
        except cm.ModemError:
            return None

    @staticmethod
    def _pseq(raw):
        try:
            return cm.unpack_packet(raw)["seq"]
        except cm.ModemError:
            return None

    def initiate_range(self, src, dst):
        """Have node `src` begin a two-way range to `dst`.

        The daemon has no autonomous ranging trigger (it only auto-beacons and
        auto-ACKs), so a scenario injects the RANGE_REQ here; the reply then
        flows back through the normal handle_frame → on_rx → 'ranged' path.
        """
        d = self.members[src]["daemon"]
        raw, tag = d.node.start_range(dst)
        d.enqueue(raw, tag)

    # ── run / queries ────────────────────────────────────────────────────────
    def run(self, until_s):
        while self.t < until_s:
            self.step()
        # flush any events the daemons produced but weren't drained on an rx tick
        for nid in self.members:
            self._drain(nid, self.t)

    def discovered_by(self, node_id):
        # combine sim-tracked discoveries with any still in the daemon's buffer
        got = set(self._discovered.get(node_id, set()))
        got |= {e["peer"] for e in self.members[node_id]["daemon"].events
                if e["kind"] == "discovered"}
        return got


# ── ported DaemonSim scenario tests (behaviour-preserving) ───────────────────

def _mac_kw():
    return dict(cw_s=3.0, id_slot_s=0.1, rechirp_min_s=4.0, rechirp_max_s=20.0)


def test_mutual_discovery():
    """5 nodes beaconing all discover each other despite the shared channel."""
    sim = MacSim()
    ids = [1, 2, 3, 4, 5]
    for k, nid in enumerate(ids):
        sim.add(nid, pos=(k * 1.0, 0.0), **_mac_kw())
    sim.run(90.0)
    for nid in ids:
        got, want = sim.discovered_by(nid), set(ids) - {nid}
        if got != want:
            return f"FAIL: node {nid} discovered {sorted(got)}, want {sorted(want)}"
    return True


def test_heavy_contention():
    """8 nodes reach full mutual discovery under a busy-but-carriable load."""
    sim = MacSim()
    ids = list(range(1, 9))
    for k, nid in enumerate(ids):
        sim.add(nid, pos=(k * 0.5, 0.0),
                cw_s=5.0, id_slot_s=0.1, rechirp_min_s=25.0, rechirp_max_s=35.0)
    sim.run(240.0)
    missing = {nid: sorted((set(ids) - {nid}) - sim.discovered_by(nid))
               for nid in ids}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        return f"FAIL: incomplete discovery, missing={missing}"
    return True


def test_drift_adapts_rechirp():
    """High relative drift shortens the re-chirp interval toward the floor;
    near-zero drift holds the ceiling."""
    kw = dict(cw_s=3.0, id_slot_s=0.1, rechirp_min_s=10.0, rechirp_max_s=3000.0)
    sim = MacSim()
    a = sim.add(1, pos=(0.0, 0.0), drift_ppm=0.0, **kw)
    sim.add(2, pos=(1.0, 0.0), drift_ppm=300.0, **kw)
    sim.run(30.0)
    fast_iv = a.rechirp_interval_s

    sim2 = MacSim()
    b = sim2.add(1, pos=(0.0, 0.0), drift_ppm=10.0, **kw)
    sim2.add(2, pos=(1.0, 0.0), drift_ppm=10.0, **kw)
    sim2.run(30.0)
    slow_iv = b.rechirp_interval_s

    if not (a.peers and b.peers):
        return "FAIL: no peer drift observed (no beacons exchanged?)"
    if fast_iv >= slow_iv:
        return f"FAIL: high-drift interval {fast_iv:.1f}s not < low-drift {slow_iv:.1f}s"
    if slow_iv < 3000.0 - 1e-6:
        return f"FAIL: near-zero drift did not hold ceiling ({slow_iv:.1f}s)"
    return True


def test_ack_stagger():
    """A node holds its ACK response by id*ack_stagger so simultaneous discovery
    doesn't fire ACKs at the same instant."""
    sim = MacSim()
    d2 = sim.add(2, pos=(0.0, 0.0), ack_stagger_s=5.0, **_mac_kw())
    beacon = cm.pack_packet(1, cm.BROADCAST_ID, cm.TYPE_BEACON, 0)
    d2.handle_frame({"bytes_hex": beacon.hex(), "at_sample": 0,
                     "grid_sps": d2.sps_eff}, now=10.0)
    acks = [e for e in d2.outbox if e[1].startswith("ack:")]
    if not acks:
        return "FAIL: node did not queue an ACK on discovery"
    if abs(acks[0][2] - (10.0 + 2 * 5.0)) > 1e-6:
        return f"FAIL: ACK ready_at={acks[0][2]}, expected 20.0"
    if d2.poll_tx(12.0, channel_busy=False) is not None:
        return "FAIL: ACK released before its id-stagger hold elapsed"
    return True


def test_settable_ids():
    """Arbitrary (non-contiguous) IDs still discover each other."""
    sim = MacSim()
    ids = [3, 9, 14]
    for k, nid in enumerate(ids):
        sim.add(nid, pos=(k * 1.0, 0.0), **_mac_kw())
    sim.run(60.0)
    for nid in ids:
        if sim.discovered_by(nid) != set(ids) - {nid}:
            return f"FAIL: node {nid} discovered {sorted(sim.discovered_by(nid))}"
    return True


def test_events_emitted():
    """A run with an EventLog produces a well-formed, renderable stream."""
    path = os.path.join(tempfile.mkdtemp(prefix="meshsim_ev_"), "events.jsonl")
    log = ev.EventLog(path, {"run_id": "test_events", "tier": "mac"})
    sim = MacSim(eventlog=log)
    for k, nid in enumerate([1, 2, 3]):
        sim.add(nid, pos=(k * 1.0, 0.0), **_mac_kw())
    sim.run(60.0)
    log.close()
    meta, recs = ev.load_events(path)
    if meta.get("tier") != "mac":
        return "FAIL: meta not written"
    kinds = {r["kind"] for r in recs}
    for need in (ev.KIND_BOOT, ev.KIND_TX_START, ev.KIND_RX_OK, ev.KIND_DISCOVERED):
        if need not in kinds:
            return f"FAIL: no {need} events (got {sorted(kinds)})"
    # every rx_ok must reference a real tx pseq and a peer
    if any(r.get("peer") is None for r in recs if r["kind"] == ev.KIND_RX_OK):
        return "FAIL: rx_ok missing peer"
    return True


def _build_ring(nodes, log, profile_mix=True, *, rechirp_min_s=22.0,
                rechirp_max_s=32.0, ack_stagger_s=12.0):
    """N-node ring on the validated healthy-discovery config.  The re-chirp /
    ACK cadence is overridable so the daemon can beacon more often (livelier
    piano roll) than the discovery preset."""
    import math
    sim = MacSim(eventlog=log, airtime_s=1.0)
    mix = ["clean", "noisy", "very_noisy"]
    for nid in range(nodes):
        a = 2 * math.pi * nid / nodes
        prof = mix[nid % 3] if profile_mix else "clean"
        sim.add(nid, pos=(1.8 * math.cos(a), 1.8 * math.sin(a)), profile=prof,
                cw_s=8.0, id_slot_s=0.1, rechirp_min_s=rechirp_min_s,
                rechirp_max_s=rechirp_max_s, ack_stagger_s=ack_stagger_s,
                first_beacon_stagger_s=0.5)
    return sim


def run_ring(nodes, run_s, emit_path, *, profile_mix=True, seed=1):
    """Emit-mode: an N-node ring under the validated healthy-discovery config.
    Writes events.jsonl and returns the sim (used by the meshrun wrapper/CLI)."""
    import meshevents as _ev
    log = _ev.EventLog(emit_path, {"run_id": f"mac{nodes}", "tier": "mac",
                                   "n_nodes": nodes,
                                   "scenario": f"ring{nodes}_healthy"})
    sim = _build_ring(nodes, log, profile_mix)
    sim.run(run_s)
    log.close()
    return sim


def run_daemon(nodes, emit_path, *, profile_mix=True, stop_check,
               pace_factor=5.0, flush_every_s=2.0):
    """Daemon-mode: run an N-node ring CONTINUOUSLY until stop_check() is true
    (the framework's Stop terminates the process).

    Time is compressed to `pace_factor` simulated seconds per wall-clock second
    (so a short watch still fills the piano roll — at 1:1 a 10 s glance shows
    almost nothing, since beacons are tens of seconds apart), and events.jsonl is
    flushed periodically so even a hard stop leaves a renderable file.

    Cadence is the FAIR discovery preset (re-chirp 22–32 s): the compression is
    what makes the roll lively, so there's no need to over-beacon.  A faster
    cadence oversubscribes the channel at 15 nodes and STARVES some nodes (they
    never win channel access → never beacon → nobody can discover them), which
    reads as artificially low discovery."""
    import time
    import meshevents as _ev
    log = _ev.EventLog(emit_path, {"run_id": f"mac{nodes}_daemon", "tier": "mac",
                                   "n_nodes": nodes, "mode": "daemon",
                                   "scenario": f"ring{nodes}_daemon"})
    sim = _build_ring(nodes, log, profile_mix)      # healthy/fair cadence
    next_flush = 0.0
    while not stop_check():
        t0 = time.time()
        sim.step()
        if sim.t >= next_flush:
            log.flush()
            next_flush = sim.t + flush_every_s
            print(f"[daemon] t={sim.t:6.1f}s  tx={sim.stats['tx']} "
                  f"rx_ok={sim.stats['rx_ok']} disc={sim.stats['discovered']}",
                  flush=True)
        if pace_factor > 0:
            dt = sim.dt / pace_factor - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    for nid in sim.members:                       # drain any buffered daemon events
        sim._drain(nid, sim.t)
    log.close()
    print(f"[daemon] stopped at t={sim.t:.1f}s  stats={sim.stats}", flush=True)
    return sim


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(add_help=True,
                                 description="Tier 1 MAC mesh sim (tests or emit).")
    ap.add_argument("--nodes", type=int, default=0,
                    help="emit-mode: run an N-node ring instead of the selftests")
    ap.add_argument("--run", type=float, default=600.0, help="emit-mode run seconds")
    ap.add_argument("--emit", help="emit-mode events.jsonl output path")
    ap.add_argument("--all-clean", action="store_true",
                    help="emit-mode: all nodes clean (default cycles noise buckets)")
    ap.add_argument("--daemon", action="store_true",
                    help="emit-mode: run continuously (wall-clock paced) until "
                         "SIGTERM/SIGINT, then flush and render")
    args, _ = ap.parse_known_args()

    if args.nodes:
        emit = args.emit or "events.jsonl"
        if args.daemon:
            import signal
            stop = {"v": False}

            def _stop(*_a):
                stop["v"] = True
            for s in (signal.SIGTERM, signal.SIGINT):
                signal.signal(s, _stop)
            print(f"[daemon] running {args.nodes}-node ring until stopped…", flush=True)
            sim = run_daemon(args.nodes, emit, profile_mix=not args.all_clean,
                             stop_check=lambda: stop["v"])
        else:
            sim = run_ring(args.nodes, args.run, emit, profile_mix=not args.all_clean)
        disc = {n: len(sim.discovered_by(n)) for n in range(args.nodes)}
        full = sum(1 for v in disc.values() if v == args.nodes - 1)
        print(f"nodes={args.nodes} run={args.run}s  stats={sim.stats}")
        print(f"full-discovery {full}/{args.nodes}  counts={disc}")
        print(f"events → {emit}")
        return 0

    tests = [
        ("mutual_discovery",     test_mutual_discovery),
        ("heavy_contention",     test_heavy_contention),
        ("drift_adapts_rechirp", test_drift_adapts_rechirp),
        ("ack_stagger",          test_ack_stagger),
        ("settable_ids",         test_settable_ids),
        ("events_emitted",       test_events_emitted),
    ]
    print("MESH SIM — TIER 1 (MAC-level, SINR/capture channel)\n")
    failures = 0
    for name, fn in tests:
        try:
            r = fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = f"EXCEPTION {type(e).__name__}: {e}"
        if r is True:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name}: {r}")
    print(f"\n*** {'ALL PASS' if not failures else str(failures)+' FAILED'} ***")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
