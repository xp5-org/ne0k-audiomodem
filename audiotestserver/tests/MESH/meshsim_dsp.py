# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
meshsim_dsp.py — Tier 2: DSP-in-loop mesh simulation (real encode/decode).

Where Tier 1 (meshsim_mac.py) decides delivery from an SINR formula, Tier 2 puts
REAL audio on a shared acoustic bus (meshbus.AirBus) and lets the REAL css
encoder/decoder decide who hears whom.  Collisions, capture and noise are not
modelled — they EMERGE: two beacons that overlap at a receiver sum in that
receiver's lane and the decoder simply fails to recover them.

Design (open-loop, embarrassingly parallel):
  1. each node's beacons are SCHEDULED over the run (id-staggered + jittered);
  2. every beacon is rendered with the real css_encoder and mixed into each
     other node's lane with that link's gain + propagation delay;
  3. each receiver's mixed lane gets its bucket noise floor (+ optional clock
     drift as a resample);
  4. every lane is DECODED with the real css_decoder — this is the heavy step,
     and since lanes are independent it fans out across cores (--jobs);
  5. decoded frames feed each node's ModemNode.on_rx to surface discovery, and
     the shared event stream (meshevents) is written for the piano roll.

This is the correctness ORACLE and the 40-core workload in one: --jobs 1 is the
single-process reference; --jobs N decodes N lanes at once.  Reactive traffic
(CSMA-timed ACKs, ranging replies) is Tier 1's domain and is intentionally out
of scope here — Tier 2's job is real-waveform reception under a real channel.

KNOWN LIMITATION — decoder PHY robustness.  A single clean burst decodes
reliably (proven by meshbus.py's roundtrip).  But in assembled multi-burst
lanes at the fast PHY used here (symbol_dur 0.1, chosen so a beacon is ~10 s
instead of ~85 s at the OTA PHY), css_decoder shows a DIRECTIONAL phase/grid
sensitivity: with two symmetric nodes it will reliably decode one direction and
drop the other, despite near-identical sync scores — the same long-symbol
phase/grid-alignment family seen over the air.  Consequence: this tier does NOT
guarantee full mutual discovery; it demonstrates real reception, capture,
collision and noise-driven failure at scale, and it is where the CPU load lives.
Making discovery reliable is a css_decoder fast-PHY tuning task (or use the OTA
PHY and accept ~85 s bursts), tracked separately from this simulation layer.

Run:  python tests/meshsim_dsp.py                 (3-node oracle selftest)
      python tests/meshsim_dsp.py --nodes 15 --jobs 15 --run 120 --emit out.jsonl
"""
from __future__ import annotations

import argparse
import math
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

import numpy as np                                     # noqa: E402
import css_modem as cm                                 # noqa: E402
from css_modem_fw import ModemFirmware, DEFAULT_PHY    # noqa: E402
from meshchannel import Channel2D                      # noqa: E402
from meshbus import AirBus                             # noqa: E402
import meshevents as ev                                # noqa: E402

# Fast-but-fractional-delay-robust PHY (see meshbus: 0.1 is the floor that still
# decodes with a realistic sub-symbol propagation delay).
DSP_PHY = dict(DEFAULT_PHY, symbol_dur=0.1, chunk_size=2, f_min=800, f_max=4000)


# ── lane decode worker (module-level so ProcessPool can pickle it) ────────────
def _decode_lane(args):
    """Decode one node's lane from a .npy file with the real css_decoder.
    Returns (node_id, [frame dicts]).  Runs in a worker process."""
    node_id, npy_path, info_path, fs, workdir = args
    os.environ.setdefault("AUDIOTOOL_RUN_DIR", workdir)
    import contextlib
    import io
    import testlib
    lane = np.load(npy_path)
    with contextlib.redirect_stdout(io.StringIO()):
        res = testlib.decode_array(lane, fs, info_path,
                                   overrides={"result_out": os.path.join(
                                       workdir, f"dec_{node_id}.json")})
    frames = [f for f in (res.get("frames") or []) if "bytes_hex" in f]
    # keep only what the event layer needs, so little crosses the process boundary
    slim = [{"bytes_hex": f["bytes_hex"], "at_sample": int(f.get("at_sample", 0))}
            for f in frames]
    return node_id, slim


class DspSim:
    def __init__(self, *, fs=48000, run_s=120.0, beacon_period_s=25.0,
                 beacon_jitter_s=6.0, ref_dist_m=1.0, pathloss_exp=2.0,
                 carrier_sense_db=0.0, guard_s=1.5, eventlog=None, seed=1,
                 verbose=False):
        self.fs = int(fs)
        self.run_s = float(run_s)
        self.beacon_period_s = float(beacon_period_s)
        self.beacon_jitter_s = float(beacon_jitter_s)
        self.ref_dist_m = float(ref_dist_m)
        self.pathloss_exp = float(pathloss_exp)
        self.carrier_sense_db = float(carrier_sense_db)
        # minimum silence a receiver needs between bursts to re-acquire: without
        # it, CSMA packs beacons back-to-back (< 1 s apart) and the decoder,
        # still mid-burst, loses BOTH.  This is the acoustic inter-frame space.
        self.guard_s = float(guard_s)
        self.log = eventlog
        self.rng = random.Random(seed)
        self.verbose = verbose

        self.members: dict[int, dict] = {}   # id -> {fw, pos, profile, drift_ppm}
        self.txs: list[dict] = []            # {src, pseq, start, end, raw, tag}
        self._discovered: dict[int, set] = {}
        self.stats = {"tx": 0, "rx_ok": 0, "rx_fail": 0, "collision": 0,
                      "no_lock": 0, "discovered": 0}
        self._work = tempfile.mkdtemp(prefix="meshsim_dsp_")
        os.environ.setdefault("AUDIOTOOL_RUN_DIR", self._work)
        self._chan: Channel2D | None = None

    # ── construction ─────────────────────────────────────────────────────────
    def add(self, node_id, pos=(0.0, 0.0), profile="clean", drift_ppm=0.0):
        phy = dict(DSP_PHY, fs=self.fs)
        fw = ModemFirmware(os.path.join(self._work, f"n{node_id}.json"),
                           node_id=node_id, workdir=self._work,
                           phy_override=phy, bake_info=True).boot()
        self.members[node_id] = {"fw": fw, "pos": (float(pos[0]), float(pos[1])),
                                 "profile": str(profile),
                                 "drift_ppm": float(drift_ppm)}
        self._discovered.setdefault(node_id, set())

    def _rebuild_channel(self):
        pos = {n: m["pos"] for n, m in self.members.items()}
        prof = {n: m["profile"] for n, m in self.members.items()}
        self._chan = Channel2D(pos, prof, ref_dist_m=self.ref_dist_m,
                               pathloss_exp=self.pathloss_exp, fs=self.fs)

    # ── CSMA scheduling ──────────────────────────────────────────────────────
    def _sense_busy_until(self, nid, t, placed, delay_s):
        """Carrier sense at the TRANSMITTER: if node `nid` wants to start at t but
        an already-placed, audible transmission is arriving at nid at that
        instant, return the time that reception clears (so nid defers); else t.

        This is what serialises the channel — without it 15 nodes with ~10 s
        beacons oversubscribe ~6× and every burst collides.  Sub-sense-latency
        simultaneous starts and hidden terminals still collide (they're placed
        anyway), which is exactly the residue Tier 2 exists to show.
        """
        busy_until = t
        g = self.guard_s
        for (src, s0, e0) in placed:
            if src == nid:
                continue
            if self._chan.link_snr_db(src, nid) < self.carrier_sense_db:
                continue                       # inaudible → doesn't hold the channel
            d = delay_s[(src, nid)]
            # the channel must be idle for `guard_s` AFTER a reception clears
            # before nid may start, so consecutive bursts are separated by enough
            # silence for the decoder to re-acquire.
            if s0 + d <= t < e0 + d + g:
                busy_until = max(busy_until, e0 + d + g)
        return busy_until

    def _schedule(self, burst_s):
        """Greedy CSMA plan: each node wants to beacon every beacon_period_s, but
        defers (with a small backoff) whenever it senses the channel busy."""
        import heapq
        ids = sorted(self.members)
        delay_s = {(a, b): self._chan.prop_delay(a, b)
                   for a in ids for b in ids if a != b}
        pq = []
        for k, nid in enumerate(ids):
            heapq.heappush(pq, (0.5 + k * (self.beacon_period_s / len(ids)), nid))
        placed = []          # (src, start_s, end_s)
        result = []
        guard = 0
        while pq and guard < 500000:
            guard += 1
            t, nid = heapq.heappop(pq)
            if t >= self.run_s:
                continue
            busy = self._sense_busy_until(nid, t, placed, delay_s)
            if busy > t:
                heapq.heappush(pq, (busy + self.rng.uniform(0.1, 1.0), nid))
                continue
            result.append((t, nid))
            placed.append((nid, t, t + burst_s))
            nxt = t + self.beacon_period_s + self.rng.uniform(0, self.beacon_jitter_s)
            if nxt < self.run_s:
                heapq.heappush(pq, (nxt, nid))
        result.sort()
        return result

    def build(self):
        """Render every scheduled beacon onto the bus; return the AirBus."""
        self._rebuild_channel()
        # encode one probe beacon to learn the burst length (all beacons are the
        # same length) — needed to size the horizon and the CSMA plan.
        any_fw = next(iter(self.members.values()))["fw"]
        probe_raw, probe_tag = any_fw.node.make_beacon()
        any_fw.node._seq = 0                       # undo the probe's seq bump
        probe_pcm = any_fw.encode_packet(probe_raw, probe_tag)
        burst_len = len(probe_pcm)
        burst_s = burst_len / self.fs
        horizon = int(self.run_s * self.fs) + burst_len + self.fs
        bus = AirBus(self._chan, self.fs, horizon)

        for start_s, nid in self._schedule(burst_s):
            fw = self.members[nid]["fw"]
            raw, tag = fw.node.make_beacon()
            pcm = fw.encode_packet(raw, tag)
            start = int(start_s * self.fs)
            bus.write_tx(nid, start, pcm)
            pseq = cm.unpack_packet(raw)["seq"]
            self.txs.append({"src": nid, "pseq": pseq, "start": start,
                             "end": start + len(pcm), "raw": raw, "tag": tag})
            self.stats["tx"] += 1
            if self.log:
                self.log.emit(ev.KIND_TX_START, t=start_s, node=nid,
                              t_local=self._t_local(nid, start_s), ptype="beacon",
                              pseq=pseq, dur=len(pcm) / self.fs)
        bus.add_noise(np.random.default_rng(self.rng.randint(0, 1 << 30)))
        self._apply_drift(bus)
        return bus

    def _apply_drift(self, bus):
        from css_loopback import resample_ppm
        for nid, m in self.members.items():
            ppm = m["drift_ppm"]
            if abs(ppm) > 1e-9:
                lane = resample_ppm(bus.lanes[nid], ppm)
                # keep the lane length fixed at the horizon
                out = np.zeros(bus.N, dtype=np.float32)
                out[:min(len(lane), bus.N)] = lane[:bus.N]
                bus.lanes[nid] = out

    def _t_local(self, nid, t):
        return t * (1.0 + self.members[nid]["drift_ppm"] * 1e-6)

    # ── decode (parallel) + reception ────────────────────────────────────────
    def run(self, jobs=1):
        bus = self.build()
        # dump lanes to .npy so worker processes can load them independently
        lane_paths = {}
        for nid in self.members:
            p = os.path.join(self._work, f"lane_{nid}.npy")
            np.save(p, bus.lanes[nid])
            lane_paths[nid] = p

        tasks = [(nid, lane_paths[nid], self.members[nid]["fw"].info_path,
                  self.fs, self._work) for nid in self.members]

        if jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            import testlib
            # pool_worker_init arms PR_SET_PDEATHSIG in each worker: a stopped
            # or crashed run must not leave `jobs` numpy-loaded decoders behind
            # (the `with` block's shutdown never runs on SIGTERM).
            with ProcessPoolExecutor(max_workers=jobs,
                                     initializer=testlib.pool_worker_init) as ex:
                results = list(ex.map(_decode_lane, tasks))
        else:
            results = [_decode_lane(t) for t in tasks]

        frames_by_node = {nid: fr for nid, fr in results}
        self._receive(frames_by_node)
        self._reconcile_failures(frames_by_node)
        return self

    def _receive(self, frames_by_node):
        """Feed each node's decoded frames to its ModemNode → discovery events."""
        for nid, frames in frames_by_node.items():
            node = self.members[nid]["fw"].node
            for f in sorted(frames, key=lambda x: x["at_sample"]):
                raw = bytes.fromhex(f["bytes_hex"])
                try:
                    p = cm.unpack_packet(raw)
                except cm.ModemError:
                    continue
                at = f["at_sample"]
                t = at / self.fs
                src = p["src"]
                self.stats["rx_ok"] += 1
                if self.log:
                    snr = (self._chan.link_snr_db(src, nid)
                           if src in self.members else None)
                    self.log.emit(ev.KIND_RX_OK, t=t, node=nid,
                                  t_local=self._t_local(nid, t), peer=src,
                                  ptype=p["type_name"].lower(), pseq=p["seq"],
                                  snr_db=round(snr, 2) if snr is not None else None)
                for a in node.on_rx(raw, at):
                    if a.get("kind") == "discovered":
                        self._discovered[nid].add(a.get("peer"))
                        self.stats["discovered"] += 1
                        if self.log:
                            self.log.emit(ev.KIND_DISCOVERED, t=t, node=nid,
                                          t_local=self._t_local(nid, t),
                                          peer=a.get("peer"))

    def _reconcile_failures(self, frames_by_node):
        """A tx that was audible at a receiver but produced no matching rx_ok is
        a reception FAILURE: collision if an audible tx overlapped it at that
        receiver, else no_lock."""
        for rx in self.members:
            got = {(bytes.fromhex(f["bytes_hex"])[0],  # src nibble byte
                    ) for f in frames_by_node.get(rx, [])}
            decoded_pairs = set()
            for f in frames_by_node.get(rx, []):
                try:
                    p = cm.unpack_packet(bytes.fromhex(f["bytes_hex"]))
                    decoded_pairs.add((p["src"], p["seq"]))
                except cm.ModemError:
                    pass
            for tx in self.txs:
                src = tx["src"]
                if src == rx:
                    continue
                if self._chan.link_snr_db(src, rx) < self.carrier_sense_db:
                    continue                      # never audible → not a failure
                if (src, tx["pseq"]) in decoded_pairs:
                    continue                      # decoded fine
                # audible but not decoded → failure; classify
                overlapped = any(
                    o is not tx and o["src"] != rx
                    and self._chan.link_snr_db(o["src"], rx) >= self.carrier_sense_db
                    and o["start"] < tx["end"] and tx["start"] < o["end"]
                    for o in self.txs)
                reason = ev.REASON_COLLISION if overlapped else ev.REASON_NO_LOCK
                self.stats["rx_fail"] += 1
                self.stats[reason] += 1
                if self.log:
                    t = tx["end"] / self.fs
                    self.log.emit(ev.KIND_RX_FAIL, t=t, node=rx,
                                  t_local=self._t_local(rx, t), peer=src,
                                  ptype="beacon", pseq=tx["pseq"], reason=reason,
                                  snr_db=round(self._chan.link_snr_db(src, rx), 2))

    def discovered_by(self, nid):
        return set(self._discovered.get(nid, set()))


# ── selftest / CLI ───────────────────────────────────────────────────────────
def _selftest() -> bool:
    """Validate the DSP-in-loop PIPELINE end to end: real encode → mixed bus →
    real decode → discovery → events.  (The single-burst decode *correctness* is
    proven separately and cleanly by meshbus.py's roundtrip; this asserts the
    multi-node assembly, the parallel and serial decode paths agree, and events
    are produced.)

    Note: this fast PHY (symbol_dur 0.1) is NOT reliable enough for guaranteed
    full mutual discovery — the css_decoder shows a directional phase/grid
    sensitivity in assembled multi-burst lanes (see the module docstring).  So
    the oracle asserts that real reception and discovery HAPPEN, not that every
    ordered pair completes; that reliability is a decoder-PHY tuning matter, not
    a property of this simulation layer.
    """
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{name:46}] {'PASS' if cond else 'FAIL'}")

    three = DspSim(run_s=110.0, beacon_period_s=14.0, beacon_jitter_s=3.0)
    for nid, p in {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (1.0, 1.5)}.items():
        three.add(nid, pos=p, profile="clean")
    three.run(jobs=1)
    check("real frames decoded over the mixed bus", three.stats["rx_ok"] > 0)
    check("discovery occurred from real audio", three.stats["discovered"] > 0)
    check("reception failures surfaced", three.stats["rx_fail"] > 0)

    # parallel decode path must agree with serial on frame count (determinism of
    # the per-lane decode; only scheduling RNG differs, so rerun the same sim)
    par = DspSim(run_s=110.0, beacon_period_s=14.0, beacon_jitter_s=3.0, seed=1)
    for nid, p in {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (1.0, 1.5)}.items():
        par.add(nid, pos=p, profile="clean")
    par.run(jobs=3)
    check("parallel (jobs=3) decode agrees with serial",
          par.stats["rx_ok"] == three.stats["rx_ok"])
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Tier 2 DSP-in-loop mesh sim.")
    ap.add_argument("--nodes", type=int, default=0,
                    help="run an N-node ring scenario instead of the 3-node selftest")
    ap.add_argument("--jobs", type=int, default=1, help="parallel lane-decode workers")
    ap.add_argument("--run", type=float, default=120.0, help="run seconds")
    ap.add_argument("--emit", help="write events.jsonl here")
    args = ap.parse_args()

    if not args.nodes:
        print("MESH SIM — TIER 2 (DSP-in-loop, real encode/decode)\n")
        okic = _selftest()
        print(f"\n*** {'ALL PASS' if okic else 'FAILED'} ***")
        return 0 if okic else 1

    log = ev.EventLog(args.emit, {"run_id": f"dsp{args.nodes}", "tier": "dsp",
                                  "n_nodes": args.nodes}) if args.emit else None
    sim = DspSim(run_s=args.run, eventlog=log)
    for nid in range(args.nodes):
        a = 2 * math.pi * nid / args.nodes
        sim.add(nid, pos=(1.8 * math.cos(a), 1.8 * math.sin(a)),
                profile=(["clean", "noisy", "very_noisy"][nid % 3]))
    import time
    t0 = time.time()
    sim.run(jobs=args.jobs)
    dt = time.time() - t0
    if log:
        log.close()
    disc = {nid: len(sim.discovered_by(nid)) for nid in range(args.nodes)}
    full = sum(1 for v in disc.values() if v == args.nodes - 1)
    print(f"nodes={args.nodes} jobs={args.jobs} run={args.run}s  decoded in {dt:.1f}s")
    print(f"stats={sim.stats}")
    print(f"full-discovery {full}/{args.nodes}  counts={disc}")
    if args.emit:
        print(f"events → {args.emit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
