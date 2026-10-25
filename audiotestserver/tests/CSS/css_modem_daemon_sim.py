# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_modem_daemon_sim.py — offline validation of the continuous-RX/TX daemon.

Fixed-tick simulation of many ModemDaemons sharing a half-duplex acoustic
channel.  It validates the MEDIUM ACCESS and the drift-driven re-chirp WITHOUT
DSP: a packet's on-air time is an abstract AIRTIME_S (the MAC logic is airtime-
agnostic, so tiny airtimes keep tests fast), and the channel models the three
things the daemon has to cope with:

  * carrier sense  — a node hears any other node's in-flight transmission;
  * collisions     — two transmissions overlapping at a receiver both corrupt
                     (neither decodes), so nobody discovers from a clash;
  * clock drift     — each node has a ppm rate error; a reception's fitted
                     grid_sps reflects the RELATIVE drift, which is what the
                     daemon uses to set its re-chirp cadence.

Between this and css_ranging_selftest (real DSP) + css_announce (over-the-air),
the daemon's protocol is exercised end to end before any hardware run.

Run:  python tests/css_modem_daemon_sim.py        (exit 0 = pass)
"""
import os
import random
import sys
import tempfile

TESTS = os.path.dirname(os.path.abspath(__file__))

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

import css_modem as cm                          # noqa: E402
from css_modem_fw import ModemFirmware, DEFAULT_PHY  # noqa: E402
from css_modem_daemon import ModemDaemon         # noqa: E402

C = cm.SPEED_OF_SOUND_M_S
AIRTIME_S = 2.0        # abstract per-packet on-air time (MAC is airtime-agnostic)
DT = 0.1              # sim tick


class DaemonSim:
    def __init__(self, fs=48000, verbose=False):
        self.fs = fs
        self.verbose = verbose
        self.members = {}     # id -> {daemon,pos,drift_ppm,tx_end}
        self.txs = []         # {id,src,raw,tag,start,end}
        self._delivered = set()   # (tx_id, receiver_id)
        self._tx_seq = 0
        self._max_prop = 0.0
        self.t = 0.0
        self._work = tempfile.mkdtemp(prefix="daemonsim_")

    def add(self, node_id, pos=0.0, drift_ppm=0.0, **daemon_kw):
        phy = dict(DEFAULT_PHY, fs=self.fs, symbol_dur=0.5, chunk_size=2)
        fw = ModemFirmware(os.path.join(self._work, f"n{node_id}.json"),
                           node_id=node_id, workdir=self._work,
                           phy_override=phy, bake_info=False).boot()
        d = ModemDaemon(fw, rng=random.Random(1000 + node_id), **daemon_kw)
        self.members[node_id] = {"daemon": d, "pos": float(pos),
                                 "drift_ppm": float(drift_ppm), "tx_end": -1.0}
        self._max_prop = max((self._prop(a, b) for a in self.members
                              for b in self.members if a != b), default=0.0)
        return d

    def _prop(self, a, b):
        return abs(self.members[a]["pos"] - self.members[b]["pos"]) / C

    def _audible(self, tx, rx_id, t):
        """Is tx audible at rx_id at global time t?"""
        p = self._prop(tx["src"], rx_id)
        return tx["start"] + p <= t < tx["end"] + p

    def _busy(self, rx_id, t):
        return any(tx["src"] != rx_id and self._audible(tx, rx_id, t)
                   for tx in self.txs)

    def _grid_sps(self, src, rx):
        sps_eff = self.members[rx]["daemon"].sps_eff
        rel = (self.members[rx]["drift_ppm"] - self.members[src]["drift_ppm"]) * 1e-6
        return sps_eff * (1.0 + rel)

    def _overlaps(self, a0, a1, b0, b1):
        return a0 < b1 and b0 < a1

    def step(self):
        t = self.t
        # 1) deliver completed receptions (collision- and half-duplex-checked)
        for tx in self.txs:
            for rx_id, m in self.members.items():
                if rx_id == tx["src"] or (tx["id"], rx_id) in self._delivered:
                    continue
                p = self._prop(tx["src"], rx_id)
                rx_start, rx_end = tx["start"] + p, tx["end"] + p
                if rx_end > t:
                    continue                     # reception not finished yet
                self._delivered.add((tx["id"], rx_id))
                # half-duplex: receiver was deaf during any of its OWN transmissions
                deaf = any(other["src"] == rx_id
                           and self._overlaps(rx_start, rx_end,
                                              other["start"], other["end"])
                           for other in self.txs)
                # collision: another node's reception window at rx overlaps this one
                clash = any(
                    other["id"] != tx["id"] and other["src"] != rx_id
                    and self._overlaps(
                        rx_start, rx_end,
                        other["start"] + self._prop(other["src"], rx_id),
                        other["end"]   + self._prop(other["src"], rx_id))
                    for other in self.txs)
                if deaf or clash:
                    if self.verbose and clash:
                        print(f"  t={t:5.1f} COLLISION at {rx_id} from {tx['src']}")
                    continue
                frame = {"bytes_hex": tx["raw"].hex(),
                         "at_sample": int(rx_end * self.fs),
                         "grid_sps": self._grid_sps(tx["src"], rx_id)}
                m["daemon"].handle_frame(frame, t)
        # 2) each idle node polls the MAC and may start a transmission
        for node_id, m in self.members.items():
            if t < m["tx_end"]:              # still transmitting (half-duplex)
                continue
            busy = self._busy(node_id, t)
            rt = m["daemon"].poll_tx(t, busy)
            if rt is not None:
                raw, tag = rt
                m["daemon"].on_tx(tag, int((t + AIRTIME_S) * self.fs))
                m["tx_end"] = t + AIRTIME_S
                self.txs.append({"id": self._tx_seq, "src": node_id,
                                 "raw": bytes(raw), "tag": tag,
                                 "start": t, "end": t + AIRTIME_S})
                self._tx_seq += 1
                if self.verbose:
                    p = cm.unpack_packet(raw)
                    print(f"  t={t:5.1f} TX  n{node_id} {p['type_name']} "
                          f"dst={p['dst']}")
        # 3) prune transmissions that can no longer be received or collide with
        #    anything (their reception window is entirely in the past).
        horizon = t - AIRTIME_S - self._max_prop - DT
        if len(self.txs) > 64:
            self.txs = [tx for tx in self.txs if tx["end"] + self._max_prop >= horizon]
        self.t += DT

    def run(self, until_s):
        while self.t < until_s:
            self.step()

    def discovered_by(self, node_id):
        return {e["peer"] for e in self.members[node_id]["daemon"].events
                if e["kind"] == "discovered"}


# ── scenario tests ───────────────────────────────────────────────────────────

def _mac_kw():
    return dict(cw_s=3.0, id_slot_s=0.1, rechirp_min_s=4.0, rechirp_max_s=20.0)


def test_mutual_discovery():
    """5 nodes beaconing all discover each other despite shared channel."""
    sim = DaemonSim()
    ids = [1, 2, 3, 4, 5]
    for k, nid in enumerate(ids):
        sim.add(nid, pos=k * 1.0, **_mac_kw())
    sim.run(90.0)
    for nid in ids:
        got = sim.discovered_by(nid)
        want = set(ids) - {nid}
        if got != want:
            return f"FAIL: node {nid} discovered {sorted(got)}, want {sorted(want)}"
    return True


def test_heavy_contention():
    """8 nodes all reach full mutual discovery under a busy-but-carriable load
    (offered load ~0.5: 8 nodes x 2s airtime / ~30s re-chirp).  Validates
    CSMA/CA + ID stagger + jitter scale to the 16-node target without livelock.
    (A shorter re-chirp would OVERSUBSCRIBE the channel — with real ~46s airtime
    you never beacon that often; that's a link-budget limit, not a MAC bug.)"""
    sim = DaemonSim()
    ids = list(range(1, 9))
    for k, nid in enumerate(ids):
        sim.add(nid, pos=k * 0.5,
                cw_s=5.0, id_slot_s=0.1,
                rechirp_min_s=25.0, rechirp_max_s=35.0)
    sim.run(240.0)
    missing = {nid: sorted((set(ids) - {nid}) - sim.discovered_by(nid))
               for nid in ids}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        return f"FAIL: incomplete discovery, missing={missing}"
    return True


def test_drift_adapts_rechirp():
    """High relative drift shortens the re-chirp interval toward the floor;
    near-zero drift leaves it at the ceiling.  Uses a high ceiling so the
    drift-driven value isn't clamped away (real drifts are only ~tens of
    samples/s, so the natural interval is minutes)."""
    kw = dict(cw_s=3.0, id_slot_s=0.1, rechirp_min_s=10.0, rechirp_max_s=3000.0)

    # Fast pair: 300 ppm apart.
    sim = DaemonSim()
    a = sim.add(1, pos=0.0, drift_ppm=0.0, **kw)
    sim.add(2, pos=1.0, drift_ppm=300.0, **kw)
    sim.run(30.0)
    fast_iv = a.rechirp_interval_s

    # Matched pair: ~0 ppm apart.
    sim2 = DaemonSim()
    b = sim2.add(1, pos=0.0, drift_ppm=10.0, **kw)
    sim2.add(2, pos=1.0, drift_ppm=10.0, **kw)
    sim2.run(30.0)
    slow_iv = b.rechirp_interval_s

    if not (a.peers and b.peers):
        return "FAIL: no peer drift observed (no beacons exchanged?)"
    if fast_iv >= slow_iv:
        return (f"FAIL: high-drift interval {fast_iv:.1f}s not < "
                f"low-drift {slow_iv:.1f}s")
    if slow_iv < 3000.0 - 1e-6:
        return f"FAIL: near-zero drift did not hold ceiling ({slow_iv:.1f}s)"
    return True


def test_ack_stagger():
    """A node holds its ACK response by id*ack_stagger so two peers that discover
    each other at the same instant don't fire ACKs simultaneously."""
    sim = DaemonSim()
    d2 = sim.add(2, pos=0.0, ack_stagger_s=5.0, **_mac_kw())
    beacon = cm.pack_packet(1, cm.BROADCAST_ID, cm.TYPE_BEACON, 0)
    d2.handle_frame({"bytes_hex": beacon.hex(), "at_sample": 0,
                     "grid_sps": d2.sps_eff}, now=10.0)
    acks = [e for e in d2.outbox if e[1].startswith("ack:")]
    if not acks:
        return "FAIL: node did not queue an ACK on discovery"
    ready_at = acks[0][2]
    if abs(ready_at - (10.0 + 2 * 5.0)) > 1e-6:        # now + id*stagger
        return f"FAIL: ACK ready_at={ready_at}, expected 20.0 (id2 x 5s hold)"
    # and it must NOT be releasable before the hold elapses
    if d2.poll_tx(12.0, channel_busy=False) is not None:
        return "FAIL: ACK released before its id-stagger hold elapsed"
    return True


def test_settable_ids():
    """Arbitrary (non-contiguous) IDs still discover each other."""
    sim = DaemonSim()
    ids = [3, 9, 14]
    for k, nid in enumerate(ids):
        sim.add(nid, pos=k * 1.0, **_mac_kw())
    sim.run(60.0)
    for nid in ids:
        if sim.discovered_by(nid) != set(ids) - {nid}:
            return f"FAIL: node {nid} discovered {sorted(sim.discovered_by(nid))}"
    return True


def main():
    tests = [
        ("mutual_discovery",     test_mutual_discovery),
        ("heavy_contention",     test_heavy_contention),
        ("drift_adapts_rechirp", test_drift_adapts_rechirp),
        ("ack_stagger",          test_ack_stagger),
        ("settable_ids",         test_settable_ids),
    ]
    print("CSS MODEM DAEMON SIM (continuous RX/TX, CSMA/CA)\n")
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
