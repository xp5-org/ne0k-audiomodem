# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_modem_sim.py — offline multi-node acoustic simulator for the modem layer.

Discrete-event simulation that drives any number of ModemNodes over a virtual
acoustic channel, with NO hardware and NO DSP.  It validates the PROTOCOL and
the ranging arithmetic end-to-end:

  * propagation delay between two nodes is their distance / speed-of-sound,
    converted to samples — so a ranging exchange measures a real, known range;
  * every node runs on its OWN clock (a distinct constant sample offset), so a
    passing ranging test proves the cross-device clock offsets truly cancel
    (they must, since ranging only ever differences timestamps within one
    node's own counter);
  * the channel quantizes arrivals to whole samples, mimicking the decoder's
    integer `at_sample`.

This is the packet-level companion to css_ranging_selftest.py (which validates
the real chirp encode/decode + at_sample).  Between them: css_ranging_selftest
proves the PHY timestamps arrivals faithfully; this proves the modem protocol
built on those timestamps behaves.

Run:  python tests/css_modem_sim.py        (exit 0 = pass)
"""
import heapq
import os
import sys

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

import css_modem as cm                 # noqa: E402
from css_modem_node import ModemNode    # noqa: E402

FS = 48000
C = cm.SPEED_OF_SOUND_M_S


class Sim:
    """Discrete-event acoustic channel.  Global time is in samples (float)."""

    def __init__(self, fs=FS, speed_of_sound=C, verbose=False):
        self.fs = fs
        self.c = speed_of_sound
        self.verbose = verbose
        self._members = {}        # node_id -> {"node":ModemNode,"pos":float,"offset":int}
        self._q = []              # heap of (global_t, counter, fn)
        self._counter = 0
        self.events = []          # (global_t, node_id, action_dict)

    def add_node(self, node: ModemNode, pos: float, offset: int):
        self._members[node.id] = {"node": node, "pos": float(pos),
                                  "offset": int(offset)}

    def _schedule(self, global_t, fn):
        heapq.heappush(self._q, (global_t, self._counter, fn))
        self._counter += 1

    def _prop_samples(self, a_id, b_id):
        dist = abs(self._members[a_id]["pos"] - self._members[b_id]["pos"])
        return dist / self.c * self.fs

    # ── the core: one node emits a packet at global time t ───────────────────
    def _transmit(self, src_id, raw, tag, global_t):
        m = self._members[src_id]
        local_tx = global_t + m["offset"]
        m["node"].on_tx(tag, local_tx)     # lets the sender fix its own t0
        if self.verbose:
            p = cm.unpack_packet(raw)
            print(f"    t={global_t:10.1f}  TX  node{src_id} "
                  f"{p['type_name']} dst={p['dst']} seq={p['seq']} tag={tag}")
        for dst_id, dm in self._members.items():
            if dst_id == src_id:
                continue
            arrival = round(global_t + self._prop_samples(src_id, dst_id))
            self._schedule(arrival,
                           lambda d=dst_id, r=raw, a=arrival: self._deliver(d, r, a))

    def _deliver(self, dst_id, raw, arrival_global):
        dm = self._members[dst_id]
        local_at = arrival_global + dm["offset"]
        actions = dm["node"].on_rx(raw, local_at)
        self._process(dst_id, actions, arrival_global)

    def _process(self, node_id, actions, now_global):
        m = self._members[node_id]
        for a in actions:
            if a["kind"] == "send":
                send_g = now_global + a["delay"]

                def _do(a=a, node_id=node_id, m=m, send_g=send_g):
                    tx_local = send_g + m["offset"]
                    raw = a["build"](tx_local)
                    self._transmit(node_id, raw, a["tag"], send_g)
                self._schedule(send_g, _do)
            else:
                self.events.append((now_global, node_id, a))
                if self.verbose:
                    print(f"    t={now_global:10.1f}  EV  node{node_id} {a}")

    # ── driving the sim ──────────────────────────────────────────────────────
    def inject(self, src_id, raw, tag, at_global=0.0):
        """Queue an application-initiated transmission (data / range / beacon)."""
        self._schedule(at_global,
                       lambda: self._transmit(src_id, raw, tag, at_global))

    def schedule_beacons(self, src_id, interval_s, until_s, phase_s=0.0):
        interval = interval_s * self.fs
        t = phase_s * self.fs
        until = until_s * self.fs
        while t <= until:
            def _beacon(src_id=src_id, t=t):
                raw, tag = self._members[src_id]["node"].make_beacon()
                self._transmit(src_id, raw, tag, t)
            self._schedule(t, _beacon)
            t += interval

    def run(self, until_s):
        until = until_s * self.fs
        while self._q and self._q[0][0] <= until:
            _, _, fn = heapq.heappop(self._q)
            fn()

    def events_of(self, kind, node_id=None):
        return [(t, nid, a) for (t, nid, a) in self.events
                if a["kind"] == kind and (node_id is None or nid == node_id)]


# ── scenario tests ───────────────────────────────────────────────────────────

def _node(nid, pos, offset, turnaround_s=0.25):
    return ModemNode(nid, fs=FS, turnaround_samples=int(turnaround_s * FS)), pos, offset


def test_discovery():
    """Two nodes beacon; each must discover the other."""
    sim = Sim()
    a, pa, oa = _node(1, 0.0, 12345)
    b, pb, ob = _node(2, 5.0, 999999)
    sim.add_node(a, pa, oa); sim.add_node(b, pb, ob)
    sim.schedule_beacons(1, interval_s=1.0, until_s=2.0, phase_s=0.1)
    sim.schedule_beacons(2, interval_s=1.0, until_s=2.0, phase_s=0.6)
    sim.run(3.0)
    d1 = {a["peer"] for _, n, a in sim.events_of("discovered", 1)}
    d2 = {a["peer"] for _, n, a in sim.events_of("discovered", 2)}
    if 2 not in d1:
        return f"FAIL: node1 did not discover node2 (discovered {d1})"
    if 1 not in d2:
        return f"FAIL: node2 did not discover node1 (discovered {d2})"
    return True


def test_data_ack():
    """Unicast DATA is delivered once and ACKed back."""
    sim = Sim()
    a, pa, oa = _node(1, 0.0, 5)
    b, pb, ob = _node(2, 3.0, 7000)
    sim.add_node(a, pa, oa); sim.add_node(b, pb, ob)
    raw, tag = a.send_data(2, b"hello")
    sim.inject(1, raw, tag, at_global=0.0)
    sim.run(1.0)
    delivered = [a for _, n, a in sim.events_of("deliver", 2)]
    if len(delivered) != 1:
        return f"FAIL: node2 delivered {len(delivered)} times (want 1)"
    if delivered[0]["packet"]["payload"] != b"hello":
        return f"FAIL: payload {delivered[0]['packet']['payload']!r}"
    acks = sim.events_of("ack", 1)
    if not acks:
        return "FAIL: node1 got no ACK"
    return True


def test_broadcast_no_ack():
    """Broadcast DATA reaches every peer and is NOT ACKed."""
    sim = Sim()
    a, pa, oa = _node(1, 0.0, 0)
    b, pb, ob = _node(2, 2.0, 111)
    c, pc, oc = _node(3, 4.0, 222)
    for n, p, o in ((a, pa, oa), (b, pb, ob), (c, pc, oc)):
        sim.add_node(n, p, o)
    raw, tag = a.send_data(cm.BROADCAST_ID, b"\x01\x02")
    sim.inject(1, raw, tag)
    sim.run(1.0)
    if not sim.events_of("deliver", 2) or not sim.events_of("deliver", 3):
        return "FAIL: broadcast not delivered to all peers"
    if sim.events_of("ack", 1):
        return "FAIL: broadcast was ACKed (should not be)"
    return True


def test_ranging_accuracy():
    """Ranging recovers a known distance despite differing clock offsets."""
    errs = []
    for dist in (2.0, 10.0, 50.0):
        sim = Sim()
        # Deliberately large, DIFFERENT offsets: if ranging leaked either node's
        # clock the answer would be wildly off.  It must cancel to ~0.
        a, pa, oa = _node(1, 0.0, 1_000_000)
        b, pb, ob = _node(2, dist, 37)
        sim.add_node(a, pa, oa); sim.add_node(b, pb, ob)
        raw, tag = a.start_range(2)
        sim.inject(1, raw, tag, at_global=0.0)
        sim.run(2.0)
        ranged = sim.events_of("ranged", 1)
        if not ranged:
            return f"FAIL: no range result at dist={dist}"
        measured = ranged[0][2]["distance_m"]
        err = abs(measured - dist)
        errs.append((dist, measured, err))
        # 1-sample channel quantization at 48k ~= 7 mm one way; allow a few cm.
        if err > 0.05:
            return (f"FAIL: dist={dist} measured={measured:.3f} err={err:.3f}m "
                    f"(offsets {oa} vs {ob} did not cancel?)")
    return True


def main():
    tests = [
        ("discovery",         test_discovery),
        ("data_ack",          test_data_ack),
        ("broadcast_no_ack",  test_broadcast_no_ack),
        ("ranging_accuracy",  test_ranging_accuracy),
    ]
    print("CSS MODEM MULTI-NODE SIM\n")
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
