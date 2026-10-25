# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
meshevents.py — shared event schema + JSONL bus for the 15-node mesh sims.

Both simulation tiers write the SAME event stream so one renderer (meshroll.py)
can draw either:

  * Tier 1  (meshsim_mac.py) — abstract MAC-level sim, delivery decided by an
    SINR/capture rule;
  * Tier 2  (meshsim_dsp.py) — real css_encoder/decoder over a mixed audio bus,
    delivery decided by an actual decode.

An `events.jsonl` file is one JSON object per line.  The FIRST line is a meta
header (kind == "_meta") carrying the run's node profiles, positions and params;
every following line is one event record:

  seq      int    monotonic emit order (tiebreaker when two events share t)
  t        float  GLOBAL sim time, seconds — the piano-roll's vertical axis
  node     int    0..14, the lane the event belongs to
  kind     str    one of KINDS
  t_local  float  the node's OWN clock, seconds — differs from t by clock drift,
                  which is the whole point of the drift scenarios; carried on
                  every record so the divergence stays visible in the viz.
  peer     int?   counterpart node (tx dst / rx src) — the arrow's other end
  ptype    str?   lowercase packet type: beacon|data|ack|range_req|range_reply|
                  ofdm_hs_req
  pseq     int?   4-bit packet seq — links a tx to the rx it produced, and dedups
  dur      float? airtime seconds — a tx block renders as a vertical extent
  reason   str?   rx-fail cause: collision|deaf|low_sinr|crc|no_lock
  snr_db   float? modeled/measured link SINR at the receiver
  detail   dict?  kind-specific extras (ranging distance, evm, backoff count, …)

Run:  python tests/meshevents.py        (exit 0 = selftest passed)
"""
from __future__ import annotations

import json
import os
import sys


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

import css_modem as cm  # noqa: E402

# ── event kinds ──────────────────────────────────────────────────────────────
KIND_TX_START      = "tx_start"       # node begins a transmission (carries dur)
KIND_TX_END        = "tx_end"         # transmission finished on the air
KIND_RX_OK         = "rx_ok"          # a frame decoded at this node
KIND_RX_FAIL       = "rx_fail"        # a reception was lost (see reason)
KIND_CARRIER_DEFER = "carrier_defer"  # CSMA deferred a queued packet (busy)
KIND_DISCOVERED    = "discovered"     # first beacon heard from a peer
KIND_ACK           = "ack"            # an ACK was observed for one of our DATAs
KIND_RANGED        = "ranged"         # a two-way range completed
KIND_DELIVER       = "deliver"        # an app payload was delivered to us
KIND_BOOT          = "boot"           # a node came up

KINDS = frozenset({
    KIND_TX_START, KIND_TX_END, KIND_RX_OK, KIND_RX_FAIL, KIND_CARRIER_DEFER,
    KIND_DISCOVERED, KIND_ACK, KIND_RANGED, KIND_DELIVER, KIND_BOOT,
})

# rx_fail reasons
REASON_COLLISION = "collision"   # two receptions overlapped, neither captured
REASON_DEAF      = "deaf"        # half-duplex: receiver was transmitting
REASON_LOW_SINR  = "low_sinr"    # below the capture threshold (too weak/noisy)
REASON_CRC       = "crc"         # decoded but CRC failed (DSP tier)
REASON_NO_LOCK   = "no_lock"     # decoder never acquired (DSP tier)

_META_KIND = "_meta"

# lowercase packet-type names for `ptype`, derived from css_modem's TYPE_NAMES so
# the two never drift apart.
PTYPE_BY_CODE = {code: name.lower() for code, name in cm.TYPE_NAMES.items()}


def ptype_of(raw: bytes) -> str | None:
    """Lowercase packet-type name for a raw packet, or None if unparseable."""
    try:
        return cm.unpack_packet(raw)["type_name"].lower()
    except cm.ModemError:
        return None


# ── writer ───────────────────────────────────────────────────────────────────
class EventLog:
    """Append-only JSONL event writer.  One per run.

    The header (run_meta) is written immediately as line 0 so a partially
    completed run is still renderable.  emit() assigns the monotonic `seq`.
    """

    def __init__(self, path, run_meta: dict | None = None):
        self.path = os.fspath(path)
        self._seq = 0
        self._fh = open(self.path, "w", encoding="utf-8")
        meta = dict(run_meta or {})
        meta["kind"] = _META_KIND
        self._write(meta)

    def _write(self, obj: dict):
        self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def emit(self, kind: str, *, t: float, node: int, t_local: float | None = None,
             **fields) -> dict:
        """Append one event record and return it.  Unknown kinds raise so a
        typo can't silently produce an unrenderable stream."""
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        rec = {"seq": self._seq, "t": round(float(t), 6), "node": int(node),
               "kind": kind, "t_local": round(float(t if t_local is None
                                                     else t_local), 6)}
        # keep only meaningful (non-None) optional fields, to keep lines compact
        for k, v in fields.items():
            if v is not None:
                rec[k] = v
        self._seq += 1
        self._write(rec)
        return rec

    def flush(self):
        self._fh.flush()

    def close(self):
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ── daemon adapter ───────────────────────────────────────────────────────────
def from_daemon_event(log: EventLog, action: dict, *, t: float, node: int,
                      t_local: float | None = None):
    """Translate one ModemDaemon.drain_events() Action dict into an emit().

    The daemon surfaces discovered / ack / ranged / deliver actions (see
    css_modem_node.on_rx).  TX / RX-success / RX-failure events are NOT produced
    by the daemon — the channel sees those — so the sim loop emits them directly.
    Returns the emitted record, or None for an action kind we don't log.
    """
    k = action.get("kind")
    if k == "discovered":
        return log.emit(KIND_DISCOVERED, t=t, node=node, t_local=t_local,
                        peer=action.get("peer"), pseq=action.get("seq"))
    if k == "ack":
        return log.emit(KIND_ACK, t=t, node=node, t_local=t_local,
                        peer=action.get("peer"), pseq=action.get("seq"))
    if k == "ranged":
        return log.emit(KIND_RANGED, t=t, node=node, t_local=t_local,
                        peer=action.get("peer"),
                        detail={"distance_m": action.get("distance_m"),
                                "rtt": action.get("rtt"),
                                "turnaround": action.get("turnaround")})
    if k == "deliver":
        pkt = action.get("packet") or {}
        return log.emit(KIND_DELIVER, t=t, node=node, t_local=t_local,
                        peer=pkt.get("src"), pseq=pkt.get("seq"),
                        detail={"nbytes": len(pkt.get("payload", b""))
                                if isinstance(pkt.get("payload"), (bytes, bytearray))
                                else None})
    return None


# ── reader ───────────────────────────────────────────────────────────────────
def load_events(path):
    """Read an events.jsonl file → (meta: dict, records: list[dict]).

    `meta` is the header object (without its "kind" marker); `records` are the
    event lines in file order.  Blank lines are skipped.  A file with no meta
    header yields meta == {} and all lines as records (forward-tolerant).
    """
    meta: dict = {}
    records: list[dict] = []
    with open(os.fspath(path), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("kind") == _META_KIND and not records and not meta:
                obj.pop("kind", None)
                meta = obj
            else:
                records.append(obj)
    return meta, records


# ── selftest ─────────────────────────────────────────────────────────────────
def _selftest() -> bool:
    import tempfile
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{name:32}] {'PASS' if cond else 'FAIL'}")

    path = os.path.join(tempfile.mkdtemp(prefix="meshevents_"), "events.jsonl")
    meta_in = {"run_id": "selftest", "nodes": {0: "clean", 1: "noisy"},
               "positions": {0: [0, 0], 1: [3, 0]}}
    with EventLog(path, meta_in) as log:
        # a beacon tx from node 0, heard ok at 1, plus a drifted local clock
        log.emit(KIND_BOOT, t=0.0, node=0)
        log.emit(KIND_TX_START, t=1.0, node=0, ptype="beacon", pseq=0, dur=2.0)
        log.emit(KIND_TX_END, t=3.0, node=0, ptype="beacon", pseq=0)
        log.emit(KIND_RX_OK, t=3.02, node=1, t_local=3.05, peer=0,
                 ptype="beacon", pseq=0, snr_db=14.3)
        log.emit(KIND_RX_FAIL, t=3.02, node=2 if False else 1, peer=0,
                 reason=REASON_COLLISION)
        # a daemon 'discovered' action routed through the adapter
        from_daemon_event(log, {"kind": "discovered", "peer": 0, "seq": 0},
                          t=3.02, node=1, t_local=3.05)
        from_daemon_event(log, {"kind": "ranged", "peer": 0, "distance_m": 3.0,
                                "rtt": 1000, "turnaround": 200},
                          t=9.0, node=1)

    meta, recs = load_events(path)
    check("meta round-trips", meta.get("run_id") == "selftest")
    check("meta has no _meta key", "kind" not in meta)
    check("all records loaded", len(recs) == 7)
    check("seq is monotonic", [r["seq"] for r in recs] == list(range(7)))
    check("tx carries dur", any(r["kind"] == "tx_start" and r.get("dur") == 2.0
                                for r in recs))
    check("global/local clocks diverge",
          any(r["kind"] == "rx_ok" and r["t"] != r["t_local"] for r in recs))
    check("None fields dropped",
          all("snr_db" not in r for r in recs if r["kind"] == "boot"))
    check("adapter mapped discovered",
          any(r["kind"] == "discovered" and r.get("peer") == 0 for r in recs))
    check("adapter mapped ranged distance",
          any(r["kind"] == "ranged" and r["detail"]["distance_m"] == 3.0
              for r in recs))

    # unknown kind must raise
    try:
        EventLog(path + ".x").emit("bogus", t=0, node=0)
        raised = False
    except ValueError:
        raised = True
    check("unknown kind raises", raised)

    return ok


def main() -> int:
    print("MESH EVENTS (shared schema + JSONL bus)\n")
    ok = _selftest()
    print(f"\n*** {'ALL PASS' if ok else 'FAILED'} ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
