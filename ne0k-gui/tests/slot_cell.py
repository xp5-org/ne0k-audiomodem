#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""One BAT-slot impairment cell for the simple (audio) OFDM link: two OfdmLinks on in-memory loop buses,
handshake clean, then the slot rule (engine/slotimpair.py) on both transmitters and a transfer.

    python tests/slot_cell.py --mode half --slot hot:4:block:data --out cell.json

Modes and what moves after the agreement:
    half    payloads both ways in timed windows; ACK-only frames are the control class
    oneway  initiator -> responder, no control frames back (a lost frame is a seq gap)
    stream  responder streams to the initiator, no control frames back
    arq     initiator -> responder, the responder sends ACK frames (control)
    full    FDD sub-bands, EC sliding window: ACKs ride inside the data chunks (no control-only frame)

Prints ``SLOT CELL PASS|FAIL <mode> <spec> ...`` and writes the JSON.  Exit 0 on PASS, 1 on FAIL, so a
cell that is expected to break (no ARQ, no re-probe) shows as FAIL with its counters: that is the
measurement.  NOHIT: no burst of the rule's class was filtered (e.g. ctrl on oneway, which sends no control
frame after the agreement), so the cell proved nothing and does not count as a pass.  --expect fail|nohit
makes that verdict exit 0, for cells that document a known break.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modemgui.audio import Registry                       # noqa: E402
from modemgui.engine import Session                       # noqa: E402
from modemgui.engine.modemcfg import OfdmConfig           # noqa: E402
from modemgui.engine.ofdm_link import OfdmLink            # noqa: E402
from modemgui.engine.slotimpair import attach_ofdm        # noqa: E402

SENDERS = {"half": "ab", "oneway": "a", "stream": "b", "arq": "a", "full": "ab"}


def run(mode: str, spec: str, nbytes: int, timeout: float, seed: int, bus: str, backend: str = "python") -> dict:
    reg, made = Registry(), []
    kw = dict(mode=mode, channel="line", backend=backend)
    if mode == "stream":
        kw.update(sender="responder", stream_poll_s=2.0)
    ends = {}
    for name, node, peer, rx, tx in (("a", 1, 2, f"loop:{bus}b", f"loop:{bus}a"),
                                     ("b", 2, 1, f"loop:{bus}a", f"loop:{bus}b")):
        s = Session(reg, 48000, 1024); s.open(rx, tx)
        link = OfdmLink(s, OfdmConfig(node_id=node, peer_id=peer, **kw),
                        log=(lambda m, n=name: print(f"{time.monotonic():.1f} {n}: {m}", file=sys.stderr, flush=True))
                        if os.environ.get("SLOT_CELL_LOG") else (lambda m: None))
        link.got = []
        link.on_data = link.got.append
        link.gate = attach_ofdm(link, spec, seed=seed + node)
        made.append((link, s)); ends[name] = link
    a, b = ends["a"], ends["b"]
    out = dict(mode=mode, backend=backend, spec=a.gate.spec, nbytes=nbytes, ok=False, why="")
    t0 = time.monotonic()
    try:
        for link in (a, b):
            link.start()
        if not (a.wait_up(180) and b.wait_up(60)):
            out["why"] = f"handshake: {a.fail_reason or b.fail_reason or 'timeout'}"
            return out
        out["hs_s"] = round(time.monotonic() - t0, 1)
        out["tx_bits"] = {k: int(v._tx_bat.sum()) for k, v in ends.items()}
        out["capacity"] = a.capacity
        senders = SENDERS[mode]
        payload = {k: bytes((i * 7 + (k == "b") * 101) & 0xFF for i in range(nbytes)) for k in senders}
        res, t1 = {}, time.monotonic()

        def push(k):
            link, ok = ends[k], True
            cap = max(1, min(link.capacity or 255, 255)) if mode == "half" else len(payload[k])
            for i in range(0, nbytes, cap):
                if time.monotonic() - t1 > timeout:
                    ok = False
                    break
                ok = link.send(payload[k][i:i + cap]) and ok
                if not ok and mode in ("half", "arq", "full"):
                    break                                  # the ARQ modes gave up on this piece
            res[k] = ok

        th = [threading.Thread(target=push, args=(k,), daemon=True) for k in senders]
        for t in th:
            t.start()
        for t in th:
            t.join(timeout + 30)
        deadline = time.monotonic() + 20                   # oneway/stream: let the last frame land
        rx_of = {"a": b, "b": a}
        while time.monotonic() < deadline and any(len(b"".join(rx_of[k].got)) < nbytes for k in senders):
            time.sleep(0.5)
        out["xfer_s"] = round(time.monotonic() - t1, 1)
        got = {k: b"".join(rx_of[k].got) for k in senders}
        out["dirs"] = {f"{k}->{'b' if k == 'a' else 'a'}": dict(
            sent_ok=bool(res.get(k)), delivered=len(got[k]), exact=got[k] == payload[k],
            retries=ends[k].retry_count, lost_frames=getattr(rx_of[k], "lost_frames", 0)) for k in senders}
        out["ok"] = all(d["exact"] for d in out["dirs"].values())
        rules = a.gate.rules
        if rules:                                          # the target must have been hit, or the cell proves nothing
            out["hits"] = {k: max(ends[k].gate.hits(r.cls) for r in rules) for k in "ab"}
            if not any(out["hits"].values()):
                out["nohit"] = True
        if not out["ok"]:
            out["why"] = "; ".join(f"{d}: {v['delivered']}/{nbytes} B, retries {v['retries']}, lost {v['lost_frames']}"
                                   for d, v in out["dirs"].items() if not v["exact"])
        return out
    finally:
        out["gate"] = {k: v.gate.report() for k, v in ends.items()}
        out["wall_s"] = round(time.monotonic() - t0, 1)
        for link, s in made:
            link.stop(); s.close()


def summary(d: str) -> int:
    """One line per cell JSON in d (in the order they ran) and d/../slot_summary.csv.  Never fails."""
    import csv
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json")), key=os.path.getmtime):
        try:
            r = json.load(open(f))
        except Exception as e:
            rows.append(dict(cell=os.path.basename(f)[:-5], verdict="ERROR", why=str(e)))
            continue
        g = r.get("gate", {})
        hit = [x for k in sorted(g) for p in g[k]["picked"][-1:] for x in p["rules"]]
        dirs = r.get("dirs", {})
        rows.append(dict(cell=os.path.basename(f)[:-5], verdict="NOHIT" if r.get("nohit") else "PASS" if r.get("ok") else "FAIL",
                         mode=r.get("mode"), backend=r.get("backend"), spec=r.get("spec"),
                         slots="+".join(str(x["n_slots"]) for x in hit), bits_hit="+".join(str(x["bits_hit"]) for x in hit),
                         filt_data=sum(v["bursts"]["data"]["filtered"] for v in g.values()),
                         filt_ctrl=sum(v["bursts"]["ctrl"]["filtered"] for v in g.values()),
                         delivered=" ".join(f"{k} {v['delivered']}/{r.get('nbytes')}" for k, v in dirs.items()),
                         retries=sum(v["retries"] for v in dirs.values()), xfer_s=r.get("xfer_s"), why=r.get("why", "")))
    if rows:
        with open(os.path.join(os.path.dirname(os.path.abspath(d)), "slot_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("%-22s %-5s %-14s %-24s %7s %9s %5s %5s %-22s %3s" % ("cell", "", "mode", "rule", "slots", "bits_hit",
                                                                 "f.dat", "f.ctl", "delivered", "rtx"))
    for r in rows:
        print("%-22s %-5s %-14s %-24s %7s %9s %5s %5s %-22s %3s%s" % (
            r["cell"], r["verdict"], f"{r.get('mode')}/{r.get('backend')}", r.get("spec", ""), r.get("slots", ""),
            r.get("bits_hit", ""), r.get("filt_data", ""), r.get("filt_ctrl", ""), r.get("delivered", ""),
            r.get("retries", ""), ("  <- " + r["why"]) if r.get("why") else ""))
    print("SLOT SUMMARY %d PASS, %d FAIL, %d NOHIT of %d cells" % tuple(
        [sum(r["verdict"] == v for r in rows) for v in ("PASS", "FAIL", "NOHIT")] + [len(rows)]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", metavar="DIR", help="print the table of every cell JSON in DIR instead of running one")
    ap.add_argument("--mode", choices=sorted(SENDERS))
    ap.add_argument("--backend", default="python", help="OfdmConfig.backend (cdx: the C full-duplex engine)")
    ap.add_argument("--slot", default="none", help="slotimpair rule(s), e.g. hot:4:block:data")
    ap.add_argument("--bytes", type=int, default=1000)
    ap.add_argument("--timeout", type=float, default=150.0, help="transfer budget after the handshake (s)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--expect", choices=["pass", "fail", "nohit"], default="pass",
                    help="the verdict this cell documents (NOHIT: no burst of the targeted class was filtered)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.summary:
        return summary(a.summary)
    if not a.mode:
        ap.error("--mode is required")
    r = run(a.mode, a.slot, a.bytes, a.timeout, a.seed, bus=f"sc{os.getpid()}", backend=a.backend)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(r, open(a.out, "w"), indent=1)
    hit = {k: [(x["rule"], x["n_slots"], x["bits_hit"]) for p in g["picked"][-1:] for x in p["rules"]]
           for k, g in r.get("gate", {}).items()}
    fil = {k: g["bursts"] for k, g in r.get("gate", {}).items()}
    verdict = "NOHIT" if r.get("nohit") else "PASS" if r["ok"] else "FAIL"
    print(f"SLOT CELL {verdict} {a.mode}{'/' + a.backend if a.backend != 'python' else ''} {r['spec']}  hs {r.get('hs_s')} s  "
          f"xfer {r.get('xfer_s')} s  bits {r.get('tx_bits')}  hit {hit}  bursts {fil}"
          + (f"  <- {r['why']}" if r["why"] else ""))
    return 0 if verdict.lower() == a.expect else 1


if __name__ == "__main__":
    sys.exit(main())
