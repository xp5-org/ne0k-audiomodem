# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Half-duplex timeline regression for the ofdmhs_node binding.

Models what ofdm_handshake's pure-logic selftest cannot: real AIRTIME and a
realistic carrier-sense detector.  Two nodes share one channel and any overlap
of their TX intervals is a collision (both frames lost).

Guards against two collision modes:

  * a binding deferring to the peer while the air is busy but then sending
    its QUEUED action anyway -- a timeout-driven RETRY going out right after
    the BAT_REPORT it was asking for, colliding with the responder's
    retransmission.  Guarded by clear_to_send() draining the receiver before
    keying up, plus HandshakeSM.hold_off() giving back time spent
    transmitting rather than listening.
  * carrier sense grading the whole DECODE window and calling anything
    shorter than 3*QUIET_TAIL_S "idle", leaving the node blind for 3s after
    every listen_start reset.  Guarded by _air_busy() sensing only the
    newest audio.

Run with the app venv (the system python3 has no reedsolo, which
ofdm_modem imports unconditionally):

    /opt/venv/bin/python tests/regression/ofdmhs_collide_regress.py

Exits non-zero if the current design collides or fails to converge.
"""
import sys
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parents[1]
for _p in (_TESTS_ROOT, _TESTS_ROOT / "OFDM"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ofdm_handshake import HandshakeSM, _MSG_NAMES

AIRTIME = {"probe": 5.5, 1: 8.0, 2: 8.0, 3: 5.0, 4: 5.0, 5: 5.0}
DT = 0.25
POLL_S = 3.0
QUIET_TAIL_S = 1.0

# Carrier-sense realism.  OLD called _window_state() on the decode window,
# which reports "idle" for anything shorter than 3*QUIET_TAIL_S -- so for 3s
# after every listen_start reset (our own TX ending, or a msg decode) the node
# is blind and keys up over whatever is on the air.  It also needs a full
# 1s tail of the peer's signal before it reads "busy".
OLD_BLIND_S  = 3 * QUIET_TAIL_S
OLD_DETECT_S = QUIET_TAIL_S
# NEW senses the newest audio only (bounded below by our own TX end), so it
# needs no warm-up window -- just enough fresh tail to judge.
NEW_DETECT_S = 0.3


class Node:
    def __init__(self, role, fixed, drop):
        self.sm = HandshakeSM(role, wait_probe_s=20.0, wait_msg_s=20.0)
        self.role, self.fixed, self.drop = role, fixed, drop
        self.queue = []          # actions awaiting transmission
        self.tx_end = None       # busy until this time
        self.tx_what = None
        self.deferred = False    # carrier sense fired for the head action
        self.turn_t0 = None      # when the head action's turn began
        self.inbox = []          # frames that landed, awaiting a decode poll
        self.last_poll = -99.0
        self.intervals = []
        self.listen_t0 = 0.0     # last listen_start reset (CS warm-up anchor)

    def senses(self, p, t):
        """Does our carrier-sense detector currently see the peer's TX?"""
        if p.tx_end is None:
            return False
        started = t - p.intervals[-1][0]
        if self.fixed:
            return started >= NEW_DETECT_S
        if t - self.listen_t0 < OLD_BLIND_S:      # decode window too short
            return False
        return started >= OLD_DETECT_S

    def bat(self):
        return bytes(range(20)) if self.role == "initiator" else bytes(range(50, 77))


def run(fixed: bool, drop: set, horizon: float = 400.0):
    ini, rsp = Node("initiator", fixed, drop), Node("responder", fixed, drop)
    peer = {id(ini): rsp, id(rsp): ini}
    sent = {}
    collisions = []
    t = 0.0
    ini.sm.start(t); rsp.sm.start(t)

    def key(n, a):
        base = f"{n.role[0]}:{'probe' if a[0]=='send_probe' else 'msg%d'%a[1]}"
        i = sent.get(base, 0); sent[base] = i + 1
        return f"{base}:{i}"

    while t < horizon:
        for n in (ini, rsp):
            n.queue.extend(x for x in n.sm.actions if x[0] in ("send_probe", "send_msg"))
            n.sm.actions = []

        for n in (ini, rsp):
            p = peer[id(n)]
            if n.tx_end is not None:                       # mid-transmission
                if t >= n.tx_end:
                    lo, hi = n.intervals[-1]
                    clash = any(not (hi <= a or b <= lo) for a, b in p.intervals)
                    if clash:
                        collisions.append((round(lo, 1), round(hi, 1), n.role, n.tx_what))
                    elif n.tx_key not in n.drop:
                        p.inbox.append((n.tx_act, n.bat(), t))
                    n.tx_end = None
                    n.listen_t0 = t          # listen_start reset after own TX
                    if n.fixed:
                        n.sm.hold_off(t - n.turn_t0)
                    n.turn_t0 = None
                continue

            # -- decode poll (a frame is only decodable once fully received) --
            if n.sm.expecting() and t - n.last_poll >= POLL_S:
                n.last_poll = t
                for f in list(n.inbox):
                    act, bat, _ = f
                    exp = n.sm.expecting() or ""
                    if act[0] == "send_probe" and "probe" in exp:
                        n.inbox.remove(f); n.sm.heard_probe(bat, t); break
                    if act[0] == "send_msg" and "msg" in exp:
                        n.inbox.remove(f); n.sm.heard_msg(act[1], act[2], act[3], t)
                        n.listen_t0 = t      # listen_start reset after msg decode
                        break

            if not n.queue:
                n.sm.tick(t)
                continue

            a = n.queue[0]
            if n.turn_t0 is None:
                n.turn_t0 = t
            if n.senses(p, t):                             # carrier sense
                n.deferred = True
                continue
            if n.fixed and n.deferred:
                # drain the receiver before keying up: a frame that landed
                # while we deferred may have made this action obsolete
                consumed = False
                for f in list(n.inbox):
                    act, bat, _ = f
                    exp = n.sm.expecting() or ""
                    if act[0] == "send_probe" and "probe" in exp:
                        n.inbox.remove(f); n.sm.heard_probe(bat, t); consumed = True; break
                    if act[0] == "send_msg" and "msg" in exp:
                        n.inbox.remove(f); n.sm.heard_msg(act[1], act[2], act[3], t)
                        n.listen_t0 = t; consumed = True; break
                if consumed:
                    n.queue.clear()
                    n.sm.hold_off(t - n.turn_t0)
                    n.turn_t0 = None; n.deferred = False
                    continue
            n.queue.pop(0)
            n.deferred = False
            n.tx_act = a
            n.tx_key = key(n, a)
            n.tx_what = "probe" if a[0] == "send_probe" else _MSG_NAMES[a[1]]
            dur = AIRTIME["probe" if a[0] == "send_probe" else a[1]]
            n.tx_end = t + dur
            n.intervals.append((t, n.tx_end))

        for n in (ini, rsp):
            if n.tx_end is None:
                n.sm.tick(t)
        if (ini.sm.done or ini.sm.failed) and (rsp.sm.done or rsp.sm.failed):
            break
        t += DT

    ok = (ini.sm.done and rsp.sm.done
          and ini.sm.my_tx_bat == ini.bat() and rsp.sm.my_tx_bat == rsp.bat())
    return ok, collisions, ini, rsp, t


CASES = [
    ("clean",                set()),
    ("drop BAT_REPORT",      {"r:msg1:0"}),
    ("drop R probe",         {"r:probe:0"}),
    ("drop BAT_ACK",         {"i:msg2:0"}),
    ("drop REPORT + ACK",    {"r:msg1:0", "i:msg2:0"}),
    ("drop GO",              {"r:msg3:0"}),
]

def main() -> int:
    totals = {}
    for label, fixed in (("OLD (pre-fix, expected to collide)", False),
                         ("NEW (current binding)", True)):
        print(f"=== {label} ===")
        tot = bad = 0
        for name, plan in CASES:
            ok, coll, ini, rsp, t = run(fixed, plan)
            tot += len(coll)
            bad += (0 if ok else 1)
            print(f"  [{name:18}] {'ok  ' if ok else 'FAIL'} "
                  f"collisions={len(coll)}  t={t:5.1f}s  "
                  f"ini={ini.sm.state} rsp={rsp.sm.state}")
            for c in coll:
                print(f"        overlap {c[0]}-{c[1]}s  {c[2]} {c[3]}")
        print(f"  total collisions: {tot}   non-converging cases: {bad}\n")
        totals[fixed] = (tot, bad)

    old_coll, _ = totals[False]
    new_coll, new_bad = totals[True]
    ok = new_coll == 0 and new_bad == 0
    if old_coll == 0:
        print("WARNING: the pre-fix model no longer collides -- this "
              "regression has stopped exercising the bug it guards")
    print("COLLIDE REGRESS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
