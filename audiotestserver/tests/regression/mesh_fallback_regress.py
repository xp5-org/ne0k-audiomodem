# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
mesh_fallback_regress.py -- offline regression for the CSS<->OFDM fallback
state machine (meshphy.PhyPolicy) wired to the REAL packet/MAC layers
(css_modem_node.ModemNode + css_modem_daemon.ModemDaemon), over a scripted
boolean per-PHY channel.  No DSP, no audio -- seconds to run.

Run:  python tests/regression/mesh_fallback_regress.py     (exit 0 = pass)

Cases:
  1. normal upgrade        REQ -> hs -> DATA+ACK over OFDM (rewiring sanity)
  2. fallback convergence  link up, then OFDM goes deaf: initiator exhausts
                           retries, drops to CSS, resends; responder infers
                           the downgrade from the CSS DATA; ACK rides CSS
  3. handshake failure     hs fails -> DATA still delivered over CSS
  4. req unanswered        peer is css_only (accept_ofdm_hs=False): REQ never
                           ACKed -> give up -> DATA over CSS
  5. css_only initiator    never emits a REQ, DATA straight over CSS
  6. no autonomous re-entry + commanded re-entry via the detector hook
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

_TESTS_ROOT = Path(__file__).parent.parent
for _p in (_TESTS_ROOT, _TESTS_ROOT / "CSS", _TESTS_ROOT / "MESH"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import css_modem as cm                                          # noqa: E402
from css_modem_node import ModemNode                            # noqa: E402
from css_modem_daemon import ModemDaemon                        # noqa: E402
from meshphy import PhyPolicy                                   # noqa: E402

FS = 48000
PAYLOAD = b"hello-fallback"


class SimNode:
    """Mirror of the meshmodem_node.py driver loop, minus audio."""

    def __init__(self, node_id: int, peer_id: int, phy_mode: str,
                 **pol_kw):
        self.id = node_id
        self.node = ModemNode(node_id, fs=FS, turnaround_samples=0)
        self.node.accept_ofdm_hs = (phy_mode != "css_only")
        fw = types.SimpleNamespace(
            node=self.node,
            cfg={"id": node_id,
                 "phy": {"fs": FS, "symbol_dur": 0.3, "symbol_gap_frac": 0.0}},
            save=lambda: None)
        self.daemon = ModemDaemon(fw, cw_s=1.0, ack_stagger_s=0.5,
                                  rechirp_min_s=1e6, rechirp_max_s=1e6,
                                  first_beacon_stagger_s=1e6)
        self.policy = PhyPolicy(node_id, peer_id, phy_mode=phy_mode, **pol_kw)
        self.hs_role_pending = None
        self.req_raw = self.req_tag = None
        self.got_data_payload = None
        self.tx_log: list[tuple[float, str, str]] = []   # (t, phy, tag)
        self.req_sends = 0

    def handle_policy(self, acts, now, sim):
        for a in acts:
            k = a["kind"]
            if k == "send_req":
                if self.req_raw is None:
                    self.req_raw, self.req_tag = \
                        self.node.request_ofdm_hs(self.policy.peer_id)
                self.daemon.enqueue(self.req_raw, self.req_tag)
                self.policy.on_req_queued(int(self.req_tag.split(":")[2]))
                self.req_sends += 1
            elif k == "send_data":
                raw, tag = self.node.send_data(self.policy.peer_id, PAYLOAD)
                self.daemon.enqueue(raw, tag)
                self.policy.on_data_queued(int(tag.split(":")[2]))
            elif k == "enter_hs":
                sim.run_handshake(self, now)
            elif k == "arm_responder":
                self.hs_role_pending = "responder"
            elif k == "link_down":
                self.req_raw = self.req_tag = None

    def handle_events(self, now, sim):
        for e in self.daemon.drain_events():
            if e["kind"] == "ofdm_hs_req" and e["peer"] == self.policy.peer_id:
                self.handle_policy(self.policy.on_hs_req(now), now, sim)
            elif e["kind"] == "ack" and e["peer"] == self.policy.peer_id:
                self.handle_policy(self.policy.on_ack(e["seq"], now), now, sim)
            elif e["kind"] == "deliver":
                self.got_data_payload = (e.get("packet") or {}).get("payload")


class Sim:
    def __init__(self, mode1="css_then_ofdm", mode2="css_then_ofdm",
                 **pol_kw):
        self.n1 = SimNode(1, 2, mode1, **pol_kw)
        self.n2 = SimNode(2, 1, mode2, **pol_kw)
        self.n1.policy.data_pending = True      # node 1 owns the demo payload
        self.css_ok = True          # CSS delivery (both directions)
        self.ofdm_ok = True         # OFDM delivery (both directions)
        self.hs_ok = True           # handshake outcome when entered
        self.t = 0.0

    def peer_of(self, node):
        return self.n2 if node is self.n1 else self.n1

    def run_handshake(self, initiator_node, now):
        """Both ends leave the air for the blocking handshake; outcome is the
        scripted hs_ok.  The responder side runs iff it armed (its ACK made it
        out), which is exactly the OTA precondition."""
        res = {"ok": self.hs_ok,
               "state": "done" if self.hs_ok else "timeout_i_wait_probe"}
        link = {"sim": True} if self.hs_ok else None
        initiator_node.handle_policy(
            initiator_node.policy.on_hs_result(res, link, now), now, self)
        peer = self.peer_of(initiator_node)
        if peer.hs_role_pending == "responder":
            peer.hs_role_pending = None
            peer.handle_policy(
                peer.policy.on_hs_result(res, dict(link) if link else None,
                                         now), now, self)

    def step(self, dt=0.5):
        self.t += dt
        now = self.t
        for node in (self.n1, self.n2):
            node.handle_policy(node.policy.tick(
                now,
                req_in_outbox=any(t.startswith("ofdmhs:")
                                  for _, t, _ in node.daemon.outbox),
                data_in_outbox=any(t.startswith("data:")
                                   for _, t, _ in node.daemon.outbox)),
                now, self)
            rt = node.daemon.poll_tx(now, False)
            if rt is None:
                continue
            raw, tag = rt
            pkt = cm.unpack_packet(raw)
            use_ofdm = node.policy.use_ofdm_for(pkt["dst"], pkt["type"])
            phy = "ofdm" if use_ofdm else "css"
            node.tx_log.append((now, phy, tag))
            node.daemon.on_tx(tag, int(now * FS))
            node.policy.on_tx(tag, now)
            delivered = self.ofdm_ok if use_ofdm else self.css_ok
            if not delivered:
                continue
            peer = self.peer_of(node)
            if phy == "css" and pkt["src"] == peer.policy.peer_id:
                # implicit downgrade BEFORE the MAC sees the packet
                peer.handle_policy(
                    peer.policy.on_css_rx_from_peer(now), now, self)
            peer.daemon.handle_frame(
                {"bytes_hex": raw.hex(), "at_sample": int(now * FS)}, now)
            peer.handle_events(now, self)

    def run(self, seconds, dt=0.5, until=None):
        end = self.t + seconds
        while self.t < end:
            self.step(dt)
            if until is not None and until():
                return True
        return until() if until is not None else True


_ok_all = True


def check(name, cond):
    global _ok_all
    _ok_all &= bool(cond)
    print(f"  [{name:44}] {'PASS' if cond else 'FAIL'}")


def phys_of(node, tag_prefix):
    return [p for _, p, t in node.tx_log if t.startswith(tag_prefix)]


def case1_normal_upgrade():
    s = Sim()
    done = s.run(600, until=lambda: s.n1.policy.got_data_ack)
    check("1 normal: DATA acked", done)
    check("1 normal: DATA rode OFDM", phys_of(s.n1, "data:") == ["ofdm"])
    check("1 normal: ACK rode OFDM", "ofdm" in phys_of(s.n2, "ack:"))
    check("1 normal: both links up", s.n1.policy.link is not None
          and s.n2.policy.link is not None)
    return s


def case2_fallback_convergence():
    s = Sim()
    s.run(600, until=lambda: s.n1.policy.link is not None
          and s.n2.policy.link is not None)
    s.ofdm_ok = False               # peer walked out of OFDM range
    s.n1.policy.got_data_ack = False
    s.n1.policy.data_pending = True
    s.n1.policy.data_tries = 0      # fresh payload after the link came up
    s.n1.policy.sent_data_seqs.clear()
    done = s.run(1200, until=lambda: s.n1.policy.got_data_ack)
    check("2 fallback: DATA acked after fallback", done)
    d_phys = phys_of(s.n1, "data:")
    check("2 fallback: OFDM tried then CSS",
          "ofdm" in d_phys and d_phys[-1] == "css")
    check("2 fallback: initiator dropped link",
          s.n1.policy.link is None and any(
              e["reason"] == "data_unacked_over_ofdm"
              for e in s.n1.policy.fallback_events))
    check("2 fallback: responder inferred downgrade",
          s.n2.policy.link is None and any(
              e["reason"] == "css_rx_from_ofdm_linked_peer"
              for e in s.n2.policy.fallback_events))
    check("2 fallback: final ACK rode CSS", phys_of(s.n2, "ack:")[-1] == "css")
    check("2 fallback: payload delivered",
          s.n2.got_data_payload == PAYLOAD)
    return s


def case3_handshake_failure():
    s = Sim()
    s.hs_ok = False
    done = s.run(1200, until=lambda: s.n1.policy.got_data_ack)
    check("3 hs-fail: DATA acked over CSS", done
          and phys_of(s.n1, "data:") and
          set(phys_of(s.n1, "data:")) == {"css"})
    check("3 hs-fail: no link either side",
          s.n1.policy.link is None and s.n2.policy.link is None)
    req_after_fail = [t for t, p, tag in s.n1.tx_log
                      if tag.startswith("ofdmhs:")
                      and s.n1.policy.fallback_events
                      and t > s.n1.policy.fallback_events[0]["t"]]
    check("3 hs-fail: no REQ after giving up", not req_after_fail)


def case4_req_unanswered():
    s = Sim(mode2="css_only", req_ack_wait_s=30.0)
    done = s.run(1200, until=lambda: s.n1.policy.got_data_ack)
    check("4 css_only peer: DATA acked over CSS", done
          and set(phys_of(s.n1, "data:")) == {"css"})
    check("4 css_only peer: REQ never ACKed / no hs",
          s.n1.policy.link is None and s.n1.policy.hs_result is None)
    check("4 css_only peer: gave up wanting OFDM",
          not s.n1.policy.upgrade_wanted and any(
              e["reason"] == "req_unanswered"
              for e in s.n1.policy.fallback_events))


def case5_css_only_initiator():
    s = Sim(mode1="css_only", mode2="css_then_ofdm")
    done = s.run(600, until=lambda: s.n1.policy.got_data_ack)
    check("5 css_only self: DATA acked over CSS", done
          and set(phys_of(s.n1, "data:")) == {"css"})
    check("5 css_only self: zero REQs sent", s.n1.req_sends == 0)


def case6_reentry(s2: Sim):
    # continue from case 2's converged fallback state: both on CSS
    reqs_before = s2.n1.req_sends
    s2.run(500)
    check("6 re-entry: NO autonomous REQ on CSS",
          s2.n1.req_sends == reqs_before and s2.n2.req_sends == 0)
    # peer moved back into OFDM range; the passive detector hears it
    s2.ofdm_ok = True
    s2.n1.handle_policy(s2.n1.policy.on_detect(s2.t, 0.9), s2.t, s2)
    done = s2.run(600, until=lambda: s2.n1.policy.link is not None
                  and s2.n2.policy.link is not None)
    check("6 re-entry: detect -> REQ -> link back up", done
          and s2.n1.req_sends > reqs_before)


def case7_no_giveup_while_queued():
    # OTA 20260720: give-up fired while the final REQ retry sat in the
    # outbox, timed from the PREVIOUS try's transmit -- the retry then aired
    # and its ACK landed on a node that no longer wanted the link.
    pol = PhyPolicy(1, 2, req_retries=1, req_ack_wait_s=10.0,
                    hs_req_delay_s=0.0)
    assert any(a["kind"] == "send_req" for a in pol.tick(1.0))
    pol.on_req_queued(5); pol.on_tx("ofdmhs:2:5", 2.0)      # try 1 airs
    assert any(a["kind"] == "send_req" for a in pol.tick(20.0))
    pol.on_req_queued(5)                                     # try 2 QUEUED
    acts = pol.tick(60.0, req_in_outbox=True)                # not yet aired
    check("7 give-up: holds while REQ queued",
          not any(a["kind"] == "link_down" for a in acts)
          and pol.upgrade_wanted)
    pol.on_tx("ofdmhs:2:5", 61.0)                            # try 2 airs
    acts = pol.tick(65.0)
    ok_wait = not any(a["kind"] == "link_down" for a in acts)
    acts = pol.tick(72.0)                                    # 11s > 10s wait
    check("7 give-up: fires after last try's own wait",
          ok_wait and any(a["kind"] == "link_down"
                          and a["reason"] == "req_unanswered" for a in acts))

    # same race on the OFDM DATA fallback trigger
    pol2 = PhyPolicy(1, 2, data_retries=0, ack_wait_s=10.0)
    pol2.link = {"sim": True}
    pol2.data_pending = True
    assert any(a["kind"] == "send_data" for a in pol2.tick(1.0))
    pol2.on_data_queued(3)                                   # queued, unaired
    acts = pol2.tick(100.0, data_in_outbox=True)
    check("7 fallback: holds while DATA queued",
          not any(a["kind"] == "link_down" for a in acts)
          and pol2.link is not None)
    pol2.on_tx("data:2:3", 101.0)
    acts = pol2.tick(112.0)
    check("7 fallback: fires after DATA's own wait",
          any(a["kind"] == "link_down"
              and a["reason"] == "data_unacked_over_ofdm" for a in acts))


def case8_third_party_css_keeps_link():
    # OTA 20260720: node 2 overheard node 1's CSS ACK addressed to node 3
    # (a css_only peer) and tore down a HEALTHY OFDM link to node 1.
    pol = PhyPolicy(2, 1)
    pol.link = {"sim": True}
    acts = pol.on_css_rx_from_peer(10.0, to_me=False)   # peer -> third party
    check("8 third-party CSS keeps OFDM link",
          not acts and pol.link is not None)
    acts = pol.on_css_rx_from_peer(20.0, to_me=True)    # peer -> us
    check("8 CSS addressed to us drops link",
          pol.link is None and any(a["kind"] == "link_down"
                                   and a["reason"] == "css_rx_from_ofdm_linked_peer"
                                   for a in acts))


def case9_ofdm_only():
    # no CSS control plane: static roles, handshake entered directly, and
    # NO fallback -- failures re-negotiate the link instead
    ini = PhyPolicy(1, 2, phy_mode="ofdm_only", initiator=True,
                    hs_req_delay_s=5.0, hs_retry_wait_s=30.0,
                    data_retries=1, ack_wait_s=10.0)
    ini.data_pending = True
    res = PhyPolicy(2, 1, phy_mode="ofdm_only", initiator=False,
                    hs_req_delay_s=5.0, hs_retry_wait_s=30.0)

    check("9 ofdm_only: no REQ before delay", not ini.tick(1.0))
    a_i, a_r = ini.tick(6.0), res.tick(6.0)
    check("9 ofdm_only: enters hs directly, static roles",
          [x["kind"] for x in a_i] == ["enter_hs"]
          and a_i[0]["role"] == "initiator"
          and a_r and a_r[0]["role"] == "responder"
          and not any(x["kind"] == "send_req" for x in a_i + a_r))

    # failed handshake must RETRY, never fall back to CSS
    ini.on_hs_result({"ok": False, "state": "timeout"}, None, 8.0)
    check("9 ofdm_only: no link after failed hs", ini.link is None)
    check("9 ofdm_only: waits retry cadence", not ini.tick(20.0))
    check("9 ofdm_only: re-attempts handshake",
          any(x["kind"] == "enter_hs" for x in ini.tick(40.0))
          and ini.hs_attempts == 2)

    # link up -> DATA rides OFDM
    ini.on_hs_result({"ok": True, "state": "done"}, {"sim": True}, 41.0)
    check("9 ofdm_only: DATA sent once linked",
          any(x["kind"] == "send_data" for x in ini.tick(42.0))
          and ini.use_ofdm_for(2, cm.TYPE_DATA))
    ini.on_data_queued(1); ini.on_tx("data:2:1", 43.0)
    ini.tick(60.0); ini.on_data_queued(2); ini.on_tx("data:2:2", 61.0)

    # unacked DATA re-negotiates (stale BAT), it does NOT drop to CSS
    acts = ini.tick(80.0)
    check("9 ofdm_only: unacked DATA drops link to re-negotiate",
          any(x["kind"] == "link_down"
              and x["reason"] == "data_unacked_ofdm_only" for x in acts)
          and ini.link is None and ini.data_pending)
    check("9 ofdm_only: never selects CSS for DATA",
          not ini.use_ofdm_for(2, cm.TYPE_DATA) and ini.final_phy == "css")

    # stray CSS must not disturb an ofdm_only link
    ini.on_hs_result({"ok": True, "state": "done"}, {"sim": True}, 90.0)
    check("9 ofdm_only: stray CSS keeps link",
          not ini.on_css_rx_from_peer(91.0, to_me=True)
          and ini.link is not None)


def case10_evm_feedback():
    # closed loop: high EVM on decoded frames -> add margin + re-negotiate;
    # very low EVM with margin already added -> shed margin to climb the rate.
    pol = PhyPolicy(2, 1, phy_mode="ofdm_only", initiator=False,
                    evm_feedback=True,
                    evm_high=45.0, evm_low=18.0, evm_margin_step=3.0,
                    evm_margin_max=9.0, evm_min_frames=2, evm_win=4)
    pol.link = {"sim": True}

    check("10 evm: one high reading holds (needs min_frames)",
          not pol.on_rx_evm(60.0, 1.0) and pol.margin_adj == 0.0)
    acts = pol.on_rx_evm(58.0, 2.0)                 # 2nd high -> median high
    check("10 evm: sustained high EVM backs off + re-negotiates",
          pol.margin_adj == 3.0 and pol.link is None
          and any(a["kind"] == "link_down"
                  and a["reason"].startswith("evm_high") for a in acts))
    check("10 evm: margin carried into next handshake",
          pol.evm_events[-1]["dir"] == "backoff")

    # re-handshake brings link back up; EVM still high -> step again, cap holds
    pol.on_hs_result({"ok": True}, {"sim": True}, 10.0)
    pol.on_rx_evm(70.0, 11.0); pol.on_rx_evm(70.0, 12.0)
    check("10 evm: second backoff steps margin to cap",
          pol.margin_adj == 6.0)
    pol.on_hs_result({"ok": True}, {"sim": True}, 20.0)
    pol.on_rx_evm(70.0, 21.0); pol.on_rx_evm(70.0, 22.0)
    pol.on_hs_result({"ok": True}, {"sim": True}, 30.0)
    pol.on_rx_evm(70.0, 31.0)
    acts = pol.on_rx_evm(70.0, 32.0)
    check("10 evm: margin capped, no further backoff at cap",
          pol.margin_adj == 9.0 and not acts)

    # now a clean, very-low-EVM channel -> climb (shed margin)
    pol.on_hs_result({"ok": True}, {"sim": True}, 40.0)
    pol.on_rx_evm(10.0, 41.0)
    acts = pol.on_rx_evm(10.0, 42.0)
    check("10 evm: low EVM climbs (sheds margin) + re-negotiates",
          pol.margin_adj == 6.0 and pol.link is None
          and any(a["reason"].startswith("evm_low") for a in acts))

    # in-band EVM (between low and high) -> stable, no churn
    pol.on_hs_result({"ok": True}, {"sim": True}, 50.0)
    m0 = pol.margin_adj
    check("10 evm: mid-band EVM is stable",
          not pol.on_rx_evm(30.0, 51.0) and not pol.on_rx_evm(30.0, 52.0)
          and pol.margin_adj == m0 and pol.link is not None)

    # climb disabled -> low EVM does nothing
    pol2 = PhyPolicy(2, 1, phy_mode="ofdm_only", evm_feedback=True,
                     evm_climb=False, evm_min_frames=2)
    pol2.link = {"sim": True}; pol2.margin_adj = 6.0
    pol2.on_rx_evm(5.0, 1.0)
    check("10 evm: climb disabled keeps margin",
          not pol2.on_rx_evm(5.0, 2.0) and pol2.margin_adj == 6.0)

    # paused by default: no feedback unless explicitly enabled
    pol3 = PhyPolicy(2, 1, phy_mode="ofdm_only", evm_min_frames=2)
    pol3.link = {"sim": True}
    pol3.on_rx_evm(100.0, 1.0)
    check("10 evm: feedback OFF by default (paused)",
          not pol3.on_rx_evm(100.0, 2.0) and pol3.link is not None
          and pol3.margin_adj == 0.0)


def case11_ofdm_only_giveup():
    # OTA 20260721: the receiver delivered + exited while the sender kept
    # re-negotiating into an empty room (reverse ACK channel too weak),
    # burning a full handshake timeout per doomed cycle = minutes of dead
    # air.  The sender must give up after max_hs_attempts and finish.
    pol = PhyPolicy(1, 2, phy_mode="ofdm_only", initiator=True,
                    hs_req_delay_s=0.0, hs_retry_wait_s=0.0, max_hs_attempts=3)
    pol.data_pending = True
    t = 0.0
    entered = 0
    for _ in range(20):
        t += 1.0
        acts = pol.tick(t)
        for a in acts:
            if a["kind"] == "enter_hs":
                entered += 1
                pol.on_hs_result({"ok": False, "state": "timeout"}, None, t)
    check("11 giveup: bounded to max_hs_attempts",
          entered == 3 and pol.hs_attempts == 3)
    check("11 giveup: emits gave_up then goes quiet",
          pol.gave_up and pol.tick(t + 5) == [])
    check("11 giveup: reneg flag set after first attempt",
          any(e.get("reason") == "gave_up_max_hs_attempts"
              for e in pol.fallback_events))

    # a re-negotiation carries the reneg flag (drives the fast timeout)
    pol2 = PhyPolicy(1, 2, phy_mode="ofdm_only", initiator=True,
                     hs_req_delay_s=0.0, hs_retry_wait_s=0.0, max_hs_attempts=5)
    pol2.data_pending = True
    a1 = [a for a in pol2.tick(1.0) if a["kind"] == "enter_hs"][0]
    pol2.on_hs_result({"ok": False}, None, 1.0)
    a2 = [a for a in pol2.tick(2.0) if a["kind"] == "enter_hs"][0]
    check("11 giveup: first=full, retries=reneg",
          a1.get("reneg") is False and a2.get("reneg") is True)


def case12_hs_schedule_gate():
    # the mesh-wide handshake schedule (driver) gates entry via hs_allowed;
    # OTA 20260721 three nodes probed at once because entry wasn't gated.
    pol = PhyPolicy(1, 2, phy_mode="ofdm_only", initiator=True,
                    hs_req_delay_s=0.0, hs_retry_wait_s=0.0)
    pol.data_pending = True
    acts = pol.tick(1.0, hs_allowed=False)
    check("12 hs-sched: gated tick does NOT enter handshake",
          not any(a["kind"] == "enter_hs" for a in acts)
          and pol.hs_attempts == 0)
    acts = pol.tick(2.0, hs_allowed=True)
    check("12 hs-sched: allowed tick enters handshake",
          any(a["kind"] == "enter_hs" for a in acts) and pol.hs_attempts == 1)

    # give-up bound still respected regardless of gating
    pol2 = PhyPolicy(1, 2, phy_mode="ofdm_only", initiator=True,
                     hs_req_delay_s=0.0, hs_retry_wait_s=0.0, max_hs_attempts=2)
    pol2.data_pending = True
    for k in range(6):
        for a in pol2.tick(float(k), hs_allowed=True):
            if a["kind"] == "enter_hs":
                pol2.on_hs_result({"ok": False}, None, float(k))
    check("12 hs-sched: gating + give-up coexist",
          pol2.gave_up and pol2.hs_attempts == 2)


def case13_link_silence_desync():
    # OTA 20260728: macdesktop's DATA frames clipped laptop's mic, so every
    # frame LOCKED and failed RS.  The sender exhausted its retries, dropped
    # the link at t=220 and re-probed three times; the RECEIVER still believed
    # in its link (nothing in ofdm_only can tell it otherwise) so it never
    # re-armed as responder, and all three handshakes died on i_wait_probe.
    # A responder's link must expire on its own when nothing decodes.
    LINK_UP, ACK_WAIT, RETRIES = 100.0, 35.0, 2
    silent_s = (1 + RETRIES) * ACK_WAIT          # 105s

    def responder():
        p = PhyPolicy(2, 1, phy_mode="ofdm_only", initiator=False,
                      ack_wait_s=ACK_WAIT, data_retries=RETRIES,
                      hs_req_delay_s=0.0, hs_retry_wait_s=0.0)
        p.on_hs_result({"ok": True, "peer_rate": 44100}, {"bat": 1}, LINK_UP)
        return p

    pol = responder()
    check("13 silence: receiver link up after hs", pol.link is not None)

    # holds the link while the sender is still inside its retry set
    acts = pol.tick(LINK_UP + silent_s - 1.0)
    check("13 silence: holds link before timeout",
          pol.link is not None and not acts)

    acts = pol.tick(LINK_UP + silent_s + 0.1)
    check("13 silence: drops a link nothing decodes on",
          pol.link is None
          and any(a["kind"] == "link_down" for a in acts)
          and any(e["reason"].startswith("link_silent")
                  for e in pol.fallback_events))

    # ...and immediately re-arms as RESPONDER, which is the whole point:
    # the sender's next probe must find somebody listening
    acts = pol.tick(LINK_UP + silent_s + 0.2, hs_allowed=True)
    check("13 silence: re-enters handshake as responder",
          any(a["kind"] == "enter_hs" and a["role"] == "responder"
              for a in acts))

    # a decoded frame resets the clock -- a healthy slow link must NOT expire
    pol2 = responder()
    t = LINK_UP
    for _ in range(6):                      # 6 * 60s = 360s, all healthy
        t += 60.0
        pol2.on_good_rx(t)
        pol2.tick(t)
    check("13 silence: good RX keeps a slow link alive",
          pol2.link is not None and not pol2.fallback_events)
    check("13 silence: expires once RX stops",
          pol2.tick(t + silent_s + 0.1) and pol2.link is None)

    # lock-then-RS-failure must NOT count as a good RX: that is exactly the
    # condition the timeout exists to escape (it is what the OTA run did)
    pol3 = responder()
    t = LINK_UP
    for _ in range(5):
        t += 20.0
        pol3.on_rx_evm(100.0, t)            # locked, RS failed
        pol3.tick(t)
    check("13 silence: RS-failures do not reset the clock",
          pol3.tick(LINK_UP + silent_s + 0.1) and pol3.link is None)

    # a SENDER keeps its own, more specific reason (its data path covers it)
    snd = PhyPolicy(1, 2, phy_mode="ofdm_only", initiator=True,
                    ack_wait_s=ACK_WAIT, data_retries=RETRIES,
                    hs_req_delay_s=0.0, hs_retry_wait_s=0.0)
    snd.on_hs_result({"ok": True, "peer_rate": 44100}, {"bat": 1}, LINK_UP)
    snd.data_pending = True
    snd.tick(LINK_UP + silent_s + 0.1)
    check("13 silence: sender unaffected (no link_silent drop)",
          not any(e["reason"].startswith("link_silent")
                  for e in snd.fallback_events))

    # opt-out
    off = PhyPolicy(2, 1, phy_mode="ofdm_only", initiator=False,
                    ack_wait_s=ACK_WAIT, data_retries=RETRIES,
                    link_silent_s=0.0, hs_req_delay_s=0.0)
    off.on_hs_result({"ok": True, "peer_rate": 44100}, {"bat": 1}, LINK_UP)
    off.tick(LINK_UP + 10000.0)
    check("13 silence: link_silent_s=0 disables the timeout",
          off.link is not None)


def case14_outbox_dedupe_and_drop():
    """Queue hygiene: no duplicate packets, and obsolete ones can be dropped.

    A CSS packet costs ~33s of air at OTA params, and the outbox is FIFO, so
    anything queued ahead of a data frame delays it by that much.  Observed
    20260813_040605 (css_then_ofdm, mutual data_map): the OFDM link came up at
    t=337.7 with ``outbox=['ofdmhs:2:0', 'ack:2:0', 'ack:2:0', 'data:2:2']`` --
    a now-obsolete handshake REQUEST plus a duplicate ACK, together ~100s of
    dead CSS air sitting in front of a 1.7s OFDM data frame.
    """
    d = ModemDaemon.__new__(ModemDaemon)   # queueing only; no PHY needed
    d.outbox = []

    check("14 dedupe: first enqueue accepted", d.enqueue(b"x", "ack:2:0"))
    check("14 dedupe: identical tag refused while queued",
          d.enqueue(b"x", "ack:2:0") is False)
    check("14 dedupe: queue holds one copy",
          [t for _, t, _ in d.outbox] == ["ack:2:0"])
    check("14 dedupe: a different packet still queues",
          d.enqueue(b"y", "data:2:2"))

    # A re-ACK is legitimate ONCE the first has aired and left the queue --
    # dedupe must be against the queue, never against history.
    d.outbox = [e for e in d.outbox if e[1] != "ack:2:0"]
    check("14 dedupe: re-ACK allowed after the first aired",
          d.enqueue(b"x", "ack:2:0"))

    # drop(): obsolete OFDM_HS_REQ for a peer whose link just came up.
    d.outbox = []
    for tag in ("ofdmhs:2:0", "ack:2:0", "data:2:2", "ofdmhs:3:1"):
        d.enqueue(b"p", tag)
    dropped = d.drop("ofdmhs:2:")
    check("14 drop: removes only the matching peer's REQ",
          dropped == ["ofdmhs:2:0"])
    check("14 drop: leaves everything else, in order",
          [t for _, t, _ in d.outbox] == ["ack:2:0", "data:2:2", "ofdmhs:3:1"])
    check("14 drop: is idempotent", d.drop("ofdmhs:2:") == [])


def main() -> int:
    case1_normal_upgrade()
    s2 = case2_fallback_convergence()
    case3_handshake_failure()
    case4_req_unanswered()
    case5_css_only_initiator()
    case6_reentry(s2)
    case7_no_giveup_while_queued()
    case8_third_party_css_keeps_link()
    case9_ofdm_only()
    case10_evm_feedback()
    case11_ofdm_only_giveup()
    case12_hs_schedule_gate()
    case13_link_silence_desync()
    case14_outbox_dedupe_and_drop()
    print("REGRESS", "PASS" if _ok_all else "FAIL")
    return 0 if _ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
