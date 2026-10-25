# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
meshphy.py -- per-peer PHY selection policy: CSS <-> OFDM with CSS fallback.

Named WITHOUT an underscore so the rig's `<test>_<role>.py` discovery does not
register it as a bogus role (meshlib pattern).

Extracted from the meshmodem_node.py driver loop so the fallback state machine
is testable offline (tests/regression/mesh_fallback_regress.py) with no audio.
The driver keeps ALL audio/MAC scaffolding -- carrier sense, outbox, blocking
TX, the actual handshake -- and asks this object what to do:

    CSS is home base.  OFDM is an upgrade a node may WANT (`upgrade_wanted`);
    wanting it authorises sending OFDM_HS_REQ over CSS.  Any give-up -- DATA
    unacked over OFDM, handshake failure, REQ retries exhausted -- clears the
    want and (if up) the link, so traffic reverts to CSS.  The want is re-armed
    ONLY by an external signal: an incoming REQ from the peer (CSS-commanded)
    or the passive OFDM-activity detector (meshlib.detect_ofdm_activity).
    There is deliberately NO timed autonomous re-upgrade: the fallback exists
    because the devices may have been moved apart with nobody at the controls.

    Both ends converge without extra control messages: the CSS decoder keeps
    running while OFDM is up, so ANY CSS packet from the OFDM-linked peer is
    proof the peer is back on CSS -- the receiver drops its own link BEFORE
    the packet is processed, which also makes the auto-ACK ride CSS.

Driver contract (all time in driver-relative seconds):
    tick(now, req_in_outbox=..., data_in_outbox=...) -> [actions]
    on_ack(seq, now)          ACK heard (either PHY)     -> [actions]
    on_css_rx_from_peer(now)  any CSS packet from peer   -> [actions]
    on_hs_req(now)            peer's OFDM_HS_REQ arrived -> [actions]
    on_hs_result(res, link, now)  handshake finished     -> [actions]
    on_detect(now, score)     passive detector hit       -> [actions]
    on_req_queued(seq) / on_data_queued(seq)   driver enqueued the packet
    on_tx(tag, now)           driver actually transmitted `tag`
    use_ofdm_for(dst, ptype)  transmit-point PHY selector

Actions (dicts, `kind`):
    send_req            enqueue an OFDM_HS_REQ to the peer (same seq on retry)
    send_data           enqueue the pending app DATA (fresh seq per try)
    enter_hs role=...   run the blocking OFDM handshake now ("initiator")
    arm_responder       peer asked for a handshake; enter as responder once
                        the CSS ACK has aired (driver-level timing)
    link_down reason=.. informational: link/want just dropped (log it)
"""
from __future__ import annotations

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

import css_modem as cm

# Every mode below "ofdm_full_duplex" is HALF duplex: one node transmits, the
# other listens, turn by turn, arbitrated by CSS carrier sense.
#
#   css_only          CSS for everything.
#   css_then_ofdm     CSS home base, upgrade a link to OFDM on request.
#   ofdm_only         OFDM half duplex, no CSS data plane.
#   ofdm_full_duplex  OFDM FULL duplex (FDD): the band is split by DIRECTION
#                     around a guard gap and BOTH ends transmit at once.
#
# Full duplex is a different animal, not another rung on the same ladder: the
# band has exactly two halves, so it is meaningful only between exactly TWO
# endpoints, and simultaneous transmission is the precise opposite of what
# carrier sense assumes.  The link is therefore TRAINED half duplex (the normal
# probe + in-band BAT handshake, CSMA-arbitrated) and only the DATA phase runs
# duplex, with carrier sense suspended for that paired link -- the same
# train-then-go-full-duplex order a wireline modem uses.  meshmodem_node
# refuses to start this mode with more than two participants.
PHY_MODES = ("css_only", "css_then_ofdm", "ofdm_only", "ofdm_full_duplex")

# Modes whose DATA plane is OFDM, i.e. "OFDM data plane, no CSS fallback" --
# true of full duplex as well as ofdm_only, so membership must be tested
# against this tuple. Comparing to the bare "ofdm_only" string instead drops
# ofdm_full_duplex into the css_then_ofdm branches, the wrong half of the
# decision.
OFDM_DATA_MODES = ("ofdm_only", "ofdm_full_duplex")

class PhyPolicy:
    def __init__(self, node_id: int, peer_id: int, *,
                 phy_mode: str = "css_then_ofdm",
                 initiator: bool | None = None,
                 hs_req_delay_s: float = 10.0,
                 req_retries: int = 3, req_ack_wait_s: float = 120.0,
                 data_retries: int = 2, ack_wait_s: float = 35.0,
                 link_silent_s: float | None = None,
                 hs_retry_wait_s: float = 20.0, max_hs_attempts: int = 4,
                 evm_high: float = 70.0, evm_low: float = 18.0,
                 evm_margin_step: float = 3.0, evm_margin_max: float = 18.0,
                 evm_min_frames: int = 2, evm_win: int = 4,
                 evm_climb: bool = True, evm_feedback: bool = False):
        if phy_mode not in PHY_MODES:
            raise ValueError(f"phy_mode {phy_mode!r} not in {PHY_MODES}")
        self.node_id = int(node_id)
        self.peer_id = int(peer_id)
        self.phy_mode = phy_mode
        self.hs_req_delay_s = float(hs_req_delay_s)
        self.req_retries = int(req_retries)
        self.req_ack_wait_s = float(req_ack_wait_s)
        self.data_retries = int(data_retries)
        self.ack_wait_s = float(ack_wait_s)

        # LINK-SILENCE TIMEOUT -- the cure for link-state desync in ofdm_only.
        # There is no control plane, so when the sender gives up on a link and
        # starts re-probing, the RECEIVER has no way to be told.  Its own link
        # was immortal: `_drop` only ever fires on data_unacked (a pure
        # receiver has no pending data) or handshake_failed (it never runs
        # one), so `link is None` -- the sole re-entry condition below -- was
        # unreachable, stranding the receiver's stale link against a sender
        # that has already torn its own link down and moved on to re-probing.
        #
        # Default (1+data_retries)*ack_wait_s deliberately expires BEFORE the
        # sender's own give-up (which needs the same retry set PLUS a final
        # ack wait), so the receiver is already re-armed when the sender's
        # next probe lands rather than being torn down by it.
        self.link_silent_s = (float(link_silent_s) if link_silent_s is not None
                              else (1 + self.data_retries) * self.ack_wait_s)
        self.link_up_t: float | None = None
        self.last_good_rx_t: float | None = None

        self.hs_retry_wait_s = float(hs_retry_wait_s)
        self.max_hs_attempts = int(max_hs_attempts)
        self.initiator = (initiator if initiator is not None
                          else node_id == 1)
        self.hs_attempts = 0
        self.hs_last_attempt: float | None = None
        self.gave_up = False               # bounded re-negotiation (ofdm_only)
        # the one flag that authorises sending OFDM_HS_REQ
        self.upgrade_wanted = (phy_mode == "css_then_ofdm" and self.initiator)
        self.link: dict | None = None      # None = CSS
        self.hs_result: dict | None = None
        self.hs_req_seq: int | None = None
        self.req_tries = 0
        self.req_last_tx: float | None = None
        self.data_pending = False          # driver sets when it has app DATA
        self.data_tries = 0
        self.data_last_tx: float | None = None
        self.got_data_ack = False
        self.sent_data_seqs: set[int] = set()
        self.fallback_events: list[dict] = []

        # closed-loop EVM feedback: the probe-negotiated BAT is optimistic
        # about the post-probe channel (fade migration over the ~60s to data),
        # so the ONE thing that tracks reality is the EVM measured on decoded
        # data frames.  High EVM -> the BAT is too hot for the live channel ->
        # add allocation margin and re-negotiate (lower, survivable rate);
        # very low EVM with margin already added -> we over-corrected, shed
        # margin to climb the rate back up (the "go faster when closer" case).
        # `margin_adj` (dB) is the learned delta the driver adds to the base
        # handshake margin on the NEXT (re)negotiation for this peer.
        # evm_high 70 (not 45): the soft-Viterbi FEC DELIVERS reliably to ~60%
        # EVM, and the real failure signal is a locked-but-RS-failed frame
        # (reported as 100%).  The EVM distribution is bimodal -- ~50-60%
        # delivered vs 100% failed -- so the threshold sits between them, or
        # the loop backs off links that are working fine.
        self.evm_high = float(evm_high)
        self.evm_low = float(evm_low)
        self.evm_margin_step = float(evm_margin_step)
        self.evm_margin_max = float(evm_margin_max)
        self.evm_min_frames = int(evm_min_frames)
        self.evm_win = int(evm_win)
        self.evm_climb = bool(evm_climb)
        # paused by default: symmetric-rate links handle asymmetry without it
        self.evm_feedback = bool(evm_feedback)
        self.margin_adj = 0.0
        self.evm_hist: list[float] = []
        self.evm_events: list[dict] = []

    # -- transmit-point selector -------------------------------------------- #
    def use_ofdm_for(self, dst: int, ptype: int) -> bool:
        return (self.link is not None and dst == self.peer_id
                and ptype in (cm.TYPE_DATA, cm.TYPE_ACK))

    @property
    def final_phy(self) -> str:
        return "ofdm" if self.link is not None else "css"

    # -- internal ----------------------------------------------------------- #
    def _drop(self, reason: str, now: float) -> list[dict]:
        """Give up on OFDM: back to CSS, clean slate for a future COMMANDED
        upgrade.  BATs are discarded on purpose -- the trigger is evidence the
        channel changed, so a stale-BAT fast re-upgrade would just fail again.
        DATA counters reset so the pending payload resends immediately (over
        CSS, since the selector now sees link=None)."""
        had_link = self.link is not None
        self.link = None
        self.link_up_t = None
        self.last_good_rx_t = None
        self.hs_result = None
        self.upgrade_wanted = False
        self.data_tries = 0
        self.data_last_tx = None
        self.hs_req_seq = None
        self.req_tries = 0
        self.req_last_tx = None
        self.fallback_events.append({"t": round(now, 1), "reason": reason,
                                     "had_link": had_link})
        return [{"kind": "link_down", "reason": reason, "had_link": had_link}]

    def _upgrade_pending(self) -> bool:
        """Still trying to get OFDM up -- DATA waits for the outcome."""
        return self.upgrade_wanted and self.link is None

    def _renegotiate(self, reason: str, now: float) -> list[dict]:
        """Drop the link and re-run the handshake (KEEPING the learned
        margin_adj), rather than falling back to CSS.  ofdm_only re-handshakes
        on its own clock; css_then_ofdm re-arms the CSS-commanded upgrade."""
        acts = self._drop(reason, now)
        self.evm_hist = []
        self.hs_last_attempt = None            # ofdm_only: re-handshake now
        if self.phy_mode == "css_then_ofdm":
            self.upgrade_wanted = True         # re-REQ over CSS
        return acts

    def on_good_rx(self, now: float) -> None:
        """A packet from this peer DECODED (any type).  Resets the
        link-silence clock; a lock-then-RS-failure must NOT call this -- an
        undecodable frame is exactly the condition the timeout exists for."""
        self.last_good_rx_t = float(now)

    def on_rx_evm(self, evm_pct: float, now: float) -> list[dict]:
        """Feed the EVM of a just-decoded OFDM data frame.  A decode that
        LOCKED but failed RS should be reported as ~100% (worst case)."""
        if not self.evm_feedback or self.link is None:
            return []
        self.evm_hist.append(float(evm_pct))
        if len(self.evm_hist) > self.evm_win:
            self.evm_hist.pop(0)
        if len(self.evm_hist) < self.evm_min_frames:
            return []
        s = sorted(self.evm_hist)
        med = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1]
                                                 + s[len(s) // 2]) / 2.0
        if med > self.evm_high and self.margin_adj < self.evm_margin_max:
            self.margin_adj = min(self.evm_margin_max,
                                  self.margin_adj + self.evm_margin_step)
            self.evm_events.append({"t": round(now, 1), "evm": round(med, 1),
                                    "dir": "backoff",
                                    "margin_adj": self.margin_adj})
            return self._renegotiate(f"evm_high_{med:.0f}pct", now)
        if self.evm_climb and med < self.evm_low and self.margin_adj > 0.0:
            self.margin_adj = max(0.0, self.margin_adj - self.evm_margin_step)
            self.evm_events.append({"t": round(now, 1), "evm": round(med, 1),
                                    "dir": "climb",
                                    "margin_adj": self.margin_adj})
            return self._renegotiate(f"evm_low_{med:.0f}pct", now)
        return []

    # -- driver entry points ------------------------------------------------ #
    def _tick_ofdm_only(self, now: float, *,
                        data_in_outbox: bool = False,
                        hs_allowed: bool = True) -> list[dict]:
        """OFDM-only: no CSS control plane at all.

        Roles are STATIC (derived from data_map), so the handshake needs no
        in-band negotiation -- each end enters it on its own clock, which
        removes the ~450s of CSS REQ/ACK airtime that precedes every OFDM
        experiment in css_then_ofdm.  There is no fallback PHY here: a failed
        handshake or an unacked DATA RE-NEGOTIATES the link (a stale BAT is
        the likely cause) instead of dropping to CSS.
        """
        acts: list[dict] = []

        # no link -> (re)run the handshake on a retry cadence, BOUNDED.
        # Without a cap the sender re-negotiates forever against a peer that
        # already completed and left, burning a full ~600s handshake timeout
        # per doomed cycle.  After the cap it gives up: the payload was likely
        # delivered; only the ACK back over the weak reverse channel failed.
        if self.link is None:
            if self.gave_up:
                return acts
            if self.hs_attempts >= self.max_hs_attempts:
                self.gave_up = True
                self.fallback_events.append(
                    {"t": round(now, 1), "reason": "gave_up_max_hs_attempts",
                     "hs_attempts": self.hs_attempts})
                return [{"kind": "gave_up", "peer": self.peer_id}]
            # hs_allowed: this link's turn in the mesh-wide handshake schedule
            # (only one link handshakes at a time -- the OFDM probe/BAT
            # exchange bypasses the data TX slots and blocks the loop, so
            # without this two links collide).
            if (hs_allowed and now >= self.hs_req_delay_s
                    and (self.hs_last_attempt is None
                         or now - self.hs_last_attempt >= self.hs_retry_wait_s)):
                self.hs_last_attempt = now
                self.hs_attempts += 1
                self.hs_result = None
                acts.append({"kind": "enter_hs",
                             "role": ("initiator" if self.initiator
                                      else "responder"),
                             "reneg": self.hs_attempts > 1})
            return acts

        # link up but SILENT -- nothing from this peer has decoded for a full
        # retry set (see link_silent_s).  Only for a node with no DATA of its
        # own: a sender's own unacked-DATA rule below already covers it, and
        # keeps the more specific `data_unacked_ofdm_only` reason.  This is
        # what lets a pure receiver notice the sender has moved on and re-arm
        # as responder instead of sitting on a link nobody else believes in.
        if (self.link is not None and not self.data_pending
                and self.link_silent_s > 0):
            ref = (self.last_good_rx_t if self.last_good_rx_t is not None
                   else self.link_up_t)
            if ref is not None and now - ref >= self.link_silent_s:
                return acts + self._renegotiate(
                    f"link_silent_{now - ref:.0f}s", now)

        # link up: DATA unacked for a full retry set -> assume the BAT went
        # stale and drop, so the next tick re-negotiates
        if (self.data_pending and not self.got_data_ack
                and self.data_tries >= 1 + self.data_retries
                and not data_in_outbox
                and self.data_last_tx is not None
                and now - self.data_last_tx >= self.ack_wait_s):
            return acts + self._drop("data_unacked_ofdm_only", now)

        if (self.data_pending and not self.got_data_ack
                and self.data_tries < 1 + self.data_retries
                and not data_in_outbox
                and (self.data_last_tx is None
                     or now - self.data_last_tx >= self.ack_wait_s)):
            acts.append({"kind": "send_data"})
        return acts

    def tick(self, now: float, *, req_in_outbox: bool = False,
             data_in_outbox: bool = False, hs_allowed: bool = True) -> list[dict]:
        if self.phy_mode in OFDM_DATA_MODES:
            return self._tick_ofdm_only(now, data_in_outbox=data_in_outbox,
                                        hs_allowed=hs_allowed)

        acts: list[dict] = []

        # fallback trigger: DATA rode OFDM into silence for a full retry set.
        # `not data_in_outbox` matters: a final try that is QUEUED but has
        # not aired yet must get its full ack wait measured from ITS OWN
        # transmit, not from the previous try's.
        if (self.link is not None and self.data_pending
                and not self.got_data_ack
                and self.data_tries >= 1 + self.data_retries
                and not data_in_outbox
                and self.data_last_tx is not None
                and now - self.data_last_tx >= self.ack_wait_s):
            acts += self._drop("data_unacked_over_ofdm", now)

        # give up wanting OFDM once the CSS REQ itself goes unanswered --
        # only after the LAST retry has actually aired and waited its turn
        if (self._upgrade_pending() and self.hs_result is None
                and self.req_tries >= 1 + self.req_retries
                and not req_in_outbox
                and self.req_last_tx is not None
                and now - self.req_last_tx >= self.req_ack_wait_s):
            acts += self._drop("req_unanswered", now)

        # (re)transmit the REQ over CSS
        if (self._upgrade_pending() and self.hs_result is None
                and now >= self.hs_req_delay_s
                and self.req_tries < 1 + self.req_retries
                and not req_in_outbox
                and (self.req_last_tx is None
                     or now - self.req_last_tx >= self.req_ack_wait_s)):
            acts.append({"kind": "send_req"})

        # (re)transmit the pending DATA -- held back only while an upgrade
        # attempt is still in flight
        if (self.data_pending and not self.got_data_ack
                and not self._upgrade_pending()
                and self.data_tries < 1 + self.data_retries
                and not data_in_outbox
                and (self.data_last_tx is None
                     or now - self.data_last_tx >= self.ack_wait_s)):
            acts.append({"kind": "send_data"})

        return acts

    def on_req_queued(self, seq: int) -> None:
        self.hs_req_seq = int(seq)
        self.req_tries += 1

    def on_data_queued(self, seq: int) -> None:
        self.sent_data_seqs.add(int(seq))
        self.data_tries += 1

    def on_tx(self, tag: str, now: float) -> None:
        if tag.startswith("data:"):
            self.data_last_tx = now
        elif tag.startswith("ofdmhs:"):
            self.req_last_tx = now

    def on_ack(self, seq: int, now: float) -> list[dict]:
        if (self.hs_req_seq is not None and seq == self.hs_req_seq
                and self.link is None and self.hs_result is None):
            return [{"kind": "enter_hs", "role": "initiator"}]
        if seq in self.sent_data_seqs:
            self.got_data_ack = True
        return []

    def on_css_rx_from_peer(self, now: float, *,
                            to_me: bool = True) -> list[dict]:
        """Called for a decoded CSS packet from the peer, BEFORE the packet is
        handed to the MAC -- so a dropped link makes the resulting auto-ACK
        (or REQ-ACK) ride CSS instead of the dead OFDM path.

        `to_me` MUST be false for traffic the peer addressed to somebody else.
        Only a packet aimed at US proves the peer gave up on OUR OFDM link;
        in an N-node mesh a peer routinely speaks CSS to css_only third
        parties while our link is perfectly healthy, and overhearing that
        traffic must not tear a working OFDM link down.
        """
        # only meaningful when CSS is this link's fallback: in ofdm_only there
        # is nothing to fall back TO, so stray CSS must not tear the link down
        if (self.phy_mode == "css_then_ofdm" and to_me
                and self.link is not None):
            return self._drop("css_rx_from_ofdm_linked_peer", now)
        return []

    def on_hs_req(self, now: float) -> list[dict]:
        """Peer commands an upgrade over CSS.  (The node layer already dropped
        the REQ without ACK when css_only -- this guard is belt-and-braces.)"""
        if self.phy_mode != "css_then_ofdm":
            return []          # css_only declines; ofdm_only never uses CSS
        return [{"kind": "arm_responder"}]

    def on_hs_result(self, res: dict, link: dict | None, now: float) -> list[dict]:
        self.hs_result = res
        if res.get("ok") and link is not None:
            self.link = link
            self.link_up_t = now    # link-silence clock starts here
            self.last_good_rx_t = None
            self.evm_hist = []      # fresh BAT -> discard old link's EVM window
            return []
        return self._drop("handshake_failed", now)

    def on_detect(self, now: float, score: float = 0.0) -> list[dict]:
        """Passive OFDM-activity hit: re-arm the COMMANDED upgrade path (send
        REQ over CSS) -- never jump straight onto OFDM."""
        if (self.phy_mode == "css_then_ofdm" and self.link is None
                and not self.upgrade_wanted):
            self.upgrade_wanted = True
            self.hs_result = None
            self.fallback_events.append({"t": round(now, 1),
                                         "reason": "ofdm_detected",
                                         "score": round(float(score), 3)})
        return []
