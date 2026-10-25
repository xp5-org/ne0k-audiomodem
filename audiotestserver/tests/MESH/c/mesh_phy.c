/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* mesh_phy.c -- see mesh_phy.h. Port of PhyPolicy in ../meshphy.py.
 *
 * Comments tagged "OTA <date>" document a specific failure the adjacent
 * logic exists to prevent; keep them when editing this file.
 */
#include "mesh_phy.h"
#include <string.h>

/* css_modem TYPE_DATA / TYPE_ACK, duplicated here (not #included) because
 * this file has no CSS dependency by design -- see mesh_phy.h's header
 * comment on why the packet header lives inline at call sites, not in a
 * shared library. Keep in sync with tests/CSS/c/'s port of css_modem.py if
 * one exists by the time this drifts. */
#define MESHPHY_CM_TYPE_DATA 0x1
#define MESHPHY_CM_TYPE_ACK  0x2

static void act_link_down(meshphy_action *a, meshphy_reason reason,
                          int had_link)
{
    a->kind = MESHPHY_ACT_LINK_DOWN;
    a->role = MESHPHY_ROLE_INITIATOR; /* unused for this kind */
    a->reneg = 0;
    a->reason = reason;
    a->had_link = had_link;
    a->detail = 0.0;
}

/* Give up on OFDM: back to CSS, clean slate for a future COMMANDED upgrade.
 * BATs are discarded on purpose -- the trigger is evidence the channel
 * changed, so a stale-BAT fast re-upgrade would just fail again. DATA
 * counters reset so the pending payload resends immediately (over CSS, since
 * the selector now sees link_up=0). Returns 1 (always writes exactly one
 * action). */
static int drop_link(meshphy *p, meshphy_reason reason, double detail,
                     meshphy_action *acts)
{
    int had_link = p->link_up;
    p->link_up = 0;
    p->link_up_t = MESHPHY_TIME_UNSET;
    p->last_good_rx_t = MESHPHY_TIME_UNSET;
    p->hs_result_set = 0;
    p->upgrade_wanted = 0;
    p->data_tries = 0;
    p->data_last_tx = MESHPHY_TIME_UNSET;
    p->hs_req_seq_set = 0;
    p->req_tries = 0;
    p->req_last_tx = MESHPHY_TIME_UNSET;
    act_link_down(&acts[0], reason, had_link);
    acts[0].detail = detail;
    return 1;
}

/* Still trying to get OFDM up -- DATA waits for the outcome. */
static int upgrade_pending(const meshphy *p)
{
    return p->upgrade_wanted && !p->link_up;
}

/* Drop the link and re-run the handshake (KEEPING the learned margin_adj)
 * rather than falling back to CSS. ofdm_only re-handshakes on its own clock;
 * css_then_ofdm re-arms the CSS-commanded upgrade. Returns 1. */
static int renegotiate(meshphy *p, meshphy_reason reason, double detail,
                       meshphy_action *acts)
{
    int n = drop_link(p, reason, detail, acts);
    p->evm_hist_len = 0;
    p->hs_last_attempt = MESHPHY_TIME_UNSET; /* ofdm_only: re-handshake now */
    if (p->phy_mode == MESHPHY_CSS_THEN_OFDM) {
        p->upgrade_wanted = 1; /* re-REQ over CSS */
    }
    return n;
}

int meshphy_init(meshphy *p, int node_id, int peer_id, meshphy_mode phy_mode,
                 int initiator,
                 double hs_req_delay_s, int req_retries, double req_ack_wait_s,
                 int data_retries, double ack_wait_s, double link_silent_s,
                 double hs_retry_wait_s, int max_hs_attempts,
                 double evm_high, double evm_low, double evm_margin_step,
                 double evm_margin_max, int evm_min_frames, int evm_win,
                 int evm_climb, int evm_feedback)
{
    if (phy_mode != MESHPHY_CSS_ONLY && phy_mode != MESHPHY_CSS_THEN_OFDM
        && phy_mode != MESHPHY_OFDM_ONLY
        && phy_mode != MESHPHY_OFDM_FULL_DUPLEX) {
        return -1;
    }
    memset(p, 0, sizeof(*p));
    p->node_id = node_id;
    p->peer_id = peer_id;
    p->phy_mode = phy_mode;
    p->hs_req_delay_s = hs_req_delay_s;
    p->req_retries = req_retries;
    p->req_ack_wait_s = req_ack_wait_s;
    p->data_retries = data_retries;
    p->ack_wait_s = ack_wait_s;

    /* LINK-SILENCE TIMEOUT: fixes link-state desync in ofdm_only. There is
     * no control plane, so when the sender gives up on a link and starts
     * re-probing, the RECEIVER has no way to be told. The default,
     * (1+data_retries)*ack_wait_s, deliberately expires BEFORE the sender's
     * own give-up (which needs the same retry set PLUS a final ack wait), so
     * the receiver is already re-armed when the sender's next probe lands
     * rather than being torn down by it. OTA 20260728: without this, a
     * receiver stayed on a dead link long enough to silently drop every one
     * of the sender's re-probes. */
    p->link_silent_s = (link_silent_s > 0.0) ? link_silent_s
        : (1 + p->data_retries) * p->ack_wait_s;
    p->link_up_t = MESHPHY_TIME_UNSET;
    p->last_good_rx_t = MESHPHY_TIME_UNSET;

    p->hs_retry_wait_s = hs_retry_wait_s;
    p->max_hs_attempts = max_hs_attempts;
    p->initiator = initiator;
    p->hs_attempts = 0;
    p->hs_last_attempt = MESHPHY_TIME_UNSET;
    p->gave_up = 0;
    /* the one flag that authorises sending OFDM_HS_REQ */
    p->upgrade_wanted = (phy_mode == MESHPHY_CSS_THEN_OFDM && initiator);
    p->link_up = 0;
    p->hs_result_set = 0;
    p->hs_req_seq_set = 0;
    p->req_tries = 0;
    p->req_last_tx = MESHPHY_TIME_UNSET;
    p->data_pending = 0;
    p->data_tries = 0;
    p->data_last_tx = MESHPHY_TIME_UNSET;
    p->got_data_ack = 0;
    p->sent_data_seq_count = 0;

    /* closed-loop EVM feedback: the probe-negotiated BAT is optimistic about
     * the post-probe channel (fade migration over the ~60s to data), so EVM
     * measured on decoded data frames is the one signal that tracks
     * reality. evm_high=70, not 45: the soft-Viterbi FEC delivers reliably
     * to ~60% EVM, and the real failure signal is a locked-but-RS-failed
     * frame (reported as 100%). EVM is bimodal -- ~50-60% delivered vs 100%
     * failed -- so the threshold must sit between them, or the loop backs
     * off links that are working fine. */
    p->evm_high = evm_high;
    p->evm_low = evm_low;
    p->evm_margin_step = evm_margin_step;
    p->evm_margin_max = evm_margin_max;
    p->evm_min_frames = evm_min_frames;
    p->evm_win = (evm_win > MESHPHY_EVM_HIST_CAP) ? MESHPHY_EVM_HIST_CAP
                                                   : evm_win;
    p->evm_climb = evm_climb;
    p->evm_feedback = evm_feedback; /* off by default: symmetric-rate links
                                      * handle asymmetry without the loop */
    p->margin_adj = 0.0;
    p->evm_hist_len = 0;
    return 0;
}

int meshphy_use_ofdm_for(const meshphy *p, int dst, int ptype)
{
    return p->link_up && dst == p->peer_id
        && (ptype == MESHPHY_CM_TYPE_DATA || ptype == MESHPHY_CM_TYPE_ACK);
}

int meshphy_final_phy_is_ofdm(const meshphy *p)
{
    return p->link_up;
}

void meshphy_on_good_rx(meshphy *p, double now)
{
    p->last_good_rx_t = now;
}

static double median_of(const double *v, int n)
{
    /* n <= MESHPHY_EVM_HIST_CAP (8): insertion sort into a local copy is
     * plenty for this size and needs no library sort. */
    double s[MESHPHY_EVM_HIST_CAP];
    int i, j;
    double key;
    memcpy(s, v, (size_t)n * sizeof(double));
    for (i = 1; i < n; i++) {
        key = s[i];
        j = i - 1;
        while (j >= 0 && s[j] > key) {
            s[j + 1] = s[j];
            j--;
        }
        s[j + 1] = key;
    }
    if (n % 2) {
        return s[n / 2];
    }
    return (s[n / 2 - 1] + s[n / 2]) / 2.0;
}

int meshphy_on_rx_evm(meshphy *p, double evm_pct, double now,
                      meshphy_action *acts, int max_acts)
{
    double med;
    (void)now; /* the reference only uses `now` to timestamp its
                * fallback_events log entry, which this port drops (see
                * mesh_phy.h: log reason+detail at the call site instead) */
    if (!p->evm_feedback || !p->link_up) {
        return 0;
    }
    if (p->evm_hist_len < p->evm_win) {
        p->evm_hist[p->evm_hist_len++] = evm_pct;
    } else {
        /* shift the ring left by one, drop oldest -- evm_win is tiny
         * (default 4, cap 8), so this is cheaper than a real ring index. */
        int i;
        for (i = 1; i < p->evm_hist_len; i++) {
            p->evm_hist[i - 1] = p->evm_hist[i];
        }
        p->evm_hist[p->evm_hist_len - 1] = evm_pct;
    }
    if (p->evm_hist_len < p->evm_min_frames) {
        return 0;
    }
    med = median_of(p->evm_hist, p->evm_hist_len);
    if (max_acts < 1) {
        return -1;
    }
    if (med > p->evm_high && p->margin_adj < p->evm_margin_max) {
        p->margin_adj += p->evm_margin_step;
        if (p->margin_adj > p->evm_margin_max) {
            p->margin_adj = p->evm_margin_max;
        }
        return renegotiate(p, MESHPHY_REASON_EVM_HIGH, med, acts);
    }
    if (p->evm_climb && med < p->evm_low && p->margin_adj > 0.0) {
        p->margin_adj -= p->evm_margin_step;
        if (p->margin_adj < 0.0) {
            p->margin_adj = 0.0;
        }
        return renegotiate(p, MESHPHY_REASON_EVM_LOW, med, acts);
    }
    return 0;
}

/* OFDM-only / OFDM-full-duplex: no CSS control plane at all. Roles are
 * STATIC, so the handshake needs no in-band negotiation -- each end enters it
 * on its own clock. There is no fallback PHY here: a failed handshake or an
 * unacked DATA RE-NEGOTIATES the link (a stale BAT is the likely cause)
 * instead of dropping to CSS. */
static int tick_ofdm_only(meshphy *p, double now, int data_in_outbox,
                          int hs_allowed, meshphy_action *acts, int max_acts)
{
    if (p->link_up == 0) {
        /* no link -> (re)run the handshake on a retry cadence, BOUNDED.
         * Without a cap the sender re-negotiates forever against a peer
         * that already completed and left, burning a full handshake
         * timeout per doomed cycle. After the cap it gives up: the payload
         * was likely delivered and only the ACK over the weak reverse
         * channel failed. */
        if (p->gave_up) {
            return 0;
        }
        if (p->hs_attempts >= p->max_hs_attempts) {
            p->gave_up = 1;
            if (max_acts < 1) {
                return -1;
            }
            acts[0].kind = MESHPHY_ACT_GAVE_UP;
            acts[0].role = MESHPHY_ROLE_INITIATOR;
            acts[0].reneg = 0;
            acts[0].reason = MESHPHY_REASON_GAVE_UP_MAX_HS_ATTEMPTS;
            acts[0].had_link = 0;
            acts[0].detail = (double)p->hs_attempts;
            return 1;
        }
        /* hs_allowed: this link's turn in the mesh-wide handshake schedule.
         * Only one link handshakes at a time -- the OFDM probe/BAT exchange
         * bypasses the data TX slots and blocks the loop, so without this,
         * two links collide. */
        if (hs_allowed && now >= p->hs_req_delay_s
            && (p->hs_last_attempt == MESHPHY_TIME_UNSET
                || now - p->hs_last_attempt >= p->hs_retry_wait_s)) {
            if (max_acts < 1) {
                return -1;
            }
            p->hs_last_attempt = now;
            p->hs_attempts++;
            p->hs_result_set = 0;
            acts[0].kind = MESHPHY_ACT_ENTER_HS;
            acts[0].role = p->initiator ? MESHPHY_ROLE_INITIATOR
                                        : MESHPHY_ROLE_RESPONDER;
            acts[0].reneg = (p->hs_attempts > 1);
            acts[0].reason = MESHPHY_REASON_NONE;
            acts[0].had_link = 0;
            acts[0].detail = 0.0;
            return 1;
        }
        return 0;
    }

    /* link up but SILENT -- nothing from this peer has decoded for a full
     * retry set (see link_silent_s). Only for a node with no DATA of its
     * own: a sender's own unacked-DATA rule below already covers it, and
     * keeps the more specific data_unacked_ofdm_only reason. This is what
     * lets a pure receiver notice the sender has moved on and re-arm as
     * responder instead of sitting on a link nobody else believes in. */
    if (!p->data_pending && p->link_silent_s > 0.0) {
        double ref = (p->last_good_rx_t != MESHPHY_TIME_UNSET)
            ? p->last_good_rx_t : p->link_up_t;
        if (ref != MESHPHY_TIME_UNSET && now - ref >= p->link_silent_s) {
            if (max_acts < 1) {
                return -1;
            }
            return renegotiate(p, MESHPHY_REASON_LINK_SILENT, now - ref,
                               acts);
        }
    }

    /* link up: DATA unacked for a full retry set -> assume the BAT went
     * stale and drop, so the next tick re-negotiates */
    if (p->data_pending && !p->got_data_ack
        && p->data_tries >= 1 + p->data_retries && !data_in_outbox
        && p->data_last_tx != MESHPHY_TIME_UNSET
        && now - p->data_last_tx >= p->ack_wait_s) {
        if (max_acts < 1) {
            return -1;
        }
        return drop_link(p, MESHPHY_REASON_DATA_UNACKED_OFDM_ONLY, 0.0,
                         acts);
    }

    if (p->data_pending && !p->got_data_ack
        && p->data_tries < 1 + p->data_retries && !data_in_outbox
        && (p->data_last_tx == MESHPHY_TIME_UNSET
            || now - p->data_last_tx >= p->ack_wait_s)) {
        if (max_acts < 1) {
            return -1;
        }
        acts[0].kind = MESHPHY_ACT_SEND_DATA;
        acts[0].role = MESHPHY_ROLE_INITIATOR;
        acts[0].reneg = 0;
        acts[0].reason = MESHPHY_REASON_NONE;
        acts[0].had_link = 0;
        acts[0].detail = 0.0;
        return 1;
    }
    return 0;
}

int meshphy_tick(meshphy *p, double now, int req_in_outbox,
                 int data_in_outbox, int hs_allowed,
                 meshphy_action *acts, int max_acts)
{
    int n = 0;

    if (MESHPHY_IS_OFDM_DATA_MODE(p->phy_mode)) {
        return tick_ofdm_only(p, now, data_in_outbox, hs_allowed, acts,
                              max_acts);
    }

    /* fallback trigger: DATA rode OFDM into silence for a full retry set.
     * `!data_in_outbox` matters: a final try that is QUEUED but has not
     * aired yet must get its full ack wait measured from ITS OWN transmit,
     * not timed from the previous try -- otherwise give-up can fire while
     * the ACK for the still-unaired try is still in flight. */
    if (p->link_up && p->data_pending && !p->got_data_ack
        && p->data_tries >= 1 + p->data_retries && !data_in_outbox
        && p->data_last_tx != MESHPHY_TIME_UNSET
        && now - p->data_last_tx >= p->ack_wait_s) {
        if (max_acts - n < 1) {
            return -1;
        }
        n += drop_link(p, MESHPHY_REASON_DATA_UNACKED_OVER_OFDM, 0.0,
                       &acts[n]);
    }

    /* give up wanting OFDM once the CSS REQ itself goes unanswered -- only
     * after the LAST retry has actually aired and waited its turn */
    if (upgrade_pending(p) && !p->hs_result_set
        && p->req_tries >= 1 + p->req_retries && !req_in_outbox
        && p->req_last_tx != MESHPHY_TIME_UNSET
        && now - p->req_last_tx >= p->req_ack_wait_s) {
        if (max_acts - n < 1) {
            return -1;
        }
        n += drop_link(p, MESHPHY_REASON_REQ_UNANSWERED, 0.0, &acts[n]);
    }

    /* (re)transmit the REQ over CSS */
    if (upgrade_pending(p) && !p->hs_result_set
        && now >= p->hs_req_delay_s && p->req_tries < 1 + p->req_retries
        && !req_in_outbox
        && (p->req_last_tx == MESHPHY_TIME_UNSET
            || now - p->req_last_tx >= p->req_ack_wait_s)) {
        if (max_acts - n < 1) {
            return -1;
        }
        acts[n].kind = MESHPHY_ACT_SEND_REQ;
        acts[n].role = MESHPHY_ROLE_INITIATOR;
        acts[n].reneg = 0;
        acts[n].reason = MESHPHY_REASON_NONE;
        acts[n].had_link = 0;
        acts[n].detail = 0.0;
        n++;
    }

    /* (re)transmit the pending DATA -- held back only while an upgrade
     * attempt is still in flight */
    if (p->data_pending && !p->got_data_ack && !upgrade_pending(p)
        && p->data_tries < 1 + p->data_retries && !data_in_outbox
        && (p->data_last_tx == MESHPHY_TIME_UNSET
            || now - p->data_last_tx >= p->ack_wait_s)) {
        if (max_acts - n < 1) {
            return -1;
        }
        acts[n].kind = MESHPHY_ACT_SEND_DATA;
        acts[n].role = MESHPHY_ROLE_INITIATOR;
        acts[n].reneg = 0;
        acts[n].reason = MESHPHY_REASON_NONE;
        acts[n].had_link = 0;
        acts[n].detail = 0.0;
        n++;
    }
    return n;
}

void meshphy_on_req_queued(meshphy *p, int seq)
{
    p->hs_req_seq_set = 1;
    p->hs_req_seq = seq;
    p->req_tries++;
}

void meshphy_on_data_queued(meshphy *p, int seq)
{
    if (p->sent_data_seq_count < MESHPHY_MAX_DATA_TRIES) {
        p->sent_data_seqs[p->sent_data_seq_count++] = seq;
    } else {
        /* Bound hit: drop the oldest tracked try. 1+data_retries should
         * never exceed MESHPHY_MAX_DATA_TRIES in practice (see mesh_phy.h);
         * this keeps the newest tries matchable if a caller ever configures
         * more retries than the cap. */
        int i;
        for (i = 1; i < MESHPHY_MAX_DATA_TRIES; i++) {
            p->sent_data_seqs[i - 1] = p->sent_data_seqs[i];
        }
        p->sent_data_seqs[MESHPHY_MAX_DATA_TRIES - 1] = seq;
    }
    p->data_tries++;
}

void meshphy_on_tx(meshphy *p, int tag_is_data, int tag_is_hs_req, double now)
{
    if (tag_is_data) {
        p->data_last_tx = now;
    } else if (tag_is_hs_req) {
        p->req_last_tx = now;
    }
}

int meshphy_on_ack(meshphy *p, int seq, double now,
                   meshphy_action *acts, int max_acts)
{
    int i;
    (void)now;
    if (p->hs_req_seq_set && seq == p->hs_req_seq && !p->link_up
        && !p->hs_result_set) {
        if (max_acts < 1) {
            return -1;
        }
        acts[0].kind = MESHPHY_ACT_ENTER_HS;
        acts[0].role = MESHPHY_ROLE_INITIATOR;
        acts[0].reneg = 0;
        acts[0].reason = MESHPHY_REASON_NONE;
        acts[0].had_link = 0;
        acts[0].detail = 0.0;
        return 1;
    }
    for (i = 0; i < p->sent_data_seq_count; i++) {
        if (p->sent_data_seqs[i] == seq) {
            p->got_data_ack = 1;
            break;
        }
    }
    return 0;
}

int meshphy_on_css_rx_from_peer(meshphy *p, double now, int to_me,
                                meshphy_action *acts, int max_acts)
{
    (void)now; /* see meshphy_on_rx_evm: only used for the dropped log
                * timestamp in the reference */
    /* Only meaningful when CSS is this link's fallback: in ofdm_only there
     * is nothing to fall back TO, so stray CSS must not tear the link down.
     *
     * `to_me` MUST be false for traffic the peer addressed to somebody else.
     * Only a packet aimed at US proves the peer gave up on OUR OFDM link; in
     * an N-node mesh a peer routinely speaks CSS to css_only third parties
     * while our link is perfectly healthy. Without this check, overhearing
     * a CSS ACK addressed to a third node tears down a working OFDM link. */
    if (p->phy_mode == MESHPHY_CSS_THEN_OFDM && to_me && p->link_up) {
        if (max_acts < 1) {
            return -1;
        }
        return drop_link(p, MESHPHY_REASON_CSS_RX_FROM_OFDM_LINKED_PEER, 0.0,
                         acts);
    }
    return 0;
}

int meshphy_on_hs_req(const meshphy *p, meshphy_action *acts, int max_acts)
{
    /* The node layer already drops the REQ without ACK when css_only;
     * this guard covers the same case defensively. */
    if (p->phy_mode != MESHPHY_CSS_THEN_OFDM) {
        return 0; /* css_only declines; ofdm_only never uses CSS */
    }
    if (max_acts < 1) {
        return -1;
    }
    acts[0].kind = MESHPHY_ACT_ARM_RESPONDER;
    acts[0].role = MESHPHY_ROLE_RESPONDER;
    acts[0].reneg = 0;
    acts[0].reason = MESHPHY_REASON_NONE;
    acts[0].had_link = 0;
    acts[0].detail = 0.0;
    return 1;
}

int meshphy_on_hs_result(meshphy *p, int hs_ok, int link_up, double now,
                         meshphy_action *acts, int max_acts)
{
    p->hs_result_set = 1;
    p->hs_result_ok = hs_ok;
    if (hs_ok && link_up) {
        p->link_up = 1;
        p->link_up_t = now;          /* link-silence clock starts here */
        p->last_good_rx_t = MESHPHY_TIME_UNSET;
        p->evm_hist_len = 0;         /* fresh BAT -> discard old EVM window */
        return 0;
    }
    if (max_acts < 1) {
        return -1;
    }
    return drop_link(p, MESHPHY_REASON_HANDSHAKE_FAILED, 0.0, acts);
}

int meshphy_on_detect(meshphy *p, double now, double score,
                      meshphy_action *acts, int max_acts)
{
    /* Re-arm the COMMANDED upgrade path (send REQ over CSS) -- never jump
     * straight onto OFDM. This only flips internal state (upgrade_wanted);
     * the reference returns no action here either (its `fallback_events`
     * log entry is a side-channel, not an action for the driver to carry
     * out) -- meshphy_tick's own SEND_REQ branch is what actually acts on
     * it, next call. `acts`/`max_acts` are accepted for interface symmetry
     * with the other on_*() entry points and so a future reason to act here
     * doesn't change this function's signature. */
    (void)acts;
    (void)max_acts;
    (void)now;
    if (p->phy_mode == MESHPHY_CSS_THEN_OFDM && !p->link_up
        && !p->upgrade_wanted) {
        p->upgrade_wanted = 1;
        p->hs_result_set = 0;
        (void)score; /* logged by the reference's fallback_events; a
                      * freestanding caller wanting this should log it
                      * itself at the call site, where `score` is already
                      * in hand. */
    }
    return 0;
}
