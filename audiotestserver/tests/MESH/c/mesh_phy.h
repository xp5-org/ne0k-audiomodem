/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* mesh_phy.h -- per-peer PHY selection policy (CSS/OFDM), freestanding C89.
 *
 * Port of PhyPolicy in ../meshphy.py.
 *
 * NOT IN THIS FILE: see tests/MESH/c/README.md.
 *   - the OFDM handshake FSM (HandshakeSM in ofdm_handshake.py).
 *   - the mesh-OFDM data-frame framing (MeshModulator/MeshDemodulator in
 *     meshlib.py); rides on ofdm_mod.c/ofdm_dem.c/rs255.c, already C89.
 *   - the CSS packet header (css_modem.py); pack/unpack inline at the call
 *     site rather than through a library here.
 *
 * CONTRACT: no malloc, no stdio, no globals. A meshphy struct is fixed-size,
 * so unlike ofdm_sync.h etc. there is no separate workspace-sizing call --
 * the caller just declares a `meshphy` on the stack or embeds one in a
 * per-peer record.
 *
 * ACTIONS: each entry point writes into a caller-owned `meshphy_action`
 * array and returns the count, or -1 if the buffer is too small to hold
 * every event the call could produce (see MESHPHY_MAX_ACTIONS).
 *
 * REASON CODES replace the Python f-string reasons with an enum plus a
 * numeric detail field -- see meshphy_action.detail.
 */
#ifndef MESH_PHY_H
#define MESH_PHY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Every mode below FULL_DUPLEX is HALF duplex: one node transmits, the other
 * listens, arbitrated by CSS carrier sense. Full duplex is a trained half
 * duplex link where carrier sense is then suspended for the paired link
 * only; see meshphy.py's module docstring. */
typedef enum {
    MESHPHY_CSS_ONLY = 0,
    MESHPHY_CSS_THEN_OFDM = 1,
    MESHPHY_OFDM_ONLY = 2,
    MESHPHY_OFDM_FULL_DUPLEX = 3
} meshphy_mode;

/* MESHPHY_OFDM_ONLY and MESHPHY_OFDM_FULL_DUPLEX share the same tick logic:
 * both mean OFDM is the data plane with no CSS fallback. Test with this
 * macro, not against the bare OFDM_ONLY mode; see OFDM_DATA_MODES in
 * meshphy.py. */
#define MESHPHY_IS_OFDM_DATA_MODE(m) \
    ((m) == MESHPHY_OFDM_ONLY || (m) == MESHPHY_OFDM_FULL_DUPLEX)

typedef enum {
    MESHPHY_ROLE_INITIATOR = 0,
    MESHPHY_ROLE_RESPONDER = 1
} meshphy_role;

typedef enum {
    MESHPHY_ACT_SEND_REQ = 0,     /* enqueue OFDM_HS_REQ (same seq on retry) */
    MESHPHY_ACT_SEND_DATA = 1,    /* enqueue the pending app DATA, fresh seq */
    MESHPHY_ACT_ENTER_HS = 2,     /* run the blocking OFDM handshake now */
    MESHPHY_ACT_ARM_RESPONDER = 3,/* peer asked for a handshake; enter as
                                    * responder once the CSS ACK has aired
                                    * (driver-level timing) */
    MESHPHY_ACT_LINK_DOWN = 4,    /* informational: link/want just dropped */
    MESHPHY_ACT_GAVE_UP = 5       /* ofdm_only: bounded re-negotiation expired */
} meshphy_act_kind;

typedef enum {
    MESHPHY_REASON_NONE = 0,
    MESHPHY_REASON_DATA_UNACKED_OVER_OFDM,
    MESHPHY_REASON_DATA_UNACKED_OFDM_ONLY,
    MESHPHY_REASON_REQ_UNANSWERED,
    MESHPHY_REASON_HANDSHAKE_FAILED,
    MESHPHY_REASON_CSS_RX_FROM_OFDM_LINKED_PEER,
    MESHPHY_REASON_LINK_SILENT,
    MESHPHY_REASON_EVM_HIGH,
    MESHPHY_REASON_EVM_LOW,
    MESHPHY_REASON_GAVE_UP_MAX_HS_ATTEMPTS,
    MESHPHY_REASON_OFDM_DETECTED
} meshphy_reason;

/* Every entry point that can act writes at most this many actions.  Bump it
 * (and re-audit every call site) before adding a branch that fires more. */
#define MESHPHY_MAX_ACTIONS 2

/* Bound on tracked DATA seqs for the currently pending payload -- see the
 * sent_data_seqs field comment. 1+data_retries must not exceed this. */
#define MESHPHY_MAX_DATA_TRIES 8

typedef struct {
    meshphy_act_kind kind;
    meshphy_role role;      /* ENTER_HS only */
    int reneg;              /* ENTER_HS only: 1 if this is a re-negotiation,
                              * i.e. hs_attempts > 1 (ofdm_only) */
    meshphy_reason reason;  /* LINK_DOWN / GAVE_UP only */
    int had_link;           /* LINK_DOWN only: was a link actually up? */
    double detail;          /* reason-specific number for logging: seconds
                              * silent, EVM percent, hs_attempts count, or
                              * detector score. 0 when the reason carries none. */
} meshphy_action;

typedef struct {
    /* identity / config, set at init and not touched again */
    int node_id;
    int peer_id;
    meshphy_mode phy_mode;
    double hs_req_delay_s;
    int req_retries;
    double req_ack_wait_s;
    int data_retries;
    double ack_wait_s;
    double link_silent_s;
    double hs_retry_wait_s;
    int max_hs_attempts;
    int initiator;              /* boolean */
    double evm_high;
    double evm_low;
    double evm_margin_step;
    double evm_margin_max;
    int evm_min_frames;
    int evm_win;
    int evm_climb;              /* boolean */
    int evm_feedback;           /* boolean, default off -- see meshphy.py */

    /* link / handshake state */
    int link_up;                /* boolean: NULL vs dict in the reference.
                                  * Link contents (negotiated BAT etc.) are
                                  * the driver's problem; this only tracks
                                  * whether a link is up. */
    double link_up_t;           /* valid iff link_up; MESHPHY_TIME_UNSET else */
    double last_good_rx_t;      /* valid iff link_up and at least one rx;
                                  * MESHPHY_TIME_UNSET else */
    int hs_result_set;          /* boolean: has a handshake outcome been
                                  * recorded since the last _drop()? */
    int hs_result_ok;           /* valid iff hs_result_set */
    int hs_req_seq_set;
    int hs_req_seq;
    int req_tries;
    double req_last_tx;         /* MESHPHY_TIME_UNSET if never sent */
    int hs_attempts;
    double hs_last_attempt;     /* MESHPHY_TIME_UNSET if never attempted */
    int gave_up;                /* boolean, ofdm_only terminal state */
    int upgrade_wanted;         /* boolean: the ONLY flag authorising a REQ */

    /* data-plane retry state: one pending payload at a time, matching the
     * reference -- a node queues the next payload only after this one
     * either delivers or the link drops. */
    int data_pending;           /* boolean: driver has app DATA queued */
    int data_tries;
    double data_last_tx;        /* MESHPHY_TIME_UNSET if never sent */
    int got_data_ack;           /* boolean */
    /* Ring of seqs sent for the CURRENTLY pending payload (fresh seq per
     * try, see on_data_queued). Bounded, unlike the reference's unbounded
     * set: only this payload's outstanding tries need matching against an
     * ACK. A 4-bit seq wraps mod 16, so "remember everything sent" and
     * "remember this payload's tries" agree in practice. See mesh_phy.c. */
    int sent_data_seqs[MESHPHY_MAX_DATA_TRIES];
    int sent_data_seq_count;

    /* EVM feedback loop, off by default (evm_feedback=0) */
    double margin_adj;
    double evm_hist[8];         /* ring, capacity >= any sane evm_win */
    int evm_hist_len;
} meshphy;

/* Sentinel for "no timestamp recorded yet" -- driver clocks are seconds since
 * an arbitrary run epoch and never negative, so -1.0 is unambiguous without a
 * separate has-fired flag for every timer field. */
#define MESHPHY_TIME_UNSET (-1.0)

/* evm_hist ring capacity; evm_win must not exceed this. */
#define MESHPHY_EVM_HIST_CAP 8

/* Zero-init `p` and fill in config from the arguments. Every field is
 * explicit; a freestanding caller has no keyword defaults to fall back on.
 *
 * `initiator` is not defaulted from node_id (the reference's `node_id == 1`
 * fallback is a Python test-rig convention) -- callers must pass it.
 *
 * `link_silent_s`: pass a value <= 0.0 to take the reference's derived
 * default, (1 + data_retries) * ack_wait_s.
 *
 * Returns -1 if phy_mode is out of range, else 0. */
int meshphy_init(meshphy *p, int node_id, int peer_id, meshphy_mode phy_mode,
                 int initiator,
                 double hs_req_delay_s, int req_retries, double req_ack_wait_s,
                 int data_retries, double ack_wait_s, double link_silent_s,
                 double hs_retry_wait_s, int max_hs_attempts,
                 double evm_high, double evm_low, double evm_margin_step,
                 double evm_margin_max, int evm_min_frames, int evm_win,
                 int evm_climb, int evm_feedback);

/* -- transmit-point selector -------------------------------------------- */
/* `ptype` is a css_modem TYPE_* code (TYPE_DATA=0x1, TYPE_ACK=0x2); pass it
 * straight from the packet header, this file has no CSS dependency. */
int meshphy_use_ofdm_for(const meshphy *p, int dst, int ptype);

/* 1 = "ofdm", 0 = "css" (meshphy.py's `final_phy` property). */
int meshphy_final_phy_is_ofdm(const meshphy *p);

/* -- state feedback from the driver -------------------------------------- */

/* A packet from this peer decoded (any type). Do NOT call this for a frame
 * that locked but failed RS -- that is exactly the condition link_silent_s
 * exists to catch (see meshphy.py:on_good_rx). */
void meshphy_on_good_rx(meshphy *p, double now);

/* Feed the EVM of a just-decoded OFDM data frame (report ~100.0 for a locked-
 * but-RS-failed frame). Writes at most 1 action. Returns the count, or 0 if
 * evm_feedback is off / no link / not enough history yet. */
int meshphy_on_rx_evm(meshphy *p, double evm_pct, double now,
                      meshphy_action *acts, int max_acts);

/* -- driver entry points -------------------------------------------------- */

/* Advance time. `req_in_outbox`/`data_in_outbox`: the driver's own outbox
 * still holds a queued-but-unaired copy of the REQ/DATA -- see the
 * data_unacked_over_ofdm comment in mesh_phy.c for why this matters.
 * `hs_allowed`: this link's turn in the mesh-wide handshake schedule
 * (ofdm_only / ofdm_full_duplex only; ignored otherwise).
 * Writes at most MESHPHY_MAX_ACTIONS actions, returns the count, or -1 if
 * max_acts is too small. */
int meshphy_tick(meshphy *p, double now, int req_in_outbox,
                 int data_in_outbox, int hs_allowed,
                 meshphy_action *acts, int max_acts);

void meshphy_on_req_queued(meshphy *p, int seq);
void meshphy_on_data_queued(meshphy *p, int seq);

/* tag_is_data / tag_is_hs_req: true if the driver's just-aired TX was the
 * pending DATA / the pending OFDM_HS_REQ respectively. The reference
 * switches on a string prefix ("data:"/"ofdmhs:"); a freestanding driver has
 * no such string and should just know which queue it drained. */
void meshphy_on_tx(meshphy *p, int tag_is_data, int tag_is_hs_req, double now);

/* Writes at most 1 action. Returns the count. */
int meshphy_on_ack(meshphy *p, int seq, double now,
                   meshphy_action *acts, int max_acts);

/* `to_me` MUST be 0 for CSS traffic the peer addressed to somebody else --
 * see the css_rx_from_ofdm_linked_peer comment in mesh_phy.c. Writes at most
 * 1 action. Returns the count. */
int meshphy_on_css_rx_from_peer(meshphy *p, double now, int to_me,
                                meshphy_action *acts, int max_acts);

/* Writes at most 1 action. Returns the count. */
int meshphy_on_hs_req(const meshphy *p, meshphy_action *acts, int max_acts);

/* `hs_ok`: handshake succeeded. `link_up`: pass 1 iff hs_ok and the driver
 * actually has a link record to install (mirrors `link is not None` guarding
 * `res.get("ok")` in the reference -- both must hold for the link to come
 * up). Writes at most 1 action. Returns the count. */
int meshphy_on_hs_result(meshphy *p, int hs_ok, int link_up, double now,
                         meshphy_action *acts, int max_acts);

/* Passive OFDM-activity detector hit: re-arms upgrade_wanted (state only,
 * see mesh_phy.c) -- meshphy_tick's SEND_REQ branch is what acts on it, on
 * the next call. Writes no actions; always returns 0. `acts`/`max_acts` are
 * accepted for interface symmetry with the other on_*() entry points. */
int meshphy_on_detect(meshphy *p, double now, double score,
                      meshphy_action *acts, int max_acts);

#ifdef __cplusplus
}
#endif
#endif
