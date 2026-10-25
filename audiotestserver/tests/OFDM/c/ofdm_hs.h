/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_hs.h -- V.22-style two-way OFDM rate negotiation, freestanding C89.
 *
 * Port of HandshakeSM in ../ofdm_handshake.py: pure protocol logic (states,
 * retries, timeouts, CRC proof-of-receipt). No DSP, no audio; time is
 * injected via the `now` parameter.
 *
 * Protocol (half-duplex, turn-based; I = initiator/caller, R = responder):
 *
 *   I: PROBE                     R measures  -> BAT_I  (I's tx bit-loading)
 *   R: PROBE                     I measures  -> BAT_R  (R's tx bit-loading)
 *   R: MSG BAT_REPORT(BAT_I)     I learns its tx table
 *   I: MSG BAT_ACK(BAT_R, proof=crc(BAT_I))   R learns its table + receipt
 *   R: MSG GO(proof=crc(BAT_R))               I gets receipt R got BAT_R
 *   I: MSG FIN                                courtesy: R can stop early
 *   both DONE -- each end holds an agreed bit-loading table for BOTH
 *   directions.
 *
 * SCOPE OF THIS PORT.  Two Python features are not ported, both opt-in and
 * off by default in the reference:
 *
 *   - band_negotiation (MSG_BAND retraining). Off by default
 *     (`band_negotiation=False`); enabling it in the reference makes the
 *     probe-wait states also listen for messages, costing an extra trial
 *     decode per poll. Would need MSG_BAND handling in every state plus the
 *     retrains_sent/retrains_recv budgets.
 *   - BAT wire compression (HS_BAT_RLE, bat_coarsen, the RobustLink rate
 *     ladder). This port carries BAT bodies raw, matching `HS_BAT_RLE =
 *     False` in the reference -- a slower but correct wire format. The
 *     compression is a wire-encoding optimization layered over this state
 *     machine, not part of it -- see ofdm_hs_pack_msg's body parameter,
 *     which accepts whatever bytes the caller already encoded.
 *
 * BAT STORAGE.  A bit-allocation table is small but variable-length (one
 * code per occupied carrier). The no-malloc contract means a fixed cap:
 * OFDM_HS_BAT_MAX bytes, sized for the handshake convention band (HS_BAND in
 * ofdm_handshake.py: 400-4000 Hz at n_fft=1024, well under 128 raw bytes). A
 * wider band or finer FFT needs a bigger cap -- rebuild with
 * -DOFDM_HS_BAT_MAX=<n> rather than silently truncating a table.
 *
 * CONTRACT: no malloc, no stdio, no globals, one caller-owned handle.
 */
#ifndef OFDM_HS_H
#define OFDM_HS_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OFDM_HS_BAT_MAX
#define OFDM_HS_BAT_MAX 128
#endif

typedef enum {
    OFDM_HS_ROLE_INITIATOR = 0,
    OFDM_HS_ROLE_RESPONDER = 1
} ofdm_hs_role;

typedef enum {
    OFDM_HS_IDLE = 0,
    OFDM_HS_I_WAIT_PROBE,
    OFDM_HS_R_WAIT_PROBE,
    OFDM_HS_R_WAIT_ACK,
    OFDM_HS_I_WAIT_BAT,
    OFDM_HS_I_WAIT_GO,
    OFDM_HS_R_LINGER,
    OFDM_HS_DONE,
    OFDM_HS_FAILED
} ofdm_hs_state;

/* Message type codes -- MUST match ofdm_handshake.py's MSG_* values; they
 * are the wire contract, not an internal choice. */
#define OFDM_HS_MSG_BAT_REPORT 1
#define OFDM_HS_MSG_BAT_ACK    2
#define OFDM_HS_MSG_GO         3
#define OFDM_HS_MSG_RETRY      4
#define OFDM_HS_MSG_FIN        5
#define OFDM_HS_MSG_BAND       6  /* recognised on the wire (unpack_msg
                                    * accepts it) but NOT acted on by this
                                    * state machine -- see header docs. */

typedef enum {
    OFDM_HS_ACT_SEND_PROBE = 0,
    OFDM_HS_ACT_SEND_MSG = 1,
    OFDM_HS_ACT_DONE = 2,
    OFDM_HS_ACT_FAIL = 3
} ofdm_hs_act_kind;

/* Fail reasons -- mirrors the reference's f-string reasons
 * (f"timeout_{state}", "peer_restarting") as an enum, plus the state the
 * timeout happened in. */
typedef enum {
    OFDM_HS_FAIL_NONE = 0,
    OFDM_HS_FAIL_TIMEOUT,       /* see .state for which wait state expired */
    OFDM_HS_FAIL_PEER_RESTARTING /* r_wait_ack: peer re-probed too many times */
} ofdm_hs_fail_reason;

/* Every event can produce at most this many actions (e.g. heard_probe in
 * r_wait_probe emits send_probe + send_msg(REPORT); a successful GO emits
 * send_msg(FIN) + done). Bump and re-audit before adding a branch that
 * fires more. */
#define OFDM_HS_MAX_ACTIONS 2

typedef struct {
    ofdm_hs_act_kind kind;

    /* SEND_MSG only */
    int mtype;
    unsigned proof_crc;
    const unsigned char *body;   /* points into the SM's own storage or a
                                   * static empty buffer -- valid only until
                                   * the next call into this handle */
    size_t body_len;

    /* DONE only -- the two negotiated tables. Same lifetime note as body. */
    const unsigned char *my_tx_bat;
    size_t my_tx_bat_len;
    const unsigned char *peer_tx_bat;
    size_t peer_tx_bat_len;

    /* FAIL only */
    ofdm_hs_fail_reason fail_reason;
    ofdm_hs_state fail_state;    /* state the timeout fired in, if TIMEOUT */
} ofdm_hs_action;

typedef struct {
    ofdm_hs_role role;
    ofdm_hs_state state;
    int max_retries;
    double wait_probe_s;
    double wait_msg_s;
    double linger_s;
    int skip_fin;                /* boolean */

    unsigned char peer_tx_bat[OFDM_HS_BAT_MAX]; /* I measured peer's probe */
    size_t peer_tx_bat_len;
    int has_peer_tx_bat;
    unsigned char my_tx_bat[OFDM_HS_BAT_MAX];   /* peer told me my table */
    size_t my_tx_bat_len;
    int has_my_tx_bat;

    int done;                    /* boolean */
    int failed;                  /* boolean */
    double deadline;             /* OFDM_HS_TIME_UNSET if not armed */
    int retries;
} ofdm_hs;

/* Sentinel: no deadline armed / no timestamp recorded. Driver clocks are
 * seconds since an arbitrary run epoch and never negative. */
#define OFDM_HS_TIME_UNSET (-1.0)

/* Zero-init `h` and fill in config. `linger_s <= 0.0` takes the reference's
 * derived default, wait_msg_s + 45.0 (sized to outlive one full initiator
 * retry cycle: its own wait plus a RETRY's airtime plus carrier-sense
 * deferral up to ~25s -- see ofdm_hs.c). Returns -1 on a bad role. */
int ofdm_hs_init(ofdm_hs *h, ofdm_hs_role role, int max_retries,
                 double wait_probe_s, double wait_msg_s, double linger_s,
                 int skip_fin);

/* Enter the opening state (send_probe for the initiator, or just start
 * listening for the responder). Writes at most 1 action. Returns the count,
 * or -1 if max_acts is too small. */
int ofdm_hs_start(ofdm_hs *h, double now, ofdm_hs_action *acts, int max_acts);

/* "probe" (1), "msg" (2), or "probe+msg" (3) via the OFDM_HS_EXPECT_* bits --
 * tells the binding what to try decoding from the mic in the current wait
 * state. 0 in a non-waiting state (idle/done/failed). */
#define OFDM_HS_EXPECT_PROBE 1
#define OFDM_HS_EXPECT_MSG   2
int ofdm_hs_expecting(const ofdm_hs *h);

/* Binding measured the peer's probe and packed the resulting BAT for the
 * peer->me direction (= the PEER's tx table). `bat`/`bat_len` are COPIED
 * (bat_len must be <= OFDM_HS_BAT_MAX). Writes at most OFDM_HS_MAX_ACTIONS
 * actions. Returns the count, -1 if max_acts too small, -2 if bat_len
 * exceeds OFDM_HS_BAT_MAX. */
int ofdm_hs_heard_probe(ofdm_hs *h, const unsigned char *bat, size_t bat_len,
                        double now, ofdm_hs_action *acts, int max_acts);

/* Binding decoded a handshake message (mtype/proof/body from
 * ofdm_hs_unpack_msg). `body` need only stay valid for the duration of this
 * call -- BAT bodies are copied into the handle's own storage.  Writes at
 * most OFDM_HS_MAX_ACTIONS actions. Returns the count, -1 if max_acts too
 * small, -2 if a BAT body exceeds OFDM_HS_BAT_MAX. */
int ofdm_hs_heard_msg(ofdm_hs *h, int mtype, unsigned proof,
                      const unsigned char *body, size_t body_len, double now,
                      ofdm_hs_action *acts, int max_acts);

/* Extend the current deadline by time spent transmitting/deferring (own
 * airtime, carrier-sense defer). Call after every transmission -- without
 * this the timer counts through our own turn and fires a spurious
 * retransmit on top of a reply already inbound. */
void ofdm_hs_hold_off(ofdm_hs *h, double seconds);

/* Drive timeouts. Writes at most OFDM_HS_MAX_ACTIONS actions (r_wait_ack's
 * retry retransmits the WHOLE turn: send_probe + send_msg(REPORT)). Returns
 * the count, or -1 if max_acts too small. */
int ofdm_hs_tick(ofdm_hs *h, double now, ofdm_hs_action *acts, int max_acts);

/* -- wire framing: crc16 / pack_msg / unpack_msg / pack_band / unpack_band */

/* CRC-16/CCITT-FALSE. */
unsigned ofdm_hs_crc16(const unsigned char *data, size_t len);

/* [1B type][2B proof_crc][2B body_crc][body] into `out` (capacity `out_cap`).
 * Returns the packed length, or -1 if out_cap is too small. */
int ofdm_hs_pack_msg(unsigned char *out, size_t out_cap, int mtype,
                     unsigned proof_crc, const unsigned char *body,
                     size_t body_len);

/* Unpacks `data` (length `len`) in place -- `*body` points into `data`, no
 * copy. Returns 0 and fills mtype/proof/body/body_len on success, -1 if
 * malformed or the body CRC does not match (identical treatment to the
 * reference's unpack_msg returning None -- caller should treat this exactly
 * like "nothing decoded"). */
int ofdm_hs_unpack_msg(const unsigned char *data, size_t len, int *mtype,
                      unsigned *proof, const unsigned char **body,
                      size_t *body_len);

/* [dir u8][lo_hz u16 BE][hi_hz u16 BE]. Recognised but unused by this port
 * (see header docs) -- exposed so a driver that itself decodes MSG_BAND to
 * ignore/log it doesn't need its own copy of the wire format. */
#define OFDM_HS_BAND_DIR_PEER_TX 0
#define OFDM_HS_BAND_DIR_SELF_TX 1
int ofdm_hs_pack_band(unsigned char *out, size_t out_cap, double lo_hz,
                     double hi_hz, int direction);
int ofdm_hs_unpack_band(const unsigned char *body, size_t len,
                       int *direction, double *lo_hz, double *hi_hz);

#ifdef __cplusplus
}
#endif
#endif
