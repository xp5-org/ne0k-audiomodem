/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026 XP5.ORG
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/* ofdm_handshake.h -- V.22-style two-way OFDM handshake state machine,
 * freestanding C89. Port of `HandshakeSM` in OFDM/ofdm_handshake.py: pure
 * protocol logic (states, retries, timeouts, CRC proof-of-receipt), no DSP,
 * no audio, no wall clock -- time is injected via `now`.
 *
 * Ported here (ofdm_dos/core/), not under mesh_c/: it's negotiated per-link
 * OFDM protocol, not mesh-specific. mesh_phy.c/.h is the layer ABOVE this one
 * that decides whether to enter a handshake at all; this is what runs once
 * it does, following the same porting conventions mesh_phy.h established:
 * action-list-by-caller-owned-buffer, enum reason codes instead of Python
 * f-strings, an OHS_TIME_UNSET sentinel instead of Optional[float], no
 * malloc/stdio/globals.
 *
 * Protocol (half-duplex, turn-based; I = initiator, R = responder):
 *
 *   I: PROBE                     R measures  -> BAT_I  (I's tx bit-loading)
 *   R: PROBE                     I measures  -> BAT_R  (R's tx bit-loading)
 *   R: MSG BAT_REPORT(BAT_I)     I learns its tx table
 *   I: MSG BAT_ACK(BAT_R, proof=crc(BAT_I))  R learns its tx table + proof
 *   R: MSG GO(proof=crc(BAT_R))              I gets proof R got BAT_R
 *   I: MSG FIN                               courtesy, lets R stop early
 *   both DONE -- each end holds an agreed bit-loading table for BOTH
 *   directions. Recovery rationale (why r_wait_ack retransmits the WHOLE
 *   turn, why r_linger must outlive one initiator retry cycle) is at each
 *   transition in ofdm_handshake.c rather than repeated here.
 *
 * SCOPE CUT, DELIBERATE: BAT WIRE ENCODING IS NOT IN THIS FILE. The
 * reference's HandshakeSM calls bat_rle_pack/bat_rle_unpack (HS_BAT_RLE) to
 * compress the BAT body before it touches the wire -- a DSP-adjacent codec
 * choice, not FSM logic, deferred for later (mirrors mesh_phy.h deferring the
 * CSS header and mesh-OFDM framing for the same reason). ofdm_hs_heard_probe/
 * heard_msg here take and emit RAW bat bytes; a driver wanting
 * HS_BAT_RLE-compatible compression does it between this state machine and
 * the OFDM transmit/receive calls.
 *
 * BAND NEGOTIATION (MSG_BAND) IS PORTED, since it is pure FSM logic (no DSP)
 * and a fixed a-priori band does not always survive real hardware (see
 * OFDMBAT.C's RATESCAN). `ofdm_hs_heard_probe`'s optional `band` argument is
 * the driver's recommendation (the reference's `recommend_band()`, itself a
 * DSP function, stays the driver's job).
 */
#ifndef OFDM_HANDSHAKE_H
#define OFDM_HANDSHAKE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OHS_ROLE_INITIATOR = 0,
    OHS_ROLE_RESPONDER = 1
} ohs_role;

typedef enum {
    OHS_STATE_IDLE = 0,
    OHS_STATE_I_WAIT_PROBE,
    OHS_STATE_I_WAIT_BAT,
    OHS_STATE_I_WAIT_GO,
    OHS_STATE_R_WAIT_PROBE,
    OHS_STATE_R_WAIT_ACK,
    OHS_STATE_R_LINGER,
    OHS_STATE_DONE,
    OHS_STATE_FAILED
} ohs_state;

/* Wire message types -- matches MSG_* in the reference exactly (wire
 * compatible: a byte value on the wire means the same thing on both ends). */
#define OHS_MSG_BAT_REPORT 1
#define OHS_MSG_BAT_ACK    2
#define OHS_MSG_GO         3
#define OHS_MSG_RETRY      4
#define OHS_MSG_FIN        5
#define OHS_MSG_BAND       6

/* MSG_BAND body: [dir u8][lo_hz u16 BE][hi_hz u16 BE] -- see pack_band/
 * unpack_band in the reference. Only PEER_TX is acted on; SELF_TX is
 * informational. */
#define OHS_BAND_DIR_PEER_TX 0
#define OHS_BAND_DIR_SELF_TX 1

/* Sentinel for "no timestamp armed" -- driver clocks are seconds since an
 * arbitrary run epoch and never negative (same convention as meshphy.h's
 * MESHPHY_TIME_UNSET). */
#define OHS_TIME_UNSET (-1.0)

/* Upper bound on a raw (uncompressed) BAT table this state machine will
 * store or emit. Sized generously above anything this project's own OFDM
 * geometry produces (n_occupied<=41 in every config used so far) -- bump if
 * a future config's data-carrier count needs more. */
#define OHS_MAX_BAT_BYTES 256

/* expecting() return values -- what the driver should try to decode from
 * the channel while in the current wait state. */
typedef enum {
    OHS_EXPECT_NONE = 0,
    OHS_EXPECT_PROBE,
    OHS_EXPECT_MSG,
    OHS_EXPECT_PROBE_OR_MSG
} ohs_expect;

typedef enum {
    OHS_ACT_SEND_PROBE = 0,
    OHS_ACT_SEND_MSG = 1,
    OHS_ACT_DONE = 2,
    OHS_ACT_FAIL = 3
} ohs_act_kind;

typedef enum {
    OHS_FAIL_NONE = 0,
    OHS_FAIL_PEER_RESTARTING, /* r_wait_ack: peer re-probed too many times */
    OHS_FAIL_TIMEOUT          /* detail = the ohs_state that timed out */
} ohs_fail_reason;

/* Every entry point that can act writes into a caller-owned ohs_action
 * array of at least OHS_MAX_ACTIONS entries and returns the count (0..
 * OHS_MAX_ACTIONS), or -1 if max_acts is smaller than that -- same contract
 * as mesh_phy.h's meshphy_action array. No call in this state machine emits
 * more than 2 (SEND_PROBE + SEND_MSG together, in the r_wait_probe/
 * r_wait_ack-retry turns that resend both at once). */
#define OHS_MAX_ACTIONS 2

typedef struct {
    ohs_act_kind kind;

    /* SEND_MSG only */
    int mtype;
    unsigned proof_crc;
    unsigned char body[OHS_MAX_BAT_BYTES];
    int body_len;

    /* DONE only -- points at the state machine's own storage, valid until
     * the next event call (same lifetime rule as any other read of `p`'s
     * fields; copy out if the driver needs it to outlive that). */
    const unsigned char *my_tx_bat;
    int my_tx_bat_len;
    const unsigned char *peer_tx_bat;
    int peer_tx_bat_len;

    /* FAIL only */
    ohs_fail_reason fail_reason;
    int fail_detail;   /* OHS_FAIL_TIMEOUT: the ohs_state that timed out */
} ohs_action;

typedef struct {
    /* config, set at init and not touched again */
    ohs_role role;
    int max_retries;
    double wait_probe_s;
    double wait_msg_s;
    double linger_s;
    int skip_fin;
    int band_negotiation;
    int max_retrains;

    /* protocol state */
    ohs_state state;
    int done;
    int failed;
    double deadline;   /* OHS_TIME_UNSET if not armed */
    int retries;

    unsigned char peer_tx_bat[OHS_MAX_BAT_BYTES]; /* I measured peer's probe */
    int peer_tx_bat_len;                          /* -1 if unset */
    unsigned char my_tx_bat[OHS_MAX_BAT_BYTES];    /* peer told me my table */
    int my_tx_bat_len;                             /* -1 if unset */

    /* per-direction bands; *_set=0 means "a-priori convention, not yet
     * renegotiated" (the reference's None). One retrain budget PER
     * DIRECTION -- see the header comment in ofdm_handshake.c. */
    int my_tx_band_set;
    double my_tx_band_lo, my_tx_band_hi;
    int peer_tx_band_set;
    double peer_tx_band_lo, peer_tx_band_hi;
    int retrains_sent;
    int retrains_recv;
} ofdm_hs;

/* Zero-init `p` and fill in config, mirroring HandshakeSM.__init__ --
 * every field explicit, no hidden Python-keyword-default fallback (same
 * convention meshphy_init uses and for the same reason: a freestanding
 * caller has no keyword defaults to lean on).
 *
 * `linger_s`: pass <= 0.0 to take the reference's derived default,
 * wait_msg_s + 45.0.
 *
 * `max_retrains`: the reference hardcodes this to the module constant
 * HS_MAX_RETRAINS rather than taking it as a constructor argument -- pass 1
 * to match it exactly (see ofdm_handshake.c's own comment at this field).
 *
 * Returns -1 if role is out of range, else 0. */
int ofdm_hs_init(ofdm_hs *p, ohs_role role, int max_retries,
                 double wait_probe_s, double wait_msg_s, double linger_s,
                 int skip_fin, int band_negotiation, int max_retrains);

/* CRC-16/CCITT-FALSE, matching the reference's crc16() exactly (init
 * 0xFFFF, poly 0x1021, no reflection) -- used for both the wire message
 * body CRC and the GO/ACK proof CRCs. */
unsigned ofdm_hs_crc16(const unsigned char *data, int n);

/* [1B type][2B proof_crc BE][2B body_crc BE][body] -- matches pack_msg/
 * unpack_msg exactly. ofdm_hs_pack_msg returns bytes written, or -1 if
 * out_cap is too small. ofdm_hs_unpack_msg returns 0 ok (writes *mtype,
 * *proof_crc, and up to body_cap bytes of body into `body`, plus the true
 * body length into *body_len even if truncated), or -1 if malformed/too
 * short/body CRC mismatch. */
int ofdm_hs_pack_msg(int mtype, unsigned proof_crc,
                     const unsigned char *body, int body_len,
                     unsigned char *out, int out_cap);
int ofdm_hs_unpack_msg(const unsigned char *data, int n,
                       int *mtype, unsigned *proof_crc,
                       unsigned char *body, int body_cap, int *body_len);

/* [dir u8][lo_hz u16 BE][hi_hz u16 BE] -- matches pack_band/unpack_band.
 * ofdm_hs_unpack_band returns 0 ok (rejects hi<=lo or hi-lo<100, same as
 * the reference), -1 malformed/nonsensical. */
int ofdm_hs_pack_band(double lo_hz, double hi_hz, int direction,
                      unsigned char *out, int out_cap);
int ofdm_hs_unpack_band(const unsigned char *body, int n,
                        int *direction, double *lo_hz, double *hi_hz);

/* Which channel decode the driver should attempt while in the current wait
 * state (mirrors HandshakeSM.expecting()). */
ohs_expect ofdm_hs_expecting(const ofdm_hs *p);

/* -- events -- */

/* Enter the opening state. Writes at most OHS_MAX_ACTIONS actions (an
 * initiator emits SEND_PROBE; a responder emits nothing, it just starts
 * listening). Returns the count, or -1 if max_acts too small. */
int ofdm_hs_start(ofdm_hs *p, double now, ohs_action *acts, int max_acts);

/* The binding measured the peer's probe and packed the resulting RAW bat
 * bytes (peer->me direction = the PEER's tx table). `has_band`/`band_lo`/
 * `band_hi`: the driver's band recommendation for the peer's tx path
 * (reference's `recommend_band()`, a DSP call, stays the driver's job) --
 * pass has_band=0 if band_negotiation is off or no recommendation applies.
 * Returns the action count, or -1 if max_acts too small. */
int ofdm_hs_heard_probe(ofdm_hs *p, const unsigned char *bat_bytes,
                        int bat_len, double now,
                        int has_band, double band_lo, double band_hi,
                        ohs_action *acts, int max_acts);

/* A decoded wire message (already CRC-verified body -- see
 * ofdm_hs_unpack_msg). Returns the action count, or -1 if max_acts too
 * small. */
int ofdm_hs_heard_msg(ofdm_hs *p, int mtype, unsigned proof,
                      const unsigned char *body, int body_len, double now,
                      ohs_action *acts, int max_acts);

/* Extend the current deadline by time the node spent unable to hear a
 * reply -- its own airtime plus any carrier-sense deferral. Call after
 * every transmission (see the reference's hold_off docstring for why the
 * wait timer would otherwise burn through the node's own turn). No-op if
 * no deadline is armed or seconds<=0. */
void ofdm_hs_hold_off(ofdm_hs *p, double seconds);

/* Drive timeouts. Returns the action count, or -1 if max_acts too small. */
int ofdm_hs_tick(ofdm_hs *p, double now, ohs_action *acts, int max_acts);

#ifdef __cplusplus
}
#endif
#endif
