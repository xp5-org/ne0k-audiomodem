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

/* ofdm_handshake.c -- see ofdm_handshake.h. A near-mechanical transliteration
 * of HandshakeSM in OFDM/ofdm_handshake.py -- every state transition below
 * has a direct line-for-line counterpart there. Logging (self._note) is
 * dropped (no stdio in this freestanding contract); a driver that wants a
 * transcript logs the state before/after each call itself, from `p->state`
 * and the returned actions.
 */
#include "ofdm_handshake.h"
#include <string.h>

unsigned ofdm_hs_crc16(const unsigned char *data, int n)
{
    unsigned crc = 0xFFFFu;
    int i, b;
    for (i = 0; i < n; i++) {
        crc ^= (unsigned)data[i] << 8;
        for (b = 0; b < 8; b++)
            crc = (crc & 0x8000u) ? (unsigned)(((crc << 1) ^ 0x1021u) & 0xFFFFu)
                                  : (unsigned)((crc << 1) & 0xFFFFu);
    }
    return crc;
}

int ofdm_hs_pack_msg(int mtype, unsigned proof_crc,
                     const unsigned char *body, int body_len,
                     unsigned char *out, int out_cap)
{
    unsigned bcrc;
    if (body_len < 0 || out_cap < body_len + 5) return -1;
    bcrc = ofdm_hs_crc16(body, body_len);
    out[0] = (unsigned char)mtype;
    out[1] = (unsigned char)((proof_crc >> 8) & 0xFF);
    out[2] = (unsigned char)(proof_crc & 0xFF);
    out[3] = (unsigned char)((bcrc >> 8) & 0xFF);
    out[4] = (unsigned char)(bcrc & 0xFF);
    if (body_len > 0) memcpy(out + 5, body, (size_t)body_len);
    return body_len + 5;
}

int ofdm_hs_unpack_msg(const unsigned char *data, int n,
                       int *mtype, unsigned *proof_crc,
                       unsigned char *body, int body_cap, int *body_len)
{
    unsigned bcrc_wire, bcrc_calc;
    int blen;
    if (n < 5) return -1;
    blen = n - 5;
    bcrc_wire = ((unsigned)data[3] << 8) | (unsigned)data[4];
    bcrc_calc = ofdm_hs_crc16(data + 5, blen);
    if (bcrc_wire != bcrc_calc) return -1;
    *mtype = data[0];
    *proof_crc = ((unsigned)data[1] << 8) | (unsigned)data[2];
    *body_len = blen;
    if (body && body_cap > 0) {
        int copy = blen < body_cap ? blen : body_cap;
        memcpy(body, data + 5, (size_t)copy);
    }
    return 0;
}

int ofdm_hs_pack_band(double lo_hz, double hi_hz, int direction,
                      unsigned char *out, int out_cap)
{
    long lo, hi;
    if (out_cap < 5) return -1;
    lo = (long)(lo_hz + 0.5); if (lo < 0) lo = 0; if (lo > 65535) lo = 65535;
    hi = (long)(hi_hz + 0.5); if (hi < 0) hi = 0; if (hi > 65535) hi = 65535;
    out[0] = (unsigned char)(direction & 0xFF);
    out[1] = (unsigned char)((lo >> 8) & 0xFF);
    out[2] = (unsigned char)(lo & 0xFF);
    out[3] = (unsigned char)((hi >> 8) & 0xFF);
    out[4] = (unsigned char)(hi & 0xFF);
    return 5;
}

int ofdm_hs_unpack_band(const unsigned char *body, int n,
                        int *direction, double *lo_hz, double *hi_hz)
{
    long lo, hi;
    if (n < 5) return -1;
    *direction = body[0];
    lo = ((long)body[1] << 8) | (long)body[2];
    hi = ((long)body[3] << 8) | (long)body[4];
    if (hi <= lo || hi - lo < 100) return -1;
    *lo_hz = (double)lo;
    *hi_hz = (double)hi;
    return 0;
}

int ofdm_hs_init(ofdm_hs *p, ohs_role role, int max_retries,
                 double wait_probe_s, double wait_msg_s, double linger_s,
                 int skip_fin, int band_negotiation, int max_retrains)
{
    if (role != OHS_ROLE_INITIATOR && role != OHS_ROLE_RESPONDER) return -1;
    memset(p, 0, sizeof(*p));
    p->role = role;
    p->state = OHS_STATE_IDLE;
    p->max_retries = max_retries;
    p->wait_probe_s = wait_probe_s;
    p->wait_msg_s = wait_msg_s;
    p->skip_fin = skip_fin;
    p->band_negotiation = band_negotiation;
    /* The reference hardcodes this to the module constant HS_MAX_RETRAINS
     * (=1) rather than taking it as a constructor argument -- exposed here
     * as an explicit parameter instead, since a freestanding caller has no
     * shared module constant to reach for (same reasoning as every other
     * field in this init call). Pass 1 to match the reference exactly. */
    p->max_retrains = max_retrains;
    p->linger_s = linger_s > 0.0 ? linger_s : (wait_msg_s + 45.0);

    p->deadline = OHS_TIME_UNSET;
    p->peer_tx_bat_len = -1;
    p->my_tx_bat_len = -1;
    p->my_tx_band_set = 0;
    p->peer_tx_band_set = 0;
    p->retrains_sent = 0;
    p->retrains_recv = 0;
    return 0;
}

/* -- small internal helpers, mirroring the reference's _emit/_arm/_goto -- */

static void arm(ofdm_hs *p, double now, double seconds) { p->deadline = now + seconds; }
static void goto_state(ofdm_hs *p, ohs_state s) { p->state = s; p->retries = 0; }

static int emit_send_probe(ohs_action *acts, int i)
{
    acts[i].kind = OHS_ACT_SEND_PROBE;
    return i + 1;
}

/* body is a pointer into `p`'s own storage (peer_tx_bat/my_tx_bat) or NULL
 * for an empty body -- copied into the action's own buffer immediately, so
 * the driver never has to worry about `p`'s storage changing under it. */
static int emit_send_msg(ohs_action *acts, int i, int mtype, unsigned proof,
                         const unsigned char *body, int body_len)
{
    acts[i].kind = OHS_ACT_SEND_MSG;
    acts[i].mtype = mtype;
    acts[i].proof_crc = proof;
    acts[i].body_len = body_len > OHS_MAX_BAT_BYTES ? OHS_MAX_BAT_BYTES : body_len;
    if (body_len > 0 && body) memcpy(acts[i].body, body, (size_t)acts[i].body_len);
    return i + 1;
}

/* Raw pass-through: BAT wire compression (HS_BAT_RLE) is a driver-level
 * concern, deliberately out of scope here -- see the header comment. */
static int send_report(ofdm_hs *p, ohs_action *acts, int i)
{
    return emit_send_msg(acts, i, OHS_MSG_BAT_REPORT, 0,
                         p->peer_tx_bat, p->peer_tx_bat_len < 0 ? 0 : p->peer_tx_bat_len);
}

static int send_ack(ofdm_hs *p, ohs_action *acts, int i)
{
    unsigned proof = ofdm_hs_crc16(p->my_tx_bat, p->my_tx_bat_len < 0 ? 0 : p->my_tx_bat_len);
    return emit_send_msg(acts, i, OHS_MSG_BAT_ACK, proof,
                         p->peer_tx_bat, p->peer_tx_bat_len < 0 ? 0 : p->peer_tx_bat_len);
}

static int send_go(ofdm_hs *p, ohs_action *acts, int i)
{
    unsigned proof = ofdm_hs_crc16(p->my_tx_bat, p->my_tx_bat_len < 0 ? 0 : p->my_tx_bat_len);
    return emit_send_msg(acts, i, OHS_MSG_GO, proof, (const unsigned char *)0, 0);
}

static int finish(ofdm_hs *p, ohs_action *acts, int i)
{
    p->done = 1;
    p->state = OHS_STATE_DONE;
    p->deadline = OHS_TIME_UNSET;
    acts[i].kind = OHS_ACT_DONE;
    acts[i].my_tx_bat = p->my_tx_bat;
    acts[i].my_tx_bat_len = p->my_tx_bat_len < 0 ? 0 : p->my_tx_bat_len;
    acts[i].peer_tx_bat = p->peer_tx_bat;
    acts[i].peer_tx_bat_len = p->peer_tx_bat_len < 0 ? 0 : p->peer_tx_bat_len;
    return i + 1;
}

static int fail(ofdm_hs *p, ohs_action *acts, int i, ohs_fail_reason reason, int detail)
{
    p->failed = 1;
    p->state = OHS_STATE_FAILED;
    p->deadline = OHS_TIME_UNSET;
    acts[i].kind = OHS_ACT_FAIL;
    acts[i].fail_reason = reason;
    acts[i].fail_detail = detail;
    return i + 1;
}

/* Enter the opening wait state for this role. Shared by start() and the
 * BAND-triggered restart (the reference's start()/_restart() are the same
 * role-branch, restart just also resets the learned tables first). */
static int enter_probe_wait(ofdm_hs *p, double now, ohs_action *acts, int i)
{
    if (p->role == OHS_ROLE_INITIATOR) {
        i = emit_send_probe(acts, i);
        goto_state(p, OHS_STATE_I_WAIT_PROBE);
        arm(p, now, p->wait_probe_s);
    } else {
        goto_state(p, OHS_STATE_R_WAIT_PROBE);
        arm(p, now, p->wait_probe_s);
    }
    return i;
}

static int do_restart(ofdm_hs *p, double now, ohs_action *acts, int i)
{
    p->peer_tx_bat_len = -1;
    p->my_tx_bat_len = -1;
    p->retries = 0;
    return enter_probe_wait(p, now, acts, i);
}

int ofdm_hs_start(ofdm_hs *p, double now, ohs_action *acts, int max_acts)
{
    if (max_acts < OHS_MAX_ACTIONS) return -1;
    return enter_probe_wait(p, now, acts, 0);
}

ohs_expect ofdm_hs_expecting(const ofdm_hs *p)
{
    switch (p->state) {
    case OHS_STATE_I_WAIT_PROBE:
    case OHS_STATE_R_WAIT_PROBE:
        return p->band_negotiation ? OHS_EXPECT_PROBE_OR_MSG : OHS_EXPECT_PROBE;
    case OHS_STATE_I_WAIT_BAT:
    case OHS_STATE_I_WAIT_GO:
        return OHS_EXPECT_MSG;
    case OHS_STATE_R_WAIT_ACK:
        return OHS_EXPECT_PROBE_OR_MSG;
    case OHS_STATE_R_LINGER:
        return OHS_EXPECT_MSG;
    default:
        return OHS_EXPECT_NONE;
    }
}

int ofdm_hs_heard_probe(ofdm_hs *p, const unsigned char *bat_bytes,
                        int bat_len, double now,
                        int has_band, double band_lo, double band_hi,
                        ohs_action *acts, int max_acts)
{
    int i = 0;

    if (max_acts < OHS_MAX_ACTIONS) return -1;
    if (bat_len > OHS_MAX_BAT_BYTES) bat_len = OHS_MAX_BAT_BYTES;

    /* Band retrain outranks the normal flow -- see ofdm_handshake.h. */
    if (has_band && p->band_negotiation
            && (!p->peer_tx_band_set || band_lo != p->peer_tx_band_lo
                || band_hi != p->peer_tx_band_hi)
            && p->retrains_sent < p->max_retrains
            && (p->state == OHS_STATE_I_WAIT_PROBE
                || p->state == OHS_STATE_R_WAIT_PROBE
                || p->state == OHS_STATE_R_WAIT_ACK)) {
        unsigned char band_body[5];
        int blen;
        p->peer_tx_band_set = 1;
        p->peer_tx_band_lo = band_lo;
        p->peer_tx_band_hi = band_hi;
        p->retrains_sent++;
        blen = ofdm_hs_pack_band(band_lo, band_hi, OHS_BAND_DIR_PEER_TX,
                                 band_body, (int)sizeof(band_body));
        i = emit_send_msg(acts, i, OHS_MSG_BAND, 0, band_body, blen);
        i = do_restart(p, now, acts, i);
        return i;
    }

    if (p->state == OHS_STATE_I_WAIT_PROBE) {
        memcpy(p->peer_tx_bat, bat_bytes, (size_t)bat_len);
        p->peer_tx_bat_len = bat_len;
        goto_state(p, OHS_STATE_I_WAIT_BAT);   /* R sends BAT_REPORT unprompted */
        arm(p, now, p->wait_msg_s);
    } else if (p->state == OHS_STATE_R_WAIT_PROBE) {
        memcpy(p->peer_tx_bat, bat_bytes, (size_t)bat_len);
        p->peer_tx_bat_len = bat_len;
        i = emit_send_probe(acts, i);
        i = send_report(p, acts, i);
        goto_state(p, OHS_STATE_R_WAIT_ACK);
        arm(p, now, p->wait_msg_s);
    } else if (p->state == OHS_STATE_R_WAIT_ACK) {
        /* Initiator re-probed: it never heard our probe/report. Re-measure
         * and repeat our whole turn (see header comment for why the whole
         * turn, not just the report). */
        memcpy(p->peer_tx_bat, bat_bytes, (size_t)bat_len);
        p->peer_tx_bat_len = bat_len;
        p->retries++;
        if (p->retries > p->max_retries) {
            i = fail(p, acts, i, OHS_FAIL_PEER_RESTARTING, 0);
            return i;
        }
        i = emit_send_probe(acts, i);
        i = send_report(p, acts, i);
        arm(p, now, p->wait_msg_s);
    }
    /* other states: stale echo, ignore */
    return i;
}

int ofdm_hs_heard_msg(ofdm_hs *p, int mtype, unsigned proof,
                      const unsigned char *body, int body_len, double now,
                      ohs_action *acts, int max_acts)
{
    int i = 0;

    if (max_acts < OHS_MAX_ACTIONS) return -1;

    /* MSG_BAND is handled in EVERY state and outranks the normal flow. */
    if (mtype == OHS_MSG_BAND) {
        int direction; double lo, hi;
        if (ofdm_hs_unpack_band(body, body_len, &direction, &lo, &hi) != 0)
            return 0;
        if (direction == OHS_BAND_DIR_SELF_TX) {
            p->peer_tx_band_set = 1;   /* informational only */
            p->peer_tx_band_lo = lo;
            p->peer_tx_band_hi = hi;
            return 0;
        }
        if (p->my_tx_band_set && lo == p->my_tx_band_lo && hi == p->my_tx_band_hi)
            return 0;   /* duplicate/retransmit */
        if (p->retrains_recv >= p->max_retrains)
            return 0;   /* retrain budget spent */
        p->my_tx_band_set = 1;
        p->my_tx_band_lo = lo;
        p->my_tx_band_hi = hi;
        p->retrains_recv++;
        i = do_restart(p, now, acts, i);
        return i;
    }

    if (body_len > OHS_MAX_BAT_BYTES) body_len = OHS_MAX_BAT_BYTES;

    if (p->state == OHS_STATE_I_WAIT_BAT && mtype == OHS_MSG_BAT_REPORT) {
        memcpy(p->my_tx_bat, body, (size_t)body_len);
        p->my_tx_bat_len = body_len;
        i = send_ack(p, acts, i);
        goto_state(p, OHS_STATE_I_WAIT_GO);
        arm(p, now, p->wait_msg_s);
    } else if (p->state == OHS_STATE_I_WAIT_GO) {
        if (mtype == OHS_MSG_GO) {
            unsigned want = ofdm_hs_crc16(p->peer_tx_bat,
                                          p->peer_tx_bat_len < 0 ? 0 : p->peer_tx_bat_len);
            if (proof == want) {
                if (!p->skip_fin)
                    i = emit_send_msg(acts, i, OHS_MSG_FIN, 0, (const unsigned char *)0, 0);
                i = finish(p, acts, i);
            } else {
                /* peer holds a different BAT_R than we sent -- ask again */
                i = emit_send_msg(acts, i, OHS_MSG_RETRY, 0, (const unsigned char *)0, 0);
                arm(p, now, p->wait_msg_s);
            }
        } else if (mtype == OHS_MSG_BAT_REPORT) {
            /* our BAT_ACK was lost; peer retransmitted its report */
            memcpy(p->my_tx_bat, body, (size_t)body_len);
            p->my_tx_bat_len = body_len;
            i = send_ack(p, acts, i);
            arm(p, now, p->wait_msg_s);
        }
    } else if (p->state == OHS_STATE_R_WAIT_ACK) {
        if (mtype == OHS_MSG_BAT_ACK) {
            unsigned want = ofdm_hs_crc16(p->peer_tx_bat,
                                          p->peer_tx_bat_len < 0 ? 0 : p->peer_tx_bat_len);
            if (proof == want) {
                memcpy(p->my_tx_bat, body, (size_t)body_len);
                p->my_tx_bat_len = body_len;
                i = send_go(p, acts, i);
                goto_state(p, OHS_STATE_R_LINGER);
                arm(p, now, p->linger_s);
            } else {
                i = send_report(p, acts, i);   /* peer got a corrupt report */
                arm(p, now, p->wait_msg_s);
            }
        } else if (mtype == OHS_MSG_RETRY) {
            i = send_report(p, acts, i);
            arm(p, now, p->wait_msg_s);
        }
    } else if (p->state == OHS_STATE_R_LINGER) {
        if (mtype == OHS_MSG_FIN) {
            i = finish(p, acts, i);   /* initiator confirmed GO landed */
        } else if (mtype == OHS_MSG_RETRY || mtype == OHS_MSG_BAT_ACK) {
            /* initiator missed GO (RETRY) or our GO crossed its ACK */
            i = send_go(p, acts, i);
            arm(p, now, p->linger_s);
        }
    }
    /* anything else: duplicate/stale, ignore */
    return i;
}

void ofdm_hs_hold_off(ofdm_hs *p, double seconds)
{
    if (p->deadline != OHS_TIME_UNSET && seconds > 0.0)
        p->deadline += seconds;
}

int ofdm_hs_tick(ofdm_hs *p, double now, ohs_action *acts, int max_acts)
{
    int i = 0;

    if (max_acts < OHS_MAX_ACTIONS) return -1;
    if (p->deadline == OHS_TIME_UNSET || now < p->deadline) return 0;

    p->retries++;
    if (p->state == OHS_STATE_R_LINGER) {
        i = finish(p, acts, i);   /* linger over, assume GO landed */
        return i;
    }
    if (p->retries > p->max_retries) {
        i = fail(p, acts, i, OHS_FAIL_TIMEOUT, (int)p->state);
        return i;
    }

    switch (p->state) {
    case OHS_STATE_I_WAIT_PROBE:
        i = emit_send_probe(acts, i);
        arm(p, now, p->wait_probe_s);
        break;
    case OHS_STATE_R_WAIT_PROBE:
        arm(p, now, p->wait_probe_s);   /* keep listening, no action */
        break;
    case OHS_STATE_I_WAIT_BAT:
    case OHS_STATE_I_WAIT_GO:
        i = emit_send_msg(acts, i, OHS_MSG_RETRY, 0, (const unsigned char *)0, 0);
        arm(p, now, p->wait_msg_s);
        break;
    case OHS_STATE_R_WAIT_ACK:
        /* Repeat the WHOLE turn (probe + report), not the report alone --
         * see the header comment for why a report-only retransmit
         * deadlocks against an initiator stuck in i_wait_probe. */
        i = emit_send_probe(acts, i);
        i = send_report(p, acts, i);
        arm(p, now, p->wait_msg_s);
        break;
    default:
        break;
    }
    return i;
}
