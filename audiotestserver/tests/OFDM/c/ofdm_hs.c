/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_hs.c -- see ofdm_hs.h. Port of HandshakeSM in ../ofdm_handshake.py. */
#include "ofdm_hs.h"
#include <string.h>

static const unsigned char EMPTY_BODY[1] = {0};

unsigned ofdm_hs_crc16(const unsigned char *data, size_t len)
{
    unsigned crc = 0xFFFFu;
    size_t i;
    int b;
    for (i = 0; i < len; i++) {
        crc ^= ((unsigned)data[i]) << 8;
        for (b = 0; b < 8; b++) {
            crc = (crc & 0x8000u) ? ((crc << 1) ^ 0x1021u) : (crc << 1);
            crc &= 0xFFFFu;
        }
    }
    return crc;
}

int ofdm_hs_pack_msg(unsigned char *out, size_t out_cap, int mtype,
                     unsigned proof_crc, const unsigned char *body,
                     size_t body_len)
{
    unsigned bcrc;
    if (out_cap < 5 + body_len) {
        return -1;
    }
    out[0] = (unsigned char)mtype;
    out[1] = (unsigned char)((proof_crc >> 8) & 0xFF);
    out[2] = (unsigned char)(proof_crc & 0xFF);
    bcrc = ofdm_hs_crc16(body_len ? body : EMPTY_BODY, body_len);
    out[3] = (unsigned char)((bcrc >> 8) & 0xFF);
    out[4] = (unsigned char)(bcrc & 0xFF);
    if (body_len) {
        memcpy(out + 5, body, body_len);
    }
    return (int)(5 + body_len);
}

/* Recognised message types -- must match the reference's _MSG_NAMES keys so
 * unpack_msg rejects the same things the Python does (an unknown mtype is
 * "malformed", identical treatment to a bad CRC). */
static int is_known_mtype(int mtype)
{
    return mtype >= OFDM_HS_MSG_BAT_REPORT && mtype <= OFDM_HS_MSG_BAND;
}

int ofdm_hs_unpack_msg(const unsigned char *data, size_t len, int *mtype,
                      unsigned *proof, const unsigned char **body,
                      size_t *body_len)
{
    unsigned bcrc, computed;
    int mt;
    if (len < 5) {
        return -1;
    }
    mt = data[0];
    if (!is_known_mtype(mt)) {
        return -1;
    }
    bcrc = ((unsigned)data[3] << 8) | (unsigned)data[4];
    computed = ofdm_hs_crc16(len > 5 ? data + 5 : EMPTY_BODY, len - 5);
    if (computed != bcrc) {
        return -1;
    }
    *mtype = mt;
    *proof = ((unsigned)data[1] << 8) | (unsigned)data[2];
    *body = data + 5;
    *body_len = len - 5;
    return 0;
}

int ofdm_hs_pack_band(unsigned char *out, size_t out_cap, double lo_hz,
                     double hi_hz, int direction)
{
    long lo, hi;
    if (out_cap < 5) {
        return -1;
    }
    lo = (long)(lo_hz + 0.5);
    if (lo < 0) lo = 0;
    if (lo > 65535) lo = 65535;
    hi = (long)(hi_hz + 0.5);
    if (hi < 0) hi = 0;
    if (hi > 65535) hi = 65535;
    out[0] = (unsigned char)(direction & 0xFF);
    out[1] = (unsigned char)((lo >> 8) & 0xFF);
    out[2] = (unsigned char)(lo & 0xFF);
    out[3] = (unsigned char)((hi >> 8) & 0xFF);
    out[4] = (unsigned char)(hi & 0xFF);
    return 5;
}

int ofdm_hs_unpack_band(const unsigned char *body, size_t len,
                       int *direction, double *lo_hz, double *hi_hz)
{
    long lo, hi;
    if (len < 5) {
        return -1;
    }
    *direction = body[0];
    lo = ((long)body[1] << 8) | (long)body[2];
    hi = ((long)body[3] << 8) | (long)body[4];
    if (hi <= lo || hi - lo < 100) { /* a band that narrow carries nothing */
        return -1;
    }
    *lo_hz = (double)lo;
    *hi_hz = (double)hi;
    return 0;
}

/* -------------------------------------------------------------------- */

int ofdm_hs_init(ofdm_hs *h, ofdm_hs_role role, int max_retries,
                 double wait_probe_s, double wait_msg_s, double linger_s,
                 int skip_fin)
{
    if (role != OFDM_HS_ROLE_INITIATOR && role != OFDM_HS_ROLE_RESPONDER) {
        return -1;
    }
    memset(h, 0, sizeof(*h));
    h->role = role;
    h->state = OFDM_HS_IDLE;
    h->max_retries = max_retries;
    h->wait_probe_s = wait_probe_s;
    h->wait_msg_s = wait_msg_s;
    /* Linger must outlive one full initiator retry cycle: its wait PLUS the
     * RETRY's airtime PLUS the binding's carrier-sense deferral (up to ~25s
     * when the air is busy), else the responder goes deaf before a lost GO
     * can be asked for again. */
    h->linger_s = (linger_s > 0.0) ? linger_s : (wait_msg_s + 45.0);
    h->skip_fin = skip_fin;
    h->has_peer_tx_bat = 0;
    h->peer_tx_bat_len = 0;
    h->has_my_tx_bat = 0;
    h->my_tx_bat_len = 0;
    h->done = 0;
    h->failed = 0;
    h->deadline = OFDM_HS_TIME_UNSET;
    h->retries = 0;
    return 0;
}

static void arm(ofdm_hs *h, double now, double seconds)
{
    h->deadline = now + seconds;
}

static void goto_state(ofdm_hs *h, ofdm_hs_state s)
{
    h->state = s;
    h->retries = 0;
}

static int emit_send_probe(ofdm_hs_action *a)
{
    memset(a, 0, sizeof(*a));
    a->kind = OFDM_HS_ACT_SEND_PROBE;
    return 1;
}

static int emit_send_msg(ofdm_hs_action *a, int mtype, unsigned proof,
                         const unsigned char *body, size_t body_len)
{
    memset(a, 0, sizeof(*a));
    a->kind = OFDM_HS_ACT_SEND_MSG;
    a->mtype = mtype;
    a->proof_crc = proof;
    a->body = body_len ? body : EMPTY_BODY;
    a->body_len = body_len;
    return 1;
}

/* The reference RLE-compresses BAT bodies, which dominate handshake airtime;
 * this port ships them raw, a valid but slower protocol mode (see ofdm_hs.h). */
static int emit_send_report(ofdm_hs *h, ofdm_hs_action *a)
{
    return emit_send_msg(a, OFDM_HS_MSG_BAT_REPORT, 0u, h->peer_tx_bat,
                         h->peer_tx_bat_len);
}

static int emit_send_ack(ofdm_hs *h, ofdm_hs_action *a)
{
    unsigned proof = ofdm_hs_crc16(h->my_tx_bat_len ? h->my_tx_bat
                                   : EMPTY_BODY, h->my_tx_bat_len);
    return emit_send_msg(a, OFDM_HS_MSG_BAT_ACK, proof, h->peer_tx_bat,
                         h->peer_tx_bat_len);
}

static int emit_send_go(ofdm_hs *h, ofdm_hs_action *a)
{
    unsigned proof = ofdm_hs_crc16(h->my_tx_bat_len ? h->my_tx_bat
                                   : EMPTY_BODY, h->my_tx_bat_len);
    return emit_send_msg(a, OFDM_HS_MSG_GO, proof, EMPTY_BODY, 0);
}

static int finish(ofdm_hs *h, ofdm_hs_action *a)
{
    h->done = 1;
    h->state = OFDM_HS_DONE;
    h->deadline = OFDM_HS_TIME_UNSET;
    memset(a, 0, sizeof(*a));
    a->kind = OFDM_HS_ACT_DONE;
    a->my_tx_bat = h->my_tx_bat;
    a->my_tx_bat_len = h->my_tx_bat_len;
    a->peer_tx_bat = h->peer_tx_bat;
    a->peer_tx_bat_len = h->peer_tx_bat_len;
    return 1;
}

static int fail(ofdm_hs *h, ofdm_hs_action *a, ofdm_hs_fail_reason reason)
{
    ofdm_hs_state prior = h->state;
    h->failed = 1;
    h->state = OFDM_HS_FAILED;
    h->deadline = OFDM_HS_TIME_UNSET;
    memset(a, 0, sizeof(*a));
    a->kind = OFDM_HS_ACT_FAIL;
    a->fail_reason = reason;
    a->fail_state = prior;
    return 1;
}

int ofdm_hs_start(ofdm_hs *h, double now, ofdm_hs_action *acts, int max_acts)
{
    if (h->role == OFDM_HS_ROLE_INITIATOR) {
        if (max_acts < 1) {
            return -1;
        }
        goto_state(h, OFDM_HS_I_WAIT_PROBE);
        arm(h, now, h->wait_probe_s);
        return emit_send_probe(&acts[0]);
    }
    goto_state(h, OFDM_HS_R_WAIT_PROBE);
    arm(h, now, h->wait_probe_s);
    return 0;
}

int ofdm_hs_expecting(const ofdm_hs *h)
{
    switch (h->state) {
    case OFDM_HS_I_WAIT_PROBE:
    case OFDM_HS_R_WAIT_PROBE:
        return OFDM_HS_EXPECT_PROBE;
    case OFDM_HS_I_WAIT_BAT:
    case OFDM_HS_I_WAIT_GO:
        return OFDM_HS_EXPECT_MSG;
    case OFDM_HS_R_WAIT_ACK:
        /* Listens for BOTH: the normal next step is the ACK (msg), but an
         * initiator that missed our probe re-probes, and hearing that is
         * the ONLY way its miss can reach us. */
        return OFDM_HS_EXPECT_PROBE | OFDM_HS_EXPECT_MSG;
    case OFDM_HS_R_LINGER:
        return OFDM_HS_EXPECT_MSG;
    default:
        return 0;
    }
}

int ofdm_hs_heard_probe(ofdm_hs *h, const unsigned char *bat, size_t bat_len,
                        double now, ofdm_hs_action *acts, int max_acts)
{
    if (bat_len > OFDM_HS_BAT_MAX) {
        return -2;
    }
    if (h->state == OFDM_HS_I_WAIT_PROBE) {
        memcpy(h->peer_tx_bat, bat, bat_len);
        h->peer_tx_bat_len = bat_len;
        h->has_peer_tx_bat = 1;
        goto_state(h, OFDM_HS_I_WAIT_BAT); /* R sends BAT_REPORT unprompted */
        arm(h, now, h->wait_msg_s);
        return 0;
    }
    if (h->state == OFDM_HS_R_WAIT_PROBE) {
        if (max_acts < 2) {
            return -1;
        }
        memcpy(h->peer_tx_bat, bat, bat_len);
        h->peer_tx_bat_len = bat_len;
        h->has_peer_tx_bat = 1;
        emit_send_probe(&acts[0]);
        emit_send_report(h, &acts[1]);
        goto_state(h, OFDM_HS_R_WAIT_ACK);
        arm(h, now, h->wait_msg_s);
        return 2;
    }
    if (h->state == OFDM_HS_R_WAIT_ACK) {
        /* Initiator re-probed: it never heard our probe/report. Re-measure
         * (fresher estimate) and repeat our whole turn. */
        memcpy(h->peer_tx_bat, bat, bat_len);
        h->peer_tx_bat_len = bat_len;
        h->has_peer_tx_bat = 1;
        h->retries++;
        if (h->retries > h->max_retries) {
            if (max_acts < 1) {
                return -1;
            }
            return fail(h, &acts[0], OFDM_HS_FAIL_PEER_RESTARTING);
        }
        if (max_acts < 2) {
            return -1;
        }
        emit_send_probe(&acts[0]);
        emit_send_report(h, &acts[1]);
        arm(h, now, h->wait_msg_s); /* state stays r_wait_ack */
        return 2;
    }
    /* other states: stale echo, ignore */
    return 0;
}

int ofdm_hs_heard_msg(ofdm_hs *h, int mtype, unsigned proof,
                      const unsigned char *body, size_t body_len, double now,
                      ofdm_hs_action *acts, int max_acts)
{
    unsigned want;

    /* MSG_BAND: recognised on the wire but not acted on -- see ofdm_hs.h.
     * Only reachable if paired with a peer running band_negotiation=True,
     * which this port does not send and both ends default off in the
     * reference. */
    if (mtype == OFDM_HS_MSG_BAND) {
        return 0;
    }
    if (body_len > OFDM_HS_BAT_MAX) {
        return -2;
    }

    if (h->state == OFDM_HS_I_WAIT_BAT && mtype == OFDM_HS_MSG_BAT_REPORT) {
        if (max_acts < 1) {
            return -1;
        }
        memcpy(h->my_tx_bat, body, body_len);
        h->my_tx_bat_len = body_len;
        h->has_my_tx_bat = 1;
        emit_send_ack(h, &acts[0]);
        goto_state(h, OFDM_HS_I_WAIT_GO);
        arm(h, now, h->wait_msg_s);
        return 1;
    }
    if (h->state == OFDM_HS_I_WAIT_GO) {
        if (mtype == OFDM_HS_MSG_GO) {
            want = ofdm_hs_crc16(h->peer_tx_bat_len ? h->peer_tx_bat
                                 : EMPTY_BODY, h->peer_tx_bat_len);
            if (proof == want) {
                int n = 0;
                if (max_acts < (h->skip_fin ? 1 : 2)) {
                    return -1;
                }
                /* FIN is a courtesy that costs airtime + a turn; skipping it
                 * is safe whenever data follows promptly (a driver whose
                 * next act is real DATA can treat that as an implicit FIN
                 * on the responder side -- see meshphy's r_linger notes). */
                if (!h->skip_fin) {
                    emit_send_msg(&acts[n++], OFDM_HS_MSG_FIN, 0u,
                                 EMPTY_BODY, 0);
                }
                n += finish(h, &acts[n]);
                return n;
            }
            /* peer holds a different BAT_R than we sent -- ask again */
            if (max_acts < 1) {
                return -1;
            }
            emit_send_msg(&acts[0], OFDM_HS_MSG_RETRY, 0u, EMPTY_BODY, 0);
            arm(h, now, h->wait_msg_s);
            return 1;
        }
        if (mtype == OFDM_HS_MSG_BAT_REPORT) {
            /* our BAT_ACK was lost; peer retransmitted its report */
            if (max_acts < 1) {
                return -1;
            }
            memcpy(h->my_tx_bat, body, body_len);
            h->my_tx_bat_len = body_len;
            h->has_my_tx_bat = 1;
            emit_send_ack(h, &acts[0]);
            arm(h, now, h->wait_msg_s);
            return 1;
        }
        return 0;
    }
    if (h->state == OFDM_HS_R_WAIT_ACK) {
        if (mtype == OFDM_HS_MSG_BAT_ACK) {
            want = ofdm_hs_crc16(h->peer_tx_bat_len ? h->peer_tx_bat
                                 : EMPTY_BODY, h->peer_tx_bat_len);
            if (max_acts < 1) {
                return -1;
            }
            if (proof == want) {
                memcpy(h->my_tx_bat, body, body_len);
                h->my_tx_bat_len = body_len;
                h->has_my_tx_bat = 1;
                emit_send_go(h, &acts[0]);
                goto_state(h, OFDM_HS_R_LINGER);
                arm(h, now, h->linger_s);
            } else {
                emit_send_report(h, &acts[0]); /* peer got a corrupt report */
                arm(h, now, h->wait_msg_s);
            }
            return 1;
        }
        if (mtype == OFDM_HS_MSG_RETRY) {
            if (max_acts < 1) {
                return -1;
            }
            emit_send_report(h, &acts[0]);
            arm(h, now, h->wait_msg_s);
            return 1;
        }
        return 0;
    }
    if (h->state == OFDM_HS_R_LINGER) {
        if (mtype == OFDM_HS_MSG_FIN) {
            if (max_acts < 1) {
                return -1;
            }
            return finish(h, &acts[0]); /* initiator confirmed GO landed */
        }
        if (mtype == OFDM_HS_MSG_RETRY || mtype == OFDM_HS_MSG_BAT_ACK) {
            /* initiator missed GO (RETRY) or our GO crossed its ACK */
            if (max_acts < 1) {
                return -1;
            }
            emit_send_go(h, &acts[0]);
            arm(h, now, h->linger_s);
            return 1;
        }
        return 0;
    }
    /* anything else: duplicate/stale, ignore */
    return 0;
}

void ofdm_hs_hold_off(ofdm_hs *h, double seconds)
{
    if (h->deadline != OFDM_HS_TIME_UNSET && seconds > 0.0) {
        h->deadline += seconds;
    }
}

int ofdm_hs_tick(ofdm_hs *h, double now, ofdm_hs_action *acts, int max_acts)
{
    if (h->deadline == OFDM_HS_TIME_UNSET || now < h->deadline) {
        return 0;
    }
    h->retries++;
    if (h->state == OFDM_HS_R_LINGER) {
        if (max_acts < 1) {
            return -1;
        }
        return finish(h, &acts[0]); /* linger over, assume GO landed */
    }
    if (h->retries > h->max_retries) {
        if (max_acts < 1) {
            return -1;
        }
        return fail(h, &acts[0], OFDM_HS_FAIL_TIMEOUT);
    }
    switch (h->state) {
    case OFDM_HS_I_WAIT_PROBE:
        if (max_acts < 1) {
            return -1;
        }
        arm(h, now, h->wait_probe_s);
        return emit_send_probe(&acts[0]);
    case OFDM_HS_R_WAIT_PROBE:
        arm(h, now, h->wait_probe_s); /* keep listening, no action */
        return 0;
    case OFDM_HS_I_WAIT_BAT:
    case OFDM_HS_I_WAIT_GO:
        if (max_acts < 1) {
            return -1;
        }
        emit_send_msg(&acts[0], OFDM_HS_MSG_RETRY, 0u, EMPTY_BODY, 0);
        arm(h, now, h->wait_msg_s);
        return 1;
    case OFDM_HS_R_WAIT_ACK:
        /* Repeat the WHOLE turn (probe + report), not the report alone. If
         * the initiator missed our probe it is stuck in i_wait_probe, where
         * a bare REPORT is undecodable to it (its binding only attempts
         * probe decode there) -- REPORT-only retransmission deadlocked both
         * ends: each re-sent something the other could not hear. */
        if (max_acts < 2) {
            return -1;
        }
        emit_send_probe(&acts[0]);
        emit_send_report(h, &acts[1]);
        arm(h, now, h->wait_msg_s);
        return 2;
    default:
        return 0;
    }
}
