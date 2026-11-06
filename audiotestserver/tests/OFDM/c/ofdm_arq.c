/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_arq.c -- see ofdm_arq.h. */
#include "ofdm_arq.h"

void ofdm_arq_init(ofdm_arq *a, int retries, double ack_wait_s, double gap_s)
{
    unsigned char *p = (unsigned char *)a;
    size_t i;
    for (i = 0; i < sizeof *a; i++) p[i] = 0;     /* no <string.h> needed */
    a->retries = retries < 0 ? 0 : retries;
    a->ack_wait_s = ack_wait_s;
    a->gap_s = gap_s < 0.0 ? 0.0 : gap_s;
}

void ofdm_arq_configure(ofdm_arq *a, int retries, double ack_wait_s, double gap_s)
{
    if (!a) return;
    a->retries = retries < 0 ? 0 : retries;
    a->ack_wait_s = ack_wait_s;
    a->gap_s = gap_s < 0.0 ? 0.0 : gap_s;
}

size_t ofdm_arq_sizeof(void) { return sizeof(ofdm_arq); }

int ofdm_arq_busy(const ofdm_arq *a)
{
    return a ? (a->inflight || a->want_done) : 0;
}

int ofdm_arq_send(ofdm_arq *a, const unsigned char *data, int n)
{
    int i;
    if (!a || n < 1 || n > OFDM_ARQ_MAX_DATA || !data) return -1;
    if (a->inflight || a->want_done) return -2;
    for (i = 0; i < n; i++) a->if_data[i] = data[i];
    a->if_len = n;
    a->if_seq = a->tx_seq;
    a->tx_seq = (a->tx_seq + 1) & 0xFF;
    a->if_sends = 0;
    a->inflight = 1;
    a->want_tx = 1;
    a->awaiting_tx_done = 0;
    a->deadline = 0.0;
    return 0;
}

int ofdm_arq_on_frame(ofdm_arq *a, int seq, int ack, const unsigned char *data,
                      int n, double now)
{
    int i, fresh = 0;
    (void)now;
    if (!a || n < 0 || n > OFDM_ARQ_MAX_DATA || (n && !data)) return -1;
    if (n && a->dlv_len) return -1;               /* application has not drained */

    a->rx_frames++;
    if (a->inflight && ack == ((a->if_seq + 1) & 0xFF)) {
        a->inflight = 0;
        a->want_tx = 0;
        a->awaiting_tx_done = 0;
        a->deadline = 0.0;
        a->want_done = 1;
        a->done_ok = 1;
    }
    if (n) {
        a->owe_ack = 1;                           /* also for a duplicate */
        if (seq == ((a->rx_next - 1) & 0xFF)) {
            a->dup_count++;
        } else {
            if (seq != a->rx_next) a->gap_count++;
            a->rx_next = (seq + 1) & 0xFF;
            for (i = 0; i < n; i++) a->dlv_data[i] = data[i];
            a->dlv_len = n;
            fresh = 1;
        }
    }
    return fresh;
}

void ofdm_arq_tx_done(ofdm_arq *a, double now)
{
    if (!a) return;
    if (a->awaiting_tx_done) {
        a->awaiting_tx_done = 0;
        /* a bare ACK says nothing about whether our own frame was heard */
        if (a->inflight && !a->pure_ack_tx) a->deadline = now + a->ack_wait_s;
        a->pure_ack_tx = 0;
        a->quiet_until = now + a->gap_s;
    }
}

void ofdm_arq_air_busy(ofdm_arq *a, int busy, double now)
{
    if (a && busy && a->deadline > 0.0 && a->deadline < now + a->ack_wait_s)
        a->deadline = now + a->ack_wait_s;
}

int ofdm_arq_poll(ofdm_arq *a, double now, ofdm_arq_action *act)
{
    if (!a || !act) return 0;
    act->kind = OFDM_ARQ_ACT_NONE;
    act->seq = act->ack = act->len = act->ok = act->retry = 0;
    act->data = 0;

    if (a->dlv_len) {
        act->kind = OFDM_ARQ_ACT_DELIVER;
        act->data = a->dlv_data;
        act->len = a->dlv_len;
        act->seq = (a->rx_next - 1) & 0xFF;
        a->dlv_len = 0;
        return 1;
    }
    if (a->want_done) {
        a->want_done = 0;
        act->kind = OFDM_ARQ_ACT_SEND_DONE;
        act->ok = a->done_ok;
        act->seq = a->if_seq;
        return 1;
    }

    /* ack timer */
    if (a->inflight && !a->want_tx && !a->awaiting_tx_done
        && a->deadline > 0.0 && now >= a->deadline) {
        a->deadline = 0.0;
        if (a->if_sends > a->retries) {
            a->inflight = 0;
            a->fail_count++;
            a->want_done = 1;
            a->done_ok = 0;
            return ofdm_arq_poll(a, now, act);
        }
        a->want_tx = 1;
        a->retry_count++;
    }

    /* a pure ACK goes out at once, on its own half of the band, and does not
     * wait for the data frame (which may be many seconds away) */
    if (a->owe_ack && !a->awaiting_tx_done && now >= a->quiet_until) {
        a->owe_ack = 0;
        act->kind = OFDM_ARQ_ACT_TX;
        act->seq = (a->tx_seq - 1) & 0xFF;
        act->ack = a->rx_next;
        act->len = 0;
        a->tx_frames++;
        a->awaiting_tx_done = 1;
        a->pure_ack_tx = 1;
        return 1;
    }
    if (a->want_tx && !a->awaiting_tx_done && now >= a->quiet_until) {
        a->want_tx = 0;
        act->kind = OFDM_ARQ_ACT_TX;
        act->seq = a->if_seq;
        act->ack = a->rx_next;
        act->data = a->if_data;
        act->len = a->if_len;
        act->retry = a->if_sends;
        a->if_sends++;
        a->tx_frames++;
        a->awaiting_tx_done = 1;
        a->pure_ack_tx = 0;
        return 1;
    }
    return 0;
}
