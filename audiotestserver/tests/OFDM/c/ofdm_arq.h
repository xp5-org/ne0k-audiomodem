/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_arq.h -- stop-and-wait ARQ with piggybacked ACK, freestanding C89.
 *
 * Port of the protocol half of OfdmLink (_on_frame, _send_full, and the
 * "ACK at once on our own half of the band" rule in _full_rx).  Pure logic:
 * no DSP, no audio, no clock.  Time is injected, in seconds, so the caller
 * can run it from a sound-card callback, a DOS timer tick, or a test.
 *
 * Wire semantics (the link frame is ofdm_dx's seq/ack/data):
 *   - a data frame carries its own seq and ack = the next seq we expect;
 *   - the sender of seq S is acked by any frame with ack == S+1;
 *   - a frame with no data is a pure ACK and never moves rx_next;
 *   - a data frame whose seq is rx_next-1 is a duplicate: it is dropped but
 *     still re-ACKed, because the ACK we owe may be the one that was lost.
 *
 * Driving it:
 *     ofdm_arq_send()      queue one frame (<=255 bytes); one in flight at a time
 *     ofdm_arq_on_frame()  feed every frame the receiver delivers
 *     ofdm_arq_tx_done()   the transmitter has finished playing the last TX action
 *     ofdm_arq_poll()      drain actions until it returns 0; call after the two above
 *                          and on every timer tick
 *
 * Contract: no malloc, no stdio, no globals.  The state is one struct the
 * caller owns (about 600 bytes).
 */
#ifndef OFDM_ARQ_H
#define OFDM_ARQ_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OFDM_ARQ_MAX_DATA 255

typedef enum {
    OFDM_ARQ_ACT_NONE = 0,
    OFDM_ARQ_ACT_TX,          /* transmit (seq, ack, data[len]); len may be 0 (pure ACK) */
    OFDM_ARQ_ACT_DELIVER,     /* data[len] arrived in order: hand it to the application */
    OFDM_ARQ_ACT_SEND_DONE    /* the frame passed to send() was acked (ok=1) or given up on (ok=0) */
} ofdm_arq_act_kind;

typedef struct {
    ofdm_arq_act_kind kind;
    int  seq, ack, len;
    int  ok;                  /* SEND_DONE */
    int  retry;               /* TX: 0 first attempt, n>0 the nth retransmission */
    const unsigned char *data;/* valid until the next call into this handle */
} ofdm_arq_action;

typedef struct {
    /* configuration */
    int    retries;           /* retransmissions after the first attempt */
    double ack_wait_s;        /* from tx_done to giving up on one attempt; must exceed the
                               * receiver's hang time plus the far end's own queued TX */
    double gap_s;             /* quiet after every TX so our bursts never run together
                               * (must exceed the receiver's hang_s) */

    /* state -- treat as private */
    int    tx_seq, rx_next;
    int    inflight;          /* 1 while a frame is awaiting its ACK */
    int    if_seq, if_len, if_sends;
    int    want_tx;           /* a TX of the in-flight frame is due */
    int    awaiting_tx_done;
    int    pure_ack_tx;       /* the TX awaiting tx_done is a bare ACK */
    double deadline;          /* 0 = none armed */
    double quiet_until;       /* no TX before this */
    int    owe_ack;
    int    want_done, done_ok;
    int    dlv_len;           /* >0 while an in-order frame waits to be polled */
    unsigned char if_data[OFDM_ARQ_MAX_DATA];
    unsigned char dlv_data[OFDM_ARQ_MAX_DATA];

    /* counters */
    unsigned long tx_frames, rx_frames, retry_count, fail_count, dup_count, gap_count;
} ofdm_arq;

void   ofdm_arq_init(ofdm_arq *a, int retries, double ack_wait_s, double gap_s);
void   ofdm_arq_configure(ofdm_arq *a, int retries, double ack_wait_s, double gap_s);  /* keeps state */
size_t ofdm_arq_sizeof(void);              /* for a caller that cannot see the struct (ctypes) */

/* 0 queued; -1 bad args; -2 a frame is already in flight. */
int  ofdm_arq_send(ofdm_arq *a, const unsigned char *data, int n);

/* A decoded link frame.  Returns 1 if it carried new in-order data, 0 otherwise,
 * -1 if the previous delivery has not been polled yet (the frame is dropped; the
 * far end will retry). */
int  ofdm_arq_on_frame(ofdm_arq *a, int seq, int ack, const unsigned char *data,
                       int n, double now);

/* The last ACT_TX has finished playing at `now`; starts the ack timer and the
 * post-TX quiet gap. */
void ofdm_arq_tx_done(ofdm_arq *a, double now);

/* Tell the ARQ whether a burst is arriving right now (ofdm_rxs_busy).  While one
 * is, the ack timer is held: it may be the ACK we are waiting for, and
 * retransmitting over it only wastes airtime.  The wait restarts when it ends. */
void ofdm_arq_air_busy(ofdm_arq *a, int busy, double now);

/* Next action, or 0 when idle.  Priority: DELIVER, SEND_DONE, pure ACK, data TX. */
int  ofdm_arq_poll(ofdm_arq *a, double now, ofdm_arq_action *act);

int  ofdm_arq_busy(const ofdm_arq *a);     /* 1 while a send() is unfinished */

#ifdef __cplusplus
}
#endif
#endif
