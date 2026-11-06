/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_rxs.h -- streaming receive path: filter -> burst gate -> decode, C89.
 *
 * Port of OfdmLink._collect / _bursts / _listen_full / _full_rx for a caller
 * that hands over samples as they arrive (sound-card callback, DOS DMA block).
 *
 *     raw mic samples -> ofdm_rxf (own band out) -> ring
 *                     -> 50 ms RMS gate, merged across gaps <= hang_s
 *                     -> when a burst has been quiet for hang_s: ofdm_dx_rx_frame
 *                     -> event queue (frame / undecodable burst)
 *
 * Each burst is decoded exactly once, so a retransmitted or duplicated frame
 * reaches the ARQ as one event per time it was actually on the air.
 *
 * POSITIONS are in the FILTERED stream, which lags the raw one by
 * ofdm_rxs_delay() samples.  Nothing here needs raw positions.
 *
 * Contract: no malloc, no stdio, no globals, one caller-owned workspace.
 */
#ifndef OFDM_RXS_H
#define OFDM_RXS_H
#include <stddef.h>
#include "ofdm_dx.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OFDM_RXS_QUEUE 4

typedef struct {
    int    fs;
    double thr_rms;       /* busy when a 50 ms block's RMS exceeds this (linear) */
    double hang_s;        /* quiet this long closes a burst */
    long   ring_samples;  /* capture memory; must hold the longest burst + lead + tail */
    const double *taps;   /* own-band rejection FIR from ofdm_rxf_design, or NULL for none */
    int    ntaps;
} ofdm_rxs_params;

typedef enum {
    OFDM_RXS_EVT_FRAME = 1,   /* a link frame decoded */
    OFDM_RXS_EVT_ERR = 2      /* a burst that did not decode (RS/CRC/missing chunk) */
} ofdm_rxs_evt_kind;

typedef struct {
    ofdm_rxs_evt_kind kind;
    long   a, b;              /* burst span, absolute filtered-stream samples */
    int    seq, ack, len;
    int    n_chunks, n_found, n_bad, errata;
    double evm;
    unsigned char data[OFDM_DX_MAX_DATA];
} ofdm_rxs_event;

typedef struct ofdm_rxs ofdm_rxs;

size_t ofdm_rxs_workspace(const ofdm_rxs_params *p);

/* `dx` must outlive the stream.  Errors: -1 bad args, -2 memlen too small,
 * -3 filter init failed. */
int ofdm_rxs_init(ofdm_rxs **s, void *mem, size_t memlen, ofdm_dx *dx,
                  const ofdm_rxs_params *p);

/* Push n raw samples.  Returns the number of events now queued (0..QUEUE). */
int ofdm_rxs_feed(ofdm_rxs *s, const double *x, long n);

/* Pop the oldest event; 1 if one was returned, 0 if the queue is empty. */
int ofdm_rxs_pop(ofdm_rxs *s, ofdm_rxs_event *ev);

/* True while a burst is open (or the last 0.4 s were busy): the far end, or our
 * own leak that the filter did not remove, is on the air. */
int  ofdm_rxs_busy(const ofdm_rxs *s);

long ofdm_rxs_total(const ofdm_rxs *s);        /* samples consumed so far */
int  ofdm_rxs_delay(const ofdm_rxs *s);        /* filter group delay */
unsigned long ofdm_rxs_dropped(const ofdm_rxs *s);   /* events lost to a full queue */

#ifdef __cplusplus
}
#endif
#endif
