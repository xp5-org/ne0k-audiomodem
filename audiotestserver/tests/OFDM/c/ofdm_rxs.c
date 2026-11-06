/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_rxs.c -- see ofdm_rxs.h. */
#include <math.h>
#include "ofdm_rxs.h"
#include "ofdm_rxf.h"

#define LEAD_S 0.2      /* decode window starts this long before the burst   */
#define TAIL_S 0.3      /* ... and ends this long after it                    */
#define BLOCK_S 0.05    /* RMS gate block, as in the reference                */
#define BUSY_LOOK_S 0.4

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

struct ofdm_rxs {
    ofdm_dx  *dx;
    ofdm_rxf *rxf;
    int       fs, win;
    double    thr2, hang_s;     /* thr squared: compare energy, no sqrt per block */
    long      ring, total;      /* ring size, filtered samples seen */
    double   *buf;              /* ring */
    double   *lin;              /* contiguous decode window */
    double   *tmp;              /* filter output block */
    long      blk_n;            /* samples accumulated into the current gate block */
    double    blk_e;
    int       in_burst;
    long      b_start, b_last;  /* first busy block start, end of the last busy block */
    long      last_busy_end;    /* for ofdm_rxs_busy */
    int       q_head, q_n;
    unsigned long dropped;
    ofdm_rxs_event q[OFDM_RXS_QUEUE];
};

#define TMP_BLOCK 512

size_t ofdm_rxs_workspace(const ofdm_rxs_params *p)
{
    size_t fw = 0;
    if (!p || p->fs <= 0 || p->ring_samples < p->fs || p->hang_s <= 0.0) return 0;
    if (p->taps && p->ntaps > 0) {
        fw = ofdm_rxf_workspace(p->ntaps);
        if (!fw) return 0;
    }
    return al8(sizeof(struct ofdm_rxs)) + al8(fw)
         + 2 * al8((size_t)p->ring_samples * sizeof(double))
         + al8(TMP_BLOCK * sizeof(double));
}

int ofdm_rxs_init(ofdm_rxs **sp, void *mem, size_t memlen, ofdm_dx *dx,
                  const ofdm_rxs_params *p)
{
    struct ofdm_rxs *s;
    unsigned char *b = (unsigned char *)mem;
    size_t need = ofdm_rxs_workspace(p), fw = 0, off;
    long i;

    if (!sp || !mem || !dx || !p || need == 0) return -1;
    if (memlen < need) return -2;
    if (p->taps && p->ntaps > 0) fw = ofdm_rxf_workspace(p->ntaps);

    s = (struct ofdm_rxs *)(void *)b;
    off = al8(sizeof(struct ofdm_rxs));
    s->rxf = 0;
    if (fw) {
        if (ofdm_rxf_init(&s->rxf, b + off, fw, p->taps, p->ntaps) != 0) return -3;
        off += al8(fw);
    }
    s->buf = (double *)(void *)(b + off);
    off += al8((size_t)p->ring_samples * sizeof(double));
    s->lin = (double *)(void *)(b + off);
    off += al8((size_t)p->ring_samples * sizeof(double));
    s->tmp = (double *)(void *)(b + off);

    s->dx = dx;
    s->fs = p->fs;
    s->win = (int)(BLOCK_S * p->fs);
    if (s->win < 1) s->win = 1;
    s->thr2 = p->thr_rms * p->thr_rms;
    s->hang_s = p->hang_s;
    s->ring = p->ring_samples;
    s->total = 0;
    s->blk_n = 0; s->blk_e = 0.0;
    s->in_burst = 0; s->b_start = s->b_last = 0;
    s->last_busy_end = -(long)(10 * p->fs);
    s->q_head = s->q_n = 0;
    s->dropped = 0;
    for (i = 0; i < OFDM_RXS_QUEUE; i++) s->q[i].kind = OFDM_RXS_EVT_ERR;
    *sp = s;
    return 0;
}

long ofdm_rxs_total(const ofdm_rxs *s) { return s ? s->total : 0; }
int  ofdm_rxs_delay(const ofdm_rxs *s) { return s && s->rxf ? ofdm_rxf_delay(s->rxf) : 0; }
unsigned long ofdm_rxs_dropped(const ofdm_rxs *s) { return s ? s->dropped : 0; }

int ofdm_rxs_busy(const ofdm_rxs *s)
{
    if (!s) return 0;
    return s->in_burst || (s->total - s->last_busy_end) < (long)(BUSY_LOOK_S * s->fs);
}

static ofdm_rxs_event *push_slot(struct ofdm_rxs *s)
{
    if (s->q_n >= OFDM_RXS_QUEUE) { s->dropped++; return 0; }
    return &s->q[(s->q_head + s->q_n) % OFDM_RXS_QUEUE];
}

/* Close the open burst and decode it. */
static void close_burst(struct ofdm_rxs *s)
{
    long a = s->b_start, b = s->b_last;
    long w0 = a - (long)(LEAD_S * s->fs), w1 = b + (long)(TAIL_S * s->fs);
    long oldest = s->total - s->ring, i, n;
    int rc, truncated = 0;
    ofdm_rxs_event *ev = push_slot(s);
    ofdm_dx_rxinfo info;

    s->in_burst = 0;
    if (!ev) return;
    if (w0 < 0) w0 = 0;
    if (w0 < oldest) { w0 = oldest; truncated = 1; }
    if (w1 > s->total) w1 = s->total;
    n = w1 - w0;

    ev->a = a; ev->b = b;
    ev->seq = ev->ack = ev->len = 0;
    ev->n_chunks = ev->n_found = ev->n_bad = ev->errata = 0;
    ev->evm = 0.0;
    ev->kind = OFDM_RXS_EVT_ERR;
    rc = 0;
    if (n > 0 && !truncated) {
        for (i = 0; i < n; i++) s->lin[i] = s->buf[(w0 + i) % s->ring];
        rc = ofdm_dx_rx_frame(s->dx, s->lin, (int)n, ev->data, &info);
        ev->n_found = info.n_found; ev->n_bad = info.n_bad;
        ev->errata = info.errata; ev->evm = info.evm;
    }
    if (rc == 1) {
        ev->kind = OFDM_RXS_EVT_FRAME;
        ev->seq = info.seq; ev->ack = info.ack; ev->len = info.len;
        ev->n_chunks = info.n_chunks;
    }
    s->q_n++;
}

int ofdm_rxs_feed(ofdm_rxs *s, const double *x, long n)
{
    long done = 0, k, m, i;
    if (!s || !x || n < 0) return -1;

    while (done < n) {
        m = n - done;
        if (m > TMP_BLOCK) m = TMP_BLOCK;
        if (s->rxf) ofdm_rxf_process(s->rxf, x + done, s->tmp, m);
        else for (i = 0; i < m; i++) s->tmp[i] = x[done + i];

        for (k = 0; k < m; k++) {
            double v = s->tmp[k];
            s->buf[s->total % s->ring] = v;
            s->total++;
            s->blk_e += v * v;
            if (++s->blk_n == s->win) {
                long end = s->total, start = end - s->win;
                if (s->blk_e / s->win > s->thr2) {
                    if (!s->in_burst) { s->in_burst = 1; s->b_start = start; }
                    s->b_last = end;
                    s->last_busy_end = end;
                }
                s->blk_n = 0; s->blk_e = 0.0;
            }
            if (s->in_burst) {
                /* quiet long enough, or a burst that has outgrown the ring */
                if (s->total - s->b_last >= (long)(s->hang_s * s->fs)
                    || s->total - s->b_start >= s->ring - (long)(s->fs / 10))
                    close_burst(s);
            }
        }
        done += m;
    }
    return s->q_n;
}

int ofdm_rxs_pop(ofdm_rxs *s, ofdm_rxs_event *ev)
{
    if (!s || !ev || s->q_n == 0) return 0;
    *ev = s->q[s->q_head];
    s->q_head = (s->q_head + 1) % OFDM_RXS_QUEUE;
    s->q_n--;
    return 1;
}
