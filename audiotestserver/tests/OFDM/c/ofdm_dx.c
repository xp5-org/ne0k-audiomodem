/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_dx.c -- see ofdm_dx.h. */
#include <math.h>
#include "ofdm_dx.h"
#include "ofdm_mod.h"
#include "ofdm_dem.h"
#include "ofdm_sync.h"
#include "ofdm_frame.h"
#include "rs255.h"

#define PILOT_AMP 0.70710678118654752

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }
static int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct ofdm_dx {
    ofdm_cfg  tx, rx;
    ofdm_mod *tx_mod;
    ofdm_mod *rx_mod;          /* only its preamble is used, at init */
    rs255    *rs;
    ofdm_sync *sync;
    ofdm_dem *dem;
    int       r_depth, r_k, r_wire, r_body;
    int       rx_max_chunks, tx_max_chunks;
    int      *rx_bat;
    int      *pilots;
    double   *ce_re, *ce_im;
    unsigned char *wire, *msg, *blk, *frame, *got, *txbuf;
    int      *starts;
};

typedef struct {
    int    t_bps, r_bps, r_ds, r_n_pilots;
    int    widths[5], n_w, n_pts, n_bits;
    size_t sz_mod_t, sz_mod_r, sz_rs, sz_sync, sz_dem;
    int    rx_max_chunks, tx_max_chunks;
    int    r_depth, r_k, r_wire;
} dx_plan;

static int sum_bits(const int *b, int n)
{
    int i, s = 0;
    for (i = 0; i < n; i++) s += b[i];
    return s;
}

static int dx_plan_make(dx_plan *pl, const ofdm_dx_params *p)
{
    int i, t;
    if (!p || !p->tx_bat || !p->rx_bat) return -1;
    if (p->tx_cfg.chunk_bytes < 1 || p->rx_cfg.chunk_bytes < 1) return -1;
    if (p->tx_cfg.n_data_carriers < 1 || p->rx_cfg.n_data_carriers < 1) return -1;

    pl->t_bps = sum_bits(p->tx_bat, p->tx_cfg.n_data_carriers);
    pl->r_bps = sum_bits(p->rx_bat, p->rx_cfg.n_data_carriers);
    if (pl->t_bps < 1 || pl->r_bps < 1) return -1;
    if (ofdm_interleave_geometry(&p->rx_cfg, pl->r_bps, &pl->r_depth, &pl->r_k,
                                 &pl->r_wire) != 0) return -1;
    pl->r_ds = ofdm_data_syms_per_pkt(&p->rx_cfg, pl->r_bps);
    if (pl->r_ds < 1) return -1;
    pl->r_n_pilots = p->rx_cfg.n_pilots;

    pl->n_w = ofdm_bat_widths(p->rx_bat, p->rx_cfg.n_data_carriers, pl->widths, 5);
    if (pl->n_w < 0) return -1;
    /* the dem wants nonzero widths only, ascending */
    t = 0;
    for (i = 0; i < pl->n_w; i++)
        if (pl->widths[i] > 0) pl->widths[t++] = pl->widths[i];
    pl->n_w = t;
    if (t < 1) return -1;
    pl->n_pts = pl->n_bits = 0;
    for (i = 0; i < t; i++) {
        pl->n_pts  += 1 << pl->widths[i];
        pl->n_bits += (1 << pl->widths[i]) * pl->widths[i];
    }

    pl->rx_max_chunks = ceil_div(OFDM_DX_MAX_FRAME, p->rx_cfg.chunk_bytes);
    pl->tx_max_chunks = ceil_div(OFDM_DX_MAX_FRAME, p->tx_cfg.chunk_bytes);

    pl->sz_mod_t = ofdm_mod_workspace(&p->tx_cfg, p->tx_bat);
    pl->sz_mod_r = ofdm_mod_workspace(&p->rx_cfg, p->rx_bat);
    pl->sz_rs    = rs255_workspace(p->rx_cfg.ecc_bytes);
    pl->sz_sync  = ofdm_sync_workspace(&p->rx_cfg);
    pl->sz_dem   = ofdm_dem_workspace(p->rx_cfg.n_fft, p->rx_cfg.cp_len,
                                      p->rx_cfg.n_occupied, p->rx_cfg.n_pilots,
                                      p->rx_cfg.n_data_carriers, pl->r_ds,
                                      p->rx_cfg.sync_period, pl->r_wire,
                                      pl->n_pts, pl->n_bits);
    if (!pl->sz_mod_t || !pl->sz_mod_r || !pl->sz_rs || !pl->sz_sync || !pl->sz_dem)
        return -1;
    return 0;
}

/* One layout routine for sizing AND carving, so the two cannot drift. */
static size_t dx_layout(const dx_plan *pl, const ofdm_dx_params *p,
                        unsigned char *base, struct ofdm_dx *d)
{
    size_t off = 0, n;
    const ofdm_cfg *r = &p->rx_cfg;

#define TAKE(field, type, bytes) \
    do { n = al8((size_t)(bytes)); \
         if (base) (field) = (type)(void *)(base + off); \
         off += n; } while (0)

    off = al8(sizeof(struct ofdm_dx));
    /* sub-core blocks are carved by the caller of this macro set below */
    off += al8(pl->sz_mod_t);
    off += al8(pl->sz_mod_r);
    off += al8(pl->sz_rs);
    off += al8(pl->sz_sync);
    off += al8(pl->sz_dem);
    if (d) {
        TAKE(d->rx_bat, int *, r->n_data_carriers * sizeof(int));
        TAKE(d->pilots, int *, r->n_pilots * sizeof(int));
        TAKE(d->ce_re, double *, r->n_occupied * sizeof(double));
        TAKE(d->ce_im, double *, r->n_occupied * sizeof(double));
        TAKE(d->wire, unsigned char *, pl->r_wire);
        TAKE(d->msg, unsigned char *, r->seq_bytes + r->chunk_bytes);
        TAKE(d->blk, unsigned char *, pl->r_k + r->ecc_bytes);
        TAKE(d->frame, unsigned char *, pl->rx_max_chunks * r->chunk_bytes);
        TAKE(d->got, unsigned char *, pl->rx_max_chunks);
        TAKE(d->txbuf, unsigned char *, pl->tx_max_chunks * p->tx_cfg.chunk_bytes);
        TAKE(d->starts, int *, OFDM_DX_MAX_STARTS * sizeof(int));
    } else {
        off += al8(r->n_data_carriers * sizeof(int));
        off += al8(r->n_pilots * sizeof(int));
        off += 2 * al8(r->n_occupied * sizeof(double));
        off += al8(pl->r_wire);
        off += al8(r->seq_bytes + r->chunk_bytes);
        off += al8(pl->r_k + r->ecc_bytes);
        off += al8(pl->rx_max_chunks * r->chunk_bytes);
        off += al8(pl->rx_max_chunks);
        off += al8(pl->tx_max_chunks * p->tx_cfg.chunk_bytes);
        off += al8(OFDM_DX_MAX_STARTS * sizeof(int));
    }
#undef TAKE
    return off;
}

size_t ofdm_dx_workspace(const ofdm_dx_params *p)
{
    dx_plan pl;
    if (dx_plan_make(&pl, p) != 0) return 0;
    return dx_layout(&pl, p, 0, 0);
}

int ofdm_dx_init(ofdm_dx **out, void *mem, size_t memlen, const ofdm_dx_params *p)
{
    dx_plan pl;
    struct ofdm_dx *d;
    unsigned char *b = (unsigned char *)mem;
    size_t off, need;
    int i, rc, tab_b[4], tab_n[4], n_pts, n_bits;
    double pts_re[96], pts_im[96];
    unsigned char tab_bits[480];
    const ofdm_cfg *r;
    unsigned char *m_t, *m_r, *m_rs, *m_sy, *m_de;
    double amp;

    if (!out || !mem || !p) return -1;
    if (dx_plan_make(&pl, p) != 0) return -1;
    if (pl.n_pts > 96 || pl.n_bits > 480) return -4;
    need = dx_layout(&pl, p, 0, 0);
    if (memlen < need) return -2;

    d = (struct ofdm_dx *)(void *)b;
    d->tx = p->tx_cfg;
    d->rx = p->rx_cfg;
    r = &d->rx;
    off = al8(sizeof(struct ofdm_dx));
    m_t  = b + off;  off += al8(pl.sz_mod_t);
    m_r  = b + off;  off += al8(pl.sz_mod_r);
    m_rs = b + off;  off += al8(pl.sz_rs);
    m_sy = b + off;  off += al8(pl.sz_sync);
    m_de = b + off;
    {
        dx_layout(&pl, p, b, d);

        if (ofdm_mod_init(&d->tx_mod, m_t, pl.sz_mod_t, &d->tx, p->tx_bat) != 0) return -3;
        if (ofdm_mod_init(&d->rx_mod, m_r, pl.sz_mod_r, &d->rx, p->rx_bat) != 0) return -3;
        if (rs255_init(&d->rs, m_rs, pl.sz_rs, r->ecc_bytes) != 0) return -3;
        if (ofdm_sync_init(&d->sync, m_sy, pl.sz_sync, &d->rx,
                           ofdm_mod_sc_preamble(d->rx_mod)) != 0) return -3;

        /* reference symbols, exactly as the modulator builds them */
        amp = OFDM_REF_SYM_BOOST / sqrt(2.0);
        ofdm_known_bpsk(d->ce_re, r->n_occupied, OFDM_SEED_CE, amp);
        for (i = 0; i < r->n_occupied; i++) d->ce_im[i] = 0.0;
        for (i = 0; i < r->n_pilots; i++) d->pilots[i] = i * r->pilot_spacing;
        for (i = 0; i < r->n_data_carriers; i++) d->rx_bat[i] = p->rx_bat[i];

        rc = ofdm_mod_tables_flat(pl.widths, pl.n_w, tab_b, tab_n, pts_re, pts_im,
                                  tab_bits, 96, 480, &n_pts, &n_bits);
        if (rc != 0) return -3;
        if (ofdm_dem_init(&d->dem, m_de, pl.sz_dem, r->n_fft, r->cp_len, r->bin_lo,
                          r->n_occupied, d->ce_re, d->ce_im, d->ce_re, d->ce_im,
                          d->pilots, r->n_pilots, PILOT_AMP, PILOT_AMP,
                          d->rx_bat, r->n_data_carriers, pl.r_ds, r->sync_period,
                          pl.r_wire, OFDM_SEED_DATA, tab_b, tab_n, pl.n_w,
                          pts_re, pts_im, tab_bits) != 0) return -3;
    }
    d->r_depth = pl.r_depth; d->r_k = pl.r_k; d->r_wire = pl.r_wire;
    d->r_body = ofdm_mod_body_syms(d->rx_mod);
    d->rx_max_chunks = pl.rx_max_chunks;
    d->tx_max_chunks = pl.tx_max_chunks;
    *out = d;
    return 0;
}

unsigned ofdm_dx_crc16(const unsigned char *data, size_t len)
{
    unsigned crc = 0xFFFFu;
    size_t i;
    int b;
    for (i = 0; i < len; i++) {
        crc ^= ((unsigned)data[i]) << 8;
        for (b = 0; b < 8; b++)
            crc = ((crc & 0x8000u) ? ((crc << 1) ^ 0x1021u) : (crc << 1)) & 0xFFFFu;
    }
    return crc;
}

int ofdm_dx_capacity(const ofdm_dx *dx, int max_chunks)
{
    int n;
    if (!dx || max_chunks < 1) return -1;
    n = max_chunks * dx->tx.chunk_bytes - OFDM_DX_HDR;
    if (n > OFDM_DX_MAX_DATA) n = OFDM_DX_MAX_DATA;
    return n;
}

long ofdm_dx_tx_samples(const ofdm_dx *dx, int n)
{
    if (!dx || n < 0 || n > OFDM_DX_MAX_DATA) return -1;
    return ofdm_mod_signal_samples(dx->tx_mod, OFDM_DX_HDR + n);
}

long ofdm_dx_tx_frame(ofdm_dx *dx, int seq, int ack, const unsigned char *data,
                      int n, double *out, long out_cap)
{
    unsigned crc;
    unsigned char *f;
    long need, ns;
    int i, total;

    if (!dx || !out || n < 0 || n > OFDM_DX_MAX_DATA || (n && !data)) return -1;
    f = dx->txbuf;
    total = OFDM_DX_HDR + n;
    f[0] = (unsigned char)(seq & 0xFF);
    f[1] = (unsigned char)(ack & 0xFF);
    f[2] = (unsigned char)n;
    for (i = 0; i < n; i++) f[OFDM_DX_HDR + i] = data[i];
    /* crc over seq ack len data, in that order, skipping the crc field */
    {
        unsigned char tmp[3 + OFDM_DX_MAX_DATA];
        for (i = 0; i < 3; i++) tmp[i] = f[i];
        for (i = 0; i < n; i++) tmp[3 + i] = data[i];
        crc = ofdm_dx_crc16(tmp, (size_t)(3 + n));
    }
    f[3] = (unsigned char)(crc >> 8);
    f[4] = (unsigned char)(crc & 0xFF);

    need = ofdm_mod_signal_samples(dx->tx_mod, total);
    if (need < 0) return -1;
    if (out_cap < need) return -2;
    ns = ofdm_mod_encode(dx->tx_mod, f, total, out, out_cap);
    if (ns < 0) return ns == -2 ? -2 : -3;
    return ns;
}

long ofdm_dx_rx_frame_samples(const ofdm_dx *dx)
{
    return dx ? ofdm_mod_frame_samples(dx->rx_mod) : 0;
}

const ofdm_cfg *ofdm_dx_tx_cfg(const ofdm_dx *dx) { return dx ? &dx->tx : 0; }
const ofdm_cfg *ofdm_dx_rx_cfg(const ofdm_dx *dx) { return dx ? &dx->rx : 0; }

/* Acquisition passes, loosest first.  The defaults are tuned for wideband
 * noise.  After the one-sided receive filter the noise is band-limited to the
 * far sub-band (a narrow one is ~1 kHz), where the Schmidl-Cox metric and the
 * matched-filter score of pure noise are far higher -- measured 3-6 candidates
 * against the true 1-2 -- and a spurious candidate ahead of a real frame can
 * suppress it.  A stricter pass drops those and is only tried when the loose
 * one did not produce a complete link frame. */
#define N_PASSES 3
static const double PASS_THR[N_PASSES] = { OFDM_SC_THRESHOLD, 0.40, 0.60 };
static const double PASS_NCC[N_PASSES] = { OFDM_SC_NCC_MIN,   0.20, 0.35 };

/* Is chunk 0 present and every chunk the header promises, with a good CRC? */
static int dx_complete(const struct ofdm_dx *dx)
{
    const ofdm_cfg *c = &dx->rx;
    int i, len, nch;
    unsigned char tmp[3 + OFDM_DX_MAX_DATA];
    unsigned crc;
    if (!dx->got[0]) return 0;
    len = dx->frame[2];
    nch = ceil_div(OFDM_DX_HDR + len, c->chunk_bytes);
    if (nch > dx->rx_max_chunks) return 0;
    for (i = 0; i < nch; i++)
        if (!dx->got[i]) return 0;
    for (i = 0; i < 3; i++) tmp[i] = dx->frame[i];
    for (i = 0; i < len; i++) tmp[3 + i] = dx->frame[OFDM_DX_HDR + i];
    crc = ofdm_dx_crc16(tmp, (size_t)(3 + len));
    return crc == (((unsigned)dx->frame[3] << 8) | dx->frame[4]);
}

int ofdm_dx_rx_frame(ofdm_dx *dx, const double *rx, int n,
                     unsigned char *data, ofdm_dx_rxinfo *info)
{
    const ofdm_cfg *c;
    int ns, s, i, rc, errata, seq, len, pass, n_dec = 0;
    double evm_sum = 0.0;
    ofdm_dem_out di;
    ofdm_dx_rxinfo local;

    if (!dx || !rx || !data || n < 0) return -1;
    if (!info) info = &local;
    info->seq = info->ack = info->len = 0;
    info->n_chunks = info->n_found = info->n_bad = info->errata = 0;
    info->evm = 0.0;
    info->first_start = -1;
    c = &dx->rx;

    for (i = 0; i < dx->rx_max_chunks; i++) dx->got[i] = 0;

    for (pass = 0; pass < N_PASSES; pass++) {
        ns = ofdm_sync_find(dx->sync, rx, n, dx->r_body, PASS_THR[pass],
                            PASS_NCC[pass], dx->starts, OFDM_DX_MAX_STARTS);
        if (ns < 0) return -1;
        for (s = 0; s < ns; s++) {
            if (info->first_start < 0) info->first_start = dx->starts[s];
            info->n_found++;
            rc = ofdm_dem_frame(dx->dem, rx, n, dx->starts[s], dx->wire, dx->r_wire, &di);
            if (rc < 0) { info->n_bad++; continue; }
            rc = ofdm_frame_decode_il(dx->rs, dx->wire, dx->r_wire, dx->r_depth, dx->r_k,
                                      c->ecc_bytes, c->seq_bytes + c->chunk_bytes,
                                      dx->msg, c->seq_bytes + c->chunk_bytes, dx->blk,
                                      &errata);
            if (rc < c->seq_bytes) { info->n_bad++; continue; }
            seq = 0;
            for (i = 0; i < c->seq_bytes; i++) seq = (seq << 8) | dx->msg[i];
            if (seq < 0 || seq >= dx->rx_max_chunks) { info->n_bad++; continue; }
            if (!dx->got[seq]) {
                info->errata += errata;
                evm_sum += di.evm;
                n_dec++;
                dx->got[seq] = 1;
                for (i = 0; i < c->chunk_bytes; i++)
                    dx->frame[seq * c->chunk_bytes + i] = dx->msg[c->seq_bytes + i];
            }
        }
        if (dx_complete(dx)) break;
    }
    if (n_dec) info->evm = evm_sum / n_dec;
    if (!dx_complete(dx)) return 0;

    len = dx->frame[2];
    for (i = 0; i < len; i++) data[i] = dx->frame[OFDM_DX_HDR + i];
    info->seq = dx->frame[0];
    info->ack = dx->frame[1];
    info->len = len;
    info->n_chunks = ceil_div(OFDM_DX_HDR + len, c->chunk_bytes);
    return 1;
}
