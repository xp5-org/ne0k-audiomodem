/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

#include "ofdm_cxmod.h"
#include "rs255.h"
#include "ofdm_frame.h"
#include "ofdm_cxfec.h"
#include <math.h>

/* Same values as ofdm_mod.c: the pilot (1+1j)/sqrt(2) and the data-frame SC
 * gain (ofdm_modem.SC_TX_GAIN). */
#define CX_PILOT_RE 0.70710678118654752
#define CX_PILOT_IM 0.70710678118654752
#define CX_SC_TX_GAIN 2.0

struct ofdm_cxmod {
    ofdm_cxcfg  x;
    int         bits_per_sym, data_syms, n_syncs, body_syms;
    int         il_depth, il_k, payload_bytes;
    int         ola_len, frame_len, n_maps;
    ofdm_cxsym *sym;
    rs255      *rs;
    int        *bpb;
    unsigned char *pilot_mask;
    double     *ce_re;
    double     *sc_iq;              /* 2 * symbol_samples           */
    double     *occ_re, *occ_im;
    double     *symbuf;             /* 2 * (symbol_samples + 2T)    */
    unsigned char *msg, *enc, *blk, *padbuf, *bits;
    int         map[5][64];         /* value -> point index per width */
    int         map_b[5];
    int         fec;                /* OFDM_CXFEC_*                   */
    long        ncod, pstep;        /* coded bits, interleaver step   */
    unsigned char *info, *cod;      /* wire bits (scrambled), coded   */
};

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

static int width_ok(int b)
{
    return b == 0 || b == 1 || b == 2 || b == 3 || b == 4 || b == 6;
}

static int bits_sum(const int *bpb, int n)
{
    int i, s = 0;
    for (i = 0; i < n; i++) s += bpb[i];
    return s;
}

/* The frame arithmetic only reads counts, so the real-path helpers serve. */
static int cxmod_plan(const ofdm_cfg *c, const int *bpb, int fec, int *bps, int *depth,
                      int *k, int *wire, int *ds, int *body)
{
    int i;
    if (!c || !bpb) return -1;
    for (i = 0; i < c->n_data_carriers; i++)
        if (!width_ok(bpb[i])) return -1;
    *bps = bits_sum(bpb, c->n_data_carriers);
    if (*bps < 1) return -1;
    if (ofdm_interleave_geometry(c, *bps, depth, k, wire) != 0) return -1;
    if (!ofdm_cxfec_valid(fec)) return -1;
    if (fec != OFDM_CXFEC_NONE) {
        long nc = ofdm_cxfec_coded_bits(fec, (long)*wire * 8);
        if (nc < 1) return -1;
        *ds = (int)((nc + *bps - 1) / *bps);
    } else
        *ds = ofdm_data_syms_per_pkt(c, *bps);
    if (*ds < 1) return -1;
    *body = 1 + *ds + ofdm_n_syncs(c, *ds);
    return 0;
}

/* Term-for-term the carving in ofdm_cxmod_init.  Edit the two together. */
size_t ofdm_cxmod_workspace(const ofdm_cxcfg *x, const int *bits_per_bin)
{
    return ofdm_cxmod_workspace_fec(x, bits_per_bin, OFDM_CXFEC_NONE);
}

size_t ofdm_cxmod_workspace_fec(const ofdm_cxcfg *x, const int *bits_per_bin, int fec)
{
    const ofdm_cfg *c;
    size_t symw, rsw;
    int    bps, depth, k, wire, ds, body, T;

    if (!x) return 0;
    c = &x->c;
    if (cxmod_plan(c, bits_per_bin, fec, &bps, &depth, &k, &wire, &ds, &body) != 0)
        return 0;
    symw = ofdm_cxsym_workspace(x);
    if (symw == 0) return 0;
    rsw = rs255_workspace(c->ecc_bytes);
    if (rsw == 0) return 0;
    T = c->taper_samples;

    return al8(sizeof(struct ofdm_cxmod))
         + al8(symw)
         + al8(rsw)
         + al8((size_t)c->n_data_carriers * sizeof(int))                  /* bpb    */
         + al8((size_t)c->n_occupied)                                     /* pilots */
         + al8((size_t)c->n_occupied * sizeof(double))                    /* ce_re  */
         + al8((size_t)2 * c->symbol_samples * sizeof(double))            /* sc_iq  */
         + al8((size_t)c->n_occupied * sizeof(double))                    /* occ_re */
         + al8((size_t)c->n_occupied * sizeof(double))                    /* occ_im */
         + al8((size_t)2 * (c->symbol_samples + 2 * T) * sizeof(double))  /* symbuf */
         + al8((size_t)(c->seq_bytes + c->chunk_bytes))                   /* msg    */
         + al8((size_t)wire)                                              /* enc    */
         + al8((size_t)(k + c->ecc_bytes))                                /* blk    */
         + al8((size_t)c->chunk_bytes)                                    /* padbuf */
         + al8((size_t)ds * bps)                                          /* bits   */
         + (fec ? al8((size_t)wire * 8) + al8((size_t)ds * bps) : 0);     /* info, cod */
}

int ofdm_cxmod_init(ofdm_cxmod **mp, void *mem, size_t memlen,
                    const ofdm_cxcfg *x, const int *bits_per_bin)
{
    return ofdm_cxmod_init_fec(mp, mem, memlen, x, bits_per_bin, OFDM_CXFEC_NONE);
}

int ofdm_cxmod_init_fec(ofdm_cxmod **mp, void *mem, size_t memlen,
                        const ofdm_cxcfg *x, const int *bits_per_bin, int fec)
{
    struct ofdm_cxmod *m;
    const ofdm_cfg *c;
    unsigned char *p = (unsigned char *)mem;
    size_t need, symw, rsw;
    int    i, j, t, bps, depth, k, wire, ds, body, T, n_sc, nw, len;
    int    widths[5];

    if (!mp || !mem || !x || !bits_per_bin) return -1;
    c = &x->c;
    for (i = 0; i < c->n_data_carriers; i++)
        if (!width_ok(bits_per_bin[i])) return -4;
    if (bits_sum(bits_per_bin, c->n_data_carriers) < 1) return -5;
    if (cxmod_plan(c, bits_per_bin, fec, &bps, &depth, &k, &wire, &ds, &body) != 0)
        return -1;
    need = ofdm_cxmod_workspace_fec(x, bits_per_bin, fec);
    if (need == 0) return -1;
    if (memlen < need) return -2;
    symw = ofdm_cxsym_workspace(x);
    rsw  = rs255_workspace(c->ecc_bytes);
    T    = c->taper_samples;

    m = (struct ofdm_cxmod *)p; p += al8(sizeof(struct ofdm_cxmod));
    m->x             = *x;
    m->bits_per_sym  = bps;
    m->il_depth      = depth;
    m->il_k          = k;
    m->payload_bytes = wire;
    m->data_syms     = ds;
    m->n_syncs       = ofdm_n_syncs(c, ds);
    m->body_syms     = body;
    m->ola_len       = body * c->symbol_samples + (T > 0 ? 2 * T : 0);
    m->frame_len     = c->symbol_samples + m->ola_len;

    if (ofdm_cxsym_init(&m->sym, p, symw, x) != 0) return -3;
    p += al8(symw);
    if (rs255_init(&m->rs, p, rsw, c->ecc_bytes) != 0) return -6;
    p += al8(rsw);

    m->bpb        = (int *)p;    p += al8((size_t)c->n_data_carriers * sizeof(int));
    m->pilot_mask = p;           p += al8((size_t)c->n_occupied);
    m->ce_re      = (double *)p; p += al8((size_t)c->n_occupied * sizeof(double));
    m->sc_iq      = (double *)p; p += al8((size_t)2 * c->symbol_samples * sizeof(double));
    m->occ_re     = (double *)p; p += al8((size_t)c->n_occupied * sizeof(double));
    m->occ_im     = (double *)p; p += al8((size_t)c->n_occupied * sizeof(double));
    m->symbuf     = (double *)p; p += al8((size_t)2 * (c->symbol_samples + 2 * T)
                                          * sizeof(double));
    m->msg        = p;           p += al8((size_t)(c->seq_bytes + c->chunk_bytes));
    m->enc        = p;           p += al8((size_t)wire);
    m->blk        = p;           p += al8((size_t)(k + c->ecc_bytes));
    m->padbuf     = p;           p += al8((size_t)c->chunk_bytes);
    m->bits       = p;           p += al8((size_t)ds * bps);
    m->fec = fec;
    m->ncod = 0; m->pstep = 1; m->info = m->cod = 0;
    if (fec) {
        m->info  = p;            p += al8((size_t)wire * 8);
        m->cod   = p;
        m->ncod  = ofdm_cxfec_coded_bits(fec, (long)wire * 8);
        m->pstep = ofdm_cxfec_perm_step(m->ncod, bps);
    }

    for (i = 0; i < c->n_data_carriers; i++) m->bpb[i] = bits_per_bin[i];
    for (i = 0; i < c->n_occupied; i++) m->pilot_mask[i] = 0;
    for (i = 0; i < c->n_occupied; i += c->pilot_spacing) m->pilot_mask[i] = 1;

    /* CE and SYNC: the same known BPSK at the same level as the real path. */
    ofdm_known_bpsk(m->ce_re, c->n_occupied, OFDM_SEED_CE, OFDM_REF_SYM_BOOST / sqrt(2.0));

    nw = ofdm_bat_widths(m->bpb, c->n_data_carriers, widths, 5);
    if (nw < 0) return -1;
    m->n_maps = 0;
    for (t = 0; t < nw; t++) {
        const unsigned char *bt;
        int n, b, v;
        if (widths[t] == 0) continue;
        if (ofdm_mod_table(widths[t], 0, 0, &bt, &n) != 0) return -4;
        b = widths[t];
        for (j = 0; j < 64; j++) m->map[m->n_maps][j] = -1;
        for (j = 0; j < n; j++) {
            v = 0;
            for (i = 0; i < b; i++) v = (v << 1) | bt[j * b + i];
            m->map[m->n_maps][v] = j;
        }
        m->map_b[m->n_maps] = b;
        m->n_maps++;
    }

    /* SC preamble: PN on the EVEN signed bins (see the header for why bins,
     * not indices).  The PN is generated into occ_im, then spread into occ_re
     * in ascending bin order (occupied order is ascending bin order). */
    n_sc = 0;
    for (i = 0; i < c->n_occupied; i++)
        if (ofdm_cx_bin(&m->x, i) % 2 == 0) n_sc++;
    if (n_sc < 1) return -1;
    ofdm_known_bpsk(m->occ_im, n_sc, OFDM_SEED_SC, 1.0 / sqrt(2.0));
    for (i = 0, j = 0; i < c->n_occupied; i++)
        m->occ_re[i] = (ofdm_cx_bin(&m->x, i) % 2 == 0) ? m->occ_im[j++] : 0.0;
    for (i = 0; i < c->n_occupied; i++) m->occ_im[i] = 0.0;
    len = ofdm_cxsym_make(m->sym, m->occ_re, m->occ_im, 0, m->symbuf);
    if (len != c->symbol_samples) return -1;
    for (i = 0; i < 2 * c->symbol_samples; i++) m->sc_iq[i] = m->symbuf[i];

    *mp = m;
    return 0;
}

int  ofdm_cxmod_bits_per_sym(const ofdm_cxmod *m)  { return m ? m->bits_per_sym : 0; }
int  ofdm_cxmod_data_syms(const ofdm_cxmod *m)     { return m ? m->data_syms : 0; }
int  ofdm_cxmod_body_syms(const ofdm_cxmod *m)     { return m ? m->body_syms : 0; }
int  ofdm_cxmod_payload_bytes(const ofdm_cxmod *m) { return m ? m->payload_bytes : 0; }
long ofdm_cxmod_frame_samples(const ofdm_cxmod *m) { return m ? m->frame_len : 0; }
const double *ofdm_cxmod_sc_preamble(const ofdm_cxmod *m) { return m ? m->sc_iq : 0; }

long ofdm_cxmod_signal_samples_ex(const ofdm_cxmod *m, int n_bytes, int gap)
{
    int n_chunks;
    if (!m || n_bytes < 0) return -1;
    n_chunks = (n_bytes + m->x.c.chunk_bytes - 1) / m->x.c.chunk_bytes;
    if (n_chunks < 1) n_chunks = 1;
    return (long)n_chunks * m->frame_len
         + (gap ? (long)(n_chunks + 1) * m->x.c.frame_gap_samples : 0L);
}

/* Build the symbol now in occ_re/occ_im and overlap-add it as body symbol
 * sym_i (out_iq already points at the start of the body). */
static void ola_sym(struct ofdm_cxmod *m, double *body_iq, int sym_i)
{
    long off = 2L * sym_i * m->x.c.symbol_samples;
    int  i, len = ofdm_cxsym_make(m->sym, m->occ_re, m->occ_im, 1, m->symbuf);
    for (i = 0; i < 2 * len; i++) body_iq[off + i] += m->symbuf[i];
}

static void load_ref(struct ofdm_cxmod *m)
{
    int i;
    for (i = 0; i < m->x.c.n_occupied; i++) {
        m->occ_re[i] = m->ce_re[i];
        m->occ_im[i] = 0.0;
    }
}

long ofdm_cxmod_frame(ofdm_cxmod *m, unsigned seq, const unsigned char *chunk,
                      double *out_iq, long out_cap)
{
    const ofdm_cfg *c;
    double *body;
    const double *pr, *pi;
    int     i, j, rc, msg_len, nbits, bit_ptr, ds, sym_i, data_i;
    unsigned state;

    if (!m || !chunk || !out_iq) return -1;
    c = &m->x.c;
    if (out_cap < m->frame_len) return -2;

    /* ---- message: big-endian sequence number, then the chunk ---- */
    msg_len = c->seq_bytes + c->chunk_bytes;
    for (i = 0; i < c->seq_bytes; i++)
        m->msg[i] = (unsigned char)((seq >> (8 * (c->seq_bytes - 1 - i))) & 0xFFu);
    for (i = 0; i < c->chunk_bytes; i++) m->msg[c->seq_bytes + i] = chunk[i];
    rc = ofdm_frame_encode_il(m->rs, m->msg, msg_len, m->il_depth, m->il_k,
                              c->ecc_bytes, m->enc, m->payload_bytes, m->blk);
    if (rc < 0) return rc;

    nbits = m->data_syms * m->bits_per_sym;
    if (m->fec) {
        /* ---- wire bits scrambled, conv encoded, interleaved into the frame;
         *      the positions past the code carry the scrambler's continuation ---- */
        long nw = (long)m->payload_bytes * 8, ii, pos = 0;
        state = OFDM_SEED_DATA & 0xFFFFu;
        for (ii = 0; ii < nw; ii++) {
            unsigned fb = ((state >> 15) ^ (state >> 14) ^ (state >> 12) ^ (state >> 3)) & 1u;
            m->info[ii] = (unsigned char)(((m->enc[ii / 8] >> (7 - ii % 8)) & 1u) ^ ((state >> 15) & 1u));
            state = ((state << 1) | fb) & 0xFFFFu;
        }
        if (ofdm_cxfec_encode(m->fec, m->info, nw, m->cod) != m->ncod) return -1;
        for (ii = 0; ii < m->ncod; ii++) {
            m->bits[pos] = m->cod[ii];
            pos += m->pstep;
            if (pos >= m->ncod) pos -= m->ncod;
        }
        for (ii = m->ncod; ii < nbits; ii++) {
            unsigned fb = ((state >> 15) ^ (state >> 14) ^ (state >> 12) ^ (state >> 3)) & 1u;
            m->bits[ii] = (unsigned char)((state >> 15) & 1u);
            state = ((state << 1) | fb) & 0xFFFFu;
        }
        goto mapped_bits;
    }
    /* ---- bits: MSB-first, zero-padded to whole symbols, then scrambled ---- */
    for (i = 0; i < nbits; i++)
        m->bits[i] = (i / 8 < m->payload_bytes)
                   ? (unsigned char)((m->enc[i / 8] >> (7 - i % 8)) & 1u) : 0;
    state = OFDM_SEED_DATA & 0xFFFFu;
    for (i = 0; i < nbits; i++) {
        unsigned fb = ((state >> 15) ^ (state >> 14)
                     ^ (state >> 12) ^ (state >> 3)) & 1u;
        m->bits[i] ^= (unsigned char)((state >> 15) & 1u);
        state = ((state << 1) | fb) & 0xFFFFu;
    }

mapped_bits:
    /* ---- SC preamble at the TX gain, then the overlap-added body ---- */
    for (i = 0; i < 2 * c->symbol_samples; i++) out_iq[i] = m->sc_iq[i] * CX_SC_TX_GAIN;
    body = out_iq + 2L * c->symbol_samples;
    for (i = 0; i < 2 * m->ola_len; i++) body[i] = 0.0;

    sym_i = 0;
    load_ref(m);
    ola_sym(m, body, sym_i++);                         /* CE */

    bit_ptr = 0;
    for (ds = 0; ds < m->data_syms; ds++) {
        if (c->sync_period > 0 && ds % c->sync_period == 0) {
            load_ref(m);
            ola_sym(m, body, sym_i++);                 /* SYNC before every period-th */
        }
        data_i = 0;
        for (i = 0; i < c->n_occupied; i++) {
            int b, t, mi, v, idx;
            if (m->pilot_mask[i]) {
                m->occ_re[i] = CX_PILOT_RE;
                m->occ_im[i] = CX_PILOT_IM;
                continue;
            }
            b = m->bpb[data_i++];
            if (b == 0) { m->occ_re[i] = 0.0; m->occ_im[i] = 0.0; continue; }
            mi = -1;
            for (t = 0; t < m->n_maps; t++)
                if (m->map_b[t] == b) { mi = t; break; }
            if (mi < 0) return -1;
            v = 0;
            for (j = 0; j < b; j++) v = (v << 1) | m->bits[bit_ptr + j];
            bit_ptr += b;
            idx = m->map[mi][v];
            if (idx < 0 || ofdm_mod_table(b, &pr, &pi, 0, 0) != 0) return -1;
            m->occ_re[i] = pr[idx];
            m->occ_im[i] = pi[idx];
        }
        ola_sym(m, body, sym_i++);
    }
    return m->frame_len;
}

long ofdm_cxmod_encode_ex(ofdm_cxmod *m, const unsigned char *data, int n_bytes,
                          unsigned long seq0, int gap,
                          double *out_iq, long out_cap)
{
    const ofdm_cfg *c;
    long need, pos, n_gap, i;
    int  n_chunks, s;
    unsigned long seq_mask;

    if (!m || (!data && n_bytes > 0) || !out_iq || n_bytes < 0) return -1;
    c = &m->x.c;
    need = ofdm_cxmod_signal_samples_ex(m, n_bytes, gap);
    if (need < 0) return -1;
    if (out_cap < need) return -2;
    n_gap = gap ? c->frame_gap_samples : 0;
    seq_mask = (c->seq_bytes >= (int)sizeof(unsigned long)) ? ~0UL
             : ((1UL << (8 * c->seq_bytes)) - 1UL);
    n_chunks = (n_bytes + c->chunk_bytes - 1) / c->chunk_bytes;
    if (n_chunks < 1) n_chunks = 1;

    pos = 0;
    for (i = 0; i < 2 * n_gap; i++) out_iq[2 * pos + i] = 0.0;
    pos += n_gap;
    for (s = 0; s < n_chunks; s++) {
        const unsigned char *src = data + (long)s * c->chunk_bytes;
        int  avail = n_bytes - s * c->chunk_bytes, k;
        long rc;
        if (avail < c->chunk_bytes) {                  /* NUL-pad, as the reference */
            for (k = 0; k < c->chunk_bytes; k++)
                m->padbuf[k] = (unsigned char)((k < avail) ? src[k] : 0);
            src = m->padbuf;
        }
        rc = ofdm_cxmod_frame(m, (unsigned)((seq0 + (unsigned long)s) & seq_mask),
                              src, out_iq + 2 * pos, out_cap - pos);
        if (rc < 0) return rc;
        pos += rc;
        for (i = 0; i < 2 * n_gap; i++) out_iq[2 * pos + i] = 0.0;
        pos += n_gap;
    }
    return pos;
}
