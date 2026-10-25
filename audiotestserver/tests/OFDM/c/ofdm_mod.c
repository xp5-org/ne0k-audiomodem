/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

#include "ofdm_mod.h"
#include "ofdm_sym.h"
#include "rs255.h"
#include "ofdm_frame.h"
#include <math.h>

/* Pilot symbol (1+1j)/sqrt(2): unit power, like every data constellation and
 * -- since REF_SYM_BOOST -- the CE/SYNC symbols too. */
#define PILOT_RE 0.70710678118654752
#define PILOT_IM 0.70710678118654752

struct ofdm_mod {
    ofdm_cfg c;
    int      bits_per_sym;
    int      data_syms;
    int      n_syncs;
    int      body_syms;
    int      il_depth, il_k, payload_bytes;
    int      ola_len;
    int      frame_len;
    int      n_maps;
    ofdm_sym *sym;
    rs255    *rs;
    int     *bpb;
    unsigned char *pilot_mask;
    double  *ce_re;
    double  *sc_td;
    double  *occ_re, *occ_im;
    double  *symbuf;
    unsigned char *msg;
    unsigned char *enc;
    unsigned char *blk;
    unsigned char *padbuf;   /* chunk_bytes, short final chunk */
    unsigned char *bits;
    /* value -> point index per width.  The tables are point->bits; the encoder
     * needs the inverse, and at 64 entries max it is a direct-mapped array
     * rather than a search per carrier per symbol. */
    int      map[4][64];
    int      map_b[4];
};

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

static int width_ok(int b)
{
    return b == 0 || b == 1 || b == 2 || b == 4 || b == 6;
}

static int bits_sum(const int *bpb, int n)
{
    int i, s = 0;
    for (i = 0; i < n; i++) s += bpb[i];
    return s;
}

/* Geometry every entry point re-derives.  Returns 0 ok. */
static int mod_plan(const ofdm_cfg *c, const int *bpb,
                    int *bps, int *depth, int *k, int *wire,
                    int *ds, int *body)
{
    int i, ns;
    if (!c || !bpb) return -1;
    for (i = 0; i < c->n_data_carriers; i++)
        if (!width_ok(bpb[i])) return -1;
    *bps = bits_sum(bpb, c->n_data_carriers);
    if (*bps < 1) return -1;
    if (ofdm_interleave_geometry(c, *bps, depth, k, wire) != 0) return -1;
    *ds = ofdm_data_syms_per_pkt(c, *bps);
    if (*ds < 1) return -1;
    ns = ofdm_n_syncs(c, *ds);
    *body = 1 + *ds + ns;
    return 0;
}

/* Must stay the term-for-term sum of the bumps ofdm_mod_init performs, in the
 * same order.  Edit the two together. */
size_t ofdm_mod_workspace(const ofdm_cfg *c, const int *bits_per_bin)
{
    size_t fftw, rsw;
    int    bps, depth, k, wire, ds, body, T;

    if (mod_plan(c, bits_per_bin, &bps, &depth, &k, &wire, &ds, &body) != 0)
        return 0;
    fftw = ofdm_sym_workspace(c);
    if (fftw == 0) return 0;
    rsw = rs255_workspace(c->ecc_bytes);
    if (rsw == 0) return 0;
    T    = c->taper_samples;

    return al8(sizeof(struct ofdm_mod))
         + fftw
         + rsw
         + al8((size_t)c->n_data_carriers * sizeof(int))
         + al8((size_t)c->n_occupied * sizeof(unsigned char))
         + al8((size_t)c->n_occupied * sizeof(double))              /* ce_re  */
         + al8((size_t)c->symbol_samples * sizeof(double))          /* sc_td  */
         + al8((size_t)c->n_occupied * sizeof(double))              /* occ_re */
         + al8((size_t)c->n_occupied * sizeof(double))              /* occ_im */
         + al8((size_t)(c->symbol_samples + 2 * T) * sizeof(double))/* symbuf */
         + al8((size_t)(c->seq_bytes + c->chunk_bytes))             /* msg    */
         + al8((size_t)wire)                                        /* enc    */
         + al8((size_t)(k + c->ecc_bytes))                          /* blk    */
         + al8((size_t)c->chunk_bytes)                              /* padbuf */
         + al8((size_t)ds * bps);                                   /* bits   */
}

int ofdm_mod_init(ofdm_mod **mp, void *mem, size_t memlen,
                  const ofdm_cfg *c, const int *bits_per_bin)
{
    struct ofdm_mod *m;
    unsigned char *p = (unsigned char *)mem;
    size_t need, fftw, rsw;
    int    i, t, rc, bps, depth, k, wire, ds, body, T, n_even, nw;
    int    widths[4];
    double amp;

    if (!mp || !mem || !c || !bits_per_bin) return -1;
    for (i = 0; i < c->n_data_carriers; i++)
        if (!width_ok(bits_per_bin[i])) return -4;
    if (bits_sum(bits_per_bin, c->n_data_carriers) < 1) return -5;
    if (mod_plan(c, bits_per_bin, &bps, &depth, &k, &wire, &ds, &body) != 0)
        return -1;

    need = ofdm_mod_workspace(c, bits_per_bin);
    if (need == 0) return -1;
    if (memlen < need) return -2;

    fftw = ofdm_sym_workspace(c);
    rsw  = rs255_workspace(c->ecc_bytes);
    T    = c->taper_samples;

    m = (struct ofdm_mod *)p; p += al8(sizeof(struct ofdm_mod));
    m->c             = *c;
    m->bits_per_sym  = bps;
    m->il_depth      = depth;
    m->il_k          = k;
    m->payload_bytes = wire;
    m->data_syms     = ds;
    m->n_syncs       = ofdm_n_syncs(c, ds);
    m->body_syms     = body;
    m->ola_len       = body * c->symbol_samples + (T > 0 ? 2 * T : 0);
    m->frame_len     = c->symbol_samples + m->ola_len;

    rc = ofdm_sym_init(&m->sym, p, fftw, c);
    if (rc != 0) return -3;
    p += fftw;
    rc = rs255_init(&m->rs, p, rsw, c->ecc_bytes);
    if (rc != 0) return -6;
    p += rsw;

    m->bpb        = (int *)p;    p += al8((size_t)c->n_data_carriers * sizeof(int));
    m->pilot_mask = p;           p += al8((size_t)c->n_occupied);
    m->ce_re      = (double *)p; p += al8((size_t)c->n_occupied * sizeof(double));
    m->sc_td      = (double *)p; p += al8((size_t)c->symbol_samples * sizeof(double));
    m->occ_re     = (double *)p; p += al8((size_t)c->n_occupied * sizeof(double));
    m->occ_im     = (double *)p; p += al8((size_t)c->n_occupied * sizeof(double));
    m->symbuf     = (double *)p; p += al8((size_t)(c->symbol_samples + 2 * T)
                                          * sizeof(double));
    m->msg        = p;           p += al8((size_t)(c->seq_bytes + c->chunk_bytes));
    m->enc        = p;           p += al8((size_t)wire);
    m->blk        = p;           p += al8((size_t)(k + c->ecc_bytes));
    m->padbuf     = p;           p += al8((size_t)c->chunk_bytes);
    m->bits       = p;

    for (i = 0; i < c->n_data_carriers; i++) m->bpb[i] = bits_per_bin[i];
    for (i = 0; i < c->n_occupied; i++) m->pilot_mask[i] = 0;
    for (i = 0; i < c->n_occupied; i += c->pilot_spacing) m->pilot_mask[i] = 1;

    /* CE and SYNC carry the SAME sequence at the SAME level; the demodulator
     * depends on that to average the two adjacent estimates. */
    amp = OFDM_REF_SYM_BOOST / sqrt(2.0);
    ofdm_known_bpsk(m->ce_re, c->n_occupied, OFDM_SEED_CE, amp);

    nw = ofdm_bat_widths(m->bpb, c->n_data_carriers, widths, 4);
    if (nw < 0) return -1;
    m->n_maps = 0;
    for (t = 0; t < nw; t++) {
        const unsigned char *bt;
        int n, j, b, v;
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

    /* SC preamble: PN-BPSK on EVEN-indexed occupied bins only.  The half-empty
     * comb is what makes the two halves of the time-domain symbol identical,
     * and that repetition is what the delay-correlator locks onto. */
    n_even = (c->n_occupied + 1) / 2;
    {
        int j = 0;
        /* Generate the PN straight into occ_re's front, then spread it: the
         * sequence is n_even long and occ_re is n_occupied >= n_even, and the
         * spread walks DOWNWARD so a value is never overwritten before use. */
        ofdm_known_bpsk(m->occ_re, n_even, OFDM_SEED_SC, 1.0 / sqrt(2.0));
        for (i = c->n_occupied - 1; i >= 0; i--) m->occ_im[i] = 0.0;
        for (j = n_even - 1; j >= 0; j--) {
            double v = m->occ_re[j];
            m->occ_re[j] = 0.0;
            if (2 * j < c->n_occupied) m->occ_re[2 * j] = v;
        }
        for (i = 1; i < c->n_occupied; i += 2) m->occ_re[i] = 0.0;
        ofdm_sym_make(m->sym, m->occ_re, m->occ_im, 0, m->symbuf);
        for (i = 0; i < c->symbol_samples; i++) m->sc_td[i] = m->symbuf[i];
    }

    *mp = m;
    return 0;
}

int  ofdm_mod_bits_per_sym(const ofdm_mod *m)  { return m ? m->bits_per_sym : 0; }
int  ofdm_mod_data_syms(const ofdm_mod *m)     { return m ? m->data_syms : 0; }
int  ofdm_mod_body_syms(const ofdm_mod *m)     { return m ? m->body_syms : 0; }
int  ofdm_mod_payload_bytes(const ofdm_mod *m) { return m ? m->payload_bytes : 0; }
long ofdm_mod_frame_samples(const ofdm_mod *m) { return m ? m->frame_len : 0; }
const double *ofdm_mod_sc_preamble(const ofdm_mod *m) { return m ? m->sc_td : 0; }

long ofdm_mod_signal_samples(const ofdm_mod *m, int n_bytes)
{
    int n_chunks;
    if (!m || n_bytes < 0) return -1;
    n_chunks = (n_bytes + m->c.chunk_bytes - 1) / m->c.chunk_bytes;
    if (n_chunks < 1) n_chunks = 1;
    return (long)n_chunks * m->frame_len
         + (long)(n_chunks + 1) * m->c.frame_gap_samples;
}

long ofdm_mod_frame(ofdm_mod *m, unsigned seq, const unsigned char *chunk,
                    double *out, long out_cap)
{
    const ofdm_cfg *c;
    int i, j, rc, msg_len, nbits, bit_ptr, ds, sym_i, data_i;
    unsigned state;
    long base;

    if (!m || !chunk || !out) return -1;
    c = &m->c;
    if (out_cap < m->frame_len) return -2;

    /* ---- message: big-endian sequence number, then the chunk ---- */
    msg_len = c->seq_bytes + c->chunk_bytes;
    for (i = 0; i < c->seq_bytes; i++)
        m->msg[i] = (unsigned char)((seq >> (8 * (c->seq_bytes - 1 - i))) & 0xFFu);
    for (i = 0; i < c->chunk_bytes; i++) m->msg[c->seq_bytes + i] = chunk[i];

    rc = ofdm_frame_encode_il(m->rs, m->msg, msg_len, m->il_depth,
                              m->il_k, c->ecc_bytes, m->enc,
                              m->payload_bytes, m->blk);
    if (rc < 0) return rc;

    /* ---- bits: MSB-first per byte, zero-padded to a whole symbol ---- */
    nbits = m->data_syms * m->bits_per_sym;
    for (i = 0; i < m->payload_bytes && i * 8 < nbits; i++)
        for (j = 0; j < 8; j++)
            if (i * 8 + j < nbits)
                m->bits[i * 8 + j] =
                    (unsigned char)((m->enc[i] >> (7 - j)) & 1u);
    for (i = m->payload_bytes * 8; i < nbits; i++) m->bits[i] = 0;

    /* ---- scramble ---- */
    state = OFDM_SEED_DATA & 0xFFFFu;
    for (i = 0; i < nbits; i++) {
        unsigned fb = ((state >> 15) ^ (state >> 14)
                     ^ (state >> 12) ^ (state >> 3)) & 1u;
        m->bits[i] ^= (unsigned char)((state >> 15) & 1u);
        state = ((state << 1) | fb) & 0xFFFFu;
    }

    /* ---- SC preamble, then the overlap-added body ---- */
    for (i = 0; i < c->symbol_samples; i++) out[i] = m->sc_td[i];
    for (i = 0; i < m->ola_len; i++) out[c->symbol_samples + i] = 0.0;
    base = c->symbol_samples;

    bit_ptr = 0;
    sym_i   = 0;

    /* CE first. */
    for (i = 0; i < c->n_occupied; i++) {
        m->occ_re[i] = m->ce_re[i];
        m->occ_im[i] = 0.0;
    }
    {
        int len = ofdm_sym_make(m->sym, m->occ_re, m->occ_im, 1, m->symbuf);
        for (i = 0; i < len; i++)
            out[base + (long)sym_i * c->symbol_samples + i] += m->symbuf[i];
        sym_i++;
    }

    for (ds = 0; ds < m->data_syms; ds++) {
        /* A SYNC is inserted BEFORE every period-th data symbol. */
        if (c->sync_period > 0 && ds % c->sync_period == 0) {
            for (i = 0; i < c->n_occupied; i++) {
                m->occ_re[i] = m->ce_re[i];      /* same sequence as CE */
                m->occ_im[i] = 0.0;
            }
            {
                int len = ofdm_sym_make(m->sym, m->occ_re, m->occ_im, 1, m->symbuf);
                for (i = 0; i < len; i++)
                    out[base + (long)sym_i * c->symbol_samples + i] += m->symbuf[i];
                sym_i++;
            }
        }
        /* Data symbol: pilots on their offsets, bit-loaded carriers elsewhere. */
        data_i = 0;
        for (i = 0; i < c->n_occupied; i++) {
            if (m->pilot_mask[i]) {
                m->occ_re[i] = PILOT_RE;
                m->occ_im[i] = PILOT_IM;
                continue;
            }
            {
                int b = m->bpb[data_i], t, mi = -1, v = 0, idx;
                const double *pr, *pi;
                data_i++;
                if (b == 0) { m->occ_re[i] = 0.0; m->occ_im[i] = 0.0; continue; }
                for (t = 0; t < m->n_maps; t++)
                    if (m->map_b[t] == b) { mi = t; break; }
                if (mi < 0) return -1;          /* rejected at init */
                for (t = 0; t < b; t++)
                    v = (v << 1) | m->bits[bit_ptr + t];
                bit_ptr += b;
                idx = m->map[mi][v];
                if (idx < 0) return -1;
                if (ofdm_mod_table(b, &pr, &pi, 0, 0) != 0) return -1;
                m->occ_re[i] = pr[idx];
                m->occ_im[i] = pi[idx];
            }
        }
        {
            int len = ofdm_sym_make(m->sym, m->occ_re, m->occ_im, 1, m->symbuf);
            for (i = 0; i < len; i++)
                out[base + (long)sym_i * c->symbol_samples + i] += m->symbuf[i];
            sym_i++;
        }
    }
    return m->frame_len;
}

long ofdm_mod_encode(ofdm_mod *m, const unsigned char *data, int n_bytes,
                     double *out, long out_cap)
{
    const ofdm_cfg *c;
    long need, pos;
    int  n_chunks, s, i;

    if (!m || (!data && n_bytes > 0) || !out || n_bytes < 0) return -1;
    c = &m->c;
    need = ofdm_mod_signal_samples(m, n_bytes);
    if (need < 0) return -1;
    if (out_cap < need) return -2;

    n_chunks = (n_bytes + c->chunk_bytes - 1) / c->chunk_bytes;
    if (n_chunks < 1) n_chunks = 1;

    pos = 0;
    for (i = 0; i < c->frame_gap_samples; i++) out[pos++] = 0.0;
    for (s = 0; s < n_chunks; s++) {
        const unsigned char *src = data + (long)s * c->chunk_bytes;
        int  avail = n_bytes - s * c->chunk_bytes;
        long rc;

        if (avail < c->chunk_bytes) {
            /* NUL-pad the final chunk, exactly as the reference does -- which
             * is why a payload whose own tail is NUL cannot be told from
             * padding on the wire, and why a caller that cares must carry an
             * explicit length. */
            for (i = 0; i < c->chunk_bytes; i++)
                m->padbuf[i] = (i < avail) ? src[i] : 0;
            src = m->padbuf;
        }
        rc = ofdm_mod_frame(m, (unsigned)s, src, out + pos, out_cap - pos);
        if (rc < 0) return rc;
        pos += rc;
        for (i = 0; i < c->frame_gap_samples; i++) out[pos++] = 0.0;
    }
    return pos;
}
