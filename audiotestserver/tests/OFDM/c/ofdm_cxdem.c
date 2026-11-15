/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

#include "ofdm_cxdem.h"
#include "ofdm_fft.h"
#include "ofdm_frame.h"
#include "rs255.h"
#include "ofdm_cxfec.h"
#include <math.h>

#define CX_PI 3.14159265358979323846
/* As ofdm_mod.c / ofdm_cxmod.c: the pilot (1+1j)/sqrt(2). */
#define CX_PILOT_RE 0.70710678118654752
#define CX_PILOT_IM 0.70710678118654752
/* As ofdm_dem.c: a dead carrier passes unequalised; +eps under the EVM ratio. */
#define CXDEM_H_EPS   1e-9
#define CXDEM_REF_EPS 1e-20
/* Erasure decoding (GMD).  Cell score r = squared decision distance over
 * (d_min/2)^2 of its constellation: 0 on the point, 1 on the boundary.  A
 * carrier whose mean |H|^2 is CXDEM_DEAD_DB below the loaded carriers' mean is
 * dead (a notch, a blocked slot) and every cell on it scores CXDEM_DEAD_SCORE.
 * A byte scores its worst bit's cell (or that cell's carrier mean, if worse).
 * Only bytes above CXDEM_ERA_FLOOR are erasure candidates; a GMD decode with E
 * erasures and e errors is accepted only when 2e + E <= ecc - CXDEM_ERA_SLACK,
 * so SLACK parity bytes stay unused as a check (a miscorrection then needs a
 * random word to land within the reduced radius: ~255^-SLACK). */
#define CXDEM_DEAD_RATIO 0.01
#define CXDEM_DEAD_SCORE 4.0f
#define CXDEM_ERA_FLOOR  0.25f
#define CXDEM_ERA_SLACK  4
/* Tracking (ofdm_cxdem_set_tracking, all off by default so the oracles hold):
 * pilots whose weighted phase coherence is under CXDEM_PILOT_COH are not
 * trusted and the previous symbol's a, b are used instead; the decision-
 * directed pass refines a, b from cells under CXDEM_DD_R; the channel update
 * moves H toward Y / decision by CXDEM_TRACK_MU on cells under CXDEM_TRACK_R. */
#define CXDEM_PILOT_COH  0.6
#define CXDEM_DD_R       0.5
#define CXDEM_TRACK_R    0.3
#define CXDEM_TRACK_MU   0.15

struct ofdm_cxdem {
    ofdm_cxcfg x;
    int        N, CP, SS, O, nd, np, w0;
    int        bps, ds, n_syncs, body, depth, k, wire_len, msg_len;
    ofdm_cfft  fft;
    rs255     *rs;
    ofdm_cpx  *X;        /* N: one window's spectrum                        */
    ofdm_cpx  *H;        /* O: live channel estimate                        */
    double    *ref;      /* O: known CE/SYNC values (real BPSK)             */
    int       *fidx;     /* O: natural-order FFT index of occupied i        */
    int       *sbin;     /* O: signed bin of occupied i                     */
    unsigned char *pmask;/* O                                               */
    int       *pilots;   /* np: occupied indices of the pilots              */
    int       *bpb;      /* nd                                              */
    unsigned char *bits; /* ds * bps                                        */
    unsigned char *wire; /* wire_len                                        */
    unsigned char *blk;  /* k + ecc                                         */
    unsigned char *msg;  /* msg_len                                         */
    /* erasure decoding */
    int        nld, era_on, erased;
    double     dmin2[7]; /* by bit width: squared min distance of the table  */
    int       *ld;       /* nld: data index of each loaded carrier, in order */
    float     *cscore;   /* ds * nld: cell score, data symbol major          */
    double    *cmean;    /* nld: mean cell score per loaded carrier          */
    double    *chpow;    /* nld: mean |H|^2 per loaded carrier               */
    float     *bscore;   /* wire_len                                         */
    int       *cand;     /* k + ecc: erasure candidates, worst first         */
    unsigned char *blk2; /* k + ecc                                          */
    /* tracking */
    int        track;    /* OFDM_CXDEM_TRACK_* flags                         */
    double     a_prev, b_prev, coh;
    ofdm_real *eqr, *eqi, *hp; /* nld: equalised point, |He|^2 per loaded col */
    int       *dec;      /* nld: decided point index                         */
    int       *colof;    /* bps: bit offset in a symbol -> loaded col        */
    int       *didx;     /* nd: data carrier -> occupied index               */
    int        n_pilot_fallback;
    /* inner FEC */
    int        fec;
    long       ncod, pstep;
    float     *llr;      /* ds * bps, frame order                            */
    float     *llrc;     /* ncod, coded order                                */
    unsigned char *ibits;/* wire_len * 8                                     */
    void      *fw;       size_t fwlen;
    /* decoder-aided re-estimation (ofdm_cxdem_set_iterate; FEC frames only) */
    int        it_passes, it_win, it_used;
    ofdm_cpx  *Ys;       /* body * O: every window's Y / sqrt(N), CE first   */
    ofdm_cpx  *Xs;       /* body * O: known / re-encoded cell, 0 = unknown   */
    unsigned char *fbits;/* ds * bps: re-encoded frame bits                 */
    unsigned char *icod; /* ncod                                             */
    unsigned char inv[7][64]; /* by width: bit value -> table point          */
};

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

static int width_ok(int b)
{
    return b == 0 || b == 1 || b == 2 || b == 4 || b == 6;
}

/* Same plan as ofdm_cxmod: the frame arithmetic only reads counts. */
static int cxdem_plan(const ofdm_cfg *c, const int *bpb, int fec, int *bps, int *depth,
                      int *k, int *wire, int *ds, int *body)
{
    int i;
    if (!c || !bpb || c->n_occupied < 1 || c->pilot_spacing < 1) return -1;
    *bps = 0;
    for (i = 0; i < c->n_data_carriers; i++) {
        if (!width_ok(bpb[i])) return -1;
        *bps += bpb[i];
    }
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

static int n_pilots_of(const ofdm_cfg *c)
{
    return (c->n_occupied + c->pilot_spacing - 1) / c->pilot_spacing;
}

/* Term-for-term the carving in ofdm_cxdem_init.  Edit the two together. */
size_t ofdm_cxdem_workspace(const ofdm_cxcfg *x, const int *bits_per_bin)
{
    return ofdm_cxdem_workspace_fec(x, bits_per_bin, OFDM_CXFEC_NONE);
}

size_t ofdm_cxdem_workspace_fec(const ofdm_cxcfg *x, const int *bits_per_bin, int fec)
{
    const ofdm_cfg *c;
    size_t fftw, rsw, O;
    int    bps, depth, k, wire, ds, body, np;

    if (!x) return 0;
    c = &x->c;
    if (cxdem_plan(c, bits_per_bin, fec, &bps, &depth, &k, &wire, &ds, &body) != 0)
        return 0;
    fftw = ofdm_cfft_workspace(c->n_fft);
    rsw  = rs255_workspace(c->ecc_bytes);
    if (fftw == 0 || rsw == 0) return 0;
    O  = (size_t)c->n_occupied;
    np = n_pilots_of(c);

    return al8(sizeof(struct ofdm_cxdem))
         + al8(fftw)
         + al8(rsw)
         + al8((size_t)c->n_fft * sizeof(ofdm_cpx))             /* X      */
         + al8(O * sizeof(ofdm_cpx))                            /* H      */
         + al8(O * sizeof(double))                              /* ref    */
         + al8(O * sizeof(int))                                 /* fidx   */
         + al8(O * sizeof(int))                                 /* sbin   */
         + al8(O)                                               /* pmask  */
         + al8((size_t)np * sizeof(int))                        /* pilots */
         + al8((size_t)c->n_data_carriers * sizeof(int))        /* bpb    */
         + al8((size_t)ds * bps)                                /* bits   */
         + al8((size_t)wire)                                    /* wire   */
         + al8((size_t)(k + c->ecc_bytes))                      /* blk    */
         + al8((size_t)(c->seq_bytes + c->chunk_bytes))         /* msg    */
         + al8((size_t)c->n_data_carriers * sizeof(int))        /* ld     */
         + al8((size_t)ds * c->n_data_carriers * sizeof(float)) /* cscore */
         + al8((size_t)c->n_data_carriers * sizeof(double))     /* cmean  */
         + al8((size_t)c->n_data_carriers * sizeof(double))     /* chpow  */
         + al8((size_t)wire * sizeof(float))                    /* bscore */
         + al8((size_t)(k + c->ecc_bytes) * sizeof(int))        /* cand   */
         + al8((size_t)(k + c->ecc_bytes))                      /* blk2   */
         + 3 * al8((size_t)c->n_data_carriers * sizeof(ofdm_real)) /* eqr eqi hp */
         + al8((size_t)c->n_data_carriers * sizeof(int))        /* dec    */
         + al8((size_t)bps * sizeof(int))                       /* colof  */
         + al8((size_t)c->n_data_carriers * sizeof(int))        /* didx   */
         + (fec ? al8((size_t)ds * bps * sizeof(float))         /* llr    */
                + al8((size_t)ofdm_cxfec_coded_bits(fec, (long)wire * 8) * sizeof(float))
                + al8((size_t)wire * 8)                         /* ibits  */
                + al8(ofdm_cxfec_workspace((long)wire * 8))
                + 2 * al8((size_t)body * O * sizeof(ofdm_cpx))  /* Ys Xs  */
                + al8((size_t)ds * bps)                         /* fbits  */
                + al8((size_t)ofdm_cxfec_coded_bits(fec, (long)wire * 8)) /* icod */
                : 0);
}

int ofdm_cxdem_init(ofdm_cxdem **dp, void *mem, size_t memlen,
                    const ofdm_cxcfg *x, const int *bits_per_bin)
{
    return ofdm_cxdem_init_fec(dp, mem, memlen, x, bits_per_bin, OFDM_CXFEC_NONE);
}

int ofdm_cxdem_init_fec(ofdm_cxdem **dp, void *mem, size_t memlen,
                        const ofdm_cxcfg *x, const int *bits_per_bin, int fec)
{
    struct ofdm_cxdem *d;
    const ofdm_cfg *c;
    unsigned char *p = (unsigned char *)mem;
    size_t need, fftw, rsw;
    int    i, j, sum = 0, bps, depth, k, wire, ds, body, O, b;

    if (!dp || !mem || !x || !bits_per_bin) return -1;
    c = &x->c;
    for (i = 0; i < c->n_data_carriers; i++) {
        if (!width_ok(bits_per_bin[i])) return -4;
        sum += bits_per_bin[i];
    }
    if (sum < 1) return -5;
    if (cxdem_plan(c, bits_per_bin, fec, &bps, &depth, &k, &wire, &ds, &body) != 0)
        return -1;
    need = ofdm_cxdem_workspace_fec(x, bits_per_bin, fec);
    if (need == 0) return -1;
    if (memlen < need) return -2;
    fftw = ofdm_cfft_workspace(c->n_fft);
    rsw  = rs255_workspace(c->ecc_bytes);
    O    = c->n_occupied;

    d = (struct ofdm_cxdem *)p; p += al8(sizeof(struct ofdm_cxdem));
    d->x  = *x;
    d->N  = c->n_fft;  d->CP = c->cp_len;  d->SS = c->symbol_samples;
    /* FFT window offset in the symbol: the middle of the ISI-free stretch.
     * Symbol s is [CP+T | body | T] overlap-added at s*SS, so the previous
     * symbol's fall ends at 2T and the next symbol's rise starts at SS: a
     * window start in [2T, CP] sees one clean cyclic period, and 2T is what
     * the taper costs out of the prefix.  The real path's cp_len is that
     * range's RIGHT edge; with the taper at its cp_len/2 clamp the range is
     * that single point and any echo is ISI. */
    d->w0 = 2 * c->taper_samples <= c->cp_len
          ? c->taper_samples + c->cp_len / 2 : c->cp_len;
    d->O  = O;         d->nd = c->n_data_carriers;  d->np = n_pilots_of(c);
    d->bps = bps; d->ds = ds; d->n_syncs = ofdm_n_syncs(c, ds); d->body = body;
    d->depth = depth; d->k = k; d->wire_len = wire;
    d->msg_len = c->seq_bytes + c->chunk_bytes;

    if (ofdm_cfft_init(&d->fft, d->N, p, fftw) != 0) return -3;
    p += al8(fftw);
    if (rs255_init(&d->rs, p, rsw, c->ecc_bytes) != 0) return -6;
    p += al8(rsw);
    d->X      = (ofdm_cpx *)p; p += al8((size_t)d->N * sizeof(ofdm_cpx));
    d->H      = (ofdm_cpx *)p; p += al8((size_t)O * sizeof(ofdm_cpx));
    d->ref    = (double *)p;   p += al8((size_t)O * sizeof(double));
    d->fidx   = (int *)p;      p += al8((size_t)O * sizeof(int));
    d->sbin   = (int *)p;      p += al8((size_t)O * sizeof(int));
    d->pmask  = p;             p += al8((size_t)O);
    d->pilots = (int *)p;      p += al8((size_t)d->np * sizeof(int));
    d->bpb    = (int *)p;      p += al8((size_t)d->nd * sizeof(int));
    d->bits   = p;             p += al8((size_t)ds * bps);
    d->wire   = p;             p += al8((size_t)wire);
    d->blk    = p;             p += al8((size_t)(k + c->ecc_bytes));
    d->msg    = p;             p += al8((size_t)d->msg_len);
    d->ld     = (int *)p;      p += al8((size_t)d->nd * sizeof(int));
    d->cscore = (float *)p;    p += al8((size_t)ds * d->nd * sizeof(float));
    d->cmean  = (double *)p;   p += al8((size_t)d->nd * sizeof(double));
    d->chpow  = (double *)p;   p += al8((size_t)d->nd * sizeof(double));
    d->bscore = (float *)p;    p += al8((size_t)wire * sizeof(float));
    d->cand   = (int *)p;      p += al8((size_t)(k + c->ecc_bytes) * sizeof(int));
    d->blk2   = p;             p += al8((size_t)(k + c->ecc_bytes));
    d->eqr    = (ofdm_real *)p; p += al8((size_t)d->nd * sizeof(ofdm_real));
    d->eqi    = (ofdm_real *)p; p += al8((size_t)d->nd * sizeof(ofdm_real));
    d->hp     = (ofdm_real *)p; p += al8((size_t)d->nd * sizeof(ofdm_real));
    d->dec    = (int *)p;      p += al8((size_t)d->nd * sizeof(int));
    d->colof  = (int *)p;      p += al8((size_t)bps * sizeof(int));
    d->didx   = (int *)p;      p += al8((size_t)d->nd * sizeof(int));
    d->n_pilot_fallback = 0;
    d->fec = fec; d->ncod = 0; d->pstep = 1;
    d->llr = d->llrc = 0; d->ibits = 0; d->fw = 0; d->fwlen = 0;
    if (fec) {
        d->ncod  = ofdm_cxfec_coded_bits(fec, (long)wire * 8);
        d->pstep = ofdm_cxfec_perm_step(d->ncod, bps);
        d->llr   = (float *)p;  p += al8((size_t)ds * bps * sizeof(float));
        d->llrc  = (float *)p;  p += al8((size_t)d->ncod * sizeof(float));
        d->ibits = p;           p += al8((size_t)wire * 8);
        d->fw    = p;           d->fwlen = ofdm_cxfec_workspace((long)wire * 8);
        p += al8(d->fwlen);
        d->Ys    = (ofdm_cpx *)p; p += al8((size_t)body * O * sizeof(ofdm_cpx));
        d->Xs    = (ofdm_cpx *)p; p += al8((size_t)body * O * sizeof(ofdm_cpx));
        d->fbits = p;           p += al8((size_t)ds * bps);
        d->icod  = p;           p += al8((size_t)d->ncod);
    } else {
        d->Ys = d->Xs = 0; d->fbits = d->icod = 0;
    }
    d->it_passes = 0; d->it_win = 2; d->it_used = 0;
    for (b = 0; b < 7; b++) {
        const double *pr, *pim;
        const unsigned char *bt;
        int np_, u, q, v;
        for (u = 0; u < 64; u++) d->inv[b][u] = 0;
        if (!width_ok(b) || b == 0 || ofdm_mod_table(b, &pr, &pim, &bt, &np_) != 0) continue;
        for (u = 0; u < np_ && u < 64; u++) {
            for (v = 0, q = 0; q < b; q++) v = (v << 1) | bt[u * b + q];
            d->inv[b][v] = (unsigned char)u;
        }
    }
    d->track = 0; d->a_prev = d->b_prev = 0.0; d->coh = 1.0;
    d->era_on = 1;
    d->erased = 0;

    /* CE and SYNC: ofdm_cxmod's known BPSK. */
    ofdm_known_bpsk(d->ref, O, OFDM_SEED_CE, OFDM_REF_SYM_BOOST / sqrt(2.0));
    for (i = 0, j = 0; i < O; i++) {
        b = ofdm_cx_bin(&d->x, i);
        d->sbin[i] = b;
        d->fidx[i] = b < 0 ? b + d->N : b;
        d->pmask[i] = (unsigned char)(i % c->pilot_spacing == 0);
        if (d->pmask[i]) d->pilots[j++] = i;
    }
    for (i = 0; i < d->nd; i++) d->bpb[i] = bits_per_bin[i];
    for (i = 0; i < d->nd; i++)
        if (d->bpb[i] && ofdm_mod_table(d->bpb[i], 0, 0, 0, 0) != 0) return -4;
    for (d->nld = 0, i = 0; i < d->nd; i++)
        if (d->bpb[i]) d->ld[d->nld++] = i;
    for (j = 0, i = 0; i < d->nld; i++)
        for (b = 0; b < d->bpb[d->ld[i]]; b++) d->colof[j++] = i;
    for (j = 0, i = 0; i < O; i++)
        if (!d->pmask[i]) d->didx[j++] = i;
    for (b = 0; b < 7; b++) {
        const double *pr, *pim;
        const unsigned char *bt;
        int np_, u, v;
        double m = 0.0;
        d->dmin2[b] = 1.0;
        if (!width_ok(b) || b == 0 || ofdm_mod_table(b, &pr, &pim, &bt, &np_) != 0) continue;
        for (u = 0; u < np_; u++)
            for (v = u + 1; v < np_; v++) {
                double dr = pr[u] - pr[v], dq = pim[u] - pim[v], e = dr * dr + dq * dq;
                if (m == 0.0 || e < m) m = e;
            }
        if (m > 0.0) d->dmin2[b] = m;
    }

    *dp = d;
    return 0;
}

int  ofdm_cxdem_data_syms(const ofdm_cxdem *d)     { return d ? d->ds : 0; }
int  ofdm_cxdem_body_syms(const ofdm_cxdem *d)     { return d ? d->body : 0; }
int  ofdm_cxdem_payload_bytes(const ofdm_cxdem *d) { return d ? d->wire_len : 0; }
long ofdm_cxdem_frame_samples(const ofdm_cxdem *d)
{
    return d ? (long)d->SS * d->body + d->w0 + d->N : 0;
}
const unsigned char *ofdm_cxdem_wire(const ofdm_cxdem *d) { return d ? d->wire : 0; }

/* numpy's complex divide (Smith), as ofdm_dem.c. */
static void cdiv(ofdm_real ar, ofdm_real ai, ofdm_real br, ofdm_real bi,
                 ofdm_real *qr, ofdm_real *qi)
{
    ofdm_real r, den;
    if (fabs((double)br) >= fabs((double)bi)) {
        r = bi / br; den = br + bi * r;
        *qr = (ar + ai * r) / den;
        *qi = (ai - ar * r) / den;
    } else {
        r = br / bi; den = br * r + bi;
        *qr = (ar * r + ai) / den;
        *qi = (ai * r - ar) / den;
    }
}

/* Window [pos + w0, + N) derotated by cfo (phase 0 at frame_start), FFT'd
 * into d->X.  fmod keeps the phase argument small, as ofdm_cx_derotate. */
static void win_fft(ofdm_cxdem *d, const double *rx, int pos, int frame_start,
                    double cfo)
{
    int    t;
    long   a = (long)pos + d->w0;
    double w = -2.0 * CX_PI / (double)d->N, ph, cs, sn, re, im;
    for (t = 0; t < d->N; t++) {
        ph = w * fmod(cfo * (double)(a + t - frame_start), (double)d->N);
        cs = cos(ph);
        sn = sin(ph);
        re = rx[2 * (a + t)];
        im = rx[2 * (a + t) + 1];
        d->X[t].re = (ofdm_real)(re * cs - im * sn);
        d->X[t].im = (ofdm_real)(re * sn + im * cs);
    }
    ofdm_cfft_fwd(&d->fft, d->X);
}

/* H = (X / sqrt(N)) / ref, or the mean with the H already there (blend). */
static void estimate_h(ofdm_cxdem *d, int blend)
{
    int i;
    ofdm_real sc = (ofdm_real)(1.0 / sqrt((double)d->N));
    for (i = 0; i < d->O; i++) {
        ofdm_cpx  Y = d->X[d->fidx[i]];
        ofdm_real nr = Y.re * sc / (ofdm_real)d->ref[i];
        ofdm_real ni = Y.im * sc / (ofdm_real)d->ref[i];
        if (blend) {
            d->H[i].re = (ofdm_real)0.5 * (d->H[i].re + nr);
            d->H[i].im = (ofdm_real)0.5 * (d->H[i].im + ni);
        } else {
            d->H[i].re = nr;
            d->H[i].im = ni;
        }
    }
}

/* r = (X/sqrt(N)) / (H * pilot) and weight |H|^2 for occupied index i.
 * 0 if the carrier is dead. */
static int pilot_resid(ofdm_cxdem *d, int i, double *rr, double *ri, double *w)
{
    ofdm_cpx  Hk = d->H[i], Y = d->X[d->fidx[i]];
    ofdm_real sc = (ofdm_real)(1.0 / sqrt((double)d->N));
    ofdm_real dr, di, qr, qi;
    *w = (double)Hk.re * (double)Hk.re + (double)Hk.im * (double)Hk.im;
    if (*w < CXDEM_H_EPS * CXDEM_H_EPS) return 0;
    dr = Hk.re * (ofdm_real)CX_PILOT_RE - Hk.im * (ofdm_real)CX_PILOT_IM;
    di = Hk.re * (ofdm_real)CX_PILOT_IM + Hk.im * (ofdm_real)CX_PILOT_RE;
    cdiv(Y.re * sc, Y.im * sc, dr, di, &qr, &qi);
    *rr = (double)qr;
    *ri = (double)qi;
    return 1;
}

/* Pilot phase a + b * signed_bin; see the header (step 3) and ofdm_dem.c's
 * estimate_cpe_ramp, of which this is the signed-bin form. */
static void cpe_ramp(ofdm_cxdem *d, double *a_out, double *b_out)
{
    int    j, prev = 0, have = 0, gap = 0;
    double rr, ri, w, prr = 0.0, pri = 0.0, pw = 0.0;
    double accr = 0.0, acci = 0.0, ss = 0.0, cs = 0.0, b = 0.0, e, c, s;

    *a_out = 0.0; *b_out = 0.0;
    for (j = 0; j < d->np; j++) {                    /* smallest usable gap */
        int bin = d->sbin[d->pilots[j]];
        if (!pilot_resid(d, d->pilots[j], &rr, &ri, &w)) continue;
        if (have && (gap == 0 || bin - prev < gap)) gap = bin - prev;
        prev = bin; have = 1;
    }
    if (gap > 0) {                                   /* slope at that gap  */
        have = 0;
        for (j = 0; j < d->np; j++) {
            int bin = d->sbin[d->pilots[j]];
            if (!pilot_resid(d, d->pilots[j], &rr, &ri, &w)) continue;
            if (have && bin - prev == gap) {
                double t = w * pw;
                accr += t * (rr * prr + ri * pri);
                acci += t * (ri * prr - rr * pri);
            }
            prev = bin; prr = rr; pri = ri; pw = w; have = 1;
        }
        if (accr != 0.0 || acci != 0.0) b = atan2(acci, accr) / (double)gap;
    }
    for (j = 0; j < d->np; j++) {                    /* intercept          */
        int bin = d->sbin[d->pilots[j]];
        if (!pilot_resid(d, d->pilots[j], &rr, &ri, &w)) continue;
        c = cos(b * (double)bin);
        s = -sin(b * (double)bin);
        e = atan2(ri * c + rr * s, rr * c - ri * s);
        ss += w * sin(e);
        cs += w * cos(e);
    }
    if (ss != 0.0 || cs != 0.0) *a_out = atan2(ss, cs);
    *b_out = b;
    {   /* weighted phase coherence of the pilots after a + b*bin (1 = all agree) */
        double tw = 0.0;
        for (j = 0; j < d->np; j++)
            if (pilot_resid(d, d->pilots[j], &rr, &ri, &w)) tw += w;
        d->coh = tw > 0.0 ? sqrt(ss * ss + cs * cs) / tw : 0.0;
    }
}

/* Per wire byte reliability (higher = worse) from the cell scores of the
 * frame just demodulated; see CXDEM_* at the top. */
static void byte_scores(ofdm_cxdem *d, int data_cnt)
{
    int    s, col, t, bit = 0, nb = d->wire_len * 8;
    double avg = 0.0;
    for (col = 0; col < d->nld; col++) {
        d->cmean[col] /= (double)data_cnt;
        d->chpow[col] /= (double)data_cnt;
        avg += d->chpow[col];
    }
    if (d->nld > 0) avg /= (double)d->nld;
    for (t = 0; t < d->wire_len; t++) d->bscore[t] = 0.0f;
    if (d->fec) {                 /* info bit j -> the cell of its first coded bit */
        long j, nw = (long)d->wire_len * 8;
        for (j = 0; j < nw; j++) {
            long ci = ofdm_cxfec_coded_index(d->fec, j), pos;
            int  sy, cl;
            float v;
            if (ci < 0 || ci >= d->ncod) continue;
            pos = (long)((ci * (double)d->pstep) - (double)d->ncod * floor((ci * (double)d->pstep) / (double)d->ncod));
            sy = (int)(pos / d->bps);
            cl = d->colof[pos % d->bps];
            if (sy >= data_cnt) continue;
            v = d->cscore[(size_t)sy * d->nld + cl];
            if ((float)d->cmean[cl] > v) v = (float)d->cmean[cl];
            if (d->chpow[cl] < CXDEM_DEAD_RATIO * avg) v = CXDEM_DEAD_SCORE;
            if (v > d->bscore[j >> 3]) d->bscore[j >> 3] = v;
        }
        return;
    }
    for (s = 0; s < data_cnt; s++)
        for (col = 0; col < d->nld; col++) {
            int   bv = d->bpb[d->ld[col]];
            float v  = d->cscore[(size_t)s * d->nld + col];
            if ((float)d->cmean[col] > v) v = (float)d->cmean[col];
            if (d->chpow[col] < CXDEM_DEAD_RATIO * avg) v = CXDEM_DEAD_SCORE;
            for (t = 0; t < bv; t++, bit++)
                if (bit < nb && v > d->bscore[bit >> 3]) d->bscore[bit >> 3] = v;
        }
}

/* De-interleave + RS like ofdm_frame_decode_il, but a codeword that fails the
 * errors-only decode is retried with erasures: the E least reliable bytes
 * (above the floor), E = 1, 2, ... while 2e + E <= ecc - SLACK can hold.
 * Returns msg_len on success, -3 if a codeword stays uncorrectable. */
static int cx_decode(ofdm_cxdem *d, int *errata)
{
    const int ecc = d->x.c.ecc_bytes, L = d->k + ecc, depth = d->depth;
    int i, j, got, err, total = 0, written = 0, failed = 0;

    for (i = 0; i < depth; i++) {
        for (j = 0; j < L; j++) d->blk[j] = d->wire[j * depth + i];
        err = 0;
        got = rs255_decode(d->rs, d->blk, L, d->blk, L, &err);
        if (got < 0 && d->era_on) {
            int nc = 0, E, a, b;
            for (j = 0; j < L; j++) {                /* candidates, worst first */
                float v = d->bscore[j * depth + i];
                if (v <= CXDEM_ERA_FLOOR) continue;
                for (a = nc; a > 0 && d->bscore[d->cand[a - 1] * depth + i] < v; a--)
                    d->cand[a] = d->cand[a - 1];
                d->cand[a] = j;
                nc++;
            }
            for (E = 1; E <= nc && E <= ecc - CXDEM_ERA_SLACK; E++) {
                int er = 0;
                for (j = 0; j < L; j++) d->blk2[j] = d->wire[j * depth + i];
                b = rs255_decode_era(d->rs, d->blk2, L, d->cand, E, d->blk, L, &er);
                if (b < 0) continue;
                if (2 * (er - E) + E > ecc - CXDEM_ERA_SLACK) continue;
                got = b; err = er; d->erased += E;
                break;
            }
        }
        if (got < 0) {
            if (!d->it_passes) return -3;
            failed = 1;                              /* keep going: the codewords */
            continue;                                /* that decode are references */
        }
        if (d->it_passes && d->fec && err > 0
            && rs255_encode(d->rs, d->blk, d->k, d->blk2, L) == L)
            for (j = 0; j < L; j++) d->wire[j * depth + i] = d->blk2[j];
        total += err;
        for (j = 0; j < got; j++) {
            int at = i * d->k + j;
            if (at < d->msg_len) {
                d->msg[at] = d->blk[j];
                if (at + 1 > written) written = at + 1;
            }
        }
    }
    *errata = total;
    if (failed) return -3;
    return written < d->msg_len ? written : d->msg_len;
}

/* Equalise and hard-decide every loaded carrier of the symbol in d->X with
 * phase a + b*bin: d->eqr/eqi (equalised), d->hp (|He|^2), d->dec (point). */
static void decide_sym(ofdm_cxdem *d, double a, double b, ofdm_real sc)
{
    int col;
    for (col = 0; col < d->nld; col++) {
        const double *pr, *pim;
        const unsigned char *bt;
        int i = d->didx[d->ld[col]], bv = d->bpb[d->ld[col]], np_, kk, best = 0;
        double ph, bestd = 0.0, dr, dq, e2;
        ofdm_cpx Y, He;
        ofdm_real cr, ci, eqr, eqi;
        ofdm_mod_table(bv, &pr, &pim, &bt, &np_);
        ph = a + b * (double)d->sbin[i];
        cr = (ofdm_real)cos(ph);
        ci = (ofdm_real)sin(ph);
        Y  = d->X[d->fidx[i]];
        He.re = d->H[i].re * cr - d->H[i].im * ci;
        He.im = d->H[i].re * ci + d->H[i].im * cr;
        d->hp[col] = (ofdm_real)((double)He.re * He.re + (double)He.im * He.im);
        if ((double)d->hp[col] > CXDEM_H_EPS * CXDEM_H_EPS)
            cdiv(Y.re * sc, Y.im * sc, He.re, He.im, &eqr, &eqi);
        else { eqr = Y.re * sc; eqi = Y.im * sc; }
        for (kk = 0; kk < np_; kk++) {               /* ascending, strict < */
            dr = (double)eqr - pr[kk];
            dq = (double)eqi - pim[kk];
            e2 = dr * dr + dq * dq;
            if (kk == 0 || e2 < bestd) { bestd = e2; best = kk; }
        }
        d->eqr[col] = eqr; d->eqi[col] = eqi; d->dec[col] = best;
    }
}

/* Residual phase of the decided cells as da + db*bin (|He|^2 |pt|^2 weighted
 * least squares on cells under CXDEM_DD_R).  0 if too few cells. */
static int dd_phase(ofdm_cxdem *d, double *da, double *db)
{
    int col, n = 0;
    double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, ssn = 0, scs = 0;
    for (col = 0; col < d->nld; col++) {
        const double *pr, *pim;
        int i = d->didx[d->ld[col]], bv = d->bpb[d->ld[col]], np_, k = d->dec[col];
        double dr, dq, r, w, e, x, pr2, er, ei;
        ofdm_mod_table(bv, &pr, &pim, 0, &np_);
        dr = (double)d->eqr[col] - pr[k];
        dq = (double)d->eqi[col] - pim[k];
        r = (dr * dr + dq * dq) / (0.25 * d->dmin2[bv]);
        if (r > CXDEM_DD_R) continue;
        pr2 = pr[k] * pr[k] + pim[k] * pim[k];
        if (pr2 <= 0.0) continue;
        /* angle(eq * conj(pt)) */
        er = (double)d->eqr[col] * pr[k] + (double)d->eqi[col] * pim[k];
        ei = (double)d->eqi[col] * pr[k] - (double)d->eqr[col] * pim[k];
        e = atan2(ei, er);
        w = (double)d->hp[col] * pr2;
        x = (double)d->sbin[i];
        sw += w; sx += w * x; sy += w * e; sxx += w * x * x; sxy += w * x * e;
        ssn += w * sin(e); scs += w * cos(e);
        n++;
    }
    if (n < 8 || sw <= 0.0) return 0;
    {
        double den = sw * sxx - sx * sx;
        *db = fabs(den) > 1e-12 ? (sw * sxy - sx * sy) / den : 0.0;
        *da = (sy - *db * sx) / sw;
    }
    if (fabs(*da) < 1e-4 && fabs(*db) < 1e-6) return 0;
    return 1;
}

/* Move H toward Y / (decision e^{j(a+b bin)}) on reliable data cells and on
 * the pilots (known), so a fade between SYNCs is followed. */
static void track_h(ofdm_cxdem *d, double a, double b, ofdm_real sc)
{
    int col, j;
    const double mu = CXDEM_TRACK_MU;
    for (col = 0; col < d->nld; col++) {
        const double *pr, *pim;
        int i = d->didx[d->ld[col]], bv = d->bpb[d->ld[col]], np_, k = d->dec[col];
        double dr, dq, r, ph, xr, xi, den, yr, yi, hr, hi;
        ofdm_cpx Y = d->X[d->fidx[i]];
        ofdm_mod_table(bv, &pr, &pim, 0, &np_);
        dr = (double)d->eqr[col] - pr[k];
        dq = (double)d->eqi[col] - pim[k];
        r = (dr * dr + dq * dq) / (0.25 * d->dmin2[bv]);
        if (r > CXDEM_TRACK_R || d->hp[col] <= (ofdm_real)0) continue;
        ph = a + b * (double)d->sbin[i];
        xr = pr[k] * cos(ph) - pim[k] * sin(ph);     /* decision with the phase */
        xi = pr[k] * sin(ph) + pim[k] * cos(ph);
        den = xr * xr + xi * xi;
        if (den <= 0.0) continue;
        yr = (double)Y.re * sc; yi = (double)Y.im * sc;
        hr = (yr * xr + yi * xi) / den;
        hi = (yi * xr - yr * xi) / den;
        d->H[i].re = (ofdm_real)((1.0 - mu) * d->H[i].re + mu * hr);
        d->H[i].im = (ofdm_real)((1.0 - mu) * d->H[i].im + mu * hi);
    }
    for (j = 0; j < d->np; j++) {
        int i = d->pilots[j];
        double ph = a + b * (double)d->sbin[i], xr, xi, yr, yi, hr, hi, hw;
        ofdm_cpx Y = d->X[d->fidx[i]];
        hw = (double)d->H[i].re * d->H[i].re + (double)d->H[i].im * d->H[i].im;
        if (hw < CXDEM_H_EPS * CXDEM_H_EPS) continue;
        xr = CX_PILOT_RE * cos(ph) - CX_PILOT_IM * sin(ph);
        xi = CX_PILOT_RE * sin(ph) + CX_PILOT_IM * cos(ph);
        yr = (double)Y.re * sc; yi = (double)Y.im * sc;
        hr = yr * xr + yi * xi;                      /* |pilot| = 1 */
        hi = yi * xr - yr * xi;
        d->H[i].re = (ofdm_real)((1.0 - mu) * d->H[i].re + mu * hr);
        d->H[i].im = (ofdm_real)((1.0 - mu) * d->H[i].im + mu * hi);
    }
}

void ofdm_cxdem_set_tracking(ofdm_cxdem *d, int flags) { if (d) d->track = flags; }
int  ofdm_cxdem_pilot_fallbacks(const ofdm_cxdem *d) { return d ? d->n_pilot_fallback : 0; }

int ofdm_cxdem_set_window(ofdm_cxdem *d, int w0)
{
    const ofdm_cfg *c;
    if (!d) return -1;
    c = &d->x.c;
    if (w0 < 0)
        d->w0 = 2 * c->taper_samples <= c->cp_len ? c->taper_samples + c->cp_len / 2 : c->cp_len;
    else if (w0 > c->cp_len)
        return -1;
    else
        d->w0 = w0;
    return d->w0;
}

void ofdm_cxdem_set_erasures(ofdm_cxdem *d, int on) { if (d) d->era_on = on ? 1 : 0; }

int ofdm_cxdem_carrier_quality(const ofdm_cxdem *d, double *score, double *hrel)
{
    int i, col;
    double avg = 0.0;
    if (!d || !score || !hrel) return -1;
    for (col = 0; col < d->nld; col++) avg += d->chpow[col];
    if (d->nld > 0) avg /= (double)d->nld;
    for (i = 0; i < d->nd; i++) { score[i] = -1.0; hrel[i] = -1.0; }
    for (col = 0; col < d->nld; col++) {
        score[d->ld[col]] = d->cmean[col];
        hrel[d->ld[col]]  = avg > 0.0 ? d->chpow[col] / avg : 0.0;
    }
    return d->nld;
}

/* For the symbol decided in d->eqr/eqi/hp/dec: LLRs (inner code), hard bits,
 * cell scores, EVM terms.  *nbits advances past the symbol's bits. */
static int emit_cells(ofdm_cxdem *d, int data_cnt, int *nbits, double *err, double *ref)
{
    int col, cnt = 0;
    for (col = 0; col < d->nld; col++) {
        const double *pr, *pim;
        const unsigned char *bt;
        int bv = d->bpb[d->ld[col]], np_, kk, best = d->dec[col];
        double dr, dq, bestd, r2;
        ofdm_mod_table(bv, &pr, &pim, &bt, &np_);
        dr = (double)d->eqr[col] - pr[best];
        dq = (double)d->eqi[col] - pim[best];
        bestd = dr * dr + dq * dq;
        if (d->fec) {                                /* max-log LLRs, |He|^2 weighted */
            double m0[6], m1[6];
            for (kk = 0; kk < bv; kk++) { m0[kk] = 1e30; m1[kk] = 1e30; }
            for (kk = 0; kk < np_; kk++) {
                int q;
                double e2;
                dr = (double)d->eqr[col] - pr[kk];
                dq = (double)d->eqi[col] - pim[kk];
                e2 = dr * dr + dq * dq;
                for (q = 0; q < bv; q++) {
                    if (bt[kk * bv + q]) { if (e2 < m1[q]) m1[q] = e2; }
                    else if (e2 < m0[q]) m0[q] = e2;
                }
            }
            for (kk = 0; kk < bv; kk++)
                d->llr[*nbits + kk] = (float)((double)d->hp[col] * (m1[kk] - m0[kk]));
        }
        for (kk = 0; kk < bv; kk++) d->bits[(*nbits)++] = bt[best * bv + kk];
        r2 = pr[best] * pr[best] + pim[best] * pim[best];
        {
            float rs_ = (float)(bestd / (0.25 * d->dmin2[bv]));
            d->cscore[(size_t)data_cnt * d->nld + col] = rs_;
            d->cmean[col] += rs_;
            d->chpow[col] += d->hp[col];
        }
        *err += bestd;
        *ref += r2;
        cnt++;
    }
    return cnt;
}

/* Inner code: de-interleave the LLRs, Viterbi, descramble (the scrambler sits
 * under the code), pack into d->wire. */
static int fec_to_wire(ofdm_cxdem *d)
{
    long ii, pp = 0, nw = (long)d->wire_len * 8;
    unsigned state;
    int i, t;
    for (ii = 0; ii < d->ncod; ii++) {
        d->llrc[ii] = d->llr[pp];
        pp += d->pstep;
        if (pp >= d->ncod) pp -= d->ncod;
    }
    if (ofdm_cxfec_decode(d->fec, d->llrc, nw, d->ibits, d->fw, d->fwlen) != 0) return -3;
    state = OFDM_SEED_DATA & 0xFFFFu;
    for (ii = 0; ii < nw; ii++) {
        unsigned fb = ((state >> 15) ^ (state >> 14) ^ (state >> 12) ^ (state >> 3)) & 1u;
        d->ibits[ii] ^= (unsigned char)((state >> 15) & 1u);
        state = ((state << 1) | fb) & 0xFFFFu;
    }
    for (i = 0; i < d->wire_len; i++) {
        unsigned char v = 0;
        for (t = 0; t < 8; t++) v = (unsigned char)((v << 1) | d->ibits[i * 8 + t]);
        d->wire[i] = v;
    }
    return 0;
}

static int is_sync(const ofdm_cxdem *d, int s)
{
    return d->x.c.sync_period > 0 && s % (d->x.c.sync_period + 1) == 0;
}

/* The transmitted data cells rebuilt from d->wire exactly as ofdm_cxmod_frame
 * does (scramble, encode, interleave, scrambler continuation past the code,
 * map) into d->Xs.  CE / SYNC / pilot cells were stored by the first pass. */
static void reencode(ofdm_cxdem *d)
{
    long nw = (long)d->wire_len * 8, nb = (long)d->ds * d->bps, ii, pos = 0, bit = 0;
    unsigned state = OFDM_SEED_DATA & 0xFFFFu;
    int s, col;
    for (ii = 0; ii < nw; ii++) {
        unsigned fb = ((state >> 15) ^ (state >> 14) ^ (state >> 12) ^ (state >> 3)) & 1u;
        d->ibits[ii] = (unsigned char)(((d->wire[ii / 8] >> (7 - ii % 8)) & 1u) ^ ((state >> 15) & 1u));
        state = ((state << 1) | fb) & 0xFFFFu;
    }
    ofdm_cxfec_encode(d->fec, d->ibits, nw, d->icod);
    for (ii = 0; ii < d->ncod; ii++) {
        d->fbits[pos] = d->icod[ii];
        pos += d->pstep;
        if (pos >= d->ncod) pos -= d->ncod;
    }
    for (ii = d->ncod; ii < nb; ii++) {
        unsigned fb = ((state >> 15) ^ (state >> 14) ^ (state >> 12) ^ (state >> 3)) & 1u;
        d->fbits[ii] = (unsigned char)((state >> 15) & 1u);
        state = ((state << 1) | fb) & 0xFFFFu;
    }
    for (s = 0; s < d->ds + d->n_syncs; s++) {
        ofdm_cpx *X = d->Xs + (size_t)(1 + s) * d->O;
        if (is_sync(d, s)) continue;
        for (col = 0; col < d->nld; col++) {
            const double *pr, *pim;
            int bv = d->bpb[d->ld[col]], v = 0, q, np_, k;
            for (q = 0; q < bv; q++) v = (v << 1) | d->fbits[bit++];
            k = d->inv[bv][v];
            ofdm_mod_table(bv, &pr, &pim, 0, &np_);
            X[d->didx[d->ld[col]]].re = (ofdm_real)pr[k];
            X[d->didx[d->ld[col]]].im = (ofdm_real)pim[k];
        }
    }
}

/* One decoder-aided pass: per loaded carrier, H at symbol t is the least-
 * squares fit sum Y X* / sum |X|^2 over the carrier's known or re-encoded cells
 * in symbols t-W..t+W, t itself left out (its own, possibly wrong, decision
 * must not vouch for itself); then one common phase per symbol from all its
 * cells against the rebuilt references.  Re-decide, LLRs, Viterbi.  The pass
 * replaces cscore / cmean / chpow, so the GMD retry and carrier_quality see it. */
static int iter_pass(ofdm_cxdem *d, int *data_cnt_out)
{
    const int O = d->O, body = d->body, W = d->it_win;
    int s, col, nbits = 0, data_cnt = 0;
    ofdm_cpx *G = d->X;                             /* nld <= N scratch */
    reencode(d);
    for (col = 0; col < d->nld; col++) { d->cmean[col] = 0.0; d->chpow[col] = 0.0; }
    for (s = 0; s < d->ds + d->n_syncs; s++) {
        const int t = 1 + s;
        const ofdm_cpx *Y = d->Ys + (size_t)t * O, *Xt = d->Xs + (size_t)t * O;
        double cr = 0.0, ci = 0.0, ph, c_, s_, er = 0.0, rf = 0.0;
        int u;
        if (is_sync(d, s)) continue;
        for (col = 0; col < d->nld; col++) {
            int i = d->didx[d->ld[col]];
            double nr = 0.0, ni = 0.0, den = 0.0;
            for (u = t - W; u <= t + W; u++) {
                const ofdm_cpx *yu, *xu;
                if (u < 0 || u >= body || u == t) continue;
                yu = d->Ys + (size_t)u * O + i;
                xu = d->Xs + (size_t)u * O + i;
                nr  += (double)yu->re * xu->re + (double)yu->im * xu->im;
                ni  += (double)yu->im * xu->re - (double)yu->re * xu->im;
                den += (double)xu->re * xu->re + (double)xu->im * xu->im;
            }
            G[col].re = (ofdm_real)(den > 0.0 ? nr / den : 0.0);
            G[col].im = (ofdm_real)(den > 0.0 ? ni / den : 0.0);
            {   /* common phase: Y (G X)* */
                double gr = G[col].re, gi = G[col].im, xr = Xt[i].re, xi = Xt[i].im;
                double pr_ = gr * xr - gi * xi, pi_ = gr * xi + gi * xr;
                cr += (double)Y[i].re * pr_ + (double)Y[i].im * pi_;
                ci += (double)Y[i].im * pr_ - (double)Y[i].re * pi_;
            }
        }
        ph = (cr != 0.0 || ci != 0.0) ? atan2(ci, cr) : 0.0;
        c_ = cos(ph); s_ = sin(ph);
        for (col = 0; col < d->nld; col++) {
            const double *pr, *pim;
            int i = d->didx[d->ld[col]], bv = d->bpb[d->ld[col]], np_, kk, best = 0;
            double bestd = 0.0, dr, dq, e2;
            ofdm_real hr = (ofdm_real)(G[col].re * c_ - G[col].im * s_);
            ofdm_real hi = (ofdm_real)(G[col].re * s_ + G[col].im * c_);
            ofdm_real eqr, eqi;
            d->hp[col] = (ofdm_real)((double)hr * hr + (double)hi * hi);
            if ((double)d->hp[col] > CXDEM_H_EPS * CXDEM_H_EPS)
                cdiv(Y[i].re, Y[i].im, hr, hi, &eqr, &eqi);
            else { eqr = Y[i].re; eqi = Y[i].im; }
            ofdm_mod_table(bv, &pr, &pim, 0, &np_);
            for (kk = 0; kk < np_; kk++) {
                dr = (double)eqr - pr[kk];
                dq = (double)eqi - pim[kk];
                e2 = dr * dr + dq * dq;
                if (kk == 0 || e2 < bestd) { bestd = e2; best = kk; }
            }
            d->eqr[col] = eqr; d->eqi[col] = eqi; d->dec[col] = best;
        }
        emit_cells(d, data_cnt, &nbits, &er, &rf);
        data_cnt++;
    }
    *data_cnt_out = data_cnt;
    return data_cnt > 0 ? fec_to_wire(d) : -3;
}

void ofdm_cxdem_set_iterate(ofdm_cxdem *d, int passes, int half_win)
{
    if (!d) return;
    d->it_passes = (d->fec && passes > 0) ? (passes > 8 ? 8 : passes) : 0;
    d->it_win = half_win < 1 ? 1 : (half_win > 16 ? 16 : half_win);
}
int ofdm_cxdem_iter_passes(const ofdm_cxdem *d) { return d ? d->it_used : 0; }

/* First pass: keep every window's Y and the cells known before decoding. */
static void it_store(ofdm_cxdem *d, int t, int known_all, ofdm_real sc)
{
    int i;
    ofdm_cpx *Y = d->Ys + (size_t)t * d->O, *X = d->Xs + (size_t)t * d->O;
    for (i = 0; i < d->O; i++) {
        Y[i].re = d->X[d->fidx[i]].re * sc;
        Y[i].im = d->X[d->fidx[i]].im * sc;
        if (known_all) { X[i].re = (ofdm_real)d->ref[i]; X[i].im = 0; }
        else if (d->pmask[i]) { X[i].re = (ofdm_real)CX_PILOT_RE; X[i].im = (ofdm_real)CX_PILOT_IM; }
        else { X[i].re = 0; X[i].im = 0; }
    }
}

int ofdm_cxdem_frame(ofdm_cxdem *d, const double *rx_iq, int n, int frame_start,
                     double cfo, unsigned char *chunk_out, int out_cap,
                     ofdm_cxdem_out *info)
{
    const ofdm_cfg *c;
    int    pos, s, i, t, nbits = 0, data_cnt = 0, rc, errata = 0, store;
    double evm_acc = 0.0;
    ofdm_real sc;
    unsigned state, seq;

    if (!d || !rx_iq || !chunk_out || !info || frame_start < 0) return -1;
    c = &d->x.c;
    if (out_cap < c->chunk_bytes) return -4;
    if ((long)frame_start + ofdm_cxdem_frame_samples(d) > (long)n) return -2;
    sc = (ofdm_real)(1.0 / sqrt((double)d->N));
    info->evm = 0.0; info->data_syms = 0; info->errata = 0; info->seq = 0;
    info->erased = 0;
    d->erased = 0;
    d->it_used = 0;
    store = d->it_passes > 0 && d->fec;
    for (i = 0; i < d->nld; i++) { d->cmean[i] = 0.0; d->chpow[i] = 0.0; }

    pos = frame_start + d->SS;                       /* skip the SC preamble */
    win_fft(d, rx_iq, pos, frame_start, cfo);
    estimate_h(d, 0);
    if (store) it_store(d, 0, 1, sc);
    d->a_prev = d->b_prev = 0.0;
    pos += d->SS;

    for (s = 0; s < d->ds + d->n_syncs; s++, pos += d->SS) {
        double a, b, err = 0.0, ref = 0.0;
        int    cnt;

        win_fft(d, rx_iq, pos, frame_start, cfo);
        /* SYNC at every (sync_period+1)-th body index after the CE */
        if (is_sync(d, s)) {
            if (store) it_store(d, 1 + s, 1, sc);
            estimate_h(d, s == 0);
            continue;
        }
        if (store) it_store(d, 1 + s, 0, sc);
        /* SIGN, as ofdm_dem: the residual has angle +phase, so H_eff =
         * H e^{+j phase} divides it out. */
        cpe_ramp(d, &a, &b);
        if ((d->track & OFDM_CXDEM_TRACK_PILOTS) && d->coh < CXDEM_PILOT_COH) {
            a = d->a_prev;                           /* pilots not trusted */
            b = d->b_prev;
            d->n_pilot_fallback++;
        }
        decide_sym(d, a, b, sc);
        if (d->track & OFDM_CXDEM_TRACK_PILOTS) {    /* decision-directed a, b */
            double da, db;
            if (dd_phase(d, &da, &db)) {
                a += da; b += db;
                decide_sym(d, a, b, sc);
            }
        }
        d->a_prev = a; d->b_prev = b;
        cnt = emit_cells(d, data_cnt, &nbits, &err, &ref);
        if (d->track & OFDM_CXDEM_TRACK_H) track_h(d, a, b, sc);
        if (cnt > 0) evm_acc += sqrt((err / cnt) / ((ref / cnt) + CXDEM_REF_EPS));
        data_cnt++;
    }
    if (data_cnt == 0) return -3;

    if (d->fec) {
        if (fec_to_wire(d) != 0) return -3;
    } else {
        /* descramble (ofdm_cxmod's scrambler, self-inverse), pack MSB-first */
        state = OFDM_SEED_DATA & 0xFFFFu;
        for (i = 0; i < nbits; i++) {
            unsigned fb = ((state >> 15) ^ (state >> 14)
                         ^ (state >> 12) ^ (state >> 3)) & 1u;
            d->bits[i] ^= (unsigned char)((state >> 15) & 1u);
            state = ((state << 1) | fb) & 0xFFFFu;
        }
        for (i = 0; i < d->wire_len; i++) {
            unsigned char v = 0;
            for (t = 0; t < 8; t++)
                v = (unsigned char)((v << 1) | (i * 8 + t < nbits ? d->bits[i * 8 + t] : 0));
            d->wire[i] = v;
        }
    }
    info->evm       = evm_acc / data_cnt;
    info->data_syms = data_cnt;

    byte_scores(d, data_cnt);
    rc = cx_decode(d, &errata);
    while (rc != d->msg_len && store && d->it_used < d->it_passes) {
        int dc = 0;
        d->it_used++;
        d->erased = 0;
        if (iter_pass(d, &dc) != 0) break;
        byte_scores(d, dc);
        rc = cx_decode(d, &errata);
    }
    info->erased = d->erased;
    if (rc != d->msg_len) return -5;
    seq = 0;
    for (i = 0; i < c->seq_bytes; i++) seq = (seq << 8) | d->msg[i];
    for (i = 0; i < c->chunk_bytes; i++) chunk_out[i] = d->msg[c->seq_bytes + i];
    info->seq    = seq;
    info->errata = errata;
    return 0;
}
