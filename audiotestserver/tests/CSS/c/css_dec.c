/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
#include "css_dec.h"
#include "css_enc.h"

/* Same generators and register convention as css_enc.c. The two files spell
 * them separately because the trellis wants them as a table and the encoder
 * wants them as a shift-register step; they must agree, and the equivalence
 * suite decodes what the encoder produced to catch a divergence. */
#define CONV_G1   0x79      /* 0o171 */
#define CONV_G2   0x5B      /* 0o133 */
#define CONV_K    7
#define CONV_MASK ((1 << CONV_K) - 1)
#define NSTATES   64

#define CRC_BYTES 4

/* Survivor bookkeeping is two bitmaps of NSTATES bits per step. */
#define SURV_BYTES_PER_STEP  (NSTATES / 8)

static int parity(unsigned int v)
{
    int p = 0;

    while (v) {
        p ^= (int)(v & 1u);
        v >>= 1;
    }
    return p;
}

static size_t align8(size_t v) { return (v + 7u) & ~(size_t)7u; }

size_t css_viterbi_workspace(int npairs)
{
    if (npairs < 1) return 0;
    return align8((size_t)NSTATES * sizeof(css_metric)) * 2   /* pm, npm    */
         + align8((size_t)npairs * SURV_BYTES_PER_STEP) * 2   /* surv, live */
         + align8((size_t)NSTATES * 2 * 3);                   /* trellis    */
}

int css_viterbi_k7(const float *conf, int npairs, void *work, size_t worklen,
                   unsigned char *bits, float *energy)
{
    /* Transition table: 128 entries of three bytes, built into the caller's
     * workspace rather than a file-scope static, to keep this translation
     * unit reentrant and stateless. On the target this becomes a const array
     * in flash and the tables come out of the workspace entirely. */
    unsigned char *tr_c1, *tr_c2, *tr_ns;
    unsigned char *p;
    css_metric *pm, *npm;
    unsigned char *surv, *live;
    unsigned char *nlive;
    int i, s, b, ns, step, nbits, cur, prev_lsb;
    css_metric bm, m;
    float pair_e;
    unsigned int reg;

    if (!conf || !bits || !work || npairs < CSS_CONV_TAIL + 1) return -1;
    if (worklen < css_viterbi_workspace(npairs)) return -2;

    p     = (unsigned char *)work;
    pm    = (css_metric *)p; p += align8((size_t)NSTATES * sizeof(css_metric));
    npm   = (css_metric *)p; p += align8((size_t)NSTATES * sizeof(css_metric));
    surv  = p;               p += align8((size_t)npairs * SURV_BYTES_PER_STEP);
    live  = p;               p += align8((size_t)npairs * SURV_BYTES_PER_STEP);
    tr_c1 = p;               p += NSTATES * 2;
    tr_c2 = p;               p += NSTATES * 2;
    tr_ns = p;

    for (s = 0; s < NSTATES; s++)
        for (b = 0; b < 2; b++) {
            reg = ((unsigned int)b << (CONV_K - 1)) | (unsigned int)s;
            reg &= CONV_MASK;
            tr_c1[s * 2 + b] = (unsigned char)parity(reg & CONV_G1);
            tr_c2[s * 2 + b] = (unsigned char)parity(reg & CONV_G2);
            tr_ns[s * 2 + b] = (unsigned char)
                ((((unsigned int)b << (CONV_K - 2)) | ((unsigned int)s >> 1))
                 & 0x3Fu);
        }

    /* live[step] marks which states any path reached; pm for a dead state is
     * never read, so it needs no -INF sentinel and no comparisons against one. */
    for (i = 0; i < npairs * SURV_BYTES_PER_STEP; i++) { surv[i] = 0; live[i] = 0; }
    for (s = 0; s < NSTATES; s++) pm[s] = (css_metric)0;

    /* The encoder starts from a cleared register, so state 0 is the only
     * legal start. */
    {
        unsigned char start[SURV_BYTES_PER_STEP];
        for (i = 0; i < SURV_BYTES_PER_STEP; i++) start[i] = 0;
        start[0] = 1;
        for (step = 0; step < npairs; step++) {
            unsigned char *lv = live + (size_t)step * SURV_BYTES_PER_STEP;
            unsigned char *sv = surv + (size_t)step * SURV_BYTES_PER_STEP;
            const unsigned char *prev_live = (step == 0)
                ? start : (live + (size_t)(step - 1) * SURV_BYTES_PER_STEP);
            float c0 = conf[step * 2 + 0];
            float c1 = conf[step * 2 + 1];

            for (i = 0; i < SURV_BYTES_PER_STEP; i++) lv[i] = 0;
            for (s = 0; s < NSTATES; s++) {
                if (!((prev_live[s >> 3] >> (s & 7)) & 1)) continue;
                for (b = 0; b < 2; b++) {
                    ns = tr_ns[s * 2 + b];
                    /* Correlation metric: reward agreement with the received
                     * soft value, penalise disagreement. Equivalent to the
                     * usual squared-distance metric up to a per-step
                     * constant, and cheaper. */
                    bm = (css_metric)(tr_c1[s * 2 + b] ? c0 : -c0)
                       + (css_metric)(tr_c2[s * 2 + b] ? c1 : -c1);
                    m  = pm[s] + bm;
                    if (!((lv[ns >> 3] >> (ns & 7)) & 1) || m > npm[ns]) {
                        npm[ns] = m;
                        lv[ns >> 3] = (unsigned char)(lv[ns >> 3] | (1 << (ns & 7)));
                        /* One bit is enough to invert the transition: the
                         * input bit is ns's top bit, and s >> 1 == ns & 0x1F,
                         * so only s's own lowest bit is unknown. */
                        if (s & 1) sv[ns >> 3] = (unsigned char)(sv[ns >> 3] | (1 << (ns & 7)));
                        else       sv[ns >> 3] = (unsigned char)(sv[ns >> 3] & ~(1 << (ns & 7)));
                    }
                }
            }
            for (s = 0; s < NSTATES; s++) pm[s] = npm[s];
        }
    }

    /* Terminate at state 0: the encoder's flush tail put it there. Fall back
     * to the best surviving state only if nothing reached 0, meaning the
     * frame is already unrecoverable. */
    nlive = live + (size_t)(npairs - 1) * SURV_BYTES_PER_STEP;
    if (nlive[0] & 1) {
        cur = 0;
    } else {
        cur = -1;
        for (s = 0; s < NSTATES; s++) {
            if (!((nlive[s >> 3] >> (s & 7)) & 1)) continue;
            if (cur < 0 || pm[s] > pm[cur]) cur = s;
        }
        if (cur < 0) return -1;
    }

    nbits = npairs - CSS_CONV_TAIL;
    for (step = npairs - 1; step >= 0; step--) {
        const unsigned char *lv = live + (size_t)step * SURV_BYTES_PER_STEP;
        const unsigned char *sv = surv + (size_t)step * SURV_BYTES_PER_STEP;
        int bit;

        pair_e = (float)0.5 * ((conf[step * 2 + 0] < 0.0f
                                ? -conf[step * 2 + 0] : conf[step * 2 + 0])
                             + (conf[step * 2 + 1] < 0.0f
                                ? -conf[step * 2 + 1] : conf[step * 2 + 1]));
        if (!((lv[cur >> 3] >> (cur & 7)) & 1)) {
            /* No survivor into this state: emit a zero and hold the state.
             * Only reachable on a frame that is already unrecoverable. */
            bit    = 0;
            pair_e = 0.0f;
        } else {
            bit      = (cur >> (CONV_K - 2)) & 1;
            prev_lsb = (sv[cur >> 3] >> (cur & 7)) & 1;
            cur      = (((cur & 0x1F) << 1) | prev_lsb) & CONV_MASK & 0x3F;
        }
        if (step < nbits) {
            bits[step] = (unsigned char)bit;
            if (energy) energy[step] = pair_e;
        }
    }
    return nbits;
}

int css_deinterleave(const unsigned char *bits, const float *conf, int n,
                     int rows, unsigned char *bits_out, float *conf_out)
{
    int cols, r, c, w, src;

    if (!bits || !bits_out || n < 0 || rows < 0) return -1;
    if (rows <= 1) {
        for (r = 0; r < n; r++) {
            bits_out[r] = bits[r] ? 1 : 0;
            if (conf && conf_out) conf_out[r] = conf[r];
        }
        return n;
    }
    if (n % rows != 0) return -1;

    /* The received stream is column-major; read it back row-major. Exact
     * inverse of css_interleave, whose destination walk is the transpose of
     * this one. */
    cols = n / rows;
    w    = 0;
    for (r = 0; r < rows; r++)
        for (c = 0; c < cols; c++) {
            src = c * rows + r;
            bits_out[w] = bits[src] ? 1 : 0;
            if (conf && conf_out) conf_out[w] = conf[src];
            w++;
        }
    return w;
}

int css_diff_decode(const unsigned char *bits, int n, int ref,
                    unsigned char *out)
{
    int i, prev, cur;

    if (!bits || !out || n < 0) return -1;
    prev = ref ? 1 : 0;
    for (i = 0; i < n; i++) {
        /* Read the input before overwriting it, so out == in is safe. */
        cur    = bits[i] ? 1 : 0;
        out[i] = (unsigned char)(cur ^ prev);
        prev   = cur;
    }
    return n;
}

int css_bits_to_bytes(const unsigned char *bits, int n, unsigned char *out)
{
    int i, k, nb;
    unsigned int v;

    if (!bits || !out || n < 0) return -1;
    nb = n / 8;
    for (i = 0; i < nb; i++) {
        v = 0u;
        for (k = 0; k < 8; k++) v = (v << 1) | (unsigned int)(bits[i * 8 + k] ? 1 : 0);
        out[i] = (unsigned char)v;
    }
    return nb;
}

int css_check_crc(const unsigned char *raw, int nraw, int chunk_size, int *seq)
{
    unsigned long want, got;
    int need;

    if (!raw || chunk_size < 0) return -1;
    need = 1 + chunk_size + CRC_BYTES;
    if (nraw < need) return 0;

    want = css_crc32(raw, 1 + chunk_size);
    got  = ((unsigned long)raw[1 + chunk_size + 0] << 24)
         | ((unsigned long)raw[1 + chunk_size + 1] << 16)
         | ((unsigned long)raw[1 + chunk_size + 2] <<  8)
         |  (unsigned long)raw[1 + chunk_size + 3];
    if (got != want) return 0;
    if (seq) *seq = (int)raw[0];
    return 1;
}

size_t css_dec_workspace(const css_enc_cfg *g)
{
    size_t need;

    if (!g) return 0;
    /* Two permuted copies of the received frame (bits + conf), the stage
     * output, and the Viterbi's own survivor store.
     *
     * Every length here is frame_symbols, not base_bits or conv_bits. The
     * Viterbi runs over the whole deinterleaved frame (frame_symbols/2
     * pairs), because that is what _frame_pipeline feeds it; with
     * interleaving on, frame_symbols is conv_bits rounded up to a whole
     * block. Sizing from conv_bits instead would be short by the interleave
     * padding, which is nonzero whenever interleaving is used. */
    need = align8((size_t)g->frame_symbols)
         + align8((size_t)g->frame_symbols * sizeof(float))
         + align8((size_t)g->frame_symbols)
         + align8((size_t)g->frame_symbols * sizeof(float));
    if (g->conv_code)
        need += align8(css_viterbi_workspace(g->frame_symbols / 2));
    return need;
}

int css_dec_raw(const css_enc_cfg *g, const unsigned char *bits,
                const float *conf, int nbits, int diff_ref,
                void *work, size_t worklen,
                unsigned char *raw, int rawcap, float *byte_conf)
{
    unsigned char *p, *db, *ob;
    float *dc, *oc;
    void *vw;
    size_t vwlen;
    int n, i, k, nraw;
    float me, mn, a;

    if (!g || !bits || !conf || !raw || !work) return -1;
    if (nbits != g->frame_symbols) return -1;
    if (rawcap < g->base_bits / 8) return -1;
    if (worklen < css_dec_workspace(g)) return -2;

    p  = (unsigned char *)work;
    db = p;               p += align8((size_t)g->frame_symbols);
    dc = (float *)p;      p += align8((size_t)g->frame_symbols * sizeof(float));
    ob = p;               p += align8((size_t)g->frame_symbols);
    oc = (float *)p;      p += align8((size_t)g->frame_symbols * sizeof(float));
    vw = p;
    vwlen = worklen - (size_t)(p - (unsigned char *)work);

    n = css_deinterleave(bits, conf, nbits,
                         g->interleave_rows > 1 ? g->interleave_rows : 0,
                         db, dc);
    if (n < 0) return -1;

    if (g->conv_code) {
        /* The soft values are the pair sequence: conf[2i], conf[2i+1] is
         * received pair i, so nothing repacks them here. */
        n = css_viterbi_k7(dc, n / 2, vw, vwlen, ob, oc);
        if (n < 0) return n;
        /* Normalise the per-bit energies to the frame's strongest, matching
         * _frame_pipeline's `fc = [e / (max(pe) + 1e-12) for e in pe]`. The
         * absolute scale of a soft energy depends on gain and symbol length,
         * so an erasure threshold is only meaningful against the frame's own
         * maximum. */
        me = 0.0f;
        for (i = 0; i < n; i++) if (oc[i] > me) me = oc[i];
        for (i = 0; i < n; i++) oc[i] = oc[i] / (me + 1e-12f);
    } else {
        n = (n > g->base_bits) ? g->base_bits : n;
        for (i = 0; i < n; i++) { ob[i] = db[i]; oc[i] = dc[i]; }
    }

    if (g->diff_enc) {
        if (css_diff_decode(ob, n, diff_ref, ob) < 0) return -1;
    }

    if (n > g->base_bits) n = g->base_bits;
    nraw = css_bits_to_bytes(ob, n, raw);
    if (nraw < 0) return -1;

    if (byte_conf) {
        /* Minimum |conf| over the byte's eight bits: one weak symbol is
         * enough to make the byte suspect, so min, not mean. */
        for (i = 0; i < nraw; i++) {
            mn = -1.0f;
            for (k = 0; k < 8; k++) {
                if (i * 8 + k >= n) break;
                a = oc[i * 8 + k];
                if (a < 0.0f) a = -a;
                if (mn < 0.0f || a < mn) mn = a;
            }
            byte_conf[i] = (mn < 0.0f) ? 0.0f : mn;
        }
    }
    return nraw;
}
