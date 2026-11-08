/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

#include "rs255.h"

/* GF(2^8) with reedsolo's defaults: prim=0x11D, generator alpha=2, fcr=0. */
#define RS_PRIM   0x11D
#define RS_GEN    2
#define RS_FCR    0
#define RS_NSIZE  255
#define RS_MAXSYM 254

struct rs255 {
    int nsym;          /* parity symbols per block            */
    int k;             /* message bytes per block = 255 - nsym */
    unsigned char *exp;    /* 512 : antilog, doubled so a+b needs no mod */
    unsigned char *log;    /* 256                                        */
    unsigned char *gen;    /* nsym+1 : generator polynomial              */
    /* decode scratch, sized for one block */
    unsigned char *synd;   /* nsym                                       */
    unsigned char *lam;    /* nsym+1                                     */
    unsigned char *prev;   /* nsym+1                                     */
    unsigned char *tmp;    /* nsym+1                                     */
    unsigned char *omega;  /* nsym+1                                     */
    unsigned char *pos;    /* nsym  : error positions within the block   */
    unsigned char *blk;    /* 255   : working copy of one block          */
    unsigned char *era;    /* RS_ERA_SCRATCH : rs255_decode_era scratch  */
};

/* rs255_decode_era works on reedsolo's own polynomial representation
 * (highest degree first, Python list semantics) in 8 rows of 512 bytes. */
#define RS_ERA_ROW     512
#define RS_ERA_SCRATCH (8 * RS_ERA_ROW)

static size_t al8(size_t v) { return (v + 7u) & ~(size_t)7u; }

/* --- GF helpers.  exp[] is doubled to 512 so exp[log[a]+log[b]] needs no
 * modulo on the hot multiply path. --- */
static unsigned char gmul(const rs255 *r, unsigned char a, unsigned char b)
{
    if (a == 0 || b == 0) return 0;
    return r->exp[(int)r->log[a] + (int)r->log[b]];
}

static unsigned char gdiv(const rs255 *r, unsigned char a, unsigned char b)
{
    if (a == 0) return 0;
    /* caller guarantees b != 0 */
    return r->exp[((int)r->log[a] + 255 - (int)r->log[b]) % 255];
}

static unsigned char ginv(const rs255 *r, unsigned char a)
{
    return r->exp[255 - (int)r->log[a]];
}

static unsigned char gpow(const rs255 *r, unsigned char a, int n)
{
    int e;
    if (a == 0) return 0;
    e = ((int)r->log[a] * n) % 255;
    if (e < 0) e += 255;
    return r->exp[e];
}

size_t rs255_workspace(int nsym)
{
    if (nsym < 1 || nsym > RS_MAXSYM) return 0;
    return al8(sizeof(struct rs255))
         + al8(512)                       /* exp   */
         + al8(256)                       /* log   */
         + al8((size_t)nsym + 1)          /* gen   */
         + al8((size_t)nsym)              /* synd  */
         + al8((size_t)nsym + 1)          /* lam   */
         + al8((size_t)nsym + 1)          /* prev  */
         + al8((size_t)nsym + 1)          /* tmp   */
         + al8((size_t)nsym + 1)          /* omega */
         + al8((size_t)nsym)              /* pos   */
         + al8(255)                       /* blk   */
         + al8(RS_ERA_SCRATCH);           /* era   */
}

int rs255_init(rs255 **rsp, void *mem, size_t memlen, int nsym)
{
    struct rs255 *r;
    unsigned char *p = (unsigned char *)mem;
    size_t need;
    int i, x;

    if (!rsp || !mem || nsym < 1 || nsym > RS_MAXSYM) return -1;
    need = rs255_workspace(nsym);
    if (need == 0 || memlen < need) return -2;

    r = (struct rs255 *)p;      p += al8(sizeof(struct rs255));
    r->exp   = p;               p += al8(512);
    r->log   = p;               p += al8(256);
    r->gen   = p;               p += al8((size_t)nsym + 1);
    r->synd  = p;               p += al8((size_t)nsym);
    r->lam   = p;               p += al8((size_t)nsym + 1);
    r->prev  = p;               p += al8((size_t)nsym + 1);
    r->tmp   = p;               p += al8((size_t)nsym + 1);
    r->omega = p;               p += al8((size_t)nsym + 1);
    r->pos   = p;               p += al8((size_t)nsym);
    r->blk   = p;               p += al8(255);
    r->era   = p;

    r->nsym = nsym;
    r->k    = RS_NSIZE - nsym;

    /* log / antilog tables */
    x = 1;
    for (i = 0; i < 255; i++) {
        r->exp[i] = (unsigned char)x;
        r->log[x] = (unsigned char)i;
        x <<= 1;
        if (x & 0x100) x ^= RS_PRIM;
    }
    for (i = 255; i < 512; i++) r->exp[i] = r->exp[i - 255];
    r->log[0] = 0;              /* never read; keeps the table defined */

    /* generator g(x) = prod_{i=0}^{nsym-1} (x - alpha^(i+fcr)), fcr=0.
     * Coefficients are stored HIGHEST DEGREE FIRST throughout this file, so
     * g(x) = sum_j gen[j] * x^(deg-j) and gen[0] is always 1.
     *
     * Multiplying a degree-d polynomial by (x + root) -- minus is plus in
     * characteristic 2 -- gives, for the new degree-(d+1) coefficients:
     *     new[i] = gen[i]                (present while i <= d)
     *            ^ root * gen[i-1]       (present while i >= 1)
     * Walking i downward lets that happen in place. */
    for (i = 0; i <= nsym; i++) r->gen[i] = 0;
    r->gen[0] = 1;
    for (i = 0; i < nsym; i++) {
        unsigned char root = gpow(r, (unsigned char)RS_GEN, i + RS_FCR);
        int j;
        for (j = i + 1; j >= 1; j--)
            r->gen[j] = (unsigned char)(r->gen[j] ^ gmul(r, root, r->gen[j - 1]));
    }

    *rsp = r;
    return 0;
}

int rs255_nsym(const rs255 *r)    { return r ? r->nsym : 0; }
int rs255_block_k(const rs255 *r) { return r ? r->k    : 0; }

int rs255_encoded_len(const rs255 *r, int n)
{
    int nblk;
    if (!r || n < 0) return -1;
    if (n == 0) return 0;
    nblk = (n + r->k - 1) / r->k;
    return n + nblk * r->nsym;
}

/* Systematic encode of ONE block: msg[len] -> par[nsym]. */
static void encode_block(rs255 *r, const unsigned char *msg, int len,
                         unsigned char *par)
{
    int i, j;
    for (i = 0; i < r->nsym; i++) par[i] = 0;
    for (i = 0; i < len; i++) {
        unsigned char fb = (unsigned char)(msg[i] ^ par[0]);
        for (j = 0; j < r->nsym - 1; j++)
            par[j] = (unsigned char)(par[j + 1] ^ gmul(r, fb, r->gen[j + 1]));
        par[r->nsym - 1] = gmul(r, fb, r->gen[r->nsym]);
    }
}

int rs255_encode(rs255 *r, const unsigned char *msg, int n,
                 unsigned char *out, int out_cap)
{
    int off = 0, wr = 0;
    if (!r || !msg || !out || n < 0) return -1;
    if (rs255_encoded_len(r, n) > out_cap) return -2;
    while (off < n) {
        int len = n - off;
        int i;
        if (len > r->k) len = r->k;
        for (i = 0; i < len; i++) out[wr + i] = msg[off + i];
        encode_block(r, msg + off, len, out + wr + len);
        wr  += len + r->nsym;
        off += len;
    }
    return wr;
}

/* Decode ONE block in place: blk[len] where len = data + nsym.
 * Returns corrected symbol count, or -1 uncorrectable. */
static int decode_block(rs255 *r, unsigned char *blk, int len)
{
    const int nsym = r->nsym;
    int i, j, nerr = 0, lam_len, L = 0, m = 1, count = 0;
    unsigned char b_scale = 1;
    int synd_zero = 1;

    /* --- syndromes: S_i = blk(alpha^(i+fcr)), Horner, blk[0] highest degree */
    for (i = 0; i < nsym; i++) {
        unsigned char xa = gpow(r, (unsigned char)RS_GEN, i + RS_FCR);
        unsigned char acc = 0;
        for (j = 0; j < len; j++) acc = (unsigned char)(gmul(r, acc, xa) ^ blk[j]);
        r->synd[i] = acc;
        if (acc) synd_zero = 0;
    }
    if (synd_zero) return 0;

    /* --- Berlekamp-Massey: error locator Lambda, LOWEST degree first here
     * (lam[0] == 1), which is the natural form for the recurrence. */
    for (i = 0; i <= nsym; i++) { r->lam[i] = 0; r->prev[i] = 0; }
    r->lam[0] = 1; r->prev[0] = 1;
    lam_len = 1;
    {
        int prev_len = 1;
        for (i = 0; i < nsym; i++) {
            unsigned char delta = r->synd[i];
            for (j = 1; j < lam_len; j++)
                delta = (unsigned char)(delta ^ gmul(r, r->lam[j], r->synd[i - j]));
            if (delta == 0) {
                m++;
            } else if (2 * L <= i) {
                /* copy lam -> tmp, update, then prev = tmp (old lam) */
                for (j = 0; j <= nsym; j++) r->tmp[j] = r->lam[j];
                {
                    int tl = lam_len;
                    unsigned char scale = gdiv(r, delta, b_scale);
                    for (j = 0; j < prev_len; j++) {
                        int idx = j + m;
                        if (idx <= nsym) {
                            r->lam[idx] = (unsigned char)
                                (r->lam[idx] ^ gmul(r, scale, r->prev[j]));
                            if (idx + 1 > lam_len) lam_len = idx + 1;
                        }
                    }
                    for (j = 0; j <= nsym; j++) r->prev[j] = r->tmp[j];
                    prev_len = tl;
                }
                L = i + 1 - L;
                b_scale = delta;
                m = 1;
            } else {
                unsigned char scale = gdiv(r, delta, b_scale);
                for (j = 0; j < prev_len; j++) {
                    int idx = j + m;
                    if (idx <= nsym) {
                        r->lam[idx] = (unsigned char)
                            (r->lam[idx] ^ gmul(r, scale, r->prev[j]));
                        if (idx + 1 > lam_len) lam_len = idx + 1;
                    }
                }
                m++;
            }
        }
    }
    while (lam_len > 1 && r->lam[lam_len - 1] == 0) lam_len--;
    nerr = lam_len - 1;
    if (nerr <= 0 || nerr > nsym / 2) return -1;

    /* --- Chien search: Lambda(alpha^-p) == 0 means an error at exponent p,
     * i.e. at index len-1-p from the left. */
    for (i = 0; i < len; i++) {
        int p = len - 1 - i;                       /* exponent at index i */
        unsigned char xinv = gpow(r, (unsigned char)RS_GEN, -p);
        unsigned char acc = 0, xp = 1;
        for (j = 0; j < lam_len; j++) {
            acc = (unsigned char)(acc ^ gmul(r, r->lam[j], xp));
            xp = gmul(r, xp, xinv);
        }
        if (acc == 0) {
            if (count >= nsym) return -1;
            r->pos[count++] = (unsigned char)i;    /* index from the left */
        }
    }
    if (count != nerr) return -1;                  /* roots outside the block */

    /* --- Omega(x) = S(x) * Lambda(x) mod x^nsym  (both lowest-degree-first) */
    for (i = 0; i <= nsym; i++) r->omega[i] = 0;
    for (i = 0; i < nsym; i++) {
        for (j = 0; j < lam_len && i + j < nsym; j++)
            r->omega[i + j] = (unsigned char)
                (r->omega[i + j] ^ gmul(r, r->synd[i], r->lam[j]));
    }

    /* --- Forney: Y = X^(1-fcr) * Omega(X^-1) / Lambda'(X^-1).  fcr=0, so the
     * X^(1-fcr) factor is a single extra X. */
    for (i = 0; i < count; i++) {
        int idx = r->pos[i];
        int p   = len - 1 - idx;
        unsigned char X    = gpow(r, (unsigned char)RS_GEN, p);
        unsigned char Xinv = ginv(r, X);
        unsigned char num = 0, den = 0, xp = 1;
        for (j = 0; j < nsym; j++) {
            num = (unsigned char)(num ^ gmul(r, r->omega[j], xp));
            xp  = gmul(r, xp, Xinv);
        }
        /* Lambda'(x): in characteristic 2 only the ODD-degree terms survive,
         * and each contributes its coefficient at degree-1. */
        xp = 1;
        for (j = 1; j < lam_len; j += 2) {
            den = (unsigned char)(den ^ gmul(r, r->lam[j], xp));
            xp = gmul(r, xp, gmul(r, Xinv, Xinv));
        }
        if (den == 0) return -1;
        {
            unsigned char y = gdiv(r, num, den);
            y = gmul(r, y, gpow(r, X, 1 - RS_FCR));
            blk[idx] = (unsigned char)(blk[idx] ^ y);
        }
    }

    /* --- verify: a corrected codeword must have all-zero syndromes.  reedsolo
     * performs the same check and raises when it fails, which is what stops a
     * beyond-capacity block from being handed back as if it were repaired. */
    for (i = 0; i < nsym; i++) {
        unsigned char xa = gpow(r, (unsigned char)RS_GEN, i + RS_FCR);
        unsigned char acc = 0;
        for (j = 0; j < len; j++) acc = (unsigned char)(gmul(r, acc, xa) ^ blk[j]);
        if (acc) return -1;
    }
    return count;
}

int rs255_decode(rs255 *r, const unsigned char *buf, int n,
                 unsigned char *out, int out_cap, int *errata)
{
    int off = 0, wr = 0, total = 0;
    if (!r || !buf || !out || !errata || n < 0) return -1;
    *errata = 0;
    while (off < n) {
        int blen = n - off;
        int dlen, ne, i;
        if (blen > RS_NSIZE) blen = RS_NSIZE;
        if (blen <= r->nsym) return -1;         /* a block with no data */
        dlen = blen - r->nsym;
        for (i = 0; i < blen; i++) r->blk[i] = buf[off + i];
        ne = decode_block(r, r->blk, blen);
        if (ne < 0) return -3;
        total += ne;
        if (wr + dlen > out_cap) return -2;
        for (i = 0; i < dlen; i++) out[wr + i] = r->blk[i];
        wr  += dlen;
        off += blen;
    }
    *errata = total;
    return wr;
}


/* ===========================================================================
 * Errors-and-erasures decode, a line-for-line port of reedsolo 1.7
 * rs_correct_msg (Forney syndromes -> Berlekamp-Massey with erase_count ->
 * brute-force Chien -> rs_correct_errata with the FULL syndrome).  Kept as a
 * separate path from decode_block so the errors-only decoder above, already
 * held byte-identical to reedsolo, is untouched; this one exists because the
 * streaming PLL decoder marks erasures and needs reedsolo's exact behaviour,
 * including its errata count (erasures are counted even when the syndrome is
 * already clean) and its failure conditions.
 *
 * Polynomials here are (array, length) pairs, highest degree first, exactly
 * like the Python lists, so the Python index arithmetic (including negative
 * indices) can be mirrored without re-deriving it.
 * ======================================================================== */

static unsigned char ra_pow(const rs255 *r, unsigned char x, int power)
{
    int e;
    if (x == 0) return 0;                 /* reedsolo: gf_exp[(log*p)%255], log[0] unused */
    e = ((int)r->log[x] * power) % 255;
    if (e < 0) e += 255;                   /* Python % is non-negative */
    return r->exp[e];
}

static unsigned char ra_eval(const rs255 *r, const unsigned char *p, int n,
                             unsigned char x)
{
    unsigned char y;
    int i;
    if (n <= 0) return 0;
    y = p[0];
    for (i = 1; i < n; i++) y = (unsigned char)(gmul(r, y, x) ^ p[i]);
    return y;
}

/* out = p + q, right-aligned (gf_poly_add); returns length. out may alias p. */
static int ra_add(const unsigned char *p, int np, const unsigned char *q,
                  int nq, unsigned char *out)
{
    int n = np > nq ? np : nq, i;
    unsigned char t[RS_ERA_ROW];
    for (i = 0; i < n; i++) t[i] = 0;
    for (i = 0; i < np; i++) t[n - np + i] = p[i];
    for (i = 0; i < nq; i++) t[i + n - nq] ^= q[i];
    for (i = 0; i < n; i++) out[i] = t[i];
    return n;
}

static int ra_mul(const rs255 *r, const unsigned char *p, int np,
                  const unsigned char *q, int nq, unsigned char *out)
{
    int n = np + nq - 1, i, j;
    unsigned char t[RS_ERA_ROW];
    for (i = 0; i < n; i++) t[i] = 0;
    for (j = 0; j < nq; j++)
        for (i = 0; i < np; i++)
            t[i + j] ^= gmul(r, p[i], q[j]);
    for (i = 0; i < n; i++) out[i] = t[i];
    return n;
}

static void ra_scale(const rs255 *r, const unsigned char *p, int n,
                     unsigned char x, unsigned char *out)
{
    int i;
    for (i = 0; i < n; i++) out[i] = gmul(r, p[i], x);
}

static int ra_syndromes_zero(const rs255 *r, const unsigned char *msg, int len,
                             unsigned char *synd /* nsym+1 */)
{
    int i, zero = 1;
    synd[0] = 0;
    for (i = 0; i < r->nsym; i++) {
        synd[i + 1] = ra_eval(r, msg, len, ra_pow(r, (unsigned char)RS_GEN, i + RS_FCR));
        if (synd[i + 1]) zero = 0;
    }
    return zero;
}

/* One block. epos: erasure indices within the block. Returns errata count
 * (len(erase_pos + err_pos)) or -1 for ReedSolomonError. */
static int decode_block_era(rs255 *r, unsigned char *msg, int len,
                            const int *epos, int ne)
{
    const int nsym = r->nsym;
    unsigned char *synd  = r->era + 0 * RS_ERA_ROW;
    unsigned char *fsynd = r->era + 1 * RS_ERA_ROW;
    unsigned char *eloc  = r->era + 2 * RS_ERA_ROW;
    unsigned char *oloc  = r->era + 3 * RS_ERA_ROW;
    unsigned char *tmp   = r->era + 4 * RS_ERA_ROW;
    unsigned char *tmp2  = r->era + 5 * RS_ERA_ROW;
    unsigned char *E     = r->era + 6 * RS_ERA_ROW;
    unsigned char *evl   = r->era + 7 * RS_ERA_ROW;
    int errpos[RS_ERA_ROW], coef[RS_ERA_ROW];
    unsigned char X[RS_ERA_ROW];
    int nel, nol, i, j, nerr = 0, nall, nfs;

    if (len > 255) return -1;
    for (i = 0; i < ne; i++) {
        if (epos[i] < 0 || epos[i] >= len) return -1;   /* Python IndexError */
        msg[epos[i]] = 0;
    }
    if (ne > nsym) return -1;                      /* "Too many erasures" */
    if (ra_syndromes_zero(r, msg, len, synd)) return ne;

    /* --- rs_forney_syndromes */
    nfs = nsym;
    for (i = 0; i < nfs; i++) fsynd[i] = synd[i + 1];
    for (i = 0; i < ne; i++) {
        unsigned char x = ra_pow(r, (unsigned char)RS_GEN, len - 1 - epos[i]);
        for (j = 0; j < nfs - 1; j++)
            fsynd[j] = (unsigned char)(gmul(r, fsynd[j], x) ^ fsynd[j + 1]);
    }

    /* --- rs_find_error_locator(fsynd, nsym, erase_count=ne) */
    eloc[0] = 1; nel = 1;
    oloc[0] = 1; nol = 1;
    for (i = 0; i < nsym - ne; i++) {
        int K = i;                                  /* synd_shift = 0 */
        unsigned char delta = fsynd[K];
        for (j = 1; j < nel; j++) {
            int si = K - j;
            if (si < 0) si += nfs;                  /* Python negative index */
            delta ^= gmul(r, eloc[nel - (j + 1)], fsynd[si]);
        }
        oloc[nol++] = 0;                            /* old_loc + [0] */
        if (delta != 0) {
            if (nol > nel) {
                int nn = nol;
                ra_scale(r, oloc, nol, delta, tmp);              /* new_loc */
                ra_scale(r, eloc, nel, ginv(r, delta), oloc);    /* old_loc */
                nol = nel;
                for (j = 0; j < nn; j++) eloc[j] = tmp[j];
                nel = nn;
            }
            ra_scale(r, oloc, nol, delta, tmp2);
            nel = ra_add(eloc, nel, tmp2, nol, eloc);
        }
    }
    /* drop leading zeros */
    i = 0;
    while (i < nel && eloc[i] == 0) i++;
    if (i) { for (j = 0; j < nel - i; j++) eloc[j] = eloc[j + i]; nel -= i; }
    {
        int errs = nel - 1;
        if ((errs - ne) * 2 + ne > nsym) return -1;  /* "Too many errors" */
    }

    /* --- rs_find_errors(err_loc[::-1], len) */
    for (i = 0; i < nel; i++) tmp[i] = eloc[nel - 1 - i];
    for (i = 0; i < len; i++) {
        if (ra_eval(r, tmp, nel, ra_pow(r, (unsigned char)RS_GEN, i)) == 0) {
            if (nerr >= RS_ERA_ROW) return -1;
            errpos[nerr++] = len - 1 - i;
        }
    }
    if (nerr != nel - 1) return -1;

    /* --- rs_correct_errata(msg, synd, erase_pos + err_pos) */
    nall = ne + nerr;
    if (nall > RS_ERA_ROW / 2) return -1;
    for (i = 0; i < ne; i++)   coef[i] = len - 1 - epos[i];
    for (i = 0; i < nerr; i++) coef[ne + i] = len - 1 - errpos[i];
    /* errata locator: prod (1 + x*alpha^c), as [alpha^c, 1] factors */
    eloc[0] = 1; nel = 1;
    for (i = 0; i < nall; i++) {
        unsigned char f[2];
        f[0] = ra_pow(r, (unsigned char)RS_GEN, coef[i]); f[1] = 1;
        nel = ra_mul(r, eloc, nel, f, 2, eloc);
    }
    /* evaluator: remainder of (synd[::-1] * err_loc) / x^(L+1) = last L+1
     * coefficients of the product (the divisor has no other terms), then
     * reversed. */
    {
        int ns = nsym + 1, np, L = nel - 1;
        for (i = 0; i < ns; i++) tmp[i] = synd[ns - 1 - i];
        np = ra_mul(r, tmp, ns, eloc, nel, tmp2);
        for (i = 0; i < L + 1; i++) evl[i] = tmp2[np - (L + 1) + i];
        /* [::-1] once for err_eval, and gf_poly_eval(err_eval[::-1]) undoes
         * it: evaluate the remainder in its original order. */
    }
    for (i = 0; i < nall; i++) {
        int l = 255 - coef[i];
        X[i] = ra_pow(r, (unsigned char)RS_GEN, -l);
    }
    for (i = 0; i < len; i++) E[i] = 0;
    {
        int L1 = nel;      /* evaluator length = L + 1 = nel */
        for (i = 0; i < nall; i++) {
            unsigned char Xi_inv = ginv(r, X[i]);
            unsigned char prime = 1, y;
            for (j = 0; j < nall; j++) {
                if (j != i)
                    prime = gmul(r, prime, (unsigned char)(1 ^ gmul(r, Xi_inv, X[j])));
            }
            if (prime == 0) return -1;
            y = ra_eval(r, evl, L1, Xi_inv);
            y = gmul(r, ra_pow(r, X[i], 1 - RS_FCR), y);
            {
                int pos = i < ne ? epos[i] : errpos[i - ne];
                E[pos] = gdiv(r, y, prime);
            }
        }
    }
    for (i = 0; i < len; i++) msg[i] ^= E[i];
    if (!ra_syndromes_zero(r, msg, len, synd)) return -1;
    return nall;
}

int rs255_decode_era(rs255 *r, const unsigned char *buf, int n,
                     const int *erase_pos, int n_erase,
                     unsigned char *out, int out_cap, int *errata)
{
    int off = 0, wr = 0, total = 0, blkno = 0;
    int ep[RS_ERA_ROW];
    if (!r || !buf || !out || !errata || n < 0 || n_erase < 0) return -1;
    if (n_erase > 0 && !erase_pos) return -1;
    *errata = 0;
    while (off < n) {
        int blen = n - off, dlen, ne = 0, i, got;
        if (blen > RS_NSIZE) blen = RS_NSIZE;
        if (blen <= r->nsym) return -1;
        dlen = blen - r->nsym;
        /* reedsolo: e_pos = [x for x in erase_pos if x < nsize], then shifts
         * the remainder down by nsize for the next chunk. */
        for (i = 0; i < n_erase; i++) {
            int x = erase_pos[i] - blkno * RS_NSIZE;
            if (x >= 0 && x < RS_NSIZE) {
                if (ne >= RS_ERA_ROW) return -3;
                ep[ne++] = x;
            }
        }
        for (i = 0; i < blen; i++) r->blk[i] = buf[off + i];
        got = decode_block_era(r, r->blk, blen, ep, ne);
        if (got < 0) return -3;
        total += got;
        if (wr + dlen > out_cap) return -2;
        for (i = 0; i < dlen; i++) out[wr + i] = r->blk[i];
        wr += dlen; off += blen; blkno++;
    }
    *errata = total;
    return wr;
}
