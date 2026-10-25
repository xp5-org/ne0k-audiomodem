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
};

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
         + al8(255);                      /* blk   */
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
    r->blk   = p;

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
