/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026 XP5.ORG
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/* rs255.c -- see rs255.h.
 *
 * A near-mechanical transliteration of the pure-Python `reedsolo` package's
 * encode/decode chain (rs_encode_msg, rs_calc_syndromes, rs_find_error_locator
 * [Berlekamp-Massey], rs_find_errors [Chien search by brute force],
 * rs_correct_errata [Forney]), specialised to the no-erasures case, which is
 * the only one ofdm_frame.c ever calls. Deliberately literal rather than
 * "cleaned up": several steps reverse a coefficient array and hand it to a
 * helper that reverses ITS OWN input again, which looks redundant read cold
 * but is exactly what the reference does, and untangling it by "simplifying"
 * is how a transliteration grows a sign/order bug that only shows up on some
 * error patterns. Trust the mechanical mapping; the correctness evidence is
 * the byte-for-byte cross-check against actual reedsolo output, not a read
 * of this file.
 *
 * All arithmetic is GF(2^8): table lookups and XOR only, no floating point.
 */
#include "rs255.h"
#include <string.h>

struct rs255 {
    int nsym;
    unsigned char gflog[256];
    unsigned char gfexp[512];              /* doubled: gf_mul never needs %255 */
    unsigned char gen[RS255_MAX_NSYM + 1]; /* generator poly, highest-degree first, gen[0]==1 */
};

static unsigned char gf_mul(const rs255 *rs, unsigned char a, unsigned char b)
{
    if (a == 0 || b == 0) return 0;
    return rs->gfexp[(int)rs->gflog[a] + (int)rs->gflog[b]];
}

/* generator is always 2 in this port -- fcr=0, c_exp=8, matching reedsolo's
 * defaults exactly (see rs255.h). n may be negative (the Forney step needs
 * alpha^-l), so reduce through a wide signed accumulator before indexing. */
static unsigned char gf_pow(const rs255 *rs, unsigned char a, long n)
{
    long e = ((long)rs->gflog[a] * n) % 255L;
    if (e < 0) e += 255L;
    return rs->gfexp[(int)e];
}

static unsigned char gf_inverse(const rs255 *rs, unsigned char a)
{
    int idx = 255 - (int)rs->gflog[a];
    if (idx == 255) idx = 0;
    return rs->gfexp[idx];
}

static unsigned char gf_div(const rs255 *rs, unsigned char a, unsigned char b)
{
    if (a == 0) return 0;
    return gf_mul(rs, a, gf_inverse(rs, b));
}

/* Horner's method. poly[0] is the highest-degree coefficient. */
static unsigned char gf_poly_eval(const rs255 *rs, const unsigned char *poly,
                                  int len, unsigned char x)
{
    unsigned char y = poly[0];
    int i;
    for (i = 1; i < len; i++) y = gf_mul(rs, y, x) ^ poly[i];
    return y;
}

/* r[Lp+Lq-1] = p*q, both highest-degree-first. */
static int gf_poly_mul(const rs255 *rs, const unsigned char *p, int Lp,
                       const unsigned char *q, int Lq, unsigned char *r)
{
    int i, j, Lr = Lp + Lq - 1;
    memset(r, 0, (size_t)Lr);
    for (j = 0; j < Lq; j++) {
        if (!q[j]) continue;
        for (i = 0; i < Lp; i++)
            if (p[i]) r[i + j] ^= gf_mul(rs, p[i], q[j]);
    }
    return Lr;
}

/* Extended synthetic division, exactly as gf_poly_div: does not normalise by
 * divisor[0] (the reference doesn't either -- every divisor it's called with
 * here is monic by construction). remainder gets Lv-1 bytes. */
static void gf_poly_div(const rs255 *rs, const unsigned char *dividend, int Ld,
                        const unsigned char *divisor, int Lv,
                        unsigned char *remainder)
{
    unsigned char work[2 * RS255_MAX_NSYM + 4];
    int i, j;
    memcpy(work, dividend, (size_t)Ld);
    for (i = 0; i < Ld - (Lv - 1); i++) {
        unsigned char coef = work[i];
        if (coef) {
            for (j = 1; j < Lv; j++)
                if (divisor[j]) work[i + j] ^= gf_mul(rs, divisor[j], coef);
        }
    }
    memcpy(remainder, work + (Ld - (Lv - 1)), (size_t)(Lv - 1));
}

size_t rs255_workspace(int nsym)
{
    if (nsym < 1 || nsym > RS255_MAX_NSYM) return 0;
    return sizeof(struct rs255);
}

int rs255_init(rs255 **rsp, void *mem, size_t memlen, int nsym)
{
    struct rs255 *rs;
    int i, j;
    unsigned x;

    if (!rsp || !mem) return -1;
    if (memlen < sizeof(struct rs255)) return -2;
    if (nsym < 1 || nsym > RS255_MAX_NSYM) return -3;

    rs = (struct rs255 *)mem;
    memset(rs, 0, sizeof(*rs));
    rs->nsym = nsym;

    /* GF(2^8) tables: prim=0x11d, generator=2 -- doubling with reduction IS
     * multiplication by 2 under this primitive polynomial. */
    x = 1;
    for (i = 0; i < 255; i++) {
        rs->gfexp[i] = (unsigned char)x;
        rs->gflog[x] = (unsigned char)i;
        x <<= 1;
        if (x & 0x100) x ^= 0x11d;
    }
    rs->gflog[0] = 0; /* undefined; never read (gf_mul/gf_pow guard on 0) */
    for (i = 255; i < 512; i++) rs->gfexp[i] = rs->gfexp[i - 255];

    /* Generator polynomial: g(x) = product_{i=0}^{nsym-1} (x + alpha^i),
     * fcr=0 so the roots are alpha^0..alpha^(nsym-1). Built the same way
     * rs_generator_poly does: g starts as [1], multiplied by [1, root] once
     * per root (see gf_poly_mul's derivation in the header comment above). */
    rs->gen[0] = 1;
    {
        int len = 1;
        for (i = 0; i < nsym; i++) {
            unsigned char root = gf_pow(rs, 2, i);
            unsigned char newg[RS255_MAX_NSYM + 1];
            memset(newg, 0, (size_t)(len + 1));
            for (j = 0; j < len; j++) {
                newg[j]     ^= rs->gen[j];
                newg[j + 1] ^= gf_mul(rs, rs->gen[j], root);
            }
            memcpy(rs->gen, newg, (size_t)(len + 1));
            len++;
        }
    }

    *rsp = rs;
    return 0;
}

int rs255_encode(rs255 *rs, const unsigned char *msg, int msg_len,
                 unsigned char *out, int out_cap)
{
    unsigned char keep[255];
    unsigned char lgen[RS255_MAX_NSYM + 1];
    int nsym, i, j;

    if (!rs || !msg || !out) return -1;
    nsym = rs->nsym;
    if (msg_len < 0 || msg_len + nsym > 255) return -1;
    if (out_cap < msg_len + nsym) return -2;

    memcpy(keep, msg, (size_t)msg_len);          /* survives in-place aliasing */
    if (out != msg) memmove(out, msg, (size_t)msg_len);
    memset(out + msg_len, 0, (size_t)nsym);

    for (i = 0; i <= nsym; i++) lgen[i] = rs->gflog[rs->gen[i]];

    for (i = 0; i < msg_len; i++) {
        unsigned char coef = out[i];
        if (coef) {
            unsigned char lcoef = rs->gflog[coef];
            for (j = 1; j <= nsym; j++)
                out[i + j] ^= rs->gfexp[(int)lcoef + (int)lgen[j]];
        }
    }
    memcpy(out, keep, (size_t)msg_len);          /* loop scribbled over this region */
    return msg_len + nsym;
}

int rs255_decode(rs255 *rs, const unsigned char *in, int in_len,
                 unsigned char *out, int out_cap, int *errata)
{
    int nsym, i, j, msg_len, errs;
    unsigned char msg[255];
    unsigned char synd[RS255_MAX_NSYM];
    unsigned char err_loc[RS255_MAX_NSYM + 1], old_loc[RS255_MAX_NSYM + 1];
    int err_loc_len, old_loc_len;
    unsigned char err_loc_rev[RS255_MAX_NSYM + 1];
    int err_pos[RS255_MAX_NSYM], n_err_pos;
    unsigned char e_loc[RS255_MAX_NSYM + 1];
    int e_loc_len;

    if (!rs || !in || !out) return -1;
    nsym = rs->nsym;
    if (in_len <= nsym || in_len > 255) return -1;
    msg_len = in_len - nsym;
    if (out_cap < msg_len) return -2;

    memcpy(msg, in, (size_t)in_len);

    /* Syndromes S[i] = msg(alpha^i), i=0..nsym-1 -- this IS reedsolo's
     * synd[1:] (its own synd[0] is always the prepended 0, so working with
     * the un-prepended array throughout is equivalent and simpler). */
    for (i = 0; i < nsym; i++)
        synd[i] = gf_poly_eval(rs, msg, in_len, gf_pow(rs, 2, i));

    {
        int any = 0;
        for (i = 0; i < nsym; i++) if (synd[i]) { any = 1; break; }
        if (!any) {
            memcpy(out, msg, (size_t)msg_len);
            if (errata) *errata = 0;
            return msg_len;
        }
    }

    /* Berlekamp-Massey, no erasures: synd_shift=0, K=i. */
    err_loc[0] = 1; err_loc_len = 1;
    old_loc[0] = 1; old_loc_len = 1;
    for (i = 0; i < nsym; i++) {
        unsigned char delta = synd[i];
        unsigned char scaled[RS255_MAX_NSYM + 1], result[RS255_MAX_NSYM + 1];
        int scaled_len, result_len;

        for (j = 1; j < err_loc_len; j++)
            delta ^= gf_mul(rs, err_loc[err_loc_len - 1 - j], synd[i - j]);

        old_loc[old_loc_len] = 0; old_loc_len++;   /* "old_loc = old_loc + [0]" */

        if (delta != 0) {
            if (old_loc_len > err_loc_len) {
                int saved_old_len = old_loc_len, t;
                unsigned char new_loc[RS255_MAX_NSYM + 1];
                for (t = 0; t < old_loc_len; t++)
                    new_loc[t] = gf_mul(rs, old_loc[t], delta);
                {
                    unsigned char inv = gf_inverse(rs, delta);
                    for (t = 0; t < err_loc_len; t++)
                        old_loc[t] = gf_mul(rs, err_loc[t], inv);
                    old_loc_len = err_loc_len;
                }
                memcpy(err_loc, new_loc, (size_t)saved_old_len);
                err_loc_len = saved_old_len;
            }
            scaled_len = old_loc_len;
            for (j = 0; j < scaled_len; j++) scaled[j] = gf_mul(rs, old_loc[j], delta);
            result_len = err_loc_len > scaled_len ? err_loc_len : scaled_len;
            memset(result, 0, (size_t)result_len);
            memcpy(result + (result_len - err_loc_len), err_loc, (size_t)err_loc_len);
            for (j = 0; j < scaled_len; j++)
                result[result_len - scaled_len + j] ^= scaled[j];
            memcpy(err_loc, result, (size_t)result_len);
            err_loc_len = result_len;
        }
    }

    {
        int start = 0;
        while (start < err_loc_len - 1 && err_loc[start] == 0) start++;
        if (start) {
            memmove(err_loc, err_loc + start, (size_t)(err_loc_len - start));
            err_loc_len -= start;
        }
    }
    errs = err_loc_len - 1;
    if (errs * 2 > nsym) return -3;

    /* Chien search over err_loc REVERSED -- see file header note. */
    for (i = 0; i < err_loc_len; i++) err_loc_rev[i] = err_loc[err_loc_len - 1 - i];
    n_err_pos = 0;
    for (i = 0; i < in_len; i++) {
        if (gf_poly_eval(rs, err_loc_rev, err_loc_len, gf_pow(rs, 2, i)) == 0) {
            if (n_err_pos < RS255_MAX_NSYM) err_pos[n_err_pos] = in_len - 1 - i;
            n_err_pos++;
        }
    }
    if (n_err_pos != errs) return -3;

    /* Errata locator e_loc = product of (root_i*x + 1), root_i = alpha^coef_pos_i. */
    e_loc[0] = 1; e_loc_len = 1;
    for (i = 0; i < errs; i++) {
        int coef_pos = in_len - 1 - err_pos[i];
        unsigned char root = gf_pow(rs, 2, coef_pos);
        unsigned char tmp[RS255_MAX_NSYM + 2];
        int t, tl = e_loc_len + 1;
        memset(tmp, 0, (size_t)tl);
        for (t = 0; t < e_loc_len; t++) {
            tmp[t]     ^= gf_mul(rs, e_loc[t], root);
            tmp[t + 1] ^= e_loc[t];
        }
        memcpy(e_loc, tmp, (size_t)tl);
        e_loc_len = tl;
    }

    /* Error evaluator: Omega(x) = [Synd_rev(x) * e_loc(x)] mod x^(errs+1),
     * then reversed again -- both reversals are the reference's, not ours.
     * Synd_rev is the reverse of the FULL reference syndrome array, which
     * has a leading 0 prepended (see the note by the `synd[]` computation
     * above) -- so it's nsym+1 long, [S[nsym-1],...,S[0], 0], not just the
     * nsym-long reverse of our own unprepended synd[]. Dropping that
     * trailing 0 shifts and effectively reverses this whole computation,
     * which is exactly the bug the cross-check against reedsolo caught. */
    {
        unsigned char synd_rev[RS255_MAX_NSYM + 1];
        unsigned char mulbuf[2 * RS255_MAX_NSYM + 2];
        unsigned char divisor[RS255_MAX_NSYM + 2];
        unsigned char eval_raw[RS255_MAX_NSYM + 1];
        unsigned char err_eval[RS255_MAX_NSYM + 1];
        int mlen, dvlen, eeh_len, t;
        unsigned char E[255];

        for (t = 0; t < nsym; t++) synd_rev[t] = synd[nsym - 1 - t];
        synd_rev[nsym] = 0;
        mlen = gf_poly_mul(rs, synd_rev, nsym + 1, e_loc, e_loc_len, mulbuf);
        dvlen = errs + 2;
        memset(divisor, 0, (size_t)dvlen);
        divisor[0] = 1;
        gf_poly_div(rs, mulbuf, mlen, divisor, dvlen, eval_raw);
        eeh_len = dvlen - 1;
        for (t = 0; t < eeh_len; t++) err_eval[t] = eval_raw[eeh_len - 1 - t];

        memset(E, 0, (size_t)in_len);
        for (i = 0; i < errs; i++) {
            int coef_pos_i = in_len - 1 - err_pos[i];
            unsigned char Xi = gf_pow(rs, 2, coef_pos_i);
            unsigned char Xi_inv = gf_inverse(rs, Xi);
            unsigned char err_loc_prime = 1;
            int j2;
            unsigned char y, magnitude;
            unsigned char eeh_rev[RS255_MAX_NSYM + 1];

            for (j2 = 0; j2 < errs; j2++) {
                int coef_pos_j;
                unsigned char Xj, term;
                if (j2 == i) continue;
                coef_pos_j = in_len - 1 - err_pos[j2];
                Xj = gf_pow(rs, 2, coef_pos_j);
                term = (unsigned char)(1 ^ gf_mul(rs, Xi_inv, Xj));
                err_loc_prime = gf_mul(rs, err_loc_prime, term);
            }
            if (err_loc_prime == 0) return -3;

            for (t = 0; t < eeh_len; t++) eeh_rev[t] = err_eval[eeh_len - 1 - t];
            y = gf_poly_eval(rs, eeh_rev, eeh_len, Xi_inv);
            y = gf_mul(rs, gf_pow(rs, Xi, 1), y);   /* fcr=0 => Xi^(1-fcr) = Xi */
            magnitude = gf_div(rs, y, err_loc_prime);
            E[err_pos[i]] = magnitude;
        }
        for (i = 0; i < in_len; i++) msg[i] ^= E[i];
    }

    /* Verify: corrected codeword must have all-zero syndromes. */
    for (i = 0; i < nsym; i++)
        if (gf_poly_eval(rs, msg, in_len, gf_pow(rs, 2, i)) != 0) return -3;

    memcpy(out, msg, (size_t)msg_len);
    if (errata) *errata = errs;
    return msg_len;
}
