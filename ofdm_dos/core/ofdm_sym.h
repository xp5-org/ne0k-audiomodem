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

/* ofdm_sym.h — occupied carriers -> one real time-domain symbol, C89.
 *
 * Its own unit because BOTH the modulator and the channel probe build symbols,
 * and there must be exactly one definition of what a symbol looks like on the
 * wire.  ofdm_mod_regress already proves this arithmetic against the Python
 * sample for sample; the probe inherits that proof by using the same code
 * rather than a second copy of it.
 */
#ifndef OFDM_SYM_H
#define OFDM_SYM_H
#include <stddef.h>
#include "ofdm_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ofdm_sym ofdm_sym;

size_t ofdm_sym_workspace(const ofdm_cfg *c);

/* Errors: -1 bad args, -2 memlen too small, -3 FFT init failed. */
int ofdm_sym_init(ofdm_sym **s, void *mem, size_t memlen, const ofdm_cfg *c);

/* Longest output ofdm_sym_make can write: symbol_samples + 2*taper_samples. */
int ofdm_sym_max_len(const ofdm_sym *s);

/* Build one symbol from n_occupied carriers.  `use_taper` selects the W-OFDM
 * raised-cosine extension; pass 0 for the SC preamble, which the reference
 * builds with _ofdm_td and concatenates ahead of the overlap-added body rather
 * than joining it.  Returns the sample count written, or -1 on bad args. */
int ofdm_sym_make(ofdm_sym *s, const double *occ_re, const double *occ_im,
                  int use_taper, double *out);

#ifdef __cplusplus
}
#endif
#endif
