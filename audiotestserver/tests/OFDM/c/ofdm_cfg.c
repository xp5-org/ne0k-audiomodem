/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

#include "ofdm_cfg.h"
#include <math.h>

/* ---------------------------------------------------------------- geometry */

static int is_pow2(int n) { return n >= 4 && (n & (n - 1)) == 0; }

int ofdm_cfg_build(ofdm_cfg *c, int n_fft, double cp_ratio,
                   double freq_lo, double freq_hi, int pilot_spacing,
                   int sample_rate, int ecc_bytes, int chunk_bytes,
                   int seq_bytes, double taper_ms, double frame_gap_ms,
                   int sync_period)
{
    double sc_spc;
    int    cp_len, bin_lo, bin_hi, n_occ, n_pilots, raw_tap, half_cp;

    if (!c) return -3;
    if (!is_pow2(n_fft)) return -1;
    if (sample_rate <= 0 || pilot_spacing < 1) return -3;
    if (ecc_bytes < 1 || chunk_bytes < 1 || seq_bytes < 1) return -3;
    if (sync_period < 0 || cp_ratio <= 0.0) return -3;

    cp_len = (int)(n_fft * cp_ratio + 0.5);
    if (cp_len < 1) cp_len = 1;
    sc_spc = (double)sample_rate / (double)n_fft;

    /* bin_lo = max(1, ceil(freq_lo/spacing)); bin_hi = min(N/2-1, floor(...)).
     * DC and Nyquist are excluded so the half spectrum handed to the inverse
     * transform is unambiguously Hermitian -- see _ofdm_td. */
    bin_lo = (int)ceil(freq_lo / sc_spc);
    if (bin_lo < 1) bin_lo = 1;
    bin_hi = (int)floor(freq_hi / sc_spc);
    if (bin_hi > n_fft / 2 - 1) bin_hi = n_fft / 2 - 1;
    if (bin_lo > bin_hi) return -2;

    n_occ = bin_hi - bin_lo + 1;
    /* Pilots sit at offsets 0, spacing, 2*spacing, ... within the occupied
     * set, so the count is a ceiling divide, not a floor. */
    n_pilots = (n_occ + pilot_spacing - 1) / pilot_spacing;

    /* The taper is clamped to cp_len/2.  It is subtracted from the usable
     * guard by _build_ola's overlap (see isi_ratio's taper_samples note), so
     * this clamp is what stops a long taper from eating the whole prefix. */
    raw_tap = (int)(taper_ms * sample_rate / 1000.0 + 0.5);
    half_cp = cp_len / 2;
    c->taper_samples = raw_tap < half_cp ? raw_tap : half_cp;
    if (c->taper_samples < 0) c->taper_samples = 0;

    c->n_fft             = n_fft;
    c->cp_len            = cp_len;
    c->bin_lo            = bin_lo;
    c->bin_hi            = bin_hi;
    c->n_occupied        = n_occ;
    c->pilot_spacing     = pilot_spacing;
    c->n_pilots          = n_pilots;
    c->n_data_carriers   = n_occ - n_pilots;
    c->sample_rate       = sample_rate;
    c->symbol_samples    = n_fft + cp_len;
    c->frame_gap_samples = (int)(frame_gap_ms * sample_rate / 1000.0 + 0.5);
    c->ecc_bytes         = ecc_bytes;
    c->chunk_bytes       = chunk_bytes;
    c->seq_bytes         = seq_bytes;
    c->sync_period       = sync_period;
    c->cp_ratio          = cp_ratio;
    c->sc_spacing_hz     = sc_spc;
    c->freq_lo_hz        = bin_lo * sc_spc;
    c->freq_hi_hz        = bin_hi * sc_spc;
    return 0;
}

/* -------------------------------------------------------------------- PRBS */

void ofdm_prbs(unsigned char *out, int n, unsigned seed)
{
    unsigned state = seed & 0xFFFFu;
    int i;
    if (!out) return;
    for (i = 0; i < n; i++) {
        unsigned fb = ((state >> 15) ^ (state >> 14)
                     ^ (state >> 12) ^ (state >> 3)) & 1u;
        out[i] = (unsigned char)((state >> 15) & 1u);
        state  = ((state << 1) | fb) & 0xFFFFu;
    }
}

void ofdm_known_bpsk(double *re, int n, unsigned seed, double amp)
{
    unsigned state = seed & 0xFFFFu;
    int i;
    if (!re) return;
    /* Inlined rather than calling ofdm_prbs into a scratch buffer: the caller
     * would otherwise have to find n bytes of workspace for a value used once. */
    for (i = 0; i < n; i++) {
        unsigned fb = ((state >> 15) ^ (state >> 14)
                     ^ (state >> 12) ^ (state >> 3)) & 1u;
        unsigned b  = (state >> 15) & 1u;
        state = ((state << 1) | fb) & 0xFFFFu;
        re[i] = b ? -amp : amp;
    }
}

/* -------------------------------------------------------- frame arithmetic */

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

int ofdm_interleave_geometry(const ofdm_cfg *c, int bits_per_ofdm_sym,
                             int *depth, int *k, int *wire_len)
{
    int msg, t, sym_bytes, d, k_max, kk;

    if (!c || bits_per_ofdm_sym < 1) return -1;
    msg = c->seq_bytes + c->chunk_bytes;
    if (msg < 1) return -1;

    /* depth >= sym_bytes/t so one fully-lost OFDM symbol cannot exceed any
     * codeword's budget, +1 byte of slack for a run straddling a column
     * boundary (which touches one codeword twice). */
    t = c->ecc_bytes / 2;
    if (t < 1) t = 1;
    sym_bytes = ceil_div(bits_per_ofdm_sym, 8);
    d = ceil_div(sym_bytes + 1, t);
    if (d < 1) d = 1;

    /* Every codeword must still hold its share of the message inside RS(255). */
    k_max = 255 - c->ecc_bytes;
    if (k_max < 1) k_max = 1;
    if (d < ceil_div(msg, k_max)) d = ceil_div(msg, k_max);

    /* More codewords than message bytes would mean empty blocks. */
    if (d > msg) d = msg;
    if (d < 1) d = 1;

    kk = ceil_div(msg, d);
    if (depth)    *depth    = d;
    if (k)        *k        = kk;
    if (wire_len) *wire_len = d * (kk + c->ecc_bytes);
    return 0;
}

int ofdm_data_syms_per_pkt(const ofdm_cfg *c, int bits_per_ofdm_sym)
{
    int wire;
    if (!c || bits_per_ofdm_sym < 1) return -1;
    if (ofdm_interleave_geometry(c, bits_per_ofdm_sym, 0, 0, &wire) != 0)
        return -1;
    return ceil_div(wire * 8, bits_per_ofdm_sym);
}

int ofdm_n_syncs(const ofdm_cfg *c, int data_syms)
{
    if (!c || c->sync_period <= 0 || data_syms <= 0) return 0;
    return ceil_div(data_syms, c->sync_period);
}

int ofdm_frame_min_gap(const ofdm_cfg *c, int n_body_syms)
{
    long ola, g;
    if (!c) return 1;
    ola = (long)n_body_syms * c->symbol_samples + 2L * c->taper_samples;
    /* cp_len is subtracted as a jitter allowance: _find_sc_peaks tests a
     * candidate by its threshold EDGE but records the accepted peak by its
     * REFINED position, and that offset scales with the metric plateau, i.e.
     * with cp_len, while the margin over the true frame stride is the fixed
     * frame_gap_samples.  Without it every second frame is discarded once
     * cp_len exceeds that margin (n_fft >= 4096 at the 20 ms default). */
    g = (long)c->symbol_samples + ola - c->cp_len;
    return g < 1 ? 1 : (int)g;
}

/* ---------------------------------------------- constellations (wire format)
 *
 * Transcribed from ofdm_modem's _MOD_TABLES.  Order matters: the bit patterns
 * are positionally paired with the points, and the demapper's tie-break is
 * lowest index, so a reordering is a wire change even though the point SET is
 * unchanged.  const, so nothing here is writable-static.
 */
#define R2  1.4142135623730951
#define R10 3.1622776601683795
#define R42 6.4807406984078604

static const double bpsk_re[2] = { 1.0, -1.0 };
static const double bpsk_im[2] = { 0.0,  0.0 };
static const unsigned char bpsk_bits[2] = { 0, 1 };

static const double qpsk_re[4] = {  1.0 / R2, -1.0 / R2, -1.0 / R2,  1.0 / R2 };
static const double qpsk_im[4] = {  1.0 / R2,  1.0 / R2, -1.0 / R2, -1.0 / R2 };
static const unsigned char qpsk_bits[8] = { 0,0,  0,1,  1,1,  1,0 };

/* 16-QAM, in _QAM16_TABLE's insertion order, scaled by 1/sqrt(10). */
static const double q16_re[16] = {
    -3/R10, -3/R10, -3/R10, -3/R10, -1/R10, -1/R10, -1/R10, -1/R10,
     1/R10,  1/R10,  1/R10,  1/R10,  3/R10,  3/R10,  3/R10,  3/R10
};
static const double q16_im[16] = {
    -3/R10, -1/R10,  1/R10,  3/R10,  3/R10,  1/R10, -1/R10, -3/R10,
    -3/R10, -1/R10,  1/R10,  3/R10,  3/R10,  1/R10, -1/R10, -3/R10
};
static const unsigned char q16_bits[64] = {
    0,0,0,0, 0,0,0,1, 0,0,1,1, 0,0,1,0,
    0,1,1,0, 0,1,1,1, 0,1,0,1, 0,1,0,0,
    1,1,0,0, 1,1,0,1, 1,1,1,1, 1,1,1,0,
    1,0,1,0, 1,0,1,1, 1,0,0,1, 1,0,0,0
};

/* 64-QAM is the outer product of a 3-bit Gray code with itself: I varies
 * slowest, Q fastest, matching the dict comprehension in ofdm_modem. Built
 * at first use rather than written out as 64 lines of literals. The buffers
 * are function-static but written once with the same values every time, so
 * this stays effectively const; a target that forbids lazy init can call
 * ofdm_mod_table(6, ...) once at startup. */
static const int    g3_lvl[8]     = { -7, -5, -3, -1, 1, 3, 5, 7 };
static const unsigned char g3_bits[24] = {
    0,0,0,  0,0,1,  0,1,1,  0,1,0,  1,1,0,  1,1,1,  1,0,1,  1,0,0
};
static double q64_re[64];
static double q64_im[64];
static unsigned char q64_bits[384];
static int q64_ready = 0;

static void q64_build(void)
{
    int i, q, n = 0, b = 0, t;
    for (i = 0; i < 8; i++) {
        for (q = 0; q < 8; q++) {
            q64_re[n] = g3_lvl[i] / R42;
            q64_im[n] = g3_lvl[q] / R42;
            n++;
            for (t = 0; t < 3; t++) q64_bits[b++] = g3_bits[i * 3 + t];
            for (t = 0; t < 3; t++) q64_bits[b++] = g3_bits[q * 3 + t];
        }
    }
    q64_ready = 1;
}

int ofdm_mod_table(int bits, const double **pts_re, const double **pts_im,
                   const unsigned char **bits_tab, int *n_pts)
{
    const double *re = 0, *im = 0;
    const unsigned char *bt = 0;
    int n = 0;

    switch (bits) {
    case 1: re = bpsk_re; im = bpsk_im; bt = bpsk_bits; n = 2;  break;
    case 2: re = qpsk_re; im = qpsk_im; bt = qpsk_bits; n = 4;  break;
    case 4: re = q16_re;  im = q16_im;  bt = q16_bits;  n = 16; break;
    case 6:
        if (!q64_ready) q64_build();
        re = q64_re; im = q64_im; bt = q64_bits; n = 64;
        break;
    default: return -1;
    }
    if (pts_re)   *pts_re   = re;
    if (pts_im)   *pts_im   = im;
    if (bits_tab) *bits_tab = bt;
    if (n_pts)    *n_pts    = n;
    return 0;
}

int ofdm_mod_tables_flat(const int *want, int n_want,
                         int *tab_b, int *tab_n,
                         double *pts_re, double *pts_im,
                         unsigned char *tab_bits, int cap_pts, int cap_bits,
                         int *n_pts_total, int *n_bits_total)
{
    int t, i, np = 0, nb = 0, rc;
    const double *re = 0, *im = 0;
    const unsigned char *bt = 0;
    int n = 0;

    if (!want || n_want < 1) return -1;

    /* Size first, so a caller can probe with cap 0 and then allocate. */
    for (t = 0; t < n_want; t++) {
        rc = ofdm_mod_table(want[t], 0, 0, 0, &n);
        if (rc != 0) return -2;
        np += n;
        nb += n * want[t];
    }
    if (n_pts_total)  *n_pts_total  = np;
    if (n_bits_total) *n_bits_total = nb;
    if (cap_pts < np || cap_bits < nb) return -3;
    if (!tab_b || !tab_n || !pts_re || !pts_im || !tab_bits) return -1;

    np = 0; nb = 0;
    for (t = 0; t < n_want; t++) {
        /* Re-checked rather than assumed: the sizing pass above proved the
         * width is valid, but leaving the result unchecked lets a future edit
         * to ofdm_mod_table introduce a null deref that the sizing pass would
         * not catch. */
        if (ofdm_mod_table(want[t], &re, &im, &bt, &n) != 0) return -2;
        tab_b[t] = want[t];
        tab_n[t] = n;
        for (i = 0; i < n; i++) {
            pts_re[np + i] = re[i];
            pts_im[np + i] = im[i];
        }
        for (i = 0; i < n * want[t]; i++) tab_bits[nb + i] = bt[i];
        np += n;
        nb += n * want[t];
    }
    return 0;
}

int ofdm_bat_widths(const int *bits_per_bin, int n, int *out, int cap)
{
    static const int order[4] = { 1, 2, 4, 6 };
    int i, j, cnt = 0;

    if (!bits_per_bin || n < 0 || !out) return -1;
    for (j = 0; j < 4; j++) {
        for (i = 0; i < n; i++) {
            if (bits_per_bin[i] == order[j]) {
                if (cnt >= cap) return -1;
                out[cnt++] = order[j];
                break;
            }
        }
    }
    return cnt;
}
