/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_cfg.h — geometry, PRBS, reference sequences and constellation tables.
 *
 * The pieces every other core needs and none of them owned: `build_config`'s
 * derivation, the 16-bit LFSR that generates the scrambler and every known
 * sequence, the interleave geometry, and the Gray-coded constellations.
 *
 * A freestanding build (DOS/OpenWatcom, or an MCU) has no Python process to
 * compute these at init, so the derivation lives here. Values match
 * ofdm_modem.build_config; `tests/regression/ofdm_cfg_regress.py` grades
 * them field by field.
 *
 * Same contract as the rest: C89, no malloc, no stdio, no writable statics.
 * The constellation tables ARE static, but const -- they are wire format,
 * not state.
 */
#ifndef OFDM_CFG_H
#define OFDM_CFG_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* PRBS seeds.  Wire format: both ends must agree, so these are fixed, not
 * configurable.  _SEED_SC is deliberately _SEED_CE ^ 0xFFFF in the reference. */
#define OFDM_SEED_DATA  0xACE1u
#define OFDM_SEED_CE    0x3D5Fu
#define OFDM_SEED_SC    (0x3D5Fu ^ 0xFFFFu)
#define OFDM_SEED_PROBE 0x7A3Fu
#define OFDM_SEED_MEAS  0x51C7u

/* Amplitude of the DATA FRAME's CE/SYNC symbols relative to _known_bpsk's
 * +-1/sqrt(2).  sqrt(2) puts them at unit power, matching the data
 * constellations and the pilot; see REF_SYM_BOOST in ofdm_modem.py for why the
 * probe's sequences are deliberately NOT boosted with them. */
#define OFDM_REF_SYM_BOOST 1.4142135623730951

typedef struct {
    int    n_fft;
    int    cp_len;
    int    bin_lo;
    int    bin_hi;
    int    n_occupied;
    int    pilot_spacing;
    int    n_pilots;
    int    n_data_carriers;
    int    sample_rate;
    int    symbol_samples;
    int    taper_samples;
    int    frame_gap_samples;
    int    ecc_bytes;
    int    chunk_bytes;
    int    seq_bytes;
    int    sync_period;
    double cp_ratio;
    double sc_spacing_hz;
    double freq_lo_hz;
    double freq_hi_hz;
} ofdm_cfg;

/* Mirrors ofdm_modem.build_config for the fields a freestanding modem needs.
 * The probe-length fields are deliberately absent: they only matter to
 * ChannelProbe, which is not part of the fixed-table path.
 *
 * Returns 0 ok, -1 n_fft not a power of two >= 4, -2 no carriers in the band,
 * -3 other bad argument. */
int ofdm_cfg_build(ofdm_cfg *c, int n_fft, double cp_ratio,
                   double freq_lo, double freq_hi, int pilot_spacing,
                   int sample_rate, int ecc_bytes, int chunk_bytes,
                   int seq_bytes, double taper_ms, double frame_gap_ms,
                   int sync_period);

/* 16-bit maximal LFSR, taps 15/14/12/3, MSB out.  `out` receives n bits as
 * 0/1 bytes.  This is the scrambler AND the generator behind every known
 * sequence, so it is the one function whose bit order must not drift. */
void ofdm_prbs(unsigned char *out, int n, unsigned seed);

/* Real BPSK reference sequence: (1 - 2*prbs) * amp.  Imaginary part is zero
 * for every caller, so only the real part is produced.  Pass
 * amp = 1/sqrt(2) for the probe's sequences, or
 * amp = OFDM_REF_SYM_BOOST/sqrt(2) (i.e. 1.0) for a data frame's CE/SYNC. */
void ofdm_known_bpsk(double *re, int n, unsigned seed, double amp);

/* Interleave geometry for one frame body: how many equal RS codewords it
 * splits into, their padded data size, and the wire length.  Mirrors
 * _interleave_geometry / _interleave_depth exactly -- get this wrong and the
 * frame length disagrees between the ends, which shows up as nothing decoding.
 * Returns 0 ok, -1 bad args. */
int ofdm_interleave_geometry(const ofdm_cfg *c, int bits_per_ofdm_sym,
                             int *depth, int *k, int *wire_len);

/* Data symbols per frame for a given table.  bits_per_ofdm_sym is the sum of
 * bits_per_bin.  Returns <=0 on bad args. */
int ofdm_data_syms_per_pkt(const ofdm_cfg *c, int bits_per_ofdm_sym);

/* SYNC symbols in a frame body, ceil(data_syms/sync_period) -- NOT an integer
 * divide.  A SYNC is inserted BEFORE every period-th data symbol, so 5 data
 * symbols at period 32 is still one SYNC. */
int ofdm_n_syncs(const ofdm_cfg *c, int data_syms);

/* Minimum sample distance between consecutive SC preamble positions.
 * Mirrors _frame_min_gap INCLUDING the cp_len jitter allowance -- without it
 * every second frame is discarded at large n_fft. */
int ofdm_frame_min_gap(const ofdm_cfg *c, int n_body_syms);

/* Gray-coded constellation for `bits` in {1,2,4,6}.  Points are unit average
 * power.  `bits_tab` holds n_pts*bits entries, MSB-first per point, in the same
 * order as the points.  Returns 0 ok, -1 no such table.
 *
 * The tables are the ONLY transcription of ofdm_modem's Gray coding, so the
 * regression compares them element by element. */
int ofdm_mod_table(int bits, const double **pts_re, const double **pts_im,
                   const unsigned char **bits_tab, int *n_pts);

/* Flattened form for ofdm_dem_init, which takes all tables concatenated in
 * ascending bit order.  `want` lists the bit widths needed (e.g. {1,2} for a
 * mixed BPSK/QPSK table).  Fills the caller's arrays and reports the totals.
 * Returns 0 ok, -1 bad args, -2 a width has no table, -3 caller arrays too
 * small (needed sizes still written to *n_pts_total / *n_bits_total). */
int ofdm_mod_tables_flat(const int *want, int n_want,
                         int *tab_b, int *tab_n,
                         double *pts_re, double *pts_im,
                         unsigned char *tab_bits, int cap_pts, int cap_bits,
                         int *n_pts_total, int *n_bits_total);

/* The distinct bit widths actually used by a table, ascending.  Handy for
 * driving ofdm_mod_tables_flat from a bits_per_bin array.  Returns the count,
 * or -1 on bad args. */
int ofdm_bat_widths(const int *bits_per_bin, int n, int *out, int cap);

#ifdef __cplusplus
}
#endif
#endif
