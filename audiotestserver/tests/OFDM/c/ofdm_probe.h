/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_probe.h — channel probe and bit allocator, freestanding C89.
 *
 * ChannelProbe.build / .estimate, _probe_snr_lin, isi_ratio, derate_snr_lin and
 * BitAllocator.allocate.  With this the whole OFDM stack is C: a freestanding
 * node can probe a path, negotiate a bit table, transmit and receive with no
 * Python anywhere.
 *
 * WHAT THE PROBE IS FOR, because it drives every design choice below.  The
 * waveform carries TWO blocks and they exist for opposite reasons:
 *
 *   PERIODIC block (identical symbols) -- what ACQUISITION needs.  Every symbol
 *   carries the same pattern, so per-symbol H estimates agree and the coherence
 *   search separates probe from noise by ~0.99 against ~0.36.
 *
 *   MEASUREMENT block (a different pattern per symbol) -- what an HONEST SNR
 *   needs.  A repeated symbol lets room reverberation reach periodic steady
 *   state: each copy's tail arrives in phase with the next and is absorbed into
 *   H as though it were channel gain, so the probe grades a reverberant path as
 *   clean.  Measured over the air at 18 dB per carrier on a path that delivered
 *   4.5 dB.  Because consecutive measurement symbols are UNCORRELATED, each
 *   one's tail lands on the next as genuine interference, exactly as a data
 *   symbol's does, and the variance-based estimator picks it up for free.
 *
 * The DIFFERENCE between the two blocks is a direct measurement of the ISI
 * penalty, and it is reported as such.
 *
 * MEMORY.  The reference keeps every per-symbol H (S x n_occupied complex, ~390
 * KB at the production geometry) and a match score per grid position.  Neither
 * is needed: the SNR estimator is a mean and a variance, which Welford
 * accumulates in O(n_occupied), and the contiguous run of matching symbols can
 * be found by walking outward from the winner instead of scoring the whole grid
 * first.  Working set here is O(n_occupied) plus the acquisition scan's ring.
 *
 * NOT PORTED: the "percentile" SNR estimator.  It needs the full S x n_occupied
 * block history, which is the allocation this design exists to avoid, and the
 * configuration comments record it as an opt-in that is NOT better here.
 * ofdm_probe_init REJECTS it rather than silently substituting the mean.
 */
#ifndef OFDM_PROBE_H
#define OFDM_PROBE_H
#include <stddef.h>
#include "ofdm_cfg.h"
#include "ofdm_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Acquisition tuning, mirroring ofdm_modem's module constants. */
#define OFDM_COH_CARRIERS   40     /* strongest carriers scored for coherence */
#define OFDM_COH_SCAN_SYMS   8     /* probe symbols averaged per candidate    */
#define OFDM_COH_GATE     0.70     /* min peak coherence for a valid probe    */
#define OFDM_COH_MATCH    0.60     /* per-symbol H match to count as "probe"  */

/* Probe symbols needed before the ISI ratio is trustworthy.  A 25-30 symbol
 * fragment gave beta 1.7-1.9 on a link a full 170-symbol run graded at 0.25 --
 * H is too noisy there to separate reverb from noise, so beta is withheld
 * rather than allowed to poison the allocation. */
#define OFDM_ISI_MIN_SYMS   64

/* Largest beta worth ACTING on.  The estimator recovers the impulse response on
 * the n_fft bin grid, so its delay axis wraps at n_fft/fs -- 23.2 ms at
 * 1024/44.1k.  A 100-400 ms room tail is 4-17x that and aliases UNIFORMLY into
 * every delay tap, so the response comes out flat and the ratio saturates near
 * 1 whatever the room is doing.  Over this bound beta means "the tail is longer
 * than this estimator can see", not "the channel carries nothing", and acting
 * on it is worse than ignoring it: a -0.5 dB ceiling zeroes the whole table. */
#define OFDM_ISI_BETA_MAX  0.5

/* SNR floors per modulation, dB.  AWGN threshold plus headroom; the dense
 * floors carry 4-6 dB over their measured threshold and that is NOT slack --
 * it covers reverberation this stack knows is there and cannot always measure.
 * Lowering them on the strength of an AWGN bench broke a real link. */
#define OFDM_FLOOR_BPSK   4.0
#define OFDM_FLOOR_QPSK   7.0
#define OFDM_FLOOR_16QAM 14.0
#define OFDM_FLOOR_64QAM 21.0

typedef struct ofdm_probe ofdm_probe;

enum {
    OFDM_PROBE_OK = 0,
    OFDM_PROBE_TOO_SHORT = 1,
    OFDM_PROBE_SYNC_FAIL = 2,
    OFDM_PROBE_TOO_FEW_SYMS = 3
};

typedef struct {
    int    valid;
    int    error;            /* one of the enum above           */
    double coherence;        /* peak coherence                  */
    double noise_floor;      /* 25th percentile, diagnostic     */
    int    frame_start;
    int    n_syms;           /* periodic symbols averaged       */
    int    n_meas_syms;      /* non-periodic symbols averaged   */
    int    has_isi_penalty;
    double isi_penalty_db;   /* periodic median - measured      */
    int    has_beta;         /* 0 when under OFDM_ISI_MIN_SYMS  */
    double isi_ratio;
} ofdm_probe_info;

/* `max_capture_samples` bounds the acquisition scan's candidate metric, the one
 * array whose length follows the capture rather than the geometry. */
size_t ofdm_probe_workspace(const ofdm_cfg *c, int n_probe_syms,
                            int n_meas_syms, int max_capture_samples);

/* Errors: -1 bad args, -2 memlen too small, -3 a sub-core failed to init,
 *         -4 percentile estimator requested (see the header note). */
int ofdm_probe_init(ofdm_probe **p, void *mem, size_t memlen,
                    const ofdm_cfg *c, int n_probe_syms, int n_meas_syms,
                    int max_capture_samples);

long ofdm_probe_build_samples(const ofdm_probe *p);

/* SC preamble, CE, the periodic block, then the measurement block.
 * Returns samples written, or -1 bad args, -2 cap too small. */
long ofdm_probe_build(ofdm_probe *p, double *out, long cap);

/* Analyse a captured probe.  The five output arrays each take n_occupied
 * doubles and may be NULL if not wanted; `snr_db_periodic` is the periodic
 * block's view, kept because the gap between it and `snr_db` IS the
 * reverberation penalty, measured rather than modelled.
 *
 * Returns 0 if a probe was found (info->valid), 1 if not (info->error set),
 * or -1 on bad args. */
int ofdm_probe_estimate(ofdm_probe *p, const double *rx, int n,
                        double threshold,
                        double *H_re, double *H_im,
                        double *snr_lin, double *snr_db,
                        double *snr_db_periodic,
                        ofdm_probe_info *info);

/* Fraction of channel energy arriving too late to be covered by the guard:
 * beta = E_late / E_covered, an interference-to-signal ratio that sets a hard
 * SINR ceiling of -10*log10(beta) which no equaliser or gain change can lift.
 *
 * `taper_samples` SHORTENS the guard: the overlap-add puts the previous
 * symbol's falling taper over the first taper_samples of every prefix, so the
 * ISI-free guard is cp_len - taper_samples.  Counting the full prefix makes
 * beta optimistic, and beta feeds the allocator.
 *
 * `snr_lin` (may be NULL) lets additive noise be subtracted from the late
 * energy -- noise spreads evenly over all taps and would otherwise masquerade
 * as reverberation.
 *
 * `fft` must be initialised for n_fft.  `scratch` needs n_fft doubles plus
 * (n_fft/2+1) ofdm_cpx; ofdm_isi_scratch() reports the byte count. */
size_t ofdm_isi_scratch(int n_fft);
double ofdm_isi_ratio(ofdm_fft *fft, const double *H_re, const double *H_im,
                      int n_occ, int n_fft, int bin_lo, int cp_len,
                      const double *snr_lin, int taper_samples,
                      void *scratch, size_t scratch_len);

/* Fold an ISI floor into a per-carrier linear SNR, in place.
 *
 *     SINR = S / (N + beta*S)   =>   1/SINR = 1/SNR + beta
 *
 * ISI power scales with the carrier's own signal power -- it IS the signal,
 * arriving late -- so a quiet-but-reverberant carrier is capped at 1/beta no
 * matter how strong it is.  That cap is the point. */
void ofdm_derate_snr_lin(double *snr_lin, int n, double beta);

/* Highest bit count each data carrier supports.  Pilots (every pilot_spacing-th
 * occupied carrier) are skipped, so `bits_out` receives n_data_carriers values.
 *
 * The margin is SPENT FIRST and the ISI ceiling applied to what is left.  Order
 * matters and this is the only order that composes: applying both as
 * independent dB penalties double-counts -- they answer the same question --
 * and at a measured beta of 0.25, subtracting a 9 dB margin from an already
 * capped 6 dB ceiling mutes every carrier and the link never comes up.
 *
 * Returns the count written, or -1 on bad args. */
int ofdm_allocate(const double *snr_db_occupied, int n_occupied,
                  int pilot_spacing, double snr_margin_db, int max_bits,
                  double isi_beta, int *bits_out, int cap);

#ifdef __cplusplus
}
#endif
#endif
