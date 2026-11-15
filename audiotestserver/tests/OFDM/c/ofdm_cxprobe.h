/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_cxprobe.h — complex-baseband (I/Q) channel probe, freestanding C89.
 *
 * The counterpart of ofdm_probe for an ofdm_cxcfg: build a probe, find it in a
 * capture, and measure per-carrier SNR over the SIGNED bins, which
 * ofdm_allocate turns into the bit table (it reads only counts and the pilot
 * spacing, so it serves complex mode unchanged).  Same waveform plan as the
 * real probe, for the same reasons (see ofdm_probe.h):
 *
 *     [ SC preamble ][ CE ][ P x n_probe_syms ][ M_0 .. M_{n_meas_syms-1} ]
 *
 *   SC + CE   exactly the first two symbols of an ofdm_cxmod frame, so
 *             ofdm_cxsync finds them and measures the CFO;
 *   P         the periodic block (OFDM_SEED_PROBE on every occupied carrier,
 *             1/sqrt(2)): what acquisition locks on;
 *   M_j       the measurement block (one PRBS stream from OFDM_SEED_MEAS, n_occupied
 *             bits per symbol, as the real probe): consecutive symbols
 *             uncorrelated, so ISI and the IQ image land as interference, as on
 *             data.  The gap between the two blocks' SNR is reported.
 *
 * Estimate, in order:
 *
 *  1. ofdm_cxacq (differential coherence, CFO-invariant) over the capture;
 *     peak >= max(threshold, OFDM_COH_GATE) or OFDM_PROBE_SYNC_FAIL.
 *  2. Walk the grid outward from the peak while each symbol matches the
 *     periodic reference (OFDM_COH_MATCH, phase-only on the strongest
 *     carriers, invariant to the per-symbol CFO rotation): the first periodic
 *     symbol, to within a fraction of a symbol.
 *  3. ofdm_cxsync_find in a window ahead of it: the SC preamble two symbols
 *     before gives the EXACT frame start and the CFO, and its CE-sequence score
 *     confirms the probe.  A tone or DC offset passes step 1 (it repeats
 *     symbol to symbol up to a phase) and fails here: OFDM_CXPROBE_NO_SC.
 *     Unlike the real probe there is then no timing ambiguity left: every
 *     symbol position is known, and both blocks are read on that one grid.
 *  4. Every window [pos + w0, + n_fft) is derotated by the CFO (phase from the
 *     frame start, as ofdm_cxdem; w0 = ofdm_cxdem's window offset).
 *     Per symbol, a common phase a + b * signed bin is fitted against the CE's
 *     H over ALL carriers (residual CFO: ~0.02 rad/symbol, which over a
 *     100-symbol probe would decorrelate the mean; sample-clock drift: a slope)
 *     and removed, as the demodulator's pilots will remove it.
 *  5. Periodic block: Welford mean/variance of H per carrier, the real probe's
 *     estimator: snr = 2|mean H|^2 / var(H), i.e. the SNR of a unit-power
 *     constellation (the known symbols are 1/sqrt(2)).
 *  6. Measurement block, per carrier k with mirror -k occupied: least squares
 *         y_j[k] = G_k x_j[k] + I_k x_j[-k] + noise
 *     (the known BPSK is real, so conj(x) = x).  G is the channel, I the IQ
 *     image: an IQ gain/phase imbalance puts nu*conj(s) beside mu*s, and
 *     conj(s) at bin k is the conjugated spectrum at -k.  Then
 *         snr_lin   = |G|^2 / (|I|^2 + sigma^2)     (unit-power data at -k)
 *         image_db  = 10 log10(|I|^2 / |G|^2)
 *     with sigma^2 the residual variance, and |G|^2, |I|^2 in snr_lin less the
 *     noise their estimates carry (unbiased at short blocks).  image_db is the
 *     raw ratio: on a clean radio it reads the fit's floor,
 *     about -10 log10(SNR * n_meas / 2).  The SNR handed to the allocator
 *     therefore COUNTS the image as interference (the demodulator does not
 *     cancel it), while image_db says how much of the loss it is.  Carriers
 *     whose mirror is not occupied, or a block too short or too correlated to
 *     separate the two, fall back to the one-regressor fit (I = 0).
 *
 * What the image model covers.  An imbalance at the TRANSMITTER, or one at the
 * receiver with no CFO, is exactly y = G x[k] + I x[-k].  At the receiver
 * BEHIND a CFO e, the image is the conjugate of the offset signal: after
 * derotation it sits at an offset of -2e and turns at twice the CFO rate, so
 * this fit sees only part of it.  The rest falls into sigma^2, so the SNR
 * stays honest and image_db reads low.
 *
 * Not ported from ofdm_probe: the ISI ratio beta.  At SDR rates the cyclic
 * prefix is hundreds of microseconds against a delay spread of a few, and
 * where ISI does occur the measurement block already counts it.
 *
 * Positions and samples as the rest of complex mode: interleaved I,Q doubles,
 * every n / start counts COMPLEX samples.  Same contract: no malloc, no stdio,
 * no globals, one caller-owned workspace.
 */
#ifndef OFDM_CXPROBE_H
#define OFDM_CXPROBE_H
#include <stddef.h>
#include "ofdm_cx.h"
#include "ofdm_probe.h"   /* OFDM_COH_*, OFDM_PROBE_* errors, ofdm_allocate */

#ifdef __cplusplus
extern "C" {
#endif

/* error code beyond ofdm_probe's: a coherent block with no SC preamble + CE
 * in front of it (a CW tone, a DC offset, or the preamble cut off). */
#define OFDM_CXPROBE_NO_SC 4

/* image_db for a carrier whose mirror carries nothing. */
#define OFDM_CXPROBE_NO_IMAGE_DB (-200.0)

typedef struct ofdm_cxprobe ofdm_cxprobe;

typedef struct {
    int    valid;
    int    error;            /* OFDM_PROBE_* or OFDM_CXPROBE_NO_SC            */
    double coherence;        /* acquisition peak                              */
    double noise_floor;      /* 25th percentile of the metric, diagnostic     */
    int    acq_start;        /* the peak's candidate start (sample)           */
    int    frame_start;      /* first sample of the SC preamble's prefix      */
    double cfo;              /* subcarriers, ofdm_cxsync's estimate           */
    double sc_score;         /* its matched-filter NCC                        */
    double int_score;        /* its CE-sequence score D                       */
    int    n_syms;           /* periodic symbols averaged                     */
    int    n_meas_syms;      /* measurement symbols used (0: none)            */
    int    has_isi_penalty;
    double isi_penalty_db;   /* median periodic SNR - median measured SNR     */
    int    has_image;
    double image_db;         /* median image_db over the mirrored carriers    */
} ofdm_cxprobe_info;

/* max_capture_samples bounds the acquisition metric (one value per grid step);
 * max_cfo_bins is ofdm_cxsync_init's integer search range.  0 if rejected. */
size_t ofdm_cxprobe_workspace(const ofdm_cxcfg *x, int n_probe_syms,
                              int n_meas_syms, int max_capture_samples,
                              int max_cfo_bins);

/* n_probe_syms >= OFDM_COH_SCAN_SYMS, n_meas_syms >= 0; both must be the
 * transmitter's.  Errors: -1 bad args, -2 memlen too small, -3 a sub-core
 * failed to init. */
int ofdm_cxprobe_init(ofdm_cxprobe **p, void *mem, size_t memlen,
                      const ofdm_cxcfg *x, int n_probe_syms, int n_meas_syms,
                      int max_capture_samples, int max_cfo_bins);

/* Complex samples ofdm_cxprobe_build writes: symbol_samples * (2 + n_probe
 * + n_meas) + 2 * taper. */
long ofdm_cxprobe_build_samples(const ofdm_cxprobe *p);

/* The probe, raw level (SC at the frame's TX gain, as ofdm_cxmod), out_iq
 * 2 * cap doubles.  Returns samples written, -1 bad args, -2 cap too small. */
long ofdm_cxprobe_build(ofdm_cxprobe *p, double *out_iq, long cap);

/* Analyse a capture.  H_re/H_im/snr_lin/snr_db: n_occupied doubles each,
 * required.  snr_db_periodic and image_db: n_occupied each, may be NULL.
 * Returns 0 found (info->valid), 1 not found (info->error), -1 bad args. */
int ofdm_cxprobe_estimate(ofdm_cxprobe *p, const double *rx_iq, int n,
                          double threshold,
                          double *H_re, double *H_im,
                          double *snr_lin, double *snr_db,
                          double *snr_db_periodic, double *image_db,
                          ofdm_cxprobe_info *info);

#ifdef __cplusplus
}
#endif
#endif
