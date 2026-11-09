/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_pll.h — C port of css_decoder.decode(), the streaming two-layer timing
 * PLL CSS decoder: impulse blanker, STFT whitening, peak AGC, the 8-slice
 * matched filter, IF-slope confidence blend, the timing PLL (acquisition and
 * settled peak_est IIR, the n_read integrator, the Type-2 rate estimate,
 * median / outlier / strength gates, snapshot and revert), the soft preamble
 * detector with its lag and early-arrival guards, whole-frame retro timing,
 * the CRC-anchored shift search (inline and deferred), and the frame
 * pipeline (css_dec_raw + reedsolo-exact RS erasures + CRC).
 *
 * Freestanding C99: caller memory, no malloc/stdio/globals; a sample counter
 * instead of wall time; events instead of prints.
 *
 * Feeding.  The reference reads exactly n_read (+ one-shot correction)
 * samples per symbol; css_pll_next_nr() says how many the next symbol
 * consumes and css_pll_symbol() runs one.  css_pll_push() is the streaming
 * convenience: it buffers arbitrary blocks and runs symbols as they fill.
 *
 * Configuration.  Every value is the RESOLVED number decode() computes in its
 * setup (§A-§F): the host does the cfg/info-file parsing (css_pll_c.py
 * reuses decode()'s own expressions), so the C never re-derives a default.
 */
#ifndef CSS_PLL_H
#define CSS_PLL_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CSS_PLL_MAX_PREAMBLE 64

typedef struct {
    /* PHY geometry */
    int    fs;
    double dur, f_min, f_max;
    int    sps, gap_samps, sps_eff, buf_size;
    int    mf_hann;                 /* mf_window == "hann"                     */
    double search_frac;             /* timing_search_frac                      */
    /* gates and loop gains (alpha already scaled for long symbols) */
    double silence_thr, sync_threshold, alpha, alpha_acq, nread_revert;
    double rate_refine_alpha, retro_rate_adopt_inlier;
    int    wpk_track_enable;
    double wpk_track_k, wpk_track_alpha;
    double agc_target, agc_attack, agc_release, agc_max;
    int    resync_after;
    double min_tim_conf;
    int    max_terr_smp, presync_max_terr_smp;
    double tim_track_alpha;
    int    impulse_blank_enable;
    double impulse_blank_k, impulse_clip_floor, impulse_blank_max_frac, impulse_bg_alpha;
    int    whiten_enable;
    double whiten_stft_win_s, whiten_bg_alpha;
    int    whiten_redesign_every, whiten_taps;
    double whiten_max_gain_db;
    double timing_min_snr_db;
    int    timing_acq_symbols, timing_median_n;
    double timing_max_step_frac;
    double pll_weight, era_thr;
    int    post_acq_search_smp;
    /* frame geometry */
    int    chunk_size, rs_t, interleave_rows, conv_k7, diff_enc;
    int    rs_enabled;              /* reference had an RS codec (rs_codec is not None) */
    int    base_bits, frame_symbols, preamble_per_frame;
    int    n_preamble;
    unsigned char preamble[CSS_PLL_MAX_PREAMBLE];
    int    training_symbols;
    /* preamble detector */
    int    syndet_lookback;
    double syndet_min_snr_db, syndet_snr_ref_db;
    int    syndet_lag_guard;
    double syndet_alt_thr;
    int    syndet_max_lag;
    double syndet_bg_alpha, syndet_margin, syndet_thr_floor;
    int    syndet_flush_terr;
    /* retro / anchor */
    long   max_retro_smp;
    int    anchor_lookahead;        /* retro_anchor_lookahead                  */
    int    retro_anchor_max_shift;  /* < 0: default (audio_hist_len - frame_symbols) */
    /* sizing (not in the reference): the largest n_read the workspace holds.
     * 0 = sps_eff + 2*sps. A symbol needing more is clamped and reported. */
    int    nr_max;
    /* sync fixes (decode() syndet_conf_floor / syndet_cov_knee /
     * syndet_mask_ineligible / impulse_silence_floor) */
    double syndet_conf_floor, syndet_cov_knee;
    int    syndet_mask_ineligible;
    double impulse_silence_floor;
    int    whiten_stft_win_max;     /* samples; 0 = no cap                    */
    int    ring_lookahead;          /* resolved; chirp end = buf_size-1-this (<= gap: old layout) */
    /* pre-sync MF search over one whole symbol period [buf_size-sps_eff, buf_size)
     * instead of +-search_range, timing error wrapped to +-sps_eff/2
     * (decode() presync_full_period) */
    int    presync_full_period;
    /* first signal symbol after this many silent ones (pre-sync) anchors the
     * early-sync guard like css_pll_mark_onset(); 0 = off
     * (decode() onset_guard_silence_syms) */
    int    onset_guard_silence_syms;
    /* phase-step recovery while synced (decode() step_recover): settled, the
     * narrow MF snr dropping step_drop_db under the frame's running snr triggers
     * a +-search_range search; in acquisition a peak beyond post_acq_search is
     * the candidate.  A strong chirp more than post_acq_search/2 away that the
     * next symbol confirms within step_tol_smp is a timing slip (jitter-buffer
     * insert/drop): step the read by it, decide both symbols there */
    int    step_recover;
    double step_drop_db, step_accept_db, step_min_conf;
    int    step_tol_smp;
    int    crc_bytes;               /* frame check: 4 = CRC-32, 2 = CRC-16; 0 reads as 4 */
} css_pll_cfg;

/* One record per symbol, emitted where the reference has finished the timing
 * update and is about to run the preamble detector / frame assembly. */
typedef struct {
    long   sym;              /* sym_count (1-based, as the reference)         */
    long   abs_start;        /* absolute input index of this symbol's first sample */
    int    nr;               /* samples consumed (n_read + one-shot)          */
    int    found_peak, bit;
    double conf, snr_db, winner_pk, pll_conf, pll_r2;
    int    timing_err, peak_est, n_read, acq_left, settled, sync_found;
    double rate_est, timing_frac, final_conf, agc_gain;
    int    outlier, wpk_strong, signal_present, n_blanked, whitened;
} css_pll_sym;

enum {
    CSS_PLL_SRC_STREAM = 0, CSS_PLL_SRC_RETRO = 1, CSS_PLL_SRC_DEFERRED = 2
};

typedef struct {
    long   sym;              /* sym_count at completion                       */
    int    seq;
    const unsigned char *data; int ndata;   /* chunk_size payload bytes        */
    double snr_db;
    int    rs_errors;
    int    source;           /* CSS_PLL_SRC_*                                 */
} css_pll_frame;

/* Event kinds (on_event). a/b/x/y meaning per kind: */
enum {
    CSS_EV_SYNC_ACQUIRED = 1,   /* a=lag  b=sync_at_symbol x=score y=eff_thr  */
    CSS_EV_SYNC_TOO_EARLY,      /* a=window_end b=since_resync x=score y=alt  */
    CSS_EV_SYNC_STALE_REJECT,   /* a=lag x=score                              */
    CSS_EV_RESYNC,              /* a=hard(0/1) b=reason code                  */
    CSS_EV_RETRO_GATE,          /* a=inliers b=n x=delta                      */
    CSS_EV_RETRO_SYNC,          /* a=one_shot b=reentered_acq x=drift y=rate  */
    CSS_EV_RETRO_ANCHOR,        /* a=shift                                    */
    CSS_EV_SNAPSHOT_RESTORE,    /* a=n_read                                   */
    CSS_EV_FRAME_ERROR,         /* a=seq b=rs_fail(0/1) x=snr                 */
    CSS_EV_PLL_OUTLIER,         /* a=terr b=median                            */
    CSS_EV_SYNC_SNAP,           /* a=old peak_est b=new                       */
    CSS_EV_IMPULSE_BLANK,       /* a=n b=skipped(0/1) x=thr                   */
    CSS_EV_NR_CLAMP,            /* a=wanted b=nr_max                          */
    CSS_EV_PHASE_STEP           /* a=step smp b=1 confirmed/0 candidate x=wide snr y=narrow snr */
};
enum {   /* CSS_EV_RESYNC reason codes */
    CSS_RS_PPF_RS = 1, CSS_RS_RS_FAIL, CSS_RS_PPF_OK, CSS_RS_PPF_CRC, CSS_RS_CRC_FAIL,
    CSS_RS_REARM
};

typedef struct {
    void *user;
    void (*on_symbol)(void *user, const css_pll_sym *s);
    void (*on_frame)(void *user, const css_pll_frame *f);
    void (*on_event)(void *user, int kind, long sym, int a, int b, double x, double y);
} css_pll_cb;

typedef struct css_pll css_pll;

size_t css_pll_workspace(const css_pll_cfg *cfg);
int    css_pll_init(css_pll **p, void *mem, size_t memlen, const css_pll_cfg *cfg,
                    const css_pll_cb *cb);
int    css_pll_next_nr(const css_pll *p);
/* raw[nr] with nr == css_pll_next_nr(). Returns 0, negative on misuse. */
int    css_pll_symbol(css_pll *p, const float *raw, int nr);
/* Streaming: buffer x[n], run every symbol that fills. Returns symbols run. */
int    css_pll_push(css_pll *p, const float *x, int n);

/* Burst boundary (NOT in the reference, which decodes one message per run):
 * the host saw the channel go quiet, so drop any half-built frame and listen
 * for the next preamble now -- a soft resync, timing state kept.  Without it a
 * continuously running decoder spends resync_after frames of CRC failures on
 * silence after every transmission before it can sync again. */
void   css_pll_rearm(css_pll *p);

/* Acquisition aid from the running chirp tracker (NOT in the reference):
 * before sync, shift the next read so the MF's expected peak lands on a chirp
 * known to end at RX sample abs_peak on a grid of `period` samples.  The
 * reference decoder misses preambles whose onset falls in a bad phase of its
 * read grid (the old GUI re-ran each burst at four sub-symbol shifts); this
 * puts the grid where the chirps are.  Returns the shift applied. */
int    css_pll_align(css_pll *p, long abs_peak, double period);

/* Acquisition aid (NOT in the reference): a transmission's first chirp starts
 * at RX sample abs_onset.  The reference anchors its SYNC_TOO_EARLY guard
 * (training + preamble - 1 symbols) to the last resync, which only means
 * "transmission start" when the capture begins with the burst; with silence
 * between bursts the guard expires before the burst arrives and the detector
 * false-locks on a Barker sidelobe in the training.  This re-anchors the
 * guard at the real onset. */
int    css_pll_mark_onset(css_pll *p, long abs_onset);

/* Diagnostics the reference puts in its result dict. */
typedef struct {
    long   symbols;
    int    frames_decoded, frames_errored;
    int    sync_acquired, sync_lag;
    long   sync_at_symbol;
    double sync_score;
    double snr_sum; long snr_n;     /* avg_snr_db = mean of every symbol's SNR */
    double rms_max, rms_sum; long rms_n;
    int    sync_found, timing_settled, acq_left;
    double rate_est, agc_gain;
    int    peak_est, n_read;
} css_pll_diag;
void   css_pll_get_diag(const css_pll *p, css_pll_diag *d);

#ifdef __cplusplus
}
#endif
#endif
