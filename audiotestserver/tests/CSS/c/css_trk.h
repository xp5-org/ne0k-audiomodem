/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_trk.h — running local PLL bank + chirp classifier.
 *
 * The streaming decoder (css_pll) locks one symbol grid at a time and decodes
 * it.  css_trk runs beside it on the same input and answers a different
 * question: for EVERY chirp on the air, whose grid is it on?
 *
 *   detector   continuous up/down chirp matched filter (window 2*sps, hop
 *              sps_eff/2), local-normalised correlation per peak, plus the
 *              window-normalised chirp score the mesh reference
 *              (meshmodem_node._make_chirp_detector) uses for carrier sense,
 *              with its ambient calibration and on/off hysteresis.
 *   grids      up to CSS_TRK_MAX_GRIDS running timing loops, one per station
 *              heard: the decoder's two-layer structure (fast acquisition
 *              gain for the first acq_hits chirps, then the settled gain plus
 *              a Type-2 rate integrator, median-of-3 residual filter,
 *              outlier gate), predicting where that station's next chirp
 *              lands in OUR sample clock.
 *   own grid   css_trk_tx() registers each transmission in the RX timebase;
 *              our own chirps are known exactly, so anything that returns on
 *              that schedule (sidetone, acoustic or RF return) is OWN.
 *
 * Every detected chirp is labelled OWN / PEER(n) / ECHO(of grid, delay, gain)
 * / UNKNOWN; grid events report NEW, LOCK, LOST, PHASE_STEP, COLLISION,
 * OWN_LAG, PEER.  Freestanding C99, caller memory. */
#ifndef CSS_TRK_H
#define CSS_TRK_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CSS_TRK_MAX_GRIDS 8

typedef struct {
    int    fs;
    double dur, f_min, f_max, gap_frac;
    double peak_thr;        /* local-normalised MF score to count a chirp (0.35) */
    double rel_floor;       /* keep peaks >= rel_floor * window max (0.06 = -24 dB) */
    double acq_alpha, alpha, rate_alpha;   /* 0.25, 0.10, 0.02                    */
    int    acq_hits;        /* chirps in acquisition gain (20)                    */
    double settle_frac;     /* settled match window, fraction of sps (0.04)       */
    double acq_frac;        /* acquisition match window (0.25)                    */
    int    lost_syms;       /* symbols without a hit before a grid is retired (12) */
    double echo_min_s, echo_max_s;  /* echo delay range (0.002 .. 0.6)             */
    double echo_tol_frac;   /* delay consistency, fraction of sps (0.02)          */
    double echo_min_db;     /* an echo is at least this much weaker (3)           */
    int    echo_hits;       /* consistent hits to declare an echo (4)             */
    double max_ppm;         /* peer clock offset accepted (800)                   */
    int    phase_m;         /* consecutive consistent off-grid hits = step (3)    */
    double collide_db;      /* two grids within this power = collision (6)        */
    double own_max_lag_s;   /* longest own-return lag searched (1.0)              */
    double cal_s;           /* carrier-sense ambient calibration span (3.0)       */
    double carrier_k, carrier_lo, carrier_hi;   /* 2.2, 0.05, 0.12                */
    double busy_hold_s;     /* off-threshold hold (2.0)                           */
    int    smooth_n;        /* median smoothing of the window score (4)           */
} css_trk_cfg;

enum { CSS_TRK_UNKNOWN = 0, CSS_TRK_OWN = 1, CSS_TRK_PEER = 2, CSS_TRK_ECHO = 3 };

enum {
    CSS_TRK_EV_NEW = 1,        /* grid a                                         */
    CSS_TRK_EV_LOCK,           /* grid a, x=ppm                                  */
    CSS_TRK_EV_LOST,           /* grid a, b=hits                                 */
    CSS_TRK_EV_PHASE_STEP,     /* grid a, x=delta samples, y=delta seconds        */
    CSS_TRK_EV_COLLISION,      /* grids a,b, x=power difference dB               */
    CSS_TRK_EV_ECHO,           /* grid a is an echo of grid b, x=delay s, y=gain dB */
    CSS_TRK_EV_OWN_LAG,        /* x=own-return lag seconds, a=hits               */
    CSS_TRK_EV_PEER,           /* grid a labelled peer b                         */
    CSS_TRK_EV_BUSY            /* a=1 busy / 0 idle, x=window score              */
};

typedef struct {
    long   abs_peak;        /* RX sample index of the chirp's MF peak (its end) */
    int    dir;             /* 1 up, 0 down                                     */
    double mag;             /* MF magnitude                                     */
    double score;           /* local-normalised correlation, 0..1               */
    int    grid;            /* grid id (-1 none / own)                          */
    int    label;           /* CSS_TRK_*                                        */
    int    peer;            /* node id when PEER, else -1                       */
    int    echo_of;         /* parent grid for ECHO, else -1                    */
    double residual;        /* samples off the grid prediction                  */
} css_trk_chirp;

typedef struct {
    int    active, id, label, peer, echo_of, settled, hits, misses;
    double t0, period, ppm, power_db, delay_s, gain_db, last_residual;
    long   last_abs;
} css_trk_grid;

typedef struct {
    void *user;
    void (*on_chirp)(void *user, const css_trk_chirp *c);
    void (*on_event)(void *user, int kind, long abs, int a, int b, double x, double y);
} css_trk_cb;

typedef struct css_trk css_trk;

void   css_trk_defaults(css_trk_cfg *c, int fs, double dur, double f_min,
                        double f_max, double gap_frac);
size_t css_trk_workspace(const css_trk_cfg *c);
int    css_trk_init(css_trk **t, void *mem, size_t memlen, const css_trk_cfg *c,
                    const css_trk_cb *cb);
/* RX samples, contiguous with everything pushed before (abs index advances). */
int    css_trk_push(css_trk *t, const float *x, int n);
/* A transmission: first sample enters the line at RX index abs_first;
 * dirs[k] = 1 up / 0 down for chirp k (nsym chirps, period sps_eff). */
int    css_trk_tx(css_trk *t, long abs_first, int nsym, const unsigned char *dirs);
/* The decoder found its chirp at abs_peak this symbol (links that grid). */
void   css_trk_decoder_peak(css_trk *t, long abs_peak);
/* The decoder just decoded a frame from node `src` (labels the linked grid). */
void   css_trk_decoder_frame(css_trk *t, int src);
/* Staged receive: `n` samples were not pushed (the host's chirp detector saw
 * nothing in them).  The RX index advances by n, so grids keep predicting on
 * the stream's own timebase and css_trk_tx() indices stay valid; the analysis
 * window restarts empty and the carrier flag drops. */
void   css_trk_skip(css_trk *t, long n);
int    css_trk_busy(const css_trk *t);
double css_trk_score(const css_trk *t);
long   css_trk_abs(const css_trk *t);
int    css_trk_get_grid(const css_trk *t, int i, css_trk_grid *out);
double css_trk_own_lag(const css_trk *t);     /* seconds, < 0 unknown */

#ifdef __cplusplus
}
#endif
#endif
