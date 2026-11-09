# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""css_pll_c.py — ctypes bridge to the C streaming PLL decoder (CSS/c/libcsspll.*).

`decode(cfg, stop_event)` is a drop-in for css_decoder.decode(): same config
keys, same info-file handling, same audio read loop (exactly n_read samples a
symbol, from the same sounddevice / mock stream), same result dict.  The DSP
and state machine run in C; this file only parses the config the way decode()
does (§A-§F, copied expression for expression so every default is the
reference's) and turns C events back into the reference's records.

css_decoder.decode() hands over to this when CSS_PLL_BACKEND is "c" (float32,
the shipping build) or "c64" (float64 grading build, to be compared with the
reference run under CSS_SIG_DTYPE=float64).  As with every other backend here,
a requested library that cannot load RAISES; it never falls back to Python.

`PllStream` is the streaming wrapper the GUI uses: build it from a decode
config + info dict, push() audio blocks of any size, get frame / symbol /
event callbacks.

`cfg["trace"]`, when a list, receives one dict per symbol and one per event,
the same records the instrumented Python decode() produces, for
regression/css_pll_regress.py.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import cbackend  # noqa: E402

MAX_PRE = 64

EVENT_NAMES = {
    1: "SYNC_ACQUIRED", 2: "SYNC_TOO_EARLY", 3: "SYNC_STALE_REJECT", 4: "RESYNC",
    5: "RETRO_GATE", 6: "RETRO_SYNC", 7: "RETRO_ANCHOR", 8: "SNAPSHOT_RESTORE",
    9: "FRAME_ERROR", 10: "PLL_OUTLIER", 11: "SYNC_SNAP", 12: "IMPULSE_BLANK",
    13: "NR_CLAMP", 14: "PHASE_STEP",
}
SOURCES = {0: "stream", 1: "retro", 2: "retro_deferred"}


class Cfg(ctypes.Structure):
    _fields_ = [
        ("fs", ctypes.c_int),
        ("dur", ctypes.c_double), ("f_min", ctypes.c_double), ("f_max", ctypes.c_double),
        ("sps", ctypes.c_int), ("gap_samps", ctypes.c_int), ("sps_eff", ctypes.c_int),
        ("buf_size", ctypes.c_int),
        ("mf_hann", ctypes.c_int),
        ("search_frac", ctypes.c_double),
        ("silence_thr", ctypes.c_double), ("sync_threshold", ctypes.c_double),
        ("alpha", ctypes.c_double), ("alpha_acq", ctypes.c_double),
        ("nread_revert", ctypes.c_double),
        ("rate_refine_alpha", ctypes.c_double), ("retro_rate_adopt_inlier", ctypes.c_double),
        ("wpk_track_enable", ctypes.c_int),
        ("wpk_track_k", ctypes.c_double), ("wpk_track_alpha", ctypes.c_double),
        ("agc_target", ctypes.c_double), ("agc_attack", ctypes.c_double),
        ("agc_release", ctypes.c_double), ("agc_max", ctypes.c_double),
        ("resync_after", ctypes.c_int),
        ("min_tim_conf", ctypes.c_double),
        ("max_terr_smp", ctypes.c_int), ("presync_max_terr_smp", ctypes.c_int),
        ("tim_track_alpha", ctypes.c_double),
        ("impulse_blank_enable", ctypes.c_int),
        ("impulse_blank_k", ctypes.c_double), ("impulse_clip_floor", ctypes.c_double),
        ("impulse_blank_max_frac", ctypes.c_double), ("impulse_bg_alpha", ctypes.c_double),
        ("whiten_enable", ctypes.c_int),
        ("whiten_stft_win_s", ctypes.c_double), ("whiten_bg_alpha", ctypes.c_double),
        ("whiten_redesign_every", ctypes.c_int), ("whiten_taps", ctypes.c_int),
        ("whiten_max_gain_db", ctypes.c_double),
        ("timing_min_snr_db", ctypes.c_double),
        ("timing_acq_symbols", ctypes.c_int), ("timing_median_n", ctypes.c_int),
        ("timing_max_step_frac", ctypes.c_double),
        ("pll_weight", ctypes.c_double), ("era_thr", ctypes.c_double),
        ("post_acq_search_smp", ctypes.c_int),
        ("chunk_size", ctypes.c_int), ("rs_t", ctypes.c_int),
        ("interleave_rows", ctypes.c_int), ("conv_k7", ctypes.c_int),
        ("diff_enc", ctypes.c_int),
        ("rs_enabled", ctypes.c_int),
        ("base_bits", ctypes.c_int), ("frame_symbols", ctypes.c_int),
        ("preamble_per_frame", ctypes.c_int),
        ("n_preamble", ctypes.c_int),
        ("preamble", ctypes.c_ubyte * MAX_PRE),
        ("training_symbols", ctypes.c_int),
        ("syndet_lookback", ctypes.c_int),
        ("syndet_min_snr_db", ctypes.c_double), ("syndet_snr_ref_db", ctypes.c_double),
        ("syndet_lag_guard", ctypes.c_int),
        ("syndet_alt_thr", ctypes.c_double),
        ("syndet_max_lag", ctypes.c_int),
        ("syndet_bg_alpha", ctypes.c_double), ("syndet_margin", ctypes.c_double),
        ("syndet_thr_floor", ctypes.c_double),
        ("syndet_flush_terr", ctypes.c_int),
        ("max_retro_smp", ctypes.c_long),
        ("anchor_lookahead", ctypes.c_int),
        ("retro_anchor_max_shift", ctypes.c_int),
        ("nr_max", ctypes.c_int),
        ("syndet_conf_floor", ctypes.c_double), ("syndet_cov_knee", ctypes.c_double),
        ("syndet_mask_ineligible", ctypes.c_int),
        ("impulse_silence_floor", ctypes.c_double),
        ("whiten_stft_win_max", ctypes.c_int),
        ("ring_lookahead", ctypes.c_int),
        ("presync_full_period", ctypes.c_int),
        ("onset_guard_silence_syms", ctypes.c_int),
        ("step_recover", ctypes.c_int),
        ("step_drop_db", ctypes.c_double), ("step_accept_db", ctypes.c_double),
        ("step_min_conf", ctypes.c_double),
        ("step_tol_smp", ctypes.c_int),
        ("crc_bytes", ctypes.c_int),
    ]


class Sym(ctypes.Structure):
    _fields_ = [
        ("sym", ctypes.c_long), ("abs_start", ctypes.c_long), ("nr", ctypes.c_int),
        ("found_peak", ctypes.c_int), ("bit", ctypes.c_int),
        ("conf", ctypes.c_double), ("snr_db", ctypes.c_double),
        ("winner_pk", ctypes.c_double), ("pll_conf", ctypes.c_double),
        ("pll_r2", ctypes.c_double),
        ("timing_err", ctypes.c_int), ("peak_est", ctypes.c_int),
        ("n_read", ctypes.c_int), ("acq_left", ctypes.c_int),
        ("settled", ctypes.c_int), ("sync_found", ctypes.c_int),
        ("rate_est", ctypes.c_double), ("timing_frac", ctypes.c_double),
        ("final_conf", ctypes.c_double), ("agc_gain", ctypes.c_double),
        ("outlier", ctypes.c_int), ("wpk_strong", ctypes.c_int),
        ("signal_present", ctypes.c_int), ("n_blanked", ctypes.c_int),
        ("whitened", ctypes.c_int),
    ]


class Frame(ctypes.Structure):
    _fields_ = [
        ("sym", ctypes.c_long), ("seq", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_ubyte)), ("ndata", ctypes.c_int),
        ("snr_db", ctypes.c_double), ("rs_errors", ctypes.c_int),
        ("source", ctypes.c_int),
    ]


class Diag(ctypes.Structure):
    _fields_ = [
        ("symbols", ctypes.c_long),
        ("frames_decoded", ctypes.c_int), ("frames_errored", ctypes.c_int),
        ("sync_acquired", ctypes.c_int), ("sync_lag", ctypes.c_int),
        ("sync_at_symbol", ctypes.c_long), ("sync_score", ctypes.c_double),
        ("snr_sum", ctypes.c_double), ("snr_n", ctypes.c_long),
        ("rms_max", ctypes.c_double), ("rms_sum", ctypes.c_double), ("rms_n", ctypes.c_long),
        ("sync_found", ctypes.c_int), ("timing_settled", ctypes.c_int),
        ("acq_left", ctypes.c_int),
        ("rate_est", ctypes.c_double), ("agc_gain", ctypes.c_double),
        ("peak_est", ctypes.c_int), ("n_read", ctypes.c_int),
    ]


SYM_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(Sym))
FRAME_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(Frame))
EVENT_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_long,
                            ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double)


class Callbacks(ctypes.Structure):
    _fields_ = [("user", ctypes.c_void_p), ("on_symbol", SYM_CB),
                ("on_frame", FRAME_CB), ("on_event", EVENT_CB)]


_LIBS: dict = {}


def _backend_key(f64: bool) -> str:
    return "css_pll64" if f64 else "css_pll"


def load(f64: bool = False):
    key = _backend_key(f64)
    lib = _LIBS.get(key)
    if lib is not None:
        return lib
    path = cbackend.require(key)
    lib = ctypes.CDLL(str(path))
    lib.css_pll_workspace.restype = ctypes.c_size_t
    lib.css_pll_workspace.argtypes = [ctypes.POINTER(Cfg)]
    lib.css_pll_init.restype = ctypes.c_int
    lib.css_pll_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                 ctypes.c_size_t, ctypes.POINTER(Cfg),
                                 ctypes.POINTER(Callbacks)]
    lib.css_pll_next_nr.restype = ctypes.c_int
    lib.css_pll_next_nr.argtypes = [ctypes.c_void_p]
    lib.css_pll_symbol.restype = ctypes.c_int
    lib.css_pll_symbol.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.css_pll_push.restype = ctypes.c_int
    lib.css_pll_push.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.css_pll_align.restype = ctypes.c_int
    lib.css_pll_align.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_double]
    lib.css_pll_mark_onset.restype = ctypes.c_int
    lib.css_pll_mark_onset.argtypes = [ctypes.c_void_p, ctypes.c_long]
    lib.css_pll_rearm.restype = None
    lib.css_pll_rearm.argtypes = [ctypes.c_void_p]
    lib.css_pll_get_diag.restype = None
    lib.css_pll_get_diag.argtypes = [ctypes.c_void_p, ctypes.POINTER(Diag)]
    _LIBS[key] = lib
    return lib


# ──────────────────────────────────────────────────────────────────────────────
# Configuration: decode() §A-§F, expression for expression
# ──────────────────────────────────────────────────────────────────────────────

def build_cfg(cfg: dict, conf: dict) -> tuple[Cfg, dict]:
    """(C config, extras) from a decode() cfg dict and its css_info.json dict."""
    import css_decoder as cd

    c = Cfg()
    extras = {}
    silence_thr    = float(cfg.get("silence_threshold",  0.002))
    sync_threshold = float(cfg.get("sync_threshold",     0.65))
    alpha          = float(cfg.get("timing_loop_gain",   0.10))
    alpha_acq      = float(cfg.get("timing_acq_gain",    0.25))
    nread_revert   = float(cfg.get("nread_revert_alpha", 0.05))
    search_frac    = float(cfg.get("timing_search_frac", 0.25))
    rate_refine_alpha = float(cfg.get("rate_refine_alpha", 0.02))
    retro_rate_adopt_inlier = float(cfg.get("retro_rate_adopt_inlier", 1.01))
    wpk_track_enable = bool(cfg.get("wpk_track_gate_enable", True))
    wpk_track_k      = float(cfg.get("wpk_track_k", 3.0))
    wpk_track_alpha  = float(cfg.get("wpk_track_ema_alpha", 0.05))
    agc_target     = float(cfg.get("agc_target_peak",    0.4))
    agc_attack     = float(cfg.get("agc_attack",         0.8))
    agc_release    = float(cfg.get("agc_release",        0.02))
    agc_max        = float(cfg.get("agc_max_gain",       200.0))
    resync_after   = int(  cfg.get("resync_after",        3))
    mf_window      = str(  cfg.get("mf_window",          "none"))

    fs            = int(  cfg.get("fs",         conf["fs"]))
    dur           = float(cfg.get("symbol_dur", conf["symbol_dur"]))
    f_min         = float(cfg.get("f_min",      conf["f_min"]))
    f_max         = float(cfg.get("f_max",      conf["f_max"]))
    info_preamble = conf.get("preamble")
    cfg_preamble  = cfg.get("preamble")
    if cfg_preamble is not None:
        full_preamble = list(cfg_preamble)
    elif info_preamble is not None:
        full_preamble = list(info_preamble)
    else:
        full_preamble = list(cd.BARKER_13)
    sync_word = full_preamble

    chunk_size      = int(  cfg.get("chunk_size", conf["chunk_size"]))
    crc_bytes       = int(  conf.get("crc_bytes",       cd.CRC_BYTES))
    rs_t            = int(  conf.get("rs_t",            0))
    rs_parity       = rs_t * 2
    interleave_rows = int(  conf.get("interleave_rows", 0))
    rs_codec = cd._make_rs_codec(rs_t)
    if rs_t > 0 and rs_codec is None:
        rs_parity = 0

    sps      = int(fs * dur)
    buf_size = sps * 3
    _gap_frac_cfg  = float(cfg.get("symbol_gap_frac", 0.0))
    _gap_frac_conf = conf.get("symbol_gap_frac")
    _gap_frac = float(_gap_frac_conf) if _gap_frac_conf is not None else _gap_frac_cfg
    _gap_frac = float(np.clip(_gap_frac, 0.0, 0.5))
    gap_samps = int(sps * _gap_frac)
    sps_eff   = sps + gap_samps

    min_tim_conf    = float(cfg.get("min_timing_conf",    0.10))
    max_terr_smp    = int(  cfg.get("max_timing_err_smp", sps // 2))
    presync_max_terr_smp = int(cfg.get("presync_max_terr_smp", max(1, int(sps * 0.02))))
    tim_track_alpha = float(cfg.get("timing_track_alpha", 0.050))
    impulse_blank_enable    = bool(cfg.get("impulse_blank_enable", True))
    impulse_blank_k         = float(cfg.get("impulse_blank_k",     6.0))
    impulse_clip_floor      = float(cfg.get("impulse_clip_floor",  0.98))
    impulse_blank_max_frac  = float(cfg.get("impulse_blank_max_frac", 0.08))
    whiten_enable          = bool(cfg.get("whiten_enable", True))
    whiten_stft_win_s      = float(cfg.get("whiten_stft_win_s", 0.05))
    whiten_bg_alpha        = float(cfg.get("whiten_bg_alpha", 0.10))
    whiten_redesign_every  = int(cfg.get("whiten_redesign_every", 4))
    whiten_taps            = int(cfg.get("whiten_taps", 2047))
    whiten_max_gain_db     = float(cfg.get("whiten_max_gain_db", 18.0))
    timing_min_snr_db = float(cfg.get("timing_min_snr_db", 3.0))
    timing_acq_symbols = int(cfg.get("timing_acq_symbols", 20))
    timing_median_n = max(1, int(cfg.get("timing_median_n", 3)))
    timing_max_step_frac = float(cfg.get("timing_max_step_frac", 0.10))
    base_bits = int(conf.get("base_bits", (1 + chunk_size + crc_bytes + rs_parity) * 8))
    frame_symbols      = int( conf.get("frame_symbols",      base_bits))
    preamble_per_frame = bool(conf.get("preamble_per_frame", False))
    diff_enc           = bool(conf.get("diff_encode",        False))
    conv_code          =      conf.get("conv_code",          None)
    pll_weight = float(cfg.get("pll_weight", 0.30))
    era_thr = float(cfg.get("erasure_conf_thr", 0.25))

    _n_preamble = len(sync_word)
    if cd._default_training_symbols is not None:
        _training_fallback = cd._default_training_symbols(dur)
    else:
        _training_fallback = 16
    _training_symbols = int(conf.get("training_symbols", _training_fallback))
    _auto_lookback    = max(_n_preamble * 3,
                            frame_symbols + 2 * _n_preamble + _training_symbols)
    _syndet_lookback  = int(cfg.get("syndet_lookback", _auto_lookback))
    syndet_min_snr_db = float(cfg.get("syndet_min_snr_db",  0.0))
    syndet_snr_ref_db = float(cfg.get("syndet_snr_ref_db", 12.0))
    syndet_lag_guard  = bool(cfg.get("syndet_lag_guard", True))
    syndet_alt_thr    = float(cfg.get("syndet_training_alt_frac", 0.85))
    syndet_max_lag    = int(cfg.get("syndet_max_lag",
                                    max(_training_symbols + _n_preamble + 4,
                                        2 * _n_preamble + 8)))
    syndet_bg_alpha   = float(cfg.get("syndet_bg_alpha", 0.02))
    syndet_margin     = float(cfg.get("syndet_margin",   0.15))
    syndet_thr_floor  = float(cfg.get("syndet_thr_floor", 0.45))

    post_acq_search_frac = float(cfg.get("post_acq_search_frac", 0.04))
    post_acq_search_smp  = max(16, int(sps * post_acq_search_frac))
    _dur_ref  = 0.25
    _eff_dur  = sps_eff / fs
    if _eff_dur > _dur_ref:
        alpha = max(0.005, alpha * (_dur_ref / _eff_dur))

    _retro_samples_cfg = int(cfg.get('max_retro_samples', 0))
    _max_retro_smp = (max(4_000_000, 24 * sps_eff) if _retro_samples_cfg == 0
                      else _retro_samples_cfg)
    _anchor_lookahead = int(cfg.get("retro_anchor_lookahead", 8))
    impulse_bg_alpha = float(cfg.get("impulse_bg_alpha", 0.05))
    syndet_flush_terr = int(cfg.get("syndet_flush_terr", sps // 2))
    _audio_hist_len = (frame_symbols + _training_symbols + _n_preamble + _anchor_lookahead + 8)
    retro_anchor_max_shift = int(cfg.get("retro_anchor_max_shift", _audio_hist_len - frame_symbols))

    if _n_preamble > MAX_PRE:
        raise ValueError(f"preamble longer than {MAX_PRE}")

    c.fs = fs; c.dur = dur; c.f_min = f_min; c.f_max = f_max
    c.sps = sps; c.gap_samps = gap_samps; c.sps_eff = sps_eff; c.buf_size = buf_size
    c.mf_hann = 1 if mf_window == "hann" else 0
    c.search_frac = search_frac
    c.silence_thr = silence_thr; c.sync_threshold = sync_threshold
    c.alpha = alpha; c.alpha_acq = alpha_acq; c.nread_revert = nread_revert
    c.rate_refine_alpha = rate_refine_alpha
    c.retro_rate_adopt_inlier = retro_rate_adopt_inlier
    c.wpk_track_enable = int(wpk_track_enable)
    c.wpk_track_k = wpk_track_k; c.wpk_track_alpha = wpk_track_alpha
    c.agc_target = agc_target; c.agc_attack = agc_attack
    c.agc_release = agc_release; c.agc_max = agc_max
    c.resync_after = resync_after
    c.min_tim_conf = min_tim_conf
    c.max_terr_smp = max_terr_smp; c.presync_max_terr_smp = presync_max_terr_smp
    c.tim_track_alpha = tim_track_alpha
    c.impulse_blank_enable = int(impulse_blank_enable)
    c.impulse_blank_k = impulse_blank_k; c.impulse_clip_floor = impulse_clip_floor
    c.impulse_blank_max_frac = impulse_blank_max_frac; c.impulse_bg_alpha = impulse_bg_alpha
    c.whiten_enable = int(whiten_enable)
    c.whiten_stft_win_s = whiten_stft_win_s; c.whiten_bg_alpha = whiten_bg_alpha
    c.whiten_redesign_every = whiten_redesign_every; c.whiten_taps = whiten_taps
    c.whiten_max_gain_db = whiten_max_gain_db
    c.timing_min_snr_db = timing_min_snr_db
    c.timing_acq_symbols = timing_acq_symbols; c.timing_median_n = timing_median_n
    c.timing_max_step_frac = timing_max_step_frac
    c.pll_weight = pll_weight; c.era_thr = era_thr
    c.post_acq_search_smp = post_acq_search_smp
    c.chunk_size = chunk_size; c.rs_t = rs_t; c.interleave_rows = interleave_rows
    c.conv_k7 = 1 if conv_code == "k7r12" else 0
    c.diff_enc = int(diff_enc)
    c.rs_enabled = 1 if rs_codec is not None else 0
    c.base_bits = base_bits; c.frame_symbols = frame_symbols; c.crc_bytes = crc_bytes
    c.preamble_per_frame = int(preamble_per_frame)
    c.n_preamble = _n_preamble
    for i, b in enumerate(sync_word):
        c.preamble[i] = 1 if b else 0
    c.training_symbols = _training_symbols
    c.syndet_lookback = _syndet_lookback
    c.syndet_min_snr_db = syndet_min_snr_db; c.syndet_snr_ref_db = syndet_snr_ref_db
    c.syndet_lag_guard = int(syndet_lag_guard)
    c.syndet_alt_thr = syndet_alt_thr; c.syndet_max_lag = syndet_max_lag
    c.syndet_bg_alpha = syndet_bg_alpha; c.syndet_margin = syndet_margin
    c.syndet_thr_floor = syndet_thr_floor
    c.syndet_flush_terr = syndet_flush_terr
    c.max_retro_smp = _max_retro_smp
    c.anchor_lookahead = _anchor_lookahead
    c.retro_anchor_max_shift = retro_anchor_max_shift
    c.nr_max = int(cfg.get("pll_nr_max", 0))
    c.syndet_conf_floor = float(cfg.get("syndet_conf_floor", 0.50))
    c.syndet_cov_knee = float(cfg.get("syndet_cov_knee", 1.00))
    c.syndet_mask_ineligible = int(bool(cfg.get("syndet_mask_ineligible", True)))
    c.impulse_silence_floor = float(cfg.get("impulse_silence_floor", 1e-6))
    c.whiten_stft_win_max = int(cfg.get("whiten_stft_win_max", sps_eff // 7))
    _sr = max(16, int(sps * search_frac))
    _rl = int(cfg.get("ring_lookahead", -1))
    _rl = max(gap_samps, _sr) if _rl < 0 else max(gap_samps, _rl)
    c.ring_lookahead = min(_rl, max(gap_samps, buf_size - 1 - sps - _sr))
    c.presync_full_period = int(bool(cfg.get("presync_full_period", True)))
    c.onset_guard_silence_syms = int(cfg.get("onset_guard_silence_syms", 2))
    c.step_recover = int(bool(cfg.get("step_recover", True)))
    c.step_drop_db = float(cfg.get("step_drop_db", 10.0))
    c.step_accept_db = float(cfg.get("step_accept_db", 6.0))
    c.step_min_conf = float(cfg.get("step_min_conf", 0.5))
    c.step_tol_smp = int(cfg.get("step_tol_smp", max(16, post_acq_search_smp // 2)))
    extras.update(fs=fs, dur=dur, sps=sps, sps_eff=sps_eff, chunk_size=chunk_size,
                  frame_symbols=frame_symbols, f_min=f_min, f_max=f_max,
                  rs_t=rs_t, gap_samps=gap_samps, alpha=alpha,
                  silence_thr=silence_thr, n_preamble=_n_preamble)
    return c, extras


# ──────────────────────────────────────────────────────────────────────────────
# The C decoder object
# ──────────────────────────────────────────────────────────────────────────────

class PllStream:
    """One C decoder instance.  on_frame(dict), on_symbol(dict), on_event(dict)
    are optional Python callables, invoked from inside push()/symbol()."""

    def __init__(self, cfg: dict, conf: dict, *, f64: bool = False,
                 on_frame=None, on_symbol=None, on_event=None):
        self.lib = load(f64)
        self.c, self.extras = build_cfg(cfg, conf)
        self._user_frame, self._user_symbol, self._user_event = on_frame, on_symbol, on_event
        self._cb = Callbacks()
        self._cb.user = None
        self._sym_cb = SYM_CB(self._on_symbol)
        self._frame_cb = FRAME_CB(self._on_frame)
        self._event_cb = EVENT_CB(self._on_event)
        self._cb.on_symbol = self._sym_cb if on_symbol is not None else SYM_CB()
        self._cb.on_frame = self._frame_cb
        self._cb.on_event = self._event_cb if on_event is not None else EVENT_CB()
        need = self.lib.css_pll_workspace(ctypes.byref(self.c))
        if need == 0:
            raise ValueError("css_pll_workspace rejected the configuration")
        self._mem = (ctypes.c_ubyte * need)()
        self._h = ctypes.c_void_p()
        rc = self.lib.css_pll_init(ctypes.byref(self._h), self._mem, need,
                                   ctypes.byref(self.c), ctypes.byref(self._cb))
        if rc != 0:
            raise RuntimeError(f"css_pll_init rc={rc}")
        self.workspace_bytes = need

    # -- callbacks -----------------------------------------------------------
    def _on_symbol(self, _u, sp):
        s = sp.contents
        self._user_symbol({k: getattr(s, k) for k, _ in Sym._fields_})

    def _on_frame(self, _u, fp):
        f = fp.contents
        data = bytes(ctypes.cast(f.data, ctypes.POINTER(ctypes.c_ubyte * f.ndata)).contents)
        if self._user_frame is not None:
            self._user_frame({"sym": f.sym, "seq": f.seq, "data": data,
                              "snr_db": f.snr_db, "rs_errors": f.rs_errors,
                              "source": SOURCES.get(f.source, str(f.source))})

    def _on_event(self, _u, kind, sym, a, b, x, y):
        self._user_event({"event": EVENT_NAMES.get(kind, str(kind)), "sym": sym,
                          "a": a, "b": b, "x": x, "y": y})

    # -- feeding -------------------------------------------------------------
    def next_nr(self) -> int:
        return self.lib.css_pll_next_nr(self._h)

    def symbol(self, raw: np.ndarray) -> None:
        x = np.ascontiguousarray(raw, dtype=np.float32)
        rc = self.lib.css_pll_symbol(self._h, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x))
        if rc != 0:
            raise RuntimeError(f"css_pll_symbol rc={rc} (n={len(x)} want {self.next_nr()})")

    def push(self, x: np.ndarray) -> int:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if not len(x):
            return 0
        return self.lib.css_pll_push(self._h, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x))

    def align(self, abs_peak: int, period: float = 0.0) -> int:
        """Pre-sync: put the next read's expected MF peak on a tracked chirp."""
        return self.lib.css_pll_align(self._h, int(abs_peak), float(period))

    def mark_onset(self, abs_onset: int) -> int:
        """A transmission's first chirp starts at abs_onset: anchor the early-sync guard there."""
        return self.lib.css_pll_mark_onset(self._h, int(abs_onset))

    def rearm(self) -> None:
        self.lib.css_pll_rearm(self._h)

    def diag(self) -> dict:
        d = Diag()
        self.lib.css_pll_get_diag(self._h, ctypes.byref(d))
        return {k: getattr(d, k) for k, _ in Diag._fields_}


# ──────────────────────────────────────────────────────────────────────────────
# decode(): the css_decoder.decode() contract on the C core
# ──────────────────────────────────────────────────────────────────────────────

def decode(cfg, stop_event=None, f64: bool | None = None):
    import sounddevice as sd
    import css_decoder as cd

    if f64 is None:
        f64 = os.environ.get("CSS_PLL_BACKEND", "").lower() == "c64"
    info_path   = Path(cfg.get("info", "css_info.json"))
    result_path = Path(cfg.get("result_out", "css_decode_result.json"))
    timeout_sec = float(cfg.get("timeout_sec", 600))
    device = cfg.get("device")
    if device is not None:
        try:
            device = int(device)
        except (TypeError, ValueError):
            device = None
    _on_frame = cfg.get("on_frame", None)
    stop_file = cfg.get("stop_file")
    trace = cfg.get("trace") if isinstance(cfg.get("trace"), list) else None
    result = {"status": "error", "failure_reason": None,
              "frames_decoded": 0, "frames_errored": 0,
              "decoded_text": "", "diagnostics": {}, "frames": []}
    if not info_path.exists():
        result["failure_reason"] = "no_info_file"
        result_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return result
    conf = json.loads(info_path.read_text())

    decoded_frames = []

    def on_frame(fr):
        msg = fr["data"].decode("utf-8", errors="replace")
        rec = {"seq": fr["seq"], "text": msg, "bytes_hex": fr["data"].hex(),
               "snr_db": round(fr["snr_db"], 1), "rs_errors": fr["rs_errors"],
               "source": fr["source"]}
        print(f"FRAME seq={fr['seq']} snr={fr['snr_db']:.1f}dB rs={fr['rs_errors']} "
              f"src={fr['source']} text={msg}", flush=True)
        decoded_frames.append(rec)
        if trace is not None:
            trace.append({"event": "FRAME", "sym": fr["sym"], "seq": fr["seq"],
                          "source": fr["source"], "hex": fr["data"].hex()})
        if _on_frame is not None and fr["source"] != "retro_deferred":
            try:
                _on_frame(dict(rec))
            except Exception as _ofe:
                print(f"on_frame hook error: {_ofe}", flush=True)

    def on_event(e):
        if e["event"] in ("SYNC_ACQUIRED", "RESYNC", "RETRO_GATE", "RETRO_SYNC",
                          "RETRO_ANCHOR", "SYNC_TOO_EARLY"):
            print(f"{e['event']} sym={e['sym']} a={e['a']} b={e['b']} "
                  f"x={e['x']:.3f} y={e['y']:.3f}", flush=True)
        if trace is not None:
            trace.append(dict(e))

    on_symbol = (lambda s: trace.append(s)) if trace is not None else None
    pll = PllStream(cfg, conf, f64=f64, on_frame=on_frame,
                    on_symbol=on_symbol, on_event=on_event)
    ex = pll.extras
    print(f"LISTENING backend=c{'64' if f64 else ''} fs={ex['fs']} sps={ex['sps']} "
          f"sps_eff={ex['sps_eff']} dur={ex['dur']}s f={ex['f_min']:.0f}-{ex['f_max']:.0f}Hz "
          f"frame_symbols={ex['frame_symbols']} alpha={ex['alpha']:.4f} "
          f"workspace={pll.workspace_bytes / 1e6:.1f}MB", flush=True)

    if hasattr(sd, "query_devices"):
        _input_ctx = cd._QueuedInputStream(sd, ex["fs"], device, channels=1)
    else:
        _input_ctx = sd.InputStream(samplerate=ex["fs"], channels=1, dtype="float32",
                                    device=device)
    started_at = time.time()
    input_overflows = 0
    try:
        with _input_ctx as stream:
            while True:
                if time.time() - started_at > timeout_sec:
                    break
                if stop_event is not None and stop_event.is_set():
                    break
                if stop_file and Path(stop_file).exists():
                    break
                nr = pll.next_nr()
                raw, overflowed = stream.read(nr)
                if overflowed:
                    input_overflows += 1
                pll.symbol(np.asarray(raw, dtype=np.float32).flatten())
    except KeyboardInterrupt:
        print("INTERRUPTED", flush=True)

    d = pll.diag()
    diag = {"sync_acquired": bool(d["sync_acquired"]),
            "sync_at_symbol": d["sync_at_symbol"] if d["sync_acquired"] else None,
            "sync_score": round(d["sync_score"], 3),
            "symbols_received": d["symbols"],
            "avg_snr_db": round(d["snr_sum"] / d["snr_n"], 1) if d["snr_n"] else 0,
            "input_overflows": input_overflows}
    if d["sync_acquired"]:
        diag["sync_lag"] = d["sync_lag"]
    if d["rms_n"]:
        diag["peak_rms"] = round(d["rms_max"], 6)
        diag["mean_rms"] = round(d["rms_sum"] / d["rms_n"], 6)
        diag["silence"] = diag["peak_rms"] < ex["silence_thr"]
    frames_decoded, frames_errored = d["frames_decoded"], d["frames_errored"]
    if diag.get("silence"):
        status, reason = "fail", f"no_audio (peak_rms={diag['peak_rms']:.6f})"
    elif not diag["sync_acquired"]:
        status, reason = "fail", f"no_sync (peak_rms={diag.get('peak_rms', 0):.4f})"
    elif frames_decoded == 0 and frames_errored > 0:
        status, reason = "fail", f"crc_fail ({frames_errored} frames)"
    elif frames_decoded == 0:
        status, reason = "fail", "sync acquired but no frames"
    else:
        expected_n = conf.get("frames")
        if expected_n and frames_decoded < expected_n:
            status, reason = "partial", f"{frames_decoded}/{expected_n} frames"
        else:
            status, reason = "ok", None
    result.update({"status": status, "failure_reason": reason,
                   "frames_decoded": frames_decoded, "frames_errored": frames_errored,
                   "diagnostics": diag, "frames": decoded_frames,
                   "backend": "c64" if f64 else "c"})
    if decoded_frames:
        result["decoded_text"], _raw_msg = cd.assemble_message(
            decoded_frames, ex["chunk_size"], conf.get("message_len"))
        result["decoded_hex"] = _raw_msg.hex()
    print(f"DONE status={status} frames={frames_decoded} "
          f"avg_snr={diag['avg_snr_db']}dB backend=c", flush=True)
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# css_trk: running local PLL bank + chirp classifier
# ──────────────────────────────────────────────────────────────────────────────

TRK_LABELS = {0: "UNKNOWN", 1: "OWN", 2: "PEER", 3: "ECHO"}
TRK_EVENTS = {1: "NEW", 2: "LOCK", 3: "LOST", 4: "PHASE_STEP", 5: "COLLISION",
              6: "ECHO", 7: "OWN_LAG", 8: "PEER", 9: "BUSY"}


class TrkCfg(ctypes.Structure):
    _fields_ = [
        ("fs", ctypes.c_int),
        ("dur", ctypes.c_double), ("f_min", ctypes.c_double), ("f_max", ctypes.c_double),
        ("gap_frac", ctypes.c_double),
        ("peak_thr", ctypes.c_double), ("rel_floor", ctypes.c_double),
        ("acq_alpha", ctypes.c_double), ("alpha", ctypes.c_double), ("rate_alpha", ctypes.c_double),
        ("acq_hits", ctypes.c_int),
        ("settle_frac", ctypes.c_double), ("acq_frac", ctypes.c_double),
        ("lost_syms", ctypes.c_int),
        ("echo_min_s", ctypes.c_double), ("echo_max_s", ctypes.c_double),
        ("echo_tol_frac", ctypes.c_double), ("echo_min_db", ctypes.c_double),
        ("echo_hits", ctypes.c_int),
        ("max_ppm", ctypes.c_double),
        ("phase_m", ctypes.c_int),
        ("collide_db", ctypes.c_double),
        ("own_max_lag_s", ctypes.c_double),
        ("cal_s", ctypes.c_double),
        ("carrier_k", ctypes.c_double), ("carrier_lo", ctypes.c_double), ("carrier_hi", ctypes.c_double),
        ("busy_hold_s", ctypes.c_double),
        ("smooth_n", ctypes.c_int),
    ]


class TrkChirp(ctypes.Structure):
    _fields_ = [("abs_peak", ctypes.c_long), ("dir", ctypes.c_int),
                ("mag", ctypes.c_double), ("score", ctypes.c_double),
                ("grid", ctypes.c_int), ("label", ctypes.c_int), ("peer", ctypes.c_int),
                ("echo_of", ctypes.c_int), ("residual", ctypes.c_double)]


class TrkGrid(ctypes.Structure):
    _fields_ = [("active", ctypes.c_int), ("id", ctypes.c_int), ("label", ctypes.c_int),
                ("peer", ctypes.c_int), ("echo_of", ctypes.c_int), ("settled", ctypes.c_int),
                ("hits", ctypes.c_int), ("misses", ctypes.c_int),
                ("t0", ctypes.c_double), ("period", ctypes.c_double), ("ppm", ctypes.c_double),
                ("power_db", ctypes.c_double), ("delay_s", ctypes.c_double),
                ("gain_db", ctypes.c_double), ("last_residual", ctypes.c_double),
                ("last_abs", ctypes.c_long)]


TRK_CHIRP_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(TrkChirp))
TRK_EVENT_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_long,
                                ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double)


class TrkCallbacks(ctypes.Structure):
    _fields_ = [("user", ctypes.c_void_p), ("on_chirp", TRK_CHIRP_CB), ("on_event", TRK_EVENT_CB)]


def _bind_trk(lib):
    if getattr(lib, "_trk_bound", False):
        return lib
    lib.css_trk_defaults.restype = None
    lib.css_trk_defaults.argtypes = [ctypes.POINTER(TrkCfg), ctypes.c_int, ctypes.c_double,
                                     ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.css_trk_workspace.restype = ctypes.c_size_t
    lib.css_trk_workspace.argtypes = [ctypes.POINTER(TrkCfg)]
    lib.css_trk_init.restype = ctypes.c_int
    lib.css_trk_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_size_t,
                                 ctypes.POINTER(TrkCfg), ctypes.POINTER(TrkCallbacks)]
    lib.css_trk_push.restype = ctypes.c_int
    lib.css_trk_push.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.css_trk_tx.restype = ctypes.c_int
    lib.css_trk_tx.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_ubyte)]
    lib.css_trk_decoder_peak.restype = None
    lib.css_trk_decoder_peak.argtypes = [ctypes.c_void_p, ctypes.c_long]
    lib.css_trk_decoder_frame.restype = None
    lib.css_trk_decoder_frame.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.css_trk_busy.restype = ctypes.c_int
    lib.css_trk_busy.argtypes = [ctypes.c_void_p]
    lib.css_trk_score.restype = ctypes.c_double
    lib.css_trk_score.argtypes = [ctypes.c_void_p]
    lib.css_trk_abs.restype = ctypes.c_long
    lib.css_trk_abs.argtypes = [ctypes.c_void_p]
    lib.css_trk_own_lag.restype = ctypes.c_double
    lib.css_trk_own_lag.argtypes = [ctypes.c_void_p]
    lib.css_trk_get_grid.restype = ctypes.c_int
    lib.css_trk_get_grid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(TrkGrid)]
    lib._trk_bound = True
    return lib


class TrkStream:
    """Running PLL bank + chirp classifier on one RX stream.

    on_chirp(dict) fires for every detected chirp with its label; on_event(dict)
    for NEW / LOCK / LOST / PHASE_STEP / COLLISION / ECHO / OWN_LAG / PEER / BUSY.
    `overrides` sets any TrkCfg field by name."""

    def __init__(self, fs, dur, f_min, f_max, gap_frac, *, f64=False,
                 on_chirp=None, on_event=None, **overrides):
        self.lib = _bind_trk(load(f64))
        self.c = TrkCfg()
        self.lib.css_trk_defaults(ctypes.byref(self.c), int(fs), float(dur), float(f_min),
                                  float(f_max), float(gap_frac))
        for k, v in overrides.items():
            setattr(self.c, k, v)
        self._uc, self._ue = on_chirp, on_event
        self._cb = TrkCallbacks()
        self._ccb = TRK_CHIRP_CB(self._on_chirp)
        self._ecb = TRK_EVENT_CB(self._on_event)
        self._cb.on_chirp = self._ccb
        self._cb.on_event = self._ecb
        need = self.lib.css_trk_workspace(ctypes.byref(self.c))
        if need == 0:
            raise ValueError("css_trk_workspace rejected the configuration")
        self._mem = (ctypes.c_ubyte * need)()
        self._h = ctypes.c_void_p()
        rc = self.lib.css_trk_init(ctypes.byref(self._h), self._mem, need,
                                   ctypes.byref(self.c), ctypes.byref(self._cb))
        if rc != 0:
            raise RuntimeError(f"css_trk_init rc={rc}")
        self.fs = int(fs)

    def _on_chirp(self, _u, cp):
        if self._uc is None:
            return
        c = cp.contents
        self._uc({"abs": c.abs_peak, "t": c.abs_peak / self.fs, "dir": c.dir, "mag": c.mag,
                  "score": c.score, "grid": c.grid, "label": TRK_LABELS.get(c.label, "?"),
                  "peer": c.peer, "echo_of": c.echo_of, "residual": c.residual})

    def _on_event(self, _u, kind, abs_, a, b, x, y):
        if self._ue is None:
            return
        self._ue({"event": TRK_EVENTS.get(kind, str(kind)), "abs": abs_, "t": abs_ / self.fs,
                  "a": a, "b": b, "x": x, "y": y})

    def push(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if len(x):
            self.lib.css_trk_push(self._h, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x))

    def tx(self, abs_first, dirs):
        d = (ctypes.c_ubyte * max(1, len(dirs)))(*[1 if b else 0 for b in dirs])
        self.lib.css_trk_tx(self._h, int(abs_first), len(dirs), d)

    def decoder_peak(self, abs_peak):
        self.lib.css_trk_decoder_peak(self._h, int(abs_peak))

    def decoder_frame(self, src):
        self.lib.css_trk_decoder_frame(self._h, int(src))

    @property
    def busy(self) -> bool:
        return bool(self.lib.css_trk_busy(self._h))

    @property
    def abs(self) -> int:
        return int(self.lib.css_trk_abs(self._h))

    def own_lag(self):
        v = self.lib.css_trk_own_lag(self._h)
        return None if v < 0 else v

    def grids(self) -> list[dict]:
        out = []
        for i in range(8):
            g = TrkGrid()
            self.lib.css_trk_get_grid(self._h, i, ctypes.byref(g))
            if g.active:
                out.append({k: getattr(g, k) for k, _ in TrkGrid._fields_}
                           | {"label": TRK_LABELS.get(g.label, "?")})
        return out
