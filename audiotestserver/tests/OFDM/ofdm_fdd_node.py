"""
ofdm_fdd_node.py -- FEASIBILITY probe for frequency-division duplex OFDM.

Assign to exactly TWO clients.  This answers ONE question before any duplex
data plane gets written:

    If both ends transmit AT THE SAME TIME on disjoint sub-bands, does either
    end still hear the other?

NOT the same test as ofdm2way_node.py.  That one is "two-way" in TIME -- probe
1->2, then probe 2->1, each over the FULL band, strictly half-duplex.  This one
splits the band by DIRECTION and has both speakers live at once.  Neither
supersedes the other.

WHY THIS MIGHT NOT WORK, which is the whole point of measuring it:
  - Self-interference.  A node's own speaker is ~0 cm from its own mic and the
    far node is metres away, so own-TX arrives tens of dB hotter than the
    signal being decoded.  Nothing here cancels it; the ONLY thing keeping the
    bands apart is that the FFT puts them in different bins.
  - Speaker/mic nonlinearity.  A loud DL band produces harmonic and
    intermodulation products at frequencies it was never given.  Those land in
    the UL band as noise that scales with DL level -- and an asymmetric split
    (dl_frac=0.9) is the worst case, because the loud direction is wide and the
    direction being buried is narrow.  A LINEAR system would show zero penalty;
    the penalty measured here IS the nonlinearity.
  - AGC.  Many USB devices ride gain on total input power.  Own-TX dominates
    that, so the far signal can be gain-ridden down into the noise even with
    perfect band isolation.

METHOD -- each direction is measured twice, and the DIFFERENCE is the result:
  Phase SOLO   (baseline, half-duplex):  one node transmits its sub-band probe
                                         while the other is silent.
  Phase DUPLEX (the real question):      both nodes transmit their own
                                         sub-band probe simultaneously.
  penalty_db = solo_snr - duplex_snr, per carrier and aggregate.

A small penalty (~0-3 dB) means FDD is worth building.  A large one, or a
duplex probe that fails to acquire at all, means own-TX leakage dominates and
a simultaneous-duplex mode is not viable on this hardware at these levels --
which is a cheap thing to learn now and an expensive thing to learn after the
data plane exists.

MEASURED ASYMMETRY -- runs 20260810_190457 (400-4000) and _190559 (400-8000),
both nodes, and the reason the defaults are what they are:

  Duplex damage flows UPWARD in frequency, essentially one-way.

  A node transmitting on the LOW band splatters into the HIGH band it is
  listening to: penalty 9.9 dB median at 250-500 Hz above its own upper edge,
  3.6-3.9 dB at 500-1000, 1.4 dB at 1000-1500, 0.4 dB beyond 1500.  A node
  transmitting on the HIGH band does almost nothing to the LOW band it is
  listening to: 0.0-1.5 dB flat, several buckets slightly NEGATIVE (noise).
  Same hardware pair, same levels, both runs, monotonic in separation -- this
  is harmonic/intermodulation splatter, which only ever lands above its
  source, not a property of either band's own quality.

  Two consequences, both baked into the defaults:

  1. dl_side='high'.  In a two-band split each node transmits on one band and
     listens on the other, so ONE direction always eats the splatter -- the
     one whose carriers sit on the HIGH band, because its receiver is the node
     transmitting low.  That damage should land on the WIDE direction, which
     has carriers to spare (run _190559: 137 active, lost 1-12), never on the
     narrow one, which has none.  So the wide/fast direction goes HIGH and the
     narrow/slow direction goes LOW, where it is received cleanly.
     Note this is the opposite of what raw SNR alone suggests: 400-4000 Hz
     measured uniformly better (16-21 dB) than 4500-8000 (7-14 dB), so the
     high band is the WORSE band -- it is still the right place for the wide
     direction, because what decides the split is who can absorb the splatter,
     not who has the best carriers.

  2. guard_hz=1200.  The 300 Hz this test originally shipped with puts the far
     band's first carriers squarely in the 10 dB damage zone.

CAVEAT on the band, unresolved: above ~4.5 kHz raw SNR falls off badly on this
hardware (7-14 dB, and 7.4 dB by 7-7.5 kHz, with 96 of 163 carriers dead).
Widening freq_hi to 8000 bought bandwidth that mostly could not carry bits.
Whether that is the room, the speakers, or the mics is not established here.

Also reported: `comb` per direction, an autocorrelation of the per-carrier SNR
curve.  Room reflections produce notches at a REGULAR frequency spacing
(spacing_hz ~= c / path_difference); a strong periodic component is direct
evidence for that, and its spacing says how wide a guard band has to be to
avoid parking a sub-band edge in a null.  Nothing else in this codebase
measures it -- ofdm_modem's isi_ratio() collapses reverberation to ONE
broadband scalar (beta), which by construction cannot show frequency
structure.

Band plan (see plan_duplex_bands): the usable span minus `guard_hz` is split
dl_frac / (1 - dl_frac).  Neither direction is allowed below `min_baud`, so a
lopsided ratio gets clamped rather than silently producing an uplink of zero
carriers.  Both nodes compute the identical plan from identical PARAMS.
"""
# Annotations are evaluated at def-time without this, so `-> dict | None` is a
# runtime TypeError on the Python 3.9 one of the nodes runs (macdesktop, run
# 20260810_194631).  Same reason ofdm_modem.py and ofdm_handshake.py carry it.
from __future__ import annotations

import hashlib
import json
import math
import random
import string
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import butter, ellip, ellipord, resample_poly, sosfiltfilt

# --- category-aware sys.path bootstrap (auto-inserted) ---
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_TESTS_ROOT = _HERE.parent if _HERE.name in ("CSS", "MESH", "OFDM", "QAM") else _HERE
_REPO_ROOT = _TESTS_ROOT.parent
for _p in (_TESTS_ROOT, _REPO_ROOT, _TESTS_ROOT / "CSS", _TESTS_ROOT / "MESH",
           _TESTS_ROOT / "OFDM", _TESTS_ROOT / "QAM", _TESTS_ROOT / "regression"):
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# --- end bootstrap ---

from testlib import (                                           # noqa: E402
    CLIENT_ID, INPUT_DEV, OUTPUT_DEV, RUN_DIR, PARAMS, RUN_ID,
    log, to_int_dev, coord_signal, coord_wait_any, coord_check, coord_check_ts,
    Recorder, play_blocking,
)
from ofdm_modem import (                                       # noqa: E402
    build_config, Modem, carrier_spectrum, SNR_FLOOR_DB,
    AdaptiveDemodulator, _sc_metric, _find_sc_peaks, _frame_min_gap,
)
# The FDD core lives in a library so the meshmodem full-duplex phy can use the
# same band planning, frame sizing and RX filter this test validated over the
# air.  These wrappers keep the old private names (and this script's PARAMS
# knobs, which are what the web UI scrapes to build its form) while there is
# exactly ONE implementation of the physics underneath.
from ofdm_fdd import (                                         # noqa: E402
    plan_duplex_bands, min_occupied_for_baud as _min_occupied_for_baud,
    chunk_for_frame_len as _chunk_for_frame_len,
    size_for_airtime as _size_for_airtime,
    make_payload as _make_payload, sum16 as _sum16,
)
import ofdm_fdd as _fdd                                        # noqa: E402


def _rx_filter(x, fs: int, far_cfg: dict, my_cfg: dict, order: int = 6):
    """This test's RX-filter selectivity knobs, applied to the shared filter."""
    return _fdd.rx_filter(x, fs, far_cfg, my_cfg, order=order,
                          ripple_db=RX_FILT_RIPPLE_DB,
                          stop_db=RX_FILT_STOP_DB,
                          max_order=RX_FILT_MAX_ORDER)

DEFAULT_ID_MAP = {"macdesktop": 1, "laptop": 2}

# RMS target, NOT peak.  Values above ~0.17 cannot be met by a sparse data
# frame (crest ~5.9x) and above ~0.22 not by the probe (crest ~4.5x): the burst
# gets scaled to the peak ceiling instead, so the knob stops doing anything and
# the transmitter sits at full scale, which is exactly where speaker
# nonlinearity manufactures the harmonics that wipe out the upper sub-band.
# Default 0.15 keeps BOTH bursts under the ceiling, so the setting stays live
# and probe/data go out at the same power.  Lower it further to trade SNR for
# less intermodulation.
AMP        = float(PARAMS.get("amplitude",    0.15))  # group: Level
# Leave a sliver of headroom rather than normalising to exactly 1.0.
PEAK_CEIL  = float(PARAMS.get("peak_ceiling", 0.95))  # group: Level
# RX filter selectivity. The guard band is the whole transition budget, so
# these decide whether own TX (~50 dB up in this node's own mic) is actually
# rejected before the SC correlation runs. 60 dB stop was what turned 1/5
# decoded frames into 4/5 on run 20260811_031659.
RX_FILT_STOP_DB   = float(PARAMS.get("rx_filt_stop_db",   60.0))  # group: Level
RX_FILT_RIPPLE_DB = float(PARAMS.get("rx_filt_ripple_db",  0.5))  # group: Level
RX_FILT_MAX_ORDER = int(PARAMS.get("rx_filt_max_order",     12))  # group: Level
SC_THRESH  = float(PARAMS.get("sc_threshold", 0.25))
TAIL_S     = float(PARAMS.get("tail_s",       1.5))
LEG_TMO    = float(PARAMS.get("leg_timeout",  90.0))
# Seconds between the barrier firing and audio starting.  Must exceed the
# spread in coord poll latency across nodes, or the "simultaneous" phase is
# only partly overlapped and the penalty reads better than it really is.
START_LEAD_S = float(PARAMS.get("start_lead_s", 3.0))
# Target airtime for the data phase, per direction.  Both nodes send for about
# this long regardless of rate, so a fast leg simply carries more bytes -- the
# comparison stays fair in TIME, which is what a duplex pipe actually shares.
# 0 disables the data phase and leaves the run probe-only.
DATA_S = float(PARAMS.get("data_s", 10.0))  # group: Data phase
# Ceiling on a single frame's airtime.  A frame is one RS codeword -- it
# decodes whole or not at all -- so its length must fit inside how long the
# room stays coherent, NOT inside the airtime budget.  See _chunk_for_frame_len.
MAX_FRAME_S = float(PARAMS.get("max_frame_s", 2.0))  # group: Data phase

# Which end of the band carries the downlink (the WIDE/fast direction, since
# dl_frac >= 0.5 by convention).  Default 'high', which is not the obvious
# choice -- see MEASURED ASYMMETRY in the module docstring.  In short: duplex
# damage flows UPWARD in frequency only, so whichever direction sits on the
# HIGH band is the one that gets hurt.  Put the wide direction there, because
# it has carriers to spare; the narrow direction cannot afford to lose any.
# Set 'low' to reproduce runs 20260810_190457 / _190559.
DL_SIDE = str(PARAMS.get("dl_side", "high"))  # choices: high, low  # group: Duplex plan


# --------------------------------------------------------------------------- #
# Band planning
# --------------------------------------------------------------------------- #

def _band_cfg(band_hz, sample_rate: int) -> dict:
    """This test's OFDM knobs, applied to the shared per-direction config.

    The PARAMS.get() calls stay HERE rather than in ofdm_fdd because the web UI
    builds its form by scraping these literals out of the script it is about to
    run -- a knob that only exists in the library never appears in the UI.
    """
    return _fdd.band_cfg(
        band_hz, sample_rate,
        n_fft            = int(PARAMS.get("n_fft",     1024)),  # choices: 512, 1024, 2048, 4096  # group: OFDM
        cp_ratio         = float(PARAMS.get("cp_ratio", 0.25)),  # choices: 0.25, 0.375, 0.5  # group: OFDM
        pilot_spacing    = int(PARAMS.get("pilot_spacing", 8)),  # group: OFDM
        probe_duration_s = float(PARAMS.get("probe_duration_s", 5.0)),  # group: OFDM
        snr_margin_db    = float(PARAMS.get("snr_margin_db", 6.0)),  # group: OFDM
        max_bits_per_bin = int(PARAMS.get("max_bits_per_bin", 6)),  # choices: 1, 2, 4, 6  # group: OFDM
        probe_meas_syms  = int(PARAMS.get("probe_meas_syms", 32)),  # group: OFDM
        # ON by default here, unlike the other OFDM tests -- see ofdm_fdd.band_cfg.
        isi_derate       = float(PARAMS.get("isi_derate", 1)),  # choices: 0, 1  # group: OFDM
    )


def _plan() -> dict:
    return plan_duplex_bands(
        freq_lo       = float(PARAMS.get("freq_lo",  400.0)),  # group: Duplex plan
        freq_hi       = float(PARAMS.get("freq_hi", 4000.0)),  # group: Duplex plan
        dl_frac       = float(PARAMS.get("dl_frac",    0.5)),  # choices: 0.5, 0.6, 0.75, 0.9  # group: Duplex plan
        # 1200, not the 300 this test shipped with.  Measured: duplex damage
        # is concentrated within ~1 kHz above the interferer's upper edge
        # (9.9 dB median at 250-500 Hz separation, 3.6-3.9 at 500-1000, 1.4 at
        # 1000-1500, 0.4 beyond).  A 300 Hz guard parks the far band's first
        # carriers exactly in the 10 dB zone -- that is what cost run
        # 20260810_190457 fifteen carriers.
        guard_hz      = float(PARAMS.get("guard_hz", 1200.0)),  # group: Duplex plan
        min_baud      = float(PARAMS.get("min_baud",  30.0)),  # group: Duplex plan
        n_fft         = int(PARAMS.get("n_fft", 1024)),
        cp_ratio      = float(PARAMS.get("cp_ratio", 0.25)),
        # NOMINAL rate, deliberately not this device's spk_fs.  Both nodes must
        # land on byte-identical band edges or they transmit past each other,
        # and sample_rate feeds the min_baud floor -- so planning at each host's
        # own rate would let two devices with different native rates compute
        # different splits from identical PARAMS.  Band edges are in Hz and so
        # are rate-independent; only the floor arithmetic reads this.
        sample_rate   = 48000,
        pilot_spacing = int(PARAMS.get("pilot_spacing", 8)),
        dl_side       = DL_SIDE,
    )


# --------------------------------------------------------------------------- #
# Measurement helpers
# --------------------------------------------------------------------------- #

def _comb_metric(freq_hz, snr_db) -> dict:
    """Look for a REGULAR notch spacing in the per-carrier SNR curve.

    A single reflection at path difference d puts nulls every c/d Hz, so a
    comb shows up as a periodic component in SNR-vs-frequency.  Detrend
    (the raw curve is dominated by the speaker/mic response rolling off, which
    is not a comb), autocorrelate, and report the strongest non-trivial lag.

    `peak` is the normalised autocorrelation at that lag.  Read it WITH
    `ripple_db_rms`, never alone -- argmax always returns some lag, so a
    featureless curve still reports a spacing.  Against synthetic references
    at this carrier spacing: noise-only and rolloff-only score peak 0.25-0.30
    at ~0.9 dB ripple, while an injected comb scores 0.83 at 4.5-6.8 dB and
    recovers its true spacing to within one carrier.  So peak > ~0.5 AND
    ripple > ~2 dB is a real comb; below that the reverberation is acting as
    the broadband floor ofdm_modem's beta already models, and `spacing_hz` is
    meaningless.
    """
    y = np.asarray(snr_db, dtype=float)
    f = np.asarray(freq_hz, dtype=float)
    if len(y) < 16:
        return {"peak": None, "spacing_hz": None}
    # Detrend with a wide moving average so only ripple survives.
    w = max(5, len(y) // 8)
    kern = np.ones(w) / w
    trend = np.convolve(np.pad(y, (w // 2, w // 2), mode="edge"), kern,
                        mode="same")[w // 2: w // 2 + len(y)]
    r = y - trend
    if not np.any(r):
        return {"peak": None, "spacing_hz": None}
    ac = np.correlate(r, r, mode="full")[len(r) - 1:]
    ac /= (ac[0] or 1.0)
    lo = 2                                   # lag 0/1 are trivially correlated
    hi = max(lo + 1, len(ac) // 2)
    seg = ac[lo:hi]
    if not len(seg):
        return {"peak": None, "spacing_hz": None}
    k = int(np.argmax(seg)) + lo
    df = float(np.median(np.diff(f))) if len(f) > 1 else 0.0
    return {"peak": round(float(seg[k - lo]), 3),
            "spacing_hz": round(k * df, 1),
            "ripple_db_rms": round(float(np.sqrt(np.mean(r ** 2))), 2)}


def _native_rates(in_dev, out_dev):
    try:
        from shared import native_samplerate
        mic = int(native_samplerate("input",  in_dev)  or 48000)
        spk = int(native_samplerate("output", out_dev) or 48000)
    except Exception as exc:
        log(f"  native-rate probe failed ({exc}); assuming 48000")
        mic = spk = 48000
    return mic, spk


def _resample_to(cap: np.ndarray, mic_fs: int, tx_fs: int) -> np.ndarray:
    if tx_fs == mic_fs or len(cap) == 0:
        return cap.astype(np.float32)
    g = gcd(tx_fs, mic_fs)
    return resample_poly(cap.astype(np.float64), tx_fs // g, mic_fs // g).astype(np.float32)


# --------------------------------------------------------------------------- #
# Mic auto-gain
# --------------------------------------------------------------------------- #
#
# DIGITAL MIC GAIN IS OFF BY DEFAULT, AND SHOULD STAY THAT WAY.
#
# It was added to spare the operator from setting levels by hand, and it does
# not work, for two independent reasons -- both measured on run
# 20260811_033012:
#
# 1. It cannot help.  Gain applied AFTER the ADC scales signal and noise by the
#    same factor, so it changes no SNR and recovers no information.  The only
#    case it would help is a capture so quiet that ADC quantisation dominates,
#    and the signal of interest here sits far above ambient, not near the LSB.
#
# 2. It actively destroys the capture.  Calibration can only run on AMBIENT --
#    before either node transmits -- but a node's own speaker then arrives
#    ~50-60 dB above that ambient in its own mic.  Setting gain to lift the
#    ambient floor therefore guarantees the loud part clips: laptop measured
#    -61 dBFS ambient, took +16 dB, and clipped 50.8% OF ALL SAMPLES during
#    its own data burst.  Replaying that same recording with gain OFF decoded
#    4 of 4 frames with a full checksum match; the live run with gain ON got 1.
#
# What IS kept is the MEASUREMENT: the ambient level is still reported, and a
# hot or dead input still produces an explicit, actionable warning. That is the
# part an operator can act on -- OS input level and mic placement -- because
# sounddevice/PortAudio exposes no OS input-gain control to set from here.
MIC_AUTO_GAIN = bool(float(PARAMS.get("mic_auto_gain", 0)))  # choices: 0, 1  # group: Level
MIC_GAIN_LIN = 1.0
# Target RMS for the CHANNEL a probe/data burst arrives on, referenced to the
# ambient floor: real OFDM signal peaks run ~2.5-3x their RMS at these crest
# factors (see _play), so -20 dBFS RMS ambient leaves headroom before a live
# burst's peaks would approach 0 dBFS.
MIC_TARGET_AMB_DBFS = float(PARAMS.get("mic_target_amb_dbfs", -45.0))  # group: Level
MIC_MAX_GAIN_DB     = float(PARAMS.get("mic_max_gain_db",      24.0))  # group: Level
# Manual trim on top of the auto gain -- 0 unless a specific mic is known to
# need correcting beyond what auto-gain alone should attempt.
MIC_GAIN_DB_MANUAL  = float(PARAMS.get("mic_gain_db",            0.0))  # group: Level


def _apply_mic_gain(cap: np.ndarray) -> np.ndarray:
    """Scale only -- NOT clipped here.  A hard-clipped edge fed into the
    polyphase resampler below rings (Gibbs), which can overshoot past the clip
    level it started from; the honest clip is applied once, after resampling,
    in `_cap`."""
    if MIC_GAIN_LIN == 1.0:
        return cap
    return (cap.astype(np.float64) * MIC_GAIN_LIN).astype(np.float32)


def _cap(rec, start: int, mic_fs: int, tx_fs: int) -> np.ndarray:
    """The one path every phase reads its capture through: gain (pre-resample,
    so quantisation from a weak signal is boosted before the resample filter's
    rounding), resample, THEN clip -- so nothing downstream ever sees a sample
    outside [-1, 1], including resampler overshoot on a boosted loud burst."""
    boosted   = _apply_mic_gain(rec.data()[start:])
    resampled = _resample_to(boosted, mic_fs, tx_fs)
    return np.clip(resampled, -1.0, 1.0).astype(np.float32)


def _calibrate_mic_gain(rec, mic_fs: int, settle_s: float = 0.6) -> dict:
    """Measure the ambient floor and set MIC_GAIN_LIN so it lands near
    MIC_TARGET_AMB_DBFS.  Runs once, before either end has transmitted
    anything, so the measurement is genuinely ambient, not self-TX bleed.

    A floor that is ALREADY hot (clipping or within a few dB of 0 dBFS) is not
    a gain problem -- boosting it only clips harder -- so that case gets a
    loud, actionable message instead of a gain value, mirroring meshmodem's
    amb_rms_max check for the same failure.
    """
    global MIC_GAIN_LIN
    time.sleep(settle_s)
    amb = rec.data()
    if len(amb) == 0:
        log("  [mic] WARNING: no ambient samples captured; leaving gain at 0dB")
        return {"amb_rms_dbfs": None, "gain_db": 0.0, "clipping": False}

    peak = float(np.max(np.abs(amb)))
    rms = float(np.sqrt(np.mean(amb.astype(np.float64) ** 2))) or 1e-9
    rms_dbfs = 20 * math.log10(rms)
    clip_pct = 100.0 * float(np.mean(np.abs(amb) > 0.985))

    if clip_pct > 0.5 or peak > 0.99:
        log(f"  [mic] WARNING: ambient input is already hot (peak={peak:.2f}, "
            f"clip {clip_pct:.1f}% of samples, {rms_dbfs:.1f} dBFS RMS) -- "
            f"digital gain cannot fix a saturated ADC. Lower the OS input "
            f"level (System Settings > Sound > Input, or disable mic "
            f"auto-gain) before the next run.")
        MIC_GAIN_LIN = 1.0
        return {"amb_rms_dbfs": round(rms_dbfs, 1), "gain_db": 0.0,
                "clipping": True}

    if not MIC_AUTO_GAIN:
        # Report only.  Gain is deliberately not applied -- see the block
        # comment above for why it cannot help and does harm.
        MIC_GAIN_LIN = 10 ** (MIC_GAIN_DB_MANUAL / 20.0) if MIC_GAIN_DB_MANUAL else 1.0
        hint = ""
        if rms_dbfs < -70:
            hint = "  (very quiet -- raise the OS input level or move closer)"
        elif rms_dbfs > -25:
            hint = "  (hot -- lower the OS input level before headroom runs out)"
        log(f"  [mic] ambient {rms_dbfs:.1f} dBFS RMS, peak {peak:.3f}, "
            f"digital gain {MIC_GAIN_DB_MANUAL:+.1f} dB{hint}")
        return {"amb_rms_dbfs": round(rms_dbfs, 1),
                "gain_db": round(MIC_GAIN_DB_MANUAL, 1), "clipping": False,
                "auto_gain": False}

    need_db = MIC_TARGET_AMB_DBFS - rms_dbfs
    gain_db = float(np.clip(need_db, 0.0, MIC_MAX_GAIN_DB)) + MIC_GAIN_DB_MANUAL
    MIC_GAIN_LIN = 10 ** (gain_db / 20.0)
    clamped = need_db > MIC_MAX_GAIN_DB
    note = (" (CLAMPED -- mic may just be far from both speakers)"
            if clamped else "")
    log(f"  [mic] auto-gain ENABLED: ambient {rms_dbfs:.1f} dBFS RMS -> "
        f"+{gain_db:.1f} dB{note}  *** this clips once TX starts; "
        f"see MIC_AUTO_GAIN comment ***")
    return {"amb_rms_dbfs": round(rms_dbfs, 1), "gain_db": round(gain_db, 1),
            "clipping": False, "clamped": clamped, "auto_gain": True}


def _play(sig, fs, out_dev, label=""):
    """Play at a level set by RMS, not peak.

    `amplitude` is an RMS target, because peak-normalising a probe and a data
    burst to the same number sends them at different POWER and a penalty
    measured that way is partly just a level difference.

    THE CEILING IS REAL AND IT BITES.  An OFDM burst has a large crest factor
    (measured here: 4.5x for the probe, 5.9x for a sparse data frame), so an
    RMS target above ~1/crest cannot be met -- the old code quietly rescaled to
    peak 1.0, which made every amplitude above ~0.22 produce BYTE-IDENTICAL
    full-scale output.  Run 20260810_203042 asked for 0.8 and got exactly what
    0.3 would have given: 0 dBFS, the level where the speaker's own
    nonlinearity is worst, which is what feeds the harmonics that were wiping
    out the upper sub-band.  Worse, probe and data have DIFFERENT crest
    factors, so they rescaled by different amounts and data went out ~2.4 dB
    quieter than the probe that graded the channel.

    So: report the level actually emitted, every time, and say plainly when the
    request was unmeetable instead of silently substituting a different one.
    """
    rms = float(np.sqrt(np.mean(sig.astype(np.float64) ** 2))) or 1e-9
    crest = float(np.max(np.abs(sig))) / rms
    out = (AMP * sig / rms).astype(np.float32)
    peak = float(np.max(np.abs(out)))
    if peak > PEAK_CEIL:
        out = (out * (PEAK_CEIL / peak)).astype(np.float32)
        eff = AMP * PEAK_CEIL / peak
        log(f"  LEVEL{label}: amplitude={AMP:.3f} UNREACHABLE (crest {crest:.1f}x "
            f"would peak at {peak:.2f}); sent RMS {eff:.3f} "
            f"({20 * math.log10(eff):+.1f} dBFS), peak {PEAK_CEIL:.2f}. "
            f"Max usable amplitude for this burst is ~{PEAK_CEIL / crest:.3f} "
            f"-- above that the setting does nothing and probe/data levels "
            f"diverge.")
    else:
        log(f"  LEVEL{label}: RMS {AMP:.3f} ({20 * math.log10(max(AMP,1e-9)):+.1f} "
            f"dBFS), peak {peak:.2f}, crest {crest:.1f}x")
    return play_blocking(out, fs, out_dev)


def _barrier(signal: str, n_expected: int, lead_s: float, timeout: float) -> bool:
    """Wait for all nodes, then sleep so everyone starts audio together.

    Uses the SERVER clock, not local wall clock: the nodes' clocks are not
    synchronised and BASE_DIR being a shared mount does not make them so.
    Each node computes its wait from the same server-side arrival timestamp,
    so their start times agree to within one coord poll.
    """
    coord_signal(signal)
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = coord_check_ts(signal)
        if len(d.get("clients", [])) >= n_expected and d.get("ts"):
            first = min(float(t) for t in d["ts"].values())
            now = d.get("server_now")
            if now is None:
                break
            delay = (first + lead_s) - float(now)
            if delay > 0:
                time.sleep(delay)
            return True
        time.sleep(0.25)
    log(f"  barrier '{signal}' incomplete after {timeout:.0f}s "
        f"-- proceeding unsynchronised (duplex overlap not guaranteed)")
    return False


def _measure(cap, cfg, label) -> dict:
    """Analyse a captured probe against `cfg` and summarise per-carrier SNR."""
    modem = Modem(cfg)
    res = modem.analyze_probe(cap, threshold=SC_THRESH)
    if not res.get("valid"):
        err = res.get("error", "probe_invalid")
        log(f"  [{label}] probe INVALID: {err}")
        return {"valid": False, "error": err}
    snr = res["snr_db"]
    bat = modem.allocate(res)
    spec = carrier_spectrum(cfg, snr, bat)
    tx_fs = int(cfg["sample_rate"])
    out = {
        "valid": True,
        "coherence": res.get("coherence"),
        # The report's OFDM block keys off PAYLOAD SHAPE, not test name
        # (shared._ofdm_node_metrics), and what it charts is a `spectrum` dict
        # carrying freq_hz + snr_db + BITS together.  Keep carrier_spectrum()'s
        # output whole: the flattened snr_db/freq_hz below are for this
        # module's own penalty arithmetic and are dropped before the result is
        # written, so dropping `bits` here is what left this test unplotted.
        "spectrum": spec,
        # The far end's TRANSMIT table -- I measured it, so only I can tell it
        # what to use.  Kept as a list for the shared-dir hand-off in the data
        # phase; dropped from the result (redundant with spectrum['bits']).
        "bat": [int(b) for b in bat],
        "snr_db": [round(float(v), 2) for v in snr],
        "freq_hz": spec["freq_hz"],
        "snr_median_db": round(float(np.median(snr)), 1),
        "snr_p25_db": round(float(np.percentile(snr, 25)), 1),
        "snr_min_db": round(float(snr.min()), 1),
        "snr_max_db": round(float(snr.max()), 1),
        "active_carriers": int(np.sum(bat > 0)),
        "n_occupied": int(cfg["n_occupied"]),
        # Reverberation ceiling, straight from the probe.  analyze_probe()
        # already computes beta (ofdm_modem.isi_ratio); dropping it here is why
        # the report's ISI CEILING column was blank for this test while every
        # other OFDM test filled it.  Same derivation as ofdm_handshake.py:1223.
        "isi_ratio": (None if res.get("isi_ratio") is None
                      else round(float(res["isi_ratio"]), 3)),
        "isi_ceiling_db": (None if not res.get("isi_ratio") else
                           round(-10.0 * math.log10(
                               max(float(res["isi_ratio"]), 1e-9)), 1)),
        "isi_applied": bool(cfg.get("isi_derate")),
        "gross_bps": int(np.sum(bat) * tx_fs // cfg["symbol_samples"]),
        "bat_summary": modem.bat_summary(bat),
        "comb": _comb_metric(spec["freq_hz"], snr),
    }
    log(f"  [{label}] coh={out['coherence']:.3f} "
        f"snr min/p25/med/max={out['snr_min_db']}/{out['snr_p25_db']}/"
        f"{out['snr_median_db']}/{out['snr_max_db']}dB  "
        f"active={out['active_carriers']}/{out['n_occupied']}  "
        f"{out['gross_bps']}bps")
    # The reverberation ceiling caps SINR no matter what the SNR says, so a
    # ceiling under the modulation's floor means the allocation is fiction.
    # This is the number that explained run 20260810_204527's total data
    # failure while its probe reported a healthy 15.8 dB median.
    ceil_db = out.get("isi_ceiling_db")
    if ceil_db is not None:
        worst = max((int(b) for b in bat if b > 0), default=0)
        need = SNR_FLOOR_DB.get(worst)
        note = ""
        if need is not None and ceil_db < need:
            note = (f"  *** BELOW the {need}dB floor for {worst}-bit carriers: "
                    f"reverb past the CP caps this path, so those carriers "
                    f"CANNOT decode however strong they read. Raise cp_ratio/"
                    f"n_fft, or move closer. ***")
        log(f"  [{label}] ISI ceiling {ceil_db}dB (beta={out['isi_ratio']}){note}")

    c = out["comb"]
    if c.get("peak") is not None:
        log(f"  [{label}] comb: peak={c['peak']} spacing={c['spacing_hz']}Hz "
            f"ripple={c['ripple_db_rms']}dB rms")
    return out


# --------------------------------------------------------------------------- #
# Data phase
# --------------------------------------------------------------------------- #

# Printable ASCII only, and deliberately so.  AdaptiveModulator.encode_bytes
# NUL-pads the final chunk out to chunk_bytes and the reassembler rstrips NULs,
# so a payload that could itself end in NUL would be indistinguishable from its
# own padding -- the wire carries no length.  Restricting the alphabet is what
# makes the checksum comparison below trustworthy rather than usually-right.
_ALPHABET = string.ascii_letters + string.digits


def _count_acquisitions(samples: np.ndarray, cfg: dict, bat) -> int:
    """How many frame preambles the receiver actually FOUND.

    `Modem.decode_data` only returns a stat for a frame whose RS decode
    SUCCEEDED -- `_decode_frame` returns None otherwise -- so a frame that was
    acquired and then failed ECC leaves no trace and is indistinguishable, from
    the caller's side, from one that was never heard at all.  Those are
    opposite problems (too much noise on a present signal vs. no signal /
    acquisition failure) and want opposite fixes, so count the SC peaks
    directly using the same geometry decode_samples derives.

    Run 20260811_014814 is why: 4 of 6 frames "missing", and the report said
    they never acquired -- but all six preambles were found within 70 ms of
    their predicted positions. They were acquired and killed by 33-43% EVM.
    """
    try:
        dem = AdaptiveDemodulator(cfg, bat)
        n_syncs = (math.ceil(dem.data_syms_per_pkt / dem.sync_period)
                   if dem.sync_period > 0 and dem.data_syms_per_pkt > 0 else 0)
        n_body = 1 + dem.data_syms_per_pkt + n_syncs
        gap = _frame_min_gap(dem.symbol_samples, n_body,
                             cfg.get("taper_samples", 0), dem.cp_len)
        M = _sc_metric(samples, dem.n_fft)
        return len(_find_sc_peaks(M, dem.symbol_samples, gap, SC_THRESH))
    except Exception:                                        # noqa: BLE001
        return -1                                            # unknown, not zero


def _data_phase(my_cfg: dict, far_cfg: dict, my_dir: str, far_dir: str,
                my_bat, far_bat, out_dev, rec, mic_fs: int, spk_fs: int,
                target_s: float) -> dict:
    """Send a random payload each way SIMULTANEOUSLY and check what survived.

    The probe phases above measure what the channel COULD carry; this measures
    what it actually delivered, which is not the same claim -- a BAT is an
    estimate and RS either closes the gap or it does not.

    Table hand-off is over the shared RUN_DIR, not in-band.  Each node measured
    the OTHER's transmit table (it is the one listening to that band) so it has
    to be told, and doing that acoustically is a solved problem living in
    ofdmhs/ofdm2way.  Re-solving it here would mean a failed negotiation could
    masquerade as a failed data transfer, which is the one confusion this test
    must not have.

    The DUPLEX bat is used, not the solo one: the payloads go out at the same
    time, so the table has to describe the channel under exactly that
    condition. Sizing off the solo table would over-promise by precisely the
    penalty this test exists to measure.
    """
    fs = int(my_cfg["sample_rate"])

    # An empty table means this direction carried NOTHING under duplex, and it
    # must not be papered over.  Run 20260810_195049 did: linkup reported DL
    # 0 bps and the node still transmitted 1.5 KB of undecodable BPSK-all for
    # 10.6 s (AdaptiveModulator used to rewrite an all-zero BAT to
    # BPSK-everywhere while AdaptiveDemodulator did not, so the two ends sized
    # frames differently and no decode was possible), adding avoidable
    # interference to the far end's uplink while proving nothing.  The modem
    # now raises on an all-off BAT instead of substituting one, but the check
    # stays HERE: this is where the decision belongs, and it is what lets the
    # far end be told (skipped=empty_bat) rather than just failing.  Staying off
    # the air is also the physically consistent choice: a direction measured at
    # 0 bps is a direction that would not be carrying traffic.
    tx_active = int(np.count_nonzero(np.asarray(my_bat)))
    if tx_active == 0:
        log(f"  [data] TX {my_dir}: NOT TRANSMITTING -- the probe graded every "
            f"carrier in this direction below the SNR floor, so there is no "
            f"table to send with. Nothing was sent and no fallback was used.")
        (RUN_DIR / f"fdd_data_{my_dir}.json").write_text(json.dumps(
            {"sum16": None, "bytes": 0, "frames": 0, "airtime_s": 0.0,
             "skipped": "empty_bat"}))
        tx_sum, raw, n_frames, air_s, sig = None, b"", 0, 0.0, None
    else:
        # Keep a codeword short enough to survive: one frame is all-or-nothing,
        # so its length must sit inside the channel's coherence, not inside the
        # airtime budget.  Both ends must frame identically, so the chosen
        # chunk is published alongside the checksum below.
        chunk = _chunk_for_frame_len(my_cfg, my_bat, MAX_FRAME_S)
        if chunk != int(my_cfg["chunk_bytes"]):
            log(f"  [data] TX {my_dir}: chunk_bytes {my_cfg['chunk_bytes']}"
                f"->{chunk} to keep one frame under {MAX_FRAME_S:.0f}s "
                f"at {tx_active} carrier(s)")
            my_cfg = dict(my_cfg, chunk_bytes=chunk)
        n_bytes, n_frames, air_s = _size_for_airtime(my_cfg, my_bat, target_s)
        payload = _make_payload(n_bytes)
        raw = payload.encode("ascii")
        tx_sum = _sum16(raw)
        # One frame is the smallest unit that exists, so a very sparse table
        # can only overshoot: at 1 active BPSK carrier a single chunk is ~60 s.
        # Say so rather than let the far end wonder why the run stalled.
        if air_s > 2.5 * target_s:
            log(f"  [data] TX {my_dir}: WARNING one frame is {air_s:.0f}s at "
                f"{tx_active} carrier(s) -- cannot hit the {target_s:.0f}s "
                f"target; this leg is simply that slow")
        log(f"  [data] TX {my_dir}: {len(raw)} B ({len(raw)/1024:.1f} KB) "
            f"in {n_frames} frame(s), ~{air_s:.1f}s airtime, "
            f"{tx_active} carriers, sum={tx_sum}")
        (RUN_DIR / f"fdd_data_{my_dir}.json").write_text(json.dumps(
            {"sum16": tx_sum, "bytes": len(raw), "frames": n_frames,
             "airtime_s": round(air_s, 2), "chunk_bytes": chunk}))
        sig, _ = Modem(my_cfg).encode_data(payload, my_bat)

    start = rec.frames_captured
    synced = _barrier("data_go", 2, START_LEAD_S, LEG_TMO)
    # Timestamp MY OWN transmission for the report's clip-attribution pass
    # (shared._attribute_own_tx): a node's speaker couples into its own mic at
    # close range and reads exactly like an over-driven input, so without this
    # window the level chart blames the input gain for what is really the
    # node hearing itself. See the module-level TX_WINDOWS list.
    tx_window = None
    if sig is not None:
        tx_window = {"t": round(rec.frames_captured / mic_fs, 2),
                    "dur_s": round(len(sig) / fs, 2)}
        _play(sig, fs, out_dev)

    # Far side's airtime is a different number (different band, different BAT),
    # so hold the window open for whichever of us is slower -- same truncation
    # trap as the duplex probe.
    far_meta = {}
    far_path = RUN_DIR / f"fdd_data_{far_dir}.json"
    if far_path.exists():
        try:
            far_meta = json.loads(far_path.read_text())
        except Exception:                                    # noqa: BLE001
            far_meta = {}
    time.sleep(max(0.0, float(far_meta.get("airtime_s") or 0.0) - air_s) + TAIL_S)

    far_fs = int(far_cfg["sample_rate"])
    cap = _cap(rec, start, mic_fs, far_fs)

    # Re-read: the far node may only have published after the barrier.
    if far_path.exists():
        try:
            far_meta = json.loads(far_path.read_text())
        except Exception:                                    # noqa: BLE001
            pass
    exp_sum = far_meta.get("sum16")
    exp_bytes = far_meta.get("bytes")
    exp_frames = far_meta.get("frames")

    # The far end may have skipped for the same reason I might have. Decoding
    # silence would report "0 frames, MISMATCH", which reads as a link failure
    # when it is actually the far end correctly declining to send.
    # The far table is checked too, not just the far end's published marker: if
    # that node died before writing fdd_data_*.json we still know it had nothing
    # to send, because WE graded its band. Without this the empty table would
    # reach Modem.decode_data, which now (correctly) raises on an all-off BAT
    # rather than silently sizing frames at 0 data symbols.
    far_skipped = far_meta.get("skipped")
    if not far_skipped and int(np.count_nonzero(np.asarray(far_bat))) == 0:
        far_skipped = "empty_bat_far"
    if far_skipped:
        log(f"  [data] RX {far_dir}: far end did not transmit "
            f"({far_skipped}) -- nothing to receive")
        return {
            "tx_dir": my_dir, "rx_dir": far_dir, "tx_sum16": tx_sum,
            "tx_bytes": len(raw), "tx_frames": n_frames,
            "tx_airtime_s": round(air_s, 2), "tx_skipped": tx_active == 0,
            "tx_active_carriers": tx_active, "barrier_synced": synced,
            "tx_window": tx_window, "rx_skipped_by_peer": far_skipped,
            "ok": False, "evm_pct": None, "frames_decoded": 0,
        }

    # Frame the same way the sender did, or every frame boundary is wrong.
    far_chunk = far_meta.get("chunk_bytes")
    if far_chunk and far_chunk != int(far_cfg["chunk_bytes"]):
        far_cfg = dict(far_cfg, chunk_bytes=int(far_chunk))

    # Decode BOTH filtered and unfiltered and keep whichever did better.
    # Not indecision -- `_find_sc_peaks` enforces a minimum gap between frame
    # starts, so a spurious peak from residual leakage can SUPPRESS the real
    # one right after it.  That makes acquisition non-monotonic in interferer
    # level (synthetically: filtering won at equal levels and lost at +20 dB,
    # then won again at +30), so neither path is reliably the better one.
    # Running both costs one extra decode and reports which actually worked,
    # which is itself the finding a real FDD receiver design needs.
    filt, filt_desc = _rx_filter(cap, far_fs, far_cfg, my_cfg)
    attempts = [("none", cap)]
    if filt is not None:
        attempts.insert(0, (filt_desc, filt))

    best = None
    for desc, samples in attempts:
        stats, chunks = Modem(far_cfg).decode_data(samples, far_bat,
                                                   threshold=SC_THRESH)
        # Reassemble on BYTES, not via Modem.reassemble's str: a UTF-8 decode
        # with errors='replace' turns a corrupt frame into valid-looking text,
        # and the checksum would then compare the damage instead of catching it.
        got = b"".join(chunks[k] for k in sorted(chunks)).rstrip(b"\x00")
        fresh = [s for s in stats if not s.get("duplicate")]
        matched = bool(exp_sum) and _sum16(got) == exp_sum
        # A verified payload beats any frame count; failing that, more frames.
        score = (1 if matched else 0, len(fresh))
        if best is None or score > best[0]:
            best = (score, desc, got, fresh, matched, set(chunks.keys()), samples)
        if matched:
            break                                # cannot do better than correct

    _, rx_path, got, fresh, ok, decoded_seqs, best_samples = best
    rx_sum = _sum16(got)
    evms = [float(s["evm_pct"]) for s in fresh if s.get("evm_pct") is not None]
    evm_avg = round(sum(evms) / len(evms), 2) if evms else None
    if len(attempts) > 1:
        log(f"  [data] RX path: {rx_path}"
            + ("" if rx_path == "none" else " (own-band rejected)"))

    # WHICH frame(s) are missing, and -- critically -- whether they were even
    # heard.  Acquired-but-ECC-failed and never-acquired are opposite problems
    # (signal present but too noisy vs. no signal / lost sync) and the frame
    # list alone cannot tell them apart, because a failed RS decode returns no
    # stat at all.  Counting preambles separates them.
    missing = sorted(set(range(exp_frames or 0)) - decoded_seqs)
    n_acquired = _count_acquisitions(best_samples, far_cfg, far_bat)

    out = {
        "tx_dir": my_dir, "rx_dir": far_dir,
        "tx_sum16": tx_sum, "tx_bytes": len(raw), "tx_frames": n_frames,
        "tx_airtime_s": round(air_s, 2), "barrier_synced": synced,
        "tx_skipped": tx_active == 0, "tx_active_carriers": tx_active,
        "tx_window": tx_window,
        "rx_sum16": rx_sum, "rx_bytes": len(got),
        "expected_sum16": exp_sum, "expected_bytes": exp_bytes,
        "frames_decoded": len(fresh), "frames_expected": exp_frames,
        "missing_frames": missing, "frames_acquired": n_acquired,
        "rx_path": rx_path,
        "evm_pct": evm_avg,
        "evm_pct_per_frame": [round(float(s["evm_pct"]), 2) for s in fresh
                              if s.get("evm_pct") is not None],
        "err_counts": [int(s.get("err_count") or 0) for s in fresh],
        "ok": ok,
    }
    size_kb = (exp_bytes or len(got)) / 1024.0
    log(f"  [data] RX {far_dir}: sent {exp_sum or '?'}, received {rx_sum}, "
        f"size: {size_kb:.1f} KB  -> {'MATCH' if ok else 'MISMATCH'}")
    rs_errs = sum(out["err_counts"])
    log(f"  [data] RX {far_dir}: frames {len(fresh)}/{exp_frames or '?'}  "
        f"EVM {evm_avg if evm_avg is not None else 'n/a'}%  rs_errors={rs_errs}")
    if missing:
        # n_acquired counts preambles FOUND; len(fresh) counts frames that also
        # survived ECC.  The gap between them is the actual diagnosis.
        # n_acquired counts SC peaks, an UPPER bound: noise and own-TX can
        # raise spurious ones. So ">= sent" only proves the preambles were
        # there to be found, which is exactly the question being asked.
        if n_acquired < 0:
            why = "could not determine whether they were acquired"
        elif n_acquired >= (exp_frames or 0):
            why = (f"NOT a sync problem -- {n_acquired} preamble candidate(s) "
                   f"found for {exp_frames} sent, so these frames arrived and "
                   f"failed ECC (EVM {evm_avg}%). Too noisy, not missing.")
        elif n_acquired > len(fresh):
            why = (f"{n_acquired} preamble candidate(s) for {exp_frames} sent, "
                   f"{len(fresh)} survived ECC -- part noise, part lost sync")
        else:
            why = (f"only {n_acquired} preamble candidate(s) for {exp_frames} "
                   f"sent -- these never acquired (lost sync / signal absent)")
        log(f"  [data] RX {far_dir}: MISSING seq {missing} of "
            f"{list(range(exp_frames or 0))} -- {why}")
    return out


def _penalty_vs_separation(solo: dict, duplex: dict, own_hi_hz: float) -> list:
    """Duplex penalty bucketed by distance above the interferer's upper edge.

    This is the guard-band sizing curve, and the reason it is computed in the
    test rather than by hand afterwards: splatter falls off with separation,
    so `guard_hz` is not a constant of the design, it is whatever this curve
    says buys an acceptable penalty on the hardware actually in the room.
    Only meaningful for the node listening ABOVE its own transmission -- the
    other node is the one doing the splattering and sees none of it, so it
    returns [] rather than a floor of zeros that would dilute a pooled read.
    """
    if not (solo.get("valid") and duplex.get("valid")):
        return []
    f = np.asarray(solo.get("freq_hz", []), dtype=float)
    a = np.asarray(solo["snr_db"], dtype=float)
    b = np.asarray(duplex["snr_db"], dtype=float)
    n = min(len(a), len(b), len(f))
    if n == 0 or f[:n].min() < own_hi_hz:
        return []
    sep = f[:n] - own_hi_hz
    d = a[:n] - b[:n]
    out = []
    edges = [0, 250, 500, 750, 1000, 1500, 2000, 3000, 4500]
    for lo, hi in zip(edges, edges[1:]):
        m = (sep >= lo) & (sep < hi)
        if not m.any():
            continue
        out.append({"sep_lo_hz": lo, "sep_hi_hz": hi, "n": int(m.sum()),
                    "median_db": round(float(np.median(d[m])), 2),
                    "p90_db": round(float(np.percentile(d[m], 90)), 2)})
    return out


def _exchange_linkup(far_dir: str, solo: dict, duplex: dict,
                     timeout: float) -> dict | None:
    """Combine both nodes' measurements into ONE system-level rate.

    Each node only measures the direction it LISTENS to -- node1 (TX on DL)
    measures UL, node2 (TX on UL) measures DL -- so neither one alone can
    report the pipe's total throughput. RUN_DIR is the shared mount both
    already use for fdd_cfg_node*.json, so the same write-then-poll pattern
    works here: publish what I measured, read what the far node published,
    and both nodes converge on an identical combined summary without needing
    a leader.
    """
    my = {"dir": far_dir,
         "solo_bps": solo.get("gross_bps") if solo.get("valid") else None,
         "duplex_bps": duplex.get("gross_bps") if duplex.get("valid") else None}
    (RUN_DIR / f"fdd_rate_{far_dir}.json").write_text(json.dumps(my))

    other_dir = "ul" if far_dir == "dl" else "dl"
    other_path = RUN_DIR / f"fdd_rate_{other_dir}.json"
    deadline = time.time() + timeout
    while not other_path.exists() and time.time() < deadline:
        time.sleep(0.3)
    if not other_path.exists():
        log(f"  == LINKUP: far node's {other_dir} rate never appeared "
            f"(timeout) -- reporting my direction only")
        return None
    other = json.loads(other_path.read_text())
    dl = my if far_dir == "dl" else other
    ul = my if far_dir == "ul" else other

    def _sum(a, b):
        return None if (a is None and b is None) else (a or 0) + (b or 0)

    linkup = {
        "dl_solo_bps": dl["solo_bps"], "dl_duplex_bps": dl["duplex_bps"],
        "ul_solo_bps": ul["solo_bps"], "ul_duplex_bps": ul["duplex_bps"],
        "total_solo_bps": _sum(dl["solo_bps"], ul["solo_bps"]),
        "total_duplex_bps": _sum(dl["duplex_bps"], ul["duplex_bps"]),
    }
    # "0bps" and "DEAD" mean different things and the distinction matters:
    # 0 is a probe that decoded fine and then graded every carrier below the
    # SNR floor (the link is up and carries nothing), DEAD is a probe that
    # never decoded at all.  Spelling out "0bps (probe ok, all carriers below
    # floor)" is what run 20260810_195049 needed -- there, DL 0bps read as a
    # glitch when it was the honest answer.
    def _fmt(side):
        if side["duplex_bps"] is None:
            return "DEAD (probe did not decode)"
        if side["duplex_bps"] == 0:
            solo = side["solo_bps"]
            return ("0bps (all carriers below SNR floor"
                    + (f", was {solo}bps solo)" if solo else ")"))
        return f"{side['duplex_bps']}bps"

    tot, tot_s = linkup["total_duplex_bps"], linkup["total_solo_bps"]
    log(f"  == LINKUP  DL {_fmt(dl)}  +  UL {_fmt(ul)}")
    log(f"  == LINKUP  TOTAL {tot}bps duplex "
        f"(vs {tot_s}bps if each direction ran alone)")
    return linkup


def _penalty(solo: dict, duplex: dict) -> dict:
    """Per-carrier SNR lost by transmitting while receiving.

    This is the number the whole test exists to produce.  Both measurements
    come from the same carriers of the same sub-band at the same level, so the
    difference isolates own-TX leakage from every other channel effect.
    """
    if not (solo.get("valid") and duplex.get("valid")):
        return {"measurable": False,
                "reason": "duplex_probe_failed" if solo.get("valid")
                          else "solo_probe_failed"}
    a = np.asarray(solo["snr_db"], dtype=float)
    b = np.asarray(duplex["snr_db"], dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return {"measurable": False, "reason": "no_overlap"}
    d = a[:n] - b[:n]
    return {
        "measurable": True,
        "median_db": round(float(np.median(d)), 2),
        "p90_db": round(float(np.percentile(d, 90)), 2),
        "max_db": round(float(d.max()), 2),
        "carriers_lost": int(solo["active_carriers"] - duplex["active_carriers"]),
        "bps_solo": solo["gross_bps"],
        "bps_duplex": duplex["gross_bps"],
    }


# --------------------------------------------------------------------------- #

def _node_id() -> int:
    id_map = PARAMS.get("id_map", DEFAULT_ID_MAP)
    return int(id_map.get(CLIENT_ID, 1))


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    in_dev  = to_int_dev(INPUT_DEV)
    out_dev = to_int_dev(OUTPUT_DEV)
    node_id = _node_id()
    mic_fs, spk_fs = _native_rates(in_dev, out_dev)

    plan = _plan()
    # node1 owns the downlink sub-band, node2 the uplink.
    my_band, far_band = ((plan["dl_hz"], plan["ul_hz"]) if node_id == 1
                         else (plan["ul_hz"], plan["dl_hz"]))
    my_dir  = "dl" if node_id == 1 else "ul"
    far_dir = "ul" if node_id == 1 else "dl"

    log(f"[{CLIENT_ID}] ofdm_fdd node_id={node_id} dir={my_dir}  "
        f"mic_fs={mic_fs} spk_fs={spk_fs}")
    log(f"  plan: DL={plan['dl_hz']}Hz UL={plan['ul_hz']}Hz "
        f"guard={plan['guard_hz']}Hz  dl_frac {plan['dl_frac_requested']}"
        f"->{plan['dl_frac_actual']}"
        + ("  (CLAMPED to min_baud floor)" if plan["ratio_clamped"] else ""))
    log(f"  my band={my_band}Hz  listening on {far_dir}={far_band}Hz")
    for h in plan.get("harmonic_overlap") or []:
        log(f"  *** WARNING: the lower sub-band's {h['harmonic']}x harmonic "
            f"({h['range_hz'][0]:.0f}-{h['range_hz'][1]:.0f}Hz) covers "
            f"{h['covers_frac']*100:.0f}% of the upper sub-band. A guard band "
            f"CANNOT fix this -- harmonics land where they land. Keep the whole "
            f"span inside one octave (freq_hi < 2*freq_lo) so 2x the lower band "
            f"falls above the upper one. ***")

    # My transmit config is at MY speaker's rate; publish it so the far node
    # can decode what I send (it resamples its capture to this rate).
    my_cfg = _band_cfg(my_band, spk_fs)
    (RUN_DIR / f"fdd_cfg_node{node_id}.json").write_text(json.dumps(my_cfg, indent=2))
    (RUN_DIR / "fdd_plan.json").write_text(json.dumps(plan, indent=2))

    rec = Recorder(in_dev, mic_fs)
    rec.start()
    mic_cal = _calibrate_mic_gain(rec, mic_fs)

    # Fetch the far node's config (its speaker rate is not knowable locally).
    far_id = 2 if node_id == 1 else 1
    far_path = RUN_DIR / f"fdd_cfg_node{far_id}.json"
    deadline = time.time() + LEG_TMO
    while not far_path.exists() and time.time() < deadline:
        time.sleep(0.3)
    if not far_path.exists():
        rec.stop()
        result = {"client_id": CLIENT_ID, "role": "node", "run_id": RUN_ID,
                  "node_id": node_id, "status": "fail", "error": "no_far_cfg"}
        (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    far_cfg = json.loads(far_path.read_text())
    far_fs  = int(far_cfg["sample_rate"])

    probe = Modem(my_cfg).build_probe()
    solo, duplex = {}, {}
    # Every span where THIS node's own speaker was live, for the report's clip
    # attribution (shared._attribute_own_tx). Without this the report had no
    # way to tell a node's own near-field TX-into-its-own-mic apart from a
    # genuinely hot input gain, so turning down ONE node's amplitude visibly
    # changed only THAT node's own level chart while the report kept flagging
    # the OTHER node as over-driven -- the other node's "over-driven" windows
    # were its own probe/data bursts bleeding into its own mic the whole time,
    # unrelated to the first node's amplitude setting.
    tx_windows: list = []

    # ---- Phase SOLO: one transmitter at a time (the baseline) --------------
    for turn_node in (1, 2):
        tag = f"solo{turn_node}"
        if node_id == turn_node:
            coord_signal(f"{tag}_ready")
            coord_wait_any(f"{tag}_armed", timeout=LEG_TMO)
            tx_windows.append({"t": round(rec.frames_captured / mic_fs, 2),
                               "dur_s": round(len(probe) / spk_fs, 2)})
            _play(probe, spk_fs, out_dev)
            coord_signal(f"{tag}_played")
            coord_wait_any(f"{tag}_done", timeout=LEG_TMO)
        else:
            if not coord_wait_any(f"{tag}_ready", timeout=LEG_TMO):
                solo = {"valid": False, "error": "no_tx_ready"}
                continue
            start = rec.frames_captured
            coord_signal(f"{tag}_armed")
            coord_wait_any(f"{tag}_played",
                           timeout=far_cfg["actual_probe_s"] + LEG_TMO)
            time.sleep(TAIL_S)
            cap = _cap(rec, start, mic_fs, far_fs)
            solo = _measure(cap, far_cfg, f"solo {far_dir}")
            coord_signal(f"{tag}_done")

    # ---- Phase DUPLEX: both transmit at once -------------------------------
    # Own TX and the far probe are both live in this capture.  We decode with
    # the FAR config, so only the far sub-band's bins are read -- own TX is
    # rejected by the FFT alone, which is precisely the assumption under test.
    start = rec.frames_captured
    synced = _barrier("duplex_go", 2, START_LEAD_S, LEG_TMO)
    tx_windows.append({"t": round(rec.frames_captured / mic_fs, 2),
                       "dur_s": round(len(probe) / spk_fs, 2)})
    _play(probe, spk_fs, out_dev)
    # Keep capturing until the FAR probe must also have finished.  Own play is
    # blocking, so without this the window closes when MY probe ends -- and the
    # far probe is a different length (same probe_duration_s, different native
    # rate) plus whatever the barrier failed to align.  A truncated far probe
    # decodes as a low-SNR one, which would read as duplex penalty that is
    # really just a clipped recording.
    time.sleep(max(0.0, float(far_cfg["actual_probe_s"])
                        - float(my_cfg["actual_probe_s"])) + TAIL_S)
    cap = _cap(rec, start, mic_fs, far_fs)
    duplex = _measure(cap, far_cfg, f"duplex {far_dir}")
    coord_signal("duplex_done")
    coord_wait_any("duplex_done", timeout=LEG_TMO)

    pen = _penalty(solo, duplex)
    # my_band is what I transmit; the curve only exists if I listen above it.
    sep_curve = _penalty_vs_separation(solo, duplex, float(my_band[1]))
    if sep_curve:
        log(f"  guard sizing (I TX up to {my_band[1]:.0f}Hz, listening above it):")
        for r in sep_curve:
            log(f"     +{r['sep_lo_hz']}-{r['sep_hi_hz']}Hz  n={r['n']:3d}  "
                f"median={r['median_db']:6.2f}dB  p90={r['p90_db']:6.2f}dB")
    if pen.get("measurable"):
        log(f"  == FDD penalty on {far_dir}: median {pen['median_db']}dB  "
            f"p90 {pen['p90_db']}dB  carriers lost {pen['carriers_lost']}  "
            f"{pen['bps_solo']}->{pen['bps_duplex']}bps")
    else:
        log(f"  == FDD penalty on {far_dir}: NOT MEASURABLE "
            f"({pen.get('reason')})")

    linkup = _exchange_linkup(far_dir, solo, duplex, LEG_TMO)

    # ---- Phase DATA: real payload each way, simultaneously ----------------
    data = None
    if not DATA_S:
        log("  [data] skipped (data_s=0)")
    elif not duplex.get("valid"):
        log("  [data] skipped: no valid duplex probe, so no table to send with")
    else:
        # Hand the far node the table I measured for IT, and collect the one it
        # measured for me.  Named by the direction each table describes, so
        # both nodes read and write the same two filenames without a leader.
        (RUN_DIR / f"fdd_bat_{far_dir}.json").write_text(
            json.dumps({"bat": duplex["bat"]}))
        my_bat_path = RUN_DIR / f"fdd_bat_{my_dir}.json"
        deadline = time.time() + LEG_TMO
        while not my_bat_path.exists() and time.time() < deadline:
            time.sleep(0.3)
        if not my_bat_path.exists():
            log(f"  [data] skipped: far node never published my {my_dir} table")
        else:
            my_bat = np.array(json.loads(my_bat_path.read_text())["bat"],
                              dtype=np.int32)
            far_bat = np.array(duplex["bat"], dtype=np.int32)
            try:
                data = _data_phase(my_cfg, far_cfg, my_dir, far_dir,
                                   my_bat, far_bat, out_dev, rec,
                                   mic_fs, spk_fs, DATA_S)
            except Exception as exc:                         # noqa: BLE001
                log(f"  [data] FAILED: {type(exc).__name__}: {exc}")
                data = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if data and data.get("tx_window"):
                tx_windows.append(data["tx_window"])

    # What the ADC actually saw, across the WHOLE run rather than the ambient
    # window.  This is the level check that can act on real content: ambient
    # says nothing about whether the loud part had headroom, and the loud part
    # is what carries the payload.
    try:
        _all = rec.data()
        if len(_all):
            _pk = float(np.max(np.abs(_all)))
            _clip = 100.0 * float(np.mean(np.abs(_all) > 0.985))
            mic_cal["run_peak"] = round(_pk, 3)
            mic_cal["run_clip_pct"] = round(_clip, 2)
            if _clip > 1.0:
                log(f"  [mic] *** INPUT CLIPPED on {_clip:.1f}% of samples "
                    f"(peak {_pk:.2f}) -- lower the OS input level, or lower "
                    f"`amplitude` so this node's own speaker stops saturating "
                    f"its own mic. Decoding cannot recover a clipped ADC. ***")
            else:
                log(f"  [mic] run peak {_pk:.2f}, clipped {_clip:.2f}% "
                    f"-- headroom OK")
    except Exception:                                        # noqa: BLE001
        pass

    rec.stop()
    try:
        rec.save(RUN_DIR / f"{CLIENT_ID}_fdd.wav")
    except Exception:
        pass

    (RUN_DIR / f"fdd_spectra_{CLIENT_ID}.json").write_text(json.dumps(
        {"solo": solo, "duplex": duplex}, indent=2))

    # Hand the report the ofdm2way-shaped binding it already knows how to chart
    # (`measured` -> per-entry `spectrum` + `gross_bps`), so this test gets BAT
    # tables and S/N charts without touching shared.py.  Both phases are listed
    # because the comparison IS the result: two charts side by side show which
    # carriers the far end's simultaneous transmission actually cost.
    measured = []
    for phase, m in (("solo", solo), ("duplex", duplex)):
        if not m.get("valid"):
            continue
        measured.append({
            "direction": f"{far_dir} {phase}",
            "spectrum":  m["spectrum"],
            "gross_bps": m["gross_bps"],
            "snr_median_db": m["snr_median_db"],
            # shared._ofdm_node_metrics._isi() reads these off each `measured`
            # entry to fill the report's ISI CEILING column.
            "isi_ratio": m.get("isi_ratio"),
            "isi_ceiling_db": m.get("isi_ceiling_db"),
            "isi_applied": m.get("isi_applied"),
        })
    # The flattened copies exist only for _penalty/_penalty_vs_separation
    # above; `spectrum` in `measured` is the one the report reads.
    for d in (solo, duplex):
        d.pop("snr_db", None)
        d.pop("freq_hz", None)
        d.pop("spectrum", None)
        d.pop("bat", None)          # already in measured[].spectrum['bits']

    result = {
        "client_id": CLIENT_ID, "role": "node", "run_id": RUN_ID,
        "node_id": node_id, "my_dir": my_dir, "measured_dir": far_dir,
        "mic_fs": mic_fs, "spk_fs": spk_fs,
        "plan": plan, "barrier_synced": synced,
        "solo": solo, "duplex": duplex, "penalty": pen,
        "penalty_vs_separation": sep_curve,
        "measured": measured,
        "linkup": linkup,
        "data": data,
        # Fields shared._ofdm_node_metrics/_ofdm_summary read to populate the
        # report's node table.  `rx` is the per-frame list its EVM average is
        # built from -- without it the EVM column is blank even though the data
        # phase measured EVM on every decoded frame.
        "avg_evm_pct": (data or {}).get("evm_pct"),
        "rx": [{"evm_pct": v}
               for v in ((data or {}).get("evm_pct_per_frame") or [])],
        "band_hz": [round(float(far_cfg["freq_lo_hz"]), 1),
                    round(float(far_cfg["freq_hi_hz"]), 1)],
        "n_carriers": int(far_cfg["n_occupied"]),
        "active_carriers": (duplex or {}).get("active_carriers"),
        # shared._track_events reads capture.tx_windows to exclude spans where
        # THIS node's own speaker was live from the "over-driven input"
        # verdict on its own mic's level chart (shared._attribute_own_tx).
        "capture": {"tx_windows": tx_windows},
        "mic_cal": mic_cal,
        # THREE states, because "fail" was hiding a successful run.  This
        # test's job is the duplex MEASUREMENT; the data phase is a bonus
        # proof that the measured link carries payload.  Collapsing both into
        # ok/fail meant run 20260810_195831 -- whose probes decoded cleanly at
        # coherence 1.0 and produced a complete penalty curve -- reported
        # "fail" purely because a 6-carrier leg could not push bytes, which
        # reads as "the test broke" when the measurement is the deliverable.
        #   ok      measurement valid AND payload verified end to end
        #   partial measurement valid, payload did not survive (a RESULT)
        #   fail    the measurement itself is missing -- nothing to report
        "status": ("fail" if not pen.get("measurable")
                   else "ok" if (data is None or data.get("ok"))
                   else "partial"),
        "data_ok": (None if data is None else bool(data.get("ok"))),
    }
    (RUN_DIR / f"result_{CLIENT_ID}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
