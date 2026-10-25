# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""
capture_track.py - time-series level / SNR tracks from a run's capture WAVs.

Probe SNR is a single 5s measurement at handshake time, and EVM is averaged
over whatever frames decoded; neither distinguishes a channel that was bad
the whole run from one that was fine but received a too-hot input (ADC
saturation on dense constellations the robust probe would not show).
Distinguishing the two needs level and S/N over time, derived here from the
run's capture WAV after the fact, so it also works on already-recorded runs.

Server-side only (numpy). shared.py imports it lazily so client nodes, which
import shared for native_samplerate, never touch numpy or these paths.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

# |sample| at/above this is a clipped sample.  Matches meshmodem_node.CLIP_LEVEL
# so the report's clip accounting agrees with the node's own live warning.
CLIP_LEVEL = 0.99

# Analysis window.  0.25s keeps a 3-minute mesh run under ~750 points per track
# (small enough to ship inline in the report JSON) while still resolving a
# single OFDM data frame, which is ~1s.
WIN_S = 0.25

# Default analysis band when the run's records don't name one.  Wide enough to
# cover every OFDM profile in use (300-8000 Hz) without letting mains hum or
# out-of-band hiss into the in-band power.
DEFAULT_BAND_HZ = (300.0, 8000.0)

# Track cache version.  Bumped when the schema or the maths changes so stale
# sidecars from an older server are recomputed instead of rendered.
TRACK_VERSION = 2      # v2: crest_db / impulse-vs-sustained clip split

_EPS = 1e-20


def _dbfs(x: np.ndarray | float) -> np.ndarray | float:
    """Amplitude (0..1 full scale) to dBFS, floored at -120."""
    return np.maximum(20.0 * np.log10(np.maximum(x, _EPS)), -120.0)


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV as mono float32 in -1..1.  Handles 16/32-bit PCM captures."""
    with wave.open(str(path), "rb") as w:
        fs = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {sw} bytes")
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    return x, fs


def analyze_capture(path: Path, band_hz=None, win_s: float = WIN_S) -> dict:
    """Level + S/N time series for one capture WAV.

    Returns arrays on a common time axis (window centres, seconds from the
    start of the capture):

      rms_dbfs / peak_dbfs  input level, so "how close to clipping" is readable
                            directly: 0 dBFS *is* the clip line.
      clip_pct              % of samples at full scale in that window -- the
                            distortion itself, not just proximity to it.
      snr_db                in-band S/N: window in-band power over the run's
                            own ambient in-band floor.

    The noise floor is taken as a low percentile of in-band power ACROSS the
    run rather than from a dedicated silence measurement: every capture here is
    mostly dead air (nodes spend most of a run listening), and a percentile
    survives a capture that starts mid-burst, which a leading-window estimate
    does not.
    """
    x, fs = _read_mono(path)
    lo, hi = (band_hz or DEFAULT_BAND_HZ)
    n = max(1, int(round(win_s * fs)))
    nwin = len(x) // n
    if nwin < 2:
        return {"error": "capture too short to plot", "seconds": len(x) / max(fs, 1)}

    blocks = x[: nwin * n].reshape(nwin, n)

    rms = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1))
    peak = np.max(np.abs(blocks), axis=1)
    clip_pct = 100.0 * np.mean(np.abs(blocks) >= CLIP_LEVEL, axis=1)

    # Tell an environmental IMPULSE apart from a genuinely hot input. Both put
    # samples at full scale, but an impulse (door/keystroke/desk knock) is a
    # few ms wide with a high crest factor, while an over-driven burst is
    # sustained across the window.
    crest_db = 20.0 * np.log10(np.maximum(peak, _EPS)
                               / np.maximum(rms, _EPS))
    hot_ms = 1000.0 * np.count_nonzero(
        np.abs(blocks) >= 0.5 * np.maximum(peak, _EPS)[:, None], axis=1) / fs
    impulsive = (crest_db >= 20.0) | (hot_ms <= 20.0)
    clipped = clip_pct > 0
    clip_imp = clipped & impulsive
    clip_lvl = clipped & ~impulsive

    # In-band power per window.  Hann-windowed rFFT; only the band bins are
    # summed, so out-of-band rumble/hiss can't masquerade as signal.
    #
    # Scaled so the bin sum is mean-square amplitude (verified against a known
    # sine and white noise): interior bins carry twice the energy of DC/Nyquist,
    # and mean(win**2) undoes the Hann taper's power loss.  Without that the
    # "noise floor" line would be in arbitrary units and could print as a
    # positive dBFS, which is nonsense for a floor.
    win = np.hanning(n).astype(np.float32)
    spec = np.abs(np.fft.rfft(blocks * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    binsc = np.full(freqs.shape, 2.0)
    binsc[0] = 1.0
    if n % 2 == 0:
        binsc[-1] = 1.0
    spec = spec * (binsc / (n ** 2 * float(np.mean(win ** 2))))
    sel = (freqs >= lo) & (freqs <= hi)
    if not sel.any():
        sel = np.ones_like(freqs, dtype=bool)
    p_in = spec[:, sel].sum(axis=1)

    noise = float(np.percentile(p_in, 10))
    if noise <= 0:
        noise = float(np.min(p_in[p_in > 0])) if np.any(p_in > 0) else _EPS
    # Signal power is the excess over the floor: during dead air that excess is
    # ~0 and the trace sits at 0 dB instead of reporting the floor's own SNR.
    snr = 10.0 * np.log10(np.maximum(p_in - noise, noise * 1e-4) / noise)
    snr = np.clip(snr, 0.0, 60.0)

    t = (np.arange(nwin) + 0.5) * win_s

    # "Active" = a burst is on the air.  6 dB over the floor is well clear of
    # the percentile estimator's own spread and matches what carrier sense
    # treats as busy in practice.
    active = snr >= 6.0
    a_rms, a_peak, a_snr = rms[active], peak[active], snr[active]

    def _r(a, k=2):
        return [round(float(v), k) for v in a]

    return {
        "version":  TRACK_VERSION,
        "fs":       fs,
        "win_s":    win_s,
        "seconds":  round(len(x) / fs, 1),
        "band_hz":  [lo, hi],
        "t":        _r(t, 2),
        "rms_dbfs": _r(_dbfs(rms), 1),
        "peak_dbfs": _r(_dbfs(peak), 1),
        "clip_pct": _r(clip_pct, 3),
        "crest_db": _r(crest_db, 1),
        # per-window flag: 1 = the clipped samples here are impulse-shaped
        "clip_impulse": [int(v) for v in clip_imp],
        "snr_db":   _r(snr, 1),
        "summary": {
            "noise_dbfs":     round(float(_dbfs(np.sqrt(noise))), 1),
            "amb_rms_dbfs":   round(float(_dbfs(np.percentile(rms, 10))), 1),
            "peak_dbfs_max":  round(float(_dbfs(peak.max())), 1),
            "headroom_db":    round(float(-_dbfs(peak.max())), 1),
            "clip_windows":   int(np.count_nonzero(clipped)),
            # Only the SUSTAINED count is a level problem worth acting on.
            "clip_windows_impulse":   int(np.count_nonzero(clip_imp)),
            "clip_windows_sustained": int(np.count_nonzero(clip_lvl)),
            "hot_windows":    int(np.count_nonzero(peak >= 0.9)),
            "n_windows":      int(nwin),
            "active_pct":     round(100.0 * float(np.mean(active)), 1),
            # Active-window stats are the ones to quote: run-wide averages are
            # dominated by dead air and would call every run quiet and clean.
            "active_peak_dbfs": (round(float(_dbfs(a_peak.max())), 1)
                                 if a_peak.size else None),
            "active_rms_dbfs":  (round(float(np.median(_dbfs(a_rms))), 1)
                                 if a_rms.size else None),
            "snr_median_db":    (round(float(np.median(a_snr)), 1)
                                 if a_snr.size else None),
            "snr_p10_db":       (round(float(np.percentile(a_snr, 10)), 1)
                                 if a_snr.size else None),
            "snr_max_db":       (round(float(a_snr.max()), 1)
                                 if a_snr.size else None),
        },
    }


def _cache_path(wav: Path) -> Path:
    return wav.with_name(f"track_{wav.stem}.json")


def _cached(wav: Path) -> dict | None:
    p = _cache_path(wav)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    st = wav.stat()
    if (d.get("version") == TRACK_VERSION
            and d.get("src_size") == st.st_size
            and abs(d.get("src_mtime", 0) - st.st_mtime) < 1.0):
        return d
    return None


def capture_track(wav: Path, band_hz=None) -> dict:
    """Cached analyze_capture.  A 3-minute capture takes ~1s to analyze, and
    the Reports pane re-fetches the detail on every annotation save, so the
    result is memoised beside the WAV keyed on its size+mtime."""
    hit = _cached(wav)
    if hit is not None:
        return hit
    try:
        tr = analyze_capture(wav, band_hz=band_hz)
    except Exception as e:                            # noqa: BLE001
        # A truncated or exotic capture must not take the whole report down.
        return {"error": f"{type(e).__name__}: {e}"}
    if tr.get("error"):
        # Don't leave a sidecar for a failure: it would never satisfy the
        # version check anyway, so it is a file in the run dir that buys
        # nothing.  Re-failing on a broken WAV is cheap.
        return tr
    st = wav.stat()
    tr["src_size"], tr["src_mtime"] = st.st_size, st.st_mtime
    try:
        _cache_path(wav).write_text(json.dumps(tr))
    except OSError:
        pass
    return tr
