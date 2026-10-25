# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_loopback.py  —  Offline closed-loop test role for the CSS modem.

Purpose
-------
Run encode → (channel impairment) → decode entirely in memory / on local
WAV files, with NO sound hardware, so the decode logic can be iterated in
seconds instead of minutes and without recording/playback variability
confounding the results.

This is a third dispatch role alongside `listen` and `play`.  A node
assigned the `loopback` role runs a self-contained battery of tests and
emits a structured JSON result, the same shape the other roles produce, so
it slots into the existing manifest/dispatch/result framework.

Test battery (each independently toggle-able via config)
-------------------------------------------------------
1. clean            — encode→decode with NO impairment.  Must pass, N times,
                      or the whole run is aborted (the loop itself is broken,
                      no point measuring noise margins).

2. pink_sweep       — add pink (1/f) noise at increasing levels until decode
                      fails; record the SNR (dB) at the last PASS and first
                      FAIL.  This is the channel's noise margin.  Repeated
                      `trials` times per level (noise is random) and averaged.

3. chirp_obscure    — inject impulse bursts of length ≈ (1/3 × chirp) at a
                      random position (start / middle / end) of a random
                      subset of symbols.  Tests the partial-chirp /
                      obscuration-recovery path specifically.  Pass/fail +
                      fraction of symbols that can be obscured before failure.

4. drift_sweep      — (added) resample the encoded audio by a small ppm factor
                      to simulate transmitter/receiver clock mismatch, sweep
                      until the PLL can no longer track.  Directly exercises
                      the timing-recovery code that's been the hard part.

5. impulse_sweep    — (added) full-amplitude impulse clicks (like a knock or
                      ADC clip) at increasing density until failure.  Distinct
                      from chirp_obscure: tests the impulse BLANKER, not the
                      partial-chirp recovery.

Why these extra two: drift and impulse are the two impairments that most
directly stress the parts of the decoder that have been hardest to get
right (timing tracking and the blanker).  Having them in the quick loop
means a regression in either shows up immediately, before a real-world run.

6. reverb_sweep     — convolve with a synthetic room impulse response (direct
                      path + exponentially-decaying tail), sweeping RT60 up
                      until inter-symbol smear breaks decode.  The realistic
                      acoustic-channel impairment: multipath/echo.

7. distortion_sweep — drive the signal into a saturating nonlinearity (tanh /
                      hard clip) at increasing drive, level held constant, so
                      the variable is harmonic splatter + intermodulation, not
                      loudness.  Models loud playback / amp / ADC over-range.

8. wobble_sweep     — resample with a SINUSOIDALLY time-varying clock rate,
                      sweeping the peak-ppm depth up.  The dynamic counterpart
                      to drift_sweep: stresses PLL tracking bandwidth, not just
                      static pull range.

Real-audio interference (separate batch mode, not in the standard battery):
  run_noise_wav_batch() sweeps every WAV in a folder (song / movie soundtrack
  / ambience) mixed into the signal at decreasing SNR until decode fails — one
  job per WAV.  See --noise-wav-dir.

Usage
-----
    # As a dispatch role (reads config from the framework):
    python css_loopback.py

    # Standalone with an explicit config file:
    python css_loopback.py --config loopback_cfg.json

    # Standalone with all defaults (quick smoke test):
    python css_loopback.py --quick
"""

import argparse
import copy
import itertools
import json
import os
import sys
import tempfile
import threading
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

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

import benchlib                                                  # noqa: E402
import css_encoder                                               # noqa: E402


# Per-call sequence number so each decode writes a unique result_out file
# (decodes can run concurrently across worker processes).
_DECODE_SEQ = itertools.count()


# ──────────────────────────────────────────────────────────────────────────────
# Decoder harness — runs css_decoder.decode() against an in-memory audio array
# ──────────────────────────────────────────────────────────────────────────────

def _decode_array(audio, info_path, decode_overrides=None):
    """
    Run css_decoder.decode() against an in-memory float32 audio array by
    mocking sounddevice.InputStream.  Runs IN-PROCESS (subprocess isolation
    was too slow for multi-hundred-second test signals).

    Determinism: css_decoder is reloaded fresh each call so no module-level
    state carries over.  The decode itself is deterministic given identical
    input; any residual non-determinism indicates decoder global state, which
    is a real bug worth surfacing, not hiding.
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    stop_event = threading.Event()
    _pos = [0]

    # css_decoder.decode() always writes its result_out file.  Under parallel
    # workers a shared path would be a write race, so give every call a unique
    # temp file and delete it afterwards (we use the returned dict, not the
    # file).
    result_out = os.path.join(
        tempfile.gettempdir(),
        f"_lb_dec_{os.getpid()}_{next(_DECODE_SEQ)}.json")

    class _FakeStream:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self, n):
            p = _pos[0]
            chunk = audio[p:p + n]
            _pos[0] += n
            if len(chunk) < n:
                chunk = np.concatenate(
                    [chunk, np.zeros(n - len(chunk), dtype=np.float32)])
                stop_event.set()
            return chunk.reshape(-1, 1), False

    sd_mock = types.ModuleType("sounddevice")
    sd_mock.InputStream = lambda **kw: _FakeStream(**kw)

    saved = sys.modules.pop("css_decoder", None)
    saved_sd = sys.modules.get("sounddevice")
    sys.modules["sounddevice"] = sd_mock
    try:
        import importlib
        import css_decoder as _dec
        importlib.reload(_dec)

        cfg = {
            "info": str(info_path),
            "result_out": result_out,
            "device": None, "timeout_sec": 99999,
            "silence_threshold": 0.002, "sync_threshold": 0.55,
            "pll_weight": 0.3, "agc_attack": 0.5,
            "syndet_min_snr_db": 0.0, "syndet_snr_ref_db": 12.0,
            "syndet_lag_guard": True, "syndet_bg_alpha": 0.02,
            "syndet_margin": 0.15, "timing_search_frac": 0.25,
        }
        if decode_overrides:
            cfg.update(decode_overrides)

        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = _dec.decode(cfg, stop_event=stop_event)
        return result
    finally:
        if saved is not None:
            sys.modules["css_decoder"] = saved
        if saved_sd is not None:
            sys.modules["sounddevice"] = saved_sd
        else:
            sys.modules.pop("sounddevice", None)
        try:
            os.remove(result_out)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Channel impairments
# ──────────────────────────────────────────────────────────────────────────────

def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-20)


def _pink_noise(n, rng):
    """
    Generate n samples of pink (1/f) noise, unit RMS.

    Voss-McCartney-ish via FFT shaping: white noise spectrum scaled by
    1/sqrt(f).  Cheap, good enough for a test channel, and deterministic
    given the rng.
    """
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # avoid div0 at DC
    spec = spec / np.sqrt(freqs)
    pink = np.fft.irfft(spec, n)
    pink = pink / (_rms(pink) + 1e-20)
    return pink.astype(np.float32)


def add_pink_noise(audio, snr_db, rng):
    """Add pink noise at the given SNR (dB) relative to the audio's RMS."""
    sig_rms = _rms(audio)
    noise = _pink_noise(len(audio), rng)
    noise_rms_target = sig_rms / (10 ** (snr_db / 20.0))
    return (audio + noise * noise_rms_target).astype(np.float32)


def add_impulse_clicks(audio, density_per_sec, fs, amplitude, rng):
    """
    Add full-amplitude impulse clicks (short broadband bursts) at the given
    average density (clicks/second).  Models knocks, clicks, ADC clipping —
    the impairment the impulse BLANKER is meant to handle.  Each click is a
    1–3 ms burst of white noise at `amplitude` (which may exceed 1.0 to model
    clipping; the array is clamped to [-1, 1] at the end).
    """
    out = audio.copy()
    n = len(audio)
    n_clicks = int(density_per_sec * (n / fs))
    for _ in range(n_clicks):
        pos = rng.integers(0, n)
        dur = int(rng.integers(int(0.001 * fs), int(0.003 * fs) + 1))
        burst = rng.standard_normal(dur).astype(np.float32) * amplitude
        end = min(pos + dur, n)
        out[pos:end] += burst[:end - pos]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _cfg_geometry(enc_cfg):
    """(active chirp samples, fs) straight from the encode config.

    Derived from enc_cfg rather than from the worker cache `_W`, which the
    PARENT process never populates when a pool is in use -- the benchmark's
    sweep driver runs here while only the workers hold the encoded audio.
    Report-only: it resolves absolute widths into symbol fractions.
    """
    fs = int(enc_cfg.get("fs", 44100))
    return int(round(fs * float(enc_cfg.get("symbol_dur", 2.0)))), fs


def _sym_stride(info, fs) -> int:
    """Samples from one chirp's start to the next: chirp + inter-symbol gap.

    `info["sym_eff"]` is written in samples by `_ensure_local_encode`, but real
    run info.json files carry seconds; both appear here because promoted
    regression cases are read straight off disk.  Disambiguated by magnitude —
    a symbol period is never 100+ seconds and never fewer than 100 samples.
    """
    v = float(info["sym_eff"])
    return int(round(v * fs)) if v < 100 else int(round(v))


def _sym_active(info, fs) -> int:
    """Samples of actual chirp inside the stride, excluding the trailing gap.

    Width and alignment are measured against this, not the stride: a burst
    dropped in the silent gap damages nothing, so measuring "end-aligned" from
    the stride would score the gap rather than the chirp.
    """
    stride = _sym_stride(info, fs)
    gap = float(info.get("symbol_gap_frac", 0.0) or 0.0)
    return max(1, int(round(stride / (1.0 + gap)))) if gap > 0 else stride


def obscure_chirps(audio, info, fs, frac_symbols_obscured, rng):
    """
    Inject an impulse burst of length ≈ (1/3 × chirp) at a random position
    (start / middle / end) within a random subset of symbol slots.

    `frac_symbols_obscured` in [0,1] is the fraction of symbol slots that get
    an obscuring burst.  Models a person/object intermittently blocking the
    acoustic path mid-chirp — the partial-obscuration case the slice-recovery
    path targets.  Returns (audio, n_obscured).
    """
    out = audio.copy()
    n = len(audio)
    sps = int(round(info["sym_eff"] * fs)) if info["sym_eff"] < 100 else int(info["sym_eff"])
    # sym_eff may be stored as seconds or samples depending on writer; normalise.
    if sps < fs * 0.05:           # implausibly short -> was seconds
        sps = int(round(info["sym_eff"] * fs))
    burst_len = max(1, sps // 3)
    n_syms = n // sps
    n_obscure = int(round(frac_symbols_obscured * n_syms))
    if n_obscure <= 0:
        return out, 0
    sig_rms = _rms(audio)
    targets = rng.choice(n_syms, size=min(n_obscure, n_syms), replace=False)
    for s in targets:
        base = s * sps
        slot = rng.integers(0, 3)   # 0=start, 1=middle, 2=end
        if slot == 0:
            pos = base
        elif slot == 1:
            pos = base + (sps - burst_len) // 2
        else:
            pos = base + (sps - burst_len)
        end = min(pos + burst_len, n)
        # Obscuring burst ~6x signal RMS (loud enough to swamp the chirp in
        # that window, like a real acoustic block, but not full-scale clip).
        burst = rng.standard_normal(end - pos).astype(np.float32) * sig_rms * 6.0
        out[pos:end] += burst
    return np.clip(out, -1.0, 1.0).astype(np.float32), len(targets)


def resample_ppm(audio, ppm):
    """
    Resample audio by (1 + ppm/1e6) to simulate clock-rate mismatch between
    transmitter and receiver.  Linear interpolation is sufficient for the
    small (≤ a few thousand ppm) factors of interest; the decoder's PLL is
    what's under test, not interpolation fidelity.
    """
    factor = 1.0 + ppm / 1e6
    n_out = int(round(len(audio) / factor))
    x_in = np.arange(len(audio))
    x_out = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(x_out, x_in, audio).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Real-audio interference (music / movie soundtrack / ambience mixed in)
# ──────────────────────────────────────────────────────────────────────────────

def load_noise_wav(path, fs):
    """
    Load an interference WAV (song, movie soundtrack, café ambience, …),
    downmix to mono, resample to `fs`, and normalise to unit RMS.

    Unit-RMS normalisation is what lets add_wav_mix() scale it to an EXACT
    signal-to-interference SNR the same way add_pink_noise() does, so a 'music'
    sweep is directly comparable to the synthetic pink-noise sweep — the only
    difference is the spectral/temporal character of the interference.

    Resampling uses scipy.signal.resample_poly (polyphase, anti-aliased):
    music is full-band, so plain linear interpolation would alias high
    frequencies down into the 400-3000 Hz chirp band and make the interference
    unrepresentative of what a real microphone captures.
    """
    data, src_fs = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if src_fs != fs and len(data) > 1:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(src_fs), int(fs))
        up, down = int(fs) // g, int(src_fs) // g
        data = resample_poly(data, up, down).astype(np.float32)
    r = _rms(data)
    if r > 1e-12:
        data = data / r
    return np.ascontiguousarray(data, dtype=np.float32)


def _noise_window(noise_unit, n, rng):
    """Extract n samples from the (unit-RMS) interference clip starting at a
    seeded random offset, tiling/wrapping if the clip is shorter than n.  The
    random offset gives each trial a different slice of the song, so a single
    lucky quiet passage can't make the whole result non-representative."""
    src = noise_unit
    if len(src) < n:
        src = np.tile(src, -(-n // len(src)))      # ceil repetitions
    off = int(rng.integers(0, len(src)))
    return np.concatenate([src[off:], src[:off]])[:n].astype(np.float32)


def add_wav_mix(audio, noise_unit, snr_db, rng):
    """
    Mix real-audio interference into the signal at the given SNR (dB).

    `noise_unit` is the unit-RMS interference from load_noise_wav().  snr_db is
    signal-to-interference: +ve = signal louder, 0 = equal RMS, -ve = the music
    is LOUDER than the modem signal.  The sum is clipped to [-1, 1], which also
    realistically models the ADC clipping that loud interference causes in a
    real acoustic capture.
    """
    sig_rms = _rms(audio)
    win = _noise_window(noise_unit, len(audio), rng)
    target = sig_rms / (10 ** (snr_db / 20.0))
    return np.clip(audio + win * target, -1.0, 1.0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Reverb / multipath
# ──────────────────────────────────────────────────────────────────────────────

def _synth_rir(rt60_s, fs, rng, wet=0.5):
    """
    Synthetic room impulse response: a unit direct-path impulse at t=0 followed
    by an exponentially-decaying Gaussian-noise tail (the standard
    Moorer/Schroeder late-reverb model).  rt60_s is the RT60 — the time for the
    reverberant tail to decay 60 dB (exp(-6.908·t/rt60)).

    The direct path is fixed at unit gain so the genuine chirp's matched-filter
    peak keeps full strength; `wet` sets the tail's RMS RELATIVE to that direct
    path, so the impairment is the delayed/smeared energy (multipath echoes +
    inter-symbol bleed), not an overall level change.
    """
    n = max(2, int(rt60_s * fs))
    t = np.arange(n) / fs
    decay = np.exp(-6.907755 * t / rt60_s)
    tail = rng.standard_normal(n).astype(np.float32) * decay
    tr = _rms(tail)
    if tr > 1e-12:
        tail = tail * (wet / tr)
    tail[0] = 1.0          # unit direct path (overwrites first tail sample)
    return tail.astype(np.float32)


def add_reverb(audio, rt60_s, fs, rng, wet=0.5):
    """
    Convolve the signal with a synthetic room impulse response (see _synth_rir).
    Longer RT60 spreads each chirp's energy across more of the following
    symbols, so this directly stresses the decoder's echo-vs-direct-path peak
    selection and the inter-symbol-interference tolerance.  Output is truncated
    to the input length (the post-signal tail already has 10 symbols of silence
    for the reverb to ring out into).
    """
    rir = _synth_rir(rt60_s, fs, rng, wet=wet)
    # FFT convolution, not np.convolve.  np.convolve is DIRECT -- O(N*M) with no
    # FFT fallback at any size -- and the signals these sweeps build are long:
    # a 1.0 s symbol_dur, 6-frame encode is ~38 M samples, and an RT60 of 1.3 s
    # is a 62 k-tap RIR, so the direct form is ~2.4e12 multiply-adds for ONE
    # cell.  That does not "run slowly", it never finishes, and because it is
    # inside a worker pool it presents as a sweep that hangs with every core
    # busy.  fftconvolve makes the same cell O(N log N) and seconds long.
    try:
        from scipy.signal import fftconvolve
        wet_sig = fftconvolve(audio, rir)[:len(audio)]
    except ImportError:
        wet_sig = np.convolve(audio, rir)[:len(audio)]
    return np.clip(wet_sig, -1.0, 1.0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Nonlinear distortion / clipping
# ──────────────────────────────────────────────────────────────────────────────

def add_distortion(audio, drive_db, rng, mode="tanh"):
    """
    Drive the signal into a saturating nonlinearity, then renormalise back to
    the original peak so the test isolates DISTORTION (harmonic splatter +
    intermodulation that lands in the chirp band) from a mere level change.

    The signal is first scaled so its peak sits at 10**(drive_db/20):
      drive_db = 0  -> peak == 1.0  (onset of saturation; nearly transparent)
      drive_db = 6  -> 2× into the nonlinearity, etc.
    making the knob independent of the encoder's amplitude setting.  mode="tanh"
    is soft saturation (loud speaker/amp); mode="hard" is hard clipping (ADC
    over-range).  A little pink noise (30 dB) is added so trials vary and the
    renormalisation peak isn't degenerate.
    """
    pk = float(np.max(np.abs(audio))) + 1e-12
    g = 10 ** (drive_db / 20.0)
    x = audio / pk * g
    y = np.clip(x, -1.0, 1.0) if mode == "hard" else np.tanh(x)
    pk_out = float(np.max(np.abs(y))) + 1e-12
    out = y * (pk / pk_out)
    return add_pink_noise(out, 30.0, rng)


# ──────────────────────────────────────────────────────────────────────────────
# Clock warp / wobble (time-varying clock rate)
# ──────────────────────────────────────────────────────────────────────────────

def warp_wobble(audio, ppm_depth, wobble_hz, fs, rng, phase=0.0):
    """
    Resample with a SINUSOIDALLY time-varying clock rate — the dynamic
    counterpart to resample_ppm()'s constant drift.  The instantaneous rate is
    1 + (ppm_depth/1e6)·sin(2π·wobble_hz·t + phase), modelling a wandering or
    jittery sound-card clock, or the slow Doppler of someone walking with a
    phone.

    Where constant drift settles to a steady PLL offset, a wobble forces the
    loop to track a continuously-changing error, so this exercises the PLL's
    tracking BANDWIDTH rather than just its static pull range.  Output length
    equals input length; a little pink noise (30 dB) keeps it off a perfectly
    clean resample.

    `phase` (radians) offsets the sinusoid's start.  It defaults to 0 — the
    original behaviour — but leaving it at 0 makes the wobble a SINGLE clock
    realisation: repeated trials that redraw only the noise all see the run-in
    land at exactly the same point in the cycle.  A result that depends on
    where the run-in falls relative to the wobble then reproduces perfectly
    across trials and reads as structural when it is one sample of clock
    phase.  Callers comparing frame geometries (training length, frame period)
    should randomise this per trial.
    """
    n = len(audio)
    a = ppm_depth / 1e6
    t_out = np.arange(n) / fs
    rate = 1.0 + a * np.sin(2 * np.pi * wobble_hz * t_out + phase)
    pos = np.cumsum(rate)
    pos = pos - pos[0]                  # start reading at input sample 0
    pos = np.clip(pos, 0, n - 1)
    warped = np.interp(pos, np.arange(n), audio).astype(np.float32)
    return add_pink_noise(warped, 30.0, rng)


# ──────────────────────────────────────────────────────────────────────────────
# Encode helper
# ──────────────────────────────────────────────────────────────────────────────

def _encode_to_array(enc_cfg, workdir):
    """
    Run css_encoder.encode() to a temp WAV, return (audio, fs, info_path,
    info_dict).  info_path is needed by the decoder; info_dict for impairments.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out_wav = workdir / "loop_encoded.wav"
    info_path = workdir / "loop_info.json"
    result_path = workdir / "loop_encode_result.json"

    cfg = dict(enc_cfg)
    cfg["output"] = str(out_wav)
    cfg["info_out"] = str(info_path)
    cfg["result_out"] = str(result_path)

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        enc_result = css_encoder.encode(cfg)
    if enc_result.get("status") != "ok":
        raise RuntimeError(f"encode failed: {enc_result.get('error')}")

    audio, fs = sf.read(str(out_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    info = json.loads(info_path.read_text())
    # Stash sym_eff (samples) for impairment helpers that need symbol length.
    sps = int(round(fs * cfg.get("symbol_dur", 2.0)))
    gap = int(round(sps * cfg.get("symbol_gap_frac", 0.05)))
    info["sym_eff"] = sps + gap
    return audio, fs, info_path, info


def _decoded_ok(enc_cfg, dec_result):
    """Did the decode recover the original message?

    The decoder returns the full chunk_size-byte payload, which for a short
    message includes the encoder's zero-padding (e.g. message "hi" with
    chunk_size=12 comes back as "hi\\x00\\x00...").  Strip trailing NULs
    before comparing, and also accept the case where decoded text starts
    with the original message (padding intact).
    """
    want = enc_cfg.get("message", "")
    got = dec_result.get("decoded_text", "")
    got_stripped = got.rstrip("\x00")
    if want and (got_stripped == want or got_stripped.startswith(want)):
        return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Parallel execution
# ──────────────────────────────────────────────────────────────────────────────
#
# Each decode is an independent, CPU-bound unit of work, so the sweeps are
# embarrassingly parallel.  We use PROCESSES (not threads) because
# _decode_array() swaps modules in sys.modules and reloads css_decoder, which is
# not thread-safe — but is perfectly fine isolated in its own process.
#
# To avoid pickling the (potentially many-MB) reference audio array on every
# task, each worker encodes it ONCE at startup (encode is deterministic, so all
# workers get byte-identical audio) and stashes it in the module-global _W.
# Tasks then carry only scalar impairment params + a seed.

_W = {}   # per-process worker state: audio, fs, info, info_path, enc_cfg
_NOISE = {}  # per-process cache: noise_wav_path -> unit-RMS interference array


def _worker_noise(path, fs):
    """Load (once per worker process) and cache the unit-RMS interference clip
    for `path`.  Caching avoids re-reading/re-resampling the WAV on every task;
    the random per-trial window is taken from the cached array, not re-loaded."""
    nz = _NOISE.get(path)
    if nz is None:
        nz = load_noise_wav(path, fs)
        _NOISE[path] = nz
    return nz


def _guard_orphan():
    """Kernel-enforced 'die with my parent' for a pool worker.

    Without it, every stopped or crashed battery leaves `n_workers` processes
    behind, each holding an encoded reference signal and a numpy heap: the
    executor's shutdown() lives in a `finally:` that SIGTERM never reaches.
    Kept local (no testlib import: this module also runs standalone).
    """
    try:
        import ctypes
        import signal as _sig
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            1, int(_sig.SIGKILL), 0, 0, 0)          # PR_SET_PDEATHSIG
        if os.getppid() == 1:                      # parent already gone
            os._exit(0)
    except Exception:
        pass


def _pool_worker_init(enc_cfg, workdir):
    """Pool initializer: orphan guard first, then the normal encode cache."""
    _guard_orphan()
    _worker_init(enc_cfg, workdir)


def _worker_init(enc_cfg, workdir):
    """Pool initializer (also called in-process for the serial path): encode
    the reference signal once and cache it for this process."""
    sub = Path(workdir) / f"worker_{os.getpid()}"
    audio, fs, info_path, info = _encode_to_array(enc_cfg, sub)
    _W.update(audio=audio, fs=fs, info_path=info_path, info=info,
              enc_cfg=enc_cfg, ready=True)


def _ensure_local_encode(enc_cfg, workdir):
    """For the serial path (no pool): make sure this process has the cached
    audio, encoding it if a pool initializer hasn't already run here."""
    if not _W.get("ready"):
        _worker_init(enc_cfg, workdir)


def _trial_seed(base_seed, level_idx, trial_idx):
    """Deterministic per-(level, trial) seed.  Replaces the old single shared
    rng stream so trials are independent and reproducible regardless of which
    worker (and in what order) runs them.  Different noise realisations than the
    old serial code, but equally valid and still fully deterministic."""
    return (int(base_seed) * 1_000_003 + level_idx * 1009 + trial_idx) & 0x7FFFFFFF


def _worker_task(task):
    """Apply one impairment to the cached audio and decode it.  Runs in a worker
    process (or in-process for the serial path)."""
    t0 = time.time()
    audio = _W["audio"]
    fs = _W["fs"]
    info = _W["info"]
    rng = np.random.default_rng(task["seed"])
    kind = task["kind"]
    if kind == "clean":
        sig = audio
    elif kind == "pink":
        sig = add_pink_noise(audio, task["snr"], rng)
    elif kind == "obscure":
        sig, _ = obscure_chirps(audio, info, fs, task["frac"], rng)
    elif kind == "obscure_grid":
        # The benchmark's deterministic cousin of "obscure": an exact 1-in-N
        # of symbols, one alignment, one width -- see benchlib.
        sig, _ = benchlib.apply_obscuration(
            audio, fs, _sym_stride(info, fs), task["every_n"],
            benchlib.Width(task["width_kind"], task["width_value"]),
            task["align"], rng,
            active=_sym_active(info, fs),
            amp_ratio=task.get("amp_ratio", 6.0))
    elif kind == "drift":
        sig = resample_ppm(audio, task["ppm"])
        # tiny pink noise so it's not a perfectly clean resample
        sig = add_pink_noise(sig, 30.0, rng)
    elif kind == "impulse":
        sig = add_impulse_clicks(audio, task["density"], fs,
                                 task["amplitude"], rng)
    elif kind == "wavmix":
        noise = _worker_noise(task["noise_path"], fs)
        sig = add_wav_mix(audio, noise, task["snr"], rng)
    elif kind == "reverb":
        sig = add_reverb(audio, task["rt60"], fs, rng)
    elif kind == "distortion":
        sig = add_distortion(audio, task["drive_db"], rng,
                             mode=task.get("mode", "tanh"))
    elif kind == "wobble":
        sig = warp_wobble(audio, task["ppm_depth"], task["wobble_hz"], fs, rng)
    else:
        raise ValueError(f"unknown task kind {kind!r}")
    res = _decode_array(sig, _W["info_path"])
    return {
        "ok": _decoded_ok(_W["enc_cfg"], res),
        "decoded_text": res.get("decoded_text", ""),
        "failure_reason": res.get("failure_reason"),
        "elapsed": time.time() - t0,
    }


def _map_tasks(tasks, executor):
    """Run _worker_task over `tasks`, returning results in the SAME order as
    `tasks`.  In parallel when `executor` is given (results stream back in
    completion order and a per-task line is printed as each finishes), serially
    otherwise.  Each task must carry a 'desc' string for the progress line."""
    results = [None] * len(tasks)
    if executor is None:
        for i, task in enumerate(tasks):
            r = _worker_task(task)
            results[i] = r
            _print_task(task, r)
        return results
    fut_to_idx = {executor.submit(_worker_task, t): i
                  for i, t in enumerate(tasks)}
    for fut in as_completed(fut_to_idx):
        i = fut_to_idx[fut]
        r = fut.result()
        results[i] = r
        _print_task(tasks[i], r)
    return results


def _print_task(task, r):
    print(f"[loopback]     {task['desc']}: "
          f"{'PASS' if r['ok'] else 'fail'}  [{r.get('elapsed', 0):.1f}s]",
          flush=True)


def _resolve_workers(n):
    """Resolve the worker count.  None/<=0 -> auto = min(8, cpu_count).
    An explicit value is honoured but still clamped to the CPU count."""
    cpu = os.cpu_count() or 1
    if n is None or n <= 0:
        return max(1, min(8, cpu))
    return max(1, min(int(n), cpu))


def _classify_failure(reason):
    """Map css_decoder's failure_reason string to a short human category so a
    sweep failure says WHY it failed (PLL/sync vs ECC/CRC vs silence)."""
    r = (reason or "").lower()
    if r.startswith("no_sync"):
        return "sync/PLL lock failure"
    if r.startswith("crc_fail"):
        return "ECC/CRC decode failure"
    if r.startswith("no_audio"):
        return "no audio / silence"
    if r.startswith("sync acquired"):
        return "sync ok but no frames decoded"
    return reason or "decode failure"


def _failure_annotation(level_results):
    """Summarise why the failing trials at a level failed, e.g.
    'sync/PLL lock failure x2  [no_sync (peak_rms=0.0200)]'."""
    from collections import Counter
    reasons = [r.get("failure_reason") for r in level_results if not r["ok"]]
    if not reasons:
        return "", []
    counts = Counter(_classify_failure(x) for x in reasons)
    parts = [f"{c} x{n}" if n > 1 else c for c, n in counts.most_common()]
    ann = "; ".join(parts)
    raw = next((x for x in reasons if x), None)
    if raw:
        ann += f"  [{raw}]"
    return ann, reasons


def _sweep_levels(level_values, trials, make_task, base_seed,
                  executor, n_workers, desc_fn):
    """
    Lazily evaluate each level in `level_values` (already in sweep order),
    yielding (level, pass_rate, level_results) per level IN ORDER so the
    caller's existing break-on-failure logic is unchanged.  `level_results` is
    the list of per-trial worker result dicts (each has 'ok' and
    'failure_reason'), so the caller can report WHY a level failed.

    Internally it runs up to ceil(n_workers / trials) levels per batch so ~
    n_workers decodes are in flight at once.  Because it's a generator, the
    caller stopping early simply means later batches are never computed; at most
    (batch-1) extra levels past the break are evaluated speculatively, which is
    the cost of keeping all CPUs busy.
    """
    per_batch = max(1, -(-n_workers // max(1, trials)))   # ceil division
    i = 0
    while i < len(level_values):
        chunk = level_values[i:i + per_batch]
        tasks = []
        owners = []   # parallel list: which chunk-row each task belongs to
        for cj, lv in enumerate(chunk):
            for t in range(trials):
                task = make_task(lv)
                task["seed"] = _trial_seed(base_seed, i + cj, t)
                task["desc"] = f"{desc_fn(lv)} trial {t+1}/{trials}"
                tasks.append(task)
                owners.append(cj)
        results = _map_tasks(tasks, executor)
        per_level = [[] for _ in chunk]
        for owner, r in zip(owners, results):
            per_level[owner].append(r)
        for cj, lv in enumerate(chunk):
            lres = per_level[cj]
            rate = sum(int(r["ok"]) for r in lres) / len(lres)
            yield lv, rate, lres
        i += per_batch


# ──────────────────────────────────────────────────────────────────────────────
# Individual tests
# ──────────────────────────────────────────────────────────────────────────────

def test_clean(enc_cfg, workdir, n=3, executor=None, n_workers=1):
    """Encode→decode with no impairment, n times.  All must pass."""
    if executor is None:
        _ensure_local_encode(enc_cfg, workdir)
    tasks = [{"kind": "clean", "seed": _trial_seed(0, 0, i),
              "desc": f"clean trial {i+1}/{n}"} for i in range(n)]
    results = _map_tasks(tasks, executor)
    passes = sum(int(r["ok"]) for r in results)
    details = [{
        "trial": i, "ok": r["ok"],
        "decoded_text": r["decoded_text"],
        "failure_reason": r["failure_reason"],
    } for i, r in enumerate(results)]
    return {
        "name": "clean",
        "passed": passes == n,
        "pass_rate": passes / n,
        "n": n,
        "details": details,
    }


def test_pink_sweep(enc_cfg, workdir, snr_start=8.0, snr_step=2.0,
                    snr_floor=-12.0, trials=3, seed=0,
                    executor=None, n_workers=1):
    """
    Sweep pink-noise SNR downward from snr_start (default 8 dB) until decode
    fails (majority of `trials` fail), then STOP — the noise margin is the
    last SNR that still passed.  The first failing level records WHY it failed
    (sync/PLL lock vs ECC/CRC decode) from the decoder's failure_reason.
    """
    if executor is None:
        _ensure_local_encode(enc_cfg, workdir)
    snrs = []
    s = snr_start
    while s >= snr_floor:
        snrs.append(round(s, 4))
        s -= snr_step
    levels = []
    last_pass_snr = None
    first_fail_snr = None
    first_fail_reason = None
    for snr, rate, lres in _sweep_levels(
            snrs, trials, lambda lv: {"kind": "pink", "snr": lv},
            seed, executor, n_workers, lambda lv: f"snr={lv:5.1f}dB"):
        rec = {"snr_db": round(snr, 1), "pass_rate": rate}
        line = f"[loopback]   snr={snr:5.1f}dB -> {rate*100:3.0f}% pass"
        if rate >= 0.5:
            last_pass_snr = snr
            levels.append(rec)
            print(line, flush=True)
        else:
            # First majority-failure: record why, then stop the sweep.
            ann, reasons = _failure_annotation(lres)
            rec["failure"] = ann
            rec["failure_reasons"] = reasons
            first_fail_snr = snr
            first_fail_reason = ann
            levels.append(rec)
            print(f"{line}  FAIL: {ann}", flush=True)
            break
    return {
        "name": "pink_sweep",
        "last_pass_snr_db": last_pass_snr,
        "first_fail_snr_db": first_fail_snr,
        "first_fail_reason": first_fail_reason,
        "margin_summary": (
            (f"decodes down to ~{last_pass_snr:.0f} dB SNR"
             + (f", fails at {first_fail_snr:.0f} dB ({first_fail_reason})"
                if first_fail_snr is not None else ""))
            if last_pass_snr is not None
            else (f"failed even at the cleanest level ({first_fail_reason})"
                  if first_fail_reason is not None
                  else "failed even at the cleanest level")),
        "trials_per_level": trials,
        "levels": levels,
    }


def test_chirp_obscure(enc_cfg, workdir, frac_start=0.1, frac_step=0.1,
                       frac_max=0.8, trials=3, seed=1,
                       executor=None, n_workers=1):
    """
    Sweep the fraction of symbols that get a (1/3-chirp-length) obscuring
    burst, upward, until decode fails.  Records max obscuration tolerated.
    """
    if executor is None:
        _ensure_local_encode(enc_cfg, workdir)
    fracs = []
    f = frac_start
    while f <= frac_max + 1e-9:
        fracs.append(f)
        f += frac_step
    levels = []
    max_pass_frac = None
    for frac, rate, lres in _sweep_levels(
            fracs, trials, lambda lv: {"kind": "obscure", "frac": lv},
            seed, executor, n_workers, lambda lv: f"frac={lv:.2f}"):
        rec = {"frac_obscured": round(frac, 2), "pass_rate": rate}
        line = f"[loopback]   frac={frac:.2f} -> {rate*100:3.0f}% pass"
        if rate >= 0.5:
            max_pass_frac = frac
            levels.append(rec)
            print(line, flush=True)
        else:
            ann, reasons = _failure_annotation(lres)
            rec["failure"] = ann
            rec["failure_reasons"] = reasons
            levels.append(rec)
            print(f"{line}  FAIL: {ann}", flush=True)
            break
    return {
        "name": "chirp_obscure",
        "max_pass_frac": max_pass_frac,
        "summary": (f"tolerates up to ~{max_pass_frac*100:.0f}% of symbols obscured"
                    if max_pass_frac is not None
                    else "failed at the lowest obscuration level"),
        "trials_per_level": trials,
        "levels": levels,
    }


# Burst amplitude for the CSS grid, as a multiple of signal RMS.  Ten times
# benchlib's OFDM-calibrated default, and measured rather than guessed: the
# chirp matched filter's spreading gain rejects broadband noise so effectively
# that at 6x this decoder decodes 100% of symbols obscured over 3/4 of every
# chirp -- every cell scores 100 and the grid resolves nothing.  It first
# breaks up between 50x and 80x, so 60x puts the coverage ladder in range.
# Re-measure if the profile's amplitude, symbol duration or FEC changes.
CSS_BENCH_AMP_RATIO = 60.0


def test_impulse_benchmark(enc_cfg, workdir, trials=3, seed=11,
                           widths=None, aligns=None, ladder=None,
                           amp_ratio=CSS_BENCH_AMP_RATIO, strict=False,
                           executor=None, n_workers=1):
    """Score the decoder 0-100 against deterministic impulse obscuration.

    The graded counterpart to `chirp_obscure`.  That test answers "roughly what
    fraction can be obscured", with a random symbol subset and a random
    alignment per symbol, which makes its answer a blend across alignments and
    move run to run.  This one sweeps a (width x alignment) grid, hits an exact
    1-in-N of symbols, and scores each cell by the highest coverage it
    survived -- so a regression says WHICH width and WHICH alignment broke, and
    the number is comparable with the last run.

    Both are kept: the old sweep is cheap and the benchmark is the graded
    surface, and they cross-check each other.
    """
    if executor is None:
        _ensure_local_encode(enc_cfg, workdir)
    cells = []
    for wi, (width, align) in enumerate(benchlib.grid_axes(widths, aligns)):
        def run_level(every_n, _w=width, _a=align, _i=wi):
            # One coverage level = `trials` decodes of the same impairment with
            # different noise draws.  Reuses the battery's parallel mapper, so
            # the benchmark saturates the same worker pool as everything else.
            tasks = [{
                "kind": "obscure_grid", "every_n": every_n,
                "width_kind": _w.kind, "width_value": _w.value,
                "align": _a, "amp_ratio": amp_ratio,
                "seed": _trial_seed(seed, _i * 100 + every_n, t),
                "desc": (f"{_w.label()}/{_a} 1-in-{every_n} "
                         f"({benchlib.coverage_pct(every_n):.0f}%) "
                         f"trial {t+1}/{trials}"),
            } for t in range(trials)]
            res = _map_tasks(tasks, executor)
            return sum(int(r["ok"]) for r in res) / len(res)

        cell = benchlib.sweep_cell(
            run_level, width, align,
            ladder=ladder or benchlib.COVERAGE_LADDER, strict=strict,
            floor_ladder=benchlib.FLOOR_LADDER,
            geom=_cfg_geometry(enc_cfg))
        print(f"[loopback]   {width.label():>10s} {align:<7s} -> "
              f"score {cell['score']:5.1f}/100  "
              f"(survived {cell['tipping_point']['survived_pct']:.0f}%, "
              f"died at "
              f"{cell['tipping_point']['failed_pct'] or '-'}%)", flush=True)
        cells.append(cell)

    report = benchlib.score_grid(cells)
    print(benchlib.format_grid(report, "[loopback] CSS impulse benchmark"),
          flush=True)
    return {
        "name": "impulse_benchmark",
        "score": report["score"],
        "summary": (f"{report['score']:.1f}/100 "
                    f"(worst cell {report['worst']:.1f})"),
        "trials_per_level": trials,
        "amp_ratio": amp_ratio,
        **report,
    }


def test_drift_sweep(enc_cfg, workdir, ppm_start=100, ppm_step=200,
                     ppm_max=6000, trials=2, seed=2,
                     executor=None, n_workers=1):
    """
    Sweep clock-drift (ppm) upward until the PLL can no longer track.
    Tests BOTH signs (drift can be +/-); reports the smaller-magnitude limit.
    """
    if executor is None:
        _ensure_local_encode(enc_cfg, workdir)
    results = {}
    for sign, label in ((+1, "pos"), (-1, "neg")):
        ppms = []
        ppm = ppm_start
        while ppm <= ppm_max:
            ppms.append(sign * ppm)
            ppm += ppm_step
        levels = []
        max_pass_ppm = 0
        # distinct base seed per sign so +/- don't reuse identical noise draws
        base = seed * 2 + (0 if sign > 0 else 1)
        for sppm, rate, lres in _sweep_levels(
                ppms, trials, lambda lv: {"kind": "drift", "ppm": lv},
                base, executor, n_workers,
                lambda lv, _lbl=label: f"ppm={lv:+6d} ({_lbl})"):
            rec = {"ppm": sppm, "pass_rate": rate}
            line = f"[loopback]   ppm={sppm:+6d} -> {rate*100:3.0f}% pass"
            if rate >= 0.5:
                max_pass_ppm = abs(sppm)
                levels.append(rec)
                print(line, flush=True)
            else:
                ann, reasons = _failure_annotation(lres)
                rec["failure"] = ann
                rec["failure_reasons"] = reasons
                levels.append(rec)
                print(f"{line}  FAIL: {ann}", flush=True)
                break
        results[label] = {"max_pass_ppm": max_pass_ppm, "levels": levels}
    return {
        "name": "drift_sweep",
        "max_pass_ppm_pos": results["pos"]["max_pass_ppm"],
        "max_pass_ppm_neg": results["neg"]["max_pass_ppm"],
        "summary": (f"tracks +{results['pos']['max_pass_ppm']} / "
                    f"-{results['neg']['max_pass_ppm']} ppm drift"),
        "detail": results,
    }


def test_impulse_sweep(enc_cfg, workdir, density_start=1, density_step=2,
                       density_max=20, amplitude=1.2, trials=3, seed=3,
                       executor=None, n_workers=1):
    """
    Sweep impulse-click density (clicks/sec) upward until decode fails.
    Tests the impulse blanker (full-amplitude broadband clicks), distinct
    from chirp_obscure (sustained mid-chirp bursts).
    """
    if executor is None:
        _ensure_local_encode(enc_cfg, workdir)
    densities = []
    d = density_start
    while d <= density_max:
        densities.append(d)
        d += density_step
    levels = []
    max_pass_density = None
    for dens, rate, lres in _sweep_levels(
            densities, trials,
            lambda lv: {"kind": "impulse", "density": lv,
                        "amplitude": amplitude},
            seed, executor, n_workers, lambda lv: f"density={lv:2d}/s"):
        rec = {"density_per_sec": dens, "pass_rate": rate}
        line = f"[loopback]   density={dens:2d}/s -> {rate*100:3.0f}% pass"
        if rate >= 0.5:
            max_pass_density = dens
            levels.append(rec)
            print(line, flush=True)
        else:
            ann, reasons = _failure_annotation(lres)
            rec["failure"] = ann
            rec["failure_reasons"] = reasons
            levels.append(rec)
            print(f"{line}  FAIL: {ann}", flush=True)
            break
    return {
        "name": "impulse_sweep",
        "max_pass_density_per_sec": max_pass_density,
        "summary": (f"tolerates up to ~{max_pass_density} clicks/sec"
                    if max_pass_density is not None
                    else "failed at the lowest click density"),
        "trials_per_level": trials,
        "levels": levels,
    }


def test_reverb_sweep(enc_cfg, workdir, rt60_start=0.05, rt60_step=0.10,
                      rt60_max=1.5, trials=3, seed=4,
                      executor=None, n_workers=1):
    """
    Sweep room RT60 (seconds) upward until reverberant inter-symbol smear
    breaks the decode.  Reports the longest RT60 still decoded — the modem's
    multipath/reverb tolerance in physical units (a small office is ~0.3 s, a
    large hall ~1-2 s)."""
    if executor is None:
        _ensure_local_encode(enc_cfg, workdir)
    rt60s = []
    r = rt60_start
    while r <= rt60_max + 1e-9:
        rt60s.append(round(r, 4))
        r += rt60_step
    levels = []
    max_pass_rt60 = None
    for rt60, rate, lres in _sweep_levels(
            rt60s, trials, lambda lv: {"kind": "reverb", "rt60": lv},
            seed, executor, n_workers, lambda lv: f"rt60={lv:.2f}s"):
        rec = {"rt60_s": round(rt60, 3), "pass_rate": rate}
        line = f"[loopback]   rt60={rt60:.2f}s -> {rate*100:3.0f}% pass"
        if rate >= 0.5:
            max_pass_rt60 = rt60
            levels.append(rec)
            print(line, flush=True)
        else:
            ann, reasons = _failure_annotation(lres)
            rec["failure"] = ann
            rec["failure_reasons"] = reasons
            levels.append(rec)
            print(f"{line}  FAIL: {ann}", flush=True)
            break
    return {
        "name": "reverb_sweep",
        "max_pass_rt60_s": max_pass_rt60,
        "summary": (f"tolerates reverb up to ~{max_pass_rt60:.2f}s RT60"
                    if max_pass_rt60 is not None
                    else "failed at the shortest RT60"),
        "trials_per_level": trials,
        "levels": levels,
    }


def test_distortion_sweep(enc_cfg, workdir, drive_start=0.0, drive_step=3.0,
                          drive_max=36.0, mode="tanh", trials=3, seed=5,
                          executor=None, n_workers=1):
    """
    Sweep nonlinear-distortion drive (dB into a saturating nonlinearity) upward
    until harmonic splatter/intermodulation breaks the decode.  Level is held
    constant (output renormalised to the original peak), so the variable is
    purely distortion, not loudness.  Reports the max drive tolerated.
    """
    if executor is None:
        _ensure_local_encode(enc_cfg, workdir)
    drives = []
    d = drive_start
    while d <= drive_max + 1e-9:
        drives.append(round(d, 4))
        d += drive_step
    levels = []
    max_pass_drive = None
    for drive, rate, lres in _sweep_levels(
            drives, trials,
            lambda lv: {"kind": "distortion", "drive_db": lv, "mode": mode},
            seed, executor, n_workers, lambda lv: f"drive={lv:4.1f}dB"):
        rec = {"drive_db": round(drive, 1), "pass_rate": rate}
        line = f"[loopback]   drive={drive:4.1f}dB -> {rate*100:3.0f}% pass"
        if rate >= 0.5:
            max_pass_drive = drive
            levels.append(rec)
            print(line, flush=True)
        else:
            ann, reasons = _failure_annotation(lres)
            rec["failure"] = ann
            rec["failure_reasons"] = reasons
            levels.append(rec)
            print(f"{line}  FAIL: {ann}", flush=True)
            break
    return {
        "name": "distortion_sweep",
        "mode": mode,
        "max_pass_drive_db": max_pass_drive,
        "summary": (f"tolerates up to ~{max_pass_drive:.0f}dB {mode} drive"
                    if max_pass_drive is not None
                    else "failed at the lowest drive"),
        "trials_per_level": trials,
        "levels": levels,
    }


def test_wobble_sweep(enc_cfg, workdir, depth_start=200, depth_step=400,
                      depth_max=8000, wobble_hz=0.5, trials=2, seed=6,
                      executor=None, n_workers=1):
    """
    Sweep clock-wobble DEPTH (peak ppm of a sinusoidal rate variation, at a
    fixed wobble_hz) upward until the PLL can no longer track the continuously-
    changing timing error.  Complements drift_sweep (constant rate): this
    exercises the loop's tracking bandwidth, not its static pull range.
    """
    if executor is None:
        _ensure_local_encode(enc_cfg, workdir)
    depths = []
    d = depth_start
    while d <= depth_max:
        depths.append(d)
        d += depth_step
    levels = []
    max_pass_depth = None
    for depth, rate, lres in _sweep_levels(
            depths, trials,
            lambda lv: {"kind": "wobble", "ppm_depth": lv,
                        "wobble_hz": wobble_hz},
            seed, executor, n_workers,
            lambda lv: f"depth={lv:5d}ppm@{wobble_hz:g}Hz"):
        rec = {"ppm_depth": depth, "wobble_hz": wobble_hz, "pass_rate": rate}
        line = (f"[loopback]   depth={depth:5d}ppm@{wobble_hz:g}Hz "
                f"-> {rate*100:3.0f}% pass")
        if rate >= 0.5:
            max_pass_depth = depth
            levels.append(rec)
            print(line, flush=True)
        else:
            ann, reasons = _failure_annotation(lres)
            rec["failure"] = ann
            rec["failure_reasons"] = reasons
            levels.append(rec)
            print(f"{line}  FAIL: {ann}", flush=True)
            break
    return {
        "name": "wobble_sweep",
        "wobble_hz": wobble_hz,
        "max_pass_depth_ppm": max_pass_depth,
        "summary": (f"tracks wobble up to ~{max_pass_depth} ppm peak "
                    f"@ {wobble_hz:g} Hz"
                    if max_pass_depth is not None
                    else "failed at the lowest wobble depth"),
        "trials_per_level": trials,
        "levels": levels,
    }


def test_wav_mix(enc_cfg, workdir, noise_path, snr_start=10.0, snr_step=2.0,
                 snr_floor=-24.0, trials=3, seed=7,
                 executor=None, n_workers=1):
    """
    Mix a real interference clip (song / movie soundtrack / ambience) into the
    signal, sweeping signal-to-interference SNR DOWNWARD from snr_start until
    decode fails (majority of `trials` fail).

    snr_db < 0 means the music is LOUDER than the modem signal, so a negative
    last_pass SNR is the headline result: "still decodes with the music N dB
    louder than the signal".  This is the real-world analogue of the synthetic
    pink_sweep, using genuine program material whose spectrum overlaps the
    chirp band and whose level is non-stationary (loud and quiet passages).
    """
    if executor is None:
        _ensure_local_encode(enc_cfg, workdir)
    name = Path(noise_path).name
    snrs = []
    s = snr_start
    while s >= snr_floor:
        snrs.append(round(s, 4))
        s -= snr_step
    levels = []
    last_pass_snr = None
    first_fail_snr = None
    first_fail_reason = None
    for snr, rate, lres in _sweep_levels(
            snrs, trials,
            lambda lv: {"kind": "wavmix", "snr": lv,
                        "noise_path": str(noise_path)},
            seed, executor, n_workers,
            lambda lv: f"{name} snr={lv:+5.1f}dB"):
        rec = {"snr_db": round(snr, 1), "pass_rate": rate}
        line = f"[loopback]   {name} snr={snr:+5.1f}dB -> {rate*100:3.0f}% pass"
        if rate >= 0.5:
            last_pass_snr = snr
            levels.append(rec)
            print(line, flush=True)
        else:
            ann, reasons = _failure_annotation(lres)
            rec["failure"] = ann
            rec["failure_reasons"] = reasons
            first_fail_snr = snr
            first_fail_reason = ann
            levels.append(rec)
            print(f"{line}  FAIL: {ann}", flush=True)
            break

    if last_pass_snr is None:
        summary = ("failed even at the quietest interference level"
                   + (f" ({first_fail_reason})" if first_fail_reason else ""))
    elif last_pass_snr <= 0:
        summary = (f"decodes with '{name}' up to {abs(last_pass_snr):.0f} dB "
                   f"LOUDER than the signal")
    else:
        summary = (f"decodes with '{name}' down to +{last_pass_snr:.0f} dB SNR "
                   f"(fails before the music reaches signal level)")

    return {
        "name": "wav_mix",
        "noise_wav": name,
        "last_pass_snr_db": last_pass_snr,
        "first_fail_snr_db": first_fail_snr,
        "first_fail_reason": first_fail_reason,
        "summary": summary,
        "trials_per_level": trials,
        "levels": levels,
    }


def run_noise_wav_batch(noise_dir, enc_cfg=None, workdir="/tmp/css_loopback",
                        quick=False, snr_start=10.0, snr_step=2.0,
                        snr_floor=-24.0, trials=3, n_workers=None):
    """
    Batch the wav-mix test over every WAV in `noise_dir`.

    Each WAV is ONE job: the reference signal is encoded once (shared across all
    jobs and workers), then a signal-to-interference SNR sweep is run with that
    WAV mixed in, from snr_start down to snr_floor or until decode fails.
    N WAVs in the folder => N jobs, all sharing one encode and one worker pool.

    A single clean-decode validation runs first; if the loop itself is broken
    the whole batch is aborted (measuring noise margins would be meaningless).
    """
    base = FAST_ENC_CFG if quick else DEFAULT_ENC_CFG
    enc_cfg = {**base, **(enc_cfg or {})}
    if quick:
        trials = 1

    noise_dir = Path(noise_dir)
    wavs = sorted(noise_dir.glob("*.wav"))
    if not wavs:
        return {"role": "loopback", "status": "error",
                "failure_reason": f"no .wav files found in {noise_dir}"}

    n_workers = _resolve_workers(n_workers)
    print(f"[loopback] noise-wav batch: {len(wavs)} job(s) from {noise_dir}  "
          f"| {n_workers} worker(s)", flush=True)

    out = {
        "role": "loopback", "status": "ok", "mode": "noise_wav_batch",
        "noise_dir": str(noise_dir), "enc_cfg": enc_cfg,
        "n_workers": n_workers, "n_jobs": len(wavs),
        "sweep": {"snr_start": snr_start, "snr_step": snr_step,
                  "snr_floor": snr_floor, "trials": trials},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "jobs": [],
    }
    t0 = time.time()

    executor = None
    if n_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_pool_worker_init, initargs=(enc_cfg, workdir))
    else:
        _worker_init(enc_cfg, workdir)

    try:
        print("[loopback] clean validation...", flush=True)
        clean = test_clean(enc_cfg, workdir, n=3,
                           executor=executor, n_workers=n_workers)
        out["clean"] = clean
        if not clean["passed"]:
            out["status"] = "fail"
            out["abort_reason"] = ("clean decode failed — the encode/decode "
                                   "loop is broken; noise-wav batch skipped")
            print(f"[loopback] ABORT: {out['abort_reason']}", flush=True)
            return out

        for i, wav in enumerate(wavs):
            print(f"[loopback] job {i+1}/{len(wavs)}: {wav.name}", flush=True)
            ts = time.time()
            r = test_wav_mix(enc_cfg, workdir, wav,
                             snr_start=snr_start, snr_step=snr_step,
                             snr_floor=snr_floor, trials=trials,
                             seed=7 + i, executor=executor, n_workers=n_workers)
            r["elapsed_s"] = round(time.time() - ts, 1)
            out["jobs"].append(r)
            print(f"  {wav.name}: {r['summary']}  [{r['elapsed_s']}s]",
                  flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    out["elapsed_s"] = round(time.time() - t0, 1)
    print(f"[loopback] noise-wav batch done in {out['elapsed_s']}s", flush=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Battery runner
# ──────────────────────────────────────────────────────────────────────────────

# Slow-symbol VALIDATION BASELINE.
#
# This is the config to get rock-solid FIRST, before adding any complexity.
# It keeps the hard part — slow 2-second symbols, where clock drift is large
# in absolute samples and timing tracking has been the real challenge — but
# removes every OTHER source of difficulty:
#   • chunk_size=1 + a 2-char message => the payload is 1+1+4 = 6 bytes =
#     48 symbols with NO long identical-symbol run.  (chunk_size=12 with a
#     2-char message zero-pads to an 83-symbol run of identical down-chirps,
#     which is genuinely ambiguous to delimit; that's a SEPARATE problem to
#     tackle only after this baseline is solid.)
#   • Higher amplitude (0.05) so the clean-channel SNR is unambiguous.
# Decode is slower than FAST (real 2s symbols) but this is the config whose
# behaviour must match the real-world slow-symbol runs, minus the zero-run.
SLOW_SIMPLE_ENC_CFG = {
    "fs": 44100,
    "f_min": 400,
    "f_max": 3000,
    "symbol_dur": 2.0,
    "chunk_size": 1,
    "amplitude": 0.05,
    "rs_t": 0,
    "interleave_rows": 0,
    "diff_encode": True,
    "preamble": [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    "training_symbols": 16,
    "symbol_gap_frac": 0.05,
    "message": "hi",
}

DEFAULT_ENC_CFG = {
    "fs": 44100,
    "f_min": 400,
    "f_max": 3000,
    "symbol_dur": 2.0,
    "chunk_size": 12,
    "amplitude": 0.03,
    "rs_t": 0,
    "interleave_rows": 0,
    "diff_encode": True,
    "preamble": [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    "training_symbols": 16,
    "symbol_gap_frac": 0.05,
    "message": "hi",
}

# Fast config for quick iteration: short symbols + small chunk = short audio
# = fast decode (~3-5s vs ~35s).  Still exercises every code path (sync,
# acquisition, settled tracking, retro, frame assembly, CRC).  Use this for
# rapid regression checks; use DEFAULT_ENC_CFG to match the real-world test.
FAST_ENC_CFG = {
    "fs": 44100,
    "f_min": 400,
    "f_max": 3000,
    "symbol_dur": 0.25,
    "chunk_size": 2,
    "amplitude": 0.05,
    "rs_t": 0,
    "interleave_rows": 0,
    "diff_encode": True,
    "preamble": [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    "training_symbols": 16,
    "symbol_gap_frac": 0.05,
    "message": "hi",
}

DEFAULT_BATTERY = {
    "clean":            {"enable": True,  "n": 3},
    "pink_sweep":       {"enable": True,  "trials": 3},
    "chirp_obscure":    {"enable": True,  "trials": 3},
    # The graded impulse surface.  Off by default in the standard battery: it
    # is a 12-cell grid and costs several times a single sweep, so it is opted
    # into (--benchmark) or run on its own (--only-benchmark) rather than
    # lengthening every quick loop.
    "impulse_benchmark": {"enable": False, "trials": 3},
    "drift_sweep":      {"enable": True,  "trials": 2},
    "impulse_sweep":    {"enable": True,  "trials": 3},
    "reverb_sweep":     {"enable": True,  "trials": 3},
    "distortion_sweep": {"enable": True,  "trials": 3},
    "wobble_sweep":     {"enable": True,  "trials": 2},
}


def run_battery(enc_cfg=None, battery=None, workdir="/tmp/css_loopback",
                quick=False, time_budget_s=None, n_workers=None):
    base = FAST_ENC_CFG if quick else DEFAULT_ENC_CFG
    enc_cfg = {**base, **(enc_cfg or {})}
    battery = {**DEFAULT_BATTERY, **(battery or {})}

    if quick:
        # Faster: fewer trials, coarser sweeps.
        for k in battery:
            if "trials" in battery[k]:
                battery[k]["trials"] = 1

    n_workers = _resolve_workers(n_workers)
    print(f"[loopback] parallelism: {n_workers} worker(s) "
          f"(cpu_count={os.cpu_count()})", flush=True)

    out = {
        "role": "loopback",
        "status": "ok",
        "enc_cfg": enc_cfg,
        "n_workers": n_workers,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tests": {},
    }

    t0 = time.time()

    # One pool for the whole battery: each worker encodes the reference signal
    # ONCE in its initializer (cached in _W), then handles many decode tasks.
    # n_workers==1 runs serially in-process (no pool) — same code path via a
    # None executor.
    executor = None
    if n_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_pool_worker_init, initargs=(enc_cfg, workdir))
    else:
        _worker_init(enc_cfg, workdir)

    def _budget_exceeded():
        return (time_budget_s is not None
                and (time.time() - t0) > time_budget_s)

    def _run(label, fn, summary_key):
        # Wrap each sweep with progress logging + budget check so a long run
        # is visible in the log (never a silent hang) and bounded in time.
        if _budget_exceeded():
            print(f"[loopback] {label}: SKIPPED (time budget "
                  f"{time_budget_s}s exceeded)", flush=True)
            out.setdefault("skipped", []).append(label)
            return
        ts = time.time()
        print(f"[loopback] {label}...", flush=True)
        r = fn()
        out["tests"][r["name"]] = r
        print(f"  {r['name']}: {r.get(summary_key, '')}  "
              f"[{time.time()-ts:.1f}s]", flush=True)

    try:
        # 1. Clean first — abort the battery if the loop itself is broken.
        if battery["clean"]["enable"]:
            ts = time.time()
            print("[loopback] clean validation...", flush=True)
            r = test_clean(enc_cfg, workdir, n=battery["clean"].get("n", 3),
                           executor=executor, n_workers=n_workers)
            out["tests"]["clean"] = r
            print(f"  clean: {'PASS' if r['passed'] else 'FAIL'} "
                  f"({r['pass_rate']*100:.0f}%)  [{time.time()-ts:.1f}s]",
                  flush=True)
            if not r["passed"]:
                out["status"] = "fail"
                out["abort_reason"] = ("clean decode failed — the encode/decode "
                                       "loop is broken; noise-margin tests skipped")
                out["elapsed_s"] = round(time.time() - t0, 1)
                print(f"[loopback] ABORT: {out['abort_reason']}", flush=True)
                return out

        if battery["pink_sweep"]["enable"]:
            _run("pink-noise sweep",
                 lambda: test_pink_sweep(
                     enc_cfg, workdir,
                     trials=battery["pink_sweep"].get("trials", 3),
                     executor=executor, n_workers=n_workers),
                 "margin_summary")

        if battery["chirp_obscure"]["enable"]:
            _run("chirp-obscuration sweep",
                 lambda: test_chirp_obscure(
                     enc_cfg, workdir,
                     trials=battery["chirp_obscure"].get("trials", 3),
                     executor=executor, n_workers=n_workers),
                 "summary")

        if battery["drift_sweep"]["enable"]:
            _run("clock-drift sweep",
                 lambda: test_drift_sweep(
                     enc_cfg, workdir,
                     trials=battery["drift_sweep"].get("trials", 2),
                     executor=executor, n_workers=n_workers),
                 "summary")

        if battery["impulse_sweep"]["enable"]:
            _run("impulse-click sweep",
                 lambda: test_impulse_sweep(
                     enc_cfg, workdir,
                     trials=battery["impulse_sweep"].get("trials", 3),
                     executor=executor, n_workers=n_workers),
                 "summary")

        if battery.get("impulse_benchmark", {}).get("enable"):
            cfg = battery["impulse_benchmark"]
            _run("impulse benchmark (scored grid)",
                 lambda: test_impulse_benchmark(
                     enc_cfg, workdir,
                     trials=cfg.get("trials", 3),
                     amp_ratio=cfg.get("amp_ratio", CSS_BENCH_AMP_RATIO),
                     strict=cfg.get("strict", False),
                     executor=executor, n_workers=n_workers),
                 "summary")

        if battery["reverb_sweep"]["enable"]:
            _run("reverb / multipath sweep",
                 lambda: test_reverb_sweep(
                     enc_cfg, workdir,
                     trials=battery["reverb_sweep"].get("trials", 3),
                     executor=executor, n_workers=n_workers),
                 "summary")

        if battery["distortion_sweep"]["enable"]:
            _run("nonlinear-distortion sweep",
                 lambda: test_distortion_sweep(
                     enc_cfg, workdir,
                     trials=battery["distortion_sweep"].get("trials", 3),
                     executor=executor, n_workers=n_workers),
                 "summary")

        if battery["wobble_sweep"]["enable"]:
            _run("clock-wobble sweep",
                 lambda: test_wobble_sweep(
                     enc_cfg, workdir,
                     trials=battery["wobble_sweep"].get("trials", 2),
                     executor=executor, n_workers=n_workers),
                 "summary")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    out["elapsed_s"] = round(time.time() - t0, 1)
    print(f"[loopback] done in {out['elapsed_s']}s", flush=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI / dispatch entry point
# ──────────────────────────────────────────────────────────────────────────────

def _run_as_framework_role():
    """
    Framework dispatch entry point (role=loopback).

    Mirrors the contract css_listen.py / css_play.py follow:
      • read job config from testlib.PARAMS
      • do the work
      • emit the result dict as JSON on the FINAL stdout line
    Unlike listen/play, loopback needs no css_ready/css_played coordination —
    it encodes and decodes entirely on its own, so it just runs the battery
    and reports.

    Returns True if it handled the run as a framework role, False if testlib
    isn't importable (i.e. we're running standalone — fall through to CLI).

    Explicit CLI arguments also mean standalone.  testlib is importable from
    anywhere inside the repo, so "testlib imports" alone selected this path for
    every in-repo invocation and silently discarded the entire CLI — a
    `--only-benchmark` run would print a full default battery and no benchmark,
    with nothing to say the flag had been dropped.  The framework dispatches
    with no argv at all (see app.py `_exec_test`, which runs
    `[sys.executable, test_path]`), so the presence of an argument is an
    unambiguous signal that a human is driving.
    """
    if len(sys.argv) > 1:
        return False
    try:
        from testlib import CLIENT_ID, RUN_DIR, PARAMS, RUN_ID, log
    except Exception:
        return False

    def _log(m):
        try:
            log(m)
        except Exception:
            print(m, flush=True)

    # ── Profile / config selection from PARAMS ────────────────────────────
    # PARAMS comes from the manifest's params block.  A "profile" key picks a
    # named base config; any individual encode keys in PARAMS override it.
    # DEFAULT here is the FAST profile, NOT the slow chunk_size=12 one — a
    # framework launch with no explicit params should return in seconds, not
    # spend a minute on the very first decode (which looked like a hang).
    profile = str(PARAMS.get("profile", "fast")).lower()  # choices: fast, slow_simple, default
    base = {
        "fast": FAST_ENC_CFG,
        "slow_simple": SLOW_SIMPLE_ENC_CFG,
        "default": DEFAULT_ENC_CFG,
    }.get(profile, FAST_ENC_CFG)

    enc_cfg = dict(base)
    # Any recognised encode key present in PARAMS overrides the profile.
    for k in ("fs", "f_min", "f_max", "symbol_dur", "chunk_size", "amplitude",
              "rs_t", "interleave_rows", "diff_encode", "preamble",
              "training_symbols", "symbol_gap_frac", "message"):
        if k in PARAMS and PARAMS[k] is not None:
            enc_cfg[k] = PARAMS[k]

    # Battery selection: PARAMS may disable sweeps or set trial counts.
    battery = {k: dict(v) for k, v in DEFAULT_BATTERY.items()}
    if PARAMS.get("clean_only"):
        for k in battery:
            battery[k]["enable"] = (k == "clean")
    for k in battery:
        tk = f"{k}_trials"
        if tk in PARAMS:
            battery[k]["trials"] = int(PARAMS[tk])
        ek = f"{k}_enable"
        if ek in PARAMS:
            battery[k]["enable"] = bool(PARAMS[ek])
    # Benchmark knobs a dispatched run may want; the generic {k}_trials /
    # {k}_enable loop above already covers the rest.
    if "impulse_benchmark_strict" in PARAMS:
        battery["impulse_benchmark"]["strict"] = bool(
            PARAMS["impulse_benchmark_strict"])
    if "impulse_benchmark_amp_ratio" in PARAMS:
        battery["impulse_benchmark"]["amp_ratio"] = float(
            PARAMS["impulse_benchmark_amp_ratio"])

    quick = bool(PARAMS.get("quick", False))
    workdir = str(RUN_DIR) if RUN_DIR else "/tmp/css_loopback"
    # Wall-clock budget so a framework run can never silently hang: after this
    # many seconds, remaining sweeps are skipped (the clean result and any
    # completed sweeps are still reported).  Default 300s; override via PARAMS.
    time_budget_s = float(PARAMS.get("time_budget_s", 300))
    # Parallel worker count.  "jobs"/"n_workers" in PARAMS; None/0 -> auto
    # (min(8, cpu_count)).
    n_workers = PARAMS.get("jobs", PARAMS.get("n_workers", None))

    _log(f"[loopback] profile={profile} "
         f"sym={enc_cfg['symbol_dur']}s chunk={enc_cfg['chunk_size']} "
         f"msg={enc_cfg['message']!r} quick={quick} budget={time_budget_s:.0f}s "
         f"jobs={n_workers or 'auto'}")

    # If a noise-wav folder is supplied, run the real-audio interference batch
    # instead of the standard impairment battery (one job per WAV).
    noise_wav_dir = PARAMS.get("noise_wav_dir")

    try:
        if noise_wav_dir:
            result = run_noise_wav_batch(
                noise_wav_dir, enc_cfg=enc_cfg, workdir=workdir, quick=quick,
                snr_start=float(PARAMS.get("snr_start", 10.0)),
                snr_step=float(PARAMS.get("snr_step", 2.0)),
                snr_floor=float(PARAMS.get("snr_floor", -24.0)),
                trials=int(PARAMS.get("wav_mix_trials", 3)),
                n_workers=n_workers)
        else:
            result = run_battery(enc_cfg=enc_cfg, battery=battery,
                                workdir=workdir, quick=quick,
                                time_budget_s=time_budget_s, n_workers=n_workers)
    except Exception as exc:
        import traceback
        _log(f"[loopback] ERROR: {exc}")
        traceback.print_exc()
        result = {"role": "loopback", "status": "error",
                  "failure_reason": str(exc)}

    # Write a result file into the run dir AND emit JSON on the final line.
    result.update({"client_id": CLIENT_ID, "role": "loopback",
                   "run_id": RUN_ID})
    if RUN_DIR:
        try:
            (Path(RUN_DIR) / f"loopback_result_{CLIENT_ID}.json").write_text(
                json.dumps(result, indent=2))
        except Exception as exc:
            _log(f"[loopback] result file write failed: {exc}")

    _log(f"[loopback] status={result.get('status')} "
         f"elapsed={result.get('elapsed_s')}s")
    print(json.dumps(result), flush=True)   # framework captures this line
    return True


def main():
    # If launched by the framework (testlib importable), run as a role.
    if _run_as_framework_role():
        return

    # Otherwise: standalone CLI.
    ap = argparse.ArgumentParser(description="CSS offline loopback test role")
    ap.add_argument("--config", help="JSON config file (enc_cfg + battery)")
    ap.add_argument("--profile",
                    choices=["fast", "slow_simple", "default"],
                    default="fast",
                    help="named encode profile (default: fast): "
                         "fast (0.25s sym, quick iteration), "
                         "slow_simple (2s sym, chunk_size=1, validation "
                         "baseline — get this solid first), "
                         "default (2s sym, chunk_size=12, real-world match)")
    ap.add_argument("--quick", action="store_true",
                    help="1 trial per level, faster smoke test")
    ap.add_argument("--clean-only", action="store_true",
                    help="run ONLY the clean-decode validation, skip sweeps")
    ap.add_argument("--out", default="loopback_result.json",
                    help="where to write the JSON result")
    ap.add_argument("--workdir", default="/tmp/css_loopback")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel decode workers; 0 = auto "
                         "(min(8, CPU count)).  Decodes run in separate "
                         "processes, so this scales near-linearly.")
    ap.add_argument("--noise-wav-dir",
                    help="real-audio interference batch: sweep every WAV in "
                         "this folder mixed into the signal at decreasing SNR "
                         "until decode fails (one job per WAV).  Overrides the "
                         "standard battery.")
    ap.add_argument("--snr-start", type=float, default=10.0,
                    help="wav-mix sweep: starting signal-to-interference SNR dB "
                         "(default 10 = signal clearly louder)")
    ap.add_argument("--snr-step", type=float, default=2.0,
                    help="wav-mix sweep: SNR decrement per step in dB (default 2)")
    ap.add_argument("--snr-floor", type=float, default=-24.0,
                    help="wav-mix sweep: lowest SNR dB to try "
                         "(default -24 = music 24 dB louder than signal)")
    ap.add_argument("--benchmark", action="store_true",
                    help="also run the scored impulse benchmark (a "
                         "width x alignment grid, 0-100 per cell)")
    ap.add_argument("--only-benchmark", action="store_true",
                    help="run ONLY the impulse benchmark (plus the clean "
                         "check that guards it), skipping every other sweep")
    ap.add_argument("--benchmark-strict", action="store_true",
                    help="benchmark: walk the whole coverage ladder instead of "
                         "stopping at the first failure, and report any level "
                         "that passes above one that failed")
    args = ap.parse_args()

    enc_cfg = None
    battery = None
    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        enc_cfg = cfg.get("enc_cfg")
        battery = cfg.get("battery")

    base = {"fast": FAST_ENC_CFG, "slow_simple": SLOW_SIMPLE_ENC_CFG,
            "default": DEFAULT_ENC_CFG}[args.profile]
    enc_cfg = {**base, **(enc_cfg or {})}

    if args.noise_wav_dir:
        result = run_noise_wav_batch(
            args.noise_wav_dir, enc_cfg=enc_cfg, workdir=args.workdir,
            quick=args.quick, snr_start=args.snr_start, snr_step=args.snr_step,
            snr_floor=args.snr_floor, n_workers=args.jobs)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[loopback] result -> {args.out}", flush=True)
        sys.exit(0 if result.get("status") == "ok" else 1)

    if args.clean_only:
        battery = {**DEFAULT_BATTERY, **(battery or {})}
        for k in battery:
            battery[k] = {**battery[k], "enable": (k == "clean")}

    if args.benchmark or args.only_benchmark:
        battery = {**DEFAULT_BATTERY, **(battery or {})}
        # `clean` stays on even for --only-benchmark: a zero score means
        # nothing if the unimpaired loop was already broken, and that check
        # costs three decodes against a grid that costs hundreds.
        if args.only_benchmark:
            for k in battery:
                battery[k] = {**battery[k],
                              "enable": k in ("clean", "impulse_benchmark")}
        battery["impulse_benchmark"] = {**battery["impulse_benchmark"],
                                        "enable": True,
                                        "strict": args.benchmark_strict}

    result = run_battery(enc_cfg=enc_cfg, battery=battery,
                         workdir=args.workdir, quick=args.quick,
                         n_workers=args.jobs)

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[loopback] result -> {args.out}", flush=True)
    sys.exit(0 if result.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()