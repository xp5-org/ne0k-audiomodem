# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_decoder.py

Live CSS (chirp spread spectrum) audio decoder — main decode loop.

Encoding is ABSOLUTE: up-chirp = bit 1, down-chirp = bit 0.

Design: read exactly sps samples per symbol period.
  The sounddevice stream.read(sps) reads exactly that many frames.
  No hop/ssls machinery. No fractional drift. No timing sign confusion.

  Clock drift correction: after each eval, timing_err = found_peak - expected_peak.
  Fractional accumulator drives a ±1 sample correction on the next read:
    terr < 0 (chirp left)  → read sps-1 next → ring advances less → chirp right ✓
    terr > 0 (chirp right) → read sps+1 next → ring advances more → chirp left  ✓

Matched-filter design — linear chirp sweep (default):
  The transmitted chirps are unwindowed linear FM sweeps, so the optimal MF
  template is also an unwindowed linear chirp.  The legacy Hann-windowed
  template (mf_window="hann") introduced a ~3 dB sensitivity loss because
  the template no longer matched the transmitted waveform.

  Default: mf_window="none" (unwindowed, properly matched).
  Set mf_window="hann" only if backward-compatibility with an encoder that
  transmits Hann-windowed chirps is required.

ECC erasure marking with convolutional coding:
  When conv_code="k7r12" is active, the previous code reset frame_confs to
  [1.0, ...] after Viterbi, which discarded all soft information and disabled
  RS erasure marking.  The revised decoder preserves per-bit soft energy from
  the Viterbi input pairs as a reliability signal after decoding.  Low
  soft-energy bits (weak signal / high noise during that symbol) are marked
  as erasures for RS.  RS corrects up to 2*t erasures vs t errors, so
  erasure marking roughly doubles the effective RS correction capacity.

Module layout (split from a single ~2490-line file for maintainability):
  css_fec.py         — Viterbi K7, diff decode, IF-slope confidence,
                        Reed-Solomon wrapper, de-interleaver, chirp
                        generation, byte/CRC utilities.
  css_correlator.py  — ChirpCorrelator (matched filter) and PeakAGC.
  css_syncdet.py      — SoftPreambleDetector (Barker/preamble sync).
  css_retro.py        — retro_timing_decode (two-pass timing recovery).
  css_decoder.py       — this file: the live decode() loop, PLL, impulse
                        blanker, frame assembly, and CLI entrypoint, tying
                        the above together.  Kept as one file because it is
                        a single tightly-coupled state machine (closures,
                        nonlocal PLL/AGC/timing state threaded through one
                        live audio loop) that does not have clean internal
                        module boundaries the way the pieces above do.
"""

import json
import os
import sys
import threading
import time
import zlib
from collections import deque
from pathlib import Path

import numpy as np

from css_fec import (
    CRC_BYTES,
    viterbi_k7,
    viterbi_k7_with_reliability,
    diff_decode_bits,
    if_slope_confidence,
    _make_rs_codec,
    rs_decode_frame,
    deinterleave_bits,
    generate_chirp,
    get_bytes_from_bits,
    check_crc,
)
from css_correlator import SIG, ChirpCorrelator, PeakAGC, corr_noise_floor
from css_syncdet import SoftPreambleDetector
from css_retro import retro_timing_decode, retro_offset_candidates

# Optional repo-root helper for audio-interface probing.  The decoder runs as a
# standalone subprocess from tests/, so reach shared.py by adding the repo root
# to sys.path; degrade gracefully (interface logging skipped) if unavailable.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from shared import log_audio_interface as _log_audio_interface
    from shared import default_training_symbols as _default_training_symbols
except Exception:
    _log_audio_interface = None
    _default_training_symbols = None

BARKER_13 = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]

EXAMPLE_CONFIG = {
    "info":               "css_info.json",
    "result_out":         "css_decode_result.json",
    "device":             None,
    "timeout_sec":        600,
    "silence_threshold":  0.002,
    "sync_threshold":     0.55,   # css_listen.py operating value (decoder code fallback: 0.65)
    "timing_loop_gain":   0.10,
    "timing_search_frac": 0.25,   # matches decode() fallback; css_listen.py default
    "agc_target_peak":    0.4,
    "agc_attack":         0.5,    # css_listen.py operating value (decoder code fallback: 0.8)
    "agc_release":        0.02,
    "agc_max_gain":       200.0,
    "snr_log_interval":   10,
    "mf_window":          "none",     # "none" = linear chirp (default), "hann" = legacy
    "erasure_conf_thr":   0.25,
    "timing_min_snr_db":  3.0,        # floor to reject empty buffers; median filter handles noise rejection
}


class _QueuedInputStream:
    """Overflow-proof audio input.

    The decode loop reads one full symbol (~sps_eff samples, ~2 s for long
    symbols) per blocking read, then runs heavy DSP (matched filter, sync,
    retro).  A plain blocking InputStream only buffers ~one `latency` worth of
    audio, so whenever that DSP gap exceeds the buffer the hardware ring
    overruns and PortAudio DROPS samples — every dropped block shifts all later
    chirps earlier and the timing loop can never settle ("symbols never
    tracking").  This is acute when the host is also resampling (e.g. a 48 kHz
    mic feeding a 44.1 kHz stream), which adds CPU.

    Fix: a PortAudio *callback* copies each input block into an in-RAM buffer the
    instant it arrives, so the hardware ring is always drained and never
    overruns regardless of DSP spikes.  `.read(n)` then serves fixed-size blocks
    from that buffer, so the decode loop is unchanged.  Memory is bounded by the
    run length (≈4 bytes/sample) and stays small because DSP keeps up on average.

    Real hardware only — the offline repro / loopback tests mock InputStream
    with their own blocking fake and bypass this class.
    """

    def __init__(self, sd, samplerate, device, channels=1):
        self._buf      = deque()
        self._navail   = 0
        self._cv       = threading.Condition()
        self._ovf      = False
        self._closed   = False
        self._stream   = sd.InputStream(
            samplerate=samplerate, channels=channels, dtype="float32",
            device=device, callback=self._callback)

    def _callback(self, indata, frames, time_info, status):
        ovf = bool(status and getattr(status, "input_overflow", False))
        with self._cv:
            if ovf:
                self._ovf = True
            self._buf.append(indata[:, 0].copy())
            self._navail += frames
            self._cv.notify()

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, *exc):
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    # passthrough attributes for the stream-open log line
    @property
    def samplerate(self): return self._stream.samplerate
    @property
    def blocksize(self):  return self._stream.blocksize
    @property
    def latency(self):    return self._stream.latency
    @property
    def device(self):     return self._stream.device

    def read(self, n):
        """Block until n samples are buffered, returning (ndarray[n,1], overflowed).
        On stream close, pads the remainder with zeros so the loop can exit."""
        out = np.empty(n, dtype=np.float32)
        got = 0
        with self._cv:
            overflowed = self._ovf
            self._ovf = False
            while got < n:
                while not self._buf and not self._closed:
                    self._cv.wait(timeout=1.0)   # periodic wake: never wedge on a missed notify
                if not self._buf and self._closed:
                    out[got:] = 0.0
                    return out.reshape(-1, 1), overflowed
                chunk = self._buf[0]
                take  = min(len(chunk), n - got)
                out[got:got + take] = chunk[:take]
                if take == len(chunk):
                    self._buf.popleft()
                else:
                    self._buf[0] = chunk[take:]
                self._navail -= take
                got += take
        return out.reshape(-1, 1), overflowed


# ---------------------------------------------------------------------------
# Matched-filter decoder for LONG symbols (symbol_dur > ~0.5 s)
# ---------------------------------------------------------------------------

class MFSymbolFilter:
    """Lightweight per-symbol matched filter for the MF tracking loop.

    ChirpCorrelator.symbol_decision runs 8 sub-slice correlations over a
    buf_size-wide FFT — far too slow per symbol and unneeded for clean long
    chirps.  Here a tight window (chirp + a small drift margin) and a single
    up/down correlation suffice; the caller's historic grid fit keeps
    predictions accurate enough that ±track_search covers residual drift.

    Pure w.r.t. the decoder: all state is precomputed template spectra fixed
    at construction; __call__(win, centre) has no side effects.
    """

    def __init__(self, up_ref, down_ref, sps, track_search, decide_search,
                 seg_M=12, seg_trim=2, blank_kmad=0.0):
        self.sps           = sps
        self.track_search  = track_search
        self.decide_search = decide_search
        self.seg_M         = seg_M
        self.seg_trim      = seg_trim
        self.blank_kmad    = blank_kmad
        win_len   = sps + 2 * track_search
        self.nfft = 1 << (win_len + sps).bit_length()
        rev_up = up_ref[:sps].astype(SIG)[::-1]
        rev_dn = down_ref[:sps].astype(SIG)[::-1]
        self._Fup = np.fft.rfft(rev_up, self.nfft)
        self._Fdn = np.fft.rfft(rev_dn, self.nfft)
        # ── Per-segment spectra for the segmented MF ─────────────────────────
        # The segments were originally masked slices of the reversed template
        # left IN PLACE, which made every segment correlation share the full
        # template's "template ends here" lag so the trimmed sum could combine
        # them sample-for-sample.  Convenient, but it forces every segment
        # transform to the FULL template's size: an in-place segment still spans
        # sps samples, so the no-alias requirement is win_len + sps - 1.
        #
        # A segment is only seg_L = sps/seg_M samples of actual support, so
        # COMPACTING it to offset 0 drops the requirement to win_len + seg_L - 1
        # and lets the transform run at the next power of two above THAT.  The
        # shared lag reference is restored exactly by an integer index shift:
        # for a segment starting at `a` in the reversed template,
        #     envelope_in_place[k] == envelope_compact[k - a]
        # (and is identically zero for k < a, which the in-place form produced
        # via its own zero padding).  So this is an EXACT restatement, not an
        # approximation — no aliasing is introduced anywhere, unlike shrinking
        # the full-template transform, which corrupts the low lags the median
        # noise reference is computed over.
        #
        # Whether it actually buys anything is geometry-dependent: it wins when
        # a power of two falls between win_len + seg_L - 1 and win_len + sps - 1.
        # At the production 0.3 s / 48 kHz profile it is a clean 2x on the 24
        # segment transforms (32768 vs 65536), which is ~83% of decode CPU; at
        # 0.5 s and 1.0 s the two sizes coincide and it costs nothing.  Falls
        # back to the full size automatically when there is no gain.
        self._seg_off  = []
        self._seg_Fup  = []
        self._seg_Fdn  = []
        self.nfft_seg  = self.nfft
        if seg_M > 1:
            seg_L  = sps // seg_M
            spans  = [(m * seg_L, sps if m == seg_M - 1 else (m + 1) * seg_L)
                      for m in range(seg_M)]
            max_L  = max(b - a for a, b in spans)
            self.nfft_seg = min(self.nfft,
                                1 << (win_len + max_L - 2).bit_length())
            for a, b in spans:
                self._seg_off.append(a)
                self._seg_Fup.append(np.fft.rfft(rev_up[a:b], self.nfft_seg))
                self._seg_Fdn.append(np.fft.rfft(rev_dn[a:b], self.nfft_seg))

    def __call__(self, win, centre):
        """Return (conf, peak_in_win, snr_db, winner_pk).

        Timing (peak_in_win, snr) is found over the WIDE ±track_search window so
        the grid fit can follow a displaced chirp.  The DIRECTION (conf), though,
        is decided ON-GRID in a TIGHT ±decide_search window around the predicted
        centre.  Rationale: taking the independent argmax of the up and down
        envelopes across the wide window lets a stronger OPPOSITE-direction
        reflection / neighbour-symbol leak — offset from the true chirp — win the
        vote and flip a clearly-decodable symbol, even though the real chirp sits
        on-grid at full strength (observed on 3 s symbols in a reverberant room:
        a DN reverb peak 0.03 sym off, amp 184, beat the true UP on-grid at amp
        153).  Deciding at the single best combined peak near the prediction
        rejects those off-grid reflections and also yields honest soft-confidence.
        """
        nfft = self.nfft
        win = win.astype(SIG)
        if self.blank_kmad > 0:
            _bmed = float(np.median(np.abs(win)))
            if _bmed > 0:
                _bmask = np.abs(win) > self.blank_kmad * _bmed
                if _bmask.any():
                    win = np.where(_bmask, 0.0, win)   # copy — never mutate buf
        if self.seg_M > 1:
            # Segmented MF: per-segment envelopes, trimmed-sum combine (see the
            # mf_segments config block in decode_longsym_mf).  float32 stacks:
            # 2·M·len(win) transient.
            #
            # Segment templates are COMPACTED to offset 0 (see __init__), so the
            # correlation for a segment starting at `a` comes out shifted: its
            # lag-k value sits at index k - a, and lags below a are zero.  The
            # shift is applied on the way into the stack, which restores the
            # common lag reference the trimmed sum needs.
            _nfs = self.nfft_seg
            _Bs  = np.fft.rfft(win, _nfs)
            _n   = len(win)
            _UE = np.zeros((self.seg_M, _n), dtype=SIG)
            _DE = np.zeros((self.seg_M, _n), dtype=SIG)
            for _m in range(self.seg_M):
                _a = self._seg_off[_m]
                if _a >= _n:
                    continue                      # segment never overlaps window
                _v = _n - _a                      # valid (alias-free) output span
                _UE[_m, _a:] = np.abs(
                    np.fft.irfft(_Bs * self._seg_Fup[_m], _nfs)[:_v])
                _DE[_m, _a:] = np.abs(
                    np.fft.irfft(_Bs * self._seg_Fdn[_m], _nfs)[:_v])
            _keep = max(1, self.seg_M - max(0, self.seg_trim))
            # This sort is the second-largest line item in the decoder (~20% of
            # decode CPU once the transforms were halved), and it LOOKS like an
            # easy win: only the _keep smallest per lag are needed, so O(M)
            # selection should beat O(M log M) ordering.  Measured, it does not.
            # ndarray.partition(axis=0), a transpose to make the reduction axis
            # contiguous, and sum-minus-the-top-(M-keep) all came out within
            # noise of the plain sort or slightly worse (1.29 / 1.29 / 1.34 vs
            # 1.20 ms per call at M=12, n=18720).  With M this small the sort is
            # effectively an insertion sort and the cost is dominated by the
            # strided memory traffic over the (M, len(win)) stack, which none of
            # the alternatives reduce.  Left as a sort deliberately — the win
            # here is a memory-layout problem, not an algorithmic one, and on
            # the embedded side the fix is to fuse the trim INTO the per-segment
            # loop (running keep-smallest per lag, never materialising the
            # stack) rather than to swap the selection algorithm.
            _UE.sort(axis=0)
            _DE.sort(axis=0)
            ue = _UE[:_keep].sum(axis=0)
            de = _DE[:_keep].sum(axis=0)
        else:
            B  = np.fft.rfft(win, nfft)
            ue = np.abs(np.fft.irfft(B * self._Fup, nfft)[:len(win)])
            de = np.abs(np.fft.irfft(B * self._Fdn, nfft)[:len(win)])
        lo = max(0, centre - self.track_search)
        hi = min(len(win), centre + self.track_search)
        if hi <= lo:
            return 0.0, centre, -60.0, 0.0
        env = np.maximum(ue, de)
        u = float(ue[lo:hi].max()); d = float(de[lo:hi].max())   # wide: snr/winner
        pk  = lo + int(np.argmax(env[lo:hi]))                    # wide: timing/found_abs
        # Direction: single best combined peak in the TIGHT on-grid window.
        dlo = max(0, centre - self.decide_search)
        dhi = min(len(win), centre + self.decide_search)
        dp  = pk if dhi <= dlo else dlo + int(np.argmax(env[dlo:dhi]))
        du  = float(ue[dp]); dd = float(de[dp])
        # Arithmetic-noise floor: below this the correlation carries no signal
        # at all, only round-off, and neither the SNR ratio nor the up/down
        # confidence ratio means anything. See corr_noise_floor() for why this
        # is computed from the window scale and template length rather than a
        # fixed constant.
        floor = corr_noise_floor(win, self.sps)
        med = max(float(np.median(env)), floor) + 1e-9
        snr = 20.0 * np.log10(max(u, d, floor, 1e-9) / med)
        if max(du, dd) <= floor:
            return 0.0, pk, snr, max(u, d)   # no detection: no direction either
        return (du - dd) / (du + dd + 1e-12), pk, snr, max(u, d)


def measure_mf_noise_floor(mf_symbol, sps, track_search, *,
                           n=128, pct=99.0, seed=0):
    """The MF symbol SNR that PURE NOISE produces on this filter, in dB.

    `snr` out of MFSymbolFilter is 20*log10(peak(env) / median(env)) -- a ratio
    of the correlation envelope to itself.  On noise-only input that ratio is
    invariant to the noise LEVEL (measured: 14.38 dB at every level from 1e-4
    to 0.5, identical to two decimals) and is set instead by the geometry: 10.3
    dB at symbol_dur=0.12 s, 14.4 at 0.5 s, 14.9 at 2.0 s, and 8.3 to 16.9
    across the seg_M/seg_trim settings in use.

    That makes "is this symbol a chirp or is it noise" answerable with an
    ABSOLUTE threshold -- but only one derived from the filter, never a
    constant.  The old fixed mf_min_snr_db=8.0 sat BELOW this floor for every
    geometry in the profile table, so every noise symbol classified STRONG.
    Three separate symptoms came out of that one number:

      * the wake-word gate needs a weak->strong edge and never saw one, so
        preamble_mode="wake" could not fire with any noise in the gap;
      * noise-only windows were admitted to the historic grid fit as timing
        evidence, on the same footing as real chirps;
      * a Barker match on 13 noise symbols looked exactly as strong as one on
        13 chirps, so nothing downstream could tell them apart.

    Measured rather than modelled: it costs a few dozen correlations once at
    startup, it is exact for whatever seg_M/seg_trim/track_search this decoder
    actually built, and it works unchanged for the C backend -- which an
    analytical expression would not, since the two would then have to be kept
    in agreement by hand.

    A high percentile rather than the mean, because the gate has to sit above
    what noise DOES reach, not above its average: at 0.5 s the mean is 14.4 dB
    but individual windows reach 16.3.  `n` is large enough that the percentile
    describes that tail rather than just landing on the largest of a handful of
    draws -- a decoder listening through a quiet minute sees thousands of noise
    symbols, so the estimate has to be of the distribution, not of the sample.

    For reference on the other side, a real chirp at -18 dB link SNR still
    reads 26.9 dB, so the separation being protected is ~10 dB wide and none of
    these choices is delicate.  Measured cost of the whole change, against the
    old constant: identical cold acquisition down to -21 dB SNR, and at -24 dB
    the old threshold managed one trial in three where this manages none.
    """
    rng = np.random.default_rng(seed)
    win_len = sps + 2 * track_search
    vals = []
    for _ in range(n):
        win = rng.standard_normal(win_len).astype(SIG)
        vals.append(mf_symbol(win, sps + track_search)[2])
    return float(np.percentile(vals, pct))


def assemble_message(decoded_frames, chunk_size=0, message_len=None):
    """Reassemble the message from decoded frames, without the encoder's padding.

    Returns (text, raw_bytes).

    The wire format is [seq(1)][data(chunk_size)][crc32(4)] and the encoder pads
    the FINAL chunk with NUL bytes to fill it (css_encoder: "Pad the last (short)
    chunk to chunk_size with zero bytes"), with the CRC computed over the padded
    chunk.  Nothing on the wire distinguishes that padding from payload, so a
    5-byte message at chunk_size=2 decodes as 'hello\\x00' -- which is what run
    20260727_194609 recorded and what the regression suite has been failing on.

    `message_len` is the only thing that can tell them apart.  The encoder writes
    it into css_info.json, which the decoder is already given (it reads
    chunk_size, preamble and frames from there), so no wire change is needed and
    ALREADY-RECORDED captures decode correctly.

    Trimming is per FRAME, keyed on its own seq, not on the joined result: when
    an earlier frame is missing the join is not contiguous, and trimming the
    total to message_len would then cut real payload off the end.

    Without message_len nothing is trimmed.  A payload whose genuine last byte is
    0x00 must survive, and there is no way to guess -- so an unknown-length
    reception keeps the padding rather than risking corruption.

    Bytes are joined and decoded ONCE, rather than per frame, so a multi-byte
    character split across two frames now decodes properly too.
    """
    frames = sorted((f for f in decoded_frames if "error" not in f),
                    key=lambda f: f.get("seq", 0))
    # seq is a byte (css_encoder: seq_num = (i // chunk_size) % 256), so past 256
    # frames it repeats and neither the ordering nor "which frame is last" can be
    # recovered from it.  Don't trim what can't be located.
    trim = (message_len is not None and chunk_size
            and message_len <= 256 * chunk_size)
    out = bytearray()
    for f in frames:
        hexs = f.get("bytes_hex")
        if hexs:
            try:
                dat = bytes.fromhex(hexs)
            except ValueError:
                dat = str(f.get("text", "")).encode("utf-8", "replace")
        else:
            dat = str(f.get("text", "")).encode("utf-8", "replace")
        if trim:
            start = int(f.get("seq", 0)) * chunk_size
            dat = dat[:max(0, min(len(dat), message_len - start))]
        out += dat
    return bytes(out).decode("utf-8", errors="replace"), bytes(out)


def _frame_pipeline(fb, fc, *, interleave_rows, conv_code, diff_enc, diff_ref,
                    base_bits, rs_codec, chunk_size, era_thr=0.0,
                    frame_snr=0.0, verbose=False):
    """deinterleave → Viterbi → diff → RS → CRC on a frame's bits.

    The ground-truth frame-alignment oracle shared by BOTH decoders (the
    streaming PLL `decode()` and the long-symbol MF `decode_longsym_mf()`) and
    by the CRC-anchored offset search that both run.  Pure: no captured state,
    the only side effects are optional diagnostic prints when verbose.

    Returns (ok, seq, dat, rs_err, rs_fail).  Callers that don't track RS error
    counts simply ignore rs_err.
    """
    fb = list(fb); fc = list(fc)
    if interleave_rows > 1:
        n_coded = len(fb)
        cols    = n_coded // interleave_rows
        idxs    = (np.arange(n_coded)
                   .reshape(cols, interleave_rows)
                   .T.flatten().tolist())
        fb = [fb[i] for i in idxs]
        fc = [fc[i] for i in idxs]
    if conv_code == "k7r12":
        sp = [(fc[i], fc[i + 1]) for i in range(0, len(fc) - 1, 2)]
        fb, pe = viterbi_k7_with_reliability(sp)
        me = max(pe) if pe else 1.0
        fc = [e / (me + 1e-12) for e in pe]
    if diff_enc:
        fb = diff_decode_bits(fb, ref_bit=diff_ref)
    raw = get_bytes_from_bits(fb[:base_bits])
    era = []
    if rs_codec is not None and era_thr > 0:
        for bi in range(len(raw)):
            b0 = bi * 8
            b1 = min(b0 + 8, len(fc))
            if b1 > b0 and min(abs(c) for c in fc[b0:b1]) < era_thr:
                era.append(bi)
        if era and verbose:
            print(f"ERASURES n={len(era)} bytes={era[:8]}", flush=True)
    rs_err = 0; rs_fail = None
    if rs_codec is not None:
        try:
            raw, rs_err = rs_decode_frame(raw, rs_codec, era or None)
            if rs_err and verbose:
                print(f"RS_CORRECTED errors={rs_err} "
                      f"snr={frame_snr:.1f}dB", flush=True)
        except Exception as e:
            rs_fail = str(e)
            if verbose:
                print(f"RS_FAIL snr={frame_snr:.1f}dB err={e}", flush=True)
            raw = b""
    ok, seq, dat = check_crc(raw, chunk_size)
    return ok, seq, dat, rs_err, rs_fail


def _anchor_decode(audio_hist, run_pipeline, *, anchor_enabled, frame_symbols,
                   up_ref, down_ref, sps, sps_eff, retro_anchor_max_shift,
                   rate_est_smp, frame_snr=0.0):
    """CRC-anchored offset search over the buffered raw audio history.

    Runs `run_pipeline` (the frame-decode oracle) at each whole-symbol shift
    proposed by retro_offset_candidates and returns (ok, seq, dat) for the
    first CRC-valid alignment, else (False, None, None).  Pure w.r.t. the
    decoder: reads only its arguments, mutates nothing.
    """
    if not (anchor_enabled and len(audio_hist) >= frame_symbols + 2):
        return False, None, None
    try:
        anchor_audio = np.concatenate(list(audio_hist))
        for _cand in retro_offset_candidates(
                anchor_audio, up_ref, down_ref, sps, sps_eff,
                frame_symbols, retro_anchor_max_shift,
                rate_est_smp=rate_est_smp):
            ok, seq, dat, _, _ = run_pipeline(_cand[0], _cand[1], frame_snr)
            if ok:
                return True, seq, dat
    except Exception as exc:
        print(f"RETRO_ANCHOR retry error: {exc}", flush=True)
    return False, None, None


def decode_longsym_mf(cfg, stop_event=None):
    """Per-symbol matched-filter decoder with a running historic grid fit.

    For long symbols each chirp is a strong, unambiguous MF target, so acquiring
    timing from noise with the streaming PLL (decode()) is both unnecessary and
    fragile — at cold start the PLL drifts on lead-in noise and walks the symbol
    grid past the preamble before it locks (a clean 0.99-correlated recording
    decoded 0/2 frames).  Here instead we:
      • predict each chirp's position and matched-filter EVERY symbol there
        (per-symbol focus), reusing the tested ChirpCorrelator;
      • keep a rolling history of detected chirp-peak positions and robustly
        line-fit (t0, sps_eff) over it to predict the next position and track
        clock drift / alignment (the longer-running historic view);
      • detect the Barker preamble in the clean up/down stream and decode each
        frame through the same FEC/CRC pipeline as decode().
    """
    import sounddevice as sd

    info_path   = Path(cfg.get("info",        "css_info.json"))
    result_path = Path(cfg.get("result_out",  "css_decode_result.json"))
    timeout_sec = float(cfg.get("timeout_sec", 600))
    device      = cfg.get("device")
    if device is not None:
        try:
            device = int(device)
        except (TypeError, ValueError):
            device = None
    wav_out  = cfg.get("wav_out")
    _capture = [] if wav_out else None

    # Cooperative stop: the wrapper (css_listen.py) hard-kills this subprocess
    # with proc.terminate() once the player signals css_played.  On Windows that
    # is TerminateProcess — uncatchable — so a SIGTERM handler can't flush the
    # buffered WAV (and this long-symbol path has no handler anyway).  Instead
    # the wrapper creates this sentinel file; we poll it each symbol read and
    # set eos, letting the loop drain and fall through to the WAV-write below.
    stop_file = cfg.get("stop_file")

    result = {
        "status": "error", "failure_reason": None,
        "frames_decoded": 0, "frames_errored": 0,
        "decoded_text": "", "diagnostics": {}, "frames": [],
    }
    conf  = json.loads(info_path.read_text())
    fs    = int(  cfg.get("fs",         conf["fs"]))
    dur   = float(cfg.get("symbol_dur", conf["symbol_dur"]))
    f_min = float(cfg.get("f_min",      conf["f_min"]))
    f_max = float(cfg.get("f_max",      conf["f_max"]))
    # Optional streaming hook: called with each decoded frame dict AS IT ARRIVES,
    # so a long-running (continuous-RX) caller gets frames live instead of only
    # from the return value.  A callback exception must never crash decoding.
    _on_frame = cfg.get("on_frame", None)
    sps   = int(fs * dur)
    gap   = int(round(sps * float(conf.get("symbol_gap_frac", 0.0))))
    sps_eff         = sps + gap
    chunk_size      = int(conf.get("chunk_size", 1))
    rs_t            = int(conf.get("rs_t", 0))
    interleave_rows = int(conf.get("interleave_rows", 0))
    conv_code       = conf.get("conv_code")
    diff_enc        = bool(conf.get("diff_encode", False))
    preamble        = list(cfg.get("preamble") or conf.get("preamble") or BARKER_13)
    P               = len(preamble)
    frame_symbols   = int(conf.get("frame_symbols", conf.get("base_bits", 48)))
    base_bits       = int(conf.get("base_bits", frame_symbols))
    train_n         = int(conf.get("training_symbols", 0))
    expected_frames = conf.get("frames")
    # Payload length before the encoder padded the final chunk (see
    # assemble_message).  None on captures older than this field.
    message_len     = conf.get("message_len")
    diff_ref        = preamble[-1]

    up_ref   = generate_chirp(f_min, f_max, dur, fs)
    down_ref = generate_chirp(f_max, f_min, dur, fs)
    search_frac = float(cfg.get("timing_search_frac", 0.25))
    corr = ChirpCorrelator(up_ref, down_ref, sps, search_frac=search_frac,
                           buf_size=4 * sps_eff,
                           mf_window=str(cfg.get("mf_window", "none")))
    search   = corr.search_range          # wide ± window, acquisition only
    rs_codec = _make_rs_codec(rs_t) if rs_t > 0 else None

    # Per-symbol matched filter parameters (see MFSymbolFilter for rationale).
    track_search = int(cfg.get("mf_track_search_smp", max(64, int(sps * 0.15))))
    # Direction-decision window: TIGHT, distinct from the wide timing search.
    # The up/down bit is decided on-grid within ±decide_search of the predicted
    # centre; the wider ±track_search is used only to locate the chirp for the
    # historic grid fit.  See _mf_symbol for why these must differ.  The default
    # (±0.06 sym) was swept end-to-end through the real decoder on both the mic
    # recording and the clean encoder output: ±0.06 recovers the reverberant mic
    # run (partial 1/2 "h" -> ok 2/2 "hi") while keeping the clean run at 2/2;
    # tighter (≤0.05) starves early-acquisition timing and slips clean sync
    # (19 -> 84, "i" only), wider (≥0.07) lets the reflection back in and the mic
    # regresses to 1/2.
    decide_search = int(cfg.get("mf_decide_search_smp", max(48, int(sps * 0.06))))
    # Impulse blanking (MF-path analogue of decode()'s blanking): zero samples
    # whose |x| exceeds mf_blank_kmad × the window's median |x| before
    # correlating.  Superseded by the segmented matched filter below (which
    # bounds burst influence structurally instead of by threshold) — kept as an
    # optional extra defence, default OFF.  0 disables.
    blank_kmad = float(cfg.get("mf_blank_kmad", 0.0))
    # ── Segmented matched filter (burst-noise-robust decision statistic) ─────
    # The plain MF decides from the envelope PEAK VALUE, which responds to
    # ENERGY, not chirp structure: any broadband input produces output speckle
    # with the same ~fs/BW correlation width as a true compressed chirp peak
    # (measured: a 185 ms thud carrying 55× the chirp's energy produced a false
    # peak 7 smp wide at 7× the signal peak — no output-domain test can reject
    # it).  The chirp's actual fingerprint is energy spread UNIFORMLY along the
    # full symbol duration, so: split the template into mf_segments time slices
    # (masked slices of the reversed template — all segment correlations align
    # at the same output lag), correlate each separately, and at every lag sum
    # the per-segment envelopes AFTER dropping the mf_seg_trim largest.  A burst
    # can only inflate the segments it overlaps (a 185 ms burst touches ≤2 of 12
    # on a 3 s symbol), and those are exactly the ones trimmed — the breakdown
    # point is trim/M by construction, with no amplitude threshold to re-tune
    # per noise type.  A real chirp is uniform across segments, so trimming
    # costs it only trim/M of amplitude.  Trade-offs: ~1–2 dB non-coherent
    # combining loss at very low SNR, and a wider timing peak (~M·fs/BW), both
    # negligible against the ±track_search grid fit and ±decide_search decision
    # windows.  mf_segments <= 1 restores the classic single-template MF.
    seg_M    = int(cfg.get("mf_segments", 12))
    # seg_trim must be derived from SYMBOL DURATION, not left constant.  The
    # breakdown point of the trimmed sum is trim/M as a FRACTION OF THE SYMBOL,
    # so a fixed trim protects a shrinking absolute burst length as symbols get
    # shorter.  The old default (2 of 12) was sized against the 185 ms thud on a
    # 3 s symbol quoted above: 250 ms per segment, so the burst touches at most
    # 2 segments.  At symbol_dur=0.5 s the same 185 ms burst spans 4-5 of 12
    # segments (41.7 ms each) and sails straight through a trim of 2.
    #
    # Measured on a real 0.5 s OTA capture whose payload carried a ~170 ms
    # impulse pair (run 20260727_180939_css): trim/M of 17% and 25% both decoded
    # 3/5 frames, 31% gave 4/5, and 33% or more gave 5/5 — the knee sits exactly
    # where the burst's symbol fraction predicts.
    #
    # So express the intent as the burst length to survive and solve for trim.
    # A burst of B ms straddles segment boundaries, so it can touch
    # ceil(B / seg_len) + 1 segments.  This reproduces trim=2 at the 3 s /
    # 185 ms design point that the paragraph above was tuned for, while scaling
    # correctly down to short symbols.  Capped at M//2: past half the segments
    # the trimmed sum has more of the chirp discarded than kept, and the
    # non-coherent combining loss starts costing more sensitivity than the burst
    # immunity buys.  Set mf_seg_trim explicitly to override entirely.
    #
    # The cap is M//3, NOT M//2, and that bound is empirical rather than
    # cosmetic.  At symbol_dur=0.5 s the raw formula asks for 6 of 12 (50%);
    # that decoded the burst-damaged OTA capture fine but cost enough
    # sensitivity to LOSE the cold-start frame in both wake-mode regression
    # cases (synth_wake_15s_suspend and synthrec_fw_wake both fell from "hi" to
    # "i").  Capping at M//3 yields 4 of 12 at 0.5 s — the smallest trim the
    # sweep showed recovering that capture at 5/5 — and keeps the wake cases
    # passing.  Raising this cap without re-running the regression suite will
    # silently trade cold-start acquisition for burst immunity.
    _seg_len_ms = (dur * 1000.0) / max(1, seg_M)
    _burst_ms   = float(cfg.get("mf_seg_burst_ms", 185.0))
    if seg_M > 1 and _seg_len_ms > 0:
        _auto_trim = int(-(-_burst_ms // _seg_len_ms)) + 1      # ceil(B/L) + 1
        _auto_trim = max(1, min(max(1, seg_M // 3), _auto_trim))
    else:
        _auto_trim = 0
    seg_trim = int(cfg.get("mf_seg_trim", _auto_trim))
    if seg_M > 1:
        print(f"MF_SEGMENTS M={seg_M} trim={seg_trim} "
              f"(seg_len={_seg_len_ms:.1f}ms, survives_burst<={_burst_ms:.0f}ms, "
              f"breakdown={seg_trim / seg_M * 100:.0f}% of symbol)", flush=True)
    # Matched-filter backend.  CSS_MF_BACKEND=c routes the hot inner loop
    # through tests/CSS/c/libcssmf.so (the freestanding C that ships to the
    # embedded target), loaded in-process via ctypes; anything else keeps the
    # numpy implementation.  Both are held to the same regression suite — that
    # equivalence is the whole point of having the switch, so a port bug shows
    # up as a suite failure rather than as a field mystery.
    _mf_kw = dict(track_search=track_search, decide_search=decide_search,
                  seg_M=seg_M, seg_trim=seg_trim, blank_kmad=blank_kmad)
    if os.environ.get("CSS_MF_BACKEND", "").lower() == "c" and seg_M > 1:
        from css_mf_c import MFSymbolFilterC
        _mf_symbol = MFSymbolFilterC(up_ref, down_ref, sps, **_mf_kw)
        print(f"MF_BACKEND=c nfft={_mf_symbol.nfft} "
              f"workspace={_mf_symbol.workspace_bytes / 1e6:.2f}MB", flush=True)
    else:
        _mf_symbol = MFSymbolFilter(up_ref, down_ref, sps, **_mf_kw)

    # ── "is this symbol a chirp?" -- referenced to the filter's own noise ────
    # See measure_mf_noise_floor.  The threshold is the noise-only value plus a
    # margin, NOT a constant: the noise-only value ranges 8.3-16.9 dB across
    # the geometries in the profile table, so any constant is below the floor
    # for some of them and above real chirps for none of them.  The old 8.0
    # default was below the floor for every geometry in use.
    #
    # The margin has ~10 dB of room to sit in (noise reaches 16.3 dB at 0.5 s;
    # a chirp at -18 dB link SNR still reads 26.9), so 4 dB is comfortably
    # clear of noise without threatening any symbol that carries signal.
    # An explicit mf_min_snr_db still wins, for bisecting against old runs.
    _mf_noise_db = measure_mf_noise_floor(_mf_symbol, sps, track_search)
    _min_snr_margin = float(cfg.get("mf_min_snr_margin_db", 4.0))
    acq_min_snr = float(cfg.get("mf_min_snr_db", _mf_noise_db + _min_snr_margin))
    print(f"MF_NOISE_FLOOR p99={_mf_noise_db:.1f}dB "
          f"margin={_min_snr_margin:.1f} -> strong>={acq_min_snr:.1f}dB"
          + (" (overridden)" if "mf_min_snr_db" in cfg else ""), flush=True)
    # ── Acquisition detection (two-stage, low-SNR capable) ──────────────────
    # Stage A gate: candidate peak vs a CFAR-style noise reference (median of
    # the MF envelope EXCLUDING a guard region around the peak).  The old
    # plain peak/median at ratio 15 was miscalibrated: the buffer median is
    # not a pure noise floor, and the measured ratio saturated ~13 even at
    # 20 dB per-symbol SNR — the decoder refused signals whose bits were
    # perfectly readable.  Stage A is now a LOW bar (default 6, just above
    # the ~4 extreme-value ratio pure noise produces over a 3-symbol buffer);
    # false candidates are cheap because stage B rejects them.
    acq_ratio   = float(cfg.get("mf_acq_peak_ratio",   6.0))
    acq_guard   = max(16, sps // 8)   # CFAR guard half-width around the peak
    # Stage B gate: multi-symbol GRID-PHASE confirmation.  A genuine chirp
    # sits on the frame's symbol grid, so its neighbours at ±k*sps_eff are
    # also chirp PEAKS, while the positions half a symbol off that grid hold
    # only correlation skirt / reverb tail.  Absolute per-slot SNR cannot
    # discriminate (measured: reverb keeps off-grid slots at 14-18 dB, within
    # a few dB of on-grid), so the test is a CONTRAST: sum of slot peak
    # amplitudes at the candidate's grid phase vs the SAME slots shifted by
    # sps_eff/2.  True grid -> on >> off (measured ~2.3x); a candidate that is
    # itself off-grid finds its half-shifted phase landing near the TRUE grid
    # -> on < off -> rejected; pure noise -> on ~ off -> rejected.  This is
    # non-coherent combining across mf_acq_confirm_slots symbols each side
    # (~10*log10(N) dB detection gain) — the same evidence a human uses on
    # the spectrograph ("several evenly spaced sweeps", never "one bright
    # blob") — and is what lets stage A run loose without the old
    # false-lock-on-early-noise failure.  mf_acq_confirm_min additionally
    # requires that many on-slots at chirp strength (>= mf_min_snr_db); set
    # it to 0 to disable stage B entirely (single-symbol behaviour).
    acq_confirm_slots    = int(cfg.get("mf_acq_confirm_slots", 3))
    acq_confirm_min      = int(cfg.get("mf_acq_confirm_min",   2))
    acq_confirm_contrast = float(cfg.get("mf_acq_confirm_contrast", 1.25))
    hist_len    = int(  cfg.get("mf_grid_hist",        24))
    refit_every = int(  cfg.get("mf_grid_refit_every",  4))
    # ── Lock-verify watchdog (false-acquisition recovery) ────────────────────
    # A loud non-chirp transient (DAC pop, door slam) can pass BOTH acquisition
    # stages and anchor the grid off the real chirp train (observed: a t=3.14s
    # impulse at 2x chirp amplitude anchored the grid 0.665 sym off; every
    # window then straddled two chirps -> 44/110 bits wrong at high apparent
    # SNR -> CRC fail, with no path back because the grid persists forever).
    # The distinguishing signature of a TRUE lock: the train's chirps are the
    # strongest thing in the buffer and they sit ON-grid, so the buffer's GLOBAL
    # MF peak has phase delta ≈ 0 from the predictions.  (Per-symbol "strong"
    # counts CANNOT discriminate: found_abs comes from the argmax inside the
    # ±track window so it is always "on-grid", and the window-median SNR metric
    # stays high even when the window straddles two chirps.)  So every
    # mf_lock_verify_syms symbols while in SEARCH state, the watchdog correlates
    # the current buffer (same stage-A CFAR gate as acquisition) and checks the
    # global peak's grid phase.  |delta| > track_search on TWO CONSECUTIVE
    # checks agreeing within mf_lock_verify_tol re-phases the grid onto the real
    # train (MF_RELOCK) and clears the bit history.  The two-check confirmation
    # rejects spurious single-window echo peaks: genuine room multipath is
    # < 0.02 sym of delay (well inside track_search) and anything else is not
    # phase-stable across a full window.  Never fires mid-frame (COLLECT).
    # Window of 4 + tol 0.08: a FALSE grid also has a bogus fitted slope, so its
    # phase delta legitimately drifts a few % of a symbol between checks — the
    # tolerance must absorb that, and shorter windows both confirm sooner and
    # keep consecutive deltas closer.  (Measured on the 20260705 false lock:
    # deltas 8 syms apart differed by ~0.06 sym; 0.05 tol missed twice and the
    # late relock cost frame 0.)
    lock_verify_syms = int(cfg.get("mf_lock_verify_syms", 4))
    lock_verify_tol  = float(cfg.get("mf_lock_verify_tol", 0.08))  # × sps_eff
    # Relock evidence must be CLEARLY chirp-strength, not just above the loose
    # stage-A ratio: at poor SNR the buffer's global MF peak is often noise
    # speckle at ratio 6-8, and treating it as evidence made the watchdog
    # re-phase onto noise repeatedly, clearing the bit history each time so the
    # preamble could never assemble (observed on a 0.3 s / 9.5 dB run: two bogus
    # MF_RELOCKs; with the gate the same run decodes fully).  Genuine chirp
    # peaks measure ratio 15-40.
    relock_min_ratio = float(cfg.get("mf_relock_min_ratio", 10.0))
    # The watchdog's evidence model assumes any strong peak OUTSIDE
    # ±track_search is a wrong grid.  That only holds when the symbol is long
    # relative to room acoustics: a phase-stable echo at ~75 ms is 0.02 sym on a
    # 3 s symbol (inside track_search, ignored) but 0.24 sym on a 0.3 s symbol —
    # outside the window, phase-stable, and strong, so it passes BOTH the ratio
    # gate and the two-check confirmation and re-phases the grid onto the echo
    # (observed: killed an otherwise fully-decodable 0.3 s run).  Below this
    # symbol duration the watchdog stays off; the segmented stage-B acquisition
    # confirm remains the primary false-lock defence at all durations.
    lock_verify_min_dur = float(cfg.get("mf_lock_verify_min_dur", 1.0))
    # Barker-13 bit-match gate.  Quantization matters: 13/13=1.00, 12/13=0.923,
    # 11/13=0.846, 10/13=0.769.  The old 0.85 default tolerated exactly ONE
    # errored preamble symbol (12/13) and rejected TWO (11/13=0.846 < 0.85).
    # That cost the FIRST frame on real cold-start runs: acquisition locks a few
    # symbols into frame 0 with a still-settling grid, so the first frame's two
    # leading Barker symbols can flip (observed on a 0.75 s / amp-0.01 over-air
    # run — frame 0 preamble decoded 11/13=0.846, dropped; frame 1, on the warm
    # grid, matched 13/13 and decoded — the classic "loses chunk 0, gets chunk 1"
    # signature).  Loosen to 0.80 so TWO errored symbols still lock (11/13
    # passes, 10/13 does not).  This does not weaken integrity: the |conf| cstr
    # gate below and the payload CRC-32 remain the real validators — a spurious
    # lock just burns one frame window and returns to SEARCH.  Barker-13's peak
    # off-alignment autocorrelation is 1/13, so the extra tolerance does not open
    # sidelobe false locks either.
    bark_thr    = float(cfg.get("mf_preamble_thr",     0.80))
    # ── Frame-cadence guard (see the block at the preamble accept site) ──────
    # bark_thr alone cannot separate a real preamble from an 11/13 accidental
    # match inside payload data; the cadence guard supplies the missing
    # constraint (a real preamble sits an integer number of frame periods from
    # the previous one).  Set mf_preamble_cadence=false to restore the old
    # first-past-the-post behaviour.
    pre_cadence_on       = bool(cfg.get("mf_preamble_cadence",          True))
    pre_cadence_tol      = int( cfg.get("mf_preamble_cadence_tol",         2))
    pre_cadence_max_miss = int( cfg.get("mf_preamble_cadence_max_miss",    3))
    # Consecutive weak symbols that mean "the transmitter stopped", which drops
    # the cadence anchor (see _weak_run).  Three, not one: a single symbol can
    # read weak under an impulse burst or a deep fade without the transmission
    # having ended, and dropping the lattice on that would give back the
    # in-payload false locks it exists to reject.  A real gap is tens of
    # symbols, so nothing is lost by requiring a few.
    cad_reset_weak  = int(cfg.get("mf_cadence_reset_weak", 3))
    # Fraction of the P preamble symbols that must read as chirps, not noise
    # (see the preamble strength gate).  0.6 = 8 of 13, comfortably looser than
    # the bit match's 11 of 13, because this gate is aimed at windows where
    # NONE of the symbols are chirps.
    pre_strong_frac = float(cfg.get("mf_pre_strong_frac", 0.6))
    # Barker match required of the FIRST preamble after a pause, where the
    # cadence lattice has been dropped and cannot vouch for it.  0.92 = 12/13,
    # so one errored symbol still locks; the sidelobe this rejects read 11/13.
    post_gap_thr = float(cfg.get("mf_post_gap_thr", 0.92))
    # Symbols between consecutive frame starts in per_frame mode:
    # training run-in + preamble + payload.
    _frame_period = train_n + P + frame_symbols
    # ── Preamble confidence gate (noise-referenced) ──────────────────────────
    # |conf| = (u-d)/(u+d) compresses toward 0 at low SNR even while every
    # hard bit decision is still CORRECT (u > d), so the old fixed
    # cstr >= 0.4 floor rejected decodable low-SNR frames outright.  Replace
    # it with a floor referenced to the RUNNING NOISE |conf| baseline
    # (_conf_bg, an EMA fed only by weak/noise-like symbols): a genuine
    # preamble must show cstr >= pre_cstr_k × that baseline, never below
    # pre_cstr_floor and never stricter than the old 0.4.  The Barker bit
    # match (bark_thr) plus the payload CRC remain the real validators.
    pre_cstr_floor = float(cfg.get("mf_pre_cstr_floor", 0.12))
    pre_cstr_k     = float(cfg.get("mf_pre_cstr_k",     1.30))

    # ── Wake-word resume detection (suspend/resume modem) ────────────────────
    # In "wake" preamble_mode the encoder sends the full training+Barker ONCE
    # (frame 0), then only a short wake_word before each resumed frame.  The
    # decoder keeps its symbol grid alive (freewheels) across the suspend
    # silence — grid_t0/grid_sps coast on the last fit, hist just stops
    # updating on the weak silence symbols — so on resume it doesn't need to
    # re-acquire timing; it only needs to re-mark the frame boundary.  The wake
    # word is detected ONLY at a silence→signal EDGE (the symbol before the
    # word was weak, the word's own symbols are strong): that edge gate makes a
    # 2-symbol marker unambiguous despite having just 4 possible patterns, and
    # keeps 15 s of intervening noise from ever false-triggering it.
    wake_word    = list(conf.get("wake_word", cfg.get("wake_word", [])) or [])
    _pre_mode    = str(conf.get("preamble_mode",
                                cfg.get("preamble_mode", "per_frame"))).lower()
    wake_enabled = (_pre_mode == "wake") and len(wake_word) > 0
    W            = len(wake_word)
    read_n      = sps_eff
    # Retro acquisition window: acquisition locks on the FIRST chirp strong
    # enough to clear acq_ratio, which is often several symbols INTO frame 0 (its
    # early chirps can be weaker), and the search discards the audio before it.
    # The forward-only tracker then starts mid-frame-0, so frame 0's preamble —
    # whose leading bits precede the anchor — never assembles its full window and
    # the frame is lost (decode is otherwise correct).  We retain this many
    # symbols of pre-lock audio and merge it back on lock so the grid anchor can
    # back up across the missed frame.  Default ≈ one full frame so the first
    # frame is reachable wherever acquisition happens to lock.
    retro_back_syms = int(cfg.get("mf_retro_back_syms", train_n + P + frame_symbols + 4))
    _retro_cap      = max(0, retro_back_syms) * sps_eff

    print(f"LISTENING_MF fs={fs} sps={sps} sps_eff={sps_eff} dur={dur}s gap={gap}smp "
          f"f={f_min:.0f}-{f_max:.0f}Hz pre={P} frame_symbols={frame_symbols} "
          f"rs_t={rs_t} diff={diff_enc} search=±{search}", flush=True)

    # ── Spectrograph telemetry (UDP, fire-and-forget) ────────────────────────
    # Same schema as decode()'s patches [A–D] (css_spectrographtool.py): the
    # viewer reads every field via .get(default), so the MF-specific subset is
    # fine.  All sends are wrapped so a missing listener can never affect decode.
    import socket as _telem_mod
    _telem      = _telem_mod.socket(_telem_mod.AF_INET, _telem_mod.SOCK_DGRAM)
    _TELEM_ADDR = ("127.0.0.1", int(cfg.get("telem_port", 55555)))

    def _send(d):
        try:
            _telem.sendto(json.dumps(d).encode(), _TELEM_ADDR)
        except Exception:
            pass

    def _pipeline(fb, fc):
        """Frame decode for the MF path — thin adapter over _frame_pipeline
        (no RS erasures here; returns the 4-tuple this path expects)."""
        ok, seq, dat, _rs_err, rs_fail = _frame_pipeline(
            fb, fc, interleave_rows=interleave_rows, conv_code=conv_code,
            diff_enc=diff_enc, diff_ref=diff_ref, base_bits=base_bits,
            rs_codec=rs_codec, chunk_size=chunk_size)
        return ok, seq, dat, rs_fail

    if _log_audio_interface is not None and hasattr(sd, "query_devices"):
        try:
            _log_audio_interface("input", device, fs, channels=1)
        except Exception:
            pass
    if hasattr(sd, "query_devices"):
        _input_ctx = _QueuedInputStream(sd, fs, device, channels=1)
    else:
        _input_ctx = sd.InputStream(samplerate=fs, channels=1, dtype="float32",
                                    device=device)

    decoded_frames = []
    frames_decoded = 0
    frames_errored = 0
    diag = {"sync_acquired": False, "sync_at_symbol": None, "sync_score": 0.0,
            "symbols_received": 0, "peak_rms": 0.0, "mean_rms": 0.0,
            "silence": False, "avg_snr_db": 0.0, "input_overflows": 0}
    snr_log = []
    _decoded_seqs = set()

    started   = time.time()
    # ── Compute-cost profiling (for porting / min-hardware sizing) ───────────
    # process_time() = CPU time consumed by THIS process (excludes time blocked
    # waiting on the audio stream), so offline (audio arrives instantly) the
    # ratio cpu_time / audio_seconds is the true real-time factor: RTF<1 means a
    # CPU this fast keeps up; 1/RTF is the headroom, i.e. how many times slower a
    # target CPU can be and still run real-time.  perf_counter() is wall time.
    _prof_cpu0  = time.process_time()
    _prof_wall0 = time.perf_counter()
    buf       = np.zeros(0, dtype=np.float32)
    buf_start = 0     # abs index of buf[0]
    total_abs = 0     # abs index one past the last sample read
    eos       = False
    peak_rms  = 0.0
    sym       = 0

    def _read_more():
        nonlocal buf, total_abs, eos, peak_rms
        try:
            raw, _ov = stream.read(read_n)
        except Exception:
            eos = True; return 0
        chunk = np.asarray(raw, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            eos = True; return 0
        if _capture is not None:
            _capture.append(chunk.copy())
        r = float(np.sqrt(np.mean(chunk ** 2)))
        if r > peak_rms:
            peak_rms = r
        buf = np.concatenate([buf, chunk])
        total_abs += chunk.size
        if stop_event is not None and stop_event.is_set():
            eos = True
        if stop_file and Path(stop_file).exists():
            eos = True
        return chunk.size

    try:
        with _input_ctx as stream:
            # ── Acquisition: first strong chirp -> grid phase ────────────────
            # Lifetime is driven by the FRAMEWORK, not a fixed acquisition
            # window: while the test runs the decoder stays active and keeps
            # writing the WAV (via _read_more -> _capture).  Termination comes
            # from end-of-stream / the cooperative stop the player triggers
            # (css_played -> listener drops stop_file / sets stop_event -> eos),
            # NOT from giving up after N seconds.  This guarantees a full-length
            # recording is always captured for offline/visual decode even when
            # sync never locks (e.g. the low-SNR challenge).  timeout_sec is
            # only a hang backstop in case the player never signals completion.
            acq_deadline = started + timeout_sec
            t0_abs = None
            # Bounded history of audio discarded during the search, kept so the
            # grid anchor can reach back over a frame the search skipped past
            # (see retro_back_syms above).  Always contiguous and immediately
            # before buf, i.e. covers abs [buf_start - len(retro_buf), buf_start).
            retro_buf = np.zeros(0, dtype=np.float32)
            while True:
                if (stop_event is not None and stop_event.is_set() and eos):
                    break
                if time.time() > acq_deadline:
                    break
                while (total_abs - buf_start) < 3 * sps_eff and not eos:
                    if _read_more() == 0:
                        break
                if len(buf) < 2 * sps:
                    if eos:
                        break
                    continue
                up_e, dn_e = corr.correlate(buf)
                comb = np.maximum(up_e[:len(buf)], dn_e[:len(buf)])
                peak_idx = int(np.argmax(comb)) if comb.size else 0
                mx  = float(comb[peak_idx]) if comb.size else 0.0
                # Stage A: CFAR noise reference — median of the envelope with
                # a guard region around the candidate excluded, so the
                # statistic measures peak-vs-NOISE, not peak-vs-everything.
                # See the acq_ratio comment block above.
                if comb.size:
                    _nmask = np.ones(len(comb), dtype=bool)
                    _nmask[max(0, peak_idx - acq_guard):peak_idx + acq_guard] = False
                    noise_ref = (float(np.median(comb[_nmask])) if _nmask.any()
                                 else float(np.median(comb))) + 1e-9
                else:
                    noise_ref = 1.0
                accept = mx > 1e-9 and (mx / noise_ref) >= acq_ratio

                # Stage B: grid-phase contrast confirmation (see the
                # acq_confirm_slots comment block above).  Read far enough
                # ahead to test forward slots, splice in just enough retro_buf
                # tail for the backward slots, then compare summed slot peaks
                # at the candidate's grid phase against the same slots shifted
                # half a symbol.
                if accept and acq_confirm_min > 0:
                    cand_abs  = buf_start + peak_idx
                    _half     = sps_eff // 2
                    _need_abs = (cand_abs + acq_confirm_slots * sps_eff
                                 + _half + track_search + 8)
                    while total_abs < _need_abs and not eos:
                        if _read_more() == 0:
                            break
                    _need_back = max(0, (sps + track_search
                                         + acq_confirm_slots * sps_eff)
                                        - (cand_abs - buf_start))
                    _rt = (retro_buf[len(retro_buf)
                                     - min(_need_back, len(retro_buf)):]
                           if _need_back else retro_buf[:0])
                    ext       = np.concatenate([_rt, buf]) if len(_rt) else buf
                    ext_start = buf_start - len(_rt)

                    def _slot_peak(c_abs):
                        """(peak_amplitude, snr_db) of the MF response in the
                        tracking slice around c_abs, or None if out of range."""
                        _lo = c_abs - sps - track_search - ext_start
                        _hi = c_abs + track_search - ext_start
                        if _lo < 0 or _hi > len(ext):
                            return None
                        _cc, _cpk, _csnr, _cwpk = _mf_symbol(
                            ext[_lo:_hi], sps + track_search)
                        return _cwpk, _csnr

                    nconf = 0
                    ntest = 0
                    _sum_on  = 0.0
                    _sum_off = 0.0
                    for _k in range(-acq_confirm_slots, acq_confirm_slots + 1):
                        if _k == 0:
                            continue
                        _on  = _slot_peak(cand_abs + _k * sps_eff)
                        _off = _slot_peak(cand_abs + _k * sps_eff + _half)
                        if _on is None or _off is None:
                            continue        # unbalanced pair: skip entirely
                        ntest += 1
                        _sum_on  += _on[0]
                        _sum_off += _off[0]
                        if _on[1] >= acq_min_snr:
                            nconf += 1
                    # Slots outside the recording don't count against the
                    # candidate; if NO pair was testable, fall through on the
                    # stage-A verdict alone rather than deadlocking.
                    _required = min(acq_confirm_min, ntest)
                    _contrast = _sum_on / (_sum_off + 1e-12) if ntest else 0.0
                    accept = (ntest == 0
                              or (nconf >= _required
                                  and _contrast >= acq_confirm_contrast))
                    print(f"MF_ACQ_CANDIDATE t={cand_abs / fs:.2f}s "
                          f"ratio={mx / noise_ref:.1f} "
                          f"confirms={nconf}/{ntest} (need {_required}) "
                          f"contrast={_contrast:.2f} "
                          f"(need {acq_confirm_contrast:.2f}) "
                          f"-> {'ACCEPT' if accept else 'REJECT'}", flush=True)

                if not accept:
                    drop = max(0, len(buf) - 2 * sps_eff)
                    if drop and _retro_cap:
                        # Retain the audio we're about to discard (contiguous,
                        # bounded to _retro_cap) so it can be merged back on lock.
                        retro_buf = np.concatenate([retro_buf, buf[:drop]])
                        if len(retro_buf) > _retro_cap:
                            retro_buf = retro_buf[len(retro_buf) - _retro_cap:]
                    buf = buf[drop:]; buf_start += drop
                    if eos:
                        break
                    continue
                t0_abs = buf_start + peak_idx
                # Merge retained pre-lock audio back so the anchor can back up
                # across an earlier frame the search skipped (retro_buf is the
                # contiguous audio immediately before buf_start).
                if len(retro_buf):
                    buf = np.concatenate([retro_buf, buf])
                    buf_start -= len(retro_buf)
                    print(f"MF_RETRO merged {len(retro_buf)} smp "
                          f"({len(retro_buf)/fs:.2f}s) of pre-lock audio "
                          f"-> buf_start={buf_start} ({buf_start/fs:.2f}s)", flush=True)
                break

            if t0_abs is not None:
                # Back up the grid anchor to the earliest chirp still in the
                # buffer (we only step forward, so anchoring on a later chirp
                # would miss the first frame's preamble).
                anchor = float(t0_abs)
                while anchor - sps_eff >= buf_start + sps + track_search:
                    anchor -= sps_eff
                grid_t0  = anchor          # peak_abs of symbol index 0
                grid_sps = float(sps_eff)
                print(f"MF_ACQUIRE chirp@{t0_abs} ({t0_abs/fs:.2f}s) "
                      f"grid_anchor={int(anchor)} sps_eff={sps_eff}", flush=True)
                hist     = deque(maxlen=hist_len)
                stream_bits = []; stream_confs = []
                # Running noise |conf| baseline for the preamble cstr gate:
                # fed only by weak (noise-like) symbols so continuous strong
                # signal cannot inflate it.  Starts at a conservative prior.
                _conf_bg = 0.25
                # Per-symbol strong/weak flags (last few), for the wake-word
                # silence→signal edge gate.  maxlen covers the word + the one
                # preceding symbol whose weakness proves we're at a resume edge.
                _strong_hist = deque(maxlen=max(2, W) + 2)
                state = "SEARCH"; payload_b = []; payload_c = []
                weak_after_eos = 0
                _lv_seen = 0; _lv_pending = None   # lock-verify watchdog state
                # Symbol index of the last ACCEPTED Barker lock, anchoring the
                # frame-cadence guard below.  None = no cadence known yet, so
                # the guard stands down (first acquisition, after a relock
                # invalidates the lattice, or after the transmission ends --
                # see _weak_run).
                _last_pre_sym = None
                # Consecutive weak (noise-like) symbols, and the total silence
                # observed since the last accepted preamble.  A transmission has
                # a chirp in every slot -- the inter-symbol gap is 5% of a
                # symbol, not a whole one -- so a run of weak symbols is the
                # transmitter being OFF.
                #
                # That total is what makes the frame-cadence lattice survive a
                # suspend gap.  The lattice says a real preamble sits a whole
                # number of frame periods from the last one, which is true in
                # SIGNAL time and false in wall-clock symbols the moment the
                # transmitter pauses.  Discounting the silence converts one to
                # the other, so the guard keeps working across a gap instead of
                # having to be dropped for it -- and dropping it was itself a
                # regression: with no lattice, a Barker sidelobe landing inside
                # the next frame's TRAINING run-in is accepted, the payload is
                # read 16 symbols early and dies on CRC.  Measured on an 8-frame
                # 24 dB run at training_symbols=12, which lost exactly one frame
                # that way; longer run-ins give the sidelobe more room, which is
                # why the damage rose with training length.
                #
                # Only runs of cad_reset_weak or more are counted, so an
                # isolated weak symbol from a fade or a blanked impulse does not
                # quietly shift the lattice.
                _weak_run = 0
                # True while the next preamble is the first after a pause, i.e.
                # the one the dropped lattice cannot vouch for.  See _post_gap
                # at the accept site.
                _post_gap = False
                # Whether a real Barker preamble has EVER locked.  Gates the
                # wake-word path (see the false-wake note there).  Distinct from
                # _last_pre_sym, which a relock deliberately clears — having seen
                # a preamble is not undone by re-phasing the grid.
                _barker_seen = False
                max_syms = int(timeout_sec / max(dur, 1e-3)) + 4000

                while sym < max_syms:
                    pred_abs = int(round(grid_t0 + sym * grid_sps))
                    left  = pred_abs - sps - track_search
                    right = pred_abs + track_search
                    while total_abs < right + 8 and not eos:
                        if _read_more() == 0:
                            break
                    if right + 8 > total_abs:
                        if eos:
                            break
                        continue
                    if left < 0:
                        sym += 1; continue          # too close to stream start
                    # Trim consumed audio, but KEEP one extra symbol of history:
                    # an MF_RELOCK can re-phase the grid up to sps_eff/2 EARLIER
                    # and re-decode the current symbol there — without this
                    # retention that window is already trimmed away and the
                    # symbol is skipped (observed: the skip cost the first
                    # preamble bit and with it the whole first frame).
                    if buf_start < left - sps_eff:
                        drop = (left - sps_eff) - buf_start
                        if 0 < drop < len(buf):
                            buf = buf[drop:]; buf_start += drop
                    if left < buf_start:
                        sym += 1; continue          # lost left-history; skip
                    win = buf[left - buf_start : right - buf_start]
                    if len(win) < sps + track_search:
                        if eos:
                            break
                        continue
                    centre_in = pred_abs - left     # = sps + track_search
                    c, pk, snr_db, wpk = _mf_symbol(win, centre_in)
                    bit = 1 if c >= 0 else 0         # up-chirp = 1 (matches encoder)
                    found_abs = left + pk
                    strong = snr_db >= acq_min_snr
                    snr_log.append(snr_db)
                    if not strong:
                        # Weak symbol => the MF up/down discrimination here is
                        # noise-like; feed the noise |conf| baseline used by
                        # the preamble confidence gate.
                        _conf_bg += 0.05 * (abs(c) - _conf_bg)
                        _weak_run += 1
                        # End of transmission: drop the cadence anchor.  The
                        # lattice describes ONE continuous transmission, and a
                        # pause of any length puts the next one at an arbitrary
                        # offset that the guard would otherwise reject.
                        #
                        # Discounting the measured silence instead -- keeping
                        # the lattice alive across the gap -- was tried and
                        # REVERTED: "is the transmitter off" cannot be measured
                        # reliably enough from weak symbols.  A fade inside a
                        # frame on a mic capture and impulse clicks between
                        # frames both banked as silence, and the airtime suite
                        # measured the cost at 20% delivery against a 100%
                        # budget in the click cell.  Dropping the anchor is
                        # cruder and much more robust; the hole it leaves is
                        # bounded and covered below by _post_gap.
                        if (_last_pre_sym is not None
                                and _weak_run >= cad_reset_weak):
                            print(f"MF_CADENCE_DROP sym={sym} after "
                                  f"{_weak_run} weak symbols — transmission "
                                  f"ended, lattice dropped", flush=True)
                            _last_pre_sym = None
                            _post_gap = True
                    else:
                        _weak_run = 0

                    # ── S/N + mic level (per symbol; UI 'signal' filter) ──────
                    # No AGC on this path, so report the raw window RMS as the
                    # mic level alongside the matched-filter SNR.  Long symbols
                    # mean this is at most a couple of lines/second.
                    _win_rms = (float(np.sqrt(np.mean(np.square(win, dtype=SIG),
                                                     dtype=np.float64)))
                                if len(win) else 0.0)
                    print(f"MIC_SN sym={sym} bit={bit} dir={'UP' if bit else 'DN'} "
                          f"snr={snr_db:.1f}dB level={_win_rms:.4f} "
                          f"peak={peak_rms:.4f} conf={c:+.2f} "
                          f"{'STRONG' if strong else 'weak'}", flush=True)

                    if eos and not strong:
                        weak_after_eos += 1
                        if weak_after_eos >= 2:
                            break
                    else:
                        weak_after_eos = 0

                    # ── [B] per-symbol telemetry ─────────────────────────────
                    # te = MF timing residual (found vs predicted); nr_dev /
                    # rate_est = the historic-fit grid drift off nominal sps_eff
                    # (the MF analogue of the PLL's n_read deviation).
                    _drift = grid_sps - sps_eff
                    _send({"t": "s", "b": bit, "c": round(float(c), 3),
                           "te": int(found_abs - pred_abs),
                           "nr_dev": int(round(_drift)),
                           "rate_est": round(_drift, 2),
                           "sc": sym, "sync": (state == "COLLECT"),
                           # wt = send wall clock; lets the spectrograph measure
                           # telemetry latency and flag overlay-sync drift.
                           "wt": round(time.time(), 3),
                           # at = the symbol's AUDIO wall-clock (when its chirp
                           # was captured), NOT the send time.  On the MF path a
                           # packet is emitted only after reading the chirp + the
                           # track_search read-ahead + running the correlation, so
                           # wt lags the chirp by a fixed ~(total_abs-found_abs)/fs
                           # (~0.15 sym).  The spectrograph back-dates overlay
                           # placement by `at`, so the drawn chirp/decode tag sits
                           # ON the heard chirp instead of a constant offset to its
                           # right.  Same wall clock (127.0.0.1 telemetry).
                           "at": round(time.time() - (total_abs - found_abs) / fs, 3),
                           "m_ring": int(buf.nbytes),
                           "m_bs": int(len(payload_b) * 8),
                           "m_misc": int(len(hist) * 16)})

                    # Historic grid fit: only trust strong, well-placed peaks.
                    _lv_seen += 1
                    if strong and abs(found_abs - pred_abs) <= track_search:
                        hist.append((sym, found_abs))
                        if len(hist) >= 6 and (sym % refit_every == 0):
                            ks = np.array([h[0] for h in hist], dtype=np.float64)
                            ps = np.array([h[1] for h in hist], dtype=np.float64)
                            A = np.vstack([np.ones_like(ks), ks]).T
                            coef, *_ = np.linalg.lstsq(A, ps, rcond=None)
                            resid = ps - (coef[0] + coef[1] * ks)
                            keepm = np.abs(resid) <= max(track_search * 0.5,
                                                         3 * np.std(resid) + 1)
                            if keepm.sum() >= 4:
                                coef, *_ = np.linalg.lstsq(A[keepm], ps[keepm],
                                                           rcond=None)
                            grid_sps = float(np.clip(coef[1], sps_eff * 0.98,
                                                     sps_eff * 1.02))
                            grid_t0  = float(coef[0])

                    # ── Lock-verify watchdog (see the config block above) ────
                    # Evaluated only in SEARCH (never mid-frame).  Every
                    # lock_verify_syms symbols, find the buffer's GLOBAL MF peak
                    # and compare its grid phase to the predictions.  Two
                    # consecutive checks agreeing on a large off-grid delta ⇒
                    # re-phase the grid onto the real train, clear the bit
                    # history (bits from a bad grid must not feed the preamble
                    # matcher), and re-decode this symbol (continue without
                    # sym += 1).
                    if (state == "SEARCH" and dur >= lock_verify_min_dur
                            and _lv_seen >= lock_verify_syms):
                        _lv_seen = 0
                        _lv_delta = None
                        if len(buf) >= 2 * sps:
                            _rp_u, _rp_d = corr.correlate(buf)
                            _rp = np.maximum(_rp_u[:len(buf)], _rp_d[:len(buf)])
                            if _rp.size:
                                _rp_pk = int(np.argmax(_rp))
                                _rp_mx = float(_rp[_rp_pk])
                                _rp_m  = np.ones(len(_rp), dtype=bool)
                                _rp_m[max(0, _rp_pk - acq_guard):_rp_pk + acq_guard] = False
                                _rp_n  = (float(np.median(_rp[_rp_m])) if _rp_m.any()
                                          else float(np.median(_rp))) + 1e-9
                                if _rp_mx / _rp_n >= max(acq_ratio, relock_min_ratio):
                                    _lv_delta = (buf_start + _rp_pk - pred_abs) % sps_eff
                                    if _lv_delta > sps_eff / 2:
                                        _lv_delta -= sps_eff
                        if _lv_delta is None or abs(_lv_delta) <= track_search:
                            _lv_pending = None          # healthy grid (or no signal)
                        elif (_lv_pending is not None and
                              abs(_lv_delta - _lv_pending) <= lock_verify_tol * sps_eff):
                            # Confirmed on two consecutive windows: re-phase.
                            grid_t0 += _lv_delta
                            grid_sps = float(sps_eff)
                            hist.clear()
                            stream_bits = []; stream_confs = []
                            _conf_bg = 0.25
                            _lv_pending = None
                            # The grid just moved by a non-integer number of
                            # symbols, so the cadence lattice measured in `sym`
                            # units is no longer trustworthy.  Drop the guard
                            # until the next accepted preamble re-anchors it.
                            _last_pre_sym = None
                            diag["relocks"] = diag.get("relocks", 0) + 1
                            print(f"MF_RELOCK sym={sym} "
                                  f"delta={int(_lv_delta)}smp "
                                  f"({_lv_delta / sps_eff:+.2f}sym) "
                                  f"new_grid_t0={grid_t0:.0f}", flush=True)
                            continue    # re-decode this sym on the new grid
                        else:
                            _lv_pending = _lv_delta     # first sighting — confirm next window
                            print(f"MF_LOCK_SUSPECT sym={sym} "
                                  f"delta={int(_lv_delta)}smp "
                                  f"({_lv_delta / sps_eff:+.2f}sym) — "
                                  f"awaiting confirmation", flush=True)

                    stream_bits.append(bit); stream_confs.append(c)
                    _strong_hist.append(bool(strong))

                    if state == "SEARCH":
                        # ── Wake-word resume (see wake_enabled comment block) ──
                        # Fire only at a silence→signal edge: the W word symbols
                        # are strong AND the symbol just before them was weak.
                        # That edge is what makes a 2-symbol marker safe — it
                        # can't match mid-payload (no weak-before) nor in the
                        # silence itself (weak symbols).  Payload starts on the
                        # symbol AFTER the word; the strong word symbols also
                        # re-tighten the grid via the normal hist path above.
                        # A wake word can only legitimately appear AFTER the
                        # first frame: in wake mode the encoder sends the full
                        # training + Barker run-in before frame 0 and the short
                        # wake word only before RESUMED frames.  Requiring a
                        # prior Barker lock closes a false-wake hole that is
                        # otherwise wide open — the training sequence is
                        # alternating [1,0,1,0,...], so it CONTAINS the default
                        # [1,0] wake word, and the silence→training transition
                        # supplies exactly the weak→strong edge the guard below
                        # looks for.  Observed on the fw fixture: a wake fired at
                        # sym=4 (mid-training, 59 dB SNR), consumed frame 0's
                        # window and CRC-failed it, while the real preamble at
                        # sym=21 never got a chance.  It stayed hidden only
                        # because the default mf_seg_trim happened to classify
                        # one training symbol as weak; any change to the
                        # strong/weak boundary re-exposes it.
                        if (wake_enabled and state == "SEARCH" and _barker_seen
                                and len(stream_bits) >= W + 1
                                and len(_strong_hist) >= W + 1):
                            _sh = list(_strong_hist)
                            if (stream_bits[-W:] == wake_word
                                    and all(_sh[-W:])
                                    and not _sh[-(W + 1)]):
                                print(f"MF_WAKE sym={sym} word={wake_word} "
                                      f"snr={snr_db:.1f}dB grid_t0={grid_t0:.0f} "
                                      f"grid_sps={grid_sps:.2f} — resumed frame",
                                      flush=True)
                                if not diag["sync_acquired"]:
                                    diag["sync_acquired"] = True
                                    diag["sync_at_symbol"] = sym
                                diag["wakes"] = diag.get("wakes", 0) + 1
                                _send({"t": "sync", "sc": 1.0,
                                       "f0": f_min, "f1": f_max, "sd": dur,
                                       "sps": sps, "sps_eff": sps_eff,
                                       "fr": frame_symbols})
                                state = "COLLECT"; payload_b = []; payload_c = []
                        # Barker preamble search — only if the wake path above
                        # did NOT just transition us to COLLECT (same iteration).
                        #
                        # In wake mode it also stops entirely once the first
                        # Barker has locked.  The encoder sends training+Barker
                        # before frame 0 and ONLY a wake word before every frame
                        # after it, so from that point on a Barker match is
                        # false by construction -- there is no Barker on the
                        # wire to find.  Leaving the matcher running just gave
                        # it P symbols of payload to accidentally match, and it
                        # did: on a 6-frame wake encode at 18 dB it stole frames
                        # 1 and 2 (one boundary match at 13.6 dB, one inside
                        # frame 2's own payload at 58 dB), while every frame
                        # after that -- where the wake path finally got its turn
                        # -- decoded cleanly.  Neither the strength gate nor the
                        # cadence lattice can catch those: the first straddles
                        # real chirps, and the second is stood down in wake mode
                        # because the suspend gap makes the cadence variable.
                        if (state == "SEARCH" and len(stream_bits) >= P
                                and not (wake_enabled and _barker_seen)):
                            lb = stream_bits[-P:]; lc = stream_confs[-P:]
                            m = sum(1 for i in range(P) if lb[i] == preamble[i]) / P
                            cstr = float(np.mean([abs(x) for x in lc]))
                            # Noise-referenced confidence gate (see the
                            # pre_cstr_floor comment block above): scales with
                            # the running noise baseline instead of the old
                            # fixed 0.4, which rejected low-SNR frames whose
                            # bits were all correct.
                            cstr_gate = max(pre_cstr_floor,
                                            min(0.40, pre_cstr_k * _conf_bg))
                            # ── Preamble strength gate ───────────────────────
                            # A preamble is thirteen CHIRPS.  Thirteen noise
                            # windows whose hard bits happen to spell Barker are
                            # not a preamble, however cleanly they spell it --
                            # and they do spell it: idle listening measured 4-12
                            # false Barker locks an hour, each one burning a
                            # whole frame window in COLLECT reading noise.
                            #
                            # Neither existing gate can see this.  `m` is a bit
                            # match, and noise bits are bits.  `cstr` is
                            # |(u-d)/(u+d)|, normalised, so a noise window can
                            # and does read 0.91 next to a real preamble's 1.00.
                            # The per-symbol SNR is what separates them -- 15 dB
                            # against 58 on the run that exposed this -- and it
                            # only became usable once `strong` was referenced to
                            # the filter's own noise floor rather than to a
                            # constant underneath it.
                            #
                            # A FRACTION, not all P: bark_thr already tolerates
                            # two wrong bits so the gate must tolerate at least
                            # as much obscuration, and a real preamble arriving
                            # under an impulse burst can lose a symbol or two to
                            # the blanker without being any less real.
                            # Alternation of the matched window, the same
                            # statistic the PLL path uses to tell a real
                            # preamble from a sidelobe off the alternating
                            # training run-in (training ~1.0, Barker-13 ~0.58).
                            _pre_alt = None
                            if len(lc) >= 2:
                                _sgn = [x >= 0 for x in lc]
                                _pre_alt = (sum(1 for a, b in zip(_sgn, _sgn[1:])
                                                if a != b) / (len(_sgn) - 1))
                            _pre_snr = snr_log[-P:]
                            _pre_frac = (sum(1 for s in _pre_snr
                                             if s >= acq_min_snr) / P
                                         if len(_pre_snr) >= P else 1.0)
                            _pre_strong_ok = _pre_frac >= pre_strong_frac
                            # ── Frame-cadence guard ──────────────────────────
                            # bark_thr is deliberately loose (0.80 = 11/13) so a
                            # cold-start frame whose preamble lands on a still-
                            # settling grid can still lock.  But the match is
                            # computed over the trailing P symbols on EVERY
                            # symbol, and arbitrary PAYLOAD data matches
                            # Barker-13 in >=11/13 positions with probability
                            # 1.12% per window — ~3.75 expected false matches in
                            # a 334-symbol transmission.  The search fires on the
                            # FIRST window past the gate, so a false match a few
                            # symbols ahead of a real preamble steals the lock,
                            # the payload is read from the wrong offset, and that
                            # frame dies on CRC.  Observed on a CLEAN encoder
                            # file at 58.8 dB: locks at sym 19/84/137/214/279
                            # against a true cadence of 65 — 137 should have been
                            # 149, and frame 2 was lost every single run.
                            #
                            # (The older note here argued Barker-13's 1/13
                            # off-alignment autocorrelation bounded this.  That
                            # bound is for Barker vs SHIFTED BARKER; it says
                            # nothing about Barker vs payload data, which is what
                            # actually matched.)
                            #
                            # Frames are emitted on an exact, known cadence and
                            # the grid freewheels, so once ONE lock exists every
                            # later preamble sits an integer number of frame
                            # periods away.  Accept only locks on that lattice
                            # (within +/-tol symbols); a false match a few
                            # symbols off-cadence is rejected and the real
                            # preamble still gets its chance.  Using the lattice
                            # rather than only the next slot means a genuinely
                            # missed frame doesn't desynchronise the guard.
                            #
                            # Stands down when: no lock yet (nothing to predict
                            # from), wake mode (subsequent frames carry a wake
                            # word, not a Barker preamble, and the suspend gap
                            # makes the cadence variable), or more than
                            # cad_max_miss periods have passed with no lock (the
                            # grid is presumably lost — better to re-acquire
                            # anywhere than to reject forever).
                            _cad_ok = True
                            _cad_why = ""
                            if (pre_cadence_on and not wake_enabled
                                    and _last_pre_sym is not None
                                    and _frame_period > 0):
                                _cd = sym - _last_pre_sym
                                _ck = int(round(_cd / float(_frame_period)))
                                _coff = _cd - _ck * _frame_period
                                if _ck < 1:
                                    _cad_ok, _cad_why = False, f"early k={_ck}"
                                elif abs(_coff) <= pre_cadence_tol:
                                    _cad_ok = True
                                elif _ck > pre_cadence_max_miss:
                                    _cad_ok, _cad_why = True, "stale-anchor"
                                else:
                                    _cad_ok = False
                                    _cad_why = f"off_cadence k={_ck} off={_coff:+d}"
                            # First preamble after a pause: the cadence
                            # lattice was dropped for it, so nothing else can
                            # tell a real resumed preamble from a Barker
                            # sidelobe landing in the next frame's TRAINING
                            # run-in.  Require a nearly clean match there --
                            # measured, every genuine post-gap preamble read
                            # 13/13 while the in-training sidelobe that cost a
                            # frame at training_symbols=12 read 11/13.  Applies
                            # ONLY after a gap: cold start and contiguous frames
                            # keep the loose bark_thr that lets a settling grid
                            # lock at 11/13.
                            _post_gap_ok = (not _post_gap
                                            or m >= post_gap_thr)
                            if (m >= bark_thr and cstr >= cstr_gate and _cad_ok
                                    and _pre_strong_ok and not _post_gap_ok):
                                print(f"MF_POST_GAP_REJECT sym={sym} "
                                      f"match={m:.2f} < {post_gap_thr:.2f} "
                                      f"(first preamble after a pause)",
                                      flush=True)
                                diag["post_gap_rejects"] = (
                                    diag.get("post_gap_rejects", 0) + 1)
                            if (m >= bark_thr and cstr >= cstr_gate
                                    and _cad_ok and not _pre_strong_ok):
                                print(f"MF_PRE_WEAK_REJECT sym={sym} "
                                      f"match={m:.2f} cstr={cstr:.2f} "
                                      f"strong={_pre_frac:.2f} "
                                      f"(need {pre_strong_frac:.2f}) "
                                      f"snr={snr_db:.1f}dB — Barker on noise",
                                      flush=True)
                                diag["weak_rejects"] = (
                                    diag.get("weak_rejects", 0) + 1)
                            if (m >= bark_thr and cstr >= cstr_gate
                                    and _pre_strong_ok and not _cad_ok):
                                print(f"MF_PRE_CADENCE_REJECT sym={sym} "
                                      f"match={m:.2f} cstr={cstr:.2f} "
                                      f"last_pre={_last_pre_sym} "
                                      f"period={_frame_period} {_cad_why}",
                                      flush=True)
                                diag["cadence_rejects"] = (
                                    diag.get("cadence_rejects", 0) + 1)
                            if (m >= bark_thr and cstr >= cstr_gate
                                    and _cad_ok and _pre_strong_ok
                                    and _post_gap_ok):
                                _last_pre_sym = sym
                                _post_gap = False
                                _barker_seen  = True
                                if _cad_why == "stale-anchor":
                                    print(f"MF_PRE_CADENCE_RESET sym={sym} — "
                                          f"no lock for >{pre_cadence_max_miss} "
                                          f"periods; re-anchoring", flush=True)
                                if not diag["sync_acquired"]:
                                    diag["sync_acquired"] = True
                                    diag["sync_at_symbol"] = sym
                                    diag["sync_score"] = round(m, 3)
                                print(f"MF_PREAMBLE sym={sym} match={m:.2f} "
                                      f"cstr={cstr:.2f} (gate={cstr_gate:.2f}) "
                                      f"alt={_pre_alt if _pre_alt is None else round(_pre_alt, 3)} "
                                      f"grid_sps={grid_sps:.1f} "
                                      f"snr={snr_db:.1f}dB", flush=True)
                                # ── [C] sync telemetry (per-frame preamble) ──
                                _send({"t": "sync", "sc": round(m, 3),
                                       "f0": f_min, "f1": f_max, "sd": dur,
                                       "sps": sps, "sps_eff": sps_eff,
                                       "fr": frame_symbols})
                                state = "COLLECT"; payload_b = []; payload_c = []
                            elif (m >= bark_thr and _cad_ok and _pre_strong_ok
                                  and _post_gap_ok):
                                # Bit pattern matches but confidence sits below
                                # the noise-referenced gate — log it so a
                                # marginal run shows WHY sync was withheld.
                                #
                                # Gated on _cad_ok as well, because a cadence
                                # rejection has ALREADY logged its own line
                                # above and this one would follow it stating a
                                # comparison that is not true: the branch fires
                                # whenever the accept failed for any reason, so
                                # an off-cadence preamble with perfect
                                # confidence printed "cstr=1.00 < gate=0.33".
                                # Two rejects for one event, the second naming
                                # the wrong gate, is how a field log sends
                                # someone after the confidence threshold when
                                # the cadence lattice is what refused them.
                                print(f"MF_PRE_REJECT sym={sym} match={m:.2f} "
                                      f"cstr={cstr:.2f} < gate={cstr_gate:.2f} "
                                      f"(conf_bg={_conf_bg:.2f})", flush=True)
                    elif state == "COLLECT":
                        payload_b.append(bit); payload_c.append(c)
                        if len(payload_b) >= frame_symbols:
                            ok, seq, dat, rs_fail = _pipeline(payload_b, payload_c)
                            fsnr = (float(np.mean(snr_log[-frame_symbols:]))
                                    if snr_log else 0.0)
                            if ok and seq not in _decoded_seqs:
                                # The frame CRC has already validated integrity.
                                # Payloads may be arbitrary BINARY (modem-layer
                                # packets), so we do NOT gate on UTF-8 — that
                                # would drop valid binary frames.  Raw bytes are
                                # kept as bytes_hex; text is a best-effort view
                                # for logs / plain-text messages.
                                txt = dat.decode("utf-8", errors="replace")
                                # at_sample: absolute capture-stream sample index
                                # of THIS frame (its current symbol), from the
                                # decoder's own grid model grid_t0 + sym*grid_sps
                                # (the same pred_abs used for tracking).  grid_t0
                                # alone is the grid ORIGIN — constant across
                                # frames because the grid freewheels — so it must
                                # be advanced by the running symbol index to get a
                                # per-frame arrival time.  The SAME detector
                                # timestamps every frame, so the difference
                                # between two frames' at_sample values is a
                                # round-trip / turnaround in samples — the basis
                                # for two-way ranging (see css_modem.py).
                                _at_sample = int(round(grid_t0 + sym * grid_sps))
                                _frame_rec = {"seq": seq, "text": txt,
                                              "bytes_hex": dat.hex(),
                                              "at_sample": _at_sample,
                                              "grid_sps": round(float(grid_sps), 3),
                                              "snr_db": round(fsnr, 1),
                                              "source": "mf"}
                                decoded_frames.append(_frame_rec)
                                _decoded_seqs.add(seq); frames_decoded += 1
                                if _on_frame is not None:
                                    try:
                                        _on_frame(dict(_frame_rec))
                                    except Exception as _ofe:
                                        print(f"on_frame hook error: {_ofe}",
                                              flush=True)
                                if "sync_at_sample" not in diag:
                                    diag["sync_at_sample"] = _at_sample
                                # grid_t0/grid_sps/hist persist across the frame
                                # boundary (the state machine only flips
                                # SEARCH<->COLLECT; nothing resets the grid):
                                # timing established at acquisition is CARRIED
                                # into the next frame's chunk and can freewheel
                                # over weak/silent stretches, where hist simply
                                # stops updating and the grid coasts on its last
                                # fit.  Logged here so runs show the handoff.
                                print(f"MF_FRAME seq={seq} snr={fsnr:.1f}dB "
                                      f"text={txt} at_sample={_at_sample} "
                                      f"grid_t0={grid_t0:.0f} "
                                      f"grid_sps={grid_sps:.2f}", flush=True)
                                # ── [D] frame telemetry ──────────────────
                                _send({"t": "frame", "seq": seq,
                                       "n": frames_decoded, "sc": sym})
                            elif not (ok and seq in _decoded_seqs):
                                frames_errored += 1
                                print(f"MF_FRAME_ERROR "
                                      f"reason={'rs' if rs_fail else 'crc'} "
                                      f"snr={fsnr:.1f}dB", flush=True)
                            state = "SEARCH"
                            if expected_frames and frames_decoded >= expected_frames:
                                break
                    sym += 1
            else:
                diag["silence"] = peak_rms < 1e-4

    except Exception as exc:
        import traceback
        result["failure_reason"] = f"mf_error: {exc}"
        print(f"MF ERROR: {exc}\n{traceback.format_exc()}", flush=True)

    diag["symbols_received"] = sym
    diag["peak_rms"]    = round(peak_rms, 6)
    diag["avg_snr_db"]  = round(float(np.mean(snr_log)), 1) if snr_log else 0.0

    if diag.get("silence"):
        status, reason = "fail", f"no_audio (peak_rms={diag['peak_rms']:.6f})"
    elif not diag["sync_acquired"]:
        status, reason = "fail", "no_sync"
    elif frames_decoded == 0 and frames_errored > 0:
        status, reason = "fail", f"crc_fail ({frames_errored} frames)"
    elif frames_decoded == 0:
        status, reason = "fail", "sync acquired but no frames"
    elif expected_frames and frames_decoded < expected_frames:
        status, reason = "partial", f"{frames_decoded}/{expected_frames} frames"
    else:
        status, reason = "ok", None

    result.update({
        "status": status, "failure_reason": reason,
        "frames_decoded": frames_decoded, "frames_errored": frames_errored,
        "diagnostics": diag, "frames": decoded_frames,
    })
    if decoded_frames:
        result["decoded_text"], _raw_msg = assemble_message(
            decoded_frames, chunk_size, message_len)
        result["decoded_hex"] = _raw_msg.hex()

    if wav_out and _capture:
        import wave as _wave_mod
        try:
            _audio = np.concatenate(_capture).astype(np.float32)
            _pcm   = (np.clip(_audio, -1.0, 1.0) * 32767).astype(np.int16)
            with _wave_mod.open(str(wav_out), "wb") as _wf:
                _wf.setnchannels(1); _wf.setsampwidth(2); _wf.setframerate(fs)
                _wf.writeframes(_pcm.tobytes())
            result["wav_capture"] = str(wav_out)
            print(f"WAV capture -> {wav_out}  ({len(_audio)/fs:.1f}s @ {fs}Hz)",
                  flush=True)
        except Exception as _e:
            print(f"WAV capture failed: {_e}", flush=True)

    # ── Compute-cost summary (min-hardware sizing) ───────────────────────────
    _prof_cpu   = time.process_time()  - _prof_cpu0
    _prof_wall  = time.perf_counter()  - _prof_wall0
    _audio_s    = total_abs / float(fs) if fs else 0.0
    _rtf        = (_prof_cpu / _audio_s) if _audio_s > 0 else 0.0
    _per_sym_ms = (_prof_cpu / sym * 1e3) if sym else 0.0
    diag["cpu_s"]        = round(_prof_cpu, 3)
    diag["audio_s"]      = round(_audio_s, 2)
    diag["rtf"]          = round(_rtf, 4)
    diag["cpu_ms_per_sym"] = round(_per_sym_ms, 2)
    result["diagnostics"] = diag
    _headroom = (1.0 / _rtf) if _rtf > 1e-9 else float("inf")
    print(f"PROFILE cpu={_prof_cpu:.2f}s wall={_prof_wall:.2f}s "
          f"audio={_audio_s:.1f}s syms={sym} "
          f"rtf={_rtf:.3f} (headroom {_headroom:.1f}x) "
          f"cpu/sym={_per_sym_ms:.1f}ms "
          f"sps={sps} nfft={_mf_symbol.nfft}", flush=True)

    print(f"DONE_MF status={status} frames={frames_decoded} "
          f"avg_snr={diag['avg_snr_db']}dB", flush=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


# ---------------------------------------------------------------------------
# Main decode loop
# ---------------------------------------------------------------------------

def decode(cfg, stop_event=None):
    import sounddevice as sd

    info_path      = Path(cfg.get("info",               "css_info.json"))
    result_path    = Path(cfg.get("result_out",          "css_decode_result.json"))
    timeout_sec    = float(cfg.get("timeout_sec",        600))
    silence_thr    = float(cfg.get("silence_threshold",  0.002))
    sync_threshold = float(cfg.get("sync_threshold",     0.65))
    alpha          = float(cfg.get("timing_loop_gain",   0.10))
    alpha_acq      = float(cfg.get("timing_acq_gain",    0.25))
    nread_revert   = float(cfg.get("nread_revert_alpha", 0.05))
    search_frac    = float(cfg.get("timing_search_frac", 0.25))
    # rate_est_smp (see init section below) is captured once from just 5
    # samples at the acq->settled transition -- a small, potentially noisy
    # window.  rate_refine_alpha lets it keep slowly self-correcting
    # throughout settled tracking: if the one-shot estimate is off by ε, the
    # PLL's int_corr will show a persistent bias of ε (it has to keep
    # compensating for the same error every symbol) rather than averaging to
    # zero.  Feeding a small fraction of each accepted symbol's int_corr
    # into rate_est_smp lets that systematic bias be slowly absorbed into the
    # baseline instead of being re-fought from scratch every single symbol —
    # the same principle as a PI controller's integral term tracking out a
    # steady-state disturbance.  Small and slow by design (default 0.02): it
    # should only correct genuine systematic bias accumulated over dozens of
    # symbols, not chase single-symbol measurement noise.
    rate_refine_alpha = float(cfg.get("rate_refine_alpha", 0.02))
    # Inlier fraction the retro drift fit must reach before it is adopted as
    # the rate baseline for the NEXT frame.  Default 1.01 = never, which
    # disables the adoption; 0.80 restores the pre-2026-09-03 behaviour.  See
    # the RETRO_SYNC adoption block for the measurements behind that default.
    retro_rate_adopt_inlier = float(cfg.get("retro_rate_adopt_inlier", 1.01))
    # ── winner_pk strength gate (settled-mode timing) ────────────────────────
    # When the payload contains a long run of identical symbols (e.g. a short
    # message zero-padded to a large chunk_size produces dozens of consecutive
    # identical down-chirps), only a sparse subset of symbols yield a strong,
    # unambiguous correlation peak — the rest are weak/noise-floor, and their
    # found_peak is a random noise argmax anywhere in the search window.
    # Updating timing from those noise positions destroys the lock the strong
    # symbols established.  The fix mirrors what a human does reading the
    # spectrograph: only re-anchor timing on the clearly-visible chirps, and
    # coast through the ambiguous stretch on the established rate model.
    #
    # A symbol's winner_pk must exceed wpk_track_k × (tracked noise-floor EMA)
    # to update peak_est/_terr_buf in settled mode.  Below that, the symbol's
    # bit is still decoded (from whatever the MF found) but its POSITION is
    # not trusted for timing — n_read continues on rate_est_smp alone.
    # Adaptive (ratio to a slow EMA of winner_pk) rather than a fixed
    # threshold so it self-calibrates to each recording's signal level.
    wpk_track_enable = bool(cfg.get("wpk_track_gate_enable", True))
    wpk_track_k      = float(cfg.get("wpk_track_k", 3.0))
    wpk_track_alpha  = float(cfg.get("wpk_track_ema_alpha", 0.05))
    agc_target     = float(cfg.get("agc_target_peak",    0.4))
    agc_attack     = float(cfg.get("agc_attack",         0.8))
    agc_release    = float(cfg.get("agc_release",        0.02))
    agc_max        = float(cfg.get("agc_max_gain",       200.0))
    snr_log_ivl    = float(cfg.get("snr_log_interval",   10))
    resync_after   = int(  cfg.get("resync_after",        3))
    mf_window      = str(  cfg.get("mf_window",          "none"))   # "none" or "hann"
    device         = cfg.get("device")
    if device is not None:
        try:
            device = int(device)
        except (TypeError, ValueError):
            device = None

    wav_out   = cfg.get("wav_out")
    _capture  = [] if wav_out else None
    # Streaming hook (see decode_longsym_mf): called per decoded frame as it
    # arrives; a callback exception must never crash decoding.
    _on_frame = cfg.get("on_frame", None)

    # Cooperative stop sentinel (see decode_longsym_mf for rationale).  Polled in
    # the main read loop so the wrapper can stop us gracefully on Windows, where
    # proc.terminate() (TerminateProcess) cannot deliver a catchable SIGTERM.
    stop_file = cfg.get("stop_file")

    result = {
        "status": "error", "failure_reason": None,
        "frames_decoded": 0, "frames_errored": 0,
        "decoded_text": "", "diagnostics": {}, "frames": [],
    }

    if not info_path.exists():
        print(f"ERROR: info file not found: {info_path}", flush=True)
        result["failure_reason"] = "no_info_file"
        result_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return result

    conf          = json.loads(info_path.read_text())
    fs            = int(  cfg.get("fs",         conf["fs"]))
    dur           = float(cfg.get("symbol_dur", conf["symbol_dur"]))
    # Dispatch to the per-symbol matched-filter decoder for everything except
    # very short symbols.  Originally MF was long-symbol-only (>0.5 s), but the
    # hardened MF path (on-grid decide window, segmented burst-robust statistic,
    # lock-verify watchdog) now beats the streaming PLL on real recordings at
    # short durations too — measured: a 0.5 s acoustic run decoded 2/2 "hi" via
    # MF vs 1/2 via the PLL (which only synced at sym 129, score 0.47), and a
    # 0.3 s run at 9.5 dB decoded fully via MF while the PLL CRC-failed.  The
    # PLL path remains for dur <= 0.25 s (e.g. the 0.25 s fast loopback profile,
    # where it is well validated) and via mf_longsym=False.
    if (dur > float(cfg.get("mf_longsym_min_dur", 0.25))
            and bool(cfg.get("mf_longsym", True))):
        return decode_longsym_mf(cfg, stop_event=stop_event)
    f_min         = float(cfg.get("f_min",      conf["f_min"]))
    f_max         = float(cfg.get("f_max",      conf["f_max"]))
    info_preamble = conf.get("preamble")
    cfg_preamble  = cfg.get("preamble")

    if cfg_preamble is not None:
        full_preamble = list(cfg_preamble)
        if info_preamble and list(cfg_preamble) != list(info_preamble):
            print(f"WARNING: cfg preamble ({len(cfg_preamble)}-sym) overrides "
                  f"info preamble ({len(info_preamble)}-sym) — confirm this is intentional",
                  flush=True)
    elif info_preamble is not None:
        full_preamble = list(info_preamble)
    else:
        full_preamble = list(BARKER_13)
        print("WARNING: no preamble in css_info.json or cfg — defaulting to Barker-13",
              flush=True)

    sync_word = full_preamble

    # ── Signal-shape parameters — conf (css_info.json) is authoritative ────────
    # These describe the waveform the encoder actually produced.  They MUST
    # come from conf (css_info.json), NOT from cfg / WebUI PARAMS.
    # Architectural rule:  WebUI PARAMS → encoder → css_info.json → decoder.
    # If a field is absent (audio predates the parameter) the default of
    # 0 / False / None matches the old-encoder behaviour for that audio.
    chunk_size      = int(  cfg.get("chunk_size", conf["chunk_size"]))  # operational override OK
    crc_bytes       = int(  conf.get("crc_bytes",       CRC_BYTES))
    rs_t            = int(  conf.get("rs_t",            0))
    rs_parity       = rs_t * 2
    interleave_rows = int(  conf.get("interleave_rows", 0))

    rs_codec = _make_rs_codec(rs_t)
    if rs_t > 0 and rs_codec is None:
        rs_parity = 0

    sps      = int(fs * dur)
    buf_size = sps * 3

    # Inter-symbol gap written by the encoder into css_info.json.
    # The encoder emits (sps + gap_samps) audio samples per symbol: sps of
    # chirp followed by gap_samps of silence.  The decoder must read
    # sps_eff = sps + gap_samps samples per symbol to stay in phase; without
    # this the ring drifts by gap_samps per symbol and sync collapses.
    # The PLL still tracks residual clock-rate differences on top of sps_eff.
    #
    # Reading order: conf (css_info.json written by the encoder) is
    # authoritative — it describes the waveform that was actually produced.
    # cfg (WebUI / direct params) is used as a fallback when conf does not
    # carry the field (css_info.json predates this parameter, or the decoder
    # started before the encoder finished writing the file).
    #
    # Unlike fs/dur/f_min/f_max — which let the operator override conf —
    # symbol_gap_frac must NOT be silently overridden from cfg when conf
    # already has a value: using the wrong gap causes the MF peak to fall
    # outside post_acq_search_smp and the decoder reads noise.  A mismatch
    # is always flagged so the race condition is visible in the logs.
    _gap_frac_cfg  = float(cfg.get("symbol_gap_frac", 0.0))
    _gap_frac_conf = conf.get("symbol_gap_frac")          # None if absent
    if _gap_frac_conf is not None:
        _gap_frac = float(_gap_frac_conf)
        if _gap_frac_cfg != 0.0 and abs(_gap_frac - _gap_frac_cfg) > 1e-6:
            print(
                f"WARNING: symbol_gap_frac mismatch — "
                f"css_info.json={_gap_frac:.4f}  cfg={_gap_frac_cfg:.4f}. "
                f"Using css_info.json value.  "
                f"If that file is stale, re-run the encoder before the decoder.",
                flush=True)
    else:
        # conf has no entry: fall back to cfg so the WebUI value reaches the
        # decoder without requiring a re-encode just to refresh css_info.json.
        _gap_frac = _gap_frac_cfg
    _gap_frac = float(np.clip(_gap_frac, 0.0, 0.5))   # mirror encoder clamp
    gap_samps = int(sps * _gap_frac)
    sps_eff   = sps + gap_samps   # effective symbol period in audio samples

    min_tim_conf    = float(cfg.get("min_timing_conf",    0.10))
    max_terr_smp    = int(  cfg.get("max_timing_err_smp", sps // 2))
    presync_max_terr_smp = int(cfg.get("presync_max_terr_smp", max(1, int(sps * 0.02))))
    tim_track_alpha = float(cfg.get("timing_track_alpha", 0.050))

    # ── Impulse blanker ──────────────────────────────────────────────────────
    # A chirp matched filter (FFT-domain correlation against an LFM template)
    # is mathematically a pulse-compression filter: a short, high-amplitude,
    # BROADBAND impulse (a knock, a click, ADC clipping) produces a sharp,
    # large peak in the matched-filter output too — the filter has no way to
    # tell "energy that sweeps through the band over 2s" from "energy that
    # hits every band at once for 20ms".  A human glancing at a spectrogram
    # sees the difference instantly (a vertical broadband streak vs. a
    # diagonal narrowband sweep); this time-domain correlator does not.
    # Impulses are typically a tiny fraction of one symbol's duration (tens
    # of ms out of a multi-second chirp), so the standard SDR/HF-receiver
    # mitigation applies directly: detect samples whose instantaneous
    # magnitude is a large multiple of the chunk's ROBUST typical level
    # (median |x|, which a handful of huge outlier samples barely moves)
    # and zero them out before they ever reach AGC or the correlator.
    # Blanking ~1-2% of a symbol costs a small fraction of a dB of matched-
    # filter processing gain on the genuine chirp — far cheaper than letting
    # one impulse hijack found_peak for the whole symbol.
    impulse_blank_enable    = bool(cfg.get("impulse_blank_enable", True))
    impulse_blank_k         = float(cfg.get("impulse_blank_k",     6.0))
    impulse_clip_floor      = float(cfg.get("impulse_clip_floor",  0.98))
    # ── Max-fraction cap ──────────────────────────────────────────────────
    # A real, brief click/knock never needs more than a few percent of a
    # symbol blanked (measured directly from captured impulses: isolated
    # clip bursts were <2% of a symbol).  A SUSTAINED noise event — speech,
    # traffic, room noise lasting hundreds of ms — can push the blank
    # fraction far higher, and at that point blanking stops helping and
    # starts hurting: it doesn't just remove noise, it punches a large,
    # often asymmetric hole straight through genuine chirp content,
    # shifting the matched filter's correlation peak (observed: 20-28% of
    # a symbol blanked in one capture, immediately followed by the timing
    # PLL going unstable for 60+ consecutive symbols).  A human — or this
    # decoder's own ChirpCorrelator._slices sub-symbol fallback — can still
    # extract a chirp that's merely OVERLAID with noise; neither can recover
    # one with a quarter of its samples zeroed out.  Past this cap, skip
    # blanking entirely for the symbol and let the matched filter (with its
    # frequency-selective rejection and partial-chirp slice fallback) work
    # on the unmodified, degraded-but-still-present signal instead.
    impulse_blank_max_frac  = float(cfg.get("impulse_blank_max_frac", 0.08))

    # ── Noise-whitening pre-filter ──────────────────────────────────────────
    # A matched filter (FFT-domain correlation against an LFM template) is
    # information-theoretically optimal ONLY against WHITE noise.  Real
    # acoustic capture environments are rarely white: mechanical/electrical
    # hum (fans, HVAC, mains harmonics) concentrates persistent, narrowband
    # energy at specific frequencies, often with harmonics extending into the
    # 400-3000Hz chirp band itself.  A human glancing at a spectrogram (or
    # Photoshop's magic wand) separates this trivially — a steady horizontal
    # hum-line looks nothing like a diagonal sweep — but a non-whitened
    # matched filter has no equivalent discrimination: a persistent tone
    # correlates against EVERY time-shift the search window tests (since,
    # unlike the chirp, it doesn't go away), so it can generate a spurious
    # correlation peak comparable to or stronger than the genuine, transient
    # chirp, and that peak can land anywhere the tone happens to align with
    # the template's instantaneous frequency — a very plausible mechanism for
    # a found_peak landing a fixed offset away from the true chirp position.
    #
    # Mitigation, and the bug in the first version of this comment: a per-bin
    # background estimate built from ONE FFT over the WHOLE symbol (~sps
    # samples) does NOT isolate persistent tones from the chirp — a single
    # full-chunk FFT smears the chirp's ENTIRE sweep across the WHOLE
    # 400-3000Hz band in that one measurement, so the chirp's own energy
    # looks just as "present in every symbol" as a real steady hum to a
    # cross-symbol EMA of that quantity (confirmed empirically: an earlier
    # version built this way measurably REDUCED winner_pk in testing — it
    # was whitening away part of the chirp along with the hum).
    #
    # The fix mirrors what a human reading a spectrogram actually does:
    # look across MANY TIME SLICES, not one. A steady tone lights up in
    # nearly every column of a spectrogram; the chirp lights up only briefly
    # in each row, since it only dwells at any given frequency for a moment
    # as it sweeps through. So: take a SHORT-TIME FFT (STFT) of each
    # incoming chunk — short enough that the chirp occupies only a small
    # minority of the time-slices at any one bin — and take the MEDIAN
    # magnitude ACROSS those time-slices, per bin.  A median is robust to a
    # small minority of elevated values (same principle as the impulse
    # blanker's median-based robust level, applied here per frequency bin
    # across time instead of per sample across one chunk): the chirp's brief
    # visit to any one bin doesn't move the median, while a persistent tone
    # — elevated in nearly every slice — dominates it directly.
    whiten_enable          = bool(cfg.get("whiten_enable", True))
    whiten_stft_win_s      = float(cfg.get("whiten_stft_win_s", 0.05))
    whiten_bg_alpha        = float(cfg.get("whiten_bg_alpha", 0.10))
    whiten_redesign_every  = int(cfg.get("whiten_redesign_every", 4))
    whiten_taps            = int(cfg.get("whiten_taps", 2047))
    whiten_max_gain_db     = float(cfg.get("whiten_max_gain_db", 18.0))

    # Minimum SNR (dB) before the timing PLL accepts a measurement.
    # This should only reject truly empty buffers (MF output is pure noise,
    # SNR typically negative).  DO NOT raise this above ~5 dB: for an
    # unwindowed linear FM chirp the sidelobes set a ceiling on the MF
    # peak-to-noise-floor ratio of ~13–16 dB regardless of acoustic
    # conditions.  Setting the gate above that ceiling makes it permanently
    # closed in any real environment, freezing the PLL completely.
    # Silence-phase drift is handled by the training sequence + median
    # filter — not by this gate.
    timing_min_snr_db = float(cfg.get("timing_min_snr_db", 3.0))

    # After sync fires, use alpha_acq for this many symbols before dropping to
    # the slow tracking alpha.  Ensures rapid convergence of whatever timing
    # error existed at sync acquisition, so frame symbols are collected with
    # good MF alignment instead of a slowly-decaying offset.
    # At alpha_acq=0.25, tau ≈ 4 symbols → effectively converged in ~12.
    timing_acq_symbols = int(cfg.get("timing_acq_symbols", 20))

    # Median filter window for timing_err before it reaches the PLL integrator.
    # In reverberant / echo environments the MF occasionally selects an echo
    # peak instead of the direct-path peak, producing a large outlier in terr
    # (e.g., terr jumping from -2000 to -8449 in one symbol).  A 3-sample
    # median turns that outlier into a non-event: the middle value of
    # [-2059, -8449, -1530] is -2059, discarding the echo spike entirely.
    # Set to 1 to disable (equivalent to the previous behaviour).
    timing_median_n = max(1, int(cfg.get("timing_median_n", 3)))

    # Maximum per-step contribution to timing_frac, expressed as a fraction
    # of sps.  Prevents a single large terr measurement from driving n_read
    # so far from sps that the audio buffer becomes significantly shorter than
    # frame_symbols × sps.  Since the retro decode uses the buffer for bit
    # decisions, a short buffer means missing end-of-frame symbols and RS
    # erasures that exceed the correction capacity.
    #
    # With max_step_frac=0.10: n_read stays within [0.9 × sps, 1.1 × sps].
    # The timing still converges — it just takes a few more symbols for large
    # initial errors.  The retro decode handles those extra symbols correctly.
    timing_max_step_frac = float(cfg.get("timing_max_step_frac", 0.10))

    base_bits = int(conf.get("base_bits",
                             (1 + chunk_size + crc_bytes + rs_parity) * 8))
    frame_symbols      = int( conf.get("frame_symbols",      base_bits))
    preamble_per_frame = bool(conf.get("preamble_per_frame", False))
    diff_enc           = bool(conf.get("diff_encode",        False))   # conf-authoritative
    conv_code          =      conf.get("conv_code",          None)     # conf-authoritative
    pll_weight = float(cfg.get("pll_weight", 0.30))

    # Erasure threshold in soft-confidence space.
    # For mf_window="none" (unwindowed MF), the conf scale is the same as before
    # so the default 0.25 remains appropriate.
    era_thr = float(cfg.get("erasure_conf_thr", 0.25))

    # Reference chirps -- no edge taper (must match encoder's generate_chirp
    # exactly; see that function's docstring).
    up_ref   = generate_chirp(f_min, f_max, dur, fs)
    down_ref = generate_chirp(f_max, f_min, dur, fs)

    correlator = ChirpCorrelator(up_ref, down_ref, sps,
                                 search_frac=search_frac, buf_size=buf_size,
                                 mf_window=mf_window)
    # verbose_syms: set of symbol indices for which symbol_decision emits
    # detailed MF_DEC / slice logging.  Empty by default; populated from
    # cfg so the web UI or harness can request per-symbol MF traces without
    # recompiling anything.
    _verbose_syms = set(int(s) for s in cfg.get("verbose_syms", []))
    agc        = PeakAGC(target_peak=agc_target, attack=agc_attack,
                         release=agc_release, max_gain=agc_max)

    # syndet lookback: read from config; auto-size if not set.
    # css_listen.py writes "syndet_lookback" into the decode config, but the
    # old code always called SoftPreambleDetector() with no lookback= argument,
    # silently defaulting to n*3 = 39 symbols.
    #
    # Auto-size to max(n*3, frame_symbols + 2*n_preamble + training_symbols):
    # after a mid-stream resync the re-armed syndet must keep the inter-frame
    # gap (training + preamble of the next frame) within its ring.  At 2s/sym
    # with frame_symbols=74 and training=16, that is max(39, 116) = 116.
    # With lookback=39, any re-arm that happens before the inter-frame training
    # starts can only look back 26 symbols past the current push; if the
    # preamble arrived earlier than that lag it is simply missed.
    _n_preamble       = len(sync_word)
    # Fallback only — real runs read training_symbols from the encoder's info
    # file.  When absent, mirror the encoder's duration-scaled default so a
    # standalone decode computes the same _min_preamble_end / lookback.
    if _default_training_symbols is not None:
        _training_fallback = _default_training_symbols(dur)
    else:
        _training_fallback = 16
    _training_symbols = int(conf.get("training_symbols", _training_fallback))
    _auto_lookback    = max(_n_preamble * 3,
                            frame_symbols + 2 * _n_preamble + _training_symbols)
    _syndet_lookback  = int(cfg.get("syndet_lookback", _auto_lookback))
    syndet = SoftPreambleDetector(sync_word, threshold=sync_threshold,
                                  lookback=_syndet_lookback)

    # -- Syndet noise-immunity parameters ------------------------------------
    # SNR gate: symbols with snr_db < syndet_min_snr_db are zeroed before
    # being pushed into the preamble correlator (deep-fade erasure).
    # The weight ramps linearly to 1.0 at syndet_snr_ref_db.
    syndet_min_snr_db = float(cfg.get("syndet_min_snr_db",  0.0))
    syndet_snr_ref_db = float(cfg.get("syndet_snr_ref_db", 12.0))
    # Lag guard: when True, reject acquisitions where the matched preamble
    # window is more than syndet_max_lag symbols in the past. Stale matches
    # arise from chance Barker correlations in the noise history.
    syndet_lag_guard  = bool(cfg.get("syndet_lag_guard", True))
    # Floor the stale-match cutoff at 2*preamble+8 so it stays wide enough even
    # when training_symbols is scaled down for long symbols — otherwise the
    # acceptance band collapses and a valid preamble ages into a STALE_REJECT
    # before it can be taken (the re-arm deadlock).
    # Alternation above this reads as "the match is on the training pattern"
    # (training = 1.0, Barker-13 = 0.58), which is what the early-arrival guard
    # exists to reject.  Well clear of both, so neither a couple of flipped
    # confidences in noise nor a partial-window match moves a decision.
    syndet_alt_thr    = float(cfg.get("syndet_training_alt_frac", 0.85))
    syndet_max_lag    = int(cfg.get("syndet_max_lag",
                                    max(_training_symbols + _n_preamble + 4,
                                        2 * _n_preamble + 8)))
    # Adaptive background threshold: the effective sync threshold is
    # clamp(bg_ema + syndet_margin, syndet_thr_floor, sync_threshold), so it
    # rises above the noise floor in bursty environments but never drops below
    # syndet_thr_floor. bg_ema is a SIGNED Barker correlation of the confidence
    # stream and decays toward (and below) zero in clean conditions; without
    # the floor, eff_thr can collapse to ~0.14 and let a weak ~0.38 spurious
    # correlation false-sync at the wrong timing offset, producing a wrong
    # peak_est, PLL-outlier storm, and a lost frame. The floor sits above the
    # ~0.40 spurious-correlation floor and below a real preamble (~0.55+), so
    # spurious hits are rejected while genuine preambles still acquire.
    syndet_bg_alpha   = float(cfg.get("syndet_bg_alpha", 0.02))
    syndet_margin     = float(cfg.get("syndet_margin",   0.15))
    syndet_thr_floor  = float(cfg.get("syndet_thr_floor", 0.45))

    # ── SIGTERM handler ───────────────────────────────────────────────────────
    # proc.terminate() in css_listen.py sends SIGTERM.  Python does NOT convert
    # SIGTERM to KeyboardInterrupt (only SIGINT does that).  Without a handler
    # the process dies instantly, result_path.write_text() is never reached,
    # and the test framework reads an empty dict → status=None even when frames
    # decoded successfully.  The flag lets the main loop exit cleanly and fall
    # through to normal result-writing.
    import signal as _signal_mod
    _sigterm_flag = [False]
    _prev_sigterm = _signal_mod.getsignal(_signal_mod.SIGTERM)
    def _handle_sigterm(signum, frame_):   # noqa: E306
        _sigterm_flag[0] = True
    _signal_mod.signal(_signal_mod.SIGTERM, _handle_sigterm)

    # ── Long-symbol stability fixes ──────────────────────────────────────────
    #
    # Post-acquisition search range.
    # Pre-sync we need the full wide window to find the initial chirp position.
    # Post-sync, restricting the search to a small neighbourhood around
    # expected_peak prevents room echoes (typically 30-200 ms ≈ 1323-8820 smp
    # at 44100 Hz) from being mistaken for the direct-path chirp peak and
    # causing large timing spikes.
    #
    # Default: 4 % of sps.  For sps=88200 that is ±3528 smp ≈ ±40 ms —
    # below the shortest common room echo delay (~30-50 ms).  Hardware clock
    # drift between transmitter and receiver is typically <100 ppm, i.e.
    # <9 smp/symbol at sps=88200, so the window is ample for genuine drift.
    #
    # Raise post_acq_search_frac (up to search_frac) if your acoustic
    # environment has a very long direct-path delay variance (e.g. outdoor
    # over a large distance with wind).
    post_acq_search_frac = float(cfg.get("post_acq_search_frac", 0.04))
    post_acq_search_smp  = max(16, int(sps * post_acq_search_frac))

    # ── gap_samps vs. post_acq_search_smp: settled-mode search symmetry ──────
    # expected_peak sits gap_samps before the ring buffer's end (see
    # expected_peak below), so the settled-mode search window
    # [expected_peak-post_acq_search_smp, expected_peak+gap_samps] can only
    # read timing_err positive up to +gap_samps before clipping at the ring
    # boundary.  If gap_samps < post_acq_search_smp, that window is
    # ONE-SIDED: a chirp arriving LATER than expected (timing_err > gap_samps)
    # clips to timing_err=gap_samps regardless of how much later it actually
    # is, giving the loop NO gradient to correct that direction -- drift in
    # that direction can run unchecked.  The encoder's default
    # symbol_gap_frac=0.05 gives gap_samps=0.05*sps > post_acq_search_smp,
    # which is symmetric; this only fires for symbol_gap_frac=0 (old configs)
    # or an unusually large post_acq_search_frac.
    if gap_samps < post_acq_search_smp:
        print(f"WARNING: gap_samps={gap_samps} < post_acq_search_smp="
              f"{post_acq_search_smp} -- settled-mode search window is "
              f"ONE-SIDED (can detect chirp arriving EARLY but not LATE). "
              f"Re-encode with symbol_gap_frac >= "
              f"{post_acq_search_frac:.3f} to fix.", flush=True)

    # Auto-scale tracking alpha for long symbol durations.
    # The PLL integrator gain is specified "per symbol".  For short symbols
    # (dur=0.01 s) that is very fast in wall-clock time; for long symbols
    # (dur=1 s) the same alpha creates a loop that is 40× slower to respond
    # to genuine drift but equally fast to amplify noise spikes.
    # Rescaling to keep bandwidth constant in *seconds* (ref=0.25 s) avoids
    # both extremes.  alpha_acq is left unscaled — it is only active for
    # timing_acq_symbols symbols right after sync fires and is intentionally
    # aggressive.
    #
    # Use the EFFECTIVE symbol period (sps_eff/fs = chirp + gap), not the
    # bare chirp duration.  With gap_frac=0.40 the effective period is 40%
    # longer than dur; using dur would leave alpha 1.4× too high, making the
    # PLL measurably noisier as the gap grows.
    _dur_ref  = 0.25
    _eff_dur  = sps_eff / fs   # includes gap: dur * (1 + gap_frac)
    if _eff_dur > _dur_ref:
        alpha = max(0.005, alpha * (_dur_ref / _eff_dur))

    print(f"LISTENING fs={fs} sps={sps} sps_eff={sps_eff} dur={dur}s "
          f"gap={gap_samps}smp({_gap_frac:.3f}) "
          f"f={f_min:.0f}-{f_max:.0f}Hz "
          f"chunk={chunk_size}B crc={crc_bytes}B rs_t={rs_t} "
          f"interleave_rows={interleave_rows} "
          f"mf_window={mf_window!r} "
          f"base_bits={base_bits} frame_symbols={frame_symbols} "
          f"sync_word={len(sync_word)} threshold={sync_threshold} "
          f"search_range=±{correlator.search_range} "
          f"post_acq_search=±{post_acq_search_smp} "
          f"alpha={alpha:.4f}",
          flush=True)

    # The MF cross-correlation peak falls at the END of the chirp, not the
    # end of the full symbol slot.  Each n_read = sps_eff block places
    # [chirp(sps) | gap(gap_samps)] at ring[-sps_eff:].  The chirp's last
    # sample is therefore at ring[buf_size - gap_samps - 1], not buf_size-1.
    #
    # Using buf_size-1 biases timing_err by -gap_samps every symbol.  In
    # settled tracking (post_acq_search = sps×0.04) the bias exceeds the
    # search radius whenever gap_frac > 0.04, meaning the true peak is
    # completely outside the search window and the decoder reads noise.
    expected_peak = buf_size - gap_samps - 1
    peak_est      = expected_peak
    timing_frac   = 0.0
    timing_settled = False

    # ── Rate-aided (Type-2) loop state ───────────────────────
    # rate_est_smp: persistent estimate of the per-symbol sample-count drift
    # between the TX and RX clocks (samples/symbol).  Applied as a constant
    # additive term to n_read, separate from int_corr (the phase-residual
    # term derived from timing_frac).
    #
    # WHY THIS IS NEEDED: in settled mode, timing_frac is clipped to
    # ±post_acq_search_smp (≈ ±sps×0.04).  A type-1 loop (phase integrator
    # only) tracking a constant-rate disturbance D (clock drift, samples/
    # symbol) settles toward steady-state error terr_ss = D / alpha.  For
    # 5 s symbols, alpha = 0.005 and post_acq_search_smp ≈ 0.04×sps, so the
    # loop can only sustain D ≤ alpha × post_acq_search_smp ≈ 44
    # samples/symbol (≈200 ppm) before terr_ss would exceed the search
    # window.  Drift above that doesn't fail instantly — terr ramps from
    # ≈0 toward the (unreachable) terr_ss = D/alpha with time constant
    # 1/alpha = 200 symbols, CROSSING post_acq_search_smp at a predictable
    # symbol count.  For D ≈ 200 smp/sym (≈900 ppm — well within the
    # range of mismatched consumer sound cards), that crossing happens at
    # ≈46 symbols ≈ 230 s at 5 s/symbol — matching "half symbol off after
    # 200-300 seconds" almost exactly.  This is real drift interacting with
    # a loop that was never given a way to absorb a rate error outside the
    # phase window — both "time drift" and "calculation gap" at once.
    #
    # rate_est_smp removes D from the phase loop's burden entirely: once
    # populated, n_read tracks sps_eff + D directly, leaving int_corr (and
    # thus terr) to track only the residual (~0), which stays comfortably
    # inside ±post_acq_search_smp regardless of how large D is.
    rate_est_smp   = 0.0
    # Rolling window of (n_read - sps_eff) from the tail of each
    # acquisition phase (alpha_acq, wide search) — used to capture
    # rate_est_smp once acquisition has converged.  See the
    # "_acq_syms_left == 0" capture below.
    _acq_nread_dev = deque(maxlen=5)
    # winner_pk noise-floor EMA for the strength gate (see wpk_track_* config).
    # Tracks the typical correlation-peak level; strong genuine chirps stand
    # out as a large multiple of this baseline.  Updated from every symbol so
    # it reflects the running noise floor; bootstrapped on first use.
    _wpk_ema = None

    ring           = np.zeros(buf_size, dtype=np.float32)
    # ── Noise-whitening filter state ──────────────────────────────────────
    # _whiten_psd_ema: slow per-bin background POWER estimate (see the
    #   whiten_enable comment above for why this is built from a SHORT-TIME
    #   FFT median across time, not one FFT over the whole symbol).  Sized
    #   to whiten_stft_nfft, a small, FIXED reference length independent of
    #   n_read (which varies slightly symbol to symbol as the PLL adjusts
    #   it).
    # _whiten_fir: current time-domain whitening filter taps, convolved with
    #   every incoming chunk.  Re-designed every whiten_redesign_every
    #   symbols, not every symbol — the background spectrum changes slowly
    #   relative to one symbol's duration.
    whiten_stft_win  = max(64, int(round(whiten_stft_win_s * fs)))
    whiten_stft_hop  = max(1, whiten_stft_win // 2)
    whiten_stft_nfft = 1 << (whiten_stft_win - 1).bit_length()
    _whiten_stft_window    = np.hanning(whiten_stft_win).astype(SIG)
    _whiten_psd_ema        = None
    _whiten_fir            = None
    _whiten_redesign_count = 0
    bit_stream     = []
    pre_sync_buf   = deque(maxlen=syndet.lookback)
    sync_found     = False
    frames_decoded = 0
    frames_errored = 0
    consec_errors  = 0
    rms_history    = []
    snr_history    = []
    decoded_frames = []
    started_at     = time.time()
    last_status_t  = time.time()
    sym_count        = 0
    timing_err       = 0
    n_read           = sps_eff
    _acq_syms_left   = 0   # counts down post-sync fast-acquisition symbols
    _last_micsn_log  = 0.0 # wall time of the last MIC_SN line (throttled ~1 Hz)
    _last_resync_sym = 0   # sym_count at the most recent resync (or 0 at start)
    _last_resync_soft = False  # True if last resync was a soft re-arm (preamble imminent)
    _terr_buf      = deque(maxlen=timing_median_n)  # median filter for echo rejection
    _frame_audio     = []   # AGC-processed audio chunks collected during frame
    _fa_nbytes       = 0    # running byte count for _frame_audio (avoids re-summing)
    # ── CRC-anchored retro decode: buffers and budget ────────────────────────
    # Retro FFT compute budget (hoisted so the look-ahead/enable decision can use
    # it).  Mirrors the per-frame auto-scale: ≥4 M samples, growing so the IRLS
    # fit always has ≥24 symbol peaks; a fixed max_retro_samples overrides.
    _retro_samples_cfg = int(cfg.get('max_retro_samples', 0))
    _max_retro_smp = (max(4_000_000, 24 * sps_eff) if _retro_samples_cfg == 0
                      else _retro_samples_cfg)
    # Look-ahead: a cold-start mis-aligned acquisition makes the first frame
    # "complete" before its true payload has fully arrived, so the tail is
    # missing from the buffer and NO shift can recover it.  Defer the frame
    # decode by a few symbols (which keep accumulating into _audio_hist) so the
    # tail is captured.  Only when the anchor path is viable (short enough
    # symbols that the wider buffer's FFT stays within the compute budget) —
    # otherwise the extra latency buys nothing.
    _anchor_lookahead = int(cfg.get("retro_anchor_lookahead", 8))
    # Rolling RAW audio history kept at ALL times (pre- and post-sync), one
    # chunk per symbol, long enough to hold a full frame even when acquisition
    # fired late, plus the look-ahead tail.  Feeds the CRC-anchored retro search
    # so the frame boundary is re-derived empirically — fixes the cold-start
    # first-frame CRC failure.
    _audio_hist_len  = (frame_symbols + _training_symbols + _n_preamble
                        + _anchor_lookahead + 8)
    _audio_hist      = deque(maxlen=_audio_hist_len)
    _anchor_enabled  = (_audio_hist_len * sps_eff <= _max_retro_smp)
    if not _anchor_enabled:
        _anchor_lookahead = 0   # long symbols: no anchor, no added latency
    # Max whole-symbol shift the CRC-anchored retro search will try when the
    # streaming acquisition lands on the wrong boundary (cold-start first frame).
    # Default covers the full history (any frame position within the buffer);
    # the search stops at the first CRC pass, so aligned frames cost one decode.
    retro_anchor_max_shift = int(cfg.get("retro_anchor_max_shift",
                                         _audio_hist_len - frame_symbols))
    _syndet_bg_ema   = 0.0  # EMA of background (non-sync) syndet scores
    # Slow cross-symbol background-amplitude EMA, used by the impulse-blanker
    # block in the main loop. A per-CHUNK median is blind to noise that
    # occupies a large fraction of that same chunk (the median rises right
    # along with the noise). This EMA updates slowly across many symbols, so
    # a burst that dominates one or two chunks barely moves it: the blanker
    # can then recognise "this whole chunk is anomalously loud vs. the
    # established baseline" even when nothing within the chunk itself looks
    # like an outlier.
    _bg_amp_ema      = None
    impulse_bg_alpha = float(cfg.get("impulse_bg_alpha", 0.05))
    # Timing snapshot: (timing_frac, n_read, peak_est) captured from a stable
    # mid-frame symbol.  Used to restore a known-good PLL state at the next
    # frame boundary if the retro decode is unavailable.
    _timing_snapshot = None
    # One-shot ring correction: added to a single stream.read() advance then
    # zeroed immediately.  Set by the retro inter-frame resync; consumed before
    # nread_revert can see it, so the phase correction cannot compound across reads.
    _one_shot_correction = 0

    diag = {
        "sync_acquired": False, "sync_at_symbol": None, "sync_score": 0.0,
        "symbols_received": 0, "peak_rms": 0.0, "mean_rms": 0.0,
        "silence": False, "avg_snr_db": 0.0, "input_overflows": 0,
    }

    def do_resync(reason="", reset_timing=False):
        nonlocal sync_found, bit_stream, consec_errors, timing_settled
        nonlocal peak_est, timing_frac, n_read, _acq_syms_left, _last_resync_sym
        nonlocal _last_resync_soft
        nonlocal _timing_snapshot, _fa_nbytes, _one_shot_correction, rate_est_smp
        print(f"RESYNC{': ' + reason if reason else ''}"
              f"{' (timing reset)' if reset_timing else ' (soft)'}",
              flush=True)
        sync_found       = False
        bit_stream       = []
        consec_errors    = 0
        timing_settled   = not reset_timing
        _acq_syms_left   = 0
        _last_resync_sym = sym_count   # anchor early-arrival guard to this resync
        _last_resync_soft = not reset_timing   # soft re-arm => next preamble imminent
        _terr_buf.clear()
        _frame_audio.clear()
        _fa_nbytes           = 0
        _timing_snapshot     = None
        _one_shot_correction = 0
        if reset_timing:
            peak_est     = expected_peak
            timing_frac  = 0.0
            n_read       = sps_eff
            # Prevent stale drift from outlasting a hard resync.
            # n_read = sps_eff + round(rate_est_smp) + int_corr, so
            # resetting n_read=sps_eff here is immediately overridden on
            # the very next PLL tick if rate_est_smp is non-zero.  Clear
            # it so the new acquisition starts from a clean n_read=sps_eff
            # baseline and captures a fresh rate estimate.
            rate_est_smp = 0.0
            _acq_nread_dev.clear()
        syndet.clear_history()
        pre_sync_buf.clear()

    # ── Spectrograph telemetry socket (UDP, fire-and-forget) ─────────────────
    # Broadcasts live decoder state to myspectrograph.py on 127.0.0.1:55555.
    # All sends are inside try/except so a missing listener can never affect
    # decoder operation.
    import socket as _telem_mod
    _telem      = _telem_mod.socket(_telem_mod.AF_INET, _telem_mod.SOCK_DGRAM)
    _TELEM_ADDR = ("127.0.0.1", 55555)

    def _run_pipeline(fb, fc, frame_snr=0.0):
        """Frame decode for the streaming PLL path — thin adapter over
        _frame_pipeline.  Used by BOTH the per-frame decode and the CRC-anchored
        offset search (at completion AND the deferred retry) as a ground-truth
        alignment oracle.  Enables RS erasures (era_thr) and diagnostic prints."""
        return _frame_pipeline(
            fb, fc, interleave_rows=interleave_rows, conv_code=conv_code,
            diff_enc=diff_enc, diff_ref=sync_word[-1], base_bits=base_bits,
            rs_codec=rs_codec, chunk_size=chunk_size, era_thr=era_thr,
            frame_snr=frame_snr, verbose=True)

    # ── Deferred anchor re-decode state ──────────────────────────────────────
    # A cold-start mis-aligned first frame "completes" before its payload tail
    # has arrived, so the at-completion anchor search can't recover it (the tail
    # isn't buffered yet).  When a frame fails, we schedule a retry a few symbols
    # later — by then _audio_hist (which fills every symbol) holds the tail.
    _decoded_seqs    = set()   # seqs already emitted (dedupe streaming vs retry)
    _anchor_retry    = 0       # symbols remaining until the deferred retry fires
    _anchor_retry_snr = 0.0

    def _try_anchor_decode(frame_snr):
        """Thin adapter binding the current PLL state into the pure
        _anchor_decode oracle (rate_est_smp drifts, so it is captured fresh)."""
        return _anchor_decode(
            _audio_hist, _run_pipeline, anchor_enabled=_anchor_enabled,
            frame_symbols=frame_symbols, up_ref=up_ref, down_ref=down_ref,
            sps=sps, sps_eff=sps_eff,
            retro_anchor_max_shift=retro_anchor_max_shift,
            rate_est_smp=rate_est_smp, frame_snr=frame_snr)

    # ── Audio interface probe ─────────────────────────────────────────────────
    # Log the real negotiated input device, host API, and native (shared-mode)
    # rate BEFORE opening.  If the device's shared-mode mixer runs at a different
    # rate than `fs` (e.g. a 48000 Hz mic feeding a 44100 Hz stream), PortAudio
    # resamples every block — a prime cause of chirp-timing jitter and dropped
    # sample blocks that desync the symbol grid.
    _input_overflows = 0
    _overflowed      = False
    # Only probe a REAL interface.  Offline repro / loopback tests mock
    # sounddevice with a fake stream (no query_devices), so gate on it.
    if _log_audio_interface is not None and hasattr(sd, "query_devices"):
        try:
            _log_audio_interface("input", device, fs, channels=1)
        except Exception as _iface_exc:
            print(f"AUDIO_IFACE input probe error: {_iface_exc}", flush=True)

    # Real hardware: use the overflow-proof callback capture so ADC blocks are
    # never dropped during per-symbol DSP.  Mocked sounddevice (offline repro /
    # loopback) lacks query_devices and a callback API, so fall back to the
    # plain blocking InputStream the tests already supply.
    if hasattr(sd, "query_devices"):
        _input_ctx = _QueuedInputStream(sd, fs, device, channels=1)
    else:
        _input_ctx = sd.InputStream(samplerate=fs, channels=1, dtype="float32",
                                    device=device)
    try:
        with _input_ctx as stream:
            # What PortAudio actually negotiated for the open stream (mock streams
            # in tests lack these attributes — skip quietly there):
            try:
                print(f"AUDIO_IFACE input stream OPEN samplerate={float(stream.samplerate):.0f}Hz "
                      f"blocksize={stream.blocksize} latency={float(stream.latency)*1000:.1f}ms "
                      f"device={stream.device} capture={'queued-callback' if hasattr(sd,'query_devices') else 'blocking-mock'}",
                      flush=True)
            except Exception:
                pass
            signal_present = True
            while True:
                if time.time() - started_at > timeout_sec:
                    break
                if stop_event is not None and stop_event.is_set():
                    break
                if _sigterm_flag[0]:
                    print("SIGTERM received — flushing result", flush=True)
                    break
                if stop_file and Path(stop_file).exists():
                    print("stop_file present — flushing result", flush=True)
                    break

                # Apply any pending inter-frame phase correction in this one
                # read, then zero it so nread_revert cannot compound it over
                # subsequent symbols.  n_read itself is unchanged — it holds
                # the PLL's ongoing rate/phase state as usual.
                _nr = n_read + _one_shot_correction
                _one_shot_correction = 0
                raw, _overflowed = stream.read(_nr)
                raw    = raw.flatten()
                # PortAudio sets overflowed=True when it had to discard captured
                # input between reads (buffer overrun).  Each overrun silently
                # removes samples from the stream, shifting every subsequent chirp
                # earlier in the ring and corrupting the symbol grid — exactly the
                # "rolls in and out of sync" failure.  Surface it loudly.
                if _overflowed:
                    _input_overflows += 1
                    print(f"INPUT_OVERFLOW count={_input_overflows} "
                          f"t={time.time()-started_at:.2f}s _nr={_nr} — ADC dropped samples "
                          f"since last read (timing slip / chirp desync likely)", flush=True)
                if _capture is not None:
                    _capture.append(raw.copy())
                # Rolling RAW audio history (pre-AGC, pre-blank) for the
                # CRC-anchored retro search.  Must be RAW: the cold-start AGC
                # gain ramp and the impulse blanker's zeroed samples distort the
                # matched-filter envelope enough to break the offset fit, while
                # the raw signal (what laptop_input.wav captures) re-anchors
                # cleanly.  Kept for every symbol, pre- and post-sync, so the
                # frame is present even when acquisition fired late.
                _audio_hist.append(raw.copy())

                # ── Deferred anchor re-decode ─────────────────────────────────
                # Fires `_anchor_lookahead` symbols after a frame failed, by
                # which point the rolling history holds that frame's payload
                # tail.  Recovers the cold-start first frame that "completed"
                # before its payload finished arriving.
                if _anchor_retry > 0:
                    _anchor_retry -= 1
                    if _anchor_retry == 0:
                        _ok, _seq, _dat = _try_anchor_decode(_anchor_retry_snr)
                        if _ok and _seq not in _decoded_seqs:
                            # CRC-valid: keep raw bytes; UTF-8 is best-effort
                            # (binary modem packets must not be dropped).
                            _msg = _dat.decode("utf-8", errors="replace")
                            print(f"FRAME seq={_seq} "
                                  f"snr={_anchor_retry_snr:.1f}dB rs=0 "
                                  f"src=retro_deferred text={_msg}", flush=True)
                            decoded_frames.append({
                                "seq": _seq, "text": _msg,
                                "bytes_hex": _dat.hex(),
                                "snr_db": round(_anchor_retry_snr, 1),
                                "rs_errors": 0, "source": "retro_deferred"})
                            frames_decoded += 1
                            frames_errored  = max(0, frames_errored - 1)
                            _decoded_seqs.add(_seq)

                # Impulse blanker. Two distinct interference regimes occur in
                # real captures: (a) brief clicks/knocks, tens of ms, a small
                # fraction of one symbol; (b) sustained moderate-amplitude
                # bursts (speech, footsteps, room noise) lasting hundreds of
                # ms up to multiple seconds, comparable to or longer than one
                # symbol itself.
                #
                # The blank threshold is anchored to a SLOW, cross-symbol EMA
                # of the background amplitude level (_bg_amp_ema), not to this
                # chunk's own median. A per-chunk median is blind to regime
                # (b): if a loud burst occupies most of the chunk, the
                # chunk's own median rises right along with it and nothing
                # looks like an "outlier" anymore.
                # Anchoring to the slowly-updated baseline instead means a
                # chunk that's mostly loud is still correctly recognised as
                # anomalous, and every sample exceeding the baseline-relative
                # threshold gets blanked — regardless of what fraction of the
                # chunk that turns out to be.  Only whatever's left below
                # threshold (we still saw clean ~20-40ms gaps even inside the
                # worst bursts) feeds the matched filter.
                _abs_raw   = np.abs(raw)
                _chunk_med = float(np.median(_abs_raw))
                if _bg_amp_ema is None:
                    _bg_amp_ema = _chunk_med          # bootstrap on first chunk
                if impulse_blank_enable:
                    # min(), not max(): blank a sample if it exceeds EITHER
                    # threshold (actual clipping, OR too far above baseline).
                    # Using max() here would mean clip_floor (0.98) always
                    # overrides the far more sensitive baseline-relative
                    # threshold whenever bg_amp_ema is small — which it
                    # normally is for a deliberately faint, sub-noise-floor
                    # signal — silently disabling the relative check entirely.
                    _blank_thr = min(impulse_clip_floor,
                                      impulse_blank_k * _bg_amp_ema + 1e-9)
                    _blank_mask = _abs_raw > _blank_thr
                    _n_blanked  = int(_blank_mask.sum())
                    _blanked_frac = _n_blanked / len(raw)
                    if _n_blanked > 0 and _blanked_frac > impulse_blank_max_frac:
                        # Sustained-noise regime: blanking this much would
                        # remove genuine chirp content, not just noise.
                        # Skip blanking for this symbol entirely and let the
                        # matched filter's own frequency selectivity and
                        # partial-chirp slice fallback work on the full,
                        # unmodified (degraded but still present) signal.
                        print(f"IMPULSE_BLANK sym={sym_count} SKIPPED — "
                              f"would blank {_n_blanked}/{len(raw)} "
                              f"({_blanked_frac*100:.2f}%) > "
                              f"max_frac={impulse_blank_max_frac*100:.0f}%; "
                              f"sustained noise, not a brief impulse — "
                              f"leaving symbol unmodified",
                              flush=True)
                    elif _n_blanked > 0:
                        raw = raw.copy()
                        raw[_blank_mask] = 0.0
                        print(f"IMPULSE_BLANK sym={sym_count} "
                              f"n={_n_blanked}/{len(raw)} "
                              f"({_blanked_frac*100:.2f}%) "
                              f"thr={_blank_thr:.3f} bg_ema={_bg_amp_ema:.4f}",
                              flush=True)
                # Update the baseline EMA from this chunk's pre-blank median,
                # but only if the chunk doesn't itself look heavily
                # contaminated (>3x current baseline) — guards against one
                # very loud chunk dragging the reference up and desensitising
                # the blanker for whatever comes next.  Genuine slow drift in
                # ambient level still tracks fine: most chunks during such a
                # drift look "not too contaminated" relative to the
                # recently-updated EMA, so it keeps adapting.
                if _bg_amp_ema == 0 or _chunk_med < 3.0 * _bg_amp_ema:
                    _bg_amp_ema += impulse_bg_alpha * (_chunk_med - _bg_amp_ema)

                # ── Noise-whitening filter ───────────────────────────────────
                # Estimate this chunk's BACKGROUND spectrum via a short-time
                # FFT (STFT) with the MEDIAN taken across time slices, per
                # bin — not a single FFT over the whole symbol (that smears
                # the chirp's own sweep across the whole band, making it
                # look just as "persistent" as real hum; see whiten_enable
                # comment above).  whiten_stft_win is short enough that the
                # chirp only dwells in a small minority of the time-slices
                # at any one bin, so the median there is dominated by
                # whatever's ACTUALLY persistent — exactly mirroring how a
                # human reads a spectrogram: a tone lights up nearly every
                # column, the chirp lights up only briefly in each row.
                if whiten_enable:
                    _n_frames = max(1, (len(raw) - whiten_stft_win) // whiten_stft_hop + 1)
                    if _n_frames >= 3:   # need enough time-slices for the median to mean anything
                        _frame_mags = np.empty((_n_frames, whiten_stft_nfft // 2 + 1),
                                                dtype=SIG)
                        _raw64 = raw.astype(SIG)
                        for _fi in range(_n_frames):
                            _s0 = _fi * whiten_stft_hop
                            _seg = _raw64[_s0 : _s0 + whiten_stft_win] * _whiten_stft_window
                            _frame_mags[_fi] = np.abs(np.fft.rfft(_seg, whiten_stft_nfft))
                        # Per-bin median magnitude across time -> power.
                        _chunk_psd = np.median(_frame_mags, axis=0) ** 2
                        if _whiten_psd_ema is None:
                            _whiten_psd_ema = _chunk_psd.copy()
                        else:
                            _whiten_psd_ema += whiten_bg_alpha * (_chunk_psd - _whiten_psd_ema)

                    _whiten_redesign_count += 1
                    if (_whiten_psd_ema is not None and
                            (_whiten_fir is None or _whiten_redesign_count >= whiten_redesign_every)):
                        _whiten_redesign_count = 0
                        # Desired response: 1/sqrt(power) -- flattens persistent
                        # peaks, regularised against near-zero bins (which would
                        # otherwise blow up into huge, noise-amplifying gain).
                        # Filter DESIGN in float64 deliberately: this runs only
                        # every whiten_redesign_every chunks (not per sample), and
                        # 1/sqrt of near-zero PSD bins is the one genuinely
                        # conditioning-sensitive step in the chain.  Only the
                        # resulting TAPS go to float32, for the per-sample
                        # convolution below.  An embedded port splits the same way:
                        # design cheaply/rarely, filter hot in single precision.
                        _psd64 = _whiten_psd_ema.astype(np.float64)
                        _valid = _psd64[_psd64 > 0]
                        _med_psd = float(np.median(_valid)) if _valid.size else 1.0
                        _floor = _med_psd * 1e-3 + 1e-20
                        _gain = 1.0 / np.sqrt(_psd64 + _floor)
                        # Normalise so a TYPICAL (median-power) bin keeps gain
                        # ≈1.0 -- whitening should re-shape the spectrum, not
                        # change its overall level (that's AGC's job, applied
                        # after this stage).
                        _ref_gain = 1.0 / np.sqrt(_med_psd + _floor)
                        _gain = _gain / (_ref_gain + 1e-20)
                        # Cap how hard any single bin can be boosted or cut --
                        # an unregularised whitening filter would try to fully
                        # cancel a bin with near-zero historical energy, which
                        # just amplifies whatever noise happens to land there
                        # next.  whiten_max_gain_db bounds this to something
                        # the filter can safely commit to.
                        _max_gain_lin = 10 ** (whiten_max_gain_db / 20.0)
                        _gain = np.clip(_gain, 1.0 / _max_gain_lin, _max_gain_lin)
                        # Build a zero-phase FIR via windowed IFFT of the gain
                        # curve, truncated to whiten_taps.  Zero-phase (not
                        # minimum-phase) is essential here: it doesn't shift
                        # the chirp in time, only reshapes its spectrum --
                        # any phase distortion would directly corrupt the
                        # matched filter's timing precision, undermining the
                        # entire reason this decoder works.  Note: the gain
                        # curve here is at whiten_stft_nfft resolution (much
                        # coarser than the old whiten_nfft approach), which is
                        # fine -- the hum/mains harmonics this targets are
                        # narrowband but not infinitely so, and whiten_taps
                        # sets the FIR's actual achievable frequency
                        # resolution regardless of the design curve's bin
                        # spacing.
                        _full_ir = np.fft.irfft(_gain, whiten_stft_nfft)
                        _full_ir = np.fft.fftshift(_full_ir)
                        _centre  = len(_full_ir) // 2
                        _half    = whiten_taps // 2
                        if len(_full_ir) >= whiten_taps:
                            _fir = _full_ir[_centre - _half : _centre + _half + 1]
                        else:
                            # whiten_stft_nfft smaller than requested taps --
                            # tile/pad rather than crash; shouldn't happen
                            # with the default config but stay safe.
                            _fir = np.zeros(whiten_taps)
                            _fir[:len(_full_ir)] = _full_ir
                        _fir     = _fir * np.hamming(len(_fir))
                        _dc_gain = float(np.sum(_fir))
                        if abs(_dc_gain) > 1e-9:
                            _fir = _fir / _dc_gain
                        _whiten_fir = _fir.astype(SIG)

                    if _whiten_fir is not None:
                        # FFT-based convolution ('same'-length, like
                        # np.convolve mode='same') -- chunks here run into the
                        # hundreds of thousands of samples, where direct-method
                        # convolution against a ~2000-tap filter would be
                        # prohibitively slow; FFT convolution keeps this
                        # comfortably within the real-time budget.
                        _n_conv  = len(raw) + len(_whiten_fir) - 1
                        _nfft_c  = 1 << (_n_conv - 1).bit_length()
                        _X       = np.fft.rfft(raw.astype(SIG), _nfft_c)
                        _H       = np.fft.rfft(_whiten_fir, _nfft_c)
                        _y_full  = np.fft.irfft(_X * _H, _nfft_c)[:_n_conv]
                        _start   = (len(_whiten_fir) - 1) // 2
                        raw      = _y_full[_start : _start + len(raw)].astype(SIG)

                rms_history.append(float(np.sqrt(np.mean(raw ** 2))))
                agc.track_rms(raw)
                signal_present = float(np.sqrt(np.mean(np.square(raw, dtype=SIG),
                                                       dtype=np.float64))) > silence_thr

                ring = np.roll(ring, -_nr)
                agc_audio = agc.apply(raw)
                ring[-_nr:] = agc_audio
                sym_count += 1

                # Buffer audio for retrospective two-pass decode.
                # Only accumulate while collecting frame bits (sync_found=True).
                # Cleared on every resync so stale audio never bleeds into
                # the next frame.
                if sync_found:
                    _frame_audio.append(agc_audio)
                    _fa_nbytes += agc_audio.nbytes

                correlator.verbose = (sym_count in _verbose_syms)
                bit, confidence, found_peak, snr_db, winner_pk = (
                    correlator.symbol_decision(
                        ring, peak_est,
                        search_range_override=(
                            post_acq_search_smp
                            if (timing_settled and sync_found and _acq_syms_left == 0)
                            else None
                        )
                    ))

                abs_conf = abs(confidence)

                # Slope-consistency check, computed BEFORE the PLL trusts
                # found_peak, not after.
                #
                # The matched filter's found_peak is a global argmax: over the
                # whole search window, it picks wherever the correlation
                # magnitude happens to be highest.  At low SNR (this decoder
                # is explicitly used down to "signal slightly above background
                # noise"), that argmax can land on a position whose magnitude
                # is merely a noise-driven local maximum, sharing nothing with
                # the genuine chirp's actual position.  Nothing about a tall
                # correlation magnitude guarantees the underlying audio at
                # that position actually traces a linear frequency sweep —
                # only if_slope_confidence() checks that directly, by fitting
                # the segment's measured instantaneous frequency against the
                # expected linear ramp and reporting how well it fits (r2).
                #
                # This is the literal equivalent of an image-tracing tool
                # verifying local shape continuity before extending a traced
                # path, versus blindly jumping to the brightest pixel in a
                # search box.  Previously this exact check existed (see the
                # confidence-blend block below) but ran AFTER peak_est had
                # already been updated from found_peak — a diagnostic with no
                # connection to the decision it should have gated.  A
                # noise-driven peak with high magnitude but a flat/incoherent
                # instantaneous-frequency profile (r2 near 0) could walk
                # peak_est step by step, each step individually within the
                # eff_max_terr clamp (Fix 8) and never flagged by the
                # statistical outlier gate (Fix 2/3, which only compares a
                # symbol to recent HISTORY — useless once a few such steps
                # have already shifted that history).  None of the existing
                # gates ask "does this position even look like a chirp" — only
                # this one does, and it wasn't wired to anything that mattered.
                #
                # found_peak (not peak_est) is used for the segment: it's
                # exactly the audio the matched filter just evaluated, so the
                # slope measurement is a genuine cross-check of THIS
                # candidate, not a stale prediction.
                pll_conf, pll_r2 = 0.0, 0.0
                if pll_weight > 0 and signal_present:
                    seg_start = max(0, found_peak - sps + 1)
                    seg_end   = min(len(ring), found_peak + 1)
                    pll_conf, pll_r2 = if_slope_confidence(
                        ring[seg_start:seg_end], f_min, f_max, sps, fs)
                pll_slope_min_r2 = float(cfg.get("pll_slope_min_r2", 0.10))
                # ── REVERTED: see PLL_SLOPE_REJECT history ──────────────────
                # This gate was disabled after a real production log showed it
                # causing exactly the failure mode it was meant to prevent.
                # Across 32 CONSECUTIVE settled-mode symbols at otherwise
                # normal SNR (10.8-12.4dB), pll_r2 never exceeded 0.063 — every
                # single symbol was rejected, n_read froze at one value for the
                # entire window (confirmed in STATUS lines: n_read=92160,
                # unchanged for 30+ symbols), and the un-corrected timing
                # walked far enough that the RETRO fit's sanity gate eventually
                # tripped (delta=-90208 smp, far past the 2x threshold),
                # forcing a full resync and failing CRC outright.
                #
                # The same pattern showed up in offline testing on a totally
                # different recording: a CONFIRMED-CORRECT decode (frame 1,
                # retro-verified) had pll_r2 = 0.001-0.026 throughout — just as
                # low as a genuinely bad symbol.  Both pieces of evidence point
                # the same way: if_slope_confidence()'s linear fit, computed
                # over a FULL 2-3 second symbol, apparently doesn't reach
                # r2 >= 0.10 for real recorded audio AT ALL, regardless of
                # whether the position is correct — almost certainly because
                # real-world effects (reverb, multipath, mic nonlinearity)
                # break a clean single linear fit across that many samples
                # even when the underlying chirp is genuinely well-aligned.
                # A threshold that's unreachable by both good and bad symbols
                # alike isn't discriminating anything — it's just an always-on
                # kill switch on the PLL's ability to correct timing, which is
                # strictly worse than having no slope check at all (the
                # statistical outlier gate, Fix 2/3, already provides
                # protection that adapts over time instead of freezing
                # completely).  pll_r2 is still computed (used by the
                # confidence blend below) but no longer gates anything here.
                # Re-enabling this needs the metric recalibrated against real
                # recordings first, not reasoned about from first principles.
                pll_slope_ok = True

                # ── Timing PLL ───────────────────────────────────────────────
                # Gate timing updates on conf AND SNR.  During silence or
                # AGC-settling, noise correlations can reach high conf at low
                # SNR and drive spurious timing drift.  Without the SNR gate,
                # that drift accumulates during the lead-silence period and is
                # revealed as a large terr (-5000+ smp) when the real signal
                # arrives — degrading MF SNR for the first ~80 frame symbols
                # at long sps.
                # pll_slope_ok is always True (gate disabled — see the
                # REVERTED comment above); kept in the condition rather than
                # removed so re-enabling it later is a one-line change.
                tim_snr_ok = snr_db >= timing_min_snr_db
                if (signal_present and abs_conf >= min_tim_conf and tim_snr_ok
                        and pll_slope_ok):
                    raw_err = found_peak - expected_peak
                    eff_max_terr = max_terr_smp if timing_settled else presync_max_terr_smp
                    clamped_err = int(np.clip(raw_err, -eff_max_terr, eff_max_terr))
                    # peak_est: IIR update ONLY while timing has not settled.
                    # Once settled, pin peak_est = expected_peak so that n_read
                    # is the *sole* corrector.  Keeping both loops active creates
                    # a second-order system that oscillates with long symbols:
                    # peak_est tracks found_peak with a 20-symbol lag, generating
                    # a persistent residual error that feeds back into timing_frac
                    # even after n_read has already corrected the drift.
                    #
                    # Clamp the IIR *target*, not just timing_err.
                    # clamped_err already bounds the deviation used for the PLL
                    # and outlier gate, but peak_est's IIR was pulling toward
                    # the RAW found_peak — completely bypassing that clamp.
                    # During acquisition the search window is wide (±sps*0.25,
                    # no override), so a single outlier found_peak (impulse,
                    # echo, sidelobe) shifts peak_est, which re-centres next
                    # symbol's wide search, which can land on an even-further
                    # outlier — a positive-feedback runaway with no leash.
                    # Observed: peak_est walked >8000 smp in 20 symbols while
                    # timing_err sat pinned at the clamp boundary the entire
                    # time, because the clamp was never applied to the thing
                    # actually driving the search centre.  Pulling toward
                    # (expected_peak + clamped_err) instead bounds every IIR
                    # step to within eff_max_terr of the true expected position,
                    # so the runaway is structurally impossible.
                    #
                    # ── peak_est tracking in settled mode ────────────────────
                    # The original code froze peak_est=expected_peak in settled
                    # mode to prevent a "second-order oscillation" where both
                    # peak_est and n_read correct simultaneously.  This works
                    # when n_read can fully absorb the clock error each symbol
                    # (error < alpha × post_acq_search_smp).  For long symbols
                    # (2-3s) where even 100ppm drift is 441 smp/symbol but
                    # alpha × search = 353 smp max, the loop CAN'T close:
                    # the chirp drifts to the window edge in 8 symbols, found_
                    # peak becomes a noise argmax, and the outlier gate median
                    # gets stuck at the boundary value permanently — proved
                    # empirically (median=+3250, all subsequent measurements
                    # rejected, n_read frozen).  Enabling the same IIR here
                    # (clamped to ±eff_max_terr, gated on not _pll_outlier)
                    # lets the search window FOLLOW the chirp as it drifts,
                    # keeping it centred rather than waiting for n_read alone
                    # to complete a correction it's arithmetically too slow to
                    # finish.  A separate slower alpha (tim_track_alpha) is
                    # used for settled vs fast alpha_acq for acquisition, so
                    # the window follows without over-reacting to single-symbol
                    # noise.  The oscillation concern doesn't apply here: both
                    # loops drive toward the same setpoint (clamped_err→0),
                    # and with the narrow settled search (±post_acq_search_smp)
                    # the outlier gate blocks noise spikes before they can move
                    # peak_est at all.
                    _track_alpha = tim_track_alpha
                    # Acquisition-phase IIR: only runs once sync is confirmed
                    # (sync_found=True).  Without this guard the IIR runs on
                    # every pre-sync symbol (timing_settled can be False at
                    # decoder start before the syndet_flush path sets it True),
                    # letting 30+ symbols of random background noise drag
                    # peak_est hundreds or thousands of samples from
                    # expected_peak before the real chirp ever arrives.
                    # Settled-mode tracking is done after the outlier gate
                    # (where _pll_outlier is available) so noise peaks don't
                    # move peak_est before being rejected.
                    if sync_found and (not timing_settled or _acq_syms_left > 0):
                        peak_est = int(round(
                            (1.0 - _track_alpha) * peak_est +
                             _track_alpha         * (expected_peak + clamped_err)
                        ))
                    # Settled-mode IIR: applied in the PLL update block below
                    # (after _pll_outlier is determined) so only accepted
                    # measurements move peak_est.
                    timing_err = clamped_err
                else:
                    timing_err = found_peak - expected_peak

                # ── FRAME_PHASE diagnostic (once per frame) ──────────
                # _acq_syms_left == timing_acq_symbols is true for exactly
                # one symbol per frame: the symbol immediately after
                # SYNC_ACQUIRED set it to that value, BEFORE this symbol's
                # countdown decrements it.  symbol_decision for THIS symbol
                # therefore used search_range_override=None (the wide
                # ±sps*0.25 acquisition window), so timing_err here is an
                # UNCLAMPED (within ±sps*0.25) measurement of how far the
                # new frame's first chirp landed from expected_peak — i.e.
                # the per-frame phase jump Δ_frame, independent of
                # rate_est_smp (the global, persistent rate estimate).
                #
                # Δ_frame ≈ 0 consistently  -> frames are phase-continuous;
                #                              the global clock (rate_est_smp)
                #                              alone predicts each new frame's
                #                              chirp position, and the 20-symbol
                #                              acquisition window is a no-op.
                # Δ_frame ≈ constant K != 0 -> a fixed inter-frame gap of K
                #                              samples exists that sps_eff /
                #                              gap_samps does not model — a
                #                              config-level term, not a PLL bug.
                # Δ_frame varies frame-to-frame -> either genuine per-frame
                #                              re-sync jitter (the wide window
                #                              + alpha_acq handles this, by
                #                              design — this IS the "per-frame
                #                              clock allowed to run separately"),
                #                              or a coarse_t0 fold issue (see
                #                              sps_eff_d in retro_timing_decode).
                if sync_found and _acq_syms_left == timing_acq_symbols:
                    print(f"FRAME_PHASE sym={sym_count} "
                          f"delta_frame={timing_err:+d}smp "
                          f"rate_est={rate_est_smp:+.2f} "
                          f"expected_peak={expected_peak} "
                          f"peak_est={peak_est}", flush=True)

                if (sync_found and timing_settled and _acq_syms_left == 0
                        and sym_count in _verbose_syms):
                    # Compact per-symbol tracking summary (enabled via verbose_syms)
                    print(f"SYM_TRACK sym={sym_count} "
                          f"wpk={winner_pk:.3f} conf={confidence:+.3f} "
                          f"snr={snr_db:.1f}dB terr={timing_err:+d} "
                          f"peak_est={peak_est} found_peak={found_peak} "
                          f"n_read={n_read} rate={rate_est_smp:+.2f}",
                          flush=True)

                # ── Spectrograph: symbol event ────────────────────────────────
                try:
                    _telem.sendto(json.dumps({
                        "t":      "s",
                        "b":      bit,
                        "c":      round(confidence, 3),
                        "te":     timing_err,
                        # ── Memory profile (raw data bytes; no Python overhead) ──
                        # These match the allocations an embedded implementation
                        # would make: audio buffers in float32, soft-decision
                        # pairs as 4-byte floats, history arrays as float64.
                        "m_ring": ring.nbytes,               # ADC ring buffer (fixed)
                        "m_fa":   _fa_nbytes,                # frame audio accumulator
                        "m_bs":   len(bit_stream) * 8,       # soft-decision bit pairs
                        "m_misc": (len(pre_sync_buf) +       # sync candidate history
                                   len(snr_history) +        # SNR log
                                   len(rms_history)) * 8,    # RMS log
                        "nr_dev": n_read - sps_eff,          # PLL deviation (timing slip)
                        "rate_est": round(rate_est_smp, 2),  # persistent clock-drift estimate (smp/sym)
                        "sync":   sync_found,
                        "sc":     sym_count,
                        "wt":     round(time.time(), 3),     # wall clock — align vs play host via app.py offset
                        "ovf":    bool(_overflowed),         # this symbol's read overran the ADC buffer
                        "novf":   _input_overflows,          # cumulative input overruns
                    }).encode(), _TELEM_ADDR)
                except Exception:
                    pass

                # ── S/N + mic gain (throttled ~1 Hz; UI 'signal' filter) ──────
                # Symbols can be short (4+/s), so rate-limit by wall clock rather
                # than per-symbol to keep the line useful without flooding.  gain
                # is the live PeakAGC gain (the mic-gain the decoder is applying).
                _now_sn = time.time()
                if _now_sn - _last_micsn_log >= 1.0:
                    _last_micsn_log = _now_sn
                    print(f"MIC_SN sym={sym_count} bit={bit} dir={'UP' if bit else 'DN'} "
                          f"snr={snr_db:.1f}dB gain={agc.gain_db:+.1f}dB "
                          f"conf={confidence:+.2f} sync={int(sync_found)}", flush=True)

                snr_history.append(snr_db)
                # Do not update AGC during silence/noise. Without this gate,
                # winner_pk ~= noise floor, so AGC winds to agc_max during lead
                # silence, the first training chirps clip, and syndet
                # confidences in the preamble correlation window get corrupted.
                if signal_present:
                    agc.update_from_peak(winner_pk)

                # -- Acq countdown (always, independent of SNR gate) ---------
                # Decrement every symbol so the fast-acquisition window has a
                # fixed wall-clock duration. If the decrement is instead inside
                # the SNR gate, _acq_syms_left freezes during noisy symbols and
                # the aggressive alpha_acq stays active long after timing has
                # nominally settled.
                #
                # When the counter hits zero, re-engage tight tracking by setting
                # timing_settled = True.  This is required for mid-frame
                # re-acquisitions (e.g. after a large retro one-shot correction):
                # those paths set timing_settled = False to open the wide search
                # window, but the only places that restore it are do_resync()
                # (called at frame boundaries) and the pre-sync block (only
                # reached when sync_found = False).  Without this hook,
                # preamble_per_frame=False sessions permanently leave
                # timing_settled=False after the first mid-frame re-acquisition,
                # keeping max_frac at sps//4 (4× wider than post_acq_search_smp)
                # and peak_est un-pinned — re-introducing the second-order
                # oscillation the tight clip was added to prevent.
                if sync_found and _acq_syms_left > 0:
                    _acq_syms_left -= 1
                    if _acq_syms_left == 0:
                        timing_settled = True   # re-engage tight tracking

                        # ── FIX A: snap peak_est to expected_peak ──────
                        # The settled-mode search window is centred on
                        # peak_est.  The design intent (see the comment
                        # above the peak_est IIR update below) is
                        # peak_est == expected_peak in settled mode — but
                        # merely *stopping* the IIR update leaves peak_est
                        # at whatever value the 0.05-alpha tracker last
                        # reached, which may differ from expected_peak by
                        # more than post_acq_search_smp if pre-sync
                        # random-walk or an under-converged acquisition
                        # left it offset.  If |peak_est - expected_peak| >
                        # post_acq_search_smp, the settled search window
                        # permanently excludes the true chirp position —
                        # found_peak becomes a noise/edge value, terr
                        # carries a roughly-constant non-zero bias, and
                        # n_read drifts the true chirp away from
                        # expected_peak indefinitely (the original "drifts
                        # to ±1/2 symbol, can't decide up/down" symptom).
                        # An explicit snap guarantees the window is always
                        # centred exactly where terr is measured from.
                        peak_est = expected_peak

                        # ── FIX B: capture rate_est_smp from converged acq ─
                        # At alpha_acq steady state for a constant-rate
                        # disturbance D, int_corr_ss = D regardless of
                        # alpha_acq (terr_ss = D/alpha_acq, step_corr_ss =
                        # alpha_acq * terr_ss = D).  So (n_read - sps_eff)
                        # averaged over the last few acquisition symbols is
                        # a direct, alpha-independent estimate of D.  This
                        # is self-correcting across frames: if a prior
                        # estimate r was off by ε = D - r, this frame's acq
                        # converges int_corr → ε (rate_est_smp already
                        # absorbs r), so (n_read-sps_eff) → r+ε = D again —
                        # each frame's capture lands closer to true D.
                        if _acq_nread_dev:
                            rate_est_smp = float(np.mean(_acq_nread_dev))
                        # Residual phase integrator starts fresh: any
                        # accumulated timing_frac reflects the *old*
                        # rate_est_smp baseline and would otherwise be
                        # double-counted now that the new rate_est_smp is
                        # applied directly in the n_read formula below.
                        timing_frac = 0.0
                        # Clean slate for settled-mode gate.
                        # The settled outlier gate seeds its median from
                        # _terr_buf.  Any acq-phase values still in the buffer
                        # (convergence transient, or worse: impulse-corrupted
                        # measurements) would bias that median, causing the
                        # gate to accept terr values in the wrong region and
                        # reject good ones near zero.  Start fresh so the
                        # first settled-mode symbols build an uncontaminated
                        # median reference.
                        _terr_buf.clear()

                # ── Single-symbol outlier gate (settled AND acquisition) ─────
                # Median-of-3 (_terr_buf, below) discards a spike IF the other
                # two slots are clean -- but a strong echo/sidelobe that wins
                # symbol_decision can itself BECOME one of those two slots for
                # the next symbol's median, and/or _terr_buf can already be
                # carrying 1-2 elevated values from a recent transient.  In
                # either case median-of-3 doesn't fully protect step_corr.
                #
                # This gate compares THIS symbol's raw timing_err to the
                # median of the EXISTING buffer (pre-append).  If it differs
                # by more than the mode-dependent threshold, this symbol is
                # excluded from _terr_buf/step_corr entirely.
                #
                # Threshold:
                #   Settled (timing_settled, _acq_syms_left==0):
                #     post_acq_search_smp // 2  (half the search radius).
                #     An echo/sidelobe win lands in the outer half; genuine
                #     drift never reaches there between adjacent symbols.
                #
                #   Acquisition (_acq_syms_left > 0):
                #     post_acq_search_smp  (the full search radius).
                #     The fast-acquisition PLL (alpha_acq=0.25, max_step=
                #     0.10×sps) can move terr by at most 0.025×sps = 2205 smp
                #     per symbol -- well within post_acq_search_smp=3528.
                #     Impulse-induced peaks jump by 5 k-22 k smp (observed),
                #     so this threshold admits every genuine convergence step
                #     while blocking impulse spikes.  The gate is inactive for
                #     the first timing_median_n symbols after sync (buffer is
                #     empty/short) so the initial large terr can bootstrap.
                _pll_outlier = False
                if sync_found and len(_terr_buf) > 0:
                    _med_existing = sorted(_terr_buf)[len(_terr_buf) // 2]
                    _outlier_thr  = (post_acq_search_smp // 2
                                     if (timing_settled and _acq_syms_left == 0)
                                     else post_acq_search_smp)
                    if abs(timing_err - _med_existing) > _outlier_thr:
                        _pll_outlier = True
                        _acq_tag = " (acq)" if _acq_syms_left > 0 else ""
                        print(f"PLL_OUTLIER sym={sym_count} terr={timing_err:+d}smp "
                              f"median={_med_existing:+d}smp "
                              f"thr=\u00b1{_outlier_thr}smp{_acq_tag}"
                              f" -- excluded from _terr_buf/step_corr this symbol",
                              flush=True)

                # NOTE \u2014 this gate can LATCH, and clearing the latch was
                # measured and REJECTED.  A rejected symbol is never appended to
                # _terr_buf, so the gate's median reference can only be updated
                # by a symbol the reference itself accepts.  Once the true
                # timing walks more than _outlier_thr from a stale median, every
                # subsequent symbol is rejected and the median stays frozen.
                # Observed at training_symbols=15 under an 800 ppm 0.5 Hz
                # wobble: 37 consecutive rejects against median=+220 while terr
                # walked smoothly from +86 to -216.
                #
                # Clearing _terr_buf after N consecutive rejects does break the
                # latch (37 rejects -> 4) and does NOT improve delivery: over a
                # 136-cell training x wobble-phase plane it was better in 16
                # cells, worse in 17, mean 88.5% vs 88.3%.  So the frames are
                # not being lost here -- the latch is real but downstream of
                # whatever loses them.  Do not re-add the reset without a
                # measurement that separates the two.

                # ── winner_pk strength gate (settled mode only) ──────────────
                # Update the noise-floor EMA from every symbol, then decide
                # whether THIS symbol's correlation peak is strong enough to
                # trust its position for timing.  Only enforced in settled
                # mode: during acquisition every measurement is needed to
                # converge, and the EMA hasn't stabilised yet.
                if _wpk_ema is None:
                    _wpk_ema = winner_pk
                _wpk_strong = True
                if (wpk_track_enable and sync_found
                        and timing_settled and _acq_syms_left == 0):
                    _wpk_strong = winner_pk >= wpk_track_k * _wpk_ema
                    if not _wpk_strong and (sym_count in _verbose_syms):
                        print(f"WPK_GATE sym={sym_count} wpk={winner_pk:.3f} "
                              f"< {wpk_track_k:.1f}×ema({_wpk_ema:.3f}) "
                              f"-- weak symbol, coasting on rate model "
                              f"(timing not updated)", flush=True)
                # EMA update: only from weak/typical symbols so the strong
                # genuine chirps don't inflate the noise-floor baseline (which
                # would then reject the NEXT strong chirp).  A symbol already
                # flagged strong is excluded from the floor estimate.
                if not (timing_settled and _acq_syms_left == 0) or not _wpk_strong:
                    _wpk_ema += wpk_track_alpha * (winner_pk - _wpk_ema)

                if (signal_present and abs_conf >= min_tim_conf and tim_snr_ok
                        and not _pll_outlier and pll_slope_ok and _wpk_strong):
                    # Post-sync fast acquisition: use alpha_acq for the first
                    # timing_acq_symbols symbols after sync fires, then drop to
                    # the slow tracking alpha.  This corrects the initial timing
                    # error (often -4000–6000 smp at long symbol durations) within
                    # ~12 symbols (3 tau at alpha_acq=0.25) instead of ~50 symbols
                    # (5 tau at alpha=0.10), keeping MF SNR high across the frame.
                    if sync_found and _acq_syms_left > 0:
                        cur_alpha = alpha_acq
                    elif sync_found:
                        cur_alpha = alpha
                    else:
                        cur_alpha = alpha_acq

                    # Median filter on timing_err before integration.
                    # In reverberant environments the MF occasionally selects an
                    # echo peak, producing a large outlier (e.g. terr jumps from
                    # -2000 to -8449 in one step).  A 3-sample median discards the
                    # spike: median([-2059, -8449, -1530]) = -2059.  The filtered
                    # value is used for the PLL update; the raw terr is still
                    # logged so the display reflects what the MF actually measured.
                    _terr_buf.append(timing_err)
                    filtered_terr = sorted(_terr_buf)[len(_terr_buf) // 2]

                    # ── Settled-mode peak_est tracking ────────────────────────
                    # Acquisition-phase IIR runs before the outlier gate (above)
                    # because fast convergence needs the window to follow the
                    # chirp symbol-by-symbol during the initial lock-on sprint.
                    # Settled-mode tracking belongs HERE — after the outlier gate
                    # and inside the PLL block — because _pll_outlier is already
                    # known False at this point, so we're guaranteed this is a
                    # measurement we're actually trusting, not a noise spike.
                    # This keeps peak_est centred on the chirp even when rate_est
                    # error causes systematic drift that n_read alone can't fully
                    # correct within one search-window width.
                    if timing_settled and _acq_syms_left == 0:
                        peak_est = int(round(
                            (1.0 - _track_alpha) * peak_est +
                             _track_alpha         * (expected_peak + clamped_err)
                        ))

                    # Clamp the per-step PLL contribution to ±timing_max_step_frac
                    # of sps.  This keeps n_read within [0.9×sps, 1.1×sps] so
                    # the audio buffer stays ≥ 90% full even when terr is large.
                    # The timing still converges — it just takes proportionally
                    # more symbols.  The retro decode handles those extra symbols
                    # correctly without relying on PLL accuracy.
                    max_step  = sps * timing_max_step_frac
                    step_corr = float(np.clip(cur_alpha * filtered_terr,
                                              -max_step, max_step))
                    timing_frac += step_corr
                    # timing_frac represents deviation from sps_eff (the
                    # effective symbol period including the inter-symbol gap).
                    # In settled tracking the clip is post_acq_search_smp —
                    # tight enough to block echo spikes (previous fix) while
                    # still covering normal clock drift.  When a non-zero gap
                    # is in use, sps_eff > sps so the already-initialised
                    # timing_frac=0 is correct (n_read = sps_eff + 0 = sps_eff)
                    # and the clip remains valid.
                    max_frac = (post_acq_search_smp
                                if (timing_settled and sync_found and _acq_syms_left == 0)
                                else sps // 4)
                    timing_frac  = float(np.clip(timing_frac, -max_frac, max_frac))
                    int_corr     = int(timing_frac)
                    timing_frac -= int_corr
                    # rate_est_smp (samples/symbol clock-drift estimate,
                    # captured at the acq→settled transition above) is
                    # applied as a constant baseline, on top of int_corr's
                    # residual-phase correction.  See the rate_est_smp
                    # comment block in the init section.
                    _rate_int = int(round(rate_est_smp))
                    n_read = max(sps // 2, sps_eff + _rate_int + int_corr)
                    # Continuous rate refinement (settled mode only): if the
                    # one-shot acquisition estimate was off, int_corr has to
                    # keep compensating for the same error every symbol
                    # instead of averaging toward zero.  Slowly folding that
                    # systematic component into rate_est_smp itself reduces
                    # how hard timing_frac/int_corr have to work each symbol
                    # — the PLL converges toward not needing per-symbol
                    # correction at all, rather than perpetually re-deriving
                    # the same correction from scratch.  Not applied during
                    # acquisition: alpha_acq's int_corr there is dominated by
                    # genuine fast convergence, not steady-state bias, and
                    # _acq_nread_dev/the one-shot capture already handle that
                    # phase specifically.
                    if (sync_found and timing_settled and _acq_syms_left == 0
                            and rate_refine_alpha > 0):
                        rate_est_smp += rate_refine_alpha * int_corr
                    # Feed the acquisition-tail capture window (see the
                    # _acq_syms_left == 0 block above).  Gated on the
                    # POST-decrement value: during acq symbols 1..K-1 this
                    # is >0 and we record (n_read - sps_eff) toward the
                    # eventual rate_est_smp capture; on the transition
                    # symbol itself (==0, n_read already reflects the
                    # freshly-captured rate_est_smp) we don't re-record.
                    if sync_found and _acq_syms_left > 0:
                        _acq_nread_dev.append(n_read - sps_eff)
                    # ── Mid-frame timing snapshot ─────────────────────────────
                    # Capture a clean PLL state from the settled body of the
                    # frame.  At the frame boundary, end-of-frame noise may have
                    # driven timing_frac/n_read away from truth; this snapshot
                    # (taken while the PLL is converged) provides a fallback.
                    # Conditions: sync running, loop settled (acq done), timing
                    # error small (< half the post-acquisition search radius),
                    # and high symbol quality — so we only save genuinely stable
                    # measurements, not noisy end-of-frame glitches.
                    if (sync_found
                            and _acq_syms_left == 0
                            and timing_settled
                            and abs(timing_err) < post_acq_search_smp // 2
                            and abs_conf > 0.55
                            and snr_db > timing_min_snr_db + 3.0):
                        _timing_snapshot = (timing_frac, n_read, peak_est)
                else:
                    # No valid timing measurement: drift n_read back toward
                    # the rate-aided baseline sps_eff + rate_est_smp, not
                    # bare sps_eff.  Reverting to bare sps_eff during a
                    # below-noise-floor symbol would erode rate_est_smp's
                    # correction by nread_revert * rate_est_smp on every
                    # such symbol — a slow leak that re-introduces drift
                    # specifically during fades, the scenario where timing
                    # accuracy matters most.
                    _rate_target = sps_eff + int(round(rate_est_smp))
                    n_read = int(round(n_read + nread_revert * (_rate_target - n_read)))

                if time.time() - last_status_t > snr_log_ivl:
                    avg_snr = float(np.mean(snr_history[-32:])) if snr_history else 0
                    fw = abs_conf < min_tim_conf
                    print(f"STATUS "
                          f"gain={agc.gain_db:+.1f}dB "
                          f"level={agc.level_dbfs:.1f}dBFS "
                          f"snr={avg_snr:.1f}dB "
                          f"conf={confidence:+.2f} "
                          f"terr={timing_err:+d}smp "
                          f"rate={rate_est_smp:+.2f} "
                          f"n_read={n_read} "
                          f"fw={'yes' if fw else 'no'} "
                          f"sync={'YES' if sync_found else 'no'}"
                          f"{f' acq={_acq_syms_left}' if _acq_syms_left > 0 else ''}",
                          flush=True)
                    last_status_t = time.time()

                # ── IF-slope (PLL) confidence blend — pre- AND post-sync ──────
                # pll_conf/pll_r2 were already computed earlier, right after
                # symbol_decision(), so the SAME slope measurement gates
                # the PLL trust decision above AND blends the bit confidence
                # here — one measurement, two consistent uses, rather than
                # computing if_slope_confidence() twice per symbol.
                # The IF-slope discriminator rejects narrowband tones (birds,
                # HVAC harmonics) that generate high MF confidence for the
                # wrong chirp direction but have near-zero instantaneous
                # frequency sweep.
                final_conf = confidence
                if pll_weight > 0 and signal_present and pll_r2 > 0.10:
                    w = pll_weight * pll_r2
                    final_conf = (1 - w) * confidence + w * pll_conf

                # ── Noise-floor erasure: zero confidence when SNR is below ──
                # the timing gate threshold.  At noise floor the MF ratio
                # (up_pk - dn_pk) / (up_pk + dn_pk) can be 0.1-0.3 from pure
                # statistical fluctuation — enough to escape the era_thr check
                # and reach the Viterbi / RS decoder as a seemingly-confident
                # but wrong bit.  Zeroing here ensures:
                #   • RS erasure marking fires (0.0 < era_thr=0.25).
                #   • Viterbi branch metric = 0 ('no information') rather than
                #     a small wrong-direction push.
                #   • The pre-sync syndet window is not polluted by fade symbols.
                if not tim_snr_ok:
                    final_conf = 0.0

                if not sync_found:
                    tim_flush_thr = int(cfg.get("syndet_flush_terr", sps // 2))
                    if abs(timing_err) <= tim_flush_thr:
                        timing_settled = True
                    # Store IF-slope-blended confidence so retrospective
                    # bit_stream reconstruction also benefits from the blend.
                    pre_sync_buf.append((1 if final_conf > 0 else 0, final_conf))

                    # SNR-weighted push: ramp weight from 0 at syndet_min_snr_db
                    # to 1 at syndet_snr_ref_db, so deep-fade symbols (random
                    # +-confidence) don't pollute the Barker cross-correlation
                    # window.
                    snr_w = float(np.clip(
                        (snr_db - syndet_min_snr_db)
                        / max(syndet_snr_ref_db - syndet_min_snr_db, 1.0),
                        0.0, 1.0))
                    _, score = syndet.push(final_conf * snr_w)

                    # Adaptive effective threshold: rises syndet_margin above
                    # the background EMA in bursty noise, clamped to
                    # [syndet_thr_floor, sync_threshold]. The floor prevents a
                    # collapse to noise-floor levels in clean conditions, which
                    # causes premature false syncs at wrong timing offsets.
                    effective_thr = min(sync_threshold,
                                        max(syndet_thr_floor,
                                            _syndet_bg_ema + syndet_margin))
                    acquired = score >= effective_thr

                    # Lag guard: reject stale acquisitions triggered by chance
                    # correlations deep in the history ring.
                    if acquired and syndet_lag_guard:
                        if syndet.best_lag > syndet_max_lag:
                            print(f"SYNC_STALE_REJECT "
                                  f"lag={syndet.best_lag} > max={syndet_max_lag} "
                                  f"score={score:.3f}", flush=True)
                            acquired = False

                    # Early-arrival guard (always active, not parameterised):
                    # reject sync if the matched preamble window overlaps the
                    # training region.  With longer symbol_dur, each training
                    # chirp produces near-perfect MF confidence (|conf|≈0.9)
                    # because the time-bandwidth product is larger.  The
                    # alternating up/down training pattern can then hit a
                    # Barker-13 sidelobe and score above the threshold even
                    # before any preamble has arrived — causing a false sync
                    # that misaligns the frame by training_symbols symbols.
                    #
                    # The preamble window's LAST symbol is sym_count - best_lag.
                    # For a valid acquisition that symbol must land at least
                    # _min_preamble_end symbols AFTER the last resync anchor — far
                    # enough that it cannot be a false lock on the alternating
                    # training pattern, AND late enough that the FULL Barker has
                    # arrived (a partial preamble peaks early and mis-aligns the
                    # frame).  Using the offset from _last_resync_sym (not absolute
                    # sym_count) keeps the guard meaningful for every frame.
                    #
                    # Soft-rearm bypass: on a soft re-arm a real preamble is
                    # imminent, and timing drift across the inter-frame gap can
                    # plant _last_resync_sym a few symbols late, making a genuine
                    # full-Barker detection read as "too early" (observed
                    # window_end_rel=25 vs min=28).  The strict positional guard
                    # then rejects it forever — it ages past syndet_max_lag into a
                    # STALE_REJECT — which is the deadlock that stalled a clean
                    # 40 dB-SNR second half.  A score at/above sync_threshold is
                    # unambiguous proof this is the real preamble (training
                    # sidelobes and partial preambles sit near the ~0.4 spurious-
                    # correlation floor, well below 0.55), so trust it and skip the
                    # positional check.  Cold start / hard resync keep the strict
                    # guard: no preamble is legitimately imminent there, so a
                    # high-scoring "early" hit could only be a training sidelobe.
                    # The positional test above is a PROXY: it assumes
                    # _last_resync_sym sits at the frame boundary, which holds
                    # for a clean soft re-arm and fails for an error resync
                    # (planted mid-stream) or a drifting clock (the symbol the
                    # anchor lands on moves).  Measured under a 0.5 Hz clock
                    # wobble, real 13/13 preambles were reported 2-8 symbols
                    # "early" and rejected -- at every training_symbols value
                    # including the shipped default, which is why the resulting
                    # failures looked like a property of training length rather
                    # than of this guard.
                    #
                    # So ask the question the guard actually cares about, and
                    # ask it of the DATA rather than of the position: is this
                    # match made of the training pattern?  Training is strictly
                    # alternating up/down, so its confidences change sign on
                    # every step (alternation 1.0); Barker-13 has runs and
                    # reads ~0.58.  A window that is not alternating is not a
                    # training sidelobe, wherever the stream thinks it is, so
                    # it is trusted regardless of the anchor.
                    #
                    # The old score-based bypass is kept for the case where the
                    # window IS alternating but a soft re-arm makes a real
                    # preamble imminent anyway.
                    _min_preamble_end = _training_symbols + _n_preamble - 1
                    _window_end_rel   = (sym_count - syndet.best_lag) - _last_resync_sym
                    _alt = syndet.alternation()
                    _looks_training = (_alt is not None
                                       and _alt >= syndet_alt_thr)
                    # The bypass stays INSIDE _last_resync_soft, as it always
                    # did.  A first attempt at this fix let any non-alternating
                    # window through regardless, and cold acquisition fell from
                    # 100% to 50-83% for the long-training candidates: the guard
                    # rejects two different things, and only one of them is the
                    # training sidelobe.  The other is a PARTIAL preamble, whose
                    # window straddles training and a half-arrived Barker --
                    # NOT alternating, so the data test happily trusts it, and
                    # the frame is then read from the wrong offset.  More
                    # training makes that straddle more likely, which is why
                    # the damage landed on exactly the longest run-ins.
                    #
                    # At cold start the anchor is reliable (it is 0, so
                    # window_end_rel is an absolute position) and the positional
                    # test is sound -- so leave it with teeth there.  Only a
                    # soft re-arm has the displaced-anchor problem this fix is
                    # for, and only there is the bypass widened.
                    _trust_early = (_last_resync_soft
                                    and (score >= sync_threshold
                                         or not _looks_training))
                    if (acquired and _window_end_rel < _min_preamble_end
                            and not _trust_early):
                        print(f"SYNC_TOO_EARLY "
                              f"window_end={sym_count - syndet.best_lag} "
                              f"(+{_window_end_rel} since resync) "
                              f"min={_min_preamble_end} "
                              f"alt={_alt if _alt is None else round(_alt, 3)} "
                              f"(training>={syndet_alt_thr:.2f}) "
                              f"score={score:.3f}", flush=True)
                        acquired = False

                    # Update background EMA only when no sync is declared.  Clamp
                    # at >= 0: 'score' is a signed Barker correlation, so a run of
                    # negative-confidence (down-chirp) symbols would otherwise pull
                    # the background level negative, which is meaningless for a
                    # threshold and only serves to depress eff_thr.
                    if not acquired:
                        _syndet_bg_ema += syndet_bg_alpha * (score - _syndet_bg_ema)
                        if _syndet_bg_ema < 0.0:
                            _syndet_bg_ema = 0.0

                    if acquired:
                        lag = syndet.best_lag
                        print(f"SYNC_ACQUIRED sym={sym_count} score={score:.3f} "
                              f"(eff_thr={effective_thr:.2f} "
                              f"bg={_syndet_bg_ema:.3f}) "
                              f"snr={snr_db:.1f}dB terr={timing_err:+d}"
                              f"{f' lag={lag}' if lag else ''}", flush=True)
                        sync_found            = True
                        _acq_syms_left        = timing_acq_symbols
                        # Open wide search during acquisition.
                        # The pre-sync scanner (line below the syndet block) sets
                        # timing_settled = True whenever abs(timing_err) is small,
                        # so timing_settled is nearly always True by the time the
                        # first SYNC_ACQUIRED fires.  Leaving it True during
                        # acquisition causes two problems:
                        #   1. eff_max_terr = max_terr_smp (≈sps//2) instead of
                        #      the narrow presync_max_terr_smp, so a bad MF
                        #      measurement at the acq→settled transition symbol
                        #      enters _terr_buf un-clamped (e.g. −20673 smp).
                        #   2. The max_frac clip on timing_frac uses
                        #      sps//4 when settled=False, giving an appropriate
                        #      wide correction range during convergence.
                        # Reset to False here; the countdown block (above) sets
                        # it back to True at the acq→settled boundary.
                        timing_settled        = False
                        # ── Snap peak_est to the detected chirp position ──────
                        # The sync detector reports timing_err = how far the
                        # actual chirp is from expected_peak at the moment of
                        # detection.  When this is large (the preamble was
                        # found but the chirp sits far from the nominal ring
                        # position — e.g. a long capture-buffer offset), the
                        # acquisition PLL would have to WALK peak_est there at
                        # ~alpha_acq×search per symbol (~2205 smp/sym max),
                        # taking many symbols.  That only works if those
                        # symbols have strong, trackable chirps — but with a
                        # sparse payload (long identical-symbol runs) the
                        # intervening symbols are noise and the walk derails,
                        # leaving peak_est stuck ~15000 smp from the real
                        # chirp for the whole frame (observed: SYNC terr=
                        # -15386, peak_est frozen ~248255 vs expected 260189,
                        # every payload symbol then a noise argmax).  Instead,
                        # snap peak_est directly to the detected position so
                        # the very first payload symbol is already centred.
                        if abs(timing_err) > post_acq_search_smp:
                            _snapped = int(np.clip(
                                expected_peak + timing_err,
                                0, buf_size - 1))
                            print(f"SYNC_SNAP peak_est {peak_est}→{_snapped} "
                                  f"(terr={timing_err:+d} exceeds "
                                  f"±{post_acq_search_smp}; centring search "
                                  f"window on detected chirp)", flush=True)
                            peak_est = _snapped
                        # Clean slate for acquisition gate.
                        # _terr_buf may hold pre-sync timing measurements taken
                        # while peak_est was IIR-scanning at wide range.  Those
                        # values are unrelated to the post-sync chirp position,
                        # so the outlier gate's median would be seeded with
                        # noise.  _acq_nread_dev similarly must not carry over
                        # fragments of a previous acquisition.
                        _terr_buf.clear()
                        _acq_nread_dev.clear()
                        if lag > 0:
                            avail      = len(pre_sync_buf)
                            take       = min(lag, avail)
                            bit_stream = list(pre_sync_buf)[-take:] if take else []
                            if take < lag:
                                print(f"  [sync] lag={lag} but only {take} "
                                      f"pre-sync symbols available", flush=True)
                        else:
                            bit_stream = []
                        diag["sync_acquired"] = True
                        diag["sync_at_symbol"]= sym_count - lag
                        diag["sync_score"]    = round(score, 3)
                        diag["sync_lag"]      = lag
                        # ── Spectrograph: sync event ──────────────────────────
                        try:
                            _telem.sendto(json.dumps({
                                "t":   "sync",
                                "sc":  round(score, 3),
                                "f0":  f_min, "f1": f_max, "sd": dur,
                                # Modem config — lets the viewer annotate axis limits
                                "sps":  sps,  "sps_eff": sps_eff,
                                "fr":   frame_symbols,
                            }).encode(), _TELEM_ADDR)
                        except Exception:
                            pass
                    elif score > 0.45:
                        # Only log candidates well above the spurious-data-correlation
                        # floor (~0.40 for high-confidence payload hitting a Barker
                        # sidelobe).  Printing every score > 0.20 generates ~40 lines
                        # of noise before the real preamble arrives.
                        print(f"SYNC_CANDIDATE sym={sym_count} score={score:.3f} "
                              f"(eff_thr={effective_thr:.2f} "
                              f"bg={_syndet_bg_ema:.3f}) "
                              f"snr={snr_db:.1f}dB conf={confidence:+.2f}", flush=True)
                else:
                    final_bit  = 1 if final_conf > 0 else 0
                    bit_stream.append((final_bit, final_conf))

                    if len(bit_stream) >= frame_symbols:
                        frame_items = bit_stream[:frame_symbols]
                        bit_stream  = bit_stream[frame_symbols:]

                        stream_bits  = [b for b, _ in frame_items]
                        stream_confs = [c for _, c in frame_items]

                        frame_snr = (float(np.mean(snr_history[-frame_symbols:]))
                                     if len(snr_history) >= frame_symbols else snr_db)

                        # ── Primary path: retrospective two-pass timing decode ─────
                        #
                        # The streaming PLL is a running estimate updated one symbol
                        # at a time.  Any timing error at sync, any noise spike, any
                        # echo-induced peak switch corrupts every subsequent bit
                        # decision because the error is never revisited.
                        #
                        # The retrospective pass treats the frame like a QR code: the
                        # chirp shape is known in advance, so we run the MF over the
                        # entire buffered audio at once, find all N peaks jointly, fit
                        # a two-parameter timing line (t0, sps_effective) with outlier
                        # rejection, and read each bit at the globally-fitted position.
                        # A single bad symbol (noise, echo) is one outlier among N —
                        # the IRLS fit discards it rather than propagating the error.
                        #
                        # The streaming bits are kept as a fallback for the rare case
                        # where insufficient audio was buffered (e.g. lag > 0 at sync).
                        frame_bits  = stream_bits
                        frame_confs = stream_confs
                        used_retro  = False

                        if _frame_audio:
                            # ── Retro audio size cap ──────────────────────────
                            # retro_timing_decode computes nfft = next_power_of_2(
                            #   N_samples + sps - 1).  At 2 s/sym with
                            #   frame_symbols=74, N = 6.5 M → nfft = 8 M.
                            # On a typical CPU each rfft of 8 M takes 1-3 s; four
                            # calls = 4-12 s of blocking on the main thread.
                            # During that time the PortAudio input FIFO overflows
                            # (default latency 100-500 ms), losing 2-4 symbols.
                            # When stream.read() resumes, the ring is misaligned by
                            # those symbol periods — exactly why 2 s symbols fail
                            # while 1 s symbols (4 M FFT, ~2 s compute, fits in
                            # the PortAudio latency buffer) succeed.
                            #
                            # Fix: limit the retro window so nfft ≤ 4 M
                            # (compute ≈ 0.5-1 s, safely under any device latency).
                            # The last MAX_RETRO_SYMS chunks are used for the retro;
                            # streaming bits cover the earlier symbols whose timing
                            # is well-settled by the time the retro runs anyway.
                            # ── max_retro_samples: config-driven auto-scale ──
                            # 0 (default) → auto: grow the window so the IRLS
                            # fit always has at least _MIN_RETRO_SYMS symbol
                            # peaks to work with, while keeping the 4 M-sample
                            # floor that guards against PortAudio FIFO overflow
                            # on short symbols.
                            # For long symbols (e.g. 5 s at 44100 Hz = 220500
                            # sps) the old hardcoded 4 M cap gave only 18 peaks
                            # — right at the edge of IRLS stability.  Auto gives
                            # 24 peaks (5.3 M samples) at the cost of a slightly
                            # longer FFT (≈1 s extra blocking); for 10 s symbols
                            # it gives 24 peaks / 10.6 M samples / ≈2 s blocking.
                            # If that is too slow for the hardware, set
                            # max_retro_samples to a fixed byte count (e.g.
                            # 4000000 restores the original behaviour).
                            # _max_retro_smp is hoisted to decode() init.  Each
                            # _frame_audio chunk is n_read ≈ sps_eff samples, so
                            # the window cap must be in units of sps_eff.
                            max_retro_syms = max(4, _max_retro_smp // sps_eff)
                            n_retro_syms      = min(len(_frame_audio), max_retro_syms)
                            retro_chunks      = _frame_audio[-n_retro_syms:]
                            retro_audio       = np.concatenate(retro_chunks)

                            # ── PRIMARY: CRC-anchored offset search ───────────
                            # The streaming sync can land on the wrong symbol
                            # boundary at cold start (first frame, AGC/background
                            # still settling), so retro over _frame_audio reads a
                            # SHIFTED set of chirps — a clean fit over the WRONG
                            # symbols → CRC fail.  Re-derive the boundary
                            # empirically: over a wider history guaranteed to
                            # contain the whole frame even if acquisition fired
                            # late, try whole-symbol shifts of the frame window
                            # and accept the first whose CRC checks out.  The
                            # envelope FFT is computed once inside the candidate
                            # generator, so the extra offsets are cheap.
                            #
                            # Compute guard: the anchor buffer is wider than the
                            # capped retro window, so for long symbols its FFT
                            # could exceed device latency and overflow the input
                            # FIFO.  Only attempt anchoring within the same
                            # sample budget the retro cap uses.
                            retro = None
                            _anchor_ok = (_anchor_enabled
                                          and len(_audio_hist) >= frame_symbols + 2)
                            if _anchor_ok:
                                anchor_audio = np.concatenate(list(_audio_hist))
                                try:
                                    for _cand in retro_offset_candidates(
                                            anchor_audio, up_ref, down_ref, sps,
                                            sps_eff, frame_symbols,
                                            retro_anchor_max_shift,
                                            rate_est_smp=rate_est_smp):
                                        c_ok, _cseq, _cdat, _, _ = _run_pipeline(
                                            _cand[0], _cand[1], frame_snr)
                                        if c_ok:
                                            retro        = _cand[:5]
                                            retro_audio  = anchor_audio
                                            n_retro_syms = frame_symbols
                                            print(f"RETRO_ANCHOR locked frame at "
                                                  f"shift={_cand[5]:+d} sym (CRC ok)",
                                                  flush=True)
                                            break
                                except Exception as _anc_exc:
                                    print(f"RETRO_ANCHOR error: {_anc_exc}",
                                          flush=True)
                                    retro = None
                                if retro is None:
                                    print("RETRO_ANCHOR: no CRC-valid shift — "
                                          "falling back to streaming-aligned retro",
                                          flush=True)

                            if retro is None:
                                retro = retro_timing_decode(
                                    retro_audio, up_ref, down_ref, sps, n_retro_syms,
                                    sps_eff=sps_eff, rate_est_smp=rate_est_smp)
                            if retro is not None:
                                retro_bits, retro_confs, retro_t0, retro_sps, retro_n_outliers = retro
                                # Prefix from streaming (earlier, settled symbols);
                                # suffix from retro (most recent n_retro_syms).
                                n_prefix    = frame_symbols - n_retro_syms
                                frame_bits  = stream_bits[:n_prefix]  + retro_bits
                                frame_confs = stream_confs[:n_prefix] + retro_confs
                                used_retro  = True

                                # ── Retro inter-frame timing resync ───────
                                #
                                # retro_t0, retro_sps describe the fit:
                                #   peak[k] = retro_t0 + k * retro_sps  (in retro_audio coords)
                                #
                                # retro_audio[-1] == ring[buf_size-1] (both are the
                                # last sample of the current AGC block), so a position
                                # p in retro_audio maps to ring position:
                                #   r = buf_size - N_retro + p
                                #
                                # For the last retro symbol (k = n_retro_syms-1):
                                #   r_last = buf_size - N_retro + retro_t0
                                #            + (n_retro_syms-1) * retro_sps
                                #
                                # The phase error (how far that peak sits from
                                # expected_peak in the ring buffer):
                                #   delta = r_last - expected_peak
                                #         = p_last - (N_retro - gap_samps - 1)
                                #
                                # Verify ideal case: t0=sps-1, sps_eff=retro_sps,
                                # N=n*sps_eff → delta = (sps-1+(n-1)*sps_eff)
                                #               - (n*sps_eff-gap_samps-1) = 0 ✓
                                #
                                # The slip the streaming PLL cannot self-correct
                                # (|delta| > post_acq_search_smp) comes from
                                # noise at end-of-frame driving int_corr to the
                                # ±max_frac clip 2+ times in a row.  The retro
                                # fit uses all N inlier peaks jointly (IRLS) so
                                # a few bad end-of-frame symbols are outliers, not
                                # contaminants.
                                N_retro   = len(retro_audio)
                                p_last    = retro_t0 + (n_retro_syms - 1) * retro_sps
                                delta     = p_last - (N_retro - gap_samps - 1)
                                sps_drift = retro_sps - sps_eff

                                # ── Phase correction (one-shot) ───────────────
                                # Applied to a single ring advance then zeroed,
                                # so nread_revert cannot repeatedly apply it and
                                # drive the chirp half-symbol off on symbol 2.
                                #
                                # ── Sanity gate: two independent checks ───────
                                #
                                # Check A — inlier fraction.
                                # If the IRLS discarded more than half the peaks
                                # as outliers, the fit is unreliable regardless
                                # of what delta says.  This fires when the PLL
                                # was already partially lost going in, so most
                                # peaks were from noise or sidelobes.  Now that
                                # n_outliers is returned we can check directly
                                # rather than relying on delta magnitude alone.
                                #
                                # Check B — delta magnitude.
                                # Hardware clock drift is ≤ ~200 ppm, so the true
                                # phase error over any frame is well within
                                # 2 × post_acq_search_smp.  A larger |delta|
                                # means the fit converged to a wrong solution.
                                #
                                # Either check failing: skip the one-shot and
                                # re-enter acquisition so the wide search window
                                # can re-find the chirp without a mis-correction
                                # first shoving it out of range.
                                _retro_n_inliers  = n_retro_syms - retro_n_outliers
                                _retro_inlier_frac = (_retro_n_inliers /
                                                      max(1, n_retro_syms))
                                _max_sane_delta   = post_acq_search_smp * 2
                                _gate_reason      = None
                                if _retro_inlier_frac < 0.5:
                                    _gate_reason = (
                                        f"inliers={_retro_n_inliers}/{n_retro_syms} "
                                        f"({_retro_inlier_frac:.0%}) < 50%")
                                elif abs(delta) > _max_sane_delta:
                                    _gate_reason = (
                                        f"delta={delta:+.1f} > "
                                        f"2×post_acq_smp ({_max_sane_delta})")
                                if _gate_reason is not None:
                                    print(f"RETRO_SYNC gate [{_gate_reason}] — "
                                          f"skipping one-shot; re-entering acq",
                                          flush=True)
                                    _one_shot_correction = 0
                                    _acq_syms_left = timing_acq_symbols
                                    timing_settled = False
                                    # Stale clipped errors (e.g. [-8820,-8820,-8820])
                                    # would make the median filter push the PLL
                                    # in the wrong direction for the first 2-3 symbols.
                                    _terr_buf.clear()
                                    # Discard stale rate estimate.
                                    # The gate fired because the retro fit was
                                    # unreliable (inlier_frac < 50% or |delta| too
                                    # large), meaning the current rate_est_smp was
                                    # captured from a corrupted acquisition.  Running
                                    # the new acquisition with a bad rate baseline
                                    # offsets n_read by ~rate_est_smp every symbol,
                                    # systematically drifting the ring away from the
                                    # chirp and corrupting the next rate capture too
                                    # (observed: rate_est=-1549 after first failure
                                    # → delta_frame=-22011 on second acquisition,
                                    # 60 sym × 1549 smp/sym of accumulated drift).
                                    rate_est_smp = 0.0
                                    _acq_nread_dev.clear()
                                else:
                                    _one_shot_correction = int(round(
                                        float(np.clip(delta,
                                                      -(sps // 2), sps // 2))))
                                    # Even a sane correction may exceed the settled
                                    # search window radius (post_acq_search_smp).
                                    # If it does, re-enter fast acquisition so
                                    # the wider alpha_acq search can re-lock within
                                    # ~12 symbols instead of ~2500.
                                    if abs(_one_shot_correction) > post_acq_search_smp:
                                        _acq_syms_left = timing_acq_symbols
                                        timing_settled = False
                                        _terr_buf.clear()   # discard stale clipped errors
                                        print(f"RETRO_SYNC large one-shot "
                                              f"({_one_shot_correction:+d} smp > "
                                              f"±{post_acq_search_smp}) → "
                                              f"re-entering acquisition",
                                              flush=True)

                                # ── Adopt retro drift as rate baseline ──────
                                # OFF BY DEFAULT since 2026-09-03.  Read the
                                # evidence below before turning it back on.
                                #
                                # The original rationale: the retro fit measures
                                # sps_drift jointly over ALL frame symbols via
                                # IRLS — far more data than rate_est_smp, which
                                # is captured from a 5-sample acquisition deque
                                # during the brief lock-on window.  The one-shot
                                # correction re-centres the chirp ONCE, but if
                                # rate_est_smp disagrees with the true drift by
                                # ~150 smp/sym the chirp re-escapes the
                                # ±post_acq window within ~20 symbols and the
                                # loop loses lock again.  Observed at the time:
                                # acq rate_est=-487 vs retro drift_fit=-672,
                                # "exactly the long-symbol tracking failure".
                                #
                                # That regime NO LONGER REACHES THIS CODE.  Long
                                # symbols now dispatch to decode_longsym_mf
                                # (symbol_dur > mf_longsym_min_dur = 0.25), which
                                # emits no RETRO_SYNC at all — verified: a 0.5 s
                                # decode logs zero "adopting drift_fit" lines.
                                # What is left reachable is the PLL path, and
                                # there the adoption is measurably not earning
                                # its keep:
                                #
                                #   constant drift, 36 cells (symbol_dur
                                #   0.12-0.8 x 0-4000 ppm, and 0.12-0.24 x
                                #   4000-40000 ppm): delivery IDENTICAL with and
                                #   without, to the cell.  The 4000->10000 ppm
                                #   cliff sits in exactly the same place.
                                #
                                #   varying clock, 136-cell training_symbols x
                                #   wobble_phase plane at 800 ppm / 0.5 Hz:
                                #   adoption is HARMFUL — mean 87.2% with vs
                                #   90.4% without, failing cells 40 vs 28,
                                #   better in 32 cells and worse in 9.
                                #
                                # The reason it hurts is structural.  The fit is
                                # a frame MEAN extrapolated to the next frame,
                                # and the inlier fraction — its only gate — asks
                                # whether the fit describes the frame it was
                                # measured over, never whether that mean still
                                # applies to the next one.  For a constant
                                # offset those are the same question.  For a
                                # clock that varies within a frame they are not:
                                # a wobble averages toward zero over the frame,
                                # the residuals stay linear enough that inliers
                                # reads 100%, and the adopted number describes
                                # no instant in particular.  Observed:
                                # drift_fit=-0.359 (inliers=100%) adopted, then
                                # the next frame's real preamble scored 0.450
                                # against eff_thr=0.45 and the frame CRC-failed.
                                #
                                # NOTE for anyone tempted by a smarter estimator
                                # (endpoint slope, half-frame slopes, quadratic
                                # fit): at 0.5 Hz the measurement-to-use delay is
                                # a frame plus a run-in, ~15 s, which is SEVERAL
                                # wobble periods.  No estimator from the previous
                                # frame can be valid there — the question is
                                # whether to adopt at all, not what to adopt.
                                # A half-frame slope test in particular is blind
                                # here: the payload spans ~6.6 wobble cycles, so
                                # both halves average over ~3.3 cycles, both read
                                # near zero, and they agree.
                                #
                                # The inlier fraction is the ONLY gate on this,
                                # and it asks whether the fit describes the
                                # frame it was measured over -- not whether that
                                # frame's mean rate still applies to the NEXT
                                # one.  For a constant clock offset those are
                                # the same question.  For a clock that varies
                                # within a frame they are not: the fit is a
                                # frame MEAN, a sinusoidal wobble averages
                                # toward zero over it, and the residuals stay
                                # linear enough that inliers reads 100% while
                                # the number adopted describes no instant in
                                # particular.  Exposed as a config key so that
                                # can be measured rather than argued about:
                                # the default 1.01 disables adoption, 0.80
                                # restores the pre-2026-09-03 behaviour.
                                if (_gate_reason is None
                                        and _retro_inlier_frac >= retro_rate_adopt_inlier):
                                    rate_est_smp = float(sps_drift)
                                    print(f"RETRO_SYNC adopting drift_fit as "
                                          f"rate baseline: rate_est "
                                          f"{rate_est_smp:+.2f} smp/sym "
                                          f"(inliers={_retro_inlier_frac:.0%})",
                                          flush=True)

                                # ── Frequency cross-check ─────────────────────
                                print(f"RETRO_SYNC "
                                      f"one_shot={_one_shot_correction:+d} "
                                      f"drift_fit={sps_drift:+.3f} "
                                      f"rate_est={rate_est_smp:+.3f} "
                                      f"n_read={n_read}",
                                      flush=True)
                                _timing_snapshot = None   # retro supersedes snapshot
                            else:
                                print("RETRO insufficient audio — using streaming bits",
                                      flush=True)
                                # ── Snapshot fallback ─────────────────────────
                                # No retro timing available; restore the last
                                # known-good mid-frame PLL state so end-of-frame
                                # noise does not carry over as a phase error.
                                if _timing_snapshot is not None:
                                    _tf, _nr, _pe = _timing_snapshot
                                    timing_frac = _tf
                                    n_read      = _nr
                                    if not timing_settled:
                                        peak_est = _pe
                                    print(f"RETRO_SYNC snapshot fallback "
                                          f"n_read={n_read}", flush=True)
                                    _timing_snapshot = None
                        else:
                            print("RETRO no audio buffered — using streaming bits",
                                  flush=True)
                            # ── Snapshot fallback ─────────────────────────────
                            if _timing_snapshot is not None:
                                _tf, _nr, _pe = _timing_snapshot
                                timing_frac = _tf
                                n_read      = _nr
                                if not timing_settled:
                                    peak_est = _pe
                                print(f"RETRO_SYNC snapshot fallback (no audio) "
                                      f"n_read={n_read}", flush=True)
                                _timing_snapshot = None

                        # Clear frame audio now — prevents unbounded growth across
                        # consecutive frames when preamble_per_frame=False (each
                        # subsequent retro would otherwise re-FFT all prior frames).
                        _fa_peak_bytes = _fa_nbytes   # save peak before dropping
                        _fa_nbytes     = 0            # reset running counter
                        _frame_audio.clear()

                        ok, seq_num, data, rs_errors, rs_fail_str = _run_pipeline(
                            frame_bits, frame_confs, frame_snr)

                        # ── Fallback: retry with streaming bits if retro failed ────
                        if not ok and used_retro:
                            print("RETRO CRC failed — retrying with streaming bits",
                                  flush=True)
                            ok2, seq2, dat2, rs2, rf2 = _run_pipeline(
                                stream_bits, stream_confs, frame_snr)
                            if ok2:
                                print("STREAM fallback succeeded", flush=True)
                                ok, seq_num, data, rs_errors, rs_fail_str = (
                                    ok2, seq2, dat2, rs2, rf2)

                        # ── Outcome handling ──────────────────────────────────────
                        if rs_fail_str is not None and not ok:
                            frames_errored += 1
                            consec_errors  += 1
                            if preamble_per_frame:
                                do_resync("preamble_per_frame: RS fail, re-arming",
                                          reset_timing=False)
                            elif consec_errors >= resync_after:
                                do_resync(f"{consec_errors} RS failures",
                                          reset_timing=True)
                            continue

                        if ok:
                            # CRC already validated integrity.  Payloads may be
                            # arbitrary BINARY (modem packets), so keep raw bytes
                            # as bytes_hex and DON'T gate on UTF-8 (text is a
                            # best-effort view).  NB: at_sample (ranging) is
                            # exposed on the MF path only — the device uses long
                            # symbols; the PLL path here serves short-symbol
                            # messaging where ranging isn't used.
                            msg = data.decode("utf-8", errors="replace")
                            src = "retro" if used_retro else "stream"
                            print(f"FRAME seq={seq_num} snr={frame_snr:.1f}dB "
                                  f"rs={rs_errors} src={src} text={msg}", flush=True)
                            _frame_rec = {"seq": seq_num, "text": msg,
                                          "bytes_hex": data.hex(),
                                          "snr_db": round(frame_snr, 1),
                                          "rs_errors": rs_errors,
                                          "source": src}
                            decoded_frames.append(_frame_rec)
                            frames_decoded += 1
                            consec_errors  = 0
                            _decoded_seqs.add(seq_num)
                            if _on_frame is not None:
                                try:
                                    _on_frame(dict(_frame_rec))
                                except Exception as _ofe:
                                    print(f"on_frame hook error: {_ofe}", flush=True)
                            # ── Spectrograph: frame event ─────────────────
                            try:
                                _telem.sendto(json.dumps({
                                    "t":        "frame",
                                    "seq":      seq_num,
                                    "n":        frames_decoded,
                                    "sc":       sym_count,
                                    "fa_peak":  _fa_peak_bytes,  # frame-audio bytes freed
                                }).encode(), _TELEM_ADDR)
                            except Exception:
                                pass
                            if preamble_per_frame:
                                do_resync("preamble_per_frame: re-arming for next frame",
                                          reset_timing=False)
                        else:
                            raw_hex = data[:6].hex() if data else "empty"
                            print(f"FRAME_ERROR seq={seq_num} "
                                  f"reason=crc_fail snr={frame_snr:.1f}dB "
                                  f"raw={raw_hex}", flush=True)
                            frames_errored += 1
                            consec_errors  += 1
                            # Schedule a deferred anchor re-decode: by the time
                            # the look-ahead elapses, _audio_hist holds this
                            # frame's payload tail (missing now if acquisition
                            # was mis-aligned at cold start).
                            if _anchor_enabled and _anchor_lookahead > 0:
                                _anchor_retry     = _anchor_lookahead
                                _anchor_retry_snr = frame_snr
                            if preamble_per_frame:
                                do_resync("preamble_per_frame: CRC fail, re-arming",
                                          reset_timing=False)

                        if not preamble_per_frame and consec_errors >= resync_after:
                            do_resync(f"{consec_errors} CRC failures",
                                      reset_timing=True)

            diag["symbols_received"] = sym_count
            diag["avg_snr_db"]       = round(float(np.mean(snr_history)), 1) if snr_history else 0
            diag["input_overflows"]  = _input_overflows

    except KeyboardInterrupt:
        print("INTERRUPTED", flush=True)
        diag["symbols_received"] = sym_count
    except Exception as exc:
        import traceback
        print(f"ERROR: {exc}", flush=True)
        traceback.print_exc()
        result["failure_reason"] = f"stream_error: {exc}"
        result["diagnostics"]    = diag
        result_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return result
    finally:
        # Restore the previous SIGTERM handler regardless of how we exited.
        _signal_mod.signal(_signal_mod.SIGTERM, _prev_sigterm)

    if rms_history:
        diag["peak_rms"] = round(float(np.max(rms_history)),  6)
        diag["mean_rms"] = round(float(np.mean(rms_history)), 6)
        diag["silence"]  = diag["peak_rms"] < silence_thr

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

    result.update({
        "status": status, "failure_reason": reason,
        "frames_decoded": frames_decoded, "frames_errored": frames_errored,
        "diagnostics": diag, "frames": decoded_frames,
    })
    if decoded_frames:
        # Assemble by frame sequence number, not decode order: a deferred
        # anchor re-decode can recover an earlier frame AFTER a later one has
        # already been appended, so completion order ≠ message order.
        # assemble_message also drops the encoder's final-chunk padding.
        result["decoded_text"], _raw_msg = assemble_message(
            decoded_frames, chunk_size, conf.get("message_len"))
        result["decoded_hex"] = _raw_msg.hex()

    if wav_out and _capture:
        import wave as _wave_mod
        try:
            _audio = np.concatenate(_capture).astype(np.float32)
            _pcm   = (np.clip(_audio, -1.0, 1.0) * 32767).astype(np.int16)
            with _wave_mod.open(str(wav_out), "wb") as _wf:
                _wf.setnchannels(1)
                _wf.setsampwidth(2)
                _wf.setframerate(fs)
                _wf.writeframes(_pcm.tobytes())
            print(f"WAV capture -> {wav_out}  ({len(_audio)/fs:.1f}s @ {fs}Hz)", flush=True)
            result["wav_capture"] = str(wav_out)
        except Exception as _e:
            print(f"WAV capture failed: {_e}", flush=True)

    print(f"DONE status={status} frames={frames_decoded} "
          f"avg_snr={diag['avg_snr_db']}dB", flush=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    if "--print-schema" in sys.argv:
        print(json.dumps(EXAMPLE_CONFIG, indent=2))
        sys.exit(0)
    config_path = sys.argv[1] if len(sys.argv) > 1 else "css_decode_config.json"
    cfg = json.loads(Path(config_path).read_text()) if Path(config_path).exists() else {}
    result = decode(cfg)
    sys.exit(0 if result["status"] in ("ok", "partial") else 1)