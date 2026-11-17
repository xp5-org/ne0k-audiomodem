# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Clear-channel assessment by signal detection: "is one of OUR signals on the air?", not "is it loud?".

The modem's carrier sense and burst gates were a level threshold (``carrier_db``).  That is right on a wire line
(SIP / G.711: silence is silence) and wrong on a microphone or a radio, where noise is always above any fixed
level and is often louder than the signal.  This module is the usual layered answer, the same split 802.11 and
802.15.4 make between energy detect and carrier sense, and LoRa makes with channel activity detection (CAD):

* stage 1, always on, cheap, here: a per-50 ms-block "our kind of signal is present" flag from detectors that
  are normalised (independent of level, so they work below the noise):

  - **OFDM**: cyclic-prefix autocorrelation.  Every OFDM symbol repeats its last ``cp`` samples ``n_fft`` samples
    apart, so the normalised correlation of the stream with itself at lag ``n_fft`` is about
    ``cp/(n_fft+cp) * snr/(1+snr)`` on any of our bursts (lead-in, Barker preamble, probe, data) and about 0 on
    noise.  Time domain, a few multiply-adds per sample, no timing or frequency search.  Narrowband sounds
    (a tone correlates with itself at every lag) are rejected by a spectral check run only on candidates.
  - **CSS**: dechirp + FFT, LoRa's CAD.  One symbol of audio times the conjugate reference chirp turns a chirp
    with any time offset into a steady tone: one FFT bin stands far above the rest.  Run every half symbol for
    the up- and the down-chirp; two hits in a row are a detection.
  - **steady tones of ours** (the key-up lead tone of the radio channel type, the CW station ID): a Goertzel
    bin at that pitch holding most of the block's energy.

* stage 2, on demand: the full acquisition and decode (Schmidl-Cox / known-preamble coherence, the CSS PLL),
  run only on what stage 1 marked.

``Presence`` is the streaming detector: ``push`` samples in, ask for block flags by absolute sample index.
``ofdm_link._Capture`` owns one when the session's ``sense`` is ``signal`` and answers every "busy?" question
from it (``_Capture.busy_blocks`` / ``busy``), so the burst gates and the pre-TX checks all change together.

Both detectors assume white noise, and real rooms are full of fan and mains lines, so the stream is band-passed
and whitened first (``Presence._whiten``).

No Qt.  scipy only for the FIR convolution.
"""
from __future__ import annotations

import collections
import dataclasses
import threading

import numpy as np

GATE_S = 0.05            # block length: the same 50 ms the level gates use
WIN_BLOCKS = 4           # OFDM correlation window: 200 ms (noise sigma ~ 1/sqrt(2 * bandwidth * window))
NARROW_TOP3 = 0.55       # top-3 FFT bins holding more than this share of a block = a tone / sweep, not OFDM
TONE_SHARE = 0.5         # key-up tone: Goertzel power share of the block
TONE_HOLD_S = 0.1

WHITEN_NFFT = 2048       # noise spectrum resolution (23 Hz at 48 kHz) ...
WHITEN_TAPS = 1025       # ... and the whitening FIR made from it
WHITEN_HOP = 12000       # one noise spectrum per 0.25 s
WHITEN_SPECTRA = 128     # median over the last 32 s of them
WHITEN_MIN = 8           # spectra before the first whitening filter (2 s)
WHITEN_REFRESH = 8       # new spectra between redesigns (2 s)
WHITEN_GUARD_BLOCKS = 10 # a spectrum within 0.5 s of a flagged block is not noise
WHITEN_STALE = 240       # ... unless nothing else has been seen for 60 s

OFDM, TONE, CHIRP = 1, 2, 4      # bits of a block's flag


@dataclasses.dataclass
class SenseSpec:
    """What to listen for.  ``ofdm`` lists (n_fft, cp_len) in samples at ``fs`` for every geometry on the air
    (handshake messages and data can differ); ``chirp`` is (f_min, f_max, symbol_dur)."""
    fs: int = 48000
    band: tuple[float, float] = (300.0, 6000.0)
    ofdm: tuple[tuple[int, int], ...] = ()
    chirp: tuple[float, float, float] | None = None
    tones: tuple[float, ...] = ()            # steady pitches of ours: the key-up lead tone, the CW ID
    ofdm_thr: float = 0.12
    chirp_thr: float = 30.0

    def key(self) -> tuple:
        return dataclasses.astuple(self)


class Presence:
    """Streaming stage-1 detector.  Thread-safe: one writer (``push``), any readers."""

    def __init__(self, spec: SenseSpec, keep_s: float = 130.0) -> None:
        from scipy.signal import fftconvolve
        self.spec, self.fs = spec, int(spec.fs)
        self.blk = int(GATE_S * self.fs)
        self._fftconvolve = fftconvolve
        # Front end: band-pass + noise whitening in one FIR (see _whiten).  Until the room has been heard for
        # WHITEN_MIN spectra it is the plain band-pass.
        f = np.fft.rfftfreq(WHITEN_NFFT, 1.0 / self.fs)
        self._band = (f >= max(20.0, spec.band[0] * 0.8)) & (f <= min(0.48 * self.fs, spec.band[1] * 1.15))
        self._psd: collections.deque = collections.deque(maxlen=WHITEN_SPECTRA)
        self._psd_buf = np.zeros(0, np.float64)
        self._psd_skip = 0
        self._psd_new = self._psd_dirty = 0
        self._last_flag = -10**9                              # last block with any flag
        self._fir = self._design(None)
        self._fir_tail = np.zeros(WHITEN_TAPS - 1, np.float64)
        self.whitened = False
        self._lags = sorted({int(n) for n, _ in spec.ofdm})
        self._sps = int(round(spec.chirp[2] * self.fs)) if spec.chirp else 0
        self._hist = max(self._lags + [self._sps, self.blk])
        self._buf = np.zeros(self._hist, np.float64)          # [history | pending]
        self._sums: collections.deque = collections.deque(maxlen=WIN_BLOCKS)   # per block: (e0, {lag: (num, eL)})
        self._flags = np.zeros(int(keep_s / GATE_S) + 8, np.uint8)             # ring by block number
        self._nblk = 0                                        # blocks decided
        self.n = 0                                            # samples consumed
        self._lock = threading.Lock()
        self.metric = 0.0                                     # last OFDM correlation (diagnostics)
        self.cad_peak = 0.0                                   # last CAD peak / mean
        self.counts = {"ofdm": 0, "tone": 0, "chirp": 0, "blocks": 0}
        if self._sps:
            f0, f1, T = spec.chirp
            t = np.arange(self._sps) / self.fs
            ph = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T * t * t)
            self._up = np.exp(-1j * ph)                       # conjugate of the up-chirp
            self._dn = np.exp(-1j * ph[::-1])                 # ... of the down-chirp (time-reversed sweep)
            fb = np.fft.fftfreq(self._sps, 1.0 / self.fs)      # a chirp up to half a symbol early or late lands
            self._cad_bins = np.abs(fb) <= 0.5 * abs(f1 - f0)   # within half the sweep of 0 Hz: look only there
            self._hop = max(1, int(round(self._sps / 2 / self.blk)))            # CAD every half symbol, in blocks
            self._hold = int(np.ceil(2.0 * self._sps / self.blk)) + 1           # blocks a detection stays up
            self._cad_last_hit = -10**9
            self._chirp_until = -1
        self._tone = [np.exp(-2j * np.pi * hz / self.fs * np.arange(self.blk)) for hz in spec.tones]
        self._tone_until = -1

    # -- writer --------------------------------------------------------------

    def push(self, x: np.ndarray) -> None:
        if not len(x):
            return
        y = self._whiten(np.asarray(x, np.float64))
        self._buf = np.concatenate([self._buf, y])
        self.n += len(x)
        k = (len(self._buf) - self._hist) // self.blk
        if k <= 0:
            return
        H, blk = self._hist, self.blk
        arr = self._buf[:H + k * blk]
        cur = arr[H:].reshape(k, blk)
        e0 = np.einsum("ij,ij->i", cur, cur)
        per_lag = {}
        for L in self._lags:
            prev = arr[H - L:H - L + k * blk].reshape(k, blk)
            per_lag[L] = (np.einsum("ij,ij->i", cur, prev), np.einsum("ij,ij->i", prev, prev))
        out = np.zeros(k, np.uint8)
        for j in range(k):
            b = self._nblk + j
            f = 0
            self._sums.append((e0[j], {L: (per_lag[L][0][j], per_lag[L][1][j]) for L in self._lags}))
            if self._lags and self._ofdm(cur[j]):
                f |= OFDM
            if self._tone and e0[j] > 0:                     # Goertzel: one bin per pitch
                if max(2.0 * abs(np.dot(cur[j], w)) ** 2 for w in self._tone) / (blk * e0[j]) > TONE_SHARE:
                    self._tone_until = b + int(TONE_HOLD_S / GATE_S)
            if b <= self._tone_until:
                f |= TONE
            if self._sps:
                if (b + 1) % self._hop == 0:
                    end = H + (j + 1) * blk
                    self._cad(arr[end - self._sps:end], b, out, j)
                if b <= self._chirp_until:
                    f |= CHIRP
            out[j] |= f
            if out[j]:
                self._last_flag = b
        with self._lock:
            for j in range(k):
                self._flags[(self._nblk + j) % len(self._flags)] = out[j]
            self._nblk += k
        self.counts["blocks"] += k
        for name, bit in (("ofdm", OFDM), ("tone", TONE), ("chirp", CHIRP)):
            self.counts[name] += int(np.count_nonzero(out & bit))
        self._buf = self._buf[k * blk:]

    def _design(self, psd: np.ndarray | None) -> np.ndarray:
        g = self._band.astype(np.float64)
        if psd is not None:
            g = np.where(self._band, 1.0 / np.sqrt(psd + 1e-30), 0.0)
        g /= np.sqrt(np.mean(g[self._band] ** 2)) + 1e-30
        h = np.roll(np.fft.irfft(g), WHITEN_TAPS // 2)[:WHITEN_TAPS]
        return h * np.hanning(WHITEN_TAPS)

    def _whiten(self, x: np.ndarray) -> np.ndarray:
        """Band-pass and whiten ``x`` (same length out, WHITEN_TAPS/2 samples late).

        Room and radio noise is not flat: fan and mains lines stand 30-50 dB above the rest and correlate with
        themselves at every lag, and a dechirped line is not noise-like either.  Both detectors assume white
        noise, so the stream is first divided by the noise spectrum -- the usual pre-whitening.  The spectrum is
        the per-bin median of the last ~30 s heard with no signal flagged, refreshed every 2 s."""
        self._psd_buf = np.concatenate([self._psd_buf, x])
        while len(self._psd_buf) >= WHITEN_NFFT + self._psd_skip:
            seg = self._psd_buf[self._psd_skip:self._psd_skip + WHITEN_NFFT]
            self._psd_buf = self._psd_buf[self._psd_skip + WHITEN_NFFT:]
            self._psd_skip = WHITEN_HOP - WHITEN_NFFT
            if not np.any(seg):                               # our own transmission is fed as zeros: not the room
                continue
            # Noise is learnt while no signal is flagged: a long frame (a CSS packet runs 20 s, a probe repeats
            # one symbol for 5 s) would otherwise become "the noise" and its comb shape the filter.  The first
            # spectra are taken regardless (nothing is whitened yet, so the flags mean little), and so is anything
            # once nothing clean has been seen for WHITEN_STALE spectra, so a wrong estimate cannot lock itself in.
            busy = self._nblk - self._last_flag <= WHITEN_GUARD_BLOCKS
            if busy and len(self._psd) >= WHITEN_MIN and self._psd_dirty < WHITEN_STALE:
                self._psd_dirty += 1
                continue
            self._psd_dirty = 0 if not busy else self._psd_dirty
            self._psd.append(np.abs(np.fft.rfft(seg * np.hanning(WHITEN_NFFT))) ** 2)
            self._psd_new += 1
        if self._psd_new >= WHITEN_REFRESH and len(self._psd) >= WHITEN_MIN:
            self._psd_new = 0
            self._fir = self._design(np.median(np.asarray(self._psd), axis=0))
            self.whitened = True
        z = np.concatenate([self._fir_tail, x])
        self._fir_tail = z[-(WHITEN_TAPS - 1):]
        return self._fftconvolve(z, self._fir, "valid")

    def _ofdm(self, block: np.ndarray) -> bool:
        """CP autocorrelation over the last WIN_BLOCKS blocks, best lag; narrowband candidates rejected."""
        e0 = sum(s[0] for s in self._sums)
        best = 0.0
        if e0 > 0:
            for L in self._lags:
                num = sum(s[1][L][0] for s in self._sums)
                eL = sum(s[1][L][1] for s in self._sums)
                if eL > 0:
                    best = max(best, num / np.sqrt(e0 * eL))
        self.metric = float(best)
        if best < self.spec.ofdm_thr:
            return False
        S = np.abs(np.fft.rfft(block * np.hanning(len(block)))) ** 2
        tot = float(S.sum())
        return tot > 0 and float(np.sort(S)[-3:].sum()) / tot < NARROW_TOP3

    def _cad(self, win: np.ndarray, b: int, out: np.ndarray, j: int) -> None:
        """Channel activity detection on the last symbol of audio.  Two hits within three half-symbols count."""
        if not np.any(win):
            return
        peak = 0.0
        for ref in (self._up, self._dn):
            S = (np.abs(np.fft.fft(win * ref)) ** 2)[self._cad_bins]
            m = float(np.median(S))
            if m > 0:
                peak = max(peak, float(S.max()) / (m / np.log(2.0)))    # median / ln2 = mean of the noise bins
        self.cad_peak = peak
        if peak < self.spec.chirp_thr:
            return
        if b - self._cad_last_hit <= 3 * self._hop:
            first = self._cad_last_hit - int(np.ceil(self._sps / self.blk))     # where the first hit's window began
            self._chirp_until = b + self._hold
            with self._lock:                                  # back-fill: the burst began before it was confirmed
                for bb in range(max(first, self._nblk - len(self._flags) + 1, 0), self._nblk):
                    self._flags[bb % len(self._flags)] |= CHIRP
            out[max(0, first - self._nblk):j] |= CHIRP
        self._cad_last_hit = b

    # -- readers -------------------------------------------------------------

    def flags(self, i0: int, i1: int, blk: int | None = None) -> np.ndarray:
        """Flag bits per ``blk``-sample block of [i0, i1) (whole blocks only, as the level gates count them).  A
        block not decided yet, or already forgotten, reads 0."""
        blk = blk or self.blk
        k = max(0, (i1 - i0) // blk)
        if not k:
            return np.zeros(0, np.uint8)
        mid = (i0 + blk // 2 + np.arange(k) * blk) // self.blk
        with self._lock:
            ok = (mid < self._nblk) & (mid >= max(0, self._nblk - len(self._flags)))
            return np.where(ok, self._flags[mid % len(self._flags)], 0).astype(np.uint8)

    def busy(self, look_s: float = 0.4) -> bool:
        """Any of our signals in the last ``look_s``."""
        return bool(np.any(self.flags(self.n - int(look_s * self.fs) - self.blk, self.n)))

    def stats(self) -> str:
        c = self.counts
        return (f"sense{'' if self.whitened else ' (not whitened yet)'} blocks={c['blocks']} ofdm={c['ofdm']} "
                f"tone={c['tone']} chirp={c['chirp']} "
                f"corr={self.metric:.2f} cad={self.cad_peak:.0f}")


def spec_from_config(ofdm=None, css=None, *, tones: tuple = (), msg_nfft: int | None = 1024, ofdm_thr: float = 0.12, chirp_thr: float = 30.0) -> SenseSpec:
    """SenseSpec for an OfdmConfig and/or CssConfig (either may be None: that modem is not in use)."""
    fs = int((ofdm or css).fs)
    lo, hi, geo = [], [], []
    if ofdm is not None:
        lo.append(ofdm.freq_lo); hi.append(ofdm.freq_hi)
        for n in {int(ofdm.n_fft), int(msg_nfft or ofdm.n_fft)}:
            geo.append((n, int(round(n * ofdm.cp_ratio))))
    chirp = None
    if css is not None and int(css.fs) == fs:
        lo.append(css.f_min); hi.append(css.f_max)
        chirp = (float(css.f_min), float(css.f_max), float(css.symbol_dur))
    return SenseSpec(fs=fs, band=(min(lo), max(hi)), ofdm=tuple(sorted(geo)), chirp=chirp,
                     tones=tuple(float(t) for t in tones), ofdm_thr=ofdm_thr, chirp_thr=chirp_thr)


def for_session(session) -> Presence | None:
    """A detector for this session's captures, or None when carrier sense is by level (``Session.sense_mode``)."""
    spec = getattr(session, "sense_spec", None)
    if getattr(session, "sense_mode", "level") != "signal" or spec is None:
        return None
    return Presence(spec)


def present_s(spec: SenseSpec, x: np.ndarray) -> float:
    """Seconds of ``x`` in which one of our signals is present (a one-off pass over a slice)."""
    p = Presence(spec, keep_s=len(x) / spec.fs + 1.0)
    p.push(x)
    return float(np.count_nonzero(p.flags(0, len(x)))) * GATE_S
