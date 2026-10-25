"""
ofdm_modem.py -- ADSL2-inspired Acoustic OFDM Modem
=====================================================

Protocol phases (mirrors ADSL2 handshake):

  PROBE    Transmit N OFDM symbols carrying known BPSK on all subcarriers
           (like ADSL2 MEDLEY).  Receiver coherently averages across symbols
           to estimate per-subcarrier H[k] and noise variance sigma^2[k].
           SNR[k] = 2|H[k]|^2 / sigma^2[k].

  ALLOCATE Greedy discrete bit loading: for each data subcarrier assign the
           highest b in {0,1,2,4,6} such that
               SNR_dB[k] >= SNR_FLOOR[b] + snr_margin_db
           0=muted  1=BPSK  2=QPSK  4=16-QAM  6=64-QAM.
           Result: bits_per_bin[k], one value per data subcarrier.

  NEGOTIATE Decoder saves bits_per_bin to ofdm_bat.json and signals
            "ofdm_bat_ready" via coord server.  Encoder reads it.
            (Acoustic BAT retransmission is optional / future work.)

  DATA     Adaptive-QAM superframe:
               [SC preamble] [CE symbol] [SYNC] [data*sync_period]
                                                 [SYNC] [data*sync_period] ...
           SYNC symbols refresh H[k] per frame; pilot subcarriers give
           per-symbol CPE correction within each data burst.

Config params (all optional, with defaults):
    n_fft          FFT size, power of 2            (1024)
    cp_ratio       cyclic-prefix fraction           (0.25)
    freq_lo        lowest occupied bin, Hz          (1000)
    freq_hi        highest occupied bin, Hz         (16000)
    pilot_spacing  1 pilot every N subcarriers      (8)
    sample_rate    Hz                               (48000)
    ecc_bytes      Reed-Solomon parity bytes        (16)
    chunk_bytes    payload bytes per RS block       (256)
    seq_bytes      sequence counter bytes           (2)
    taper_ms       W-OFDM raised-cosine taper, ms  (2.0)
    frame_gap_ms   inter-frame silence, ms          (20.0)
    n_probe_syms   probe symbols for SNR averaging  (16)
    snr_margin_db  headroom above loading floor     (6.0)
    max_bits_per_bin  ceiling per subcarrier        (6)
    sync_period    SYNC symbol every N data syms    (32)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
# RS_BACKEND=c swaps the pure-Python reedsolo for the C core in
# tests/c/ (rs255_c.RSCodec is a drop-in and defaults to reedsolo),
# which is why this imports the shim rather than the package.
from rs255_c import RSCodec

# -
# PRBS seeds  (distinct to avoid correlation between sequences)
# -
_SEED_DATA  = 0xACE1
_SEED_CE    = 0x3D5F
_SEED_PROBE = 0x7A3F
_SEED_SYNC  = 0x1B4E
_SEED_SC    = 0x3D5F ^ 0xFFFF   # = _SEED_CE ^ 0xFFFF
# Base seed for the probe's NON-PERIODIC measurement segment.  Symbol j carries
# _known_bpsk(n_occupied, _SEED_MEAS + j), so consecutive symbols are
# uncorrelated -- see ChannelProbe.build for why that is the whole point.
_SEED_MEAS  = 0x51C7

# -
# Bit-loading thresholds
# Minimum SNR (dB) needed to USE b bits per subcarrier.
# Derived from BER curves plus ~3 dB RS coding gain.
# Receiver applies an additional snr_margin_db on top.
# -
def _rs_encoded_len(rsc, seq_bytes: int, chunk_bytes: int,
                    ecc_bytes: int) -> int:
    """Wire length of one RS-coded frame body, measured rather than assumed.

    This used to be `seq + chunk + ecc`, which is only right while the body fits
    ONE Reed-Solomon codeword.  reedsolo splits the message into blocks of
    (255 - ecc) bytes and appends `ecc` parity to EACH, so at seq+chunk of 240
    the encoded length jumps to 272, not 256 -- and a frame geometry built on the
    short figure allocates too few symbols, so nothing decodes.  That single
    assumption was the whole reason chunk_bytes could not exceed 237 (which is
    why an application payload over ~233 B was impossible regardless of link
    rate).  Deriving the length from the codec removes the ceiling, and stays
    deterministic: both ends compute it from the same params.

    Bonus property: parity now scales with length (16 bytes per 239-byte block),
    so a longer frame is not proportionally less protected.
    """
    return len(rsc.encode(bytes(seq_bytes + chunk_bytes)))


# ---------------------------------------------------------------------------
# Block interleaver for the DATA path
# ---------------------------------------------------------------------------
#
# WHY.  An impulse destroys a whole OFDM symbol -- the burst is time-domain, the
# FFT spreads it over every subcarrier of the symbol it lands in, so the symbol
# is lost regardless of how narrow the burst was.  One symbol carries
# bits_per_ofdm_sym/8 bytes (~16.8 B at QPSK on 67 carriers), and those bytes
# are CONTIGUOUS in the RS stream.  RS(255) with ecc_bytes=16 corrects t=8 bytes
# per codeword, so a single lost symbol is already double the budget: the
# impulse benchmark scored the data path 0/100, failing at one burst.
#
# Interleaving fixes that without adding a single parity byte, because it is a
# permutation: the same bytes are sent, in a different order.  Split the message
# into `depth` EQUAL RS codewords, then transmit byte-column-wise, so
# consecutive wire bytes come from different codewords.  A contiguous run of B
# lost wire bytes then puts only ceil(B/depth) errors into any one codeword.
#
# Two things this deliberately does NOT do:
#
#   * It does not reuse reedsolo's internal blocking.  That splits at 239 data
#     bytes and leaves a ragged remainder -- at chunk 256 the blocks are 239+16
#     and 17+16, so a burst landing in the short block is hopeless and the
#     interleave would be uneven anyway.  Equal blocks are the whole point.
#
#   * It does not touch the handshake.  RobustLink negotiates the link and has
#     to stay wire-stable across versions; the data path is where the payload
#     is, and where the loss was measured.
#
# The depth is DERIVED, never signalled: both ends know cfg and the negotiated
# BAT, so both compute the same number. A field on the wire would be one more
# thing to disagree about, and it would have to survive the very burst it is
# there to protect against.


def _interleave_depth(bits_per_ofdm_sym: int, ecc_bytes: int,
                      msg_bytes: int) -> int:
    """How many equal RS codewords one frame body splits into.

    Chosen so that ONE fully-lost OFDM symbol cannot exceed any codeword's
    correction budget: with t = ecc_bytes//2 correctable bytes per codeword and
    S bytes carried per symbol, depth >= S/t.  One extra byte of slack covers
    the case where the lost run straddles a column boundary and so touches one
    codeword twice.

    Clamped so each codeword's data part still fits RS(255)'s k = 255 -
    ecc_bytes, and to at least 1 (depth 1 is the old un-interleaved layout).
    """
    t = max(1, int(ecc_bytes) // 2)
    sym_bytes = math.ceil(int(bits_per_ofdm_sym) / 8)
    depth = max(1, math.ceil((sym_bytes + 1) / t))
    k_max = max(1, 255 - int(ecc_bytes))
    # Every codeword must hold its share of the message.
    depth = max(depth, math.ceil(int(msg_bytes) / k_max))
    # More codewords than message bytes would mean empty blocks.
    return max(1, min(depth, max(1, int(msg_bytes))))


def _interleave_geometry(seq_bytes: int, chunk_bytes: int, ecc_bytes: int,
                         bits_per_ofdm_sym: int) -> tuple[int, int, int]:
    """(depth, k_per_block, wire_len) for one frame body.

    `k_per_block` is the padded data size of each equal codeword; the message is
    zero-padded up to depth*k. The padding is not sent as a length -- both ends
    derive it from cfg, and the true message length (seq_bytes + chunk_bytes) is
    fixed, so the receiver simply truncates.
    """
    msg = int(seq_bytes) + int(chunk_bytes)
    depth = _interleave_depth(bits_per_ofdm_sym, ecc_bytes, msg)
    k = math.ceil(msg / depth)
    return depth, k, depth * (k + int(ecc_bytes))


def _interleave(blocks: list[bytes]) -> bytes:
    """Column-major read of equal-length codewords."""
    if len(blocks) == 1:
        return blocks[0]
    return bytes(np.asarray([bytearray(b) for b in blocks],
                            dtype=np.uint8).T.reshape(-1))


def _deinterleave(buf: bytes, depth: int) -> list[bytes]:
    """Inverse of _interleave. `len(buf)` must be a multiple of `depth`."""
    if depth == 1:
        return [bytes(buf)]
    a = np.frombuffer(bytes(buf), dtype=np.uint8)
    return [bytes(r) for r in a.reshape(-1, depth).T]


def rs_encode_interleaved(rsc, msg: bytes, depth: int, k: int,
                          ecc_bytes: int) -> bytes:
    """RS-encode `msg` as `depth` equal codewords and interleave them."""
    pad = bytes(msg) + bytes(depth * k - len(msg))
    blocks = [bytes(rsc.encode(pad[i * k:(i + 1) * k])) for i in range(depth)]
    L = len(blocks[0])
    if any(len(b) != L for b in blocks):
        # Equal inputs must give equal outputs; if they ever do not, the
        # column-major transpose below would silently misalign every byte.
        raise ValueError(f"RS produced ragged blocks: {[len(b) for b in blocks]}")
    return _interleave(blocks)


def rs_decode_interleaved(rsc, buf: bytes, depth: int, msg_len: int):
    """Inverse of rs_encode_interleaved. Returns (msg, total_errata).

    Raises if ANY codeword is uncorrectable -- a partially recovered frame is
    not useful here, and the caller already treats a raised decode as a lost
    frame.
    """
    total = 0
    out = bytearray()
    for blk in _deinterleave(buf, depth):
        dec, _, errata = rsc.decode(bytearray(blk))
        out += bytes(dec)
        total += len(errata)
    return bytes(out[:msg_len]), total


# Per-carrier SNR each modulation needs, for THIS implementation's FEC chain
# (rate-1/2 K=7 convolutional + RS outer, soft max-log LLRs).  These are AWGN
# thresholds and nothing else: reverberation is handled separately by the ISI
# ceiling (isi_ratio) and probe->data fade migration by snr_margin_db, so a floor
# that also hedged against those would be counting the same risk twice.
#
# MEASURED 20260804 by sweeping encode -> AWGN -> decode, 12 frames per point,
# lowest SNR with 12/12 frames recovered:
#     BPSK 3 dB, QPSK 6 dB, 16-QAM 10 dB, 64-QAM 15 dB
# Each floor below is that plus ~1 dB.  BPSK/QPSK were already about right; the
# dense modes were 4-6 dB conservative and could never be allocated even on a
# channel that would carry them.
#
# On textbook figures: published rate-1/2 coded QPSK thresholds of ~1.5-2.0 dB
# Eb/N0 are not reachable here, and the gap is accounted for rather than
# mysterious -- K=7 Viterbi alone is ~3 dB Eb/N0 even with an RS outer (1.5-2.0
# is turbo/LDPC territory), the cyclic prefix spends 20% of transmit energy on
# no new information (~0.97 dB), 1 carrier in 8 is a pilot (~0.58 dB), and the
# one-shot CE/SYNC channel estimate costs another ~1 dB (which is what the
# data-aided second demod pass claws back).  That sums to the measured 6 dB.
# THE DENSE MODES KEEP THEIR ORIGINAL MARGIN, and the AWGN measurement is NOT
# reason enough to lower them.  Lowering 16-QAM to 11 dB on the strength of that
# measurement put 67-72 carriers into 16-QAM over the air (run 20260804_064547,
# max_bits_per_bin=4): every DATA frame then locked and failed RS, the receiver
# never ACKed, and the link tore down as `data_unacked` four times before the
# node gave up.  Reverted to 14/21.
#
# The lesson is about WHERE a threshold was measured.  A 4-6 dB gap over the AWGN
# threshold is not slack -- it is covering the reverberation this stack knows is
# present and cannot measure (the ISI estimator saturates once the room's tail
# outlasts n_fft/fs; see _ISI_BETA_MAX).  An AWGN bench number cannot authorise
# spending margin that exists for a non-AWGN impairment.
SNR_FLOOR_DB: dict[int, float] = {
    1:  4.0,   # BPSK    (AWGN threshold measured at 3 dB)
    2:  7.0,   # QPSK    (AWGN threshold measured at 6 dB)
    4: 14.0,   # 16-QAM  (AWGN threshold 10 dB; the rest covers unmeasured ISI)
    6: 21.0,   # 64-QAM  (AWGN threshold 15 dB; same reason)
}
VALID_BIT_LEVELS = [0, 1, 2, 4, 6]

# -
# Probe acquisition (coherence search -- replaces Schmidl-Cox, which
# false-locks on the audio-stream startup click).  See ChannelProbe.estimate.
# -
_COH_CARRIERS = 40    # strongest carriers used to score coherence
_COH_SCAN_SYMS = 8    # probe symbols averaged per candidate start
_COH_GATE     = 0.70  # min peak coherence to declare a valid probe
_COH_MATCH    = 0.60  # per-symbol H match to count a symbol as "probe"

# Minimum normalised correlation with the known SC preamble for an SC-metric
# peak to be allowed to SUPPRESS its neighbours (see _find_sc_peaks).
#
# Deliberately LOW.  This is not "is this a good frame?" -- it is "is this a
# preamble at all?", and it only has to catch things that are obviously not, so
# that they stop consuming a min_gap slot.  Measured across both sim directions
# at 14/10/6 dB SNR: the audio-stream startup click scores 0.037 every time,
# while frames that actually decoded scored 0.29 to 0.90.  0.10 sits ~2.7x
# above the click and ~2.9x below the weakest real frame -- balanced in log
# space between the two populations.
#
# Do NOT raise this to try to reject spurious peaks generally: the populations
# overlap badly at low SNR (a 6 dB capture produced spurious peaks at 0.72 and
# real ones at 0.35), so a discriminating threshold would cost real frames.
# Rejecting junk-that-blocks is the job; rejecting junk-that-merely-fails is
# what RS is for.
SC_NCC_MIN    = 0.10
# Candidates per block in _grid_coherence_scan.  Sets the working-set bound:
# the scan holds (_COH_BLOCK + stride*(nsyms-1)) H rows, never the whole
# capture, so memory is O(1) in capture length.  Purely a host tuning knob --
# numpy wants a batch big enough to amortise the FFT call; the C port uses the
# equivalent of 1 (a true ring buffer).  Does not affect results.
_COH_BLOCK    = 512

# -
# Constellation tables
# -
_R2  = math.sqrt(2.0)
_R10 = math.sqrt(10.0)
_R42 = math.sqrt(42.0)

# BPSK: bit 0 -> +1, bit 1 -> -1
_BPSK_TABLE: dict[tuple, complex] = {(0,): 1.0+0j, (1,): -1.0+0j}
BPSK_POINTS = np.array([1.0+0j, -1.0+0j])
BPSK_BITS   = np.array([[0], [1]], dtype=np.uint8)

# QPSK  (Gray coded)
_QPSK_TABLE: dict[tuple, complex] = {
    (0,0): ( 1+1j)/_R2, (0,1): (-1+1j)/_R2,
    (1,1): (-1-1j)/_R2, (1,0): ( 1-1j)/_R2,
}
QPSK_POINTS = np.array(list(_QPSK_TABLE.values()))
QPSK_BITS   = np.array(list(_QPSK_TABLE.keys()), dtype=np.uint8)

# 16-QAM  (Gray coded, same layout as qam_writer)
_QAM16_TABLE: dict[tuple, complex] = {
    (0,0,0,0): -3-3j, (0,0,0,1): -3-1j, (0,0,1,1): -3+1j, (0,0,1,0): -3+3j,
    (0,1,1,0): -1+3j, (0,1,1,1): -1+1j, (0,1,0,1): -1-1j, (0,1,0,0): -1-3j,
    (1,1,0,0):  1-3j, (1,1,0,1):  1-1j, (1,1,1,1):  1+1j, (1,1,1,0):  1+3j,
    (1,0,1,0):  3+3j, (1,0,1,1):  3+1j, (1,0,0,1):  3-1j, (1,0,0,0):  3-3j,
}
QAM16_POINTS = np.array(list(_QAM16_TABLE.values())) / _R10
QAM16_BITS   = np.array(list(_QAM16_TABLE.keys()), dtype=np.uint8)

# 64-QAM  (Gray coded 8x8, 3 bits I + 3 bits Q)
_G3: dict[tuple, int] = {
    (0,0,0): -7, (0,0,1): -5, (0,1,1): -3, (0,1,0): -1,
    (1,1,0):  1, (1,1,1):  3, (1,0,1):  5, (1,0,0):  7,
}
_QAM64_TABLE: dict[tuple, complex] = {
    ib + qb: (ia + 1j*qa) / _R42
    for ib, ia in _G3.items()
    for qb, qa in _G3.items()
}
QAM64_POINTS = np.array(list(_QAM64_TABLE.values()))
QAM64_BITS   = np.array(list(_QAM64_TABLE.keys()), dtype=np.uint8)

# Fast lookup: bits-per-symbol -> (points_array, bits_array)
_MOD_TABLES: dict[int, tuple] = {
    1: (BPSK_POINTS,  BPSK_BITS),
    2: (QPSK_POINTS,  QPSK_BITS),
    4: (QAM16_POINTS, QAM16_BITS),
    6: (QAM64_POINTS, QAM64_BITS),
}

_MOD_ENCODE: dict[int, dict] = {
    1: _BPSK_TABLE,
    2: _QPSK_TABLE,
    4: {k: v/_R10 for k, v in _QAM16_TABLE.items()},
    6: _QAM64_TABLE,
}

# -
# Shared OFDM primitives
# -

def _prbs(length: int, seed: int = _SEED_DATA) -> np.ndarray:
    """16-bit maximal LFSR, same polynomial as ofdm_writer."""
    state = seed & 0xFFFF
    out   = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        fb     = ((state>>15)^(state>>14)^(state>>12)^(state>>3)) & 1
        out[i] = (state >> 15) & 1
        state  = ((state << 1) | fb) & 0xFFFF
    return out


def _known_bpsk(n: int, seed: int) -> np.ndarray:
    pn = _prbs(n, seed)
    return (1.0 - 2.0 * pn.astype(float)).astype(complex) / math.sqrt(2)


# Amplitude boost applied to the DATA FRAME's CE and SYNC symbols, and to
# nothing else.
#
# _known_bpsk returns +-1/sqrt(2), i.e. power 0.5, while every data
# constellation and PILOT_SYM are unit power.  So the two symbols the receiver
# measures H from were the only thing in the frame transmitted 3 dB DOWN, and
# the channel estimate came out 3 dB noisier than the signal it corrects --
# a self-inflicted penalty that lands hardest on 16/64-QAM, which is where the
# bits are.
#
# Applied at the CE/SYNC construction sites rather than inside _known_bpsk on
# purpose: the same helper builds the PROBE's symbols, and the probe's average
# power is what scale_hs_tx pegs every burst to (see probe_rms_ratio).  Moving
# the probe's level here would silently re-calibrate every transmission in the
# system against a level the far end already measured.  meshlib's _SEED_MD
# pattern is likewise left alone.
#
# The cost is PAPR on two symbols per frame, and those two are the most
# PAPR-benign in it: random BPSK on every occupied carrier.
REF_SYM_BOOST = math.sqrt(2.0)


# NO CHANNEL-ESTIMATE SMOOTHING HERE, AND THAT IS A MEASURED RESULT.
#
# The obvious denoiser for a least-squares H is to window it in DELAY: transform
# H to the delay domain, keep only the taps the cyclic prefix covers, transform
# back.  The prior is that delay spread <= cp_len, and where that holds it is
# worth 5.5 dB of estimator noise at the 1024/256 geometry.
#
# It does not hold in the rooms this modem actually runs in, and the sign of
# the effect flips.  Graded over the simulated room profiles (48 kHz, 14 dB,
# 400-8000 Hz), mean frame EVM and frames decoded:
#
#     window        anechoic        desk room
#     off           11.3%  9/9      27.9%  8/8
#     cp_len        10.1%  9/9      30.5%  7/8      <- best clean, worst real
#     2*cp_len      10.4%  9/9      28.2%  8/8
#     noise-adaptive 10.7%  9/9     29.3%  6/8
#
# The reason is that a reverberant room's response is LONGER than the prefix by
# construction -- that is what isi_ratio measures -- and the tail beyond the
# prefix is not purely ISI: across a frame of many symbols it reaches periodic
# steady state and contributes real, usable gain that H legitimately captures.
# Zeroing those taps deletes signal the equaliser was using.  The 1.2 dB the
# window buys on a single-tap channel is not a channel this link ever sees.
#
# Do not reintroduce this without re-running that grid.  If it is ever wanted,
# the honest version needs a delay-spread estimate driving the window length,
# not a constant -- and the flat result above says the payoff does not justify
# it.


def _ofdm_td(occ: np.ndarray, n_fft: int, bin_lo: int, cp_len: int) -> np.ndarray:
    """Hermitian-symmetric IFFT -> real time-domain symbol with CP prepended.

    Only the positive-frequency half is filled; irfft IS the Hermitian inverse,
    so the conjugate mirror X[N-b]=conj(X[b]) is implied rather than
    materialised.  That halves the transform's flops -- a real-output inverse
    is an N/2 complex FFT plus an O(N) split step -- and is the form a
    fixed-point C port should mirror.  Occupied bins never reach DC or Nyquist
    (bin_lo >= 1, bin_hi <= N/2-1), so the half spectrum is unambiguous.
    """
    X = np.zeros(n_fft // 2 + 1, dtype=complex)
    X[bin_lo:bin_lo + len(occ)] = occ
    td = np.fft.irfft(X, n=n_fft) * math.sqrt(n_fft)
    return np.concatenate([td[-cp_len:], td])


def _ofdm_td_windowed(occ, n_fft, bin_lo, cp_len, T):
    """W-OFDM: raised-cosine taper of T samples on each side of symbol."""
    if T == 0:
        return _ofdm_td(occ, n_fft, bin_lo, cp_len)
    X = np.zeros(n_fft // 2 + 1, dtype=complex)   # half spectrum; see _ofdm_td
    X[bin_lo:bin_lo + len(occ)] = occ
    td  = np.fft.irfft(X, n=n_fft) * math.sqrt(n_fft)
    ext = np.concatenate([td[n_fft - cp_len - T:], td, td[:T]])
    t   = (np.arange(T) + 0.5) / T
    win = np.ones(len(ext))
    win[:T]  = 0.5 * (1.0 - np.cos(np.pi * t))  # rise
    win[-T:] = 0.5 * (1.0 + np.cos(np.pi * t))  # fall (complementary)
    return ext * win


def _build_ola(occ_list, n_fft, bin_lo, cp_len, T):
    """Overlap-and-add assembly.  stride = n_fft+cp_len; taper width T."""
    SYM = n_fft + cp_len
    n   = len(occ_list)
    if T == 0:
        return np.concatenate([_ofdm_td(o, n_fft, bin_lo, cp_len) for o in occ_list])
    out = np.zeros(n * SYM + 2 * T)
    for i, occ in enumerate(occ_list):
        sym = _ofdm_td_windowed(occ, n_fft, bin_lo, cp_len, T)
        out[i * SYM : i * SYM + SYM + 2 * T] += sym
    return out


def _sc_metric(samples: np.ndarray, n_fft: int) -> np.ndarray:
    """Schmidl-Cox delay-correlate metric, normalised to [0, 1].

        M[d] = |P[d]|^2 / (R1[d] * R2[d])

    where P correlates the two half-symbol windows and R1/R2 are their
    energies.  By Cauchy-Schwarz M <= 1, so a fixed threshold (e.g. 0.25) is
    meaningful.  The original form used R2^2 in the denominator, so a DECAYING
    transient (first half loud, second half quiet -> P >> R2) drove M far above
    1 -- the audio-stream startup click scored ~317 and swamped the real
    preamble.  Using R1*R2 removes that amplitude sensitivity."""
    r = samples.astype(np.float64)
    L = n_fft // 2
    n = len(r)
    if n < n_fft:
        return np.zeros(0)
    prod = r[:n - L] * r[L:n]
    sq1  = r[:n - L] ** 2   # energy of the FIRST half-window
    sq2  = r[L:n] ** 2      # energy of the SECOND half-window
    cP  = np.empty(len(prod) + 1); cP[0]  = 0.0; np.cumsum(prod, out=cP[1:])
    cR1 = np.empty(len(sq1)  + 1); cR1[0] = 0.0; np.cumsum(sq1,  out=cR1[1:])
    cR2 = np.empty(len(sq2)  + 1); cR2[0] = 0.0; np.cumsum(sq2,  out=cR2[1:])
    md = len(cP) - L - 1
    if md < 1:
        return np.zeros(0)
    d = np.arange(md)
    P   = cP[d + L]  - cP[d]
    R1  = cR1[d + L] - cR1[d]
    R2  = cR2[d + L] - cR2[d]
    den = R1 * R2
    return np.where(den > 1e-18, P * P / (den + 1e-24), 0.0)


def _find_sc_peaks(
    M: np.ndarray,
    symbol_samples:  int,
    min_gap_samples: int,   # minimum sample distance between consecutive frame starts
    threshold: float = 0.25,
    validate = None,        # optional (pos) -> bool; see below
) -> list[int]:
    """Return refined frame-start sample indices from SC metric.

    `validate` rejects a candidate that does not actually look like an SC
    preamble.  A rejected candidate is skipped AND DOES NOT ADVANCE `last`,
    which is the entire point: the delay-correlator scores ~1 on any decaying
    transient, so the audio-stream startup click lands a peak at sample ~30,
    and a click that consumed the min_gap slot suppressed the REAL first frame
    behind it.  Measured in the handshake sim: a 36-carrier link lost frame 0
    every time (click-to-frame0 26427 samples against a min_gap of 47536), and
    a 64-carrier link kept it by 412 samples out of 28352 -- i.e. by luck.  The
    failure gets MORE likely as the link gets weaker, because fewer carriers
    means more symbols per frame means a longer min_gap.

    This is the same false-lock that made ChannelProbe.estimate abandon
    Schmidl-Cox for a coherence search; the data path kept SC (correctly -- it
    needs sample timing, which coherence does not give) and so needs its own
    guard.  Default None keeps the previous behaviour exactly, for callers that
    only count peaks (ofdm_fdd_node's activity metric).
    """
    above = M > threshold
    if not np.any(above):
        return []
    edges = np.where(np.diff(above.astype(np.int8)) > 0)[0] + 1
    if above[0]:
        edges = np.concatenate([[0], edges])
    min_gap    = min_gap_samples
    PLATEAU_F   = 0.95
    refine_win  = symbol_samples
    peaks: list[int] = []
    last = -(min_gap * 2)
    for d0 in edges:
        if d0 - last < min_gap:
            continue
        d1     = min(d0 + refine_win, len(M))
        region = M[d0:d1]
        if not len(region):
            continue
        r_max = float(np.max(region))
        hits  = np.where(region >= PLATEAU_F * r_max)[0]
        ref   = int(d0 + hits[0]) if len(hits) else d0
        if validate is not None and not validate(ref):
            continue          # deliberately does NOT set `last`
        peaks.append(ref)
        last = ref
    return peaks


def _grid_coherence_scan(rx: np.ndarray, n_fft: int, cp_len: int, bin_lo: int,
                         n_occupied: int, known: np.ndarray, nsyms: int,
                         step: int, q: np.ndarray | None = None,
                         top_k: int = 40):
    """Vectorised sign-matched coherence scan over a whole capture.

    Computes H at every grid position (one chunked batch FFT for the entire
    capture), then evaluates the power-normalised coherence metric

        c_k = |sum_j q_j H_jk|^2 / (nsyms * sum_j |H_jk|^2)

    for every candidate start, averaged over each candidate's top_k strongest
    carriers.  `step` MUST divide symbol_samples so that consecutive symbol
    windows land exactly on grid rows -- callers use
    gcd(symbol_samples, desired_step).  q=None means an all-ones key
    (identical repeated symbols, e.g. the channel probe).

    Replaces the per-candidate python loops that cost ~45 s per decode attempt
    on the laptop during the first OTA handshake -- the H grid is shared by
    all candidates, cutting the FFT count ~8x and vectorising the metric.

    STREAMING.  An earlier version materialised the H grid for the WHOLE
    capture and then fancy-indexed (nc, nsyms, nocc) batches out of it.  That
    is O(capture length) in memory and it does not fit the target: 9.6 MB of
    grid plus a 77 MB gather for a 10 s capture at n_fft=1024, against 520 KB
    of SRAM on an RP2350.  Candidate c only ever reads grid rows
    c + j*stride, so the grid never needs to be wider than
    stride*(nsyms-1)+1 rows -- 71 rows, 178 KB at the production geometry, and
    CONSTANT in capture length.  Here that bound is applied a block of
    candidates at a time (numpy wants the batch FFT); the C port walks it one
    row at a time as a true ring buffer.  Same arithmetic either way.

    The per-j accumulation below is deliberately SEQUENTIAL rather than a
    numpy reduction over a stacked (nc, nsyms, nocc) array.  It has to be: the
    stack is exactly the allocation this rewrite exists to remove, and a
    sequential accumulator is what the C core will do, so keeping the
    reference in the same summation order means a C-vs-numpy diff reports a
    port bug and not a reduction-tree difference.

    `step` MUST divide symbol_samples so consecutive symbol windows land
    exactly on grid rows -- callers use gcd(symbol_samples, desired_step).
    q=None means an all-ones key (identical repeated symbols, e.g. the probe).

    Returns (cand_starts, metric) as ndarrays (possibly empty).
    """
    SYM = n_fft + cp_len
    assert SYM % step == 0, "step must divide symbol_samples"

    # Backend switch lives HERE rather than at the two call sites, so
    # ChannelProbe.estimate and RobustLink.decode both get it and neither has
    # to know the C exists.  Default is numpy: the C is strictly opt-in and a
    # missing .so cannot break normal operation.
    if os.environ.get("OFDM_ACQ_BACKEND", "").lower() == "c":
        import ofdm_acq_c
        if ofdm_acq_c.available():
            return ofdm_acq_c.grid_coherence_scan_c(
                rx, n_fft, cp_len, bin_lo, n_occupied, known, nsyms, step,
                q=q, top_k=top_k)

    stride = SYM // step
    n_pos = (len(rx) - cp_len - n_fft) // step + 1
    if n_pos < stride * (nsyms - 1) + 1:
        return np.zeros(0, dtype=int), np.zeros(0)

    n_cand = n_pos - stride * (nsyms - 1)
    span   = stride * (nsyms - 1)   # extra rows a candidate reaches forward
    qv     = (np.ones(nsyms) if q is None else np.asarray(q, dtype=float))
    k      = min(top_k, n_occupied)
    scale  = 1.0 / math.sqrt(n_fft)
    base   = np.arange(n_fft)
    metric = np.empty(n_cand)

    for c0 in range(0, n_cand, _COH_BLOCK):
        c1   = min(c0 + _COH_BLOCK, n_cand)
        nc   = c1 - c0
        # Rows [c0, c0+nc+span) are everything this block of candidates reads.
        # The `span` overlap with the next block is recomputed rather than
        # carried: it is 70 rows against a block of 512, and not keeping a
        # carry buffer is what makes the bound obviously correct.
        idx  = ((np.arange(c0, c0 + nc + span) * step + cp_len)[:, None]
                + base[None, :])
        fd   = np.fft.rfft(rx[idx], n=n_fft, axis=1)
        win  = (fd[:, bin_lo:bin_lo + n_occupied] * scale
                / known).astype(np.complex64)

        acc = np.zeros((nc, n_occupied), dtype=np.complex64)  # sum_j q_j H_jk
        den = np.zeros((nc, n_occupied), dtype=np.float32)    # sum_j |H_jk|^2
        mag = np.zeros((nc, n_occupied), dtype=np.float32)    # sum_j |H_jk|
        for j in range(nsyms):
            Hj = win[j * stride: j * stride + nc]             # contiguous view
            acc += Hj * qv[j]
            a    = np.abs(Hj)
            den += a * a
            mag += a

        c_k = np.abs(acc) ** 2 / (nsyms * den + 1e-18)
        sel = np.argpartition(mag, -k, axis=1)[:, -k:]
        metric[c0:c1] = np.take_along_axis(c_k, sel, axis=1).mean(axis=1)

    return np.arange(n_cand) * step, metric


def _frame_min_gap(symbol_samples: int, n_body_syms: int, taper_samples: int,
                   cp_len: int = 0) -> int:
    """
    Minimum sample distance between consecutive SC preamble positions.

    Actual frame length = SC symbol + OLA output for body symbols.
    OLA output = n_body_syms * symbol_samples + 2 * taper_samples.
    We use this (without the inter-frame gap) as min_gap so:
      - false peaks inside the OLA taper overhang are blocked
      - the next real frame's SC preamble (which starts after the gap) is allowed

    `cp_len` is subtracted as a JITTER ALLOWANCE, and without it this silently
    deletes every second frame at large n_fft.

    The margin between this bound and the true frame stride is exactly
    frame_gap_samples -- 960 samples at the 20 ms default, and CONSTANT in
    n_fft.  What eats it is inside `_find_sc_peaks`: that function tests a
    candidate by its threshold-crossing EDGE (`d0`) but records the accepted
    peak by its refined position (`last = ref`), and ref sits later than d0 by
    however far into the metric plateau the 95%-of-max point falls.  The
    plateau is the cyclic prefix, so that offset scales with cp_len while the
    960-sample margin does not.  Once it dominates, `d0 - last` for the NEXT
    frame comes out short and a REAL frame is suppressed -- thrown away before
    it is ever demodulated, and without advancing `last`.

    The crossover lands between n_fft 2048 and 4096 (cp_len 512 -> 1024 at
    cp_ratio 0.25), and it is unmistakable once looked for.  On the `desk` room
    the decoded sequence numbers were [0,1,2,...,8] at 2048 but [0,2,4,6] at
    4096 -- every second frame -- with the scan returning 4 candidates for 7
    transmitted frames, while per-frame EVM kept IMPROVING (26% -> 17%).  The
    raw candidate spacings were ~52360 samples against a bound of 51392, i.e.
    the frames were there and correctly spaced; only the edge-vs-peak offset
    made them look too close.  It reads as a bell curve in n_fft and is not
    one; it is this bound.

    Subtracting cp_len costs almost nothing on the blocking side, since cp_len
    is a few percent of a frame.  Defaults to 0 so a caller that has not been
    updated keeps the old bound rather than silently changing behaviour.
    """
    ola_len = n_body_syms * symbol_samples + 2 * taper_samples
    return max(1, symbol_samples + ola_len - int(cp_len))   # SC + OLA - jitter


def _make_sc_preamble(n_fft, cp_len, bin_lo, n_occupied) -> np.ndarray:
    """Schmidl-Cox preamble: PN-BPSK on even-indexed occupied bins."""
    n_even = (n_occupied + 1) // 2
    pn     = _prbs(n_even, seed=_SEED_SC)
    bpsk   = (1.0 - 2.0 * pn.astype(float)).astype(complex) / math.sqrt(2)
    occ    = np.zeros(n_occupied, dtype=complex)
    j = 0
    for i in range(0, n_occupied, 2):
        occ[i] = bpsk[j] if j < len(bpsk) else 0.0
        j += 1
    return _ofdm_td(occ, n_fft, bin_lo, cp_len)


def _estimate_cpe(sym_fd, n_fft, bin_lo, pilot_offsets, H, pilot_sym) -> float:
    """Common phase error from pilot subcarriers, weighted by pilot gain.

    The weight is |H_k|**2, and in a multipath channel it is what makes this
    estimator usable at all.  Frequency-selective fading puts some pilots in
    nulls, where the residual angle is essentially uniform noise; giving those
    the same vote as a pilot sitting on a strong carrier drags the correction
    applied to EVERY data carrier in the symbol.  |H|**2 is the right weight
    because the angle variance of a pilot residual goes as 1/(|H|**2 * SNR), so
    inverse-variance weighting IS |H|**2 weighting.

    Same reasoning as the gain-weighting the constellation EVM tool needed, and
    the same failure mode if it is left out: an artefact of the estimator that
    reads as channel damage.

    Note this replaces an unconditional unit weight, not a gate: the old
    `abs(Hk) < 1e-9` skip is an absolute epsilon inherited from a float64
    reference and never fired in practice, so a pilot in a 40 dB null was
    previously counted at full strength.  The gate is kept only as a guard
    against a true zero, where the division is undefined.
    """
    ssum = 0.0
    csum = 0.0
    for pi in pilot_offsets:
        b    = bin_lo + pi
        Y    = sym_fd[b] / math.sqrt(n_fft)
        Hk   = H[pi]
        w    = abs(Hk) ** 2
        if w < 1e-18:
            continue
        r = Y / (Hk * pilot_sym)
        e = math.atan2(r.imag, r.real)
        ssum += w * math.sin(e)
        csum += w * math.cos(e)
    if ssum == 0.0 and csum == 0.0:
        return 0.0
    return float(math.atan2(ssum, csum))


def _estimate_cpe_ramp(sym_fd, n_fft, bin_lo, pilot_offsets, H, pilot_sym):
    """Pilot phase residual as (intercept, slope): phase(p) = a + b*p.

    WHY A SLOPE AND NOT JUST A CONSTANT.  Two sound cards never share a sample
    clock; consumer parts sit tens of ppm apart.  That offset is not a carrier
    rotation the scalar CPE can absorb -- it is a TIMING drift, and a timing
    error of d samples is a phase 2*pi*d*k/N, i.e. LINEAR IN CARRIER INDEX.  It
    also grows across the frame, so it is largest exactly where the frame has
    been running longest.

    At 50 ppm over a 33-symbol body (0.7 s) the drift is ~1.7 samples, which is
    ~1.8 rad of differential phase across a 400-8000 Hz carrier set.  A single
    common phase cannot represent that at all: measured over the simulated
    rooms at 48 kHz / 14 dB, a scalar CPE took a clean channel from 11.3% EVM
    at 0 ppm to 25.0% at 30 ppm and to total loss (0 of 7 frames) at 60 ppm,
    and on the mild `desk` room 30 ppm alone lost every frame.  30 ppm is an
    unremarkable pair of sound cards.

    The slope is estimated from DIFFERENTIAL phase between adjacent pilots --
    the angle of sum(w_j*w_j-1 * r_j * conj(r_j-1)) -- rather than by
    unwrapping and regressing.  Residual angles wrap at +-pi and an unwrapper
    is exactly the thing that fails silently under noise; the differential form
    has no wrap to undo as long as the step BETWEEN ADJACENT PILOTS is under
    pi, which at spacing 8 and n_fft 1024 means 64 samples of timing error --
    far more than the cyclic prefix tolerates anyway, so the estimator's range
    is not the binding constraint.

    Weights are |H|**2 throughout, for the reason given in _estimate_cpe.
    """
    ps, rs, ws = [], [], []
    for pi in pilot_offsets:
        Hk = H[pi]
        w  = abs(Hk) ** 2
        if w < 1e-18:
            continue
        Y = sym_fd[bin_lo + pi] / math.sqrt(n_fft)
        ps.append(pi)
        rs.append(Y / (Hk * pilot_sym))
        ws.append(w)
    if not rs:
        return 0.0, 0.0

    # Slope from adjacent pairs at the SMALLEST gap present.  Restricting to
    # one gap keeps every term the same angle-per-carrier so they can be summed
    # as complex numbers; mixing gaps would need each term divided by its own
    # gap first, which reintroduces the wrap this form exists to avoid.
    b = 0.0
    if len(ps) > 1:
        gaps = [ps[j] - ps[j - 1] for j in range(1, len(ps))]
        g    = min(gaps)
        acc  = 0j
        for j in range(1, len(ps)):
            if ps[j] - ps[j - 1] != g:
                continue
            acc += ws[j] * ws[j - 1] * rs[j] * rs[j - 1].conjugate()
        if acc != 0j:
            b = math.atan2(acc.imag, acc.real) / g

    # Intercept: derotate by the slope, then the weighted circular mean.
    ssum = csum = 0.0
    for pj, rj, wj in zip(ps, rs, ws):
        d = rj * complex(math.cos(b * pj), -math.sin(b * pj))
        e = math.atan2(d.imag, d.real)
        ssum += wj * math.sin(e)
        csum += wj * math.cos(e)
    a = 0.0 if (ssum == 0.0 and csum == 0.0) else math.atan2(ssum, csum)
    return a, b


# -
# Channel probe
# -

# Probe symbols needed before beta is trustworthy.  A 25-30 symbol fragment of
# a probe (a poll that caught the tail) gave beta 1.7-1.9 on the same link a
# full 170-symbol run graded at 0.25 -- H is too noisy to separate reverb from
# noise there, so beta is reported as None instead of poisoning the allocation.
_ISI_MIN_SYMS = 64

# Largest beta worth ACTING on.  The estimator recovers the impulse response by
# inverse-transforming H on the n_fft bin grid, so its delay axis wraps at
# n_fft/fs -- only 23.2 ms at 1024/44.1k.  A room tail of 100-400 ms is 4-17x
# that, and aliases UNIFORMLY into every delay tap: the "impulse response" comes
# out flat instead of decaying, and the late/early ratio saturates near or above
# 1 no matter what the room is really doing.
#
# Measured both sides of this (OTA, same room, laptops 1 ft apart):
#   coherence 0.996-0.998 -> beta 0.17-0.19, IR decays properly, and the ceiling
#                            predicted the delivered SINR within ~1 dB
#   coherence 0.944-0.952 -> beta 0.98-1.21 with a FLAT IR (~7% of energy in
#                            every 2 ms slice out to the wrap) = unmeasurable
# A beta over this bound therefore means "the tail is longer than this estimator
# can see", not "the channel carries nothing" -- and acting on it is worse than
# ignoring it, because a -0.5 dB ceiling zeroes the whole table (run
# 20260804_051433 fell from 8785 bps of measured capacity to 2928 bps of
# backed-off BPSK, and its handshake then failed outright).
_ISI_BETA_MAX = 0.5


def isi_ratio(H: np.ndarray, n_fft: int, bin_lo: int, cp_len: int,
              snr_lin: np.ndarray | None = None,
              pre_guard_frac: float = 0.125,
              taper_samples: int = 0) -> float:
    """Fraction of channel energy arriving too late to be covered by the CP.

    Returns beta = E_late / E_covered, an interference-to-signal ratio.  The
    probe measures H at the occupied carriers; an inverse FFT of that (zeros
    elsewhere) is a band-limited impulse response whose delay axis wraps at
    n_fft.  Energy landing more than cp_len after the main peak is not
    cyclically covered, so it leaks into the FOLLOWING symbol as ISI/ICI, and
    no equaliser or gain change can remove it -- it sets a hard SINR ceiling of
    -10*log10(beta).

    Why the probe cannot see this itself: it is ~150 IDENTICAL symbols, so late
    reflections reach periodic steady state and are absorbed into H as though
    they were channel gain.  The probe therefore grades a reverberant room as
    clean (17-20 dB measured OTA) while data symbols, which differ every symbol,
    ran at 4 dB SINR / 63% EVM on the same link (run 20260803_194332).

    `pre_guard_frac` counts a little energy BEFORE the peak as covered: a
    band-limited IR estimate rings symmetrically about the peak, and that
    pre-ringing is an artefact of measuring only n_occupied carriers, not a
    real pre-echo.

    `snr_lin` (per-carrier) lets additive noise be subtracted from the late
    energy.  Noise spreads evenly over all taps, so on a quiet-but-noisy
    channel it would otherwise masquerade as reverberation.

    `taper_samples` is the W-OFDM raised-cosine width, and it SHORTENS the
    guard.  _build_ola overlaps each symbol's rising taper with the previous
    symbol's falling one, so the first `taper_samples` of every cyclic prefix
    carry the tail of the symbol before it: the ISI-free guard is
    cp_len - taper_samples, not cp_len.  At the 1024/0.25 geometry with the
    2 ms default that is 160 samples rather than 256 -- 3.33 ms of echo budget
    instead of 5.33, i.e. 1.14 m of excess path instead of 1.83.

    Counting the full cp_len therefore made beta OPTIMISTIC by the taper width,
    and beta feeds derate_snr_lin and so the whole bit allocation.  Defaults to
    0 for callers that reason about a bare cyclic prefix (meshlib's selftest
    asserts against explicit tap positions and must not move).
    """
    n_occ = len(H)
    if n_occ == 0 or n_fft <= 0:
        return 0.0
    # Hann-taper the occupied band before transforming.  Only n_occ of the
    # n_fft/2 bins are measured, and that hard band edge leaks a sinc skirt
    # across EVERY delay tap: untapered, a perfectly echo-free single-tap
    # channel reads beta=0.025 (a phantom 16 dB ceiling in an anechoic room),
    # and a known 0.0625 echo reads 0.085.  The taper costs a slightly wider
    # main lobe -- irrelevant at the CP scale, ~12 samples against 256 -- and
    # buys an essentially unbiased estimate: 0.0002 clean, 0.0627 on the same
    # 0.0625 echo.
    full = np.zeros(n_fft // 2 + 1, dtype=complex)
    full[bin_lo:bin_lo + n_occ] = H * np.hanning(n_occ)
    h = np.fft.irfft(full, n=n_fft)
    e = h ** 2
    tot = float(e.sum())
    if tot <= 0:
        return 0.0

    guard = max(1, int(cp_len) - int(taper_samples))
    idx = (np.arange(n_fft) - int(np.argmax(e))) % n_fft
    pre = int(pre_guard_frac * cp_len)
    covered = (idx < guard) | (idx >= n_fft - pre)
    e_in = float(e[covered].sum())
    e_out = float(e[~covered].sum())

    if snr_lin is not None and len(snr_lin) == n_occ:
        # Expected noise energy, split across taps in proportion to tap count.
        p_sig = float(np.sum(np.abs(H) ** 2))
        p_noise = float(np.sum(np.abs(H) ** 2 / np.maximum(snr_lin, 1e-12)))
        if p_sig > 0:
            e_noise = tot * min(p_noise / (p_sig + p_noise), 0.9)
            e_out = max(e_out - e_noise * float(np.count_nonzero(~covered))
                        / n_fft, 0.0)
    return float(e_out / max(e_in, 1e-30))


def derate_snr_lin(snr_lin: np.ndarray, beta: float) -> np.ndarray:
    """Fold an ISI floor into a per-carrier linear SNR.

    ISI power scales with the carrier's own signal power (it IS the signal,
    arriving late), so it enters as a signal-proportional interference term:

        SINR = S / (N + beta*S)  =>  1/SINR = 1/SNR + beta

    which leaves a quiet-but-reverberant carrier capped at 1/beta no matter how
    strong it is.  That cap is the whole point: it is what the probe's
    coherently-averaged SNR cannot express.
    """
    if beta <= 0:
        return snr_lin
    return snr_lin / (1.0 + beta * snr_lin)


def _probe_snr_lin(H_arr: np.ndarray, estimator: str = "mean",
                   pctile: float = 15.0, n_blocks: int = 8) -> np.ndarray:
    """Per-carrier LINEAR SNR from stacked per-probe-symbol H estimates.

    H_arr is (S, n_occupied) complex -- S per-symbol channel estimates from
    one probe, each already divided by the known probe symbol.

    'mean' (legacy): SNR = 2|mean H|^2 / var(H).  The signal numerator uses
        the COHERENTLY-AVERAGED H (best case) while the noise is the whole-
        probe variance.  A carrier that is strong-and-steady for the 5s probe
        scores full marks even if it fades before data -- the acoustic channel
        is time-varying, so this over-promises (observed OTA 20260720: probe
        graded >=16dB, data came in at ~5dB / 56% EVM).

    'percentile': grade each carrier on its WORST time sub-block, not its
        average.  Split the probe into `n_blocks` contiguous sub-blocks; the
        channel is ~constant within a ~0.5s block, so the within-block
        variance is the pure additive-noise floor (NOT inflated by fading),
        while the spread of block-mean |H|^2 across blocks IS the fade.  Take
        the `pctile` (low) percentile of per-block SNR per carrier, so a
        carrier that fades during any part of the probe is graded down.  This
        predicts data-time robustness and makes the BAT negotiate a lower,
        survivable rate instead of a loose one.
    """
    S = H_arr.shape[0]
    nv = np.maximum(np.var(H_arr.real, axis=0, ddof=1) +
                    np.var(H_arr.imag, axis=0, ddof=1), 1e-12)
    if estimator == "percentile" and S >= 4:
        K = int(max(2, min(n_blocks, S // 2)))
        blocks = np.array_split(H_arr, K, axis=0)
        nvb = [np.var(b.real, axis=0, ddof=1) + np.var(b.imag, axis=0, ddof=1)
               for b in blocks if b.shape[0] >= 2]
        nv_add = np.maximum(np.mean(nvb, axis=0), 1e-12) if nvb else nv
        blk_snr = np.array([2.0 * np.abs(b.mean(axis=0)) ** 2 / nv_add
                            for b in blocks])              # (K, n_occupied)
        return np.maximum(np.percentile(blk_snr, pctile, axis=0), 1e-10)
    return np.maximum(2.0 * np.abs(H_arr.mean(axis=0)) ** 2 / nv, 1e-10)


class ChannelProbe:
    """
    Builds and analyzes the MEDLEY-equivalent channel probe.

    Probe audio layout:
        [ SC_preamble | CE_symbol | probe_sym * n_probe_syms ]

    SC gives timing.  CE gives a one-tap LS channel estimate.  The N probe
    symbols (all carrying an identical known BPSK pattern) are coherently
    averaged to obtain a low-noise H estimate; variance across the N estimates
    gives per-subcarrier noise power -> SNR.
    """

    def __init__(self, cfg: dict) -> None:
        self.n_fft          = cfg['n_fft']
        self.cp_len         = cfg['cp_len']
        self.bin_lo         = cfg['bin_lo']
        self.n_occupied     = cfg['n_occupied']
        self.symbol_samples = cfg['symbol_samples']
        self.sample_rate    = cfg['sample_rate']
        self.taper_samples  = cfg.get('taper_samples', 0)
        self.n_probe_syms   = cfg.get('n_probe_syms', 16)
        self.snr_estimator  = cfg.get('snr_estimator', 'mean')
        self.snr_pctile     = cfg.get('snr_pctile', 15.0)
        self.snr_blocks     = cfg.get('snr_blocks', 8)
        self.isi_derate     = float(cfg.get('isi_derate', 0.0))

        self.sc_td      = _make_sc_preamble(self.n_fft, self.cp_len,
                                             self.bin_lo, self.n_occupied)
        self.ce_known   = _known_bpsk(self.n_occupied, _SEED_CE)
        self.probe_known = _known_bpsk(self.n_occupied, _SEED_PROBE)
        self.n_meas_syms = int(cfg.get('n_probe_meas_syms', 0))
        # One independent known pattern per measurement symbol.
        self.meas_known = [_known_bpsk(self.n_occupied, _SEED_MEAS + j)
                           for j in range(self.n_meas_syms)]

    def build(self) -> np.ndarray:
        """Build probe audio: SC + CE + periodic block + measurement block.

        The two blocks exist for opposite reasons, and one waveform cannot serve
        both:

        PERIODIC block (identical symbols) -- what acquisition needs.  Every
        symbol carries the same pattern, so per-symbol H estimates agree and the
        coherence search separates a probe from noise by ~0.99 vs ~0.36.

        MEASUREMENT block (a different known pattern per symbol) -- what an
        HONEST SNR needs.  A repeated symbol lets room reverberation reach
        periodic steady state: the tail of each copy arrives in phase with the
        next and is absorbed into H as though it were channel gain.  The probe
        then grades a reverberant path as clean -- measured OTA at 18 dB per
        carrier on a path that delivered 4.5 dB, an ~15 dB over-promise that no
        margin could be tuned against because it varied with the room.

        Because consecutive measurement symbols are UNCORRELATED, each one's
        reverb tail lands on the next as genuine interference, exactly as a data
        symbol's does.  The existing variance-based estimator then picks it up
        for free: ISI shows up as per-symbol spread in H, which is what
        var(H) measures.  No new estimator, no assumption about the room, and
        nothing to calibrate -- the measurement simply experiences what the data
        will.

        Both blocks are reported, and the DIFFERENCE between them is a direct
        measurement of the ISI penalty (see estimate()).
        """
        T = self.taper_samples
        occ_list = [self.ce_known.copy()]
        for _ in range(self.n_probe_syms):
            occ_list.append(self.probe_known.copy())
        for j in range(self.n_meas_syms):
            occ_list.append(self.meas_known[j].copy())
        sig = np.concatenate([
            self.sc_td,
            _build_ola(occ_list, self.n_fft, self.bin_lo, self.cp_len, T),
        ])
        dur = len(sig) / self.sample_rate
        print(
            f"Probe: {self.n_probe_syms} periodic + {self.n_meas_syms} "
            f"non-periodic symbols  {dur:.2f}s",
            flush=True,
        )
        return sig

    def _align_meas(self, rx: np.ndarray, approx: int, n_try: int = 8) -> int:
        """Exact symbol timing for the measurement block.

        The periodic block does not need this -- identical symbols make any
        sub-symbol offset a pure phase ramp, which is why acquisition never
        bothered to resolve it and frame_start is only good to a fraction of a
        symbol (measured: -0.25 symbols).  A NON-periodic block cannot tolerate
        that: an offset makes the FFT window straddle two different symbols and
        the H estimate collapses to noise (123 dB -> -2 dB in test).

        MATCHED FILTER against the first measurement symbol, whose pattern is
        known exactly.  The peak sits on the direct path even in multipath, which
        is what the data receivers here already rely on for their preambles.

        Two simpler metrics were tried and both fail, instructively:

        * coherent energy |mean_j H_j|^2 is FLAT across the whole prefix
          (measured: 176.0 at every offset from -64 to +64) -- an offset inside
          the CP is harmless to the MAGNITUDE, which is exactly what a cyclic
          prefix is for.  Its argmax then lands wherever noise puts it, and at
          +64 the window spills into the next symbol and SNR drops 123 -> 23 dB.
          Flat metric, cliff-edged consequence.
        * cyclic-prefix correlation is sharp on a single-tap channel but goes
          AMBIGUOUS in multipath -- it can lock onto an echo.  With a 0.5
          amplitude echo only 140 samples into a 256-sample prefix it mis-locked
          and reported a false +15 dB ISI penalty on a channel whose echo the
          prefix fully covers.  Since multipath is the entire point of this
          measurement, a metric that fails there is useless here.

        Scoring on SNR would be wrong for a third reason: that is selecting on
        the quantity being measured.
        """
        N, CP, SYM = self.n_fft, self.cp_len, self.symbol_samples
        ref = _ofdm_td(self.meas_known[0], N, self.bin_lo, CP)
        span = SYM
        lo = max(0, approx - span)
        hi = min(len(rx), approx + span + len(ref))
        seg = np.asarray(rx[lo:hi], dtype=np.float64)
        if len(seg) < len(ref) + 8:
            return approx
        # normalised cross-correlation, so a loud echo cannot win on energy alone
        c = np.correlate(seg, np.asarray(ref, dtype=np.float64), mode="valid")
        e = np.sqrt(np.convolve(seg ** 2, np.ones(len(ref)), mode="valid")) + 1e-12
        k = int(np.argmax(np.abs(c) / e))
        # The reference from _ofdm_td is UNTAPERED, while the transmitted symbol
        # comes from _ofdm_td_windowed, which prepends `taper_samples` for the
        # overlap-add cross-fade.  The correlation therefore peaks exactly one
        # taper late (verified: +88 with taper=88), so correct for it.
        return lo + k - self.taper_samples

    def _measure_H(self, rx: np.ndarray, start: int, n_syms: int | None = None,
                   aligned: bool = False):
        """Per-symbol H from the NON-PERIODIC measurement block.

        `start` is the CP start of the first measurement symbol.  Each symbol is
        divided by its own known pattern, so the result is directly comparable
        with the periodic block's H -- but carries the ISI the data will see.
        """
        N, CP, SYM = self.n_fft, self.cp_len, self.symbol_samples
        if not aligned:
            start = self._align_meas(rx, start)
        out = []
        for j in range(n_syms if n_syms else self.n_meas_syms):
            g = start + j * SYM
            if g < 0 or g + CP + N > len(rx):
                break
            fd = np.fft.rfft(rx[g + CP:g + CP + N], n=N) / math.sqrt(N)
            Y = fd[self.bin_lo:self.bin_lo + self.n_occupied]
            out.append(Y / self.meas_known[j])
        return np.array(out) if len(out) >= 4 else None

    # -

    # - acquisition helpers -

    def _probe_H(self, rx: np.ndarray, start: int, nsyms: int):
        """Per-symbol H estimates for `nsyms` probe symbols starting at `start`.

        `start` is the CP start of a probe symbol.  Returns (nsyms, n_occupied)
        complex, or None if the window runs past the end of `rx`.
        """
        N, CP, SYM = self.n_fft, self.cp_len, self.symbol_samples
        if start < 0 or start + (nsyms - 1) * SYM + CP + N > len(rx):
            return None
        idx = (start + CP
               + np.arange(nsyms)[:, None] * SYM
               + np.arange(N)[None, :])
        fd  = np.fft.rfft(rx[idx], n=N, axis=1) / math.sqrt(N)
        return fd[:, self.bin_lo:self.bin_lo + self.n_occupied] / self.probe_known

    def _coherence(self, rx: np.ndarray, start: int, nsyms: int = 8) -> float:
        """How consistent are the per-symbol H estimates?  1.0 = identical.

        All probe symbols carry the SAME known BPSK, so a correct start makes
        every H[k] agree.  Power-normalised per carrier:

            c_k = |sum_j H_jk|^2 / (nsyms * sum_j |H_jk|^2)

        averaged over the strongest carriers.  Random noise scores ~1/nsyms.
        (An amplitude-normalised |mean|/mean|.| ratio is degenerate when one
        energetic window sits among near-silent ones -- a startup click
        followed by silence scored 0.998 and false-validated a probe at the
        wrong sample rate in the handshake rate-detection path.)
        """
        H = self._probe_H(rx, start, nsyms)
        if H is None:
            return 0.0
        num = np.abs(H.sum(axis=0)) ** 2
        den = nsyms * np.sum(np.abs(H) ** 2, axis=0) + 1e-18
        c_k = num / den
        mag = np.abs(H).mean(axis=0)
        k   = min(_COH_CARRIERS, len(mag))
        idx = np.argpartition(mag, -k)[-k:]
        return float(np.mean(c_k[idx]))

    def estimate(self, rx: np.ndarray, threshold: float = 0.25) -> dict:
        """
        Estimate per-subcarrier channel from received probe audio.

        Acquisition is a coherence search, NOT Schmidl-Cox.  SC false-locks on
        the audio-stream startup click (a decaying transient scores ~1 on any
        delay-correlator), and its min_gap then blocks the real preamble for a
        whole probe length.  Coherence separates probe from noise by ~0.99 vs
        ~0.36 on real captures, where SC managed 0.3 vs 0.5 -- the wrong way
        round.  `threshold` is kept for call compatibility and is used as the
        coherence gate.

        Exact timing is not required here: the probe is a periodic excitation,
        so a start offset only adds a phase ramp, and |H| (hence SNR) is
        unaffected.  We refine alignment anyway by maximising coherent energy.

        Returns
        -------
        dict with keys:
            valid       bool
            H           complex ndarray (n_occupied,)
            snr_db      float ndarray (n_occupied,)  -- SNR per subcarrier
            snr_lin     float ndarray (n_occupied,)
            n_syms      int -- probe symbols successfully averaged
            frame_start int
        """
        N, CP, SYM = self.n_fft, self.cp_len, self.symbol_samples
        rx = np.asarray(rx, dtype=np.float64)
        gate = max(float(threshold), _COH_GATE)

        span = (_COH_SCAN_SYMS - 1) * SYM + CP + N
        if len(rx) < span + SYM:
            return {'valid': False, 'error': 'too_short'}

        # -- 1. coarse scan: where is the probe? (vectorised H-grid) --
        step  = math.gcd(SYM, max(8, CP // 2))
        cands, coh = _grid_coherence_scan(
            rx, N, CP, self.bin_lo, self.n_occupied, self.probe_known,
            _COH_SCAN_SYMS, step, q=None, top_k=_COH_CARRIERS)
        if not len(cands):
            return {'valid': False, 'error': 'too_short'}
        best  = int(cands[int(np.argmax(coh))])
        peak  = float(coh.max())
        # Noise floor = low percentile of candidate coherence (positions that
        # missed the probe).  Reported as a diagnostic only; the absolute gate
        # is the discriminator.  A relative peak/floor test is wrong here: when
        # the probe fills most of the capture, most candidates land inside it
        # and score high, so the floor legitimately approaches the peak.
        floor = float(np.percentile(coh, 25))
        if peak < gate:
            return {'valid': False, 'error': 'sync_fail',
                    'coherence': round(peak, 3),
                    'noise_floor': round(floor, 3)}

        # -- 2. fine align: maximise coherent energy within +-CP --
        fine = np.arange(max(0, best - CP), best + CP, 4, dtype=int)
        energy = []
        for s in fine:
            H = self._probe_H(rx, int(s), _COH_SCAN_SYMS)
            energy.append(0.0 if H is None
                          else float(np.sum(np.abs(H.mean(axis=0)) ** 2)))
        best = int(fine[int(np.argmax(energy))])

        # -- 3. collect every probe symbol on the grid through `best` --
        # The coarse winner can sit anywhere inside the probe (all probe symbols
        # are identical, so coherence is flat across it).  Walk the whole grid
        # and keep only symbols whose H matches the local reference: that drops
        # the silence either side, the SC preamble, and the CE symbol (which
        # carries a different sequence and so scores ~0.1 here).
        Hr = self._probe_H(rx, best, _COH_SCAN_SYMS)
        Href = Hr.mean(axis=0)
        magr = np.abs(Href)
        kk   = min(_COH_CARRIERS, len(magr))
        strong = np.argpartition(magr, -kk)[-kk:]
        ref  = Href[strong] / (magr[strong] + 1e-12)

        first = best - (best // SYM) * SYM
        grid  = np.arange(first, len(rx) - CP - N, SYM, dtype=int)
        i_best = int(np.argmin(np.abs(grid - best)))

        H_all: list = []
        match: list = []
        for g in grid:
            h = self._probe_H(rx, int(g), 1)
            if h is None:
                H_all.append(None); match.append(0.0); continue
            h = h[0]
            hh = h[strong] / (np.abs(h[strong]) + 1e-12)
            H_all.append(h)
            match.append(float(np.abs(np.mean(hh * np.conj(ref)))))

        # contiguous run of matching symbols containing the winner
        lo_i = i_best
        while lo_i - 1 >= 0 and match[lo_i - 1] > _COH_MATCH:
            lo_i -= 1
        hi_i = i_best
        while hi_i + 1 < len(grid) and match[hi_i + 1] > _COH_MATCH:
            hi_i += 1

        H_list = [H_all[i] for i in range(lo_i, hi_i + 1)
                  if H_all[i] is not None][:self.n_probe_syms]
        frame_start = int(grid[lo_i])

        if len(H_list) < 2:
            return {'valid': False, 'error': 'too_few_probe_syms',
                    'coherence': round(peak, 3)}

        # Put BOTH blocks on ONE cyclic-prefix-aligned grid before measuring.
        # frame_start is only good to a fraction of a symbol (the periodic block
        # never needed better), and here it was 318 samples out -- past the
        # 256-sample CP -- so the periodic H estimates were straddling the
        # previous symbol and reading ~9 dB LOW on a clean channel.  Comparing a
        # misaligned periodic SNR against a properly aligned one measures the
        # misalignment, not the channel: the two blocks must differ only in their
        # excitation, which is the whole point of the comparison.
        # The delay must be measured on the MEASUREMENT block.  CP correlation
        # is degenerate in the periodic block: identical symbols make the signal
        # periodic with SYM, so rx[t:t+CP] matches rx[t+N:t+N+CP] at EVERY offset
        # and the estimator has nothing to lock to.  Only where consecutive
        # symbols differ does the correlation peak.
        dly = 0
        meas_at = None
        if self.n_meas_syms and hi_i + 1 < len(grid):
            m0 = int(grid[hi_i + 1])
            meas_at = self._align_meas(rx, m0)
            dly = meas_at - m0
            if abs(dly) < SYM:
                H_re = [self._probe_H(rx, int(grid[i]) + dly, 1)
                        for i in range(lo_i, hi_i + 1)]
                H_re = [h[0] for h in H_re if h is not None][:self.n_probe_syms]
                if len(H_re) >= 2:
                    H_list = H_re
                    frame_start = int(frame_start + dly)
            else:
                dly, meas_at = 0, m0

        H_arr  = np.array(H_list)
        H_mean = np.mean(H_arr, axis=0)
        snr_lin_per = _probe_snr_lin(H_arr, self.snr_estimator,
                                     self.snr_pctile, self.snr_blocks)
        snr_db_per  = 10.0 * np.log10(snr_lin_per + 1e-10)

        # -- the honest measurement: the NON-PERIODIC block --
        # The periodic run's match test stops exactly where the patterns change,
        # so the measurement block begins at the next grid position.
        snr_lin, snr_db = snr_lin_per, snr_db_per
        H_meas = None
        isi_penalty = None
        if meas_at is not None:
            # same grid, same alignment -- only the excitation differs
            H_meas = self._measure_H(rx, meas_at, aligned=True)
        if H_meas is not None:
            snr_lin_meas = _probe_snr_lin(H_meas, self.snr_estimator,
                                          self.snr_pctile, self.snr_blocks)
            snr_db_meas = 10.0 * np.log10(snr_lin_meas + 1e-10)
            # This is what the data will get, so this is what the allocator uses.
            snr_lin, snr_db = snr_lin_meas, snr_db_meas
            isi_penalty = round(float(np.median(snr_db_per)
                                      - np.median(snr_db_meas)), 1)

        # ISI ceiling.  Always MEASURED (it is the single most useful number
        # about a reverberant room and costs one FFT); it is APPLIED, if the
        # run asks, in BitAllocator.allocate -- where the SNR margin also lives,
        # because the two only compose correctly in one place (see there).
        # A short symbol run gives a noisy H and so a meaningless beta, hence
        # the symbol-count gate.
        beta = (isi_ratio(H_mean, self.n_fft, self.bin_lo, self.cp_len, snr_lin,
                          taper_samples=self.taper_samples)
                if len(H_list) >= _ISI_MIN_SYMS else None)
        # Never let the ceiling go missing quietly: with no beta the allocator
        # falls back to the flat snr_margin_db guess, and the only evidence used
        # to be a null in the record.
        isi_reason = None
        if beta is None:
            isi_reason = (f"only {len(H_list)} probe symbols, under the "
                          f"{_ISI_MIN_SYMS} needed for a trustworthy beta -- "
                          f"no reverberation ceiling measured, so the "
                          f"allocation rests on snr_margin_db alone")
            print(f"Probe: ISI ceiling NOT measured -- {isi_reason}", flush=True)

        meas_note = ""
        if isi_penalty is not None:
            meas_note = (f"  non-periodic SNR median "
                         f"{float(np.median(snr_db)):.1f}dB vs periodic "
                         f"{float(np.median(snr_db_per)):.1f}dB "
                         f"-> ISI penalty {isi_penalty:+.1f}dB "
                         f"({len(H_meas)} syms)")
        elif self.n_meas_syms:
            meas_note = "  (measurement block not found -- periodic SNR only)"

        isi_note = ""
        if beta is not None:
            wrap_ms = 1000.0 * self.n_fft / max(self.sample_rate, 1)
            if beta > _ISI_BETA_MAX:
                isi_note = (f"  ISI beta={beta:.3f} UNMEASURABLE (over "
                            f"{_ISI_BETA_MAX}: the tail outlasts this "
                            f"estimator's {wrap_ms:.1f}ms window and aliases "
                            f"flat) -- NOT applied")
            else:
                isi_note = (f"  ISI beta={beta:.3f} (ceiling "
                            f"{-10 * math.log10(max(beta, 1e-9)):.1f}dB"
                            f"{', APPLIED' if self.isi_derate > 0 else ', not applied'})")
        print(f"Probe acquired: coherence={peak:.3f} (noise floor {floor:.3f})  "
              f"start={frame_start} ({frame_start / self.sample_rate:.3f}s)  "
              f"{len(H_list)} symbols{meas_note}{isi_note}", flush=True)

        return {
            'valid': True,
            'H': H_mean,
            'snr_db': snr_db,
            'snr_lin': snr_lin,
            # the periodic block's view, kept for comparison: the gap between
            # the two IS the reverberation penalty, measured rather than modelled
            'snr_db_periodic': snr_db_per,
            'isi_penalty_db': isi_penalty,
            'n_meas_syms': (0 if H_meas is None else len(H_meas)),
            'isi_ratio': beta,
            'isi_reason': isi_reason,
            'isi_derate': self.isi_derate,
            'n_syms': len(H_list),
            'frame_start': frame_start,
            'coherence': round(peak, 3),
            'noise_floor': round(floor, 3),
        }

    def _estimate_sc_legacy(self, rx: np.ndarray, threshold: float = 0.25) -> dict:
        """Original Schmidl-Cox acquisition.  Retained for comparison only --
        it false-locks on stream startup transients; see estimate()."""
        M     = _sc_metric(rx, self.n_fft)
        # Probe frame: SC (unwindowed) + OLA(CE + n_probe_syms symbols)
        probe_min_gap = _frame_min_gap(
            self.symbol_samples,
            1 + self.n_probe_syms,   # CE + probes in OLA
            self.taper_samples,
            self.cp_len,
        )
        peaks = _find_sc_peaks(M, self.symbol_samples, probe_min_gap, threshold)
        if not peaks:
            return {'valid': False, 'error': 'sync_fail'}

        frame_start = peaks[0]
        SYM = self.symbol_samples
        N   = self.n_fft
        CP  = self.cp_len
        pos = frame_start + SYM  # skip SC preamble

        if pos + CP + N > len(rx):
            return {'valid': False, 'error': 'truncated_ce'}

        # CE symbol -> initial H estimate
        ce_fd = np.fft.rfft(rx[pos + CP : pos + CP + N], n=N)
        H_ce  = np.array([
            ce_fd[self.bin_lo + i] / math.sqrt(N) / (self.ce_known[i] or 1e-12)
            for i in range(self.n_occupied)
        ])
        pos += SYM

        # Accumulate per-probe-symbol H estimates
        H_list: list[np.ndarray] = []
        while len(H_list) < self.n_probe_syms and pos + CP + N <= len(rx):
            pfd = np.fft.rfft(rx[pos + CP : pos + CP + N], n=N)
            H_est = np.array([
                pfd[self.bin_lo + i] / math.sqrt(N) / (self.probe_known[i] or 1e-12)
                for i in range(self.n_occupied)
            ])
            H_list.append(H_est)
            pos += SYM

        if not H_list:
            # Fall back to CE only; assign conservative flat SNR
            return {
                'valid': True, 'H': H_ce,
                'snr_db': np.full(self.n_occupied, 15.0),
                'snr_lin': np.full(self.n_occupied, 10 ** 1.5),
                'n_syms': 0, 'frame_start': frame_start,
            }

        H_arr  = np.array(H_list)        # (n_syms, n_occupied) complex
        H_mean = np.mean(H_arr, axis=0)

        # Per-subcarrier noise variance (real + imag separately for unbiased)
        if len(H_list) >= 2:
            nv = (np.var(H_arr.real, axis=0, ddof=1) +
                  np.var(H_arr.imag, axis=0, ddof=1))
        else:
            # Single probe symbol: use CE vs probe cross-check
            nv = np.abs(H_list[0] - H_ce) ** 2 + 1e-12

        nv = np.maximum(nv, 1e-12)

        # SNR for unit-power data constellations:
        #   signal power  = |H[k]|^2   (H normalised against unit pilot)
        #   noise in H_est = sigma_n^2 * 2  (probe X has |X|=1/sqrt(2))
        #   sigma_n^2     = nv / 2
        #   SNR           = |H|^2 / (nv / 2) = 2|H|^2 / nv
        snr_lin = 2.0 * np.abs(H_mean) ** 2 / nv
        snr_db  = 10.0 * np.log10(snr_lin + 1e-10)

        return {
            'valid': True,
            'H': H_mean,
            'snr_db': snr_db,
            'snr_lin': snr_lin,
            'n_syms': len(H_list),
            'frame_start': frame_start,
        }


# -
# Bit allocator  (greedy / Levin-Campello for flat transmit power)
# -

class BitAllocator:
    """
    Assigns the highest supported bit count to each data subcarrier such that
    SNR[k] >= SNR_FLOOR_DB[b] + snr_margin_db.

    The pilot subcarrier SNR values in the input are skipped automatically;
    the output length equals n_data_carriers (pilots excluded).
    """

    def allocate(
        self,
        snr_db_occupied: np.ndarray,
        pilot_set:       set,
        n_occupied:      int,
        snr_margin_db:   float = 6.0,
        max_bits:        int   = 6,
        isi_beta:        float = 0.0,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        snr_db_occupied  SNR per occupied subcarrier (length n_occupied)
        pilot_set        set of pilot indices within occupied block
        n_occupied       total occupied subcarriers
        snr_margin_db    headroom above floor (dB)
        max_bits         maximum bits per carrier (1/2/4/6)
        isi_beta         measured ISI-to-signal ratio (0 = ignore); see
                         isi_ratio()

        The margin is SPENT FIRST and the ISI ceiling applied to what is left:

            usable = derate(snr / margin, beta)   and require usable >= floor

        Order matters and this is the only order that composes.  Applying both
        as independent dB penalties double-counts -- they answer the same
        question ("how much worse is data than the probe says") and, at a
        measured beta of 0.25, subtracting a 9 dB margin from an already-capped
        6 dB ceiling mutes every carrier and the link never comes up.  Spending
        the margin first keeps its real meaning (headroom against fade
        migration between probe and data) while leaving the ceiling intact:
        a carrier cannot be dragged below the ISI floor by a big margin,
        because the floor is not something margin can buy off.

        With isi_beta=0 this reduces EXACTLY to the previous behaviour.
        """
        levels = sorted([b for b in VALID_BIT_LEVELS if 0 < b <= max_bits],
                        reverse=True)
        m_lin = 10.0 ** (snr_margin_db / 10.0)
        bits = []
        for i in range(n_occupied):
            if i in pilot_set:
                continue
            lin = 10.0 ** (float(snr_db_occupied[i]) / 10.0) / m_lin
            if isi_beta > 0:
                lin = float(derate_snr_lin(np.array([lin]), isi_beta)[0])
            usable = 10.0 * math.log10(max(lin, 1e-12))
            assigned = 0
            for b in levels:
                if usable >= SNR_FLOOR_DB[b]:
                    assigned = b
                    break
            bits.append(assigned)
        return np.array(bits, dtype=np.int32)

    def summary(
        self,
        bits_per_bin:   np.ndarray,
        symbol_samples: int,
        sample_rate:    int,
    ) -> str:
        counts  = {b: int(np.sum(bits_per_bin == b)) for b in VALID_BIT_LEVELS}
        total   = int(np.sum(bits_per_bin))
        active  = int(np.sum(bits_per_bin > 0))
        bps     = total * sample_rate // symbol_samples
        avg     = total / active if active else 0.0
        return (
            f"active={active}  avg={avg:.2f}b/carrier  gross~{bps} bps  "
            f"[off={counts[0]} bpsk={counts[1]} qpsk={counts[2]} "
            f"16q={counts[4]} 64q={counts[6]}]"
        )


# -
# Empty-BAT policy (shared by modulator and demodulator)
# -

def bpsk_all_bat(n_data_carriers: int) -> np.ndarray:
    """BPSK on every data carrier -- the diagnostic table for a dead channel.

    A caller holding an all-off allocation and wanting to put something on the
    air anyway asks for this EXPLICITLY, and must hand the returned table to
    BOTH ends (write it to the shared BAT file / send it over the wire), not
    just to the transmitter.  See _reject_empty_bat for why the modem can no
    longer make that substitution on the caller's behalf.
    """
    return np.ones(int(n_data_carriers), dtype=np.int32)


def _reject_empty_bat(bits_per_ofdm_sym: int, who: str) -> None:
    """Refuse to build a modem around an all-off BAT.

    AdaptiveModulator used to rewrite an all-zero table to BPSK-on-every-carrier
    ("so the operator can see what arrived") while AdaptiveDemodulator had no
    matching rule -- it kept bits_per_ofdm_sym=0 and so data_syms_per_pkt=0.
    The two ends therefore computed DIFFERENT frame geometries from the same
    table, and the transfer was undecodable by construction: the receiver was
    not merely getting errors, it was not looking for the symbols being sent.
    Run 20260810_195049_ofdm_fdd hit exactly this -- a direction measured at
    0 bps still aired 1.5 KB of undecodable audio over the far end's uplink.

    Raising in BOTH classes is the fix rather than teaching the demodulator the
    same fallback, because the substitution is a POLICY choice and every caller
    that can reach here already makes it, better, with the context to do so:
    ofdm_fdd_node declines to transmit an empty direction, meshmodem_node
    rejects a one-way link outright (and MeshModulator.encode_packet raises --
    a guard the silent rewrite had rendered unreachable), and ofdm_decode wants
    the BPSK table WRITTEN OUT so its peer transmits with it.  A library that
    quietly picks one of those for them can only pick wrong.
    """
    if bits_per_ofdm_sym <= 0:
        raise ValueError(
            f"{who}: all-off BAT (0 bits per OFDM symbol) -- this channel was "
            f"graded as carrying nothing, so there is no frame geometry to "
            f"build. Decide at the call site: skip the transfer, or pass "
            f"bpsk_all_bat(n_data_carriers) to BOTH ends if you want a "
            f"diagnostic frame on the air."
        )


# -
# Adaptive modulator
# -

class AdaptiveModulator:
    """
    OFDM modulator with per-subcarrier bit loading.

    Superframe layout per RS-coded chunk:
        [ SC_preamble ]
        [ CE_symbol   ]
        [ SYNC_symbol ]
        [ data * sync_period ]
        [ SYNC_symbol ]
        [ data * sync_period ]
        ...  (until data_syms_per_pkt exhausted)

    SYNC symbols carry known BPSK on all occupied subcarriers (same sequence
    as the CE symbol) and allow the decoder to refresh H[k] estimates mid-frame.
    """

    PILOT_SYM: complex = (1 + 1j) / math.sqrt(2)

    def __init__(self, cfg: dict, bits_per_bin: np.ndarray) -> None:
        self.n_fft          = cfg['n_fft']
        self.cp_len         = cfg['cp_len']
        self.bin_lo         = cfg['bin_lo']
        self.n_occupied     = cfg['n_occupied']
        self.sample_rate    = cfg['sample_rate']
        self.symbol_samples = cfg['symbol_samples']
        self.taper_samples  = cfg.get('taper_samples', 0)
        self.frame_gap_samp = cfg.get('frame_gap_samples', 0)
        self.ecc_bytes      = cfg['ecc_bytes']
        self.chunk_bytes    = cfg['chunk_bytes']
        self.seq_bytes      = cfg['seq_bytes']
        self.pilot_offsets  = list(range(0, self.n_occupied, cfg['pilot_spacing']))
        self._pilot_set     = set(self.pilot_offsets)
        self.n_data_carriers = cfg['n_data_carriers']
        self.sync_period    = cfg.get('sync_period', 32)

        if len(bits_per_bin) != self.n_data_carriers:
            raise ValueError(
                f"bits_per_bin length {len(bits_per_bin)} "
                f"!= n_data_carriers {self.n_data_carriers}"
            )
        self.bits_per_bin      = np.array(bits_per_bin, dtype=np.int32)
        self.bits_per_ofdm_sym = int(np.sum(self.bits_per_bin))
        _reject_empty_bat(self.bits_per_ofdm_sym, 'AdaptiveModulator')

        self.rsc           = RSCodec(self.ecc_bytes)
        # Interleaved layout: equal codewords, depth derived from the BAT (see
        # _interleave_geometry).  Depth 1 reproduces the previous single-call
        # reedsolo layout exactly, so a link that needs no interleaving pays
        # nothing for it.
        (self.il_depth, self.il_k,
         self.payload_bytes) = _interleave_geometry(
            self.seq_bytes, self.chunk_bytes, self.ecc_bytes,
            self.bits_per_ofdm_sym)
        payload_bits       = self.payload_bytes * 8
        self.data_syms_per_pkt = math.ceil(payload_bits / self.bits_per_ofdm_sym)

        self.sc_td      = _make_sc_preamble(self.n_fft, self.cp_len,
                                             self.bin_lo, self.n_occupied)
        # Unit power, not the +-1/sqrt(2) the helper returns: see REF_SYM_BOOST.
        self.ce_known   = _known_bpsk(self.n_occupied, _SEED_CE) * REF_SYM_BOOST
        self.sync_known = self.ce_known.copy()   # same pattern, same level

        self._scale: float = 1.0
        self.total_frames:  int   = 0
        self.tx_signal:     np.ndarray | None = None

    # - public -

    def encode(self, input_str: str) -> np.ndarray:
        return self.encode_bytes(input_str.encode('utf-8'))

    def _frame_occ(self, seq_idx: int, chunk: bytes) -> list[np.ndarray]:
        """Occupied-carrier symbol list for one frame body:
        CE + interleaved SYNC + data symbols (RS-coded, scrambled)."""
        seq_b    = seq_idx.to_bytes(self.seq_bytes, 'big')
        enc      = rs_encode_interleaved(self.rsc, seq_b + chunk,
                                         self.il_depth, self.il_k,
                                         self.ecc_bytes)
        raw_bits = np.unpackbits(np.frombuffer(enc, dtype=np.uint8))
        needed   = self.data_syms_per_pkt * self.bits_per_ofdm_sym
        if len(raw_bits) < needed:
            raw_bits = np.concatenate(
                [raw_bits, np.zeros(needed - len(raw_bits), dtype=np.uint8)]
            )
        scr_bits = raw_bits ^ _prbs(len(raw_bits), _SEED_DATA)

        occ_list = [self.ce_known.copy()]
        bit_ptr  = 0
        for ds in range(self.data_syms_per_pkt):
            if self.sync_period > 0 and ds % self.sync_period == 0:
                occ_list.append(self.sync_known.copy())
            seg = scr_bits[bit_ptr : bit_ptr + self.bits_per_ofdm_sym]
            occ_list.append(self._data_occ(seg))
            bit_ptr += self.bits_per_ofdm_sym
        return occ_list

    def encode_bytes(self, data: bytes) -> np.ndarray:
        data = bytes(data)
        pad  = (-len(data)) % self.chunk_bytes
        data += b'\x00' * pad
        chunks = [data[i : i + self.chunk_bytes]
                  for i in range(0, len(data), self.chunk_bytes)]
        self.total_frames = len(chunks)

        gap    = (np.zeros(self.frame_gap_samp)
                  if self.frame_gap_samp > 0 else None)
        pieces: list[np.ndarray] = []

        for seq_idx, chunk in enumerate(chunks):
            occ_list = self._frame_occ(seq_idx, chunk)
            T     = self.taper_samples
            frame = np.concatenate([
                self.sc_td,
                _build_ola(occ_list, self.n_fft, self.bin_lo, self.cp_len, T),
            ])
            if gap is not None:
                pieces.append(gap)
            pieces.append(frame)

        if gap is not None:
            pieces.append(gap)

        sig  = np.concatenate(pieces)
        peak = np.max(np.abs(sig))
        self._scale    = 32700.0 / peak if peak > 0 else 1.0
        self.tx_signal = sig
        return sig

    def write_wav(self, filename: str) -> None:
        if self.tx_signal is None:
            raise RuntimeError("call encode() first")
        pcm = (self.tx_signal * self._scale).astype(np.int16)
        wavfile.write(filename, self.sample_rate, pcm)
        dur = len(self.tx_signal) / self.sample_rate
        print(f"Written {filename}  ({dur:.2f}s, {self.total_frames} frames)",
              flush=True)

    def write_config(self, cfg: dict, filename: str) -> None:
        out = dict(cfg)
        out['bits_per_bin']        = self.bits_per_bin.tolist()
        out['bits_per_ofdm_sym']   = self.bits_per_ofdm_sym
        out['data_syms_per_pkt']   = self.data_syms_per_pkt
        out['total_frames']        = self.total_frames
        out['scale_factor']        = float(self._scale)
        out['ce_known_re']         = self.ce_known.real.tolist()
        out['ce_known_im']         = self.ce_known.imag.tolist()
        Path(filename).write_text(json.dumps(out, indent=2))
        print(f"Written {filename}", flush=True)

    # - private -

    def _data_occ(self, bits: np.ndarray) -> np.ndarray:
        occ    = np.zeros(self.n_occupied, dtype=complex)
        data_i = 0
        bp     = 0
        for i in range(self.n_occupied):
            if i in self._pilot_set:
                occ[i] = self.PILOT_SYM
            else:
                b = int(self.bits_per_bin[data_i])
                if b > 0:
                    key = tuple(int(bits[bp + j]) for j in range(b))
                    occ[i] = _MOD_ENCODE[b][key]
                    bp += b
                else:
                    occ[i] = 0.0
                data_i += 1
        return occ


# -
# Adaptive demodulator
# -

class AdaptiveDemodulator:
    """
    OFDM demodulator matching AdaptiveModulator's superframe structure.

    Channel tracking:
      - CE symbol: initial LS estimate H[k] for all occupied bins
      - SYNC symbols (every sync_period data syms): refresh H[k]
      - Pilot subcarriers: per-symbol CPE correction between SYNCs
    """

    PILOT_SYM: complex = (1 + 1j) / math.sqrt(2)

    def __init__(self, cfg: dict, bits_per_bin: np.ndarray) -> None:
        self.n_fft           = cfg['n_fft']
        self.cp_len          = cfg['cp_len']
        self.bin_lo          = cfg['bin_lo']
        self.n_occupied      = cfg['n_occupied']
        self.sample_rate     = cfg['sample_rate']
        self.symbol_samples  = cfg['symbol_samples']
        self.ecc_bytes       = cfg['ecc_bytes']
        self.chunk_bytes     = cfg['chunk_bytes']
        self.seq_bytes       = cfg['seq_bytes']
        self.pilot_offsets   = list(range(0, self.n_occupied, cfg['pilot_spacing']))
        self._pilot_set      = set(self.pilot_offsets)
        self.n_data_carriers = cfg['n_data_carriers']
        self.sync_period     = cfg.get('sync_period', 32)
        self.taper_samples   = cfg.get('taper_samples', 0)
        self.total_frames    = cfg.get('total_frames')

        if len(bits_per_bin) != self.n_data_carriers:
            raise ValueError(
                f"bits_per_bin length {len(bits_per_bin)} "
                f"!= n_data_carriers {self.n_data_carriers}"
            )
        self.bits_per_bin      = np.array(bits_per_bin, dtype=np.int32)
        self.bits_per_ofdm_sym = int(np.sum(self.bits_per_bin))
        _reject_empty_bat(self.bits_per_ofdm_sym, 'AdaptiveDemodulator')

        self.rsc         = RSCodec(self.ecc_bytes)
        # Same derivation as the modulator, from the same cfg and the same
        # negotiated BAT -- moved below bits_per_ofdm_sym because the depth now
        # depends on it.  If these two ever disagree the frame length disagrees
        # too, which shows up as nothing decoding rather than as subtle errors.
        (self.il_depth, self.il_k,
         self.payload_bytes) = _interleave_geometry(
            self.seq_bytes, self.chunk_bytes, self.ecc_bytes,
            self.bits_per_ofdm_sym)
        payload_bits     = self.payload_bytes * 8
        self.data_syms_per_pkt = math.ceil(payload_bits / self.bits_per_ofdm_sym)
        self.ce_known    = _known_bpsk(self.n_occupied, _SEED_CE) * REF_SYM_BOOST
        self.sync_known  = self.ce_known.copy()
        self._sc_ref     = None      # lazy: matched filter for start refinement

    # - public -

    def _refine_frame_start(self, samples: np.ndarray, d: int) -> int:
        """Snap an SC-metric peak onto the true frame start by matched filter.

        `_sc_metric` only ever sees the metric, never the samples, so it cannot
        do better than the plateau the cyclic prefix creates: M[] is flat for
        ~cp_len samples and `_find_sc_peaks` takes the first sample above 95%
        of the plateau max.  Under noise the metric RAMPS into that plateau, so
        that crossing lands part-way in and the returned start is LATE.

        Late is the destructive direction.  Starting early inside the CP is
        harmless -- the prefix is cyclic, so it costs only a phase ramp the
        per-carrier equaliser already removes -- but starting late drags the
        NEXT symbol into the FFT window, which is ICI no equaliser can undo.

        Measured on run 20260811_031659 (ofdm_fdd, 5 data frames): SC peaks ran
        +159/+285/+345/+295 samples late against a cp_len of 512, and every
        frame failed RS with 44-49% EVM.  Correlating the known SC preamble
        around each peak recovered the true starts and all 5 frames decoded
        with a full checksum match, using the SAME channel estimator.

        Only ever moves the start within +/-cp_len and only to a position that
        correlates BETTER with the known preamble, so a capture that was
        already aligned is left where it was.
        """
        return self._sc_match(samples, d)[0]

    def _sc_match(self, samples: np.ndarray, d: int) -> tuple[int, float]:
        """(best start, normalised correlation score) near `d`.

        Split out of _refine_frame_start because the peak VALIDATOR needs the
        same correlation the refinement already computes -- see the `validate`
        argument of _find_sc_peaks.  Computing it once and caching keeps the
        false-lock guard free rather than doubling the matched-filter work.

        The score is a normalised cross-correlation, so it is in [0, 1] by
        Cauchy-Schwarz and comparable across capture levels.
        """
        if self._sc_ref is None:
            ref = _make_sc_preamble(self.n_fft, self.cp_len,
                                    self.bin_lo, self.n_occupied)
            ref = ref - ref.mean()
            n = float(np.linalg.norm(ref))
            self._sc_ref = (ref / n) if n > 0 else None
        ref = self._sc_ref
        if ref is None or len(samples) < len(ref):
            return d, 1.0
        lo = max(0, d - self.cp_len)
        hi = min(len(samples) - len(ref), d + self.cp_len)
        if hi <= lo:
            return d, 1.0
        seg = samples[lo: hi + len(ref)].astype(np.float64)
        num = np.correlate(seg, ref, mode="valid")
        eng = np.sqrt(np.convolve(seg ** 2, np.ones(len(ref)), mode="valid"))
        ncc = np.abs(num) / (eng + 1e-12)
        if not len(ncc):
            return d, 1.0
        i = int(np.argmax(ncc))
        return int(lo + i), float(ncc[i])

    def decode_samples(
        self,
        samples:        np.ndarray,
        threshold:      float = 0.25,
        stats_callback  = None,
    ) -> tuple[list[dict], dict[int, bytes]]:
        M      = _sc_metric(samples, self.n_fft)
        n_syncs_est = (math.ceil(self.data_syms_per_pkt / self.sync_period)
                       if self.sync_period > 0 and self.data_syms_per_pkt > 0 else 0)
        # Body syms in OLA = CE + data + SYNCs
        n_body = 1 + self.data_syms_per_pkt + n_syncs_est
        data_min_gap = _frame_min_gap(
            self.symbol_samples, n_body, self.taper_samples, self.cp_len
        )
        # Reject candidates that are not preamble-shaped BEFORE they can
        # suppress a real frame via min_gap (see _find_sc_peaks).  The match is
        # cached because the accepted peaks need exactly the same correlation
        # for start refinement a few lines below -- so the guard costs nothing.
        match: dict[int, tuple[int, float]] = {}

        def _validate(pos: int) -> bool:
            m = match.get(pos)
            if m is None:
                m = match[pos] = self._sc_match(samples, pos)
            return m[1] >= SC_NCC_MIN

        peaks  = _find_sc_peaks(M, self.symbol_samples, data_min_gap, threshold,
                                validate=_validate)
        stats:  list[dict]      = []
        chunks: dict[int, bytes] = {}

        for d in peaks:
            # Snap onto the true frame start before demodulating: the SC
            # metric's plateau leaves a systematic LATE bias that costs whole
            # frames.  See _refine_frame_start.
            d = match[d][0] if d in match else self._refine_frame_start(samples, d)
            st = self._decode_frame(samples, d)
            if st is None:
                continue
            seq = st['seq']
            if seq not in chunks:
                chunks[seq] = st.pop('_chunk')
                st['duplicate'] = False
            else:
                st.pop('_chunk', None)
                st['duplicate'] = True
            stats.append(st)
            print(f"FRAME_STAT:{json.dumps(st)}", flush=True)
            if stats_callback:
                stats_callback(st)

        return stats, chunks

    def decode_wav(
        self,
        wav_path:       str,
        threshold:      float = 0.25,
        gain_db:        float = 0.0,
        stats_out:      str | None = None,
        stats_callback  = None,
    ) -> tuple[list[dict], str | None]:
        sr, pcm = wavfile.read(wav_path)
        if sr != self.sample_rate:
            raise ValueError(f"WAV sr={sr} != config sr={self.sample_rate}")
        samples = pcm.astype(np.float32) / 32768.0
        if gain_db:
            samples *= 10 ** (gain_db / 20)
        stats, chunks = self.decode_samples(samples, threshold, stats_callback)
        text = self.reassemble(chunks)
        if stats_out and stats:
            Path(stats_out).write_text(json.dumps(stats, indent=2))
        return stats, text

    @staticmethod
    def reassemble(chunks: dict[int, bytes]) -> str | None:
        if not chunks:
            return None
        raw = b''.join(chunks[k] for k in sorted(chunks)).rstrip(b'\x00')
        return raw.decode('utf-8', errors='replace')

    # - private -

    def _demod_frame(self, samples: np.ndarray, frame_start: int):
        """The DSP half of a frame: samples in, descrambled pre-RS bytes out.

        Returns (raw_bytes, evm_avg, data_syms) or None.

        Split out of _decode_frame so the whole per-symbol FFT / channel
        estimate / CPE / equalise / demap chain can be swapped for the C core
        while RS decoding and the stats dict -- which are not DSP and not on
        the embedded critical path -- stay in one place for both backends.
        """
        if os.environ.get("OFDM_ACQ_BACKEND", "").lower() == "c":
            import ofdm_acq_c
            if ofdm_acq_c.available():
                return ofdm_acq_c.demod_frame_c(self, samples, frame_start)

        SYM  = self.symbol_samples
        N    = self.n_fft
        CP   = self.cp_len
        # SYNC is inserted before every sync_period-th data symbol (ds=0,16,32,...).
        # Count = ceil(data_syms / sync_period), NOT integer divide.
        n_syncs = (math.ceil(self.data_syms_per_pkt / self.sync_period)
                   if self.sync_period > 0 and self.data_syms_per_pkt > 0 else 0)
        total_body_syms = 1 + self.data_syms_per_pkt + n_syncs  # CE + data + SYNCs

        if frame_start + SYM * (total_body_syms + 1) > len(samples):
            return None

        pos = frame_start + SYM  # skip SC preamble

        # WINDOW PLACEMENT.  The FFT window starts at pos+CP -- the latest legal
        # position, so the whole prefix is spent on LATE echoes and none on
        # frame-start error.  Backing off a few samples into the prefix is the
        # textbook insurance (it costs only a phase ramp, which H absorbs
        # because CE is measured at the same offset), and it does NOT pay here.
        # Swept over the room profiles, 4 seeds, mean frame EVM / frames:
        #
        #   backoff   anechoic      desk        desk @ 30 ppm
        #     0       11.5% 32/32   32.2% 15/27   30.7%  9/24
        #    16       11.5% 32/32   32.8% 15/27   32.1%  8/24
        #    32       11.5% 32/32   32.4% 11/27   32.9% 11/24
        #    64       11.5% 32/32   34.9%  5/27   34.4%  9/24
        #    96       11.6% 32/32   (no frames)   35.7%  7/24
        #
        # Monotonically worse, and worst at 96 -- which is exactly
        # taper_samples.  That is the reason: _build_ola overlaps each symbol's
        # rising taper with the previous symbol's falling one, so the first
        # taper_samples of every prefix already carry the PREVIOUS symbol.
        # Backing off walks straight into that contamination.  _refine_frame_start
        # is already placing the window correctly; there is nothing to insure.
        ce_fd = np.fft.rfft(samples[pos + CP : pos + CP + N], n=N)
        H     = np.zeros(self.n_occupied, dtype=complex)
        for i in range(self.n_occupied):
            b    = self.bin_lo + i
            Y    = ce_fd[b] / math.sqrt(N)
            H[i] = Y / (self.ce_known[i] or 1e-12)
        pos += SYM

        # - data symbols (with interleaved SYNC) -
        # Modulator inserts a SYNC symbol before every (sync_period)-th data
        # symbol: at ds=0, ds=sync_period, ds=2*sync_period, ...
        # Within the body stream (excluding CE) that means symbol index
        # 0, sync_period+1, 2*(sync_period+1), ...
        # Detect with: body_sym_idx % (sync_period + 1) == 0
        all_bits:  list[np.ndarray] = []
        evm_acc  = 0.0
        data_cnt = 0
        body_sym = 0   # symbol counter within body (CE already consumed)

        for _ in range(self.data_syms_per_pkt + n_syncs):
            if pos + CP + N > len(samples):
                break
            sym_fd = np.fft.rfft(samples[pos + CP : pos + CP + N], n=N)
            pos   += SYM

            # Is this a SYNC symbol-  Pattern: SYNC, data*P, SYNC, data*P, ...
            block_size = self.sync_period + 1   # 1 SYNC + sync_period data
            is_sync    = (self.sync_period > 0 and body_sym % block_size == 0)
            body_sym  += 1

            if is_sync:
                # Refresh H from SYNC symbol
                H_new = np.zeros(self.n_occupied, dtype=complex)
                for i in range(self.n_occupied):
                    b    = self.bin_lo + i
                    Y    = sym_fd[b] / math.sqrt(N)
                    H_new[i] = Y / (self.sync_known[i] or 1e-12)
                if body_sym == 1:
                    # The FIRST SYNC is adjacent to the CE symbol and carries
                    # the IDENTICAL known sequence (both are _SEED_CE) -- the
                    # modulator emits CE then SYNC back to back.  Overwriting
                    # threw away a whole full-band reference symbol, one
                    # symbol old, every frame.  Averaging the two halves the
                    # estimator noise for free; nothing about the channel can
                    # have changed in 26 ms that a mid-frame SYNC would not
                    # also miss.
                    #
                    # Later SYNCs must still OVERWRITE: they exist precisely
                    # because the channel has moved since the last one, so
                    # averaging them with a stale estimate would smear the
                    # thing they are there to track.
                    H = 0.5 * (H + H_new)
                else:
                    H = H_new
                continue

            # Pilot phase correction: intercept AND slope.  The slope is the
            # sample-clock offset between the two sound cards; see
            # _estimate_cpe_ramp for why a scalar cannot stand in for it.
            cpe_a, cpe_b = _estimate_cpe_ramp(
                sym_fd, N, self.bin_lo, self.pilot_offsets, H, self.PILOT_SYM)
            ph   = cpe_a + cpe_b * np.arange(self.n_occupied)
            # SIGN: the residual r = Y/(H*pilot) has angle +cpe, so removing
            # it needs H_eff = H*e^{+j*cpe} -- eq = Y/H_eff then divides the
            # error out.  This read e^{-j*cpe} and so DOUBLED the phase error
            # every symbol instead of cancelling it.
            #
            # It survived because nothing ever presented it with a phase error
            # to get wrong: the simulated channel was a zero-phase magnitude
            # filter with no delay and no clock offset, and 2*0 == 0.  It only
            # became visible once the sim grew a sample-clock offset, where a
            # 30 ppm pair -- an unremarkable one -- lost every frame.
            corr = np.cos(ph) + 1j * np.sin(ph)

            # Equalise data subcarriers and demodulate
            sym_bits: list[int] = []
            eq_syms:  list[complex] = []
            ref_syms: list[complex] = []
            data_i = 0
            for i in range(self.n_occupied):
                if i in self._pilot_set:
                    continue
                b_val = int(self.bits_per_bin[data_i])
                data_i += 1
                if b_val == 0:
                    continue
                b_bin  = self.bin_lo + i
                Y      = sym_fd[b_bin] / math.sqrt(N)
                H_eff  = H[i] * corr[i]
                eq     = Y / (H_eff if abs(H_eff) > 1e-9 else 1.0)
                eq_syms.append(eq)

                pts, bts = _MOD_TABLES[b_val]
                idx = int(np.argmin(np.abs(eq - pts) ** 2))
                sym_bits.extend(bts[idx].tolist())
                ref_syms.append(pts[idx])

            if eq_syms:
                eq_arr  = np.array(eq_syms)
                ref_arr = np.array(ref_syms)
                err_pwr = float(np.mean(np.abs(eq_arr - ref_arr) ** 2))
                ref_pwr = float(np.mean(np.abs(ref_arr) ** 2))
                evm_acc += math.sqrt(err_pwr / (ref_pwr + 1e-20))

            all_bits.append(np.array(sym_bits, dtype=np.uint8))
            data_cnt += 1

        if data_cnt == 0:
            return None

        scr_bits = np.concatenate(all_bits)[: self.payload_bytes * 8]
        raw_bits = scr_bits ^ _prbs(len(scr_bits), _SEED_DATA)
        nb       = (len(raw_bits) // 8) * 8
        return np.packbits(raw_bits[:nb]).tobytes(), evm_acc / data_cnt, data_cnt

    def _decode_frame(self, samples: np.ndarray, frame_start: int) -> dict | None:
        SYM  = self.symbol_samples
        n_syncs = (math.ceil(self.data_syms_per_pkt / self.sync_period)
                   if self.sync_period > 0 and self.data_syms_per_pkt > 0 else 0)
        total_body_syms = 1 + self.data_syms_per_pkt + n_syncs

        demod = self._demod_frame(samples, frame_start)
        if demod is None:
            return None
        raw_byt, evm_avg, _data_cnt = demod
        snr_evm = -20 * math.log10(evm_avg + 1e-9)

        try:
            dec, err_count = rs_decode_interleaved(
                self.rsc, bytes(raw_byt[: self.payload_bytes]),
                self.il_depth, self.seq_bytes + self.chunk_bytes)
        except Exception:
            return None

        if len(dec) < self.seq_bytes:
            return None

        seq   = int.from_bytes(dec[: self.seq_bytes], 'big')
        chunk = dec[self.seq_bytes :]

        ecc_cap  = self.ecc_bytes // 2
        ecc_hroom = max(0.0, (ecc_cap - err_count) / ecc_cap * 100)
        rms_in   = float(np.sqrt(np.mean(
            samples[frame_start : frame_start + SYM * (total_body_syms + 1)] ** 2
        )))
        rms_db   = 20 * math.log10(rms_in + 1e-9)

        active    = int(np.sum(self.bits_per_bin > 0))
        SYM_total = SYM * (total_body_syms + 1)
        gross_bps = int(self.bits_per_ofdm_sym * self.sample_rate / SYM)
        net_bps   = int(self.chunk_bytes * 8 * self.sample_rate / SYM_total)

        return {
            'seq':              seq,
            'frame_start':      frame_start,
            'err_count':        err_count,
            'ecc_capacity':     ecc_cap,
            'ecc_headroom_pct': round(ecc_hroom, 1),
            'evm_pct':          round(evm_avg * 100, 2),
            'snr_evm_db':       round(snr_evm, 1),
            'active_carriers':  active,
            'avg_bits_per_bin': round(float(np.sum(self.bits_per_bin)) / active, 2)
                                if active else 0,
            'input_rms_db':     round(rms_db, 1),
            'gross_bps':        gross_bps,
            'net_bps':          net_bps,
            '_chunk':           chunk,
        }


# -
# Robust bootstrap link  (in-band back-channel, no negotiation required)
# -

_SEED_RL_PRE  = 0x51C7    # RobustLink carrier PN pattern seed
_SEED_RL_PN   = 0x2E8B    # RobustLink per-symbol payload sign PRBS
# Preamble per-symbol sign key: Barker-7 (+1 pad) -- low shifted autocorrelation
# so a partially-overlapping window can't score near a true match.
_RL_Q         = np.array([1, 1, 1, -1, -1, 1, -1, 1], dtype=float)
_RL_N_PRE     = len(_RL_Q)  # preamble symbols (sign-matched acquisition)
_RL_RESYNC    = 24        # refresh H every this many payload bits
_RL_N_LEAD    = 22        # plain lead-in symbols (~0.6s) absorbing speaker
                          # wake-up / mic-AGC onset ramps: the first OTA
                          # handshake's BAT_ACK arrived ~5 dB quiet for its
                          # first 0.5s and the ramp ate the preamble
_RL_GATE      = 0.60      # acquisition gate (that same ramped preamble scored
                          # 0.688 -- 0.70 rejected a recoverable frame)
_RL_CLUSTERS  = 3         # distinct hit clusters to attempt (earliest first)

# BAT values {0,1,2,4,6} <-> 3-bit codes, for compact back-channel transport.
_BAT_LEVELS = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4}
_BAT_INV    = {v: k for k, v in _BAT_LEVELS.items()}


def bat_to_bytes(bat: np.ndarray) -> bytes:
    """Pack a bit-allocation table (values in {0,1,2,4,6}) to bytes, 3 bits each."""
    codes = np.array([_BAT_LEVELS.get(int(b), 0) for b in bat], dtype=np.uint8)
    bits  = np.zeros(len(codes) * 3, dtype=np.uint8)
    bits[0::3] = (codes >> 2) & 1
    bits[1::3] = (codes >> 1) & 1
    bits[2::3] = codes & 1
    return np.packbits(bits).tobytes()


def bat_wire_carriers(data: bytes) -> tuple[int, int]:
    """Carrier count a packed BAT can represent: (min, max).

    The wire format carries no length -- 3 bits per carrier, zero-padded to a
    byte -- so a table packed for one geometry unpacks silently against another.
    Callers that know their expected geometry should check it against this.
    """
    hi = len(data) * 8 // 3
    return (max(0, (len(data) * 8 - 7 + 2) // 3), hi)


def bytes_to_bat(data: bytes, n_carriers: int, *, strict: bool = False,
                 what: str = "BAT") -> np.ndarray:
    """Inverse of bat_to_bytes for `n_carriers` carriers.

    Out-of-range `n_carriers` means the sender and receiver disagree about the
    carrier geometry: the table gets truncated (or zero-padded) and every
    carrier ends up running an allocation measured for a DIFFERENT frequency.
    That went unnoticed for a whole OTA campaign (see meshlib.mesh_cfg), so it
    is now at least loud, and fatal where the caller can assert its geometry.
    """
    lo, hi = bat_wire_carriers(data)
    if data and not (lo <= n_carriers <= hi):
        msg = (f"{what}: wire table holds {lo}-{hi} carriers but this node "
               f"expects {n_carriers} -- the two ends disagree about the "
               f"carrier geometry (band / n_fft / cp_ratio)")
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg}", flush=True)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[: n_carriers * 3]
    if len(bits) < n_carriers * 3:
        bits = np.concatenate([bits, np.zeros(n_carriers * 3 - len(bits), np.uint8)])
    codes = (bits[0::3] << 2) | (bits[1::3] << 1) | bits[2::3]
    return np.array([_BAT_INV.get(int(c), 0) for c in codes], dtype=np.int32)


def _rl_groups(n_occupied: int, G: int) -> list:
    """Carrier index sets for the G payload groups -- INTERLEAVED (stride G).

    Contiguous blocks (np.array_split, the original split) fail on exactly the
    links that need the speed: acoustic fades are contiguous in FREQUENCY, so a
    single null can swallow a whole block and leave that group's bit with no
    live carrier at all -- unrecoverable at any SNR, and one dead bit kills the
    frame.  Measured on the fragmented table from run 20260729_212926 (85/237
    carriers active), G=8 contiguous gives per-group live counts
    [15,22,7,17,14,1,6,3]: one group holds a SINGLE carrier.  Striding the same
    groups across the band spreads every fade over all of them --
    [7,13,13,11,10,10,9,12], worst group 7 carriers = 8.5dB of MRC instead of
    0dB.  On an already-flat table the two are equivalent (worst group 23 vs 22
    carriers on that run's healthy direction), so the stride costs nothing
    where it is not needed.

    Encoder and decoder MUST agree, so both take their groups from here.
    """
    if G <= 1:
        return [np.arange(n_occupied)]
    return [np.arange(g, n_occupied, G) for g in range(G)]


class RobustLink:
    """Fixed, negotiation-free OFDM link used to carry the measured BAT back to
    the far end (the in-band back-channel bootstrap).

    The two acoustic directions have near-disjoint usable bands (a good speaker
    favours low frequencies, a tiny laptop speaker only passes 3-4 kHz), and the
    transmitter does not yet know its outgoing channel -- so a fixed sub-band
    mode cannot work.  Instead every payload BIT is sent as BPSK on ALL occupied
    data carriers (full-band repetition); the receiver does maximal-ratio
    combining weighted by the channel H it just measured in the probe, so
    whichever carriers are good for that direction carry the bit.  1 bit per
    OFDM symbol -- slow, but decodable at a few dB SNR without any prior rate
    agreement.  RS(parity) cleans up residual bit errors.

    Frame layout:
        [ SC preamble | N_PRE sign-keyed known symbols | payload symbols ]
    A known re-sync symbol is reinserted every _RL_RESYNC payload bits to
    track slow phase drift.

    Symbol structure (both design points are load-bearing):
    - Every symbol is the PN `known` carrier pattern times a per-symbol sign.
      The PN pattern keeps PAPR low (an all-ones spectrum is an impulse in
      time).  Payload sign: s_i = (1-2*bit_i)*(1-2*pn_i) with pn a PRBS, so a
      run of equal bits never produces identical consecutive symbols.
    - The preamble symbols carry a KNOWN sign sequence q_j, not a constant.
      Identical repeated symbols would be cyclically continuous (CP makes each
      symbol periodic), so timing within such a run is unobservable -- the
      coherence lock drifted anywhere inside it and windows straddled symbol
      boundaries (observed offline).  Sign-matched acquisition against q peaks
      only at the true grid alignment AND resolves the absolute sign of H.
    """

    def __init__(self, cfg: dict, ecc_bytes: int = 10,
                 n_groups: int = 1) -> None:
        self.n_fft         = cfg['n_fft']
        self.cp_len        = cfg['cp_len']
        self.bin_lo        = cfg['bin_lo']
        self.n_occupied    = cfg['n_occupied']
        self.sample_rate   = cfg['sample_rate']
        self.symbol_samples = cfg['symbol_samples']
        self.taper_samples = cfg.get('taper_samples', 0)
        self.ecc_bytes     = ecc_bytes
        self.rsc           = RSCodec(ecc_bytes)
        self.known   = _known_bpsk(self.n_occupied, _SEED_RL_PRE)
        self.q       = _RL_Q
        # Spreading control.  n_groups=1 is the bootstrap mode described above:
        # one bit smeared over every occupied carrier, ~24dB of processing gain,
        # usable with no knowledge of the outgoing channel.  n_groups=G splits
        # the occupied band into G contiguous groups carrying one bit each, so
        # the frame runs G times faster with spreading (and gain) divided by G.
        # Only worth using once the transmitter KNOWS its outgoing table --
        # i.e. after the peer has told it (see ofdm_handshake: every message
        # after BAT_REPORT qualifies).  Acquisition is untouched: the preamble
        # and re-sync symbols stay full-band, so the OTA-proven q-keyed lock
        # behaves identically at any G.
        self.n_groups = max(1, int(n_groups))
        self._grp = _rl_groups(self.n_occupied, self.n_groups)
        # Telemetry from the most recent decode (EVM etc); see _walk_bits.
        self.last_stats: dict | None = None

    # - framing helpers -

    def _bytes_to_bits(self, payload: bytes) -> tuple[np.ndarray, int]:
        enc = bytes(self.rsc.encode(bytearray(payload)))
        hdr = len(payload).to_bytes(2, 'big')            # length prefix (2B, plain)
        allb = hdr + enc
        return np.unpackbits(np.frombuffer(allb, dtype=np.uint8)), len(allb)

    @staticmethod
    def _sym_pn(n: int) -> np.ndarray:
        """Per-symbol PRBS sign sequence (+1/-1), one entry per payload bit."""
        return 1.0 - 2.0 * _prbs(n, _SEED_RL_PN).astype(float)

    def encode(self, payload: bytes) -> np.ndarray:
        """Build back-channel audio carrying `payload` bytes.

        Frame: [lead-in * N_LEAD | q-preamble * N_PRE | payload symbols].
        The lead-in is plain `known` symbols -- pure wake-up energy for
        powered speakers / mic AGC, invisible to acquisition (a constant-sign
        run scores ~(sum q)^2/N_PRE^2 = 1/16 under the Barker key).
        """
        bits, _ = self._bytes_to_bits(payload)
        G = self.n_groups
        n_sym = -(-len(bits) // G)                        # ceil
        pn = self._sym_pn(n_sym * G)
        T = self.taper_samples
        occ_list = [self.known.copy() for _ in range(_RL_N_LEAD)]
        occ_list += [self.q[j] * self.known for j in range(_RL_N_PRE)]
        for s in range(n_sym):
            # re-sync cadence counts SYMBOLS (identical to the old bit count
            # at G=1) so drift tracking stays tied to elapsed time, not bits
            if s > 0 and s % _RL_RESYNC == 0:
                occ_list.append(self.known.copy())        # periodic re-sync
            # match `known`'s dtype: it is complex128 (all-real for BPSK today,
            # but coercing to float would silently drop any future phase)
            occ = np.zeros(self.n_occupied, dtype=self.known.dtype)
            for g in range(G):
                k = s * G + g
                b = int(bits[k]) if k < len(bits) else 0
                sl = self._grp[g]
                occ[sl] = (1.0 - 2.0 * b) * pn[k] * self.known[sl]
            occ_list.append(occ)
        sig = _build_ola(occ_list, self.n_fft, self.bin_lo, self.cp_len, T)
        print(f"RobustLink: {len(payload)}B payload -> {len(bits)} bits"
              f"{'' if G == 1 else f' x{G} groups'}, "
              f"{len(sig)/self.sample_rate:.2f}s", flush=True)
        return sig

    # - acquisition -

    def _block_H(self, rx: np.ndarray, start: int) -> np.ndarray | None:
        """Per-symbol H estimates for the _RL_N_PRE preamble windows at `start`.

        Returns (_RL_N_PRE, n_occupied) complex = fd/known, or None if the
        block runs past the end of `rx`.
        """
        N, CP, SYM = self.n_fft, self.cp_len, self.symbol_samples
        if start < 0 or start + (_RL_N_PRE - 1) * SYM + CP + N > len(rx):
            return None
        idx = (start + CP
               + np.arange(_RL_N_PRE)[:, None] * SYM
               + np.arange(N)[None, :])
        fd = np.fft.rfft(rx[idx], n=N, axis=1) / math.sqrt(N)
        return fd[:, self.bin_lo:self.bin_lo + self.n_occupied] / self.known

    def _q_coherence(self, rx: np.ndarray, start: int) -> float:
        """Sign-matched, power-normalised coherence:

            c_k = |sum_j q_j H_jk|^2 / (N_PRE * sum_j |H_jk|^2)

        averaged over the strongest carriers.  1.0 only when all N_PRE windows
        agree under the q key; a single energetic window among silent ones
        scores 1/N_PRE (an amplitude-normalised mean is trivially coherent in
        that degenerate case -- it false-locked on the zero padding offline)."""
        Hb = self._block_H(rx, start)
        if Hb is None:
            return 0.0
        Hq  = Hb * self.q[:, None]
        num = np.abs(Hq.sum(axis=0)) ** 2
        den = _RL_N_PRE * np.sum(np.abs(Hb) ** 2, axis=0) + 1e-18
        c_k = num / den
        mag = np.abs(Hb).mean(axis=0)
        k   = min(_COH_CARRIERS, len(mag))
        idx = np.argpartition(mag, -k)[-k:]
        return float(np.mean(c_k[idx]))

    def decode(self, rx: np.ndarray, groups=None) -> bytes | None:
        """Recover payload bytes from received back-channel audio.

        Acquisition: vectorised sign-matched scan on the q-keyed preamble
        (the match only holds while every window stays inside its own symbol,
        so the peak is ~CP wide).  Hit clusters are attempted EARLIEST FIRST
        -- the preamble always precedes the payload, so a chance
        sign-collision inside the payload (an 8-symbol key has only 8 bits of
        distinctiveness) can never pre-empt the true lock -- and a cluster
        that fails the RS decode falls through to the next one.

        MRC weights come from the frame's own preamble only.  Deliberately no
        external H hint: an H measured in a different capture carries a
        different timing-offset phase ramp, and blending the two is
        destructive per carrier.
        """
        N, CP, SYM = self.n_fft, self.cp_len, self.symbol_samples
        rx = np.asarray(rx, dtype=np.float64)

        span = (_RL_N_PRE - 1) * SYM + CP + N
        if len(rx) < span + SYM:
            return None
        step = math.gcd(SYM, max(8, CP // 4))
        cands, coh = _grid_coherence_scan(
            rx, N, CP, self.bin_lo, self.n_occupied, self.known,
            _RL_N_PRE, step, q=self.q, top_k=_COH_CARRIERS)
        if not len(cands):
            return None
        hit_idx = np.where(coh >= _RL_GATE)[0]

        tried = 0
        pos_i = 0
        while tried < _RL_CLUSTERS and pos_i < len(hit_idx):
            h0 = int(hit_idx[pos_i])
            h1 = min(len(cands), h0 + max(1, (2 * SYM) // step))
            best = int(cands[h0 + int(np.argmax(coh[h0:h1]))])
            fine = np.arange(max(0, best - step), best + step, 4, dtype=int)
            fcoh = np.array([self._q_coherence(rx, int(s)) for s in fine])
            best = int(fine[int(np.argmax(fcoh))])

            # Acquisition is independent of the payload grouping, so try every
            # candidate spreading at THIS lock rather than re-scanning per mode
            # -- the coherence scan dominates decode cost, and re-running it
            # per group doubled per-poll CPU for nothing.
            for g in (groups or (self.n_groups,)):
                self.n_groups = g
                self._grp = _rl_groups(self.n_occupied, g)
                out = self._walk_bits(rx, best)
                if out is not None:
                    if self.last_stats is not None:
                        # Separates the two ways a BAT exchange fails: a low
                        # coherence means acquisition never locked, a high
                        # coherence with bad EVM means it locked and the
                        # channel was too weak to carry the payload.
                        self.last_stats["coherence"] = round(
                            float(fcoh[int(np.argmax(fcoh))]), 3)
                        self.last_stats["frame_start"] = int(best)
                    return out
            tried += 1
            # advance past this cluster to the next distinct hit
            nxt = cands[h0] + 2 * SYM
            while pos_i < len(hit_idx) and cands[hit_idx[pos_i]] < nxt:
                pos_i += 1
        return None

    def _walk_bits(self, rx: np.ndarray, best: int) -> bytes | None:
        """Demodulate a frame assuming its q-preamble starts at `best`."""
        N, CP, SYM = self.n_fft, self.cp_len, self.symbol_samples
        Hb = self._block_H(rx, best)
        if Hb is None:
            return None
        H = (Hb * self.q[:, None]).mean(axis=0)

        # -- walk payload symbols, MRC-combine, re-sync periodically --
        # Mirror the encoder: a re-sync known symbol precedes bit i whenever
        # i > 0 and i % _RL_RESYNC == 0.  Payload bins carry
        # H_ch * (1-2b)*pn_i * known, so divide by `known`, MRC against H,
        # and de-rotate the per-symbol PN sign.
        pos = best + _RL_N_PRE * SYM
        bits: list[int] = []
        G = self.n_groups
        w = np.conj(H)

        def _wnorms(Hv):
            return [float(np.sum(np.abs(Hv[sl]) ** 2)) + 1e-12
                    for sl in self._grp]

        wn = _wnorms(H)
        cap = 16 + self.ecc_bytes * 8 + 4096
        pn  = self._sym_pn(cap + G)
        # Soft-decision telemetry.  After MRC the decision variable is
        # normalised: y is already divided by `known`, so y = H*(+-1)*pn and
        # r = sum(conj(H)*y)/sum(|H|^2) = (+-1)*pn exactly in the noiseless
        # case.  De-rotating by pn therefore lands on +-1.0, which makes the
        # residual a true EVM against a unit reference -- no scaling needed.
        # Kept PER GROUP because that is the diagnostic that matters once the
        # band is split: a starved group shows up here long before it costs a
        # frame, and it is invisible in an aggregate figure.
        g_err2 = [0.0] * G                                # sum |d - decision|^2
        g_n    = [0]   * G
        g_pow0 = list(wn)                                 # acquisition-time |H|^2
        margin_min = float("inf")
        i = 0                                             # symbol index
        while pos + CP + N <= len(rx) and len(bits) < cap:
            if i > 0 and i % _RL_RESYNC == 0:             # consume re-sync symbol
                fd = np.fft.rfft(rx[pos + CP:pos + CP + N], n=N) / math.sqrt(N)
                Hs = fd[self.bin_lo:self.bin_lo + self.n_occupied] / self.known
                H  = 0.5 * H + 0.5 * Hs                   # track slow drift
                w  = np.conj(H); wn = _wnorms(H)
                pos += SYM
                if pos + CP + N > len(rx):
                    break
            fd = np.fft.rfft(rx[pos + CP:pos + CP + N], n=N) / math.sqrt(N)
            y  = fd[self.bin_lo:self.bin_lo + self.n_occupied] / self.known
            for g in range(G):                            # one bit per group
                sl = self._grp[g]
                r  = np.sum(w[sl] * y[sl]) / wn[g]        # maximal-ratio combine
                d  = r * pn[i * G + g]                    # de-rotate symbol PN
                s  = d.real
                bits.append(0 if s > 0 else 1)
                dec = 1.0 if s > 0 else -1.0              # ideal constellation pt
                g_err2[g] += float(abs(d - dec) ** 2)
                g_n[g]    += 1
                if abs(s) < margin_min:
                    margin_min = abs(s)
            i   += 1
            pos += SYM

        # Publish telemetry even when the frame goes on to fail RS: a caller
        # debugging a failed BAT exchange needs the EVM of the attempt that
        # failed.  decode() returns on the first success, so whatever is here
        # when it returns belongs to the frame it returned.
        tot_n = sum(g_n)
        self.last_stats = {
            "n_groups":   G,
            "n_bits":     tot_n,
            "evm_pct":    (100.0 * math.sqrt(sum(g_err2) / tot_n)) if tot_n else None,
            "evm_pct_per_group": [
                round(100.0 * math.sqrt(e / n), 1) if n else None
                for e, n in zip(g_err2, g_n)],
            "margin_min": (None if margin_min == float("inf")
                           else round(margin_min, 4)),
            # Per-group MRC weight sum, in dB relative to the strongest group.
            # A group far below 0 dB is one the interleave failed to feed.
            "group_power_db": [
                round(10.0 * math.log10(p / max(g_pow0)), 1) if p > 0 else None
                for p in g_pow0],
        }

        if len(bits) < 16:
            return None
        # length prefix (first 2 bytes) tells us how many payload bytes to expect
        hdr_bits = np.array(bits[:16], dtype=np.uint8)
        plen = int.from_bytes(np.packbits(hdr_bits).tobytes(), 'big')
        need_bytes = plen + self.ecc_bytes
        need_bits  = 16 + need_bytes * 8
        if plen <= 0 or plen > 4096 or len(bits) < need_bits:
            return None
        body = np.array(bits[16:need_bits], dtype=np.uint8)
        raw  = np.packbits(body).tobytes()[:need_bytes]
        try:
            dec, _, _ = self.rsc.decode(bytearray(raw))
            return bytes(dec)
        except Exception:
            return None


# -
# Modem  (orchestration)
# -

class Modem:
    """
    Full ADSL2-inspired OFDM modem.

    TX flow
    -------
        modem = Modem(cfg)
        probe_audio = modem.build_probe()
        # play probe; decoder measures channel and writes ofdm_bat.json
        bat = Modem.load_bat('ofdm_bat.json')
        data_audio, n_frames = modem.encode_data(text, bat)

    RX flow
    -------
        modem = Modem(cfg)
        result = modem.analyze_probe(probe_rx_audio)
        bat = modem.allocate(result)
        Modem.save_bat(bat, 'ofdm_bat.json')
        print(modem.bat_summary(bat))
        stats, chunks = modem.decode_data(data_rx_audio, bat)
        text = AdaptiveDemodulator.reassemble(chunks)
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg     = cfg
        self._probe  = ChannelProbe(cfg)
        self._alloc  = BitAllocator()

    # - probe -

    def build_probe(self) -> np.ndarray:
        """Build probe audio (SC + CE + N_PROBE_SYMS known BPSK symbols)."""
        return self._probe.build()

    def analyze_probe(
        self, rx: np.ndarray, threshold: float = 0.25
    ) -> dict:
        """Estimate per-subcarrier SNR from received probe recording."""
        return self._probe.estimate(rx, threshold=threshold)

    # - allocation -

    def allocate(
        self,
        probe_result:  dict,
        snr_margin_db: float | None = None,
        max_bits:      int   | None = None,
    ) -> np.ndarray:
        """Compute bits_per_bin from a probe_result dict."""
        if not probe_result.get('valid'):
            raise RuntimeError(f"probe failed: {probe_result.get('error')}")
        margin = (snr_margin_db if snr_margin_db is not None
                  else self.cfg.get('snr_margin_db', 6.0))
        mx     = (max_bits if max_bits is not None
                  else self.cfg.get('max_bits_per_bin', 6))
        pilot_set = set(range(0, self.cfg['n_occupied'],
                              self.cfg['pilot_spacing']))
        # isi_derate scales the measured ceiling (0 = ignore it, 1 = trust it).
        beta = probe_result.get('isi_ratio')
        strength = float(self.cfg.get('isi_derate', 0.0))
        # Out-of-range beta is a failed MEASUREMENT, not a bad channel: ignore it
        # rather than de-rate on it (see _ISI_BETA_MAX).
        if beta is not None and beta > _ISI_BETA_MAX:
            beta = None
        beta_eff = (strength * beta if beta and strength > 0 else 0.0)
        bits = self._alloc.allocate(
            probe_result['snr_db'], pilot_set,
            self.cfg['n_occupied'], margin, mx, isi_beta=beta_eff,
        )
        if beta_eff <= 0 or bits.any():
            return bits

        # An ALL-OFF table is not a negotiation outcome, it is a dead run.
        # Observed OTA 20260804_034714: one node measured beta=1.21 (a -0.8 dB
        # ceiling, below BPSK's floor) and allocated zero carriers; the peer's
        # handshake then read the 0-active table as "probe-shaped frame, likely
        # my own message" and ignored it on every poll until the window expired.
        #
        # So back the de-rating off until something survives, and say so loudly.
        # This is deliberately a POLICY choice over a measurement one: a link
        # this reverberant may genuinely not carry data, but the run learns far
        # more from a weak link that completes and reports its EVM than from a
        # handshake that dies silently -- and the fallback ladder (margin climb,
        # band renegotiation, CSS) can only act on a link that came up.
        for scale in (0.5, 0.25, 0.1, 0.0):
            trial = self._alloc.allocate(
                probe_result['snr_db'], pilot_set,
                self.cfg['n_occupied'], margin, mx, isi_beta=beta_eff * scale,
            )
            if trial.any():
                print(f"WARNING: ISI de-rating (beta={beta_eff:.3f}, ceiling "
                      f"{-10 * math.log10(max(beta_eff, 1e-9)):.1f}dB) left "
                      f"NO usable carriers -- backing it off to "
                      f"{scale:g}x beta ({int((trial > 0).sum())} carriers "
                      f"survive). The measurement stands: this path is "
                      f"reverberation-limited and its data EVM will be poor. "
                      f"A probe SNR far below the peer's usually means a weak "
                      f"DIRECT path (mic blocked/aimed away), which raises the "
                      f"late/early ratio without the room changing.",
                      flush=True)
                return trial
        return bits

    def bat_summary(self, bits_per_bin: np.ndarray) -> str:
        return self._alloc.summary(
            bits_per_bin, self.cfg['symbol_samples'], self.cfg['sample_rate']
        )

    # - data -

    def encode_data(
        self, input_str: str, bits_per_bin: np.ndarray
    ) -> tuple[np.ndarray, int]:
        mod = AdaptiveModulator(self.cfg, bits_per_bin)
        sig = mod.encode(input_str)
        return sig, mod.total_frames

    def decode_data(
        self,
        samples:       np.ndarray,
        bits_per_bin:  np.ndarray,
        threshold:     float = 0.25,
        stats_callback = None,
    ) -> tuple[list[dict], dict[int, bytes]]:
        dem = AdaptiveDemodulator(self.cfg, bits_per_bin)
        return dem.decode_samples(samples, threshold, stats_callback)

    # - persistence -

    @staticmethod
    def save_bat(bits_per_bin: np.ndarray, path: str) -> None:
        Path(path).write_text(
            json.dumps({'bits_per_bin': bits_per_bin.tolist()}, indent=2)
        )
        active = int(np.sum(bits_per_bin > 0))
        print(f"BAT saved -> {path}  ({active} active carriers)", flush=True)

    @staticmethod
    def load_bat(path: str) -> np.ndarray:
        return np.array(
            json.loads(Path(path).read_text())['bits_per_bin'], dtype=np.int32
        )


# -
# Report helpers
# -

def carrier_spectrum(cfg: dict, snr_db_occupied, bits_per_bin=None) -> dict:
    """Per-subcarrier probe measurement tagged with its audio frequency, for
    the report generator's BAT chart.

    The two arrays a probe produces are indexed DIFFERENTLY, which is the whole
    reason this helper exists: `snr_db` runs over all `n_occupied` subcarriers,
    while a BAT runs over DATA carriers only (allocate() skips pilots).  Line
    them up on the occupied grid and put -1 at the pilot positions, so a
    consumer can plot frequency vs SNR without re-deriving the layout.

    Returns {'freq_hz': [...], 'snr_db': [...], 'bits': [...]}, all the same
    length (n_occupied); bits[i] == -1 marks a pilot.
    """
    snr = np.asarray(snr_db_occupied, dtype=float).ravel()
    n_occ = int(cfg['n_occupied'])
    n = min(n_occ, len(snr))
    sc_hz, bin_lo = float(cfg['sc_spacing_hz']), int(cfg['bin_lo'])
    pilot_set = set(range(0, n_occ, int(cfg['pilot_spacing'])))

    bits_out = [-1] * n
    if bits_per_bin is not None:
        bat = np.asarray(bits_per_bin, dtype=int).ravel()
        k = 0
        for i in range(n):
            if i in pilot_set:
                continue
            if k < len(bat):
                bits_out[i] = int(bat[k])
            k += 1

    return {
        'freq_hz': [round((bin_lo + i) * sc_hz, 1) for i in range(n)],
        'snr_db':  [round(float(snr[i]), 1) for i in range(n)],
        'bits':    bits_out,
    }


# -
# Config builder
# -

def build_config(
    n_fft:            int   = 1024,
    cp_ratio:         float = 0.25,
    freq_lo:          float = 1000.0,
    freq_hi:          float = 16000.0,
    pilot_spacing:    int   = 8,
    sample_rate:      int   = 48000,
    ecc_bytes:        int   = 16,
    chunk_bytes:      int   = 256,
    seq_bytes:        int   = 2,
    taper_ms:         float = 2.0,
    frame_gap_ms:     float = 20.0,
    probe_duration_s: float = 4.0,   # how long the probe plays (seconds)
    snr_margin_db:    float = 6.0,
    max_bits_per_bin: int   = 6,
    sync_period:      int   = 32,
    snr_estimator:    str   = 'mean',
    snr_pctile:       float = 15.0,
    snr_blocks:       int   = 8,
    isi_derate:       float = 0.0,
    data_aided:       bool  = True,
    probe_meas_syms:  int   = 32,
) -> dict:
    """
    Derive all config fields from primary parameters.

    probe_duration_s controls the acoustic probe length.  The probe must be
    long enough for:
      - the recording thread on the decode side to start before the probe ends
      - enough OFDM symbol averages for a low-variance SNR estimate
      - room reverberation to settle between averages

    At 48 kHz / n_fft=1024 / cp_ratio=0.25 each symbol is 26.7 ms.
    probe_duration_s=4.0 gives ~147 probe symbols -- 147 coherent averages.
    Minimum recommended: 3.0 s.  For reverberant rooms: 5.0 s.
    """
    if n_fft & (n_fft - 1):
        raise ValueError(f"n_fft must be power-of-2, got {n_fft}")
    cp_len    = max(1, int(round(n_fft * cp_ratio)))
    sc_spc    = sample_rate / n_fft
    bin_lo    = max(1, math.ceil(freq_lo  / sc_spc))
    bin_hi    = min(n_fft // 2 - 1, math.floor(freq_hi / sc_spc))
    if bin_lo > bin_hi:
        raise ValueError(f"No subcarriers in [{freq_lo},{freq_hi}] Hz")
    n_occ     = bin_hi - bin_lo + 1
    p_offs    = list(range(0, n_occ, pilot_spacing))
    n_pilots  = len(p_offs)
    n_data    = n_occ - n_pilots
    sym_samp  = n_fft + cp_len
    raw_tap   = int(round(taper_ms * sample_rate / 1000))
    taper_samp = min(raw_tap, cp_len // 2)
    gap_samp  = int(round(frame_gap_ms * sample_rate / 1000))

    # Compute n_probe_syms from probe_duration_s.
    # Total probe audio = (2 + n_probe_syms) symbols (SC + CE + probes).
    # Solve: (2 + n) * sym_samp / sample_rate >= probe_duration_s  =>  n >= ...
    n_probe_syms = max(8, int(math.ceil(
        probe_duration_s * sample_rate / sym_samp
    )) - 2)
    # Split the probe: a PERIODIC block for acquisition (identical symbols, which
    # is what the coherence search needs) and a NON-PERIODIC block for the SNR
    # measurement (see ChannelProbe.build).  Taken OUT of the existing symbol
    # count, so total probe airtime is unchanged.
    n_meas = int(min(max(0, probe_meas_syms), n_probe_syms // 3))
    n_probe_syms = max(8, n_probe_syms - n_meas)
    # The ISI estimator needs a floor of symbols to average H over
    # (_ISI_MIN_SYMS), and the symbol count falls as n_fft RISES for a fixed
    # probe DURATION: 5 s yields 139 periodic symbols at n_fft=1024 but only 57
    # at 2048 and 28 at 4096.  So buying cyclic-prefix guard time with a bigger
    # FFT -- the correct move against reverberation -- silently dropped the
    # measurement under the gate and handed the allocator back to the blind
    # snr_margin_db toll, which is exactly the guesswork the ceiling exists to
    # replace.  Runs 20260805_050849 and _054955 lost it this way and
    # over-allocated one direction by ~10 dB.
    #
    # Airtime is the honest price of the measurement, so lengthen the probe
    # rather than lowering the bar: the handshake budget is derived by encoding
    # the actual probe, so the schedule follows automatically.
    if n_probe_syms < _ISI_MIN_SYMS:
        n_probe_syms = _ISI_MIN_SYMS
    actual_probe_s = (2 + n_probe_syms + n_meas) * sym_samp / sample_rate

    return {
        'n_fft':            n_fft,
        'cp_ratio':         cp_ratio,
        'cp_len':           cp_len,
        'freq_lo_hz':       float(bin_lo * sc_spc),
        'freq_hi_hz':       float(bin_hi * sc_spc),
        'bin_lo':           bin_lo,
        'bin_hi':           bin_hi,
        'n_occupied':       n_occ,
        'pilot_spacing':    pilot_spacing,
        'n_pilots':         n_pilots,
        'n_data_carriers':  n_data,
        'sample_rate':      sample_rate,
        'sc_spacing_hz':    float(sc_spc),
        'symbol_samples':   sym_samp,
        'ecc_bytes':        ecc_bytes,
        'chunk_bytes':      chunk_bytes,
        'seq_bytes':        seq_bytes,
        'taper_ms':         taper_ms,
        'taper_samples':    taper_samp,
        'frame_gap_ms':     frame_gap_ms,
        'frame_gap_samples': gap_samp,
        'probe_duration_s': probe_duration_s,
        'n_probe_syms':     n_probe_syms,
        'n_probe_meas_syms': n_meas,
        'actual_probe_s':   round(actual_probe_s, 2),
        'snr_margin_db':    snr_margin_db,
        'max_bits_per_bin': max_bits_per_bin,
        'sync_period':      sync_period,
        'snr_estimator':    snr_estimator,
        'snr_pctile':       snr_pctile,
        'snr_blocks':       snr_blocks,
        'isi_derate':       isi_derate,
        'data_aided':       data_aided,
    }


# -
# CLI (standalone test / probe / encode)
# -

EXAMPLE_CONFIG = {
    'n_fft': 1024, 'cp_ratio': 0.25,
    'freq_lo': 1000, 'freq_hi': 16000,
    'pilot_spacing': 8, 'sample_rate': 48000,
    'ecc_bytes': 16, 'chunk_bytes': 256, 'seq_bytes': 2,
    'taper_ms': 2.0, 'frame_gap_ms': 20.0,
    'probe_duration_s': 4.0,
    'snr_margin_db': 6.0,
    'max_bits_per_bin': 6, 'sync_period': 32,
}

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='ADSL2-style acoustic OFDM modem')
    ap.add_argument('--print-schema', action='store_true')
    ap.add_argument('--config',       default=None, help='JSON config file')
    ap.add_argument('--probe-out',    default=None, help='Write probe WAV')
    ap.add_argument('--probe-in',     default=None, help='Analyse probe WAV -> BAT')
    ap.add_argument('--bat-out',      default=None, help='Save BAT JSON')
    ap.add_argument('--bat-in',       default=None, help='Load BAT JSON for encoding')
    ap.add_argument('--encode',       default=None, help='Text to encode')
    ap.add_argument('--wav-out',      default='ofdm_output.wav')
    ap.add_argument('--threshold',    type=float, default=0.25)
    args = ap.parse_args()

    if args.print_schema:
        print(json.dumps(EXAMPLE_CONFIG, indent=2)); sys.exit(0)

    raw_cfg = {}
    if args.config and Path(args.config).exists():
        raw_cfg = json.loads(Path(args.config).read_text())

    cfg = build_config(**{k: raw_cfg[k] for k in raw_cfg if k in build_config.__code__.co_varnames})
    modem = Modem(cfg)

    if args.probe_out:
        sig  = modem.build_probe()
        peak = np.max(np.abs(sig))
        scale = 32700.0 / peak if peak > 0 else 1.0
        wavfile.write(args.probe_out, cfg['sample_rate'],
                      (sig * scale).astype(np.int16))
        print(f"Probe written: {args.probe_out}  ({len(sig)/cfg['sample_rate']:.2f}s)")

    if args.probe_in:
        sr, pcm = wavfile.read(args.probe_in)
        samp    = pcm.astype(np.float32) / 32768.0
        result  = modem.analyze_probe(samp, threshold=args.threshold)
        if not result['valid']:
            print(f"Probe analysis failed: {result.get('error')}"); sys.exit(1)
        bat = modem.allocate(result)
        print(f"SNR range: {result['snr_db'].min():.1f} .. {result['snr_db'].max():.1f} dB")
        print(f"BAT: {modem.bat_summary(bat)}")
        if args.bat_out:
            Modem.save_bat(bat, args.bat_out)

    if args.encode:
        bat_path = args.bat_in
        if bat_path and Path(bat_path).exists():
            bat = Modem.load_bat(bat_path)
        else:
            # No BAT: fall back to conservative QPSK on all carriers
            print("No BAT supplied -- using fallback QPSK on all carriers")
            bat = np.full(cfg['n_data_carriers'], 2, dtype=np.int32)
        sig, nf = modem.encode_data(args.encode, bat)
        peak    = np.max(np.abs(sig))
        wavfile.write(args.wav_out, cfg['sample_rate'],
                      (sig * 32700.0 / peak).astype(np.int16))
        print(f"Encoded {nf} frames -> {args.wav_out}")
