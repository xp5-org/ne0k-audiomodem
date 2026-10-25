"""
ofdm_writer.py - Acoustic OFDM Modulator
==========================================
Encodes a UTF-8 string as a sequence of OFDM frames and writes a WAV file.
Drop-in design companion to qam_writer.py.

Frame structure (one RS-coded chunk per frame):
  [ Schmidl-Cox sync symbol  (n_fft + cp_len samples, real) ]
  [ Channel-estimation symbol (n_fft + cp_len samples, real) ]
  [ data_syms_per_pkt x data OFDM symbols                   ]

All OFDM symbols use Hermitian (conjugate) symmetry so the IFFT output is
real and can be written directly as a mono WAV  -  no carrier mixing needed.
Occupied bins run from bin_lo to bin_hi (positive frequency only); the
matching negative-frequency bins are set automatically.

Pilots are inserted every pilot_spacing subcarriers within the occupied
block and carry a fixed known symbol used by the receiver for per-symbol
phase-drift correction.

Standalone usage:
    python ofdm_writer.py                      # encode built-in demo text
    python ofdm_writer.py config.json          # read params from JSON
    python ofdm_writer.py --print-schema       # print example config JSON

Config JSON (all optional):
{
    "input_str":     "Hello world",
    "output":        "ofdm_output.wav",
    "config_out":    "ofdm_config.json",
    "n_fft":         1024,
    "cp_ratio":      0.25,
    "freq_lo":       1000,
    "freq_hi":       16000,
    "pilot_spacing": 8,
    "modulation":    "qam16",
    "sample_rate":   48000,
    "ecc_bytes":     16,
    "chunk_bytes":   256,
    "seq_bytes":     2
}
"""
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
# RS_BACKEND=c swaps the pure-Python reedsolo for the C core in
# tests/c/ (rs255_c.RSCodec is a drop-in and defaults to reedsolo),
# which is why this imports the shim rather than the package.
from rs255_c import RSCodec


# -- Shared constellation tables (also imported by ofdm_reader) ---------------

# Gray-coded 16-QAM (same layout as qam_writer for consistency)
QAM16_CONST: dict[tuple, complex] = {
    (0,0,0,0): -3-3j, (0,0,0,1): -3-1j, (0,0,1,1): -3+1j, (0,0,1,0): -3+3j,
    (0,1,1,0): -1+3j, (0,1,1,1): -1+1j, (0,1,0,1): -1-1j, (0,1,0,0): -1-3j,
    (1,1,0,0):  1-3j, (1,1,0,1):  1-1j, (1,1,1,1):  1+1j, (1,1,1,0):  1+3j,
    (1,0,1,0):  3+3j, (1,0,1,1):  3+1j, (1,0,0,1):  3-1j, (1,0,0,0):  3-3j,
}
QAM16_NORM   = math.sqrt(10)
QAM16_POINTS = np.array(list(QAM16_CONST.values())) / QAM16_NORM
QAM16_BITS   = np.array(list(QAM16_CONST.keys()),   dtype=np.uint8)

# Gray-coded QPSK
QPSK_CONST: dict[tuple, complex] = {
    (0, 0): ( 1+1j) / math.sqrt(2),
    (0, 1): (-1+1j) / math.sqrt(2),
    (1, 1): (-1-1j) / math.sqrt(2),
    (1, 0): ( 1-1j) / math.sqrt(2),
}
QPSK_POINTS = np.array(list(QPSK_CONST.values()))
QPSK_BITS   = np.array(list(QPSK_CONST.keys()),  dtype=np.uint8)

BITS_PER_MOD = {"bpsk": 1, "qpsk": 2, "qam16": 4}


def prbs(length: int, seed: int = 0xACE1) -> np.ndarray:
    """16-bit maximal-length LFSR  -  identical poly to qam_writer."""
    state = seed & 0xFFFF
    out   = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        fb      = ((state >> 15) ^ (state >> 14) ^ (state >> 12) ^ (state >> 3)) & 1
        out[i]  = (state >> 15) & 1
        state   = ((state << 1) | fb) & 0xFFFF
    return out


# -- Modulator -----------------------------------------------------------------

class Modulator:
    # Two separate PRBS seeds so CE and data scrambler never interfere
    PRBS_SEED_DATA: int     = 0xACE1
    PRBS_SEED_CE:   int     = 0x3D5F
    PILOT_SYMBOL:   complex = (1 + 1j) / math.sqrt(2)

    def __init__(
        self,
        input_str:     str,
        n_fft:         int   = 1024,
        cp_ratio:      float = 0.25,
        freq_lo:       float = 1000.0,
        freq_hi:       float = 16000.0,
        pilot_spacing: int   = 8,
        modulation:    str   = "qam16",
        sample_rate:   int   = 48000,
        ecc_bytes:     int   = 16,
        chunk_bytes:   int   = 256,
        seq_bytes:     int   = 2,
        taper_ms:      float = 2.0,
        frame_gap_ms:  float = 20.0,
    ) -> None:
        self.input_str     = input_str
        self.n_fft         = n_fft
        self.cp_len        = max(1, int(round(n_fft * cp_ratio)))
        self.cp_ratio      = cp_ratio
        self.sample_rate   = sample_rate
        self.pilot_spacing = pilot_spacing
        self.modulation    = modulation.lower()
        self.ecc_bytes     = ecc_bytes
        self.chunk_bytes   = chunk_bytes
        self.seq_bytes     = seq_bytes
        self.taper_ms      = taper_ms
        self.frame_gap_ms  = frame_gap_ms

        if n_fft & (n_fft - 1):
            raise ValueError(f"n_fft must be a power of 2, got {n_fft}")
        if self.modulation not in BITS_PER_MOD:
            raise ValueError(f"modulation must be bpsk/qpsk/qam16, got {modulation}")

        # Windowing: taper must fit inside the CP so the FFT window is unaffected.
        # Hard cap at cp_len // 2 to leave at least half the CP as clean guard.
        raw_taper             = int(round(taper_ms * sample_rate / 1000))
        self.taper_samples    = min(raw_taper, self.cp_len // 2)
        self.frame_gap_samples = int(round(frame_gap_ms * sample_rate / 1000))

        # Subcarrier grid
        self.sc_spacing   = sample_rate / n_fft                         # Hz
        self.bin_lo       = max(1, math.ceil(freq_lo  / self.sc_spacing))
        self.bin_hi       = min(n_fft // 2 - 1,
                                math.floor(freq_hi / self.sc_spacing))
        if self.bin_lo > self.bin_hi:
            raise ValueError(f"No subcarriers in [{freq_lo}, {freq_hi}] Hz with "
                             f"n_fft={n_fft}, sample_rate={sample_rate}")
        self.n_occupied   = self.bin_hi - self.bin_lo + 1

        # Pilot layout within the occupied block
        self.pilot_offsets: list[int] = list(range(0, self.n_occupied, pilot_spacing))
        self._pilot_set               = set(self.pilot_offsets)
        self.n_pilots                 = len(self.pilot_offsets)
        self.n_data_carriers          = self.n_occupied - self.n_pilots

        if self.n_data_carriers <= 0:
            raise ValueError("pilot_spacing too small  -  no data carriers remain")

        self.bits_per_qam      = BITS_PER_MOD[self.modulation]
        self.bits_per_ofdm_sym = self.n_data_carriers * self.bits_per_qam

        self.rsc           = RSCodec(ecc_bytes)
        self.payload_bytes = seq_bytes + chunk_bytes + ecc_bytes
        self.data_syms_per_pkt = math.ceil(
            self.payload_bytes * 8 / self.bits_per_ofdm_sym
        )
        self.symbol_samples = n_fft + self.cp_len

        # Pre-build fixed symbols
        self._ce_known       = self._make_ce_sequence()
        self._sc_preamble_td = self._make_sc_preamble()

        # Set by encode()
        self.tx_signal:    np.ndarray | None = None
        self.scale_factor: float | None      = None
        self.total_frames: int               = 0

        self._print_summary()

    # -- public API ------------------------------------------------------------

    def encode(self) -> np.ndarray:
        data = self.input_str.encode("utf-8")
        pad  = (-len(data)) % self.chunk_bytes
        data += b"\x00" * pad
        chunks = [data[i : i + self.chunk_bytes]
                  for i in range(0, len(data), self.chunk_bytes)]
        self.total_frames = len(chunks)

        gap    = (np.zeros(self.frame_gap_samples, dtype=np.float64)
                  if self.frame_gap_samples > 0 else None)
        pieces: list[np.ndarray] = []

        for seq_idx, chunk in enumerate(chunks):
            seq_b   = seq_idx.to_bytes(self.seq_bytes, "big")
            encoded = bytes(self.rsc.encode(seq_b + chunk))

            raw_bits = np.unpackbits(np.frombuffer(encoded, dtype=np.uint8))
            needed   = self.data_syms_per_pkt * self.bits_per_ofdm_sym
            if len(raw_bits) < needed:
                raw_bits = np.concatenate(
                    [raw_bits, np.zeros(needed - len(raw_bits), dtype=np.uint8)]
                )

            scr_bits = prbs(len(raw_bits), self.PRBS_SEED_DATA) ^ raw_bits

            # Collect per-symbol occupied-subcarrier arrays for this frame
            sym_occ = [self._ce_known.copy()]   # CE symbol (SC preamble handled separately)
            for si in range(self.data_syms_per_pkt):
                b = scr_bits[si * self.bits_per_ofdm_sym :
                             (si + 1) * self.bits_per_ofdm_sym]
                sym_occ.append(self._data_occ(b))

            # Build frame: SC preamble (always unwindowed for clean sync metric)
            # followed by CE + data symbols assembled with windowed overlap-add
            frame = np.concatenate([
                self._sc_preamble_td,       # unwindowed (SC metric needs clean amplitude)
                self._build_ola(sym_occ),   # windowed CE + data
            ])

            if gap is not None:
                pieces.append(gap)
            pieces.append(frame)

        if gap is not None:
            pieces.append(gap)   # trailing gap so the last frame fades to silence

        sig  = np.concatenate(pieces)
        peak = np.max(np.abs(sig))
        self.scale_factor = 32700.0 / peak if peak > 0 else 1.0
        self.tx_signal    = sig
        return sig

    def write_wav(self, filename: str = "ofdm_output.wav") -> None:
        if self.tx_signal is None:
            self.encode()
        pcm = (self.tx_signal * self.scale_factor).astype(np.int16)
        wavfile.write(filename, self.sample_rate, pcm)
        dur = len(self.tx_signal) / self.sample_rate
        print(f"Written {filename}  ({dur:.2f}s, {self.total_frames} frames)",
              flush=True)

    def write_config(self, filename: str = "ofdm_config.json") -> None:
        if self.scale_factor is None:
            self.encode()
        cfg = {
            "input_str":          self.input_str,
            "n_fft":              self.n_fft,
            "cp_ratio":           self.cp_ratio,
            "cp_len":             self.cp_len,
            "freq_lo_hz":         float(self.bin_lo * self.sc_spacing),
            "freq_hi_hz":         float(self.bin_hi * self.sc_spacing),
            "bin_lo":             self.bin_lo,
            "bin_hi":             self.bin_hi,
            "n_occupied":         self.n_occupied,
            "pilot_spacing":      self.pilot_spacing,
            "n_pilots":           self.n_pilots,
            "n_data_carriers":    self.n_data_carriers,
            "modulation":         self.modulation,
            "bits_per_qam":       self.bits_per_qam,
            "bits_per_ofdm_sym":  self.bits_per_ofdm_sym,
            "sample_rate":        self.sample_rate,
            "sc_spacing_hz":      float(self.sc_spacing),
            "symbol_samples":     self.symbol_samples,
            "ecc_bytes":          self.ecc_bytes,
            "chunk_bytes":        self.chunk_bytes,
            "seq_bytes":          self.seq_bytes,
            "payload_bytes":      self.payload_bytes,
            "data_syms_per_pkt":  self.data_syms_per_pkt,
            "total_frames":       self.total_frames,
            "scale_factor":       float(self.scale_factor),
            "taper_ms":           self.taper_ms,
            "taper_samples":      self.taper_samples,
            "frame_gap_ms":       self.frame_gap_ms,
            "frame_gap_samples":  self.frame_gap_samples,
            "PRBS_SEED_DATA":     self.PRBS_SEED_DATA,
            "PRBS_SEED_CE":       self.PRBS_SEED_CE,
            "ce_known_re":        self._ce_known.real.tolist(),
            "ce_known_im":        self._ce_known.imag.tolist(),
        }
        with open(filename, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Written {filename}", flush=True)

    # -- private helpers -------------------------------------------------------

    def _ofdm_td(self, occ: np.ndarray) -> np.ndarray:
        """
        Convert n_occupied complex values (occupied block) to a real time-domain
        OFDM symbol with cyclic prefix.

        Hermitian symmetry: X[N-b] = conj(X[b]) makes IFFT output real.  Only
        the positive-frequency half is filled -- irfft IS that Hermitian
        inverse, so the mirror is implied rather than materialised, halving the
        transform's flops (a real-output inverse is an N/2 complex FFT plus an
        O(N) split).  Occupied bins never reach DC or Nyquist, so the half
        spectrum is unambiguous.
        """
        X = np.zeros(self.n_fft // 2 + 1, dtype=complex)
        X[self.bin_lo:self.bin_lo + len(occ)] = occ
        td = np.fft.irfft(X, n=self.n_fft) * math.sqrt(self.n_fft)
        return np.concatenate([td[-self.cp_len:], td])   # prepend CP

    def _ofdm_td_windowed(self, occ: np.ndarray) -> np.ndarray:
        """
        W-OFDM: build a time-domain OFDM symbol with a raised-cosine taper on
        both ends, using an EXTENDED cyclic prefix so the FFT window stays clean.

        Extended symbol layout  (T = taper_samples, CP = cp_len, N = n_fft):

          [ T extra CP | CP | body N | T body-start repeat ]
          total length = N + CP + 2*T

        Window applied:
          first T  samples: 0 -> 1  (rising half-Hann)
          middle N+CP:      flat 1
          last  T  samples: 1 -> 0  (falling half-Hann)

        Complementary property: w_rise[t] + w_fall[t] = 1  =>  overlap-add of
        adjacent windowed symbols reconstructs the signal exactly at boundaries.
        """
        T = self.taper_samples
        if T == 0:
            return self._ofdm_td(occ)

        N  = self.n_fft
        CP = self.cp_len

        X = np.zeros(N // 2 + 1, dtype=complex)   # half spectrum; see _ofdm_td
        X[self.bin_lo:self.bin_lo + len(occ)] = occ
        td = np.fft.irfft(X, n=N) * math.sqrt(N)

        # Extended: [td[N-CP-T : N]  |  td[0:N]  |  td[0:T]]
        extended = np.concatenate([td[N - CP - T :], td, td[:T]])
        # length = T + CP + N + T = N + CP + 2T  (check)

        # Raised-cosine taper (complementary: rise[t] + fall[t] = 1 for all t)
        t     = (np.arange(T) + 0.5) / T          # 0.5/T .. (T-0.5)/T
        rise  = 0.5 * (1.0 - np.cos(np.pi * t))   # 0 -> 1
        fall  = 0.5 * (1.0 + np.cos(np.pi * t))   # 1 -> 0

        win         = np.ones(len(extended))
        win[:T]     = rise
        win[-T:]    = fall

        return extended * win

    def _build_ola(self, sym_occ_list: list) -> np.ndarray:
        """
        Overlap-and-add assembly of windowed OFDM symbols.

        Each windowed symbol has length SYM + 2*T.  Symbols are placed at stride
        SYM so the T-sample taper tails/heads overlap exactly, and by the
        complementary property their overlap-add sums to 1 -> no discontinuity.

        Output length = n_syms * SYM + T
          (first T samples ramp up from 0; last T samples ramp down to 0)
        """
        T   = self.taper_samples
        SYM = self.symbol_samples
        n   = len(sym_occ_list)

        if T == 0:
            return np.concatenate([self._ofdm_td(occ) for occ in sym_occ_list])

        out_len = n * SYM + 2 * T
        out     = np.zeros(out_len)

        for i, occ in enumerate(sym_occ_list):
            sym = self._ofdm_td_windowed(occ)   # length SYM + 2T
            start = i * SYM
            out[start : start + SYM + 2 * T] += sym

        # Trim the extra T that overhangs past the last symbol's nominal end
        # (it's the falling taper of the last symbol, which we keep)
        return out

    def _data_occ(self, bits: np.ndarray) -> np.ndarray:
        """Return occupied-subcarrier array for one data OFDM symbol (pilots + data)."""
        qam = self._modulate(bits)
        occ = np.zeros(self.n_occupied, dtype=complex)
        di  = 0
        for i in range(self.n_occupied):
            if i in self._pilot_set:
                occ[i] = self.PILOT_SYMBOL
            else:
                occ[i] = qam[di]
                di += 1
        return occ

    def _make_sc_preamble(self) -> np.ndarray:
        """
        Schmidl-Cox sync preamble: PN-BPSK on *even-indexed* occupied bins,
        zeros on odd.  The IFFT output has period n_fft/2 samples, enabling
        the delay-and-correlate timing metric at the receiver.
        """
        n_even = (self.n_occupied + 1) // 2
        pn     = prbs(n_even, seed=self.PRBS_SEED_CE ^ 0xFFFF)
        bpsk   = (1.0 - 2.0 * pn.astype(float)).astype(complex) / math.sqrt(2)
        occ    = np.zeros(self.n_occupied, dtype=complex)
        j = 0
        for i in range(0, self.n_occupied, 2):
            occ[i] = bpsk[j] if j < len(bpsk) else 0.0
            j += 1
        return self._ofdm_td(occ)

    def _make_ce_sequence(self) -> np.ndarray:
        """Known BPSK PN on all occupied bins  -  used for LS channel estimation."""
        pn = prbs(self.n_occupied, seed=self.PRBS_SEED_CE)
        return (1.0 - 2.0 * pn.astype(float)).astype(complex) / math.sqrt(2)

    def _modulate(self, bits: np.ndarray) -> np.ndarray:
        if self.modulation == "bpsk":
            return (1.0 - 2.0 * bits.astype(float)).astype(complex)
        elif self.modulation == "qpsk":
            b = bits.reshape(-1, 2)
            return np.array([QPSK_CONST[tuple(row)] for row in b])
        else:  # qam16
            b = bits.reshape(-1, 4)
            return np.array([QAM16_CONST[tuple(row)] for row in b]) / QAM16_NORM

    def _print_summary(self) -> None:
        bw       = (self.bin_hi - self.bin_lo) * self.sc_spacing
        sym_ms   = 1000 * self.symbol_samples / self.sample_rate
        fr_samps = self.symbol_samples * (self.data_syms_per_pkt + 2)
        net_bps  = int(self.chunk_bytes * 8 * self.sample_rate / fr_samps)
        taper_ms_actual = 1000 * self.taper_samples / self.sample_rate
        print(
            f"OFDM Modulator  n_fft={self.n_fft}  "
            f"df={self.sc_spacing:.2f} Hz  "
            f"CP={self.cp_len}/{self.n_fft} ({self.cp_ratio*100:.0f}%)  "
            f"sym={sym_ms:.1f} ms",
            flush=True,
        )
        print(
            f"Band  {self.bin_lo*self.sc_spacing:.0f}-"
            f"{self.bin_hi*self.sc_spacing:.0f} Hz  "
            f"BW~{bw:.0f} Hz  "
            f"{self.n_occupied} subcarriers  "
            f"({self.n_pilots} pilots / {self.n_data_carriers} data)",
            flush=True,
        )
        print(
            f"Modulation {self.modulation.upper()}  "
            f"{self.bits_per_ofdm_sym} bits/OFDM-sym  "
            f"Net~{net_bps} bps",
            flush=True,
        )
        print(
            f"Frame: 1 SC + 1 CE + {self.data_syms_per_pkt} data "
            f"({self.data_syms_per_pkt + 2} OFDM symbols total)  "
            f"taper={taper_ms_actual:.1f}ms ({self.taper_samples} samp)  "
            f"gap={self.frame_gap_ms:.0f}ms",
            flush=True,
        )


# -- CLI -----------------------------------------------------------------------

EXAMPLE_CONFIG = {
    "input_str":     "Hello world",
    "output":        "ofdm_output.wav",
    "config_out":    "ofdm_config.json",
    "n_fft":         1024,
    "cp_ratio":      0.25,
    "freq_lo":       1000,
    "freq_hi":       16000,
    "pilot_spacing": 8,
    "modulation":    "qam16",
    "sample_rate":   48000,
    "ecc_bytes":     16,
    "chunk_bytes":   256,
    "seq_bytes":     2,
    "taper_ms":      2.0,
    "frame_gap_ms":  20.0,
}

DEFAULT_MSG = """Do you ever get bored doing the same job every day-
Son, let me tell you about stirring.
You grab that stick, and you just move it around, and you stir it around.
You get yourself into a rhythm.
It's a beautiful thing.
You know, Dad, the more I think about it,
maybe the honey field just isn't right for me.
You were thinking of what, making balloon animals-
That's a bad job for a guy with a stinger.
Janet, your son's not sure he wants to go into honey!
Barry, you are so funny sometimes.
I'm not trying to be funny.
You're not funny! You're going into honey. Our son, the stirrer!
You're gonna be a stirrer-
No one's listening to me!
Wait till you see the sticks I have.
I could say anything right now.
I'm gonna get an ant tattoo!
Let's open some honey and celebrate!
Maybe I'll pierce my thorax. Shave my antennae.
Shack up with a grasshopper. Get a gold tooth and call everybody "dawg"!
I'm so proud. We're starting work today!
"""


def encode_from_config(cfg: dict) -> dict:
    mod = Modulator(
        input_str     = cfg.get("input_str",     DEFAULT_MSG),
        n_fft         = int(cfg.get("n_fft",         1024)),
        cp_ratio      = float(cfg.get("cp_ratio",      0.25)),
        freq_lo       = float(cfg.get("freq_lo",       1000)),
        freq_hi       = float(cfg.get("freq_hi",       16000)),
        pilot_spacing = int(cfg.get("pilot_spacing", 8)),
        modulation    = cfg.get("modulation",    "qam16"),
        sample_rate   = int(cfg.get("sample_rate",   48000)),
        ecc_bytes     = int(cfg.get("ecc_bytes",     16)),
        chunk_bytes   = int(cfg.get("chunk_bytes",   256)),
        seq_bytes     = int(cfg.get("seq_bytes",     2)),
        taper_ms      = float(cfg.get("taper_ms",      2.0)),
        frame_gap_ms  = float(cfg.get("frame_gap_ms",  20.0)),
    )
    mod.encode()
    mod.write_wav(cfg.get("output",     "ofdm_output.wav"))
    mod.write_config(cfg.get("config_out", "ofdm_config.json"))
    return {
        "status":       "ok",
        "total_frames": mod.total_frames,
        "duration_s":   round(len(mod.tx_signal) / mod.sample_rate, 2),
        "wav_path":     cfg.get("output",     "ofdm_output.wav"),
        "config_path":  cfg.get("config_out", "ofdm_config.json"),
    }


if __name__ == "__main__":
    if "--print-schema" in sys.argv:
        print(json.dumps(EXAMPLE_CONFIG, indent=2))
        sys.exit(0)

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = {}
    if config_path and Path(config_path).exists():
        cfg = json.loads(Path(config_path).read_text())
    elif config_path:
        print(f"Config '{config_path}' not found  -  using defaults.", flush=True)

    result = encode_from_config(cfg)
    print(json.dumps(result), flush=True)
    sys.exit(0 if result["status"] == "ok" else 1)
