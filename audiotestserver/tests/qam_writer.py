"""
qam_writer.py - Acoustic QAM Modulator (standalone)
=====================================================
Encodes a UTF-8 string as a sequence of 16-QAM frames and writes a WAV file.

Frame structure per chunk:
  [ Zadoff-Chu preamble (ZC_LENGTH syms) ]
  [ pilot-interleaved data (raw_data_syms + n_pilots syms) ]

Standalone usage:
    python qam_writer.py                        # encodes built-in demo text
    python qam_writer.py config.json            # reads input_str + params from JSON
    python qam_writer.py --print-schema         # print example config

Config JSON (all optional):
{
    "input_str":    "Hello world",
    "output":       "output.wav",
    "config_out":   "qam_config.json",
    "chunk_bytes":  512,
    "ecc_bytes":    8,
    "seq_bytes":    2,
    "sample_rate":  48000,
    "carrier_freq": 12000,
    "symbol_rate":  6000
}
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import upfirdn
from scipy.io import wavfile
from reedsolo import RSCodec



class Modulator:
    PILOT_INTERVAL: int     = 8
    PILOT_SYMBOL:   complex = (1 + 1j) / math.sqrt(2)
    ZC_LENGTH:      int     = 63
    ZC_ROOT:        int     = 25
    RRC_SPAN:       int     = 6
    PRBS_SEED:      int     = 0xACE1

    _CONSTELLATION: dict[tuple, complex] = {
        (0,0,0,0): -3-3j, (0,0,0,1): -3-1j, (0,0,1,1): -3+1j, (0,0,1,0): -3+3j,
        (0,1,1,0): -1+3j, (0,1,1,1): -1+1j, (0,1,0,1): -1-1j, (0,1,0,0): -1-3j,
        (1,1,0,0):  1-3j, (1,1,0,1):  1-1j, (1,1,1,1):  1+1j, (1,1,1,0):  1+3j,
        (1,0,1,0):  3+3j, (1,0,1,1):  3+1j, (1,0,0,1):  3-1j, (1,0,0,0):  3-3j,
    }

    def __init__(
        self,
        input_str:    str,
        chunk_bytes:  int = 512,
        ecc_bytes:    int = 8,
        seq_bytes:    int = 2,
        sample_rate:  int = 48000,
        carrier_freq: int = 12000,
        symbol_rate:  int = 6000,
    ) -> None:
        self.input_str    = input_str
        self.CHUNK_BYTES  = chunk_bytes
        self.ECC_BYTES    = ecc_bytes
        self.SEQ_BYTES    = seq_bytes
        self.sample_rate  = sample_rate
        self.carrier_freq = carrier_freq
        self.symbol_rate  = symbol_rate

        self.samples_per_symbol: int = sample_rate // symbol_rate
        self._validate()

        self.RRC_BETA: float = self._auto_beta()
        self.rrc              = self._rrc_filter(self.RRC_BETA, self.RRC_SPAN,
                                                  self.samples_per_symbol)
        self.preamble_syms    = self._make_zadoff_chu(self.ZC_LENGTH, self.ZC_ROOT)
        self.n_pre:       int = len(self.preamble_syms)
        self.rsc              = RSCodec(self.ECC_BYTES)

        self._raw_data_syms: int = (self.SEQ_BYTES + self.CHUNK_BYTES + self.ECC_BYTES) * 2
        self._n_pilots: int = math.ceil(self._raw_data_syms / self.PILOT_INTERVAL)
        self.frame_syms_with_pilots: int = self._raw_data_syms + self._n_pilots

        self.tx_signal:     np.ndarray | None = None
        self.scale_factor:  float | None      = None
        self.total_frames:  int               = 0

        bw = symbol_rate * (1 + self.RRC_BETA)
        print(
            f"RRC_BETA={self.RRC_BETA}  BW={bw:.0f} Hz  "
            f"[{carrier_freq - bw/2:.0f}-{carrier_freq + bw/2:.0f} Hz]  "
            f"{self.samples_per_symbol} samp/sym",
            flush=True
        )
        print(
            f"Frame: {self.n_pre} ZC + {self.frame_syms_with_pilots} data+pilots "
            f"({self._n_pilots} pilots @ interval {self.PILOT_INTERVAL})",
            flush=True
        )

    def encode(self) -> np.ndarray:
        tx_bytes = self.input_str.encode("utf-8")
        pad      = (self.CHUNK_BYTES - len(tx_bytes) % self.CHUNK_BYTES) % self.CHUNK_BYTES
        tx_bytes += b"\x00" * pad

        chunks = [
            tx_bytes[i : i + self.CHUNK_BYTES]
            for i in range(0, len(tx_bytes), self.CHUNK_BYTES)
        ]
        self.total_frames = len(chunks)

        all_symbols: list[complex] = []
        for seq_idx, chunk in enumerate(chunks):
            seq_b   = seq_idx.to_bytes(self.SEQ_BYTES, "big")
            encoded = bytes(self.rsc.encode(seq_b + chunk))

            raw_bits = np.array([int(b) for byte in encoded for b in f"{byte:08b}"])
            scr_bits = self._scramble(raw_bits)
            data_syms, _ = self._bits_to_symbols(scr_bits)
            mixed    = self.insert_pilots(data_syms)

            all_symbols.extend(self.preamble_syms)
            all_symbols.extend(mixed)

        symbols  = np.array(all_symbols)
        bb       = upfirdn(self.rrc, symbols, self.samples_per_symbol)
        t        = np.arange(len(bb)) / self.sample_rate
        passband = (bb.real * np.cos(2 * np.pi * self.carrier_freq * t)
                  - bb.imag * np.sin(2 * np.pi * self.carrier_freq * t))

        self.scale_factor = 32700.0 / np.max(np.abs(passband))
        self.tx_signal    = passband
        return passband

    def write_wav(self, filename: str = "output.wav") -> None:
        if self.tx_signal is None:
            self.encode()
        pcm = (self.tx_signal * self.scale_factor).astype(np.int16)
        wavfile.write(filename, self.sample_rate, pcm)
        dur = len(self.tx_signal) / self.sample_rate
        print(f"Written {filename}  ({dur:.2f} s,  {self.total_frames} frames)",
              flush=True)

    def write_config(self, filename: str = "config.json") -> None:
        if self.scale_factor is None:
            self.encode()
        cfg = {
            "input_str":              self.input_str,
            "CHUNK_BYTES":            self.CHUNK_BYTES,
            "ECC_BYTES":              self.ECC_BYTES,
            "SEQ_BYTES":              self.SEQ_BYTES,
            "total_frames":           self.total_frames,
            "sample_rate":            self.sample_rate,
            "carrier_freq":           self.carrier_freq,
            "symbol_rate":            self.symbol_rate,
            "RRC_BETA":               float(self.RRC_BETA),
            "ZC_LENGTH":              self.ZC_LENGTH,
            "ZC_ROOT":                self.ZC_ROOT,
            "PILOT_INTERVAL":         self.PILOT_INTERVAL,
            "PILOT_SYMBOL_RE":        float(self.PILOT_SYMBOL.real),
            "PILOT_SYMBOL_IM":        float(self.PILOT_SYMBOL.imag),
            "frame_syms_with_pilots": self.frame_syms_with_pilots,
            "scale_factor":           float(self.scale_factor),
        }
        with open(filename, "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"Written {filename}", flush=True)

    def insert_pilots(self, data_syms: np.ndarray) -> np.ndarray:
        out: list[complex] = []
        for i, sym in enumerate(data_syms):
            if i % self.PILOT_INTERVAL == 0:
                out.append(self.PILOT_SYMBOL)
            out.append(sym)
        return np.array(out)

    def _validate(self) -> None:
        if self.sample_rate % self.symbol_rate != 0:
            # Auto-snap to nearest valid (lower) divisor instead of crashing
            best = 1
            for d in range(1, self.symbol_rate + 1):
                if self.sample_rate % d == 0:
                    best = d
            print(f"symbol_rate {self.symbol_rate} not a divisor of "
                  f"{self.sample_rate} - snapping to {best}", flush=True)
            self.symbol_rate         = best
            self.samples_per_symbol  = self.sample_rate // self.symbol_rate
        if self.samples_per_symbol < 4:
            raise ValueError(
                f"symbol_rate ({self.symbol_rate}) too high - only "
                f"{self.samples_per_symbol} samples/symbol (need >= 4)"
            )

    def _auto_beta(self) -> float:
        beta_a = self.carrier_freq / (self.symbol_rate * 1.5) - 1.0
        beta_b = (self.sample_rate - 2 * self.carrier_freq) / self.symbol_rate - 1.0
        if beta_a < 0.1:
            raise ValueError(
                f"carrier_freq ({self.carrier_freq} Hz) too low. "
                f"Minimum: {int(self.symbol_rate * 1.5 * 1.1)} Hz"
            )
        if beta_b < 0.1:
            raise ValueError(
                "Upper sideband exceeds Nyquist. Lower symbol_rate or raise carrier_freq."
            )
        return float(min(0.99, round(min(beta_a, beta_b) - 0.01, 4)))

    def _scramble(self, bits: np.ndarray) -> np.ndarray:
        return (bits ^ self._make_prbs(len(bits))).astype(int)

    def _bits_to_symbols(self, bits: np.ndarray) -> tuple[np.ndarray, int]:
        padding = (4 - len(bits) % 4) % 4
        if padding:
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])
        syms = [
            self._CONSTELLATION[tuple(bits[i : i + 4])]
            for i in range(0, len(bits), 4)
        ]
        return np.array(syms) / np.sqrt(10), padding

    def _make_prbs(self, length: int) -> np.ndarray:
        state = self.PRBS_SEED
        bits  = np.zeros(length, dtype=int)
        for i in range(length):
            bit     = ((state >> 15) ^ (state >> 14) ^ (state >> 12) ^ (state >> 3)) & 1
            bits[i] = (state >> 15) & 1
            state   = ((state << 1) | bit) & 0xFFFF
        return bits

    @staticmethod
    def _make_zadoff_chu(length: int = 63, root: int = 25) -> np.ndarray:
        n = np.arange(length)
        return np.exp(-1j * np.pi * root * n * (n + 1) / length)

    @staticmethod
    def _rrc_filter(beta: float, span: int, sps: int) -> np.ndarray:
        N = span * sps
        if N % 2 == 0:
            N += 1
        t = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1) / float(sps)
        h = np.zeros(len(t))
        for i, ti in enumerate(t):
            if ti == 0.0:
                h[i] = 1.0 - beta + (4.0 * beta / np.pi)
            elif abs(abs(ti) - 1.0 / (2.0 * beta)) < 1e-6:
                h[i] = (beta / np.sqrt(2.0)) * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                  + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
                )
            elif abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-6:
                ti2  = ti + 1e-7
                h[i] = (
                    np.sin(np.pi * ti2 * (1.0 - beta))
                    + 4.0 * beta * ti2 * np.cos(np.pi * ti2 * (1.0 + beta))
                ) / (np.pi * ti2 * (1.0 - (4.0 * beta * ti2) ** 2))
            else:
                h[i] = (
                    np.sin(np.pi * ti * (1.0 - beta))
                    + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
                ) / (np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2))
        h /= np.sqrt(np.sum(h ** 2))
        return h



EXAMPLE_CONFIG = {
    "input_str":    "Hello world",
    "output":       "output.wav",
    "config_out":   "qam_config.json",
    "chunk_bytes":  512,
    "ecc_bytes":    8,
    "seq_bytes":    2,
    "sample_rate":  48000,
    "carrier_freq": 12000,
    "symbol_rate":  6000,
}

DEFAULT_MSG = """Do you ever get bored doing the same job every day?
Son, let me tell you about stirring.
You grab that stick, and you just move it around, and you stir it around.
You get yourself into a rhythm.
It's a beautiful thing.
You know, Dad, the more I think about it,
maybe the honey field just isn't right for me.
You were thinking of what, making balloon animals?
That's a bad job for a guy with a stinger.
Janet, your son's not sure he wants to go into honey!
Barry, you are so funny sometimes.
I'm not trying to be funny.
You're not funny! You're going into honey. Our son, the stirrer!
You're gonna be a stirrer?
No one's listening to me!
Wait till you see the sticks I have.
I could say anything right now.
I'm gonna get an ant tattoo!
Let's open some honey and celebrate!
Maybe I'll pierce my thorax. Shave my antennae. Shack up with a grasshopper.
Get a gold tooth and call everybody "dawg"!
I'm so proud.
We're starting work today!
Today's the day.
"""


def encode_from_config(cfg: dict) -> dict:
    """Encode using a config dict. Returns result dict."""
    mod = Modulator(
        input_str    = cfg.get("input_str",    DEFAULT_MSG),
        chunk_bytes  = int(cfg.get("chunk_bytes",  512)),
        ecc_bytes    = int(cfg.get("ecc_bytes",    8)),
        seq_bytes    = int(cfg.get("seq_bytes",    2)),
        sample_rate  = int(cfg.get("sample_rate",  48000)),
        carrier_freq = int(cfg.get("carrier_freq", 12000)),
        symbol_rate  = int(cfg.get("symbol_rate",  6000)),
    )
    mod.encode()
    mod.write_wav(cfg.get("output",     "output.wav"))
    mod.write_config(cfg.get("config_out", "qam_config.json"))
    return {
        "status":       "ok",
        "total_frames": mod.total_frames,
        "duration_s":   round(len(mod.tx_signal) / mod.sample_rate, 2),
        "wav_path":     cfg.get("output",     "output.wav"),
        "config_path":  cfg.get("config_out", "qam_config.json"),
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
        print(f"Config '{config_path}' not found, using defaults.", flush=True)

    result = encode_from_config(cfg)
    print(json.dumps(result), flush=True)
    sys.exit(0 if result["status"] == "ok" else 1)