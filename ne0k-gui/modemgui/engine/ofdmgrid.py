# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""The OFDM carrier grid the engine enforces: sample rate, FFT size and band, reconciled once, in one place.

Carrier k sits at k * fs / n_fft.  The PHY (ofdm_modem.build_config and the C ofdm_cfg_build) takes freq_lo /
freq_hi and SILENTLY rounds them onto that grid: bin_lo = max(1, ceil(lo/spacing)), bin_hi = min(n_fft/2 - 1,
floor(hi/spacing)), DC and Nyquist excluded.  So a UI band of 400-9000 Hz at n_fft 256 really runs 562.5-8812.5
Hz, nothing says so, and a combination with no usable carrier (n_fft 32 over 400-4000 Hz: two 1.5 kHz carriers,
one of them a pilot) is accepted and dies later as a dead link.

``check`` computes the same geometry as the PHY (``tests/test_ofdmgrid.py`` holds it to both the Python and the C
builder), refuses what cannot carry data, and gives the band snapped onto the carriers.  The engine runs it:
* at SET time on the control port (``validate_set``): a combination that cannot work is refused with the reason;
* at link start (``OfdmLink.__init__``): the link runs on the snapped band, so the log, the saved BAT and every
  later band computation name the frequencies actually on the air;
* as the base of the line-probe comb (``lineprobe.plan_for_grid``): comb tones sit ON the OFDM carriers, and an
  n_fft-point window resolves them with no bleed.

Every value here is exact: n_fft is a power of two, so spacing = fs / n_fft is an exact binary fraction and
bin * spacing round-trips through ceil/floor without drift (snapping a snapped band changes nothing).
"""
from __future__ import annotations

import dataclasses
import math

MIN_N_FFT = 32            # smallest size that streams in loopback on both backends (16 loses chunks, 8 fails the
MAX_N_FFT = 4096          # probe's measurement block: README "TODO: n_fft 16 and 8")
MIN_DATA_CARRIERS = 2
KEYS = ("fs", "n_fft", "freq_lo", "freq_hi", "pilot_spacing", "cp_ratio")   # fields that move the grid


class GridError(ValueError):
    """A sample rate / FFT size / band combination the modem cannot run."""


@dataclasses.dataclass(frozen=True)
class OfdmGrid:
    fs: int
    n_fft: int
    spacing_hz: float
    cp_len: int
    bin_lo: int
    bin_hi: int
    pilot_spacing: int
    asked_lo: float           # what the settings said
    asked_hi: float

    @property
    def n_occupied(self) -> int:
        return self.bin_hi - self.bin_lo + 1

    @property
    def pilot_bins(self) -> list[int]:
        return [self.bin_lo + i for i in range(0, self.n_occupied, self.pilot_spacing)]

    @property
    def data_bins(self) -> list[int]:
        return [self.bin_lo + i for i in range(self.n_occupied) if i % self.pilot_spacing]

    @property
    def n_pilots(self) -> int:
        return len(self.pilot_bins)

    @property
    def n_data(self) -> int:
        return self.n_occupied - self.n_pilots

    @property
    def freq_lo(self) -> float:
        return self.bin_lo * self.spacing_hz

    @property
    def freq_hi(self) -> float:
        return self.bin_hi * self.spacing_hz

    @property
    def snapped(self) -> bool:
        return self.freq_lo != self.asked_lo or self.freq_hi != self.asked_hi

    def carrier_hz(self, b: int) -> float:
        return b * self.spacing_hz

    def describe(self) -> str:
        s = (f"fs={self.fs} n_fft={self.n_fft} spacing={self.spacing_hz:g}Hz "
             f"band={self.freq_lo:g}-{self.freq_hi:g}Hz bins={self.bin_lo}-{self.bin_hi} "
             f"carriers={self.n_occupied} (data {self.n_data}, pilots {self.n_pilots}) cp={self.cp_len}")
        if self.snapped:
            s += f" [asked {self.asked_lo:g}-{self.asked_hi:g}Hz, snapped to carriers]"
        return s


def grid(fs: int, n_fft: int, freq_lo: float, freq_hi: float, pilot_spacing: int = 8,
         cp_ratio: float = 0.25) -> OfdmGrid:
    """The carrier grid for these settings, or GridError saying why it cannot run."""
    fs, n_fft, pilot_spacing = int(fs), int(n_fft), int(pilot_spacing)
    if fs <= 0:
        raise GridError(f"sample rate {fs} Hz")
    if n_fft < 4 or n_fft & (n_fft - 1):
        raise GridError(f"n_fft {n_fft} is not a power of two")
    if not MIN_N_FFT <= n_fft <= MAX_N_FFT:
        raise GridError(f"n_fft {n_fft} outside {MIN_N_FFT}-{MAX_N_FFT} (below {MIN_N_FFT} the stream or the "
                        f"probe does not work yet)")
    if pilot_spacing < 1:
        raise GridError(f"pilot spacing {pilot_spacing}")
    if not cp_ratio > 0:
        raise GridError(f"cyclic prefix ratio {cp_ratio}")
    if not freq_lo < freq_hi:
        raise GridError(f"band low {freq_lo:g} Hz is not below band high {freq_hi:g} Hz")
    spacing = fs / n_fft
    bin_lo = max(1, math.ceil(freq_lo / spacing))
    bin_hi = min(n_fft // 2 - 1, math.floor(freq_hi / spacing))
    cp_len = max(1, int(round(n_fft * cp_ratio)))
    g = OfdmGrid(fs, n_fft, spacing, cp_len, bin_lo, bin_hi, pilot_spacing, float(freq_lo), float(freq_hi))
    if bin_hi < bin_lo or g.n_data < MIN_DATA_CARRIERS:
        have = max(0, bin_hi - bin_lo + 1)
        raise GridError(f"{freq_lo:g}-{freq_hi:g} Hz at fs={fs}, n_fft={n_fft} holds {have} carrier(s) of "
                        f"{spacing:g} Hz ({max(0, g.n_data) if have else 0} for data, pilot every {pilot_spacing}); "
                        f"need {MIN_DATA_CARRIERS} data carriers: widen the band or raise n_fft")
    return g


def check(cfg) -> OfdmGrid:
    """``grid`` from an OfdmConfig (or anything with its field names)."""
    return grid(cfg.fs, cfg.n_fft, cfg.freq_lo, cfg.freq_hi, cfg.pilot_spacing, cfg.cp_ratio)


def snap(cfg):
    """(cfg with freq_lo / freq_hi on the carriers, its grid).  Raises GridError."""
    g = check(cfg)
    return dataclasses.replace(cfg, freq_lo=g.freq_lo, freq_hi=g.freq_hi), g


def validate_set(cfg, key: str, value):
    """The config a SET of ``key`` would produce, refused (GridError) when its grid cannot run."""
    new = dataclasses.replace(cfg, **{key: value})
    if key in KEYS:
        check(new)
    return new
