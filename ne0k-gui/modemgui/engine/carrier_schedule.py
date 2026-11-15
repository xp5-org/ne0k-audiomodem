# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Carrier-schedule OFDM: a piano roll for the data carriers.

A *carrier score* is a list of steps,
each a set of MIDI notes.  On a link with ``OfdmConfig.schedule_tx`` set, every chunk of a DATA frame
plays the next step: its bit-allocation table is cut down to carriers consonant with those notes until they carry ``schedule_step_carriers`` typical carriers' bits per symbol, so every
note lasts the same time.  Nothing else changes -- same chunks, same symbols, same preamble / CE / pilots -- and the
music follows the modem's pace, not the MIDI's tempo.  A resent frame repeats its own steps.

The receiver holds the same score (``schedule_rx``).  It cannot read the step from the frame, so it ranks the next few expected steps by how well each step's carrier set matches the burst's
measured spectrum and decodes with the best one first.

Pitch: the carrier grid is fixed (``sc_spacing_hz``, 46.875 Hz at 48 kHz / n_fft 1024), so a carrier is a harmonic of
that spacing and only carriers above ~``RES_SEMITONE`` spacings apart resolve a semitone.  The score's pitch span is
transposed to sit as high in the band as it fits, above that resolution floor where it can; notes
that still fall outside are octave-folded.  Expect "recognisable", not "in tune".

score JSON:  {"version": 1, "name": str, "source": str, "steps": [{"notes": [midi, ...], "beats": float}, ...]}
"""
from __future__ import annotations

import json
import math

import numpy as np

SCORE_VERSION = 1
RES_SEMITONE = 0.75          # a semitone must span at least this many carrier spacings to count as resolved
# a note's carriers, in the order they are taken: the note, its octave, the octave below, 3rd / 4th harmonic, the
# fifth above, ... -- consonant with the note, so a note that needs more carriers still sounds like that note
NOTE_RATIOS = (1.0, 2.0, 0.5, 3.0, 4.0, 1.5, 6.0, 0.25, 8.0, 3.0 / 4.0, 5.0, 2.5, 1.0 / 3.0, 7.0)


def load_score(path: str) -> dict:
    with open(path) as f:
        sc = json.load(f)
    steps = sc.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{path}: carrier score has no steps")
    for i, st in enumerate(steps):
        notes = st.get("notes") if isinstance(st, dict) else None
        if not notes or not all(isinstance(n, int) and 0 <= n < 128 for n in notes):
            raise ValueError(f"{path}: step {i} needs a non-empty list of MIDI notes")
    return sc


def _et(n: float) -> float:
    return 440.0 * 2 ** ((n - 69) / 12.0)


def fit_transpose(lo: int, hi: int, f_min: float, f_max: float) -> int:
    """Semitone transposition putting [lo, hi] inside [f_min, f_max], as high as it goes.  When the span is wider than the band, the top is kept and the bottom folds."""
    return math.floor(12 * math.log2(f_max / _et(hi)))


class CarrierScore:
    """A score mapped onto one link's carrier set (``hs_data_cfg`` of that direction's band)."""

    def __init__(self, score: dict, cfg: dict, note_carriers: int = 4) -> None:
        self.name = str(score.get("name") or score.get("source") or "score")
        self.steps = [sorted(set(st["notes"])) for st in score["steps"]]
        self.note_carriers = max(1, int(note_carriers))
        sc = float(cfg["sc_spacing_hz"])
        self.sc, self.n_fft = sc, int(cfg["n_fft"])
        occ = [cfg["bin_lo"] + i for i in range(cfg["n_occupied"])]
        self.data_bins = np.array([b for i, b in enumerate(occ) if i % cfg["pilot_spacing"]], dtype=int)
        f_lo, f_hi = self.data_bins[0] * sc, self.data_bins[-1] * sc
        floor = sc * RES_SEMITONE / (2 ** (1 / 12) - 1)          # semitone >= RES_SEMITONE carriers above here
        notes = [n for st in self.steps for n in st]
        lo, hi = min(notes), max(notes)
        self.f_min, self.f_max = max(f_lo, min(floor, f_hi / 2)), f_hi   # keep at least an octave
        self.transpose = fit_transpose(lo, hi, self.f_min, self.f_max)
        self.folded = sorted({n for n in notes if not self.f_min <= _et(n + self.transpose) <= f_hi})
        self.cands = [self._cands(st) for st in self.steps]
        self.masks = [self._mask_of(c, np.ones(len(self.data_bins), int)) for c in self.cands]

    def tone(self, n: int) -> float:
        f = _et(n + self.transpose)
        while f < self.f_min:
            f *= 2
        while f > self.f_max:
            f /= 2
        return f

    def _idx(self, f: float) -> int:
        """Data-carrier index nearest frequency f."""
        return int(np.argmin(np.abs(self.data_bins * self.sc - f)))

    def _cands(self, notes: list[int]) -> list[int]:
        """Data-carrier indices for a step, in the order taken (NOTE_RATIOS of each note, notes round-robin)."""
        f_lo = self.data_bins[0] * self.sc - self.sc / 2
        f_hi = self.f_max + self.sc / 2
        per = [[self._idx(self.tone(n) * r) for r in NOTE_RATIOS if f_lo <= self.tone(n) * r <= f_hi] for n in notes]
        out = []
        for k in range(max(len(p) for p in per)):
            for p in per:
                if k < len(p) and p[k] not in out:
                    out.append(p[k])
        return out

    def _mask_of(self, cands: list[int], bat: np.ndarray, target: int | None = None) -> np.ndarray:
        """Carriers of ``cands`` taken until their bits reach ``target``
        (default: note_carriers of this table's typical carrier), so every note carries the same bits per symbol
        and lasts the same time whatever its pitch."""
        bat = np.asarray(bat)
        live = bat[bat > 0]
        if target is None:
            target = self.note_carriers * (int(np.median(live)) if len(live) else 1)
        m = np.zeros(len(bat), dtype=bool)
        got = 0
        # short of consonant candidates: the carriers nearest the note fill the rest
        rest = sorted((i for i in range(len(bat)) if i not in cands), key=lambda i: (abs(i - cands[0]), i)) \
            if cands else list(range(len(bat)))
        for i in list(cands) + rest:
            for j in (i, i - 1, i + 1):
                if 0 <= j < len(bat) and bat[j] > 0 and not m[j]:
                    m[j] = True
                    got += int(bat[j])
                    break
            if got >= target:
                break
        return m

    def __len__(self) -> int:
        return len(self.steps)

    def mask(self, step: int) -> np.ndarray:
        return self.masks[step % len(self.masks)]

    def bat(self, bat: np.ndarray, step: int) -> np.ndarray:
        """``bat`` with every carrier outside this step's set switched off.  A note carrier the channel cannot carry
        moves to a usable neighbour.  A step left with nothing usable falls back to the full table, so a
        frame always goes out."""
        bat = np.asarray(bat)
        m = self._mask_of(self.cands[step % len(self.cands)], bat)
        out = np.where(m, bat, 0).astype(bat.dtype)
        return out if out.any() else bat.copy()

    def describe(self, step: int) -> str:
        names = "C C# D D# E F F# G G# A A# B".split()
        notes = self.steps[step % len(self.steps)]
        return " ".join(f"{names[n % 12]}{n // 12 - 1}" for n in notes)

    def summary(self) -> str:
        return (f"score '{self.name}': {len(self.steps)} steps, transpose {self.transpose:+d} st, "
                f"notes {self.f_min:.0f}-{self.f_max:.0f} Hz on {self.sc:.1f} Hz carriers, "
                f"{self.note_carriers} carriers' bits per note, folded {len(self.folded)} pitch(es)")

    # -- transmitter: the frame's audio ------------------------------------------

    def audio(self, H, fs: int, body: bytes, bats, band, chunk_bytes: int,
              boost_db: float = 0.0, seq0: int = 0) -> np.ndarray:
        """``H.make_data_audio`` with chunk k of ``body`` on table ``bats[k]`` and the data carriers ``boost_db`` above the pilots.  A masked chunk has 2-4 note carriers next to ~7 unit-power pilots, so at 0 dB the pilots, a
        fixed chord, carry most of its energy and drown the tune.  CE and SYNC get the same boost on those carriers, so
        the receiver's channel estimate takes it in and a chunk of any QAM order decodes with the ordinary demodulator.  The
        chunks are numbered 0..n-1 exactly as make_data_audio numbers them."""
        from ofdm_modem import AdaptiveModulator
        if isinstance(bats, np.ndarray) and bats.ndim == 1:
            bats = [bats]
        gain = 10 ** (max(0.0, boost_db) / 20.0)
        cfg = H.hs_data_cfg(fs, band, chunk_bytes)

        class _Boosted(AdaptiveModulator):
            def __init__(self, cfg_, bat_):
                super().__init__(cfg_, bat_)
                # CE and SYNC carry the same boost on this chunk's note carriers, so the receiver's channel estimate
                # takes it in and equalised symbols land on the unit grid: 16/64-QAM decide on amplitude too
                data_occ = [i for i in range(self.n_occupied) if i not in self._pilot_set]
                on = [data_occ[d] for d in np.flatnonzero(self.bits_per_bin > 0)]
                self.ce_known = self.ce_known.copy()
                self.ce_known[on] *= gain
                self.sync_known = self.sync_known.copy()
                self.sync_known[on] *= gain

            def _data_occ(self, bits):
                occ = super()._data_occ(bits)
                data = np.ones(len(occ), dtype=bool)
                data[self.pilot_offsets] = False
                occ[data] *= gain
                return occ

        body = bytes(body)
        n = max(1, -(-len(body) // chunk_bytes))
        out = []
        for k in range(n):
            mod = (_Boosted if gain != 1.0 else AdaptiveModulator)(cfg, bats[min(k, len(bats) - 1)])
            a = mod.encode_bytes(body[k * chunk_bytes:(k + 1) * chunk_bytes], seq0=seq0 + k)
            out.append(a if k == n - 1 else a[:len(a) - mod.frame_gap_samp])   # one gap between chunks, as one encode
        return np.concatenate(out)

    # -- receiver: which step is this burst? ----------------------------------

    def spectrum(self, x: np.ndarray) -> np.ndarray:
        """Mean power per data carrier over the burst (non-overlapping n_fft blocks, Hann)."""
        n = self.n_fft
        k = len(x) // n
        if k < 1:
            return np.ones(len(self.data_bins))
        blocks = np.asarray(x[:k * n], dtype=np.float64).reshape(k, n) * np.hanning(n)
        p = np.mean(np.abs(np.fft.rfft(blocks, axis=1)) ** 2, axis=0)
        return p[self.data_bins] + 1e-20

    def rank(self, x: np.ndarray, bat: np.ndarray, steps: list[int]) -> list[int]:
        """``steps`` ordered best match first: mean log-power on the step's carriers minus on the rest."""
        lp = np.log10(self.spectrum(x))
        scored = []
        for s in steps:
            on = self.bat(bat, s) > 0
            if on.all() or not on.any():
                scored.append((0.0, s))
                continue
            scored.append((float(lp[on].mean() - lp[~on].mean()), s))
        return [s for _, s in sorted(scored, key=lambda t: -t[0])]
