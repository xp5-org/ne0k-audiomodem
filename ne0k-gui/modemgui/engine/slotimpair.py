# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Targeted BAT-slot impairment: block or fuzz chosen OFDM carriers once the bit table is agreed.

A channel impairment that hits a random frequency usually lands on a carrier the table loads lightly,
or not at all, and proves nothing.  This one aims.  When the link has agreed its table, the slots are
picked FROM THAT TABLE by role, so every run hits the same kind of carrier:

    hot     the N carriers loaded with the most bits (the first to fail, the most bits lost)
    cold    the N loaded carriers with the fewest bits
    run     N adjacent data carriers, the window holding the most bits (a burst across the symbol)
    spread  N loaded carriers evenly through the band (one per RS byte, not all in one)
    pilot   N pilots, evenly spaced ("all" = every pilot): the phase/channel tracking, not the data
    idle    N carriers the table leaves at 0 bits (negative control: data must not notice)

and only frames of one CLASS are touched, so data and control are tested apart:

    data    payload frames (OfdmLink DATA/STREAM and the full-duplex/EC chunks; IqLink MSG)
    ctrl    control frames after the agreement (OfdmLink ACK/GO; IqLink READY/BAT/PING)
    any     both

Modes: ``block[@DEPTH_DB]`` (the slots attenuated, default 60 dB: gone) and ``fuzz[@SNR_DB]`` (Gaussian
noise added in the slots only, at SNR_DB below the signal measured in those slots; default 0 dB).

Spec: one or more rules, comma-separated:  ``ROLE:N:MODE[@LEVEL]:CLASS[:pin]``, e.g. ``hot:4:block:data``,
``pilot:all:fuzz@3:any``, ``idle:4:block:ctrl``.  ``none`` = no rule.

Tracking: by default a rule re-picks its slots whenever the table changes (an adversary that always hits
whatever is loaded most now).  ``:pin`` picks them once, from the first agreed table, and keeps those
frequencies: a fixed channel fault, which is what a receiver-driven table update (iq_link ARQ) must
route around.

The filter is applied at the transmitter (``attach``), after the link has agreed the table and before
the samples reach the device: only the TX side knows which table and which frame class a burst is,
so this is the one place the target can be exact.  To the far end it is the channel.  A whole burst
is filtered in one FFT of its own length (a brick-wall mask one carrier spacing wide per slot, applied
circularly), so there is no filter state between bursts.  Deterministic: the fuzz noise comes from ``seed``.

    gate = attach_ofdm(link, "hot:4:block:data")     # an OfdmLink (simple, real audio)
    gate = attach_iq(link, "run:8:fuzz@6:ctrl")      # an IqLink (complex I/Q)
    gate.report()  -> what was picked and how many bursts of each class were filtered or passed
"""
from __future__ import annotations

import dataclasses
import threading

import numpy as np

ROLES = ("hot", "cold", "run", "spread", "pilot", "idle")
CLASSES = ("data", "ctrl", "any")


@dataclasses.dataclass(frozen=True)
class Rule:
    role: str
    n: int | None          # None = all
    mode: str              # block | fuzz
    level: float           # block: depth dB; fuzz: SNR dB
    cls: str
    pin: bool = False      # slots picked once, from the first agreed table

    def __str__(self) -> str:
        return (f"{self.role}:{'all' if self.n is None else self.n}:{self.mode}@{self.level:g}:{self.cls}"
                + (":pin" if self.pin else ""))


def parse(spec: str | None) -> list[Rule]:
    spec = (spec or "").strip()
    if spec.lower() in ("", "none", "clean"):
        return []
    out = []
    for part in spec.split(","):
        f = part.strip().split(":")
        pin = len(f) == 5 and f[4] == "pin"
        if len(f) != 4 and not pin:
            raise ValueError(f"slot rule {part!r}: want ROLE:N:MODE[@LEVEL]:CLASS[:pin]")
        role, n, mode, cls = f[:4]
        if role not in ROLES:
            raise ValueError(f"slot rule {part!r}: role must be one of {ROLES}")
        if cls not in CLASSES:
            raise ValueError(f"slot rule {part!r}: class must be one of {CLASSES}")
        mode, _, lvl = mode.partition("@")
        if mode not in ("block", "fuzz"):
            raise ValueError(f"slot rule {part!r}: mode must be block or fuzz")
        level = float(lvl) if lvl else (60.0 if mode == "block" else 0.0)
        nn = None if n == "all" else int(n)
        if nn is not None and nn < 1:
            raise ValueError(f"slot rule {part!r}: N must be >= 1 or all")
        out.append(Rule(role, nn, mode, level, cls, pin))
    return out


# -- picking ----------------------------------------------------------------

def _even(idx: list[int], n: int | None) -> list[int]:
    """n of idx evenly spaced (all when n is None or >= len)."""
    if n is None or n >= len(idx):
        return list(idx)
    return [idx[int((k + 0.5) * len(idx) / n)] for k in range(n)]


def pick(role: str, n: int | None, bat, data_hz, pilot_hz) -> list[float]:
    """The carriers (Hz) a role names in this table.  Ties go to the lower carrier, so the same table
    always gives the same slots.  An empty list means the table has no such carrier (e.g. no idle one)."""
    bat = [int(b) for b in bat]
    data_hz = [float(h) for h in data_hz]
    if len(bat) != len(data_hz):
        raise ValueError(f"table has {len(bat)} entries for {len(data_hz)} data carriers")
    loaded = [i for i, b in enumerate(bat) if b > 0]
    if role == "pilot":
        return [float(h) for h in _even(list(pilot_hz), n)]
    if role == "idle":
        return [data_hz[i] for i in _even([i for i, b in enumerate(bat) if b == 0], n)]
    if role == "spread":
        return [data_hz[i] for i in _even(loaded, n)]
    k = len(loaded) if n is None else min(n, len(loaded))
    if role == "hot":
        sel = sorted(loaded, key=lambda i: (-bat[i], i))[:k]
    elif role == "cold":
        sel = sorted(loaded, key=lambda i: (bat[i], i))[:k]
    elif role == "run":
        w = len(bat) if n is None else min(n, len(bat))
        sums = [sum(bat[i:i + w]) for i in range(len(bat) - w + 1)]
        i0 = int(np.argmax(sums)) if sums else 0
        sel = [i for i in range(i0, i0 + w)]
    else:
        raise ValueError(role)
    return [data_hz[i] for i in sorted(sel)]


# -- the filter -------------------------------------------------------------

def apply(x: np.ndarray, fs: float, slots_hz, width_hz: float, mode: str, level: float,
          rng: np.random.Generator) -> np.ndarray:
    """``x`` with every slot (centre +- width/2) blocked or fuzzed, same length and dtype.  Complex
    input: slots are signed baseband Hz; real input: audio Hz."""
    if not len(slots_hz) or not len(x):
        return x
    cplx = np.iscomplexobj(x)
    n = len(x)                     # circular over the burst itself: no padding to truncate, so nothing leaks out of
    if cplx:                       # the slot (a burst starts and ends in its frame gap, so the wrap is silence)
        X = np.fft.fft(x.astype(np.complex128))
        f = np.fft.fftfreq(n, 1.0 / fs)
    else:
        X = np.fft.rfft(x.astype(np.float64))
        f = np.fft.rfftfreq(n, 1.0 / fs)
    m = np.zeros(len(f), bool)
    for s in slots_hz:
        m |= np.abs(f - float(s)) <= width_hz / 2
    if not m.any():
        return x
    if mode == "block":
        X[m] *= 10 ** (-level / 20)
    else:
        p_sig = float(np.mean(np.abs(X[m]) ** 2))
        z = rng.standard_normal(n) + (1j * rng.standard_normal(n) if cplx else 0)
        Z = np.fft.fft(z) if cplx else np.fft.rfft(z)
        p_z = float(np.mean(np.abs(Z[m]) ** 2)) or 1.0
        X[m] += Z[m] * np.sqrt(p_sig / p_z / 10 ** (level / 10))
    y = np.fft.ifft(X) if cplx else np.fft.irfft(X, n)
    return y.astype(x.dtype)


# -- the gate ----------------------------------------------------------------

class Gate:
    """Filters TX bursts by class once armed.  ``grid()`` returns (bat, data_hz, pilot_hz, fs, width)
    for the table now in use; slots are re-picked whenever that table changes."""

    def __init__(self, spec: str, grid, seed: int = 1, armed=None) -> None:
        self.rules = parse(spec)
        self.armed_fn = armed                   # () -> bool, polled per burst until it says yes
        self.spec = ",".join(map(str, self.rules)) or "none"
        self.grid = grid
        self.rng = np.random.default_rng(seed)
        self.armed = False
        self.cls = threading.local()
        self.counts = {"data": [0, 0], "ctrl": [0, 0], "other": [0, 0]}    # [filtered, passed]
        self.picked: list[dict] = []
        self._key = None
        self._slots: list[tuple[Rule, list[float]]] = []
        self.lock = threading.Lock()

    def _pick(self):
        bat, data_hz, pilot_hz, fs, width = self.grid()
        key = (tuple(int(b) for b in bat), round(float(fs)), round(float(width), 3))
        if key != self._key:
            self._key = key
            old = dict(self._slots)
            self._slots = [(r, old[r] if r.pin and r in old else pick(r.role, r.n, bat, data_hz, pilot_hz))
                           for r in self.rules]
            self.picked.append(dict(bits=int(sum(key[0])), fs=fs, width_hz=width, rules=[
                dict(rule=str(r), n_slots=len(s), slots_hz=[round(h, 1) for h in s],
                     bits_hit=int(sum(b for b, h in zip(bat, data_hz) if float(h) in set(s))))
                for r, s in self._slots]))
        return fs, width

    def filter(self, x: np.ndarray) -> np.ndarray:
        cls = getattr(self.cls, "v", "data")
        bucket = cls if cls in ("data", "ctrl") else "other"
        with self.lock:
            if not self.armed and self.armed_fn is not None:
                self.armed = bool(self.armed_fn())
            if not self.armed or not self.rules or bucket == "other":
                self.counts[bucket][1] += 1
                return x
            fs, width = self._pick()
            hit = False
            for r, slots in self._slots:
                if slots and r.cls in (cls, "any"):
                    x = apply(x, fs, slots, width, r.mode, r.level, self.rng)
                    hit = True
            self.counts[bucket][0 if hit else 1] += 1
            return x

    def hits(self, cls: str) -> int:
        """Bursts of a class the rules actually filtered ("any": all classes)."""
        return sum(v[0] for k, v in self.counts.items() if cls in (k, "any"))

    def report(self) -> dict:
        return dict(spec=self.spec, armed=self.armed, picked=self.picked,
                    bursts={k: dict(filtered=v[0], passed=v[1]) for k, v in self.counts.items()})


def _shadow_play_raw(session, gate: Gate) -> None:
    raw = session.play_raw
    session.play_raw = lambda samples: raw(gate.filter(samples))


def attach_ofdm(link, spec: str, seed: int = 1) -> Gate:
    """Gate an OfdmLink (before ``start``).  Armed while the link is up; the table is ``_tx_bat`` on
    ``_tx_band`` (the sub-band in full duplex).  Classes come from what ``_play`` says it sends: DATA/
    STREAM -> data, ACK/GO -> ctrl, the rest (handshake, probe, ID) never filtered.  Bursts played
    outside ``_play`` (the EC full-duplex chunks) are data."""
    from .ofdm_link import _imports, OfdmLink

    def grid():
        H, _ = _imports()
        c = link.cfg
        band = getattr(link, "_tx_band", None) or link._band
        g = H.hs_data_cfg(c.fs, band, c.chunk_bytes)
        df = c.fs / g["n_fft"]
        occ = [g["bin_lo"] + i for i in range(g["n_occupied"])]
        pil = [b * df for i, b in enumerate(occ) if i % g["pilot_spacing"] == 0]
        data = [b * df for b in OfdmLink._data_bins(g)]
        return link._tx_bat, data, pil, c.fs, df

    gate = Gate(spec, grid, seed, armed=lambda: bool(link.up) and getattr(link, "_tx_bat", None) is not None)
    play = link._play

    def _play(audio, what, ident=False):
        w = what.split(" ", 1)[0]
        gate.cls.v = "data" if w in ("DATA", "STREAM") else "ctrl" if w in ("ACK", "GO") else "other"
        try:
            return play(audio, what, ident=ident)
        finally:
            gate.cls.v = "data"

    link._play = _play
    _shadow_play_raw(link.session, gate)
    return gate


def attach_iq(link, spec: str, seed: int = 1) -> Gate:
    """Gate an IqLink.  Armed from the moment a probed TX table is set (``set_tx_bat``/``set_bat`` with
    a table), so the READY that negotiate sends at the new table is already in scope.  Classes by
    message magic: MSG -> data, BAT/READY/PING -> ctrl; probes never filtered."""
    from . import iq_link as il
    from .ofdm_cx import bins

    def grid():
        c, x = link.cfg, link.mod.x
        df = c.fs / x.c.n_fft
        occ = bins(x)
        sp = x.c.pilot_spacing
        pil = [b * df for i, b in enumerate(occ) if i % sp == 0]
        data = [b * df for i, b in enumerate(occ) if i % sp]
        return link.bat, data, pil, c.fs, df

    gate = Gate(spec, grid, seed)
    set_tx, set_bat, send_msg, probe = link.set_tx_bat, link.set_bat, link.send_message, link.send_probe

    def set_tx_bat(bat=None):
        set_tx(bat)
        gate.armed = gate.armed or bat is not None

    def set_bat_(bat=None):
        set_bat(bat)
        gate.armed = gate.armed or bat is not None

    def send_message(data, seq0=None, magic=il.MSG_MAGIC, mod=None, repeat_s=0.0):
        gate.cls.v = "data" if magic == il.MSG_MAGIC else "ctrl"
        try:
            return send_msg(data, seq0, magic, mod, repeat_s=repeat_s)
        finally:
            gate.cls.v = "data"

    def send_probe():
        gate.cls.v = "other"
        try:
            return probe()
        finally:
            gate.cls.v = "data"

    link.set_tx_bat, link.set_bat, link.send_message, link.send_probe = set_tx_bat, set_bat_, send_message, send_probe
    _shadow_play_raw(link.session, gate)
    return gate
