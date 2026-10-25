# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
meshchannel.py — per-link acoustic channel model for the mesh sims.

This module supplies the physical layer both tiers share:

  * geometry      — 2-D node positions give a propagation DELAY and an amplitude
                    GAIN per ordered (tx, rx) pair (spherical-spreading path loss);
  * noise buckets — each node has a profile (clean / noisy / very_noisy) setting
                    its receiver noise floor;
  * SINR + capture — a reception decodes iff its signal-to-(noise+interference)
                    ratio clears a capture margin.  This replaces the old
                    all-or-nothing collision: a strong near signal can CAPTURE
                    over a weak far interferer, which is how real radios behave.

Tier 1 (meshsim_mac.py) uses sinr_db()/decodes() to decide delivery abstractly.
Tier 2 (meshsim_dsp.py) additionally renders real audio, so it uses gain(),
prop_delay(), add_awgn() and the shape_channel()/notch() frequency responses to
build each node's mixed mic signal for a real decode.

Run:  python tests/meshchannel.py       (exit 0 = selftest passed)
"""
from __future__ import annotations

import math
import os
import sys


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

import numpy as np                       # noqa: E402
import css_modem as cm                   # noqa: E402

SPEED_OF_SOUND = cm.SPEED_OF_SOUND_M_S   # reuse the PHY's constant, don't redefine

# Receiver noise floor per profile, in dB relative to a unit-RMS transmit
# reference.  A node transmitting at unit level, heard at the reference distance
# with no interference, therefore lands at ≈ -floor dB SNR: clean ≈ 40 dB,
# noisy ≈ 22 dB, very_noisy ≈ 12 dB — a spread that spans "trivial" to "marginal"
# for the CSS PHY without being hopeless.
NOISE_BUCKETS = {
    "clean":      -40.0,
    "half_noisy": -31.0,   # midway clean↔noisy — a partial-recovery level
    "noisy":      -22.0,
    "very_noisy": -12.0,
}

# Default SINR a reception must clear to decode.  Tunable per-run; calibrated
# against the DSP tier (Tier 2) where a real decoder sets the true floor.
DEFAULT_CAPTURE_MARGIN_DB = 6.0


def add_awgn(audio, snr_db, rng):
    """Add white Gaussian noise at `snr_db` relative to the signal's RMS.

    Mirrors css_loopback.add_pink_noise (amplitude form: noise_rms =
    sig_rms / 10**(snr/20)) so the two impairment helpers compose the same way.
    """
    audio = np.asarray(audio, dtype=np.float32)
    sig_rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-20)
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    noise_rms_target = sig_rms / (10 ** (snr_db / 20.0))
    return (audio + noise * noise_rms_target).astype(np.float32)


class Channel2D:
    """A fixed constellation of nodes on a 2-D plane sharing one acoustic pool.

    positions : {node_id: (x_m, y_m)}
    profiles  : {node_id: "clean"|"noisy"|"very_noisy"}
    tx_levels : {node_id: float} transmit RMS (default 1.0 = the reference)

    ref_dist_m   — distance at which gain == 1.0 (near-field clamp)
    pathloss_exp — power falls as 1/dist**pathloss_exp (2.0 = spherical spreading)
    """

    def __init__(self, positions, profiles, *, tx_levels=None,
                 ref_dist_m=1.0, pathloss_exp=2.0,
                 capture_margin_db=DEFAULT_CAPTURE_MARGIN_DB, fs=48000, rng=None):
        self.pos = {int(k): (float(v[0]), float(v[1])) for k, v in positions.items()}
        self.profiles = {int(k): str(v) for k, v in profiles.items()}
        self.tx_levels = {int(k): float((tx_levels or {}).get(k, 1.0))
                          for k in self.pos}
        self.ref_dist = float(ref_dist_m)
        self.pathloss_exp = float(pathloss_exp)
        self.capture_margin_db = float(capture_margin_db)
        self.fs = int(fs)
        self.rng = rng or np.random.default_rng(0)
        for nid, prof in self.profiles.items():
            if prof not in NOISE_BUCKETS:
                raise ValueError(f"node {nid}: unknown noise profile {prof!r}")

    # ── geometry ─────────────────────────────────────────────────────────────
    def distance(self, a: int, b: int) -> float:
        (xa, ya), (xb, yb) = self.pos[a], self.pos[b]
        return math.hypot(xa - xb, ya - yb)

    def prop_delay(self, tx: int, rx: int) -> float:
        """Propagation delay tx→rx in seconds."""
        return self.distance(tx, rx) / SPEED_OF_SOUND

    def gain(self, tx: int, rx: int) -> float:
        """Amplitude gain tx→rx (≤ 1.0), spherical spreading clamped at ref_dist."""
        d = max(self.distance(tx, rx), self.ref_dist)
        return (self.ref_dist / d) ** (self.pathloss_exp / 2.0)

    # ── noise / SNR ──────────────────────────────────────────────────────────
    def noise_power(self, rx: int) -> float:
        floor_db = NOISE_BUCKETS[self.profiles[rx]]
        return 10.0 ** (floor_db / 10.0)          # (unit-RMS reference)**2 scale

    def rx_signal_power(self, tx: int, rx: int) -> float:
        amp = self.tx_levels[tx] * self.gain(tx, rx)
        return amp * amp

    def link_snr_db(self, tx: int, rx: int) -> float:
        """Noise-limited SNR of a lone tx→rx reception (no interferers)."""
        return 10.0 * math.log10(self.rx_signal_power(tx, rx)
                                 / self.noise_power(rx))

    def sinr_db(self, tx: int, rx: int, interferers) -> float:
        """SINR of tx→rx with `interferers` (an iterable of node ids) also on air.

        Interference adds in power; the wanted signal competes with noise + the
        summed interferer power at the receiver.
        """
        sig = self.rx_signal_power(tx, rx)
        intf = sum(self.rx_signal_power(i, rx) for i in interferers if i != tx)
        return 10.0 * math.log10(sig / (self.noise_power(rx) + intf))

    def decodes(self, tx: int, rx: int, interferers) -> bool:
        """Capture rule: does tx→rx clear the capture margin against interferers?"""
        return self.sinr_db(tx, rx, interferers) >= self.capture_margin_db

    # ── DSP-tier frequency responses (used by Tier 2) ────────────────────────
    def shape_channel(self, x, kind="good"):
        """Apply a band-limited acoustic frequency response (lifted from the
        meshlib selftest's `chan`).  No noise — Tier 2 adds noise per receiver."""
        x = np.asarray(x, dtype=np.float64)
        F = np.fft.rfftfreq(len(x), 1.0 / self.fs)
        if kind == "good":
            H = np.exp(-((F - 900) / 2600) ** 2) + 0.35 * np.exp(-((F - 2700) / 900) ** 2)
        else:
            H = 1 / (1 + np.exp(-(F - 2500) / 300)) * np.exp(-((F - 3500) / 2500) ** 2)
        y = np.fft.irfft(np.fft.rfft(x) * (H / H.max()), n=len(x))
        return y.astype(np.float32)

    def notch(self, x, kill_hz, width_hz=30.0):
        """Zero out narrow bands (dead carriers), lifted from meshlib's `notch`."""
        x = np.asarray(x, dtype=np.float64)
        F = np.fft.rfftfreq(len(x), 1.0 / self.fs)
        Hn = np.ones_like(F)
        for f0 in kill_hz:
            Hn[np.abs(F - f0) < width_hz] = 0.0
        return np.fft.irfft(np.fft.rfft(x) * Hn, n=len(x)).astype(np.float32)


# ── selftest ─────────────────────────────────────────────────────────────────
def _selftest() -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{name:38}] {'PASS' if cond else 'FAIL'}")

    # three nodes on a line: 0 at origin, 1 near, 2 far
    pos = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (20.0, 0.0)}
    prof = {0: "clean", 1: "clean", 2: "clean"}
    ch = Channel2D(pos, prof, fs=48000)

    # geometry
    check("prop delay grows with distance",
          ch.prop_delay(0, 2) > ch.prop_delay(0, 1) > 0)
    check("delay ~ dist/c", abs(ch.prop_delay(0, 1) - 2.0 / SPEED_OF_SOUND) < 1e-9)
    check("gain <= 1 and falls with distance",
          1.0 >= ch.gain(0, 1) > ch.gain(0, 2) > 0)
    check("gain symmetric", abs(ch.gain(0, 2) - ch.gain(2, 0)) < 1e-12)

    # SNR ordering: near link stronger than far link
    check("near SNR > far SNR", ch.link_snr_db(0, 1) > ch.link_snr_db(0, 2))

    # noise bucket lowers SNR by the expected delta (clean vs noisy at same geom)
    ch_noisy = Channel2D(pos, {0: "clean", 1: "noisy", 2: "clean"}, fs=48000)
    d = ch.link_snr_db(0, 1) - ch_noisy.link_snr_db(0, 1)
    check("noisy bucket drops SNR by bucket delta",
          abs(d - (NOISE_BUCKETS["noisy"] - NOISE_BUCKETS["clean"])) < 1e-6)

    # capture effect: strong near signal survives a weak far interferer;
    # the weak far signal does NOT survive the strong near one.
    strong_ok = ch.decodes(1, 0, interferers=[2])   # node1→0 with node2 jamming
    weak_ok = ch.decodes(2, 0, interferers=[1])     # node2→0 with node1 jamming
    check("strong signal captures over weak", strong_ok and not weak_ok)

    # symmetric two equal-power interferers at equal distance => neither captures
    sym = Channel2D({0: (0.0, 0.0), 1: (5.0, 0.0), 2: (-5.0, 0.0)},
                    {0: "clean", 1: "clean", 2: "clean"}, fs=48000,
                    capture_margin_db=6.0)
    check("equal-power collision: neither captures",
          not sym.decodes(1, 0, [2]) and not sym.decodes(2, 0, [1]))

    # lone weak link below margin fails on noise alone
    faraway = Channel2D({0: (0.0, 0.0), 1: (5000.0, 0.0)},
                        {0: "very_noisy", 1: "very_noisy"}, fs=48000)
    check("distant weak link fails on noise", not faraway.decodes(1, 0, []))

    # add_awgn hits the requested SNR within tolerance
    rng = np.random.default_rng(3)
    sig = np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000).astype(np.float32)
    noisy = add_awgn(sig, 10.0, rng)
    meas = 10 * np.log10(np.mean(sig ** 2) / (np.mean((noisy - sig) ** 2) + 1e-20))
    check("add_awgn achieves ~10 dB SNR", abs(meas - 10.0) < 0.5)

    return ok


def main() -> int:
    print("MESH CHANNEL (2-D pathloss + noise buckets + capture)\n")
    ok = _selftest()
    print(f"\n*** {'ALL PASS' if ok else 'FAILED'} ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
