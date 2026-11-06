# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Loopback check shared by the CLI and the UI's Test button."""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from ..audio import AudioPipe, Registry, StreamSpec


def _db(x: float) -> float:
    return 20 * math.log10(x) if x > 1e-9 else -180.0


def tone(freq: float, secs: float, rate: int, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(secs * rate)) / rate
    x = amp * np.sin(2 * np.pi * freq * t)
    ramp = min(int(0.01 * rate), len(x) // 2)          # 10 ms edges: no click
    if ramp:
        w = np.linspace(0, 1, ramp)
        x[:ramp] *= w
        x[-ramp:] *= w[::-1]
    return x.astype(np.float32)


def tone_power_db(x: np.ndarray, freq: float, rate: int) -> float:
    """Level of `freq` in x, dBFS-ish (single-bin projection)."""
    t = np.arange(len(x)) / rate
    z = np.dot(x, np.exp(-2j * np.pi * freq * t)) * 2 / max(len(x), 1)
    return _db(abs(z))


@dataclasses.dataclass
class LoopbackResult:
    ok: bool
    floor_db: float
    heard_db: float
    gain_db: float             # heard vs sent
    rx_samples: int
    tx_samples: int
    clipped: int
    error: str | None = None

    def summary(self) -> str:
        return (f"floor {self.floor_db:.1f} dB  heard {self.heard_db:.1f} dB  "
                f"({self.gain_db:+.1f} dB vs sent)  rx={self.rx_samples} "
                f"tx={self.tx_samples} clipped={self.clipped}  "
                f"{'PASS' if self.ok else 'FAIL: tone not heard'}")


def loopback(reg: Registry, tx: str, rx: str, rate: int = 48000, block: int = 1024,
             freq: float = 1500.0, seconds: float = 1.5) -> LoopbackResult:
    """Play a tone out of `tx` and look for it on `rx` using a private pipe."""
    with AudioPipe(reg, StreamSpec(rate, block)).open(rx, tx) as p:
        p.subscribe(4096)
        p.start()
        time.sleep(0.5)                                    # let capture settle
        noise = p.history()[0]
        floor = tone_power_db(noise, freq, rate) if len(noise) else -180.0
        p.play(tone(freq, seconds, rate))
        p.wait_tx(seconds + 5)
        time.sleep(0.5)
        heard, _ = p.history(int((seconds + 0.5) * rate))
        level = tone_power_db(heard, freq, rate)
        s = p.stats
    return LoopbackResult(level - floor > 20, floor, level, level - _db(0.3),
                          s.rx_samples, s.tx_samples, s.clipped,
                          s.rx_error or s.tx_error)
