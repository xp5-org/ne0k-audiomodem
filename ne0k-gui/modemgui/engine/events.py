# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Plain-data events the engine publishes.  numpy arrays + scalars only, so the
same objects can later be serialised (msgpack) for a remote engine."""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class Event:
    pass


@dataclasses.dataclass(frozen=True)
class Status(Event):
    state: str                 # "closed" | "open" | "error"
    message: str = ""
    rx: str | None = None
    tx: str | None = None


@dataclasses.dataclass(frozen=True)
class Levels(Event):
    peak_db: float             # dBFS over the interval
    rms_db: float
    clipped: int               # cumulative since open
    tx_active: bool
    rx_samples: int
    dropped: int               # blocks this consumer lost (it fell behind)


@dataclasses.dataclass(frozen=True)
class Spectrum(Event):
    """A batch of new STFT columns, oldest first.  ``cols`` is (n, bins) dB."""
    index: int                 # absolute RX sample index of the first column's window start
    cols: np.ndarray
    tx_active: np.ndarray      # (n,) bool: column overlaps our own TX
    freqs: np.ndarray          # (bins,) Hz


@dataclasses.dataclass(frozen=True)
class ConstellationFrame(Event):
    """One demodulated OFDM frame (phase 5).  Defined now so the view can be
    built against it.  ``points`` is (n_symbols, n_carriers) complex, equalised;
    only the sync candidate that was REPORTED may be put here, and ``evm`` must be
    gain-weighted (see memory: constellation-points-per-attempt, -evm-gain-weighted)."""
    index: int
    points: np.ndarray
    carriers: np.ndarray       # carrier bin per column
    bits_per_carrier: np.ndarray
    evm: float
    backend: str               # "c" | "python"


@dataclasses.dataclass(frozen=True)
class Decoded(Event):
    """One CSS packet recovered from a received burst."""
    text: str
    src: int
    dst: int
    seq: int
    type_name: str
    ok: bool                   # no frame errors
    frames: int
    frames_errored: int
    snr_db: float
    rs_errors: int
    payload: bytes = b""      # raw packet payload (text is its lossy decode)
    acq_s: float | None = None  # est. seconds from burst start to preamble lock
    sync_score: float = 0.0     # preamble correlation score at lock (0..1)
    grid_sps: float | None = None   # fitted samples per symbol: the sender's clock error vs ours


@dataclasses.dataclass(frozen=True)
class LinkEvent(Event):
    """Something happened on the CSS link; drives the pianoroll / spectrogram overlay.

    kind: tx | rx | err | lock | ack | retry | fail | blockout.  ``node`` is the local node that
    observed or did it, ``peer`` the other end (-1 unknown).  ``t0``/``t1`` are
    time.monotonic() seconds: the airtime span for tx/rx bursts, t0 == t1 for points.
    ``lock`` = the decoder acquired the preamble and recovered frames (the CSS
    receiver has no separate PLL state to report).  ``blockout`` = an exclusive mesh-wide quiet
    window opened or closed (``MeshNode.blockout``); ``t0``..``t1`` spans the announced window
    (``t0``==``t1`` for the END/resume point), ``peer`` is the OTHER participant when this node
    is one of the two, else -1 (muted bystander)."""
    kind: str
    node: int
    peer: int
    t0: float
    t1: float
    detail: str = ""


@dataclasses.dataclass(frozen=True)
class ProbeReport(Event):
    """One per-carrier channel probe from the OFDM BAT exchange, for the bar graph.

    ``source`` says whose measurement it is:
      * ``measured``    -- THIS node graded the peer's probe (SNR and bits per carrier).
      * ``peer_report`` -- the peer graded OUR probe and sent the table back in its BAT report
        (bits only: the wire format carries no SNR, so ``snr_db`` is NaN).
    ``node`` is the node whose transmit path the table describes; ``measured_by`` who graded it.
    ``bits`` is -1 at pilot positions, else 0/1/2/4/6 bits per carrier."""
    source: str
    node: int
    measured_by: int
    freq_hz: np.ndarray
    snr_db: np.ndarray
    bits: np.ndarray
    summary: str = ""


@dataclasses.dataclass(frozen=True)
class OfdmData(Event):
    """A payload the OFDM link delivered (all modes)."""
    nbytes: int
    text: str
    payload: bytes = b""
