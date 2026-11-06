# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Connector API: the one interface every audio endpoint implements.

The modems work on REAL, MONO, float32 audio at a fixed sample rate.  Anything
that is not that (a stereo sound card, a PulseAudio sink, a PlutoSDR delivering
complex baseband) is adapted INSIDE its connector, so nothing above this layer
needs to know what is physically underneath.  For an SDR that means the
connector owns the (de)modulation -- e.g. SSB/NFM -- and presents an audio-like
stream.

Direction and ownership:
  * a Source PUSHES blocks to a callback from its own thread;
  * a Sink is PULLED: it calls ``pull(n)`` from its own thread whenever it needs
    n samples, and must be given exactly n back (silence when nothing is queued).
    That matches PortAudio's callback model and gives pacing to pipe-based
    sinks for free.

Device addresses are strings, ``<connector>:<device id>`` (``pulse:alsa_output.x``,
``sd:3``, ``file:/tmp/in.wav``), so a CLI flag or a saved setting is one value.
"""
from __future__ import annotations

import abc
import dataclasses
from typing import Callable

import numpy as np

SOURCE = "source"
SINK = "sink"


class AudioError(RuntimeError):
    """A device could not be opened, or died while running."""


@dataclasses.dataclass(frozen=True)
class StreamSpec:
    rate: int = 48000      # Hz; the connector asks the OS/device for exactly this
    block: int = 1024      # samples per callback (a hint; sinks may be exact)


@dataclasses.dataclass(frozen=True)
class DeviceInfo:
    connector: str
    dev_id: str
    name: str
    kind: str                      # SOURCE | SINK
    default: bool = False
    monitor: bool = False          # a source that taps a sink's output
    rate_hint: int | None = None   # device native rate, informational
    extra: dict = dataclasses.field(default_factory=dict, compare=False, hash=False)

    @property
    def address(self) -> str:
        return f"{self.connector}:{self.dev_id}"


class Source(abc.ABC):
    """Produces mono float32 blocks."""

    spec: StreamSpec

    @abc.abstractmethod
    def start(self, on_block: Callable[[np.ndarray], None]) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    # Set by the connector when the stream ends or dies, so the pipe can report it.
    error: str | None = None
    finished: bool = False         # True once a finite source (a file) is exhausted


class Sink(abc.ABC):
    """Consumes mono float32 blocks by pulling them."""

    spec: StreamSpec
    error: str | None = None

    @abc.abstractmethod
    def start(self, pull: Callable[[int], np.ndarray]) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...


class Connector(abc.ABC):
    """A family of endpoints: system audio, PulseAudio, a file, an SDR driver."""

    name: str = ""           # address prefix; unique
    description: str = ""

    def available(self) -> tuple[bool, str]:
        """(usable on this host, reason if not).  Must be cheap and never raise."""
        return True, ""

    @abc.abstractmethod
    def list_devices(self) -> list[DeviceInfo]: ...

    @abc.abstractmethod
    def open_source(self, dev_id: str, spec: StreamSpec) -> Source: ...

    @abc.abstractmethod
    def open_sink(self, dev_id: str, spec: StreamSpec) -> Sink: ...


def as_mono_f32(x: np.ndarray) -> np.ndarray:
    """Normalise whatever a driver hands over to contiguous mono float32."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return np.ascontiguousarray(x)
