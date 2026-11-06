# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""System audio devices through PortAudio (``sounddevice``) -- Linux/ALSA,
Windows (MME/WASAPI), macOS (CoreAudio).

Device ids are PortAudio device indices as strings (``sd:3``); the empty id is
the default device.  Indices are only stable while the hardware set is, so a
saved setting should prefer the device *name*: ``sd:name=USB Audio`` matches the
first device whose name contains that text.

``import sounddevice`` fails on a host without libportaudio; that is reported by
available() rather than at import time, so the other connectors still load.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .base import (SINK, SOURCE, AudioError, Connector, DeviceInfo, Sink,
                   Source, StreamSpec, as_mono_f32)

try:
    import sounddevice as _sd
    _IMPORT_ERR = ""
except Exception as e:                      # ImportError or OSError (no PortAudio)
    _sd = None
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


def _resolve(dev_id: str, kind: str):
    """'' -> default (None), '3' -> 3, 'name=foo' -> first matching device index."""
    if dev_id == "":
        return None
    if dev_id.startswith("name="):
        want = dev_id[5:].lower()
        key = "max_input_channels" if kind == SOURCE else "max_output_channels"
        for i, d in enumerate(_sd.query_devices()):
            if want in d["name"].lower() and d[key] > 0:
                return i
        raise AudioError(f"no {kind} device whose name contains {want!r}")
    try:
        return int(dev_id)
    except ValueError as e:
        raise AudioError(f"bad sounddevice id {dev_id!r}") from e


class SdSource(Source):
    def __init__(self, dev, spec: StreamSpec) -> None:
        self.spec, self._dev, self._st = spec, dev, None

    def start(self, on_block: Callable[[np.ndarray], None]) -> None:
        def cb(indata, frames, t, status):
            if status:
                self.error = str(status)       # overflow etc.; non-fatal, pipe counts it
            on_block(as_mono_f32(indata[:, 0]))
        try:
            self._st = _sd.InputStream(device=self._dev, samplerate=self.spec.rate,
                                       blocksize=self.spec.block, channels=1,
                                       dtype="float32", callback=cb)
            self._st.start()
        except Exception as e:
            raise AudioError(f"cannot open input: {e}") from e

    def stop(self) -> None:
        if self._st:
            self._st.abort()
            self._st.close()
            self._st = None


class SdSink(Sink):
    def __init__(self, dev, spec: StreamSpec) -> None:
        self.spec, self._dev, self._st = spec, dev, None

    def start(self, pull: Callable[[int], np.ndarray]) -> None:
        def cb(outdata, frames, t, status):
            if status:
                self.error = str(status)
            outdata[:, 0] = pull(frames)
        try:
            self._st = _sd.OutputStream(device=self._dev, samplerate=self.spec.rate,
                                        blocksize=self.spec.block, channels=1,
                                        dtype="float32", callback=cb)
            self._st.start()
        except Exception as e:
            raise AudioError(f"cannot open output: {e}") from e

    def stop(self) -> None:
        if self._st:
            self._st.abort()                  # abort, not stop: stop() drains the backlog
            self._st.close()
            self._st = None


class SoundDeviceConnector(Connector):
    name = "sd"
    description = "System audio devices (PortAudio: ALSA / WASAPI / CoreAudio)"

    def available(self) -> tuple[bool, str]:
        return (_sd is not None), _IMPORT_ERR

    def list_devices(self) -> list[DeviceInfo]:
        out = []
        try:
            default = _sd.default.device
        except Exception:
            default = (None, None)
        for i, d in enumerate(_sd.query_devices()):
            for kind, key, di in ((SOURCE, "max_input_channels", 0),
                                  (SINK, "max_output_channels", 1)):
                if d[key] > 0:
                    out.append(DeviceInfo(
                        self.name, str(i), d["name"], kind,
                        default=(i == default[di]),
                        rate_hint=int(d["default_samplerate"]),
                        extra={"channels": d[key],
                               "hostapi": _sd.query_hostapis(d["hostapi"])["name"]}))
        return out

    def open_source(self, dev_id: str, spec: StreamSpec) -> Source:
        return SdSource(_resolve(dev_id, SOURCE), spec)

    def open_sink(self, dev_id: str, spec: StreamSpec) -> Sink:
        return SdSink(_resolve(dev_id, SINK), spec)


CONNECTORS = [SoundDeviceConnector]
