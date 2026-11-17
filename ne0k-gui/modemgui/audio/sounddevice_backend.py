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
                   Source, StreamSpec, as_mono_f32, spread)

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
                self.overflows = getattr(self, "overflows", 0) + 1
            # copy: indata is PortAudio's buffer, reused for the next callback.  as_mono_f32 returns a single
            # contiguous float32 channel as is, and the pipe queues blocks to its consumers -- without the copy the
            # decoder read blocks already overwritten by later audio: macOS/Windows RX never decoded, 2026-11-09
            on_block(as_mono_f32(indata[:, 0]).copy())
        try:
            # latency="high": the device buffers enough to ride out the callback waiting for the GIL while the DSP
            # threads run on a slow host such as the macOS mesh VM.  RX latency is irrelevant to a half-duplex modem; TX
            # keeps its own setting.
            self._st = _sd.InputStream(device=self._dev, samplerate=self.spec.rate,
                                       blocksize=self.spec.block, channels=1,
                                       dtype="float32", callback=cb, latency="high")
            self._st.start()
        except Exception as e:
            raise AudioError(f"cannot open input: {e}") from e

    def stop(self) -> None:
        if self._st:
            self._st.abort()
            self._st.close()
            self._st = None
            print(f"sd input closed: {getattr(self, 'overflows', 0)} overflow callbacks", flush=True)


def _out_channels(dev) -> int:
    """Channels to open a sink with: 2 on a stereo (or wider) device, so ``StreamSpec.tx_channels`` decides where the
    mono signal plays.  A one-channel stream reaches the left channel only on CoreAudio (a MacBook played the modem
    on one speaker, quieter than a stereo file at the same volume, 2026-11-16), while PulseAudio upmixes it to both."""
    try:
        n = int(_sd.query_devices(dev, "output")["max_output_channels"])
    except Exception:
        return 1
    return 2 if n >= 2 else 1


class SdSink(Sink):
    def __init__(self, dev, spec: StreamSpec) -> None:
        self.spec, self._dev, self._st = spec, dev, None

    def start(self, pull: Callable[[int], np.ndarray]) -> None:
        def cb(outdata, frames, t, status):
            if status:
                self.error = str(status)
            outdata[:] = spread(pull(frames), outdata.shape[1], self.spec.tx_channels)
        try:
            self.channels = _out_channels(self._dev)
            self._st = _sd.OutputStream(device=self._dev, samplerate=self.spec.rate,
                                        blocksize=self.spec.block, channels=self.channels,
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
