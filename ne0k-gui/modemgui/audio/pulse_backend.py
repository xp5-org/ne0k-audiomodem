# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""PulseAudio (and PipeWire-pulse) connector, driven through pactl/parec/pacat.

Why subprocesses and not PortAudio: PortAudio's view of Pulse is whatever the
ALSA plugin exposes, which hides the thing this tool needs -- picking an
arbitrary *sink* by name, and capturing a sink's **monitor** (the audio a sink
is playing), which gives a software loopback between two modem instances with
no hardware.  parec/pacat also honour ``PULSE_SERVER`` unchanged, so a
container talking to the host over TCP needs no special code.

Device ids are Pulse object names (``alsa_output.pci-0000_00_1f.3.analog-stereo``);
the empty id ``pulse:`` means "the server's default".  Sinks list as
kind=sink, sources as kind=source, and a monitor source carries monitor=True.

Latency is a trade: the sink pulls ahead of the server by roughly
``latency_ms`` plus the pipe, so a Stop must kill pacat (dropping that backlog)
rather than wait for it to drain -- see PulseSink.stop().
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from typing import Callable

import numpy as np

from .base import (SINK, SOURCE, AudioError, Connector, DeviceInfo, Sink,
                   Source, StreamSpec, as_mono_f32)

_TOOLS = ("pactl", "parec", "pacat")


def _die_with_parent():
    """Linux only: have the kernel kill the child if this process dies.

    A leaked parec/pacat otherwise holds the capture device (and, if it
    inherited our stdout, the caller's pipe) until someone kills it by hand.
    """
    if not sys.platform.startswith("linux"):
        return None
    import ctypes
    import signal

    def _set():
        try:
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
        except OSError:
            pass
    return _set


def _spawn(argv: list[str], **kw) -> subprocess.Popen:
    return subprocess.Popen(argv, preexec_fn=_die_with_parent(),
                            stderr=subprocess.PIPE, **kw)


def _stderr_tail(p: subprocess.Popen) -> str:
    try:
        return p.stderr.read(2000).decode("utf-8", "replace").strip()
    except Exception:
        return ""


def _stop_proc(p: subprocess.Popen | None) -> None:
    if p is None:
        return
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    # Close our pipe ends now: a buffered stdin would otherwise flush into the
    # dead process when the Popen is garbage collected (BrokenPipeError noise).
    for f in (p.stdin, p.stdout, p.stderr):
        if f:
            try:
                f.close()
            except OSError:
                pass


def _pactl(*args: str, timeout: float = 4.0) -> str:
    r = subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise AudioError(f"pactl {' '.join(args)}: {r.stderr.strip() or r.returncode}")
    return r.stdout


def _rate_of(spec: str | None) -> int | None:
    # "float32le 2ch 48000Hz"
    for tok in (spec or "").split():
        if tok.endswith("Hz") and tok[:-2].isdigit():
            return int(tok[:-2])
    return None


def parse_json_listing(text: str, kind: str, default_name: str | None) -> list[DeviceInfo]:
    out = []
    for o in json.loads(text):
        name = o["name"]
        mon = kind == SOURCE and o.get("monitor_of_sink") not in (None, "n/a")
        if kind == SOURCE and name.endswith(".monitor"):
            mon = True
        out.append(DeviceInfo(
            "pulse", name, o.get("description") or name, kind,
            default=(name == default_name), monitor=mon,
            rate_hint=_rate_of(o.get("sample_specification")),
            extra={"state": o.get("state"), "index": o.get("index")}))
    return out


def parse_short_listing(text: str, kind: str, default_name: str | None) -> list[DeviceInfo]:
    """`pactl list short sinks` -- the fallback for servers without --format=json."""
    out = []
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) < 2:
            continue
        out.append(DeviceInfo(
            "pulse", f[1], f[1], kind, default=(f[1] == default_name),
            monitor=(kind == SOURCE and f[1].endswith(".monitor")),
            rate_hint=_rate_of(f[3]) if len(f) > 3 else None,
            extra={"state": f[4] if len(f) > 4 else None}))
    return out


class PulseSource(Source):
    def __init__(self, dev: str, spec: StreamSpec, latency_ms: int) -> None:
        self.spec, self._dev, self._lat = spec, dev, latency_ms
        self._proc: subprocess.Popen | None = None
        self._th: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self, on_block: Callable[[np.ndarray], None]) -> None:
        argv = ["parec", "--raw", "--format=float32le", "--channels=1",
                f"--rate={self.spec.rate}", f"--latency-msec={self._lat}"]
        if self._dev:
            argv.append(f"--device={self._dev}")
        try:
            self._proc = _spawn(argv, stdout=subprocess.PIPE)
        except OSError as e:
            raise AudioError(f"cannot run parec: {e}") from e
        self._stop.clear()
        self._th = threading.Thread(target=self._run, args=(on_block,),
                                    name="pulse-src", daemon=True)
        self._th.start()

    def _run(self, on_block) -> None:
        p, n = self._proc, self.spec.block * 4
        while not self._stop.is_set():
            buf = p.stdout.read(n)               # blocks; EOF when parec exits
            if not buf:
                if not self._stop.is_set():
                    self.error = _stderr_tail(p) or "parec exited"
                return
            buf = buf[: len(buf) // 4 * 4]
            if buf:
                on_block(as_mono_f32(np.frombuffer(buf, dtype="<f4")))

    def stop(self) -> None:
        self._stop.set()
        _stop_proc(self._proc)
        if self._th:
            self._th.join(timeout=2.0)


class PulseSink(Sink):
    def __init__(self, dev: str, spec: StreamSpec, latency_ms: int) -> None:
        self.spec, self._dev, self._lat = spec, dev, latency_ms
        self._proc: subprocess.Popen | None = None
        self._th: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self, pull: Callable[[int], np.ndarray]) -> None:
        argv = ["pacat", "--playback", "--raw", "--format=float32le", "--channels=1",
                f"--rate={self.spec.rate}", f"--latency-msec={self._lat}",
                "--client-name=modemgui"]
        if self._dev:
            argv.append(f"--device={self._dev}")
        try:
            self._proc = _spawn(argv, stdin=subprocess.PIPE)
        except OSError as e:
            raise AudioError(f"cannot run pacat: {e}") from e
        self._stop.clear()
        self._th = threading.Thread(target=self._run, args=(pull,),
                                    name="pulse-sink", daemon=True)
        self._th.start()

    def _run(self, pull) -> None:
        p, n = self._proc, self.spec.block
        try:
            while not self._stop.is_set():
                blk = as_mono_f32(pull(n))
                p.stdin.write(blk.astype("<f4", copy=False).tobytes())
                p.stdin.flush()                  # the blocking write is the pacing
        except (BrokenPipeError, ValueError, OSError):
            if not self._stop.is_set():
                self.error = _stderr_tail(p) or "pacat exited"

    def stop(self) -> None:
        # Kill, don't close-and-wait: closing stdin lets pacat play out its
        # buffered backlog, so Stop would keep making sound for the latency.
        self._stop.set()
        _stop_proc(self._proc)
        if self._th:
            self._th.join(timeout=2.0)


class PulseConnector(Connector):
    name = "pulse"
    description = "PulseAudio / PipeWire-pulse sinks, sources and sink monitors"

    def __init__(self, latency_ms: int = 40) -> None:
        self.latency_ms = int(os.environ.get("PULSE_LATENCY_MS", latency_ms))   # experiment: larger buffers vs sample slips

    def available(self) -> tuple[bool, str]:
        missing = [t for t in _TOOLS if shutil.which(t) is None]
        if missing:
            return False, f"missing {', '.join(missing)} (install pulseaudio-utils)"
        try:
            _pactl("info", timeout=3.0)
        except Exception as e:
            srv = os.environ.get("PULSE_SERVER", "<default>")
            why = " ".join(str(e).split())
            return False, f"no Pulse server reachable (PULSE_SERVER={srv}): {why}"
        return True, ""

    def _default(self, which: str) -> str | None:
        try:
            return _pactl(f"get-default-{which}").strip() or None
        except Exception:
            return None

    def list_devices(self) -> list[DeviceInfo]:
        out: list[DeviceInfo] = []
        for kind, noun in ((SINK, "sinks"), (SOURCE, "sources")):
            default = self._default(kind)
            try:
                out += parse_json_listing(_pactl("--format=json", "list", noun), kind, default)
            except (AudioError, ValueError, KeyError):
                out += parse_short_listing(_pactl("list", "short", noun), kind, default)
        return out

    def open_source(self, dev_id: str, spec: StreamSpec) -> Source:
        return PulseSource(dev_id, spec, self.latency_ms)

    def open_sink(self, dev_id: str, spec: StreamSpec) -> Sink:
        return PulseSink(dev_id, spec, self.latency_ms)


CONNECTORS = [PulseConnector]
