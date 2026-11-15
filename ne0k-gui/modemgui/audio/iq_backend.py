# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""``iq:`` -- complex baseband over a TCP I/Q socket (the GNU Radio medium, iq_connector.py).

Address::

    iq:HOST:PORT          cf32 (little-endian float32 I,Q), the medium's default
    iq:HOST:PORT/sc16     int16 I,Q, full scale 32768 (the PlutoSDR native)

The wire is the one gr_rf_medium.py and iq_connector.py speak: a bare stream, no header, ONE
connection per node carrying both directions -- the node writes its TX and reads what the medium
says it hears.  So the source and the sink of one address share one socket here: whichever opens
first dials, the second joins, and the socket closes when both have stopped.

Clocking.  The medium is the sample clock: it emits every node's RX at the sample rate on the
wall clock and buffers TX it is handed early (dropping the OLDEST past its --max-buffer-s).  This
connector therefore never paces itself: for every block of n samples RECEIVED it pulls exactly n
from the sink and writes them back (zeros when nothing is queued), so TX runs at the medium's rate
with a constant, block-sized lead, and our own transmissions land on the RX timebase exactly as
AudioPipe's ``tx_onset_hooks`` assume.  A sink with no source still gets that pacing (the reader
runs and discards RX).

Lead.  Lockstep alone leaves the medium's TX buffer one block deep, and the medium takes TX
BEFORE it hands out the RX of the same tick: any scheduling lateness (100-200 ms seen on a busy
host) then finds the buffer empty and the medium fills the hole with zeros -- in the middle of a
frame.  So when the sink starts, ``TX_LEAD_S`` of zeros goes out once; lockstep then holds it.
What it holds is the lead LESS the RX still in flight inside the flowgraph and the socket, and that
swings: at 0.25 s the medium's TX buffer for a node was seen down to 18 ms (and empty three times
in a 100 kB transfer) on a host where the medium ran up to 0.4 s late.  0.5 s covers that and
stays inside the medium's --max-buffer-s of 1 s.  It is pure latency on our TX.

Only for an I/Q session (``StreamSpec.iq``); the registry refuses it on an audio pipe.
"""
from __future__ import annotations

import os
import socket
import threading
import time
from typing import Callable

import numpy as np

from .base import SINK, SOURCE, AudioError, Connector, DeviceInfo, Sink, Source, StreamSpec

FORMATS = {"cf32": 8, "sc16": 4}          # bytes per complex sample
CONNECT_TIMEOUT_S = 10.0                  # the medium may still be starting
TX_LEAD_S = 0.5                           # zeros primed into the medium's TX buffer (see Lead)
DEVICES_ENV = "MODEMGUI_IQ_DEVICES"       # comma list of HOST:PORT[/FMT] to offer in pickers


def parse_dev(dev_id: str) -> tuple[str, int, str]:
    """``HOST:PORT[/FMT]`` -> (host, port, fmt).  IPv6 hosts in brackets: ``[::1]:5601``."""
    addr, _, fmt = dev_id.partition("/")
    fmt = fmt or "cf32"
    if fmt not in FORMATS:
        raise AudioError(f"iq:{dev_id}: format {fmt!r}, want one of {sorted(FORMATS)}")
    host, sep, port = addr.rpartition(":")
    if not sep or not host:
        raise AudioError(f"iq:{dev_id}: want HOST:PORT[/cf32|/sc16]")
    try:
        p = int(port)
    except ValueError:
        raise AudioError(f"iq:{dev_id}: bad port {port!r}") from None
    if not 0 < p < 65536:
        raise AudioError(f"iq:{dev_id}: port {p} out of range")
    return host.strip("[]"), p, fmt


def decode(buf: bytes | bytearray | memoryview, fmt: str) -> np.ndarray:
    if fmt == "cf32":
        return np.frombuffer(buf, "<f4").view("<c8").astype(np.complex64)
    v = np.frombuffer(buf, "<i2").astype(np.float32) * (1.0 / 32768.0)
    return (v[0::2] + 1j * v[1::2]).astype(np.complex64)


def encode(x: np.ndarray, fmt: str) -> bytes:
    x = np.asarray(x, np.complex64)
    if fmt == "cf32":
        return x.astype("<c8").tobytes()
    v = np.empty(2 * len(x), np.float32)
    v[0::2], v[1::2] = x.real, x.imag
    return np.clip(np.round(v * 32768.0), -32768, 32767).astype("<i2").tobytes()


class _Link:
    """One socket shared by an address's source and sink."""

    def __init__(self, owner: "IqConnector", dev_id: str, spec: StreamSpec) -> None:
        self.owner, self.dev_id, self.spec = owner, dev_id, spec
        self.host, self.port, self.fmt = parse_dev(dev_id)
        self.isz = FORMATS[self.fmt]
        self.block = max(1, int(spec.block))
        self.lock = threading.Lock()
        self.src: "IqSource | None" = None
        self.snk: "IqSink | None" = None
        self.sock: socket.socket | None = None
        self.thread: threading.Thread | None = None
        self.stop_ev = threading.Event()
        self.users = 0
        self.rx_samples = self.tx_samples = 0
        self.primed = False                   # TX lead sent (once per connection)

    # -- ends ---------------------------------------------------------------

    def attach(self, end) -> None:
        with self.lock:
            if isinstance(end, IqSource):
                if self.src is not None:
                    raise AudioError(f"iq:{self.dev_id}: source already open")
                self.src = end
            else:
                if self.snk is not None:
                    raise AudioError(f"iq:{self.dev_id}: sink already open")
                self.snk = end
            self.users += 1

    def detach(self, end) -> None:
        with self.lock:
            if end is self.src:
                self.src = None
            elif end is self.snk:
                self.snk = None
            else:
                return
            self.users -= 1
            last = self.users == 0
        if last:
            self.close()

    # -- socket -------------------------------------------------------------

    def ensure_running(self) -> None:
        with self.lock:
            if self.thread is not None:
                return
            self.sock = self._dial()
            self.stop_ev.clear()
            self.thread = threading.Thread(target=self._run, daemon=True, name=f"iq:{self.dev_id}")
            self.thread.start()

    def _dial(self) -> socket.socket:
        end = time.monotonic() + CONNECT_TIMEOUT_S
        last: Exception | None = None
        while time.monotonic() < end:
            try:
                s = socket.create_connection((self.host, self.port), timeout=2.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(0.5)
                return s
            except OSError as e:
                last = e
                time.sleep(0.2)
        raise AudioError(f"iq:{self.dev_id}: cannot connect to {self.host}:{self.port}: {last}")

    def close(self) -> None:
        # Socket first: the medium then sees the node leave at once (no TX gap counted as underrun
        # while the reader winds down), and a recv() in progress returns.
        self.stop_ev.set()
        s, self.sock = self.sock, None
        if s is not None:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()
        t, self.thread = self.thread, None
        if t is not None and t is not threading.current_thread():
            t.join(2.0)
        self.owner._forget(self.dev_id, self)

    def _fail(self, msg: str) -> None:
        with self.lock:
            src, snk = self.src, self.snk
        for e in (src, snk):
            if e is not None and e.error is None:
                e.error = msg
        if src is not None:
            src.finished = True

    def _run(self) -> None:
        need = self.block * self.isz
        buf = bytearray()
        sock = self.sock
        while not self.stop_ev.is_set():
            try:
                chunk = sock.recv(max(65536, need))
            except socket.timeout:
                continue
            except OSError as e:
                if not self.stop_ev.is_set():
                    self._fail(f"iq:{self.dev_id}: {e}")
                return
            if not chunk:
                if not self.stop_ev.is_set():
                    self._fail(f"iq:{self.dev_id}: medium closed the connection")
                return
            buf += chunk
            whole = (len(buf) // self.isz) * self.isz
            if whole < need:
                continue
            x = decode(memoryview(buf)[:whole], self.fmt)
            del buf[:whole]
            n = len(x)
            self.rx_samples += n
            with self.lock:
                src, snk = self.src, self.snk
            if src is not None and src.on_block is not None:
                try:
                    src.on_block(x)
                except Exception as e:          # a consumer bug must not kill the socket
                    src.error = f"iq:{self.dev_id}: RX callback: {type(e).__name__}: {e}"
            if snk is not None and snk.pull is not None:
                try:
                    if not self.primed:
                        self.primed = True
                        sock.sendall(encode(np.zeros(int(TX_LEAD_S * self.spec.rate), np.complex64), self.fmt))
                    y = snk.pull(n)
                    sock.sendall(encode(y, self.fmt))
                    self.tx_samples += n
                except OSError as e:
                    if not self.stop_ev.is_set():
                        self._fail(f"iq:{self.dev_id}: TX: {e}")
                    return
                except Exception as e:
                    snk.error = f"iq:{self.dev_id}: TX pull: {type(e).__name__}: {e}"


class IqSource(Source):
    def __init__(self, link: _Link, spec: StreamSpec) -> None:
        self.link, self.spec = link, spec
        self.on_block: Callable[[np.ndarray], None] | None = None
        self.error = None
        self.finished = False
        link.attach(self)

    def start(self, on_block: Callable[[np.ndarray], None]) -> None:
        self.on_block = on_block
        self.link.ensure_running()

    def stop(self) -> None:
        self.on_block = None
        self.link.detach(self)


class IqSink(Sink):
    def __init__(self, link: _Link, spec: StreamSpec) -> None:
        self.link, self.spec = link, spec
        self.pull: Callable[[int], np.ndarray] | None = None
        self.error = None
        link.attach(self)

    def start(self, pull: Callable[[int], np.ndarray]) -> None:
        self.pull = pull
        self.link.ensure_running()

    def stop(self) -> None:
        self.pull = None
        self.link.detach(self)


class IqConnector(Connector):
    name = "iq"
    description = "Complex baseband over a TCP I/Q socket (GNU Radio medium, iq_connector.py)"
    iq = True

    def __init__(self) -> None:
        self._links: dict[str, _Link] = {}
        self._lock = threading.Lock()

    def _link(self, dev_id: str, spec: StreamSpec) -> _Link:
        if not spec.iq:
            raise AudioError("iq: needs an I/Q session (StreamSpec.iq)")
        parse_dev(dev_id)
        with self._lock:
            l = self._links.get(dev_id)
            if l is None:
                l = self._links[dev_id] = _Link(self, dev_id, spec)
            elif l.spec.block != spec.block or l.spec.rate != spec.rate:
                raise AudioError(f"iq:{dev_id}: already open with another rate/block")
            return l

    def _forget(self, dev_id: str, link: _Link) -> None:
        with self._lock:
            if self._links.get(dev_id) is link:
                del self._links[dev_id]

    def list_devices(self) -> list[DeviceInfo]:
        devs = [d.strip() for d in os.environ.get(DEVICES_ENV, "127.0.0.1:5601").split(",") if d.strip()]
        out = []
        for i, d in enumerate(devs):
            for kind in (SOURCE, SINK):
                out.append(DeviceInfo(self.name, d, f"I/Q socket {d}", kind, default=(i == 0)))
        return out

    def open_source(self, dev_id: str, spec: StreamSpec) -> Source:
        return IqSource(self._link(dev_id, spec), spec)

    def open_sink(self, dev_id: str, spec: StreamSpec) -> Sink:
        return IqSink(self._link(dev_id, spec), spec)


CONNECTORS = [IqConnector]
