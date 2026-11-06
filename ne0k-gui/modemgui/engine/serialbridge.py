# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Virtual serial port on top of the CSS link (Linux, pty).

CSS airtime is seconds per few characters, so the port is a *buffered, fully
handshaken* half-duplex modem rather than a wire:

    host writes <= N bytes to the tty        (modem accepts only while CTS is high)
    host raises RTS                          (rising edge = "send the buffer")
    modem drops CTS                          (TX-busy; receiver is off, we are not in RX)
    ... airtime ...
    modem raises CTS, host drops RTS         (ready for the next buffer)

With ``ack`` the buffer is only done when the peer's ACK (the DATA seq echoed, as in
css_modem_node) arrives: CTS stays low through the wait, a missing ACK retransmits the
same seq up to ``retries`` times, then ``TXFAIL`` replaces ``TXDONE``.  The receiver
ACKs every unicast DATA, including a duplicate (its ACK may be what was lost) but hands
a duplicate to the host only once.

With an ``OfdmConfig`` the bridge runs over the half-duplex OFDM link instead: no CSS packets,
the link does its own BAT handshake and windowed ARQ (``--retries`` still applies).  The modem
is "not ready" until the BAT exchange finishes -- CTS stays low and ``CD 1`` (carrier detect)
marks the link up -- and the buffer is clamped to what one window carries.

A pty has no modem-control lines (TIOCMGET is ENOTTY), so the lines travel on a
unix control socket next to the tty, ``<link>.ctl``, one text line per event:

    host -> modem   RTS 1 | RTS 0 | STATUS
    modem -> host   CTS 1|0   CD 1|0   BUF <n>/<N>   RX <n>   TXDONE   TXFAIL   RETRY <k>   ACK <seq>
                    ERR <why>

``HandshakePort`` is the host-side client.  Received data is written to the tty
as it arrives; there is no handshake on that direction (the host may be slow, in
which case bytes wait in the engine, not the air).

Packet payload = 1 length byte + data, so binary data with trailing NULs survives.
"""
from __future__ import annotations

import os
import queue
import select
import socket
import threading
import time

try:
    import tty                     # pty side only; Windows has no tty module and no pty to put in raw mode
except ImportError:                # pragma: no cover
    tty = None

from .css_link import CssLink
from .events import Decoded, LinkEvent
from .modemcfg import CssConfig, OfdmConfig
from .mesh_node import MeshNode
from .ofdm_link import OfdmLink

MAX_BUF = 250                      # length fits one byte
MAX_BIG_BUF = 1 << 20              # OFDM oneway/stream: the link carries MiB per burst


def frame(data: bytes) -> bytes:
    if len(data) > MAX_BUF:
        raise ValueError(f"{len(data)} bytes exceeds MAX_BUF={MAX_BUF}")
    return bytes([len(data)]) + data


def unframe(payload: bytes) -> bytes:
    if not payload:
        return b""
    return payload[1:1 + payload[0]]


class SerialBridge:
    def __init__(self, session, cfg: CssConfig | OfdmConfig | MeshNode, link_path: str,
                 buf_size: int = 32, mode: str = "rts", idle_s: float = 2.0, log=lambda m: None,
                 ack: bool = True, retries: int = 3, ack_timeout: float | None = None,
                 dst: int | None = None) -> None:
        # MAX_BUF is the CSS frame's one-byte length.  The OFDM one-way/stream link frames its own bursts
        # (4-byte length, MiB capacity), so its serial buffer is limited only by what the link reports.
        big = isinstance(cfg, OfdmConfig) and cfg.mode in ("oneway", "stream")
        if not 1 <= buf_size <= (MAX_BIG_BUF if big else MAX_BUF):
            raise ValueError(f"buf_size must be 1..{MAX_BIG_BUF if big else MAX_BUF}")
        if mode not in ("rts", "auto", "fifo"):
            raise ValueError("mode must be 'rts', 'auto' or 'fifo'")
        if mode == "fifo" and not big:
            raise ValueError("mode 'fifo' needs an OFDM link in oneway or stream mode")
        self.fifo = mode == "fifo"
        self._chain = False                      # fifo: bytes arrived while the last burst was on the air
        self.mesh = isinstance(cfg, MeshNode)
        if self.mesh and dst is None:
            raise ValueError("a mesh bridge needs dst (the node its bytes are sent to)")
        self.dst = dst
        self.session, self.cfg = session, (cfg.mesh if self.mesh else cfg)
        self.link_path, self.ctl_path = link_path, link_path + ".ctl"
        self.buf_size, self.mode, self.idle_s, self.log = buf_size, mode, idle_s, log
        self.ofdm = isinstance(cfg, OfdmConfig)
        if self.mesh:
            self.link = cfg
            cfg.on_data = self._mesh_rx
        elif self.ofdm:
            cfg.retries = retries
            self.link = OfdmLink(session, cfg, log=self.log)
            self.link.on_data = self._rxq_put
            self.link.on_retry = self._ofdm_retry
            self.link.on_state = self._ofdm_state
        else:
            self.link = CssLink(session, cfg)
        self._rxq: "queue.Queue[bytes]" = queue.Queue()
        self.txbuf = bytearray()
        self.busy = False
        self._rts = False
        self._last_byte = 0.0
        self._last_rx: tuple[int, int] | None = None
        self._out = bytearray()                  # received data waiting for the host
        self._clients: list[socket.socket] = []
        self._inbuf: dict[socket.socket, bytes] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.tx_count = self.rx_count = self.retry_count = self.fail_count = 0
        self.ack, self.retries, self.ack_timeout = ack, retries, ack_timeout
        self.rx_filter = None                    # test hook: f(Decoded) -> False drops the packet
        self._want_ack: int | None = None
        self._ack_evt = threading.Event()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> str:
        import tty                                   # POSIX only; keep the module importable on Windows
        self.master, self._slave = os.openpty()      # keep the slave open: no EIO when the host closes
        tty.setraw(self._slave)
        os.set_blocking(self.master, False)
        name = os.ttyname(self._slave)
        for p in (self.link_path, self.ctl_path):
            with _suppress(FileNotFoundError):
                os.unlink(p)
        os.symlink(name, self.link_path)
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.ctl_path); self._srv.listen(4); self._srv.setblocking(False)
        self.link.start()
        self._thread = threading.Thread(target=self._run, name="serial-bridge", daemon=True)
        self._thread.start()
        self.log(f"ready link={self.link_path} tty={name} buf={self.buf_size} mode={self.mode}"
                 + (f" modem=ofdm role={self.link.role}" if self.ofdm else "")
                 + (f" modem=mesh node={self.cfg.node_id} dst={self.dst}" if self.mesh else ""))
        return name

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(5.0)
        self.link.stop()
        for s in self._clients + [self._srv]:
            s.close()
        for fd in (self.master, self._slave):
            os.close(fd)
        for p in (self.link_path, self.ctl_path):
            with _suppress(FileNotFoundError):
                os.unlink(p)

    # -- line state ---------------------------------------------------------

    @property
    def cts(self) -> bool:
        if self.fifo:                            # a UART with a FIFO: ready while there is room, TX or not
            return self.link.up and len(self.txbuf) < self.buf_size
        return not self.busy and (not self.ofdm or self.link.up)

    def _rxq_put(self, data: bytes) -> None:
        self._rxq.put(data)

    def _mesh_rx(self, src: int, data: bytes) -> None:
        # mesh rx_filter: f(src) -> bool (CSS's is f(Decoded) -> bool; different link, different shape).
        if self.rx_filter is not None and not self.rx_filter(src):
            self.log(f"rx {len(data)}B from {src} dropped by filter")
            return
        self._rxq_put(data)

    def _ofdm_retry(self, k: int) -> None:
        self.retry_count += 1
        self._notify(f"RETRY {k}")

    def _ofdm_state(self, st: str) -> None:
        if st == "up":
            self.buf_size = min(self.buf_size, self.link.capacity)
            self.log(f"buf={self.buf_size} window capacity={self.link.capacity}B")
        self._notify(f"CD {int(st == 'up')}")
        self._status()

    def _notify(self, line: str) -> None:
        data = (line + "\n").encode()
        for c in list(self._clients):
            try:
                c.sendall(data)
            except OSError:
                self._drop(c)

    def _status(self, c: socket.socket | None = None) -> None:
        lines = [f"CTS {int(self.cts)}", f"BUF {len(self.txbuf)}/{self.buf_size}"]
        if self.ofdm:
            lines.insert(1, f"CD {int(self.link.up)}")
        for ln in lines:
            if c is None:
                self._notify(ln)
            else:
                c.sendall((ln + "\n").encode())

    def _drop(self, c: socket.socket) -> None:
        if c in self._clients:
            self._clients.remove(c)
        self._inbuf.pop(c, None)
        c.close()

    # -- main loop ----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            rfds = [self._srv, *self._clients]
            if self.cts and len(self.txbuf) < self.buf_size:
                rfds.append(self.master)         # not reading = pty back-pressure on the host
            wfds = [self.master] if self._out else []
            r, w, _ = select.select(rfds, wfds, [], 0.1)
            for f in r:
                if f is self._srv:
                    c, _ = self._srv.accept(); c.setblocking(True); c.settimeout(2.0)
                    self._clients.append(c); self._status(c)
                elif f is self.master:
                    self._load()
                else:
                    self._control(f)
            if w:
                self._flush_out()
            self._poll_rx()
            if self.mode in ("auto", "fifo"):
                self._auto()

    def _load(self) -> None:
        try:
            data = os.read(self.master, self.buf_size - len(self.txbuf))
        except (BlockingIOError, OSError):
            return
        if data:
            self.txbuf += data
            self._last_byte = time.monotonic()
            self._notify(f"BUF {len(self.txbuf)}/{self.buf_size}")
            if self.fifo and not self.cts:
                self._notify("CTS 0")

    def _control(self, c: socket.socket) -> None:
        try:
            d = c.recv(256)
        except OSError:
            d = b""
        if not d:
            self._drop(c)
            return
        buf = self._inbuf.get(c, b"") + d
        *lines, self._inbuf[c] = buf.split(b"\n")
        for ln in lines:
            cmd = ln.decode("ascii", "replace").split()
            if cmd[:1] == ["RTS"] and len(cmd) == 2:
                rising = cmd[1] == "1" and not self._rts
                self._rts = cmd[1] == "1"
                if rising:
                    self._try_send(c)
            elif cmd[:1] == ["STATUS"]:
                self._status(c)

    def _try_send(self, c: socket.socket | None) -> None:
        if self.fifo:                            # RTS means nothing on a FIFO port
            return
        if self.busy:
            if c:
                c.sendall(b"ERR busy\n")
        elif not self.txbuf:
            if c:
                c.sendall(b"ERR empty\n")
        else:
            self._transmit()

    def _auto(self) -> None:
        if self.busy or not self.txbuf:
            return
        if self._chain or len(self.txbuf) >= self.buf_size or time.monotonic() - self._last_byte >= self.idle_s:
            self._transmit()

    def _transmit(self) -> None:
        data, self.txbuf = bytes(self.txbuf), bytearray()
        self._chain = False
        self.busy = True
        self._notify(f"BUF 0/{self.buf_size}")
        if not self.fifo:
            self._notify("CTS 0")
        self.tx_count += len(data)
        threading.Thread(target=self._tx_job, args=(data,), name="serial-tx", daemon=True).start()

    def _tx_job(self, data: bytes) -> None:
        if self.mesh:
            return self._tx_job_mesh(data)
        if self.ofdm:
            return self._tx_job_ofdm(data)
        me, peer = self.cfg.node_id, self.cfg.peer_id
        seq, ok = None, not self.ack
        for attempt in range(self.retries + 1 if self.ack else 1):
            if attempt:
                self.retry_count += 1
                self._notify(f"RETRY {attempt}")
                self.session._emit(LinkEvent("retry", me, peer, time.monotonic(),
                                             time.monotonic(), f"attempt {attempt}"))
            air = self.link.send_bytes(frame(data), seq)
            seq = self.link.last_seq
            self.log(f"tx {len(data)}B seq={seq} airtime={air:.1f}s" + (f" retry={attempt}" if attempt else ""))
            pipe = self.session.pipe
            if pipe:
                pipe.wait_tx(air + 10)
            time.sleep(0.3)                      # sink tail; RX stays muted until here
            if not self.ack:
                break
            if self.ack_timeout is None:         # ack airtime + both ends' hang/decode + margin
                self.ack_timeout = self.link.ack_airtime() + 2 * self.cfg.hang_s + 10
            self._ack_evt.clear(); self._want_ack = seq
            if self._ack_evt.wait(self.ack_timeout):
                ok = True
                break
        self._want_ack = None
        if self.ack and not ok:
            self.fail_count += 1
            now = time.monotonic()
            self.session._emit(LinkEvent("fail", me, peer, now, now, f"no ACK after {self.retries} retries"))
        self.busy = False
        self._notify("TXDONE" if ok else "TXFAIL"); self._notify("CTS 1")
        self.log("tx done" if ok else "tx FAILED (no ack)")

    def _tx_job_mesh(self, data: bytes) -> None:
        try:
            dst = self.dst() if callable(self.dst) else self.dst      # mesh: the node to talk to is learned
            if dst is None:
                raise RuntimeError("no neighbour heard yet")
            ok = self.link.send(dst, data)
        except Exception as e:
            # An OFDM_ONLY VIOLATION message is already self-describing and alarming on its own
            # (mesh_node.py logged and recorded it) -- don't bury it inside the generic wrapper.
            self.log(str(e) if "OFDM_ONLY VIOLATION" in str(e) else f"tx refused: {e}")
            ok = False
        if not ok:
            self.fail_count += 1
        self.busy = False
        self._notify("TXDONE" if ok else "TXFAIL"); self._notify("CTS 1")
        self.log("tx done" if ok else "tx FAILED")

    def _tx_job_ofdm(self, data: bytes) -> None:
        try:
            ok = self.link.send(data)
        except RuntimeError as e:                    # oneway/stream: this end only listens
            self.log(f"tx refused: {e}")
            ok = False
        if not ok:
            self.fail_count += 1
        self._chain = self.fifo and bool(self.txbuf)
        self.busy = False
        self._notify("TXDONE" if ok else "TXFAIL"); self._notify(f"CTS {int(self.cts)}")
        self.log("tx done" if ok else "tx FAILED (no ack)")

    def _send_ack(self, dst: int, seq: int) -> None:
        def go() -> None:
            self.link.send_ack(dst, seq)
            self.log(f"ack -> {dst} seq={seq}")
        threading.Thread(target=go, name="serial-ack", daemon=True).start()

    # -- receive ------------------------------------------------------------

    def _poll_rx(self) -> None:
        me = self.cfg.node_id
        if self.mesh:                            # MeshNode owns the session's event stream
            self._poll_rx_ofdm(drain=False)
            return
        if self.ofdm:
            self._poll_rx_ofdm()
            return
        for ev in self.session.drain():
            if isinstance(ev, LinkEvent) and ev.kind == "err":
                self.log(f"rx error: {ev.detail}")
            if not isinstance(ev, Decoded) or not ev.ok or ev.dst not in (me, 0xF):
                continue
            if self.rx_filter and not self.rx_filter(ev):
                self.log(f"rx {ev.type_name} seq={ev.seq} dropped by filter")
                continue
            if ev.type_name == "ACK":
                if ev.dst == me and ev.seq == self._want_ack:
                    now = time.monotonic()
                    self.session._emit(LinkEvent("ack", me, ev.src, now, now, f"seq={ev.seq}"))
                    self._notify(f"ACK {ev.seq}")
                    self._ack_evt.set()
                continue
            if ev.type_name != "DATA":
                continue
            if self.ack and ev.dst == me:        # unicast only; also re-ACK a duplicate
                self._send_ack(ev.src, ev.seq)
            key = (ev.src, ev.seq)
            if key == self._last_rx:
                self.log(f"rx duplicate seq={ev.seq} dropped")
                continue
            self._last_rx = key
            data = unframe(ev.payload)
            self._out += data
            self.rx_count += len(data)
            self.log(f"rx {len(data)}B snr={ev.snr_db:.1f}dB")
            self._notify(f"RX {len(data)}")
        if self._out:
            self._flush_out()

    def _poll_rx_ofdm(self, drain: bool = True) -> None:
        for ev in self.session.drain() if drain else ():
            if isinstance(ev, LinkEvent) and ev.kind == "err":
                self.log(f"rx error: {ev.detail}")
        while True:
            try:
                data = self._rxq.get_nowait()
            except queue.Empty:
                break
            self._out += data
            self.rx_count += len(data)
            self.log(f"rx {len(data)}B")
            self._notify(f"RX {len(data)}")
        if self._out:
            self._flush_out()

    def _flush_out(self) -> None:
        try:
            n = os.write(self.master, bytes(self._out))
            del self._out[:n]
        except (BlockingIOError, OSError):
            pass


class _suppress:
    def __init__(self, *exc): self.exc = exc
    def __enter__(self): return self
    def __exit__(self, t, v, tb): return t is not None and issubclass(t, self.exc)


class HandshakePort:
    """Host side: the tty for data, the control socket for RTS/CTS.  ``send`` returns True
    only when the whole buffer was delivered (ACKed when the bridge runs with ack)."""

    def __init__(self, link_path: str) -> None:
        self.fd = os.open(link_path, os.O_RDWR | os.O_NOCTTY)
        tty.setraw(self.fd)
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.connect(link_path + ".ctl")
        self.cts, self.cd, self.buf = True, None, (0, 0)
        self._buf_hi = 0                 # highest BUF level since the last write: a full buffer goes out at once,
                                         # and "BUF n/N" + "BUF 0/N" often arrive in one recv
        self.rx_events: "queue.Queue[int]" = queue.Queue()
        self.errors: list[str] = []
        self._cv = threading.Condition()
        self._txdone = self._txfail = 0
        self.retries = 0
        threading.Thread(target=self._reader, daemon=True, name="hs-port").start()

    def _reader(self) -> None:
        buf = b""
        while True:
            try:
                d = self.sock.recv(256)
            except OSError:
                return
            if not d:
                return
            buf += d
            *lines, buf = buf.split(b"\n")
            with self._cv:
                for ln in lines:
                    w = ln.decode().split()
                    if w[0] == "CTS":
                        self.cts = w[1] == "1"
                    elif w[0] == "CD":
                        self.cd = w[1] == "1"
                    elif w[0] == "BUF":
                        n, N = w[1].split("/"); self.buf = (int(n), int(N))
                        self._buf_hi = max(self._buf_hi, int(n))
                    elif w[0] == "RX":
                        self.rx_events.put(int(w[1]))
                    elif w[0] == "TXDONE":
                        self._txdone += 1
                    elif w[0] == "TXFAIL":
                        self._txfail += 1
                    elif w[0] == "RETRY":
                        self.retries += 1
                    elif w[0] == "ERR":
                        self.errors.append(" ".join(w[1:]))
                self._cv.notify_all()

    def _wait(self, pred, timeout: float) -> bool:
        with self._cv:
            return self._cv.wait_for(pred, timeout)

    def wait_cts(self, value: bool, timeout: float) -> bool:
        return self._wait(lambda: self.cts == value, timeout)

    def wait_cd(self, value: bool, timeout: float) -> bool:
        return self._wait(lambda: self.cd == value, timeout)

    def wait_buffered(self, n: int, timeout: float) -> bool:
        """The bridge took ``n`` bytes since the last ``write`` (even if it has already sent them on)."""
        return self._wait(lambda: self._buf_hi >= n, timeout)

    def write(self, data: bytes) -> None:
        with self._cv:
            self._buf_hi = self.buf[0]
        os.write(self.fd, data)

    def read(self, n: int = 4096, timeout: float = 1.0) -> bytes:
        r, _, _ = select.select([self.fd], [], [], timeout)
        return os.read(self.fd, n) if r else b""

    def set_rts(self, on: bool) -> None:
        self.sock.sendall(f"RTS {int(on)}\n".encode())

    def send(self, data: bytes, timeout: float = 120.0) -> bool:
        """Full handshake for one buffer: wait CTS, load, RTS, wait busy->ready."""
        if not self.wait_cts(True, timeout):
            return False
        with self._cv:
            before, failed = self._txdone, self._txfail     # before the write: a full buffer is sent at once
        self.write(data)
        if not self.wait_buffered(len(data), 5.0):
            return False
        self.set_rts(True)
        self._wait(lambda: self._txdone > before or self._txfail > failed, timeout)
        self.set_rts(False)
        return self._txdone > before

    def close(self) -> None:
        self.sock.close(); os.close(self.fd)
