# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""CSS broadcast port: a one-way text stream with a time code, on a virtual serial port (Linux, pty).

The data-stream sibling of ``serialbridge`` for one sender and any number of late-joining listeners.  No ACKs, no
handshake, no turns: the sender keys up and keeps airing packets, every listener locks its PLL on whatever packet
comes next and keeps going.

send     what the host writes to the tty is queued and aired as broadcast DATA packets (dst 0xF), up to
         ``text_bytes`` per packet, payload = 1 length byte + text.  Every ``time_every``-th packet -- and every packet while the queue is empty --
         is a TIME packet instead: u32 UTC seconds at the instant that packet's last chirp ends (the sender
         schedules it so that lands on the whole second).  Packets are separated by ``gap_s`` of silence so the
         listeners' streaming decoders see a burst end and re-arm for the next preamble.
receive  DATA from ``sender_id`` is written to the tty as it decodes.  A TIME packet sets the clock: offset =
         its UTC stamp - our wall clock when its last frame decoded.  The 4-bit packet seq counts lost packets.
         CSS puts no packet length on the air (a packet ends at the next chunk 0 or when the channel goes quiet),
         so a packet whose tail frames were lost still decodes, short.  Every packet's frame count is checked
         against its length (DATA: the length byte, TIME: 4 bytes); a short one is counted as damaged and dropped,
         never handed to the host or the clock.

The tty carries only the text.  Everything else is on the control socket next to it, ``<link>.ctl``, one line each:

    host -> modem   TIME | STATUS
    modem -> host   TIME <utc.sss> LOCK <age_s> N <time packets> OFFSET <s>   or   TIME NOLOCK
                    PLL 0|1   BCAST role=.. pkts=.. data=.. text=..B lost=.. damaged=.. time=..   (STATUS, plus CTS 1)
                    PLL 1 (first lock)   TIMELOCK <utc> OFFSET <s>   RX <n>   (unsolicited)

The receive path is fed with ``handle(ev)`` (the GUI forwards its session events); headless, ``own_events=True``
drains the session on a thread of its own.  ``python -m modemgui.engine.css_bcast`` runs one end headless.
"""
from __future__ import annotations

import math
import os
import select
import socket
import sys
import threading
import time

from .css_link import CssLink, _imports, encode_type
from .events import Decoded, LinkEvent
from .modemcfg import CssBcastConfig, CssConfig

BROADCAST = 0xF
MAX_QUEUE = 1 << 16                # sender: bytes waiting for the air; beyond this the tty back-pressures


class CssBcastPort:
    def __init__(self, session, css: CssConfig, bc: CssBcastConfig, log=lambda m: None,
                 own_events: bool = False) -> None:
        self.session, self.bc, self.log = session, bc, log
        self.send_role = bc.role == "send"
        import dataclasses
        self.css = dataclasses.replace(css, peer_id=BROADCAST)
        self.link = CssLink(session, self.css, log=log)
        self.link_path = os.path.expanduser(bc.link_path)
        self.ctl_path = self.link_path + ".ctl"
        self.own_events = own_events
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._txq = bytearray()
        self._out = bytearray()
        self._clients: list[socket.socket] = []
        self._inbuf: dict[socket.socket, bytes] = {}
        # counters: sender = aired, listener = delivered
        self.pkts = self.data_pkts = self.time_pkts = 0
        self.tx_count = self.rx_count = 0
        self.lost = 0
        self.damaged = 0                      # listener: decoded packets whose frame count is short of their length
        self._last_seq: int | None = None
        self._seq = 0
        self.pll_ever = False
        self.offset: float | None = None      # broadcast clock - our clock, from the latest TIME packet
        self.offset_mono: float | None = None
        self.offsets: list[float] = []

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> str:
        import tty
        self.master, self._slave = os.openpty()
        tty.setraw(self._slave)
        os.set_blocking(self.master, False)
        name = os.ttyname(self._slave)
        for p in (self.link_path, self.ctl_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        os.symlink(name, self.link_path)
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.ctl_path); self._srv.listen(8); self._srv.setblocking(False)
        self.link.start()
        self._spawn(self._run, "bcast-port")
        if self.send_role:
            self._spawn(self._tx_loop, "bcast-tx")
        if self.own_events:
            self._spawn(self._pump, "bcast-events")
        c, b = self.css, self.bc
        what = (f"tty -> CSS broadcast from node {c.node_id}" if self.send_role
                else f"CSS broadcast of node {b.sender_id} -> tty")
        self.log(f"bcast port ready: {what} link={self.link_path} tty={name} symbol={c.symbol_dur}s "
                 f"chunk={c.chunk_size}B text={b.text_bytes}B/pkt time_every={b.time_every}")
        return what

    def _spawn(self, fn, name: str) -> None:
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(5.0)
        self.link.stop()
        if self.send_role:
            self.session.stop_tx()
        for s in self._clients + [self._srv]:
            s.close()
        for fd in (self.master, self._slave):
            os.close(fd)
        for p in (self.link_path, self.ctl_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        if self.send_role:
            self.log(f"bcast port closed: sent pkts={self.pkts} data={self.data_pkts} time={self.time_pkts} "
                     f"text={self.tx_count}B queued={len(self._txq)}B")
        else:
            self.log(f"bcast port closed: rx pkts={self.pkts} data={self.data_pkts} time={self.time_pkts} "
                     f"text={self.rx_count}B lost={self.lost}")

    # -- tty + control socket ----------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            rfds = [self._srv, *self._clients]
            if self.send_role and len(self._txq) < MAX_QUEUE:
                rfds.append(self.master)
            with self._lock:
                wfds = [self.master] if self._out else []
            r, w, _ = select.select(rfds, wfds, [], 0.1)
            for f in r:
                if f is self._srv:
                    c, _ = self._srv.accept(); c.setblocking(True); c.settimeout(2.0)
                    self._clients.append(c)
                elif f is self.master:
                    try:
                        d = os.read(self.master, MAX_QUEUE - len(self._txq))
                    except (BlockingIOError, OSError):
                        d = b""
                    if d:
                        with self._lock:
                            self._txq += d
                else:
                    self._control(f)
            if w or self._out:
                self._flush_out()

    def _flush_out(self) -> None:
        with self._lock:
            if not self._out:
                return
            try:
                n = os.write(self.master, bytes(self._out))
                del self._out[:n]
            except (BlockingIOError, OSError):
                pass

    def _notify(self, line: str) -> None:
        data = (line + "\n").encode()
        for c in list(self._clients):
            try:
                c.sendall(data)
            except OSError:
                self._drop(c)

    def _drop(self, c: socket.socket) -> None:
        if c in self._clients:
            self._clients.remove(c)
        self._inbuf.pop(c, None)
        c.close()

    def _control(self, c: socket.socket) -> None:
        try:
            d = c.recv(256)
        except OSError:
            d = b""
        if not d:
            self._drop(c)
            return
        *lines, self._inbuf[c] = (self._inbuf.get(c, b"") + d).split(b"\n")
        for ln in lines:
            cmd = ln.decode("ascii", "replace").strip().upper()
            try:
                if cmd == "TIME":
                    c.sendall((self.time_line() + "\n").encode())
                elif cmd == "STATUS":
                    c.sendall(("\n".join(self.status_lines()) + "\n").encode())
                elif cmd:
                    c.sendall(f"ERR unknown {cmd}\n".encode())
            except OSError:
                self._drop(c)
                return

    # -- state ---------------------------------------------------------------

    @property
    def pll_locked(self) -> bool:
        st = self.link.stream
        return bool(st is not None and st._synced)

    def now(self) -> float | None:
        """The broadcast clock (UTC seconds) as this node has it, or None before the first TIME packet.
        The sender's own clock is the broadcast clock."""
        if self.send_role:
            return time.time()
        return None if self.offset is None else time.time() + self.offset

    def time_line(self) -> str:
        t = self.now()
        if t is None:
            return "TIME NOLOCK"
        age = 0.0 if self.send_role else time.monotonic() - self.offset_mono
        return f"TIME {t:.3f} LOCK {age:.1f} N {self.time_pkts} OFFSET {0.0 if self.send_role else self.offset:+.3f}"

    def status_lines(self) -> list[str]:
        role = "send" if self.send_role else "receive"
        text = self.tx_count if self.send_role else self.rx_count
        return ["CTS 1", f"PLL {int(self.pll_locked)}",
                f"BCAST role={role} pkts={self.pkts} data={self.data_pkts} text={text}B lost={self.lost} "
                f"damaged={self.damaged} time={self.time_pkts}"]

    # -- send ------------------------------------------------------------------

    def _tx_loop(self) -> None:
        _, cm, _ = _imports()
        b, c = self.bc, self.css
        k = 0
        while not self._stop.is_set():
            with self._lock:
                have = bool(self._txq)
            if k % b.time_every == 0 or not have or b.time_every == 1:
                self._send_time(cm)
            else:
                with self._lock:
                    data, self._txq[:] = bytes(self._txq[:b.text_bytes]), self._txq[b.text_bytes:]
                seq = self._next_seq()
                air = self._air(encode_type(c, BROADCAST, cm.TYPE_DATA, seq, bytes([len(data)]) + data,
                                            tag="bcast_data"))
                self.data_pkts += 1
                self.tx_count += len(data)
                self.log(f"bcast tx DATA seq={seq} {len(data)}B {data!r} airtime={air:.1f}s "
                         f"(text sent={self.tx_count}B queued={len(self._txq)}B)")
            k += 1
            self.pkts += 1
            self._stop.wait(b.gap_s)

    def _next_seq(self) -> int:
        s, self._seq = self._seq, (self._seq + 1) % 16
        return s

    def _send_time(self, cm) -> None:
        """A TIME packet whose last chirp ends on a whole UTC second (minus tx_latency_s).  The encoder's audio
        carries silence after the last chirp (~0.9 s at 0.15 s symbols), so the stamp is placed at the last
        audible sample, which is where a listener's decoder finishes the frame -- not at the end of the buffer."""
        import numpy as np
        c, b = self.css, self.bc
        seq = self._next_seq()
        probe = encode_type(c, BROADCAST, cm.TYPE_TIME, seq, cm.pack_time_payload(0), tag="bcast_time")
        air = len(probe) / c.fs
        loud = np.flatnonzero(np.abs(probe) > 1e-3)
        chirp_end = (int(loud[-1]) + 1 if len(loud) else len(probe)) / c.fs     # seconds into the audio
        end = math.ceil(time.time() + chirp_end + 0.5)           # 0.5 s for the encode below
        pcm = encode_type(c, BROADCAST, cm.TYPE_TIME, seq, cm.pack_time_payload(end), tag="bcast_time")
        start = end - chirp_end - b.tx_latency_s
        while not self._stop.is_set() and time.time() < start:
            time.sleep(min(0.01, max(0.0, start - time.time())))
        late = time.time() - start
        self._air(pcm, ident=False)
        self.time_pkts += 1
        self.log(f"bcast tx TIME seq={seq} utc={end} airtime={air:.1f}s last chirp at {chirp_end:.2f}s "
                 f"keyed {late * 1e3:+.0f} ms vs schedule")

    def _air(self, pcm, ident: bool = True) -> float:
        air = len(pcm) / self.css.fs
        now = time.monotonic()
        self.session._emit(LinkEvent("tx", self.css.node_id, BROADCAST, now, now + air, "bcast"))
        self.session.play(pcm, ident=ident)
        pipe = self.session.pipe
        if pipe:
            pipe.wait_tx(air + 10)
        return air

    # -- receive ---------------------------------------------------------------

    def _pump(self) -> None:
        while not self._stop.is_set():
            for ev in self.session.drain():
                self.handle(ev)
            time.sleep(0.05)

    def handle(self, ev) -> None:
        """One session event (the GUI forwards them all; headless, ``_pump`` does)."""
        if self.send_role:
            return
        if isinstance(ev, LinkEvent) and ev.kind in ("lock", "sync") and not self.pll_ever and self.pll_locked:
            self.pll_ever = True
            self.log("bcast pll lock")
            self._notify("PLL 1")
            return
        if not isinstance(ev, Decoded) or not ev.ok or ev.dst != BROADCAST or ev.src != self.bc.sender_id:
            return
        if not self.pll_ever:
            self.pll_ever = True
            self.log("bcast pll lock")
            self._notify("PLL 1")
        if self._last_seq is not None:
            gap = (ev.seq - self._last_seq - 1) % 16
            if gap:
                self.lost += gap
                self.log(f"bcast gap: {gap} packet(s) lost before seq={ev.seq} (total lost={self.lost})")
        else:
            self.log(f"bcast join at seq={ev.seq}")
        self._last_seq = ev.seq
        self.pkts += 1
        if ev.type_name == "TIME":
            _, cm, _ = _imports()
            if not self._whole(ev, 4):
                return
            # the streaming RX strips trailing NULs off a packet (chunk padding): a second ending in 0x00 arrives
            # short, which the frame count above already told apart from a lost frame
            utc = cm.unpack_time_payload(ev.payload.ljust(4, b"\x00"))
            t_local = ev.t_last if ev.t_last is not None else time.time()
            off = utc - t_local
            self.offsets.append(off)
            self.offset, self.offset_mono = off, time.monotonic() - (time.time() - t_local)
            self.time_pkts += 1
            spread = max(self.offsets) - min(self.offsets)
            self.log(f"bcast time lock utc={utc} offset={off:+.3f}s n={self.time_pkts} spread={spread:.3f}s "
                     f"snr={ev.snr_db:.1f}dB")
            self._notify(f"TIMELOCK {utc} OFFSET {off:+.3f}")
        elif ev.type_name == "DATA":
            n = ev.payload[0] if ev.payload else -1
            if n < 1 or not self._whole(ev, 1 + n):
                return
            data = ev.payload[1:1 + n]
            with self._lock:
                self._out += data
            self.data_pkts += 1
            self.rx_count += len(data)
            self.log(f"bcast rx DATA seq={ev.seq} {len(data)}B {data!r} snr={ev.snr_db:.1f}dB "
                     f"(text={self.rx_count}B lost={self.lost})")
            self._notify(f"RX {len(data)}")


    def _whole(self, ev, payload_len: int) -> bool:
        """The packet arrived with every frame its length needs (2-byte header + payload, chunk_size per frame)."""
        need = -(-(2 + payload_len) // self.css.chunk_size)
        if ev.frames == need:
            return True
        self.damaged += 1
        self.log(f"bcast damaged {ev.type_name} seq={ev.seq}: {ev.frames}/{need} frames (tail lost), dropped "
                 f"(total damaged={self.damaged})")
        return False


class BcastClient:
    """Host side of a listener's control socket: ``time()`` / ``status()`` round trips; unsolicited lines
    (PLL, TIMELOCK, RX) are kept in ``events``."""

    def __init__(self, link_path: str) -> None:
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.connect(os.path.expanduser(link_path) + ".ctl")
        self.sock.settimeout(5.0)
        self._buf = b""
        self.events: list[str] = []

    def _lines(self, until) -> list[str]:
        """Reply lines up to and including the first one ``until`` accepts; unsolicited lines go to events."""
        out = []
        while True:
            while b"\n" in self._buf:
                ln, self._buf = self._buf.split(b"\n", 1)
                s = ln.decode("ascii", "replace")
                if s.split()[:1] in (["TIMELOCK"], ["RX"]) or (s.startswith("PLL") and not out):
                    self.events.append(s)
                    continue
                out.append(s)
                if until(s):
                    return out
            d = self.sock.recv(1024)
            if not d:
                raise ConnectionError("control socket closed")
            self._buf += d

    def time(self) -> str:
        self.sock.sendall(b"TIME\n")
        return self._lines(lambda s: s.startswith("TIME"))[-1]

    def status(self) -> list[str]:
        self.sock.sendall(b"STATUS\n")
        return self._lines(lambda s: s.startswith("BCAST"))

    def close(self) -> None:
        self.sock.close()


def main(argv=None) -> int:
    """Headless end: python -m modemgui.engine.css_bcast --settings S.json --rx DEV --tx DEV [--seconds N]
    (settings: the GUI's settings.json; its css and cssbcast sections are used)."""
    import argparse
    import json
    from .modemcfg import HamConfig
    from .session import Session
    from . import css_link
    ap = argparse.ArgumentParser(prog="css_bcast")
    ap.add_argument("--settings", required=True)
    ap.add_argument("--rx", required=True)
    ap.add_argument("--tx", required=True)
    ap.add_argument("--seconds", type=float, default=0)
    a = ap.parse_args(argv)
    st = json.load(open(a.settings))
    css = CssConfig.from_dict(st.get("css", {}))
    bc = CssBcastConfig.from_dict(st.get("cssbcast", {}))
    s = Session()
    s.ident.cfg = HamConfig(enabled=False)
    s.open(a.rx, a.tx)
    say = lambda m: print("bcast:", m, flush=True)    # noqa: E731
    port = CssBcastPort(s, css, bc, log=say, own_events=True)
    port.start()
    say(f"css dsp: {css_link.BACKEND}")
    end = time.monotonic() + a.seconds if a.seconds else None
    try:
        while end is None or time.monotonic() < end:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    port.stop(); s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
