# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""TCP control and debug port: drive the whole modem from another host, one text line per command.

Off by default; ``--control-port PORT`` turns it on, bound to 127.0.0.1 unless
``--control-bind`` names another address.  **No authentication**: anyone who can
reach the port controls the modem, so open it beyond localhost only on a network you trust.

Protocol: ASCII lines ending in ``\\n``; the command word is case-insensitive.
Every command is answered by zero or more data lines and then exactly one ``OK [detail]`` or ``ERR <why>``.
On connect the server sends ``OK ne0k-control <protocol> <version>``.

Virtual serial port -- the same handler the pty's ``.ctl`` socket uses (``SerialBridge._command``), so the two
stay in step: the same commands give the same lines, plus the closing OK/ERR::

    RTS 0|1           the RTS line; rising edge sends the buffer
    STATUS            CTS n / [CD n] / BUF n/N
    WRITE <text>      bytes into the TX buffer as a tty write: only while CTS, only up to the free room
    WRITEHEX <hex>    the same, binary.  Both answer ``OK <accepted>/<offered>``
    SEND <text>       load the whole text and trigger it; ``SENDWAIT`` waits for TXDONE
    READ [n]          received bytes since the last READ: ``DATA <hex>``; ``READTEXT [n]`` gives ``TEXT <escaped>``
    EVENTS 0|1        stream the .ctl socket's event lines and
                      ``PORT 1 <path>`` / ``PORT 0`` when the port opens or closes

Configuration -- the ``modemcfg`` dataclasses::

    SECTIONS                       section names
    LIST [section]                 section.key=value<TAB>type=..<TAB>apply=live|restart[<TAB>lo=..][<TAB>hi=..][<TAB>choices=a|b]
    GET section.key                section.key=value
    SET section.key value          OK applied | OK pending-restart | OK next-start
    DESCRIBE section.key           label / group / tip / default

``apply=live`` fields take effect at once.  ``apply=restart`` fields are read when the modem (re)starts: set while
the modem runs, they answer ``OK pending-restart`` and the running link keeps its old copy until ``RESTART``; set
while it is off, ``OK next-start``.  SET does not write the settings to disk; ``SAVE`` does.

Debug: ``INFO`` (key=value status dump), ``LOG [n]``, ``HELP``, ``QUIT``.
The host adds its own actions.

Several clients may connect at once (``max_clients``, default 4); one more is refused with ``ERR busy ...``.
"""
from __future__ import annotations

import dataclasses
import select
import socket
import threading
import time

PROTOCOL = 1
RX_KEEP = 1 << 20                  # bytes of received data held for READ


class ControlError(Exception):
    """A refused command; the message becomes ``ERR <message>``."""


def direct(fn):
    """Mark a host action to run on the control thread instead of through ``ControlHost.run``: for actions that
    wait on subprocesses or the OS and hop to the modem's thread themselves only for the state they need."""
    def wrapper(a):
        return fn(a)
    wrapper.direct = True
    return wrapper


# -- values -------------------------------------------------------------------

def fmt_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def parse_value(f: dataclasses.Field, raw: str):
    """Text from the wire -> the field's type, checked against its F() choices / lo / hi."""
    cur = f.default
    md = f.metadata or {}
    try:
        if isinstance(cur, bool):
            low = raw.strip().lower()
            if low in ("1", "true", "yes", "on"):
                v = True
            elif low in ("0", "false", "no", "off"):
                v = False
            else:
                raise ValueError
        elif isinstance(cur, int):
            v = int(raw.strip(), 0)
        elif isinstance(cur, float):
            v = float(raw.strip())
        else:
            v = raw
    except ValueError:
        raise ControlError(f"bad value {raw!r} for {type(cur).__name__}") from None
    if md.get("choices") and v not in md["choices"]:
        raise ControlError(f"{v!r} not one of {'|'.join(map(str, md['choices']))}")
    if md.get("lo") is not None and v < md["lo"]:
        raise ControlError(f"{v} below minimum {md['lo']}")
    if md.get("hi") is not None and v > md["hi"]:
        raise ControlError(f"{v} above maximum {md['hi']}")
    return v


# -- hosts --------------------------------------------------------------------

class ControlHost:
    """What the server drives.  The GUI (ui/control_host.py) and the ``serial`` CLI subclass this."""

    name = "modem"
    live_sections: frozenset = frozenset()       # sections whose fields apply at once

    def run(self, fn):
        """Run ``fn()`` on the thread that owns the modem."""
        return fn()

    def serial(self):
        """The running SerialBridge, or None."""
        return None

    def sections(self) -> dict:
        """name -> dataclass instance."""
        return {}

    def set_param(self, section: str, key: str, value) -> str:
        """Store ``value``; answer ``applied`` / ``pending-restart`` / ``next-start``."""
        raise ControlError("settings are read-only here")

    def apply_class(self, section: str) -> str:
        return "live" if section in self.live_sections else "restart"

    def info(self) -> list[tuple[str, object]]:
        return []

    def info_offthread(self) -> list[tuple[str, object]]:
        """More INFO rows, called on the control thread rather than through ``run``: for slow reads (the host OS
        mixer shells out to pactl / amixer) that must not stall the modem's thread."""
        return []

    def actions(self) -> dict:
        """COMMAND -> (help text, fn(args: str) -> str | list[str] | None).  fn runs via ``run``; a str is the
        OK detail, a list the data lines before the OK; ControlError refuses."""
        return {}

    def log_tail(self, n: int) -> list[str] | None:
        return None


class EngineHost(ControlHost):
    """``modemgui serial``: the port and the link config it was started with.  The link is built once from the
    command line, so the config is readable here but changed only by restarting the process."""

    name = "serial-cli"

    def __init__(self, bridge, cfgs: dict) -> None:
        self.bridge, self.cfgs = bridge, cfgs

    def serial(self):
        return self.bridge

    def sections(self) -> dict:
        return dict(self.cfgs)

    def set_param(self, section: str, key: str, value) -> str:
        raise ControlError("the serial CLI builds its link once from the command line: restart it with new "
                           "options (or use 'modemgui gui --control-port' for live settings)")

    def info(self):
        br = self.bridge
        return [("link", type(br.link).__name__)]


# -- server -------------------------------------------------------------------

class _Client:
    def __init__(self, sock: socket.socket, addr) -> None:
        self.sock, self.addr = sock, addr
        self.events = False
        self.lock = threading.Lock()
        self.alive = True

    def send(self, line: str) -> None:
        if not self.alive:
            return
        try:
            with self.lock:
                self.sock.sendall((line + "\n").encode("utf-8", "replace"))
        except OSError:
            self.alive = False


class ControlServer:
    def __init__(self, host: ControlHost, port: int = 0, bind: str = "127.0.0.1",
                 max_clients: int = 4, log=lambda m: None, version: str = "") -> None:
        self.host, self.bind, self.port = host, bind, port
        self.max_clients, self.log = max_clients, log
        if not version:
            from .. import __version__ as version
        self.version = version
        self._clients: list[_Client] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._bridge = None
        self._rx = bytearray()
        self._rx_lock = threading.Lock()
        self.rx_dropped = 0
        self._txwait: list = []                  # SENDWAIT: (event, box) waiting for TXDONE / TXFAIL

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> int:
        fam = socket.AF_INET6 if ":" in self.bind else socket.AF_INET
        self._srv = socket.socket(fam, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.bind, self.port))
        self._srv.listen(8)
        self.port = self._srv.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop, name="control-accept", daemon=True)
        self._thread.start()
        self.log(f"control port listening on {self.bind}:{self.port} (no authentication"
                 + ("; LOCALHOST ONLY" if self.bind in ("127.0.0.1", "::1", "localhost") else "; OPEN TO THE NETWORK")
                 + ")")
        return self.port

    def stop(self) -> None:
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass
        if self._thread:
            self._thread.join(2.0)
        with self._lock:
            clients, self._clients = self._clients, []
        for c in clients:
            c.alive = False
            try:
                c.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            c.sock.close()
        self._attach(None)

    # -- serial port tracking ---------------------------------------------------

    def _attach(self, br) -> None:
        old, self._bridge = self._bridge, br
        if old is not None:
            old.remove_listener(self._on_line); old.remove_rx_tap(self._on_rx)
        if br is not None:
            br.add_listener(self._on_line); br.add_rx_tap(self._on_rx)
            self._broadcast(f"PORT 1 {br.link_path}")
        elif old is not None:
            self._broadcast("PORT 0")

    def _sync_bridge(self):
        try:
            br = self.host.serial()
        except Exception:
            br = None
        if br is not self._bridge:
            self._attach(br)
        return br

    def _on_line(self, line: str) -> None:
        self._broadcast(line)
        if line in ("TXDONE", "TXFAIL"):
            with self._lock:
                waiting, self._txwait = self._txwait, []
            for evt, box in waiting:
                box.append(line); evt.set()

    def _on_rx(self, data: bytes) -> None:
        with self._rx_lock:
            self._rx += data
            if len(self._rx) > RX_KEEP:
                drop = len(self._rx) - RX_KEEP
                del self._rx[:drop]
                self.rx_dropped += drop

    def _broadcast(self, line: str) -> None:
        with self._lock:
            targets = [c for c in self._clients if c.events]
        for c in targets:
            c.send(line)

    # -- connections ------------------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            self._sync_bridge()
            try:
                r, _, _ = select.select([self._srv], [], [], 0.25)
            except (OSError, ValueError):
                return
            if not r:
                continue
            try:
                sock, addr = self._srv.accept()
            except OSError:
                continue
            sock.settimeout(5.0)
            with self._lock:
                self._clients = [c for c in self._clients if c.alive]
                full = len(self._clients) >= self.max_clients
                if not full:
                    cl = _Client(sock, addr); self._clients.append(cl)
            if full:
                try:
                    sock.sendall(f"ERR busy: {self.max_clients} control clients already connected\n".encode())
                finally:
                    sock.close()
                self.log(f"control client {addr[0]}:{addr[1]} refused (max {self.max_clients})")
                continue
            self.log(f"control client {addr[0]}:{addr[1]} connected")
            threading.Thread(target=self._serve, args=(cl,), name="control-client", daemon=True).start()

    def _serve(self, c: _Client) -> None:
        c.send(f"OK ne0k-control {PROTOCOL} {self.version}")
        buf = b""
        # The 5 s socket timeout bounds sends; an idle client just loops here.
        try:
            while c.alive and not self._stop.is_set():
                try:
                    d = c.sock.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not d:
                    break
                buf += d
                if len(buf) > (4 << 20):
                    c.send("ERR line too long"); break
                *lines, buf = buf.split(b"\n")
                for ln in lines:
                    text = ln.decode("utf-8", "replace").rstrip("\r")
                    if not text.strip():
                        continue
                    if not self._handle(c, text):
                        c.alive = False
                        break
        finally:
            c.alive = False
            with self._lock:
                if c in self._clients:
                    self._clients.remove(c)
            try:
                c.sock.close()
            except OSError:
                pass
            self.log(f"control client {c.addr[0]}:{c.addr[1]} gone")

    def _handle(self, c: _Client, text: str) -> bool:
        word, _, rest = text.lstrip().partition(" ")
        word = word.upper()
        out: list[str] = []
        try:
            if word == "QUIT":
                c.send("OK bye")
                return False
            fn = getattr(self, "_cmd_" + word.lower(), None)
            if fn is not None:
                detail = fn(c, rest, out)
            else:
                acts = self.host.actions()
                if word not in acts:
                    raise ControlError(f"unknown command {word!r} (HELP lists them)")
                act = acts[word][1]
                detail = (act(rest.strip()) if getattr(act, "direct", False)
                          else self.host.run(lambda: act(rest.strip())))
                if isinstance(detail, list):        # an action that answers with data lines
                    out.extend(detail); detail = None
            for ln in out:
                c.send(ln)
            c.send("OK" + (f" {detail}" if detail not in (None, "") else ""))
        except ControlError as e:
            for ln in out:
                c.send(ln)
            c.send(f"ERR {e}")
        except Exception as e:                       # never let one bad command take the port down
            c.send(f"ERR {type(e).__name__}: {e}")
        return True

    # -- generic commands -------------------------------------------------------

    def _port(self):
        br = self._sync_bridge()
        if br is None:
            raise ControlError("serial port closed")
        return br

    def _cmd_help(self, c, rest, out):
        out += ["RTS 0|1 | STATUS | WRITE <text> | WRITEHEX <hex> | SEND <text> | SENDWAIT <text> "
                "| READ [n] | READTEXT [n] | EVENTS 0|1",
                "SECTIONS | LIST [section] | GET section.key | SET section.key value | DESCRIBE section.key",
                "INFO | LOG [n] | HELP | QUIT"]
        for k, (h, _) in sorted(self.host.actions().items()):
            out.append(f"{k} {h}")

    def _cmd_events(self, c, rest, out):
        on = rest.strip() not in ("0", "off", "OFF")
        c.events = on
        if on:
            br = self._sync_bridge()
            out.append(f"PORT 1 {br.link_path}" if br else "PORT 0")
            if br is not None:
                out.extend(br.status_lines())
        return f"events {int(on)}"

    def _serial_cmd(self, line: str, out):
        lines = self._port().command(line)
        errs = [ln for ln in lines if ln.startswith("ERR ")]
        out.extend(ln for ln in lines if not ln.startswith("ERR "))
        if errs:
            raise ControlError(errs[-1][4:])

    def _cmd_rts(self, c, rest, out):
        if rest.strip() not in ("0", "1"):
            raise ControlError("RTS 0 or RTS 1")
        self._serial_cmd("RTS " + rest.strip(), out)

    def _cmd_status(self, c, rest, out):
        self._serial_cmd("STATUS", out)

    def _write(self, data: bytes) -> str:
        n = self._port().write(data)
        return f"{n}/{len(data)}"

    def _cmd_write(self, c, rest, out):
        return self._write(rest.encode("utf-8"))

    def _cmd_writehex(self, c, rest, out):
        try:
            data = bytes.fromhex(rest.strip())
        except ValueError:
            raise ControlError("WRITEHEX takes hex digits") from None
        return self._write(data)

    def _send(self, data: bytes) -> None:
        br = self._port()

        def go():
            room = br.buf_size - len(br.txbuf)
            if not br.cts:
                return "not ready (CTS 0)"
            if len(data) > room:
                return f"too long: {len(data)} bytes, {room} free (buffer {br.buf_size})"
            br._accept(data)
            if br.mode == "rts":
                errs: list[str] = []
                br._command("RTS 0", errs.append)
                br._command("RTS 1", errs.append)
                br._command("RTS 0", errs.append)
                return errs[-1][4:] if errs else None
            return None
        why = br.call(go)
        if why:
            raise ControlError(why)

    def _cmd_send(self, c, rest, out):
        if not rest:
            raise ControlError("SEND <text>")
        self._send(rest.encode("utf-8"))
        return "queued"

    def _cmd_sendwait(self, c, rest, out):
        if not rest:
            raise ControlError("SENDWAIT <text>")
        evt, box = threading.Event(), []
        with self._lock:
            self._txwait.append((evt, box))
        try:
            self._send(rest.encode("utf-8"))
        except ControlError:
            with self._lock:
                self._txwait = [w for w in self._txwait if w[0] is not evt]
            raise
        if not evt.wait(600.0):
            raise ControlError("timeout waiting for TXDONE")
        if box[0] != "TXDONE":
            raise ControlError(box[0])
        return box[0]

    def _take_rx(self, rest: str) -> bytes:
        try:
            n = int(rest.strip() or 4096)
        except ValueError:
            raise ControlError("READ [n]") from None
        with self._rx_lock:
            data = bytes(self._rx[:n]); del self._rx[:n]
        return data

    def _cmd_read(self, c, rest, out):
        data = self._take_rx(rest)
        if data:
            out.append("DATA " + data.hex())
        return str(len(data))

    def _cmd_readtext(self, c, rest, out):
        data = self._take_rx(rest)
        if data:
            t = data.decode("utf-8", "backslashreplace")
            out.append("TEXT " + t.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n"))
        return str(len(data))

    # -- configuration ------------------------------------------------------------

    def _field(self, ref: str):
        sec, _, key = ref.strip().partition(".")
        secs = self.host.run(self.host.sections)
        if sec not in secs:
            raise ControlError(f"no section {sec!r} (SECTIONS lists them)")
        obj = secs[sec]
        fields = {f.name: f for f in dataclasses.fields(obj)}
        if key not in fields:
            raise ControlError(f"no key {key!r} in {sec}")
        return sec, key, obj, fields[key]

    def _cmd_sections(self, c, rest, out):
        secs = self.host.run(self.host.sections)
        for name in secs:
            out.append(f"{name}\tapply={self.host.apply_class(name)}")

    def _cmd_list(self, c, rest, out):
        secs = self.host.run(self.host.sections)
        want = rest.strip()
        if want and want not in secs:
            raise ControlError(f"no section {want!r}")
        for name, obj in secs.items():
            if want and name != want:
                continue
            cls = self.host.apply_class(name)
            for f in dataclasses.fields(obj):
                md = f.metadata or {}
                parts = [f"{name}.{f.name}={fmt_value(getattr(obj, f.name))}", f"type={type(f.default).__name__}",
                         f"apply={cls}"]
                if md.get("lo") is not None:
                    parts.append(f"lo={md['lo']}")
                if md.get("hi") is not None:
                    parts.append(f"hi={md['hi']}")
                if md.get("choices"):
                    parts.append("choices=" + "|".join(map(str, md["choices"])))
                out.append("\t".join(parts))

    def _cmd_get(self, c, rest, out):
        sec, key, obj, _ = self._field(rest)
        out.append(f"{sec}.{key}={fmt_value(getattr(obj, key))}")

    def _cmd_describe(self, c, rest, out):
        sec, key, obj, f = self._field(rest)
        md = f.metadata or {}
        out += [f"{sec}.{key}={fmt_value(getattr(obj, key))}", f"default={fmt_value(f.default)}",
                f"type={type(f.default).__name__}", f"apply={self.host.apply_class(sec)}",
                f"label={md.get('label', '')}", f"group={md.get('group', '')}", f"tip={md.get('tip', '')}"]

    def _cmd_set(self, c, rest, out):
        ref, _, raw = rest.strip().partition(" ")
        if not raw:
            raise ControlError("SET section.key value")
        sec, key, _, f = self._field(ref)
        v = parse_value(f, raw)
        how = self.host.run(lambda: self.host.set_param(sec, key, v))
        out.append(f"{sec}.{key}={fmt_value(v)}")
        return how

    # -- debug ------------------------------------------------------------------------

    def _cmd_info(self, c, rest, out):
        rows: list[tuple[str, object]] = [("version", self.version), ("host", self.host.name),
                                          ("control.bind", f"{self.bind}:{self.port}"),
                                          ("control.clients", len(self._clients)),
                                          ("control.rx_buffered", len(self._rx)),
                                          ("control.rx_dropped", self.rx_dropped)]
        rows += self.host.run(self.host.info)
        rows += self.host.info_offthread()
        br = self._sync_bridge()
        if br is None:
            rows.append(("serial", "closed"))
        else:
            rows += [("serial", br.link_path), ("serial.ctl", br.ctl_path), ("serial.mode", br.mode),
                     ("serial.buffer", br.buf_size), ("serial.busy", int(br.busy)),
                     ("serial.tx_bytes", br.tx_count), ("serial.rx_bytes", br.rx_count),
                     ("serial.retries", br.retry_count), ("serial.fails", br.fail_count)]
            try:
                rows += [("serial.status", " / ".join(br.status_lines()))]
            except Exception as e:
                rows.append(("serial.status", f"unavailable: {e}"))
        rows.append(("time", time.strftime("%Y-%m-%d %H:%M:%S")))
        out += [f"{k}={fmt_value(v)}" for k, v in rows]

    def _cmd_log(self, c, rest, out):
        try:
            n = int(rest.strip() or 50)
        except ValueError:
            raise ControlError("LOG [n]") from None
        lines = self.host.run(lambda: self.host.log_tail(n))
        if lines is None:
            raise ControlError("no log on this host")
        out += lines
