# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""TCP control port (engine/control.py): the serial-port commands answer as the .ctl socket does, data moves,
settings are listed / read / set and take effect, the GUI host's actions drive the window."""
import concurrent.futures
import dataclasses
import os
import socket
import time

import pytest

from modemgui.audio import Registry
from modemgui.engine import Session
from modemgui.engine.control import ControlError, ControlServer, EngineHost, parse_value
from modemgui.engine.modemcfg import PRESETS, GeneralConfig
from modemgui.engine.serialbridge import HandshakePort, SerialBridge


class Client:
    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        self.s = socket.create_connection((host, port), timeout=120)
        self.f = self.s.makefile("r", encoding="utf-8", newline="\n")
        self.events: list[str] = []
        self.banner = self.line()

    def line(self) -> str:
        return self.f.readline().rstrip("\n")

    def cmd(self, text: str) -> tuple[list[str], str]:
        """Send one command; return (data lines, final OK/ERR line).  Event lines (EVENTS 1) go to .events."""
        self.s.sendall((text + "\n").encode())
        out = []
        while True:
            ln = self.line()
            if ln == "OK" or ln.startswith(("OK ", "ERR ")):
                return out, ln
            out.append(ln)

    def wait_event(self, want: str, timeout: float) -> bool:
        """Read event lines until one equals ``want``."""
        self.s.settimeout(timeout)
        try:
            while True:
                ln = self.line()
                self.events.append(ln)
                if ln == want:
                    return True
        except (socket.timeout, OSError):
            return False
        finally:
            self.s.settimeout(120)

    def close(self) -> None:
        self.s.close()


def test_values_are_checked_against_the_field_description():
    f = {x.name: x for x in dataclasses.fields(GeneralConfig)}
    assert parse_value(f["node_id"], "7") == 7
    assert parse_value(f["beacon"], "off") is False and parse_value(f["beacon"], "1") is True
    assert parse_value(f["beacon_min_s"], "30") == 30.0
    with pytest.raises(ControlError, match="above maximum"):
        parse_value(f["node_id"], "16")
    with pytest.raises(ControlError, match="not one of"):
        parse_value(f["modem"], "fm")
    with pytest.raises(ControlError, match="bad value"):
        parse_value(f["node_id"], "seven")


@pytest.fixture
def bridges(tmp_path):
    reg = Registry()
    made = []

    def mk(name, node, peer, rx, tx, **kw):
        s = Session(reg, 48000, 1024); s.open(rx, tx)
        # decoder auto: the Python CSS decoder, so the test does not depend on libcsspll loading on this host
        cfg = dataclasses.replace(PRESETS["Fast (POC)"], node_id=node, peer_id=peer, decoder="auto")
        b = SerialBridge(s, cfg, str(tmp_path / name), **kw); b.start()
        srv = ControlServer(EngineHost(b, {"css": cfg}), 0, max_clients=2); srv.start()
        made.append((srv, b, s))
        return b, srv
    yield mk, tmp_path
    for srv, b, s in made:
        srv.stop(); b.stop(); s.close()


def test_serial_commands_match_the_ctl_socket_and_data_crosses(bridges):
    mk, d = bridges
    a, sa = mk("A", 1, 2, "loop:cba", "loop:cab", buf_size=8, ack=False)
    b, sb = mk("B", 2, 1, "loop:cab", "loop:cba", buf_size=8, ack=False)
    ca, cb = Client(sa.port), Client(sb.port)
    assert ca.banner.startswith("OK ne0k-control 1 ")

    ctl = socket.socket(socket.AF_UNIX); ctl.connect(str(d / "A") + ".ctl"); ctl.settimeout(2)
    time.sleep(0.2); ctl.recv(256)                                  # the status a new .ctl client is greeted with
    ctl.sendall(b"STATUS\n"); time.sleep(0.2)
    assert ca.cmd("STATUS") == (ctl.recv(256).decode().splitlines(), "OK")   # same lines, plus OK
    assert ca.cmd("STATUS")[0] == ["CTS 1", "BUF 0/8"]
    ctl.sendall(b"RTS 1\n"); time.sleep(0.2); ctl.sendall(b"RTS 0\n")
    assert ctl.recv(256) == b"ERR empty\n"
    assert ca.cmd("RTS 1") == ([], "ERR empty")                    # same refusal, same words
    assert ca.cmd("RTS 0") == ([], "OK")
    ctl.close()

    assert cb.cmd("EVENTS 1")[1] == "OK events 1"
    assert ca.cmd("WRITE " + "x" * 20) == ([], "OK 8/20")         # a tty write: only the free room is taken
    assert ca.cmd("WRITE more") == ([], "OK 0/4")
    assert ca.cmd("STATUS")[0] == ["CTS 1", "BUF 8/8"]
    pb = HandshakePort(str(d / "B"))
    assert ca.cmd("RTS 1") == ([], "OK")
    assert cb.wait_event("RX 8", 60)                                # the .ctl event stream reaches TCP clients
    assert pb.read(100, 5) == b"x" * 8                             # the pty still gets the bytes ...
    assert cb.cmd("READ") == (["DATA " + (b"x" * 8).hex()], "OK 8")   # ... and so does READ
    assert cb.cmd("READ") == ([], "OK 0")
    ca.cmd("RTS 0")
    lines, ok = ca.cmd("INFO")
    assert ok == "OK" and "serial.tx_bytes=8" in lines and "host=serial-cli" in lines
    pb.close()


def test_config_read_only_on_the_cli_host_and_clients_limited(bridges):
    mk, _ = bridges
    _, srv = mk("A", 1, 2, "loop:dba", "loop:dab", buf_size=8, ack=False)
    c1, c2 = Client(srv.port), Client(srv.port)
    assert c1.cmd("GET css.node_id") == (["css.node_id=1"], "OK")
    assert c1.cmd("SECTIONS") == (["css\tapply=restart"], "OK")
    assert c1.cmd("SET css.node_id 3")[1].startswith("ERR the serial CLI builds its link once")
    assert c1.cmd("GET css.nope")[1] == "ERR no key 'nope' in css"
    assert c1.cmd("FROB")[1].startswith("ERR unknown command")
    assert c2.cmd("HELP")[1] == "OK"
    c3 = Client(srv.port)                                            # max_clients=2: refused, clearly
    assert c3.banner.startswith("ERR busy: 2 control clients")
    assert c1.cmd("QUIT") == ([], "OK bye")
    time.sleep(0.5)
    c4 = Client(srv.port)                                            # a slot freed up
    assert c4.banner.startswith("OK ne0k-control")
    for c in (c1, c2, c3, c4):
        c.close()


# -- desktop app host ----------------------------------------------------------------------------------------

def test_gui_host_settings_actions_and_serial_port(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from modemgui.ui.control_host import GuiHost
    from modemgui.ui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    w = MainWindow(Session(Registry()), QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    srv = ControlServer(GuiHost(w), 0); srv.start()
    pool = concurrent.futures.ThreadPoolExecutor(1)
    c = pool.submit(Client, srv.port).result(10)

    def ask(text: str, timeout: float = 30.0):
        """The server hops to the Qt thread, so keep the event loop turning while the client waits."""
        fut = pool.submit(c.cmd, text)
        end = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < end:
            app.processEvents(); time.sleep(0.01)
        return fut.result(0)

    def spin(pred, timeout: float = 10.0) -> bool:
        end = time.monotonic() + timeout
        while not pred() and time.monotonic() < end:
            app.processEvents(); time.sleep(0.02)
        return pred()

    try:
        lines, ok = ask("SECTIONS")
        assert ok == "OK" and "general\tapply=restart" in lines and "ham\tapply=live" in lines
        assert ask("SET general.node_id 7") == (["general.node_id=7"], "OK next-start")
        assert w.app.general.node_id == 7
        assert ask("SET general.node_id 99")[1] == "ERR 99 above maximum 15"
        assert ask("SET general.modem fm")[1].startswith("ERR 'fm' not one of")
        assert ask("SET diag.detail basic")[1] == "OK applied" and w.feed.detailed is False   # live section
        assert ask("SET serial.link_path " + str(tmp_path / "m0"))[1] == "OK next-start"

        assert ask("SET css.decoder auto")[1] == "OK next-start"   # Python decoder: no libcsspll needed here
        assert ask("POWER ON")[1].startswith("ERR audio is not open")
        assert ask("AUDIO OPEN loop:ctl loop:ctl")[1] == "OK audio open rx=loop:ctl tx=loop:ctl"
        assert w.session.is_open
        assert ask("POWER ON")[1].startswith("OK")
        assert w.modem_on and w.link is not None and w.link.me == 7
        assert ask("SET general.node_id 5")[1] == "OK pending-restart"
        assert w.link.me == 7                                       # the running node keeps its copy ...
        assert ask("RESTART")[1].startswith("OK")
        assert w.link.me == 5                                       # ... until RESTART
        lines, _ = ask("LIST general")
        assert "general.node_id=5\ttype=int\tapply=restart\tlo=0\thi=15" in lines
        assert ask("GET general.node_id") == (["general.node_id=5"], "OK")

        lines, ok = ask("PORT OPEN")
        assert ok.startswith("OK " + str(tmp_path / "m0")) and w.serial is not None
        assert os.path.islink(tmp_path / "m0")
        assert spin(lambda: srv._bridge is w.serial)               # the server follows the port the GUI opened
        lines, ok = ask("STATUS")
        assert ok == "OK" and lines[0].startswith("CTS ")
        lines, _ = ask("INFO")
        assert "modem=on" in lines and "modem.node_id=5" in lines and f"serial={tmp_path / 'm0'}" in lines
        assert ask("PORT CLOSE")[1].startswith("OK") and w.serial is None
        assert ask("STATUS")[1] == "ERR serial port closed"
        lines, ok = ask("LOG 200")
        assert ok == "OK" and any("modem powered on" in ln for ln in lines)
        assert ask("POWER OFF")[1].startswith("OK") and not w.modem_on
        assert ask("AUDIO CLOSE")[1] == "OK audio closed" and not w.session.is_open
    finally:
        pool.submit(c.close).result(5)
        pool.shutdown()
        srv.stop()
        w.close()
