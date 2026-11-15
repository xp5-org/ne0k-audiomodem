# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""The desktop app as a TCP control-port host: everything the window does from its buttons,
the settings sections and a status dump.  Every call runs on the Qt thread."""
from __future__ import annotations

import collections
import dataclasses
import threading

from PySide6.QtCore import QObject, Qt, QThread, Signal

from .. import native
from ..audio import SINK, SOURCE
from ..engine.control import ControlError, ControlHost
from .appsettings import SECTIONS


class _Invoker(QObject):
    job = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.job.connect(self._run, Qt.QueuedConnection)

    def _run(self, job) -> None:
        job()


class GuiHost(ControlHost):
    name = "gui"
    # Read on every use: the ident timers / callsign and diagnostics capture + log detail.  Everything else is read when the modem starts.
    live_sections = frozenset({"ham", "diag"})

    def __init__(self, win, log_lines: int = 2000) -> None:
        self.w = win
        self._inv = _Invoker()
        self._log: collections.deque[str] = collections.deque(maxlen=log_lines)
        prev = win.diag.on_line

        def tee(line: str, prev=prev) -> None:
            self._log.append(line)
            if prev:
                prev(line)
        win.diag.on_line = tee

    # -- thread hop ---------------------------------------------------------------

    def run(self, fn, timeout: float = 15.0):
        if QThread.currentThread() is self._inv.thread():
            return fn()
        box: list = []
        done = threading.Event()

        def job() -> None:
            try:
                box.append((True, fn()))
            except BaseException as e:
                box.append((False, e))
            done.set()
        self._inv.job.emit(job)
        if not done.wait(timeout):
            raise ControlError("GUI thread did not answer (busy or blocked)")
        ok, v = box[0]
        if not ok:
            raise v
        return v

    # -- state --------------------------------------------------------------------

    def serial(self):
        return self.w.serial

    def sections(self) -> dict:
        return {name: getattr(self.w.app, name) for name in SECTIONS}

    def set_param(self, section: str, key: str, value) -> str:
        w = self.w
        cur = getattr(w.app, section)
        # A fresh object: the running link / bridge keep the copy they were built with until the modem restarts.
        setattr(w.app, section, dataclasses.replace(cur, **{key: value}))
        w._after_settings()
        if section in self.live_sections:
            return "applied"
        return "pending-restart" if w.modem_on else "next-start"

    def log_tail(self, n: int) -> list[str]:
        return list(self._log)[-n:] if n > 0 else []

    def _msg(self) -> str:
        return self.w.statusBar().currentMessage()

    def info(self):
        w = self.w
        pipe = w.session.pipe
        rows = [("audio", "open" if w.session.is_open else "closed")]
        if pipe is not None:
            rows += [("audio.rx", w.io.describe(w.io.rx.currentData(), SOURCE)),
                     ("audio.tx", w.io.describe(w.io.tx.currentData(), SINK)),
                     ("audio.rate", w.session.spec.rate)]
        g = w.app.general
        rows += [("modem", "on" if w.modem_on else "off"), ("modem.mode", g.modem), ("modem.topology", g.topology),
                 ("modem.node_id", g.node_id), ("modem.peer_id", g.peer_id)]
        m = w.link
        if m is not None:
            rows += [("link", "mesh node %d" % m.me),
                     ("link.states", ", ".join(f"{k} {v}" for k, v in m.link_states().items()) or "none"),
                     ("link.neighbors", ", ".join(f"{n} ({snr:.0f} dB)" for n, (snr, _) in sorted(m.neighbors.items()))
                      or "none"),
                     ("link.tx", m.tx_count), ("link.rx", m.rx_count), ("link.css_fallbacks", m.css_fallbacks),
                     ("link.ofdm_only_violations", len(m.ofdm_only_violations))]
        else:
            rows.append(("link", "none"))
        rows += [("udp", "open" if w.udp else "closed"), ("cssbcast", "open" if w.bcast else "closed"),
                 ("link_info", w.column.link_info.detail.text().replace("\n", " | ")),
                 ("status", self._msg())]
        try:
            libs = native.inventory()
            rows.append(("c_libs", f"{sum(1 for v in libs if v['ok'])}/{len(libs)}"))
        except Exception as e:
            rows.append(("c_libs", f"unavailable: {e}"))
        return rows

    # -- actions (what the buttons do) -----------------------------------------------

    def actions(self) -> dict:
        return {
            "POWER": ("ON|OFF  start / stop the modem (the Start modem button)", self._power),
            "RESTART": ("  power-cycle the modem so pending settings apply (reopens the serial port if it was open)",
                        self._restart),
            "AUDIO": ("OPEN <rx> <tx> | CLOSE  open audio devices (addresses as on the command line: pulse:, sd:N, "
                      "loop:x, file:x.wav, null:)", self._audio),
            "DEVICES": ("  audio devices this host can open", self._devices),
            "PORT": ("OPEN|CLOSE  the virtual serial port (Ctrl+P)", self._port),
            "IDENT": ("  send the station ID now", lambda a: self._after(self.w._send_id_now)),
            "TEXT": ("<message>  send one message on the mesh link (not while the serial port owns it)",
                     lambda a: self._after(lambda: self.w.send_text(a)) if a else self._need("TEXT <message>")),
            "SAVE": ("  write the current settings to disk (as the settings dialog's OK does)", self._save),
        }

    @staticmethod
    def _need(usage: str):
        raise ControlError(usage)

    def _after(self, fn) -> str:
        fn()
        return self._msg()

    def _power(self, a: str) -> str:
        a = a.upper()
        if a == "ON":
            if not self.w.session.is_open:
                raise ControlError("audio is not open (AUDIO OPEN <rx> <tx>)")
            if not self.w.modem_on:
                self.w._power_on()
        elif a == "OFF":
            if self.w.modem_on:
                self.w._power_off()
        else:
            raise ControlError("POWER ON or POWER OFF")
        return self._msg()

    def _restart(self, a: str) -> str:
        w = self.w
        if not w.modem_on:
            raise ControlError("modem is off (POWER ON starts it with the current settings)")
        had_port = w.serial is not None
        w._power_off()
        w._power_on()
        if had_port and w.serial is None:
            w._start_serial()
        return self._msg()

    def _audio(self, a: str) -> str:
        w = self.w
        words = a.split()
        if words[:1] and words[0].upper() == "CLOSE":
            w.session.close()
            return "audio closed"
        if not words or words[0].upper() != "OPEN" or len(words) not in (2, 3):
            raise ControlError("AUDIO OPEN <rx> [<tx>] | AUDIO CLOSE  (use - for none)")
        rx = None if words[1] == "-" else words[1]
        tx = None if len(words) < 3 or words[2] == "-" else words[2]
        w.io.select(rx, tx)
        try:
            w.session.open(rx, tx)
        except Exception as e:
            raise ControlError(f"open failed: {e}") from None
        return f"audio open rx={rx or 'none'} tx={tx or 'none'}"

    def _devices(self, a: str) -> str:
        rows = []
        for kind in (SOURCE, SINK):
            for d in self.w.session.devices(kind):
                rows.append(f"{kind} {d.address}  {d.name}" + ("  (default)" if d.default else ""))
        return rows

    def _port(self, a: str) -> str:
        w = self.w
        a = a.upper()
        if a == "OPEN":
            if w.serial is None:
                w._start_serial()
            if w.serial is None:
                raise ControlError(self._msg() or "serial port did not open")
            return f"{w.serial.link_path} ctl={w.serial.ctl_path}"
        if a == "CLOSE":
            if w.serial is not None:
                w._stop_serial()
                w._restart_link()
            return self._msg()
        raise ControlError("PORT OPEN or PORT CLOSE")

    def _save(self, a: str) -> str:
        self.w.app.save()
        return "settings saved"
