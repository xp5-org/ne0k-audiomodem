# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Feeds decoded CSS packets and OFDM payloads to the one Log and drives the shared PLL bar.

Only one modem runs at a time, so there is no per-modem view: this has no widgets of its own."""
from __future__ import annotations

import time
from typing import Callable

from PySide6.QtCore import QObject, QTimer

from ..engine import Decoded, LinkEvent, OfdmData
from .lockbar import LockBar


class ModemFeed(QObject):
    def __init__(self, lockbar: LockBar, log: Callable[[str], None], parent=None) -> None:
        super().__init__(parent)
        self.lockbar = lockbar
        self._log = log
        self.link = None
        self.detailed = True                       # Debug & Logging > Event log detail
        self.count = 0                             # packets / payloads received since the link started
        self._t_hs: float | None = None            # when the current OFDM handshake began
        self._timer = QTimer(self); self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll_handshake)

    def _stamp(self, line: str) -> None:
        self._log(f"{time.strftime('%H:%M:%S')} {line}")

    def set_open(self, on: bool) -> None:
        self.lockbar.set_open(on)

    def set_link(self, link) -> None:
        self.link = link
        self._t_hs = None
        self.lockbar.reset()
        self.count = 0
        self.lockbar.align.setVisible(link is None or not hasattr(link, "cfg") or not hasattr(link.cfg, "mode"))
        if link is not None and hasattr(link, "state"):
            self._timer.start()
        else:
            self._timer.stop()

    def on_link(self, ev: LinkEvent) -> None:
        if ev.kind == "lock":
            self.lockbar.locked()                  # on_decoded() refines it with timing

    def _poll_handshake(self) -> None:
        l = self.link
        if not l:
            return
        if l.state == "handshake" and self._t_hs is None:
            self._t_hs = time.monotonic()
        elif l.state == "up" and self._t_hs is not None:
            self.lockbar.locked(time.monotonic() - self._t_hs)   # handshake -> link up
            self._t_hs = None
        elif l.state not in ("handshake", "up"):
            self._t_hs = None

    def on_decoded(self, d: Decoded) -> None:
        self.count += 1
        self.lockbar.locked(d.acq_s, d.sync_score)
        print(f"css rx: {d.text!r} ok={d.ok} snr={d.snr_db:.1f}", flush=True)   # demo greps the log
        flag = "OK " if d.ok else "ERR"
        if not self.detailed:
            return self._stamp(f"CSS rx {flag} {d.text!r}")
        self._stamp(f"CSS rx {flag} src={d.src} dst={d.dst} seq={d.seq} "
                    f"{d.type_name} snr={d.snr_db:.1f}dB frames={d.frames}/{d.frames_errored}err "
                    f"rs={d.rs_errors}  {d.text!r}")

    def on_data(self, d: OfdmData) -> None:
        self.count += 1
        print(f"mesh rx: {d.text!r} {d.nbytes}B", flush=True)             # demo greps the log
        self._stamp(f"rx {d.nbytes} B  {d.text!r}")
