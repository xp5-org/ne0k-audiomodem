# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""The desktop app as a TCP control-port host: everything the window does from its buttons,
the settings sections and a status dump.  Every call runs on the Qt thread."""
from __future__ import annotations

import base64
import collections
import dataclasses
import io
import json
import os
import re
import threading
import time

import numpy as np

from PySide6.QtCore import QObject, Qt, QThread, Signal

from .. import native
from ..audio import SINK, SOURCE, hostmixer
from ..engine import calibrate, ranging, volcal
from ..engine.control import ControlError, ControlHost, direct
from .appsettings import SECTIONS


def device_name(addr: str, devices) -> str:
    """The listed name of the device ``addr`` opens.  ``sd:name=<text>`` is resolved the way the sounddevice
    connector resolves it (the first device of that kind whose name contains the text, case-insensitive), so the
    mixer acts on the same device as the audio stream rather than on the OS default."""
    for d in devices:
        if d.address == addr:
            return d.name
    conn, _, dev = addr.partition(":")
    if dev.startswith("name="):
        want = dev[5:].lower()
        for d in devices:
            if d.connector == conn and want in d.name.lower():
                return d.name
    return ""


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
        self._mixer_snap: dict | None = None
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
            # input overflows drop samples (PortAudio "input overflow"): a lost burst with no other trace
            src = getattr(pipe, "_src", None)
            rows += [("audio.rx.overflows", getattr(src, "overflows", 0)),
                     ("audio.rx.last_error", pipe.stats.rx_error or "none")]
            rows += self._level_rows()
        # diag.wav_rx / wav_tx / event_db write here (rx_*.wav, tx_*.wav, events_*.sqlite)
        rows += [("diag.dir", str(w.diag.dir) if w.diag.dir else "none")]
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
            if getattr(m, "rx", None) is not None:
                rows += [("link.rx_stats", m.rx.stats()), ("link.rx_busy", int(m.rx.channel_busy()))]
            # CSS carrier sense part by part, the MAC's whole busy verdict and its outbox (why a DATA waits)
            try:
                rows += [("css.cs", m.css.cs_state()), ("mac.channel_busy", int(m._channel_busy())),
                         ("mac.csma", m.csma.state(time.monotonic())),
                         ("mac.hold_left", f"{max(0.0, m._hold_until - time.monotonic()):.1f}s")]
            except Exception as e:
                rows.append(("css.cs", f"unavailable: {e}"))
        else:
            rows.append(("link", "none"))
        rows.append(("ui.pll", w.pll.describe()))
        rows += [("udp", "open" if w.udp else "closed"), ("cssbcast", "open" if w.bcast else "closed"),
                 ("link_info", w.column.link_info.detail.text().replace("\n", " | ")),
                 ("status", self._msg())]
        cpu = w.strip.meter.last
        rows.append(("cpu.total_pct", f"{cpu['total']:.1f}"))
        rows += [(f"cpu.{g}_pct", f"{v:.1f}") for g, v in cpu["groups"].items()]
        rows.append(("cpu.threads", ", ".join(f"{n}={p:.1f}" for n, _, p in cpu["threads"][:10] if p >= 0.05) or "none"))
        try:
            libs = native.inventory()
            rows.append(("c_libs", f"{sum(1 for v in libs if v['ok'])}/{len(libs)}"))
        except Exception as e:
            rows.append(("c_libs", f"unavailable: {e}"))
        return rows

    def _level_rows(self, window_s: float = 10.0) -> list:
        """The front-page level meter as numbers, over the last ``window_s`` seconds.  ``noise_floor`` is the
        quietest block heard while we were not transmitting (our own playback hearing itself is left out)."""
        now = time.monotonic()
        recent = [lv for t, lv in self.w.level_hist if now - t <= window_s]
        if not recent:
            return [("audio.rx.level", "no data")]
        quiet = [lv.rms_db for lv in recent if not lv.tx_active]
        loud = [lv.rms_db for lv in recent if lv.tx_active]
        last = recent[-1]
        rows = [("audio.rx.peak_dbfs", f"{max(lv.peak_db for lv in recent):.1f}"),
                ("audio.rx.rms_dbfs", f"{last.rms_db:.1f}"),
                ("audio.rx.noise_floor_dbfs", f"{min(quiet):.1f}" if quiet else "n/a (transmitting)"),
                ("audio.rx.own_tx_rms_dbfs", f"{max(loud):.1f}" if loud else "n/a"),
                ("audio.rx.clipped_total", last.clipped),
                ("audio.rx.level_window_s", f"{window_s:g}")]
        return rows

    # -- host OS mixer (off the Qt thread: pactl / amixer / CoreAudio calls can take a while) ----------------------

    _MIX_KIND = {"in": SOURCE, "out": SINK}

    def _mixer_targets(self) -> dict:
        """{"in"|"out": (address, device name)} for the devices the session has open."""
        def addrs():
            w = self.w
            if w.session.pipe is None:
                return {}
            return {"in": w.io.rx.currentData(), "out": w.io.tx.currentData()}
        out = {}
        for side, addr in self.run(addrs).items():
            if not addr:
                continue
            kind = self._MIX_KIND[side]
            out[side] = (addr, device_name(addr, self.w.session.devices(kind)))
        return out

    def _mixer_rows(self, targets: dict | None = None) -> list:
        targets = self._mixer_targets() if targets is None else targets
        rows = []
        for side, (addr, name) in targets.items():
            rows += hostmixer.get(addr, self._MIX_KIND[side], name).rows(f"audio.host.{side}")
        return rows

    def info_offthread(self):
        return self._mixer_rows()

    @staticmethod
    def _parse_level(v: str) -> float:
        try:
            x = float(v[:-1]) / 100.0 if v.endswith("%") else float(v)
        except ValueError:
            raise ControlError(f"bad level {v!r} (0..1 or N%)") from None
        if not 0.0 <= x <= 1.5:
            raise ControlError(f"level {v} out of range (0..1, Pulse allows up to 1.5)")
        return x

    def _mixer(self, a: str):
        words = a.split()
        op = words[0].upper() if words else "GET"
        usage = ("MIXER [GET] | SET in|out <0..1|N%> | MUTE in|out on|off | SNAPSHOT | RESTORE")
        targets = self._mixer_targets()
        if not targets:
            raise ControlError("audio is not open (AUDIO OPEN <rx> <tx>)")
        kv = (lambda rows: [f"{k}={v}" for k, v in rows])
        if op == "GET":
            return kv(self._mixer_rows(targets))
        if op in ("SET", "MUTE"):
            if len(words) != 3 or words[1].lower() not in self._MIX_KIND:
                raise ControlError(usage)
            side = words[1].lower()
            if side not in targets:
                raise ControlError(f"no {side}put device open")
            addr, name = targets[side]
            if op == "SET":
                kw = {"volume": self._parse_level(words[2])}
            else:
                if words[2].lower() not in ("on", "off", "1", "0", "yes", "no"):
                    raise ControlError(usage)
                kw = {"mute": words[2].lower() in ("on", "1", "yes")}
            try:
                st = hostmixer.apply(addr, self._MIX_KIND[side], name, **kw)
            except hostmixer.MixerError as e:
                raise ControlError(str(e)) from None
            return kv(st.rows(f"audio.host.{side}"))
        if op == "SNAPSHOT":
            self._mixer_snap = {side: (addr, name, hostmixer.get(addr, self._MIX_KIND[side], name))
                                for side, (addr, name) in targets.items()}
            return [f"snapshot {side} volume={st.volume} mute={st.mute}"
                    for side, (_, _, st) in self._mixer_snap.items()]
        if op == "RESTORE":
            if not self._mixer_snap:
                raise ControlError("no snapshot (MIXER SNAPSHOT first)")
            rows = []
            for side, (addr, name, st) in self._mixer_snap.items():
                if not st.settable:
                    rows.append(f"restore {side} skipped (not settable)")
                    continue
                try:
                    hostmixer.apply(addr, self._MIX_KIND[side], name, volume=st.volume, mute=st.mute)
                    rows.append(f"restore {side} volume={st.volume} mute={st.mute}")
                except hostmixer.MixerError as e:
                    rows.append(f"restore {side} failed: {e}")
            return rows
        raise ControlError(usage)

    # -- acoustic calibration (engine/calibrate.py), off the Qt thread: each takes seconds ------------------------

    @staticmethod
    def _band_arg(v: str) -> tuple[float, float]:
        try:
            lo, hi = (float(x) for x in v.split("-"))
        except ValueError:
            raise ControlError(f"bad band {v!r} (lo-hi in Hz, e.g. 400-9000)") from None
        if not 50 <= lo < hi <= 24000:
            raise ControlError(f"bad band {v!r}")
        return lo, hi

    def _cal_setup(self, band=None, css=None):
        """(pipe, probe, current (n_fft, cp_ratio)) or ControlError: audio open, modem off (the link would answer the
        probe or transmit over it).  ``band`` / ``css`` override the OFDM / CSS bands from the settings: LISTEN and
        SEND must build the same probe, so two hosts with different settings pass the same band= / css=."""
        def grab():
            w = self.w
            if w.session.pipe is None:
                raise ControlError("audio is not open (AUDIO OPEN <rx> <tx>)")
            if w.modem_on:
                raise ControlError("modem is on (POWER OFF first: the link would transmit over the probe)")
            o, c = w.app.ofdm, w.app.css
            return (w.session.pipe, w.session.spec.rate, (o.freq_lo, o.freq_hi), (c.f_min, c.f_max),
                    (o.n_fft, o.cp_ratio))
        pipe, fs, ob, cb, cur = self.run(grab)
        ob, cb = band or ob, css or cb
        return pipe, calibrate.make_probe(fs, ob[0], ob[1], cb), cur

    @staticmethod
    def _record(pipe, secs: float):
        time.sleep(secs)
        return pipe.history(int(secs * pipe.spec.rate))[0]

    def _calibrate(self, a: str):
        usage = ("CALIBRATE NOISE [secs] | SELF | LISTEN [secs] | SEND  [band=lo-hi] [css=lo-hi]  (modem off; two "
                 "hosts: LISTEN on the receiver, then SEND on the sender 1-5 s later, both with the same band= css=)"
                 " | " + self._VOL_USAGE)
        kw = dict(w.split("=", 1) for w in a.split() if "=" in w)
        words = [w for w in a.split() if "=" not in w]
        op = words[0].upper() if words else ""
        if op == "VOLUME":
            return self._cal_volume(kw)
        kv = (lambda rows: [f"{k}={v}" for k, v in rows])
        try:
            arg = float(words[1]) if len(words) > 1 else None
        except ValueError:
            raise ControlError(usage) from None
        if op not in ("NOISE", "SELF", "LISTEN", "SEND") or set(kw) - {"band", "css"}:
            raise ControlError(usage)
        pipe, probe, cur = self._cal_setup(self._band_arg(kw["band"]) if "band" in kw else None,
                                           self._band_arg(kw["css"]) if "css" in kw else None)
        fs = probe.fs
        dur = len(probe.signal) / fs
        if op == "NOISE":
            x = self._record(pipe, min(max(arg or 3.0, 0.5), 25.0))
            r = calibrate.analyse(probe, x[:0], x)              # floor only (fails "too short" on purpose)
            return kv([("cal.floor_rms_dbfs", f"{r.floor_rms_db:.1f}"), ("cal.floor_band_dbfs", f"{r.floor_band_db:.1f}"),
                       ("cal.band_hz", f"{probe.band[0]:g}-{probe.band[1]:g}")])
        if op == "SEND":
            pipe.play(probe.signal)
            if not pipe.wait_tx(dur + 10):
                raise ControlError("TX did not drain")
            return kv([("cal.sent_s", f"{dur:.2f}"), ("cal.band_hz", f"{probe.band[0]:g}-{probe.band[1]:g}"),
                       ("cal.css_band_hz", f"{probe.css_band[0]:g}-{probe.css_band[1]:g}")])
        if op == "SELF":
            floor = self._record(pipe, 1.5)
            pipe.play(probe.signal)
            pipe.wait_tx(dur + 10)
            time.sleep(1.0)                                      # the room's ring-out
            rec = pipe.history(int((dur + 2.5) * fs))[0]
            res = calibrate.analyse(probe, rec, floor)
            return kv(res.rows("cal", cur)) + self._mixer_rows_kv()
        secs = min(max(arg or dur + 6.0, dur + 1.0), 29.0)      # LISTEN: the pipe keeps 30 s of history
        rec = self._record(pipe, secs)
        floor, why = calibrate.split_floor(probe, rec)
        if why:
            return kv([("cal.ok", "no"), ("cal.error", why + "  (same band= / css= on both hosts?)"
                        if "not found" in why else why)])
        res = calibrate.analyse(probe, rec, floor)
        return kv(res.rows("cal", cur)) + self._mixer_rows_kv()

    _VOL_USAGE = ("CALIBRATE VOLUME [from=0.2] [to=1.0] [steps=9] [level=-6] [band=lo-hi] [apply=0|1] [stop=1|0]  "
                  "sweep the OS output volume with the digital level held at level= dBFS peak; recommends the system "
                  "volume (modem off; restores the volume after unless apply=1)")

    def _cal_volume(self, kw: dict):
        """OS output-volume sweep (engine/volcal.py).  Quiet to loud; stop=1 ends early once it clearly distorts."""
        bad = set(kw) - {"band", "from", "to", "steps", "level", "apply", "stop"}
        if bad:
            raise ControlError(self._VOL_USAGE)
        try:
            lo, hi = float(kw.get("from", 0.2)), float(kw.get("to", 1.0))
            n, level = int(kw.get("steps", 9)), float(kw.get("level", volcal.LEVEL_DB))
            apply_, stop = kw.get("apply", "0") in ("1", "yes", "on"), kw.get("stop", "1") in ("1", "yes", "on")
        except ValueError:
            raise ControlError(self._VOL_USAGE) from None
        if not (0.0 <= lo < hi <= 1.0 and 2 <= n <= 21 and -40.0 <= level <= -1.0):
            raise ControlError("from/to 0..1 (from < to), steps 2..21, level -40..-1 dBFS")
        pipe, probe, _ = self._cal_setup(self._band_arg(kw["band"]) if "band" in kw else None)
        targets = self._mixer_targets()
        if "out" not in targets:
            raise ControlError("no output device open")
        addr, name = targets["out"]
        st0 = hostmixer.get(addr, SINK, name)
        if not st0.settable:
            raise ControlError(f"output volume is not settable here (backend {st0.backend}"
                               f"{': ' + '; '.join(st0.notes) if st0.notes else ''}); use CALIBRATE SELF at the "
                               f"current volume instead")
        fs = probe.fs
        burst = volcal.make_burst(fs, probe.band[0], probe.band[1], level)
        dur = len(burst.signal) / fs
        steps: list = []
        res = None
        try:
            for v in volcal.volume_grid(lo, hi, n):
                try:
                    st = hostmixer.apply(addr, SINK, name, volume=v, mute=False)
                except hostmixer.MixerError as e:
                    raise ControlError(f"setting the output volume to {v:.2f} failed: {e}") from None
                time.sleep(0.3)                                       # the OS volume ramp
                idle = self._record(pipe, volcal.IDLE_S)
                hx, hj = pipe.history(1)
                i0 = hj + len(hx)                                     # index of the next RX sample
                pipe.play(burst.signal)
                if not pipe.wait_tx(dur + 10):
                    raise ControlError("TX did not drain")
                # wait_tx means "handed to the sink", not "played": the device buffer (latency="high", Pulse adds
                # its own) still holds up to a second or so.  Record from the play call until the burst + 1.5 s.
                want = i0 + len(burst.signal) + int(1.5 * fs)
                t_end = time.time() + 6.0
                while time.time() < t_end:
                    x, j0 = pipe.history(1)
                    if j0 >= want:
                        break
                    time.sleep(0.1)
                x, j0 = pipe.history(int(want - i0 + 0.5 * fs))
                rec = x[max(0, i0 - j0):]
                step = volcal.measure_step(burst, rec, idle, st.volume if st.volume is not None else v, st.volume_db)
                steps.append(step)
                heard = [s for s in steps if s.heard]
                if stop and step.heard and len(heard) >= 3:
                    best = max(s.sinad_db for s in heard)
                    if step.clipped or (step.sinad_db < best - 8.0 and step.thd_db > -15.0):
                        break                                        # well past the knee: spare the speaker
            res = volcal.analyse_volume(steps, level)
        finally:
            target = res.rec_volume if (apply_ and res is not None and res.rec_volume is not None) else None
            try:
                if target is not None:
                    hostmixer.apply(addr, SINK, name, volume=target, mute=False)
                else:
                    hostmixer.apply(addr, SINK, name, volume=st0.volume, mute=st0.mute)
            except hostmixer.MixerError:
                pass
        kv = (lambda rows: [f"{k}={v}" for k, v in rows])
        out = kv(res.rows("vol"))
        out.append(f"vol.applied={'%.2f' % target if target is not None else 'no (restored %s)' % st0.volume}")
        return out + self._mixer_rows_kv()

    # -- two-way acoustic ranging (engine/ranging.py), off the Qt thread -------------------------------------------

    _RANGE_USAGE = ("RANGE RESPOND [secs] | INIT [secs] | SOLVE rtt=<s> turn=<s> [ppm=] [speed=343] [bias=m]  "
                    "[band=lo-hi] [sym=0.3] [amp=0..1] [margin=1] [ppm=] [save=file.wav]  (modem off; RESPOND on one "
                    "host, then INIT on the other, both with the same band= sym=; SOLVE takes INIT's range.rtt_s and "
                    "RESPOND's range.turnaround_s)")

    def _range(self, a: str):
        usage = self._RANGE_USAGE
        kw = dict(w.split("=", 1) for w in a.split() if "=" in w)
        words = [w for w in a.split() if "=" not in w]
        op = words[0].upper() if words else ""
        if op not in ("RESPOND", "INIT", "SOLVE") or len(words) > 2:
            raise ControlError(usage)

        def num(key, default=None):
            if key not in kw:
                if default is None:
                    raise ControlError(f"{key}= is required  ({usage})")
                return default
            try:
                return float(kw[key])
            except ValueError:
                raise ControlError(f"bad {key}={kw[key]!r}") from None

        if op == "SOLVE":
            if set(kw) - {"rtt", "turn", "ppm", "speed", "bias"}:
                raise ControlError(usage)
            d = ranging.distance_m(num("rtt"), num("turn"), speed=num("speed", ranging.SPEED_OF_SOUND_M_S),
                                   ppm=num("ppm", 0.0), bias_m=num("bias", 0.0))
            rows = [f"range.distance_m={d:.3f}", f"range.flight_ms={(num('rtt') - num('turn')) / 2 * 1000:.3f}"]
            if d < 0:
                rows.append("range.error=negative distance: a burst was mis-timed (check self_db / peer_db)")
            return rows
        if set(kw) - {"band", "sym", "amp", "margin", "ppm", "save"}:
            raise ControlError(usage)
        try:
            secs = float(words[1]) if len(words) > 1 else None
        except ValueError:
            raise ControlError(usage) from None

        def grab():
            w = self.w
            if w.session.pipe is None:
                raise ControlError("audio is not open (AUDIO OPEN <rx> <tx>)")
            if w.modem_on:
                raise ControlError("modem is on (POWER OFF first: the link would transmit over the burst)")
            c = w.app.css
            return w.session.pipe, (c.f_min, c.f_max), c.amplitude
        pipe, band, amp = self.run(grab)
        lo, hi = self._band_arg(kw["band"]) if "band" in kw else band
        p = ranging.BurstParams(lo, hi, num("sym", 0.3), num("amp", amp))
        try:
            p.check()
        except ValueError as e:
            raise ControlError(str(e)) from None
        ppm = num("ppm", 0.0)
        try:
            if op == "RESPOND":
                leg = ranging.respond(pipe, p, timeout=min(max(secs or 60.0, 1.0), 600.0),
                                      margin=min(max(num("margin", 1.0), 0.2), 10.0), ppm=ppm)
            else:
                room = 27.0 - len(ranging.burst(pipe.spec.rate, p)) / pipe.spec.rate     # the pipe keeps 30 s
                leg = ranging.initiate(pipe, p, wait=min(max(secs or 20.0, 2.0), max(room, 2.0)), ppm=ppm)
        except ranging.RangingError as e:
            return ["range.ok=no", f"range.error={e}"]
        rows = [f"{k}={v}" for k, v in leg.rows("range")]
        rows += [f"range.band_hz={p.f_min:g}-{p.f_max:g}", f"range.symbol_dur={p.symbol_dur:g}",
                 f"range.amplitude={p.amplitude:g}", f"range.burst_s={len(ranging.preamble(leg.fs, p)) / leg.fs:.2f}"]
        if "save" in kw:
            try:
                ranging.save_wav(kw["save"], leg.capture, leg.fs)
                rows.append(f"range.saved={kw['save']}")
            except OSError as e:
                rows.append(f"range.save_error={e}")
        return rows

    # -- the last seconds of received audio, for debugging a remote host ------------------------------------------

    def _capture(self, a: str):
        """``CAPTURE [secs]``: the newest RX audio as a 16-bit mono WAV, base64 on one line (the pipe keeps 30 s).
        Works with the modem on: it is how a failed handshake on a host without file access gets looked at."""
        words = a.split()
        try:
            secs = float(words[0]) if words else 20.0
        except ValueError:
            raise ControlError("CAPTURE [secs]  (1-29, default 20)") from None
        secs = min(max(secs, 1.0), 29.0)

        def grab():
            if self.w.session.pipe is None:
                raise ControlError("audio is not open (AUDIO OPEN <rx> <tx>)")
            return self.w.session.pipe
        pipe = self.run(grab)
        fs = pipe.spec.rate
        x, first = pipe.history(int(secs * fs))
        if x.dtype.kind == "c":
            raise ControlError("CAPTURE takes real audio, not an iq: device")
        buf = io.BytesIO()
        ranging.save_wav(buf, x, fs)
        return [f"capture.fs={fs}", f"capture.samples={len(x)}", f"capture.first_index={first}",
                f"capture.end_time={time.time():.3f}",
                f"capture.peak_dbfs={20 * np.log10(max(float(np.abs(x).max()) if len(x) else 0.0, 1e-9)):.1f}",
                "capture.wav_b64=" + base64.b64encode(buf.getvalue()).decode("ascii")]

    def _mixer_rows_kv(self) -> list[str]:
        """The mixer state the measurement was taken at, so a result can be reproduced."""
        try:
            return [f"{k}={v}" for k, v in self._mixer_rows() if k.endswith((".volume", ".volume_db", ".device"))]
        except Exception as e:
            return [f"audio.host.error={e}"]

    # -- actions (what the buttons do) -----------------------------------------------

    def actions(self) -> dict:
        return {
            "POWER": ("ON|OFF  start / stop the modem (the Start modem button)", self._power),
            "RESTART": ("  power-cycle the modem so pending settings apply (reopens the serial port if it was open)",
                        self._restart),
            "AUDIO": ("OPEN <rx> <tx> | CLOSE  open audio devices (addresses as on the command line: pulse:, sd:N, "
                      "loop:x, file:x.wav, null:)", self._audio),
            "DEVICES": ("  audio devices this host can open", self._devices),
            "CALIBRATE": ("NOISE [secs] | SELF | LISTEN [secs] | SEND  measure the acoustic channel: noise floor, best "
                          "TX level, input gain change, AGC, echo / clock offset and the OFDM cp_ratio / n_fft it "
                          "wants (modem off) | VOLUME [from= to= steps= level= apply=]  sweep the OS output volume and "
                          "recommend the system volume", direct(self._calibrate)),
            "RANGE": ("RESPOND [secs] | INIT [secs] | SOLVE rtt= turn=  two-way acoustic ranging with a CSS Barker "
                      "burst: RESPOND arms one host, INIT on the other times the round trip; SOLVE turns the two "
                      "into metres (modem off)", direct(self._range)),
            "CAPTURE": ("[secs]  the last secs (default 20, max 29) of received audio as a base64 16-bit WAV on one "
                        "line (capture.wav_b64=); works with the modem on", direct(self._capture)),
            "MIXER": ("[GET] | SET in|out <0..1|N%> | MUTE in|out on|off | SNAPSHOT | RESTORE  the host OS volume / "
                      "mute / AGC of the open audio devices (CoreAudio, Pulse, ALSA)", direct(self._mixer)),
            "PORT": ("OPEN|CLOSE  the virtual serial port (Ctrl+P)", self._port),
            "IDENT": ("  send the station ID now", lambda a: self._after(self.w._send_id_now)),
            "TEXT": ("<message>  send one message on the mesh link (not while the serial port owns it)",
                     lambda a: self._after(lambda: self.w.send_text(a)) if a else self._need("TEXT <message>")),
            "SAVE": ("  write the current settings to disk (as the settings dialog's OK does)", self._save),
            "PROFILE": ("LIST | SAVE <name> | LOAD <name> | SHOW <name> | DELETE <name>  named snapshots of every "
                        "setting plus the OS speaker / mic volume (e.g. a calibration for one placement: \"1ft apart\"); "
                        "LOAD applies, SAVEs to disk and sets the mixer when audio is open", direct(self._profile)),
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
                # native rate: the modem streams at session.rate, so a 44.1 kHz device is resampled by the OS
                rows.append(f"{kind} {d.address}  {d.name}" + (f"  {d.rate_hint} Hz" if d.rate_hint else "")
                            + ("  (default)" if d.default else ""))
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

    # -- named setting profiles ---------------------------------------------------------------------------------

    def _profile_dir(self) -> str:
        d = os.path.join(os.path.dirname(os.path.abspath(self.w.app.qs.fileName())), "ne0k-profiles")
        os.makedirs(d, exist_ok=True)
        return d

    def _profile_path(self, name: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_")
        if not slug:
            raise ControlError("profile name needed")
        return os.path.join(self._profile_dir(), slug + ".json")

    def _profile(self, a: str):
        usage = "PROFILE LIST | SAVE <name> | LOAD <name> | SHOW <name> | DELETE <name>"
        op, _, name = a.strip().partition(" ")
        op, name = op.upper(), name.strip()
        if op == "LIST":
            rows = []
            for fn in sorted(os.listdir(self._profile_dir())):
                if fn.endswith(".json"):
                    try:
                        with open(os.path.join(self._profile_dir(), fn), encoding="utf-8") as f:
                            d = json.load(f)
                        rows.append(f"profile={d.get('name', fn[:-5])} saved={d.get('saved', '?')} "
                                    f"out={d.get('mixer', {}).get('out', {}).get('volume')} "
                                    f"in={d.get('mixer', {}).get('in', {}).get('volume')}")
                    except (OSError, ValueError) as e:
                        rows.append(f"profile={fn[:-5]} unreadable: {e}")
            return rows or ["no profiles"]
        if op not in ("SAVE", "LOAD", "SHOW", "DELETE") or not name:
            raise ControlError(usage)
        path = self._profile_path(name)
        if op == "SAVE":
            mixer = {}
            for side, (addr, dev) in self._mixer_targets().items():
                st = hostmixer.get(addr, self._MIX_KIND[side], dev)
                if st.volume is not None:
                    mixer[side] = {"device": st.device, "volume": round(st.volume, 3), "volume_db": st.volume_db}
            d = {"name": name, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "mixer": mixer,
                 "settings": self.run(self.w.app.dump)}
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(d, f, indent=1)
            os.replace(path + ".tmp", path)
            if not mixer:
                return [f"profile saved {path}", "mixer not stored (audio closed: AUDIO OPEN first to keep the volumes)"]
            return [f"profile saved {path}"] + [f"profile.mixer.{k}.volume={v['volume']}" for k, v in mixer.items()]
        if not os.path.exists(path):
            raise ControlError(f"no profile {name!r} (PROFILE LIST)")
        if op == "DELETE":
            os.remove(path)
            return f"profile deleted {name}"
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if op == "SHOW":
            rows = [f"profile.name={d.get('name')}", f"profile.saved={d.get('saved')}"]
            rows += [f"profile.mixer.{k}.{kk}={vv}" for k, v in d.get("mixer", {}).items() for kk, vv in v.items()]
            rows += [f"{sec}.{k}={v}" for sec, kv in d.get("settings", {}).items() for k, v in kv.items()
                     if sec in ("ofdm", "css", "output", "general")]
            return rows

        def load():
            w = self.w
            w.app.load_dict(d.get("settings", {}))
            w._after_settings()
            w.app.save()
            return w.modem_on
        on = self.run(load)
        rows = [f"profile loaded {d.get('name')} (settings saved{'; RESTART to apply them to the running modem' if on else ''})"]
        targets = self._mixer_targets()
        for side, m in d.get("mixer", {}).items():
            if side not in targets:
                rows.append(f"profile.mixer.{side} not applied (audio closed)")
                continue
            addr, dev = targets[side]
            try:
                st = hostmixer.apply(addr, self._MIX_KIND[side], dev, volume=float(m["volume"]))
                rows.append(f"profile.mixer.{side}.volume={st.volume}")
            except (hostmixer.MixerError, KeyError, ValueError) as e:
                rows.append(f"profile.mixer.{side} failed: {e}")
        return rows

    def _save(self, a: str) -> str:
        self.w.app.save()
        return "settings saved"
