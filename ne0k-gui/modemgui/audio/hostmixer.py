# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""The host OS mixer for the audio device the modem has open: read and set input / output volume and mute, and
report what the OS says about automatic gain control.

Why: on a real speaker + mic link the OS gain stages (mic input volume, speaker volume, boost, an AGC in an echo
canceller) set the SNR as much as the modem's own ``amplitude`` does, and a test framework cannot ask a person to
move sliders.  This module lets the control port snapshot, force and restore them.

Backends, picked from the modem's device address:
  * ``pulse:<name>``      -- pactl (PulseAudio / PipeWire-pulse).  ALSA boost / AGC switches on the card beneath
                             are listed as notes.
  * ``sd:`` on macOS      -- CoreAudio via ctypes, on the device whose name PortAudio reports.
  * ``sd:`` on Linux      -- the PortAudio "pulse"/"default"/"pipewire" devices map to the Pulse default; a
                             ``(hw:N,M)`` device maps to ``amixer -c N``.
  * anything else         -- (loop:, file:, rtp:, null:, iq:, Windows) backend "none": nothing to read or set.

Volumes are 0..1 of the OS slider (Pulse may report above 1 for software boost).  Nothing here raises for a
missing tool or an unsupported OS: the state says ``backend="none"`` / ``volume=None`` and why in ``notes``.

AGC: ``agc`` is what the OS *reports*, never a measurement -- "on", "off" or "unknown".  macOS has no HAL property
for it (Voice Isolation / Mic Modes act only on VoiceProcessingIO streams, which PortAudio does not open; a USB
headset can still have AGC in its own hardware), so a mac always says "unknown"; the calibration step measures it.
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
import sys

from .base import SINK, SOURCE
from .sysenv import host_env


@dataclasses.dataclass
class MixerState:
    kind: str                         # SOURCE | SINK
    backend: str                      # "coreaudio" | "pulse" | "alsa" | "none"
    device: str = ""                  # the OS object the numbers belong to
    volume: float | None = None       # 0..1 slider position (Pulse: may exceed 1)
    volume_db: float | None = None
    mute: bool | None = None
    settable: bool = False
    agc: str = "unknown"              # what the OS reports: on | off | unknown
    notes: list[str] = dataclasses.field(default_factory=list)

    def rows(self, prefix: str) -> list[tuple[str, str]]:
        """``key=value`` rows for the control port's INFO / MIXER GET."""
        f = (lambda v, fmt: "unknown" if v is None else fmt % v)
        rows = [(f"{prefix}.backend", self.backend)]
        if self.backend != "none":
            rows += [(f"{prefix}.device", self.device or "default"),
                     (f"{prefix}.volume", f(self.volume, "%.2f")),
                     (f"{prefix}.volume_db", f(self.volume_db, "%.1f")),
                     (f"{prefix}.mute", "unknown" if self.mute is None else ("yes" if self.mute else "no")),
                     (f"{prefix}.settable", "yes" if self.settable else "no")]
            if self.kind == SOURCE:
                rows.append((f"{prefix}.agc", self.agc))
        if self.notes:
            # one line per row: pactl / amixer errors carry newlines, which would end the control-port reply early
            rows.append((f"{prefix}.notes", " ".join("; ".join(self.notes).split())))
        return rows


class MixerError(RuntimeError):
    """A set was asked of a device whose volume cannot be set."""


# -- shared helpers -----------------------------------------------------------------------------------------------------

def _run(argv: list[str], timeout: float = 4.0) -> str:
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=host_env())
    if r.returncode != 0:
        raise MixerError(f"{' '.join(argv)}: {(r.stderr or r.stdout).strip() or r.returncode}")
    return r.stdout


def _clamp(v: float, hi: float = 1.0) -> float:
    return max(0.0, min(hi, float(v)))


# -- Pulse --------------------------------------------------------------------------------------------------------------

_PCT = re.compile(r"(\d+)%")
_DB = re.compile(r"(-?[\d.]+|-inf) dB")


def parse_pactl_volume(text: str) -> tuple[float | None, float | None]:
    """``Volume: front-left: 49152 /  75% / -7.50 dB,   front-right: ...`` -> (0.75, -7.5): channel mean."""
    pcts = [int(p) for p in _PCT.findall(text)]
    dbs = [float(d) for d in _DB.findall(text) if d != "-inf"]
    vol = sum(pcts) / len(pcts) / 100.0 if pcts else None
    db = sum(dbs) / len(dbs) if dbs else (None if not pcts else float("-inf"))
    return vol, db


def parse_pactl_mute(text: str) -> bool | None:
    m = re.search(r"Mute:\s*(yes|no)", text)
    return None if m is None else m.group(1) == "yes"


def pulse_agc(source: str, modules_short: str, sources_json: str | None) -> tuple[str, list[str]]:
    """'on' if ``source`` is an echo-cancel source (WebRTC AEC carries AGC), else 'off'; notes say what was seen."""
    notes = []
    ec = [ln for ln in modules_short.splitlines() if "echo-cancel" in ln]
    is_ec = "echo-cancel" in source or "echo_cancel" in source
    if sources_json and not is_ec:
        try:
            for o in json.loads(sources_json):
                if o.get("name") == source:
                    drv = str(o.get("driver", "")) + " " + str(o.get("properties", {}).get("media.class", ""))
                    is_ec = "echo-cancel" in drv or "echo_cancel" in drv
        except (ValueError, AttributeError):
            pass
    if is_ec:
        args = " ".join(ec)
        notes.append("source is an echo-cancel source" + ("" if "agc" not in args.lower() else f" ({args.strip()})"))
        if re.search(r"(analog_gain_control|digital_gain_control|agc)\s*=\s*(0|false|no)", args, re.I):
            return "off", notes
        return "on", notes
    if ec:
        notes.append("module-echo-cancel is loaded but this is not its source")
    return "off", notes


def parse_amixer_scontrols(text: str) -> list[str]:
    """``Simple mixer control 'Mic Boost',0`` lines -> names."""
    return re.findall(r"Simple mixer control '([^']+)'", text)


_OUT_CTLS = ("master", "speaker", "pcm", "headphone")


def alsa_wanted(name: str, kind: str) -> bool:
    """The ALSA controls worth noting under a Pulse device: capture gain / boost / AGC for a source, the playback
    stages (Master, Speaker, PCM, Headphone) for a sink."""
    low = name.lower()
    if kind == SOURCE:
        return "boost" in low or "auto gain" in low or "agc" in low or low == "capture"
    return low in _OUT_CTLS


def alsa_gain_notes(card: str | int, kind: str = SOURCE) -> tuple[str | None, list[str]]:
    """The ``kind`` side's gain controls on an ALSA card (alsa_wanted), as notes; agc 'on'/'off' if a source card has
    an AGC switch, else None."""
    if not shutil.which("amixer"):
        return None, []
    try:
        names = parse_amixer_scontrols(_run(["amixer", "-c", str(card), "scontrols"]))
    except Exception:
        return None, []
    notes, agc = [], None
    for n in names:
        low = n.lower()
        if alsa_wanted(n, kind):
            try:
                out = _run(["amixer", "-c", str(card), "sget", n])
            except Exception:
                continue
            val = _summarise_amixer(out)
            notes.append(f"alsa card {card} '{n}' {val}")
            if kind == SOURCE and ("auto gain" in low or "agc" in low):
                agc = "on" if "[on]" in out else "off"
    return agc, notes


def _summarise_amixer(out: str) -> str:
    pct = _PCT.findall(out)
    db = re.findall(r"\[(-?[\d.]+dB)\]", out)
    sw = re.findall(r"\[(on|off)\]", out)
    parts = []
    if pct:
        parts.append(f"{pct[0]}%")
    if db:
        parts.append(db[0])
    if sw:
        parts.append(sw[0])
    return " ".join(parts) or "present"


def _pulse_get(kind: str, name: str) -> MixerState:
    noun = "source" if kind == SOURCE else "sink"
    obj = name or ("@DEFAULT_SOURCE@" if kind == SOURCE else "@DEFAULT_SINK@")
    st = MixerState(kind, "pulse", device=name or f"default {noun}")
    if not shutil.which("pactl"):
        return MixerState(kind, "none", notes=["pactl not installed"])
    try:
        st.volume, st.volume_db = parse_pactl_volume(_run(["pactl", f"get-{noun}-volume", obj]))
        st.mute = parse_pactl_mute(_run(["pactl", f"get-{noun}-mute", obj]))
        st.settable = st.volume is not None
    except Exception as e:
        st.notes.append(str(e))
        return st
    sources_json = None
    card = None
    try:
        listing = _run(["pactl", "--format=json", "list", noun + "s"])
        for o in json.loads(listing):
            if o.get("name") == name or (not name and o.get("name") == _pulse_default(noun)):
                st.device = o.get("name", st.device)
                card = o.get("properties", {}).get("alsa.card")
        sources_json = listing if kind == SOURCE else None
    except Exception:
        pass
    if kind == SOURCE:
        try:
            mods = _run(["pactl", "list", "modules", "short"])
        except Exception:
            mods = ""
        st.agc, notes = pulse_agc(st.device, mods, sources_json)
        st.notes += notes
    if card is not None:
        agc, notes = alsa_gain_notes(card, kind)
        st.notes += notes
        if kind == SOURCE and agc == "on":
            st.agc = "on"
    return st


def _pulse_default(noun: str) -> str | None:
    try:
        return _run(["pactl", f"get-default-{noun}"]).strip() or None
    except Exception:
        return None


def _pulse_set(kind: str, name: str, volume: float | None, mute: bool | None) -> None:
    noun = "source" if kind == SOURCE else "sink"
    obj = name or ("@DEFAULT_SOURCE@" if kind == SOURCE else "@DEFAULT_SINK@")
    if volume is not None:
        _run(["pactl", f"set-{noun}-volume", obj, f"{round(_clamp(volume, 1.5) * 100)}%"])
    if mute is not None:
        _run(["pactl", f"set-{noun}-mute", obj, "1" if mute else "0"])


# -- ALSA (amixer) ------------------------------------------------------------------------------------------------------

_ALSA_CTL = {SOURCE: ("Capture",), SINK: ("Master", "PCM", "Speaker")}


def _alsa_ctl(card: str, kind: str) -> str | None:
    try:
        names = parse_amixer_scontrols(_run(["amixer", "-c", card, "scontrols"]))
    except Exception:
        return None
    for want in _ALSA_CTL[kind]:
        if want in names:
            return want
    return None


def _alsa_get(kind: str, card: str) -> MixerState:
    st = MixerState(kind, "alsa", device=f"hw:{card}")
    if not shutil.which("amixer"):
        return MixerState(kind, "none", notes=["amixer not installed"])
    ctl = _alsa_ctl(card, kind)
    if ctl is None:
        st.notes.append(f"card {card} has no {'/'.join(_ALSA_CTL[kind])} control")
        return st
    out = _run(["amixer", "-c", card, "sget", ctl])
    st.device = f"hw:{card} '{ctl}'"
    pct = [int(p) for p in _PCT.findall(out)]
    db = [float(d[:-2]) for d in re.findall(r"\[(-?[\d.]+dB)\]", out)]
    sw = re.findall(r"\[(on|off)\]", out)
    st.volume = sum(pct) / len(pct) / 100.0 if pct else None
    st.volume_db = sum(db) / len(db) if db else None
    st.mute = (sw[0] == "off") if sw else None      # capture/playback switch off = muted
    st.settable = st.volume is not None
    if kind == SOURCE:
        agc, notes = alsa_gain_notes(card)
        st.agc = agc or "off"
        st.notes += [n for n in notes if f"'{ctl}'" not in n]
    return st


def _alsa_set(kind: str, card: str, volume: float | None, mute: bool | None) -> None:
    ctl = _alsa_ctl(card, kind)
    if ctl is None:
        raise MixerError(f"card {card} has no {'/'.join(_ALSA_CTL[kind])} control")
    if volume is not None:
        _run(["amixer", "-c", card, "sset", ctl, f"{round(_clamp(volume) * 100)}%"])
    if mute is not None:
        _run(["amixer", "-c", card, "sset", ctl, "nocap" if kind == SOURCE and mute else
              "cap" if kind == SOURCE else "mute" if mute else "unmute"])


# -- CoreAudio (macOS) ------------------------------------------------------------------------------------------------

def _fourcc(s: str) -> int:
    return int.from_bytes(s.encode("ascii"), "big")


class _CoreAudio:
    """The few AudioObject calls needed, via ctypes.  Built lazily: importing this module on Linux loads nothing."""

    SYSTEM = 1
    GLOBAL, INPUT, OUTPUT = _fourcc("glob"), _fourcc("inpt"), _fourcc("outp")
    DEVICES, NAME = _fourcc("dev#"), _fourcc("lnam")
    DEF_IN, DEF_OUT = _fourcc("dIn "), _fourcc("dOut")
    VOLUME, VOLUME_DB, MUTE = _fourcc("volm"), _fourcc("vold"), _fourcc("mute")
    VIRTUAL_MAIN = _fourcc("vmvc")          # kAudioHardwareServiceDeviceProperty_VirtualMainVolume
    DATA_SOURCE, DS_NAME = _fourcc("ssrc"), _fourcc("lscn")   # kAudioDevicePropertyDataSource / ...NameForIDCFString
    STREAM_CFG = _fourcc("slay")            # kAudioDevicePropertyStreamConfiguration

    def __init__(self) -> None:
        import ctypes
        import ctypes.util
        self.c = ctypes
        self.ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
        self.cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

        class Addr(ctypes.Structure):
            _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]
        self.Addr = Addr
        u32, p = ctypes.c_uint32, ctypes.POINTER
        ca = self.ca
        ca.AudioObjectHasProperty.argtypes = [u32, p(Addr)]
        ca.AudioObjectHasProperty.restype = ctypes.c_ubyte
        ca.AudioObjectIsPropertySettable.argtypes = [u32, p(Addr), p(ctypes.c_ubyte)]
        ca.AudioObjectIsPropertySettable.restype = ctypes.c_int32
        ca.AudioObjectGetPropertyDataSize.argtypes = [u32, p(Addr), u32, ctypes.c_void_p, p(u32)]
        ca.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
        ca.AudioObjectGetPropertyData.argtypes = [u32, p(Addr), u32, ctypes.c_void_p, p(u32), ctypes.c_void_p]
        ca.AudioObjectGetPropertyData.restype = ctypes.c_int32
        ca.AudioObjectSetPropertyData.argtypes = [u32, p(Addr), u32, ctypes.c_void_p, u32, ctypes.c_void_p]
        ca.AudioObjectSetPropertyData.restype = ctypes.c_int32
        self.cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, u32]
        self.cf.CFStringGetCString.restype = ctypes.c_ubyte
        self.cf.CFRelease.argtypes = [ctypes.c_void_p]

    def _a(self, sel: int, scope: int, elem: int = 0):
        return self.Addr(sel, scope, elem)

    def has(self, dev: int, sel: int, scope: int, elem: int = 0) -> bool:
        a = self._a(sel, scope, elem)
        return bool(self.ca.AudioObjectHasProperty(dev, self.c.byref(a)))

    def settable(self, dev: int, sel: int, scope: int, elem: int = 0) -> bool:
        a = self._a(sel, scope, elem)
        out = self.c.c_ubyte(0)
        return self.ca.AudioObjectIsPropertySettable(dev, self.c.byref(a), self.c.byref(out)) == 0 and bool(out.value)

    def get(self, dev: int, sel: int, scope: int, ctype, elem: int = 0, qual=None):
        a = self._a(sel, scope, elem)
        v = ctype()
        size = self.c.c_uint32(self.c.sizeof(v))
        qs, qp = (0, None) if qual is None else (self.c.sizeof(qual), self.c.byref(qual))
        err = self.ca.AudioObjectGetPropertyData(dev, self.c.byref(a), qs, qp, self.c.byref(size), self.c.byref(v))
        if err:
            raise MixerError(f"CoreAudio get {sel:#x}: OSStatus {err}")
        return v.value

    def set(self, dev: int, sel: int, scope: int, ctype, value, elem: int = 0) -> None:
        a = self._a(sel, scope, elem)
        v = ctype(value)
        err = self.ca.AudioObjectSetPropertyData(dev, self.c.byref(a), 0, None, self.c.sizeof(v), self.c.byref(v))
        if err:
            raise MixerError(f"CoreAudio set {sel:#x}: OSStatus {err}")

    def get_bytes(self, dev: int, sel: int, scope: int) -> bytes:
        a = self._a(sel, scope)
        size = self.c.c_uint32(0)
        if self.ca.AudioObjectGetPropertyDataSize(dev, self.c.byref(a), 0, None, self.c.byref(size)):
            return b""
        buf = self.c.create_string_buffer(size.value)
        if self.ca.AudioObjectGetPropertyData(dev, self.c.byref(a), 0, None, self.c.byref(size), buf):
            return b""
        return buf.raw[:size.value]

    def cfstr(self, ref) -> str:
        if not ref:
            return ""
        buf = self.c.create_string_buffer(512)
        ok = self.cf.CFStringGetCString(ref, buf, 512, 0x08000100)   # kCFStringEncodingUTF8
        self.cf.CFRelease(ref)
        return buf.value.decode("utf-8", "replace") if ok else ""

    def devices(self) -> list[int]:
        raw = self.get_bytes(self.SYSTEM, self.DEVICES, self.GLOBAL)
        return [int.from_bytes(raw[i:i + 4], sys.byteorder) for i in range(0, len(raw), 4)]

    def name(self, dev: int) -> str:
        return self.cfstr(self.get(dev, self.NAME, self.GLOBAL, self.c.c_void_p))

    def channels(self, dev: int, scope: int) -> int:
        """Channel count in ``scope`` from the AudioBufferList of kAudioDevicePropertyStreamConfiguration."""
        raw = self.get_bytes(dev, self.STREAM_CFG, scope)
        if len(raw) < 4:
            return 0
        n = int.from_bytes(raw[:4], sys.byteorder)
        # AudioBuffer {UInt32 mNumberChannels; UInt32 mDataByteSize; void* mData} after the 4-byte count, 8-aligned.
        off, step, total = 8 if self.c.sizeof(self.c.c_void_p) == 8 else 4, 8 + self.c.sizeof(self.c.c_void_p), 0
        for i in range(n):
            o = off + i * step
            total += int.from_bytes(raw[o:o + 4], sys.byteorder)
        return total

    def find(self, name: str, scope: int) -> int | None:
        default = self.get(self.SYSTEM, self.DEF_IN if scope == self.INPUT else self.DEF_OUT, self.GLOBAL,
                           self.c.c_uint32)
        if not name:
            return default
        want = name.lower()
        hits = [d for d in self.devices() if self.name(d).lower() == want and self.channels(d, scope) > 0]
        if not hits:
            hits = [d for d in self.devices() if want in self.name(d).lower() and self.channels(d, scope) > 0]
        return hits[0] if hits else None

    def volume_elems(self, dev: int, scope: int) -> list[int]:
        """Elements that carry a volume: the main element 0 if present, else the per-channel ones."""
        if self.has(dev, self.VOLUME, scope, 0):
            return [0]
        return [e for e in range(1, self.channels(dev, scope) + 1) if self.has(dev, self.VOLUME, scope, e)]


_CA: _CoreAudio | None = None


def _ca() -> _CoreAudio:
    global _CA
    if _CA is None:
        _CA = _CoreAudio()
    return _CA


def _ca_get(kind: str, name: str) -> MixerState:
    try:
        ca = _ca()
    except Exception as e:
        return MixerState(kind, "none", notes=[f"CoreAudio unavailable: {e}"])
    c = ca.c
    scope = ca.INPUT if kind == SOURCE else ca.OUTPUT
    dev = ca.find(name, scope)
    if dev is None:
        return MixerState(kind, "none", notes=[f"no CoreAudio device named {name!r}"])
    st = MixerState(kind, "coreaudio", device=ca.name(dev))
    elems = ca.volume_elems(dev, scope)
    if elems:
        vols = [ca.get(dev, ca.VOLUME, scope, c.c_float, e) for e in elems]
        st.volume = sum(vols) / len(vols)
        st.settable = any(ca.settable(dev, ca.VOLUME, scope, e) for e in elems)
        dbs = [ca.get(dev, ca.VOLUME_DB, scope, c.c_float, e) for e in elems if ca.has(dev, ca.VOLUME_DB, scope, e)]
        st.volume_db = sum(dbs) / len(dbs) if dbs else None
        if len(elems) > 1:
            st.notes.append(f"per-channel volume (elements {elems})")
    elif ca.has(dev, ca.VIRTUAL_MAIN, scope):
        st.volume = ca.get(dev, ca.VIRTUAL_MAIN, scope, c.c_float)
        st.settable = ca.settable(dev, ca.VIRTUAL_MAIN, scope)
        st.notes.append("virtual main volume")
    else:
        st.notes.append("device has no volume control (fixed gain)")
    if ca.has(dev, ca.MUTE, scope):
        st.mute = bool(ca.get(dev, ca.MUTE, scope, c.c_uint32))
    if ca.has(dev, ca.DATA_SOURCE, scope):
        try:
            src = c.c_uint32(ca.get(dev, ca.DATA_SOURCE, scope, c.c_uint32))
            st.notes.append(f"data source {_ca_ds_name(ca, dev, scope, src) or src.value}")
        except Exception:
            pass
    if kind == SOURCE:
        st.agc = "unknown"
        st.notes.append("macOS reports no AGC state; measure it")
    return st


def _ca_ds_name(ca: _CoreAudio, dev: int, scope: int, src) -> str:
    """kAudioDevicePropertyDataSourceNameForIDCFString takes an AudioValueTranslation (in: UInt32, out: CFStringRef)."""
    c = ca.c

    class AVT(c.Structure):
        _fields_ = [("inp", c.c_void_p), ("inSize", c.c_uint32), ("out", c.c_void_p), ("outSize", c.c_uint32)]
    ref = c.c_void_p()
    t = AVT(c.cast(c.pointer(src), c.c_void_p), 4, c.cast(c.pointer(ref), c.c_void_p), c.sizeof(ref))
    a = ca._a(ca.DS_NAME, scope)
    size = c.c_uint32(c.sizeof(t))
    if ca.ca.AudioObjectGetPropertyData(dev, c.byref(a), 0, None, c.byref(size), c.byref(t)):
        return ""
    return ca.cfstr(ref.value)


def _ca_set(kind: str, name: str, volume: float | None, mute: bool | None) -> None:
    ca = _ca()
    c = ca.c
    scope = ca.INPUT if kind == SOURCE else ca.OUTPUT
    dev = ca.find(name, scope)
    if dev is None:
        raise MixerError(f"no CoreAudio device named {name!r}")
    if volume is not None:
        elems = ca.volume_elems(dev, scope)
        if elems:
            for e in elems:
                ca.set(dev, ca.VOLUME, scope, c.c_float, _clamp(volume), e)
        elif ca.has(dev, ca.VIRTUAL_MAIN, scope):
            ca.set(dev, ca.VIRTUAL_MAIN, scope, c.c_float, _clamp(volume))
        else:
            raise MixerError(f"{ca.name(dev)}: no settable volume")
    if mute is not None:
        if not ca.has(dev, ca.MUTE, scope):
            raise MixerError(f"{ca.name(dev)}: no mute control")
        ca.set(dev, ca.MUTE, scope, c.c_uint32, 1 if mute else 0)


# -- address -> backend ---------------------------------------------------------------------------------------------

_HW = re.compile(r"\(hw:(\d+),\d+\)")


def route(address: str | None, kind: str, device_name: str = "", platform: str | None = None) -> tuple[str, str]:
    """(backend, target) for a modem device address.  ``device_name`` is the name the connector lists (PortAudio's
    device name for ``sd:``).  Pure: no OS calls, so it is unit-testable everywhere."""
    platform = platform or sys.platform
    if not address:
        return "none", "no device open"
    conn, _, dev = address.partition(":")
    if conn == "pulse":
        return "pulse", dev
    if conn == "sd":
        if platform == "darwin":
            return "coreaudio", device_name
        if platform.startswith("linux"):
            low = device_name.lower()
            if low in ("pulse", "default", "pipewire", "sysdefault") or dev == "":
                return "pulse", ""
            m = _HW.search(device_name)
            if m:
                return "alsa", m.group(1)
            return "none", f"PortAudio device {device_name!r} has no known mixer"
        return "none", f"no mixer support on {platform} yet"
    return "none", f"{conn}: is not a hardware device"


def get(address: str | None, kind: str, device_name: str = "") -> MixerState:
    backend, target = route(address, kind, device_name)
    try:
        if backend == "pulse":
            return _pulse_get(kind, target)
        if backend == "alsa":
            return _alsa_get(kind, target)
        if backend == "coreaudio":
            return _ca_get(kind, target)
    except Exception as e:
        return MixerState(kind, backend, device=target, notes=[f"{type(e).__name__}: {e}"])
    return MixerState(kind, "none", notes=[target])


def apply(address: str | None, kind: str, device_name: str = "", volume: float | None = None,
        mute: bool | None = None) -> MixerState:
    """Set volume (0..1) and/or mute, then read back.  Raises MixerError when the device cannot be set."""
    backend, target = route(address, kind, device_name)
    if backend == "pulse":
        _pulse_set(kind, target, volume, mute)
    elif backend == "alsa":
        _alsa_set(kind, target, volume, mute)
    elif backend == "coreaudio":
        _ca_set(kind, target, volume, mute)
    else:
        raise MixerError(f"cannot set the mixer: {target}")
    return get(address, kind, device_name)
