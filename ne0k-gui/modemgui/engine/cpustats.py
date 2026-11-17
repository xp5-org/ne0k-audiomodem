# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Where the process's CPU goes: per-thread CPU time, plus the modem's DSP stages.

Two measurements, combined by ``CpuMeter``:

* **Stages** -- ``with stage("decode"):`` / ``stage("encode")`` / ``stage("pll")`` around the DSP calls.  It adds the
  calling thread's own CPU time (``time.thread_time``) spent inside, so time spent waiting for the DSP lock or the GIL
  is not counted.  Nested stages count once, for the outermost.  Encode runs on whichever thread sends (the MAC, a
  link, the GUI's send worker), so only a stage timer can separate it from that thread's other work.
* **Threads** -- CPU time of every OS thread in the process, read from the OS: ``/proc/self/task`` on Linux (every
  thread, named by Python name or the kernel's ``comm``), ``task_threads`` + ``thread_info`` on macOS (every thread,
  named by Python name or ``pthread_getname_np`` -- CoreAudio's IO thread included), ``GetThreadTimes`` on Windows
  (Python threads; the rest shows as the remainder).

``CpuMeter.sample()`` returns CPU % per group over the interval since the last call (100 % = one core): the stage
time is taken out of the thread that did it, and each thread's remainder goes to its group (``group_of``).
"""
from __future__ import annotations

import collections
import os
import sys
import threading
import time
from contextlib import contextmanager

# -- stage timers ----------------------------------------------------------------------------------------------------

STAGES = ("decode", "encode", "pll", "dsp")
_lock = threading.Lock()
_stage_cpu: dict[tuple[str, int], float] = collections.defaultdict(float)   # (stage, native thread id) -> seconds
_tls = threading.local()


@contextmanager
def stage(name: str):
    if getattr(_tls, "depth", 0):
        _tls.depth += 1
        try:
            yield
        finally:
            _tls.depth -= 1
        return
    _tls.depth = 1
    t0 = time.thread_time()
    try:
        yield
    finally:
        dt = time.thread_time() - t0
        _tls.depth = 0
        with _lock:
            _stage_cpu[(name, threading.get_native_id())] += dt


def stage_totals() -> dict[tuple[str, int], float]:
    with _lock:
        return dict(_stage_cpu)


# -- thread groups -----------------------------------------------------------------------------------------------------

GROUPS = ("decode", "encode", "pll", "dsp", "ui", "rx", "link", "audio", "io", "other", "native")
LABELS = {"decode": "Decode", "encode": "Encode", "pll": "CSS PLL", "dsp": "DSP", "ui": "UI", "rx": "RX capture",
          "link": "Link/MAC", "audio": "Audio I/O", "io": "Ports", "other": "Other", "native": "Native"}

_PREFIX = (
    ("ui", ("MainThread",)),
    ("rx", ("mesh-rx", "ofdm-cap", "modemgui-session", "diag-rx", "mesh-detect", "css-rx", "bcast-port")),
    ("link", ("mesh-", "ofdm-", "hub", "bcast-", "ident-", "iq-link", "iq-arq", "css-")),
    ("audio", ("pulse-", "rtp-", "pump-", "iq:", "sd-", "com.apple.audio", "PortAudio", "pa_", "pw-", "alsa",
               "AudioIO", "caulk")),
    ("io", ("control-", "serial-", "udp-", "hs-port")),
)


def group_of(name: str, python: bool) -> str:
    """The group of a thread by its name; ``python`` = a threading.Thread (else an OS thread Python never named)."""
    for g, prefixes in _PREFIX:
        if any(name.startswith(p) for p in prefixes):
            return g
    return "other" if python else "native"


# -- per-thread CPU from the OS --------------------------------------------------------------------------------------

def _py_threads() -> dict[int, threading.Thread]:
    return {t.native_id: t for t in threading.enumerate() if t.native_id is not None}


def _linux_threads() -> dict[int, tuple[str, float]]:
    tick = os.sysconf("SC_CLK_TCK")
    out = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/stat") as f:
                s = f.read()
        except OSError:
            continue
        comm = s[s.index("(") + 1:s.rindex(")")]
        fields = s[s.rindex(")") + 2:].split()
        out[int(tid)] = (comm, (int(fields[11]) + int(fields[12])) / tick)    # utime + stime
    return out


class _Mach:
    """task_threads / thread_info / pthread_getname_np through ctypes."""

    def __init__(self) -> None:
        import ctypes
        import ctypes.util
        c = self.c = ctypes
        lib = self.lib = ctypes.CDLL(ctypes.util.find_library("c") or "libSystem.B.dylib")
        self.task = c.c_uint.in_dll(lib, "mach_task_self_").value
        lib.task_threads.argtypes = [c.c_uint, c.POINTER(c.POINTER(c.c_uint)), c.POINTER(c.c_uint)]
        lib.thread_info.argtypes = [c.c_uint, c.c_int, c.POINTER(c.c_int), c.POINTER(c.c_uint)]
        lib.pthread_from_mach_thread_np.argtypes = [c.c_uint]
        lib.pthread_from_mach_thread_np.restype = c.c_void_p
        lib.pthread_getname_np.argtypes = [c.c_void_p, c.c_char_p, c.c_size_t]
        lib.pthread_threadid_np.argtypes = [c.c_void_p, c.POINTER(c.c_uint64)]
        lib.mach_port_deallocate.argtypes = [c.c_uint, c.c_uint]
        lib.vm_deallocate.argtypes = [c.c_uint, c.c_size_t, c.c_size_t]

    def threads(self) -> dict[int, tuple[str, float]]:
        c, lib = self.c, self.lib
        arr = c.POINTER(c.c_uint)()
        n = c.c_uint(0)
        if lib.task_threads(self.task, c.byref(arr), c.byref(n)):
            return {}
        out = {}
        try:
            for i in range(n.value):
                port = arr[i]
                info = (c.c_int * 10)()
                cnt = c.c_uint(10)                      # THREAD_BASIC_INFO_COUNT
                if lib.thread_info(port, 3, info, c.byref(cnt)) == 0:   # THREAD_BASIC_INFO
                    cpu = info[0] + info[1] / 1e6 + info[2] + info[3] / 1e6
                    pt = lib.pthread_from_mach_thread_np(port)
                    name, tid = "", port
                    if pt:
                        buf = c.create_string_buffer(64)
                        lib.pthread_getname_np(pt, buf, 64)
                        name = buf.value.decode("utf-8", "replace")
                        t64 = c.c_uint64(0)
                        if lib.pthread_threadid_np(pt, c.byref(t64)) == 0:
                            tid = t64.value                     # == threading.get_native_id() on macOS
                    out[tid] = (name or f"thread {port}", cpu)
                lib.mach_port_deallocate(self.task, port)
        finally:
            lib.vm_deallocate(self.task, c.cast(arr, c.c_void_p).value, n.value * 4)
        return out


class _Win:
    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes
        self.c, self.wt = ctypes, wintypes
        self.k = ctypes.WinDLL("kernel32", use_last_error=True)

    def cpu(self, tid: int) -> float | None:
        c, wt, k = self.c, self.wt, self.k
        h = k.OpenThread(0x0800, False, tid)            # THREAD_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        try:
            ft = [wt.FILETIME() for _ in range(4)]
            if not k.GetThreadTimes(h, *(c.byref(f) for f in ft)):
                return None
            return sum((f.dwHighDateTime << 32 | f.dwLowDateTime) for f in ft[2:]) / 1e7
        finally:
            k.CloseHandle(h)


_mach: _Mach | None = None
_win: _Win | None = None


def thread_cpu() -> dict[int, tuple[str, bool, float]]:
    """{native thread id: (name, is a Python thread, CPU seconds)} for the threads the OS lets us read."""
    global _mach, _win
    py = _py_threads()
    raw: dict[int, tuple[str, float]] = {}
    try:
        if sys.platform.startswith("linux"):
            raw = _linux_threads()
        elif sys.platform == "darwin":
            _mach = _mach or _Mach()
            raw = _mach.threads()
        elif sys.platform == "win32":
            _win = _win or _Win()
            for tid, t in py.items():
                v = _win.cpu(tid)
                if v is not None:
                    raw[tid] = (t.name, v)
        else:
            for tid, t in py.items():
                try:
                    raw[tid] = (t.name, time.clock_gettime(time.pthread_getcpuclockid(t.ident)))
                except (AttributeError, OSError):
                    pass
    except Exception:
        raw = {}
    return {tid: ((py[tid].name, True, cpu) if tid in py else (name, False, cpu)) for tid, (name, cpu) in raw.items()}


# -- the meter -------------------------------------------------------------------------------------------------------

class CpuMeter:
    """Call ``sample()`` periodically; each call reports the interval since the previous one."""

    def __init__(self) -> None:
        self._t = time.monotonic()
        self._proc = time.process_time()
        self._threads = thread_cpu()
        self._stages = stage_totals()
        self.last: dict = {"total": 0.0, "groups": {}, "threads": []}

    def sample(self) -> dict:
        """{"total": %, "groups": {group: %}, "threads": [(name, group, %), ...] busiest first}."""
        now, proc = time.monotonic(), time.process_time()
        threads, stages = thread_cpu(), stage_totals()
        dt = max(now - self._t, 1e-6)
        pct = (lambda sec: 100.0 * sec / dt)
        groups: dict[str, float] = collections.defaultdict(float)
        stage_by_tid: dict[int, float] = collections.defaultdict(float)
        for (st, tid), sec in stages.items():
            d = sec - self._stages.get((st, tid), 0.0)
            if d > 0:
                groups[st] += pct(d)
                stage_by_tid[tid] += d
        rows, seen = [], 0.0
        for tid, (name, is_py, cpu) in threads.items():
            prev = self._threads.get(tid)
            d = cpu - (prev[2] if prev else 0.0)
            if d <= 0:
                continue
            seen += d
            g = group_of(name, is_py)
            rest = max(0.0, d - stage_by_tid.get(tid, 0.0))
            groups[g] += pct(rest)
            rows.append((name, g, pct(d)))
        total = pct(proc - self._proc)
        if threads:
            # threads the OS would not show us (non-Python threads on Windows) and threads that came and went
            unseen = (proc - self._proc) - seen
            if unseen > 0.01 * dt:
                groups["native"] += pct(unseen)
        else:
            groups["other"] += max(0.0, total - sum(groups.values()))
        self._t, self._proc, self._threads, self._stages = now, proc, threads, stages
        rows.sort(key=lambda r: -r[2])
        self.last = {"total": total, "groups": {g: groups[g] for g in GROUPS if groups.get(g, 0.0) > 0.05},
                     "threads": rows}
        return self.last


# Always on the strip, in this order, so the row never reflows as groups go idle; the rest (CSS PLL, unlabelled DSP,
# ports, other) join once they have used CPU and then stay.
CORE = ("decode", "encode", "ui", "link", "rx", "audio", "native")
_FIG = "\u2007"                      # figure space: as wide as a digit, so "  7%" and "107%" take the same room


def summary(sample: dict, shown=None) -> str:
    """``Decode   41% · Encode    6% · UI   12% · ...``: every group in ``shown`` (default: CORE plus any group
    with CPU in this sample), in GROUPS order, zeros included, numbers padded to a fixed width."""
    vals = sample.get("groups", {})
    shown = set(CORE) | {g for g, v in vals.items() if v >= 0.5} if shown is None else set(shown)
    return " · ".join(f"{LABELS[g]} {f'{vals.get(g, 0.0):.0f}'.rjust(3, _FIG)}%" for g in GROUPS if g in shown)
