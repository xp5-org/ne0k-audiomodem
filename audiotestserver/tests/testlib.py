# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""
tests/testlib.py  –  shared utilities for all AudioTool test scripts.

Centralises the boilerplate that was copy-pasted into every wrapper:
  · environment-variable parsing  (CLIENT_ID, RUN_DIR, PARAMS, …)
  · logging helper
  · coordination HTTP helpers     (coord_signal / coord_check / coord_wait_any)
  · device coercion               (to_int_dev)
  · WAV capture utility           (save_wav)

Usage in any test script::

    from testlib import (
        CLIENT_ID, INPUT_DEV, OUTPUT_DEV, RUN_DIR, PARAMS, RUN_ID, SERVER_URL,
        log, to_int_dev,
        coord_signal, coord_check, coord_wait_any, coord_wait,
        save_wav,
    )
"""

from __future__ import annotations

import json
import os
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Environment variables  (read once at import time)
# ---------------------------------------------------------------------------


# --- category-aware sys.path bootstrap (auto-inserted) ---
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_TESTS_ROOT = _HERE.parent if _HERE.name in ("CSS", "MESH", "OFDM", "QAM") else _HERE
_REPO_ROOT = _TESTS_ROOT.parent
for _p in (_TESTS_ROOT, _REPO_ROOT, _TESTS_ROOT / "CSS", _TESTS_ROOT / "MESH",
           _TESTS_ROOT / "OFDM", _TESTS_ROOT / "QAM", _TESTS_ROOT / "regression"):
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# --- end bootstrap ---

CLIENT_ID  = os.environ.get("AUDIOTOOL_CLIENT_ID",     "unknown")
INPUT_DEV  = os.environ.get("AUDIOTOOL_INPUT_DEVICE",  "")
OUTPUT_DEV = os.environ.get("AUDIOTOOL_OUTPUT_DEVICE", "")
RUN_DIR    = Path(os.environ.get("AUDIOTOOL_RUN_DIR",  "."))
PARAMS     = json.loads(os.environ.get("AUDIOTOOL_PARAMS",  "{}"))
RUN_ID     = os.environ.get("AUDIOTOOL_RUN_ID",        "unknown")
SERVER_URL = os.environ.get("AUDIOTOOL_SERVER_URL",    "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_shutting_down = False      # set by run_shutdown_hooks(); see log() below


def log(msg: str) -> None:
    # During shutdown this runs from a signal handler, which can interrupt
    # the main thread inside print()'s non-reentrant BufferedWriter lock.
    # A raw fd write takes no Python lock, so hooks keep logging safely.
    if _shutting_down:
        data = (msg + "\n").encode("utf-8", "replace")
        while data:
            try:
                data = data[os.write(1, data):]
            except OSError:
                return
    else:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# C backend preflight
# ---------------------------------------------------------------------------

import cbackend                                                  # noqa: E402


def backend_inventory() -> dict:
    """This host's CPU and the state of every C core, for the registration
    payload. Never raises."""
    try:
        return {
            "host_tag":  cbackend.host_tag(),
            "backends":  cbackend.inventory(),
        }
    except Exception as exc:                                     # noqa: BLE001
        return {"host_tag": "unknown", "backends": [], "error": str(exc)}


def preflight_backends(keys=None, fatal: bool = True) -> list[str]:
    """Fail this role script now if a C backend it was told to use is unusable.

    Returns the list of problems (empty when clean). With `fatal` (the
    default) a non-empty list exits non-zero after writing a
    `preflight_fail` result into RUN_DIR.

    Call this at the top of a role script, before opening audio devices: a
    wrong-architecture library is a property of the host, not the run.

    Backends that were not requested are not graded — an unbuilt core with
    no env var pointing at it is the default configuration.
    """
    ok, problems = cbackend.check(keys)
    log(f"[preflight] {cbackend.banner()}")
    if ok:
        return []
    for p in problems:
        log(f"[preflight] FAIL: {p}")
    if not fatal:
        return problems
    try:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        (RUN_DIR / f"result_{CLIENT_ID}_preflight.json").write_text(json.dumps({
            "run_id":    RUN_ID,
            "client_id": CLIENT_ID,
            "status":    "preflight_fail",
            "error":     "; ".join(problems),
            "host_tag":  cbackend.host_tag(),
            "backends":  cbackend.inventory(),
            "ts":        time.time(),
        }, indent=2))
    except Exception as exc:                                     # noqa: BLE001
        log(f"[preflight] could not write result stub: {exc}")
    raise SystemExit(3)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def to_int_dev(val) -> Optional[int]:
    """Coerce a device value to int, returning None on failure."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Coordination helpers
# ---------------------------------------------------------------------------

def coord_signal(signal: str) -> list:
    """POST this client's arrival on *signal*.  Returns list of all signalers."""
    import urllib.request
    url  = f"{SERVER_URL}/api/coord/{RUN_ID}/{signal}"
    body = json.dumps({"client_id": CLIENT_ID}).encode()
    req  = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read()).get("clients", [])
    except Exception as exc:
        log(f"  [coord] {signal} failed: {exc}")
        return []


def coord_check(signal: str) -> list:
    """GET current signaler list for *signal* (non-blocking)."""
    import urllib.request
    try:
        url = f"{SERVER_URL}/api/coord/{RUN_ID}/{signal}"
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read()).get("clients", [])
    except Exception:
        return []


def coord_check_ts(signal: str) -> dict:
    """GET signaler list WITH server-clock arrival times (non-blocking).

    Returns {"clients": [...], "ts": {node: server_t}, "server_now": t}
    (empty dict fields on error). The timestamps let every poller compute
    the same schedule arithmetic regardless of when it polls.
    """
    import urllib.request
    try:
        url = f"{SERVER_URL}/api/coord/{RUN_ID}/{signal}"
        with urllib.request.urlopen(url, timeout=5) as r:
            d = json.loads(r.read())
            return {"clients": d.get("clients", []), "ts": d.get("ts", {}),
                    "server_now": d.get("server_now")}
    except Exception:
        return {"clients": [], "ts": {}, "server_now": None}


def coord_wait_any(signal: str,
                   timeout: float = 30.0,
                   poll: float = 0.3) -> list:
    """Block until ≥ 1 client signals *signal* or *timeout* expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        clients = coord_check(signal)
        if clients:
            return clients
        time.sleep(poll)
    return []


def coord_wait(signal: str,
               timeout: float = 60.0,
               poll: float = 0.3) -> bool:
    """Block until *signal* appears or *timeout* expires.  Returns True on success."""
    return bool(coord_wait_any(signal, timeout=timeout, poll=poll))


# ---------------------------------------------------------------------------
# WAV capture
# ---------------------------------------------------------------------------

def save_wav(audio: np.ndarray, fs: int, path: "str | Path") -> None:
    """
    Save a float32 mono array (values in [-1, 1]) as a 16-bit WAV.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 samples.  Multi-channel arrays are mixed to mono by
        taking the first channel.
    fs    : int
        Sample rate in Hz.
    path  : str or Path
        Destination file path (will be created / overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat = audio[:, 0] if (audio.ndim > 1) else audio.flatten()
    pcm  = (np.clip(flat.astype(np.float32), -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(pcm.tobytes())

    duration = len(pcm) / max(fs, 1)
    log(f"  WAV capture -> {path.name}  ({duration:.1f}s @ {fs} Hz, {len(pcm)} samples)")


class _WavSpool:
    """A WAV file that is valid on disk after every write.

    `save_wav` at the end of a run loses the whole capture if the process
    dies first, since the capture lives in RAM until the loop returns.

    `wave.Wave_write.writeframes` re-patches the RIFF header on every call,
    so appending + flushing as we go leaves a playable file at all times;
    the spill interval is the most audio that can ever be lost.
    """

    def __init__(self, path: "str | Path", fs: int):
        self.path = Path(path)
        self.fs = int(fs)
        self.frames = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "wb")
        self._wf = wave.open(self._fh, "wb")
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)
        self._wf.setframerate(self.fs)

    def write(self, audio: np.ndarray) -> None:
        flat = audio[:, 0] if (audio.ndim > 1) else audio.reshape(-1)
        pcm = (np.clip(flat.astype(np.float32), -1.0, 1.0) * 32767).astype(np.int16)
        if not len(pcm):
            return
        self._wf.writeframes(pcm.tobytes())   # patches the header every call
        self._fh.flush()
        self.frames += len(pcm)

    def close(self) -> None:
        try:
            self._wf.close()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Shutdown hooks
# ---------------------------------------------------------------------------
#
# The rig stops a test with proc.terminate() (SIGTERM). CPython's default
# SIGTERM disposition kills the interpreter outright -- no `finally`, no
# atexit -- so role scripts register what must survive here instead.

_shutdown_hooks: list = []

# Open PortAudio streams, so a stop can silence the device before doing
# anything else. The shutdown hooks (capture spill, result file) can take
# seconds against a network share, during which the PortAudio callback
# thread would otherwise keep feeding the speaker.
_streams: list = []


def track_stream(stream):
    """Register an open sd.InputStream/OutputStream for shutdown silencing."""
    try:
        _streams.append(stream)
    except Exception:
        pass
    return stream


def abort_audio() -> int:
    """Stop every tracked stream immediately, discarding what is queued.

    `abort()` rather than `stop()`: stop() drains the buffer first, which
    on a 0.5s-latency output device is another half second of transmission.
    """
    n = 0
    for st in list(_streams):
        for meth in ("abort", "close"):
            try:
                getattr(st, meth)()
                n += 1
                break
            except Exception:
                continue
    try:
        import sys as _sys
        sd = _sys.modules.get("sounddevice")
        if sd is not None:
            sd.stop()          # module-level sd.play() convenience stream
    except Exception:
        pass
    del _streams[:]
    return n


def on_shutdown(fn) -> None:
    """Register `fn()` to run before the process dies on SIGTERM/SIGINT."""
    _shutdown_hooks.append(fn)


def run_shutdown_hooks() -> None:
    """Run every registered hook once, best-effort, never raising."""
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    for fn in list(_shutdown_hooks):
        try:
            fn()
        except Exception as exc:
            log(f"  [shutdown] hook {getattr(fn, '__name__', fn)!r} "
                f"failed: {type(exc).__name__}: {exc}")


def install_shutdown_handler() -> None:
    """Flush registered hooks on SIGTERM/SIGINT, then exit.

    Must be called from the main thread (signal handlers can only be
    installed there); silently skipped otherwise.  Exits via os._exit after
    the hooks so a blocking audio call cannot swallow the unwind.
    """
    import signal as _sig

    def _handler(signum, _frame):
        global _shutting_down
        _shutting_down = True     # route logging off the stdout buffer lock
        # Silence the device before the hooks: the hooks are the slow part
        # (a capture spill to the shared mount), and until they finish the
        # output stream is still playing.
        abort_audio()
        log(f"  *** signal {signum} -- flushing capture + result before exit ***")
        run_shutdown_hooks()
        os._exit(128 + int(signum))

    for s in (_sig.SIGTERM, _sig.SIGINT):
        try:
            _sig.signal(s, _handler)
        except (ValueError, OSError, AttributeError):
            pass
    install_child_reaper()


# ---------------------------------------------------------------------------
# Child processes
#
# A role script is a process tree: a capture worker, a decode pool of
# numpy-loaded workers, sometimes a second daemon. Two invariants:
#
#   · a child must not outlive this process. The shutdown handler exits via
#     os._exit (so a blocking audio call cannot swallow the unwind), which
#     runs no atexit and no pool shutdown, so pool workers must be killed
#     explicitly.
#   · a child must not outlive this process even if this process is
#     SIGKILLed and never runs any handler. Only the kernel can enforce
#     that: PR_SET_PDEATHSIG, set by the child on itself.
#
# `guard_orphan()` is the child-side call, `track_child()` the parent-side one.
# ---------------------------------------------------------------------------

_children: list = []          # (pid, label) launched by this process


def _orphan_watchdog(poll_s: float = 1.0) -> bool:
    """Userspace stand-in for PR_SET_PDEATHSIG, for kernels without it.

    macOS has no PDEATHSIG. A daemon thread watching getppid() is weaker (it
    cannot survive a SIGSTOPped process, and costs one wakeup a second) but
    closes the common case: parent gone -> reparented to init/launchd -> exit.
    """
    import threading as _th

    def _watch():
        while True:
            try:
                if os.getppid() in (1, 0):
                    os._exit(0)
            except Exception:
                return
            time.sleep(poll_s)

    try:
        t = _th.Thread(target=_watch, name="orphan-watchdog", daemon=True)
        t.start()
        return True
    except Exception:
        return False


def guard_orphan(sig: int | None = None) -> bool:
    """Ask the kernel to kill THIS process when its parent dies.

    Call first thing in any child process (mp target, pool initializer).
    Uses PR_SET_PDEATHSIG on Linux and a getppid watchdog elsewhere.
    Returns True if a guard is armed.
    """
    import signal as _sig
    armed = False
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        armed = libc.prctl(PR_SET_PDEATHSIG,
                           int(sig or _sig.SIGKILL), 0, 0, 0) == 0
    except Exception:
        armed = False
    if not armed:
        armed = _orphan_watchdog()
    # Race: the parent may already have died between fork/spawn and here, in
    # which case the signal has been and gone.  Check and exit if so.
    try:
        if os.getppid() in (1, 0):
            os._exit(0)
    except Exception:
        pass
    return armed


def track_child(proc, label: str = "") -> None:
    """Register a child (Popen, mp.Process, or raw pid) for shutdown killing."""
    pid = getattr(proc, "pid", proc)
    try:
        _children.append((int(pid), label or str(pid)))
    except Exception:
        pass


def kill_tracked_children(grace: float = 2.0) -> int:
    """SIGTERM then SIGKILL every tracked child *and its descendants*.

    Called from the shutdown hooks, so it also covers the SIGTERM path where
    `finally:` blocks and pool shutdowns never run.
    """
    import signal as _sig
    victims = []
    for pid, label in list(_children):
        for p in _descendants(pid) + [pid]:
            if p not in victims:
                victims.append(p)
    if not victims:
        return 0
    for p in victims:
        try:
            os.kill(p, _sig.SIGTERM)
        except Exception:
            pass
    deadline = time.time() + grace
    while time.time() < deadline and any(_pid_alive(p) for p in victims):
        time.sleep(0.1)
    left = [p for p in victims if _pid_alive(p)]
    for p in left:
        try:
            os.kill(p, _sig.SIGKILL)
        except Exception:
            pass
    if victims:
        log(f"  [shutdown] killed {len(victims)} child process(es)"
            f"{' (SIGKILL for ' + str(len(left)) + ')' if left else ''}")
    return len(victims)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _ppid_map() -> dict:
    """pid -> [child pids], from /proc where it exists and `ps` where it does
    not (macOS has no /proc)."""
    kids: dict = {}
    try:
        entries = [e for e in os.listdir("/proc") if e.isdigit()]
    except Exception:
        entries = []
    if entries:
        for entry in entries:
            try:
                with open(f"/proc/{entry}/stat") as fh:
                    fields = fh.read().rsplit(") ", 1)[-1].split()
                ppid = int(fields[1])
            except Exception:
                continue
            kids.setdefault(ppid, []).append(int(entry))
        return kids
    try:
        import subprocess as _sp
        out = _sp.run(["ps", "-Ao", "pid=,ppid="], capture_output=True,
                      text=True, timeout=10).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                kids.setdefault(int(parts[1]), []).append(int(parts[0]))
            except ValueError:
                continue
    except Exception:
        return {}
    return kids


def _descendants(pid: int) -> list:
    """Child pids of `pid`, recursively (no psutil in tests)."""
    kids = _ppid_map()
    out, stack = [], list(kids.get(pid, []))
    while stack:
        p = stack.pop()
        if p in out:
            continue
        out.append(p)
        stack.extend(kids.get(p, []))
    return out


def install_child_reaper() -> None:
    """Register kill_tracked_children() as a shutdown hook (idempotent)."""
    if getattr(install_child_reaper, "_done", False):
        return
    install_child_reaper._done = True
    on_shutdown(kill_tracked_children)
    import atexit
    atexit.register(kill_tracked_children)


def pool_worker_init(fn=None, *args):
    """ProcessPoolExecutor initializer that arms the orphan guard first.

    Use as ``initializer=testlib.pool_worker_init, initargs=(real_init, ...)``
    -- or with no initargs when the pool has no initializer of its own.
    """
    guard_orphan()
    if fn is not None:
        fn(*args)


# ---------------------------------------------------------------------------
# Real-time capture / playback  (for half-duplex tests like two-way ranging)
# ---------------------------------------------------------------------------

class _InlineRecorder:
    """In-process capture engine (sounddevice callback thread).

    The callback shares this process's GIL, so sustained pure-Python decode
    work starves it and the device queue overflows. Use `Recorder` below,
    which runs this engine inside a dedicated capture process; this class
    remains as its worker core and as the fallback when the worker can't
    start.
    """

    def __init__(self, in_dev, fs: int, channels: int = 1):
        import sounddevice as sd
        self._sd = sd
        self.fs = int(fs)
        self._chunks: list = []
        self.frames_captured = 0
        self.overflows = 0          # input_overflow callbacks (dropped audio)
        # latency='high': deep host-side buffering so a scheduling stall in
        # the main thread (long decode polls) doesn't overflow the device
        # queue. Periodic sample drops on a USB mic break long coherent OFDM
        # frames while short/robust signals survive.
        self._in_dev = in_dev
        self._channels = channels
        # Explicit 0.5s: latency='high' resolves to the device's own figure,
        # which can be too short to absorb a decode-poll stall.
        self._stream = self._open(0.5)

    def _open(self, latency):
        self._latency = latency
        return track_stream(self._sd.InputStream(
            samplerate=self.fs, channels=self._channels, dtype="float32",
            device=self._in_dev, latency=latency, callback=self._cb))

    def _cb(self, indata, frames, time_info, status):
        if status and status.input_overflow:
            self.overflows += 1
        self._chunks.append(indata[:, 0].copy())
        self.frames_captured += frames

    def _alsa_kick(self):
        """Open the device briefly outside PortAudio (direct ALSA arecord).

        Some devices wedge to all-zero output after a stream close and stay
        wedged for every later PortAudio open; a direct-ALSA open revives
        them. Mimics that with a 1s arecord of the raw hw device, output
        discarded. Best-effort -- silently skipped where arecord or the hw
        name is unavailable."""
        import re, subprocess
        try:
            name = str(self._sd.query_devices(self._in_dev)["name"])
            m = re.search(r"\((hw:\d+,\d+)\)", name)
            hw = m.group(1) if m else None
            if hw is None:
                return
            log(f"  [rec] ALSA kick: 1s arecord on {hw}")
            subprocess.run(
                ["arecord", "-D", hw, "-f", "S16_LE", "-r", str(self.fs),
                 "-c", "1", "-d", "1", "/dev/null"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=8)
        except Exception as exc:
            log(f"  [rec] ALSA kick failed: {exc}")

    def _stream_silent(self):
        """Wait for real frames; True if the stream delivers EXACT zeros.

        Real ambience is never all-zero float.  Waits for actual frames
        before judging: with a 0.5s buffer the FIRST callback can land
        late, and an empty data() must not pass a dead stream through.
        """
        import time as _t
        deadline = _t.time() + 2.5
        min_n = int(0.3 * self.fs)
        while _t.time() < deadline and len(self.data()) < min_n:
            _t.sleep(0.1)
        d = self.data()
        return len(d) >= min_n and float(np.max(np.abs(d))) == 0.0

    def start(self):
        # All-zero output is ambiguous: some capture paths deliver exact
        # zeros when a non-default latency config fails (dead stream), but
        # a mic with firmware noise blanking can legitimately gate a quiet
        # room to digital zero while passing real signals. One reopen at
        # device default rules out the config-failure case; the node-level
        # self-echo test is the authoritative aliveness arbiter, not the
        # idle floor.
        self._stream.start()
        if self._stream_silent():
            log(f"  [rec] latency={self._latency!r} reads all-zero -- one "
                f"reopen at device default (zeros may be normal blanking)")
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
            self._chunks = []
            self.frames_captured = 0
            self._stream = self._open(None)
            self._stream.start()
        log(f"  [rec] capture OPEN @ {self.fs} Hz "
            f"latency={float(self._stream.latency)*1000:.1f}ms")

    def stop(self):
        try:
            self._stream.stop(); self._stream.close()
        except Exception:
            pass
        if self.overflows:
            log(f"  [rec] WARNING: {self.overflows} input overflow(s) -- "
                f"capture has dropped-sample gaps")

    def data(self) -> np.ndarray:
        return (np.concatenate(self._chunks) if self._chunks
                else np.zeros(0, dtype=np.float32))

    def take(self) -> list:
        """Pop accumulated chunks (used by the capture worker's forward loop
        so the child process never re-concatenates its whole history)."""
        n = len(self._chunks)
        out = self._chunks[:n]
        del self._chunks[:n]
        return out

    def save(self, path):
        save_wav(self.data(), self.fs, path)


def _capture_proc_main(in_dev, fs, channels, q, stop_evt):
    """Capture worker entrypoint (own process -> own GIL).

    Runs the inline engine (with its full open-cycle/kick/silence recovery)
    and forwards chunks to the parent over `q`, so decode load in the test
    process cannot starve the audio callback. stdout is inherited, so the
    engine's own log lines land in the test log as usual.
    """
    import time as _t
    guard_orphan()      # never outlive the role script, even if it is SIGKILLed
    try:
        rec = _InlineRecorder(in_dev, fs, channels)
        rec.start()
    except Exception as exc:
        q.put(("fail", f"{type(exc).__name__}: {exc}"))
        return
    q.put(("open", float(rec._stream.latency)))
    try:
        while not stop_evt.is_set():
            for arr in rec.take():
                q.put(("c", arr.tobytes(), rec.overflows))
            _t.sleep(0.1)
    finally:
        try:
            rec.stop()
        except Exception:
            pass
        for arr in rec.take():
            q.put(("c", arr.tobytes(), rec.overflows))
        q.put(("done", rec.overflows))


class Recorder:
    """Process-isolated capture with the same API as the old Recorder.

    The stream runs inside a dedicated capture process (`_capture_proc_main`)
    whose GIL is untouched by the test's decode work, so long pure-Python
    decode loops cannot cause buffer-sized capture gaps. A parent-side
    drain thread mirrors chunks into memory, so `data()` /
    `frames_captured` / `overflows` behave as before. If the worker can't
    start (spawn or device failure), falls back to the inline engine with a
    warning rather than refusing to run.
    """

    def __init__(self, in_dev, fs: int, channels: int = 1):
        import multiprocessing as mp
        self.fs = int(fs)
        self._chunks: list = []
        self._fc = 0
        self._ov = 0
        self._in_dev = in_dev
        self._channels = channels
        self._inline = None
        self._done = False
        self._dead = False        # capture worker died mid-run
        self._spool = None        # incremental WAV (see spill_to)
        self._spilled = 0         # chunks already written to the spool
        self._spill_lock = None
        self._spill_stop = None
        # spawn, never fork: the parent may already hold live PortAudio
        # state (device queries, playback), which a forked child would clone
        self._ctx = mp.get_context("spawn")
        self._q = self._ctx.Queue()
        self._stop_evt = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=_capture_proc_main,
            args=(in_dev, self.fs, channels, self._q, self._stop_evt),
            daemon=True)

    @property
    def frames_captured(self):
        return (self._inline.frames_captured if self._inline is not None
                else self._fc)

    @property
    def overflows(self):
        return (self._inline.overflows if self._inline is not None
                else self._ov)

    @property
    def alive(self) -> bool:
        """False once the capture worker has died -- data() is frozen."""
        return self._inline is not None or not self._dead

    def _handle(self, m):
        if m[0] == "c":
            arr = np.frombuffer(m[1], dtype=np.float32)
            self._chunks.append(arr)
            self._fc += len(arr)
            self._ov = m[2]
        elif m[0] == "done":
            self._ov = m[1]
            self._done = True

    def _drain_loop(self):
        import queue as _qm
        while not self._done:
            try:
                self._handle(self._q.get(timeout=0.5))
            except _qm.Empty:
                if not self._proc.is_alive():
                    # A worker death leaves _done False and data() frozen
                    # at whatever already arrived; log it explicitly and let
                    # `alive` fail the run (meshmodem capture guard).
                    if not self._done:
                        self._dead = True
                        log(f"  [rec] *** CAPTURE WORKER DIED "
                            f"(exitcode={self._proc.exitcode}) -- input "
                            f"audio has STOPPED, this node is now deaf ***")
                    return

    def start(self):
        import queue as _qm
        import threading
        import time as _t
        self._proc.start()
        track_child(self._proc, "capture worker")
        install_child_reaper()
        # worker may cycle latencies / ALSA kicks before opening: allow 30s
        opened = None
        deadline = _t.time() + 30.0
        while _t.time() < deadline and opened is None:
            try:
                m = self._q.get(timeout=0.5)
            except _qm.Empty:
                if not self._proc.is_alive():
                    break
                continue
            if m[0] == "open":
                opened = m[1]
            elif m[0] == "fail":
                log(f"  [rec] capture worker failed: {m[1]}")
                break
            else:
                self._handle(m)
        if opened is None:
            log("  [rec] capture process unavailable -- falling back to "
                "in-process capture (decode load may starve it)")
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._inline = _InlineRecorder(self._in_dev, self.fs,
                                           self._channels)
            self._inline.start()
            return
        threading.Thread(target=self._drain_loop, daemon=True).start()
        log(f"  [rec] capture OPEN @ {self.fs} Hz "
            f"latency={opened*1000:.1f}ms (isolated process)")

    def _chunk_src(self) -> list:
        return (self._inline._chunks if self._inline is not None
                else self._chunks)

    def spill_to(self, path, interval_s: float = 15.0):
        """Stream the capture to `path` for the life of the run.

        Call once, right after start(). The file is a valid WAV from the
        first flush onward, so a killed / crashed / stopped run still leaves
        everything up to the last flush on disk. Returns the _WavSpool so
        the caller can close it (stop() does that too).
        """
        import threading

        if self._spool is not None:
            return self._spool
        self._spool = _WavSpool(path, self.fs)
        # RLock: SIGTERM can land while THIS thread already holds it inside
        # stop()->close_spill(), and the handler calls close_spill() again --
        # a plain Lock would deadlock until the rig's force-kill.
        self._spill_lock = threading.RLock()
        self._spill_stop = threading.Event()
        log(f"  [rec] capture spilling to {Path(path).name} "
            f"every {interval_s:g}s (survives SIGTERM/crash)")

        def _loop():
            while not self._spill_stop.wait(interval_s):
                try:
                    self.spill_flush()
                except Exception as exc:
                    log(f"  [rec] spill write failed: "
                        f"{type(exc).__name__}: {exc}")
                    return

        threading.Thread(target=_loop, daemon=True).start()
        return self._spool

    def spill_flush(self) -> None:
        """Append everything captured since the last flush.  Safe to call
        from a signal handler or a shutdown hook."""
        if self._spool is None:
            return
        with self._spill_lock:
            src = self._chunk_src()
            n = len(src)
            if n <= self._spilled:
                return
            pending = src[self._spilled:n]
            self._spilled = n
            for arr in pending:
                self._spool.write(arr)

    def close_spill(self) -> None:
        """Final flush + close.  Idempotent."""
        if self._spool is None:
            return
        if self._spill_stop is not None:
            self._spill_stop.set()
        try:
            self.spill_flush()
        except Exception:
            pass
        sp, self._spool = self._spool, None
        sp.close()
        log(f"  WAV capture -> {sp.path.name}  "
            f"({sp.frames / max(self.fs, 1):.1f}s @ {self.fs} Hz, "
            f"{sp.frames} samples)")

    def stop(self):
        if self._inline is not None:
            self._inline.stop()
            self.close_spill()
            return
        import time as _t
        self._stop_evt.set()
        deadline = _t.time() + 8.0
        while not self._done and _t.time() < deadline:
            _t.sleep(0.1)
        try:
            self._proc.join(timeout=3.0)
        except Exception:
            pass
        # daemon=True only reaps the worker via multiprocessing's atexit hook,
        # which the SIGTERM path (os._exit) never runs -- and a wedged worker
        # ignores the stop event anyway.  Make the death explicit.
        if self._proc.is_alive():
            log("  [rec] capture worker still alive after stop -- terminating")
            try:
                self._proc.terminate()
                self._proc.join(timeout=2.0)
            except Exception:
                pass
            if self._proc.is_alive():
                try:
                    self._proc.kill()
                except Exception:
                    pass
        while True:                       # drain any tail left in the queue
            try:
                self._handle(self._q.get_nowait())
            except Exception:
                break
        if self._ov:
            log(f"  [rec] WARNING: {self._ov} input overflow(s) in capture "
                f"worker -- capture has dropped-sample gaps")
        self.close_spill()

    def data(self) -> np.ndarray:
        if self._inline is not None:
            return self._inline.data()
        n = len(self._chunks)
        return (np.concatenate(self._chunks[:n]) if n
                else np.zeros(0, dtype=np.float32))

    def save(self, path):
        save_wav(self.data(), self.fs, path)


def decode_array(pcm: np.ndarray, fs: int, info_path, overrides: dict | None = None):
    """Run the real css_decoder.decode() over an in-memory capture array.

    Feeds `pcm` to the decoder through a fake sounddevice module (the same
    technique the offline regression harness uses), so a role script can decode
    exactly what its Recorder captured — no second live pass over the mic.
    css_decoder gates its hardware paths on hasattr(sd,'query_devices'); the bare
    mock has none, so it takes the blocking/test path, and an empty read (via the
    stop_event) signals end-of-input.

    Returns the decoder result dict (status, frames[], diagnostics, ...).
    IMPORTANT: restores sys.modules['sounddevice'] before returning, so a later
    play_blocking()/Recorder gets the real module back.
    """
    import contextlib as _cl
    import importlib as _il
    import io as _io
    import sys as _sys
    import threading as _th
    import types as _types

    pcm = np.ascontiguousarray(pcm, dtype=np.float32)
    pos = [0]
    stop = _th.Event()

    class _FakeStream:
        def __init__(s, **k): pass
        def __enter__(s): return s
        def __exit__(s, *a): return False

        def read(s, n):
            c = pcm[pos[0]:pos[0] + n]
            pos[0] += len(c)
            if len(c) < n:
                stop.set()
                c = np.concatenate([c, np.zeros(n - len(c), dtype=np.float32)])
            return c.astype(np.float32), False

    fake = _types.ModuleType("sounddevice")
    fake.InputStream = lambda **k: _FakeStream()

    saved_dec = _sys.modules.pop("css_decoder", None)
    saved_sd = _sys.modules.get("sounddevice")
    _sys.modules["sounddevice"] = fake
    try:
        import css_decoder
        _il.reload(css_decoder)
        cfg = {"info": str(info_path),
               "result_out": str(Path(RUN_DIR) / "_decode_array_result.json"),
               "timeout_sec": 100000, "fs": int(fs)}
        if overrides:
            cfg.update(overrides)
        buf = _io.StringIO()
        with _cl.redirect_stdout(buf):
            res = css_decoder.decode(cfg, stop_event=stop)
        res["_log"] = buf.getvalue()
        return res
    finally:
        if saved_dec is not None:
            _sys.modules["css_decoder"] = saved_dec
        else:
            _sys.modules.pop("css_decoder", None)
        if saved_sd is not None:
            _sys.modules["sounddevice"] = saved_sd
        else:
            _sys.modules.pop("sounddevice", None)


def play_blocking(audio: np.ndarray, fs: int, out_dev, tail_s: float = 0.3) -> float:
    """Play a mono float32 array through out_dev, blocking until it has flushed.

    Uses a callback OutputStream opened at `fs` so a device running at its
    native rate is not resampled.  Returns the stream's output latency (s).
    """
    import sounddevice as sd
    import threading

    a = audio.reshape(-1, 1) if audio.ndim == 1 else audio
    nframes, nch = a.shape
    done = threading.Event()
    state = {"pos": 0}

    def cb(outdata, frames, time_info, status):
        pos = state["pos"]; chunk = a[pos:pos + frames]; n = len(chunk)
        outdata[:n] = chunk
        if n < frames:
            outdata[n:] = 0
            state["pos"] = pos + n
            done.set()
            raise sd.CallbackStop
        state["pos"] = pos + frames

    stream = sd.OutputStream(samplerate=int(fs), channels=nch, dtype="float32",
                             device=out_dev, callback=cb)
    track_stream(stream)          # so Stop can silence it mid-burst
    try:
        with stream:
            done.wait(timeout=nframes / float(fs) + 5)
            time.sleep(float(stream.latency) + tail_s)
    finally:
        try:
            _streams.remove(stream)
        except ValueError:
            pass
    return float(stream.latency)
