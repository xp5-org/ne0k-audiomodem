# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""
procreg.py  --  process-group registry and reaper for test subprocesses.

Why this exists
---------------
A test is not one process.  `_exec_test` starts the role script, the role
script starts a multiprocessing capture worker, a ProcessPoolExecutor with
`jobs` numpy-loaded decode workers, and in the MESH case a whole second
daemon in its own session.  `proc.terminate()` reaches exactly one of them.
Everything else survives -- hundreds of MB each, holding the audio device --
until someone notices with `ps` and kills them by hand.  Worse, if the node
app itself dies (crash, container restart, kill -9) nothing at all is
terminated and the leftovers outlive every trace of the run that made them.

The design has three layers, so no single failure mode leaks:

  1. Every test is launched with `start_new_session=True`, so the script and
     all its descendants share one process-group id.  Stopping a test kills
     the GROUP (`os.killpg`), not the leader.
  2. Every launched group is written to an on-disk registry BEFORE it is
     started and removed when it exits.  The registry survives the app, so a
     crashed/killed/restarted node reaps the previous instance's leftovers at
     boot (`sweep_stale`).
  3. A registry entry can still be missed -- a child that calls setsid itself
     (meshrun_node does), or a process started before this module existed.
     Those are caught by environment scan: every test descendant inherits
     AUDIOTOOL_RUN_ID, so `scan_untracked()` finds them regardless of how
     they were spawned.

Identity is checked with the kernel's process start time on every kill, so a
recycled pid can never be mistaken for the process that was registered.
"""

from __future__ import annotations

import errno
import json
import os
import signal
import threading
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import psutil

import shared

def _host_tag() -> str:
    """Filesystem-safe identity of the machine this process runs on."""
    import socket
    raw = os.environ.get("AUDIOTOOL_HOST_TAG") or socket.gethostname() or "host"
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in raw)[:48]


# The registry is keyed BY HOST, and that is load-bearing rather than tidiness.
# BASE_DIR is frequently a network share mounted by several machines at once
# (the mac nodes run this straight off an SMB mount of the same checkout), and
# a pid only means something on the machine that issued it.  One shared file
# would let this host read another host's pid, match it against a local process
# that happens to have the same number, and reap something innocent -- and on
# SMB the atomic replace also fails EBUSY the moment another client holds the
# file open, which is what killed a run outright.
#
# Per-HOST, not per-node: two nodes on one box (server + client is the normal
# deployment) must still share a registry, or `scan_untracked`'s busy_runs
# check cannot see the other node's live run and a booting node would kill its
# escaped children mid-test.
_HOST     = _host_tag()


# ---------------------------------------------------------------------------
# Where the registry lives
#
# Must be machine-local, not on the SMB share every node has mounted: a pid
# registry has no meaning off the machine that wrote it.
#
#   · macOS writes an ACL alongside the POSIX mode on an SMB file, and the
#     ACL wins: `chmod -R 0777` sets the mode bits but the node still gets
#     EACCES opening its own lock file.
#   · smbfs refuses rename-over-open (EBUSY), so the atomic replace is not
#     atomic, which is what `_write`'s retry ladder below exists to handle.
#   · one host's registry file is readable by another host, whose pids mean
#     something entirely different.
#
# So: machine-local by default, first writable candidate wins.  The share is
# kept as the last resort only so a locked-down host still gets *something*.
# ---------------------------------------------------------------------------

def _candidate_dirs() -> list:
    out = []
    env = os.environ.get("AUDIOTOOL_PROCREG_DIR")
    if env:
        out.append(Path(env))
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        out.append(Path(xdg) / "audiotool")
    home = Path(os.path.expanduser("~"))
    if sys.platform == "darwin":
        out.append(home / "Library" / "Caches" / "audiotool" / "run")
    out.append(home / ".cache" / "audiotool" / "run")
    out.append(Path(tempfile.gettempdir()) / f"audiotool-{os.getuid()}")
    out.append(shared.BASE_DIR / "run")          # legacy / last resort
    return out


def _pick_reg_dir() -> Path:
    """First candidate we can actually create AND write a file in.

    Writability is proved by writing, not by os.access: over SMB with an ACL
    the mode bits say yes and the open still fails, which is the whole reason
    this function exists.
    """
    last = None
    for d in _candidate_dirs():
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / f".probe_{os.getpid()}"
            probe.write_text("")
            probe.unlink()
            return d
        except Exception:
            last = d
            continue
    return last or (shared.BASE_DIR / "run")


REG_DIR   = _pick_reg_dir()
REG_FILE  = REG_DIR / f"procs_{_HOST}.json"
LOCK_FILE = REG_DIR / f".procs_{_HOST}.lock"

# The pre-move location, read once at boot so a node restarting after this
# change still reaps the previous instance's groups instead of orphaning them.
LEGACY_REG_FILE = shared.BASE_DIR / "run" / f"procs_{_HOST}.json"

# Marker env var carried by every descendant of a test launch (set by
# _exec_test).  Used by the environment scan to recognise strays.
ENV_MARK = "AUDIOTOOL_RUN_ID"

_TERM_GRACE = 5.0     # seconds between SIGTERM and SIGKILL for a group


# ---------------------------------------------------------------------------
# Registry file  (JSON dict pid -> entry, rewritten atomically under flock)
# ---------------------------------------------------------------------------

# A single in-process lock, always held.  The file lock below is the
# cross-process half and is best-effort; this one is not, so two threads in
# one node can never interleave a read-modify-write of the registry even when
# the filesystem refuses to co-operate.
_thread_lock = threading.RLock()
_lock_warned = False


class _Locked:
    """Exclusive lock on the registry: thread lock always, file lock if it can.

    The file lock must never be load-bearing.  Opening it failed outright on a
    mac client whose SMB-mounted lock file carried a deny ACL (EACCES on every
    call, 0777 mode bits notwithstanding), and because __enter__ raised, the
    exception came out of `entries()` -> `live_entries()` -> `scan_untracked()`
    and killed the reaper loop's pass -- so a permissions problem on a lock
    file stopped the node reaping anything at all.  Degrade instead: warn
    once, run unlocked, and let the callers' existing tolerance for a torn
    read do its job.
    """

    def __init__(self):
        self._fh = None

    def __enter__(self):
        global _lock_warned
        _thread_lock.acquire()
        try:
            REG_DIR.mkdir(parents=True, exist_ok=True)
            self._fh = open(LOCK_FILE, "a+")
        except Exception as exc:
            self._fh = None
            if not _lock_warned:
                _lock_warned = True
                print(f"[procreg] cannot open {LOCK_FILE}: "
                      f"{type(exc).__name__}: {exc} -- continuing without the "
                      f"cross-process lock (registry is per-host anyway)")
            return self
        try:
            import fcntl
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
        return self

    def __exit__(self, *_exc):
        if self._fh is not None:
            try:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        _thread_lock.release()
        return False


_legacy_read = False


def _read() -> dict:
    global _legacy_read
    reg = {}
    try:
        reg = json.loads(REG_FILE.read_text())
    except Exception:
        reg = {}
    # One-shot: fold in whatever the pre-move (shared-mount) registry held for
    # this host, so a node restarting across the move still knows about the
    # groups its previous instance launched.
    if not _legacy_read:
        _legacy_read = True
        try:
            if LEGACY_REG_FILE.resolve() != REG_FILE.resolve():
                for pid, entry in json.loads(
                        LEGACY_REG_FILE.read_text()).items():
                    reg.setdefault(pid, entry)
        except Exception:
            pass
    return reg if isinstance(reg, dict) else {}


# Errnos a network filesystem raises for "someone else is touching this right
# now", as opposed to a real problem worth surfacing.
_TRANSIENT = {errno.EBUSY, errno.EACCES, errno.EPERM, errno.ENOTEMPTY,
              errno.EEXIST, errno.EAGAIN}


def _write(reg: dict) -> None:
    """Replace the registry, tolerating a network filesystem underneath.

    macOS smbfs refuses to rename over a file another process still has open
    (EBUSY), so the textbook write-temp-then-replace is not actually atomic on
    a share -- it just fails.  Retry briefly, then fall back to writing in
    place: a torn read degrades to an empty registry, which the environment
    scan still backstops, whereas raising here aborts the whole test run.
    """
    REG_DIR.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(reg, indent=2)
    # Per-pid temp name: two node processes on one box would otherwise write
    # the same temp path and clobber each other whenever flock is a no-op,
    # which it is on most network mounts.
    tmp = REG_FILE.with_name(f"{REG_FILE.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(blob)
        delay = 0.05
        for _ in range(5):
            try:
                tmp.replace(REG_FILE)
                return
            except OSError as exc:
                if exc.errno not in _TRANSIENT:
                    raise
                time.sleep(delay)
                delay *= 2
        REG_FILE.write_text(blob)          # last resort, non-atomic
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def _create_time(pid: int) -> float | None:
    """Kernel start time of `pid`, or None if it is gone.

    (pid, create_time) is the only pair that identifies a process across
    time: pids are recycled, and killing a recycled pid's group is how a
    cleanup routine turns into an outage.
    """
    try:
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def _same_process(pid: int, create_time) -> bool:
    if create_time is None:
        return _create_time(pid) is not None
    ct = _create_time(pid)
    # granularity of /proc clock ticks: compare loosely
    return ct is not None and abs(ct - float(create_time)) < 1.0


def _protected_pids() -> set:
    """This process and its ancestors -- never killable by a sweep."""
    keep = {os.getpid()}
    try:
        for p in psutil.Process(os.getpid()).parents():
            keep.add(p.pid)
    except Exception:
        pass
    return keep


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(pid: int, log=print, **meta) -> dict:
    """Record a live process group.  Call right after Popen/start.

    `pid` must be a session/group leader (started with start_new_session=True)
    for group kills to reach the whole tree; if it is not, kill_entry falls
    back to walking the process tree with psutil.

    Never raises. The caller has already started the process by the time this
    runs, so a registry error here must not abort an otherwise-fine test.
    A failure here costs tracking, not correctness: the process still carries
    AUDIOTOOL_RUN_ID, so `scan_untracked` reaps it regardless.
    """
    try:
        pgid = os.getpgid(pid)
    except Exception:
        pgid = pid
    entry = {
        "pid":         int(pid),
        "pgid":        int(pgid),
        "create_time": _create_time(pid),
        "owner_pid":   os.getpid(),
        "host":        _HOST,
        "started":     datetime.now().isoformat(timespec="seconds"),
        **meta,
    }
    try:
        with _Locked():
            reg = _read()
            reg[str(pid)] = entry
            _write(reg)
    except Exception as exc:                              # noqa: BLE001
        log(f"[procreg] could not record pid={pid} in {REG_FILE.name}: "
            f"{type(exc).__name__}: {exc} -- the run continues and the "
            f"environment scan will still reap it")
    return entry


def unregister(pid: int) -> None:
    """Drop a process group that has already exited (or been killed).

    Also never raises: this runs in cleanup paths, where an exception would
    replace the run's real result with a registry error.  A leftover entry is
    harmless -- the next sweep sees the pid is gone and prunes it.
    """
    try:
        with _Locked():
            reg = _read()
            if reg.pop(str(pid), None) is not None:
                _write(reg)
    except Exception:                                     # noqa: BLE001
        pass


def entries() -> list:
    """Every registered entry, dead ones included."""
    with _Locked():
        return list(_read().values())


def live_entries() -> list:
    """Registered entries whose leader is still the process we registered."""
    return [e for e in entries()
            if _same_process(e["pid"], e.get("create_time"))]


# ---------------------------------------------------------------------------
# Killing
# ---------------------------------------------------------------------------

def _tree_pids(pid: int) -> list:
    try:
        p = psutil.Process(pid)
        return [c.pid for c in p.children(recursive=True)] + [pid]
    except Exception:
        return [pid]


def kill_entry(entry: dict, grace: float = _TERM_GRACE, log=print) -> dict:
    """SIGTERM the whole group, then SIGKILL whatever is left.

    Returns {"pid", "killed": n, "forced": bool}.  Never raises; a group that
    was already gone reports killed=0.
    """
    pid  = int(entry["pid"])
    pgid = int(entry.get("pgid") or pid)
    leader_alive = _same_process(pid, entry.get("create_time"))

    # A dead leader is the interesting case, not a reason to stop: the role
    # script exiting is exactly when its decode pool gets orphaned, and the
    # orphans keep the leader's pgid.  Kill by group id, but -- since the
    # leader's identity can no longer vouch for the group -- only members that
    # still carry this run's AUDIOTOOL_RUN_ID, so a recycled pgid is safe.
    members = _group_members(pgid)
    if not leader_alive:
        want = entry.get("run_id")
        members = [m for m in members if _run_id_of(m)
                   and (not want or _run_id_of(m) == want)]
        if not members:
            return {"pid": pid, "killed": 0, "forced": False,
                    "note": "already gone"}

    # Never signal our own group: that would take down the node app with it.
    use_group = leader_alive
    try:
        if pgid in (os.getpgid(0), 0, 1):
            use_group = False
    except Exception:
        pass

    victims = members if not use_group else None

    def _signal(sig) -> int:
        if use_group:
            try:
                os.killpg(pgid, sig)
                return 1
            except ProcessLookupError:
                return 0
            except Exception:
                return 0
        n = 0
        for vp in victims or []:
            try:
                os.kill(vp, sig)
                n += 1
            except Exception:
                pass
        return n

    before = list(members) if use_group else list(victims or [])
    _signal(signal.SIGTERM)

    deadline = time.time() + max(0.0, grace)
    while time.time() < deadline:
        if not _alive_any(before):
            break
        time.sleep(0.2)

    forced = False
    if _alive_any(before):
        forced = True
        _signal(signal.SIGKILL)
        time.sleep(0.3)

    remaining = [p for p in before if _is_alive(p)]
    killed = len(before) - len(remaining)
    log(f"[procreg] reaped pgid={pgid} pid={pid} "
        f"run={entry.get('run_id')} role={entry.get('role')} "
        f"procs={killed}{' (SIGKILL)' if forced else ''}"
        + (f" STILL ALIVE: {remaining}" if remaining else ""))
    return {"pid": pid, "pgid": pgid, "killed": killed, "forced": forced,
            "remaining": remaining, "run_id": entry.get("run_id"),
            "role": entry.get("role")}


def _run_id_of(pid: int) -> str | None:
    """AUDIOTOOL_RUN_ID from a process's environment, if readable."""
    try:
        return psutil.Process(pid).environ().get(ENV_MARK)
    except Exception:
        return None


def _group_members(pgid: int) -> list:
    out = []
    for p in psutil.process_iter(["pid"]):
        try:
            if os.getpgid(p.pid) == pgid:
                out.append(p.pid)
        except Exception:
            continue
    return out


def _is_alive(pid: int) -> bool:
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def _alive_any(pids) -> bool:
    return any(_is_alive(p) for p in pids)


def kill_pid_group(pid: int, grace: float = _TERM_GRACE, log=print,
                   **meta) -> dict:
    """Kill the group of a pid we launched, registered or not.

    Works after the leader has exited: the registered entry (or `meta`)
    supplies the pgid and run_id needed to identify the survivors.
    """
    known = {}
    with _Locked():
        known = dict(_read().get(str(pid)) or {})
    try:
        pgid = os.getpgid(pid)
    except Exception:
        pgid = int(known.get("pgid") or pid)
    entry = {**known, **meta, "pid": pid, "pgid": pgid}
    entry.setdefault("create_time", known.get("create_time")
                     if known else _create_time(pid))
    res = kill_entry(entry, grace, log)
    unregister(pid)
    return res


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def sweep_stale(log=print, exclude_owner: bool = True) -> list:
    """Reap every registered group left over from a previous app instance.

    Call at node startup.  `exclude_owner` keeps groups launched by a still
    running app (our own pid, or another live node process) alive -- only
    entries whose owner is gone, or whose owner is us after a restart with
    the same pid, are reaped.
    """
    out = []
    me = os.getpid()
    protected = _protected_pids()
    for e in entries():
        pid = int(e["pid"])
        if pid in protected:
            continue
        if not _same_process(pid, e.get("create_time")):
            unregister(pid)                      # dead: just prune the entry
            continue
        owner = int(e.get("owner_pid") or 0)
        if exclude_owner and owner and owner != me and _is_alive(owner):
            continue                             # another live node owns it
        log(f"[procreg] stale test group from a previous run: pid={pid} "
            f"run={e.get('run_id')} role={e.get('role')} "
            f"started={e.get('started')}")
        out.append(kill_entry(e, log=log))
        unregister(pid)
    return out


def scan_untracked(log=print, run_id: str | None = None,
                   ignore_busy: bool = False, include_covered: bool = False,
                   node: str | None = None) -> list:
    """Find test descendants that no registry entry covers.

    Anything spawned under a test inherits AUDIOTOOL_RUN_ID, including a
    child that called setsid and escaped its parent's group.  This is the
    backstop for exactly those, and for processes started before the registry
    existed.  Returns entry-shaped dicts (not killed).

    `run_id`      restricts the scan to one run.
    `ignore_busy` drops the "somebody's run is still live" guard.  That guard
                  exists so a *booting* node cannot kill another node's
                  in-flight test; a deliberate Stop of a named run is the
                  exact opposite intent, and leaving the guard on there is
                  what let a playback daemon keep the speaker after Stop.
    `include_covered` also returns members of a tracked leader's tree.  A
                  group kill normally covers those, but after the leader has
                  exited the group is only identifiable by the env marker.
    `node`        restricts to processes launched by one node id (plus any
                  that carry no node id at all).  Two nodes commonly share a
                  host, and an unscoped sweep on one of them must not reach
                  into the other's live test.
    """
    protected = _protected_pids()
    live = live_entries()
    tracked = {int(e["pid"]) for e in live}
    # a tracked leader's whole group is covered
    covered = set(tracked)
    if not include_covered:
        for pid in tracked:
            covered.update(_tree_pids(pid))
    # Two nodes share this host and this registry (server + client on one
    # box is the normal deployment).  A run that is still tracked by ANY live
    # entry is somebody's live run: its escaped children are not strays, and
    # the other node booting must not kill them mid-test.
    busy_runs = set() if ignore_busy else {
        e.get("run_id") for e in live if e.get("run_id")}

    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        pid = p.info["pid"]
        if pid in protected or pid in covered:
            continue
        try:
            env = p.environ()
        except Exception:
            continue
        proc_run = env.get(ENV_MARK)
        if not proc_run or proc_run in busy_runs:
            continue
        if run_id and proc_run != run_id:
            continue
        proc_node = env.get("AUDIOTOOL_CLIENT_ID") or ""
        if node and proc_node and proc_node != node:
            continue
        try:
            rss = p.memory_info().rss
        except Exception:
            rss = 0
        found.append({
            "pid":         pid,
            "pgid":        (os.getpgid(pid) if _is_alive(pid) else pid),
            "create_time": p.info.get("create_time"),
            "run_id":      proc_run,
            "role":        env.get("AUDIOTOOL_ROLE") or "",
            "node":        env.get("AUDIOTOOL_CLIENT_ID") or "",
            "cmd":         " ".join(p.info.get("cmdline") or [])[:160],
            "rss_mb":      round(rss / 1e6, 1),
            "untracked":   True,
        })
    if found:
        log(f"[procreg] {len(found)} untracked test process(es) found by "
            f"environment scan")
    return found


def reap_untracked(log=print, run_id: str | None = None,
                   ignore_busy: bool = False, grace: float = _TERM_GRACE,
                   node: str | None = None) -> list:
    """Kill everything scan_untracked() finds (optionally one run only)."""
    out = []
    for e in scan_untracked(log=log, run_id=run_id, ignore_busy=ignore_busy,
                            include_covered=ignore_busy, node=node):
        out.append(kill_entry(e, grace=grace, log=log))
    return out


def run_survivors(run_id: str | None = None, node: str | None = None) -> list:
    """Live pids still carrying AUDIOTOOL_RUN_ID (optionally one run).

    The verification half of a stop: "the group was signalled" is not the
    same claim as "nothing is holding the speaker any more", and only the
    second one is what the operator sees.
    """
    protected = _protected_pids()
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        pid = p.info["pid"]
        if pid in protected:
            continue
        try:
            env = p.environ()
        except Exception:
            continue
        proc_run = env.get(ENV_MARK)
        if not proc_run or (run_id and proc_run != run_id):
            continue
        proc_node = env.get("AUDIOTOOL_CLIENT_ID") or ""
        if node and proc_node and proc_node != node:
            continue
        try:
            rss = p.memory_info().rss
        except Exception:
            rss = 0
        out.append({
            "pid":     pid,
            "run_id":  proc_run,
            "role":    env.get("AUDIOTOOL_ROLE") or "",
            "node":    env.get("AUDIOTOOL_CLIENT_ID") or "",
            "cmd":     " ".join(p.info.get("cmdline") or [])[:160],
            "rss_mb":  round(rss / 1e6, 1),
            "age_s":   round(time.time() - (p.info.get("create_time") or 0), 1),
        })
    return out


def reap_run(run_id: str | None = None, log=print, grace: float = 2.0,
             passes: int = 3, node: str | None = None) -> dict:
    """Kill everything belonging to `run_id` (or every run, if None).

    This is the Stop path, and it differs from the boot/idle sweeps in three
    ways that all matter:

      · it does not wait for the run to look finished.  `scan_untracked`'s
        busy-run guard protects an in-flight test from a *booting* node; a
        Stop is a deliberate order to end that same in-flight test, so the
        guard is dropped here (`ignore_busy`).  With it on, a role script's
        setsid'd daemon -- the mesh runner's, and anything else that leaves
        its parent's group -- survived Stop and kept the output device until
        the 60s reaper loop noticed.
      · it uses a short grace.  SIGTERM runs the role script's shutdown
        hooks, and those hooks flush a capture buffer to a network share
        while the PortAudio callback thread is still feeding the speaker.
        A five-second grace is five more seconds of audible transmission.
      · it re-scans.  A group kill races a process that is still forking
        (a decode pool respawning a worker), so one pass can miss a child
        that did not exist when the member list was taken.

    Returns {"killed", "registered", "untracked", "remaining"}.
    """
    registered, untracked = [], []
    for e in live_entries():
        if int(e["pid"]) in _protected_pids():
            continue
        if run_id and e.get("run_id") != run_id:
            continue
        if node and e.get("node") and e.get("node") != node:
            continue
        registered.append(kill_entry(e, grace=grace, log=log))
        unregister(int(e["pid"]))

    for i in range(max(1, passes)):
        found = reap_untracked(log=log, run_id=run_id, ignore_busy=True,
                               grace=(grace if i == 0 else 0.0), node=node)
        untracked += found
        if not found:
            break

    remaining = run_survivors(run_id, node=node)
    killed = (sum(r.get("killed", 0) for r in registered)
              + sum(r.get("killed", 0) for r in untracked))
    if remaining:
        log(f"[procreg] stop of run={run_id or '*'} left {len(remaining)} "
            f"process(es) alive: {[r['pid'] for r in remaining]}")
    return {"killed": killed, "registered": registered,
            "untracked": untracked, "remaining": remaining}


def reap_all(log=print) -> dict:
    """Everything: registered groups owned by us, plus untracked strays."""
    killed = []
    for e in live_entries():
        if int(e["pid"]) in _protected_pids():
            continue
        killed.append(kill_entry(e, log=log))
        unregister(int(e["pid"]))
    strays = reap_untracked(log=log)
    return {"registered": killed, "untracked": strays}


def status() -> dict:
    """What the registry and the scan currently see (for /api/procs)."""
    live = live_entries()
    for e in live:
        try:
            e["rss_mb"] = round(sum(
                psutil.Process(p).memory_info().rss for p in _tree_pids(e["pid"])
            ) / 1e6, 1)
            e["procs"] = len(_tree_pids(e["pid"]))
        except Exception:
            pass
    return {"tracked": live, "untracked": scan_untracked(log=lambda *_a: None),
            "host": _HOST, "reg_dir": str(REG_DIR)}


def report(run_id: str | None = None) -> dict:
    """Compact roll-up of this node's test processes, for the server.

    `status()` answers "what should I clean up here"; this answers "what is
    this client actually running right now", which is what the operator at
    the server needs before and after pressing Stop.  Cheap enough to send
    on a timer: one process_iter, no tree walks per entry.

    Every env-marked process is listed, tracked or not -- the ones that
    escaped their group are precisely the ones worth seeing.
    """
    live = live_entries()
    tracked_pids = {int(e["pid"]) for e in live}
    procs = run_survivors(run_id)
    for pr in procs:
        pr["tracked"] = pr["pid"] in tracked_pids
    by_run: dict = {}
    for pr in procs:
        r = by_run.setdefault(pr["run_id"], {"procs": 0, "rss_mb": 0.0,
                                             "pids": [], "roles": []})
        r["procs"] += 1
        r["rss_mb"] = round(r["rss_mb"] + pr["rss_mb"], 1)
        r["pids"].append(pr["pid"])
        if pr["role"] and pr["role"] not in r["roles"]:
            r["roles"].append(pr["role"])
    return {
        "host":      _HOST,
        "reg_dir":   str(REG_DIR),
        "ts":        time.time(),
        "n_procs":   len(procs),
        "rss_mb":    round(sum(p["rss_mb"] for p in procs), 1),
        "runs":      by_run,
        "procs":     sorted(procs, key=lambda d: -d["rss_mb"])[:40],
        "tracked":   [{"pid": e["pid"], "run_id": e.get("run_id"),
                       "role": e.get("role"), "test": e.get("test"),
                       "started": e.get("started")} for e in live],
    }


if __name__ == "__main__":      # python3 procreg.py [list|reap|sweep]
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        print(json.dumps(status(), indent=2, default=str))
    elif cmd == "sweep":
        # same pair the node runs at boot: registered groups, then strays
        sweep_stale()
        reap_untracked()
    elif cmd == "reap":
        print(json.dumps(reap_all(), indent=2, default=str))
    elif cmd == "report":
        print(json.dumps(report(), indent=2, default=str))
    elif cmd == "stop":
        run = sys.argv[2] if len(sys.argv) > 2 else None
        print(json.dumps(reap_run(run), indent=2, default=str))
    else:
        print(f"usage: {sys.argv[0]} [list|report|reap|stop <run_id>|sweep]")
