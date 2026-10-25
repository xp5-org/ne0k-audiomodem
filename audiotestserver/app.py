# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""
app.py  –  Unified server/client node.

Every node runs a Flask+SocketIO *server* on port 8001 (local web UI).
Client nodes additionally maintain a socketio *client* connection to the server.

Entry points
------------
  run_node(role="server",    node_id="server",  host="0.0.0.0", port=8001)
  run_node(role="client", node_id="client1", host="0.0.0.0", port=8001,
           server_ip="192.168.1.10", server_port=8001)

CLI (main block at bottom):
  python node_core.py server
  python node_core.py client --id client1 --server 192.168.1.10
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
from flask import Flask, render_template, jsonify, request, send_from_directory, abort
from flask_socketio import SocketIO, emit, join_room
import socketio as sio_module          # python-socketio *client*

import procreg
import shared

# ---------------------------------------------------------------------------
# Flask / SocketIO server (runs on every node)
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "audiotool-node-secret"
sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ---------------------------------------------------------------------------
# Mutable node state
# ---------------------------------------------------------------------------

state: dict = {
    # identity
    "role":      "client",   # "server" | "client"
    "node_id":   "node1",
    "port":      8001,
    # server connection (client role only)
    "server_ip":    None,
    "server_port":  8001,
    "connected": False,      # True  = SocketIO client connected to server
    # audio
    "selected_input":  None,
    "selected_output": None,
    # runtime
    "status":      "disconnected",
    "logs":        [],
    "current_run": None,
    # Latest self-scan of this node's test processes (procreg.report()).
    # Clients push this to the server so the operator can see leftover pids
    # on every node, not just the one whose web UI they happen to have open.
    "procs":       {},
}

# server-only registry (populated only when role == "server")
_clients:     dict[str, dict] = {}   # node_id  -> live info
_client_sids: dict[str, str]  = {}   # node_id  -> socket SID
_sid_to_node: dict[str, str]  = {}   # SID      -> node_id
_coord:       dict[str, dict] = {}   # run_id   -> {signal: [node_ids]}
_coord_ts:    dict[str, dict] = {}   # run_id   -> {signal: {node: server_t}}
_coord_lock   = threading.Lock()

# Outstanding dispatch for the current run (server role only). Tracks who owes
# a result so a node that is dispatched and never reports leaves an explicit
# gap in the run directory rather than silently fewer participants.
_run_pending: dict = {"run_id": None, "nodes": set()}
_run_pending_lock = threading.Lock()

# Active subprocesses (both roles)
_active_procs:    dict[str, subprocess.Popen] = {}   # role_key -> proc
_proc_gen:        dict[str, int] = {}               # role_key -> generation counter
_proc_runs:       dict[str, str] = {}               # role_key -> run_id
_proc_lock        = threading.Lock()

# Last stop summary (what was killed, what survived) — surfaced by /api/stop
# and pushed to the server by a client, so "stopped" is a verified claim
# rather than "the signal was sent".
_last_stop: dict = {}
_server_proc_lock    = threading.Lock()   # extra lock for server's own proc reference

# SocketIO *client* (client role only)
_server_client:      sio_module.Client | None = None
_server_client_lock  = threading.Lock()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str, level: str = "info", client_id: str | None = None):
    cid = client_id or state["node_id"]
    entry = {
        "ts":     datetime.now().isoformat(timespec="seconds"),
        "level":  level,
        "client": cid,
        "msg":    msg,
    }
    state["logs"].append(entry)
    if len(state["logs"]) > 500:
        state["logs"].pop(0)
    print(f"[{entry['ts']}][{level.upper()}][{cid}] {msg}")

    # server: broadcast to dashboard
    if state["role"] == "server":
        sio.emit("server_log", entry)

    # Client: forward to server if connected
    if state["role"] == "client" and state["connected"] and _server_client:
        try:
            _server_client.emit("client_log", {"level": level, "msg": msg,
                                             "client_id": state["node_id"]})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _backend_inventory() -> dict:
    """This node's CPU tag and C-core status, for the registration payload.

    Imported lazily and defensively: `tests/` is a sibling package that a
    stripped deployment may not carry, and failing to register because the
    inventory could not be built would be a far worse outcome than registering
    without it.  An absent inventory reads as "unknown" at the preflight gate.
    """
    try:
        sys.path.insert(0, str(shared.TESTS_DIR))
        import cbackend
        return {"host_tag": cbackend.host_tag(),
                "backends": cbackend.inventory()}
    except Exception as exc:                                    # noqa: BLE001
        _log(f"could not build C backend inventory: {exc}", "warn")
        return {"host_tag": "unknown", "backends": []}


# Backend key -> the (env var, value) that selects it.  Mirrors
# cbackend.BACKENDS[...].env so a run can name backends symbolically
# ("c_backends": ["rs255"]) instead of hand-setting env vars, and so the UI has
# a fixed list to offer.  Kept as a literal because the server must answer even
# when tests/ is not importable.
_CBACKEND_ENV = {
    "css_mf":     ("CSS_MF_BACKEND", "c"),
    "ofdm_acq":   ("OFDM_ACQ_BACKEND", "c"),
    "ofdm_acq32": ("OFDM_ACQ_BACKEND", "c"),
    "rs255":      ("RS_BACKEND", "c"),
}


def _cbackend_requested(env: dict) -> list[str]:
    """Backend keys an env mapping selects.  Delegates to cbackend when it can
    be imported so the precision-selection rule lives in exactly one place."""
    try:
        sys.path.insert(0, str(shared.TESTS_DIR))
        import cbackend
        return cbackend.requested(env)
    except Exception:                                           # noqa: BLE001
        keys = []
        for key, (var, want) in _CBACKEND_ENV.items():
            if env.get(var, "").lower() != want:
                continue
            is32 = env.get("OFDM_ACQ_LIB", "") == "32"
            if key == "ofdm_acq" and is32:
                continue
            if key == "ofdm_acq32" and not is32:
                continue
            keys.append(key)
        return keys


def _backend_problems(nid: str, env_keys: list[str]) -> list[str]:
    """Problems that would stop node `nid` from delivering `env_keys`.

    Grades the inventory the node reported at registration against the
    backends this run asks for.  A node that reported no inventory is a
    problem in itself when the run requests C: it may be an older node that
    would silently fall back, and that is precisely what cannot be allowed
    through.
    """
    if not env_keys:
        return []
    info = _clients.get(nid, {})
    inv  = info.get("backends")
    if not inv:
        return [f"{nid}: reported no C backend inventory "
                f"(host {info.get('host_tag', 'unknown')}) — cannot confirm it "
                f"would not silently fall back"]
    by_key = {r.get("key"): r for r in inv}
    out = []
    for key in env_keys:
        rec = by_key.get(key)
        if rec is None:
            out.append(f"{nid}: does not know backend '{key}'")
        elif not rec.get("ok"):
            out.append(f"{nid} [{info.get('host_tag','?')}]: {rec.get('reason')}")
        elif rec.get("stale"):
            out.append(f"{nid} [{info.get('host_tag','?')}]: {key} is stale vs "
                       f"{', '.join(rec['stale'])}")
    return out


def _server_only(fn):
    """Decorator: returns 403 JSON when this node is not the server."""
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if state["role"] != "server":
            return jsonify({"error": "This node is not the server"}), 403
        return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Test execution  (shared by server self-run and client dispatch)
# ---------------------------------------------------------------------------

def _exec_test(test_path: str, params: dict):
    """
    Run a test script in a subprocess.  Works identically for server and client
    nodes; the only difference is how the result is reported back.
    """
    run_id   = params.get("run_id", "unknown")
    role_key = params.get("role") or test_path    # unique key per concurrent role

    # Each invocation gets a unique generation id. The finally block only
    # cleans _active_procs when it is still the current generation, so a
    # late-exiting cancelled thread cannot evict its successor's proc.
    with _proc_lock:
        _proc_gen[role_key] = _proc_gen.get(role_key, 0) + 1
        my_gen = _proc_gen[role_key]
        stale  = _active_procs.pop(role_key, None)
        _proc_runs.pop(role_key, None)
    if stale and stale.poll() is None:
        _log(f"Evicting stale proc role={role_key!r} (gen={my_gen-1}) before re-run", "warn")
        # Kill the GROUP: the role script's mp capture worker and decode pool
        # outlive a terminate() aimed at the leader alone.
        procreg.kill_pid_group(stale.pid, log=lambda m: _log(m, "warn"))
        try:
            stale.wait(timeout=3)
        except Exception:
            pass

    state["status"]      = "running"
    state["current_run"] = run_id
    _log(f"Starting test: {test_path}  run_id={run_id}  role={role_key}")

    local_run_dir = shared.RESULTS_DIR / run_id
    local_run_dir.mkdir(parents=True, exist_ok=True)

    server_url = (
        params.get("server_url")
        or (f"http://127.0.0.1:{state['port']}"
            if state["role"] == "server"
            else f"http://{state['server_ip']}:{state['server_port']}")
    )

    env = {
        **os.environ,
        "AUDIOTOOL_CLIENT_ID":     state["node_id"],
        "AUDIOTOOL_INPUT_DEVICE":  ("" if state["selected_input"]  is None
                                    else str(state["selected_input"])),
        "AUDIOTOOL_OUTPUT_DEVICE": ("" if state["selected_output"] is None
                                    else str(state["selected_output"])),
        "AUDIOTOOL_PARAMS":        json.dumps(params),
        "AUDIOTOOL_ROLE":          str(params.get("role", "")),
        "AUDIOTOOL_RUN_ID":        run_id,
        "AUDIOTOOL_SERVER_URL":    server_url,
        "AUDIOTOOL_RUN_DIR":       str(shared.RESULTS_DIR / run_id),
        # params["env"] carries the run's backend selection (CSS_MF_BACKEND=c,
        # …), already preflighted against every participant before dispatch.
        # Applied last so a run can override an operator's shell setting and
        # the whole fleet demonstrably runs one configuration.
        **{str(k): str(v) for k, v in (params.get("env") or {}).items()},
    }

    proc = None
    try:
        # start_new_session: the script leads its own process group, so every
        # descendant it spawns (mp capture worker, ProcessPoolExecutor decode
        # workers, helper Popens) can be killed in one signal.
        proc = subprocess.Popen(
            [sys.executable, test_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        with _proc_lock:
            _active_procs[role_key] = proc
            _proc_runs[role_key]    = run_id
        procreg.register(proc.pid, log=lambda m: _log(m, "warn"),
                         run_id=run_id, role=str(params.get("role", "")),
                         test=Path(test_path).stem, node=state["node_id"],
                         role_key=role_key)

        # Read the log on a pump thread rather than inline.  A leaked child
        # inherits the stdout pipe, so `for line in proc.stdout` does NOT end
        # when the script does -- it blocks until the last orphan exits, which
        # for an orphaned decode pool is never.  The run then never reported a
        # result even though the test itself had finished.  Wait on the
        # PROCESS; the pipe drains when the group is reaped in `finally`.
        output_lines: list[str] = []

        def _pump():
            try:
                for line in proc.stdout:
                    output_lines.append(line.rstrip())
                    _log(f"[test] {line.rstrip()}")
            except Exception:
                pass

        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()
        proc.wait()
        pump.join(timeout=5.0)
        if pump.is_alive():
            _log(f"{Path(test_path).stem} exited but a child still holds its "
                 f"output pipe — leftover processes are being reaped", "warn")

        # A negative returncode is a SIGNAL, not a crash: the stop button and
        # the stale-proc eviction both SIGTERM the child.  Label it so a
        # stopped node is distinguishable from one that failed on its own.
        killed_by = -proc.returncode if (proc.returncode or 0) < 0 else None
        result = {
            "run_id":     run_id,
            "client_id":  state["node_id"],
            "test":       Path(test_path).stem,
            "returncode": proc.returncode,
            "status":     ("ok" if proc.returncode == 0
                           else "terminated" if killed_by else "error"),
            **({"signal": killed_by,
                "error": f"test process terminated by signal {killed_by}"}
               if killed_by else {}),
            "output":     "\n".join(output_lines[-100:]),
            "ts":         datetime.now().isoformat(),
        }

        # Load structured per-test result file if the script wrote one
        result_file = local_run_dir / f"result_{state['node_id']}.json"
        if result_file.exists():
            try:
                result["data"] = json.loads(result_file.read_text())
            except Exception:
                pass

        _report_result(result, run_id)

    except Exception as exc:
        _log(f"Test execution error: {exc}", "error")
        _report_result({
            "run_id":    run_id,
            "client_id": state["node_id"],
            "status":    "error",
            "error":     str(exc),
            "ts":        datetime.now().isoformat(),
        }, run_id)
    finally:
        # The role script exiting does NOT end the test: an orphaned decode
        # pool or capture worker keeps its numpy heap and the audio device
        # for as long as the machine is up.  Sweep the group unconditionally,
        # whether the script finished, failed, or was stopped.
        if proc is not None:
            res = procreg.kill_pid_group(proc.pid, grace=2.0, run_id=run_id,
                                         log=lambda m: _log(m, "debug"))
            n = res.get("killed", 0)
            # ...and anything that escaped the group by starting its own
            # session (the mesh daemon does exactly that).
            n += sum(r.get("killed", 0) for r in procreg.reap_untracked(
                run_id=run_id, log=lambda m: _log(m, "debug")))
            if n:
                _log(f"Reaped {n} leftover process(es) from "
                     f"{Path(test_path).stem} (run {run_id})", "warn")
        with _proc_lock:
            # Only remove our slot — don't evict a newer generation's proc
            if _proc_gen.get(role_key) == my_gen:
                _active_procs.pop(role_key, None)
                _proc_gen.pop(role_key, None)
                _proc_runs.pop(role_key, None)
        if not _active_procs:
            state["status"]      = "idle"
            state["current_run"] = None


def _run_pending_arm(run_id: str, nodes) -> None:
    """Record who was dispatched for `run_id` and still owes a result."""
    with _run_pending_lock:
        _run_pending["run_id"] = run_id
        _run_pending["nodes"] = set(nodes)


def _run_pending_clear(run_id: str, nid: str) -> None:
    with _run_pending_lock:
        if _run_pending["run_id"] == run_id:
            _run_pending["nodes"].discard(nid)


def _mark_node_lost(nid: str, why: str) -> None:
    """Write a stub result for a dispatched node that never reported, so the
    gap is visible in the results list and in any analysis pass instead of
    the node being silently absent from the run directory."""
    with _run_pending_lock:
        run_id = _run_pending["run_id"]
        if not run_id or nid not in _run_pending["nodes"]:
            return
        _run_pending["nodes"].discard(nid)
    result = {
        "run_id":    run_id,
        "client_id": nid,
        "status":    "lost",
        "error":     why,
        "ts":        datetime.now().isoformat(),
    }
    try:
        run_dir = shared.RESULTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"result_{nid}_lost_{int(time.time())}.json").write_text(
            json.dumps(result, indent=2))
    except Exception as exc:
        _log(f"Could not write lost-node stub for {nid}: {exc}", "error")
    _log(f"Node {nid} never reported for run {run_id}: {why}", "warn",
         client_id=nid)
    sio.emit("test_result_update", {"client_id": nid, **result})


def _render_css_spectrograph_bg(run_id: str, node_id: str) -> None:
    """Render a css_listen node's spectrograph PNG server-side, off-thread.

    Rendering scales with capture duration (~10-17 min for a 400-580s WAV),
    so it runs server-side rather than on the listen node. BASE_DIR is a
    shared SMB mount, so the server can read the listen node's WAV straight
    off disk once its result lands; it doesn't need the file transferred.
    """
    try:
        run_dir     = shared.RESULTS_DIR / run_id
        wav_path    = run_dir / f"{node_id}_input.wav"
        result_path = run_dir / f"decode_result_{node_id}.json"
        out_path    = run_dir / f"{node_id}_spectrograph.png"
        if not wav_path.exists() or out_path.exists():
            return

        info = None
        info_path = run_dir / "css_info.json"
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text())
            except Exception:
                info = None

        decode_result = None
        if result_path.exists():
            try:
                decode_result = json.loads(result_path.read_text())
            except Exception:
                decode_result = None

        tests_dir = str(shared.TESTS_DIR)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        from analysis import make_css_spectrograph_png

        ok = make_css_spectrograph_png(wav_path, out_path, info=info,
                                       result=decode_result)
        if ok and decode_result is not None:
            decode_result["spectrograph_png"] = str(out_path)
            result_path.write_text(json.dumps(decode_result, indent=2))
    except Exception as exc:
        _log(f"CSS spectrograph render failed ({node_id}/{run_id}): {exc}", "warn")


def _maybe_render_css_spectrograph(result: dict, run_id: str, node_id: str) -> None:
    """Kick off a background spectrograph render for a css_listen result.

    Server role only — a client node never has another node's WAV to render,
    and shouldn't import matplotlib/scipy just to check.
    """
    if state["role"] != "server":
        return
    if result.get("test") != "css_listen":
        return
    threading.Thread(target=_render_css_spectrograph_bg, args=(run_id, node_id),
                     daemon=True).start()


def _report_result(result: dict, run_id: str):
    """Deliver a test result to whoever needs it."""
    node_id = state["node_id"]

    if state["role"] == "server":
        # server node: emit directly into the SocketIO server bus
        sio.emit("test_result_update", {"client_id": node_id, **result})
        run_dir = shared.RESULTS_DIR / run_id
        run_dir.mkdir(exist_ok=True)
        (run_dir / f"result_{node_id}_{int(time.time())}.json").write_text(
            json.dumps(result, indent=2)
        )
        _run_pending_clear(run_id, node_id)
        _maybe_render_css_spectrograph(result, run_id, node_id)
    else:
        # Client node: send back to server
        if _server_client:
            _server_client.emit("test_result", result)


# ---------------------------------------------------------------------------
# Process stop helper (both roles)
# ---------------------------------------------------------------------------

# Grace between SIGTERM and SIGKILL on the Stop path.  Deliberately short:
# SIGTERM lets the role script flush its capture and result file, but the
# PortAudio callback thread keeps feeding the speaker for the whole window,
# so every extra second of grace is an extra second of audible transmission
# after the operator pressed Stop.  testlib's shutdown handler silences the
# output stream first thing, so this is a backstop, not the mechanism.
_STOP_GRACE = 1.5


def _do_stop(run_id: str | None = None, grace: float = _STOP_GRACE) -> dict:
    """End this node's test processes now and report what survived.

      · Does not give up when `_active_procs` is empty: a role script that
        has already exited can still have live descendants (e.g. the mesh
        runner's setsid'd daemon) holding the output device.
      · Sweeps by run marker, not only by process group: a child that called
        setsid is in nobody's group, and AUDIOTOOL_RUN_ID is the only thing
        that still identifies it.
      · Verifies: survivors are counted and named rather than assumed gone.
    """
    global _last_stop
    with _proc_lock:
        procs = dict(_active_procs)
        runs  = dict(_proc_runs)

    # Which runs are we ending?  An explicit run_id wins (the server names the
    # run it dispatched); otherwise everything this node believes it started.
    targets: set[str] = set()
    if run_id:
        targets.add(run_id)
    else:
        targets.update(r for r in runs.values() if r)
        if state.get("current_run"):
            targets.add(state["current_run"])

    killed = 0
    details: list[dict] = []

    # 1. The groups we launched, by pid — fastest and most precise route.
    for role_key, proc in procs.items():
        if run_id and runs.get(role_key) not in (None, run_id):
            continue
        if proc and proc.poll() is None:
            _log(f"Terminating process group: {role_key}", "warn")
            try:
                res = procreg.kill_pid_group(
                    proc.pid, grace=grace, run_id=runs.get(role_key),
                    log=lambda m: _log(m, "warn"))
                killed += res.get("killed", 0)
                details.append({"role_key": role_key, **res})
                _log(f"Stopped {role_key}: {res.get('killed', 0)} process(es)"
                     + (" (SIGKILL)" if res.get("forced") else ""), "warn")
            except Exception as exc:
                _log(f"Stop error ({role_key}): {exc}", "error")

    # 2. Everything else carrying the run marker, group or no group.  With no
    #    known run this still sweeps every marked stray (run_id=None means
    #    "any run" to reap_run), which is what makes Stop work after the role
    #    script has already exited and left a daemon behind.  That unscoped
    #    sweep is limited to processes this node launched: server and client
    #    routinely share one box, and "stop my tests" must not end the other
    #    node's live run.
    for target in (sorted(targets) or [None]):
        try:
            res = procreg.reap_run(target, grace=grace,
                                   node=(None if target else state["node_id"]),
                                   log=lambda m: _log(m, "debug"))
            killed += res.get("killed", 0)
            details.append({"run_id": target, "swept": res.get("killed", 0),
                            "remaining": res.get("remaining", [])})
        except Exception as exc:
            _log(f"Stop sweep error (run={target}): {exc}", "error")

    remaining = procreg.run_survivors(
        run_id, node=(None if (run_id or targets) else state["node_id"]))
    summary = {
        "node":      state["node_id"],
        "run_id":    run_id,
        "runs":      sorted(targets),
        "killed":    killed,
        "remaining": remaining,
        "ts":        time.time(),
    }
    _last_stop = summary

    if killed:
        _log(f"Stop: reaped {killed} test process(es)"
             + (f" for run {run_id}" if run_id else ""), "warn")
    elif not procs:
        _log("Stop requested but no test processes were running")
    if remaining:
        _log(f"Stop: {len(remaining)} process(es) SURVIVED the reap: "
             + ", ".join(f"pid={r['pid']} role={r['role'] or '-'} "
                         f"run={r['run_id']}" for r in remaining[:6]), "error")

    state["status"]      = "idle"
    state["current_run"] = None
    return summary


# ---------------------------------------------------------------------------
# Leftover-process reaping
#
# Three independent nets, because each one has a hole the others cover:
#   _boot_reap        previous instance died without stopping its tests
#   _install_exit_reaper   this instance is going down now
#   _reaper_loop      a run ended but something escaped its process group
# ---------------------------------------------------------------------------

def _boot_reap() -> None:
    try:
        killed = procreg.sweep_stale(log=lambda m: _log(m, "warn"))
        strays = procreg.reap_untracked(log=lambda m: _log(m, "warn"))
        n = sum(k.get("killed", 0) for k in killed + strays)
        if n:
            _log(f"Startup: reaped {n} process(es) left by a previous run",
                 "warn")
    except Exception as exc:
        _log(f"Startup reap failed: {exc}", "error")


def _install_exit_reaper() -> None:
    """Kill every test group when this node exits, however it exits."""
    import atexit
    import signal as _sig

    def _reap_all(_why: str = "exit"):
        try:
            procreg.reap_all(log=lambda m: _log(m, "warn"))
        except Exception:
            pass

    atexit.register(_reap_all)

    def _handler(signum, _frame):
        _log(f"Node received signal {signum} -- stopping test processes",
             "warn")
        _reap_all(f"signal {signum}")
        os._exit(128 + int(signum))

    for s in (_sig.SIGTERM, _sig.SIGINT, _sig.SIGHUP):
        try:
            _sig.signal(s, _handler)
        except (ValueError, OSError, AttributeError):
            pass


def _active_run_ids() -> set:
    runs = {e.get("run_id") for e in procreg.live_entries()}
    if state.get("current_run"):
        runs.add(state["current_run"])
    with _proc_lock:
        procs = list(_active_procs.values())
    if procs:
        runs.add(state.get("current_run"))
    return {r for r in runs if r}


def _reaper_loop(period: float = 60.0, min_age: float = 30.0) -> None:
    """Kill test descendants whose run is over.

    A child that calls setsid (meshrun_node's daemon) leaves its parent's
    group, so a group kill never reaches it; it is only recognisable by the
    AUDIOTOOL_RUN_ID it inherited.  Runs still in flight are left alone, and
    so is anything younger than `min_age` (a run that has just been launched
    may not be registered yet).
    """
    last_err: str | None = None
    err_count = 0
    while True:
        time.sleep(period)
        try:
            active = _active_run_ids()
            now    = time.time()
            for e in procreg.scan_untracked(log=lambda *_a: None):
                if e.get("run_id") in active:
                    continue
                ct = e.get("create_time") or 0
                if ct and now - ct < min_age:
                    continue
                _log(f"Orphan from finished run {e.get('run_id')}: pid="
                     f"{e['pid']} rss={e.get('rss_mb')}MB — {e.get('cmd')}",
                     "warn")
                procreg.kill_entry(e, log=lambda m: _log(m, "warn"))
            last_err = None
        except Exception as exc:
            # Do not repeat the same failure every 60s forever.  A permission
            # problem on the registry produced one identical error line per
            # minute for hours, which buried everything else in the log.
            msg = f"{type(exc).__name__}: {exc}"
            if msg != last_err:
                last_err = msg
                err_count = 1
                _log(f"Reaper loop error: {msg}", "error")
            else:
                err_count += 1
                if err_count % 30 == 0:
                    _log(f"Reaper loop error (x{err_count}): {msg}", "error")


# ---------------------------------------------------------------------------
# Stats broadcast loop (both roles)
# ---------------------------------------------------------------------------

# How often a node re-scans its own process table for the server's benefit.
# `procreg.report()` reads every process's environment, which is cheap on
# Linux (/proc) and not cheap on macOS (one sysctl per process), so it runs on
# its own slower cadence rather than with the 2s stats tick.
_PROC_REPORT_BUSY = 6.0     # seconds, while a test is running
_PROC_REPORT_IDLE = 30.0    # seconds, while idle


def _proc_report_local() -> dict:
    """This node's process roll-up, cached against the scan cost."""
    try:
        return procreg.report()
    except Exception as exc:
        return {"host": "?", "n_procs": 0, "rss_mb": 0.0, "runs": {},
                "procs": [], "tracked": [], "error": str(exc)}


def _stats_loop():
    next_procs = 0.0
    while True:
        time.sleep(2)

        # --- process roll-up (own cadence) ---------------------------------
        if time.time() >= next_procs:
            busy = bool(_active_procs) or state["status"] == "running"
            next_procs = time.time() + (_PROC_REPORT_BUSY if busy
                                        else _PROC_REPORT_IDLE)
            rep = _proc_report_local()
            state["procs"] = rep
            if state["role"] == "server":
                if state["node_id"] in _clients:
                    _clients[state["node_id"]]["procs"] = rep
                sio.emit("client_procs", {"client_id": state["node_id"],
                                          "procs": rep})
            elif state["connected"] and _server_client:
                try:
                    _server_client.emit("procs_report", rep)
                except Exception:
                    pass

        try:
            s = {
                "cpu_pct":     psutil.cpu_percent(interval=None),
                "mem_pct":     psutil.virtual_memory().percent,
                "mem_used_mb": int(psutil.virtual_memory().used / 1e6),
                "disk_pct":    psutil.disk_usage(str(shared.BASE_DIR)).percent,
                "status":      state["status"],
                "current_run": state["current_run"],
            }
        except Exception:
            continue

        node_id = state["node_id"]

        if state["role"] == "server":
            # Update self in registry + push to dashboard
            if node_id in _clients:
                _clients[node_id]["stats"]     = s
                _clients[node_id]["last_seen"] = time.time()
            sio.emit("client_stats", {
                "client_id": node_id,
                "stats":     s,
                "status":    state["status"],
            })
        else:
            # Forward to server
            if state["connected"] and _server_client:
                try:
                    _server_client.emit("stats_update", {
                        "stats":  s,
                        "status": state["status"],
                        "client_id": node_id,
                    })
                except Exception:
                    pass


# ===========================================================================
# SocketIO SERVER handlers  (active on every node; meaningful on server)
# ===========================================================================

@sio.on("connect")
def on_connect():
    _log(f"Incoming socket connection sid={request.sid}", "debug")


@sio.on("disconnect")
def on_disconnect():
    sid = request.sid
    nid = _sid_to_node.pop(sid, None)
    if nid:
        _client_sids.pop(nid, None)
        _clients.pop(nid, None)
        shared.unregister_client(nid)
        _log(f"Node disconnected: {nid}", "warn")
        sio.emit("client_left", {"client_id": nid})
        _mark_node_lost(nid, "client disconnected before reporting a result")


@sio.on("register")
def on_register(data):
    if state["role"] != "server":
        return   # non-server nodes ignore registration events
    sid = request.sid
    nid = data.get("client_id", f"unknown_{sid[:6]}")

    old_sid = _client_sids.get(nid)
    if old_sid and old_sid != sid:
        _sid_to_node.pop(old_sid, None)

    _client_sids[nid] = sid
    _sid_to_node[sid]  = nid

    info = {
        "client_id":       nid,
        "ip":              data.get("ip", request.remote_addr),
        "port":            data.get("port", 8001),
        "devices":         data.get("devices", {}),
        "selected_input":  data.get("selected_input"),
        "selected_output": data.get("selected_output"),
        "status":          "idle",
        "connected_at":    datetime.now().isoformat(timespec="seconds"),
        "last_seen":       time.time(),
        "stats":           {},
        # CPU type and per-core load status, used by the pre-dispatch gate in
        # api_run_test.  Absent from older nodes, which the gate treats as
        # "unconfirmed" rather than "fine".
        "host_tag":        data.get("host_tag", "unknown"),
        "backends":        data.get("backends", []),
    }
    _clients[nid] = info
    shared.register_client(nid, info)

    join_room(nid)
    _log(f"Node registered: {nid} from {info['ip']}:{info['port']}")
    sio.emit("client_joined", {"client_id": nid, "info": info})
    emit("registered", {"ok": True, "server_time": time.time()})


@sio.on("stats_update")
def on_stats_update(data):
    if state["role"] != "server":
        return
    sid = request.sid
    nid = _sid_to_node.get(sid)
    if nid and nid in _clients:
        _clients[nid]["stats"]     = data.get("stats", {})
        _clients[nid]["last_seen"] = time.time()
        _clients[nid]["status"]    = data.get("status", _clients[nid].get("status", "idle"))
        sio.emit("client_stats", {
            "client_id": nid,
            "stats":     data.get("stats", {}),
            "status":    _clients[nid]["status"],
        })


@sio.on("procs_report")
def on_procs_report(data):
    """A client told us which test processes it is holding.

    Kept per-node so Stop can be verified rather than assumed.
    """
    if state["role"] != "server":
        return
    nid = _sid_to_node.get(request.sid)
    if not nid or nid not in _clients:
        return
    _clients[nid]["procs"] = data or {}
    _clients[nid]["last_seen"] = time.time()
    sio.emit("client_procs", {"client_id": nid, "procs": data or {}})


@sio.on("stop_report")
def on_stop_report(data):
    """Outcome of a stop we asked a client for: killed count and survivors."""
    if state["role"] != "server":
        return
    nid = _sid_to_node.get(request.sid, "?")
    data = data or {}
    if nid in _clients:
        _clients[nid]["last_stop"] = data
    left = data.get("remaining") or []
    _log(f"Stop on {nid}: reaped {data.get('killed', 0)} process(es)"
         + (f"; {len(left)} STILL ALIVE "
            f"({', '.join(str(r.get('pid')) for r in left[:6])})" if left else ""),
         "error" if left else "warn", client_id=nid)
    sio.emit("client_stop_report", {"client_id": nid, **data})


@sio.on("client_log")
def on_client_log(data):
    if state["role"] != "server":
        return
    sid = request.sid
    nid = _sid_to_node.get(sid, "?")
    _log(data.get("msg", ""), level=data.get("level", "info"), client_id=nid)


@sio.on("device_selected")
def on_device_selected(data):
    if state["role"] != "server":
        return
    sid = request.sid
    nid = _sid_to_node.get(sid)
    if nid and nid in _clients:
        _clients[nid]["selected_input"]  = data.get("input")
        _clients[nid]["selected_output"] = data.get("output")
        sio.emit("client_device_changed", {"client_id": nid, **data})


@sio.on("test_result")
def on_test_result(data):
    if state["role"] != "server":
        return
    sid    = request.sid
    nid    = _sid_to_node.get(sid, "?")
    run_id = data.get("run_id", "unknown")
    _log(f"Result from {nid}  run={run_id}  status={data.get('status')}", client_id=nid)
    sio.emit("test_result_update", {"client_id": nid, **data})

    run_dir = shared.RESULTS_DIR / run_id
    run_dir.mkdir(exist_ok=True)
    (run_dir / f"result_{nid}_{int(time.time())}.json").write_text(
        json.dumps(data, indent=2)
    )
    _run_pending_clear(run_id, nid)
    _maybe_render_css_spectrograph(data, run_id, nid)


@sio.on("command_result")
def on_command_result(data):
    if state["role"] != "server":
        return
    sid = request.sid
    nid = _sid_to_node.get(sid, "?")
    _log(f"cmd result [{nid}]: rc={data.get('returncode')} {data.get('stdout', '')[:120]}",
         client_id=nid)
    sio.emit("command_result_update", {"client_id": nid, **data})


# ===========================================================================
# SocketIO CLIENT  (client role: connects to server)
# ===========================================================================

def _make_server_client() -> sio_module.Client:
    c = sio_module.Client(
        reconnection=True, reconnection_attempts=10,
        reconnection_delay=2, logger=False, engineio_logger=False
    )

    @c.on("connect")
    def _on_connect():
        _log("SocketIO client connected to server")
        c.emit("register", {
            "client_id":       state["node_id"],
            "ip":              _local_ip(),
            "port":            state["port"],
            "devices":         shared.get_audio_devices(),
            "selected_input":  state["selected_input"],
            "selected_output": state["selected_output"],
            # CPU type + which C cores this host can actually load.  Reported
            # at registration so the server can refuse a run BEFORE dispatching
            # it, rather than discovering mid-flight that one participant was
            # silently on the numpy path.
            **_backend_inventory(),
        })
        state["connected"] = True
        state["status"]    = "idle"

    @c.on("disconnect")
    def _on_disconnect():
        _log("SocketIO client disconnected from server", "warn")
        state["connected"] = False
        state["status"]    = "disconnected"

    @c.on("registered")
    def _on_registered(data):
        offset = time.time() - data.get("server_time", time.time())
        _log(f"Registered with server. Clock offset: {offset:.2f}s")

    @c.on("run_test")
    def _on_run_test(data):
        test_name = data.get("test", "")
        params    = data.get("params", {})
        test_path = str(shared.find_test_script(f"{test_name}.py"))
        _log(f"Received run_test: {test_name}")
        threading.Thread(target=_exec_test, args=(test_path, params), daemon=True).start()

    @c.on("stop")
    def _on_stop(data):
        """Stop the run the server names, and tell it what actually died.

        Runs on a worker thread: the reap is bounded but not instant (SIGTERM,
        short grace, SIGKILL, re-scan) and blocking the socket event loop for
        it stalls this client's heartbeat, which the server then reads as a
        lost node.
        """
        run_id = (data or {}).get("run_id") if isinstance(data, dict) else None

        def _work():
            try:
                summary = _do_stop(run_id)
            except Exception as exc:
                _log(f"Stop failed: {exc}", "error")
                summary = {"node": state["node_id"], "run_id": run_id,
                           "killed": 0, "error": str(exc), "remaining": []}
            try:
                c.emit("stop_report", summary)
                c.emit("procs_report", procreg.report())
            except Exception:
                pass

        threading.Thread(target=_work, daemon=True).start()

    @c.on("procs_query")
    def _on_procs_query(_data):
        """Server asked what test processes this node is holding."""
        def _work():
            try:
                c.emit("procs_report", procreg.report())
            except Exception as exc:
                _log(f"procs report failed: {exc}", "debug")
        threading.Thread(target=_work, daemon=True).start()

    @c.on("reap")
    def _on_reap(_data):
        """Server asked every node to clear leftover test processes."""
        def _work():
            try:
                res = procreg.reap_all(log=lambda m: _log(m, "warn"))
                n = sum(k.get("killed", 0)
                        for k in res["registered"] + res["untracked"])
                _log(f"Reap requested by server: {n} process(es) killed",
                     "warn")
                state["status"]      = "idle"
                state["current_run"] = None
                c.emit("stop_report", {
                    "node": state["node_id"], "run_id": None, "killed": n,
                    "remaining": procreg.run_survivors(), "ts": time.time(),
                    "reason": "reap"})
                c.emit("procs_report", procreg.report())
            except Exception as exc:
                _log(f"Reap failed: {exc}", "error")
        threading.Thread(target=_work, daemon=True).start()

    @c.on("set_device")
    def _on_set_device(data):
        if "input"  in data: state["selected_input"]  = data["input"]
        if "output" in data: state["selected_output"] = data["output"]
        _log(f"Device updated — input={state['selected_input']}  output={state['selected_output']}")
        c.emit("device_selected", {
            "input":  state["selected_input"],
            "output": state["selected_output"],
        })

    @c.on("ping")
    def _on_ping(_data):
        c.emit("pong", {"ts": time.time(), "client_id": state["node_id"]})

    return c


# ===========================================================================
# HTTP API  (all routes on every node; server-only routes return 403 elsewhere)
# ===========================================================================

@app.route("/")
def index():
    return render_template(
        "server.html",               # single unified template for both roles
        node_id=state["node_id"],
        role=state["role"],
        port=state["port"],
        server_ip=state.get("server_ip") or "",
        server_port=state.get("server_port") or 8001,
    )


@app.route("/api/status")
def api_status():
    base = {
        "role":            state["role"],
        "node_id":         state["node_id"],
        "status":          state["status"],
        "selected_input":  state["selected_input"],
        "selected_output": state["selected_output"],
        "current_run":     state["current_run"],
    }
    if state["role"] == "server":
        base["clients_connected"] = len(_clients)
        base["ts"]                = time.time()
    else:
        base["server_ip"]    = state["server_ip"]
        base["connected"] = state["connected"]
    return jsonify(base)


@app.route("/api/devices")
def api_devices():
    return jsonify(shared.get_audio_devices())


@app.route("/api/logs")
def api_logs():
    return jsonify({"logs": state["logs"][-200:]})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Stop this node's test processes.  Blocks until the reap is verified."""
    body   = request.get_json(silent=True) or {}
    run_id = body.get("run_id") or request.args.get("run_id")
    summary = _do_stop(run_id)
    return jsonify({"ok": not summary.get("remaining"), **summary})


@app.route("/api/reload", methods=["GET", "POST"])
def api_reload():
    """Restart this node's Python process to pick up bind-mounted code changes.

    The Docker entrypoint supervises `app.py` in a restart loop, so exiting the
    interpreter here causes the loop to relaunch it against the freshly-mounted
    /audiotestserver source. Any active test subprocesses are stopped first.
    Callable either from the LLM (`curl -X POST .../api/reload`) or by hitting
    the URL directly in a browser, which only ever issues GET.
    """
    _log("Reload requested — stopping procs and exiting for supervised restart", "warn")
    _do_stop()

    def _bye():
        time.sleep(0.5)   # let the HTTP response flush first
        os._exit(0)       # hard exit; the entrypoint while-loop relaunches us

    threading.Thread(target=_bye, daemon=True).start()
    return jsonify({"ok": True, "reloading": True, "node_id": state["node_id"]})


@app.route("/api/select_device", methods=["POST"])
def api_select_device():
    data = request.get_json() or {}
    if "input"  in data: state["selected_input"]  = data["input"]
    if "output" in data: state["selected_output"] = data["output"]

    if state["role"] == "server" and state["node_id"] in _clients:
        _clients[state["node_id"]]["selected_input"]  = state["selected_input"]
        _clients[state["node_id"]]["selected_output"] = state["selected_output"]
        sio.emit("client_device_changed", {
            "client_id": state["node_id"],
            "input":     state["selected_input"],
            "output":    state["selected_output"],
        })
    elif state["role"] == "client" and state["connected"] and _server_client:
        _server_client.emit("device_selected", {
            "input":  state["selected_input"],
            "output": state["selected_output"],
        })

    return jsonify({"ok": True,
                    "selected_input":  state["selected_input"],
                    "selected_output": state["selected_output"]})


# -- Connect / disconnect  (client role only) ---------------------------------

@app.route("/api/connect", methods=["POST"])
def api_connect():
    if state["role"] == "server":
        return jsonify({"error": "server nodes do not connect to other servers"}), 400

    global _server_client
    data      = request.get_json() or {}
    server_ip    = data.get("server_ip",    state["server_ip"]    or "127.0.0.1").strip()
    server_port  = int(data.get("server_port", state["server_port"] or 8001))
    node_id   = data.get("node_id",   state["node_id"]).strip()

    state["server_ip"]   = server_ip
    state["server_port"] = server_port
    state["node_id"]  = node_id

    with _server_client_lock:
        if _server_client and _server_client.connected:
            try:
                _server_client.disconnect()
            except Exception:
                pass
        _server_client = _make_server_client()
        url = f"http://{server_ip}:{server_port}"
        try:
            _server_client.connect(url, wait_timeout=5)
            _log(f"Connecting to server at {url}")
            return jsonify({"ok": True, "url": url})
        except Exception as exc:
            state["connected"] = False
            _log(f"Connection failed: {exc}", "error")
            return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/disconnect", methods=["POST"])
def api_disconnect():
    if state["role"] == "server":
        return jsonify({"error": "server nodes do not disconnect from servers"}), 400

    global _server_client
    with _server_client_lock:
        if _server_client:
            try:
                _server_client.disconnect()
            except Exception:
                pass
            _server_client = None
    state["connected"] = False
    state["status"]    = "disconnected"
    return jsonify({"ok": True})


# -- server-only routes ----------------------------------------------------------

@app.route("/api/clients")
@_server_only
def api_clients():
    return jsonify({"clients": _clients})


import ast
import re as _re

def _parse_test_params(script_path: str) -> list[dict]:
    """
    Scan a test .py file for every  PARAMS.get("key")  or
    PARAMS.get("key", default)  call and return a list of:

        { "key": str, "default": <python value | None>, "type": str,
          "choices": [<value>, ...] | None, "group": str | None }

    Type is inferred from the default: int/float → "number",
    bool → "boolean", list/dict → "json", str → "string", None → "string".

    "choices" is populated when the line carries a trailing annotation comment
    of the form  ``# choices: a, b, c``  — the web UI renders such a param as a
    dropdown instead of a free-text field.  Choice values are coerced to the
    param's inferred type (numbers stay numeric).

    "group" is populated from a trailing ``# group: <name>`` annotation — the
    web UI inserts a section divider whenever consecutive params' groups
    differ, so e.g. a modem test that exposes both a CSS and an OFDM PHY can
    tag each PARAMS.get(...) call to keep the two settings blocks visually
    separate instead of interleaving. Both annotations may appear on the same
    line, in either order (e.g. ``# choices: 1, 2  # group: ofdm``).

    Duplicate keys within the same file are deduplicated (first wins).
    """
    try:
        source = Path(script_path).read_text(encoding="utf-8", errors="replace")
    except (OSError, IOError):
        return []

    # Match: PARAMS.get( "key" )  or  PARAMS.get( "key", <expr> )
    # We capture everything after the comma up to the closing paren so we can
    # eval it safely with ast.literal_eval.
    pattern = _re.compile(
        r"""PARAMS\s*\.\s*get\s*\(\s*["']([^"']+)["']\s*(?:,\s*(.*?))?\s*\)""",
        _re.DOTALL,
    )

    seen: dict[str, dict] = {}
    for m in pattern.finditer(source):
        key      = m.group(1)
        raw_expr = (m.group(2) or "").strip()

        if key in seen:
            continue  # first occurrence wins

        default = None
        py_type = "string"

        if raw_expr:
            # Strip trailing comment (e.g.  0.8,  # amplitude)
            raw_expr = _re.sub(r"#.*$", "", raw_expr, flags=_re.MULTILINE).strip()
            # Remove trailing comma that sometimes slips in
            raw_expr = raw_expr.rstrip(",").strip()
            try:
                default = ast.literal_eval(raw_expr)
            except Exception:
                default = None   # variable reference, not a literal — don't prefill

        if isinstance(default, bool):
            py_type = "boolean"
        elif isinstance(default, (int, float)):
            py_type = "number"
        elif isinstance(default, (list, dict)):
            py_type = "json"
        else:
            py_type = "string"

        # Look for trailing  "# choices: a, b, c"  and  "# group: name"
        # annotations on the same line as the PARAMS.get(...) call. Both use
        # [^#]+ so either can follow the other on one line without one
        # swallowing the other's text.
        choices = None
        line_end = source.find("\n", m.end())
        trailing = source[m.end(): line_end if line_end != -1 else len(source)]
        cmatch = _re.search(r"#.*\bchoices:\s*([^#]+)", trailing)
        if cmatch:
            raw_choices = [c.strip() for c in cmatch.group(1).split(",")]
            choices = []
            for c in raw_choices:
                if not c:
                    continue
                if py_type == "number":
                    try:
                        choices.append(int(c) if "." not in c else float(c))
                        continue
                    except ValueError:
                        pass
                choices.append(c)
            if not choices:
                choices = None

        gmatch = _re.search(r"#.*\bgroup:\s*([^#]+)", trailing)
        group = gmatch.group(1).strip() if gmatch else None
        if group == "":
            group = None

        seen[key] = {"key": key, "default": default, "type": py_type,
                     "choices": choices, "group": group}

    return list(seen.values())


def _merge_params_schema(role_scripts: list[tuple[str, str]]) -> list[dict]:
    """
    Merge param schemas from multiple role scripts into one list.

    role_scripts: [(role_name, script_path), ...]

    Returns a list of param dicts, each with an added "sources" list and an
    optional "conflict" flag when the same key appears with different defaults
    across files.
    """
    merged: dict[str, dict] = {}

    for role_name, script_path in role_scripts:
        for p in _parse_test_params(script_path):
            key = p["key"]
            if key not in merged:
                merged[key] = {**p, "sources": [role_name], "conflict": False}
            else:
                merged[key]["sources"].append(role_name)
                # Inherit choices/group from whichever role script declared them.
                if not merged[key].get("choices") and p.get("choices"):
                    merged[key]["choices"] = p["choices"]
                if not merged[key].get("group") and p.get("group"):
                    merged[key]["group"] = p["group"]
                if merged[key]["default"] != p["default"]:
                    merged[key]["conflict"] = True
                    # Keep both defaults in a dict keyed by role for UI display
                    if not isinstance(merged[key].get("conflict_defaults"), dict):
                        merged[key]["conflict_defaults"] = {
                            merged[key]["sources"][0]: merged[key]["default"]
                        }
                    merged[key]["conflict_defaults"][role_name] = p["default"]

    return list(merged.values())


@app.route("/api/tests")
@_server_only
def api_tests():
    tests = shared.list_tests()   # existing call — returns list of test dicts

    # Offline regression suites are synthetic entries: they're not role
    # scripts dispatched over AUDIOTOOL_PARAMS, they run tests/regression/
    # {suite}_regress.py right here via the /api/regression/<suite>/* routes.
    # Flagged "synthetic" so the loop below (and the frontend) skip the
    # normal script-based params_schema lookup.
    tests += [
        {"name": "css_regression",  "filename": None, "roles": [],
         "synthetic": "regression", "suite": "css",  "params_schema": []},
        {"name": "ofdm_regression", "filename": None, "roles": [],
         "synthetic": "regression", "suite": "ofdm", "params_schema": []},
    ]

    for t in tests:
        if t.get("synthetic"):
            continue
        roles = t.get("roles", [])
        if roles:
            # Role-based test: aggregate across all role scripts
            role_scripts = [
                (r["role"], str(shared.find_test_script(r["filename"])))
                for r in roles
            ]
            t["params_schema"] = _merge_params_schema(role_scripts)
        else:
            # Standalone test: single script
            script_path = str(shared.find_test_script(f"{t['name']}.py"))
            params = _parse_test_params(script_path)
            for p in params:
                p["sources"] = [t["name"]]
                p["conflict"] = False
            t["params_schema"] = params

    return jsonify({"tests": tests})


@app.route("/api/results")
@_server_only
def api_results():
    return jsonify({"results": shared.list_results()})


@app.route("/api/report_meta")
@_server_only
def api_report_meta():
    """Vocabulary for the Reports tab: verdicts + label suggestions."""
    return jsonify({
        "verdicts": shared.VERDICTS,
        "labels":   shared.known_labels(),
    })


@app.route("/api/results/<run_id>/analyze_ranging", methods=["POST"])
@_server_only
def api_analyze_ranging(run_id: str):
    """Post-hoc two-way ranging for a css_ranging run.

    The ranging test itself only records each node's capture; distance is
    recovered offline by matched-filtering both captures against the burst
    template (css_ranging_analyze.py), which writes ranging_result.json into the
    run dir.  Runs here on the server host, synchronously — it takes seconds.
    """
    run_dir = shared.RESULTS_DIR / run_id
    if not run_dir.is_dir():
        return jsonify({"error": "run not found"}), 404
    script = shared.find_test_script("css_ranging_analyze.py")
    if not script.exists():
        return jsonify({"error": f"missing {script}"}), 404

    rep = shared.run_report(run_id, include_files=False,
                            include_tracks=False)
    if not rep.get("ranging_ready"):
        return jsonify({"error": "run has no initiator+responder ranging "
                                 f"sidecars (found {rep.get('ranging_roles')})"}), 400

    cmd = [sys.executable, str(script), str(run_dir)]
    true_dist = (request.get_json() or {}).get("true_dist")
    if true_dist not in (None, ""):
        try:
            cmd += ["--true-dist", str(float(true_dist))]
        except (TypeError, ValueError):
            return jsonify({"error": "true_dist must be a number"}), 400

    _log(f"Analyzing ranging for {run_id}")
    try:
        proc = subprocess.run(cmd, cwd=str(shared.BASE_DIR), text=True,
                              capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "ranging analysis timed out"}), 504
    out = (proc.stdout or "") + (proc.stderr or "")
    for line in out.splitlines():
        _log(f"[ranging] {line}")
    ranging = shared._read_json(run_dir / "ranging_result.json")
    if ranging:
        _log(f"Ranging {run_id}: distance = {ranging.get('distance_m')} m")
    return jsonify({"ok": proc.returncode == 0, "returncode": proc.returncode,
                    "output": out, "ranging": ranging})


@app.route("/api/results/<run_id>/analyze_constellation", methods=["POST"])
@_server_only
def api_analyze_constellation(run_id: str):
    """Build the per-phase constellation composite for every node in a run.

    An explicit pass, like the ranging analyzer: it re-decodes the run's frames
    out of the capture WAV (~11s per node), which is far too slow to sit in the
    report fetch.  The PNG lands in the run dir and run_report lists it from
    then on.
    """
    run_dir = shared.RESULTS_DIR / run_id
    if not run_dir.is_dir():
        return jsonify({"error": "run not found"}), 404
    try:
        import constellation
    except Exception as exc:                                   # noqa: BLE001
        return jsonify({"error": f"constellation module unavailable: {exc}"}), 500

    manifest = shared._read_json(run_dir / "manifest.json") or {}
    ids = shared._run_node_ids(run_dir, manifest)
    built, notes = [], []
    sink: list = []          # (node, phases-with-histograms) for the timeline
    for wav, node in shared._capture_wavs(run_dir, ids):
        rec = None
        for p in sorted(run_dir.glob("result_*.json")):
            if shared._WRAPPER_RE.search(p.name):
                continue
            d = shared._read_json(p)
            if isinstance(d, dict) and (d.get("client_id") == node):
                rec = d
                break
        if rec is None:
            notes.append(f"{node}: no per-node result record, skipped")
            continue
        _log(f"Constellation: analyzing {node} in {run_id}")
        try:
            res = constellation.build(run_dir, node, rec, wav,
                                      manifest.get("params") or {}, sink=sink)
        except Exception as exc:                               # noqa: BLE001
            notes.append(f"{node}: {type(exc).__name__}: {exc}")
            continue
        for line in (res or {}).get("notes") or []:
            notes.append(f"{node}: {line}")
            _log(f"Constellation {node}: {line}")
        if not res or not res.get("phases"):
            notes.append(f"{node}: no decodable phase found in the capture")
            continue
        built.append({"client_id": node, **{k: v for k, v in res.items() if k != "notes"}})
        _log(f"Constellation {node}: {len(res.get('phases') or [])} phases")
    # Run-level call flow across every node that produced panels.  Needs at
    # least two, since the whole point is comparing the two ends of a moment.
    timeline = None
    if len(sink) > 1:
        try:
            timeline = constellation.render_timeline(run_dir, sink)
            if timeline:
                _log(f"Constellation timeline: {len(timeline['rows'])} moments "
                     f"across {len(timeline['nodes'])} nodes")
        except Exception as exc:                           # noqa: BLE001
            notes.append(f"timeline: {type(exc).__name__}: {exc}")
    return jsonify({"ok": bool(built), "built": built, "notes": notes,
                    "constellation_timeline": timeline})


@app.route("/api/labels/delete", methods=["POST"])
@_server_only
def api_delete_label():
    """Forget a label across every run annotation and regression case."""
    label = (request.get_json() or {}).get("label", "")
    res = shared.delete_label(label)
    if not res.get("ok"):
        return jsonify(res), 400
    _log(f"Deleted label '{res['label']}' from {len(res['runs'])} run(s), "
         f"{len(res['cases'])} case(s)")
    return jsonify(res)


@app.route("/api/results/<run_id>")
@_server_only
def api_result_detail(run_id: str):
    rep = shared.run_report(run_id)
    if rep is None:
        return jsonify({"error": "run not found"}), 404
    # Full per-node stdout, which the list view deliberately omits.
    logs = []
    for p in sorted((shared.RESULTS_DIR / run_id).glob("result_*.json")):
        d = shared._read_json(p) or {}
        if d.get("output"):
            logs.append({"client_id": d.get("client_id"), "file": p.name,
                         "output": d["output"]})
    rep["logs"] = logs
    rep["suggested_expect"] = shared.suggested_expect(run_id)
    return jsonify(rep)


@app.route("/api/results/<run_id>/annotation", methods=["POST"])
@_server_only
def api_annotate_result(run_id: str):
    data = request.get_json() or {}
    try:
        ann = shared.write_annotation(run_id, data)
    except FileNotFoundError:
        return jsonify({"error": "run not found"}), 404
    _log(f"Annotated {run_id}: verdict={ann['verdict']} labels={ann['labels']}")
    return jsonify({"ok": True, "annotation": ann})


@app.route("/api/results/<run_id>", methods=["DELETE"])
@_server_only
def api_delete_result(run_id: str):
    ann = shared.read_annotation(run_id)
    if ann.get("case_name") and not request.args.get("force"):
        return jsonify({
            "error": f"run backs regression case '{ann['case_name']}' — "
                     f"remove it from the suite first (or pass force=1)",
        }), 409
    res = shared.delete_run(run_id)
    if not res.get("ok"):
        return jsonify(res), 404
    _log(f"Deleted result run {run_id}", "warn")
    return jsonify(res)


@app.route("/api/results/delete", methods=["POST"])
@_server_only
def api_delete_results_bulk():
    run_ids = (request.get_json() or {}).get("run_ids") or []
    deleted, skipped = [], []
    for rid in run_ids:
        ann = shared.read_annotation(rid)
        if ann.get("case_name"):
            skipped.append({"run_id": rid,
                            "reason": f"backs regression case '{ann['case_name']}'"})
            continue
        res = shared.delete_run(rid)
        (deleted if res.get("ok") else skipped).append(
            rid if res.get("ok") else {"run_id": rid, "reason": res.get("error")})
    if deleted:
        _log(f"Deleted {len(deleted)} result run(s)", "warn")
    return jsonify({"ok": True, "deleted": deleted, "skipped": skipped})


@app.route("/api/results/<run_id>/promote", methods=["POST"])
@_server_only
def api_promote_result(run_id: str):
    data = request.get_json() or {}
    res = shared.promote_run(
        run_id,
        expect=data.get("expect"),
        labels=data.get("labels"),
        note=data.get("note", ""),
        case_name=data.get("case_name"),
    )
    if not res.get("ok"):
        return jsonify(res), 400
    _log(f"Promoted {run_id} to regression suite as '{res['case']['name']}'")
    return jsonify(res)


@app.route("/api/results/<run_id>/unpromote", methods=["POST"])
@_server_only
def api_unpromote_result(run_id: str):
    res = shared.unpromote_run(run_id)
    _log(f"Removed {run_id} from regression suite")
    return jsonify(res)


@app.route("/api/regression/cases")
@_server_only
def api_regression_cases():
    return jsonify({"cases": shared.load_regression_cases()})


# ---------------------------------------------------------------------------
# Built-in regression suite runner
#
# tests/regression/css_regress.py and tests/regression/ofdm_regress.py each
# replay synthetic encodes and recorded WAVs through their decoder offline —
# no audio hardware, no client nodes, so they run right here on the server
# host.  One run at a time per suite; output is streamed into the log and
# parsed into a per-case table for the Reports tab.
# ---------------------------------------------------------------------------

_REGRESS_SCRIPTS = {
    "css":  shared.TESTS_DIR / "regression" / "css_regress.py",
    "ofdm": shared.TESTS_DIR / "regression" / "ofdm_regress.py",
}

# "  PASS  mic_li2_1.0s                 [  12.3s]  status=ok text='hi' ..."
_REGRESS_CASE_RE = re.compile(
    r"^\s*(PASS|FAIL|SKIP)\s+(\S+)\s+\[\s*([\d.]+)s\]\s*(.*)$")

_regress_lock = {suite: threading.Lock() for suite in _REGRESS_SCRIPTS}
_regress_proc: dict[str, subprocess.Popen | None] = {suite: None for suite in _REGRESS_SCRIPTS}
_regress = {suite: {
    "running":     False,
    "args":        [],
    "started_at":  None,
    "finished_at": None,
    "returncode":  None,
    "cases":       [],
    "summary":     None,
    "output":      [],
} for suite in _REGRESS_SCRIPTS}


def _regress_worker(suite: str, args: list[str]):
    proc = None
    script = _REGRESS_SCRIPTS[suite]
    r = _regress[suite]
    # The suite fans out over a ProcessPoolExecutor; the pool workers are the
    # heavy ones.  Same treatment as a test: own session, registered group,
    # and an AUDIOTOOL_RUN_ID so the environment scan can recognise strays.
    run_id = f"regression_{suite}_{int(time.time())}"
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(script), *args],
            cwd=str(shared.BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1",
                 "AUDIOTOOL_RUN_ID": run_id},
            start_new_session=True,
        )
        with _regress_lock[suite]:
            _regress_proc[suite] = proc
        procreg.register(proc.pid, log=lambda m: _log(m, "warn"),
                         run_id=run_id, role="regression",
                         test=f"{suite}_regress", node=state["node_id"])

        for line in proc.stdout:
            line = line.rstrip()
            r["output"].append(line)
            del r["output"][:-500]        # keep the tail bounded
            m = _REGRESS_CASE_RE.match(line)
            if m:
                r["cases"].append({
                    "status": m.group(1),
                    "name":   m.group(2),
                    "secs":   float(m.group(3)),
                    "detail": m.group(4).strip(),
                })
            elif "REGRESSION:" in line:
                r["summary"] = line.strip(" *")
            _log(f"[regress:{suite}] {line}")
        proc.wait()
        r["returncode"] = proc.returncode
        _log(f"{suite} regression suite finished rc={proc.returncode} "
             f"({r['summary'] or 'no summary'})",
             "info" if proc.returncode == 0 else "warn")
    except Exception as exc:
        r["output"].append(f"runner error: {exc}")
        r["returncode"] = -1
        _log(f"{suite} regression runner error: {exc}", "error")
    finally:
        if proc is not None:
            res = procreg.kill_pid_group(proc.pid, grace=2.0, run_id=run_id,
                                         log=lambda m: _log(m, "debug"))
            if res.get("killed"):
                _log(f"Reaped {res['killed']} leftover {suite} regression worker(s)",
                     "warn")
        r["running"]     = False
        r["finished_at"] = datetime.now().isoformat()
        with _regress_lock[suite]:
            _regress_proc[suite] = None
        sio.emit("regression_finished", {
            "suite":      suite,
            "returncode": r["returncode"],
            "summary":    r["summary"],
            "cases":      len(r["cases"]),
        })


@app.route("/api/procs", methods=["GET"])
def api_procs():
    """What test processes this node currently owns (tracked + strays)."""
    try:
        return jsonify(procreg.status())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/procs/reap", methods=["POST"])
def api_procs_reap():
    """Kill every test process group this node knows about.

    The manual escape hatch for the case the automatic nets miss.  Also stops
    whatever is currently running -- that is the point of it.  On the server,
    `?all=1` fans the same request out to every connected client first.
    """
    fanout = []
    if state["role"] == "server" and (request.args.get("all")
                                      or (request.get_json(silent=True) or {}).get("all")):
        for nid, sid in list(_client_sids.items()):
            sio.emit("reap", {}, to=sid)
            fanout.append(nid)
    try:
        res = procreg.reap_all(log=lambda m: _log(m, "warn"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    n = sum(k.get("killed", 0) for k in res["registered"] + res["untracked"])
    remaining = procreg.run_survivors()
    _log(f"Manual reap: {n} process(es) killed"
         + (f"; {len(remaining)} STILL ALIVE" if remaining else ""),
         "error" if remaining else "warn")
    state["status"]      = "idle"
    state["current_run"] = None
    return jsonify({"ok": not remaining, "killed": n, "asked": fanout,
                    "remaining": remaining, **res})


@app.route("/api/regression/<suite>/run", methods=["POST"])
@_server_only
def api_regression_run(suite):
    """Start the offline decoder regression suite (css|ofdm) on this host."""
    script = _REGRESS_SCRIPTS.get(suite)
    if script is None:
        return jsonify({"error": f"unknown suite '{suite}'"}), 404
    if _regress[suite]["running"]:
        return jsonify({"error": "a regression run is already in progress"}), 409
    if not script.exists():
        return jsonify({"error": f"missing {script}"}), 404

    data = request.get_json() or {}
    args: list[str] = []
    only = str(data.get("only") or "").strip()
    if only:
        args += ["--only", only]
    for lab in (data.get("labels") or []):
        lab = str(lab).strip()
        if lab:
            args += ["--label", lab]
    if data.get("slow"):
        args.append("--slow")
    if data.get("verbose"):
        args.append("-v")

    _regress[suite].update({
        "running": True, "args": args, "started_at": datetime.now().isoformat(),
        "finished_at": None, "returncode": None, "cases": [], "summary": None,
        "output": [],
    })
    threading.Thread(target=_regress_worker, args=(suite, args), daemon=True).start()
    _log(f"{suite} regression suite started: {script.name} {' '.join(args)}")
    return jsonify({"ok": True, "args": args})


@app.route("/api/regression/<suite>/status")
@_server_only
def api_regression_status(suite):
    r = _regress.get(suite)
    if r is None:
        return jsonify({"error": f"unknown suite '{suite}'"}), 404
    started = r["started_at"]
    elapsed = None
    if started:
        end = r["finished_at"]
        try:
            t0 = datetime.fromisoformat(started)
            t1 = datetime.fromisoformat(end) if end else datetime.now()
            elapsed = round((t1 - t0).total_seconds(), 1)
        except Exception:
            pass
    return jsonify({**r, "elapsed_s": elapsed,
                    "output": r["output"][-200:]})


@app.route("/api/regression/<suite>/stop", methods=["POST"])
@_server_only
def api_regression_stop(suite):
    if suite not in _REGRESS_SCRIPTS:
        return jsonify({"error": f"unknown suite '{suite}'"}), 404
    with _regress_lock[suite]:
        proc = _regress_proc[suite]
    if not (_regress[suite]["running"] and proc):
        return jsonify({"error": "no regression run in progress"}), 409
    try:
        threading.Thread(
            target=procreg.kill_pid_group, args=(proc.pid,),
            kwargs={"log": lambda m: _log(m, "warn")}, daemon=True).start()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    _log(f"{suite} regression suite stop requested", "warn")
    return jsonify({"ok": True})


@app.route("/api/results/<run_id>/artifact/<path:filename>")
@_server_only
def api_result_artifact(run_id: str, filename: str):
    """Serve a file from a run's results directory (e.g. the mesh piano-roll
    roll.html, or events.jsonl).  run_id/filename are constrained to the results
    tree by send_from_directory's safe-join."""
    run_dir = (shared.RESULTS_DIR / run_id).resolve()
    if shared.RESULTS_DIR.resolve() not in run_dir.parents and run_dir != shared.RESULTS_DIR.resolve():
        abort(404)
    if not run_dir.is_dir():
        abort(404)
    return send_from_directory(str(run_dir), filename)


@app.route("/api/clients/<node_id>/set_device", methods=["POST"])
@_server_only
def api_set_device(node_id: str):
    data = request.get_json() or {}
    if node_id == state["node_id"]:
        # server node itself
        if "input"  in data: state["selected_input"]  = data["input"]
        if "output" in data: state["selected_output"] = data["output"]
        if node_id in _clients:
            _clients[node_id]["selected_input"]  = state["selected_input"]
            _clients[node_id]["selected_output"] = state["selected_output"]
        sio.emit("client_device_changed", {"client_id": node_id, **data})
        return jsonify({"ok": True})
    if node_id not in _client_sids:
        return jsonify({"error": "node not connected"}), 404
    sio.emit("set_device", data, to=_client_sids[node_id])
    _log(f"Set device on {node_id}: {data}")
    return jsonify({"ok": True})


@app.route("/api/clients/<node_id>/command", methods=["POST"])
@_server_only
def api_send_command(node_id: str):
    data    = request.get_json() or {}
    command = data.get("command", "")
    if not command:
        return jsonify({"error": "no command"}), 400
    if node_id not in _client_sids:
        return jsonify({"error": "node not connected"}), 404
    sio.emit("run_command", {"command": command}, to=_client_sids[node_id])
    _log(f"Sent command to {node_id}: {command}")
    return jsonify({"ok": True})


@app.route("/api/tests/run", methods=["POST"])
@_server_only
def api_run_test():
    """
    Flat dispatch:
      { "test": "audiotest1", "params": {}, "clients": ["client1", "client2"] }

    Role-based dispatch:
      { "test": "audiotest1",
        "params": {},
        "roles": { "play": ["client1"], "listen": ["client2", "server"] } }
    """
    data      = request.get_json() or {}
    test_name = data.get("test", "")
    params    = data.get("params", {})
    roles_map = data.get("roles", {})
    flat_list = data.get("clients", [])

    # Build dispatch list: [(node_id, script_name, role_str), ...]
    dispatch: list[tuple[str, str, str]] = []

    if roles_map:
        for role, node_ids in roles_map.items():
            script      = f"{test_name}_{role}"
            script_path = shared.find_test_script(f"{script}.py")
            if not script_path.exists():
                return jsonify({"error": f"role script '{script}.py' not found"}), 404
            for nid in node_ids:
                dispatch.append((nid, script, role))
    else:
        script_path = shared.find_test_script(f"{test_name}.py")
        if not script_path.exists():
            return jsonify({"error": f"test '{test_name}.py' not found"}), 404
        for nid in (flat_list or list(_clients.keys())):
            dispatch.append((nid, test_name, ""))

    if not dispatch:
        return jsonify({"error": "no nodes targeted"}), 400

    # ---- C backend preflight, before anything is dispatched -----------------
    # If this run asks for a C core (params.env / c_backends), every targeted
    # node must be able to load it.  A node whose library was built for another
    # CPU falls back to numpy/reedsolo and still reports a green pass, so a
    # single mismatched participant does not degrade the run, it invalidates
    # it — the results look like the C port's and are the reference path's.
    # BASE_DIR being a shared SMB mount makes this the normal case, not an edge
    # one: all nodes see one tests/ directory.  Hence: nobody starts.
    env_req = {k: str(v) for k, v in (params.get("env") or {}).items()}
    for key in (params.get("c_backends") or []):
        b = _CBACKEND_ENV.get(key)
        if b:
            env_req[b[0]] = b[1]
            if key == "ofdm_acq32":
                env_req["OFDM_ACQ_LIB"] = "32"
    want_keys = _cbackend_requested(env_req)
    if want_keys:
        problems = []
        for nid, _s, _r in dispatch:
            problems += _backend_problems(nid, want_keys)
        if problems:
            _log("Run refused: C backend preflight failed on "
                 f"{len(problems)} participant check(s)", "error")
            for p in problems:
                _log(f"  {p}", "error")
            return jsonify({
                "error": "c_backend_preflight_failed",
                "requested": want_keys,
                "problems": problems,
            }), 409
        params["env"] = env_req

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{test_name}"
    run_dir = shared.RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # In a container, _local_ip() is the (unroutable) bridge address, so the
    # server's self-reported IP is useless to off-host clients. Prefer an
    # operator-set host address (DOCKERHOST=<host LAN ip/name>); when it's unset
    # we drop server_url entirely for remote clients below so each one
    # coordinates back to the address it actually connected on.
    public_host = os.environ.get("DOCKERHOST", "").strip()
    server_ip = public_host or _clients.get(state["node_id"], {}).get("ip", "127.0.0.1")
    remote_server_url = f"http://{server_ip}:{state['port']}"
    local_server_url  = f"http://127.0.0.1:{state['port']}"   # for local subprocs

    params.update({
        "run_id":     run_id,
        "test_name":  test_name,
        "server_url": remote_server_url,   # default; overridden for local below
        # Shared schedule anchor: tests with wall-clock-cycled airtime windows
        # (meshmodem) count their schedule from dispatch time instead of a
        # random time.time() phase, so window 0 opens at run start on every
        # node (unix time -> timezone-free; NTP skew ~0.3s is negligible).
        "sched_epoch": time.time(),
    })

    role_to_nodes = {}
    for nid, _script, role in dispatch:
        role_to_nodes.setdefault(role, []).append(nid)

    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id":      run_id,
        "test":        test_name,
        "params":      params,
        "roles_map":   roles_map,
        "dispatch":    [{"node": n, "script": s, "role": r} for n, s, r in dispatch],
        "started_at":  datetime.now().isoformat(),
    }, indent=2))

    _run_pending_arm(run_id, [n for n, _s, _r in dispatch])

    unreachable: list[str] = []
    for nid, script, role in dispatch:
        node_params = {**params, "role": role, "role_assignments": role_to_nodes}
        test_path   = str(shared.find_test_script(f"{script}.py"))

        if nid == state["node_id"]:
            # Server running its own slice: force 127.0.0.1 so coord HTTP calls
            # always succeed regardless of LAN firewall / routing rules.
            local_params = {**node_params, "server_url": local_server_url}
            threading.Thread(
                target=_exec_test, args=(test_path, local_params), daemon=True
            ).start()
            _log(f"Running '{script}' locally (server node, role={role or '-'})", client_id=nid)
        elif nid in _client_sids:
            # Remote client. If no public host is configured, strip server_url so
            # the client falls back to the exact address it connected on
            # (state["server_ip"]:server_port) — always routable, unlike the
            # server's container-internal _local_ip().
            remote_params = dict(node_params)
            if not public_host:
                remote_params.pop("server_url", None)
            sio.emit("run_test", {"test": script, "params": remote_params},
                     to=_client_sids[nid])
            _log(f"Dispatched '{script}' -> {nid} (role={role or '-'})", client_id=nid)
        else:
            unreachable.append(nid)
            _run_pending_clear(run_id, nid)   # never dispatched: not "lost"
            _log(f"Node {nid} not connected  skipped", "warn", client_id=nid)

    return jsonify({"ok": True, "run_id": run_id, "unreachable": unreachable})

# How long the server waits for every client to confirm its reap before it
# answers the operator.  Long enough for SIGTERM + _STOP_GRACE + SIGKILL +
# a re-scan on a slow machine; short enough that Stop still feels like a
# button and not a job.
_STOP_CONFIRM_TIMEOUT = 8.0


@app.route("/api/stop/all", methods=["POST"])
@_server_only
def api_stop_all():
    """Stop every node and confirm it, instead of firing and hoping.

    The old version emitted "stop" and returned `ok: True` in the same
    breath.  Anything that escaped its process group on a client — the
    playback daemon being the one operators actually hear — kept running,
    and the UI said the run was stopped.  Now: name the run, wait for each
    node's reap report, escalate to a full reap on any node that still has
    a marked process alive, and tell the operator what is left.
    """
    body   = request.get_json(silent=True) or {}
    with _run_pending_lock:
        pending_run = _run_pending["run_id"]
    run_id = body.get("run_id") or pending_run or state.get("current_run")

    t0      = time.time()
    stopped = []
    for nid, sid in list(_client_sids.items()):
        _clients.get(nid, {}).pop("last_stop", None)
        sio.emit("stop", {"run_id": run_id}, to=sid)
        stopped.append(nid)
        _log(f"Stop signal -> {nid} (run={run_id or '*'})", client_id=nid)

    own = _do_stop(run_id)          # server's own procs, synchronously

    # --- wait for each client to confirm ------------------------------------
    nodes: dict[str, dict] = {}
    deadline = t0 + _STOP_CONFIRM_TIMEOUT
    while time.time() < deadline:
        missing = [n for n in stopped
                   if float((_clients.get(n, {}).get("last_stop") or {})
                            .get("ts", 0)) < t0]
        if not missing:
            break
        time.sleep(0.2)

    escalated = []
    for nid in stopped:
        rep = _clients.get(nid, {}).get("last_stop") or {}
        if float(rep.get("ts", 0)) < t0:
            nodes[nid] = {"reported": False,
                          "note": "no stop confirmation within "
                                  f"{_STOP_CONFIRM_TIMEOUT:g}s"}
            _log(f"{nid} did not confirm the stop — asking for a full reap",
                 "error", client_id=nid)
            sio.emit("reap", {}, to=_client_sids.get(nid))
            escalated.append(nid)
            continue
        left = rep.get("remaining") or []
        nodes[nid] = {"reported": True, "killed": rep.get("killed", 0),
                      "remaining": left}
        if left:
            # Its own group kill missed something.  reap_all ignores run
            # scoping entirely, so it catches a stray whose marker no longer
            # matches the run we asked about.
            _log(f"{nid} still holds {len(left)} process(es) after stop — "
                 f"escalating to a full reap", "error", client_id=nid)
            sio.emit("reap", {}, to=_client_sids.get(nid))
            escalated.append(nid)

    left_here = own.get("remaining") or []
    clean = not left_here and not escalated
    return jsonify({
        "ok":        clean,
        "run_id":    run_id,
        "stopped":   stopped,
        "server":    {"killed": own.get("killed", 0), "remaining": left_here},
        "nodes":     nodes,
        "escalated": escalated,
        "elapsed_s": round(time.time() - t0, 2),
    })


@app.route("/api/procs/all")
@_server_only
def api_procs_all():
    """Every node's test processes, as last reported.

    `?refresh=1` asks each client for a fresh scan first and waits briefly
    for the answers, so the operator is not reading a 30s-old picture when
    deciding whether something is still running.
    """
    t0 = time.time()
    if request.args.get("refresh"):
        for nid, sid in list(_client_sids.items()):
            sio.emit("procs_query", {}, to=sid)
        deadline = t0 + 4.0
        while time.time() < deadline:
            if all(float((_clients.get(n, {}).get("procs") or {}).get("ts", 0))
                   >= t0 for n in _client_sids):
                break
            time.sleep(0.2)
        try:
            rep = procreg.report()
            state["procs"] = rep
            if state["node_id"] in _clients:
                _clients[state["node_id"]]["procs"] = rep
        except Exception:
            pass

    out = {}
    for nid, info in _clients.items():
        rep = info.get("procs") or {}
        out[nid] = {
            "n_procs":  rep.get("n_procs", 0),
            "rss_mb":   rep.get("rss_mb", 0.0),
            "runs":     rep.get("runs", {}),
            "procs":    rep.get("procs", []),
            "tracked":  rep.get("tracked", []),
            "age_s":    (round(time.time() - float(rep["ts"]), 1)
                         if rep.get("ts") else None),
            "stale":    (not rep.get("ts")
                         or time.time() - float(rep["ts"]) > 90),
            "last_stop": info.get("last_stop"),
        }
    return jsonify({"ok": True, "nodes": out,
                    "total": sum(v["n_procs"] for v in out.values())})


# -- Coordination endpoints (server-only) ----------------------------------------

@app.route("/api/coord/<run_id>/<signal>", methods=["POST"])
@_server_only
def coord_post(run_id: str, signal: str):
    data    = request.get_json() or {}
    node_id = data.get("client_id", "unknown")
    with _coord_lock:
        _coord.setdefault(run_id, {}).setdefault(signal, [])
        if node_id not in _coord[run_id][signal]:
            _coord[run_id][signal].append(node_id)
            # server-clock arrival time: gives every poller the SAME number
            # to compute schedule advances from, regardless of when it polls
            _coord_ts.setdefault(run_id, {}).setdefault(signal, {})[node_id] \
                = time.time()
    _log(f"coord [{run_id}] {signal} <- {node_id}", "debug")
    return jsonify({"ok": True, "signal": signal, "clients": _coord[run_id][signal]})


@app.route("/api/coord/<run_id>/<signal>", methods=["GET"])
@_server_only
def coord_get(run_id: str, signal: str):
    with _coord_lock:
        nodes = _coord.get(run_id, {}).get(signal, [])
        ts    = dict(_coord_ts.get(run_id, {}).get(signal, {}))
    return jsonify({"signal": signal, "clients": nodes, "count": len(nodes),
                    "ts": ts, "server_now": time.time()})


@app.route("/api/coord/<run_id>", methods=["DELETE"])
@_server_only
def coord_clear(run_id: str):
    with _coord_lock:
        _coord.pop(run_id, None)
    return jsonify({"ok": True})

@app.route("/api/coord/<run_id>", methods=["GET"])
@_server_only
def coord_dump(run_id: str):
    """Return all signals registered for a run — useful for debugging timeouts."""
    with _coord_lock:
        data = {k: list(v) for k, v in _coord.get(run_id, {}).items()}
    return jsonify({"run_id": run_id, "signals": data})



# ===========================================================================
# server helpers
# ===========================================================================

def _watchdog():
    """Mark nodes as stale if they haven't sent stats recently."""
    while True:
        now = time.time()
        for nid, info in list(_clients.items()):
            if now - info.get("last_seen", now) > 15:
                info["status"] = "stale"
        time.sleep(5)


def _register_server_self():
    """Register the server node in its own _clients registry."""
    ip = _local_ip()
    info = {
        "client_id":       state["node_id"],
        "ip":              ip,
        "port":            state["port"],
        "devices":         shared.get_audio_devices(),
        "selected_input":  state["selected_input"],
        "selected_output": state["selected_output"],
        "status":          "idle",
        "connected_at":    datetime.now().isoformat(timespec="seconds"),
        "last_seen":       time.time(),
        "stats":           {},
        "is_server":          True,
        # The server takes dispatch slices like any other node, so it is graded
        # by the same pre-dispatch gate and needs the same inventory.
        **_backend_inventory(),
    }
    _clients[state["node_id"]] = info
    shared.register_client(state["node_id"], info)
    state["status"] = "idle"
    _log(f"server self-registered as '{state['node_id']}' ({ip}:{state['port']})")
    sio.emit("client_joined", {"client_id": state["node_id"], "info": info})




# ===========================================================================
# Entry point
# ===========================================================================

def run_node(
    role:      str = "client",
    node_id:   str = "node1",
    host:      str = "0.0.0.0",
    port:      int = 8001,
    server_ip:    str | None = None,
    server_port:  int = 8001,
):
    """
    Start this node.

    Parameters
    ----------
    role      "server" or "client"
    node_id   Unique name for this node (e.g. "server", "client1")
    host      Bind address for the local Flask server
    port      Local Flask+SocketIO port (default 8001)
    server_ip    IP of the server node (client role only)
    server_port  Port of the server node (client role only, default 8001)
    """
    state["role"]     = role
    state["node_id"]  = node_id
    state["port"]     = port
    state["server_ip"]   = server_ip
    state["server_port"] = server_port

    # Reap anything the previous instance left behind before doing anything
    # else: a node that was killed (crash, docker restart, kill -9) never got
    # to stop its tests, and those groups are still holding the audio device
    # this node is about to open.
    _boot_reap()
    _install_exit_reaper()

    # Shared background threads
    threading.Thread(target=_stats_loop, daemon=True).start()
    threading.Thread(target=_reaper_loop, daemon=True).start()

    if role == "server":
        threading.Thread(target=_watchdog, daemon=True).start()
        # Delay self-registration until SocketIO server is up
        def _delayed():
            time.sleep(1.5)
            _register_server_self()
        threading.Thread(target=_delayed, daemon=True).start()

    elif role == "client":
        state["status"] = "disconnected"
        # Auto-connect if server_ip was provided at startup
        if server_ip:
            def _auto_connect():
                time.sleep(1.0)
                global _server_client
                with _server_client_lock:
                    _server_client = _make_server_client()
                url = f"http://{server_ip}:{server_port}"
                try:
                    _server_client.connect(url, wait_timeout=8)
                    _log(f"Auto-connected to server at {url}")
                except Exception as exc:
                    _log(f"Auto-connect failed: {exc}", "error")
            threading.Thread(target=_auto_connect, daemon=True).start()

    _log(f"Starting node role={role} id={node_id} port={port}")
    sio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


# ===========================================================================
# CLI shim
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AudioTool node")
    parser.add_argument("role", choices=["server", "client"],
                        help="Node role: 'server' (coordinator) or 'client'")
    parser.add_argument("--id", default=None,
                        help="Node ID (default: 'server' for server, 'client1' for client)")
    parser.add_argument("--port", type=int, default=8001,
                        help="bind port (default: 8001)")
    parser.add_argument("--host", default=None,
                        help="host IP address to connect")

    args = parser.parse_args()

    default_id = "server" if args.role == "server" else "client1"
    run_node(
        role=args.role,
        node_id=args.id or default_id,
        port=args.port,
        server_ip=args.host
    )