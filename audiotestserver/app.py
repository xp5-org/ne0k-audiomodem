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
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit, join_room
import socketio as sio_module          # python-socketio *client*

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
}

# server-only registry (populated only when role == "server")
_clients:     dict[str, dict] = {}   # node_id  -> live info
_client_sids: dict[str, str]  = {}   # node_id  -> socket SID
_sid_to_node: dict[str, str]  = {}   # SID      -> node_id
_coord:       dict[str, dict] = {}   # run_id   -> {signal: [node_ids]}
_coord_lock   = threading.Lock()

# Active subprocesses (both roles)
_active_procs:    dict[str, subprocess.Popen] = {}   # role_key -> proc
_proc_gen:        dict[str, int] = {}               # role_key -> generation counter
_proc_lock        = threading.Lock()
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
    if stale and stale.poll() is None:
        _log(f"Evicting stale proc role={role_key!r} (gen={my_gen-1}) before re-run", "warn")
        try:
            stale.terminate()
            stale.wait(timeout=3)
        except Exception:
            try:
                stale.kill()
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
    }

    try:
        proc = subprocess.Popen(
            [sys.executable, test_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        with _proc_lock:
            _active_procs[role_key] = proc

        output_lines: list[str] = []
        for line in proc.stdout:
            line = line.rstrip()
            output_lines.append(line)
            _log(f"[test] {line}")
        proc.wait()

        result = {
            "run_id":     run_id,
            "client_id":  state["node_id"],
            "test":       Path(test_path).stem,
            "returncode": proc.returncode,
            "status":     "ok" if proc.returncode == 0 else "error",
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
        with _proc_lock:
            # Only remove our slot — don't evict a newer generation's proc
            if _proc_gen.get(role_key) == my_gen:
                _active_procs.pop(role_key, None)
                _proc_gen.pop(role_key, None)
        if not _active_procs:
            state["status"]      = "idle"
            state["current_run"] = None


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
    else:
        # Client node: send back to server
        if _server_client:
            _server_client.emit("test_result", result)


# ---------------------------------------------------------------------------
# Process stop helper (both roles)
# ---------------------------------------------------------------------------

def _do_stop():
    with _proc_lock:
        procs = dict(_active_procs)
    if not procs:
        _log("Stop requested but no active processes")
        return
    for role_key, proc in procs.items():
        if proc and proc.poll() is None:
            _log(f"Terminating process: {role_key}", "warn")
            try:
                proc.terminate()
                def _force(p=proc, r=role_key):
                    time.sleep(3)
                    try:
                        if p.poll() is None:
                            p.kill()
                            _log(f"Force-killed: {r}", "warn")
                    except Exception:
                        pass
                threading.Thread(target=_force, daemon=True).start()
            except Exception as exc:
                _log(f"Stop error ({role_key}): {exc}", "error")
    state["status"]      = "idle"
    state["current_run"] = None


# ---------------------------------------------------------------------------
# Stats broadcast loop (both roles)
# ---------------------------------------------------------------------------

def _stats_loop():
    while True:
        time.sleep(2)
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
        test_path = str(shared.TESTS_DIR / f"{test_name}.py")
        _log(f"Received run_test: {test_name}")
        threading.Thread(target=_exec_test, args=(test_path, params), daemon=True).start()

    @c.on("stop")
    def _on_stop(_data):
        _do_stop()

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
    _do_stop()
    return jsonify({"ok": True})


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


# ── PATCH for app.py ──────────────────────────────────────────────────────────
# 1. Add _parse_test_params() anywhere near the top of the file (after imports).
# 2. Replace the existing api_tests() route with the one below.
# ─────────────────────────────────────────────────────────────────────────────

import ast
import re as _re

def _parse_test_params(script_path: str) -> list[dict]:
    """
    Scan a test .py file for every  PARAMS.get("key")  or
    PARAMS.get("key", default)  call and return a list of:

        { "key": str, "default": <python value | None>, "type": str }

    Type is inferred from the default: int/float → "number",
    bool → "boolean", list/dict → "json", str → "string", None → "string".

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

        seen[key] = {"key": key, "default": default, "type": py_type}

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
                if merged[key]["default"] != p["default"]:
                    merged[key]["conflict"] = True
                    # Keep both defaults in a dict keyed by role for UI display
                    if not isinstance(merged[key].get("conflict_defaults"), dict):
                        merged[key]["conflict_defaults"] = {
                            merged[key]["sources"][0]: merged[key]["default"]
                        }
                    merged[key]["conflict_defaults"][role_name] = p["default"]

    return list(merged.values())


# ── Replace existing api_tests() with this ────────────────────────────────────

@app.route("/api/tests")
@_server_only
def api_tests():
    tests = shared.list_tests()   # existing call — returns list of test dicts

    for t in tests:
        roles = t.get("roles", [])
        if roles:
            # Role-based test: aggregate across all role scripts
            role_scripts = [
                (r["role"], str(shared.TESTS_DIR / r["filename"]))
                for r in roles
            ]
            t["params_schema"] = _merge_params_schema(role_scripts)
        else:
            # Standalone test: single script
            script_path = str(shared.TESTS_DIR / f"{t['name']}.py")
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
            script_path = shared.TESTS_DIR / f"{script}.py"
            if not script_path.exists():
                return jsonify({"error": f"role script '{script}.py' not found"}), 404
            for nid in node_ids:
                dispatch.append((nid, script, role))
    else:
        script_path = shared.TESTS_DIR / f"{test_name}.py"
        if not script_path.exists():
            return jsonify({"error": f"test '{test_name}.py' not found"}), 404
        for nid in (flat_list or list(_clients.keys())):
            dispatch.append((nid, test_name, ""))

    if not dispatch:
        return jsonify({"error": "no nodes targeted"}), 400

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{test_name}"
    run_dir = shared.RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    server_ip = _clients.get(state["node_id"], {}).get("ip", "127.0.0.1")
    remote_server_url = f"http://{server_ip}:{state['port']}"
    local_server_url  = f"http://127.0.0.1:{state['port']}"   # for local subprocs

    params.update({
        "run_id":     run_id,
        "test_name":  test_name,
        "server_url": remote_server_url,   # default; overridden for local below
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

    unreachable: list[str] = []
    for nid, script, role in dispatch:
        node_params = {**params, "role": role, "role_assignments": role_to_nodes}
        test_path   = str(shared.TESTS_DIR / f"{script}.py")

        if nid == state["node_id"]:
            # Server running its own slice: force 127.0.0.1 so coord HTTP calls
            # always succeed regardless of LAN firewall / routing rules.
            local_params = {**node_params, "server_url": local_server_url}
            threading.Thread(
                target=_exec_test, args=(test_path, local_params), daemon=True
            ).start()
            _log(f"Running '{script}' locally (server node, role={role or '-'})", client_id=nid)
        elif nid in _client_sids:
            # Remote client: keep the LAN IP so it can reach back to the server.
            sio.emit("run_test", {"test": script, "params": node_params},
                     to=_client_sids[nid])
            _log(f"Dispatched '{script}' -> {nid} (role={role or '-'})", client_id=nid)
        else:
            unreachable.append(nid)
            _log(f"Node {nid} not connected  skipped", "warn", client_id=nid)

    return jsonify({"ok": True, "run_id": run_id, "unreachable": unreachable})

@app.route("/api/stop/all", methods=["POST"])
@_server_only
def api_stop_all():
    stopped = []
    for nid, sid in list(_client_sids.items()):
        sio.emit("stop", {}, to=sid)
        stopped.append(nid)
        _log(f"Stop signal -> {nid}", client_id=nid)
    _do_stop()   # also stop server's own procs
    return jsonify({"ok": True, "stopped": stopped})


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
    _log(f"coord [{run_id}] {signal} <- {node_id}", "debug")
    return jsonify({"ok": True, "signal": signal, "clients": _coord[run_id][signal]})


@app.route("/api/coord/<run_id>/<signal>", methods=["GET"])
@_server_only
def coord_get(run_id: str, signal: str):
    with _coord_lock:
        nodes = _coord.get(run_id, {}).get(signal, [])
    return jsonify({"signal": signal, "clients": nodes, "count": len(nodes)})


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

    # Shared background threads
    threading.Thread(target=_stats_loop, daemon=True).start()

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