"""
server_core.py - Server hub: web dashboard, client registry, test launcher.
http://server-ip:8000 in a browser.
"""
import json
import os
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit, join_room, leave_room

import shared

# -- Flask / SocketIO setup ----------------------------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "audiotool-secret"
sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# In-memory state (authoritative, also mirrored to NFS via shared.py)
_clients: dict[str, dict] = {}          # client_id -> live info
_client_sids: dict[str, str] = {}       # client_id -> socket SID
_sid_to_client: dict[str, str] = {}     # SID -> client_id
_server_logs: list[dict] = []           # rolling server log


def _log(msg: str, level: str = "info", client_id: str = "server"):
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "client": client_id,
        "msg": msg,
    }
    _server_logs.append(entry)
    if len(_server_logs) > 500:
        _server_logs.pop(0)
    print(f"[{entry['ts']}][{level.upper()}][{client_id}] {msg}")
    sio.emit("server_log", entry)


# -- HTTP routes ---------------------------------------------------------------

@app.route("/")
def index():
    return render_template("server.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "role": "server",
        "clients_connected": len(_clients),
        "ts": time.time(),
    })


@app.route("/api/clients")
def api_clients():
    return jsonify({"clients": _clients})


@app.route("/api/devices")
def api_devices():
    #servers own audio devices
    return jsonify(shared.get_audio_devices())




_coord: dict[str, dict[str, list]] = {}
_coord_lock = threading.Lock()


@app.route("/api/coord/<run_id>/<signal>", methods=["POST"])
def coord_post(run_id: str, signal: str):
    #announce that a client has reached a named coordination point
    data      = request.get_json() or {}
    client_id = data.get("client_id", "unknown")
    with _coord_lock:
        _coord.setdefault(run_id, {}).setdefault(signal, [])
        if client_id not in _coord[run_id][signal]:
            _coord[run_id][signal].append(client_id)
    _log(f"coord [{run_id}] {signal} <- {client_id}", level="debug")
    return jsonify({"ok": True, "signal": signal, "clients": _coord[run_id][signal]})


@app.route("/api/coord/<run_id>/<signal>", methods=["GET"])
def coord_get(run_id: str, signal: str):
    """Return which clients have posted this signal."""
    with _coord_lock:
        clients = _coord.get(run_id, {}).get(signal, [])
    return jsonify({"signal": signal, "clients": clients, "count": len(clients)})


@app.route("/api/coord/<run_id>", methods=["DELETE"])
def coord_clear(run_id: str):
    """Clear all coordination state for a run (called after test completes)."""
    with _coord_lock:
        _coord.pop(run_id, None)
    return jsonify({"ok": True})


@app.route("/api/tests")
def api_tests():
    return jsonify({"tests": shared.list_tests()})


@app.route("/api/results")
def api_results():
    return jsonify({"results": shared.list_results()})


@app.route("/api/logs")
def api_logs():
    return jsonify({"logs": _server_logs[-200:]})


@app.route("/api/clients/<client_id>/command", methods=["POST"])
def api_send_command(client_id: str):
    """Send an arbitrary shell command to a specific client."""
    data = request.get_json() or {}
    command = data.get("command", "")
    if not command:
        return jsonify({"error": "no command"}), 400
    if client_id not in _client_sids:
        return jsonify({"error": "client not connected"}), 404
    sid = _client_sids[client_id]
    sio.emit("run_command", {"command": command}, to=sid)
    _log(f"Sent command to {client_id}: {command}")
    return jsonify({"ok": True})


@app.route("/api/clients/<client_id>/set_device", methods=["POST"])
def api_set_device(client_id: str):
    data = request.get_json() or {}
    if client_id == _SERVER_CLIENT_ID and client_id in _clients:
        if "input"  in data: _clients[client_id]["selected_input"]  = data["input"]
        if "output" in data: _clients[client_id]["selected_output"] = data["output"]
        _log(f"Set device on server node: {data}")
        sio.emit("client_device_changed", {"client_id": client_id, **data})
        return jsonify({"ok": True})
    if client_id not in _client_sids:
        return jsonify({"error": "client not connected"}), 404
    sid = _client_sids[client_id]
    sio.emit("set_device", data, to=sid)
    _log(f"Set device on {client_id}: {data}")
    return jsonify({"ok": True})


@app.route("/api/tests/run", methods=["POST"])
def api_run_test():
    """
    Standalone test (no roles):
      { "test": "audiotest1", "params": {}, "clients": ["client1","client2"] }
      Every listed client runs the same audiotest1.py script.

    Grouped test (roles):
      { "test": "audiotest1",
        "params": {},
        "roles": {
          "play":          ["client1"],
          "listen":        ["client2", "server"],
          "bidirectional": ["client3"]
        }
      }
    """
    data      = request.get_json() or {}
    test_name = data.get("test", "")
    params    = data.get("params", {})
    roles_map = data.get("roles", {})
    flat_list = data.get("clients", [])

    dispatch = []

    if roles_map:
        for role, client_ids in roles_map.items():
            script = f"{test_name}_{role}"
            script_path = shared.TESTS_DIR / f"{script}.py"
            if not script_path.exists():
                return jsonify({"error": f"role script '{script}.py' not found"}), 404
            for cid in client_ids:
                dispatch.append((cid, script, role))
    else:
        script_path = shared.TESTS_DIR / f"{test_name}.py"
        if not script_path.exists():
            return jsonify({"error": f"test '{test_name}.py' not found"}), 404
        targets = flat_list or list(_clients.keys())
        for cid in targets:
            dispatch.append((cid, test_name, ""))

    if not dispatch:
        return jsonify({"error": "no clients targeted"}), 400

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{test_name}"
    run_dir = shared.RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    params["run_id"]    = run_id
    params["test_name"] = test_name
    _real_ip = _clients.get(_SERVER_CLIENT_ID, {}).get("ip", "127.0.0.1")
    params["server_url"] = f"http://{_real_ip}:{_server_port}"
    # run_dir not sent to clients, use local to client dir

    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id":     run_id,
        "test":       test_name,
        "params":     params,
        "roles_map":  roles_map,
        "dispatch":   [{"client": c, "script": s, "role": r} for c,s,r in dispatch],
        "started_at": datetime.now().isoformat(),
    }, indent=2))

    role_to_clients = {}
    for cid, script, role in dispatch:
        role_to_clients.setdefault(role, [])
        role_to_clients[role].append(cid)

    # If the same client is assigned multiple roles, client_core spawns each
    unreachable = []
    for cid, script, role in dispatch:
        client_params = {**params, "role": role, "role_assignments": role_to_clients}
        if cid == _SERVER_CLIENT_ID and cid in _clients:
            threading.Thread(
                target=_run_test_locally,
                args=(cid, script, client_params),
                daemon=True,
            ).start()
            _log(f"Running '{script}' locally (server node, role={role or '-'})", client_id=cid)
        elif cid in _client_sids:
            sio.emit("run_test", {"test": script, "params": client_params},
                     to=_client_sids[cid])
            _log(f"Dispatched '{script}' -> {cid} (role={role or '-'})", client_id=cid)
        else:
            unreachable.append(cid)
            _log(f"Client {cid} not connected - skipped", level="warn", client_id=cid)

    return jsonify({"ok": True, "run_id": run_id, "unreachable": unreachable})


_server_active_proc: subprocess.Popen | None = None
_server_proc_lock = threading.Lock()

@app.route("/api/stop/all", methods=["POST"])
def api_stop_all():
    stopped = []
    # stop remote
    for cid, sid in list(_client_sids.items()):
        sio.emit("stop", {}, to=sid)
        stopped.append(cid)
        _log(f"Stop signal sent to {cid}", client_id=cid)

    # stop local process
    with _server_proc_lock:
        proc = _server_active_proc
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            threading.Timer(3, lambda: proc.kill() if proc.poll() is None else None).start()
            _log("Stop signal sent to server-local process", client_id=_SERVER_CLIENT_ID)
            stopped.append(_SERVER_CLIENT_ID + " (local)")
        except Exception as exc:
            _log(f"Server-local stop error: {exc}", level="error")

    return jsonify({"ok": True, "stopped": stopped})


@sio.on("connect")
def on_connect():
    _log(f"New socket connection: {request.sid}", level="debug")


@sio.on("disconnect")
def on_disconnect():
    sid = request.sid
    cid = _sid_to_client.pop(sid, None)
    if cid:
        _client_sids.pop(cid, None)
        _clients.pop(cid, None)
        shared.unregister_client(cid)
        _log(f"Client disconnected: {cid}", level="warn")
        sio.emit("client_left", {"client_id": cid})


@sio.on("register")
def on_register(data):
    sid = request.sid
    cid = data.get("client_id", f"unknown_{sid[:6]}")

    # Kick out stale sid for same client_id
    old_sid = _client_sids.get(cid)
    if old_sid and old_sid != sid:
        _sid_to_client.pop(old_sid, None)

    _client_sids[cid] = sid
    _sid_to_client[sid] = cid

    info = {
        "client_id": cid,
        "ip": data.get("ip", request.remote_addr),
        "port": data.get("port", 5001),
        "devices": data.get("devices", {}),
        "selected_input": data.get("selected_input"),
        "selected_output": data.get("selected_output"),
        "status": "idle",
        "connected_at": datetime.now().isoformat(timespec="seconds"),
        "last_seen": time.time(),
        "stats": {},
    }
    _clients[cid] = info
    shared.register_client(cid, info)

    join_room(cid)
    _log(f"Client registered: {cid} from {info['ip']}:{info['port']}")
    sio.emit("client_joined", {"client_id": cid, "info": info})
    emit("registered", {"ok": True, "server_time": time.time()})


@sio.on("stats_update")
def on_stats(data):
    sid = request.sid
    cid = _sid_to_client.get(sid)
    if cid and cid in _clients:
        _clients[cid]["stats"] = data.get("stats", {})
        _clients[cid]["last_seen"] = time.time()
        _clients[cid]["status"] = data.get("status", _clients[cid].get("status", "idle"))
        sio.emit("client_stats", {"client_id": cid, "stats": data.get("stats", {}),
                                   "status": _clients[cid]["status"]})


@sio.on("client_log")
def on_client_log(data):
    sid = request.sid
    cid = _sid_to_client.get(sid, "?")
    _log(data.get("msg", ""), level=data.get("level", "info"), client_id=cid)


@sio.on("device_selected")
def on_device_selected(data):
    sid = request.sid
    cid = _sid_to_client.get(sid)
    if cid and cid in _clients:
        _clients[cid]["selected_input"]  = data.get("input")
        _clients[cid]["selected_output"] = data.get("output")
        sio.emit("client_device_changed", {"client_id": cid, **data})


@sio.on("test_result")
def on_test_result(data):
    sid = request.sid
    cid = _sid_to_client.get(sid, "?")
    run_id = data.get("run_id", "unknown")
    _log(f"Test result from {cid} run={run_id}: {data.get('status')}", client_id=cid)
    sio.emit("test_result_update", {"client_id": cid, **data})

    # Persist to NFS
    run_dir = shared.RESULTS_DIR / run_id
    run_dir.mkdir(exist_ok=True)
    out_f = run_dir / f"result_{cid}_{int(time.time())}.json"
    out_f.write_text(json.dumps(data, indent=2))


@sio.on("command_result")
def on_command_result(data):
    sid = request.sid
    cid = _sid_to_client.get(sid, "?")
    _log(f"cmd result [{cid}]: rc={data.get('returncode')} {data.get('stdout','')[:120]}",
         client_id=cid)
    sio.emit("command_result_update", {"client_id": cid, **data})


def _watchdog():
    while True:
        now = time.time()
        for cid, info in list(_clients.items()):
            if now - info.get("last_seen", now) > 15:
                info["status"] = "stale"
        time.sleep(5)


_SERVER_CLIENT_ID = "server"


def _register_self(server_id: str, port: int):
    import socket as _socket
    import psutil

    def _local_ip():
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    ip = _local_ip()
    info = {
        "client_id":       server_id,
        "ip":              ip,
        "port":            port,
        "devices":         shared.get_audio_devices(),
        "selected_input":  None,
        "selected_output": None,
        "status":          "idle",
        "connected_at":    datetime.now().isoformat(timespec="seconds"),
        "last_seen":       time.time(),
        "stats":           {},
        "is_server":       True,
    }
    _clients[server_id] = info
    shared.register_client(server_id, info)
    _log(f"Server self-registered as client '{server_id}' ({ip}:{port})")
    sio.emit("client_joined", {"client_id": server_id, "info": info})

    def _stats_loop():
        while True:
            time.sleep(2)
            if server_id not in _clients:
                break
            try:
                s = {
                    "cpu_pct":     psutil.cpu_percent(interval=None),
                    "mem_pct":     psutil.virtual_memory().percent,
                    "mem_used_mb": int(psutil.virtual_memory().used / 1e6),
                    "disk_pct":    psutil.disk_usage(str(shared.BASE_DIR)).percent,
                }
                _clients[server_id]["stats"]     = s
                _clients[server_id]["last_seen"] = time.time()
                sio.emit("client_stats", {
                    "client_id": server_id,
                    "stats":     s,
                    "status":    _clients[server_id].get("status", "idle"),
                })
            except Exception:
                pass

    threading.Thread(target=_stats_loop, daemon=True).start()


def _run_test_locally(server_id: str, test_name: str, params: dict):
    """Run a test script on the server node itself (same logic as client_core)."""
    test_path = shared.TESTS_DIR / f"{test_name}.py"
    if not test_path.exists():
        _log(f"Local test not found: {test_path}", level="error", client_id=server_id)
        return

    _clients[server_id]["status"] = "running"
    sio.emit("client_stats", {
        "client_id": server_id,
        "stats":     _clients[server_id].get("stats", {}),
        "status":    "running",
    })

    env = {
        **os.environ,
        "AUDIOTOOL_CLIENT_ID":     server_id,
        "AUDIOTOOL_INPUT_DEVICE":  ("" if _clients[server_id].get("selected_input")  is None else str(_clients[server_id]["selected_input"])),
        "AUDIOTOOL_OUTPUT_DEVICE": ("" if _clients[server_id].get("selected_output") is None else str(_clients[server_id]["selected_output"])),
        "AUDIOTOOL_PARAMS":        json.dumps(params),
        "AUDIOTOOL_ROLE":          str(params.get("role", "")),
        "AUDIOTOOL_RUN_ID":        params.get("run_id", "unknown"),
        "AUDIOTOOL_SERVER_URL":    f"http://127.0.0.1:{_server_port}",
        "AUDIOTOOL_RUN_DIR":       str(shared.RESULTS_DIR / params.get("run_id", "unknown")),
    }

    global _server_active_proc  # declared at top of function, before any try/finally

    try:
        proc = subprocess.Popen(
            [sys.executable, str(test_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
        with _server_proc_lock:
            _server_active_proc = proc
        lines = []
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            _log(f"[test] {line}", client_id=server_id)
        proc.wait()

        result = {
            "run_id":     params.get("run_id"),
            "client_id":  server_id,
            "test":       test_name,
            "returncode": proc.returncode,
            "status":     "ok" if proc.returncode == 0 else "error",
            "output":     "\n".join(lines[-100:]),
            "ts":         datetime.now().isoformat(),
        }
        run_dir = shared.RESULTS_DIR / params.get("run_id", "unknown")
        result_file = run_dir / f"result_{server_id}.json"
        if result_file.exists():
            try:
                result["data"] = json.loads(result_file.read_text())
            except Exception:
                pass

        sio.emit("test_result_update", {"client_id": server_id, **result})
        _log(f"Local test '{test_name}' done rc={proc.returncode}", client_id=server_id)

        if run_dir.exists():
            out_f = run_dir / f"result_{server_id}_{int(time.time())}.json"
            out_f.write_text(json.dumps(result, indent=2))

    except Exception as exc:
        _log(f"Local test error: {exc}", level="error", client_id=server_id)
        sio.emit("test_result_update", {
            "client_id": server_id, "status": "error", "error": str(exc),
            "run_id": params.get("run_id"), "ts": datetime.now().isoformat(),
        })
    finally:
        with _server_proc_lock:
            _server_active_proc = None
        if server_id in _clients:
            _clients[server_id]["status"] = "idle"


_server_port: int = 8000

def run_server(host: str = "0.0.0.0", port: int = 8000, server_id: str = "server"):
    global _SERVER_CLIENT_ID, _server_port
    _SERVER_CLIENT_ID = server_id
    _server_port = port

    threading.Thread(target=_watchdog, daemon=True).start()

    def _delayed_register():
        time.sleep(1.5)
        _register_self(server_id, port)

    threading.Thread(target=_delayed_register, daemon=True).start()
    sio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)