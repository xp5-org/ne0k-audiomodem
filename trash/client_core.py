# client_core.py

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
import socketio as sio_module

import shared

# -- Flask app -----------------------------------------------------------------

app  = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "audiotool-client-secret"

# Mutable state container so threads can share it
state = {
    "client_id":       "client1",
    "client_port":     5001,
    "server_ip":       None,
    "server_port":     5000,
    "connected":       False,
    "selected_input":  None,
    "selected_output": None,
    "status":          "disconnected",
    "logs":            [],
    "current_run":     None,
}

_sio_client: sio_module.Client | None = None
_sio_lock = threading.Lock()
_active_procs: dict[str, subprocess.Popen] = {}  # role -> proc, supports concurrent roles on one client
_proc_lock = threading.Lock()


# -- Logging helper ------------------------------------------------------------

def _log(msg: str, level: str = "info"):
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "msg": msg,
    }
    state["logs"].append(entry)
    if len(state["logs"]) > 300:
        state["logs"].pop(0)
    print(f"[{entry['ts']}][{level.upper()}] {msg}")

    if state["connected"] and _sio_client:
        try:
            _sio_client.emit("client_log", {"level": level, "msg": msg,
                                             "client_id": state["client_id"]})
        except Exception:
            pass


# -- Server Socket.IO client ---------------------------------------------------

def _make_sio_client() -> sio_module.Client:
    c = sio_module.Client(reconnection=True, reconnection_attempts=10,
                           reconnection_delay=2, logger=False, engineio_logger=False)

    @c.on("connect")
    def _on_connect():
        _log("Socket.IO connected to server")
        # Send registration
        c.emit("register", {
            "client_id":       state["client_id"],
            "ip":              _local_ip(),
            "port":            state["client_port"],
            "devices":         shared.get_audio_devices(),
            "selected_input":  state["selected_input"],
            "selected_output": state["selected_output"],
        })
        state["connected"] = True
        state["status"] = "idle"

    @c.on("disconnect")
    def _on_disconnect():
        _log("Disconnected from server", "warn")
        state["connected"] = False
        state["status"] = "disconnected"

    @c.on("registered")
    def _on_registered(data):
        _log(f"Server acknowledged registration. Server time offset: "
             f"{time.time() - data.get('server_time', time.time()):.2f}s")

    @c.on("run_test")
    def _on_run_test(data):
        test_name = data.get("test", "")
        params    = data.get("params", {})
        # Resolve path locally - never trust a remote path (breaks on SMB/Windows)
        test_path = str(shared.TESTS_DIR / f"{test_name}.py")
        _log(f"Received run_test: {test_name} -> {test_path}")
        threading.Thread(target=_exec_test, args=(test_path, params), daemon=True).start()

    # @c.on("run_command")
    # def _on_run_command(data):
    #     cmd = data.get("command", "")
    #     _log(f"Received run_command: {cmd}")
    #     threading.Thread(target=_exec_command, args=(cmd,), daemon=True).start()

    @c.on("stop")
    def _on_stop(data):
        _do_stop()

    @c.on("set_device")
    def _on_set_device(data):
        inp  = data.get("input")
        out  = data.get("output")
        if inp  is not None: state["selected_input"]  = inp
        if out is not None: state["selected_output"] = out
        _log(f"Device updated - input={state['selected_input']} output={state['selected_output']}")
        c.emit("device_selected", {
            "input":  state["selected_input"],
            "output": state["selected_output"],
        })

    @c.on("ping")
    def _on_ping(data):
        c.emit("pong", {"ts": time.time(), "client_id": state["client_id"]})

    return c


def _exec_test(test_path: str, params: dict):
    state["status"] = "running"
    state["current_run"] = params.get("run_id")
    run_id = params.get("run_id", "unknown")
    _log(f"Starting test script: {test_path}")

    env = {
        **os.environ,
        "AUDIOTOOL_CLIENT_ID":      state["client_id"],
        "AUDIOTOOL_INPUT_DEVICE":   ("" if state["selected_input"]  is None else str(state["selected_input"])),
        "AUDIOTOOL_OUTPUT_DEVICE":  ("" if state["selected_output"] is None else str(state["selected_output"])),
        "AUDIOTOOL_PARAMS":         json.dumps(params),
        "AUDIOTOOL_ROLE":           str(params.get("role", "")),
        "AUDIOTOOL_RUN_ID":         run_id,
        "AUDIOTOOL_SERVER_URL":     (params.get("server_url") or f"http://{state['server_ip']}:{state['server_port']}"),
        "AUDIOTOOL_RUN_DIR":        str(shared.RESULTS_DIR / run_id),
    }

    global _active_procs  # declared at top of function, before any try/finally

    role = params.get("role") or test_path  # unique key per concurrent role

    # Create the local results dir
    local_run_dir = shared.RESULTS_DIR / run_id
    local_run_dir.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.Popen(
            [sys.executable, test_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        with _proc_lock:
            _active_procs[role] = proc
        output_lines = []
        for line in proc.stdout:
            line = line.rstrip()
            output_lines.append(line)
            _log(f"[test] {line}")
        proc.wait()

        result = {
            "run_id": run_id,
            "client_id": state["client_id"],
            "test": Path(test_path).stem,
            "returncode": proc.returncode,
            "status": "ok" if proc.returncode == 0 else "error",
            "output": "\n".join(output_lines[-100:]),
            "ts": datetime.now().isoformat(),
        }

        # Try to load structured results if test wrote them
        result_file = (shared.RESULTS_DIR / run_id) / f"result_{state['client_id']}.json"
        if result_file.exists():
            try:
                result["data"] = json.loads(result_file.read_text())
            except Exception:
                pass

        if _sio_client:
            _sio_client.emit("test_result", result)

    except Exception as exc:
        _log(f"Test execution error: {exc}", "error")
        if _sio_client:
            _sio_client.emit("test_result", {
                "run_id": run_id, "client_id": state["client_id"],
                "status": "error", "error": str(exc), "ts": datetime.now().isoformat(),
            })
    finally:
        with _proc_lock:
            _active_procs.pop(role, None)
        # Only go idle when all concurrent roles are done
        if not _active_procs:
            state["status"] = "idle"
            state["current_run"] = None


def _exec_command(cmd: str):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        result = {
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-500:],
            "ts": datetime.now().isoformat(),
        }
        if _sio_client:
            _sio_client.emit("command_result", result)
        _log(f"cmd done rc={proc.returncode}")
    except Exception as exc:
        _log(f"Command error: {exc}", "error")


def _stats_loop():
    while True:
        time.sleep(2)
        if not state["connected"] or not _sio_client:
            continue
        try:
            cpu  = psutil.cpu_percent(interval=None)
            mem  = psutil.virtual_memory()
            disk = psutil.disk_usage(str(shared.BASE_DIR))
            stats = {
                "cpu_pct":   cpu,
                "mem_pct":   mem.percent,
                "mem_used_mb": int(mem.used / 1e6),
                "disk_pct":  disk.percent,
                "status":    state["status"],
                "current_run": state["current_run"],
            }
            _sio_client.emit("stats_update", {"stats": stats, "status": state["status"]})
        except Exception:
            pass


# -- HTTP API ------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("client.html",
                           client_id=state["client_id"],
                           client_port=state["client_port"])


@app.route("/api/status")
def api_status():
    return jsonify({
        "role":            "client",
        "client_id":       state["client_id"],
        "server_ip":       state["server_ip"],
        "connected":       state["connected"],
        "status":          state["status"],
        "selected_input":  state["selected_input"],
        "selected_output": state["selected_output"],
        "current_run":     state["current_run"],
    })


@app.route("/api/devices")
def api_devices():
    return jsonify(shared.get_audio_devices())


@app.route("/api/logs")
def api_logs():
    return jsonify({"logs": state["logs"][-200:]})


@app.route("/api/connect", methods=["POST"])
def api_connect():
    global _sio_client
    data = request.get_json() or {}
    server_ip   = data.get("server_ip", "127.0.0.1").strip()
    server_port = int(data.get("server_port", 5000))
    client_id   = data.get("client_id", state["client_id"]).strip()

    state["server_ip"]   = server_ip
    state["server_port"] = server_port
    state["client_id"]   = client_id

    with _sio_lock:
        if _sio_client and _sio_client.connected:
            try:
                _sio_client.disconnect()
            except Exception:
                pass

        _sio_client = _make_sio_client()
        url = f"http://{server_ip}:{server_port}"
        try:
            _sio_client.connect(url, wait_timeout=5)
            _log(f"Connecting to server at {url}")
            return jsonify({"ok": True, "url": url})
        except Exception as exc:
            state["connected"] = False
            _log(f"Connection failed: {exc}", "error")
            return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/disconnect", methods=["POST"])
def api_disconnect():
    global _sio_client
    with _sio_lock:
        if _sio_client:
            try:
                _sio_client.disconnect()
            except Exception:
                pass
            _sio_client = None
    state["connected"] = False
    state["status"] = "disconnected"
    return jsonify({"ok": True})


def _do_stop():
    """Kill all active test processes."""
    global _active_procs
    with _proc_lock:
        procs = dict(_active_procs)
    if not procs:
        _log("Stop requested but no active processes")
        return
    for role, proc in procs.items():
        if proc and proc.poll() is None:
            _log(f"Stop requested - terminating {role}", "warn")
            try:
                proc.terminate()
                import threading as _t
                def _force(p=proc, r=role):
                    import time as _time
                    _time.sleep(3)
                    try:
                        if p.poll() is None:
                            p.kill()
                            _log(f"Force-killed {r}", "warn")
                    except Exception:
                        pass
                _t.Thread(target=_force, daemon=True).start()
            except Exception as exc:
                _log(f"Stop error ({role}): {exc}", "error")
    state["status"] = "idle"
    state["current_run"] = None
    if _sio_client and state["connected"]:
        _sio_client.emit("stats_update", {"stats": state.get("stats", {}), "status": "idle"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    _do_stop()
    return jsonify({"ok": True})


@app.route("/api/select_device", methods=["POST"])
def api_select_device():
    data = request.get_json() or {}
    if "input"  in data: state["selected_input"]  = data["input"]
    if "output" in data: state["selected_output"] = data["output"]
    if _sio_client and state["connected"]:
        _sio_client.emit("device_selected", {
            "input":  state["selected_input"],
            "output": state["selected_output"],
        })
    return jsonify({"ok": True,
                    "selected_input":  state["selected_input"],
                    "selected_output": state["selected_output"]})


# -- Helpers -------------------------------------------------------------------

def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# -- Entry ---------------------------------------------------------------------

def run_client(client_id: str = "client1", host: str = "0.0.0.0", port: int = 5001):
    state["client_id"]   = client_id
    state["client_port"] = port

    threading.Thread(target=_stats_loop, daemon=True).start()

    from flask_socketio import SocketIO
    local_sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    app.config["SECRET_KEY"] = "client-secret"
    local_sio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)