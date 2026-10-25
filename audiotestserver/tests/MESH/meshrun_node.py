# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
meshrun_node.py — framework wrapper that runs a mesh SIMULATION as a test.

Unlike the audio role scripts, the mesh sim is a single self-contained process
that models every node internally (see meshsim_mac.py / meshsim_dsp.py), so it
runs on ONE host — "the viewed server".  The Flask app dispatches this wrapper
to the server node via the normal /api/tests/run path; it then:

  1. picks a numpy-capable Python (the app itself runs under a numpy-less
     interpreter, but the sim needs numpy/scipy — usually /opt/venv/bin/python);
  2. runs the requested tier into <run_dir>/events.jsonl
       tier=mac : meshsim_mac.py --nodes N --run R   (fast, MAC-level)
       tier=dsp : meshsim_dsp.py --nodes N --run R --jobs J   (real DSP, 40-core)
  3. renders <run_dir>/roll.html (the piano-roll / receipt) with meshroll.py;
  4. writes result_<client>.json so the UI can link the artifact.

This wrapper is deliberately STDLIB-ONLY so it imports cleanly under the app's
interpreter; all numpy work happens in the child process it launches.

PARAMS (scraped by the app's param UI, overridable per run):
  tier     : "mac" or "dsp"
  nodes    : node count (1..15; 15 is the address-space ceiling)
  run_s    : simulated seconds
  jobs     : dsp only — parallel lane-decode workers
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time

TESTS = os.path.dirname(os.path.abspath(__file__))


def _params() -> dict:
    try:
        return json.loads(os.environ.get("AUDIOTOOL_PARAMS", "{}")) or {}
    except Exception:
        return {}


PARAMS = _params()
RUN_DIR = os.environ.get("AUDIOTOOL_RUN_DIR", os.getcwd())
CLIENT_ID = os.environ.get("AUDIOTOOL_CLIENT_ID", "server")


def _find_python() -> str:
    """First interpreter that can import numpy: an explicit override, the known
    venv, this interpreter, or a python3 on PATH."""
    candidates = [
        os.environ.get("MESH_PYTHON"),
        "/opt/venv/bin/python",
        sys.executable,
        shutil.which("python3"),
        shutil.which("python"),
    ]
    for py in candidates:
        if not py or not os.path.exists(py) and not shutil.which(py or ""):
            continue
        try:
            subprocess.run([py, "-c", "import numpy"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return py
        except Exception:
            continue
    return ""


# Window the mesh daemon gets to flush events after Stop before it is
# SIGKILLed.  It is a graceful-stop handler, so without a deadline "stop"
# means "stop at the end of the current step" — which is not what the button
# says.
_DAEMON_STOP_GRACE = 3.0


def _run(cmd):
    """Run a child, streaming its output to our stdout (the app tails it as
    logs) and collecting it.  A SIGTERM/SIGINT from the framework (Stop) is
    forwarded to the child as SIGTERM — for the daemon that's a GRACEFUL stop
    (it flushes events and exits), after which we still render the roll.
    Returns (returncode, lines)."""
    print(f"[meshrun] $ {' '.join(cmd)}", flush=True)
    def _pdeathsig():
        """Die with this runner even if it is SIGKILLed.

        start_new_session (needed to signal the daemon's whole group) also
        takes the child OUT of the runner's group, so the framework's group
        kill cannot reach it — this is what makes an abandoned mesh daemon
        possible at all.  PR_SET_PDEATHSIG closes that hole in the kernel;
        macOS has no equivalent, so there the daemon is only reachable by the
        AUDIOTOOL_RUN_ID it inherited (procreg.reap_run sweeps by that).
        """
        try:
            import ctypes
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(
                1, int(signal.SIGKILL), 0, 0, 0)     # PR_SET_PDEATHSIG
        except Exception:
            pass

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True,
                            preexec_fn=_pdeathsig)

    def _fwd(_signo, _frame):
        """Forward the stop to the daemon, then make sure it is actually gone.

        The daemon treats SIGTERM as a graceful stop and finishes what it is
        doing first, which from the operator's side is "I pressed Stop and it
        carried on to the end of the step".  Give it a bounded window to
        flush, then SIGKILL the group and exit rather than continuing to
        stream its stdout.
        """
        pgid = None
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        deadline = time.time() + _DAEMON_STOP_GRACE
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.1)
        if proc.poll() is None:
            print(f"[meshrun] daemon did not stop in {_DAEMON_STOP_GRACE:g}s "
                  f"— SIGKILL", flush=True)
            try:
                os.killpg(pgid, signal.SIGKILL) if pgid else proc.kill()
            except Exception:
                pass
        os._exit(128 + int(_signo))

    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, _fwd)
        except Exception:
            pass
    lines = []
    for line in proc.stdout:
        line = line.rstrip()
        lines.append(line)
        print(line, flush=True)
    proc.wait()
    return proc.returncode, lines


def _summarize(events_path: str) -> dict:
    """Stdlib summary of the emitted event stream (no numpy)."""
    kinds: dict = {}
    disc: dict = {}
    n_nodes = 0
    meta = {}
    if not os.path.exists(events_path):
        return {"events": 0}
    with open(events_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") == "_meta":
                meta = r
                n_nodes = int(r.get("n_nodes", 0))
                continue
            k = r.get("kind", "?")
            kinds[k] = kinds.get(k, 0) + 1
            if k == "discovered":
                disc.setdefault(r["node"], set()).add(r.get("peer"))
    counts = {n: len(v) for n, v in disc.items()}
    full = sum(1 for n, c in counts.items() if n_nodes and c == n_nodes - 1)
    total = sum(kinds.values())
    return {"events": total, "kinds": kinds, "n_nodes": n_nodes,
            "full_discovery": full, "discovery_counts": counts,
            "scenario": meta.get("scenario")}


def _render(py, events, roll) -> bool:
    """Render events.jsonl → roll.html; return True on success."""
    if not os.path.exists(events):
        return False
    rrc, _ = _run([py, os.path.join(TESTS, "meshroll.py"), events, "-o", roll])
    return rrc == 0 and os.path.exists(roll)


def _run_suite(py) -> int:
    """Suite mode: run the whole scenario battery (meshtests.py), report per-
    scenario pass/fail, and render the headline discovery roll."""
    rc, lines = _run([py, os.path.join(TESTS, "meshtests.py"),
                      "--emit-dir", RUN_DIR])
    scenarios = []
    for ln in lines:
        s = ln.strip()
        if s.startswith(("PASS ", "FAIL ")):
            verdict, _, rest = s.partition(" ")
            scenarios.append({"name": rest.split()[0], "pass": verdict == "PASS"})
    # headline artifact: the full 15-node discovery scenario
    headline = os.path.join(RUN_DIR, "discovery15_clean.jsonl")
    roll = os.path.join(RUN_DIR, "roll.html")
    roll_ok = os.path.exists(headline) and _render(py, headline, roll)
    passed = sum(1 for s in scenarios if s["pass"])
    _write_result({"status": "ok" if rc == 0 else "error", "mode": "suite",
                   "returncode": rc, "scenarios": scenarios,
                   "passed": passed, "total": len(scenarios),
                   "roll_file": "roll.html" if roll_ok else None})
    print(f"[meshrun] suite done — {passed}/{len(scenarios)} scenarios passed",
          flush=True)
    return rc


def main() -> int:
    mode = str(PARAMS.get("mode", "single")).lower()
    tier = str(PARAMS.get("tier", "mac")).lower()
    nodes = max(1, min(15, int(PARAMS.get("nodes", 15))))   # ids 0..14
    run_s = float(PARAMS.get("run_s", 600.0 if tier == "mac" else 200.0))
    jobs = int(PARAMS.get("jobs", nodes if tier == "dsp" else 1))

    events = os.path.join(RUN_DIR, "events.jsonl")
    roll = os.path.join(RUN_DIR, "roll.html")

    py = _find_python()
    if not py:
        print("[meshrun] ERROR: no numpy-capable Python found "
              "(set MESH_PYTHON to e.g. /opt/venv/bin/python)", flush=True)
        _write_result({"status": "error", "error": "no numpy-capable python"})
        return 2
    print(f"[meshrun] mode={mode} tier={tier} nodes={nodes} "
          f"run_s={run_s} jobs={jobs}", flush=True)
    print(f"[meshrun] python={py}", flush=True)

    if mode == "suite":
        return _run_suite(py)

    if mode == "daemon":
        # continuous run until the framework's Stop (SIGTERM) — MAC tier only
        # (the DSP tier is a batch decode, not a live loop).
        print("[meshrun] daemon mode — runs until you press Stop All (MAC tier)",
              flush=True)
        cmd = [py, os.path.join(TESTS, "meshsim_mac.py"),
               "--nodes", str(nodes), "--daemon", "--emit", events]
        _run(cmd)                                  # returns when stopped/flushed
        roll_ok = _render(py, events, roll)
        summary = _summarize(events)
        _write_result({"status": "ok", "mode": "daemon", "tier": "mac",
                       "nodes": nodes,
                       "roll_file": "roll.html" if roll_ok else None,
                       "events_file": "events.jsonl", **summary})
        print(f"[meshrun] daemon stopped — {summary.get('events')} events", flush=True)
        return 0

    # single run
    if tier == "dsp":
        cmd = [py, os.path.join(TESTS, "meshsim_dsp.py"),
               "--nodes", str(nodes), "--jobs", str(jobs),
               "--run", str(run_s), "--emit", events]
    else:
        cmd = [py, os.path.join(TESTS, "meshsim_mac.py"),
               "--nodes", str(nodes), "--run", str(run_s), "--emit", events]
    rc, _ = _run(cmd)
    if rc != 0:
        _write_result({"status": "error", "returncode": rc, "tier": tier})
        return rc
    roll_ok = _render(py, events, roll)
    summary = _summarize(events)
    _write_result({"status": "ok", "mode": "single", "tier": tier, "nodes": nodes,
                   "run_s": run_s, "jobs": jobs,
                   "roll_file": "roll.html" if roll_ok else None,
                   "events_file": "events.jsonl", **summary})
    print(f"[meshrun] done — {summary.get('events')} events, "
          f"full-discovery {summary.get('full_discovery')}/{summary.get('n_nodes')}",
          flush=True)
    return 0


def _write_result(data: dict):
    try:
        with open(os.path.join(RUN_DIR, f"result_{CLIENT_ID}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception as e:
        print(f"[meshrun] could not write result file: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
