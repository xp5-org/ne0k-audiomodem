"""
shared.py - utilities shared by server and client
"""
import json
import os
import time
from pathlib import Path

BASE_DIR   = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
RESULTS_DIR = BASE_DIR / "results"
TESTS_DIR  = BASE_DIR / "tests"
WAV_DIR    = BASE_DIR / "wavfiles"

for _d in (CONFIG_DIR, RESULTS_DIR, TESTS_DIR, WAV_DIR):
    _d.mkdir(exist_ok=True)

CLIENTS_FILE = CONFIG_DIR / "clients.json"


def get_audio_devices() -> dict:
    """Return {'inputs': [...], 'outputs': [...]} using sounddevice."""
    try:
        import sounddevice as sd
        raw = sd.query_devices()
        inputs, outputs = [], []
        for i, d in enumerate(raw):
            base = {
                "id": i,
                "name": d["name"],
                "sample_rate": int(d["default_samplerate"]),
            }
            if d["max_input_channels"] > 0:
                inputs.append({**base, "channels": d["max_input_channels"]})
            if d["max_output_channels"] > 0:
                outputs.append({**base, "channels": d["max_output_channels"]})
        return {"inputs": inputs, "outputs": outputs, "error": None}
    except Exception as exc:
        return {"inputs": [], "outputs": [], "error": str(exc)}



def _load_clients() -> dict:
    if CLIENTS_FILE.exists():
        try:
            return json.loads(CLIENTS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_clients(data: dict):
    tmp = CLIENTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(CLIENTS_FILE)


def register_client(client_id: str, info: dict):
    clients = _load_clients()
    clients[client_id] = {**info, "last_seen": time.time()}
    _save_clients(clients)


def unregister_client(client_id: str):
    clients = _load_clients()
    clients.pop(client_id, None)
    _save_clients(clients)


def get_clients() -> dict:
    return _load_clients()


def list_tests() -> list[dict]:
    """
    Returns a list of dicts:
      standalone: {"name": "audiotest1", "filename": ..., "roles": []}
      grouped:    {"name": "audiotest1", "filename": None, "roles": [
                      {"role": "play",   "name": "audiotest1_play",   "filename": ...},
                      {"role": "listen", "name": "audiotest1_listen", "filename": ...},
                  ]}
    """
    EXCLUDE = {"decoder", "css_encoder", "css_decoder", "coord", "qam_writer", "qam_reader"}   # helper/tool scripts, not framework-launchable

    all_stems = {p.stem: p for p in sorted(TESTS_DIR.glob("*.py"))
                 if p.stem not in EXCLUDE}
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)

    for stem in sorted(all_stems):
        if "_" in stem:
            prefix, role = stem.rsplit("_", 1)
            groups[prefix].append((role, stem))
        else:
            groups[stem]

    result = []
    for parent, role_pairs in sorted(groups.items()):
        if not role_pairs:
            p = all_stems.get(parent)
            if p:
                result.append({
                    "name":     parent,
                    "filename": p.name,
                    "roles":    [],
                })
        else:
            roles = []
            for role, stem in role_pairs:
                p = all_stems[stem]
                roles.append({
                    "role":     role,
                    "name":     stem,
                    "filename": p.name,
                })
            result.append({
                "name":     parent,
                "filename": None,
                "roles":    roles,
            })

    return result



def list_results() -> list[dict]:
    runs = []
    for run_dir in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if run_dir.is_dir():
            summary_f = run_dir / "summary.json"
            summary = {}
            if summary_f.exists():
                try:
                    summary = json.loads(summary_f.read_text())
                except Exception:
                    pass
            runs.append({
                "run_id": run_dir.name,
                "summary": summary,
                "files": [f.name for f in run_dir.iterdir()],
            })
    return runs