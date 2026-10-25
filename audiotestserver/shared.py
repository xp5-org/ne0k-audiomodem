# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""
shared.py - utilities shared by server and client
"""
# Defer annotation evaluation.  Client nodes run whatever interpreter their
# host has, and PEP 604 unions ("dict | None") are evaluated AT DEF TIME
# without this, so on Python < 3.10 merely importing this module raised
# TypeError.  That broke app.py outright on such a node, and -- quieter --
# made every test's `from shared import native_samplerate` fail into its
# except branch, so those nodes silently assumed 48000 Hz instead of probing
# the device's real rate.
from __future__ import annotations

import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

BASE_DIR   = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
RESULTS_DIR = BASE_DIR / "results"
TESTS_DIR  = BASE_DIR / "tests"
WAV_DIR    = BASE_DIR / "wavfiles"

for _d in (CONFIG_DIR, RESULTS_DIR, TESTS_DIR, WAV_DIR):
    _d.mkdir(exist_ok=True)

CLIENTS_FILE = CONFIG_DIR / "clients.json"

# tests/ is split into per-protocol subdirs; scripts are looked up by bare
# filename across the flat root first, then each of these in turn.
TEST_CATEGORY_DIRS = ["CSS", "MESH", "OFDM", "QAM"]


def find_test_script(filename: str) -> Path:
    """Locate a tests/ script by filename (e.g. 'css_modem.py'), checking the
    flat tests/ root first, then each category subdir. Returns the tests/
    root path (possibly non-existent) if nothing matches, so callers can keep
    doing their own .exists() checks unchanged."""
    direct = TESTS_DIR / filename
    if direct.exists():
        return direct
    for d in TEST_CATEGORY_DIRS:
        p = TESTS_DIR / d / filename
        if p.exists():
            return p
    return direct

# ── report review / regression promotion ─────────────────────────────────────
REGRESSION_DIR   = TESTS_DIR / "regression"
REGRESSION_CASES = REGRESSION_DIR / "cases.json"
RUN_FIXTURES_DIR = REGRESSION_DIR / "fixtures" / "runs"
ANNOTATION_NAME  = "annotation.json"

# Review verdicts a run can carry.  'unreviewed' is the implicit default.
VERDICTS = ["unreviewed", "valid", "expected_fail", "bug", "junk"]

# Condition labels are entirely user-driven: nothing is applied or suggested by
# default.  The picker offers only labels that already exist on some run or
# regression case (see known_labels()), and any new one is typed in.


def default_training_symbols(symbol_dur: float, lead_silence_syms: int = 4) -> int:
    """Default count of pre-preamble training chirps, scaled by symbol duration.

    Training chirps exist purely to let the decoder's timing PLL converge
    before the Barker preamble arrives.  The PLL time-constant is measured in
    *symbols* (~4 symbols at alpha_acq=0.25), so a naive "fixed 20-symbol
    run-in" is duration-independent in symbol count — but at long symbol
    durations that is a wall-clock disaster: 16 training chirps × 5 s = 80 s of
    dead air just to settle, and it pushes the earliest possible sync out past
    2.5 minutes.

    Long symbols also need *fewer* training chirps: each one is a high-
    confidence timing measurement (large time-bandwidth product → sharp
    matched-filter peak), so the PLL converges in a handful of them.  We
    therefore target a fixed wall-clock-ish run-in (~5 s worth of symbols),
    clamped to a sane [8, 20]-symbol band, then subtract the lead silence that
    already precedes the chirps.

    Examples (lead_silence_syms=4):
        symbol_dur=0.25s -> runin 20 -> training 16   (unchanged: fast default)
        symbol_dur=0.5s  -> runin 10 -> training  6
        symbol_dur=1.0s  -> runin  8 -> training  4
        symbol_dur=5.0s  -> runin  8 -> training  4   (was 16 -> saves 60 s)
    """
    runin_target = int(min(20, max(8, round(5.0 / max(symbol_dur, 1e-6)))))
    return max(0, runin_target - int(lead_silence_syms))


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


def probe_audio_interface(direction: str, device=None,
                          requested_fs: float = 44100.0, channels: int = 1) -> dict:
    """Inspect an audio interface and detect host-side sample-rate conversion.

    direction    : 'input' or 'output'
    device       : device index, name, or None (PortAudio default for direction)
    requested_fs : the rate the stream will actually be opened at

    Returns a dict describing the device, its native (shared-mode mixer) rate,
    the host API, and whether opening at requested_fs forces PortAudio to
    resample.  The usual Windows pitfall: a 48000 Hz shared-mode mixer feeding a
    44100 Hz stream silently resamples every block — a non-integer 48/44.1 ratio
    that can smear chirp-to-chirp timing and, under load, drop sample blocks.

    Never raises — on error returns {... 'error': str}.
    """
    out = {"direction": direction, "requested_fs": float(requested_fs)}
    try:
        import sounddevice as sd
        is_in = (direction == "input")
        if device is None or device == "":
            device = sd.default.device[0 if is_in else 1]
        try:
            device = int(device)
        except (TypeError, ValueError):
            pass  # allow a device-name string

        dev      = sd.query_devices(device, "input" if is_in else "output")
        hostapis = sd.query_hostapis()
        ha_name  = (hostapis[dev["hostapi"]]["name"]
                    if 0 <= dev["hostapi"] < len(hostapis) else "?")
        native   = float(dev["default_samplerate"])
        max_ch   = dev["max_input_channels"] if is_in else dev["max_output_channels"]
        lat_key  = "default_low_input_latency" if is_in else "default_low_output_latency"

        out.update({
            "device_index":   device if isinstance(device, int) else None,
            "device_name":    dev["name"],
            "host_api":       ha_name,
            "native_fs":      native,
            "max_channels":   max_ch,
            "default_latency": round(float(dev.get(lat_key, 0.0)), 4),
            "resample":       abs(native - requested_fs) > 0.5,
            "resample_ratio": round(native / requested_fs, 6) if requested_fs else None,
        })

        # Actively validate configs (raises if a rate is truly unsupported).
        # In shared mode PortAudio accepts the requested rate even when it has to
        # resample, so 'requested_supported is True' does NOT mean "no resample" —
        # the native_fs vs requested_fs comparison above is the resample signal.
        check = sd.check_input_settings if is_in else sd.check_output_settings
        def _ck(rate):
            try:
                check(device=device, samplerate=rate, channels=min(channels, max_ch) or 1)
                return True
            except Exception as exc:
                return str(exc)
        out["requested_supported"] = _ck(requested_fs)
        out["native_supported"]    = _ck(native)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def native_samplerate(direction: str, device=None):
    """Return a device's native (shared-mode mixer) sample rate in Hz as an int,
    or None if it can't be determined.

    This is the rate the modem should run at so the host never resamples: open
    the stream at the device's own rate and the 48k↔44.1k smear that destroys
    long-chirp phase coherence simply never happens.  `device` may be an index,
    a name, or None (PortAudio default for the direction).
    """
    info = probe_audio_interface(direction, device, requested_fs=0.0)
    nf = info.get("native_fs")
    try:
        return int(round(float(nf))) if nf else None
    except (TypeError, ValueError):
        return None


def log_audio_interface(direction, device=None, requested_fs=44100.0,
                        channels=1, log=print, tag="AUDIO_IFACE") -> dict:
    """Probe an interface (see probe_audio_interface) and emit a human-readable
    block plus a single machine-parseable JSON line tagged `tag`.  Returns the
    probe dict so callers can react (e.g. record the resample ratio)."""
    info = probe_audio_interface(direction, device, requested_fs, channels)
    if info.get("error"):
        log(f"  [{tag}] {direction} probe FAILED: {info['error']}")
        return info
    if info["resample"]:
        flag = (f"RESAMPLE {info['native_fs']:.0f}->{requested_fs:.0f}Hz "
                f"(x{info['native_fs']/requested_fs:.4f})")
    else:
        flag = "no resample (rates match)"
    log(f"  [{tag}] {direction.upper()} dev[{info.get('device_index')}] "
        f"'{info['device_name']}' via {info['host_api']}")
    log(f"  [{tag}]   requested={requested_fs:.0f}Hz  "
        f"native(shared-mode)={info['native_fs']:.0f}Hz  ->  {flag}")
    if info["resample"]:
        log(f"  [{tag}]   ** host mixer will sample-rate-convert every block; a "
            f"non-integer ratio adds timing jitter and can drop blocks under load **")
    if info.get("requested_supported") is not True:
        log(f"  [{tag}]   requested rate rejected by host: {info['requested_supported']}")
    try:
        import json as _json
        log(f"  [{tag}] {_json.dumps({k: v for k, v in info.items() if k != 'error'})}")
    except Exception:
        pass
    return info



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


def _is_framework_runnable(path) -> bool:
    """True if a tests/ file is a launchable role script rather than a library.

    A framework test is driven by the launcher's environment contract — it reads
    PARAMS (the injected AUDIOTOOL_PARAMS dict) or other AUDIOTOOL_* variables.
    Pure DSP/packet libraries, selftests and offline tools (css_modem.py,
    meshchannel.py, css_correlator.py, …) never touch that contract, so they must
    not show up in the runnable-test list.  This replaces an unmaintainable
    per-file denylist with a positive signal.
    """
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return ("PARAMS" in src) or ("AUDIOTOOL_" in src)


def list_tests() -> list[dict]:
    """
    Returns a list of dicts:
      standalone: {"name": "audiotest1", "filename": ..., "roles": []}
      grouped:    {"name": "audiotest1", "filename": None, "roles": [
                      {"role": "play",   "name": "audiotest1_play",   "filename": ...},
                      {"role": "listen", "name": "audiotest1_listen", "filename": ...},
                  ]}
    """
    # Explicit denylist: files that DO reference the launcher contract (so the
    # heuristic below would keep them) but are still not user-launchable — tools
    # and the mesh sim internals that run only via the meshrun wrapper.
    EXCLUDE = {"decoder", "css_encoder", "css_decoder", "coord", "qam_writer", "qam_reader",
               "channel_sound",
               "meshsim_mac", "meshsim_dsp", "meshtests", "meshbus",
               "meshchannel", "meshevents", "meshroll"}

    py_files = list(TESTS_DIR.glob("*.py"))
    for _d in TEST_CATEGORY_DIRS:
        py_files.extend((TESTS_DIR / _d).glob("*.py"))

    all_stems = {p.stem: p for p in sorted(py_files)
                 if p.stem not in EXCLUDE
                 and not p.stem.endswith("lib")        # library modules (testlib, meshlib, …)
                 and _is_framework_runnable(p)}         # must drive off the launcher contract
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



def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _write_json_atomic(path: Path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


# ── annotations ──────────────────────────────────────────────────────────────
# One sidecar per run (results/<run_id>/annotation.json) so a review travels
# with its run and disappears when the run is deleted.

def default_annotation(run_id: str) -> dict:
    return {
        "run_id":     run_id,
        "verdict":    "unreviewed",
        "labels":     [],
        "note":       "",
        "expect":     None,      # ground-truth text for a promoted case
        "case_name":  None,      # regression case name once promoted
        "updated_at": None,
    }


def read_annotation(run_id: str) -> dict:
    ann = _read_json(RESULTS_DIR / run_id / ANNOTATION_NAME) or {}
    return {**default_annotation(run_id), **ann}


def write_annotation(run_id: str, patch: dict) -> dict:
    """Merge `patch` into a run's annotation and persist it."""
    run_dir = RESULTS_DIR / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(run_id)
    ann = read_annotation(run_id)
    for key in ("verdict", "labels", "note", "expect", "case_name"):
        if key in patch:
            ann[key] = patch[key]
    if ann["verdict"] not in VERDICTS:
        ann["verdict"] = "unreviewed"
    ann["labels"] = sorted({str(x).strip() for x in (ann["labels"] or []) if str(x).strip()})
    ann["note"] = str(ann["note"] or "")
    ann["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json_atomic(run_dir / ANNOTATION_NAME, ann)
    return ann


def known_labels() -> list[str]:
    """Every label in use, across run annotations and regression cases."""
    used: set[str] = set()
    for run_dir in RESULTS_DIR.iterdir():
        if run_dir.is_dir():
            ann = _read_json(run_dir / ANNOTATION_NAME) or {}
            used.update(str(x) for x in (ann.get("labels") or []))
    for case in load_regression_cases():
        used.update(str(x) for x in (case.get("labels") or []))
    return sorted(used)


def delete_label(label: str) -> dict:
    """Forget a label everywhere: strip it from every run annotation and from
    every regression case.  Used to clean up typos — nothing else references a
    label, so this removes it from the vocabulary too."""
    label = str(label).strip()
    if not label:
        return {"ok": False, "error": "empty label"}

    runs = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        ann = _read_json(run_dir / ANNOTATION_NAME)
        if not ann or label not in (ann.get("labels") or []):
            continue
        write_annotation(run_dir.name,
                         {"labels": [l for l in ann["labels"] if l != label]})
        runs.append(run_dir.name)

    cases = load_regression_cases()
    hit_cases = []
    for case in cases:
        if label in (case.get("labels") or []):
            case["labels"] = [l for l in case["labels"] if l != label]
            hit_cases.append(case.get("name"))
    if hit_cases:
        _save_regression_cases(cases)

    return {"ok": True, "label": label, "runs": runs, "cases": hit_cases}


# ── run inspection ───────────────────────────────────────────────────────────

def find_run_capture(run_id: str) -> dict:
    """Locate the listener capture + encoder info sidecar for a run.

    Returns {'wav': Path|None, 'info': Path|None, 'reason': str|None}.  These
    are the two files the offline regression harness needs to replay a run.
    """
    run_dir = RESULTS_DIR / run_id
    if not run_dir.is_dir():
        return {"wav": None, "info": None, "reason": "run not found"}

    wavs = [p for p in run_dir.glob("*_input.wav") if p.stat().st_size > 0]
    if not wavs:
        # 0-byte captures happen when the capture thread dies — call it out
        # rather than silently promoting an unreplayable case.
        zero = list(run_dir.glob("*_input.wav"))
        return {"wav": None, "info": None,
                "reason": "capture WAV is 0 bytes" if zero else "no capture WAV in run"}
    wav = max(wavs, key=lambda p: p.stat().st_size)

    info = run_dir / "css_info.json"
    if not info.exists():
        cands = sorted(run_dir.glob("*info*.json"))
        info = cands[0] if cands else None
    if info is None:
        return {"wav": wav, "info": None, "reason": "no encoder info sidecar in run"}
    return {"wav": wav, "info": info, "reason": None}


def _mesh_delivery(run_dir: Path, manifest: dict) -> list[dict]:
    """Did the payload actually ARRIVE?  One row per node with a role in it.

    A mesh run has no decode sidecar -- delivery is recorded in each node's own
    result record (`data_delivered` on the receiver, `data_ack` on the sender).
    Without this the report showed "No decode sidecar in this run" for a run
    that had in fact carried its whole message end to end, which reads as if
    nothing but the handshake happened.
    """
    want = (manifest.get("params") or {}).get("data_payload")
    out: list[dict] = []
    for p in sorted(run_dir.glob("result_*.json")):
        if _WRAPPER_RE.search(p.name):
            continue
        d = _read_json(p)
        if not isinstance(d, dict):
            continue
        got  = d.get("data_delivered") or {}
        dst  = d.get("data_dst")
        exp  = d.get("expected_srcs") or []
        if not (got or dst is not None or exp):
            continue
        peers = d.get("peers") or {}
        # An ACK only means something for the node that SENT data.  Reporting
        # every peer's data_ack made a pure receiver read "1:no" next to
        # "delivered", which looks like a contradiction and is really just a
        # field that does not apply to it.
        acked = ({str(dst): bool((peers.get(str(dst))
                                  or peers.get(dst) or {}).get("data_ack"))}
                 if dst is not None else {})
        # A receiver is satisfied when every source it expected delivered, and
        # the text matches what the manifest said to send.
        missing = [str(s) for s in exp if str(s) not in {str(k) for k in got}]
        mismatched = [str(k) for k, v in got.items()
                      if want is not None and v != want]
        out.append({
            "client_id":  d.get("client_id") or p.stem.replace("result_", ""),
            "sends_to":   dst,
            "expects":    [str(s) for s in exp],
            "received":   {str(k): v for k, v in got.items()},
            "acked":      acked,
            "missing":    missing,
            "mismatched": mismatched,
            "ok": (not missing and not mismatched
                   and (dst is None or any(acked.values()))),
        })
    return out


def _decode_results(run_dir: Path) -> list[dict]:
    """Per-listener decode sidecars, trimmed to the fields a report needs."""
    out = []
    for p in sorted(run_dir.glob("decode_result_*.json")):
        d = _read_json(p)
        if not isinstance(d, dict):
            continue
        diag = d.get("diagnostics") or {}
        out.append({
            "file":           p.name,
            "client_id":      d.get("client_id") or p.stem.replace("decode_result_", ""),
            "status":         d.get("status"),
            "decoded_text":   d.get("decoded_text", ""),
            "failure_reason": d.get("failure_reason"),
            "frames_decoded": d.get("frames_decoded"),
            "frames_errored": d.get("frames_errored"),
            "avg_snr_db":     diag.get("avg_snr_db"),
            "sync_acquired":  diag.get("sync_acquired"),
            "sync_score":     diag.get("sync_score"),
            "peak_rms":       diag.get("peak_rms"),
            "silence":        diag.get("silence"),
        })
    return out


def _node_results(run_dir: Path) -> list[dict]:
    """Per-node run records (result_<node>_<ts>.json), minus the huge log blob."""
    out = []
    for p in sorted(run_dir.glob("result_*.json")):
        d = _read_json(p)
        if not isinstance(d, dict):
            continue
        out.append({
            "file":       p.name,
            "client_id":  d.get("client_id"),
            "test":       d.get("test"),
            "status":     d.get("status"),
            "returncode": d.get("returncode"),
            "error":      d.get("error"),
            "ts":         d.get("ts"),
            "has_output": bool(d.get("output")),
        })
    return out


# ── OFDM-family report block ─────────────────────────────────────────────────
#
# Applies to every OFDM test (ofdmhs / ofdm2way / ofdm_decode / meshmodem),
# which is why it keys off PAYLOAD SHAPE rather than the test name -- a new
# OFDM test gets charted for free as long as it writes the same fields.
#
# Source of truth is the RICH per-test record, result_<node>.json, written by
# the test itself.  The result_<node>_<epoch>.json files are process-level
# wrappers from app._report_result and carry no modem data.

_WRAPPER_RE = re.compile(r"_\d{9,}\.json$")


def _ofdm_node_metrics(d: dict, node: str) -> dict:
    """Extract link rates, EVM samples and probe spectra from one node record.

    Every field is optional: a decode-only run has EVM but one spectrum, while
    a handshake-only run has rates and spectra and no DATA-phase EVM.  Callers
    treat empty lists as "not measured".

    `bat_evms` is kept separate from `evms` on purpose.  Both are error-vector
    magnitudes, but they measure different links: `evms` is the negotiated data
    carrier (QPSK/16-QAM, thousands of bps), `bat_evms` is the RobustLink
    back-channel that carries the BAT itself (BPSK spread over many carriers,
    tens to hundreds of bps).  Averaging them together would produce a number
    describing neither -- a handshake-only run would silently report its
    bootstrap EVM as though it were the data link's.
    """
    links: list[dict] = []
    evms:  list[float] = []
    bat_evms: list[float] = []
    spectra: list[dict] = []
    # Reverberation ceilings measured by each probe, and the one-pass EVM of any
    # frame whose channel estimate was refined.  Both exist to answer "is this
    # link ISI-limited or estimate-limited", which the averages cannot.
    isi: list[dict] = []
    evms_1pass: list[float] = []
    peer_hs: list[dict] = []

    def _spec(label, s):
        if isinstance(s, dict) and s.get("freq_hz"):
            spectra.append({"label": label, **s})

    def _isi(label, s):
        if isinstance(s, dict) and s.get("isi_ceiling_db") is not None:
            isi.append({"label": label,
                        "ceiling_db": s["isi_ceiling_db"],
                        "ratio": s.get("isi_ratio"),
                        "applied": bool(s.get("isi_applied")),
                        "snr_median_db": s.get("snr_median_db")})

    # -- handshake bindings (ofdmhs, ofdm2way): one measured probe + own table
    pr = d.get("probe_rx")
    if isinstance(pr, dict):
        _spec(f"{node} ← peer", pr.get("spectrum"))
        _isi(f"{node} ← peer", pr)
    if d.get("my_tx_gross_bps"):
        links.append({"label": f"{node} tx", "bps": float(d["my_tx_gross_bps"])})

    # -- ofdm_decode: a single decode pass over a known config
    if d.get("net_bps"):
        links.append({"label": f"{node} rx", "bps": float(d["net_bps"])})
    if d.get("avg_evm_pct") is not None:
        evms.append(float(d["avg_evm_pct"]))
    _spec(f"{node} probe", d.get("spectrum"))

    # -- meshmodem: one negotiated link per peer, both directions
    peers = d.get("peers")
    if isinstance(peers, dict):
        for pid, pv in peers.items():
            pv = pv or {}
            lk = pv.get("link")
            if isinstance(lk, dict):
                for key, dirn in (("tx_gross_bps", "tx"), ("rx_gross_bps", "rx")):
                    if lk.get(key):
                        links.append({"label": f"{node}→{pid} {dirn}"
                                      if dirn == "tx" else f"{node}←{pid}",
                                      "bps": float(lk[key])})
            hs = pv.get("hs") or {}
            if isinstance(hs.get("probe_rx"), dict):
                _spec(f"{node} ← {pid}", hs["probe_rx"].get("spectrum"))
                _isi(f"{node} ← {pid}", hs["probe_rx"])
            # Record the OUTCOME, not just the measurement. A node whose
            # handshake never completed has no probe, no BAT and no spectrum,
            # so it contributes nothing to the lists above; "ok" here reflects
            # hs.get("ok"), not the test script's exit status.
            peer_hs.append({
                "peer":         pid,
                "final_phy":    pv.get("final_phy"),
                "ok":           bool(hs.get("ok")),
                "state":        hs.get("state"),
                "attempts":     pv.get("hs_attempts"),
                "gave_up":      bool(pv.get("gave_up")),
                "has_spectrum": isinstance(hs.get("probe_rx"), dict)
                                and bool((hs["probe_rx"].get("spectrum") or {})
                                         .get("freq_hz")),
            })
    # per-frame EVM from the mesh receive log
    for r in (d.get("rx") or []):
        if isinstance(r, dict) and r.get("evm_pct") is not None:
            evms.append(float(r["evm_pct"]))
            if r.get("evm_pct_1pass") is not None:
                evms_1pass.append(float(r["evm_pct_1pass"]))

    # -- ofdm2way: probes land in `measured`, back-channel legs in
    # `back_channel`.  Neither key is shared with the handshake bindings above,
    # so this run type contributed NOTHING to the report before -- no spectra,
    # no rates, no EVM -- and rendered as "no data phase in this test".
    for mr in (d.get("measured") or []):
        if not isinstance(mr, dict):
            continue
        dirn = mr.get("direction") or "peer"
        _spec(f"{node} ← {dirn}", mr.get("spectrum"))
        _isi(f"{node} ← {dirn}", mr)
        if mr.get("gross_bps"):
            links.append({"label": f"{node} ← {dirn}",
                          "bps": float(mr["gross_bps"])})
    for leg in (d.get("back_channel") or []):
        if not isinstance(leg, dict):
            continue
        rl = leg.get("rl")
        if isinstance(rl, dict) and rl.get("evm_pct") is not None:
            bat_evms.append(float(rl["evm_pct"]))

    # -- handshake bindings that recorded their BAT-exchange decode telemetry
    for key in ("hs", "probe_rx"):
        sub = d.get(key)
        if isinstance(sub, dict) and isinstance(sub.get("rl"), dict):
            if sub["rl"].get("evm_pct") is not None:
                bat_evms.append(float(sub["rl"]["evm_pct"]))
    for pv in (d.get("peers") or {}).values():
        hs = (pv or {}).get("hs") or {}
        rl = hs.get("rl") if isinstance(hs, dict) else None
        if isinstance(rl, dict) and rl.get("evm_pct") is not None:
            bat_evms.append(float(rl["evm_pct"]))

    # -- payload exchange (ofdm_fdd): a real transfer over the measured link.
    # Distinct from every rate above, which are all CAPACITY estimates from a
    # probe; this is bytes that actually arrived, verified by checksum.  A run
    # can grade well and still deliver nothing, and that gap is the finding.
    xfer = d.get("data")
    if isinstance(xfer, dict) and (xfer.get("tx_bytes") is not None
                                   or xfer.get("rx_bytes") is not None):
        xfer = {"node": node, **xfer}
        xfer.pop("evm_pct_per_frame", None)
        data_xfer = [xfer]
    else:
        data_xfer = []

    return {"links": links, "evms": evms, "bat_evms": bat_evms,
            "spectra": spectra, "isi": isi, "evms_1pass": evms_1pass,
            "peer_hs": peer_hs, "data_xfer": data_xfer}


def _ofdm_summary(run_dir: Path, include_spectrum: bool = True) -> dict | None:
    """Roll a run's OFDM records up for the report.  None when there is no
    OFDM payload, so non-OFDM runs render exactly as before.

    `include_spectrum` is dropped for the Reports LIST (list_results calls
    run_report once per run); the per-carrier arrays are only worth shipping
    for the one run whose detail pane is open.
    """
    links: list[dict] = []
    evms:  list[float] = []
    bat_evms: list[float] = []
    spectra: list[dict] = []
    nodes: list[dict] = []
    isi: list[dict] = []
    evms_1pass: list[float] = []
    data_xfer: list[dict] = []

    for p in sorted(run_dir.glob("result_*.json")):
        if _WRAPPER_RE.search(p.name):
            continue
        d = _read_json(p)
        if not isinstance(d, dict):
            continue
        node = d.get("client_id") or p.stem.replace("result_", "")
        m = _ofdm_node_metrics(d, node)
        # A failed handshake has neither a rate nor a spectrum but still
        # measured a probe -- worth listing (that measurement is usually the
        # whole story of why it failed).
        has_probe = isinstance(d.get("probe_rx"), dict)
        if not (m["links"] or m["evms"] or m["bat_evms"] or m["spectra"]
                or has_probe):
            continue
        links.extend(m["links"])
        evms.extend(m["evms"])
        bat_evms.extend(m["bat_evms"])
        spectra.extend(m["spectra"])
        isi.extend(m["isi"])
        evms_1pass.extend(m["evms_1pass"])
        data_xfer.extend(m["data_xfer"])
        pr = d.get("probe_rx") if isinstance(d.get("probe_rx"), dict) else {}
        # Why this node has no BAT chart, in words, so an absent chart reads as
        # a finding rather than as nothing at all.
        hs_note = None
        phy_used = None
        bad = [h for h in m["peer_hs"] if not h["has_spectrum"]]
        if m["peer_hs"]:
            phys = {h["final_phy"] for h in m["peer_hs"] if h["final_phy"]}
            phy_used = "/".join(sorted(phys)) if phys else None
        if bad and not m["spectra"]:
            h = bad[0]
            parts = ["handshake never completed" if not h["ok"]
                     else "handshake completed but recorded no probe spectrum"]
            if h.get("attempts"):
                parts.append(f"after {h['attempts']} attempt(s)")
            if h.get("final_phy"):
                parts.append(f"fell back to {h['final_phy']}")
            hs_note = ", ".join(parts)
        nodes.append({
            "client_id":  node,
            "role":       d.get("role"),
            "status":     d.get("status"),
            # The PHY actually used, and the handshake outcome. `status` above
            # is the test script's exit code and says "ok" even when the link
            # fell back.
            "phy_used":   phy_used,
            "hs_note":    hs_note,
            "band_hz":    d.get("band_hz"),
            "n_carriers": d.get("n_carriers") or pr.get("n_carriers"),
            "n_active":   d.get("my_tx_bat_active") or pr.get("n_active")
                          or d.get("active_carriers"),
            "link_bps":   [l["bps"] for l in m["links"]],
            "evm_pct":    (round(sum(m["evms"]) / len(m["evms"]), 1)
                           if m["evms"] else None),
            "bat_evm_pct": (round(sum(m["bat_evms"]) / len(m["bat_evms"]), 1)
                            if m["bat_evms"] else None),
            # Worst (lowest) reverberation ceiling this node measured: the
            # cap on what ANY allocation over that path can deliver.
            "isi_ceiling_db": (min(x["ceiling_db"] for x in m["isi"])
                               if m["isi"] else None),
        })

    if not nodes:
        return None

    def _avg(xs):
        return round(sum(xs) / len(xs), 1) if xs else None

    out = {
        "nodes":         nodes,
        "links":         links,
        "avg_link_bps":  (int(round(sum(l["bps"] for l in links) / len(links)))
                          if links else None),
        "min_link_bps":  int(min(l["bps"] for l in links)) if links else None,
        "max_link_bps":  int(max(l["bps"] for l in links)) if links else None,
        "n_links":       len(links),
        "avg_evm_pct":   _avg(evms),
        "n_evm_samples": len(evms),
        "avg_bat_evm_pct":   _avg(bat_evms),
        "n_bat_evm_samples": len(bat_evms),
        # ISI ceilings, and the EVM before data-aided re-estimation.  The gap
        # avg_evm_pct_1pass -> avg_evm_pct is channel-ESTIMATE noise the second
        # demod pass removed; the gap from there to the ISI ceiling is what the
        # room itself imposes.
        "isi":               isi,
        "isi_ceiling_db":    (min(x["ceiling_db"] for x in isi)
                              if isi else None),
        "isi_applied":       any(x["applied"] for x in isi),
        "avg_evm_pct_1pass": _avg(evms_1pass),
        # Bytes that actually crossed the air, per direction.
        "data_xfer":         data_xfer,
    }
    if include_spectrum:
        # Rendered server-side to PNG: an inline SVG at report width gave under
        # 2px per carrier on a 350-carrier table, merging neighbours and hiding
        # single dead bins.  Only on the detail view -- the Reports LIST calls
        # run_report once per run and must not draw charts for all of them.
        try:
            import bat_chart
            spectra = bat_chart.render_all(run_dir, spectra)
        except Exception:                                # noqa: BLE001
            pass                                         # chart is optional
        out["spectra"] = spectra
    return out


# ── time-series tracks (level / clipping / S/N vs time) ──────────────────────
#
# The OFDM block above is all run-AVERAGES: one probe SNR, one mean EVM.  That
# cannot distinguish "bad channel" from "input too hot", which is the first
# question a high-EVM run raises.  These tracks are derived from the capture WAV
# by capture_track.py, so they also exist for runs recorded before this code.

def _track_band(d: dict, params: dict) -> tuple[float, float] | None:
    """Analysis band for one node: whatever this run actually used on the air.

    Each test names it differently (a bound handshake reports band_hz, meshmodem
    only has the hs_freq_* request params), and the band matters -- measuring
    in-band S/N over 0-24 kHz would count room hiss above the top carrier as
    signal.
    """
    b = d.get("band_hz")
    if isinstance(b, (list, tuple)) and len(b) == 2 and all(
            isinstance(v, (int, float)) for v in b):
        return float(b[0]), float(b[1])
    for lo_k, hi_k in (("hs_freq_lo", "hs_freq_hi"), ("freq_lo", "freq_hi"),
                       ("f_min", "f_max")):        # CSS names its band this way
        lo, hi = params.get(lo_k), params.get(hi_k)
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return float(lo), float(hi)
    return None


def _track_events(d: dict) -> dict:
    """Timestamped level / EVM points from a node record, for chart overlays.

    These are what the run itself measured, plotted on the same axis as the
    derived tracks: an EVM marker sitting on a clipped window is the whole
    diagnosis in one glance.
    """
    frames: list[dict] = []
    for key, dirn in (("rx", "rx"), ("tx", "tx")):
        for r in (d.get(key) or []):
            if not isinstance(r, dict) or r.get("t") is None:
                continue
            lv = r.get("level") or {}
            frames.append({
                "t":        round(float(r["t"]), 2),
                "dir":      dirn,
                "type":     r.get("type"),
                "phy":      r.get("phy"),
                "evm_pct":  r.get("evm_pct"),
                "peak":     lv.get("peak"),
                "rms":      lv.get("rms"),
                "clip_pct": lv.get("clip_pct"),
            })
    cap = d.get("capture") if isinstance(d.get("capture"), dict) else {}
    # Probe SNR is a single 5s handshake measurement with no reliable timestamp,
    # so it is a reference LINE on the S/N chart rather than a point: it says
    # what the negotiator believed the channel was worth.
    probe_snr = []
    pr = d.get("probe_rx")
    if isinstance(pr, dict) and pr.get("snr_median_db") is not None:
        probe_snr.append({"label": "probe", "snr_db": pr["snr_median_db"]})
    for mr in (d.get("measured") or []):
        if isinstance(mr, dict) and mr.get("snr_median_db") is not None:
            probe_snr.append({"label": mr.get("direction") or "probe",
                              "snr_db": mr["snr_median_db"]})
    for pid, pv in (d.get("peers") or {}).items():
        hs = (pv or {}).get("hs") or {}
        hpr = hs.get("probe_rx") if isinstance(hs, dict) else None
        if isinstance(hpr, dict) and hpr.get("snr_median_db") is not None:
            probe_snr.append({"label": f"probe ←{pid}",
                              "snr_db": hpr["snr_median_db"]})
    return {
        "frames":      sorted(frames, key=lambda f: f["t"])[:400],
        "clip_events": [c for c in (cap.get("clip_events") or [])
                        if isinstance(c, dict)][:20],
        "amb_rms":     cap.get("amb_rms"),
        "rx_peak_worst": cap.get("rx_peak_worst"),
        "probe_snr":   probe_snr,
        # Spans where this node was TRANSMITTING.  Its own speaker couples into
        # its own mic at full scale, so clipped windows inside one of these say
        # nothing about the input gain -- they are the node hearing itself.
        "tx_windows":  [w for w in (cap.get("tx_windows") or [])
                        if isinstance(w, dict) and w.get("t") is not None][:60],
    }


def _run_node_ids(run_dir: Path, manifest: dict) -> list[str]:
    """Every client_id that took part in a run, longest first.

    Drawn from the manifest's roles_map AND every result sidecar including the
    process-level wrappers, because some tests (CSS) write only wrappers -- and
    without an id list a capture cannot be told apart from a TX asset below.
    """
    ids = set()
    for lst in (manifest.get("roles_map") or {}).values():
        ids.update(str(c) for c in (lst or []))
    for p in run_dir.glob("result_*.json"):
        d = _read_json(p)
        if isinstance(d, dict) and d.get("client_id"):
            ids.add(str(d["client_id"]))
    return sorted(ids, key=len, reverse=True)


def _capture_wavs(run_dir: Path, node_ids: list[str]) -> list[tuple[Path, str]]:
    """(capture WAV, owning node) pairs, largest capture first.

    A run dir mixes recordings with TX-side assets the test wrote for its own
    playback (css_output.wav, tx_beacon_*.wav, *_enc.wav).  Charting one of those
    as "input level" would be actively misleading -- its headroom is just the
    amplitude param, and it never went through anyone's ADC.  So this matches
    RECORDINGS only: `<client_id>_*.wav` (what every node names its capture) or
    the CSS listener's `<node>_input.wav`.
    """
    out = []
    for p in sorted(run_dir.glob("*.wav")):
        if p.stat().st_size <= 1024:
            continue
        node = next((cid for cid in node_ids if p.stem.startswith(cid)), None)
        if node is None and p.name.endswith("_input.wav"):
            node = p.name[: -len("_input.wav")]
        if node is None:
            continue
        out.append((p, node))
    return sorted(out, key=lambda t: -t[0].stat().st_size)


def _capture_tracks(run_dir: Path, manifest: dict, params: dict) -> list[dict]:
    """Per-node level / S/N tracks for the report detail pane.

    Returns [] when nothing can be derived (no capture uploaded, or numpy is
    missing on the server) -- the caller renders an explanatory note, never an
    empty chart.
    """
    try:
        import capture_track
    except Exception:                                  # noqa: BLE001
        return []

    # Rich per-node records, keyed by client_id, so each WAV can be matched to
    # the record that says which band it used and what it decoded.
    recs: dict[str, dict] = {}
    for p in sorted(run_dir.glob("result_*.json")):
        if _WRAPPER_RE.search(p.name):
            continue
        d = _read_json(p)
        if isinstance(d, dict):
            recs[d.get("client_id") or p.stem.replace("result_", "")] = d

    tracks = []
    for wav, node in _capture_wavs(run_dir, _run_node_ids(run_dir, manifest)):
        d = recs.get(node) or {}
        tr = capture_track.capture_track(wav, band_hz=_track_band(d, params))
        ev = _track_events(d)
        _attribute_own_tx(tr, ev.get("tx_windows") or [])
        tracks.append({
            "client_id": node,
            "file":      wav.name,
            "role":      d.get("role"),
            "status":    d.get("status"),
            "track":     tr,
            **ev,
        })
    return tracks


def _attribute_own_tx(tr: dict, tx_windows: list) -> None:
    """Reclassify clipped windows that fall inside this node's own TX spans.

    capture_track only sees the WAV, so it can tell an impulse from a sustained
    burst but not a sustained burst from THE NODE'S OWN SPEAKER -- which couples
    into its own mic at full scale and reads exactly like an over-driven input.
    Correlating here, where both the track and the TX log exist, keeps the
    chips and the verdict agreeing with the chart.
    """
    s = tr.get("summary")
    if not isinstance(s, dict) or not tx_windows:
        return
    ts = tr.get("t") or []
    cp = tr.get("clip_pct") or []
    imp = tr.get("clip_impulse") or []
    spans = [(float(w["t"]) - 0.3,
              float(w["t"]) + float(w.get("dur_s") or 0) + 0.3)
             for w in tx_windows if w.get("t") is not None]
    own = imp_n = sus = 0
    flags = []
    for i, t in enumerate(ts):
        clipped = i < len(cp) and cp[i] > 0
        in_tx = any(a <= t <= b for a, b in spans)
        flags.append(1 if (clipped and in_tx) else 0)
        if not clipped:
            continue
        if in_tx:
            own += 1
        elif i < len(imp) and imp[i]:
            imp_n += 1
        else:
            sus += 1
    tr["clip_own_tx"] = flags
    s["clip_windows_own_tx"] = own
    s["clip_windows_impulse"] = imp_n
    s["clip_windows_sustained"] = sus


def _overall_status(decodes: list[dict], nodes: list[dict]) -> str:
    """Roll a run up to one word.  Decode outcome wins when present — a node
    can exit 0 while the modem decoded nothing."""
    statuses = [d.get("status") for d in decodes] or [n.get("status") for n in nodes]
    statuses = [s for s in statuses if s]
    if not statuses:
        return "unknown"
    if any(s == "error" for s in statuses):
        return "error"
    if all(s == "ok" for s in statuses):
        return "ok"
    # "partial" means the run produced its measurement but a secondary stage
    # (e.g. ofdm_fdd's payload exchange) did not verify.  Rolling that up to
    # "fail" hid a complete, valid duplex measurement behind a red word and
    # sent the operator looking for a broken test instead of reading the
    # result -- so it keeps its own state rather than collapsing.
    if all(s in ("ok", "partial") for s in statuses):
        return "partial"
    return "fail"


def run_report(run_id: str, include_files: bool = True,
               include_tracks: bool = True) -> dict | None:
    """Summarise one run directory for the Reports tab.

    `include_tracks` is separate from `include_files` because the level/S/N
    tracks are the one expensive part of this function: they read and FFT every
    capture WAV in the run (~0.5s per 3-minute capture, cached afterwards).  The
    Reports LIST calls this once per run and never draws them, so it opts out --
    with them on, /api/results took 12s on a 20-run results dir.
    """
    run_dir = RESULTS_DIR / run_id
    if not run_dir.is_dir():
        return None

    manifest = _read_json(run_dir / "manifest.json") or {}
    params   = manifest.get("params") or {}
    info     = _read_json(run_dir / "css_info.json") or {}
    decodes  = _decode_results(run_dir)
    nodes    = _node_results(run_dir)

    files, size = [], 0
    for f in run_dir.iterdir():
        try:
            if f.is_file():
                size += f.stat().st_size
        except OSError:
            pass
        files.append(f.name)

    cap = find_run_capture(run_id)
    ann = read_annotation(run_id)

    # Two-way ranging: the test only CAPTURES; distance comes from the post-hoc
    # css_ranging_analyze.py pass, which writes ranging_result.json.  Report
    # both whether that pass can run and its result if it already has.
    ranging_roles = set()
    for p in run_dir.glob("ranging_*.json"):
        meta = _read_json(p) or {}
        if meta.get("role"):
            ranging_roles.add(meta["role"])

    rep = {
        "run_id":     run_id,
        "test":       manifest.get("test") or (nodes[0]["test"] if nodes else None),
        "started_at": manifest.get("started_at"),
        "roles_map":  manifest.get("roles_map") or {},
        "status":     _overall_status(decodes, nodes),
        "message":    params.get("message"),
        "symbol_dur": params.get("symbol_dur", info.get("symbol_dur")),
        "amplitude":  params.get("amplitude", info.get("amplitude")),
        "decodes":    decodes,
        # Mesh runs carry their payload without producing a decode sidecar, so
        # delivery has to be read out of the node records instead.
        "delivery":   _mesh_delivery(run_dir, manifest),
        "nodes":      nodes,
        "size_bytes": size,
        "annotation": ann,
        "replayable": bool(cap["wav"] and cap["info"]),
        "replay_note": cap["reason"],
        "spectrographs": sorted(p.name for p in run_dir.glob("*spectrograph*.png")),
        # Constellations are built by an explicit post-hoc pass (~11s per
        # node), so the report only LISTS what already exists -- same contract
        # as the spectrographs and the ranging analyzer.  The row list drives
        # the call-flow table; one cell PNG per moment per node sits beside it.
        "constellation_timeline": _read_json(
            run_dir / "constellation_timeline.json") or None,
        "summary":    _read_json(run_dir / "summary.json") or {},
        "ofdm":          _ofdm_summary(run_dir, include_spectrum=include_files),
        # Output side of the level story: the only gain this stack actually
        # CONTROLS is TX amplitude (input gain lives in the OS mixer), so the
        # level chart names it next to the measured input level.
        "tx_gain":       {"amplitude": params.get("amplitude"),
                          "ofdm_amplitude": params.get("ofdm_amplitude"),
                          "amp_map": params.get("amp_map") or {}},
        "ranging":       _read_json(run_dir / "ranging_result.json"),
        "ranging_ready": {"initiator", "responder"} <= ranging_roles,
        "ranging_roles": sorted(ranging_roles),
    }
    if include_files:
        rep["files"] = sorted(files)
    if include_tracks:
        rep["tracks"] = _capture_tracks(run_dir, manifest, params)
    return rep


def suggested_expect(run_id: str) -> str:
    """Best guess at ground-truth text for a run: the message that was sent."""
    manifest = _read_json(RESULTS_DIR / run_id / "manifest.json") or {}
    msg = (manifest.get("params") or {}).get("message")
    if msg:
        return str(msg)
    for d in _decode_results(RESULTS_DIR / run_id):
        if d.get("status") == "ok" and d.get("decoded_text"):
            return d["decoded_text"]
    return ""


def list_results() -> list[dict]:
    runs = []
    for run_dir in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if run_dir.is_dir():
            rep = run_report(run_dir.name, include_tracks=False)
            if rep:
                runs.append(rep)
    return runs


def delete_run(run_id: str) -> dict:
    """Delete a run directory.  Refuses if the run backs a regression case —
    unpromote it first (which copies nothing back, the fixture is already a
    standalone copy, but the refusal keeps the pairing honest)."""
    run_dir = (RESULTS_DIR / run_id).resolve()
    if RESULTS_DIR.resolve() not in run_dir.parents:
        return {"ok": False, "error": "path escapes results dir"}
    if not run_dir.is_dir():
        return {"ok": False, "error": "run not found"}
    shutil.rmtree(run_dir)
    return {"ok": True, "run_id": run_id}


# ── regression promotion ─────────────────────────────────────────────────────
# A promoted run is COPIED into tests/regression/fixtures/runs/<case>/ so the
# case survives deleting the result dir, and recorded in cases.json, which
# css_regress.py reads at startup.

def load_regression_cases() -> list[dict]:
    data = _read_json(REGRESSION_CASES) or {}
    cases = data.get("cases") if isinstance(data, dict) else data
    return cases if isinstance(cases, list) else []


def _save_regression_cases(cases: list[dict]):
    _write_json_atomic(REGRESSION_CASES, {"cases": cases})


def promote_run(run_id: str, expect: str | None = None,
                labels: list[str] | None = None, note: str = "",
                case_name: str | None = None) -> dict:
    """Copy a run's capture into the regression fixtures and register a case."""
    rep = run_report(run_id, include_files=False, include_tracks=False)
    if rep is None:
        return {"ok": False, "error": "run not found"}

    cap = find_run_capture(run_id)
    if not (cap["wav"] and cap["info"]):
        return {"ok": False, "error": cap["reason"] or "run is not replayable"}

    ann = read_annotation(run_id)
    name = case_name or ann.get("case_name") or f"run_{run_id}"
    if expect is None:
        expect = ann.get("expect")
    if expect is None:
        expect = suggested_expect(run_id)
    labels = sorted({str(x).strip() for x in (labels if labels is not None
                                              else ann.get("labels") or []) if str(x).strip()})

    dest = RUN_FIXTURES_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cap["wav"], dest / "input.wav")
    shutil.copy2(cap["info"], dest / "info.json")

    case = {
        "name":       name,
        "run_id":     run_id,
        "wav":        str((dest / "input.wav").relative_to(REGRESSION_DIR)).replace("\\", "/"),
        "info":       str((dest / "info.json").relative_to(REGRESSION_DIR)).replace("\\", "/"),
        "expect":     expect,
        "labels":     labels,
        "note":       note or ann.get("note") or "",
        "test":       rep.get("test"),
        "symbol_dur": rep.get("symbol_dur"),
        "added_at":   datetime.now().isoformat(timespec="seconds"),
    }
    cases = [c for c in load_regression_cases() if c.get("name") != name]
    cases.append(case)
    cases.sort(key=lambda c: c.get("name", ""))
    _save_regression_cases(cases)

    write_annotation(run_id, {"verdict": "valid", "expect": expect,
                              "labels": labels, "case_name": name,
                              **({"note": note} if note else {})})
    return {"ok": True, "case": case}


def unpromote_run(run_id: str) -> dict:
    """Remove a run's regression case and its copied fixture."""
    ann  = read_annotation(run_id)
    name = ann.get("case_name") or f"run_{run_id}"
    cases = load_regression_cases()
    kept  = [c for c in cases if c.get("name") != name and c.get("run_id") != run_id]
    dropped = [c for c in cases if c not in kept]
    _save_regression_cases(kept)
    for c in dropped:
        fixture = RUN_FIXTURES_DIR / c.get("name", "")
        if fixture.is_dir() and RUN_FIXTURES_DIR.resolve() in fixture.resolve().parents:
            shutil.rmtree(fixture, ignore_errors=True)
    if (RESULTS_DIR / run_id).is_dir():
        write_annotation(run_id, {"case_name": None})
    return {"ok": True, "removed": [c.get("name") for c in dropped]}