#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Boot each node script far enough to prove its CONFIG BLOCK executes.

No offline sim runs a node script's startup path: the sim tiers exercise the
libraries (meshlib, ofdm_modem) directly and skip the scripts that configure
them, so the whole PARAMS -> HS_OVERRIDES -> log region is untested and a
config-block bug there only surfaces as a dead OTA run.

This runs each script in a subprocess with representative PARAMS and asserts
it reaches the point where it opens an audio device. It needs no sound card:
the device failure IS the success marker, so anything failing earlier than
that is a config-block bug. Every param combination that changes geometry or
allocation policy gets a case, since those are the lines most often edited.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent

# Reaching ANY of these means the config block ran to completion.
REACHED = ("MESH NODE", "sounddevice", "InputStream", "OutputStream",
           "capture worker", "AUDIO_IFACE", "LISTENING", "no such device",
           "PortAudio", "Error querying device")

# Names that mean the config block itself is broken.
BROKEN = ("NameError", "AttributeError", "KeyError", "TypeError",
          "ImportError", "ModuleNotFoundError", "SyntaxError",
          "UnboundLocalError", "ValueError", "ZeroDivisionError")

BASE = {
    "run_s": 3, "run_timeout": 30, "wait_probe_s": 5, "wait_msg_s": 0,
    # Mutual, matching meshmodem_node's own default. Both endpoints get a
    # data_dst, so this is the path that needs _link_initiator to pick one
    # leader.
    "hs_freq_lo": 300, "hs_freq_hi": 6000, "data_map": {"1": 2, "2": 1},
    "data_payload": "hi",
}

CASES = [
    ("meshmodem_node.py", "defaults",           {}),
    ("meshmodem_node.py", "isi_derate on",      {"isi_derate": 1,
                                                 "snr_margin_db": 2.0}),
    ("meshmodem_node.py", "isi_derate + big margin",
                                                {"isi_derate": 1,
                                                 "snr_margin_db": 9.0}),
    ("meshmodem_node.py", "long CP (n_fft)",    {"n_fft": 2048}),
    ("meshmodem_node.py", "long CP (cp_ratio)", {"cp_ratio": 0.5}),
    ("meshmodem_node.py", "css_only",           {"phy_mode": "css_only"}),
    ("meshmodem_node.py", "percentile SNR",     {"snr_estimator": "percentile"}),
    ("meshmodem_node.py", "big mesh_chunk",     {"mesh_chunk": 512,
                                                 "data_payload": "y" * 500}),
    # Full duplex needs EXACTLY two participants, so it must be given a 2-entry
    # id_map -- the default map has three and the mode correctly refuses it
    # (that refusal is asserted separately below, as a non-boot check).
    ("meshmodem_node.py", "ofdm_full_duplex",   {"phy_mode": "ofdm_full_duplex",
                                                 "id_map": {"macdesktop": 1,
                                                            "laptop": 2}}),
    ("ofdmhs_node.py",    "defaults",           {}),
    ("ofdmhs_node.py",    "isi_derate on",      {"isi_derate": 1,
                                                 "snr_margin_db": 2.0}),
    ("ofdmhs_node.py",    "long CP (n_fft)",    {"n_fft": 2048}),
    ("ofdm_fdd_node.py",  "defaults",           {}),
    ("ofdm_fdd_node.py",  "asymmetric 90/10",   {"dl_frac": 0.9}),
    ("ofdm_fdd_node.py",  "dl_side low",        {"dl_side": "low"}),
    ("ofdm_fdd_node.py",  "no data phase",      {"data_s": 0}),
]

# The oldest Python any node host runs. Node scripts are launched by whatever
# interpreter that machine has, not the server's, so a syntax feature newer
# than this is a crash on that host alone, invisible to every offline suite
# (which all run here).
MIN_PY = (3, 9)

# Cases that must be REJECTED at config time rather than reaching the air.
# (script, label, params, substring the rejection must contain)
REJECT_CASES = [
    ("meshmodem_node.py", "oversized data_payload",
     {"data_payload": "z" * 1504}, "mesh_chunk>=1508"),
]

ok_all = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"  [{name:34s}] {'PASS' if cond else 'FAIL'}"
          f"{('  ' + detail) if detail and not cond else ''}")


def find(script: str) -> Path:
    for sub in ("", "MESH", "OFDM", "QAM", "CSS"):
        p = TESTS / sub / script if sub else TESTS / script
        if p.exists():
            return p
    raise FileNotFoundError(script)


def boot(script: str, params: dict, run_dir: Path) -> str:
    path = find(script)
    env = dict(os.environ)
    env["AUDIOTOOL_PARAMS"] = json.dumps({**BASE, **params})
    env["AUDIOTOOL_CLIENT_ID"] = "laptop"
    env["AUDIOTOOL_RUN_DIR"] = str(run_dir)
    env.setdefault("AUDIOTOOL_ROLE", "node")
    try:
        p = subprocess.run([sys.executable, str(path)], cwd=str(path.parent),
                           env=env, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired as e:
        return ((e.stdout or "") if isinstance(e.stdout, str)
                else (e.stdout or b"").decode()) + "\n[TIMEOUT]"
    return (p.stdout or "") + (p.stderr or "")


def py_compat_offenders() -> list[tuple[str, int, str]]:
    """Node scripts using PEP-604 (`X | Y`) annotations without __future__.

    Checked statically, by parsing, because booting cannot find this: the
    smoke test runs under THIS host's interpreter and the bug only exists
    under the older one on a node. Annotations are evaluated at def-time, so
    `-> dict | None` raises TypeError at import on < 3.10 -- unless
    `from __future__ import annotations` makes them lazy strings, which is why
    ofdm_modem.py and ofdm_handshake.py carry it and why anything launched on
    a node must too.
    """
    import ast
    out: list[tuple[str, int, str]] = []
    for p in sorted(TESTS.rglob("*.py")):
        src = p.read_text(encoding="utf-8", errors="replace")
        if "PARAMS" not in src and "AUDIOTOOL_" not in src:
            continue                       # not launchable on a node
        if "from __future__ import annotations" in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            out.append((str(p.relative_to(TESTS)), e.lineno or 0, "syntax error"))
            continue

        def flag(ann, where):
            if ann is None:
                return
            for s in ast.walk(ann):
                if isinstance(s, ast.BinOp) and isinstance(s.op, ast.BitOr):
                    out.append((str(p.relative_to(TESTS)),
                                getattr(ann, "lineno", 0), where))
                    return

        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                flag(n.returns, f"{n.name}() return annotation")
                a = n.args
                for arg in (list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)
                            + ([a.vararg] if a.vararg else [])
                            + ([a.kwarg] if a.kwarg else [])):
                    flag(arg.annotation, f"{n.name}(arg {arg.arg}) annotation")
            elif isinstance(n, ast.AnnAssign):
                flag(n.annotation, "variable annotation")
    return out


def main() -> int:
    run_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "node_boot_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    print("node config-block boot smoke")

    offenders = py_compat_offenders()
    check(f"py{MIN_PY[0]}.{MIN_PY[1]} annotation compat", not offenders,
          "; ".join(f"{f}:{ln} {w}" for f, ln, w in offenders[:4]))
    for script, label, params in CASES:
        out = boot(script, params, run_dir)
        reached = any(m in out for m in REACHED)
        # A crash BEFORE the audio stage is a config-block bug.  One after it is
        # this host having no sound card, which is not what we are testing.
        head = out.split("MESH NODE")[0] if "MESH NODE" in out else out
        early = None
        if not reached:
            for b in BROKEN:
                m = re.search(rf"^{b}:.*$", head, re.M)
                if m:
                    early = m.group(0)
                    break
        check(f"{script.replace('_node.py','')}: {label}", reached,
              early or (out.strip().splitlines() or ["no output"])[-1][:160])

    # A bad config must be caught before the run starts, not crash the node
    # partway through an OTA run.
    for script, label, params, want in REJECT_CASES:
        out = boot(script, params, run_dir)
        good = ("FATAL" in out and want in out
                and "Traceback" not in out.split("FATAL")[-1])
        check(f"{script.replace('_node.py','')}: rejects {label}", good,
              (out.strip().splitlines() or ["no output"])[-1][:160])
    # level_stats runs only when a burst ARRIVES, so no boot case reaches it --
    # yet it is where an impulse gets told apart from a hot input, and where a
    # stray `math.` reference would kill a live run the first time the room made
    # a loud noise.  Exercise it directly, in a subprocess so importing the node
    # script's module-level code stays contained.
    probe_src = """
import numpy as np, sys
import meshmodem_node as M
fs = 48000
# a) environmental impulse: 2ms full-scale click in an otherwise quiet window
imp = (np.random.RandomState(1).randn(fs * 3) * 0.02).astype(np.float64)
imp[fs] = 1.0; imp[fs + 1:fs + 96] = 0.999
a = M.level_stats(imp, fs)
# b) sustained hot burst: a loud tone filling the window
t = np.arange(fs * 3) / fs
hot = 0.999 * np.sin(2 * np.pi * 1000 * t)
b = M.level_stats(hot, fs)
assert a["clip_pct"] > 0 and b["clip_pct"] > 0, (a, b)
imp_kind = a["crest_db"] >= 20.0 or a["hot_ms"] <= 20.0
hot_kind = b["crest_db"] >= 20.0 or b["hot_ms"] <= 20.0
print("IMPULSE", a["crest_db"], a["hot_ms"], imp_kind)
print("SUSTAINED", b["crest_db"], b["hot_ms"], hot_kind)
assert imp_kind, "impulse not classified as impulsive"
assert not hot_kind, "sustained tone misclassified as impulsive"
print("LEVEL_STATS_OK")
"""
    path = find("meshmodem_node.py")
    env = dict(os.environ)
    env["AUDIOTOOL_PARAMS"] = json.dumps(BASE)
    env["AUDIOTOOL_CLIENT_ID"] = "laptop"
    env["AUDIOTOOL_RUN_DIR"] = str(run_dir)
    try:
        p = subprocess.run([sys.executable, "-c", probe_src], cwd=str(path.parent),
                           env=env, capture_output=True, text=True, timeout=120)
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        out = "[TIMEOUT]"
    check("meshmodem: level_stats tells impulse from hot input",
          "LEVEL_STATS_OK" in out,
          (out.strip().splitlines() or ["no output"])[-1][:160])

    # _link_initiator: on a MUTUAL data_map both endpoints have data_dst ==
    # the other, and initiator=True unconditionally would make BOTH sides
    # enter the shared handshake as "initiator" at once -- a live collision no
    # OTA report has ever exercised, because every data_map used so far was
    # one-way.  Pure function, but probed the same subprocess way as
    # level_stats above so importing the node script stays contained.
    li_src = """
import meshmodem_node as M
one_way = {1: 2}
two_way = {1: 2, 2: 1}
li_1 = M._link_initiator(one_way, 1, 2)
li_2 = M._link_initiator(two_way, 1, 2)
li_3 = M._link_initiator(two_way, 2, 1)
print("ONE_WAY", li_1)
print("TWO_WAY", li_2, li_3)
assert li_1 is True, "one-way data_map behaviour changed"
assert li_2 != li_3, "mutual data_map: both sides would lead -- collision"

# --- full duplex refuses anything but exactly 2 participants ---
# Full duplex splits ONE band into a DL and a UL half; there is no third half,
# and a third node's carrier sense would read the duplex pair as permanently
# busy.  It must refuse LOUDLY rather than quietly degrade to half duplex,
# which would look like the mode working badly instead of not running.
R = M._full_duplex_refusal
two   = {"macdesktop": 1, "laptop": 2}
three = {"macdesktop": 1, "laptop": 2, "delllaptop": 3}
assert R("ofdm_full_duplex", two) is None, "refused a legal 2-node full duplex"
assert R("ofdm_full_duplex", three), "allowed full duplex with 3 participants"
assert R("ofdm_full_duplex", {"laptop": 2}), "allowed full duplex with 1"
# ...and never interferes with the half-duplex modes at any node count.
for hd in ("css_only", "css_then_ofdm", "ofdm_only"):
    assert R(hd, three) is None, f"{hd} wrongly refused"
    assert R(hd, two) is None, f"{hd} wrongly refused"
print("FULL_DUPLEX_GUARD_OK")

# --- _scenario_exit: when may a node stop early? ---
E = M._scenario_exit
base = dict(acked=False, gave_up=False, dst_gone=False, done_seen=False,
            timed_out=False, outbox_empty=True)

# Mutual data_map: peer finished and announced DONE while my own DATA was
# still unacked.  No ACK is ever coming, so I must stop rather than grind
# through retries and a re-handshake spiral at a departed peer.
stop, why = E(1, [1], {1: b"x"}, **{**base, "dst_gone": True, "done_seen": True})
assert stop, "stranded node will not stop after its peer announced DONE"
assert why == "peer left before acking", why

# ...but NOT before the peer has actually announced anything: still retrying.
stop, _ = E(1, [1], {1: b"x"}, **base)
assert not stop, "stopped while the peer was still live and might yet ACK"

# Normal mutual completion: acked both ways, peer done.
stop, why = E(1, [1], {1: b"x"}, **{**base, "acked": True, "done_seen": True})
assert stop and why == "peers reported done", (stop, why)

# One-way sender (no expected_srcs): unchanged -- exits on its own ACK.
stop, _ = E(2, [], {}, **{**base, "acked": True})
assert stop, "one-way sender behaviour changed"
stop, _ = E(2, [], {}, **base)
assert not stop, "one-way sender exited before its DATA was acked"

# One-way receiver (no data_dst): unchanged -- exits once delivered + quiet.
stop, _ = E(None, [1], {1: b"x"}, **{**base, "done_seen": True})
assert stop, "one-way receiver behaviour changed"
stop, _ = E(None, [1], {}, **{**base, "done_seen": True})
assert not stop, "receiver exited before delivery"

# A queued outbox always blocks the exit, whatever else is true.
stop, _ = E(1, [1], {1: b"x"}, **{**base, "acked": True, "done_seen": True,
                                  "outbox_empty": False})
assert not stop, "exited with packets still queued"
print("SCENARIO_EXIT_OK")
print("LINK_INITIATOR_OK")
"""
    try:
        p = subprocess.run([sys.executable, "-c", li_src], cwd=str(path.parent),
                           env=env, capture_output=True, text=True, timeout=60)
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        out = "[TIMEOUT]"
    check("meshmodem: mutual data_map does not collide on initiator role",
          "LINK_INITIATOR_OK" in out,
          (out.strip().splitlines() or ["no output"])[-1][:160])
    check("meshmodem: node stops when its data_dst announced DONE",
          "SCENARIO_EXIT_OK" in out,
          (out.strip().splitlines() or ["no output"])[-1][:160])
    check("meshmodem: full duplex refuses != 2 participants",
          "FULL_DUPLEX_GUARD_OK" in out,
          (out.strip().splitlines() or ["no output"])[-1][:160])

    print("BOOT SMOKE", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
