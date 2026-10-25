# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""
ofdm_regress.py -- runs the standalone OFDM regression scripts
(ofdm_cfg/mod/sync/probe/acq_regress.py, ofdmhs_collide_regress.py) as one
suite, each as its own case, in the same "PASS/FAIL/SKIP name [secs]s detail"
line format css_regress.py prints -- so the web UI's regression runner and
table work for OFDM the same way they do for CSS.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

CASES = [
    "ofdm_cfg_regress",
    "ofdm_mod_regress",
    "ofdm_sync_regress",
    "ofdm_probe_regress",
    "ofdm_acq_regress",
    "ofdmhs_collide_regress",
]


def run_case(name, verbose):
    t0 = time.time()
    args = [sys.executable, "-u", str(HERE / f"{name}.py")]
    if verbose:
        args.append("-v")
    proc = subprocess.run(args, capture_output=True, text=True)
    dt = time.time() - t0
    out = (proc.stdout + proc.stderr).strip()
    lines = out.splitlines()
    stdout_lines = proc.stdout.strip().splitlines()
    detail = next((ln.strip() for ln in reversed(stdout_lines) if ln.strip()), "")
    if any(ln.lstrip().startswith("SKIP") for ln in lines):
        status = "SKIP"
    elif proc.returncode == 0:
        status = "PASS"
    else:
        status = "FAIL"
    return status, detail, out, dt


def main():
    ap = argparse.ArgumentParser(description="OFDM regression suite")
    ap.add_argument("--only", help="run only cases whose name contains this")
    ap.add_argument("--list", action="store_true",
                    help="list cases and exit")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="pass -v through to each case; print output for failing cases")
    args = ap.parse_args()

    cases = [c for c in CASES if not args.only or args.only in c]
    if not cases:
        print("no cases match filter"); return 1

    if args.list:
        for c in cases:
            print(f"  {c}")
        print(f"\n{len(cases)} case(s)")
        return 0

    print(f"OFDM REGRESSION -- {len(cases)} cases\n")
    all_ok = True
    n_skip = 0
    for name in cases:
        status, detail, out, dt = run_case(name, args.verbose)
        if status == "SKIP":
            n_skip += 1
        else:
            all_ok &= status == "PASS"
        print(f"  {status}  {name:28s} [{dt:6.1f}s]  {detail}")
        if status == "FAIL" and args.verbose and out:
            print("        --- case output tail ---")
            for ln in out.splitlines()[-40:]:
                print("        " + ln)

    skip_note = f" ({n_skip} skipped)" if n_skip else ""
    print(f"\n*** OFDM REGRESSION: {'ALL PASS' if all_ok else 'FAILURES'}"
          f"{skip_note} ***")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
