#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Pull a modem's diagnostics sessions (rx_*.wav, events_*.sqlite, settings_*.json, debug.log) over its control port
(DIAG LIST / DIAG GET) into DEST/<label>/<session>/, for tools/corpus_replay.py.  Resumes: files already complete
are skipped.  tx_*.wav is skipped unless --tx.

    fetchdiag.py DEST label=host[:port] [label=host ...] [--tx] [--since session_20261117]
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import socket
import sys
import time
from pathlib import Path

CHUNK = 2 << 20


def ask(host: str, port: int, cmd: str, maxwait: float = 120.0) -> str:
    s = socket.create_connection((host, port), timeout=10)
    s.settimeout(2)
    try:
        try:
            s.recv(200)                                  # banner
        except socket.timeout:
            pass
        s.sendall((cmd + "\n").encode())
        buf, t = b"", time.time()
        while time.time() - t < maxwait:
            try:
                d = s.recv(1 << 20)
            except socket.timeout:
                continue
            if not d:
                break
            buf += d
            last = buf.rstrip().split(b"\n")[-1]
            if last.startswith((b"OK", b"ERR")):
                break
        return buf.decode(errors="replace")
    finally:
        s.close()


def kv(lines: str, key: str) -> str | None:
    for l in lines.splitlines():
        if l.startswith(key + "="):
            return l.split("=", 1)[1]
    return None


def fetch_host(dest: Path, label: str, addr: str, tx: bool, since: str) -> tuple[int, int]:
    host, _, port = addr.partition(":")
    port = int(port or 5555)
    files = []
    for l in ask(host, port, "DIAG LIST", 60).splitlines():
        if l.startswith("diag.file="):
            name, size = l.split("=", 1)[1].rsplit(" ", 1)
            if name.split("/")[0] < since:
                continue
            if not tx and Path(name).name.startswith("tx_"):
                continue
            files.append((name, int(size)))
    got = 0
    for name, size in files:
        out = dest / label / name
        if out.exists() and out.stat().st_size == size:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_suffix(out.suffix + ".part")
        off = part.stat().st_size if part.exists() else 0
        with open(part, "ab") as fh:
            while off < size:
                for attempt in range(3):
                    r = ask(host, port, f"DIAG GET {name} {off} {CHUNK}", 120)
                    b, n = kv(r, "diag.b64"), kv(r, "diag.n")
                    try:
                        data = base64.b64decode(b or "", validate=True)
                    except Exception:
                        data = None
                    if b is not None and data is not None and n is not None and len(data) == int(n):
                        break
                else:
                    raise RuntimeError(f"{label} {name} @{off}: short or bad chunk ({r[-120:]!r})")
                if not data:
                    break
                fh.write(data)
                off += len(data)
        part.rename(out)
        got += size
        print(f"{label}: {name} {size / 1e6:.1f} MB", flush=True)
    return len(files), got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", type=Path)
    ap.add_argument("hosts", nargs="+", help="label=host[:port]")
    ap.add_argument("--tx", action="store_true")
    ap.add_argument("--since", default="", help="only sessions whose name sorts >= this")
    a = ap.parse_args()
    with cf.ThreadPoolExecutor(len(a.hosts)) as ex:
        futs = {ex.submit(fetch_host, a.dest, *h.split("=", 1), a.tx, a.since): h for h in a.hosts}
        for f in cf.as_completed(futs):
            n, b = f.result()
            print(f"{futs[f]}: {n} files listed, {b / 1e6:.0f} MB fetched", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
