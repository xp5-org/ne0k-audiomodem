# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Programmable BAT: the bit-allocation tables this modem measured, and saved BATs loaded from outside.

A receiver measures the BAT its peer should transmit with from that peer's probe (``OfdmLink._attempt_rx``).  On a
two-way link the handshake carries it back on the air.  On a one-way link (an FM transmitter into an RTL-SDR, a
broadcast) there is no air return path, so the table travels over the control port instead:

    receiver:  BAT LIST              what was measured, per peer
               BAT GET [peer]        one saved-BAT JSON line (bat.json=...)
    sender:    BAT PUT <json>        store it and point udp.bat_file at it; the UDP port comes up from it
                                     with no handshake (engine/udpbridge.py)

Saved-BAT JSON (what ``udpbridge.load_bat`` reads)::

    {"config": {OfdmConfig PHY/band fields}, "tx_bat": "<hex>", "rx_bat": "<hex>", ...provenance}

``tx_bat`` is the sender's TX table.  A one-way measurement has no reverse table, so ``rx_bat`` repeats it (the
listener never transmits on it).  Process-wide like the PHY overrides: every link of this process records here.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import threading
import time

from .modemcfg import OfdmConfig

KEEP = 16                                  # newest measurements kept
# fields that are this node's own business, never imposed by a saved BAT
_LOCAL = ("node_id", "peer_id", "backend", "amplitude", "role", "mode", "sender", "hear_own", "carrier_db",
          "chunk_bytes", "keyup_lead_s")

_lock = threading.Lock()
_measured: list[dict] = []


def record(peer: int, by: int, bat: bytes, cfg: OfdmConfig, info: dict | None = None) -> dict:
    """Keep one probe measurement: ``bat`` is the table ``peer`` should transmit with to reach ``by``."""
    info = info or {}
    e = {"peer": int(peer), "by": int(by), "time": time.time(), "bat": bytes(bat).hex(),
         "config": {k: v for k, v in dataclasses.asdict(cfg).items() if k not in _LOCAL},
         "summary": str(info.get("bat_summary", "")).strip(),
         "snr_median_db": info.get("snr_median_db"), "snr_max_db": info.get("snr_max_db"),
         "isi_ceiling_db": info.get("isi_ceiling_db")}
    with _lock:
        _measured.append(e)
        del _measured[:-KEEP]
    return e


def measured() -> list[dict]:
    with _lock:
        return list(_measured)


def clear() -> None:
    with _lock:
        _measured.clear()


def latest(peer: int | None = None) -> dict | None:
    for e in reversed(measured()):
        if peer is None or e["peer"] == peer:
            return e
    return None


def to_saved(e: dict) -> dict:
    """A measurement -> the saved-BAT JSON a sender (udp.bat_file) loads."""
    return {"config": dict(e["config"]), "tx_bat": e["bat"], "rx_bat": e["bat"],
            "measured_by": e["by"], "measured_from": e["peer"], "measured_at": round(e["time"], 3),
            "summary": e["summary"], "snr_median_db": e["snr_median_db"]}


def parse_saved(text: str) -> dict:
    """Check a saved-BAT JSON line; returns the dict.  Raises ValueError with the reason."""
    try:
        d = json.loads(text)
    except json.JSONDecodeError as ex:
        raise ValueError(f"not JSON: {ex.msg} at {ex.pos}") from None
    if not isinstance(d, dict) or not isinstance(d.get("config"), dict):
        raise ValueError("want an object with config, tx_bat, rx_bat")
    for k in ("tx_bat", "rx_bat"):
        try:
            if not bytes.fromhex(str(d.get(k, ""))):
                raise ValueError
        except ValueError:
            raise ValueError(f"{k} missing or not hex") from None
    names = {f.name for f in dataclasses.fields(OfdmConfig)}
    bad = sorted(set(d["config"]) - names)
    if bad:
        raise ValueError(f"unknown config fields: {', '.join(bad)}")
    try:
        OfdmConfig(**{**dataclasses.asdict(OfdmConfig()), **d["config"]})
    except TypeError as ex:
        raise ValueError(f"config: {ex}") from None
    return d


def describe(d: dict) -> str:
    """One line about a saved BAT or a measurement."""
    c = d.get("config", {})
    tab = bytes.fromhex(d.get("tx_bat") or d.get("bat") or "")
    return (f"n_fft={c.get('n_fft')} cp={c.get('cp_ratio')} band={c.get('freq_lo')}-{c.get('freq_hi')} "
            f"table={len(tab)}B {d.get('summary', '')}".strip())


def write(d: dict, directory: str, name: str = "") -> str:
    """Write a saved BAT under ``directory``; returns the path."""
    os.makedirs(directory, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_") or time.strftime("bat_%Y%m%d_%H%M%S")
    path = os.path.join(directory, slug + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, path)
    return path


def command(arg: str, put=None) -> list[str] | str:
    """The control port's BAT verb.  ``put(saved_dict, name) -> str`` stores a saved BAT (the host decides where
    and what it switches); without it PUT is refused."""
    from .control import ControlError
    op, _, rest = arg.strip().partition(" ")
    op, rest = op.upper(), rest.strip()
    if op in ("", "LIST"):
        now = time.time()
        return [f"bat.measured peer={e['peer']} by={e['by']} age_s={now - e['time']:.0f} "
                f"snr_median_db={e['snr_median_db']} {describe(e)}" for e in measured()]
    if op == "GET":
        try:
            peer = int(rest) if rest else None
        except ValueError:
            raise ControlError("BAT GET [peer]") from None
        e = latest(peer)
        if e is None:
            raise ControlError("no probe measured" + (f" from node {peer}" if peer is not None else "")
                               + " yet: the peer must send a probe (a handshake) first")
        return ["bat.json=" + json.dumps(to_saved(e), separators=(",", ":"))]
    if op == "PUT":
        if put is None:
            raise ControlError("this host cannot take a BAT (use 'modemgui gui --control-port')")
        name = ""
        m = re.match(r"name=(\S+)\s+(.*)$", rest)
        if m:
            name, rest = m.group(1), m.group(2)
        try:
            d = parse_saved(rest)
        except ValueError as ex:
            raise ControlError(f"bad BAT: {ex}") from None
        return put(d, name)
    if op == "CLEAR":
        clear()
        return "measurements cleared"
    raise ControlError("BAT LIST | GET [peer] | PUT [name=N] <json> | CLEAR")
