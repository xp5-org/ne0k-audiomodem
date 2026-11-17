# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Event log in SQLite, for sessions too long to read as text.  Qt-free.

One file per diagnostics session (``events.sqlite`` beside debug.log and the WAVs)::

    meta(key, value)                         rate, versions, the RX sample index the rx WAV starts at
    log(id, t, rx_index, line)               every Diagnostics.log line
    events(id, t, rx_index, type, kind,      every engine event except the UI-rate Levels / Spectrum;
           node, peer, t0, t1, detail, data) data = the event's other fields as JSON

``t`` is wall-clock (time.time()); ``rx_index`` the count of RX samples received when the row was made, so
``(rx_index - meta.wav_rx_start) / rate`` is the row's position in the rx WAV.  LinkEvent ``t0``/``t1`` stay in
time.monotonic() seconds as the engine reports them; ``meta.mono_minus_wall`` converts.

Rows are queued and written by one thread in batches (WAL, commit every 0.5 s), so a caller on an audio or
decoder thread never waits on the disk.  Query with any SQLite tool, e.g.::

    sqlite3 events.sqlite "select datetime(t,'unixepoch','localtime'), kind, detail from events where type='LinkEvent'"
"""
from __future__ import annotations

import dataclasses
import json
import queue
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

SKIP_TYPES = ("Levels", "Spectrum")       # 20 Hz UI meters: the WAV is the record of those
_MAX_ARRAY = 8192                          # bigger arrays are summarised by shape, not stored

_SCHEMA = """
create table if not exists meta(key text primary key, value text);
create table if not exists log(id integer primary key, t real, rx_index integer, line text);
create table if not exists events(id integer primary key, t real, rx_index integer, type text, kind text,
                                  node integer, peer integer, t0 real, t1 real, detail text, data text);
create index if not exists events_t on events(t);
create index if not exists events_type on events(type, kind);
create index if not exists log_t on log(t);
"""


def _jsonable(v):
    if isinstance(v, np.ndarray):
        if v.size > _MAX_ARRAY:
            return {"shape": list(v.shape), "dtype": str(v.dtype)}
        if np.iscomplexobj(v):
            return {"re": _jsonable(np.round(v.real, 5)), "im": _jsonable(np.round(v.imag, 5))}
        if v.dtype.kind == "f" and not np.isfinite(v).all():       # NaN / inf are not JSON: null
            return np.where(np.isfinite(v), v, None).tolist()
        return v.tolist()
    if isinstance(v, (bytes, bytearray)):
        return v.hex()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and v != v:
        return None                        # NaN is not JSON
    return v


def event_row(ev) -> tuple:
    """(type, kind, node, peer, t0, t1, detail, data_json) for one engine event."""
    typ = type(ev).__name__
    f = {k.name: getattr(ev, k.name) for k in dataclasses.fields(ev)} if dataclasses.is_dataclass(ev) else {}
    kind = f.pop("kind", None) or f.get("state") or f.get("source") or f.get("type_name")
    node, peer = f.pop("node", None), f.pop("peer", None)
    t0, t1 = f.pop("t0", None), f.pop("t1", None)
    detail = f.pop("detail", None) or f.get("message") or f.get("text") or f.get("summary")
    data = json.dumps({k: _jsonable(v) for k, v in f.items()}, default=str) if f else None
    return typ, kind, node, peer, t0, t1, detail, data


class EventDb:
    def __init__(self, path: str | Path, meta: dict | None = None) -> None:
        self.path = Path(path)
        self.rx_index = lambda: None       # set by the owner: current RX sample count
        self.rows = 0
        self._q: "queue.Queue[tuple | None]" = queue.Queue()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("pragma journal_mode=wal")
        self._db.executescript(_SCHEMA)
        m = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "mono_minus_wall": time.monotonic() - time.time()}
        m.update(meta or {})
        self._db.executemany("insert or replace into meta values (?, ?)", [(k, str(v)) for k, v in m.items()])
        self._db.commit()
        self._th = threading.Thread(target=self._run, name="eventdb", daemon=True)
        self._th.start()

    def set_meta(self, key: str, value) -> None:
        self._q.put(("meta", key, str(value)))

    def log(self, line: str) -> None:
        self._q.put(("log", time.time(), self.rx_index(), line))

    def event(self, ev) -> None:
        if type(ev).__name__ in SKIP_TYPES:
            return
        try:
            row = event_row(ev)
        except Exception as e:             # a diagnostics tap must never break the engine
            row = (type(ev).__name__, "unserialisable", None, None, None, None, repr(e), None)
        self._q.put(("ev", time.time(), self.rx_index()) + row)

    def _run(self) -> None:
        stop = False
        while not stop:
            items = []
            try:
                items.append(self._q.get(timeout=0.5))
                while len(items) < 1000:
                    items.append(self._q.get_nowait())
            except queue.Empty:
                pass
            if None in items:
                stop = True
                items = [i for i in items if i is not None]
            if not items:
                continue
            try:
                for it in items:
                    if it[0] == "log":
                        self._db.execute("insert into log(t, rx_index, line) values (?, ?, ?)", it[1:])
                    elif it[0] == "ev":
                        self._db.execute("insert into events(t, rx_index, type, kind, node, peer, t0, t1, detail, "
                                         "data) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", it[1:])
                    else:
                        self._db.execute("insert or replace into meta values (?, ?)", it[1:])
                self._db.commit()
                self.rows += len(items)
            except sqlite3.Error:
                pass                       # disk full etc.: drop the batch, keep the modem running

    def close(self) -> None:
        self._q.put(None)
        self._th.join(5.0)
        try:
            self._db.close()
        except sqlite3.Error:
            pass
