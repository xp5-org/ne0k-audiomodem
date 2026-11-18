# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Live CSS receive for long symbols (symbol_dur > 0.25 s): the clock stays in focus at every decode tier.

``CssLink`` uses this when ``CssConfig.decoder == "pll_c"`` and the symbols are longer than the C streaming
PLL handles (0.25 s).  Before it, those bursts were recorded whole and decoded cold after they ended, with no
timing loop running while they arrived.  Here every RX sample goes, as it arrives, to:

* ``css_pll_c.TrkStream`` -- the running chirp-tracker PLL bank (one grid per station heard, our own TX
  schedule, carrier sense, ppm against our clock).  It never stops, so the clock survives across bursts.
* a per-burst live session of ``css_decoder.decode_longsym_mf``, fed through a queue.  Each burst
  re-acquires its symbol phase (a new transmission starts anywhere), seeded with the sender's clock rate
  as last measured (``mf_grid_sps_seed``), and works through the decode tiers:

  1. PLL active: the current symbol on the historic grid fit (t0, samples/symbol): the CSS clock.
  2. Each symbol searched in a predicted window (±track_search), with a 12-slice matched filter.
  3. PLL lost or in doubt (a CRC failure; weak / off-grid symbols counted as doubt): re-scan the
     failed chunk's own audio -- least-confident-symbol chase, whole-symbol search with a local grid
     refit, preamble offset ±1 -- and re-centre the live grid on what the chunk says
     (css_decoder ``mf_rescan``).
  4. Chunk still missing at burst end (a CRC-failed chunk, a gap in the chunk numbers, or fewer chunks
     than the burst could hold): a batch re-scan of the whole burst on a worker thread, merged by chunk
     number.  Clock tracking does not wait for it.
  5. Still missing: the chunk / packet is declared lost.

Chunks are reassembled into css_modem packets by chunk number (chunk 0 starts a packet).
"""
from __future__ import annotations

import collections
import itertools
import json
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import cpustats

# decoder log lines worth a GUI log line (the rest -- MIC_SN per symbol etc. -- are dropped)
_LOG_PREFIX = ("MF_ACQUIRE", "MF_RESCAN", "MF_FRAME_ERROR", "MF_RELOCK", "MF ERROR", "hook error")
_SEED_MAX_AGE_S = 1800.0         # a measured clock older than this is not used as a seed
_AUDIO_CAP_S = 1200.0            # tier 4 keeps at most this much of a burst


class _QStream:
    """The decoder's input: read(n) blocks until n samples arrived (or the burst ended)."""

    def __init__(self) -> None:
        self._q: collections.deque = collections.deque()
        self._n = 0
        self._closed = False
        self.waiting = False          # the reader is blocked for more input (tests pace on it)
        self._cv = threading.Condition()

    def push(self, x: np.ndarray) -> None:
        with self._cv:
            self._q.append(x); self._n += len(x)
            self._cv.notify()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n: int):
        with self._cv:
            while self._n < n and not self._closed:
                self.waiting = True
                self._cv.wait(0.5)
            self.waiting = False
            take = min(n, self._n)
            out, got = [], 0
            while got < take:
                a = self._q.popleft()
                if got + len(a) > take:
                    k = take - got
                    self._q.appendleft(a[k:]); a = a[:k]
                out.append(a); got += len(a)
            self._n -= got
        return (np.concatenate(out) if out else np.zeros(0, np.float32)), False


class _ArrayStream(_QStream):
    def __init__(self, x: np.ndarray) -> None:
        super().__init__()
        self.push(np.asarray(x, np.float32))
        self.close()


def _packets(chunks: list[dict]) -> list[dict]:
    """Chunk records in arrival order -> packets {seq: data}: chunk 0 (or a number that goes back) starts one."""
    out: list[dict] = []
    for f in chunks:
        s = int(f["seq"])
        if not out or s == 0 or s in out[-1]["chunks"] or s < max(out[-1]["chunks"]):
            out.append({"chunks": {}, "recs": []})
        out[-1]["chunks"][s] = bytes.fromhex(f["bytes_hex"])
        out[-1]["recs"].append(f)
    return out


def _gaps(p: dict) -> list[int]:
    have = p["chunks"]
    return [s for s in range(0, max(have) + 1) if s not in have] if have else [0]


class _Session(threading.Thread):
    """One burst: a live decoder run, then tiers 4-5 and packet delivery, all on this thread."""

    def __init__(self, rx: "CssLongRx", pre: np.ndarray, seed_sps: float | None, n: int) -> None:
        super().__init__(name=f"css-mf-{n}", daemon=True)
        self.rx, self.seed_sps, self.n = rx, seed_sps, n
        self.q = _QStream()
        self.audio: list[np.ndarray] = []
        self.audio_n = 0
        self.chunks: list[dict] = []
        self.errors: list[dict] = []
        self.rescans = {"ok": 0, "lost": 0}
        self.tier4_added = 0
        self.grid_sps: float | None = None
        self.t_start = time.monotonic()
        self.push(pre)

    def push(self, x: np.ndarray) -> None:
        self.q.push(x)
        if self.audio_n < _AUDIO_CAP_S * self.rx.fs:
            self.audio.append(x); self.audio_n += len(x)

    def end(self) -> None:
        self.q.close()

    # -- decoder hooks (this thread) --------------------------------------------------------------
    def _print(self, *a, **k) -> None:
        line = " ".join(str(x) for x in a)
        if line.startswith(_LOG_PREFIX):
            self.rx.log(f"css mf[{self.n}]: {line[:200]}")

    def _on_symbol(self, s: dict) -> None:
        self.grid_sps = s["grid_sps"]
        self.rx._clock_update(s, self)

    def _on_event(self, e: dict) -> None:
        k = e["event"]
        rx = self.rx
        now = time.monotonic()
        if k == "acquire":
            rx.emit_link("sync", -1, now, now, f"mf acquired, grid {rx.ppm(e['grid_sps']):+.0f} ppm (seed)")
        elif k == "preamble":
            rx.clock["state"] = "locked"
        elif k == "frame_error":
            self.errors.append(e)
            rx.clock["state"] = "doubt"
        elif k == "rescan":
            self.rescans["ok" if e["ok"] else "lost"] += 1
            rx.clock["state"] = "locked" if e["ok"] else "doubt"
            if e["ok"]:
                rx.emit_link("phase", -1, now, now, f"tier3 rescan recovered chunk ({e['how']})")
        elif k == "relock":
            rx.emit_link("phase", -1, now, now, f"mf relock {e['delta']:+d} smp")
        elif k == "cadence_drop":
            rx.clock["state"] = "coast"

    def _on_frame(self, f: dict) -> None:
        f = dict(f, t=time.time())
        self.chunks.append(f)
        rx = self.rx
        rx.log(f"css rx chunk {f['seq']} via {f.get('source', 'mf')} snr={f.get('snr_db', 0):.1f}dB "
               f"grid {rx.ppm(f.get('grid_sps')):+.1f} ppm doubt={f.get('doubt', 0)}")
        now = time.monotonic()
        rx.emit_link("lock", -1, now, now, f"chunk {f['seq']}")

    def _cfg(self, info: str, stream, live: bool) -> dict:
        c = {"info": info, "input_stream": stream, "fs": self.rx.fs, "timeout_sec": 1e7,
             "result_out": str(self.rx.work / f"mf_{self.n}_{'live' if live else 'batch'}.json"),
             "mf_dedupe_seqs": False, "print": self._print if live else (lambda *a, **k: None),
             "mf_grid_sps_seed": self.seed_sps, **self.rx.dec}
        if live:
            c.update(on_symbol=self._on_symbol, on_event=self._on_event, on_frame=self._on_frame)
        return c

    # -- the burst -------------------------------------------------------------------------------
    def run(self) -> None:
        rx = self.rx
        dec = rx.decoder_mod()
        try:
            with cpustats.stage("decode"):
                dec.decode_longsym_mf(self._cfg(rx.info, self.q, True), stop_event=rx.stop_event)
        except Exception as e:                          # the decoder must never kill RX
            rx.log(f"css mf[{self.n}]: decoder error: {e}")
        try:
            self._finish()
        except Exception as e:
            rx.log(f"css mf[{self.n}]: finish error: {e}")
        finally:
            self.audio = []
            rx._session_done(self)

    def _expected_chunks(self) -> int:
        """Whole frames the burst could hold (an upper bound on what was sent; lead-in and ID make it loose)."""
        return int(self.audio_n / self.rx.fs / self.rx.frame_s)

    def _finish(self) -> None:
        rx = self.rx
        pkts = _packets(self.chunks)
        want4 = (bool(self.errors) or any(_gaps(p) for p in pkts)
                 or (not pkts and self._expected_chunks() >= 1)
                 or len(self.chunks) < self._expected_chunks())
        tier4 = None
        if want4 and not rx.stop_event.is_set():
            tier4 = self._tier4(pkts)
        for p in pkts:
            self._deliver(p)
        if not pkts and tier4 is not None:
            rx.log(f"css mf[{self.n}]: burst {self.audio_n / rx.fs:.1f}s: no chunks decoded (tier 5)")

    def _tier4(self, pkts: list[dict]) -> list[dict]:
        """Batch re-scan of the whole burst, merged into the live packets by chunk number."""
        rx = self.rx
        t = time.perf_counter()
        x = np.concatenate(self.audio) if self.audio else np.zeros(0, np.float32)
        pad = np.zeros(rx.fs, np.float32)
        x = np.concatenate([pad, x, pad])
        dec = rx.decoder_mod()
        found: list[dict] = []
        infos = [rx.info] + ([rx.info_short] if rx.info_short else [])
        for info, shift in itertools.product(infos, (0.0, 0.5)):
            got: list[dict] = []
            y = np.concatenate([np.zeros(int(shift * rx.sps_eff), np.float32), x]) if shift else x
            cfg = self._cfg(info, _ArrayStream(y), False)
            cfg["on_frame"] = lambda f, got=got: got.append(dict(f, source=f"tier4_{f.get('source', 'mf')}"))
            try:
                with cpustats.stage("decode"):
                    dec.decode_longsym_mf(cfg, stop_event=rx.stop_event)
            except Exception as e:
                rx.log(f"css mf[{self.n}]: tier4 error: {e}")
            found.append(got)
            if not pkts:
                if got:
                    pkts.extend(_packets(got))
                    self.tier4_added += len(got)
                    break
                continue
            self.tier4_added += self._merge(pkts, _packets(got))
            if not any(_gaps(p) for p in pkts) and self.tier4_added >= len(self.errors):
                break
        rx.log(f"css mf[{self.n}]: tier4 burst re-scan {len(x) / rx.fs:.0f}s in {time.perf_counter() - t:.1f}s: "
               f"{sum(len(g) for g in found)} chunk(s) read, live had {len(self.chunks)}, "
               f"{len(self.errors)} failed live")
        return found

    @staticmethod
    def _merge(pkts: list[dict], batch: list[dict]) -> int:
        """Fill missing chunk numbers of each live packet from the batch packet that agrees with it most."""
        added = 0
        for p in pkts:
            best, score = None, -1
            for b in batch:
                same = sum(1 for s, d in p["chunks"].items() if b["chunks"].get(s) == d)
                diff = sum(1 for s, d in p["chunks"].items() if s in b["chunks"] and b["chunks"][s] != d)
                if diff == 0 and same > score:
                    best, score = b, same
            if best is None:
                continue
            for s, d in best["chunks"].items():
                if s not in p["chunks"]:
                    p["chunks"][s] = d
                    p["recs"].append(next(r for r in best["recs"] if int(r["seq"]) == s))
                    added += 1
        return added

    def _deliver(self, p: dict) -> None:
        rx = self.rx
        gaps = _gaps(p)
        now = time.monotonic()
        if gaps:
            rx.log(f"css mf[{self.n}]: tier5 packet lost: chunks {gaps} missing of 0..{max(p['chunks'])}")
            rx.emit_link("err", -1, now, now, f"chunk loss {gaps}")
            return
        raw = b"".join(p["chunks"][s] for s in sorted(p["chunks"])).rstrip(b"\x00")
        try:
            pk = rx.unpack(raw)
        except Exception:
            rx.log(f"css mf[{self.n}]: {len(p['chunks'])} chunk(s) did not form a packet ({raw[:6].hex()})")
            return
        recs = p["recs"]
        # a live CRC failure that tier 4 did not make up for may be a trailing chunk (no gap shows it):
        # the packet is delivered, marked errored (tier 5), never silently short
        unresolved = max(0, len(self.errors) - self.tier4_added)
        snr = float(np.mean([r.get("snr_db", 0.0) for r in recs]))
        grid = recs[-1].get("grid_sps") or self.grid_sps
        if pk["src"] != rx.node_id and grid:
            rx.peer_clock[pk["src"]] = (float(grid), time.monotonic())
        d = rx.Decoded(text=pk["payload"].rstrip(b"\x00").decode("utf-8", "replace"), src=pk["src"],
                       dst=pk["dst"], seq=pk["seq"], type_name=pk["type_name"], ok=unresolved == 0,
                       frames=len(recs), frames_errored=unresolved, snr_db=snr, rs_errors=0,
                       payload=pk["payload"], acq_s=None, sync_score=1.0, grid_sps=grid,
                       t_last=max(r.get("t", 0) for r in recs) or None)
        how = collections.Counter(str(r.get("source", "mf")) for r in recs)
        rx.log(f"css rx src={d.src} seq={d.seq} {d.type_name} ok={d.ok} snr={snr:.1f}dB "
               f"frames={d.frames}/{d.frames_errored}err grid {rx.ppm(grid):+.1f} ppm "
               f"via {dict(how)} rescans={self.rescans} {len(raw)}B")
        rx.emit_link("rx" if d.ok else "err", d.src, now, now, f"{d.type_name} seq={d.seq} snr={snr:.1f}dB")
        rx.emit(d)


class CssLongRx:
    """Same surface as ``CssStreamRx`` (feed / skip / tx_onset / busy / burst_end / trk), plus
    ``burst_start`` (the RX level gate opened) and ``clock`` (the live CSS clock)."""

    def __init__(self, cfg, info_path: str, info_short: str | None, *, node_id: int, hear_own: bool,
                 log: Callable[[str], None], emit: Callable, emit_link: Callable, decoded_cls,
                 unpack: Callable, decoder_mod: Callable, work: Path, dec: dict | None = None,
                 stop_event: threading.Event | None = None, track: bool = True) -> None:
        self.cfg, self.node_id, self.hear_own = cfg, node_id, hear_own
        self.log, self.emit, self.emit_link = log, emit, emit_link
        self.Decoded, self.unpack, self.decoder_mod = decoded_cls, unpack, decoder_mod
        self.info, self.info_short, self.work = info_path, info_short, Path(work)
        self.dec = dict(dec or {})
        self.stop_event = stop_event or threading.Event()
        conf = json.loads(Path(info_path).read_text())
        self.fs = int(cfg.fs)
        self.sps = int(self.fs * float(conf["symbol_dur"]))
        self.sps_eff = self.sps + int(round(self.sps * float(conf.get("symbol_gap_frac", 0.0))))
        fsym = int(conf.get("frame_symbols", conf.get("base_bits", 48)))
        self.frame_s = ((int(conf.get("training_symbols", 0)) + len(conf.get("preamble") or [0] * 13) + fsym)
                        * self.sps_eff / self.fs)
        self.lock = threading.Lock()
        self.trk = None
        if track:
            import css_pll_c
            self.trk = css_pll_c.TrkStream(self.fs, float(conf["symbol_dur"]), float(conf["f_min"]),
                                           float(conf["f_max"]), float(conf.get("symbol_gap_frac", 0.0)),
                                           on_chirp=self._on_chirp, on_event=self._on_trk_event)
        self.busy_changes = 0
        self.counts = {"OWN": 0, "PEER": 0, "ECHO": 0, "UNKNOWN": 0}
        self.pre_n = int((4 * self.sps_eff) + self.fs)
        self._pre: collections.deque = collections.deque()
        self._pre_len = 0
        self.session: _Session | None = None
        self.sessions = 0
        self.peer_clock: dict[int, tuple[float, float]] = {}   # src -> (samples/symbol, monotonic)
        self._last_clock: tuple[float, float] | None = None
        # the live CSS clock: what the decoder's grid (tier 1) says right now
        self.clock = {"state": "idle", "grid_sps": None, "ppm": None, "sym": None, "t": None,
                      "trk_ppm": None}
        self.fed = 0
        self.skipped = 0
        self.log(f"css rx: live long-symbol MF decoder (tiers 1-5) + chirp tracker, sps={self.sps} "
                 f"sps_eff={self.sps_eff} frame {self.frame_s:.1f}s")

    def ppm(self, grid_sps) -> float:
        return float("nan") if not grid_sps else (float(grid_sps) / self.sps_eff - 1) * 1e6

    # -- feeding (css-rx thread) ---------------------------------------------------------------------
    def feed(self, samples: np.ndarray, tx_active: bool) -> None:
        x = np.asarray(samples, np.float32)
        self.fed += len(x)
        live = self.hear_own or not tx_active
        y = x if live else np.zeros_like(x)
        with self.lock, cpustats.stage("pll"):
            if self.trk is not None:
                self.trk.push(x)
        s = self.session
        if s is not None:
            s.push(y)
        else:
            self._pre.append(y); self._pre_len += len(y)
            while self._pre_len - len(self._pre[0]) >= self.pre_n:
                self._pre_len -= len(self._pre.popleft())

    @property
    def can_skip(self) -> bool:
        return False

    def skip(self, n: int) -> None:
        if n > 0 and self.trk is not None:
            with self.lock:
                self.trk.skip(n)
            self.skipped += n

    def tx_onset(self, rx_index: int, pcm: np.ndarray) -> None:
        from .css_stream import chirp_dirs
        first, dirs = chirp_dirs(pcm, self.fs, self.cfg.f_min, self.cfg.f_max, self.cfg.symbol_dur,
                                 self.sps, self.sps_eff)
        if dirs and self.trk is not None:
            with self.lock:
                self.trk.tx(rx_index + first, dirs)

    def busy(self) -> bool:
        return (self.trk is not None and self.trk.busy) or self.session is not None

    def _seed(self) -> float | None:
        now = time.monotonic()
        best = [v for v in self.peer_clock.values() if now - v[1] < _SEED_MAX_AGE_S]
        if self._last_clock and now - self._last_clock[1] < _SEED_MAX_AGE_S:
            best.append(self._last_clock)
        if best:
            return max(best, key=lambda v: v[1])[0]
        if self.clock.get("trk_ppm") is not None:                # nothing decoded yet: the tracker's estimate
            return self.sps_eff * (1 + self.clock["trk_ppm"] * 1e-6)
        return None

    def burst_start(self) -> None:
        """The RX gate opened: start a live decoder session on the pre-roll plus everything after it."""
        if self.session is not None:
            return
        pre = np.concatenate(list(self._pre)) if self._pre else np.zeros(0, np.float32)
        self._pre.clear(); self._pre_len = 0
        self.sessions += 1
        seed = self._seed()
        self.clock["state"] = "acquiring"
        s = _Session(self, pre, seed, self.sessions)
        self.session = s
        s.start()

    def burst_end(self) -> None:
        """The channel went quiet: the session drains, then runs tiers 4-5 and delivers on its own thread."""
        s, self.session = self.session, None
        if s is not None:
            s.end()

    def stop(self) -> None:
        self.stop_event.set()
        self.burst_end()

    def _session_done(self, s: _Session) -> None:
        if s.grid_sps:
            self._last_clock = (float(s.grid_sps), time.monotonic())
        if self.session is None and self.clock["state"] != "idle":
            self.clock["state"] = "coast" if s.grid_sps else "idle"

    def _clock_update(self, s: dict, sess: _Session) -> None:
        c = self.clock
        c.update(grid_sps=s["grid_sps"], ppm=self.ppm(s["grid_sps"]), sym=s["sym"], t=time.monotonic(),
                 te=s["te"], snr_db=s["snr_db"])
        if c["state"] in ("acquiring", "coast", "idle") and s["strong"]:
            c["state"] = "tracking"

    # -- tracker callbacks (under self.lock) -------------------------------------------------------------
    def _on_chirp(self, ch: dict) -> None:
        self.counts[ch["label"]] = self.counts.get(ch["label"], 0) + 1

    def _on_trk_event(self, e: dict) -> None:
        k = e["event"]
        now = time.monotonic()
        if k == "LOCK":
            self.clock["trk_ppm"] = float(e["x"])
            self.log(f"css pll: grid {e['a']} locked, {e['x']:+.0f} ppm vs our clock")
            self.emit_link("sync", -1, now, now, f"grid {e['a']} locked {e['x']:+.0f} ppm")
        elif k == "LOST":
            self.log(f"css pll: grid {e['a']} lost after {e['b']} chirps")
        elif k == "PHASE_STEP":
            self.log(f"css pll: grid {e['a']} PHASE STEP {e['x']:+.0f} smp ({e['y'] * 1e3:+.2f} ms)")
            self.emit_link("phase", -1, now, now, f"grid {e['a']} step {e['x']:+.0f} smp")
        elif k == "PEER":
            self.clock["trk_ppm"] = float(e["x"])
            self.log(f"css pll: grid {e['a']} is node {e['b']} ({e['x']:+.0f} ppm)")
        elif k == "BUSY":
            self.busy_changes += 1
