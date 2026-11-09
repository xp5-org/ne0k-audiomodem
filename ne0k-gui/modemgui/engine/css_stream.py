# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Continuous CSS receive on the C streaming decoder + the running local PLL bank.

``CssLink`` uses this when ``CssConfig.decoder == "pll_c"``.  Instead of cutting
each burst out with a level gate and decoding it cold, every RX sample goes to:

* ``css_pll_c.PllStream`` -- the C port of css_decoder.decode(), the streaming
  two-layer timing PLL decoder, fed continuously (blocks we transmitted are fed
  as silence unless hear_own, so its timebase never breaks);
* ``css_pll_c.TrkStream`` -- the running local PLL bank: one timing grid per
  station heard plus our own TX schedule, labelling every chirp OWN / PEER /
  ECHO / UNKNOWN and reporting phase steps and collisions.  It always gets the
  real samples, so a returning copy of our own audio is recognised.

Frames are reassembled into css_modem packets by their chunk sequence numbers
(chunk 0 starts a packet, as meshmodem_node does).  A packet is finished when
the next chunk 0 arrives or when the channel goes quiet; at that burst
boundary the decoder is re-armed (``rearm``: a soft resync, so it listens for
the next preamble straight away instead of burning resync_after frames of CRC
failures).  Each finished packet becomes one ``Decoded`` carrying the tracked
grid's period as ``grid_sps`` (the sender's clock against ours).
"""
from __future__ import annotations

import ctypes
import threading
import time
from typing import Callable

import numpy as np


class CssStreamRx:
    def __init__(self, cfg, info: dict, *, node_id: int, hear_own: bool,
                 log: Callable[[str], None], emit: Callable, emit_link: Callable,
                 decoded_cls, unpack: Callable, f64: bool = False,
                 dec: dict | None = None, track: bool = True, quiet: bool = False) -> None:
        """``dec``: decode() cfg overrides for the C decoder (tests switch single
        decoder features off with it); None = the reference defaults.
        ``track=False, quiet=True``: the second decoder CssLink runs on the short header-only
        frame geometry -- no chirp tracker of its own (carrier sense and grids stay with the
        main one), and it only speaks up for a packet: it syncs on every frame's preamble and
        fails the CRC of every full-size one, which is expected, not news."""
        import css_pll_c
        self.cfg, self.node_id, self.hear_own = cfg, node_id, hear_own
        self.log, self.emit, self.emit_link = log, emit, emit_link
        self.Decoded, self.unpack = decoded_cls, unpack
        self.lock = threading.Lock()
        dec = {"fs": cfg.fs, **(dec or {})}
        self.pll = css_pll_c.PllStream(dec, info, f64=f64, on_frame=self._on_frame,
                                       on_symbol=self._on_symbol, on_event=self._on_event)
        self.lib = self.pll.lib
        if not hasattr(self.lib, "_rearm_bound"):
            self.lib.css_pll_rearm.restype = None
            self.lib.css_pll_rearm.argtypes = [ctypes.c_void_p]
            self.lib._rearm_bound = True
        ex = self.pll.extras
        self.fs, self.sps, self.sps_eff = ex["fs"], ex["sps"], ex["sps_eff"]
        self.buf_size = 3 * self.sps
        self.quiet = quiet
        self.trk = None if not track else css_pll_c.TrkStream(ex["fs"], ex["dur"], ex["f_min"], ex["f_max"],
                                       ex["gap_samps"] / ex["sps"] if ex["sps"] else 0.0,
                                       on_chirp=self._on_chirp, on_event=self._on_trk_event)
        self._pkt: list | None = None          # [bytearray, [frame dicts], last_seq]
        self._pkt_t: float | None = None
        self._sync_score = 0.0
        self._sync_mono = None
        self.locked_peer = None
        self.counts = {"OWN": 0, "PEER": 0, "ECHO": 0, "UNKNOWN": 0}
        self._synced = False
        self.log(f"css rx: C streaming PLL decoder{' (short frames)' if quiet else ''} (workspace {self.pll.workspace_bytes / 1e6:.0f} MB) "
                 f"+ chirp tracker, sps={self.sps} sps_eff={self.sps_eff}")

    # -- feeding -------------------------------------------------------------
    def feed(self, samples: np.ndarray, tx_active: bool) -> None:
        x = np.asarray(samples, np.float32)
        with self.lock:
            if self.trk is not None:
                self.trk.push(x)
            # The decoder hears the line continuously (its preamble score is
            # energy/coverage-gated and its ring sees late chirps, so noise,
            # silence and signal onset no longer false-lock it).  Only our own
            # transmission is blanked, unless we are meant to hear it.
            live = self.hear_own or not tx_active
            self.pll.push(x if live else np.zeros_like(x))

    def tx_onset(self, rx_index: int, pcm: np.ndarray) -> None:
        """Our transmission ``pcm`` entered the sink at RX index rx_index: register its
        chirp schedule (directions read back off the waveform) with the tracker."""
        first, dirs = chirp_dirs(pcm, self.fs, self.cfg.f_min, self.cfg.f_max,
                                 self.cfg.symbol_dur, self.sps, self.sps_eff)
        if dirs and self.trk is not None:
            with self.lock:
                self.trk.tx(rx_index + first, dirs)

    def busy(self) -> bool:
        return self.trk is not None and self.trk.busy

    def burst_end(self) -> None:
        """The channel went quiet: finish the packet in progress and re-arm the decoder."""
        with self.lock:
            self._finish()
            self.lib.css_pll_rearm(self.pll._h)

    # -- decoder callbacks (called inside push, under self.lock) --------------
    def _on_symbol(self, s: dict) -> None:
        if s["sync_found"] and self.trk is not None:
            # where this symbol's chirp sits in the RX stream
            abs_peak = s["abs_start"] + s["nr"] - self.buf_size + s["found_peak"]
            self.trk.decoder_peak(abs_peak)

    def _on_event(self, e: dict) -> None:
        if self.quiet:
            if e["event"] == "SYNC_ACQUIRED":
                self._sync_score = e["x"]
                self._sync_mono = time.monotonic()
            return
        if e["event"] in ("SYNC_TOO_EARLY", "SYNC_STALE_REJECT", "FRAME_ERROR", "RETRO_GATE"):
            self.log(f"css rx: {e['event']} sym={e['sym']} a={e['a']} b={e['b']} x={e['x']:.3f}")
        if e["event"] == "RESYNC":
            self._synced = False
        if e["event"] == "SYNC_ACQUIRED":
            self._synced = True
            self._sync_score = e["x"]
            self._sync_mono = time.monotonic()
            now = time.monotonic()
            self.emit_link("lock", -1, now, now, f"preamble sync score {e['x']:.2f}")
            self.log(f"css rx: PLL sync acquired score={e['x']:.2f} lag={e['a']}")
        elif e["event"] == "RESYNC":
            self.log(f"css rx: decoder resync ({'hard' if e['a'] else 'soft'})")
        elif e["event"] in ("RETRO_GATE", "RETRO_SYNC") and e["event"] == "RETRO_GATE":
            self.log("css rx: retro timing gate tripped, re-acquiring")
        elif e["event"] == "PHASE_STEP":
            # a=step (smp), b=1 confirmed / 0 candidate, x=snr there, y=snr in the window
            self.log(f"css rx: PHASE_STEP {e['a']:+d} smp ({e['a'] / self.fs * 1e3:+.2f} ms) "
                     f"{'confirmed, read stepped' if e['b'] else 'candidate'} "
                     f"snr {e['y']:.1f} -> {e['x']:.1f} dB")

    def _on_frame(self, f: dict) -> None:
        seq = f["seq"]
        if seq == 0 or self._pkt is None or seq != self._pkt[2] + 1:
            if seq != 0:
                if self._pkt is not None:
                    if not self.quiet:
                        self.log(f"css rx: chunk gap (got {seq} after {self._pkt[2]}) -- partial dropped")
                self._pkt = None
                return
            self._finish()
            self._pkt = [bytearray(), [], -1]
        self._pkt[0] += f["data"]
        self._pkt[1].append(f)
        self._pkt_t = time.time()             # wall clock at this frame's decode (TIME packets use the last one)
        self._pkt[2] = seq

    def _finish(self) -> None:
        p, self._pkt = self._pkt, None
        if not p:
            return
        raw = bytes(p[0]).rstrip(b"\x00")
        frames = p[1]
        try:
            pk = self.unpack(raw)
        except Exception:
            if not self.quiet:
                self.log(f"css rx: {len(frames)} frame(s) did not form a packet ({raw[:6].hex()})")
            return
        grid_sps, peer_db = None, None
        if pk["src"] != self.node_id and self.trk is not None:
            self.trk.decoder_frame(pk["src"])
            for g in self.trk.grids():
                if g["label"] == "PEER" and g["peer"] == pk["src"]:
                    grid_sps, peer_db = g["period"], g["power_db"]
        snr = float(np.mean([f["snr_db"] for f in frames]))
        text = pk["payload"].rstrip(b"\x00").decode("utf-8", "replace")
        acq = (time.monotonic() - self._sync_mono) if self._sync_mono else None
        d = self.Decoded(text=text, src=pk["src"], dst=pk["dst"], seq=pk["seq"],
                         type_name=pk["type_name"], ok=True, frames=len(frames),
                         frames_errored=0, snr_db=snr,
                         rs_errors=sum(f["rs_errors"] for f in frames),
                         payload=pk["payload"], acq_s=None,
                         sync_score=self._sync_score, grid_sps=grid_sps, t_last=self._pkt_t)
        self.log(f"css rx src={d.src} seq={d.seq} {d.type_name} frames={d.frames} snr={snr:.1f}dB "
                 f"{'grid %.2f smp/sym (%+.0f ppm)' % (grid_sps, (grid_sps / self.sps_eff - 1) * 1e6) if grid_sps else 'grid ?'}")
        now = time.monotonic()
        self.emit_link("rx", d.src, now, now, f"{d.type_name} seq={d.seq} snr={snr:.1f}dB")
        self.emit(d)
        _ = acq, peer_db

    # -- tracker callbacks -----------------------------------------------------
    def _on_chirp(self, c: dict) -> None:
        self.counts[c["label"]] = self.counts.get(c["label"], 0) + 1
        # labels only: the decoder acquires on its own (no onset/align steering)

    def _on_trk_event(self, e: dict) -> None:
        k = e["event"]
        now = time.monotonic()
        if k == "LOCK":
            self.log(f"css pll: grid {e['a']} locked, {e['x']:+.0f} ppm vs our clock")
            self.emit_link("lock", -1, now, now, f"grid {e['a']} locked {e['x']:+.0f} ppm")
        elif k == "LOST":
            self.log(f"css pll: grid {e['a']} lost after {e['b']} chirps")
        elif k == "PHASE_STEP":
            self.log(f"css pll: grid {e['a']} PHASE STEP {e['x']:+.0f} smp ({e['y'] * 1e3:+.2f} ms)")
            self.emit_link("phase", -1, now, now, f"grid {e['a']} step {e['x']:+.0f} smp")
        elif k == "COLLISION":
            self.log(f"css pll: COLLISION grids {e['a']} and {e['b']} ({e['x']:+.1f} dB)")
            self.emit_link("collision", -1, now, now, f"grids {e['a']}+{e['b']}")
        elif k == "ECHO":
            self.log(f"css pll: grid {e['a']} is an ECHO of grid {e['b']}: "
                     f"{e['x'] * 1e3:.1f} ms, {e['y']:+.1f} dB")
            self.emit_link("echo", -1, now, now, f"echo {e['x'] * 1e3:.1f} ms {e['y']:+.1f} dB")
        elif k == "OWN_LAG":
            self.log(f"css pll: our own chirps return after {e['x'] * 1e3:.1f} ms")
        elif k == "PEER":
            self.log(f"css pll: grid {e['a']} is node {e['b']} ({e['x']:+.0f} ppm)")
        elif k == "NEW":
            self.log(f"css pll: new grid {e['a']}")


def chirp_dirs(pcm: np.ndarray, fs: int, f_min: float, f_max: float, dur: float,
               sps: int, sps_eff: int) -> tuple[int, list[int]]:
    """(offset of the first chirp's first sample, up/down per chirp) of a CSS
    transmission, read off the waveform with the matched filter on the
    sps_eff grid (independent of the encoder's lead silence / ID wrapping)."""
    x = np.asarray(pcm, np.float64)
    if len(x) < sps:
        return 0, []
    t = np.arange(sps) / fs
    def ch(f0, f1):
        k = (f1 - f0) / dur
        return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    up, dn = ch(f_min, f_max), ch(f_max, f_min)
    n = len(x) + sps - 1
    nf = 1 << (n - 1).bit_length()
    X = np.fft.rfft(x, nf)
    cu = np.abs(np.fft.irfft(X * np.fft.rfft(up[::-1], nf), nf)[sps - 1:len(x)])
    cd = np.abs(np.fft.irfft(X * np.fft.rfft(dn[::-1], nf), nf)[sps - 1:len(x)])
    env = np.maximum(cu, cd)
    if env.max() <= 0:
        return 0, []
    thr = 0.4 * env.max()
    first = int(np.argmax(env > thr))
    first = int(np.argmax(env[first:first + sps // 4])) + first
    dirs = []
    k = first
    while k < len(env):
        lo, hi = max(0, k - sps // 20), min(len(env), k + sps // 20)
        j = int(np.argmax(env[lo:hi])) + lo
        if env[j] < thr:
            break
        dirs.append(1 if cu[j] >= cd[j] else 0)
        k = j + sps_eff
    return first, dirs
