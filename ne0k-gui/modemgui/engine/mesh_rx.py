# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""One receiver per mesh node: one audio channel, one capture, one decode per burst.

The medium is a single shared half-duplex channel (a radio, a conference line).  Every station hears every
burst, so the node listens ONCE and routes what it hears, as the reference ``meshmodem_node`` does (one
capture, one RX poll, the packet's own src/dst deciding where it goes).  Before this module every OFDM link,
the announce round, the blockout listener and the CSS link each kept a capture and decoded every burst on its
own, serialised on ofdm_link's process-wide DSP lock: on 3 nodes that was ~3 s of decode per hop.

Per closed burst (0.8 s of quiet, measured on the samples, ends it), split again at any inner quiet of
``SPLIT_GAP_S`` so two messages that ran close together are decoded as two (a GO 1.0 s after a BAT_ACK was lost
when they were decoded as one burst), each part:

1. **Station ID** -- the audio before the frame's keyup tone goes to the session's Identifier (CW / ASCII).
2. **Data frame** -- when OFDM links are up: one decode
   with the rx BAT of the station expected to be transmitting (the data turn's owner, then a peer we await an
   ACK from, then the rest), a further BAT only if it actually differs and the first found no frame.  The frame
   header ``[src:4 | dst:4]`` routes it: to the link with that peer (DATA to the listening link, an ACK to the
   one waiting for it), or nowhere when it is addressed to another station.
3. **Message** -- otherwise one ``try_decode_msg``: ANNOUNCE / GO go to the announce round, BLOCKOUT_REQ / END
   to the node.  Outside the data turns the message is tried first and the data frame second: the phase only
   orders the two (nodes can disagree about it), it never skips one.

Pairwise handshakes (probe, BAT report, BAT ack, GO) are still decoded by the link running them: they happen one
pair at a time in the BAT turns, and the handshake state machine owns its own expectations.  The broadcast BAT
round's probes go to the announce round first (``rx_probe``); its BAT_TABLES messages arrive as messages.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from .ofdm_link import BROADCAST_ID, _Capture, _imports, _quiet, unpack_frame

BURST_END_S = 0.8          # quiet that closes a burst (as AnnounceRound)
SPLIT_GAP_S = 0.7          # inner quiet that separates two transmissions inside one closed burst: above a CW
                           # word gap (7 dots = 0.56 s at 15 wpm), below the ~1 s stations leave between frames
GATE_S = 0.05              # RMS gate block
PRE_S, POST_S = 0.2, 0.3   # data decode window around the gated burst
MSG_PRE_S = 1.5            # message decode looks further back (the gate opens late on a slow ramp)
MAX_MSGS_PER_PART = 3      # messages decoded back to back out of one part (see _msgs)


class MeshReceiver:
    def __init__(self, session, cfg, node_id: int, log: Callable[[str], None]) -> None:
        self.session, self.cfg, self.me, self.log = session, cfg, node_id, log
        self.links: Callable[[], list] = lambda: []                 # -> the node's OfdmLinks (any state)
        self.expected: Callable[[], list[int]] = lambda: []         # -> station ids most likely on the air, best first
        self.data_first: Callable[[], bool] = lambda: True          # payload frames most likely (data turns)?
        self.announce = None                                        # AnnounceRound (shared_rx) or None
        self.on_msg: Callable[[object], None] | None = None         # blockout etc.: f(try_decode_msg result)
        self.on_level: Callable[[float], None] | None = None        # f(rms) of each quiet 100 ms: ambient estimate
        self._last_busy = -1e9
        self._in_burst = False
        self.bursts = self.frames = self.foreign = self.undecodable = self.msgs = 0
        self.decode_s = 0.0
        self._cap = _Capture(cfg.fs, keep_s=60.0)
        self._tap = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        pipe = self.session.pipe
        if not pipe:
            raise RuntimeError("session not open")
        self._stop.clear()
        self._tap = pipe.subscribe(2048)
        for fn, name in ((self._collect, "mesh-rx-cap"), (self._run, "mesh-rx")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(5.0)
        self._threads.clear()
        pipe = self.session.pipe
        if self._tap and pipe:
            pipe.unsubscribe(self._tap)
        self._tap = None

    def _collect(self) -> None:
        while not self._stop.is_set():
            b = self._tap.get(timeout=0.2)
            if b is None:
                continue
            x = b.samples
            if b.tx_active and not self.cfg.hear_own:
                x = np.zeros_like(x)             # never decode our own transmission
            self._cap.add(x)

    def _thr(self) -> float:
        return 10 ** (self.cfg.carrier_db / 20)

    # -- burst gate ----------------------------------------------------------

    def _busy_blocks(self, i0: int, i1: int) -> np.ndarray:
        """Per GATE_S block of capture [i0, i1): above the carrier threshold?"""
        blk = int(GATE_S * self.cfg.fs)
        x = self._cap.get(i0, i1)
        k = len(x) // blk
        if not k:
            return np.zeros(0, bool)
        return np.sqrt(np.mean(x[:k * blk].reshape(k, blk) ** 2, axis=1)) > self._thr()

    def _run(self) -> None:
        """Burst gate.  It scans the capture from where it left off, not just the last poll's 100 ms: a decode takes
        1-3 s, and a burst that began during it (the GO answering the BAT_ACK being decoded) or began and ended in it
        (a short ACK) must keep its real start -- a start taken at "now" put the GO's frame before the decode window."""
        fs, win = self.cfg.fs, int(0.1 * self.cfg.fs)
        blk = int(GATE_S * fs)
        start = last = None                       # capture index the open burst began at / its last busy block ended
        scan = self._cap.n                        # everything before this was looked at
        while not self._stop.wait(0.25):
            try:
                n = self._cap.n
                x = self._cap.get(n - win)
                lvl = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
                busy = lvl > self._thr()
                if self.announce:
                    self.announce.rx_air(busy)
                if busy:
                    self._last_busy = time.monotonic()
                elif start is None and len(x) and self.on_level:
                    self.on_level(lvl)
                lo = max(scan, n - int(50 * fs))      # the capture keeps 60 s
                k = (n - lo) // blk
                if k:
                    idx = np.flatnonzero(self._busy_blocks(lo, lo + k * blk))
                    if len(idx):
                        if start is None:
                            start = lo + int(idx[0]) * blk
                            if self.announce:
                                self.announce.rx_open()
                        last = lo + (int(idx[-1]) + 1) * blk
                    scan = lo + k * blk
                self._in_burst = busy or start is not None
                if start is None or n - last < int(BURST_END_S * fs):   # quiet measured on the samples
                    continue
                a, b = start, last
                start = last = None
                self._closed(a, max(a + 1, b))
            except Exception as e:               # the node's only ear must never die silently
                self.log(f"mesh rx: {type(e).__name__}: {e}")
                start = last = None
                if self.announce:
                    self.announce.rx_done(None, 0.0)

    def channel_busy(self, settle_s: float = 0.4) -> bool:
        """Carrier sense: a burst is open, or energy was on the air within ``settle_s``."""
        return self._in_burst or time.monotonic() - self._last_busy < settle_s

    # -- one burst -----------------------------------------------------------

    def _parts(self, a: int, b: int) -> list[tuple[int, int]]:
        """[a, b) split at every inner quiet of SPLIT_GAP_S or more (the burst gate needs BURST_END_S)."""
        blk, gap = int(GATE_S * self.cfg.fs), int(round(SPLIT_GAP_S / GATE_S))
        busy = self._busy_blocks(a, b)
        idx = np.flatnonzero(busy)
        if not len(idx):
            return [(a, b)]
        out, s0, prev = [], idx[0], idx[0]
        for k in idx[1:]:
            if k - prev > gap:
                out.append((a + s0 * blk, a + (prev + 1) * blk))
                s0 = k
            prev = k
        out.append((a + s0 * blk, a + (prev + 1) * blk))
        return out

    def _closed(self, a: int, b: int) -> None:
        try:
            fs = self.cfg.fs
            if self._ident(self._cap.get(a - int(MSG_PRE_S * fs), b + int(POST_S * fs))):
                return                           # the whole burst was one station ID
            if self.announce and self.announce.rx_probe(self._cap.get(a - int(MSG_PRE_S * fs),
                                                                      b + int(0.7 * fs)),
                                                        self._cap.t_of(a), self._cap.t_of(b)):
                self.bursts += 1
                return                           # a probe of the broadcast BAT round (measured there)
            for pa, pb in self._parts(a, b):
                self._burst(pa, pb)
        finally:
            if self.announce:
                self.announce.rx_done(None, 0.0)

    def _burst(self, a: int, b: int) -> None:
        fs = self.cfg.fs
        self.bursts += 1
        end_t = self._cap.t_of(b)
        msg_cap = self._cap.get(a - int(MSG_PRE_S * fs), b + int(POST_S * fs))
        if b - a < fs // 4:                      # shorter than any frame: a click
            return
        if self._ident(msg_cap):                 # a bare ID part (no keyup, so no frame after it): nothing to decode
            return
        if self.data_first() and self._data(a, b):
            return
        if self._msgs(msg_cap, end_t):
            return
        if not self.data_first():
            self._data(a, b)

    def _msgs(self, cap: np.ndarray, end_t: float) -> int:
        """Decode the message(s) in one part.  Two stations can follow each other closer than SPLIT_GAP_S (a GO keyed
        0.5 s after the BAT_ACK it answers), and try_decode_msg returns the first message only: after each one, the
        audio past its end is decoded again.  Returns how many messages came out."""
        H, _ = _imports()
        fs, band = self.cfg.fs, (self.cfg.freq_lo, self.cfg.freq_hi)
        n = 0
        while n < MAX_MSGS_PER_PART and len(cap) >= fs:
            st: dict = {}
            t = time.monotonic()
            with _quiet():
                got = H.try_decode_msg(cap, fs, rates=(fs,), band=band, stats_out=st)
            self.decode_s += time.monotonic() - t
            if not got:
                break
            n += 1
            self.msgs += 1
            if self.announce:
                self.announce.rx_msg(got, end_t)
            if self.on_msg:
                self.on_msg(got)
            if st.get("frame_start") is None:
                break
            with _quiet():
                air = len(H.make_msg_audio(fs, got[1], got[2], got[3], band, n_groups=st.get("n_groups")))
            cap = cap[int(st["frame_start"]) + air:]
            if not self._busy_in(cap):
                break
        return n

    def _busy_in(self, x: np.ndarray) -> bool:
        """Any GATE_S block of ``x`` above the carrier threshold for 0.25 s or more in all?"""
        blk = int(GATE_S * self.cfg.fs)
        k = len(x) // blk
        if not k:
            return False
        busy = np.sqrt(np.mean(x[:k * blk].reshape(k, blk) ** 2, axis=1)) > self._thr()
        return int(busy.sum()) * GATE_S >= 0.25

    def _ident(self, cap: np.ndarray) -> bool:
        """Hand an ID at the start of the burst to the Identifier.  True when the burst was ONLY that ID."""
        from .ofdm_announce import _before_keyup
        ident = getattr(self.session, "ident", None)
        if ident is None:
            return False
        try:
            lead = _before_keyup(cap, self.cfg.fs)
            if len(lead) >= self.cfg.fs // 4 and ident.note_heard(lead, self.cfg.fs):
                return len(lead) >= len(cap) - int(0.1 * self.cfg.fs)
        except Exception as e:
            self.log(f"mesh rx: ident check failed: {e}")
        return False

    def _candidates(self) -> list:
        """Up links whose rx BAT may decode this burst, the expected sender's first, one per distinct table."""
        up = [lk for lk in self.links() if lk.up and getattr(lk, "shared_rx", False) and lk.rx_demod() is not None]
        if not up:
            return []
        rank = {p: i for i, p in enumerate(self.expected())}
        up.sort(key=lambda lk: (rank.get(lk.cfg.peer_id, len(rank)), 0 if lk.wants_ack() else 1))
        out, seen = [], []
        for lk in up:
            _, bat, band = lk.rx_demod()
            if any(tuple(band) == sb and np.array_equal(bat, sbat) for sbat, sb in seen):
                continue                         # same table, same band: the same decode
            seen.append((bat, tuple(band)))
            out.append(lk)
        return out

    def _data(self, a: int, b: int) -> bool:
        """True when the burst held a good data frame (ours or another station's): no message decode needed."""
        cands = self._candidates()
        if not cands:
            return False
        fs = self.cfg.fs
        x = self._cap.get(a - int(PRE_S * fs), b + int(POST_S * fs))
        t0, t1 = self._cap.t_of(a), self._cap.t_of(b)
        for lk in cands:
            dem, _, _ = lk.rx_demod()
            t = time.monotonic()
            with _quiet():
                stats, chunks = dem.decode_samples(x)
            dt = time.monotonic() - t
            self.decode_s += dt
            if not (chunks and sorted(chunks) == list(range(len(chunks)))):
                continue
            fr = unpack_frame(b"".join(chunks[k] for k in sorted(chunks)))
            if fr is None:
                continue                         # CRC: this table is not the sender's (or the frame is bad)
            src, dst, seq, ack, data = fr
            self.frames += 1
            what = f"seq={seq} ack={ack} {len(data)}B ({src}->{dst}, decode {dt:.1f}s)"
            if dst not in (self.me, BROADCAST_ID) or src == self.me:
                self.foreign += 1
                self.log(f"rx frame {what}: for another station")
                return True
            target = self._route(src, data)
            if target is None:
                self.log(f"rx frame {what}: no link to {src} -- dropped")
                return True
            self.log(f"rx frame ok {what} -> ofdm[{src}]")
            target.deliver(seq, ack, data, stats, len(chunks), t0, t1)
            return True
        self.undecodable += 1
        return False

    def _route(self, src: int, data: bytes):
        """DATA goes to the link that listens to ``src`` (arq: the end that is not the sender); a pure ACK to the
        link that has a frame in flight to it."""
        links = [lk for lk in self.links() if lk.up and lk.cfg.peer_id == src]
        if not links:
            return None
        if data:
            listening = [lk for lk in links if lk.role != lk.cfg.sender]
            return (listening or links)[0]
        waiting = [lk for lk in links if lk.wants_ack()]
        return (waiting or links)[0]

    def stats(self) -> str:
        return (f"bursts={self.bursts} frames={self.frames} foreign={self.foreign} msgs={self.msgs} "
                f"undecodable={self.undecodable} decode={self.decode_s:.1f}s")
