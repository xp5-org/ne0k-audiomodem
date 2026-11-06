"""Presence round and handshake turn order for an OFDM-only mesh.

CSS beacons are the mesh's normal discovery; under ``phy_mode="ofdm_only"`` they are never sent, so without
this module every node handshakes blind with its preset neighbour, all at once, and probes (which carry no
node id) collide on the one shared channel.  This runs the same idea over OFDM, by node id and event-driven
(nobody needs a shared clock, and nodes may start or join at any time):

1. **Round.**  Each node listens, then announces (a short OFDM message: its id and a mask of the ids it has
   heard) when it is its turn: after the highest lower id it heard, plus one empty slot for every id it did
   not hear in between.  Carrier sense first: it only keys when the channel has been quiet.
2. **Catch-up.**  A node that missed its turn, or that another node's mask shows did not hear it, announces
   once more in an id-ranked slot before BAT training.  It does this every cycle until it is confirmed heard,
   backing off 1, 2, 4 .. cycles after each unconfirmed try, and it never delays anyone: one announce, a
   clear channel, in its own slot only.
3. **BAT turns.**  In id order, one initiator at a time.  A turn ends when its handshake's GO (or a
   turn-pass announce) is heard, or on timeout; the node whose link is already up just passes the turn.
   Responders listen only during their initiator's turn, so a bystander never answers someone else's probe.

   ``bat_mode="broadcast"`` (SIP: a clean shared line) replaces them with one round for everybody:

   a. **Probes**, in id order, one each.  Everyone else measures it: the table for sender->me.  Probes carry no
      node id, so a probe belongs to the slot it was heard in (a burst that was not a probe, or not decodable,
      still uses up its slot; nobody guesses).
   b. **Reports**, in id order, everyone but the coordinator (the lowest id): one BAT_TABLES message with the
      tables it measured.
   c. **Confirm**: the coordinator sends one BAT_TABLES message holding every table it has (its own measurements
      and everyone's reports) and whether that is every direction between every known pair.  It is the single
      source of truth: links come up from the confirm's tables only, so both ends of a pair use the same ones.
   A confirm marked complete ends the BAT phase.  An incomplete one sends everybody to the pairwise turns above
   (a link that is already up just passes); a node that missed the confirm does its own turn alone.  In this mode
   an initiator announces FLAG_HANDSHAKE before a pairwise handshake, so its peer replaces the link it holds
   from the broadcast round with a fresh responder, and everybody holds its data until the GO.

Hooks (set by MeshNode): ``on_turn() -> bool`` runs this node's handshake and returns True when the link is
up; ``link_needed() -> bool``; ``on_peer_turn(initiator_id)`` lets the node (re)arm the responder for it;
``on_tables(tables) -> bool`` brings links up from the confirm's tables (True: every link it needs is up);
``on_peer_handshake(initiator_id)``: a FLAG_HANDSHAKE was heard.
"""
from __future__ import annotations

import struct
import threading
import time
from typing import Callable

import numpy as np

from .ident import KEYUP_TONE_HZ
from .ofdm_link import _Capture, _imports, _quiet, keyup, KEYUP_LEAD_S

FLAG_CATCHUP = 1
FLAG_PASS = 2            # "my turn is done, nothing to train" -- advances everyone's turn like a GO does
FLAG_HANDSHAKE = 4       # bat_mode broadcast: "a pairwise handshake of mine follows" (see the module docstring)
ANN_BODY = "<BHB"        # id, heard mask, flags

TAB_REPORT, TAB_CONFIRM = 0, 1
TAB_COMPLETE = 1         # confirm flag: every direction between every known pair is in it
BAT_MODES = ("pairwise", "broadcast")


def pack_tables(kind: int, frm: int, tables: dict, complete: bool = False) -> bytes:
    """BAT_TABLES body: [kind u8][from u8][flags u8][n u8] + n x ([src:4|dst:4][len u8][RLE table]).
    ``tables`` maps (src, dst) -> the wire table for src's transmissions as dst measured them."""
    H, _ = _imports()
    out = bytearray([kind, frm, TAB_COMPLETE if complete else 0, len(tables)])
    for (src, dst), bat in sorted(tables.items()):
        blob = H.bat_rle_pack(bat)
        out += bytes([(src << 4) | dst, len(blob)]) + blob
    return bytes(out)


def unpack_tables(body: bytes):
    """(kind, from, complete, {(src, dst): wire table}) or None when malformed."""
    H, _ = _imports()
    if len(body) < 4:
        return None
    kind, frm, flags, n = body[0], body[1], body[2], body[3]
    k, tables = 4, {}
    for _ in range(n):
        if k + 2 > len(body):
            return None
        addr, ln = body[k], body[k + 1]
        blob = body[k + 2:k + 2 + ln]
        if len(blob) != ln:
            return None
        k += 2 + ln
        tables[(addr >> 4, addr & 0xF)] = H.bat_rle_unpack(blob)
    return kind, frm, bool(flags & TAB_COMPLETE), tables


def _before_keyup(x: np.ndarray, fs: int) -> np.ndarray:
    """``x`` up to where an OFDM keyup lead tone starts (all of it when there is none)."""
    hop = int(0.05 * fs)
    win = np.hanning(hop)
    for k in range(0, len(x) - hop + 1, hop):
        s = np.abs(np.fft.rfft(x[k:k + hop] * win)) ** 2
        pk = int(s.argmax())
        if abs(pk * fs / hop - KEYUP_TONE_HZ) <= 50 and s[max(0, pk - 2):pk + 3].sum() > 0.6 * s.sum():
            return x[:max(0, k - hop)]            # the frame before holds the tone's first ms already
    return x


class AnnounceRound:
    def __init__(self, session, cfg, node_id: int, log: Callable[[str], None], *, listen_s: float = 1.5,
                 guard_s: float = 1.0, bat_turn_s: float = 60.0, cycle_s: float = 180.0,
                 backoff_cap: int = 8, data_turn_s: float = 24.0, shared_rx: bool = False,
                 bat_mode: str = "pairwise") -> None:
        if bat_mode not in BAT_MODES:
            raise ValueError(f"bat_mode must be one of {', '.join(BAT_MODES)}")
        if bat_mode == "broadcast" and not shared_rx:
            raise ValueError("bat_mode broadcast needs the node's shared receiver (probes reach it via rx_probe)")
        self.session, self.cfg, self.me, self.log = session, cfg, node_id, log
        self.listen_s, self.guard_s, self.bat_turn_s = listen_s, guard_s, bat_turn_s
        self.cycle_s, self.backoff_cap = cycle_s, backoff_cap
        self.data_turn_s = data_turn_s
        self.shared_rx = shared_rx                 # True: the node's MeshReceiver feeds rx_* below (no _rx thread)
        self._data_who: int | None = None
        self._data_next = 0                        # rotation index: data turns carry on across cycles, never restart at
                                                   # the lowest id (the last id waited 96 s and lost its link to the reaper)
        self.data_since: float | None = None       # monotonic start of the first data turns (link idle clock)
        self._data_end = 0.0
        self.on_turn: Callable[[], bool] | None = None
        self.link_needed: Callable[[], bool] = lambda: True
        self.on_peer_turn: Callable[[int], None] | None = None
        self.bat_mode = bat_mode
        self.on_tables: Callable[[dict], bool] | None = None
        self.on_peer_handshake: Callable[[int], None] | None = None
        self.tables: dict[tuple[int, int], bytes] = {}   # broadcast round: (src, dst) -> wire table, as known here
        self._confirm = None                       # (complete, tables) from the coordinator's confirm
        self._seq: dict | None = None              # broadcast round: whose item is next (see _seq_start)
        self._unattr = None                        # (start, end) of a probe-slot burst nothing decoded yet
        self._hs_hold_until = 0.0                  # a FLAG_HANDSHAKE was heard: no data until its GO
        self.phase = "listen"                      # listen | round | catchup | bcast | bat | data | idle
        self.known: set[int] = set()
        self.heard_end: dict[int, float] = {}      # id -> monotonic time its last announce ended
        self.masks: dict[int, int] = {}            # id -> mask it announced
        self._own_ann_start = float("-inf")        # monotonic start of our latest announce
        self.missed_turn = False
        self.unconfirmed_cycles = 0
        self._skip_left = 0
        self._announced = False
        self.idx = 0
        self.go_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cap = _Capture(cfg.fs, keep_s=40.0)
        self._tap = None
        self._threads: list[threading.Thread] = []
        self._tx_until = 0.0
        self._last_busy = 0.0                      # monotonic time energy (any burst, CW id included) was last on the air
        self._last_go = -100.0
        self._pending = False                      # a burst is on the air / not yet decoded
        self._band = (cfg.freq_lo, cfg.freq_hi)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        pipe = self.session.pipe
        if not pipe:
            raise RuntimeError("session not open")
        self._stop.clear()
        self._tap = pipe.subscribe(2048)
        for fn, name in ((self._collect, "ann-cap"),) + (() if self.shared_rx else ((self._rx, "ann-rx"),)) \
                + ((self._run, "ann-run"),):
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

    # -- state the node consults ------------------------------------------

    def order(self) -> list[int]:
        with self._lock:
            return sorted(self.known | {self.me})

    def initiator_may_start(self) -> bool:
        with self._lock:
            order = sorted(self.known | {self.me})
            if self.phase in ("idle", "data"):
                return True
            return self.phase == "bat" and self.idx < len(order) and order[self.idx] == self.me

    def responder_may_listen(self, initiator_id: int) -> bool:
        with self._lock:
            order = sorted(self.known | {self.me})
            if self.phase in ("idle", "catchup", "data"):
                return True
            return self.phase == "bat" and self.idx < len(order) and order[self.idx] == initiator_id

    def wait_data_window(self, timeout: float, need_s: float = 12.0) -> bool:
        """Payload traffic shares the one half-duplex channel with the round and the BAT turns, and after them takes
        turns by id: True when it is this node's data turn with at least ``need_s`` of it left.  False on timeout/stop."""
        end = time.monotonic() + timeout
        while not self._stop.is_set():
            with self._lock:
                if (self.phase == "data" and self._data_who == self.me and self._data_end - time.monotonic() >= need_s
                        and time.monotonic() >= self._hs_hold_until):
                    return True
            if time.monotonic() >= end:
                return False
            self._stop.wait(0.5)
        return False

    def _data_turns(self, until: float) -> None:
        """Id-ordered payload windows of ``data_turn_s`` each, repeated until ``until`` (start of the next cycle)."""
        order = self.order()
        if self.data_since is None:
            self.data_since = time.monotonic()
        self.log(f"announce: data turns order={order} window={self.data_turn_s:.0f}s")
        while not self._stop.is_set() and time.monotonic() < until:
            for _ in range(len(order)):
                who = order[self._data_next % len(order)]
                self._data_next = (self._data_next + 1) % len(order)
                with self._lock:
                    self.phase = "data"
                    self._data_who = who
                    self._data_end = time.monotonic() + self.data_turn_s
                if self._stop.wait(self.data_turn_s):
                    return
                if time.monotonic() >= until:
                    break
        with self._lock:
            self._data_who = None

    def max_turn_gap_s(self) -> float:
        """Longest a node can wait between two of its own data turns (one full rotation, plus a cycle's catch-up
        and BAT turns that may cut in): a link's idle limit must outlast it."""
        return 2.0 * len(self.order()) * self.data_turn_s + self.bat_turn_s

    def note_own_go(self) -> None:
        self._count_go()

    def _count_go(self) -> bool:
        """One BAT turn ends with one GO, but this node can see it three ways (the GO on the air, its own link
        coming up, and sending it); count it once.  Returns True when it was new."""
        with self._lock:
            now = time.monotonic()
            if now - self._last_go < 5.0:
                return False
            self._last_go = now
            self.go_count += 1
            self._hs_hold_until = 0.0
            return True

    # -- capture / receive -------------------------------------------------

    def _collect(self) -> None:
        while not self._stop.is_set():
            b = self._tap.get(timeout=0.2)
            if b is None:
                continue
            x = b.samples
            if b.tx_active:
                x = np.zeros_like(x)                 # never decode our own transmission
            self._cap.add(x)

    def _thr(self) -> float:
        return 10 ** (self.cfg.carrier_db / 20)

    def _clear(self, quiet_s: float = 0.8) -> bool:
        x = self._cap.get(self._cap.n - int(quiet_s * self.cfg.fs))
        return time.monotonic() >= self._tx_until and (len(x) == 0 or float(np.sqrt(np.mean(x * x))) <= self._thr())

    # -- receive: driven by _rx below, or by the node's MeshReceiver (shared_rx) -----------------

    def rx_air(self, busy: bool) -> None:
        """Energy on the air right now (any burst, CW id included)."""
        if busy:
            self._last_busy = time.monotonic()

    def rx_open(self) -> None:
        """A burst started: nothing may announce until it has been decoded (it may be a lower id's announce)."""
        with self._lock:
            self._pending = True

    def rx_done(self, got, end_t: float) -> None:
        """The burst ended and was decoded: ``got`` is try_decode_msg's result (or None: not a message)."""
        try:
            if got:
                self._on_msg(got, end_t)
            with self._lock:
                unattr, self._unattr = self._unattr, None
            if unattr is not None:               # a burst in a probe slot that held neither a probe nor a message
                who = self._seq_attribute("probe", *unattr)
                if who is not None:
                    self.log(f"announce: undecodable burst in the probe slot of {who}")
        finally:
            with self._lock:                       # under the lock: "not pending" then means heard_end is current
                self._pending = False

    def rx_msg(self, got, end_t: float) -> None:
        """One message out of a burst that may hold several (the burst stays pending until rx_done)."""
        with self._lock:
            self._unattr = None
        self._on_msg(got, end_t)

    def rx_probe(self, cap: np.ndarray, start_t: float, end_t: float) -> bool:
        """A whole closed burst, while the broadcast round is in its probes: True when it was a probe (measured and
        consumed).  False otherwise -- the receiver then decodes it as usual, and if nothing comes out of it either,
        rx_done still counts it as that slot's (undecodable) probe."""
        with self._lock:
            if self._seq is None or self._seq["kind"] != "probe":
                return False
        H, _ = _imports()
        fs = self.cfg.fs
        full = int(H.hs_cfg(fs, self._band)["n_probe_syms"])
        with _quiet():
            got = H.try_decode_probe(cap, fs, rates=(fs,), min_syms=max(12, int(0.6 * full)), band=self._band)
        if not got or got[2].get("n_active", 1) == 0:
            with self._lock:
                self._unattr = (start_t, end_t)
            return False
        _, bat_bytes, info = got
        who = self._seq_attribute("probe", start_t, end_t)
        if who is None:
            self.log("announce: probe heard outside any probe slot -- ignored")
            return True
        with self._lock:
            self.tables[(who, self.me)] = bat_bytes
        self.log(f"announce: probe of {who}: {info['bat_summary']} snr median {info['snr_median_db']} dB "
                 f"max {info['snr_max_db']} dB")
        self._emit_probe(who, info)
        return True

    def _emit_probe(self, who: int, info: dict) -> None:
        try:
            from .events import ProbeReport
            spec = info["spectrum"]
            freq = np.asarray(spec["freq_hz"], float)
            snr = spec.get("snr_db")
            snr = np.asarray(snr if snr is not None else [float("nan")] * len(freq), float)
            self.session._emit(ProbeReport("measured", who, self.me, freq, snr, np.asarray(spec["bits"], int),
                                           info["bat_summary"]))
        except Exception as e:
            self.log(f"announce: probe report failed: {e}")

    def _on_msg(self, got, end_t: float) -> None:
        H, _ = _imports()
        _, mtype, _, body = got
        if mtype == H.MSG_GO:
            new = self._count_go()
            self.log(f"announce: heard GO (turn {self.idx}, go_count={self.go_count}{'' if new else ', dup'})")
        elif mtype == H.MSG_ANNOUNCE and len(body) >= struct.calcsize(ANN_BODY):
            nid, mask, flags = struct.unpack(ANN_BODY, body[:struct.calcsize(ANN_BODY)])
            if nid == self.me:
                return
            with self._lock:
                self.known.add(nid)
                self.known.update(i for i in range(16) if mask & (1 << i) and i != self.me)
                self.heard_end[nid] = end_t
                self.masks[nid] = mask
                if self.phase in ("listen", "round") and nid > self.me and not self._announced:
                    self.missed_turn = True
            if flags & FLAG_PASS:
                self._count_go()
            self.log(f"announce: rx id={nid} mask={mask:#06x} flags={flags} known={sorted(self.known)}")
            if flags & FLAG_HANDSHAKE:
                with self._lock:
                    self._hs_hold_until = time.monotonic() + self.bat_turn_s
                if self.on_peer_handshake:
                    try:
                        self.on_peer_handshake(nid)
                    except Exception as e:
                        self.log(f"announce: on_peer_handshake failed: {e}")
        elif mtype == getattr(H, "MSG_BAT_TABLES", -1):
            tab = unpack_tables(body)
            if tab is None:
                self.log("announce: malformed BAT_TABLES -- ignored")
                return
            kind, frm, complete, tables = tab
            if frm == self.me:
                return
            if kind == TAB_REPORT:
                with self._lock:
                    for (src, dst), bat in tables.items():
                        if dst == frm:                       # a report only speaks for its sender's measurements
                            self.tables[(src, dst)] = bat
                self._seq_heard("report", frm, end_t)
                self.log(f"announce: rx BAT report from {frm}: {sorted(tables)}")
            elif kind == TAB_CONFIRM:
                with self._lock:
                    self._confirm = (complete, dict(tables))
                self.log(f"announce: rx BAT confirm from {frm} complete={complete}: {sorted(tables)}")

    def expected_sender(self) -> int | None:
        """Whose data turn it is (the station most likely on the air), or None outside the data turns."""
        with self._lock:
            return self._data_who if self.phase == "data" else None

    def _rx(self) -> None:
        """Stand-alone receiver (no MeshReceiver): decode a burst once it has ended."""
        H, _ = _imports()
        fs, win = self.cfg.fs, int(0.1 * self.cfg.fs)
        burst_from = None
        while not self._stop.wait(0.25):
            n = self._cap.n
            x = self._cap.get(n - win)
            busy = len(x) > 0 and float(np.sqrt(np.mean(x * x))) > self._thr()
            self.rx_air(busy)
            if busy and burst_from is None:
                burst_from = n - int(1.0 * fs)
                self.rx_open()
            if burst_from is None or busy:
                continue
            tail = self._cap.get(n - int(0.8 * fs))
            if len(tail) and float(np.sqrt(np.mean(tail * tail))) > self._thr():
                continue
            cap = self._cap.get(max(burst_from - int(0.5 * fs), n - int(40 * fs)))
            burst_from = None
            if len(cap) < int(1.0 * fs):
                self.rx_done(None, 0.0)
                continue
            end_t = time.monotonic() - 0.8
            with _quiet():
                got = H.try_decode_msg(cap, fs, rates=(fs,), band=self._band)
            self._check_ident(cap)
            self.rx_done(got, end_t)

    def _check_ident(self, cap: np.ndarray) -> None:
        """Another station's CW/ASCII ID at the start of this burst.  The start-of-session ID goes out in the
        round, before any link exists to hear it, and its announce follows it closer than the 0.8 s quiet that
        ends a burst here: look at the audio before the frame's keyup tone, whether or not a message decoded."""
        ident = getattr(self.session, "ident", None)
        if ident is None:
            return
        try:
            lead = _before_keyup(cap, self.cfg.fs)
            if len(lead) >= self.cfg.fs // 4:
                ident.note_heard(lead, self.cfg.fs)
        except Exception as e:
            self.log(f"announce: ident check failed: {e}")

    # -- transmit ----------------------------------------------------------

    def _airtime(self) -> float:
        H, _ = _imports()
        a = H.make_msg_audio(self.cfg.fs, H.MSG_ANNOUNCE, 0, struct.pack(ANN_BODY, self.me, 0, 0), self._band)
        return len(a) / self.cfg.fs

    def _mask(self) -> int:
        with self._lock:
            m = 0
            for i in self.heard_end:
                m |= 1 << i
            return m | (1 << self.me)

    def _announce(self, flags: int = 0) -> None:
        H, _ = _imports()
        body = struct.pack(ANN_BODY, self.me, self._mask(), flags)
        audio = H.make_msg_audio(self.cfg.fs, H.MSG_ANNOUNCE, 0, body, self._band)
        audio, _ = H.scale_hs_tx(audio, self.cfg.amplitude, self.cfg.fs, is_probe=False, band=self._band)
        audio = keyup(self.session, audio, self.cfg.fs)
        air = len(audio) / self.cfg.fs
        self._own_ann_start = time.monotonic()
        self._tx_until = time.monotonic() + air + 0.5
        self.log(f"announce: tx id={self.me} mask={self._mask():#06x} flags={flags} ({air:.1f}s)")
        self.session.play(audio, ident=False)       # a CW ID after the frame would overrun the slot and deafen us
        pipe = self.session.pipe
        if pipe:
            pipe.wait_tx(air + 10)
        time.sleep(0.3)

    def _wait_clear_then(self, target: float) -> bool:
        """Sleep to ``target`` (monotonic), then until the channel has been quiet; False if stopped."""
        while not self._stop.is_set():
            if time.monotonic() >= target and self._clear():
                return True
            self._stop.wait(0.2)
        return False

    # -- the schedule ------------------------------------------------------

    def _run(self) -> None:
        try:
            slot = self._airtime() + self.guard_s
            self.log(f"announce: id={self.me} slot={slot:.1f}s (frame {slot - self.guard_s:.1f}s)")
            self._round(slot)
            cycle_t0 = time.monotonic()
            self._catchup(slot, first=True)
            if self.bat_mode == "broadcast":
                res = self._bat_broadcast()
                if res == "incomplete":
                    self._bat()                      # everyone who heard the confirm: pairwise turns for the gaps
                elif res == "missed":
                    self._bat_solo()                 # the others are past the BAT phase: our own turn, alone
            else:
                self._bat()
            while not self._stop.is_set():
                self._data_turns(cycle_t0 + self.cycle_s)
                with self._lock:
                    self.phase = "idle"
                if self._stop.is_set():
                    return
                cycle_t0 = time.monotonic()
                if not self._flagged() and not self.link_needed():
                    self.log("announce: all links up, nobody flagged -- no new cycle")
                    continue
                self._catchup(slot, first=False)
                self._bat()
        except Exception as e:                       # never die silently
            self.log(f"announce: stopped on {type(e).__name__}: {e}")

    def _round(self, slot: float) -> None:
        t_start = time.monotonic()
        with self._lock:
            self.phase = "listen"
        self._stop.wait(self.listen_s)
        with self._lock:
            self.phase = "round"
        while not self._stop.is_set():
            # the slot target and the pending flag are read together: a target computed before a lower id's
            # announce was recorded must never meet a "not pending" read after it (C keyed in B's slot so)
            clear = self._clear()
            with self._lock:
                lower = [j for j in self.heard_end if j < self.me]
                if lower:
                    j = max(lower)
                    target = self.heard_end[j] + self.guard_s + (self.me - j - 1) * slot
                else:
                    target = t_start + self.listen_s + max(0, self.me - 1) * slot
                go = clear and not self._pending and time.monotonic() >= target
            if go:
                break
            self._stop.wait(0.2)
        if self._stop.is_set():
            return
        self._announced = True
        ident = getattr(self.session, "ident", None)
        if ident is not None and ident.due():            # ham: one CW ID at start, in our own slot, before the beacon
            try:
                air = ident.send_now("start")
                self._tx_until = time.monotonic() + air + 0.5
                self.session.pipe.wait_tx(air + 10)
                time.sleep(0.3)
            except Exception as e:
                self.log(f"announce: ident failed: {e}")
        self._announce()
        last = time.monotonic()
        # round ends once the air has been quiet of announces for one slot (absent ids go to the catch-up)
        while not self._stop.is_set():
            with self._lock:
                recent = max([last, self._last_busy, *self.heard_end.values()])
            if time.monotonic() - recent > slot:
                break
            self._stop.wait(0.25)
        self.log(f"announce: round done known={self.order()} missed_turn={self.missed_turn}")

    def _flagged(self) -> bool:
        with self._lock:
            # only an announce that ended after ours began can carry our bit: the first node of the round announces
            # knowing nobody, and that old mask must not flag everyone after it for the rest of the session
            lacking = any(not (m & (1 << self.me)) for nid, m in self.masks.items()
                          if self.heard_end.get(nid, 0.0) > self._own_ann_start)
            return self.missed_turn or lacking

    def _catchup(self, slot: float, first: bool) -> None:
        with self._lock:
            self.phase = "catchup"
        order = self.order()
        rank = order.index(self.me)
        t0 = time.monotonic()
        with self._lock:
            lacking = {i for i in order for m in self.masks.values() if not m & (1 << i)}
            lacking |= {self.me} if self.missed_turn else set()
        window = slot * len(order)               # fixed: every node must leave catch-up together (rank slots, one each)
        flagged = self._flagged()
        skip = flagged and not first and self._skip_left > 0
        if skip:
            self._skip_left -= 1
        elif flagged:
            if self._wait_clear_then(t0 + rank * slot):
                self._announce(FLAG_CATCHUP)
            self.unconfirmed_cycles += 1
            self._skip_left = min(self.backoff_cap, 2 ** self.unconfirmed_cycles) - 1   # 1, 3, 7 .. cycles off
            self.missed_turn = False
        else:
            self.unconfirmed_cycles = 0
            self._skip_left = 0
        remain = t0 + window - time.monotonic()
        if remain > 0:
            self._stop.wait(remain)
        self.log(f"announce: catch-up done flagged={flagged} skipped={skip} known={self.order()}")

    def _bat(self) -> None:
        order = self.order()
        with self._lock:
            self.phase = "bat"
            self.idx = 0
            base = self.go_count
        self.log(f"announce: BAT turns order={order}")
        while not self._stop.is_set():
            with self._lock:
                idx = self.idx
            if idx >= len(order):
                break
            who = order[idx]
            if self.on_peer_turn and who != self.me:
                try:
                    self.on_peer_turn(who)
                except Exception as e:
                    self.log(f"announce: on_peer_turn failed: {e}")
            t_turn = time.monotonic()
            if who == self.me:
                ok = False
                if self.link_needed() and self.on_turn:
                    self._announce_handshake()
                    ok = bool(self.on_turn())
                    self.log(f"announce: my BAT turn -> {'link up' if ok else 'failed'}")
                elif not self.link_needed():
                    if self._wait_clear_then(time.monotonic()):
                        self._announce(FLAG_PASS)
                with self._lock:
                    self.idx += 1
                    base = self.go_count
                continue
            t_rearm = 0.0
            while not self._stop.is_set():
                if self.on_peer_turn and who != self.me and time.monotonic() - t_rearm > 3.0:
                    t_rearm = time.monotonic()              # keep a live responder waiting for this initiator
                    try:
                        self.on_peer_turn(who)
                    except Exception:
                        pass
                with self._lock:
                    if self.go_count > base:
                        base = self.go_count
                        break
                if time.monotonic() - t_turn > self.bat_turn_s:
                    self.log(f"announce: turn of {who} timed out")
                    break
                self._stop.wait(0.25)
            with self._lock:
                self.idx += 1
        self.log("announce: BAT turns done")

    # -- broadcast BAT round (bat_mode="broadcast") --------------------------

    def _announce_handshake(self) -> None:
        """bat_mode broadcast: tell the peer (which may hold a link from the broadcast round) to re-arm."""
        if self.bat_mode == "broadcast" and self._wait_clear_then(time.monotonic()):
            self._announce(FLAG_HANDSHAKE)

    def _bat_solo(self) -> None:
        """Our own pairwise turn with nobody else in the BAT phase (we missed the confirm, or it did not cover us)."""
        with self._lock:
            self.phase = "bat"
            self.idx = self.order().index(self.me)
        if self.link_needed() and self.on_turn:
            self._announce_handshake()
            ok = bool(self.on_turn())
            self.log(f"announce: solo BAT turn -> {'link up' if ok else 'failed'}")

    def _seq_start(self, kind: str, ids: list[int], slot: float) -> None:
        """Items (probes, reports) go out in ``ids`` order, each after the previous one ended, an absent one costing
        one ``slot``.  ``pos`` is the index of the next expected item, ``last_end`` when the last one ended."""
        with self._lock:
            self._seq = {"kind": kind, "ids": list(ids), "pos": 0, "slot": slot,
                         "last_end": time.monotonic() - self.guard_s}

    def _seq_heard(self, kind: str, who: int, end_t: float) -> None:
        with self._lock:
            q = self._seq
            if q is None or q["kind"] != kind or who not in q["ids"]:
                return
            r = q["ids"].index(who)
            if r >= q["pos"]:
                q["pos"], q["last_end"] = r + 1, end_t

    def _seq_attribute(self, kind: str, start_t: float, end_t: float) -> int | None:
        """Whose item a burst heard from ``start_t`` to ``end_t`` was: the next expected sender (never us), or a later
        one when the burst started a whole slot or more after that sender was due."""
        with self._lock:
            q = self._seq
            if q is None or q["kind"] != kind:
                return None
            cand = [i for i in range(q["pos"], len(q["ids"])) if q["ids"][i] != self.me]
            if not cand:
                return None
            late = max(0, int((start_t - q["last_end"] - self.guard_s) // q["slot"]))
            i = cand[min(late, len(cand) - 1)]
            q["pos"], q["last_end"] = i + 1, end_t
            return q["ids"][i]

    def _seq_my_turn(self) -> bool:
        """Wait for our item's turn on a clear channel.  False when stopped, or when later items already went (our
        slot is gone: a probe out of its slot would be credited to someone else)."""
        while not self._stop.is_set():
            clear = self._clear()
            with self._lock:
                q = self._seq
                r = q["ids"].index(self.me)
                if q["pos"] > r:
                    return False
                target = q["last_end"] + self.guard_s + (r - q["pos"]) * q["slot"]
                if clear and not self._pending and time.monotonic() >= target:
                    return True
            self._stop.wait(0.2)
        return False

    def _seq_wait_done(self, slack_s: float = 2.0) -> None:
        """Until every item was heard (or its slot passed) and nothing is left to decode."""
        while not self._stop.is_set():
            with self._lock:
                q = self._seq
                left = len(q["ids"]) - q["pos"]
                due = q["last_end"] + self.guard_s + left * q["slot"] + slack_s
                if not self._pending and (left == 0 or time.monotonic() >= due):
                    return
            self._stop.wait(0.2)

    def _tx_hs(self, audio: np.ndarray, what: str, probe: bool = False) -> float:
        """Key one handshake burst (probe or message); returns the monotonic time it ended."""
        H, _ = _imports()
        audio, _ = H.scale_hs_tx(audio, self.cfg.amplitude, self.cfg.fs, is_probe=probe, band=self._band)
        audio = keyup(self.session, audio, self.cfg.fs)
        air = len(audio) / self.cfg.fs
        self._tx_until = time.monotonic() + air + 0.5
        self.log(f"announce: tx {what} ({air:.1f}s)")
        self.session.play(audio, ident=False)
        pipe = self.session.pipe
        if pipe:
            pipe.wait_tx(air + 10)
        end = time.monotonic()
        time.sleep(0.3)
        return end

    def _sample_tables(self, pairs) -> dict:
        """Stand-in tables (every carrier at 2 bits, a few off) for sizing slots before the real ones exist."""
        H, _ = _imports()
        n = int(H.hs_cfg(self.cfg.fs, self._band)["n_data_carriers"])
        bits = np.full(n, 2)
        bits[: max(1, n // 10)] = 0
        bat = H.bat_to_bytes(bits)
        return {p: bat for p in pairs}

    def _msg_audio(self, body: bytes) -> np.ndarray:
        H, _ = _imports()
        return H.make_msg_audio(self.cfg.fs, H.MSG_BAT_TABLES, 0, body, self._band)

    def _bat_broadcast(self) -> str:
        """The broadcast round (module docstring).  Returns "complete" (links up from the confirm, no pairwise turns),
        "incomplete" (the confirm left gaps: pairwise turns follow) or "missed" (no confirm heard here, or it did not
        give this node the links it needs)."""
        H, _ = _imports()
        order = self.order()
        with self._lock:
            self.phase = "bcast"                   # nobody's pairwise turn: no responder may act on these probes
            self.idx = 0
            self.tables = {}
            self._confirm = None
        if len(order) < 2:
            return "complete"
        fs, guard = self.cfg.fs, self.guard_s
        t0 = time.monotonic()
        probe = H.make_probe_audio(fs, self._band)
        slot_p = len(probe) / fs + KEYUP_LEAD_S + guard + 2.0              # + the decode before the next one keys
        self.log(f"announce: broadcast BAT order={order} probe slot={slot_p:.1f}s")
        self._seq_start("probe", order, slot_p)
        if self._seq_my_turn():
            self._seq_heard("probe", self.me, self._tx_hs(probe, "BAT probe", probe=True))
        else:
            self.log("announce: our probe slot passed -- no probe from us this round")
        self._seq_wait_done()
        coord, reporters = order[0], order[1:]
        sample = pack_tables(TAB_REPORT, self.me, self._sample_tables([(i, self.me) for i in order if i != self.me]))
        slot_r = len(self._msg_audio(sample)) / fs + KEYUP_LEAD_S + guard + 2.0
        self._seq_start("report", reporters, slot_r)
        if self.me != coord:
            with self._lock:
                mine = {k: v for k, v in self.tables.items() if k[1] == self.me}
            if self._seq_my_turn():
                self._seq_heard("report", self.me,
                                self._tx_hs(self._msg_audio(pack_tables(TAB_REPORT, self.me, mine)),
                                            f"BAT report {sorted(mine)}"))
        self._seq_wait_done()
        with self._lock:
            self._seq = None
        if self.me == coord:
            with self._lock:
                tables = dict(self.tables)
            complete = all((s, d) in tables for s in order for d in order if s != d)
            if not self._wait_clear_then(time.monotonic()):
                return "missed"
            body = pack_tables(TAB_CONFIRM, self.me, tables, complete)
            self._tx_hs(self._msg_audio(body), f"BAT confirm complete={complete} {sorted(tables)}")
            with self._lock:
                self._confirm = (complete, tables)
        else:
            est = len(self._msg_audio(pack_tables(TAB_CONFIRM, coord, self._sample_tables(
                [(a, b) for a in order for b in order if a != b]), True))) / fs
            deadline = time.monotonic() + est + KEYUP_LEAD_S + 3 * guard + slot_r + 8.0
            while not self._stop.is_set() and time.monotonic() < deadline:
                with self._lock:
                    if self._confirm is not None and not self._pending:
                        break
                self._stop.wait(0.2)
        with self._lock:
            conf = self._confirm
        took = time.monotonic() - t0
        if conf is None:
            self.log(f"announce: broadcast BAT: no confirm heard ({took:.0f}s)")
            return "missed"
        complete, tables = conf
        with self._lock:
            self.tables = dict(tables)
        ok = bool(self.on_tables(tables)) if self.on_tables else True
        self.log(f"announce: broadcast BAT done in {took:.0f}s complete={complete} my links {'up' if ok else 'missing'}")
        if complete:
            return "complete" if ok else "missed"
        return "incomplete"
