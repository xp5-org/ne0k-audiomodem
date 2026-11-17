# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Half-duplex OFDM link with a BAT exchange and fixed TX/RX windows.

Backend is the Python reference modem (ofdm_modem / ofdm_handshake from tests/OFDM);
``BACKEND`` says so for the UI.

Two phases:

1. **Handshake** -- ``HandshakeSM`` (the V.22-style state machine the mesh stack also
   uses) driven over the audio pipe: each end probes the other, grades the probe into a
   bit-allocation table (BAT) for the peer's transmit path, and the tables are swapped
   and proved with CRCs (BAT_REPORT / BAT_ACK / GO).  Afterwards each end holds the
   table for BOTH directions.

2. **Windows** -- time is cut into fixed windows of ``window_s``.  The epoch is the end of
   the GO message (the responder knows it exactly, the initiator when it decoded it); the
   initiator transmits in even windows, the responder in odd ones, and each side only
   keys up between ``guard_s`` after the window start and ``guard_s`` before its end, so
   the skew between the two epochs (well under a guard) can never put both on the air.
   Stop-and-wait ARQ rides in the frame header: every frame carries ``seq``, a piggy-backed
   ``ack`` (= next seq expected) and a CRC; a window with nothing to say stays silent.

    frame body = [addr u8][seq u8][ack u8][len u8][crc16 u16] + data      (len 0 = pure ACK)

   ``addr`` = sender id << 4 | destination id (4-bit ids, destination 0xF = broadcast, the same header byte as
   css_modem.pack_packet); the CRC covers it.
   On a shared channel (the mesh, a conference line) every link hears every station's frames, so a link acts
   only on a frame whose sender is its peer and whose destination is this node or broadcast.

``cfg.mode`` picks what happens after the BAT exchange (the handshake is the same for all):

* ``half``   -- the windows above (two-way in time, ARQ).
* ``oneway`` -- only ``cfg.sender`` transmits; each ``send`` is cut into frames of up to 250 B
  (same frame body, ``ack`` unused), separated by ``gap_s`` of silence so the listener sees
  one burst per frame.  No ACK, no retransmit: a seq gap is counted in ``lost_frames``.
* ``stream`` -- only ``cfg.sender`` transmits; one ``send`` is ONE burst of back-to-back OFDM
  chunks (each is its own frame with its own preamble, so any chunk decodes alone).  The
  listener decodes the growing burst every ``stream_poll_s`` and hands bytes to ``on_data``
  as soon as the chunks are contiguous; ``on_stream`` reports the end.

      stream body = [len u32] + data + [crc16 of data]

* ``bcast``  -- a continuous broadcast: only ``cfg.sender`` transmits, and once it has data it keeps the carrier
  up, one chunk after another with a running 16-bit chunk seq and no gaps, filling with empty chunks while
  ``send``/``feed`` has nothing queued (it unkeys after ``BCAST_IDLE_S`` with nothing to say).  There is no start
  and no end: a listener locks on whichever chunk it hears first, delivers every newer chunk in seq order, counts
  a seq gap in ``lost_frames`` and re-acquires after a stall.  Lossy by design, like UDP: what it carries must
  survive a gap (an H.264 / MPEG-TS stream recovers at the next IDR).

      bcast chunk = [len u16] + data (len <= chunk_bytes - 2) + zero pad        (len 0 = filler)

* ``full``   -- full duplex (FDD): after the full-band BAT exchange the band is split in two
  around ``duplex_guard_hz`` (``ofdm_fdd.plan_duplex_bands``), the initiator transmitting on
  the low half and the responder on the high one, and BOTH may be on the air at once.  Each
  half is its own carrier set (own preamble, CE and sync symbols), so its BAT is the full-band
  table re-indexed onto the sub-band's data bins.  The receiver low/high-passes away its own
  band first (``ofdm_fdd.rx_filter``).  The data plane is the same continuous carrier and sliding
  window as ``ec`` (see "full duplex, echo cancelled" below; logged with an ``fdd`` prefix): each
  end keeps its carrier up on its own half, one chunk after another, and ACKs ride the stream.
  (The C engine, backend ``cdx``, still runs fdd stop-and-wait per frame.)

* ``arq``    -- one-way data WITH acknowledgement and no windows: only ``cfg.sender`` sends data,
  the listener answers every frame with a pure-ACK frame on the same (full) band, the sender
  resends after ``ack_wait_s``.  Turns are enforced by the protocol (the sender waits for the
  ACK), not by a clock.  This is the data plane the mesh node uses for a src -> dst transfer.
  With ``sender="both"`` (the mesh node's links) it is connected-mode ARQ: once the handshake has
  exchanged both tables, either end sends DATA on the one link (sequence and ack numbers are per
  direction), listening before it talks and backing a resend off at random.

The mesh node (``mesh_node``) sits on top: it brings these links up on demand.
"""
from __future__ import annotations

import collections
import contextlib
import importlib.util
import io
import os
import random
import sys
import struct
import threading
import time
from pathlib import Path

import numpy as np

from .events import ProbeReport, LinkEvent
from . import carrier_schedule, cpustats
from .ident import KEYUP_TONE_HZ
from .modemcfg import OfdmConfig
from . import cca

BACKEND = "python"
HDR = 6                             # addr, seq, ack, len, crc16
# P2P TX window reservation (OfdmConfig.tx_window_chunks): control frames carried as ordinary frame data
RSV_MAGIC, STA_MAGIC = b"RSV1", b"STA1"
_RSV = struct.Struct("<HHIIH")      # window id, chunks, first schedule step, bytes, crc16 of the bytes
_STA = struct.Struct("<HH")         # window id, chunks listed: the ones to stream again
STA_MAX = 100                       # chunk numbers per status
BROADCAST_ID = 0xF                  # frame destination 0xF = every station (as css_modem.BROADCAST_ID)
MODES = ("half", "oneway", "stream", "full", "arq", "bcast")
MAX_CHUNKS = 65535                  # chunk seq is 2 bytes
STREAM_STALL_S = 3.0                          # quiet that ends a stream still short of its length
BCAST_HDR = 2                       # bcast chunk length prefix
BCAST_IDLE_S = 10.0                 # bcast sender: unkey after this long with nothing queued
BCAST_BACKLOG_S = 20.0              # bcast sender: older queued bytes beyond this much air time are dropped
BCAST_LOG_S = 10.0                  # bcast: one progress line per this long
EC_LOG_S = 10.0                     # duplex ec stream: one progress line per this long
_TESTS = Path(os.environ.get("MODEMGUI_TESTS_DIR",
                             Path(__file__).resolve().parents[3] / "audiotestserver" / "tests"))
_dsp_lock = threading.Lock()        # redirect_stdout is process-global: one DSP call at a time


def _imports():
    for p in (_TESTS, _TESTS / "OFDM"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if importlib.util.find_spec("reedsolo") is None:      # fall back to the C RS core
        os.environ.setdefault("RS_BACKEND", "c")
    import ofdm_handshake
    import ofdm_modem
    _stage_wrap(ofdm_handshake)
    return ofdm_handshake, ofdm_modem


# Charged to a CPU stage wherever they are called from, inside _quiet or not (the handshake messages, beacons and
# announce rounds build their audio outside it).  Nested inside a labelled _quiet they count once, for the outer label.
_STAGED = {"make_msg_audio": "encode", "make_probe_audio": "encode", "make_data_audio": "encode",
           "scale_hs_tx": "encode", "try_decode_msg": "decode", "try_decode_probe": "decode"}


def _stage_wrap(mod) -> None:
    if getattr(mod, "_cpu_staged", False):
        return
    for name, what in _STAGED.items():
        fn = getattr(mod, name, None)
        if fn is None:
            continue

        def staged(*a, _fn=fn, _what=what, **kw):
            with cpustats.stage(_what):
                return _fn(*a, **kw)
        staged.__wrapped__ = fn
        staged.__name__ = name
        setattr(mod, name, staged)
    mod._cpu_staged = True



KEYUP_LEAD_S = 2.0       # sacrificial lead-in on every key-up (channel=radio): the transmitter's PTT attack, the far
                         # squelch / PL decode, and an RTP/SIP leg (jitter buffer, talkspurt restart) all eat the first
                         # ~1.5 s of a burst; a frame that starts there loses its sync
KEYUP_IDLE_S = 0.0       # was 4.0: a station that keyed again 1.2-4 s after its own last burst (the short ACK right
                         # after a DATA) went out with no lead tone at all.  OWN_GAP_S below means every burst is a
                         # new key-up (the PTT dropped between them), so on a radio each one gets its lead.
OWN_GAP_S = 1.2          # least quiet between two of OUR bursts.  Our own TX is muted in our capture, so carrier
                         # sense alone let one node key back to back (its ACK, then its DATA 0.9 s later); the
                         # receivers' gate (mesh_rx: 0.8 s closes, 0.7 s splits) then saw ONE burst and lost the 2nd


_keyup_locks_lock = threading.Lock()


def _keyup_lock(session) -> threading.Lock:
    """One lock per station (session): two nodes in one process (tests) never wait on each other."""
    with _keyup_locks_lock:
        lk = getattr(session, "_keyup_lock", None)
        if lk is None:
            lk = session._keyup_lock = threading.Lock()
        return lk


def keyup(session, audio: np.ndarray, fs: int, lead_s: float | None = None, idle_s: float | None = None) -> np.ndarray:
    """Prefix ``audio`` with a steady tone when this session has not transmitted for ``idle_s``, like a radio's
    key-up preamble, so a re-timed burst start lands on throw-away audio.  Waits until our previous burst ended
    ``OWN_GAP_S`` ago (so receivers see two bursts), and marks the transmit end time.  The wait and the mark are
    one step: two threads of one station (an ACK, a DATA) queue their bursts a gap apart, never back to back.

    ``session.keyup_tone`` False (OfdmConfig.channel = line: SIP / dialup / fax-modem audio, no PTT, no PL) drops
    the lead tone; the OWN_GAP_S spacing and the transmit-end mark stay, the receivers' burst gate needs them."""
    lead = KEYUP_LEAD_S if lead_s is None else lead_s
    if not getattr(session, "keyup_tone", True):
        lead = 0.0
    idle = KEYUP_IDLE_S if idle_s is None else idle_s
    with _keyup_lock(session):
        while True:
            wait = getattr(session, "_keyup_end", 0.0) + OWN_GAP_S - time.monotonic()
            if wait <= 0:
                break
            time.sleep(min(wait, 0.2))
        now = time.monotonic()
        last = getattr(session, "_keyup_end", 0.0)
        if lead > 0 and now - last >= idle:
            amp = float(np.max(np.abs(audio))) if len(audio) else 0.3
            pad = amp * np.sin(2 * np.pi * KEYUP_TONE_HZ * np.arange(int(lead * fs)) / fs)
            audio = np.concatenate([pad, audio])
        session._keyup_end = now + len(audio) / fs
    return audio


BACKENDS = ("python", "c", "cdx")


def set_backend(name: str) -> str:
    """Select the DSP cores for this process: "c" = libofdmacq (acquisition + demod) and librs255,
    "python" = numpy/reedsolo.  The reference modules read the env at call time, so the choice is
    process-wide.  "c" fails loudly when a library is missing or built for another CPU instead of
    quietly running the numpy path.  "cdx" is "python" for the handshake plus the whole full-duplex data
    plane (codec, own-band filter, burst gate, ARQ) in the C engine ``ofdm_cdx``; it only changes
    mode ``full``; the handshake stays on the reference modem.  Returns the description to log."""
    _imports()
    if name not in BACKENDS:
        raise ValueError(f"ofdm backend {name!r} not in {BACKENDS}")
    if name == "cdx":
        from . import ofdm_cdx
        ofdm_cdx._lib()                                  # fail now, not mid-call, if libofdm is stale
        return "c engine (libofdm: full-duplex data plane in C; handshake: " + set_backend("python") + ")"
    if name == "python":
        os.environ.pop("OFDM_ACQ_BACKEND", None)
        if importlib.util.find_spec("reedsolo") is None:
            os.environ["RS_BACKEND"] = "c"
        else:
            os.environ.pop("RS_BACKEND", None)
        return "python (numpy" + (", reedsolo)" if importlib.util.find_spec("reedsolo") else ", librs255)")
    import cbackend
    libs = []
    for key in ("ofdm_acq", "rs255"):
        rec = cbackend.BACKENDS[key].probe()
        if not rec["ok"]:
            raise RuntimeError(f"ofdm backend c: {rec['reason']}")
        libs.append(Path(rec["path"]).name)
    os.environ["OFDM_ACQ_BACKEND"] = "c"
    os.environ["RS_BACKEND"] = "c"
    return "c (" + ", ".join(libs) + ")"


_PHY_KEYS = ("n_fft", "cp_ratio", "pilot_spacing", "probe_duration_s", "probe_meas_syms",
             "snr_margin_db", "max_bits_per_bin", "snr_estimator", "isi_derate")


def apply_phy(c: OfdmConfig) -> dict:
    """Push the PHY group into ``HS_OVERRIDES`` (read by ``hs_cfg`` at call time, so the probe,
    the messages and the data frames all follow it).  Process-wide like the backend; the peer
    must run the same values or the carrier geometry disagrees.  Returns what was set."""
    H, _ = _imports()
    over = {k: getattr(c, k) for k in _PHY_KEYS}
    over["isi_derate"] = float(over["isi_derate"])
    H.HS_OVERRIDES = dict(over)
    return over


@contextlib.contextmanager
def _quiet(what: str = "dsp"):
    """The reference modules print a progress line per call; keep them out of the CLI log.  ``what`` is the CPU stage
    the call is charged to ("decode" / "encode", cpustats.py), timed inside the lock so waiting for it is not counted."""
    with _dsp_lock, cpustats.stage(what), contextlib.redirect_stdout(io.StringIO()):
        yield


def pack_frame(seq: int, ack: int, data: bytes, src: int = 0, dst: int = 0xF) -> bytes:
    """Frame body for station ``src`` to ``dst`` (4-bit ids, dst 0xF = broadcast)."""
    _imports()
    from ofdm_handshake import crc16
    if len(data) > 255:
        raise ValueError("frame data over 255 bytes")
    if not (0 <= src <= 15 and 0 <= dst <= 15):
        raise ValueError(f"station ids are 4 bits: src={src} dst={dst}")
    head = bytes([(src << 4) | dst, seq & 0xFF, ack & 0xFF, len(data)])
    return head + crc16(head + data).to_bytes(2, "big") + data


def unpack_frame(raw: bytes):
    """(src, dst, seq, ack, data) or None when short or the CRC fails."""
    _imports()
    from ofdm_handshake import crc16
    if len(raw) < HDR:
        return None
    addr, seq, ack, n = raw[0], raw[1], raw[2], raw[3]
    if len(raw) < HDR + n:
        return None
    data = raw[HDR:HDR + n]
    if crc16(raw[:4] + data) != int.from_bytes(raw[4:6], "big"):
        return None
    return addr >> 4, addr & 0xF, seq, ack, data


class _Capture:
    """Rolling mono capture with absolute sample indices and a monotonic-time mapping."""

    def __init__(self, fs: int, keep_s: float = 120.0, sense=None) -> None:
        self.fs, self.keep = fs, int(keep_s * fs)
        self.sense = sense                  # cca.Presence: "busy" = one of our signals, not a level (or None)
        self._chunks: collections.deque = collections.deque()
        self._start = 0
        self.n = 0
        self._ref = (0, time.monotonic())
        self._lock = threading.Lock()

    def add(self, x: np.ndarray) -> None:
        with self._lock:
            self._chunks.append(x)
            self.n += len(x)
            self._ref = (self.n, time.monotonic())
            while self.n - self._start > self.keep and len(self._chunks) > 1:
                self._start += len(self._chunks.popleft())
        if self.sense is not None:
            self.sense.push(x)              # stage 1 runs on every sample, once, here

    def busy_blocks(self, i0: int, i1: int | None, thr: float, blk: int) -> tuple[int, np.ndarray]:
        """(first index, busy flag per ``blk`` samples) for capture [i0, i1): one of our signals is there (signal
        sense), or the block's RMS is above ``thr`` (level sense)."""
        i0 = max(i0, self._start)
        i1 = self.n if i1 is None else min(i1, self.n)
        if self.sense is not None:
            return i0, self.sense.flags(i0, i1, blk) > 0
        x = self.get(i0, i1)
        k = len(x) // blk
        if not k:
            return i0, np.zeros(0, bool)
        return i0, np.sqrt(np.mean(x[:k * blk].reshape(k, blk) ** 2, axis=1)) > thr

    def busy(self, look_s: float, thr: float) -> bool:
        """The last ``look_s``: one of our signals in it (signal sense), or its RMS above ``thr`` (level sense)."""
        if self.sense is not None:
            return self.sense.busy(look_s)
        x = self.get(self.n - int(look_s * self.fs))
        return len(x) > 0 and float(np.sqrt(np.mean(x * x))) > thr

    def get(self, i0: int, i1: int | None = None) -> np.ndarray:
        with self._lock:
            chunks = list(self._chunks)
            start, n = self._start, self.n
        if not chunks:
            return np.zeros(0, np.float32)
        x = np.concatenate(chunks)
        i0 = max(i0, start)
        i1 = n if i1 is None else min(i1, n)
        return x[i0 - start:max(i0, i1) - start]

    def t_of(self, idx: int) -> float:
        n, t = self._ref
        return t - (n - idx) / self.fs

    def idx_of(self, t: float) -> int:
        n, tr = self._ref
        return int(n - (tr - t) * self.fs)


class OfdmLink:
    def __init__(self, session, cfg: OfdmConfig | None = None, log=lambda m: None) -> None:
        self.session, self.cfg, self.log = session, cfg or OfdmConfig(), log
        c = self.cfg
        self.role = c.role if c.role != "auto" else ("initiator" if c.node_id < c.peer_id
                                                      else "responder")
        if self.role not in ("initiator", "responder"):
            raise ValueError("role must be initiator, responder or auto")
        if c.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if c.sender not in ("initiator", "responder", "both"):
            raise ValueError("sender must be initiator, responder or both")
        self.state = "idle"             # idle | handshake | up | failed | closed
        self.fail_reason = ""
        self.on_data = None             # f(bytes): a delivered payload
        self.on_state = None            # f(state)
        self.rx_gate = None             # f() -> bool: may a responder act on a probe heard NOW (its pair's window)
        self.on_retry = None            # f(k): window resent for the k-th time
        self._id_seen: set[int] = set()
        self.on_stream = None           # f(ok, nbytes): a stream burst ended (listener side)
        self.rx_filter = None           # test hook: f({"seq","ack","len"}) -> False drops the frame
        self.shared_rx = False          # True (mesh): once up, frames come from the node's one MeshReceiver via
                                        # deliver(); this link runs no data listener of its own (arq mode only)
        self._dem = None                # cached demodulator for _rx_bat
        self.sm = None
        self.preset = None              # (my_tx_bat, peer_tx_bat) wire tables from the mesh's broadcast BAT round:
                                        # the link comes up on them without a handshake of its own
        self.epoch = 0.0
        self.capacity = 0               # payload bytes per window, known once up
        self.retry_count = self.fail_count = 0
        self.lost_frames = self.rx_errors = 0          # oneway/stream listener
        self._txlock = threading.Lock()
        self._feed = bytearray()                        # bcast sender: bytes queued for the carrier
        self._feed_lock = threading.Lock()
        self._feed_max = 1 << 20                        # set from the chunk air time once up
        self.bcast_dropped = 0                          # bcast sender: backlog bytes dropped (fed faster than the air)
        self.bcast_chunks = 0                           # bcast: chunks aired (sender) / delivered (listener)
        self.bcast_bytes = 0                            # bcast: payload bytes aired (sender) / delivered (listener)
        self.phy_bps = 0.0                              # payload bit/s of the carrier while keyed (0 = not known yet)
        self._hs_rx_end = 0                             # capture index where the last HS message ended
        self._go_end = None                             # monotonic time the responder saw GO (set by the handshake; None when up from a preset)
        self.tx_frames = self.rx_frames = 0
        self.foreign_frames = 0                        # good frames between other stations (shared channel)
        self._cap = _Capture(c.fs, sense=cca.for_session(session))
        self._tap = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._pending = None            # (data, done Event, box) waiting for the next TX window
        self._inflight = None           # dict: the frame awaiting its ACK
        self._tx_seq = 0
        self._rx_next = 0               # next seq expected from the peer
        self._owe_ack = False
        self._resend_go = False
        self._tx_bat = self._rx_bat = None
        self._band = (c.freq_lo, c.freq_hi)
        self._tx_band = self._rx_band = self._band      # sub-bands once up in full mode
        self._filt = None                               # (far_cfg, my_cfg) in full mode
        self._cdx = None                                # C duplex engine (backend "cdx", mode full)
        self._ec = None                                 # echo canceller (mode full, duplex ec)
        # duplex ec sliding window: frames queued by send(), not yet numbered / numbered and unacked (oldest first) /
        # peer frames kept past a gap {seq: (data, t)}; round trip estimate for the resend timer
        self._ec_q: collections.deque = collections.deque()
        self._ec_win: dict = {}
        self._ec_rxbuf: dict = {}
        self._ec_srtt = self._ec_rttvar = None
        self.ec_acked_bytes = self.ec_rx_bytes = 0
        self._ec_duty_mine = 1.0
        self._cdx_base = None                           # capture index the engine's sample 0 sits at
        self._cdx_box = None                            # {"evt", "ok"} of the send in progress
        self._send_lock = threading.Lock()
        self._dup_cfgs = None                           # (tx sub-band cfg, rx sub-band cfg)
        self._ack_evt = threading.Event()
        # carrier-schedule OFDM (carrier_schedule.py): score for our DATA frames / the peer's, frames counted per direction, receive demodulators per step
        self._mel_tx = self._mel_rx = None
        self._mel_tx_n = self._mel_rx_n = 0
        self._mel_rx_prev = 0                           # score step the last frame we took began on
        self._mel_dems: dict = {}
        # TX window reservation: our next window id, the window we are sending, the
        # window we are receiving
        self._rsv_id = 0
        self._rsv_tx: dict | None = None
        self._rsv_evt = threading.Event()
        self._rsv_rx: dict | None = None
        self.plan = None
        self.backend = ""
        self.last_activity = time.monotonic()
        self._up = threading.Event()
        self._hs_t0 = time.monotonic()
        self._last_rx_t0 = -1e9
        self._peer_reported = False

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        pipe = self.session.pipe
        if not pipe:
            raise RuntimeError("session not open")
        if self.cfg.fs != self.session.spec.rate:
            raise ValueError(f"ofdm fs {self.cfg.fs} != session rate {self.session.spec.rate}")
        self.backend = set_backend(self.cfg.backend)
        self.log(f"dsp backend: {self.backend}")
        self.log("ofdm phy: " + ", ".join(f"{k}={v}" for k, v in apply_phy(self.cfg).items()))
        self.log("ofdm config: " + ", ".join(f"{k}={v}" for k, v in self.cfg.to_dict().items()))
        self._stop.clear()
        if self.cfg.mode == "full" and self.cfg.duplex == "ec":
            from .ofdm_ec import EchoCanceller
            self._ec = EchoCanceller(self.cfg.fs, taps=max(1, self.cfg.ec_taps))
            hooks = getattr(pipe, "tx_onset_hooks", None)
            if hooks is not None:
                hooks.append(self._ec.add_ref)       # our own audio on the RX timebase: the canceller's reference
        self._tap = pipe.subscribe(2048)
        self._set_state("handshake")
        for fn, name in ((self._collect, "ofdm-cap"), (self._run, "ofdm-link")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def passive_listen(self) -> bool:
        """A responder still waiting for the initiator's first probe: it only listens, nothing is on
        the air from this link, so it must not hold off our own initiator handshake."""
        sm = getattr(self, "sm", None)
        return self.role == "responder" and sm is not None and sm.state == "r_wait_probe"

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(5.0)
        self._threads.clear()
        self._cdx = None
        self._cdx_base = None
        pipe = self.session.pipe
        if self._ec is not None and pipe is not None:
            with contextlib.suppress(ValueError, AttributeError):
                pipe.tx_onset_hooks.remove(self._ec.add_ref)
        if self._tap and pipe:
            pipe.unsubscribe(self._tap)
        self._tap = None
        self._complete(False)
        self._set_state("closed")

    def wait_up(self, timeout: float) -> bool:
        return self._up.wait(timeout)

    def wait_settled(self, timeout: float) -> bool:
        """Like wait_up, but returns False as soon as the handshake has failed or the link closed, instead of
        sitting out the whole timeout."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._up.wait(0.25):
                return True
            if self.state in ("failed", "closed"):
                return False
        return False

    @property
    def up(self) -> bool:
        return self.state == "up"

    def _set_state(self, st: str) -> None:
        self.state = st
        if st == "up":
            self._up.set()
        else:
            self._up.clear()
        self.log(f"link {st}" + (f" ({self.fail_reason})" if st == "failed" else ""))
        if self.on_state:
            self.on_state(st)

    def _ev(self, kind: str, t0: float, t1: float | None = None, detail: str = "") -> None:
        self.session._emit(LinkEvent(kind, self.cfg.node_id, self.cfg.peer_id, t0,
                                     t0 if t1 is None else t1, detail))

    # -- capture ------------------------------------------------------------

    def _collect(self) -> None:
        while not self._stop.is_set():
            b = self._tap.get(timeout=0.2)
            if b is None:
                continue
            x = b.samples
            ec = self._ec
            if ec is not None:
                ec.add_raw(b.index, x)               # training data: the echo of our own probe
                if ec.on and self.up:
                    x = ec.cancel(b.index, x)
            # never decode our own transmission: mute it, except in full duplex once the far end can talk over it
            # (fdd: the own-band filter removes it; ec: the canceller, trained at the end of the handshake)
            full_rx = self.cfg.mode == "full" and (ec is None or self.up)
            if b.tx_active and not (self.cfg.hear_own or full_rx):
                x = np.zeros_like(x)
            self._cap.add(x)
            if self._cdx is not None:
                self._cdx_rx(x)

    def _thr(self) -> float:
        return 10 ** (self.cfg.carrier_db / 20)

    def _bursts(self, i0: int, i1: int | None = None, gap_s: float = 0.5):
        """Contiguous busy spans (idx0, idx1) in the capture, 50 ms RMS gate.  Callers poll this
        every 0.25s against a still-growing capture, so a span here may not be closed yet --
        _is_ident is deliberately NOT applied here: judging a truncated prefix of an ongoing
        burst risks a wrong, PERMANENT verdict once the caller caches it by span start (see
        _is_ident's _id_seen).  Callers must call _is_ident themselves, only once a span is
        actually closed."""
        fs = self.cfg.fs
        win = int(0.05 * fs)
        base = max(i0, self._cap._start)
        if self._cap.sense is not None and not self._filt:
            busy = np.flatnonzero(self._cap.busy_blocks(i0, i1, 0.0, win)[1])
        else:                                # level sense; fdd always: our own carrier is in the raw capture
            x = self._rx_get(i0, i1)
            n = len(x) // win
            if not n:
                return []
            rms = np.sqrt(np.mean(x[:n * win].reshape(n, win) ** 2, axis=1))
            busy = np.flatnonzero(rms > self._thr())
        out: list[list[int]] = []
        gap = int(gap_s / 0.05)
        for k in busy:
            if out and k - out[-1][1] <= gap:
                out[-1][1] = int(k)
            else:
                out.append([int(k), int(k)])
        spans = [(base + a * win, base + (b + 1) * win) for a, b in out]
        return spans

    def _is_ident(self, span: tuple[int, int]) -> bool:
        """Another station's ID burst (CW or binary ASCII) is not ours to decode."""
        idt = self.session.ident
        a, b = span
        dur_s = (b - a) / self.cfg.fs
        if not (0.25 <= dur_s <= 30.0) or a in self._id_seen:
            return a in self._id_seen
        # every real OFDM frame is keyup()-prefixed (same span, no gap); a genuine CW/ASCII id
        # never is (send_now plays it raw) -- see note_heard's tail_from.
        tail_from = int(KEYUP_LEAD_S * self.cfg.fs) if dur_s > KEYUP_LEAD_S else None
        hit = idt.note_heard(self._cap.get(a, b), self.cfg.fs, tail_from=tail_from)
        if hit:
            self._id_seen.add(a)
            self._ev("rx", self._cap.t_of(a), self._cap.t_of(b), f"ID heard: {idt.heard[-1][2]}")
        return hit

    def _rx_get(self, i0: int, i1: int | None = None, raw: bool = False) -> np.ndarray:
        """Capture slice as the demodulator should see it (full duplex: own band removed)."""
        x = self._cap.get(i0, i1)
        if self._filt and len(x) > 64 and not raw:
            try:
                from ofdm_fdd import rx_filter
                y, _ = rx_filter(np.asarray(x, np.float64), self.cfg.fs, *self._filt,
                                 ripple_db=0.05, stop_db=50.0)
                if y is not None:
                    return y
            except Exception as e:
                self.log(f"rx filter off: {e}")
        return x

    def _air_busy(self, look_s: float = 0.4) -> bool:
        if getattr(self.session, "ignore_busy", False):
            return False                          # GeneralConfig.ignore_busy: microphone use, noise always present
        return self._cap.busy(look_s, self._thr())

    def _note_rx(self, i0: int, i1: int, detail: str) -> None:
        t0, t1 = self._cap.t_of(i0), self._cap.t_of(i1)
        self._ev("lock", t0, t0, detail)
        if abs(t0 - self._last_rx_t0) > 1.0:
            self._ev("rx", t0, t1, detail)
            self._last_rx_t0 = t0

    # -- TX -----------------------------------------------------------------

    def _defer_for_air(self, max_s: float = 40.0) -> None:
        """Carrier sense before a payload frame: the channel is shared and half-duplex, so wait for other stations'
        bursts to end, then hold off a few hundred ms by node id so two waiting nodes do not key together."""
        end = time.monotonic() + max_s
        while time.monotonic() < end and not self._stop.is_set():
            if self._air_busy(0.6):
                self._stop.wait(0.2)
                continue
            self._stop.wait((self.cfg.node_id % 8) * 0.25)
            if not self._air_busy(0.6):
                return

    def _play(self, audio: np.ndarray, what: str, ident: bool = False) -> float:
        if what.startswith(("DATA", "STREAM")) and self.cfg.mode != "full":
            self._defer_for_air()                # full duplex: our sub-band is ours alone, the peer is always "on air"
        audio = keyup(self.session, audio, self.cfg.fs)
        air = len(audio) / self.cfg.fs
        now = time.monotonic()
        self._ev("tx", now, now + air, what)
        self.log(f"tx {what} ({air:.1f}s)")
        if os.environ.get("OFDM_FAIL_DUMP") and what.startswith("STREAM"):   # keep what we played, to diff against the capture
            os.makedirs(os.environ["OFDM_FAIL_DUMP"], exist_ok=True)
            np.save(os.path.join(os.environ["OFDM_FAIL_DUMP"], f"tx_{os.getpid()}_{self.tx_frames}.npy"),
                    np.asarray(audio, np.float32))
        self.session.play(audio, ident=ident)
        pipe = self.session.pipe
        if pipe:
            pipe.wait_tx(air + 10)
        time.sleep(0.3)                          # sink tail; RX is muted until here
        return air

    def _sleep_until(self, t: float) -> bool:
        while not self._stop.is_set():
            rem = t - time.monotonic()
            if rem <= 0:
                return True
            self._stop.wait(min(rem, 0.2))
        return False

    # -- handshake ----------------------------------------------------------

    def _run(self) -> None:
        try:
            if self._preset_up() if self.preset else self._handshake():
                m = self.cfg.mode
                if m == "half":
                    self._windows()
                elif m == "full" and self._cdx is not None:
                    self._run_cdx()
                elif m == "full":
                    self._run_ec()               # ec and fdd: the windowed continuous carrier
                elif m == "arq" and self.shared_rx:
                    self._stop.wait()            # frames arrive through deliver(); TX runs on the caller's thread
                elif m in ("full", "arq"):
                    self._listen_full()
                elif m == "bcast":
                    self._bcast_tx() if self.role == self.cfg.sender else self._listen_bcast()
                elif self.role != self.cfg.sender or self.cfg.sender == "both":
                    self._listen(stream=(m == "stream"))
        except Exception as e:                   # the link thread must never die silently
            self.fail_reason = f"{type(e).__name__}: {e}"
            self.session._emit_raw_error(f"ofdm link: {self.fail_reason}")
            if self.state != "closed":
                self._set_state("failed")

    def _send_hs(self, H, act, sm) -> None:
        c, band = self.cfg, self._band
        if act[0] == "send_probe":
            audio, what = H.make_probe_audio(c.fs, band), "probe"
        else:
            audio = H.make_msg_audio(c.fs, act[1], act[2], act[3], band, my_tx_bat=sm.my_tx_bat)
            what = H._MSG_NAMES.get(act[1], str(act[1]))
        audio, _ = H.scale_hs_tx(audio, c.amplitude, c.fs, is_probe=(act[0] == "send_probe"),
                                 band=band)
        t_turn = time.monotonic()
        deadline = t_turn + 15.0                # never talk over the peer's tail
        while self._air_busy() and time.monotonic() < deadline and not self._stop.is_set():
            time.sleep(0.25)
        self._play(audio, what)
        if act[0] == "send_msg" and act[1] == H.MSG_GO:
            self._go_end = time.monotonic()
        sm.hold_off(time.monotonic() - t_turn)

    def _handshake(self) -> bool:
        H, _ = _imports()
        c = self.cfg
        self._hs_t0 = time.monotonic()
        now = self._hs_now
        wait_msg = max(20.0, c.wait_msg_s) + (10.0 if self.role == "responder" else 0.0)
        sm = self.sm = H.HandshakeSM(self.role, max_retries=c.hs_retries,
                                     wait_probe_s=max(20.0, c.wait_probe_s),
                                     wait_msg_s=wait_msg, skip_fin=True)
        listen = max(0, self._cap.n - int(20.0 * c.fs))      # peer may have probed already
        self._go_end = None
        if self.role == "initiator":
            self._stop.wait(3.0)                 # let the responder get its capture running
        sm.start(now())
        last = -1e9
        while now() < c.hs_timeout_s and not self._stop.is_set():
            acts, sm.actions = sm.actions, []
            for a in acts:
                if a[0] in ("send_probe", "send_msg"):
                    self._send_hs(H, a, sm)
                    listen = self._cap.n
                elif a[0] == "fail":
                    self.fail_reason = a[1]
            if sm.done or sm.failed or (self.role == "responder" and sm.state == "r_linger"
                                        and self._go_end):
                break
            if now() - last >= c.poll_s:
                last = now()
                listen = self._attempt_rx(H, sm, listen)
            sm.tick(now())
            time.sleep(0.1)
        ok = sm.done or (self.role == "responder" and sm.state == "r_linger" and self._go_end)
        if not ok or not (sm.my_tx_bat and sm.peer_tx_bat):
            self.fail_reason = self.fail_reason or sm.state
            self.fail_count += 1
            now_m = time.monotonic()
            self._ev("fail", now_m, now_m, f"handshake {self.fail_reason}")
            self._set_state("failed")
            return False
        self.epoch = self._go_end if self.role == "responder" else time.monotonic()
        return self._finish_up(H, sm.my_tx_bat, sm.peer_tx_bat, "BAT exchanged")

    def _preset_up(self) -> bool:
        """Up on the tables the mesh's broadcast BAT round measured (``preset``), no handshake on the air.  Only the
        shared-receiver arq mode uses it: no windows, no epoch to agree on."""
        H, _ = _imports()
        my_tx, peer_tx = self.preset
        self.epoch = time.monotonic()
        return self._finish_up(H, my_tx, peer_tx, "BAT from the broadcast round")

    def _finish_up(self, H, my_tx_bat: bytes, peer_tx_bat: bytes, how: str) -> bool:
        c = self.cfg
        self._tx_bat = H.data_bat(my_tx_bat, c.fs, self._band, c.chunk_bytes)
        self._rx_bat = H.data_bat(peer_tx_bat, c.fs, self._band, c.chunk_bytes)
        m = c.mode
        self._setup_schedule(H)
        if m == "full":
            self._setup_duplex(H)
            if c.backend == "cdx" and c.duplex == "ec":
                self.log("c engine: its data plane is fdd only (own-band filter); duplex ec runs the python one")
            elif c.backend == "cdx":
                self._start_cdx()
        self.capacity = (self._capacity(H) if m == "half"
                         else c.chunk_bytes - HDR if m == "full" and self._cdx is None
                         else 255 - HDR if m in ("oneway", "full", "arq")
                         else c.chunk_bytes - BCAST_HDR if m == "bcast" else MAX_CHUNKS * c.chunk_bytes - 6)
        now_m = time.monotonic()
        self._ev("lock", now_m, now_m, how)
        self.log(f"bat exchanged{'' if how == 'BAT exchanged' else ' (broadcast round)'} tx={int((self._tx_bat > 0).sum())} carriers "
                 f"rx={int((self._rx_bat > 0).sum())} carriers capacity={self.capacity}B "
                 f"({c.mode})")
        self.last_activity = time.monotonic()      # a handshake can outlast link_idle_s: do not let the reaper see a fresh link as idle
        self._set_state("up")
        return True

    def _attempt_rx(self, H, sm, listen: int) -> int:
        """Decode whatever the SM expects (a port of meshlib.run_handshake's attempt_rx).
        Returns the new listen start."""
        c = self.cfg
        expect = sm.expecting()
        if not expect:
            return listen
        if sm.state == "r_wait_probe" and self.rx_gate is not None and not self.rx_gate():
            return self._cap.n          # probes carry no node id: one heard outside our pair's window is some other link's
        cap = self._cap.get(listen)
        if len(cap) > int(60 * c.fs):
            listen += len(cap) - int(60 * c.fs)
            cap = cap[-int(60 * c.fs):]
        if len(cap) <= int(4.0 * c.fs):
            return listen
        rates, band = (c.fs,), self._band
        if "msg" in expect:
            with _quiet("decode"):
                got = H.try_decode_msg(cap, c.fs, rates=rates, band=band)
            if got:
                _, mtype, proof, body = got
                self.log(f"hs rx {H._MSG_NAMES.get(mtype)}")
                self._mark_rx(listen, f"{H._MSG_NAMES.get(mtype)}")
                sm.heard_msg(mtype, proof, body, self._hs_now())
                self._emit_peer_report(H, sm)
                return self._cap.n
        if "probe" in expect:
            full = int(H.hs_cfg(c.fs, band)["n_probe_syms"])
            near_full = max(12, int(0.6 * full))
            min_syms = near_full if sm.state == "r_wait_ack" else 12
            with _quiet("decode"):
                got = H.try_decode_probe(cap, c.fs, rates=rates, min_syms=min_syms, band=band)
                if got and got[2]["n_syms"] < near_full and got[2].get("tail_s", 99.0) < 6.0:
                    got = None                   # still playing: wait for the full train
                if got and got[2].get("n_active", 1) == 0:      # a message's lead-in, not a probe
                    cut = got[2].get("start_mic", 0)
                    got2 = (H.try_decode_probe(cap[:cut], c.fs, rates=rates, min_syms=min_syms,
                                               band=band) if cut > int(4.0 * c.fs) else None)
                    got = got2 if got2 and got2[2].get("n_active", 1) > 0 else None
            if got:
                _, bat_bytes, info = got
                self.log(f"hs rx probe {info['bat_summary']} snr median {info['snr_median_db']} dB "
                         f"max {info['snr_max_db']} dB isi_ceiling {info.get('isi_ceiling_db')}")
                self._mark_rx(listen, "probe")
                self._emit_probe("measured", c.peer_id, c.node_id, info["spectrum"],
                                 info["bat_summary"])
                sm.heard_probe(bat_bytes, self._hs_now())
        return listen

    def _emit_probe(self, source: str, node: int, by: int, spec: dict, summary: str) -> None:
        nan = float("nan")
        freq = np.asarray(spec["freq_hz"], float)
        snr = np.asarray(spec.get("snr_db") if spec.get("snr_db") is not None else [nan] * len(freq),
                         float)
        self.session._emit(ProbeReport(source, node, by, freq, snr,
                                       np.asarray(spec["bits"], int), summary))

    def _emit_peer_report(self, H, sm) -> None:
        """The peer's BAT_REPORT tells us how OUR probe looked to it (bits only)."""
        if self._peer_reported or not sm.my_tx_bat:
            return
        self._peer_reported = True
        c = self.cfg
        cfg = H.hs_cfg(c.fs, self._band)
        bat = H.bytes_to_bat(sm.my_tx_bat, cfg["n_data_carriers"])
        spec = H.carrier_spectrum(cfg, np.full(int(cfg["n_occupied"]), np.nan), bat)
        summ = H.Modem(cfg).bat_summary(bat)
        self.log(f"hs rx bat report (our probe as heard by node {c.peer_id}): {summ}")
        self._emit_probe("peer_report", c.node_id, c.peer_id, spec, summ)

    def _hs_now(self) -> float:
        return time.monotonic() - self._hs_t0

    def _mark_rx(self, listen: int, what: str) -> None:
        spans = self._bursts(listen)
        if spans:
            self._hs_rx_end = spans[-1][1]
            self._note_rx(spans[-1][0], spans[-1][1], what)

    # -- data windows -------------------------------------------------------

    def _pack(self, seq: int, ack: int, data: bytes) -> bytes:
        c = self.cfg
        return pack_frame(seq, ack, data, c.node_id & 0xF, c.peer_id & 0xF)

    def _for_us(self, src: int, dst: int) -> bool:
        """Sender is our peer and the frame is addressed to us (or broadcast).  A link whose ids are unset or equal
        (an old point-to-point setup) accepts everything, as before addressing."""
        me, peer = self.cfg.node_id, self.cfg.peer_id
        if not (0 <= me < BROADCAST_ID and 0 <= peer < BROADCAST_ID) or me == peer:
            return True
        return src == peer and dst in (me, BROADCAST_ID)

    def _frame_from(self, chunks: dict):
        """(seq, ack, data) of a frame for this link; None = undecodable; False = a good frame between other
        stations (shared channel), which the caller ignores without counting an error."""
        if not (chunks and sorted(chunks) == list(range(len(chunks)))):
            return None
        fr = unpack_frame(b"".join(chunks[k] for k in sorted(chunks)))
        if fr is None:
            self.log("rx frame CRC fail")
            return None
        src, dst, seq, ack, data = fr
        if not self._for_us(src, dst):
            self.foreign_frames += 1
            self.log(f"rx frame {src}->{dst} seq={seq} ack={ack} {len(data)}B: not for this link "
                     f"({self.cfg.peer_id}->{self.cfg.node_id}), ignored")
            return False
        self.log(f"rx frame ok seq={seq} ack={ack} {len(data)}B ({src}->{dst})")
        return seq, ack, data

    def _frame_audio(self, H, body: bytes, bat=None) -> np.ndarray:
        c = self.cfg
        if isinstance(bat, list):              # a scheduled frame: one table per chunk, note carriers over the pilots
            audio = self._mel_tx.audio(H, c.fs, body, bat, self._tx_band, c.chunk_bytes, c.schedule_boost_db)
        else:
            audio = H.make_data_audio(c.fs, body, self._tx_bat if bat is None else bat, self._tx_band, c.chunk_bytes)
        return H.scale_hs_tx(audio, c.amplitude, c.fs, is_probe=False, band=self._tx_band)[0]

    def _capacity(self, H) -> int:
        c = self.cfg
        usable = c.window_s - 2 * c.guard_s
        n = 0
        with _quiet("encode"):
            while n < 64:
                air = len(H.make_data_audio(c.fs, b"\0" * ((n + 1) * c.chunk_bytes), self._tx_bat,
                                            self._band, c.chunk_bytes)) / c.fs
                if air > usable:
                    break
                n += 1
        if not n:
            raise RuntimeError(f"window_s={c.window_s} guard_s={c.guard_s} leaves no room for "
                               f"one {c.chunk_bytes}-byte frame")
        return min(255, n * c.chunk_bytes - HDR)

    def _windows(self) -> None:
        H, _ = _imports()
        c = self.cfg
        mine = 0 if self.role == "initiator" else 1
        k = 0
        while not self._stop.is_set():
            w0 = self.epoch + k * c.window_s
            if k % 2 == mine:
                if not self._sleep_until(w0 + c.guard_s):
                    return
                self._tx_window(H, k)
            else:
                if not self._sleep_until(w0 + c.window_s + c.guard_s / 2):
                    return
                self._rx_window(H, w0 - c.guard_s / 2, w0 + c.window_s + c.guard_s / 2)
            k += 1

    def _next_frame(self):
        """(seq, ack, body) for this window or None to stay silent."""
        c = self.cfg
        with self._lock:
            if self._inflight is None and self._pending is not None:
                data = self._pending[0]
                self._inflight = dict(seq=self._tx_seq, data=data, sends=0)
                self._tx_seq = (self._tx_seq + 1) & 0xFF
            f = self._inflight
            if f and f["sends"] > c.retries:
                self.fail_count += 1
                now = time.monotonic()
                self._ev("fail", now, now, f"no ACK after {c.retries} retries")
                self._inflight = None
                self._complete_locked(False)
                f = None
            if f:
                f["sends"] += 1
                if f["sends"] > 1:
                    self.retry_count += 1
                    now = time.monotonic()
                    self._ev("retry", now, now, f"attempt {f['sends'] - 1}")
                    if self.on_retry:
                        self.on_retry(f["sends"] - 1)
                self._owe_ack = False
                return f["seq"], self._rx_next, f["data"]
            if self._owe_ack:
                self._owe_ack = False
                return (self._tx_seq - 1) & 0xFF, self._rx_next, b""
        return None

    def _window_id(self) -> bool:
        """Half duplex has no slack for in-line audio, so a due station ID takes this window's
        slot instead of the frame (which goes out in our next window).  CW that is longer than
        the window falls back to binary ASCII; if neither fits, the end-of-transmission timer
        is all that is left and we say so."""
        idt = self.session.ident
        if not idt.due():
            return False
        c = self.cfg
        usable = c.window_s - 2 * c.guard_s
        pcm = idt.standalone()
        if len(pcm) / c.fs > usable:
            pcm = idt.ascii()
        if len(pcm) / c.fs > usable:
            if not getattr(self, "_id_warned", False):
                self._id_warned = True
                self.log(f"ident: {len(pcm) / c.fs:.1f}s does not fit a {usable:.1f}s window; "
                         "lengthen window_s or rely on the end-of-transmission ID")
            return False
        idt.mark_sent("window slot", len(pcm))
        self._play(pcm, "ID")
        return True

    def _tx_window(self, H, k: int) -> None:
        if self._resend_go:
            self._resend_go = False
            sm = self.sm
            audio = H.make_msg_audio(self.cfg.fs, H.MSG_GO, H.crc16(sm.my_tx_bat), b"",
                                     self._band, my_tx_bat=sm.my_tx_bat)
            audio, _ = H.scale_hs_tx(audio, self.cfg.amplitude, self.cfg.fs, is_probe=False,
                                     band=self._band)
            self._play(audio, "GO (resent)")
            return
        if self._window_id():
            return
        fr = self._next_frame()
        if fr is None:
            return
        seq, ack, data = fr
        with _quiet("encode"):
            audio = self._frame_audio(H, self._pack(seq, ack, data))
        self.tx_frames += 1
        self._play(audio, f"DATA seq={seq} ack={ack} {len(data)}B" if data
                   else f"ACK ack={ack}")
        self.log(f"window {k} tx seq={seq} ack={ack} {len(data)}B")

    def _rx_window(self, H, t_from: float, t_to: float) -> None:
        c = self.cfg
        i0, i1 = self._cap.idx_of(t_from), self._cap.idx_of(t_to)
        spans = self._bursts(i0, i1)
        if not spans:
            return
        a, b = spans[0][0], spans[-1][1]
        cap = self._cap.get(i0, i1)
        with _quiet("decode"):
            dem = H_dem(H, c, self._rx_bat, self._rx_band)
            stats, chunks = dem.decode_samples(cap)
        fr = self._frame_from(chunks)
        if fr is False:
            return
        t0, t1 = self._cap.t_of(a), self._cap.t_of(b)
        if fr is None:
            self._ev("err", t0, t1, f"undecodable window ({len(chunks)} chunks)")
            self.log(f"rx window undecodable ({len(chunks)} chunks)")
            self._maybe_msg(H, cap)
            return
        seq, ack, data = fr
        if self.rx_filter and not self.rx_filter({"seq": seq, "ack": ack, "len": len(data)}):
            self.log(f"rx seq={seq} dropped by filter")
            return
        self.rx_frames += 1
        evm = sum(s.get("evm_pct", 0.0) for s in stats) / max(1, len(stats))
        self._ev("lock", t0, t0, f"{len(chunks)} frames")
        self._ev("rx", t0, t1, f"DATA seq={seq} ack={ack} {len(data)}B evm={evm:.1f}%"
                 if data else f"ACK ack={ack}")
        self.log(f"rx seq={seq} ack={ack} {len(data)}B evm={evm:.1f}%")
        self._on_frame(seq, ack, data)

    def _maybe_msg(self, H, cap: np.ndarray) -> None:
        """Responder only: an undecodable window may be the initiator re-asking for GO."""
        if self.role != "responder" or not self.sm:
            return
        with _quiet("decode"):
            got = H.try_decode_msg(cap, self.cfg.fs, rates=(self.cfg.fs,), band=self._band)
        if got and got[1] in (H.MSG_RETRY, H.MSG_BAT_ACK):
            self.log("initiator missed GO -- resending in the next window")
            self._resend_go = True

    def _on_frame(self, seq: int, ack: int, data: bytes) -> None:
        deliver = None
        with self._lock:
            f = self._inflight
            if f and ack == ((f["seq"] + 1) & 0xFF):
                now = time.monotonic()
                self._ev("ack", now, now, f"seq={f['seq']}")
                self._inflight = None
                self._complete_locked(True)
                self._ack_evt.set()
            if data:
                self._owe_ack = True              # also for a duplicate: our ACK may be the lost one
                if seq == ((self._rx_next - 1) & 0xFF):
                    self.log(f"rx duplicate seq={seq} dropped")
                else:
                    if seq != self._rx_next:
                        self.log(f"rx seq gap: expected {self._rx_next} got {seq} (resync)")
                    self._rx_next = (seq + 1) & 0xFF
                    # schedule: one score step per chunk of the frame; the peer's next new frame starts after them
                    self._mel_rx_prev = self._mel_rx_n
                    self._mel_rx_n += self._chunks_of(data)
                    deliver = data
        if deliver is not None and self.on_data:
            self.on_data(deliver)

    # -- send ---------------------------------------------------------------

    def _complete_locked(self, ok: bool) -> None:
        p, self._pending = self._pending, None
        if p:
            p[2]["ok"] = ok
            p[1].set()

    def _complete(self, ok: bool) -> None:
        with self._lock:
            self._inflight = None
            self._complete_locked(ok)

    def send(self, data: bytes, timeout: float | None = None) -> bool:
        """half: queue one payload for the next TX window; True once the peer ACKed it.
        oneway / stream: transmit now; True once it has aired (nobody ACKs).
        bcast: queue for the running broadcast; True at once."""
        if self.cfg.mode == "bcast":
            self.feed(bytes(data))
            return True
        if self.cfg.mode in ("full", "arq"):
            if self.cfg.sender not in (self.role, "both") and self.cfg.mode == "arq":
                raise RuntimeError(f"mode arq: this end ({self.role}) only listens")
            return self._send_full(bytes(data))
        if self.cfg.mode != "half":
            return self._send_oneway(bytes(data))
        if len(data) > max(1, self.capacity or 255):
            raise ValueError(f"{len(data)} bytes exceeds the window capacity {self.capacity}")
        if not self._up.wait(self.cfg.hs_timeout_s):
            return False
        done, box = threading.Event(), {"ok": False}
        with self._lock:
            if self._pending or self._inflight:
                raise RuntimeError("a payload is already in flight")
            self._pending = (bytes(data), done, box)
        c = self.cfg
        done.wait(timeout or (c.retries + 2) * 2 * c.window_s + 30)
        return box["ok"]


    # -- oneway / stream ----------------------------------------------------

    def _send_oneway(self, data: bytes) -> bool:
        c = self.cfg
        if c.sender not in (self.role, "both"):
            raise RuntimeError(f"mode {c.mode}: this end ({self.role}) only listens")
        if not self._up.wait(c.hs_timeout_s):
            return False
        if not data:
            return True
        H, _ = _imports()
        if c.sender == "both" and self.role == "responder":
            # the initiator is still decoding our GO message; a burst that lands before it is listening is lost
            self._sleep_until(self.epoch + 4.0)
        if c.sender == "both":                   # turn-taking: let the peer's burst end before ours starts
            end = time.monotonic() + 30.0
            while self._air_busy(c.hang_s) and time.monotonic() < end and not self._stop.is_set():
                self._stop.wait(0.1)
            if self._air_busy(1.5):              # the peer just finished: its sink tail still mutes its receiver
                self._stop.wait(0.6)
        with self._txlock:
            if c.mode == "stream":
                body = len(data).to_bytes(4, "big") + data + H.crc16(data).to_bytes(2, "big")
                if -(-len(body) // c.chunk_bytes) > MAX_CHUNKS:
                    raise ValueError(f"{len(data)} bytes exceeds the stream capacity {self.capacity}")
                pad = int(os.environ.get("OFDM_TRAILER", "0"))       # experiment: padding chunks past the CRC, never delivered
                if pad:
                    body += b"\0" * (pad * c.chunk_bytes)
                with _quiet("encode"):
                    audio = self._frame_audio(H, body)
                self.tx_frames += 1
                self._play(audio, f"STREAM {len(data)}B", ident=True)
                return not self._stop.wait(c.gap_s)
            for i in range(0, len(data), self.capacity):
                piece = data[i:i + self.capacity]
                seq, self._tx_seq = self._tx_seq, (self._tx_seq + 1) & 0xFF
                with _quiet("encode"):
                    audio = self._frame_audio(H, self._pack(seq, 0, piece))
                self.tx_frames += 1
                self._play(audio, f"DATA seq={seq} {len(piece)}B", ident=True)
                if self._stop.wait(c.gap_s):         # silence so the listener sees separate bursts
                    return False
        return True

    def _listen(self, stream: bool) -> None:
        H, _ = _imports()
        c = self.cfg
        scan = (self._cap.idx_of(self._go_end) if self.role == "responder" and self._go_end
                else self._hs_rx_end)
        scan = max(scan, self._cap._start)
        st = None
        # a stream is one long burst whose chunks are encoded and played one by one, so quiet inside it
        # (encode time, audio underruns under load) is not its end: it ends on its length, or a real stall
        hang = max(c.hang_s, STREAM_STALL_S) if stream else c.hang_s
        while not self._stop.wait(0.25):
            n = self._cap.n
            spans = self._bursts(scan, n, gap_s=hang)
            if not spans:
                scan = max(scan, n - c.fs)           # a burst may be starting right now
                continue
            a, b = spans[0]
            closed = (n - b) >= hang * c.fs
            if stream:
                scan, st = self._stream_step(H, a, n, closed, scan, st)
            elif closed:
                if not self._is_ident((a, b)):
                    self._oneway_rx(H, a, b)
                scan = b

    def _decode_from(self, H, i0: int, i1: int | None, raw_retry: bool = True):
        """(base index, stats, chunks) for the capture slice [i0, i1).  raw_retry=False: the stream listeners, whose
        chunk counters never start at 0 and whose raw capture holds our own fdd carrier."""
        c = self.cfg
        base = max(i0, self._cap._start)
        dem = H_dem(H, c, self._rx_bat, self._rx_band)
        with _quiet("decode"):
            stats, chunks = dem.decode_samples(self._rx_get(i0, i1))
            if raw_retry and self._filt and (sorted(chunks) != list(range(len(chunks))) or not chunks):
                # the filter's ringing can cost the first frame; clean audio decodes unfiltered
                st2, ch2 = dem.decode_samples(self._rx_get(i0, i1, raw=True))
                if len(ch2) > len(chunks):
                    stats, chunks = st2, ch2
        return base, stats, chunks

    def _oneway_rx(self, H, a: int, b: int) -> None:
        fs = self.cfg.fs
        _, stats, chunks = self._decode_from(H, a - int(0.2 * fs), b + int(0.3 * fs))
        t0, t1 = self._cap.t_of(a), self._cap.t_of(b)
        fr = self._frame_from(chunks)
        if fr is False:
            return
        if fr is None:
            self.rx_errors += 1
            self._ev("err", t0, t1, f"undecodable frame ({len(chunks)} chunks)")
            self.log(f"rx burst undecodable ({len(chunks)} chunks)")
            return
        seq, ack, data = fr
        if self.rx_filter and not self.rx_filter({"seq": seq, "ack": ack, "len": len(data)}):
            self.log(f"rx seq={seq} dropped by filter")
            return
        with self._lock:
            if seq == ((self._rx_next - 1) & 0xFF):
                return
            gap = (seq - self._rx_next) & 0xFF
            self._rx_next = (seq + 1) & 0xFF
        if gap:
            self.lost_frames += gap
            self._ev("err", t0, t0, f"{gap} frame(s) lost before seq={seq}")
        self.rx_frames += 1
        evm = sum(s.get("evm_pct", 0.0) for s in stats) / max(1, len(stats))
        self._ev("lock", t0, t0, f"{len(chunks)} frames")
        self._ev("rx", t0, t1, f"DATA seq={seq} {len(data)}B evm={evm:.1f}%")
        self.log(f"rx seq={seq} {len(data)}B evm={evm:.1f}%")
        if data and self.on_data:
            self.on_data(data)

    def _stream_step(self, H, a: int, n: int, closed: bool, scan: int, st):
        """Decode the growing burst; hand contiguous bytes on.  Returns (scan, st)."""
        c = self.cfg
        now = time.monotonic()
        if st is None:
            st = dict(a=a, frm=a, next=0, buf=bytearray(), total=None, sent=0, t_poll=0.0)
            self._ev("lock", self._cap.t_of(a), self._cap.t_of(a), "stream start")
        if not closed and now - st["t_poll"] < c.stream_poll_s:
            return scan, st
        st["t_poll"] = now
        base, stats, chunks = self._decode_from(H, st["frm"] - int(0.2 * c.fs), None)
        starts = {s["seq"]: base + int(s["frame_start"]) for s in stats}
        buf = st["buf"]
        while st["next"] in chunks:
            k = st["next"]
            buf += chunks[k]
            st["next"] += 1
            st["frm"] = starts[k] + 1                # resume inside the last delivered frame
        if st["total"] is None and len(buf) >= 4:
            st["total"] = 4 + int.from_bytes(buf[:4], "big") + 2
            if st["total"] > MAX_CHUNKS * c.chunk_bytes:
                return self._stream_end(st, False, "bad stream length", n)
        if st["total"]:
            end = min(len(buf), st["total"] - 2)
            if end > 4 + st["sent"]:
                piece = bytes(buf[4 + st["sent"]:end])
                st["sent"] += len(piece)
                if self.on_data:
                    self.on_data(piece)
            if len(buf) >= st["total"]:
                data = bytes(buf[4:st["total"] - 2])
                ok = H.crc16(data) == int.from_bytes(buf[st["total"] - 2:st["total"]], "big")
                return self._stream_end(st, ok, "" if ok else "crc mismatch", self._after_frame(st["frm"], n) if ok else n)
        if closed:
            want = f" of {st['total'] - 6}" if st["total"] else ""
            return self._stream_end(st, False, f"truncated: {st['sent']}{want} bytes", n)
        return scan, st

    def _after_frame(self, i0: int, n: int) -> int:
        """End of the frame in progress at ``i0``: the next stream may already be on the air by now, so
        scanning on from the present would lose its first chunks."""
        spans = self._bursts(i0, n, gap_s=0.1)
        return spans[0][1] if spans else n

    def _stream_end(self, st, ok: bool, why: str, n: int):
        t0, t1 = self._cap.t_of(st["a"]), self._cap.t_of(n)
        if ok:
            self.rx_frames += 1
            self._ev("rx", t0, t1, f"STREAM {st['sent']}B")
        else:
            self.rx_errors += 1
            self._ev("err", t0, t1, f"stream {why}")
        self.log(f"stream end ok={ok} {st['sent']}B {why}")
        if not ok and os.environ.get("OFDM_FAIL_DUMP"):      # keep the audio of a failed stream for offline replay
            i0 = max(self._cap._start, st["a"] - self.cfg.fs)
            os.makedirs(os.environ["OFDM_FAIL_DUMP"], exist_ok=True)
            np.savez(os.path.join(os.environ["OFDM_FAIL_DUMP"], f"fail_{os.getpid()}_{i0}.npz"),
                     x=np.asarray(self._cap.get(i0, n), np.float32), bat=np.asarray(self._rx_bat),
                     band=np.asarray(self._rx_band), i0=i0, a=st["a"], frm=st["frm"], nxt=st["next"],
                     filt=repr(self._filt), fs=self.cfg.fs, chunk=self.cfg.chunk_bytes)
        if self.on_stream:
            self.on_stream(ok, st["sent"])
        return n, None

    # -- bcast (continuous broadcast) ----------------------------------------

    def feed(self, data: bytes) -> None:
        """bcast sender: queue bytes for the carrier.  Fed faster than the air for longer than BCAST_BACKLOG_S, the
        OLDEST bytes go (a live stream wants the present, and its next IDR recovers it)."""
        with self._feed_lock:
            self._feed += data
            over = len(self._feed) - self._feed_max
            if over > 0:
                del self._feed[:over]
                self.bcast_dropped += over

    def _bcast_audio(self, H, piece: bytes, seq: int, bat=None, band=None, gap: bool = False) -> np.ndarray:
        """One chunk's audio.  Gapless by default: while keyed, chunks play back to back with no inter-frame
        silence (it was 40 ms per chunk, 6% of the air and an audible pause).  ``gap`` keeps the silence, used for
        the first chunk of a keying so the lead-in tone stays apart from the first preamble."""
        c = self.cfg
        bat = self._tx_bat if bat is None else bat
        band = self._tx_band if band is None else band
        body = len(piece).to_bytes(BCAST_HDR, "big") + piece
        body += b"\0" * (c.chunk_bytes - len(body))
        with _quiet("encode"):
            audio = self._bcast_cmod_encode(H, body, seq, bat, band, gap)
            if audio is None:
                audio = H.make_data_audio(c.fs, body, bat, band, c.chunk_bytes, seq0=seq, gap=gap)
            return H.scale_hs_tx(audio, c.amplitude, c.fs, is_probe=False, band=band)[0]

    def _bcast_cmod_encode(self, H, body: bytes, seq: int, bat, band, gap: bool):
        """The chunk on the C modulator (ofdm_mod_encode_ex: the same samples as make_data_audio, graded by
        ofdm_mod_regress), one per (bat, band); None -> the caller encodes in numpy.  A library that is missing or
        lacks encode_ex is logged once and the broadcast stays on numpy for the rest of the link."""
        if getattr(self, "_bcast_cmod_off", False):
            return None
        c = self.cfg
        key = (np.asarray(bat, dtype=np.int32).tobytes(), None if band is None else tuple(np.asarray(band).tolist()))
        cache = self.__dict__.setdefault("_bcast_cmods", {})
        try:
            mod = cache.get(key)
            if mod is None:
                from .ofdm_cmod import CMod
                mod = cache[key] = CMod(H.hs_data_cfg(c.fs, band, c.chunk_bytes), bat)
                if len(cache) == 1:
                    self.log("bcast encoder: c (libofdm ofdm_mod_encode_ex)")
            with self.__dict__.setdefault("_bcast_cmod_lock", threading.Lock()):
                return mod.encode(body, seq, gap)
        except Exception as e:                       # NativeError, OSError: no/stale/foreign library
            self._bcast_cmod_off = True
            self.log(f"bcast encoder: numpy ({e})")
            return None

    def _bcast_rate(self, H, bat, band) -> tuple[int, float]:
        """(samples per chunk, payload bytes/s while keyed) of a carrier modulated with ``bat``."""
        n_one = len(self._bcast_audio(H, b"", 0, bat, band))
        return n_one, (self.cfg.chunk_bytes - BCAST_HDR) * self.cfg.fs / n_one

    def _bcast_tx(self) -> None:
        """Keep exactly one chunk queued ahead of the sink, so the carrier never breaks while there is data (and
        filler while there briefly is not); key up with the usual lead-in tone, unkey after BCAST_IDLE_S idle."""
        H, _ = _imports()
        c, pipe, fs = self.cfg, self.session.pipe, self.cfg.fs
        room = c.chunk_bytes - BCAST_HDR
        n_one, rate = self._bcast_rate(H, self._tx_bat, self._tx_band)   # payload bytes/s while keyed
        self.phy_bps = rate * 8
        with self._feed_lock:
            self._feed_max = int(BCAST_BACKLOG_S * rate)
        self.log(f"bcast ready: {n_one / fs:.2f} s per {c.chunk_bytes} B chunk, {rate * 8:.0f} bit/s payload, "
                 f"backlog cap {self._feed_max} B")
        seq, keyed, t_data = 0, False, 0.0
        tot = dict(data=0, fill=0, chunks=0)
        t_log = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            with self._feed_lock:
                have = len(self._feed)
            if have:
                t_data = now
            if not keyed:
                if not have:
                    self._stop.wait(0.05)
                    continue
                self._defer_for_air()
                keyed = True
                self.log(f"bcast carrier up at seq={seq}")
            elif not have and now - t_data > BCAST_IDLE_S:
                if pipe.tx_pending:
                    self._stop.wait(0.05)
                    continue
                keyed = False
                self.log(f"bcast carrier down at seq={seq}: nothing queued for {BCAST_IDLE_S:.0f} s")
                continue
            pending = pipe.tx_pending
            if pending > n_one // 2:                               # one chunk ahead is enough; wait for it to drain
                self._stop.wait(min(0.05, (pending - n_one // 2) / fs))
                continue
            with self._feed_lock:
                piece = bytes(self._feed[:room])
                del self._feed[:room]
            first = pending == 0 and not pipe.tx_pending          # first chunk of a keying (or after an underrun)
            audio = self._bcast_audio(H, piece, seq, gap=first)
            if first:                                             # lead-in tone, then the silence, then the chunk
                audio = keyup(self.session, audio, fs)
            t0 = time.monotonic() + pipe.tx_pending / fs
            self.session._keyup_end = t0 + len(audio) / fs
            self.session.play(audio, ident=True)
            self._ev("tx", t0, t0 + len(audio) / fs, f"BCAST seq={seq} {len(piece)}B")
            self.tx_frames += 1
            self.bcast_chunks += 1
            self.bcast_bytes += len(piece)
            tot["chunks"] += 1
            tot["data" if piece else "fill"] += len(piece) if piece else 1
            seq = (seq + 1) & 0xFFFF
            if now - t_log >= BCAST_LOG_S:
                t_log = now
                with self._feed_lock:
                    backlog = len(self._feed)
                self.log(f"bcast tx: chunks={tot['chunks']} data={tot['data']}B filler={tot['fill']} "
                         f"backlog={backlog}B dropped={self.bcast_dropped}B")

    def _listen_bcast(self) -> None:
        """Lock on whatever chunk comes first, then deliver every newer one in seq order.  The decode window starts
        just inside the last delivered frame, so each poll only demodulates the new audio; audio that will not
        decode (a fade, a station ID between chunks) is skipped once it is clearly not a frame still arriving."""
        H, _ = _imports()
        c, fs = self.cfg, self.cfg.fs
        scan = max(self._hs_rx_end, self._cap._start)
        st = None
        n_one, rate = self._bcast_rate(H, self._rx_bat, self._rx_band)  # the sender modulates with our rx BAT
        self.phy_bps = rate * 8
        while not self._stop.wait(0.25):
            n, now = self._cap.n, time.monotonic()
            if st is None:
                spans = self._bursts(scan, n, gap_s=max(c.hang_s, 0.5))
                if not spans:
                    scan = max(scan, n - fs)
                    continue
                st = dict(frm=spans[0][0], last=None, t_poll=0.0, t_got=now, chunks=0, data=0, lost=0, t_log=now)
                self.log("bcast carrier heard, acquiring")
            if now - st["t_poll"] < c.stream_poll_s:
                continue
            st["t_poll"] = now
            base, stats, chunks = self._decode_from(H, st["frm"] - int(0.2 * fs), None, raw_retry=False)
            evm = {s["seq"]: s.get("evm_pct", 0.0) for s in stats if s.get("seq") in chunks}
            got = sorted((base + int(s["frame_start"]), s["seq"]) for s in stats if s.get("seq") in chunks)
            got = self._ec_fill(H, got, chunks, evm)
            for start, seq in got:
                if st["last"] is None:
                    self.log(f"bcast lock at seq={seq}")
                    self._ev("lock", self._cap.t_of(start), self._cap.t_of(start), f"bcast seq={seq}")
                else:
                    d = (seq - st["last"]) & 0xFFFF
                    if d == 0 or d >= 0x8000:                     # already delivered (or a stale echo)
                        continue
                    if d > 1:
                        st["lost"] += d - 1
                        self.lost_frames += d - 1
                        self._ev("err", self._cap.t_of(start), self._cap.t_of(start), f"{d - 1} chunk(s) lost before seq={seq}")
                        self.log(f"bcast gap: {d - 1} chunk(s) lost before seq={seq} (total lost={st['lost']})")
                st["last"], st["frm"], st["t_got"] = seq, start + 1, now
                st["chunks"] += 1
                self.bcast_chunks += 1
                body = chunks[seq]
                ln = int.from_bytes(body[:BCAST_HDR], "big")
                t0 = self._cap.t_of(start)
                if ln > len(body) - BCAST_HDR:
                    self.rx_errors += 1
                    self._ev("err", t0, t0 + n_one / fs, f"BCAST seq={seq} bad length {ln}")
                    continue
                self._ev("rx", t0, t0 + n_one / fs, f"BCAST seq={seq} {ln}B evm={evm.get(seq, 0.0):.1f}%")
                if ln:
                    self.rx_frames += 1
                    self.bcast_bytes += ln
                    st["data"] += ln
                    if self.on_data:
                        self.on_data(bytes(body[BCAST_HDR:BCAST_HDR + ln]))
            if not got:
                if n - st["frm"] > 6 * fs:                         # several chunks' worth that will not decode
                    st["frm"] = n - 2 * fs
                if now - st["t_got"] > STREAM_STALL_S + 2 * c.stream_poll_s and not self._air_busy(STREAM_STALL_S):
                    self.log(f"bcast carrier lost after seq={st['last']}: chunks={st['chunks']} data={st['data']}B "
                             f"lost={st['lost']}")
                    self._ev("err", now, now, "bcast carrier lost")
                    st, scan = None, n
                    continue
            if now - st["t_log"] >= BCAST_LOG_S:
                st["t_log"] = now
                self.log(f"bcast rx: seq={st['last']} chunks={st['chunks']} data={st['data']}B lost={st['lost']}")


    # -- full duplex --------------------------------------------------------

    @staticmethod
    def _data_bins(cfg: dict) -> list[int]:
        """FFT bin of every data carrier (occupied minus pilots), in BAT order."""
        return [cfg["bin_lo"] + i for i in range(cfg["n_occupied"]) if i % cfg["pilot_spacing"]]

    def _sub_bat(self, full_bat, full_cfg: dict, sub_cfg: dict) -> np.ndarray:
        """Re-index a full-band table onto a sub-band's data bins.  A bin that is a pilot in
        the full band was never measured as data: it gets BPSK."""
        by_bin = dict(zip(self._data_bins(full_cfg), (int(b) for b in full_bat)))
        return np.array([by_bin.get(b, 1) for b in self._data_bins(sub_cfg)], dtype=np.int32)

    def _bits_dl_frac(self, full: dict, plan_fn) -> tuple[float, int, int]:
        """duplex_split=bits: the Hz share (for ``plan_duplex_bands``) that gives the responder ``dl_frac`` of the
        capacity.  ul = the initiator's TX table, dl = the responder's: both ends hold both after the exchange, so
        both compute the same split.  Returns (hz share, ul bits, dl bits); (cfg share, 0, 0) without usable tables."""
        c = self.cfg
        ul_bat, dl_bat = ((self._tx_bat, self._rx_bat) if self.role == "initiator"
                          else (self._rx_bat, self._tx_bat))
        hz = np.array(self._data_bins(full), np.float64) * c.fs / full["n_fft"]
        ul_bat, dl_bat = np.asarray(ul_bat, np.int64), np.asarray(dl_bat, np.int64)
        if len(ul_bat) != len(hz) or len(dl_bat) != len(hz) or not (ul_bat.sum() and dl_bat.sum()):
            return self._share(), 0, 0
        lo, hi, guard = float(c.freq_lo), float(c.freq_hi), float(c.duplex_guard_hz)
        usable = hi - lo - guard
        min_bw = plan_fn(0.5)["min_bw_hz"]
        step = c.fs / full["n_fft"]
        best = None
        ul_bw = min_bw
        while ul_bw <= usable - min_bw + 1e-9:
            edge = lo + ul_bw
            ub = int(ul_bat[hz <= edge].sum())
            db = int(dl_bat[hz >= edge + guard].sum())
            if ub + db:
                err = abs(db / (ub + db) - self._share())
                if best is None or err < best[0] - 1e-9:
                    best = (err, ul_bw, ub, db)
            ul_bw += step
        if best is None:
            return self._share(), 0, 0
        _, ul_bw, ub, db = best
        return (usable - ul_bw) / usable, ub, db

    @staticmethod
    def _ec_duty(rate_i: float, rate_r: float, dl_frac: float, strict: bool = False) -> tuple[float, float]:
        """duplex ec: (duty_initiator, duty_responder) -- the fraction of its chunks in which each end may carry a
        data frame (fillers, which still carry its ACKs, fill the rest; a credit of ``duty`` per chunk, spent 1 per
        data chunk, so 0.11 = about 1 chunk in 9).  ``rate_*`` = payload bytes/s of each direction's carrier at
        the rate its own BAT measured.  Not strict (``dl_frac``) and 0.5 = no cap (V.34: each direction runs what
        its line supports); otherwise the end over its share is held to it: share = duty*mine / (other +
        duty*mine).  ``strict`` (the share came from ``ratio``) enforces 50/50 as well, holding the faster
        direction to the slower one's rate.

        The cap is deliberately NOT a trimmed BAT: zero-bit carriers (a table trimmed to ~half its bits) made
        back-to-back chunks undecodable 1 time in 3 even on a clean loop (2026-11-13), so the PHY keeps its
        measured tables and the share is enforced by the link."""
        if rate_i <= 0 or rate_r <= 0 or (abs(dl_frac - 0.5) < 1e-9 and not strict):
            return 1.0, 1.0
        # the responder's share at full duty on both ends; whoever is over its share gets capped
        free = rate_r / (rate_i + rate_r)
        if abs(free - dl_frac) < 1e-9:
            return 1.0, 1.0
        capped_r = free > dl_frac
        f = dl_frac if capped_r else 1.0 - dl_frac
        mine, other = (rate_r, rate_i) if capped_r else (rate_i, rate_r)
        duty = min(1.0, f * other / ((1.0 - f) * mine))
        return (1.0, duty) if capped_r else (duty, 1.0)

    def _share(self) -> float:
        return self.cfg.responder_share()[0]

    def _setup_ec(self, H) -> None:
        """duplex ec: the whole band both ways, each direction on the table measured for it; the canceller trains
        on the echo of our own probe (the longest thing we played in the handshake, the far end listening)."""
        c = self.cfg
        initiator = self.role == "initiator"
        full = H.hs_data_cfg(c.fs, self._band, c.chunk_bytes)
        self._tx_band = self._rx_band = self._band
        self._filt = None
        self._dup_cfgs = (full, full)
        # payload bytes/s of each direction's continuous carrier, from the chunk length its table gives
        r_tx = (c.chunk_bytes - HDR) * c.fs / len(self._ec_chunk_audio(H, b"", 0))
        r_rx = (c.chunk_bytes - HDR) * c.fs / len(self._ec_chunk_audio(H, b"", 0, bat=self._rx_bat))
        rate_i, rate_r = (r_tx, r_rx) if initiator else (r_rx, r_tx)
        d_i, d_r = self._ec_duty(rate_i, rate_r, *c.responder_share())
        self._ec_duty_mine = d_i if initiator else d_r
        share = rate_r * d_r / (rate_i * d_i + rate_r * d_r)
        st = (self._ec.train() if self._ec is not None and c.ec_taps > 0
              else {"on": False, "why": "ec_taps=0"})
        if self._ec is not None and c.ec_taps <= 0:
            self._ec.on = False
        self.plan = {"duplex": "ec", "echo": st, "duty": (d_i, d_r), "rates": (rate_i, rate_r), "share": share}
        echo = (f"canceller on: delay {st['delay_ms']} ms, {st['taps']} taps, training ERLE {st['erle_db']} dB, "
                f"corr {st['corr']}" if st.get("on") else f"canceller off ({st.get('why')}"
                + (f", corr {st['corr']}" if "corr" in st else "") + ")")
        self.log(f"duplex tx {self._band[0]:.0f}-{self._band[1]:.0f} Hz rx {self._band[0]:.0f}-{self._band[1]:.0f} Hz"
                 f" tx_bits={int(np.asarray(self._tx_bat).sum())} rx_bits={int(np.asarray(self._rx_bat).sum())}"
                 f" (ec: {echo}; data duty initiator {d_i:.2f} responder {d_r:.2f}, window {c.ec_window}, carriers {rate_i * 8:.0f}/"
                 f"{rate_r * 8:.0f} bit/s, responder share {share:.2f})")

    def _setup_duplex(self, H) -> None:
        if self.cfg.duplex == "ec":
            self._setup_ec(H)
            return
        from ofdm_fdd import plan_duplex_bands
        c = self.cfg
        full = H.hs_data_cfg(c.fs, self._band, c.chunk_bytes)
        plan_fn = lambda frac: plan_duplex_bands(c.freq_lo, c.freq_hi, frac, c.duplex_guard_hz,
                                                 full["n_fft"], full["cp_ratio"], c.fs,
                                                 full["pilot_spacing"], 100.0, "high")
        frac, ub, db = (self._bits_dl_frac(full, plan_fn) if c.duplex_split == "bits" else (self._share(), 0, 0))
        plan = self.plan = plan_fn(frac)
        if ub + db:
            self.log(f"duplex split by bits: responder share {self._share():.2f} of {ub + db} bits/symbol "
                     f"(initiator {ub}, responder {db}) -> {plan['dl_frac_actual']:.2f} of the band in Hz")
        ul, dl = tuple(plan["ul_hz"]), tuple(plan["dl_hz"])
        mine, far = (ul, dl) if self.role == "initiator" else (dl, ul)
        mc, fc = H.hs_data_cfg(c.fs, mine, c.chunk_bytes), H.hs_data_cfg(c.fs, far, c.chunk_bytes)
        self._tx_bat = self._sub_bat(self._tx_bat, full, mc)
        self._rx_bat = self._sub_bat(self._rx_bat, full, fc)
        self._tx_band, self._rx_band = mine, far
        self._filt = (fc, mc)
        self._dup_cfgs = (mc, fc)
        self.log(f"duplex tx {mine[0]:.0f}-{mine[1]:.0f} Hz rx {far[0]:.0f}-{far[1]:.0f} Hz"
                 f" tx_bits={int(self._tx_bat.sum())} rx_bits={int(self._rx_bat.sum())}"
                 + (f" (harmonics overlap: {plan['harmonic_overlap']})"
                    if plan["harmonic_overlap"] else ""))

    def _listen_full(self) -> None:
        H, _ = _imports()
        c = self.cfg
        scan = (self._cap.idx_of(self._go_end) if self.role == "responder" and self._go_end
                else self._hs_rx_end)
        scan = max(scan, self._cap._start)
        while not self._stop.wait(0.25):
            n = self._cap.n
            spans = self._bursts(scan, n, gap_s=c.hang_s)
            if not spans:
                scan = max(scan, n - c.fs)
                continue
            a, b = spans[0]
            if (n - b) >= c.hang_s * c.fs:
                if not self._is_ident((a, b)):
                    self._full_rx(H, a, b)
                scan = b

    def _full_rx(self, H, a: int, b: int) -> None:
        fs = self.cfg.fs
        _, stats, chunks = self._decode_from(H, a - int(0.2 * fs), b + int(0.3 * fs))
        t0, t1 = self._cap.t_of(a), self._cap.t_of(b)
        fr = self._frame_from(chunks)
        if fr is False:
            return
        if fr is None:
            self.rx_errors += 1
            self._ev("err", t0, t1, f"undecodable frame ({len(chunks)} chunks)")
            # decode_samples' stats holds one entry per frame that DEMODULATED (RS passed), not
            # per SC peak: a peak whose _decode_frame failed leaves no trace in it.  So "0 frames"
            # means no peak, OR peaks that all failed to demodulate (e.g. the wrong one won, as
            # when the keyup tone false-locked ahead of the preamble).  Non-empty stats with no
            # frame means chunks arrived but the frame CRC/sequence did not check out.
            self.log(f"rx frame undecodable: {len(chunks)} chunk(s) from {len(stats)} "
                     f"demodulated frame(s)" + (f" first={stats[0]}" if stats else
                     " (no frame in the burst demodulated)"))
            return
        self.deliver(*fr, stats, len(chunks), t0, t1)

    # -- shared receiver (mesh) ------------------------------------------------

    def rx_demod(self):
        """(demodulator for what the peer transmits, its BAT, its band), or None before the link is up."""
        if self._rx_bat is None:
            return None
        if self._dem is None:
            H, _ = _imports()
            self._dem = H_dem(H, self.cfg, self._rx_bat, self._rx_band)
        return self._dem, self._rx_bat, self._rx_band

    def deliver(self, seq: int, ack: int, data: bytes, stats, nchunks: int, t0: float, t1: float) -> None:
        """A frame from our peer to us (or broadcast): ARQ, delivery, and the ACK we owe (sent on this thread)."""
        if self.rx_filter and not self.rx_filter({"seq": seq, "ack": ack, "len": len(data)}):
            self.log(f"rx seq={seq} dropped by filter")
            return
        self.rx_frames += 1
        self.last_activity = time.monotonic()
        evm = sum(s.get("evm_pct", 0.0) for s in stats) / max(1, len(stats))
        self._ev("lock", t0, t0, f"{nchunks} frames")
        self._ev("rx", t0, t1, f"DATA seq={seq} ack={ack} {len(data)}B evm={evm:.1f}%"
                 if data else f"ACK ack={ack}")
        if self.cfg.tx_window_chunks and data[:4] in (RSV_MAGIC, STA_MAGIC):
            self._on_rsv_ctrl(data)               # a window reservation's control frame, not data
            return
        self._on_frame(seq, ack, data)
        with self._lock:
            owe, self._owe_ack = self._owe_ack, False
            nxt, last = self._rx_next, (self._tx_seq - 1) & 0xFF
        if owe:                                    # ACK at once
            self._tx_frame_full(last, nxt, b"")

    def wants_ack(self) -> bool:
        """A frame of ours is waiting for the peer's ACK (so the peer is the station due on the air)."""
        with self._lock:
            return self._inflight is not None

    def _tx_frame_full(self, seq: int, ack: int, data: bytes, step: int | None = None) -> None:
        """One frame on the full band.  ``step``: schedule step of a DATA frame; None = the full table."""
        H, _ = _imports()
        bat, tag = None, ""
        if step is not None and self._mel_tx is not None:
            sc, n = self._mel_tx, self._chunks_of(data)
            bat = [sc.bat(self._tx_bat, step + k) for k in range(n)]      # one score step per chunk: a phrase
            tag = (f" schedule steps {step % len(sc)}-{(step + n - 1) % len(sc)}/{len(sc)} "
                   f"[{' '.join(sc.describe(step + k) for k in range(n))}] "
                   f"{'/'.join(str(int((b > 0).sum())) for b in bat)} carriers")
        with self._txlock:
            with _quiet("encode"):
                audio = self._frame_audio(H, self._pack(seq, ack, data), bat)
            self.tx_frames += 1
            self._play(audio, (f"DATA seq={seq} ack={ack} {len(data)}B" if data
                               else f"ACK ack={ack}") + tag, ident=True)
            self.last_activity = time.monotonic()
            self._stop.wait(self.cfg.gap_s)        # quiet so our bursts never run together

    # -- full duplex, echo cancelled: a continuous carrier (V.34 style) ------------------------------------------
    #
    # A line has no airtime to save and a receiver that re-acquires every burst loses lock in between, so once up
    # each end keeps its carrier on for good: one chunk queued ahead of the sink at all times, no keyup, no gap, no
    # unkey.  Every chunk is one frame [src|dst, seq, ack, len, crc] on its own preamble with a running 16-bit chunk
    # counter; it carries the in-flight DATA when that is due (first send, or a resend after ack_wait_s), else a
    # filler (len 0).  Every chunk carries our current ack, so ACKs ride the stream: there are no ACK bursts.

    @property
    def _tag(self) -> str:
        """Log prefix of the continuous-carrier data plane: 'ec' (echo cancelled) or 'fdd' (band split)."""
        return "ec" if self.cfg.duplex == "ec" else "fdd"

    def _ec_chunk_audio(self, H, body: bytes, cseq: int, bat=None, band=None) -> np.ndarray:
        """One stream chunk on our tx band, or (bat=, band=) the peer's, to size what we receive."""
        c = self.cfg
        body = body + b"\0" * (c.chunk_bytes - len(body))
        bat = self._tx_bat if bat is None else bat
        band = self._tx_band if band is None else band
        with _quiet("encode"):
            audio = H.make_data_audio(c.fs, body, bat, band, c.chunk_bytes, seq0=cseq, gap=False)
            return H.scale_hs_tx(audio, c.amplitude, c.fs, is_probe=False, band=band)[0]

    def _far_busy(self, look_s: float) -> bool:
        """The peer's carrier is on the line.  fdd: judged on the own-band-filtered capture, since our own carrier
        is always there in the raw one."""
        if not self._filt:
            return self._air_busy(look_s)
        if getattr(self.session, "ignore_busy", False):
            return False
        x = self._rx_get(self._cap.n - int(look_s * self.cfg.fs))
        return len(x) > 0 and float(np.sqrt(np.mean(x * x))) > self._thr()

    def _run_ec(self) -> None:
        t = threading.Thread(target=self._ec_tx, name="ofdm-ec-tx", daemon=True)
        t.start()
        self._threads.append(t)
        self._ec_listen()

    # Sliding window (LAPM style, k = ec_window): send() queues its pieces and returns once all are acked; the TX
    # loop numbers queued pieces into the window while it has room and puts one data frame per chunk on the air,
    # subject to the duty credit.  ack = the next seq the peer wants (cumulative); the peer keeps frames that land
    # past a gap.  The stream is FIFO, so only the OLDEST unacked frame can be missing for real: it alone has a
    # resend timer (srtt + 4 rttvar, Karn's rule, at most ack_wait_s); an ack that moves but stops short of what we
    # sent before that resend names the next hole, which then goes out at once.

    def _ec_rto(self) -> float:
        c = self.cfg
        if self._ec_srtt is None:
            return c.ack_wait_s
        # floor: both ends decode once per stream_poll_s, so an ack can lag a poll at each end
        return min(c.ack_wait_s, max(2.0 * c.stream_poll_s, self._ec_srtt + 4.0 * self._ec_rttvar))

    def _ec_pick_locked(self, now: float, t_end: float):
        """Number queued pieces into the window and pick the frame this chunk carries (or None).  Fails the oldest
        frame once it has used up its retries."""
        c = self.cfg
        while self._ec_q and len(self._ec_win) < max(1, c.ec_window):
            f = self._ec_q.popleft()
            f["seq"], self._tx_seq = self._tx_seq, (self._tx_seq + 1) & 0xFF
            self._ec_win[f["seq"]] = f
        if not self._ec_win:
            return None
        old = next(iter(self._ec_win.values()))
        if old["sent"] is not None and now >= old["sent"] + self._ec_rto():
            if old["sends"] > c.retries:
                del self._ec_win[old["seq"]]
                self.fail_count += 1
                self._ev("fail", now, now, f"seq={old['seq']}: no ACK after {c.retries} retries")
                self.log(f"{self._tag}: seq={old['seq']} failed after {c.retries} retries")
                old["box"]["ok"] = False
                old["box"]["evt"].set()
                return self._ec_pick_locked(now, t_end)
            pick = old
        else:
            pick = next((f for f in self._ec_win.values() if f["sent"] is None), None)
        if pick is not None:
            pick["sends"] += 1
            pick["sent"] = t_end                     # on the air until then; its ACK clock starts there
        return pick

    def _ec_acked_locked(self, ack: int, now: float) -> None:
        done = [s for s in self._ec_win if 1 <= ((ack - s) & 0xFF) <= 128]
        # one RTT sample per ack, from the OLDEST frame it covers (the peer decodes and acks in jumps, one per
        # stream poll, so the newest frame of a jump under-reads the round trip by up to a poll and a timer built on
        # it fires early), and only when nothing it covers was resent: a resent frame's ack is ambiguous (Karn) and
        # the frames behind a hole were held by it, not by the line
        clean = done and all(self._ec_win[s]["sends"] == 1 for s in done)
        for seq in done:
            f = self._ec_win.pop(seq)
            if clean and seq == done[0] and f["sent"] is not None:
                r = max(0.0, now - f["sent"])
                if self._ec_srtt is None:
                    self._ec_srtt, self._ec_rttvar = r, r / 2
                else:
                    self._ec_rttvar = 0.75 * self._ec_rttvar + 0.25 * abs(self._ec_srtt - r)
                    self._ec_srtt = 0.875 * self._ec_srtt + 0.125 * r
            self.ec_acked_bytes += len(f["data"])
            self._ev("ack", now, now, f"seq={seq}")
            box = f["box"]
            box["left"] -= 1
            if box["left"] <= 0:
                box["evt"].set()

    def _on_frame_ec(self, seq: int, ack: int, data: bytes) -> None:
        """A chunk from the peer's stream: its ack clears our window, its data goes into the receive window."""
        c, now, out = self.cfg, time.monotonic(), []
        with self._lock:
            self._ec_acked_locked(ack, now)
            if data:
                d = (seq - self._rx_next) & 0xFF
                if d >= 128:
                    pass                                                # a resend of what we already have
                elif d == 0:
                    out.append(data)
                    self._rx_next = (seq + 1) & 0xFF
                elif seq not in self._ec_rxbuf:
                    self._ec_rxbuf[seq] = (data, now)
            # a hole the sender gave up on (it failed that frame): skip it rather than hold the rest for good
            skip_s = (c.retries + 2) * c.ack_wait_s
            if self._ec_rxbuf and self._rx_next not in self._ec_rxbuf:
                oldest = min(t for _, t in self._ec_rxbuf.values())
                if now - oldest > skip_s:
                    nxt = min(self._ec_rxbuf, key=lambda s: (s - self._rx_next) & 0xFF)
                    self.log(f"{self._tag} rx: gap {self._rx_next}..{(nxt - 1) & 0xFF} never filled, skipped")
                    self._rx_next = nxt
            while self._rx_next in self._ec_rxbuf:
                out.append(self._ec_rxbuf.pop(self._rx_next)[0])
                self._rx_next = (self._rx_next + 1) & 0xFF
            for s_ in [s_ for s_ in self._ec_rxbuf if ((s_ - self._rx_next) & 0xFF) >= 128]:
                del self._ec_rxbuf[s_]
            self.ec_rx_bytes += sum(len(x) for x in out)
        if self.on_data:
            for x in out:
                self.on_data(x)

    def _ec_tx(self) -> None:
        H, _ = _imports()
        c, pipe, fs = self.cfg, self.session.pipe, self.cfg.fs
        n_one = len(self._ec_chunk_audio(H, b"", 0))
        self.phy_bps = (c.chunk_bytes - HDR) * 8 * fs / n_one
        self.log(f"{self._tag} carrier up: continuous, {n_one / fs:.2f} s per {c.chunk_bytes} B chunk, "
                 f"{self.phy_bps:.0f} bit/s payload, window {c.ec_window}")
        cseq, t_log = 0, time.monotonic()
        duty = min(1.0, max(1e-3, self._ec_duty_mine))
        credit = 1.0
        tot = dict(chunks=0, data=0, fill=0, resend=0, underrun=0)
        if duty < 1.0:
            self.log(f"{self._tag} data duty: {duty:.2f} of chunks may carry data (responder share {self._share():.2f})")
        try:
            while not self._stop.is_set():
                pending = pipe.tx_pending
                if pending > n_one // 2:                       # one chunk ahead is enough
                    self._stop.wait(min(0.05, (pending - n_one // 2) / fs))
                    continue
                if pending == 0 and tot["chunks"]:
                    tot["underrun"] += 1                         # the carrier broke: we were late
                now = time.monotonic()
                t0 = now + pending / fs
                credit = min(1.0, credit + duty)
                with self._lock:
                    nxt, last = self._rx_next, (self._tx_seq - 1) & 0xFF
                    send = self._ec_pick_locked(now, t0 + n_one / fs) if credit >= 1.0 - 1e-9 else None
                    self._owe_ack = False
                    inflight = len(self._ec_win)
                if send is not None:
                    credit -= 1.0
                    body = self._pack(send["seq"], nxt, send["data"])
                    what = f"DATA seq={send['seq']} ack={nxt} {len(send['data'])}B"
                    tot["data"] += 1
                    if send["sends"] > 1:
                        tot["resend"] += 1
                        self.retry_count += 1
                        self._ev("retry", now, now, f"seq={send['seq']} attempt {send['sends'] - 1}")
                        if self.on_retry:
                            self.on_retry(send["sends"] - 1)
                else:
                    body = self._pack(last, nxt, b"")
                    tot["fill"] += 1
                audio = self._ec_chunk_audio(H, body, cseq)
                self.session._keyup_end = t0 + len(audio) / fs
                self.session.play(audio, ident=False)
                self.tx_frames += 1
                tot["chunks"] += 1
                if send is not None:
                    self._ev("tx", t0, t0 + len(audio) / fs, what)
                    self.log(f"tx {what} (stream chunk {cseq}, {len(audio) / fs:.2f}s, {inflight} in flight)")
                cseq = (cseq + 1) & 0xFFFF
                if now - t_log >= EC_LOG_S:
                    t_log = now
                    self.log(f"{self._tag} tx: chunks={tot['chunks']} data={tot['data']} filler={tot['fill']} "
                             f"resends={tot['resend']} underruns={tot['underrun']} acked_bytes={self.ec_acked_bytes} "
                             f"in_flight={inflight} queued={len(self._ec_q)} rto={self._ec_rto():.1f}")
        finally:
            self.log(f"{self._tag} carrier down: chunks={tot['chunks']} data={tot['data']} filler={tot['fill']} "
                     f"resends={tot['resend']} underruns={tot['underrun']} acked_bytes={self.ec_acked_bytes}")
            with self._lock:
                left = list(self._ec_win.values()) + list(self._ec_q)
                self._ec_win.clear(); self._ec_q.clear()
            for f in left:
                f["box"]["ok"] = False
                f["box"]["evt"].set()

    def _send_ec(self, data: bytes) -> bool:
        """Queue the pieces for the window and wait until the peer has acked all of them (False as soon as one
        fails after its retries, or the link stops).  Several callers may send at once; their pieces interleave
        in the window, each caller's in order."""
        c = self.cfg
        if not self._up.wait(c.hs_timeout_s):
            return False
        room = c.chunk_bytes - HDR
        pieces = [data[i:i + room] for i in range(0, len(data), room)]
        if not pieces:
            return True
        box = dict(evt=threading.Event(), left=len(pieces), ok=True)
        with self._lock:
            for p in pieces:
                self._ec_q.append(dict(data=p, box=box, sends=0, sent=None))
        while not box["evt"].wait(1.0):
            if self._stop.is_set():
                return False
        return box["ok"] and box["left"] <= 0

    def _ec_fill(self, H, got: list, chunks: dict, evm: dict) -> list:
        """Timing recovery for the back-to-back stream.  The frame search finds a preamble by the RISING edge of its
        Schmidl-Cox metric; with no gap between chunks the previous chunk's last symbols can hold the metric above
        threshold right into the next preamble, so that preamble has no edge and is never tried -- data-dependent,
        about 1 data chunk in 3 on a full bulk stream (2026-11-13).  The stream's chunk length is fixed, so a chunk
        missing between two decoded ones sits at a known place: demodulate it there (matched-filter snap only)."""
        if len(got) < 2:
            return got
        c = self.cfg
        if getattr(self, "_ec_rx_len", None) is None:
            self._ec_rx_len = len(self._ec_chunk_audio(H, b"", 0, bat=self._rx_bat, band=self._rx_band))
            self._ec_dem = H_dem(H, c, self._rx_bat, self._rx_band)
        L, dem, pad = self._ec_rx_len, self._ec_dem, self.cfg.fs // 50
        extra = []
        for (s1, c1), (s2, c2) in zip(got, got[1:]):
            dk = (c2 - c1) & 0xFFFF
            if not 2 <= dk <= 16 or abs((s2 - s1) - dk * L) > L // 20:
                continue
            for j in range(1, dk):
                cs, pos = (c1 + j) & 0xFFFF, s1 + round(j * (s2 - s1) / dk)
                x = self._rx_get(pos - pad, pos + L + pad)
                if x is None or len(x) < L + pad:
                    continue
                with _quiet("decode"):
                    start, _score = dem._sc_match(x, pad)
                    fr = dem._decode_frame(x, start)
                if fr is None or fr.get("seq") != cs or "_chunk" not in fr:
                    continue
                chunks[cs] = fr["_chunk"]
                evm[cs] = fr.get("evm_pct", 0.0)
                extra.append((pos - pad + start, cs))
        if extra:
            self.ec_refilled = getattr(self, "ec_refilled", 0) + len(extra)
        return sorted(got + extra)

    def _ec_listen(self) -> None:
        """Lock on the peer's carrier once and follow its chunk counter (as the bcast listener does); every chunk
        is a frame for the ARQ.  A chunk-counter gap is counted in ``lost_frames``; ARQ resends what mattered."""
        H, _ = _imports()
        c, fs = self.cfg, self.cfg.fs
        scan = max(self._hs_rx_end, self._cap._start)
        st = None
        while not self._stop.wait(0.25):
            n, now = self._cap.n, time.monotonic()
            if st is None:
                spans = self._bursts(scan, n, gap_s=max(c.hang_s, 0.5))
                if not spans:
                    scan = max(scan, n - fs)
                    continue
                st = dict(frm=spans[0][0], last=None, t_poll=0.0, t_got=now, chunks=0, data=0, lost=0, t_log=now)
                self.log(f"{self._tag}: far carrier heard, acquiring")
            if now - st["t_poll"] < c.stream_poll_s:
                continue
            st["t_poll"] = now
            base, stats, chunks = self._decode_from(H, st["frm"] - int(0.2 * fs), None, raw_retry=False)
            evm = {s["seq"]: s.get("evm_pct", 0.0) for s in stats if s.get("seq") in chunks}
            got = sorted((base + int(s["frame_start"]), s["seq"]) for s in stats if s.get("seq") in chunks)
            got = self._ec_fill(H, got, chunks, evm)
            for start, cs in got:
                if st["last"] is None:
                    self.log(f"{self._tag}: lock at chunk {cs}")
                    self._ev("lock", self._cap.t_of(start), self._cap.t_of(start), f"{self._tag} chunk {cs}")
                else:
                    d = (cs - st["last"]) & 0xFFFF
                    if d == 0 or d >= 0x8000:
                        continue
                    if d > 1:
                        st["lost"] += d - 1
                        self.lost_frames += d - 1
                        self.log(f"{self._tag}: {d - 1} chunk(s) lost before chunk {cs} (total lost={st['lost']})")
                st["last"], st["frm"], st["t_got"] = cs, start + 1, now
                st["chunks"] += 1
                fr = self._frame_from({0: chunks[cs]})
                if fr is None:
                    self.rx_errors += 1
                    continue
                if fr is False:
                    continue
                seq, ack, data = fr
                t0 = self._cap.t_of(start)
                if self.rx_filter and not self.rx_filter({"seq": seq, "ack": ack, "len": len(data)}):
                    self.log(f"rx seq={seq} dropped by filter")
                    continue
                if data:
                    self.rx_frames += 1
                    st["data"] += 1
                    self._ev("rx", t0, t0, f"DATA seq={seq} ack={ack} {len(data)}B evm={evm.get(cs, 0.0):.1f}%")
                self.last_activity = time.monotonic()
                self._on_frame_ec(seq, ack, data)
            if not got:
                if n - st["frm"] > 6 * fs:
                    st["frm"] = n - 2 * fs
                if now - st["t_got"] > STREAM_STALL_S + 2 * c.stream_poll_s and not self._far_busy(STREAM_STALL_S):
                    self.log(f"{self._tag}: far carrier lost after chunk {st['last']}: chunks={st['chunks']} "
                             f"data={st['data']} lost={st['lost']}")
                    self._ev("err", now, now, f"{self._tag} carrier lost")
                    st, scan = None, n
                    continue
            if now - st["t_log"] >= EC_LOG_S:
                st["t_log"] = now
                self.log(f"{self._tag} rx: chunk={st['last']} chunks={st['chunks']} data={st['data']} lost={st['lost']} "
                         f"bytes={self.ec_rx_bytes} held={len(self._ec_rxbuf)} refilled={getattr(self, 'ec_refilled', 0)}")

    # -- full duplex, C engine (backend "cdx") ------------------------------

    def _start_cdx(self) -> None:
        """Build the C engine for this end's sub-bands and bit tables."""
        from .ofdm_cdx import CDuplexEnd
        c = self.cfg
        tx_cfg, rx_cfg = self._dup_cfgs
        hang = max(c.hang_s, 0.2)
        self._cdx = CDuplexEnd(c.fs, tx_cfg, rx_cfg, self._tx_bat, self._rx_bat,
                               thr_rms=self._thr(), hang_s=hang, retries=c.retries,
                               ack_wait_s=c.ack_wait_s, gap_s=max(c.gap_s, hang + 0.25))
        self.log(f"c engine: own-band filter {self._cdx.filter_taps} taps "
                 f"({self._cdx.filter_stop_db:.0f} dB), delay {self._cdx.delay} samples")

    def _cdx_rx(self, x: np.ndarray) -> None:
        """Capture thread: feed the engine, route decoded frames into its ARQ."""
        eng, c = self._cdx, self.cfg
        now = time.monotonic()
        if self._cdx_base is None:
            self._cdx_base = self._cap.n - len(x)
        base = self._cdx_base - eng.delay
        for e in eng.feed(x, now):
            t0, t1 = self._cap.t_of(base + e.a), self._cap.t_of(base + e.b)
            if not e.ok:
                if e.found:                     # a preamble was seen but the frame did not come out
                    self.rx_errors += 1
                    self._ev("err", t0, t1, f"undecodable frame ({e.found} found, {e.bad} bad)")
                continue
            if self.rx_filter and not self.rx_filter({"seq": e.seq, "ack": e.ack, "len": len(e.data)}):
                self.log(f"rx seq={e.seq} dropped by filter")
                continue
            self.rx_frames += 1
            self.last_activity = time.monotonic()
            self._ev("lock", t0, t0, f"{e.chunks} frames")
            self._ev("rx", t0, t1, f"DATA seq={e.seq} ack={e.ack} {len(e.data)}B evm={100 * e.evm:.1f}%"
                     if e.data else f"ACK ack={e.ack}")
            eng.on_frame(e, now)
        eng.air_busy(now)

    def _run_cdx(self) -> None:
        """Link thread: turn the ARQ's actions into audio, deliveries and send results."""
        H, _ = _imports()
        eng = self._cdx
        while not self._stop.wait(0.05):
            for a in eng.poll(time.monotonic()):
                if a.kind == "tx":
                    self._tx_cdx(H, a)
                elif a.kind == "deliver":
                    self.last_activity = time.monotonic()
                    if self.on_data:
                        self.on_data(a.data)
                else:                           # done
                    now = time.monotonic()
                    box = self._cdx_box
                    if a.ok:
                        self._ev("ack", now, now, f"seq={a.seq}")
                    else:
                        self.fail_count += 1
                        self._ev("fail", now, now, f"no ACK after {self.cfg.retries} retries")
                    if box:
                        box["ok"] = a.ok
                        box["evt"].set()

    def _tx_cdx(self, H, a) -> None:
        c = self.cfg
        if a.retry:
            self.retry_count += 1
            now = time.monotonic()
            self._ev("retry", now, now, f"attempt {a.retry}")
            if self.on_retry:
                self.on_retry(a.retry)
        audio = self._cdx.tx_audio(a.seq, a.ack, a.data)
        audio = H.scale_hs_tx(audio, c.amplitude, c.fs, is_probe=False, band=self._tx_band)[0]
        self.tx_frames += 1
        with self._txlock:
            self._play(audio, f"DATA seq={a.seq} ack={a.ack} {len(a.data)}B" if a.data
                       else f"ACK ack={a.ack}", ident=True)
        self.last_activity = time.monotonic()
        self._cdx.tx_done(time.monotonic())

    def _send_full_c(self, data: bytes) -> bool:
        c = self.cfg
        if not self._up.wait(c.hs_timeout_s):
            return False
        hang = max(c.hang_s, 0.2)
        self._cdx.configure(c.retries, c.ack_wait_s, max(c.gap_s, hang + 0.25))   # cfg may change after up
        with self._send_lock:                   # one payload at a time, like the Python path
            for i in range(0, max(len(data), 1), self.capacity):
                piece = data[i:i + self.capacity]
                if not piece:
                    break
                box = self._cdx_box = {"evt": threading.Event(), "ok": False}
                while self._cdx.send(piece) != 0:        # the previous frame is still in flight
                    if self._stop.wait(0.05):
                        return False
                if not box["evt"].wait((c.retries + 2) * (c.ack_wait_s + 15)) or not box["ok"]:
                    return False
        return True

    def _send_full(self, data: bytes) -> bool:
        if self._cdx is not None:
            return self._send_full_c(data)
        if self.cfg.mode == "arq":
            return self._send_arq(data)          # bursts, one frame in flight
        return self._send_ec(data)               # ec and fdd: the sliding window on the continuous carrier

    def _send_arq(self, data: bytes) -> bool:
        """arq: stop-and-wait, one burst per frame, resent after ack_wait_s. """
        c = self.cfg
        if c.tx_window_chunks > 0:
            return self._send_rsv(data)
        if not self._up.wait(c.hs_timeout_s):
            return False
        for i in range(0, max(len(data), 1), self.capacity):
            piece = data[i:i + self.capacity]
            if not piece:
                break
            with self._lock:
                seq, self._tx_seq = self._tx_seq, (self._tx_seq + 1) & 0xFF
                self._inflight = dict(seq=seq, data=piece, sends=0)
                self._ack_evt.clear()
                step, self._mel_tx_n = self._mel_tx_n, self._mel_tx_n + self._chunks_of(piece)
            ok = False
            for attempt in range(c.retries + 1):
                if self._stop.is_set():
                    return False
                if attempt:
                    self.retry_count += 1
                    now = time.monotonic()
                    self._ev("retry", now, now, f"attempt {attempt}")
                    if self.on_retry:
                        self.on_retry(attempt)
                if c.sender == "both":
                    # both ends may send DATA on this link: listen before talking, and back a resend off by a random
                    # fraction of a frame gap so two ends that collided do not collide again on the same timer
                    if attempt:
                        self._stop.wait(random.uniform(0.0, 2.0 * c.gap_s + 1.0))
                    end = time.monotonic() + 30.0
                    while self._air_busy(c.hang_s) and time.monotonic() < end and not self._stop.is_set():
                        self._stop.wait(0.1)
                with self._lock:
                    nxt = self._rx_next
                self._tx_frame_full(seq, nxt, piece, step)
                if self._ack_evt.wait(c.ack_wait_s):
                    ok = True
                    break
            with self._lock:
                self._inflight = None
            if not ok:
                self.fail_count += 1
                now = time.monotonic()
                self._ev("fail", now, now, f"no ACK after {c.retries} retries")
                return False
        return True

    # -- P2P TX window reservation ------------------------------------------------
    #
    # RESERVE (window id, N chunks, first schedule step, bytes, crc16) -> the receiver arms for N chunks and answers
    # STATUS (all N to stream) -> the sender streams them back to back in one burst, no ACKs, no pauses -> the
    # receiver reads the burst slot by slot and answers STATUS: the chunks to stream again.  A
    # lost STATUS: the sender reserves the same window again and the receiver repeats its status.  With a carrier
    # score, chunk k plays score step first+k: the tone map is the slot schedule, and every slot carries data.

    def _rsv_send_ctrl(self, payload: bytes) -> None:
        with self._lock:
            seq, self._tx_seq = self._tx_seq, (self._tx_seq + 1) & 0xFF
        self._tx_frame_full(seq, 0, payload)

    def _send_rsv(self, data: bytes) -> bool:
        from ofdm_handshake import crc16
        c = self.cfg
        if not self._up.wait(c.hs_timeout_s):
            return False
        cb = c.chunk_bytes
        per = max(1, min(c.tx_window_chunks, STA_MAX))
        for off in range(0, len(data), per * cb):
            win = bytes(data[off:off + per * cb])
            n = -(-len(win) // cb)
            with self._lock:
                wid, self._rsv_id = self._rsv_id, (self._rsv_id + 1) & 0xFFFF
                step0, self._mel_tx_n = self._mel_tx_n, self._mel_tx_n + n
            if not self._rsv_window(wid, win, n, step0, crc16(win)):
                return False
        return True

    def _rsv_status(self, timeout: float):
        """The peer's status for the window in flight, or None on timeout."""
        if not self._rsv_evt.wait(timeout):
            return None
        with self._lock:
            self._rsv_evt.clear()
            return list(self._rsv_tx["status"])

    def _rsv_window(self, wid: int, win: bytes, n: int, step0: int, crc: int) -> bool:
        c = self.cfg
        hdr = RSV_MAGIC + _RSV.pack(wid, n, step0 & 0xFFFFFFFF, len(win), crc)
        with self._lock:
            self._rsv_tx = dict(wid=wid, status=None)
            self._rsv_evt.clear()
        todo, tries, reserved = None, 0, False
        try:
            while not self._stop.is_set():
                if not reserved:
                    self.log(f"window {wid}: reserve {n} chunks ({len(win)} B)")
                    self._rsv_send_ctrl(hdr)
                    st = self._rsv_status(c.ack_wait_s)
                    if st is None:
                        tries += 1
                        if tries > c.retries:
                            break
                        continue
                    reserved, todo = True, st
                if not todo:
                    self.log(f"window {wid}: all {n} chunks in")
                    return True
                air = self._rsv_stream(wid, win, todo, step0, first=len(todo) == n)
                st = self._rsv_status(c.ack_wait_s + 0.25 * len(todo) + air * 0.05)
                if st is None:
                    reserved = False             # status lost: reserve again, the receiver repeats it
                    tries += 1
                    if tries > c.retries:
                        break
                    continue
                if set(st) >= set(todo):
                    tries += 1                   # nothing got through
                    if tries > c.retries:
                        break
                else:
                    tries = 0
                todo = st
            self.fail_count += 1
            now = time.monotonic()
            self._ev("fail", now, now, f"window {wid}: no progress after {c.retries} retries")
            return False
        finally:
            with self._lock:
                self._rsv_tx = None

    def _rsv_chunk_bat(self, sc, bat, step: int | None):
        return bat if sc is None or step is None else sc.bat(bat, step)

    def _rsv_stream(self, wid: int, win: bytes, todo: list, step0: int, first: bool) -> float:
        """Chunks ``todo`` of the window, back to back in one burst (chunk k numbered k, on step step0+k)."""
        H, _ = _imports()
        from ofdm_modem import AdaptiveModulator
        c, sc, cb = self.cfg, self._mel_tx, self.cfg.chunk_bytes
        cfg = H.hs_data_cfg(c.fs, self._tx_band, cb)
        parts = []
        with self._txlock:
            with _quiet("encode"):
                for i, k in enumerate(todo):
                    body = win[k * cb:(k + 1) * cb].ljust(cb, b"\0")
                    if sc is not None:
                        a = sc.audio(H, c.fs, body, [sc.bat(self._tx_bat, step0 + k)], self._tx_band, cb,
                                     c.schedule_boost_db, seq0=k)
                    else:
                        a = AdaptiveModulator(cfg, self._tx_bat).encode_bytes(body, seq0=k)
                    parts.append(a if i == len(todo) - 1 else a[:len(a) - int(cfg.get("frame_gap_samples", 0))])
                audio = H.scale_hs_tx(np.concatenate(parts), c.amplitude, c.fs, is_probe=False, band=self._tx_band)[0]
            self.tx_frames += 1
            if sc is not None and first:
                tag = f" schedule steps {step0 % len(sc)}+{len(todo)}/{len(sc)}"
            elif sc is not None:
                tag = f" schedule resend steps {','.join(str((step0 + k) % len(sc)) for k in todo)}"
            else:
                tag = ""
            n_b = min(len(win), len(todo) * cb)
            air = self._play(audio, f"DATA seq={wid} ack=0 {n_b}B window {wid}: {len(todo)} chunks"
                                    f"{'' if first else ' (again)'}{tag}", ident=True)
            self.last_activity = time.monotonic()
            self._stop.wait(c.gap_s)
        return air

    def _on_rsv_ctrl(self, data: bytes) -> None:
        if data[:4] == STA_MAGIC and len(data) >= 4 + _STA.size:
            wid, k = _STA.unpack(data[4:4 + _STA.size])
            miss = list(struct.unpack(f"<{k}H", data[4 + _STA.size:4 + _STA.size + 2 * k]))
            with self._lock:
                t = self._rsv_tx
                if t is None or t["wid"] != wid:
                    return
                t["status"] = miss
                self._rsv_evt.set()
            self.log(f"window {wid}: status {'all in' if not miss else f'{len(miss)} chunk(s) to send again'}")
        elif data[:4] == RSV_MAGIC and len(data) >= 4 + _RSV.size:
            wid, n, step0, nbytes, crc = _RSV.unpack(data[4:4 + _RSV.size])
            with self._lock:
                r = self._rsv_rx
                if r is None or r["wid"] != wid:
                    r = self._rsv_rx = dict(wid=wid, n=n, step0=step0, nbytes=nbytes, crc=crc, got={},
                                            expect=list(range(n))[:STA_MAX], done=False)
                    self.log(f"window {wid}: reserved {n} chunks ({nbytes} B) from step {step0}")
            self._rsv_reply(r)

    def _rsv_reply(self, r: dict) -> None:
        miss = [] if r["done"] else r["expect"]
        self._rsv_send_ctrl(STA_MAGIC + _STA.pack(r["wid"], len(miss)) + struct.pack(f"<{len(miss)}H", *miss))

    def rsv_expecting(self) -> bool:
        r = self._rsv_rx
        return bool(r and not r["done"] and r["expect"])

    def _rsv_air(self, step) -> int:
        """Samples from one chunk's preamble to the next one's on this slot's table."""
        key = ("rsv-air", None if step is None else step % len(self._mel_rx))
        if key not in self._mel_dems:
            H, _ = _imports()
            from ofdm_modem import AdaptiveModulator
            mod = AdaptiveModulator(H.hs_data_cfg(self.cfg.fs, self._rx_band, self.cfg.chunk_bytes),
                                    self._rsv_chunk_bat(self._mel_rx, self._rx_bat, step))
            with contextlib.redirect_stdout(io.StringIO()):
                n = len(mod.encode_bytes(b"\0" * self.cfg.chunk_bytes))
            self._mel_dems[key] = n - mod.frame_gap_samp
        return self._mel_dems[key]

    def _rsv_dem(self, step):
        key = ("rsv-dem", None if step is None else step % len(self._mel_rx))
        if key not in self._mel_dems:
            H, _ = _imports()
            self._mel_dems[key] = H_dem(H, self.cfg, self._rsv_chunk_bat(self._mel_rx, self._rx_bat, step),
                                        self._rx_band)
        return self._mel_dems[key]

    def rx_window(self, x: np.ndarray) -> bool:
        """The reserved stream in burst ``x``: every expected chunk decoded from its own slot, then our status.
        False when the burst is not the stream, for the ordinary decode."""
        r = self._rsv_rx
        if not r or r["done"] or not r["expect"]:
            return False
        fs, lead = self.cfg.fs, int(0.015 * self.cfg.fs)
        mel = self._mel_rx is not None
        at, skipped, got = None, 0, []
        for k in r["expect"]:
            step = r["step0"] + k if mel else None
            air = self._rsv_air(step)
            if at is None:                        # not anchored yet: search the start of the burst
                y0, y = 0, x[:skipped + air + 2 * fs]
            else:
                y0 = max(0, at - lead)
                y = x[y0:at + air + lead]
            with _quiet("decode"):
                st, ch = self._rsv_dem(step).decode_samples(y)
            if k in ch:
                if not got and unpack_frame(ch[k]) is not None and ch[k][HDR:HDR + 4] in (RSV_MAGIC, STA_MAGIC):
                    return False                  # a control frame, not the stream
                r["got"][k] = ch[k]
                got.append(k)
                f = next((d.get("frame_start") for d in st if d.get("seq") == k), None)
                at = (y0 + int(f) if f is not None else (at or 0)) + air
            elif at is not None:
                at += air                         # the slot is lost; the next one starts where the schedule says
            else:
                skipped += air
        if not got:
            return False
        miss = [k for k in range(r["n"]) if k not in r["got"]]
        self.log(f"window {r['wid']}: {len(got)}/{len(r['expect'])} chunks this burst, "
                 f"{r['n'] - len(miss)}/{r['n']} in")
        if not miss:
            from ofdm_handshake import crc16
            data = b"".join(r["got"][k] for k in range(r["n"]))[:r["nbytes"]]
            if crc16(data) == r["crc"]:
                r["done"], r["expect"] = True, []
                self.rx_frames += 1
                self.last_activity = time.monotonic()
                if self.on_data:
                    self.on_data(data)
            else:
                self.log(f"window {r['wid']}: crc mismatch -- every chunk again")
                r["got"], miss = {}, list(range(r["n"]))
        if not r["done"]:
            r["expect"] = miss[:STA_MAX]
        self._rsv_reply(r)
        return True

    # -- carrier-schedule OFDM (carrier_schedule.py) ----------------------------------------------

    def _setup_schedule(self, H) -> None:
        c = self.cfg
        for attr, path, band in (("_mel_tx", c.schedule_tx, self._tx_band), ("_mel_rx", c.schedule_rx, self._rx_band)):
            sc = None
            if path:
                try:
                    sc = carrier_schedule.CarrierScore(carrier_schedule.load_score(path), H.hs_data_cfg(c.fs, band, c.chunk_bytes),
                                             c.schedule_step_carriers)
                    self.log(f"schedule {attr[-2:]}: {sc.summary()}")
                except (OSError, ValueError, KeyError) as e:
                    self.log(f"schedule {attr[-2:]}: {path}: {e} -- off")
            setattr(self, attr, sc)
        self._mel_tx_n = self._mel_rx_n = 0
        self._mel_dems = {}

    @property
    def schedule_rx(self):
        return self._mel_rx

    def _chunks_of(self, data: bytes) -> int:
        """Chunks one link frame carrying ``data`` takes on the air."""
        return max(1, -(-(len(data) + HDR) // self.cfg.chunk_bytes))

    def rx_demods_for(self, x: np.ndarray) -> list:
        """Demodulators to try on burst ``x``, in order.  With a receive score: the peer's next new frame, a resend of the last frame, then the full table."""
        base = self.rx_demod()
        if base is None:
            return []
        dem, bat, band = base
        if self._mel_rx is None:
            return [dem]
        starts = [self._mel_rx_n] + ([self._mel_rx_prev] if self._mel_rx_prev != self._mel_rx_n else [])
        return [_PhraseDem(self, s, bat, band) for s in starts] + [dem]

    def _mel_chunk_air(self, step: int, bat, band) -> int:
        """Samples from one scheduled chunk's preamble to the next one's: its frame on this step's table plus one gap
       ."""
        sc = self._mel_rx
        key = ("air", step % len(sc))
        if key not in self._mel_dems:
            H, _ = _imports()
            from ofdm_modem import AdaptiveModulator
            mod = AdaptiveModulator(H.hs_data_cfg(self.cfg.fs, band, self.cfg.chunk_bytes), sc.bat(bat, step))
            with contextlib.redirect_stdout(io.StringIO()):        # not _quiet(): the caller holds its DSP lock
                n = len(mod.encode_bytes(b"\0" * self.cfg.chunk_bytes))
            self._mel_dems[key] = n - mod.frame_gap_samp            # [gap, frame, gap] -> frame + one gap
        return self._mel_dems[key]

    def _mel_dem(self, step: int, bat, band):
        sc = self._mel_rx
        k = step % len(sc)
        if k not in self._mel_dems:
            H, _ = _imports()
            self._mel_dems[k] = H_dem(H, self.cfg, sc.bat(bat, step), band)
        return self._mel_dems[k]


class _PhraseDem:
    """A scheduled frame's demodulator: chunk k was sent on score step ``start + k``, so it is decoded with that step's
    table until the chunks so far make a whole frame or a chunk is missing.

    Every chunk starts with the same full-band preamble but lasts as long as its own table needs, so a demodulator
    run over the whole burst locks onto the chunk BEFORE its own, fails it, and skips ahead by its own frame length
    -- past the start of the chunk it is for.  So chunk k is decoded from just before its own preamble: where
    chunk k-1 was found plus k-1's length on its table."""

    LEAD_S = 0.015                       # decode window opens this far before the expected preamble

    def __init__(self, link, start: int, bat, band):
        self.link, self.start, self.bat, self.band = link, start, bat, band

    def decode_samples(self, x):
        stats, chunks = [], {}
        lead = int(self.LEAD_S * self.link.cfg.fs)
        at = None                         # sample index in x where chunk k's preamble is due
        for k in range(-(-255 // self.link.cfg.chunk_bytes)):      # a frame is 255 B at most
            step = self.start + k
            off = 0 if at is None else max(0, at - lead)
            st, ch = self.link._mel_dem(step, self.bat, self.band).decode_samples(x[off:])
            if k not in ch:
                break
            st_k = [d for d in st if d.get("seq") == k]
            chunks[k] = ch[k]
            stats += st_k
            if unpack_frame(b"".join(chunks[i] for i in range(k + 1))) is not None:
                break
            found = st_k[0].get("frame_start") if st_k else None
            here = off + int(found) if found is not None else (at or 0)
            at = here + self.link._mel_chunk_air(step, self.bat, self.band)
        return stats, chunks


def H_dem(H, c: OfdmConfig, bat, band):
    from ofdm_modem import AdaptiveDemodulator
    return AdaptiveDemodulator(H.hs_data_cfg(c.fs, band, c.chunk_bytes), bat)
