# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""IqLink: the complex-baseband OFDM modem on an I/Q session.

The engine path for complex samples, the counterpart of OfdmLink for an ``iq:`` device:

    session = open_iq_session(cfg, "iq:127.0.0.1:5601")      # Session(rate=fs, iq=True)
    link = IqLink(session, cfg, on_frame=print).start()
    link.send(b"hello")                  # frames at the agreed bit table, tx_gain level
    link.send_probe()                    # the channel probe, same level
    res = link.grade_probe(window_s)     # the peer's probe in the last window_s of RX history
    link.set_bat(res.bat)                # both ends, by hand ...
    neg = link.negotiate(initiator=True) # ... or the probe -> BAT exchange with the peer (below)

What it does: one worker thread drains a pipe tap into a CxReceiver and hands every decoded
frame to ``on_frame`` and to ``frames`` (a bounded deque); TX goes through ``Session.play_raw``
(no station-ID audio: that is real-valued audio and has no place in an I/Q stream).  Every burst
is sent at ``ofdm_cx.tx_gain(cfg)``, probes included, so the probe SNR is the frames' SNR.

Frames carry ``seq`` and a fixed-size chunk; ``Reassembler`` puts a run of them back together, and
``send_message`` / ``receive_message`` wrap one payload as a 4-byte magic + length ... CRC-32 so the
far end knows how many frames to wait for and where the NUL padding starts.

The probe -> BAT exchange (``negotiate``, step 9).  Each direction gets its own table: the one the
FAR end measured from our probe.  Control messages always go at the base table (``default_bits``
everywhere) and the receiver keeps the base table next to the probed one (CxReceiver tries both),
so a lost message is just retried and nobody has to switch tables at an agreed instant:

    initiator I                                   responder R
    PROBE  ------------------------------------->  grade -> table for I->R, add it to RX
           <---------------  BAT{I->R, have=0} + PROBE   (have: "I hold my own TX table")
    TX = I->R; grade -> table for R->I, add to RX
    BAT{R->I, have=1}  ------------------------->  TX = R->I; peer has its table: done
           <---------------  READY (sent at R->I: the new table proves itself)
    done; READY (at I->R)  --------------------->  settled
    data at I->R ...                               data at R->I ...

Round-trip delay (RTD), measured as V.34 does in its ranging phase rather than assumed: every PROBE is
preceded by a PING control message carrying a token, and every BAT echoes the newest token its sender
heard plus how long it held it before answering.  RTD = (answer decoded - ping queued) - hold, on the
wall clock, so it is the whole protocol loop: both modems' TX buffering, both air times' worth of
detection, and the path delay twice.  The token keeps it right when retries cross in flight.  I
measures it from R's first BAT, R from I's BAT{have=1}.

Retries: before the RTD is known, no answer -> resend what the peer is missing (PING + PROBE while we
have no measurement of it or it has none of us, else our BAT), backing off ``retry_s``, 2x, 4x ... up to
``cfg.max_rtd_s``; once it is known, every retry and the linger after we are done are sized from it.
A repeated BAT{have=1} after we are done is answered with READY again.

Selective-repeat ARQ (``send_reliable`` / ``receive_reliable``, 2026-11-14).  The data frames go out as
``send_message`` frames, then a POLL control message names them (message id, first seq, frame count,
data table).  Every link runs a responder thread that answers a POLL with STAT: a bitmap of the frames
it holds and, when frames were lost or needed erasures, a TABLE UPDATE graded from what it decoded
(per-carrier decision distance and |H| from ofdm_cxdem_carrier_quality: a dead carrier goes to 0 bits,
a noisy one down to what its measured SNR carries; never up).  The sender adopts the table, resends only
the missing frames, and polls again; the receiver tries the new table first and keeps the old ones.  A
lost POLL or STAT costs one more POLL round (sized from the measured RTD).  Control stays at the base
table, so a table that went bad never takes the control channel with it.

    sender                                           receiver
    data frames s0..s0+n-1 (TX table)  ---------->   held in a seq -> chunk store
    POLL{id, s0, n}  (base)            ---------->   bitmap; lost/erased frames -> grade a new table
            <-----------------------------------     STAT{id, bitmap[, table for the sender's TX]}
    set_tx_bat(table); resend missing  ---------->   decoded at the new table (tried first)
    POLL ...                                         complete + CRC ok: deliver, STAT{complete}
"""

from __future__ import annotations

import collections
import dataclasses
import os
import struct
import sys
import threading
import time
import zlib
from typing import Callable

import numpy as np

from ..audio import Registry
from .events import OfdmData, ProbeReport
from .modemcfg import IqConfig
from .ofdm_cx import CxFrame, CxModem, CxProber, CxProbeResult, CxReceiver, ToneExciser, table_id, tx_gain
from .session import Session

MSG_MAGIC = b"IQM1"                    # data message
BAT_MAGIC = b"IQB1"                    # control: a bit table for the recipient's TX
READY_MAGIC = b"IQR1"                  # control: negotiation done at this end
PING_MAGIC = b"IQP1"                   # control: ranging token, sent just before every PROBE
POLL_MAGIC = b"IQQ1"                   # control (ARQ): which frames of message id do you hold?
STAT_MAGIC = b"IQS1"                   # control (ARQ): the answer, a bitmap [+ a table for our TX]
CONTROL_MAGICS = (BAT_MAGIC, READY_MAGIC, PING_MAGIC, POLL_MAGIC, STAT_MAGIC)
_POLL = struct.Struct(">HIHI")         # message id, first seq, frames, table_id of the data frames
_STAT_HDR = struct.Struct(">HIHB")     # message id, first seq, frames, flags (1 complete, 2 table follows,
                                       # 4 CRC failed: everything is resent, 8 send me a probe)
# Bit-loading floors (dB, before the margin) as ofdm_allocate: BPSK, QPSK, 16-QAM, 64-QAM; and the
# squared minimum distance of each unit-energy constellation (ofdm_mod_table).
_FLOOR_DB = {1: 4.0, 2: 7.0, 4: 14.0, 6: 21.0}
_DMIN2 = {1: 4.0, 2: 2.0, 4: 0.4, 6: 4.0 / 42.0}
DEAD_HREL = 0.1                        # mean |H|^2 under 10 dB below the loaded carriers': dead carrier
ARQ_MARGIN_DB = 3.0                    # regrade margin: measured on the data itself, so less than the probe's
                                       # snr_margin_db (64-QAM kept down to score 0.17: SER ~1e-3 before RS)
BAT_VERSION = 2                        # v2 adds the ranging echo (v1 nodes are not around: test bench only)
_BAT_HDR = struct.Struct(">BBH")       # version, flags (bit 0: sender holds its own TX table), n
_BAT_TAIL = struct.Struct(">ffHI")     # the measurement it came from: SNR median dB, CFO Hz;
                                       # ranging echo: newest PING token heard (0xFFFF none), held ms
_PING = struct.Struct(">H")            # token
NO_TOKEN = 0xFFFF
_MSG_HDR = struct.Struct(">4sI")              # magic, payload length


def open_iq_session(cfg: IqConfig, rx: str | None, tx: str | None = "",
                    registry: Registry | None = None) -> Session:
    """A Session at the profile's rate and block, complex, opened on ``rx``/``tx`` (``tx=""`` =
    the same address as rx: one ``iq:`` socket carries both directions)."""
    s = Session(registry, rate=int(cfg.fs), block=int(cfg.block), iq=True)
    s.open(rx, rx if tx == "" else tx)
    return s


class Reassembler:
    """Chunks by sequence number -> bytes.  ``add`` returns True when seq0..seq0+n-1 are all in.
    ``table`` / ``since``: only frames decoded with that bit table, at or after that RX index
    (so an older message or a control frame that reuses a sequence number is not mixed in)."""

    def __init__(self, seq0: int, n: int, seq_mod: int, table: int | None = None, since: int | None = None) -> None:
        self.seq0, self.n, self.mod = seq0, n, seq_mod
        self.table, self.since = table, since
        self.chunks: dict[int, bytes] = {}

    def add(self, f: CxFrame) -> bool:
        if f.ok and (self.table is None or f.table == self.table) and (self.since is None or f.index >= self.since):
            k = (f.seq - self.seq0) % self.mod
            if k < self.n:
                self.chunks.setdefault(k, f.payload)
        return self.complete

    @property
    def complete(self) -> bool:
        return len(self.chunks) == self.n

    def data(self, nbytes: int | None = None) -> bytes:
        b = b"".join(self.chunks[k] for k in range(self.n))
        return b if nbytes is None else b[:nbytes]


class IqLink:
    def __init__(self, session: Session, cfg: IqConfig, bat=None,
                 on_frame: Callable[[CxFrame], None] | None = None, hear_own: bool | None = None) -> None:
        if not session.spec.iq:
            raise ValueError("IqLink needs an I/Q session (open_iq_session)")
        if session.spec.rate != int(cfg.fs):
            raise ValueError(f"session rate {session.spec.rate} != IqConfig.fs {cfg.fs}")
        self.session, self.cfg = session, cfg
        self.on_frame = on_frame
        # False: zero RX blocks captured while we transmit.  None: from cfg.duplex (full = keep them).
        self.hear_own = (str(getattr(cfg, "duplex", "half")) == "full") if hear_own is None else hear_own
        self.exciser = ToneExciser(cfg.fs, int(cfg.block)) if getattr(cfg, "excise", False) else None
        self._ident_at: float | None = None      # last station ID (monotonic); None = never sent
        self._rx_pow: collections.deque[float] = collections.deque(maxlen=500)   # block powers (LBT floor)
        self.level = dict(rx_rms_dbfs=float("-inf"), rx_peak=0.0, rx_clipped=0)
        self._probe_t = time.monotonic()         # last probe exchanged (re-probe timer)
        self._want_probe: int | None = None      # RX index since which we wait for a requested probe
        self._neg_rx: list[int] | None = None    # the RX table negotiate settled (carriers ARQ may zero)
        self.gain = tx_gain(cfg)
        self.prober = CxProber(cfg)
        self.frames: collections.deque[CxFrame] = collections.deque(maxlen=4096)
        self.stats = collections.Counter()
        self._lock = threading.Lock()
        self._rxlock = threading.Lock()         # the receiver: _run feeds it, set_rx_tables edits it
        # control messages: default_bits everywhere, in short frames (cfg.ctl_chunk_bytes) so a fade
        # cannot move under one; the receiver tries this modem on every hit (CxReceiver.set_ctl)
        ctl = int(getattr(cfg, "ctl_chunk_bytes", 0) or 0)
        self.base = CxModem(dataclasses.replace(cfg, chunk_bytes=ctl) if ctl and ctl != cfg.chunk_bytes else cfg)
        self._seq = 0
        self.rtd_s: float | None = None          # round trip measured by negotiate()
        # ARQ (send_reliable / the responder): received chunks by seq, delivered messages
        self._arq_lock = threading.Lock()
        self._arq_q: collections.deque[CxFrame] = collections.deque(maxlen=8192)
        self._arq_ev = threading.Event()
        self._store: dict[int, tuple[bytes, int]] = {}     # seq -> (chunk, rx index)
        self._qual: list[tuple[int, np.ndarray, np.ndarray, int]] = []   # (qtable, score, hrel, erased)
        self._ctl_seen: collections.deque = collections.deque(maxlen=256)  # (magic, seq) of POLL/STAT acted on
        self._negotiating = self._neg_ok = False
        self._late_ready_t = 0.0
        self._delivered: collections.OrderedDict[int, int] = collections.OrderedDict()  # msg id -> seq0
        self._msg_id = 0
        self._arq_thread: threading.Thread | None = None
        self.messages: collections.deque[bytes] = collections.deque(maxlen=256)
        self._msg_ev = threading.Event()
        self.on_message: Callable[[bytes], None] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tap = None
        self.set_bat(bat)

    # -- configuration ------------------------------------------------------

    def set_bat(self, bat=None) -> None:
        """Switch both directions to a bit table (None: ``default_bits`` everywhere).  Both ends
        must switch together; frames in flight at the old table are lost."""
        with self._lock, self._rxlock:
            self.mod = CxModem(self.cfg, bat)
            self.rx = CxReceiver(self.cfg, self.mod.bat)
            if self.base.x.c.chunk_bytes != self.mod.x.c.chunk_bytes:
                self.rx.set_ctl(self.base)
            self.bat = self.mod.bat

    def set_tx_bat(self, bat=None) -> None:
        """Our TX table only (the one the peer measured from our probe); RX is untouched."""
        with self._lock:
            self.mod = CxModem(self.cfg, bat)
            self.bat = self.mod.bat

    def set_rx_tables(self, tables) -> None:
        """The tables the receiver tries, first preferred; the buffer and timebase are kept."""
        with self._rxlock:
            self.rx.set_tables(list(tables))

    @property
    def rx_tables(self) -> list[list[int]]:
        return self.rx.tables

    @property
    def seq_mod(self) -> int:
        return 1 << (8 * int(self.cfg.seq_bytes))

    @property
    def bits_per_symbol(self) -> int:
        return int(sum(self.bat))

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "IqLink":
        pipe = self.session.pipe
        if pipe is None:
            raise RuntimeError("session not open")
        self._tap = pipe.subscribe(1024)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="iq-link", daemon=True)
        self._thread.start()
        self._arq_thread = threading.Thread(target=self._arq_serve, name="iq-arq", daemon=True)
        self._arq_thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._arq_ev.set()
        for t in (self._thread, self._arq_thread):
            if t:
                t.join(2.0)
        self._arq_thread = None
        if self._tap is not None and self.session.pipe is not None:
            self.session.pipe.unsubscribe(self._tap)
        self._thread = self._tap = None

    def _run(self) -> None:
        tap = self._tap
        while not self._stop.is_set():
            b = tap.get(timeout=0.1)
            if b is None:
                continue
            self.stats["rx_blocks"] += 1
            if b.tx_active and not self.hear_own:
                self.stats["rx_muted"] += 1
                x = np.zeros_like(b.samples)    # keep the timebase continuous, drop the content
            else:
                x = b.samples
                p = float(np.mean(np.abs(x) ** 2)) if len(x) else 0.0
                self._rx_pow.append(p)
                pk = float(np.max(np.abs(x))) if len(x) else 0.0
                self.level["rx_rms_dbfs"] = 10 * np.log10(max(p, 1e-20))
                self.level["rx_peak"] = max(self.level["rx_peak"] * 0.99, pk)
                if pk >= 0.999:
                    self.level["rx_clipped"] += int(np.count_nonzero(np.abs(x) >= 0.999))
            if self.exciser is not None:
                x = self.exciser.process(x)
            try:
                with self._rxlock:
                    got = self.rx.feed(x, b.index)
            except Exception as e:              # pragma: no cover - a C error must not kill RX
                self.stats["rx_errors"] += 1
                self.session._emit_raw_error(f"iq rx: {type(e).__name__}: {e}")
                continue
            for f in got:
                self.stats["frames_ok" if f.ok else "frames_bad"] += 1
                self.frames.append(f)
                self._arq_q.append(f)
                self._arq_ev.set()
                if f.ok:
                    self.session._emit(OfdmData(len(f.payload), "", f.payload))
                if self.on_frame:
                    try:
                        self.on_frame(f)
                    except Exception:
                        pass

    # -- TX -----------------------------------------------------------------

    def send(self, data: bytes, seq0: int | None = None, mod: CxModem | None = None,
             keep_seq: bool = False, repeat_s: float = 0.0) -> tuple[int, int]:
        """Queue ``data`` as frames (NUL-padded to whole chunks) at our TX table (``mod``: another
        modem, e.g. ``self.base`` for control).  Returns (seq0, n_frames).  ``keep_seq``: a resend at
        an old ``seq0``, the sequence counter stays where it is."""
        with self._lock:
            mod = mod or self.mod
            s0 = self._seq if seq0 is None else int(seq0) % self.seq_mod
            n = mod.frames_for(len(data))
            if not keep_seq:
                self._seq = (s0 + n) % self.seq_mod
        if self.cfg.gapless:
            q = np.zeros(int(mod.x.c.frame_gap_samples), np.complex128)
            x = np.concatenate([q, mod.encode(data, seq0=s0, gap=False), q]) * self.gain
        else:
            x = mod.encode(data, seq0=s0) * self.gain
        if repeat_s > 0:                      # the same frames again (same seq) after a fade's worth of air
            x = np.concatenate([x, np.zeros(int(repeat_s * self.cfg.fs), np.complex128), x])
            self.stats["tx_repeats"] += n
        self._tx(x)
        self.stats["tx_frames"] += n
        return s0, n

    def send_message(self, data: bytes, seq0: int | None = None, magic: bytes = MSG_MAGIC,
                     mod: CxModem | None = None, repeat_s: float = 0.0) -> tuple[int, int]:
        """One payload with a length header and CRC-32, so ``receive_message`` can size and check it."""
        data = bytes(data)
        return self.send(_MSG_HDR.pack(magic, len(data)) + data + struct.pack(">I", zlib.crc32(data)), seq0, mod,
                         repeat_s=repeat_s)

    def _ctl_repeat(self) -> float:
        return float(getattr(self.cfg, "ctl_repeat_s", 0.0) or 0.0)

    def _late_bat(self, f: CxFrame) -> None:
        """A BAT after negotiate returned: if it says the peer has our table (it is re-sending because
        our READY never arrived), READY again at our TX table, at most once a second."""
        _m, ln = _MSG_HDR.unpack(f.payload[:_MSG_HDR.size])
        body = f.payload[_MSG_HDR.size:_MSG_HDR.size + ln]
        if len(body) < ln or self._ctl_dup(f):
            return                                   # a multi-frame BAT (not ours) or the repeated copy
        try:
            _bat, have, *_rest = self._parse_bat(body)
        except (ValueError, struct.error):
            return
        if not have or time.monotonic() - self._late_ready_t < 1.0:
            return
        self._late_ready_t = time.monotonic()
        self.send_message(b"", magic=READY_MAGIC, repeat_s=self._ctl_repeat())
        self.stats["tx_ready_late"] += 1

    def _ctl_dup(self, f: CxFrame) -> bool:
        """True if this POLL / STAT frame's seq was already acted on (the ctl_repeat_s copy); else mark it."""
        k = (f.payload[:4], f.seq)
        if k in self._ctl_seen:
            self.stats["rx_ctl_dup"] += 1
            return True
        self._ctl_seen.append(k)
        return False

    def send_probe(self) -> int:
        x = self.prober.build() * self.gain
        self._tx(x)
        self._probe_t = time.monotonic()
        self.stats["tx_probes"] += 1
        return len(x)

    def _tx(self, x: np.ndarray) -> None:
        """Every burst goes out here: station ID when due, listen-before-talk, PAPR clip, then the device."""
        cfg = self.cfg
        if getattr(cfg, "callsign", "") and (self._ident_at is None
                                             or time.monotonic() - self._ident_at >= float(cfg.ident_s)):
            self.send_ident()
        if getattr(cfg, "lbt", False):
            self._listen_before_talk()
        clip = float(getattr(cfg, "papr_clip_db", 0.0) or 0.0)
        if clip > 0:
            x = self._papr_clip(x, clip)
        self.session.play_raw(np.asarray(x).astype(np.complex64))

    def send_ident(self) -> int:
        """The station ID now: ``cfg.callsign`` in CW (on/off keyed carrier at ``ident_hz``, 5 ms
        raised-cosine edges, PARIS timing at ``ident_wpm``), at the frames' RMS.  Returns samples."""
        x = cw_iq(str(self.cfg.callsign), float(self.cfg.fs), float(self.cfg.ident_wpm), float(self.cfg.ident_hz),
                  float(self.cfg.tx_rms))
        self._ident_at = time.monotonic()
        if len(x):
            q = np.zeros(int(0.05 * self.cfg.fs), np.complex128)
            self.session.play_raw(np.concatenate([q, x, q]).astype(np.complex64))
            self.stats["tx_ident"] += 1
        return len(x)

    def _listen_before_talk(self) -> None:
        """Wait until the last ~20 ms of RX sit within 6 dB of the noise floor (10th percentile of the
        recent block powers), then a random 0-50 ms backoff; give up after lbt_max_s."""
        end = time.monotonic() + float(self.cfg.lbt_max_s)
        rng = np.random.default_rng()
        while time.monotonic() < end:
            pw = list(self._rx_pow)
            if len(pw) < 10:
                break
            floor = float(np.percentile(pw, 10))
            if pw[-1] <= 4.0 * max(floor, 1e-20):
                time.sleep(float(rng.uniform(0.0, 0.05)))
                pw = list(self._rx_pow)
                if pw[-1] <= 4.0 * max(floor, 1e-20):
                    return
            self.stats["lbt_busy"] += 1
            time.sleep(0.02)
        self.stats["lbt_timeout"] += 1

    def _papr_clip(self, x: np.ndarray, clip_db: float) -> np.ndarray:
        """Clip at clip_db over the burst's RMS, mask the spectrum back to the occupied band; 2 passes."""
        x = np.asarray(x, np.complex128)
        nz = np.abs(x) > 0
        if not nz.any():
            return x
        c = self.base.x.c
        f = np.fft.fftfreq(len(x), 1.0 / float(self.cfg.fs))
        df = float(self.cfg.fs) / c.n_fft
        lo, hi = float(self.cfg.band_lo) - df, float(self.cfg.band_hi) + df
        band = (f >= lo) & (f <= hi)
        rms = float(np.sqrt(np.mean(np.abs(x[nz]) ** 2)))
        lim = rms * 10 ** (clip_db / 20)
        for _ in range(2):
            a = np.abs(x)
            over = a > lim
            if not over.any():
                break
            x = x.copy()
            x[over] *= lim / a[over]
            X = np.fft.fft(x)
            X[~band] = 0
            x = np.fft.ifft(X)
            self.stats["papr_clipped"] += int(over.sum())
        return x

    def wait_tx(self, timeout: float | None = None) -> bool:
        """Until everything queued has gone to the device.  False on timeout, and at once when the
        device died: on iq: the medium's RX is the TX clock, so a dead socket never drains."""
        p = self.session.pipe
        end = None if timeout is None else time.monotonic() + timeout
        while p is not None:
            if p.wait_tx(0.2):
                return True
            if p.rx_finished or p.stats.tx_error:
                return False
            if end is not None and time.monotonic() >= end:
                return False
            p = self.session.pipe
        return False

    # -- probe --------------------------------------------------------------

    def grade_probe(self, window_s: float | None = None, emit: bool = True) -> CxProbeResult:
        """Look for the peer's probe in the last ``window_s`` of RX history (default: as much as
        the prober takes) and grade it.  Call after the probe has fully arrived."""
        pipe = self.session.pipe
        n = int((window_s or self.prober.max_cap / self.cfg.fs) * self.cfg.fs)
        x, _i0 = pipe.history(min(n, self.prober.max_cap))
        if self.exciser is not None:
            x = self.exciser.filter_snippet(x)
        res = self.prober.estimate(x)
        if emit and res.ok:
            c = self.prober.x.c
            bits = np.full(c.n_occupied, -1)
            data_idx = [i for i in range(c.n_occupied) if i % c.pilot_spacing != 0]
            if len(data_idx) == len(res.bat):
                bits[data_idx] = res.bat
            self.session._emit(ProbeReport("measured", int(self.cfg.peer_id), int(self.cfg.node_id),
                                           res.freq_hz, res.snr_db, bits, res.summary()))
        return res

    # -- RX helpers ---------------------------------------------------------

    def receive(self, seq0: int, n_frames: int, timeout: float) -> Reassembler:
        """Wait until frames seq0..seq0+n-1 have all arrived (or timeout); scans ``frames``."""
        r = Reassembler(seq0, n_frames, self.seq_mod)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for f in list(self.frames):       # bounded deque: rescanning is cheap and never misses
                r.add(f)
            if r.complete:
                break
            time.sleep(0.02)
        return r

    def receive_message(self, timeout: float, seq0: int | None = None) -> tuple[bytes | None, str]:
        """Wait for a ``send_message`` payload.  ``seq0`` None: the first ok frame that opens with
        MSG_MAGIC starts it.  Returns (payload, "") or (None, why)."""
        data, why, _f = self.receive_msg(MSG_MAGIC, timeout, seq0=seq0)
        return data, why

    def receive_msg(self, magic: bytes, timeout: float, since: int | None = None,
                    seq0: int | None = None) -> tuple[bytes | None, str, CxFrame | None]:
        """``receive_message`` for any magic, from RX index ``since`` on.  Returns (payload, "",
        first frame) or (None, why, first frame or None); pass ``first.index + 1`` as the next
        ``since`` to get the message after it."""
        end = time.monotonic() + timeout
        first: CxFrame | None = None
        while first is None:
            for f in list(self.frames):
                if (f.ok and f.payload[:4] == magic and (seq0 is None or f.seq == seq0)
                        and (since is None or f.index >= since)):
                    first = f
                    break
            else:
                if time.monotonic() >= end:
                    break
                time.sleep(0.02)
        if first is None:
            return None, "no message header frame", None
        _m, n = _MSG_HDR.unpack(first.payload[:_MSG_HDR.size])
        total = _MSG_HDR.size + n + 4
        nf = self.mod.frames_for(total)
        r = Reassembler(first.seq, nf, self.seq_mod, table=first.table, since=first.index)
        while True:
            for f in list(self.frames):       # bounded deque: rescanning is cheap and never misses
                r.add(f)
            if r.complete or time.monotonic() >= end:
                break
            time.sleep(0.02)
        if not r.complete:
            return None, f"{len(r.chunks)}/{nf} frames", first
        blob = r.data(total)
        data, crc = blob[_MSG_HDR.size:_MSG_HDR.size + n], struct.unpack(">I", blob[-4:])[0]
        if zlib.crc32(data) != crc:
            return None, "CRC mismatch", first
        return data, "", first

    # -- probe -> BAT exchange (step 9) ---------------------------------------

    def rx_now(self) -> int:
        """Absolute RX index just past the newest sample received."""
        x, i0 = self.session.pipe.history(1)
        return int(i0 + len(x))

    def poll_probe(self, since: int = -1) -> tuple[CxProbeResult, int] | None:
        """The peer's probe, if a whole one starting at or after RX index ``since`` is in the
        history now: (result, absolute start).  Cheap enough to call a few times a second."""
        n = self.prober.samples + int(0.5 * self.cfg.fs)
        x, i0 = self.session.pipe.history(min(n, self.prober.max_cap))
        if len(x) < self.prober.samples:
            return None
        if self.exciser is not None:
            x = self.exciser.filter_snippet(x)
        res = self.prober.estimate(x)
        if not res.ok or i0 + res.frame_start < since:
            return None
        return res, int(i0 + res.frame_start)

    def _send_bat(self, bat, have: bool, res: CxProbeResult | None, echo: int = NO_TOKEN,
                  held_s: float = 0.0) -> None:
        snr = float(np.median(res.snr_db)) if res is not None else float("nan")
        cfo = float(res.cfo_hz) if res is not None else float("nan")
        held_ms = int(min(max(held_s, 0.0) * 1e3, 0xFFFFFFFF))
        body = (_BAT_HDR.pack(BAT_VERSION, int(bool(have)), len(bat)) + bytes(int(v) for v in bat)
                + _BAT_TAIL.pack(snr, cfo, echo & 0xFFFF, held_ms))
        self.send_message(body, magic=BAT_MAGIC, mod=self.base, repeat_s=self._ctl_repeat())
        self.stats["tx_bat"] += 1

    def _parse_bat(self, data: bytes):
        """-> bat, have, snr, cfo, echo token (NO_TOKEN: none), held seconds."""
        ver, flags, n = _BAT_HDR.unpack(data[:_BAT_HDR.size])
        nd = self.base.x.c.n_data_carriers
        if ver != BAT_VERSION or n != nd or len(data) != _BAT_HDR.size + n + _BAT_TAIL.size:
            raise ValueError(f"BAT message v{ver} n={n} len={len(data)} (want v{BAT_VERSION} n={nd})")
        bat = list(data[_BAT_HDR.size:_BAT_HDR.size + n])
        if any(b not in (0, 1, 2, 4, 6) or b > int(self.cfg.max_bits) for b in bat):
            raise ValueError("BAT message holds a bit count this profile does not allow")
        if sum(bat) == 0:
            raise ValueError("BAT message holds an empty table")
        snr, cfo, echo, held_ms = _BAT_TAIL.unpack(data[_BAT_HDR.size + n:])
        return bat, bool(flags & 1), snr, cfo, echo, held_ms / 1e3

    def negotiate(self, *args, **kw) -> dict:
        """See ``_negotiate``.  While it runs the background responder leaves BATs alone; once one has
        succeeded, that responder answers a BAT the peer re-sends (it never heard our READY: we had
        already stopped lingering) with another READY, however late."""
        self._negotiating = True
        try:
            r = self._negotiate(*args, **kw)
        finally:
            self._negotiating = False
        self._neg_ok = bool(r.get("ok"))
        return r

    def _negotiate(self, initiator: bool, timeout: float | None = None, retry_s: float = 3.0,
                  linger_s: float | None = None, log: Callable[[str], None] | None = None) -> dict:
        """The probe -> BAT exchange with the peer (module docstring).  Returns a dict: ok,
        settled (the peer's READY was heard too), tx_bat / rx_bat and their bits per symbol, the
        probe result we measured (``probe``), the far end's SNR/CFO of our probe, seconds, message
        counts, ``rtd_s`` (the measured round trip, None if never measured; also kept as
        ``self.rtd_s``) and ``since``: the RX index it started at, where to look for the peer's first
        data (it may follow its READY closely enough to arrive before we return).  ``timeout``
        (default 3 x max_rtd_s + 20 s) is the whole budget.  ``retry_s``: the first retry; it backs
        off 2x up to max_rtd_s until the RTD is measured, then retry = max(retry_s, RTD + 1.5 s).
        ``linger_s`` (default 2 x that retry): after we are done, how long to wait for the peer's
        READY, answering its repeats.  On failure ok is False and the link keeps the base table
        both ways."""
        log = log or (lambda m: None)
        max_rtd = float(self.cfg.max_rtd_s)
        timeout = 3.0 * max_rtd + 20.0 if timeout is None else timeout
        t0 = time.monotonic()
        end = t0 + timeout
        rtd: float | None = None
        rtds: list[float] = []
        backoff = float(retry_s)
        ping_sent: dict[int, float] = {}             # our token -> when queued
        token = 0
        heard_token, heard_at = NO_TOKEN, 0.0        # the peer's newest PING and when we decoded it
        since_ping = None

        def retry_now() -> float:
            return max(retry_s, rtd + 1.5) if rtd is not None else backoff

        def send_ping_probe():
            nonlocal token
            token = (token + 1) % NO_TOKEN
            ping_sent[token] = time.monotonic()
            self.send_message(_PING.pack(token), magic=PING_MAGIC, mod=self.base)
            self.send_probe()
        base_id = table_id(self.base.bat)
        nd_bits = self.base.x.c.n_data_carriers

        def weak(t) -> bool:
            # under one bit per carrier: our probe reached the peer in a fade (vhf-mobile: 3 and 60 bits/sym
            # of 112 carriers).  Keep probing (the peer keeps the best per carrier) and hold READY until
            # half the budget.
            return t is not None and int(sum(t)) < nd_bits
        start = self.rx_now()
        since_probe = since_bat = since_ready = since_ping = start
        probe_gap = self.prober.samples // 2
        rx_bat = tx_bat = None
        res: CxProbeResult | None = None
        far = (float("nan"), float("nan"))
        peer_has = sent_ready = settled = False
        done_at = None
        need_reply = False
        bat_sent = None                              # the table our last BAT carried
        next_act = t0 if initiator else float("inf")
        while True:
            now = time.monotonic()
            linger = (2.0 * retry_now()) if linger_s is None else linger_s
            if now >= end or (done_at is not None and (settled or now >= done_at + linger)):
                break
            # -- what arrived --
            while True:
                data, _why, f = self.receive_msg(PING_MAGIC, 0, since=since_ping)
                if f is None:
                    break
                since_ping = f.index + 1
                if data is not None and len(data) == _PING.size:
                    heard_token, heard_at = _PING.unpack(data)[0], time.monotonic()
                    self.stats["rx_ping"] += 1
            got = self.poll_probe(since_probe)
            if got and sum(got[0].bat) == 0:          # the probe arrived but carries nothing: no table to give
                since_probe = got[1] + probe_gap
                self.stats["rx_probes_empty"] += 1
                log(f"NEG probe gives an empty table, ignored: {got[0].summary()}")
                got = None
            if got:
                res, at = got
                since_probe = at + probe_gap
                prev = rx_bat
                if prev is not None and len(prev) == len(res.bat):
                    # best per carrier over the probes heard: one probe in a fade must not sink the table
                    # (vhf-mobile: a 0.6 dB probe gave 4 bits/sym); ARQ's grading lowers what frames show
                    rx_bat = [max(int(a), int(b)) for a, b in zip(prev, res.bat)]
                else:
                    rx_bat = list(res.bat)
                self.set_rx_tables([rx_bat] + ([prev] if prev is not None and prev != rx_bat else [])
                                   + [self.base.bat])
                self.stats["rx_probes"] += 1
                log(f"NEG heard probe: {res.summary()}"
                    + (f" -> best of probes {sum(rx_bat)} per symbol" if prev is not None else ""))
                need_reply = True
                if done_at is not None and rx_bat != bat_sent:
                    # the peer is still probing: give it the better one.
                    # Against what we SENT, not the previous best: a probe that improved the table in
                    # the pass we went READY never reached the peer otherwise.
                    self._send_bat(rx_bat, True, res, NO_TOKEN, 0.0)
                    bat_sent = list(rx_bat)
                    log(f"NEG sent BAT (after done): {sum(rx_bat)} bits/sym")
                    done_at, settled = time.monotonic(), False
            while True:
                data, _why, f = self.receive_msg(BAT_MAGIC, 0, since=since_bat)
                if f is None:
                    break
                since_bat = f.index + 1
                if data is None:
                    continue
                try:
                    bat, have, snr, cfo, echo, held = self._parse_bat(data)
                except ValueError as e:
                    log(f"NEG bad BAT: {e}")
                    continue
                self.stats["rx_bat"] += 1
                if echo in ping_sent:
                    r = time.monotonic() - ping_sent.pop(echo) - held
                    rtds.append(r)
                    first = rtd is None
                    rtd = min(rtds)                  # the least-delayed answer: queueing only ever adds
                    if first:
                        log(f"NEG RTD {rtd:.2f} s (ping {echo} answered, peer held it {held:.2f} s)")
                if tx_bat != bat:
                    tx_bat = bat
                    self.set_tx_bat(bat)
                    log(f"NEG got BAT: TX {sum(bat)} bits/sym (peer measured SNR {snr:.1f} dB, CFO {cfo:+.1f} Hz)")
                far = (snr, cfo)
                if have:
                    peer_has = True
                    if sent_ready:
                        self.send_message(b"", magic=READY_MAGIC, repeat_s=self._ctl_repeat())  # it missed it
                        self.stats["tx_ready"] += 1
                elif rx_bat is not None:
                    need_reply = True
            while True:
                _d, _why, f = self.receive_msg(READY_MAGIC, 0, since=since_ready)
                if f is None:
                    break
                since_ready = f.index + 1
                peer_has = settled = True
                self.stats["rx_ready"] += 1
            if rx_bat is not None and table_id(rx_bat) != base_id and any(
                    f.ok and f.table == table_id(rx_bat) and f.index >= start for f in list(self.frames)):
                peer_has = settled = True     # the peer already sends at the table we gave it
            # -- what to send --
            if (tx_bat is not None and peer_has and not sent_ready
                    and (not weak(tx_bat) or time.monotonic() >= t0 + (end - t0) / 2)):
                if rx_bat is not None and rx_bat != bat_sent:
                    # a probe this pass changed our table after the peer's BAT came in: send it before
                    # READY (same queue, so it lands first), or the reply branch below never runs again
                    self._send_bat(rx_bat, True, res, NO_TOKEN, 0.0)
                    bat_sent = list(rx_bat)
                    log(f"NEG sent BAT (before READY): {sum(rx_bat)} bits/sym")
                self.send_message(b"", magic=READY_MAGIC, repeat_s=self._ctl_repeat())  # at the new TX table
                self.stats["tx_ready"] += 1
                sent_ready = True
                done_at = time.monotonic()
                log(f"NEG done in {done_at - t0:.1f} s: TX {sum(tx_bat)} bits/sym, "
                    f"RX {sum(rx_bat) if rx_bat else '-'} bits/sym")
                continue
            if done_at is None and (need_reply or time.monotonic() >= next_act):
                timed_out = not need_reply
                if rx_bat is not None:
                    held = time.monotonic() - heard_at if heard_token != NO_TOKEN else 0.0
                    self._send_bat(rx_bat, tx_bat is not None, res, heard_token, held)
                    bat_sent = list(rx_bat)
                if rx_bat is None or tx_bat is None or weak(tx_bat):
                    send_ping_probe()         # the peer has no measurement of us yet
                    log("NEG sent ping + probe" + (" + BAT" if rx_bat is not None else ""))
                else:
                    log("NEG sent BAT")
                need_reply = False
                if timed_out and rtd is None:
                    backoff = min(2.0 * backoff, max(retry_s, max_rtd + 1.5))
                next_act = time.monotonic() + retry_now()
            time.sleep(0.1)
        ok = done_at is not None
        if not ok:
            self.set_tx_bat(None)
            self.set_rx_tables([self.base.bat])
        else:
            self._neg_rx = list(rx_bat) if rx_bat else None
            self._probe_t = time.monotonic()
        self.rtd_s = rtd
        return dict(ok=ok, settled=settled, seconds=time.monotonic() - t0, since=start,
                    rtd_s=rtd, rtd_all_s=[round(r, 3) for r in rtds],
                    tx_bat=list(tx_bat) if ok else None, rx_bat=list(rx_bat) if ok and rx_bat else None,
                    tx_bits=int(sum(tx_bat)) if ok else int(sum(self.base.bat)),
                    rx_bits=int(sum(rx_bat)) if ok and rx_bat else int(sum(self.base.bat)),
                    probe=res, far_snr_db=far[0], far_cfo_hz=far[1],
                    counts={k: int(self.stats[k]) for k in ("tx_probes", "rx_probes", "tx_bat", "rx_bat",
                                                           "tx_ready", "rx_ready", "rx_ping")})

    # -- selective-repeat ARQ (module docstring) --------------------------------

    def _ctrl_air_s(self) -> float:
        return self.base.frame_samples / float(self.cfg.fs)

    def _arq_wait_s(self) -> float:
        """How long a POLL waits for its STAT: the measured round trip (else 3 s) plus both control
        frames' air time and a margin for detection and the peer's own queued TX."""
        rtd = self.rtd_s if self.rtd_s is not None else 3.0
        return rtd + 2.0 * self._ctrl_air_s() + 1.5 + 2.0 * float(getattr(self.cfg, "ctl_repeat_s", 0.0) or 0.0)

    def _send_poll(self, mid: int, s0: int, n: int, table: int) -> None:
        self.send_message(_POLL.pack(mid, s0, n, table & 0xFFFFFFFF), magic=POLL_MAGIC, mod=self.base,
                          repeat_s=self._ctl_repeat())
        self.stats["tx_poll"] += 1

    def send_reliable(self, data: bytes, rounds: int | None = None, log: Callable[[str], None] | None = None,
                      deadline: float | None = None) -> dict:
        """``send_message`` with selective repeat: POLL, resend what the STAT says is missing (at the
        table it sends, if any), until the peer has the whole message or ``rounds`` (default
        ``cfg.arq_rounds``) POLLs went by.  Returns a dict: ok, rounds, resent (frames), frames,
        table_updates, tx_bits (bits per symbol at the end), seconds, why.  ``deadline``
        (time.monotonic()): no POLL after it."""
        log = log or (lambda m: None)
        rounds = int(self.cfg.arq_rounds) if rounds is None else int(rounds)
        t0 = time.monotonic()
        data = bytes(data)
        blob = _MSG_HDR.pack(MSG_MAGIC, len(data)) + data + struct.pack(">I", zlib.crc32(data))
        pb = self.mod.payload_bytes
        s0, n = self.send(blob)
        with self._lock:
            self._msg_id = (self._msg_id + 1) & 0xFFFF
            mid = self._msg_id
        out = dict(ok=False, rounds=0, resent=0, frames=n, table_updates=0, why="", polls_unanswered=0)
        since = self.rx_now()
        # an unanswered POLL (POLL or STAT lost to a fade) costs no round: re-POLL, up to 2 x rounds in
        # a row (hf-moderate's 0.5 Hz fades took 6 of 8 POLLs and failed a link that was converging)
        r = npoll = miss = 0
        while r < max(rounds, 1) and miss < 2 * max(rounds, 1):
            if deadline is not None and time.monotonic() >= deadline:
                out["why"] = f"deadline after {out['rounds']} rounds ({miss} POLLs in a row unanswered)"
                break
            npoll += 1
            out["rounds"] = r + 1
            self._send_poll(mid, s0, n, table_id(self.bat))
            if not self.wait_tx(60.0):
                out["why"] = "TX did not drain"
                break
            st = self._wait_stat(mid, since, self._arq_wait_s())
            if st is None:
                out["polls_unanswered"] += 1
                miss += 1
                log(f"ARQ msg {mid}: no STAT for POLL {npoll}")
                continue
            miss = 0
            r += 1
            since = st["index"] + 1
            if st["complete"]:
                out["ok"] = True
                break
            if st["table"] is not None and list(st["table"]) != list(self.bat):
                self.set_tx_bat(st["table"])
                out["table_updates"] += 1
                log(f"ARQ msg {mid}: peer sent a table: TX {sum(st['table'])} bits/sym")
            if st["probe"]:
                self.send_probe()                  # the peer re-grades our channel from it
                self.stats["tx_reprobe"] += 1
                log(f"ARQ msg {mid}: peer asked for a probe")
            if st["crc_failed"]:
                missing = list(range(n))
            else:
                missing = [k for k in range(n) if not st["have"][k]]
            log(f"ARQ msg {mid}: round {r}: resend {len(missing)}/{n}")
            for a, b in _runs(missing):
                self.send(blob[a * pb:b * pb], seq0=(s0 + a) % self.seq_mod, keep_seq=True)
                out["resent"] += b - a
            if rounds == 0:
                break
        out["seconds"] = round(time.monotonic() - t0, 2)
        out["tx_bits"] = int(sum(self.bat))
        if not out["ok"] and not out["why"]:
            out["why"] = (f"peer still missing frames after {out['rounds']} rounds"
                          + (f" ({miss} POLLs in a row unanswered)" if miss else ""))
        self.stats["arq_ok" if out["ok"] else "arq_fail"] += 1
        return out

    def send_window(self, messages, window: int = 4, rounds: int | None = None,
                    log: Callable[[str], None] | None = None) -> list[dict]:
        """Pipelined selective repeat: up to ``window`` messages in flight, each followed by its POLL,
        without waiting for the STAT of the one before; a STAT for any of them resends that message's
        missing frames (at the table it carries) and re-polls it; a message waits the usual STAT time
        before its POLL is repeated.  Returns one ``send_reliable``-style dict per message, in order."""
        log = log or (lambda m: None)
        rounds = int(self.cfg.arq_rounds) if rounds is None else int(rounds)
        msgs = [bytes(m) for m in messages]
        out = [dict(ok=False, rounds=0, resent=0, frames=0, table_updates=0, why="", polls_unanswered=0)
               for _ in msgs]
        live: dict[int, dict] = {}                    # mid -> state
        nxt, t0 = 0, time.monotonic()
        since = self.rx_now()
        pb = self.mod.payload_bytes

        def start(i: int) -> None:
            data = msgs[i]
            blob = _MSG_HDR.pack(MSG_MAGIC, len(data)) + data + struct.pack(">I", zlib.crc32(data))
            s0, n = self.send(blob)
            with self._lock:
                self._msg_id = (self._msg_id + 1) & 0xFFFF
                mid = self._msg_id
            live[mid] = dict(i=i, blob=blob, s0=s0, n=n, polled=0.0)
            out[i]["frames"] = n
            poll(mid)

        def poll(mid: int) -> None:
            st = live[mid]
            self._send_poll(mid, st["s0"], st["n"], table_id(self.bat))
            st["polled"] = time.monotonic()
            out[st["i"]]["rounds"] += 1

        while nxt < len(msgs) or live:
            while nxt < len(msgs) and len(live) < max(1, window):
                start(nxt)
                nxt += 1
            self.wait_tx(60.0)
            wait = self._arq_wait_s()
            st = self._wait_stat(set(live), since, 0.2)
            now = time.monotonic()
            if st is not None:
                since = st["index"] + 1
                mid = st["mid"]
                m = live.get(mid)
                if m is None:
                    continue
                r = out[m["i"]]
                if st["complete"]:
                    r["ok"] = True
                    del live[mid]
                else:
                    if st["table"] is not None and list(st["table"]) != list(self.bat):
                        self.set_tx_bat(st["table"])
                        r["table_updates"] += 1
                        log(f"ARQ msg {mid}: peer sent a table: TX {sum(st['table'])} bits/sym")
                    missing = list(range(m["n"])) if st["crc_failed"] else [
                        k for k in range(m["n"]) if not st["have"][k]]
                    for a, b in _runs(missing):
                        self.send(m["blob"][a * pb:b * pb], seq0=(m["s0"] + a) % self.seq_mod, keep_seq=True)
                        r["resent"] += b - a
                    if r["rounds"] >= max(rounds, 1):
                        r["why"] = f"peer still missing frames after {r['rounds']} rounds"
                        del live[mid]
                    else:
                        poll(mid)
                if st["probe"]:
                    self.send_probe()
                    self.stats["tx_reprobe"] += 1
                continue
            for mid, m in list(live.items()):         # no STAT in time: poll again
                if now - m["polled"] > wait:
                    r = out[m["i"]]
                    r["polls_unanswered"] += 1
                    if r["rounds"] >= max(rounds, 1):
                        r["why"] = f"no STAT after {r['rounds']} POLLs"
                        del live[mid]
                    else:
                        poll(mid)
        for r in out:
            r["seconds"] = round(time.monotonic() - t0, 2)
            r["tx_bits"] = int(sum(self.bat))
            self.stats["arq_ok" if r["ok"] else "arq_fail"] += 1
        return out

    def _wait_stat(self, mid, since: int, timeout: float) -> dict | None:
        end = time.monotonic() + timeout
        while True:
            data, _why, f = self.receive_msg(STAT_MAGIC, max(0.0, end - time.monotonic()), since=since)
            if f is None:
                return None
            since = f.index + 1
            if data is None or len(data) < _STAT_HDR.size:
                continue
            if self._ctl_dup(f):
                continue
            m, s0, n, flags = _STAT_HDR.unpack(data[:_STAT_HDR.size])
            if (m not in mid) if isinstance(mid, (set, frozenset, dict)) else (m != mid):
                continue
            nb = (n + 7) // 8
            bm = data[_STAT_HDR.size:_STAT_HDR.size + nb]
            have = [bool(bm[k >> 3] & (0x80 >> (k & 7))) if (k >> 3) < len(bm) else False for k in range(n)]
            table = None
            if flags & 2:
                t = list(data[_STAT_HDR.size + nb:])
                if len(t) == self.base.x.c.n_data_carriers and all(v in (0, 1, 2, 4, 6) for v in t) and sum(t) > 0:
                    table = t
            self.stats["rx_stat"] += 1
            return dict(complete=bool(flags & 1), crc_failed=bool(flags & 4), have=have, table=table,
                        index=f.index, probe=bool(flags & 8), mid=m)

    def linger(self, max_s: float = 60.0, quiet_s: float | None = None) -> float:
        """Keep answering POLLs after the last message arrived, until none has come for ``quiet_s``
        (default two STAT waits) or ``max_s`` passed: our last STAT may be lost, and the sender
        re-POLLs until it hears one; closing at once leaves it polling into silence.  Returns the
        seconds spent."""
        t0 = time.monotonic()
        quiet = 2.0 * self._arq_wait_s() if quiet_s is None else float(quiet_s)
        while time.monotonic() - t0 < max_s:
            last = max(getattr(self, "_poll_t", 0.0), t0)
            if time.monotonic() - last >= quiet:
                break
            time.sleep(0.1)
        self.wait_tx(10.0)
        return time.monotonic() - t0

    def receive_reliable(self, timeout: float) -> bytes | None:
        """The next message the ARQ responder delivered (whole, CRC checked), or None on timeout."""
        end = time.monotonic() + timeout
        while True:
            try:
                return self.messages.popleft()
            except IndexError:
                pass
            left = end - time.monotonic()
            if left <= 0:
                return None
            self._msg_ev.wait(min(left, 0.2))
            self._msg_ev.clear()

    def _arq_serve(self) -> None:
        """The responder: keep every data chunk by seq, answer each POLL with a STAT."""
        while not self._stop.is_set():
            self._arq_ev.wait(0.2)
            self._arq_ev.clear()
            while True:
                try:
                    f = self._arq_q.popleft()
                except IndexError:
                    break
                if f.quality is not None and f.qtable and f.qtable != table_id(self.base.bat):
                    self._qual.append((f.qtable, f.quality[0], f.quality[1], f.erased if f.ok else -1))
                    del self._qual[:-64]
                if not f.ok:
                    continue
                magic = f.payload[:4]
                if magic == BAT_MAGIC and self._neg_ok and not self._negotiating:
                    try:
                        self._late_bat(f)
                    except Exception as e:      # pragma: no cover - never kill the responder
                        self.stats["arq_errors"] += 1
                        self.session._emit_raw_error(f"iq arq: {type(e).__name__}: {e}")
                    continue
                if magic == POLL_MAGIC:
                    if self._ctl_dup(f):
                        continue
                    try:
                        self._answer_poll(f)
                    except Exception as e:      # pragma: no cover - never kill the responder
                        self.stats["arq_errors"] += 1
                        self.session._emit_raw_error(f"iq arq: {type(e).__name__}: {e}")
                elif magic not in CONTROL_MAGICS:
                    with self._arq_lock:
                        self._store[f.seq] = (f.payload, f.index)
                        if len(self._store) > 4 * 4096:
                            for k in sorted(self._store, key=lambda k: self._store[k][1])[:4096]:
                                del self._store[k]

    def _answer_poll(self, f: CxFrame) -> None:
        body = f.payload[_MSG_HDR.size:]
        _m, ln = _MSG_HDR.unpack(f.payload[:_MSG_HDR.size])
        if ln != _POLL.size:
            return
        mid, s0, n, table = _POLL.unpack(body[:_POLL.size])
        self.stats["rx_poll"] += 1
        self._poll_t = time.monotonic()
        nb = (n + 7) // 8
        if self._delivered.get(mid) == s0:                         # its last STAT was lost
            self._send_stat(mid, s0, n, 1, b"\xff" * nb)
            return
        seqs = [(s0 + k) % self.seq_mod for k in range(n)]
        with self._arq_lock:
            have = [q in self._store for q in seqs]
        extra = self._probe_flags(table)
        if all(have):
            with self._arq_lock:
                blob = b"".join(self._store[q][0] for q in seqs)
            data, ok = _unwrap(blob)
            if ok:
                with self._arq_lock:
                    for q in seqs:
                        self._store.pop(q, None)
                self._delivered[mid] = s0
                while len(self._delivered) > 64:
                    self._delivered.popitem(last=False)
                self.messages.append(data)
                self._msg_ev.set()
                self.stats["arq_delivered"] += 1
                if self.on_message:
                    try:
                        self.on_message(data)
                    except Exception:
                        pass
                new = self._probe_table(table)
                if new is None:                                    # clean: only an up-shift is worth sending
                    g = self._grade_table(table)
                    cur = next((t for t in self.rx_tables if table_id(t) == table), None)
                    if g is not None and cur is not None and sum(g) > sum(cur) and min(
                            gb - cb for gb, cb in zip(g, cur)) >= 0:
                        new = g
                if new is not None:
                    self._adopt_rx(new)
                    self._send_stat(mid, s0, n, 1 | 2 | extra, b"\xff" * nb + bytes(new))
                else:
                    self._send_stat(mid, s0, n, 1 | extra, b"\xff" * nb)
                return
            with self._arq_lock:                                   # CRC failed: start over
                for q in seqs:
                    self._store.pop(q, None)
            self._send_stat(mid, s0, n, 4, bytes(nb))
            return
        bm = bytearray(nb)
        for k, h in enumerate(have):
            if h:
                bm[k >> 3] |= 0x80 >> (k & 7)
        new = self._probe_table(table) or self._grade_table(table)
        if new is not None:
            self._adopt_rx(new)
            self._send_stat(mid, s0, n, 2 | extra, bytes(bm) + bytes(new))
        else:
            self._send_stat(mid, s0, n, extra, bytes(bm))

    def _adopt_rx(self, new: list[int]) -> None:
        """Try ``new`` first from now on; keep the last few tables (frames in flight) and the base."""
        rx = [new] + [t for t in self.rx_tables if table_id(t) != table_id(new)]
        self.set_rx_tables(rx[:4] + ([self.base.bat] if self.base.bat not in rx[:4] else []))

    def _probe_flags(self, table: int) -> int:
        """8 (ask the sender for a probe) when cfg.reprobe_s has passed since the last one, or a table
        update has left carriers at 0 bits that the negotiated table loaded; remember since when."""
        every = float(getattr(self.cfg, "reprobe_s", 0.0) or 0.0)
        if every <= 0 or self._want_probe is not None:
            return 0
        cur = next((t for t in self.rx_tables if table_id(t) == table), None)
        zeroed = (cur is not None and self._neg_rx is not None
                  and any(b == 0 and nb > 0 for b, nb in zip(cur, self._neg_rx)))
        due = time.monotonic() - self._probe_t >= (every / 4 if zeroed else every)
        if not due:
            return 0
        self._want_probe = self.rx_now()
        self.stats["reprobe_asked"] += 1
        return 8

    def _probe_table(self, table: int) -> list[int] | None:
        """The table from the probe we asked for, once it has arrived (None before, or if it brings
        nothing new).  Lets carriers come back and bits go up."""
        if self._want_probe is None:
            return None
        got = self.poll_probe(self._want_probe)
        if got is None:
            if self.rx_now() - self._want_probe > 4 * self.prober.samples + 10 * self.cfg.fs:
                self._want_probe = None                            # it never came: ask again later
            return None
        res, _at = got
        self._want_probe = None
        self._probe_t = time.monotonic()
        self.stats["reprobe_got"] += 1
        if not res.ok or sum(res.bat) == 0:
            return None
        new = [int(b) for b in res.bat]
        cur = next((t for t in self.rx_tables if table_id(t) == table), None)
        if cur is not None and len(cur) == len(new):
            # the probe only brings back carriers that are at 0: a probe that lands in a fade would hand
            # out a near-empty table (a 0.5 Hz HF fade took 355 bits/sym to 5), and one at a fade peak
            # raises every carrier for grading to pull down again (205 -> 384 -> 88 on hf-moderate).
            # Lowering, and raising loaded carriers one step, is _grade_table's job, from frames.
            new = [int(a) if int(a) > 0 else int(b) for a, b in zip(cur, new)]
        self._neg_rx = new
        return None if cur is not None and list(cur) == new else new

    def _send_stat(self, mid: int, s0: int, n: int, flags: int, rest: bytes) -> None:
        self.send_message(_STAT_HDR.pack(mid, s0, n, flags) + rest, magic=STAT_MAGIC, mod=self.base,
                          repeat_s=self._ctl_repeat())
        self.stats["tx_stat"] += 1

    def _grade_table(self, table: int) -> list[int] | None:
        """A new table for the peer's TX from the frames we demodulated at ``table`` (its current one),
        or None if it stays.  Dead carriers (|H|^2 DEAD_HREL under the mean) go to 0; the rest to what
        their measured SNR carries above the floors + ARQ_MARGIN_DB; a loaded carrier goes ONE step up
        when its SNR clears the next floor by 3 dB more (0-bit carriers come back by re-probe only)."""
        cur = next((t for t in self.rx_tables if table_id(t) == table), None)
        q = [(sc, hr) for qt, sc, hr, _e in self._qual if qt == table]
        if cur is None or not q:
            return None
        sc = np.median([a for a, _ in q], axis=0)    # median: one frame lost to a burst must not
        hr = np.median([b for _, b in q], axis=0)    # demote every carrier
        if os.environ.get("IQ_ARQ_DEBUG"):
            print("ARQ grade", len(q), "frames; dead", [i for i, h in enumerate(hr) if 0 <= h < DEAD_HREL],
                  "score>0.5", [i for i, r in enumerate(sc) if r > 0.5], "median", float(np.median(sc[sc >= 0])),
                  file=sys.stderr)
        margin = ARQ_MARGIN_DB
        new = []
        for b, r, h in zip(cur, sc, hr):
            b = int(b)
            if b == 0:
                new.append(0)
                continue
            if h < DEAD_HREL:
                new.append(0)
                continue
            snr_db = 10 * np.log10(1.0 / max(r * _DMIN2[b] / 4.0, 1e-6))
            if r > 0.5:                       # decisions too often wrong for the estimate: two steps down
                steps = [w for w in (6, 4, 2, 1, 0) if w < b]
                new.append(steps[1] if len(steps) > 1 else 0)
                continue
            # one step up when the data itself shows the room for it (+3 dB hysteresis), else down to fit
            up = [w for w in (1, 2, 4, 6) if w > b and w <= int(self.cfg.max_bits)][:1]
            if up and snr_db >= _FLOOR_DB[up[0]] + margin + 3.0 and r < 0.1:
                new.append(up[0])
                continue
            # down to fit, but one step per update: under a fade the median of a few frames swings, and
            # the next update can always take another step
            fit = [w for w in (6, 4, 2, 1) if w <= b and snr_db >= _FLOOR_DB[w] + margin]
            one = [w for w in (6, 4, 2, 1, 0) if w < b][:1]
            new.append(max(fit[0] if fit else 0, one[0] if one else 0))
        if new == [int(b) for b in cur] or sum(new) == 0:
            return None
        # Floor: never below 1 bit per data carrier on average.  Frames graded inside a deep fade
        # walked hf-moderate's table 882 -> 726 -> 444 -> 84 -> less: each step makes frames longer (84
        # bits/sym = 1.25 s), so more of each frame sits in the next fade.  Below the floor the table
        # stays and ARQ resends at it; a re-probe (asked for while carriers are at 0) can still restore.
        if sum(new) < len(new) and sum(new) < sum(int(b) for b in cur):
            return None
        self._qual = [e for e in self._qual if e[0] != table]   # grade the next table on its own frames
        return new


def cw_iq(text: str, fs: float, wpm: float, offset_hz: float, rms: float) -> np.ndarray:
    """Morse as an on/off keyed complex carrier at ``offset_hz`` (station ID on an I/Q link): PARIS
    timing (dot = 1.2 / wpm s), 5 ms raised-cosine edges as ident.cw_audio, carrier amplitude ``rms``."""
    from .ident import MORSE
    dot = int(round(1.2 / max(wpm, 1.0) * fs))
    ramp = min(int(0.005 * fs), dot // 2)
    edge = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / max(ramp, 1))
    env: list[np.ndarray] = []

    def key(n: int) -> np.ndarray:
        e = np.ones(n)
        e[:ramp], e[n - ramp:] = edge, edge[::-1]
        return e

    for w, word in enumerate(str(text).upper().split()):
        if w:
            env.append(np.zeros(7 * dot))
        for ci, ch in enumerate(word):
            code = MORSE.get(ch)
            if not code:
                continue
            if ci:
                env.append(np.zeros(3 * dot))
            for si, sym in enumerate(code):
                if si:
                    env.append(np.zeros(dot))
                env.append(key(dot if sym == "." else 3 * dot))
    if not env:
        return np.zeros(0, np.complex128)
    e = np.concatenate(env)
    return rms * e * np.exp(2j * np.pi * offset_hz * np.arange(len(e)) / fs)


def _runs(idx: list[int]) -> list[tuple[int, int]]:
    """[1, 2, 3, 7] -> [(1, 4), (7, 8)]: contiguous half-open runs."""
    out: list[tuple[int, int]] = []
    for k in idx:
        if out and out[-1][1] == k:
            out[-1] = (out[-1][0], k + 1)
        else:
            out.append((k, k + 1))
    return out


def _unwrap(blob: bytes) -> tuple[bytes, bool]:
    """send_message framing -> (payload, CRC ok)."""
    if len(blob) < _MSG_HDR.size + 4:
        return b"", False
    m, n = _MSG_HDR.unpack(blob[:_MSG_HDR.size])
    if m != MSG_MAGIC or _MSG_HDR.size + n + 4 > len(blob):
        return b"", False
    data = blob[_MSG_HDR.size:_MSG_HDR.size + n]
    crc = struct.unpack(">I", blob[_MSG_HDR.size + n:_MSG_HDR.size + n + 4])[0]
    return data, zlib.crc32(data) == crc
