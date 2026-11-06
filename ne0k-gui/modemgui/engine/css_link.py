# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""CSS text link: encode/decode with the reference CSS modem, TX and burst RX.

Backend is the Python reference (css_encoder / css_decoder from tests/CSS); the
C libs are not yet on this path, and ``BACKEND`` says so for the UI.
"""
from __future__ import annotations

import collections
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

from .events import Decoded, LinkEvent
from .modemcfg import CssConfig
from .ofdm_link import KEYUP_LEAD_S

BACKEND = "python"
_TESTS = Path(os.environ.get("MODEMGUI_TESTS_DIR",
                             Path(__file__).resolve().parents[3] / "audiotestserver" / "tests"))
_work = Path(tempfile.mkdtemp(prefix="modemgui_css_"))


def _imports():
    for p in (_TESTS, _TESTS / "CSS"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.environ.setdefault("AUDIOTOOL_RUN_DIR", str(_work))   # testlib scratch
    import css_encoder, css_modem        # noqa: E401
    import testlib
    return css_encoder, css_modem, testlib


def _enc_cfg(c: CssConfig, raw: bytes, base: Path) -> dict:
    return dict(fs=c.fs, f_min=c.f_min, f_max=c.f_max, symbol_dur=c.symbol_dur,
                amplitude=c.amplitude, symbol_gap_frac=c.symbol_gap_frac,
                chunk_size=c.chunk_size, diff_encode=c.diff_encode, rs_t=c.rs_t,
                conv_code=(None if c.conv_code in ('', 'none') else c.conv_code),
                interleave_rows=c.interleave_rows, message_bytes=bytes(raw),
                output=str(base) + ".wav", info_out=str(base) + "_i.json",
                result_out=str(base) + "_e.json")


def encode_packet(c: CssConfig, raw: bytes, tag: str = "tx") -> np.ndarray:
    enc, _, _ = _imports()
    with contextlib.redirect_stdout(io.StringIO()):
        enc.encode(_enc_cfg(c, raw, _work / tag))
    with wave.open(str(_work / tag) + ".wav", "rb") as w:
        return (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                .astype(np.float32) / 32768.0)


def encode_bytes(c: CssConfig, payload: bytes, seq: int = 0, dst: int | None = None) -> np.ndarray:
    _, cm, _ = _imports()
    raw = cm.pack_packet(c.node_id, c.peer_id if dst is None else dst, cm.TYPE_DATA, seq % 16,
                         payload)
    return encode_packet(c, raw)


def encode_ack(c: CssConfig, dst: int, seq: int) -> np.ndarray:
    """Header-only ACK to ``dst`` echoing the acknowledged DATA seq (as css_modem_node does)."""
    _, cm, _ = _imports()
    return encode_packet(c, cm.pack_packet(c.node_id, dst, cm.TYPE_ACK, seq % 16), "ack")


def encode_type(c: CssConfig, dst: int, mtype: int, seq: int, payload: bytes = b"",
                tag: str = "ctl") -> np.ndarray:
    """Any css_modem packet type (REQ, BEACON, ...) from this node to ``dst``."""
    _, cm, _ = _imports()
    return encode_packet(c, cm.pack_packet(c.node_id, dst, mtype, seq % 16, payload), tag)


def encode_text(c: CssConfig, text: str, seq: int = 0) -> np.ndarray:
    return encode_bytes(c, text.encode("utf-8"), seq)


def _info(c: CssConfig, max_frames: int = 256) -> str:
    """Frame-geometry info.json the decoder needs, from a throwaway encode."""
    _, cm, _ = _imports()
    encode_packet(c, cm.pack_packet(0, 0, 0, 0), "dec")
    p = _work / "dec_i.json"
    info = json.loads(p.read_text())
    info["frames"] = max_frames
    p.write_text(json.dumps(info))
    return str(p)


_decode_lock = threading.Lock()     # decoder swaps sys.modules; one at a time


@contextlib.contextmanager
def _no_signals():
    """css_decoder.decode() installs a SIGTERM handler, which only works on the
    main thread; our RX runs on a worker, so make that call a no-op there."""
    import signal
    if threading.current_thread() is threading.main_thread():
        yield
        return
    real = signal.signal
    signal.signal = lambda *a, **k: None
    try:
        yield
    finally:
        signal.signal = real


def decode_samples(c: CssConfig, pcm: np.ndarray) -> list[Decoded]:
    """Decode one captured burst.  Returns [] when nothing was recovered."""
    _, cm, tl = _imports()
    pad = np.zeros(c.fs, np.float32)
    cap = np.concatenate([pad, np.asarray(pcm, np.float32), pad])
    with _decode_lock, _no_signals():
        res = tl.decode_array(cap, c.fs, _info(c))
    frames = res.get("frames") or []
    if not frames:
        return []
    raw = b"".join(bytes.fromhex(f["bytes_hex"]) for f in frames)
    try:
        pk = cm.unpack_packet(raw)
    except cm.ModemError:
        return []
    errs = int(res.get("frames_errored") or 0)
    snr = float(np.mean([f.get("snr_db", 0.0) for f in frames]))
    text = pk["payload"].rstrip(b"\x00").decode("utf-8", "replace")
    diag = res.get("diagnostics") or {}
    acq = None
    if diag.get("sync_at_symbol") is not None:        # symbols counted from the 1 s lead-in pad
        acq = max(0.0, diag["sync_at_symbol"] * c.symbol_dur * (1 + c.symbol_gap_frac) - 1.0)
    return [Decoded(text=text, src=pk["src"], dst=pk["dst"], seq=pk["seq"],
                    type_name=pk["type_name"], ok=(errs == 0), frames=len(frames),
                    frames_errored=errs, snr_db=snr,
                    rs_errors=sum(int(f.get("rs_errors") or 0) for f in frames),
                    payload=pk["payload"], acq_s=acq,
                    sync_score=float(diag.get("sync_score") or 0.0),
                    grid_sps=frames[0].get("grid_sps"))]


class CssLink:
    """TX + burst RX on a Session's pipe.  Carrier sense is a 50 ms RMS gate; a
    burst ends after ``hang_s`` of quiet and is decoded on this thread."""

    def __init__(self, session, cfg: CssConfig | None = None, log=lambda m: None) -> None:
        self.session, self.cfg, self.log = session, cfg or CssConfig(), log
        self._seq = 0
        self.last_seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_busy = -1e9
        self._in_burst = False
        self.on_level = None             # f(rms) per 50 ms block: feeds the ambient estimate

    def channel_busy(self, settle_s: float = 0.4) -> bool:
        """Carrier sense: a burst is being received, or the gate was open within ``settle_s``."""
        return self._in_burst or time.monotonic() - self._last_busy < settle_s

    def start(self) -> None:
        pipe = self.session.pipe
        if not pipe:
            raise RuntimeError("session not open")
        self._stop.clear()
        self._thread = threading.Thread(target=self._rx, args=(pipe.subscribe(1024),),
                                        name="css-rx", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(5.0)
            self._thread = None

    def _play(self, pcm: np.ndarray, peer: int, detail: str) -> float:
        air = len(pcm) / self.cfg.fs
        now = time.monotonic()
        self.session._emit(LinkEvent("tx", self.cfg.node_id, peer, now, now + air, detail))
        self.log(f"css tx to {peer}: {detail} airtime {air:.1f}s")
        self.session.play(pcm)
        return air

    def send(self, text: str) -> float:
        """Encode and queue ``text``; returns the airtime in seconds."""
        return self.send_bytes(text.encode("utf-8"))

    def send_bytes(self, payload: bytes, seq: int | None = None, dst: int | None = None) -> float:
        """Send raw bytes (the serial bridge frames its own length).  ``seq`` repeats an
        earlier sequence number for a retransmission; None takes the next one."""
        if seq is None:
            seq, self._seq = self._seq, (self._seq + 1) % 16
        self.last_seq = seq
        return self._play(encode_bytes(self.cfg, payload, seq, dst),
                          self.cfg.peer_id if dst is None else dst, f"DATA seq={seq} {len(payload)}B")

    def send_ack(self, dst: int, seq: int) -> float:
        return self._play(encode_ack(self.cfg, dst, seq), dst, f"ACK seq={seq}")

    def send_type(self, dst: int, mtype: int, seq: int, payload: bytes = b"",
                  detail: str = "") -> float:
        return self._play(encode_type(self.cfg, dst, mtype, seq, payload), dst,
                          detail or f"type={mtype} seq={seq}")

    def ack_airtime(self) -> float:
        return len(encode_ack(self.cfg, self.cfg.peer_id, 0)) / self.cfg.fs

    def _rx(self, tap) -> None:
        c = self.cfg
        win = int(0.05 * c.fs)
        thr = 10 ** (c.carrier_db / 20)
        buf: list[np.ndarray] = []
        quiet = 0
        pre: collections.deque = collections.deque(maxlen=4)   # pre-roll: the gate opens late
        rem = np.zeros(0, np.float32)
        while not self._stop.is_set():
            b = tap.get(timeout=0.2)
            if b is None:
                continue
            if b.tx_active and not c.hear_own:
                continue
            x = np.concatenate([rem, b.samples])
            n = len(x) // win * win
            rem = x[n:]
            for seg in x[:n].reshape(-1, win):
                lvl = float(np.sqrt(np.mean(seg * seg)))
                busy = lvl > thr
                if busy:
                    self._last_busy = time.monotonic()
                elif not buf and self.on_level:
                    self.on_level(lvl)
                if busy and not buf:
                    buf.extend(pre); pre.clear()
                self._in_burst = bool(busy or buf)
                if busy or buf:
                    buf.append(seg)
                    quiet = 0 if busy else quiet + win
                    if quiet >= c.hang_s * c.fs:
                        # keep the trailing quiet: the gate closes before the last
                        # symbols' tail, and a truncated final frame fails to decode
                        burst = np.concatenate(buf)
                        t1 = time.monotonic() - quiet / c.fs
                        t0 = t1 - (len(burst) - quiet) / c.fs
                        buf.clear(); quiet = 0
                        self._decode(burst, t0, t1)
                else:
                    pre.append(seg)

    def _decode(self, burst: np.ndarray, t0: float, t1: float) -> None:
        me = self.cfg.node_id
        # OFDM and CSS never intermix: this listener keeps running under ofdm_only (CssLink.start
        # is unconditional), so it hears every OFDM burst too -- the same keyup()-tone-dominance
        # problem as ofdm_link._is_ident applies here (see note_heard's tail_from).
        tail_from = int(KEYUP_LEAD_S * self.cfg.fs) if len(burst) / self.cfg.fs > KEYUP_LEAD_S else None
        if self.session.ident.note_heard(burst, self.cfg.fs, lambda d: self.session._emit(
                LinkEvent("rx", me, -1, t0, t1, d)), tail_from=tail_from):
            return
        out: list[Decoded] = []
        # The reference decoder reads in symbol-sized chunks and misses the preamble when the
        # onset lands in one phase of that grid (about 20-45 ms of each 50 ms symbol).  The gate
        # puts the onset anywhere, so re-try with the burst shifted by fractions of a symbol.
        sym = self.cfg.symbol_dur
        for shift in (0, 0.5, 0.25, 0.75):
            try:
                out = decode_samples(self.cfg, np.concatenate(
                    [np.zeros(int(shift * sym * self.cfg.fs), np.float32), burst]))
            except Exception as e:               # decoder must never kill RX
                self.session._emit_raw_error(f"css decode: {e}")
                break
            if out and out[0].ok:
                break
        if not out:
            self.log(f"css rx burst {t1 - t0:.1f}s: no frames decoded")
            self.session._emit(LinkEvent("err", me, -1, t0, t1, "no frames decoded"))
        for d in out:
            self.log(f"css rx src={d.src} seq={d.seq} {d.type_name} ok={d.ok} snr={d.snr_db:.1f}dB "
                     f"frames={d.frames}/{d.frames_errored}err rs={d.rs_errors} {len(d.payload)}B")
            self.session._emit(LinkEvent("lock", me, d.src, t0, t0, f"{d.frames} frames"))
            if d.ok:
                self.session._emit(LinkEvent("rx", me, d.src, t0, t1,
                                             f"{d.type_name} seq={d.seq} snr={d.snr_db:.1f}dB"))
            else:
                self.session._emit(LinkEvent("err", me, d.src, t0, t1,
                                             f"{d.frames_errored} frame(s) errored"))
            self.session._emit(d)
