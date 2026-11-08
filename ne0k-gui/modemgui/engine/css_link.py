# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""CSS text link: encode/decode with the reference CSS modem, TX and burst RX.

The modem is the audiotestserver reference (css_encoder / css_decoder from tests/CSS).
``set_backend("c")`` (CssConfig.backend) switches its C cores on, process-wide:
libcssenc for the frame codec both ways (CSS_ENC_BACKEND), libcssmf for the
matched-filter decoder (CSS_MF_BACKEND) and librs255 (RS_BACKEND).  The decoder
itself only enters the matched filter above symbol_dur 0.25 s; CssConfig.decoder
"mf" forces it at any length.  What actually ran is read back from the encoder /
decoder output (``last_codec``) and logged, so the log never claims C on a
Python run.
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

BACKEND = "python"                 # what set_backend last selected
BACKENDS = ("python", "c")
_C_ENV = ("CSS_ENC_BACKEND", "CSS_MF_BACKEND", "RS_BACKEND")
last_codec: dict = {}              # {"enc": c|python, "path": mf|pll, "mf": c|numpy|None, "dec": c|python}
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


def set_backend(name: str) -> str:
    """Select the CSS DSP cores for this process.  "c" = libcssenc + libcssmf + librs255, and fails
    loudly when one is missing or built for another CPU instead of quietly grading numpy; "python" =
    numpy/reedsolo (RS_BACKEND is left alone: OFDM shares it).  Returns the description to log."""
    global BACKEND
    _imports()
    if name not in BACKENDS:
        raise ValueError(f"css backend {name!r} not in {BACKENDS}")
    if name == "python":
        os.environ.pop("CSS_ENC_BACKEND", None)
        os.environ.pop("CSS_MF_BACKEND", None)
        BACKEND = "python"
        return "python (numpy)"
    import cbackend
    libs = []
    for key in ("css_enc", "css_mf", "rs255"):
        rec = cbackend.BACKENDS[key].probe()
        if not rec["ok"]:
            raise RuntimeError(f"css backend c: {rec['reason']}")
        libs.append(Path(rec["path"]).name + (" (STALE)" if rec.get("stale") else ""))
    for var in _C_ENV:
        os.environ[var] = "c"
    BACKEND = "c"
    return "c (" + ", ".join(libs) + ")"


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
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        enc.encode(_enc_cfg(c, raw, _work / tag))
    last_codec["enc"] = "c" if "ENC_BACKEND=c" in out.getvalue() else "python"
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
    ov = {"mf_longsym": True, "mf_longsym_min_dur": 0.0} if c.decoder == "mf" else None
    with _decode_lock, _no_signals():
        res = tl.decode_array(cap, c.fs, _info(c), ov)
    log = res.get("_log") or ""
    mf = "MF_SEGMENTS" in log or "MF_BACKEND=" in log
    last_codec.update(path="mf" if mf else "pll",
                      mf=("c" if "MF_BACKEND=c" in log else "numpy") if mf else None,
                      dec="c" if "DEC_BACKEND=c" in log else "python")
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
        self.backend = ""                # set_backend's description, once applied
        self._codec_logged: dict = {}
        self.stream = None               # CssStreamRx when decoder == "pll_c"
        self._tx_onsets: collections.deque = collections.deque()
        self._pipe_hooked = None

    def _ensure_backend(self) -> None:
        if self.backend:
            return
        c = self.cfg
        self.backend = set_backend(c.backend)
        self.log(f"css dsp backend: {self.backend} decoder={c.decoder} symbol_dur={c.symbol_dur}s")
        if c.backend == "c" and c.decoder == "auto" and c.symbol_dur <= 0.25:
            self.log(f"css WARNING: symbol_dur {c.symbol_dur}s <= 0.25 s decodes on the streaming PLL, "
                     "so libcssmf is not used (libcssenc still is); set css.decoder=mf to run the "
                     "matched filter in C")

    def _log_codec(self, keys) -> None:
        """Log what the reference modem reported it ran, on first use and whenever it changes."""
        now = {k: last_codec.get(k) for k in keys}
        if any(now[k] != self._codec_logged.get(k) for k in keys):
            self._codec_logged.update(now)
            self.log("css codec: " + " ".join(f"{k}={v}" for k, v in now.items()))

    def channel_busy(self, settle_s: float = 0.4) -> bool:
        """Carrier sense: a burst is being received, or the gate was open within ``settle_s``."""
        if self.stream is not None and self.stream.busy():
            return True
        return self._in_burst or time.monotonic() - self._last_busy < settle_s

    def start(self) -> None:
        pipe = self.session.pipe
        if not pipe:
            raise RuntimeError("session not open")
        self._ensure_backend()
        # Same dispatch as the reference decode(): the streaming PLL up to
        # mf_longsym_min_dur (0.25 s); above it the burst path runs decode(),
        # which hands long symbols to the 12-segment matched-filter decoder --
        # on the recorded mic/speaker challenge captures (0.5-1.5 s, TV,
        # parakeets, impulses) that decodes 7/7 where the PLL got 2/7.
        if self.cfg.decoder == "pll_c" and self.cfg.symbol_dur <= 0.25:
            self._start_stream(pipe)
        elif self.cfg.decoder == "pll_c":
            self.log(f"css: symbol_dur {self.cfg.symbol_dur}s > 0.25 s: decoding bursts on the "
                     "matched-filter long-symbol decoder (reference dispatch), not the streaming PLL")
        self._stop.clear()
        self._thread = threading.Thread(target=self._rx, args=(pipe.subscribe(1024),),
                                        name="css-rx", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(5.0)
            self._thread = None
        if self._pipe_hooked is not None:
            try:
                self._pipe_hooked.tx_onset_hooks.remove(self._on_tx_onset)
            except ValueError:
                pass
            self._pipe_hooked = None

    def _start_stream(self, pipe) -> None:
        """decoder "pll_c": continuous RX on the C streaming PLL decoder + chirp tracker."""
        from .css_stream import CssStreamRx
        _, cm, _ = _imports()
        info = json.loads(Path(_info(self.cfg)).read_text())
        me = self.cfg.node_id
        self.stream = CssStreamRx(
            self.cfg, info, node_id=me, hear_own=self.cfg.hear_own, log=self.log,
            emit=self.session._emit,
            emit_link=lambda kind, peer, t0, t1, d: self.session._emit(LinkEvent(kind, me, peer, t0, t1, d)),
            decoded_cls=Decoded, unpack=cm.unpack_packet)
        pipe.tx_onset_hooks.append(self._on_tx_onset)
        self._pipe_hooked = pipe
        last_codec.update(path="pll_c", mf=None, dec="c")

    def _on_tx_onset(self, rx_index: int, samples) -> None:
        # sink thread: just queue it; the RX thread reads the chirp schedule off it
        self._tx_onsets.append((rx_index, samples))

    def _play(self, pcm: np.ndarray, peer: int, detail: str) -> float:
        self._log_codec(("enc",))
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
        self._ensure_backend()
        if seq is None:
            seq, self._seq = self._seq, (self._seq + 1) % 16
        self.last_seq = seq
        return self._play(encode_bytes(self.cfg, payload, seq, dst),
                          self.cfg.peer_id if dst is None else dst, f"DATA seq={seq} {len(payload)}B")

    def send_ack(self, dst: int, seq: int) -> float:
        self._ensure_backend()
        return self._play(encode_ack(self.cfg, dst, seq), dst, f"ACK seq={seq}")

    def send_type(self, dst: int, mtype: int, seq: int, payload: bytes = b"",
                  detail: str = "") -> float:
        self._ensure_backend()
        return self._play(encode_type(self.cfg, dst, mtype, seq, payload), dst,
                          detail or f"type={mtype} seq={seq}")

    def ack_airtime(self) -> float:
        self._ensure_backend()
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
            st = self.stream
            if st is not None:
                while self._tx_onsets:
                    idx, pcm = self._tx_onsets.popleft()
                    try:
                        st.tx_onset(idx, pcm)
                    except Exception as e:
                        self.log(f"css rx: tx schedule failed: {e}")
            if b is None:
                continue
            if st is not None:
                st.feed(b.samples, b.tx_active)
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
            if self.stream is not None:
                self.stream.burst_end()
            return
        if self.stream is not None:
            # frames came off the streaming decoder as they arrived; the burst
            # boundary only finishes the packet and re-arms the decoder
            self.stream.burst_end()
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
        if last_codec.get("path"):
            self._log_codec(("path", "mf", "dec"))
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
