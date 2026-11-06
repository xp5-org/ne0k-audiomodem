# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""One end of the full-duplex OFDM link, running in C.

ctypes wrapper over the C89 cores in ``tests/OFDM/c``: ``ofdm_dx`` (link-frame codec),
``ofdm_rxf`` (own-band rejection), ``ofdm_rxs`` (burst gate + decode) and ``ofdm_arq``
(stop-and-wait with piggybacked ACK).  Everything between "a mic block arrives" and "a
payload is delivered / an audio frame is due" happens in C; this module only moves samples
and turns C structs into Python values.

It replaces the Python data plane of ``OfdmLink`` mode ``full`` when ``cfg.backend == "cdx"``.
The handshake that produces the bit tables is still the reference one.

Thread-safety: the capture thread feeds, the link thread polls; one lock serialises every
call into C because the state is shared.
"""
from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass

import numpy as np

from .. import native

D, I = ctypes.c_double, ctypes.c_int
MAX_TAPS = 1025


class _Params(ctypes.Structure):
    _fields_ = [("tx_cfg", native.OfdmCfg), ("rx_cfg", native.OfdmCfg),
                ("tx_bat", ctypes.POINTER(I)), ("rx_bat", ctypes.POINTER(I))]


class _RxsParams(ctypes.Structure):
    _fields_ = [("fs", I), ("thr_rms", D), ("hang_s", D), ("ring_samples", ctypes.c_long),
                ("taps", ctypes.POINTER(D)), ("ntaps", I)]


class _Evt(ctypes.Structure):
    _fields_ = [("kind", I), ("a", ctypes.c_long), ("b", ctypes.c_long), ("seq", I), ("ack", I),
                ("len", I), ("n_chunks", I), ("n_found", I), ("n_bad", I), ("errata", I),
                ("evm", D), ("data", ctypes.c_ubyte * 255)]


class _Act(ctypes.Structure):
    _fields_ = [("kind", I), ("seq", I), ("ack", I), ("len", I), ("ok", I), ("retry", I),
                ("data", ctypes.POINTER(ctypes.c_ubyte))]


ACT_TX, ACT_DELIVER, ACT_DONE = 1, 2, 3


@dataclass
class RxEvent:
    ok: bool                      # a link frame decoded (False: a burst that did not)
    a: int                        # burst span in the FILTERED stream (see delay)
    b: int
    seq: int = 0
    ack: int = 0
    data: bytes = b""
    chunks: int = 0
    found: int = 0
    bad: int = 0
    errata: int = 0
    evm: float = 0.0


@dataclass
class Action:
    kind: str                     # "tx" | "deliver" | "done"
    seq: int = 0
    ack: int = 0
    data: bytes = b""
    retry: int = 0
    ok: bool = False


def _lib() -> ctypes.CDLL:
    lib = native.load("libofdm")
    if getattr(lib, "_cdx_ready", False):
        return lib
    P = ctypes.POINTER
    vp = ctypes.c_void_p
    lib.ofdm_cfg_build.argtypes = [P(native.OfdmCfg), I, D, D, D, I, I, I, I, I, D, D, I]
    lib.ofdm_dx_workspace.argtypes = [P(_Params)]
    lib.ofdm_dx_workspace.restype = ctypes.c_size_t
    lib.ofdm_dx_init.argtypes = [P(vp), vp, ctypes.c_size_t, P(_Params)]
    lib.ofdm_dx_capacity.argtypes = [vp, I]
    lib.ofdm_dx_tx_samples.argtypes = [vp, I]
    lib.ofdm_dx_tx_samples.restype = ctypes.c_long
    lib.ofdm_dx_tx_frame.argtypes = [vp, I, I, ctypes.c_char_p, I, P(D), ctypes.c_long]
    lib.ofdm_dx_tx_frame.restype = ctypes.c_long
    lib.ofdm_rxf_design.argtypes = [P(D), I, I, D, D, D, D, D, P(D)]
    lib.ofdm_rxs_workspace.argtypes = [P(_RxsParams)]
    lib.ofdm_rxs_workspace.restype = ctypes.c_size_t
    lib.ofdm_rxs_init.argtypes = [P(vp), vp, ctypes.c_size_t, vp, P(_RxsParams)]
    lib.ofdm_rxs_feed.argtypes = [vp, P(D), ctypes.c_long]
    lib.ofdm_rxs_pop.argtypes = [vp, P(_Evt)]
    lib.ofdm_rxs_busy.argtypes = [vp]
    lib.ofdm_rxs_delay.argtypes = [vp]
    lib.ofdm_arq_sizeof.restype = ctypes.c_size_t
    lib.ofdm_arq_init.argtypes = [vp, I, D, D]
    lib.ofdm_arq_configure.argtypes = [vp, I, D, D]
    lib.ofdm_arq_send.argtypes = [vp, ctypes.c_char_p, I]
    lib.ofdm_arq_on_frame.argtypes = [vp, I, I, ctypes.c_char_p, I, D]
    lib.ofdm_arq_tx_done.argtypes = [vp, D]
    lib.ofdm_arq_air_busy.argtypes = [vp, I, D]
    lib.ofdm_arq_poll.argtypes = [vp, D, P(_Act)]
    lib.ofdm_arq_busy.argtypes = [vp]
    lib._cdx_ready = True
    return lib


def _build_cfg(lib, d: dict) -> native.OfdmCfg:
    c = native.OfdmCfg()
    rc = lib.ofdm_cfg_build(ctypes.byref(c), int(d["n_fft"]), float(d["cp_ratio"]),
                            float(d["freq_lo_hz"]), float(d["freq_hi_hz"]), int(d["pilot_spacing"]),
                            int(d["sample_rate"]), int(d["ecc_bytes"]), int(d["chunk_bytes"]),
                            int(d["seq_bytes"]), float(d["taper_ms"]), float(d["frame_gap_ms"]),
                            int(d["sync_period"]))
    if rc != 0:
        raise native.NativeError(f"ofdm_cfg_build rc={rc} for {d.get('freq_lo_hz')}-{d.get('freq_hi_hz')} Hz")
    return c


class CDuplexEnd:
    """tx_cfg / rx_cfg are the Python config dicts of the two sub-bands (``hs_data_cfg``)."""

    def __init__(self, fs: int, tx_cfg: dict, rx_cfg: dict, tx_bat, rx_bat, *, thr_rms: float,
                 hang_s: float, retries: int, ack_wait_s: float, gap_s: float,
                 stop_db: float = 60.0, ring_s: float = 12.0) -> None:
        self.lib = lib = _lib()
        self.fs = fs
        self._lock = threading.RLock()
        self._txb = (I * len(tx_bat))(*[int(b) for b in tx_bat])
        self._rxb = (I * len(rx_bat))(*[int(b) for b in rx_bat])
        p = _Params()
        p.tx_cfg, p.rx_cfg = _build_cfg(lib, tx_cfg), _build_cfg(lib, rx_cfg)
        p.tx_bat = ctypes.cast(self._txb, ctypes.POINTER(I))
        p.rx_bat = ctypes.cast(self._rxb, ctypes.POINTER(I))
        need = lib.ofdm_dx_workspace(ctypes.byref(p))
        if not need:
            raise native.NativeError("ofdm_dx rejected the sub-band geometry or bit tables")
        self._dx_mem = ctypes.create_string_buffer(need)
        self._dx = ctypes.c_void_p()
        rc = lib.ofdm_dx_init(ctypes.byref(self._dx), self._dx_mem, need, ctypes.byref(p))
        if rc:
            raise native.NativeError(f"ofdm_dx_init rc={rc}")

        # own-band rejection: pass the far band, stop ours (one-sided, like the reference)
        taps = (D * MAX_TAPS)()
        got = D()
        nt = lib.ofdm_rxf_design(taps, MAX_TAPS, fs, float(rx_cfg["freq_lo_hz"]),
                                 float(rx_cfg["freq_hi_hz"]), float(tx_cfg["freq_lo_hz"]),
                                 float(tx_cfg["freq_hi_hz"]), stop_db, ctypes.byref(got))
        if nt < 0:
            raise native.NativeError("ofdm_rxf_design failed")
        self.filter_taps, self.filter_stop_db = max(nt, 0), (got.value if nt > 0 else 0.0)
        self._taps = taps
        rp = _RxsParams(fs, thr_rms, hang_s, int(ring_s * fs),
                        ctypes.cast(taps, ctypes.POINTER(D)) if nt > 0 else None, max(nt, 0))
        need = lib.ofdm_rxs_workspace(ctypes.byref(rp))
        if not need:
            raise native.NativeError("ofdm_rxs rejected its parameters")
        self._rxs_mem = ctypes.create_string_buffer(need)
        self._rxs = ctypes.c_void_p()
        rc = lib.ofdm_rxs_init(ctypes.byref(self._rxs), self._rxs_mem, need, self._dx,
                               ctypes.byref(rp))
        if rc:
            raise native.NativeError(f"ofdm_rxs_init rc={rc}")
        self.delay = lib.ofdm_rxs_delay(self._rxs)

        self._arq = ctypes.create_string_buffer(lib.ofdm_arq_sizeof())
        lib.ofdm_arq_init(self._arq, int(retries), float(ack_wait_s), float(gap_s))
        self.hang_s = hang_s

    # -- geometry -----------------------------------------------------------

    def capacity(self, window_s: float) -> int:
        """Largest payload whose frame airs within ``window_s`` (one link frame, 255 B max)."""
        best = 0
        with self._lock:
            for k in range(1, 65):
                n = self.lib.ofdm_dx_capacity(self._dx, k)
                if n < 1:
                    break
                if self.lib.ofdm_dx_tx_samples(self._dx, min(n, 255)) / self.fs > window_s:
                    break
                best = n
                if n >= 255:
                    break
        return best

    # -- receive ------------------------------------------------------------

    def feed(self, x: np.ndarray, now: float) -> list[RxEvent]:
        """Push mic samples.  Frames go straight to the ARQ, except those ``accept`` vetoes."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        out: list[RxEvent] = []
        with self._lock:
            self.lib.ofdm_rxs_feed(self._rxs, x.ctypes.data_as(ctypes.POINTER(D)), len(x))
            ev = _Evt()
            while self.lib.ofdm_rxs_pop(self._rxs, ctypes.byref(ev)):
                out.append(RxEvent(ev.kind == 1, ev.a, ev.b, ev.seq, ev.ack, bytes(ev.data[:ev.len]),
                                   ev.n_chunks, ev.n_found, ev.n_bad, ev.errata, ev.evm))
        return out

    def on_frame(self, e: RxEvent, now: float) -> bool:
        """Hand a decoded frame to the ARQ.  True if it carried new in-order data."""
        with self._lock:
            return self.lib.ofdm_arq_on_frame(self._arq, e.seq, e.ack, e.data, len(e.data), now) == 1

    def air_busy(self, now: float) -> bool:
        with self._lock:
            busy = bool(self.lib.ofdm_rxs_busy(self._rxs))
            self.lib.ofdm_arq_air_busy(self._arq, int(busy), now)
        return busy

    # -- send ---------------------------------------------------------------

    def configure(self, retries: int, ack_wait_s: float, gap_s: float) -> None:
        """Change the ARQ timing without disturbing sequence numbers or a frame in flight."""
        with self._lock:
            self.lib.ofdm_arq_configure(self._arq, int(retries), float(ack_wait_s), float(gap_s))

    def send(self, data: bytes) -> int:
        """0 queued, -2 a frame is already in flight."""
        with self._lock:
            return self.lib.ofdm_arq_send(self._arq, bytes(data), len(data))

    def busy(self) -> bool:
        with self._lock:
            return bool(self.lib.ofdm_arq_busy(self._arq))

    def tx_audio(self, seq: int, ack: int, data: bytes) -> np.ndarray:
        """The link frame as raw modulator audio (levelling is the caller's)."""
        with self._lock:
            n = self.lib.ofdm_dx_tx_samples(self._dx, len(data))
            if n <= 0:
                raise native.NativeError(f"frame of {len(data)} B does not fit")
            buf = (D * n)()
            got = self.lib.ofdm_dx_tx_frame(self._dx, seq, ack, bytes(data), len(data), buf, n)
            if got != n:
                raise native.NativeError(f"ofdm_dx_tx_frame rc={got}")
        return np.frombuffer(buf, dtype=np.float64).copy()

    def tx_done(self, now: float) -> None:
        with self._lock:
            self.lib.ofdm_arq_tx_done(self._arq, now)

    def poll(self, now: float) -> list[Action]:
        out: list[Action] = []
        act = _Act()
        with self._lock:
            while self.lib.ofdm_arq_poll(self._arq, now, ctypes.byref(act)):
                data = bytes(act.data[:act.len]) if act.len and act.data else b""
                if act.kind == ACT_TX:
                    out.append(Action("tx", act.seq, act.ack, data, act.retry))
                elif act.kind == ACT_DELIVER:
                    out.append(Action("deliver", act.seq, data=data))
                elif act.kind == ACT_DONE:
                    out.append(Action("done", act.seq, ok=bool(act.ok)))
                if act.kind == ACT_TX:
                    break                       # one transmission at a time; wait for tx_done
        return out
