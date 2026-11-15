# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Complex-baseband (I/Q) OFDM, in C: ctypes over libofdm's ofdm_cx* cores.

    CxModem      ofdm_cxmod: bytes -> frames of complex samples (SC + CE + data + SYNCs)
    CxReceiver   ofdm_cxsync + ofdm_cxdem over a STREAM: feed blocks, get decoded chunks back
    CxProber     ofdm_cxprobe: build the channel probe; grade a capture -> per-carrier SNR,
                 IQ image, and the bit table (ofdm_allocate)

Every one is built from an ``IqConfig`` (modemcfg; the ``iq`` profile is 200 ksps, +-50 kHz).
Both ends must use the same config and the same bit table: the probe measures, the caller agrees
the table (step 9's exchange), then ``CxModem(cfg, bat)`` / ``CxReceiver(cfg, bat)``.

The cores are graded against numpy and truth by audiotestserver/tests/OFDM/ofdm_cx_selftest.py;
this module only carries samples across.  Samples here are numpy complex (I + jQ), one value per
complex sample; the C side wants interleaved doubles, which complex128 already is in memory.

Not thread-safe: one object per thread (each core's scratch lives in its workspace).
"""
from __future__ import annotations

import collections
import ctypes
import dataclasses

import numpy as np

from .. import native

D, I, VP = ctypes.c_double, ctypes.c_int, ctypes.c_void_p
P = ctypes.POINTER


class CxCfg(ctypes.Structure):
    _fields_ = [("c", native.OfdmCfg), ("dc_null", I), ("dc_slot", I)]


class CxHit(ctypes.Structure):
    _fields_ = [("start", I), ("score", D), ("metric", D), ("cfo", D), ("cfo_frac", D),
                ("cfo_int", I), ("int_score", D)]


class CxDemOut(ctypes.Structure):
    _fields_ = [("evm", D), ("data_syms", I), ("errata", I), ("seq", ctypes.c_uint),
                ("erased", I)]          # bytes decoded as erasures (GMD); absent in a pre-2026-11-14 lib


class CxProbeInfo(ctypes.Structure):
    _fields_ = [("valid", I), ("error", I), ("coherence", D), ("noise_floor", D), ("acq_start", I),
                ("frame_start", I), ("cfo", D), ("sc_score", D), ("int_score", D), ("n_syms", I),
                ("n_meas_syms", I), ("has_isi_penalty", I), ("isi_penalty_db", D), ("has_image", I),
                ("image_db", D)]


PROBE_ERRORS = {1: "too short", 2: "no coherent block (sync fail)", 3: "bad geometry",
                4: "no SC preamble + CE in front of the coherent block (tone / DC / cut off)"}

_NEED = ("ofdm_cxcfg_build", "ofdm_cx_bin", "ofdm_cxmod_init", "ofdm_cxmod_encode_ex",
         "ofdm_cxsync_init", "ofdm_cxsync_find", "ofdm_cxdem_init", "ofdm_cxdem_frame",
         "ofdm_cxprobe_init", "ofdm_cxprobe_estimate", "ofdm_allocate")


def _lib() -> ctypes.CDLL:
    lib = native.load("libofdm")
    if getattr(lib, "_cx_ready", False):
        return lib
    missing = [f for f in _NEED if not hasattr(lib, f)]
    if missing:
        raise native.NativeError(f"libofdm lacks {', '.join(missing)} (stale build: run make -C ne0k-gui)")
    cfgp = P(CxCfg)
    lib.ofdm_cxcfg_build.argtypes = [cfgp, I, D, D, D, I, I, I, I, I, D, D, I, I]
    lib.ofdm_cx_bin.argtypes = [cfgp, I]
    lib.ofdm_cxmod_workspace.restype = ctypes.c_size_t
    lib.ofdm_cxmod_workspace.argtypes = [cfgp, VP]
    lib.ofdm_cxmod_init.argtypes = [P(VP), VP, ctypes.c_size_t, cfgp, VP]
    for f in ("ofdm_cxmod_body_syms", "ofdm_cxmod_data_syms", "ofdm_cxmod_payload_bytes"):
        getattr(lib, f).argtypes = [VP]
    lib.ofdm_cxmod_frame_samples.restype = ctypes.c_long
    lib.ofdm_cxmod_frame_samples.argtypes = [VP]
    lib.ofdm_cxmod_signal_samples_ex.restype = ctypes.c_long
    lib.ofdm_cxmod_signal_samples_ex.argtypes = [VP, I, I]
    lib.ofdm_cxmod_encode_ex.restype = ctypes.c_long
    lib.ofdm_cxmod_encode_ex.argtypes = [VP, ctypes.c_char_p, I, ctypes.c_ulong, I, VP, ctypes.c_long]
    lib.ofdm_cxmod_sc_preamble.restype = P(D)
    lib.ofdm_cxmod_sc_preamble.argtypes = [VP]
    lib.ofdm_frame_min_gap.argtypes = [P(native.OfdmCfg), I]
    lib.ofdm_cxsync_workspace.restype = ctypes.c_size_t
    lib.ofdm_cxsync_workspace.argtypes = [cfgp]
    lib.ofdm_cxsync_init.argtypes = [P(VP), VP, ctypes.c_size_t, cfgp, VP, I]
    lib.ofdm_cxsync_find.argtypes = [VP, VP, I, I, D, D, P(CxHit), I]
    lib.ofdm_cxdem_workspace.restype = ctypes.c_size_t
    lib.ofdm_cxdem_workspace.argtypes = [cfgp, VP]
    lib.ofdm_cxdem_init.argtypes = [P(VP), VP, ctypes.c_size_t, cfgp, VP]
    for f in ("ofdm_cxdem_body_syms", "ofdm_cxdem_payload_bytes"):
        getattr(lib, f).argtypes = [VP]
    lib.ofdm_cxdem_frame_samples.restype = ctypes.c_long
    lib.ofdm_cxdem_frame_samples.argtypes = [VP]
    lib.ofdm_cxdem_frame.argtypes = [VP, VP, I, I, D, ctypes.c_char_p, I, P(CxDemOut)]
    if hasattr(lib, "ofdm_cxdem_set_erasures"):
        lib.ofdm_cxdem_set_erasures.argtypes = [VP, I]
        lib.ofdm_cxdem_set_erasures.restype = None
    if hasattr(lib, "ofdm_cxdem_carrier_quality"):
        lib.ofdm_cxdem_carrier_quality.argtypes = [VP, VP, VP]
    if hasattr(lib, "ofdm_cxmod_init_fec"):           # 2026-11-14: inner FEC, tracking, window
        lib.ofdm_cxmod_workspace_fec.restype = ctypes.c_size_t
        lib.ofdm_cxmod_workspace_fec.argtypes = [cfgp, VP, I]
        lib.ofdm_cxmod_init_fec.argtypes = [P(VP), VP, ctypes.c_size_t, cfgp, VP, I]
        lib.ofdm_cxdem_workspace_fec.restype = ctypes.c_size_t
        lib.ofdm_cxdem_workspace_fec.argtypes = [cfgp, VP, I]
        lib.ofdm_cxdem_init_fec.argtypes = [P(VP), VP, ctypes.c_size_t, cfgp, VP, I]
        lib.ofdm_cxdem_set_tracking.argtypes = [VP, I]
        lib.ofdm_cxdem_set_tracking.restype = None
        lib.ofdm_cxdem_set_window.argtypes = [VP, I]
        lib.ofdm_cxdem_pilot_fallbacks.argtypes = [VP]
    if hasattr(lib, "ofdm_cxdem_set_iterate"):        # 2026-11-15: decoder-aided re-estimation
        lib.ofdm_cxdem_set_iterate.argtypes = [VP, I, I]
        lib.ofdm_cxdem_set_iterate.restype = None
        lib.ofdm_cxdem_iter_passes.argtypes = [VP]
    lib.ofdm_cxprobe_workspace.restype = ctypes.c_size_t
    lib.ofdm_cxprobe_workspace.argtypes = [cfgp, I, I, I, I]
    lib.ofdm_cxprobe_init.argtypes = [P(VP), VP, ctypes.c_size_t, cfgp, I, I, I, I]
    lib.ofdm_cxprobe_build_samples.restype = ctypes.c_long
    lib.ofdm_cxprobe_build_samples.argtypes = [VP]
    lib.ofdm_cxprobe_build.restype = ctypes.c_long
    lib.ofdm_cxprobe_build.argtypes = [VP, VP, ctypes.c_long]
    lib.ofdm_cxprobe_estimate.argtypes = [VP, VP, I, D] + [VP] * 6 + [P(CxProbeInfo)]
    lib.ofdm_allocate.argtypes = [VP, I, I, D, I, D, VP, I]
    lib._cx_ready = True
    return lib


def build_cfg(cfg, lib: ctypes.CDLL | None = None) -> CxCfg:
    """``IqConfig`` -> ofdm_cxcfg (ofdm_cxcfg_build)."""
    lib = lib or _lib()
    x = CxCfg()
    rc = lib.ofdm_cxcfg_build(ctypes.byref(x), int(cfg.n_fft), float(cfg.cp_ratio), float(cfg.band_lo),
                              float(cfg.band_hi), int(cfg.pilot_spacing), int(cfg.fs), int(cfg.ecc_bytes),
                              int(cfg.chunk_bytes), int(cfg.seq_bytes), float(cfg.taper_ms),
                              float(cfg.frame_gap_ms), int(cfg.sync_period), int(bool(cfg.dc_null)))
    if rc:
        raise native.NativeError(f"ofdm_cxcfg_build rc={rc} for {cfg}")
    c = x.c
    if c.taper_samples * 4 > c.cp_len:
        # cxdem windows at taper + CP/2: past CP/4 the window's ISI margin is gone (COMPLEX_BB.md).
        raise native.NativeError(f"taper {c.taper_samples} samples > CP/4 ({c.cp_len / 4:g}): no ISI margin")
    return x


def bins(x: CxCfg, lib: ctypes.CDLL | None = None) -> np.ndarray:
    """Signed FFT bin of every occupied carrier, occupied order."""
    lib = lib or _lib()
    return np.array([lib.ofdm_cx_bin(ctypes.byref(x), i) for i in range(x.c.n_occupied)], int)


FEC_CODES = {"none": 0, "1/2": 1, "2/3": 2, "3/4": 3}
# Coding gain the bit loading may spend (dB off snr_margin_db): K=7 soft Viterbi is ~5 dB at BER 1e-5 for
# rate 1/2; taken conservatively, and less for the punctured rates.
FEC_GAIN_DB = {"none": 0.0, "1/2": 4.0, "2/3": 3.0, "3/4": 2.5}


def fec_code(cfg) -> int:
    f = str(getattr(cfg, "fec", "none"))
    if f not in FEC_CODES:
        raise ValueError(f"IqConfig.fec {f!r}: want one of {sorted(FEC_CODES)}")
    return FEC_CODES[f]


def window_offset(cfg, x: CxCfg) -> int:
    """cxdem window offset for IqConfig.window: mid (-1, the default) or late (cp_len - 2: the
    longest echo, delay <= cp_len - 2 - 2*taper, at the cost of timing margin)."""
    w = str(getattr(cfg, "window", "mid"))
    if w == "late":
        return max(0, int(x.c.cp_len) - 2)
    return -1


def default_bat(cfg, x: CxCfg) -> list[int]:
    return [int(cfg.default_bits)] * x.c.n_data_carriers


def _ints(v, n: int, what: str):
    v = [int(b) for b in v]
    if len(v) != n:
        raise native.NativeError(f"{what} has {len(v)} entries, config has {n} data carriers")
    return (I * n)(*v)


def _iq(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, np.complex128).reshape(-1))


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(VP)


def rms_scale(x: np.ndarray, rms: float) -> np.ndarray:
    """Scale a burst to complex RMS ``rms`` over its non-silent samples (gaps don't count).
    Not for the link: see ``tx_gain``."""
    p = np.abs(x) ** 2
    on = p > 0
    if not on.any():
        return x.astype(np.complex64)
    return (x * (rms / np.sqrt(p[on].mean()))).astype(np.complex64)


def tx_gain(cfg) -> float:
    """The ONE gain every burst of a config is sent at: puts a unit-power data frame at
    ``cfg.tx_rms``.  The probe must go out at the same gain, never RMS-matched on its own: its
    known carriers are 1/sqrt(2) and ofdm_cxprobe reports the SNR of a unit-power constellation,
    so a probe scaled up to the frames' RMS reads 3 dB better than the frames will see."""
    m = CxModem(cfg)
    x = m.encode(bytes(m.payload_bytes), gap=False)
    return float(cfg.tx_rms / np.sqrt(np.mean(np.abs(x) ** 2)))


class CxModem:
    """ofdm_cxmod for one IqConfig and bit table (default: ``default_bits`` everywhere)."""

    def __init__(self, cfg, bat=None) -> None:
        self.lib = lib = _lib()
        self.cfg = cfg
        self.x = build_cfg(cfg, lib)
        self.bat = list(bat) if bat is not None else default_bat(cfg, self.x)
        self._bat = _ints(self.bat, self.x.c.n_data_carriers, "bat")
        self.fec = fec_code(cfg)
        if self.fec and not hasattr(lib, "ofdm_cxmod_init_fec"):
            raise native.NativeError("IqConfig.fec needs a libofdm with ofdm_cxmod_init_fec (rebuild)")
        if hasattr(lib, "ofdm_cxmod_init_fec"):
            need = lib.ofdm_cxmod_workspace_fec(ctypes.byref(self.x), self._bat, self.fec)
        else:
            need = lib.ofdm_cxmod_workspace(ctypes.byref(self.x), self._bat)
        if not need:
            raise native.NativeError("ofdm_cxmod_workspace rejected the geometry or bit table")
        self._mem = ctypes.create_string_buffer(need)
        self._h = VP()
        if hasattr(lib, "ofdm_cxmod_init_fec"):
            rc = lib.ofdm_cxmod_init_fec(ctypes.byref(self._h), self._mem, need, ctypes.byref(self.x), self._bat,
                                         self.fec)
        else:
            rc = lib.ofdm_cxmod_init(ctypes.byref(self._h), self._mem, need, ctypes.byref(self.x), self._bat)
        if rc:
            raise native.NativeError(f"ofdm_cxmod_init rc={rc}")

    @property
    def body_syms(self) -> int:
        return self.lib.ofdm_cxmod_body_syms(self._h)

    @property
    def frame_samples(self) -> int:
        return int(self.lib.ofdm_cxmod_frame_samples(self._h))

    @property
    def payload_bytes(self) -> int:
        """User bytes per frame (chunk_bytes; the sequence number and RS parity ride on top)."""
        return self.x.c.chunk_bytes

    def sc_preamble(self) -> np.ndarray:
        """The SC preamble the receiver correlates against (interleaved doubles, as C wants)."""
        p = self.lib.ofdm_cxmod_sc_preamble(self._h)
        n = 2 * self.x.c.symbol_samples
        return np.ctypeslib.as_array(p, shape=(n,)).copy()

    def encode(self, data: bytes, seq0: int = 0, gap: bool = True) -> np.ndarray:
        """Unscaled complex samples: data NUL-padded to whole frames numbered seq0, seq0+1, ..."""
        data = bytes(data)
        n = self.lib.ofdm_cxmod_signal_samples_ex(self._h, len(data), int(bool(gap)))
        if n < 0:
            raise native.NativeError(f"ofdm_cxmod_signal_samples_ex rc={n}")
        out = np.zeros(n, np.complex128)
        got = self.lib.ofdm_cxmod_encode_ex(self._h, data, len(data), int(seq0) & 0xFFFFFFFF,
                                            int(bool(gap)), _ptr(out), n)
        if got != n:
            raise native.NativeError(f"ofdm_cxmod_encode_ex rc={got} (expected {n} samples)")
        return out

    def frames_for(self, nbytes: int) -> int:
        return max(1, -(-int(nbytes) // self.payload_bytes))


def table_id(bat) -> int:
    """A short name for a bit table (CRC-32 of its bytes): which table decoded a frame."""
    import zlib
    return zlib.crc32(bytes(int(v) for v in bat))


@dataclasses.dataclass(frozen=True)
class CxFrame:
    index: int          # absolute RX sample index of the frame start (SC prefix)
    seq: int
    payload: bytes      # the chunk, chunk_bytes long (NUL-padded on the last frame); b"" when not ok
    ok: bool            # RS decoded
    evm: float
    errata: int
    cfo: float          # subcarriers
    cfo_hz: float
    score: float        # SC matched-filter NCC
    table: int = 0      # table_id() of the bit table that decoded it (0 when not ok)
    bits: int = 0       # bits per symbol of that table
    erased: int = 0     # RS bytes decoded as erasures (0: the errors-only decode was enough)
    quality: object = None  # (score, hrel) per data carrier of the table it was demodulated with
                            # (ofdm_cxdem_carrier_quality), also when RS failed; None from an old lib
    qtable: int = 0         # table_id() of that table (== table when ok)


class _Dem:
    """One ofdm_cxdem for one bit table."""

    def __init__(self, lib, cfg, bat) -> None:
        mod = CxModem(cfg, bat)
        self.mod, self.bat = mod, mod.bat
        self.id, self.bits = table_id(self.bat), int(sum(self.bat))
        self._bat = _ints(self.bat, mod.x.c.n_data_carriers, "bat")
        self.x = mod.x
        fec = mod.fec
        if hasattr(lib, "ofdm_cxdem_init_fec"):
            need = lib.ofdm_cxdem_workspace_fec(ctypes.byref(self.x), self._bat, fec)
        else:
            need = lib.ofdm_cxdem_workspace(ctypes.byref(self.x), self._bat)
        self._mem = ctypes.create_string_buffer(need)
        self.h = VP()
        if hasattr(lib, "ofdm_cxdem_init_fec"):
            rc = lib.ofdm_cxdem_init_fec(ctypes.byref(self.h), self._mem, need, ctypes.byref(self.x), self._bat, fec)
        else:
            rc = lib.ofdm_cxdem_init(ctypes.byref(self.h), self._mem, need, ctypes.byref(self.x), self._bat)
        if rc:
            raise native.NativeError(f"ofdm_cxdem_init rc={rc}")
        self.erasures = bool(getattr(cfg, "erasures", True)) and hasattr(lib, "ofdm_cxdem_set_erasures")
        if hasattr(lib, "ofdm_cxdem_set_erasures"):
            lib.ofdm_cxdem_set_erasures(self.h, int(self.erasures))
        if hasattr(lib, "ofdm_cxdem_set_tracking"):
            lib.ofdm_cxdem_set_tracking(self.h, 3 if getattr(cfg, "track", True) else 0)
            if lib.ofdm_cxdem_set_window(self.h, window_offset(cfg, self.x)) < 0:
                raise native.NativeError("ofdm_cxdem_set_window refused the window")
        self.iterate = int(getattr(cfg, "iterate", 0) or 0) if fec else 0
        if self.iterate and not hasattr(lib, "ofdm_cxdem_set_iterate"):
            raise native.NativeError("IqConfig.iterate needs a libofdm with ofdm_cxdem_set_iterate (rebuild)")
        if hasattr(lib, "ofdm_cxdem_set_iterate"):
            lib.ofdm_cxdem_set_iterate(self.h, self.iterate, int(getattr(cfg, "iterate_win", 2) or 2))
        self.body_syms = mod.body_syms
        self.frame_samples = int(lib.ofdm_cxdem_frame_samples(self.h))


class CxReceiver:
    """Streaming frame receiver: ``feed()`` RX blocks as they arrive, get decoded frames back.

    Keeps a rolling buffer at least one frame plus two symbols long (ofdm_cxsync_find does not
    report a frame whose CE runs past the buffer, and cxdem needs the whole frame), searches it
    every ``hop_s`` of new samples, demodulates every complete hit and drops what is behind it.
    A frame is reported once: hits within the sync's min_gap of a decoded one are skipped.

    Several bit tables at once (``set_tables``): the preamble, CE and SYNC do not depend on the
    table, so one sync serves them all and each hit is demodulated with the table that last
    worked first, then the others, until RS takes it.  That is how a link hears the base-table
    control messages and the probed-table data in the same stream (iq_link's BAT exchange).
    A wrong table fails RS (and the sequence number) rather than decoding to garbage."""

    def __init__(self, cfg, bat=None, hop_s: float = 0.05, tables=None) -> None:
        self.lib = lib = _lib()
        self.cfg = cfg
        self.hop_s = hop_s
        mod = CxModem(cfg, bat)               # the SC reference and the frame geometry
        self.x = mod.x
        self._sc = mod.sc_preamble()
        need = lib.ofdm_cxsync_workspace(ctypes.byref(self.x))
        self._smem = ctypes.create_string_buffer(need)
        self._sync = VP()
        rc = lib.ofdm_cxsync_init(ctypes.byref(self._sync), self._smem, need, ctypes.byref(self.x),
                                  _ptr(self._sc), int(cfg.max_cfo_bins))
        if rc:
            raise native.NativeError(f"ofdm_cxsync_init rc={rc}")
        c = self.x.c
        self.payload_bytes = mod.payload_bytes
        self._chunk = ctypes.create_string_buffer(max(c.chunk_bytes, 4080))
        self._hits = (CxHit * 64)()
        self.searches = 0
        self._ctl: _Dem | None = None
        self.set_tables([mod.bat] + [list(t) for t in (tables or [])])
        self.reset()

    def set_ctl(self, modem: "CxModem | None") -> None:
        """A control-frame modem (its own chunk size, e.g. iq_link's short base-table frames) that
        is tried on every hit besides ``tables``.  It is not one of ``tables``; its frames report
        ``table_id`` of its bat like any other."""
        if modem is None:
            self._ctl = None
        elif (self._ctl is None or self._ctl.id != table_id(modem.bat)
              or self._ctl.x.c.chunk_bytes != modem.x.c.chunk_bytes):
            self._ctl = _Dem(self.lib, modem.cfg, modem.bat)
        self.set_tables(self.tables)

    def set_tables(self, tables) -> None:
        """The bit tables to try, first = the primary (``bat``).  Duplicates are dropped.  Keeps
        the buffer: a frame already in it is demodulated with the new set."""
        lib, c = self.lib, self.x.c
        old = {d.id: d for d in getattr(self, "_dems", []) if d is not self._ctl}
        dems, seen = [], set()
        for t in tables:
            i = table_id(t)
            if i not in seen:
                seen.add(i)
                dems.append(old.get(i) or _Dem(lib, self.cfg, t))
        self._ntab = len(dems)
        if self._ctl is not None:
            dems.append(self._ctl)
        self._dems = dems
        self._pref = 0
        self.bat = dems[0].bat
        self.body_syms = min(d.body_syms for d in dems)
        self.frame_samples = dems[0].frame_samples
        self._fmin = min(d.frame_samples for d in dems)
        self._fmax = max(d.frame_samples for d in dems)
        self.min_gap = int(lib.ofdm_frame_min_gap(ctypes.byref(c), self.body_syms))
        # Retained / needed before a search: one SHORTEST frame.  A hit whose longer tables do not fit yet
        # holds the buffer itself (_search); waiting for the longest table before searching at all made a
        # receiver holding one thin ARQ table (84 bits/sym = 1.25 s frames, 8 = 13 s) deaf to short POLLs.
        self.keep = self._fmin + 2 * c.symbol_samples + 2 * c.taper_samples
        self.hop = max(c.symbol_samples, int(self.hop_s * c.sample_rate))

    @property
    def tables(self) -> list[list[int]]:
        return [d.bat for d in self._dems[:self._ntab]]

    def reset(self, index: int = 0) -> None:
        self._buf = np.zeros(0, np.complex128)
        self._i0, self._pending, self._last = index, 0, -10 ** 18
        # the held hit (one waiting for longer tables): its abs index, tables tried, first failed attempt
        self._tried_at, self._tried, self._held_bad = None, set(), None

    def feed(self, x: np.ndarray, index: int | None = None) -> list[CxFrame]:
        """Add RX samples (``index`` = absolute index of x[0]; a gap restarts the buffer)."""
        x = np.asarray(x, np.complex128).reshape(-1)
        if index is not None and index != self._i0 + len(self._buf):
            self.reset(index)
        self._buf = np.concatenate([self._buf, x]) if len(self._buf) else x.copy()
        self._pending += len(x)
        if self._pending < self.hop or len(self._buf) < self.keep:
            return []
        self._pending = 0
        return self._search()

    def flush(self) -> list[CxFrame]:
        """Search whatever is buffered now (end of a capture)."""
        self._pending = 0
        return self._search(final=True) if len(self._buf) >= self._fmin else []

    def _report(self, h, a: int, got) -> CxFrame:
        """The CxFrame for hit ``h`` (absolute index ``a``) from a demod attempt (i or None, dem, info)."""
        lib, c = self.lib, self.x.c
        i, d, info = got
        ok = i is not None
        q = None
        if hasattr(lib, "ofdm_cxdem_carrier_quality"):   # each table has its own cxdem state
            sc, hr = np.zeros(c.n_data_carriers), np.zeros(c.n_data_carriers)
            lib.ofdm_cxdem_carrier_quality(d.h, _ptr(sc), _ptr(hr))
            q = (sc, hr)
        raw = bytes(self._chunk.raw[:d.x.c.chunk_bytes]) if ok else b""
        return CxFrame(index=a, seq=int(info.seq) if ok else -1, payload=raw,
                       ok=ok, evm=float(info.evm), errata=int(info.errata) if ok else -1,
                       cfo=float(h.cfo), cfo_hz=float(h.cfo) * c.sc_spacing_hz, score=float(h.score),
                       table=d.id if ok else 0, bits=d.bits if ok else 0,
                       erased=int(info.erased) if ok else 0, quality=q, qtable=d.id)

    def _search(self, final: bool = False) -> list[CxFrame]:
        lib, c = self.lib, self.x.c
        buf = np.ascontiguousarray(self._buf)
        n = len(buf)
        self.searches += 1
        k = lib.ofdm_cxsync_find(self._sync, _ptr(buf), n, self.body_syms, float(self.cfg.sc_threshold),
                                 float(self.cfg.sc_ncc_min), self._hits, len(self._hits))
        out: list[CxFrame] = []
        cut = max(0, n - self.keep)
        held = None                           # first hit some table could still decode once more arrives
        for h in self._hits[:max(k, 0)]:
            a = self._i0 + h.start
            if a - self._last < self.min_gap:
                continue                      # already reported
            if h.start + self._fmin > n:
                if held is None:
                    held = h.start            # not even the shortest frame is in yet
                break
            order = [self._pref] + [i for i in range(len(self._dems)) if i != self._pref]
            same = self._tried_at is not None and abs(a - self._tried_at) < c.symbol_samples
            tried = self._tried if same else set()
            got, later = None, False
            for i in order:
                d = self._dems[i]
                if id(d) in tried:
                    continue
                if h.start + d.frame_samples > n:
                    later = True              # this table's frame is not all here yet
                    continue
                tried.add(id(d))
                info = CxDemOut()
                rc = lib.ofdm_cxdem_frame(d.h, _ptr(buf), n, h.start, h.cfo, self._chunk,
                                          d.x.c.chunk_bytes, ctypes.byref(info))
                if rc == 0:
                    got = (i, d, info)
                    break
                if rc == -5 and got is None:
                    got = (None, d, info)     # demodulated, RS failed: remember the EVM
            ok = got is not None and got[0] is not None
            # a failed attempt is reported once the hit is settled; read its quality now, while that
            # table's cxdem state still holds this frame
            if same and self._held_bad is not None:
                bad = self._held_bad          # the first failure: the preferred table, tried first
            else:
                bad = self._report(h, a, got) if got is not None and not ok else None
            if not ok and later and not final:
                if held is None:              # wait for the longer tables; later hits may still decode
                    held = h.start
                    self._tried_at, self._tried, self._held_bad = a, tried, bad
                continue
            if ok and held is not None and held < h.start:
                # a frame decoded after it: the held hit was no long frame.  Still report what it was
                # (grading counts the failed frames: dropping them silently stalled ARQ table updates)
                if self._held_bad is not None and self._held_bad.index != a:
                    out.append(self._held_bad)
                held = None
            if same or ok:
                self._tried_at, self._tried, self._held_bad = None, set(), None
            f = self._report(h, a, got) if ok else bad
            if f is None:
                continue
            if ok:
                self._pref = got[0]
            out.append(f)
            self._last = a
            cut = max(cut, h.start + (got[1].frame_samples if ok else self._fmin))
        if held is not None:
            cut = min(cut, max(0, held - c.symbol_samples))   # keep it for the next pass
        if cut:
            self._buf = self._buf[cut:]
            self._i0 += cut
        return out


@dataclasses.dataclass(frozen=True)
class CxProbeResult:
    ok: bool
    error: str
    freq_hz: np.ndarray       # signed carrier frequency, occupied order
    snr_db: np.ndarray        # per occupied carrier, image counted as interference
    snr_db_periodic: np.ndarray
    image_db: np.ndarray
    bat: list[int]            # per data carrier (ofdm_allocate), [] when not ok
    cfo_hz: float
    coherence: float
    frame_start: int          # in the capture passed to estimate()
    image_db_median: float | None
    isi_penalty_db: float | None

    def summary(self) -> str:
        if not self.ok:
            return f"probe: not found ({self.error})"
        from collections import Counter
        hist = dict(sorted(Counter(self.bat).items()))
        img = f", image {self.image_db_median:.1f} dB" if self.image_db_median is not None else ""
        return (f"probe: SNR median {np.median(self.snr_db):.1f} dB, CFO {self.cfo_hz:+.1f} Hz, "
                f"coherence {self.coherence:.2f}{img}; bits {hist} = {sum(self.bat)} per symbol")


class CxProber:
    """ofdm_cxprobe: the probe waveform, and its estimate on a capture."""

    def __init__(self, cfg, max_capture_s: float = 4.0) -> None:
        self.lib = lib = _lib()
        self.cfg = cfg
        self.x = build_cfg(cfg, lib)
        self.max_cap = int(max_capture_s * cfg.fs)
        args = (int(cfg.probe_syms), int(cfg.meas_syms), self.max_cap, int(cfg.max_cfo_bins))
        need = lib.ofdm_cxprobe_workspace(ctypes.byref(self.x), *args)
        if not need:
            raise native.NativeError("ofdm_cxprobe_workspace rejected the parameters")
        self._mem = ctypes.create_string_buffer(need)
        self._h = VP()
        rc = lib.ofdm_cxprobe_init(ctypes.byref(self._h), self._mem, need, ctypes.byref(self.x), *args)
        if rc:
            raise native.NativeError(f"ofdm_cxprobe_init rc={rc}")
        self.freq_hz = bins(self.x, lib) * self.x.c.sc_spacing_hz

    @property
    def samples(self) -> int:
        return int(self.lib.ofdm_cxprobe_build_samples(self._h))

    def build(self) -> np.ndarray:
        n = self.samples
        out = np.zeros(n, np.complex128)
        got = self.lib.ofdm_cxprobe_build(self._h, _ptr(out), n)
        if got != n:
            raise native.NativeError(f"ofdm_cxprobe_build rc={got}")
        return out

    def estimate(self, rx: np.ndarray, threshold: float = 0.0) -> CxProbeResult:
        """Grade a capture that holds the whole probe (at most ``max_capture_s``; the last part
        is used when longer)."""
        rx = _iq(rx)
        if len(rx) > self.max_cap:
            rx = rx[-self.max_cap:]
        c = self.x.c
        O, nd = c.n_occupied, c.n_data_carriers
        arrs = [np.zeros(O) for _ in range(6)]
        info = CxProbeInfo()
        rc = self.lib.ofdm_cxprobe_estimate(self._h, _ptr(rx), len(rx), float(threshold),
                                            *[_ptr(a) for a in arrs], ctypes.byref(info))
        if rc < 0:
            raise native.NativeError(f"ofdm_cxprobe_estimate rc={rc}")
        H_re, H_im, snr_lin, snr_db, snr_p, img = arrs
        bat: list[int] = []
        if rc == 0:
            b = (I * nd)()
            k = self.lib.ofdm_allocate(_ptr(np.ascontiguousarray(snr_db)), O, c.pilot_spacing,
                                       float(self.cfg.snr_margin_db)
                                       - FEC_GAIN_DB.get(str(getattr(self.cfg, "fec", "none")), 0.0),
                                       int(self.cfg.max_bits), 0.0, b, nd)
            if k != nd:
                raise native.NativeError(f"ofdm_allocate returned {k}, want {nd}")
            bat = list(b)
        return CxProbeResult(ok=rc == 0, error="" if rc == 0 else PROBE_ERRORS.get(info.error, f"error {info.error}"),
                             freq_hz=self.freq_hz, snr_db=snr_db, snr_db_periodic=snr_p, image_db=img, bat=bat,
                             cfo_hz=float(info.cfo) * c.sc_spacing_hz, coherence=float(info.coherence),
                             frame_start=int(info.frame_start),
                             image_db_median=float(info.image_db) if info.has_image else None,
                             isi_penalty_db=float(info.isi_penalty_db) if info.has_isi_penalty else None)


class ToneExciser:
    """Find stationary narrowband interferers (CW tones, birdies) in the RX stream and notch them out
    before sync and the probe see them (step 10c: one CW tone 6 dB under the signal used to wreck the
    probe's SNR on EVERY carrier and the sync's CFO).

    Detection: a 1024-point Hann spectrum of every block goes into a ring (~1.3 s); per bin the MEDIAN
    over time (a tone is there all the time, OFDM bursts come and go and data carriers vary) is compared
    with the 75th percentile of its +-16 bin neighbourhood (a flat OFDM band never stands out from its own
    neighbours; a band edge has half its neighbours at signal level).  ``thr_db`` above = a tone.  Up to
    ``max_notches``; a notch is dropped once its bin has been under thr_db - 5 for ``hold_s``.

    Filtering: a linear-phase FIR (``taps``, Hann-windowed inverse of the notch mask; each notch about one
    OFDM carrier wide) by overlap-save, ALWAYS applied (pure delay of taps//2 samples when there is no
    notch), so turning a notch on never shifts the timebase.  ``filter_snippet`` is the stateless version
    for a history snippet (the probe)."""

    def __init__(self, fs: float, block: int = 4000, thr_db: float = 12.0, nfft: int = 1024, ring: int = 64,
                 taps: int = 513, max_notches: int = 8, hold_s: float = 2.0, update_s: float = 0.5) -> None:
        self.fs, self.nfft, self.taps = float(fs), int(nfft), int(taps) | 1
        self.thr, self.max_notches, self.hold_s, self.update_s = float(thr_db), int(max_notches), hold_s, update_s
        self.win = np.hanning(self.nfft)
        self.psd: collections.deque = collections.deque(maxlen=int(ring))
        self.notch_bins: dict[int, float] = {}       # 1024-bin index -> last time it was above thr - 5
        self._t = 0.0
        self._last_update = -1e9
        self.ols = 8192
        self.hop = self.ols - self.taps + 1
        self._tail = np.zeros(self.taps - 1, np.complex128)
        self._pend = np.zeros(0, np.complex128)
        self._H = None                                 # frequency response for overlap-save, None = delay only
        self._h = None
        self.delay = self.taps // 2
        self._dl = np.zeros(self.delay, np.complex128)
        self.stats = collections.Counter()

    @property
    def notches_hz(self) -> list[float]:
        return sorted(((k + self.nfft // 2) % self.nfft - self.nfft // 2) * self.fs / self.nfft
                      for k in self.notch_bins)

    def _detect(self) -> None:
        if len(self.psd) < 8:
            return
        med = np.median(np.array(self.psd), axis=0)
        n = self.nfft
        k = np.arange(n)
        ref = np.empty(n)
        for i in range(n):                             # neighbourhood 75th percentile, +-16 bins, skip +-2
            idx = np.r_[i - 16:i - 2, i + 3:i + 17] % n
            ref[i] = np.percentile(med[idx], 75)
        ratio = 10 * np.log10(np.maximum(med, 1e-30) / np.maximum(ref, 1e-30))
        changed = False
        for i in k[ratio > self.thr - 5.0]:
            if i in self.notch_bins:
                self.notch_bins[i] = self._t
        for i in np.argsort(-ratio):
            if ratio[i] <= self.thr or len(self.notch_bins) >= self.max_notches:
                break
            if all(min(abs(i - j), n - abs(i - j)) > 2 for j in self.notch_bins):
                self.notch_bins[int(i)] = self._t
                changed = True
                self.stats["notches_added"] += 1
        for i in [i for i, t in self.notch_bins.items() if self._t - t > self.hold_s]:
            del self.notch_bins[i]
            changed = True
        if changed:
            self._design()

    def _design(self) -> None:
        if not self.notch_bins:
            self._H = self._h = None
            return
        m = 4096
        mask = np.ones(m)
        f = np.fft.fftfreq(m, 1.0 / self.fs)
        width = 1.5 * self.fs / self.nfft              # +-1.5 analysis bins around each tone
        for hz in self.notches_hz:
            mask[np.abs(f - hz) <= width] = 0.0
        h = np.fft.fftshift(np.fft.ifft(mask))
        c = m // 2
        h = h[c - self.taps // 2:c + self.taps // 2 + 1] * np.hanning(self.taps + 2)[1:-1]
        self._h = h
        self._H = np.fft.fft(h, self.ols)

    def process(self, x: np.ndarray) -> np.ndarray:
        """One RX block in, the same number of samples out, delayed by ``delay`` samples."""
        x = np.asarray(x, np.complex128).reshape(-1)
        self._t += len(x) / self.fs
        if len(x) >= self.nfft:
            seg = x[:self.nfft] * self.win
            self.psd.append(np.abs(np.fft.fft(seg)) ** 2)
        if self._t - self._last_update >= self.update_s:
            self._last_update = self._t
            self._detect()
        if self._H is None:                            # pure delay, same alignment as the FIR
            y = np.concatenate([self._dl, x])
            self._dl = y[len(x):]
            self._tail = np.concatenate([self._tail, x])[-(self.taps - 1):]
            return y[:len(x)]
        buf = np.concatenate([self._tail, x])
        out = np.empty(len(x), np.complex128)
        o = 0
        while o < len(x):
            n = min(self.hop, len(x) - o)
            seg = buf[o:o + n + self.taps - 1]
            Y = np.fft.ifft(np.fft.fft(seg, self.ols) * self._H)
            out[o:o + n] = Y[self.taps - 1:self.taps - 1 + n]
            o += n
        self._tail = buf[-(self.taps - 1):]
        self._dl = np.concatenate([self._dl, x])[-self.delay:]
        self.stats["filtered_blocks"] += 1
        return out

    def filter_snippet(self, x: np.ndarray) -> np.ndarray:
        """Stateless: a history snippet with the current notches (same length, zero-phase)."""
        if self._h is None or len(x) < self.taps:
            return x
        y = np.convolve(np.asarray(x, np.complex128), self._h, mode="same")
        return y
