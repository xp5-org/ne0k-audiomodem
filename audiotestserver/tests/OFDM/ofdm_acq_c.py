"""ofdm_acq_c.py — ctypes bridge to the C acquisition core (c/libofdmacq.so).

Loaded in-process rather than run as a subprocess: the C in tests/OFDM/c/ is
freestanding (caller-owned memory, no malloc/stdio), the artifact that ships
to the 32-bit target. In-process ctypes runs that exact object code and
passes the capture as a pointer into the numpy buffer with no copy.

Set OFDM_ACQ_BACKEND=c to route _grid_coherence_scan through this; anything
else (the default) keeps the numpy implementation. Both are held to the same
handshake sim.

OFDM_ACQ_LIB=32 selects the float32 build (libofdmacq32.*), the target
precision, instead of the default double build that matches the numpy oracle.

Both libraries are resolved through `cbackend`, which picks the build for this
host's CPU (`libofdmacq.x86_64-linux.so`, `libofdmacq.arm64-darwin.dylib`, …).
With OFDM_ACQ_BACKEND=c set, a missing or wrong-architecture library raises
rather than falling back.
"""
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

import cbackend

_C_DIR = cbackend.BACKENDS["ofdm_acq"].directory


def _backend():
    """The cbackend entry for the precision OFDM_ACQ_LIB currently selects."""
    return cbackend.BACKENDS["ofdm_acq32"
                             if os.environ.get("OFDM_ACQ_LIB", "") == "32"
                             else "ofdm_acq"]

# PRBS seed the demodulator descrambles with. Imported rather than restated:
# must match the modulator's scrambling constant exactly.
from ofdm_modem import _SEED_DATA                                # noqa: E402


def _lib_path() -> Path:
    """Where the selected precision's library is expected for this host.

    Used only for messages and as a cache key; `available()` resolves the
    real path through cbackend, which also accepts a legacy unsuffixed build
    when its header says it matches this CPU.
    """
    return _backend().path


class _Info(ctypes.Structure):
    _fields_ = [("n_cand", ctypes.c_int),
                ("n_pos", ctypes.c_int),
                ("start_step", ctypes.c_int)]


class _DemOut(ctypes.Structure):
    _fields_ = [("evm", ctypes.c_double),
                ("data_syms", ctypes.c_int),
                ("n_bytes", ctypes.c_int)]


def _load(path: Path):
    lib = ctypes.CDLL(str(path))
    lib.ofdm_acq_workspace.restype  = ctypes.c_size_t
    lib.ofdm_acq_workspace.argtypes = [ctypes.c_int] * 6
    lib.ofdm_acq_init.restype  = ctypes.c_int
    lib.ofdm_acq_init.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.ofdm_acq_run.restype  = ctypes.c_int
    lib.ofdm_acq_run.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(_Info)]
    lib.ofdm_acq_ring_rows.restype  = ctypes.c_int
    lib.ofdm_acq_ring_rows.argtypes = [ctypes.c_void_p]
    lib.ofdm_acq_nfft.restype  = ctypes.c_int
    lib.ofdm_acq_nfft.argtypes = [ctypes.c_void_p]
    lib.ofdm_acq_row.restype  = ctypes.c_int
    lib.ofdm_acq_row.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_float)]

    _dp = ctypes.POINTER(ctypes.c_double)
    _ip = ctypes.POINTER(ctypes.c_int)
    _up = ctypes.POINTER(ctypes.c_ubyte)
    lib.ofdm_dem_workspace.restype  = ctypes.c_size_t
    lib.ofdm_dem_workspace.argtypes = [ctypes.c_int] * 9
    lib.ofdm_dem_init.restype  = ctypes.c_int
    lib.ofdm_dem_init.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,     # geometry
        _dp, _dp, _dp, _dp,                                          # ce, sync
        _ip, ctypes.c_int, ctypes.c_double, ctypes.c_double,         # pilots
        _ip, ctypes.c_int,                                           # bits_per_bin
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint,     # framing
        _ip, _ip, ctypes.c_int, _dp, _dp, _up]                       # tables
    lib.ofdm_dem_frame.restype  = ctypes.c_int
    lib.ofdm_dem_frame.argtypes = [
        ctypes.c_void_p, _dp, ctypes.c_int, ctypes.c_int,
        _up, ctypes.c_int, ctypes.POINTER(_DemOut)]
    lib.ofdm_dem_total_body_syms.restype  = ctypes.c_int
    lib.ofdm_dem_total_body_syms.argtypes = [ctypes.c_void_p]
    return lib


_LIB: dict = {}


def available() -> bool:
    """True if a library for the selected precision AND this host's CPU is
    present and loadable.

    Never raises: a missing library means the numpy path stays in use. A stale
    library loads but warns. A wrong-architecture library is not available and
    is reported as such by `require()`.
    """
    b   = _backend()
    key = b.key
    if key in _LIB:
        return _LIB[key] is not None
    rec = b.probe()
    if not rec["ok"]:
        _LIB[key] = None
        return False
    lib = _load(Path(rec["path"]))
    if rec["stale"]:
        print(f"WARNING: {Path(rec['path']).name} is older than "
              f"{', '.join(rec['stale'])} — run `make -C {_C_DIR}` "
              f"(loading it anyway)", file=sys.stderr, flush=True)
    _LIB[key] = lib
    return True


def require() -> None:
    """Raise `cbackend.BackendUnavailable` unless the selected precision is
    usable here. For the OFDM_ACQ_BACKEND=c path, which must not silently
    fall back to numpy."""
    if not available():
        raise cbackend.BackendUnavailable(_backend().probe()["reason"])


def _dp(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class AcqScan:
    """One configured scan plan. Owns the C workspace for its lifetime.

    The workspace is allocated once here and reused across calls; `self._mem`
    is held as an attribute so the GC keeps it alive as long as the C handle
    points into it.
    """

    def __init__(self, n_fft, cp_len, bin_lo, n_occupied, known, nsyms, step,
                 q=None, top_k=40):
        require()
        lib = _LIB[_backend().key]
        self._lib = lib
        self.n_fft, self.cp_len, self.step = int(n_fft), int(cp_len), int(step)
        self.n_occupied, self.nsyms = int(n_occupied), int(nsyms)

        nbytes = lib.ofdm_acq_workspace(self.n_fft, self.cp_len,
                                        self.n_occupied, self.nsyms,
                                        self.step, int(top_k))
        if nbytes == 0:
            raise RuntimeError("ofdm_acq_workspace rejected these parameters "
                               f"(n_fft={n_fft} cp_len={cp_len} "
                               f"n_occupied={n_occupied} nsyms={nsyms} "
                               f"step={step} top_k={top_k})")
        self.workspace_bytes = int(nbytes)
        self._mem = ctypes.create_string_buffer(int(nbytes))
        self._h   = ctypes.c_void_p()

        k = np.ascontiguousarray(known, dtype=np.complex128)
        self._kre = np.ascontiguousarray(k.real, dtype=np.float64)
        self._kim = np.ascontiguousarray(k.imag, dtype=np.float64)
        if q is None:
            self._q = None
            qp = None
        else:
            self._q = np.ascontiguousarray(q, dtype=np.float64)
            qp = _dp(self._q)

        rc = lib.ofdm_acq_init(
            ctypes.byref(self._h),
            ctypes.cast(self._mem, ctypes.c_void_p), nbytes,
            _dp(self._kre), _dp(self._kim), qp,
            self.n_fft, self.cp_len, int(bin_lo), self.n_occupied,
            self.nsyms, self.step, int(top_k))
        if rc != 0:
            raise RuntimeError(f"ofdm_acq_init failed rc={rc}")
        self.ring_rows = lib.ofdm_acq_ring_rows(self._h)
        self._info = _Info()

    def __call__(self, rx):
        """Returns (cand_starts, metric), same contract as the numpy scan."""
        r = np.ascontiguousarray(rx, dtype=np.float64)
        n = len(r)
        SYM  = self.n_fft + self.cp_len
        span = (SYM // self.step) * (self.nsyms - 1)
        n_pos = ((n - self.cp_len - self.n_fft) // self.step + 1
                 if n >= self.cp_len + self.n_fft else 0)
        n_cand = max(0, n_pos - span)
        if n_cand == 0:
            return np.zeros(0, dtype=int), np.zeros(0)
        out = np.empty(n_cand, dtype=np.float64)
        rc = self._lib.ofdm_acq_run(self._h, _dp(r), n, _dp(out), n_cand,
                                    ctypes.byref(self._info))
        if rc != 0:
            raise RuntimeError(f"ofdm_acq_run failed rc={rc}")
        if self._info.n_cand != n_cand:
            raise RuntimeError(f"n_cand mismatch: C says {self._info.n_cand}, "
                               f"python computed {n_cand}")
        return np.arange(n_cand) * self.step, out

    def row(self, rx, pos):
        """One H row as complex64 — for diffing the FFT+normalise stage alone."""
        r = np.ascontiguousarray(rx, dtype=np.float64)
        buf = np.empty(2 * self.n_occupied, dtype=np.float32)
        rc = self._lib.ofdm_acq_row(
            self._h, _dp(r), len(r), int(pos),
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if rc != 0:
            raise RuntimeError(f"ofdm_acq_row failed rc={rc}")
        return buf.view(np.complex64).copy()


# Plans are keyed by geometry and reused: re-carving the workspace (which
# rebuilds the FFT twiddles) on every call would be wasteful.
_PLANS: dict = {}


def grid_coherence_scan_c(rx, n_fft, cp_len, bin_lo, n_occupied, known, nsyms,
                          step, q=None, top_k=40):
    key = (n_fft, cp_len, bin_lo, n_occupied, nsyms, step, top_k,
           None if q is None else tuple(np.asarray(q, dtype=float).tolist()),
           known.tobytes())
    plan = _PLANS.get(key)
    if plan is None:
        plan = AcqScan(n_fft, cp_len, bin_lo, n_occupied, known, nsyms, step,
                       q=q, top_k=top_k)
        _PLANS[key] = plan
        # Announce each new geometry once, so which backend ran is visible in the log.
        print(f"ACQ_BACKEND=c lib={_lib_path().name} n_fft={n_fft} "
              f"nsyms={nsyms} step={step} ring_rows={plan.ring_rows} "
              f"workspace={plan.workspace_bytes / 1024:.0f}KB", flush=True)
    return plan(rx)


class DemodPlan:
    """One configured data-frame demodulator. Owns its C workspace.

    Constellation tables are handed to the C at init rather than compiled
    into it, so ofdm_modem.py stays the single definition of the Gray coding.
    """

    def __init__(self, dem, mod_tables):
        require()
        lib = _LIB[_backend().key]
        self._lib = lib

        # Flatten the constellation tables in ascending bits-per-carrier order.
        bs = sorted(mod_tables)
        pts = np.concatenate([np.asarray(mod_tables[b][0], dtype=np.complex128)
                              for b in bs])
        bits = np.concatenate([np.asarray(mod_tables[b][1],
                                          dtype=np.uint8).reshape(-1)
                               for b in bs])
        self._tb = np.ascontiguousarray(bs, dtype=np.int32)
        self._tn = np.ascontiguousarray([len(mod_tables[b][0]) for b in bs],
                                        dtype=np.int32)
        self._pre = np.ascontiguousarray(pts.real, dtype=np.float64)
        self._pim = np.ascontiguousarray(pts.imag, dtype=np.float64)
        self._tbits = np.ascontiguousarray(bits, dtype=np.uint8)

        ce = np.ascontiguousarray(dem.ce_known, dtype=np.complex128)
        sy = np.ascontiguousarray(dem.sync_known, dtype=np.complex128)
        self._cere = np.ascontiguousarray(ce.real); self._ceim = np.ascontiguousarray(ce.imag)
        self._syre = np.ascontiguousarray(sy.real); self._syim = np.ascontiguousarray(sy.imag)
        self._pil = np.ascontiguousarray(dem.pilot_offsets, dtype=np.int32)
        self._bpb = np.ascontiguousarray(dem.bits_per_bin, dtype=np.int32)

        self.payload_bytes = int(dem.payload_bytes)
        nbytes = lib.ofdm_dem_workspace(
            int(dem.n_fft), int(dem.cp_len), int(dem.n_occupied),
            len(self._pil), len(self._bpb), int(dem.data_syms_per_pkt),
            int(dem.sync_period), self.payload_bytes,
            int(len(pts)), int(len(bits)))
        if nbytes == 0:
            raise RuntimeError("ofdm_dem_workspace rejected this geometry")
        self.workspace_bytes = int(nbytes)
        self._mem = ctypes.create_string_buffer(int(nbytes))
        self._h = ctypes.c_void_p()

        rc = lib.ofdm_dem_init(
            ctypes.byref(self._h),
            ctypes.cast(self._mem, ctypes.c_void_p), nbytes,
            int(dem.n_fft), int(dem.cp_len), int(dem.bin_lo), int(dem.n_occupied),
            _dp(self._cere), _dp(self._ceim), _dp(self._syre), _dp(self._syim),
            self._pil.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(self._pil),
            float(np.real(dem.PILOT_SYM)), float(np.imag(dem.PILOT_SYM)),
            self._bpb.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(self._bpb),
            int(dem.data_syms_per_pkt), int(dem.sync_period),
            self.payload_bytes, int(_SEED_DATA),
            self._tb.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._tn.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(self._tb),
            _dp(self._pre), _dp(self._pim),
            self._tbits.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        if rc != 0:
            raise RuntimeError(f"ofdm_dem_init failed rc={rc}")
        self._out = np.empty(self.payload_bytes, dtype=np.uint8)
        self._info = _DemOut()
        self.total_body_syms = lib.ofdm_dem_total_body_syms(self._h)

    def frame(self, samples, frame_start):
        """Returns (descrambled_bytes, evm_linear, data_syms) or None.

        None means the C declined the frame the same way the numpy path does:
        capture too short, or no data symbol survived.
        """
        r = np.ascontiguousarray(samples, dtype=np.float64)
        rc = self._lib.ofdm_dem_frame(
            self._h, _dp(r), len(r), int(frame_start),
            self._out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            self.payload_bytes, ctypes.byref(self._info))
        if rc in (-2, -3):
            return None
        if rc != 0:
            raise RuntimeError(f"ofdm_dem_frame failed rc={rc}")
        return (self._out[:self._info.n_bytes].tobytes(),
                float(self._info.evm), int(self._info.data_syms))


def demod_frame_c(dem, samples, frame_start):
    """Route AdaptiveDemodulator._demod_frame through the C core.

    Same contract as the numpy path: (raw_bytes, evm, data_syms) or None.

    The plan is cached ON THE DEMODULATOR, not in a module-level dict keyed by
    id(dem): CPython reuses id() once an object is collected, so a demodulator
    built after an earlier one was freed could receive the earlier one's plan
    (a different BAT and geometry) and decode against the wrong table. Caching
    on the object ties the plan's lifetime to the thing it describes. The lib
    name is part of the key so switching precision mid-process rebuilds.
    """
    from ofdm_modem import _MOD_TABLES
    lib = _lib_path().name
    cached = getattr(dem, "_c_demod_plan", None)
    if cached is not None and cached[0] == lib:
        return cached[1].frame(samples, frame_start)

    plan = DemodPlan(dem, _MOD_TABLES)
    dem._c_demod_plan = (lib, plan)
    # Announce once per demodulator, so which backend ran is visible in the log.
    print(f"DEM_BACKEND=c lib={lib} n_fft={dem.n_fft} "
          f"n_occ={dem.n_occupied} data_syms={dem.data_syms_per_pkt} "
          f"workspace={plan.workspace_bytes / 1024:.0f}KB", flush=True)
    return plan.frame(samples, frame_start)
