# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""css_mf_c.py — ctypes bridge to the C matched filter (tests/CSS/c/libcssmf.*).

Loaded IN-PROCESS, deliberately, rather than run as a subprocess over a pipe or
socket.  The C in tests/CSS/c/ is the artifact that ships to the embedded
target, so it is written freestanding (caller-owned memory, no malloc/stdio).
A subprocess design would force a framing/marshalling transport that exists
only on the host and gets thrown away — and what the regression suite would
then be validating is "C core + transport", not the core that ships.  In-process
ctypes runs the exact object code built from the exact .c files, and passes the
symbol window as a pointer into the numpy buffer with no copy at all (the window
is ~75 KB and this is called once per symbol, so a copy each way would show).

Set CSS_MF_BACKEND=c to route MFSymbolFilter through this; anything else (the
default) keeps the numpy implementation.  Both are held to the same regression
suite, which is the only way the two stay honest about each other.

The library is resolved through `cbackend`, which picks the build for this
host's CPU (`libcssmf.x86_64-linux.so`, `libcssmf.arm64-darwin.dylib`, …).  With
CSS_MF_BACKEND=c set, a missing or wrong-architecture library RAISES rather
than falling back: silently grading numpy is indistinguishable from grading the
port, which is the one thing this bridge exists to tell apart.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

import cbackend

_BACKEND = cbackend.BACKENDS["css_mf"]


class _Out(ctypes.Structure):
    _fields_ = [("conf", ctypes.c_float), ("peak", ctypes.c_int),
                ("snr_db", ctypes.c_float), ("winner_pk", ctypes.c_float)]


def _load(path: Path):
    lib = ctypes.CDLL(str(path))
    lib.css_mf_workspace.restype  = ctypes.c_size_t
    lib.css_mf_workspace.argtypes = [ctypes.c_int] * 5
    lib.css_mf_init.restype  = ctypes.c_int
    lib.css_mf_init.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.css_mf_run.restype  = ctypes.c_int
    lib.css_mf_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int, ctypes.c_int, ctypes.POINTER(_Out)]
    lib.css_mf_nfft.restype  = ctypes.c_int
    lib.css_mf_nfft.argtypes = [ctypes.c_void_p]
    return lib


_LIB = None


def available() -> bool:
    """True if a libcssmf for THIS host's CPU is present and loadable.

    Never raises.  A cross-architecture library on the shared mount used to
    land here as a plain False and silently return the decoder to the numpy
    MFSymbolFilter — the very implementation the C port is measured against.
    """
    global _LIB
    if _LIB is None:
        rec = _BACKEND.probe()
        if not rec["ok"]:
            return False
        _LIB = _load(Path(rec["path"]))
        if rec["stale"]:
            print(f"WARNING: {Path(rec['path']).name} is older than "
                  f"{', '.join(rec['stale'])} — run "
                  f"`make -C {_BACKEND.directory}` (loading it anyway)",
                  file=sys.stderr, flush=True)
    return True


def require() -> None:
    """Raise `cbackend.BackendUnavailable` unless the C core is usable here.

    For the CSS_MF_BACKEND=c path, where a fallback would grade numpy while the
    log claims C.
    """
    if not available():
        raise cbackend.BackendUnavailable(_BACKEND.probe()["reason"])


def _f32p(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class MFSymbolFilterC:
    """Drop-in for css_decoder.MFSymbolFilter backed by the C core.

    Same constructor signature and the same (conf, peak, snr_db, winner_pk)
    return, so decode_longsym_mf is untouched.
    """

    def __init__(self, up_ref, down_ref, sps, track_search, decide_search,
                 seg_M=12, seg_trim=2, blank_kmad=0.0):
        require()
        if seg_M < 2:
            raise ValueError("C core handles seg_M >= 2 only")
        if blank_kmad:
            raise ValueError("blank_kmad unsupported in the C core (default is off)")
        self.sps           = int(sps)
        self.track_search  = int(track_search)
        self.decide_search = int(decide_search)
        self.seg_M         = int(seg_M)
        self.seg_trim      = int(seg_trim)
        self.blank_kmad    = 0.0
        self._max_win      = int(sps) + 2 * int(track_search)

        self._up = np.ascontiguousarray(up_ref[:sps], dtype=np.float32)
        self._dn = np.ascontiguousarray(down_ref[:sps], dtype=np.float32)

        nbytes = _LIB.css_mf_workspace(self.sps, self.track_search, self.seg_M,
                                       self.seg_trim, self._max_win)
        if nbytes == 0:
            raise RuntimeError("css_mf_workspace rejected these parameters")
        # One block, owned by Python, handed to C — mirrors how the target will
        # hand it a static buffer.  Kept alive by this object.
        self._mem = ctypes.create_string_buffer(int(nbytes))
        self.workspace_bytes = int(nbytes)
        self._h = ctypes.c_void_p()
        rc = _LIB.css_mf_init(ctypes.byref(self._h),
                              ctypes.cast(self._mem, ctypes.c_void_p), nbytes,
                              _f32p(self._up), _f32p(self._dn),
                              self.sps, self.track_search, self.decide_search,
                              self.seg_M, self.seg_trim, self._max_win)
        if rc != 0:
            raise RuntimeError(f"css_mf_init failed rc={rc}")
        self.nfft = _LIB.css_mf_nfft(self._h)
        self._out = _Out()

    def __call__(self, win, centre):
        w = np.ascontiguousarray(win, dtype=np.float32)
        rc = _LIB.css_mf_run(self._h, _f32p(w), w.size, int(centre),
                             ctypes.byref(self._out))
        if rc != 0:
            raise RuntimeError(f"css_mf_run failed rc={rc}")
        o = self._out
        return float(o.conf), int(o.peak), float(o.snr_db), float(o.winner_pk)
