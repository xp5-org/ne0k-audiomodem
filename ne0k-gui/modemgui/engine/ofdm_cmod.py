# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""The OFDM data-frame modulator, in C.

ctypes wrapper over ``ofdm_mod_encode_ex`` (tests/OFDM/c/ofdm_mod.c), the port of
``AdaptiveModulator.encode_bytes(data, seq0, gap)``.  ofdm_mod_regress.py grades it sample for
sample against the reference, so ``CMod(cfg, bat).encode(body, seq0, gap)`` is a drop-in for
``ofdm_handshake.make_data_audio(..., seq0=, gap=)`` before ``scale_hs_tx``.

Used by the continuous broadcast (``OfdmLink._bcast_audio``), which encodes one chunk per call
for as long as the carrier is up: a per-chunk numpy encode there is the TX thread's main cost.

Not thread-safe: one CMod per thread (the modulator's scratch lives in its workspace).
"""
from __future__ import annotations

import ctypes

import numpy as np

from .. import native

D, I = ctypes.c_double, ctypes.c_int


def _lib() -> ctypes.CDLL:
    lib = native.load("libofdm")
    if getattr(lib, "_cmod_ready", False):
        return lib
    if not hasattr(lib, "ofdm_mod_encode_ex"):
        raise native.NativeError("libofdm lacks ofdm_mod_encode_ex (stale build: run make -C ne0k-gui)")
    P, vp = ctypes.POINTER, ctypes.c_void_p
    lib.ofdm_cfg_build.argtypes = [P(native.OfdmCfg), I, D, D, D, I, I, I, I, I, D, D, I]
    lib.ofdm_mod_workspace.restype = ctypes.c_size_t
    lib.ofdm_mod_workspace.argtypes = [P(native.OfdmCfg), P(I)]
    lib.ofdm_mod_init.restype = I
    lib.ofdm_mod_init.argtypes = [P(vp), vp, ctypes.c_size_t, P(native.OfdmCfg), P(I)]
    lib.ofdm_mod_signal_samples_ex.restype = ctypes.c_long
    lib.ofdm_mod_signal_samples_ex.argtypes = [vp, I, I]
    lib.ofdm_mod_encode_ex.restype = ctypes.c_long
    lib.ofdm_mod_encode_ex.argtypes = [vp, ctypes.c_char_p, I, ctypes.c_ulong, I, P(D), ctypes.c_long]
    lib._cmod_ready = True
    return lib


class CMod:
    """A C modulator for one data config (``hs_data_cfg``) and bit table."""

    def __init__(self, cfg: dict, bat) -> None:
        from .ofdm_cdx import _build_cfg
        self.lib = lib = _lib()
        self._cfg = _build_cfg(lib, cfg)
        if len(bat) != self._cfg.n_data_carriers:
            raise native.NativeError(f"bat has {len(bat)} entries, config has {self._cfg.n_data_carriers} data carriers")
        self._bat = (I * len(bat))(*[int(b) for b in bat])
        need = lib.ofdm_mod_workspace(ctypes.byref(self._cfg), self._bat)
        if not need:
            raise native.NativeError("ofdm_mod_workspace rejected the geometry or bit table")
        self._mem = ctypes.create_string_buffer(need)
        self._h = ctypes.c_void_p()
        rc = lib.ofdm_mod_init(ctypes.byref(self._h), self._mem, need, ctypes.byref(self._cfg), self._bat)
        if rc:
            raise native.NativeError(f"ofdm_mod_init rc={rc}")

    def encode(self, data: bytes, seq0: int = 0, gap: bool = True) -> np.ndarray:
        """``AdaptiveModulator(cfg, bat).encode_bytes(data, seq0, gap=gap)``, in C."""
        data = bytes(data)
        n = self.lib.ofdm_mod_signal_samples_ex(self._h, len(data), int(bool(gap)))
        if n < 0:
            raise native.NativeError(f"ofdm_mod_signal_samples_ex rc={n}")
        out = np.empty(n, dtype=np.float64)
        got = self.lib.ofdm_mod_encode_ex(self._h, data, len(data), int(seq0) & 0xFFFFFFFF, int(bool(gap)),
                                          out.ctypes.data_as(ctypes.POINTER(D)), n)
        if got != n:
            raise native.NativeError(f"ofdm_mod_encode_ex rc={got} (expected {n} samples)")
        return out
