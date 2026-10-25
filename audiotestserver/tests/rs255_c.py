# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""rs255_c.py — ctypes bridge to the C Reed-Solomon core (c/librs255.so).

Drop-in for `reedsolo.RSCodec(nsym)`: same encode bytes, same decode return
shape `(msg, msg+ecc, errata)`, same ReedSolomonError on an uncorrectable
block.

Set RS_BACKEND=c to route the modems through it; anything else (the default)
keeps the pure-Python `reedsolo`. `tests/regression/rs255_regress.py`
compares both over random error patterns including beyond-capacity ones.

Scope: only RSCodec(nsym) with reedsolo's defaults (nsize=255, fcr=0,
prim=0x11D, generator=2, c_exp=8) is supported. Anything else raises rather
than decoding against a different field.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import cbackend

_BACKEND = cbackend.BACKENDS["rs255"]
_C_DIR = _BACKEND.directory


class ReedSolomonError(Exception):
    """Mirrors reedsolo.ReedSolomonError so callers can catch either.

    Bound to the real class when reedsolo is importable, so
    `except reedsolo.ReedSolomonError` keeps working with the C backend.
    """


try:                                                    # pragma: no cover
    from reedsolo import ReedSolomonError as _RSE
    ReedSolomonError = _RSE                             # noqa: F811
except Exception:                                       # noqa: BLE001
    pass


def _load(path: Path):
    lib = ctypes.CDLL(str(path))
    ub = ctypes.POINTER(ctypes.c_ubyte)
    lib.rs255_workspace.restype = ctypes.c_size_t
    lib.rs255_workspace.argtypes = [ctypes.c_int]
    lib.rs255_init.restype = ctypes.c_int
    lib.rs255_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                               ctypes.c_size_t, ctypes.c_int]
    lib.rs255_encoded_len.restype = ctypes.c_int
    lib.rs255_encoded_len.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rs255_encode.restype = ctypes.c_int
    lib.rs255_encode.argtypes = [ctypes.c_void_p, ub, ctypes.c_int,
                                 ub, ctypes.c_int]
    lib.rs255_decode.restype = ctypes.c_int
    lib.rs255_decode.argtypes = [ctypes.c_void_p, ub, ctypes.c_int, ub,
                                 ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.rs255_nsym.restype = ctypes.c_int
    lib.rs255_nsym.argtypes = [ctypes.c_void_p]
    lib.rs255_block_k.restype = ctypes.c_int
    lib.rs255_block_k.argtypes = [ctypes.c_void_p]
    return lib


_LIB = None
_PROBED = False


def available() -> bool:
    """True if a librs255 for this host's CPU is present and loadable.

    Never raises. Callers that require C go through `require()` instead.
    """
    global _LIB, _PROBED
    if _PROBED:
        return _LIB is not None
    _PROBED = True
    rec = _BACKEND.probe()
    if not rec["ok"]:
        _LIB = None
        return False
    _LIB = _load(Path(rec["path"]))
    if rec["stale"]:
        print(f"WARNING: {Path(rec['path']).name} is older than "
              f"{', '.join(rec['stale'])} — run `make -C {_C_DIR}` "
              f"(loading it anyway)", file=sys.stderr, flush=True)
    return True


def require() -> None:
    """Raise `cbackend.BackendUnavailable` unless the C core is usable here."""
    if not available():
        raise cbackend.BackendUnavailable(_BACKEND.probe()["reason"])


class RSCodecC:
    """reedsolo.RSCodec-compatible façade over the C core."""

    def __init__(self, nsym=10, nsize=255, fcr=0, prim=0x11D,
                 generator=2, c_exp=8, single_gen=True):
        if (nsize, fcr, prim, generator, c_exp) != (255, 0, 0x11D, 2, 8):
            raise ValueError(
                "rs255_c supports only reedsolo's default field "
                "(nsize=255, fcr=0, prim=0x11D, generator=2, c_exp=8); "
                f"got nsize={nsize} fcr={fcr} prim={prim:#x} "
                f"generator={generator} c_exp={c_exp}")
        require()
        self.nsym = int(nsym)
        nb = _LIB.rs255_workspace(self.nsym)
        if nb == 0:
            raise ValueError(f"rs255 rejects nsym={nsym} (need 1..254)")
        self.workspace_bytes = int(nb)
        self._mem = ctypes.create_string_buffer(int(nb))
        self._h = ctypes.c_void_p()
        rc = _LIB.rs255_init(ctypes.byref(self._h),
                             ctypes.cast(self._mem, ctypes.c_void_p),
                             nb, self.nsym)
        if rc != 0:
            raise RuntimeError(f"rs255_init failed rc={rc}")
        self.block_k = _LIB.rs255_block_k(self._h)

    @staticmethod
    def _buf(data):
        b = bytes(data)
        return (ctypes.c_ubyte * max(len(b), 1)).from_buffer_copy(
            b or b"\x00"), len(b)

    def encode(self, data):
        src, n = self._buf(data)
        cap = _LIB.rs255_encoded_len(self._h, n)
        out = (ctypes.c_ubyte * max(cap, 1))()
        wr = _LIB.rs255_encode(self._h, src, n, out, cap)
        if wr < 0:
            raise RuntimeError(f"rs255_encode failed rc={wr}")
        return bytearray(bytes(out[:wr]))

    def decode(self, data, erase_pos=None, only_erasures=False):
        # Erasure decoding is not implemented. css_fec.rs_decode_frame can
        # pass erase_pos when erasure_conf_thr > 0; that raises here rather
        # than silently ignoring the erasure list. An empty list is not an
        # erasure request and passes through.
        if erase_pos or only_erasures:
            raise NotImplementedError(
                "rs255_c does not implement erasure decoding; "
                "unset RS_BACKEND to use reedsolo for this call")
        src, n = self._buf(data)
        out = (ctypes.c_ubyte * max(n, 1))()
        er = ctypes.c_int()
        wr = _LIB.rs255_decode(self._h, src, n, out, n, ctypes.byref(er))
        if wr == -3:
            raise ReedSolomonError("Could not correct message")
        if wr < 0:
            raise RuntimeError(f"rs255_decode failed rc={wr}")
        msg = bytes(out[:wr])
        # reedsolo returns (msg, msg+ecc, errata_positions); the middle is
        # re-encoded here rather than stubbed.
        full = self.encode(msg)
        return bytearray(msg), bytearray(full), bytearray(b"\x00" * er.value)


def RSCodec(nsym=10, **kw):
    """RS_BACKEND=c -> the C core; anything else -> reedsolo (the default).

    One env var switches the FEC layer for CSS, OFDM and QAM at once.
    RS_BACKEND=c raises when the core is unusable rather than falling back
    to reedsolo. An unset RS_BACKEND keeps the reedsolo default.
    """
    if os.environ.get("RS_BACKEND", "").lower() == "c":
        require()
        return RSCodecC(nsym, **kw)
    from reedsolo import RSCodec as _RS
    return _RS(nsym, **kw)
