# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""css_enc_c.py — ctypes bridge to the CSS frame codec (tests/CSS/c/libcssenc.*).

Covers BOTH directions: css_enc.c (chirp, CRC, convolutional encode,
interleave, differential encode) and css_dec.c (deinterleave, soft-decision
Viterbi, differential decode, CRC check).  One module rather than two because
both are driven by the same css_enc_cfg geometry struct, and two ctypes
mirrors of one struct is exactly the drift that sharing it prevents.

Same design as css_mf_c.py and for the same reason: the C in tests/CSS/c/ is
the artifact that ships to the embedded target, so it is loaded IN-PROCESS
and the regression suite exercises the exact object code the build produced,
not a host-only wrapper around it.  See css_mf_c's module docstring for the
full argument against a subprocess/IPC design.

TWO LIBRARIES, deliberately.  Every entry point takes an `f64` flag:

    f64=False  -> libcssenc     float32 chirp phase + Viterbi metric,
                                THE SHIPPING BUILD
    f64=True   -> libcssenc64   float64 both, grading only

Stage 1 of the equivalence check grades the f64 build against numpy: the phase
arithmetic matches, so any difference there is a PORT bug and nothing else.
Stage 2 grades the f32 build against the f64 one, where a difference is a
PRECISION effect by construction.  Collapsing the two into one comparison is
what makes a float32 port hard to debug -- you get one number that could mean
either thing.  Same split ofdm_acq/ofdm_acq32 uses.

Unlike the matched filter this is not on any hot path: it runs once per frame
over a few hundred bits, plus one chirp table per run.  The bridge exists to
GRADE the port, not to accelerate Python -- which is why the wrappers below do
no work of their own beyond buffer setup.  Logic added here would be logic the
equivalence suites are not testing.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

import cbackend


class CssEncCfg(ctypes.Structure):
    """Mirror of css_enc_cfg.  Field order and types must track css_enc.h."""
    _fields_ = [("chunk_size", ctypes.c_int),
                ("rs_parity", ctypes.c_int),
                ("base_bits", ctypes.c_int),
                ("conv_code", ctypes.c_int),
                ("diff_enc", ctypes.c_int),
                ("interleave_rows", ctypes.c_int),
                ("conv_bits", ctypes.c_int),
                ("frame_symbols", ctypes.c_int)]

    def as_dict(self):
        return {f: getattr(self, f) for f, _ in self._fields_}


_U8P = ctypes.POINTER(ctypes.c_ubyte)


def _bind(lib):
    lib.css_enc_geometry.restype = ctypes.c_int
    lib.css_enc_geometry.argtypes = [ctypes.POINTER(CssEncCfg)] + [ctypes.c_int] * 5
    lib.css_crc32.restype = ctypes.c_ulong
    lib.css_crc32.argtypes = [_U8P, ctypes.c_int]
    lib.css_enc_inner.restype = ctypes.c_int
    lib.css_enc_inner.argtypes = [ctypes.POINTER(CssEncCfg), ctypes.c_int,
                                  _U8P, _U8P]
    lib.css_bytes_to_bits.restype = ctypes.c_int
    lib.css_bytes_to_bits.argtypes = [_U8P, ctypes.c_int, _U8P]
    lib.css_conv_encode_k7.restype = ctypes.c_int
    lib.css_conv_encode_k7.argtypes = [_U8P, ctypes.c_int, _U8P]
    lib.css_diff_encode.restype = ctypes.c_int
    lib.css_diff_encode.argtypes = [_U8P, ctypes.c_int, ctypes.c_int, _U8P]
    lib.css_interleave.restype = ctypes.c_int
    lib.css_interleave.argtypes = [_U8P, ctypes.c_int, ctypes.c_int, _U8P]
    lib.css_enc_scratch.restype = ctypes.c_int
    lib.css_enc_scratch.argtypes = [ctypes.POINTER(CssEncCfg)]
    lib.css_enc_bits.restype = ctypes.c_int
    lib.css_enc_bits.argtypes = [ctypes.POINTER(CssEncCfg), _U8P, ctypes.c_int,
                                 ctypes.c_int, _U8P, ctypes.c_int,
                                 _U8P, ctypes.c_int]
    lib.css_enc_chirp.restype = ctypes.c_int
    lib.css_enc_chirp.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                  ctypes.c_double, ctypes.c_double,
                                  ctypes.c_double]
    return lib


_LIBS: dict[bool, object] = {}


def available(f64: bool = False) -> bool:
    """True if the requested build is present for THIS host's CPU.

    Never raises, matching css_mf_c.available().  A cross-architecture library
    on the shared mount is a plain False here; it is `require()` that turns
    that into an error, so a caller that has not asked for the C backend is
    not broken by one being unbuildable.
    """
    key = "css_enc64" if f64 else "css_enc"
    if f64 in _LIBS:
        return True
    rec = cbackend.BACKENDS[key].probe()
    if not rec["ok"]:
        return False
    _LIBS[f64] = _bind(ctypes.CDLL(str(Path(rec["path"]))))
    return True


def require(f64: bool = False):
    """The library, or BackendUnavailable naming why it could not load."""
    if not available(f64):
        cbackend.BACKENDS["css_enc64" if f64 else "css_enc"].require()
    return _LIBS[f64]


# ── thin numpy-facing wrappers ───────────────────────────────────────────────
#
# These exist for the harness's convenience only.  Each is a direct call with
# no logic of its own beyond buffer setup: any adjustment made here rather than
# in the C would be adjusting the thing under test.

def _u8(arr) -> ctypes.Array:
    b = bytes(bytearray(int(x) & 0xFF for x in arr))
    return (ctypes.c_ubyte * max(len(b), 1)).from_buffer_copy(b or b"\x00")


def geometry(chunk_size, rs_t, conv_code, diff_enc, interleave_rows,
             f64=False) -> CssEncCfg:
    lib = require(f64)
    g = CssEncCfg()
    rc = lib.css_enc_geometry(ctypes.byref(g), chunk_size, rs_t,
                              1 if conv_code == "k7r12" else 0,
                              1 if diff_enc else 0, interleave_rows)
    if rc != 0:
        raise ValueError(f"css_enc_geometry rc={rc}")
    return g


def crc32(data: bytes, f64=False) -> int:
    lib = require(f64)
    src = _u8(data)
    return int(lib.css_crc32(src, len(data))) & 0xFFFFFFFF


def enc_inner(g: CssEncCfg, seq: int, data: bytes, f64=False) -> bytes:
    lib = require(f64)
    src = _u8(data)
    out = (ctypes.c_ubyte * (1 + g.chunk_size + 4))()
    n = lib.css_enc_inner(ctypes.byref(g), seq, src, out)
    if n < 0:
        raise ValueError(f"css_enc_inner rc={n}")
    return bytes(out[:n])


def bytes_to_bits(data: bytes, f64=False) -> list:
    lib = require(f64)
    src = _u8(data)
    out = (ctypes.c_ubyte * max(len(data) * 8, 1))()
    n = lib.css_bytes_to_bits(src, len(data), out)
    return list(out[:n])


def conv_encode_k7(bits, f64=False) -> list:
    lib = require(f64)
    src = _u8(bits)
    out = (ctypes.c_ubyte * ((len(bits) + 6) * 2))()
    n = lib.css_conv_encode_k7(src, len(bits), out)
    return list(out[:n])


def diff_encode(bits, ref=0, f64=False) -> list:
    lib = require(f64)
    src = _u8(bits)
    out = (ctypes.c_ubyte * max(len(bits), 1))()
    n = lib.css_diff_encode(src, len(bits), int(ref), out)
    return list(out[:n])


def interleave(bits, rows, f64=False) -> list:
    lib = require(f64)
    src = _u8(bits)
    cols = -(-len(bits) // rows) if rows > 1 else len(bits)
    out = (ctypes.c_ubyte * max((rows if rows > 1 else 1) * cols, 1))()
    n = lib.css_interleave(src, len(bits), rows, out)
    return list(out[:n])


def enc_bits(g: CssEncCfg, frame: bytes, diff_ref=0, f64=False) -> list:
    lib = require(f64)
    src = _u8(frame)
    cap = g.frame_symbols
    out = (ctypes.c_ubyte * max(cap, 1))()
    scap = lib.css_enc_scratch(ctypes.byref(g))
    scratch = (ctypes.c_ubyte * max(scap, 1))()
    n = lib.css_enc_bits(ctypes.byref(g), src, len(frame), int(diff_ref),
                         out, cap, scratch, scap)
    if n < 0:
        raise ValueError(f"css_enc_bits rc={n}")
    return list(out[:n])


def chirp(f0, f1, duration, fs, f64=False) -> np.ndarray:
    """Linear FM sweep as float32, sized int(fs * duration) like the Python.

    `duration` is passed through to C rather than fs: the time base is
    linspace(0, duration, n, endpoint=False), which is i/fs only when
    fs*duration is integral.  See css_enc.h.
    """
    lib = require(f64)
    n = int(fs * duration)
    out = (ctypes.c_float * max(n, 1))()
    rc = lib.css_enc_chirp(out, n, float(f0), float(f1), float(duration))
    if rc != 0:
        raise ValueError(f"css_enc_chirp rc={rc}")
    return np.frombuffer(out, dtype=np.float32, count=n).copy()


# ── receive side (css_dec.c) ─────────────────────────────────────────────────
#
# Same library, same two builds.  Bound here rather than in a separate module
# because the RX pipeline is driven by the SAME css_enc_cfg the TX derived --
# splitting the bridge would mean two ctypes mirrors of one struct, which is
# the drift this shares it to avoid.

def _bind_dec(lib):
    if getattr(lib, "_css_dec_bound", False):
        return lib
    lib.css_viterbi_workspace.restype = ctypes.c_size_t
    lib.css_viterbi_workspace.argtypes = [ctypes.c_int]
    lib.css_viterbi_k7.restype = ctypes.c_int
    lib.css_viterbi_k7.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                   ctypes.c_void_p, ctypes.c_size_t, _U8P,
                                   ctypes.POINTER(ctypes.c_float)]
    lib.css_deinterleave.restype = ctypes.c_int
    lib.css_deinterleave.argtypes = [_U8P, ctypes.POINTER(ctypes.c_float),
                                     ctypes.c_int, ctypes.c_int, _U8P,
                                     ctypes.POINTER(ctypes.c_float)]
    lib.css_diff_decode.restype = ctypes.c_int
    lib.css_diff_decode.argtypes = [_U8P, ctypes.c_int, ctypes.c_int, _U8P]
    lib.css_bits_to_bytes.restype = ctypes.c_int
    lib.css_bits_to_bytes.argtypes = [_U8P, ctypes.c_int, _U8P]
    lib.css_check_crc.restype = ctypes.c_int
    lib.css_check_crc.argtypes = [_U8P, ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int)]
    lib.css_dec_workspace.restype = ctypes.c_size_t
    lib.css_dec_workspace.argtypes = [ctypes.POINTER(CssEncCfg)]
    lib.css_dec_raw.restype = ctypes.c_int
    lib.css_dec_raw.argtypes = [ctypes.POINTER(CssEncCfg), _U8P,
                                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                                _U8P, ctypes.c_int,
                                ctypes.POINTER(ctypes.c_float)]
    lib._css_dec_bound = True
    return lib


def _f32(arr) -> ctypes.Array:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    buf = (ctypes.c_float * max(a.size, 1))()
    if a.size:
        ctypes.memmove(buf, a.ctypes.data, a.nbytes)
    return buf


def viterbi_k7(conf, f64=False):
    """(bits, energies) for a flat soft-value array of 2*npairs values."""
    lib = _bind_dec(require(f64))
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)
    npairs = conf.size // 2
    src = _f32(conf)
    wlen = lib.css_viterbi_workspace(npairs)
    work = (ctypes.c_ubyte * max(int(wlen), 1))()
    nb = npairs - 6
    bits = (ctypes.c_ubyte * max(nb, 1))()
    energy = (ctypes.c_float * max(nb, 1))()
    n = lib.css_viterbi_k7(src, npairs, work, wlen, bits, energy)
    if n < 0:
        raise ValueError(f"css_viterbi_k7 rc={n}")
    return list(bits[:n]), [float(x) for x in energy[:n]]


def deinterleave(bits, conf, rows, f64=False):
    lib = _bind_dec(require(f64))
    n = len(bits)
    src = _u8(bits)
    sc = _f32(conf) if conf is not None else None
    out = (ctypes.c_ubyte * max(n, 1))()
    oc = (ctypes.c_float * max(n, 1))() if conf is not None else None
    r = lib.css_deinterleave(src, sc, n, rows, out, oc)
    if r < 0:
        raise ValueError(f"css_deinterleave rc={r}")
    return list(out[:r]), ([float(x) for x in oc[:r]] if conf is not None
                           else None)


def diff_decode(bits, ref=0, f64=False):
    lib = _bind_dec(require(f64))
    src = _u8(bits)
    out = (ctypes.c_ubyte * max(len(bits), 1))()
    n = lib.css_diff_decode(src, len(bits), int(ref), out)
    return list(out[:n])


def bits_to_bytes(bits, f64=False) -> bytes:
    lib = _bind_dec(require(f64))
    src = _u8(bits)
    out = (ctypes.c_ubyte * max(len(bits) // 8, 1))()
    n = lib.css_bits_to_bytes(src, len(bits), out)
    return bytes(out[:n])


def check_crc(raw: bytes, chunk_size: int, f64=False):
    """(ok, seq, data) — the same triple css_fec.check_crc returns."""
    lib = _bind_dec(require(f64))
    src = _u8(raw)
    seq = ctypes.c_int(-1)
    rc = lib.css_check_crc(src, len(raw), chunk_size, ctypes.byref(seq))
    if rc < 0:
        raise ValueError(f"css_check_crc rc={rc}")
    if rc != 1:
        return False, 0, b""
    return True, seq.value, bytes(raw[1:1 + chunk_size])


def dec_raw(g: CssEncCfg, bits, conf, diff_ref=0, f64=False):
    """(raw_bytes, byte_conf) — the RX pipeline up to but not including RS."""
    lib = _bind_dec(require(f64))
    n = len(bits)
    src = _u8(bits)
    sc = _f32(conf)
    wlen = lib.css_dec_workspace(ctypes.byref(g))
    work = (ctypes.c_ubyte * max(int(wlen), 1))()
    cap = g.base_bits // 8
    raw = (ctypes.c_ubyte * max(cap, 1))()
    bc = (ctypes.c_float * max(cap, 1))()
    nr = lib.css_dec_raw(ctypes.byref(g), src, sc, n, int(diff_ref),
                         work, wlen, raw, cap, bc)
    if nr < 0:
        raise ValueError(f"css_dec_raw rc={nr}")
    return bytes(raw[:nr]), [float(x) for x in bc[:nr]]
