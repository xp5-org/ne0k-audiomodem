# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""ofdm_cfg.c vs ofdm_modem: geometry, PRBS, tables, frame arithmetic.

The C cores used to be handed all of this by the Python bridge at init --
geometry, reference sequences, constellations.  A freestanding build (DOS /
OpenWatcom, or an MCU) has no Python, so ofdm_cfg.c derives it instead.  That
makes it a TRANSCRIPTION, and transcriptions drift silently: a constellation
reordered, a ceiling turned into a floor, a taper clamp dropped.  Nothing about
a modem that still runs tells you the tables agree, because both ends are the
same C.  So this compares field by field against the Python that defines the
wire format.

    python tests/regression/ofdm_cfg_regress.py [-v]

Exit 0 = pass.  SKIP (still 0) if the C is not built.
"""
from __future__ import annotations

import argparse
import ctypes
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "OFDM"))

import numpy as np                                              # noqa: E402
import cbackend                                                 # noqa: E402
import ofdm_modem as M                                          # noqa: E402


class Check:
    def __init__(self, verbose): self.bad = 0; self.n = 0; self.v = verbose
    def eq(self, got, want, what):
        self.n += 1
        ok = got == want
        if not ok:
            self.bad += 1
            print(f"  FAIL {what}: C={got!r} py={want!r}")
        elif self.v:
            print(f"  ok   {what} = {got!r}")
    def close(self, got, want, what, tol=0.0):
        self.n += 1
        d = abs(float(got) - float(want))
        if d > tol:
            self.bad += 1
            print(f"  FAIL {what}: C={got!r} py={want!r} (|d|={d:g} > {tol:g})")
        elif self.v:
            print(f"  ok   {what} = {got!r}")


class Cfg(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in (
        "n_fft", "cp_len", "bin_lo", "bin_hi", "n_occupied", "pilot_spacing",
        "n_pilots", "n_data_carriers", "sample_rate", "symbol_samples",
        "taper_samples", "frame_gap_samples", "ecc_bytes", "chunk_bytes",
        "seq_bytes", "sync_period")] + [(n, ctypes.c_double) for n in (
        "cp_ratio", "sc_spacing_hz", "freq_lo_hz", "freq_hi_hz")]


def load():
    so = ROOT / "OFDM" / "c" / cbackend.lib_filename("libofdmacq")
    if not so.exists():
        return None
    lib = ctypes.CDLL(str(so))
    if not hasattr(lib, "ofdm_cfg_build"):
        return None
    lib.ofdm_cfg_build.restype = ctypes.c_int
    lib.ofdm_cfg_build.argtypes = [
        ctypes.POINTER(Cfg), ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_int]
    lib.ofdm_prbs.restype = None
    lib.ofdm_prbs.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
                              ctypes.c_uint]
    lib.ofdm_known_bpsk.restype = None
    lib.ofdm_known_bpsk.argtypes = [ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int, ctypes.c_uint,
                                    ctypes.c_double]
    lib.ofdm_interleave_geometry.restype = ctypes.c_int
    lib.ofdm_interleave_geometry.argtypes = [
        ctypes.POINTER(Cfg), ctypes.c_int] + [ctypes.POINTER(ctypes.c_int)] * 3
    for f in ("ofdm_data_syms_per_pkt", "ofdm_n_syncs", "ofdm_frame_min_gap"):
        getattr(lib, f).restype = ctypes.c_int
        getattr(lib, f).argtypes = [ctypes.POINTER(Cfg), ctypes.c_int]
    lib.ofdm_mod_table.restype = ctypes.c_int
    lib.ofdm_mod_table.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
        ctypes.POINTER(ctypes.c_int)]
    lib.ofdm_bat_widths.restype = ctypes.c_int
    lib.ofdm_bat_widths.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    return lib


GEOMS = [
    # n_fft, cp_ratio, lo,   hi,     pilot, fs,    ecc, chunk, seq, taper, gap, sync
    (1024, 0.25,  400.0, 4000.0,  8, 48000, 16, 256, 2, 2.0, 20.0, 32),
    (1024, 0.25,  400.0, 8000.0,  8, 48000, 16, 256, 2, 2.0, 20.0, 32),
    (2048, 0.375, 400.0, 8000.0,  8, 48000, 16, 256, 2, 2.0, 20.0, 32),
    ( 512, 0.25,  400.0, 3600.0,  8, 20000, 16,  48, 2, 2.0, 20.0, 32),
    (4096, 0.25, 1000.0,16000.0,  4, 44100, 10, 237, 2, 1.0, 10.0, 16),
    (8192, 0.5,   300.0, 6000.0, 16, 48000, 32, 200, 3, 3.0, 25.0,  8),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    lib = load()
    if lib is None:
        print("SKIP: ofdm_cfg not in the built library "
              "(run `make -C tests/OFDM/c both`)")
        return 0
    c = Check(a.verbose)

    # ---- constellations: the one pure transcription ----------------------
    for bits in (1, 2, 4, 6):
        pr = ctypes.POINTER(ctypes.c_double)()
        pi = ctypes.POINTER(ctypes.c_double)()
        pb = ctypes.POINTER(ctypes.c_ubyte)()
        n = ctypes.c_int()
        rc = lib.ofdm_mod_table(bits, ctypes.byref(pr), ctypes.byref(pi),
                                ctypes.byref(pb), ctypes.byref(n))
        c.eq(rc, 0, f"mod_table({bits}) rc")
        pts, bts = M._MOD_TABLES[bits]
        c.eq(n.value, len(pts), f"mod_table({bits}) n_pts")
        for i in range(n.value):
            c.close(pr[i], pts[i].real, f"tab{bits}[{i}].re", 1e-15)
            c.close(pi[i], pts[i].imag, f"tab{bits}[{i}].im", 1e-15)
        for i in range(n.value):
            for j in range(bits):
                c.eq(pb[i * bits + j], int(bts[i][j]), f"tab{bits}bits[{i}][{j}]")

    # ---- PRBS and the known sequences ------------------------------------
    for seed, name in ((M._SEED_DATA, "DATA"), (M._SEED_CE, "CE"),
                       (M._SEED_SC, "SC"), (M._SEED_PROBE, "PROBE")):
        n = 200
        buf = (ctypes.c_ubyte * n)()
        lib.ofdm_prbs(buf, n, seed)
        want = M._prbs(n, seed)
        c.eq(bytes(buf), bytes(want.tolist()), f"prbs[{name}]")
        d = (ctypes.c_double * n)()
        lib.ofdm_known_bpsk(d, n, seed, M.REF_SYM_BOOST / math.sqrt(2.0))
        ref = M._known_bpsk(n, seed) * M.REF_SYM_BOOST
        bad = max(abs(d[i] - ref[i].real) for i in range(n))
        c.close(bad, 0.0, f"known_bpsk[{name}] boosted", 1e-15)
        c.close(max(abs(ref[i].imag) for i in range(n)), 0.0,
                f"known_bpsk[{name}] imag is zero", 0.0)

    # ---- geometry --------------------------------------------------------
    for g in GEOMS:
        (n_fft, cpr, lo, hi, pil, fs, ecc, chunk, seq, tap, gap, sync) = g
        tag = f"{n_fft}/{cpr}/{lo:.0f}-{hi:.0f}/{fs}"
        cc = Cfg()
        rc = lib.ofdm_cfg_build(ctypes.byref(cc), n_fft, cpr, lo, hi, pil, fs,
                                ecc, chunk, seq, tap, gap, sync)
        py = M.build_config(n_fft=n_fft, cp_ratio=cpr, freq_lo=lo, freq_hi=hi,
                            pilot_spacing=pil, sample_rate=fs, ecc_bytes=ecc,
                            chunk_bytes=chunk, seq_bytes=seq, taper_ms=tap,
                            frame_gap_ms=gap, sync_period=sync)
        c.eq(rc, 0, f"[{tag}] cfg_build rc")
        for f in ("n_fft", "cp_len", "bin_lo", "bin_hi", "n_occupied",
                  "n_pilots", "n_data_carriers", "sample_rate",
                  "symbol_samples", "taper_samples", "frame_gap_samples",
                  "sync_period"):
            c.eq(getattr(cc, f), py[f], f"[{tag}] {f}")
        for f in ("sc_spacing_hz", "freq_lo_hz", "freq_hi_hz"):
            c.close(getattr(cc, f), py[f], f"[{tag}] {f}", 1e-9)

        # pilot offsets are range(0, n_occ, spacing); check the count matches
        c.eq(cc.n_pilots, len(list(range(0, py["n_occupied"], pil))),
             f"[{tag}] pilot count")

        # ---- frame arithmetic over a spread of tables --------------------
        nd = py["n_data_carriers"]
        for label, bat in (("bpsk", np.ones(nd, dtype=np.int32)),
                           ("qpsk", np.full(nd, 2, dtype=np.int32)),
                           ("16qam", np.full(nd, 4, dtype=np.int32)),
                           ("mixed", np.array([(1, 2, 4, 6)[i % 4]
                                               for i in range(nd)],
                                              dtype=np.int32))):
            bps = int(np.sum(bat))
            d_, k_, w_ = (ctypes.c_int(), ctypes.c_int(), ctypes.c_int())
            rc = lib.ofdm_interleave_geometry(
                ctypes.byref(cc), bps, ctypes.byref(d_), ctypes.byref(k_),
                ctypes.byref(w_))
            pd, pk, pw = M._interleave_geometry(seq, chunk, ecc, bps)
            c.eq(rc, 0, f"[{tag}/{label}] il rc")
            c.eq((d_.value, k_.value, w_.value), (pd, pk, pw),
                 f"[{tag}/{label}] interleave geometry")

            ds = lib.ofdm_data_syms_per_pkt(ctypes.byref(cc), bps)
            c.eq(ds, math.ceil(pw * 8 / bps), f"[{tag}/{label}] data_syms")
            ns = lib.ofdm_n_syncs(ctypes.byref(cc), ds)
            c.eq(ns, math.ceil(ds / sync) if sync > 0 and ds > 0 else 0,
                 f"[{tag}/{label}] n_syncs")
            n_body = 1 + ds + ns
            c.eq(lib.ofdm_frame_min_gap(ctypes.byref(cc), n_body),
                 M._frame_min_gap(py["symbol_samples"], n_body,
                                  py["taper_samples"], py["cp_len"]),
                 f"[{tag}/{label}] min_gap")

            arr = (ctypes.c_int * nd)(*[int(x) for x in bat])
            out = (ctypes.c_int * 4)()
            cnt = lib.ofdm_bat_widths(arr, nd, out, 4)
            want = sorted({int(x) for x in bat})
            c.eq([out[i] for i in range(cnt)], want, f"[{tag}/{label}] widths")

    print(f"\n{c.n - c.bad}/{c.n} checks passed")
    print("OFDM CFG REGRESS " + ("PASS" if c.bad == 0 else "FAIL"))
    return 1 if c.bad else 0


if __name__ == "__main__":
    sys.exit(main())
