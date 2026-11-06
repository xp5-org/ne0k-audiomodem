# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""ofdm_sync.c vs _sc_metric / _find_sc_peaks / _sc_match.

Acquisition is graded on DECISIONS, not on the metric: the metric is an
intermediate, and the C computes it from running window sums where the
reference differences two cumulative sums.  Both are sequential sums with the
same error class -- the reference's is arguably worse conditioned, since it
subtracts two large numbers -- so bit-equality is not the bar and would not mean
anything if it held.

What must agree is what a caller acts on: the same frame starts, in the same
order, refined to the same sample.

    python tests/regression/ofdm_sync_regress.py [-v]

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
from ofdm_cfg_regress import Cfg                                # noqa: E402


def load():
    so = ROOT / "OFDM" / "c" / cbackend.lib_filename("libofdmacq")
    if not so.exists():
        return None
    lib = ctypes.CDLL(str(so))
    if not hasattr(lib, "ofdm_sync_init"):
        return None
    lib.ofdm_cfg_build.restype = ctypes.c_int
    lib.ofdm_cfg_build.argtypes = [
        ctypes.POINTER(Cfg), ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_int]
    lib.ofdm_sync_workspace.restype = ctypes.c_size_t
    lib.ofdm_sync_workspace.argtypes = [ctypes.POINTER(Cfg)]
    lib.ofdm_sync_init.restype = ctypes.c_int
    lib.ofdm_sync_init.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.c_void_p, ctypes.c_size_t,
                                   ctypes.POINTER(Cfg),
                                   ctypes.POINTER(ctypes.c_double)]
    lib.ofdm_sync_find.restype = ctypes.c_int
    lib.ofdm_sync_find.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.ofdm_sync_refine.restype = ctypes.c_int
    lib.ofdm_sync_refine.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double)]
    return lib


CASES = [
    (1024, 0.25, 400.0, 8000.0, 8, 48000, 16, 256, 2, 2.0, 20.0, 32, 2),
    (1024, 0.25, 400.0, 4000.0, 8, 48000, 16, 256, 2, 2.0, 20.0, 32, 1),
    (2048, 0.375, 400.0, 8000.0, 8, 48000, 16, 256, 2, 2.0, 20.0, 32, 2),
    (512, 0.25, 400.0, 3600.0, 8, 20000, 16, 48, 2, 2.0, 20.0, 32, 2),
]
SNRS = (30.0, 20.0, 12.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    lib = load()
    if lib is None:
        print("SKIP: ofdm_sync not in the built library "
              "(run `make -C tests/OFDM/c both`)")
        return 0

    bad = 0
    for (n_fft, cpr, lo, hi, pil, fs, ecc, chunk, seq, tap, gap, sync,
         bits) in CASES:
        cc = Cfg()
        rc = lib.ofdm_cfg_build(ctypes.byref(cc), n_fft, cpr, lo, hi, pil, fs,
                                ecc, chunk, seq, tap, gap, sync)
        assert rc == 0
        py = M.build_config(n_fft=n_fft, cp_ratio=cpr, freq_lo=lo, freq_hi=hi,
                            pilot_spacing=pil, sample_rate=fs, ecc_bytes=ecc,
                            chunk_bytes=chunk, seq_bytes=seq, taper_ms=tap,
                            frame_gap_ms=gap, sync_period=sync)
        bat = np.full(py["n_data_carriers"], bits, dtype=np.int32)
        mod = M.AdaptiveModulator(py, bat)
        dem = M.AdaptiveDemodulator(py, bat)
        n_syncs = math.ceil(mod.data_syms_per_pkt / sync)
        n_body = 1 + mod.data_syms_per_pkt + n_syncs

        need = lib.ofdm_sync_workspace(ctypes.byref(cc))
        mem = ctypes.create_string_buffer(need)
        h = ctypes.c_void_p()
        ref = (ctypes.c_double * py["symbol_samples"])(*mod.sc_td)
        rc = lib.ofdm_sync_init(ctypes.byref(h), mem, need, ctypes.byref(cc), ref)
        if rc != 0:
            print(f"  FAIL sync_init rc={rc}"); bad += 1; continue

        payload = bytes((i * 37 + 11) & 0xFF for i in range(chunk * 3))
        sig = mod.encode_bytes(payload)

        # keyup=True prepends ofdm_link.keyup()'s 1 kHz lead tone.  Its SC
        # metric sits on the 0.25 threshold and its NCC (~0.12) passes
        # SC_NCC_MIN, so without the replace rule the last tone edge locks just
        # ahead of frame 0 and hides it (mesh run 20261106_011344 _73f5).
        for snr, keyup in [(s_, False) for s_ in SNRS] + [(s_, True) for s_ in SNRS]:
            tag = f"{n_fft}/{fs}/b{bits}/{snr:.0f}dB" + ("/keyup" if keyup else "")
            rs = np.random.RandomState(int(snr) * 7 + n_fft)
            lead = 4000
            if keyup:
                ku = int(0.5 * fs)
                tone = float(np.max(np.abs(sig))) * np.sin(2 * np.pi * 1000.0 * np.arange(ku) / fs)
                sig_k = np.concatenate([tone, sig])
            else:
                sig_k = sig
            rx = np.zeros(len(sig_k) + 2 * lead)
            rx[lead:lead + len(sig_k)] = sig_k
            p = float(np.mean(sig ** 2))
            rx += rs.randn(len(rx)) * math.sqrt(p / 10 ** (snr / 10))

            cbuf = (ctypes.c_double * len(rx))(*rx)
            out = (ctypes.c_int * 64)()
            n = lib.ofdm_sync_find(h, cbuf, len(rx), n_body,
                                   M.SC_NCC_MIN * 0 + 0.25, M.SC_NCC_MIN,
                                   out, 64)
            c_starts = [out[i] for i in range(max(0, n))]

            # Python reference: the same pipeline decode_samples runs.
            Mm = M._sc_metric(rx, py["n_fft"])
            min_gap = M._frame_min_gap(py["symbol_samples"], n_body,
                                       py["taper_samples"], py["cp_len"])
            match = {}

            def _validate(pos):
                m = match.get(pos)
                if m is None:
                    m = match[pos] = dem._sc_match(rx, pos)
                return m[1] >= M.SC_NCC_MIN

            peaks = M._find_sc_peaks(Mm, py["symbol_samples"], min_gap, 0.25,
                                     validate=_validate,
                                     score=lambda pos: match[pos][1])
            py_starts = [match[d][0] if d in match else
                         dem._refine_frame_start(rx, d) for d in peaks]

            if c_starts != py_starts:
                print(f"  FAIL [{tag}] starts C={c_starts} py={py_starts}")
                bad += 1
            elif keyup and len(c_starts) != 3:
                print(f"  FAIL [{tag}] keyup tone hid a frame: {len(c_starts)}/3 "
                      f"starts {c_starts}")
                bad += 1
            elif a.verbose:
                print(f"  ok   [{tag}] {len(c_starts)} starts {c_starts}")

            # refine() on its own, at deliberately offset probe points
            for d in py_starts[:2]:
                for off in (-py["cp_len"] // 2, 0, py["cp_len"] // 3):
                    probe = max(0, d + off)
                    bi = ctypes.c_int()
                    sc = ctypes.c_double()
                    lib.ofdm_sync_refine(h, cbuf, len(rx), probe,
                                         ctypes.byref(bi), ctypes.byref(sc))
                    pb, ps = dem._sc_match(rx, probe)
                    if bi.value != pb or abs(sc.value - ps) > 1e-9:
                        print(f"  FAIL [{tag}] refine({probe}): "
                              f"C=({bi.value},{sc.value:.6f}) "
                              f"py=({pb},{ps:.6f})")
                        bad += 1
        if a.verbose:
            print(f"  -- {n_fft}/{fs} done")

    print("\nOFDM SYNC REGRESS " + ("PASS" if bad == 0 else f"FAIL ({bad})"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
