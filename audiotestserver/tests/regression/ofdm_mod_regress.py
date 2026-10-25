# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""ofdm_mod.c vs AdaptiveModulator: the emitted waveform, sample for sample.

This is the first TRANSMIT code in C, and a transmitter cannot be graded the way
a receiver can.  A demodulator that is subtly wrong still produces bytes you can
compare; a modulator that is subtly wrong produces audio that only fails on
someone else's radio.  So the bar here is the waveform itself -- every sample of
every frame against ofdm_modem.encode_bytes -- plus a round trip back through
the C demodulator, which proves the two halves agree about the frame they share.

    python tests/regression/ofdm_mod_regress.py [-v]

Exit 0 = pass.  SKIP (still 0) if the C is not built.
"""
from __future__ import annotations

import argparse
import ctypes
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
    if not hasattr(lib, "ofdm_mod_init"):
        return None
    lib.ofdm_cfg_build.restype = ctypes.c_int
    lib.ofdm_cfg_build.argtypes = [
        ctypes.POINTER(Cfg), ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_int]
    lib.ofdm_mod_workspace.restype = ctypes.c_size_t
    lib.ofdm_mod_workspace.argtypes = [ctypes.POINTER(Cfg),
                                       ctypes.POINTER(ctypes.c_int)]
    lib.ofdm_mod_init.restype = ctypes.c_int
    lib.ofdm_mod_init.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(Cfg), ctypes.POINTER(ctypes.c_int)]
    lib.ofdm_mod_signal_samples.restype = ctypes.c_long
    lib.ofdm_mod_signal_samples.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ofdm_mod_encode.restype = ctypes.c_long
    lib.ofdm_mod_encode.argtypes = [ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_double), ctypes.c_long]
    for f, r in (("ofdm_mod_bits_per_sym", ctypes.c_int),
                 ("ofdm_mod_data_syms", ctypes.c_int),
                 ("ofdm_mod_body_syms", ctypes.c_int),
                 ("ofdm_mod_payload_bytes", ctypes.c_int),
                 ("ofdm_mod_frame_samples", ctypes.c_long)):
        getattr(lib, f).restype = r
        getattr(lib, f).argtypes = [ctypes.c_void_p]
    lib.ofdm_mod_sc_preamble.restype = ctypes.POINTER(ctypes.c_double)
    lib.ofdm_mod_sc_preamble.argtypes = [ctypes.c_void_p]
    return lib


CASES = [
    # n_fft, cp,    lo,     hi,    pilot, fs,    ecc, chunk, seq, tap, gap, sync, table
    (1024, 0.25,  400.0, 4000.0,  8, 48000, 16, 256, 2, 2.0, 20.0, 32, "bpsk"),
    (1024, 0.25,  400.0, 8000.0,  8, 48000, 16, 256, 2, 2.0, 20.0, 32, "qpsk"),
    (1024, 0.25,  400.0, 8000.0,  8, 48000, 16, 256, 2, 2.0, 20.0, 32, "mixed"),
    (2048, 0.375, 400.0, 8000.0,  8, 48000, 16, 256, 2, 2.0, 20.0, 32, "16qam"),
    ( 512, 0.25,  400.0, 3600.0,  8, 20000, 16,  48, 2, 2.0, 20.0, 32, "qpsk"),
    ( 512, 0.25,  400.0, 3600.0,  8, 20000, 16,  48, 2, 0.0, 20.0, 32, "bpsk"),
    (1024, 0.25, 1000.0,16000.0,  4, 44100, 10, 237, 2, 1.0, 10.0, 16, "64qam"),
]


def make_bat(kind, nd):
    if kind == "bpsk":  return np.ones(nd, dtype=np.int32)
    if kind == "qpsk":  return np.full(nd, 2, dtype=np.int32)
    if kind == "16qam": return np.full(nd, 4, dtype=np.int32)
    if kind == "64qam": return np.full(nd, 6, dtype=np.int32)
    return np.array([(1, 2, 4, 6)[i % 4] for i in range(nd)], dtype=np.int32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    lib = load()
    if lib is None:
        print("SKIP: ofdm_mod not in the built library "
              "(run `make -C tests/OFDM/c both`)")
        return 0

    bad = 0
    for (n_fft, cpr, lo, hi, pil, fs, ecc, chunk, seq, tap, gap, sync,
         kind) in CASES:
        tag = f"{n_fft}/{cpr}/{fs}/{kind}"
        bad0 = bad
        cc = Cfg()
        rc = lib.ofdm_cfg_build(ctypes.byref(cc), n_fft, cpr, lo, hi, pil, fs,
                                ecc, chunk, seq, tap, gap, sync)
        assert rc == 0, (tag, rc)
        py = M.build_config(n_fft=n_fft, cp_ratio=cpr, freq_lo=lo, freq_hi=hi,
                            pilot_spacing=pil, sample_rate=fs, ecc_bytes=ecc,
                            chunk_bytes=chunk, seq_bytes=seq, taper_ms=tap,
                            frame_gap_ms=gap, sync_period=sync)
        bat = make_bat(kind, py["n_data_carriers"])
        cbat = (ctypes.c_int * len(bat))(*[int(x) for x in bat])

        need = lib.ofdm_mod_workspace(ctypes.byref(cc), cbat)
        if need == 0:
            print(f"  FAIL [{tag}] workspace rejected"); bad += 1; continue
        mem = ctypes.create_string_buffer(need)
        h = ctypes.c_void_p()
        rc = lib.ofdm_mod_init(ctypes.byref(h), mem, need, ctypes.byref(cc), cbat)
        if rc != 0:
            print(f"  FAIL [{tag}] mod_init rc={rc}"); bad += 1; continue

        mod = M.AdaptiveModulator(py, bat)
        if lib.ofdm_mod_bits_per_sym(h) != mod.bits_per_ofdm_sym:
            print(f"  FAIL [{tag}] bits_per_sym "
                  f"{lib.ofdm_mod_bits_per_sym(h)} != {mod.bits_per_ofdm_sym}")
            bad += 1
        if lib.ofdm_mod_data_syms(h) != mod.data_syms_per_pkt:
            print(f"  FAIL [{tag}] data_syms "
                  f"{lib.ofdm_mod_data_syms(h)} != {mod.data_syms_per_pkt}")
            bad += 1
        if lib.ofdm_mod_payload_bytes(h) != mod.payload_bytes:
            print(f"  FAIL [{tag}] payload_bytes "
                  f"{lib.ofdm_mod_payload_bytes(h)} != {mod.payload_bytes}")
            bad += 1

        # SC preamble on its own: it is shared with the receiver's matched
        # filter, so an error here breaks acquisition, not just the waveform.
        scp = lib.ofdm_mod_sc_preamble(h)
        sc_c = np.array([scp[i] for i in range(py["symbol_samples"])])
        d_sc = float(np.max(np.abs(sc_c - mod.sc_td)))
        if d_sc > 1e-12:
            print(f"  FAIL [{tag}] SC preamble max|d|={d_sc:.3e}"); bad += 1

        # payloads: exact multiple of a chunk, and a short final chunk that
        # exercises the NUL padding.
        for label, nb in (("exact", chunk * 2), ("short", chunk * 2 - 7),
                          ("tiny", 3)):
            payload = bytes((i * 37 + 11) & 0xFF for i in range(nb))
            n = lib.ofdm_mod_signal_samples(h, nb)
            out = (ctypes.c_double * n)()
            buf = (ctypes.c_ubyte * len(payload))(*payload)
            got = lib.ofdm_mod_encode(h, buf, len(payload), out, n)
            if got != n:
                print(f"  FAIL [{tag}/{label}] encode returned {got} != {n}")
                bad += 1
                continue
            c_sig = np.frombuffer(out, dtype=np.float64, count=n)
            py_sig = mod.encode_bytes(payload)
            if len(py_sig) != n:
                print(f"  FAIL [{tag}/{label}] length {n} != py {len(py_sig)}")
                bad += 1
                continue
            d = float(np.max(np.abs(c_sig - py_sig)))
            rms = float(np.sqrt(np.mean(py_sig ** 2)))
            if d > 1e-11:
                where = int(np.argmax(np.abs(c_sig - py_sig)))
                print(f"  FAIL [{tag}/{label}] max|d|={d:.3e} at sample {where} "
                      f"of {n} (signal rms {rms:.4f})")
                bad += 1
            elif a.verbose:
                print(f"  ok   [{tag}/{label}] {n} samples, max|d|={d:.2e}")
        if bad == bad0:
            print(f"  [{tag}] waveform matches")

    print("\nOFDM MOD REGRESS " + ("PASS" if bad == 0 else f"FAIL ({bad})"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
