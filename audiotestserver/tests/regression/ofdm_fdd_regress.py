# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""ofdm_fdd.c vs ofdm_fdd.py: band plan, frame sizing, sub-band BAT re-index.

    python tests/regression/ofdm_fdd_regress.py [-v]

Compiles ofdm_cfg.c + ofdm_fdd.c into a throwaway shared library (no dependence
on the make-built libs).  Exit 0 = pass.  The receive filter is not ported yet,
so it is not graded here.
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CDIR = ROOT / "OFDM" / "c"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "OFDM"))

import ofdm_fdd as F                                            # noqa: E402
import ofdm_modem as M                                          # noqa: E402

D = ctypes.c_double
I = ctypes.c_int


class Plan(ctypes.Structure):
    _fields_ = [(n, D) for n in ("dl_lo", "dl_hi", "ul_lo", "ul_hi", "guard_lo", "guard_hi",
                                 "dl_frac_actual", "min_bw_hz", "symbol_rate_hz",
                                 "carrier_spacing_hz")] + \
               [(n, I) for n in ("ratio_clamped", "within_one_octave", "harm2_overlap",
                                 "harm3_overlap")]


class Cfg(ctypes.Structure):
    _fields_ = [(n, I) for n in (
        "n_fft", "cp_len", "bin_lo", "bin_hi", "n_occupied", "pilot_spacing", "n_pilots",
        "n_data_carriers", "sample_rate", "symbol_samples", "taper_samples",
        "frame_gap_samples", "ecc_bytes", "chunk_bytes", "seq_bytes", "sync_period")] + \
        [(n, D) for n in ("cp_ratio", "sc_spacing_hz", "freq_lo_hz", "freq_hi_hz")]


def build_lib(tmp: Path) -> ctypes.CDLL:
    so = tmp / "fdd.so"
    subprocess.run(["cc", "-O1", "-std=c99", "-fPIC", "-shared", "-Wall", "-Wextra", "-pedantic",
                    f"-I{CDIR}", "-o", str(so), str(CDIR / "ofdm_cfg.c"), str(CDIR / "ofdm_fdd.c"),
                    "-lm"], check=True)
    lib = ctypes.CDLL(str(so))
    lib.ofdm_fdd_plan_bands.argtypes = [ctypes.POINTER(Plan), D, D, D, D, I, D, I, I, D, I]
    lib.ofdm_fdd_min_occupied.argtypes = [D, D, I]
    lib.ofdm_fdd_chunk_for_frame_len.argtypes = [ctypes.POINTER(Cfg), I, D]
    lib.ofdm_fdd_sub_bat.argtypes = [ctypes.POINTER(Cfg), ctypes.POINTER(I),
                                     ctypes.POINTER(Cfg), ctypes.POINTER(I)]
    lib.ofdm_cfg_build.argtypes = [ctypes.POINTER(Cfg), I, D, D, D, I, I, I, I, I, D, D, I]
    return lib


def c_cfg(lib, cfg: dict) -> Cfg:
    c = Cfg()
    rc = lib.ofdm_cfg_build(ctypes.byref(c), cfg["n_fft"], cfg["cp_ratio"], cfg["freq_lo_hz"],
                            cfg["freq_hi_hz"], cfg["pilot_spacing"], cfg["sample_rate"],
                            cfg["ecc_bytes"], cfg["chunk_bytes"], cfg["seq_bytes"],
                            cfg["taper_ms"], cfg["frame_gap_ms"], cfg["sync_period"])
    assert rc == 0, rc
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", action="store_true")
    v = ap.parse_args().v
    bad = n = 0

    def chk(ok, what):
        nonlocal bad, n
        n += 1
        if not ok:
            bad += 1
            print(f"  FAIL {what}")
        elif v:
            print(f"  ok   {what}")

    with tempfile.TemporaryDirectory() as t:
        lib = build_lib(Path(t))

        for mb in (0, 50, 100, 400):
            for sr in (40.0, 43.07, 46.875):
                for ps in (4, 8, 16):
                    chk(lib.ofdm_fdd_min_occupied(mb, sr, ps) == F.min_occupied_for_baud(mb, sr, ps),
                        f"min_occupied({mb},{sr},{ps})")

        for (lo, hi) in ((400, 3300), (400, 4000), (300, 2450), (400, 6000)):
            for frac in (0.3, 0.5, 0.7, 0.9):
                for guard in (300.0, 1200.0):
                    for side in ("high", "low"):
                        tag = f"plan {lo}-{hi} frac={frac} guard={guard} {side}"
                        args = (lo, hi, frac, guard, 1024, 0.25, 48000, 8, 100.0)
                        try:
                            py = F.plan_duplex_bands(*args, side)
                        except ValueError:
                            py = None
                        p = Plan()
                        rc = lib.ofdm_fdd_plan_bands(ctypes.byref(p), *args, 1 if side == "low" else 0)
                        if py is None:
                            chk(rc < 0, tag + " rejected")
                            continue
                        chk(rc == 0, tag + " accepted")
                        ok = (abs(p.dl_lo - py["dl_hz"][0]) < 0.06 and abs(p.dl_hi - py["dl_hz"][1]) < 0.06
                              and abs(p.ul_lo - py["ul_hz"][0]) < 0.06 and abs(p.ul_hi - py["ul_hz"][1]) < 0.06
                              and abs(p.guard_lo - py["guard_hz"][0]) < 0.06
                              and abs(p.guard_hi - py["guard_hz"][1]) < 0.06
                              and abs(p.dl_frac_actual - py["dl_frac_actual"]) < 5e-4
                              and bool(p.ratio_clamped) == py["ratio_clamped"]
                              and bool(p.within_one_octave) == py["within_one_octave"])
                        hs = {h["harmonic"] for h in py["harmonic_overlap"]}
                        ok = ok and bool(p.harm2_overlap) == (2 in hs) and bool(p.harm3_overlap) == (3 in hs)
                        chk(ok, tag)

        full = M.build_config(n_fft=1024, cp_ratio=0.25, freq_lo=400.0, freq_hi=3300.0,
                              pilot_spacing=8, sample_rate=48000)
        cfull = c_cfg(lib, full)
        for bits in (1, 8, 40, 200, 400):
            for ms in (0.5, 2.0, 5.0, 11.0):
                chk(lib.ofdm_fdd_chunk_for_frame_len(ctypes.byref(cfull), bits, ms)
                    == F.chunk_for_frame_len(full, [bits], ms), f"chunk bits={bits} max={ms}s")

        plan = F.plan_duplex_bands(400, 3300, 0.5, 1200.0, 1024, 0.25, 48000, 8, 100.0, "high")
        for band in (plan["ul_hz"], plan["dl_hz"]):
            sub = F.band_cfg(band, 48000)
            csub = c_cfg(lib, sub)
            fb = [((i * 7) % 4) and (1, 2, 4, 6)[i % 4] or 2 for i in range(int(full["n_data_carriers"]))]
            bins = [full["bin_lo"] + i for i in range(full["n_occupied"]) if i % full["pilot_spacing"]]
            by_bin = dict(zip(bins, fb))
            sbins = [sub["bin_lo"] + i for i in range(sub["n_occupied"]) if i % sub["pilot_spacing"]]
            want = [by_bin.get(b, 1) for b in sbins]
            arr = (I * len(fb))(*fb)
            out = (I * int(sub["n_data_carriers"]))()
            rc = lib.ofdm_fdd_sub_bat(ctypes.byref(cfull), arr, ctypes.byref(csub), out)
            chk(rc == 0 and list(out) == want, f"sub_bat {band}")

    print(f"ofdm_fdd_regress: {n - bad}/{n} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
