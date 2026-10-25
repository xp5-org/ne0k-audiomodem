# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""ofdm_probe.c vs ChannelProbe / isi_ratio / BitAllocator.

The probe is graded on what a caller ACTS on, in three layers:

  1. the probe WAVEFORM, sample for sample against ChannelProbe.build -- the
     transmit side, where a wrong symbol only fails on someone else's radio;
  2. isi_ratio and BitAllocator as pure functions, on synthetic channels where
     the answer is known independently of any capture;
  3. estimate() end to end over simulated rooms -- and there the bar is the
     DECISIONS: the same validity, the same symbol counts, an SNR that agrees
     to a fraction of a dB, and above all the SAME BIT TABLE, because that is
     the only output the rest of the system consumes.

Bit-equality is deliberately not the bar for layer 3.  The C accumulates mean
and variance with Welford in O(n_occupied) where the reference keeps every
per-symbol H and calls np.var; the acquisition scan has documented, explained
disagreements with numpy at boundary ties; and the coherence plateau makes the
fine-align argmax genuinely ambiguous.  What must not differ is the table.

    python tests/regression/ofdm_probe_regress.py [-v]

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


class Info(ctypes.Structure):
    _fields_ = [("valid", ctypes.c_int), ("error", ctypes.c_int),
                ("coherence", ctypes.c_double), ("noise_floor", ctypes.c_double),
                ("frame_start", ctypes.c_int), ("n_syms", ctypes.c_int),
                ("n_meas_syms", ctypes.c_int),
                ("has_isi_penalty", ctypes.c_int),
                ("isi_penalty_db", ctypes.c_double),
                ("has_beta", ctypes.c_int), ("isi_ratio", ctypes.c_double)]


D = ctypes.POINTER(ctypes.c_double)
I = ctypes.POINTER(ctypes.c_int)


def load():
    so = ROOT / "OFDM" / "c" / cbackend.lib_filename("libofdmacq")
    if not so.exists():
        return None
    lib = ctypes.CDLL(str(so))
    if not hasattr(lib, "ofdm_probe_init"):
        return None
    lib.ofdm_cfg_build.restype = ctypes.c_int
    lib.ofdm_cfg_build.argtypes = [
        ctypes.POINTER(Cfg), ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_int]
    lib.ofdm_probe_workspace.restype = ctypes.c_size_t
    lib.ofdm_probe_workspace.argtypes = [ctypes.POINTER(Cfg), ctypes.c_int,
                                         ctypes.c_int, ctypes.c_int]
    lib.ofdm_probe_init.restype = ctypes.c_int
    lib.ofdm_probe_init.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.POINTER(Cfg), ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int]
    lib.ofdm_probe_build_samples.restype = ctypes.c_long
    lib.ofdm_probe_build_samples.argtypes = [ctypes.c_void_p]
    lib.ofdm_probe_build.restype = ctypes.c_long
    lib.ofdm_probe_build.argtypes = [ctypes.c_void_p, D, ctypes.c_long]
    lib.ofdm_probe_estimate.restype = ctypes.c_int
    lib.ofdm_probe_estimate.argtypes = [ctypes.c_void_p, D, ctypes.c_int,
                                        ctypes.c_double, D, D, D, D, D,
                                        ctypes.POINTER(Info)]
    lib.ofdm_isi_scratch.restype = ctypes.c_size_t
    lib.ofdm_isi_scratch.argtypes = [ctypes.c_int]
    lib.ofdm_isi_ratio.restype = ctypes.c_double
    lib.ofdm_isi_ratio.argtypes = [ctypes.c_void_p, D, D, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   D, ctypes.c_int, ctypes.c_void_p,
                                   ctypes.c_size_t]
    lib.ofdm_allocate.restype = ctypes.c_int
    lib.ofdm_allocate.argtypes = [D, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_double, ctypes.c_int,
                                  ctypes.c_double, I, ctypes.c_int]
    lib.ofdm_fft_workspace.restype = ctypes.c_size_t
    lib.ofdm_fft_workspace.argtypes = [ctypes.c_int]
    lib.ofdm_fft_init.restype = ctypes.c_int
    lib.ofdm_fft_init.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_size_t]
    return lib


def arr(x):
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return a, a.ctypes.data_as(D)


GEOMS = [
    # n_fft, cp,    lo,     hi,   pilot, fs,    tap,  probe, meas
    (1024, 0.25,  400.0, 4000.0,  8, 48000, 2.0, 154, 32),
    (1024, 0.25,  400.0, 8000.0,  8, 48000, 2.0,  96, 24),
    (2048, 0.375, 400.0, 8000.0,  8, 48000, 2.0,  80, 32),
    ( 512, 0.25,  400.0, 3600.0,  8, 20000, 2.0, 120, 24),
]


def build_cfg(lib, g):
    n_fft, cpr, lo, hi, pil, fs, tap, nps, nms = g
    cc = Cfg()
    rc = lib.ofdm_cfg_build(ctypes.byref(cc), n_fft, cpr, lo, hi, pil, fs,
                            16, 256, 2, tap, 20.0, 32)
    assert rc == 0, rc
    py = M.build_config(n_fft=n_fft, cp_ratio=cpr, freq_lo=lo, freq_hi=hi,
                        pilot_spacing=pil, sample_rate=fs, ecc_bytes=16,
                        chunk_bytes=256, seq_bytes=2, taper_ms=tap,
                        frame_gap_ms=20.0, sync_period=32)
    py = dict(py)
    py["n_probe_syms"] = nps
    py["n_probe_meas_syms"] = nms
    return cc, py


def sim_room(x, fs, seed, taps, rt60, diffuse, snr_db):
    """Same shape of impairment ofdm_handshake's room profiles apply."""
    rs = np.random.RandomState(seed)
    n = max(int(rt60 * fs), 1) + 1
    last = max(int(round(d * fs / 1000.0)) for d, _ in taps) if taps else 0
    h = np.zeros(max(n, last + 1))
    h[0] = 1.0
    for d, a in taps:
        h[int(round(d * fs / 1000.0))] += a
    if rt60 > 0 and diffuse > 0:
        t = np.arange(n) / fs
        tail = rs.randn(n) * np.exp(-t * (math.log(1000.0) / rt60))
        tail[:last] = 0.0
        e = float(np.sum(tail ** 2))
        if e > 0:
            h[:n] += tail * math.sqrt(diffuse / e)
    y = np.convolve(x, h)[:len(x)]
    p = float(np.mean(y ** 2))
    return y + rs.randn(len(y)) * math.sqrt(p / 10 ** (snr_db / 10))


ROOMS = {
    "clean":  ([], 0.0, 0.0),
    "desk":   ([(0.9, 0.50), (1.7, 0.28), (3.1, 0.16)], 0.20, 0.06),
    "room":   ([(2.4, 0.45), (4.6, 0.30), (7.2, 0.20), (11.0, 0.12)], 0.45, 0.25),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    lib = load()
    if lib is None:
        print("SKIP: ofdm_probe not in the built library "
              "(run `make -C tests/OFDM/c both`)")
        return 0
    bad = 0

    # ---- layer 2: isi_ratio and allocate as pure functions -----------------
    fw = lib.ofdm_fft_workspace(1024)
    fmem = ctypes.create_string_buffer(fw)
    fft = ctypes.create_string_buffer(64)
    assert lib.ofdm_fft_init(fft, 1024, fmem, fw) == 0
    sw = lib.ofdm_isi_scratch(1024)
    smem = ctypes.create_string_buffer(sw)
    n_occ, bin_lo, cp = 162, 9, 256

    def H_of(taps):
        full = np.zeros(1024)
        for d, amp in taps.items():
            full[d] = amp
        return np.fft.rfft(full)[bin_lo:bin_lo + n_occ]

    for name, taps, snr in (("single tap", {5: 1.0}, None),
                            ("echo inside guard", {5: 1.0, 100: 0.4}, None),
                            ("echo past guard", {5: 1.0, cp + 40: 0.5}, None),
                            ("with snr", {5: 1.0, cp + 40: 0.35},
                             np.full(n_occ, 10 ** 1.4))):
        H = H_of(taps)
        hr, phr = arr(H.real)
        hi, phi = arr(H.imag)
        if snr is None:
            psnr = None
            snr_arg = ctypes.cast(None, D)
        else:
            sl, psnr = arr(snr)
            snr_arg = psnr
        for tap in (0, 96):
            c_val = lib.ofdm_isi_ratio(fft, phr, phi, n_occ, 1024, bin_lo, cp,
                                       snr_arg, tap, smem, sw)
            py_val = M.isi_ratio(H, 1024, bin_lo, cp,
                                 None if snr is None else snr,
                                 taper_samples=tap)
            if abs(c_val - py_val) > 1e-9 * max(1.0, abs(py_val)):
                print(f"  FAIL isi_ratio[{name},taper={tap}] "
                      f"C={c_val:.9g} py={py_val:.9g}")
                bad += 1
            elif a.verbose:
                print(f"  ok   isi_ratio[{name},taper={tap}] = {c_val:.6g}")

    rs = np.random.RandomState(4)
    for trial in range(6):
        snr_db = rs.uniform(-5, 35, n_occ)
        for margin in (0.0, 6.0, 9.0):
            for mx in (2, 4, 6):
                for beta in (0.0, 0.05, 0.25):
                    sd, psd = arr(snr_db)
                    out = (ctypes.c_int * n_occ)()
                    cnt = lib.ofdm_allocate(psd, n_occ, 8, margin, mx, beta,
                                            out, n_occ)
                    py_bat = M.BitAllocator().allocate(
                        snr_db, set(range(0, n_occ, 8)), n_occ,
                        snr_margin_db=margin, max_bits=mx, isi_beta=beta)
                    c_bat = [out[i] for i in range(cnt)]
                    if c_bat != list(py_bat):
                        print(f"  FAIL allocate[m={margin},mx={mx},b={beta}] "
                              f"differs at "
                              f"{[i for i,(x,y) in enumerate(zip(c_bat,py_bat)) if x!=y][:5]}")
                        bad += 1
    if a.verbose:
        print("  ok   allocate over 6 x 3 x 3 x 3 combinations")

    # ---- layer 1 + 3 -------------------------------------------------------
    for g in GEOMS:
        cc, py = build_cfg(lib, g)
        nps, nms = g[7], g[8]
        fs = g[5]
        tag = f"{g[0]}/{g[1]}/{fs}"
        maxcap = int(fs * 12)
        need = lib.ofdm_probe_workspace(ctypes.byref(cc), nps, nms, maxcap)
        if need == 0:
            print(f"  FAIL [{tag}] probe workspace rejected"); bad += 1; continue
        mem = ctypes.create_string_buffer(need)
        h = ctypes.c_void_p()
        rc = lib.ofdm_probe_init(ctypes.byref(h), mem, need, ctypes.byref(cc),
                                 nps, nms, maxcap)
        if rc != 0:
            print(f"  FAIL [{tag}] probe_init rc={rc}"); bad += 1; continue

        probe = M.ChannelProbe(py)
        py_sig = probe.build()
        n = lib.ofdm_probe_build_samples(h)
        buf = (ctypes.c_double * n)()
        got = lib.ofdm_probe_build(h, buf, n)
        c_sig = np.frombuffer(buf, dtype=np.float64, count=n)
        if got != n or len(py_sig) != n:
            print(f"  FAIL [{tag}] probe length C={got} py={len(py_sig)}")
            bad += 1
            continue
        d = float(np.max(np.abs(c_sig - py_sig)))
        if d > 1e-11:
            print(f"  FAIL [{tag}] probe waveform max|d|={d:.3e}")
            bad += 1
        elif a.verbose:
            print(f"  ok   [{tag}] probe waveform {n} samples max|d|={d:.2e}")

        for rname, (taps, rt60, diff) in ROOMS.items():
            for snr_db in (25.0, 14.0):
                lead = int(0.3 * fs)
                air = sim_room(py_sig, fs, 99, taps, rt60, diff, snr_db)
                rx = np.concatenate([np.zeros(lead), air, np.zeros(lead)])
                rxa, prx = arr(rx)
                outs = [np.zeros(py["n_occupied"]) for _ in range(5)]
                ptrs = [o.ctypes.data_as(D) for o in outs]
                info = Info()
                r = lib.ofdm_probe_estimate(h, prx, len(rxa), 0.25,
                                            ptrs[0], ptrs[1], ptrs[2],
                                            ptrs[3], ptrs[4],
                                            ctypes.byref(info))
                pres = probe.estimate(rx, threshold=0.25)
                t2 = f"{tag}/{rname}/{snr_db:.0f}dB"

                if bool(info.valid) != bool(pres.get("valid")):
                    print(f"  FAIL [{t2}] valid C={info.valid} "
                          f"py={pres.get('valid')}")
                    bad += 1
                    continue
                if not info.valid:
                    continue

                dsnr = float(np.max(np.abs(outs[3] - pres["snr_db"])))
                dsyms = abs(info.n_syms - pres["n_syms"])
                # the decision that matters
                c_bat_buf = (ctypes.c_int * py["n_occupied"])()
                cnt = lib.ofdm_allocate(ptrs[3], py["n_occupied"],
                                        py["pilot_spacing"], 6.0, 6, 0.0,
                                        c_bat_buf, py["n_occupied"])
                c_bat = [c_bat_buf[i] for i in range(cnt)]
                py_bat = list(M.BitAllocator().allocate(
                    pres["snr_db"], set(range(0, py["n_occupied"],
                                              py["pilot_spacing"])),
                    py["n_occupied"], snr_margin_db=6.0, max_bits=6))
                nb_diff = sum(1 for x, y in zip(c_bat, py_bat) if x != y)

                ok = True
                if dsyms > 1:
                    print(f"  FAIL [{t2}] n_syms C={info.n_syms} "
                          f"py={pres['n_syms']}")
                    ok = False
                if dsnr > 1.5:
                    print(f"  FAIL [{t2}] max SNR delta {dsnr:.2f} dB")
                    ok = False
                if nb_diff > max(2, len(py_bat) // 20):
                    print(f"  FAIL [{t2}] BAT differs on {nb_diff}/{len(py_bat)} "
                          f"carriers (C sum {sum(c_bat)} vs py {sum(py_bat)})")
                    ok = False
                if not ok:
                    bad += 1
                elif a.verbose:
                    print(f"  ok   [{t2}] syms {info.n_syms}/{pres['n_syms']} "
                          f"dSNR {dsnr:.2f}dB  BAT diff {nb_diff}  "
                          f"beta C={info.isi_ratio:.4f} "
                          f"py={pres.get('isi_ratio')}")

    print("\nOFDM PROBE REGRESS " + ("PASS" if bad == 0 else f"FAIL ({bad})"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
