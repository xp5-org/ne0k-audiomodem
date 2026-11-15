#!/usr/bin/env python3
"""Self-test of the complex-baseband OFDM pieces (c/ofdm_cx.c, ofdm_cfft_* in c/ofdm_fft.c) against numpy.

    ofdm_cx_selftest.py [--keep DIR]        prints one line per check, then CX PASS / CX FAIL (exit 0 / 1)

Builds the C twice into a temp dir (double, and -DOFDM_ACQ_F32 float) plus a strict -std=c89 -pedantic
compile of the new file, then grades:
  * ofdm_cfft_fwd / _inv == numpy.fft.fft / ifft for n = 4 .. 8192;
  * ofdm_cxcfg_build: signed bins, DC hole, edge cases (band touching DC, DC only, clamp at +-n/2);
  * ofdm_cxsym_make == numpy reference sample for sample (no taper and with taper);
  * make -> strip CP -> ofdm_cxsym_demod returns the carriers;
  * every occupied carrier lands at its signed frequency (a single-carrier symbol's spectrum peak);
  * reused real-path frame arithmetic gives the same counts for a cx config as for the equivalent real one;
  * ofdm_cxmod_frame == numpy frame rebuilt from the carriers the REAL ofdm_mod_frame put on the air (same
    BAT, chunk, seq), sample for sample; SC halves repeat; encode_ex gaps / NUL padding / seq wrap; init errors;
  * ofdm_cxsync_find == numpy reference (starts, even integer CFO, frac CFO / scores to 1e-9) and == truth over a
    CFO sweep (-14.6 .. 7.25 subcarriers, wrap at +-1) with a 2-path channel at 25 dB and at 6 dB; noiseless CFO
    exact; derotated CE -> last SYNC phase as the residual CFO predicts; no frames from noise, DC or CW tones;
  * ofdm_cxdem_frame == numpy reference (hard-decision wire bytes identical, EVM to 1e-9 / 1e-4) on the frames
    ofdm_cxsync finds over a CFO sweep, 2-path, 25 dB; == truth (chunks, seq numbers) there, noiseless, with a
    deliberately wrong CFO (pilot phase tracking), a 200 ppm sample-clock offset (pilot slope in signed bin)
    and starts 12 samples early / late (window margin); error codes;
  * ofdm_cxacq_run == numpy reference (float32 grid, double sums) and CFO-invariant: a periodic block scores
    >= 0.95 at CFO 0 .. -14.6 subcarriers, noise stays under the 0.70 gate;
  * ofdm_cxprobe: build == numpy (SC + CE = ofdm_cxmod's); estimate == numpy reference at the C's start / CFO
    (SNR, periodic SNR, image to 1e-8 dB) and == truth: exact start, CFO, per-carrier SNR over a CFO sweep
    (2-path, 25 dB), at 10 dB and 6 dB, a short 16+8 probe, the IQ image (transmitter side at any CFO, receiver side at CFO 0) against
    mu/nu theory, a 200 ppm sample clock; probe -> ofdm_allocate -> a frame on that BAT decodes; noise, a CW
    tone, a DC offset, a probe missing its SC, data frames are refused; cut / short / long captures.
"""
import argparse, ctypes, os, shutil, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.join(HERE, "c")
RSDIR = os.path.normpath(os.path.join(HERE, "..", "c"))     # rs255.c, as c/Makefile's -I../../c
# ofdm_sym.c + ofdm_mod.c are the REAL modulator: the oracle cxmod is graded against.
SRC = ["ofdm_fft.c", "ofdm_cfg.c", "ofdm_cx.c", "ofdm_cxmod.c", "ofdm_cxsync.c", "ofdm_cxdem.c", "ofdm_cxacq.c", "ofdm_cxprobe.c", "ofdm_cxfec.c", "ofdm_frame.c", "ofdm_sym.c", "ofdm_mod.c", "ofdm_probe.c", "ofdm_acq.c",
       os.path.join(RSDIR, "rs255.c")]
CX_NEW = ["ofdm_cx.c", "ofdm_cxmod.c", "ofdm_cxsync.c", "ofdm_cxdem.c", "ofdm_cxacq.c", "ofdm_cxprobe.c", "ofdm_cxfec.c"]
SEED_CE, SEED_SC = 0x3D5F, 0x3D5F ^ 0xFFFF
SC_TX_GAIN = 2.0


class OfdmCfg(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in (
        "n_fft", "cp_len", "bin_lo", "bin_hi", "n_occupied", "pilot_spacing",
        "n_pilots", "n_data_carriers", "sample_rate", "symbol_samples",
        "taper_samples", "frame_gap_samples", "ecc_bytes", "chunk_bytes",
        "seq_bytes", "sync_period")] + [(n, ctypes.c_double) for n in (
        "cp_ratio", "sc_spacing_hz", "freq_lo_hz", "freq_hi_hz")]


class CxCfg(ctypes.Structure):
    _fields_ = [("c", OfdmCfg), ("dc_null", ctypes.c_int), ("dc_slot", ctypes.c_int)]


class Cpx(ctypes.Structure):
    pass


class CxHit(ctypes.Structure):
    _fields_ = [("start", ctypes.c_int), ("score", ctypes.c_double), ("metric", ctypes.c_double),
                ("cfo", ctypes.c_double), ("cfo_frac", ctypes.c_double), ("cfo_int", ctypes.c_int),
                ("int_score", ctypes.c_double)]


class CxDemOut(ctypes.Structure):
    _fields_ = [("evm", ctypes.c_double), ("data_syms", ctypes.c_int), ("errata", ctypes.c_int),
                ("seq", ctypes.c_uint)]


class AcqInfo(ctypes.Structure):
    _fields_ = [("n_cand", ctypes.c_int), ("n_pos", ctypes.c_int), ("start_step", ctypes.c_int)]


class CxProbeInfo(ctypes.Structure):
    _fields_ = [("valid", ctypes.c_int), ("error", ctypes.c_int), ("coherence", ctypes.c_double),
                ("noise_floor", ctypes.c_double), ("acq_start", ctypes.c_int), ("frame_start", ctypes.c_int),
                ("cfo", ctypes.c_double), ("sc_score", ctypes.c_double), ("int_score", ctypes.c_double),
                ("n_syms", ctypes.c_int), ("n_meas_syms", ctypes.c_int), ("has_isi_penalty", ctypes.c_int),
                ("isi_penalty_db", ctypes.c_double), ("has_image", ctypes.c_int), ("image_db", ctypes.c_double)]


def build(out, f32):
    so = os.path.join(out, "libofdmcx32.so" if f32 else "libofdmcx.so")
    cmd = ["cc", "-O2", "-std=c99", "-fPIC", "-Wall", "-Wextra", "-pedantic", "-shared", "-I", CDIR, "-I", RSDIR,
           *(["-DOFDM_ACQ_F32"] if f32 else []), "-o", so, *[os.path.join(CDIR, s) for s in SRC], "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode or r.stderr.strip():
        raise SystemExit(f"build failed ({'f32' if f32 else 'f64'}):\n{r.stderr}")
    lib = ctypes.CDLL(so)
    real = ctypes.c_float if f32 else ctypes.c_double
    Cpx = type("Cpx", (ctypes.Structure,), {"_fields_": [("re", real), ("im", real)]})
    Cfft = type("Cfft", (ctypes.Structure,), {"_fields_": [("n", ctypes.c_int), ("tw", ctypes.POINTER(Cpx))]})
    lib.ofdm_cfft_workspace.restype = ctypes.c_size_t
    lib.ofdm_cxsym_workspace.restype = ctypes.c_size_t
    lib.ofdm_cxcfg_build.argtypes = [ctypes.POINTER(CxCfg), ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                     ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
    lib.ofdm_cx_bin.argtypes = [ctypes.POINTER(CxCfg), ctypes.c_int]
    lib.ofdm_known_bpsk.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_uint, ctypes.c_double]
    lib.ofdm_cfg_build.argtypes = [ctypes.POINTER(OfdmCfg), ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                   ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int]
    for f in ("ofdm_mod_workspace", "ofdm_cxmod_workspace"):
        getattr(lib, f).restype = ctypes.c_size_t
    for f in ("ofdm_mod_frame", "ofdm_mod_frame_samples", "ofdm_cxmod_frame", "ofdm_cxmod_frame_samples",
              "ofdm_cxmod_signal_samples_ex", "ofdm_cxmod_encode_ex"):
        getattr(lib, f).restype = ctypes.c_long
    for f in ("ofdm_mod_frame", "ofdm_cxmod_frame"):
        getattr(lib, f).argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_long]
    lib.ofdm_cxmod_encode_ex.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_ulong,
                                         ctypes.c_int, ctypes.c_void_p, ctypes.c_long]
    lib.ofdm_cxmod_signal_samples_ex.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    for f in ("ofdm_cxmod_frame_samples", "ofdm_mod_frame_samples", "ofdm_cxmod_data_syms",
              "ofdm_mod_data_syms", "ofdm_cxmod_body_syms", "ofdm_mod_body_syms"):
        getattr(lib, f).argtypes = [ctypes.c_void_p]
    lib.ofdm_cxmod_sc_preamble.restype = ctypes.POINTER(ctypes.c_double)
    lib.ofdm_cxmod_sc_preamble.argtypes = [ctypes.c_void_p]
    lib.ofdm_frame_min_gap.argtypes = [ctypes.POINTER(OfdmCfg), ctypes.c_int]
    lib.ofdm_cxsync_workspace.restype = ctypes.c_size_t
    lib.ofdm_cxsync_workspace.argtypes = [ctypes.POINTER(CxCfg)]
    lib.ofdm_cxsync_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_size_t,
                                     ctypes.POINTER(CxCfg), ctypes.c_void_p, ctypes.c_int]
    lib.ofdm_cxsync_find.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_double, ctypes.c_double, ctypes.POINTER(CxHit), ctypes.c_int]
    lib.ofdm_cxsync_eval.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.POINTER(CxHit)]
    lib.ofdm_cx_derotate.restype = None
    lib.ofdm_cx_derotate.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_double, ctypes.c_int, ctypes.c_long]
    lib.ofdm_cxdem_workspace.restype = ctypes.c_size_t
    lib.ofdm_cxdem_workspace.argtypes = [ctypes.POINTER(CxCfg), ctypes.c_void_p]
    lib.ofdm_cxdem_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.POINTER(CxCfg), ctypes.c_void_p]
    for f in ("ofdm_cxdem_data_syms", "ofdm_cxdem_body_syms", "ofdm_cxdem_payload_bytes"):
        getattr(lib, f).argtypes = [ctypes.c_void_p]
    lib.ofdm_cxdem_frame_samples.restype = ctypes.c_long
    lib.ofdm_cxdem_frame_samples.argtypes = [ctypes.c_void_p]
    lib.ofdm_cxdem_frame.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                                     ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(CxDemOut)]
    lib.ofdm_cxdem_wire.restype = ctypes.POINTER(ctypes.c_ubyte)
    lib.ofdm_cxdem_wire.argtypes = [ctypes.c_void_p]
    lib.ofdm_mod_table.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                                   ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                                   ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_int)]
    vp, ci, cd = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    lib.ofdm_cxacq_workspace.restype = ctypes.c_size_t
    lib.ofdm_cxacq_workspace.argtypes = [ctypes.POINTER(CxCfg), ci, ci, ci]
    lib.ofdm_cxacq_init.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_size_t, ctypes.POINTER(CxCfg), vp, ci, ci, ci]
    lib.ofdm_cxacq_run.argtypes = [vp, vp, ci, vp, ci, ctypes.POINTER(AcqInfo)]
    lib.ofdm_cxacq_n_cand.argtypes = [vp, ci]
    lib.ofdm_cxacq_ring_rows.argtypes = [vp]
    lib.ofdm_cxacq_row.argtypes = [vp, vp, ci, ci, vp]
    lib.ofdm_cxprobe_workspace.restype = ctypes.c_size_t
    lib.ofdm_cxprobe_workspace.argtypes = [ctypes.POINTER(CxCfg), ci, ci, ci, ci]
    lib.ofdm_cxprobe_init.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_size_t, ctypes.POINTER(CxCfg), ci, ci, ci, ci]
    lib.ofdm_cxprobe_build_samples.restype = ctypes.c_long
    lib.ofdm_cxprobe_build_samples.argtypes = [vp]
    lib.ofdm_cxprobe_build.restype = ctypes.c_long
    lib.ofdm_cxprobe_build.argtypes = [vp, vp, ctypes.c_long]
    lib.ofdm_cxprobe_estimate.argtypes = [vp, vp, ci, cd] + [vp] * 6 + [ctypes.POINTER(CxProbeInfo)]
    lib.ofdm_allocate.argtypes = [vp, ci, ci, cd, ci, cd, vp, ci]
    return lib, real, Cpx, Cfft


def c89_check():
    errs = []
    for f in CX_NEW:
        r = subprocess.run(["cc", "-std=c89", "-pedantic", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
                            "-Wdeclaration-after-statement", "-Wno-long-long", "-I", CDIR, "-I", RSDIR,
                            os.path.join(CDIR, f)], capture_output=True, text=True)
        if r.returncode:
            errs.append(f"{f} is not clean C89:\n" + r.stderr)
    return errs


def check_fft(lib, real, Cpx, Cfft, tol):
    errs = []
    rng = np.random.default_rng(1)
    for n in [4, 8, 16, 64, 256, 1024, 4096, 8192]:
        f = Cfft()
        mem = ctypes.create_string_buffer(lib.ofdm_cfft_workspace(n))
        if lib.ofdm_cfft_init(ctypes.byref(f), n, mem, len(mem)) != 0:
            errs.append(f"cfft_init n={n} failed"); continue
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        buf = (Cpx * n)(*[Cpx(v.real, v.imag) for v in x])
        lib.ofdm_cfft_fwd(ctypes.byref(f), buf)
        got = np.array([b.re + 1j * b.im for b in buf])
        e = np.abs(got - np.fft.fft(x)).max() / np.sqrt(n)
        if e > tol:
            errs.append(f"cfft_fwd n={n} err {e:.2e}")
        lib.ofdm_cfft_inv(ctypes.byref(f), buf)
        back = np.array([b.re + 1j * b.im for b in buf])
        e2 = np.abs(back - x).max()
        if e2 > tol:
            errs.append(f"cfft_inv n={n} round-trip err {e2:.2e}")
    bad = Cfft()
    if lib.ofdm_cfft_init(ctypes.byref(bad), 12, ctypes.create_string_buffer(64), 64) != -1:
        errs.append("cfft_init accepted n=12")
    return errs


def cxcfg(lib, n, lo, hi, fs, dc=1, ps=8, cp=0.25, taper_ms=2.0, sync=32):
    x = CxCfg()
    rc = lib.ofdm_cxcfg_build(ctypes.byref(x), n, cp, lo, hi, ps, fs, 16, 256, 2, taper_ms, 20.0, sync, dc)
    return rc, x


def bins_of(lib, x):
    return [lib.ofdm_cx_bin(ctypes.byref(x), i) for i in range(x.c.n_occupied)]


def check_geometry(lib):
    errs = []
    # 100 kHz channel at 200 ksps, n=1024: spacing 195.3125 Hz, bins -256..256 minus DC = 512 carriers
    rc, x = cxcfg(lib, 1024, -50e3, 50e3, 200000)
    b = bins_of(lib, x)
    want = [k for k in range(-256, 257) if k != 0]
    if rc or b != want or x.c.n_occupied != 512 or x.dc_slot != 256:
        errs.append(f"100k@200k: rc={rc} n_occ={x.c.n_occupied} dc_slot={x.dc_slot} bins {b[:2]}..{b[-2:]}")
    if abs(x.c.freq_lo_hz + 50000) > 1e-6 or abs(x.c.freq_hi_hz - 50000) > 1e-6:
        errs.append(f"100k@200k freqs {x.c.freq_lo_hz} {x.c.freq_hi_hz}")
    if x.c.n_pilots != 64 or x.c.n_data_carriers != 448:
        errs.append(f"100k@200k pilots {x.c.n_pilots} data {x.c.n_data_carriers}")
    # dc_null off: DC is a carrier
    rc, x = cxcfg(lib, 1024, -50e3, 50e3, 200000, dc=0)
    if rc or x.c.n_occupied != 513 or x.dc_slot != -1 or 0 not in bins_of(lib, x):
        errs.append(f"dc_null=0: n_occ={x.c.n_occupied} dc_slot={x.dc_slot}")
    # band touching DC from above / below: edge moves, no hole
    rc, x = cxcfg(lib, 64, 0.0, 1000.0, 6400)     # spacing 100 Hz -> bins 0..10 -> 1..10
    if rc or bins_of(lib, x) != list(range(1, 11)) or x.dc_slot != -1 or x.c.freq_lo_hz != 100.0:
        errs.append(f"band from DC up: {bins_of(lib, x)} dc_slot={x.dc_slot} lo={x.c.freq_lo_hz}")
    rc, x = cxcfg(lib, 64, -1000.0, 0.0, 6400)
    if rc or bins_of(lib, x) != list(range(-10, 0)) or x.c.freq_hi_hz != -100.0:
        errs.append(f"band from DC down: {bins_of(lib, x)} hi={x.c.freq_hi_hz}")
    # DC only with dc_null: nothing to carry
    rc, _ = cxcfg(lib, 64, -10.0, 10.0, 6400)
    if rc != -2:
        errs.append(f"DC-only band rc={rc}, want -2")
    # clamp: asking for the whole of +-fs/2 gives -n/2+1 .. n/2-1
    rc, x = cxcfg(lib, 64, -1e9, 1e9, 6400)
    b = bins_of(lib, x)
    if rc or b[0] != -31 or b[-1] != 31 or len(b) != 62:
        errs.append(f"clamp: rc={rc} {b[:1]}..{b[-1:]} n={len(b)}")
    # all-negative band, no DC involved
    rc, x = cxcfg(lib, 64, -2000.0, -500.0, 6400)
    if rc or bins_of(lib, x) != list(range(-20, -4)) or x.dc_slot != -1:
        errs.append(f"negative band: {bins_of(lib, x)}")
    # out-of-range index -> 0
    if lib.ofdm_cx_bin(ctypes.byref(x), x.c.n_occupied) != 0 or lib.ofdm_cx_bin(ctypes.byref(x), -1) != 0:
        errs.append("cx_bin out of range not 0")
    rc, _ = cxcfg(lib, 1000, -1, 1, 6400)
    if rc != -1:
        errs.append(f"n_fft 1000 rc={rc}, want -1")
    return errs


def ref_symbol(x, occ, use_taper):
    """numpy reference of ofdm_cxsym_make."""
    N, CP, T = x.c.n_fft, x.c.cp_len, (x.c.taper_samples if use_taper else 0)
    X = np.zeros(N, complex)
    for i, v in enumerate(occ):
        X[x._bins[i] % N] = v
    td = np.fft.ifft(X) * np.sqrt(N)
    idx = (np.arange(CP + N + 2 * T) - CP - T) % N
    y = td[idx]
    if T:
        t = (np.arange(T) + 0.5) / T
        y[:T] *= 0.5 * (1 - np.cos(np.pi * t))
        y[-T:] *= 0.5 * (1 + np.cos(np.pi * t))
    return y


def make_sym(lib, x):
    mem = ctypes.create_string_buffer(lib.ofdm_cxsym_workspace(ctypes.byref(x)))
    sp = ctypes.c_void_p()
    rc = lib.ofdm_cxsym_init(ctypes.byref(sp), mem, len(mem), ctypes.byref(x))
    return rc, sp, mem


def run_sym(lib, sp, x, occ, use_taper):
    n = x.c.n_occupied
    re = (ctypes.c_double * n)(*occ.real)
    im = (ctypes.c_double * n)(*occ.imag)
    cap = lib.ofdm_cxsym_max_len(sp)
    out = (ctypes.c_double * (2 * cap))()
    ln = lib.ofdm_cxsym_make(sp, re, im, use_taper, out)
    a = np.frombuffer(out, np.float64)[:2 * ln]
    return a[0::2] + 1j * a[1::2]


def demod(lib, sp, x, body):
    n = x.c.n_occupied
    iq = np.empty(2 * body.size)
    iq[0::2], iq[1::2] = body.real, body.imag
    ore, oim = (ctypes.c_double * n)(), (ctypes.c_double * n)()
    k = lib.ofdm_cxsym_demod(sp, iq.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ore, oim)
    return k, np.array(ore[:]) + 1j * np.array(oim[:])


def check_symbol(lib, tol):
    errs = []
    rng = np.random.default_rng(7)
    qpsk = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    for (n, lo, hi, fs, dc) in [(1024, -50e3, 50e3, 200000, 1), (256, -20e3, 35e3, 125000, 1),
                                (64, -2000.0, -500.0, 6400, 1), (1024, -50e3, 50e3, 200000, 0)]:
        rc, x = cxcfg(lib, n, lo, hi, fs, dc=dc)
        if rc:
            errs.append(f"cfg n={n} rc={rc}"); continue
        x._bins = bins_of(lib, x)
        rc, sp, mem = make_sym(lib, x)
        if rc:
            errs.append(f"cxsym_init rc={rc}"); continue
        occ = qpsk[rng.integers(0, 4, x.c.n_occupied)]
        tag = f"n={n} band {lo / 1e3:+.1f}..{hi / 1e3:+.1f} kHz dc_null={dc}"
        for taper in (0, 1):
            got = run_sym(lib, sp, x, occ, taper)
            want = ref_symbol(x, occ, taper)
            if got.size != want.size:
                errs.append(f"{tag} taper={taper}: {got.size} samples, want {want.size}"); continue
            e = np.abs(got - want).max()
            if e > tol:
                errs.append(f"{tag} taper={taper}: max err {e:.2e}")
        # level: mean |x|^2 over the body == n_occupied / n_fft for a unit-power constellation
        sym = run_sym(lib, sp, x, occ, 0)
        body = sym[x.c.cp_len:]
        p = float(np.mean(np.abs(body) ** 2))
        if abs(p / (x.c.n_occupied / n) - 1) > 1e-3 + tol:
            errs.append(f"{tag}: body power {p:.5f}, want {x.c.n_occupied / n:.5f}")
        # round trip through the demodulator
        k, back = demod(lib, sp, x, body)
        if k != x.c.n_occupied or np.abs(back - occ).max() > tol * 10:
            errs.append(f"{tag}: demod k={k} err {np.abs(back - occ).max():.2e}")
        # a fixed delay inside the CP is a per-carrier phase ramp, magnitudes untouched
        d = x.c.cp_len // 2
        k, rot = demod(lib, sp, x, sym[x.c.cp_len - d: x.c.cp_len - d + n])
        ramp = np.exp(-2j * np.pi * np.array(x._bins) * d / n)
        if np.abs(rot - occ * ramp).max() > tol * 10:
            errs.append(f"{tag}: delayed demod is not the expected phase ramp")
        # placement: one carrier at a time lands at its signed frequency
        for i in (0, x.c.n_occupied // 2, x.c.n_occupied - 1):
            one = np.zeros(x.c.n_occupied, complex)
            one[i] = 1
            b = run_sym(lib, sp, x, one, 0)[x.c.cp_len:]
            f = np.fft.fftfreq(n, 1 / fs)[np.argmax(np.abs(np.fft.fft(b)))]
            if abs(f - x._bins[i] * fs / n) > 1e-6:
                errs.append(f"{tag}: carrier {i} at {f:.1f} Hz, want {x._bins[i] * fs / n:.1f}")
    return errs


def check_reuse(lib):
    """The real-path frame arithmetic only reads counts, so a cx config with the same data-carrier count
    must give the same frame geometry as a real one."""
    errs = []
    lib.ofdm_data_syms_per_pkt.argtypes = [ctypes.POINTER(OfdmCfg), ctypes.c_int]
    rc, x = cxcfg(lib, 1024, -50e3, 50e3, 200000)
    bits = 2 * x.c.n_data_carriers
    d_cx = lib.ofdm_data_syms_per_pkt(ctypes.byref(x.c), bits)
    r = OfdmCfg()
    lib.ofdm_cfg_build.argtypes = [ctypes.POINTER(OfdmCfg), ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                   ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int]
    lib.ofdm_cfg_build(ctypes.byref(r), 1024, 0.25, 1000.0, 16000.0, 8, 48000, 16, 256, 2, 2.0, 20.0, 32)
    d_re = lib.ofdm_data_syms_per_pkt(ctypes.byref(r), bits)
    if d_cx <= 0 or d_cx != d_re:
        errs.append(f"data_syms_per_pkt cx {d_cx} != real {d_re} for {bits} bits/sym")
    return errs


def known_bpsk(lib, n, seed, amp):
    buf = (ctypes.c_double * n)()
    lib.ofdm_known_bpsk(buf, n, seed, amp)
    return np.array(buf[:])


def body_types(ds, sp):
    """'R' (CE/SYNC reference) or 'D' (data) per body symbol, in frame order."""
    t = ["R"]
    for d in range(ds):
        if sp > 0 and d % sp == 0:
            t.append("R")
        t.append("D")
    return t


def real_frame_carriers(lib, bpb, seq, chunk, sp):
    """Run the REAL modulator (itself graded sample-for-sample against ofdm_modem.py by ofdm_mod_regress) and
    read back every body symbol's occupied carriers, CE-equalised: the values the complex frame must carry."""
    r = OfdmCfg()
    # 512 @ 48 kHz, bins 11..138 = 128 occupied: the same count as the cx config below, so the pilot layout agrees
    lib.ofdm_cfg_build(ctypes.byref(r), 512, 0.25, 1000.0, 12937.5, 8, 48000, 16, 256, 2, 2.0, 20.0, sp)
    nb = (ctypes.c_int * r.n_data_carriers)(*bpb)
    mem = ctypes.create_string_buffer(lib.ofdm_mod_workspace(ctypes.byref(r), nb))
    m = ctypes.c_void_p()
    rc = lib.ofdm_mod_init(ctypes.byref(m), mem, len(mem), ctypes.byref(r), nb)
    if rc:
        raise RuntimeError(f"ofdm_mod_init rc={rc}")
    fl = lib.ofdm_mod_frame_samples(m)
    out = (ctypes.c_double * fl)()
    if lib.ofdm_mod_frame(m, seq, bytes(chunk), out, fl) != fl:
        raise RuntimeError("ofdm_mod_frame failed")
    y = np.frombuffer(out, np.float64)
    N, CP, SS = r.n_fft, r.cp_len, r.symbol_samples
    Y = np.array([np.fft.rfft(y[SS + s * SS + CP: SS + s * SS + CP + N])[r.bin_lo:r.bin_hi + 1]
                  for s in range(lib.ofdm_mod_body_syms(m))])
    ce = known_bpsk(lib, r.n_occupied, SEED_CE, 1.0)
    geo = dict(n_occ=r.n_occupied, n_data=r.n_data_carriers, ds=lib.ofdm_mod_data_syms(m),
               body=lib.ofdm_mod_body_syms(m))
    return Y / Y[0] * ce, geo, (mem, m)


def ref_cx_frame(lib, x, M):
    """numpy reference of ofdm_cxmod_frame from the carrier matrix."""
    SS, T, N = x.c.symbol_samples, x.c.taper_samples, x.c.n_fft
    ev = [i for i, b in enumerate(x._bins) if b % 2 == 0]
    occ = np.zeros(x.c.n_occupied, complex)
    occ[ev] = known_bpsk(lib, len(ev), SEED_SC, 1 / np.sqrt(2))
    sc = ref_symbol(x, occ, 0)
    body = np.zeros(len(M) * SS + 2 * T, complex)
    for s, row in enumerate(M):
        y = ref_symbol(x, row, 1)
        body[s * SS: s * SS + y.size] += y
    return np.concatenate([sc * SC_TX_GAIN, body]), sc


def cx_out(buf, n):
    a = np.frombuffer(buf, np.float64)[:2 * n]
    return a[0::2] + 1j * a[1::2]


def check_cxmod(lib, tol, cache):
    errs = []
    sp, seq = 4, 0x1234
    rng = np.random.default_rng(11)
    rc, x = cxcfg(lib, 256, -50e3, 50e3, 200000, sync=sp)         # bins -64..64 minus DC = 128 occupied
    if rc:
        return [f"cxcfg rc={rc}"]
    x._bins = bins_of(lib, x)
    if "M" not in cache:                                          # graded once from the f64 build
        cache["bpb"] = list(rng.choice([0, 1, 2, 2, 4, 4, 6], x.c.n_data_carriers))
        cache["chunk"] = rng.integers(0, 256, 256, dtype=np.uint8).tobytes()
        cache["M"], cache["geo"], _ = real_frame_carriers(lib, cache["bpb"], seq, cache["chunk"], sp)
    M, geo, chunk = cache["M"], cache["geo"], cache["chunk"]
    if geo["n_occ"] != x.c.n_occupied or geo["n_data"] != x.c.n_data_carriers:
        return [f"real oracle config has {geo['n_occ']}/{geo['n_data']} carriers, cx {x.c.n_occupied}/{x.c.n_data_carriers}"]
    nb = (ctypes.c_int * x.c.n_data_carriers)(*cache["bpb"])

    # init errors, as ofdm_mod_init
    bad = (ctypes.c_int * x.c.n_data_carriers)(*([0] * x.c.n_data_carriers))
    tmp = ctypes.create_string_buffer(1 << 20)
    m = ctypes.c_void_p()
    if lib.ofdm_cxmod_init(ctypes.byref(m), tmp, len(tmp), ctypes.byref(x), bad) != -5:
        errs.append("empty BAT not rejected with -5")
    bad[0] = 3
    if lib.ofdm_cxmod_init(ctypes.byref(m), tmp, len(tmp), ctypes.byref(x), bad) != -4:
        errs.append("width 3 not rejected with -4")
    need = lib.ofdm_cxmod_workspace(ctypes.byref(x), nb)
    if lib.ofdm_cxmod_init(ctypes.byref(m), tmp, need - 1, ctypes.byref(x), nb) != -2:
        errs.append("short workspace not rejected with -2")

    mem = ctypes.create_string_buffer(need)
    rc = lib.ofdm_cxmod_init(ctypes.byref(m), mem, need, ctypes.byref(x), nb)
    if rc:
        return errs + [f"ofdm_cxmod_init rc={rc}"]
    ds, body = lib.ofdm_cxmod_data_syms(m), lib.ofdm_cxmod_body_syms(m)
    SS, T, N, CP = x.c.symbol_samples, x.c.taper_samples, x.c.n_fft, x.c.cp_len
    fl = lib.ofdm_cxmod_frame_samples(m)
    if ds != geo["ds"] or body != geo["body"] or len(M) != body:
        errs.append(f"geometry: data_syms {ds} body {body}, real {geo['ds']} {geo['body']}")
    if fl != SS + body * SS + 2 * T:
        errs.append(f"frame_samples {fl}, want {SS + body * SS + 2 * T}")
    types = body_types(ds, sp)
    ce = known_bpsk(lib, x.c.n_occupied, SEED_CE, 1.0)
    if types.count("R") < 3 or max(np.abs(M[s] - ce).max() for s, t in enumerate(types) if t == "R") > 1e-9:
        errs.append("real oracle: CE/SYNC symbols are not the known BPSK (layout assumption broken)")

    # the frame, sample for sample
    out = (ctypes.c_double * (2 * fl))()
    if lib.ofdm_cxmod_frame(m, seq, chunk, out, fl - 1) != -2:
        errs.append("short out_cap not rejected with -2")
    if lib.ofdm_cxmod_frame(m, seq, chunk, out, fl) != fl:
        return errs + ["ofdm_cxmod_frame failed"]
    got = cx_out(out, fl)
    want, sc = ref_cx_frame(lib, x, M)
    e = np.abs(got - want).max()
    if got.size != want.size or e > tol:
        errs.append(f"frame vs numpy (real-oracle carriers): max err {e:.2e} over {got.size} samples")
    pre = np.frombuffer(ctypes.cast(lib.ofdm_cxmod_sc_preamble(m), ctypes.POINTER(ctypes.c_double * (2 * SS))).contents)
    if np.abs((pre[0::2] + 1j * pre[1::2]) - sc).max() > tol:
        errs.append("sc_preamble is not the unscaled SC symbol")
    # SC body: two identical halves (even signed bins), the property step 5's CFO estimate needs
    b = got[CP:CP + N]
    if np.abs(b[:N // 2] - b[N // 2:]).max() > tol:
        errs.append("SC preamble halves differ")
    # level: data symbols carry ~ the real path's power per carrier
    p = float(np.mean(np.abs(got[SS:]) ** 2))
    if not 0.2 < p < 1.0:
        errs.append(f"body mean power {p:.3f} implausible")

    # encode_ex: gaps, NUL padding, seq wrap 0xFFFF -> 0
    data = bytes(rng.integers(0, 256, 300, dtype=np.uint8))
    G = x.c.frame_gap_samples
    for gap in (1, 0):
        n = lib.ofdm_cxmod_signal_samples_ex(m, len(data), gap)
        if n != 2 * fl + (3 * G if gap else 0):
            errs.append(f"signal_samples_ex gap={gap}: {n}"); continue
        buf = (ctypes.c_double * (2 * n))(*([7.0] * (2 * n)))
        if lib.ofdm_cxmod_encode_ex(m, data, len(data), 0xFFFF, gap, buf, n - 1) != -2:
            errs.append(f"encode_ex gap={gap}: short cap not -2")
        if lib.ofdm_cxmod_encode_ex(m, data, len(data), 0xFFFF, gap, buf, n) != n:
            errs.append(f"encode_ex gap={gap} failed"); continue
        sig = cx_out(buf, n)
        g = G if gap else 0
        f0, f1 = (ctypes.c_double * (2 * fl))(), (ctypes.c_double * (2 * fl))()
        lib.ofdm_cxmod_frame(m, 0xFFFF, data[:256], f0, fl)
        lib.ofdm_cxmod_frame(m, 0, data[256:] + bytes(212), f1, fl)
        want = np.concatenate([np.zeros(g), cx_out(f0, fl), np.zeros(g), cx_out(f1, fl), np.zeros(g)])
        if np.abs(sig - want).max() != 0:
            errs.append(f"encode_ex gap={gap} != gap+frame(0xFFFF)+gap+frame(0, NUL-padded)+gap")
    return errs


# ---------------------------------------------------------------- step 5: acquisition + CFO

SC_THRESHOLD, SC_NCC_MIN, SC_REPLACE_RATIO, CX_INT_MIN = 0.25, 0.10, 2.0, 0.35


def ref_cxsync(rx, x, sc, ce, bins, min_gap, thr, ncc_min, max_int):
    """numpy reference of ofdm_cxsync_find (see c/ofdm_cxsync.h for the algorithm): the SC metric from
    cumulative sums, the same region / leading-edge / min_gap / NCC / replace selection, then per candidate
    fractional CFO = arg(P)/pi mid-plateau, even integer CFO from the CE's differential correlation, and the
    matched filter against the CFO-rotated reference.  Returns a list of hit dicts."""
    N, CP, SS = x.c.n_fft, x.c.cp_len, x.c.symbol_samples
    L, n = N // 2, rx.size
    ref = sc - sc.mean()
    ref = ref / np.linalg.norm(ref)
    bins = np.array(bins) % N
    md = n - 2 * L
    if md < 1:
        return []
    cp = np.concatenate([[0], np.cumsum(np.conj(rx[:n - L]) * rx[L:])])
    ce_ = np.concatenate([[0], np.cumsum(np.abs(rx) ** 2)])
    P = cp[L:L + md] - cp[:md]
    R1, R2 = ce_[L:L + md] - ce_[:md], ce_[2 * L:2 * L + md] - ce_[L:L + md]
    den = R1 * R2
    M = np.where(den <= 1e-18, 0.0, np.abs(P) ** 2 / (den + 1e-24))

    def ev(d):
        pf = min(d + CP // 2, n - 2 * L)
        p = np.vdot(rx[pf:pf + L], rx[pf + L:pf + 2 * L])
        frac = 0.0 if p == 0 else float(np.angle(p)) / np.pi
        q, ds = 0, -1.0
        c0 = d + SS + CP // 2
        if c0 + N <= n:
            t = np.arange(c0, c0 + N)
            Y = np.fft.fft(rx[t] * np.exp(-2j * np.pi * frac * t / N))
            for k in range(max_int // 2 + 1):
                for qq in ([0] if k == 0 else [2 * k, -2 * k]):
                    a = Y[(bins + qq) % N] * ce
                    z = np.conj(a[:-1]) * a[1:]
                    v = abs((z / np.abs(z)).sum()) / z.size                   # phase-only
                    if v > ds:
                        ds, q = v, qq
        cfo = frac + q
        refc = ref * np.exp(2j * np.pi * cfo * np.arange(SS) / N)
        lo, hi = max(d - CP, 0), min(d + CP, n - SS)
        W = np.lib.stride_tricks.sliding_window_view(rx, SS)[lo:hi + 1]
        W = W - W.mean(1, keepdims=True)                           # DC-proof: the rotated ref is not zero-mean
        v = np.abs(W @ np.conj(refc)) / (np.sqrt((np.abs(W) ** 2).sum(1)) + 1e-12)
        i = int(np.argmax(v))
        return dict(start=lo + i, score=float(v[i]), cfo=cfo, cfo_frac=frac, cfo_int=q, int_score=ds)

    hits, last, in_reg, last_sc, last_rmax = [], -2 * min_gap, False, 0.0, 0.0
    for d in range(md):
        m_ = M[d]
        if not in_reg:
            if m_ > thr:
                in_reg, r0, rend, rmax = True, d, min(d + SS, md), m_
        elif m_ > rmax:
            rmax = m_
        if in_reg and d + 1 >= rend:
            in_reg = False
            seg = M[r0:rend]
            k = np.nonzero(seg >= 0.95 * rmax)[0]
            ref_pos = r0 + (int(k[0]) if k.size else 0)
            inside = ref_pos - last < min_gap      # edge vs edge (ofdm_cxsync.c says why)
            if not inside or (hits and rmax > last_rmax):
                h = ev(ref_pos)
                if h["score"] >= ncc_min and h["int_score"] >= CX_INT_MIN:
                    h["metric"] = rmax
                    if not inside:
                        hits.append(h)
                        last, last_sc, last_rmax = ref_pos, h["score"], rmax
                    elif h["score"] >= SC_REPLACE_RATIO * last_sc:
                        hits[-1] = h
                        last, last_sc, last_rmax = ref_pos, h["score"], rmax
    return hits


def cx_in(z):
    a = np.empty(2 * z.size)
    a[0::2], a[1::2] = z.real, z.imag
    return (ctypes.c_double * a.size)(*a)


def make_cxmod(lib, x, bpb):
    nb = (ctypes.c_int * x.c.n_data_carriers)(*bpb)
    need = lib.ofdm_cxmod_workspace(ctypes.byref(x), nb)
    mem = ctypes.create_string_buffer(need)
    m = ctypes.c_void_p()
    rc = lib.ofdm_cxmod_init(ctypes.byref(m), mem, need, ctypes.byref(x), nb)
    if rc:
        raise RuntimeError(f"ofdm_cxmod_init rc={rc}")
    return m, mem


def c_find(lib, s, rx, nbody, max_hits=16):
    hits = (CxHit * max_hits)()
    buf = cx_in(rx)
    k = lib.ofdm_cxsync_find(s, buf, rx.size, nbody, SC_THRESHOLD, SC_NCC_MIN, hits, max_hits)
    return [hits[i] for i in range(max(k, 0))], k


def check_cxsync(lib, tol, ids_tol, cache):
    """Step 5: frame acquisition + CFO on complex samples, C against the numpy reference and against truth."""
    errs = []
    sp, MAXI = 4, 16
    rc, x = cxcfg(lib, 256, -50e3, 50e3, 200000, sync=sp)
    if rc or "bpb" not in cache:
        return [f"cxcfg rc={rc}" if rc else "no BAT cached (cxmod check did not run first)"]
    N, CP, SS, T, G = x.c.n_fft, x.c.cp_len, x.c.symbol_samples, x.c.taper_samples, x.c.frame_gap_samples
    m, _mm = make_cxmod(lib, x, cache["bpb"])
    nbody, ds = lib.ofdm_cxmod_body_syms(m), lib.ofdm_cxmod_data_syms(m)
    fl = lib.ofdm_cxmod_frame_samples(m)
    pre = np.frombuffer(ctypes.cast(lib.ofdm_cxmod_sc_preamble(m), ctypes.POINTER(ctypes.c_double * (2 * SS))).contents)
    sc = pre[0::2] + 1j * pre[1::2]
    ce = known_bpsk(lib, x.c.n_occupied, SEED_CE, 1.0)
    bins = bins_of(lib, x)
    min_gap = lib.ofdm_frame_min_gap(ctypes.byref(x.c), nbody)

    # init errors
    need = lib.ofdm_cxsync_workspace(ctypes.byref(x))
    s = ctypes.c_void_p()
    tmp = ctypes.create_string_buffer(need)
    if lib.ofdm_cxsync_init(ctypes.byref(s), tmp, need, ctypes.byref(x), pre.ctypes.data, -1) != -1:
        errs.append("max_cfo_bins -1 not rejected with -1")
    if lib.ofdm_cxsync_init(ctypes.byref(s), tmp, need - 1, ctypes.byref(x), pre.ctypes.data, MAXI) != -2:
        errs.append("short workspace not rejected with -2")
    smem = ctypes.create_string_buffer(need)
    if lib.ofdm_cxsync_init(ctypes.byref(s), smem, need, ctypes.byref(x), pre.ctypes.data, MAXI):
        return errs + ["ofdm_cxsync_init failed"]

    # three frames with gaps, as ofdm_cxmod_encode_ex puts them on the air
    rng = np.random.default_rng(55)
    data = bytes(rng.integers(0, 256, 3 * 256 - 40, dtype=np.uint8))
    ns = lib.ofdm_cxmod_signal_samples_ex(m, len(data), 1)
    buf = (ctypes.c_double * (2 * ns))()
    if lib.ofdm_cxmod_encode_ex(m, data, len(data), 7, 1, buf, ns) != ns:
        return errs + ["encode_ex failed"]
    sig = cx_out(buf, ns)
    D = 1237                                                     # arbitrary lead-in, not symbol-aligned
    truth = [D + G + k * (fl + G) for k in range(3)]
    psig = float(np.mean(np.abs(sig[G:G + fl]) ** 2))
    types = body_types(ds, sp)
    last_r = max(i for i, t in enumerate(types) if t == "R")
    _, sp_, _spm = make_sym(lib, x)

    worst = dict(cfo=0.0, dt=0, ref=0.0, ids=0.0, drift=0.0)
    for eps in (0.0, 0.37, -0.81, 0.999, -1.0, 2.5, -3.3, 7.25, -14.6):
        phi = rng.uniform(-np.pi, np.pi)
        y = np.concatenate([np.zeros(D, complex), sig, np.zeros(500, complex)])
        y = y + 0.3j * np.concatenate([np.zeros(3, complex), y[:-3]])           # 2-path channel
        t = np.arange(y.size)
        y = y * np.exp(1j * (2 * np.pi * eps * t / N + phi))
        y = y + (rng.standard_normal(y.size) + 1j * rng.standard_normal(y.size)) * np.sqrt(psig / 10 ** 2.5 / 2)

        got, k = c_find(lib, s, y, nbody)
        want = ref_cxsync(y, x, sc, ce, bins, min_gap, SC_THRESHOLD, SC_NCC_MIN, MAXI)
        if k != 3 or len(want) != 3:
            errs.append(f"eps {eps}: C found {k} frames, numpy {len(want)}, want 3"); continue
        for i, (h, w) in enumerate(zip(got, want)):
            if h.start != w["start"] or h.cfo_int != w["cfo_int"]:
                errs.append(f"eps {eps} frame {i}: C start {h.start} int {h.cfo_int}, numpy {w['start']} {w['cfo_int']}")
            for f in ("cfo", "cfo_frac", "score", "metric"):
                worst["ref"] = max(worst["ref"], abs(getattr(h, f) - w[f]))
            worst["ids"] = max(worst["ids"], abs(h.int_score - w["int_score"]))   # through the (f32) FFT
            worst["cfo"] = max(worst["cfo"], abs(h.cfo - eps))
            worst["dt"] = max(worst["dt"], abs(h.start - truth[i]))
            if h.cfo_int % 2 or not -1.0 < h.cfo_frac <= 1.0 or h.int_score < 0.8 or h.score < 0.5:
                errs.append(f"eps {eps} frame {i}: frac {h.cfo_frac:.3f} int {h.cfo_int} D {h.int_score:.2f} ncc {h.score:.2f}")
        # what the demodulator will see: derotate with the estimate; the CE and the frame's last SYNC must
        # then agree in phase (residual CFO * the time between them), where before they differ by 2 pi eps dt/N
        h = got[1]
        cb = cx_in(y)
        lib.ofdm_cx_derotate(cb, y.size, h.cfo, N, 0)
        z = cx_out(cb, y.size)
        ys = [demod(lib, sp_, x, z[h.start + SS + j * SS + CP: h.start + SS + j * SS + CP + N])[1]
              for j in (0, last_r)]
        # the drift must be exactly what the residual CFO predicts (sign and derotation right), and with the
        # estimate left in it would be 2 pi eps dt / N
        pred = 2 * np.pi * (eps - h.cfo) * last_r * SS / N
        drift = abs(np.angle(np.vdot(ys[0] * ce, ys[1] * ce) * np.exp(-1j * pred)))
        worst["drift"] = max(worst["drift"], drift)
    if worst["cfo"] > 0.01:
        errs.append(f"CFO estimate off by up to {worst['cfo']:.4f} subcarriers at 25 dB SNR")
    if worst["dt"] > 2:
        errs.append(f"frame start off by up to {worst['dt']} samples")
    if worst["ref"] > tol:
        errs.append(f"C vs numpy reference: max diff {worst['ref']:.2e}")
    if worst["ids"] > ids_tol:
        errs.append(f"C vs numpy integer-CFO score: max diff {worst['ids']:.2e}")
    if worst["drift"] > 0.05:
        errs.append(f"after derotation the CE->last SYNC phase is {worst['drift']:.3f} rad off the residual-CFO prediction")
    cache.setdefault("cxsync_info", []).append(
        f"cfo err {worst['cfo']:.4f} sc, start err {worst['dt']}, CE->SYNC phase vs prediction {worst['drift']:.3f} rad")

    # low SNR (6 dB, frame-average): still found, integer part still right
    lo = []
    for eps in (1.7, -6.4):
        y = np.concatenate([np.zeros(D, complex), sig, np.zeros(500, complex)])
        y = y * np.exp(2j * np.pi * eps * np.arange(y.size) / N)
        y = y + (rng.standard_normal(y.size) + 1j * rng.standard_normal(y.size)) * np.sqrt(psig / 10 ** 0.6 / 2)
        got, k = c_find(lib, s, y, nbody)
        lo += [h.int_score for h in got]
        if k != 3 or max(abs(h.cfo - eps) for h in got) > 0.05 or max(abs(h.start - t) for h, t in zip(got, truth)) > 4:
            errs.append(f"6 dB eps {eps}: {k} hits, cfo {[round(h.cfo, 3) for h in got]}, "
                        f"starts {[h.start - t for h, t in zip(got, truth)]}")
    cache.setdefault("cxsync_info", []).append(f"6 dB: integer-CFO score D {min(lo, default=0):.2f}..{max(lo, default=0):.2f}")

    # noiseless: exact
    eps = 5.123
    y = np.concatenate([np.zeros(D, complex), sig])
    y = y * np.exp(2j * np.pi * eps * np.arange(y.size) / N)
    got, k = c_find(lib, s, y, nbody)
    if k != 3 or max(abs(h.cfo - eps) for h in got) > 1e-9 or [h.start for h in got] != truth:
        errs.append(f"noiseless eps {eps}: {k} hits, starts {[h.start for h in got]} vs {truth}, "
                    f"cfo {[round(h.cfo, 12) for h in got]}")

    # back to back (encode gap=0, the throughput mode): every frame, C and numpy.  The stride is only
    # cp_len over min_gap; comparing the threshold edge with the last accepted edge lost 1 in 4 here.
    nb = 12
    datb = bytes(rng.integers(0, 256, nb * 256 - 40, dtype=np.uint8))
    nsb = lib.ofdm_cxmod_signal_samples_ex(m, len(datb), 0)
    bufb = (ctypes.c_double * (2 * nsb))()
    if lib.ofdm_cxmod_encode_ex(m, datb, len(datb), 7, 0, bufb, nsb) != nsb or nsb != nb * fl:
        errs.append(f"gapless encode_ex: {nsb} samples, want {nb * fl}")
    else:
        sb = cx_out(bufb, nsb)
        tb = [D + k * fl for k in range(nb)]
        for epb in (0.0, -3.3, 7.25):
            yb = np.concatenate([np.zeros(D, complex), sb, np.zeros(500, complex)])
            yb = yb * np.exp(2j * np.pi * epb * np.arange(yb.size) / N)
            yb = yb + (rng.standard_normal(yb.size) + 1j * rng.standard_normal(yb.size)) * np.sqrt(psig / 10 ** 2.5 / 2)
            got, k = c_find(lib, s, yb, nbody)
            want = ref_cxsync(yb, x, sc, ce, bins, min_gap, SC_THRESHOLD, SC_NCC_MIN, MAXI)
            if k != nb or len(want) != nb or max(abs(h.start - t) for h, t in zip(got, tb)) > 2 \
                    or [h.start for h in got] != [w["start"] for w in want]:
                errs.append(f"gapless eps {epb}: C {k} hits, numpy {len(want)}, want {nb}; "
                            f"start errs {[h.start - t for h, t in zip(got, tb)]}")

    # the CE cut off: integer part not measured, flagged
    cut = y[:truth[0] + SS + CP]
    hh = CxHit()
    if lib.ofdm_cxsync_eval(s, cx_in(cut), cut.size, truth[0], ctypes.byref(hh)) != 0 \
            or hh.int_score != -1.0 or hh.cfo_int != 0 or abs(hh.cfo_frac - (eps - 6)) > 1e-9:
        errs.append(f"truncated CE: rc/int_score {hh.int_score} int {hh.cfo_int} frac {hh.cfo_frac}")

    # noise only (and a lone DC offset, the LO leak): nothing
    nz = (rng.standard_normal(30000) + 1j * rng.standard_normal(30000)) * 0.01
    tone = 0.5 * np.exp(2j * np.pi * 0.1234 * np.arange(30000))
    for name, z in (("AWGN", nz * 10), ("DC + AWGN", 0.5 + nz), ("CW tone + AWGN", tone + nz),
                    ("CW tone + DC + AWGN", tone + 0.3 + nz)):
        _, k = c_find(lib, s, z, nbody)
        if k != 0:
            errs.append(f"{name}: {k} false frames")

    # derotate: continuous phase across blocks
    z = np.exp(1j * rng.uniform(0, 6, 1000))
    a, b = cx_in(z), cx_in(z)
    lib.ofdm_cx_derotate(a, 1000, -2.7, N, 123456)
    lib.ofdm_cx_derotate(b, 400, -2.7, N, 123456)
    lib.ofdm_cx_derotate(ctypes.cast(ctypes.addressof(b) + 800 * 8, ctypes.c_void_p), 600, -2.7, N, 123856)
    w = z * np.exp(2j * np.pi * 2.7 * (np.arange(1000) + 123456) / N)
    if np.abs(cx_out(a, 1000) - w).max() > 1e-9 or np.abs(cx_out(b, 1000) - w).max() > 1e-9:
        errs.append("ofdm_cx_derotate wrong or not block-continuous")
    return errs


# ---------------------------------------------------------------- step 6: demodulator

PILOT = (1 + 1j) / np.sqrt(2)


def mod_table(lib, b):
    pr, pi, bt, n = (ctypes.POINTER(ctypes.c_double)(), ctypes.POINTER(ctypes.c_double)(),
                     ctypes.POINTER(ctypes.c_ubyte)(), ctypes.c_int())
    if lib.ofdm_mod_table(b, ctypes.byref(pr), ctypes.byref(pi), ctypes.byref(bt), ctypes.byref(n)):
        raise RuntimeError(f"no table for {b}")
    k = n.value
    return np.array(pr[:k]) + 1j * np.array(pi[:k]), np.array(bt[:k * b], np.uint8).reshape(k, b)


def prbs_data(n):
    out, st = np.empty(n, np.uint8), 0xACE1
    for i in range(n):
        out[i] = (st >> 15) & 1
        fb = ((st >> 15) ^ (st >> 14) ^ (st >> 12) ^ (st >> 3)) & 1
        st = ((st << 1) | fb) & 0xFFFF
    return out


def ref_cxdem(rx, x, start, cfo, bpb, ds, wire_len, tabs, bins, ce):
    """numpy reference of ofdm_cxdem_frame up to the descrambled wire bytes (see c/ofdm_cxdem.h).
    Returns (wire bytes, mean EVM, data symbols)."""
    N, CP, SS, T, O, ps, sp = (x.c.n_fft, x.c.cp_len, x.c.symbol_samples, x.c.taper_samples, x.c.n_occupied,
                               x.c.pilot_spacing, x.c.sync_period)
    w0 = T + CP // 2 if 2 * T <= CP else CP
    bins = np.array(bins)
    fidx = bins % N
    pil = np.arange(0, O, ps)
    pb = bins[pil]
    dat = [i for i in range(O) if i % ps]
    nsync = -(-ds // sp) if sp > 0 else 0

    def win(pos):
        t = np.arange(pos + w0, pos + w0 + N)
        ph = -2 * np.pi / N * np.fmod(cfo * (t - start), N)
        return np.fft.fft(rx[t] * np.exp(1j * ph))[fidx] / np.sqrt(N)

    H = win(start + SS) / ce
    pos, bits, evms = start + 2 * SS, [], []
    for s in range(ds + nsync):
        Y = win(pos)
        pos += SS
        if sp > 0 and s % (sp + 1) == 0:
            H = 0.5 * (H + Y / ce) if s == 0 else Y / ce
            continue
        r = Y[pil] / (H[pil] * PILOT)
        w = np.abs(H[pil]) ** 2
        gap = int(np.diff(pb).min())
        k = np.nonzero(np.diff(pb) == gap)[0]
        acc = (w[k + 1] * w[k] * r[k + 1] * np.conj(r[k])).sum()
        b = float(np.angle(acc)) / gap if acc != 0 else 0.0
        e = np.angle(r * np.exp(-1j * b * pb))
        a = float(np.angle((w * np.exp(1j * e)).sum()))
        err = ref = 0.0
        cnt = 0
        for j, i in enumerate(dat):
            bv = bpb[j]
            if bv == 0:
                continue
            pts, bt = tabs[bv]
            eq = Y[i] / (H[i] * np.exp(1j * (a + b * bins[i])))
            dd = np.abs(eq - pts) ** 2
            q = int(np.argmin(dd))
            bits.extend(bt[q])
            err += dd[q]
            ref += abs(pts[q]) ** 2
            cnt += 1
        evms.append(np.sqrt((err / cnt) / (ref / cnt + 1e-20)))
    bits = np.array(bits, np.uint8) ^ prbs_data(len(bits))
    return np.packbits(bits)[:wire_len].tobytes(), float(np.mean(evms)), len(evms)


def make_cxdem(lib, x, bpb):
    nb = (ctypes.c_int * x.c.n_data_carriers)(*bpb)
    need = lib.ofdm_cxdem_workspace(ctypes.byref(x), nb)
    mem = ctypes.create_string_buffer(need)
    d = ctypes.c_void_p()
    rc = lib.ofdm_cxdem_init(ctypes.byref(d), mem, need, ctypes.byref(x), nb)
    return rc, d, mem


def c_dem(lib, d, rx, start, cfo, cap=256, buf=None):
    out = ctypes.create_string_buffer(max(cap, 1))
    info = CxDemOut()
    rc = lib.ofdm_cxdem_frame(d, buf if buf is not None else cx_in(rx), rx.size, start, cfo, out, cap,
                              ctypes.byref(info))
    return rc, out.raw[:cap], info


def check_cxdem(lib, tol, tag, cache):
    """Step 6: demodulate + RS on complex samples, C against the numpy reference and against truth."""
    errs = []
    sp, MAXI = 4, 16
    # taper 16 = CP/4, not the 32 (CP/2 clamp) the cxmod/cxsync checks use: the window then has 16 samples of
    # ISI-free margin either side for the echo and an early start (see c/ofdm_cxdem.h, step 1)
    rc, x = cxcfg(lib, 256, -50e3, 50e3, 200000, sync=sp, taper_ms=0.08)
    if rc or "bpb" not in cache or x.c.taper_samples != 16:
        return [f"cxcfg rc={rc} taper {x.c.taper_samples}" if rc or "bpb" in cache
                else "no BAT cached (cxmod check did not run first)"]
    bpb = cache["bpb"]
    N, SS, G = x.c.n_fft, x.c.symbol_samples, x.c.frame_gap_samples
    m, _mm = make_cxmod(lib, x, bpb)
    fl = lib.ofdm_cxmod_frame_samples(m)
    pre = np.frombuffer(ctypes.cast(lib.ofdm_cxmod_sc_preamble(m), ctypes.POINTER(ctypes.c_double * (2 * SS))).contents)
    ce = known_bpsk(lib, x.c.n_occupied, SEED_CE, 1.0)
    bins = bins_of(lib, x)
    tabs = {b: mod_table(lib, b) for b in set(bpb) if b}
    nbody = lib.ofdm_cxmod_body_syms(m)

    # init errors, as ofdm_cxmod_init
    nd = x.c.n_data_carriers
    zero = [0] * nd
    if make_cxdem(lib, x, zero)[0] != -5:
        errs.append("empty BAT not rejected with -5")
    if make_cxdem(lib, x, [3] + list(bpb[1:]))[0] != -4:
        errs.append("width 3 not rejected with -4")
    nb = (ctypes.c_int * nd)(*bpb)
    need = lib.ofdm_cxdem_workspace(ctypes.byref(x), nb)
    tmp, dd = ctypes.create_string_buffer(need), ctypes.c_void_p()
    if lib.ofdm_cxdem_init(ctypes.byref(dd), tmp, need - 1, ctypes.byref(x), nb) != -2:
        errs.append("short workspace not rejected with -2")
    rc, d, _dm = make_cxdem(lib, x, bpb)
    if rc:
        return errs + [f"ofdm_cxdem_init rc={rc}"]
    ds, wl = lib.ofdm_cxdem_data_syms(d), lib.ofdm_cxdem_payload_bytes(d)
    if (ds, lib.ofdm_cxdem_body_syms(d), wl) != (lib.ofdm_cxmod_data_syms(m), nbody, lib.ofdm_cxmod_payload_bytes(m)):
        errs.append("geometry differs from ofdm_cxmod's")
    if not 0 < lib.ofdm_cxdem_frame_samples(d) <= fl:
        errs.append(f"frame_samples {lib.ofdm_cxdem_frame_samples(d)} > cxmod's {fl}")
    smem = ctypes.create_string_buffer(lib.ofdm_cxsync_workspace(ctypes.byref(x)))
    s = ctypes.c_void_p()
    if lib.ofdm_cxsync_init(ctypes.byref(s), smem, len(smem), ctypes.byref(x), pre.ctypes.data, MAXI):
        return errs + ["ofdm_cxsync_init failed"]

    rng = np.random.default_rng(66)
    data = bytes(rng.integers(0, 256, 3 * 256 - 40, dtype=np.uint8))
    chunks = [data[i:i + 256].ljust(256, b"\0") for i in range(0, len(data), 256)]
    ns = lib.ofdm_cxmod_signal_samples_ex(m, len(data), 1)
    buf = (ctypes.c_double * (2 * ns))()
    if lib.ofdm_cxmod_encode_ex(m, data, len(data), 0xFFFE, 1, buf, ns) != ns:
        return errs + ["encode_ex failed"]
    sig = cx_out(buf, ns)
    seqs = [0xFFFE, 0xFFFF, 0]
    D = 1237
    truth = [D + G + k * (fl + G) for k in range(3)]
    psig = float(np.mean(np.abs(sig[G:G + fl]) ** 2))

    def channel(eps, snr_db, echo=True, z=None):
        y = np.concatenate([np.zeros(D, complex), sig if z is None else z, np.zeros(500, complex)])
        if echo:
            y = y + 0.3j * np.concatenate([np.zeros(3, complex), y[:-3]])
        y = y * np.exp(1j * (2 * np.pi * eps * np.arange(y.size) / N + rng.uniform(-np.pi, np.pi)))
        if snr_db is not None:
            y = y + (rng.standard_normal(y.size) + 1j * rng.standard_normal(y.size)) * np.sqrt(psig / 10 ** (snr_db / 10) / 2)
        return y

    def frames(y, starts_cfos, label, ref=False):
        """Demodulate each (start, cfo); return (worst evm, worst errata)."""
        cb = cx_in(y)
        we, wr = 0.0, 0
        for i, (st, cf) in enumerate(starts_cfos):
            rc, ch, info = c_dem(lib, d, y, st, cf, buf=cb)
            if rc or ch != chunks[i] or info.seq != seqs[i] or info.data_syms != ds:
                errs.append(f"{label} frame {i}: rc {rc} seq {info.seq:#x} chunk {'ok' if ch == chunks[i] else 'WRONG'} "
                            f"syms {info.data_syms} evm {info.evm:.3f}")
                continue
            we, wr = max(we, info.evm), max(wr, info.errata)
            if ref:
                w, evm, nds = ref_cxdem(y, x, st, cf, bpb, ds, wl, tabs, bins, ce)
                cw = bytes(lib.ofdm_cxdem_wire(d)[:wl])
                if cw != w:
                    nbad = sum(a != b for a, b in zip(cw, w))
                    errs.append(f"{label} frame {i}: wire bytes differ from numpy in {nbad} of {wl}")
                worst["ref"] = max(worst["ref"], abs(info.evm - evm))
        return we, wr

    # against numpy and truth: the frames ofdm_cxsync finds, its start and CFO
    worst = dict(ref=0.0, evm=0.0, err=0)
    for eps in (0.0, 0.37, -0.81, 0.999, -3.3, 7.25, -14.6):
        y = channel(eps, 25)
        got, k = c_find(lib, s, y, nbody)
        if k != 3:
            errs.append(f"eps {eps}: cxsync found {k} frames"); continue
        we, wr = frames(y, [(h.start, h.cfo) for h in got], f"25 dB eps {eps}", ref=True)
        worst["evm"], worst["err"] = max(worst["evm"], we), max(worst["err"], wr)
    if worst["ref"] > tol:
        errs.append(f"EVM C vs numpy: max diff {worst['ref']:.2e}")
    if worst["evm"] > 0.09:
        errs.append(f"25 dB: EVM up to {worst['evm']:.3f} (> 0.09)")
    info = [f"{tag} 25 dB 2-path, cxsync start+CFO: evm <= {worst['evm']:.3f}, RS errata <= {worst['err']}"]

    # noiseless, exact start and CFO: clean
    y = channel(5.123, None, echo=False)
    we, wr = frames(y, [(t, 5.123) for t in truth], "noiseless")
    if wr or we > (1e-9 if tag == "f64" else 1e-4):
        errs.append(f"noiseless: evm {we:.2e}, errata {wr}")

    # CFO estimate deliberately 0.03 subcarriers off: the pilots must track ~0.24 rad/symbol
    y = channel(-2.2, 25)
    we, wr = frames(y, [(t, -2.2 + 0.03) for t in truth], "CFO error 0.03")
    if we > 0.15:
        errs.append(f"CFO error 0.03: EVM {we:.3f} (> 0.15)")
    info.append(f"{tag} CFO given 0.03 sc wrong: evm {we:.3f}, errata {wr}")

    # 200 ppm slow sample clock: a timing drift of ~2.4 samples over a frame, a phase slope in bin
    from scipy.signal import resample_poly
    z = resample_poly(sig, 5000, 5001)
    y = channel(1.4, 25, z=z)
    got, k = c_find(lib, s, y, nbody)
    if k != 3:
        errs.append(f"200 ppm: cxsync found {k} frames")
    else:
        we, wr = frames(y, [(h.start, h.cfo) for h in got], "200 ppm SRO")
        if we > 0.09:
            errs.append(f"200 ppm SRO: EVM {we:.3f} (> 0.09)")
        info.append(f"{tag} 200 ppm sample clock: evm {we:.3f}, errata {wr}")

    # a start 12 samples early / late (2-path, 25 dB): inside the window's 16-sample margin either way
    for off in (-12, 12):
        y = channel(0.6, 25)
        we, wr = frames(y, [(t + off, 0.6) for t in truth], f"start {off:+d}")
        info.append(f"{tag} start {off:+d} samples: evm {we:.3f}, errata {wr}")
        if we > 0.09:
            errs.append(f"start {off:+d}: EVM {we:.3f} (> 0.09), the window is not centred in the ISI-free stretch")

    # error codes
    y = channel(0.0, None, echo=False)
    t0 = truth[0]
    fs = lib.ofdm_cxdem_frame_samples(d)
    if c_dem(lib, d, y[:t0 + fs - 1], t0, 0.0)[0] != -2:
        errs.append("buffer one sample short not rejected with -2")
    if c_dem(lib, d, y[:t0 + fs], t0, 0.0)[0] != 0:
        errs.append("buffer exactly frame_samples long not decoded")
    if c_dem(lib, d, y, t0, 0.0, cap=255)[0] != -4:
        errs.append("out_cap 255 not rejected with -4")
    if c_dem(lib, d, y, -1, 0.0)[0] != -1:
        errs.append("negative start not rejected with -1")
    nz = (rng.standard_normal(30000) + 1j * rng.standard_normal(30000)) * 0.3
    if c_dem(lib, d, nz, 100, 0.0)[0] != -5:
        errs.append("noise not rejected with -5 (RS uncorrectable)")
    cache.setdefault("cxdem_info", []).extend(info)
    return errs


SEED_PROBE, SEED_MEAS = 0x7A3F, 0x51C7
COH_SCAN_SYMS, COH_CARRIERS, COH_GATE = 8, 40, 0.70
PROBE_SYNC_FAIL, PROBE_TOO_SHORT, CXPROBE_NO_SC = 2, 1, 4
NO_IMAGE_DB = -200.0


def ref_cxacq(rx, x, nsyms, step, top_k):
    """numpy reference of ofdm_cxacq_run: float32 grid rows, double sums, top-k by magnitude (ties lowest index)."""
    N, CP, SS = x.c.n_fft, x.c.cp_len, x.c.symbol_samples
    fidx = np.array(x._bins) % N
    n_pos = (rx.size - CP - N) // step + 1
    rows = np.array([(np.fft.fft(rx[p * step + CP: p * step + CP + N]) * (1.0 / np.sqrt(N)))[fidx]
                     for p in range(n_pos)]).astype(np.complex64).astype(complex)
    stride, span = SS // step, SS // step * (nsyms - 1)
    out = np.empty(n_pos - span)
    for c in range(n_pos - span):
        acc, den, mag = 0j, 0.0, 0.0
        for j in range(nsyms):
            r1 = rows[c + j * stride]
            p1 = r1.real ** 2 + r1.imag ** 2
            mag = mag + np.sqrt(p1)
            if j:
                r0 = rows[c + (j - 1) * stride]
                acc = acc + np.conj(r0) * r1
                den = den + (r0.real ** 2 + r0.imag ** 2) * p1
        ck = (acc.real ** 2 + acc.imag ** 2) / ((nsyms - 1) * den + 1e-300)
        thr = np.sort(mag)[::-1][top_k - 1]
        sel = list(np.nonzero(mag > thr)[0])
        sel += list(np.nonzero(mag == thr)[0][:top_k - len(sel)])
        out[c] = ck[sel].sum() / top_k
    return out, rows


def ref_track(H, Href, bins):
    """numpy reference of ofdm_cxprobe.c track(): theta = a + b*bin, three stages."""
    z = H * np.conj(Href)
    k1 = np.nonzero(np.diff(bins) == 1)[0]
    acc = (z[k1 + 1] * np.conj(z[k1])).sum()
    b = float(np.angle(acc)) if acc != 0 else 0.0
    i8 = np.array([i for i in range(8, bins.size) if bins[i] - bins[i - 8] == 8])
    acc = (z[i8] * np.conj(z[i8 - 8])).sum()
    if acc != 0:
        b += ((float(np.angle(acc)) - 8 * b + np.pi) % (2 * np.pi) - np.pi) / 8
    a = float(np.angle((z * np.exp(-1j * b * bins)).sum()))
    for _ in range(2):
        e = np.angle(z * np.exp(-1j * (a + b * bins)))
        w = np.abs(z)
        sw, sx, sxx, se, sxe = w.sum(), (w * bins).sum(), (w * bins * bins).sum(), (w * e).sum(), (w * bins * e).sum()
        den = sw * sxx - sx * sx
        if sw <= 0 or den <= 0:
            break
        db = (sw * sxe - sx * se) / den
        a += (se - db * sx) / sw
        b += db
    return a, b


def ref_cxprobe(rx, x, s0, cfo, n_p, n_m, pk, ce, mks, mir):
    """numpy reference of ofdm_cxprobe_estimate steps 4-6 at the C's frame start and CFO.
    Returns (H, snr_db, snr_db_periodic, image_db, isi_penalty_db, image median)."""
    N, CP, SS, T = x.c.n_fft, x.c.cp_len, x.c.symbol_samples, x.c.taper_samples
    w0 = T + CP // 2 if 2 * T <= CP else CP
    bins = np.array(x._bins, float)
    fidx = np.array(x._bins) % N

    def win(a):
        t = np.arange(a, a + N)
        ph = -2 * np.pi / N * np.fmod(cfo * (t - s0), N)
        return np.fft.fft(rx[t] * np.exp(1j * ph))[fidx] / np.sqrt(N)

    Href = win(s0 + SS + w0) / ce
    Hs = []
    for j in range(n_p):
        H = win(s0 + (2 + j) * SS + w0) / pk
        a, b = ref_track(H, Href, bins)
        Hs.append(H * np.exp(-1j * (a + b * bins)))
    Hs = np.array(Hs)
    m = Hs.mean(0)
    nv = np.maximum(Hs.real.var(0, ddof=1) + Hs.imag.var(0, ddof=1), 1e-12)
    pdb = 10 * np.log10(np.maximum(2 * np.abs(m) ** 2 / nv, 1e-10) + 1e-10)
    Ys, X1, X2 = [], [], []
    for j in range(n_m):
        y = win(s0 + (2 + n_p + j) * SS + w0)
        a, b = ref_track(y / mks[j], Href, bins)
        Ys.append(y * np.exp(-1j * (a + b * bins)))
        X1.append(mks[j])
        X2.append(np.where(mir >= 0, mks[j][np.maximum(mir, 0)], 0.0))
    Ys, X1, X2 = np.array(Ys), np.array(X1), np.array(X2)
    s11, s12, s22 = (X1 * X1).sum(0), (X1 * X2).sum(0), (X2 * X2).sum(0)
    b1, b2, yy = (X1 * Ys).sum(0), (X2 * Ys).sum(0), (np.abs(Ys) ** 2).sum(0)
    sdb, idb = pdb.copy(), np.full(x.c.n_occupied, NO_IMAGE_DB)
    for i in range(x.c.n_occupied):
        det = s11[i] * s22[i] - s12[i] ** 2
        if mir[i] >= 0 and det > 1e-3 * s11[i] * s22[i]:
            G = (s22[i] * b1[i] - s12[i] * b2[i]) / det
            I = (s11[i] * b2[i] - s12[i] * b1[i]) / det
            rss = yy[i] - (np.conj(G) * b1[i] + np.conj(I) * b2[i]).real
            nv_, g2, i2 = max(rss / (n_m - 2), 1e-12), abs(G) ** 2, abs(I) ** 2
            g2u, i2u = g2 - nv_ * s22[i] / det, i2 - nv_ * s11[i] / det
        else:
            G = b1[i] / s11[i]
            rss = yy[i] - (np.conj(G) * b1[i]).real
            nv_, g2, i2 = max(rss / (n_m - 1), 1e-12), abs(G) ** 2, 0.0
            g2u, i2u = g2 - nv_ / s11[i], 0.0
        s = max(max(g2u, 0.0) / max(i2u + nv_, 1e-12), 1e-10)
        sdb[i] = 10 * np.log10(s + 1e-10)
        if mir[i] >= 0 and i2 > 0:
            idb[i] = 10 * np.log10(i2 / (g2 + 1e-300) + 1e-30)
    have = idb > NO_IMAGE_DB
    return m, sdb, pdb, idb, float(np.median(pdb) - np.median(sdb)), float(np.median(idb[have]))


def make_cxacq(lib, x, nsyms, step, top_k, memlen=None):
    need = lib.ofdm_cxacq_workspace(ctypes.byref(x), nsyms, step, top_k)
    mem = ctypes.create_string_buffer(max(need, 1))
    a = ctypes.c_void_p()
    rc = lib.ofdm_cxacq_init(ctypes.byref(a), mem, need if memlen is None else memlen, ctypes.byref(x), None,
                             nsyms, step, top_k)
    return rc, a, mem


def c_acq(lib, a, rx):
    cap = max(lib.ofdm_cxacq_n_cand(a, rx.size), 1)
    out = (ctypes.c_double * cap)()
    info = AcqInfo()
    rc = lib.ofdm_cxacq_run(a, cx_in(rx), rx.size, out, cap, ctypes.byref(info))
    return rc, np.array(out[:info.n_cand]), info


def check_cxacq(lib, tol, tag, cache):
    """Step 7a: CFO-invariant coherence acquisition, C against numpy and against truth."""
    errs = []
    rc, x = cxcfg(lib, 256, -50e3, 50e3, 200000, sync=4, taper_ms=0.08)
    x._bins = bins_of(lib, x)
    N, SS, CP = x.c.n_fft, x.c.symbol_samples, x.c.cp_len
    step, K = 32, 40
    if make_cxacq(lib, x, 1, step, K)[0] != -1:
        errs.append("nsyms 1 not rejected with -1")
    if make_cxacq(lib, x, 8, 33, K)[0] != -4:
        errs.append("step 33 (does not divide 320) not rejected with -4")
    if make_cxacq(lib, x, 8, step, K, memlen=lib.ofdm_cxacq_workspace(ctypes.byref(x), 8, step, K) - 1)[0] != -2:
        errs.append("short workspace not rejected with -2")
    rc, a, _am = make_cxacq(lib, x, 8, step, K)
    if rc:
        return errs + [f"ofdm_cxacq_init rc={rc}"]
    if lib.ofdm_cxacq_ring_rows(a) != (SS // step) * 7 + 1:
        errs.append(f"ring rows {lib.ofdm_cxacq_ring_rows(a)}")
    # a periodic block: 14 copies of one random symbol, after silence
    rng = np.random.default_rng(71)
    occ = (rng.integers(0, 2, x.c.n_occupied) * 2 - 1) / np.sqrt(2)
    sym = ref_symbol(x, occ, 0)
    blk = np.concatenate([np.zeros(1500, complex), np.tile(sym, 14), np.zeros(1500, complex)])
    worst, info = 0.0, []
    for eps in (0.0, 0.4, -0.81, 3.3, -14.6):
        y = blk * np.exp(2j * np.pi * eps * np.arange(blk.size) / N)
        y = y + (rng.standard_normal(y.size) + 1j * rng.standard_normal(y.size)) * 0.05
        rc, m, ai = c_acq(lib, a, y)
        if rc or ai.n_cand != lib.ofdm_cxacq_n_cand(a, y.size) or ai.start_step != step:
            errs.append(f"eps {eps}: rc {rc} n_cand {ai.n_cand}"); continue
        r, rows = ref_cxacq(y, x, 8, step, K)
        if r.size != m.size:
            errs.append(f"eps {eps}: {m.size} candidates, numpy {r.size}"); continue
        worst = max(worst, float(np.max(np.abs(r - m))))
        if eps == 0.4:
            row = (ctypes.c_float * (2 * x.c.n_occupied))()
            if lib.ofdm_cxacq_row(a, cx_in(y), y.size, 17, row):
                errs.append("ofdm_cxacq_row failed")
            else:
                cr = np.array(row[0::2]) + 1j * np.array(row[1::2])
                d = float(np.max(np.abs(cr - rows[17]) / (np.abs(rows[17]) + 1e-3)))
                if d > (1e-6 if tag == "f64" else 1e-4):
                    errs.append(f"row 17 differs from numpy by {d:.2e}")
        # truth: inside the block (all 8 symbols periodic) the metric is ~1; the peak is there, well above noise
        inside = [c for c in range(m.size) if 1500 <= c * step and c * step + 7 * SS + CP + N <= 1500 + 14 * SS]
        lo = float(min(m[inside]))
        info.append(f"eps {eps:+.2f}: inside {lo:.3f} peak {m.max():.3f} median {np.median(m):.3f}")
        if lo < 0.95:
            errs.append(f"eps {eps}: metric inside the periodic block down to {lo:.3f} (< 0.95)")
    if worst > tol:
        errs.append(f"metric C vs numpy: max diff {worst:.2e} (> {tol:.0e})")
    nz = (rng.standard_normal(20000) + 1j * rng.standard_normal(20000))
    rc, m, ai = c_acq(lib, a, nz)
    if m.max() >= COH_GATE:
        errs.append(f"noise peak {m.max():.3f} >= gate {COH_GATE}")
    info.append(f"noise: peak {m.max():.3f}, median {np.median(m):.3f}")
    rc, m, ai = c_acq(lib, a, nz[:CP + N + 7 * SS - 1])
    if rc or ai.n_cand:
        errs.append(f"capture one sample short of a candidate: rc {rc} n_cand {ai.n_cand}")
    if lib.ofdm_cxacq_run(a, cx_in(nz), nz.size, (ctypes.c_double * 1)(), 1, ctypes.byref(AcqInfo())) != -2:
        errs.append("metric_cap too small not rejected with -2")
    cache.setdefault("cxacq_info", []).append(f"{tag} worst |C - numpy| {worst:.1e}; " + "; ".join(info))
    return errs


class CxProbe:
    def __init__(self, lib, x, n_p, n_m, cap, maxi):
        self.lib, self.x, self.O = lib, x, x.c.n_occupied
        need = lib.ofdm_cxprobe_workspace(ctypes.byref(x), n_p, n_m, cap, maxi)
        self.mem = ctypes.create_string_buffer(max(need, 1))
        self.p = ctypes.c_void_p()
        self.rc = lib.ofdm_cxprobe_init(ctypes.byref(self.p), self.mem, need, ctypes.byref(x), n_p, n_m, cap, maxi)

    def build(self):
        L = self.lib.ofdm_cxprobe_build_samples(self.p)
        buf = (ctypes.c_double * (2 * L))()
        n = self.lib.ofdm_cxprobe_build(self.p, buf, L)
        return cx_out(buf, n) if n == L else None

    def estimate(self, y, thr=0.0):
        arrs = [(ctypes.c_double * self.O)() for _ in range(6)]
        info = CxProbeInfo()
        rc = self.lib.ofdm_cxprobe_estimate(self.p, cx_in(y), y.size, thr, *arrs, ctypes.byref(info))
        H = np.array(arrs[0][:]) + 1j * np.array(arrs[1][:])
        return rc, info, H, np.array(arrs[3][:]), np.array(arrs[4][:]), np.array(arrs[5][:])


def check_cxprobe(lib, tol, tag, cache):
    """Step 7b: the complex channel probe, C against numpy and against truth."""
    errs = []
    rc, x = cxcfg(lib, 256, -50e3, 50e3, 200000, sync=4, taper_ms=0.08)
    x._bins = bins_of(lib, x)
    N, SS, CP, T, O = x.c.n_fft, x.c.symbol_samples, x.c.cp_len, x.c.taper_samples, x.c.n_occupied
    NP, NM, CAP, MAXI, D = 64, 32, 60000, 16, 1237
    bins = np.array(x._bins)
    mir = np.array([int(np.nonzero(bins == -b)[0][0]) if (bins == -b).any() else -1 for b in bins])
    pk = known_bpsk(lib, O, SEED_PROBE, 1 / np.sqrt(2))
    ce = known_bpsk(lib, O, SEED_CE, 1.0)
    mst = known_bpsk(lib, O * NM, SEED_MEAS, 1 / np.sqrt(2))      # one stream, O bits per symbol
    mks = [mst[j * O:(j + 1) * O] for j in range(NM)]

    for args, want in (((NP, NM, CAP, -1), -1), ((7, NM, CAP, MAXI), -1), ((NP, -1, CAP, MAXI), -1)):
        if CxProbe(lib, x, *args).rc != want:
            errs.append(f"init{args} not rejected with {want}")
    need = lib.ofdm_cxprobe_workspace(ctypes.byref(x), NP, NM, CAP, MAXI)
    tmp, pp = ctypes.create_string_buffer(need), ctypes.c_void_p()
    if lib.ofdm_cxprobe_init(ctypes.byref(pp), tmp, need - 1, ctypes.byref(x), NP, NM, CAP, MAXI) != -2:
        errs.append("short workspace not rejected with -2")
    P = CxProbe(lib, x, NP, NM, CAP, MAXI)
    if P.rc:
        return errs + [f"ofdm_cxprobe_init rc={P.rc}"]

    # build == numpy: cxmod's SC (TX gain 2) + CE, then the periodic and measurement blocks, overlap-added
    sig = P.build()
    rows = [ce.astype(complex)] + [pk.astype(complex)] * NP + [m.astype(complex) for m in mks]
    want, _sc = ref_cx_frame(lib, x, rows)
    if sig is None or sig.size != want.size or sig.size != SS * (2 + NP + NM) + 2 * T:
        return errs + [f"build length {None if sig is None else sig.size}, numpy {want.size}"]
    d = float(np.max(np.abs(sig - want)))
    if d > (1e-11 if tag == "f64" else 2e-4):
        errs.append(f"build differs from numpy by {d:.2e}")
    m, _mm = make_cxmod(lib, x, cache["bpb"])
    fl = lib.ofdm_cxmod_frame_samples(m)
    fb = (ctypes.c_double * (2 * fl))()
    lib.ofdm_cxmod_frame(m, 7, bytes(256), fb, fl)
    if float(np.max(np.abs(cx_out(fb, fl)[:SS] - sig[:SS]))) > 1e-12:
        errs.append("SC preamble differs from ofdm_cxmod's")

    rng = np.random.default_rng(77)
    h = np.array([1, 0, 0, 0.3j])
    Hc = np.fft.fft(h, N)[bins % N]

    def channel(eps, snr_db, z=None, iq=None, tail=3000):
        src = sig if z is None else z
        if iq and iq[0] == "tx":                    # imbalance at the transmitter: before the channel and the CFO
            src = iq_mix(src, *iq[1:])
        s = np.convolve(src, h)
        y = np.concatenate([np.zeros(D, complex), s, np.zeros(tail, complex)])
        y = y * np.exp(1j * (2 * np.pi * eps * np.arange(y.size) / N + rng.uniform(-np.pi, np.pi)))
        if iq and iq[0] == "rx":                    # at the receiver: after the CFO
            y = iq_mix(y, *iq[1:])
        var = 10 ** (-snr_db / 10)
        return y + (rng.standard_normal(y.size) + 1j * rng.standard_normal(y.size)) * np.sqrt(var / 2), var

    def iq_mix(s, g, ph):
        mu, nu = (1 + g * np.exp(-1j * ph)) / 2, (1 - g * np.exp(1j * ph)) / 2
        return mu * s + nu * np.conj(s)

    def est(y, label, want_valid=True):
        rc, info, H, sdb, pdb, idb = P.estimate(y)
        if want_valid and (rc or not info.valid or info.frame_start != D):
            errs.append(f"{label}: rc {rc} err {info.error} start {info.frame_start - D:+d} coh {info.coherence:.3f}")
            return None
        return info, H, sdb, pdb, idb

    # against numpy (steps 4-6 at the C's start and CFO) and truth, over a CFO sweep, 2-path, 25 dB
    worst = dict(ref=0.0, H=0.0, cfo=0.0, med=0.0, p90=0.0, pmed=0.0)
    for eps in (0.0, 0.37, -0.81, 0.999, 3.3, 7.25, -14.6):
        y, var = channel(eps, 25)
        r = est(y, f"25 dB eps {eps}")
        if r is None:
            continue
        info, H, sdb, pdb, idb = r
        if info.n_syms != NP or info.n_meas_syms != NM or not info.has_isi_penalty or not info.has_image:
            errs.append(f"eps {eps}: n {info.n_syms}/{info.n_meas_syms} isi {info.has_isi_penalty} image {info.has_image}")
        rH, rs, rp, ri, risi, rimg = ref_cxprobe(y, x, info.frame_start, info.cfo, NP, NM, pk, ce, mks, mir)
        worst["ref"] = max(worst["ref"], float(np.max(np.abs(sdb - rs))), float(np.max(np.abs(pdb - rp))),
                           float(np.max(np.abs(idb - ri))), abs(info.isi_penalty_db - risi), abs(info.image_db - rimg))
        worst["H"] = max(worst["H"], float(np.max(np.abs(H - rH))))
        worst["cfo"] = max(worst["cfo"], abs(info.cfo - eps))
        true = 10 * np.log10(np.abs(Hc) ** 2 / var)
        e = sdb - true
        worst["med"] = max(worst["med"], abs(float(np.median(e))))
        worst["p90"] = max(worst["p90"], float(np.percentile(np.abs(e), 90)))
        worst["pmed"] = max(worst["pmed"], abs(float(np.median(pdb - true))))
        # H is the channel up to a phase a + b*bin (the CE's random phase, and the window offset w0 != cp_len)
        a_, b_ = ref_track(H, Hc, bins.astype(float))
        dh = float(np.max(np.abs(H * np.exp(-1j * (a_ + b_ * bins)) - Hc)))
        worst["Hc"] = max(worst.get("Hc", 0.0), dh)
        if dh > 0.15:
            errs.append(f"eps {eps}: H differs from the channel by {dh:.3f} (after a phase ramp)")
    if worst["ref"] > (1e-8 if tag == "f64" else 2e-3) or worst["H"] > tol:
        errs.append(f"C vs numpy: dB {worst['ref']:.2e}, H {worst['H']:.2e}")
    # 0.005 ~ 3 sigma here: per sample this is ~6 dB below the cxsync check's 25 dB (the probe and measurement
    # symbols run at 1/sqrt(2) on half the band); cxsync's own check grades the estimator
    if worst["cfo"] > 0.005:
        errs.append(f"CFO error up to {worst['cfo']:.4f}")
    if worst["med"] > 0.5 or worst["pmed"] > 0.5 or worst["p90"] > 2.0:
        errs.append(f"25 dB SNR vs truth: median off by {worst['med']:.2f} dB (periodic {worst['pmed']:.2f}), "
                    f"90th pct |err| {worst['p90']:.2f} dB")
    info_l = [f"{tag} 25 dB 2-path CFO sweep: start exact, |cfo err| <= {worst['cfo']:.4f}, snr median err <= "
              f"{worst['med']:.2f} dB (periodic {worst['pmed']:.2f}), 90th pct {worst['p90']:.2f} dB, "
              f"|H - channel| <= {worst.get('Hc', 0.0):.3f}; C vs numpy {worst['ref']:.1e} dB"]

    # 10 dB, and the same object twice (no state left from the last call)
    y, var = channel(0.37, 10)
    r1, r2 = est(y, "10 dB"), est(y, "10 dB again")
    if r1 and r2:
        e = float(np.median(r1[2] - 10 * np.log10(np.abs(Hc) ** 2 / var)))
        info_l.append(f"{tag} 10 dB: snr median err {e:+.2f} dB, coherence {r1[0].coherence:.3f}")
        if abs(e) > 0.75:
            errs.append(f"10 dB: SNR median off by {e:+.2f} dB")
        if not (np.array_equal(r1[2], r2[2]) and r1[0].frame_start == r2[0].frame_start):
            errs.append("a second estimate on the same capture differs")

    # 6 dB, four captures: the slope's lag stage keeps the residual phase unwrapped (without it: -5 dB)
    e6 = []
    for _ in range(4):
        y, var = channel(0.37, 6)
        r = est(y, "6 dB")
        if r:
            e6.append(float(np.median(r[2] - 10 * np.log10(np.abs(Hc) ** 2 / var))))
    if e6:
        info_l.append(f"{tag} 6 dB x4: snr median err {min(e6):+.2f} .. {max(e6):+.2f} dB")
        if max(abs(v) for v in e6) > 0.6:
            errs.append(f"6 dB: SNR median off by up to {max(e6, key=abs):+.2f} dB")

    # a short probe (16 + 8 symbols), eight captures at 10 dB: the SINR's noise correction matters here
    # (uncorrected, the fit's noise in |G|^2 and |I|^2 reads ~0.7 dB low)
    Ps = CxProbe(lib, x, 16, 8, CAP, MAXI)
    sig_s = Ps.build()
    es = []
    for _ in range(8):
        y, var = channel(0.37, 10, z=sig_s)
        rc, inf, H, sdb, pdb, idb = Ps.estimate(y)
        if rc or inf.frame_start != D or inf.n_meas_syms != 8:
            errs.append(f"short probe: rc {rc} start {inf.frame_start - D:+d} n {inf.n_syms}/{inf.n_meas_syms}"); break
        es.append(float(np.median(sdb - 10 * np.log10(np.abs(Hc) ** 2 / var))))
    if len(es) == 8:
        info_l.append(f"{tag} short probe 16+8, 10 dB x8: snr median err mean {np.mean(es):+.2f} dB "
                      f"({min(es):+.2f} .. {max(es):+.2f})")
        if abs(np.mean(es)) > 0.45:
            errs.append(f"short probe: SNR median err averages {np.mean(es):+.2f} dB over 8 captures")

    # IQ image.  Theory: |nu|^2/|mu|^2 at the TX (any CFO, any channel); |nu H(-k)|^2/|mu H(k)|^2 at the RX, CFO 0
    g, phi = 1.05, np.deg2rad(3.0)
    mu, nu = (1 + g * np.exp(-1j * phi)) / 2, (1 - g * np.exp(1j * phi)) / 2
    irr = 10 * np.log10(abs(nu) ** 2 / abs(mu) ** 2)
    for side, eps in (("tx", 0.37), ("tx", -3.3), ("rx", 0.0)):
        y, var = channel(eps, 40, iq=(side, g, phi))
        r = est(y, f"IQ {side} eps {eps}")
        if r is None:
            continue
        info, H, sdb, pdb, idb = r
        if side == "tx":
            th_img = np.full(O, irr)
            Gp, Ip = np.abs(mu * Hc) ** 2, np.abs(nu * Hc) ** 2       # data power 1 at both k and -k
        else:
            Hm = Hc[mir]
            th_img = 10 * np.log10(np.abs(nu * Hm) ** 2 / np.abs(mu * Hc) ** 2)
            Gp, Ip = np.abs(mu * Hc) ** 2, np.abs(nu * Hm) ** 2
        th_sinr = 10 * np.log10(Gp / (Ip + var))
        ie, se = float(np.median(np.abs(idb - th_img))), float(np.median(sdb - th_sinr))
        info_l.append(f"{tag} IQ at {side}, eps {eps}, 40 dB: image median {info.image_db:.2f} dB (theory "
                      f"{float(np.median(th_img)):.2f}), median |image err| {ie:.2f} dB, SINR median err {se:+.2f} dB, "
                      f"isi penalty {info.isi_penalty_db:.1f} dB")
        if ie > 1.0 or abs(se) > 0.75:
            errs.append(f"IQ {side} eps {eps}: image err {ie:.2f} dB, SINR median err {se:+.2f} dB")
        if info.isi_penalty_db < 8.0:
            errs.append(f"IQ {side} eps {eps}: periodic-vs-measured gap {info.isi_penalty_db:.1f} dB, the image was absorbed")

    # 200 ppm slow sample clock over the ~100-symbol probe: ~6 samples of drift, a growing phase slope
    from scipy.signal import resample_poly
    z = resample_poly(sig, 5000, 5001)
    y, var = channel(1.4, 25, z=z)
    r = est(y, "200 ppm")
    if r:
        true = 10 * np.log10(np.abs(Hc) ** 2 / var)
        e, ep = float(np.median(r[2] - true)), float(np.median(r[3] - true))
        info_l.append(f"{tag} 200 ppm sample clock: snr median err {e:+.2f} dB (periodic {ep:+.2f})")
        if abs(e) > 0.5 or abs(ep) > 0.5:
            errs.append(f"200 ppm: SNR median off by {e:+.2f} dB (periodic {ep:+.2f})")

    # probe -> ofdm_allocate -> BAT -> a frame through the same channel decodes
    y, var = channel(-2.6, 25)
    r = est(y, "allocate")
    if r:
        nd = x.c.n_data_carriers
        bat = (ctypes.c_int * nd)()
        sd = (ctypes.c_double * O)(*r[2])
        k = lib.ofdm_allocate(sd, O, x.c.pilot_spacing, 3.0, 6, 0.0, bat, nd)
        bpb = list(bat[:nd])
        if k != nd or sum(bpb) < 1:
            errs.append(f"ofdm_allocate returned {k}, {sum(bpb)} bits")
        else:
            mm, _m2 = make_cxmod(lib, x, bpb)
            rcd, dd, _dm = make_cxdem(lib, x, bpb)
            chunk = bytes(rng.integers(0, 256, 256, dtype=np.uint8))
            fl = lib.ofdm_cxmod_frame_samples(mm)
            fb = (ctypes.c_double * (2 * fl))()
            lib.ofdm_cxmod_frame(mm, 0x1234, chunk, fb, fl)
            yf, _v = channel(-2.6, 25, z=cx_out(fb, fl))
            rcd, ch, dinfo = c_dem(lib, dd, yf, D, r[0].cfo)
            hist = {b: bpb.count(b) for b in sorted(set(bpb))}
            info_l.append(f"{tag} BAT from the 25 dB probe (3 dB margin): {hist}, frame rc {rcd} evm {dinfo.evm:.3f} "
                          f"errata {dinfo.errata}")
            if rcd or ch != chunk or dinfo.seq != 0x1234:
                errs.append(f"frame on the probe's BAT: rc {rcd}, chunk {'ok' if ch == chunk else 'WRONG'}")

    # what must NOT be a probe
    var = 10 ** (-25 / 10)
    nz = (rng.standard_normal(30000) + 1j * rng.standard_normal(30000)) * np.sqrt(var / 2)
    t = np.arange(nz.size)
    for label, y, want in (("noise", nz, PROBE_SYNC_FAIL),
                           ("CW tone", nz + 0.5 * np.exp(2j * np.pi * 0.1234 * t), CXPROBE_NO_SC),
                           # the DC bin is the dc_null hole, not an occupied carrier: acquisition never sees it
                           ("DC offset", nz + 0.3, PROBE_SYNC_FAIL),
                           ("probe without its SC preamble", channel(0.37, 25)[0][D + SS - 40:], CXPROBE_NO_SC)):
        rc, info, *_ = P.estimate(y)
        info_l.append(f"{tag} {label}: rc {rc} error {info.error} coherence {info.coherence:.3f}")
        if rc != 1 or info.valid or info.error != want:
            errs.append(f"{label}: rc {rc} valid {info.valid} error {info.error} (want {want}), "
                        f"coherence {info.coherence:.3f}")
    ns = lib.ofdm_cxmod_signal_samples_ex(m, 3 * 256, 1)
    fb = (ctypes.c_double * (2 * ns))()
    lib.ofdm_cxmod_encode_ex(m, bytes(range(256)) * 3, 768, 0, 1, fb, ns)
    yd = cx_out(fb, ns) + nz[:ns] if ns <= nz.size else None
    if yd is not None:
        rc, info, *_ = P.estimate(yd)
        info_l.append(f"{tag} three data frames: rc {rc} error {info.error} coherence {info.coherence:.3f}")
        if rc != 1 or info.valid:
            errs.append(f"data frames taken for a probe (coherence {info.coherence:.3f})")

    # a capture cut in the measurement block: valid, fewer measurement symbols; too short / too long refused
    y, var = channel(0.2, 25, tail=0)
    cut = D + SS * (2 + NP + NM // 2) + CP + N
    r = est(y[:cut], "cut capture")
    if r and not (4 <= r[0].n_meas_syms < NM and r[0].n_syms == NP):
        errs.append(f"cut capture: n {r[0].n_syms}/{r[0].n_meas_syms}")
    for label, yy_ in (("too short", y[:7 * SS + CP + N + SS - 1]), ("over max_capture", np.zeros(CAP + 1, complex))):
        rc, info, *_ = P.estimate(yy_)
        if rc != 1 or info.error != PROBE_TOO_SHORT:
            errs.append(f"{label}: rc {rc} error {info.error}")
    cache.setdefault("cxprobe_info", []).extend(info_l)
    return errs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keep", help="build into this dir and keep it")
    a = p.parse_args()
    out = a.keep or tempfile.mkdtemp(prefix="ofdmcx_")
    os.makedirs(out, exist_ok=True)
    try:
        return run(out)
    finally:
        if not a.keep:
            shutil.rmtree(out, ignore_errors=True)


def run(out):
    allbad = []

    def report(name, errs):
        print(f"[{name}]", "ok" if not errs else "\n    ".join(["FAIL"] + errs), flush=True)
        allbad.extend(errs)

    report("c89 " + " ".join(CX_NEW), c89_check())
    cache = {}
    for f32, tol in ((False, 1e-12), (True, 2e-5)):
        lib, real, Cpx, Cfft = build(out, f32)
        tag = "f32" if f32 else "f64"
        report(f"{tag} cfft vs numpy", check_fft(lib, real, Cpx, Cfft, tol))
        report(f"{tag} cxcfg geometry", check_geometry(lib))
        report(f"{tag} cxsym make/demod", check_symbol(lib, tol))
        report(f"{tag} frame arithmetic reuse", check_reuse(lib))
        report(f"{tag} cxmod frame vs real-modulator carriers", check_cxmod(lib, tol * 10, cache))
        report(f"{tag} cxsync acquisition + CFO", check_cxsync(lib, 1e-9, tol * 10, cache))
        report(f"{tag} cxdem demodulate + RS", check_cxdem(lib, 1e-9 if not f32 else 1e-4, tag, cache))
        report(f"{tag} cxacq coherence acquisition", check_cxacq(lib, 1e-6 if not f32 else 1e-4, tag, cache))
        report(f"{tag} cxprobe channel probe", check_cxprobe(lib, 1e-9 if not f32 else 1e-4, tag, cache))
    for line in cache.get("cxsync_info", []):
        print("    cxsync:", line)
    for line in cache.get("cxdem_info", []):
        print("    cxdem:", line)
    for line in cache.get("cxacq_info", []):
        print("    cxacq:", line)
    for line in cache.get("cxprobe_info", []):
        print("    cxprobe:", line)
    print("CX FAIL" if allbad else "CX PASS")
    return 1 if allbad else 0


if __name__ == "__main__":
    sys.exit(main())
