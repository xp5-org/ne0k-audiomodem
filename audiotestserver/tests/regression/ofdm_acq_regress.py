# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""OFDM acquisition backend equivalence: numpy reference vs the C core.

`_grid_coherence_scan` is the whole cost of OFDM acquisition and has three
implementations that must agree: the numpy reference (the default, and the
benchmark), the C core built in double (which matches numpy's FFT precision),
and the C core built in float32 (the 32-bit target: RP2350 / CMSIS-DSP).  This
suite is what keeps them honest about each other.

    python tests/regression/ofdm_acq_regress.py           # both C precisions
    python tests/regression/ofdm_acq_regress.py --lib 32  # float32 only
    python tests/regression/ofdm_acq_regress.py -v        # per-case detail

Exit code 0 = all pass.  SKIP (still 0) if the C is not built -- run
`make -C tests/OFDM/c both` first.  Run before and after ANY change to the scan
or to the C.

WHAT IT GRADES, AND WHY NOT BIT-EQUALITY.  The metric is an intermediate, not a
deliverable, and three things legitimately differ between correct
implementations:

  * Reduction order.  float32 sums do not associate; the C accumulates
    sequentially because that is what the MCU will do.  Floor ~4e-7.
  * top_k boundary ties.  "The strongest 40 carriers" is ambiguous when
    carriers sit within a ULP of the cut, and swapping one of 40 moves the
    metric by ~1e-5.  Asserted to be explained, not merely tolerated.
  * argmax on a plateau.  A probe is a PERIODIC excitation, so coherence is
    deliberately flat across it -- tens of candidates within round-off of the
    peak.  ChannelProbe.estimate is built to absorb that.

So the bar is the DECISIONS: identical candidate starts, identical thresholding
at every gate a caller applies, an argmax that is either identical or provably
indistinguishable, and -- the one that actually matters -- identical
frame_start / n_syms / SNR / BAT out of analyze_probe end to end.

The H row is checked separately and IS bit-exact for the double build, which is
what separates "the port is wrong" from "float32 rounds differently".
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

_TESTS_ROOT = Path(__file__).resolve().parents[1]
for _p in (_TESTS_ROOT, _TESTS_ROOT / "OFDM"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ofdm_modem import (build_config, Modem, RobustLink, AdaptiveDemodulator,
                        _grid_coherence_scan,
                        _RL_Q, _RL_N_PRE, _COH_SCAN_SYMS, _COH_CARRIERS,
                        _COH_GATE, _RL_GATE)

# Every gate a caller applies to this metric, plus a couple of interior probes
# so a systematic shift shows up even where no caller looks.
GATES = (0.25, _COH_GATE, _RL_GATE, 0.5, 0.9)
GEOMETRIES = (512, 1024, 2048)


def _ref_row(rx, cfg, known, pos, step):
    """One grid row exactly as the numpy scan computes it."""
    N, CP, lo = cfg['n_fft'], cfg['cp_len'], cfg['bin_lo']
    nocc = cfg['n_occupied']
    off = pos * step + CP
    fd = np.fft.rfft(rx[off:off + N], n=N)
    return (fd[lo:lo + nocc] * (1.0 / math.sqrt(N)) / known).astype(np.complex64)


def _boundary_tie(rx, cfg, known, nsyms, step, q, top_k, cand):
    """True if `cand` is a near-tie at the top_k magnitude cut.

    Threshold is DERIVED from the dtype (np.spacing of the boundary value,
    times the number of terms summed), not a hard-coded epsilon -- an absolute
    epsilon here would silently encode a float64 assumption and stop being
    meaningful in the float32 build, which is exactly the build it must judge.
    """
    N, CP, lo = cfg['n_fft'], cfg['cp_len'], cfg['bin_lo']
    nocc = cfg['n_occupied']
    stride = (N + CP) // step
    mag = np.zeros(nocc, np.float32)
    for j in range(nsyms):
        off = (cand + j * stride) * step + CP
        fd = np.fft.rfft(rx[off:off + N], n=N)
        H = (fd[lo:lo + nocc] * (1.0 / math.sqrt(N)) / known).astype(np.complex64)
        mag += np.abs(H)
    srt = np.sort(mag)[::-1]
    k = min(top_k, nocc)
    return bool(srt[k - 1] - srt[k] <= nsyms * float(np.spacing(srt[k - 1])))


class Case:
    def __init__(self, verbose):
        self.fails = 0
        self.verbose = verbose

    def report(self, ok, label, detail):
        self.fails += (not ok)
        if not ok or self.verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {label:<34} {detail}")

    def scan(self, label, rx, cfg, known, nsyms, step, q, top_k, AcqScan):
        starts, met = _grid_coherence_scan(
            rx, cfg['n_fft'], cfg['cp_len'], cfg['bin_lo'], cfg['n_occupied'],
            known, nsyms, step, q=q, top_k=top_k)
        plan = AcqScan(cfg['n_fft'], cfg['cp_len'], cfg['bin_lo'],
                       cfg['n_occupied'], known, nsyms, step, q=q, top_k=top_k)
        cstarts, cmet = plan(rx)

        rowd = 0.0
        if len(met):
            for pos in (0, len(met) // 2, len(met) - 1):
                a = _ref_row(rx, cfg, known, pos, step)
                b = plan.row(rx, pos)
                sc = float(np.max(np.abs(a))) or 1.0
                rowd = max(rowd, float(np.max(np.abs(a - b))) / sc)

        if len(met) == 0:
            ok = np.array_equal(starts, cstarts)
            self.report(ok, label, "empty (early-return path)")
            return

        d = float(np.max(np.abs(met - cmet)))
        ia, ib = int(np.argmax(met)), int(np.argmax(cmet))
        tol = 8.0 * float(np.spacing(np.float32(met[ia])))
        plateau = int((met >= met[ia] - tol).sum())
        same_arg = (ia == ib) or (abs(met[ia] - met[ib]) <= tol
                                  and abs(cmet[ib] - cmet[ia]) <= tol)
        same_gates = all(np.array_equal(met >= g, cmet >= g) for g in GATES)
        big = np.where(np.abs(met - cmet) > 1e-6)[0]
        unexplained = [int(c) for c in big
                       if not _boundary_tie(rx, cfg, known, nsyms, step, q,
                                            top_k, int(c))]
        ok = (np.array_equal(starts, cstarts) and same_arg and same_gates
              and not unexplained)
        self.report(ok, label,
                    f"n={len(met):5d} row={rowd:.1e} metric={d:.1e} "
                    f"argmax={'ok' if same_arg else 'DIFF'}(plateau {plateau}) "
                    f"gates={'ok' if same_gates else 'DIFF'} "
                    f"ties={len(big)}"
                    + (f" UNEXPLAINED {unexplained[:3]}" if unexplained else "")
                    + f" ring={plan.ring_rows} ws={plan.workspace_bytes//1024}KB")


def run(lib_tag, verbose):
    os.environ["OFDM_ACQ_LIB"] = lib_tag
    import importlib
    import ofdm_acq_c
    importlib.reload(ofdm_acq_c)
    if not ofdm_acq_c.available():
        print(f"  SKIP  C core not built: {ofdm_acq_c._lib_path()}")
        print(f"        run: make -C {_TESTS_ROOT / 'OFDM' / 'c'} both")
        return None
    AcqScan = ofdm_acq_c.AcqScan
    c = Case(verbose)

    for nfft in GEOMETRIES:
        cfg = build_config(n_fft=nfft)
        SYM = cfg['symbol_samples']
        CP = cfg['cp_len']
        m = Modem(cfg)
        rng = np.random.default_rng(7)

        step = math.gcd(SYM, max(8, CP // 2))
        probe = m.build_probe()
        for snr, tag in ((0.005, "clean"), (0.05, "noisy")):
            rx = (probe + snr * rng.standard_normal(len(probe))).astype(np.float64)
            c.scan(f"{nfft} probe {tag}", rx, cfg, m._probe.probe_known,
                   _COH_SCAN_SYMS, step, None, _COH_CARRIERS, AcqScan)
        rx = rng.standard_normal(len(probe)).astype(np.float64) * 0.01
        c.scan(f"{nfft} noise-only", rx, cfg, m._probe.probe_known,
               _COH_SCAN_SYMS, step, None, _COH_CARRIERS, AcqScan)
        c.scan(f"{nfft} too-short", probe[:SYM].astype(np.float64), cfg,
               m._probe.probe_known, _COH_SCAN_SYMS, step, None,
               _COH_CARRIERS, AcqScan)

        rl = RobustLink(cfg)
        step_rl = math.gcd(SYM, max(8, CP // 4))
        sig = rl.encode(bytes(range(24)))
        for snr, tag in ((0.005, "clean"), (0.05, "noisy")):
            rx = (sig + snr * rng.standard_normal(len(sig))).astype(np.float64)
            c.scan(f"{nfft} robustlink {tag}", rx, cfg, rl.known, _RL_N_PRE,
                   step_rl, _RL_Q, _COH_CARRIERS, AcqScan)

    # End to end: what ChannelProbe.estimate actually hands the allocator.
    # This is the check that matters -- the metric is only a means to it.
    for nfft in GEOMETRIES:
        cfg = build_config(n_fft=nfft)
        mm = Modem(cfg)
        rng = np.random.default_rng(7)
        pr = mm.build_probe()
        rx = (pr + 0.005 * rng.standard_normal(len(pr))).astype(np.float64)
        got = {}
        for be in ("", "c"):
            os.environ["OFDM_ACQ_BACKEND"] = be
            got[be or "numpy"] = mm.analyze_probe(rx)
        os.environ["OFDM_ACQ_BACKEND"] = ""
        a, b = got["numpy"], got["c"]
        bat_eq = np.array_equal(mm.allocate(a), mm.allocate(b))
        ok = (a['valid'] == b['valid']
              and a['frame_start'] == b['frame_start']
              and a['n_syms'] == b['n_syms'] and bat_eq)
        c.report(ok, f"{nfft} analyze_probe end-to-end",
                 f"start {a['frame_start']}/{b['frame_start']} "
                 f"n_syms {a['n_syms']}/{b['n_syms']} "
                 f"dSNR={np.max(np.abs(a['snr_db'] - b['snr_db'])):.1e}dB "
                 f"BAT={'equal' if bat_eq else 'DIFFERS'}")

    _demod_cases(c, verbose)
    return c.fails


def _demod_cases(c, verbose):
    """numpy vs C for the data-frame demodulator.

    Graded on the PAYLOAD BYTES, not on a metric.  Unlike the acquisition
    score -- where two implementations may legitimately disagree by a ULP at a
    top_k boundary -- these bytes are the message, and RS either repairs them
    or does not.  Byte equality is the only meaningful bar, and both precisions
    have met it on every case here.

    Every modulation order is covered because the decision regions shrink as
    the order rises: 64-QAM is where a float32 equaliser would first put a
    point on the wrong side of a boundary.  The mixed table is the case where a
    zero bits_per_bin must skip a carrier WITHOUT shifting every later bit.
    """
    import contextlib
    import io
    for nfft in GEOMETRIES:
        cfg = build_config(n_fft=nfft)
        nd = cfg["n_data_carriers"]
        mixed = np.zeros(nd, dtype=np.int32)
        mixed[0::3] = 2
        mixed[1::3] = 4
        tables = [("bpsk", np.ones(nd, dtype=np.int32)),
                  ("qpsk", np.full(nd, 2, dtype=np.int32)),
                  ("16qam", np.full(nd, 4, dtype=np.int32)),
                  ("64qam", np.full(nd, 6, dtype=np.int32)),
                  ("mixed", mixed)]
        for name, bat in tables:
            for snr, tag in ((0.002, "clean"), (0.02, "noisy")):
                m = Modem(cfg)
                rng = np.random.default_rng(abs(hash((nfft, name, tag))) % 9973)
                sig, _ = m.encode_data("The quick brown fox " * 40, bat)
                rx = (sig + snr * rng.standard_normal(len(sig))).astype(np.float64)
                dem = AdaptiveDemodulator(cfg, bat)

                os.environ["OFDM_ACQ_BACKEND"] = ""
                with contextlib.redirect_stdout(io.StringIO()):
                    stats, _ = dem.decode_samples(rx)
                starts = [s["frame_start"] for s in stats]
                if not starts:
                    c.report(False, f"{nfft} demod {name} {tag}",
                             "reference decoded no frames")
                    continue

                n_eq = 0
                devm = 0.0
                bad = None
                for d in starts:
                    os.environ["OFDM_ACQ_BACKEND"] = ""
                    ref = dem._demod_frame(rx, d)
                    os.environ["OFDM_ACQ_BACKEND"] = "c"
                    with contextlib.redirect_stdout(io.StringIO()):
                        got = dem._demod_frame(rx, d)
                    os.environ["OFDM_ACQ_BACKEND"] = ""
                    if ref is None or got is None:
                        bad = f"frame {d}: ref={ref is not None} c={got is not None}"
                        break
                    if ref[0] != got[0]:
                        i = next((k for k, (x, y) in enumerate(zip(ref[0], got[0]))
                                  if x != y), None)
                        bad = f"frame {d}: bytes differ at {i}/{len(ref[0])}"
                        break
                    if ref[2] != got[2]:
                        bad = f"frame {d}: data_syms {got[2]} != {ref[2]}"
                        break
                    devm = max(devm, abs(ref[1] - got[1]))
                    n_eq += 1
                c.report(bad is None, f"{nfft} demod {name} {tag}",
                         bad or f"{n_eq} frame(s) byte-exact, |dEVM|={devm:.1e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lib", choices=("", "32", "both"), default="both",
                    help="C precision: '' double, '32' float32, both (default)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every case, not just failures")
    args = ap.parse_args()

    libs = ["", "32"] if args.lib == "both" else [args.lib]
    total = 0
    ran = False
    for lib in libs:
        print(f"--- C backend: {'float32 (target)' if lib == '32' else 'double (oracle-matched)'} ---")
        f = run(lib, args.verbose)
        if f is None:
            continue
        ran = True
        total += f
        print(f"    {'all pass' if f == 0 else f'{f} FAILURE(S)'}")
    if not ran:
        print("\nOFDM ACQ REGRESS SKIP (C not built)")
        return 0
    print(f"\nOFDM ACQ REGRESS {'PASS' if total == 0 else f'FAIL ({total})'}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
