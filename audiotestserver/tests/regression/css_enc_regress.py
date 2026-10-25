# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""CSS transmit-chain C-port equivalence suite.

Grades tests/CSS/c/css_enc.c against the Python it was ported from
(tests/CSS/css_encoder.py), stage by stage, over the geometries this repo
actually transmits at.

    python tests/regression/css_enc_regress.py
    python tests/regression/css_enc_regress.py -v

WHY STAGE BY STAGE.  A whole-pipeline comparison can only say "the bits
differ".  The transmit chain is five transforms deep (bytes -> diff -> conv ->
interleave, plus CRC and the chirp), several of which have a plausible-looking
wrong convention -- a shift register clocked from the LSB instead of the MSB,
a differential encoder fed its own input instead of its own output, a block
interleaver transposed the other way.  Each of those produces a self-
consistent stream that only fails at the far end of a decode, so each stage is
compared against its Python original on its own.

WHY TWO PRECISIONS.  The chirp is graded in two stages:

  stage 1  libcssenc64 (float64 phase) vs numpy -- the phase arithmetic is the
           same, so ANY difference here is a port bug.
  stage 2  libcssenc (float32 phase, the shipping build) vs stage 1 -- graded
           on CORRELATION, not equality, because a difference here is the
           precision choice doing exactly what it was chosen to do.

The float32 threshold is the number that justifies the default in css_enc.h:
if the shipping build ever correlates below 0.99999 with the float64 chirp,
that comment is no longer true and the default should change.
"""
import argparse
import os
import sys

import numpy as np

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(TESTS, "CSS"))

import css_enc_c                      # noqa: E402
import css_encoder                    # noqa: E402


# Correlation floor for the float32-phase chirp against the float64 one.
# Measured across every geometry below: the worst is ~0.999993 at 3.0 s, which
# is 0.00003 dB of matched-filter loss.  Set at 0.99999 -- tight enough that a
# real phase bug (a wrong k, an off-by-one time base) blows straight through
# it, loose enough that round-off never will.
CHIRP_CORR_MIN = 0.99999

# (fs, f_min, f_max, symbol_dur) -- the geometries in the regression fixtures
# and the promoted runs, plus the 3.0 s worst case for phase round-off.
CHIRP_GEOMS = [
    (48000, 800, 4000, 0.25),
    (48000, 800, 4000, 0.5),
    (48000, 400, 6000, 1.0),
    (48000, 400, 6000, 3.0),
    (44100, 400, 6000, 0.3),      # fs*dur = 13230, integral
    (44100, 400, 6000, 0.35),     # fs*dur = 15435, integral
    (44100, 500, 5000, 0.27),     # fs*dur = 11907.0 -> non-integral duration/n
]

# (chunk_size, rs_t, conv_code, diff_enc, interleave_rows)
FRAME_GEOMS = [
    (1,  0, None,    False, 0),
    (1,  0, None,    True,  0),
    (2,  2, None,    False, 0),
    (12, 0, "k7r12", False, 0),
    (12, 0, "k7r12", True,  0),
    (8,  2, "k7r12", True,  8),
    (16, 4, None,    False, 5),
    (48, 0, "k7r12", False, 16),
]


class Fail(Exception):
    pass


def check(name, ok, detail=""):
    if not ok:
        raise Fail(f"{name}: {detail}")


# ── stage checks ─────────────────────────────────────────────────────────────

def t_crc32(rng, v):
    """css_crc32 vs zlib.crc32, which is what frame_crc uses."""
    import zlib
    for n in (0, 1, 4, 5, 17, 64, 255, 1000):
        data = bytes(rng.integers(0, 256, n, dtype=np.uint8))
        want = zlib.crc32(data) & 0xFFFFFFFF
        got = css_enc_c.crc32(data)
        check("crc32", got == want, f"n={n} got={got:08x} want={want:08x}")
    # All-ones and all-zeros: the two inputs where a missing init or final
    # xor still produces a plausible-looking value.
    for data in (b"\x00" * 32, b"\xff" * 32):
        want = zlib.crc32(data) & 0xFFFFFFFF
        check("crc32", css_enc_c.crc32(data) == want, f"data={data[:4]!r}...")
    return "8 lengths + edge patterns"


def t_bytes_to_bits(rng, v):
    for n in (1, 3, 16, 64):
        data = bytes(rng.integers(0, 256, n, dtype=np.uint8))
        want = []
        for b in data:
            want.extend(css_encoder.get_bits(b))
        check("bytes_to_bits", css_enc_c.bytes_to_bits(data) == want, f"n={n}")
    return "MSB-first, 4 lengths"


def t_conv(rng, v):
    for n in (1, 8, 40, 137, 400):
        bits = list(rng.integers(0, 2, n).tolist())
        want = css_encoder.conv_encode_k7(bits)
        got = css_enc_c.conv_encode_k7(bits)
        bad = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                   None)
        check("conv_k7", got == want, f"n={n} first_diff={bad}")
    # All-zero input must give all-zero output: catches a register that is not
    # actually being cleared, which random data hides.
    check("conv_k7", css_enc_c.conv_encode_k7([0] * 32) == [0] * 76,
          "all-zero input did not give all-zero output")
    return "5 lengths + all-zero"


def t_diff(rng, v):
    for n in (1, 8, 137):
        for ref in (0, 1):
            bits = list(rng.integers(0, 2, n).tolist())
            want = css_encoder.diff_encode_bits(bits, ref_bit=ref)
            got = css_enc_c.diff_encode(bits, ref)
            check("diff", got == want, f"n={n} ref={ref}")
    # Round-trip through the decoder's inverse: the encoder XORs its own
    # OUTPUT, the decoder its own INPUT, and getting that backwards is
    # self-consistent for one step and wrong from the second on.
    import css_fec
    bits = list(rng.integers(0, 2, 64).tolist())
    enc = css_enc_c.diff_encode(bits, 1)
    check("diff", css_fec.diff_decode_bits(enc, ref_bit=1) == bits,
          "diff_decode(diff_encode(x)) != x")
    return "3 lengths x 2 refs + decoder round-trip"


def t_interleave(rng, v):
    import css_fec
    for n, rows in ((40, 4), (40, 7), (137, 8), (400, 16), (10, 1), (10, 0)):
        bits = list(rng.integers(0, 2, n).tolist())
        want = css_encoder.interleave_bits(bits, rows)
        got = css_enc_c.interleave(bits, rows)
        check("interleave", got == want, f"n={n} rows={rows}")
        if rows > 1:
            back = css_fec.deinterleave_bits(got, rows, n)
            check("interleave", back == bits,
                  f"deinterleave(interleave(x)) != x at n={n} rows={rows}")
    return "6 shapes incl. non-divisible + deinterleave round-trip"


def t_geometry(rng, v):
    """css_enc_geometry vs the arithmetic in css_encoder.encode()."""
    for chunk_size, rs_t, conv_code, diff_enc, il_rows in FRAME_GEOMS:
        g = css_enc_c.geometry(chunk_size, rs_t, conv_code, diff_enc, il_rows)
        parity = rs_t * 2
        base_bits = (1 + chunk_size + 4 + parity) * 8
        conv_bits = (base_bits + 6) * 2 if conv_code == "k7r12" else base_bits
        if il_rows > 1:
            frame_symbols = il_rows * (-(-conv_bits // il_rows))
        else:
            frame_symbols = conv_bits
        check("geometry", g.base_bits == base_bits,
              f"base_bits {g.base_bits} != {base_bits}")
        check("geometry", g.conv_bits == conv_bits,
              f"conv_bits {g.conv_bits} != {conv_bits}")
        check("geometry", g.frame_symbols == frame_symbols,
              f"frame_symbols {g.frame_symbols} != {frame_symbols} "
              f"(cs={chunk_size} rs={rs_t} conv={conv_code} il={il_rows})")
    return f"{len(FRAME_GEOMS)} configs"


def _python_frame_bits(chunk_size, rs_t, conv_code, diff_enc, il_rows,
                       seq, data, diff_ref):
    """The frame pipeline exactly as css_encoder.encode() runs it."""
    codec = css_encoder._make_rs_codec(rs_t)
    crc = css_encoder.frame_crc(seq, data)
    inner = bytes([seq]) + data + crc
    frame = css_encoder.rs_encode_frame(inner, codec)
    bits = []
    for byte in frame:
        bits.extend(css_encoder.get_bits(byte))
    if diff_enc:
        bits = css_encoder.diff_encode_bits(bits, ref_bit=diff_ref)
    if conv_code == "k7r12":
        bits = css_encoder.conv_encode_k7(bits)
    if il_rows > 1:
        bits = css_encoder.interleave_bits(bits, il_rows)
    return inner, frame, bits


def t_frame(rng, v):
    """Whole-frame equivalence: inner frame bytes AND the emitted bit stream.

    RS runs on the Python side for both, because it is not part of css_enc.c
    by design -- the C caller pipes css_enc_inner through rs255_encode.  Using
    one RS for both is what keeps this test about the transmit chain rather
    than about Reed-Solomon.
    """
    n = 0
    for chunk_size, rs_t, conv_code, diff_enc, il_rows in FRAME_GEOMS:
        g = css_enc_c.geometry(chunk_size, rs_t, conv_code, diff_enc, il_rows)
        for trial in range(4):
            seq = int(rng.integers(0, 256))
            data = bytes(rng.integers(0, 256, chunk_size, dtype=np.uint8))
            diff_ref = int(rng.integers(0, 2))
            inner, frame, want = _python_frame_bits(
                chunk_size, rs_t, conv_code, diff_enc, il_rows,
                seq, data, diff_ref)

            got_inner = css_enc_c.enc_inner(g, seq, data)
            check("frame/inner", got_inner == inner,
                  f"cs={chunk_size} seq={seq} {got_inner.hex()} != {inner.hex()}")

            got = css_enc_c.enc_bits(g, frame, diff_ref)
            check("frame/bits", len(got) == len(want),
                  f"len {len(got)} != {len(want)} at cs={chunk_size} "
                  f"rs={rs_t} conv={conv_code} diff={diff_enc} il={il_rows}")
            # `bad` is computed with a default rather than inline in the
            # message: check()'s detail argument is evaluated eagerly, so a
            # bare next() over an equal pair raises StopIteration on the
            # PASSING path and reports a failure that is not there.
            bad = next((i for i, (a, b) in enumerate(zip(got, want))
                        if a != b), None)
            check("frame/bits", got == want,
                  f"cs={chunk_size} rs={rs_t} conv={conv_code} "
                  f"diff={diff_enc} il={il_rows} trial={trial} "
                  f"first_diff={bad}")
            n += 1
    return f"{n} frames over {len(FRAME_GEOMS)} configs"


def t_chirp_f64(rng, v):
    """Stage 1: float64-phase C chirp vs numpy.  A difference here is a port bug."""
    worst = 0.0
    worst_g = None
    for fs, f0, f1, dur in CHIRP_GEOMS:
        for lo, hi in ((f0, f1), (f1, f0)):       # up AND down sweeps
            want = css_encoder.generate_chirp(lo, hi, dur, fs)
            got = css_enc_c.chirp(lo, hi, dur, fs, f64=True)
            check("chirp/f64", got.shape == want.shape,
                  f"n {got.shape} != {want.shape} at fs={fs} dur={dur}")
            d = float(np.max(np.abs(got - want)))
            if d > worst:
                worst, worst_g = d, (fs, lo, hi, dur)
            # float32 STORAGE round-off only: both computed the phase in
            # double, so this is the last-ulp difference between numpy's sin
            # and libm's, not an arithmetic disagreement.
            check("chirp/f64", d < 1e-6,
                  f"maxabs={d:.3e} at fs={fs} {lo}->{hi} dur={dur}")
    return f"{len(CHIRP_GEOMS) * 2} sweeps, worst maxabs={worst:.2e} at {worst_g}"


def t_chirp_f32(rng, v):
    """Stage 2: the SHIPPING float32-phase build, graded on correlation.

    Equality is the wrong test here and asserting it would force the default
    back to double for no link benefit.  What matters is how much matched-
    filter output the precision choice costs, which is the correlation
    against the float64 chirp.
    """
    worst = 1.0
    worst_g = None
    for fs, f0, f1, dur in CHIRP_GEOMS:
        for lo, hi in ((f0, f1), (f1, f0)):
            ref = css_enc_c.chirp(lo, hi, dur, fs, f64=True).astype(np.float64)
            got = css_enc_c.chirp(lo, hi, dur, fs).astype(np.float64)
            corr = float(np.dot(ref, got)
                         / np.sqrt(np.dot(ref, ref) * np.dot(got, got)))
            if corr < worst:
                worst, worst_g = corr, (fs, lo, hi, dur)
            check("chirp/f32", corr >= CHIRP_CORR_MIN,
                  f"corr={corr:.7f} < {CHIRP_CORR_MIN} at fs={fs} "
                  f"{lo}->{hi} dur={dur} -- float32 phase is no longer "
                  f"adequate; see the PHASE PRECISION note in css_enc.h")
    loss_db = -10.0 * np.log10(max(worst, 1e-12))
    return (f"{len(CHIRP_GEOMS) * 2} sweeps, worst corr={worst:.7f} "
            f"({loss_db:.5f} dB loss) at {worst_g}")


def t_chirp_vs_mf_template(rng, v):
    """The C chirp must match the DECODER's template, not just the encoder's.

    css_fec.generate_chirp is a separate copy of the same formula, used to
    build the matched-filter reference.  The two Python copies drifting apart
    would be invisible to a test that only compares C against one of them --
    and a TX/RX template mismatch shows up only as correlation loss, never as
    an error.
    """
    import css_fec
    for fs, f0, f1, dur in CHIRP_GEOMS:
        tmpl = css_fec.generate_chirp(f0, f1, dur, fs)
        got = css_enc_c.chirp(f0, f1, dur, fs, f64=True)
        check("chirp/template", tmpl.shape == got.shape,
              f"decoder template n={tmpl.shape} != C n={got.shape} "
              f"at fs={fs} dur={dur}")
        check("chirp/template", float(np.max(np.abs(tmpl - got))) < 1e-6,
              f"decoder template differs from C chirp at fs={fs} dur={dur}")
    return f"{len(CHIRP_GEOMS)} geometries vs css_fec.generate_chirp"


TESTS_LIST = [
    ("crc32", t_crc32),
    ("bytes_to_bits", t_bytes_to_bits),
    ("conv_encode_k7", t_conv),
    ("diff_encode", t_diff),
    ("interleave", t_interleave),
    ("geometry", t_geometry),
    ("frame", t_frame),
    ("chirp_f64_vs_numpy", t_chirp_f64),
    ("chirp_f32_precision", t_chirp_f32),
    ("chirp_vs_mf_template", t_chirp_vs_mf_template),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="run only tests whose name contains this")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()

    for f64 in (False, True):
        if not css_enc_c.available(f64):
            rec = __import__("cbackend").BACKENDS[
                "css_enc64" if f64 else "css_enc"].probe()
            print(f"SKIP: C core not built ({rec['reason']})")
            print("  build it with: make -C tests/CSS/c")
            return 0

    cases = [(n, f) for n, f in TESTS_LIST
             if not args.only or args.only in n]
    print(f"CSS ENCODER C-PORT EQUIVALENCE -- {len(cases)} checks\n")
    rng = np.random.default_rng(args.seed)
    failed = 0
    for name, fn in cases:
        try:
            detail = fn(rng, args.verbose)
            print(f"  PASS  {name:<24} {detail}")
        except Fail as e:
            failed += 1
            print(f"  FAIL  {name:<24} {e}")
        except Exception as e:      # noqa: BLE001
            failed += 1
            print(f"  ERROR {name:<24} {type(e).__name__}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print()
    if failed:
        print(f"*** {failed} FAILED ***")
        return 1
    print("*** ALL PASS ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
