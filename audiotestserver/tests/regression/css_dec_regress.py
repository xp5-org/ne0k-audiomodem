# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""CSS receive-pipeline C-port equivalence suite.

Grades tests/CSS/c/css_dec.c against css_fec.py and css_decoder._frame_pipeline
-- deinterleave, soft-decision Viterbi, differential decode, bit packing, CRC.

    python tests/regression/css_dec_regress.py
    python tests/regression/css_dec_regress.py -v

The headline check is `roundtrip`: encode a frame with the C TX chain, push it
through a simulated soft channel at a range of SNRs, and decode it with the C
RX chain, asserting the payload comes back AND that Python's _frame_pipeline
agrees bit for bit on the same input.  Every stage is also compared on its own,
because a whole-pipeline test can only say "the bytes differ".

VITERBI PRECISION.  Path metrics accumulate over the whole frame, so float32
and float64 can in principle pick different survivors when two paths are
within a float32 ulp of each other.  `viterbi_precision` measures how often
that actually changes a decoded bit, deliberately including SNRs low enough
that the decoder is failing anyway -- the interesting question is not whether
the two agree on easy frames but whether float32 is ever WORSE, and near-ties
are dense exactly where the decoder is struggling.
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
import css_fec                        # noqa: E402
import css_decoder                    # noqa: E402


# (chunk_size, rs_t, conv_code, diff_enc, interleave_rows)
FRAME_GEOMS = [
    (1,  0, None,    False, 0),
    (2,  0, "k7r12", False, 0),
    (12, 0, "k7r12", True,  0),
    (8,  0, "k7r12", True,  8),
    (16, 0, None,    False, 5),
    (48, 0, "k7r12", False, 16),
]

# Soft-channel SNRs to sweep, in "confidence" terms rather than dB: the MF
# hands the frame pipeline values in [-1, +1], so the channel model here adds
# Gaussian noise to an ideal +-1 and clips.  1.2 is comfortable, 0.35 is past
# the point where frames start failing -- which is where near-ties live.
NOISE_SIGMAS = [0.10, 0.30, 0.55, 0.80, 1.20]


class Fail(Exception):
    pass


def check(name, ok, detail=""):
    if not ok:
        raise Fail(f"{name}: {detail}")


def soft_channel(bits, sigma, rng):
    """Transmitted bits -> soft confidences in [-1, +1].

    Not a chirp channel: this suite is about the FRAME pipeline, and putting a
    real matched filter in front of it would make a Viterbi disagreement
    indistinguishable from an MF disagreement.  css_regress.py covers the
    acoustic path end to end; this covers the bits.
    """
    ideal = np.where(np.asarray(bits) > 0, 1.0, -1.0)
    noisy = ideal + rng.normal(0.0, sigma, size=ideal.shape)
    return np.clip(noisy, -1.0, 1.0).astype(np.float32)


def _py_pipeline(g, bits, conf, diff_ref, conv_code, il_rows):
    """css_decoder._frame_pipeline's transforms, stopping before RS.

    Reimplemented rather than called because _frame_pipeline runs RS and CRC
    in the same breath and returns only the verdict; this suite needs the
    intermediate bytes so a mismatch names the stage.  The ORDER and each step
    are lifted from it directly -- see t_pipeline_matches_decoder, which pins
    this reimplementation against the real thing.
    """
    fb = list(bits)
    fc = list(conf)
    if il_rows > 1:
        n_coded = len(fb)
        cols = n_coded // il_rows
        idxs = (np.arange(n_coded).reshape(cols, il_rows).T.flatten().tolist())
        fb = [fb[i] for i in idxs]
        fc = [fc[i] for i in idxs]
    if conv_code == "k7r12":
        sp = [(fc[i], fc[i + 1]) for i in range(0, len(fc) - 1, 2)]
        fb, pe = css_fec.viterbi_k7_with_reliability(sp)
        me = max(pe) if pe else 1.0
        fc = [e / (me + 1e-12) for e in pe]
    if g.diff_enc:
        fb = css_fec.diff_decode_bits(fb, ref_bit=diff_ref)
    raw = css_fec.get_bytes_from_bits(fb[:g.base_bits])
    return raw, fc


# ── stage checks ─────────────────────────────────────────────────────────────

def t_deinterleave(rng, v):
    for n, rows in ((40, 4), (40, 8), (256, 16), (10, 1), (10, 0)):
        bits = list(rng.integers(0, 2, n).tolist())
        conf = rng.uniform(-1, 1, n).astype(np.float32)
        gb, gc = css_enc_c.deinterleave(bits, conf, rows)
        want_b = css_fec.deinterleave_bits(bits, rows, n)
        check("deinterleave", gb == want_b, f"bits differ at n={n} rows={rows}")
        if rows > 1:
            # The confidences must follow the SAME permutation.  The index
            # map is built here with numpy rather than by pushing range(n)
            # through css_fec.deinterleave_bits: that helper casts to int8,
            # so it silently cannot carry an index past 127 -- it raises on
            # n=256, which is a real limit of the helper and not of the C.
            idx = (np.arange(n).reshape(n // rows, rows).T.flatten().tolist())
            want_c = [float(conf[i]) for i in idx]
            check("deinterleave", np.allclose(gc, want_c),
                  f"conf did not follow bits at n={n} rows={rows}")
        # Inverse of the transmit interleaver, which is the property that
        # actually matters on the wire.
        if rows > 1:
            tx = css_enc_c.interleave(bits, rows)
            back, _ = css_enc_c.deinterleave(tx, None, rows)
            check("deinterleave", back == bits,
                  f"deinterleave(interleave(x)) != x at n={n} rows={rows}")
    return "5 shapes, bits + conf + TX inverse"


def t_diff_decode(rng, v):
    for n in (1, 8, 137):
        for ref in (0, 1):
            bits = list(rng.integers(0, 2, n).tolist())
            want = css_fec.diff_decode_bits(bits, ref_bit=ref)
            check("diff_decode", css_enc_c.diff_decode(bits, ref) == want,
                  f"n={n} ref={ref}")
    return "3 lengths x 2 refs"


def t_bits_to_bytes(rng, v):
    for n in (8, 64, 65, 71, 256):
        bits = list(rng.integers(0, 2, n).tolist())
        want = css_fec.get_bytes_from_bits(bits)
        check("bits_to_bytes", css_enc_c.bits_to_bytes(bits) == want,
              f"n={n}")
    return "5 lengths incl. non-multiples of 8"


def t_check_crc(rng, v):
    for chunk_size in (1, 4, 12, 48):
        g = css_enc_c.geometry(chunk_size, 0, None, False, 0)
        data = bytes(rng.integers(0, 256, chunk_size, dtype=np.uint8))
        seq = int(rng.integers(0, 256))
        raw = css_enc_c.enc_inner(g, seq, data)
        ok, got_seq, got_data = css_enc_c.check_crc(raw, chunk_size)
        want = css_fec.check_crc(raw, chunk_size)
        check("check_crc", (ok, got_seq, got_data) == want,
              f"cs={chunk_size} C={(ok, got_seq, got_data)} py={want}")
        check("check_crc", ok and got_data == data, f"good frame rejected")
        # One flipped bit anywhere must be rejected.  Walking every byte
        # rather than one at random: a CRC that ignores its last byte, or
        # checks the payload but not the seq, passes a spot check.
        for i in range(len(raw)):
            bad = bytearray(raw)
            bad[i] ^= 0x01
            bok, _, _ = css_enc_c.check_crc(bytes(bad), chunk_size)
            check("check_crc", not bok,
                  f"corrupt byte {i} of {len(raw)} accepted (cs={chunk_size})")
        # Short buffer is a reject, not a read past the end.
        sok, _, _ = css_enc_c.check_crc(raw[:-1], chunk_size)
        check("check_crc", not sok, "truncated frame accepted")
    return "4 chunk sizes, every single-bit corruption + truncation"


def t_viterbi_clean(rng, v):
    """Noise-free: the C Viterbi must recover exactly what was encoded."""
    for n in (8, 40, 137, 400):
        bits = list(rng.integers(0, 2, n).tolist())
        coded = css_enc_c.conv_encode_k7(bits)
        conf = np.where(np.array(coded) > 0, 1.0, -1.0).astype(np.float32)
        got, _ = css_enc_c.viterbi_k7(conf)
        check("viterbi_clean", got == bits,
              f"n={n} recovered {sum(a != b for a, b in zip(got, bits))} wrong")
    return "4 lengths, exact recovery"


def t_viterbi_vs_python(rng, v):
    """Stage 1: the f64-metric C build vs css_fec.viterbi_k7_with_reliability.

    Python computes its metrics in float64, so the f64 build should agree
    exactly and any difference is a PORT bug -- a wrong generator, a
    misordered trellis, an off-by-one traceback.
    """
    trials = 0
    for n in (40, 137, 400):
        for sigma in NOISE_SIGMAS:
            for _ in range(3):
                bits = list(rng.integers(0, 2, n).tolist())
                coded = css_enc_c.conv_encode_k7(bits)
                conf = soft_channel(coded, sigma, rng)
                sp = [(float(conf[i]), float(conf[i + 1]))
                      for i in range(0, len(conf) - 1, 2)]
                want_b, want_e = css_fec.viterbi_k7_with_reliability(sp)
                got_b, got_e = css_enc_c.viterbi_k7(conf, f64=True)
                check("viterbi/f64", got_b == want_b,
                      f"n={n} sigma={sigma} "
                      f"{sum(a != b for a, b in zip(got_b, want_b))} bits differ")
                check("viterbi/f64", np.allclose(got_e, want_e, atol=1e-6),
                      f"energies differ at n={n} sigma={sigma}")
                trials += 1
    return f"{trials} frames vs Python float64, bits AND energies exact"


def t_viterbi_precision(rng, v):
    """Stage 2: the SHIPPING float32-metric build vs the float64 one.

    Reported rather than merely asserted: the number is the evidence behind
    the PATH-METRIC PRECISION note in css_dec.h, and a run where it stops
    being zero is worth seeing even if it stays under the tolerance.
    """
    diff_bits = 0
    total_bits = 0
    frames = 0
    for n in (137, 400):
        for sigma in NOISE_SIGMAS:
            for _ in range(8):
                bits = list(rng.integers(0, 2, n).tolist())
                coded = css_enc_c.conv_encode_k7(bits)
                conf = soft_channel(coded, sigma, rng)
                a, _ = css_enc_c.viterbi_k7(conf, f64=True)
                b, _ = css_enc_c.viterbi_k7(conf)
                diff_bits += sum(x != y for x, y in zip(a, b))
                total_bits += len(a)
                frames += 1
    frac = diff_bits / max(total_bits, 1)
    # Zero is what this has measured; the assert is a hair looser so a single
    # near-tie on some other libm does not fail the suite, while a real
    # precision problem (which shows up as percentages, not ppm) does.
    check("viterbi/f32", frac < 1e-4,
          f"float32 metrics changed {diff_bits}/{total_bits} bits "
          f"({frac:.2e}) -- see PATH-METRIC PRECISION in css_dec.h")
    return (f"{frames} frames, {diff_bits}/{total_bits} bits differ "
            f"({frac:.2e}) from float64 metrics")


def t_pipeline_matches_decoder(rng, v):
    """Pin this file's _py_pipeline against the real _frame_pipeline.

    _py_pipeline exists so a mismatch can name the stage, but a
    reimplementation that has quietly drifted from the decoder would make
    every other check in this suite grade the wrong reference.  This asserts
    the two reach the same verdict on the same input.
    """
    n = 0
    for chunk_size, rs_t, conv_code, diff_enc, il_rows in FRAME_GEOMS:
        g = css_enc_c.geometry(chunk_size, rs_t, conv_code, diff_enc, il_rows)
        for sigma in (0.1, 0.55):
            data = bytes(rng.integers(0, 256, chunk_size, dtype=np.uint8))
            seq = int(rng.integers(0, 256))
            diff_ref = int(rng.integers(0, 2))
            inner = css_enc_c.enc_inner(g, seq, data)
            tx = css_enc_c.enc_bits(g, inner, diff_ref)
            conf = soft_channel(tx, sigma, rng)
            hard = [1 if c > 0 else 0 for c in conf]

            raw, _ = _py_pipeline(g, hard, conf, diff_ref, conv_code, il_rows)
            ok_mine, seq_mine, data_mine = css_fec.check_crc(raw, chunk_size)

            ok_real, seq_real, data_real, _, _ = css_decoder._frame_pipeline(
                hard, list(conf), interleave_rows=il_rows,
                conv_code=conv_code, diff_enc=diff_enc, diff_ref=diff_ref,
                base_bits=g.base_bits, rs_codec=None, chunk_size=chunk_size)
            check("pipeline_pin",
                  (ok_mine, seq_mine, data_mine) == (ok_real, seq_real, data_real),
                  f"local pipeline disagrees with _frame_pipeline at "
                  f"cs={chunk_size} conv={conv_code} il={il_rows} sigma={sigma}")
            n += 1
    return f"{n} frames vs css_decoder._frame_pipeline"


def t_roundtrip(rng, v):
    """C TX -> soft channel -> C RX, cross-checked against Python's RX.

    Two assertions per frame, and they are different assertions:
      1. the C RX agrees with the Python RX BIT FOR BIT on the same input --
         the port check, which must hold at every SNR including ones where
         both are wrong;
      2. at low noise the payload actually comes back -- the sanity check,
         which catches two implementations that agree because both are broken.
    """
    n = 0
    recovered = 0
    clean = 0
    for chunk_size, rs_t, conv_code, diff_enc, il_rows in FRAME_GEOMS:
        g = css_enc_c.geometry(chunk_size, rs_t, conv_code, diff_enc, il_rows)
        for sigma in NOISE_SIGMAS:
            for _ in range(3):
                data = bytes(rng.integers(0, 256, chunk_size, dtype=np.uint8))
                seq = int(rng.integers(0, 256))
                diff_ref = int(rng.integers(0, 2))
                inner = css_enc_c.enc_inner(g, seq, data)
                tx = css_enc_c.enc_bits(g, inner, diff_ref)
                conf = soft_channel(tx, sigma, rng)
                hard = [1 if c > 0 else 0 for c in conf]

                want_raw, _ = _py_pipeline(g, hard, conf, diff_ref,
                                           conv_code, il_rows)
                got_raw, _ = css_enc_c.dec_raw(g, hard, conf, diff_ref,
                                               f64=True)
                check("roundtrip", got_raw == want_raw,
                      f"C RX != Python RX at cs={chunk_size} conv={conv_code} "
                      f"diff={diff_enc} il={il_rows} sigma={sigma}: "
                      f"{got_raw.hex()} vs {want_raw.hex()}")

                ok, got_seq, got_data = css_enc_c.check_crc(got_raw, chunk_size)
                if ok:
                    check("roundtrip", got_seq == seq and got_data == data,
                          f"CRC passed but payload wrong at sigma={sigma}")
                    recovered += 1
                # Only the near-noiseless point is asserted for UNCODED
                # configs: at sigma=0.30 a bare 80-bit frame flips a bit
                # roughly 7% of the time, which is the channel behaving as
                # specified.  Convolutionally coded frames are held to the
                # higher noise level as well.
                if sigma <= 0.10 or (conv_code and sigma <= 0.30):
                    clean += 1
                    check("roundtrip", ok,
                          f"clean frame FAILED CRC at sigma={sigma} "
                          f"cs={chunk_size} conv={conv_code} il={il_rows}")
                n += 1
    return (f"{n} frames over {len(FRAME_GEOMS)} configs x "
            f"{len(NOISE_SIGMAS)} SNRs; {recovered} CRC-ok, "
            f"{clean} low-noise frames all recovered")


def t_roundtrip_f32(rng, v):
    """The same round trip on the SHIPPING build, graded on payload recovery.

    Not on byte equality against float64: that is what t_viterbi_precision
    measures, and asserting it here too would just report the same fact twice
    under a name that hides which precision was at fault.
    """
    n = 0
    for chunk_size, rs_t, conv_code, diff_enc, il_rows in FRAME_GEOMS:
        g = css_enc_c.geometry(chunk_size, rs_t, conv_code, diff_enc, il_rows)
        # Same rationale as t_roundtrip: 0.30 is only a "must recover" point
        # when the convolutional code is carrying the frame.
        for sigma in ((0.10, 0.30) if conv_code else (0.10,)):
            for _ in range(4):
                data = bytes(rng.integers(0, 256, chunk_size, dtype=np.uint8))
                seq = int(rng.integers(0, 256))
                diff_ref = int(rng.integers(0, 2))
                inner = css_enc_c.enc_inner(g, seq, data)
                tx = css_enc_c.enc_bits(g, inner, diff_ref)
                conf = soft_channel(tx, sigma, rng)
                hard = [1 if c > 0 else 0 for c in conf]
                raw, _ = css_enc_c.dec_raw(g, hard, conf, diff_ref)
                ok, got_seq, got_data = css_enc_c.check_crc(raw, chunk_size)
                check("roundtrip/f32", ok and got_seq == seq and got_data == data,
                      f"shipping build lost a clean frame at sigma={sigma} "
                      f"cs={chunk_size} conv={conv_code} il={il_rows}")
                n += 1
    return f"{n} clean frames recovered on the float32 build"


def t_byte_conf(rng, v):
    """The per-byte confidence css_dec_raw exports vs the Python erasure key.

    _frame_pipeline marks byte i as an erasure when min(|fc|) over its eight
    bits falls under era_thr, so that minimum -- not some other summary -- is
    the number the C has to reproduce for erasure marking to mean the same
    thing on both sides.
    """
    n = 0
    for chunk_size, rs_t, conv_code, diff_enc, il_rows in FRAME_GEOMS:
        g = css_enc_c.geometry(chunk_size, rs_t, conv_code, diff_enc, il_rows)
        for sigma in (0.30, 0.80):
            data = bytes(rng.integers(0, 256, chunk_size, dtype=np.uint8))
            seq = int(rng.integers(0, 256))
            inner = css_enc_c.enc_inner(g, seq, data)
            tx = css_enc_c.enc_bits(g, inner, 0)
            conf = soft_channel(tx, sigma, rng)
            hard = [1 if c > 0 else 0 for c in conf]

            _, fc = _py_pipeline(g, hard, conf, 0, conv_code, il_rows)
            _, got = css_enc_c.dec_raw(g, hard, conf, 0, f64=True)
            want = []
            for bi in range(len(got)):
                b0, b1 = bi * 8, min(bi * 8 + 8, len(fc))
                want.append(min(abs(c) for c in fc[b0:b1]) if b1 > b0 else 0.0)
            check("byte_conf", np.allclose(got, want, atol=1e-6),
                  f"cs={chunk_size} conv={conv_code} il={il_rows} "
                  f"sigma={sigma} max_dev="
                  f"{float(np.max(np.abs(np.array(got) - np.array(want)))):.3e}")
            n += 1
    return f"{n} frames, per-byte min|conf| matches the erasure key"


TESTS_LIST = [
    ("deinterleave", t_deinterleave),
    ("diff_decode", t_diff_decode),
    ("bits_to_bytes", t_bits_to_bytes),
    ("check_crc", t_check_crc),
    ("viterbi_clean", t_viterbi_clean),
    ("viterbi_vs_python", t_viterbi_vs_python),
    ("viterbi_precision", t_viterbi_precision),
    ("pipeline_pin", t_pipeline_matches_decoder),
    ("roundtrip", t_roundtrip),
    ("roundtrip_f32", t_roundtrip_f32),
    ("byte_conf", t_byte_conf),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="run only tests whose name contains this")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()

    for f64 in (False, True):
        if not css_enc_c.available(f64):
            import cbackend
            rec = cbackend.BACKENDS["css_enc64" if f64 else "css_enc"].probe()
            print(f"SKIP: C core not built ({rec['reason']})")
            print("  build it with: make -C tests/CSS/c")
            return 0

    cases = [(n, f) for n, f in TESTS_LIST if not args.only or args.only in n]
    print(f"CSS DECODER C-PORT EQUIVALENCE -- {len(cases)} checks\n")
    rng = np.random.default_rng(args.seed)
    failed = 0
    for name, fn in cases:
        try:
            detail = fn(rng, args.verbose)
            print(f"  PASS  {name:<20} {detail}")
        except Fail as e:
            failed += 1
            print(f"  FAIL  {name:<20} {e}")
        except Exception as e:      # noqa: BLE001
            failed += 1
            print(f"  ERROR {name:<20} {type(e).__name__}: {e}")
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
