# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""rs255_regress.py -- the C Reed-Solomon core against `reedsolo`, byte for byte.

Run:  python tests/regression/rs255_regress.py            (exit 0 = pass)
      python tests/regression/rs255_regress.py --trials 5000 -v

Why this suite is adversarial rather than a smoke test: a wrong RS does NOT
fail loudly.  It hands back plausible bytes, and the damage surfaces later as a
checksum mismatch or a frame that "just didn't decode".  So every case here
compares the C against reedsolo on THREE things at once -- the corrected
message, the errata count the modems use as their error metric, and whether the
decode raises at all -- and the error counts sweep from zero to past the
correction capacity, where the right answer is "both must fail".

Covered:
  * encode, including the multi-block chunking (255-nsym per block) that
    _rs_encoded_len derives frame geometry from;
  * decode at 0..t errors (must repair), and t+1..t+3 (must refuse);
  * message lengths either side of a block boundary, where an off-by-one in
    the chunking would otherwise hide;
  * nsym values the repo actually builds: 10 (RobustLink), 16 (OFDM data),
    plus 2/4/32 as range checks.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parent.parent
for _p in (_TESTS_ROOT, _TESTS_ROOT / "CSS", _TESTS_ROOT / "MESH",
           _TESTS_ROOT / "OFDM", _TESTS_ROOT / "QAM"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rs255_c                                                   # noqa: E402

try:
    from reedsolo import RSCodec as RefRSCodec, ReedSolomonError
except ImportError:                                              # pragma: no cover
    print("SKIP: reedsolo not installed, nothing to compare against")
    sys.exit(0)

NSYMS = (2, 4, 10, 16, 32)
# Lengths that straddle a block boundary at nsym=16 (k=239) and nsym=10 (k=245).
LENGTHS = (1, 2, 17, 100, 238, 239, 240, 244, 245, 246, 300, 490, 512)

_fails = 0
_checks = 0


def check(ok, label, detail="", verbose=False):
    global _fails, _checks
    _checks += 1
    _fails += (not ok)
    if not ok or verbose:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<44} {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=1200,
                    help="randomised decode trials per nsym (default 1200)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not rs255_c.available():
        print(f"SKIP: C core not built: {rs255_c._LIB_PATH}")
        print(f"      run: make -C {_TESTS_ROOT / 'c'}")
        return 0

    rnd = random.Random(20260814)

    for nsym in NSYMS:
        ref = RefRSCodec(nsym)
        got = rs255_c.RSCodecC(nsym)
        t = nsym // 2
        k = got.block_k

        # --- encode, including multi-block ---
        bad = None
        for L in LENGTHS:
            msg = bytes(rnd.randrange(256) for _ in range(L))
            a, b = bytes(ref.encode(msg)), bytes(got.encode(msg))
            if a != b:
                bad = f"len {L}: {len(a)}B ref vs {len(b)}B C"
                break
        check(bad is None, f"nsym={nsym} encode (incl. chunking)",
              bad or f"{len(LENGTHS)} lengths, block_k={k}", args.verbose)

        # --- decode: repairable, and beyond capacity ---
        n_rep = n_ref = 0
        bad = None
        for _ in range(args.trials):
            L = rnd.choice(LENGTHS)
            msg = bytes(rnd.randrange(256) for _ in range(L))
            enc = bytearray(ref.encode(msg))
            # Errors are spread over the WHOLE stream, so a multi-block message
            # gets them distributed across blocks -- each block has its own
            # capacity and the totals must still agree.
            nerr = rnd.randint(0, t + 3)
            for p in rnd.sample(range(len(enc)), min(nerr, len(enc))):
                enc[p] ^= rnd.randrange(1, 256)

            try:
                rmsg, _, rerr = ref.decode(bytearray(enc))
                r_ok, r_n = True, len(rerr)
            except ReedSolomonError:
                r_ok, r_n, rmsg = False, None, None

            try:
                cmsg, _, cerr = got.decode(bytearray(enc))
                c_ok, c_n = True, len(cerr)
            except rs255_c.ReedSolomonError:
                c_ok, c_n, cmsg = False, None, None

            if r_ok != c_ok:
                bad = f"L={L} nerr={nerr}: ref_ok={r_ok} c_ok={c_ok}"
                break
            if r_ok:
                if bytes(rmsg) != bytes(cmsg):
                    bad = f"L={L} nerr={nerr}: corrected bytes differ"
                    break
                if r_n != c_n:
                    bad = f"L={L} nerr={nerr}: errata {c_n} != {r_n}"
                    break
                n_rep += 1
            else:
                n_ref += 1
        check(bad is None, f"nsym={nsym} decode vs reedsolo",
              bad or f"{args.trials} trials: {n_rep} repaired, "
                     f"{n_ref} correctly refused (t={t})", args.verbose)

    # --- the guardrails that stop a silently-wrong configuration ---
    bad = None
    try:
        rs255_c.RSCodecC(16, fcr=1)
        bad = "accepted fcr=1"
    except ValueError:
        pass
    if bad is None:
        try:
            rs255_c.RSCodecC(16).decode(bytearray(b"\x00" * 40),
                                        erase_pos=[1, 2])
            bad = "accepted erasures it cannot perform"
        except NotImplementedError:
            pass
    check(bad is None, "rejects unsupported configurations",
          bad or "non-default field and erasures both refused", args.verbose)

    print(f"\nRS255 REGRESS {'PASS' if _fails == 0 else f'FAIL ({_fails})'} "
          f"({_checks} checks)")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
