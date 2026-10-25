# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""CSS decoder regression harness.

Replays synthetic encodes AND real mic recordings through css_decoder offline
(no hardware) and asserts the expected decoded text.  Run this before/after
any decoder change -- it is the safety net for the hard-won fixes (MF
direction window, false-lock watchdog, soft-rearm deadlock, freewheel/wake).

Usage:
    python tests/regression/css_regress.py            # full suite
    python tests/regression/css_regress.py --only li2 # substring filter
    python tests/regression/css_regress.py -v         # decoder log on FAIL
    python tests/regression/css_regress.py --slow     # include 3.0s synthetic

How it works: sys.modules["sounddevice"] is replaced with a fake module whose
InputStream feeds the WAV in chunks.  css_decoder gates its real-hardware
paths on hasattr(sd, "query_devices"), so the bare mock automatically takes
the blocking/test path, and an empty read signals end-of-stream.

Opening WAVs directly on the Z: SMB share can raise a file-open "System
error" -- all recorded fixtures are copied to a local cache dir first, and
synthetic encodes are written locally.
"""
import argparse
import contextlib
import hashlib
import io
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import types
import wave

import numpy as np

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGRESS = os.path.join(TESTS, "regression")
FIXTURES = os.path.join(REGRESS, "fixtures")
RESULTS = os.path.join(os.path.dirname(TESTS), "results")

CACHE = os.path.join(tempfile.gettempdir(), "css_regress_cache")
WORK = os.path.join(tempfile.gettempdir(), "css_regress_work")

sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.dirname(TESTS))
sys.path.insert(0, os.path.join(TESTS, "CSS"))

import css_encoder  # noqa: E402


# ── local file cache (SMB gotcha) ────────────────────────────────────────────

def local_copy(path):
    """Copy a (possibly SMB-hosted) file to a local cache dir and return the
    local path.  Keyed by absolute path + size + mtime so a changed source is
    re-copied and an unchanged one is reused."""
    path = os.path.abspath(path)
    st = os.stat(path)
    key = hashlib.sha1(
        f"{path}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, key + "_" + os.path.basename(path))
    if not (os.path.exists(dst) and os.path.getsize(dst) == st.st_size):
        shutil.copy2(path, dst)
    return dst


def load_wav(path):
    with wave.open(path, "rb") as w:
        fs = w.getframerate()
        pcm = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
               .astype(np.float32) / 32768.0)
    return pcm, fs


def save_wav(path, pcm, fs):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())


# ── offline decode via mocked sounddevice ────────────────────────────────────

def decode_offline(pcm, fs, info_path, overrides=None):
    """Run css_decoder.decode() (which self-dispatches PLL vs MF) on an
    in-memory PCM array via a fake sounddevice module.  Returns
    (result_dict, decoder_stdout)."""
    import threading
    pos = [0]
    stop = threading.Event()

    class FakeStream:
        def __init__(s, **k): pass
        def __enter__(s): return s
        def __exit__(s, *a): return False

        def read(s, n):
            c = pcm[pos[0]:pos[0] + n]
            pos[0] += len(c)
            # Real sounddevice blocking reads always deliver exactly n frames
            # and a live mic never "ends" -- so pad/extend with silence and
            # signal end-of-input via stop_event (both decode paths poll it),
            # exactly like the wrapper's cooperative stop on hardware.
            if len(c) < n:
                stop.set()
                c = np.concatenate([c, np.zeros(n - len(c), dtype=np.float32)])
            return c.astype(np.float32), False

    fake = types.ModuleType("sounddevice")
    fake.InputStream = lambda **k: FakeStream()

    saved_dec = sys.modules.pop("css_decoder", None)
    saved_sd = sys.modules.get("sounddevice")
    sys.modules["sounddevice"] = fake
    try:
        import css_decoder
        importlib.reload(css_decoder)
        cfg = {"info": local_copy(info_path),
               "result_out": os.path.join(WORK, "_result.json"),
               "timeout_sec": 100000, "fs": fs}
        if overrides:
            cfg.update(overrides)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = css_decoder.decode(cfg, stop_event=stop)
        return res, buf.getvalue()
    finally:
        if saved_dec is not None:
            sys.modules["css_decoder"] = saved_dec
        else:
            sys.modules.pop("css_decoder", None)
        if saved_sd is not None:
            sys.modules["sounddevice"] = saved_sd
        else:
            sys.modules.pop("sounddevice", None)


# ── synthetic encode helpers ─────────────────────────────────────────────────

def encode_synth(name, **enc_overrides):
    """Encode 'hi' into WORK/<name>.wav; return (wav_path, info_path)."""
    os.makedirs(WORK, exist_ok=True)
    wav = os.path.join(WORK, name + ".wav")
    info = os.path.join(WORK, name + "_info.json")
    enc = dict(message="hi", output=wav, info_out=info,
               result_out=os.path.join(WORK, name + "_enc.json"),
               fs=48000, f_min=800, f_max=4000, chunk_size=1,
               amplitude=0.3, diff_encode=False, symbol_gap_frac=0.05)
    enc.update(enc_overrides)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        css_encoder.encode(enc)
    return wav, info


def make_dropped(name, symbol_dur=0.5, drop_syms=4, silence_s=3.0):
    """Dropped-sample resync case: encode per_frame with an inter-frame
    silence gap, then excise drop_syms whole symbol-periods from the END of
    that gap so frame 1's training arrives drop_syms symbols early.  This is
    the offline emulation of the ADC-overflow sample loss that exposed the
    soft-rearm deadlock (old decoder: SYNC_TOO_EARLY loop, 1 frame 'h';
    fixed decoder: full 'hi')."""
    wav, info = encode_synth(name, symbol_dur=symbol_dur,
                             inter_frame_silence_s=silence_s)
    pcm, fs = load_wav(wav)
    conf = json.load(open(info))
    sps = int(fs * conf["symbol_dur"])
    gap = int(sps * conf["symbol_gap_frac"])
    sps_eff = sps + gap
    lead = sps * 4                                   # lead_silence_syms default
    frame0_syms = (conf["training_symbols"] + len(conf["preamble"])
                   + conf["frame_symbols"])
    gap_slots = max(1, int(round(silence_s * fs / sps_eff)))
    gap_end = lead + frame0_syms * sps_eff + gap_slots * sps_eff
    cut = drop_syms * sps_eff
    pcm2 = np.concatenate([pcm[:gap_end - cut], pcm[gap_end:]])
    wav2 = os.path.join(WORK, name + "_dropped.wav")
    save_wav(wav2, pcm2, fs)
    return wav2, info


# ── case table ───────────────────────────────────────────────────────────────
# Each case: name, how to obtain (wav, info), expected text, extra checks.

def _expected_from_run(run_id):
    d = json.load(open(os.path.join(RESULTS, run_id,
                                    "decode_result_laptop.json")))
    return d.get("decoded_text", "")


def load_promoted_cases():
    """Cases promoted from the server UI's Reports tab.

    `cases.json` is written by shared.promote_run(); each entry points at a
    WAV + info sidecar that was COPIED into fixtures/runs/<name>/ at promotion
    time, so the case keeps working after the original results/<run_id>/ dir
    is deleted.  Missing files are left in the list and reported as SKIP by
    run_case(), same as the hand-written fixture cases.
    """
    path = os.path.join(REGRESS, "cases.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"  WARN: cannot read cases.json: {e}")
        return []
    entries = data.get("cases", []) if isinstance(data, dict) else data
    cases = []
    for c in entries:
        if not (c.get("name") and c.get("wav") and c.get("info")):
            continue
        wav = os.path.join(REGRESS, c["wav"])
        info = os.path.join(REGRESS, c["info"])
        cases.append(dict(
            name=c["name"],
            make=lambda w=wav, i=info: (w, i),
            expect=c.get("expect", ""),
            labels=c.get("labels", []),
            note=c.get("note", ""),
        ))
    return cases


def build_cases(slow=False):
    cases = []

    # Synthetic per_frame roundtrips (MF path via decode() auto-dispatch)
    durs = [0.5, 1.0] + ([3.0] if slow else [])
    for dur in durs:
        cases.append(dict(
            name=f"synth_perframe_{dur}s",
            make=lambda dur=dur: encode_synth(f"pf_{dur}", symbol_dur=dur),
            expect="hi"))

    # Short-symbol roundtrip -> PLL decode() path (dur <= mf_longsym_min_dur)
    cases.append(dict(
        name="synth_pll_0.25s",
        make=lambda: encode_synth("pll_025", symbol_dur=0.25),
        expect="hi"))

    # Wake mode: full preamble once, 15s suspend, 2-symbol wake word
    cases.append(dict(
        name="synth_wake_15s_suspend",
        make=lambda: encode_synth("wake", symbol_dur=0.5,
                                  preamble_mode="wake", wake_word=[1, 0],
                                  inter_frame_silence_s=15.0),
        expect="hi",
        check=lambda r: r.get("diagnostics", {}).get("wakes", 0) >= 1
        or "no wake detected"))

    # Reed-Solomon roundtrip.  The rest of the synthetic cases encode at rs_t=0.
    # rs_t=2 gives 4 parity bytes and a codec invoked on both encode and
    # decode, so this is the case that exercises the RS_BACKEND selection.
    cases.append(dict(
        name="synth_rs_t2_0.5s",
        make=lambda: encode_synth("rs_t2", symbol_dur=0.5, rs_t=2),
        expect="hi"))

    # Dropped-sample resync (soft-rearm deadlock regression)
    cases.append(dict(
        name="synth_dropped_4sym",
        make=lambda: make_dropped("drop4"),
        expect="hi"))

    # Real mic recordings (rescued fixtures) -- validated the MF-direction fix
    for rec in ("li2", "co2"):
        cases.append(dict(
            name=f"mic_{rec}_1.0s",
            make=lambda rec=rec: (os.path.join(FIXTURES, rec + ".wav"),
                                  os.path.join(FIXTURES, "info2.json")),
            expect="hi"))

    # Freewheel/wake recording fixture
    cases.append(dict(
        name="synthrec_fw_wake",
        make=lambda: (os.path.join(FIXTURES, "fw.wav"),
                      os.path.join(FIXTURES, "fw_info.json")),
        expect="hi",
        check=lambda r: r.get("diagnostics", {}).get("wakes", 0) >= 1
        or "no wake detected"))

    # Real mic captures from result runs (0.5s symbols, laptop mic)
    for run_id in ("20260706_204233_css", "20260706_200807_css"):
        wav = os.path.join(RESULTS, run_id, "laptop_input.wav")
        info = os.path.join(RESULTS, run_id, "css_info.json")
        if not os.path.exists(wav) or os.path.getsize(wav) == 0:
            continue        # known gotcha: capture thread can leave 0-byte WAV
        cases.append(dict(
            name=f"mic_run_{run_id}",
            make=lambda w=wav, i=info: (w, i),
            expect=_expected_from_run(run_id)))

    # Cold-start first-frame drop regression (0.75s, amp 0.01, diff off): frame 0's
    # Barker preamble decoded 11/13=0.846 on the still-settling grid, just under the
    # old mf_preamble_thr=0.85, so sync fired only on frame 1 -> "i" instead of "hi".
    # The 0.80 gate (tolerates 2 errored preamble symbols) recovers frame 0.
    _cold = os.path.join(RESULTS, "20260712_042306_css")
    _cold_wav = os.path.join(_cold, "laptop_input.wav")
    _cold_info = os.path.join(_cold, "css_info.json")
    if os.path.exists(_cold_wav) and os.path.getsize(_cold_wav) > 0:
        cases.append(dict(
            name="mic_coldstart_0.75s_2frame",
            make=lambda w=_cold_wav, i=_cold_info: (w, i),
            expect="hi"))

    # Cases promoted from the Reports tab of the server UI
    cases.extend(load_promoted_cases())

    return cases


# ── runner ───────────────────────────────────────────────────────────────────

def run_case(case, verbose):
    t0 = time.time()
    try:
        wav, info = case["make"]()
        # Recorded fixtures (WAVs) are gitignored and kept only on local disk;
        # a fresh clone won't have them, so skip rather than fail.  Synthetic
        # cases generate their own WAV in make() and always exist.
        if not (os.path.exists(wav) and os.path.exists(info)):
            return None, "SKIP (fixture not on disk)", "", time.time() - t0
        pcm, fs = load_wav(local_copy(wav))
        res, log = decode_offline(pcm, fs, info)
    except Exception as e:
        return False, f"EXCEPTION {type(e).__name__}: {e}", "", time.time() - t0
    got = res.get("decoded_text")
    ok = res.get("status") == "ok" and got == case["expect"]
    detail = (f"status={res.get('status')} text={got!r} "
              f"expect={case['expect']!r}")
    if ok and "check" in case:
        c = case["check"](res)
        if c is not True:
            ok = False
            detail += f" extra-check: {c}"
    return ok, detail, log, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="CSS decoder regression suite")
    ap.add_argument("--only", help="run only cases whose name contains this")
    ap.add_argument("--label", action="append", default=[],
                    help="run only promoted cases carrying this label "
                         "(repeatable; matches any)")
    ap.add_argument("--list", action="store_true",
                    help="list cases (name, labels, expect) and exit")
    ap.add_argument("--slow", action="store_true",
                    help="include slow cases (3.0s symbol_dur)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print decoder log for failing cases")
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    cases = build_cases(slow=args.slow)
    if args.only:
        cases = [c for c in cases if args.only in c["name"]]
    if args.label:
        want = set(args.label)
        cases = [c for c in cases if want & set(c.get("labels", []))]
    if not cases:
        print("no cases match filter"); return 1

    if args.list:
        for c in cases:
            labels = ",".join(c.get("labels", [])) or "-"
            print(f"  {c['name']:32s} labels={labels:24s} expect={c['expect']!r}")
        print(f"\n{len(cases)} case(s)")
        return 0

    print(f"CSS DECODER REGRESSION -- {len(cases)} cases\n")
    all_ok = True
    n_skip = 0
    for case in cases:
        ok, detail, log, dt = run_case(case, args.verbose)
        if ok is None:              # fixture absent -> skipped, not a failure
            n_skip += 1
            tag = "SKIP"
        else:
            all_ok &= ok
            tag = "PASS" if ok else "FAIL"
        print(f"  {tag}  {case['name']:28s} [{dt:6.1f}s]  {detail}")
        if ok is False and args.verbose and log:
            tail = log.splitlines()[-40:]
            print("        --- decoder log tail ---")
            for ln in tail:
                print("        " + ln)

    skip_note = f" ({n_skip} skipped)" if n_skip else ""
    print(f"\n*** REGRESSION: {'ALL PASS' if all_ok else 'FAILURES'}"
          f"{skip_note} ***")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
