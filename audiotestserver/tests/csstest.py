"""
css_test.py  -  Offline test harness. No sounddevice required.

Reads exactly sps samples per symbol. No hop/ssls complexity.

Usage:
    python css_test.py                                          # self-test
    python css_test.py --wav capture.wav --info css_info.json
    python css_test.py --wav capture.wav --info css_info.json --verbose

Columns:
    terr    : found_peak - expected_peak  (timing drives this to 0)
    n_read  : samples read for this symbol (sps ± timing correction)
    peak_est: where MF peak was found in ring buffer
    score   : syndet soft score
"""

import argparse
import json
import sys
import wave
import zlib
from pathlib import Path

import numpy as np

try:
    from css_decoder import (
        generate_chirp, ChirpCorrelator, PeakAGC,
        SoftPreambleDetector, get_bytes_from_bits, check_crc,
        _make_rs_codec, rs_decode_frame, deinterleave_bits,
        viterbi_k7, diff_decode_bits, if_slope_confidence,
        BARKER_13, CRC_BYTES, _CONV_TAIL,
    )
except ImportError as e:
    sys.exit(f"ERROR: cannot import css_decoder.py -- {e}")

try:
    from css_encoder import encode as css_encode
except ImportError:
    css_encode = None


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        fs        = wf.getframerate()
        n_ch      = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw       = wf.readframes(wf.getnframes())
    dtype   = np.int16 if sampwidth == 2 else np.int32
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch)[:, 0]
    samples /= float(np.iinfo(dtype).max)
    return samples, fs


def generate_test_wav(cfg, out_path="css_test_generated.wav"):
    if css_encode is None:
        sys.exit("ERROR: css_encoder.py not found")
    c = dict(cfg)
    c["output"]   = out_path
    c["info_out"] = "css_test_info.json"
    result = css_encode(c)
    if result["status"] != "ok":
        sys.exit(f"ERROR: encoder failed: {result}")
    return out_path, "css_test_info.json"


def print_mf_envelope(ring, correlator, centre, label=""):
    up_env, dn_env = correlator.correlate(ring)
    sps = correlator.sps
    lo  = max(0, centre - sps)
    hi  = min(len(up_env), centre + sps)
    pu  = int(np.argmax(up_env[lo:hi])) + lo
    pd  = int(np.argmax(dn_env[lo:hi])) + lo
    print(f"\n  ={'='*64}")
    print(f"  MF{(' -- ' + label) if label else ''}")
    print(f"  centre={centre}  window=[{lo},{hi}]")
    print(f"  up @ {pu:6d}  offset={pu-centre:+d}  amp={up_env[pu]:.4f}")
    print(f"  dn @ {pd:6d}  offset={pd-centre:+d}  amp={dn_env[pd]:.4f}")
    stride = max(1, (hi-lo)//64)
    bar    = up_env[lo:hi][::stride][:64]
    sc     = max(bar.max(), 1e-9)
    ch     = "".join("█" if v>sc*.6 else ("▄" if v>sc*.2 else " ") for v in bar)
    cur    = min(63, (centre-lo)//stride)
    cl     = list(ch); cl[cur] = "│"
    print(f"  up-MF: {''.join(cl)}")
    print(f"  ={'='*64}\n")


def print_header():
    print(f"\n{'sym':>5} {'bit':>3} {'conf':>6} {'snr':>7} {'terr':>7} "
          f"{'n_read':>7} {'peak_est':>9} {'score':>6}  event")
    print("-" * 82)


def sym_row(sym, bit, conf, snr, terr, n_read, peak_est, score, event=""):
    print(f"{sym:5d} {bit:3d} {conf:+6.3f} {snr:7.1f}dB "
          f"{terr:+7d} {n_read:7d} {peak_est:9d} {score:6.3f}  {event}")


def offline_decode(samples, fs, info_path, cfg, verbose=False):
    conf_j = json.loads(Path(info_path).read_text())
    print(f"\n  css_info.json:")
    for k, v in conf_j.items():
        print(f"    {k}: {v}")
    print()

    dur           = float(cfg.get("symbol_dur",     conf_j["symbol_dur"]))
    f_min         = float(cfg.get("f_min",           conf_j["f_min"]))
    f_max         = float(cfg.get("f_max",           conf_j["f_max"]))
    full_preamble = list( cfg.get("preamble",        conf_j.get("preamble", BARKER_13)))
    sync_word     = full_preamble   # always use full preamble; ignore stale sync_word in JSON
    chunk_size    = int(  cfg.get("chunk_size",      conf_j["chunk_size"]))
    crc_bytes_n   = int(  conf_j.get("crc_bytes",    CRC_BYTES))
    rs_t          = int(  conf_j.get("rs_t",         0))
    interleave_rows = int(conf_j.get("interleave_rows", 0))
    sync_thr      = float(cfg.get("sync_threshold",  0.65))
    alpha         = float(cfg.get("timing_loop_gain",   0.10))
    search_frac   = float(cfg.get("timing_search_frac", 0.25))

    rs_parity = rs_t * 2
    rs_codec  = _make_rs_codec(rs_t)

    sps      = int(fs * dur)
    buf_size = sps * 3

    base_bits     = int(conf_j.get("base_bits",
                                   (1 + chunk_size + crc_bytes_n + rs_parity) * 8))
    frame_symbols = int(conf_j.get("frame_symbols", base_bits))
    preamble_per_frame = bool(conf_j.get("preamble_per_frame", False))
    diff_enc  = bool(conf_j.get("diff_encode", False))
    conv_code = conf_j.get("conv_code", None)
    pll_weight = float(cfg.get("pll_weight", 0.30))

    up_ref   = generate_chirp(f_min, f_max, dur, fs)
    down_ref = generate_chirp(f_max, f_min, dur, fs)

    correlator = ChirpCorrelator(up_ref, down_ref, sps,
                                 search_frac=search_frac, buf_size=buf_size)
    agc        = PeakAGC()
    syndet     = SoftPreambleDetector(sync_word, threshold=sync_thr)

    expected_peak = buf_size - 1
    peak_est      = expected_peak
    timing_frac   = 0.0

    print(f"{'='*82}")
    print(f"OFFLINE DECODE")
    print(f"  WAV:    {len(samples)} samples @ {fs}Hz  ({len(samples)/fs:.2f}s)")
    print(f"  sps={sps}  buf_size={buf_size}  expected_peak={expected_peak}")
    print(f"  search_range=±{correlator.search_range}  base_bits={base_bits}  frame_symbols={frame_symbols}")
    print(f"  sync_threshold={sync_thr}  timing_alpha={alpha}  interleave_rows={interleave_rows}")
    print(f"  sync_word ({len(sync_word)}): {sync_word}")
    print(f"{'='*82}")

    ring           = np.zeros(buf_size, dtype=np.float32)
    bit_stream     = []
    sync_found     = False
    timing_settled = False
    frames_decoded = 0
    frames_errored = 0
    consec_errors  = 0
    sym_count      = 0
    read_pos       = 0
    n_read         = sps
    mf_dumped      = False

    print_header()

    while read_pos < len(samples):
        # Read exactly n_read samples (timing-adjusted)
        chunk = samples[read_pos : read_pos + n_read]
        actual = len(chunk)
        if actual == 0:
            break
        read_pos += actual
        if actual < n_read:
            chunk = np.pad(chunk.astype(np.float32), (0, n_read - actual))

        agc.track_rms(chunk)
        ring = np.roll(ring, -n_read)
        ring[-n_read:] = agc.apply(chunk)
        sym_count += 1

        bit, confidence, found_peak, snr_db, winner_pk = correlator.symbol_decision(ring, peak_est)
        peak_est   = found_peak
        timing_err = found_peak - expected_peak

        # AGC
        up_env, dn_env = correlator.correlate(ring)
        sr = correlator.search_range
        lo = max(0,            peak_est - sr)
        hi = min(len(up_env), peak_est + sr)
        agc.update_from_peak(float(max(np.max(up_env[lo:hi]), np.max(dn_env[lo:hi]))))

        if verbose and not mf_dumped:
            print_mf_envelope(ring, correlator, peak_est, f"sym={sym_count}")
            mf_dumped = True

        # Timing correction for NEXT read
        timing_frac += alpha * timing_err
        timing_frac  = float(np.clip(timing_frac, -sps//4, sps//4))
        int_corr     = int(timing_frac)
        timing_frac -= int_corr
        next_n_read  = max(sps // 2, sps + int_corr)

        score = 0.0
        event = ""

        if not sync_found:
            tim_flush_thr = sps // 8
            if abs(timing_err) <= tim_flush_thr:
                timing_settled = True
            if not timing_settled:
                syndet.reset()
            acquired, score = syndet.push(confidence)
            if acquired:
                event = f"*** SYNC_ACQUIRED score={score:.3f} peak_est={peak_est}"
            elif score > 0.3:
                event = f"CANDIDATE score={score:.3f}"

            sym_row(sym_count, bit, confidence, snr_db, timing_err,
                    n_read, peak_est, score, event)

            if acquired:
                if verbose:
                    print_mf_envelope(ring, correlator, peak_est,
                                      f"AT SYNC sym={sym_count}")
                sync_found = True
                bit_stream = []
                print(f"\n  >> SYNC sym={sym_count}  score={score:.3f}"
                      f"  snr={snr_db:.1f}dB  terr={timing_err:+d}\n")
                print_header()

        else:
            # PLL blend (same logic as css_decoder.py)
            final_conf = confidence
            if pll_weight > 0:
                seg_start = max(0, peak_est - sps + 1)
                seg_end   = min(len(ring), peak_est + 1)
                pll_conf, pll_r2 = if_slope_confidence(
                    ring[seg_start:seg_end], f_min, f_max, sps, fs)
                if pll_r2 > 0.10:
                    w = pll_weight * pll_r2
                    final_conf = (1 - w) * confidence + w * pll_conf
            final_bit = 1 if final_conf > 0 else 0
            bit_stream.append((final_bit, final_conf))
            event = ""

            if len(bit_stream) >= frame_symbols:
                frame_items = bit_stream[:frame_symbols]
                bit_stream  = bit_stream[frame_symbols:]

                frame_bits  = [b for b, _ in frame_items]
                frame_confs = [c for _, c in frame_items]

                # Step 1: de-interleave coded bits+confs together
                if interleave_rows > 1:
                    n_coded = len(frame_bits)
                    cols    = n_coded // interleave_rows
                    idxs    = (np.arange(n_coded)
                                 .reshape(cols, interleave_rows)
                                 .T.flatten().tolist())
                    frame_bits  = [frame_bits[i]  for i in idxs]
                    frame_confs = [frame_confs[i] for i in idxs]

                # Step 2: Viterbi
                if conv_code == "k7r12":
                    soft_pairs = [(frame_confs[i], frame_confs[i + 1])
                                  for i in range(0, len(frame_confs) - 1, 2)]
                    frame_bits  = viterbi_k7(soft_pairs)
                    frame_confs = [1.0] * len(frame_bits)

                # Step 3: differential decode
                if diff_enc:
                    frame_bits = diff_decode_bits(frame_bits, ref_bit=sync_word[-1])

                raw_bytes  = get_bytes_from_bits(frame_bits[:base_bits])

                rs_errors = 0
                if rs_codec is not None:
                    try:
                        raw_bytes, rs_errors = rs_decode_frame(raw_bytes, rs_codec)
                    except Exception as rs_exc:
                        event = f"RS_FAIL {rs_exc}"
                        frames_errored += 1; consec_errors += 1
                        sym_row(sym_count, bit, confidence, snr_db, timing_err,
                                n_read, peak_est, 0.0, event)
                        if consec_errors >= 3:
                            print(f"\n  >> RESYNC\n")
                            sync_found = False; bit_stream = []; consec_errors = 0
                            timing_settled = False
                            syndet.reset(); print_header()
                        n_read = next_n_read
                        continue

                ok, seq_num, data = check_crc(raw_bytes, chunk_size)
                if ok:
                    try:
                        msg   = data.decode("utf-8", errors="strict")
                        event = f"FRAME seq={seq_num} rs={rs_errors} text={msg!r}"
                        frames_decoded += 1; consec_errors = 0
                        if preamble_per_frame:
                            sync_found = False; bit_stream = []; consec_errors = 0
                            timing_settled = False
                            syndet.reset()
                            print(f"\n  >> RE-ARM for next preamble\n")
                            print_header()
                    except UnicodeDecodeError:
                        event = f"FRAME_UTF8_ERR seq={seq_num}"
                        frames_errored += 1; consec_errors += 1
                else:
                    need = 1 + chunk_size + crc_bytes_n
                    if verbose and len(raw_bytes) >= need:
                        seq_b    = raw_bytes[0]
                        data_b   = raw_bytes[1:1+chunk_size]
                        recv_crc = raw_bytes[1+chunk_size:need]
                        calc_crc = (zlib.crc32(bytes([seq_b]) + data_b)
                                    & 0xFFFFFFFF).to_bytes(4, "big")
                        print(f"\n  CRC_FAIL detail sym={sym_count}:")
                        print(f"    seq={seq_b}  data={data_b.hex()}")
                        print(f"    recv={recv_crc.hex()}  calc={calc_crc.hex()}")
                        print(f"    bits[0:16]={frame_bits[:16]}\n")
                    event = f"CRC_FAIL raw={raw_bytes[:6].hex()}"
                    frames_errored += 1; consec_errors += 1

                    if consec_errors >= 3:
                        print(f"\n  >> RESYNC after {consec_errors} failures\n")
                        sync_found = False; bit_stream = []; consec_errors = 0
                        timing_settled = False
                        syndet.reset(); print_header()

            sym_row(sym_count, bit, confidence, snr_db, timing_err,
                    n_read, peak_est, 0.0, event)

        n_read = next_n_read

    print(f"\n{'='*82}")
    print(f"DONE  frames_decoded={frames_decoded}  frames_errored={frames_errored}")
    print(f"{'='*82}\n")
    return frames_decoded, frames_errored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav",     default=None)
    ap.add_argument("--config",  default=None)
    ap.add_argument("--info",    default=None)
    ap.add_argument("--message", default="Hello world from CSS test harness!")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = {}
    if args.config and Path(args.config).exists():
        cfg = json.loads(Path(args.config).read_text())

    wav_path  = args.wav
    info_path = args.info

    if wav_path is None:
        print("No --wav specified. Generating test WAV...")
        enc_cfg = {
            "message":         args.message,
            "fs":              cfg.get("fs",             44100),
            "f_min":           cfg.get("f_min",           400),
            "f_max":           cfg.get("f_max",          6000),
            "symbol_dur":      cfg.get("symbol_dur",      0.25),
            "chunk_size":      cfg.get("chunk_size",      8),
            "amplitude":       cfg.get("amplitude",       0.8),
            "rs_t":            cfg.get("rs_t",            0),
            "interleave_rows": cfg.get("interleave_rows", 0),
        }
        wav_path, info_path = generate_test_wav(enc_cfg)
        print(f"Generated: {wav_path}  info: {info_path}")

    if info_path is None:
        for c in ["css_info.json", "css_test_info.json"]:
            if Path(c).exists():
                info_path = c; break
    if info_path is None:
        sys.exit("ERROR: pass --info path/to/css_info.json")
    if not Path(wav_path).exists():
        sys.exit(f"ERROR: WAV not found: {wav_path}")
    if not Path(info_path).exists():
        sys.exit(f"ERROR: info not found: {info_path}")

    print(f"WAV:  {wav_path}")
    print(f"Info: {info_path}")
    samples, fs = read_wav(wav_path)
    print(f"Read {len(samples)} samples @ {fs}Hz  ({len(samples)/fs:.2f}s)")

    offline_decode(samples, fs, info_path, cfg, verbose=args.verbose)


if __name__ == "__main__":
    main()