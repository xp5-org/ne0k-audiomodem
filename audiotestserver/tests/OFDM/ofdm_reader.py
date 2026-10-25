"""
ofdm_reader.py - Acoustic OFDM Demodulator
============================================
Decodes OFDM frames produced by ofdm_writer.py from a WAV file or live mic.
Drop-in design companion to qam_reader.py.

Algorithm overview
------------------
  1. Schmidl-Cox timing sync
       Delay-and-correlate over lag n_fft/2 exploits the half-period
       property of the SC preamble.  The metric M[d] = P2/R2 ~ 1 at
       frame-start positions; plateau edges locate each frame.

  2. Channel estimation (LS)
       The CE symbol immediately after the SC preamble carries a known
       BPSK PN sequence on all occupied subcarriers.
       H[k] = Y_ce[k] / C[k]   (one-tap flat-fading per subcarrier)

  3. Per-symbol pilot correction (common phase error)
       Pilots inserted every pilot_spacing subcarriers carry a fixed
       known symbol.  The mean phase of Y_pilot / (H * pilot_ref) gives
       the common phase error (CPE) for that symbol, which is removed
       before data detection.

  4. Data detection
       Equalised:  X[k] = Y[k] / (H[k] * e^{j*CPE})
       Hard-decision nearest-constellation-point demodulation.
       PRBS descrambling -> RS decode -> reassemble chunks.

Standalone usage:
    python ofdm_reader.py wav  --file recording.wav --config ofdm_config.json
    python ofdm_reader.py live --config ofdm_config.json [--device N]
                               [--timeout 60] [--threshold 0.35]
                               [--mic-gain-db 0] [--wav-out capture.wav]
                               [--stats-out stats.json]
"""
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import threading
from pathlib import Path

import numpy as np
from scipy.io import wavfile
# RS_BACKEND=c swaps the pure-Python reedsolo for the C core in
# tests/c/ (rs255_c.RSCodec is a drop-in and defaults to reedsolo),
# which is why this imports the shim rather than the package.
from rs255_c import RSCodec

# Re-use constellation tables and PRBS from writer
from ofdm_writer import (
    QAM16_CONST, QAM16_NORM, QAM16_POINTS, QAM16_BITS,
    QPSK_CONST, QPSK_POINTS, QPSK_BITS,
    BITS_PER_MOD, prbs,
    Modulator,          # for class constants (PRBS seeds)
)


class Demodulator:
    def __init__(self, config: dict) -> None:
        self.n_fft              = config["n_fft"]
        self.cp_len             = config["cp_len"]
        self.bin_lo             = config["bin_lo"]
        self.bin_hi             = config["bin_hi"]
        self.n_occupied         = config["n_occupied"]
        self.pilot_spacing      = config["pilot_spacing"]
        self.n_pilots           = config["n_pilots"]
        self.n_data_carriers    = config["n_data_carriers"]
        self.modulation         = config["modulation"]
        self.bits_per_qam       = config["bits_per_qam"]
        self.bits_per_ofdm_sym  = config["bits_per_ofdm_sym"]
        self.sample_rate        = config["sample_rate"]
        self.symbol_samples     = config["symbol_samples"]
        self.ecc_bytes          = config["ecc_bytes"]
        self.chunk_bytes        = config["chunk_bytes"]
        self.seq_bytes          = config["seq_bytes"]
        self.payload_bytes      = config["payload_bytes"]
        self.data_syms_per_pkt  = config["data_syms_per_pkt"]
        self.total_frames       = config.get("total_frames")
        self.PRBS_SEED_DATA     = config.get("PRBS_SEED_DATA", Modulator.PRBS_SEED_DATA)

        ce_re = np.array(config["ce_known_re"])
        ce_im = np.array(config["ce_known_im"])
        self.ce_known = (ce_re + 1j * ce_im).astype(complex)

        self.rsc           = RSCodec(self.ecc_bytes)
        self.pilot_offsets = list(range(0, self.n_occupied, self.pilot_spacing))
        self._pilot_set    = set(self.pilot_offsets)
        self._data_offsets = [i for i in range(self.n_occupied)
                              if i not in self._pilot_set]
        self.PILOT_SYMBOL  = Modulator.PILOT_SYMBOL

        # Bitrate helpers (computed once)
        SYM = self.symbol_samples
        fr  = SYM * (self.data_syms_per_pkt + 2)
        self._gross_bps  = int(self.bits_per_ofdm_sym * self.sample_rate / SYM)
        self._coded_bps  = int(self.payload_bytes   * 8 * self.sample_rate / fr)
        self._net_bps    = int(self.chunk_bytes      * 8 * self.sample_rate / fr)

    # -- main entry points -----------------------------------------------------

    def decode_samples(
        self,
        samples:       np.ndarray,
        threshold:     float = 0.35,
        stats_callback = None,
    ) -> tuple[list[dict], dict[int, bytes]]:
        """
        Decode all OFDM frames found in *samples* (float32, normalised +/-1).

        Returns
        -------
        (frame_stats, chunks)
          frame_stats : list of per-frame stat dicts
          chunks      : {seq_idx: bytes}   -  may contain duplicates resolved
        """
        peaks  = self._sc_peaks(samples, threshold)
        stats  = []
        chunks: dict[int, bytes] = {}

        for d in peaks:
            stat = self._decode_frame(samples, d)
            if stat is None:
                continue
            seq = stat["seq"]
            if seq in chunks:
                stat["duplicate"] = True
                print(f"FRAME_DUP:{json.dumps({'seq': seq})}", flush=True)
            else:
                chunks[seq] = stat.pop("_chunk")
                stat["duplicate"] = False
            stats.append(stat)
            if stats_callback:
                stats_callback(stat)
            print(f"FRAME_STAT:{json.dumps(stat)}", flush=True)

        return stats, chunks

    def decode_wav(
        self,
        wav_path:  str,
        threshold: float = 0.35,
        gain_db:   float = 0.0,
        stats_out: str | None = None,
        stats_callback = None,
    ) -> tuple[list[dict], str | None]:
        """Read a WAV file and decode all frames. Returns (stats, decoded_text)."""
        sr, pcm = wavfile.read(wav_path)
        if sr != self.sample_rate:
            raise ValueError(f"WAV sample rate {sr} != config {self.sample_rate}")
        samples = pcm.astype(np.float32) / 32768.0
        if gain_db:
            samples *= 10 ** (gain_db / 20)

        stats, chunks = self.decode_samples(samples, threshold, stats_callback)

        text = self._reassemble(chunks)

        if stats_out and stats:
            Path(stats_out).write_text(json.dumps(stats, indent=2))

        return stats, text

    def decode_live(
        self,
        device:    int | None = None,
        timeout:   float      = 60.0,
        threshold: float      = 0.35,
        gain_db:   float      = 0.0,
        wav_out:   str | None = None,
        stats_out: str | None = None,
        stats_callback = None,
    ) -> tuple[list[dict], str | None]:
        """Record from mic until all frames decoded or timeout, then decode."""
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError("sounddevice not installed  -  pip install sounddevice")

        buf: list[np.ndarray] = []
        lock   = threading.Lock()
        done   = threading.Event()

        def _cb(indata, frames, time_info, status):
            with lock:
                buf.append(indata[:, 0].copy())

        stream_kw = dict(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=_cb,
        )
        if device is not None:
            stream_kw["device"] = device

        all_stats: list[dict]        = []
        seen:      dict[int, bytes]  = {}
        total                        = self.total_frames or 0

        print(f"[ofdm_reader] recording (timeout={timeout}s) ...", flush=True)
        with sd.InputStream(**stream_kw):
            t0 = time.time()
            while time.time() - t0 < timeout:
                time.sleep(2.0)
                with lock:
                    if not buf:
                        continue
                    samples = np.concatenate(buf)

                if gain_db:
                    samples = samples * (10 ** (gain_db / 20))

                stats, chunks = self.decode_samples(samples, threshold, stats_callback)
                for s in stats:
                    seq = s["seq"]
                    if seq not in seen and not s.get("duplicate"):
                        seen[seq] = chunks.get(seq, b"")
                all_stats = stats  # replaced each iteration (duplicates filtered by seen)

                if total and len(seen) >= total:
                    print(f"[ofdm_reader] all {total} frames received", flush=True)
                    break

        # Final decode of everything accumulated
        with lock:
            if buf:
                samples = np.concatenate(buf)
        if gain_db:
            samples = samples * (10 ** (gain_db / 20))

        if wav_out:
            pcm = (samples * 32767).astype(np.int16)
            wavfile.write(wav_out, self.sample_rate, pcm)
            print(f"[ofdm_reader] WAV saved: {wav_out}", flush=True)

        final_stats, final_chunks = self.decode_samples(samples, threshold)
        seen.update(final_chunks)
        text = self._reassemble(seen)

        if stats_out and final_stats:
            Path(stats_out).write_text(json.dumps(final_stats, indent=2))

        return final_stats, text

    # -- timing sync -----------------------------------------------------------

    def _sc_peaks(self, samples: np.ndarray, threshold: float) -> list[int]:
        """
        Schmidl-Cox timing metric.

        M[d] = P[d]2 / R[d]2
          P[d] = sum_{m=0}^{L-1}  r[d+m] * r[d+m+L]   (L = n_fft//2)
          R[d] = sum_{m=0}^{L-1}  r[d+m+L]2

        Returns list of frame-start sample indices (start of SC plateau).
        """
        r = samples.astype(np.float64)
        N = self.n_fft
        L = N // 2
        n = len(r)

        if n < N + self.cp_len:
            return []

        # Element-wise products for sliding windows
        max_i = n - L
        prod  = r[:max_i] * r[L:n]          # prod[i] = r[i]*r[i+L]
        sq    = r[L:n]    ** 2               # sq[i]   = r[i+L]2

        # Prefix sums for O(1) window queries
        cP = np.empty(len(prod) + 1); cP[0] = 0.0; np.cumsum(prod, out=cP[1:])
        cR = np.empty(len(sq)   + 1); cR[0] = 0.0; np.cumsum(sq,   out=cR[1:])

        max_d = len(cP) - L - 1
        if max_d < 1:
            return []

        d    = np.arange(max_d)
        P    = cP[d + L] - cP[d]
        R    = cR[d + L] - cR[d]
        M    = np.where(R > 1e-12, P * P / (R * R + 1e-24), 0.0)

        # Detect rising edges of the plateau (threshold crossing going up)
        above  = M > threshold
        if not np.any(above):
            return []

        # Rising edge = transition False->True
        edges  = np.where(np.diff(above.astype(np.int8)) > 0)[0] + 1
        if above[0]:
            edges = np.concatenate([[0], edges])

        # min_gap = full frame duration so peaks inside the current frame body
        # are rejected.  Frame = 1 SC + 1 CE + data_syms_per_pkt data symbols.
        # Use (data_syms_per_pkt + 2) * SYM which covers SC + CE + data without
        # the OLA taper overhang (overhang <= cp_len, well within tolerance).
        min_gap = self.symbol_samples * (self.data_syms_per_pkt + 2)

        # The SC metric ramps from 0 up to 1 over a region ~cp_len samples wide
        # BEFORE the plateau (because the CP starts feeding the correlation
        # window early).  Using the threshold crossing as the frame start
        # undershoots by up to cp_len samples.
        #
        # Refinement: from the threshold-crossing candidate, scan forward to
        # the first sample where M reaches >= PLATEAU_FRAC of the local max.
        # That sample is the true plateau entry = true frame start.
        PLATEAU_FRAC  = 0.95
        refine_window = self.cp_len * 2   # max refinement scan distance

        peaks: list[int] = []
        last  = -(min_gap * 2)
        for d_start in edges:
            if d_start - last < min_gap:
                continue
            # Refine: find first d where M reaches 95% of local max
            d_end   = min(d_start + refine_window, len(M))
            region  = M[d_start : d_end]
            if len(region) == 0:
                continue
            r_max = float(np.max(region))
            # find first index where region >= PLATEAU_FRAC * r_max
            hits = np.where(region >= PLATEAU_FRAC * r_max)[0]
            refined = int(d_start + hits[0]) if len(hits) > 0 else d_start
            peaks.append(refined)
            last = refined

        return peaks

    # -- frame decoder ---------------------------------------------------------

    def _decode_frame(
        self, samples: np.ndarray, frame_start: int
    ) -> dict | None:
        """
        Decode one OFDM frame starting at *frame_start*.
        Returns a stat dict (with '_chunk' key) or None on failure.
        """
        SYM = self.symbol_samples
        N   = self.n_fft
        CP  = self.cp_len
        needed = SYM * (2 + self.data_syms_per_pkt)
        if frame_start + needed > len(samples):
            return None

        pos = frame_start + SYM      # skip SC preamble

        # -- channel estimation ------------------------------------------------
        ce_td = samples[pos + CP : pos + CP + N]
        ce_fd = np.fft.rfft(ce_td, n=N)          # rfft for real input

        H = np.zeros(self.n_occupied, dtype=complex)
        for i in range(self.n_occupied):
            b    = self.bin_lo + i
            Y_ce = ce_fd[b] / math.sqrt(N)
            # Avoid divide-by-zero on null CE tones
            if abs(self.ce_known[i]) > 1e-9:
                H[i] = Y_ce / self.ce_known[i]
            else:
                H[i] = 1.0 + 0j

        pos += SYM

        # -- data symbols ------------------------------------------------------
        all_bits: list[np.ndarray] = []
        evm_acc  = 0.0
        cpe_acc  = 0.0

        for _ in range(self.data_syms_per_pkt):
            sym_td = samples[pos + CP : pos + CP + N]
            sym_fd = np.fft.rfft(sym_td, n=N)
            pos   += SYM

            # Common phase error from pilots
            cpe = self._estimate_cpe(sym_fd, H)
            # Residual r = Y/(H*pilot) has angle +cpe, so cancelling it needs
            # H_eff = H*e^{+j*cpe}; eq = Y/H_eff then divides the error out.
            corr = math.cos(cpe) + 1j * math.sin(cpe)  # e^{+j*CPE}

            # Equalise data subcarriers
            eq = np.zeros(self.n_data_carriers, dtype=complex)
            for j, di in enumerate(self._data_offsets):
                b     = self.bin_lo + di
                Y     = sym_fd[b] / math.sqrt(N)
                H_eff = H[di] * corr
                eq[j] = Y / H_eff if abs(H_eff) > 1e-9 else 0.0 + 0j

            bits, ref = self._demodulate(eq)
            evm_val   = self._evm(eq, ref)
            evm_acc  += evm_val
            cpe_acc  += abs(math.degrees(cpe))
            all_bits.append(bits)

        evm_avg   = evm_acc / self.data_syms_per_pkt
        evm_pct   = evm_avg * 100.0
        snr_evm   = -20 * math.log10(evm_avg + 1e-9)

        # -- RS decode ---------------------------------------------------------
        scr_bits = np.concatenate(all_bits)[: self.payload_bytes * 8]
        raw_bits = prbs(len(scr_bits), self.PRBS_SEED_DATA) ^ scr_bits.astype(np.uint8)

        n_bytes  = (len(raw_bits) // 8) * 8
        raw_bytes = np.packbits(raw_bits[:n_bytes]).tobytes()

        try:
            dec, _, errata = self.rsc.decode(bytearray(raw_bytes[: self.payload_bytes]))
            dec       = bytes(dec)
            err_count = len(errata)
        except Exception:
            return None         # uncorrectable  -  skip this candidate

        if len(dec) < self.seq_bytes:
            return None

        seq_idx = int.from_bytes(dec[: self.seq_bytes], "big")
        chunk   = dec[self.seq_bytes :]

        ecc_cap      = self.ecc_bytes // 2
        ecc_headroom = max(0.0, (ecc_cap - err_count) / ecc_cap * 100)

        # Channel power (mean |H|2) as rough SNR proxy
        H_power      = float(np.mean(np.abs(H) ** 2))
        input_rms_db = 20 * math.log10(
            float(np.sqrt(np.mean(samples[frame_start:frame_start+needed] ** 2))) + 1e-9
        )

        return {
            "seq":               seq_idx,
            "frame_start":       frame_start,
            "err_count":         err_count,
            "ecc_capacity":      ecc_cap,
            "ecc_headroom_pct":  round(ecc_headroom, 1),
            "evm_pct":           round(evm_pct, 2),
            "snr_evm_db":        round(snr_evm, 1),
            "avg_cpe_deg":       round(cpe_acc / self.data_syms_per_pkt, 2),
            "H_power_db":        round(10 * math.log10(H_power + 1e-9), 1),
            "input_rms_db":      round(input_rms_db, 1),
            "gross_bps":         self._gross_bps,
            "coded_bps":         self._coded_bps,
            "net_bps":           self._net_bps,
            "_chunk":            chunk,    # popped before external delivery
        }

    # -- equalization helpers --------------------------------------------------

    def _estimate_cpe(self, sym_fd: np.ndarray, H: np.ndarray) -> float:
        """
        Common phase error (radians) estimated from pilot subcarriers.
        Uses mean angle of (Y_pilot / H_pilot / pilot_ref).
        """
        if not self.pilot_offsets:
            return 0.0
        errors = []
        for pi in self.pilot_offsets:
            b     = self.bin_lo + pi
            Y_p   = sym_fd[b] / math.sqrt(self.n_fft)
            H_eff = H[pi]
            if abs(H_eff) < 1e-9:
                continue
            ratio = Y_p / (H_eff * self.PILOT_SYMBOL)
            errors.append(math.atan2(ratio.imag, ratio.real))
        if not errors:
            return 0.0
        # Circular mean
        return float(math.atan2(
            sum(math.sin(e) for e in errors) / len(errors),
            sum(math.cos(e) for e in errors) / len(errors),
        ))

    @staticmethod
    def _evm(rx: np.ndarray, ideal: np.ndarray) -> float:
        err_pwr  = float(np.mean(np.abs(rx - ideal) ** 2))
        ref_pwr  = float(np.mean(np.abs(ideal) ** 2))
        return math.sqrt(err_pwr / (ref_pwr + 1e-20))

    # -- demodulation ----------------------------------------------------------

    def _demodulate(
        self, syms: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hard-decision demodulation. Returns (bits uint8, ideal_ref)."""
        if self.modulation == "bpsk":
            bits = (syms.real < 0).astype(np.uint8)
            ref  = (1.0 - 2.0 * bits).astype(complex)
            return bits, ref
        elif self.modulation == "qpsk":
            d   = np.abs(syms[:, None] - QPSK_POINTS[None, :]) ** 2
            idx = np.argmin(d, axis=1)
            return QPSK_BITS[idx].ravel().astype(np.uint8), QPSK_POINTS[idx]
        else:  # qam16
            d   = np.abs(syms[:, None] - QAM16_POINTS[None, :]) ** 2
            idx = np.argmin(d, axis=1)
            return QAM16_BITS[idx].ravel().astype(np.uint8), QAM16_POINTS[idx]

    # -- reassembly ------------------------------------------------------------

    @staticmethod
    def _reassemble(chunks: dict[int, bytes]) -> str | None:
        if not chunks:
            return None
        ordered = [chunks[k] for k in sorted(chunks)]
        raw     = b"".join(ordered).rstrip(b"\x00")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace")


# -- CLI -----------------------------------------------------------------------

def _load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description="OFDM acoustic decoder")
    sub = ap.add_subparsers(dest="mode", required=True)

    # wav mode
    pw = sub.add_parser("wav", help="Decode a WAV file")
    pw.add_argument("--file",       required=True)
    pw.add_argument("--config",     required=True)
    pw.add_argument("--threshold",  type=float, default=0.35)
    pw.add_argument("--mic-gain-db", type=float, default=0.0)
    pw.add_argument("--stats-out")

    # live mode
    pl = sub.add_parser("live", help="Record from mic and decode")
    pl.add_argument("--config",     required=True)
    pl.add_argument("--device",     type=int)
    pl.add_argument("--timeout",    type=float, default=60.0)
    pl.add_argument("--threshold",  type=float, default=0.35)
    pl.add_argument("--mic-gain-db", type=float, default=0.0)
    pl.add_argument("--wav-out")
    pl.add_argument("--stats-out")

    args = ap.parse_args()
    cfg  = _load_config(args.config)
    dem  = Demodulator(cfg)

    def _on_frame(stat):
        # Print to stdout for parent process to capture
        s = stat.copy()
        s.pop("_chunk", None)
        pass  # already printed inside decode_samples

    if args.mode == "wav":
        stats, text = dem.decode_wav(
            wav_path   = args.file,
            threshold  = args.threshold,
            gain_db    = args.mic_gain_db,
            stats_out  = args.stats_out,
        )
    else:
        stats, text = dem.decode_live(
            device    = args.device,
            timeout   = args.timeout,
            threshold = args.threshold,
            gain_db   = args.mic_gain_db,
            wav_out   = args.wav_out,
            stats_out = args.stats_out,
        )

    n = len(stats)
    ok = sum(1 for s in stats if not s.get("duplicate"))
    total = cfg.get("total_frames", "-")
    print(f"\n[ofdm_reader] decoded {ok}/{total} frames  "
          f"({n} detections incl. dups)", flush=True)
    if text:
        print(f"[ofdm_reader] text: {text[:120]}{'...' if len(text)>120 else ''}",
              flush=True)

    result = {
        "status":         "ok" if ok == total or total == "-" else "partial",
        "frames_decoded": ok,
        "frames_expected": total,
        "text":           text,
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
