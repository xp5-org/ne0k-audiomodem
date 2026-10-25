# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""
tests/analysis.py  –  Post-run visualisation helpers.

Three public functions, all optional (gracefully skip if matplotlib is absent):

    make_qam_constellation_png(frame_stats_path, output_path, config, title=None)
        Three-panel summary for a single QAM decode run:
          · Ideal 16-QAM reference constellation
          · ECC headroom per frame (green / orange / red bars)
          · SNR (EVM) over time

    make_qam_probe_heatmap_png(step_results, output_path, title=None)
        Carrier-freq × symbol-rate heatmap for probe sweeps.
        Two sub-plots side by side:
          · Left:  frames decoded (fraction of expected)
          · Right: net bitrate (bps)

    make_css_spectrograph_png(wav_path, output_path, info=None,
                              result=None, title=None)
        STFT spectrogram of a raw CSS mic capture.
          · Frequency on y-axis (Hz), time on x-axis (s)
          · Power in dB (viridis colormap)
          · Dashed horizontal lines at f_min / f_max from info dict
          · Vertical line at sync acquisition time (from result dict)
          · Decoded frame boundaries if result contains frame timestamps

All functions return True on success, False on failure (e.g. missing deps or
empty data), and log a one-line message either way.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_mpl():
    """Import matplotlib in non-interactive mode.  Raises ImportError if absent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    return plt, ticker


def _log(msg: str) -> None:
    print(msg, flush=True)


def _read_wav_float32(path: "str | Path") -> tuple[np.ndarray, int]:
    """Read a WAV file, return (float32 mono array, sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        fs        = wf.getframerate()
        n_ch      = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw       = wf.readframes(wf.getnframes())
    dtype   = np.int16 if sampwidth == 2 else np.int32
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch)[:, 0]
    samples /= float(np.iinfo(dtype).max)
    return samples.astype(np.float32), fs


# ---------------------------------------------------------------------------
# QAM constellation + ECC headroom
# ---------------------------------------------------------------------------

_IDEAL_16QAM = [
    -3-3j, -3-1j, -3+1j, -3+3j,
    -1-3j, -1-1j, -1+1j, -1+3j,
     1-3j,  1-1j,  1+1j,  1+3j,
     3-3j,  3-1j,  3+1j,  3+3j,
]


def make_qam_constellation_png(
    frame_stats_path: "str | Path",
    output_path: "str | Path",
    config: Optional[dict] = None,
    title: Optional[str] = None,
) -> bool:
    """
    Three-panel QAM decode summary.

    Panel 1 — Ideal 16-QAM reference constellation.
    Panel 2 — ECC headroom % per frame (green ≥50 %, orange >0 %, red = 0 %).
               RS error count annotated on red / orange bars.
    Panel 3 — SNR (EVM-derived) and EVM % over frame sequence.

    Parameters
    ----------
    frame_stats_path : path to qam_frame_stats.json (written by qam_reader).
    output_path      : destination PNG path.
    config           : qam_config.json dict (used for title metadata).
    title            : optional override title.
    """
    try:
        plt, ticker = _require_mpl()
    except ImportError:
        _log("  analysis: matplotlib not installed — skipping QAM constellation PNG")
        return False

    try:
        stats  = json.loads(Path(frame_stats_path).read_text())
        frames = stats.get("frames", [])
        if not frames:
            _log("  analysis: no frame stats — skipping QAM constellation PNG")
            return False

        n      = len(frames)
        seqs   = [f["seq"]              for f in frames]
        hdroom = [f["ecc_headroom_pct"] for f in frames]
        errs   = [f["err_count"]        for f in frames]
        snrs   = [f.get("snr_evm_db",  0)   for f in frames]
        evms   = [f.get("evm_pct",     100) for f in frames]

        net_bps    = stats.get("net_bps",    frames[0].get("net_bps",    "?"))
        coded_bps  = stats.get("coded_bps",  frames[0].get("coded_bps",  "?"))
        gross_bps  = stats.get("gross_bps",  frames[0].get("gross_bps",  "?"))
        n_total    = (config or {}).get("total_frames", "?")

        if title is None:
            cf = (config or {}).get("carrier_freq", "?")
            sr = (config or {}).get("symbol_rate",  "?")
            title = (f"QAM Decode — carrier {cf} Hz  symbol_rate {sr} baud\n"
                     f"{n}/{n_total} frames  •  "
                     f"net {net_bps} / coded {coded_bps} / gross {gross_bps} bps")

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle(title, fontsize=10)

        # ── Panel 1: Ideal constellation ──────────────────────────────────
        ax = axes[0]
        ix = [p.real for p in _IDEAL_16QAM]
        iy = [p.imag for p in _IDEAL_16QAM]
        ax.scatter(ix, iy, s=80, color="steelblue", zorder=3)
        for p in _IDEAL_16QAM:
            ax.text(p.real + 0.1, p.imag + 0.1,
                    f"({int(p.real)},{int(p.imag)})",
                    fontsize=5, color="gray", va="bottom")
        ax.axhline(0, color="lightgray", lw=0.8, zorder=0)
        ax.axvline(0, color="lightgray", lw=0.8, zorder=0)
        for v in (-2, 0, 2):
            ax.axhline(v, color="#e8e8e8", lw=0.4, zorder=0)
            ax.axvline(v, color="#e8e8e8", lw=0.4, zorder=0)
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect("equal")
        ax.set_title("Ideal 16-QAM", fontsize=9)
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.grid(False)

        # ── Panel 2: ECC headroom bars ────────────────────────────────────
        ax2 = axes[1]
        colors = ["#2ecc71" if h >= 50 else "#e67e22" if h > 0 else "#e74c3c"
                  for h in hdroom]
        bars = ax2.bar(seqs, hdroom, color=colors, edgecolor="white",
                       linewidth=0.4, width=0.8)
        ax2.axhline(50,  color="#2ecc71", lw=1.0, ls="--", alpha=0.6,
                    label="50 % headroom")
        ax2.axhline(100, color="#888",    lw=0.4, ls=":", alpha=0.5)
        ax2.set_ylim(0, 115)
        ax2.set_xlabel("Frame sequence")
        ax2.set_ylabel("ECC headroom (%)")
        ax2.set_title("ECC Headroom per Frame", fontsize=9)
        ax2.legend(fontsize=7, loc="upper right")
        for bar, err, h in zip(bars, errs, hdroom):
            if err > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         min(h + 2, 108),
                         f"+{err}", ha="center", va="bottom",
                         fontsize=6, color="#c0392b", fontweight="bold")

        # ── Panel 3: SNR and EVM over time ────────────────────────────────
        ax3 = axes[2]
        color_snr = "#2980b9"
        color_evm = "#e74c3c"
        ln1, = ax3.plot(seqs, snrs, color=color_snr, lw=1.5, marker="o",
                        markersize=3, label="SNR (dB)")
        ax3.set_xlabel("Frame sequence")
        ax3.set_ylabel("SNR (EVM-derived, dB)", color=color_snr)
        ax3.tick_params(axis="y", labelcolor=color_snr)
        ax3.set_title("SNR & EVM over Time", fontsize=9)

        ax3b = ax3.twinx()
        ln2, = ax3b.plot(seqs, evms, color=color_evm, lw=1.2, ls="--",
                         marker="s", markersize=3, label="EVM (%)")
        ax3b.set_ylabel("EVM (%)", color=color_evm)
        ax3b.tick_params(axis="y", labelcolor=color_evm)
        ax3b.set_ylim(bottom=0)

        lines  = [ln1, ln2]
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, fontsize=7, loc="upper right")
        ax3.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        _log(f"  analysis: QAM constellation PNG -> {Path(output_path).name}")
        return True

    except Exception as exc:
        import traceback
        _log(f"  analysis: QAM constellation PNG failed: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# QAM probe heatmap
# ---------------------------------------------------------------------------

def make_qam_probe_heatmap_png(
    step_results: list,
    output_path: "str | Path",
    title: Optional[str] = None,
) -> bool:
    """
    Carrier-freq × symbol-rate heatmap for a completed qam_probe run.

    Left panel  — fraction of frames decoded (0–1), white = none, green = all.
    Right panel — net bitrate (bps), white = none, blue scale.

    If only one symbol-rate was probed, a bar-chart fallback is used instead.
    """
    try:
        plt, ticker = _require_mpl()
        import matplotlib.colors as mcolors
    except ImportError:
        _log("  analysis: matplotlib not installed — skipping QAM probe heatmap")
        return False

    try:
        if not step_results:
            _log("  analysis: no probe step results — skipping heatmap")
            return False

        carriers = sorted({r["carrier_freq"] for r in step_results})
        symrates = sorted({r["symbol_rate"]  for r in step_results})

        def _grid(key, default=0.0):
            m = np.full((len(symrates), len(carriers)), np.nan)
            for r in step_results:
                ci = carriers.index(r["carrier_freq"])
                si = symrates.index(r["symbol_rate"])
                exp = r.get("frames_expected", 1) or 1
                if key == "frac":
                    m[si, ci] = r["frames_decoded"] / exp
                elif key == "bps":
                    m[si, ci] = r.get("net_bps", 0) if r["frames_decoded"] > 0 else 0
                elif key == "snr":
                    m[si, ci] = r.get("avg_snr_evm_db", 0) if r["frames_decoded"] > 0 else 0
            return m

        if title is None:
            title = (f"QAM Probe Sweep — "
                     f"{len(carriers)} carriers × {len(symrates)} symbol rates  "
                     f"({len(step_results)} steps)")

        # Bar-chart fallback when there is only one symbol rate
        use_bars = (len(symrates) == 1)

        if use_bars:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            fig.suptitle(title, fontsize=10)

            frac_row = _grid("frac")[0]
            bps_row  = _grid("bps")[0]
            colors   = ["#2ecc71" if f >= 1.0 else "#e67e22" if f > 0 else "#e74c3c"
                        for f in frac_row]

            ax = axes[0]
            ax.bar(range(len(carriers)), frac_row * 100, color=colors,
                   edgecolor="white", lw=0.4)
            ax.set_xticks(range(len(carriers)))
            ax.set_xticklabels([f"{c}" for c in carriers], rotation=45, ha="right",
                                fontsize=8)
            ax.set_ylabel("Frames decoded (%)")
            ax.set_title(f"Decode success @ {symrates[0]} baud", fontsize=9)
            ax.set_ylim(0, 115)

            ax2 = axes[1]
            ax2.bar(range(len(carriers)), bps_row, color="#3498db",
                    edgecolor="white", lw=0.4)
            ax2.set_xticks(range(len(carriers)))
            ax2.set_xticklabels([f"{c}" for c in carriers], rotation=45, ha="right",
                                 fontsize=8)
            ax2.set_ylabel("Net bitrate (bps)")
            ax2.set_title(f"Net bitrate @ {symrates[0]} baud", fontsize=9)
            for i, bps in enumerate(bps_row):
                if bps > 0:
                    ax2.text(i, bps + max(bps_row) * 0.01, f"{int(bps)}",
                             ha="center", va="bottom", fontsize=7)

        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(symrates) * 0.7 + 2)))
            fig.suptitle(title, fontsize=10)

            frac_grid = _grid("frac")
            bps_grid  = _grid("bps")

            cmap_frac = mcolors.LinearSegmentedColormap.from_list(
                "frac", ["#e74c3c", "#f39c12", "#2ecc71"])
            cmap_bps  = plt.cm.Blues

            def _heatmap(ax, data, cmap, vmin, vmax, fmt, clabel):
                im = ax.imshow(data, aspect="auto", cmap=cmap,
                               vmin=vmin, vmax=vmax, origin="lower")
                ax.set_xticks(range(len(carriers)))
                ax.set_xticklabels([f"{c//1000:.1f}k" if c >= 1000 else str(c)
                                    for c in carriers], rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(len(symrates)))
                ax.set_yticklabels([str(s) for s in symrates], fontsize=8)
                ax.set_xlabel("Carrier frequency (Hz)")
                ax.set_ylabel("Symbol rate (baud)")
                plt.colorbar(im, ax=ax, label=clabel, fraction=0.046, pad=0.04)
                for si in range(len(symrates)):
                    for ci in range(len(carriers)):
                        v = data[si, ci]
                        if not np.isnan(v):
                            txt = fmt(v)
                            fc  = "white" if v > (vmax - vmin) * 0.6 + vmin else "black"
                            ax.text(ci, si, txt, ha="center", va="center",
                                    fontsize=6.5, color=fc)

            _heatmap(axes[0], frac_grid, cmap_frac, 0, 1,
                     lambda v: f"{v*100:.0f}%",
                     "Frames decoded (%)")
            axes[0].set_title("Decode success rate", fontsize=9)

            max_bps = float(np.nanmax(bps_grid)) if not np.all(np.isnan(bps_grid)) else 1
            _heatmap(axes[1], bps_grid, cmap_bps, 0, max_bps,
                     lambda v: f"{int(v)}" if v > 0 else "—",
                     "Net bitrate (bps)")
            axes[1].set_title("Net bitrate (bps)", fontsize=9)

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        _log(f"  analysis: QAM probe heatmap PNG -> {Path(output_path).name}")
        return True

    except Exception as exc:
        import traceback
        _log(f"  analysis: QAM probe heatmap PNG failed: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# CSS spectrograph
# ---------------------------------------------------------------------------

def make_css_spectrograph_png(
    wav_path: "str | Path",
    output_path: "str | Path",
    info: Optional[dict] = None,
    result: Optional[dict] = None,
    title: Optional[str] = None,
) -> bool:
    """
    STFT spectrogram of a CSS mic capture WAV.

    Layout
    ------
    Single axes — time (s) on x, frequency (Hz) on y, power (dBFS) as
    viridis heatmap.

    Overlays (when data is available)
    ----------------------------------
    · Dashed horizontal lines at f_min / f_max (from info dict or css_info.json).
    · A light-blue band spanning [f_min, f_max] to highlight the chirp band.
    · Vertical dashed line at sync-acquisition time (result["diagnostics"]
      ["sync_at_symbol"] × symbol_dur).
    · Frame boundary tick-marks along the top x-axis edge when result contains
      per-frame data.

    Parameters
    ----------
    wav_path    : path to the raw mic capture WAV (float32 or int16, mono or
                  stereo — first channel used).
    output_path : destination PNG path.
    info        : css_info.json dict (provides f_min, f_max, symbol_dur, fs).
    result      : css_decode_result.json dict (provides sync timing, frames).
    title       : optional override title string.
    """
    try:
        plt, ticker = _require_mpl()
        from scipy.signal import spectrogram as _spectrogram
    except ImportError:
        _log("  analysis: matplotlib / scipy not installed — skipping CSS spectrograph")
        return False

    try:
        samples, fs = _read_wav_float32(wav_path)
    except Exception as exc:
        _log(f"  analysis: cannot read WAV {wav_path}: {exc}")
        return False

    if len(samples) < 512:
        _log("  analysis: WAV too short for spectrograph")
        return False

    try:
        duration = len(samples) / fs

        # ── STFT parameters ──────────────────────────────────────────────
        # Target ~50–100 ms time resolution, ~20 Hz frequency resolution.
        nperseg = min(len(samples), max(512, int(fs * 0.05)))
        noverlap = nperseg * 3 // 4

        f_arr, t_arr, Sxx = _spectrogram(
            samples.astype(np.float64),
            fs=fs, nperseg=nperseg, noverlap=noverlap,
            window="hann", scaling="density",
        )

        # Convert to dBFS (reference = full-scale sine RMS = 1/√2)
        Sxx_db = 10 * np.log10(Sxx + 1e-12) - 10 * np.log10(0.5)

        # ── Extract overlay parameters from info / result ─────────────────
        f_min      = float((info or {}).get("f_min",       200))
        f_max      = float((info or {}).get("f_max",      8000))
        symbol_dur = float((info or {}).get("symbol_dur",  0.25))
        gap_frac   = float((info or {}).get("symbol_gap_frac", 0.0))
        # Symbols are spaced by the full period (chirp + inter-symbol gap),
        # not the bare chirp duration.
        symbol_period = symbol_dur * (1.0 + gap_frac)
        diag       = (result or {}).get("diagnostics",       {})
        frames     = (result or {}).get("frames",             [])

        sync_at_sym = diag.get("sync_at_symbol")
        sync_time   = (sync_at_sym * symbol_period) if sync_at_sym is not None else None

        # ── Dynamic range: clip to [median - 3 dB, p99] ──────────────────
        valid = Sxx_db[np.isfinite(Sxx_db)]
        vmin  = float(np.percentile(valid, 10))
        vmax  = float(np.percentile(valid, 99))
        vmin  = max(vmin, vmax - 60)   # cap dynamic range at 60 dB

        # Limit display to a useful frequency band
        f_display_max = min(float(f_arr[-1]), max(f_max * 2, 12000))
        f_mask        = f_arr <= f_display_max

        if title is None:
            n_frames  = (result or {}).get("frames_decoded", "?")
            status    = (result or {}).get("status", "")
            title = (f"CSS Spectrograph — {Path(wav_path).name}\n"
                     f"{duration:.1f}s @ {fs} Hz  •  "
                     f"f_min={f_min:.0f} Hz  f_max={f_max:.0f} Hz  •  "
                     f"frames={n_frames}  status={status}")

        # ── Plot ──────────────────────────────────────────────────────────
        # Width scales with capture length so individual chirps stay
        # resolvable; both DPI and width are capped to keep the PNG a
        # viewable size (~40 px/audio-second at dpi=100, max 14000 px wide).
        _dpi  = 100
        _w_px = min(14000, max(1300, int(duration * 40)))
        fig, ax = plt.subplots(figsize=(_w_px / _dpi, 6.0))
        fig.suptitle(title, fontsize=9)

        im = ax.pcolormesh(
            t_arr, f_arr[f_mask], Sxx_db[f_mask, :],
            shading="gouraud", cmap="viridis",
            vmin=vmin, vmax=vmax,
        )
        cbar = plt.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label("Power (dBFS)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        # Chirp band highlight
        ax.axhspan(f_min, f_max, color="white", alpha=0.06, zorder=2)
        ax.axhline(f_min, color="cyan", lw=1.2, ls="--", alpha=0.8,
                   label=f"f_min = {f_min:.0f} Hz", zorder=3)
        ax.axhline(f_max, color="lime",  lw=1.2, ls="--", alpha=0.8,
                   label=f"f_max = {f_max:.0f} Hz", zorder=3)

        # Sync acquisition marker
        if sync_time is not None:
            ax.axvline(sync_time, color="orange", lw=1.5, ls="-.",
                       alpha=0.9, label=f"sync @ {sync_time:.2f}s", zorder=4)

        # Frame boundary ticks. Frames repeat every (training + preamble +
        # payload) symbols; spaced by that symbol count × the symbol period.
        if frames and sync_time is not None:
            _frame_syms = (int((info or {}).get("training_symbols", 0))
                           + len((info or {}).get("preamble", []) or [])
                           + int((info or {}).get("frame_symbols", 0)))
            for frame_info in frames:
                seq = frame_info.get("seq", 0)
                frame_t = sync_time + seq * max(1, _frame_syms) * symbol_period
                ax.axvline(frame_t, color="yellow", lw=0.6, alpha=0.4,
                           ymax=0.05, zorder=4)

        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", fontsize=9)
        ax.set_xlim(0, duration)
        ax.set_ylim(0, f_display_max)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v)))
        )
        ax.tick_params(labelsize=8)

        legend = ax.legend(fontsize=7, loc="upper right",
                           framealpha=0.6, edgecolor="gray")
        for text in legend.get_texts():
            text.set_color("white")
        legend.get_frame().set_facecolor("#222222")

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=_dpi, bbox_inches="tight")
        plt.close(fig)
        _log(f"  analysis: CSS spectrograph PNG -> {Path(output_path).name}")
        return True

    except Exception as exc:
        import traceback
        _log(f"  analysis: CSS spectrograph PNG failed: {exc}")
        traceback.print_exc()
        return False
