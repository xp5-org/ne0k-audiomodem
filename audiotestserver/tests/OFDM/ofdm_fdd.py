"""ofdm_fdd.py -- full-duplex (FDD) OFDM core, shared by the standalone
feasibility test and the meshmodem phy.

Everything else in this repo is half duplex: one node transmits, the other
listens, turn by turn. Full duplex splits the band by direction around a
guard gap and has both ends transmitting at the same time, which is only
meaningful between exactly two endpoints (the band has two halves) -- the
meshmodem phy that uses this refuses to start with more than two participants.

All functions take explicit arguments. PARAMS.get() calls stay in the node
scripts, since the web UI builds its form by scraping those literals out of
the script it runs; a knob that only exists in a library module never shows
in the UI.

Defaults, measured over the air:
  * dl_side='high' -- duplex damage flows upward in frequency only, so the
    direction on the high band is the one that gets hurt. Put the wide
    direction there; it has carriers to spare, the narrow one does not.
  * guard_hz=1200 -- 300 Hz leaves the far side's skirts inside your bins.
  * Keep the whole span inside one octave (freq_hi < 2*freq_lo). A guard band
    cannot stop a harmonic: with 400-2450 under 3950-6000 the lower band's
    2nd and 3rd harmonics blanket the upper one and its allocation collapses
    to zero carriers at any guard width. `plan_duplex_bands` reports this
    rather than producing a plan that cannot work.
  * The RX filter is not optional -- see rx_filter().
"""
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
from __future__ import annotations

import hashlib
import math
import random

import numpy as np
from scipy.signal import butter, ellip, ellipord, sosfiltfilt

from ofdm_modem import build_config, Modem

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


# --------------------------------------------------------------------------- #
# Band planning                                                                #
# --------------------------------------------------------------------------- #

def min_occupied_for_baud(min_baud: float, symbol_rate: float,
                          pilot_spacing: int) -> int:
    """Occupied carriers needed to reach `min_baud` at 1 bit/carrier.

    Counts pilots, which carry no payload: allocate() skips them, so an
    occupied span of N yields only N - ceil(N/pilot_spacing) data carriers.
    BPSK (1 bit) is assumed because the floor must hold in the worst case a
    link can negotiate.
    """
    if min_baud <= 0 or symbol_rate <= 0:
        return 1
    need_data = int(math.ceil(min_baud / symbol_rate))
    n = need_data
    while n - len(range(0, n, pilot_spacing)) < need_data:
        n += 1
    return n


def plan_duplex_bands(freq_lo: float, freq_hi: float, dl_frac: float,
                      guard_hz: float, n_fft: int, cp_ratio: float,
                      sample_rate: int, pilot_spacing: int,
                      min_baud: float, dl_side: str = "high") -> dict:
    """Split [freq_lo, freq_hi] into DL and UL sub-bands around a guard gap.

    The guard is real bandwidth spent, not free: it comes off the usable span
    BEFORE the ratio is applied, so dl_frac always describes the split of what
    is left to carry data.  Its job is to keep one direction's spectral
    skirts -- and the far side's frequency error -- out of the other's bins.

    `min_baud` is a floor on each direction, enforced after the ratio. A
    lopsided request (0.9/0.1 of a narrow band) can leave the minority
    direction with too few carriers to carry anything; clamping to the floor
    and reporting `ratio_clamped` is more useful than negotiating an uplink
    of zero carriers.

    `dl_side` defaults to 'high', the measured-correct choice.
    """
    sc_spc      = sample_rate / n_fft
    cp_len      = max(1, int(round(n_fft * cp_ratio)))
    symbol_rate = sample_rate / (n_fft + cp_len)

    span   = float(freq_hi) - float(freq_lo)
    usable = span - float(guard_hz)
    if usable <= 0:
        raise ValueError(
            f"guard_hz={guard_hz} consumes the whole {span:.0f}Hz band")

    min_occ = min_occupied_for_baud(min_baud, symbol_rate, pilot_spacing)
    min_bw  = min_occ * sc_spc
    if 2 * min_bw > usable:
        raise ValueError(
            f"band too narrow for duplex: {usable:.0f}Hz usable after a "
            f"{guard_hz:.0f}Hz guard, but each direction needs {min_bw:.0f}Hz "
            f"to sustain {min_baud:.0f} baud (raise freq_hi, lower min_baud, "
            f"or shrink guard_hz)")

    dl_bw = max(min_bw, min(usable - min_bw, dl_frac * usable))
    ul_bw = usable - dl_bw
    clamped = abs(dl_bw - dl_frac * usable) > 0.5 * sc_spc

    if dl_side == "low":
        dl = (freq_lo, freq_lo + dl_bw)
        ul = (freq_lo + dl_bw + guard_hz, freq_hi)
    else:
        ul = (freq_lo, freq_lo + ul_bw)
        dl = (freq_lo + ul_bw + guard_hz, freq_hi)

    # Does the lower band's distortion land inside the upper one? Splatter
    # travels upward; a guard band only helps against the skirt just above
    # the edge, not a harmonic (at 2x or 3x the source frequency). The fix is
    # geometric: keep the whole duplex span inside one octave and the lower
    # band's 2nd harmonic falls above the upper band entirely.
    lo_b, hi_b = (ul, dl) if ul[0] < dl[0] else (dl, ul)
    harm = []
    for n in (2, 3):
        h0, h1 = n * lo_b[0], n * lo_b[1]
        if h0 < hi_b[1] and h1 > hi_b[0]:
            ov = (min(h1, hi_b[1]) - max(h0, hi_b[0])) / (hi_b[1] - hi_b[0])
            harm.append({"harmonic": n,
                         "range_hz": [round(h0, 1), round(h1, 1)],
                         "covers_frac": round(ov, 3)})

    return {
        "dl_hz": [round(dl[0], 1), round(dl[1], 1)],
        "ul_hz": [round(ul[0], 1), round(ul[1], 1)],
        "harmonic_overlap": harm,
        "within_one_octave": bool(freq_hi < 2.0 * freq_lo),
        "guard_hz": [round(min(dl[1], ul[1]), 1), round(max(dl[0], ul[0]), 1)],
        "dl_frac_requested": round(float(dl_frac), 3),
        "dl_frac_actual": round(dl_bw / usable, 3),
        "ratio_clamped": bool(clamped),
        "min_baud": float(min_baud),
        "min_bw_hz": round(min_bw, 1),
        "symbol_rate_hz": round(symbol_rate, 2),
        "carrier_spacing_hz": round(sc_spc, 2),
        "dl_side": dl_side,
    }


def band_cfg(band_hz, sample_rate: int, *, n_fft: int = 1024,
             cp_ratio: float = 0.25, pilot_spacing: int = 8,
             probe_duration_s: float = 5.0, snr_margin_db: float = 6.0,
             max_bits_per_bin: int = 6, probe_meas_syms: int = 32,
             isi_derate: float = 1.0, **extra) -> dict:
    """build_config restricted to ONE direction's sub-band.

    The band split needs no allocator change: build_config already derives
    bin_lo/bin_hi from freq_lo/freq_hi, so two configs over disjoint frequency
    ranges are two disjoint carrier sets, with the gap between them as guard.

    `isi_derate` defaults on here, unlike the half-duplex OFDM paths. The
    probe measures the reverberation ceiling; with derating off it is
    recorded and then ignored, so the allocator sizes carriers against an
    SNR the path cannot deliver. Derating and snr_margin_db answer the same
    question, so with this on the margin should come down.
    """
    return build_config(
        n_fft            = int(n_fft),
        cp_ratio         = float(cp_ratio),
        freq_lo          = float(band_hz[0]),
        freq_hi          = float(band_hz[1]),
        pilot_spacing    = int(pilot_spacing),
        sample_rate      = int(sample_rate),
        probe_duration_s = float(probe_duration_s),
        snr_margin_db    = float(snr_margin_db),
        max_bits_per_bin = int(max_bits_per_bin),
        probe_meas_syms  = int(probe_meas_syms),
        isi_derate       = float(isi_derate),
        **extra,
    )


# --------------------------------------------------------------------------- #
# Frame sizing                                                                 #
# --------------------------------------------------------------------------- #

def chunk_for_frame_len(cfg: dict, bat, max_frame_s: float) -> int:
    """Shrink chunk_bytes until ONE frame fits inside `max_frame_s`.

    A frame is a single RS codeword: it decodes completely or not at all, and
    its length in symbols is chunk_bytes*8 / sum(bat). On a thin table that
    can reach e.g. 6 carriers -> 366 data symbols -> an eleven-second frame,
    far longer than this channel stays coherent, so the whole codeword fails
    together.

    Smaller chunks are strictly better here: more, shorter codewords mean a
    fade costs one frame instead of everything, and the receiver gets partial
    payload plus a real EVM number instead of silence. The cost is
    proportional RS/preamble overhead.
    """
    fs = float(cfg["sample_rate"])
    sym = float(cfg["symbol_samples"])
    bits = max(int(np.sum(np.asarray(bat))), 1)
    over = int(cfg["seq_bytes"]) + int(cfg["ecc_bytes"])
    for chunk in (256, 192, 128, 96, 64, 48, 32, 24, 16):
        n_syms = math.ceil((chunk + over) * 8 / bits)
        n_syms += 2 + math.ceil(n_syms / max(int(cfg.get("sync_period", 32)), 1))
        if n_syms * sym / fs <= max_frame_s:
            return chunk
    return 16


def size_for_airtime(cfg: dict, bat, target_s: float) -> tuple[int, int, float]:
    """Payload bytes whose transmission lasts about `target_s` seconds.

    Measured, not derived: frame airtime depends on n_fft, cp_ratio,
    sync_period, the inter-frame gap, and the BAT (a sparse table needs more
    symbols per chunk), so a closed form would rot as the geometry changes.
    Encode one frame, encode two, and the difference is the marginal cost.
    """
    modem = Modem(cfg)
    chunk = int(cfg["chunk_bytes"])
    fs = float(cfg["sample_rate"])
    one, _ = modem.encode_data(make_payload(chunk), bat)
    two, _ = modem.encode_data(make_payload(chunk * 2), bat)
    base_s = len(one) / fs
    marg_s = max((len(two) - len(one)) / fs, 1e-6)
    n_frames = max(1, int(round((target_s - base_s) / marg_s)) + 1)
    return n_frames * chunk, n_frames, base_s + (n_frames - 1) * marg_s


def make_payload(n_bytes: int) -> str:
    return "".join(random.choices(_ALPHABET, k=int(n_bytes)))


def sum16(data: bytes) -> str:
    """Short stable digest for a log line: 16 hex chars of SHA-256."""
    return hashlib.sha256(data).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Receive filter                                                               #
# --------------------------------------------------------------------------- #

def rx_filter(x: np.ndarray, fs: int, far_cfg: dict, my_cfg: dict,
              *, order: int = 6, ripple_db: float = 0.5,
              stop_db: float = 60.0, max_order: int = 12):
    """Reject this node's own band from a capture before decoding the far one.

    Not optional. The probe phase gets away without it because it analyses
    per-bin after an FFT, so own-TX lands in bins it never reads. The data
    path acquires frames in the time domain, before any FFT, so own-TX is
    fully present in the correlation and its normalisation. Measured on a
    synthetic two-band mix at equal RMS: 0 of 10 frames acquired unfiltered,
    10 of 10 filtered.

    One-sided: each band only needs rejection on the side the other band is
    on. A two-sided bandpass's extra highpass transition rang long enough to
    eat the first frame (9/10) where a plain lowpass took all 10.

    Elliptic, sized to the guard, not a fixed-order Butterworth: own TX
    arrives ~50 dB above the far signal in this node's own mic, and a
    6th-order Butterworth cutting at the middle of a 300 Hz guard only
    attenuates it 4.6-14.9 dB, leaving it +36..+45 dB. Because own TX is not
    symbol-aligned with the far node's frames it smears across the SC
    correlation, so acquisition -- not SNR -- is what gets lost. Passband
    edge is the far band's near edge, stopband edge is this node's near
    edge, so the whole guard is spent as transition.

    Returns (filtered, description) or (None, None) if nothing to reject.
    """
    ny = fs / 2.0
    f_lo, f_hi = float(far_cfg["freq_lo_hz"]), float(far_cfg["freq_hi_hz"])
    m_lo, m_hi = float(my_cfg["freq_lo_hz"]), float(my_cfg["freq_hi_hz"])
    if f_lo > m_hi:                      # far band sits above mine
        wp, ws, btype = f_lo, m_hi, "high"
    elif f_hi < m_lo:                    # far band sits below mine
        wp, ws, btype = f_hi, m_lo, "low"
    else:
        return None, None                # overlapping bands: nothing to reject
    wp = min(max(wp, 20.0), ny * 0.99)
    ws = min(max(ws, 20.0), ny * 0.99)

    try:
        n, wn = ellipord(wp / ny, ws / ny, ripple_db, stop_db)
        n = int(min(n, max_order))
        sos = ellip(n, ripple_db, stop_db, wn, btype=btype, output="sos")
        desc = f"{btype}pass ellip n={n} {wp:.0f}/{ws:.0f}Hz"
    except Exception:                                        # noqa: BLE001
        # Guard too narrow for the requested selectivity: fall back to a
        # Butterworth and label the description accordingly.
        cut = (wp + ws) / 2.0
        try:
            sos = butter(order, min(max(cut, 20.0), ny * 0.99) / ny,
                         btype=btype, output="sos")
        except Exception:                                    # noqa: BLE001
            return None, None
        desc = f"{btype}pass butter{order}@{cut:.0f}Hz (ellip unavailable)"
    try:
        return sosfiltfilt(sos, x).astype(np.float32), desc
    except Exception:                                        # noqa: BLE001
        return None, None
