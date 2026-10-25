# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""
myspectrograph.py  —  Live Audio Spectrograph with CSS Decoder Overlay
=======================================================================

KEYBOARD CONTROLS
  T       toggle timing markers   (white tick at each symbol boundary)
  O       toggle chirp overlay    (diagonal line showing ideal chirp path)
  L       toggle per-symbol overlay-sync log to stdout (sc/bit/lag/latency;
           SYNC GAP lines always print when a telemetry packet is dropped)
  R       reload css_info.json    (picks up a new encoder run mid-session)
  + / -   adjust decoder lag compensation (step scales with symbol width; hold
           SHIFT for a 2 px fine trim).  Tune until the overlay + decode tags
           align with the visible chirp.  Positive = shift overlay left = older
           audio.  Marks are already back-dated by the decoder "wt" timestamp,
           so this only trims the constant capture offset between the two
           independent audio streams.
  Esc     quit

MOUSE
  Click (in the spectrogram, live mode)  flag the nearest symbol overlay as a
           suspected wrong decode.  A red box + "⚑ FLAG #sc" marks it and a JSON
           line is appended to spectrograph_flags.jsonl (sc = decoder symbol
           index — the locator for offline confirmation; also dir/conf/wt/chirp
           params).  Note the chirp yourself within a few positions; sc anchors
           the search.  (Clicks in the UI panel still drag the sliders.)

SCROLLBACK (history is kept in a numpy ring buffer — default 10 min, set via
  "hist_seconds" in spectrograph.json):
  Wheel / ← →     scroll through history (wheel up / ← = back in time)
  PgUp / PgDn     scroll one screen-width
  Space           pause/resume live auto-scroll
  Home / End      jump to oldest retained / snap back to live
  While frozen the decoder symbol/sync/frame marks are hidden (they track the
  live edge); the frequency guides remain.  A scrollbar + "-Ns" readout at the
  top shows your position.

OVERLAY VISUALS
  ──────────────── f_max ···········  (dashed blue guide line)
       ╱ up-chirp (cyan, bit=1)
      ╲ dn-chirp (magenta, bit=0)
  ──────────────── f_min ···········  (dashed blue guide line)
  │ timing mark (white vertical bar at symbol left-edge)
  │ SYNC mark   (yellow, labelled)
  │ Frame mark  (green, labelled with frame count)

  Brightness of each chirp line scales with decoder confidence.

DECODER INTEGRATION — built in to css_decoder.py
====================================================
This telemetry is ALREADY integrated into css_decoder.py — both the streaming
PLL path decode() and the long-symbol matched-filter path decode_longsym_mf()
emit it (try/except wrapped, so it can never affect decoding).  The four blocks
below are kept only as the wire-schema reference.  If the overlay shows
"waiting on UDP", the decoder process is not sending here — check it is running
on THIS host (telemetry goes to 127.0.0.1:55555, so the decoder and this viewer
must share a machine).

Note (matched-filter path): there is no PLL, so "te" is the per-symbol MF timing
residual (found − predicted) and "nr_dev"/"rate_est" report the historic-grid
drift off nominal sps_eff rather than an n_read deviation.

  ── [A] Once, near the top of decode(), before the sd.InputStream context: ──

    import socket as _telem_mod
    _telem      = _telem_mod.socket(_telem_mod.AF_INET, _telem_mod.SOCK_DGRAM)
    _TELEM_ADDR = ("127.0.0.1", 55555)

  ── [B] Every symbol — after the Timing PLL block (after timing_err is set): ──

    try:
        _telem.sendto(json.dumps({
            "t":       "s",
            "b":       bit,
            "c":       round(confidence, 3),
            "te":      timing_err,
            # ── Memory profile (raw data bytes; no Python overhead) ──────────
            "m_ring":  ring.nbytes,                        # ADC ring buffer (fixed)
            "m_fa":    _fa_nbytes,                         # frame audio accumulator
            "m_bs":    len(bit_stream) * 8,                # soft-decision bit pairs
            "m_misc":  (len(pre_sync_buf) +                # sync candidate history
                        len(snr_history) +                 # SNR log
                        len(rms_history)) * 8,             # RMS log
            # ── PLL state ────────────────────────────────────────────────────
            "nr_dev":   n_read - sps_eff,                  # instantaneous PLL deviation (smp)
            "rate_est": round(rate_est_smp, 2),            # persistent clock-drift estimate (smp/sym)
            # ── Misc ─────────────────────────────────────────────────────────
            "sync":    sync_found,                         # True while collecting frame bits
            "sc":      sym_count,                          # total symbols processed
        }).encode(), _TELEM_ADDR)
    except Exception:
        pass

  ── [C] Inside "if acquired:" after sync_found = True: ──

    try:
        _telem.sendto(json.dumps({
            "t":      "sync",
            "sc":     round(score, 3),
            "f0":     f_min, "f1": f_max, "sd": dur,
            # Modem sizing — used by the viewer to scale memory-meter axes
            "sps":     sps,        # samples per chirp (without gap)
            "sps_eff": sps_eff,    # effective symbol period (chirp + gap)
            "fr":      frame_symbols,
        }).encode(), _TELEM_ADDR)
    except Exception:
        pass

  ── [D] Inside the successful frame decode block, after frames_decoded += 1: ──

    try:
        _telem.sendto(json.dumps({
            "t":       "frame",
            "seq":     seq_num,      # frame sequence number
            "n":       frames_decoded,
            "sc":      sym_count,    # total symbols at frame decode time
            "fa_peak": _fa_peak_bytes,   # frame-audio bytes freed at this boundary
        }).encode(), _TELEM_ADDR)
    except Exception:
        pass
"""

import sys
import os
import json
import queue
import threading
import socket
import time
import pygame
import pyaudio
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
CONFIG_FILE   = "spectrograph.json"
CSS_INFO_FILE = "css_info.json"
FLAG_LOG      = "spectrograph_flags.jsonl"   # click-to-flag records (for offline confirm)
TELEM_PORT    = 55555

CHANNELS   = 1
FPS        = 30

# Open the capture at the input device's NATIVE rate rather than forcing 44100.
# The decoder already derives fs from the device's native rate (a 48 kHz mic
# forced to 44.1 k destroys long-chirp phase and, here, mislabels both the time
# and frequency axes — the chirp overlay then diverges from the spectrogram, and
# worse the longer the chirp).  Matching the device rate makes chunk_size, the
# FFT Nyquist and the column rate all self-consistent.  Falls back to 44100.
RATE = 44100
p    = pyaudio.PyAudio()
try:
    _dev_rate = int(round(p.get_default_input_device_info()["defaultSampleRate"]))
    if 8000 <= _dev_rate <= 192000:
        RATE = _dev_rate
except Exception as exc:
    print(f"[spec] default-input rate query failed ({exc}); using {RATE} Hz", flush=True)
print(f"[spec] capture rate {RATE} Hz", flush=True)

chunk_size = RATE // FPS

UI_HEIGHT = 310   # expanded: sliders + status + 3 meter bars (mem / PLL / S/N)

# ─────────────────────────────────────────────────────────────────────────────
cfg = {
    "min_db":      20.0,
    "max_db":     100.0,
    "max_hz":   22050.0,
    "color":        1.0,
    "scroll":       1.0,
    "width":      800,
    "height":     700,
    "show_timing":  True,
    "show_chirp":   True,
    "decoder_lag":  0,
    "hist_seconds": 600.0,   # scrollback history depth (10 min default)
    "log_telem":    False,   # per-symbol overlay-sync log to stdout (toggle: L)
}

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as fh:
            loaded = json.load(fh)
        for k, v in loaded.items():
            if k in cfg:
                cfg[k] = v
    except Exception:
        pass

WIDTH       = int(cfg["width"])
HEIGHT      = int(cfg["height"])
SPEC_HEIGHT = max(100, HEIGHT - UI_HEIGHT)

# ─────────────────────────────────────────────────────────────────────────────
# Spectrogram history  (numpy ring buffer of raw-dB columns)
# ─────────────────────────────────────────────────────────────────────────────
# We store one column per audio chunk (the natural time slice — independent of
# the Scroll-Width zoom factor) at full FFT-bin resolution.  Storing the *data*
# (not rendered pixels) means the Min/Max dB, Color and Max-Hz sliders re-apply
# retroactively to history, and a window resize re-interpolates cleanly.
#
# float16 keeps it cheap: N_BINS≈736 → 1.47 KB/col → ~26 MB for 600 s @ 30 fps.
N_BINS   = chunk_size // 2 + 1
MAX_COLS = max(1, int(float(cfg["hist_seconds"]) * FPS))
hist     = np.full((MAX_COLS, N_BINS), -120.0, dtype=np.float16)  # dB silence floor
hist_w   = 0    # ring write pointer  (next slot to write, mod MAX_COLS)
hist_n   = 0    # total columns ever written (monotonic; clamps available window)

# View state: live edge tracking vs. frozen scrollback.
live           = True   # True = auto-scroll at the newest column
view_right_abs = 0      # absolute column index shown at the screen's right edge
resync         = False  # one-shot: rebuild spec_surface from hist on next frame

# ─────────────────────────────────────────────────────────────────────────────
# Chirp parameters  (from css_info.json, updated live on sync events)
# ─────────────────────────────────────────────────────────────────────────────
chirp = {
    "f_min":      400.0,
    "f_max":     6000.0,
    "symbol_dur":  0.25,
    "loaded":     False,
}

def load_css_info():
    if not os.path.exists(CSS_INFO_FILE):
        return False
    try:
        with open(CSS_INFO_FILE) as fh:
            info = json.load(fh)
        chirp.update({
            "f_min":      float(info.get("f_min",       400.0)),
            "f_max":      float(info.get("f_max",      6000.0)),
            "symbol_dur": float(info.get("symbol_dur",   0.25)),
            "loaded":     True,
        })
        print(f"[spec] css_info: {chirp['f_min']:.0f}–{chirp['f_max']:.0f} Hz  "
              f"sym={chirp['symbol_dur']}s", flush=True)
        return True
    except Exception as exc:
        print(f"[spec] css_info error: {exc}", flush=True)
        return False

load_css_info()

# ─────────────────────────────────────────────────────────────────────────────
# Decoder telemetry  (background UDP receiver thread)
# ─────────────────────────────────────────────────────────────────────────────
telem_queue  = queue.Queue()
telem_status = {
    "connected":  False,
    "last_rx":    0.0,
    "sync":       False,
    "sync_score": 0.0,
    "frames":     0,
    "last_seq":   -1,    # sequence number of the most-recently decoded frame
    "last_terr":  0,
    "last_conf":  0.0,
    # ── Memory profile (raw data bytes from css_decoder.py telemetry) ─────────
    "m_ring":     0,     # fixed ADC ring buffer
    "m_fa":       0,     # _frame_audio accumulator (the sawtooth)
    "m_bs":       0,     # bit_stream soft-decision pairs
    "m_misc":     0,     # pre_sync / SNR / RMS history
    "fa_peak":    0,     # peak m_fa of the last completed frame (high-water mark)
    "mem_peak":   0,     # high-water mark of TOTAL live bytes (auto-range axis)
    # ── PLL health ────────────────────────────────────────────────────────────
    "last_nr_dev": 0,    # raw n_read − sps_eff (instantaneous PLL correction)
    "nr_dev_ema":  0.0,  # EMA-smoothed version for bar display
    "rate_est":    0.0,  # persistent TX/RX clock-drift estimate (samples/symbol)
    # ── Signal quality ────────────────────────────────────────────────────────
    "conf_ema":    0.0,  # EMA of |confidence| for smooth S/N bar
    # ── Decoder heartbeat ─────────────────────────────────────────────────────
    "sym_count":   0,    # total symbols processed by the decoder (rolling counter)
    # ── Overlay-sync diagnostics (detect dropped/misordered telemetry) ────────
    # The overlay's per-symbol direction is correct only if every "s" packet
    # arrives in order: a single dropped UDP packet shifts ALL later overlays
    # one chirp to the right, so a symbol's direction is then drawn over its
    # neighbour (opposite direction) — the "symbol decoded reversed in the
    # spectrograph" symptom.  These counters + the gap log below make that
    # visible and let it be cross-checked against the decoder's own log.
    "last_sc":    -1,    # sc (sym index) of the previous "s" event received
    "n_sym_rx":    0,    # total "s" events received
    "n_sym_gap":   0,    # count of sc discontinuities (≈ packets dropped/reordered)
    "n_sym_lost":  0,    # total symbols implied missing by those gaps
    "lat_ema":     0.0,  # EMA of telemetry latency (recv − send wall clock), s
    # ── Modem config (populated from the "sync" telemetry event) ─────────────
    "sps_eff":    44100, # effective samples-per-symbol (sps + gap_samps)
    "fr":         74,    # frame_symbols (used to size the theoretical-max line)
}

def _udp_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(0.5)
    try:
        sock.bind(("127.0.0.1", TELEM_PORT))
        print(f"[spec] telemetry UDP 127.0.0.1:{TELEM_PORT}", flush=True)
    except OSError as exc:
        print(f"[spec] UDP bind failed ({exc}) — decoder overlay disabled", flush=True)
        return
    while True:
        try:
            data, _ = sock.recvfrom(4096)
            telem_queue.put(json.loads(data.decode()))
            telem_status["connected"] = True
            telem_status["last_rx"]   = time.time()
        except socket.timeout:
            # Dynamic threshold: must survive at least one full retro-FFT
            # without a packet.  For large symbols (e.g. sps_eff=44100,
            # fr=74) the blocking FFT can take 5–15 s; a fixed 3 s fires
            # false disconnects and causes the meter bars to flash.
            sps_e = telem_status.get("sps_eff", 44100)
            fr_   = telem_status.get("fr", 74)
            dyn_timeout = max(6.0, sps_e * fr_ / 44100 * 3.0)
            if telem_status["connected"] and (
                    time.time() - telem_status["last_rx"] > dyn_timeout):
                telem_status["connected"] = False
        except Exception:
            pass

threading.Thread(target=_udp_receiver, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# Audio
# ─────────────────────────────────────────────────────────────────────────────
audio_queue = queue.Queue()

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def _open_stream(rate, frames):
    s = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=rate,
        input=True,
        frames_per_buffer=frames,
        stream_callback=audio_callback,
    )
    s.start_stream()
    return s

try:
    stream = _open_stream(RATE, chunk_size)
except Exception as exc:
    # The device's reported native rate may not actually be openable in this
    # format — fall back to 44100.
    print(f"[spec] open @ {RATE} Hz failed ({exc}); falling back to 44100", flush=True)
    if RATE != 44100:
        RATE       = 44100
        chunk_size = RATE // FPS
        N_BINS     = chunk_size // 2 + 1
        hist       = np.full((MAX_COLS, N_BINS), -120.0, dtype=np.float16)
        hist_w     = 0
        hist_n     = 0
        try:
            stream = _open_stream(RATE, chunk_size)
        except Exception as exc2:
            print(exc2); p.terminate(); sys.exit(1)
    else:
        p.terminate(); sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Pygame
# ─────────────────────────────────────────────────────────────────────────────
pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT),
    pygame.RESIZABLE | pygame.DOUBLEBUF,
    vsync=1,
)
pygame.display.set_caption(
    "Spectrograph  |  T=timing  O=overlay  L=synclog  R=reload  +/-=lag  |  "
    "click=flag symbol  |  wheel/←→/PgUp-Dn=scroll  Space=pause  Home/End=oldest/live")

clock   = pygame.time.Clock()
font    = pygame.font.SysFont(None, 24)
font_sm = pygame.font.SysFont(None, 18)
font_xs = pygame.font.SysFont(None, 15)   # compact legend text inside meter bars

spec_surface = pygame.Surface((WIDTH, SPEC_HEIGHT))
spec_surface.fill((0, 0, 0))

# ─────────────────────────────────────────────────────────────────────────────
# Sliders
# ─────────────────────────────────────────────────────────────────────────────
sliders = [
    {"key": "min_db",  "label": "Min dB",       "min":    0.0, "max": 140.0,    "row": 0, "col": 0},
    {"key": "max_db",  "label": "Max dB",        "min":    0.0, "max": 140.0,    "row": 0, "col": 1},
    {"key": "max_hz",  "label": "Max Hz",        "min":  500.0, "max": RATE/2,   "row": 1, "col": 0},
    {"key": "color",   "label": "Color",         "min":    0.0, "max":   2.0,    "row": 1, "col": 1},
    {"key": "scroll",  "label": "Scroll Width",  "min":    1.0, "max":   8.0,    "row": 2, "col": 0},
]

dragging = None

def get_rect(row, col, w, sh):
    margin_x = 50
    margin_y = 30
    col_w    = w // 2
    return pygame.Rect(
        margin_x + col * col_w,
        sh + margin_y + row * 40,
        col_w - 2 * margin_x,
        10,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────────────────────────────────────
#
# overlay_annotations  — list of dicts, one per decoder event still on-screen.
#
#   {"type": "sym",   "x": int, "bit": 0|1, "conf": float}
#   {"type": "sync",  "x": int}
#   {"type": "frame", "x": int, "n": int}
#
# "x" is the RIGHT edge of the symbol in spectrograph pixel coordinates.
# Each live display frame, x is decremented by the rendered scroll width so
# annotations travel left with the spectrogram and are pruned once off-screen.
# Annotations are only updated/drawn in live mode (see set_view_right).
#
overlay_annotations = []

# Actual audio-column production rate (columns/sec), MEASURED live from the
# capture — do NOT assume it equals FPS.  The spectrogram scrolls one column per
# delivered audio chunk at (real_device_rate / chunk_size) cols/sec; if the mic's
# native rate differs from RATE (e.g. a 48 kHz device opened at 44100, delivering
# 48000/1470 = 32.65 cols/sec), a drawn chirp keyed to the nominal FPS is scaled
# ~9 % wrong — an error that accumulates over the chirp's length.  Measuring the
# true rate makes the drawn chirp width match the audio regardless of platform
# resampling behaviour.  Seeded to FPS until enough audio is measured.
cols_per_sec = float(FPS)

def sym_pixels():
    """Width of one symbol's chirp in display pixels, using the MEASURED audio
    column rate (not nominal FPS) so the drawn slope matches the spectrogram."""
    return max(1.0, chirp["symbol_dur"] * cols_per_sec * max(1, int(cfg["scroll"])))

def freq_to_y(f):
    """
    Map a frequency (Hz) → spectrograph y-pixel.
    Convention: y=0 = top = max_hz,  y=SPEC_HEIGHT-1 = bottom = 0 Hz.
    """
    mhz = max(1.0, cfg["max_hz"])
    return max(0, min(SPEC_HEIGHT - 1, int((1.0 - float(f) / mhz) * SPEC_HEIGHT)))

def flag_symbol_at(click_x):
    """Flag the on-screen symbol overlay nearest `click_x` (a suspected wrong
    decode) and append a JSONL record so it can be located/confirmed offline.

    The KEY locator is `sc`, the decoder's symbol index: it indexes both the
    decoder's `MIC_SN` stdout log AND the matched-filter grid
    (pred = grid_t0 + sc·grid_sps), so the exact audio window can be pulled from
    the run's WAV.  `wt` (decoder send wall-clock) is a secondary/cross-host
    anchor.  Returns the flagged annotation, or None if no symbol was near the
    click.  The record is deliberately self-contained (chirp params included) so
    a reader needs only this line + the run's WAV + css_info.json.
    """
    # Nearest "sym" annotation within one symbol width of the click.
    tol   = max(20.0, sym_pixels())
    best  = None
    for ann in overlay_annotations:
        if ann.get("type") != "sym":
            continue
        d = abs(int(ann["x"]) - click_x)
        if best is None or d < best[0]:
            best = (d, ann)
    if best is None or best[0] > tol:
        return None
    ann = best[1]
    ann["flagged"] = True
    bit = ann.get("bit", 0)
    rec = {
        "ts":            time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event":         "flag",
        "sc":            ann.get("sc", -1),          # decoder symbol index (primary locator)
        "dir":           "UP" if bit == 1 else "DN",  # what the decoder decided
        "bit":           bit,
        "conf":          round(float(ann.get("conf", 0.0)), 3),
        "wt":            ann.get("wt"),               # decoder send wall-clock (s)
        "f_min":         chirp["f_min"],
        "f_max":         chirp["f_max"],
        "symbol_dur":    chirp["symbol_dur"],
        "decoder_lag_px": int(decoder_lag),
        "cols_per_sec":  round(cols_per_sec, 2),
        "note":          "user-flagged suspected wrong symbol; confirm sc +/- a few",
    }
    try:
        with open(FLAG_LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as exc:
        print(f"[spec] flag log write failed: {exc}", flush=True)
    print(f"[spec] FLAG sc={rec['sc']} dir={rec['dir']} conf={rec['conf']:+.2f} "
          f"wt={rec['wt']} -> {FLAG_LOG}", flush=True)
    return rec

def draw_overlays(surface, show_annotations=True):
    """
    Render all active annotations onto `surface` (the screen, already showing
    the live spectrogram).  Called once per display frame.

    Draws:
      • Dashed frequency guides at f_min / f_max.
      • Per-symbol: timing tick (T) and/or chirp diagonal (O).
      • Sync events: yellow vertical bar + SYNC label.
      • Frame events: green vertical bar + frame-count label.

    `show_annotations` is False during frozen scrollback — the moving symbol /
    sync / frame marks are positioned in live-edge coordinates and would be
    meaningless against a historical window, so only the static guides remain.
    """
    sp = sym_pixels()

    # ── Frequency guide lines ──────────────────────────────────────────────
    if chirp["loaded"] and show_chirp:
        for freq in (chirp["f_min"], chirp["f_max"]):
            yg = freq_to_y(freq)
            for xx in range(0, WIDTH, 16):
                pygame.draw.line(surface, (40, 40, 110),
                                 (xx, yg), (min(xx + 6, WIDTH - 1), yg))

    if not show_annotations:
        return

    # ── Per-annotation rendering ───────────────────────────────────────────
    for ann in overlay_annotations:
        atype = ann["type"]
        xr    = int(ann["x"])   # right edge of symbol

        if atype == "sym":
            xl   = int(xr - sp + 1)
            bit  = ann["bit"]
            conf = ann["conf"]
            sc   = ann.get("sc", -1)

            if show_chirp and chirp["loaded"]:
                bright = max(80, min(255, int(abs(conf) * 255)))
                if bit == 1:            # up-chirp: f_min → f_max  (lower-left → upper-right)
                    y0, y1  = freq_to_y(chirp["f_min"]), freq_to_y(chirp["f_max"])
                    col     = (0, bright, bright)           # cyan
                    dim     = (0, bright // 5, bright // 5)
                else:                   # down-chirp: f_max → f_min (upper-left → lower-right)
                    y0, y1  = freq_to_y(chirp["f_max"]), freq_to_y(chirp["f_min"])
                    col     = (bright, 0, bright)           # magenta
                    dim     = (bright // 5, 0, bright // 5)
                xl_c = max(0, xl)
                xr_c = min(WIDTH - 1, xr)
                if xl_c <= xr_c and xr > xl:
                    # Clip the diagonal to the visible x-range, INTERPOLATING
                    # the y endpoints so the slope is preserved.  Clamping x
                    # without recomputing y (the old behaviour) squeezed the
                    # full f_min→f_max span into a shorter x-range, bending the
                    # chirp line up/down at the screen border.
                    span = xr - xl
                    yl   = int(round(y0 + (y1 - y0) * (xl_c - xl) / span))
                    yr   = int(round(y0 + (y1 - y0) * (xr_c - xl) / span))
                    # Glow backing, then bright line on top
                    pygame.draw.line(surface, dim, (xl_c, yl), (xr_c, yr), 4)
                    pygame.draw.line(surface, col, (xl_c, yl), (xr_c, yr), 2)

            if show_timing:
                xm = xl     # timing mark at the LEFT edge = symbol boundary
                if 0 <= xm < WIDTH:
                    pygame.draw.line(surface, (170, 170, 170),
                                     (xm, 0), (xm, SPEC_HEIGHT - 1), 1)

            # ── Per-symbol decode-status label ────────────────────────────────
            # Draw WHAT the decoder decided (sym index · direction · confidence)
            # anchored to the chirp's left edge, so the decode can be read off
            # against the visible chirp underneath it.  This is the whole point
            # of the overlay: a genuine symbol flip shows as a "DN" tag sitting
            # on an obviously rising chirp (or vice-versa).  sc lets you cross-
            # reference the decoder's MIC_SN log for that exact symbol.
            if show_chirp and chirp["loaded"]:
                dir_txt = "UP" if bit == 1 else "DN"
                tag_col = (60, 230, 230) if bit == 1 else (230, 60, 230)  # cyan / magenta
                lbl = f"#{sc} {dir_txt} {conf:+.2f}" if sc >= 0 else f"{dir_txt} {conf:+.2f}"
                lbl_s = font_xs.render(lbl, True, tag_col)
                tx    = max(1, min(WIDTH - lbl_s.get_width() - 1, xl + 2))
                ty    = SPEC_HEIGHT - 16
                pygame.draw.rect(surface, (12, 12, 16),
                                 (tx - 1, ty - 1,
                                  lbl_s.get_width() + 2, lbl_s.get_height() + 1))
                surface.blit(lbl_s, (tx, ty))

            # ── User flag (click-to-mark suspected wrong decode) ──────────────
            if ann.get("flagged"):
                fxl = max(0, xl); fxr = min(WIDTH - 1, xr)
                if fxl < fxr:
                    pygame.draw.rect(surface, (255, 60, 60),
                                     (fxl, 0, fxr - fxl, SPEC_HEIGHT - 1), 2)
                    ftag = font_sm.render(f"⚑ FLAG #{sc}", True, (255, 80, 80))
                    fbx  = max(0, min(WIDTH - ftag.get_width(), xl + 2))
                    pygame.draw.rect(surface, (30, 0, 0),
                                     (fbx - 1, 32, ftag.get_width() + 2,
                                      ftag.get_height() + 1))
                    surface.blit(ftag, (fbx, 33))

        elif atype == "sync":
            if show_timing and 0 <= xr < WIDTH:
                pygame.draw.line(surface, (255, 215, 40),
                                 (xr, 0), (xr, SPEC_HEIGHT - 1), 2)
                lbl = font_sm.render("SYNC", True, (255, 215, 40))
                surface.blit(lbl, (max(0, xr - lbl.get_width() // 2), 2))

        elif atype == "frame":
            if show_timing and 0 <= xr < WIDTH:
                pygame.draw.line(surface, (40, 255, 100),
                                 (xr, 0), (xr, SPEC_HEIGHT - 1), 2)
                lbl = font_sm.render(f"F{ann.get('n', '')}", True, (40, 255, 100))
                surface.blit(lbl, (max(0, xr - lbl.get_width() // 2), 18))

# ─────────────────────────────────────────────────────────────────────────────
# Decoder meters  (memory stacked bar · PLL deviation bar · S/N confidence bar)
# ─────────────────────────────────────────────────────────────────────────────
#
# Layout of one meter row (LABEL_W | gap | BAR_W | gap | NUM_W):
#
#   "MEM"  ▕ring▏──────── frame_audio ────────▏bs▏m▏   6.2 / 6.5 KB
#   "PLL"  ◄───[danger]──╠══╪══╣──[danger]───►         +  42 smp
#   "S/N"  [████████████░░░░░░░░░░░░░░░░░░░░]           +0.82  te=+5
#
# All colours match modem_membench.py for visual consistency across tools.
#
METER_LABEL_W = 38
METER_NUM_W   = 106
METER_BAR_H   = 14
METER_ROW_H   = 30

# Segment colours
_C_RING = (108, 117, 125)   # grey  – ring buffer (fixed baseline)
_C_FA   = ( 13, 110, 253)   # blue  – _frame_audio (grows/clears)
_C_BS   = ( 32, 201, 151)   # teal  – bit_stream
_C_MISC = (245, 158,  11)   # amber – misc history
_C_BG   = ( 25,  25,  32)   # dark bar background
_C_BRD  = ( 65,  68,  80)   # bar border
_C_TXT  = (200, 200, 205)   # label text
_C_DIM  = (120, 122, 132)   # muted numeric text

# PLL / confidence threshold colours
_C_OK   = ( 50, 210, 120)   # green  – within safe zone
_C_WARN = (255, 193,   7)   # yellow – borderline
_C_BAD  = (220,  53,  69)   # red    – slip / weak signal

# Ghost line (frame audio high-water mark)
_C_GHOST = (255, 107, 107)


def draw_decoder_meters(surface, px, py, avail_w, stale_age=None):
    """
    Draw the three instrumentation meters at pixel position (px, py).
    avail_w   — available horizontal pixels (usually WIDTH − 2*margin).
    stale_age — seconds since last telemetry packet, or None if never received.
                Bars are dimmed (not hidden) when stale so they don't flash
                during long retro FFTs or the inter-frame gap.
    """
    # ── No data yet: show a waiting placeholder and return ────────────────────
    if stale_age is None:
        ph = font_xs.render(
            f"── decoder meters: run css_decoder.py on THIS host "
            f"(UDP {TELEM_PORT}) to activate ──",
            True, (70, 72, 80),
        )
        surface.blit(ph, (px, py + 16))
        return

    # ── Dimming factor: full brightness when fresh, 35 % when stale ──────────
    STALE_THRESH = 2.5   # seconds before dimming kicks in
    dim = max(0.35, 1.0 - max(0.0, stale_age - STALE_THRESH) / 8.0)
    # Helper: scale a colour tuple by dim factor
    def _c(col):
        return tuple(int(c * dim) for c in col)

    bar_w = avail_w - METER_LABEL_W - METER_NUM_W - 16

    def _bar_rect(row_y):
        """Return (lx, bx, by, nx) for a row at row_y."""
        lx = px
        bx = px + METER_LABEL_W + 6
        by = row_y + (METER_ROW_H - METER_BAR_H) // 2
        nx = bx + bar_w + 6
        return lx, bx, by, nx

    sps_e   = max(1, telem_status["sps_eff"])
    fr      = max(1, telem_status["fr"])
    m_ring  = telem_status["m_ring"]
    m_fa    = telem_status["m_fa"]
    m_bs    = telem_status["m_bs"]
    m_misc  = telem_status["m_misc"]
    fa_pk   = telem_status["fa_peak"]
    nr_ema  = telem_status["nr_dev_ema"]
    nr_raw  = telem_status["last_nr_dev"]
    rate_est = telem_status["rate_est"]
    conf_e  = telem_status["conf_ema"]
    conf_r  = telem_status["last_conf"]
    terr    = telem_status["last_terr"]

    # ── Row 1: Memory stacked bar ─────────────────────────────────────────────
    r1y          = py
    lx, bx, by, nx = _bar_rect(r1y)
    ring_bytes   = m_ring if m_ring else sps_e * 3 * 4
    fa_max_bytes = fr * sps_e * 4        # theoretical ceiling (full frame buffered)
    total_cur    = m_ring + m_fa + m_bs + m_misc
    mem_peak     = max(telem_status["mem_peak"], total_cur)
    # Auto-range the bar axis to the observed high-water mark of ACTUAL usage,
    # not the theoretical frame-audio ceiling.  During PLL training / sync only
    # the fixed ring buffer + small history are allocated — a tiny fraction of
    # fr*sps_eff — so anchoring the axis to fa_max made the bar read as empty
    # ("0 memory") even though memory IS in use.  The theoretical ceiling is
    # still reported in the numeric readout and as the reference tick below.
    # Floor at ~1.5× ring so the baseline buffer fills a meaningful, visible
    # portion the instant telemetry starts.
    total_max    = max(mem_peak * 1.15, ring_bytes * 1.5, total_cur + 1)

    pygame.draw.rect(surface, _c(_C_BG),  (bx, by, bar_w, METER_BAR_H))
    pygame.draw.rect(surface, _c(_C_BRD), (bx, by, bar_w, METER_BAR_H), 1)

    # Stacked segments — clip each to the remaining bar width to prevent overflow
    cx          = bx
    remaining_w = bar_w
    seg_info    = []
    for nbytes, col in ((m_ring, _C_RING), (m_fa, _C_FA),
                        (m_bs, _C_BS), (m_misc, _C_MISC)):
        seg_w = max(0, min(remaining_w, int(nbytes / total_max * bar_w)))
        # Guarantee any nonzero allocation is at least 1px so small but real
        # buffers (ring/history during training) never vanish to "0".
        if nbytes > 0 and seg_w == 0 and remaining_w > 0:
            seg_w = 1
        if seg_w > 0:
            pygame.draw.rect(surface, _c(col), (cx, by, seg_w, METER_BAR_H))
            seg_info.append((cx, seg_w, col))
        cx          += seg_w
        remaining_w -= seg_w
        if remaining_w <= 0:
            break

    # Ghost high-water line (last frame's peak)
    if fa_pk > 0 and total_max > 0:
        gx = min(bx + bar_w - 1, bx + int((ring_bytes + fa_pk) / total_max * bar_w))
        if bx < gx:
            pygame.draw.line(surface, _c(_C_GHOST),
                             (gx, by - 2), (gx, by + METER_BAR_H + 1), 1)

    # Theoretical-ceiling reference tick (fr*sps_eff full-frame buffer).  With
    # the auto-ranged axis this normally sits off the right edge during
    # training (lots of headroom) — draw a small ">" at the edge in that case
    # so the ceiling is acknowledged without distorting the live scale.
    if fa_max_bytes > 0:
        if fa_max_bytes <= total_max:
            tx = bx + int(fa_max_bytes / total_max * bar_w)
            pygame.draw.line(surface, _c((90, 92, 104)),
                             (tx, by - 2), (tx, by + METER_BAR_H + 1), 1)
        else:
            surface.blit(font_xs.render("›", True, _c((110, 112, 124))),
                         (bx + bar_w - 6, by))

    # Segment labels inside bar where width allows
    labels_in_bar = ((_C_RING, "ring"), (_C_FA, "fa"),
                     (_C_BS, "bs"), (_C_MISC, "m"))
    for (sx, sw, _scol), (_, short_lbl) in zip(seg_info, labels_in_bar):
        lbl_surf = font_xs.render(short_lbl, True, (220, 220, 220))
        if sw > lbl_surf.get_width() + 4:
            surface.blit(lbl_surf,
                         (sx + (sw - lbl_surf.get_width()) // 2,
                          by + (METER_BAR_H - lbl_surf.get_height()) // 2))

    surface.blit(font_sm.render("MEM", True, _c(_C_TXT)), (lx, r1y + 7))
    cur_s  = (f"{total_cur/1048576:.2f}MB" if total_cur >= 1048576
              else f"{total_cur/1024:.1f}KB")
    max_s  = (f"/{fa_max_bytes/1048576:.1f}MB" if fa_max_bytes >= 1048576
              else f"/{fa_max_bytes/1024:.0f}KB")
    # Warn colour reflects real headroom against the THEORETICAL ceiling, not
    # the auto-ranged display axis (the latter would warn at every frame peak).
    num_col = _C_OK if total_cur < fa_max_bytes * 0.80 else _C_WARN
    surface.blit(font_xs.render(cur_s + max_s + " max", True, _c(num_col)),
                 (nx, r1y + 8))

    # ── Row 2: PLL deviation bar ──────────────────────────────────────────────
    r2y             = py + METER_ROW_H + 4
    lx, bx, by, nx = _bar_rect(r2y)
    safe_smp        = sps_e * 0.04
    disp_range      = max(safe_smp * 3.5, abs(nr_ema) * 1.8, 1.0)
    cx_bar          = bx + bar_w // 2

    safe_px = max(1, int(safe_smp / disp_range * (bar_w // 2)))
    pygame.draw.rect(surface, _c((48, 18, 18)),
                     (bx, by, cx_bar - bx - safe_px, METER_BAR_H))
    pygame.draw.rect(surface, _c((48, 18, 18)),
                     (cx_bar + safe_px, by,
                      (bx + bar_w) - (cx_bar + safe_px), METER_BAR_H))
    pygame.draw.rect(surface, _c((18, 44, 28)),
                     (cx_bar - safe_px, by, safe_px * 2, METER_BAR_H))

    dev_px     = int(nr_ema / disp_range * (bar_w // 2))
    dev_px     = max(-bar_w // 2, min(bar_w // 2, dev_px))
    frac_safe  = abs(nr_ema) / max(safe_smp, 1)
    pll_col    = (_C_OK   if frac_safe < 0.55 else
                  _C_WARN if frac_safe < 1.0  else _C_BAD)
    if dev_px > 0:
        pygame.draw.rect(surface, _c(pll_col),
                         (cx_bar, by, dev_px, METER_BAR_H))
    elif dev_px < 0:
        pygame.draw.rect(surface, _c(pll_col),
                         (cx_bar + dev_px, by, -dev_px, METER_BAR_H))

    pygame.draw.line(surface, _c((210, 210, 215)),
                     (cx_bar, by - 1), (cx_bar, by + METER_BAR_H), 1)
    pygame.draw.line(surface, _c((90, 90, 90)),
                     (cx_bar + safe_px, by),
                     (cx_bar + safe_px, by + METER_BAR_H - 1), 1)
    pygame.draw.line(surface, _c((90, 90, 90)),
                     (cx_bar - safe_px, by),
                     (cx_bar - safe_px, by + METER_BAR_H - 1), 1)
    surface.blit(font_xs.render("◄", True, _c(_C_DIM)), (bx - 1,      by - 1))
    surface.blit(font_xs.render("►", True, _c(_C_DIM)), (bx + bar_w - 8, by - 1))
    pygame.draw.rect(surface, _c(_C_BRD), (bx, by, bar_w, METER_BAR_H), 1)

    surface.blit(font_sm.render("PLL", True, _c(_C_TXT)), (lx, r2y + 7))
    pll_num = f"{nr_raw:+d} smp"
    surface.blit(font_sm.render(pll_num, True, _c(pll_col)), (nx, r2y + 2))
    # Second line: persistent clock-drift estimate (rate_est, samples/symbol).
    # Distinct from nr_raw: nr_raw is the noisy per-symbol PLL output; rate_est
    # is the smoothed baseline absorbed by the type-2 loop after acquisition.
    # Near zero means TX and RX clocks are well matched; a persistent non-zero
    # value (e.g. +3 smp/sym at sps=11025 → +270 ppm) shows a real clock offset.
    rate_ppm = rate_est / sps_e * 1e6
    surface.blit(font_xs.render(f"r={rate_est:+.1f}smp ({rate_ppm:+.0f}ppm)",
                                True, _c(_C_DIM)), (nx, r2y + 17))
    surface.blit(font_xs.render(f"±{int(safe_smp)}smp safe", True, _c(_C_DIM)),
                 (cx_bar - safe_px - 4, by + METER_BAR_H + 1))

    # ── Row 3: Confidence / S/N bar ───────────────────────────────────────────
    r3y             = py + (METER_ROW_H + 4) * 2
    lx, bx, by, nx = _bar_rect(r3y)
    fill_px         = max(0, min(bar_w, int(conf_e * bar_w)))
    sn_col          = (_C_OK   if conf_e >= 0.70 else
                       _C_WARN if conf_e >= 0.40 else _C_BAD)
    dim_col         = tuple(max(0, c - 150) for c in sn_col)

    pygame.draw.rect(surface, _c(_C_BG), (bx, by, bar_w, METER_BAR_H))
    if fill_px > 0:
        pygame.draw.rect(surface, _c(sn_col), (bx, by, fill_px, METER_BAR_H))
    if fill_px < bar_w:
        pygame.draw.rect(surface, _c(dim_col),
                         (bx + fill_px, by, bar_w - fill_px, METER_BAR_H))
    for frac in (0.25, 0.50, 0.75):
        tx = bx + int(frac * bar_w)
        pygame.draw.line(surface, _c(_C_BRD), (tx, by), (tx, by + METER_BAR_H - 1), 1)
    pygame.draw.rect(surface, _c(_C_BRD), (bx, by, bar_w, METER_BAR_H), 1)

    surface.blit(font_sm.render("S/N", True, _c(_C_TXT)), (lx, r3y + 7))
    sn_txt = f"{conf_r:+.2f}  te={terr:+d}"
    surface.blit(font_sm.render(sn_txt, True, _c(sn_col)), (nx, r3y + 7))

    # ── Colour key row ────────────────────────────────────────────────────────
    ky = py + (METER_ROW_H + 4) * 3 + 2
    kx = bx
    for col, lbl in ((_C_RING, "ring"), (_C_FA, "frame_audio"),
                     (_C_BS, "bit_stream"), (_C_MISC, "misc")):
        pygame.draw.rect(surface, _c(col), (kx, ky, 8, 8))
        t = font_xs.render(lbl, True, _c(_C_DIM))
        surface.blit(t, (kx + 11, ky - 1))
        kx += 11 + t.get_width() + 12

    # ── Stale indicator (top-right of meter block) ────────────────────────────
    if stale_age is not None and stale_age > STALE_THRESH:
        stale_txt = font_xs.render(
            f"  ⏸ stale {stale_age:.0f}s", True, (110, 110, 120))
        surface.blit(stale_txt, (nx, py + 8))


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized spectrogram rendering + history access
# ─────────────────────────────────────────────────────────────────────────────
#
# The old renderer set pixels one-at-a-time in a Python double loop
# (SPEC_HEIGHT × scroll_width set_at() calls per chunk).  That is what caused
# the frame-rate hitching: a single 1470-sample chunk meant ~400 attribute
# lookups + tuple builds, multiplied by however many chunks were queued in a
# slow frame.  Everything below is pure-numpy and blitted in one surfarray op.

_vmap_cache = {}   # (SPEC_HEIGHT, max_bin) → (lo, hi, frac) freq-resample indices

def _vertical_map(max_bin):
    """Linear freq-bin → SPEC_HEIGHT resample indices (cached; recomputed only
    when the height or Max-Hz cutoff changes)."""
    key = (SPEC_HEIGHT, max_bin)
    m = _vmap_cache.get(key)
    if m is None:
        xi   = np.linspace(0.0, max_bin - 1, SPEC_HEIGHT)
        lo   = np.floor(xi).astype(np.intp)
        hi   = np.minimum(lo + 1, max_bin - 1)
        frac = (xi - lo).astype(np.float32)
        m    = (lo, hi, frac)
        _vmap_cache.clear()   # only ever need the current geometry
        _vmap_cache[key] = m
    return m

def render_cols(db_block):
    """
    Vectorized column renderer.

    db_block : (ncols, N_BINS) float array of dB magnitudes, low→high freq.
    Returns  : (ncols*scroll_width, SPEC_HEIGHT, 3) uint8 array suitable for
               pygame.surfarray.make_surface (x along axis 0, y top→bottom).
    """
    ncols   = db_block.shape[0]
    # Frequency axis keyed to the MEASURED capture rate, not the nominal RATE.
    # An rfft of `chunk_size` samples has bins spaced actual_fs/chunk_size Hz,
    # which equals the measured cols_per_sec exactly (cols_per_sec = actual_fs /
    # chunk_size).  If the device's true delivered rate differs from RATE — a
    # 48 kHz mic opened at 44.1 k, or PortAudio resampling — keying off RATE
    # mislabels every frequency, so the straight-line chirp overlay (drawn at
    # freq_to_y(f_min..f_max)) diverges from the real chirp, worsening toward the
    # high-frequency end of the sweep.  cols_per_sec self-calibrates both cases.
    freq_per_bin = cols_per_sec if cols_per_sec > 1e-6 else (RATE / chunk_size)
    max_bin = int(round(cfg["max_hz"] / freq_per_bin))
    max_bin = max(2, min(N_BINS, max_bin))
    lo, hi, frac = _vertical_map(max_bin)

    d   = db_block[:, :max_bin].astype(np.float32, copy=False)
    # Interpolate every column to SPEC_HEIGHT in one shot (low→high freq).
    col = d[:, lo] * (1.0 - frac) + d[:, hi] * frac          # (ncols, SPEC_HEIGHT)

    span  = max(1e-6, cfg["max_db"] - cfg["min_db"])
    inten = (col - cfg["min_db"]) / span
    np.clip(inten, 0.0, 1.0, out=inten)
    inten = inten[:, ::-1]    # flip so row 0 = top = high freq (matches freq_to_y)

    c = cfg["color"]
    if c <= 1.0:
        br, bg, bb = 255.0 * (1.0 - c), 255.0 * c, 0.0
    else:
        br, bg, bb = 0.0, 255.0 * (2.0 - c), 255.0 * (c - 1.0)

    rgb = np.empty((ncols, SPEC_HEIGHT, 3), dtype=np.uint8)
    rgb[..., 0] = (inten * br).astype(np.uint8)
    rgb[..., 1] = (inten * bg).astype(np.uint8)
    rgb[..., 2] = (inten * bb).astype(np.uint8)

    sw = max(1, int(cfg["scroll"]))
    if sw > 1:
        rgb = np.repeat(rgb, sw, axis=0)
    return rgb

def hist_view(right_abs, ncols):
    """Fetch `ncols` history columns ending at absolute index `right_abs`
    (inclusive), oldest→newest left→right.  Columns outside the retained ring
    window are filled with the dB silence floor (render to black)."""
    left_abs = right_abs - ncols + 1
    idxs     = np.arange(left_abs, right_abs + 1)
    valid    = (idxs >= 0) & (idxs < hist_n) & (idxs >= hist_n - MAX_COLS)
    block    = np.full((ncols, N_BINS), -120.0, dtype=np.float32)
    if valid.any():
        block[valid] = hist[idxs[valid] % MAX_COLS].astype(np.float32)
    return block

def render_window(right_abs):
    """Rebuild the entire spec_surface from history for a window whose right
    edge is absolute column `right_abs`.  Used for frozen scrollback and to
    repopulate after a resize so on-screen history survives."""
    sw      = max(1, int(cfg["scroll"]))
    visible = max(1, WIDTH // sw)
    spec_surface.fill((0, 0, 0))
    if hist_n <= 0:
        return
    arr = render_cols(hist_view(right_abs, visible))
    w   = min(WIDTH, arr.shape[0])
    if w > 0:
        spec_surface.blit(pygame.surfarray.make_surface(arr[:w]), (0, 0))

def set_view_right(nr):
    """Move the right-edge view to absolute column `nr`, clamped to the
    retained window.  Snaps to live auto-scroll when at/after the newest
    column; otherwise freezes scrollback (and clears stale live annotations)."""
    global live, view_right_abs, resync, overlay_annotations
    sw      = max(1, int(cfg["scroll"]))
    visible = max(1, WIDTH // sw)
    avail   = min(hist_n, MAX_COLS)
    newest  = hist_n - 1
    oldest  = hist_n - avail
    min_right = (oldest + visible - 1) if avail >= visible else newest
    min_right = max(0, min_right)
    nr        = max(min_right, min(newest, nr))
    if nr >= newest:
        if not live:
            resync = True          # rebuild from hist on the next frame
        live           = True
        view_right_abs = newest
    else:
        if live:
            overlay_annotations = []   # live-edge marks are meaningless when frozen
        live           = False
        view_right_abs = nr

def cur_view_right():
    return (hist_n - 1) if live else view_right_abs

# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
_cap_t0        = None   # wall clock of the first measured audio batch (rate calib)
_cap_cols      = 0      # columns counted since _cap_t0 (drives cols_per_sec)
_rate_measured = False  # True once cols_per_sec reflects real capture (post-warmup)

show_timing = bool(cfg.get("show_timing", True))
show_chirp  = bool(cfg.get("show_chirp",  True))
decoder_lag = int( cfg.get("decoder_lag", 0))
log_telem   = bool(cfg.get("log_telem",  False))  # 'L' toggles per-symbol overlay log

flash_msg = ""
flash_t   = 0.0
_last_sync_log = 0.0   # wall time of the last periodic SYNC-status summary

print(f"[spec] overlay-sync logging ready  decoder_lag={decoder_lag}px  "
      f"FPS={FPS}  scroll={int(cfg['scroll'])}  per-symbol log={'ON' if log_telem else 'off (press L)'}",
      flush=True)

running = True

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
while running:

    if not pygame.mouse.get_pressed()[0]:
        dragging = None

    # ── Events ────────────────────────────────────────────────────────────
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_t:
                show_timing = not show_timing
                flash_msg   = f"Timing markers: {'ON' if show_timing else 'off'}"
                flash_t     = time.time()

            elif event.key == pygame.K_o:
                show_chirp = not show_chirp
                flash_msg  = f"Chirp overlay: {'ON' if show_chirp else 'off'}"
                flash_t    = time.time()

            elif event.key == pygame.K_l:
                log_telem = not log_telem
                flash_msg = f"Per-symbol overlay log: {'ON' if log_telem else 'off'}"
                flash_t   = time.time()
                print(f"[spec] per-symbol overlay log {'ON' if log_telem else 'off'}",
                      flush=True)

            elif event.key == pygame.K_r:
                ok        = load_css_info()
                flash_msg = (
                    f"Loaded: {chirp['f_min']:.0f}–{chirp['f_max']:.0f} Hz  "
                    f"{chirp['symbol_dur']}s/sym"
                    if ok else "css_info.json not found"
                )
                flash_t = time.time()

            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS,
                               pygame.K_MINUS, pygame.K_KP_MINUS):
                # Step scales with the symbol width so a full-symbol offset is a
                # few presses, not ~135 at the old fixed 2 px.  Hold SHIFT for a
                # 2 px fine trim.  Readout shows px and the equivalent ms.
                mods = pygame.key.get_mods()
                step = 2 if (mods & pygame.KMOD_SHIFT) else max(4, int(sym_pixels() / 16))
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    step = -step
                decoder_lag += step
                _lag_ms = decoder_lag / (cols_per_sec * max(1, int(cfg["scroll"]))) * 1000.0
                flash_msg = f"Lag: {decoder_lag:+d} px  ({_lag_ms:+.0f} ms)"
                flash_t   = time.time()

            # ── Scrollback navigation ──────────────────────────────────────
            elif event.key in (pygame.K_LEFT, pygame.K_RIGHT,
                                pygame.K_PAGEUP, pygame.K_PAGEDOWN,
                                pygame.K_HOME, pygame.K_END, pygame.K_SPACE):
                _sw      = max(1, int(cfg["scroll"]))
                _visible = max(1, WIDTH // _sw)
                if event.key == pygame.K_LEFT:
                    set_view_right(cur_view_right() - max(1, _visible // 8))
                elif event.key == pygame.K_RIGHT:
                    set_view_right(cur_view_right() + max(1, _visible // 8))
                elif event.key == pygame.K_PAGEUP:
                    set_view_right(cur_view_right() - _visible)
                elif event.key == pygame.K_PAGEDOWN:
                    set_view_right(cur_view_right() + _visible)
                elif event.key == pygame.K_HOME:
                    set_view_right(0)              # oldest retained
                elif event.key == pygame.K_END:
                    set_view_right(hist_n)         # snap to live
                elif event.key == pygame.K_SPACE:
                    if live:
                        live           = False     # pause at the current edge
                        view_right_abs = max(0, hist_n - 1)
                        overlay_annotations.clear()
                    else:
                        set_view_right(hist_n)
                flash_msg = ("LIVE" if live
                             else f"HISTORY  -{((hist_n - 1) - view_right_abs) / FPS:.1f}s")
                flash_t = time.time()

        elif event.type == pygame.MOUSEWHEEL:
            # Wheel up = scroll back in time; down = toward live.
            _sw      = max(1, int(cfg["scroll"]))
            _visible = max(1, WIDTH // _sw)
            set_view_right(cur_view_right() - event.y * max(1, _visible // 6))

        elif event.type == pygame.VIDEORESIZE:
            WIDTH       = event.w
            HEIGHT      = event.h
            SPEC_HEIGHT = max(100, HEIGHT - UI_HEIGHT)

            screen = pygame.display.set_mode(
                (WIDTH, HEIGHT),
                pygame.RESIZABLE | pygame.DOUBLEBUF,
                vsync=1,
            )

            spec_surface = pygame.Surface((WIDTH, SPEC_HEIGHT))
            spec_surface.fill((0, 0, 0))
            # Repopulate from history at the new geometry so on-screen
            # spectrogram survives the resize (re-interpolated to the new
            # height / column width rather than pixel-stretched).
            _vmap_cache.clear()
            render_window(cur_view_right())

            # Stale annotation positions are relative to the old WIDTH; clear them.
            overlay_annotations.clear()

            cfg["width"]  = WIDTH
            cfg["height"] = HEIGHT

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Click inside the spectrogram (above the UI panel & scrollbar)
                # flags the nearest symbol overlay as a suspected wrong decode.
                # Only meaningful live (annotations track the live edge).
                if event.pos[1] < SPEC_HEIGHT - 6:
                    if live:
                        rec = flag_symbol_at(event.pos[0])
                        if rec is not None:
                            flash_msg = (f"⚑ Flagged sc={rec['sc']} "
                                         f"{rec['dir']} conf={rec['conf']:+.2f} "
                                         f"→ {FLAG_LOG}")
                        else:
                            flash_msg = "No symbol near click to flag"
                    else:
                        flash_msg = "Flagging only in LIVE mode (press End)"
                    flash_t = time.time()
                else:
                    for s in sliders:
                        rect = get_rect(s["row"], s["col"], WIDTH, SPEC_HEIGHT)
                        if rect.collidepoint(event.pos):
                            dragging = s["key"]

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging = None

        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                for s in sliders:
                    if s["key"] == dragging:
                        rect = get_rect(s["row"], s["col"], WIDTH, SPEC_HEIGHT)
                        mx  = max(rect.left, min(event.pos[0], rect.right))
                        val = float(mx - rect.left) / float(rect.width)
                        cfg[s["key"]] = s["min"] + val * (s["max"] - s["min"])
                        if dragging == "min_db" and cfg["min_db"] >= cfg["max_db"]:
                            cfg["min_db"] = cfg["max_db"] - 1.0
                        if dragging == "max_db" and cfg["max_db"] <= cfg["min_db"]:
                            cfg["max_db"] = cfg["min_db"] + 1.0

    # ── Decoder telemetry ─────────────────────────────────────────────────
    # Each event becomes an annotation placed at x = (right edge) - lag, then it
    # travels left with the spectrogram via the scroll shift below.
    #
    # Placement is BACK-DATED by the decoder's "wt" send timestamp instead of
    # always landing at the live edge.  The decoder stamps wt right after it
    # finishes a symbol's chirp, so (now - wt) is how long ago that chirp ended
    # in real time; converting that to pixels (columns advance at ~FPS, each
    # rendered `scroll` px wide) puts the mark on the audio it describes.  In the
    # live real-time case (now ≈ wt) this is a no-op vs the old behaviour, but it
    # removes per-symbol telemetry JITTER (a late packet no longer smears its
    # chirp onto the next, opposite-direction symbol) and makes replayed/bursty
    # sources — where many symbols arrive at once — land at their true times.
    # decoder_lag remains the manual trim for the constant capture offset between
    # the decoder's audio stream and this viewer's independent one.
    _scroll_px  = max(1, int(cfg["scroll"]))
    _px_per_sec = cols_per_sec * _scroll_px   # MEASURED rate, matches sym_pixels
    while not telem_queue.empty():
        try:
            evt = telem_queue.get_nowait()
        except queue.Empty:
            break

        et  = evt.get("t", "")
        # Place by the symbol's AUDIO time ("at") when the decoder provides it —
        # that is the wall clock at which the chirp was captured, so back-dating
        # by (now - at) lands the overlay ON the chirp.  Fall back to "wt" (send
        # time) for older decoders; note wt lags the chirp by the MF read-ahead,
        # leaving a constant right-shift that only decoder_lag could trim.
        _at = evt.get("at", evt.get("wt"))
        _back_px = (int(max(0.0, time.time() - float(_at)) * _px_per_sec)
                    if _at is not None else 0)
        ox = WIDTH - 1 - int(decoder_lag) - _back_px   # audio-aligned right edge

        if et == "s":
            bit  = int(evt.get("b", 0))
            conf = float(evt.get("c", 0.0))
            telem_status["last_terr"] = int(evt.get("te", 0))
            telem_status["last_conf"] = conf
            # Memory profile
            telem_status["m_ring"]     = int(evt.get("m_ring", telem_status["m_ring"]))
            telem_status["m_fa"]       = int(evt.get("m_fa",   telem_status["m_fa"]))
            telem_status["m_bs"]       = int(evt.get("m_bs",   telem_status["m_bs"]))
            telem_status["m_misc"]     = int(evt.get("m_misc", telem_status["m_misc"]))
            # High-water mark of total live bytes — drives the auto-ranged MEM
            # axis so baseline allocations are visible during training, not just
            # the brief full-frame peak.
            telem_status["mem_peak"]   = max(
                telem_status["mem_peak"],
                telem_status["m_ring"] + telem_status["m_fa"]
                + telem_status["m_bs"] + telem_status["m_misc"])
            telem_status["last_nr_dev"] = int(evt.get("nr_dev", telem_status["last_nr_dev"]))
            telem_status["rate_est"]   = float(evt.get("rate_est", telem_status["rate_est"]))
            telem_status["sym_count"]  = int(evt.get("sc", telem_status["sym_count"]))
            # EMA smoothing for visual meters (α=0.25 → ~4-symbol time constant)
            a = 0.25
            telem_status["nr_dev_ema"] = (a * telem_status["last_nr_dev"]
                                          + (1 - a) * telem_status["nr_dev_ema"])
            telem_status["conf_ema"]   = (a * abs(conf)
                                          + (1 - a) * telem_status["conf_ema"])

            # ── Overlay-sync check ────────────────────────────────────────────
            # Verify telemetry continuity: sc must increase by exactly 1 each
            # packet.  A jump means UDP dropped/reordered an "s" packet, after
            # which every overlay is mis-indexed against the spectrogram and
            # directions appear reversed.  Latency = recv − send wall clock; a
            # growing/jittery latency also drifts the overlay vs the live edge.
            sc   = int(evt.get("sc", telem_status["sym_count"]))
            wt   = evt.get("wt")          # decoder send wall clock (s), if present
            now  = time.time()
            lat  = (now - float(wt)) if wt is not None else None
            if lat is not None:
                telem_status["lat_ema"] = (0.1 * lat + 0.9 * telem_status["lat_ema"]
                                           if telem_status["n_sym_rx"] else lat)
            last_sc = telem_status["last_sc"]
            telem_status["n_sym_rx"] += 1
            if last_sc >= 0 and sc != last_sc + 1:
                gap = sc - last_sc - 1
                telem_status["n_sym_gap"]  += 1
                telem_status["n_sym_lost"] += max(0, gap)
                print(f"[spec] SYNC GAP: expected sc={last_sc + 1} got {sc} "
                      f"({'lost ' + str(gap) if gap > 0 else 'reorder ' + str(gap)}); "
                      f"overlays now shifted — total gaps={telem_status['n_sym_gap']} "
                      f"lost={telem_status['n_sym_lost']}", flush=True)
            telem_status["last_sc"] = sc
            if log_telem:
                _lat_s = f"{lat * 1000:+.0f}ms" if lat is not None else "n/a"
                print(f"[spec] OVL sc={sc} bit={bit} dir={'UP' if bit else 'DN'} "
                      f"conf={conf:+.2f} place_x={ox} lag={decoder_lag}px "
                      f"edge_col={hist_n} lat={_lat_s}", flush=True)

            overlay_annotations.append({
                "type": "sym", "x": ox, "bit": bit, "conf": conf, "sc": sc,
                "wt": (float(wt) if wt is not None else None), "flagged": False,
            })

        elif et == "sync":
            telem_status["sync"]       = True
            telem_status["sync_score"] = float(evt.get("sc", 0.0))
            # Decoder broadcasts live chirp params with every sync event —
            # update so the overlay stays accurate after an encoder change.
            if "f0" in evt:
                chirp.update({
                    "f_min":      float(evt["f0"]),
                    "f_max":      float(evt["f1"]),
                    "symbol_dur": float(evt["sd"]),
                    "loaded":     True,
                })
            # Modem config for meter axis scaling
            if "sps_eff" in evt:
                telem_status["sps_eff"] = int(evt["sps_eff"])
            if "fr" in evt:
                telem_status["fr"]      = int(evt["fr"])
            overlay_annotations.append({"type": "sync", "x": ox})

        elif et == "frame":
            telem_status["sync"]   = False   # re-arm detection for next frame
            telem_status["frames"] = int(evt.get("n", telem_status["frames"]))
            telem_status["last_seq"] = int(evt.get("seq", telem_status["last_seq"]))
            # fa_peak: bytes freed at this boundary → drives the ghost line
            if "fa_peak" in evt:
                telem_status["fa_peak"] = int(evt["fa_peak"])
            overlay_annotations.append({
                "type": "frame", "x": ox,
                "n":    telem_status["frames"],
            })

    # ── Periodic overlay-sync summary (every 5 s while connected) ──────────
    # One-line health check to cross-reference against the decoder's log: if
    # gaps>0 the overlay is out of sync (dropped telemetry) and reversed-looking
    # symbols are a positioning artifact, not a decode error.  lat is the mean
    # send→display latency (needs a fresh decoder that stamps "wt").
    if telem_status["connected"] and (time.time() - _last_sync_log) >= 5.0:
        _last_sync_log = time.time()
        print(f"[spec] SYNC status: rx={telem_status['n_sym_rx']} "
              f"last_sc={telem_status['last_sc']} gaps={telem_status['n_sym_gap']} "
              f"lost={telem_status['n_sym_lost']} lat~{telem_status['lat_ema']*1000:.0f}ms "
              f"lag={decoder_lag}px", flush=True)

    # ── Audio → spectrogram ───────────────────────────────────────────────
    scroll_width = max(1, int(cfg["scroll"]))

    # Drain all queued chunks and convert each to a full-resolution dB column,
    # storing it in the history ring buffer regardless of live/frozen state.
    pending_audio = []
    while not audio_queue.empty():
        try:
            pending_audio.append(audio_queue.get_nowait())
        except queue.Empty:
            break

    n_new = len(pending_audio)
    if n_new:
        # ── Measure the true audio-column rate ────────────────────────────────
        # One column is produced per delivered chunk, so counting columns over
        # wall-clock gives the real cols/sec = device_rate / chunk_size, whatever
        # the device actually delivers.  Skip the first batch (start-up prefill
        # can arrive as a burst and would skew the cumulative rate) and only
        # trust it after a couple of seconds of steady capture.
        _now_t = time.time()
        if _cap_t0 is None:
            _cap_t0 = _now_t
        else:
            _cap_cols += n_new
            _cap_el    = _now_t - _cap_t0
            if _cap_el > 2.0 and _cap_cols > FPS:
                cols_per_sec   = _cap_cols / _cap_el
                _rate_measured = True

        db_block = np.empty((n_new, N_BINS), dtype=np.float32)
        for i, data in enumerate(pending_audio):
            audio_data = np.frombuffer(data, dtype=np.int16)
            if audio_data.size == 0:
                db_block[i, :] = -120.0
                continue
            db = 20.0 * np.log10(np.abs(np.fft.rfft(audio_data)) + 1e-6)
            if db.size != N_BINS:   # buffer length jitter — resample to N_BINS
                db = np.interp(np.linspace(0.0, 1.0, N_BINS),
                               np.linspace(0.0, 1.0, db.size), db)
            db_block[i, :] = db
            hist[hist_w] = db.astype(np.float16)
            hist_w       = (hist_w + 1) % MAX_COLS
            hist_n      += 1

    if live:
        if resync:
            # Returned from scrollback — rebuild the whole window from history.
            render_window(hist_n - 1)
            resync = False
            overlay_annotations.clear()
        elif n_new:
            # Incremental: scroll left by the rendered width, blit new columns.
            arr   = render_cols(db_block)
            shift = arr.shape[0]
            if shift >= WIDTH:
                spec_surface.fill((0, 0, 0))
                spec_surface.blit(
                    pygame.surfarray.make_surface(arr[-WIDTH:]), (0, 0))
            else:
                spec_surface.scroll(-shift, 0)
                spec_surface.blit(
                    pygame.surfarray.make_surface(arr), (WIDTH - shift, 0))
            # Scroll annotation x-positions in sync with the spectrogram.
            for ann in overlay_annotations:
                ann["x"] -= shift
            prune_edge = -(sym_pixels() + 20)
            overlay_annotations = [a for a in overlay_annotations
                                   if a["x"] > prune_edge]
    else:
        # Frozen scrollback — clamp the anchor (the ring window slides as new
        # audio arrives) and re-render the historical window.
        avail   = min(hist_n, MAX_COLS)
        visible = max(1, WIDTH // scroll_width)
        oldest  = hist_n - avail
        lo_edge = (oldest + visible - 1) if avail >= visible else (hist_n - 1)
        view_right_abs = max(max(0, lo_edge),
                             min(hist_n - 1, view_right_abs))
        render_window(view_right_abs)

    # ── Compose ───────────────────────────────────────────────────────────
    screen.blit(spec_surface, (0, 0))

    if show_timing or show_chirp:
        draw_overlays(screen, show_annotations=live)

    # ── Scrollback position indicator + scrollbar ─────────────────────────
    avail = min(hist_n, MAX_COLS)
    if live:
        tag = font_sm.render("● LIVE", True, (0, 230, 80))
    else:
        back_s = ((hist_n - 1) - view_right_abs) / FPS
        tag = font_sm.render(
            f"⏸ HISTORY  -{back_s:.1f}s / {avail / FPS:.0f}s  "
            f"(wheel/←→ scroll · End=live)",
            True, (255, 200, 60))
    pygame.draw.rect(screen, (0, 0, 0), (6, 2, tag.get_width() + 6, 18))
    screen.blit(tag, (9, 3))
    if avail > 0:
        visible = max(1, WIDTH // scroll_width)
        newest  = hist_n - 1
        oldest  = hist_n - avail
        right   = newest if live else view_right_abs
        left    = right - visible + 1
        sb_y    = SPEC_HEIGHT - 5
        pygame.draw.rect(screen, (40, 40, 48), (0, sb_y, WIDTH, 4))
        fs = max(0.0, min(1.0, (left - oldest) / max(1, avail)))
        fw = max(0.02, min(1.0, visible / max(1, avail)))
        tw = max(6, int(fw * WIDTH))
        tx = min(WIDTH - tw, int(fs * WIDTH))
        pygame.draw.rect(screen, (0, 200, 90) if live else (255, 200, 60),
                         (tx, sb_y, tw, 4))

    # ── UI panel ──────────────────────────────────────────────────────────
    pygame.draw.rect(screen, (30, 30, 30), (0, SPEC_HEIGHT, WIDTH, UI_HEIGHT))

    for s in sliders:
        rect = get_rect(s["row"], s["col"], WIDTH, SPEC_HEIGHT)
        pygame.draw.rect(screen, (100, 100, 100), rect)

        val_pct = (cfg[s["key"]] - s["min"]) / (s["max"] - s["min"])
        knob_x  = rect.left + int(val_pct * rect.width)
        pygame.draw.circle(screen, (200, 200, 255), (knob_x, rect.centery), 8)

        if s["key"] == "color":
            lbl = "Color (R → G → B)"
        else:
            lbl = f"{s['label']}: {int(cfg[s['key']])}"
        screen.blit(font.render(lbl, True, (255, 255, 255)),
                    (rect.left, rect.top - 20))

    # ── Decoder status strip ──────────────────────────────────────────────
    sy  = SPEC_HEIGHT + 135
    sy2 = sy + 18

    # Connection dot + live stats
    dot_col = (0, 230, 80) if telem_status["connected"] else (160, 160, 0)
    pygame.draw.circle(screen, dot_col, (12, sy + 7), 5)

    if telem_status["connected"]:
        seq_txt = (f"  seq={telem_status['last_seq']}"
                   if telem_status["last_seq"] >= 0 else "")
        dec_txt = (
            f"DECODER  sync={'YES' if telem_status['sync'] else 'no '}"
            f"  frames={telem_status['frames']}{seq_txt}"
            f"  syms={telem_status['sym_count']}"
            f"  terr={telem_status['last_terr']:+d}smp"
            f"  conf={telem_status['last_conf']:+.2f}"
        )
    else:
        dec_txt = f"DECODER  waiting on UDP {TELEM_PORT}  (run css_decoder on THIS host — telemetry is built-in)"

    screen.blit(font_sm.render(dec_txt, True, (200, 200, 200)), (24, sy))

    # Toggle labels in state-dependent colour + param/lag info
    xc = 10
    for label, active, col_on in (
        ("[T]", show_timing, (80, 210, 80)),
        ("[O]", show_chirp,  (80, 210, 210)),
    ):
        col  = col_on if active else (90, 90, 90)
        surf = font_sm.render(label, True, col)
        screen.blit(surf, (xc, sy2))
        xc  += surf.get_width() + 2

    _lag_ms = decoder_lag / (cols_per_sec * max(1, int(cfg["scroll"]))) * 1000.0
    rest = (f"  [R]reload  [+/-] lag={decoder_lag:+d}px ({_lag_ms:+.0f}ms)"
            f"  {RATE} Hz · {cols_per_sec:.1f}col/s")
    if chirp["loaded"]:
        rest += (f"   {chirp['f_min']:.0f}–{chirp['f_max']:.0f} Hz"
                 f"  {chirp['symbol_dur']}s/sym")
    screen.blit(font_sm.render(rest, True, (155, 155, 195)), (xc, sy2))

    # ── Decoder meters (memory · PLL · S/N) ──────────────────────────────
    # Always rendered — stale data is dimmed rather than hidden so the bars
    # never flash during long retro FFTs or the inter-frame gap.
    meter_x   = 8
    meter_y   = SPEC_HEIGHT + 178
    meter_w   = WIDTH - 16
    stale_age = (time.time() - telem_status["last_rx"]
                 if telem_status["last_rx"] > 0 else None)
    draw_decoder_meters(screen, meter_x, meter_y, meter_w, stale_age)

    # Mini chirp-direction legend (top-right corner of UI)
    if chirp["loaded"]:
        lx = WIDTH - 80
        ly = sy - 1
        # Up-chirp: line goes from lower-left to upper-right → cyan
        pygame.draw.line(screen, (0, 190, 190), (lx, ly + 13), (lx + 22, ly + 3), 2)
        screen.blit(font_sm.render("up", True, (0, 190, 190)),   (lx + 25, ly + 2))
        # Down-chirp: line goes upper-left to lower-right → magenta
        pygame.draw.line(screen, (190, 0, 190), (lx, ly + 3), (lx + 22, ly + 13), 2)
        screen.blit(font_sm.render("dn", True, (190, 0, 190)), (lx + 25, ly + 10))

    # ── Sample-rate mismatch banner ───────────────────────────────────────
    # We opened at the device's reported native RATE, but the number of columns
    # actually delivered per second implies measured_fs = cols_per_sec*chunk_size.
    # If those disagree, a silent resample / wrong-rate is in play and BOTH axes
    # would be mis-scaled — the user can't otherwise tell this is happening.  The
    # display self-calibrates off cols_per_sec so the overlay still tracks, but
    # flag it loudly so a sampling error/mismatch is never invisible.
    if _rate_measured:
        _measured_fs = cols_per_sec * chunk_size
        if abs(_measured_fs - RATE) > 0.02 * RATE:
            _mm = font.render(
                f"⚠ SAMPLE-RATE MISMATCH: opened {RATE} Hz, "
                f"measuring {_measured_fs:.0f} Hz (display self-calibrated)",
                True, (255, 235, 120))
            mbx = WIDTH // 2 - _mm.get_width() // 2
            pygame.draw.rect(screen, (140, 20, 20),
                             (mbx - 8, 26, _mm.get_width() + 16,
                              _mm.get_height() + 6))
            pygame.draw.rect(screen, (255, 90, 90),
                             (mbx - 8, 26, _mm.get_width() + 16,
                              _mm.get_height() + 6), 1)
            screen.blit(_mm, (mbx, 29))

    # ── Flash message (top-centre of spectrogram) ─────────────────────────
    if flash_msg and (time.time() - flash_t) < 2.5:
        fs  = font.render(flash_msg, True, (255, 255, 80))
        bx  = WIDTH // 2 - fs.get_width() // 2
        pygame.draw.rect(screen, (20, 20, 20),
                         (bx - 5, 6, fs.get_width() + 10, fs.get_height() + 4))
        screen.blit(fs, (bx, 8))

    pygame.display.flip()
    clock.tick(FPS)

# ─────────────────────────────────────────────────────────────────────────────
# Save state and clean up
# ─────────────────────────────────────────────────────────────────────────────
cfg.update({
    "show_timing": show_timing,
    "show_chirp":  show_chirp,
    "decoder_lag": decoder_lag,
    "log_telem":   log_telem,
})
try:
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=4)
except Exception:
    pass

stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()
sys.exit()
