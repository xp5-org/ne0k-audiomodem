# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""
bat_chart.py -- per-carrier bit-allocation chart, rendered to PNG.

Rendered at 1200px+ so a 353-carrier table gets one column per carrier;
below ~2px per carrier adjacent columns merge and a dead bin disappears.

Bar per subcarrier: height = SNR the probe measured there, colour = bits
the allocator handed it. Pairing the two shows why the table looks the way
it does (dead top octave, multipath notch), not just what was allocated.
"""
from __future__ import annotations

import math
from pathlib import Path

# Same palette as the report's legend, so the PNG and the page agree.
BAT_COLORS = {
    -1: (240, 196, 25),     # pilot
    0:  (201, 206, 214),    # off
    1:  (109, 168, 224),    # BPSK
    2:  (47, 127, 208),     # QPSK
    4:  (31, 157, 85),      # 16-QAM
    6:  (142, 68, 173),     # 64-QAM
}
BAT_NAMES = {-1: "pilot", 0: "off", 1: "BPSK", 2: "QPSK", 4: "16-QAM",
             6: "64-QAM"}

MIN_WIDTH = 1200


def _font(px: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=px)
    except Exception:                                    # noqa: BLE001
        try:
            return ImageFont.load_default()
        except Exception:                                # noqa: BLE001
            return None


def render(spec: dict, out: Path, width: int = 1400,
           height: int = 460) -> Path | None:
    """Draw one direction's BAT/SNR chart.  Returns the path, or None."""
    freq = spec.get("freq_hz") or []
    snr = spec.get("snr_db") or []
    bits = spec.get("bits") or []
    if not freq or len(snr) != len(freq):
        return None
    from PIL import Image, ImageDraw

    n = len(freq)
    # One column per carrier, never under a pixel, or a table wider than the
    # plot drops carriers silently.
    width = max(int(width), MIN_WIDTH, n * 2 + 120)
    pl, pr, pt, pb = 62, 18, 34, 54
    plot_w, plot_h = width - pl - pr, height - pt - pb

    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f_lbl, f_ax, f_ttl = _font(12), _font(11), _font(15)

    # Clamp the SNR floor at 0 dB: negative excursions on dead carriers would
    # otherwise squash the useful range into the top third.
    lo = 0.0
    hi = max(6.0, math.ceil(max(snr) / 6.0) * 6.0)

    def x_of(i):
        return pl + (i + 0.5) * (plot_w / n)

    def y_of(v):
        v = max(lo, min(hi, v))
        return pt + plot_h * (1.0 - (v - lo) / (hi - lo))

    d.text((pl, 8), str(spec.get("label") or "BAT"), fill=(20, 20, 20),
           font=f_ttl)

    # y gridlines every 6 dB
    v = lo
    while v <= hi:
        y = y_of(v)
        d.line([pl, y, width - pr, y], fill=(227, 230, 234))
        d.text((pl - 8 - (18 if v >= 10 else 10), y - 6), f"{int(v)}",
               fill=(90, 90, 90), font=f_ax)
        v += 6
    d.text((6, pt + plot_h / 2 - 24), "SNR", fill=(90, 90, 90), font=f_ax)
    d.text((6, pt + plot_h / 2 - 10), "(dB)", fill=(90, 90, 90), font=f_ax)

    bw = max(1.0, plot_w / n - 0.4)
    for i, fr in enumerate(freq):
        b = int(bits[i]) if i < len(bits) else 0
        col = BAT_COLORS.get(b, (201, 206, 214))
        y = y_of(snr[i])
        x = x_of(i) - bw / 2.0
        d.rectangle([x, y, x + bw, pt + plot_h], fill=col)

    d.rectangle([pl, pt, width - pr, pt + plot_h], outline=(150, 155, 162))

    # x labels every ~1 kHz
    step_hz = 1000.0
    nxt = math.ceil(float(freq[0]) / step_hz) * step_hz
    while nxt <= float(freq[-1]):
        i = min(range(n), key=lambda k: abs(float(freq[k]) - nxt))
        x = x_of(i)
        d.line([x, pt + plot_h, x, pt + plot_h + 4], fill=(150, 155, 162))
        d.text((x - 12, pt + plot_h + 7), f"{nxt / 1000:.0f}k",
               fill=(90, 90, 90), font=f_ax)
        nxt += step_hz
    d.text((pl, height - 18), "carrier frequency", fill=(90, 90, 90), font=f_ax)

    # legend: only the orders actually present, with their carrier counts
    present = []
    for b in (-1, 0, 1, 2, 4, 6):
        c = sum(1 for v in bits if int(v) == b)
        if c:
            present.append((b, c))
    lx = pl + 150
    for b, c in present:
        d.rectangle([lx, height - 20, lx + 11, height - 10],
                    fill=BAT_COLORS.get(b, (200, 200, 200)),
                    outline=(120, 120, 120))
        txt = f"{BAT_NAMES.get(b, b)} ({c})"
        d.text((lx + 15, height - 20), txt, fill=(60, 60, 60), font=f_lbl)
        lx += 20 + 7 * len(txt)

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_"
                   for ch in str(s))[:60]


def render_all(run_dir: Path, spectra: list) -> list:
    """Render every direction's chart into the run dir; returns the specs with
    a `png` name attached.  Rebuilt only when missing, so opening a report
    stays cheap."""
    out = []
    for i, spec in enumerate(spectra or []):
        name = f"bat_{i:02d}_{_slug(spec.get('label') or i)}.png"
        p = run_dir / name
        try:
            if not p.exists():
                if render(spec, p) is None:
                    out.append(spec)
                    continue
            out.append({**spec, "png": name})
        except Exception:                                # noqa: BLE001
            out.append(spec)                             # chart is optional
    return out
