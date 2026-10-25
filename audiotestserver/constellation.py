# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""
constellation.py - constellation heatmaps per run phase, from the capture WAV.

The report could say a frame arrived at 72% EVM but not WHY.  A single number
cannot distinguish the failure modes that need completely different responses:

  * radial smear, cloud centred on the reference points  -> noise / low SNR
  * whole constellation ROTATED                          -> uncorrected phase
  * points pulled toward the origin, edges worst         -> clipping / compression
  * a fuzzy ring instead of dots                         -> residual ISI / reverb
  * two overlaid constellations                          -> BAT mismatch, or two
                                                            transmitters at once

A heatmap of every demapped symbol in a run shows which one it is at a glance.

Built post-hoc from the capture WAV plus the run's own result record, so it works
on runs that were already recorded -- including a before/after pair when
something physical changed in the room.

Phases are plotted SEPARATELY because they are different modems: the handshake
probe is known BPSK on every carrier, the RobustLink messages are BPSK spread
over carrier groups, and the DATA frames use the negotiated per-carrier mix.
Pooling them would average away the thing you are looking for.

Server-side only (numpy + PIL).  shared.py imports it lazily.
"""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np

# Own transmissions arrive ~20 dB above anything from a peer, and a node's own
# speaker into its own mic is not a channel measurement.  Anything above this
# is muted before analysis (see _mute_own_tx).
OWN_TX_DBFS = -10.0

GRID = 96          # heatmap bins per axis
SPAN = 1.8         # plot range in reference-constellation units (1.0 = outer pt)


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        fs, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128) / 128.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    return x, fs


def _mute_own_tx(x: np.ndarray, fs: int, tx_windows=None) -> tuple[np.ndarray, float]:
    """Zero the spans where this node was transmitting.

    Uses the node's own TX log when the record has one, and falls back to a
    level gate (own TX is ~20 dB hotter than anything received).  Without this
    every analysis keeps locking onto the node's own probe -- it scores higher
    coherence than the peer's, so it MASKS the received one entirely.
    """
    xm = x.copy()
    muted = 0.0
    if tx_windows:
        for w in tx_windows:
            t = w.get("t")
            if t is None:
                continue
            a = max(0, int((float(t) - 0.3) * fs))
            b = min(len(xm), int((float(t) + float(w.get("dur_s") or 0) + 0.5) * fs))
            if b > a:
                muted += (b - a) / fs
                xm[a:b] = 0.0
        return xm, muted
    nb = max(1, int(0.05 * fs))
    nblk = len(xm) // nb
    if nblk < 2:
        return xm, 0.0
    env = np.max(np.abs(xm[: nblk * nb].reshape(nblk, nb)), axis=1)
    loud = env > 10 ** (OWN_TX_DBFS / 20.0)
    if loud.any():
        # widen so the transducer's ring-out is muted with the burst
        idx = np.flatnonzero(loud)
        for i in idx:
            loud[max(0, i - 10):i + 10] = True
        for i in np.flatnonzero(loud):
            xm[i * nb:(i + 1) * nb] = 0.0
        muted = float(np.count_nonzero(loud)) * nb / fs
    return xm, muted


def _hist(points: np.ndarray) -> tuple[np.ndarray, int]:
    """2-D histogram of equalised symbols on a fixed reference-unit grid."""
    if not len(points):
        return np.zeros((GRID, GRID), dtype=np.int32), 0
    edges = np.linspace(-SPAN, SPAN, GRID + 1)
    h, _, _ = np.histogram2d(np.real(points), np.imag(points),
                             bins=(edges, edges))
    inside = int(np.count_nonzero(
        (np.abs(np.real(points)) <= SPAN) & (np.abs(np.imag(points)) <= SPAN)))
    return h.astype(np.int32), inside


# The overrides meshmodem_node.py installs from PARAMS.  Mirrored here so the
# report rebuilds the geometry the run ACTUALLY FLEW rather than the library
# defaults: mesh_cfg derives its carrier set from HS_BAND merged with these, so
# a run that moved n_fft or cp_ratio produces a table this process cannot
# reproduce unless it applies the same values.  Missing the FFT keys is what
# made an n_fft=2048 run (329 carriers) unreadable against a default-1024
# rebuild, and the DATA phase was skipped for every node in it.
#   PARAMS key -> HS_OVERRIDES key, converter
_GEOM_PARAMS = (
    ("hs_freq_lo",       "freq_lo",          float),
    ("hs_freq_hi",       "freq_hi",          float),
    ("n_fft",            "n_fft",            int),
    ("cp_ratio",         "cp_ratio",         float),
    ("probe_meas_syms",  "probe_meas_syms",  int),
    ("max_bits_per_bin", "max_bits_per_bin", int),
    ("snr_margin_db",    "snr_margin_db",    float),
    ("isi_derate",       "isi_derate",       float),
)


def _run_overrides(rec: dict, params: dict) -> dict:
    """HS_OVERRIDES as this run flew, reconstructed from its manifest params.

    The recorded BAT length stays the arbiter -- these are what the run ASKED
    for, and a run predating a geometry fix flew something else.  The caller
    tries this first and falls back only when the rebuilt carrier count fails
    to match the table that was actually negotiated.
    """
    ov: dict = {}
    for pkey, okey, cast in _GEOM_PARAMS:
        v = params.get(pkey)
        if v is None:
            continue
        try:
            ov[okey] = cast(v)
        except (TypeError, ValueError):
            continue
    # A band recorded on the node record itself beats the params: it is what
    # that direction ended up on after any retrain.
    band = _run_band(rec, params)
    if band:
        ov["freq_lo"], ov["freq_hi"] = band
    return ov


def _geom_desc(ov: dict) -> str:
    bits = []
    if ov.get("freq_lo") is not None and ov.get("freq_hi") is not None:
        bits.append(f"{ov['freq_lo']:.0f}-{ov['freq_hi']:.0f} Hz")
    if ov.get("n_fft"):
        bits.append(f"n_fft {ov['n_fft']}")
    if ov.get("cp_ratio"):
        bits.append(f"cp {ov['cp_ratio']}")
    return ", ".join(bits) or "run params"


def _phase_data(run_dir: Path, node: str, rec: dict, x: np.ndarray, fs: int,
                mesh, hs_mod, params: dict) -> dict | None:
    """DATA-frame constellation, decoded with the table the node negotiated."""
    peers = rec.get("peers") or {}
    bat = rate = None
    # `link` is None once a link tears down, so fall back to `last_link` -- the
    # table the frames were actually sent with.  Without it a run that ended
    # disconnected (exactly the kind worth inspecting) yields no DATA panel.
    for pv in peers.values():
        for key in ("link", "last_link"):
            lk = (pv or {}).get(key)
            if isinstance(lk, dict) and lk.get("rx_bat"):
                bat = np.asarray(lk["rx_bat"], dtype=np.int32)
                rate = int(lk.get("rx_rate") or 0)
                break
        if bat is not None:
            break
    if bat is None or not rate:
        print(f"constellation: {node} has no negotiated table in its record "
              f"(no link and no last_link) -- DATA phase needs the table the "
              f"sender used, and re-deriving it from the probe would only match "
              f"if the allocator still behaves exactly as it did at run time",
              flush=True)
        return None
    # Same rule for the data geometry: drive mesh_cfg from the run's own band
    # (mesh_cfg reads HS_OVERRIDES, which the report process does not set).
    # Pick the geometry that MATCHES THE RECORDED TABLE's length.  The run's
    # params are the first guess, but runs recorded before the mesh_cfg geometry
    # fix transmitted on the raw HS_BAND default even when their params said
    # otherwise -- their stored BAT is the ground truth for what went on air, so
    # decoding them needs the geometry they actually used, not the one they
    # meant to use.
    saved = dict(hs_mod.HS_OVERRIDES or {})
    if params.get("mesh_chunk"):
        mesh.MESH_CHUNK = int(params["mesh_chunk"])
    band = _run_band(rec, params)
    tries = []
    full = _run_overrides(rec, params)
    if full:
        tries.append((full, "run params (" + _geom_desc(full) + ")"))
    if band:
        tries.append(({**saved, "freq_lo": band[0], "freq_hi": band[1]},
                      f"band only ({band[0]:.0f}-{band[1]:.0f} Hz, "
                      f"default FFT geometry)"))
    tries.append(({}, "HS_BAND default (pre-geometry-fix run)"))
    cfg = geom_note = None
    for ov, why in tries:
        hs_mod.HS_OVERRIDES = ov
        c = mesh.mesh_cfg(rate)
        if len(bat) == c["n_data_carriers"]:
            cfg, geom_note = c, why
            break
    if cfg is None:
        hs_mod.HS_OVERRIDES = saved
        print(f"constellation: {node} BAT is {len(bat)} carriers, which matches "
              f"neither the run's params nor the default geometry -- skipping "
              f"the DATA phase rather than demapping with the wrong table",
              flush=True)
        return None
    if "default" in geom_note:
        print(f"constellation: {node} DATA geometry from {geom_note}", flush=True)
    frame_pts: list = []
    # Plot ONLY frames the node itself logged receiving.
    #
    # A blind envelope scan finds every burst in the capture, but "a burst the
    # acquisition search could latch onto" is not the same thing as "a frame".
    # The gate is coherence 0.60, so handshake traffic (a different modem
    # entirely) and its reverb clear it, get force-demapped with the DATA
    # table, and produce swirling half-rotated garbage -- the mesh demod
    # re-estimates the common phase error PER SYMBOL from pilot carriers, and
    # on a non-frame those pilots are noise, so every symbol is corrected by a
    # different wrong angle.  Those panels are indistinguishable at a glance
    # from a real frame that arrived badly, which is worse than not plotting
    # them.
    #
    # The rx log is the receiver's OWN record, in the same clock as its own
    # capture, of frames it actually took delivery of.  Anchor on it: no log
    # entry, no panel.
    rx_log = [e for e in (rec.get("rx") or [])
              if isinstance(e, dict) and e.get("t") is not None]
    if not rx_log:
        print(f"constellation: {node} logged no received frames -- nothing to "
              f"correlate against, so no DATA panels (a burst that merely "
              f"acquires is not a frame)", flush=True)
        hs_mod.HS_OVERRIDES = saved
        return None
    # The log timestamp sits near the END of the frame (the peer put a 2.2s
    # DATA burst on air at 113.3s; the receiver logged it at 115.7s), so the
    # window has to LEAD it generously.  It also has to TRAIL it: the demod
    # reads a whole frame body forward from the sync it finds, and a window
    # that stops too soon truncates the body -- which does not fail cleanly,
    # it makes acquisition settle on something else and report a bogus 77%
    # EVM instead of the frame's real 9.2%.  Measured on this rig, trail=2s
    # recovered 2 of 4 logged frames and misread one of them; trail>=6s
    # recovers 4 of 4 with every offline EVM matching the live log exactly.
    lead, trail = 14.0, 8.0
    seen_at: list[float] = []
    for e in rx_log:
        t0 = max(0.0, float(e["t"]) - lead)
        t1 = min(len(x) / fs, float(e["t"]) + trail)
        if t1 - t0 < 1.0:
            continue
        seg = x[int(t0 * fs):int(t1 * fs)]
        if not np.any(seg):
            continue
        dem = mesh.MeshDemodulator(cfg, bat)
        dem.collect_points = True
        got, st = dem.decode_packet(
            mesh._resample_for(seg, fs, rate) if hasattr(mesh, "_resample_for")
            else _to_rate(seg, fs, rate))
        if not st or (got is None and not st.get("rs_failed")):
            print(f"constellation: {node} could not re-decode the "
                  f"{e.get('type', '?')} logged at t={e['t']}s "
                  f"(live EVM {e.get('evm_pct')}%)", flush=True)
            continue
        at = st.get("at_sample")
        key = round(float(at or 0) / max(rate, 1) + t0, 1)
        if key in seen_at:                    # one frame, two log entries
            continue
        seen_at.append(key)
        ev = st.get("evm_pct")
        frame_pts.append({
            "t": key,
            "pts": list(dem.points),
            "evm": float(ev) if ev is not None else None,
            "decoded": got is not None,
            "coh": st.get("acq_coh"),
            # Identity straight from the log, so a panel names the frame the
            # node reported rather than a position in a scan.
            "kind": str(e.get("type") or "frame"),
            "seq": e.get("seq"),
            "log_t": e.get("t"),
            "log_evm": e.get("evm_pct"),
        })
    hs_mod.HS_OVERRIDES = saved
    if not frame_pts:
        return None
    # One panel PER decoded frame PER modulation order present in it -- not
    # pooled across the run's frames. Pooling would hide whether quality
    # holds steady exchange to exchange or one round mid-run degraded while
    # its neighbours stayed clean.
    frame_pts.sort(key=lambda f: f["t"])
    out = []
    for f in frame_pts:
        t, pts, ev = f["t"], f["pts"], f["evm"]
        if not pts:
            continue
        # Name the frame the way the node's own log named it, so a panel can be
        # traced back to a line in the record rather than to a scan position.
        who = f["kind"]
        if f.get("seq") is not None:
            who += f" seq {f['seq']}"
        ord_name = _ORDER_NAME
        orders = sorted({int(p[2]) for p in pts})
        for order in orders:
            sel = [p for p in pts if int(p[2]) == order]
            eq = np.array([p[0] for p in sel])
            if not len(eq):
                continue
            h, inside = _hist(eq)
            refs = np.array([p[1] for p in sel])
            gains = np.array([p[3] if len(p) > 3 else 1.0 for p in sel])
            ev_w, ev_u = _order_evm(eq, refs, gains)
            coh = f["coh"]
            detail = f"{who} @ {t:.1f}s, {len(eq)} symbols"
            if ev is not None:
                detail += f", frame EVM {round(ev, 1)}%"
            if f.get("log_evm") is not None:
                detail += f" (live log {f['log_evm']}%)"
            if not f["decoded"]:
                detail += " — logged by the node but RS failed on re-decode"
                if coh is not None:
                    detail += f", acq coherence {coh}"
            out.append({
                "label": f"{who} {ord_name.get(order, order)} @ {t:.1f}s",
                "hist": h, "n": len(eq), "inside": inside, "frames": 1,
                "orders": [order],
                "t": round(float(t), 1),
                "short": f"{who} · {ord_name.get(order, order)}",
                "decoded": f["decoded"],
                "acq_coh": coh,
                "log_t": f.get("log_t"),
                "log_evm": f.get("log_evm"),
                "evm_pct": ev_w,
                "evm_unweighted_pct": ev_u,
                "detail": detail,
            })
    return out


_ORDER_NAME = {1: "BPSK", 2: "QPSK", 4: "16-QAM", 6: "64-QAM"}


def _order_evm(eq: np.ndarray, refs: np.ndarray,
               gains: np.ndarray) -> tuple[float, float]:
    """EVM for one modulation order, energy-weighted and unweighted.

    The plotted points are equalised (Y/He), and that division scales each
    carrier's NOISE by 1/|He| as well as its signal.  A carrier sitting in a
    deep null therefore lands arbitrarily far from its reference no matter how
    little absolute noise it collected, so a plain mean of |eq-ref|^2 is set by
    whichever handful of carriers were most deeply faded -- measured on a real
    run, the worst 1% of points carried ~35% of the total error energy and the
    weakest gain-decile read 249% against 74% for the strongest.  That is what
    produced the >100% BPSK numbers: not a 100%-error channel, but an unbounded
    statistic.  It is also unstable, because the extreme tail moves frame to
    frame.

    Weighting each point by |He|^2 undoes exactly that amplification --
    |He|*(eq-ref) == Y - He*ref -- so the weighted figure is the error the
    demodulator actually saw, and pooling every order reproduces its own frame
    EVM.  It is the headline number.  The unweighted one is kept alongside it
    because "every symbol counts once" is still the right question when asking
    how many carriers are unusable, and returned so the caption can show both.
    """
    err2 = np.abs(eq - refs) ** 2
    ref2 = np.abs(refs) ** 2
    w = gains ** 2
    num_w, den_w = float(np.sum(w * err2)), float(np.sum(w * ref2))
    ev_w = 100 * math.sqrt(num_w / den_w) if den_w > 0 else None
    den_u = float(np.mean(ref2))
    ev_u = (100 * math.sqrt(float(np.mean(err2)) / den_u) if den_u > 0 else None)
    return (None if ev_w is None else round(ev_w, 1),
            None if ev_u is None else round(ev_u, 1))


def _to_rate(seg: np.ndarray, fs: int, rate: int) -> np.ndarray:
    from ofdm_handshake import _resample
    return np.asarray(_resample(seg, fs, rate), dtype=np.float64)


def _run_band(rec: dict, params: dict):
    """The band this RUN used.  Must not be left to the server's own
    HS_OVERRIDES, which are empty in the report process -- that silently analysed
    a 300-9000 Hz run at the 400-4000 Hz default (83 carriers instead of ~200),
    the same class of geometry mismatch that once broke the BAT itself."""
    b = rec.get("band_hz")
    if isinstance(b, (list, tuple)) and len(b) == 2:
        return float(b[0]), float(b[1])
    for lo_k, hi_k in (("hs_freq_lo", "hs_freq_hi"), ("freq_lo", "freq_hi")):
        lo, hi = params.get(lo_k), params.get(hi_k)
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return float(lo), float(hi)
    return None


def _phase_probe(node: str, rec: dict, x: np.ndarray, fs: int,
                 hs_mod, params: dict) -> dict | None:
    """Handshake-probe constellation: known BPSK on every occupied carrier.

    Every probe symbol is the SAME known pattern, so Y/(H*known) collapses to
    the BPSK alphabet and the spread is the per-symbol channel variation the
    coherent average hides.
    """
    from ofdm_modem import ChannelProbe
    # hs_cfg reads n_fft/cp_ratio out of HS_OVERRIDES, which is EMPTY in the
    # report process -- passing only the band would rebuild the probe at the
    # library's default FFT size and fail to acquire anything a run that moved
    # n_fft transmitted.  Install the run's own geometry for the decode.
    saved = dict(hs_mod.HS_OVERRIDES or {})
    hs_mod.HS_OVERRIDES = {**saved, **_run_overrides(rec, params)}
    try:
        return _probe_scan(node, rec, x, fs, hs_mod, params, ChannelProbe)
    finally:
        hs_mod.HS_OVERRIDES = saved


def _probe_scan(node: str, rec: dict, x: np.ndarray, fs: int,
                hs_mod, params: dict, ChannelProbe) -> dict | None:
    cfg = hs_mod.hs_cfg(44100, _run_band(rec, params))
    best = None
    t = 0.0
    while t + 16.0 <= len(x) / fs:
        seg = x[int(t * fs):int((t + 16) * fs)]
        if np.any(seg):
            cp = ChannelProbe(cfg)
            r = cp.estimate(_to_rate(seg, fs, cfg["sample_rate"]))
            if r.get("valid") and (best is None or r["n_syms"] > best[1]["n_syms"]):
                best = (t, r, cp, seg)
        t += 4.0
    if best is None:
        return None
    t, res, cp, seg = best
    rx = _to_rate(seg, fs, cfg["sample_rate"])
    H = cp._probe_H(rx, res["frame_start"], min(res["n_syms"], 64))
    if H is None:
        return None
    Hm = res["H"]
    ok = np.abs(Hm) > 1e-12
    # _probe_H already divided the known symbol out, so H/Hm alone would put
    # every point at +1 -- a residual plot, not a constellation.  Multiplying the
    # known BPSK pattern back in restores the real +-1 alphabet, which is what
    # the red reference marks show.
    known = cp.probe_known[ok]
    eq = ((H[:, ok] / Hm[ok]) * known).ravel()
    h, inside = _hist(eq)
    return {"label": "handshake probe", "hist": h, "n": len(eq),
            "inside": inside, "frames": 1, "orders": [1],
            # t/short place this panel on the run-level call-flow timeline
            # (render_timeline).  t is the probe's ACTUAL start in the capture,
            # not the start of the 16s window that happened to contain it --
            # the scan steps in 4s, so the window start ran up to 16s early and
            # put the probe's arrival BEFORE the peer's own log of sending it.
            "t": round(float(t) + float(res.get("frame_start") or 0)
                       / max(int(cfg["sample_rate"]), 1), 1),
            "short": "handshake probe",
            "evm_pct": None,
            "detail": f"{res['n_syms']} symbols, coherence "
                      f"{res.get('coherence')}, {int(ok.sum())} carriers"}


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def _panel(draw, ox: int, oy: int, size: int, ph: dict, font) -> None:
    """One heatmap panel, drawn into an existing image.

    Bare plot only -- no label or EVM burned into the pixels.  Each panel is
    one cell of the call-flow table, which carries all of that as HTML text in
    its own column, where it stays selectable and legible at any zoom.
    """
    h = ph["hist"].astype(np.float64)
    # log scale: a constellation's centres hold orders of magnitude more hits
    # than the tails, and the tails are the interesting part
    v = np.log1p(h)
    v = v / max(v.max(), 1e-9)
    for iy in range(GRID):
        for ix in range(GRID):
            a = v[ix, GRID - 1 - iy]
            if a <= 0.004:
                continue
            # dark blue -> cyan -> yellow -> white
            if a < 0.4:
                f = a / 0.4
                col = (int(20 + 20 * f), int(30 + 120 * f), int(90 + 130 * f))
            elif a < 0.75:
                f = (a - 0.4) / 0.35
                col = (int(40 + 200 * f), int(150 + 90 * f), int(220 - 130 * f))
            else:
                f = (a - 0.75) / 0.25
                col = (255, int(240 + 15 * f), int(90 + 165 * f))
            px = ox + int(ix * size / GRID)
            py = oy + int(iy * size / GRID)
            w = max(1, size // GRID)
            draw.rectangle([px, py, px + w, py + w], fill=col)
    # axes + unit circle marks
    mid = size // 2
    draw.line([ox, oy + mid, ox + size, oy + mid], fill=(70, 80, 95))
    draw.line([ox + mid, oy, ox + mid, oy + size], fill=(70, 80, 95))
    draw.rectangle([ox, oy, ox + size, oy + size], outline=(90, 100, 115))
    # reference points for the orders present
    for order in ph.get("orders") or []:
        for pt in _ref_points(order):
            cx = ox + int((np.real(pt) + SPAN) / (2 * SPAN) * size)
            cy = oy + int((SPAN - np.imag(pt)) / (2 * SPAN) * size)
            draw.line([cx - 3, cy, cx + 3, cy], fill=(255, 80, 80))
            draw.line([cx, cy - 3, cx, cy + 3], fill=(255, 80, 80))


def _ref_points(order: int) -> np.ndarray:
    from ofdm_modem import _MOD_TABLES
    tab = _MOD_TABLES.get(order)
    return tab[0] if tab is not None else np.array([])


def _font(px: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=px)
    except Exception:                                    # noqa: BLE001
        try:
            return ImageFont.load_default()
        except Exception:                                # noqa: BLE001
            return None


CELL_PX = 230          # one constellation cell in the call-flow table


def render_timeline(run_dir: Path, per_node: list) -> dict | None:
    """Run-level call flow: one row per moment, one column per node.

    Each node's composite answers "how did THIS node's reception look", which
    is the wrong axis for a two-way exchange -- the question is what the two
    ends saw of the SAME moment.  So this emits one SMALL PNG per cell plus a
    JSON row list, and the report lays them out as an HTML table (rather than
    baking a table into one giant image, which cannot be zoomed, linked, or
    read on a phone).

    Every panel lands in the column of the node that HEARD it, leaving the
    sender's column empty on that row: an ack/data/ack exchange then reads as
    a zig-zag down the page, and a moment where one direction degraded while
    the other stayed clean shows up as a pair.

    Times come from each node's own capture, which start independently, so
    treat the ordering as the call flow rather than a synchronised clock.
    """
    if not per_node:
        return None
    from PIL import Image, ImageDraw
    nodes = [entry[0] for entry in per_node]
    rows = []
    for node, phases, _tx in per_node:
        for ph in phases:
            if ph.get("hist") is not None:
                rows.append((float(ph.get("t") or 0.0), node, ph))
    if not rows:
        return None
    rows.sort(key=lambda r: (r[0], r[1]))

    # Drop the previous build's cells first: a rebuild after a code change can
    # produce a different number of them, and orphans would keep being served.
    # The per-node composites are swept with them -- one giant image of every
    # panel is what this table replaces, and runs built before it still have
    # theirs on disk.
    for old in (*run_dir.glob("cell_*.png"),
                *run_dir.glob("*_constellation.png")):
        try:
            old.unlink()
        except OSError:
            pass

    font = _font(11)
    # Render the plots first, keyed by their position in the sorted list, then
    # merge them with the SEND side into one chronological list.
    plots = {}
    for i, (t, rx, ph) in enumerate(rows):
        img = Image.new("RGB", (CELL_PX, CELL_PX), (16, 20, 28))
        _panel(ImageDraw.Draw(img), 0, 0, CELL_PX - 1, ph, font)
        name = f"cell_{i:03d}_{rx}.png"
        img.save(run_dir / name)
        plots[i] = name

    # ---- the SEND side -----------------------------------------------------
    # Every transmission each node logged, including the RobustLink control
    # frames of the handshake (BAT_REPORT / BAT_ACK / GO).  Those carry the bit
    # table itself and are a different modem from the mesh DATA frames, so
    # there is no constellation to draw for them -- but leaving them out made
    # the flow jump from "handshake" straight to "DATA seq 1" with the
    # negotiation that produced the table invisible.
    #
    # A transmission and its reception are listed SEPARATELY, each from the log
    # of the node that recorded it, because the two captures start
    # independently: pairing them would mean guessing a clock offset, and a
    # wrong guess silently reattributes a frame.  A send with no matching
    # receive is then also visible as what it is -- something that went out and
    # was not decoded.
    merged: list = []
    for i, (t, rx, ph) in enumerate(rows):
        merged.append({"t": float(t), "side": "rx", "node": rx,
                       "ph": ph, "img": plots[i]})
    for node, _phases, txw in per_node:
        for w in txw:
            if not isinstance(w, dict) or w.get("t") is None:
                continue
            merged.append({"t": float(w["t"]), "side": "tx", "node": node,
                           "what": str(w.get("what") or "tx"),
                           "dur_s": w.get("dur_s")})
    merged.sort(key=lambda r: (r["t"], 0 if r["side"] == "tx" else 1))

    out_rows = []
    # Event numbering: rows that share a timestamp AND a receiver are the same
    # transmission seen through different modulations -- bit-loading spreads
    # BPSK/QPSK/16-QAM across the carriers of ONE frame, and they are plotted
    # apart only so the alphabets do not smear together.  Without a shared id
    # the reader has to notice the duplicate timestamps to work out which rows
    # belong to each other.
    ev_no = 0
    prev_key = None
    for m in merged:
        others = [n for n in nodes if n != m["node"]]
        peer = others[0] if len(others) == 1 else None
        if m["side"] == "tx":
            ev_no += 1
            prev_key = None
            out_rows.append({
                "t": round(m["t"], 1),
                "ev": ev_no,
                "side": "tx",
                "tx": m["node"],
                "rx": None,
                "peer": peer,
                "event": m["what"],
                "mod": None,
                "label": m["what"],
                "dur_s": m["dur_s"],
                "n": None,
                "img": None,
            })
            continue
        ph, rx = m["ph"], m["node"]
        key = (round(m["t"], 1), rx)
        if key != prev_key:
            ev_no += 1
            prev_key = key
        short = ph.get("short") or ph.get("label") or ""
        out_rows.append({
            "t": round(m["t"], 1),
            "ev": ev_no,
            "side": "rx",
            "tx": peer,
            "rx": rx,
            "peer": peer,
            # The event this row belongs to, without the per-modulation
            # suffix: every row of one frame carries the identical string.
            "event": short.split(" · ")[0],
            "mod": (short.split(" · ")[1] if " · " in short else None),
            "label": short,
            # False acquisitions are plotted on purpose, but they are not
            # frames the link delivered -- the table must say so.
            "decoded": ph.get("decoded", True),
            "acq_coh": ph.get("acq_coh"),
            "log_t": ph.get("log_t"),
            "log_evm": ph.get("log_evm"),
            "evm_pct": ph.get("evm_pct"),
            "evm_unweighted_pct": ph.get("evm_unweighted_pct"),
            "n": ph.get("n"),
            "inside": ph.get("inside"),
            "img": m["img"],
        })
    doc = {"nodes": nodes, "rows": out_rows, "cell_px": CELL_PX}
    (run_dir / "constellation_timeline.json").write_text(json.dumps(doc))
    return doc


def build(run_dir: Path, node: str, rec: dict, wav: Path,
          params: dict | None = None,
          sink: list | None = None) -> dict | None:
    """Analyse one node's capture and write its constellation composite.

    Returns {'png': name, 'phases': [...]} or None when nothing could be built.
    The returned phases are JSON-safe (no histograms).  Pass `sink` to also
    collect (node, phases-with-histograms) for the run-level call-flow view --
    render_timeline needs the histograms, and re-decoding to get them back
    would cost another ~11s per node.
    """
    import sys
    for p in ("tests/MESH", "tests/OFDM", "tests"):
        d = str((Path(__file__).parent / p).resolve())
        if d not in sys.path:
            sys.path.insert(0, d)
    import meshlib
    import ofdm_handshake as hs_mod

    params = params or {}
    x, fs = _read_mono(wav)
    tx_windows = ((rec.get("capture") or {}).get("tx_windows")
                  if isinstance(rec.get("capture"), dict) else None)
    xm, muted = _mute_own_tx(x, fs, tx_windows)

    import io, contextlib
    phases = []
    diag: list[str] = []
    for fn in (_phase_probe_wrapper, _phase_data_wrapper):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                ph = fn(run_dir, node, rec, xm, fs, meshlib, hs_mod, params)
        except Exception as e:                            # noqa: BLE001
            ph = None
            buf.write(f"constellation: {node} phase failed: "
                       f"{type(e).__name__}: {e}\n")
        got_any = any(one and one.get("n") for one in
                       (ph if isinstance(ph, list) else [ph]))
        # The per-phase decoders print() their skip/failure reasoning rather
        # than returning it, since they're also called standalone from a shell.
        # Surface it through the notes path.
        #
        # The DATA phase is always worth reporting: now that it correlates
        # against the rx log, its output is one line per LOGGED frame it could
        # not reproduce -- which stays interesting even when the other frames
        # came out fine, and is silently lost if reporting waits for the phase
        # to fail entirely.  The probe phase is the noisy one (ChannelProbe
        # traces every acquisition attempt), so it still only reports when it
        # produced nothing.
        if not got_any or fn is _phase_data_wrapper:
            diag.extend(ln for ln in buf.getvalue().splitlines() if ln.strip())
        for one in (ph if isinstance(ph, list) else [ph]):
            if one and one.get("n"):
                phases.append(one)
    if not phases:
        return {"phases": [], "notes": diag} if diag else None
    if sink is not None:
        # The node's own TX log rides along: the call-flow table lists what
        # each node SENT as well as what it decoded, and the handshake control
        # frames (BAT_REPORT/BAT_ACK/GO) exist only there -- they are
        # RobustLink, not mesh DATA, so no constellation can be plotted for
        # them and without this they were simply missing from the flow.
        sink.append((node, phases, tx_windows or []))
    return {"muted_s": round(muted, 1),
            "phases": [{k: v for k, v in p.items() if k != "hist"}
                       for p in phases],
            "notes": diag}


def _phase_probe_wrapper(run_dir, node, rec, x, fs, mesh, hs_mod, params):
    return _phase_probe(node, rec, x, fs, hs_mod, params)


def _phase_data_wrapper(run_dir, node, rec, x, fs, mesh, hs_mod, params):
    return _phase_data(run_dir, node, rec, x, fs, mesh, hs_mod, params)
