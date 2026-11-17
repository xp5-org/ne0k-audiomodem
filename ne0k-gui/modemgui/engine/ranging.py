# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Two-way acoustic ranging with a CSS Barker burst (port of audiotestserver tests/CSS/css_ranging_*).

The ranging waveform is decode-free: a short lead silence plus the Barker-13 preamble of up / down chirps (the
same waveform ``css_ranging_analyze.ranging_burst`` sends, so a GUI node can range against the script).  Barker-13
has an aperiodic, near-sidelobe-free autocorrelation, so its matched-filter peak cannot slip a symbol.

Each node keeps one continuous capture and times everything inside it, so neither node's playback latency nor the
two clocks' offsets enter the answer::

    initiator capture:  req_tx  (its own burst, self-echo)   reply_rx (the responder's burst)
    responder capture:  req_rx  (the initiator's burst)      reply_tx (its own burst, self-echo)

    rtt        = reply_rx - req_tx        (initiator clock)
    turnaround = reply_tx - req_rx        (responder clock)
    distance   = (rtt - turnaround) / 2 * speed_of_sound

The fiducials sit on the DIRECT-path arrival (``leading_edge``): indoors the strongest matched-filter peak is often
a reflection a few ms after the weaker direct path.  What remains:

* each self-echo includes that node's own speaker-to-mic path, so the answer is the distance minus the mean of the
  two nodes' speaker-to-mic spacings (``bias_m`` adds it back);
* the turnaround is counted on the responder's clock: ``ppm`` (responder minus initiator, as CALIBRATE LISTEN on the
  initiator reports for the responder's probe, sign flipped) corrects it.  The turnaround is kept short (burst plus
  ``margin``) so an uncorrected 50 ppm costs only a few cm.

Live protocol (``respond`` / ``initiate``) runs over an AudioPipe-like object (``history``, ``play``, ``wait_tx``,
``spec.rate``): the responder listens for a complete burst in its own capture and replies ``margin`` seconds later,
so no coordination server is needed.  The responder's turnaround still has to reach the initiator's side for
``distance_m``: the control port prints both and ``RANGE SOLVE`` combines them.

Offline: ``python -m modemgui.engine.ranging initiator.wav responder.wav`` (captures saved with ``save=``, or the
script's ``<client>_ranging.wav``).
"""
from __future__ import annotations

import dataclasses
import math
import time
import wave

import numpy as np

SPEED_OF_SOUND_M_S = 343.0          # dry air, ~20 C
BARKER_13 = (1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1)
GAP_FRAC = 0.05                     # silence after each chirp, as a fraction of the symbol
LEAD_S = 0.5                        # silence before the burst
DETECT_DB = 20.0                    # matched-filter peak over the median that counts as a burst
POLL_S = 0.25


@dataclasses.dataclass
class BurstParams:
    f_min: float = 800.0
    f_max: float = 4000.0
    symbol_dur: float = 0.3
    amplitude: float = 0.3

    def check(self) -> None:
        if not 50 <= self.f_min < self.f_max <= 24000:
            raise ValueError(f"bad band {self.f_min:g}-{self.f_max:g} Hz")
        if not 0.02 <= self.symbol_dur <= 1.0:
            raise ValueError(f"symbol_dur {self.symbol_dur:g} s outside 0.02-1")
        if not 0.0 < self.amplitude <= 1.0:
            raise ValueError(f"amplitude {self.amplitude:g} outside 0-1")


# -- waveform --------------------------------------------------------------------------------------------------------

def chirp(f0: float, f1: float, dur: float, fs: int) -> np.ndarray:
    """Linear FM chirp, rectangular envelope (css_encoder.generate_chirp)."""
    n = int(fs * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    return np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / dur * t ** 2))


def preamble(fs: float, p: BurstParams, ratio: float = 1.0) -> np.ndarray:
    """The Barker-13 burst at unit amplitude: the matched-filter template.  ``ratio`` = listener clock / player
    clock: a burst from a sound card running fast arrives compressed, and over a 1-2 s burst even 100 ppm smears
    the peak by several samples, which the leading-edge pick turns into a few cm."""
    up, dn = chirp(p.f_min, p.f_max, p.symbol_dur, fs), chirp(p.f_max, p.f_min, p.symbol_dur, fs)
    gap = np.zeros(int(int(fs * p.symbol_dur) * GAP_FRAC))
    x = np.concatenate([np.concatenate([up if b else dn, gap]) for b in BARKER_13])
    if ratio == 1.0:
        return x
    return np.interp(np.arange(int(len(x) * ratio)) / ratio, np.arange(len(x)), x)


def burst(fs: int, p: BurstParams, lead_s: float = LEAD_S) -> np.ndarray:
    """What is transmitted: lead silence + the preamble, scaled by amplitude."""
    return (np.concatenate([np.zeros(int(lead_s * fs)), preamble(fs, p)]) * p.amplitude).astype(np.float32)


# -- detection -------------------------------------------------------------------------------------------------------

def envelope(cap: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    """Analytic |cross-correlation| of ``tmpl`` against ``cap``; index i = the template starting at cap[i].
    The analytic (one-sided spectrum) envelope has no carrier ripple, so the leading-edge threshold cannot land
    on an earlier half-cycle."""
    cap = np.asarray(cap, np.float64)
    n = 1 << int(math.ceil(math.log2(len(cap) + len(tmpl))))
    c = np.fft.fft(cap, n) * np.conj(np.fft.fft(tmpl, n))
    c[n // 2 + 1:] = 0.0
    c[1:n // 2] *= 2.0
    return np.abs(np.fft.ifft(c))[:len(cap)]


def _frac(a: np.ndarray, k: int) -> float:
    if 0 < k < len(a) - 1:
        y0, y1, y2 = a[k - 1], a[k], a[k + 1]
        d = y0 - 2 * y1 + y2
        if d != 0:
            return k + 0.5 * (y0 - y2) / d
    return float(k)


def leading_edge(env: np.ndarray, p: int, fs: int, back_ms: float = 35.0, fwd_ms: float = 6.0,
                 thr: float = 0.4) -> float:
    """Move a correlation peak ``p`` back to the direct-path arrival: the first sample within ``back_ms`` before it
    that rises above ``thr`` x the local maximum, then the local peak within 3 ms after that edge (sub-sample).
    A clean single peak comes back as itself."""
    a = max(0, p - int(back_ms / 1000 * fs))
    b = min(len(env), p + int(fwd_ms / 1000 * fs) + 1)
    seg = env[a:b]
    edge = int(np.argmax(seg >= thr * float(seg.max())))
    k = edge + int(np.argmax(seg[edge:edge + max(1, int(0.003 * fs))]))
    return a + _frac(seg, k)


@dataclasses.dataclass
class Fiducial:
    pos: float          # sample index of the burst's start (direct path), within the capture given
    score_db: float     # matched-filter peak over the capture's median


def fiducials(cap: np.ndarray, tmpl: np.ndarray, fs: int, n: int = 2, complete: bool = True) -> list[Fiducial]:
    """The ``n`` strongest bursts at least one burst length apart, sorted by time.  ``complete``: leave out a burst
    whose template runs past the end of ``cap``.  Its partial peak is still found (it beats its own symbol-slipped
    sidelobes) and blanks its neighbourhood, so a sidelobe of a burst still arriving is never taken for a burst."""
    env = envelope(cap, tmpl)
    med = float(np.median(env)) + 1e-12
    work = env.copy()
    last = len(cap) - len(tmpl)
    out = []
    for _ in range(n + 1):
        k = int(np.argmax(work))
        if work[k] <= 0 or len(out) == n:
            break
        if not (complete and k > last):
            out.append(Fiducial(leading_edge(env, k, fs), 20 * math.log10(env[k] / med)))
        work[max(0, k - len(tmpl)):k + len(tmpl)] = 0.0
    return sorted(out, key=lambda f: f.pos)


def refine(cap: np.ndarray, f: Fiducial, tmpl: np.ndarray, fs: int, slack_s: float = 0.05) -> Fiducial:
    """Re-time one burst found near ``f.pos`` with another template (the peer's clock-skewed one)."""
    a = max(0, int(f.pos) - int(slack_s * fs))
    seg = cap[a:int(f.pos) + len(tmpl) + int(slack_s * fs)]
    got = fiducials(seg, tmpl, fs, 1, complete=False)
    return Fiducial(a + got[0].pos, f.score_db) if got else f


def distance_m(rtt_s: float, turnaround_s: float, speed: float = SPEED_OF_SOUND_M_S, ppm: float = 0.0,
               bias_m: float = 0.0) -> float:
    """One-way metres from the initiator's round trip and the responder's turnaround (seconds on each node's own
    sample clock).  ``ppm`` = how fast the responder's clock runs against the initiator's.  Negative = a
    mis-detection (the reply was timed before the request)."""
    return (rtt_s - turnaround_s / (1 + ppm * 1e-6)) / 2 * speed + bias_m


# -- offline: both captures ------------------------------------------------------------------------------------------

@dataclasses.dataclass
class RangeResult:
    fs: int
    req_tx: float
    reply_rx: float
    req_rx: float
    reply_tx: float
    scores_db: tuple[float, float, float, float]

    @property
    def rtt_s(self) -> float:
        return (self.reply_rx - self.req_tx) / self.fs

    @property
    def turnaround_s(self) -> float:
        return (self.reply_tx - self.req_rx) / self.fs

    def distance_m(self, **kw) -> float:
        return distance_m(self.rtt_s, self.turnaround_s, **kw)


def analyse(ini_cap: np.ndarray, res_cap: np.ndarray, fs: int, p: BurstParams,
            res_fs: int | None = None, ppm: float = 0.0) -> RangeResult:
    """Post-hoc ranging from the initiator's and the responder's whole captures (css_ranging_analyze).  ``ppm``:
    the responder's clock against the initiator's, when known (pass it to distance_m too)."""
    rfs = res_fs or fs
    a = fiducials(ini_cap, preamble(fs, p), fs)
    b = fiducials(res_cap, preamble(rfs, p), rfs)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("a capture does not hold two bursts")
    if ppm:
        k = 1 + ppm * 1e-6
        a[1] = refine(ini_cap, a[1], preamble(fs, p, 1 / k), fs)       # the reply, sent on the responder's clock
        b[0] = refine(res_cap, b[0], preamble(rfs, p, k), rfs)         # the request, heard by the responder
    # the turnaround is converted to initiator samples so RangeResult keeps one fs
    k = fs / rfs
    return RangeResult(fs, a[0].pos, a[1].pos, b[0].pos * k, b[1].pos * k,
                       (a[0].score_db, a[1].score_db, b[0].score_db, b[1].score_db))


def load_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        fs, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"{path}: {8 * sw}-bit WAV, want 16-bit")
    x = np.frombuffer(raw, np.int16).astype(np.float64) / 32768.0
    return (x.reshape(-1, ch)[:, 0] if ch > 1 else x), fs


def save_wav(path, x: np.ndarray, fs: int) -> None:
    """16-bit mono WAV to a path or a writable binary file object."""
    with wave.open(path if hasattr(path, "write") else str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs)
        w.writeframes((np.clip(np.asarray(x, np.float64), -1, 1) * 32767).astype(np.int16).tobytes())


# -- live protocol over an AudioPipe ---------------------------------------------------------------------------------

class RangingError(RuntimeError):
    pass


def _now(pipe) -> int:
    """Absolute index of the next RX sample."""
    x, i0 = pipe.history(1)
    return i0 + len(x)


def _since(pipe, start: int) -> tuple[np.ndarray, int]:
    end = _now(pipe)
    return pipe.history(max(1, end - start))


@dataclasses.dataclass
class LegResult:
    """One node's half: ``first`` / ``second`` are absolute RX sample positions of the two bursts it heard."""
    role: str
    fs: int
    first: float
    second: float
    scores_db: tuple[float, float]
    capture: np.ndarray

    @property
    def interval_s(self) -> float:
        return (self.second - self.first) / self.fs

    def rows(self, prefix: str = "range") -> list[tuple[str, str]]:
        ini = self.role == "initiator"
        names = ("req_tx", "reply_rx") if ini else ("req_rx", "reply_tx")
        selfs, peer = (self.scores_db[0], self.scores_db[1]) if ini else (self.scores_db[1], self.scores_db[0])
        return [(f"{prefix}.ok", "yes"), (f"{prefix}.role", self.role), (f"{prefix}.fs", str(self.fs)),
                (f"{prefix}.{'rtt_s' if ini else 'turnaround_s'}", f"{self.interval_s:.7f}"),
                (f"{prefix}.{names[0]}", f"{self.first:.2f}"), (f"{prefix}.{names[1]}", f"{self.second:.2f}"),
                (f"{prefix}.self_db", f"{selfs:.1f}"), (f"{prefix}.peer_db", f"{peer:.1f}")]


def _wait_bursts(pipe, start: int, tmpl: np.ndarray, fs: int, want: int, deadline: float,
                 min_db: float, window: int | None = None) -> list[Fiducial] | None:
    """Poll the capture from ``start`` until ``want`` complete bursts above ``min_db`` are in it.  Returns their
    fiducials with ABSOLUTE positions, or None at the deadline.  ``window``: only search the newest samples
    (listening for one burst does not need the whole wait in every FFT)."""
    while True:
        x, i0 = _since(pipe, start)
        if window and len(x) > window:
            x, i0 = x[-window:], i0 + len(x) - window
        if len(x) >= len(tmpl):
            got = [f for f in fiducials(x, tmpl, fs, want) if f.score_db >= min_db]
            if len(got) >= want:
                return [Fiducial(f.pos + i0, f.score_db) for f in got]
        if time.monotonic() >= deadline:
            return None
        time.sleep(POLL_S)


def respond(pipe, p: BurstParams, timeout: float = 60.0, margin: float = 1.0, tail: float = 0.5,
            min_db: float = DETECT_DB, ppm: float = 0.0) -> LegResult:
    """Listen up to ``timeout`` s for the initiator's burst, reply ``margin`` s after it is complete, and time
    both in this node's capture.  ``ppm``: this node's clock against the initiator's, when known."""
    p.check()
    fs = pipe.spec.rate
    tmpl, sig = preamble(fs, p), burst(fs, p)
    start = _now(pipe)
    got = _wait_bursts(pipe, start, tmpl, fs, 1, time.monotonic() + timeout, min_db,
                       window=len(tmpl) + int(2 * POLL_S * fs) + fs)
    if not got:
        raise RangingError(f"no ranging burst heard in {timeout:g} s (same band= / sym= on both hosts?)")
    req = got[0].pos
    time.sleep(margin)
    pipe.play(sig)
    if not pipe.wait_tx(len(sig) / fs + 10):
        raise RangingError("TX did not drain")
    time.sleep(tail)
    seg_start = int(req) - int(0.2 * fs)
    found = _wait_bursts(pipe, seg_start, tmpl, fs, 2, time.monotonic() + 2.0, min_db)
    if not found:
        raise RangingError("did not hear its own reply (speaker muted or mic not hearing it?)")
    a, b = found
    if abs(a.pos - req) > 0.05 * fs:
        raise RangingError(f"request moved {abs(a.pos - req) / fs * 1000:.0f} ms between detection and timing")
    x, i0 = _since(pipe, seg_start)
    if ppm:
        a = _refine_abs(x, i0, a, preamble(fs, p, 1 + ppm * 1e-6), fs)
    return LegResult("responder", fs, a.pos, b.pos, (a.score_db, b.score_db), x)


def initiate(pipe, p: BurstParams, wait: float = 20.0, tail: float = 0.5, min_db: float = DETECT_DB,
             ppm: float = 0.0) -> LegResult:
    """Send the request burst and time it (self-echo) and the responder's reply in this node's capture.
    ``ppm``: the responder's clock against this node's, when known."""
    p.check()
    fs = pipe.spec.rate
    tmpl, sig = preamble(fs, p), burst(fs, p)
    start = _now(pipe) - int(0.2 * fs)
    pipe.play(sig)
    if not pipe.wait_tx(len(sig) / fs + 10):
        raise RangingError("TX did not drain")
    t_end = time.monotonic() + wait
    one = _wait_bursts(pipe, start, tmpl, fs, 1, time.monotonic() + 3.0, min_db)
    if not one:
        raise RangingError("did not hear its own request (speaker muted or mic not hearing it?)")
    found = _wait_bursts(pipe, start, tmpl, fs, 2, t_end, min_db)
    if not found:
        raise RangingError(f"no reply within {wait:g} s (is RANGE RESPOND armed on the other host?)")
    time.sleep(tail)
    x, i0 = _since(pipe, start)
    a, b = found
    if ppm:
        b = _refine_abs(x, i0, b, preamble(fs, p, 1 / (1 + ppm * 1e-6)), fs)
    return LegResult("initiator", fs, a.pos, b.pos, (a.score_db, b.score_db), x)


def _refine_abs(x: np.ndarray, i0: int, f: Fiducial, tmpl: np.ndarray, fs: int) -> Fiducial:
    r = refine(x, Fiducial(f.pos - i0, f.score_db), tmpl, fs)
    return Fiducial(r.pos + i0, r.score_db)


# -- CLI: the offline analyser ---------------------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="two-way CSS ranging from the initiator's and responder's captures")
    ap.add_argument("initiator_wav")
    ap.add_argument("responder_wav")
    ap.add_argument("--f-min", type=float, default=800.0)
    ap.add_argument("--f-max", type=float, default=4000.0)
    ap.add_argument("--symbol-dur", type=float, default=0.3)
    ap.add_argument("--speed", type=float, default=SPEED_OF_SOUND_M_S)
    ap.add_argument("--ppm", type=float, default=0.0, help="responder clock minus initiator clock")
    ap.add_argument("--bias", type=float, default=0.0, help="metres added (mean speaker-to-mic spacing)")
    ap.add_argument("--true-dist", type=float, default=None)
    a = ap.parse_args(argv)
    p = BurstParams(a.f_min, a.f_max, a.symbol_dur)
    ini, fs = load_wav(a.initiator_wav)
    res, rfs = load_wav(a.responder_wav)
    r = analyse(ini, res, fs, p, rfs, ppm=a.ppm)
    d = r.distance_m(speed=a.speed, ppm=a.ppm, bias_m=a.bias)
    print(f"CSS TWO-WAY RANGING  fs={fs}/{rfs}  burst {len(preamble(fs, p)) / fs:.2f} s")
    print(f"  initiator: req_tx={r.req_tx:.1f}  reply_rx={r.reply_rx:.1f}  ({r.scores_db[0]:.0f}/{r.scores_db[1]:.0f} dB)")
    print(f"  responder: req_rx={r.req_rx:.1f}  reply_tx={r.reply_tx:.1f}  ({r.scores_db[2]:.0f}/{r.scores_db[3]:.0f} dB)")
    print(f"  rtt={r.rtt_s * 1000:.3f} ms  turnaround={r.turnaround_s * 1000:.3f} ms")
    print(f"  => distance = {d:.3f} m")
    if a.true_dist is not None:
        print(f"     true     = {a.true_dist:.3f} m   error = {abs(d - a.true_dist) * 100:.1f} cm")
    return 0 if d > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
