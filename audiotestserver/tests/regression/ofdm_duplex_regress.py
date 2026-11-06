# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Full-duplex OFDM engine in C: two ends talking at the same time over one shared channel.

    python tests/regression/ofdm_duplex_regress.py [-v] [--seconds N]

Each end is the complete C stack -- ofdm_dx (codec) + ofdm_rxf (own-band
rejection) + ofdm_rxs (burst gate / decode) + ofdm_arq (stop-and-wait with
piggybacked ACK) -- and the only Python is the sound-card stand-in: it mixes what
each end plays into what the other hears, at a realistic imbalance (an end's own
speaker is far hotter in its mic than the far end), and moves samples in blocks.

What it grades:
  rxf        the designed filter rejects the near band by the requested amount and
             leaves the far band alone
  ping-pong  one frame each way, simultaneously, and ACKed
  stream     many frames each way at once, delivered exactly, in order, once
  loss       the same with random bursts of the channel killed, so the ARQ has to
             retransmit; still exact and in order
  simultaneous  both ends start a frame at the same instant: decodes with the filter,
             and must NOT without it -- a check that the filter is what makes it work,
             not an easy channel
Needs numpy and the project venv.  Exit 0 = pass.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CDIR = ROOT / "OFDM" / "c"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "OFDM"))
sys.path.insert(0, str(HERE))

import ofdm_fdd as F                                            # noqa: E402
import ofdm_handshake as H                                      # noqa: E402
import ofdm_dx_regress as DX                                    # noqa: E402

D, I = ctypes.c_double, ctypes.c_int
FS = DX.FS
BLOCK = 1024


class RxsParams(ctypes.Structure):
    _fields_ = [("fs", I), ("thr_rms", D), ("hang_s", D), ("ring_samples", ctypes.c_long),
                ("taps", ctypes.POINTER(D)), ("ntaps", I)]


class Evt(ctypes.Structure):
    _fields_ = [("kind", I), ("a", ctypes.c_long), ("b", ctypes.c_long), ("seq", I), ("ack", I),
                ("len", I), ("n_chunks", I), ("n_found", I), ("n_bad", I), ("errata", I),
                ("evm", D), ("data", ctypes.c_ubyte * 255)]


class Act(ctypes.Structure):
    _fields_ = [("kind", I), ("seq", I), ("ack", I), ("len", I), ("ok", I), ("retry", I),
                ("data", ctypes.POINTER(ctypes.c_ubyte))]


def build_lib(tmp: Path) -> ctypes.CDLL:
    lib = DX.build_lib(tmp, extra=("ofdm_arq.c", "ofdm_rxf.c", "ofdm_rxs.c"), opt="-O2")
    lib.ofdm_rxf_design.argtypes = [ctypes.POINTER(D), I, I, D, D, D, D, D, ctypes.POINTER(D)]
    lib.ofdm_rxs_workspace.argtypes = [ctypes.POINTER(RxsParams)]
    lib.ofdm_rxs_workspace.restype = ctypes.c_size_t
    lib.ofdm_rxs_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_void_p, ctypes.POINTER(RxsParams)]
    lib.ofdm_rxs_feed.argtypes = [ctypes.c_void_p, ctypes.POINTER(D), ctypes.c_long]
    lib.ofdm_rxs_pop.argtypes = [ctypes.c_void_p, ctypes.POINTER(Evt)]
    lib.ofdm_rxs_busy.argtypes = [ctypes.c_void_p]
    lib.ofdm_arq_sizeof.restype = ctypes.c_size_t
    lib.ofdm_arq_init.argtypes = [ctypes.c_void_p, I, D, D]
    lib.ofdm_arq_air_busy.argtypes = [ctypes.c_void_p, I, D]
    lib.ofdm_arq_send.argtypes = [ctypes.c_void_p, ctypes.c_char_p, I]
    lib.ofdm_arq_on_frame.argtypes = [ctypes.c_void_p, I, I, ctypes.c_char_p, I, D]
    lib.ofdm_arq_tx_done.argtypes = [ctypes.c_void_p, D]
    lib.ofdm_arq_poll.argtypes = [ctypes.c_void_p, D, ctypes.POINTER(Act)]
    lib.ofdm_arq_busy.argtypes = [ctypes.c_void_p]
    return lib


def design_taps(lib, far, mine, stop_db=70.0):
    cap = 1025
    t = (D * cap)()
    got = D()
    n = lib.ofdm_rxf_design(t, cap, FS, far[0], far[1], mine[0], mine[1], stop_db, ctypes.byref(got))
    return (np.array(t[:n]) if n > 0 else np.zeros(0)), got.value


class Node:
    """The whole C stack for one end, plus the host glue the real engine would own."""

    def __init__(self, lib, name, tx_band, rx_band, tx_bat, rx_bat, taps, thr, hang_s=0.5,
                 retries=4, ack_wait_s=2.5, gap_s=0.7):
        self.lib, self.name = lib, name
        self.end = DX.End(lib, tx_band, rx_band, tx_bat, rx_bat)
        self._taps = (D * max(len(taps), 1))(*taps)
        prm = RxsParams(FS, thr, hang_s, FS * 8,
                        ctypes.cast(self._taps, ctypes.POINTER(D)) if len(taps) else None, len(taps))
        need = lib.ofdm_rxs_workspace(ctypes.byref(prm))
        assert need > 0
        self._rmem = ctypes.create_string_buffer(need)
        self.rxs = ctypes.c_void_p()
        rc = lib.ofdm_rxs_init(ctypes.byref(self.rxs), self._rmem, need, self.end.h, ctypes.byref(prm))
        assert rc == 0, rc
        self.arq = ctypes.create_string_buffer(lib.ofdm_arq_sizeof())
        lib.ofdm_arq_init(self.arq, retries, ack_wait_s, gap_s)
        self.segs: list = []              # (start sample, audio) still to play
        self.tx_end = 0                   # sample index when the last queued audio finishes
        self.pending_done = None
        self.delivered, self.sent_ok, self.sent_fail = [], 0, 0
        self.tx_count = self.retry_tx = self.err_evts = self.frame_evts = 0
        self.todo: list[bytes] = []

    def play_block(self, t0: int) -> np.ndarray:
        """The next BLOCK samples of what this end's speaker plays."""
        out = np.zeros(BLOCK)
        for start, sig in self.segs:
            lo, hi = max(start, t0), min(start + len(sig), t0 + BLOCK)
            if lo < hi:
                out[lo - t0:hi - t0] += sig[lo - start:hi - start]
        self.segs = [s for s in self.segs if s[0] + len(s[1]) > t0 + BLOCK]
        return out

    def hear(self, mic: np.ndarray, now_s: float):
        x = np.ascontiguousarray(mic, dtype=np.float64)
        self.lib.ofdm_rxs_feed(self.rxs, x.ctypes.data_as(ctypes.POINTER(D)), len(x))
        ev = Evt()
        while self.lib.ofdm_rxs_pop(self.rxs, ctypes.byref(ev)):
            if ev.kind == 1:
                self.frame_evts += 1
                self.lib.ofdm_arq_on_frame(self.arq, ev.seq, ev.ack, bytes(ev.data[:ev.len]),
                                           ev.len, now_s)
            else:
                self.err_evts += 1

    def drive(self, t_samples: int, level: float):
        """Poll the ARQ: queue TX audio, deliver data, finish sends."""
        now = t_samples / FS
        self.lib.ofdm_arq_air_busy(self.arq, self.lib.ofdm_rxs_busy(self.rxs), now)
        if self.pending_done is not None and t_samples >= self.pending_done:
            self.lib.ofdm_arq_tx_done(self.arq, now)
            self.pending_done = None
        if self.todo and not self.lib.ofdm_arq_busy(self.arq):
            d = self.todo.pop(0)
            assert self.lib.ofdm_arq_send(self.arq, d, len(d)) == 0
        act = Act()
        while self.lib.ofdm_arq_poll(self.arq, now, ctypes.byref(act)):
            if act.kind == 1:                                        # TX
                data = bytes(act.data[:act.len]) if act.len else b""
                sig = self.end.tx(act.seq, act.ack, data)
                rms = np.sqrt(np.mean(sig[sig != 0] ** 2)) or 1.0
                sig = sig * (level / rms)
                start = max(t_samples, self.tx_end)
                self.segs.append((start, sig))
                self.tx_end = start + len(sig)
                self.pending_done = self.tx_end
                self.tx_count += 1
                self.retry_tx += 1 if act.retry else 0
            elif act.kind == 2:
                self.delivered.append(bytes(act.data[:act.len]))
            elif act.kind == 3:
                if act.ok:
                    self.sent_ok += 1
                else:
                    self.sent_fail += 1


def run(lib, *, n_msgs, msg_len, seconds, filt, loss, seed, stop_db=70.0, log=None):
    rng = np.random.default_rng(seed)
    plan = F.plan_duplex_bands(400, 3300, 0.5, 1200.0, 1024, 0.25, FS, 8, 100.0, "high")
    ul, dl = tuple(plan["ul_hz"]), tuple(plan["dl_hz"])
    cu, cd = H.hs_data_cfg(FS, ul, DX.CHUNK), H.hs_data_cfg(FS, dl, DX.CHUNK)
    bu = DX.bat_for(int(cu["n_data_carriers"]), 0, False, 4)
    bd = DX.bat_for(int(cd["n_data_carriers"]), 1, True, 4)

    level, own_gain, far_gain = 0.1, 1.0, 10 ** (-40 / 20)
    far_rms = level * far_gain
    noise = far_rms * 10 ** (-25 / 20)
    thr = 4 * noise                  # a 50 ms block of the far signal (18 dB over this) is well above

    ta = design_taps(lib, dl, ul, stop_db)[0] if filt else np.zeros(0)      # A hears DL, plays UL
    tb = design_taps(lib, ul, dl, stop_db)[0] if filt else np.zeros(0)
    A = Node(lib, "A", ul, dl, bu, bd, ta, thr)
    B = Node(lib, "B", dl, ul, bd, bu, tb, thr)
    A.todo = [bytes((i * 13 + k) & 0xFF for k in range(msg_len)) for i in range(n_msgs)]
    B.todo = [bytes((i * 29 + 5 + k) & 0xFF for k in range(msg_len)) for i in range(n_msgs)]
    want_a, want_b = list(A.todo), list(B.todo)

    n_blocks = int(seconds * FS / BLOCK)
    dead_until = 0
    for k in range(n_blocks):
        t0 = k * BLOCK
        now_s = (t0 + BLOCK) / FS
        pa, pb = A.play_block(t0), B.play_block(t0)
        gain = 1.0
        if loss and t0 >= dead_until and rng.random() < loss / (FS / BLOCK):
            dead_until = t0 + int(rng.uniform(0.15, 0.6) * FS)
        if t0 < dead_until:
            gain = 0.0                                              # the channel drops out
        A.hear(own_gain * pa + far_gain * gain * pb + rng.normal(0, noise, BLOCK), now_s)
        B.hear(own_gain * pb + far_gain * gain * pa + rng.normal(0, noise, BLOCK), now_s)
        A.drive(t0 + BLOCK, level)
        B.drive(t0 + BLOCK, level)
        if (len(B.delivered) >= n_msgs and len(A.delivered) >= n_msgs
                and not A.todo and not B.todo and A.sent_ok + A.sent_fail >= n_msgs
                and B.sent_ok + B.sent_fail >= n_msgs):
            break
    t_used = (k + 1) * BLOCK / FS
    return dict(A=A, B=B, want_a=want_a, want_b=want_b, t=t_used)


def simultaneous(lib, *, filt, seed, payload=60):
    """Both ends start a data frame at the same instant.  Returns (A decoded B's, B decoded A's)."""
    rng = np.random.default_rng(seed)
    plan = F.plan_duplex_bands(400, 3300, 0.5, 1200.0, 1024, 0.25, FS, 8, 100.0, "high")
    ul, dl = tuple(plan["ul_hz"]), tuple(plan["dl_hz"])
    cu, cd = H.hs_data_cfg(FS, ul, DX.CHUNK), H.hs_data_cfg(FS, dl, DX.CHUNK)
    bu = DX.bat_for(int(cu["n_data_carriers"]), 0, False, 4)
    bd = DX.bat_for(int(cd["n_data_carriers"]), 1, True, 4)
    level, far_gain = 0.1, 10 ** (-40 / 20)
    noise = level * far_gain * 10 ** (-25 / 20)
    ta = design_taps(lib, dl, ul)[0] if filt else np.zeros(0)
    tb = design_taps(lib, ul, dl)[0] if filt else np.zeros(0)
    A = Node(lib, "A", ul, dl, bu, bd, ta, 4 * noise)
    B = Node(lib, "B", dl, ul, bd, bu, tb, 4 * noise)
    da, db = bytes(range(payload)), bytes(range(100, 100 + payload))
    sa, sb = A.end.tx(1, 0, da), B.end.tx(1, 0, db)
    sa, sb = (s * (level / np.sqrt(np.mean(s[s != 0] ** 2))) for s in (sa, sb))
    n = max(len(sa), len(sb)) + 3 * FS
    pa, pb = np.zeros(n), np.zeros(n)
    pa[FS:FS + len(sa)], pb[FS:FS + len(sb)] = sa, sb
    got = {}
    for node, own, far in ((A, pa, pb), (B, pb, pa)):
        mic = own + far_gain * far + rng.normal(0, noise, n)
        for i in range(0, n, BLOCK):
            node.hear(mic[i:i + BLOCK], i / FS)
        got[node.name] = node.frame_evts
    return got["A"] > 0, got["B"] > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--seconds", type=float, default=240.0)
    a = ap.parse_args()
    bad = n = 0

    def chk(ok, what):
        nonlocal bad, n
        n += 1
        if not ok:
            bad += 1
            print(f"  FAIL {what}")
        elif a.v:
            print(f"  ok   {what}")

    with tempfile.TemporaryDirectory() as t:
        lib = build_lib(Path(t))

        # --- filter ---------------------------------------------------------
        plan = F.plan_duplex_bands(400, 3300, 0.5, 1200.0, 1024, 0.25, FS, 8, 100.0, "high")
        ul, dl = tuple(plan["ul_hz"]), tuple(plan["dl_hz"])
        for far, mine, nm in ((dl, ul, "highpass"), (ul, dl, "lowpass")):
            taps, got = design_taps(lib, far, mine, 60.0)
            chk(len(taps) % 2 == 1 and 20 < len(taps) < 1025 and got >= 59.0, f"rxf {nm} designed ({len(taps)} taps)")
            w = np.fft.rfft(taps, 16384)
            f = np.fft.rfftfreq(16384, 1 / FS)
            mag = 20 * np.log10(np.maximum(np.abs(w), 1e-12))
            near = (f >= mine[0]) & (f <= mine[1])
            pas = (f >= far[0] + 100) & (f <= far[1])
            chk(mag[near].max() < -57.0, f"rxf {nm} stopband {mag[near].max():.1f} dB")
            chk(np.abs(mag[pas]).max() < 0.2, f"rxf {nm} passband ripple {np.abs(mag[pas]).max():.3f} dB")
        chk(len(design_taps(lib, (400, 2000), (1500, 3300))[0]) == 0, "rxf overlapping bands -> nothing to reject")

        # --- the link -------------------------------------------------------
        def report(tag, r):
            A, B = r["A"], r["B"]
            if a.v:
                print(f"    {tag}: {r['t']:.1f}s  A tx={A.tx_count} (retx {A.retry_tx}) frames={A.frame_evts} "
                      f"err={A.err_evts} | B tx={B.tx_count} (retx {B.retry_tx}) frames={B.frame_evts} err={B.err_evts}")

        r = run(lib, n_msgs=1, msg_len=40, seconds=60, filt=True, loss=0, seed=1)
        report("ping-pong", r)
        chk(r["B"].delivered == r["want_a"] and r["A"].delivered == r["want_b"]
            and r["A"].sent_ok == 1 and r["B"].sent_ok == 1, "ping-pong: one frame each way, both ACKed")

        r = run(lib, n_msgs=8, msg_len=120, seconds=a.seconds, filt=True, loss=0, seed=2)
        report("stream", r)
        chk(r["B"].delivered == r["want_a"], "stream A->B exact, in order, once")
        chk(r["A"].delivered == r["want_b"], "stream B->A exact, in order, once")
        chk(r["A"].sent_fail == 0 and r["B"].sent_fail == 0, "stream: no send gave up")

        r = run(lib, n_msgs=6, msg_len=80, seconds=a.seconds, filt=True, loss=0.3, seed=3)
        report("loss", r)
        chk(r["B"].delivered == r["want_a"] and r["A"].delivered == r["want_b"],
            "loss: exact and in order despite dropouts")
        chk(r["A"].retry_tx + r["B"].retry_tx > 0, "loss: the dropouts actually forced retransmissions")

        ok_f = [simultaneous(lib, filt=True, seed=s) for s in range(4)]
        ok_u = [simultaneous(lib, filt=False, seed=s) for s in range(4)]
        if a.v:
            print(f"    simultaneous frames, (A heard B, B heard A): filtered {ok_f}  unfiltered {ok_u}")
        chk(all(x and y for x, y in ok_f), "simultaneous TX: filtered, both ends decode every time")
        chk(not all(x and y for x, y in ok_u),
            "simultaneous TX: unfiltered fails -- the filter is what makes duplex work")

    print(f"ofdm_duplex_regress: {n - bad}/{n} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
