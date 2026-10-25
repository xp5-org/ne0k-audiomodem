# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
meshbus.py — the mixed acoustic "air" bus for the DSP-in-loop tier (Tier 2).

Where Tier 1 (meshsim_mac.py) decides delivery from an SINR formula, Tier 2
puts REAL audio on a shared bus and lets the REAL css_decoder decide.  Each
receiver hears

    rx_j(t) = Σ_{i≠j} gain(i,j) · tx_i(t − delay(i,j)) + noise_j(t)

with gain/delay from Channel2D (meshchannel.py) and a per-node noise floor from
its bucket.  A node renders packets to PCM with the real css_encoder (via
ModemFirmware.encode_packet) and writes them onto the bus; its virtual mic reads
its own mixed lane back and feeds the real decoder.

This module provides the bus itself and a streaming virtual-soundcard shim.  The
drivers live in meshsim_dsp.py: an in-process stepped "oracle" (correctness
reference) and a multiprocess/shared-memory runner (the 40-core workload).

AirBus keeps ONE mixed float32 lane per receiver (not a single shared air),
because gain and delay differ per (tx, rx) link.  For the oracle a lane is a
plain preallocated numpy array indexed by absolute sample; a node only ever
writes into the FUTURE of a lane (start ≥ the driver's current clock), so a
reader that has consumed up to the clock never races an unwritten sample.

Run:  python tests/meshbus.py        (exit 0 = encode→air→real-decode roundtrip OK)
"""
from __future__ import annotations

import os
import sys


# --- category-aware sys.path bootstrap (auto-inserted) ---
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_TESTS_ROOT = _HERE.parent if _HERE.name in ("CSS", "MESH", "OFDM", "QAM") else _HERE
_REPO_ROOT = _TESTS_ROOT.parent
for _p in (_TESTS_ROOT, _REPO_ROOT, _TESTS_ROOT / "CSS", _TESTS_ROOT / "MESH",
           _TESTS_ROOT / "OFDM", _TESTS_ROOT / "QAM", _TESTS_ROOT / "regression"):
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# --- end bootstrap ---

import numpy as np                       # noqa: E402
from meshchannel import Channel2D, NOISE_BUCKETS   # noqa: E402


class AirBus:
    """Per-receiver mixed-signal lanes over a fixed sample horizon.

    channel      : Channel2D giving gain()/prop_delay() per (tx, rx)
    fs           : sample rate
    total_samples: lane length (fs × run seconds, + tail for propagation)
    """

    def __init__(self, channel: Channel2D, fs: int, total_samples: int):
        self.ch = channel
        self.fs = int(fs)
        self.N = int(total_samples)
        self.ids = sorted(channel.pos)
        # one mixed lane per receiver (its own transmissions are NOT mixed in —
        # a half-duplex node is deaf to itself; the driver enforces TX/RX overlap)
        self.lanes = {nid: np.zeros(self.N, dtype=np.float32) for nid in self.ids}
        self._delay_smp = {(a, b): int(round(channel.prop_delay(a, b) * self.fs))
                           for a in self.ids for b in self.ids if a != b}

    def delay_samples(self, tx, rx) -> int:
        return self._delay_smp[(tx, rx)]

    def write_tx(self, src: int, start_sample: int, pcm: np.ndarray):
        """Mix one node's transmission into every other node's lane, applying
        that link's amplitude gain and propagation delay."""
        pcm = np.asarray(pcm, dtype=np.float32)
        for rx in self.ids:
            if rx == src:
                continue
            g = self.ch.gain(src, rx)
            s0 = start_sample + self.delay_samples(src, rx)
            s1 = s0 + len(pcm)
            if s0 >= self.N:
                continue
            s1c = min(s1, self.N)
            self.lanes[rx][s0:s1c] += g * pcm[:s1c - s0]

    def add_noise(self, rng=None):
        """Bake each lane's receiver noise floor in as additive white Gaussian
        noise at the node's bucket level (absolute, relative to unit-RMS ref)."""
        rng = rng or np.random.default_rng(0)
        for nid in self.ids:
            floor_rms = 10.0 ** (NOISE_BUCKETS[self.ch.profiles[nid]] / 20.0)
            self.lanes[nid] += (rng.standard_normal(self.N).astype(np.float32)
                                * floor_rms)

    def mic(self, node: int, start: int, n: int) -> np.ndarray:
        """Read n samples of a node's mixed lane starting at absolute `start`
        (zero-padded past the horizon)."""
        s0 = max(0, start)
        s1 = min(self.N, start + n)
        out = np.zeros(n, dtype=np.float32)
        if s1 > s0:
            out[s0 - start:s1 - start] = self.lanes[node][s0:s1]
        return out


class BusInputStream:
    """A streaming fake-sounddevice InputStream over one node's AirBus lane.

    Serves the decoder a node's mixed lane in `blocksize` chunks.  `clock_fn()`
    returns the driver's current global sample count; read() blocks (cooperative
    spin via a supplied waiter) until the lane is filled up to cursor+n, so the
    decoder only ever sees causal audio.  On reaching the horizon it signals EOF
    exactly like testlib.decode_array's fake stream (short read → stop_event).
    """

    def __init__(self, bus: AirBus, node: int, *, clock_fn, stop_event,
                 wait_fn=None, **_):
        self.bus = bus
        self.node = node
        self.cursor = 0
        self.clock_fn = clock_fn
        self.stop = stop_event
        self.wait_fn = wait_fn or (lambda: None)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        # block until the driver clock has produced n samples past our cursor,
        # or the run has ended / horizon reached
        while (self.clock_fn() < self.cursor + n
               and not self.stop.is_set()
               and self.cursor + n <= self.bus.N):
            self.wait_fn()
        if self.cursor >= self.bus.N or self.stop.is_set():
            # EOF: short read tells the decoder the stream is done
            self.stop.set()
            return np.zeros(n, dtype=np.float32), False
        chunk = self.bus.mic(self.node, self.cursor, n)
        self.cursor += n
        return chunk.astype(np.float32), False


# ── selftest: encode → air → REAL decode roundtrip ───────────────────────────
def _selftest() -> bool:
    import tempfile
    import contextlib
    import io as _io
    from css_modem_fw import ModemFirmware, DEFAULT_PHY
    import testlib

    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{name:40}] {'PASS' if cond else 'FAIL'}")

    fs = 48000
    work = tempfile.mkdtemp(prefix="meshbus_")
    os.environ.setdefault("AUDIOTOOL_RUN_DIR", work)

    # three nodes on a small line; node 1 (middle) will beacon
    pos = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (5.0, 0.0)}
    prof = {0: "clean", 1: "clean", 2: "clean"}
    ch = Channel2D(pos, prof, fs=fs)

    # Fast PHY for the DSP-in-loop: symbol_dur 0.1 (vs OTA's 0.5) with the OTA
    # band (800-4000 Hz). The decoder tolerates ~6% sub-symbol offset (0.1 s ->
    # 4800 smp) but not ~12% (0.05 s -> 2400 smp), so 0.1 is the floor. Beacon ~10 s.
    phy = dict(DEFAULT_PHY, fs=fs, symbol_dur=0.1, chunk_size=2,
               f_min=800, f_max=4000)
    fws = {}
    for nid in pos:
        fws[nid] = ModemFirmware(os.path.join(work, f"n{nid}.json"),
                                 node_id=nid, workdir=work, phy_override=phy,
                                 bake_info=True).boot()

    # node 1 beacons (a beacon matches the beacon-baked info.json frame contract)
    raw, tag = fws[1].node.make_beacon()
    pcm = fws[1].encode_packet(raw, tag)
    check("encoder produced PCM", pcm is not None and len(pcm) > fs // 2)

    # lay it on the bus with 1s of lead-in so there's silence before the burst
    horizon = len(pcm) + 3 * fs
    bus = AirBus(ch, fs, horizon)
    start = fs
    bus.write_tx(1, start, pcm)

    # near receiver (node 0, 2 m) and far receiver (node 2, 5 m) hear it
    g0, g2 = ch.gain(1, 0), ch.gain(1, 2)
    check("near gain > far gain", g0 > g2 > 0)
    check("lane 0 has energy", float(np.max(np.abs(bus.lanes[0]))) > 0)
    d0 = bus.delay_samples(1, 0)
    check("delay applied (2m ≈ 279 smp)", abs(d0 - round(2.0 / 343.0 * fs)) <= 1)

    # add each receiver's noise floor, then run the REAL decoder over node 0's lane
    bus.add_noise(np.random.default_rng(1))

    def decode_lane(node):
        res = testlib.decode_array(bus.lanes[node], fs, fws[node].info_path,
                                   overrides={"result_out":
                                              os.path.join(work, f"dec{node}.json")})
        frames = res.get("frames", []) or []
        return [bytes.fromhex(f["bytes_hex"]) for f in frames
                if "bytes_hex" in f]

    with contextlib.redirect_stdout(_io.StringIO()):
        got0 = decode_lane(0)
    check("node 0 decoded the real burst", raw in got0)

    return ok


def main() -> int:
    print("MESH BUS (encode → mixed air → REAL css_decoder)\n")
    ok = _selftest()
    print(f"\n*** {'ALL PASS' if ok else 'FAILED'} ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
