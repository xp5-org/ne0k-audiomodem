# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""ofdm_dx.c link-frame codec: CRC pin, C<->C loopback, C<->Python interop.

    python tests/regression/ofdm_dx_regress.py [-v]

Compiles the OFDM cores plus ofdm_dx.c into a throwaway shared library.  Needs
numpy/scipy, so run it with the project venv.  Exit 0 = pass.

What it grades:
  crc        ofdm_dx_crc16 == ofdm_handshake.crc16
  loop       C tx -> C rx, both sub-bands of a duplex plan, several payload sizes,
             clean and with AWGN
  c->py      C-built frame decoded by the Python AdaptiveDemodulator
  py->c      Python-built frame (make_data_audio) decoded by the C receive chain
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CDIR = ROOT / "OFDM" / "c"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "OFDM"))

import ofdm_fdd as F                                            # noqa: E402
import ofdm_handshake as H                                      # noqa: E402
from ofdm_modem import AdaptiveDemodulator                      # noqa: E402

D = ctypes.c_double
I = ctypes.c_int
FS = 48000
CHUNK = 48

SRCS = ["ofdm_acq.c", "ofdm_dem.c", "ofdm_fft.c", "ofdm_cfg.c", "ofdm_sym.c", "ofdm_mod.c",
        "ofdm_probe.c", "ofdm_sync.c", "ofdm_frame.c", "ofdm_fdd.c", "ofdm_dx.c",
        "../../c/rs255.c"]


class Cfg(ctypes.Structure):
    _fields_ = [(n, I) for n in (
        "n_fft", "cp_len", "bin_lo", "bin_hi", "n_occupied", "pilot_spacing", "n_pilots",
        "n_data_carriers", "sample_rate", "symbol_samples", "taper_samples",
        "frame_gap_samples", "ecc_bytes", "chunk_bytes", "seq_bytes", "sync_period")] + \
        [(n, D) for n in ("cp_ratio", "sc_spacing_hz", "freq_lo_hz", "freq_hi_hz")]


class Params(ctypes.Structure):
    _fields_ = [("tx_cfg", Cfg), ("rx_cfg", Cfg),
                ("tx_bat", ctypes.POINTER(I)), ("rx_bat", ctypes.POINTER(I))]


class RxInfo(ctypes.Structure):
    _fields_ = [(n, I) for n in ("seq", "ack", "len", "n_chunks", "n_found", "n_bad", "errata")] + \
               [("evm", D), ("first_start", I)]


def build_lib(tmp: Path, extra=(), opt="-O1") -> ctypes.CDLL:
    so = tmp / "dx.so"
    subprocess.run(["cc", opt, "-std=c99", "-fPIC", "-shared", "-Wall", "-Wextra", "-pedantic",
                    f"-I{CDIR}", f"-I{CDIR}/../../c", "-o", str(so)]
                   + [str(CDIR / s) for s in list(SRCS) + list(extra)] + ["-lm"], check=True)
    lib = ctypes.CDLL(str(so))
    declare(lib)
    return lib


def declare(lib) -> None:
    lib.ofdm_cfg_build.argtypes = [ctypes.POINTER(Cfg), I, D, D, D, I, I, I, I, I, D, D, I]
    lib.ofdm_dx_workspace.argtypes = [ctypes.POINTER(Params)]
    lib.ofdm_dx_workspace.restype = ctypes.c_size_t
    lib.ofdm_dx_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                 ctypes.c_size_t, ctypes.POINTER(Params)]
    lib.ofdm_dx_tx_samples.argtypes = [ctypes.c_void_p, I]
    lib.ofdm_dx_tx_samples.restype = ctypes.c_long
    lib.ofdm_dx_tx_frame.argtypes = [ctypes.c_void_p, I, I, ctypes.c_char_p, I,
                                     ctypes.POINTER(D), ctypes.c_long]
    lib.ofdm_dx_tx_frame.restype = ctypes.c_long
    lib.ofdm_dx_rx_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(D), I,
                                     ctypes.c_char_p, ctypes.POINTER(RxInfo)]
    lib.ofdm_dx_crc16.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.ofdm_dx_crc16.restype = ctypes.c_uint
    lib.ofdm_dx_capacity.argtypes = [ctypes.c_void_p, I]


def c_cfg(lib, cfg: dict) -> Cfg:
    c = Cfg()
    rc = lib.ofdm_cfg_build(ctypes.byref(c), cfg["n_fft"], cfg["cp_ratio"], cfg["freq_lo_hz"],
                            cfg["freq_hi_hz"], cfg["pilot_spacing"], cfg["sample_rate"],
                            cfg["ecc_bytes"], cfg["chunk_bytes"], cfg["seq_bytes"],
                            cfg["taper_ms"], cfg["frame_gap_ms"], cfg["sync_period"])
    assert rc == 0, rc
    return c


class End:
    """One duplex end: C engine plus the Python configs it was built from."""

    def __init__(self, lib, tx_band, rx_band, tx_bat, rx_bat):
        self.lib = lib
        self.tx_py = H.hs_data_cfg(FS, tx_band, CHUNK)
        self.rx_py = H.hs_data_cfg(FS, rx_band, CHUNK)
        self.tx_bat = np.asarray(tx_bat, dtype=np.int32)
        self.rx_bat = np.asarray(rx_bat, dtype=np.int32)
        p = Params()
        p.tx_cfg, p.rx_cfg = c_cfg(lib, self.tx_py), c_cfg(lib, self.rx_py)
        self._tb = (I * len(self.tx_bat))(*self.tx_bat)
        self._rb = (I * len(self.rx_bat))(*self.rx_bat)
        p.tx_bat = ctypes.cast(self._tb, ctypes.POINTER(I))
        p.rx_bat = ctypes.cast(self._rb, ctypes.POINTER(I))
        need = lib.ofdm_dx_workspace(ctypes.byref(p))
        assert need > 0, "workspace rejected the parameters"
        self.mem = ctypes.create_string_buffer(need)
        self.h = ctypes.c_void_p()
        rc = lib.ofdm_dx_init(ctypes.byref(self.h), self.mem, need, ctypes.byref(p))
        assert rc == 0, f"ofdm_dx_init rc={rc}"
        self.n_tx = len(self.tx_bat)

    def tx(self, seq, ack, data: bytes) -> np.ndarray:
        n = self.lib.ofdm_dx_tx_samples(self.h, len(data))
        assert n > 0
        out = (D * n)()
        got = self.lib.ofdm_dx_tx_frame(self.h, seq, ack, data, len(data), out, n)
        assert got == n, f"tx_frame rc={got} want {n}"
        return np.frombuffer(out, dtype=np.float64).copy()

    def rx(self, x: np.ndarray):
        x = np.ascontiguousarray(x, dtype=np.float64)
        buf = ctypes.create_string_buffer(256)
        info = RxInfo()
        rc = self.lib.ofdm_dx_rx_frame(self.h, x.ctypes.data_as(ctypes.POINTER(D)), len(x),
                                       buf, ctypes.byref(info))
        if rc != 1:
            return None, info
        return (info.seq, info.ack, buf.raw[:info.len]), info


def py_pack(seq, ack, data: bytes) -> bytes:
    head = bytes([seq & 0xFF, ack & 0xFF, len(data)])
    return head + H.crc16(head + data).to_bytes(2, "big") + data


def py_unpack(raw: bytes):
    if len(raw) < 5:
        return None
    seq, ack, n = raw[0], raw[1], raw[2]
    if len(raw) < 5 + n:
        return None
    data = raw[5:5 + n]
    if H.crc16(raw[:3] + data) != int.from_bytes(raw[3:5], "big"):
        return None
    return seq, ack, data


def bat_for(n: int, salt: int, zeros: bool, max_bits: int = 6) -> list:
    pat = [min(b, max_bits) for b in (2, 4, 1, 2, 6, 2, 4, 1)]
    out = [pat[(i * 3 + salt) % len(pat)] for i in range(n)]
    if zeros:
        for i in range(0, n, 11):
            out[i] = 0
    return out


def payload(n: int, k: int) -> bytes:
    return bytes((i * 31 + k * 7 + 1) & 0xFF for i in range(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", action="store_true")
    v = ap.parse_args().v
    bad = n = 0

    def chk(ok, what):
        nonlocal bad, n
        n += 1
        if not ok:
            bad += 1
            print(f"  FAIL {what}")
        elif v:
            print(f"  ok   {what}")

    rng = np.random.default_rng(7)

    with tempfile.TemporaryDirectory() as t:
        lib = build_lib(Path(t))

        for sz in (0, 1, 17, 100, 255):
            d = payload(sz, 3)
            head = bytes([5, 9, sz])
            chk(lib.ofdm_dx_crc16(head + d, 3 + sz) == H.crc16(head + d), f"crc16 len={sz}")

        plan = F.plan_duplex_bands(400, 3300, 0.5, 1200.0, 1024, 0.25, FS, 8, 100.0, "high")
        ul, dl = tuple(plan["ul_hz"]), tuple(plan["dl_hz"])
        cu, cd = H.hs_data_cfg(FS, ul, CHUNK), H.hs_data_cfg(FS, dl, CHUNK)
        bu = bat_for(int(cu["n_data_carriers"]), 0, False)
        bd = bat_for(int(cd["n_data_carriers"]), 1, True)

        # initiator: tx UL, rx DL.  responder: the mirror.
        A = End(lib, ul, dl, bu, bd)
        B = End(lib, dl, ul, bd, bu)

        # --- loopback both directions, clean and noisy -----------------------
        for name, src, dst in (("UL->", A, B), ("DL->", B, A)):
            for sz in (0, 1, 17, 100, 230):
                d = payload(sz, sz)
                sig = src.tx(len(d) & 0xFF, 0x42, d)
                for snr in (None, 25.0):
                    x = np.concatenate([np.zeros(4000), sig, np.zeros(4000)])
                    if snr is not None:
                        rms = np.sqrt(np.mean(sig[sig != 0] ** 2))
                        x = x + rng.normal(0, rms * 10 ** (-snr / 20), len(x))
                    got, info = dst.rx(x)
                    ok = got is not None and got == (len(d) & 0xFF, 0x42, d)
                    chk(ok, f"loop {name} {sz}B snr={snr}"
                        + ("" if ok else f" (found={info.n_found} bad={info.n_bad})"))

        # --- C tx -> Python decode -------------------------------------------
        for sz in (1, 40, 150):
            d = payload(sz, 9)
            sig = np.concatenate([np.zeros(4000), A.tx(11, 22, d), np.zeros(4000)])
            dem = AdaptiveDemodulator(cu, np.asarray(bu, dtype=np.int32))
            _, chunks = dem.decode_samples(sig.astype(np.float32))
            raw = b"".join(chunks[k] for k in sorted(chunks)) if chunks else b""
            chk(sorted(chunks) == list(range(len(chunks))) and chunks
                and py_unpack(raw) == (11, 22, d), f"c->py {sz}B")

        # --- Python tx -> C decode -------------------------------------------
        for sz in (1, 40, 150):
            d = payload(sz, 5)
            audio = H.make_data_audio(FS, py_pack(33, 44, d), np.asarray(bd, dtype=np.int32),
                                      dl, CHUNK)
            x = np.concatenate([np.zeros(4000), np.asarray(audio, dtype=np.float64), np.zeros(4000)])
            got, info = A.rx(x)             # A listens on DL
            chk(got == (33, 44, d), f"py->c {sz}B" + ("" if got else f" (found={info.n_found})"))

        # --- a damaged frame must not be delivered ----------------------------
        d = payload(100, 1)
        sig = B.tx(1, 2, d)
        hole = sig.copy()
        a = len(hole) // 3
        hole[a:a + len(hole) // 2] = 0.0
        got, _ = A.rx(np.concatenate([np.zeros(4000), hole, np.zeros(4000)]))
        chk(got is None, "gap in a multi-chunk frame is rejected, not mis-delivered")

    print(f"ofdm_dx_regress: {n - bad}/{n} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
