# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""The broadcast carrier's chunks come from the C modulator (ofdm_mod_encode_ex) and must be the
reference's samples: seq0 numbering with wrap, gapless and gapped, on the default and a narrowed band."""
import numpy as np
import pytest

pytest.importorskip("scipy")             # the reference OFDM modem needs it

from modemgui import native
from modemgui.engine.modemcfg import OfdmConfig
from modemgui.engine.ofdm_link import BCAST_HDR, OfdmLink, _imports


def _cmod_or_skip():
    try:
        from modemgui.engine.ofdm_cmod import _lib
        _lib()
    except (native.NativeError, OSError) as e:
        pytest.skip(f"libofdm with ofdm_mod_encode_ex not built: {e}")


def _link(logs):
    lk = OfdmLink.__new__(OfdmLink)
    lk.cfg, lk.log = OfdmConfig(), logs.append
    lk._tx_band = None                   # band=None means "our tx band": the handshake default here
    return lk


@pytest.mark.parametrize("band", [None, (600.0, 3000.0)])
def test_bcast_chunk_is_reference_samples(band):
    _cmod_or_skip()
    H, M = _imports()
    logs: list[str] = []
    lk = _link(logs)
    c = lk.cfg
    nd = H.hs_data_cfg(c.fs, band, c.chunk_bytes)["n_data_carriers"]
    bat = np.array([(1, 2, 4, 2)[i % 4] for i in range(nd)], dtype=np.int32)
    seq_mod = 1 << (8 * H.hs_data_cfg(c.fs, band, c.chunk_bytes)["seq_bytes"])
    piece = bytes((i * 29 + 3) & 0xFF for i in range(c.chunk_bytes - BCAST_HDR - 9))
    body = len(piece).to_bytes(BCAST_HDR, "big") + piece
    body += b"\0" * (c.chunk_bytes - len(body))
    for seq, gap in ((0, True), (17, False), (seq_mod - 1, False), (seq_mod + 3, True)):
        got = lk._bcast_audio(H, piece, seq, bat, band, gap=gap)
        ref = H.make_data_audio(c.fs, body, bat, band, c.chunk_bytes, seq0=seq, gap=gap)
        ref = H.scale_hs_tx(ref, c.amplitude, c.fs, is_probe=False, band=band)[0]
        assert len(got) == len(ref), (seq, gap)
        assert float(np.max(np.abs(got - ref))) < 1e-9, (seq, gap)
    assert not getattr(lk, "_bcast_cmod_off", False)
    assert logs == ["bcast encoder: c (libofdm ofdm_mod_encode_ex)"]


def test_bcast_falls_back_to_numpy(monkeypatch):
    H, _ = _imports()
    logs: list[str] = []
    lk = _link(logs)

    def broken(stem):
        raise native.NativeError("no libofdm here")
    monkeypatch.setattr(native, "load", broken)
    import modemgui.engine.ofdm_cmod as cm
    monkeypatch.setattr(cm, "_lib", lambda: broken("libofdm"))
    nd = H.hs_data_cfg(lk.cfg.fs, None, lk.cfg.chunk_bytes)["n_data_carriers"]
    bat = np.full(nd, 2, dtype=np.int32)
    a = lk._bcast_audio(H, b"hello", 4, bat, None)
    assert len(a) > 0 and lk._bcast_cmod_off
    assert logs and logs[0].startswith("bcast encoder: numpy")
