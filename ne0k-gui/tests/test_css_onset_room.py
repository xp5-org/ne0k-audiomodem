# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""CSS onset guard in a room whose noise sits over the decoder's absolute silence threshold.

2026-11-17 laptops (0.15 s, m6600 -> MacBook Air): most m6600 packets were lost on the Mac.  Its sync came 1.2-2.9 s
after the peer keyed instead of 5.6 s, at score ~0.5: an early lock on the training run-in, read at the wrong
offset -> FRAME_ERROR with every symbol clean (29 dB).  The onset guard that blocks that lock anchors on "signal
after >= 2 silent symbols", and "silent" was rms < silence_threshold 0.002 (-54 dBFS); the MacBook mic floor is
-64..-44 dBFS (its noise gate adds ~15 dB once open), so no symbol was ever silent and the guard never anchored.
m6600's quieter room passed.  Fix (py reference + C, parity kept): "silent" also = under the pre-sync room floor
+ onset_rel_db, a relative onset needs onset_sustain_syms loud symbols in a row and anchors HARD.

Fixture: a real m6600 -> Mac capture (sync 1.00 live) with the Mac's own idle room audio mixed in at 5x (floor
~-35 dBFS), 8 s before the burst through the first frame.  Corpus-wide check: tools/corpus_replay.py."""
import json
import wave
from pathlib import Path

import numpy as np
import pytest

FIX = Path(__file__).with_name("data") / "css_mac_room_onset_48k.wav"
MAC_CSS = dict(node_id=2, peer_id=1, fs=48000, f_min=400.0, f_max=6000.0, symbol_dur=0.15, symbol_gap_frac=0.05,
               edge_taper_cycles=3.0, amplitude=0.1, chunk_size=8, short_frames=True, crc_bytes=2, diff_encode=True,
               rs_t=0, interleave_rows=0, conv_code="none", backend="c", decoder="pll_c")


@pytest.fixture(scope="module")
def css():
    from modemgui.engine import css_link as cl
    try:
        cl.set_backend("c")
        import css_pll_c
        css_pll_c.load(False)
    except Exception as e:
        pytest.skip(f"C CSS libraries unavailable: {e}")
    return cl, cl._imports()[1]


def _replay(css, dec):
    from modemgui.engine.css_stream import CssStreamRx
    from modemgui.engine.events import Decoded
    from modemgui.engine.modemcfg import CssConfig
    cl, cm = css
    cfg = CssConfig(**MAC_CSS)
    info = json.loads(Path(cl._info(cfg)).read_text())
    out, logs = [], []
    rx = CssStreamRx(cfg, info, node_id=2, hear_own=False, log=logs.append, emit=out.append,
                     emit_link=lambda *a: None, decoded_cls=Decoded, unpack=cm.unpack_packet, dec=dec)
    with wave.open(str(FIX)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    for i in range(0, len(x), 1024):
        rx.feed(x[i:i + 1024], False)
    rx.feed(np.zeros(3 * 48000, np.float32), False)
    rx.burst_end()
    syncs = [float(l.split("score=")[1].split()[0]) for l in logs if "sync acquired score=" in l]
    return out, syncs


def test_room_noise_onset_no_longer_locks_on_the_training(css):
    out, syncs = _replay(css, None)
    assert syncs and syncs[0] >= 0.9
    assert any(d.type_name == "DATA" and d.src == 1 and "cap m" in d.text for d in out)


def test_fixture_still_reproduces_the_loss_without_the_fix(css):
    out, syncs = _replay(css, {"onset_rel_db": 0})
    assert syncs and syncs[0] < 0.6                      # the early training lock
    assert not any(d.type_name == "DATA" for d in out)
