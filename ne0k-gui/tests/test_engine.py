import time

import numpy as np

from modemgui.audio import Registry
from modemgui.engine import Levels, Session, Spectrum, Status
from modemgui.engine.spectro import Stft


def test_stft_peak_bin_and_continuity():
    r, f = 48000, 3000.0
    x = (0.5 * np.sin(2 * np.pi * f * np.arange(r) / r)).astype(np.float32)
    s = Stft(r, 2048, 512)
    got = []
    for i in range(0, len(x), 1000):               # odd chunking must not matter
        idx, cols = s.feed(x[i:i + 1000], i)
        got.append(cols)
    cols = np.concatenate(got)
    assert len(cols) == (len(x) - 2048) // 512 + 1
    assert abs(s.freqs[cols[10].argmax()] - f) < 25
    assert -8 < cols[10].max() < 0                 # 0.5 amplitude -> about -6 dBFS


def test_stft_gap_resets():
    s = Stft(48000, 1024, 256)
    s.feed(np.zeros(1500, np.float32), 0)
    idx, cols = s.feed(np.zeros(1024, np.float32), 10_000)   # gap
    assert idx == 10_000 and len(cols) == 1


def test_session_events_from_file_source(tmp_path):
    import wave
    r = 48000
    x = (0.4 * np.sin(2 * np.pi * 2000 * np.arange(r) / r) * 32767).astype("<i2")
    p = tmp_path / "t.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(r); w.writeframes(x.tobytes())
    s = Session(Registry())
    s.open(f"file:{p}#fast", None)
    evs = []
    t0 = time.time()
    while time.time() - t0 < 5 and not (any(isinstance(e, Spectrum) for e in evs)
                                        and any(isinstance(e, Levels) for e in evs)):
        evs += s.drain(); time.sleep(0.05)
    s.close()
    evs += s.drain()
    assert [e.state for e in evs if isinstance(e, Status)][0] == "open"
    assert evs[-1].state == "closed"
    lv = next(e for e in evs if isinstance(e, Levels))
    assert -12 < lv.peak_db < -6
    sp = next(e for e in evs if isinstance(e, Spectrum))
    assert abs(sp.freqs[sp.cols[0].argmax()] - 2000) < 50


def test_css_text_roundtrip_offline():
    from modemgui.engine.css_link import decode_samples, encode_text
    from modemgui.engine.modemcfg import CssConfig
    c = CssConfig()
    out = decode_samples(c, encode_text(c, "hello modem", seq=5))
    assert [(d.text, d.seq, d.ok) for d in out] == [("hello modem", 5, True)]


def test_css_config_roundtrip():
    from modemgui.engine.modemcfg import CssConfig
    c = CssConfig(symbol_dur=0.1, diff_encode=False, node_id=3)
    assert CssConfig.from_dict({k: str(v) for k, v in c.to_dict().items()}) == c
