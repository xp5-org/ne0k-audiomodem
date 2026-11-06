# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Ham mode: station ID audio (CW, binary ASCII) and the two timers."""
import time

import numpy as np
import pytest

from modemgui.audio import Registry
from modemgui.engine import Session
from modemgui.engine.ident import (MORSE, Identifier, ascii_audio, cw_audio, decode_ascii, id_text)
from modemgui.engine.modemcfg import HamConfig, describe

FS = 48000


def _cw_decode(pcm: np.ndarray, wpm: float) -> str:
    """Key the envelope, then read dots/dashes/gaps by length."""
    dot = 1.2 / wpm * FS
    on = np.convolve(np.abs(pcm), np.ones(int(dot / 4)) / (dot / 4), "same") > 0.05
    edges = np.flatnonzero(np.diff(on.astype(int))) + 1
    runs = np.diff(np.concatenate([[0], edges, [len(on)]]))
    state = bool(on[0]); out = ""; code = ""
    inv = {v: k for k, v in MORSE.items()}
    for i, r in enumerate(runs):
        if state:
            code += "." if r < 2 * dot else "-"
        elif i and r >= 2 * dot:
            out += inv.get(code, "?"); code = ""
            if r >= 5 * dot:
                out += " "
        state = not state
    return (out + inv.get(code, "?")) if code else out


def test_every_ham_field_is_described():
    for d in describe(HamConfig):
        assert d["label"] and d["group"], d["name"]


def test_cw_carries_the_callsign():
    pcm = cw_audio(id_text("n0call"), FS, 15, 800, 0.3)
    assert _cw_decode(pcm, 15) == "DE N0CALL"
    assert np.max(np.abs(pcm)) <= 0.3 + 1e-6


@pytest.mark.parametrize("baud", [110.0, 300.0, 600.0])
def test_binary_ascii_round_trip(baud):
    pcm = ascii_audio("w1aw/p", FS, baud, 0.3)
    assert decode_ascii(pcm, FS, baud) == "DE W1AW/P"
    noisy = pcm + np.random.default_rng(1).normal(0, 0.05, len(pcm)).astype(np.float32)
    assert decode_ascii(noisy, FS, baud) == "DE W1AW/P"


class _S:
    class spec:
        rate = FS


def _ident(**kw):
    return Identifier(_S(), HamConfig(enabled=True, callsign="N0CALL", **kw))


def test_disabled_or_no_callsign_leaves_frames_alone():
    x = np.ones(1000, np.float32)
    assert Identifier(_S(), HamConfig()).wrap(x) is x
    assert Identifier(_S(), HamConfig(enabled=True, callsign="  ")).wrap(x) is x


def test_interval_timer_ascii_goes_in_front_cw_goes_behind():
    x = np.full(FS, 0.5, np.float32)
    a = _ident(id_mode="ascii", id_interval_s=60)
    out = a.wrap(x)                                 # first frame of the session is an ID
    assert len(out) > len(x) and np.array_equal(out[-len(x):], x)
    assert decode_ascii(out[:-len(x)], FS, 300.0) == "DE N0CALL"
    assert a.wrap(x) is x                           # inside the interval: untouched

    c = _ident(id_mode="cw", id_interval_s=60)
    out = c.wrap(x)
    assert np.array_equal(out[:len(x)], x) and len(out) > len(x) + int(1.5 * FS)
    assert not np.any(out[len(x):len(x) + int(1.4 * FS)])      # the gap

    b = _ident(id_mode="both", id_interval_s=60)
    assert len(b.wrap(x)) > len(out)


def test_interval_elapses_and_ids_again():
    i = _ident(id_mode="cw", id_interval_s=10)
    x = np.zeros(100, np.float32)
    i.wrap(x)
    assert i.wrap(x) is x
    i._last_id -= 11
    assert len(i.wrap(x)) > len(x)
    assert len(i.sent) == 2


def test_end_of_transmission_timer_sends_a_lone_id():
    s = Session(Registry(), FS, 1024)
    s.ident.cfg = HamConfig(enabled=True, callsign="N0CALL", id_mode="cw", id_interval_s=3600,
                            id_end_delay_s=1.0, cw_wpm=30)
    s.open("loop:id", "loop:id")
    try:
        frame = np.sin(np.arange(FS // 2) * 0.2).astype(np.float32) * 0.5
        s.play(frame)                               # first frame: ID rides with it
        s.play(frame)                               # second: ID is now "owed" at the end
        assert [r for _, _, r in s.ident.sent] == ["interval"]
        t0 = time.monotonic()
        while len(s.ident.sent) < 2 and time.monotonic() - t0 < 12:
            time.sleep(0.1)
        assert [r for _, _, r in s.ident.sent] == ["interval", "end of transmission"]
        assert time.monotonic() - t0 >= 1.0            # waited out the quiet time
        time.sleep(2.0)
        assert len(s.ident.sent) == 2                # nothing more owed
    finally:
        s.close()


def test_end_timer_off_when_zero():
    s = Session(Registry(), FS, 1024)
    s.ident.cfg = HamConfig(enabled=True, callsign="N0CALL", id_interval_s=3600, id_end_delay_s=0)
    s.open("loop:id0", "loop:id0")
    try:
        f = np.zeros(2000, np.float32)
        s.play(f); s.play(f)
        time.sleep(1.5)
        assert len(s.ident.sent) == 1
    finally:
        s.close()


def test_send_now_needs_a_callsign():
    s = Session(Registry(), FS, 1024)
    s.open("loop:idn", "loop:idn")
    try:
        with pytest.raises(ValueError):
            s.ident.send_now()
        s.ident.cfg = HamConfig(enabled=True, callsign="K1ABC", id_mode="ascii")
        assert s.ident.send_now() > 0.5
    finally:
        s.close()


@pytest.mark.parametrize("mode,kind", [("ascii", "ascii"), ("cw", "cw")])
def test_css_link_hears_the_id_and_still_decodes_the_frame(mode, kind):
    """ID rides with a real CSS frame through a loop device; RX takes the ID burst as an ID (not
    an error) and decodes the frame that follows/precedes it."""
    from modemgui.engine.css_link import CssLink
    from modemgui.engine.events import Decoded
    from modemgui.engine.modemcfg import CssConfig
    s = Session(Registry(), FS, 1024)
    s.ident.cfg = HamConfig(enabled=True, callsign="N0CALL", id_mode=mode, id_end_delay_s=0)
    lg: list[str] = []
    s.open("loop:idcss", "loop:idcss")
    link = CssLink(s, CssConfig(symbol_dur=0.05, hear_own=True), log=lg.append)
    link.start()
    got: list = []
    try:
        link.send("hi")
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:
            got += s.drain()
            if any(isinstance(e, Decoded) for e in got) and s.ident.heard:
                break
            time.sleep(0.2)
    finally:
        link.stop(); s.close()
    dec = [e for e in got if isinstance(e, Decoded)]
    assert dec and dec[0].text == "hi" and dec[0].ok, lg
    assert s.ident.heard and s.ident.heard[0][1:] == (kind, "DE N0CALL"), (s.ident.heard, lg)
    assert not any("no frames decoded" in m for m in lg)


def test_keyup_tone_is_not_an_id():
    """The OFDM keyup lead (1 kHz) alone, or two of them beating on a shared line, used to decode as E/T strings."""
    from modemgui.engine.ident import KEYUP_TONE_HZ, heard_ident
    fs = 48000
    t = np.arange(int(2.0 * fs)) / fs
    one = 0.3 * np.sin(2 * np.pi * KEYUP_TONE_HZ * t)
    beat = one + 0.3 * np.sin(2 * np.pi * (KEYUP_TONE_HZ + 3.0) * t)
    for x in (one, beat, np.concatenate([np.zeros(fs // 2), one[: fs // 3], np.zeros(fs // 2)])):
        assert heard_ident(x.astype(np.float32), fs, tone_hz=800.0) is None
        assert heard_ident(x.astype(np.float32), fs) is None
    cw = cw_audio(id_text("CEC6U"), fs, 15.0, 800.0, 0.3)
    assert heard_ident(cw, fs, tone_hz=800.0) == ("cw", "DE CEC6U")


def test_same_id_through_several_listeners_is_logged_once():
    from modemgui.engine.ident import Identifier
    lines = []
    idt = Identifier(None, HamConfig(enabled=True, callsign="6Z8CC"), log=lines.append)
    cw = cw_audio(id_text("CEMER"), 48000, 15.0, 800.0, 0.3)
    assert idt.note_heard(cw, 48000) and idt.note_heard(cw, 48000)
    assert len(idt.heard) == 1 and lines == ["ident heard (cw): DE CEMER"]


def test_announce_hears_an_id_glued_to_its_frame():
    """The start-of-session ID runs into the announce's keyup with less than the burst-ending quiet between."""
    from modemgui.engine.ident import KEYUP_TONE_HZ, heard_ident
    from modemgui.engine.ofdm_announce import _before_keyup
    fs = 48000
    cw = cw_audio(id_text("CEMER"), fs, 15.0, 800.0, 0.3)
    rng = np.random.default_rng(1)
    burst = np.concatenate([np.zeros(fs // 2), cw, np.zeros(int(0.4 * fs)),
                            0.3 * np.sin(2 * np.pi * KEYUP_TONE_HZ * np.arange(2 * fs) / fs),
                            0.1 * rng.standard_normal(5 * fs)]).astype(np.float32)
    assert heard_ident(_before_keyup(burst, fs), fs, tone_hz=800.0) == ("cw", "DE CEMER")


def test_a_click_at_the_end_does_not_turn_the_id_into_dashes():
    """Seen live: the first 4 ms of the next keyup tone at the end of the ID set the dot length -> 'T T T ... E'."""
    from modemgui.engine.ident import heard_ident
    fs = 48000
    cw = cw_audio(id_text("QKRE0"), fs, 15.0, 800.0, 0.3)
    blip = 0.3 * np.sin(2 * np.pi * 800.0 * np.arange(int(0.004 * fs)) / fs)
    x = np.concatenate([cw, np.zeros(int(0.3 * fs)), blip]).astype(np.float32)
    assert heard_ident(x, fs, tone_hz=800.0) == ("cw", "DE QKRE0")
