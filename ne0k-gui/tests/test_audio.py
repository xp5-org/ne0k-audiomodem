# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
import json
import os
import textwrap
import time
import wave

import numpy as np
import pytest

from modemgui.audio import AudioError, AudioPipe, Registry, StreamSpec
from modemgui.audio.base import SINK, SOURCE, Connector
from modemgui.audio.file_backend import read_wav
from modemgui.audio.pulse_backend import (PulseConnector, parse_json_listing,
                                          parse_short_listing)

RATE = 48000
SPEC = StreamSpec(RATE, 480)


def tone(f=1000, s=0.5, amp=0.3):
    t = np.arange(int(s * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def write_wav(path, x, rate=RATE):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


# -- registry ---------------------------------------------------------------

def test_builtins_load_without_errors_for_file_loop_null():
    r = Registry(plugins=False)
    assert {"file", "loop", "null", "pulse", "sd"} <= set(r.connectors)
    assert not r.errors


def test_bad_address_and_unknown_connector():
    r = Registry(plugins=False)
    with pytest.raises(AudioError, match="want <connector>"):
        r.open_source("nocolon", SPEC)
    with pytest.raises(AudioError, match="unknown connector"):
        r.open_sink("nope:x", SPEC)


def test_plugin_dir_connector_is_found_and_broken_plugin_is_isolated(tmp_path, monkeypatch):
    (tmp_path / "good.py").write_text(textwrap.dedent('''
        import numpy as np
        from modemgui.audio.base import Connector, DeviceInfo, Source, Sink, SOURCE
        class FakeSdr(Connector):
            name = "fakesdr"
            def list_devices(self):
                return [DeviceInfo("fakesdr", "0", "Fake SDR", SOURCE)]
            def open_source(self, d, spec): raise NotImplementedError
            def open_sink(self, d, spec): raise NotImplementedError
        CONNECTORS = [FakeSdr]
    '''))
    (tmp_path / "bad.py").write_text("raise RuntimeError('driver missing')")
    monkeypatch.setenv("MODEMGUI_PLUGIN_PATH", str(tmp_path))
    r = Registry()
    assert "fakesdr" in r.connectors
    assert "file" in r.connectors                       # others unharmed
    assert "driver missing" in r.errors["plugin:bad.py"]
    assert any(d.address == "fakesdr:0" for d in r.list_devices(SOURCE))


def test_duplicate_connector_name_rejected():
    r = Registry(plugins=False)
    with pytest.raises(ValueError):
        r.add(r.connectors["file"])


def test_unavailable_connector_gives_reason_not_traceback():
    class Down(Connector):
        name = "down"
        def available(self): return False, "driver not installed"
        def list_devices(self): raise AssertionError("must not be called")
        def open_source(self, d, s): ...
        def open_sink(self, d, s): ...
    r = Registry(builtins=False, plugins=False)
    r.add(Down())
    assert r.list_devices() == []
    with pytest.raises(AudioError, match="driver not installed"):
        r.open_source("down:x", SPEC)


# -- pulse parsing ----------------------------------------------------------

def test_pulse_json_listing_marks_monitor_and_default():
    sources = json.dumps([
        {"index": 1, "name": "alsa_input.usb", "description": "USB mic", "state": "RUNNING",
         "sample_specification": "s16le 1ch 48000Hz", "monitor_of_sink": None},
        {"index": 2, "name": "alsa_output.x.monitor", "description": "Monitor of X",
         "sample_specification": "float32le 2ch 44100Hz", "monitor_of_sink": "alsa_output.x"},
    ])
    d = parse_json_listing(sources, SOURCE, "alsa_input.usb")
    assert [(x.dev_id, x.default, x.monitor, x.rate_hint) for x in d] == [
        ("alsa_input.usb", True, False, 48000), ("alsa_output.x.monitor", False, True, 44100)]
    assert d[0].address == "pulse:alsa_input.usb"


def test_pulse_short_listing_fallback():
    txt = "0\talsa_output.a\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tIDLE\n" \
          "1\talsa_output.a.monitor\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tIDLE\n"
    d = parse_short_listing(txt, SOURCE, None)
    assert [x.dev_id for x in d] == ["alsa_output.a", "alsa_output.a.monitor"]
    assert d[1].monitor and not d[0].monitor and d[0].rate_hint == 44100


def test_pulse_unavailable_when_tools_missing(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    ok, why = PulseConnector().available()
    assert not ok and "pulseaudio-utils" in why


# -- file / loop connectors -------------------------------------------------

def test_file_source_fast_reads_everything_and_resamples(tmp_path):
    x = tone(1000, 0.5)
    write_wav(tmp_path / "in.wav", x)
    assert np.allclose(read_wav(str(tmp_path / "in.wav"), RATE), x, atol=1e-3)
    assert len(read_wav(str(tmp_path / "in.wav"), 24000)) == len(x) // 2

    r = Registry(plugins=False)
    with AudioPipe(r, SPEC).open(f"file:{tmp_path/'in.wav'}#fast", None) as p:
        tap = p.subscribe(10000)
        p.start()
        t0 = time.monotonic()
        while not p.rx_finished and time.monotonic() - t0 < 3:
            time.sleep(0.01)
        got = np.concatenate([b.samples for b in tap.drain()])
    assert p.stats.rx_samples == len(x) and np.allclose(got, x, atol=1e-3)


def test_file_source_missing_fails_at_open(tmp_path):
    with pytest.raises(AudioError, match="cannot read"):
        Registry(plugins=False).open_source(f"file:{tmp_path/'nope.wav'}", SPEC)


def test_loop_roundtrip_preserves_samples_and_index_is_contiguous():
    r = Registry(plugins=False)
    x = tone(1200, 0.4)
    with AudioPipe(r, SPEC).open("loop:t1", "loop:t1") as p:
        tap = p.subscribe(10000)
        p.start()
        p.play(x)
        assert p.wait_tx(3)
        time.sleep(0.3)
        blocks = tap.drain()
    idx = [b.index for b in blocks]
    assert idx == [i * 480 for i in range(len(idx))]
    rx = np.concatenate([b.samples for b in blocks])
    pos = int(np.argmax(np.abs(rx) > 0)) - 1      # x[0] == 0, so first nonzero is x[1]
    assert np.allclose(rx[pos:pos + len(x)], x, atol=1e-6)


# -- pipe behaviour ---------------------------------------------------------

def test_slow_tap_drops_own_blocks_without_starving_others():
    r = Registry(plugins=False)
    with AudioPipe(r, SPEC).open("loop:t2", None) as p:
        slow, fast = p.subscribe(2), p.subscribe(10000)
        p.start()
        time.sleep(0.5)
        assert slow.dropped > 0 and fast.dropped == 0
        assert len(fast.drain()) > 20


def test_tx_flag_covers_playback_plus_guard_then_clears():
    r = Registry(plugins=False)
    with AudioPipe(r, SPEC, tx_guard_s=0.25).open("loop:t3", "loop:t3") as p:
        tap = p.subscribe(10000)
        p.start()
        time.sleep(0.2)
        p.play(tone(1000, 0.3))
        p.wait_tx(3)
        time.sleep(0.9)
        flags = [b.tx_active for b in tap.drain()]
    assert not flags[0] and any(flags) and not flags[-1]
    first, last = flags.index(True), len(flags) - 1 - flags[::-1].index(True)
    assert all(flags[first:last + 1])                  # one contiguous TX window


def test_stop_tx_silences_immediately_and_discards_queue():
    r = Registry(plugins=False)
    with AudioPipe(r, SPEC).open(None, "null:") as p:
        p.start()
        p.play(tone(1000, 5.0))
        assert p.tx_pending > 0
        p.stop_tx()
        assert p.tx_pending == 0 and p.wait_tx(0.1)
        assert not np.any(p._pull(480))


def test_peak_and_clip_counting():
    r = Registry(plugins=False)
    with AudioPipe(r, SPEC).open("loop:t4", "loop:t4") as p:
        p.start()
        p.play(np.ones(2400, np.float32))             # 50 ms at full scale
        p.wait_tx(3)
        time.sleep(0.4)
        assert p.stats.clipped >= 2000 and p.reset_peak() >= 0.999
        assert p.stats.peak == 0.0


def test_open_twice_and_play_without_sink_are_errors():
    r = Registry(plugins=False)
    p = AudioPipe(r, SPEC).open("null:", None)
    with pytest.raises(AudioError):
        p.open("null:", None)
    with pytest.raises(AudioError, match="no TX sink"):
        p.play(tone())
    p.close()


# -- live Pulse (skipped without a server) ----------------------------------

pulse_ok = PulseConnector().available()[0]


@pytest.mark.skipif(not pulse_ok, reason="no PulseAudio server")
def test_pulse_null_sink_monitor_loopback():
    r = Registry(plugins=False)
    sinks = [d for d in r.list_devices(SINK) if d.connector == "pulse"]
    mons = {d.dev_id for d in r.list_devices(SOURCE) if d.monitor}
    pair = next((s for s in sinks if s.dev_id + ".monitor" in mons), None)
    if pair is None:
        pytest.skip("no sink with a monitor")
    x = tone(1500, 0.6)
    with AudioPipe(r, SPEC).open(f"pulse:{pair.dev_id}.monitor", f"pulse:{pair.dev_id}") as p:
        p.start()
        time.sleep(0.5)
        p.play(x)
        p.wait_tx(5)
        time.sleep(0.5)
        heard, _ = p.history()
    t = np.arange(len(heard)) / RATE
    assert abs(np.dot(heard, np.exp(-2j * np.pi * 1500 * t))) * 2 / len(heard) > 0.02
    import subprocess
    for name in ("parec", "pacat"):        # -x: exact name, or pgrep matches its own shell
        assert subprocess.run(["pgrep", "-x", name], capture_output=True).returncode == 1, \
            f"leaked {name}"
