# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""The sounddevice sink plays the mono signal on both channels of a stereo device (CoreAudio sends a one-channel
stream to the left speaker only), or on one with the other silent (output.tx_channels left / right: a TX/RX switch
on the spare channel)."""
from __future__ import annotations

import numpy as np

from modemgui.audio import sounddevice_backend as sdb
from modemgui.audio.base import StreamSpec


class _FakeSd:
    def __init__(self, max_out):
        self.max_out, self.opened = max_out, None

    def query_devices(self, dev=None, kind=None):
        return {"max_output_channels": self.max_out}

    def OutputStream(self, channels, callback, **kw):
        self.opened = (channels, callback)
        return type("S", (), {"start": lambda s: None, "abort": lambda s: None, "close": lambda s: None})()


def _run(monkeypatch, max_out, mode="both"):
    fake = _FakeSd(max_out)
    monkeypatch.setattr(sdb, "_sd", fake)
    sink = sdb.SdSink(3, StreamSpec(rate=48000, block=4, tx_channels=mode))
    sig = np.array([0.1, -0.2, 0.3, -0.4], np.float32)
    sink.start(lambda n: sig[:n])
    ch, cb = fake.opened
    out = np.zeros((4, ch), np.float32)
    cb(out, 4, None, None)
    return ch, out, sig


def test_stereo_device_gets_both_channels(monkeypatch):
    ch, out, sig = _run(monkeypatch, 2)
    assert ch == 2 and np.array_equal(out[:, 0], sig) and np.array_equal(out[:, 1], sig)


def test_wide_device_opens_two_and_mono_device_one(monkeypatch):
    assert _run(monkeypatch, 8)[0] == 2
    ch, out, sig = _run(monkeypatch, 1)
    assert ch == 1 and np.array_equal(out[:, 0], sig)


def test_left_or_right_only_keeps_the_other_channel_silent(monkeypatch):
    for mode, on in (("left", 0), ("right", 1)):
        ch, out, sig = _run(monkeypatch, 2, mode)
        assert ch == 2 and np.array_equal(out[:, on], sig) and not out[:, 1 - on].any()
    ch, out, sig = _run(monkeypatch, 1, "right")          # a mono device has only the one channel
    assert ch == 1 and np.array_equal(out[:, 0], sig)


def test_session_takes_the_setting_at_open():
    from modemgui.engine.session import Session
    s = Session()
    s.tx_channels = "left"
    try:
        s.open(None, "null:")
        assert s.spec.tx_channels == "left"
    finally:
        s.close()
