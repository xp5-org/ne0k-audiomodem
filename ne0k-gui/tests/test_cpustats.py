"""engine/cpustats.py: stage timers, thread groups and the meter that splits CPU between them."""
import threading
import time

from modemgui.engine import cpustats as cs


def _spin(sec: float) -> None:
    t = time.thread_time()
    while time.thread_time() - t < sec:
        pass


def test_group_of_names():
    assert cs.group_of("MainThread", True) == "ui"
    assert cs.group_of("mesh-rx-cap", True) == "rx"
    assert cs.group_of("mesh-mac", True) == "link"
    assert cs.group_of("ofdm-link", True) == "link"
    assert cs.group_of("pulse-src", True) == "audio"
    assert cs.group_of("com.apple.audio.IOThread.client", False) == "audio"
    assert cs.group_of("control-client", True) == "io"
    assert cs.group_of("Thread-7", True) == "other"
    assert cs.group_of("QXcbEventQueue", False) == "native"


def test_nested_stage_counts_once_for_the_outer_label():
    before = sum(v for (s, _), v in cs.stage_totals().items() if s in ("decode", "encode"))
    with cs.stage("decode"):
        with cs.stage("encode"):
            _spin(0.05)
    tot = cs.stage_totals()
    tid = threading.get_native_id()
    assert tot.get(("decode", tid), 0) >= 0.05
    after = sum(v for (s, _), v in tot.items() if s in ("decode", "encode"))
    assert after - before < 0.09                 # not counted twice


def test_meter_splits_stage_time_out_of_its_thread():
    m = cs.CpuMeter()

    def work():
        with cs.stage("encode"):
            _spin(0.15)
        _spin(0.1)                               # the thread's own (non-stage) work
    t = threading.Thread(target=work, name="mesh-mac")
    m.sample()
    t.start(); t.join()
    smp = m.sample()
    g = smp["groups"]
    assert g.get("encode", 0) > 0 and smp["total"] > 0
    assert "Encode" in cs.summary(smp)
    if smp["threads"]:                          # the OS let us read per-thread times (Linux / macOS / Windows)
        assert abs(g["encode"] / max(g.get("link", 0.0), 1e-9) - 1.5) < 0.8, g


def test_summary_keeps_its_layout_when_groups_go_idle():
    busy = cs.summary({"groups": {"decode": 41.2, "ui": 12.0, "pll": 3.0}})
    idle = cs.summary({"groups": {}})
    for lab in ("Decode", "Encode", "UI", "Link/MAC", "RX capture", "Audio I/O", "Native"):
        assert lab in idle and lab in busy
    assert "CSS PLL" in busy and "CSS PLL" not in idle
    assert len(idle) == len(cs.summary({"groups": {"decode": 107.0}}))        # padded: same character count
    assert "CSS PLL" in cs.summary({"groups": {}}, shown=set(cs.CORE) | {"pll"})   # a joined group stays
