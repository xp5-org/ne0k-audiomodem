# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Offline checks for the capture-survival fixes (no audio hardware needed).

1. _WavSpool leaves a valid, readable WAV on disk BEFORE close.
2. A SIGTERMed process still leaves that WAV (shutdown hook path).
3. Recorder.spill_to / spill_flush / close_spill stream the chunk list.
4. A dead capture worker flips Recorder.alive to False instead of freezing
   data() silently.
"""
import os
import signal
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

TESTS_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, TESTS_DIR)
import testlib
from testlib import Recorder, _WavSpool

TMP = Path(tempfile.mkdtemp(prefix="capsurv_"))
FS = 8000
fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def wav_frames(path):
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes(), wf.getframerate()


# -- 1. valid WAV before close ----------------------------------------------
p = TMP / "spool.wav"
sp = _WavSpool(p, FS)
tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(FS) / FS)).astype(np.float32)
sp.write(tone)
sp.write(tone)
n, fs = wav_frames(p)                      # read while still OPEN
check("WAV readable before close", n == 2 * FS and fs == FS, f"{n} frames @ {fs}")
sp.write(tone)
sp.close()
n, _ = wav_frames(p)
check("WAV complete after close", n == 3 * FS, f"{n} frames")

# -- 2. survives SIGTERM -----------------------------------------------------
child_wav = TMP / "sigterm.wav"
child = f"""
import sys, time, numpy as np
sys.path.insert(0, {TESTS_DIR!r})
from testlib import _WavSpool, install_shutdown_handler, on_shutdown
install_shutdown_handler()
sp = _WavSpool({str(child_wav)!r}, {FS})
sp.write(np.zeros({FS}, dtype=np.float32) + 0.25)
on_shutdown(sp.close)
print("ready", flush=True)
# hammer stdout so SIGTERM is likely to land inside a print() call
end = time.time() + 60
while time.time() < end:
    print("tick " * 40, file=sys.stderr)
"""
proc = subprocess.Popen([sys.executable, "-c", child],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
assert proc.stdout.readline().strip() == "ready"
proc.terminate()                            # exactly what _do_stop() sends
rc = proc.wait(timeout=10)
n, _ = wav_frames(child_wav)
check("WAV survives SIGTERM", n == FS, f"{n} frames, rc={rc}")
check("SIGTERM exit code is 128+15", rc == 143, f"rc={rc}")

# -- 3. Recorder spill over a live chunk list --------------------------------
class FakeInline:
    def __init__(self):
        self._chunks = []
        self.frames_captured = 0
        self.overflows = 0

    def stop(self):
        pass

    def data(self):
        return (np.concatenate(self._chunks) if self._chunks
                else np.zeros(0, dtype=np.float32))


rec = Recorder(None, FS)          # never started: no device touched
rec._inline = FakeInline()
rp = TMP / "rec.wav"
rec.spill_to(rp, interval_s=0.2)
for _ in range(3):
    rec._inline._chunks.append(np.full(FS // 2, 0.1, dtype=np.float32))
time.sleep(0.6)                    # let the spill thread run
n, _ = wav_frames(rp)
check("spill thread writes during run", n == 3 * (FS // 2), f"{n} frames")

rec._inline._chunks.append(np.full(FS // 2, 0.1, dtype=np.float32))
rec.stop()                         # final flush + close
n, _ = wav_frames(rp)
check("stop() flushes the tail", n == 4 * (FS // 2), f"{n} frames")
check("close_spill is idempotent", (rec.close_spill() or True))

# -- 4. dead worker is detected ----------------------------------------------
import queue as _q


class DeadProc:
    exitcode = -9

    def is_alive(self):
        return False


rec2 = Recorder(None, FS)
rec2._proc = DeadProc()
rec2._q = _q.Queue()               # always empty -> Empty after timeout
check("alive True before worker death", rec2.alive)
rec2._drain_loop()                 # returns once it notices the dead proc
check("alive False after worker death", not rec2.alive)
check("frozen buffer is reported empty", len(rec2.data()) == 0)

print()
print(("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
