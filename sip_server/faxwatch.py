# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 XP5.ORG
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""The Python watcher that sits in the middle of a fax call (the `watcher` option of a sipserver.py
`bridge` line). Every byte the two modem mocks exchange passes through it, in both directions.

WATCHER INTERFACE (write your own, point `watcher.module` at the file, `watcher.class` at the class):

    class Watcher:
        def __init__(self, cfg, label, log):
            # cfg: the "watcher" object from the server config; label: "call c3"; log: log(text)
        def process(self, direction, data):
            # direction: "a2b" (caller -> callee) or "b2a"; data: raw bytes off the wire.
            # Return the list of byte strings to forward on, in order ([] swallows, [data] passes).
        def close(self):
            # the call is over: flush files, log a summary

FaxWatcher below is the default. It understands the faxcodec wire frames, logs every frame
(`call c1 watch: a2b S 1,3,0,2,0,0,0,0`), saves every page it sees in each direction as .t4 plus a
TIFF per direction and a JSON frame trace per call into `dir`, and can break things on purpose:

    "watcher": {"dir": "/path/watch",
                "faults": [
                  {"frame": "R", "direction": "b2a", "nth": 1, "action": "set", "payload": "2"},
                  {"frame": "P", "direction": "a2b", "nth": 3, "action": "mangle"},
                  {"frame": "N", "nth": 2, "action": "drop"},
                  {"frame": "I", "action": "duplicate"}]}

  frame      I A S C P N R D        (the faxcodec frame types; required)
  direction  a2b | b2a | omitted = both
  nth        the Nth matching frame, counted from 1 per rule; "every": K = every Kth; omitted = all
  action     drop       the frame never arrives (the far end's timers decide what happens)
             set        replace the payload with "payload" (e.g. turn a good page response, 1, into 2)
             mangle     flip one bit in the payload but keep the frame valid, so a page carries bad data
             duplicate  deliver the frame twice

Stable phrases for test_filecontains:
  call c1 watch: a2b I "TEST DOS"          call c1 watch: b2a A 1,3,0,2,0,0,0,0
  call c1 watch: a2b S 1,3,0,2,0,0,0,0     call c1 watch: b2a C
  call c1 watch: a2b page 1 complete, 1065 bytes (EOP)
  call c1 watch: b2a R 1                   call c1 watch: b2a D 0
  call c1 watch: fault set b2a R #1        call c1 watch: summary pages=1 ...
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import faxcodec as fc  # noqa: E402

POST = {"0": "MPS", "1": "EOM", "2": "EOP"}


class FaxWatcher:
    def __init__(self, cfg, label, log):
        self.cfg, self.label, self.log = cfg, label, log
        self.df = {"a2b": fc.Deframer(), "b2a": fc.Deframer()}
        self.pages = {"a2b": [], "b2a": []}
        self.chunks = {"a2b": [], "b2a": []}
        self.trace = []
        self.t0 = time.monotonic()
        self.faults = []
        for f in cfg.get("faults", []):
            if f.get("frame") not in fc.FRAME_TYPES or f.get("action") not in ("drop", "set", "mangle", "duplicate"):
                raise ValueError(f"bad watcher fault {f!r}")
            self.faults.append(dict(f, seen=0))
        self.caps = None
        self.closed = False

    def say(self, text):
        self.log(f"{self.label} watch: {text}")

    @staticmethod
    def text(payload):
        return payload.decode("latin-1", "replace")

    def process(self, direction, data):
        out = []
        for ftype, payload in self.df[direction].feed(data):
            out.extend(self.one(direction, ftype, payload))
        return out

    def one(self, direction, ftype, payload):
        self.trace.append([round(time.monotonic() - self.t0, 3), direction, ftype, len(payload)])
        self.describe(direction, ftype, payload)
        frames = [(ftype, payload)]
        for f in self.faults:
            if f["frame"] != ftype or f.get("direction", direction) != direction:
                continue
            f["seen"] += 1
            n = f["seen"]
            if "nth" in f and n != int(f["nth"]):
                continue
            if "every" in f and n % int(f["every"]):
                continue
            act = f["action"]
            self.say(f"fault {act} {direction} {ftype} #{n}")
            if act == "drop":
                return []
            if act == "set":
                frames = [(ftype, str(f.get("payload", "")).encode("latin-1"))]
            elif act == "mangle" and payload:
                i = len(payload) // 2
                frames = [(ftype, payload[:i] + bytes([payload[i] ^ 0x01]) + payload[i + 1:])]
            elif act == "duplicate":
                frames = [(ftype, payload), (ftype, payload)]
            break
        return [fc.pack(t, p) for t, p in frames]

    def describe(self, d, ftype, payload):
        if ftype == "I":
            self.say(f'{d} I "{self.text(payload)}"')
        elif ftype in ("A", "S"):
            if ftype == "S":
                self.caps = self.text(payload)
            self.say(f"{d} {ftype} {self.text(payload)}")
        elif ftype == "C":
            self.say(f"{d} C")
        elif ftype == "P":
            self.chunks[d].append(payload)
        elif ftype == "N":
            data = b"".join(self.chunks[d])
            self.chunks[d] = []
            self.pages[d].append(data)
            self.say(f"{d} page {len(self.pages[d])} complete, {len(data)} bytes "
                     f"({POST.get(self.text(payload), self.text(payload))})")
            self.save(d, f"_p{len(self.pages[d])}.t4", data)
        elif ftype in ("R", "D"):
            self.say(f"{d} {ftype} {self.text(payload)}")

    def path(self, d, ext):
        base = self.cfg.get("dir")
        if not base:
            return None
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f"{self.label.split()[-1]}_{d}{ext}")

    def save(self, d, ext, data):
        path = self.path(d, ext)
        if path:
            with open(path, "wb") as f:
                f.write(data)
        return path

    def close(self):
        if self.closed:
            return
        self.closed = True
        tifs = {}
        for d, pages in self.pages.items():
            if pages and self.cfg.get("dir"):
                path = self.path(d, ".tif")
                try:
                    fc.write_tiff(path, pages, self.caps)
                    tifs[d] = path
                except OSError as e:
                    self.say(f"tiff failed: {e}")
        n = {d: len(p) for d, p in self.pages.items()}
        base = self.cfg.get("dir")
        if base:
            os.makedirs(base, exist_ok=True)
            with open(os.path.join(base, f"{self.label.split()[-1]}.json"), "w") as f:
                json.dump({"pages": n, "tiff": tifs, "dropped": {d: x.dropped for d, x in self.df.items()},
                           "trace": self.trace}, f)
        self.say(f"summary pages a2b={n['a2b']} b2a={n['b2a']} frames={len(self.trace)} "
                 f"resync_dropped a2b={self.df['a2b'].dropped} b2a={self.df['b2a'].dropped}"
                 + (f" tiff={tifs.get('a2b')}" if tifs.get("a2b") else ""))


def load_watcher(cfg, label, log):
    """Instantiate the configured watcher: {"module": path, "class": name} or the default."""
    mod = cfg.get("module")
    if not mod:
        return FaxWatcher(cfg, label, log)
    import importlib.util
    spec = importlib.util.spec_from_file_location("custom_fax_watcher", mod)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, cfg.get("class", "Watcher"))(cfg, label, log)
