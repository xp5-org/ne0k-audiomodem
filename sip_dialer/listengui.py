#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""listengui.py -- testrunnerapp's "SIP call monitor" window meshgui.py, tkinter for a desktop that is NOT a test node.

Joins the mesh conference receive-only (sipbridge.py listen) and shows meshgui's window: call ACTIVE /
DISCONNECTED, RX/TX bit/s, latency (ping), dropped RTP frames, and the Monitor audio toggle.
Works on Windows/macOS/Linux; needs numpy, sounddevice and tkinte.

    python listengui.py                                   # mesh conference 3000 on the external SIP server
    python listengui.py --to HOST:PORT --number 3000

Monitor audio OFF/ON flips sipbridge's --mute-file; the call and the audio device stay open.
meshgui.py is the one in prereqs/meshsip; it is not copied.
"""
import argparse
import atexit
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BRIDGE = os.path.join(HERE, "sipbridge.py")


def find_meshgui(hint):
    for d in ([hint] if hint else []) + [os.path.join(HERE, p, "prereqs", "meshsip") for p in
                                         ("..", "../..", "../../..", "../../../..")] + [HERE]:
        if os.path.isfile(os.path.join(d, "meshgui.py")):
            return os.path.abspath(d)
    sys.exit("listengui: meshgui.py not found (it lives in <project>/prereqs/meshsip); pass --meshgui DIR")


class Listener:
    """Owns the one sipbridge listen subprocess for the whole session; Monitor audio only flips its mute file."""

    def __init__(self, a, stats):
        self.a, self.stats, self.proc = a, stats, None
        self.mute = os.path.join(os.path.dirname(stats), "mute")
        open(self.mute, "w").close()                         # start muted, like meshgui's OFF button
        self.mod = None                                      # truthy while audio is ON: meshgui's button reads this
        self.source = f"conference {a.number} @ {a.to}"
        argv = [sys.executable, BRIDGE, "listen", "--to", a.to, "--number", a.number,
                "--player", a.player, "--mute-file", self.mute, "--stats-file", stats]
        self.proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def toggle(self):
        if self.mod:
            open(self.mute, "w").close()
            self.mod = None
            return True, "off"
        try:
            os.remove(self.mute)
        except OSError:
            pass
        self.mod = "on"
        return True, "on: speakers"

    def close(self):
        self.stop()
        shutil.rmtree(os.path.dirname(self.stats), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--to", default="127.0.0.1:15064", help="SIP server HOST:PORT (default: the external mesh server)")
    ap.add_argument("--number", default="3000", help="conference number (default 3000)")
    ap.add_argument("--player", default="auto", help="sipbridge --player when Monitor audio is ON")
    ap.add_argument("--meshgui", help="directory containing meshgui.py")
    a = ap.parse_args()

    sys.path.insert(0, find_meshgui(a.meshgui))
    import meshgui

    if platform.system() == "Windows":
        def run(self):                                       # meshgui pings with Linux flags; Windows: -n 1 -w ms
            while True:
                try:
                    r = subprocess.run(["ping", "-n", "1", "-w", "1000", self.host], capture_output=True, text=True)
                    m = re.search(r"time[=<]\s*([\d.]+)\s*ms", r.stdout)
                    self.rtt = float(m.group(1)) if m else None
                except OSError:
                    self.rtt = None
                    return
                time.sleep(2)
        meshgui.Pinger.run = run

    stats_dir = tempfile.mkdtemp(prefix="listengui_")
    lis = Listener(a, os.path.join(stats_dir, "listen-conference_call.json"))
    atexit.register(lis.close)
    g = argparse.Namespace(stats_dir=stats_dir, node="listen", role="", prefix="mesh", direction="rx",
                           ping_host=a.to.rsplit(":", 1)[0])
    app = meshgui.App(g)
    app.mon = lis
    app.root.title(f"SIP call monitor - listening to conference {a.number}")
    app.note.config(text=app.mon.source)
    app.root.mainloop()
    lis.close()


if __name__ == "__main__":
    main()
