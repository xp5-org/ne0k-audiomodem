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

"""Python side of spanmodem (the spandsp helper, see spanmodem.c).

    sm = SpanModem(["--mode", "data", "--role", "caller"], on_audio=..., on_data=...,
                   on_status=..., on_done=...)
    await sm.start()
    sm.tick()               # every 20 ms of wall clock: helper answers with on_audio(bytes)
    sm.audio(payload)       # G.711 payload from RTP
    sm.dte(b"hello")        # serial bytes from the guest
    await sm.stop()

`PacedModem` adds the 20 ms clock, so a caller only has to feed it RTP payloads.
Everything here is asyncio; callbacks run in the loop.
"""
import asyncio
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "spanmodem")
SAMPLES = 160           # 20 ms at 8 kHz


def available():
    return os.access(BINARY, os.X_OK)


class SpanModem:
    def __init__(self, args, on_audio=None, on_data=None, on_status=None, on_done=None,
                 on_stderr=None, binary=BINARY):
        self.args = [binary] + list(args)
        self.on_audio = on_audio or (lambda b: None)
        self.on_data = on_data or (lambda b: None)
        self.on_status = on_status or (lambda s: None)
        self.on_done = on_done or (lambda r: None)
        self.on_stderr = on_stderr or (lambda s: None)
        self.proc = None
        self.result = None
        self._readers = []
        self.pending_ticks = 0

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(
            *self.args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True)
        self._readers = [asyncio.ensure_future(self._read_out()),
                         asyncio.ensure_future(self._read_err())]

    async def _read_out(self):
        try:
            while True:
                hdr = await self.proc.stdout.readexactly(3)
                n = (hdr[1] << 8) | hdr[2]
                pay = await self.proc.stdout.readexactly(n) if n else b""
                t = chr(hdr[0])
                if t == "A":
                    self.pending_ticks = max(0, self.pending_ticks - 1)
                    self.on_audio(pay)
                elif t == "D":
                    self.on_data(pay)
                elif t == "S":
                    self.on_status(pay.decode("latin-1"))
                elif t == "X":
                    self.result = pay.decode("latin-1")
                    self.on_done(self.result)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        if self.result is None:
            self.result = "exited"
            self.on_done(self.result)

    async def _read_err(self):
        try:
            while True:
                line = await self.proc.stderr.readline()
                if not line:
                    return
                self.on_stderr(line.decode("latin-1").rstrip())
        except ConnectionError:
            pass

    def _send(self, t, payload=b""):
        if self.proc is None or self.proc.stdin.is_closing():
            return
        try:
            self.proc.stdin.write(struct.pack("!cH", t.encode(), len(payload)) + payload)
        except (ConnectionError, RuntimeError):
            pass

    def tick(self, samples=SAMPLES):
        self.pending_ticks += 1
        self._send("K", struct.pack("!H", samples))

    def audio(self, payload):
        if payload:
            self._send("A", bytes(payload))

    def go(self, path):
        """faxtx --hold: start sending this TIFF now."""
        self._send("G", path.encode())

    def dte(self, data):
        for i in range(0, len(data), 2048):
            self._send("D", bytes(data[i:i + 2048]))

    async def stop(self):
        if self.proc is None:
            return
        self._send("Q")
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            await asyncio.wait_for(self.proc.wait(), 2)
        except asyncio.TimeoutError:
            try:
                os.killpg(self.proc.pid, 9)
            except ProcessLookupError:
                pass
            await self.proc.wait()
        for r in self._readers:
            r.cancel()
        self.proc = None


class PacedModem(SpanModem):
    """SpanModem plus the 20 ms wall clock. Ticks are skipped (not queued up) when the
    helper is more than a few packets behind, so a stalled helper cannot build a backlog."""

    def __init__(self, *a, tick_s=0.02, **kw):
        super().__init__(*a, **kw)
        self.tick_s = tick_s
        self._clock = None

    async def start(self):
        await super().start()
        self._clock = asyncio.ensure_future(self._run_clock())

    async def _run_clock(self):
        loop = asyncio.get_running_loop()
        nxt = loop.time()
        while True:
            nxt += self.tick_s
            await asyncio.sleep(max(0, nxt - loop.time()))
            if loop.time() - nxt > 0.5:         # long stall: do not burst to catch up
                nxt = loop.time()
            if self.pending_ticks < 5:
                self.tick()

    async def stop(self):
        if self._clock:
            self._clock.cancel()
        await super().stop()
