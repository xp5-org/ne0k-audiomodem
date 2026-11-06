# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Amateur-radio station identification ("ham mode").

Two timers drive it, both on the TX path in ``Session.play`` so CSS, OFDM and mesh all get it:

* ``id_interval_s``  -- no more than this long between IDs while transmitting.  Checked on every
  outgoing frame; the first frame of a session is an ID too.
* ``id_end_delay_s`` -- after the last frame, once the channel has been quiet this long, an ID
  is sent on its own (the "end of transmission" ID).  0 turns it off.

The ID itself is sent as ``id_mode``:

* ``cw``    -- Morse tone *after* the frame, separated by ``cw_gap_s`` of silence so the peer's burst
  detector ends the data burst first and sees the beeps as a burst of their own.
* ``ascii`` -- binary-coded ASCII in front of the frame, as AFSK (8N1, LSB first):
  ``SYN SYN STX "DE <CALL>" ETX``, then ``ascii_gap_s`` of silence, then the normal framing.
* ``both``  -- both of the above.

No Qt here: the UI and the CLI edit ``HamConfig`` and hand it over.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .modemcfg import HamConfig

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.", "H": "....",
    "I": "..", "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---", "P": ".--.",
    "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..", "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.", "/": "-..-.", "-": "-....-",
    "?": "..--..", ".": ".-.-.-",
}
SYN, STX, ETX = 0x16, 0x02, 0x03
ASCII_MARK_HZ, ASCII_SPACE_HZ = 2200.0, 1200.0       # Bell 202 pair: inside a phone line
LEADER_S = 0.1
KEYUP_TONE_HZ = 1000.0      # the steady lead tone OFDM puts before a frame (ofdm_link.keyup): never an ID
CW_TONE_TOL_HZ = 100.0      # a CW ID must sit this close to the configured cw_tone_hz
MIN_MARK_S = 0.015          # shortest key-down decode_cw takes as an element
HEARD_DEDUP_S = 5.0         # several listeners (announce, one link per peer) see the same burst


def id_text(call: str) -> str:
    return f"DE {call.strip().upper()}"


def cw_audio(text: str, fs: int, wpm: float, tone_hz: float, amp: float) -> np.ndarray:
    """Morse keying with 5 ms raised-cosine edges (no key clicks).  PARIS timing: dot = 1.2/wpm."""
    dot = int(round(1.2 / max(wpm, 1.0) * fs))
    ramp = min(int(0.005 * fs), dot // 2)
    edge = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / max(ramp, 1))

    def tone(n: int) -> np.ndarray:
        t = np.sin(2 * np.pi * tone_hz * np.arange(n) / fs)
        env = np.ones(n)
        env[:ramp], env[n - ramp:] = edge, edge[::-1]
        return amp * t * env

    out: list[np.ndarray] = []
    for w, word in enumerate(text.upper().split()):
        if w:
            out.append(np.zeros(7 * dot))
        for ci, ch in enumerate(word):
            code = MORSE.get(ch)
            if not code:
                continue
            if ci:
                out.append(np.zeros(3 * dot))
            for si, sym in enumerate(code):
                if si:
                    out.append(np.zeros(dot))
                out.append(tone(dot if sym == "." else 3 * dot))
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def ascii_bytes(call: str) -> bytes:
    return bytes([SYN, SYN, STX]) + id_text(call).encode("ascii", "replace") + bytes([ETX])


def ascii_audio(call: str, fs: int, baud: float, amp: float) -> np.ndarray:
    bits: list[int] = []
    for b in ascii_bytes(call):
        bits += [0] + [(b >> i) & 1 for i in range(8)] + [1]     # start, 8 data LSB first, stop
    spb = fs / baud
    idx = np.minimum((np.arange(int(len(bits) * spb)) / spb).astype(int), len(bits) - 1)
    levels = np.concatenate([np.ones(int(LEADER_S * fs), np.int8), np.array(bits, np.int8)[idx],
                             np.ones(int(0.5 * spb), np.int8)])
    freq = np.where(levels == 1, ASCII_MARK_HZ, ASCII_SPACE_HZ)
    phase = 2 * np.pi * np.cumsum(freq) / fs                    # continuous phase across bits
    return (amp * np.sin(phase)).astype(np.float32)


def decode_ascii(pcm: np.ndarray, fs: int, baud: float) -> str | None:
    """Recover the ID text from ``ascii_audio`` output (or the same heard back off the air)."""
    spb = fs / baud

    def bit(pos: int, n: int) -> int:
        """Mark or space over the n samples at ``pos`` (1 past the end of the audio)."""
        seg = pcm[max(pos, 0):pos + n]
        if len(seg) < n:
            return 1
        k = np.arange(n)
        mark = abs(np.dot(seg, np.exp(-2j * np.pi * ASCII_MARK_HZ * k / fs)))
        space = abs(np.dot(seg, np.exp(-2j * np.pi * ASCII_SPACE_HZ * k / fs)))
        return int(mark >= space)

    quarter, step = max(int(spb / 4), 8), max(int(spb / 16), 1)
    pos, out = 0, bytearray()
    while pos + quarter < len(pcm):
        if bit(pos, quarter) == 1:                      # idle (mark): keep looking for a start bit
            pos += step
            continue
        edge = pos
        while edge > 0 and bit(edge - step, quarter) == 0:
            edge -= step
        word = [bit(edge + int((i + 0.25) * spb), max(int(spb / 2), 8)) for i in range(10)]
        if word[0] != 0 or word[9] != 1:
            pos += step
            continue
        out.append(sum(b << i for i, b in enumerate(word[1:9])))
        pos = edge + int(9.5 * spb)
        if out[-1] == ETX:
            break
    if STX in out and ETX in out:
        return out[out.index(STX) + 1:out.index(ETX)].decode("ascii", "replace")
    return None


def decode_cw(pcm: np.ndarray, fs: int, tone_hz: float | None = None) -> str | None:
    """Read Morse out of a burst.  The dot length is the shortest key-down; the burst must be a
    narrow tone (anything wide-band, such as a modem frame, is not CW)."""
    if len(pcm) < fs // 4:
        return None
    spec = np.abs(np.fft.rfft(pcm * np.hanning(len(pcm)))) ** 2
    freqs = np.fft.rfftfreq(len(pcm), 1 / fs)
    peak = freqs[int(np.argmax(spec))]
    near = spec[np.abs(freqs - peak) <= 60].sum() / max(spec.sum(), 1e-12)
    if near < 0.7 or (tone_hz and abs(peak - tone_hz) > CW_TONE_TOL_HZ):
        return None
    # A keyup lead tone (or two of them beating against each other on a shared line) is narrow too, and its
    # envelope fluctuations read as "E"/"T" strings.  Unless the CW tone itself is set to that pitch, a tone
    # at the keyup pitch is not an ID.
    if abs(peak - KEYUP_TONE_HZ) <= 50 and not (tone_hz and abs(tone_hz - KEYUP_TONE_HZ) <= 50):
        return None
    w = max(int(0.004 * fs), 1)
    env = np.convolve(np.abs(pcm), np.ones(w) / w, "same")
    on = env > 0.25 * env.max()
    edges = np.flatnonzero(np.diff(on.astype(int))) + 1
    runs = np.diff(np.concatenate([[0], edges, [len(on)]]))
    state = bool(on[0])
    # a key-down shorter than any real CW element (50 wpm dot = 24 ms) is a click or the edge of the next burst's
    # keyup tone; the dot length is the shortest key-down, so one such blip read every element as a dash ("T T T")
    min_mark = int(MIN_MARK_S * fs)
    marks = [r for i, r in enumerate(runs) if (i % 2 == 0) == state and r >= min_mark]
    if not marks:
        return None
    dot = min(marks)
    inv = {v: k for k, v in MORSE.items()}
    out, code = "", ""
    for i, r in enumerate(runs):
        if state and r < min_mark:
            pass
        elif state:
            code += "." if r < 2 * dot else "-"
        elif code and r >= 2 * dot:
            out += inv.get(code, "?"); code = ""
            if r >= 5 * dot:
                out += " "
        state = not state
    if code:
        out += inv.get(code, "?")
    return out.strip() or None


MAX_ID_BURST_S = 30.0


def heard_ident(pcm: np.ndarray, fs: int, bauds=(300.0, 110.0, 600.0), tone_hz: float | None = None):
    """Is this burst another station's ID?  ('ascii'|'cw', text) or None.  Cheap for anything
    that is not: long bursts are skipped and ASCII needs the mark-tone leader first."""
    n = len(pcm)
    if n < fs // 4 or n > MAX_ID_BURST_S * fs:
        return None
    pcm = np.asarray(pcm, np.float32)
    lead = pcm[int(0.02 * fs):int(0.02 * fs) + int(0.06 * fs)]
    if len(lead) == int(0.06 * fs):
        k = np.arange(len(lead))
        pm = abs(np.dot(lead, np.exp(-2j * np.pi * ASCII_MARK_HZ * k / fs)))
        ps = abs(np.dot(lead, np.exp(-2j * np.pi * ASCII_SPACE_HZ * k / fs)))
        if pm > 4 * ps and pm > 0.2 * len(lead) * float(np.abs(lead).max()):
            for b in bauds:
                t = decode_ascii(pcm, fs, b)
                if t:
                    return "ascii", t
    t = decode_cw(pcm, fs, tone_hz)
    return ("cw", t) if t else None


class Identifier:
    """Owned by a Session.  ``wrap`` is called with every outgoing frame; a background thread
    sends the end-of-transmission ID.

    ``channel_busy`` is a hook for a MAC layer that shares the medium with other transmitters
    (mesh CSMA): the background timer defers the standalone end-of-transmission ID while it
    returns True, rather than keying over someone else's traffic.  Unset (the default) for a
    point-to-point link with nothing else on the channel."""

    def __init__(self, session, cfg: HamConfig | None = None, log=lambda m: None) -> None:
        self.session, self.cfg, self.log = session, cfg or HamConfig(), log
        self.channel_busy = lambda: False
        self._lock = threading.Lock()
        self._last_id: float | None = None
        self._owed = False
        self._tx_end = 0.0
        self.heard: list[tuple[float, str, str]] = []       # (monotonic, kind, text) from other stations
        self.sent: list[tuple[float, str, str]] = []        # (monotonic, mode, reason)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- audio ---------------------------------------------------------------

    def _fs(self) -> int:
        return self.session.spec.rate

    def _active(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.callsign.strip())

    def cw(self) -> np.ndarray:
        c = self.cfg
        return cw_audio(id_text(c.callsign), self._fs(), c.cw_wpm, c.cw_tone_hz, c.id_amplitude)

    def ascii(self) -> np.ndarray:
        c = self.cfg
        return ascii_audio(c.callsign, self._fs(), c.ascii_baud, c.id_amplitude)

    def _gap(self, s: float) -> np.ndarray:
        return np.zeros(int(s * self._fs()), np.float32)

    def standalone(self) -> np.ndarray:
        """The ID by itself: what the end-of-transmission and manual IDs send."""
        m, parts = self.cfg.id_mode, []
        if m in ("ascii", "both"):
            parts += [self.ascii()]
        if m == "both":
            parts += [self._gap(self.cfg.ascii_gap_s)]
        if m in ("cw", "both"):
            parts += [self.cw()]
        return np.concatenate(parts)

    # -- TX path -------------------------------------------------------------

    def due(self) -> bool:
        """Timer 1: an ID is owed (nothing sent yet, or the interval has run out)."""
        return self._active() and (self._last_id is None or
                                   time.monotonic() - self._last_id >= self.cfg.id_interval_s)

    def mark_sent(self, reason: str, samples: int) -> None:
        """Book an ID the caller is about to play itself (OFDM window slots)."""
        now = time.monotonic()
        with self._lock:
            self._last_id, self._owed = now, False
            self._tx_end = max(self._tx_end, now) + samples / self._fs()
            self.sent.append((now, self.cfg.id_mode, reason))
        self.log(f"ident {self.cfg.callsign.upper()} sent ({self.cfg.id_mode}, {reason})")

    def wrap(self, samples: np.ndarray, inline: bool = True) -> np.ndarray:
        """Put the ID around ``samples`` when the interval timer says one is due.  ``inline=False``
        is for timed protocols that cannot take extra audio: the frame is only booked as traffic
        (so the end-of-transmission ID still follows) and the caller finds its own slot."""
        if not self._active():
            return samples
        now = time.monotonic()
        with self._lock:
            due = inline and (self._last_id is None or now - self._last_id >= self.cfg.id_interval_s)
            if not due:
                self._owed = True
                out = samples
            else:
                m, c = self.cfg.id_mode, self.cfg
                pre = np.concatenate([self.ascii(), self._gap(c.ascii_gap_s)]) if m in ("ascii", "both") else None
                post = np.concatenate([self._gap(c.cw_gap_s), self.cw()]) if m in ("cw", "both") else None
                out = np.concatenate([x for x in (pre, samples, post) if x is not None])
                self._last_id, self._owed = now, False
                self.sent.append((now, m, "interval"))
                self.log(f"ident {c.callsign.upper()} sent ({m}, with frame)")
            self._tx_end = max(self._tx_end, now) + len(out) / self._fs()
        return out

    def send_now(self, reason: str = "manual") -> float:
        """Transmit the ID on its own.  Returns its airtime."""
        if not self.cfg.callsign.strip():
            raise ValueError("no callsign set")
        pcm = self.standalone()
        self.mark_sent(reason, len(pcm))
        self.session.play_raw(pcm)
        return len(pcm) / self._fs()

    # -- receive ----------------------------------------------------------------

    def note_heard(self, burst: np.ndarray, fs: int, emit=None, tail_from: int | None = None) -> bool:
        """True when ``burst`` is another station's ID (logged, remembered, not for the modem).
        Runs whether or not our own ham mode is on: hearing an ID costs nothing.

        ``tail_from``: OFDM and CSS never intermix, so when the caller knows ``burst`` could be
        its OWN (or anyone's) keyup tone glued onto a wideband OFDM frame (see ofdm_link.keyup,
        KEYUP_LEAD_S), it passes the sample index where that tone could plausibly end.
        decode_cw's narrowband spectral-concentration check can be fooled by the tone's
        energy for the WHOLE burst even though the frame after it is wideband; requiring the
        tail past that index to look narrowband on its own too is what tells a real CW/ASCII id
        apart from a keyup artifact -- a genuine id is never keyup-prefixed, so this never clips
        one short."""
        tone_hz = self.cfg.cw_tone_hz if self.cfg.enabled else None
        hit = heard_ident(burst, fs, tone_hz=tone_hz)
        if hit and tail_from is not None and tail_from < len(burst):
            if not heard_ident(burst[tail_from:], fs, tone_hz=tone_hz):
                return False
        if not hit:
            return False
        kind, text = hit
        now = time.monotonic()
        with self._lock:
            if any(k == kind and t == text and now - ts < HEARD_DEDUP_S for ts, k, t in self.heard):
                return True                        # the same burst through another listener: already logged
            self.heard.append((now, kind, text))
        self.log(f"ident heard ({kind}): {text}")
        if emit:
            emit(f"ID heard: {text} ({kind})")
        return True

    # -- end-of-transmission timer ---------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        if not self._thread or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._timer, name="ident-timer", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2.0)
            self._thread = None

    def reset(self) -> None:
        with self._lock:
            self._last_id, self._owed, self._tx_end = None, False, 0.0

    def _timer(self) -> None:
        while not self._stop.wait(0.25):
            c = self.cfg
            if not (self._active() and c.id_end_delay_s > 0):
                continue
            with self._lock:
                fire = self._owed and time.monotonic() >= self._tx_end + c.id_end_delay_s
            if fire:
                if self.channel_busy():                  # someone else is on the air: wait, don't key over it
                    continue
                try:
                    self.send_now("end of transmission")
                except Exception as e:                  # audio closed under us: try again next lap
                    self.log(f"ident failed: {e}")
