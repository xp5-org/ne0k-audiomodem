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

"""Shared fax pieces for modemmock.py (the DTE-facing Class 2 modem) and sipserver.py (the
`fax` line that plays the remote fax machine). Pure Python, no third-party modules.

SERIAL SIDE -- DLE shielding. Page data crosses the serial port as T.4 bytes with the
modem-standard escapes: DLE DLE is one data byte 0x10, DLE ETX ends the data, DLE SUB is two
data bytes 0x10 (Class 1 convention), other DLE <x> pairs are control codes we ignore.
`DleDecoder` is the DTE -> modem direction, `dle_encode` the modem -> DTE one.

WIRE SIDE -- the two fax ends do not exchange audio or HDLC; the mock IS the modem and the
server line IS the far fax machine, so they speak a small private framed protocol inside the
RTP "x-modem-bytes" payload stream (which is what the server's delay / jitter / loss rules act on):

    [0xFA][type][len: 2 bytes BE][payload][checksum: 1 byte]    checksum = sum(type+len+payload) & 0xFF

  I  station id text (the answerer's CSI, the caller's TSI)
  A  the answerer's capabilities (DIS):  "vr,br,wd,ln,df,ec,bf,st"
  S  the sender's chosen parameters (DCS), same format
  C  the answerer is ready for the page: "confirmation to receive" (CFR). The sender's modem
     holds its CONNECT until this arrives, so page data never races ahead of the receiving DTE
  P  a chunk of page data (T.4 bytes exactly as the DTE sent them, bit order BOR from S)
  N  end of page: one ASCII digit, 0 = more pages (MPS), 1 = new document (EOM), 2 = last page (EOP)
  R  page response: one ASCII digit, the +FPTS value (1 good, 2 bad / retrain, 3 good but retrain)
  D  disconnect (DCN), payload one ASCII digit, the +FHNG code (0 = normal)

`Deframer` resynchronises on the 0xFA magic and drops frames with a bad checksum, so a lost or
reordered RTP packet (a delay rule with loss_pct) costs a frame instead of wedging the stream;
the protocol timers then do what T.30's do.

TIFF -- `write_tiff` wraps received T.4 data (CCITT Group 3 1D) in a TIFF so ordinary tools can
open it. G3 data does not carry its own length, so the line count is found by counting EOL codes.
"""
import struct

DLE, ETX, SUB, XON, XOFF, DC2 = 0x10, 0x03, 0x1A, 0x11, 0x13, 0x12

MAGIC = 0xFA
FRAME_TYPES = "IASCPNRD"

# +FDIS / +FDCC field order and the pixel width of each WD value (ITU T.30 / TR-29.2)
FIELDS = ("vr", "br", "wd", "ln", "df", "ec", "bf", "st")
WIDTHS = {0: 1728, 1: 2048, 2: 2432, 3: 1216, 4: 864}


# ---- serial side -----------------------------------------------------------

def dle_encode(data):
    return bytes(data).replace(b"\x10", b"\x10\x10")


class DleDecoder:
    """DTE -> modem direction. feed() returns a list of ("data", bytes) and ("end",) events;
    an escape split across two reads is carried over."""

    def __init__(self):
        self.pending = False

    def feed(self, chunk):
        out, buf = [], bytearray()
        for b in chunk:
            if self.pending:
                self.pending = False
                if b == DLE:
                    buf.append(DLE)
                elif b == SUB:
                    buf.extend((DLE, DLE))
                elif b == ETX:
                    if buf:
                        out.append(("data", bytes(buf)))
                        buf.clear()
                    out.append(("end",))
                # any other DLE <x> is a control code: ignored
            elif b == DLE:
                self.pending = True
            else:
                buf.append(b)
        if buf:
            out.append(("data", bytes(buf)))
        return out


# ---- wire side -------------------------------------------------------------

def pack(ftype, payload=b""):
    if isinstance(payload, str):
        payload = payload.encode("latin-1", "replace")
    if ftype not in FRAME_TYPES or len(payload) > 0xFFFF:
        raise ValueError("bad fax frame")
    body = ftype.encode() + struct.pack(">H", len(payload)) + payload
    return bytes([MAGIC]) + body + bytes([sum(body) & 0xFF])


class Deframer:
    """feed(bytes) -> [(type, payload)]. `dropped` counts bytes skipped while resyncing."""

    def __init__(self):
        self.buf = bytearray()
        self.dropped = 0

    def feed(self, data):
        self.buf.extend(data)
        out = []
        while True:
            while self.buf and self.buf[0] != MAGIC:
                del self.buf[0]
                self.dropped += 1
            if len(self.buf) < 5:
                return out
            ftype = chr(self.buf[1])
            n = struct.unpack(">H", self.buf[2:4])[0]
            if ftype not in FRAME_TYPES:
                del self.buf[0]
                self.dropped += 1
                continue
            if len(self.buf) < 5 + n:
                return out
            body = bytes(self.buf[1:4 + n])
            if sum(body) & 0xFF != self.buf[4 + n]:
                del self.buf[0]
                self.dropped += 1
                continue
            out.append((ftype, bytes(self.buf[4:4 + n])))
            del self.buf[:5 + n]


def parse_caps(text):
    """"1,3,0,2,0,0,0,0" -> dict of ints (missing trailing fields are 0)."""
    vals = [int(v) if v.strip().lstrip("-").isdigit() else 0 for v in str(text).split(",")]
    vals += [0] * (len(FIELDS) - len(vals))
    return dict(zip(FIELDS, vals))


def caps_text(caps):
    return ",".join(str(int(caps.get(f, 0))) for f in FIELDS)


def intersect(dte, answerer):
    """What the sender may use: never faster / wider / finer than the answerer offered."""
    d, a = parse_caps(dte), parse_caps(answerer)
    out = dict(d)
    out["br"] = min(d["br"], a["br"])
    out["vr"] = min(d["vr"], a["vr"])
    out["wd"] = d["wd"] if d["wd"] <= a["wd"] else a["wd"]
    out["ec"], out["bf"] = 0, 0
    return out


# ---- T.4 data and TIFF -----------------------------------------------------

def _reverse_bits(data):
    return bytes(int(f"{b:08b}"[::-1], 2) for b in data)


def count_eols(data):
    """Number of EOL codes (eleven 0 bits then a 1) in an MSB-first bit stream, with the
    fill zeros before an EOL absorbed. Used as the line count of a 1D G3 page."""
    n, zeros = 0, 0
    for byte in data:
        for i in range(7, -1, -1):
            if (byte >> i) & 1:
                if zeros >= 11:
                    n += 1
                zeros = 0
            else:
                zeros += 1
    return n


def detect_fill_order(data):
    """1 = the bytes are MSB-first (TIFF FillOrder 1), 2 = LSB-first. A valid G3 stream is
    dense with EOLs only in the right order, so count both ways and take the larger."""
    if not data:
        return 1
    return 1 if count_eols(data[:65536]) >= count_eols(_reverse_bits(data[:65536])) else 2


def _ifd(entries, next_off=0):
    entries = sorted(entries)
    return struct.pack("<H", len(entries)) + b"".join(struct.pack("<HHII", *e) for e in entries) \
        + struct.pack("<I", next_off)


def write_tiff(path, pages, caps=None, fill_order=None):
    """pages: list of bytes (T.4 data per page). Little-endian, one strip per page,
    Compression 3 (CCITT G3), WhiteIsZero. Returns {"pages": n, "width": w, "lines": [..]}."""
    c = parse_caps(caps) if isinstance(caps, str) else dict(parse_caps("1,3,0,2,0,0,0,0"), **(caps or {}))
    width = WIDTHS.get(c["wd"], 1728)
    yres = 196 if c["vr"] else 98            # lines per inch: fine / normal
    info = {"pages": len(pages), "width": width, "lines": [], "fill_order": []}
    body, ifds, pos = bytearray(), [], 8
    # layout: header(8) | strip data and resolution rationals for every page | IFDs
    placed = []
    for data in pages:
        fo = fill_order or detect_fill_order(data)
        lines = max(1, count_eols(data if fo == 1 else _reverse_bits(data)))
        info["lines"].append(lines)
        info["fill_order"].append(fo)
        off = 8 + len(body)
        body.extend(data)
        if len(body) % 2:
            body.append(0)
        res_off = 8 + len(body)
        body.extend(struct.pack("<IIII", 204, 1, yres, 1))
        placed.append((off, len(data), res_off, lines, fo))
    ifd_pos = 8 + len(body)
    out = bytearray(b"II*\x00" + struct.pack("<I", ifd_pos) + body)
    for i, (off, size, res_off, lines, fo) in enumerate(placed):
        entries = [
            (254, 4, 1, 0),
            (256, 4, 1, width), (257, 4, 1, lines), (258, 3, 1, 1), (259, 3, 1, 3),
            (262, 3, 1, 0), (266, 3, 1, fo), (273, 4, 1, off), (277, 3, 1, 1),
            (278, 4, 1, lines), (279, 4, 1, size), (282, 5, 1, res_off), (283, 5, 1, res_off + 8),
            (292, 4, 1, 0), (296, 3, 1, 2),
        ]
        nxt = ifd_pos + 2 + 12 * len(entries) + 4 if i + 1 < len(placed) else 0
        out.extend(_ifd(entries, nxt))
        ifd_pos += 2 + 12 * len(entries) + 4
    with open(path, "wb") as f:
        f.write(out)
    return info
