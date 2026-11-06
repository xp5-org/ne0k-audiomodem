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

"""Minimal SIP/SDP message handling shared by sipserver.py and the modem mock.

Only what a UDP test rig needs: parse a datagram into a SipMessage, build
requests/responses, and read/write the one SDP shape this rig negotiates
(a single audio m-line carrying the private "x-modem-bytes" payload plus
RFC 4733 telephone-event). No TCP/TLS, no auth, no transaction layer -- the
callers do their own retransmits.
"""
import re
import random
import string

CRLF = "\r\n"
COMPACT = {"v": "via", "f": "from", "t": "to", "i": "call-id",
           "m": "contact", "l": "content-length", "c": "content-type"}
DATA_CODEC = "x-modem-bytes"
DTMF_CODEC = "telephone-event"
USER_AGENT = "bughabitat-sipserver/1"


class SipMessage:
    def __init__(self, first, headers, body):
        self.first = first
        self.headers = headers          # [(lname, name, value)] in wire order
        self.body = body                # str
        parts = first.split(" ", 2)
        self.is_request = not first.startswith("SIP/2.0")
        if self.is_request:
            self.method = parts[0].upper()
            self.uri = parts[1] if len(parts) > 1 else ""
            self.status, self.reason = None, None
        else:
            self.method, self.uri = None, None
            self.status = int(parts[1])
            self.reason = parts[2] if len(parts) > 2 else ""

    def get(self, name, default=None):
        n = name.lower()
        for ln, _, v in self.headers:
            if ln == n:
                return v
        return default

    def get_all(self, name):
        n = name.lower()
        return [v for ln, _, v in self.headers if ln == n]

    def cseq(self):
        m = re.match(r"\s*(\d+)\s+(\S+)", self.get("cseq", ""))
        return (int(m.group(1)), m.group(2).upper()) if m else (0, "")

    @property
    def call_id(self):
        return self.get("call-id", "")


def parse(data):
    """bytes -> SipMessage, or None if it is not a SIP message."""
    try:
        text = data.decode("utf-8", errors="replace")
    except AttributeError:
        text = data
    head, sep, body = text.partition("\r\n\r\n")
    if not sep:
        head, sep, body = text.partition("\n\n")
    lines = head.replace("\r\n", "\n").split("\n")
    first = lines[0].strip()
    if not first or not (first.startswith("SIP/2.0") or first.endswith("SIP/2.0")):
        return None
    unfolded = []
    for ln in lines[1:]:
        if ln[:1] in (" ", "\t") and unfolded:
            unfolded[-1] += " " + ln.strip()
        elif ln.strip():
            unfolded.append(ln)
    headers = []
    for ln in unfolded:
        name, colon, value = ln.partition(":")
        if not colon:
            continue
        name = name.strip()
        lname = COMPACT.get(name.lower(), name.lower())
        headers.append((lname, name, value.strip()))
    try:
        return SipMessage(first, headers, body)
    except (ValueError, IndexError):
        return None


# ---- small header helpers -------------------------------------------------

def get_tag(value):
    m = re.search(r";\s*tag=([^;>\s]+)", value or "")
    return m.group(1) if m else None


def uri_of(value):
    """The URI inside a name-addr or addr-spec header value."""
    m = re.search(r"<([^>]+)>", value or "")
    if m:
        return m.group(1)
    return (value or "").split(";")[0].strip()


def user_of(uri):
    m = re.match(r"\s*sips?:([^@;>]+)@", uri or "")
    return m.group(1) if m else ""


def new_token(n=10):
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def new_branch():
    return "z9hG4bK" + new_token(12)


# ---- builders ---------------------------------------------------------------

def _finish(lines, body):
    body = body or ""
    lines.append(f"Content-Length: {len(body.encode())}")
    return (CRLF.join(lines) + CRLF + CRLF + body).encode()


def build_response(req, status, reason, to_tag=None, extra=(), body=None,
                   content_type="application/sdp"):
    lines = [f"SIP/2.0 {status} {reason}"]
    for v in req.get_all("via"):
        lines.append(f"Via: {v}")
    lines.append(f"From: {req.get('from', '')}")
    to = req.get("to", "")
    if to_tag and not get_tag(to):
        to = f"{to};tag={to_tag}"
    lines.append(f"To: {to}")
    lines.append(f"Call-ID: {req.call_id}")
    lines.append(f"CSeq: {req.get('cseq', '')}")
    lines.extend(extra)
    lines.append(f"Server: {USER_AGENT}")
    if body:
        lines.append(f"Content-Type: {content_type}")
    return _finish(lines, body)


def build_request(method, uri, via_hostport, from_hdr, to_hdr, call_id, cseq,
                  contact=None, body=None, content_type="application/sdp",
                  extra=()):
    lines = [f"{method} {uri} SIP/2.0",
             f"Via: SIP/2.0/UDP {via_hostport};branch={new_branch()};rport",
             "Max-Forwards: 70",
             f"From: {from_hdr}",
             f"To: {to_hdr}",
             f"Call-ID: {call_id}",
             f"CSeq: {cseq} {method}"]
    if contact:
        lines.append(f"Contact: {contact}")
    lines.extend(extra)
    lines.append(f"User-Agent: {USER_AGENT}")
    if body:
        lines.append(f"Content-Type: {content_type}")
    return _finish(lines, body)


# ---- SDP ----------------------------------------------------------------------

def build_sdp(addr, port, data_pt=96, dtmf_pt=101, session_id=None):
    sid = session_id or random.randint(10 ** 9, 10 ** 10 - 1)
    pts = [str(data_pt)] + ([str(dtmf_pt)] if dtmf_pt is not None else [])
    lines = ["v=0",
             f"o=- {sid} {sid} IN IP4 {addr}",
             "s=modem-call",
             f"c=IN IP4 {addr}",
             "t=0 0",
             f"m=audio {port} RTP/AVP {' '.join(pts)}",
             f"a=rtpmap:{data_pt} {DATA_CODEC}/8000"]
    if dtmf_pt is not None:
        lines += [f"a=rtpmap:{dtmf_pt} {DTMF_CODEC}/8000", f"a=fmtp:{dtmf_pt} 0-15"]
    lines += ["a=ptime:20", "a=sendrecv"]
    return CRLF.join(lines) + CRLF


G711 = {0: ("PCMU", "ulaw"), 8: ("PCMA", "alaw")}


def build_sdp_g711(addr, port, pts=(0, 8), dtmf_pt=101, session_id=None):
    """SDP for real audio (spandsp media): G.711 u-law / A-law plus telephone-event.
    pts is the preference order for an offer; an answer passes the one it chose."""
    sid = session_id or random.randint(10 ** 9, 10 ** 10 - 1)
    allp = [str(p) for p in pts] + ([str(dtmf_pt)] if dtmf_pt is not None else [])
    lines = ["v=0",
             f"o=- {sid} {sid} IN IP4 {addr}",
             "s=modem-call",
             f"c=IN IP4 {addr}",
             "t=0 0",
             f"m=audio {port} RTP/AVP {' '.join(allp)}"]
    lines += [f"a=rtpmap:{p} {G711[p][0]}/8000" for p in pts]
    if dtmf_pt is not None:
        lines += [f"a=rtpmap:{dtmf_pt} {DTMF_CODEC}/8000", f"a=fmtp:{dtmf_pt} 0-15"]
    lines += ["a=ptime:20", "a=sendrecv"]
    return CRLF.join(lines) + CRLF


def parse_sdp(text):
    """-> {"addr", "port", "data_pt", "dtmf_pt"} (plus "g711": True, "law" when the offer
    has real audio instead of the byte pipe) or None when the offer has no audio line,
    no usable address, or neither an x-modem-bytes nor a G.711 payload."""
    addr, port, offered, names = None, None, [], {}
    for ln in (text or "").replace("\r\n", "\n").split("\n"):
        ln = ln.strip()
        if ln.startswith("c=") and addr is None:
            parts = ln[2:].split()
            if len(parts) >= 3:
                addr = parts[2].split("/")[0]
        elif ln.startswith("m=audio") and port is None:
            parts = ln.split()
            try:
                port = int(parts[1])
            except (IndexError, ValueError):
                return None
            offered = [int(p) for p in parts[3:] if p.isdigit()]
        elif ln.startswith("a=rtpmap:"):
            m = re.match(r"a=rtpmap:(\d+)\s+([^/\s]+)", ln)
            if m:
                names[int(m.group(1))] = m.group(2).lower()
    if addr is None or port is None:
        return None
    data_pt = next((p for p in offered if names.get(p) == DATA_CODEC), None)
    dtmf_pt = next((p for p in offered if names.get(p) == DTMF_CODEC), None)
    if data_pt is None:
        # no byte pipe: take real G.711 audio (static payload types 0 / 8 need no rtpmap)
        g = next((p for p in offered if p in G711), None)
        if g is None:
            return None
        return {"addr": addr, "port": port, "data_pt": g, "dtmf_pt": dtmf_pt,
                "g711": True, "law": G711[g][1]}
    return {"addr": addr, "port": port, "data_pt": data_pt, "dtmf_pt": dtmf_pt}
