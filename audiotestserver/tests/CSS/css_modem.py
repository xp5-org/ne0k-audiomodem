# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_modem.py — Modem layer riding on top of the CSS physical layer.

The physical layer (css_encoder.encode / css_decoder.decode) owns chirps,
preamble, symbol timing, per-PHY-frame sequencing, CRC and optional FEC.  It is
fed OPAQUE payload bytes and is left untouched by this module.

This modem layer owns everything above the wire:

  * addressing        — 4-bit source / destination device IDs (16 nodes)
  * message typing     — BEACON / DATA / ACK / RANGE_REQ / RANGE_REPLY
  * application seq     — 4-bit sequence number (dedup, ranging correlation)
  * ranging payloads    — the fixed turnaround value a RANGE_REPLY carries back

A modem packet is the byte string handed to encode() as its ``message`` and
recovered from decode() as a frame's raw bytes.

Packet layout (2-byte header, variable payload)
-----------------------------------------------
    byte 0:  [ src_id : 4 | dst_id : 4 ]     dst 0xF = broadcast
    byte 1:  [ type   : 4 | seq    : 4 ]
    byte 2+: payload (0..MAX_PAYLOAD bytes)

Nibble packing keeps the on-air frame as short as possible: a header-only
packet is just two bytes, which matters because every extra byte is more
symbols, more airtime, and one more chance for an error on a channel with no
retransmit guarantee.

This file has NO audio / numpy dependency on purpose: it is pure bytes so the
packet format can be unit-tested (and ported to a constrained device) without
pulling in the DSP stack.  Run ``python css_modem.py`` for the self-test.
"""

from __future__ import annotations

# ── Message types (the 4-bit `type` nibble; 16 possible, 5 defined) ──────────
TYPE_BEACON       = 0x0   # "I am here" — full-preamble presence / wake
TYPE_DATA         = 0x1   # application payload for dst (or broadcast)
TYPE_ACK          = 0x2   # acknowledge a received DATA (payload = acked seq)
TYPE_RANGE_REQ    = 0x3   # ranging ping (initiator marks its own TX time)
TYPE_RANGE_REPLY  = 0x4   # ranging echo (payload carries the turnaround value)
# 0x5 reserved for FWD (store-and-forward relay) — deferred, see design notes.
TYPE_OFDM_HS_REQ  = 0x6   # "become OFDM handshake responder now" — the CSS
                          # control plane arranges the high-rate negotiation
                          # (V.8-style); the addressed peer ACKs, then both
                          # drop into the in-band OFDM handshake proper.
# BLOCKOUT: an exclusive, scheduled quiet window for one peer-to-peer transfer.
# Unlike OFDM_HS_REQ (unicast, bystanders only infer the hold from overhearing
# someone else's REQ), both messages below are BROADCAST so every node gets
# the window explicitly, including the duration -- a bystander that missed
# the REQ (woke mid-window) still knows when to resume from BLOCKOUT_END, and
# the REQ's own duration field is the fallback if END never arrives.
TYPE_BLOCKOUT_REQ = 0x7   # "quiet down for <duration>: <peer_b> and I are
                          # about to transfer" — broadcast; payload names the
                          # OTHER participant (src is the announcer) and the
                          # window length so every other node can hold off.
TYPE_BLOCKOUT_END = 0x8   # "the window is over, resume" — broadcast by
                          # whichever participant finished the transfer;
                          # lets bystanders resume EARLY instead of waiting
                          # out the full announced duration.

TYPE_NAMES = {
    TYPE_BEACON:       "BEACON",
    TYPE_DATA:         "DATA",
    TYPE_ACK:          "ACK",
    TYPE_RANGE_REQ:    "RANGE_REQ",
    TYPE_RANGE_REPLY:  "RANGE_REPLY",
    TYPE_OFDM_HS_REQ:  "OFDM_HS_REQ",
    TYPE_BLOCKOUT_REQ: "BLOCKOUT_REQ",
    TYPE_BLOCKOUT_END: "BLOCKOUT_END",
}

# ── Field ranges ─────────────────────────────────────────────────────────────
ID_MIN, ID_MAX     = 0x0, 0xF     # 4-bit device IDs
BROADCAST_ID       = 0xF          # dst == 0xF addresses every listener
TYPE_MIN, TYPE_MAX = 0x0, 0xF     # 4-bit message type
SEQ_MIN, SEQ_MAX   = 0x0, 0xF     # 4-bit sequence number (wraps mod 16)
SEQ_MODULO         = SEQ_MAX + 1  # 16

HEADER_LEN         = 2

# Sanity bound only -- NOT the transmittable size.  This was 253 ("keep total
# packet <= 255 bytes"), which read like a format constraint but is not one: the
# header carries no length field, so a packet's length comes from the PHY frame
# that holds it, and both PHYs chunk long messages themselves (css_encoder by
# chunk_size, the OFDM mesh by its frame body).  The 255 was a keep-it-small
# convention, and it became the binding limit on application payload once links
# reached ~10 kbps -- a 1504 B payload failed here (20260804) long before any
# PHY objected.
#
# The REAL per-hop limit is the PHY's frame capacity, enforced where it is known:
# meshlib.MeshModulator.encode_packet checks it against chunk_bytes and names the
# `mesh_chunk` param in the error.  This cap now only catches implausible sizes.
MAX_PAYLOAD        = 4093        # sanity bound (total packet <= 4095 B)


class ModemError(ValueError):
    """Raised for malformed packets or out-of-range field values."""


def _check_nibble(name: str, value: int, lo: int, hi: int) -> int:
    v = int(value)
    if not (lo <= v <= hi):
        raise ModemError(f"{name}={value} out of range [{lo}..{hi}]")
    return v


def pack_packet(src_id: int, dst_id: int, mtype: int, seq: int,
                payload: bytes = b"") -> bytes:
    """Encode one modem packet to bytes (the PHY payload).

    src_id / dst_id : 0..15  (dst 0xF = broadcast)
    mtype           : 0..15  (see TYPE_* constants)
    seq             : 0..15  (application sequence, wraps mod 16)
    payload         : 0..MAX_PAYLOAD bytes
    """
    s = _check_nibble("src_id", src_id, ID_MIN, ID_MAX)
    d = _check_nibble("dst_id", dst_id, ID_MIN, ID_MAX)
    t = _check_nibble("mtype",  mtype,  TYPE_MIN, TYPE_MAX)
    q = _check_nibble("seq",    seq,    SEQ_MIN, SEQ_MAX)
    payload = bytes(payload)
    if len(payload) > MAX_PAYLOAD:
        raise ModemError(f"payload {len(payload)}B exceeds MAX_PAYLOAD={MAX_PAYLOAD}")
    return bytes(((s << 4) | d, (t << 4) | q)) + payload


def unpack_packet(raw: bytes) -> dict:
    """Decode one modem packet.  Returns a dict with keys
    src, dst, type, type_name, seq, payload, is_broadcast.

    Raises ModemError if the buffer is shorter than the 2-byte header.
    """
    raw = bytes(raw)
    if len(raw) < HEADER_LEN:
        raise ModemError(f"packet too short: {len(raw)}B < {HEADER_LEN}B header")
    b0, b1 = raw[0], raw[1]
    src = (b0 >> 4) & 0xF
    dst =  b0       & 0xF
    typ = (b1 >> 4) & 0xF
    seq =  b1       & 0xF
    return {
        "src":          src,
        "dst":          dst,
        "type":         typ,
        "type_name":    TYPE_NAMES.get(typ, f"0x{typ:X}"),
        "seq":          seq,
        "payload":      raw[HEADER_LEN:],
        "is_broadcast": dst == BROADCAST_ID,
    }


def next_seq(seq: int) -> int:
    """Increment a 4-bit sequence number with wraparound."""
    return (int(seq) + 1) % SEQ_MODULO


# ── Blockout payload helpers ─────────────────────────────────────────────────
# TYPE_BLOCKOUT_REQ payload: [peer_b : 1 byte][duration : 2 bytes, big-endian
# deciseconds].  peer_b gets a full byte (not packed into the header nibbles)
# because the header's dst nibble is already spent on BROADCAST_ID -- the
# announcer (src) and peer_b together name the exclusive pair.  Deciseconds
# (0.1 s steps) cover 0..6553.5 s, comfortably past any one-way/two-way
# transfer this mode is for, at a resolution nobody transferring audio-rate
# payload will ever notice.
BLOCKOUT_DURATION_SCALE = 0.1
_BLOCKOUT_DUR_MAX = 0xFFFF


def pack_blockout_req_payload(peer_b: int, duration_s: float) -> bytes:
    """Encode a BLOCKOUT_REQ payload: the other participant + the window length."""
    p = _check_nibble("peer_b", peer_b, ID_MIN, ID_MAX)
    ds = int(round(float(duration_s) / BLOCKOUT_DURATION_SCALE))
    if not (0 <= ds <= _BLOCKOUT_DUR_MAX):
        raise ModemError(f"blockout duration {duration_s}s out of range "
                         f"[0..{_BLOCKOUT_DUR_MAX * BLOCKOUT_DURATION_SCALE}]")
    return bytes([p]) + ds.to_bytes(2, "big")


def unpack_blockout_req_payload(payload: bytes) -> tuple[int, float]:
    """Decode a BLOCKOUT_REQ payload.  Returns (peer_b, duration_s)."""
    payload = bytes(payload)
    if len(payload) < 3:
        raise ModemError(f"blockout_req payload {len(payload)}B < 3B")
    peer_b = payload[0]
    ds = int.from_bytes(payload[1:3], "big")
    return peer_b, ds * BLOCKOUT_DURATION_SCALE


# ── Ranging payload helpers ──────────────────────────────────────────────────
# A RANGE_REPLY carries back the responder's *turnaround* — the number of audio
# samples it deliberately waited between hearing the REQ and emitting the REPLY.
# The initiator subtracts this from the measured round-trip so only true
# time-of-flight remains.  3 bytes (big-endian) holds up to ~16.7 M samples
# (~380 s at 44.1 kHz) — far more than any realistic turnaround.
TURNAROUND_BYTES = 3
_TURN_MAX = (1 << (8 * TURNAROUND_BYTES)) - 1


def pack_range_reply_payload(turnaround_samples: int) -> bytes:
    """Encode the responder turnaround (in audio samples) for a RANGE_REPLY."""
    t = int(turnaround_samples)
    if not (0 <= t <= _TURN_MAX):
        raise ModemError(f"turnaround {t} out of range [0..{_TURN_MAX}]")
    return t.to_bytes(TURNAROUND_BYTES, "big")


def unpack_range_reply_payload(payload: bytes) -> int:
    """Recover the responder turnaround (in audio samples) from a RANGE_REPLY."""
    payload = bytes(payload)
    if len(payload) < TURNAROUND_BYTES:
        raise ModemError(
            f"range reply payload {len(payload)}B < {TURNAROUND_BYTES}B")
    return int.from_bytes(payload[:TURNAROUND_BYTES], "big")


SPEED_OF_SOUND_M_S = 343.0   # dry air, ~20 C; override per environment if needed


def rtt_to_distance_m(round_trip_samples: int, turnaround_samples: int,
                      fs: int, speed_of_sound: float = SPEED_OF_SOUND_M_S):
    """Convert a measured round-trip to a one-way distance in metres.

    round_trip_samples : (reply_arrival - req_emit) in the INITIATOR's own
                         continuous capture-sample counter — no cross-device
                         clock is ever compared.
    turnaround_samples : the value echoed in the RANGE_REPLY payload.
    fs                 : sample rate (Hz).

    tof_one_way = ((round_trip - turnaround) / 2) / fs   seconds
    distance    = tof_one_way * speed_of_sound

    Returns a float (metres).  Negative results (turnaround exceeds the measured
    round-trip — a mis-detection) are returned as-is so the caller can reject
    them rather than silently clamping.
    """
    net_samples = int(round_trip_samples) - int(turnaround_samples)
    tof_one_way_s = (net_samples / 2.0) / float(fs)
    return tof_one_way_s * float(speed_of_sound)


# ── Self-test ────────────────────────────────────────────────────────────────
def _selftest() -> int:
    failures = 0

    def check(cond, msg):
        nonlocal failures
        if not cond:
            failures += 1
            print(f"  FAIL: {msg}")

    # round-trip every field at its extremes
    for src in (0, 7, 15):
        for dst in (0, 8, 15):
            for typ in (0, 15):
                for seq in (0, 15):
                    for pl in (b"", b"\x00", b"\xff\x01\x02", bytes(range(20))):
                        raw = pack_packet(src, dst, typ, seq, pl)
                        got = unpack_packet(raw)
                        check(got["src"] == src, f"src {got['src']}!={src}")
                        check(got["dst"] == dst, f"dst {got['dst']}!={dst}")
                        check(got["type"] == typ, f"type {got['type']}!={typ}")
                        check(got["seq"] == seq, f"seq {got['seq']}!={seq}")
                        check(got["payload"] == pl, f"payload mismatch")
                        check(len(raw) == 2 + len(pl), "length wrong")

    # broadcast flag
    check(unpack_packet(pack_packet(1, BROADCAST_ID, TYPE_BEACON, 0))["is_broadcast"],
          "broadcast not flagged")
    check(not unpack_packet(pack_packet(1, 2, TYPE_DATA, 0))["is_broadcast"],
          "unicast flagged as broadcast")

    # header non-UTF-8 safety: a packet with high bytes must survive round-trip
    raw = pack_packet(0xF, 0xF, 0xF, 0xF, b"\x80\x81\xfe")
    check(unpack_packet(raw)["payload"] == b"\x80\x81\xfe", "hi-byte payload lost")

    # seq wrap
    check(next_seq(15) == 0, "seq wrap 15->0 failed")
    check(next_seq(0) == 1, "seq inc 0->1 failed")

    # blockout payload round-trip, including the 0.1s rounding and both ends
    for peer_b, dur in ((2, 0.0), (9, 90.0), (15, 6553.5)):
        pb, got_dur = unpack_blockout_req_payload(
            pack_blockout_req_payload(peer_b, dur))
        check(pb == peer_b, f"blockout peer_b {pb}!={peer_b}")
        check(abs(got_dur - dur) < 1e-9, f"blockout duration {got_dur}!={dur}")
    try:
        pack_blockout_req_payload(2, 6553.6)
        check(False, "expected ModemError for an out-of-range duration")
    except ModemError:
        pass

    # range payload round-trip + distance math
    turn = 44100  # 1.0 s turnaround
    p = pack_range_reply_payload(turn)
    check(unpack_range_reply_payload(p) == turn, "turnaround round-trip failed")
    # round-trip 1.0s turnaround + 0.02s net flight => 0.01s one-way => 3.43 m
    rt = turn + int(0.02 * 44100)
    d = rtt_to_distance_m(rt, turn, 44100)
    check(abs(d - 3.43) < 0.05, f"distance {d:.3f} != ~3.43 m")

    # out-of-range rejection
    for bad in (
        lambda: pack_packet(16, 0, 0, 0),
        lambda: pack_packet(0, 16, 0, 0),
        lambda: pack_packet(0, 0, 16, 0),
        lambda: pack_packet(0, 0, 0, 16),
        lambda: pack_packet(0, 0, 0, 0, b"x" * (MAX_PAYLOAD + 1)),
        lambda: unpack_packet(b"\x00"),
    ):
        try:
            bad()
            check(False, "expected ModemError not raised")
        except ModemError:
            pass

    if failures:
        print(f"SELFTEST FAILED: {failures} check(s)")
    else:
        print("SELFTEST OK")
    return failures


if __name__ == "__main__":
    import sys
    sys.exit(1 if _selftest() else 0)
