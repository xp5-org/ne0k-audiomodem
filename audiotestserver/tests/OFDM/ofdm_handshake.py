"""
ofdm_handshake.py -- V.22-style two-way OFDM rate negotiation.

Layering mirrors the CSS mesh stack (css_modem_node.py state machine /
css_modem_daemon.py driver / css_cmodem_node.py hardware binding), so the mesh
firmware can later import this module unchanged while it stays independently
testable:

  HandshakeSM     pure protocol logic: states, retries, timeouts, CRC
                  proof-of-receipt.  Consumes events, emits actions.  No DSP,
                  no audio, no wall clock (time is injected).  THE import
                  target for the meshmodem.
  DSP helpers     probe / RobustLink encode+decode wrappers, including
                  unknown-sample-rate trial detection -- the receiver does not
                  need to be told the transmitter's rate out-of-band.
  ofdmhs_node.py  test-rig binding (real mic/speaker, one node per host).
                  Coord only starts the run; the handshake itself is fully
                  in-band: no shared files, no coord signals.

Protocol (half-duplex, turn-based; I = initiator/caller, R = responder):

    I: PROBE                     R measures  -> BAT_I  (I's tx bit-loading)
    R: PROBE                     I measures  -> BAT_R  (R's tx bit-loading)
    R: MSG BAT_REPORT(BAT_I)     I learns its tx table
    I: MSG BAT_ACK(BAT_R,
                   proof=crc(BAT_I))   R learns its tx table + proof I got BAT_I
    R: MSG GO(proof=crc(BAT_R))        I gets proof R got BAT_R
    I: MSG FIN                         courtesy: lets R stop listening early
    both DONE -- each end holds an agreed bit-loading table for BOTH directions.

Recovery: every wait state has a deadline.  On expiry a node retransmits its
last transmission (responder) or sends MSG_RETRY asking the peer to repeat
(initiator), up to max_retries, then FAILs.  The responder's r_wait_ack
retransmission repeats its WHOLE turn (probe + report), because the one thing
the initiator cannot ask for is the responder's probe: stuck in i_wait_probe
it can neither decode a REPORT nor signal RETRY the responder would act on --
a REPORT-only retransmit deadlocked both ends when the responder's first
probe was lost (each side re-sending something the other could not hear).
For the same reason r_wait_ack keeps listening for probes ("probe+msg"): an
initiator re-probe means it heard nothing and the whole turn must repeat.
A CRC proof mismatch is treated as a failed transfer and triggers the same
retransmission.  The GO/FIN tail is two-generals: the responder lingers after
GO long enough to cover one full initiator retry cycle including the RETRY's
airtime and carrier-sense deferral (re-armed on every RETRY), exits early on
FIN, and declares itself done at linger expiry -- by then a live initiator
would have either FINed or RETRYed.

The handshake channel starts from a fixed CONVENTION (band, FFT geometry, probe
length below) so no prior agreement is needed; only the sample rate varies per
transmitter (its speaker's native rate) and the receiver detects that by trial
acquisition.

Band renegotiation (MSG_BAND) relaxes the band half of that convention.  The
probe measures the whole convention band, but its edges are often dead, and a
dead edge is not free: the BAT is ~3 bits per PROBED carrier, so every
handshake message carries it.  At 400-14000 Hz the table is 95B (~25s on the
air) against 26B (~7s) at 400-4000.  So the receiver of a probe -- the only end
that can judge what survives the path -- may answer with MSG_BAND instead of a
report, naming the band the peer should transmit in; both ends then restart the
handshake on the new geometry (capped at HS_MAX_RETRAINS, so it converges).

The two directions renegotiate INDEPENDENTLY: each is chosen by its own
receiver, so a pair whose paths differ (16 active carriers one way, 72 the
other, observed OTA) keeps the healthy direction wide.  Consequently a node
transmits with sm.my_tx_band and decodes the peer with sm.peer_tx_band; the
DSP helpers all take that band as an argument rather than reading a global.
"""
from __future__ import annotations

import sys
import math
from math import gcd
from pathlib import Path

import numpy as np


# --- category-aware sys.path bootstrap (auto-inserted) ---
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_TESTS_ROOT = _HERE.parent if _HERE.name in ("CSS", "MESH", "OFDM", "QAM") else _HERE
_REPO_ROOT = _TESTS_ROOT.parent
for _p in (_TESTS_ROOT, _REPO_ROOT, _TESTS_ROOT / "CSS", _TESTS_ROOT / "MESH",
           _TESTS_ROOT / "OFDM", _TESTS_ROOT / "QAM", _TESTS_ROOT / "regression"):
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# --- end bootstrap ---

import benchlib                                                   # noqa: E402
from ofdm_modem import (                                          # noqa: E402
    build_config, Modem, ChannelProbe, RobustLink,
    AdaptiveModulator, AdaptiveDemodulator,
    bat_to_bytes, bytes_to_bat, carrier_spectrum, _ISI_BETA_MAX,
)

# --------------------------------------------------------------------------- #
# Fixed handshake convention (both ends know this a priori -- like V.22's
# answer tone frequencies).  Only sample_rate varies (TX speaker native rate).
# --------------------------------------------------------------------------- #

HS_BAND = dict(
    n_fft            = 1024,
    cp_ratio         = 0.25,
    freq_lo          = 400.0,
    freq_hi          = 4000.0,
    pilot_spacing    = 8,
    probe_duration_s = 5.0,
    # Allocation policy: the acoustic channel is TIME-VARYING (a person moving
    # reshapes sub-kHz multipath fades completely, observed OTA: carriers
    # strong at probe time were in deep fades 60 s later at data time).  So,
    # unlike DSL, the margin must cover fade migration, and dense QAM is not
    # survivable yet -- cap at QPSK.
    snr_margin_db    = 9.0,
    max_bits_per_bin = 2,
    # SNR estimator stays 'mean': its variance term already folds within-probe
    # fading into the noise floor (verified -- a brief carrier null it grades
    # ~18dB down, MORE conservatively than a sub-block percentile, which can
    # skip a single bad block).  'percentile' is available as an opt-in but is
    # NOT better here; the real optimism is the probe->data TIME gap (fade
    # migration over the ~60s to data), which no single probe can measure --
    # cover that with snr_margin_db or closed-loop EVM feedback.
    # Non-periodic measurement symbols at the tail of the probe.  This is what
    # makes the probe's SNR representative: a repeated symbol lets reverberation
    # reach periodic steady state and be absorbed into H as gain, so the probe
    # graded an 18 dB path that delivered 4.5 dB (OTA 20260803/04).  These
    # symbols each carry a DIFFERENT known pattern, so their tails interfere with
    # each other exactly as data does, and the variance-based estimator picks the
    # ISI up with no new maths.  Taken out of the existing probe length, so
    # airtime is unchanged.  0 restores the old periodic-only behaviour.
    #
    # NOTE ON MARGIN: snr_margin_db exists partly to cover this over-promise.
    # With an honest probe that cover is no longer needed, so the useful margin
    # drops to ~3-6 dB (fade migration only).  isi_derate is likewise superseded
    # -- the penalty is now measured directly, not inferred.
    probe_meas_syms  = 32,
    snr_estimator    = "mean",
    snr_pctile       = 15.0,
    snr_blocks       = 8,
    # ISI de-rating: fold the measured reverberation ceiling into the SNR the
    # allocator sees (ofdm_modem.isi_ratio, applied in BitAllocator.allocate).
    # Default OFF -- it changes negotiated rates, so it is opt-in until an OTA
    # run confirms it -- but it is what snr_margin_db above is really standing
    # in for, and unlike a fixed margin it ADAPTS: a dead room de-rates by
    # almost nothing, a live one by 15-20 dB.  Measured OTA (20260803_194332):
    # beta=0.25-0.29, a ~5-6 dB ceiling on a probe claiming 25 dB, against
    # 4-7 dB actually delivered.  Turning it on pairs naturally with a smaller
    # margin (the node scripts drop the default to 2 dB) since the ceiling now
    # does the job the margin was guessing at.
    isi_derate       = 0.0,
)

# Per-run overrides for the fixed handshake convention, injected from the
# test's PARAMS (e.g. meshmodem_node) so snr_margin_db / max_bits_per_bin /
# the SNR-estimator knobs can be tuned per distance WITHOUT editing the
# a-priori convention.  Every node in a run gets identical PARAMS, so both
# ends stay in agreement.  Empty = use HS_BAND as-is.
HS_OVERRIDES: dict = {}
HS_RATES   = (48000, 44100)      # candidate TX rates for trial detection
# FFT size for handshake MESSAGES, independent of the probe's (see hs_msg_cfg).
# None = follow the probe geometry (the old coupled behaviour).
HS_MSG_NFFT = 1024
HS_ECC     = 10                  # RobustLink RS parity bytes

MSG_BAT_REPORT = 1
MSG_BAT_ACK    = 2
MSG_GO         = 3
MSG_RETRY      = 4
MSG_FIN        = 5
MSG_BAND       = 6      # renegotiate the band for ONE direction (see below)
MSG_ANNOUNCE   = 7      # mesh presence round: body = [id u8][heard mask u16][flags u8] (see modemgui ofdm_announce)
MSG_BLOCKOUT_REQ = 8    # mesh blockout window open: body = [announcer u8][peer u8][duration u16] (see modemgui
                        # mesh_node.send_blockout) -- OFDM-only: CSS's baud rate can't carry this
MSG_BLOCKOUT_END = 9    # mesh blockout window closed early: body = [announcer u8] (see mesh_node.end_blockout)
MSG_BAT_TABLES = 10     # mesh broadcast BAT round: measured tables, a report or the coordinator's confirm (see
                        # modemgui ofdm_announce.pack_tables) -- bootstrap ladder step, like BAT_REPORT

# Band renegotiation.  Only the RECEIVER of a probe can judge which part of the
# spectrum actually survives a path, so each direction's band is chosen by the
# far end and pushed to the transmitter with MSG_BAND.  The two directions are
# independent: an asymmetric pair of speakers/mics routinely grades 16 active
# carriers one way and 72 the other (observed OTA 20260727), and forcing the
# healthy direction down to the sick one's band throws away real capacity.
#
# Body: [dir u8][lo_hz u16 BE][hi_hz u16 BE]
BAND_DIR_PEER_TX = 0    # "transmit in this band"  (the only one acted on today)
BAND_DIR_SELF_TX = 1    # "I will transmit in this band" (informational)
# Retrains are capped: two ends can both propose at once, and an uncapped
# exchange could ping-pong bands instead of converging.  One retrain per
# direction is enough to escape a dead octave; after that we take what we get.
HS_MAX_RETRAINS = 1

# BAT body compression for BAT_REPORT / BAT_ACK.  RobustLink runs at ~1 bit
# per 33ms symbol, so body BYTES map 1:1 to seconds on the air (58B wide-band
# BAT = 19s).  Measured BATs are ~85% runs of identical codes (long off runs,
# contiguous bpsk/qpsk stretches), so run-length coding the 3-bit code stream
# cuts the message to a third or less.  Toggleable so both ends can be pinned
# to raw for A/B (same code ships to every node -- no mixed-version risk).
HS_BAT_RLE = True

# BAT coarsening factor: real OTA BATs are FRAGMENTED (off/bpsk/qpsk
# interleaved per-carrier with channel ripple -- observed 20260723: RLE fell
# back to raw on every real BAT and the 19s reports stayed 19s).  Adjacent
# carriers are highly correlated, so quantize the table to groups of N
# carriers AT MEASURE TIME -- both ends then hold the identical coarse
# table, proof CRCs agree, and the grouped wire format is a fixed ~(n/N)*3
# bits however fragmented the channel is.  Group level = the group's
# SECOND-LOWEST code: min() collapses a group whenever one carrier ripple-
# nulls (98->20 active on a measured-shape table), while second-lowest
# overdrives at most one carrier per group -- inside what the FEC absorbs.
# 1 disables.
HS_BAT_GROUP = 4

# Upper bound on the number of GROUP LEVELS that ever go on the wire.  The BAT
# is ~3 bits per probed carrier, and RobustLink airs ~1 bit per symbol, so an
# uncapped table makes handshake airtime grow with carrier count -- and with
# n_fft it grows QUADRATICALLY, because a bigger FFT means both more carriers
# AND a longer symbol.  Measured: n_fft=4096 over 400-14000 Hz is 1105
# carriers, a 415B raw table, ~400 SECONDS for one BAT_REPORT.  Capping the
# level count makes the message a bounded size no matter how fine the probe
# grid is; adjacent carriers are highly correlated, so the resolution given up
# is mostly redundant anyway.
HS_BAT_TARGET_CODES = 128


def bat_group_for(n_carriers: int) -> int:
    """Group size that keeps a table of `n_carriers` inside the wire budget."""
    if n_carriers <= 0:
        return HS_BAT_GROUP
    need = -(-n_carriers // max(1, HS_BAT_TARGET_CODES))    # ceil
    return max(HS_BAT_GROUP, need)


def bat_coarsen(bat: np.ndarray, group: int = None) -> np.ndarray:
    """Quantize a per-carrier BAT to per-`group` levels (same length).

    `group=None` sizes the grouping to the table (see bat_group_for): a fixed
    group cannot bound the wire cost once the probe grid gets fine.
    """
    g = bat_group_for(len(bat)) if group is None else group
    if g <= 1 or len(bat) == 0:
        return bat
    n = len(bat)
    pad = (-n) % g
    a = np.concatenate([bat, np.zeros(pad, dtype=bat.dtype)]) if pad else bat
    grp = np.sort(a.reshape(-1, g), axis=1)
    lvl = grp[:, 1] if g >= 2 else grp[:, 0]
    return np.repeat(lvl, g)[:n]


# Hard ceiling on the packed BAT.  The table rides RobustLink's bootstrap mode
# at ~1 bit per symbol, so wire bytes ARE airtime: 48B is ~15s, 354B is ~93s
# (measured OTA 20260728_003130, which is where this budget came from).
# 128B is ~33s in the bootstrap mode -- and BAT_REPORT is now the ONLY message
# paying that rate (see fast_groups_for), so it is a bounded one-off.  Do not
# lower it casually: coarsening buys wire bytes with CAPACITY, and on a
# fragmented table the exchange rate is brutal (measured on the 942-carrier
# table from run 20260728_003130: 355B/86s keeps 827 bits per symbol, 93B/25s
# keeps 468, 31B/11s keeps 56).  There is no setting that is small AND
# faithful; a table that fine is simply expensive to describe.
HS_BAT_MAX_WIRE = 128

# Fraction of link capacity that must survive coarsening.  This is the knob that
# actually decides BAT_REPORT airtime on a normal table -- HS_BAT_MAX_WIRE only
# catches the pathological ones.  0.90 buys the g=4 grouping on measured tables
# (94% kept, wire cut ~3x); raise it toward 1.0 to prefer capacity over airtime.
HS_BAT_MIN_KEEP = 0.90


def bat_fit_wire(bat: np.ndarray, budget: int = None) -> np.ndarray:
    """Finest version of `bat` whose packed form fits the wire budget.

    Replaces an earlier all-or-nothing rule that coarsened only when doing so
    cost under 40% of the active carriers, on the reasoning that "wire size is
    the only thing lost".  That held at 72 carriers (27B raw).  At 942 it does
    not: a fragmented real table failed the capacity test, shipped RAW at 354B,
    and turned one BAT_REPORT into 93 SECONDS of airtime.

    That fixed the unbounded case but left the common one: keeping the FINEST
    table that fits means a table already under budget is never coarsened at
    all.  Run 20260729_212926 shipped 89B raw and spent 25.3s of airtime on one
    BAT_REPORT -- while g=4 on that same table costs 6% of link capacity and
    saves 62 of those 89 bytes (25.3s -> 10.3s at ladder step 0, 3.9s -> 2.0s at
    step 3).  Wire bytes ARE airtime, and near the top of the table the exchange
    rate is very cheap; a fits/doesn't-fit test cannot see that.

    So: coarsen freely while HS_BAT_MIN_KEEP of the capacity survives, and fall
    back to the old finest-that-fits rule when nothing clears that bar.
    """
    budget = HS_BAT_MAX_WIRE if budget is None else budget
    if len(bat) == 0:
        return bat
    full = float(np.sum(bat))
    # g=2 is deliberately absent: bat_coarsen takes each group's SECOND-LOWEST
    # code, which for a 2-element group is the LARGER of the pair -- it
    # over-drives one carrier in every pair and reports MORE capacity than was
    # measured (117% of full on the 20260729_212926 table).  The heuristic is
    # only conservative at g>=4.
    keep_best = fit_best = None
    cand = bat
    for g in (1, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128):
        cand = bat if g == 1 else bat_coarsen(bat, g)
        if len(bat_rle_pack(bat_to_bytes(cand))) > budget:
            continue                      # coarser only shrinks; keep trying
        if fit_best is None:
            fit_best = cand               # finest that fits -- the old rule
        # DELIVERABLE bits, not claimed ones.  bat_coarsen levels each group to
        # its second-lowest code, so carriers under that level are told to carry
        # MORE than they measured -- and an overdriven carrier is worse than a
        # muted one, because it returns errors instead of nothing.  Summing the
        # coarse table counts those phantom bits as capacity and can pass a
        # table that is materially worse on the air (measured on run
        # 20260729_212926: g=4 claims 368 of 411 bits but delivers 331, having
        # overdriven 28 carriers).  min() against the measured table charges
        # each carrier only what it can actually carry.
        deliverable = float(np.minimum(cand, bat).sum())
        if full <= 0 or deliverable >= HS_BAT_MIN_KEEP * full:
            keep_best = cand              # coarsest that also keeps capacity
    if keep_best is not None:
        return keep_best
    if fit_best is not None:
        return fit_best
    return cand                           # nothing fit; coarsest is least bad


def _codes_of(bat_bytes: bytes) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(bat_bytes, dtype=np.uint8))
    n = (len(bits) // 3) * 3
    return (bits[0:n:3] << 2) | (bits[1:n:3] << 1) | bits[2:n:3]


def _codes_pack(codes: np.ndarray) -> bytes:
    bits = np.zeros(len(codes) * 3, dtype=np.uint8)
    arr = np.asarray(codes, dtype=np.uint8)
    bits[0::3] = (arr >> 2) & 1
    bits[1::3] = (arr >> 1) & 1
    bits[2::3] = arr & 1
    return np.packbits(bits).tobytes()


def bat_rle_pack(bat_bytes: bytes) -> bytes:
    """Smallest of three formats:
    [0x48][g][n hi][n lo][packed codes[::g]]  grouped (tables from bat_coarsen)
    [0x52][code|runlen pairs...]              run-length
    [0x00][raw bytes]                         fallback

    `n` is 16-bit.  It used to be one byte, which silently disabled grouping
    for any table over 255 carriers -- precisely the tables that needed it, so
    a 1105-carrier BAT fell back to run-length and shipped 201B where grouping
    gives ~50B.  The group size is DETECTED here rather than assumed, since
    bat_coarsen now picks it from the table length.
    """
    raw = bytes([0x00]) + bat_bytes
    if not bat_bytes:
        return raw
    codes = _codes_of(bat_bytes)
    best = raw
    # Largest grouping the table actually satisfies wins (fewest levels on the
    # wire).  bat_group_for's answer is tried first; the rest cover tables
    # coarsened under a different policy or not coarsened at all.
    cands = sorted({bat_group_for(len(codes)), HS_BAT_GROUP, 32, 16, 8, 4, 2},
                   reverse=True)
    for g in cands:
        if g <= 1 or len(codes) > 0xFFFF:
            continue
        mins = codes[::g]
        if np.array_equal(np.repeat(mins, g)[:len(codes)], codes):
            blob = (bytes([0x48, g]) + len(codes).to_bytes(2, "big")
                    + _codes_pack(mins))
            if len(blob) < len(best):
                best = blob
            break
    pairs = bytearray([0x52])
    i = 0
    while i < len(codes):
        j = i
        while j < len(codes) and codes[j] == codes[i] and j - i < 255:
            j += 1
        pairs += bytes([int(codes[i]), j - i])
        i = j
    if len(pairs) < len(best):
        best = bytes(pairs)
    return best


def bat_rle_unpack(blob: bytes) -> bytes:
    if not blob:
        return b""
    if blob[0] == 0x00:
        return blob[1:]
    if blob[0] == 0x48:
        g = blob[1]
        n = int.from_bytes(blob[2:4], "big")
        mins = _codes_of(blob[4:])[: (n + g - 1) // g]
        return _codes_pack(np.repeat(mins, g)[:n])
    if blob[0] == 0x47:                     # legacy 1-byte-length grouped form
        g, n = blob[1], blob[2]
        mins = _codes_of(blob[3:])[: (n + g - 1) // g]
        return _codes_pack(np.repeat(mins, g)[:n])
    codes = []
    for k in range(1, len(blob) - 1, 2):
        codes.extend([blob[k]] * blob[k + 1])
    if not codes:
        return b""
    return _codes_pack(np.array(codes, dtype=np.uint8))

_MSG_NAMES = {1: "BAT_REPORT", 2: "BAT_ACK", 3: "GO", 4: "RETRY", 5: "FIN",
              6: "BAND", 7: "ANNOUNCE", 8: "BLOCKOUT_REQ", 9: "BLOCKOUT_END",
              10: "BAT_TABLES"}


def pack_band(lo_hz: float, hi_hz: float,
              direction: int = BAND_DIR_PEER_TX) -> bytes:
    """[dir][lo_hz u16][hi_hz u16] -- Hz fits u16 up to 65535, well past the
    22.05kHz Nyquist ceiling of the lowest supported rate."""
    return (bytes([direction & 0xFF])
            + int(max(0, min(65535, round(lo_hz)))).to_bytes(2, "big")
            + int(max(0, min(65535, round(hi_hz)))).to_bytes(2, "big"))


def unpack_band(body: bytes):
    """Returns (direction, lo_hz, hi_hz) or None if malformed/nonsensical."""
    if len(body) < 5:
        return None
    direction = body[0]
    lo = int.from_bytes(body[1:3], "big")
    hi = int.from_bytes(body[3:5], "big")
    if hi <= lo or hi - lo < 100:        # a band that narrow carries nothing
        return None
    return direction, float(lo), float(hi)


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
            crc &= 0xFFFF
    return crc


def pack_msg(mtype: int, proof_crc: int, body: bytes) -> bytes:
    """[1B type][2B proof crc][2B body crc][body]"""
    return (bytes([mtype]) + proof_crc.to_bytes(2, 'big')
            + crc16(body).to_bytes(2, 'big') + body)


def unpack_msg(data: bytes):
    """Returns (mtype, proof_crc, body) or None if malformed/corrupt."""
    if len(data) < 5:
        return None
    mtype = data[0]
    proof = int.from_bytes(data[1:3], 'big')
    bcrc  = int.from_bytes(data[3:5], 'big')
    body  = data[5:]
    if mtype not in _MSG_NAMES or crc16(body) != bcrc:
        return None
    return mtype, proof, body


# --------------------------------------------------------------------------- #
# HandshakeSM -- pure protocol logic
# --------------------------------------------------------------------------- #

class HandshakeSM:
    """Transport-agnostic V.22-style handshake state machine.

    Events (call these; each may append actions):
        start(now)
        heard_probe(bat_bytes, now)   binding measured the peer's probe and
                                      packed the resulting BAT for the
                                      peer->me direction (= the PEER's tx table)
        heard_msg(mtype, proof, body, now)
        tick(now)                     drive timeouts

    Actions (drain from .actions; tuples):
        ("send_probe",)
        ("send_msg", mtype, proof_crc, body_bytes)
        ("done", {"my_tx_bat": bytes, "peer_tx_bat": bytes})
        ("fail", reason_str)

    `expecting()` -> "probe" | "msg" | None tells the binding what to try to
    decode from the mic during the current wait state.
    """

    def __init__(self, role: str, max_retries: int = 2,
                 wait_probe_s: float = 60.0, wait_msg_s: float = 60.0,
                 linger_s: float | None = None, skip_fin: bool = False,
                 band_negotiation: bool = False):
        assert role in ("initiator", "responder")
        self.role         = role
        self.state        = "idle"
        self.max_retries  = max_retries
        self.wait_probe_s = wait_probe_s
        self.wait_msg_s   = wait_msg_s
        self.skip_fin     = skip_fin
        # Off by default so a binding that does not implement band
        # renegotiation is bit-for-bit unaffected by it.  Turning it on makes
        # the probe-wait states listen for messages too (MSG_BAND arrives
        # exactly there) -- which costs an extra trial decode per poll, and
        # decode CPU is what starves the capture callback on slower nodes.
        self.band_negotiation = band_negotiation
        # Linger must outlive one full initiator retry cycle: its wait PLUS
        # the RETRY's airtime PLUS the binding's carrier-sense deferral (up to
        # ~25s when the air is busy), else the responder goes deaf before a
        # lost GO can be asked for again.  Re-armed on every RETRY, cut short
        # by FIN (or, in the mesh binding, by the peer's first data frame).
        self.linger_s     = (wait_msg_s + 45.0) if linger_s is None else linger_s

        self.peer_tx_bat: bytes | None = None   # I measured peer's probe
        self.my_tx_bat:   bytes | None = None   # peer told me my table
        # Per-direction bands (None = the a-priori convention).  my_tx_band is
        # set by the PEER via MSG_BAND; peer_tx_band is what we told the peer
        # to use, and therefore what we must decode it with.
        self.my_tx_band:   tuple | None = None
        self.peer_tx_band: tuple | None = None
        # One budget per DIRECTION, not one shared budget: adopting the peer's
        # band for our tx path must not spend the allowance we need to fix the
        # other path.  Otherwise whichever direction renegotiated first would
        # lock the other one to the starting band forever.
        self.retrains_sent = 0        # bands we imposed on the peer's tx
        self.retrains_recv = 0        # bands the peer imposed on ours
        self.max_retrains = HS_MAX_RETRAINS
        self.actions: list[tuple] = []
        self.done = False
        self.failed = False
        self._deadline: float | None = None
        self._retries = 0
        self._log: list[str] = []

    # - helpers -

    def _emit(self, *action):
        self.actions.append(action)

    def _note(self, now, msg):
        self._log.append(f"{now:8.2f} [{self.role[:4]}/{self.state}] {msg}")

    def _arm(self, now, seconds):
        self._deadline = now + seconds

    def _goto(self, state):
        self.state = state
        self._retries = 0

    def _bat_body(self):
        # BAT bodies dominate handshake airtime; ship them compressed.
        # proof CRCs stay over the RAW bat bytes on both ends.
        return (bat_rle_pack(self.peer_tx_bat) if HS_BAT_RLE
                else self.peer_tx_bat)

    @staticmethod
    def _bat_from_body(body: bytes) -> bytes:
        return bat_rle_unpack(body) if HS_BAT_RLE else body

    def _send_report(self):
        self._emit("send_msg", MSG_BAT_REPORT, 0, self._bat_body())

    def _send_ack(self):
        self._emit("send_msg", MSG_BAT_ACK, crc16(self.my_tx_bat),
                   self._bat_body())

    def _send_go(self):
        self._emit("send_msg", MSG_GO, crc16(self.my_tx_bat), b"")

    def _finish(self, now):
        self.done = True
        self.state = "done"
        self._deadline = None
        self._note(now, "DONE")
        self._emit("done", {"my_tx_bat": self.my_tx_bat,
                            "peer_tx_bat": self.peer_tx_bat})

    def _fail(self, now, reason):
        self.failed = True
        self.state = "failed"
        self._deadline = None
        self._note(now, f"FAIL {reason}")
        self._emit("fail", reason)

    def expecting(self) -> str | None:
        # r_wait_ack listens for BOTH: the normal next step is the ACK (msg),
        # but an initiator that missed our probe re-probes, and hearing that
        # is the ONLY way its miss can reach us (see module docstring).
        # With band negotiation on, the probe-wait states listen for msgs too,
        # because MSG_BAND arrives exactly there: a peer that has measured us
        # answers our probe with a band instead of a report, and a node deaf to
        # msgs while waiting for a probe could never be told to retrain.
        pw = "probe+msg" if self.band_negotiation else "probe"
        return {
            "i_wait_probe": pw,
            "r_wait_probe": pw,
            "i_wait_bat":   "msg",
            "i_wait_go":    "msg",
            "r_wait_ack":   "probe+msg",
            "r_linger":     "msg",
        }.get(self.state)

    def _restart(self, now, why):
        """Re-enter the opening state, keeping the negotiated bands.  Used by
        both ends of a retrain so they re-probe on the new geometry."""
        self.peer_tx_bat = None
        self.my_tx_bat = None
        self._retries = 0
        if self.role == "initiator":
            self._emit("send_probe")
            self._goto("i_wait_probe")
            self._arm(now, self.wait_probe_s)
        else:
            self._goto("r_wait_probe")
            self._arm(now, self.wait_probe_s)
        self._note(now, f"restart ({why})")

    # - events -

    def start(self, now: float):
        if self.role == "initiator":
            self._emit("send_probe")
            self._goto("i_wait_probe")
            self._arm(now, self.wait_probe_s)
        else:
            self._goto("r_wait_probe")
            self._arm(now, self.wait_probe_s)
        self._note(now, "start")

    def heard_probe(self, bat_bytes: bytes, now: float,
                    band: tuple | None = None):
        """`band` is the binding's recommendation for the PEER's tx path,
        derived from the probe we just measured (see recommend_band).  When it
        differs from what the peer is using, push it and retrain instead of
        continuing: the BAT we just measured is expressed on the OLD carrier
        grid and would be meaningless once the peer narrows."""
        if (band and self.band_negotiation and band != self.peer_tx_band
                and self.retrains_sent < self.max_retrains
                and self.state in ("i_wait_probe", "r_wait_probe", "r_wait_ack")):
            self.peer_tx_band = band
            self.retrains_sent += 1
            self._emit("send_msg", MSG_BAND, 0,
                       pack_band(band[0], band[1], BAND_DIR_PEER_TX))
            self._note(now, f"band retrain -> peer tx {band[0]:.0f}-"
                            f"{band[1]:.0f} Hz")
            self._restart(now, "sent BAND")
            return
        if self.state == "i_wait_probe":
            self.peer_tx_bat = bat_bytes
            self._goto("i_wait_bat")            # R sends BAT_REPORT unprompted
            self._arm(now, self.wait_msg_s)
            self._note(now, f"peer probe measured ({len(bat_bytes)}B bat)")
        elif self.state == "r_wait_probe":
            self.peer_tx_bat = bat_bytes
            self._emit("send_probe")
            self._send_report()
            self._goto("r_wait_ack")
            self._arm(now, self.wait_msg_s)
            self._note(now, f"initiator probe measured ({len(bat_bytes)}B bat)")
        elif self.state == "r_wait_ack":
            # Initiator re-probed: it never heard our probe/report.  Re-measure
            # (fresher estimate) and repeat our whole turn.
            self.peer_tx_bat = bat_bytes
            self._retries += 1
            if self._retries > self.max_retries:
                self._fail(now, "peer_restarting")
                return
            self._emit("send_probe")
            self._send_report()
            self._arm(now, self.wait_msg_s)
            self._note(now, "initiator re-probed -- repeating probe+report")
        # other states: stale echo, ignore

    def heard_msg(self, mtype: int, proof: int, body: bytes, now: float):
        self._note(now, f"rx {_MSG_NAMES.get(mtype, mtype)}")
        # MSG_BAND is handled in EVERY state and outranks the normal flow: the
        # peer is telling us the geometry it will decode us on, so anything we
        # had queued for the old band is already undeliverable.
        if mtype == MSG_BAND:
            got = unpack_band(body)
            if got is None:
                return
            direction, lo, hi = got
            if direction == BAND_DIR_SELF_TX:
                self.peer_tx_band = (lo, hi)      # informational
                self._note(now, f"peer announces its tx {lo:.0f}-{hi:.0f} Hz")
                return
            if (lo, hi) == self.my_tx_band:       # duplicate/retransmit
                return
            if self.retrains_recv >= self.max_retrains:
                self._note(now, "ignoring BAND -- retrain budget spent")
                return
            self.my_tx_band = (lo, hi)
            self.retrains_recv += 1
            self._note(now, f"band retrain -> my tx {lo:.0f}-{hi:.0f} Hz")
            self._restart(now, "got BAND")
            return
        if self.state == "i_wait_bat" and mtype == MSG_BAT_REPORT:
            self.my_tx_bat = self._bat_from_body(body)
            self._send_ack()
            self._goto("i_wait_go")
            self._arm(now, self.wait_msg_s)
        elif self.state == "i_wait_go":
            if mtype == MSG_GO:
                if proof == crc16(self.peer_tx_bat or b""):
                    # FIN is a courtesy that costs ~5s airtime + a turn; the
                    # mesh binding's r_linger already treats the initiator's
                    # first DATA frame as an implicit FIN, so skipping it is
                    # free whenever data follows promptly (it does here: the
                    # hs initiator is the scripted data sender).
                    if not self.skip_fin:
                        self._emit("send_msg", MSG_FIN, 0, b"")
                    self._finish(now)
                else:
                    # peer holds a different BAT_R than we sent -- ask again
                    self._emit("send_msg", MSG_RETRY, 0, b"")
                    self._arm(now, self.wait_msg_s)
            elif mtype == MSG_BAT_REPORT:
                # our BAT_ACK was lost; peer retransmitted its report
                self.my_tx_bat = self._bat_from_body(body)
                self._send_ack()
                self._arm(now, self.wait_msg_s)
        elif self.state == "r_wait_ack":
            if mtype == MSG_BAT_ACK:
                if proof == crc16(self.peer_tx_bat or b""):
                    self.my_tx_bat = self._bat_from_body(body)
                    self._send_go()
                    self._goto("r_linger")
                    self._arm(now, self.linger_s)
                else:
                    self._send_report()         # peer got a corrupt report
                    self._arm(now, self.wait_msg_s)
            elif mtype == MSG_RETRY:
                self._send_report()
                self._arm(now, self.wait_msg_s)
        elif self.state == "r_linger":
            if mtype == MSG_FIN:
                self._finish(now)               # initiator confirmed GO landed
            elif mtype in (MSG_RETRY, MSG_BAT_ACK):
                # initiator missed GO (RETRY) or our GO crossed its ACK
                self._send_go()
                self._arm(now, self.linger_s)
        # anything else: duplicate/stale, ignore

    def hold_off(self, seconds: float):
        """Extend the current deadline by time the node spent unable to hear a
        reply -- its own airtime plus any carrier-sense deferral.

        Wait states are armed when the SM *decides* to transmit, but the peer
        cannot answer until that transmission actually ends.  Without this the
        timer burns through our own turn (the responder's probe+report is ~13s
        of airtime) and fires a retransmission on top of the reply already on
        the air.  The binding calls this after every transmission."""
        if self._deadline is not None and seconds > 0:
            self._deadline += float(seconds)

    def tick(self, now: float):
        if self._deadline is None or now < self._deadline:
            return
        self._retries += 1
        if self.state == "r_linger":
            self._finish(now)                   # linger over, assume GO landed
            return
        if self._retries > self.max_retries:
            self._fail(now, f"timeout_{self.state}")
            return
        self._note(now, f"timeout -> retry {self._retries}")
        if self.state == "i_wait_probe":
            self._emit("send_probe")
            self._arm(now, self.wait_probe_s)
        elif self.state == "r_wait_probe":
            self._arm(now, self.wait_probe_s)   # keep listening
        elif self.state == "i_wait_bat":
            self._emit("send_msg", MSG_RETRY, 0, b"")
            self._arm(now, self.wait_msg_s)
        elif self.state == "i_wait_go":
            self._emit("send_msg", MSG_RETRY, 0, b"")
            self._arm(now, self.wait_msg_s)
        elif self.state == "r_wait_ack":
            # Repeat the WHOLE turn (probe + report), not the report alone.
            # If the initiator missed our probe it is stuck in i_wait_probe,
            # where a bare REPORT is undecodable to it (its binding only
            # attempts probe decode there) -- REPORT-only retransmission
            # deadlocked both ends OTA-style: each re-sent something the
            # other could not hear until both failed.
            self._emit("send_probe")
            self._send_report()
            self._arm(now, self.wait_msg_s)

    def log(self) -> str:
        return "\n".join(self._log)


# --------------------------------------------------------------------------- #
# DSP helpers -- shared by the test binding and (later) the mesh firmware
# --------------------------------------------------------------------------- #

def hs_cfg(sample_rate: int, band: tuple | None = None, **overrides) -> dict:
    """Handshake config at `sample_rate`.

    `band` is an optional (freq_lo, freq_hi) override for ONE direction, set by
    band renegotiation (MSG_BAND).  It is deliberately a per-call argument
    rather than another global: after a retrain the two directions can run in
    DIFFERENT bands, so a node builds its transmissions with its own tx band
    and decodes the peer with the peer's -- a single global cannot express that.
    None keeps the a-priori convention.

    `**overrides` exists so the DATA config can be derived from this one
    function (see hs_data_cfg) rather than restating the band anywhere.  A
    second band definition is how run 20260803_194332 negotiated a table for
    one carrier set and transmitted on another; see meshlib.mesh_cfg.  No
    overrides = byte-identical to the previous behaviour.
    """
    over = dict(HS_OVERRIDES)
    if band:
        over["freq_lo"], over["freq_hi"] = float(band[0]), float(band[1])
    over.update(overrides)
    return build_config(sample_rate=int(sample_rate), **{**HS_BAND, **over})


def make_probe_audio(sample_rate: int, band: tuple | None = None) -> np.ndarray:
    return Modem(hs_cfg(sample_rate, band)).build_probe()


_PROBE_RMS_RATIO: dict = {}


def probe_rms_ratio(sample_rate: int, band: tuple | None = None) -> float:
    """rms/peak of the probe waveform at this rate+band (cached).

    This is the conversion between the one number a caller sets (`amplitude`, a
    PEAK limit) and the thing the channel measurement is actually pegged to (the
    probe's average power).
    """
    key = (int(sample_rate),
           None if band is None else (round(float(band[0]), 1),
                                      round(float(band[1]), 1)))
    if key not in _PROBE_RMS_RATIO:
        p = np.asarray(make_probe_audio(sample_rate, band), dtype=np.float64)
        _PROBE_RMS_RATIO[key] = (float(np.sqrt(np.mean(p ** 2)))
                                 / max(float(np.max(np.abs(p))), 1e-9))
    return _PROBE_RMS_RATIO[key]


def scale_hs_tx(audio, amplitude: float, sample_rate: int, *,
                is_probe: bool, band: tuple | None = None) -> tuple:
    """Level one handshake burst so it delivers the PROBE's average power.

    Returns (scaled float32 audio, telemetry dict).

    Every burst used to be PEAK-normalised to `amplitude`, which silently sends
    each waveform at a different average power -- its PAPR excess over the
    probe's.  Measured at amplitude=0.3, 400-8000 Hz: probe 10.8 dB PAPR,
    BAT_REPORT 13.3, BAT_ACK at ladder step G=8 14.9 dB, so the messages went
    out 2.5-4.1 dB QUIETER than the probe that graded the channel.  Audible as
    the handshake dropping ~25% in loudness after the probe (reported 20260804),
    and the reason a BAT_ACK is the most fragile message in the exchange: it is
    both the most heavily grouped and the quietest.  Run 20260804_034714 logged
    it at 69% EVM.
    The same reasoning, and the same fix, as meshlib.ofdm_tx_pcm for DATA frames
    ("Peak-normalising both to amplitude under-drives the data frame by its PAPR
    excess ... which erased the 6 dB allocation margin") -- the handshake simply
    never got it.

    The PROBE keeps peak normalisation: it defines the reference, so its own
    level must stay a pure function of `amplitude`.
    """
    a = np.asarray(audio, dtype=np.float64)
    pk = float(np.max(np.abs(a)))
    if pk <= 0:
        return a.astype(np.float32), {"scale": 0.0}
    rms = float(np.sqrt(np.mean(a ** 2)))
    info = {"papr_db": round(20.0 * math.log10(pk / max(rms, 1e-12)), 1)}
    if is_probe:
        scale = amplitude / pk
        info["mode"] = "peak(reference)"
    else:
        target = amplitude * probe_rms_ratio(sample_rate, band)
        scale = target / max(rms, 1e-12)
        info["mode"] = "rms-matched"
        # Headroom guard: high-PAPR bursts at a high `amplitude` would clip, and
        # a clipped burst is worse than a quiet one (broadband distortion that
        # also corrupts the ISI measurement).  Accept the deficit and report it.
        if pk * scale > 0.9:
            info["clamped_db"] = round(20.0 * math.log10(0.9 / (pk * scale)), 1)
            scale = 0.9 / pk
            info["mode"] = "rms-matched(peak-clamped)"
    info["delivered_rms"] = round(rms * scale, 5)
    info["peak"] = round(pk * scale, 4)
    return (a * scale).astype(np.float32), info


def hs_msg_cfg(sample_rate: int, band: tuple | None = None) -> dict:
    """Geometry for handshake MESSAGES, deliberately decoupled from the probe's.

    The probe's n_fft is whatever resolution we want to measure the channel at.
    A RobustLink message is a fixed ~1-bit-per-symbol BPSK mode, so inheriting
    that n_fft only stretches each of those bits: at n_fft=4096/44.1k a symbol
    is 116ms, versus 29ms at 1024.  Nothing about a message benefits from the
    finer grid, so messages always ride HS_MSG_NFFT.  Both ends run this same
    code, so the two agree without negotiating it.
    """
    return hs_cfg(sample_rate, band) if HS_MSG_NFFT is None else build_config(
        sample_rate=int(sample_rate),
        **{**HS_BAND, **HS_OVERRIDES, "n_fft": HS_MSG_NFFT,
           **({"freq_lo": float(band[0]), "freq_hi": float(band[1])}
              if band else {})})


# Payload spreading for messages sent AFTER the sender knows its own tx table.
# RobustLink's default is one bit smeared over every occupied carrier (~24dB
# processing gain) because the bootstrap transmitter is blind to its outgoing
# channel.  Once the peer has told it that table, that much spreading is pure
# waste: measured OTA 20260728, RobustLink ran at 31 bps while the link it had
# just negotiated ran at 7364 bps -- 234x.  Splitting the band into G groups
# (one bit each) recovers most of that; the ladder below is the set of G worth
# offering, and fast_groups_for picks from it.
# Bench (56B, simulated speaker/mic response): G=1 17.30s decodes at 0dB;
# G=8 2.91s needs ~5dB; G=32 1.37s needs ~10dB.  Step 3 (G=8) is the default
# ceiling because the margin at 5dB is what survives a channel that drifts
# between probe and data.
# RobustLink rate ladder.  RobustLink carries G bits per OFDM symbol: the band
# is split into G INTERLEAVED groups (ofdm_modem._rl_groups), each carrying one
# BPSK bit backed by n_occupied/G carriers of MRC combining.  So
#     bps = G * sample_rate / symbol_samples     (symbol_samples = n_fft*(1+cp))
# and every doubling of G doubles the rate for exactly 3dB of spreading gain.
# At HS_MSG_NFFT=1024 / cp 0.25, with worst-group occupancy measured on both
# directions of run 20260729_212926 (good link 198/237 active, bad link 85/237):
#
#  step  G  bps@48k bps@44.1k spread  cost |     good link |      bad link
#     0  1     37.5      34.5   269c   -0dB | 198c  23.0dB |  85c  19.3dB
#     1  2     75.0      68.9   134c   -3dB |  97c  19.9dB |  39c  15.9dB
#     2  4    150.0     137.8    67c   -6dB |  47c  16.7dB |  17c  12.3dB
#     3  8    300.0     275.6    33c   -9dB |  23c  13.6dB |   7c   8.5dB
#     4 16    600.0     551.2    16c  -12dB |   9c   9.5dB |   3c   4.8dB
#     5 32   1200.0    1102.5     8c  -15dB |   4c   6.0dB |   0c   DEAD
#
# G, not a bps figure, is the wire contract: sample_rate is DISCOVERED (HS_RATES
# trial detection), not known a priori, so two ends deriving G from a target bps
# would diverge whenever they disagreed about the peer's rate.  Pick a STEP; the
# achieved bps follows from whatever rate the link turns out to run at.
HS_RL_STEPS = (1, 2, 4, 8, 16, 32)

# Ceiling on the ladder step any message may use.  Step 3 (G=8, ~300bps at 48k)
# is the default: carriers are only marked active once they clear BPSK's 4.0dB
# threshold plus snr_margin_db=9, so every live carrier is >=13dB and even the
# fragmented direction's worst 7-carrier group lands ~21dB.  Step 4 is where
# that link gets thin (3 carriers).  Lower for a marginal deployment; raise when
# the band is known wide and clean.
HS_RL_MAX_STEP = 3

# Below this many LIVE carriers a group has too little spreading gain of its own
# to carry its bit, whatever the airtime saved.
#
# 6, not 8: carriers are only marked active once they clear BPSK's 4.0dB
# threshold plus snr_margin_db, so a live carrier is >=13dB and a group of 6
# combines to ~21dB -- far more than the ~4dB BPSK needs, before RS.  At 8 the
# fragmented direction of run 20260729_212926 missed step 3 by exactly one
# carrier (worst interleaved group = 7) and fell back to G=4, halving the rate
# on the link that most needs the airtime.  Raise it back toward 8 if OTA shows
# marginal groups failing; that is the first knob to blame for a step-3 decode
# failure on a sparse table.
HS_FAST_MIN_PER_GROUP = 6

# Step used by a sender that does not YET know its own tx table -- BAT_REPORT,
# and the first back-channel leg of ofdm2way.  There is no table to consult, so
# this is a BET.  The bet is cheap to lose on the DECODE side, because the
# receiver trials the whole ladder (try_decode_msg) and so never needs to be
# told which step was used; what it cannot cover is raw SNR, which is why this
# sits below HS_RL_MAX_STEP.  Step 2 gives every bit a quarter of the band --
# on the fragmented direction of run 20260729_212926 that is 17 live carriers
# (12.3dB) against 7 (8.5dB) at step 3 -- and a failed bootstrap message costs a
# WHOLE TURN to retransmit (probe + report), so the asymmetry favours caution
# here even though step 3 would probably survive.  0 restores the old
# always-robust G=1 behaviour.
HS_RL_BOOTSTRAP_STEP = 2

# Message types that NEVER take the bootstrap bet, whatever the step is set to.
# Not an SNR judgement -- a consequence-asymmetry one:
#   MSG_BAND is emitted BECAUSE the sender just graded the link as poor, so
#   spending spreading gain on the message that says "this link is bad" argues
#   against its own evidence.  Worse, the sender adopts the new band and
#   _restart()s IMMEDIATELY (see set_peer_band), while r_wait_probe's timeout
#   only re-listens and never re-sends BAND.  A lost BAND therefore leaves the
#   two ends decoding on DIFFERENT carrier grids permanently, with no path back
#   -- the one message in the protocol whose loss is unrecoverable.
#   MSG_RETRY is the recovery path itself; if it needs recovering, nothing does.
HS_RL_ALWAYS_ROBUST = frozenset({6, 4, 7, 8, 9})     # MSG_BAND, MSG_RETRY, MSG_ANNOUNCE, MSG_BLOCKOUT_REQ,
                                                      # MSG_BLOCKOUT_END (all sent before any BAT exists)


def geometry_step_cap(msg_cfg: dict) -> int:
    """Highest ladder step THIS message geometry can carry -- derived, not set.

    The ladder was characterised on the a-priori convention (HS_BAND at
    HS_MSG_NFFT).  Move the band or the FFT and the same step number means
    something different on the air, so an operator who narrows the band would
    otherwise have to re-derive a safe rate by hand.  This does it from the two
    things a blind sender can know without measuring the channel:

      spreading    carriers per group.  Narrowing the band leaves each group
                   fewer carriers to combine, so G must come down to hold the
                   per-bit processing gain the step was validated at.
      echo budget  the CP in SECONDS -- what the room's reverb is measured
                   against, and independent of FFT size.  A CP shorter than the
                   convention's leaves inter-symbol interference that spreading
                   cannot recover, because ISI scales with the signal rather
                   than sitting under it like noise.

    Both are ratios against the convention, so the convention itself always
    returns the configured step unchanged.  Returns an INDEX into HS_RL_STEPS.
    """
    fs = int(msg_cfg.get("sample_rate") or 48000)
    try:
        ref = build_config(sample_rate=fs,
                           **{**HS_BAND,
                              "n_fft": HS_MSG_NFFT or HS_BAND.get("n_fft", 1024)})
    except Exception:
        return len(HS_RL_STEPS) - 1          # cannot compare -- do not restrict
    ref_occ  = max(1, int(ref["n_occupied"]))
    ref_cp_s = ref["cp_len"] / max(1, int(ref["sample_rate"]))
    occ  = max(1, int(msg_cfg.get("n_occupied", 0) or 1))
    cp_s = int(msg_cfg.get("cp_len", 0)) / max(1, fs)

    # Reference against the CEILING step, not the bootstrap one.  This function
    # answers "what is the highest step this geometry can carry at all"; the
    # configured bootstrap/max steps are applied on top by the callers' min().
    # Referencing the bootstrap step instead would ceiling the measured-table
    # path at the blind rate, throwing away the speed a known table buys.
    base = HS_RL_STEPS[max(0, min(int(HS_RL_MAX_STEP), len(HS_RL_STEPS) - 1))]
    g_spread = occ / (ref_occ / base)                 # hold carriers-per-group
    g_cp     = base * (cp_s / ref_cp_s) if ref_cp_s > 0 else base
    g_max    = min(g_spread, g_cp)

    # Snap to the NEAREST step in log2, not the next one down.  The ladder is
    # 3dB per step, so flooring would charge a full halving for any shortfall
    # at all -- a 7% narrower band cost a 2x rate cut, which is not a tradeoff
    # anyone would make deliberately.  Nearest-in-log2 picks the step whose
    # spreading is closest to what the geometry actually supports, and callers
    # min() this against the configured step, so the result can only ever
    # LOWER the rate, never raise it above what the operator asked for.
    if g_max <= 1.0:
        return 0
    idx = int(round(math.log2(g_max)))
    return max(0, min(idx, len(HS_RL_STEPS) - 1))


def bootstrap_groups(msg_cfg: dict) -> int:
    """G for a sender with no tx table yet (see HS_RL_BOOTSTRAP_STEP).

    Capped by the geometry as well as the configured step, so changing the band
    or FFT lowers the blind rate automatically instead of silently keeping a
    step that was safe only for the convention.
    """
    step = max(0, min(int(HS_RL_BOOTSTRAP_STEP), int(HS_RL_MAX_STEP),
                      geometry_step_cap(msg_cfg), len(HS_RL_STEPS) - 1))
    cap = max(1, int(msg_cfg.get("n_occupied", 0)) // HS_FAST_MIN_PER_GROUP)
    return max(1, min(HS_RL_STEPS[step], cap))


# Back-compat alias for the old single "fast" group count.
HS_FAST_GROUPS = HS_RL_STEPS[HS_RL_MAX_STEP]


def rl_ladder() -> tuple:
    """Ladder steps permitted by HS_RL_MAX_STEP, coarsest last."""
    top = max(0, min(int(HS_RL_MAX_STEP), len(HS_RL_STEPS) - 1))
    return HS_RL_STEPS[:top + 1]


def rl_rate_bps(n_groups: int, cfg: dict) -> float:
    """Actual bit rate a given G realises on `cfg` -- what the step really buys."""
    return n_groups * cfg["sample_rate"] / cfg["symbol_samples"]


def fast_groups_for(bat_bytes: bytes | None, msg_cfg: dict) -> int:
    """How many groups a message may use, given the sender's known tx table.

    Returns 1 (full-band repetition, the always-safe bootstrap mode) when the
    table is unknown or too sparse to split.

    The previous rule tested `frac >= 0.5` globally and then estimated per-group
    occupancy as `n_occupied * frac` -- a UNIFORM smear, wrong in both
    directions on a real table.  On the fragmented direction of run
    20260729_212926 (85/237 active, frac 0.36) it refused to split at all,
    pinning that node to ~31bps for every message it sent; drop the frac test
    and the same smear then claims 12 carriers per group at G=8 where the
    contiguous split really had ONE.  Interleaved groups make near-uniform
    occupancy actually true, so this counts live carriers in the real groups
    rather than assuming them -- which also catches the one hazard striding
    introduces, a comb-shaped channel whose fade period aliases with G.

    Group occupancy is measured on the BAT's own grid (codes are per data
    carrier, the message rides n_occupied bins).  The two grids differ only by
    pilot insertion and both are near-uniform strided samplings, so the live
    COUNT per group carries over; it is a proxy, but a well-behaved one, and it
    only ever gates a threshold.
    """
    if not bat_bytes:
        return 1
    codes = _codes_of(bat_bytes)
    if not len(codes):
        return 1
    live = np.flatnonzero(codes > 0)
    if not len(live):
        return 1
    # Hard ceiling from the message geometry: a group cannot be given fewer
    # carriers than exist to split.  Independent of how alive the table looks.
    g_cap = max(1, int(msg_cfg.get("n_occupied", 0)) // HS_FAST_MIN_PER_GROUP)
    # Geometry ceiling applies even with a measured table: counting live
    # carriers proves a group has ENERGY, but says nothing about whether the CP
    # covers the room's echo.  ISI scales with the signal, so a carrier can
    # grade fully active and still smear -- no live-carrier count detects that.
    g_cap = min(g_cap, HS_RL_STEPS[geometry_step_cap(msg_cfg)])
    best = 1
    for G in rl_ladder():
        if G > g_cap:
            break
        counts = np.bincount(live % G, minlength=G)
        if counts.min() >= HS_FAST_MIN_PER_GROUP:
            best = G                      # coarsest step every group survives
    return best


def make_msg_audio(sample_rate: int, mtype: int, proof: int,
                   body: bytes, band: tuple | None = None,
                   my_tx_bat: bytes | None = None,
                   n_groups: int | None = None) -> np.ndarray:
    """`my_tx_bat` is the sender's OWN tx table if the peer has already told it
    (None before then).  With a table the step is chosen from it; without one
    the sender falls back to HS_RL_BOOTSTRAP_STEP rather than all the way to
    G=1, since the receiver trials the whole ladder either way.

    `n_groups` forces a specific G, bypassing both -- for airtime BUDGETING,
    which must size the worst case rather than the likely one.
    """
    cfg = hs_msg_cfg(sample_rate, band)
    if n_groups is None:
        if mtype in HS_RL_ALWAYS_ROBUST:
            n_groups = 1
        elif my_tx_bat:
            n_groups = fast_groups_for(my_tx_bat, cfg)
        else:
            n_groups = bootstrap_groups(cfg)
    rl = RobustLink(cfg, ecc_bytes=HS_ECC, n_groups=n_groups)
    return rl.encode(pack_msg(mtype, proof, body))


# --------------------------------------------------------------------------- #
# Data phase -- what the handshake is FOR                                      #
#                                                                              #
# The handshake ends the moment both ends agree on a bit table, so everything  #
# above exercises ChannelProbe and RobustLink and nothing else.  The main      #
# modem's data path -- AdaptiveModulator / AdaptiveDemodulator, the per-symbol #
# equaliser, pilot CPE tracking, hard demapping, RS -- had no offline coverage #
# at all.  These helpers close that: they take a NEGOTIATED table and move real #
# payload with it, so a data-path regression fails here instead of over the air.#
#                                                                              #
# Geometry is the trap.  A negotiated BAT is indexed on the handshake carrier   #
# set, and n_data_carriers depends on sample_rate, so the table for a 48 kHz    #
# transmitter is NOT the table for a 44.1 kHz one.  Every function here takes   #
# the TRANSMITTER's rate and derives the config from hs_cfg, which is why       #
# hs_cfg grew **overrides rather than this file growing a second band.          #
# --------------------------------------------------------------------------- #

HS_DATA_CHUNK = 256     # payload bytes per data frame


def hs_data_cfg(sample_rate: int, band: tuple | None = None,
                chunk_bytes: int = HS_DATA_CHUNK) -> dict:
    """Data-frame config: the handshake band and geometry, data-sized chunk.

    Same carrier set as hs_cfg by construction, so a negotiated BAT indexes it
    directly.  meshlib.mesh_cfg is this same config with MESH_CHUNK (48 B)
    instead -- a mesh payload is deliberately one frame, which is a MAC choice,
    not a link-rate one.
    """
    return hs_cfg(sample_rate, band, chunk_bytes=int(chunk_bytes))


def data_bat(bat_bytes: bytes, sample_rate: int, band: tuple | None = None,
             chunk_bytes: int = HS_DATA_CHUNK) -> np.ndarray:
    """Unpack a wire BAT onto the data carrier set at the TRANSMITTER's rate."""
    cfg = hs_data_cfg(sample_rate, band, chunk_bytes)
    return bytes_to_bat(bat_bytes, cfg["n_data_carriers"], what="data BAT")


def data_link_rates(bat: np.ndarray, sample_rate: int,
                    band: tuple | None = None,
                    chunk_bytes: int = HS_DATA_CHUNK) -> dict:
    """Airtime and throughput this table actually delivers.

    Measured by encoding one frame rather than re-deriving the framing
    arithmetic: the modulator already owns CE/SYNC insertion, RS expansion,
    OLA taper and the inter-frame gap, and a second copy of that sum here would
    be one more thing to drift.
    """
    cfg = hs_data_cfg(sample_rate, band, chunk_bytes)
    mod = AdaptiveModulator(cfg, bat)
    one = mod.encode_bytes(b"\x01" * cfg["chunk_bytes"])
    frame_s = len(one) / cfg["sample_rate"]
    return {
        "frame_s":   frame_s,
        "gross_bps": int(mod.bits_per_ofdm_sym * cfg["sample_rate"]
                         / cfg["symbol_samples"]),
        "net_bps":   int(cfg["chunk_bytes"] * 8 / frame_s),
        "n_carriers": int(np.sum(bat > 0)),
        "chunk_bytes": int(cfg["chunk_bytes"]),
    }


def data_payload_for(seconds: float, bat: np.ndarray, sample_rate: int,
                     band: tuple | None = None,
                     chunk_bytes: int = HS_DATA_CHUNK) -> bytes:
    """A deterministic payload that occupies about `seconds` of airtime.

    Printable ASCII, and never ending in NUL: encode_bytes NUL-pads the final
    chunk and reassemble() rstrips it, so a payload whose own tail is NUL
    cannot be told from padding on the wire -- the same ambiguity CSS hit and
    had to fix with an explicit message length.  Sizing it to a whole number of
    chunks sidesteps the padding entirely.
    """
    rates = data_link_rates(bat, sample_rate, band, chunk_bytes)
    n_frames = max(1, int(round(seconds / rates["frame_s"])))
    n = n_frames * rates["chunk_bytes"]
    alphabet = (b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                b"0123456789")
    return bytes(alphabet[i % len(alphabet)] for i in range(n))


def make_data_audio(sample_rate: int, payload: bytes, bat: np.ndarray,
                    band: tuple | None = None,
                    chunk_bytes: int = HS_DATA_CHUNK) -> np.ndarray:
    """Encode `payload` as data frames at the transmitter's rate."""
    cfg = hs_data_cfg(sample_rate, band, chunk_bytes)
    return AdaptiveModulator(cfg, bat).encode_bytes(payload)


def try_decode_data(capture: np.ndarray, mic_fs: int, tx_rate: int,
                    bat: np.ndarray, band: tuple | None = None,
                    chunk_bytes: int = HS_DATA_CHUNK, quiet: bool = True):
    """Decode data frames from a mic capture.  Returns (text, stats).

    Resamples to the TRANSMITTER's rate first, exactly as
    meshlib.try_decode_packet_ofdm does -- the BAT and the symbol grid are both
    defined there, not at the microphone.

    `quiet` swallows AdaptiveDemodulator's per-frame FRAME_STAT lines.  Those
    are a live protocol line that node scripts parse out of a subprocess's
    stdout, so they cannot be removed at the source; a sim that decodes tens of
    frames just should not spray them.
    """
    cfg = hs_data_cfg(tx_rate, band, chunk_bytes)
    dem = AdaptiveDemodulator(cfg, bat)
    cap = _resample(np.asarray(capture, dtype=np.float64), mic_fs, tx_rate)
    if quiet:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stats, chunks = dem.decode_samples(cap)
        # Re-emit backend banners that the suppression would otherwise eat.
        # Which backend actually ran must never be invisible: the whole value
        # of an opt-in C path is that a silent fallback to numpy cannot
        # masquerade as a passing C run, and swallowing the banner would hand
        # that failure mode straight back.  FRAME_STAT stays suppressed --
        # that is per-frame noise, not a fact about the run.
        for _l in buf.getvalue().splitlines():
            if "BACKEND=" in _l:
                print(_l, flush=True)
    else:
        stats, chunks = dem.decode_samples(cap)
    return dem.reassemble(chunks), stats


def recommend_band(spectrum: dict, shrink_thresh: float = 0.8) -> tuple | None:
    """Narrowest band still covering everything the allocator could use.

    The probe measures the whole convention band; when its edges grade dead the
    only thing they buy is airtime -- and airtime is the scarce resource,
    because the BAT (hence every handshake message) is ~3 bits per probed
    carrier.  Trimming a dead top octave at 14kHz takes the table from 95B to
    ~40B, which more than halves the message length.

    Returns (lo_hz, hi_hz) worth renegotiating for, or None to stay put:
    None when nothing is active (a failed measurement -- narrowing on that
    would lock the link into whatever noise happened to grade highest), and
    None when the trim saves less than `shrink_thresh` of the span (not worth
    a retrain).
    """
    freqs = spectrum.get("freq_hz") or []
    bits  = spectrum.get("bits") or []
    active = [f for f, b in zip(freqs, bits) if b > 0]
    if not active or len(freqs) < 2:
        return None
    step = (freqs[-1] - freqs[0]) / max(1, len(freqs) - 1)
    lo = max(0.0, min(active) - step)          # one carrier of guard each side
    hi = max(active) + step
    span_old = freqs[-1] - freqs[0]
    if span_old <= 0 or (hi - lo) / span_old > shrink_thresh:
        return None
    return (round(lo, 1), round(hi, 1))


def _resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out or len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    from scipy.signal import resample_poly
    g = gcd(fs_in, fs_out)
    return resample_poly(np.asarray(x, np.float64),
                         fs_out // g, fs_in // g).astype(np.float32)


def try_decode_probe(capture: np.ndarray, mic_fs: int,
                     rates=HS_RATES, threshold: float = 0.25,
                     min_syms: int = 12, band: tuple | None = None):
    """Trial-decode a probe whose TX sample rate is unknown.

    Evaluates ALL candidate rates and keeps the best by coherence (a probe at
    the wrong rate can partially validate -- 1/9 of carriers land phase-aligned
    at the 44.1k/48k ratio -- so first-valid is not safe).  `min_syms` rejects
    fragmentary locks; a real handshake probe yields >150 symbols.

    Returns (tx_rate, bat_bytes, info) or None.  `bat_bytes` is the packed
    bit-allocation table for the probe's direction (the PEER's tx table).
    """
    best = None
    for rate in rates:
        cap = _resample(capture, mic_fs, rate)
        cfg = hs_cfg(rate, band)
        modem = Modem(cfg)
        res = modem.analyze_probe(cap, threshold=threshold)
        if not res.get("valid") or res["n_syms"] < min_syms:
            continue
        # Completeness: if the symbol run extends to the end of the capture,
        # the probe is still PLAYING -- deciding now yields a truncated
        # measurement (84/171 syms seen OTA) and, worse, replying talks over
        # the peer.  Wait for the next poll instead.
        end = res["frame_start"] + (res["n_syms"] + 2) * cfg["symbol_samples"]
        if end > len(cap):
            continue
        if best is not None and res.get("coherence", 0) <= best[2].get("coherence", 0):
            continue
        fine = modem.allocate(res)
        bat  = bat_fit_wire(fine)
        bat_cost = {"wire_b": len(bat_rle_pack(bat_to_bytes(bat))),
                    "bits_fine": int(fine.sum()),
                    "bits_sent": int(bat.sum())}
        snr = res["snr_db"]
        info = {
            "tx_rate":       rate,
            "coherence":     res.get("coherence"),
            "n_syms":        res["n_syms"],
            # frame_start in MIC-rate samples (callers slice mic-rate capture)
            "start_mic":     int(res["frame_start"] * mic_fs / rate),
            # capture time remaining AFTER the symbol run ended: small values
            # mean the run ended at the live edge -- a still-playing probe
            # whose coherence run broke early (dropout), not a finished one
            "tail_s":        round((len(cap) - end) / rate, 2),
            "snr_median_db": round(float(np.median(snr)), 1),
            "snr_max_db":    round(float(snr.max()), 1),
            # Reverberation ceiling.  snr_* above is what the allocator used;
            # when isi_derate is on that is already de-rated, so the RAW probe
            # numbers are kept beside it -- the gap between the two pairs is
            # exactly how much the periodic probe was over-promising.
            "isi_ratio":     (None if res.get("isi_ratio") is None
                              else round(float(res["isi_ratio"]), 3)),
            "isi_ceiling_db": (None if not res.get("isi_ratio") else
                               round(-10.0 * math.log10(
                                   max(float(res["isi_ratio"]), 1e-9)), 1)),
            "isi_applied":   bool(res.get("isi_ratio")
                                  and res.get("isi_derate", 0) > 0
                                  and res["isi_ratio"] <= _ISI_BETA_MAX),
            "isi_unmeasurable": bool(res.get("isi_ratio")
                                     and res["isi_ratio"] > _ISI_BETA_MAX),
            # Why there is no ceiling, when there is none.  A bare null here
            # is indistinguishable from "the room is clean", and the allocator
            # quietly falls back to the flat snr_margin_db guess.
            "isi_reason":    res.get("isi_reason"),
            "bat_summary":   modem.bat_summary(bat),
            "n_carriers":    int(cfg["n_data_carriers"]),
            "n_active":      int(np.sum(bat > 0)),
            "bat_cost":      bat_cost,
            # per-subcarrier frequency/SNR/bits for the report's BAT chart;
            # this is the only place the measurement still exists unreduced
            "spectrum":      carrier_spectrum(cfg, snr, bat),
        }
        best = (rate, bat_to_bytes(bat), info)
    return best


def try_decode_msg(capture: np.ndarray, mic_fs: int, rates=HS_RATES,
                   band: tuple | None = None, fast_ok: bool = True,
                   stats_out: dict | None = None):
    """Trial-decode a RobustLink message whose TX sample rate is unknown.

    Returns (tx_rate, mtype, proof_crc, body) or None.

    EVERY ladder step is attempted, not an agreed one: which step a peer used
    is not something the receiver can infer -- a lost BAT_REPORT leaves the peer
    still transmitting in the robust mode, and its retransmission must stay
    decodable.  Trying the whole ladder means the sender's step is discovered
    rather than negotiated, so a node's rate can change without a coordinated
    deploy, and BAT_REPORT (sent before its sender knows its own tx table, so
    never eligible for fast_groups_for) can still ride a fast step.

    Cheap: the resync cadence counts SYMBOL index in both encode and
    _walk_bits, so the symbol grid is identical at every G -- one acquisition
    and one set of FFTs serve all steps, and only the bit unpacking differs.
    The coherence scan dominates decode cost and is run once.

    Wrong-step attempts cannot produce a false positive: RS parity plus the
    body CRC in unpack_msg both have to pass.

    `stats_out`, when given, is filled with the successful decode's telemetry
    (EVM, per-group EVM, margin, coherence -- see RobustLink._walk_bits).  That
    is the only quality figure a handshake-only run produces: there is no data
    phase to measure, so without it a report has nothing to say about how much
    margin the BAT exchange actually had.
    """
    groups = (1,) if not fast_ok else rl_ladder()
    for rate in rates:
        cap = _resample(capture, mic_fs, rate)
        rl = RobustLink(hs_msg_cfg(rate, band), ecc_bytes=HS_ECC)
        # one acquisition, both spreadings tried at each lock
        payload = rl.decode(cap, groups=groups)
        if payload is None:
            continue
        msg = unpack_msg(payload)
        if msg is None:
            continue
        if stats_out is not None and rl.last_stats:
            stats_out.clear()
            stats_out.update(rl.last_stats)
            stats_out["mtype"] = msg[0]
            stats_out["tx_rate"] = rate
        return (rate,) + msg
    return None


# --------------------------------------------------------------------------- #
# selftest -- pure logic, lossy delivery, no DSP
# --------------------------------------------------------------------------- #

def _run_logic_pair(drop_plan: set[str], max_steps: int = 60) -> tuple:
    """Run two SMs through in-memory delivery.  `drop_plan` names
    transmissions to drop once, e.g. {"r:probe:0", "r:msg1:0", "i:msg2:1"}
    (side : probe|msg<type> : occurrence-index).

    Delivery is GATED by the receiver's expecting(), exactly like the real
    bindings, which only attempt the decode type(s) the SM asks for.  An
    ungated selftest once passed a "drop R probe" case whose recovery path
    was unreachable over real audio (the responder in r_wait_ack could never
    hear the initiator's re-probe)."""
    ini = HandshakeSM("initiator")
    rsp = HandshakeSM("responder")
    bat_i = bytes(range(20))          # stand-in packed BATs
    bat_r = bytes(range(50, 77))
    now = 0.0
    ini.start(now); rsp.start(now)
    counts: dict[str, int] = {}
    dropped: list[str] = []

    def deliver(side, sm_from, sm_to, peer_bat):
        nonlocal now
        acts, sm_from.actions = sm_from.actions, []
        for a in acts:
            if a[0] in ("done", "fail"):
                continue
            if a[0] == "send_probe":
                key_base = f"{side}:probe"
            else:
                key_base = f"{side}:msg{a[1]}"
            n = counts.get(key_base, 0)
            counts[key_base] = n + 1
            key = f"{key_base}:{n}"
            if key in drop_plan:
                dropped.append(key)
                continue
            now += 8.0                                   # airtime
            exp = sm_to.expecting() or ""
            if a[0] == "send_probe":
                if "probe" in exp:                       # binding decode gate
                    sm_to.heard_probe(peer_bat, now)
            else:
                if "msg" in exp:
                    sm_to.heard_msg(a[1], a[2], a[3], now)

    for _ in range(max_steps):
        if (ini.done or ini.failed) and (rsp.done or rsp.failed):
            break
        deliver("i", ini, rsp, bat_i)   # rsp measures BAT_I from ini's probe
        deliver("r", rsp, ini, bat_r)
        now += 5.0
        ini.tick(now); rsp.tick(now)
    return ini, rsp, bat_i, bat_r, dropped


def cmd_selftest() -> int:
    cases = [
        ("clean",             set()),
        ("drop R probe",      {"r:probe:0"}),
        ("drop R probe x2",   {"r:probe:0", "r:probe:1"}),
        ("drop BAT_REPORT",   {"r:msg1:0"}),
        ("drop BAT_ACK",      {"i:msg2:0"}),
        ("drop GO",           {"r:msg3:0"}),
        ("drop FIN",          {"i:msg5:0"}),
        ("drop GO and FIN",   {"r:msg3:0", "i:msg5:0"}),
        ("drop I probe",      {"i:probe:0"}),
        ("drop two in a row", {"r:msg1:0", "i:msg2:0"}),
    ]
    ok_all = True
    for name, plan in cases:
        ini, rsp, bat_i, bat_r, dropped = _run_logic_pair(plan)
        ok = (ini.done and rsp.done
              and ini.my_tx_bat == bat_i and rsp.my_tx_bat == bat_r
              and ini.peer_tx_bat == bat_r and rsp.peer_tx_bat == bat_i)
        ok_all &= ok
        print(f"  [{name:18}] {'PASS' if ok else 'FAIL'}  "
              f"(dropped {len(dropped)}; ini={ini.state} rsp={rsp.state})")
        if not ok:
            print("--- initiator log ---"); print(ini.log())
            print("--- responder log ---"); print(rsp.log())
    # a hopeless channel must FAIL, not hang
    ini, rsp, *_ = _run_logic_pair({f"r:msg1:{i}" for i in range(10)})
    hopeless_ok = ini.failed and not ini.done
    ok_all &= hopeless_ok
    print(f"  [{'unrecoverable loss':18}] {'PASS' if hopeless_ok else 'FAIL'}  "
          f"(ini={ini.state})")
    ok_all &= _selftest_band()
    print("SELFTEST", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


def _selftest_band() -> bool:
    """Band renegotiation: recommend_band's policy, the MSG_BAND wire format,
    and that a retrain converges instead of ping-ponging."""
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{name:18}] {'PASS' if cond else 'FAIL'}")

    # -- policy: propose only when a real slice of the band is dead --
    freqs = [400.0 + 50 * i for i in range(200)]        # 400..10350 Hz
    live_low = {"freq_hz": freqs,
                "bits": [2 if f < 4000 else 0 for f in freqs]}
    all_live = {"freq_hz": freqs, "bits": [2] * len(freqs)}
    all_dead = {"freq_hz": freqs, "bits": [0] * len(freqs)}
    b = recommend_band(live_low)
    check("dead top -> narrow", b and b[1] < 4200 and b[0] <= 400)
    check("all live -> keep",   recommend_band(all_live) is None)
    check("all dead -> keep",   recommend_band(all_dead) is None)
    # a trim too small to pay for a retrain must be declined
    minor = {"freq_hz": freqs, "bits": [2 if f < 9800 else 0 for f in freqs]}
    check("minor trim -> keep", recommend_band(minor) is None)

    # -- wire format round-trips, and rejects nonsense --
    rt = unpack_band(pack_band(400, 6000))
    check("band round-trip", rt == (BAND_DIR_PEER_TX, 400.0, 6000.0))
    check("band rejects junk", unpack_band(b"\x00\x00\x00\x00\x00") is None
                               and unpack_band(b"\x00") is None)

    # -- retrain converges: responder pushes a band, both ends restart once --
    ini = HandshakeSM("initiator", band_negotiation=True)
    rsp = HandshakeSM("responder", band_negotiation=True)
    ini.start(0.0); rsp.start(0.0)
    ini.actions.clear()
    new_band = (400.0, 6000.0)
    rsp.heard_probe(bytes(range(20)), 1.0, band=new_band)       # R -> BAND
    band_msgs = [a for a in rsp.actions if a[0] == "send_msg" and a[1] == MSG_BAND]
    check("responder emits BAND", len(band_msgs) == 1)
    check("responder restarted", rsp.state == "r_wait_probe"
                                 and rsp.peer_tx_band == new_band)
    ini.heard_msg(MSG_BAND, 0, band_msgs[0][3], 2.0)
    check("initiator adopts band", ini.my_tx_band == new_band
                                   and ini.state == "i_wait_probe")
    check("initiator re-probes",
          any(a[0] == "send_probe" for a in ini.actions))
    # cap holds: a second proposal must NOT retrain again
    rsp.actions.clear()
    rsp.heard_probe(bytes(range(20)), 3.0, band=(400.0, 3000.0))
    check("retrain cap holds",
          not any(a[0] == "send_msg" and a[1] == MSG_BAND for a in rsp.actions))

    # -- the two directions stay independent --
    check("directions independent",
          ini.peer_tx_band is None and rsp.my_tx_band is None)
    # having ADOPTED a band, the initiator must still be able to IMPOSE one on
    # the other direction -- the budgets are per-direction, not shared
    ini.actions.clear()
    ini.heard_probe(bytes(range(20)), 4.0, band=(400.0, 5000.0))
    check("adopt != spend propose",
          any(a[0] == "send_msg" and a[1] == MSG_BAND for a in ini.actions)
          and ini.peer_tx_band == (400.0, 5000.0))
    return ok


# --------------------------------------------------------------------------- #
# simtest -- full DSP through simulated acoustic channels
# --------------------------------------------------------------------------- #

# Room impulse responses.  The frequency-shaping filters above model the
# SPEAKER and MIC; this models the ROOM between them, and until it existed the
# entire simulated suite ran on a zero-phase magnitude filter -- no delay
# spread, no reflections, no reverberation.  That is the one impairment the
# acoustic link is actually limited by (see isi_ratio's docstring and the
# OTA runs it cites), so every sim result was blind to it: simtest and the
# impulse benchmark passed while OTA frames ran 44-63% EVM.
#
# Each profile is (early_taps, rt60_s, diffuse_ratio):
#   early_taps     (delay_ms, amplitude) discrete reflections, direct = 1.0 @ 0
#   rt60_s         decay time of the diffuse tail
#   diffuse_ratio  tail ENERGY as a fraction of the direct path's
#
# The delays are chosen against the guard the modem actually has.  At 48 kHz /
# n_fft=1024 / cp_ratio=0.25 the cyclic prefix is 5.33 ms, and the W-OFDM taper
# eats `taper_samples` of it (2 ms by default), leaving ~3.3 ms = 1.14 m of
# excess path.  So "desk" sits entirely inside the guard, "room" straddles it,
# and "reverb" is mostly outside it -- which is what makes the three grade
# different things rather than the same thing at three volumes.
SIM_ROOMS: dict = {
    # Control.  Single tap, no tail: reproduces the pre-room behaviour exactly
    # and is what a regression should show no change against.
    "none":     ([], 0.0, 0.0),
    # Two laptops a foot apart on a hard desk.  One strong desk bounce, short
    # tail.  Matches the OTA case isi_ratio records as coherence 0.996-0.998,
    # beta 0.17-0.19 -- measurable, and the ISI ceiling predicted it.
    "desk":     ([(0.9, 0.50), (1.7, 0.28), (3.1, 0.16)], 0.20, 0.06),
    # Ordinary office, a few metres apart.  Reflections straddle the guard:
    # this is the profile the CP/taper trade is decided on.
    "room":     ([(2.4, 0.45), (4.6, 0.30), (7.2, 0.20), (11.0, 0.12)],
                 0.45, 0.25),
    # Hard-surfaced room.  Most of the energy arrives after the guard; beta
    # saturates and the ISI estimator cannot see past its own n_fft wrap, which
    # is the regime _ISI_BETA_MAX exists to refuse to act on.
    "reverb":   ([(3.0, 0.55), (6.5, 0.45), (9.8, 0.35), (14.0, 0.28),
                  (19.0, 0.20)], 0.80, 0.70),
}
SIM_ROOM_DEFAULT = "none"

# Sample-clock offset between the two sound cards, in ppm.  Independent
# consumer ADC/DACs sit tens of ppm apart, and the effect is NOT a frequency
# shift the CPE estimator can absorb: it is a timing drift that accumulates
# across the frame, i.e. a phase ramp LINEAR IN CARRIER INDEX that grows with
# symbol index.  At 50 ppm over a 33-symbol body (0.7 s) that is ~1.7 samples,
# which is ~1.8 rad of differential phase at bin 170.  `_estimate_cpe` fits a
# single scalar and structurally cannot represent it, so with ppm=0 the sim
# could never show the cost.  0 keeps the old behaviour.
SIM_PPM_DEFAULT = 0.0


def _room_ir(fs: int, room: str, seed: int) -> np.ndarray:
    """Impulse response for a named profile.  Direct path is tap 0, gain 1."""
    spec = SIM_ROOMS.get(room)
    if spec is None:
        raise ValueError(f"unknown room {room!r}; have {sorted(SIM_ROOMS)}")
    early, rt60, diffuse = spec
    if not early and diffuse <= 0.0:
        return np.array([1.0])

    tail_n = int(round(rt60 * fs)) if rt60 > 0 else 0
    last_early = int(round(max((d for d, _ in early), default=0.0)
                           * fs / 1000.0))
    n = max(last_early, tail_n) + 1
    h = np.zeros(n)
    h[0] = 1.0
    for delay_ms, amp in early:
        h[int(round(delay_ms * fs / 1000.0))] += amp

    if tail_n > 0 and diffuse > 0.0:
        rs = np.random.RandomState(seed ^ 0x5EED)
        t = np.arange(tail_n) / fs
        # -60 dB at rt60 => amplitude decay exp(-t * ln(1000)/rt60).
        env = np.exp(-t * (math.log(1000.0) / rt60))
        tail = rs.randn(tail_n) * env
        # Start the diffuse field after the last discrete reflection so the
        # two are not double-counted, then scale it to the requested energy
        # RELATIVE TO THE DIRECT PATH (h[0]**2 == 1), which is what makes
        # diffuse_ratio comparable across profiles.
        tail[:last_early] = 0.0
        e = float(np.sum(tail ** 2))
        if e > 0:
            h[:tail_n] += tail * math.sqrt(diffuse / e)
    return h


def _apply_ppm(x: np.ndarray, ppm: float) -> np.ndarray:
    """Resample by (1 + ppm*1e-6) to model a sound-card clock offset.

    Band-limited (FFT) resampling, not linear interpolation: at these ratios a
    linear interpolator's own lowpass would be a bigger impairment than the
    offset it is meant to model.  The signal is zero-padded before the
    transform and trimmed after, because scipy's FFT resampler is circular and
    the capture is not periodic.
    """
    if not ppm or len(x) == 0:
        return np.asarray(x, dtype=np.float64)
    from scipy.signal import resample
    pad = 2048
    xp = np.concatenate([np.zeros(pad), np.asarray(x, np.float64),
                         np.zeros(pad)])
    ratio = 1.0 / (1.0 + ppm * 1e-6)
    m = max(8, int(round(len(xp) * ratio)))
    y = resample(xp, m)
    lo = int(round(pad * ratio))
    hi = m - int(round(pad * ratio))
    return y[lo:hi] if hi > lo else y


def _sim_channel(x: np.ndarray, fs: int, kind: str, snr_db: float,
                 seed: int, room: str | None = None,
                 ppm: float | None = None) -> np.ndarray:
    """Speaker/mic colouration, then the ROOM, then clock offset, then noise.

    The two LTI stages commute, so applying the magnitude filter before the
    room convolution is equivalent to the physical order.  The clock offset
    belongs at the ADC, hence after the room; the noise is mic noise, so it is
    added last and is NOT resampled.

    `room`/`ppm` default to the module-level SIM_ROOM_DEFAULT / SIM_PPM_DEFAULT
    so a suite can select a profile once at startup without threading it
    through every call site.  Both default to the pre-room behaviour.
    """
    from numpy.fft import rfft, irfft, rfftfreq
    F = rfftfreq(len(x), 1 / fs)
    if kind == "good":       # decent speaker: strong lows, HF rolloff
        H = np.exp(-((F - 900) / 2600) ** 2) + 0.35 * np.exp(-((F - 2700) / 900) ** 2)
    else:                    # tiny speaker: highpass, only 3-4 kHz alive
        H = 1 / (1 + np.exp(-(F - 2500) / 300)) * np.exp(-((F - 3500) / 2500) ** 2)
    H = H / H.max()
    y = irfft(rfft(x) * H, n=len(x))

    ir = _room_ir(fs, SIM_ROOM_DEFAULT if room is None else room, seed)
    if len(ir) > 1:
        # 'full' then trim to len(y): the reverb tail past the end of the
        # transmission is real and the capture assembly appends quiet after
        # this, so keeping it would just move the trim point.  Trimming here
        # keeps every caller's length arithmetic unchanged.
        y = np.convolve(y, ir)[:len(y)]

    y = _apply_ppm(y, SIM_PPM_DEFAULT if ppm is None else ppm)

    p = float(np.mean(y ** 2))
    return y + np.random.RandomState(seed).randn(len(y)) * np.sqrt(p / 10 ** (snr_db / 10))


DATA_SIM_S = 10.0    # payload airtime per direction in the sim's data phase

# Chunk sizes the data phase sweeps.  Not arbitrary: frame length scales with
# chunk, and _find_sc_peaks' min_gap scales with frame length, so the chunk
# size sets how much of a capture a single false SC lock can suppress.  48 B is
# meshlib's MESH_CHUNK (a mesh payload is deliberately one frame); 256 B is a
# bulk frame, and it is the size at which the startup-click false-lock was
# first caught costing frame 0 on the weak direction.  Sweeping both keeps that
# interaction covered from both ends instead of at one point.
DATA_SIM_CHUNKS = (48, 256)


def _sim_leading_junk(fs: int, seed: int) -> np.ndarray:
    """Audio-stream startup click plus a noise floor, ahead of every capture.

    The decaying impulse is the thing Schmidl-Cox false-locks on (it scores ~1
    on any delay-correlator), so every capture in the sim carries one -- the
    data phase needs it as much as the handshake does.
    """
    c = np.zeros(int(0.2 * fs)); c[40] = 0.8
    for i in range(41, len(c)):
        c[i] = c[i - 1] * 0.994
    return np.concatenate(
        [c, np.random.RandomState(seed).randn(int(0.4 * fs)) * 1e-4])


def _sim_data_phase(ini, rsp, spk, mic, chan, rng,
                    seconds: float = DATA_SIM_S,
                    chunk_sizes=DATA_SIM_CHUNKS) -> bool:
    """Move real payload both ways using the tables the handshake negotiated.

    Same channel, levelling and capture assembly as the message exchange, so
    the data sees exactly the conditions its BAT was measured under.  Each
    direction transmits at its OWN speaker rate and is decoded against that
    rate, which is the pairing that a geometry mistake breaks.
    """
    print(f"\n--- data phase ({seconds:.0f}s payload per direction) ---")
    all_ok = True
    for chunk_bytes in chunk_sizes:
        for side, sm in (("initiator", ini), ("responder", rsp)):
            tx_rate = spk[side]
            fs_rx   = mic["responder" if side == "initiator" else "initiator"]

            bat     = data_bat(sm.my_tx_bat, tx_rate, chunk_bytes=chunk_bytes)
            rates   = data_link_rates(bat, tx_rate, chunk_bytes=chunk_bytes)
            payload = data_payload_for(seconds, bat, tx_rate,
                                       chunk_bytes=chunk_bytes)

            audio      = make_data_audio(tx_rate, payload, bat,
                                         chunk_bytes=chunk_bytes)
            audio, lvl = scale_hs_tx(audio, 0.3, tx_rate, is_probe=False)
            air = _sim_channel(audio, tx_rate, chan[side], snr_db=14,
                               seed=int(rng.randint(1e6)))
            capture = np.concatenate([
                _sim_leading_junk(fs_rx, int(rng.randint(1e6))),
                _resample(air, tx_rate, fs_rx),
                np.random.RandomState(int(rng.randint(1e6))).randn(
                    int(0.3 * fs_rx)) * 1e-4,
            ])

            text, stats = try_decode_data(capture, fs_rx, tx_rate, bat,
                                          chunk_bytes=chunk_bytes)
            got   = (text or "").encode("utf-8", errors="replace")
            exact = (got == payload)
            n_exp = len(payload) // rates["chunk_bytes"]
            n_got = len({s["seq"] for s in stats})
            evm   = (sum(s["evm_pct"] for s in stats) / len(stats)
                     ) if stats else 0.0
            all_ok &= exact

            print(f"  chunk {chunk_bytes:3d}B  {side:9s} tx@{tx_rate} "
                  f"{rates['n_carriers']:3d} carriers  "
                  f"{len(payload):5d}B in {len(audio) / tx_rate:5.1f}s  "
                  f"gross {rates['gross_bps']:5d} / net {rates['net_bps']:5d} bps  "
                  f"frames {n_got:3d}/{n_exp:<3d}  EVM {evm:4.1f}%  "
                  f"payload {'EXACT' if exact else 'MISMATCH'}")
            if not exact:
                i = next((j for j, (a, b) in enumerate(zip(payload, got))
                          if a != b), min(len(payload), len(got)))
                print(f"      expected {len(payload)}B got {len(got)}B, "
                      f"first difference at byte {i}")
    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Impulse-obscuration benchmark
# ──────────────────────────────────────────────────────────────────────────────

# Symbols at the head of every superframe that are not payload: the
# Schmidl-Cox preamble, the channel-estimate symbol and the first SYNC.  See
# AdaptiveModulator's layout diagram.  The benchmark's symbol grid is anchored
# past these so it grades the DATA path; obscuring the SC preamble instead
# grades acquisition, which is a different question with its own coverage
# (the false-lock work) and which would otherwise dominate every cell.
BENCH_HEADER_SYMS = 3

BENCH_CHUNK   = 256      # bulk frame, not meshlib's 48 B single-frame payload
BENCH_SECONDS = 3.0      # payload airtime per trial; the grid runs many trials
BENCH_SNR_DB  = 14       # same channel SNR the handshake sim uses

# Burst amplitude as a multiple of signal RMS.  MEASURED, not inherited: at
# benchlib's default 6x every cell scored 0 and the grid resolved nothing.
# Sweeping a single 0.5-symbol burst found the cliff between 0.75x and 1.0x --
# at 0.75x the burst puts 7 byte errors into the frame and the interleaver
# corrects them (6/1/0 across the three codewords, all within t=8), while at
# 1.0x and above the frame is lost outright with ALL codewords failing, i.e.
# far more damage than the ~17 bytes one OFDM symbol carries.  That cliff is
# upstream of the FEC and is not a coding problem; see the benchmark notes in
# tests/regression/README.md.  0.75x puts the coverage ladder in range.
BENCH_AMP_RATIO = 0.75


def _bench_reference(fs: int, chan: str, rng):
    """Encode one payload and measure the BAT the link would actually negotiate.

    The table comes from a real probe over the same channel rather than from
    `bpsk_all_bat()`: an all-BPSK table is far more robust than anything the
    link would choose in practice, so it would score near 100 everywhere and
    the grid would say nothing.  Encoding once and reusing the audio across
    every cell keeps the impairment the only variable.
    """
    probe = make_probe_audio(fs)
    probe, _ = scale_hs_tx(probe, 0.3, fs, is_probe=True)
    air = _sim_channel(probe, fs, chan, snr_db=BENCH_SNR_DB,
                       seed=int(rng.randint(1e6)))
    capture = np.concatenate([
        _sim_leading_junk(fs, int(rng.randint(1e6))), air,
        np.random.RandomState(int(rng.randint(1e6))).randn(int(0.3 * fs)) * 1e-4])
    got = try_decode_probe(capture, fs)
    if got is None:
        raise RuntimeError("benchmark: probe decode failed on a clean channel")
    _rate, bat_bytes, info = got

    bat     = data_bat(bat_bytes, fs, chunk_bytes=BENCH_CHUNK)
    payload = data_payload_for(BENCH_SECONDS, bat, fs, chunk_bytes=BENCH_CHUNK)
    audio   = make_data_audio(fs, payload, bat, chunk_bytes=BENCH_CHUNK)
    audio, _lvl = scale_hs_tx(audio, 0.3, fs, is_probe=False)
    return bat, payload, audio, info


def cmd_benchmark(seconds=BENCH_SECONDS, trials=2, chan="good",
                  strict=False, widths=None, aligns=None, out=None,
                  amp_ratio=BENCH_AMP_RATIO) -> int:
    """Score the OFDM data path 0-100 against deterministic impulse obscuration.

    Same axes, ladder and scoring as the CSS benchmark, so the two numbers are
    read the same way -- but note the symbol is ~27 ms here against ~1 s there,
    so a 100 ms absolute burst spans several OFDM symbols and is clamped to
    one; the fractional widths are the meaningful ones for this modem, and the
    absolute one is kept because real clicks do not scale with n_fft.

    The impairment is applied to the CAPTURE, after the channel, because that
    is where an acoustic impulse actually enters -- injecting it before the
    speaker response would filter the click by the transmitter's band and
    measure something softer than the thing being modelled.

    Measured behaviour, which is NOT what the cyclic prefix would suggest: the
    `end` column is the strongest (100 at 1/2 symbol) and `start`/`middle`
    collapse to ~10 at the same width.  A 1/2-symbol burst is 640 samples
    against a 256-sample CP, so most of it lands inside the FFT window wherever
    it starts, and the CP absorbs nothing.  The alignment axis is swept rather
    than reasoned about precisely because this went the other way.
    """
    fs = 48000
    rng = np.random.RandomState(7)
    cfg = hs_data_cfg(fs, chunk_bytes=BENCH_CHUNK)
    stride = int(cfg["symbol_samples"])

    global BENCH_SECONDS
    BENCH_SECONDS = float(seconds)
    bat, payload, audio, info = _bench_reference(fs, chan, rng)
    print(f"OFDM impulse benchmark  fs={fs} chan={chan} snr={BENCH_SNR_DB}dB "
          f"chunk={BENCH_CHUNK}B amp={amp_ratio}x")
    print(f"  reference: {info['bat_summary']}")
    print(f"  payload {len(payload)}B in {len(audio)/fs:.1f}s, "
          f"symbol {stride} samples ({stride/fs*1e3:.1f} ms, "
          f"cp {cfg['cp_len']} = {cfg['cp_len']/stride*100:.0f}% of it)")
    print(f"  {trials} trial(s) per level\n")

    junk_len = len(_sim_leading_junk(fs, 0))
    offset = junk_len + BENCH_HEADER_SYMS * stride

    def _decode(capture):
        text, _stats = try_decode_data(capture, fs, fs, bat,
                                       chunk_bytes=BENCH_CHUNK)
        return (text or "").encode("utf-8", errors="replace") == payload

    def _channel(seed_rng):
        air = _sim_channel(audio, fs, chan, snr_db=BENCH_SNR_DB,
                           seed=int(seed_rng.randint(1e6)))
        return np.concatenate([
            _sim_leading_junk(fs, int(seed_rng.randint(1e6))), air,
            np.random.RandomState(int(seed_rng.randint(1e6))).randn(
                int(0.3 * fs)) * 1e-4])

    # Clean control, before any cell.  This grid can legitimately come back all
    # zeros -- one obscured OFDM symbol puts more byte errors into a chunk than
    # its RS parity can correct -- and an all-zero grid is indistinguishable
    # from a benchmark whose decode never worked at all.  So prove the
    # unimpaired path first and refuse to report a score without it.
    if not _decode(_channel(rng)):
        print("  CONTROL FAILED: the unimpaired payload did not decode. "
              "The benchmark cannot attribute anything to the impairment; "
              "fix the clean path first.")
        return 1
    print("  control: unimpaired payload decodes exactly\n")

    def make_run_level(width, align):
        def run_level(every_n):
            ok = n = 0
            for _t in range(trials):
                capture, n = benchlib.apply_obscuration(
                    _channel(rng), fs, stride, every_n, width, align,
                    np.random.default_rng(int(rng.randint(1e6))),
                    offset=offset, amp_ratio=amp_ratio)
                ok += int(_decode(capture))
            rate = ok / trials
            print(f"    {width.label():>10s} {align:<7s} 1-in-{every_n:<3d} "
                  f"({benchlib.coverage_pct(every_n):5.2f}%, {n:3d} bursts) -> "
                  f"{rate*100:3.0f}% pass", flush=True)
            return rate
        return run_level

    cells = []
    for width, align in benchlib.grid_axes(widths, aligns):
        cells.append(benchlib.sweep_cell(
            make_run_level(width, align), width, align, strict=strict,
            floor_ladder=benchlib.FLOOR_LADDER, geom=(stride, fs)))

    report = benchlib.score_grid(cells)
    print()
    print(benchlib.format_grid(report, "OFDM impulse benchmark"))

    if out:
        import json
        # The geometry rides along with the score: a score is only comparable
        # against another run of the SAME symbol length, chunk size and BAT,
        # and all three move when the band or the channel does.
        Path(out).write_text(json.dumps({
            "name": "ofdm_impulse_benchmark",
            "sample_rate": fs, "chan": chan, "snr_db": BENCH_SNR_DB,
            "chunk_bytes": BENCH_CHUNK, "seconds": seconds, "trials": trials,
            "amp_ratio": amp_ratio,
            "symbol_samples": stride, "cp_len": int(cfg["cp_len"]),
            "ecc_bytes": cfg.get("ecc_bytes"),
            "bat_summary": info["bat_summary"],
            "payload_bytes": len(payload),
            **report,
        }, indent=2))
        print(f"\n  result -> {out}")

    # Always exit 0: this command MEASURES, it does not assert.  A score is a
    # number to track, and there is no threshold here that is defensible
    # without a baseline to compare against -- gating on one would only teach
    # people to pass a flag to silence it.
    return 0


def cmd_simtest() -> int:
    """Two nodes, full DSP: initiator spk 48k / responder spk 44.1k, both mics
    48k, asymmetric channels, noise floors + startup clicks.  The handshake
    must converge and both learned BATs must equal what the peer measured."""
    ini = HandshakeSM("initiator", wait_probe_s=200, wait_msg_s=200)
    rsp = HandshakeSM("responder", wait_probe_s=200, wait_msg_s=200)
    spk = {"initiator": 48000, "responder": 44100}
    mic = {"initiator": 48000, "responder": 48000}
    chan = {"initiator": "good", "responder": "weak"}   # kind of MY tx channel
    measured = {}                                        # what each RX measured
    now = [0.0]
    rng = np.random.RandomState(99)

    leading_junk = _sim_leading_junk

    def deliver(sm_from, sm_to, side):
        acts, sm_from.actions = sm_from.actions, []
        for a in acts:
            if a[0] in ("done", "fail"):
                continue
            fs_tx = spk[side]
            if a[0] == "send_probe":
                audio = make_probe_audio(fs_tx)
            else:
                # mirror the real binding: pass our own tx table once we have
                # it, so the fast grouped payload mode is exercised end-to-end
                audio = make_msg_audio(fs_tx, a[1], a[2], a[3],
                                       my_tx_bat=sm_from.my_tx_bat)
            # Same levelling as the OTA binding, so the sim's per-message SNR
            # matches the real one -- peak normalisation here would quietly test
            # the exchange at 2.5-4 dB below what the probe graded.
            audio, _lvl = scale_hs_tx(audio, 0.3, fs_tx,
                                      is_probe=(a[0] == "send_probe"))
            air = _sim_channel(audio, fs_tx, chan[side], snr_db=14,
                               seed=int(rng.randint(1e6)))
            fs_rx = mic["responder" if side == "initiator" else "initiator"]
            capture = np.concatenate([
                leading_junk(fs_rx, int(rng.randint(1e6))),
                _resample(air, fs_tx, fs_rx),
                np.random.RandomState(int(rng.randint(1e6))).randn(int(0.3 * fs_rx)) * 1e-4,
            ])
            now[0] += len(audio) / fs_tx + 2.0
            exp = sm_to.expecting() or ""       # binding decode gate
            if a[0] == "send_probe":
                if "probe" not in exp:
                    continue
                got = try_decode_probe(capture, fs_rx)
                if got is None:
                    print(f"  !! probe decode failed ({side})"); continue
                rate, bat_bytes, info = got
                assert rate == fs_tx, f"rate detect wrong: {rate} != {fs_tx}"
                measured[side] = bat_bytes
                print(f"  {side} probe -> rate {rate}, {info['bat_summary']}")
                sm_to.heard_probe(bat_bytes, now[0])
            else:
                if "msg" not in exp:
                    continue
                got = try_decode_msg(capture, fs_rx)
                if got is None:
                    print(f"  !! msg decode failed ({side} {_MSG_NAMES.get(a[1])})")
                    continue
                rate, mtype, proof, body = got
                print(f"  {side} msg {_MSG_NAMES.get(mtype)} -> rate {rate}")
                sm_to.heard_msg(mtype, proof, body, now[0])

    ini.start(now[0]); rsp.start(now[0])
    for _ in range(30):
        if (ini.done or ini.failed) and (rsp.done or rsp.failed):
            break
        deliver(ini, rsp, "initiator")
        deliver(rsp, ini, "responder")
        now[0] += 3.0
        ini.tick(now[0]); rsp.tick(now[0])

    ok = (ini.done and rsp.done
          and ini.my_tx_bat == measured.get("initiator")
          and rsp.my_tx_bat == measured.get("responder"))
    print("\n--- initiator log ---"); print(ini.log())
    print("--- responder log ---"); print(rsp.log())

    # --- data phase: actually USE the tables that were just negotiated ---
    # Converging on a BAT proves the handshake works, not that the link does.
    # This runs real payload through the main modem's data path over the SAME
    # simulated channel the tables were measured on, in both directions, at
    # each transmitter's own rate.  A table that is subtly misaligned -- the
    # wrong carrier set, the wrong rate, an off-by-one in the wire unpack --
    # converges happily above and fails here.
    if ok:
        ok &= _sim_data_phase(ini, rsp, spk, mic, chan, rng, seconds=DATA_SIM_S)

    print("\nSIMTEST", "PASS" if ok else "FAIL")
    if ok:
        n_i = len(bytes_to_bat(ini.my_tx_bat, hs_cfg(48000)["n_data_carriers"]))
        print(f"  initiator learned its tx BAT ({n_i} carriers) in-band; "
              f"responder likewise -- both proofs verified")
    return 0 if ok else 1


def _bench_argv(argv) -> int:
    """`benchmark [--trials N] [--seconds S] [--chan good|weak] [--strict]`."""
    import argparse
    ap = argparse.ArgumentParser(prog="ofdm_handshake.py benchmark")
    ap.add_argument("--trials", type=int, default=2,
                    help="decodes per coverage level (default 2)")
    ap.add_argument("--seconds", type=float, default=BENCH_SECONDS,
                    help="payload airtime per trial (default 3)")
    ap.add_argument("--chan", default="good", choices=("good", "weak"),
                    help="simulated speaker/room response (default good)")
    ap.add_argument("--strict", action="store_true",
                    help="walk the whole coverage ladder instead of stopping "
                         "at the first failure, and report non-monotonicity")
    ap.add_argument("--out", help="write the full report as JSON here")
    ap.add_argument("--amp", type=float, default=BENCH_AMP_RATIO,
                    help="burst amplitude as a multiple of signal RMS "
                         f"(default {BENCH_AMP_RATIO}); sweep it if every cell "
                         "scores 0 or 100, which means the grid is out of range")
    ap.add_argument("--room", default=SIM_ROOM_DEFAULT, choices=sorted(SIM_ROOMS),
                    help="room impulse response (default none)")
    ap.add_argument("--ppm", type=float, default=SIM_PPM_DEFAULT,
                    help="sound-card sample-clock offset in ppm (default 0)")
    a = ap.parse_args(argv)
    globals()["SIM_ROOM_DEFAULT"] = a.room
    globals()["SIM_PPM_DEFAULT"]  = a.ppm
    return cmd_benchmark(seconds=a.seconds, trials=a.trials, chan=a.chan,
                         strict=a.strict, out=a.out, amp_ratio=a.amp)


def _sim_argv(argv) -> int:
    """`simtest [--room none|desk|room|reverb] [--ppm P]`.

    The room and the clock offset are set GLOBALLY rather than passed down,
    because they are properties of the simulated world, not of one call: every
    transmission in the run -- probe, handshake message, data frame -- has to
    see the same channel or the BAT is measured under conditions the data
    never experiences, which is precisely the bug the ISI ceiling exists for.
    """
    import argparse
    global SIM_ROOM_DEFAULT, SIM_PPM_DEFAULT
    ap = argparse.ArgumentParser(prog="ofdm_handshake.py simtest")
    ap.add_argument("--room", default=SIM_ROOM_DEFAULT,
                    choices=sorted(SIM_ROOMS),
                    help="room impulse response (default none = the old "
                         "delay-free channel)")
    ap.add_argument("--ppm", type=float, default=SIM_PPM_DEFAULT,
                    help="sound-card sample-clock offset in ppm (default 0)")
    a = ap.parse_args(argv)
    SIM_ROOM_DEFAULT = a.room
    SIM_PPM_DEFAULT  = a.ppm
    if a.room != "none" or a.ppm:
        print(f"[sim] room={a.room} clock_offset={a.ppm:g} ppm")
    return cmd_simtest()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "simtest":
        sys.exit(_sim_argv(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        sys.exit(_bench_argv(sys.argv[2:]))
    sys.exit(cmd_selftest())
