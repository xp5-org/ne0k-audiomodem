# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""CSS modem settings.  Plain data, no Qt: the UI edits a copy and the engine
reads it, so the same object works for the CLI and a future remote engine."""
from __future__ import annotations

import dataclasses
from pathlib import Path


class _Config:
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        known = {f.name: f for f in dataclasses.fields(cls)}
        out = cls()
        for k, v in d.items():
            if k in known:
                cur = getattr(out, k)
                if isinstance(cur, bool):
                    v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
                setattr(out, k, type(cur)(v))
        return out


def F(default, label: str, group: str, tip: str = "", lo=None, hi=None, choices=None, step=None):
    """A config field that also describes itself, so the settings dialog is generated from the
    dataclass and a new setting needs no UI code.  ``choices`` makes it a drop-down."""
    return dataclasses.field(default=default, metadata=dict(
        label=label, group=group, tip=tip, lo=lo, hi=hi, choices=choices, step=step))


def describe(cls) -> list[dict]:
    """Field descriptions in declaration order: name, default, type and the F() metadata."""
    out = []
    for f in dataclasses.fields(cls):
        out.append(dict(name=f.name, default=f.default, kind=type(f.default), **f.metadata))
    return out


@dataclasses.dataclass
class CssConfig(_Config):
    node_id: int = F(1, "Node id", "Link", "This end's address, 0..15.", 0, 15)
    peer_id: int = F(15, "Peer id", "Link", "Where packets are addressed; 15 = broadcast.", 0, 15)
    fs: int = F(48000, "Sample rate (Hz)", "Band and timing",
                "Must equal the audio device rate.", 8000, 192000, [8000, 16000, 22050, 44100, 48000, 96000])
    f_min: float = F(400.0, "Lowest chirp frequency (Hz)", "Band and timing",
                     "Chirp sweep start.  Use 400 for a phone line.", 50, 20000)
    f_max: float = F(6000.0, "Highest chirp frequency (Hz)", "Band and timing",
                     "Chirp sweep end.  Use 3300 for a phone line.", 100, 24000)
    symbol_dur: float = F(0.05, "Symbol duration (s)", "Band and timing",
                          "Longer = slower but more robust.  0.03 and below fail to decode.",
                          0.005, 2.0, step=0.01)
    symbol_gap_frac: float = F(0.05, "Symbol gap (fraction)", "Band and timing",
                               "Silence between chirps, as a fraction of the symbol.", 0.0, 0.5, step=0.01)
    amplitude: float = F(0.8, "TX amplitude", "Band and timing", "Peak level, 0..1.", 0.01, 1.0, step=0.05)
    chunk_size: int = F(8, "Payload bytes per frame", "Framing and FEC", "", 1, 255)
    short_frames: bool = F(True, "Short header-only frames", "Framing and FEC",
                           "Send ACKs and beacons (2-byte packets) as 2-byte frames, and listen for them: "
                           "15.5 s instead of 23 s at 0.15 s symbols.  Every station must have it on.")
    crc_bytes: int = F(2, "Frame check bytes", "Framing and FEC",
                       "4 = CRC-32, 2 = CRC-16.  CRC-16 is 16 chirps (2.4 s at 0.15 s symbols) shorter "
                       "per frame; the packet header still has to make sense on top.  Every station "
                       "must use the same value.", 2, 4, step=2)
    diff_encode: bool = F(True, "Differential encoding", "Framing and FEC",
                          "Survives a constant phase offset.")
    rs_t: int = F(0, "Reed-Solomon t", "Framing and FEC", "Correctable bytes per frame, 0 = off.", 0, 32)
    interleave_rows: int = F(0, "Interleave rows", "Framing and FEC",
                             "Block interleaver depth for burst errors, 0 = off.", 0, 64)
    conv_code: str = F("none", "Convolutional code", "Framing and FEC",
                       "k7r12 = rate 1/2, constraint 7, Viterbi decoded.", choices=["none", "k7r12"])
    carrier_db: float = F(-55.0, "Carrier threshold (dBFS)", "Receive",
                          "A burst starts when 50 ms RMS passes this.", -100, 0)
    hang_s: float = F(1.0, "Burst end after quiet (s)", "Receive", "", 0.1, 10)
    hear_own: bool = F(False, "Decode own transmissions", "Receive",
                       "For self-loop through a null sink.")
    backend: str = F("python", "DSP backend", "Engine",
                     "c = libcssenc (chirp, CRC, conv/Viterbi, interleave), libcssmf (matched filter) "
                     "and librs255, process-wide; fails if a library is missing.  python = numpy/reedsolo.",
                     choices=["python", "c"])
    decoder: str = F("pll_c", "Decoder", "Engine",
                     "pll_c = continuous RX on the C streaming PLL decoder (libcsspll, the port of "
                     "css_decoder.decode()) plus the running chirp tracker (own / peer / echo / phase "
                     "step / collision).  auto = css_decoder per burst (PLL at symbol_dur <= 0.25 s, "
                     "matched filter above).  mf = per-burst matched filter at any symbol length.",
                     choices=["pll_c", "auto", "mf"])


@dataclasses.dataclass
class OfdmConfig(_Config):
    """OFDM link.  Both ends must agree on the PHY group, ``window_s``, ``guard_s``,
    ``chunk_bytes`` and the band; the BAT exchange only settles the bit loading."""
    # settings saved before the carrier schedule got its name used the "melody_" keys
    _LEGACY_KEYS = {"melody_tx": "schedule_tx", "melody_rx": "schedule_rx",
                    "melody_note_carriers": "schedule_step_carriers", "melody_boost_db": "schedule_boost_db"}

    @classmethod
    def from_dict(cls, d: dict):
        return super().from_dict({cls._LEGACY_KEYS.get(k, k): v for k, v in d.items()})

    node_id: int = F(1, "Node id", "Link", "", 0, 255)
    peer_id: int = F(2, "Peer id", "Link", "", 0, 255)
    channel: str = F("radio", "Channel", "Link",
                     "radio = a keyed transmitter: every burst starts with a 2 s 1000 Hz key-up (PL) lead tone for the "
                     "PTT, squelch and the far end's receive gate.  line = SIP / dialup / fax-modem audio: no PTT, no "
                     "lead tone, the frame goes out as is.", choices=["radio", "line"])
    role: str = F("auto", "Role", "Link", "auto = the lower node id initiates.",
                  choices=["auto", "initiator", "responder"])
    mode: str = F("half", "Mode", "Link",
                  "half = timed windows both ways, oneway/stream = one sender, full = FDD, "
                  "arq = one-way with ACKs (mesh data plane), bcast = continuous broadcast carrier (late joiners lock "
                  "on any chunk).",
                  choices=["half", "oneway", "stream", "full", "arq", "bcast"])
    sender: str = F("initiator", "Sender (oneway/stream/arq)", "Link",
                    "Which role transmits data; the other listens.  both = oneway/stream in either "
                    "direction, a sender waits for the peer's burst to end (turn-taking, no ACK).",
                    choices=["initiator", "responder", "both"])
    fs: int = F(48000, "Sample rate (Hz)", "Band and level",
                "The handshake only runs at 48000 or 44100.", choices=[48000, 44100])
    freq_lo: float = F(400.0, "Band low (Hz)", "Band and level", "", 50, 20000)
    freq_hi: float = F(4000.0, "Band high (Hz)", "Band and level", "", 100, 24000)
    amplitude: float = F(0.3, "TX amplitude", "Band and level",
                         "Peak of the probe; later bursts are RMS-matched to it.", 0.01, 1.0, step=0.05)
    wait_probe_s: float = F(40.0, "Wait for probe (s)", "Handshake", "", 5, 600)
    wait_msg_s: float = F(40.0, "Wait for message (s)", "Handshake", "", 5, 600)
    hs_retries: int = F(2, "Handshake retries", "Handshake", "", 0, 20)
    hs_timeout_s: float = F(300.0, "Handshake timeout (s)", "Handshake", "", 20, 3600)
    poll_s: float = F(0.5, "Receive poll (s)", "Handshake", "", 0.1, 10)
    n_fft: int = F(1024, "FFT size", "OFDM PHY (both ends equal)",
                   "Carrier spacing = rate / FFT size.", choices=[512, 1024, 2048, 4096])
    cp_ratio: float = F(0.25, "Cyclic prefix ratio", "OFDM PHY (both ends equal)",
                        "The echo budget: must outlast the room's reflections.",
                        choices=[0.25, 0.375, 0.5])
    pilot_spacing: int = F(8, "Pilot spacing", "OFDM PHY (both ends equal)", "", 2, 32)
    probe_duration_s: float = F(5.0, "Probe duration (s)", "OFDM PHY (both ends equal)", "", 1, 30)
    probe_meas_syms: int = F(32, "Probe measurement symbols", "OFDM PHY (both ends equal)",
                             "Non-periodic symbols that make the probe's SNR honest. 0 = periodic only.",
                             0, 128)
    snr_margin_db: float = F(9.0, "SNR margin (dB)", "OFDM PHY (both ends equal)",
                             "Headroom over the modulation floor; covers fade migration.", 0, 30)
    max_bits_per_bin: int = F(2, "Max bits per carrier", "OFDM PHY (both ends equal)",
                              "1 BPSK, 2 QPSK, 4 16-QAM, 6 64-QAM.", choices=[1, 2, 4, 6])
    snr_estimator: str = F("mean", "SNR estimator", "OFDM PHY (both ends equal)", "",
                           choices=["mean", "percentile"])
    isi_derate: float = F(0.0, "ISI de-rating", "OFDM PHY (both ends equal)",
                          "1 folds the measured reverberation ceiling into the allocation.",
                          choices=[0.0, 1.0])
    window_s: float = F(5.0, "Window (s)", "Half-duplex windows",
                        "Fixed TX/RX window; the initiator owns the even ones.", 1, 120)
    guard_s: float = F(1.0, "Guard (s)", "Half-duplex windows",
                       "Dead time at both ends of a window (clock skew).", 0, 10)
    chunk_bytes: int = F(32, "Payload bytes per frame", "Half-duplex windows", "", 4, 255)
    retries: int = F(3, "Window retries", "Half-duplex windows",
                     "Resent without an ACK this many times before TX fails.", 0, 20)
    gap_s: float = F(1.0, "Frame gap (s)", "One-way and stream",
                     "Quiet between frames so the receiver sees separate bursts.", 0, 10)
    hang_s: float = F(0.6, "Burst end after quiet (s)", "One-way and stream", "", 0.1, 10)
    stream_poll_s: float = F(1.0, "Stream decode interval (s)", "One-way and stream", "", 0.5, 30)
    duplex: str = F("fdd", "Duplex mechanism", "Full duplex",
                    "How mode full carries both directions at once.  fdd = band split around a guard band, each end "
                    "filters its own half away: radio (a transmitter deafens its own receiver) and acoustic audio.  "
                    "ec = V.34 style: both ends use the WHOLE band and each subtracts an echo canceller's estimate "
                    "of its own signal, trained on the echo of its own BAT probe: SIP / dialup / fax-modem lines.  "
                    "Both ends must agree.", choices=["fdd", "ec"])
    ec_taps: int = F(256, "Echo canceller taps", "Full duplex",
                     "duplex ec: impulse response length the canceller fits after the bulk delay (256 = 5.3 ms at "
                     "48 kHz).  0 = no canceller (a 4-wire line that never returns our own audio).", 0, 4096)
    duplex_guard_hz: float = F(600.0, "Duplex guard band (Hz)", "Full duplex", "", 0, 5000)
    dl_frac: float = F(0.5, "Responder share of band", "Full duplex",
                       "fdd: the responder's share of the split (see the next setting).  ec: both directions use "
                       "the whole band, each at the rate its own BAT measured (0.5 = leave them be); below 0.5 the "
                       "responder may start a data frame in only 1 of every k chunks (k holding its share of the "
                       "payload capacity to this), above 0.5 the initiator.  Its ACKs still ride every chunk.", 0.1, 0.9, step=0.05)
    ratio: str = F("", "Data ratio initiator/responder", "Full duplex",
                   "The split of the link's data capacity as initiator/responder percent (in p2p the lower node id, "
                   "A, initiates), e.g. 50/50, 90/10 or 10/90.  Overrides the share above, and unlike it 50/50 is "
                   "enforced: the faster direction is held to an equal share.  Empty = use the share above.  Both "
                   "ends must agree.")
    ec_window: int = F(32, "ARQ window (frames)", "Full duplex",
                       "duplex ec: data frames in flight unacknowledged (LAPM's k).  The receiver keeps frames that "
                       "arrive past a gap and the sender resends only the frames whose own timer ran out.  1 = "
                       "stop-and-wait.", 1, 64)
    duplex_split: str = F("bits", "Split the share by", "Full duplex",
                          "bits = place the split so the responder's share of the CAPACITY (bits per symbol from the "
                          "exchanged BATs, both ends hold both) is the share above: a G.711 line carries little above "
                          "~3.4 kHz, so a 50/50 split in Hz leaves the high (responder) half short.  hz = split the "
                          "bandwidth itself.  Both ends must agree, like the share.", choices=["bits", "hz"])
    ack_wait_s: float = F(8.0, "Resend after (s)", "Full duplex / ARQ",
                          "Resend a frame when no ACK arrived within this.", 1, 120)
    schedule_tx: str = F("", "Score for frames we send", "Carrier schedule",
                       "Carrier score JSON (carrier_schedule_score.py, made from a MIDI file).  Each new DATA frame this end "
                       "sends plays the next score step: its bit table is cut down to the carriers of that step's "
                       "notes (+ overtones).  Frame and symbol sizes are unchanged; ACKs keep the full table.  "
                       "arq mode (mesh P2P).  Empty = off.")
    schedule_rx: str = F("", "Score for frames we receive", "Carrier schedule",
                       "The peer's schedule_tx score: the receiver knows the sequence in advance, matches each burst's "
                       "spectrum against the next expected steps and decodes with that step's table.  May differ "
                       "per direction (A->B one tune, B->A another).  Empty = off.")
    schedule_step_carriers: int = F(4, "Carriers per step", "Carrier schedule",
                                  "Every note carries this many of the link's typical carrier's bits per symbol "
                                  "(the note, its octave, the octave below, 3rd / 4th harmonic, the fifth ... until "
                                  "they add up), so all notes last the same time whatever their pitch.  Both ends "
                                  "must agree.", 1, 16)
    tx_window_chunks: int = F(0, "TX window reservation (chunks)", "P2P window",
                              "arq: reserve up to this many chunks with one control frame, then stream them back to "
                              "back -- no ACKs, no pauses -- and take one status at the end (all in, or which chunks "
                              "to stream again, each in its own slot).  Both ends must agree.  0 = stop-and-wait "
                              "frames of up to 255 B.", 0, 100)
    schedule_boost_db: float = F(12.0, "Step carrier boost (dB)", "Carrier schedule",
                               "Level of the note carriers over the pilots in a scheduled frame.  Pilots, CE and sync stay at "
                               "unit power; with only 2-4 note carriers against the pilots, 0 dB leaves the pilot chord "
                               "louder than the tune.  CE / sync get the same boost there, so the receiver's channel "
                               "estimate absorbs it: no receive setting, any QAM order.",
                               0, 20)
    carrier_db: float = F(-55.0, "Carrier threshold (dBFS)", "Receive", "", -100, 0)
    hear_own: bool = F(False, "Decode own transmissions", "Receive", "")
    backend: str = F("python", "DSP backend", "Engine",
                     "c = libofdmacq + librs255 (process-wide). "
                     "cdx = c, plus the full-duplex data plane (mode full) run by the C engine.",
                     choices=["python", "c", "cdx"])

    def responder_share(self) -> tuple[float, bool]:
        """(responder share of the data capacity, strict).  strict = it came from ``ratio`` and 0.5 is enforced
        too; otherwise it is ``dl_frac`` and 0.5 means "leave each direction at its own rate"."""
        r = str(self.ratio or "").strip()
        if not r:
            return float(self.dl_frac), False
        try:
            a, b = (float(x) for x in r.replace(":", "/").split("/"))
        except ValueError:
            raise ValueError(f"ratio {r!r}: want initiator/responder, e.g. 90/10") from None
        if a <= 0 or b <= 0:
            raise ValueError(f"ratio {r!r}: both directions need a share above 0")
        return b / (a + b), True


@dataclasses.dataclass
class GeneralConfig(_Config):
    # settings saved before the mixed mode got its own name called it "ofdm"
    _MODEM_ALIASES = {"ofdm": "css_promote_ofdm"}

    @classmethod
    def from_dict(cls, d: dict):
        if d.get("modem") in cls._MODEM_ALIASES:
            d = {**d, "modem": cls._MODEM_ALIASES[d["modem"]]}
        return super().from_dict(d)

    modem: str = F("css", "Modem", "Mode", "css = CSS only: robust, slow.  css_promote_ofdm = mixed: a CSS "
                   "control plane, beacons and CSS chirps on the air, with an OFDM link brought up per transfer and "
                   "CSS data as the fallback if OFDM fails.  ofdm_only = OFDM only: no CSS at all, nothing chirps; "
                   "beacons are OFDM frames.",
                   choices=["css", "css_promote_ofdm", "ofdm_only"])
    topology: str = F("p2p", "Topology", "Mode", "p2p = one pair of stations; mesh = N stations that discover "
                      "each other with beacons.  hub = hub & spoke: one leader (hub_id) polls, every other station "
                      "answers it only, pages go station <-> hub (the hub may send one to every station at once).  "
                      "hub_relay = hub, plus station -> station pages the hub stores and forwards.  Hub modes are "
                      "ofdm_only; no failover: when the hub is gone the network is down.",
                      choices=["p2p", "mesh", "hub", "hub_relay"])
    hub_id: int = F(1, "Hub id", "Hub & spoke", "Hub topologies: the leader's node id.  A client knows this and "
                    "the number of stations, nothing else.", 0, 14)
    hub_nodes: int = F(3, "Stations", "Hub & spoke", "Hub topologies: stations in the network, the hub included.  "
                       "The hub closes its join window once this many have joined.", 2, 15)
    hub_turn_s: float = F(60.0, "Polled turn (s)", "Hub & spoke", "Longest turn the hub grants one station.",
                          15.0, 600.0)
    hub_cycle_s: float = F(15.0, "Least cycle (s)", "Hub & spoke", "An idle network polls every station at most "
                           "this often.", 2.0, 600.0)
    hub_page_timeout_s: float = F(300.0, "Page timeout (s)", "Hub & spoke", "A page not delivered within this "
                                  "fails (TXFAIL on the serial port).  It waits for this station's next turn and, "
                                  "relayed, for the hub's delivery report.", 30.0, 3600.0)
    node_id: int = F(1, "Node id", "Addressing", "This station's address, 0..15.", 0, 15)
    peer_id: int = F(15, "Peer id", "Addressing", "p2p only: the other station.  In a mesh the peer is "
                     "not configured: it is whoever answers.", 0, 15)
    mesh_dst: int = F(0, "Pinned mesh peer", "Addressing", "Mesh only: the virtual serial port always talks "
                      "to this station instead of whoever is heard strongest; 0 = auto (best neighbour).", 0, 15)
    beacon: bool = F(True, "Send beacons", "Mesh beacons", "Mesh only: announce this node so the others can find it -- a CSS "
                     "beacon, or an OFDM beacon frame under ofdm_only.")
    beacon_min_s: float = F(20.0, "Beacon interval min (s)", "Mesh beacons",
                            "The beacon interval follows the measured clock drift of the other nodes "
                            "(faster drift, more often), kept between min and max.", 1.0, 3600.0)
    beacon_max_s: float = F(300.0, "Beacon interval max (s)", "Mesh beacons", "", 1.0, 3600.0)
    first_beacon_s: float = F(20.0, "Listen before first beacon (s)", "Mesh beacons",
                              "Stay silent and listen for up to this long (random, plus 1 s per node id) before "
                              "the first beacon, so nodes started together do not collide.", 0.0, 600.0)
    ignore_busy: bool = F(False, "Transmit without waiting for silence", "Transmit",
                          "Skip the checks that wait for a quiet channel before keying up: mesh carrier sense, OFDM "
                          "handshake turns and data frames, the announce round.  For a microphone and speaker, where "
                          "room noise always reads as someone transmitting and the modem would never send.  Protocol "
                          "holds stay: blockout windows, our own handshake in progress.  With it on, two stations can "
                          "key over each other.")
    slot_s: float = F(2.5, "Backoff slot (s)", "Mesh channel access",
                      "CSMA/CA: before sending, wait a random whole number of these slots, plus 0.1 s per "
                      "node id.  Must be longer than it takes to hear another node key up (path delay + "
                      "carrier detect), or two nodes in neighbouring slots still collide.", 0.1, 30.0)
    cw_slots: int = F(8, "Contention window (slots)", "Mesh channel access",
                      "Initial number of backoff slots to pick from.", 1, 64)
    cw_max_slots: int = F(16, "Max contention window (slots)", "Mesh channel access",
                          "Each unanswered packet doubles the window, up to this; an ACK resets it.", 1, 256)
    ack_stagger_s: float = F(0.5, "ACK stagger per node id (s)", "Mesh channel access",
                             "A node holds its ACK by id x this, so two peers never answer at the same instant.",
                             0.0, 5.0)
    ofdm_detect: bool = F(True, "Passive OFDM detector", "Mesh channel access",
                          "OFDM mesh: after giving up on OFDM, listen for OFDM on the air and ask again over CSS.")
    airtime_windows: bool = F(False, "Per-link handshake windows", "Mesh channel access",
                              "Give each link its own wall-clock window for handshakes so two never overlap.  "
                              "Needs both ends' clocks within a second or so.")
    auto_link: bool = F(True, "Train the link at power-on", "Mesh channel access",
                        "The modem handshakes (and BAT-trains) with its peer as soon as it is powered on, without "
                        "waiting for data to send.  The initiator opens the link, the responder listens for it; "
                        "Role is under OFDM settings.  Off = handshake only when something is sent.")
    announce_round: bool = F(False, "OFDM announce round", "Mesh channel access",
                             "ofdm_only: nodes announce in id order (listen, then speak after the highest lower id), "
                             "late/unheard nodes retry in a catch-up slot, then BAT training runs one initiator at a "
                             "time. Replaces the blind simultaneous handshakes.")
    announce_bat_turn_s: float = F(60.0, "BAT turn timeout (s)", "Mesh channel access", "", 10.0, 600.0)
    announce_bat_mode: str = F("pairwise", "BAT training", "Mesh channel access",
                               "ofdm_only + announce round.  pairwise: one initiator at a time, each pair probes both "
                               "ways (the robust acoustic setting).  broadcast: each node probes once while everyone "
                               "measures, then one report each and a coordinator confirm -- for a clean shared line "
                               "(SIP); pairs it leaves out fall back to pairwise.",
                               choices=["pairwise", "broadcast"])
    announce_cycle_s: float = F(180.0, "Announce cycle (s)", "Mesh channel access",
                                "Interval between catch-up windows / retry passes.", 30.0, 3600.0)
    hs_window_s: float = F(150.0, "Handshake window (s)", "Mesh channel access", "", 10.0, 600.0)
    hs_slots: int = F(4, "Window slots", "Mesh channel access", "Links are spread over this many windows.", 1, 16)
    hs_guard_s: float = F(45.0, "Handshake guard (s)", "Mesh channel access",
                          "A handshake may only START in the first (window - guard) seconds of its window, so it can "
                          "finish before the next link's window opens.", 0.0, 300.0)
    blockout_role: str = F("off", "Blockout role", "Blockout",
                           "Mesh only. 'announce': at blockout_at_s after this node's mesh link starts, "
                           "broadcast an exclusive window with blockout_peer and send blockout_payload "
                           "over it (MeshNode.send_blockout). 'reply': wait to be named as the other "
                           "participant, then send blockout_payload back once (the two-way case). "
                           "'off': do nothing of our own -- a node named in someone else's announce still "
                           "mutes for the window regardless of this setting.",
                           choices=["off", "announce", "reply"])
    blockout_peer: int = F(0, "Blockout peer", "Blockout",
                           "The OTHER participant's node id (announce and reply both need this).", 0, 15)
    blockout_at_s: float = F(60.0, "Announce after (s)", "Blockout",
                             "role=announce only: seconds after this node's mesh link starts -- for "
                             "manual use (a human operator arming the feature). Ignored when "
                             "blockout_go_file is set.", 0.0, 3600.0)
    blockout_go_file: str = F("", "Announce go-file", "Blockout",
                              "role=announce only: wait for this file to exist, then announce "
                              "immediately -- no guessed delay. For a distributed test group that "
                              "already knows (via its own barrier) when every member is actually up "
                              "and listening; that synchronization belongs in the test, not as a "
                              "timer here. Empty (manual use) falls back to blockout_at_s.")
    blockout_duration_s: float = F(120.0, "Window duration (s)", "Blockout",
                                   "role=announce only: how long every other node mutes for.", 1.0, 6553.0)
    blockout_payload: str = F("blockout-demo-payload", "Payload to send", "Blockout",
                              "Text sent to blockout_peer inside the window (announce and reply both).")


@dataclasses.dataclass
class OutputConfig(_Config):
    """Where the modem's audio / IQ goes.  Only Audio has a connector so far; the GNURadio and
    libsdr fields are stored for the connectors still to come."""
    io_type: str = F("audio", "I/O type", "Output", "Audio uses the sound devices on the front page.",
                     choices=["audio", "gnuradio", "libsdr"])
    location: str = F("local", "Location", "GNURadio / libsdr", "local = on this machine, remote = over the network.",
                      choices=["local", "remote"])
    host: str = F("127.0.0.1", "Remote IP", "GNURadio / libsdr", "Address of the remote GNURadio / libsdr end.")
    port: int = F(5555, "Port", "GNURadio / libsdr", "", 1, 65535)


@dataclasses.dataclass
class DiagConfig(_Config):
    """Troubleshooting capture, for sending back to the developers."""
    debug_log: bool = F(False, "Save debug log", "Capture",
                        "Timestamped engine log with versions, settings and backends.")
    wav_rx: bool = F(False, "Record received audio (WAV)", "Capture",
                     "Everything the input device delivers, while audio is open.")
    wav_tx: bool = F(False, "Record transmitted audio (WAV)", "Capture",
                     "What we played, time-aligned to the RX file.")
    out_dir: str = F(str(Path.home() / "modemgui-diagnostics"), "Folder", "Capture", "")
    detail: str = F("detailed", "Event log detail", "Logging",
                    "basic = one short line per packet; detailed = SNR, frame and RS error counts too.",
                    choices=["basic", "detailed"])
    log_serial: bool = F(True, "Log serial port activity", "Logging",
                         "Put the virtual serial port's tx/rx lines in the log.")


@dataclasses.dataclass
class HamConfig(_Config):
    """Amateur-radio station identification.  Off by default; the operator is responsible for
    meeting the ID rules of their licence (see README)."""
    ident_mode: str = F("ham", "Ident mode", "Station",
                        "ham = alphanumeric callsign and CW announce timers.  numeric = regular mode: "
                        "only the 0-15 node / peer ids (CSS and OFDM pages) identify stations; "
                        "nothing is announced.", choices=["ham", "numeric"])
    enabled: bool = F(False, "Identify this station", "Station",
                      "Transmit the callsign as set below.")
    callsign: str = F("", "Callsign", "Station", "Sent as 'DE <CALLSIGN>'.")
    id_interval_s: float = F(600.0, "Timer 1: ID at least every (s)", "Timers",
                             "Checked on every outgoing frame; the first frame of a session counts. "
                             "600 = every 10 minutes.", 10, 7200, step=30)
    id_end_delay_s: float = F(5.0, "Timer 2: ID after the channel is quiet for (s)", "Timers",
                              "End-of-transmission ID, sent on its own once the last frame is this "
                              "old and an ID is owed.  0 = off.", 0, 600, step=1)
    id_mode: str = F("cw", "ID mode", "Mode",
                     "cw = Morse beeps after the frame.  ascii = binary-coded ASCII (AFSK) in front "
                     "of the framing.  both = ascii in front, cw after.",
                     choices=["cw", "ascii", "both"])
    id_amplitude: float = F(0.3, "ID level", "Mode", "Peak amplitude of the ID audio, 0..1.",
                            0.01, 1.0, step=0.05)
    cw_wpm: float = F(15.0, "CW speed (wpm)", "CW", "", 5, 40, step=1)
    cw_tone_hz: float = F(800.0, "CW tone (Hz)", "CW", "", 300, 3000, step=50)
    cw_gap_s: float = F(1.5, "Gap before CW (s)", "CW",
                        "Silence between the frame and the beeps; keep it longer than the receive "
                        "'burst end after quiet' so the peer finishes the data burst first.", 0, 10, step=0.1)
    ascii_baud: float = F(300.0, "ASCII baud", "Binary ASCII", "AFSK 2200/1200 Hz, 8N1.",
                          choices=[110.0, 300.0, 600.0])
    ascii_gap_s: float = F(1.25, "Gap after ASCII (s)", "Binary ASCII",
                           "Silence between the ID and the frame.  Keep it longer than the receive "
                           "'burst end after quiet' so the peer takes the ID as a burst of its own.",
                           0, 5, step=0.05)


@dataclasses.dataclass
class SerialConfig(_Config):
    """Virtual serial port: a pty with RTS/CTS on a control socket, over the active modem
    (General > Active modem).  Node/peer ids come from the CSS / OFDM pages."""
    link_path: str = F(str(Path.home() / "modem0"), "Port path", "Port",
                       "Symlink to the pty the host opens as a serial port; the RTS/CTS control "
                       "socket is this path + '.ctl'.")
    autostart: bool = F(False, "Start with audio", "Port",
                        "Open the port whenever audio is opened.")
    buffer: int = F(32, "TX buffer (bytes)", "Handshake",
                    "Bytes accepted per send; the modem raises CTS again after each.  Up to 250 on a frame-per-"
                    "send link; OFDM oneway/stream and a full-duplex p2p link take up to 1 MiB.", 1, 1 << 20)
    mode: str = F("rts", "Send trigger", "Handshake",
                  "rts = send on the RTS rising edge.  auto = send when the buffer is full or idle.  fifo = a UART "
                  "with a FIFO: CTS stays up while there is room, the next buffer follows at once (OFDM "
                  "oneway/stream, full-duplex p2p: keeps the link's window full).",
                  choices=["rts", "auto", "fifo"])
    idle_s: float = F(2.0, "Auto idle (s)", "Handshake", "auto mode: send after this long without input.",
                      0.1, 60, step=0.5)
    ack: bool = F(True, "Peer ACKs each buffer", "Reliability",
                  "Retransmit the buffer when no ACK arrives (CSS).")
    retries: int = F(3, "Retries", "Reliability", "Retransmissions before TXFAIL.", 0, 20)
    rx_host: int = F(0, "RX host filter", "Port", "Mesh only: only hand the host bytes received from this "
                     "station id; 0 = accept from any neighbour (CSS is mostly multicast/broadcast).", 0, 15)
    page_addr: str = F("off", "Page address", "Port",
                       "Hub topologies: one buffer is one page (load it, raise RTS, CTS stays low until it is "
                       "delivered or failed -- nothing queues behind it).  ascii: a page may start with '@<id>:' "
                       "(id one hex digit 0-E, or '*' = every station) naming where it goes, and a received page "
                       "starts with '@<origin>:'.  off: pages go to the hub (from the hub: to every station), "
                       "received pages arrive bare.", choices=["off", "ascii"])


@dataclasses.dataclass
class UdpConfig(_Config):
    """UDP data port: the alternative to the virtual serial port.  A sender node takes the UDP datagrams an
    encoder (ffmpeg) sends to ``listen`` and airs them on a continuous OFDM broadcast carrier (mode bcast); a
    receiver node hands what it decodes to ``dest`` (VLC, ffmpeg).  Use this or the serial port, not both."""
    autostart: bool = F(False, "Start with audio", "Port", "Open the UDP port whenever audio is opened.")
    role: str = F("send", "Role", "Port", "send = datagrams in, OFDM out.  receive = OFDM in, datagrams out.",
                  choices=["send", "receive"])
    listen_host: str = F("127.0.0.1", "Listen address", "Send", "Where the encoder sends its UDP datagrams.")
    listen_port: int = F(5000, "Listen port", "Send", "", 1, 65535)
    dest_host: str = F("127.0.0.1", "Destination address", "Receive", "Where decoded datagrams go (the player).")
    dest_port: int = F(5001, "Destination port", "Receive", "", 1, 65535)
    chunk_bytes: int = F(256, "Bytes per chunk", "Send",
                         "OFDM chunk size (2 of them are the length prefix); larger = higher net rate, but a lost chunk "
                         "costs more.  Sender and receivers must agree.", 16, 1024)
    bat_file: str = F("", "Saved BAT (JSON)", "Port",
                      "Engine config + the two BAT tables from a saved handshake.  The link comes up from it "
                      "with no handshake on the air, so a listener can join at any time.")
    sender_id: int = F(1, "Sender node id", "Receive", "", 0, 15)



@dataclasses.dataclass
class CssBcastConfig(_Config):
    """CSS broadcast port: a one-way text stream with a time code, on the virtual serial port (engine/css_bcast.py).
    The sender airs what the host writes to its tty as broadcast DATA packets, with a TIME packet every
    ``time_every`` packets; every listener writes the text to its own tty and keeps the broadcast's clock, which
    the host reads with ``TIME`` on the control socket.  No ACKs, no handshake: a listener can join at any time.
    Uses the CSS page's band, symbol and chunk settings.  Use this, the serial port or the UDP port, not two."""
    autostart: bool = F(False, "Start with audio", "Port", "Open the broadcast port whenever audio is opened.")
    role: str = F("receive", "Role", "Port", "send = tty in, CSS broadcast out.  receive = CSS in, tty out + clock.",
                  choices=["send", "receive"])
    link_path: str = F(str(Path.home() / "bcast0"), "Port path", "Port",
                       "Symlink to the pty; the control socket (TIME, STATUS) is this path + '.ctl'.")
    sender_id: int = F(1, "Sender node id", "Receive", "Only packets from this node are taken.", 0, 15)
    text_bytes: int = F(5, "Text bytes per packet", "Send",
                        "Each packet is the 2-byte header + a length byte + this much text, cut into CSS chunks.  Longer = less "
                        "header and preamble overhead, but a lost frame loses the whole packet.", 1, 64)
    time_every: int = F(3, "TIME every N packets", "Send", "1 = only TIME packets.", 1, 64)
    gap_s: float = F(1.5, "Gap between packets (s)", "Send",
                     "Silence after each packet; keep it above the CSS page's 'burst end after quiet' so the "
                     "listeners' decoders re-arm for the next preamble.", 0.0, 30.0, step=0.1)
    tx_latency_s: float = F(0.0, "TX latency (s)", "Send",
                            "Played this much earlier so a TIME packet's last chirp leaves the sink on the second.", 0.0, 5.0,
                            step=0.05)

PRESETS = {
    "Fast (POC)":   CssConfig(),
    "Robust":       CssConfig(symbol_dur=0.25, chunk_size=2),
    "Over the air": CssConfig(symbol_dur=0.5, chunk_size=2),
}

OFDM_PRESETS = {
    "Default":      {},
    "Cable / loopback": dict(snr_margin_db=6.0, max_bits_per_bin=6, isi_derate=0.0),
    "Over the air": dict(snr_margin_db=9.0, max_bits_per_bin=2, cp_ratio=0.375, isi_derate=1.0),
    "Phone line":   dict(freq_lo=400.0, freq_hi=3300.0, snr_margin_db=9.0, max_bits_per_bin=2),
}


@dataclasses.dataclass
class IqConfig(_Config):
    """Complex-baseband OFDM on an I/Q device (``iq:`` connector, GNU Radio medium, an SDR).

    The ``iq`` profile: 200 ksps complex, the occupied band +-50 kHz around the carrier (a 100 kHz
    channel with guard), n_fft 256 -> 781.25 Hz carrier spacing and a 64-sample (320 us) cyclic
    prefix.  The taper is kept at CP/4 (0.08 ms = 16 samples): ofdm_cxdem windows at taper + CP/2
    and the config clamps the taper to CP/2, which would leave zero ISI margin (COMPLEX_BB.md).
    Both ends must agree on every field in "PHY"; the probe only settles the bit loading."""
    node_id: int = F(1, "Node id", "Link", "", 0, 255)
    peer_id: int = F(2, "Peer id", "Link", "", 0, 255)
    fs: int = F(200000, "Sample rate (complex sps)", "PHY",
                "Must equal the I/Q device / GNU Radio medium rate.", 50000, 2000000,
                [200000, 250000, 400000, 521000, 1000000])
    band_lo: float = F(-50000.0, "Band low (Hz, baseband)", "PHY", "Signed: below the carrier.", -1e6, 1e6)
    band_hi: float = F(50000.0, "Band high (Hz, baseband)", "PHY", "Signed: above the carrier.", -1e6, 1e6)
    n_fft: int = F(256, "FFT size", "PHY", "", choices=[64, 128, 256, 512, 1024, 2048])
    cp_ratio: float = F(0.25, "Cyclic prefix (fraction of n_fft)", "PHY", "", 0.0625, 0.5, step=0.0625)
    taper_ms: float = F(0.08, "Symbol taper (ms)", "PHY",
                        "Keep <= CP/4: the demodulator windows at taper + CP/2.", 0.0, 10.0, step=0.01)
    pilot_spacing: int = F(8, "Pilot spacing (carriers)", "PHY", "", 2, 32)
    dc_null: bool = F(True, "Leave the DC carrier empty", "PHY",
                      "An SDR's LO leak and DC offset sit on bin 0.")
    sync_period: int = F(32, "SYNC symbol every N data symbols", "PHY", "", 1, 256)
    ecc_bytes: int = F(16, "Reed-Solomon parity bytes per chunk", "PHY", "", 0, 64)
    chunk_bytes: int = F(2040, "Payload bytes per frame", "PHY",
                         "Sequence number and RS parity go on top.  Parity is sized so one lost OFDM symbol is "
                         "correctable (~176 B at 64-QAM whatever the chunk), so bigger frames amortise it: "
                         "141 / 282 / 339 / 357 kbit/s at 256 / 1020 / 2040 / 4080 B (64-QAM, gapless, 200 k).  "
                         "A short message is NUL-padded to a whole frame.", 32, 4080)
    ctl_chunk_bytes: int = F(512, "Payload bytes per control frame", "PHY",
                             "Control messages (PING, BAT, READY, POLL, STAT) at the base table go in frames of "
                             "this size instead of chunk_bytes: a 2040 B QPSK frame lasts 0.24 s on iq-hf, long "
                             "enough for a 0.5 Hz two-ray fade to move its notches mid-frame (hf-moderate lost 6 "
                             "of 8 POLLs).  0 = chunk_bytes (the old frames).  Both ends must agree.", 0, 4080)
    ctl_repeat_s: float = F(0.0, "Send each POLL / STAT twice, this far apart (s)", "PHY",
                            "Time diversity for the ARQ control frames on a fading channel: the copy goes this "
                            "long after the first (same seq; the receiver drops whichever arrives second).  On "
                            "hf-moderate (0.5 Hz fade) one 512 B frame got through ~65 % of the time, either of "
                            "two 0.6 s apart ~90 %; 0.3 s apart the fades are still correlated.  0 = once.", 0.0, 5.0)
    seq_bytes: int = F(2, "Sequence number bytes", "PHY", "", 1, 4)
    frame_gap_ms: float = F(20.0, "Gap between frames (ms)", "PHY", "", 0.0, 1000.0)
    gapless: bool = F(True, "Frames back to back within a burst", "PHY",
                      "One frame gap before and after a burst, none between its frames.  Off: a gap after "
                      "every frame (20 ms default: 256 B frames 59 vs 141 kbit/s at 64-QAM).")
    max_cfo_bins: int = F(16, "CFO search range (carriers)", "Receive",
                          "Integer CFO searched, +- this many carrier spacings (781 Hz each at 200 k / 256).", 0, 64)
    sc_threshold: float = F(0.25, "SC metric threshold", "Receive", "", 0.05, 0.95, step=0.05)
    sc_ncc_min: float = F(0.10, "SC matched-filter minimum", "Receive", "", 0.0, 0.95, step=0.05)
    fec: str = F("2/3", "Inner FEC", "PHY",
                 "K=7 convolutional code (133, 171) with soft-decision Viterbi between the RS and the mapper, "
                 "bit-interleaved over the frame.  Buys ~4-5 dB (the bit loading spends 4 / 3 / 2.5 dB of the "
                 "margin at 1/2 / 2/3 / 3/4); costs the rate in air time.  none = the RS-only frame of steps 4-10.",
                 choices=["none", "1/2", "2/3", "3/4"])
    window: str = F("mid", "FFT window position", "Receive",
                    "mid: the middle of the ISI-free stretch (echo up to cp/2 - taper, timing slack both ways).  "
                    "late: cp_len - 2 samples (echo up to cp - 2*taper - 2, ~2x as long; a late sync costs ISI).",
                    choices=["mid", "late"])
    track: bool = F(True, "Track between SYNCs", "Receive",
                    "Decision-directed: the phase falls back on the data when the pilots disagree (dead or "
                    "noisy pilots) and the channel estimate follows a fade between SYNC symbols.")
    iterate: int = F(0, "Decoder-aided re-estimation passes", "Receive",
                     "Prototype (2026-11-15, off by default): a frame whose RS decode fails is re-encoded from the "
                     "Viterbi output (and every RS codeword that did decode), H is refit per carrier from the "
                     "neighbouring symbols' re-encoded cells, and the frame is decoded again, up to this many "
                     "times.  Needs an inner code; frames that decode first time are untouched.", 0, 8)
    iterate_win: int = F(2, "Re-estimation window (+- symbols)", "Receive",
                         "Per-carrier time window of the refit: short follows a fast fade, long averages noise.",
                         1, 16)
    erasures: bool = F(True, "Erasure decoding", "Receive",
                       "A frame the errors-only RS decode cannot correct is retried with its least reliable bytes "
                       "(decision distance, dead carriers by |H|) erased: up to twice the correction on a notched "
                       "or noisy carrier.  Off = errors-only, as before 2026-11-14.")
    default_bits: int = F(2, "Bits per carrier before a probe", "Bit loading",
                          "Every data carrier, until a probe has measured the channel.", choices=[1, 2, 4, 6])
    snr_margin_db: float = F(7.0, "SNR margin (dB)", "Bit loading",
                             "Subtracted before ofdm_allocate (floors 4/7/14/21 dB for BPSK..64-QAM are "
                             "raw).  7 dB: 30/30 frames just above every 16/64-QAM step (6 dB loses "
                             "frames at the QPSK edge, 3 dB loses up to half at each step).", 0.0, 20.0)
    max_bits: int = F(6, "Most bits per carrier", "Bit loading", "", choices=[1, 2, 4, 6])
    probe_syms: int = F(64, "Probe periodic symbols", "Bit loading", "", 8, 1024)
    meas_syms: int = F(32, "Probe measurement symbols", "Bit loading", "", 0, 1024)
    max_rtd_s: float = F(15.0, "Longest round trip allowed for (s)", "Link",
                         "Before the probe exchange has measured the round-trip delay, probe retries back off "
                         "(3, 6, 12 s ...) up to this, and the exchange's time budget is sized from it.  After "
                         "the first answer every retry / wait is sized from the MEASURED round trip.  15 s covers "
                         "speaker+mic (~0.3 s), G.711 paths (1-2 s) and a ham satellite hop (5+ s one way).",
                         0.5, 120.0, step=0.5)
    arq_rounds: int = F(8, "ARQ rounds per message", "Link",
                        "send_reliable: after the data and a POLL, the receiver answers with a bitmap of the frames "
                        "it holds and the missing ones are sent again, up to this many rounds (POLL-only rounds "
                        "when no answer comes count too).  0 = fire and forget.", 0, 64)
    reprobe_s: float = F(60.0, "Re-probe interval (s)", "Link",
                         "ARQ: the receiver asks for a fresh probe (STAT flag) this often, and at once when a table "
                         "update left carriers at 0 bits, then sends the probe's table: carriers come back and bits go "
                         "up when the channel recovers.  0 = never.", 0.0, 3600.0, step=5.0)
    duplex: str = F("full", "Duplex", "Link",
                    "full: keep receiving while we transmit (a mix-minus medium, FDD, or an SDR with "
                    "cancellation).  half: RX blocks captured during our own TX are zeroed (a radio that hears "
                    "itself).", choices=["full", "half"])
    excise: bool = F(True, "Notch narrowband interferers", "Receive",
                     "Stationary tones standing 12 dB over their spectral neighbourhood are notched (one carrier "
                     "wide) before the sync and the probe; a constant 2.6 ms delay either way.")
    lbt: bool = F(False, "Listen before talk", "Link",
                  "Before a burst, wait (up to lbt_max_s) until the last 20 ms of RX are within 6 dB of the noise "
                  "floor, plus a random 0-50 ms backoff.  For shared channels; off on a point-to-point link.")
    lbt_max_s: float = F(2.0, "LBT longest wait (s)", "Link", "", 0.0, 30.0, step=0.5)
    papr_clip_db: float = F(0.0, "PAPR clip (dB over RMS)", "Level",
                            "Clip each burst at this many dB over its RMS and filter the clipping noise back into "
                            "the occupied band (2 passes).  ~8 dB buys a few dB of PA headroom; 0 = off.",
                            0.0, 15.0, step=0.5)
    callsign: str = F("", "Station ID callsign", "Ident",
                      "Sent as CW (on/off keyed carrier at ident_hz) before the first burst and every ident_s.  "
                      "Required on amateur bands (e.g. FCC 97.119: every 10 minutes and at the end).")
    ident_s: float = F(600.0, "Station ID interval (s)", "Ident", "", 30.0, 3600.0, step=30.0)
    ident_wpm: float = F(20.0, "Station ID speed (wpm)", "Ident", "", 5.0, 40.0)
    ident_hz: float = F(25000.0, "Station ID offset (Hz)", "Ident", "Baseband offset of the CW carrier.",
                        -100000.0, 100000.0)
    tx_rms: float = F(0.2, "TX RMS level (full scale 1.0)", "Level",
                      "Complex RMS of every burst; OFDM peaks run ~10 dB above it, so 0.2 stays clear of "
                      "an sc16 / ADC full scale.", 0.001, 0.7, step=0.01)
    block: int = F(4000, "I/Q block (samples)", "Engine", "20 ms at 200 ksps.", 256, 65536)

# Named I/Q profiles: IqConfig field overrides.  "iq" is the defaults themselves (200 ksps, +-50 kHz).
IQ_PROFILES = {
    "iq": {},
    # Multipath: CP 0.5 x 256 = 640 us and the late window: echoes to ~470 us (impair "echo": 50 + 200 us).
    "iq-echo": {"cp_ratio": 0.5, "window": "late"},
    # HF-like long echoes: 512-point FFT (390 Hz carriers), CP 0.5 = 1.28 ms, late window: echoes to ~1.1 ms
    # (impair "echo-isi" / "hf-moderate": 1 ms).  Half the carrier spacing also halves the CFO range in Hz.
    "iq-hf": {"n_fft": 512, "cp_ratio": 0.5, "window": "late", "ctl_repeat_s": 0.6},
}


def iq_profile(name: str = "iq", **over) -> IqConfig:
    """``IqConfig`` for a named profile, with field overrides on top."""
    if name not in IQ_PROFILES:
        raise KeyError(f"unknown I/Q profile {name!r}; have {sorted(IQ_PROFILES)}")
    d = IqConfig().to_dict()
    d.update(IQ_PROFILES[name])
    d.update(over)
    return IqConfig.from_dict(d)
