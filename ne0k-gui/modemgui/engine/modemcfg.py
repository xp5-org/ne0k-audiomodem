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


@dataclasses.dataclass
class OfdmConfig(_Config):
    """OFDM link.  Both ends must agree on the PHY group, ``window_s``, ``guard_s``,
    ``chunk_bytes`` and the band; the BAT exchange only settles the bit loading."""
    node_id: int = F(1, "Node id", "Link", "", 0, 255)
    peer_id: int = F(2, "Peer id", "Link", "", 0, 255)
    role: str = F("auto", "Role", "Link", "auto = the lower node id initiates.",
                  choices=["auto", "initiator", "responder"])
    mode: str = F("half", "Mode", "Link",
                  "half = timed windows both ways, oneway/stream = one sender, full = FDD, "
                  "arq = one-way with ACKs (mesh data plane).",
                  choices=["half", "oneway", "stream", "full", "arq"])
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
    duplex_guard_hz: float = F(600.0, "Duplex guard band (Hz)", "Full duplex", "", 0, 5000)
    dl_frac: float = F(0.5, "Responder share of band", "Full duplex", "", 0.1, 0.9, step=0.05)
    ack_wait_s: float = F(8.0, "Resend after (s)", "Full duplex / ARQ",
                          "Resend a frame when no ACK arrived within this.", 1, 120)
    carrier_db: float = F(-55.0, "Carrier threshold (dBFS)", "Receive", "", -100, 0)
    hear_own: bool = F(False, "Decode own transmissions", "Receive", "")
    backend: str = F("python", "DSP backend", "Engine",
                     "c = libofdmacq + librs255 (process-wide). "
                     "cdx = c, plus the full-duplex data plane (mode full) run by the C engine.",
                     choices=["python", "c", "cdx"])


@dataclasses.dataclass
class GeneralConfig(_Config):
    modem: str = F("css", "Modem", "Mode", "css = control plane only (robust, slow); ofdm = CSS control plane "
                   "plus an OFDM link brought up per transfer, falling back to CSS data if OFDM fails; "
                   "ofdm_only = same OFDM upgrade, but a failed transfer stays failed -- no silent CSS "
                   "fallback (that state machine has its own regression, mesh_fallback_regress.py).",
                   choices=["css", "ofdm", "ofdm_only"])
    topology: str = F("p2p", "Topology", "Mode", "p2p = one pair of stations; mesh = N stations that discover "
                      "each other with beacons.", choices=["p2p", "mesh"])
    node_id: int = F(1, "Node id", "Addressing", "This station's address, 0..15.", 0, 15)
    peer_id: int = F(15, "Peer id", "Addressing", "p2p only: the other station.  In a mesh the peer is "
                     "not configured: it is whoever answers.", 0, 15)
    mesh_dst: int = F(0, "Pinned mesh peer", "Addressing", "Mesh only: the virtual serial port always talks "
                      "to this station instead of whoever is heard strongest; 0 = auto (best neighbour).", 0, 15)
    beacon: bool = F(True, "Send beacons", "Mesh beacons", "Mesh only: announce this node on the CSS "
                     "control plane so the others can find it.")
    beacon_min_s: float = F(20.0, "Beacon interval min (s)", "Mesh beacons",
                            "The beacon interval follows the measured clock drift of the other nodes "
                            "(faster drift, more often), kept between min and max.", 1.0, 3600.0)
    beacon_max_s: float = F(300.0, "Beacon interval max (s)", "Mesh beacons", "", 1.0, 3600.0)
    first_beacon_s: float = F(20.0, "Listen before first beacon (s)", "Mesh beacons",
                              "Stay silent and listen for up to this long (random, plus 1 s per node id) before "
                              "the first beacon, so nodes started together do not collide.", 0.0, 600.0)
    cw_s: float = F(4.0, "Contention window (s)", "Mesh channel access",
                    "CSMA/CA: before sending, wait a random time up to this, plus 0.1 s per node id.", 0.0, 60.0)
    ack_stagger_s: float = F(0.5, "ACK stagger per node id (s)", "Mesh channel access",
                             "A node holds its ACK by id x this, so two peers never answer at the same instant.",
                             0.0, 5.0)
    ofdm_detect: bool = F(True, "Passive OFDM detector", "Mesh channel access",
                          "OFDM mesh: after giving up on OFDM, listen for OFDM on the air and ask again over CSS.")
    airtime_windows: bool = F(False, "Per-link handshake windows", "Mesh channel access",
                              "Give each link its own wall-clock window for handshakes so two never overlap.  "
                              "Needs both ends' clocks within a second or so.")
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
                    "Bytes accepted per send; the modem raises CTS again after each.", 1, 250)
    mode: str = F("rts", "Send trigger", "Handshake",
                  "rts = send on the RTS rising edge.  auto = send when the buffer is full or idle.",
                  choices=["rts", "auto"])
    idle_s: float = F(2.0, "Auto idle (s)", "Handshake", "auto mode: send after this long without input.",
                      0.1, 60, step=0.5)
    ack: bool = F(True, "Peer ACKs each buffer", "Reliability",
                  "Retransmit the buffer when no ACK arrives (CSS).")
    retries: int = F(3, "Retries", "Reliability", "Retransmissions before TXFAIL.", 0, 20)
    rx_host: int = F(0, "RX host filter", "Port", "Mesh only: only hand the host bytes received from this "
                     "station id; 0 = accept from any neighbour (CSS is mostly multicast/broadcast).", 0, 15)


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
