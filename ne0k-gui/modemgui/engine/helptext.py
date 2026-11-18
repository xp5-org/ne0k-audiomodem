# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Built-in help for every setting.  Plain data, no Qt.

The key is ``<section>.<field>``, the same dotted name the UI inspector shows (Help mode is part of
View > Inspect UI names).  Each entry is ``(impact, ofdm)``:

* impact -- what changing it does to speed, robustness or behaviour;
* ofdm   -- how it relates to the OFDM link (carriers, cyclic prefix, bit loading, ARQ...).

*What the setting is* comes from the field's own ``F()`` label and tip in ``modemcfg``, so a tip
edited there shows up here without touching this file.  ``topic()`` joins the two."""
from __future__ import annotations

from . import modemcfg as mc

SECTION_TITLES = {"general": "Modem Mode", "output": "Output Device", "css": "CSS Modem", "ofdm": "OFDM Modem",
                  "ham": "Ident", "serial": "Virtual serial port", "udp": "UDP port",
                  "cssbcast": "CSS broadcast port", "diag": "Debug & Logging", "remote": "Remote Control"}
SECTION_CLASSES = {"general": mc.GeneralConfig, "output": mc.OutputConfig, "css": mc.CssConfig, "ofdm": mc.OfdmConfig,
                   "ham": mc.HamConfig, "serial": mc.SerialConfig, "udp": mc.UdpConfig,
                   "cssbcast": mc.CssBcastConfig, "diag": mc.DiagConfig, "remote": mc.RemoteConfig}

OVERVIEW = """\
<b>OFDM in one paragraph.</b> The OFDM modem splits the audio band (Band low to Band high) into many
narrow <i>carriers</i>, spaced sample rate / FFT size apart (1024 at 48 kHz = 46.9 Hz).  Each carrier
carries 1, 2, 3, 4 or 6 bits per symbol (BPSK, QPSK, 8-PSK, 16-QAM, 64-QAM).  A <i>probe</i> measures the SNR of
every carrier and a <i>bit allocation table</i> (BAT) gives each carrier as many bits as it can hold, so a
good channel is fast and a bad one slows down instead of failing.  A <i>cyclic prefix</i> at the front of
each symbol absorbs echoes, <i>pilot</i> carriers track phase and level, and Reed-Solomon parity repairs
what is left.  Both ends must agree on everything marked "both ends equal"; the BAT exchange only settles
the bit loading."""

# -- general (Modem Mode) ------------------------------------------------------------------------------
_GENERAL = {
    "modem": ("Chooses the whole PHY.  CSS is slow and survives bad paths; OFDM is far faster but needs a cleaner channel; "
              "the mixed mode uses CSS only to coordinate.", "Every OFDM setting is ignored while this is css.  The OFDM page "
              "only matters for the OFDM and mixed choices."),
    "topology": ("p2p needs a configured peer; a mesh finds peers by beacon; hub topologies poll the stations in turn.  It changes who "
                 "may transmit when, not the modulation.", "Each OFDM link inside a mesh or hub is still a pair that probes and exchanges "
                 "BATs.  Mesh adds beacon, backoff and handshake windows around those pairs."),
    "hub_id": ("Wrong value and clients never find the leader.", "The hub runs the OFDM handshake with each client in turn."),
    "hub_nodes": ("Too low closes the join window before everyone arrived; too high keeps it open longer.", "Each station is an OFDM link to the hub."),
    "hub_turn_s": ("Longer turns move more data per grant but make others wait.", "Must be longer than a handshake plus a frame at the BAT rate."),
    "hub_cycle_s": ("Lower = lower latency on an idle network, more airtime spent polling.", "Polls are short OFDM control frames."),
    "hub_page_timeout_s": ("Too short fails pages that only needed to wait their turn.", "Counts airtime at the OFDM data rate."),
    "node_id": ("Sets the station address, also the tie-break for who initiates and how long backoff waits.",
                "With OFDM role auto the lower id initiates the probe/BAT handshake."),
    "peer_id": ("Frames are addressed to it in p2p.", "The OFDM handshake is made with this peer only."),
    "mesh_dst": ("Fixes the serial port's destination so traffic does not jump to whichever neighbour is loudest.",
                 "Chooses which OFDM link the serial bytes use."),
    "beacon": ("Without beacons others cannot find this node.", "ofdm_only mesh sends an OFDM beacon frame; the mixed mode sends it over CSS."),
    "beacon_min_s": ("Lower = faster discovery and more airtime.", "Beacon frames cost an OFDM burst each."),
    "beacon_max_s": ("Upper bound of the drift-adaptive interval.", "Stale beacons mean stale BATs after clocks drift."),
    "first_beacon_s": ("Longer listening avoids everyone beaconing on power-up together.", "Avoids OFDM bursts colliding at start-up."),
    "ignore_busy": ("Skips the wait-for-quiet checks.  Faster but collides.  A test aid, not for a shared channel.",
                    "The OFDM handshake turns are skipped too, so handshakes can overlap and fail."),
    "carrier_detect": ("level trusts loudness, signal recognises the modem's own waveforms.  Use signal with a "
                       "microphone or a radio; every station can choose for itself.",
                       "OFDM is found by its cyclic prefix at the FFT size and prefix ratio set on the OFDM page."),
    "sense_ofdm_thr": ("Lower detects weaker OFDM and false-triggers more; a false trigger costs a decode attempt, "
                       "not a lost frame.", "Compared with the cyclic-prefix correlation of the last 0.2 s."),
    "sense_chirp_thr": ("Lower detects weaker chirps and false-triggers more.", "OFDM bursts are found separately."),
    "slot_s": ("Backoff granularity.  Must be longer than the time to hear a transmission start.", "Should exceed the OFDM lead tone plus detection time."),
    "cw_slots": ("A larger initial window means fewer collisions and more average delay.", "Collisions cost a whole OFDM burst."),
    "cw_max_slots": ("Caps the doubling after unanswered packets.", "Bounds worst-case delay at the OFDM burst rate."),
    "ack_stagger_s": ("Keeps two peers from ACKing at the same instant.", "ACKs are short OFDM frames on the full bit table."),
    "ofdm_detect": ("Lets a node that gave up on OFDM notice OFDM again and ask over CSS.", "Looks for OFDM sync (probe/lead tone) on the air."),
    "airtime_windows": ("Gives each link a wall-clock window so handshakes never overlap.  Needs clocks within about a second.",
                        "Keeps probe and BAT exchanges, the longest OFDM bursts, from colliding."),
    "auto_link": ("Trains the link at power-on instead of on the first send, so the first message does not pay for the handshake.",
                  "Runs the probe and BAT exchange early, giving each carrier its bit count before data."),
    "announce_round": ("ofdm_only: stations announce in id order instead of all at once.", "Orders the OFDM announce/BAT traffic."),
    "announce_bat_turn_s": ("How long a station waits for its BAT turn.", "Covers a probe plus BAT exchange."),
    "announce_bat_mode": ("pairwise is the robust choice acoustically; other modes train faster.", "Chooses how BATs are measured between stations."),
    "announce_cycle_s": ("Lower = faster catch-up for late joiners, more airtime.", "Each pass can repeat OFDM probes."),
    "hs_window_s": ("Length of each handshake window.", "Must fit a complete OFDM handshake."),
    "hs_slots": ("More slots = fewer links sharing one window.", "Spreads probes and BATs over time."),
    "hs_guard_s": ("A handshake may only start early enough to finish before the window ends.", "OFDM probe + BAT length sets the minimum."),
    "blockout_role": ("A test/operator feature: announce makes all other nodes mute for a window.", "Used to give an OFDM exchange clear air."),
    "blockout_peer": ("The other participant of the exclusive window.", "The OFDM pair that has the air."),
    "blockout_at_s": ("When the window is announced.", "Time the exclusive OFDM exchange starts."),
    "blockout_go_file": ("Starts the window on demand instead of a guessed delay.", "Lets a test line up the OFDM exchange exactly."),
    "blockout_duration_s": ("How long everyone else stays muted.", "Long enough for the OFDM payload at the BAT rate."),
    "blockout_payload": ("The text sent in the window.", "Sent as OFDM frames to the peer."),
}

# -- output ---------------------------------------------------------------------------------------------
_OUTPUT = {
    "io_type": ("Audio uses the sound card (acoustic, radio or a SIP line); GNU Radio / libsdr use I/Q samples.",
                "The audio path is real-valued OFDM; the SDR path uses complex baseband OFDM (a separate, additive PHY)."),
    "tx_channels": ("left / right keep the other channel silent (free for a TX/RX switch); both doubles the speakers "
                    "playing, about 6 dB louder on a laptop than one channel.", "Same level setting for CSS and OFDM."),
    "location": ("Local runs the connector in this process's machine; remote goes over TCP.", "Does not change the modulation."),
    "host": ("Wrong address and the link never opens (Test connection shows it).", "Carries the OFDM I/Q stream."),
    "port": ("Must match the remote end.", "Carries the OFDM I/Q stream."),
}

# -- css ------------------------------------------------------------------------------------------------
_CSS_NOTE = "CSS is the alternative to OFDM: chirps instead of carriers.  It matters to OFDM only in the mixed mode, where CSS carries control and beacons."
_CSS = {
    "node_id": ("This end's address.", _CSS_NOTE),
    "peer_id": ("15 broadcasts to all.", _CSS_NOTE),
    "fs": ("Must equal the audio device rate or the chirps are mistimed.", "OFDM also runs at 48000 or 44100 only."),
    "f_min": ("Sets the bottom of the chirp sweep.  Use 400 for a phone line.", "Compare OFDM band low, which clips the carriers instead."),
    "f_max": ("Sets the top of the sweep.  Use 3300 for a phone line.", "Compare OFDM band high."),
    "symbol_dur": ("Longer symbols are slower and more robust; 0.03 s and below fail to decode.", "The OFDM equivalent is FFT size: longer symbols, narrower carriers."),
    "symbol_gap_frac": ("Silence between chirps keeps echoes from smearing the next symbol.", "Plays the role of the OFDM cyclic prefix."),
    "amplitude": ("Higher is more SNR until the audio path clips.", "OFDM has a larger peak-to-average ratio, so it needs lower amplitude."),
    "chunk_size": ("Bigger frames are more efficient but a lost one costs more.", "Compare OFDM payload bytes per frame."),
    "short_frames": ("2-byte packets (ACKs, beacons) go as 2-byte frames: 15.5 s instead of 23 s at 0.15 s symbols.  Every station must match.", _CSS_NOTE),
    "crc_bytes": ("CRC-16 saves 16 chirps per frame; CRC-32 detects more errors.  Every station must match.", "OFDM frames use Reed-Solomon parity instead."),
    "diff_encode": ("Survives a constant phase offset.", "OFDM gets the same from its pilot carriers."),
    "rs_t": ("More parity corrects more bytes and costs airtime.  0 = off.", "OFDM has its own RS parity, sized per frame."),
    "interleave_rows": ("Spreads burst errors across codewords.", "OFDM bit interleaving serves the same purpose."),
    "conv_code": ("k7r12 halves the rate for soft-decision gain.", "OFDM's inner FEC option is the same code."),
    "carrier_db": ("Lower hears weak signals and false-triggers on noise.", "OFDM uses its own carrier threshold on the Receive group."),
    "hang_s": ("How long quiet must last before a burst is declared over.", "Keep shorter than the gap between frames."),
    "hear_own": ("Turn on for a self-loop through a null sink.", _CSS_NOTE),
    "backend": ("c is faster and needs the libraries; python is the fallback.", _CSS_NOTE),
    "decoder": ("pll_c is continuous; mf is per burst and works at any symbol length.", _CSS_NOTE),
}

# -- ofdm -----------------------------------------------------------------------------------------------
_OFDM = {
    "node_id": ("Station address.", "With role auto the lower id initiates the probe."),
    "peer_id": ("Frames are addressed here.", "The OFDM handshake is made with this peer."),
    "channel": ("radio adds a 2 s 1000 Hz lead tone for PTT, squelch and the far receive gate; line sends the frame as is.  Wrong choice costs 2 s per burst or loses the PTT.",
                "The lead tone precedes the OFDM probe and every burst."),
    "keyup_lead_s": ("Length of the radio lead tone.  Too short and a PTT radio or a waking Bluetooth link eats the "
                     "start of the frame; too long wastes air time on every burst (2 s for a PTT radio, ~0.5 s for "
                     "a Bluetooth FM transmitter).",
                     "The probe and the first chunk of every broadcast keying follow the tone."),
    "role": ("Decides who sends the first probe.", "Both ends cannot be initiator; auto uses the lower node id."),
    "mode": ("half alternates timed windows; oneway/stream has one sender; full is simultaneous; arq adds ACKs; bcast is continuous.  Changes the speed, delay and what the receiver may do.",
             "Chooses how frames are scheduled; the carriers and BAT are the same in every mode."),
    "sender": ("Which side transmits data in oneway/stream/arq.", "The BAT is still exchanged both ways."),
    "fs": ("Only 48000 or 44100 work.  Must match the device.", "Carrier spacing is rate / FFT size, so this scales it."),
    "freq_lo": ("Raises or lowers the bottom of the band.  Higher cuts low hum and rumble but loses carriers.", "Carriers exist only between band low and high; fewer carriers means less total rate."),
    "freq_hi": ("Raising it adds carriers if the channel passes them.  A phone line stops near 3.4 kHz.", "More band = more carriers = more bits, but only while their SNR holds."),
    "amplitude": ("Raises SNR until the audio path clips.", "OFDM peaks run well above RMS; the level is set on the probe and later bursts are RMS-matched."),
    "wait_probe_s": ("How long to wait for the peer's probe.", "The probe is the 5 s measurement burst."),
    "wait_msg_s": ("How long to wait for the next handshake message.", "Covers the BAT exchange."),
    "hs_retries": ("More retries survive a failed probe and cost time.", "Each retry repeats the probe."),
    "hs_timeout_s": ("Whole-handshake limit.", "Probe + BAT + retries must fit."),
    "poll_s": ("Lower reacts faster and uses more CPU.", "How often the receiver looks for an OFDM burst."),
    "n_fft": ("Bigger = narrower carriers, more of them, longer symbols and more tolerance of echo; smaller = shorter symbols and more tolerance of fast fading and clock drift.  Both ends must match.",
              "The core OFDM size: carrier spacing = rate / FFT size."),
    "cp_ratio": ("Larger outlasts longer room reflections and loses more of each symbol to the prefix.  Both ends must match.",
                 "The cyclic prefix is copied from the end of each symbol so echoes fall inside it and carriers stay orthogonal."),
    "pilot_spacing": ("Smaller = better tracking of a changing channel and fewer data carriers.", "Pilots are known carriers used to track phase and level."),
    "probe_duration_s": ("Longer probes measure SNR more accurately and take longer.  Do not shorten it for speed on a fading radio.", "The probe measures SNR per carrier for the BAT."),
    "probe_meas_syms": ("Non-periodic symbols make the SNR estimate honest.  0 = periodic only.", "A periodic probe can look better than the channel is."),
    "snr_margin_db": ("More margin = fewer errors and fewer bits; less margin = faster and fragile.", "Subtracted from each carrier's SNR before assigning bits."),
    "max_bits_per_bin": ("1 BPSK, 2 QPSK, 3 8-PSK, 4 16-QAM, 6 64-QAM.  Higher caps allow more speed on clean channels.", "The ceiling of bit loading on any carrier."),
    "snr_estimator": ("percentile is more cautious than mean.", "Chooses how per-carrier SNR is summarised."),
    "isi_derate": ("1 folds the measured reverberation ceiling into the allocation.", "ISI is energy from the previous symbol beyond the prefix."),
    "window_s": ("Longer windows carry more per turn and add latency.  Both ends equal.", "Each window holds OFDM frames at the BAT rate."),
    "guard_s": ("Absorbs clock skew between windows.  Too small causes overlap.", "Dead time at both ends of a window."),
    "chunk_bytes": ("Bigger frames are efficient and costly to lose.  Both ends equal.", "Each frame is a number of OFDM symbols; a smaller chunk is fewer."),
    "retries": ("More retries before TX fails.", "Each retry resends the whole OFDM frame."),
    "gap_s": ("Quiet between frames so the receiver sees separate bursts.", "Gap between OFDM bursts."),
    "hang_s": ("How long quiet must last before a burst ends.", "Shorter than the frame gap."),
    "stream_poll_s": ("How often the stream decoder runs.", "Decodes whatever OFDM has arrived."),
    "duplex": ("fdd splits the band; ec uses all of it and cancels its own echo.  Both ends must agree.", "fdd gives each direction half of the carriers; ec gives each direction all carriers."),
    "ec_taps": ("Longer cancels longer echoes and costs CPU.  0 = none.", "The canceller subtracts our own signal from what the receiver hears."),
    "duplex_guard_hz": ("Wider guard cuts leakage and wastes band.", "Carriers inside the guard are left unused."),
    "dl_frac": ("Moves the split of the band or capacity between directions.", "Fewer carriers for one direction = less rate."),
    "ratio": ("Sets the data split initiator/responder.  Both ends agree.", "Acts on the OFDM data capacity."),
    "ec_window": ("Frames in flight before waiting for an ACK.  1 = stop-and-wait.", "Larger windows hide the OFDM round-trip delay."),
    "duplex_split": ("bits splits by capacity, hz by bandwidth.", "Uses the exchanged BATs."),
    "ack_wait_s": ("Resend when no ACK arrives within this.", "Must exceed an OFDM frame plus the ACK delay."),
    "schedule_tx": ("A score restricts each frame to the carriers of a tune: a demo effect, not for speed.", "Cuts the bit table to a few carriers."),
    "schedule_rx": ("The receiver expects the peer's score.", "Matches spectra against the expected tune."),
    "schedule_step_carriers": ("More carriers per step = shorter notes.  Both ends equal.", "How many carriers' worth of bits each note carries."),
    "tx_window_chunks": ("Reserves many chunks and streams them, with one status at the end.  0 = stop-and-wait.  Both ends equal.", "Removes the per-frame OFDM ACK gap."),
    "schedule_boost_db": ("Raises the tune carriers over the pilots.", "Pilots stay at unit power."),
    "carrier_db": ("Lower hears weak bursts and may trigger on noise.", "Where an OFDM burst is first detected."),
    "hear_own": ("Decode our own audio, for a self-loop.", "The receiver sees our OFDM as a peer's."),
    "backend": ("c is faster; cdx moves the full-duplex data plane into C.", "The OFDM engine itself, not the signal."),
}

HELP: dict[str, tuple[str, str]] = {}
for _sec, _tbl in (("general", _GENERAL), ("output", _OUTPUT), ("css", _CSS), ("ofdm", _OFDM)):
    for _k, _v in _tbl.items():
        HELP[f"{_sec}.{_k}"] = _v

# -- ham / serial / udp / cssbcast / diag ---------------------------------------------------------------
_SIMPLE = {
    "ham": ("Station identification is a legal requirement on amateur bands; it does not change the modulation.",
            "The ID audio is sent beside the OFDM frames, and its gaps must exceed the receiver's burst-end hold."),
    "serial": ("Controls the virtual serial port between the host program and the modem.",
               "The bytes you write are the OFDM payload; buffer size and ACKs shape how OFDM frames are filled."),
    "udp": ("Moves datagrams in or out of the OFDM link, e.g. a video stream.", "A one-way OFDM stream with no ACKs."),
    "cssbcast": ("Broadcast text and a time signal over CSS.", "This port is CSS only and has no OFDM involvement."),
    "remote": ("Lets another program or host drive the modem over a TCP text protocol; it does not alter the signal.",
               "Whatever it sends goes through the same OFDM engine as the window's own buttons."),
    "diag": ("Logging and capture; none of it alters the signal.", "Captured WAVs let you replay an OFDM decode offline."),
}
for _sec in _SIMPLE:
    for _d in mc.describe(SECTION_CLASSES[_sec]):
        HELP.setdefault(f"{_sec}.{_d['name']}", _SIMPLE[_sec])


def keys() -> list[str]:
    return [f"{s}.{d['name']}" for s, c in SECTION_CLASSES.items() for d in mc.describe(c)]


def _field(key: str):
    sec, _, name = key.partition(".")
    cls = SECTION_CLASSES.get(sec)
    if cls is None:
        return None
    for d in mc.describe(cls):
        if d["name"] == name:
            return d
    return None


def topic(key: str) -> dict | None:
    """Everything the help window shows for ``key``, or None for an unknown setting."""
    d = _field(key)
    if d is None:
        return None
    impact, ofdm = HELP.get(key, ("", ""))
    return dict(key=key, section=SECTION_TITLES[key.partition('.')[0]], label=d["label"], group=d["group"],
                what=d.get("tip") or "", default=d["default"], lo=d.get("lo"), hi=d.get("hi"),
                choices=d.get("choices"), impact=impact, ofdm=ofdm)


def html(key: str | None = None) -> str:
    """The help page: the overview, then every setting (anchor ``key``) grouped by section."""
    from html import escape as e
    out = [f"<h2>Help</h2><p>{OVERVIEW}</p>"]
    for sec, cls in SECTION_CLASSES.items():
        out.append(f"<h3>{e(SECTION_TITLES[sec])}</h3>")
        for d in mc.describe(cls):
            t = topic(f"{sec}.{d['name']}")
            rng = f" (range {t['lo']} to {t['hi']})" if t["lo"] is not None and t["hi"] is not None else ""
            ch = f" Choices: {', '.join(str(c) for c in t['choices'])}." if t["choices"] else ""
            out.append(
                f"<a name='{t['key']}'></a><h4>{e(t['label'])} <small>[{t['key']}]</small></h4>"
                f"<p><b>What:</b> {e(t['what'] or t['label'])}{e(ch)} Default {e(str(t['default']))}{e(rng)}.<br>"
                f"<b>Impact:</b> {e(t['impact'])}<br><b>OFDM:</b> {e(t['ofdm'])}</p>")
    return "".join(out)


# -- the window itself: every panel, control and menu entry ---------------------------------------------
# Key = "ui.<name>" or "ui.<parent>.<name>" from the widgets' object names (the attribute that holds them; see
# ui/context_help.py for the lookup), or "menu.<Menu>[.<Submenu>].<Entry>" with the menu text minus "&".
# Right-click any element for its entry; the window class names (ui.MainWindow, ui.SettingsDialog...) catch
# what has no entry of its own.

UI_GROUPS: list[tuple[str, dict[str, tuple[str, str]]]] = [
    ("Windows", {
        "ui.MainWindow": ("Main window", "The modem's front page: the I/O column on the left (audio devices, start / stop, "
                          "link and serial-port figures), the PLL and CPU bars on top, and the Event Log, BAT, Log and "
                          "Spectrograph panes.  Right-click any element for its help; View > Layout rearranges the panes."),
        "ui.SettingsDialog": ("Settings dialog", "Every modem setting, one page per section.  Right-click a field for what it "
                              "does, its impact and how it relates to OFDM.  OK applies and saves; settings marked "
                              "restart take effect the next time the modem is started."),
        "ui.HelpWindow": ("Help window", "This window: the built-in help for every setting and every element of the window."),
        "ui.statusbar": ("Status bar", "The last message from the modem or a button (the same lines go to the log).  "
                         "Edit > Copy status copies it."),
        "ui.libs_lbl": ("C libraries", "How many of the native DSP libraries loaded.  Missing ones fall back to Python, "
                        "which is slower but decodes the same.  Help > C library inventory lists them."),
        "ui.qt_dockwidget_floatbutton": ("Float pane", "Detach this pane into its own window; double-click its title to "
                                         "dock it again."),
        "ui.qt_dockwidget_closebutton": ("Close pane", "Hide this pane.  Bring it back from the View menu."),
    }),
    ("I/O column", {
        "ui.dock-io": ("I/O column", "Picks the audio devices, starts and stops the modem, and shows the link and "
                       "serial-port figures."),
        "ui.column": ("I/O column", "Picks the audio devices, starts and stops the modem, and shows the link and "
                      "serial-port figures."),
        "ui.io_type": ("I/O Type", "Where the modem's audio goes: sound-card audio now; the greyed-out entries are "
                       "planned transports."),
        "ui.column.mode": ("Mode", "The modem, topology and node ids in use.  Change them under Settings > Modem Mode "
                           "Settings."),
        "ui.io": ("Audio I/O", "The sound devices the modem listens and talks on, the buttons that open them and start "
                  "the modem, and the receive level meter."),
        "ui.io.rx": ("Input (RX)", "The device the modem listens on (microphone, line in, a loopback or a Pulse source).  "
                     "Takes effect on Open audio."),
        "ui.io.tx": ("Output (TX)", "The device the modem transmits on (speaker, line out, a Pulse sink).  Takes effect on "
                     "Open audio."),
        "ui.unavail": ("Unavailable devices", "Why some audio backends are missing on this host (a library not installed, "
                       "no permission...)."),
        "ui.refresh_btn": ("Refresh", "List the audio devices again, e.g. after plugging one in."),
        "ui.open_btn": ("Open audio", "Open (or close) the chosen input and output.  The level meter and spectrograph run "
                        "while audio is open, even with the modem off."),
        "ui.power_btn": ("Start modem", "Power the modem on: it handshakes and trains with its peer right away.  Stop it "
                         "to change modem settings.  Needs audio open."),
        "ui.test_btn": ("Test loopback", "Play a test tone on the output and look for it on the input: checks the devices "
                        "and the level before a real link."),
        "ui.peak": ("RX level", "Peak level of what the input hears.  Aim for speech-like peaks well below 0 dBFS; a bar "
                    "at the top means clipping, an empty one means the mic hears nothing."),
        "ui.peak_lbl": ("RX peak dBFS", "The input's recent peak in dB below full scale (0 dBFS = clipping)."),
        "ui.clip_lbl": ("Clip count", "How many input blocks reached full scale since audio opened.  Rising means the input "
                        "gain (or the far end's volume) is too high: decoding fails on clipped audio."),
        "ui.tx_lbl": ("TX indicator", "Lit while this node is transmitting."),
        "ui.link_info": ("Link Info", "Rates of the running link: the raw rate on the air and the payload bitrate after "
                         "framing, FEC and handshakes."),
        "ui.rx_rate": ("RX Linkrate", "Raw bit rate of the frames being received (before FEC and framing)."),
        "ui.rx_bitrate": ("RX Bitrate", "Payload bits per second actually delivered from the peer."),
        "ui.tx_rate": ("TX Linkrate", "Raw bit rate this node transmits at; for OFDM, the carrier's rate while keyed."),
        "ui.tx_bitrate": ("TX Bitrate", "Payload bits per second this node gets across."),
        "ui.overhead": ("% Overhead", "Share of airtime spent on preambles, FEC, ACKs and handshakes instead of payload."),
        "ui.link_info.detail": ("Link detail", "Extra figures from the engine about the current link (BAT, retries...)."),
        "ui.uart": ("Serial uart info", "The virtual serial port that host programs use to send and receive data over the "
                    "modem."),
        "ui.uart.state": ("Port", "Whether the virtual serial port is open, and its path."),
        "ui.uart.mode": ("Mode", "How the port moves data (raw bytes, line mode, flow control)."),
        "ui.uart.buf": ("Buffer", "Bytes waiting in the port's transmit buffer / its size.  Full means the host writes "
                        "faster than the link sends."),
        "ui.uart.rx": ("RX", "Bytes received from the peer and handed to the port."),
        "ui.uart.tx": ("TX", "Bytes taken from the port and sent to the peer."),
        "ui.retries": ("Retries / fails", "Frames that had to be resent, and frames given up on after all retries."),
        "ui.port_btn": ("Open virtual serial port", "Create (or remove) the pty that host programs open as a serial port "
                        "(Ctrl+P).  Needs a POSIX system."),
    }),
    ("Bars", {
        "ui.bar-pll": ("PLL bar", "Receiver lock: state, time to acquire and preamble alignment."),
        "ui.pll": ("PLL bar", "Receiver lock: state, time to acquire and preamble alignment."),
        "ui.pll.state": ("PLL state", "Whether the receiver is locked onto a burst, idle, or audio is closed.  'locked' "
                         "fades to idle a few seconds after the last lock."),
        "ui.timing": ("Time to acquire", "How long the receiver took to lock onto the last burst, from its start."),
        "ui.align": ("Align", "Preamble correlation score at the last lock (100% = perfectly aligned).  Low values mean a "
                     "weak or distorted signal."),
        "ui.bar-status": ("CPU bar", "Process CPU, per stage and thread, plus buffer and memory figures."),
        "ui.strip": ("CPU bar", "Process CPU, per stage and thread, plus buffer and memory figures."),
        "ui.cpu": ("CPU", "Total CPU of the modem process over the last second (100% = one core)."),
        "ui.stages": ("CPU per stage", "CPU per DSP stage (Decode, Encode, CSS PLL) and per thread group (UI, link, RX "
                      "capture, audio...).  Hover for the busiest threads."),
        "ui.strip.buf": ("Buffer / memory", "Engine event queue depth, events dropped because the UI fell behind, and the "
                         "process memory."),
    }),
    ("Event Log", {
        "ui.dock-event log": ("Event Log", "Timeline of TX, RX, errors, PLL locks and ACKs, one lane per node."),
        "ui.pianoroll": ("Event Log", "Timeline of TX, RX, errors, PLL locks and ACKs, one lane per node."),
        "ui.pianoroll.window": ("Window", "How many seconds of the timeline are shown."),
        "ui.follow": ("Follow live", "Keep the newest events in view.  Untick to scroll back."),
        "ui.pianoroll.plot": ("Timeline", "Each lane is a node; bars are transmissions, dots are receptions, errors, locks "
                              "and ACKs.  Drag to pan, wheel to zoom."),
        "ui.scroll": ("Scroll", "Move back through the event history."),
        "ui.summary": ("Event counts", "Totals of each kind of event since the log was cleared (View > Clear events)."),
    }),
    ("BAT", {
        "ui.dock-bat": ("BAT", "The OFDM channel probe: per-carrier SNR and the bits each carrier was given."),
        "ui.bat": ("BAT", "One chart per probe: bar height is the SNR the probe saw on that carrier and its colour the "
                   "bits the allocator gave it.  The first chart is what this end measured, the next what the peer "
                   "measured of us.  Right-click a chart for zoom and export."),
        "ui.bat.legend": ("BAT colours", "Bar colour = bits per symbol the allocator gave the carrier: pilot, off, BPSK (1), "
                          "QPSK (2), 8-PSK (3), 16-QAM (4), 64-QAM (6)."),
        "ui.bat.hint": ("No probe yet", "The charts appear once an OFDM probe has been exchanged with the peer."),
    }),
    ("Log", {
        "ui.dock-log": ("Log", "The modem's text log: status, link events, decoder messages."),
        "ui.logview": ("Log", "The modem's text log: status, link events, decoder messages."),
        "ui.text": ("Log text", "Select and copy lines from the log.  The same log goes to the capture folder."),
    }),
    ("Spectrograph", {
        "ui.dock-spectrograph": ("Spectrograph", "Live spectrogram of the input: frequency up, time to the right."),
        "ui.spectro": ("Spectrograph", "Live spectrogram of the input: frequency up, time to the right."),
        "ui.floor": ("Floor dB", "Level shown as the darkest colour.  Raise it to hide the noise floor."),
        "ui.ceil": ("Ceil dB", "Level shown as the brightest colour.  Lower it to see weak signals."),
        "ui.maxhz": ("Max Hz", "Highest frequency shown."),
        "ui.overlay": ("Event overlay", "Draw node TX/RX/ERR/PLL-lock/ACK events over the spectrogram."),
        "ui.spectro.plot": ("Spectrogram", "Drag to pan, wheel to zoom; right-click for the plot menu (view all, export)."),
    }),
    ("Menus", {
        "menu.File": ("File menu", "Load and save settings files, quit."),
        "menu.File.Open Settings...": ("Open Settings", "Load all settings from a file (Ctrl+O)."),
        "menu.File.Save Settings...": ("Save Settings", "Save all settings to a file (Ctrl+S)."),
        "menu.File.Exit": ("Exit", "Close the modem (Ctrl+Q); settings and the window layout are saved."),
        "menu.Edit": ("Edit menu", "Copy the status line, send the station ID."),
        "menu.Edit.Copy status": ("Copy status", "Copy the status bar's message to the clipboard (Ctrl+C)."),
        "menu.Edit.Send ID now": ("Send ID now", "Transmit the station identification immediately (Ctrl+I)."),
        "menu.Settings": ("Settings menu", "Open the settings dialog at one section."),
        "menu.Settings.Output Device Settings...": ("Output Device Settings", "Sample rate, block size and output options."),
        "menu.Settings.Modem Mode Settings...": ("Modem Mode Settings", "Modem (CSS / OFDM / mixed), topology, node ids, "
                                                 "beacons and channel access."),
        "menu.Settings.CSS Modem Settings...": ("CSS Modem Settings", "Chirp band, symbol length, FEC and level of the CSS "
                                                "modem."),
        "menu.Settings.OFDM Modem Settings...": ("OFDM Modem Settings", "Band, FFT size, cyclic prefix, bit loading, ARQ "
                                                 "and level of the OFDM modem."),
        "menu.Settings.Ident Settings...": ("Ident Settings", "Call sign and station identification timing."),
        "menu.Settings.Ports (serial, UDP, broadcast)...": ("Ports", "Virtual serial port, UDP port and CSS broadcast port."),
        "menu.Settings.Remote Control (TCP/IP)...": ("Remote Control", "The TCP control port other programs drive the "
                                                     "modem through."),
        "menu.Settings.Debug & Logging...": ("Debug & Logging", "Log detail, captures and the capture folder."),
        "menu.View": ("View menu", "Show or hide panes, arrange them, and the inspect / help modes."),
        "menu.View.I/O": ("I/O", "Show or hide the I/O column."),
        "menu.View.Event overlay on spectrograph": ("Event overlay", "Draw node events over the spectrogram."),
        "menu.View.Clear events": ("Clear events", "Empty the Event Log and its counts."),
        "menu.View.Spectrograph": ("Spectrograph", "Show or hide the Spectrograph pane."),
        "menu.View.Event Log": ("Event Log", "Show or hide the Event Log pane."),
        "menu.View.BAT": ("BAT", "Show or hide the BAT (channel probe) pane."),
        "menu.View.Log": ("Log", "Show or hide the Log pane."),
        "menu.View.Layout": ("Layout", "Arrange the panes: stacked with tabs, all tabbed, stacked vertically, side by side "
                             "or a two-column grid."),
        "menu.View.Reset layout": ("Reset layout", "Put every pane back where it started."),
        "menu.View.Inspect UI names": ("Inspect UI names", "Hover any control for its name in the code; click to copy it "
                                       "(Ctrl+Shift+I)."),
        "menu.View.Setting help (click a setting)": ("Setting help", "Left-click mode for help on settings (Ctrl+Shift+H).  "
                                                     "Right-click works on any element without it."),
        "menu.Help": ("Help menu", "Native library inventory, the capture folder, a diagnostic bundle, About."),
        "menu.Help.C library inventory": ("C library inventory", "Which native DSP libraries loaded."),
        "menu.Help.Open capture folder": ("Open capture folder", "Open the folder logs and audio captures are written to."),
        "menu.Help.Save diagnostic bundle...": ("Save diagnostic bundle", "Zip the logs, settings and recent captures for a "
                                                "bug report."),
        "menu.Help.About NE0K Modem": ("About", "Version, licence and which DSP backend is in use."),
    }),
]
UI: dict[str, tuple[str, str]] = {k: v for _, tbl in UI_GROUPS for k, v in tbl.items()}


def anchor(key: str) -> str:
    """The HTML anchor of a help key (menu keys hold spaces and slashes)."""
    import re
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", key)


def lookup_menu(key: str) -> str:
    """The most specific menu topic: menu.View.Layout.Side by side -> menu.View.Layout."""
    parts = key.split(".")
    while len(parts) > 1:
        k = ".".join(parts)
        if k in UI:
            return k
        parts.pop()
    return ""


def title(key: str) -> str:
    if key in UI:
        return UI[key][0]
    t = topic(key)
    return t["label"] if t else key


def focus_html(key: str) -> str:
    """The clicked element's topic, shown above the full help page."""
    from html import escape as e
    if key in UI:
        name, text = UI[key]
        return (f"<div style='background:#fff8d0;color:#222;padding:6px;border:1px solid #d8c060'>"
                f"<h3>{e(name)}</h3><p>{e(text)}</p></div>")
    t = topic(key)
    if t is None:
        return ""
    return (f"<div style='background:#fff8d0;color:#222;padding:6px;border:1px solid #d8c060'>"
            f"<h3>{e(t['label'])} <small>[{e(key)}] - {e(t['section'])}</small></h3>"
            f"<p><b>What:</b> {e(t['what'] or t['label'])}<br><b>Impact:</b> {e(t['impact'])}<br>"
            f"<b>OFDM:</b> {e(t['ofdm'])}</p></div>")


def ui_html() -> str:
    from html import escape as e
    out = ["<h2>The window</h2>"]
    for group, tbl in UI_GROUPS:
        out.append(f"<h3>{e(group)}</h3>")
        seen: set[tuple[str, str]] = set()
        for k, (name, text) in tbl.items():
            if (name, text) in seen:                 # a panel and its dock share one entry
                out.append(f"<a name='{anchor(k)}'></a>")
                continue
            seen.add((name, text))
            out.append(f"<a name='{anchor(k)}'></a><p><b>{e(name)}</b> - {e(text)}</p>")
    return "".join(out)
