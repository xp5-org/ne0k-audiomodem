# NE0K Modem (modemgui)

Front end for the OFDM and CSS modems. The Python modems in `../tests` are the
reference; the C ports beside them (`../tests/{OFDM,CSS}/c`, `../tests/c/rs255.c`)
are what this project loads. Nothing is copied: `make` compiles them in place.

**Status: audio pipe + Linux/Docker environment only.** No GUI, no DSP wiring yet.

## Layout

```
modemgui/
  Makefile            builds build/lib{ofdm,ofdm32,cssmf,cssenc}.so from ../tests
  modemgui/
    native.py         ctypes loader: CPU check, entry-point manifest, ABI smoke test
    cli.py            `python -m modemgui ...`
    audio/
      base.py         Connector / Source / Sink / DeviceInfo / StreamSpec
      registry.py     built-ins + entry points + MODEMGUI_PLUGIN_PATH plugins
      pipe.py         AudioPipe: RX taps, history, TX queue, own-TX flag, meters
      pulse_backend.py       pulse:  sinks, sources, sink monitors (pactl/parec/pacat)
      sounddevice_backend.py sd:     PortAudio (ALSA / WASAPI / CoreAudio)
      file_backend.py        file: / loop: / null:  (no hardware)
  docker/  docker-compose.yml  tests/
```

## Audio model

Everything the modems see is **mono float32 at one fixed rate** (default 48 kHz).
A connector adapts whatever is underneath. Devices are addressed `connector:id`:

| address | meaning |
|---|---|
| `pulse:` / `pulse:<sink-or-source name>` | PulseAudio/PipeWire; empty = server default |
| `pulse:<sink>.monitor` | what a sink is playing — a software loopback, no hardware |
| `sd:<index>` / `sd:name=USB Audio` / `sd:` | PortAudio device (name match survives re-enumeration) |
| `file:<path.wav>[#fast,repeat]` | WAV source/sink; paced to real time unless `fast` |
| `loop:<name>` | in-memory bus: sink `loop:x` is heard by source `loop:x` |
| `null:` | silence in / discard out |

**Source** pushes blocks; **Sink** is pulled (`pull(n)` must return exactly n
samples). `AudioPipe` owns one of each and adds: per-consumer bounded taps (a slow
spectrograph drops *its* blocks, never the decoder's), a 30 s history addressed by
absolute sample index, a TX queue with `stop_tx()` for immediate silence, and a
per-block `tx_active` flag so analysis can discard the node's own transmissions
(~20 dB hotter than a peer's, and not a channel measurement).

### Adding an SDR / network connector

Subclass `Connector` (`modemgui/audio/base.py`), set `name`, and ship it as
either a setuptools entry point in group `modemgui.connectors`, or a `.py` file
in a directory on `MODEMGUI_PLUGIN_PATH` exposing `CONNECTORS = [YourClass]`.
A plugin that fails to import is listed in `Registry.errors` and skipped.

The contract is audio, not IQ: a PlutoSDR or network SDR connector owns the
radio-side (de)modulation and presents mono float32 at the pipe rate. Not built yet.

IQ passthrough to gnuradio or libsdr is on the todo list

## Run it (host, no Docker)

```
make                                   # C cores
python3 -m modemgui native             # load + ABI smoke
python3 -m modemgui devices
python3 -m modemgui monitor --rx pulse: --seconds 10
python3 -m modemgui loopback --tx pulse:SINK --rx pulse:SINK.monitor   # exit 0 = heard
python3 -m pytest -q tests
```

## Docker (Linux)

Build context is the **repo root** (the C sources live in `tests/`):

```
docker compose -f modemgui/docker-compose.yml build
docker compose -f modemgui/docker-compose.yml run --rm modemgui
```

Audio reaches the container one of three ways (details in the compose file):
**A** Pulse over TCP to the host (default, same as `../docker/README.md`; also the
only route from a macOS/Windows host), **B** the host's Pulse/PipeWire unix socket
bind-mounted in, **C** raw `/dev/snd` for the `sd:` connector.

The entrypoint runs an incremental `make` on every start, so the loop for a C
change is edit → restart container.

## GUI (PySide6) — framework state

`python -m modemgui gui [--rx ADDR --tx ADDR]` (needs `pip install -e .[gui]`).

```
modemgui/engine/   no Qt.  Session (owns AudioPipe + worker) -> events.py (Status, Levels,
                   Spectrum, ConstellationFrame) on a queue; spectro.py (streaming STFT);
                   selftest.py (loopback, shared with the CLI)
modemgui/ui/       PySide6 only.  bridge.py drains the queue on a QTimer and emits signals;
                   io_panel.py (device pickers, Open/Close, Test loopback, meters),
                   spectro_view.py (pyqtgraph waterfall), css_view / ofdm_view / tx_view / log_view (tabs),
                   probe_view.py (BAT bar graphs), settings_dialog.py (generated from the
                   config dataclasses), appsettings.py (QSettings persistence)
tests/test_layering.py   fails if anything outside ui/ imports Qt/pyqtgraph
tests/test_ui.py         offscreen pytest-qt smoke tests (skipped without PySide6)
```

Working now: device picker over every connector (unavailable ones explained under the
pickers), settings persisted by address *and* name, Open/Close, loopback test, peak/clip/TX
meters, live spectrogram with floor/ceil/max-Hz, C-library status in the status bar.
Stubbed: the OFDM constellation (`PLAN.md` phase 5).

### Settings, persistence and troubleshooting capture

* **BAT exchange chart** (OFDM tab): per-carrier bars, height = the SNR the probe saw, colour =
  bits given to the carrier (pilot/off/BPSK/QPSK/16/64-QAM).  The chart for what *this* end
  measured of the peer's probe appears as soon as that probe is decoded; the one for what the
  *peer* measured of our probe appears when its BAT report arrives (bits only: the wire format
  carries no SNR).  A mesh adds one chart per participant.
* **Settings > Settings...** (Ctrl+,): General (active modem), CSS, OFDM, Diagnostics.  The pages
  are generated from the field descriptions in `engine/modemcfg.py`, so a new field there is a new
  control and a new persisted value with no UI code.  The CSS page covers the chirp band, symbol
  and gap timing, framing, Reed-Solomon, interleaver and the rate-1/2 convolutional code; the OFDM
  page covers link/mode, band, handshake timers, the PHY group the original audiotestserver
  exposes (FFT size, cyclic prefix, pilot spacing, probe length and measurement symbols, SNR margin,
  max bits per carrier, SNR estimator, ISI de-rating), the window/stream/full-duplex timing and
  the DSP backend.  PHY values are pushed into the reference handshake's `HS_OVERRIDES`, i.e.
  process-wide, and both ends must use the same ones.
* **Persistence**: every setting is saved to QSettings when the dialog is accepted and at exit.
  The window, dock layout, selected tab, the TX draft and the overlay toggle are restored too.
  Switching tabs never touches them.
* **Diagnostics** menu: *Save debug log*, *Record received audio (WAV)*, *Record transmitted audio
  (WAV)*, *Open capture folder*, *Save diagnostic bundle...*.  Each session gets a folder under the
  capture folder (default `~/modemgui-diagnostics/session_<time>/`) with `debug.log` (versions,
  C-library status, settings, link events), `rx_*.wav` and `tx_*.wav`.  The TX file is padded with
  silence to the RX sample index at each play, so the two share one timeline.  The bundle is a zip
  of that folder plus `settings.json` and `c_libraries.json` -- the thing to send back to the dev
  team.  Close audio first so the WAV headers are complete.  The Log tab shows the same log live.
  The CLI has the same capture: `serial --debug-log --capture-wav both --diag-dir DIR`.

### Virtual serial port over OFDM (half duplex)

`python -m modemgui serial --modem ofdm --link /tmp/modem0 --rx ... --tx ... --node 1 --peer 2`
puts a pty (`/tmp/modem0`, control socket `/tmp/modem0.ctl`) on the OFDM modem
(`engine/ofdm_link.py`, Python reference modem, so `scipy` is required and `reedsolo` or the
built `tests/c/librs255` for Reed-Solomon). It works in two phases:

1. **BAT exchange** (`ofdm_handshake.HandshakeSM`): each end probes the other (5 s), grades the
   probe into a bit-allocation table for the *peer's* transmit path, and the tables are swapped and
   proved with CRCs (BAT_REPORT, BAT_ACK, GO). About 22 s on the default 400-4000 Hz band. The lower
   node id initiates (`--role` overrides). The modem keeps CTS low and raises `CD 1` when it is done.
2. **Fixed windows**: time is cut into `--window` seconds (default 5). The epoch is the end of GO;
   the initiator transmits in even windows, the responder in odd ones, each only between `--guard`
   seconds (default 1) after the start and before the end of its window. A frame is
   `[seq][ack][len][crc16] + data`; the ack is piggy-backed, a duplicate is re-ACKed but delivered
   once, and an unACKed frame is resent in the next own window up to `--retries` times, then `TXFAIL`.
   A window with nothing to say stays silent. Both ends must use the same window, guard, chunk and band.

The host side is the same as for CSS (`HandshakePort`; RTS/CTS plus the extra `CD` line).

#### Other data planes (`--ofdm-mode`)

Every mode starts with the same BAT exchange; only what follows differs. Defaults in `OfdmConfig`.

| `--ofdm-mode` | who transmits | ACK | notes |
|---|---|---|---|
| `half` | both, in alternating windows | per frame | the windows described above |
| `oneway` | only `--sender` (initiator by default) | none | frames of up to 250 B, `gap_s` of silence between them; a lost frame shows up as a seq gap (`lost_frames`). The listener's `send` / `TXFAIL` is refused. |
| `stream` | only `--sender` | none | one `send` is one continuous burst (`[len u32] + data + crc16`, chunks back to back); the listener decodes the growing burst every `stream_poll_s` and delivers bytes as they become contiguous (`on_stream(ok, n)` reports the end). Up to 65535 chunks (about 2 MB at 32 B). |
| `full` | both at once | piggybacked, sliding window | after the full-band BAT exchange each end keys a continuous carrier (chunks back to back, no keyup, no gap) and keeps it up; every chunk is one frame `[src\|dst, seq, ack, len, crc]` carrying data when due, else a filler, and always our current ACK. How the two directions share the line is the `duplex` setting, below. |
| `arq` | only `sender` | per frame | one-way with acknowledgement and no windows; the data plane of the mesh node. |

##### Full-duplex mechanism (`duplex`: `fdd` or `ec`)

Both ends must use the same `duplex` value (OFDM settings, group *Full duplex*); the difference is whether the
two directions share the spectrum:

| `duplex` | spectrum | how each end hears the other | use it for |
|---|---|---|---|
| `fdd` (default) | **split**: the band is cut around a `--duplex-guard` dead band (`ofdm_fdd.plan_duplex_bands`), initiator low, responder high | each half is its own carrier set with the full-band table re-indexed onto it (`_sub_bat`); the receiver filters its own half out (`ofdm_fdd.rx_filter`) | radio (a transmitter deafens its own receiver) and acoustic audio |
| `ec` | **whole band both ways**, V.34 style, each direction on the table measured for it | each end subtracts an echo canceller's estimate of its own signal (`engine/ofdm_ec.py`: bulk delay + `ec_taps` FIR, fitted by least squares on the echo of its own BAT probe while the peer listens) | SIP / dial-up / fax-modem lines (2-wire hybrids) |

Settings in the same group:

* `duplex_split` (fdd only): `bits` (default) places the cut so the responder's share of the *capacity* (bits per
  symbol from both BATs) is `dl_frac`, since a G.711 line carries little above ~3.4 kHz and a 50/50 cut in Hz
  leaves the high (responder) half short; `hz` cuts the bandwidth itself.
* `dl_frac` / `ratio`: the responder's share. In `fdd` it moves the band cut. In `ec` the PHY keeps its measured
  tables and the link enforces the share instead: the end over its share may start a data frame in only some of
  its chunks (a duty credit, the rest are fillers that still carry ACKs). At `dl_frac` 0.5 with no `ratio`, each
  direction runs whatever its line supports; `ratio` 50/50 holds the faster direction to the slower one's rate.
  (Trimming the BAT instead made back-to-back chunks undecodable 1 time in 3 on a clean loop, 2026-11-13.)
* `ec_taps` (ec only): canceller length after the bulk delay (256 = 5.3 ms at 48 kHz); `0` = no canceller, for a
  4-wire line that never returns our own audio. With no echo path found in training the canceller stays off and
  passes audio through, and the link log says so.
* `ec_window`: data frames in flight unacknowledged (LAPM's k; 1 = stop-and-wait). The receiver keeps frames past
  a gap, and only the oldest unacked frame has a resend timer (srtt + 4 rttvar, at most `ack_wait_s`).

The canceller's taps and delay are frozen after training. Tracking through data is not done yet: a plain re-fit
under double talk locks onto false lags, because both ends' chunks carry the same preambles and pilots.

##### The full-duplex engine in C (`--backend cdx`)

`--ofdm-mode full --backend cdx` (with `duplex` `fdd` only; `ec` falls back to the Python data plane, and the link logs that) runs the whole full-duplex data plane in C89 (`tests/OFDM/c`), loaded through
`engine/ofdm_cdx.py`; the BAT-exchange handshake still runs on the reference modem. The OFDM tab's state line shows
`backend: c engine (libofdm: ...)` so a run says which plane it used (the global status bar only counts loaded C libs).

| C module | does | Python it replaces |
|---|---|---|
| `ofdm_fdd` | band plan, frame sizing, sub-band BAT re-index | `ofdm_fdd.plan_duplex_bands`, `chunk_for_frame_len`, `_sub_bat` |
| `ofdm_dx` | link frame (`seq ack len crc16 data`) over `ofdm_mod` / sync / `ofdm_dem` / RS | `pack_frame`, `unpack_frame`, `_frame_audio`, `_decode_from` |
| `ofdm_rxf` | rejects this end's own band before acquisition | `ofdm_fdd.rx_filter` |
| `ofdm_rxs` | 50 ms RMS burst gate, decode once per burst | `_collect`, `_bursts`, `_full_rx` |
| `ofdm_arq` | stop-and-wait, piggybacked ACK, retries, TX gap | `_on_frame`, `_send_full`, `_tx_frame_full` |

Two deliberate differences from the reference, both because a streaming receiver cannot do what the
reference does:

* **The receive filter is causal.** The reference is an elliptic filter run forward and backward
  (`sosfiltfilt`, zero phase), which needs the whole capture. `ofdm_rxf` is a Kaiser-window linear-phase FIR
  designed in C from the same four band edges (about 150-175 taps for a 1200 Hz guard at 48 kHz, 60 dB). Its
  delay is a constant `(taps-1)/2` samples.
* **Acquisition retries with stricter thresholds.** After the filter the noise is band-limited to the far
  sub-band, where the Schmidl-Cox metric of pure noise is much higher than the wideband tuning expects (3-6
  candidates against 1-2 real ones, and a spurious one can suppress a real frame). `ofdm_dx_rx_frame` tries
  0.25/NCC 0.10, then 0.40/0.20, then 0.60/0.35, and stops at the first pass that yields a complete frame.

Graded by `tests/regression/ofdm_fdd_regress.py`, `ofdm_dx_regress.py` and `ofdm_duplex_regress.py` (C against
Python, then two complete C ends talking at once over a simulated shared channel with own-TX leakage and
dropouts) and by the `Full-duplex C engine` testlist. Not done: the C handshake on this path, DOS/OpenWatcom
build of the new modules, and anything over the air.

#### Mesh (`--modem mesh --dst N`)

`engine/mesh_node.py`. One `MeshNode` per radio on a shared medium. `send(dst, data)` is
`src -> dst`, one way: a CSS `OFDM_HS_REQ` and its CSS ACK (everyone else that overhears the REQ stays
quiet for `hold_s`), then the BAT exchange and an `arq` OFDM link between the two, then the data. If the REQ
goes unanswered, the handshake fails or the data is not ACKed, the node resends over CSS (stop-and-wait with a
CSS ACK). OFDM links idle for `link_idle_s` are closed; there is no timed re-upgrade. The serial bridge for
`--modem mesh` sends everything written to the pty to node `--dst` and writes whatever any node sends to it
(`--mesh-phy css_only` skips OFDM). Tests use `mix:<net>/<node>`, a many-node in-memory medium whose sources
hear the sum of all other nodes' sinks.

Not ported from `meshmodem_node.py`: the conv-coded 48-byte mesh frame wrapper (data rides the same OFDM
frames as the point-to-point modes) and the multi-host test rig. Not done anywhere: band renegotiation
(`MSG_BAND` is supported by the state machine but off), the C engine for modes other than `full`, a GUI tab for the
link, over-the-air runs (everything was exercised on loop/mix buses and PulseAudio null sinks).

#### Hub & spoke (General > Topology `hub` / `hub_relay`, ofdm_only)

`engine/hub_mac.py`, in place of the mesh announce round. One leader, `hub_id`, runs the channel; every other
station answers it and nothing else. A client is configured with `hub_id` and `hub_nodes` (stations, hub
included) and is never sent a list of station ids. There is no failover: when the hub is gone the network is down.

Each hub cycle: BEACON (join open while fewer than `hub_nodes - 1` clients are registered; clients JOIN in an
own-id slot) -> per new client TRAIN + one pairwise handshake (both directions trained at once) -> BTAB, the
broadcast table (per carrier the least bits of all hub -> client tables) -> the hub's turn (its pages, and under
`hub_relay` the pages it stored for forwarding, each followed by a DLVR report to the source) -> POLL each client,
which sends its page and then DONE. A broadcast frame is ACKed by every client (BACK, own-id slot) and resent to the
missing ones; a client that still lacks it gets the page by unicast. Three silent polls drop a client and the join
window reopens.

The serial port in a hub topology is a page port, like a fax machine: one buffer is one page, RTS sends it, CTS stays
low until it was delivered (TXDONE) or failed (TXFAIL), and nothing queues behind it. A relayed page reports TXDONE
only once the hub has delivered it. With `serial.page_addr = ascii` a page may start with `@<id>:` (one hex digit)
or `@*:` (every station) and a received page starts with `@<origin>:`; with `off` a client's pages go to the hub and
the hub's to every station. Test: `tests/test_hub.py` (3 nodes on `mix:`), group tests `hub_sip_3node_group` and
`hub_relay_sip_3node_group`.

### TCP control port (remote control and debug)

`engine/control.py` (+ `ui/control_host.py` for the desktop app). A line-based TCP port that lets another
host drive the whole modem. It covers everything the virtual serial port does, reading and setting every
settings key, the front-panel actions, and a status dump. It is **off unless you ask for it**:

```bash
python3 -m modemgui gui --control-port 7322                            # 127.0.0.1 only
python3 -m modemgui gui --control-port 7322 --control-bind 0.0.0.0     # every interface: remote hosts
python3 -m modemgui serial --link ~/modem0 ... --control-port 7322     # headless port; settings read-only
```

**The port has no authentication and no encryption.** Anyone who can reach it can key the transmitter,
change the station's settings and read the received data. It binds to 127.0.0.1 unless `--control-bind`
names another address. Open it to the network only on a trusted test LAN, or keep it on localhost and
tunnel (`ssh -L 7322:127.0.0.1:7322 laptop`).

From the remote host, `nc laptop 7322` is enough. Every command gets zero or more data lines, then one
`OK [detail]` or `ERR <why>`. The greeting is `OK ne0k-control 1 <version>`. Up to 4 clients can connect at
once; one more is told `ERR busy: 4 control clients already connected` and disconnected.

| command | what it does |
|---|---|
| `RTS 0\|1`, `STATUS` | the serial port's control lines. Same handler as the pty's `.ctl` socket, so the same lines come back (`CTS n`, `CD n`, `BUF n/N`, `ERR empty`, `ERR busy`), plus the closing `OK`/`ERR`. RTS is one line shared with `.ctl` clients |
| `WRITE <text>`, `WRITEHEX <hex>` | bytes into the TX buffer, as a tty write: only while CTS is up, only up to the free room. Answers `OK accepted/offered` |
| `SEND <text>` / `SENDWAIT <text>` | load the whole text and trigger it (RTS edge in `rts` mode). `SENDWAIT` answers after `TXDONE`, or `ERR TXFAIL` |
| `READ [n]` / `READTEXT [n]` | received bytes since the last read, as `DATA <hex>` / `TEXT <escaped>`. The pty still gets them too. Up to 1 MiB is kept, shared by all clients |
| `EVENTS 1` | stream the `.ctl` event lines (`CTS`, `CD`, `BUF`, `RX n`, `TXDONE`, `TXFAIL`, `RETRY`, `ACK`), plus `PORT 1 <path>` / `PORT 0` when the serial port opens or closes. They arrive whenever they happen, so one can land between a command and its `OK`; tell them apart by their first word |
| `SECTIONS`, `LIST [section]`, `GET s.key`, `DESCRIBE s.key` | the settings sections (`general`, `css`, `ofdm`, `serial`, `ham`, `diag`, ...: the `modemcfg` dataclasses, the same keys a `--settings` file holds). `LIST` prints `s.key=value<TAB>type=..<TAB>apply=live\|restart` plus `lo=`/`hi=`/`choices=` |
| `SET s.key value` | checked against the field's type, range and choices. Answers `OK applied` (live), `OK pending-restart` (the modem is on, so the running link keeps its old copy until `RESTART`) or `OK next-start` (the modem is off). Not written to disk until `SAVE` |
| `INFO`, `LOG [n]` | `key=value` status dump (audio, modem, link states, neighbours, serial counters, status bar text); the last n log lines |
| `AUDIO OPEN <rx> [<tx>]`, `AUDIO CLOSE`, `DEVICES` | gui: open the sound devices (same addresses as `--rx/--tx`) |
| `POWER ON\|OFF`, `RESTART` | gui: the Start modem button. `RESTART` power-cycles so pending settings apply, and reopens the serial port if it was open |
| `PORT OPEN\|CLOSE`, `IDENT`, `TEXT <msg>`, `SAVE` | gui: the virtual serial port, send ID now, one message on the mesh link, persist settings |
| `HELP`, `QUIT` | |

Live sections are `ham` and `diag`. They apply at once, as the settings dialog applies them while the
modem runs. Every other section is read when the modem starts. Under `modemgui serial` the link is built
once from the command line, so `SET` is refused there (restart the process with new options). An
unattended laptop run looks like this:

```
AUDIO OPEN sd:3 sd:3                 (DEVICES lists the addresses; no spaces in an address)
SET general.node_id 3
SET serial.mode auto
POWER ON
PORT OPEN
EVENTS 1
SENDWAIT hello from the test host
READTEXT
```

Docker (`docker/Dockerfile` targets `engine` → `gui`):

```bash
docker compose -f modemgui/docker-compose.yml --profile gui run --rm gui-test   # offscreen tests
docker compose -f modemgui/docker-compose.yml --profile gui run --rm gui        # window on $DISPLAY (X11/xrdp)
```

Moving this tree elsewhere: it needs the C sources at `$(REF)` (default `../audiotestserver/tests`,
containing `c/rs255.c`, `OFDM/c`, `CSS/c`); override with `make REF=/path/to/tests`, or set
`MODEMGUI_LIB_DIR` to prebuilt libraries.

## Not done yet

- The Docker image (both stages) has **not been built** (no Docker in the dev sandbox);
  the apt/pip steps were exercised by hand on Debian 13, the Dockerfile itself was not.
  The GUI was run only under `QT_QPA_PLATFORM=offscreen`, not against xrdp/X11 or a real
  display, and the `gui` compose service (X11 socket, XAUTHORITY, Pulse socket) is untested.
- Pulse and PortAudio were tested only against a local null-sink server
- Windows/macOS: `sd:` is the intended path (WASAPI/CoreAudio via PortAudio); untested.
  The `Makefile` handles `.dylib`/`.dll` extensions but a Windows toolchain is unproven.
- Resampling between a device's native rate and the pipe rate is left to the OS/Pulse.
- OFDM constellation viewer (decision A in `PLAN.md`),
  Pluto/Pico connectors, PyInstaller packaging.
- Spectrograph has no scroll-back, own-TX shading or time axis yet (the data is in
  `Spectrum.tx_active` and `AudioPipe.history`).

## Ham mode (station identification)

Off by default.  **Ham > Ident settings** (or Settings > Ham ID) sets the callsign and two timers; **Ham > Send ID now**
(Ctrl+I) sends one on demand.  The CLI `serial` command takes `--ident CALL --id-mode cw|ascii|both
--id-interval S --id-end-delay S`.

| Setting | Meaning |
|---|---|
| Timer 1, *ID at least every* | Checked on every outgoing frame; the first frame of a session counts. Default 600 s. |
| Timer 2, *ID after the channel is quiet for* | End-of-transmission ID, sent alone once the last frame is that old and an ID is owed. 0 = off. Default 5 s. |
| Mode `cw` | Morse (`DE <CALL>`, wpm and tone settable) *after* the frame, `cw_gap_s` of silence first. |
| Mode `ascii` | Binary-coded ASCII in *front* of the framing: AFSK 2200/1200 Hz, 8N1 LSB-first, `SYN SYN STX "DE <CALL>" ETX`, then `ascii_gap_s` of silence. 110/300/600 baud. |
| Mode `both` | ASCII in front, CW after. |

Both gaps are longer than the receivers' burst-end quiet time on purpose: the peer sees the ID as a burst of its own,
recognises it (CW or ASCII; logged as `ident heard`, shown as an overlay event) and does not count it as a decode error,
and the data frame decodes exactly as without ham mode.

OFDM half duplex has no slack for in-line audio, so a due ID takes one of our TX windows instead of the frame (the frame
goes out in the next window).  CW longer than the window falls back to ASCII; if neither fits, only the end-of-transmission
timer can send it (lengthen `window_s`).  The handshake probe/BAT exchange is never wrapped.

The operator is responsible for meeting the ID rules of their licence (callsign format, interval, permitted ID
methods; some services require the ID code to be publicly documented, which is the paragraph above).
