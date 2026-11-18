# modemgui — Qt (PySide6) plan

## Principle: the engine never imports Qt

```
modemgui/
  audio/ native.py        (done)  connectors, AudioPipe, C loaders
  engine/                 (new)   headless: STFT, modem sessions, events. No Qt.
  ui/                     (new)   PySide6 only. Talks to engine via signals/queue.
```

The engine publishes plain data (numpy arrays + dicts) on queues; `ui/` adapts them
to Qt signals in QThread workers. Consequences: engine tests run in Docker/CI with no
display, and a remote engine (below) is a transport swap, not a rewrite.

## Docker and a desktop app

Docker is the wrong *runtime* for a desktop audio/USB app on Windows/macOS (Docker runs
in a VM: no USB passthrough for a Pico/Pluto, audio only via Pulse-over-TCP, X11 is
painful). It is the right *build/test/CI* environment, and a workable Linux run target.

| use | how |
|---|---|
| CI / tests | image + `QT_QPA_PLATFORM=offscreen`, pytest-qt, engine tests; grab widgets to PNG for regression |
| Linux dev run | compose profile `gui`: mount `/tmp/.X11-unix` + `$XAUTHORITY` (or the Wayland socket), `DISPLAY`; audio via existing Pulse socket/TCP; USB via `devices:` |
| Linux build | PyInstaller/AppImage built in the image (old glibc base = portable binary) |
| Windows / macOS | run natively (PyInstaller bundle); C libs built in a CI matrix, not Docker |
| optional later | engine in container, GUI on host, joined by a local websocket (msgpack frames). Only worth it for a headless Pi/remote node; not in v1 |

Image additions for the `gui` stage: `libegl1 libxkbcommon-x11-0 libxcb-cursor0 libdbus-1-3 libgl1`, pip `PySide6 pyqtgraph pytest-qt` (kept out of the base stage so the headless image stays small).

## Status

Framework (phase 1) and the generic I/O panel + spectrograph (phases 2-3, minus scroll-back/TX shading) are built; see README "GUI (PySide6)".

## Phases

1. **Foundation** — `engine/` + `ui/` split, `gui` Docker stage, offscreen smoke test that opens the main window. Settings via `QSettings` (store the connector address *and* name; re-resolve by name, as `sd:name=` already does).
2. **I/O panel** — device picker over `Registry`: tree by connector, sinks/sources/monitors, unavailable connectors shown greyed with the reason, Refresh, a "Test" button running the existing loopback check, live peak/clip meters (uses `Block.peak/clipped/tx_active`; attribute clipping to own-TX vs input).
3. **Spectrograph (generic)** — STFT worker on a `Tap`, ring of columns, pyqtgraph `ImageItem`, dB floor/ceiling/max-Hz/scroll from `spectrograph.json`, scroll-back from `AudioPipe.history()`, own-TX shading. This alone serves OFDM's spectrograph requirement.
4. **CSS view** — port `css_spectrographtool.py` overlays and decoder meters. It currently reads decoder state over UDP/files (`load_css_info`, `_udp_receiver`); replace that with engine events so there is one path. Largest port (~1500 lines); do overlays first, meters second.
5. **OFDM constellation-per-symbol** — see open decision A.
6. **TX / modem control** — message entry, probe → BAT → data flow, status. Needs the Python fallback for probe/BAT (not in C yet); engine picks C where a function exists, else `tests/OFDM/ofdm_modem.py`, and says which in the UI.
7. **Connectors** — PlutoSDR (network/USB via libiio; owns modulation, presents audio) and Pico USB (serial/bulk). Plugins, so they ship separately.
8. **Packaging** — PyInstaller per OS from a CI matrix.

## Open decisions

**A. Where do constellation points come from?** `ofdm_dem_frame` returns descrambled bytes and one mean EVM — the equalised points are discarded. Options: (1) add an opt-in output buffer to `ofdm_dem.c` (touches the embedded-shared C; must keep byte-exact regress and stay C89); (2) re-run the reference demod in Python on frames the C already located (slow, but zero C risk, and `constellation.py` already does it post-hoc). Constraints from past bugs: EVM must be gain-weighted; plot only the sync candidate that was reported, not every attempt. Recommendation: (2) first to get the viewer working, (1) later if CPU matters.

**B. Engine placement.** In-process (QThreads) for v1; remote engine deferred. Recommended.

**C. Plotting.** pyqtgraph for spectrogram + constellation heatmap (fast `ImageItem`, PySide6-supported); custom QPainter only if it falls short. Recommended.

**D. Python fallback policy.** Until probe/BAT are in C, the app is not "C only". Acceptable for the GUI; label the active backend in the status bar.

## Design defects (TODO)

- **Carrier sense treats sound level as "channel busy" (2026-11-17).** On an acoustic link (speaker + mic) room noise
  always reads as someone transmitting, so a station never keys up: Mac <-> m6600 1 ft apart, `ofdm_only` p2p, both
  sides logged `no handshake slot with <peer> within 60s`. The only escape is `general.ignore_busy`, which turns
  carrier sense off entirely and lets two stations key over each other. Sound level is not an indicator of a busy
  carrier: busy must mean *our* signal is on the air -- a CSS preamble/chirp or OFDM sync/probe detected by
  correlation (`mesh_rx.channel_busy` energy gate, `css.channel_busy`, `meshlib.detect_ofdm_activity` thresholds
  vs `ambient.rms`), not energy above a floor. Affects `engine/mesh_rx.py`, `engine/css_link.py` carrier sense,
  `engine/mesh_node.py::_channel_busy` / `_hs_gate`, the announce round and OFDM data-frame gating.
  The RX side has the same defect: the `mesh_rx` burst gate is a fixed absolute RMS threshold (`ofdm.carrier_db`,
  default -55 dBFS) and the laptops' room floor is -53..-57 dBFS, so the gate never closes, a "burst" lasts the whole
  session (`bursts=1` over 6 minutes) and no handshake probe is ever decoded -- even with `ignore_busy` on. Until it
  is fixed, the calibration tool should at least set `carrier_db` from the measured floor (floor + ~6 dB).
  **Addressed 2026-11-17 (not yet tried on the air):** `general.carrier_detect = signal` (`engine/cca.py`,
  `tests/test_cca.py`).  A small always-on detector marks each 50 ms block as "ours" or not: OFDM by cyclic-prefix
  autocorrelation at lag n_fft, CSS by dechirp + FFT (LoRa-style CAD), key-up tone and CW ID by a Goertzel bin.
  Every burst gate and pre-TX check that compared RMS with `carrier_db` now asks the capture (`_Capture.busy_blocks`
  / `busy`), so `level` (default, wire lines) and `signal` switch them all together.  Synthetic results: OFDM found
  at 0 dB in-band SNR, CSS at -6 dB (0.05 s symbols) / -10 dB (0.15 s), no false flags on pink noise, tones or
  clicks; under 1 % of a core.  Real room noise (fan lines 30-49 dB above the rest at 300-800 Hz) false-triggered
  both detectors on the laptops, so the detector input is pre-whitened: a 1025-tap FIR (band-pass x 1/sqrt(noise
  PSD)), PSD = per-bin median over ~32 s, learnt only from unflagged blocks (`tests/data/room_*_48k.wav`).
  The CSS streaming PLL decoder + tracker are now fed only while a burst is open, for 2 s after own TX and for the
  first 5 s (`css_link._feed_streams`); idle blocks are skipped with `css_pll_skip` / `css_trk_skip`, which keep
  the timebase.  Measured idle (builds 83/84, signal, css): m6600 total 90 % -> 56 % (pll 44 % -> 0), Mac pll 20 %
  -> 0; the UI MainThread (24-35 %) is now the largest cost and is not addressed.  Note `PROFILE LOAD` of a profile
  saved before this setting existed resets `carrier_detect` to `level`.  Open: (2) each capture (mesh receiver, every OfdmLink, the announce round) runs its
  own detector: share one per session if the CPU shows; (3) a station on another FFT size / prefix ratio / chirp
  band is invisible to it (level sense heard anything); (4) speech-like pitched sounds can read as chirps or OFDM
  for a moment: that costs a decode attempt or a short deferral, never a frame.
- **ofdm_only p2p two-way does not work over the air (2026-11-17, Mac <-> m6600 1 ft).** With `ignore_busy`,
  `carrier_db -47`, `ofdm.role auto`, `auto_link true` the link came up (m6600->Mac 59 carriers ~2470 bps,
  Mac->m6600 12 carriers ~406 bps), but: (1) 4 DATA frames m6600->Mac were never decoded or ACKed on the Mac and
  the link closed; (2) once the Mac had its own message to send, both nodes initiated at once: each heard the
  other's probe as if it were the reply, sent RETRY and waited for a BAT that never came (`timeout_i_wait_bat` on
  one side, `timeout_i_wait_probe` on the other), so it never recovered. Responders are only armed by the
  auto-link keeper or an overheard BLOCKOUT_REQ, and both nodes had `ofdm.role=initiator`, so with `auto_link`
  off nobody ever listens.
  **Re-run 2026-11-17, builds 85/86, carrier_detect signal, ignore_busy OFF:** m6600->Mac 3/3 (28-62 s, link
  43/112 carriers), Mac->m6600 0/3: each direction is its own link and `_keeper` arms a responder only on the
  higher id, so the Mac's probes (3 per send) reach a node that is not listening (`timeout_i_wait_probe`).
  Arming one on m6600 too (`serial.rx_host 2`) makes it worse: its rx and tx links to the same peer take each
  other's handshake messages (`i_wait_bat`, `r_wait_ack`), 0/1.  Needs one link per peer pair that carries data
  both ways (or a responder that hands the turn back), not more links.  CSS p2p on the same builds: 6/6, every
  DATA first try, ACK ~1 s after decode (after the staged-receive stale-busy fix in `css_link.channel_busy`).
  **Fixed 2026-11-17 (builds 87/88):** mesh arq links are connected-mode (`sender="both"`): either end sends DATA
  on the one link (`MeshNode._up_link`); in ofdm_only + auto_link the higher id never initiates, it waits for the
  keeper's link (`_await_rx_link`); `_send_arq` listens before talking and backs a resend off at random.
  `tests/test_p2p_arq_both_ways.py` (fails on the old code with the laptops' `timeout_i_wait_bat`).  On the air:
  n_fft 1024/cp 0.5 got 5/6 then 2/4 (the Mac probes report `isi_ceiling 7.9` dB: room echo caps m6600->Mac);
  **n_fft 4096 / cp 0.375 (the pair calibration's recommendation): 6/6**, Mac->m6600 ~15 s, m6600->Mac ~55 s.
  Open: (1) in the 4096 runs the Mac missed the first two copies of m6600's DATA in most rounds though its captures
  of all copies agree to 23-28 dB (same audio in, different outcome).  The MacBook mic noise gate lifts the floor
  ~10 dB (-64 -> -54 dBFS) for ~40 s after any loud burst, and in that state the whitened detector false-flags ~20 %
  of blocks offline; the copy that decodes lands ~60 s after the Mac's own TX, the whitening stale-fallback time.
  **Refuted 2026-11-17 (builds 91/92, 4/6):** the Mac's per-burst log shows every DATA/ACK copy detected and handed
  to the decoder at full length (2.95-3.65 s bursts, 1.6-3.2 s decode) and `nothing decoded`; 0 input overflows.  So
  it is the OFDM decode, not detection.  The Mac's probe of m6600 swings from 73 carriers / median SNR 3.7 dB to 528 /
  19.5 dB between handshakes minutes apart, and even with the 528-carrier BAT no DATA copy decoded.  Sample rates:
  Mac mic 48 kHz native, Mac speakers 44.1 kHz (CoreAudio resamples the 48 kHz TX); m6600 Pulse 48 kHz both ways.
  Baseline rerun with `carrier_detect level` (no detect/whitening/gating): 3/6, same pattern -- Mac detects every
  m6600 copy, decodes almost none; probes swing both ways (Mac hears m6600 at 60 -> 512 carriers, m6600 hears the
  Mac at 13 -> 36 carriers, vs 464-480 an hour earlier).  So the power-saving sense is not the cause.
  FFT sweep, level sense, cp 0.5, 3 rounds each: **n_fft 512: 6/6** (one handshake, 48/28 carriers of 80,
  sends 8-16 s after the first), 1024: 3/6 (Mac probes 26-33 of 161 carriers, isi_ceiling ~7 dB), 4096/0.375: 3/6.
  Shorter symbols win here although the echo budget says the opposite -- points at something that grows with
  symbol length (sample-clock offset / drift, the 44.1 kHz resampler, a time-varying channel), not echo.
  Capture now exists for this: `diag.wav_rx` / `wav_tx` (already there) + new `diag.event_db` (`engine/eventdb.py`,
  SQLite: every log line and engine event with its rx-WAV sample position); INFO `diag.dir`.
  Next: capture failing bursts on the Mac and decode offline, measuring sample-clock offset and per-symbol SNR;
  and rerun with the Mac speakers at 48 kHz (Audio MIDI Setup) to take the CoreAudio resampler out.
  Rerun 2026-11-17 on Linux 93 / macOS 94 with **no PL lead tone** (`ofdm.channel line`), level sense, auto_link,
  capture on, 3 rounds each: **512/0.5: 6/6, 1024/0.5: 6/6, 4096/0.375: 5/6** (the miss: round 1, the m6600 copy
  arrived but its ACK did not, then the Mac's send failed).  Sends 3.5-6 s once the link is up.  With the 2 s
  1000 Hz lead the same pair gave 3/6 at 1024 and 4096 -- the lead tone (or what it does to the receive gate /
  probe scan, cf. the 1 kHz keyup false-lock) is now the prime suspect for the long-FFT losses, ahead of clock
  drift.  Captures: m6600 `~/modemgui-diagnostics/session_20261117_083428`, Mac `session_20261117_090034`.
  (The macOS CI smoke failures were the `loop:` bus zero-padding a stalled writer: fixed in `file_backend._Bus`,
  `test_loop_bus_waits_out_a_stalled_writer_instead_of_splicing_silence`.  DEVICES now prints native rates; INFO
  shows `audio.rx.overflows`.)  (2) a failed
  send on the responder closes its end while the initiator still holds the link up: the initiator's next send
  burns its retries before it re-handshakes.  (3) `test_mesh` bystander/never_beacons and `test_hub` relay fail on
  HEAD as well; `test_mesh_bat` broadcast-round failed once with the change, passed twice on rerun.
- **Calibration missed the weak Mac -> m6600 direction.** Pair calibration showed SNR ~28 dB at -12 dBFS, but the
  modem's own handshake probe measured median 8.2 dB (12 carriers) at amplitude 0.25 (-12 dBFS); a later probe
  reached 15.5 dB. The pair step has to measure what the modem's probe/BAT will see (same signal, same
  amplitudes, same estimator) and flag a direction whose expected carrier count / rate is low, instead of
  stopping at speaker compression (the -6 dB mask that set the Mac to 0.25).
- **DEFECT (2026-11-17): calibration passes a pair that cannot hold an OFDM link.**  Release users (non-technical)
  only have the calibration steps/tool -- built into ne0k modem or an external Python GUI button -- so whatever the
  modem will fail at, calibration must catch and say so.  Mac <-> m6600 1 ft: calibration recommended n_fft 4096 /
  cp 0.375 and called the pair fine, then on the air the modem's own handshake probes swung from 464-528 usable
  carriers (median SNR ~19 dB) to 13-73 carriers (median ~3-5 dB) between handshakes minutes apart, in both
  directions, and m6600->Mac DATA frames were detected at full length but not decoded (3/6 and 4/6 runs, with and
  without signal carrier sense).  `engine/calibrate.py` measures one probe at one moment with its own multitone and
  estimator, so it can neither see the swing nor what the BAT probe will see.  Calibration (the 2-node pair step at
  least; a self-loop through own speaker/mic may not reproduce a 2-machine problem such as clock offset) must:
  run the modem's real handshake probe + BAT both ways at the recommended n_fft/cp, several times over a minute or
  more, and report min / median usable carriers and SNR; send real DATA frames both ways at that BAT and count
  decodes; fail the pair (in plain words) when the worst probe or the frame decode rate is below what the link needs;
  report each device's native sample rate vs the modem's (the Mac speakers run at 44.1 kHz, CoreAudio resamples
  the 48 kHz TX) and the measured clock offset; and try the other n_fft sizes when the recommended one fails.
- **FIXED (2026-11-17): plain CSS sent OFDM_HS_REQ.**  With `general.auto_link` on (left from an OFDM run), modem
  `css` (phy_mode css_only) ran the link keeper, which opened an OFDM tx link and put a CSS OFDM_HS_REQ (13 s at
  0.15 s symbols, 43 s at 0.5 s) on the air before sends.  CSS -> OFDM promotion is a planned mode, not built:
  css_only now never opens an OFDM tx link (`MeshNode._keeper`, `_open_tx_link`);
  `tests/test_css_only_no_ofdm_req.py`.
- **FIXED (2026-11-17): CSS ACK wait shorter than the long-symbol decode.**  At 0.5 s symbols both legs (1 round,
  16 B, 191 s DATA, 43 s ACK) were delivered but each sender gave up 1 s / 3 s before the ACK finished decoding
  (decode ~10 s after a 43 s ACK, ~11 s after a 191 s DATA on m6600) and re-sent the whole 191 s DATA; SENDWAIT hit
  its 600 s cap.  `_wait_ack` now stretches the deadline while the CSS receiver is busy (receiving or decoding),
  capped at twice the wait.  CSS 0.15 s (before both fixes): 4/4 delivered, one send took 465 s (lost ACK, 76 s
  DATA resent); after them (Linux 97 / macOS 98, auto_link on, no REQ): 4/4, 98-110 s each, no resends.
- **FIXED (2026-11-17): noise blips froze CSS transmit at long symbols.**  m6600's 40 s-late ACK: at 0.5 s symbols
  every room-noise blip that crossed the -55 dBFS level gate (0.2-11 s) went through four matched-filter decode
  passes on the css-rx thread -- 23 s for a 0.2 s blip on the manager, ~38-40 s on m6600 -- and `channel_busy()`
  stayed True for the whole decode (`_in_burst` is only reset by the next block).  ACKs and DATA key only on a
  clear channel, so they waited: on builds 97/98 m6600's DATA keyed 170 s after the send, and its ACK of the Mac's
  DATA never went out in 5 min (back-to-back junk decodes, one every ~40 s; the Mac every ~13 s); the GUI's RX CPU
  meter cycled 99 % / 0 % on both laptops.  The 22 s RX tap queue also overflowed during such a decode.  Fix:
  `CssLink._decode` skips a burst shorter than half the shortest frame (`MIN_BURST_FRAC` x the header-only ACK:
  6.5 s at 0.15 s symbols, 21.6 s at 0.5 s; `short_bursts` counts them, an `err` LinkEvent records each);
  `tests/test_css_burst_gate_and_taper.py`.  Still open: a long junk burst (> half an ACK) still decodes on the
  RX thread with the channel busy; decoding off that thread is the real cure.
  Verified on Linux 99 / macOS 100 (gate + taper, auto_link on): 0.5 s 2/2 and 0.15 s 4/4 delivered, no junk
  decodes, no resends, every ACK keyed ~1 s (0.5 s) / 2-6 s (0.15 s) after its DATA decoded; mac->m6600 at 0.5 s
  250 s = pure airtime + decode.
- **FIXED (2026-11-17, builds 103/104): CSS level carrier sense read the room as busy.**  The INFO rows below
  (`css.cs`, `mac.csma`, `mac.hold_left`, `ui.pll`, builds 101/102) showed it at once: `in_burst=1`,
  `last_busy_age=0.0s` on both laptops from power-on with the tracker idle (`trk=0`).  `css.carrier_db` -55 dBFS
  fixed vs a room floor of -57/-56 dBFS and -51 dBFS RMS: every 50 ms window was "busy", DATA never got DIFS,
  and a send waited for a lull (nothing to do with power-on; round two was just quieter).  The burst never closed
  either, so the level buffer grew without bound.  `engine/levelfloor.py`: threshold = max(carrier_db, room floor
  + `css.carrier_margin_db` 10 dB), floor = 10th percentile of 5 s of window RMS, falls at once, rises 0.5 dB/s
  (0.02 dB/s inside a burst, so a 450 s frame cannot cut itself off).  Weak signals still hold the channel via
  the chirp tracker.  OFDM's `carrier_db` (mesh_rx) is the same design defect and is not changed yet.
- **FIXED (2026-11-17, decoder py + C): the Mac's repeat CSS decode failures.**  Mac sync scores were bimodal:
  1.00 (good) vs 0.46-0.70 (25 of 68 syncs; m6600 had 2 of 52), the weak ones -> FRAME_ERROR / RETRO_GATE with
  every symbol at 29 dB.  They came 1.2-2.9 s after the peer keyed, the good ones 5.6 s: early locks on the
  training run-in (partial-preamble windows), read at the wrong offset.  The onset guard that blocks them anchors
  on "signal after >= 2 silent symbols" and "silent" was rms < silence_threshold 0.002 (-54 dBFS) absolute; the
  MacBook mic floor runs -64..-44 dBFS (noise gate +15 dB once open), so no symbol was silent and the guard
  never anchored.  Fix: `onset_rel_db` 6 (silent also = under the pre-sync room floor + 6 dB; floor falls at once,
  rises 0.2 dB/sym on quiet symbols, 0.02 while loud), `onset_sustain_syms` 2 loud symbols in a row, anchor HARD
  (a soft anchor trusts partial-preamble windows: tried, made 0.48 locks, net worse).  Corpus (tools/
  corpus_replay.py, 80 vCPU): Mac-room mixes 81 -> 91 packets, frame errors 16 -> 6, weak syncs 33 -> 9; group
  runs 567 units identical; laptops +1; 0 regressions.  Parity case pll_015_room_lead; fixture test
  tests/test_css_onset_room.py (real capture + Mac room audio).  Today's laptop sessions (fetched with DIAG,
  404 units, 0.15 s): 49 -> 50 packets, 0 regressions, the one changed unit a real Mac 0.49/0.53 lock now 1.00.
  Live on builds 105/106: 10/10 sends first try (56-81 s), m6600 0 weak syncs, Mac 2 -- both right after a burst
  ended (room tail, "carrier idle"), no packet lost.  OPEN: those post-burst tail locks (would blind the decoder
  ~14 s if an ACK followed at once); 61 weak syncs remain across the sessions; 0.5 s (MF path) not in the corpus
  replay yet.
- **Corpus replay** (`tools/corpus_replay.py`): real RX recordings (laptop diag sessions, group-run per-member
  rx.wav, CAPTURE grabs) cut into burst units and decoded in a process pool under decode() variants; judges a
  receiver change by packets / frame errors / weak syncs and lists every unit that differs.  Corpus lives in
  test_objoutput/css_corpus/ (group_runs = symlinks to meshsip output, laptops_*, mac_noise_mix).  New control
  command `DIAG LIST | GET` pulls laptop session files; each modem start now writes settings_<time>.json (+ rx
  sample index) beside rx_*.wav so sessions decode per segment.  Seen in passing: group-run CSS units show 330
  frame errors / 398 weak syncs against 319 packets -- worth its own look.
- **FIXED (2026-11-17): PLL bar LOCKED for minutes on an idle Mac.**  Relit by the CSS decoder's preamble sync on
  room noise (score 0.50), tracker grid locks on own-TX echoes, and CW-ident guesses on noise (`rx`, peer -1), 8 s
  per relight.  CSS sync / grid lock are now LinkEvent `sync` (pianoroll shows them; css_bcast still takes them
  for "PLL 1"); the bar locks only on a decoded frame, a peer rx or an OFDM lock.
- **(was OPEN) first CSS DATA after power-on held for minutes** -- root cause above.  Same run: m6600's first DATA keyed
  3.5 min (0.5 s) / 2 min (0.15 s) after the send, the Mac's first 0.15 s DATA 1 min 47 s after; second-round
  sends keyed within seconds, ACKs are never held.  ACKs only need one clear instant, DATA needs DIFS (2.5 s
  idle) + backoff, so carrier sense is busy most of the time early on.  At 0.15 s `channel_busy()` also takes the
  streaming chirp tracker's `busy` (C css_trk: on-threshold calibrated on the first 3 s of ambient, 2 s off
  hold); m6600 logged a false CW ident (`I EE EE I I E N EE D`) 8 s after power-on and keyed exactly 120 s later.
  INFO shows no CSS carrier-sense state (`link.rx_busy` is the OFDM receiver's) -- add one (tracker busy,
  `_in_burst`, last-busy age, CSMA backoff/idle) before guessing further.
- **FIXED (2026-11-17): speaker click at CSS chirp edges.**  Every chirp started and stopped at an arbitrary phase
  (rectangular envelope + 5 % gap), a step at each symbol edge.  `css_encoder.edge_taper` (encoder option
  `edge_taper_cycles`, reference default 0) fades each chirp in/out with a raised cosine over N cycles of f_max;
  the GUI sets `css.edge_taper_cycles` = 3 (0.5 ms at 6 kHz).  A tapered 43 s ACK at 0.5 s symbols decodes at
  62 dB SNR.

## Risks

- PySide6 wheels + PortAudio + libs in PyInstaller on 3 OSes is the real schedule risk; get a packaged hello-world early in phase 1, not at the end.
- Spectrogram at 48 kHz in Python: keep FFT in numpy on a worker thread, hand the UI a prebuilt column; never FFT in the paint path.
- Pulse over TCP adds latency/jitter in a container; fine for display, check before trusting for timing-sensitive modem work.
