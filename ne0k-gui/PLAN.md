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

## Risks

- PySide6 wheels + PortAudio + libs in PyInstaller on 3 OSes is the real schedule risk; get a packaged hello-world early in phase 1, not at the end.
- Spectrogram at 48 kHz in Python: keep FFT in numpy on a worker thread, hand the UI a prebuilt column; never FFT in the paint path.
- Pulse over TCP adds latency/jitter in a container; fine for display, check before trusting for timing-sensitive modem work.
