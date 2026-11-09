# sip_dialer

The one SIP dialer for the mesh tests: connects modem nodes to the SIP conference and monitors the call.

## Files

| File | Role |
|---|---|
| `meshdialer.py` | Launcher. One `sipbridge.py` leg per node, or per pair in full-mesh mode. |
| `sipbridge.py` | One SIP leg: G.711 u-law RTP over PulseAudio. Modes: `call`, `answer`, `listen`. |
| `listengui.py` | Windows/macOS front end: `sipbridge.py listen` plus the `meshgui.py` window. |
| `meshgui.py` | "SIP call monitor" window, tkinter. Reads leg stats files only. |

## Conference

- Server: `your_server_ip:15064`, config `prereqs/meshsip/sipserver_mesh_external.json`.
- Number `3000`, line type `conference`, `max_members` 10.
- Each member hears the mix of everyone else, never itself.
- A listener sends digital silence, which mixes in as zero, so extra members do not disturb a test.

## Test node pipeline

What the 3-node group tests run on each Linux node:

1. `demo_env.sh` creates null sinks `mesh_<node>` = modem TX and `mesh_<node>_rx` = modem RX.
2. `meshdialer.py --server your-server-ip:15064 --conference 3000 --stats-dir OUT/stats ...` starts one `sipbridge.py call` leg: sends `mesh_<node>.monitor`, plays into `mesh_<node>_rx`.
3. `meshgui.py --stats-dir OUT/stats --node <node> --ping-host 0x13h.org` shows call state, RX/TX bit/s, latency, dropped RTP frames and a Monitor audio toggle.

## Linux user PC: same path as the testlist

Needs PulseAudio or PipeWire-pulse, numpy, tkinter. Use an unused node name such as `D`. A fresh null sink sends silence.

```bash
pactl load-module module-null-sink sink_name=mesh_D; pactl load-module module-null-sink sink_name=mesh_D_rx
python3 meshdialer.py --nodes D --local D --prefix mesh --out ~/md --stats-dir ~/md/stats --server your-server-ip:15064 --conference 3000
python3 meshgui.py --stats-dir ~/md/stats --node D --ping-host 0x13h.org
```

Monitor audio plays `mesh_D_rx.monitor`: what the conference sends to D.

## macOS / Windows test nodes

The group tests run on macOS and Windows hosts too. Two loopback audio devices stand in for the null sinks: BlackHole 16ch/2ch on macOS, VB-Cable plus VAC Lite on Windows. The `audio_loopback_check` testlist proves them and writes their names to `loopback.json`. `sipbridge.py --source/--sink sd:name=TEXT` opens those devices through PortAudio instead of `parec`/`pacat`. `meshdialer.py --tx-source/--rx-sink` passes the names through, and `--hosts NODE=auto` advertises the address that routes to `--server`. The container's `__testplan__.py` picks each node's steps by host OS.

## Windows user PC

Outside the test framework, `meshgui.py` alone still assumes Linux:

- `sipbridge.py call` without `sd:` devices needs `parec`/`pacat`.
- meshgui's Monitor audio needs `pactl`.
- meshgui's latency uses Linux `ping -c`/`-W` flags.

`listengui.py` substitutes those three pieces. It shows the same window but does not validate the testlist path.

```powershell
pip install numpy sounddevice
python .\listengui.py
```

- Joins conference 3000 receive-only via `sipbridge.py listen`, with `--stats-file` feeding meshgui.
- Monitor audio starts muted. The button flips `sipbridge.py listen --mute-file`: silence goes to the player while the file exists, so the call never drops.
- Options: `--to HOST:PORT`, `--number N`, `--player auto|sounddevice|ffplay|wav:FILE`, `--meshgui DIR`.
- Not yet run against the live server or on Windows.

## Open items

- Some testlists still point at `GIT_SRC_DIR/audiomodem_linux/sipbridge.py`: `build_win`, `build_linux`, `build_macos`, `build_freebsd`, `modem_serial_sip`, `modem_serial_perf`, `modem_css_ratesweep`. Repoint them here, then delete that copy. Wait for the running streaming test to finish first.
- Stale copies to leave alone: `ab_src/ab_A/...` is a build pin slot, `ne0k-audiomodemold` is an old tree.
- Option: make `meshgui.py` itself portable so the testlist and Windows run one script and `listengui.py` shrinks to a launcher.
