# tests/MESH/c/ — mesh state machine, C89

Mirrors `tests/OFDM/c/`: freestanding C89, no malloc/stdio/globals, built for
the DOS/OpenWatcom target ([[dos-openwatcom-port]] in memory terms — the CSS
and OFDM cores are already there; this is the mesh layer's turn).

## What's here

**`mesh_phy.c`/`.h`** — port of `PhyPolicy` (`../meshphy.py`): the per-peer
CSS↔OFDM PHY selection policy — REQ/DATA retry counters, give-up logic, the
CSS-fallback decision, and the EVM-feedback margin controller. Chosen first
because it was *already* the one dependency-free piece in the Python mesh
stack: no numpy, no audio I/O, just a `tick()`/`on_*()` state machine
returning actions for a driver to carry out. That means it needed no DSP
groundwork to port — just a faithful state-machine translation — so it's the
place to prove out the porting conventions (action-list-by-output-buffer,
enum reason codes instead of Python f-strings, `-1.0` timestamp sentinel
instead of `Optional[float]`) before tackling the harder pieces below.

`mesh_phy_selftest.c` exercises four of the scenarios from
`../../regression/mesh_fallback_regress.py` directly against the state
machine (no packet layer): no-autonomous-reentry, bounded ofdm_only give-up,
the link-silence receiver-desync fix, and the `to_me` CSS-rx gate. Run with
`make selftest`.

## What's NOT here yet, and why

1. **DONE — the OFDM handshake FSM.** `HandshakeSM` is now ported as
   `../../OFDM/c/ofdm_hs.c`/`.h` (`make hs_selftest` / `make c89check` in that
   dir). It lives under `tests/OFDM/c/`, not here, since it's negotiated
   per-link OFDM protocol, not mesh-specific — but it's what
   `meshlib.run_handshake()` drives, so a mesh node needed it to be
   handshake-capable in C at all. Two Python features are deliberately NOT
   in the port because both are opt-in and off by default in the reference
   (see `ofdm_hs.h`'s header comment for the full rationale):
   - **band renegotiation** (`MSG_BAND`/`band_negotiation`) — recognised on
     the wire (won't be misread as corrupt) but not acted on.
   - **BAT wire compression** (`HS_BAT_RLE`, `bat_coarsen`, the RobustLink
     rate ladder) — BAT bodies ship raw, which is exactly what
     `HS_BAT_RLE = False` already means in the reference. A real airtime win
     worth porting later, but it's a wire-encoding optimization layered OVER
     the state machine, not part of it.

2. **The mesh-OFDM data-frame framing** (`MeshModulator`/`MeshDemodulator` in
   `../meshlib.py`) rides on `ofdm_mod.c`/`ofdm_dem.c`/`ofdm_frame.c`/`rs255.c`
   in `../../OFDM/c/`, which already exist. What's missing is the
   mesh-specific wrapper: the `[2B length][packet][zero pad to chunk_bytes]`
   framing, the dedicated preamble carrier-PN seed that keeps a data frame
   from ever being mistaken for a handshake probe, the golden-ratio bit
   interleaver (not a block interleaver — see `meshlib.py:124-135` for why),
   and the RMS-to-probe TX level match. This is real work because it's the
   piece that determines whether a byte survives the channel, not just
   whether two ends agree on what to try next.

3. **The CSS packet header** (`../../CSS/css_modem.py`'s
   `[src:4|dst:4][type:4|seq:4][payload]`) is two bytes and a handful of
   `#define`s — deliberately NOT a library here (see `mesh_phy.h`'s header
   comment). Whoever wires a real driver loop around `mesh_phy.c` packs/unpacks
   it inline; duplicating `MESHPHY_CM_TYPE_DATA`/`MESHPHY_CM_TYPE_ACK` in
   `mesh_phy.c` already shows the whole of what's needed.

4. **Everything in `meshmodem_node.py`, `meshbus.py`, `meshchannel.py`,
   `meshevents.py`, `meshroll.py`, `meshrun_node.py`, `meshsim_dsp.py`,
   `meshsim_mac.py`, `meshtests.py`** is simulation/test-rig scaffolding for
   this Python OTA test harness — multi-host coordination, wall-clock TX-slot
   scheduling, JSONL event logs, HTML piano-roll rendering. None of it has an
   embedded-target equivalent, and none of it is planned for a C89 port. What
   IS worth carrying over from `meshmodem_node.py` as design guidance (not
   code) is its guard conditions — deaf-node detection, input-clipping
   detection, capture-stall watchdog — since a freestanding node needs
   *some* answer to "is my mic actually receiving," even if the mechanism
   looks nothing like the Python rig's.

## Interface notes carried over from the Python state machine

- **No timed autonomous re-upgrade.** `upgrade_wanted` is only re-armed by an
  external signal (a peer's REQ, or `meshphy_on_detect`) — never by time
  alone. The fallback exists because the two ends may have been physically
  separated with nobody watching; retrying blind into that is how you turn a
  quiet mesh into a wasted-airtime one.
- **Every retry loop is bounded.** `req_retries`/`data_retries` bound the
  fallback path; `max_hs_attempts` bounds `ofdm_only` re-negotiation. An
  unbounded loop against a peer that already left burns a full handshake
  timeout per doomed cycle (measured OTA at ~600s/cycle).
- **`link_silent_s` (receiver-side) must expire before the sender's own
  give-up.** Sized as `(1 + data_retries) * ack_wait_s` by default so a pure
  receiver notices the sender moved on before the sender's next probe lands
  on a peer that still believes the old link is up.
- **A queued-but-unaired retry needs its own ack-wait timer.** `req_in_outbox`/
  `data_in_outbox` in `meshphy_tick` exist so a give-up can't fire while a
  final retry is sitting in an outbox waiting for airtime, timed from an
  earlier, already-superseded transmit.
- **`to_me` must gate CSS-rx link teardown.** Overhearing the peer talk CSS to
  a third node is not evidence the peer left our link.
