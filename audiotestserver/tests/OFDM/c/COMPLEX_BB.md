# Complex-baseband (I/Q) OFDM — plan and status

Goal: the OFDM modem running natively on complex baseband, so it can sit on a GNU Radio / PlutoSDR
I/Q stream with a 100 kHz+ channel centred on DC, instead of only real audio passband
(G.711 / sound card, 48 kHz, carriers 1–16 kHz).

**Rule: additive.** The real (audio) path keeps its ABI, its numpy oracles and its regressions
byte for byte. Complex mode is a parallel set of `ofdm_cx*` entry points next to it, sharing only
the code that never looks at a bin number or a sample. Nothing in `native.py`, `ofdm_acq_c.py` or
the DOS build has to change until it opts in.

Test bench: `css-ofdm-audiomodem.gnuradio-mesh-2node`. `prereqs/gnuradio/gr_rf_medium.py` is the
shared air (one TCP I/Q socket per node, mix-minus). Today `gr_medium_probe.py` stands in for the
nodes; the modem replaces it at step 9 below.

## What differs from the real path

| | real (audio) | complex (I/Q) |
|---|---|---|
| sample | 1 double | I,Q pair, interleaved doubles (cf32 wire order) |
| bins | 1 .. N/2-1, Hermitian mirror implied | signed −N/2+1 .. N/2-1, no mirror |
| DC | never in band (bin_lo ≥ 1) | in the middle of the band: `dc_null` hole (LO leakage on AD936x) |
| transform | `ofdm_rfft` / `ofdm_irfft` (N/2 complex + pack) | `ofdm_cfft_fwd` / `_inv`, N complex |
| band at the same fs | fs/2 − guard | ≈ fs − guard (100 kHz at 125–200 ksps) |
| carrier frequency offset | a real channel has none | the receiver must estimate and remove CFO (LO mismatch, ppm × RF) |
| IQ imbalance | n/a | carrier k leaks into −k; pilots/BAT measure it later |

## Status

| step | piece | state |
|---|---|---|
| 1 | `ofdm_cfft_*` in ofdm_fft.c (additive) | **done**, graded vs numpy f64 + f32 |
| 2 | `ofdm_cxcfg_build` / `ofdm_cx_bin` (signed bins, DC hole), ofdm_cx.c | **done** |
| 3 | `ofdm_cxsym_make` / `ofdm_cxsym_demod` (one symbol, CP + taper, both ways) | **done** |
| 4 | `ofdm_cxmod` (ofdm_cxmod.c): frame = SC preamble + CE + SYNC + data, OLA, RS/interleave/scramble reused | **done**, graded vs the real modulator's carriers |
| 5 | `ofdm_cxsync` (ofdm_cxsync.c): Schmidl-Cox on complex samples, **CFO estimate** (fractional from the SC half-symbol phase, even integer from the CE), timing, `ofdm_cx_derotate` | **done**, graded vs numpy + truth over a CFO / multipath / AWGN sweep |
| 6 | `ofdm_cxdem` (ofdm_cxdem.c): per-window CFO derotation, CE/SYNC channel estimate, pilot phase a + b·bin (residual CFO, sample-clock drift), demap, descramble, de-interleave + RS | **done**, graded vs numpy + truth (CFO sweep, wrong CFO, 200 ppm SRO, ±12-sample start) |
| 7 | `ofdm_cxacq` (ofdm_cxacq.c): CFO-invariant (time-differential) coherence acquisition; `ofdm_cxprobe` (ofdm_cxprobe.c): probe build + estimate, per signed bin SNR and the **IQ image**, SC/CE confirmation, feeds `ofdm_allocate` unchanged | **done**, graded vs numpy + truth (CFO sweep, 10 dB, IQ imbalance TX/RX vs theory, 200 ppm, probe → BAT → frame) |
| 8 | `iq:` connector (modemgui/audio/iq_backend.py) + I/Q pipe/session path (`StreamSpec.iq`), `engine/ofdm_cx.py` (ctypes: CxModem, streaming CxReceiver, CxProber, `tx_gain`), `engine/iq_link.py` (IqLink, message framing), modemcfg `IqConfig` + `IQ_PROFILES["iq"]` (fs 200 k, band ±50 kHz, taper 0.08 ms = CP/4), CLI `iq-send` / `iq-recv` | **done**, tests/test_iq.py + 100 kB 64-QAM transfers over the real gr_rf_medium.py |
| 9 | probe → BAT exchange (`IqLink.negotiate`, a receiver that holds the base and the probed table at once), `iq-p2p` CLI, group testlist `gnuradio-iq-p2p-2node` (R = medium, A/B = modemgui nodes on `iq:`); throughput: gapless + 2 kB frames (step 8 notes) | **done** 2026-11-14, group PASS over the real medium at ~23 dB |
| 10 | impairments in the medium (impair.py: delay, CFO, echo, fade, pink, impulse, tone, dropout; runtime `--ctl-port`), RTD ranging in `negotiate` (V.34-style), matrix group test `gnuradio-iq-impair-2node`; **10b** targeted BAT-slot impairments (engine/slotimpair.py) for complex (`gnuradio-iq-slot-2node`) and simple OFDM (`ofdm-slot-simple`) | **tests done** 2026-11-14; most frequency-selective cells red: see step 10 / 10b notes |
| 10c | **erasure decoding** in ofdm_cxdem (GMD retry per codeword from per-cell decision distance + dead-carrier |H|, `ofdm_cxdem_set_erasures`, `ofdm_cxdem_carrier_quality`); **selective-repeat ARQ** in IqLink (`send_reliable` / `receive_reliable`: POLL → STAT bitmap + receiver-graded table update, resend missing); `iq-p2p` uses both by default (`--no-arq`, `--no-erasures`); slot rule `:pin` | **done** 2026-11-14: see step 10c notes (cluster 10/19 cells PASS, was 4/16) |
| 10d | the 10c missing-feature list: inner FEC (ofdm_cxfec.c, K=7 soft Viterbi, default 2/3), decision-directed tracking + pilot fallback, late window + `iq-echo` / `iq-hf` profiles, ARQ up-shift + re-probe on request, tone exciser, pipelined ARQ (`send_window`), full-duplex RX, CW station ID, LBT / PAPR clip / RX level | **done** 2026-11-14: see step 10d notes (cluster impair group 13/14 PASS) |
| 10e | fading-channel ARQ: short control frames (`ctl_chunk_bytes` 512), best-of-probes negotiation + weak-table re-probe, re-probe restores only zeroed carriers, grading one step down per update, unanswered POLLs cost no round, receiver lingers for re-POLLs, sender deadline; impair matrix gains `profile` + `echo-isi-hf` / `hf-moderate-hf` | **done** 2026-11-14: cluster impair group 15/16 cells PASS (only hf-moderate on the default profile) |
| 10f | **decoder-aided re-estimation** (`ofdm_cxdem_set_iterate`, `IqConfig.iterate` / `iterate_win`, CLI `--iterate` / `--iterate-win`): a frame whose RS fails is re-encoded from the Viterbi output (+ RS-corrected codewords), H refit per carrier over ±W symbols, decoded again | **prototype** 2026-11-15, off by default: offline bench rescues most fast-fade losses, nothing on HF notch losses (step 10f notes); not yet run on the cluster |
| 11 | Pluto: same `iq:` socket via iq_connector.py `--pluto-uri` (fs ≥ 521 ksps on AD936x; decimate in GR) | |

Step 4 test: the selftest runs the REAL `ofdm_mod_frame` (itself graded sample-for-sample against
ofdm_modem.py by `ofdm_mod_regress`) on a real config with the same occupied/data carrier counts,
reads every body symbol's carriers back (CE-equalised), rebuilds the complex frame in numpy from
them and compares it sample-for-sample with `ofdm_cxmod_frame` (f64 1e-11, f32 2e-4). A wrong
scrambler seed fails it at 2.7. It also checks encode_ex gaps, NUL padding and the seq wrap.

**SC preamble, the one deliberate difference.** The real path puts the PN on even occupied
*indices*; across the DC hole index parity flips bin parity and the halves would no longer repeat.
`ofdm_cxmod` puts it on even *signed bins* (ascending order, same seed, amp, TX gain 2.0), so
the body is exactly periodic in n_fft/2 and a CFO of ε subcarriers rotates the second half by
e^{jπε}: step 5's fractional estimate (range ±1 subcarrier), then the CE gives the integer part.

Step 5 test: a numpy reference of the whole find (cumsum metric, same region/gate logic, frac, integer,
rotated matched filter) must give the C the same starts and integer CFO and frac/scores to 1e-9, in f64
and f32 (only the CE FFT is float). Against truth: three encode_ex frames, 2-path channel, 25 dB, CFO
0, ±0.37..0.999, −1.0, 2.5, −3.3, 7.25, −14.6 subcarriers: start exact, CFO within 0.0033 (≈2σ of the
L=128 estimator), and after `ofdm_cx_derotate` the CE→last-SYNC phase equals what the residual CFO
predicts (0.009 rad). Also 6 dB, noiseless (CFO to 1e-9, starts exact), a truncated CE, block-continuous
derotation, and no frames from noise, DC, a CW tone, or tone+DC. Four mutations (frac sign, no integer
search, unrotated matched filter, no D gate) each fail it.

**Step 5 design notes.**
- The SC phase repeats every 2 subcarriers, so the integer part is always **even**; the search runs
  q = 0, ±2, ±4 … up to `max_cfo_bins` (init argument: 2 × ppm × f_rf / spacing).
- The integer score D(q) is **differential** (neighbouring carriers, so timing error and channel
  phase cancel) and **phase-only** (each pair a unit phasor). Energy-weighted, one strong bin wins.
- On complex samples a DC offset or **any** CW tone is periodic in n_fft/2 and scores the SC metric 1.
  At n_fft 256 the matched filter's noise maximum (~0.15) also clears the real path's NCC gate (0.10).
  So find also requires D ≥ `OFDM_CXSYNC_INT_MIN` (0.35; noise ≲ 0.2, a CE at 6 dB scores 0.92).
- The matched filter correlates against the reference **rotated by the estimate** (unrotated, ε≈1
  drops the NCC to 0.17), and mean-removes the window, since a rotated reference is no longer zero-mean.
- find drops a candidate whose CE runs past the buffer (it can't be scored or demodulated); stream
  with ≥ 1 frame of overlap. `ofdm_cxsync_eval` still reports it with `int_score` −1.

Step 6 test: a numpy reference of the demodulator up to the descrambled wire bytes must give the C the
same hard decisions byte for byte and the EVM to 1e-9 (f64) / 1e-4 (f32), on the frames `ofdm_cxsync`
finds (its start and CFO) over the 25 dB 2-path CFO sweep. Against truth: every chunk and sequence number
(0xFFFE, 0xFFFF, 0: the wrap) decodes there (EVM ≤ 0.065, RS errata ≤ 2), noiseless (EVM ~0, no errata),
with the CFO handed in 0.03 subcarriers wrong (the pilots track ~0.24 rad/symbol), with a 200 ppm slow
sample clock (`resample_poly`, ~2.4 samples of drift per frame), and with the start 12 samples early or late;
plus the error codes. Six mutations (pilot phase sign, no pilot phase, no slope, slope by occupied index
instead of signed bin, CFO ignored, window at `cp_len`) each fail it **against truth**, not only against numpy.

**Step 6 design notes.**
- The CFO is removed **per FFT window** inside `ofdm_cxdem_frame` (phase referenced to the frame start, so
  the constant left over is the same every symbol and the CE absorbs it); the caller's buffer stays const.
- Pilot slope is in **signed bin**, not occupied index: across the DC hole they differ by one, and a timing
  drift is linear in frequency. The slope uses only adjacent pilots at the smallest bin gap, which leaves
  out the pair straddling the hole.
- **FFT window = taper + cp_len/2**, the middle of the ISI-free stretch [2T, CP] (the previous symbol's taper
  ends at 2T, the next one starts at the symbol length). The real path windows at `cp_len`, that stretch's
  right edge. With the cfg's taper clamp at CP/2 the stretch is a single point, so a 3-sample echo alone
  gave errata ≈ 24 per frame at 25 dB. The step 6 test uses taper = CP/4 (16 samples at 200 ksps); **step
  8's iq profile must keep the taper ≤ CP/4**. (The real audio path has the same zero margin at its clamp;
  not changed here, by the additive rule.)
- Constellations come from `ofdm_mod_table`, the tables `ofdm_cxmod` maps with, instead of being passed in as
  `ofdm_dem_init` does (that guards the C against the Python tables; complex mode has no Python modulator).
- RS is in the core (`ofdm_frame_decode_il`), unlike `ofdm_dem`; `ofdm_cxdem_wire` keeps the pre-RS bytes.

Step 7 test: `ofdm_cxacq_run` against a numpy reference (float32 grid, double sums) to 6e-16 (f64) / 4e-9 (f32);
a periodic block scores ≥ 0.98 inside at CFO 0, 0.4, −0.81, 3.3, −14.6 subcarriers, noise ≤ 0.22 against the 0.70 gate.
`ofdm_cxprobe`: build against numpy sample for sample (its SC is `ofdm_cxmod`'s); estimate against a numpy reference of
steps 4–6 at the C's start and CFO (SNR, periodic SNR, image to 1e-12 dB f64 / 2e-4 f32, H to 1e-9). Against truth
(64 periodic + 32 measurement symbols, 2-path, 25 dB, CFO 0 .. −14.6): start exact, CFO within 0.0042, H within 0.04 of
the channel after a phase ramp, per-carrier SNR median within 0.14 dB (90th percentile 1.6 dB); 10 dB within 0.07 dB;
6 dB (4 captures) within 0.21 dB; a short 16 + 8 probe at 10 dB averages +0.07 dB over 8 captures; a 200 ppm sample
clock within 0.04 dB. IQ imbalance (1.05 gain, 3°, theory −28.93 dB) at the transmitter with CFO 0.37 and −3.3 and at
the receiver with CFO 0 reads −28.9..−29.0 dB, the SINR it reports is within 0.1 dB of μ/ν theory, and the periodic
block shows the image as a 12 dB "ISI penalty". The probe's SNR → `ofdm_allocate` (3 dB margin) → a 16/64-QAM frame
through the same channel decodes (EVM 0.081, 3 RS errata). Refused: noise and a DC offset (SYNC_FAIL), a CW tone
(coherence 0.92, then NO_SC), a probe missing its SC, three data frames (coherence 0.36); a capture cut inside the
measurement block still measures, too short / over max_capture is TOO_SHORT. Eleven mutations each fail it, all
against truth: non-differential acquisition, no phase tracking, no slope, no 8-bin lag stage (6 dB reads −7.6),
no LS refinement, tracking sign, no image regressor, wrong mirror, no CFO derotation, no SC/CE confirmation (the CW
tone is taken for a probe), no noise correction in the SINR (the short probe reads −0.8 dB).

**Step 7 design notes.**
- **Acquisition must not care about CFO.** `ofdm_acq` sums H across symbols; a CFO of 0.4 subcarriers turns each
  symbol by π (SS/N = 1.25) and the sum cancels. `ofdm_cxacq` scores conj(Y_j)·Y_{j+1} instead, which drops any
  constant per-symbol rotation (fractional, integer, and the ICI it smears, which repeats too). No known pattern is
  needed: it cancels from the score. Accumulators are double: the denominator is a fourth power of the level and
  underflowed float32 for quiet input.
- The price: **any stationary signal repeats** up to a phase and scores 1 on its bins (a CW tone reads 0.93). So a
  coherence peak is only a candidate. The probe then runs `ofdm_cxsync_find` in a window ahead of the periodic block:
  its SC/CE (the frame's first two symbols) give the exact start, the CFO, and the CE-sequence check a tone fails
  (`OFDM_CXPROBE_NO_SC`). With the start exact there is no prefix-alignment search (`align_meas`) to port.
- **Phase tracking per symbol**, against the CE's H over all carriers: residual CFO (~0.02 rad/symbol) over a
  100-symbol probe would decorrelate the mean, and a 200 ppm clock drifts ~6 samples. The slope needs three stages:
  neighbour differences (unambiguous, noisy: ~3 rad of error at the band edge at 10 dB), then an 8-bin lag unwrapped
  against it, then weighted least squares on the residual. Without the lag stage the 10 dB SNR swung 2 dB run to run.
  (`ofdm_cxdem`'s pilot slope is the plain differential at the pilot spacing; fine at decodable SNR, but the same LS
  refinement would tighten it.)
- **IQ image:** per carrier, least squares y = G·x[k] + I·x[−k] over the measurement block (BPSK is real, so conj(x) = x).
  The SNR handed to the allocator is |G|²/(|I|² + σ²) with the noise each estimate carries taken out, and |I|² NOT
  clamped at zero afterwards (clamping biased the denominator high: −0.4 dB at 8 symbols); `image_db` = |I|²/|G|² raw (on a clean radio it reads the fit's floor, ≈ −10·log10(SNR·M/2)). An RX
  imbalance behind a CFO turns at 2× the CFO and is only partly fitted; the remainder lands in σ², so the SNR stays honest.
- **Not ported:** the ISI ratio β (SDR delay spreads are microseconds against a 320 µs prefix; the measurement block
  counts ISI anyway) and the percentile estimator (as in the real probe).
- **Measurement-block seeds (fixed 2026-11-14, real and cx together):** the measurement patterns used seeds
  `OFDM_SEED_MEAS + j`, and adjacent LFSR seeds gave the **first ~11 occupied carriers the same value in every
  measurement symbol**, so on those carriers the "uncorrelated" block was periodic and read reverb as gain (simulated
  room: 18.9 dB vs 10.5 dB on the rest). The block is now one PRBS stream from `OFDM_SEED_MEAS`, n_occupied bits per
  symbol (`ofdm_known_bpsk_next` / `_known_bpsk_stream`), in `ofdm_probe.c`, `ofdm_modem.py`, `ofdm_dos/core`,
  `ofdm_cxprobe.c` and the selftest; after it those carriers read 8.7 dB. It changes the probe on the air: an old and a
  new node grade each other's measurement block at about -16 dB, so every node and prebuilt lib moves together.

Each step gets the same treatment: a numpy reference in `ofdm_cx_selftest.py`, f64 then f32,
sample-for-sample where the math is deterministic, and mutations that must fail against truth.

Step 8 test: `tests/test_iq.py` (ne0k-gui, 10 cases, 2.5 s): address parsing, cf32/sc16 wire
round trip (sc16 clips, never wraps), the registry refusing an I/Q session on an audio device and
vice versa, the two-sided STFT (a full-scale complex tone at −31.25 kHz reads 0 dB in its bin),
the `iq` profile; a mock medium (wall-clock clocked, zero-fills TX gaps and counts them) carries
30 000 samples through the pipe bit-exact and in order with no underrun, and an `IqLink` message
(3 kB, sequence numbers wrapping at 65535) through it at −6.1 kHz CFO and 25 dB; the probe at the
link's TX gain reads the in-band SNR to 1 dB and its bit table carries frames; a taper past CP/4
is refused. Over the real `gr_rf_medium.py` (2 nodes, localhost, noise 0.002): probe 43 dB → all
64-QAM → 20 kB in 79 frames, 4/4 runs CRC-good at EVM 0.010; 100 kB in 391 frames 3/3 with zero
underruns. CPU at 200 ksps: 13–19 % of one core per node while transferring, ~11 % idle-receiving
(Python + C, one process per node) — far under the estimate below.

**Step 8 design notes.**

- *One socket, both ends.* The medium's wire is one bare duplex TCP stream per node, so an address's
  source and sink share one `_Link`; it closes when both have stopped (socket first, so the medium
  sees the node leave instead of counting the reader's wind-down as underrun).
- *The medium is the clock.* For every n samples received the connector pulls n from the pipe and
  writes them (zeros when idle). No local pacing, no drift, own TX on the RX timebase.
- *TX lead 0.5 s.* Lockstep alone keeps the medium's TX buffer one block deep, and the medium takes
  TX before it emits the same tick's RX, so any lateness zero-fills the middle of a frame. A lead of
  zeros is primed once; lockstep holds it LESS the RX in flight in the flowgraph, which swung by
  >0.2 s here (0.25 s lead: buffer seen at 18 ms, 3 underruns per 100 kB, 4–5 frames lost; 0.5 s: none).
- *A slow node stalls the whole medium.* `gr_rf_medium.py` writes each node's RX with a blocking
  `sendall` (0.5 s timeout) from the flowgraph thread, so a node that stops reading stalls the AIR
  for every node. Found when `AudioPipe.history()` joined all 30 s of history (48 MB at 200 ksps)
  under the GIL four times a second: the receiver's reader stalled and the SENDER underran.
  `history(n)` now joins only the tail. A medium-side fix (non-blocking per-node writer) belongs to
  step 10.
- *One TX gain for everything.* `tx_gain(cfg)` puts a unit-power data frame at `tx_rms` and the probe
  goes out at the same gain. RMS-matching each burst instead made the probe read 3 dB optimistic
  (its known carriers are 1/√2 and cxprobe reports the SNR of a unit-power constellation).
- *Throughput (fixed 2026-11-14): gapless bursts + 2 kB frames.* Back-to-back frames
  (`encode(gap=False)`) lost 1 in 4 in `ofdm_cxsync_find` itself: the min_gap test compared the
  threshold edge of the next region with the previous ACCEPTED (95 %) edge, which sits later by the
  metric's ramp, and min_gap's cp_len allowance only covers that when frame_gap separates frames;
  back to back the stride is exactly min_gap + cp_len, so the next frame read as a second region of
  the last and replaced it. Now 95 % edge against 95 % edge (C + numpy reference; the selftest has
  a 12-frame gapless case). `IqConfig.gapless` (default on): one frame gap before and after a burst,
  none between frames; `--gapped` on iq-send/iq-recv restores the old layout.
  Then frame size: RS parity is sized so one lost OFDM symbol stays correctable
  (`ofdm_interleave_geometry`, shared wire format, unchanged): at 672 bits/symbol that is 11
  codewords x 16 = 176 B per frame whatever the chunk, plus SC + CE + SYNC, so ~5 symbols of fixed
  overhead per frame. Chunk -> 64-QAM gapless rate at 200 ksps (PHY ceiling 420 kbit/s), all 30/30
  at 26 dB + CFO offline: 256 B 141, 1020 B 282, 2040 B 339, 4080 B 357 kbit/s. Default now
  2040 B (48 ms frames; past it gains are small and a lost frame costs more). A short message is
  NUL-padded to one frame.
  Over the real gr_rf_medium.py (2 nodes, noise 0.002), 100 kB, 3 runs each, all CRC-good, zero
  underruns: gapped 256 B 391 frames 13.5 s air / 15.2 s wall; gapless 256 B 5.7 s / 7.4 s;
  gapless 2040 B 50 frames 2.4 s / 4.0 s.
- *Not covered yet:* `iq_connector.py` / Pluto (step 11), real CFO from the medium (step 10), the
  GUI's settings dialog for `IqConfig` (it is generated from the dataclass, not wired to a page).

**Step 9 design notes.**
- *One table per direction, measured by the far end.* A's TX table is what B measured from A's probe,
  and the other way round, so an asymmetric channel (different noise or image per node) gets two tables.
- *No agreed switch instant.* Only the data symbols depend on the table: SC, CE and SYNC do not. So
  `CxReceiver(tables=...)` runs one sync and demodulates each hit with the table that worked last,
  then the others, until RS accepts it (a wrong table fails RS, it does not decode to garbage).
  `CxFrame.table` (CRC-32 of the table) / `.bits` say which one took it. Control messages always go
  at the base table (`default_bits`, QPSK); every receiver keeps the base table next to the probed
  one; data goes at the probed table. A lost control message is simply sent again, and no frame is
  ever lost because the two ends switched at different moments.
- *The exchange* (`IqLink.negotiate`, messages `IQB1` = BAT {table for the recipient's TX, have = "I
  hold my own TX table"} and `IQR1` = READY):
  I PROBE → R grades it, BAT{I→R, have 0} + PROBE → I grades it, BAT{R→I, have 1} → R: done, READY
  at its new table → I: done, READY at its new table → R settled. Retry after `retry_s` (3 s) of
  silence: a PROBE while either side lacks a measurement, else our BAT; a repeated BAT{have 1} after
  we are done gets READY again; an ok frame at the table we handed out counts as READY. On the
  medium it is one round, 3–5 s, mostly the 0.3 s probe plus the base-table control frames.
  `negotiate` returns `since` (its starting RX index): the peer's first data can follow its READY
  closely enough to arrive before we return, so a receive must look from there, not from "now".
- *SNR margin 7 dB for iq (was 3).* The `ofdm_allocate` floors (4/7/14/21 dB, shared with the real
  path, unchanged) are raw. Offline, the probe reads the frame SNR within 0.2 dB, but with 3 dB just
  above each step RS lost up to half the frames (16-QAM at 19 dB: 6/20; 64-QAM at 24 dB: 9/20).
  At 6 dB the QPSK/BPSK edge still loses frames (13 dB: 16/30); at 7 dB every 16/64-QAM edge is
  30/30, and 14 dB is 29/30. The real-path profiles already use 6–9 dB.
- *`iq-p2p`* (cli): `--role initiator|responder`. The initiator sends `--send-bytes` of seeded random
  data (seed + sender's node id, so the far end checks it without a file); the responder checks it
  and answers with its own; then the initiator checks that. Prints NEG / SENT / RECV and
  `P2P PASS|FAIL`; `--json` writes the tables, the probe SNR/CFO/image, air time and frame counts.
  `--min-bits` fails a link that ends up no better than QPSK.
- *Results.*
  - Local gr_rf_medium.py, 100 kB + 20 kB reply:
    - noise 0.002: 43 dB, 672 bits/sym both ways, 332 kbit/s.
    - noise 0.03: 19 dB, 236 bits/sym, 125 kbit/s.
  - Group test `gnuradio-iq-p2p-2node`, noise 0.02, R xp5kvm13, A docker manager, B xp5kvm14:
    - The two directions measured 22.8 / 23.0 dB and got 440 / 448 bits/sym, 227 kbit/s.
    - 100 kB took 3.5 s air, and the run took 38 s end to end.
    - Medium: no underrun or overflow, late_max 12 ms.
  - Each node counts one "bad" frame: the peer's probe opens with an SC preamble, so the sync takes
    it for a frame and RS rejects it. That is harmless.
- *Not covered yet:*
  - ARQ: one lost frame fails the message.
  - Re-probing during a long transfer.
  - The low-SNR edge: 14 dB is 29/30 even at a 7 dB margin, probably because the single CE symbol
    gives a noisy channel estimate.
  - 3+ nodes (BAT per peer).
  - The GUI.

**Step 10 design notes (medium impairments).**

- *Where:* `prereqs/gnuradio/impair.py` holds the models; `gr_rf_medium.py --impair SPEC` builds them into the graph.
  - Per path: CFO rotator → `PathImpair` (delay line, echo taps, Rayleigh/Rician fade per tap, dropout) → gain.
  - Per receiver: AWGN (native) + `RxImpair` (pink 1/f, Poisson impulse crashes, CW tone).
  - SPEC: `delay=MS,cfo=HZ,echo=MS@DB+MS@DB,fade=HZ[@K_DB],pink=AMP,impulse=RATE@AMP@MS,tone=HZ@AMP,dropout=RATE@MS`,
    or a preset (latency, latency-acoustic / -g711 / -satellite, cfo, echo, echo-isi, fade, pink, impulse, tone,
    dropout, vhf-mobile, hf-moderate).  Amplitudes are in noise_amp units: the signal arrives at 0.2.
  - `--ctl-port`: every block is in the graph and `IMPAIR <spec>` retunes it live (`gr_medium_ctl.py`), so one
    medium serves a whole matrix.  `STATS` returns one JSON line.
- *Medium fixes found on the way (the medium must not be the impairment):*
  - RX writer thread per node, batch `sendall` (the GIL starved per-chunk sends; clean runs dropped).
  - `DelayLine` ring buffer instead of GR `blocks.delay` (a large history put the graph seconds late).
  - `NodeTxSource` min_chunk 5 ms (200-sample chunks cost 154% CPU; now 19%).
- *RTD ranging (V.34 phase 2 analogue):* a PING control message precedes every PROBE; BAT v2 echoes the newest
  token and how long it was held.  RTD = answer decoded − ping queued − held, minimum over measurements.
  Before RTD is known retries back off 2× up to max_rtd_s; after, retry = max(retry_s, RTD + 1.5 s),
  linger = 2 × retry.  `IqConfig.max_rtd_s` = 15 s (satellite paths); default budget 3·max_rtd + 20 s.
- *Sweep (local medium, noise 0.02 ≈ 23 dB, 100 kB + 20 kB, 2 runs each):*
  - PASS: clean, delay 0.3 s and 2 s (RTD measured, settles in 5 / 15 s), CFO 200 Hz – 12 kHz (measured within a
    few Hz).
  - FAIL: delay 5 s (settles in 24 s, RTD right, but the first message frame is lost: each node mutes its RX while
    it transmits (hear_own=False) and the READY linger overlaps the peer's message at this RTD).
  - FAIL: every echo (even 0.1 ms @ −3 dB, inside the CP: EVM 0.16 vs 0.10 clean, 35/50 frames), every fade,
    every pink level.  Frequency-selective or time-varying channels lose a few frames per message, and with no ARQ
    one lost frame loses the message.  Step 10b says why.
  - FAIL: impulse (even 1/s @ 0.5 for 0.6 ms: 48/50 frames), dropout (0.2/s × 5 ms: 48/50), a CW tone in band
    (0.05 and 0.2: the BAT exchange never settles), vhf-mobile and hf-moderate (never settle).  PASS: a tone at DC
    (the DC carrier is nulled).  Every frame-loss failure here is the missing ARQ: 2-7 frames of 50 lost.

**Step 10b design notes (targeted BAT-slot impairments).**

- *Why:* a random impairment usually lands on carriers the table loads lightly, and proves nothing.
  `ne0k-gui/modemgui/engine/slotimpair.py` aims: once the link has agreed its table, the TRANSMITTER blocks
  (60 dB or `@DEPTH`) or fuzzes (noise at `@SNR` in the slot) carriers picked FROM THAT TABLE, on one frame
  class only.  Only the TX side knows which table and which frame class a burst is, so that is where the target
  can be exact; to the far end it is the channel.
  - Roles: `hot` (most bits), `cold` (fewest loaded), `run` (adjacent window with the most bits), `spread` (even
    through the loaded carriers), `pilot`, `idle` (0-bit: negative control).  Ties go to the lower carrier.
  - Classes: `data` (OfdmLink DATA/STREAM and EC chunks; IqLink MSG), `ctrl` (OfdmLink ACK/GO; IqLink
    BAT/READY/PING after the agreement), `any`.
  - Spec `ROLE:N:MODE[@LEVEL]:CLASS[,...]`, e.g. `hot:4:block:data`, `pilot:all:fuzz@6:ctrl`.
  - A cell whose rule filtered no burst is **NOHIT**, never PASS (e.g. ctrl on oneway: no control frame exists
    after the agreement).
  - Hooks: `iq-p2p --slot-impair`, `tests/slot_cell.py` (two OfdmLinks on loop buses), `attach_ofdm` /
    `attach_iq`.  Unit tests: tests/test_slotimpair.py.
- *Complex (I/Q), local medium, 20 kB each way:*
  - data: 1 hot carrier blocked (4 of 448 bits/sym) → FAIL (8/10 frames; also with only a 10 or 20 dB dip);
    4 hot, 4 cold, run 8, pilots blocked, 4 pilots fuzzed at 0 dB → FAIL; every data carrier fuzzed at 20 dB SNR
    → PASS, at 15 dB → FAIL; pilots fuzzed at 20 dB → PASS, 10 dB → FAIL.
  - ctrl: 1 hot carrier blocked on control frames → PASS; 4 hot or all pilots → the exchange never settles
    (base-table BATs fail RS, 65 s).
  - idle: a CW tone on one carrier (to make a 0-bit carrier) breaks the link before the rule matters: at
    0.05 (−12 dB) every BAT fails RS; at 0.1 the probed table makes `ofdm_cxmod_workspace` refuse it (NativeError).
  - Group run `gnuradio-iq-slot-2node` (2026-11-14, R xp5kvm13, A docker manager, B xp5kvm14), 100 kB / 20 kB:
    4/16 PASS (clean, hot 1 at 10 dB, all fuzzed at 25 dB, ctrl hot 1); the rest as above.  Edges move between
    runs: hot 1 at 10 dB failed locally and passed here, all at 20 dB passed locally and failed here (23/50 frames).
    The group ends "aborted" by design of the group runner: a member that finishes red aborts it after the cells.
  - Interleave depth covers one lost SYMBOL (`ofdm_interleave_geometry`), not a lost CARRIER: one dead carrier
    is a few byte errors in every codeword, at the edge of t = 8, so a few frames per message fail.
- *Simple (audio, 2 bits/carrier max, 64 carriers), loop buses, 600 B:*

  | rule (data class) | oneway | stream | full (Python EC) | full (C cdx) |
  |---|---|---|---|---|
  | hot 4 blocked | PASS | PASS | FAIL | FAIL |
  | run 16 blocked (25% of bits) | PASS | PASS | FAIL | FAIL |
  | run 32 / spread 16 blocked | FAIL | FAIL | FAIL | FAIL |
  | all fuzzed at 9 dB / 6 dB | FAIL | FAIL | PASS | FAIL |
  | 2 or 4 pilots blocked | PASS | PASS | FAIL | FAIL |
  | all pilots blocked | FAIL | FAIL | – | – |
  | all pilots fuzzed at 6 dB (0 dB) | PASS (PASS) | PASS (FAIL) | PASS | PASS |
  | idle (0-bit) blocked | PASS | PASS | – | – |
  | ctrl, hot 4 blocked | NOHIT | NOHIT | NOHIT (ACKs ride in chunks) | PASS (ACK-only frames) |

  - An adjacent run survives where the same number spread out does not: bytes map to adjacent carriers.
  - Full duplex has half the carriers per direction, so 4 carriers is 15% of the table.
- *Known breaks the tests expose (not fixed here):*
  - half mode on loop buses fails clean (also at HEAD): the 255 B window frame airs 4.4 s in a 3 s usable window,
    the far end cuts it (CRC fail) and the window numbering drifts (`test_payload_both_ways_binary_safe` red).
  - standalone `arq` never transmits: `send()` queues for the EC window, but arq runs `_listen_full` only.
  - complex: no ARQ, no re-probe, no erasure decoding (a carrier the pilots/CE show dead could be marked as an
    erasure: RS corrects 16 erasures vs 8 errors).
- *Not covered:* broadcast/streaming (no return path: FEC headroom alone); mesh announce frames; a slot rule that
  switches on mid-transfer (it arms at the agreement).

## Step 10c design notes (erasure decoding, ARQ, missing features)

- *Erasure decoding (ofdm_cxdem.c, additive; the real path's ofdm_frame_decode_il is untouched).*
  - Every data cell gets a score r = squared decision distance / (d_min/2)^2 of its constellation (0 on the
    point, 1 on the boundary); per loaded carrier the frame mean of r and of |H|^2.  A carrier whose mean |H|^2
    is 20 dB under the loaded mean (`CXDEM_DEAD_RATIO`) is dead: every cell on it scores 4.  A wire byte scores
    its worst bit's cell (or that cell's carrier mean, if worse).
  - De-interleave + RS as before; a codeword the errors-only decode cannot correct is retried with
    `rs255_decode_era` (already in rs255.c, reedsolo-exact) on its E least reliable bytes above score 0.25,
    E = 1, 2, ... .  Accepted only when 2e + E <= ecc - 4: four parity bytes stay unused as the miscorrection check
    (a wrong-table trial in the multi-table receiver must still fail, not decode to garbage).
  - Clean frames never reach the retry, so every selftest oracle and errata count is unchanged (CX PASS).
  - Offline, 64-QAM table, 32 dB, 20 frames: 2 dead hot carriers 14/20 → 20/20; 4 dead → 0 → 1/20 (12 erasures
    per codeword is the budget: a dead 6-bit carrier costs ~1 byte per symbol in every codeword).  So erasures
    alone fix one or two notches; past that the table has to change, which is what the ARQ's update does.
  - `CxFrame.erased` (bytes decoded as erasures) and `CxFrame.quality` (score, |H|^2 rel per data carrier, also
    for a frame RS failed) come back from the receiver; `IqConfig.erasures` (default on).
- *Selective-repeat ARQ (iq_link.py).*
  - `send_reliable(data)`: the `send_message` frames, then POLL{id, seq0, n, table} at the base table.  The
    peer's responder thread (every IqLink runs one; it keeps every data chunk by seq) answers STAT{id, bitmap,
    flags[, table]}.  Missing frames go again at their old seq (`send(..., keep_seq=True)`), then POLL again,
    up to `IqConfig.arq_rounds` (8).  A lost POLL/STAT is just another POLL round, waited for RTD + 2 control
    frames + 1.5 s (`rtd_s` from negotiate, else 3 s).  Complete + CRC → `receive_reliable()` / `on_message`;
    a repeated POLL for a delivered id is answered complete again.
  - *Table update in the STAT:* the receiver grades the sender's current table from the median per-carrier
    quality of the frames it demodulated at it (ok or not): dead (|H|^2 10 dB under) → 0 bits; r > 0.5 → two
    steps down; else the widest width whose floor (4/7/14/21 dB) + 3 dB (`ARQ_MARGIN_DB`: measured on the data,
    so less than the probe's 7) fits SNR = 1/(r d_min^2/4); never up.  It adds the new table to its RX set first
    (CxReceiver tries it first, keeps the old ones), the sender switches with `set_tx_bat`.  Control stays at
    the base table, so a bad data table never takes the control channel with it.
  - Why the update matters: a resend on the same table hits the same dead carriers and fails the same way.
  - Local pair medium (32 dB, ±2.1 kHz CFO, 20 kB = 10 frames), rule on the sender's data frames:

    | rule | rounds | resent | table updates | bits/sym after | result |
    |---|---|---|---|---|---|
    | none | 1 | 0 | 0 | 672 | PASS 1.9 s |
    | hot:4:block:data:pin | 2 | 7 | 1 | 604 (dead slots now 0 bits) | PASS 3.8 s |
    | run:8:block:data:pin | 2 | 10 | 1 | 612 | PASS 3.9 s |
    | spread:8:fuzz@6:data:pin | 2 | 10 | 1 | 490 | PASS 4.1 s |
    | hot:all:fuzz@20:data | 2 | 10 | 1 | 376 (20 dB in-slot: 16-QAM/QPSK) | PASS 4.4 s |
    | hot:1:block:data (tracking) | 3 | 8 | 2 | 589 | PASS 5.4 s |
    | hot:4:block:ctrl | 1 | 0 | 0 | 672 (erasures carry the QPSK control) | PASS 1.9 s |
    | hot:4:block:data (tracking) | 8 | 72 | 8 | 356 | FAIL: the rule re-aims at the 4 hottest of every new table |

  - Slot rule `:pin` (slotimpair): slots picked once from the first agreed table and kept (a fixed channel fault);
    without it a rule re-aims at every new table, an adversary no table update outruns (data-hot4 keeps that case).
  - `iq-p2p` uses `send_reliable` / `receive_reliable` by default; `--no-arq` is the old fire-and-forget,
    `--no-erasures` the errors-only decode, `--arq-rounds N`.
- *Bug fixed on the way:* an in-band CW tone ~6 dB under the signal makes the probe report ~-5 dB on EVERY
  carrier (CFO off by 80 Hz, image -9 dB), so `ofdm_allocate` gives an all-zero table and cxmod refused it
  (the NativeError of the 10b idle cells).  negotiate now ignores a probe with an empty table and `_parse_bat`
  refuses one; the probe's tone sensitivity itself is open (below).
- *Cluster (gnuradio-iq-slot-2node, ARQ + erasures on; group ..._group_934c, A docker, B xp5kvm14, R xp5kvm13;
  100 kB A→B, 20 kB back):* **10/19 PASS** (step 10b: 4/16).
  - PASS: clean (146 kbit/s wall), ctrl-hot1, data-hot1 (2 rounds, 13 resent, 1 update), hot1@10 dB, all-f25,
    all-f20 (2 rounds, 448 → 224 bits/sym), spread8-f12, and every pinned fault: hot4-pin (2 rounds, 50 resent,
    448 → 419), run8-pin (448 → 403), spread8-f6-pin (3 rounds, → 344).
  - FAIL, tracking adversary (by design): data-hot4, data-run8, data-cold4: 8 rounds, 8 updates, the rule re-aims
    at each new table.
  - FAIL, pilots: data-pilots (every pilot blocked), data-pilot4-f0: phase tracking gone (missing feature 6).
  - FAIL, control during negotiation: ctrl-hot4, ctrl-pilots (both ends filter their BAT/READY; base-table
    erasures are not enough once both directions lose them); ctrl-idle / data-idle: the tone breaks the probe
    (missing feature 5).
  - The group ends "aborted" because a red member finishes the group (as in 10b), after all cells ran.
- *Missing features (assessment, most valuable first):*
  1. **Inner FEC / soft decisions.**  RS(255, 239)-style hard decisions only: no LLRs, no convolutional / LDPC
     code.  A rate-3/4 LDPC or a convolutional code with bit interleaving typically buys 4-6 dB, i.e. one or two
     constellation steps at the same margin, and would carry the fades and pink-noise cells that fail now.
  2. **Channel tracking inside a frame.**  H comes from the CE and is replaced at each SYNC; between them only a
     common phase + slope (pilots) is tracked.  Fading (`fade`, `vhf-mobile`, `hf-moderate`) needs per-carrier
     amplitude/phase tracking: pilot interpolation across frequency and time, or decision-directed updates.
  3. **Multipath beyond the CP margin.**  The window sits at taper + CP/2 with taper <= CP/4, so an echo longer
     than ~CP/4 is ISI (every `echo` cell fails).  Needs a long-CP profile (`IQ_PROFILES`) chosen from the probe's
     delay spread, or a timing search that moves the window to the channel's energy.
  4. **Table up-shift / periodic re-probe.**  The ARQ table update only lowers bits; nothing raises them when
     the channel recovers until a new `negotiate`.  A re-probe on a timer or on sustained clean frames.
  5. **Narrowband interference robustness.**  One CW tone 6 dB down wrecks the probe (above) and sync; a notch
     (detect + excise strong bins) before sync/probe, and an SNR estimate per carrier that a single bin cannot
     poison.  Same family: 4 notched carriers raise far carriers' score 0.023 → 0.12 (CFO/timing bias), worth a
     look at the sync's estimate under notches.
  6. **Pilot loss fallback.**  All pilots blocked = every frame lost (phase tracking gone); decision-directed
     phase tracking from the data cells would survive it.
  7. **Pipelined ARQ / flow control.**  `send_reliable` is one message in flight (POLL after the whole message):
     fine for bulk, slow for many small messages at a long RTD.  A window of outstanding messages and
     piggybacked STAT on data.
  8. **RX during own TX.**  `hear_own=False` mutes RX while transmitting, so at 5 s RTD the first frame after a
     READY linger overlap is lost (delay5000); ARQ now resends it, but a full-duplex echo canceller (as the
     Pluto 1-pair mock) or TX scheduling would avoid the loss.
  9. **Station ID on I/Q.**  The real path sends CW ID audio; the I/Q path sends none (by design: it is audio).
     A ham-band Pluto link needs a CW (OOK carrier) or FM ID burst in I/Q before step 11 goes on air.
  10. **Mesh / multi-node MAC** (the real path's mesh_mac CSMA), AGC / level control and PAPR reduction:
      not needed for the 2-node bench, needed for SDR deployment.

## Step 10d design notes (the 10c missing features)

Defaults moved with it: `IqConfig.fec = "2/3"`, `track`, `excise`, `duplex = "full"`, `reprobe_s = 60`.  The C
additions are off unless asked for (`*_fec(..., OFDM_CXFEC_NONE)` = the old frame, tracking flags 0, window -1),
so every ofdm_cx_selftest oracle is unchanged (CX PASS; the new c89 file is in its list).

1. *Inner FEC (ofdm_cxfec.c/.h; cxmod/cxdem `*_workspace_fec` / `*_init_fec`).*  Wire bytes → scrambler → K=7
   (133, 171) conv code with 6 tail bits → puncture (1/2, 2/3 = [c0 c1][c0 -], 3/4 = [c0 c1][c0 -][- c1]) → bit
   interleave (coded bit i → frame position i·P mod n, P >= bps + bps/3 + 1 coprime: neighbours one symbol and a
   third of the band apart) → mapper; the bits past the code carry the scrambler's continuation.  RX: max-log LLRs
   weighted by |He|^2 (a dead carrier is an erasure for free), de-interleave, 0 for punctured bits, soft Viterbi
   (64 states, decisions as two 32-bit words per step, terminated traceback), descramble, RS + the GMD retry
   (byte scores from the cell of each info bit's first coded bit).  Standalone BER check matches K=7 textbook
   curves (rate 1/2 error-free by 4 dB Eb/N0).  Frame (iq, 64-QAM, 4 frames): RS-only dies at 20 dB, 2/3 decodes
   at 16 dB; the probe spends `FEC_GAIN_DB` (4 / 3 / 2.5 dB) of `snr_margin_db`.  Cost: air time / rate.
2. *Tracking between SYNCs (`ofdm_cxdem_set_tracking`, TRACK_H).*  After the decisions, H moves toward Y / (decision
   · e^{j(a+b bin)}) on cells under score 0.3 and on the pilots (mu 0.15).  Fade 5 Hz bench: 2/6 → 3/6 frames
   (helps, not a cure: the SYNC spacing still bounds it).
3. *Pilot loss (TRACK_PILOTS).*  Pilot coherence (weighted resultant after a + b·bin) under 0.6 → the previous
   symbol's phase; then a decision-directed least-squares fit of da + db·bin on cells under score 0.5 and a
   re-decision.  Pilots fuzzed at 6 dB: 1/4 → 4/4 frames, EVM 0.21 → 0.14; phase noise EVM -30 %.  Pilots BLOCKED
   still fail: the notch comb (every 8th carrier) leaves EVM 0.25 on the data carriers themselves with or without
   tracking (the brick-wall block also damages the neighbours and the SC/CE), so it is not a phase problem.
4. *Echo.*  `ofdm_cxdem_set_window` (late = cp_len - 2: echo tolerance cp - 2T - 2 instead of cp/2 - T) and two
   profiles: `iq-echo` (CP 0.5, late; ~470 us) and `iq-hf` (n_fft 512, CP 0.5, late; ~1.1 ms, 390 Hz carriers).
   With FEC 2/3 even the default profile now decodes 50 + 200 us and 1 ms @ -15 dB echoes (bench, 16-QAM, 30 dB).
   Profiles are PHY: both ends pass the same `--profile`.  Not done: choosing the profile from the probe's delay
   spread.
5. *Up-shift / re-probe (iq_link ARQ).*  A clean message's STAT may carry an up-shift (one step on a loaded carrier
   whose data-measured SNR clears the next floor + 3 dB margin + 3 dB hysteresis, score < 0.1).  STAT flag 8 asks the
   sender for a probe every `reprobe_s` (60 s), or after reprobe_s/4 once a table update left carriers at 0 bits;
   the sender probes, the receiver's next STAT carries the probe's table (dead carriers come back).
6. *Narrowband interference (`ofdm_cx.ToneExciser`, `IqConfig.excise`).*  1024-point spectra per block, per-bin
   median over ~1.3 s vs the 75th percentile of the +-16-bin neighbourhood, 12 dB → notch (513-tap FIR from the
   mask, ~1 carrier wide, overlap-save; always a fixed 256-sample delay so a new notch never shifts the timebase;
   `filter_snippet` for the probe's history).  The 10b tone (0.1 at 19531 Hz) is found and notched, and the probe
   grades the channel again instead of -5 dB everywhere.
7. *Pipelined ARQ (`IqLink.send_window(messages, window=4)`).*  Up to `window` messages in flight, each with its
   POLL; STATs handled per message id; per-message re-POLL on timeout.  Not yet used by iq-p2p (one message each way).
8. *RX during own TX (`IqConfig.duplex`).*  `full` (default; the GR medium is mix-minus) keeps RX blocks captured
   while we transmit, so a long-RTD peer's frames that overlap our READY linger / STAT are no longer zeroed
   (the delay5000 loss).  `half` = the old mute, for a radio that hears itself.
9. *Station ID (`IqConfig.callsign`, `ident_s`, `ident_wpm`, `ident_hz`; `iq_link.cw_iq`, `IqLink.send_ident`).*  CW as an
   on/off keyed carrier at +25 kHz, 5 ms raised-cosine edges, at the frames' RMS, before the first burst and every
   10 min (FCC 97.119).  Not yet: the ID at the end of a contact (call `send_ident()` before stopping).
10. *Shared channel and level.*  `lbt` (wait until the last block is within 6 dB of the floor = 10th percentile of
    recent block powers, + 0-50 ms backoff, up to `lbt_max_s`), `papr_clip_db` (clip at N dB over RMS, mask back to
    the band, 2 passes), `IqLink.level` (RX RMS dBFS, peak, clipped samples).  Not done: a mesh MAC for complex
    (the real path's mesh_mac CSMA slots / ACK rules) and hardware AGC (Pluto gain control belongs to step 11).

CLI: `iq-*` take `--fec --window --duplex --no-track --no-excise --lbt --papr-clip --callsign --reprobe`.

*Cluster (gnuradio-iq-impair-2node, group ..._group_bc3a, all defaults: FEC 2/3, ARQ, erasures, tracking, exciser,
full duplex; 100 kB A→B, 20 kB back):* **13/14 PASS** (step 10: echo, fade, pink, impulse, dropout, tone, the
satellite delay and both channel presets all failed).  clean 112 kbit/s wall; cfo, fade, impulse, pink, tone,
lat-acoustic / g711 / satellite (46 kbit/s at the 5 s delay) all first round; dropout and echo one ARQ round + one
table update; echo-isi (1 ms) first round at 304 bits/sym; vhf-mobile passes but slowly (7 rounds, table 448 → 24
bits/sym: the fades drive the down-shifts faster than up-shift / re-probe bring them back).  FAIL: hf-moderate
(1 ms echo at -3 dB + fade + pink + impulse) on the default profile: next to try with `--profile iq-hf`.

## Step 10e notes (fading channels: hf-moderate on iq-hf, vhf-mobile)

Started as "try hf-moderate on the iq-hf profile".  The impair matrix cells now carry `profile` (passed as
`--profile` to iq-p2p) and two cells run the long-echo channels on `iq-hf`: `echo-isi-hf`, `hf-moderate-hf`.

What broke on the fading channels and what changed (engine/iq_link.py, engine/ofdm_cx.py, cli.py):

1. *Control frames were full 2040 B QPSK frames* (0.24 s on iq-hf).  A two-ray channel whose rays fade apart
   (hf-moderate = echo 1 ms @ -3 dB + 0.5 Hz Rayleigh per ray) moves its notches inside one frame: on the bench
   (scratchpad ctl_bench.py) control frames decoded 9/20 vs data 15/20; each part alone 20/20.  Now
   `IqConfig.ctl_chunk_bytes` (512, 0 = old) sizes the base-table control frames (PING, BAT, READY, POLL, STAT),
   and `CxReceiver.set_ctl(modem)` tries that modem on every hit beside the data tables (`tables` excludes it).
   Bench: control 9/20 -> 21/30.  Both ends must agree (flag day for complex links).
2. *A probe in a fade sank the table.*  ARQ re-probe once took 355 -> 5 bits/sym; vhf-mobile negotiation took a
   0.6 dB probe (3 bits/sym).  Now: negotiate keeps the per-carrier best of the probes it hears; a side whose
   TX table is weak (< 1 bit/carrier) keeps probing and holds READY until half the budget, and a peer that is
   already done answers a new probe with the better BAT.  An ARQ re-probe only restores carriers at 0 (it used to
   raise every carrier and grading pulled them back: 205 -> 384 -> 88).  `_grade_table` lowers a carrier at most
   one step per update (dead carriers and r > 0.5 excepted).
3. *POLL/STAT lost to fades ate the ARQ rounds.*  An unanswered POLL no longer counts as a round: re-POLL, up to
   2 x arq_rounds in a row; `send_reliable(deadline=)` stops at iq-p2p's `--seconds` (it used to be killed by the
   step timeout).
4. *The last STAT.*  The initiator passed and closed while its final STAT was lost, leaving the replier polling
   into silence (hf-moderate-hf B FAIL with A PASS).  `IqLink.linger()` keeps answering POLLs until none comes
   for two STAT waits; iq-p2p calls it after the reply.

*Cluster (group ..._group_1fe6):* **15/16 cells PASS.**  echo-isi-hf 906 bits/sym first round (90 kbit/s wall,
vs 52-79 on iq); hf-moderate-hf PASS (4 rounds, 3 table updates, 17 kbit/s wall); vhf-mobile PASS (5 rounds,
table ends at 18 bits/sym: grading still drifts down under 5 Hz fades).  FAIL: hf-moderate on the default
profile (CP shorter than the 1 ms echo: use iq-hf, or pick the profile from the measured delay spread, not done).
The group still ends red: R's medium-stats step flags TX overflow on node 1 (B, xp5kvm14; 222-5133 samples per
run, seen before 10e too), a medium/host issue, not the modem.  iq-hf is the wrong profile for vhf-mobile
(bench: data 5/30): long symbols do not follow a 5 Hz fade.

*Node-1 medium overflow (2026-11-14, after 10e).*  Not the modem: the iq connector writes exactly one TX
sample per RX sample plus a one-off 0.5 s lead, so a node's TX queue in the medium is lead + (how far the
OTHER node's source, which paces this node's RX, is ahead of this node's source) - RX in flight.  The medium
routinely runs ~250-300 ms late per source thread (GIL/scheduler), so the queue sat at 0.7-1.0 s against
--max-buffer-s 1.0 and the faster-answering node (B on a KVM; A sat behind the docker bridge) tipped over by a few ms.  Fix in
gr_rf_medium.py: the cap counts only TX ahead of the air clock (max-buffer-s PAST what the source already
owes), and STATS gains tx_queue_max_ms / src_lag_max_ms.  Group ..._group_235f (R on xp5kvm15): overflow 0
on both nodes, R 56/56 green, but tx_queue_max 995/998 ms stayed just under the old 1 s cap, so that run did
not exercise the new allowance; it shows how thin the old margin was.  Same run: 14/16 cells
(hf-moderate-hf failed this time, ARQ 5 rounds / 16 unanswered POLLs: still flaky under the 0.5 Hz fade).

*hf-moderate-hf flakiness (2026-11-15).*  Three failure modes, one per run, all control-plane:
1. **Thin table made the receiver deaf.**  CxReceiver searched only once one frame of its LONGEST table
   was buffered and decoded no hit before that much followed it; ARQ keeps the last 4 tables, so one thin
   graded table (84 bits/sym = 1.25 s frames, 8 = 13 s) delayed every POLL past the sender's wait.  Now it
   searches after one SHORTEST frame and tries each table as soon as that table's frame fits; a hit waiting
   for longer tables is held (later hits still decode; a later decode ends the hold) and its first failed
   attempt is still reported (grading needs bad frames: dropping them stalled table updates).
   Test: test_thin_table_does_not_delay_short_frames.
2. **Grading collapse.**  _grade_table no longer goes below 1 bit per data carrier on average (882 -> 726 ->
   444 -> 84 -> ... under the fade); below that the table stays and ARQ resends at it.
3. **Control frames lost in fades.**  One 512 B control frame got through ~65 % of the time; bench: either of
   two copies 0.6 s apart ~90 % (0.3 s apart still correlated).  IqConfig.ctl_repeat_s (iq-hf 0.6) sends
   POLL / STAT / BAT / READY twice with the same seq; the receiver drops the copy by (magic, seq).
   Test: test_arq_repeated_control_frames_are_acted_on_once.
4. **Lost READY after the peer stopped lingering.**  The initiator re-sent BATs for 65 s to a peer that had
   left negotiate; the background responder now answers a post-negotiation BAT (have=1) with READY.
   Test: test_late_bat_after_negotiate_gets_a_ready.
Cluster ..._group_6ddf: **15/16 PASS**, hf-moderate-hf 100 kB in 30.7 s wall (26 kbit/s, was 88-111 s), reply
OK; only hf-moderate on the default profile fails (CP < echo).  R green, overflow 0.  Repeat run ..._group_26e5
(A on the docker manager this time): 15/16 again, hf-moderate-hf 100 kB in 38.4 s wall, reply OK.

## Step 10f notes (decoder-aided channel re-estimation, prototype 2026-11-15)

Prompted by Xiong/Gan/Zhu 2010 (FH/OFDM, iterative noncoherent sequence detection + SOVA): the
DPSK / noncoherent part does not fit a bit-loaded coherent modem, but its iterate-between-demod-and-decoder
idea does, as the coherent form of turbo channel estimation.

- *What it does (ofdm_cxdem.c `iter_pass`, inner code only).* The first pass keeps every window's Y and
  the cells known before decoding (CE, SYNCs, pilots).  If RS fails: every codeword that DID decode is
  RS-re-encoded back into the wire (so its cells become exact references), the wire is re-encoded
  exactly as ofdm_cxmod_frame (scramble, K=7 encode, interleave, scrambler continuation, map via
  `inv[width][value]`); per loaded carrier, H(t) = sum Y X* / sum |X|^2 over symbols t-W..t+W of that
  carrier, **t itself excluded** (its own possibly wrong decision must not vouch for itself); one common
  phase per symbol from all its cells; re-decide, LLRs, Viterbi, byte scores, RS + GMD.  Up to
  `passes` rounds.  Frames that decode first time are untouched, so it cannot lose a frame the old
  path got, and the RS (with GMD's 4-byte slack) still guards against miscorrection.
- *Cost.* Workspace + body x n_occupied x 2 complex whenever fec != none.  CPU only on failed frames:
  bench decode wall +10-90 % on cells that fail a lot (vhf-mobile 0.66 -> 1.27 s for 48 frames), ~0 on
  clean ones.
- *C ABI.* Additive: `ofdm_cxdem_set_iterate(d, passes, half_win)`, `ofdm_cxdem_iter_passes(d)`;
  ofdm_cxdem_out unchanged.  cx_decode now walks all codewords when iterate is on (to harvest the good
  ones) and writes corrected codewords back into `ofdm_cxdem_wire`.  Off = byte-identical behaviour:
  ofdm_cx_selftest CX PASS, tests/test_iq.py + test_slotimpair.py pass.
- *Offline bench* (scratch iter_bench.py: impair.py PathImpair/RxImpair on the encoded frames, nominal
  per-carrier SNR, fixed table, 6 seeds x 8 frames, every rescued payload compared with the data):

  | cell | off | it2 w2 | it3 w4 | it3 w6 | it4 w8 |
  |---|---|---|---|---|---|
  | AWGN, fade 0.3 Hz, fade 5 Hz QPSK | 48/48 | 48/48 | 48/48 | 48/48 | 48/48 |
  | fade 2 Hz Rayleigh, 16-QAM | 40/48 | 46/48 | 48/48 | 48/48 | 48/48 |
  | fade 5 Hz K=3 dB, 16-QAM | 42/48 | 46/48 | 48/48 | 48/48 | 48/48 |
  | vhf-mobile, 16-QAM | 28/48 | 35/48 | 40/48 | 41/48 | 42/48 |
  | hf-moderate on iq-hf, 16-QAM | 24/36 seen | 26 | 27 | 27 | 27 |
  | hf-moderate on iq-hf, QPSK; echo-isi-hf | unchanged | | | | |

  Fast fades between SYNCs were estimation-limited and are now mostly recovered.  HF losses are not:
  failed HF frames have EVM 0.23-0.35 with a 0.5 Hz fade (H barely moves in a frame); they lose energy
  in the echo notches, which re-estimation cannot add back (deeper interleaving / ARQ territory), and
  a quarter of HF frames are never found (sync).  Best setting in the bench: passes 3, half_win 4-8.
- *Test:* tests/test_iq.py `test_iterate_rescues_frames_on_a_fast_selective_fade` (echo 20 samples @ 0.5
  with its own 6 Hz Doppler: 0/6 frames off, 6/6 at iterate 3 / win 4, payload checked).
- *Next:* cluster `gnuradio-iq-impair-2node` with `--iterate 3 --iterate-win 4` vs without (vhf-mobile,
  fade cells; ARQ rounds and wall throughput, not just frames); then decide the default.  Not done: soft
  (SOVA/BCJR) reliabilities for the reference cells, mixing in neighbouring carriers' cells (wrong under
  a long echo), applying it to control frames at the base table.

## Reuse map (real-path code that complex mode calls unchanged)

`ofdm_interleave_geometry`, `ofdm_data_syms_per_pkt`, `ofdm_n_syncs`, `ofdm_frame_min_gap`,
`ofdm_prbs`, `ofdm_known_bpsk`, `ofdm_mod_table(s_flat)`, `ofdm_bat_widths`, `ofdm_allocate`, `rs255.c`.
They read only byte and carrier counts from `ofdm_cfg`, which `ofdm_cxcfg.c` fills with the same
meaning (the selftest checks frame geometry agrees for equal data-carrier counts).

Never pass `ofdm_cxcfg.c` to `ofdm_sym_*`, `ofdm_mod_*`, `ofdm_dem_*`, `ofdm_acq_*`: they index
`bin_lo + i` into a half spectrum and a negative bin_lo is out of bounds.

## CPU (estimates; step 8 measured 13–19 % of one core per node at 200 ksps, see above)

Per second of signal the cost goes as fs × log2(N) at a fixed carrier spacing. Against today's
48 kHz real modem: 200 ksps complex ≈ 4–8× (complex FFT ≈ 2× a real FFT of the same N); 8 ksps
telephone audio ≈ 1/6. The likely hot spot is not the FFT but the per-sample loops (SC correlation,
echo canceller taps ∝ fs, Python block overhead); keep them in C and use 20 ms+ blocks.

## Test

    python3 audiotestserver/tests/OFDM/ofdm_cx_selftest.py      # CX PASS / CX FAIL
