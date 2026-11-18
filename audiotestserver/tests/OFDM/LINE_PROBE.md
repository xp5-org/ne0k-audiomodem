# Staged line probe (plan, 2026-11-18)

## Why

Today's probe (`ChannelProbe` in `ofdm_modem.py`, C port `c/ofdm_probe.c`) sends:

- an SC preamble,
- one CE symbol,
- N periodic BPSK symbols,
- M non-periodic BPSK symbols, all carriers loaded.

It reports **one number per carrier**: SNR = |mean H|^2 / var(H). Every impairment lands in that one variance:

- additive noise,
- phase noise and timing jitter,
- intermodulation from the other carriers,
- codec damage (A2DP SBC/AAC),
- ISI.

The BT-Car -> FM -> RTL-SDR link shows an SNR that slopes down toward 9 kHz. From that one number we cannot tell which of these causes it is:

| cause | signature | fix |
|---|---|---|
| frequency response tilt (pre-emphasis mismatch, codec low-pass) | gain falls with f; the noise floor stays flat | TX EQ / pre-emphasis (calibration engine TODO) |
| phase noise / timing jitter (BT clock, resampler) | phase error grows ~ with f, so the SNR loss grows ~f^2; the error is **correlated** across tones | per-symbol common-phase and phase-slope tracking from pilots, shorter symbols. Not EQ: EQ cannot help |
| nonlinearity (FM limiter, deviation clipping) | harmonics and IMD products; the loss grows with TX level | back off the level, crest-factor control |
| codec | tones quantised or dropped under masking; level-dependent and spectrum-dependent | stay in the codec's comfortable region |

The probe has to be able to tell these apart. It does that by building up from the simplest stimulus, the way V.34's L1/L2 line probing does.

## Stages

One probe sequence, sent as one burst after the key-up lead (the BT stream must already be awake, see the README chunk-2 TODO). The receiver analyses each stage separately.

### 0. Quiet: the floor under no load (~0.5 s)
The carrier is up with zero modulation; for BT this means a silent A2DP stream.

- Measure: noise PSD per bin.

This is the additive floor. Every later stage is compared against it.

### 1. Single tones (a stepped sweep, ~100-200 ms per tone)
One tone at a time across the band (for example on the OFDM carrier grid, every k-th carrier). Each tone gets the full crest budget, so this is the best-case SNR. Per tone:

- **Gain and phase.** This gives |H(f)| and the group delay. A slope in gain with a flat floor means EQ is needed.
- **Frequency error.** This gives the clock offset in ppm, which should scale with f.
- **Phase-noise skirt and jitter series.** Fit a frequency and demodulate the tone. The instantaneous phase and amplitude give:
  - PM jitter RMS and its spectrum,
  - AM jitter, kept separate from the PM.

  If the PM RMS grows linearly with f, the cause is timing jitter. If it is flat, the cause is oscillator or additive.
- **Harmonics at 2f and 3f.** This gives the THD.
- **Level steps.** Repeat a few tones at -6, -3 and 0 dB relative to the full level. This finds the compression point of the limiter or codec.

### 2. Tone pairs
- **Close pairs (f, f+Δ).** Measure the IMD3 products 2f1-f2 and 2f2-f1, at the same level steps as stage 1.
- **Far pairs (one low tone, one high tone).** Compare the two tones' phase-jitter series:
  - Correlation, with the high tone's jitter about f_hi/f_lo times the low tone's, means common timing jitter. Pilot tracking fixes it.
  - Uncorrelated jitter means independent impairment.

  This is the direct answer to "is the SNR lower at high tones because of phase noise".

### 3. Multi-tone comb (V.34 L1/L2), then phase rotations

**What V.34 does.** Source: "T-REC-V.34-199802-I!!PDF-E-2.pdf page 31", section 10.1.2.4 and Table 17.

- **L1:**
  - A periodic sum of cosines, cos(2πft + ϕ), repeating at 150 Hz ± 0.01%.
  - The tones sit on a 150 Hz grid from 150 Hz to 3750 Hz.
  - The tones at 900, 1200, 1800 and 2400 Hz are **omitted**.
  - L1 lasts 160 ms (24 repetitions) at **+6 dB** above nominal power.
- **L2:** the same signal as L1 at the **nominal** level, for at most 550 ms plus a round trip.
- **Table 17 initial phases.** The tones at 300, 1650, 2250, 2700, 3000, 3150, 3300 and 3450 Hz start at 180°. All the others start at 0°:

  | f (Hz) | ϕ | f (Hz) | ϕ | f (Hz) | ϕ |
  |---|---|---|---|---|---|
  | 150 | 0 | 1500 | 0 | 2850 | 0 |
  | 300 | 180 | 1650 | 180 | 3000 | 180 |
  | 450 | 0 | 1950 | 0 | 3150 | 180 |
  | 600 | 0 | 2100 | 0 | 3300 | 180 |
  | 750 | 0 | 2250 | 180 | 3450 | 180 |
  | 1050 | 0 | 2550 | 0 | 3600 | 0 |
  | 1350 | 0 | 2700 | 180 | 3750 | 0 |

  The 180° entries are fixed **initial phases**, not reversals in time. The spec gives no reason for them. Presumably they keep the comb's crest factor low; the generator will measure it.

**Why this suits our analyzer:**
- **One period is an integer number of samples.** It is 320 samples at 48 kHz and 294 at 44.1 kHz (the BT-Car rate). So a window of exactly one period puts each tone on its own FFT bin, with no window leakage.
- **Repetitions average coherently.** Averaging the periods gives H(f) at every tone, and the spread between periods gives noise and jitter per tone.
- **The omitted bins hold no signal.** Whatever lands there is noise, harmonics and IMD under multi-tone load. Note that 900 = 2×450 and 1200 = 2×600, so second harmonics of present tones fall into two of the empty bins. A third-harmonic check (1800 = 3×600) also fits.
- **L1 at +6 dB versus L2 at nominal is a built-in level step.** If the omitted-bin power rises faster than the tone power between L2 and L1, the cause is nonlinearity, not noise.

**One comb, driven by the UI parameters (no fixed modes).** V.34 is a guideline only: the audio and radio paths will add complications that a phone line does not have.

The code is in `ne0k-gui/modemgui/engine/lineprobe.py` and the tests are in `tests/test_lineprobe.py`.

- **Inputs:**
  - `freq_lo` and `freq_hi`, from the OFDM band. The band may be very narrow or up to 20 kHz wide.
  - The tone spacing `spacing_hz`, default 150.
  - Every sample rate the path touches (for example 44100 on BT-Car and 48000 at the receiver).
  - The number of periods per analysis window (`window_periods`).
- **The math that guarantees no bin bleed:**
  - The period is P = fs / spacing. P must be an integer at **every** sample rate given, so the spacing must divide the gcd of the rates: gcd(48000, 44100) = 300, which allows 150, 100, 75, ... Hz. The validator rejects anything else and offers the nearest valid spacings.
  - The tones sit on bins k = f / spacing, with DC and Nyquist excluded.
  - Any window of exactly M·P samples inside the steady-state comb holds a whole number of periods of every tone. The DFT then puts tone k on bin k·M exactly, with zero leakage in exact arithmetic, whatever the window's start offset.
  - With M > 1, the M-1 bins between neighbouring tones are empty. They measure noise and the phase-noise skirt next to every tone.
- **Omitted tones** (empty grid slots):
  - Omitted where the 2nd and 3rd harmonics of low "reference" tones land, so harmonic distortion is seen on-grid.
  - Plus a spread of empty slots across the band at least every `omit_every` tones.
  - The plan records which slot catches what.
- **Phases:**
  - The default is Newman phases, φ_k = π(k-1)²/K, which keep the crest factor low.
  - An explicit table may override them. The tests use V.34 Table 17 as a reference case.
  - The crest factor is reported with the plan.
- **What leaks in a real channel:** a clock offset ε moves tone k by k·M·ε bins, which leaks into its neighbours. The analyzer:
  1. estimates ε from each tone's phase drift between periods,
  2. predicts the leakage,
  3. measures the empty bins on a residual. The residual is what remains after a least-squares fit at the corrected tone frequencies is subtracted.

  The tests inject ε analytically and check both the prediction and the residual floor.
- **On the modem's own carriers** (`lineprobe.plan_for_grid`, built on the enforced grid in `engine/ofdmgrid.py`).
  - Comb slot k is OFDM bin k·`every`: the spacing is fs·every/n_fft, and the period is n_fft/every samples, an integer because n_fft is a power of two.
  - The default window is `every` periods, which is n_fft samples: the modem's own FFT, with every tone on its carrier's bin.
  - So the line probe measures exactly the channels the data uses, in the band the engine actually runs: freq_lo and freq_hi snapped to carriers.
  - `tests/test_ofdmgrid.py` checks:
    - each tone frequency equals its carrier frequency, exactly,
    - no bleed in the n_fft window, from n_fft 32 to 4096, at 44.1 and 48 kHz.

**Our additions after the comb.** These are not part of L1/L2.

- **180° phase reversals at known instants.** Measure:
  - the reversal timing, which gives a timing reference and group delay,
  - the **transient and settling** after the abrupt change: what the codec, limiter or AGC does to a symbol edge.

  The settling measurement bears on the chunk-2 loss.
- **A slow, known phase ramp.** This checks the tracker's ability to follow a frequency shift.

### 4. Full channel (the existing OFDM probe, extended)
- Keep the periodic and non-periodic blocks.
- **Add null carriers to the measurement block**: every 8th carrier off, with the pattern rotating per symbol, so every bin is empty sometimes.
  - Null power under full load, minus the stage-0 floor, gives the **signal-dependent impairment per bin** (IMD, phase-noise leakage, codec).
  - Null power equal to the floor means the SNR is noise-limited.
- **Report a per-carrier breakdown** instead of one SNR:

  ```
  SNR_total ~ 1 / (1/SNR_noise + 1/SNR_phase(f) + 1/SNR_imd + 1/SNR_isi)
  ```

  The terms come from stages 0-3. Whatever the model does not explain is reported as a residual, which is probably the codec.

## Outputs

- **A line-probe report (JSON)** with:
  - the floor PSD,
  - |H| and group delay,
  - ppm,
  - PM/AM jitter vs f and the correlation verdict,
  - THD and IMD3 vs level, and the compression point,
  - the reversal settling time,
  - the per-carrier SNR breakdown.
- **Recommendations** the calibration engine can act on:
  - an EQ / pre-emphasis curve (only when the tilt is gain, not phase),
  - a TX level backoff,
  - pilot-tracking or n_fft hints. The phase-jitter limit argues for shorter symbols, which fits 256 beating 512 on this link.
- **A BAT** built from the full-load measurement as today, plus a BAT predicted after the recommended fixes.

## Build order

1. **Python generator and analyzer**, offline:
   - `line_probe.py` beside `ofdm_modem.py`.
   - Self-tests on synthetic channels. Each must be identified correctly: tilt only, f-proportional timing jitter, cubic nonlinearity, additive noise.
2. **WAV round trip on the real link.** Generate the WAV, play it through BT-Car, and capture on m6600 (`parec rtlsdr.monitor`). Analyse offline. This needs no modem change beyond a way to play a WAV out of the TX device. There are two options:
   - a control-port `PLAY` verb,
   - extending `CALIBRATE`.
3. **Control-port verbs.**
   - `LINEPROBE SEND` on TX.
   - `LINEPROBE LISTEN` and `LINEPROBE REPORT` on RX.
   - The report moves over the TCP port the same way `BAT GET` does.
4. **Stage-4 null carriers** in the OFDM probe, Python and the C `ofdm_probe.c` in step. Old peers must still parse the probe, so this needs a probe version flag.
5. **The calibration engine consumes the report**: EQ / pre-emphasis, level, then a re-probe to confirm.

## References

Cite these in the code with the file name and page, as given here.

- **V.34 transmit spectrum template:** "T-REC-V.34-199802-I!!PDF-E-2.pdf page 5".
  - What it says:
    - The template uses normalized frequency f/S, where S is the symbol rate.
    - The magnitude of the transmitted spectrum must conform to Figures 1 and 2 over (d/e - 0.45) to (d/e + 0.45).
    - That range is the carrier ± 0.45 S, so the occupied band is 0.9 S wide.
  - Our use:
    - The line-probe analyzer checks the received spectrum of the stage-4 full-load probe against a template: in-band flatness, and how steeply it falls at freq_lo and freq_hi.
    - The single tones (stage 1) give |H(f)| to tell a TX spectrum problem from a channel one.
    - `line_probe.py` cites this at the template check.
- **V.34 line probing signals L1/L2 and Table 17:** "T-REC-V.34-199802-I!!PDF-E-2.pdf page 31", section 10.1.2.4. Stage 3 copies the content and the `v34` mode follows it exactly; `line_probe.py` cites it at the comb generator and the phase table.
- **For later, not part of this plan:**
  - Shell mapper, precoder and trellis encoder: "T-REC-V.34-199802-I!!PDF-E-2.pdf page 20".
  - 16- and 64-state convolutional encoders: "T-REC-V.34-199802-I!!PDF-E-2.pdf pages 23-25".

  These matter for the data path once the line probe says what limits each carrier:
  - shaping gain from shell mapping,
  - a precoder for channels limited by ISI or tilt,
  - trellis coding gain on top of the BAT.

## Open questions

- Band for this link: the modem uses 400-9000 Hz; the FM transmitter accepts audio to 20 kHz. Sweep the single tones to 15-20 kHz anyway, to see where the codec cuts.
- A2DP codec behaviour on pure tones may differ from noise-like OFDM. That is why stage 4 keeps the realistic load, and why the residual term matters.

## First on-air comb: BT-Car FM -> m6600 RTL-SDR (2026-11-18)

The `LINEPROBE PLAN | SEND | LISTEN` control verb (build 113 m6600, 114 mac) runs stage 3. Command, the same
comb args on both hosts, LISTEN on m6600 first, SEND on the mac 1.5 s later:
`LINEPROBE SEND lo=150 hi=15000 secs=7 peak=0.5` (comb 150 Hz x 86 tones, 14 empty slots, crest 4.4 dB, m=2, so
there is one gap bin between tones). Run with peaks 0.5 and 0.25 (scratchpad lpair.sh).

| | peak 0.5 | peak 0.25 |
|---|---|---|
| top tone level | -42.9 dBFS | -49.0 dBFS (the 6 dB step arrived) |
| median tone SNR | 26.1 dB | 28.5 dB |
| empty slots (median) | -33.8 dBc | -34.8 dBc |
| clock | -529 ppm | -529 ppm (a later run: -12.7 ppm) |

| tone | 150 | 1500 | 4650 | 8700 | 12750 | 14850 Hz |
|---|---|---|---|---|---|---|
| level vs top | 0 | -0.9 | -2.7 | -3.6 | -4.4 | -5.0 dB |
| SNR (peak 0.5) | 53 | 40 | 30 | 25 | 21 | 20 dB |

What it says:
- **Response:** a smooth -5 dB tilt over 150 Hz-15 kHz, with no cut-off below 15 kHz. The FM transmitter's
  pre-emphasis and rtl_fm's de-emphasis roughly cancel. Amplitude EQ is worth at most about 5 dB.
- **The noise follows the signal:** 6 dB less signal costs the tones only 2.5 dB of SNR, and the empty slots
  keep the same dBc. ~~The likeliest source is the A2DP codec~~ -- **wrong: stage 1 (below) found it is an
  alias.** rtl_fm ran `-s 250k -r 48k`, a non-integer resampling that folds every tone around
  250000 - 5 x 48000 = 10000 Hz at about -30 dBc. With 86 tones the folds land on every slot: that was the
  -31 dBc floor.
- **Not timing jitter:** removing each window's common timing (what OFDM pilots do) gains 0.0 dB at every tone.
  The first analyzer read 0.7-0.9 us of "jitter" with timing_fraction 0.85, but that was a DRIFTING clock (the
  offset moves during a capture, and between runs from -529 to -13 ppm). The analyzer now fits the drift
  apart (`timing_drift_us`, 2.5 us over 6 s); jitter is 0.03 us and timing_fraction 0.01. The CSS "+530 ppm
  false PLL grid" on this link may have been this real offset.
- **What the BAT should do:** superseded by stages 0-2 below (the cause was the RX resampler, now fixed).

Open: `CAPTURE 12` right after `LISTEN` returned only the noise floor in both of the first runs, though LISTEN
had the comb; `CAPTURE` after a plain SEND works. Not chased yet.

## Stages 0-2 on the same link: the sweep and the classifier (2026-11-18)

`engine/linesweep.py` + `LINEPROBE SWEEP SEND|LISTEN|PLAN [lo=] [hi=] [peak=] [tones=]` (builds 115/116+). One
14.2 s burst: wake noise, up-chirp, 0.6 s quiet, 11 log-spaced single tones at peak/2, level steps (-12/-6/0 dB) at
low/mid/high, close pairs (300 Hz apart) at two levels, a 2 s far pair (lowest + highest tone), down-chirp. The
chirps find the burst and give the clock ratio (re-correlated with stretched references, since a chirp's matched
filter turns a frequency offset into a time offset). `classify` turns the measurements into `ls.finding.*`
lines: clock, floor/spurs, band edge, tilt (+ EQ curve), **mixing** (images at F +/- f, names a resampler fold),
noise mechanism (dB of SNR per dB of level), nonlinearity per frequency (names pre-emphasis when only the top
saturates), phase (timing jitter vs independent), codec (noise beside a lone tone vs the quiet floor), dropouts.
`tests/test_linesweep.py` builds each fault synthetically and checks it is named and nothing else trips (15 pass).

Run 1, rtl_fm `-s 250k` (as set up): lone-tone SNR **25 dB at every frequency and level**, PM tiny, THD -75 dBc.
The residual sat at 10000 -/+ f for every tone (750 -> 9250/10750, 1800 -> 8200/11800, 4200 -> 5800/14200), about
-30 dBc each. Finding: `[bad] mixing: each tone reappears mirrored around 10012 Hz ... 250000 -> 48000 Hz
resampling folds at 250000 - 5x48000 = 10000 Hz`.

Run 2, rtl_fm `-s 240k` (5 x 48 kHz, changed on m6600; old script kept as ~/rtlfm_pulse_250k.sh): lone-tone SNR
**~49 dB** from 150 Hz to 15 kHz, the images and the 10012 Hz idle spur gone, SNR follows the level (additive).
What is left:
- `[bad] nonlinearity`: 15 kHz at full level (peak 0.5) loses 2.7 dB of gain and its SNR collapses; the
  14550+14850 pair has IMD3 -28 dBc. Clean at -6 dB and at the low/mid picks: FM pre-emphasis driving the
  deviation limiter. Keep high carriers ~6 dB down (TX de-emphasis / BAT power tilt) or freq_hi lower.
- `[bad] dropouts`: one 13 ms hit during the 750/1200 Hz tones (BT).
- `[warn] clock`: -668 ppm this run, -529 the last; it moves between runs.
- `[warn] floor`: idle spurs at 150, 375, 1012 Hz (+16-26 dB over the floor).
- `[info] tilt`: 5.4 dB over 150 Hz-15 kHz.

Not yet: OFDM throughput re-measured at 240k (the BAT should improve a lot: ~25 dB caps were the alias); the
comb (stage 3) re-run at 240k; thresholds tuned on more links; a JSON report for the calibration engine.

### Correction: the 240k "dropouts" were the RX bridge, not Bluetooth

RTL-SDR streams cleanly only at certain hardware rates (250k, 900k, 1.2M, 2.4M S/s); other rates drop samples.
rtl_fm `-s 250k` captured at 1.25 MS/s (not one of them) and `-s 240k` captures at 1.2 MS/s (one of them, and an
integer 25 x 48 kHz). The 10-13 ms "dropouts" in the 240k runs were exact-zero runs (40-228 samples, 19-27 per
27 s capture, up to -92 dB below the tone; an FM demodulator never outputs exact zeros): the `pacat` -> Pulse
null-sink bridge underrunning at `--latency-msec=40`. At 200 ms the idle channel had 0 zero runs in 2 minutes
(15 in 20 s at 40 ms). The gaps also corrupted the clock reading (-668 / -785 ppm and 1320 ppm per-tone spread).

The classifier now reports these as `rx_gaps` (exact-zero runs inside the burst, with the underrun fix and the
RTL rate rule) and keeps `dropouts` for analog dips only; the alias finding names the 1.25 MS/s rate when it sees
the 250k fold.

Verified (builds 119/120, rtl_fm -s 240k, pacat 200 ms), two sweeps: 0 zero runs, no dropouts, clock -529.6 /
-529.7 ppm with 0.2-0.3 ppm per-tone spread -- the clock is steady; the "moving" -668/-785 ppm were the gaps.
Remaining findings, identical in both: 15 kHz saturates at full level (pre-emphasis -> limiter), idle spurs at
375 Hz (+23 dB) and 1012 Hz (+15 dB), 5.4 dB tilt, additive noise.

## CSS 0.5 s symbols, chunk 1, on the fixed link (2026-11-18)

Mac (node 2, BT-Car TX, css.amplitude 0.5) -> m6600 (node 1, rtl_fm -s 240k), general.modem css,
css.symbol_dur 0.5, css.chunk_size 1, 400-6000 Hz, python backend.  Symbols > 0.25 s go to the
matched-filter long-symbol decoder (burst path), not the streaming PLL.

- One 8 B DATA frame = 10 chunks, 327 s airtime as the mac logs it.  The m6600 decoded it ~21 s after the
  burst ended.
- Two transmissions (the original and the first retry, since no ACK gets back): both `ok=True snr=61.6/61.9 dB
  frames=10/0err rs=0`, so 20/20 chunks were clean with the -529.6 ppm clock (145 ms, ~0.28 symbol, of
  accumulated slip over the frame).
- Offline (scratchpad csssim.py: 44.1k encode -> 48k resample, clock offset, white noise): 10/0err at
  0, -529.6, -1500, -3000, -6000 and -12000 ppm at 20 dB, and at -529.6 ppm down to 0 dB.  Drift is not the
  limit for this decoder at 0.5 s.
- Cost: chunk 1 at 0.5 s is ~0.2 bit/s of payload.  The header-only ACK alone is 71 s.

### Live long-symbol receive (builds 123 m6600 / 124 mac)

Long-symbol bursts (0.5 s here) are no longer recorded whole and decoded after they end.
`ne0k-gui/modemgui/engine/css_longrx.py` runs `decode_longsym_mf` live: the clock is tracked while the
audio arrives, with the C chirp tracker beside it. See the README section "CSS receive for long symbols".

Same link, 0.5 s symbols, chunk size 1:

| Run | Clock | Chunks | Result |
|---|---|---|---|
| 1 | +13.3 ppm, locked 6 s after the first chirp; tracker +13 | 10, one every 32 s as each frame ended; 62.6–63.0 dB, doubt=0 | packet 0.9 s after the last chunk |
| 2 (retry) | +12.9 ppm | chunk 0 on a false preamble, rescued by tier 3 at offset +20; 1–9 live | `ok=True`, 10/0 err |

Before the tier-3 widening (build 121), the same retry failed outright. The bundles' encoder falls back to
a 16-symbol training run-in, because `shared.py` is not packaged. With that run-in the decoder accepted
11/13 Barker matches inside it (+24) and inside a payload (−13). Its cadence lattice then rejected the real
preambles for three frames. The replay of that recording now decodes every chunk it holds.

The mac→m6600 clock was +13 ppm at 08:00 UTC, against −529.6 ppm in this morning's sweeps: the BT link
does not keep one rate.
