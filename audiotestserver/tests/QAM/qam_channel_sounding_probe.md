# Fast Channel-Sounding Probe — Design Review & Plan

## The problem with the current probe

`qam_probe_play` / `qam_probe_listen` treat probing as **empirical brute force**:
encode a full message at every `carrier × symbol_rate` combo, transmit *many
frames* per combo, decode, and rank by frames-decoded. That's `N` separate
transmissions (`N` = 18+ today), each carrying enough data to get statistics.
Even with the continuous-burst rewrite it's still fundamentally "try everything
and see what sticks."

Real QAM modems don't do this. They **measure the channel once**, then
**compute** the best parameters. This doc reviews how the well-known specs do it
and proposes a channel-sounding probe for this codebase.

---

## How real modems probe the channel

### 1. ITU-T V.34 (33.6 kbps voiceband) — *line probing*
The closest analogue to what we want. During handshake Phase 2, V.34 sends a
**line-probing signal**: a set of tones spaced across the voiceband (~150 Hz
spacing, roughly 150–3750 Hz — verify exact set against the ITU-T V.34 spec).
The receiver does an FFT and measures, at each tone:
- **amplitude** → channel magnitude response |H(f)|
- **phase** → group delay / delay distortion

From that single measurement it estimates SNR vs frequency and **selects from a
small fixed menu**:
- **symbol rate (baud):** one of ~{2400, 2743, 2800, 3000, 3200, 3429}
- **carrier frequency:** paired with the chosen baud
- **pre-emphasis filter** and **power** to fight the measured tilt

Key principle: *one wideband measurement → analytically choose baud + carrier +
shaping.* No "try each and count errors."

### 2. ADSL / VDSL (DMT — Discrete MultiTone) — *bit-loading / water-filling*
The channel is split into hundreds of ~4.3 kHz subcarriers. A wideband
pseudo-random training signal (the C-MEDLEY / R-MEDLEY phase) lets the receiver
estimate **SNR per subcarrier**. Then **bit-loading** assigns QAM order per tone
proportional to that SNR (water-filling): 15 bits on a clean tone, 2 or 0 on a
noisy one. Measure SNR(f) once → compute a per-frequency modulation map.

### 3. 802.11 OFDM / DOCSIS — *known training symbols + pilots*
The preamble carries known training symbols (e.g. the 802.11 Long Training
Field). **One FFT** of a known symbol yields the channel estimate H(f) on every
subcarrier at once; the MCS (modulation + code rate) is then chosen from a table
against the measured SNR. DOCSIS adds upstream **pre-equalization** and *ranging*
built on the same "known probe → estimate → adapt" idea.

### 4. Room-acoustics measurement (Farina ESS) — because this is an *acoustic* channel
The exponential sine sweep (ESS) deconvolution method is the standard way to
measure a **reverberant acoustic** system's impulse response. It:
- excites the whole band with a **constant-envelope** sweep (low crest factor →
  more energy through the speaker without clipping → better measurement SNR),
- recovers the **impulse response** `h(t)` by deconvolving with the inverse sweep,
- **separates harmonic distortion** (speaker/mic nonlinearity) into a pre-arrival
  region so it doesn't pollute the linear estimate.

From `h(t)` you get `H(f)` **and** the **delay spread** (multipath/reverb tail),
which directly bounds the usable symbol rate — something a tone-only probe can't
give you cleanly.

### The common principle
> Send **one** known wideband excitation → estimate **H(f)** and **noise N(f)** →
> derive **SNR(f)** (and delay spread) → **compute** the operating point from a
> formula/table, not from trial transmissions.

---

## Proposed probe for this codebase

Two-phase probe that replaces the combo sweep:

```
Phase A  Channel sounding  (ONE short transmission, no data frames)
Phase B  Parameter selection (pure computation from the measurement)
Phase C  Single confirmation burst at the computed best config (optional)
```

### Phase A — sound the channel

**Signal:** an **exponential sine sweep** (Farina ESS) from `f_lo` to
`f_hi ≈ 0.45·fs`, ~1.5–3 s, precomputed once. Bracket it with **~0.5 s silence**
before and after so the receiver can measure the **noise floor** with no signal.

Optionally offer a **Schroeder-phased multisine** (V.34-style) as an alternative
excitation — simpler receiver (one FFT, read tone bins) but worse crest factor
and no clean distortion separation. ESS is the better default for a reverberant
acoustic link; keep multisine as a config switch.

**Receiver:** open the mic **once** at the device's **native rate** (do *not*
resample — see [[css-longsym-phase-incoherence]] / native-rate lessons in memory;
resampling 48k↔44.1k wrecks phase and would corrupt the deconvolution). Record
`[silence | sweep | silence]` in a single capture.

**Analysis (all offline, in-process):**
1. Deconvolve capture with the analytic inverse sweep → impulse response `h(t)`.
2. Window out the linear IR (drop the pre-arrival harmonic-distortion cluster).
3. `H(f) = FFT(h)` → magnitude + group delay across the band.
4. Noise PSD `N(f)` from the leading silent segment.
5. `SNR(f) = |H(f)|²·P_tx / N(f)` (relative, dB).
6. **RMS delay spread** `τ_rms` and an RT60-style tail estimate from `h(t)`.

Deliverables: `SNR(f)` curve, `|H(f)|`, group delay, `τ_rms`, noise floor — one
PNG + a JSON, from a single ~3–4 s transmission.

### Phase B — compute the best parameters (no transmissions)

Reuse the modulator's own constraints (`Modulator._auto_beta`) so we only ever
recommend a **legal** config. For a candidate `(cf, sr)`:

```
beta = auto_beta(cf, sr)                    # from qam_writer
BW   = sr * (1 + beta)                       # RRC occupied bandwidth
band = [cf - BW/2, cf + BW/2]                # occupied passband
```

Constraints already enforced by the modem:
- `cf > 1.65 · sr` (lower-sideband guard, from `_auto_beta`)
- `2·cf + BW < fs` (upper sideband under Nyquist)
- `fs / sr ≥ 4` (samples/symbol)

Selection (throughput is `sr·4` bits/s gross since QAM order is fixed at 16):

```
for cf in coarse_carrier_grid:
    for sr in candidate_symbol_rates (divisors of fs):
        if not legal(cf, sr): continue
        band   = occupied_band(cf, sr)
        snr_min = min(SNR(f) for f in band)        # worst point in the band
        snr_avg = mean(SNR(f) for f in band)
        # 16-QAM needs ~17–20 dB Es/N0 uncoded for ~1e-3 BER; RS ECC relaxes it.
        if snr_min < MARGIN_DB: continue
        # delay-spread / ISI guard: symbol period must cover the reverb tail
        if (1/sr) < K_ISI * tau_rms: continue      # else equalizer can't cope
        score = sr                                  # maximise baud = throughput
    pick (cf, sr) with best score, tie-break on snr_avg
```

Output the **same** `probe_best.json` (`carrier_freq`, `symbol_rate`) the rest of
the pipeline already consumes, plus rich diagnostics (SNR curve, why others were
rejected: "band clipped Nyquist", "snr_min 11 dB < 16 dB margin", "delay spread
too long for baud"). This is the whole point — **the answer is computed from one
measurement**, not searched by transmitting each combo.

> The `MARGIN_DB` ↔ modulation mapping is exactly the V.34 "SNR → baud/carrier
> menu" and the DMT "SNR → bits" idea, specialised to a fixed-16-QAM menu.

### Phase C — one confirmation burst (optional but recommended)

Transmit **a single short frame** at the computed `(cf, sr)` and decode it to
confirm real-world EVM/RS-headroom matches the prediction. Optionally confirm the
top *2–3* predicted configs instead of all N. This is the only data-bearing
transmission in the whole probe — replacing "many frames × N combos" with
"one frame × 1 (few) config(s)."

---

## Why this is faster

| | Current combo sweep | Channel-sounding probe |
|---|---|---|
| Transmissions | N combos × many frames | 1 sweep (+1 confirm) |
| Air time | Σ(all combos, seconds each) → minutes | ~4 s sweep + ~1 s confirm |
| What limits speed | number of configs tried | fixed, independent of grid size |
| Adding more candidates | linearly slower | **free** (just more math on H(f)) |
| Output | best of the tried set | optimum over the *continuous* SNR(f) |
| Bonus | — | H(f), delay spread, noise floor for the decoder work |

Grid resolution becomes free: want to consider 200 carrier positions and every
legal baud? It's an argmax over the measured curve, not 200 more transmissions.

---

## Fit with this codebase

**New role scripts** (parallel to the existing ones, keep the dispatcher happy):
- `qam_sound_play.py` — precompute ESS (or multisine), `coord_signal("sound_ready")`,
  wait `sound_armed`, `coord_signal("sound_start")`, single `sd.play`,
  `sound_done`. Save `probe_sweep_tx.wav`.
- `qam_sound_listen.py` — record once at native rate, `coord_signal("sound_armed")`,
  anchor on `sound_start`, do the deconvolution + selection, emit `probe_best.json`
  + `channel_snr.png` + `channel_report.json`. Optional Phase C reuses
  `Modulator`/`Demodulator` in-process for the confirmation frame.

**Coordination:** 4 signals total (`sound_ready → sound_armed → sound_start →
sound_done`), same lightweight pattern as the continuous-burst plan — no per-combo
handshakes.

**Reuse, don't modify DSP:** `Modulator._auto_beta` for legality/BW,
`Demodulator` for the confirmation decode. The sweep generation, deconvolution,
and selection are new but self-contained (numpy/scipy already deps).

**Native-rate discipline:** open input/output at the device native rate and do
all math there; never resample the sweep or capture (memory:
[[css-longsym-phase-incoherence]], [[css-longsym-overflow-rootcause]]).

---

## Signal-design notes

- **ESS vs multisine vs MLS:** ESS default (constant envelope, distortion
  separation, gives delay spread). Schroeder multisine = V.34-style, one-FFT
  readout, offer as a switch. MLS/PN is a third classic option (correlate for IR)
  but ESS handles loudspeaker nonlinearity better.
- **Crest factor matters acoustically:** a constant-envelope sweep pushes far more
  energy through a small laptop speaker before clipping than a multitone burst,
  which directly raises measurement SNR — the single biggest practical win.
- **Band limits:** sweep `f_lo ≈ 150–300 Hz` to `f_hi ≈ 0.45·fs`. Below `f_lo`
  laptop speakers roll off hard; above `0.45·fs` you risk the anti-alias edge.
- **Distortion check:** the harmonic-distortion cluster ESS isolates is itself a
  useful diagnostic — heavy distortion means back off amplitude before trusting
  any high-baud recommendation.

---

## Risks & validation

1. **Deconvolution alignment / latency.** Same handshake-latency issue as the
   burst plan; ESS deconvolution is *self-aligning* (the IR peak marks t0), so it's
   actually more robust than offset-gating. ✔
2. **Noise estimate needs true silence.** Guarantee the leading silence window is
   long enough (~0.5 s) and gate against accidental room noise bursts; report the
   measured noise floor so the operator can sanity-check.
3. **Predicted ≠ actual.** That's exactly why Phase C exists — one confirmation
   frame catches model error cheaply. If confirmation fails, fall back to
   confirming the next-best 1–2 candidates (still far cheaper than the full sweep).
4. **Nonlinear speaker at high amplitude.** ESS separates it; use the distortion
   region to cap amplitude before selection.
5. **Delay-spread → baud rule is heuristic.** Calibrate `K_ISI` against a couple
   of known-good rooms; the existing adaptive equalizer buys margin here.

---

## Forward-looking: this unlocks bit-loading

Once `SNR(f)` is measured, the natural next step (DMT-style) is **per-frequency
modulation**: the current modem is fixed 16-QAM on one carrier, but the same SNR
curve would drive a water-filling bit-loader if you ever move to a multicarrier /
OFDM front end (there are already `ofdm_*` prototypes in `tests/`). The
channel-sounding probe is the shared foundation for both the single-carrier
parameter pick *and* a future multicarrier bit map.

---

## Suggested build order

1. Standalone `channel_sound.py`: ESS generation + inverse filter + deconvolution
   + `SNR(f)` / `τ_rms`, validated on a **loopback WAV** (play→file→analyse) with
   a known synthetic channel (add delay + band-limit + noise, confirm it's
   recovered). No hardware yet.
2. Selection function `select_qam_params(SNR(f), τ_rms, fs)` using
   `Modulator._auto_beta` for legality; unit-test against hand-picked SNR curves.
3. `qam_sound_play.py` / `qam_sound_listen.py` role scripts + coordination.
4. Wire Phase C confirmation reusing `Demodulator`.
5. Acoustic A/B: run old combo sweep and new sounding probe in the same room,
   confirm they pick the same (or better) `(cf, sr)` and report the wall-clock
   speedup.

---

## What stays the same
- `Modulator` / `Demodulator` DSP and the fixed 16-QAM front end.
- `probe_best.json` output contract → the rest of the pipeline is untouched.
- `app.py` dispatch, `testlib`, coord endpoints, ranking/plot helpers where reused.
- The separate noisy-conditions *decoder* work — but now it's fed by a measured
  `H(f)` / delay spread / noise floor instead of guesswork.
