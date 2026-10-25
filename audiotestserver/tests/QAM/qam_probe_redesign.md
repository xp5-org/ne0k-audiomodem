# QAM Probe Redesign — Continuous Training Prober

## Goal

Replace the current stop-start probe (spawn a reader subprocess + reopen the mic +
30 s deadline pad **per combo**) with a real modem-style trainer:

1. **Play** builds one *continuous* training waveform (all combos back-to-back with
   short guard gaps) and plays it in a **single** `sd.play()`.
2. **Listen** opens the mic **once**, records the whole sweep to one buffer, then
   decodes every combo **in-process** by slicing each combo's known time-window out
   of the single recording (time-gated by manifest offsets).

Net effect: sweep time collapses from
`N × (subprocess spawn + mic reopen + 30 s pad + 2 s silence + 3 s tail)`
down to `≈ Σ burst_durations + N × guard_gap`.

Both files keep their names (`qam_probe_play.py`, `qam_probe_listen.py`) and the
`PARAMS` / `RUN_DIR` / `coord_*` env interface so the dispatcher in `app.py`
(`qam_probe` → `_play` / `_listen`) is unchanged. `qam_writer.py` /
`qam_reader.py` are **not** modified — the plan reuses `Modulator` and
`Demodulator` as libraries.

---

## Decisions locked in

- **Scope:** full continuous-burst rewrite of both role scripts.
- **Segmentation:** time-gate by manifest offsets. Play writes each burst's
  sample offset + length into the manifest; receiver slices that window (+ guard)
  before handing it to that combo's `Demodulator`. This is robust when two combos
  share a carrier frequency (the sweep is carrier × symbol_rate, so collisions are
  guaranteed), which pure ZC self-sync is not.

---

## Play side — `qam_probe_play.py`

### Build one signal in memory (no per-combo WAV files needed)

Reuse `Modulator` directly instead of shelling out to `qam_writer.py`:

```python
from qam_writer import Modulator

def build_burst(msg, cf, sr, chunk, ecc, seq, fs, amp):
    m = Modulator(msg, chunk_bytes=chunk, ecc_bytes=ecc, seq_bytes=seq,
                  sample_rate=fs, carrier_freq=cf, symbol_rate=sr)
    sig = m.encode()                     # float passband, pre-scale
    pcm = sig * m.scale_factor / 32700.0 # normalise to ~[-1,1] float32
    cfg = m.config_dict()                # NEW small helper, or replicate write_config dict
    return pcm.astype(np.float32), cfg, m.total_frames
```

> `Modulator` already exposes `.encode()`, `.scale_factor`, `.total_frames`,
> and `write_config()`. Add a tiny `config_dict()` returning the same dict
> `write_config()` dumps, so we can keep per-combo configs in the manifest
> without touching disk N times. (Alternatively call `write_config` per combo —
> disk cost is trivial; the expensive part was the subprocess, not the file.)

### Concatenate with structure

```
[ LEAD_IN silence  ~1.0 s ]     # lets receiver AGC/stream settle, marks t0
[ burst 0 ][ GAP ~0.4 s ]
[ burst 1 ][ GAP ~0.4 s ]
...
[ burst N-1 ][ TAIL silence ~0.5 s ]
```

- `GAP` guard silence (fixed) separates bursts so windows never overlap even with
  a little clock skew. 0.3–0.5 s is plenty.
- Track a running sample cursor while concatenating; record each burst's
  `start_sample` (relative to the concatenated buffer, i.e. after LEAD_IN) and
  `n_samples`.
- Amplitude-normalise the **whole** concatenated buffer once at the end so no
  single burst clips and relative levels are preserved.

### Manifest (drives receiver time-gating)

```json
{
  "run_id": "...",
  "player": "client1",
  "sample_rate": 48000,
  "lead_in_samples": 48000,
  "gap_samples": 19200,
  "steps": [
    {"step":0,"carrier_freq":800,"symbol_rate":48,"symbol_rate_requested":40,
     "chunk_bytes":8,"ecc_bytes":8,"seq_bytes":2,
     "start_sample":48000,"n_samples":211200,"total_frames":37,
     "config": {<full demod config dict>}}
  ]
}
```

Embedding the full per-combo config dict in the manifest means the receiver needs
**no** separate `probe_NNN_config.json` files and can build each `Demodulator`
straight from the manifest.

### Play & coordinate (only TWO handshakes total, not per-step)

```python
write_manifest()
coord_signal("probe_ready")
coord_wait_any("probe_armed", timeout=30)   # receiver has mic open + is recording

t_play_start = time.monotonic()
coord_signal("probe_play_start")            # marks t0 for the receiver
sd.play(full_signal, samplerate=fs, device=out_dev)
sd.wait()
coord_signal("probe_all_played")
```

No `probe_step_*` signals at all. The receiver keys everything off
`probe_play_start` + manifest offsets.

### Optional: still dump the concatenated WAV

Write the full training signal to `RUN_DIR/probe_full_tx.wav` for debugging /
offline replay. Cheap and very useful for the noisy-conditions work.

---

## Listen side — `qam_probe_listen.py`

### Record once

Open a single `sd.InputStream` (or `sd.rec` for a fixed duration). Compute the
expected total duration from the manifest:

```
total_samples = lead_in + Σ(burst_i + gap) + tail
record_secs   = total_samples/fs + RECORD_PAD   # e.g. +2 s slack
```

Sequence:
1. Wait for `probe_manifest.json` (as today).
2. Open mic stream, start appending to one capture buffer in the callback
   (reuse the float32 capture pattern already in `qam_reader.listen_live`).
3. `coord_signal("probe_armed")`.
4. Wait for `coord_check("probe_play_start")`; record the local monotonic time
   `t0_local` at the moment it appears. Also note how many samples were already
   captured at that instant → `start_offset_samples` (the alignment anchor).
5. Keep recording until `probe_all_played` seen **and** `record_secs` elapsed
   (whichever gives the full tail), then stop the stream.
6. Save the full capture to `RUN_DIR/<client>_probe_capture.wav`.

> **Alignment anchor.** The manifest offsets are relative to the *transmitted*
> buffer's t0 (start of LEAD_IN... actually start of first burst). We align the
> recording to transmit-t0 using the `probe_play_start` signal: the sample index
> in the capture buffer at the moment `probe_play_start` was observed ≈ transmit
> t0 (minus network + acoustic latency). Convert each manifest `start_sample`
> (which is relative to end-of-lead-in) to a capture index:
>
> ```
> capture_idx = anchor_idx + lead_in_samples? ...
> ```
>
> Keep it simple: make manifest `start_sample` **include** the lead-in (i.e.
> absolute from transmit buffer sample 0). Then
> `capture_idx = anchor_idx + start_sample`, where `anchor_idx` = capture-buffer
> length when `probe_play_start` fired. The LEAD_IN silence (~1 s) is deliberately
> generous so that even ±200 ms of handshake/acoustic latency still lands the
> slice inside the burst; the per-combo `GUARD` below absorbs the rest.

### Decode each combo in-process (no subprocess)

```python
from qam_reader import Demodulator

capture = np.concatenate(capture_chunks)          # float32 [-1,1]
pcm16   = (np.clip(capture, -1, 1) * 32767).astype(np.int16)

for step in manifest["steps"]:
    cfg   = step["config"]
    demod = Demodulator.from_config_dict(cfg)     # NEW: ctor from dict, not file
    demod.ZC_THRESHOLD = ZC_THRESH

    lo = anchor_idx + step["start_sample"] - GUARD_SAMPLES
    hi = anchor_idx + step["start_sample"] + step["n_samples"] + GUARD_SAMPLES
    seg = pcm16[max(0,lo):hi]

    # process_wav path, but from an array. Either:
    #   (a) add Demodulator.process_array(seg) that mirrors process_wav, or
    #   (b) write seg to a temp wav and call process_wav (simplest, still fast).
    demod._stats_out = None
    text = demod.process_array(seg)               # collects demod._frame_stats
    results.append(summarize(demod, step))
```

Two small **additions to `qam_reader.py`** (kept backward-compatible):
- `Demodulator.from_config_dict(cfg)` — factory that skips the file read
  (`__init__` currently opens `config_path`; refactor so it can take a dict).
- `Demodulator.process_array(audio_int16)` — the body of `process_wav` minus the
  `wavfile.read`, so the receiver can decode an in-memory slice without temp
  files. `process_wav` becomes `read wav → process_array`.

> If you'd rather not touch `qam_reader.py` at all in v1, fall back to option (b):
> write each slice to a temp WAV and call `process_wav`. Still eliminates the
> subprocess spawn and mic reopen — the two big costs — so it already captures
> ~90% of the speedup. The `from_config_dict`/`process_array` refactor is the
> clean version.

### Metrics / ranking / reporting

`process_wav`/`process_array` already populates `demod._frame_stats` (via
`_emit_frame_stat`). Reuse the **existing** per-step summary math from the current
`decode_step` (frames_decoded, total_rs_errors, avg SNR/ECC/ZC, net_bps) — just
read it off `demod._frame_stats` instead of parsing `FRAME_STAT:` stdout lines.
`rank_results`, `print_report`, `make_qam_constellation_png`,
`make_qam_probe_heatmap_png`, and the `probe_results.json` / `probe_best.json`
outputs stay **exactly as they are**. Only the *acquisition* of the stats changes.

---

## New / changed params

| Param | Where | Default | Purpose |
|---|---|---|---|
| `lead_in_secs` | play | 1.0 | settle time + alignment margin (replaces per-step `head_silence_secs`) |
| `gap_secs` | play | 0.4 | guard silence between bursts |
| `tail_secs` | play/listen | 0.5 | trailing silence |
| `guard_secs` | listen | 0.3 | ± slack added to each decode window |
| `record_pad_secs` | listen | 2.0 | extra recording slack after expected end |

Remove per-step `timeout_per_step`, `step_deadline_pad`, `head_silence_secs`
(no longer meaningful). Keep `mic_gain_db`, `zc_threshold`, `amplitude`,
and the carrier/symbol sweep params unchanged.

---

## Coordination protocol (before → after)

| Before (per step, ×N) | After (once) |
|---|---|
| `probe_ready` | `probe_ready` |
| `probe_armed` | `probe_armed` |
| `probe_step_i_listen_ready` | — |
| `probe_step_i_play` | `probe_play_start` |
| `probe_step_i_played` | — |
| `probe_all_played` | `probe_all_played` |

From 3 signals/step down to a fixed 4 signals total.

---

## Risks & mitigations

1. **Clock drift over a 1–2 min continuous playout.** Two independent sound cards
   (TX card vs mic ADC) drift ~10–100 ppm → up to ~10 ms over 2 min. That's tiny
   next to the 300 ms `guard` + 1 s `lead_in`, and each burst *re-syncs* on its own
   ZC preamble inside the slice, so residual drift only shifts where the ZC peak
   lands within the window — harmless. ✔ covered by guards.
2. **Handshake/acoustic latency on the alignment anchor.** `probe_play_start` fires
   just before `sd.play`; actual sound arrives tens of ms later. LEAD_IN (1 s) +
   guard absorb it. If it ever proves marginal, add a **chirp marker** at the start
   of the signal and cross-correlate to find true t0 (defer — offsets+guards should
   suffice).
3. **Two combos sharing a carrier decoding each other's burst.** Prevented by
   time-gating: each `Demodulator` only ever sees its own slice. ✔ (this is why we
   chose offset gating over pure ZC self-sync.)
4. **Total record length wrong if a combo failed to encode.** Build the manifest
   from *successfully* encoded bursts only (as today), and derive `record_secs`
   from the manifest, so lengths always match what was actually concatenated.
5. **Level clipping across concatenation.** Normalise the whole buffer once at the
   end, not per burst.

---

## Suggested implementation order

1. `qam_reader.py`: add `Demodulator.from_config_dict()` + `process_array()`;
   refactor `process_wav`/`__init__` to reuse them. Verify offline against an
   existing captured step WAV → identical decode to before.
2. `qam_writer.py`: add `Modulator.config_dict()` (extract from `write_config`).
3. Rewrite `qam_probe_play.py`: build concatenated signal + manifest w/ offsets,
   single `sd.play`, 4-signal coord. Dump `probe_full_tx.wav`.
4. Rewrite `qam_probe_listen.py`: record-once, anchor on `probe_play_start`, slice
   + in-process decode, reuse existing summary/rank/report/plots.
5. Loopback test (play+listen same box, speaker→mic or virtual cable) to validate
   alignment & that every combo still decodes, then a real acoustic run.

---

## What is explicitly NOT changing

- `Modulator` / `Demodulator` DSP (constellation, ZC sync, pilots, equalizer).
- Ranking, reporting, constellation PNG, heatmap, `probe_best.json` format.
- `app.py` dispatch, `testlib`, coord server endpoints.
- The separate "noisy-conditions decode robustness" work — that's a follow-up on
  the DSP; this redesign just makes the prober fast enough to sweep enough
  settings to *find* the robust operating point, and gives you a clean single
  capture WAV per run to iterate the decoder against offline.
```
