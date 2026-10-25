> This directory holds several suites. `css_regress.py` (below) is the CSS
> decoder safety net; `css_enc_regress.py`, `css_dec_regress.py` and
> `css_sm_regress.py` guard the CSS C port (see below);
> `ofdm_acq_regress.py` guards the OFDM acquisition C port;
> `ofdmhs_collide_regress.py`, `mesh_fallback_regress.py`,
> `capture_survival_regress.py` and `node_boot_smoke.py` cover the MAC timeline,
> mesh fallback, capture survival and node startup respectively.
> Two cross-cutting modules live one level up: `tests/cbackend.py` (which C core
> this host can load) and `tests/benchlib.py` (the scored impulse benchmark).

# C backends: which binary, for which CPU (`tests/cbackend.py`)

The CSS matched filter, the OFDM acquisition/demodulator and Reed-Solomon all
have C cores selected by env var. Each has a numpy or pure-Python default, and
until now a library that could not be loaded silently returned the caller to
that default — including when the library was built for a **different CPU**.

That is not a corner case here. `BASE_DIR` is a shared SMB mount, so every node
sees one `tests/` directory. With a plain `libcssmf.so`, an arm64-apple node and
an x86_64 Linux node overwrite each other's build; the loser falls back to numpy
and still reports a green pass. A whole OTA run could grade the reference
implementation while its log claimed to be exercising the port.

Two changes:

**Arch-suffixed filenames.** Builds are now `libcssmf.x86_64-linux.so`,
`libcssmf.arm64-darwin.dylib`, and so on, so both hosts can hold a working build
at once. The suffix comes from `cbackend.py --tag`, which the Makefiles call —
the name the compiler writes is by construction the name the loader looks for.
A legacy unsuffixed `.so` is still accepted, but only when its header says it
matches this CPU.

```bash
make -C tests/c && make -C tests/CSS/c && make -C tests/OFDM/c both
```

(`tests/CSS/c/Makefile` is new. `libcssmf.so` used to be a committed binary with
no build rule, so there was no way to tell edited C from compiled C and no way
for a second CPU type to get a working copy at all.) The build products are
gitignored; `make clean` removes only this host's, since a blanket `rm` on the
shared mount would delete the other nodes' working builds.

**Requested is not the same as available.** With no env var set, a missing core
is the default configuration and stays quiet. With one set, a missing or
wrong-architecture core now **raises** instead of falling back. `RSCodec()` was
the worst of these: both implementations emit identical bytes, so the fallback
was undetectable downstream.

```bash
python tests/cbackend.py          # what this host can load, and why not
python tests/cbackend.py --json   # same, machine-readable
```

Every run logs a `C_BACKENDS=…` banner naming the resolved filename, so a run
that quietly used numpy is visible rather than passing vacuously.

**Across nodes.** Each node reports its CPU tag and core inventory in its
socketio `register` payload. `POST /api/tests/run` grades every targeted node
*before* dispatching and refuses the whole run with **409** if any one of them
would fall back — nobody starts. One mismatched participant does not degrade a
run, it invalidates it, and on an OTA test it also burns airtime for everyone
else. A run selects backends with `params.c_backends` (`["rs255", "css_mf", …]`)
or `params.env`; the selection is passed through to every node's environment, so
the fleet demonstrably runs one configuration. Role scripts additionally call
`testlib.preflight_backends()` at startup, which writes a `preflight_fail`
result and exits 3 — a node that is wrong reports as wrong rather than as lost.

# Impulse benchmark: scoring the tipping point (`tests/benchlib.py`)

A 0-100 score for how much impulse obscuration a modem survives, shared by the
CSS and OFDM suites so the two numbers read the same way.

Three axes, swept separately so a regression is attributable:

* **coverage** — "1 in every N symbols" obscured, N=10 (10%) up to N=1 (100%).
  This is the score axis.
* **width** — how much of a symbol one burst destroys, given either absolutely
  (100 ms) or as a fraction of a symbol (up to 3/4).
* **alignment** — start, middle or end of the symbol it hits.

**Each (width, alignment) cell scores 0-100 = the highest coverage percentage
that still decoded.** Surviving 1-in-1 is 100, surviving only 1-in-10 is 10,
failing 1-in-10 is 0. The headline is the mean over cells; the worst cell is
carried alongside, because a mean can hold up while one alignment collapses.

```bash
python tests/benchlib.py                                  # geometry selftest
python tests/CSS/css_loopback.py --only-benchmark --jobs 8
python tests/OFDM/ofdm_handshake.py benchmark --trials 2 --out bench.json
```

This supersedes nothing — `chirp_obscure` stays. That sweep picks a *random*
symbol subset and a *random* alignment per symbol, so its answer is a blend
across alignments and moves run to run; it is cheap and stays in the quick loop.
The benchmark is deterministic (`s % every_n == phase`), isolates alignment, and
is opted into because a 12-cell grid costs several times a single sweep.

Two things worth knowing before reading a score:

* **Alignment is measured against the ACTIVE part of the symbol,** not the
  stride. CSS puts a 5% silent gap after each chirp; a burst dropped in the gap
  damages nothing, so measuring "end-aligned" from the stride would score the
  gap instead of the chirp.
* **An all-zero OFDM grid is a real result, not a broken harness.** One
  obscured OFDM symbol puts ~16 byte errors into a chunk that RS(272,256) can
  correct 8 of, and there is no interleaver, so a single burst is fatal. The
  benchmark runs a clean control first and refuses to report a score if the
  unimpaired path did not decode; below a zero score it walks a sub-10%
  `FLOOR_LADDER` purely to say *how* it failed ("survives 1.00%" vs "a single
  burst is fatal"). The floor levels never affect the score.

# OFDM acquisition backend suite (`ofdm_acq_regress.py`)

Holds the three implementations of `_grid_coherence_scan` to each other: the
numpy reference (default, and the benchmark), the C core built in double, and
the C core built in float32 (the 32-bit target — RP2350 / CMSIS-DSP).

```
make -C tests/OFDM/c both                          # build first, or it SKIPs
python tests/regression/ofdm_acq_regress.py        # both C precisions
python tests/regression/ofdm_acq_regress.py --lib 32 -v   # float32, per-case
```

Grades **decisions, not bit-equality** — reduction order, top-k boundary ties
and argmax-on-a-plateau all differ legitimately between correct
implementations. The bar is identical candidate starts, identical gate
thresholding, an argmax that is identical or provably indistinguishable, and
identical `frame_start` / `n_syms` / SNR / BAT out of `analyze_probe` end to
end. The H row is checked separately and *is* bit-exact for the double build,
which is what separates a port bug from a precision effect. See the module
docstring for the full argument.

The C is opt-in and the numpy path stays the default:
`OFDM_ACQ_BACKEND=c` selects it, `OFDM_ACQ_LIB=32` picks the float32 build. A
missing `.so` falls back to numpy silently; a `.so` older than its sources loads
but warns. Every selection prints an `ACQ_BACKEND=c …` banner, so a run that
silently fell back to numpy is visible rather than passing vacuously.

The end-to-end gate for the OFDM DSP is the handshake sim, which must produce
byte-identical stdout on every backend:

```
python tests/OFDM/ofdm_handshake.py simtest
OFDM_ACQ_BACKEND=c python tests/OFDM/ofdm_handshake.py simtest
```

# CSS decoder regression suite

Safety net for `tests/css_decoder.py` refactors. Replays synthetic encodes and
real mic recordings through the decoder offline (mocked `sounddevice`, no
hardware) and asserts expected decoded text.

```
python tests/regression/css_regress.py          # full suite (~2-4 min)
python tests/regression/css_regress.py --slow   # + 3.0s symbol case
python tests/regression/css_regress.py --only mic -v   # filter + log-on-fail
python tests/regression/css_regress.py --list          # show cases + labels
python tests/regression/css_regress.py --label distance --label quiet
```

Exit code 0 = all pass. Run before and after ANY decoder change.

**Adding a case from the UI (preferred).** The server's **Reports** tab lists
every run in `results/`. Review one, give it a verdict (`valid`,
`expected_fail`, `bug`, `junk`), tag the conditions, set the expected text, then
hit **Add to regression suite**. Labels are free-form and entirely user-driven —
nothing is applied by default; type a new one (`distance`, `quiet`,
`impulse_noise`, …) and it joins the picker for later runs. Promotion:

* copies the run's capture WAV + `css_info.json` into
  `fixtures/runs/<case>/` — so the case survives deleting the results dir;
* appends an entry to `cases.json`, which `load_promoted_cases()` reads at
  startup and appends to `build_cases()`.

Labels ride along into `cases.json`, so `--label distance` runs just the
distance cases. Deleting a run from the Reports tab is blocked while it backs a
suite case — remove it from the suite first.

Runs with a 0-byte capture WAV (capture-thread failure) can't be promoted; the
tab shows them as `n/a` with the reason.

**Adding a case by hand:** add the `run_id` to the loop at the bottom of
`build_cases()` in `css_regress.py` — expected text is read from that run's
`decode_result_laptop.json`. Unlike promotion this reads `results/` in place,
so the case breaks if the run is deleted.

**Fixtures** (`fixtures/`): `li2.wav`/`co2.wav` (real mic, 1.0s symbols,
validated the MF-direction fix), `fw.wav` (freewheel/wake 15s-suspend).
These WAVs (and their `*_info.json`) are **gitignored — local disk only**
(ZFS-snapshotted, not in git). A fresh clone has no fixtures, so the mic cases
report `SKIP` and only the synthetic cases run; drop the WAVs back into
`fixtures/` to re-enable them. WAVs are copied to a local temp cache before
opening (opening directly on the Z: SMB share can raise a file-open "System
error").


# CSS C-port suites (`css_enc_regress.py`, `css_dec_regress.py`, `css_sm_regress.py`)

Three suites cover the C port in `tests/CSS/c/`, split by what they can
actually compare.

```
python tests/regression/css_enc_regress.py     # TX chain vs css_encoder.py
python tests/regression/css_dec_regress.py     # RX pipeline vs css_fec.py
python tests/regression/css_sm_regress.py      # state machine, trace vs golden
python tests/regression/css_sm_regress.py --update   # re-record the golden
python tests/regression/css_sm_regress.py --show wake_15s
```

**`css_enc_regress`** and **`css_dec_regress`** are equivalence suites: they
hand the same bytes to the C and to the Python and compare, stage by stage
(CRC, bit packing, convolutional encode, interleave, chirp / deinterleave,
Viterbi, differential decode, CRC check). Both grade in two precision stages —
the `libcssenc64` build against Python's float64 arithmetic, where any
difference is a *port* bug, then the shipping float32 build against that,
where a difference is a *precision* effect. As measured today the float32
chirp costs 0.00002 dB of matched-filter correlation and the float32 Viterbi
metric changes 0 of 21480 decoded bits, which is what justifies float being
the default in `css_enc.h` and `css_dec.h`.

**`css_sm_regress`** is different in kind, because the acquisition and
tracking loop in `decode_longsym_mf` is a state machine over a stream and has
no pure function to compare against. It runs the decoder on nine synthetic
inputs, parses the `MF_*` log lines into a normalised event trace, and diffs
that against a checked-in golden with per-field tolerances — exact on the
integers the state machine *decided* (symbol index, sequence number, grid
anchor, accept/reject verdict), tolerant on the floats that merely describe
those decisions. `css_regress.py` only ever asserted the final decoded string,
which a port can reach two symbols late, via a relock that should not have
fired, and still pass.

Both the numpy and the `CSS_MF_BACKEND=c` path produce identical traces,
including on the deliberately marginal −28 dB case — which is the evidence
that the tolerances are calibrated rather than merely loose.

Coverage is complete: every `MF_*` event the state machine can emit is reached
by some case, including `MF_RELOCK` and `MF_LOCK_SUSPECT`.

## Finding the hard cases (`css_sm_sweep.py`)

The false-lock watchdog resisted hand-picked inputs, so `css_sm_sweep.py`
searches for them instead — a grid of perturbation families (`shift`,
`excise`, `echo`, `decoy`, `prelude`) swept over offsets expressed as
fractions of a symbol slot, run across every core.

```
python tests/regression/css_sm_sweep.py --jobs 40
python tests/regression/css_sm_sweep.py --family echo --dur 1.0 -v
python tests/regression/css_sm_sweep.py --target MF_RELOCK --promote
```

320 variants in ~95 s on 40 workers. It exits non-zero when a hunted event was
never reached, and `--promote` writes the winning specs out for promotion into
`css_sm_regress.CASES`. Re-run it after any change to the acquisition or
tracking gates.

Two things were blocking the watchdog, and neither was guessable from reading
the code:

1. **`mf_lock_verify_min_dur` defaults to 1.0 s.** Below that the watchdog
   never evaluates. Every case in `css_sm_regress` used 0.5 s symbols, so no
   perturbation could ever have fired it. `dur` is a swept axis for this
   reason, and `watchdog_gated_0.5s` now pins the gate itself.
2. **It keys on the buffer's *global* MF peak**, so a competing signal must be
   at least as loud as the direct path (`gain >= 1.0`) to be seen at all. A
   quieter echo corrupts symbol decisions without ever reaching the watchdog —
   which is why "add an echo" alone did not work.

Measured: 35 of 320 variants reach `MF_RELOCK` and 98 reach `MF_LOCK_SUSPECT`
at 1.0 s, against **0 of 64** at 0.5 s. That contrast is the evidence the gate
works, not that the test is flaky.

The C89 syntax gate is a make target, not a Python suite:

```
make -C tests/CSS/c c89check
```
