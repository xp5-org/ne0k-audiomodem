# rp2350_eval — can DSP be offloaded to an RP2350 over USB?

Question: the CSS and OFDM decoders cost too much CPU for a Pi Zero (and maybe a Pi 4).
Could an RP2350 (dual Cortex-M33 @ 150 MHz, single-precision FPU, 520 KB SRAM) take the
DSP while the host only moves samples over USB?

This tool answers the *compute* half from simulation, with the real code:

1. `fw/` — bare-metal M33 firmware. The kernels are the **unmodified** NE0K C sources
   (`../ofdm_dos/core`: `ofdm_fft.c` in both `double` (as shipped) and `OFDM_ACQ_F32`
   builds, `rs255.c`) plus loops standing in for the CSS matched filter / dechirp
   (f32/f64 dot, Q15 FIR, complex multiply). clang cross-compiles the C; Rust provides
   startup, soft-double (`compiler_builtins`) and the mailbox loop. No arm-gcc needed.
2. `host/` — Rust harness on picoem (`GIT_SRC_DIR/picoem`). Posts a kernel id to a SRAM
   mailbox, counts emulated cycles until done. Per-call cost = (cycles(1+R) − cycles(1)) / R,
   so one-time setup cancels and is reported separately.
3. Every output is checksummed and compared **bit for bit** with a native x86 build of the same
   C (`fw/c/ref_main.c`, shared `mathrt.c` so sin/cos/sqrt are identical). A difference is a
   finding (it found two picoem bugs, below), not noise.
4. The load tables turn cycles into "fraction of one core" at 8/16/48 kHz.

`sh run_eval.sh build OUTDIR && sh run_eval.sh run OUTDIR`, or the testlist
`css-ofdm-audiomodem.rp2350-dsp-eval`. About 15 s. Exit 0 = every kernel bit-identical.
`--dump ID PARAM ADDR FILE` (host) + `fw/c/ref_dump.c` diff a kernel's working memory.

## What it does not cover (read before trusting a verdict)

- **USB is not emulated.** The link table is arithmetic (FS bulk ≈ 1 MB/s ceiling, 1 ms frame
  latency floor). Interrupt load from USB, and real packet jitter, are untested. The real-board
  step is the same mailbox protocol over CDC; not built yet.
- **Timing model.** picoem is cycle-accurate for the core; bus contention and XIP-cache
  misses are only as good as picoem models them (`picoem/tech_debt.md`). Run the same kernels
  from SRAM vs flash before relying on a ±20 % number.
- **The workload model is a stated approximation**: demod = 1 rfft/symbol, acq = 1 rfft per
  n_fft/4 samples, CSS = one L-tap dot per sample. Equalisation, pilot tracking, the PLL, the
  real `ofdm_dem_frame`/`ofdm_acq_run` orchestration and the C CSS libs (libcssmf, libcsspll)
  are not run yet. Those are the next kernels to add.
- Single core only; the second M33 core is unused.

## picoem fixes this needed (in GIT_SRC_DIR/picoem)

- `VMOV Rt,Rt2,Dm` / `VMOV Dm,Rt,Rt2` (MCRR/MRRC on cp10/11) were UNDEFINED. clang uses
  them to hand soft-double operands to `__aeabi_*` helpers, so any double code faulted.
- VSEL / VMAXNM / VMINNM decoded D from `hw0[4]` and cc from `hw0[6:5]`; correct is
  D = `hw0[6]`, cc = `hw0[5:4]` (`fe36 1a01` = `vselgt s2,s12,s2`). The unit-test encoders had the
  same mistake, so the tests passed. Visible as `irfft f32` differing from x86.
