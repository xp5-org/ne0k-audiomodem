# ofdm_dos

DOS-side OFDM/BAT modem test tools for the 86Box Mic/Line-In Echo Box project
(audio path testing — SB16 DAC/ADC, COM1 reporting — between a TX/RX pair of
86Box "echo box" VMs). `core/` holds the portable OFDM/Reed-Solomon codec
(ported from `sourcedir/86box_mic_echo/OFDM/c/`); the top-level `.C` files are
the on-device test programs built against it.

## Build (OpenWatcom, 32-bit DOS-extended target)

```
wcl386 -l=dos4g -3s -fpc -oneatx -DOFDM_C89_MATH -i=core <FILE>.C <needed core/*.c>
```

32-bit target is required, not 16-bit real mode: `ofdm_cfg.h` stores
`sample_rate`/`n_fft`/etc. as plain `int`, which overflows at 16 bits.

| file | needs from `core/` | purpose |
|---|---|---|
| `OFDMBAT.C` | `ofdm_cfg ofdm_fft ofdm_sym ofdm_acq ofdm_probe` | SB16 channel-probe + bit-allocation ("BAT") exchange. `ofdmbat TX`, `ofdmbat RX [out.wav]`, `ofdmbat RATESCAN`. Geometry comes from `OFDMBAT.INI`. |
| `OFDMDATA.C` | `ofdm_cfg ofdm_mod ofdm_dem ofdm_sync ofdm_frame rs255` | Two-VM OFDM data-frame TX/RX validation test over SB16. |
| `MODFTEST.C` | `ofdm_cfg ofdm_mod ofdm_dem ofdm_sync ofdm_frame rs255` | On-hardware self-test of the data-frame codec: CPU/DSP self-loop only (no SB16/DMA), reports per-stage BIOS-tick timing over COM1. |
| `RSFTEST.C` | `rs255 ofdm_frame` | On-hardware self-test of the RS-coded interleave layer alone, same self-loop/timing convention as `MODFTEST.C`. |

`battest.c` is the odd one out: a **host-side** (not DOS) round-trip harness
for `batcodec.c`, built with `gcc -std=c89 -DOFDM_C89_MATH -I core -o battest
battest.c batcodec.c core/ofdm_cfg.c core/ofdm_fft.c core/ofdm_sym.c
core/ofdm_acq.c -lm` — run it on the dev host before spending an emulator run
on a codec change.

`*.INI` files are modem geometry config read by the matching `.EXE` at
runtime (TX and RX must read the same file).
