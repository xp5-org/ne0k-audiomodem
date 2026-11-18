#!/bin/sh
# Build and run the RP2350 DSP-offload eval.
#   run_eval.sh build OUTDIR      builds fw (thumbv8m), x86 reference, host harness
#   run_eval.sh run   OUTDIR      runs it; exit 0 = PASS
# Reuses the local Rust toolchain that ad9850-sim's run_sim.sh installs into
# test_objoutput/ad9850_sim (rustup/cargo) when the host has none.
set -eu
cmd=${1:?usage: run_eval.sh build|run OUTDIR}
out=${2:?usage: run_eval.sh build|run OUTDIR}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
if ! command -v cargo >/dev/null 2>&1; then
    tc=${RUST_TOOLCHAIN_DIR:-$out/../ad9850_sim}
    RUSTUP_HOME=$tc/rustup; CARGO_HOME=$tc/cargo; export RUSTUP_HOME CARGO_HOME
    [ -x "$CARGO_HOME/bin/cargo" ] || { echo "no cargo: run the ad9850-picoem-usb testlist once first (it installs one)"; exit 1; }
    PATH=$CARGO_HOME/bin:$PATH; export PATH
fi
core=${OFDM_CORE:-$here/../ofdm_dos/core}
OFDM_CORE=$core; export OFDM_CORE
# kernel list the reference build must cover (id param ...); keep in sync with host/src/main.rs
KLIST="1 256 1 512 1 1024 1 2048 2 256 2 512 2 1024 2 2048 3 512 3 1024 4 512 4 1024 5 16 5 32 6 16 6 32 7 16 7 32 8 64 8 256 9 64 9 256 10 64 10 256 11 512 11 1024"
case "$cmd" in
build)
    rustup target add thumbv8m.main-none-eabihf >"$out/rustup_target.log" 2>&1 || { cat "$out/rustup_target.log"; exit 1; }
    ( cd "$here/fw" && CARGO_TARGET_DIR=$out/fwtarget cargo build --release >"$out/build_fw.log" 2>&1 ) || { tail -n 40 "$out/build_fw.log"; exit 1; }
    llvm-objcopy -O binary "$out/fwtarget/thumbv8m.main-none-eabihf/release/rp2350-eval-fw" "$out/fw.bin" 2>/dev/null \
        || objcopy -O binary "$out/fwtarget/thumbv8m.main-none-eabihf/release/rp2350-eval-fw" "$out/fw.bin"
    # native reference: same C, same flags that matter for bit-exactness
    c=$here/fw/c
    cc -O2 -ffp-contract=off -fno-builtin-sqrt -fno-builtin-sin -fno-builtin-cos -I"$c" -I"$core" -include "$c/mathmap.h" \
        -Dofdm_rfft=ofdm_rfft_f32 -Dofdm_irfft=ofdm_irfft_f32 -Dofdm_fft_init=ofdm_fft_init_f32 -Dofdm_fft_workspace=ofdm_fft_workspace_f32 \
        -DOFDM_ACQ_F32 -DPREC_F32 -c "$c/kern_fft.c" -o "$out/ref_fft32.o"
    cc -O2 -ffp-contract=off -fno-builtin-sqrt -fno-builtin-sin -fno-builtin-cos -I"$c" -I"$core" -include "$c/mathmap.h" \
        -Dofdm_rfft=ofdm_rfft_f32 -Dofdm_irfft=ofdm_irfft_f32 -Dofdm_fft_init=ofdm_fft_init_f32 -Dofdm_fft_workspace=ofdm_fft_workspace_f32 \
        -DOFDM_ACQ_F32 -c "$core/ofdm_fft.c" -o "$out/ref_fftcore32.o"
    cc -O2 -ffp-contract=off -fno-builtin-sqrt -fno-builtin-sin -fno-builtin-cos -I"$c" -I"$core" -include "$c/mathmap.h" -c "$c/kern_fft.c" -o "$out/ref_fft64.o"
    cc -O2 -ffp-contract=off -fno-builtin-sqrt -fno-builtin-sin -fno-builtin-cos -I"$c" -I"$core" -include "$c/mathmap.h" -c "$core/ofdm_fft.c" -o "$out/ref_fftcore64.o"
    for f in rs255.c; do cc -O2 -ffp-contract=off -I"$c" -I"$core" -include "$c/mathmap.h" -c "$core/$f" -o "$out/ref_${f%.c}.o"; done
    for f in kern_main mathrt ref_main; do cc -O2 -ffp-contract=off -fno-builtin-sqrt -fno-builtin-sin -fno-builtin-cos -I"$c" -I"$core" -include "$c/mathmap.h" -c "$c/$f.c" -o "$out/ref_$f.o"; done
    cc "$out"/ref_*.o -o "$out/host_ref" -lm
    "$out/host_ref" $KLIST >"$out/host_ref.txt"
    ( cd "$here/host" && CARGO_TARGET_DIR=$out/hosttarget cargo build --release >"$out/build_host.log" 2>&1 ) || { tail -n 40 "$out/build_host.log"; exit 1; }
    tail -n 2 "$out/build_host.log"
    ;;
run)
    rc=0
    "$out/hosttarget/release/rp2350-eval" --bin "$out/fw.bin" --ref "$out/host_ref.txt" >"$out/rp2350_eval.log" 2>&1 || rc=$?
    cat "$out/rp2350_eval.log"
    exit $rc
    ;;
*) echo "unknown command: $cmd" >&2; exit 2 ;;
esac
