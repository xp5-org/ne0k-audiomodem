//! Compiles the NE0K C kernels with clang for thumbv8m.main (hard float) into one static
//! archive. No arm-none-eabi-gcc needed; Rust's compiler_builtins covers soft-double and mem*.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let core = std::env::var("OFDM_CORE").unwrap_or_else(|_| format!("{}/../../ofdm_dos/core", dir.display()));
    let c = dir.join("c");
    let common = [
        "--target=thumbv8m.main-none-eabihf", "-mcpu=cortex-m33", "-mfloat-abi=hard",
        "-O2", "-ffreestanding", "-fno-math-errno", "-ffp-contract=off", "-fno-builtin-sqrt",
        "-fno-builtin-sin", "-fno-builtin-cos", "-Wno-unused-command-line-argument",
    ];
    let inc = [format!("-I{}", c.join("inc").display()), format!("-I{}", c.display()), format!("-I{}", core)];
    let mathmap = ["-include".to_string(), c.join("mathmap.h").display().to_string()];
    // (object, source, extra flags)
    let f32r = ["-DOFDM_ACQ_F32", "-DPREC_F32", "-Dofdm_rfft=ofdm_rfft_f32", "-Dofdm_irfft=ofdm_irfft_f32",
                "-Dofdm_fft_init=ofdm_fft_init_f32", "-Dofdm_fft_workspace=ofdm_fft_workspace_f32"];
    let jobs: Vec<(&str, String, Vec<&str>)> = vec![
        ("fft64_core.o", format!("{core}/ofdm_fft.c"), vec![]),
        ("fft32_core.o", format!("{core}/ofdm_fft.c"), f32r.to_vec()),
        ("kern_fft64.o", c.join("kern_fft.c").display().to_string(), vec![]),
        ("kern_fft32.o", c.join("kern_fft.c").display().to_string(), f32r.to_vec()),
        ("rs255.o", format!("{core}/rs255.c"), vec![]),
        ("kern_main.o", c.join("kern_main.c").display().to_string(), vec![]),
        ("mathrt.o", c.join("mathrt.c").display().to_string(), vec![]),
    ];
    let mut objs = Vec::new();
    for (name, src, extra) in &jobs {
        let o = out.join(name);
        let st = Command::new("clang").args(common).args(&inc).args(&mathmap).args(extra)
            .arg("-c").arg(src).arg("-o").arg(&o).status().expect("clang not found");
        assert!(st.success(), "clang failed on {src}");
        objs.push(o);
    }
    let lib = out.join("libkernels.a");
    let _ = std::fs::remove_file(&lib);
    let st = Command::new("ar").arg("crs").arg(&lib).args(&objs).status().expect("ar not found");
    assert!(st.success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=kernels");
    println!("cargo:rerun-if-changed={}", c.display());
    println!("cargo:rerun-if-changed={core}");
    println!("cargo:rerun-if-changed=link.x");
    println!("cargo:rustc-link-arg=-L{}", dir.display());
}
