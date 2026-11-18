//! rp2350-eval: boots the kernel-bench firmware (fw/) in picoem, times each DSP kernel in
//! emulated Cortex-M33 cycles, checks its output bit-for-bit against a native x86 build of
//! the same C, then turns cycles into per-core load for the modem workloads.
//!
//! Timing = (cycles(reps = 1+R) - cycles(reps = 1)) / R, so one-time setup cancels and is
//! reported separately. Not modelled: USB, XIP-cache misses beyond what picoem models
//! (see README caveats). Exit 0 = firmware alive and every kernel bit-identical to host.

use rp2350_emu::{Config, Emulator};
use std::collections::HashMap;
use std::fs;
use std::process::exit;

const MB: u32 = 0x2004_1000;
const SYS_HZ: f64 = 150e6;

const K_RFFT_F64: u32 = 1;
const K_RFFT_F32: u32 = 2;
const K_IRFFT_F64: u32 = 3;
const K_IRFFT_F32: u32 = 4;
const K_RS_ENC: u32 = 5;
const K_RS_DEC_CLEAN: u32 = 6;
const K_RS_DEC_ERR: u32 = 7;
const K_DOT_F32: u32 = 8;
const K_DOT_F64: u32 = 9;
const K_FIR_Q15: u32 = 10;
const K_CMUL_F32: u32 = 11;

struct Bench {
    emu: Emulator,
    seq: u32,
}

impl Bench {
    fn new(fw: &[u8], rom: &[u8]) -> Bench {
        let mut emu = Emulator::new(Config::default());
        emu.load_bootrom(rom);
        emu.load_flash(fw);
        emu.reset();
        emu.core_mut(1).halt();
        emu.bus.atomics.rcp_salt_set(0, 0x1357_9bdf);
        emu.bus.atomics.rcp_salt_set(1, 0x2468_ace0);
        let sp = u32::from_le_bytes(fw[0..4].try_into().unwrap());
        let pc = u32::from_le_bytes(fw[4..8].try_into().unwrap());
        for i in 0..2 {
            let c = emu.core_mut(i);
            c.ppb.cpacr |= 0x0000_CD03;
            c.regs.faultmask = 0;
            c.regs.primask = 0;
        }
        let c = emu.core_mut(0);
        c.ppb.vtor = 0x1000_0000;
        c.regs.r[13] = sp;
        c.regs.msp = sp;
        c.regs.set_pc(pc & !1);
        emu.step_quantum = 32;
        for w in 0..8 {
            emu.mmio_write32(MB + 4 * w, 0);
        }
        Bench { emu, seq: 0 }
    }

    fn rd(&mut self, w: u32) -> u32 {
        self.emu.mmio_read32(MB + 4 * w)
    }

    fn boot(&mut self) -> bool {
        for _ in 0..200_000 {
            self.emu.run(32).unwrap();
            if self.rd(7) == 0xE7A1_0001 {
                return true;
            }
        }
        false
    }

    /// Run one kernel call to completion. Returns (cycles, checksum, workspace bytes).
    fn call(&mut self, id: u32, param: u32, reps: u32, limit: u64) -> Option<(u64, u32, u32)> {
        self.seq += 1;
        self.emu.mmio_write32(MB + 4, id);
        self.emu.mmio_write32(MB + 8, param);
        self.emu.mmio_write32(MB + 12, reps);
        let t0 = self.emu.cycles();
        self.emu.mmio_write32(MB, self.seq);
        loop {
            self.emu.run(32).unwrap();
            if self.rd(4) == self.seq {
                let c = self.emu.cycles() - t0;
                return Some((c, self.rd(5), self.rd(6)));
            }
            if self.emu.cycles() - t0 > limit {
                let c = self.emu.core_mut(0);
                let sp = c.regs.r[13];
                let (cfsr, hfsr) = (c.ppb.cfsr, c.ppb.hfsr);
                let spv = sp;
                let _ = c;
                let frame: Vec<u32> = (0..8).map(|k| self.emu.mmio_read32(spv + 4 * k)).collect();
                println!("fault: cfsr={cfsr:#x} hfsr={hfsr:#x} frame r0..xpsr={frame:x?}");
                let c = self.emu.core_mut(0);
                println!("timeout: core0 pc={:#010x} lr={:#010x} sp={:#010x}", c.regs.r[15], c.regs.r[14], c.regs.r[13]);
                return None;
            }
        }
    }
}

struct Row {
    name: String,
    id: u32,
    param: u32,
    per: f64,   // cycles per rep
    setup: f64, // one-time cycles
    ws: u32,
    ok: Option<bool>,
}

fn main() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let mut bin = String::new();
    let mut rom = format!("{dir}/../../../picoem/roms/rp2350/bootrom-combined.bin");
    let mut refp = String::new();
    let mut quick = false;
    // --dump ID PARAM ARENA_ADDR FILE: write the kernel's working memory after reps=2 (debug aid)
    let mut dump: Option<(u32, u32, u32, String)> = None;
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--bin" => { bin = argv[i + 1].clone(); i += 1; }
            "--rom" => { rom = argv[i + 1].clone(); i += 1; }
            "--ref" => { refp = argv[i + 1].clone(); i += 1; }
            "--quick" => quick = true,
            "--dump" => { dump = Some((argv[i + 1].parse().unwrap(), argv[i + 2].parse().unwrap(), argv[i + 3].parse().unwrap(), argv[i + 4].clone())); i += 4; }
            "-h" | "--help" => {
                println!("usage: rp2350-eval --bin fw.bin --ref host_ref.txt [--rom F] [--quick]");
                exit(0);
            }
            a => fail(&format!("unknown argument {a}")),
        }
        i += 1;
    }
    if bin.is_empty() { fail("--bin required"); }
    let fw = fs::read(&bin).unwrap_or_else(|e| fail(&format!("{bin}: {e}")));
    let rom = fs::read(&rom).unwrap_or_else(|e| fail(&format!("rom: {e}")));
    let mut refs: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
    if !refp.is_empty() {
        let t = fs::read_to_string(&refp).unwrap_or_else(|e| fail(&format!("{refp}: {e}")));
        for l in t.lines() {
            let f: Vec<&str> = l.split_whitespace().collect();
            if f.len() == 4 {
                refs.insert((f[0].parse().unwrap(), f[1].parse().unwrap()),
                            (u32::from_str_radix(f[2], 16).unwrap(), f[3].parse().unwrap()));
            }
        }
    }

    let mut b = Bench::new(&fw, &rom);
    if !b.boot() { fail("firmware did not post its alive word (boot failed)"); }
    println!("firmware: {} bytes, alive", fw.len());
    if let Some((id, p, addr, file)) = dump {
        let (_, _, ws) = b.call(id, p, 2, 400_000_000).unwrap_or_else(|| fail("dump call timed out"));
        let mut bytes = Vec::new();
        for k in 0..(ws + 3) / 4 { bytes.extend_from_slice(&b.emu.mmio_read32(addr + 4 * k).to_le_bytes()); }
        bytes.truncate(ws as usize);
        fs::write(&file, bytes).unwrap();
        println!("dumped {ws} bytes to {file}");
        exit(0);
    }
    match b.call(0, 0, 1, 2_000_000) {
        Some((c, _, _)) => println!("nop call: {c} cycles round trip (mailbox overhead)"),
        None => fail("NOP kernel call did not complete: mailbox/dispatch broken"),
    }

    // (label, id, param list, reps R)
    let sweeps: Vec<(&str, u32, Vec<u32>, u32)> = vec![
        ("rfft  f64 (as shipped)", K_RFFT_F64, vec![256, 512, 1024, 2048], 1),
        ("rfft  f32 (OFDM_ACQ_F32)", K_RFFT_F32, vec![256, 512, 1024, 2048], 2),
        ("irfft f64", K_IRFFT_F64, vec![512, 1024], 1),
        ("irfft f32", K_IRFFT_F32, vec![512, 1024], 2),
        ("rs255 encode nsym", K_RS_ENC, vec![16, 32], 4),
        ("rs255 decode clean nsym", K_RS_DEC_CLEAN, vec![16, 32], 4),
        ("rs255 decode nsym/2 errors", K_RS_DEC_ERR, vec![16, 32], 2),
        ("dot f32 (correlator) taps", K_DOT_F32, vec![64, 256], 32),
        ("dot f64 taps", K_DOT_F64, vec![64, 256], 8),
        ("fir q15 taps", K_FIR_Q15, vec![64, 256], 32),
        ("cmul f32 (dechirp) n", K_CMUL_F32, vec![512, 1024], 16),
    ];
    let mut rows: Vec<Row> = Vec::new();
    let mut bad = 0;
    for (name, id, params, r) in &sweeps {
        for &p in params {
            if quick && p > 512 { continue; }
            let c1 = b.call(*id, p, 1, 400_000_000).unwrap_or_else(|| fail(&format!("{name} {p}: timed out")));
            let cn = b.call(*id, p, 1 + r, 400_000_000).unwrap_or_else(|| fail(&format!("{name} {p}: timed out")));
            let per = (cn.0 as f64 - c1.0 as f64) / *r as f64;
            let setup = c1.0 as f64 - per;
            let chk = b.call(*id, p, 2, 400_000_000).unwrap();
            let ok = refs.get(&(*id, p)).map(|&(s, w)| s == chk.1 && w == chk.2);
            if ok == Some(false) { bad += 1; }
            if cn.2 == 0 && *id != 0 { bad += 1; println!("note: {name} {p}: kernel refused (workspace over arena?)"); }
            rows.push(Row { name: name.to_string(), id: *id, param: p, per, setup, ws: cn.2, ok });
            println!("  measured {name} {p}: {:.0} cycles/call", per);
        }
    }

    println!("\n== kernel cost on emulated Cortex-M33 @ 150 MHz ==");
    println!("{:<30} {:>6} {:>12} {:>10} {:>11} {:>9} {:>7}", "kernel", "param", "cycles/call", "us/call", "setup cyc", "RAM B", "x86 ref");
    for r in &rows {
        println!("{:<30} {:>6} {:>12.0} {:>10.1} {:>11.0} {:>9} {:>7}", r.name, r.param, r.per, r.per / SYS_HZ * 1e6, r.setup, r.ws,
                 match r.ok { Some(true) => "bit-ok", Some(false) => "DIFF", None => "-" });
    }

    let per = |id: u32, p: u32| rows.iter().find(|r| r.id == id && r.param == p).map(|r| r.per);

    // --- workload model -------------------------------------------------------------
    // Assumptions are printed with the numbers; change them in the model, not the kernels.
    println!("\n== OFDM workload (load = cycles/s needed / 150e6; 1.00 = one full core) ==");
    println!("model: 1 rfft per received symbol (demod) + 1 irfft per sent symbol (mod); acq/sync adds");
    println!("       one rfft per hop of n_fft/4 samples while searching; RS and equalise not included.");
    println!("{:>8} {:>6} {:>9} | {:>9} {:>9} {:>9} {:>9}", "fs Hz", "n_fft", "sym/s", "demod f64", "demod f32", "acq f64", "acq f32");
    for &fs in &[8000.0f64, 16000.0, 48000.0] {
        for &n in &[256u32, 512, 1024, 2048] {
            let (Some(f64c), Some(f32c)) = (per(K_RFFT_F64, n), per(K_RFFT_F32, n)) else { continue };
            let cp = (n as f64 * 0.25).round();
            let sym = fs / (n as f64 + cp);
            let hop = fs / (n as f64 / 4.0);
            println!("{:>8.0} {:>6} {:>9.1} | {:>9.2} {:>9.2} {:>9.2} {:>9.2}", fs, n, sym,
                     f64c * sym / SYS_HZ, f32c * sym / SYS_HZ, f64c * hop / SYS_HZ, f32c * hop / SYS_HZ);
        }
    }
    println!("\n== correlator / CSS-style streaming (per received sample at fs) ==");
    println!("model: one L-tap dot product per input sample (sliding matched filter).");
    println!("{:>8} {:>6} | {:>10} {:>10} {:>10}", "fs Hz", "taps", "f32", "f64", "q15");
    for &fs in &[8000.0f64, 16000.0, 48000.0] {
        for &t in &[64u32, 256] {
            let (Some(a), Some(c), Some(q)) = (per(K_DOT_F32, t), per(K_DOT_F64, t), per(K_FIR_Q15, t)) else { continue };
            println!("{:>8.0} {:>6} | {:>10.2} {:>10.2} {:>10.2}", fs, t, a * fs / SYS_HZ, c * fs / SYS_HZ, q * fs / SYS_HZ);
        }
    }
    println!("\n== Reed-Solomon (255-byte codewords per second one core can sustain at 100% load) ==");
    for &n in &[16u32, 32] {
        if let (Some(e), Some(d0), Some(d1)) = (per(K_RS_ENC, n), per(K_RS_DEC_CLEAN, n), per(K_RS_DEC_ERR, n)) {
            println!("  nsym {n}: encode {:.0}/s  decode clean {:.0}/s  decode worst-case {:.0}/s", SYS_HZ / e, SYS_HZ / d0, SYS_HZ / d1);
        }
    }

    // --- USB link (analytic; USB is NOT emulated) -----------------------------------
    println!("\n== USB full-speed bulk link budget (analytic, not emulated) ==");
    println!("raw FS = 12 Mbit/s; practical bulk ceiling ~1.0 MB/s (19 x 64 B per 1 ms frame, shared).");
    println!("{:>8} {:>6} | {:>12} {:>9} {:>12}", "fs Hz", "format", "MB/s both ways", "% of 1 MB/s", "min latency");
    for &fs in &[8000.0f64, 16000.0, 48000.0] {
        for &(fmt, bytes) in &[("int16", 2.0f64), ("f32", 4.0)] {
            let mbs = fs * bytes * 2.0 / 1e6;
            println!("{:>8.0} {:>6} | {:>12.3} {:>9.1} {:>12}", fs, fmt, mbs, mbs / 1.0 * 100.0, "1 ms + chunk");
        }
    }
    println!("latency floor = one 1 ms USB frame each way plus the audio chunk; the DSP time above adds to it.");

    println!();
    if bad == 0 {
        println!("RESULT: PASS ({} kernel measurements, all outputs bit-identical to x86 where a reference was supplied)", rows.len());
    } else {
        println!("RESULT: FAIL ({bad} kernel(s) differ from the x86 reference or were refused)");
        exit(1);
    }
}

fn fail(msg: &str) -> ! {
    println!("RESULT: FAIL ({msg})");
    exit(1);
}
