//! Kernel-bench firmware. Polls a mailbox in SRAM, runs one kernel, posts the checksum.
//! Mailbox (0x2004_1000), word index:
//!   [0] cmd seq (host writes last)  [1] id  [2] param  [3] reps
//!   [4] done seq (fw)  [5] checksum  [6] workspace bytes  [7] alive magic
#![no_std]
#![no_main]

use core::panic::PanicInfo;
use core::ptr::{read_volatile, write_volatile};

const MB: *mut u32 = 0x2004_1000 as *mut u32;

extern "C" {
    fn kernel_run(id: u32, param: u32, reps: u32, ws: *mut u32) -> u32;
    static mut __sdata: u32;
    static mut __edata: u32;
    static __sidata: u32;
    static mut __sbss: u32;
    static mut __ebss: u32;
}

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

// Vector table: initial SP, reset, then fault slots that spin (a hang is visible to the
// harness as a missing done-seq, with the PC readable from the emulator).
core::arch::global_asm!(
    ".section .vector_table,\"a\"",
    ".word __stack_top",
    ".word reset",
    ".word hang",
    ".word hang",
    ".word hang",
    ".word hang",
    ".word hang",
    ".text",
    ".global hang",
    ".thumb_func",
    "hang: b hang",
);

#[no_mangle]
pub unsafe extern "C" fn reset() -> ! {
    // FPU: full access to CP10/CP11.
    write_volatile(0xE000_ED88 as *mut u32, read_volatile(0xE000_ED88 as *const u32) | (0xF << 20));
    core::arch::asm!("dsb", "isb");

    let mut s = &raw const __sidata;
    let mut d = &raw mut __sdata;
    while d < &raw mut __edata {
        write_volatile(d, read_volatile(s));
        d = d.add(1);
        s = s.add(1);
    }
    let mut b = &raw mut __sbss;
    while b < &raw mut __ebss {
        write_volatile(b, 0);
        b = b.add(1);
    }

    write_volatile(MB.add(7), 0xE7A1_0001);
    loop {
        let seq = read_volatile(MB);
        if seq == read_volatile(MB.add(4)) {
            continue;
        }
        let mut ws = 0u32;
        let sum = kernel_run(
            read_volatile(MB.add(1)),
            read_volatile(MB.add(2)),
            read_volatile(MB.add(3)),
            &mut ws,
        );
        write_volatile(MB.add(5), sum);
        write_volatile(MB.add(6), ws);
        write_volatile(MB.add(4), seq); // done: last
    }
}
