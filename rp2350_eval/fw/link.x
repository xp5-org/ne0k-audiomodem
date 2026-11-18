/* RP2350 image run from XIP flash like an SDK image; data in SRAM. */
MEMORY
{
  FLASH (rx)  : ORIGIN = 0x10000000, LENGTH = 2M
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 256K   /* mailbox lives at 0x20041000 */
}
ENTRY(reset)
SECTIONS
{
  .vector_table ORIGIN(FLASH) : { KEEP(*(.vector_table)) } > FLASH
  .text : { *(.text .text.*) } > FLASH
  .rodata : { *(.rodata .rodata.*) } > FLASH
  .data : ALIGN(4) { __sdata = .; *(.data .data.*) . = ALIGN(4); __edata = .; } > RAM AT > FLASH
  __sidata = LOADADDR(.data);
  .bss (NOLOAD) : ALIGN(8) { __sbss = .; *(.bss .bss.* COMMON) . = ALIGN(8); __ebss = .; } > RAM
  __end__ = .;
  __stack_top = ORIGIN(RAM) + LENGTH(RAM);
  /DISCARD/ : { *(.ARM.exidx*) *(.ARM.extab*) *(.comment) }
}
