# Build the OFDM acquisition + demodulator core.
#
#   make            -> libofdmacq.<arch>-<os>.so   (double FFT; numpy oracle)
#   make f32        -> libofdmacq32.<arch>-<os>.so (float FFT; 32-bit target)
#   make both
#   make clean      -> removes this host's builds only
#   make clean-all
#
# The .so depends on the sources AND the headers, so a stale library cannot be
# loaded against edited C.  cbackend also reports a library older than its
# sources as a preflight failure when the backend is explicitly requested.
#
# WHY THE ARCH SUFFIX.  BASE_DIR is a shared SMB mount: every node sees this
# same directory.  With plain `libofdmacq.so` an arm64-apple node and an x86_64
# Linux node overwrite each other's build, and the loser silently falls back to
# the numpy scan while still reporting a pass — which is exactly the comparison
# this port's regression suite exists to make honest.  Suffixed names let both
# hosts hold a working build at once.
#
# The suffix comes from `cbackend.py --tag`, not from a second copy of the
# uname normalisation here, so the name the compiler writes is by construction
# the name the loader looks for.
#
# -Wall -Wextra -pedantic is not decoration: this code has to survive a strict
# embedded toolchain, so warnings are worth keeping at zero here where they are
# cheap to fix.
#
# C89 / OPENWATCOM.  Every core here is C89: no // comments, no declarations
# after statements, no stdint/bool/restrict.  A DOS, Win3.1 or NT4 build with
# OpenWatcom (32-bit protected mode) compiles the same sources with
#
#     wcc386 -za -ot -DOFDM_C89_MATH ...
#
# `-DOFDM_C89_MATH` swaps hypot/hypotf -- C99 and POSIX, absent from a strict
# C89 library, where an implicit int declaration would silently corrupt every
# magnitude in the acquisition metric -- for sqrt forms.  It is opt-in because
# the substitution is not bit-identical and the host equivalence suites grade
# the C against numpy, whose complex abs IS hypot.  Verify a change still
# builds for that target with:
#
#     make c89check

CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -fPIC -Wall -Wextra -pedantic -I../../c
LDLIBS  ?= -lm
PYTHON  ?= python3

_MK := $(dir $(lastword $(MAKEFILE_LIST)))
TAG := $(shell $(PYTHON) $(_MK)../../cbackend.py --tag)
EXT := $(shell $(PYTHON) $(_MK)../../cbackend.py --ext)

ifeq ($(strip $(TAG)),)
$(error could not determine host tag — is $(PYTHON) on PATH? \
        Set PYTHON=/path/to/python3)
endif

LIB64 := libofdmacq.$(TAG)$(EXT)
LIB32 := libofdmacq32.$(TAG)$(EXT)

SRC := ofdm_acq.c ofdm_dem.c ofdm_fft.c ofdm_cfg.c ofdm_sym.c ofdm_mod.c ofdm_probe.c ofdm_sync.c ofdm_frame.c ../../c/rs255.c
HDR := ofdm_acq.h ofdm_dem.h ofdm_fft.h ofdm_cfg.h ofdm_sym.h ofdm_mod.h ofdm_probe.h ofdm_sync.h ofdm_frame.h ../../c/rs255.h

# ofdm_hs (HandshakeSM) is DSP-free and has no numpy oracle to grade against
# -- it is not part of the .so, and is checked separately with its own host
# selftest below rather than a regression .py.
HSSRC := ofdm_hs.c
HSHDR := ofdm_hs.h

all: $(LIB64)

$(LIB64): $(SRC) $(HDR)
	$(CC) $(CFLAGS) -shared -o $@ $(SRC) $(LDLIBS)

# 32-bit target precision: ofdm_real becomes float, which is what CMSIS-DSP
# and the RP2350 FPU want.  Built as a separate library so both precisions can
# be loaded and compared in one process.
f32: $(LIB32)

$(LIB32): $(SRC) $(HDR)
	$(CC) $(CFLAGS) -DOFDM_ACQ_F32 -shared -o $@ $(SRC) $(LDLIBS)

both: $(LIB64) $(LIB32)

# HandshakeSM: no .so, no numpy oracle -- just a host test binary.
hs_selftest: ofdm_hs_selftest
	./ofdm_hs_selftest

ofdm_hs_selftest: $(HSSRC) $(HSHDR) ofdm_hs_selftest.c
	$(CC) $(CFLAGS) $(HSSRC) ofdm_hs_selftest.c -o $@

# Only this host's artifacts: on the shared mount a blanket rm would delete the
# other nodes' working builds.
clean:
	rm -f $(LIB64) $(LIB32) ofdm_hs_selftest

clean-all:
	rm -f libofdmacq.*.so libofdmacq.*.dylib libofdmacq.so \
	      libofdmacq32.*.so libofdmacq32.*.dylib libofdmacq32.so

# Strict-C89 syntax gate for the OpenWatcom target.  Compile-only, host
# compiler: it does not prove OpenWatcom accepts the code, but it catches the
# things that actually drift -- a // comment, a mid-block declaration, a C99
# libm call -- at the cost of one second.
c89check:
	@for f in $(SRC); do \
	  $(CC) -std=c89 -pedantic -Wall -Wextra -Wdeclaration-after-statement \
	        -DOFDM_C89_MATH -I$(_MK)../../c -c $$f -o /dev/null || exit 1; \
	done
	@$(CC) -std=c89 -pedantic -Wall -Wextra -Wdeclaration-after-statement \
	      -c $(HSSRC) -o /dev/null
	@echo "C89 OK: $(SRC) $(HSSRC)"

.PHONY: all f32 both hs_selftest clean clean-all c89check
