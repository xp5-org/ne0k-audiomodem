# Build the CSS matched-filter core.
#
#   make        -> libcssmf.<arch>-<os>.so
#   make clean  -> removes this host's build only
#   make clean-all
#
# This build rule is new.  libcssmf.so was previously a committed binary with
# no way to rebuild it, so there was no way to tell an edited css_mf.c from a
# compiled one, and no way for a second CPU type to obtain a working copy at
# all.  Both of those are now preflight failures rather than silent fallbacks,
# which only works if the library can actually be built — hence this file.
#
# WHY THE ARCH SUFFIX.  BASE_DIR is a shared SMB mount: every node sees this
# same directory.  With a plain `libcssmf.so` an arm64-apple node and an x86_64
# Linux node overwrite each other's build, and the loser silently falls back to
# the numpy MFSymbolFilter while still reporting a pass.  Suffixed names let
# both hosts hold a working build at once.
#
# The suffix comes from `cbackend.py --tag`, not from a second copy of the
# uname normalisation here, so the name the compiler writes is by construction
# the name the loader looks for.
#
# -Wall -Wextra -pedantic matches the OFDM core: this code ships to the
# embedded target, so warnings are worth keeping at zero here where they are
# cheap to fix.

CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -fPIC -Wall -Wextra -pedantic
LDLIBS  ?= -lm
PYTHON  ?= python3

_MK := $(dir $(lastword $(MAKEFILE_LIST)))
TAG := $(shell $(PYTHON) $(_MK)../../cbackend.py --tag)
EXT := $(shell $(PYTHON) $(_MK)../../cbackend.py --ext)

ifeq ($(strip $(TAG)),)
$(error could not determine host tag — is $(PYTHON) on PATH? \
        Set PYTHON=/path/to/python3)
endif

LIB     := libcssmf.$(TAG)$(EXT)
ENCLIB  := libcssenc.$(TAG)$(EXT)
ENC64   := libcssenc64.$(TAG)$(EXT)

SRC     := css_mf.c css_fft.c
HDR     := css_mf.h css_fft.h
ENCSRC  := css_enc.c css_dec.c
ENCHDR  := css_enc.h css_dec.h

all: $(LIB) $(ENCLIB) $(ENC64)

$(LIB): $(SRC) $(HDR)
	$(CC) $(CFLAGS) -shared -o $@ $(SRC) $(LDLIBS)

# The transmit chain is its own library rather than more objects inside
# libcssmf: a TX-only node has no matched filter, no FFT and no reason to carry
# 4.7 MB of segment-template workspace code, and keeping the split visible in
# the build is what stops that from quietly regrowing.
$(ENCLIB): $(ENCSRC) $(ENCHDR)
	$(CC) $(CFLAGS) -shared -o $@ $(ENCSRC) $(LDLIBS)

# Double-precision chirp phase AND Viterbi path metric.  NOT the shipping
# build -- it exists so the equivalence harness can separate "is this a port
# bug?" (grade the f64 build against numpy, which computes both in double)
# from "is float32 enough?" (grade the f32 build against the f64 one, on
# correlation for the chirp and on decoded bits for the Viterbi).  Same
# two-stage split the OFDM core uses for ofdm_real.
$(ENC64): $(ENCSRC) $(ENCHDR)
	$(CC) $(CFLAGS) -DCSS_ENC_F64_PHASE -DCSS_DEC_F64_METRIC \
	      -shared -o $@ $(ENCSRC) $(LDLIBS)

# Only this host's artifact: on the shared mount a blanket rm would delete the
# other nodes' working builds.
clean:
	rm -f $(LIB) $(ENCLIB) $(ENC64)

clean-all:
	rm -f libcssmf.*.so libcssmf.*.dylib libcssmf.so \
	      libcssenc.*.so libcssenc.*.dylib \
	      libcssenc64.*.so libcssenc64.*.dylib

# Strict-C89 syntax gate for the OpenWatcom / embedded target, mirroring the
# OFDM core's c89check.  Compile-only with the host compiler: it does not prove
# OpenWatcom accepts the code, but it catches the things that actually drift --
# a // comment, a mid-block declaration, a C99 libm call -- in about a second.
#
# -DCSS_C89_MATH is REQUIRED here, not optional: without it css_fabsf/css_log10f
# expand to the f-suffixed C99 entry points, which a C89 library does not
# declare.  Gating without the define would pass a build that cannot exist.
c89check:
	@for f in $(SRC) $(ENCSRC); do \
	  $(CC) -std=c89 -pedantic -Wall -Wextra -Wdeclaration-after-statement \
	        -Werror -DCSS_C89_MATH -c $$f -o /dev/null || exit 1; \
	  $(CC) -std=c89 -pedantic -Wall -Wextra -Wdeclaration-after-statement \
	        -Werror -DCSS_C89_MATH -DCSS_ENC_F64_PHASE -DCSS_DEC_F64_METRIC \
	        -c $$f -o /dev/null || exit 1; \
	done
	@echo "C89 OK: $(SRC) $(ENCSRC)"

.PHONY: all clean clean-all c89check
