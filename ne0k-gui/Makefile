# Builds the C cores that modemgui imports.
#
# The sources are NOT copied: they live in ../audiotestserver/tests/{OFDM,CSS,}/c, which stay the
# reference.  REF points at that tests/ directory (override for a Docker build
# context that mounts it elsewhere).  Output goes to build/ with unsuffixed
# names -- unlike tests/, this project is not run from a shared mount, so there
# is no cross-arch overwrite to defend against; the loader still checks the
# ELF/Mach-O machine before trusting a library (modemgui/native.py).
#
#   make            -> build/libofdm.so  build/libofdm32.so  build/libcssmf.so
#                      build/libcssenc.so
#   make check      -> load every library and resolve its entry points
#   make clean

REF     ?= ../audiotestserver/tests
CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -fPIC -Wall -Wextra -pedantic
LDLIBS  ?= -lm

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
EXT := .dylib
else ifneq (,$(findstring MINGW,$(UNAME_S))$(findstring MSYS,$(UNAME_S)))
EXT := .dll
else
EXT := .so
endif

OFDMC := $(REF)/OFDM/c
CSSC  := $(REF)/CSS/c
RSC   := $(REF)/c

OFDM_SRC := $(addprefix $(OFDMC)/,ofdm_acq.c ofdm_dem.c ofdm_fft.c ofdm_cfg.c ofdm_sym.c \
            ofdm_mod.c ofdm_probe.c ofdm_sync.c ofdm_frame.c ofdm_fdd.c ofdm_dx.c ofdm_arq.c ofdm_rxf.c ofdm_rxs.c) $(RSC)/rs255.c
CSSMF_SRC  := $(addprefix $(CSSC)/,css_mf.c css_fft.c)
CSSENC_SRC := $(addprefix $(CSSC)/,css_enc.c css_dec.c)

INC := -I$(OFDMC) -I$(CSSC) -I$(RSC)
LIBS := build/libofdm$(EXT) build/libofdm32$(EXT) build/libcssmf$(EXT) build/libcssenc$(EXT)

all: $(LIBS)

build:
	mkdir -p build

# Any header or source change in the reference tree rebuilds: a stale library
# loaded against edited C is the failure this guards.
DEPS := $(wildcard $(OFDMC)/*.h $(OFDMC)/*.c $(CSSC)/*.h $(CSSC)/*.c $(RSC)/*.h $(RSC)/*.c)

build/libofdm$(EXT): $(OFDM_SRC) $(DEPS) | build
	$(CC) $(CFLAGS) $(INC) -shared -o $@ $(OFDM_SRC) $(LDLIBS)

# float FFT: the 32-bit-target precision; kept loadable beside the double build.
build/libofdm32$(EXT): $(OFDM_SRC) $(DEPS) | build
	$(CC) $(CFLAGS) $(INC) -DOFDM_ACQ_F32 -shared -o $@ $(OFDM_SRC) $(LDLIBS)

build/libcssmf$(EXT): $(CSSMF_SRC) $(DEPS) | build
	$(CC) $(CFLAGS) $(INC) -shared -o $@ $(CSSMF_SRC) $(LDLIBS)

build/libcssenc$(EXT): $(CSSENC_SRC) $(DEPS) | build
	$(CC) $(CFLAGS) $(INC) -shared -o $@ $(CSSENC_SRC) $(LDLIBS)

check: all
	python3 -m modemgui.native

clean:
	rm -rf build

.PHONY: all check clean
