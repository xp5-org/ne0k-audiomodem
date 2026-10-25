# Shared C cores — used by more than one modem, so they live here rather than
# under tests/<MODEM>/c/.
#
#   make        -> librs255.<arch>-<os>.so   (Reed-Solomon, GF(256))
#   make clean  -> removes this host's build only
#   make clean-all
#
# rs255 has no floating point and no FFT, so it builds standalone with only
# libc's headers for size_t.  It does not even need -lm.
#
# WHY THE ARCH SUFFIX.  BASE_DIR is a shared SMB mount: every node sees this
# same directory.  With a plain `librs255.so` an arm64-apple node and an x86_64
# Linux node overwrite each other's build, and the loser silently falls back to
# pure-Python reedsolo while still reporting a pass.  Suffixed names let both
# hosts hold a working build at once.
#
# The suffix comes from `cbackend.py --tag`, not from a second copy of the
# uname normalisation here, so the name the compiler writes is by construction
# the name the loader looks for.
#
# The .so depends on sources AND headers so an edited core cannot be silently
# left uncompiled; cbackend also reports a library older than its sources as a
# preflight failure when the backend is explicitly requested.

CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -fPIC -Wall -Wextra -pedantic
PYTHON  ?= python3

TAG := $(shell $(PYTHON) $(dir $(lastword $(MAKEFILE_LIST)))../cbackend.py --tag)
EXT := $(shell $(PYTHON) $(dir $(lastword $(MAKEFILE_LIST)))../cbackend.py --ext)

ifeq ($(strip $(TAG)),)
$(error could not determine host tag — is $(PYTHON) on PATH? \
        Set PYTHON=/path/to/python3)
endif

LIB := librs255.$(TAG)$(EXT)

SRC := rs255.c
HDR := rs255.h

all: $(LIB)

$(LIB): $(SRC) $(HDR)
	$(CC) $(CFLAGS) -shared -o $@ $(SRC)

# Only this host's artifact: on the shared mount a blanket rm would delete the
# other nodes' working builds.
clean:
	rm -f $(LIB)

clean-all:
	rm -f librs255.*.so librs255.*.dylib librs255.so

.PHONY: all clean clean-all
