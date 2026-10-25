# Build/check the mesh-layer C89 port.
#
#   make selftest   -> builds and runs mesh_phy_selftest
#   make c89check   -> strict-C89 syntax gate for the OpenWatcom target
#   make clean
#
# Unlike tests/OFDM/c/, there is no numpy oracle to grade against here and no
# shared library to load into Python -- mesh_phy.c has no DSP and nothing in
# this repo calls it from Python, so it builds straight to a host test binary.
# See ../README.md (this directory's) for what's ported and what's not.

CC     ?= cc
CFLAGS ?= -O2 -std=c99 -fPIC -Wall -Wextra -pedantic

SRC := mesh_phy.c
HDR := mesh_phy.h

selftest: mesh_phy_selftest
	./mesh_phy_selftest

mesh_phy_selftest: $(SRC) $(HDR) mesh_phy_selftest.c
	$(CC) $(CFLAGS) $(SRC) mesh_phy_selftest.c -o $@

clean:
	rm -f mesh_phy_selftest

# Strict-C89 syntax gate for the OpenWatcom target -- see tests/OFDM/c/Makefile
# for the rationale (host compiler, compile-only, catches // comments, mid-
# block declarations, etc. at negligible cost).
c89check:
	@$(CC) -std=c89 -pedantic -Wall -Wextra -Wdeclaration-after-statement \
	      -c $(SRC) -o /dev/null
	@echo "C89 OK: $(SRC)"

.PHONY: selftest clean c89check
