#!/bin/sh
# Incremental rebuild of the C cores (a no-op when nothing changed), then exec.
# A build failure is fatal: running the GUI against a stale library is worse
# than not starting.
set -e
cd /audiotestserver/modemgui
make -s
exec "$@"
