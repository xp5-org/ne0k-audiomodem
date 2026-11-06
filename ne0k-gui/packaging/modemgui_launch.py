# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Entry point of the frozen (PyInstaller) release.  The C libraries ship NEXT TO the executable
(build/), so users can see and replace them.  The reference DSP modules are frozen into the
executable; the release has no tests/ dir, but one placed beside the executable still overrides
them (MODEMGUI_TESTS_DIR).  No argument opens the GUI."""
import multiprocessing
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    base = Path(sys.executable).resolve().parent
    os.environ.setdefault("MODEMGUI_LIB_DIR", str(base / "build"))
    os.environ.setdefault("MODEMGUI_TESTS_DIR", str(base / "tests"))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from modemgui.cli import main
    sys.exit(main(sys.argv[1:] or ["gui"]))
