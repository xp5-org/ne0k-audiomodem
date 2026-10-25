# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""cbackend.py — one place that knows which C cores exist, for which CPU, and
whether this host can actually load them.

`BASE_DIR` is a shared SMB mount: all nodes see the same `tests/` directory,
so a wrong-CPU `.so` left by another host must be detectable rather than
silently falling back to numpy/reedsolo while a run claims to exercise C.

Two mechanisms:

  * Arch-suffixed filenames. `libcssmf.x86_64-linux.so` and
    `libcssmf.arm64-darwin.dylib` are different paths, so multiple hosts can
    hold a working build on the shared mount at once. `host_tag()` defines
    the suffix; the Makefiles use the identical expression.

  * requested() vs available(). `requested()` reads the env vars that select
    a backend; `require()` raises rather than returning False. A bridge
    keeps its silent numpy default only when nobody asked for C.

The arch check reads the ELF / Mach-O header directly instead of `dlopen`
and catching the error, because dlopen's failure text is platform-specific
and does not distinguish "wrong CPU" from "missing libm".
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────────────
# Host identity
# ──────────────────────────────────────────────────────────────────────────────

# platform.machine() is not canonical across hosts: Apple silicon reports
# "arm64", Linux on the same ISA reports "aarch64", x86 answers "x86_64" or
# "AMD64" depending on OS. Collapsed so the tag baked into a filename by one
# host matches what another host looks for.
_MACHINE_ALIASES = {
    "amd64": "x86_64", "x86_64": "x86_64", "x64": "x86_64",
    "aarch64": "arm64", "arm64": "arm64",
    "armv7l": "armv7", "armv6l": "armv6",
    "i386": "i386", "i686": "i386",
}


def norm_machine(m: str) -> str:
    return _MACHINE_ALIASES.get(str(m).lower(), str(m).lower())


def host_machine() -> str:
    return norm_machine(platform.machine())


def host_os() -> str:
    s = sys.platform
    if s.startswith("linux"):
        return "linux"
    if s == "darwin":
        return "darwin"
    if s.startswith("win"):
        return "windows"
    return s


def host_tag() -> str:
    """`<machine>-<os>`, e.g. `x86_64-linux` or `arm64-darwin`.

    The Makefiles compute this same string; keep the two in step.
    """
    return f"{host_machine()}-{host_os()}"


def so_ext() -> str:
    return ".dylib" if host_os() == "darwin" else ".so"


def lib_filename(stem: str) -> str:
    """`libcssmf` -> `libcssmf.x86_64-linux.so` for this host."""
    return f"{stem}.{host_tag()}{so_ext()}"


# ──────────────────────────────────────────────────────────────────────────────
# Reading the CPU out of a binary, without loading it
# ──────────────────────────────────────────────────────────────────────────────

_ELF_MACHINE = {
    3: "i386", 40: "armv7", 62: "x86_64", 183: "arm64", 243: "riscv",
}

# Mach-O cputype, low 32 bits; bit 24 (CPU_ARCH_ABI64) is set for 64-bit.
_MACHO_CPU = {
    7: "i386", 0x01000007: "x86_64", 12: "armv7", 0x0100000C: "arm64",
}


def binary_machine(path: Path) -> str | None:
    """Normalised CPU name a shared library was built for, or None if unknown.

    Handles ELF (Linux) and both endiannesses of 64-bit Mach-O (macOS).  A
    Mach-O universal ("fat") binary reports `universal`, which loads anywhere
    and so is never a mismatch.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
    except OSError:
        return None
    if len(head) < 20:
        return None

    if head[:4] == b"\x7fELF":
        little = head[5] == 1
        e_machine = int.from_bytes(head[18:20], "little" if little else "big")
        return _ELF_MACHINE.get(e_machine, f"elf:{e_machine}")

    magic = int.from_bytes(head[:4], "little")
    if magic in (0xFEEDFACF, 0xFEEDFACE):            # thin Mach-O, LE host
        cpu = int.from_bytes(head[4:8], "little")
        return _MACHO_CPU.get(cpu, f"macho:{cpu}")
    if magic in (0xCFFAEDFE, 0xCEFAEDFE):            # byte-swapped
        cpu = int.from_bytes(head[4:8], "big")
        return _MACHO_CPU.get(cpu, f"macho:{cpu}")
    if int.from_bytes(head[:4], "big") in (0xCAFEBABE, 0xCAFEBABF):
        return "universal"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# The registry of C cores
# ──────────────────────────────────────────────────────────────────────────────

class Backend:
    """One buildable C core: where it lives, what it is built from, and the env
    var that selects it."""

    def __init__(self, key, directory, stem, sources, env, doc):
        self.key       = key
        self.directory = directory
        self.stem      = stem
        self.sources   = sources
        self.env       = env          # (VAR, value) that selects this backend
        self.doc       = doc

    # -- paths -----------------------------------------------------------
    @property
    def path(self) -> Path:
        """The arch-suffixed path for THIS host.  Always the build target."""
        return self.directory / lib_filename(self.stem)

    @property
    def legacy_path(self) -> Path:
        """Pre-arch-suffix name. Accepted only when its header matches this
        host."""
        return self.directory / f"{self.stem}.so"

    def resolve(self) -> Path | None:
        """Path this host should load, or None if nothing usable is present."""
        if self.path.exists():
            return self.path
        legacy = self.legacy_path
        if legacy.exists() and binary_machine(legacy) in (host_machine(),
                                                          "universal"):
            return legacy
        return None

    # -- state -----------------------------------------------------------
    def stale_sources(self, lib: Path) -> list[str]:
        try:
            so_mtime = lib.stat().st_mtime
        except OSError:
            return []
        return [s for s in self.sources
                if (self.directory / s).exists()
                and (self.directory / s).stat().st_mtime > so_mtime]

    def probe(self) -> dict:
        """Everything known about this backend on this host.  Never raises.

        `ok` means: a library for this CPU is present and dlopen succeeded.
        `reason` is filled in only when `ok` is False, and names the specific
        problem so a failing node's log says what to rebuild and where.
        """
        rec = {
            "key": self.key, "stem": self.stem, "host": host_tag(),
            "expected": str(self.path), "path": None, "arch": None,
            "present": False, "arch_ok": False, "loadable": False,
            "stale": [], "ok": False, "reason": None,
        }
        lib = self.resolve()

        if lib is None:
            # Distinguish "nothing there" from "there, but for another CPU".
            other = None
            for cand in sorted(self.directory.glob(f"{self.stem}*")):
                if cand.suffix in (".so", ".dylib"):
                    other = (cand, binary_machine(cand))
                    break
            if other:
                rec["reason"] = (
                    f"{self.stem}: no build for {host_tag()}; found "
                    f"{other[0].name} built for {other[1] or 'unknown CPU'}. "
                    f"Run `make -C {self.directory}` on this host.")
            else:
                rec["reason"] = (f"{self.stem}: not built. Run "
                                 f"`make -C {self.directory}` on this host.")
            return rec

        rec["path"] = str(lib)
        rec["present"] = True
        rec["arch"] = binary_machine(lib)
        rec["arch_ok"] = rec["arch"] in (host_machine(), "universal", None)
        rec["stale"] = self.stale_sources(lib)

        if not rec["arch_ok"]:
            rec["reason"] = (f"{lib.name} is built for {rec['arch']}, this host "
                             f"is {host_machine()}. Run "
                             f"`make -C {self.directory}` on this host.")
            return rec

        try:
            ctypes.CDLL(str(lib))
        except OSError as exc:
            # Right CPU but still fails to load: missing dependency, corrupt
            # file, or hardening policy. Pass the loader's own message through.
            rec["reason"] = f"{lib.name} failed to load: {exc}"
            return rec

        rec["loadable"] = True
        rec["ok"] = True
        return rec


_TESTS = _HERE

BACKENDS: dict[str, Backend] = {
    b.key: b for b in (
        Backend("css_mf", _TESTS / "CSS" / "c", "libcssmf",
                ("css_mf.c", "css_mf.h", "css_fft.c", "css_fft.h"),
                ("CSS_MF_BACKEND", "c"),
                "CSS matched filter (MFSymbolFilter)"),
        Backend("css_enc", _TESTS / "CSS" / "c", "libcssenc",
                ("css_enc.c", "css_enc.h", "css_dec.c", "css_dec.h"),
                ("CSS_ENC_BACKEND", "c"),
                "CSS frame codec (chirp, conv/Viterbi, interleave, CRC)"),
        Backend("css_enc64", _TESTS / "CSS" / "c", "libcssenc64",
                ("css_enc.c", "css_enc.h", "css_dec.c", "css_dec.h"),
                ("CSS_ENC_BACKEND", "c"),
                "CSS frame codec, float64 phase/metric (grading build)"),
        Backend("ofdm_acq", _TESTS / "OFDM" / "c", "libofdmacq",
                ("ofdm_acq.c", "ofdm_dem.c", "ofdm_fft.c",
                 "ofdm_acq.h", "ofdm_dem.h", "ofdm_fft.h"),
                ("OFDM_ACQ_BACKEND", "c"),
                "OFDM acquisition + demodulator, double"),
        Backend("ofdm_acq32", _TESTS / "OFDM" / "c", "libofdmacq32",
                ("ofdm_acq.c", "ofdm_dem.c", "ofdm_fft.c",
                 "ofdm_acq.h", "ofdm_dem.h", "ofdm_fft.h"),
                ("OFDM_ACQ_BACKEND", "c"),
                "OFDM acquisition + demodulator, float32 target build"),
        Backend("rs255", _TESTS / "c", "librs255",
                ("rs255.c", "rs255.h"),
                ("RS_BACKEND", "c"),
                "Reed-Solomon GF(256), shared by CSS/OFDM/QAM"),
    )
}


# ──────────────────────────────────────────────────────────────────────────────
# What the environment is asking for
# ──────────────────────────────────────────────────────────────────────────────

def requested(env=None) -> list[str]:
    """Backend keys the current environment explicitly selects.

    Empty means the numpy/reedsolo defaults are in play and a missing `.so`
    is not a problem; any key here means it must load or the run is invalid.
    """
    env = os.environ if env is None else env
    keys = []
    for key, b in BACKENDS.items():
        var, want = b.env
        if env.get(var, "").lower() != want:
            continue
        if key == "ofdm_acq" and env.get("OFDM_ACQ_LIB", "") == "32":
            continue      # the 32 build is the one that will be loaded
        if key == "ofdm_acq32" and env.get("OFDM_ACQ_LIB", "") != "32":
            continue
        keys.append(key)
    return keys


class BackendUnavailable(RuntimeError):
    """Raised when a backend was explicitly requested and cannot be delivered."""


def require(key: str) -> Path:
    """Path to the loadable library for `key`, or raise with the reason.

    Bridges call this on the requested path; they must NOT call it on the
    default path, or building nothing would stop being a valid state.
    """
    rec = BACKENDS[key].probe()
    if not rec["ok"]:
        raise BackendUnavailable(rec["reason"])
    return Path(rec["path"])


def inventory(keys=None) -> list[dict]:
    """probe() for every backend (or the named subset). Never raises."""
    keys = list(BACKENDS) if keys is None else list(keys)
    return [BACKENDS[k].probe() for k in keys]


def check(keys=None, env=None) -> tuple[bool, list[str]]:
    """Grade the backends `env` requests.  Returns (ok, problems).

    Only requested backends can fail: an unrequested, unbuilt core is the
    default configuration, not a fault. Staleness is also reported as a
    problem for a requested backend.
    """
    want = [k for k in requested(env) if keys is None or k in keys]
    problems = []
    for key in want:
        rec = BACKENDS[key].probe()
        if not rec["ok"]:
            problems.append(rec["reason"])
        elif rec["stale"]:
            problems.append(
                f"{Path(rec['path']).name} is older than "
                f"{', '.join(rec['stale'])} — run "
                f"`make -C {BACKENDS[key].directory}`")
    return (not problems), problems


def banner(env=None) -> str:
    """One line naming every backend actually in force, for the run log."""
    want = requested(env)
    if not want:
        return f"C_BACKENDS=none (numpy/reedsolo defaults) host={host_tag()}"
    parts = []
    for key in want:
        rec = BACKENDS[key].probe()
        parts.append(f"{key}={Path(rec['path']).name if rec['ok'] else 'BROKEN'}")
    return f"C_BACKENDS={' '.join(parts)} host={host_tag()}"


def _cli(argv) -> int:
    """`python tests/cbackend.py [--json|--tag|--ext]` — what this host can load.

    `--tag` and `--ext` exist for the Makefiles, which ask this function
    rather than keeping a separate copy of the normalisation table.
    """
    if "--tag" in argv:
        print(host_tag())
        return 0
    if "--ext" in argv:
        print(so_ext())
        return 0
    recs = inventory()
    if "--json" in argv:
        import json
        print(json.dumps({"host": host_tag(), "requested": requested(),
                          "backends": recs}, indent=2))
    else:
        print(f"host {host_tag()}   requested: {', '.join(requested()) or 'none'}")
        for r in recs:
            mark = "ok  " if r["ok"] else "MISS"
            name = Path(r["path"]).name if r["path"] else "-"
            print(f"  [{mark}] {r['key']:<11} {name}")
            if r["reason"]:
                print(f"         {r['reason']}")
            if r["stale"]:
                print(f"         STALE vs {', '.join(r['stale'])}")
    ok, problems = check()
    for p in problems:
        print(f"FAIL: {p}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
