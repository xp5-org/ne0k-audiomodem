# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Loads the C cores (built by ``make``) and proves they are the ones we expect.

Three independent failures look identical from the GUI -- "nothing decodes" --
so each gets its own message here instead of a bare OSError:

  * the library is missing                      -> run ``make``
  * it was built for another CPU/OS              -> rebuild on this host
  * it loads but lacks an entry point            -> stale build vs edited headers

``python -m modemgui.native`` prints the inventory and runs a smoke test that
exercises real ABI (a struct crossing the boundary, a workspace-carved
modulator) rather than just dlopen.
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
import zlib
from pathlib import Path

LIB_DIR_ENV = "MODEMGUI_LIB_DIR"
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "build"

_EXT = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")

# stem -> entry points the GUI depends on.  The set is the contract: a build
# that drops one has diverged from the headers this was written against.
MANIFEST: dict[str, tuple[str, ...]] = {
    "libofdm": ("ofdm_cfg_build", "ofdm_mod_init", "ofdm_mod_workspace",
                "ofdm_mod_encode", "ofdm_mod_signal_samples", "ofdm_acq_init",
                "ofdm_acq_run", "ofdm_dem_init", "ofdm_dem_frame", "ofdm_sync_find",
                "ofdm_probe_estimate", "ofdm_mod_tables_flat", "rs255_init",
                "ofdm_fdd_plan_bands", "ofdm_fdd_sub_bat", "ofdm_fdd_chunk_for_frame_len",
                "ofdm_dx_init", "ofdm_dx_tx_frame", "ofdm_dx_rx_frame", "ofdm_dx_capacity",
                "ofdm_rxf_design", "ofdm_rxs_init", "ofdm_rxs_feed", "ofdm_rxs_pop",
                "ofdm_arq_init", "ofdm_arq_send", "ofdm_arq_on_frame", "ofdm_arq_poll"),
    "libofdm32": ("ofdm_acq_init", "ofdm_dem_init", "ofdm_mod_init"),
    "libcssmf": ("css_mf_init", "css_mf_run", "css_mf_workspace"),
    "libcssenc": ("css_enc_bits", "css_dec_raw", "css_viterbi_k7", "css_crc32",
                  "css_check_crc"),
}

_ELF_MACHINE = {3: "i386", 40: "armv7", 62: "x86_64", 183: "arm64", 243: "riscv"}
_ALIAS = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64",
          "armv7l": "armv7", "i686": "i386"}


class NativeError(RuntimeError):
    pass


def lib_dir() -> Path:
    return Path(os.environ.get(LIB_DIR_ENV) or _DEFAULT_DIR)


def _host_machine() -> str:
    m = platform.machine().lower()
    return _ALIAS.get(m, m)


def _elf_machine(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            h = f.read(20)
    except OSError:
        return None
    if h[:4] != b"\x7fELF" or len(h) < 20:
        return None                       # Mach-O / PE: leave it to dlopen
    return _ELF_MACHINE.get(int.from_bytes(h[18:20], "little" if h[5] == 1 else "big"))


_cache: dict[str, ctypes.CDLL] = {}


def load(stem: str) -> ctypes.CDLL:
    """dlopen build/<stem>.so, verifying CPU and the manifest's entry points."""
    if stem in _cache:
        return _cache[stem]
    if stem not in MANIFEST:
        raise NativeError(f"unknown library {stem!r}; have {sorted(MANIFEST)}")
    path = lib_dir() / f"{stem}{_EXT}"
    if not path.exists():
        raise NativeError(f"{path} not found -- run `make` in {_DEFAULT_DIR.parent} "
                          f"(or set {LIB_DIR_ENV})")
    built = _elf_machine(path)
    if built and built != _host_machine():
        raise NativeError(f"{path.name} is built for {built}, this host is "
                          f"{_host_machine()} -- `make clean && make`")
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as e:
        raise NativeError(f"cannot load {path}: {e}") from e
    missing = [s for s in MANIFEST[stem] if not hasattr(lib, s)]
    if missing:
        raise NativeError(f"{path.name} lacks {', '.join(missing)} -- stale build, "
                          f"`make clean && make`")
    _cache[stem] = lib
    return lib


def inventory() -> list[dict]:
    out = []
    for stem in MANIFEST:
        try:
            load(stem)
            out.append({"lib": stem, "ok": True, "path": str(lib_dir() / f"{stem}{_EXT}")})
        except NativeError as e:
            out.append({"lib": stem, "ok": False, "error": str(e)})
    return out


# -- ABI mirrors -------------------------------------------------------------

class OfdmCfg(ctypes.Structure):
    """ofdm_cfg from ofdm_cfg.h; field order IS the ABI."""
    _fields_ = [(n, ctypes.c_int) for n in (
        "n_fft", "cp_len", "bin_lo", "bin_hi", "n_occupied", "pilot_spacing",
        "n_pilots", "n_data_carriers", "sample_rate", "symbol_samples",
        "taper_samples", "frame_gap_samples", "ecc_bytes", "chunk_bytes",
        "seq_bytes", "sync_period")] + [(n, ctypes.c_double) for n in (
        "cp_ratio", "sc_spacing_hz", "freq_lo_hz", "freq_hi_hz")]


def smoke() -> list[str]:
    """Exercise real calls.  Returns a list of failures (empty = all good)."""
    bad: list[str] = []

    css = load("libcssenc")
    css.css_crc32.restype = ctypes.c_ulong
    css.css_crc32.argtypes = [ctypes.c_char_p, ctypes.c_int]
    msg = b"modemgui smoke 123456789"
    if css.css_crc32(msg, len(msg)) != zlib.crc32(msg):
        bad.append("css_crc32 disagrees with zlib.crc32")

    of = load("libofdm")
    c = OfdmCfg()
    of.ofdm_cfg_build.argtypes = [ctypes.POINTER(OfdmCfg), ctypes.c_int, ctypes.c_double,
                                  ctypes.c_double, ctypes.c_double, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_double, ctypes.c_double, ctypes.c_int]
    # Same defaults as ofdm_modem.build_config.
    rc = of.ofdm_cfg_build(ctypes.byref(c), 1024, 0.25, 1000.0, 16000.0, 8, 48000,
                           16, 256, 2, 2.0, 20.0, 32)
    if rc != 0:
        return bad + [f"ofdm_cfg_build rc={rc}"]
    if (c.n_occupied, c.n_pilots + c.n_data_carriers) != (c.n_occupied, c.n_occupied):
        bad.append("cfg: pilots + data != occupied")

    bpb = (ctypes.c_int * c.n_data_carriers)(*([2] * c.n_data_carriers))   # all-QPSK
    of.ofdm_mod_workspace.restype = ctypes.c_size_t
    of.ofdm_mod_workspace.argtypes = [ctypes.POINTER(OfdmCfg), ctypes.POINTER(ctypes.c_int)]
    ws = of.ofdm_mod_workspace(ctypes.byref(c), bpb)
    if not ws:
        return bad + ["ofdm_mod_workspace rejected the default geometry"]
    mem = ctypes.create_string_buffer(ws)
    mod = ctypes.c_void_p()
    of.ofdm_mod_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                 ctypes.c_size_t, ctypes.POINTER(OfdmCfg),
                                 ctypes.POINTER(ctypes.c_int)]
    rc = of.ofdm_mod_init(ctypes.byref(mod), mem, ws, ctypes.byref(c), bpb)
    if rc != 0:
        return bad + [f"ofdm_mod_init rc={rc}"]
    payload = b"hello ofdm"
    of.ofdm_mod_signal_samples.restype = ctypes.c_long
    of.ofdm_mod_signal_samples.argtypes = [ctypes.c_void_p, ctypes.c_int]
    n = of.ofdm_mod_signal_samples(mod, len(payload))
    out = (ctypes.c_double * n)()
    of.ofdm_mod_encode.restype = ctypes.c_long
    of.ofdm_mod_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_double), ctypes.c_long]
    wrote = of.ofdm_mod_encode(mod, payload, len(payload), out, n)
    if wrote != n:
        bad.append(f"ofdm_mod_encode wrote {wrote}, signal_samples said {n}")
    elif not any(out[i] for i in range(0, n, 97)):
        bad.append("ofdm_mod_encode produced silence")
    return bad


def _main() -> int:
    ok = True
    for row in inventory():
        print(("ok   " if row["ok"] else "FAIL ") + row["lib"] + "  " +
              row.get("path", row.get("error", "")))
        ok &= row["ok"]
    if ok:
        problems = smoke()
        for p in problems:
            print("FAIL smoke:", p)
        ok = not problems
        if ok:
            print("ok   smoke (crc32, cfg struct ABI, modulator encode)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main())
