# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Command line for exercising the audio layer before any GUI exists.

    python -m modemgui devices                    what can I open?
    python -m modemgui monitor --rx pulse:        live level meter
    python -m modemgui play --tx sd:3 --tone 1500 send a tone
    python -m modemgui loopback --tx A --rx B     play a tone, check it is heard
    python -m modemgui selfloop --rx A --tx B     send text, decode it on the same pipe
    python -m modemgui serial --link /tmp/modem0 ...  pty + RTS/CTS handshake
    python -m modemgui native                     check the C libraries
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .audio import AudioError, AudioPipe, Registry, StreamSpec
from .audio.file_backend import read_wav
from .engine.selftest import _db, loopback, tone


def cmd_devices(a, reg: Registry) -> int:
    if a.json:
        print(json.dumps({
            "connectors": {n: {"available": ok, "reason": why,
                               "description": reg.connectors[n].description}
                           for n, (ok, why) in reg.status().items()},
            "devices": [{"address": d.address, "name": d.name, "kind": d.kind,
                         "default": d.default, "monitor": d.monitor,
                         "rate_hint": d.rate_hint} for d in reg.list_devices(a.kind)],
            "errors": reg.errors}, indent=2))
        return 0
    for n, (ok, why) in reg.status().items():
        print(f"[{n}] {'ok' if ok else 'UNAVAILABLE: ' + why}  - {reg.connectors[n].description}")
    for d in reg.list_devices(a.kind):
        flags = ("*" if d.default else " ") + ("M" if d.monitor else " ")
        rate = f"{d.rate_hint}Hz" if d.rate_hint else ""
        print(f"  {flags} {d.kind:6} {d.address:60} {rate:8} {d.name}")
    for k, v in reg.errors.items():
        print(f"  ! {k}: {v}", file=sys.stderr)
    return 0


def cmd_monitor(a, reg: Registry) -> int:
    with AudioPipe(reg, StreamSpec(a.rate, a.block)).open(a.rx, None) as p:
        tap = p.subscribe()
        p.start()
        end = time.monotonic() + a.seconds if a.seconds else None
        try:
            while end is None or time.monotonic() < end:
                b = tap.get(timeout=1.0)
                if b is None:
                    if p.rx_finished:
                        break
                    if p.stats.rx_error:
                        print("error:", p.stats.rx_error, file=sys.stderr)
                        return 1
                    continue
                rms = float(np.sqrt(np.mean(b.samples ** 2)))
                bar = "#" * max(0, int((_db(rms) + 80) / 2))
                print(f"\r{_db(b.peak):7.1f} dBpk {_db(rms):7.1f} dBrms "
                      f"clip={p.stats.clipped:<5} {bar:40.40}", end="", flush=True)
        except KeyboardInterrupt:
            pass
    print()
    return 0


def cmd_play(a, reg: Registry) -> int:
    x = read_wav(a.wav, a.rate) if a.wav else tone(a.tone, a.seconds, a.rate)
    with AudioPipe(reg, StreamSpec(a.rate, a.block)).open(None, a.tx) as p:
        p.start()
        p.play(x)
        if not p.wait_tx(len(x) / a.rate + 5):
            print("timed out waiting for TX to drain", file=sys.stderr)
            return 1
        time.sleep(0.3)                    # let the OS buffer play out
    return 0


def cmd_loopback(a, reg: Registry) -> int:
    """Play a tone out of --tx and look for it on --rx.  Exit 0 when heard."""
    r = loopback(reg, a.tx, a.rx, a.rate, a.block, a.tone, a.seconds)
    print(r.summary())
    if r.error:
        print("errors:", r.error, file=sys.stderr)
    return 0 if r.ok else 1


def _css_cfg(a):
    from .engine.modemcfg import PRESETS
    c = PRESETS[a.preset]
    c.hear_own = getattr(a, "hear_own", False)
    if getattr(a, "band_lo", None):
        c.f_min = a.band_lo
    if getattr(a, "band_hi", None):
        c.f_max = a.band_hi
    return c


def _wait_decoded(sess, timeout: float, seen: list | None = None):
    """Block until a Decoded event arrives; None on timeout.  Other events go to ``seen``."""
    from .engine import Decoded
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for ev in sess.drain():
            if isinstance(ev, Decoded):
                return ev
            if seen is not None:
                seen.append(ev)
        time.sleep(0.1)
    return None


def cmd_send(a, reg: Registry) -> int:
    from .engine import Session
    from .engine.css_link import CssLink
    s = Session(reg, a.rate, a.block)
    s.open(None, a.tx)
    air = CssLink(s, _css_cfg(a)).send(a.text)
    print(f"sending {len(a.text)} chars, airtime {air:.1f}s")
    s.pipe.wait_tx(air + 10)
    time.sleep(0.5)
    s.close()
    return 0


def cmd_recv(a, reg: Registry) -> int:
    from .engine import Session
    from .engine.css_link import CssLink
    s = Session(reg, a.rate, a.block)
    s.open(a.rx, None)
    link = CssLink(s, _css_cfg(a)); link.start()
    d = _wait_decoded(s, a.seconds)
    link.stop(); s.close()
    if d is None:
        print("RX: nothing decoded", file=sys.stderr)
        return 1
    print(f"RX: {d.text!r} src={d.src} seq={d.seq} ok={d.ok} snr={d.snr_db:.1f}dB")
    return 0 if (a.expect is None or d.text == a.expect) else 1


def cmd_selfloop(a, reg: Registry) -> int:
    """Send --text out of --tx and decode it from --rx in one process."""
    from .engine import Session
    from .engine.css_link import BACKEND, CssLink
    a.hear_own = True
    s = Session(reg, a.rate, a.block)
    s.open(a.rx, a.tx)
    log: list[str] = []
    link = CssLink(s, _css_cfg(a), log=log.append); link.start()
    air = link.send(a.text)
    print(f"backend={BACKEND} airtime={air:.1f}s")
    seen: list = []
    d = _wait_decoded(s, air + a.timeout, seen)
    st = s.pipe.stats if s.pipe else None
    link.stop(); s.close()
    if d is None:
        print("SELFLOOP FAIL: nothing decoded", file=sys.stderr)
        print(f"  rx={a.rx} tx={a.tx} rate={a.rate}", file=sys.stderr)
        if st:
            print(f"  pipe: rx_blocks={st.rx_blocks} rx_samples={st.rx_samples} tx_samples={st.tx_samples} "
                  f"peak={st.peak:.4f} clipped={st.clipped} rx_error={st.rx_error} tx_error={st.tx_error}",
                  file=sys.stderr)
        for m in log:
            print(f"  link: {m}", file=sys.stderr)
        for ev in seen:
            print(f"  event: {ev}", file=sys.stderr)
        return 1
    print(f"RX: {d.text!r} src={d.src} seq={d.seq} ok={d.ok} snr={d.snr_db:.1f}dB")
    good = d.ok and d.text == a.text
    print("SELFLOOP " + ("PASS" if good else "FAIL"))
    return 0 if good else 1


def cmd_decodecheck(a, reg) -> int:
    """No audio device: encode a packet and decode it straight back, printing what the decoder saw.
    Separates a broken codec (frozen build, C library) from a broken audio path."""
    import numpy as np
    from .engine import css_link
    from .engine.modemcfg import CssConfig
    cfg = CssConfig(hear_own=True)
    pcm = css_link.encode_text(cfg, a.text)
    print(f"encoded {len(pcm)} samples, peak {float(np.max(np.abs(pcm))):.3f}")
    cap = np.concatenate([np.zeros(int(0.37 * cfg.fs), np.float32), pcm])
    for shift in (0, 0.5, 0.25, 0.75):
        out = css_link.decode_samples(cfg, np.concatenate(
            [np.zeros(int(shift * cfg.symbol_dur * cfg.fs), np.float32), cap]))
        print(f"shift {shift}: {[(d.text, d.ok, d.frames) for d in out]}")
        if out and out[0].ok and out[0].text == a.text:
            print("DECODECHECK PASS")
            return 0
    _, cm, tl = css_link._imports()
    res = tl.decode_array(np.concatenate([np.zeros(cfg.fs, np.float32), cap, np.zeros(cfg.fs, np.float32)]),
                          cfg.fs, css_link._info(cfg))
    print({k: (v if k != "frames" else len(v)) for k, v in res.items() if k != "diagnostics"})
    print("DECODECHECK FAIL")
    return 1


def cmd_serial(a, reg: Registry) -> int:
    """Virtual serial port: a pty plus RTS/CTS control socket over the CSS or OFDM link."""
    import dataclasses
    from .engine import Session
    from .engine.modemcfg import OfdmConfig
    from .engine.serialbridge import SerialBridge
    from .engine.diag import Diagnostics
    from .engine.modemcfg import DiagConfig
    s = Session(reg, a.rate, a.block)
    s.open(a.rx, a.tx)
    cap = a.capture_wav
    diag = Diagnostics(DiagConfig(debug_log=a.debug_log, wav_rx=cap in ("rx", "both"),
                                  wav_tx=cap in ("tx", "both"), out_dir=a.diag_dir))
    diag.start_log(); diag.attach(s)
    if a.ident:
        from .engine.modemcfg import HamConfig
        s.ident.cfg = HamConfig(enabled=True, callsign=a.ident, id_mode=a.id_mode,
                                id_interval_s=a.id_interval, id_end_delay_s=a.id_end_delay)
        s.ident.log = lambda m: (diag.log(m), print("serial:", m, flush=True, file=sys.__stdout__))
    dst = None
    if a.modem == "mesh":
        from .engine.mesh_node import MeshConfig, MeshNode
        if a.dst is None:
            raise SystemExit("serial --modem mesh needs --dst (the node id the bytes go to)")
        ocfg = OfdmConfig(fs=a.rate, freq_lo=a.band_lo or 400.0, freq_hi=a.band_hi or 4000.0,
                          chunk_bytes=a.chunk, hear_own=a.hear_own, retries=a.retries, backend=a.backend)
        cfg = MeshNode(s, MeshConfig(node_id=a.node, phy_mode=a.mesh_phy), _css_cfg(a), ocfg,
                       log=lambda m: (diag.log("mesh: " + m), print("mesh:", m, flush=True, file=sys.__stdout__)))
        dst = a.dst
    elif a.modem == "ofdm":
        cfg = OfdmConfig(fs=a.rate, node_id=a.node, peer_id=a.peer, role=a.role,
                         window_s=a.window, guard_s=a.guard, chunk_bytes=a.chunk,
                         freq_lo=a.band_lo or 400.0, freq_hi=a.band_hi or 4000.0,
                         hear_own=a.hear_own, mode=a.ofdm_mode, sender=a.sender, duplex_guard_hz=a.duplex_guard,
                         backend=a.backend,
                         **{k: v for k, v in (("gap_s", a.gap), ("hang_s", a.hang), ("max_bits_per_bin", a.max_bits),
                                              ("snr_margin_db", a.snr_margin), ("stream_poll_s", a.stream_poll))
                            if v is not None})
    else:
        cfg = dataclasses.replace(_css_cfg(a), node_id=a.node, peer_id=a.peer,
                                  hear_own=a.hear_own)
    buf = a.buffer if a.buffer is not None else (4000 if a.modem == "ofdm" and a.ofdm_mode == "stream" else 32)
    br = SerialBridge(s, cfg, a.link, buf, a.mode, a.idle,
                      log=lambda m: (diag.log("serial: " + m), print("serial:", m, flush=True, file=sys.__stdout__)),
                      ack=a.ack, retries=a.retries, dst=dst)
    br.start()
    end = time.monotonic() + a.seconds if a.seconds else None
    try:
        while end is None or time.monotonic() < end:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    br.stop(); s.close(); diag.close()
    if diag.dir:
        print(f"serial: diagnostics in {diag.dir}", flush=True)
    print(f"serial: done tx={br.tx_count}B rx={br.rx_count}B retries={br.retry_count} failed={br.fail_count}", flush=True)
    return 0


def cmd_faxmodem(a, reg: Registry) -> int:
    """Hayes + Class 2 fax modem on an 86Box COM port; the phone line is a SIP call carrying the OFDM link."""
    if sys.platform.startswith("win"):
        print("faxmodem: POSIX only (no pty to attach to on Windows)", file=sys.stderr)
        return 2
    import asyncio
    from .faxmodem import mock, ptyhelpers
    pty = a.pty
    if not pty:
        if not a.vm_path:
            print("faxmodem: give --vm-path (the 86Box -P directory) or --pty", file=sys.stderr)
            return 2
        try:
            pty = ptyhelpers.discover_ptys(a.vm_path, a.com, a.find_timeout)[a.com - 1]
        except RuntimeError as e:
            print(f"faxmodem: {e}", file=sys.stderr)
            return 1
    fax = {"connect_timeout_s": 40.0, "page_timeout_s": 150.0, "train_timeout_s": 60.0,
           "dcn_timeout_s": 20.0, "train_ms": 0, "answer_delay_ms": 0}
    if a.save_dir:
        fax["save_dir"] = a.save_dir
    modem = {"port": a.com, "name": a.name or f"ne0k{a.user}", "user": a.user, "baud": a.baud,
             "media": "ne0k", "fclass": a.fclass, "fax": fax, "listen_port": a.listen_port,
             "ring_interval_ms": a.ring_ms, "pty": pty, "carrier_timeout_s": a.carrier_timeout,
             "ofdm": {"freq_lo": a.band_lo, "freq_hi": a.band_hi, "max_bits_per_bin": a.max_bits}}
    cfg = {"sip": {"server": a.sip_server, "local_host": a.local_host, "register": not a.no_register},
           "modems": [modem]}
    asyncio.run(mock.amain(cfg, [pty]))
    return 0


def cmd_gui(a, reg: Registry) -> int:
    from .ui.app import main as gui_main
    return gui_main(rx=a.rx, tx=a.tx, autosend=a.autosend, hear_own=a.hear_own, settings_file=a.settings)


def cmd_native(a, reg) -> int:
    from . import native
    return native._main()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ne0k-modem", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, rx=False, tx=False):
        p.add_argument("--rate", type=int, default=48000)
        p.add_argument("--block", type=int, default=1024)
        if rx:
            p.add_argument("--rx", required=True, help="source address, e.g. pulse:NAME, sd:3, file:x.wav")
        if tx:
            p.add_argument("--tx", required=True, help="sink address")

    p = sub.add_parser("devices"); p.set_defaults(fn=cmd_devices)
    p.add_argument("--kind", choices=["source", "sink"]); p.add_argument("--json", action="store_true")
    p = sub.add_parser("monitor"); common(p, rx=True); p.set_defaults(fn=cmd_monitor)
    p.add_argument("--seconds", type=float, default=0)
    p = sub.add_parser("play"); common(p, tx=True); p.set_defaults(fn=cmd_play)
    p.add_argument("--tone", type=float, default=1000.0); p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--wav")
    p = sub.add_parser("loopback"); common(p, rx=True, tx=True); p.set_defaults(fn=cmd_loopback)
    p.add_argument("--tone", type=float, default=1500.0); p.add_argument("--seconds", type=float, default=1.5)
    def css(p):
        p.add_argument("--preset", default="Fast (POC)", choices=["Fast (POC)", "Robust", "Over the air"])
    p = sub.add_parser("send"); common(p, tx=True); css(p); p.set_defaults(fn=cmd_send)
    p.add_argument("--text", required=True)
    p = sub.add_parser("recv"); common(p, rx=True); css(p); p.set_defaults(fn=cmd_recv)
    p.add_argument("--seconds", type=float, default=60.0); p.add_argument("--expect")
    p = sub.add_parser("selfloop"); common(p, rx=True, tx=True); css(p); p.set_defaults(fn=cmd_selfloop)
    p.add_argument("--text", default="hello modem"); p.add_argument("--timeout", type=float, default=30.0)
    p = sub.add_parser("decodecheck", help="encode then decode in memory (no audio device)")
    p.add_argument("--text", default="one click"); p.set_defaults(fn=cmd_decodecheck)
    p = sub.add_parser("serial", help="virtual serial port (pty + RTS/CTS socket) over CSS or OFDM")
    common(p, rx=True, tx=True); css(p); p.set_defaults(fn=cmd_serial)
    p.add_argument("--link", required=True, help="symlink to the pty; control socket is LINK.ctl")
    p.add_argument("--modem", choices=["css", "ofdm", "mesh"], default="css",
                   help="ofdm: BAT handshake, then the --ofdm-mode data plane; "
                        "mesh: CSS control + on-demand OFDM link to --dst")
    p.add_argument("--dst", type=int, help="mesh: node id this port's bytes are sent to")
    p.add_argument("--mesh-phy", choices=["css_then_ofdm", "css_only"], default="css_then_ofdm")
    p.add_argument("--role", choices=["auto", "initiator", "responder"], default="auto",
                   help="ofdm: who probes first (auto: the lower node id)")
    p.add_argument("--ofdm-mode", choices=["half", "oneway", "stream", "full"], default="half",
                   help="ofdm: half = windows with ARQ; oneway/stream = only --sender transmits, no ACK; "
                        "full = band split by direction, both ends talk at once")
    p.add_argument("--sender", choices=["initiator", "responder", "both"], default="initiator",
                   help="ofdm oneway/stream: which role transmits (the other end listens)")
    p.add_argument("--duplex-guard", type=float, default=600.0, help="ofdm full: dead band between directions, Hz")
    p.add_argument("--window", type=float, default=5.0, help="ofdm: TX/RX window seconds (both ends equal)")
    p.add_argument("--guard", type=float, default=1.0, help="ofdm: dead time at each window edge")
    p.add_argument("--chunk", type=int, default=32, help="ofdm: payload bytes per data frame")
    p.add_argument("--gap", type=float, default=None, help="ofdm oneway/stream: quiet between bursts, s (default 1.0)")
    p.add_argument("--hang", type=float, default=None, help="ofdm oneway/stream: burst ends after this much quiet, s (0.6)")
    p.add_argument("--stream-poll", type=float, default=None, help="ofdm stream: decode interval, s (1.0)")
    p.add_argument("--max-bits", type=int, choices=[1, 2, 4, 6], default=None,
                   help="ofdm: max bits per carrier (2 = QPSK default; 4 = 16-QAM needs ~25 dB SNR)")
    p.add_argument("--snr-margin", type=float, default=None, help="ofdm: dB headroom over the modulation floor (9)")
    p.add_argument("--band-lo", type=float, default=None, help="Hz; OFDM default 400, CSS keeps its preset")
    p.add_argument("--band-hi", type=float, default=None, help="Hz; OFDM default 4000.  3300 keeps both inside a phone line")
    p.add_argument("--backend", choices=["python", "c", "cdx"], default="python",
                   help="ofdm/mesh: DSP cores. c = libofdmacq + librs255 (fails if missing), python = numpy/reedsolo; "
                        "cdx = c + the full-duplex data plane (ofdm --mode full) run by the C engine")
    p.add_argument("--buffer", type=int, default=None, help="TX buffer bytes (1..250; ofdm oneway/stream up to 1048576; default 4000 for ofdm stream, else 32)")
    p.add_argument("--mode", choices=["rts", "auto", "fifo"], default="rts",
                   help="rts: send on RTS rising edge; auto: send when full or idle; "
                        "fifo (ofdm oneway/stream): CTS stays up while the buffer has room, bursts chain back to back")
    p.add_argument("--idle", type=float, default=2.0, help="auto mode idle seconds")
    p.add_argument("--ack", action=argparse.BooleanOptionalAction, default=True,
                   help="peer ACKs each buffer; retransmit when missing (default on)")
    p.add_argument("--retries", type=int, default=3, help="retransmissions before TXFAIL")
    p.add_argument("--node", type=int, default=1); p.add_argument("--peer", type=int, default=2)
    p.add_argument("--seconds", type=float, default=0, help="exit after this long (0 = until Ctrl-C)")
    p.add_argument("--hear-own", action="store_true")
    p.add_argument("--ident", metavar="CALLSIGN", help="ham mode: transmit this callsign as a station ID")
    p.add_argument("--id-mode", choices=["cw", "ascii", "both"], default="cw",
                   help="cw = Morse after frames; ascii = binary ASCII (AFSK) before the framing")
    p.add_argument("--id-interval", type=float, default=600.0, help="timer 1: ID at least this often, seconds")
    p.add_argument("--id-end-delay", type=float, default=5.0, help="timer 2: ID after this much quiet, seconds (0 = off)")
    p.add_argument("--debug-log", action="store_true", help="write debug.log (versions, settings, link events)")
    p.add_argument("--capture-wav", choices=["none", "rx", "tx", "both"], default="none",
                   help="record the audio we received / transmitted (time-aligned WAVs)")
    p.add_argument("--diag-dir", default=str(__import__("pathlib").Path.home() / "modemgui-diagnostics"),
                   help="where --debug-log / --capture-wav write their session folder")
    p = sub.add_parser("gui", help="desktop app (needs PySide6)")
    p.add_argument("--rx"); p.add_argument("--tx"); p.set_defaults(fn=cmd_gui)
    p.add_argument("--autosend", help="transmit this text once the window is up (demo)")
    p.add_argument("--hear-own", action="store_true", help="decode our own TX (null-sink self-loop)")
    p.add_argument("--settings", metavar="FILE.json", help="apply this settings file (File > Save settings format) for this run only")
    p = sub.add_parser("native"); p.set_defaults(fn=cmd_native)
    p = sub.add_parser("faxmodem", help="Hayes + Class 2 fax modem on an 86Box COM pty, line = SIP call with OFDM over G.711")
    p.set_defaults(fn=cmd_faxmodem)
    p.add_argument("--vm-path", help="86Box -P directory; the COM pty is found from its process")
    p.add_argument("--pty", help="explicit pty slave path (skips discovery)")
    p.add_argument("--com", type=int, default=1, help="COM port number (1 = lowest 86Box console pty)")
    p.add_argument("--baud", type=int, default=19200, help="DTE speed reported in CONNECT")
    p.add_argument("--class", dest="fclass", default="2", choices=["0", "2"], help="service class at power-up")
    p.add_argument("--sip-server", required=True, help="HOST:PORT of the SIP server that routes the numbers")
    p.add_argument("--no-register", action="store_true", help="do not REGISTER the user with the SIP server (default: register, refresh, "
                   "and print READY after the first 200 OK)")
    p.add_argument("--local-host", default=None, help="address to advertise in SIP/SDP (default: auto)")
    p.add_argument("--user", required=True, help="this modem's SIP user / caller id")
    p.add_argument("--name", default=None, help="name in log lines (default ne0k<user>)")
    p.add_argument("--listen-port", type=int, default=0, help="UDP port for incoming SIP INVITEs (the server's target)")
    p.add_argument("--ring-ms", type=int, default=3000, help="RING interval toward the DTE")
    p.add_argument("--band-lo", type=float, default=400.0); p.add_argument("--band-hi", type=float, default=3300.0)
    p.add_argument("--max-bits", type=int, default=6, choices=[1, 2, 4, 6], help="OFDM bits per carrier cap (6 = 64-QAM)")
    p.add_argument("--carrier-timeout", type=float, default=90.0, help="seconds to wait for the OFDM handshake after answer")
    p.add_argument("--save-dir", help="received fax pages are written here")
    p.add_argument("--find-timeout", type=float, default=60.0, help="seconds to wait for the 86Box pty")

    a = ap.parse_args(argv)
    try:
        return a.fn(a, Registry())
    except AudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
