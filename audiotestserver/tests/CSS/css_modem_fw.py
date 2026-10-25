# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
css_modem_fw.py — basic modem "firmware" over the CSS PHY + ModemNode.

This is the device-resident layer: it owns identity and persisted parameters,
drives the real css_encoder to put packets on the wire, and applies behaviour
policy on top of the transport-level ModemNode (announce, ack-a-neighbour).
Receiving is driven by the transport: whoever runs css_decoder.decode() (a live
mic loop on hardware, or the offline loopback harness in css_modem_fw_demo.py)
feeds the decoded frames back in via handle_frame().  Either way it is the SAME
css_encoder / css_decoder code — the core stays testable end-to-end.

What it provides now (per the brief):
  * self-initialisation      — boot() loads or creates a JSON config
  * stored params + id        — CONFIG_PATH holds id, seq, PHY profile, cadence
  * seq-number tracking       — restored on boot, persisted on every TX
  * periodic self-announce    — announce_due() / announce() on a cadence
  * announce + neighbour-ack   — policy: on discovering a peer, ACK it; and an
                                 originator learns its announce was heard
  * wake/preamble             — an announce is a full-preamble frame, which is
                                 exactly the training+Barker a cold neighbour
                                 needs to wake and sync (the 2-symbol wake_word
                                 only re-syncs an ALREADY-locked link).

Stretch targets it is shaped for (not built): mesh few-byte messaging with
sender tracking (already have src_id + dedup), then store-and-forward.
"""
from __future__ import annotations

import json
import os
import tempfile
import wave

import numpy as np

import css_encoder
import css_modem as cm
from css_modem_node import ModemNode


# ── persisted PHY profile ────────────────────────────────────────────────────
# The PHY parameters are FIXED in firmware and identical on every node, so the
# on-air frame geometry is constant and a receiver can decode with its own baked
# profile.  chunk_size is fixed (packets are padded to it) so every packet is
# exactly one PHY frame of constant length.
DEFAULT_PHY = {
    "fs":              48000,
    "f_min":           800,
    "f_max":           4000,
    "symbol_dur":      0.12,
    "chunk_size":      8,      # max modem packet bytes (2 hdr + up to 6 payload)
    "amplitude":       0.3,
    "symbol_gap_frac": 0.05,
    "diff_encode":     False,  # over-the-air recipe sets this True (polarity-safe)
    "rs_t":            0,
    "conv_code":       None,
}

DEFAULT_ANNOUNCE_INTERVAL_S = 5.0
DEFAULT_TURNAROUND_S        = 0.25


class ModemFirmware:
    def __init__(self, config_path: str, *, node_id: int | None = None,
                 workdir: str | None = None, verbose: bool = False,
                 phy_override: dict | None = None, bake_info: bool = True):
        self.config_path = config_path
        self.verbose = verbose
        self._boot_node_id = node_id
        # phy_override lets a caller (e.g. an over-the-air role script) inject
        # the device-native sample rate and the validated PHY recipe without
        # editing the persisted config.
        self._phy_override = dict(phy_override) if phy_override else None
        # bake_info=False skips the boot-time probe encode (canonical decode
        # info.json).  Callers that never use this node's own decode path — e.g.
        # the abstract-airtime daemon sim — set it False to boot instantly.
        self._bake_info = bake_info
        self.workdir = workdir or tempfile.mkdtemp(prefix="modemfw_")
        os.makedirs(self.workdir, exist_ok=True)

        self.cfg: dict = {}
        self.node: ModemNode | None = None
        self.info_path: str | None = None     # canonical decode profile
        self._tx_clock = 0                     # stand-in local sample counter
        self._last_announce_s: float | None = None
        # collected, application-visible events since last drain
        self.events: list[dict] = []

    # ── lifecycle ────────────────────────────────────────────────────────────
    def boot(self):
        """Load config (or create defaults) and bring up the node + PHY."""
        self.cfg = self._load_or_default()
        phy = self.cfg["phy"]
        self.node = ModemNode(
            self.cfg["id"], fs=phy["fs"],
            turnaround_samples=int(self.cfg["turnaround_s"] * phy["fs"]))
        self.node._seq = int(self.cfg.get("seq", 0))   # restore seq across boots
        if self._bake_info:
            self._bake_profile_info()
        self._log(f"BOOT id={self.cfg['id']} seq={self.node._seq} "
                  f"phy={phy['symbol_dur']}s/{phy['fs']}Hz "
                  f"announce_every={self.cfg['announce_interval_s']}s")
        return self

    def _load_or_default(self) -> dict:
        if os.path.exists(self.config_path):
            cfg = json.load(open(self.config_path))
            if self._boot_node_id is not None:
                cfg["id"] = self._boot_node_id
        elif self._boot_node_id is None:
            raise ValueError("no config on disk and no node_id given to create one")
        else:
            cfg = {
                "id": int(self._boot_node_id),
                "seq": 0,
                "announce_interval_s": DEFAULT_ANNOUNCE_INTERVAL_S,
                "turnaround_s": DEFAULT_TURNAROUND_S,
                "phy": dict(DEFAULT_PHY),
            }
            self._save(cfg)
        if self._phy_override:                     # runtime PHY (fs, recipe)
            cfg.setdefault("phy", dict(DEFAULT_PHY))
            cfg["phy"].update(self._phy_override)
        return cfg

    def _save(self, cfg: dict | None = None):
        cfg = cfg if cfg is not None else self.cfg
        if self.node is not None:
            cfg["seq"] = self.node._seq            # persist live seq
        tmp = self.config_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, self.config_path)          # atomic-ish param store

    def save(self):
        self._save()

    # ── PHY binding (real css_encoder / css_decoder) ─────────────────────────
    def _encode_cfg(self, raw: bytes, tag: str) -> dict:
        phy = self.cfg["phy"]
        base = os.path.join(self.workdir, f"tx_{tag.replace(':', '_')}")
        return dict(
            fs=phy["fs"], f_min=phy["f_min"], f_max=phy["f_max"],
            symbol_dur=phy["symbol_dur"], amplitude=phy["amplitude"],
            symbol_gap_frac=phy["symbol_gap_frac"], chunk_size=phy["chunk_size"],
            diff_encode=phy.get("diff_encode", False),
            conv_code=phy.get("conv_code", None),
            rs_t=phy.get("rs_t", 0), interleave_rows=0,
            message_bytes=bytes(raw),
            output=base + ".wav", info_out=base + "_info.json",
            result_out=base + "_enc.json",
        )

    def _bake_profile_info(self):
        """Encode a probe packet once to produce the canonical info.json this
        node will decode against — the firmware's fixed PHY 'contract'."""
        saved_seq = self.node._seq
        raw, tag = self.node.make_beacon()          # consumes a seq for the probe
        self.node._seq = saved_seq                    # undo: probe is not real TX
        cfg = self._encode_cfg(raw, "profile_probe")
        self._run_encode(cfg)
        self.info_path = cfg["info_out"]

    @staticmethod
    def _run_encode(cfg: dict) -> np.ndarray:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            css_encoder.encode(cfg)
        with wave.open(cfg["output"], "rb") as w:
            pcm = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                   .astype(np.float32) / 32768.0)
        return pcm

    def encode_packet(self, raw: bytes, tag: str) -> np.ndarray:
        """Turn a modem packet into transmit PCM via the real css_encoder."""
        return self._run_encode(self._encode_cfg(raw, tag))

    # ── transmit path ────────────────────────────────────────────────────────
    def _next_tx_sample(self, pcm_len: int = 0) -> int:
        ts = self._tx_clock
        self._tx_clock += max(1, pcm_len)
        return ts

    def announce(self) -> np.ndarray:
        """Emit a self-announce (broadcast BEACON) and return its PCM."""
        raw, tag = self.node.make_beacon()
        pcm = self.encode_packet(raw, tag)
        self.node.on_tx(tag, self._next_tx_sample(len(pcm)))
        self._save()                                  # persist advanced seq
        self._log(f"ANNOUNCE seq={raw[1] & 0xF}")
        return pcm

    def announce_due(self, now_s: float) -> bool:
        """Cadence check for periodic self-announce."""
        iv = self.cfg["announce_interval_s"]
        return (self._last_announce_s is None
                or (now_s - self._last_announce_s) >= iv)

    def announce_now(self, now_s: float) -> np.ndarray:
        self._last_announce_s = now_s
        return self.announce()

    # ── receive path (driven by the transport's decode()) ────────────────────
    def handle_frames(self, frames: list[dict]) -> list[np.ndarray]:
        """Feed decoded PHY frames (from css_decoder) to the node, run policy,
        and return any PCM buffers this node must transmit in response."""
        outgoing_pcm: list[np.ndarray] = []
        for f in frames:
            if "bytes_hex" not in f:
                continue
            raw = bytes.fromhex(f["bytes_hex"])
            at = f.get("at_sample", self._next_tx_sample())
            actions = self.node.on_rx(raw, at)
            outgoing_pcm.extend(self._apply_actions(actions))
        return outgoing_pcm

    def _apply_actions(self, actions: list[dict]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for a in actions:
            kind = a["kind"]
            if kind == "send":
                # node-originated reply (auto-ACK of DATA, or RANGE_REPLY)
                ts = self._next_tx_sample()
                raw = a["build"](ts)
                pcm = self.encode_packet(raw, a["tag"])
                self.node.on_tx(a["tag"], self._next_tx_sample(len(pcm)))
                out.append(pcm)
            elif kind == "discovered":
                # POLICY: acknowledge a newly-heard neighbour's announce so it
                # learns it was received.  This is the announce->ack behaviour.
                peer, seq = a["peer"], a.get("seq", 0)
                self.events.append({"kind": "discovered", "peer": peer})
                self._log(f"DISCOVERED peer={peer} -> ACK")
                raw, tag = self.node.make_ack(peer, seq)
                pcm = self.encode_packet(raw, tag)
                self.node.on_tx(tag, self._next_tx_sample(len(pcm)))
                out.append(pcm)
            elif kind == "ack":
                self.events.append(a)
                self._log(f"ACK from peer={a['peer']} seq={a['seq']} "
                          f"(neighbour heard my announce)")
            elif kind == "deliver":
                p = a["packet"]
                self.events.append(a)
                self._log(f"DATA from peer={p['src']} "
                          f"payload={p['payload'].hex()}")
            elif kind == "ranged":
                self.events.append(a)
                self._log(f"RANGED peer={a['peer']} dist={a['distance_m']:.2f}m")
        return out

    def drain_events(self) -> list[dict]:
        e, self.events = self.events, []
        return e

    # ── misc ─────────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        if self.verbose:
            print(f"[fw{self.cfg.get('id', '?')}] {msg}", flush=True)
