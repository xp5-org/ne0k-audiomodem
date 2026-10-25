# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
meshlib.py -- OFDM as a selectable PHY for the mesh modem.

Named WITHOUT an underscore so the rig's `<test>_<role>.py` discovery does not
register it as a bogus role (announcelib pattern).

The mesh packet layer (css_modem.py) is opaque bytes, so the SAME packets ride
either PHY: CSS chirps (control/discovery -- robust, slow) or OFDM data frames
at the bit-loading negotiated by ofdm_handshake (fast).  The PHY choice is one
decision at the transmit point: a peer with a negotiated link gets OFDM.

Mesh-OFDM data frames deliberately do NOT use the legacy Schmidl-Cox data
preamble (which never survived this room OTA).  They reuse the acquisition
that is OTA-proven end-to-end in the handshake:

    [ lead-in * N_LEAD | Barker-q-keyed known preamble * N_PRE | CE | SYNC/data ]

with a dedicated carrier-PN seed so a data frame can never be mistaken for a
handshake message (RobustLink), a channel probe, or vice versa.  The frame
body -- bit-loaded data symbols, pilots, SYNC refresh, RS -- is exactly
AdaptiveModulator's, driven at the negotiated BAT.

Packets are length-prefixed inside the frame ([2B len][packet bytes], padded),
so binary payloads with trailing zeros survive.  A mesh packet is always a
single frame (MESH_CHUNK bounds it).

Also here: `run_handshake()` -- the single mic/speaker binding for HandshakeSM.
Both the mesh (per peer, keeping the negotiated link record) and the
standalone ofdmhs test drive it.  Anything mesh-specific stays behind a
keyword (`mesh_fin` for the implicit-FIN preamble check, `abort_cb` for the
airtime kill switch) so the standalone caller pays neither cost.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np


# --- category-aware sys.path bootstrap (auto-inserted) ---
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_TESTS_ROOT = _HERE.parent if _HERE.name in ("CSS", "MESH", "OFDM", "QAM") else _HERE
_REPO_ROOT = _TESTS_ROOT.parent
for _p in (_TESTS_ROOT, _REPO_ROOT, _TESTS_ROOT / "CSS", _TESTS_ROOT / "MESH",
           _TESTS_ROOT / "OFDM", _TESTS_ROOT / "QAM", _TESTS_ROOT / "regression"):
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# --- end bootstrap ---

from ofdm_modem import (                                          # noqa: E402
    AdaptiveModulator, AdaptiveDemodulator, build_config,
    _known_bpsk, _build_ola, _grid_coherence_scan, _estimate_cpe,
    _prbs, _MOD_TABLES, _SEED_DATA,
    _RL_Q, _RL_N_PRE, _RL_N_LEAD, _COH_CARRIERS,
    bytes_to_bat, bpsk_all_bat,
    rs_encode_interleaved, rs_decode_interleaved,
)
from css_fec import (                                             # noqa: E402
    viterbi_k7, _CONV_G1, _CONV_G2, _CONV_K, _CONV_TAIL,
)
from ofdm_handshake import (                                      # noqa: E402
    HandshakeSM, HS_BAND, HS_RATES, hs_cfg, make_probe_audio, make_msg_audio,
    scale_hs_tx, probe_rms_ratio,
    try_decode_probe, try_decode_msg, _MSG_NAMES, MSG_FIN,
    MSG_BAT_REPORT, MSG_BAT_ACK, MSG_GO, MSG_BAND, recommend_band,
)
from ofdm_modem import bat_to_bytes                               # noqa: E402

_SEED_MD   = 0x4A21     # mesh-data preamble carrier PN (distinct from probe
                        # _SEED_PROBE and RobustLink _SEED_RL_PRE seeds)
_SEED_FILL = 0x39D5     # interleaver filler-bit PRBS
# Frame body size in bytes (2B length prefix + packet must fit).  Node scripts
# override it from the `mesh_chunk` param -- both ends get identical PARAMS, so
# they stay in agreement, exactly like the band and FFT geometry.
#
# 48 is a SMALL default and it costs most of the link: a frame's preamble, CE and
# SYNC symbols are fixed overhead, so at QPSK on 141 carriers (10.6 kbps gross)
# a 48-byte chunk delivers 336 bps of application goodput while a 512-byte chunk
# delivers 2253 bps -- 6.7x, for a frame 1.05s -> 1.80s longer.  Measured
# round-trips stay clean to 12 dB SNR at every size (see _selftest).
# The trade is exposure: one long frame rides out a fade or a collision less well
# than several short ones, and a mesh TX slot is 8 s by default.
MESH_CHUNK = 48
MESH_GATE  = 0.60       # acquisition gate (matches RobustLink's)
MESH_CLUSTERS = 3       # hit clusters to attempt, earliest first


# --------------------------------------------------------------------------- #
# Cross-carrier FEC (concatenated: RS outer + rate-1/2 K=7 conv inner)
#
# The OTA failure mode this defends against: the acoustic channel is
# TIME-VARYING, so carriers that measured strong at probe time can be in deep
# fades at data time.  Hard-decision demap turned each faded carrier into a
# contiguous byte burst that overwhelmed RS.  Here every coded bit gets a
# SOFT confidence computed in Y-space (|Y - H*point|^2), so a faded carrier's
# bits arrive as near-zero-confidence punctures that the Viterbi absorbs, and
# the interleaver scatters them through the coded stream.
# --------------------------------------------------------------------------- #

def conv_encode_k7(bits) -> np.ndarray:
    """Rate-1/2 K=7 (NASA 0o171/0o133) encoder matching css_fec.viterbi_k7,
    including the 6 zero tail bits.  Returns the coded bit array (2*(n+6))."""
    state = 0
    out = np.empty(2 * (len(bits) + _CONV_TAIL), dtype=np.uint8)
    j = 0
    for b in list(map(int, bits)) + [0] * _CONV_TAIL:
        reg = (b << (_CONV_K - 1)) | state
        out[j]     = bin(reg & _CONV_G1).count("1") & 1
        out[j + 1] = bin(reg & _CONV_G2).count("1") & 1
        state = ((b << (_CONV_K - 2)) | (state >> 1)) & 0x3F
        j += 2
    return out


def _interleave_map(n_total: int) -> np.ndarray:
    """Permutation pi: coded-bit index k -> transmit slot pi[k].

    Multiplicative golden-ratio stride, coprime to n_total: consecutive coded
    bits land far apart in BOTH symbol index and carrier position, and -- the
    part a row/column block interleaver gets wrong -- a dead carrier's slots
    map back to an evenly-strided (not contiguous) set of coded bits.
    """
    p = max(1, round(n_total * 0.6180339887))
    while math.gcd(p, n_total) != 1:
        p += 1
    return (np.arange(n_total, dtype=np.int64) * p) % n_total


# per-modulation bit-value point masks for max-log LLR: _LLR_MASKS[b][j] =
# (points_with_bit_j_0, points_with_bit_j_1)
_LLR_MASKS: dict[int, list] = {}
for _b, (_pts, _bts) in _MOD_TABLES.items():
    _LLR_MASKS[_b] = [(np.where(_bts[:, _j] == 0)[0], np.where(_bts[:, _j] == 1)[0])
                      for _j in range(_b)]


def hs_band_effective() -> dict:
    """The handshake convention AS OVERRIDDEN for this run.

    HS_OVERRIDES is read live rather than captured at import: node scripts set
    it after importing this module, and a stale copy silently splits the run in
    two geometries (see mesh_cfg).
    """
    import ofdm_handshake as _hs
    return {**HS_BAND, **(_hs.HS_OVERRIDES or {})}


def mesh_cfg(sample_rate: int) -> dict:
    """OFDM config for mesh data frames: handshake band/geometry, small chunk.

    MUST derive from the same effective band as ofdm_handshake.hs_cfg. Using
    the raw HS_BAND constant instead of the effective one desyncs the two
    geometries whenever the band or FFT geometry (hs_freq_lo/hs_freq_hi, n_fft,
    cp_ratio) is overridden: the negotiated table is for one carrier set but
    transmission uses another, and bytes_to_bat's silent truncation unpacks the
    bit-allocation onto the wrong carriers -- a direct cause of high data-frame
    EVM that both ends can agree on without either side detecting it.
    """
    return build_config(sample_rate=int(sample_rate),
                        chunk_bytes=MESH_CHUNK, ecc_bytes=16, seq_bytes=2,
                        **hs_band_effective())


class _MeshFecGeometry:
    """Deterministic FEC frame geometry shared by modulator and demodulator.

    info bits  = 8 * (seq_bytes + chunk_bytes + ecc_bytes)   (RS-coded block)
    coded bits = 2 * (info + 6 tail)
    C          = bits per OFDM symbol (from the BAT)
    R          = data symbols = ceil(coded / C); slots = R*C, PRBS filler
    """

    def _fec_geometry(self):
        info_bits  = 8 * self.payload_bytes
        coded_len  = 2 * (info_bits + _CONV_TAIL)
        C = self.bits_per_ofdm_sym
        R = math.ceil(coded_len / C)
        n_total = R * C
        return info_bits, coded_len, C, R, n_total


class MeshModulator(AdaptiveModulator, _MeshFecGeometry):
    """AdaptiveModulator body with the q-keyed acquisition preamble and
    concatenated cross-carrier FEC (RS outer + soft-Viterbi inner)."""

    def __init__(self, cfg: dict, bits_per_bin: np.ndarray) -> None:
        super().__init__(cfg, bits_per_bin)
        self.md_known = _known_bpsk(self.n_occupied, _SEED_MD)

    def encode_packet(self, packet: bytes) -> np.ndarray:
        """One mesh packet -> one OFDM frame (audio float array)."""
        if int(self.bits_per_ofdm_sym) <= 0:
            raise ValueError("cannot encode on an all-off BAT (0 carriers) -- "
                             "this direction of the link has no capacity")
        packet = bytes(packet)
        framed = len(packet).to_bytes(2, 'big') + packet
        if len(framed) > self.chunk_bytes:
            raise ValueError(
                f"packet {len(packet)}B exceeds this frame's capacity "
                f"({self.chunk_bytes - 2}B = mesh_chunk {self.chunk_bytes} "
                f"minus the 2B length prefix). Raise the `mesh_chunk` param "
                f"(both nodes get the same PARAMS, so they stay in agreement) "
                f"or split the payload across packets -- there is no "
                f"fragmentation layer, one packet is always one frame.")
        framed += b'\x00' * (self.chunk_bytes - len(framed))

        # RS -> scramble -> conv -> interleave
        # Interleaved RS, same geometry both ends inherit from
        # AdaptiveModulator.  Mesh gets the burst protection for free here: its
        # own bit-interleaver spreads CONVOLUTIONAL symbols across carriers,
        # which helps frequency-selective fades, but the RS bytes underneath
        # were still contiguous, so a lost OFDM symbol hit one codeword.
        enc  = rs_encode_interleaved(
            self.rsc, (0).to_bytes(self.seq_bytes, 'big') + framed,
            self.il_depth, self.il_k, self.ecc_bytes)
        info = np.unpackbits(np.frombuffer(enc, dtype=np.uint8))
        info = info ^ _prbs(len(info), _SEED_DATA)
        coded = conv_encode_k7(info)
        _, coded_len, C, R, n_total = self._fec_geometry()
        stream = np.concatenate([coded,
                                 _prbs(n_total - coded_len, _SEED_FILL)])
        slots = np.empty(n_total, dtype=np.uint8)
        slots[_interleave_map(n_total)] = stream

        occ  = [self.md_known.copy() for _ in range(_RL_N_LEAD)]
        occ += [_RL_Q[j] * self.md_known for j in range(_RL_N_PRE)]
        occ += [self.ce_known.copy()]
        for ds in range(R):
            if self.sync_period > 0 and ds % self.sync_period == 0:
                occ.append(self.sync_known.copy())
            occ.append(self._data_occ(slots[ds * C:(ds + 1) * C]))
        self.total_frames = 1
        return _build_ola(occ, self.n_fft, self.bin_lo, self.cp_len,
                          self.taper_samples)


class MeshDemodulator(AdaptiveDemodulator, _MeshFecGeometry):
    """AdaptiveDemodulator acquiring on the q-keyed mesh preamble, with
    soft-decision (max-log LLR) demapping feeding the Viterbi."""

    def __init__(self, cfg: dict, bits_per_bin: np.ndarray) -> None:
        super().__init__(cfg, bits_per_bin)
        self.md_known = _known_bpsk(self.n_occupied, _SEED_MD)
        # Receive-side only and self-checking (the refinement is kept only when
        # it lowers EVM), so it defaults ON: there is nothing for the peer to
        # agree to and a bad refinement is discarded rather than transmitted.
        self.data_aided = bool(cfg.get('data_aided', True))
        # Optional constellation tap.  Off during live decoding (it allocates
        # per symbol); the report's post-hoc analyser turns it on to plot what
        # the demapper actually saw.  Filled with (eq, ref, bits) triples for
        # the pass whose result was KEPT, so it matches the reported EVM.
        self.collect_points = False
        self.points: list = []
        # Points from the acquisition attempt in progress.  decode_packet may
        # try several candidate sync positions before one takes; only the
        # attempt it ends up REPORTING may be published into self.points (see
        # _commit_points), or the plot pools a good frame with garbage
        # demapped at a wrong offset.
        self._pending_points: list = []

    def _loaded_carriers(self) -> list:
        """(occupied index, bits) for every carrier the BAT actually loads."""
        out, di = [], 0
        for i in range(self.n_occupied):
            if i in self._pilot_set:
                continue
            b = int(self.bits_per_bin[di]); di += 1
            if b:
                out.append((i, b))
        return out

    def _demap(self, fds: list, H_per_sym: list, n_total: int):
        """Soft-demap collected data symbols against a given channel estimate.

        Split out of the acquisition loop so the SAME demapping can be run
        twice: once on the one-shot CE/SYNC estimate and once on an estimate
        refined from the first pass's own decisions (see _soft_decode_body).

        Returns (conf, decisions, evm_num, evm_den, bits_written).

        When `collect_points` is set, also returns the equalised constellation
        points via `pts_out` -- Y/He, i.e. what the demapper compared against the
        reference alphabet, which is the only view that shows HOW a frame failed
        (rotation, radial smear, one bad quadrant) rather than just how much.
        """
        loaded = self._loaded_carriers()
        conf = np.empty(n_total)
        dec = []
        pts_out: list = []
        evm_n = evm_d = 0.0
        w = 0
        for fd, H in zip(fds, H_per_sym):
            cpe = _estimate_cpe(fd * math.sqrt(self.n_fft), self.n_fft,
                                self.bin_lo, self.pilot_offsets, H,
                                self.PILOT_SYM)
            # SIGN: residual r = Y/(H*pilot) has angle +cpe, so correct removal
            # needs H_eff = H*e^{+j*cpe} (eq = Y/H_eff). Using e^{-j*cpe} doubles
            # the phase error instead of cancelling it. Only manifests with a
            # nonzero sample-clock offset; a zero-phase channel has cpe=0.
            corr = math.cos(cpe) + 1j * math.sin(cpe)
            row = np.empty(len(loaded), dtype=complex)
            for c, (i, b) in enumerate(loaded):
                Y = fd[self.bin_lo + i]
                He = H[i] * corr
                pts, _ = _MOD_TABLES[b]
                d = np.abs(Y - He * pts) ** 2
                k = int(np.argmin(d))
                row[c] = pts[k]
                evm_n += float(d[k]); evm_d += float(np.abs(He * pts[k]) ** 2)
                if self.collect_points and abs(He) > 1e-12:
                    # Y/He puts every carrier on the SAME reference alphabet
                    # regardless of its channel gain, so points from carriers
                    # 30 dB apart can share one plot.  |He| rides along because
                    # that normalisation also divides the noise up: a carrier in
                    # a deep null lands arbitrarily far from its reference, so
                    # any EVM over these points has to be able to weight them
                    # back down (see constellation._order_evm).
                    pts_out.append((Y / He, pts[k], b, abs(He)))
                if w + b <= n_total:
                    for j in range(b):
                        m0, m1 = _LLR_MASKS[b][j]
                        conf[w] = float(d[m0].min() - d[m1].min())
                        w += 1
                else:
                    w += b
            dec.append(row)
        return conf, dec, evm_n, evm_d, w, pts_out

    def _refine_H(self, fds: list, H_per_sym: list, dec: list):
        """Least-squares per-carrier H from the whole frame's own decisions.

        The CE/SYNC estimate comes from ONE symbol, so its noise lands on every
        data symbol that follows it -- and with only one SYNC anchor inside a
        short mesh frame there is nothing to interpolate between.  Averaging
        Y*conj(s) over all R data symbols instead cuts that estimation noise by
        ~sqrt(R).  Measured on an OTA capture: EVM 63% -> 46%, closing most of
        the gap between the as-decoded SINR (4.0 dB) and the channel's actual
        ISI ceiling (~7 dB).

        Decisions from a 60%-EVM first pass are not all correct, so this is
        only ever ACCEPTED if it lowers the measured EVM (the caller checks).
        """
        loaded = self._loaded_carriers()
        if not loaded or not dec:
            return None
        num = np.zeros(len(loaded), dtype=complex)
        den = np.zeros(len(loaded))
        for fd, H, row in zip(fds, H_per_sym, dec):
            cpe = _estimate_cpe(fd * math.sqrt(self.n_fft), self.n_fft,
                                self.bin_lo, self.pilot_offsets, H,
                                self.PILOT_SYM)
            # SIGN: residual r = Y/(H*pilot) has angle +cpe, so correct removal
            # needs H_eff = H*e^{+j*cpe} (eq = Y/H_eff). Using e^{-j*cpe} doubles
            # the phase error instead of cancelling it. Only manifests with a
            # nonzero sample-clock offset; a zero-phase channel has cpe=0.
            corr = math.cos(cpe) + 1j * math.sin(cpe)
            for c, (i, _b) in enumerate(loaded):
                # Y is de-rotated by the symbol's own CPE, so the refined H is
                # in the same (CPE-free) frame the demapper re-applies it in.
                num[c] += (fd[self.bin_lo + i] / corr) * np.conj(row[c])
                den[c] += float(np.abs(row[c]) ** 2)
        H_new = np.array(H_per_sym[0], dtype=complex)
        for c, (i, _b) in enumerate(loaded):
            if den[c] > 0:
                H_new[i] = num[c] / den[c]
        return [H_new] * len(fds)

    def _soft_decode_body(self, rx: np.ndarray, body_start: int):
        """Demodulate the frame body (CE at `body_start`) to a packet.

        Per-bit confidences are computed in Y-space: LLR_j =
        min_{s: bit_j=0}|Y - H*s|^2 - min_{s: bit_j=1}|Y - H*s|^2, so a faded
        carrier (tiny |H|) yields near-zero confidence for its bits instead of
        confidently-wrong hard decisions.  Returns (packet, stats) or
        (None, evm_stats).

        Two passes when `data_aided` is on: the second re-estimates H from the
        first pass's decisions across the whole frame and is kept only if the
        EVM improves.  Receive-side only -- nothing about it is negotiated, so
        one end can run it without the other knowing.
        """
        N, CP, SYM = self.n_fft, self.cp_len, self.symbol_samples
        info_bits, coded_len, C, R, n_total = self._fec_geometry()
        n_syncs = (math.ceil(R / self.sync_period)
                   if self.sync_period > 0 else 0)
        total_body = 1 + R + n_syncs                     # CE + data + SYNCs
        need = body_start + SYM * total_body + CP + N
        if need > len(rx):
            # The frame is still ARRIVING -- the preamble is here but the body
            # is not.  This is NOT a decode failure; treating it as RS failure
            # discards the live-captured prefix and loses the rest of the frame
            # while it is still transmitting. A frame longer than the poll
            # interval (e.g. a 4.8s frame against a 4s poll) hits this routinely.
            return None, {"incomplete": True,
                          "have_s": round(len(rx) / max(self.sample_rate, 1), 2),
                          "need_s": round(need / max(self.sample_rate, 1), 2)}

        pos = body_start
        ce  = np.fft.rfft(rx[pos + CP:pos + CP + N], n=N) / math.sqrt(N)
        H   = ce[self.bin_lo:self.bin_lo + self.n_occupied] / self.ce_known
        pos += SYM

        # -- collect the data symbols and the estimate in force at each --
        fds: list = []
        H_per_sym: list = []
        body = 0
        for _ in range(R + n_syncs):
            fd = np.fft.rfft(rx[pos + CP:pos + CP + N], n=N) / math.sqrt(N)
            pos += SYM
            blk = self.sync_period + 1
            if self.sync_period > 0 and body % blk == 0:
                H = (fd[self.bin_lo:self.bin_lo + self.n_occupied]
                     / self.sync_known)
                body += 1
                continue
            body += 1
            fds.append(fd)
            H_per_sym.append(H)

        conf, dec, evm_n, evm_d, w, pts1 = self._demap(fds, H_per_sym, n_total)
        evm_pct = (100 * math.sqrt(evm_n / max(evm_d, 1e-12))
                   if evm_d > 0 else None)
        evm_1st = evm_pct
        refined = False
        if self.data_aided and w >= n_total and len(fds) >= 3:
            H2 = self._refine_H(fds, H_per_sym, dec)
            if H2 is not None:
                c2, d2, n2, d2d, w2, pts2 = self._demap(fds, H2, n_total)
                evm2 = (100 * math.sqrt(n2 / max(d2d, 1e-12))
                        if d2d > 0 else None)
                # Strictly better only: a wrong-decision-biased refinement
                # makes EVM worse, and that is exactly when to discard it.
                if evm2 is not None and evm_pct is not None and evm2 < evm_pct:
                    conf, evm_pct, w, refined = c2, evm2, w2, True
                    pts1 = pts2          # keep the pass we actually used

        if self.collect_points:
            self._pending_points = list(pts1)
        if w < n_total:
            return None, None
        # normalise to [-1, 1]; robust scale so a few huge LLRs don't crush
        # the rest
        scale = float(np.percentile(np.abs(conf), 70)) + 1e-12
        conf = np.clip(conf / scale, -1.0, 1.0)

        # de-interleave, Viterbi, descramble, RS
        coded_conf = conf[_interleave_map(n_total)][:coded_len]
        pairs = list(zip(coded_conf[0::2], coded_conf[1::2]))
        bits = np.array(viterbi_k7(pairs), dtype=np.uint8)
        bits = bits ^ _prbs(len(bits), _SEED_DATA)
        raw = np.packbits(bits).tobytes()[:self.payload_bytes]
        stats = {"evm_pct": (None if evm_pct is None else round(evm_pct, 1)),
                 "mean_conf": round(float(np.mean(np.abs(conf))), 3)}
        if refined:
            # Keep the one-shot figure too: the two together say how much of
            # the error was estimation noise rather than channel.
            stats["evm_pct_1pass"] = round(evm_1st, 1)
            stats["data_aided"] = True
        try:
            decoded, n_errata = rs_decode_interleaved(
                self.rsc, raw, self.il_depth,
                self.seq_bytes + self.chunk_bytes)
        except Exception:
            return None, stats
        chunk = decoded[self.seq_bytes:]
        stats["rs_errata"] = n_errata
        return chunk, stats

    def _commit_points(self, staged: list | None = None) -> None:
        """Publish one attempt's constellation points (see _pending_points)."""
        if self.collect_points:
            self.points.extend(self._pending_points if staged is None
                               else staged)
        self._pending_points = []

    def decode_packet(self, samples: np.ndarray):
        """Find and decode one mesh frame.  Returns (packet_bytes, stats) or
        (None, None)."""
        # An all-off BAT (0 usable carriers) carries no data symbols, so the
        # FEC geometry has zero bits per symbol -- nothing to decode, and
        # _fec_geometry would divide by C=0.  Never bring such a "link" this
        # far, but guard the shared decoder regardless.
        if int(self.bits_per_ofdm_sym) <= 0:
            return None, None
        N, CP, SYM = self.n_fft, self.cp_len, self.symbol_samples
        rx = np.asarray(samples, dtype=np.float64)
        span = (_RL_N_PRE - 1) * SYM + CP + N
        if len(rx) < span + SYM:
            return None, None
        step = math.gcd(SYM, max(8, CP // 4))
        cands, coh = _grid_coherence_scan(
            rx, N, CP, self.bin_lo, self.n_occupied, self.md_known,
            _RL_N_PRE, step, q=_RL_Q, top_k=_COH_CARRIERS)
        if not len(cands):
            return None, None
        hit_idx = np.where(coh >= MESH_GATE)[0]

        tried = 0
        pos_i = 0
        last_fail = None
        last_fail_points: list = []
        while tried < MESH_CLUSTERS and pos_i < len(hit_idx):
            h0 = int(hit_idx[pos_i])
            h1 = min(len(cands), h0 + max(1, (2 * SYM) // step))
            best = int(cands[h0 + int(np.argmax(coh[h0:h1]))])
            # fine align on coherent energy of the preamble block
            fine, fscore = [], []
            for s in range(max(0, best - step), best + step, 4):
                Hb = self._pre_H(rx, s)
                fine.append(s)
                fscore.append(0.0 if Hb is None else
                              float(np.sum(np.abs((Hb * _RL_Q[:, None])
                                                  .mean(axis=0)) ** 2)))
            best = int(fine[int(np.argmax(fscore))])

            # frame body (CE first) begins right after the preamble block
            body_start = best + _RL_N_PRE * SYM
            chunk, st = self._soft_decode_body(rx, body_start)
            if chunk is not None:
                plen = int.from_bytes(chunk[:2], 'big')
                if 0 < plen <= len(chunk) - 2:
                    self._commit_points()
                    return chunk[2:2 + plen], st
            if st and st.get("incomplete"):
                # Report it and stop: later candidates start LATER, so they are
                # even less complete.  The caller must keep the capture and try
                # again next poll rather than count this as a failure.  Not a
                # decoded frame, so its half-demapped points are dropped.
                self._pending_points = []
                return None, {"acq_locked": True,
                              "acq_coh": round(float(coh[h0]), 3),
                              "at_sample": int(best), **st}
            tried += 1
            # Where THIS frame ends, so the caller can skip past just the failed
            # frame rather than everything captured so far -- discarding to the
            # live edge also throws away any frame that began during the poll.
            _, _, _, _R, _ = self._fec_geometry()
            _nsync = (math.ceil(_R / self.sync_period)
                      if self.sync_period > 0 else 0)
            frame_end = body_start + SYM * (1 + _R + _nsync) + CP + N
            last_fail = {"acq_locked": True, "acq_coh": round(float(coh[h0]), 3),
                         "at_sample": int(best), "rs_failed": True,
                         "frame_end": int(frame_end),
                         "rate": int(self.sample_rate),
                         **(st or {})}
            # A failed frame is still worth plotting -- it is the one worth
            # plotting -- but only the attempt actually reported below.
            last_fail_points = self._pending_points
            self._pending_points = []
            nxt = cands[h0] + 2 * SYM
            while pos_i < len(hit_idx) and cands[hit_idx[pos_i]] < nxt:
                pos_i += 1
        # no packet -- report whether we at least locked a preamble, so the
        # caller can tell "nothing there" from "frame present but demod failed"
        if tried and last_fail:
            self._commit_points(last_fail_points)
        self._pending_points = []
        return None, (last_fail if tried else None)

    def _pre_H(self, rx: np.ndarray, start: int):
        N, CP, SYM = self.n_fft, self.cp_len, self.symbol_samples
        if start < 0 or start + (_RL_N_PRE - 1) * SYM + CP + N > len(rx):
            return None
        idx = (start + CP + np.arange(_RL_N_PRE)[:, None] * SYM
               + np.arange(N)[None, :])
        fd = np.fft.rfft(rx[idx], n=N, axis=1) / math.sqrt(N)
        return fd[:, self.bin_lo:self.bin_lo + self.n_occupied] / self.md_known


# --------------------------------------------------------------------------- #
# Link records + PHY-selection helpers
# --------------------------------------------------------------------------- #

def make_link(hs_result: dict, my_rate: int, *, symmetric: bool = True) -> dict:
    """Per-peer OFDM link record from a completed handshake.

    symmetric=True (default): the acoustic link is EXPECTED to be asymmetric
    (different speaker/mic sensitivities each way), which strands the return
    ACK when its direction is the weak one -- data got through but the ACK
    can't come back.  So collapse both directions onto the WEAKER one:
    per-carrier bits = min(tx_bat, rx_bat), used for TX and RX alike.  This
    is provably consistent WITHOUT extra signalling: both ends hold the SAME
    pair of directional BATs (my tx == peer's rx == the A->B allocation; my
    rx == peer's tx == the B->A allocation), so min() yields the identical
    table on each node.  Requires both directions at one rate/geometry (true
    on this rig -- both TX at out_fs); falls back to independent BATs if not.
    """
    my_cfg = mesh_cfg(my_rate)
    tx_bat = bytes_to_bat(hs_result["my_tx_bat"], my_cfg["n_data_carriers"],
                          what="tx BAT")
    peer_rate = int(hs_result["peer_rate"])
    peer_cfg = mesh_cfg(peer_rate)
    rx_bat = bytes_to_bat(hs_result["peer_tx_bat"],
                          peer_cfg["n_data_carriers"], what="rx BAT")
    if symmetric and my_rate == peer_rate and len(tx_bat) == len(rx_bat):
        sym = np.minimum(tx_bat, rx_bat).astype(tx_bat.dtype)
        tx_bat = rx_bat = sym
    return {
        "tx_rate":  my_rate,   "tx_bat": tx_bat,   # what I transmit with
        "rx_rate":  peer_rate, "rx_bat": rx_bat,   # what the peer transmits with
        "symmetric": bool(symmetric and my_rate == peer_rate
                          and tx_bat is rx_bat),
        "tx_gross_bps": int(tx_bat.sum()) * my_rate // my_cfg["symbol_samples"],
        "rx_gross_bps": int(rx_bat.sum()) * peer_rate // peer_cfg["symbol_samples"],
    }


def _probe_rms_ratio(rate: int) -> float:
    """rms/peak of the handshake probe waveform at `rate`.

    Delegates to ofdm_handshake.probe_rms_ratio so the DATA path and the
    HANDSHAKE path share one definition of the reference level -- they were
    independent, which is how the handshake kept peak-normalising for so long
    after the data path was fixed."""
    return probe_rms_ratio(rate)


def encode_packet_ofdm(link: dict, packet: bytes) -> np.ndarray:
    return MeshModulator(mesh_cfg(link["tx_rate"]),
                         link["tx_bat"]).encode_packet(packet)


def ofdm_tx_pcm(link: dict, packet: bytes, amplitude: float) -> np.ndarray:
    """Encode a packet and scale it for transmission, RMS-MATCHED to the probe.

    The negotiated BAT is only valid at the power spectral density the probe
    was measured at.  Peak-normalising both to `amplitude` under-drives the
    data frame by its PAPR excess (measured OTA: -4.7 dB average power vs the
    probe), which erased the 6 dB allocation margin -- every data symbol came
    out at ~100% EVM while the frame's preamble acquired at 0.876 coherence.
    So: scale the data frame so its RMS equals what the probe's RMS would be
    at this amplitude setting; clip-guard the peak.
    """
    sig = encode_packet_ofdm(link, packet)
    target_rms = amplitude * _probe_rms_ratio(link["tx_rate"])
    scale = target_rms / max(float(np.sqrt(np.mean(sig ** 2))), 1e-12)
    peak = float(np.max(np.abs(sig))) * scale
    if peak > 0.9:                       # never clip; accept the small deficit
        scale *= 0.9 / peak
    return (sig * scale).astype(np.float32)


def try_decode_packet_ofdm(link: dict, capture: np.ndarray, mic_fs: int):
    """Decode a mesh-OFDM frame from a mic capture (resamples to the peer's
    TX rate).  Returns (packet_bytes, stats) or (None, None)."""
    from ofdm_handshake import _resample
    cap = _resample(capture, mic_fs, link["rx_rate"])
    dem = MeshDemodulator(mesh_cfg(link["rx_rate"]), link["rx_bat"])
    return dem.decode_packet(cap)


# --------------------------------------------------------------------------- #
# Passive OFDM-activity detection (fallback re-entry sensing)
# --------------------------------------------------------------------------- #

def detect_ofdm_activity(capture: np.ndarray, mic_fs: int, *,
                         thr: float = 0.5, amb_rms: float | None = None,
                         rates=HS_RATES, max_s: float = 6.0):
    """Cheap "is anyone using OFDM nearby?" detector for duty-cycled polling.

    NOT a decoder: only the acquisition coherence metric is evaluated, at a
    coarse grid (SYM/4), so a hit means "OFDM-shaped signal on the air", not
    "frame for me".  Deliberately lossy -- the caller polls every few seconds
    and can afford to miss frames (re-entry budget is 10-15 s).

    Two scans per candidate rate share the geometry:
      * q=_RL_Q  (Barker-keyed): mesh data frames + RobustLink handshake
        messages.  The carrier-PN seed divides out of the power-normalised
        coherence (a wrong seed is a per-carrier constant), so ONE scan covers
        every q-keyed preamble regardless of seed; the sign key rejects tones
        and repeated-symbol mimics.
      * q=None (repeated symbols): handshake probes (~150 identical symbols)
        and the _RL_N_LEAD plain lead-in.  Tones are rejected by the top_k
        carrier averaging (a tone lights one carrier, the other 39 score
        noise).

    `amb_rms` (broadband ambient from carrier-sense calibration) enables the
    stage-1 energy gate: skip the scans entirely unless the last 0.5 s carries
    in-band energy well above ambient -- the steady-state cost is then one
    small rfft per poll.

    Returns (hit: bool, best_coherence: float).
    """
    x = np.asarray(capture, dtype=np.float64)
    if len(x) < int(1.5 * mic_fs):
        return False, 0.0
    if len(x) > int(max_s * mic_fs):
        x = x[-int(max_s * mic_fs):]

    if amb_rms is not None and amb_rms > 0:
        tail = x[-int(0.5 * mic_fs):]
        F = np.fft.rfftfreq(len(tail), 1.0 / mic_fs)
        S = np.abs(np.fft.rfft(tail)) ** 2
        _hb = hs_band_effective()
        sel = (F >= _hb["freq_lo"]) & (F <= _hb["freq_hi"])
        # Parseval: rms of the band-limited part of the tail
        band_rms = math.sqrt(2.0 * float(S[sel].sum()) / (len(tail) ** 2))
        if band_rms < 3.0 * amb_rms:
            return False, 0.0

    # stage 1.5: narrowband rejection.  A tone or CSS chirp sweep puts the
    # SAME phase progression on every spectral-leakage bin, and the power-
    # normalised coherence metric cannot tell tiny-but-coherent leakage from
    # a genuinely flat preamble (a slow sweep scored 0.505 against the Barker
    # key in selftest).  Structurally: within one ~43 ms window a sweep moves
    # < 1 bin, so its in-band energy sits in a few bins, while an OFDM symbol
    # fills ~77 carriers.  Reject when the energetic windows are dominated by
    # their top 3 bins.
    N1 = 2048
    nwin = len(x) // N1
    fr = x[:nwin * N1].reshape(nwin, N1)
    S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    F1 = np.fft.rfftfreq(N1, 1.0 / mic_fs)
    _hb = hs_band_effective()
    S = S[:, (F1 >= _hb["freq_lo"]) & (F1 <= _hb["freq_hi"])]
    e = S.sum(axis=1)
    strong = e > 0.25 * float(e.max() + 1e-30)
    if np.any(strong):
        top3 = np.sort(S[strong], axis=1)[:, -3:].sum(axis=1)
        if float(np.median(top3 / (e[strong] + 1e-30))) > 0.55:
            return False, 0.0

    best = 0.0
    for rate in rates:
        from ofdm_handshake import _resample
        cap = _resample(x, mic_fs, rate)
        cfg = mesh_cfg(rate)
        N, CP = cfg["n_fft"], cfg["cp_len"]
        SYM = N + CP
        step = SYM // 4
        while SYM % step:
            step -= 1
        known = _known_bpsk(cfg["n_occupied"], _SEED_MD)
        # q=None runs over 2x the symbols: a slow swept tone is quasi-
        # stationary across 8 symbols (scored 0.51 in selftest) but decoheres
        # over 16; probes (~150+ syms) and the 22-sym lead-in still cover it.
        for q, nsyms in ((_RL_Q, _RL_N_PRE), (None, 2 * _RL_N_PRE)):
            _, coh = _grid_coherence_scan(
                cap, N, CP, cfg["bin_lo"], cfg["n_occupied"], known,
                nsyms, step, q=q, top_k=_COH_CARRIERS)
            if len(coh):
                best = max(best, float(coh.max()))
        if best >= thr:
            break
    return best >= thr, round(best, 3)


# --------------------------------------------------------------------------- #
# Handshake binding (mic/speaker loop around HandshakeSM)
# --------------------------------------------------------------------------- #

def _tail_busy(cap: np.ndarray, fs: int, ambient: float,
               tail_s: float = 1.0) -> bool:
    """Is the last `tail_s` of the listen window loud vs the ambient floor?
    Mirrors ofdmhs_node's carrier-sense gate (peer TX still in progress)."""
    n = int(tail_s * fs)
    if len(cap) < n:
        return False
    rms = float(np.sqrt(np.mean(cap[-n:] ** 2)))
    return rms > 4.0 * max(ambient, 1e-7)


def _mesh_preamble_coherence(capture: np.ndarray, mic_fs: int,
                             tx_rate: int) -> float:
    """Best q-keyed OFDM preamble coherence in `capture` at the peer's rate.

    Used by the responder during r_linger as an implicit FIN: once the
    initiator is transmitting mesh data frames it has necessarily finished
    the handshake, so lingering deaf serves nothing.  (The carrier-PN seed
    divides out of the power-normalised coherence, so this also fires on a
    RobustLink message whose payload did not decode -- acceptable: linger
    expiry reaches the same terminal state anyway, and the msg decode is
    always attempted first in the same poll.)
    """
    from ofdm_handshake import _resample
    x = _resample(np.asarray(capture, np.float64), mic_fs, tx_rate)
    cfg = mesh_cfg(tx_rate)
    N, CP = cfg["n_fft"], cfg["cp_len"]
    SYM = N + CP
    span = (_RL_N_PRE - 1) * SYM + CP + N
    if len(x) < span + SYM:
        return 0.0
    step = math.gcd(SYM, max(8, CP // 4))
    known = _known_bpsk(cfg["n_occupied"], _SEED_MD)
    _, coh = _grid_coherence_scan(x, N, CP, cfg["bin_lo"], cfg["n_occupied"],
                                  known, _RL_N_PRE, step, q=_RL_Q,
                                  top_k=_COH_CARRIERS)
    return float(coh.max()) if len(coh) else 0.0

def hs_airtime_budget(spk_fs: int, *, turn_overhead_s: float = 4.0) -> dict:
    """Exact airtime of one clean handshake transaction at *spk_fs* under the
    ACTIVE HS_OVERRIDES (so the chosen mode/allocation knobs are reflected):
    encodes the real probe and RobustLink message segments and measures them,
    rather than assuming durations.

    A clean transaction is 5 half-duplex turns:
        I:probe  R:probe+BAT_REPORT  I:BAT_ACK  R:GO  I:FIN
    `turn_overhead_s` covers per-turn decode-poll latency + processing.

    Returns segment durations plus:
        clean_s   one full transaction incl. turn overheads -- sizes the
                  window-entry guard (don't start what can't finish)
        max_tx_s  longest single segment -- the clamp margin a handshake
                  timeout must leave before a window boundary so an
                  already-playing burst still ends inside the window
    """
    import ofdm_handshake as _hs
    cfg = hs_cfg(spk_fs)
    bat = bat_to_bytes(np.zeros(cfg["n_data_carriers"]))
    if getattr(_hs, "HS_BAT_RLE", False):
        # BAT bodies ship RLE-compressed; budget with a conservative-typical
        # compressed size (24B covers every BAT observed OTA except the
        # fully-fragmented raw-fallback case) instead of the raw table.
        bat = b"\x00" * min(24, len(bat))
    dur = lambda a: len(a) / spk_fs
    # n_groups=1 forces the SLOWEST ladder step for every message.  Real sends
    # ride HS_RL_BOOTSTRAP_STEP or better, but fast_groups_for still drops to
    # G=1 on a table too sparse to split -- exactly the marginal link least able
    # to afford being aborted for overrun -- so the budget has to cover it.
    probe_s  = dur(make_probe_audio(spk_fs))
    report_s = dur(make_msg_audio(spk_fs, MSG_BAT_REPORT, 0, bat, n_groups=1))
    ack_s    = dur(make_msg_audio(spk_fs, MSG_BAT_ACK, 0, bat, n_groups=1))
    go_s     = dur(make_msg_audio(spk_fs, MSG_GO, 0, b"", n_groups=1))
    fin_s    = dur(make_msg_audio(spk_fs, MSG_FIN, 0, b"", n_groups=1))
    clean_s  = 2 * probe_s + report_s + ack_s + go_s + fin_s \
        + 5 * turn_overhead_s
    return {"probe_s": round(probe_s, 2), "report_s": round(report_s, 2),
            "ack_s": round(ack_s, 2), "go_s": round(go_s, 2),
            "fin_s": round(fin_s, 2),
            "max_tx_s": round(max(probe_s, report_s, ack_s, go_s, fin_s), 2),
            "clean_s": round(clean_s, 1)}


def _air_busy(rec, mic_fs: int, ambient: float, since_frame: int,
              tail_s: float = 1.0) -> bool:
    """Carrier sense: is the peer transmitting RIGHT NOW?

    Looks only at the newest `tail_s` of audio, floored at `since_frame` so we
    never sense our own transmission's tail.  Deliberately not _tail_busy on
    the decode window: that returns False whenever the window is shorter than
    the tail, so for a second after every listen_start reset (our own TX
    ending, or a message decode) the node was blind and could key up over
    whatever was on the air.
    """
    n_tail = int(tail_s * mic_fs)
    total  = rec.frames_captured
    tail   = rec.data()[max(int(since_frame), total - n_tail):]
    if len(tail) < int(0.3 * mic_fs):        # not enough fresh audio to judge
        return False
    return float(np.sqrt(np.mean(tail ** 2))) > 4.0 * max(ambient, 1e-7)


def run_handshake(role: str, rec, mic_fs: int, spk_fs: int, out_dev,
                  *, amplitude: float = 0.3, poll_s: float = 3.0,
                  timeout: float = 300.0, wait_probe_s: float = 75.0,
                  wait_msg_s: float = 75.0, max_retries: int = 2,
                  amb_rms: float | None = None, log=print,
                  abort_cb=None, skip_fin: bool = False,
                  band_negotiation: bool = False, band_shrink: float = 0.8,
                  mesh_fin: bool = True, delockstep_s: float = 30.0,
                  lookback_s: float = 0.0) -> dict:
    """Drive HandshakeSM over real audio (testlib Recorder + play_blocking).

    The single hardware binding for the handshake: both meshmodem (via
    meshmodem_node) and the standalone ofdmhs test drive it.  Generic:
    everything mesh-specific is behind a keyword (`mesh_fin`, `abort_cb`).

    Composed from the OTA-validated ofdmhs_node loop: listen windows start
    after our own TX; the window is NOT advanced on probe rx (the responder
    queues probe+BAT_REPORT back-to-back); decode attempts are ungated (probe
    completeness check + RS/CRC self-guard); every TX carrier-senses first
    (a timeout-triggered retransmission must not talk over a late-arriving
    message).  `amb_rms` is the broadband ambient floor for that gate --
    pass the mesh run's calibrated value; if None it is sampled from the
    recorder tail at entry (which mis-calibrates if the peer is already
    transmitting, so callers that have a boot-time floor should pass it).

    Two additions over ofdmhs_node: r_wait_ack decodes BOTH probe and msg
    (an initiator that missed our probe re-probes -- hearing that is the only
    recovery signal), and the responder's r_linger treats a q-keyed mesh
    preamble lock as an implicit FIN (once the peer sends data frames it has
    finished; lingering deaf just strands its first DATA+ACK exchanges).

    Returns {"ok", "state", "my_tx_bat", "peer_tx_bat", "peer_rate",
             "probe_rx", "elapsed_s"}.
    """
    from testlib import play_blocking

    # De-lockstep the two ends' retry timers. With identical waits both ends
    # time out near-simultaneously (arm times differ only by one message
    # flight), so on a weak link their retransmissions CROSS, and carrier
    # sense cannot referee once the received level sits under the TX-gate
    # threshold. The responder waits an extra 30s -- more than its longest
    # retransmit turn (probe+report ~17s + cs deferral) -- so the initiator
    # times out first, alone, and the responder answers the RETRY it then
    # hears instead of colliding with it.
    if role == "responder":
        wait_msg_s = wait_msg_s + delockstep_s

    sm = HandshakeSM(role, max_retries=max_retries,
                     wait_probe_s=wait_probe_s, wait_msg_s=wait_msg_s,
                     skip_fin=skip_fin, band_negotiation=band_negotiation)
    t0 = time.time()
    now = lambda: time.time() - t0
    # Start the listen window `lookback_s` in the PAST.  The Recorder captures
    # continuously, so entering the handshake even a second after the peer's
    # probe finished discards audio already on disk, forcing the peer through
    # its whole wait_probe_s before re-probing.
    #
    # The hazard is reaching back far enough to decode our OWN probe and grade
    # it as the peer's, so this must stay well inside one handshake cycle: a
    # node's previous probe is at least a handshake timeout away (tens of
    # seconds), while what we want to recover is only the last few seconds.
    # 0 keeps the old behaviour.
    listen_start = max(0, rec.frames_captured - int(max(0.0, lookback_s) * mic_fs))
    if amb_rms is None:
        tail = rec.data()[-int(0.4 * mic_fs):]
        amb_rms = (float(np.sqrt(np.mean(tail ** 2))) if len(tail) else 0.0)
    peer_rate = None
    # Worst-margin RobustLink decode of the handshake -- the only
    # link-quality figure a handshake-only run produces.
    rl_stats: dict | None = None
    probe_info = None
    last_attempt = -1e9
    tx_log: list[str] = []
    tx_windows: list[dict] = []      # wall-clock spans this node was talking
    fail_reason = None

    def _aborted():
        return abort_cb is not None and abort_cb()

    def _ofdm_hs_rates():
        return HS_RATES

    def attempt_rx() -> bool:
        """Decode whatever the SM currently expects.  Returns True if a
        frame was accepted (the SM advanced and may have emitted actions)."""
        nonlocal listen_start, peer_rate, probe_info, rl_stats
        expect = sm.expecting()
        if not expect:
            return False
        cap = rec.data()[listen_start:]
        max_n = int(60.0 * mic_fs)
        if len(cap) > max_n:
            listen_start += len(cap) - max_n
            cap = cap[-max_n:]
        if len(cap) <= int(4.0 * mic_fs):
            return False
        # expect is "probe", "msg", or "probe+msg"; msg first (the
        # normal next step), probe as the recovery path.
        # RATE LOCK: once the peer's TX rate is known from its first
        # decodable signal, stop trial-decoding the other candidate
        # -- it halves decode cost per poll (turnaround latency and
        # the CPU pressure that used to starve capture).
        rates_try = ((peer_rate,) if peer_rate else
                     _ofdm_hs_rates())
        
        if "msg" in expect:
            _rlst: dict = {}
            got = try_decode_msg(cap, mic_fs, rates=rates_try,
                                 band=sm.peer_tx_band, stats_out=_rlst)
            if got:
                rate, mtype, proof, body = got
                peer_rate = peer_rate or rate
                # Keep the WORST-margin message of the handshake, not the last:
                # a report is only useful if it shows the exchange's tightest
                # moment, and BAT_REPORT (longest frame, blind step) is usually
                # it while GO/FIN are short and easy.
                if _rlst and (rl_stats is None
                              or (_rlst.get("margin_min") or 1.0)
                              < (rl_stats.get("margin_min") or 1.0)):
                    rl_stats = dict(_rlst)
                log(f"    [hs] RX {_MSG_NAMES.get(mtype)}  "
                    f"G={_rlst.get('n_groups')} "
                    f"evm={_rlst.get('evm_pct')}%  t={now():.1f}")
                listen_start = rec.frames_captured
                sm.heard_msg(mtype, proof, body, now())
                return True
        if "probe" in expect:
            # In r_wait_ack a "probe" claims the initiator RESTARTED its whole
            # turn, and acting on it means transmitting over whatever the peer
            # is actually sending. The RobustLink lead-in of the peer's own
            # BAT_ACK/GO also validates as a SHORT high-coherence probe
            # fragment (25 syms, coh .998), so a real re-probe -- the full
            # ~171-symbol train, repeating on the initiator's own timer -- is
            # distinguished by requiring a near-complete train here; this
            # costs at most one poll. Fragments stay acceptable in the initial
            # probe waits, where the peer is silent until we reply.
            # Thresholds scale with the probe's OWN symbol count: a 5s probe
            # is ~171 symbols at n_fft=1024 but only 42 at 4096, so a fixed
            # "100 symbols = complete" test misclassifies every full probe at
            # higher n_fft as a fragment.
            full_syms = int(hs_cfg(spk_fs, sm.peer_tx_band)["n_probe_syms"])
            near_full = max(12, int(0.6 * full_syms))
            probe_min_syms = near_full if sm.state == "r_wait_ack" else 12
            got = try_decode_probe(cap, mic_fs, rates=rates_try,
                                   min_syms=probe_min_syms,
                                   band=sm.peer_tx_band)
            # Mid-flight guard: a SHORT symbol run ending near the live edge
            # of the capture is a probe still PLAYING whose coherence run
            # broke early (a dropout defeats try_decode_probe's completeness
            # check). Measuring the head alone is wildly optimistic and can
            # seed a degenerate exchange, so wait a poll -- the next attempt
            # sees the full train.
            if (got and got[2]["n_syms"] < near_full
                    and got[2].get("tail_s", 99.0) < 6.0):
                log(f"    [hs] probe fragment ({got[2]['n_syms']} "
                    f"syms, tail {got[2]['tail_s']:.1f}s) at live "
                    f"edge -- waiting for full train  t={now():.1f}")
                got = None
            # Zero-active gate, ALL probe-wait states: RobustLink SYNC symbols
            # carry the SAME known-BPSK sequence as the probe, so a long
            # message frame (e.g. a BAT_ACK) can validate as a FULL-length
            # probe train (coh 1.000) while grading 0 active carriers, which
            # no real probe on a viable channel does. Acting on one is never
            # right: in r_wait_ack it repeats our turn over the peer's
            # message; in i/r_wait_probe it advances the handshake with an
            # all-zero BAT that can only end one_way-rejected. Ignoring costs
            # one poll; a real awful channel just retries and re-measures.
            if got and got[2].get("n_active", 1) == 0:
                # try_decode_probe keeps the best candidate BY COHERENCE, and
                # a completed message frame's phantom (coh 1.0) outscores the
                # real probe (~0.99) that aired just BEFORE it (probe always
                # precedes the REPORT that spawns the phantom), shadowing the
                # real probe for the rest of the window. So: re-decode the
                # window TRUNCATED just before the phantom -- the earlier
                # real probe, if present, now wins.
                cut = got[2].get("start_mic", 0)
                got2 = (try_decode_probe(cap[:cut], mic_fs,
                                         rates=rates_try,
                                         min_syms=probe_min_syms,
                                         band=sm.peer_tx_band)
                        if cut > int(4.0 * mic_fs) else None)
                if got2 and got2[2].get("n_active", 1) > 0:
                    log(f"    [hs] phantom at +{cut/mic_fs:.1f}s "
                        f"shadowed a real probe -- recovered from "
                        f"truncated window  t={now():.1f}")
                    got = got2
                else:
                    # A REAL probe can also grade 0-active if the allocation
                    # policy is too harsh (ISI de-rating zeroing a valid probe).
                    # Modem.allocate backs the de-rating off rather than emit
                    # an empty table, so reaching this point should mean a
                    # genuine phantom -- but the two look identical from here.
                    log(f"    [hs] probe-shaped frame with 0 active "
                        f"carriers in {sm.state} -- ignoring (likely "
                        f"the peer's own message; if it repeats every "
                        f"poll, check for an allocation that zeroes "
                        f"everything)  t={now():.1f}")
                    got = None
            if got:
                rate, bat_bytes, info = got
                peer_rate = rate
                probe_info = info
                log(f"    [hs] RX probe rate={rate} "
                    f"coh={info['coherence']} {info['bat_summary']}")
                # Only this end can see which part of the peer's band survived
                # the path; if the edges are dead the SM turns this into
                # MSG_BAND + retrain (see ofdm_handshake.recommend_band).
                rec_band = None
                if band_negotiation and info.get("spectrum"):
                    rec_band = recommend_band(info["spectrum"], band_shrink)
                    if rec_band:
                        log(f"    [hs] band fallback: {rec_band[0]:.0f}-"
                            f"{rec_band[1]:.0f} Hz usable -- asking peer "
                            f"to narrow  t={now():.1f}")
                sm.heard_probe(bat_bytes, now(), band=rec_band)
                return True
        # implicit FIN: the peer is already sending mesh data frames
        # -> it finished the handshake, stop lingering deaf
        if (mesh_fin and sm.state == "r_linger"
                and peer_rate is not None):
            mcoh = _mesh_preamble_coherence(cap, mic_fs,
                                            int(peer_rate))
            if mcoh >= MESH_GATE:
                log(f"    [hs] mesh frame on air (coh={mcoh:.2f}) "
                    f"-> implicit FIN  t={now():.1f}")
                listen_start = rec.frames_captured
                sm.heard_msg(MSG_FIN, 0, b"", now())
                return True
        return False

    sm.start(now())
    while time.time() - t0 < timeout:
        if _aborted():
            # external kill switch (e.g. the mesh run's airtime-overrun
            # guard): stop transmitting NOW -- every further probe/BAT we
            # air lands in someone else's window
            log(f"    [hs] ABORTED externally  t={now():.1f}")
            return {"ok": False, "state": "aborted",
                    "my_tx_bat": sm.my_tx_bat,
                    "peer_tx_bat": sm.peer_tx_bat,
                    "peer_rate": peer_rate, "probe_rx": probe_info,
                    "tx_windows": tx_windows,
                    "elapsed_s": round(now(), 1)}
        acts, sm.actions = sm.actions, []
        for a in acts:
            if a[0] == "send_probe":
                audio = make_probe_audio(spk_fs, sm.my_tx_band)
            elif a[0] == "send_msg":
                # sm.my_tx_bat is set only once the peer has told us our own
                # table -- i.e. every message after BAT_REPORT.  Passing it
                # unlocks the faster grouped payload mode (see
                # ofdm_handshake.fast_groups_for); before that it is None and
                # the message goes out in the always-safe bootstrap mode.
                audio = make_msg_audio(spk_fs, a[1], a[2], a[3],
                                       sm.my_tx_band, my_tx_bat=sm.my_tx_bat)
            else:
                if a[0] == "fail":
                    fail_reason = a[1]
                continue                       # done/fail read from sm state
            # Level every burst to the PROBE's average power, not to a common
            # peak: peak-normalising sent each message 2.5-4.1 dB quieter than
            # the probe that graded the channel (see scale_hs_tx).
            audio, lvl = scale_hs_tx(audio, amplitude, spk_fs,
                                     is_probe=(a[0] == "send_probe"),
                                     band=sm.my_tx_band)
            what = "probe" if a[0] == "send_probe" else _MSG_NAMES.get(a[1])
            # carrier sense: never talk over the peer -- a timeout-triggered
            # RETRY could otherwise collide with a late-arriving message.
            # No TX may BEGIN at/after the timeout deadline: the caller sizes
            # its window clamp for one in-flight burst past the deadline at
            # most, so a cs-deferred TX leaking past it would overrun the
            # airtime window this timeout protects.
            hard_stop = t0 + timeout
            cs_deadline = min(time.time() + 25.0, hard_stop)
            t_turn = time.time()
            deferred = False
            while (time.time() < cs_deadline and not _aborted()
                   and _air_busy(rec, mic_fs, amb_rms, listen_start)):
                deferred = True
                time.sleep(0.5)
            if _aborted() or time.time() >= hard_stop:
                continue          # loop top returns aborted / times out
            # Deferring alone is not enough.  A frame that landed while we
            # waited can make this queued action obsolete, and sending it
            # anyway is what puts both ends on the air together: a timeout
            # RETRY goes out right after the BAT_REPORT it asked for, the peer
            # re-sends the report, and our ACK for the first copy lands on top
            # of it.  So drain the receiver first and drop what went stale.
            # MSG_BAND is exempt -- the SM commits peer_tx_band when it queues
            # one, so discarding it leaves the two ends on different bands.
            is_band = (a[0] == "send_msg" and a[1] == MSG_BAND)
            if deferred and not is_band and attempt_rx():
                log(f"    [hs] TX {what} abandoned: peer frame arrived first")
                sm.hold_off(time.time() - t_turn)
                break
            # Level is logged per burst: the whole point is that these are now
            # equal in AVERAGE power, which a peak figure would hide.
            log(f"    [hs] TX {what} ({len(audio)/spk_fs:.1f}s)  t={now():.1f}"
                f"  [rms={lvl.get('delivered_rms')} peak={lvl.get('peak')} "
                f"papr={lvl.get('papr_db')}dB {lvl.get('mode')}"
                + (f" -{abs(lvl['clamped_db'])}dB" if lvl.get("clamped_db")
                   else "") + "]")
            tx_log.append(f"{now():.1f} TX {what}")
            # WALL-clock TX intervals, returned to the caller.  A node's own
            # speaker couples into its own mic at full scale, which the
            # post-hoc capture analysis would otherwise misread as sustained
            # input clipping.  Only the transmitter knows when it was talking.
            tx_windows.append({"wall": time.time(),
                               "dur_s": round(len(audio) / spk_fs, 2),
                               "what": what})
            play_blocking(audio, spk_fs, out_dev)
            listen_start = rec.frames_captured
            # The reply timer was armed when the SM decided to send; the peer
            # could not answer during our turn, so give that time back.
            sm.hold_off(time.time() - t_turn)

        if sm.done or sm.failed:
            break

        if (time.time() - t0) - last_attempt >= poll_s:
            last_attempt = time.time() - t0
            attempt_rx()

        sm.tick(now())
        time.sleep(0.25)

    if not sm.done and not sm.failed and sm.state == "r_linger":
        # The binding timeout truncated the linger (reneg timeouts are shorter
        # than a full linger).  The SM's own expiry semantics apply: a
        # lingering responder finishes ok -- GO is presumed to have landed.
        sm.tick(sm._deadline)

    return {
        "ok":          bool(sm.done),
        "sm":          sm,             # callers reporting detail want the log
        "fail_reason": fail_reason,
        "tx_log":      tx_log,
        "tx_windows":  tx_windows,
        "state":       sm.state,
        "my_tx_bat":   sm.my_tx_bat,
        "peer_tx_bat": sm.peer_tx_bat,
        "peer_rate":   peer_rate,
        "my_tx_band":  sm.my_tx_band,
        "peer_tx_band": sm.peer_tx_band,
        "probe_rx":    probe_info,
        "rl":          rl_stats,
        "elapsed_s":   round(now(), 1),
    }


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #

def _selftest() -> int:
    import css_modem as cm
    from numpy.fft import rfft, irfft, rfftfreq
    from ofdm_handshake import _resample
    rng = np.random.RandomState(5)
    ok_all = True

    def chan(x, fs, snr_db, kind="good", seed=1):
        F = rfftfreq(len(x), 1 / fs)
        if kind == "good":
            H = np.exp(-((F - 900) / 2600) ** 2) + 0.35 * np.exp(-((F - 2700) / 900) ** 2)
        else:
            H = 1 / (1 + np.exp(-(F - 2500) / 300)) * np.exp(-((F - 3500) / 2500) ** 2)
        y = irfft(rfft(x) * (H / H.max()), n=len(x))
        p = float(np.mean(y ** 2))
        return y + np.random.RandomState(seed).randn(len(y)) * np.sqrt(p / 10 ** (snr_db / 10))

    def clicky(fs, seed):
        c = np.zeros(int(0.2 * fs)); c[40] = 0.8
        for i in range(41, len(c)):
            c[i] = c[i - 1] * 0.994
        return np.concatenate([c, np.random.RandomState(seed).randn(int(0.4 * fs)) * 1e-4])

    def check(name, cond):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  [{name:34}] {'PASS' if cond else 'FAIL'}")

    # a realistic negotiated BAT (mixed, some carriers off)
    cfg48 = mesh_cfg(48000)
    bat48 = rng.choice([0, 1, 2, 2, 4], size=cfg48["n_data_carriers"]).astype(np.int32)
    pkt = cm.pack_packet(1, 2, cm.TYPE_DATA, 7, b"hello-ofdm\x00\x01\x00")
    link12 = {"tx_rate": 48000, "tx_bat": bat48, "rx_rate": 48000, "rx_bat": bat48}

    sig = encode_packet_ofdm(link12, pkt)
    sig = 0.3 * sig / np.max(np.abs(sig))
    pad = np.zeros(int(0.4 * 48000))

    got, st = try_decode_packet_ofdm(link12, np.concatenate([pad, sig, pad]).astype(np.float32), 48000)
    check("clean round-trip (binary payload)", got == pkt)

    rx = np.concatenate([clicky(48000, 3), chan(sig, 48000, 15, "good", 4), pad])
    got, st = try_decode_packet_ofdm(link12, rx.astype(np.float32), 48000)
    check("good channel 15dB + click", got == pkt)

    # 44.1k TX -> 48k mic -> resample (the reverse mesh leg).  The BAT must be
    # band-realistic: a NEGOTIATED table zeroes the carriers this highpass
    # channel kills (that is the whole point of the handshake) -- a random BAT
    # loading bits onto -20 dB carriers fails RS by construction, not by bug.
    cfg441 = mesh_cfg(44100)
    sc = cfg441["sc_spacing_hz"]
    f0 = cfg441["freq_lo_hz"]
    pilots441 = set(range(0, cfg441["n_occupied"], cfg441["pilot_spacing"]))
    freqs_d = [f0 + i * sc for i in range(cfg441["n_occupied"])
               if i not in pilots441]
    bat441 = np.array([0 if f < 2700 else rng.choice([1, 2])
                       for f in freqs_d], dtype=np.int32)
    link21 = {"tx_rate": 44100, "tx_bat": bat441, "rx_rate": 44100, "rx_bat": bat441}
    sig2 = encode_packet_ofdm(link21, pkt)
    sig2 = 0.3 * sig2 / np.max(np.abs(sig2))
    air = chan(sig2, 44100, 10, "weak", 6)
    mic = _resample(air, 44100, 48000)
    rx2 = np.concatenate([clicky(48000, 7), mic, np.zeros(int(0.3 * 48000))])
    got, st = try_decode_packet_ofdm(link21, rx2.astype(np.float32), 48000)
    check("weak channel 10dB + resample leg", got == pkt)

    # THE OTA killer scenario -- fade migration: carriers healthy when the BAT
    # was negotiated go DEAD at data time.  Notch out 4 active data carriers
    # entirely (plus noise); the interleaved soft FEC must ride through what
    # previously produced ~100% EVM byte bursts and total RS failure.
    def notch(x, fs, kill_hz, snr_db, seed):
        from numpy.fft import rfft, irfft, rfftfreq
        F = rfftfreq(len(x), 1 / fs)
        Hn = np.ones_like(F)
        for f0 in kill_hz:
            Hn[np.abs(F - f0) < 30.0] = 0.0
        y = irfft(rfft(x) * Hn, n=len(x))
        p = float(np.mean(y ** 2))
        return y + np.random.RandomState(seed).randn(len(y)) * np.sqrt(p / 10 ** (snr_db / 10))

    pilots48 = set(range(0, cfg48["n_occupied"], cfg48["pilot_spacing"]))
    sc48, f048 = cfg48["sc_spacing_hz"], cfg48["freq_lo_hz"]
    data_freqs = [f048 + i * sc48 for i in range(cfg48["n_occupied"])
                  if i not in pilots48]
    active_hz = [f for f, b in zip(data_freqs, bat48) if b > 0]
    kill = active_hz[2::max(1, len(active_hz) // 4)][:4]
    rx3 = np.concatenate([pad, notch(sig, 48000, kill, 12, 9), pad])
    got, st = try_decode_packet_ofdm(link12, rx3.astype(np.float32), 48000)
    check("fade migration: 4 dead carriers", got == pkt)

    # isolation: a handshake message must NOT decode as a mesh packet ...
    msg_audio = make_msg_audio(48000, 3, 0x1234, b"")
    msg_audio = 0.3 * msg_audio / np.max(np.abs(msg_audio))
    got, _ = try_decode_packet_ofdm(link12, np.concatenate([pad, msg_audio, pad]).astype(np.float32), 48000)
    check("RL msg does not decode as data", got is None)
    # ... and a mesh data frame must not decode as a handshake message
    got2 = try_decode_msg(np.concatenate([pad, sig, pad]).astype(np.float32), 48000)
    check("data frame does not decode as msg", got2 is None)

    # wrong BAT must not yield a packet
    bad = dict(link12, rx_bat=np.roll(bat48, 7))
    got, _ = try_decode_packet_ofdm(bad, np.concatenate([pad, sig, pad]).astype(np.float32), 48000)
    check("wrong BAT rejected", got != pkt)

    # ---- passive OFDM-activity detector ---------------------------------- #
    hit, sc_ = detect_ofdm_activity(
        np.concatenate([pad, chan(sig, 48000, 12, "good", 11), pad]), 48000)
    check(f"detect: data frame 12dB ({sc_})", hit)

    air_d = chan(sig2, 44100, 10, "weak", 12)          # 44.1k TX, 48k mic leg
    hit, sc_ = detect_ofdm_activity(_resample(air_d, 44100, 48000), 48000)
    check(f"detect: 44.1k leg 10dB ({sc_})", hit)

    probe = make_probe_audio(48000)
    probe = 0.3 * probe / np.max(np.abs(probe))
    hit, sc_ = detect_ofdm_activity(
        np.concatenate([pad, chan(probe[:int(4.0 * 48000)], 48000, 12, "good", 13)]),
        48000)
    check(f"detect: handshake probe ({sc_})", hit)

    noise = np.random.RandomState(14).randn(int(5.0 * 48000)) * 0.05
    hit, sc_ = detect_ofdm_activity(noise, 48000)
    check(f"detect: rejects noise ({sc_})", not hit)

    t = np.arange(int(5.0 * 48000)) / 48000            # CSS-style swept chirp
    sweep = 0.3 * np.sin(2 * np.pi * (600 * t + (3000 - 600) / (2 * 5.0) * t ** 2))
    hit, sc_ = detect_ofdm_activity(sweep + noise, 48000)
    check(f"detect: rejects chirp sweep ({sc_})", not hit)

    tone = 0.3 * np.sin(2 * np.pi * 1500 * t)
    hit, sc_ = detect_ofdm_activity(tone + noise, 48000)
    check(f"detect: rejects steady tone ({sc_})", not hit)

    hit, sc_ = detect_ofdm_activity(noise, 48000, amb_rms=0.05)
    check("detect: stage-1 gate on quiet", not hit and sc_ == 0.0)

    # ---- probe-SNR estimator characterisation --------------------------- #
    # Finding (20260720): the DEFAULT 'mean' estimator, snr = 2|mean H|^2 /
    # var(H), is already fade-aware -- its var(H) term folds any within-probe
    # dropout into the noise floor, so a carrier that briefly nulls is graded
    # sharply DOWN.  A sub-block 'percentile' estimator is NOT more
    # conservative here (it can skip a single bad block), so 'mean' stays the
    # default.  The real probe->data optimism is the post-probe TIME gap,
    # which no single-probe statistic can see -- covered by snr_margin_db.
    from ofdm_modem import _probe_snr_lin
    rng2 = np.random.RandomState(21)
    S, NC = 160, 6
    H = np.ones((S, NC), dtype=complex)
    H += 0.03 * (rng2.randn(S, NC) + 1j * rng2.randn(S, NC))
    H[S - S // 8:, 2] *= 0.02                           # carrier 2: brief null
    H[:S // 8, 4] *= 0.02                               # carrier 4: brief null
    snr_mean = 10 * np.log10(_probe_snr_lin(H, "mean"))
    faded, steady = [2, 4], [0, 1, 3, 5]
    steady_min = min(snr_mean[c] for c in steady)
    check(f"snr-est: mean var-term grades brief fades down "
          f"({steady_min - max(snr_mean[c] for c in faded):.0f}dB)",
          all(snr_mean[c] < steady_min - 10 for c in faded))
    # percentile option runs and stays finite / sane on the same input
    snr_pct = 10 * np.log10(_probe_snr_lin(H, "percentile", 15.0, 8))
    check("snr-est: percentile option returns finite SNR",
          np.all(np.isfinite(snr_pct)) and snr_pct.shape == (NC,))

    # ---- symmetric-rate link selection ---------------------------------- #
    # An asymmetric handshake (A->B strong, B->A weak) must collapse to the
    # weaker per-carrier allocation on BOTH ends, computed identically without
    # extra signalling.  D_ab = A->B allocation (bytes), D_ba = B->A.
    from ofdm_modem import bat_to_bytes
    ncar = mesh_cfg(48000)["n_data_carriers"]
    rs = np.random.RandomState(31)
    D_ab = rs.choice([0, 1, 2, 2, 4], size=ncar).astype(np.int32)   # strong
    D_ba = rs.choice([0, 0, 1, 2], size=ncar).astype(np.int32)      # weak
    # A holds: my_tx = D_ab (what A transmits, allocated by B) ... but by the
    # handshake convention my_tx_bat is what the PEER measured of MY probe =
    # my transmit direction.  For node A: my_tx=D_ab, peer_tx=D_ba.
    linkA = make_link({"my_tx_bat": bat_to_bytes(D_ab),
                       "peer_tx_bat": bat_to_bytes(D_ba), "peer_rate": 48000},
                      48000, symmetric=True)
    linkB = make_link({"my_tx_bat": bat_to_bytes(D_ba),
                       "peer_tx_bat": bat_to_bytes(D_ab), "peer_rate": 48000},
                      48000, symmetric=True)
    expect = np.minimum(D_ab, D_ba)
    check("symlink: both ends agree on min BAT",
          np.array_equal(linkA["tx_bat"], expect)
          and np.array_equal(linkB["tx_bat"], expect)
          and np.array_equal(linkA["tx_bat"], linkB["tx_bat"]))
    check("symlink: tx==rx (both directions on weak rate)",
          np.array_equal(linkA["tx_bat"], linkA["rx_bat"])
          and linkA["symmetric"] and linkA["tx_gross_bps"] == linkB["tx_gross_bps"])
    # independent policy keeps the two directions distinct
    ind = make_link({"my_tx_bat": bat_to_bytes(D_ab),
                     "peer_tx_bat": bat_to_bytes(D_ba), "peer_rate": 48000},
                    48000, symmetric=False)
    check("symlink: independent policy keeps directions distinct",
          np.array_equal(ind["tx_bat"], D_ab)
          and np.array_equal(ind["rx_bat"], D_ba) and not ind["symmetric"])
    # a round-trip over the symmetric (weak) BAT still carries a packet
    got, _ = try_decode_packet_ofdm(
        linkA, np.concatenate([pad, 0.3 * encode_packet_ofdm(linkA, pkt)
                               / np.max(np.abs(encode_packet_ofdm(linkA, pkt))),
                               pad]).astype(np.float32), 48000)
    check("symlink: round-trip over min BAT decodes", got == pkt)

    # ---- allocator floors vs what the FEC chain actually needs ----------- #
    # SNR_FLOOR_DB is the AWGN threshold per modulation.  Too HIGH and carriers
    # that would carry data are switched off (16-QAM sat 4 dB high until
    # 20260804, so it was unreachable on channels that would have held it); too
    # LOW and the allocator over-drives, which returns errors instead of nothing.
    # A full waterfall is too slow for a selftest, so bracket each floor: a frame
    # must decode just above it, and must NOT decode well below it.
    # Brackets are the MEASURED AWGN waterfall, not SNR_FLOOR_DB minus a
    # constant.  They were the latter, and that coupling made this test assert
    # the chain's own weakness: it required 16-QAM to FAIL at floor-5 dB
    # (9 dB), so fixing the inverted CPE sign and adding the pilot phase slope
    # -- which moved the knee down -- broke it in the GOOD direction.
    #
    # Re-measured after those fixes, 6 trials per point, mesh geometry at
    # 48 kHz (first SNR with 6/6, and the shoulder below it):
    #
    #   BPSK    3 dB   (0/6 at 1, 5/6 at 2)
    #   QPSK    4 dB   (0/6 at 2, 4/6 at 3)          was 6 dB
    #   16-QAM 10 dB   (0/6 at 5, 1/6 at 6, 5/6 at 9)
    #   64-QAM 11 dB   (0/6 at 9, 2/6 at 10)         was 15 dB
    #
    # 16-QAM has a long shoulder, hence the wide lower bracket; a tighter one
    # would flake.  SNR_FLOOR_DB itself is deliberately NOT changed here --
    # lowering it changes negotiated rates over the air, which is a separate
    # decision from a demodulator fix.
    _AWGN_PASS = {1: 4.0, 2: 5.0, 4: 11.0, 6: 12.0}
    _AWGN_FAIL = {1: 1.0, 2: 2.0, 4:  5.0, 6:  9.0}

    def _frame_at(bits, snr_db, seed):
        _rs5 = np.random.RandomState(seed)
        _cfg_f = mesh_cfg(48000)
        _bat_f = np.full(_cfg_f["n_data_carriers"], bits, dtype=np.int32)
        _lk_f = {"tx_rate": 48000, "tx_bat": _bat_f,
                 "rx_rate": 48000, "rx_bat": _bat_f}
        _pk_f = bytes(_rs5.randint(0, 256, MESH_CHUNK - 2).tolist())
        _sg = encode_packet_ofdm(_lk_f, _pk_f)
        _sg = 0.3 * _sg / np.max(np.abs(_sg))
        # per-carrier SNR -> white-noise variance (see waterfall calibration)
        _occ = _cfg_f["n_occupied"] / (_cfg_f["n_fft"] / 2)
        _s2 = float(np.mean(_sg ** 2)) / (10 ** (snr_db / 10.0) * _occ)
        _rxf = np.concatenate([pad, _sg, pad])
        _rxf = _rxf + _rs5.randn(len(_rxf)) * math.sqrt(_s2)
        _g, _ = try_decode_packet_ofdm(_lk_f, _rxf.astype(np.float32), 48000)
        return _g == _pk_f

    for _b, _nm in ((1, "BPSK"), (2, "QPSK"), (4, "16-QAM"), (6, "64-QAM")):
        check(f"awgn {_nm}: decodes at {_AWGN_PASS[_b]:.0f}dB",
              _frame_at(_b, _AWGN_PASS[_b], 100 + _b))
        check(f"awgn {_nm}: still fails at {_AWGN_FAIL[_b]:.0f}dB",
              not _frame_at(_b, _AWGN_FAIL[_b], 200 + _b))

    # ---- frame body size (mesh_chunk) ------------------------------------ #
    # One packet is one frame, so chunk_bytes is what bounds an application
    # payload -- and it used to be stuck at 48B (44B of payload) because the FEC
    # geometry assumed the RS-coded body fitted ONE codeword.  Past 237B reedsolo
    # adds a second codeword's parity, so the geometry has to be measured; see
    # ofdm_modem._rs_encoded_len.  Verify round-trips at sizes on both sides of
    # that boundary, since nothing else in the suite leaves the default.
    _saved_chunk = MESH_CHUNK
    try:
        for _ch in (48, 237, 512, 1024):
            globals()["MESH_CHUNK"] = _ch
            _cfg_c = mesh_cfg(48000)
            _bat_c = np.full(_cfg_c["n_data_carriers"], 2, dtype=np.int32)
            _lk = {"tx_rate": 48000, "tx_bat": _bat_c,
                   "rx_rate": 48000, "rx_bat": _bat_c}
            _pkt_c = bytes(rng.randint(0, 256, _ch - 2).tolist())
            _sig_c = encode_packet_ofdm(_lk, _pkt_c)
            _sig_c = 0.3 * _sig_c / np.max(np.abs(_sig_c))
            _rx_c = np.concatenate([pad, chan(_sig_c, 48000, 15, "good", 9),
                                    pad]).astype(np.float32)
            _got_c, _ = try_decode_packet_ofdm(_lk, _rx_c, 48000)
            check(f"chunk {_ch:4d}B: {_ch - 2}B packet round-trips at 15dB",
                  _got_c == _pkt_c)
            # The frame must be sized from the REAL encoder output, not from
            # seq+chunk+ecc.  Now that the body is interleaved across equal RS
            # codewords the measured length is the interleaved one -- still
            # measured rather than assumed, which is the property under test --
            # and BOTH ends must derive the same figure without signalling it.
            _mod_c = MeshModulator(_cfg_c, _bat_c)
            _dem_c = MeshDemodulator(_cfg_c, _bat_c)
            _exp = len(rs_encode_interleaved(
                _mod_c.rsc, bytes(_cfg_c["seq_bytes"] + _ch),
                _mod_c.il_depth, _mod_c.il_k, _mod_c.ecc_bytes))
            check(f"chunk {_ch:4d}B: geometry matches interleaved RS output "
                  f"({_exp}B, depth {_mod_c.il_depth})",
                  _mod_c.payload_bytes == _exp
                  and _dem_c.payload_bytes == _exp)
            check(f"chunk {_ch:4d}B: both ends derive the same interleave",
                  (_mod_c.il_depth, _mod_c.il_k)
                  == (_dem_c.il_depth, _dem_c.il_k))
        # and an oversized packet is refused with an actionable message
        globals()["MESH_CHUNK"] = 48
        try:
            encode_packet_ofdm({"tx_rate": 48000,
                                "tx_bat": np.full(mesh_cfg(48000)["n_data_carriers"],
                                                  2, dtype=np.int32)}, bytes(200))
            _refused = ""
        except ValueError as e:
            _refused = str(e)
        check("chunk: oversized packet names the mesh_chunk param",
              "mesh_chunk" in _refused)

        # An all-off BAT must be refused SYMMETRICALLY: a modulator that
        # silently rewrites it to BPSK-everywhere while the demodulator keeps
        # 0 bits per symbol sizes frames differently on each end, making the
        # transfer undecodable by construction. A silent asymmetry has no
        # symptom at this layer -- both objects construct fine -- so it is
        # checked directly.
        _cfg_z = mesh_cfg(48000)
        _zero  = np.zeros(_cfg_z["n_data_carriers"], dtype=np.int32)
        _raised = []
        for _cls in (MeshModulator, MeshDemodulator,
                     AdaptiveModulator, AdaptiveDemodulator):
            try:
                _cls(_cfg_z, _zero)
            except ValueError:
                _raised.append(_cls.__name__)
        check("empty BAT: both ends refuse it (no silent BPSK rewrite)",
              len(_raised) == 4)
        # and the explicit opt-in yields ONE table both ends agree on
        _b1 = bpsk_all_bat(_cfg_z["n_data_carriers"])
        check("empty BAT: bpsk_all_bat gives both ends the same geometry",
              AdaptiveModulator(_cfg_z, _b1).data_syms_per_pkt
              == AdaptiveDemodulator(_cfg_z, _b1).data_syms_per_pkt)
    finally:
        globals()["MESH_CHUNK"] = _saved_chunk

    # ---- handshake TX levelling ------------------------------------------ #
    # The negotiated BAT is only valid at the PSD the probe was measured at, so
    # every burst must deliver the probe's AVERAGE power.  Peak-normalising
    # instead sends each one at its own PAPR deficit: -2.5 dB for a
    # BAT_REPORT, -4.1 dB for a BAT_ACK at ladder step G=8.
    # scale_hs_tx / make_msg_audio / the MSG_* codes are imported at module
    # level; re-importing them here would shadow them for this whole function,
    # including the earlier cases that use them.
    _fs_l = 48000
    _bat_b = bat_to_bytes(np.ones(mesh_cfg(_fs_l)["n_data_carriers"],
                                  dtype=np.int32))
    _probe_a = make_probe_audio(_fs_l)

    def _rms(x):
        return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))

    def _pk(x):
        return float(np.max(np.abs(np.asarray(x, np.float64))))

    _p_scaled, _p_info = scale_hs_tx(_probe_a, 0.3, _fs_l, is_probe=True)
    check(f"tx level: probe stays peak-normalised to amplitude "
          f"({_pk(_p_scaled):.3f})", abs(_pk(_p_scaled) - 0.3) < 1e-6)
    _ref_rms = _rms(_p_scaled)
    _worst = 0.0
    for _nm, _mt, _body, _fast in (("REPORT boot", MSG_BAT_REPORT, _bat_b, False),
                                   ("REPORT fast", MSG_BAT_REPORT, _bat_b, True),
                                   ("ACK fast", MSG_BAT_ACK, _bat_b, True),
                                   ("GO", MSG_GO, b"", True)):
        _a = make_msg_audio(_fs_l, _mt, 0x1234, _body,
                            my_tx_bat=(bytes(_bat_b) if _fast else None))
        _s, _i = scale_hs_tx(_a, 0.3, _fs_l, is_probe=False)
        _worst = max(_worst, abs(20 * math.log10(_rms(_s) / _ref_rms)))
        # the old behaviour, for contrast
        _old = _rms(_a) * 0.3 / _pk(_a)
        check(f"tx level: {_nm} matches probe power "
              f"(was {20*math.log10(_old/_ref_rms):+.1f}dB)",
              abs(20 * math.log10(_rms(_s) / _ref_rms)) < 0.1
              and _pk(_s) <= 0.9 + 1e-9)
    check(f"tx level: every burst within 0.1dB of the probe ({_worst:.2f}dB)",
          _worst < 0.1)
    # No headroom left => the clamp binds and says so, rather than clipping.
    _a_hi = make_msg_audio(_fs_l, MSG_BAT_ACK, 0x1234, _bat_b,
                           my_tx_bat=bytes(_bat_b))
    _s_hi, _i_hi = scale_hs_tx(_a_hi, 0.95, _fs_l, is_probe=False)
    check(f"tx level: peak clamp holds at high amplitude "
          f"(peak {_pk(_s_hi):.3f}, reported {_i_hi.get('clamped_db')}dB)",
          _pk(_s_hi) <= 0.9 + 1e-9 and _i_hi.get("clamped_db") is not None)

    # ---- data-aided channel re-estimation (two-pass demod) --------------- #
    # A one-shot CE estimate carries its own noise into every data symbol that
    # follows it; re-fitting H from the frame's own decisions averages that
    # down.  Tested where it matters: a REVERBERANT channel (late echo beyond
    # the CP), which is what the OTA runs actually see.
    _rs4 = np.random.RandomState(5)
    _sig3 = encode_packet_ofdm(link12, pkt)
    _sig3 = 0.3 * _sig3 / np.max(np.abs(_sig3))
    _cp48 = mesh_cfg(48000)["cp_len"]
    _echo = np.zeros(len(_sig3) + 2 * _cp48)
    _echo[:len(_sig3)] += _sig3
    _echo[2 * _cp48:2 * _cp48 + len(_sig3)] += 0.4 * _sig3   # tail past the CP
    _p3 = float(np.mean(_echo ** 2))
    _rx3 = np.concatenate([
        np.zeros(int(0.4 * 48000)),
        _echo + _rs4.randn(len(_echo)) * math.sqrt(_p3 / 10 ** (18 / 10.0)),
        np.zeros(int(0.4 * 48000))]).astype(np.float32)

    def _dec(aided):
        cfg = dict(mesh_cfg(48000)); cfg["data_aided"] = aided
        dem = MeshDemodulator(cfg, link12["rx_bat"])
        return dem.decode_packet(_resample(_rx3, 48000, 48000))

    _got_a, _st_a = _dec(True)
    _got_b, _st_b = _dec(False)
    check("data-aided: still decodes over a post-CP echo",
          _got_a == pkt and _got_b == pkt)
    check(f"data-aided: lowers EVM "
          f"({(_st_b or {}).get('evm_pct')}% -> {(_st_a or {}).get('evm_pct')}%)",
          _st_a and _st_b and _st_a["evm_pct"] <= _st_b["evm_pct"])
    check("data-aided: reports the one-pass figure alongside",
          not _st_a.get("data_aided")
          or _st_a.get("evm_pct_1pass", 0) >= _st_a["evm_pct"])
    check("data-aided: off means untouched single-pass stats",
          "evm_pct_1pass" not in (_st_b or {}))

    # ---- a frame still on the air is not a failed frame ------------------ #
    # The RX poll runs on a live capture, so it routinely sees a frame whose
    # preamble has arrived and whose body has not.  Reporting that as an RS
    # failure makes the caller discard the capture and lose the frame for
    # good. With a 512-byte chunk the frame (4.8s) outlasts the poll interval
    # (4s), so this is the normal case, not an edge case.
    _saved_chunk2 = MESH_CHUNK
    try:
        globals()["MESH_CHUNK"] = 512
        _cfg_i = mesh_cfg(48000)
        _bat_i = np.full(_cfg_i["n_data_carriers"], 2, dtype=np.int32)
        _lk_i = {"tx_rate": 48000, "tx_bat": _bat_i,
                 "rx_rate": 48000, "rx_bat": _bat_i}
        _pkt_i = bytes(rng.randint(0, 256, MESH_CHUNK - 2).tolist())
        _sig_i = encode_packet_ofdm(_lk_i, _pkt_i)
        _sig_i = 0.3 * _sig_i / np.max(np.abs(_sig_i))
        _full = np.concatenate([pad, _sig_i]).astype(np.float32)
        # Mid-transmission snapshots: preamble present, body incomplete.  Below
        # ~30% the coherence scan has not locked yet and reports nothing at all,
        # which is also fine (the caller waits) -- the dangerous case is a LOCK
        # with an unfinished body, so that is what is asserted.
        for _frac in (0.3, 0.5, 0.9):
            _cut = len(pad) + int(_frac * len(_sig_i))
            _g, _s = try_decode_packet_ofdm(_lk_i, _full[:_cut], 48000)
            check(f"partial frame at {int(_frac*100)}%: reported incomplete, "
                  f"not RS failure",
                  _g is None and bool(_s) and _s.get("incomplete")
                  and not _s.get("rs_failed"))
        # ...and once the whole thing has landed it decodes as normal
        _g, _s = try_decode_packet_ofdm(
            _lk_i, np.concatenate([_full, pad]).astype(np.float32), 48000)
        check("complete frame after the partial polls still decodes",
              _g == _pkt_i)
        # A genuinely corrupt frame must still report rs_failed, with the frame
        # end so the caller can skip precisely past it.  Note ATTENUATING a third
        # of the body is not enough to break it -- the soft LLRs treat faded
        # carriers as erasures and it still decoded at 84% EVM -- so the body is
        # overwritten with noise instead.
        _bad = np.concatenate([pad, _sig_i, pad]).astype(np.float32).copy()
        _b0 = len(pad) + int(0.45 * len(_sig_i))
        _b1 = len(pad) + int(0.75 * len(_sig_i))
        _bad[_b0:_b1] = (rng.randn(_b1 - _b0) * 0.3).astype(np.float32)
        _g, _s = try_decode_packet_ofdm(_lk_i, _bad, 48000)
        check("corrupt frame still reports rs_failed (not incomplete)",
              _g is None and bool(_s) and _s.get("rs_failed")
              and not _s.get("incomplete"))
        check("rs_failed carries frame_end/rate so the caller can skip past it",
              bool(_s) and _s.get("frame_end") and _s.get("rate") == 48000
              and _s["frame_end"] <= len(_bad))
    finally:
        globals()["MESH_CHUNK"] = _saved_chunk2

    # ---- the probe must experience what the data will ------------------- #
    # A probe of identical symbols lets reverberation reach periodic steady
    # state, so its tail is absorbed into H as channel gain and the probe grades
    # a reverberant path as clean (measured OTA: 18 dB graded on a path that
    # delivered 4.5 dB).  The probe therefore ends with a NON-PERIODIC block
    # whose symbols each carry a different known pattern, so its tail lands on
    # the next symbol as real interference -- exactly as data does.  The two
    # blocks differ ONLY in excitation, so their difference is the ISI penalty.
    from ofdm_modem import build_config as _bc, ChannelProbe as _CProbe, _ofdm_td
    _pcfg = _bc(sample_rate=44100, freq_lo=400.0, freq_hi=8000.0,
                probe_duration_s=5.0)
    _CP_p, _N_p = _pcfg["cp_len"], _pcfg["n_fft"]
    check(f"probe: split into periodic + non-periodic "
          f"({_pcfg['n_probe_syms']} + {_pcfg['n_probe_meas_syms']})",
          _pcfg["n_probe_meas_syms"] >= 8 and _pcfg["n_probe_syms"] >= 8)
    _pb = _CProbe(_pcfg)
    _psig = _pb.build()
    _rs_p = np.random.RandomState(5)

    def _probe_penalty(taps, snr_db=30.0):
        y = np.convolve(_psig, taps)[:len(_psig) + 4 * _N_p]
        pw = float(np.mean(y ** 2))
        y = y + _rs_p.randn(len(y)) * math.sqrt(pw / 10 ** (snr_db / 10.0))
        rx = np.concatenate([np.zeros(13230), y, np.zeros(13230)])
        r = _CProbe(_pcfg).estimate(rx)
        if not r.get("valid") or r.get("isi_penalty_db") is None:
            return None
        return r["isi_penalty_db"]

    _one = np.zeros(_CP_p * 4); _one[0] = 1.0
    _p_clean = _probe_penalty(_one)
    check(f"probe: clean channel shows no ISI penalty ({_p_clean:+.1f}dB)",
          _p_clean is not None and abs(_p_clean) <= 1.5)
    # an echo the prefix COVERS must not be charged as ISI
    _t_in = _one.copy(); _t_in[_CP_p // 4] = 0.5
    _p_in = _probe_penalty(_t_in)
    check(f"probe: echo inside the prefix is not charged ({_p_in:+.1f}dB)",
          _p_in is not None and _p_in <= 2.5)
    # ...and one the prefix does NOT cover must be
    _t_out = _one.copy(); _t_out[_CP_p + 120] = 0.5
    _p_out = _probe_penalty(_t_out)
    check(f"probe: echo beyond the prefix IS charged ({_p_out:+.1f}dB)",
          _p_out is not None and _p_out >= 6.0)
    check("probe: penalty grows with the part of the tail past the prefix",
          _p_out > _p_in + 4.0)
    # the allocator must consume the REPRESENTATIVE number, not the periodic one
    _rx_o = np.concatenate([np.zeros(13230),
                            np.convolve(_psig, _t_out)[:len(_psig) + 4 * _N_p],
                            np.zeros(13230)])
    _r_o = _CProbe(_pcfg).estimate(_rx_o)
    check("probe: snr_db is the non-periodic measurement (allocator input)",
          _r_o.get("valid")
          and float(np.median(_r_o["snr_db"]))
              < float(np.median(_r_o["snr_db_periodic"])) - 4.0)

    # ---- ISI ceiling: measurement + how it composes with the margin ------ #
    # Synthetic channels with a KNOWN late-tap energy, so beta is checked
    # against arithmetic rather than against another estimate.
    from ofdm_modem import (isi_ratio, derate_snr_lin, BitAllocator,
                            SNR_FLOOR_DB)
    _cfg = mesh_cfg(48000)
    _N, _CP, _BL, _NO = (_cfg["n_fft"], _cfg["cp_len"], _cfg["bin_lo"],
                         _cfg["n_occupied"])

    def _H_of(taps):
        """Channel response at the occupied carriers for {delay: amplitude}."""
        h = np.zeros(_N)
        for d, a in taps.items():
            h[d] = a
        return np.fft.rfft(h)[_BL:_BL + _NO]

    b_clean = isi_ratio(_H_of({5: 1.0}), _N, _BL, _CP)
    check(f"isi: single tap inside CP -> beta~0 ({b_clean:.5f})",
          b_clean < 0.002)
    # one late tap at 25% amplitude = 6.25% energy, all of it beyond the CP
    b_late = isi_ratio(_H_of({5: 1.0, _CP + 40: 0.25}), _N, _BL, _CP)
    check(f"isi: tap beyond CP is counted ({b_late:.4f} vs 0.0625 expected)",
          abs(b_late - 0.0625) < 0.008)
    b_early = isi_ratio(_H_of({5: 1.0, _CP - 40: 0.25}), _N, _BL, _CP)
    check(f"isi: same tap INSIDE the CP is not ({b_early:.5f})",
          b_early < 0.004)
    check("isi: beta grows with reverberation",
          isi_ratio(_H_of({5: 1.0, _CP + 40: 0.5}), _N, _BL, _CP) > b_late)
    # a noisy but echo-free channel must not read as reverberant
    _rs3 = np.random.RandomState(11)
    _Hn = _H_of({5: 1.0}) + (_rs3.randn(_NO) + 1j * _rs3.randn(_NO)) * 0.02
    _snr = np.full(_NO, 10 ** (14 / 10.0))
    check("isi: additive noise is not mistaken for late energy",
          isi_ratio(_Hn, _N, _BL, _CP, _snr)
          < isi_ratio(_Hn, _N, _BL, _CP) * 0.75)

    # de-rating caps SINR at 1/beta no matter how loud the carrier is
    _capped = 10 * np.log10(derate_snr_lin(np.array([1e6]), 0.25))
    check(f"isi: derate caps a 60dB carrier at 1/beta ({_capped[0]:.1f}dB)",
          abs(_capped[0] - 6.0) < 0.2)

    _alloc = BitAllocator()
    _ps, _snr_db = set(), np.full(_NO, 25.0)
    a_off = _alloc.allocate(_snr_db, _ps, _NO, 9.0, 4)
    a_on = _alloc.allocate(_snr_db, _ps, _NO, 9.0, 4, isi_beta=0.25)
    a_on_lo = _alloc.allocate(_snr_db, _ps, _NO, 2.0, 4, isi_beta=0.25)
    check("isi: beta=0 leaves allocation exactly as before",
          np.array_equal(a_off, _alloc.allocate(_snr_db, _ps, _NO, 9.0, 4,
                                                isi_beta=0.0)))
    check(f"isi: ceiling pulls 25dB carriers off 16-QAM "
          f"({a_off.max()}b -> {a_on.max()}b)", a_on.max() < a_off.max())
    check("isi: a large margin does NOT mute the link (composition order)",
          int((a_on > 0).sum()) == _NO and int((a_on_lo > 0).sum()) == _NO)
    check("isi: margin still tightens allocation on a capped link",
          int(a_on.sum()) <= int(a_on_lo.sum()))

    # An all-off table must never leave the allocator because of de-rating: a
    # measured beta with ceiling below BPSK's floor allocating nothing leaves
    # a 0-active table that the peer reads as a phantom probe and ignores
    # every poll until the window expires.
    from ofdm_modem import Modem as _Modem
    _cfg_d = dict(_cfg); _cfg_d["isi_derate"] = 1.0
    _cfg_d["snr_margin_db"] = 2.0; _cfg_d["max_bits_per_bin"] = 2
    _m = _Modem(_cfg_d)
    _nd = _cfg["n_data_carriers"]
    _res_bad = {"valid": True, "snr_db": np.full(_NO, 14.0),
                "snr_lin": np.full(_NO, 10 ** 1.4), "isi_ratio": 1.25}
    _bat_bad = _m.allocate(_res_bad)
    check(f"isi: a below-BPSK ceiling backs off instead of muting the link "
          f"({int((_bat_bad > 0).sum())}/{len(_bat_bad)} active)",
          _bat_bad.any() and len(_bat_bad) == _nd)
    # An out-of-range beta is a failed MEASUREMENT, not a bad channel.  The IR is
    # recovered on the n_fft bin grid so its delay axis wraps at n_fft/fs (23.2ms
    # at 1024/44.1k); a longer room tail aliases flat and saturates the ratio
    # near 1.  Acting on that zeroes a working link's capacity down to a
    # backed-off BPSK allocation and fails the handshake, so beta above
    # _ISI_BETA_MAX must be IGNORED rather than de-rated on.
    from ofdm_modem import _ISI_BETA_MAX
    _res_wild = {"valid": True, "snr_db": np.full(_NO, 25.0),
                 "snr_lin": np.full(_NO, 10 ** 2.5),
                 "isi_ratio": _ISI_BETA_MAX + 0.6}
    _bat_wild = _m.allocate(_res_wild)
    _res_none = {"valid": True, "snr_db": np.full(_NO, 25.0),
                 "snr_lin": np.full(_NO, 10 ** 2.5), "isi_ratio": None}
    check(f"isi: unmeasurable beta ({_ISI_BETA_MAX + 0.6:.1f}) is ignored, not "
          f"applied", np.array_equal(_bat_wild, _m.allocate(_res_none)))
    check("isi: a beta just inside the bound is still applied",
          int(_m.allocate({"valid": True, "snr_db": np.full(_NO, 25.0),
                           "snr_lin": np.full(_NO, 10 ** 2.5),
                           "isi_ratio": _ISI_BETA_MAX - 0.05}).sum())
          < int(_bat_wild.sum()))

    # ...but a genuinely dead channel still allocates nothing (the back-off is
    # only allowed to undo the DE-RATING, never the underlying measurement).
    _res_dead = {"valid": True, "snr_db": np.full(_NO, -5.0),
                 "snr_lin": np.full(_NO, 10 ** -0.5), "isi_ratio": 1.25}
    check("isi: back-off does not resurrect a dead channel",
          not _m.allocate(_res_dead).any())
    # and with a sane ceiling nothing is backed off at all
    _res_ok = {"valid": True, "snr_db": np.full(_NO, 25.0),
               "snr_lin": np.full(_NO, 10 ** 2.5), "isi_ratio": 0.2}
    check("isi: a workable ceiling is applied as measured",
          int((_m.allocate(_res_ok) > 0).sum()) == _nd)

    # ---- data geometry must follow the run's overrides ------------------- #
    # Every case above runs with HS_OVERRIDES empty, where hs_cfg and mesh_cfg
    # trivially agree -- which is exactly why the two drifted apart unnoticed.
    # Any run that overrides the band or FFT geometry must still see ONE
    # geometry: the handshake allocates a table of length n_data_carriers and
    # make_link unpacks the same count, or every carrier is loaded from a
    # different frequency's measurement.
    import ofdm_handshake as _hs
    _saved_ov = _hs.HS_OVERRIDES
    try:
        for ov in ({"freq_lo": 300.0, "freq_hi": 6000.0},
                   {"n_fft": 2048, "cp_ratio": 0.25},
                   {"freq_lo": 300.0, "freq_hi": 6000.0, "cp_ratio": 0.5}):
            _hs.HS_OVERRIDES = ov
            hs_n = _hs.hs_cfg(44100)["n_data_carriers"]
            mesh_n = mesh_cfg(44100)["n_data_carriers"]
            check(f"geometry: mesh follows override {ov} "
                  f"({mesh_n} carriers)", hs_n == mesh_n)
            # and the wire table round-trips at that geometry
            rs2 = np.random.RandomState(7)
            bt = rs2.choice([0, 1, 2], size=hs_n).astype(np.int32)
            lk = make_link({"my_tx_bat": bat_to_bytes(bt),
                            "peer_tx_bat": bat_to_bytes(bt),
                            "peer_rate": 44100}, 44100, symmetric=True)
            check(f"geometry: BAT survives the wire at {mesh_n} carriers",
                  np.array_equal(lk["tx_bat"], bt))
    finally:
        _hs.HS_OVERRIDES = _saved_ov

    # A genuine geometry disagreement must be reported, not silently truncated.
    from ofdm_modem import bat_wire_carriers
    wire = bat_to_bytes(np.ones(116, dtype=np.int32))
    lo_w, hi_w = bat_wire_carriers(wire)
    check("geometry: wire carrier count is recoverable",
          lo_w <= 116 <= hi_w and hi_w - lo_w <= 2)
    try:
        bytes_to_bat(wire, 72, strict=True)
        mismatch_raised = False
    except ValueError:
        mismatch_raised = True
    check("geometry: truncating unpack is refused in strict mode",
          mismatch_raised)

    print("SELFTEST", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(_selftest())
