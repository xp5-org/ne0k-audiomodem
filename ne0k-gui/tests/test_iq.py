# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Complex baseband (COMPLEX_BB.md step 8): the iq: connector, the I/Q pipe/session path, the iq
profile, and the complex OFDM modem through it.

The cores themselves are graded by audiotestserver/tests/OFDM/ofdm_cx_selftest.py; here the point
is the plumbing: samples arrive in order and complete through a socket that behaves like the GNU
Radio medium (it is the clock, it buffers TX, it fills a TX gap with zeros), and the modem decodes
what comes out the other side.
"""
import collections
import socket
import threading
import time

import numpy as np
import pytest

from modemgui import native
from modemgui.audio import AudioError, AudioPipe, Registry, StreamSpec
from modemgui.audio.iq_backend import decode, encode, parse_dev
from modemgui.engine.modemcfg import IQ_PROFILES, IqConfig, iq_profile
from modemgui.engine.spectro import Stft


def _cx_or_skip():
    try:
        from modemgui.engine.ofdm_cx import _lib
        _lib()
    except (native.NativeError, OSError) as e:
        pytest.skip(f"libofdm with the ofdm_cx* cores not built: {e}")


# -- connector pieces -------------------------------------------------------------------------

def test_parse_dev():
    assert parse_dev("127.0.0.1:5601") == ("127.0.0.1", 5601, "cf32")
    assert parse_dev("host.lan:7/sc16") == ("host.lan", 7, "sc16")
    assert parse_dev("[::1]:5601") == ("::1", 5601, "cf32")
    for bad in ("5601", "h:x", "h:0", "h:5601/cs8", ":5601"):
        with pytest.raises(AudioError):
            parse_dev(bad)


@pytest.mark.parametrize("fmt", ["cf32", "sc16"])
def test_wire_roundtrip(fmt):
    rng = np.random.default_rng(0)
    x = (0.5 * (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)) / 2).astype(np.complex64)
    x = np.clip(x.real, -0.99, 0.99) + 1j * np.clip(x.imag, -0.99, 0.99)
    b = encode(x, fmt)
    assert len(b) == len(x) * (8 if fmt == "cf32" else 4)
    y = decode(b, fmt)
    assert y.dtype == np.complex64 and len(y) == len(x)
    assert np.max(np.abs(y - x)) <= (0 if fmt == "cf32" else 1 / 32768 * 0.75)
    if fmt == "sc16":                       # full scale clips, never wraps
        z = decode(encode(np.array([1.5 - 2j], np.complex64), fmt), fmt)
        assert z[0].real == pytest.approx(32767 / 32768) and z[0].imag == -1.0


def test_registry_keeps_iq_and_audio_apart(tmp_path):
    reg = Registry(plugins=False)
    assert "iq" in reg.connectors and reg.connectors["iq"].iq
    with pytest.raises(AudioError, match="I/Q"):            # audio session on an iq: device
        reg.open_source("iq:127.0.0.1:1", StreamSpec(48000, 1024))
    wav = tmp_path / "x.wav"
    with pytest.raises(AudioError, match="audio device"):   # I/Q session on a file/sound card
        reg.open_source(f"file:{wav}", StreamSpec(200000, 4000, iq=True))


def test_stft_two_sided():
    fs, f0 = 200000, -31250.0
    t = np.arange(8192) / fs
    st = Stft(fs, 1024, 512, iq=True)
    assert st.freqs[0] == pytest.approx(-fs / 2) and np.all(np.diff(st.freqs) > 0)
    _i, cols = st.feed(np.exp(2j * np.pi * f0 * t).astype(np.complex64), 0)
    k = np.argmax(cols.mean(axis=0))
    assert st.freqs[k] == pytest.approx(f0, abs=fs / 1024)
    assert cols.mean(axis=0)[k] == pytest.approx(0.0, abs=0.5)     # full-scale complex tone = 0 dB


def test_iq_profile():
    c = iq_profile()
    assert (c.fs, c.band_lo, c.band_hi) == (200000, -50000.0, 50000.0)
    assert "iq" in IQ_PROFILES and iq_profile("iq", node_id=7).node_id == 7
    with pytest.raises(KeyError):
        iq_profile("nope")
    assert IqConfig.from_dict(c.to_dict()) == c


# -- a socket that behaves like gr_rf_medium.py, one node, hearing itself ----------------------

class LoopMedium:
    """One node port: emits RX at ``fs`` on the wall clock = the node's own queued TX (else zeros),
    plus AWGN and a carrier offset; a TX gap is filled with zeros and counted, as the medium does."""

    def __init__(self, fs=200000, cfo_hz=0.0, noise=0.0, block=2000):
        self.fs, self.cfo, self.noise, self.block = fs, cfo_hz, noise, block
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.underrun = self.started = 0
        self.rng = np.random.default_rng(1)
        threading.Thread(target=self._serve, daemon=True).start()

    def _reader(self, c):
        while not self.stop.is_set():
            try:
                d = c.recv(1 << 16)
            except OSError:
                return
            if not d:
                return
            with self.lock:
                self.buf += d
                self.started = 1

    def _serve(self):
        self.srv.settimeout(0.2)
        while not self.stop.is_set():
            try:
                c, _ = self.srv.accept()
                break
            except socket.timeout:
                continue
        else:
            return
        threading.Thread(target=self._reader, args=(c,), daemon=True).start()
        t0, made = time.monotonic(), 0
        while not self.stop.is_set():
            due = int((time.monotonic() - t0) * self.fs) - made
            if due < self.block:
                time.sleep(0.002)
                continue
            n = self.block
            with self.lock:
                k = min(n, len(self.buf) // 8)
                x = np.frombuffer(bytes(self.buf[:8 * k]), "<c8").astype(np.complex128)
                del self.buf[:8 * k]
                if self.started:
                    self.underrun += n - k
            y = np.zeros(n, complex)
            y[:k] = x
            y *= np.exp(2j * np.pi * self.cfo * (made + np.arange(n)) / self.fs)
            if self.noise:
                y += self.noise * (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)) / np.sqrt(2)
            made += n
            try:
                c.sendall(y.astype("<c8").tobytes())
            except OSError:
                return

    def close(self):
        self.stop.set()
        self.srv.close()


def test_pipe_over_iq_socket_is_complete_and_ordered():
    med = LoopMedium()
    try:
        pipe = AudioPipe(Registry(plugins=False), StreamSpec(200000, 2000, iq=True))
        addr = f"iq:127.0.0.1:{med.port}"
        pipe.open(addr, addr)
        tap = pipe.subscribe(4096)
        pipe.start()
        time.sleep(0.3)
        ramp = (np.arange(30000) % 1000 / 1000.0 * np.exp(1j * 0.3)).astype(np.complex64) + 0.001
        pipe.play(ramp)
        assert pipe.wait_tx(5)
        time.sleep(0.6)
        x, i0 = pipe.history()
        idx = [b.index for b in tap.drain()]
        pipe.close()
    finally:
        med.close()
    assert x.dtype == np.complex64
    assert all(b - a == 2000 for a, b in zip(idx, idx[1:]))        # contiguous
    on = np.flatnonzero(np.abs(x) > 0.0005)
    got = x[on[0]:on[0] + len(ramp)]
    assert np.array_equal(got, ramp)                              # every sample, in order, bit-exact
    assert med.underrun == 0


# -- the modem through it ---------------------------------------------------------------------

def test_iq_link_message_over_socket_with_cfo():
    _cx_or_skip()
    from modemgui.engine.iq_link import IqLink, open_iq_session
    cfg = iq_profile(block=2000)
    med = LoopMedium(cfo_hz=-6100.0, noise=0.2 * 10 ** (-25 / 20))
    try:
        sess = open_iq_session(cfg, f"iq:127.0.0.1:{med.port}")
        link = IqLink(sess, cfg, hear_own=True).start()
        time.sleep(0.3)
        msg = np.random.default_rng(3).bytes(3000)
        s0, nf = link.send_message(msg, seq0=65530)               # sequence numbers wrap
        assert nf == link.mod.frames_for(8 + len(msg) + 4)
        data, why = link.receive_message(10.0, seq0=65530)
        link.stop(); sess.close()
    finally:
        med.close()
    assert why == "" and data == msg
    good = [f for f in link.frames if f.ok]
    assert len(good) == nf
    assert np.median([f.cfo_hz for f in good]) == pytest.approx(-6100.0, abs=15.0)


def test_probe_reads_the_frames_snr_and_drives_the_bat():
    """At the link's one TX gain the probe SNR is the data's SNR (not 3 dB above it), and the bit
    table it yields carries frames through the same channel."""
    _cx_or_skip()
    from modemgui.engine.ofdm_cx import CxModem, CxProber, CxReceiver, tx_gain
    cfg = iq_profile()
    g = tx_gain(cfg)
    rng = np.random.default_rng(5)
    snr_in_band = 24.0                       # dB per carrier: noise per bin vs a unit-power carrier
    m0 = CxModem(cfg)
    occ = m0.x.c.n_occupied / m0.x.c.n_fft
    nv = cfg.tx_rms ** 2 / occ * 10 ** (-snr_in_band / 10)  # per-sample noise power for that

    def chan(x, cfo=3300.0):
        y = np.concatenate([np.zeros(7000), x * g, np.zeros(9000)])
        y = y * np.exp(2j * np.pi * cfo * np.arange(len(y)) / cfg.fs)
        return y + np.sqrt(nv / 2) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))

    p = CxProber(cfg)
    res = p.estimate(chan(p.build()))
    assert res.ok, res.error
    assert np.median(res.snr_db) == pytest.approx(snr_in_band, abs=1.0)
    assert res.cfo_hz == pytest.approx(3300.0, abs=10.0)
    assert sum(res.bat) > sum(m0.bat)        # better than the default QPSK at 24 dB
    data = rng.bytes(2000)
    m, r = CxModem(cfg, res.bat), CxReceiver(cfg, res.bat)
    y = chan(m.encode(data))
    got = []
    for i in range(0, len(y), 4000):
        got += r.feed(y[i:i + 4000], i)
    got += r.flush()
    assert [f.seq for f in got] == list(range(m.frames_for(len(data))))
    assert all(f.ok for f in got)
    assert b"".join(f.payload for f in got)[:len(data)] == data


def test_iterate_rescues_frames_on_a_fast_selective_fade():
    """Decoder-aided re-estimation (IqConfig.iterate): an echo with 6 Hz Doppler of its own makes
    the channel move between SYNCs; the CE/SYNC + pilot estimate loses every frame, the re-encoded
    second pass decodes them all, with the right payload."""
    _cx_or_skip()
    from modemgui.engine import ofdm_cx
    from modemgui.engine.ofdm_cx import CxModem, CxReceiver
    if not hasattr(ofdm_cx._lib(), "ofdm_cxdem_set_iterate"):
        pytest.skip("libofdm without ofdm_cxdem_set_iterate (rebuild)")
    cfg = iq_profile()
    bat = [4] * CxModem(cfg).x.c.n_data_carriers
    m = CxModem(cfg, bat)
    rng = np.random.default_rng(3)
    data = rng.bytes(cfg.chunk_bytes * 6)
    x = np.concatenate([np.zeros(3000), m.encode(data) * 0.2, np.zeros(20000)])
    t = np.arange(len(x)) / cfg.fs
    y = x + np.concatenate([np.zeros(20), x[:-20]]) * 0.5 * np.exp(2j * np.pi * 6.0 * t)
    y = y + 0.2 * 10 ** (-28 / 20) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y))) / np.sqrt(2)

    def run(it):
        r = CxReceiver(iq_profile(iterate=it, iterate_win=4), bat)
        got = []
        for i in range(0, len(y), 4000):
            got += r.feed(y[i:i + 4000], i)
        return [f for f in got + r.flush() if f.ok]

    n = m.frames_for(len(data))
    assert len(run(0)) < n
    ok = run(3)
    assert sorted(f.seq for f in ok) == list(range(n))
    assert b"".join(f.payload for f in sorted(ok, key=lambda f: f.seq))[:len(data)] == data


def test_taper_past_cp_quarter_is_refused():
    _cx_or_skip()
    from modemgui.engine.ofdm_cx import build_cfg
    with pytest.raises(native.NativeError, match="CP/4"):
        build_cfg(iq_profile(taper_ms=0.16))


# -- step 9: two tables in one stream, and the probe -> BAT exchange between two nodes -----------

def test_receiver_decodes_base_and_probed_tables_in_one_stream():
    """Control frames at the base table and data at a probed table, back to back in one stream:
    one receiver holding both tables gets every frame and says which table took it."""
    _cx_or_skip()
    from modemgui.engine.ofdm_cx import CxModem, CxReceiver, table_id
    cfg = iq_profile()
    base = CxModem(cfg)
    hi = [6 if i % 3 else 4 for i in range(base.x.c.n_data_carriers)]
    m = CxModem(cfg, hi)
    rng = np.random.default_rng(11)
    a, b, c = rng.bytes(300), rng.bytes(5000), rng.bytes(300)
    y = np.concatenate([base.encode(a, seq0=0), m.encode(b, seq0=1), base.encode(c, seq0=9)]) * 0.2
    y = y + 0.2 * 10 ** (-30 / 20) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
    r = CxReceiver(cfg, hi, tables=[base.bat])
    got = []
    for i in range(0, len(y), 4000):
        got += r.feed(y[i:i + 4000], i)
    got += r.flush()
    assert all(f.ok for f in got)
    assert [f.seq for f in got] == [0, 1, 2, 3, 9]
    tid = [f.table for f in got]
    assert tid == [table_id(base.bat)] + [table_id(hi)] * 3 + [table_id(base.bat)]
    assert b"".join(f.payload for f in got[1:4])[:len(b)] == b


class PairMedium:
    """Two node ports, mix-minus like gr_rf_medium.py: each node hears the other's TX (else
    zeros) plus AWGN, with its own carrier offset; the medium is the clock for both.  ``delay_s``:
    one-way path delay, both directions (zeros first, like gr_rf_medium's delay line)."""

    def __init__(self, fs=200000, cfo_hz=(0.0, 0.0), noise=0.0, block=2000, delay_s=0.0):
        self.fs, self.cfo, self.noise, self.block = fs, cfo_hz, noise, block
        self.dline = [collections.deque([np.zeros(block, complex)] * int(round(delay_s * fs / block)))
                      for _ in range(2)]
        self.srv = []
        for _ in range(2):
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            s.settimeout(0.2)
            self.srv.append(s)
        self.ports = [s.getsockname()[1] for s in self.srv]
        self.buf = [bytearray(), bytearray()]
        self.conn = [None, None]
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.rng = np.random.default_rng(2)
        threading.Thread(target=self._serve, daemon=True).start()

    def _reader(self, k, c):
        while not self.stop.is_set():
            try:
                d = c.recv(1 << 16)
            except OSError:
                return
            if not d:
                return
            with self.lock:
                self.buf[k] += d

    def _serve(self):
        while not self.stop.is_set() and None in self.conn:
            for k, s in enumerate(self.srv):
                if self.conn[k] is None:
                    try:
                        self.conn[k], _ = s.accept()
                        threading.Thread(target=self._reader, args=(k, self.conn[k]), daemon=True).start()
                    except socket.timeout:
                        pass
        t0, made = time.monotonic(), 0
        while not self.stop.is_set():
            if int((time.monotonic() - t0) * self.fs) - made < self.block:
                time.sleep(0.002)
                continue
            n = self.block
            tx = []
            with self.lock:
                for k in range(2):
                    m = min(n, len(self.buf[k]) // 8)
                    x = np.zeros(n, complex)
                    x[:m] = np.frombuffer(bytes(self.buf[k][:8 * m]), "<c8")
                    del self.buf[k][:8 * m]
                    if self.dline[k]:
                        self.dline[k].append(x)
                        x = self.dline[k].popleft()
                    tx.append(x)
            t = (made + np.arange(n)) / self.fs
            for k in range(2):
                y = tx[1 - k] * np.exp(2j * np.pi * self.cfo[k] * t)
                if self.noise:
                    y = y + self.noise * (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)) / np.sqrt(2)
                try:
                    self.conn[k].sendall(y.astype("<c8").tobytes())
                except OSError:
                    return
            made += n

    def close(self):
        self.stop.set()
        for s in self.srv:
            s.close()


def test_negotiate_then_a_message_each_way_over_a_pair_medium():
    """Both nodes on one mix-minus medium: the probe -> BAT exchange settles a table per
    direction above the base QPSK, then a message goes each way at those tables."""
    _cx_or_skip()
    from modemgui.engine.iq_link import IqLink, MSG_MAGIC, open_iq_session
    from modemgui.engine.ofdm_cx import table_id
    cfg = [iq_profile(block=2000, node_id=k, peer_id=1 - k) for k in range(2)]
    med = PairMedium(cfo_hz=(2100.0, -2100.0), noise=0.2 * 10 ** (-28 / 20))
    res = [None, None]
    try:
        sess = [open_iq_session(cfg[k], f"iq:127.0.0.1:{med.ports[k]}") for k in range(2)]
        link = [IqLink(sess[k], cfg[k]).start() for k in range(2)]
        time.sleep(0.3)

        def run(k):
            res[k] = link[k].negotiate(initiator=k == 0, timeout=30, retry_s=2.0)
        th = [threading.Thread(target=run, args=(k,)) for k in range(2)]
        for t in th:
            t.start()
        for t in th:
            t.join(40)
        msg = [np.random.default_rng(20 + k).bytes(6000) for k in range(2)]
        since = res[1]["since"]
        link[0].send_message(msg[0])
        got1, why1, f1 = link[1].receive_msg(MSG_MAGIC, 15, since=since)
        since = link[0].rx_now()
        link[1].send_message(msg[1])
        got0, why0, f0 = link[0].receive_msg(MSG_MAGIC, 15, since=since)
        for k in range(2):
            link[k].stop(); sess[k].close()
    finally:
        med.close()
    for k in range(2):
        assert res[k]["ok"], res[k]
        assert res[k]["tx_bits"] > sum(link[k].base.bat)     # better than QPSK at 28 dB
    assert res[0]["tx_bat"] == res[1]["rx_bat"] and res[1]["tx_bat"] == res[0]["rx_bat"]
    assert res[0]["settled"]
    assert why1 == "" and got1 == msg[0] and f1.table == table_id(res[0]["tx_bat"])
    assert why0 == "" and got0 == msg[1] and f0.table == table_id(res[1]["tx_bat"])


def test_negotiate_measures_a_long_round_trip_and_still_settles():
    """2.5 s each way (a 5 s+ round trip, past the 2 s first retry): the probe retries back off
    instead of piling up, the PING/BAT echo measures the round trip, both ends settle and data
    still flows.  The RTD is the whole protocol loop: 2 x 2.5 s path plus both modems' buffering."""
    _cx_or_skip()
    from modemgui.engine.iq_link import IqLink, MSG_MAGIC, open_iq_session
    cfg = [iq_profile(block=2000, node_id=k, peer_id=1 - k, max_rtd_s=10.0) for k in range(2)]
    med = PairMedium(noise=0.2 * 10 ** (-28 / 20), delay_s=2.5)
    res = [None, None]
    try:
        sess = [open_iq_session(cfg[k], f"iq:127.0.0.1:{med.ports[k]}") for k in range(2)]
        link = [IqLink(sess[k], cfg[k]).start() for k in range(2)]
        time.sleep(0.3)

        def run(k):
            res[k] = link[k].negotiate(initiator=k == 0, retry_s=2.0)
        th = [threading.Thread(target=run, args=(k,)) for k in range(2)]
        for t in th:
            t.start()
        for t in th:
            t.join(80)
        msg = np.random.default_rng(30).bytes(4000)
        link[0].send_message(msg)
        got, why, _f = link[1].receive_msg(MSG_MAGIC, 20, since=res[1]["since"])
        for k in range(2):
            link[k].stop(); sess[k].close()
    finally:
        med.close()
    for k in range(2):
        assert res[k]["ok"], res[k]
        assert 5.0 <= res[k]["rtd_s"] <= 8.0, res[k]["rtd_s"]
    assert res[0]["counts"]["tx_probes"] <= 2, res[0]["counts"]       # backed off, did not pile up
    assert why == "" and got == msg


# -- erasure decoding and selective-repeat ARQ (2026-11-14) ------------------------------------

def _blocked_frames(cfg, bat, dead_idx, nfr=12, snr_db=32.0):
    """nfr frames at ``bat`` with the data carriers ``dead_idx`` blocked 60 dB, AWGN at snr_db,
    through a CxReceiver; returns the frames."""
    from modemgui.engine.ofdm_cx import CxModem, CxReceiver, bins, tx_gain
    from modemgui.engine import slotimpair as si
    mod = CxModem(cfg, bat)
    data = np.random.default_rng(1).bytes(mod.payload_bytes * nfr)
    x = mod.encode(data, seq0=0) * tx_gain(cfg)
    x = np.concatenate([np.zeros(4000, complex), x, np.zeros(20000, complex)])
    df = cfg.fs / mod.x.c.n_fft
    occ, sp = bins(mod.x), mod.x.c.pilot_spacing
    dat = [b * df for i, b in enumerate(occ) if i % sp]
    x = si.apply(x, cfg.fs, [dat[i] for i in dead_idx], df, "block", 60.0, np.random.default_rng(2))
    rng = np.random.default_rng(3)
    x = x + 0.2 * 10 ** (-snr_db / 20) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))) / np.sqrt(2)
    rx, got = CxReceiver(cfg, bat), []
    for i in range(0, len(x), 4000):
        got += rx.feed(x[i:i + 4000], i)
    return got + rx.flush()


def test_erasure_decoding_rescues_frames_errors_only_loses():
    """Two dead 64-QAM carriers: errors-only RS loses frames, the GMD retry with the least reliable
    bytes erased decodes every one, and reports the erasures it used.  Per-carrier quality marks
    exactly the dead carriers by |H|."""
    _cx_or_skip()
    import dataclasses
    from modemgui.engine.ofdm_cx import CxModem
    from modemgui.engine.iq_link import DEAD_HREL
    cfg = iq_profile(fec="none")                     # the RS-only frame: the inner code would hide the loss
    nd = CxModem(cfg).x.c.n_data_carriers
    bat = [6 if i % 7 else 4 for i in range(nd)]
    dead = [1, 2]
    off = _blocked_frames(dataclasses.replace(cfg, erasures=False), bat, dead)
    on = _blocked_frames(cfg, bat, dead)
    assert sum(f.ok for f in off) < 12
    assert sum(f.ok for f in on) == 12 and sum(f.erased for f in on) > 0
    hr = np.median([f.quality[1] for f in on], axis=0)
    assert [i for i, h in enumerate(hr) if 0 <= h < DEAD_HREL] == dead


def _arq_pair(spec: str, nbytes: int = 20000, snr_db: float = 32.0, **over):
    from modemgui.engine.iq_link import IqLink, open_iq_session
    from modemgui.engine.slotimpair import attach_iq
    cfg = [iq_profile(block=2000, node_id=k, peer_id=1 - k, **over) for k in range(2)]
    med = PairMedium(cfo_hz=(2100.0, -2100.0), noise=0.2 * 10 ** (-snr_db / 20))
    try:
        sess = [open_iq_session(cfg[k], f"iq:127.0.0.1:{med.ports[k]}") for k in range(2)]
        link = [IqLink(sess[k], cfg[k]).start() for k in range(2)]
        gate = attach_iq(link[0], spec, seed=3)
        time.sleep(0.3)
        res = [None, None]

        def run(k):
            res[k] = link[k].negotiate(initiator=k == 0, timeout=30, retry_s=2.0)
        th = [threading.Thread(target=run, args=(k,)) for k in range(2)]
        for t in th:
            t.start()
        for t in th:
            t.join(40)
        msg = np.random.default_rng(5).bytes(nbytes)
        out = link[0].send_reliable(msg)
        got = link[1].receive_reliable(5)
        for k in range(2):
            link[k].stop(); sess[k].close()
    finally:
        med.close()
    assert res[0]["ok"] and res[1]["ok"], res
    return out, got == msg, gate.report()


def test_arq_clean_needs_one_poll():
    _cx_or_skip()
    out, ok, _rep = _arq_pair("none")
    assert ok and out["ok"] and out["rounds"] == 1 and out["resent"] == 0 and out["table_updates"] == 0


def test_arq_repeated_control_frames_are_acted_on_once():
    """ctl_repeat_s sends every POLL / STAT twice (same seq); the copies are dropped, so a clean transfer
    still takes one POLL and nothing is resent."""
    _cx_or_skip()
    out, ok, _rep = _arq_pair("none", ctl_repeat_s=0.6)
    assert ok and out["ok"] and out["rounds"] == 1 and out["resent"] == 0


def test_late_bat_after_negotiate_gets_a_ready():
    """The peer that finished negotiating still answers a re-sent BAT (its READY was lost and it had stopped
    lingering) with another READY, so the other end can finish: hf-moderate's initiator failed for this."""
    _cx_or_skip()
    from modemgui.engine.iq_link import IqLink, open_iq_session, READY_MAGIC
    cfg = [iq_profile(block=2000, node_id=k, peer_id=1 - k) for k in range(2)]
    med = PairMedium(cfo_hz=(0.0, 0.0), noise=0.2 * 10 ** (-32 / 20))
    try:
        sess = [open_iq_session(cfg[k], f"iq:127.0.0.1:{med.ports[k]}") for k in range(2)]
        link = [IqLink(sess[k], cfg[k]).start() for k in range(2)]
        time.sleep(0.3)
        res = [None, None]

        def run(k):
            res[k] = link[k].negotiate(initiator=k == 0, timeout=30, retry_s=2.0)
        th = [threading.Thread(target=run, args=(k,)) for k in range(2)]
        for t in th:
            t.start()
        for t in th:
            t.join(40)
        assert res[0]["ok"] and res[1]["ok"], res
        since = link[0].rx_now()
        link[0]._send_bat(res[0]["rx_bat"], True, None)
        _d, _why, f = link[0].receive_msg(READY_MAGIC, 8.0, since=since)
        late = link[1].stats["tx_ready_late"]
        for k in range(2):
            link[k].stop(); sess[k].close()
    finally:
        med.close()
    assert late == 1 and f is not None


def test_negotiate_sends_a_table_improved_in_the_ready_pass():
    """vhf-mobile 2026-11-15: the responder's first probe was in a fade, its second
    probe improved the table in the same pass the peer's BAT{have} arrived; READY went first and the
    better table was never sent, so the initiator stayed at the faded table.  Forced here: node 1's
    first probe is cut to a weak table, its second is held until node 0's BAT{have} is decodable."""
    _cx_or_skip()
    import dataclasses
    from modemgui.engine.iq_link import IqLink, open_iq_session, BAT_MAGIC
    cfg = [iq_profile(block=2000, node_id=k, peer_id=1 - k) for k in range(2)]
    med = PairMedium(cfo_hz=(0.0, 0.0), noise=0.2 * 10 ** (-30 / 20))
    try:
        sess = [open_iq_session(cfg[k], f"iq:127.0.0.1:{med.ports[k]}") for k in range(2)]
        link = [IqLink(sess[k], cfg[k]).start() for k in range(2)]
        time.sleep(0.3)
        real, n_probe, good, start = link[1].poll_probe, [0], [], link[1].rx_now()

        def peer_bat_have() -> bool:
            for f in list(link[1].frames):
                if f.ok and f.payload[:4] == BAT_MAGIC and f.index >= start:
                    try:
                        ln = int.from_bytes(f.payload[4:8], "big")
                        if link[1]._parse_bat(f.payload[8:8 + ln])[1]:
                            return True
                    except ValueError:
                        pass
            return False

        def poll(since):
            got = real(since)
            if not got:
                return got
            if n_probe[0] == 0:                       # faded: 2 bits on carrier 0, nothing else
                n_probe[0] = 1
                weak = [0] * len(got[0].bat)
                weak[0] = 2
                return dataclasses.replace(got[0], bat=weak), got[1]
            if n_probe[0] == 1 and not peer_bat_have():
                return None                           # hold the good probe for the peer's BAT pass
            if n_probe[0] == 1:
                n_probe[0], good[:] = 2, list(got[0].bat)
            # later probes add nothing: only the BAT sent before
            # READY can carry it, not the after-done resend on an improvement
            return dataclasses.replace(got[0], bat=list(good)), got[1]
        link[1].poll_probe = poll
        res = [None, None]

        def run(k):
            res[k] = link[k].negotiate(initiator=k == 0, timeout=40, retry_s=2.0)
        th = [threading.Thread(target=run, args=(k,)) for k in range(2)]
        for t in th:
            t.start()
        for t in th:
            t.join(50)
        for k in range(2):
            link[k].stop(); sess[k].close()
    finally:
        med.close()
    assert n_probe[0] == 2, "the race was not set up"
    assert res[0]["ok"] and res[1]["ok"], res
    assert res[0]["tx_bat"] == res[1]["rx_bat"], (res[0]["tx_bits"], res[1]["rx_bits"])
    assert res[0]["tx_bits"] > sum(link[0].base.bat)


def test_arq_table_update_routes_around_pinned_dead_carriers():
    """Eight dead hot carriers that stay put (:pin), RS-only frame: past the erasure budget, so frames are lost;
    with ARQ the receiver's STAT carries a table with those carriers at 0 bits, the missing frames go
    again at it, and the message arrives whole.  The gate shows no bits left on the dead slots."""
    _cx_or_skip()
    out, ok, rep = _arq_pair("hot:8:block:data:pin", fec="none", reprobe_s=0.0)
    assert ok and out["ok"], out
    assert out["table_updates"] >= 1 and out["resent"] >= 1
    assert rep["picked"][0]["rules"][0]["bits_hit"] > 0 and rep["picked"][-1]["rules"][0]["bits_hit"] == 0


def test_inner_fec_decodes_where_the_rs_only_frame_fails():
    """64-QAM at 18 dB: the RS-only frame loses every frame, the rate-2/3 soft-Viterbi frame none."""
    _cx_or_skip()
    from modemgui.engine.ofdm_cx import CxModem
    for fec, want in (("none", 0), ("2/3", 12)):
        cfg = iq_profile(fec=fec)
        nd = CxModem(cfg).x.c.n_data_carriers
        got = _blocked_frames(cfg, [6] * nd, [], snr_db=18.0)
        assert sum(f.ok for f in got) == want, (fec, [f.ok for f in got])


def test_cw_station_id_keys_the_callsign():
    from modemgui.engine.iq_link import cw_iq
    x = cw_iq("E", 200000.0, 20.0, 25000.0, 0.2)           # one dot: 60 ms keyed
    assert abs(len(x) - 12000) <= 1 and abs(np.max(np.abs(x)) - 0.2) < 1e-9
    f = np.fft.fftfreq(len(x), 1 / 200000.0)[np.argmax(np.abs(np.fft.fft(x)))]
    assert abs(f - 25000.0) < 50


def test_tone_exciser_notches_a_cw_and_leaves_the_ofdm():
    """A CW at -6 dB under the frames: found and notched; the frames still decode and the probe
    grades the channel instead of reporting ~-5 dB everywhere (the step 10b idle-cell failure)."""
    _cx_or_skip()
    from modemgui.engine.ofdm_cx import CxModem, CxProber, ToneExciser, tx_gain
    cfg = iq_profile()
    ex = ToneExciser(cfg.fs, 4000)
    pr = CxProber(cfg)
    p = pr.build() * tx_gain(cfg)
    x = np.concatenate([np.zeros(8000, complex), p, np.zeros(8000, complex)])
    x = np.tile(x, 4)
    n = np.arange(len(x))
    rng = np.random.default_rng(1)
    x = x + 0.1 * np.exp(2j * np.pi * 19531.25 * n / cfg.fs) + 0.2 * 10 ** (-30 / 20) * (
        rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))) / np.sqrt(2)
    for i in range(0, len(x), 4000):
        ex.process(x[i:i + 4000])
    assert any(abs(h - 19531.25) < 400 for h in ex.notches_hz), ex.notches_hz
    tail = x[-(pr.samples + 16000):]
    assert not pr.estimate(tail).ok or sum(pr.estimate(tail).bat) == 0      # unfiltered: wrecked
    res = pr.estimate(ex.filter_snippet(tail))
    assert res.ok and sum(res.bat) > 200, res.summary()


def test_short_control_frames_ride_beside_data_frames():
    """Control messages go in ctl_chunk_bytes frames; one receiver hears them and full data frames."""
    import dataclasses
    from modemgui.engine.ofdm_cx import CxModem, CxReceiver
    cfg = iq_profile()
    ctl = CxModem(dataclasses.replace(cfg, chunk_bytes=cfg.ctl_chunk_bytes))
    nd = ctl.x.c.n_data_carriers
    dat = CxModem(cfg, [4 if i % 2 else 2 for i in range(nd)])
    assert ctl.payload_bytes == cfg.ctl_chunk_bytes < dat.payload_bytes
    assert ctl.frame_samples < dat.frame_samples / 2
    gap = np.zeros(4000, complex)
    x = np.concatenate([gap, ctl.encode(b"poll", seq0=1), gap, dat.encode(b"d" * 3000, seq0=2), gap,
                        ctl.encode(b"stat", seq0=4), gap])
    rx = CxReceiver(cfg, dat.bat)
    rx.set_ctl(ctl)
    assert rx.tables == [dat.bat]
    got = []
    for i in range(0, len(x), 3000):
        got += rx.feed(x[i:i + 3000], i)
    got += rx.flush()
    ok = {f.seq: f for f in got if f.ok}
    assert sorted(ok) == [1, 2, 3, 4]
    assert ok[1].payload.startswith(b"poll") and len(ok[1].payload) == cfg.ctl_chunk_bytes
    assert ok[4].payload.startswith(b"stat")
    assert ok[2].payload == b"d" * dat.payload_bytes


def test_thin_table_does_not_delay_short_frames():
    """A receiver also holding a thin ARQ table (frames many seconds long) still hears a short POLL as
    soon as the POLL itself is in: hf-moderate's responder went deaf with an 8-bit/sym table kept."""
    import dataclasses
    from modemgui.engine.ofdm_cx import CxModem, CxReceiver
    cfg = iq_profile()
    ctl = CxModem(dataclasses.replace(cfg, chunk_bytes=cfg.ctl_chunk_bytes))
    nd = ctl.x.c.n_data_carriers
    thin = CxModem(cfg, [2 if i < 4 else 0 for i in range(nd)])
    assert thin.frame_samples > 20 * ctl.frame_samples
    gap = np.zeros(4000, complex)
    x = np.concatenate([gap, ctl.encode(b"poll", seq0=7), np.zeros(ctl.frame_samples, complex)])
    rx = CxReceiver(cfg, thin.bat)
    rx.set_ctl(ctl)
    got = []
    for i in range(0, len(x), 2000):
        got += rx.feed(x[i:i + 2000], i)
    assert [f.seq for f in got if f.ok] == [7]          # without flush: within one frame's time
