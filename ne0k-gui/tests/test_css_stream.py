"""CSS streaming receive (decoder "pll_c": libcsspll + chirp tracker) on the two
mesh-conference losses of 2026-11-08 (css-ofdm-audiomodem.mesh-grp-css-ham-5min):

* a frame that had the air to itself, starting ~5 s after our own TX + CW ID, was
  never synced ("grid locked ... lost after 133 chirps", no PLL sync): the
  pre-sync matched-filter window covered only half a symbol period, so for some
  onset phases the conference (G.711) chirps sat outside it and the acquisition
  loop random-walked.  Fixture: the captured ACK (node B rx, group a194),
  stored at the conference's 8 kHz.
* a +1026 sample (21.4 ms) jitter-buffer insert mid-frame: the settled loop's
  +-post_acq_search window lost the chirp and the frame failed its CRC
  (RETRO_GATE, FRAME_ERROR).  Synthesised: an encoded ACK through a G.711 leg
  with 1026 zeros inserted.

Each case is also run with the fix switched off (decode() cfg overrides), to
prove the input still reproduces the loss.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

FS = 48000
BLK = 1024
SPS_EFF = 7560                  # 0.15 s + 5 % gap at 48 kHz
DATA = Path(__file__).with_name("data")
OLD_ACQ = {"presync_full_period": False, "onset_guard_silence_syms": 0}
OLD_STEP = {"step_recover": False}


def _cfg(node, crc_bytes=4):
    from modemgui.engine.modemcfg import CssConfig
    # the mesh test's CSS settings (settings.json of the failing groups, sent with CRC-32 then, as
    # the captured fixtures were; the short-frame tests below use today's CRC-16 default)
    return CssConfig(node_id=node, peer_id=15, fs=FS, f_min=400.0, f_max=3500.0,
                     symbol_dur=0.15, symbol_gap_frac=0.05, amplitude=0.6, chunk_size=8,
                     crc_bytes=crc_bytes,
                     diff_encode=True, rs_t=0, interleave_rows=0, conv_code="none",
                     backend="c", decoder="pll_c")


@pytest.fixture(scope="module")
def css():
    from modemgui.engine import css_link as cl
    try:
        cl.set_backend("c")
        import css_pll_c
        css_pll_c.load(False)
    except Exception as e:                       # libraries not built on this host
        pytest.skip(f"C CSS libraries unavailable: {e}")
    _, cm, _ = cl._imports()
    info = json.loads(Path(cl._info(_cfg(2))).read_text())
    return cl, cm, info


def _rx(css, dec=None):
    from modemgui.engine.css_stream import CssStreamRx
    from modemgui.engine.events import Decoded
    cl, cm, info = css
    out, logs = [], []
    rx = CssStreamRx(_cfg(2), info, node_id=2, hear_own=False, log=logs.append,
                     emit=out.append, emit_link=lambda *a: None, decoded_cls=Decoded,
                     unpack=cm.unpack_packet, dec=dec)
    return rx, out, logs


def _feed(rx, x, tx_active=False):
    for i in range(0, len(x), BLK):
        rx.feed(np.asarray(x[i:i + BLK], np.float32), tx_active)


def _lp(n=191, fc=3600.0):
    t = np.arange(n) - (n - 1) / 2
    h = np.sinc(2 * fc / FS * t) * np.hamming(n)
    return h / h.sum()


def _up8k(y):
    up = np.zeros(len(y) * 6)
    up[::6] = y * 6
    return np.convolve(up, _lp(), mode="same").astype(np.float32)


def _g711(x):
    """48 kHz -> 8 kHz mu-law -> 48 kHz: the SIP conference leg."""
    y = np.convolve(np.asarray(x, np.float64), _lp(), mode="same")[::6]
    mu = 255.0
    c = np.round(np.sign(y) * np.log1p(mu * np.minimum(np.abs(y), 1.0)) / np.log1p(mu) * 127) / 127
    return _up8k(np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu)


def _conference_ack():
    with wave.open(str(DATA / "css_conf_ack_8k.wav"), "rb") as w:
        y = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32767
    return _up8k(y)


def _after_own_tx(css, phase, dec=None):
    """Our DATA + CW ID (decoder blanked by tx_active, tracker hears it), ~5 s of
    conference silence, then the captured peer ACK with its first sample on read
    phase ``phase`` of the decoder's symbol grid."""
    from modemgui.engine import ident
    cl, _, _ = css
    own = np.concatenate([cl.encode_bytes(_cfg(2), b"hello", 1, 3),
                          np.zeros(int(1.5 * FS), np.float32),
                          ident.cw_audio(ident.id_text("HEJ5W"), FS, 15.0, 800.0, 0.3)])
    own = np.concatenate([own, np.zeros(int(0.3 * FS), np.float32)])   # pipe tx guard
    gap = int(4.7 * FS)
    gap += (phase - len(own) - gap) % SPS_EFF
    rx, out, logs = _rx(css, dec)
    rx.tx_onset(0, own)
    _feed(rx, own, tx_active=True)
    _feed(rx, np.zeros(gap, np.float32))
    _feed(rx, _conference_ack())
    _feed(rx, np.zeros(2 * FS, np.float32))
    rx.burst_end()
    return out, logs


# onset phases at which the captured frame was lost before the fix
@pytest.mark.parametrize("phase", [630, 1134])
def test_conference_frame_after_own_tx_decodes(css, phase):
    out, logs = _after_own_tx(css, phase)
    assert [(d.src, d.type_name, d.seq) for d in out] == [(3, "ACK", 1)], logs


def test_conference_frame_after_own_tx_was_lost_without_full_period_search(css):
    out, logs = _after_own_tx(css, 630, dec=OLD_ACQ)
    assert out == []
    assert not any("sync acquired" in m for m in logs)          # the live signature
    assert any("locked" in m for m in logs)                     # the tracker saw it


def _slipped_ack(css, at_s, n, impulse_frac=0.0, seed=0):
    cl, _, _ = css
    c = _cfg(3)
    c.short_frames = False          # the full-size frame the live slip hit (a short ACK has its own test)
    x = _g711(np.concatenate([np.zeros(FS // 2, np.float32), cl.encode_ack(c, 2, 1)]))
    i = int(at_s * FS)
    x = np.concatenate([x[:i], np.zeros(n, np.float32), x[i:]]) if n > 0 else \
        np.concatenate([x[:i], x[i - n:]])
    x = np.concatenate([np.zeros(630, np.float32), x, np.zeros(2 * FS, np.float32)])
    if impulse_frac:
        # SIP outage: ~impulse_frac of the 20 ms packets carry a noise burst
        r = np.random.default_rng(seed)
        for k in range(0, len(x) - 960, 960):
            if r.random() < impulse_frac:
                x[k:k + 960] += (r.standard_normal(960) * r.uniform(0.1, 0.6)).astype(np.float32)
        x = np.clip(x, -1, 1)
    return x


def _decode(css, x, dec=None):
    rx, out, logs = _rx(css, dec)
    _feed(rx, x)
    rx.burst_end()
    return out, logs


# 12.33 s: settled tracking (the live case, ~6 s after sync); 8.05 s: inside the
# post-sync acquisition window; -1026: a dropped packet instead of an insert
@pytest.mark.parametrize("at_s,n", [(12.33, 1026), (8.05, 1026), (15.0, -1026)])
def test_mid_frame_timing_slip_decodes(css, at_s, n):
    out, logs = _decode(css, _slipped_ack(css, at_s, n))
    assert [(d.src, d.type_name) for d in out] == [(3, "ACK")], logs
    assert any("PHASE_STEP" in m and "confirmed" in m for m in logs), logs


def test_mid_frame_timing_slip_was_lost_without_step_recovery(css):
    out, _ = _decode(css, _slipped_ack(css, 12.33, 1026), dec=OLD_STEP)
    assert out == []


def test_timing_slip_under_impulse_noise_decodes(css):
    out, logs = _decode(css, _slipped_ack(css, 11.0, 1026, impulse_frac=0.10, seed=3))
    assert [(d.src, d.type_name) for d in out] == [(3, "ACK")], logs


# -- short header-only frames (CssConfig.short_frames) ---------------------------------------------

class _Sess:
    def __init__(self):
        self.got = []

    def _emit(self, ev):
        self.got.append(ev)


class _Pipe:
    def __init__(self):
        self.tx_onset_hooks = []


def _link_rx(css, short_frames=True, crc_bytes=None):
    """A CssLink's receive side as _start_stream builds it (main decoder + the short-frame twin)."""
    cl, _, _ = css
    cfg = _cfg(2, 2 if crc_bytes is None else crc_bytes)
    cfg.short_frames = short_frames
    sess, logs = _Sess(), []
    link = cl.CssLink(sess, cfg, log=logs.append)
    link._start_stream(_Pipe())
    return link, sess, logs


def _link_feed(link, x):
    for i in range(0, len(x), BLK):
        b = np.asarray(x[i:i + BLK], np.float32)
        link.stream.feed(b, False)
        if link.stream_short is not None:
            link.stream_short.feed(b, False)
    for st in (link.stream, link.stream_short):
        if st is not None:
            st.burst_end()


def _decoded(sess):
    from modemgui.engine.events import Decoded
    return [(d.src, d.type_name, d.seq, d.payload) for d in sess.got if isinstance(d, Decoded)]


def test_header_only_packets_go_out_in_short_frames(css):
    cl, _, _ = css
    c = _cfg(3, 2)
    from modemgui.engine.modemcfg import CssConfig
    assert CssConfig().crc_bytes == 2                                       # CRC-16 by default
    assert abs(len(cl.encode_ack(c, 2, 1)) / FS - 12.97) < 0.05            # 73 symbols, not 121
    assert abs(len(cl.encode_type(c, 15, 0, 0)) / FS - 12.97) < 0.05       # the beacon too
    assert abs(len(cl.encode_bytes(c, b"\x04abcd", 1, 2)) / FS - 20.53) < 0.05
    c.short_frames = False
    assert abs(len(cl.encode_ack(c, 2, 1)) / FS - 20.53) < 0.05
    c.crc_bytes = 4                                                         # CRC-32: 16 chirps longer
    assert abs(len(cl.encode_ack(c, 2, 1)) / FS - 23.05) < 0.05
    c.short_frames = True
    assert abs(len(cl.encode_ack(c, 2, 1)) / FS - 15.49) < 0.05


@pytest.mark.parametrize("crc", [4, 2])
def test_each_crc_width_decodes_its_own_frames(css, crc):
    """The receiver follows the sender's crc_bytes through the geometry sidecar; both widths
    decode a short ACK and a full DATA on the streaming RX."""
    cl, _, _ = css
    tx = _cfg(3, 2); tx.crc_bytes = crc
    ack = cl.encode_ack(tx, 2, 5)
    data = cl.encode_bytes(tx, b"\x04 \x0b\x1a\x01", 7, 2)
    gap = np.zeros(5 * FS, np.float32)
    x = _g711(np.concatenate([np.zeros(630, np.float32), ack, gap, data, np.zeros(2 * FS, np.float32)]))
    link, sess, logs = _link_rx(css, crc_bytes=crc)
    _link_feed(link, x)
    # two decoders, so the order they report in is not the order on the line
    assert sorted(_decoded(sess)) == [(3, "ACK", 5, b""), (3, "DATA", 7, b"\x04 \x0b\x1a\x01")], logs


def test_short_ack_and_full_data_on_one_line_each_decode_once(css):
    cl, _, _ = css
    ack = cl.encode_ack(_cfg(3, 2), 2, 5)
    data = cl.encode_bytes(_cfg(1, 2), b"\x04 \x0b\x1a\x01", 7, 2)
    gap = np.zeros(5 * FS, np.float32)
    x = _g711(np.concatenate([np.zeros(630, np.float32), ack, gap, data, gap, ack, np.zeros(2 * FS, np.float32)]))
    link, sess, logs = _link_rx(css)
    _link_feed(link, x)
    assert _decoded(sess) == [(3, "ACK", 5, b""), (1, "DATA", 7, b"\x04 \x0b\x1a\x01"), (3, "ACK", 5, b"")], logs


@pytest.mark.parametrize("phase", [0, 630, 1890, 3780, 5670])
def test_short_ack_decodes_at_any_onset_phase_through_the_conference(css, phase):
    cl, _, _ = css
    x = _g711(np.concatenate([np.zeros(phase, np.float32), cl.encode_ack(_cfg(3, 2), 2, 9),
                              np.zeros(2 * FS, np.float32)]))
    link, sess, logs = _link_rx(css)
    _link_feed(link, x)
    assert _decoded(sess) == [(3, "ACK", 9, b"")], logs


def test_short_ack_survives_a_jitter_buffer_insert_and_impulse_noise(css):
    cl, _, _ = css
    x = _g711(np.concatenate([np.zeros(FS // 2, np.float32), cl.encode_ack(_cfg(3, 2), 2, 4)]))
    i = int(8.0 * FS)
    x = np.concatenate([np.zeros(630, np.float32), x[:i], np.zeros(1026, np.float32), x[i:],
                        np.zeros(2 * FS, np.float32)])
    r = np.random.default_rng(5)
    for k in range(0, len(x) - 960, 960):
        if r.random() < 0.10:
            x[k:k + 960] += (r.standard_normal(960) * r.uniform(0.1, 0.6)).astype(np.float32)
    link, sess, logs = _link_rx(css)
    _link_feed(link, np.clip(x, -1, 1))
    assert _decoded(sess) == [(3, "ACK", 4, b"")], logs


def test_short_frames_off_keeps_one_decoder_and_full_size_acks(css):
    cl, _, _ = css
    c = _cfg(3, 2)
    c.short_frames = False
    link, sess, logs = _link_rx(css, short_frames=False)
    assert link.stream_short is None
    _link_feed(link, _g711(np.concatenate([np.zeros(630, np.float32), cl.encode_ack(c, 2, 2),
                                           np.zeros(2 * FS, np.float32)])))
    assert _decoded(sess) == [(3, "ACK", 2, b"")], logs


# -- staged receive: the streaming decoders run only on what the chirp detector marks (carrier_detect = signal) ------

class _Tap:
    def __init__(self, x, link):
        self.x, self.i, self.link = np.asarray(x, np.float32), 0, link

    def get(self, timeout=0.0):
        from modemgui.audio.pipe import Block
        if self.i >= len(self.x):
            self.link._stop.set()
            return None
        s = self.x[self.i:self.i + BLK]
        b = Block(s, self.i, False, float(np.abs(s).max()), 0)
        self.i += BLK
        return b


def _staged(css, x, mode):
    import types
    from modemgui.engine import cca
    cl, _, _ = css
    c = _cfg(2, crc_bytes=2)
    out, logs = [], []
    ses = types.SimpleNamespace(
        pipe=types.SimpleNamespace(tx_onset_hooks=[]), _emit=out.append, _emit_raw_error=logs.append,
        ident=types.SimpleNamespace(note_heard=lambda *a, **k: False),
        sense_mode=mode, sense_spec=cca.spec_from_config(None, c))
    link = cl.CssLink(ses, c, log=logs.append)
    link._ensure_backend()
    link._start_stream(ses.pipe)
    link._rx(_Tap(x, link))
    pk = [e for e in out if type(e).__name__ == "Decoded"]
    return link, pk, logs


def _room(css, snr_db=10.0, lead_s=25.0, tail_s=12.0):
    """A DATA frame from node 3 in room noise at -45 dBFS: above the -55 dBFS carrier threshold all the time."""
    cl, _, _ = css
    sig = np.asarray(cl.encode_bytes(_cfg(3, crc_bytes=2), b"hi", 1, 2), np.float64)
    r = np.random.default_rng(11)
    n0, n1 = int(lead_s * FS), int(tail_s * FS)
    nz = r.standard_normal(n0 + len(sig) + n1) * 10 ** (-45 / 20)
    nz[n0:n0 + len(sig)] += sig / np.sqrt(np.mean(sig ** 2)) * 10 ** ((-45 + snr_db) / 20)
    return nz.astype(np.float32)


def test_staged_receive_decodes_the_frame_and_skips_the_noise(css):
    cl, _, _ = css
    if not __import__("css_pll_c").load(False).has_skip:
        pytest.skip("libcsspll built before css_pll_skip")
    x = _room(css)
    link, pk, logs = _staged(css, x, "signal")
    assert [(d.src, d.type_name, d.text) for d in pk] == [(3, "DATA", "hi")], logs[-12:]
    st = link.stream
    idle = st.skipped + link._skip_n                      # the tail's skip is applied when the gate next opens
    assert idle + st.fed <= len(x) and idle > 0.45 * len(x), (idle, st.fed, len(x))   # the frame is 36 % of it
    assert st.trk.abs == st.fed + st.skipped              # one timebase: nothing lost between skip and feed
    # 2026-11-17 on the laptops: the gate closed right after the frame, the unfed tracker kept reporting busy and
    # the node never sent its ACK (or anything else).  Once the gate is closed, carrier sense is the detector's.
    assert link._gated and not link._in_burst
    link._last_busy = -1e9
    assert not link.channel_busy()


def test_level_sense_still_feeds_every_sample(css):
    x = _room(css, lead_s=6.0, tail_s=6.0)
    link, pk, logs = _staged(css, x, "level")
    assert link.stream.skipped == 0 and link.stream.fed == len(x)
