"""
melodic_ofdm_demo.py  —  OFDM synthesis on a musical scale (audition only, no decoder).

Standard OFDM places subcarriers on a uniform FFT grid (k * fs/n_fft Hz).
This variant instead:
  * builds a musical scale over [f_min, f_max],
  * snaps each scale note to its nearest FFT bin (still orthogonal OFDM),
  * activates only those bins, mutes everything else,
  * QAM-modulates data onto each active bin.
One OFDM symbol (n_fft/fs seconds) is a held chord of scale notes, with data
in the per-note amplitude/phase. QPSK keeps every note at equal loudness;
16-QAM carries 2x the bits but the notes shimmer in level.

Modes
-----
  --chord         every symbol rings the whole scale (sustained OFDM drone,
                  max carriers / max data rate).
  (default)       follow a melody (MIDI or built-in Fur Elise): each symbol
                  activates the current note's bin + its octave (dual-tone).

Melody pitch mapping (dual-tone mode) is chromatic, not scale-quantized: it
reuses build_mapper()'s equal-tempered note->Hz from melodic_css_demo.py.
--scale only shapes --chord's drone and the note->bin snap grid.

Note timing uses the MIDI file's own tempo (Set Tempo meta event) by default,
via get_midi_tempo(). Override with --beat-unit.

Usage
-----
  python melodic_ofdm_demo.py                         # Fur Elise melody, QPSK
  python melodic_ofdm_demo.py --chord                 # full-scale drone
  python melodic_ofdm_demo.py fur_elise.mid --use-note-dur
  python melodic_ofdm_demo.py --fmin 400 --fmax 4000 --nfft 8192

Reuses parse_midi_melody / build_mapper / get_midi_tempo from melodic_css_demo.py.
"""
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

import argparse
import math
import wave

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

from melodic_css_demo import parse_midi_melody, build_mapper, get_midi_tempo, _FUR_ELISE

# --- QAM constellations (unit average power), Gray-coded ---------------------
_R2, _R10 = math.sqrt(2), math.sqrt(10)
_QPSK = np.array([(1+1j), (-1+1j), (-1-1j), (1-1j)]) / _R2
_QAM16 = np.array([complex(i, q) for i in (-3, -1, 1, 3) for q in (-3, -1, 1, 3)]) / _R10
_CONST = {2: _QPSK, 4: _QAM16}

# --- Scale definitions (semitone offsets within an octave) ------------------
_SCALES = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 3, 5, 7, 10],          # minor pentatonic
    "chromatic":  list(range(12)),
}


def scale_freqs(f_min, f_max, scale):
    """Musical scale frequencies (Hz) spanning [f_min, f_max], A440 tuning."""
    degs = _SCALES[scale]
    # midi 69 = A4 = 440 Hz. Walk octaves overlapping the band.
    out = []
    for octave in range(-2, 9):
        for d in degs:
            n = 69 + d + 12 * (octave - 4)
            f = 440.0 * 2 ** ((n - 69) / 12.0)
            if f_min <= f <= f_max:
                out.append((n, f))
    return sorted(out, key=lambda x: x[1])


def snap_to_bins(freqs, fs, n_fft, bin_lo, bin_hi):
    """Map (midi, hz) -> unique FFT bin indices nearest each note."""
    sc = fs / n_fft
    seen, bins = {}, []
    for midi, f in freqs:
        b = int(round(f / sc))
        if b < bin_lo or b > bin_hi or b in seen:
            continue
        seen[b] = midi
        bins.append((midi, b, b * sc))
    return bins


def ofdm_symbol(active_bins, syms, n_fft, cp_len):
    """Hermitian IFFT of QAM values placed on active bins -> real CP'd symbol."""
    X = np.zeros(n_fft, dtype=complex)
    for b, s in zip(active_bins, syms):
        X[b] = s
        X[n_fft - b] = np.conj(s)
    td = np.fft.ifft(X).real * math.sqrt(n_fft)
    return np.concatenate([td[-cp_len:], td])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("midi", nargs="?", help="input .mid (omit for Fur Elise)")
    ap.add_argument("-o", "--out", default="melodic_ofdm.wav")
    ap.add_argument("--message", default=None, help="data bytes (else random)")
    ap.add_argument("--fmin", type=float, default=400.0)
    ap.add_argument("--fmax", type=float, default=4000.0)
    ap.add_argument("--fs", type=int, default=44100)
    ap.add_argument("--nfft", type=int, default=8192,
                    help="FFT size: symbol dur = nfft/fs s (bin res = fs/nfft Hz)")
    ap.add_argument("--cp", type=float, default=0.25, help="cyclic-prefix ratio")
    ap.add_argument("--scale", choices=list(_SCALES), default="major")
    ap.add_argument("--bits", type=int, choices=[2, 4], default=2,
                    help="bits/carrier: 2=QPSK (clean chord), 4=16-QAM (shimmer)")
    ap.add_argument("--chord", action="store_true",
                    help="ring the whole scale every symbol (OFDM drone)")
    ap.add_argument("--headroom", type=float, default=0.85,
                    help="fraction of fmax the octave-doubled tone may reach")
    ap.add_argument("--track", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-note-dur", action="store_true",
                    help="hold each note's chord across enough OFDM symbols to "
                         "match its MIDI duration, instead of 1 symbol/note")
    ap.add_argument("--beat-unit", type=float, default=None,
                    help="seconds per quarter-note beat (tempo); default: read "
                         "the MIDI file's own Set Tempo event")
    ap.add_argument("--gap", type=float, default=0.0, help="silence between symbols s")
    ap.add_argument("--amp", type=float, default=0.7)
    ap.add_argument("--extra-carriers", type=int, default=0,
                    help="extra data-only subcarriers ABOVE fmax, always active "
                         "alongside the melody tones (e.g. 12). Orthogonal FFT "
                         "bins, so they don't disturb the melody carriers; "
                         "snapped to the same --scale (transposed up) so if "
                         "they're audible they blend instead of hissing.")
    ap.add_argument("--extra-fmax", type=float, default=None,
                    help="top of the extra-carrier band (default: 3x fmax)")
    args = ap.parse_args()

    fs, n_fft = args.fs, args.nfft
    cp_len = int(n_fft * args.cp)
    sym_dur = (n_fft + cp_len) / fs
    sc = fs / n_fft
    bin_lo = max(1, int(math.floor(args.fmin / sc)))
    bin_hi = min(n_fft // 2 - 1, int(math.ceil(args.fmax / sc)))

    scale_bins = snap_to_bins(scale_freqs(args.fmin, args.fmax, args.scale),
                              fs, n_fft, bin_lo, bin_hi)
    if not scale_bins:
        raise SystemExit("no scale notes fit the band / FFT resolution")
    bin_of = {midi: b for midi, b, _ in scale_bins}
    all_active = [b for _, b, _ in scale_bins]

    # Extra data-only carriers track the current note as integer-multiple
    # harmonics of its fundamental. b0=1st harmonic (fundamental), b1=2nd
    # (octave, existing dual-tone); extras are the 3rd..(2+extra_carriers)th
    # harmonics, capped at --extra-fmax (default 3x fmax) / Nyquist.
    extra_fmax = args.extra_fmax or args.fmax * 3
    extra_bin_cap = min(n_fft // 2 - 1, int(math.floor(extra_fmax / sc)))

    def harmonics(base_freq, count, taken):
        """Bins for harmonics 3..(2+count) of base_freq, deduped vs `taken`."""
        out = []
        k = 3
        while len(out) < count and base_freq * k / sc <= extra_bin_cap:
            b = int(round(base_freq * k / sc))
            if b not in taken and b not in out:
                out.append(b)
            k += 1
        return out

    const = _CONST[args.bits]
    rng = np.random.default_rng(0)
    msg_bits = None
    if args.message:
        raw = np.unpackbits(np.frombuffer(args.message.encode(), dtype=np.uint8))
        msg_bits = raw
    bitpos = 0

    def next_syms(k):
        """k QAM symbols from the message (looped) or random."""
        nonlocal bitpos
        idxs = np.empty(k, dtype=int)
        for j in range(k):
            if msg_bits is not None:
                chunk = [msg_bits[(bitpos + t) % len(msg_bits)] for t in range(args.bits)]
                bitpos += args.bits
                idxs[j] = int("".join(map(str, chunk)), 2)
            else:
                idxs[j] = rng.integers(len(const))
        return const[idxs]

    # Build the list of active-bin sets, one per OFDM symbol.
    if args.chord:
        # --extra-carriers is unsupported in --chord mode: no per-note
        # fundamental to track, the drone rings the whole scale at once.
        symbol_bins = [all_active] * max(1, args.limit or 64)
        src = f"scale drone ({args.scale})"
    else:
        if args.midi:
            melody = parse_midi_melody(args.midi, track=args.track)
            src = args.midi
            beat_unit = (args.beat_unit if args.beat_unit is not None
                        else get_midi_tempo(args.midi))
        else:
            melody = _FUR_ELISE
            src = "built-in Fur Elise"
            beat_unit = args.beat_unit if args.beat_unit is not None else 0.5
        if args.limit > 0:
            melody = melody[:args.limit]

        # Dual-tone mapping is chromatic (equal-tempered), not snapped to the
        # --scale grid: reuses build_mapper(), preserving every semitone interval.
        notes = [n for n, _ in melody]
        tone  = build_mapper(notes, args.fmin, args.fmax, "musical", args.headroom)
        all_bins = list(range(bin_lo, bin_hi + 1))

        def nearest_bin(f):
            b = int(round(f / sc))
            return min(all_bins, key=lambda x: abs(x - b))

        symbol_bins = []
        for note, beats in melody:
            base = tone(note)
            b0 = nearest_bin(base)
            b1 = nearest_bin(base * 2)          # octave companion (dual-tone)
            active = [b0] if b1 == b0 else [b0, b1]
            if args.extra_carriers > 0:
                active = active + harmonics(base, args.extra_carriers, set(active))
            # Hold this note's chord across enough OFDM symbols to match its
            # MIDI duration. Each held symbol still draws fresh QAM data.
            n_slots = (max(1, round(beats * beat_unit / sym_dur))
                       if args.use_note_dur else 1)
            symbol_bins.extend([active] * n_slots)

    # Synthesize.
    gap = np.zeros(int(fs * args.gap), dtype=np.float32) if args.gap > 0 else None
    taper = max(1, int(fs * 0.004))
    out = [np.zeros(int(fs * 0.1), dtype=np.float32)]
    for active in symbol_bins:
        s = ofdm_symbol(active, next_syms(len(active)), n_fft, cp_len)
        s = s.astype(np.float32)
        ramp = np.linspace(0, 1, taper, dtype=np.float32)
        s[:taper] *= ramp
        s[-taper:] *= ramp[::-1]
        out.append(s)
        if gap is not None:
            out.append(gap)

    audio = np.concatenate(out)
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = np.clip(audio / peak * args.amp, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(args.out, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs)
        wf.writeframes(pcm.tobytes())

    carriers = (len(all_active) if args.chord
                else float(np.mean([len(a) for a in symbol_bins])))
    gross_bps = carriers * args.bits / sym_dur
    print(f"source        : {src}")
    print(f"scale         : {args.scale}   notes in band: {len(scale_bins)} "
          f"({scale_bins[0][2]:.0f}-{scale_bins[-1][2]:.0f} Hz)")
    print(f"fft/symbol    : n_fft={n_fft} cp={cp_len}  sym_dur={sym_dur*1000:.0f} ms  "
          f"bin_res={sc:.1f} Hz")
    print(f"modulation    : {'QPSK' if args.bits==2 else '16-QAM'} "
          f"({args.bits} b/carrier)   avg carriers/sym={carriers:.1f}")
    print(f"data rate     : ~{gross_bps:.0f} bits/s gross "
          f"({'message' if args.message else 'random'} payload)")
    if not args.chord:
        print(f"tempo         : {beat_unit:.3f} s/beat ({60/beat_unit:.1f} bpm)"
              f"{'  [--beat-unit override]' if args.beat_unit is not None else '  [from MIDI]'}"
              f"   note-dur={'on' if args.use_note_dur else 'off (1 symbol/note)'}")
    print(f"wrote         : {args.out}   ({len(audio)/fs:.1f} s)")


if __name__ == "__main__":
    main()
