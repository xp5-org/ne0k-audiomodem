# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""
melodic_css_demo.py  —  THROWAWAY proof-of-concept (not the real encoder).

Goal: hear what "melodic CSS chirps" sound like before committing to anything.
Does NOT touch css_encoder.py / css_decoder.py.

Idea
----
  * Melody drives PITCH: one chirp per MIDI note; the note's pitch sets the
    chirp's base frequency (scaled into [f_min, f_max]).  You hear the tune.
  * Each chirp still SWEEPS a small bandwidth (chirp_bw) around that base — that
    is what gives it the CSS "chirp" texture instead of a plain piano tone.
  * 1 byte of DATA rides on each note as a fine start-frequency offset: the byte
    (0..255) picks one of 256 sub-slots inside a small window around the pitch,
    so it stays musically on-note but is recoverable by measuring start freq.
    This is the "n discrete tone slots" idea nested INSIDE each musical note.
  * A short wideband up/down chirp preamble is emitted first (nod to Barker).

There is a tiny built-in Standard MIDI File parser so a real `.mid` works if you
pass one; otherwise it falls back to a hardcoded Fur Elise so it runs with no
inputs at all.

Usage
-----
  python melodic_css_demo.py                       # Fur Elise fallback -> wav
  python melodic_css_demo.py song.mid              # your midi
  python melodic_css_demo.py song.mid --message "hi there"
  python melodic_css_demo.py --fmin 400 --fmax 4000 --dur 0.18 --bw 300
  python melodic_css_demo.py --map linear          # stretch pitch range -> freq range

Knobs worth turning by ear: --fmin/--fmax (range), --dur (note length),
--bw (sweep width -> more/less "chirpy"), --map musical|linear, --gap.
"""

import argparse
import struct
import sys
import wave
from pathlib import Path

import numpy as np

# --- Fur Elise (melody only), as (midi_note, beats) -------------------------
# Enough of the opening phrase to be instantly recognizable.
E5, DS5, B4, D5, C5, A4, C4, E4, A4b, GS4, B4b = 76, 75, 71, 74, 72, 69, 60, 64, 45, 68, 47
_FUR_ELISE = [
    (76, 1), (75, 1), (76, 1), (75, 1), (76, 1), (71, 1), (74, 1), (72, 1),
    (69, 2), (60, 1), (64, 1), (69, 1), (71, 2), (64, 1), (68, 1), (71, 1),
    (72, 2), (64, 1), (76, 1), (75, 1), (76, 1), (75, 1), (76, 1), (71, 1),
    (74, 1), (72, 1), (69, 2),
]


def get_midi_tempo(path):
    """Seconds per quarter-note beat, from the first Set Tempo meta event
    (0xFF 0x51) found in any track.  Defaults to 120 bpm (0.5 s/beat) if the
    file has none -- that is also the MIDI spec's implicit default tempo.
    """
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise ValueError("not a Standard MIDI File (missing MThd)")
    hlen = struct.unpack(">I", data[4:8])[0]
    _fmt, ntracks, _division = struct.unpack(">HHH", data[8:14])
    pos = 8 + hlen
    for _ in range(ntracks):
        assert data[pos:pos + 4] == b"MTrk", "bad track header"
        tlen = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        i, end = pos + 8, pos + 8 + tlen
        pos = end
        status = 0
        while i < end:
            _dt, i = _read_vlq(data, i)
            b = data[i]
            if b & 0x80:
                status = b; i += 1
            if status == 0xFF:
                mtype = data[i]; i += 1
                mlen, i = _read_vlq(data, i)
                if mtype == 0x51 and mlen == 3:        # Set Tempo
                    us_per_beat = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2]
                    return us_per_beat / 1_000_000.0
                i += mlen
            elif status in (0xF0, 0xF7):
                slen, i = _read_vlq(data, i)
                i += slen
            else:
                i += 1 if (status & 0xF0) in (0xC0, 0xD0) else 2
    return 0.5  # default 120 bpm


# --- Minimal Standard MIDI File parser (type 0/1, note-on melody) ------------

def _read_vlq(data, i):
    val = 0
    while True:
        b = data[i]; i += 1
        val = (val << 7) | (b & 0x7F)
        if not (b & 0x80):
            return val, i


def parse_midi_melody(path, track=None):
    """Return list of (midi_note, duration_beats) — a monophonic melody line.

    Real MIDI files carry multiple tracks (melody + accompaniment, often even
    DUPLICATED — e.g. two identical "Piano RH" tracks), so merging every
    note-on mangles the tune.  Instead: parse each track separately, pick the
    melody track (highest mean pitch among the note-bearing tracks, ties to
    the one with more notes), and within it keep only the TOP note of any
    simultaneous group (skyline).  `track` overrides the auto-pick.
    """
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise ValueError("not a Standard MIDI File (missing MThd)")
    hlen = struct.unpack(">I", data[4:8])[0]           # header chunk length (=6)
    _fmt, ntracks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        division = 480  # SMPTE timing not handled; assume a sane PPQ
    pos = 8 + hlen

    tracks = []  # per track: (name, [(abs_tick, note), ...])
    for _ in range(ntracks):
        assert data[pos:pos + 4] == b"MTrk", "bad track header"
        tlen = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        i, end = pos + 8, pos + 8 + tlen
        pos = end
        tick, status, name = 0, 0, ""
        events = []
        while i < end:
            dt, i = _read_vlq(data, i)
            tick += dt
            b = data[i]
            if b & 0x80:
                status = b; i += 1
            # running status: status carries over, b was already a data byte
            if status == 0xFF:                      # meta
                mtype = data[i]; i += 1
                mlen, i = _read_vlq(data, i)
                if mtype == 0x03:                   # track name
                    name = data[i:i + mlen].decode("latin1", "ignore")
                i += mlen
            elif status in (0xF0, 0xF7):            # sysex
                slen, i = _read_vlq(data, i)
                i += slen
            else:
                hi = status & 0xF0
                if hi == 0x90:                      # note on
                    note, vel = data[i], data[i + 1]; i += 2
                    if vel > 0:
                        events.append((tick, note))
                elif hi in (0x80, 0xA0, 0xB0, 0xE0):  # 2-byte events we skip
                    i += 2
                else:                               # program/channel pressure
                    i += 1
        tracks.append((name, events))

    candidates = [(k, name, ev) for k, (name, ev) in enumerate(tracks) if ev]
    if not candidates:
        raise ValueError("no note-on events found")
    for k, name, ev in candidates:
        print(f"  track {k}: '{name}' {len(ev)} notes, "
              f"range {min(n for _, n in ev)}-{max(n for _, n in ev)}")

    if track is not None:
        pick = next((c for c in candidates if c[0] == track), None)
        if pick is None:
            raise ValueError(f"track {track} has no notes "
                             f"(candidates: {[c[0] for c in candidates]})")
    else:
        # Melody heuristic: highest mean pitch wins, more notes breaks ties.
        pick = max(candidates,
                   key=lambda c: (np.mean([n for _, n in c[2]]), len(c[2])))
    k, name, events = pick
    print(f"  using track {k} ('{name}')")

    # Skyline: one note per distinct tick — keep the highest of a chord.
    events.sort()
    sky = {}
    for tick, note in events:
        if tick not in sky or note > sky[tick]:
            sky[tick] = note
    events = sorted(sky.items())

    # Duration = gap to next note (fallback 1 beat), in beats.
    out = []
    for j, (tick, note) in enumerate(events):
        nxt = events[j + 1][0] if j + 1 < len(events) else tick + division
        beats = max(0.125, (nxt - tick) / division)
        out.append((note, beats))
    return out


# --- Pitch -> frequency mapping ---------------------------------------------

def build_mapper(notes, f_min, f_max, mode, headroom):
    """Return f(midi_note) -> base_freq, keeping notes inside [f_min, f_max*headroom]."""
    lo, hi = min(notes), max(notes)
    top = f_max * headroom  # leave room for the chirp sweep + data offset above base

    if mode == "linear":
        # Stretch the used pitch range across the frequency range (distorts
        # musical intervals, but uses the whole band).
        span = max(1, hi - lo)
        return lambda n: f_min + (n - lo) / span * (top - f_min)

    # "musical": equal-tempered (real Hz ratios), then octave-shift the whole
    # melody so it lands inside the band.  Intervals stay correct -> sounds right.
    def et(n):
        return 440.0 * 2 ** ((n - 69) / 12.0)
    shift = 0
    while et(lo) * 2 ** shift < f_min:
        shift += 1
    while et(hi) * 2 ** shift > top and shift > -8:
        shift -= 1
    return lambda n: et(n) * 2 ** shift


# --- Chirp synthesis ---------------------------------------------------------

def chirp(f0, f1, n, fs):
    t = np.arange(n) / fs
    k = (f1 - f0) / (n / fs)
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t)).astype(np.float32)


def struck_tone(f0, n, fs, env="struck"):
    """A discrete pitch at f0 — the MFSK "xylophone bar".

    This is what a xylophone modem actually wants: energy in ONE frequency bin,
    held steady, so a receiver detects it with a Goertzel/FFT (not a swept
    correlator).  Two envelopes:
      struck — mallet feel: fast attack, exponential decay, with a touch of the
               3rd harmonic (xylophone bars ring ~a 12th above the fundamental).
               Musical, but the decay smears bin energy a little.
      flat   — rectangular gate: constant amplitude for the whole symbol.  Ugly
               to the ear but the cleanest thing to detect — closest to a real
               MFSK symbol.
    """
    t = np.arange(n) / fs
    if env == "flat":
        amp = np.ones(n)
        tone = np.sin(2 * np.pi * f0 * t)
    else:
        # 5 ms attack, exponential decay over the note; +3rd harmonic for timbre.
        atk = min(n, int(fs * 0.005))
        amp = np.exp(-t * 6.0)
        if atk > 1:
            amp[:atk] *= np.linspace(0, 1, atk)
        tone = (np.sin(2 * np.pi * f0 * t)
                + 0.25 * np.sin(2 * np.pi * 3 * f0 * t))
    return (tone * amp).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("midi", nargs="?", help="input .mid (omit for Fur Elise demo)")
    ap.add_argument("-o", "--out", default="melodic_css.wav")
    ap.add_argument("--message", default=None,
                    help="data bytes to ride on the notes (cycled/truncated to fit)")
    ap.add_argument("--fmin", type=float, default=400.0)
    ap.add_argument("--fmax", type=float, default=4000.0)
    ap.add_argument("--dur", type=float, default=0.18, help="seconds per note")
    ap.add_argument("--bw", type=float, default=250.0, help="chirp sweep bandwidth Hz")
    ap.add_argument("--synth", choices=["tone", "chirp"], default="tone",
                    help="tone=MFSK struck bar (xylophone), chirp=CSS glide")
    ap.add_argument("--env", choices=["struck", "flat"], default="struck",
                    help="tone envelope: struck=mallet decay, flat=rectangular gate")
    ap.add_argument("--gap", type=float, default=0.03, help="silence between notes s")
    ap.add_argument("--map", choices=["musical", "linear"], default="musical")
    ap.add_argument("--headroom", type=float, default=0.85,
                    help="fraction of fmax used for the highest base freq")
    ap.add_argument("--use-note-dur", action="store_true",
                    help="use MIDI note durations instead of fixed --dur")
    ap.add_argument("--limit", type=int, default=0,
                    help="only render the first N notes (0 = all)")
    ap.add_argument("--track", type=int, default=None,
                    help="MIDI track to use (default: auto-pick melody track)")
    ap.add_argument("--fs", type=int, default=44100)
    ap.add_argument("--amp", type=float, default=0.7)
    args = ap.parse_args()

    if args.midi:
        melody = parse_midi_melody(args.midi, track=args.track)
        src = args.midi
    else:
        melody = _FUR_ELISE
        src = "built-in Fur Elise"

    if args.limit > 0:
        melody = melody[:args.limit]

    fs, notes = args.fs, [n for n, _ in melody]
    tone = build_mapper(notes, args.fmin, args.fmax, args.map, args.headroom)

    # Data byte per note.  Offset window is a small fraction of the note's own
    # frequency so the byte never pushes the pitch audibly off-key.
    msg = (args.message or "").encode("utf-8")
    payload = bytes(msg[i % len(msg)] for i in range(len(melody))) if msg else None

    out = [np.zeros(int(fs * 0.15), dtype=np.float32)]  # lead-in silence

    # Preamble: 2 wideband up/down chirps across the whole band.
    pn = int(fs * args.dur)
    for f0, f1 in [(args.fmin, args.fmax), (args.fmax, args.fmin)]:
        out.append(chirp(f0, f1, pn, fs))
        out.append(np.zeros(int(fs * args.gap), dtype=np.float32))

    # Each chirp GLIDES from this note's pitch to the NEXT note's pitch, so the
    # sweep endpoints ARE the MIDI tones (portamento between notes) — the melody
    # is heard as the pitch contour, and it's still a real chirp (never a flat
    # tone) except when two consecutive notes repeat.  Data byte, if any, rides
    # as a small offset on the START frequency only, leaving the target honest.
    freqs = [tone(n) for n, _ in melody]
    gap = np.zeros(int(fs * args.gap), dtype=np.float32)
    for i, (note, beats) in enumerate(melody):
        dur = beats * args.dur if args.use_note_dur else args.dur
        n = int(fs * dur)
        f0 = freqs[i]
        if payload is not None:
            f0 += (payload[i] / 255.0 - 0.5) * (f0 * 0.06)
        if args.synth == "chirp":
            f1 = freqs[i + 1] if i + 1 < len(freqs) else freqs[i]
            out.append(chirp(f0, f1, n, fs))
        else:
            out.append(struck_tone(f0, n, fs, env=args.env))
        out.append(gap)

    audio = np.clip(np.concatenate(out) * args.amp, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(args.out, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs)
        wf.writeframes(pcm.tobytes())

    dur_s = len(audio) / fs
    fr = [tone(n) for n in notes]
    print(f"source        : {src}")
    print(f"notes         : {len(melody)}   map={args.map}")
    print(f"freq range    : {min(fr):.0f}-{max(fr):.0f} Hz  (band {args.fmin:.0f}-{args.fmax:.0f})")
    print(f"synth         : {args.synth} ({args.env})   note dur: {args.dur*1000:.0f} ms")
    print(f"data          : {len(payload) if payload else 0} bytes ({'1/note' if payload else 'none'})")
    print(f"wrote         : {args.out}   ({dur_s:.1f} s)")


if __name__ == "__main__":
    main()
