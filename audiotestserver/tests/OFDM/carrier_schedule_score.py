"""
carrier_schedule_score.py  --  compose a carrier score (JSON) for carrier-schedule OFDM from a MIDI file.

The score is what both ends of a carrier-schedule link hold in advance (ne0k-gui OfdmConfig.schedule_tx / schedule_rx,
modemgui/engine/carrier_schedule.py): a list of steps, one per OFDM data frame, each a set of MIDI notes.  The modem maps
the notes onto its own carrier grid when the link comes up, so the score carries pitches, not bins, and the same
file works on any band / sample rate.  Each frame plays one step for as long as the frame takes; the beats are kept
for reference only.

The melody line is picked the same way as schedule_ofdm_demo.py: the highest note still SOUNDING, windowed in seconds and octave-free.

Usage
-----
  python carrier_schedule_score.py joplin.mid --t-end 28.2 -o joplin_score.json
  python carrier_schedule_score.py --fur-elise -o fur_elise_score.json         # built-in tune, no MIDI needed
"""
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "CSS"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from schedule_css_demo import get_midi_tempo, _FUR_ELISE          # noqa: E402
from schedule_ofdm_demo import parse_midi_held_top, clean_melody   # noqa: E402

SCORE_VERSION = 1


def compose(melody, name, source, chord=1):
    """Score dict from [(note, beats)].  chord>1 stacks the previous chord-1 distinct notes under each step."""
    steps, held = [], []
    for n, beats in melody:
        held = [n] + [h for h in held if h != n][:max(0, chord - 1)]
        steps.append({"notes": sorted(held), "beats": round(float(beats), 4)})
    notes = [n for st in steps for n in st["notes"]]
    return {"version": SCORE_VERSION, "name": name, "source": source,
            "lo": min(notes), "hi": max(notes), "steps": steps}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("midi", nargs="?", help="input .mid")
    ap.add_argument("--fur-elise", action="store_true", help="use the built-in Fur Elise line instead of a MIDI")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--track", type=int, default=None)
    ap.add_argument("--t-start", type=float, default=0.0, help="window start s (MIDI tempo)")
    ap.add_argument("--t-end", type=float, default=None, help="window end s; cut where the piece loops")
    ap.add_argument("--chord", type=int, default=1, help="notes per step (1 = melody only)")
    args = ap.parse_args()

    if args.fur_elise:
        melody, source = list(_FUR_ELISE), "built-in Fur Elise (schedule_css_demo._FUR_ELISE)"
    elif args.midi:
        melody, source = parse_midi_held_top(args.midi, track=args.track), Path(args.midi).name
        bu = get_midi_tempo(args.midi)
        if args.t_start or args.t_end is not None:
            melody = clean_melody(melody, 0, 0.0, args.t_start / bu,
                                  None if args.t_end is None else args.t_end / bu)
    else:
        ap.error("give a MIDI file or --fur-elise")
    name = args.name or (Path(args.midi).stem if args.midi else "fur_elise")
    sc = compose(melody, name, source, args.chord)
    with open(args.out, "w") as f:
        json.dump(sc, f, indent=0)
    print(f"{args.out}: '{name}' {len(sc['steps'])} steps, MIDI notes {sc['lo']}-{sc['hi']}")


if __name__ == "__main__":
    main()
