# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""CSS wake-preamble cost sweep: per-frame run-in cost vs noise.

    python tests/regression/css_wake_sweep.py                  # full sweep
    python tests/regression/css_wake_sweep.py --quick          # smoke, ~2 min
    python tests/regression/css_wake_sweep.py --leg idle -v
    python tests/regression/css_wake_sweep.py --candidate wake_w2 --candidate wake_w1
    python tests/regression/css_wake_sweep.py --out /tmp/wake_sweep.json

Every chunk frame pays a full run-in: training chirps plus the 13-symbol
Barker preamble (19 symbols at symbol_dur=0.5s, 29 at the firmware PHY's
0.12s) ahead of 104 payload symbols, 15-22% of airtime. css_encoder has a
cheaper alternative, preamble_mode="wake" (a short wake_word before each
resumed frame), but its cost was unmeasured.

Decode-quality sweeps (css_loopback's pink/impulse/reverb) cannot measure
this: they grade decoded text, dominated by payload SNR, which a shorter
preamble does not touch. A shorter marker changes ACQUISITION, which fails
in two directions:

  MISSED WAKE   marker sent, not recognised -- chunk lost even though its
                payload was receivable.
  FALSE WAKE    marker "recognised" in ambient noise with nothing sent. The
                decoder enters COLLECT, spends a full frame window
                (frame_symbols, ~55s at the default geometry) reading noise,
                CRC-fails it, and is deaf for that window -- so the real next
                chunk arriving inside it is lost.

Those two trade off against each other, so this sweep always runs both legs
over the same candidates, impairment families and levels, and reports the
SNR at which each leg gives out.

THE TWO LEGS.

  leg "wake"  multi-frame encode, impaired, decoded. Scored on two numbers:

                cold      did frame 0 decode. Prices the TRAINING chirps,
                          which let the timing loop converge before the
                          first Barker window; only protects cold
                          acquisition.
                deliver   fraction of RESUMED frames decoded. Prices the
                          per-frame MARKER. Frame 0 excluded (it carries the
                          full run-in in every candidate).

              Delivery alone is a trap: later frames inherit a locked grid,
              so delivery is blind to training cuts and reports them as
              free at every impairment level.

              --gap chooses the regime. With a gap (default) one decode
              session freewheels across the pause, which is what wake mode
              requires. With --gap 0 frames are contiguous inside one
              transmission, matching how a live node sends (it decodes each
              transmission separately, see css_modem_loopback.py).

  leg "idle"  the same encode truncated after frame 0, with `idle_s` of dead
              air appended, impairment applied across the whole stream.
              Nothing is transmitted in that window, so every acquisition
              logged there (MF_WAKE or MF_PREAMBLE) is false. Expected
              frames in the info sidecar stays >1, so the decoder keeps
              listening across the idle as it would while waiting for the
              next chunk.

The per_frame candidates are in the grid because they measure the current
scheme on the identical stream, so a false-acquisition rate reads relative
to today. The Barker matcher runs on every symbol regardless of whether a
frame is expected, and arbitrary data matches Barker-13 in >=11/13
positions with p=1.12% per window, so the long preamble is itself a
false-acquisition source; the frame-cadence guard suppresses this in
per_frame mode and cannot in wake mode (the suspend gap makes cadence
variable). Both legs must be measured.

OUTPUT. Per candidate, two floors in dB of pink-noise SNR:

  deliver_floor  lowest SNR at which chunks still land (>= --deliver-thr).
  quiet_floor    lowest SNR at which the receiver is not chasing ghosts
                 (false acquisitions <= --false-budget per hour).

The usable floor is the worse of the two; which one binds decides the fix
(deliver_floor wants a longer/more distinctive marker, quiet_floor wants a
tighter gate). Airtime saved per candidate is computed from geometry alone.

Impairment families: pink noise, impulse clicks, clock drift (ppm). Drift
tests what the training chirps are for; without it a sweep reports the
run-in as free to delete at any length.

These are simulation floors on synthetic impairments, used to pick
candidates and an SNR band worth OTA time, and what to instrument when you
do (count MF_WAKE and MF_PREAMBLE against frames actually sent -- a field
run that only logs decoded text cannot distinguish a missed wake from a
false one). The OTA PLAN block at the end of the report covers both;
--out writes the whole thing as JSON.
"""
import argparse
import collections
import json
import math
import os
import sys
import time
import traceback

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGRESS = os.path.join(TESTS, "regression")

sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(TESTS, "CSS"))
sys.path.insert(0, REGRESS)


# ── the candidates ───────────────────────────────────────────────────────────
#
# Encoder overrides only; everything else about the encode is held identical
# so the per-frame run-in is the only variable. Ordered longest run-in first.
#
# The decoder requires an exact bit match over the wake word AND a
# weak->strong edge before it (wake block in css_decoder.py): the word adds
# bit entropy on top of that edge, 2^-W of the false-edge rate survives a
# W-symbol word. [1,0] is the current default and alternates like the
# training sequence; w3/w4 deliberately break that alternation so a training
# run-in cannot contain them.
# The per_frame ladder is an explicit training-symbol count, not a delta from
# the auto default: shared.default_training_symbols scales with symbol_dur
# (6 symbols at 0.5s, 16 at the production 0.12s), so "t8" means the same
# encode at every geometry while "full" does not. t12/t16 are above the auto
# default at 0.5s and below it at 0.12s.
CANDIDATES = {
    "per_frame_full": dict(preamble_mode="per_frame"),
    "per_frame_t12":  dict(preamble_mode="per_frame", training_symbols=12),
    "per_frame_t8":   dict(preamble_mode="per_frame", training_symbols=8),
    "per_frame_t4":   dict(preamble_mode="per_frame", training_symbols=4),
    "per_frame_t2":   dict(preamble_mode="per_frame", training_symbols=2),
    "per_frame_t0":   dict(preamble_mode="per_frame", training_symbols=0),
    "wake_w4":        dict(preamble_mode="wake", wake_word=[1, 1, 0, 1]),
    "wake_w3":        dict(preamble_mode="wake", wake_word=[1, 1, 0]),
    "wake_w2":        dict(preamble_mode="wake", wake_word=[1, 0]),
    "wake_w1":        dict(preamble_mode="wake", wake_word=[1]),
}

# Impairment levels. Pink noise is the shared sweep axis the floors are
# reported in. Clicks are a second family: an impulse is broadband and
# arrives on no grid, so it is the impairment most likely to read in the
# matched filter as a strong peak at an arbitrary phase. Density is in
# clicks/second; amplitude is a multiple of signal RMS (CLICK_AMP_RATIO).
PINK_SNRS = (24.0, 18.0, 12.0, 9.0, 6.0, 3.0, 0.0)
CLICK_DENSITIES = (0.5, 2.0, 8.0)
# Clock offset in ppm. The only family that tests what the training chirps
# are for: training lets the timing loop converge before the Barker window
# opens (shared.default_training_symbols). With a perfect clock there is
# nothing to converge, so a sweep without drift reports training as free to
# delete at any length.
DRIFT_PPMS = (0.0, 200.0, 800.0, 2000.0, 4000.0)
# Peak ppm of a sinusoidally varying clock (css_loopback.warp_wobble). A
# constant offset is converged on once and tracked, so a training-symbol
# sweep against drift alone reports every length as equally good; a
# wobbling clock never lets the loop settle, which is the condition
# training exists to buy.
WOBBLE_PPMS = (500.0, 2000.0, 6000.0)
WOBBLE_HZ = 0.5

# Click amplitude as a multiple of reference signal RMS. css_loopback's
# impulse benchmark measured 60x for clicks landing on chirps (the matched
# filter's spreading gain). Here clicks land mostly on silence, where there
# is no spreading gain to overcome, so a click a few times signal level is
# the realistic case.
CLICK_AMP_RATIO = 6.0

# ── geometry, and the one constraint that decides it ─────────────────────────
#
# WAKE MODE IS MF-PATH ONLY.  css_decoder.decode() dispatches on symbol
# duration: above mf_longsym_min_dur (0.25 s) it runs decode_longsym_mf, and
# the entire wake-word gate lives inside that function.  Below it, the
# streaming PLL decoder runs and has no wake-word support: pointed at a
# wake-mode encode it looks for a per-frame preamble that is not there, so
# every resumed frame is lost with zero MF_WAKE events logged.
#
# That matters for the production PHY, which is css_modem_fw.DEFAULT_PHY at
# symbol_dur=0.12 s -- BELOW the dispatch threshold.  So "shorten the per-frame
# run-in" cannot be done today at the geometry the modem actually transmits,
# whatever this sweep says about wake words: it would need the gate ported to
# the PLL path, or the link moved to long symbols.  The default here is
# therefore 0.5 s, where the feature exists and the question is answerable;
# --dur 0.12 --force-mf runs the production symbol rate through the MF decoder
# to price that second option.
#
# chunk_size stays at the PHY's value: packets are padded to it, so one packet
# is exactly one frame, and the run-in-to-payload ratio this sweep is trying to
# improve is only meaningful against a real payload length.
DEF_DUR = 0.5
DEF_CHUNK = 8
# Frames per run: each one after the first is a resume, so this is also the
# sample size for the delivery rate.  Eight makes one missed resume a visible
# 1/7 rather than all-or-nothing, without making a run long enough to matter.
DEF_FRAMES = 8
# Idle between packets.  A real chunked link is bursty and irregular; 10 s is
# short enough to keep runs cheap and, deliberately, is NOT a whole number of
# frame periods -- a gap that happened to land on the cadence lattice would
# flatter the per_frame candidates for a reason that has nothing to do with
# their preamble (see the cadence guard in css_decoder.py).
DEF_GAP_S = 10.0
# The idle-leg listening window.  It has to be several frame periods long or
# the metrics derived from it saturate rather than measure: `deaf_frac` is
# false_acqs x frame_dur / idle_s, so at one frame period per idle window a
# single false acquisition reads as 100% deaf and every candidate that fires at
# all looks identical.  15 minutes is ~16 frame periods at the default
# geometry, and the decode costs ~45 s of CPU at RTF 0.05.
DEF_IDLE_S = 900.0
# Below this many frame periods of idle, the false-acquisition metrics are
# reported with a warning rather than silently.
MIN_IDLE_FRAMES = 5.0
DEF_TRIALS = 3


def make_message(frames, chunk):
    """Payload for `frames` frames of `chunk` bytes, one packet per frame.

    Printable and per-frame distinct so a partial decode is legible in the log
    ("AAAAAAAACCCCCCCC" says frame 1 was the one that went missing), and
    deterministic so two runs of the sweep compare.
    """
    return "".join(chr(65 + (i % 26)) * chunk for i in range(frames))


# ── worker side ──────────────────────────────────────────────────────────────
#
# Imports are lazy and inside the worker: the parent only builds specs and
# prints tables, and paying for the decoder stack in every forked child that
# does not need it is the difference between a fast startup and a slow one.

# (candidate, dur, gap, message, chunk) -> (pcm, fs, info, geom), LRU-bounded
_ENC = collections.OrderedDict()
_ENC_MAX = 2


def _geometry(conf, fs):
    """Frame geometry in samples and symbols, straight from the info sidecar.

    Everything downstream -- where frame 0 ends, how long a bogus COLLECT
    window lasts, how much airtime the run-in costs -- is derived here rather
    than re-computed at each use, because these are exactly the numbers that
    have silently drifted apart before (see the encoder's CHIRP_SCHEDULE line,
    which exists for the same reason).
    """
    sps = int(fs * conf["symbol_dur"])
    gap = int(sps * conf["symbol_gap_frac"])
    sps_eff = sps + gap
    lead_syms = 4                       # css_encoder lead_silence_syms default
    run_in = conf["training_symbols"] + len(conf["preamble"])
    wake_syms = (len(conf.get("wake_word") or [])
                 if conf.get("preamble_mode") == "wake" else run_in)
    frame0_syms = run_in + conf["frame_symbols"]
    return {
        "sps": sps, "gap": gap, "sps_eff": sps_eff,
        "lead_syms": lead_syms,
        "run_in_syms": run_in,
        "resume_syms": wake_syms,       # run-in paid by frames 1..n-1
        "frame_symbols": conf["frame_symbols"],
        "frame0_syms": frame0_syms,
        # Sample where frame 0's payload ends: lead silence has no
        # inter-symbol gaps, every symbol after it occupies a full slot.
        "frame0_end": sps * lead_syms + frame0_syms * sps_eff,
        # Decoder symbol counter runs from the grid anchor at the first
        # chirp; lead_syms + frame0_syms is that index +/- acquisition slack.
        # Separates idle-window events from frame 0's own (~50 symbols apart).
        "idle_start_sym": lead_syms + frame0_syms,
        "frame_dur_s": conf["frame_symbols"] * sps_eff / fs,
        "resume_dur_s": wake_syms * sps_eff / fs,
        "frames": conf["frames"],
    }


def _encode(candidate, dur, gap_s, message, chunk, enc_extra=()):
    """The unperturbed wake-mode encode for one candidate, cached per process.

    Encoded in the worker rather than shipped from the parent: a 300s encode
    is ~14M float32 samples, and pickling that to every task would cost more
    than the decodes. The pid is in the filename because css_regress writes
    into one shared work dir and two workers encoding the same geometry would
    otherwise race on the same path.
    """
    key = (candidate, dur, gap_s, message, chunk, enc_extra)
    if key in _ENC:
        _ENC.move_to_end(key)
        return _ENC[key]
    import numpy as np
    import css_regress
    kw = dict(CANDIDATES[candidate])
    # enc_extra is an arbitrary (key, value) tuple of encoder overrides, passed
    # as a TUPLE rather than a dict so it can key the cache.  It is what lets a
    # caller sweep an encoder parameter the CANDIDATES table does not enumerate
    # -- css_matrix.py sweeps symbol_dur and training_symbols this way -- while
    # every candidate keeps its own identity.
    kw.update(dict(enc_extra))
    wav, info = css_regress.encode_synth(
        f"wksw_{candidate}_{dur}_{os.getpid()}",
        symbol_dur=dur, message=message, chunk_size=chunk,
        inter_frame_silence_s=gap_s, **kw)
    pcm, fs = css_regress.load_wav(wav)
    conf = json.load(open(info))
    geom = _geometry(conf, fs)
    # Reference RMS for every impairment level: measured over frame 0's chirps
    # ONLY.  The stream is mostly silence (suspend gaps, lead-in, tail), so a
    # whole-stream RMS would read several dB low and every "12 dB SNR" would
    # silently be a different, quieter noise level for every candidate --
    # exactly the confound this sweep exists to avoid.
    a = pcm[fs // 2 + geom["sps"] * geom["lead_syms"]:geom["frame0_end"]]
    geom["ref_rms"] = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    _ENC[key] = (pcm, fs, info, geom)
    # Bounded LRU rather than an unbounded dict: a worker sees every candidate
    # in the grid, each encode is ~100 MB at the default geometry, and with the
    # pool now long-lived (see the max_tasks_per_child note in main) nothing
    # else ever frees them.  Two entries is enough that the common case -- a
    # run of tasks on the same candidate -- still hits the cache.
    while len(_ENC) > _ENC_MAX:
        _ENC.popitem(last=False)
    return _ENC[key]


def _pink(n, rng, block=1 << 21):
    """css_loopback's pink noise, generated in blocks so it fits in memory.

    LB._pink_noise is an FFT shaping of a white draw, and every intermediate is
    float64 or complex128: a 15-minute idle stream at 48 kHz is 43 M samples,
    so one call transiently holds ~2 GB.  Multiplied by a worker per core, the
    sweep runs out of memory.

    Blocking it fixes that for the price of the 1/f correlation not extending
    past a block (~43 s).  For a noise FLOOR that is not a meaningful loss, and
    it is arguably the more realistic of the two: real ambient noise is not one
    stationary draw for fifteen minutes either.  Blocks are equal-sized rather
    than "full blocks plus a remainder" so the last one is shaped over the same
    bandwidth as the rest and does not join at a different level.
    """
    import numpy as np
    import css_loopback as LB
    if n <= block:
        return LB._pink_noise(n, rng)
    nb = -(-n // block)
    size = -(-n // nb)
    return np.concatenate(
        [LB._pink_noise(min(size, n - i * size), rng) for i in range(nb)])


def _impair(sig, fs, spec, geom, rng):
    """Apply one impairment family at one level.  Pure: returns a new array.

    Levels are expressed against `ref_rms` (frame 0's chirp RMS) rather than
    against the array's own RMS, because in the idle leg the array is almost
    entirely silence -- css_loopback's add_pink_noise would reference that and
    produce a wildly quieter noise floor than the same nominal SNR does in the
    wake leg, making the two legs incomparable.
    """
    import numpy as np
    import css_loopback as LB
    ref = geom["ref_rms"]
    fam = spec["family"]
    if fam == "none":
        return sig
    if fam == "pink":
        nz = _pink(len(sig), rng)
        return np.clip(sig + nz * (ref / (10.0 ** (spec["level"] / 20.0))),
                       -1.0, 1.0).astype(np.float32)
    if fam == "click":
        # A quiet pink floor underneath so the between-click stretches are not
        # digital silence: a receiver in a real room never sees a dead channel,
        # and the strong/weak classifier behaves differently when it does.
        nz = _pink(len(sig), rng)
        out = np.clip(sig + nz * (ref / (10.0 ** (24.0 / 20.0))), -1.0, 1.0)
        return LB.add_impulse_clicks(out.astype(np.float32), spec["level"], fs,
                                     ref * CLICK_AMP_RATIO, rng)
    if fam == "wobble":
        # Time-varying clock, then the noise floor -- same ordering rationale
        # as drift: the wobble is the transmitter's clock, the noise is the
        # receiver's room.
        #
        # The phase matters as much as the depth here, and for a reason that
        # is specific to what this sweep compares.  Candidates differ in
        # run-in length, so they differ in WHERE each frame's acquisition
        # lands inside the wobble cycle.  With a fixed phase every trial of a
        # candidate lands in the same place, so a candidate that happens to
        # acquire on the wobble's steepest slope loses frames in all six
        # trials identically -- a per-candidate ranking that is really one
        # sample of clock phase, and which reads as reproducible precisely
        # because it is not being resampled.  See spec["wobble_phase"].
        sig = LB.warp_wobble(sig, spec["level"],
                             float(spec.get("wobble_hz") or WOBBLE_HZ),
                             fs, rng, phase=float(spec.get("wobble_phase", 0.0)))
        nz = _pink(len(sig), rng)
        return np.clip(sig + nz * (ref / (10.0 ** (18.0 / 20.0))),
                       -1.0, 1.0).astype(np.float32)
    if fam == "drift":
        # Resample first, THEN add the noise floor: the drift is the
        # transmitter's clock, which the receiver's ambient noise is not
        # subject to.  Resampling the noise too would be a subtly easier
        # channel (correlated noise), and it is the sort of thing that makes a
        # run-in look adequate when it is not.  A fixed 18 dB floor keeps
        # timing, not level, as the variable.
        sig = LB.resample_ppm(sig, spec["level"])
        nz = _pink(len(sig), rng)
        return np.clip(sig + nz * (ref / (10.0 ** (18.0 / 20.0))),
                       -1.0, 1.0).astype(np.float32)
    if fam == "reverb":
        # RT60 in seconds.  This is the family the other four cannot stand in
        # for, and the reason it exists: pink, click, drift and wobble are all
        # ADDITIVE or CLOCK impairments, and both decoders survive them almost
        # identically.  Reverb is neither -- it puts a delayed copy of the
        # signal's OWN chirps into the buffer, which is the one thing that
        # looks like a valid symbol to a correlator.  That is precisely where a
        # matched filter (which can reject an echo peak for being off the
        # symbol grid) and a PLL (which chases whichever peak wins the search
        # window) should diverge, and no other family here exercises it.
        #
        # Reverb first, THEN the noise floor: the room acts on the transmitted
        # signal, and the receiver's own noise is not reverberated.  Same
        # fixed 18 dB floor as drift/wobble so the room, not the level, is the
        # variable.
        sig = LB.add_reverb(sig, spec["level"], fs, rng,
                            wet=float(spec.get("reverb_wet") or 0.5))
        nz = _pink(len(sig), rng)
        return np.clip(sig + nz * (ref / (10.0 ** (18.0 / 20.0))),
                       -1.0, 1.0).astype(np.float32)
    raise ValueError(f"unknown family {spec['family']!r}")


def _build(spec):
    """(pcm, fs, info_path, geom) for one spec, impairment applied."""
    import numpy as np
    pcm, fs, info, geom = _encode(spec["candidate"], spec["dur"],
                                  spec["gap_s"], spec["message"],
                                  spec["chunk"],
                                  tuple(spec.get("enc_overrides") or ()))
    rng = np.random.default_rng(spec["seed"])
    if spec["leg"] == "idle":
        # Truncate after frame 0 and append dead air.  The info sidecar still
        # says `frames` > 1, so the decoder keeps listening across the idle
        # exactly as a node waiting for a chunk that has not been sent yet --
        # which is the whole point: every acquisition logged in that window is
        # one the link would have paid for.
        idle = np.zeros(int(spec["idle_s"] * fs), np.float32)
        pcm = np.concatenate([pcm[:geom["frame0_end"]], idle])
    return _impair(pcm, fs, spec, geom, rng), fs, info, geom


def _events(log, geom, leg):
    """Acquisition events from the decoder log, split by where they landed.

    MF_WAKE and MF_PREAMBLE are the two ways the state machine commits to a
    frame boundary, and they are counted together: which door a false
    acquisition came through is a diagnostic, but the COST is identical (a
    frame window spent reading noise), so the headline metric must not care.
    """
    out = collections.Counter()
    for line in log.splitlines():
        name = line.split(" ", 1)[0] if line else ""
        # The two REJECT events are counted but never treated as acquisitions:
        # they are the decoder's own account of why a frame that WAS on the
        # wire did not land, and without them a 0% delivery row cannot be told
        # apart from "the marker was never detectable" (an edge/entropy
        # problem) versus "the marker was seen and refused" (a gate problem).
        # Those want opposite fixes.
        if name in ("MF_PRE_CADENCE_REJECT", "MF_PRE_REJECT"):
            out[name] += 1
            continue
        if name not in ("MF_WAKE", "MF_PREAMBLE"):
            continue
        out[name] += 1
        if leg == "idle" and "sym=" in line:
            try:
                sym = int(line.split("sym=")[1].split()[0])
            except (IndexError, ValueError):
                continue
            if sym >= geom["idle_start_sym"]:
                out["false_acq"] += 1
                out[f"false_{name}"] += 1
    return out


def _edge_stats(log, geom, leg):
    """Is the weak->strong edge the wake gate needs even PRESENT in this run?

    The wake word is only half of the wake gate.  The other half is an edge:
    the W word symbols must read STRONG and the symbol before them must read
    weak (css_decoder.py, wake-word resume block).  "Weak" is
    `snr_db < mf_min_snr_db`, an ABSOLUTE threshold on the matched filter's
    per-symbol peak-to-background ratio -- and that ratio does not go to zero
    in silence, it goes to whatever the filter reads on noise, which is a
    property of the FILTER, not of the channel.  If that floor sits above
    mf_min_snr_db, every symbol of the suspend gap reads STRONG, no edge ever
    occurs, and the wake word cannot fire no matter how clean it is.

    So the two numbers here are not decoration -- they are the mechanism:

      strong_frac    fraction of symbols in the quiet window reading STRONG.
                     Anything near 1.0 means the edge is gone.
      snr_floor_db   10th-percentile per-symbol MF SNR, i.e. where the
                     decoder's own statistic sits when nothing is being sent.
                     Compare it to mf_min_snr_db (default 8 dB): the gate has
                     to sit ABOVE this for a gap to read as quiet.

    Parsed from MIC_SN, which the decoder already prints once per symbol -- at
    0.5 s symbols that is two lines a second, so scanning them costs nothing
    next to the decode that produced them.
    """
    import numpy as np
    # MIC_SN is printed by BOTH decode paths, but only the MF one carries the
    # STRONG/weak verdict (the PLL path's line ends in sync=0/1 and is throttled
    # to ~1 Hz).  Scoring the PLL path here would report a clean 0% strong
    # fraction that means "this decoder does not classify symbols that way", not
    # "the edge is present" -- so refuse to answer instead.
    if "MF_SEGMENTS" not in log:
        return {"strong_frac": None, "snr_floor_db": None, "gate_db": None}
    # The gate is DERIVED per decode (measure_mf_noise_floor), so read it back
    # from the decoder's own line rather than restating a default here -- this
    # table used to say "default 8 dB" and would now be quoting a number the
    # decoder stopped using.
    gate = None
    for line in log.splitlines():
        if line.startswith("MF_NOISE_FLOOR") and "strong>=" in line:
            try:
                gate = float(line.split("strong>=")[1].split("dB")[0])
            except (IndexError, ValueError):
                pass
            break
    snrs, strong = [], 0
    lo = geom["idle_start_sym"] if leg == "idle" else 0
    for line in log.splitlines():
        if not line.startswith("MIC_SN"):
            continue
        try:
            sym = int(line.split("sym=")[1].split()[0])
            if sym < lo:
                continue
            snrs.append(float(line.split("snr=")[1].split("dB")[0]))
        except (IndexError, ValueError):
            continue
        strong += line.rstrip().endswith("STRONG")
    if not snrs:
        return {"strong_frac": None, "snr_floor_db": None,
                "gate_db": gate}
    return {"gate_db": gate,
            "strong_frac": strong / len(snrs),
            # p10 rather than the minimum: one symbol clipped by a click is not
            # the floor, and rather than the median because in the wake leg the
            # window also contains chirps at ~59 dB.
            "snr_floor_db": round(float(np.percentile(snrs, 10)), 1)}


def run_variant(spec):
    """Decode one variant.  Returns a plain (picklable) dict."""
    t0 = time.time()
    try:
        import css_regress
        pcm, fs, info, geom = _build(spec)
        res, log = css_regress.decode_offline(
            pcm, fs, info, overrides=spec.get("dec_overrides") or None)
        ev = _events(log, geom, spec["leg"])
        frames = geom["frames"]
        out = {
            "spec": spec, "ok": True,
            "status": res.get("status"),
            "frames_sent": frames,
            "frames_decoded": int(res.get("frames_decoded") or 0),
            "frames_errored": int(res.get("frames_errored") or 0),
            "wakes": int(ev.get("MF_WAKE", 0)),
            "pre_locks": int(ev.get("MF_PREAMBLE", 0)),
            "cadence_rejects": int(ev.get("MF_PRE_CADENCE_REJECT", 0)),
            "pre_rejects": int(ev.get("MF_PRE_REJECT", 0)),
            # Which decoder actually ran.  Recorded per variant rather than
            # assumed from symbol_dur, because a --dec-override can move the
            # dispatch threshold and a sweep that silently measured the wrong
            # decoder would be worse than one that did not run.
            "path": "mf" if "MF_SEGMENTS" in log else "pll",
            "elapsed": round(time.time() - t0, 1),
            **_edge_stats(log, geom, spec["leg"]),
        }
        if spec["leg"] == "wake":
            # Frame 0 is scored SEPARATELY, not excluded.  Two different things
            # are being priced and they load onto different frames:
            #
            #   deliver  resumed frames -- what a shorter per-frame MARKER
            #            costs.  Frame 0 is left out because it carries the
            #            full run-in in every candidate and would dilute the
            #            one difference under test.
            #   cold     frame 0 -- what shorter TRAINING costs.  The training
            #            chirps exist so the timing loop converges before the
            #            first Barker window; once the grid is locked, later
            #            frames inherit it.  So resumed-frame delivery is
            #            structurally blind to cutting training, and a sweep
            #            scored only on `deliver` will report training as free
            #            at every impairment level.  It did.
            # Read from the RESULT, not from an MF_FRAME log line: the PLL
            # decoder (symbol_dur below mf_longsym_min_dur, which is where the
            # production 0.12 s PHY lives) emits no MF_* lines at all, so a
            # log-scraped `cold` would read 0% everywhere on exactly the
            # geometry this metric was added to measure.
            out["cold"] = any(f.get("seq") == 0 and "error" not in f
                              for f in (res.get("frames") or []))
            # Resumed frames delivered, counted by subtracting frame 0 only if
            # it landed.  The obvious form, (frames_decoded - 1) / (frames - 1),
            # assumes frame 0 always decodes; when it does not, it reports a
            # negative delivery rate instead of clamping at 0%.
            _resumed = out["frames_decoded"] - (1 if out["cold"] else 0)
            out["deliver"] = (max(0, _resumed) / (frames - 1)
                              if frames > 1 else 0.0)
            out["text_ok"] = (res.get("decoded_text", "").rstrip("\x00")
                              == spec["message"])
        else:
            n = int(ev.get("false_acq", 0))
            out["false_acq"] = n
            out["false_wake"] = int(ev.get("false_MF_WAKE", 0))
            out["false_pre"] = int(ev.get("false_MF_PREAMBLE", 0))
            out["false_per_hr"] = n * 3600.0 / spec["idle_s"]
            # The cost of a false acquisition is not the event, it is the frame
            # window it burns: the decoder is in COLLECT and deaf for
            # frame_dur_s afterwards.  Expressed as a fraction of the idle
            # window, this is the probability that a chunk arriving at a random
            # moment finds the receiver already busy with a ghost.
            out["deaf_frac"] = min(1.0, n * geom["frame_dur_s"] / spec["idle_s"])
        return out
    except Exception as e:                      # noqa: BLE001
        return {"spec": spec, "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc()[-800:],
                "elapsed": round(time.time() - t0, 1)}


# ── the grid ─────────────────────────────────────────────────────────────────

def _seed(spec_key, trial, base):
    """Deterministic per-(variant, trial) seed.

    Hashed from the spec's identity rather than from its position in the grid
    so that restricting the sweep with --candidate or --level reproduces the
    SAME noise draws as the full run: a floor measured in a narrowed re-run has
    to be the same number, or narrowing the sweep to investigate a result would
    change the result.
    """
    h = 2166136261
    for ch in f"{spec_key}|{trial}|{base}".encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def build_specs(args, dec_overrides, message):
    legs = [args.leg] if args.leg else ["wake", "idle"]
    cands = args.candidate or list(CANDIDATES)
    fams = args.family or ["pink", "click", "drift", "wobble"]
    out = []
    for leg in legs:
        for cand in cands:
            for fam in fams:
                levels = {"pink": args.snr, "click": args.density,
                          "drift": args.ppm,
                          "wobble": args.wobble}[fam]
                for lv in levels:
                    for t in range(args.trials):
                        key = (f"{leg}|{cand}|{fam}|{lv}|{args.dur}"
                               f"|{args.wobble_hz:g}")
                        # Clock phase is drawn per TRIAL and, deliberately,
                        # independently of the candidate: every candidate sees
                        # the same set of clock realisations, so a difference
                        # between them is a difference in the modem and not in
                        # which phases each was handed.  --wobble-phase fixed
                        # pins it at 0 to reproduce pre-2026-09 numbers.
                        _ph = (0.0 if args.wobble_phase == "fixed" else
                               2.0 * math.pi
                               * ((_seed(f"wobphase|{fam}|{lv}", t,
                                         args.seed) % 100000) / 100000.0))
                        out.append({
                            "leg": leg, "candidate": cand, "family": fam,
                            "level": lv, "dur": args.dur, "trial": t,
                            "wobble_hz": args.wobble_hz,
                            "wobble_phase": _ph,
                            "gap_s": args.gap, "idle_s": args.idle,
                            "message": message, "chunk": args.chunk_size,
                            "dec_overrides": dec_overrides,
                            "seed": _seed(key, t, args.seed),
                        })
    return out


def parse_overrides(pairs):
    """--dec-override k=v ... -> a decoder config dict.

    Exists so a proposed FIX can be swept on the same grid as the thing it
    fixes.  The gates this sweep measures (mf_min_snr_db above all) are all
    plain css_decoder config keys, so raising one and re-running is a complete
    experiment -- without this the tool could only document a failure, not
    price the repair.  Values are typed by trying int, then float, then the
    JSON literals, then falling back to the string.
    """
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--dec-override needs key=value, got {p!r}")
        k, v = p.split("=", 1)
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k] = {"true": True, "false": False,
                      "null": None}.get(v.lower(), v)
    return out


def spec_label(s):
    unit = {"pink": "dB", "click": "/s", "drift": "ppm",
            "wobble": "ppm~"}[s["family"]]
    return (f"{s['leg']:<4} {s['candidate']:<14} "
            f"{s['family']}={s['level']:g}{unit}")


# ── aggregation ──────────────────────────────────────────────────────────────

def aggregate(results):
    """(leg, candidate, family, level) -> averaged metrics over trials."""
    by = collections.defaultdict(list)
    for r in results:
        if not r.get("ok"):
            continue
        s = r["spec"]
        by[(s["leg"], s["candidate"], s["family"], s["level"])].append(r)
    out = {}
    for k, rs in by.items():
        n = len(rs)
        rec = {"trials": n}
        sf = [r["strong_frac"] for r in rs if r.get("strong_frac") is not None]
        nf = [r["snr_floor_db"] for r in rs if r.get("snr_floor_db") is not None]
        rec["strong_frac"] = sum(sf) / len(sf) if sf else None
        rec["snr_floor_db"] = sum(nf) / len(nf) if nf else None
        gd = [r["gate_db"] for r in rs if r.get("gate_db") is not None]
        rec["gate_db"] = sum(gd) / len(gd) if gd else None
        rec["path"] = "/".join(sorted({r.get("path", "?") for r in rs}))
        if k[0] == "wake":
            rec["deliver"] = sum(r["deliver"] for r in rs) / n
            rec["cold"] = sum(int(r["cold"]) for r in rs) / n
            rec["text_ok"] = sum(int(r["text_ok"]) for r in rs) / n
            rec["wakes"] = sum(r["wakes"] for r in rs) / n
            rec["pre_locks"] = sum(r["pre_locks"] for r in rs) / n
            rec["cadence_rejects"] = sum(r["cadence_rejects"] for r in rs) / n
            rec["pre_rejects"] = sum(r["pre_rejects"] for r in rs) / n
        else:
            rec["false_per_hr"] = sum(r["false_per_hr"] for r in rs) / n
            rec["deaf_frac"] = sum(r["deaf_frac"] for r in rs) / n
            rec["false_wake"] = sum(r["false_wake"] for r in rs) / n
            rec["false_pre"] = sum(r["false_pre"] for r in rs) / n
        out[k] = rec
    return out


def floors(agg, candidate, deliver_thr, false_budget):
    """The two pink-noise floors for one candidate, in dB, each with a status.

    A floor is the LOWEST swept SNR at which the metric still holds with every
    higher SNR also holding -- not merely the lowest that happens to pass.  A
    single level passing below a failure is noise in the measurement (or a
    threshold effect worth looking at directly), and letting it set the floor
    would report a margin that is not there.

    Three outcomes, and conflating them is how a sweep lies about its own
    range.  A bare "no floor found" covers both a candidate that passed
    everything and one that failed everything -- opposite results:

      "ok"       failed somewhere inside the range; the floor is real.
      "below"    passed every level including the noisiest; the true floor is
                 BELOW the range, so the reported number is a lower bound on
                 how good it is.
      "never"    failed at the cleanest level swept; the true floor is ABOVE
                 the range and the candidate did not work anywhere in it.
    """
    def scan(leg, ok):
        lvls = sorted((k[3] for k in agg if k[0] == leg and k[1] == candidate
                       and k[2] == "pink"), reverse=True)
        if not lvls:
            return None, "none", lvls
        best = None
        for lv in lvls:
            if not ok(agg[(leg, candidate, "pink", lv)]):
                break
            best = lv
        if best is None:
            return None, "never", lvls
        return best, ("below" if best == lvls[-1] else "ok"), lvls

    dfloor, dstat, lv1 = scan("wake", lambda r: r["deliver"] >= deliver_thr)
    qfloor, qstat, lv2 = scan("idle", lambda r: r["false_per_hr"] <= false_budget)
    swept = lv1 + lv2
    # A "never" leg dominates the usable floor no matter what the other leg
    # did: the candidate has no working operating point in the swept range.
    if "never" in (dstat, qstat):
        usable, ustat = None, "never"
    else:
        cand_f = [x for x in (dfloor, qfloor) if x is not None]
        usable = max(cand_f) if cand_f else None
        ustat = ("below" if usable is not None and dstat == qstat == "below"
                 else "ok" if usable is not None else "none")
    return {
        "deliver_floor_db": dfloor, "deliver_status": dstat,
        "quiet_floor_db": qfloor, "quiet_status": qstat,
        "swept_to_db": min(swept) if swept else None,
        "usable_floor_db": usable, "usable_status": ustat,
        "binding": ("deliver" if dstat == "never" and qstat != "never" else
                    "quiet" if qstat == "never" and dstat != "never" else
                    "both" if dstat == qstat == "never" else
                    None if dfloor is None or qfloor is None else
                    "deliver" if dfloor > qfloor else
                    "quiet" if qfloor > dfloor else "both"),
    }


def airtime(candidate, dur, gap_s, message, chunk):
    """Static per-frame overhead for one candidate -- geometry, no decode."""
    pcm, fs, info, geom = _encode(candidate, dur, gap_s, message, chunk)
    del pcm
    pay = geom["frame_symbols"]
    res = geom["resume_syms"]
    return {
        "resume_syms": res,
        "resume_s": round(geom["resume_dur_s"], 2),
        "frame_s": round((res + pay) * geom["sps_eff"] / fs, 2),
        "frame_window_s": round(geom["frame_dur_s"], 2),
        "efficiency": round(pay / (res + pay), 3),
    }


# ── report ───────────────────────────────────────────────────────────────────

def report(agg, cands, args, air):
    lines = []
    p = lines.append

    # ── is this sweep even measuring the feature? ────────────────────────────
    # First, before any table: a wake candidate decoded on the PLL path was
    # never given the chance to fail on its own merits, and every number below
    # it in the report would be an artifact of a decoder that has no wake gate.
    # Say so at the top, where it cannot be read past.
    paths = {v.get("path") for v in agg.values()}
    if "pll" in paths:
        wake_on_pll = [c for c in cands if c.startswith("wake_") and any(
            k[1] == c and agg[k].get("path") == "pll" for k in agg)]
        p("")
        p("!! DECODER PATH: the PLL decoder ran for some or all of this sweep.")
        p("   The wake-word gate exists only in decode_longsym_mf, which runs")
        p("   above mf_longsym_min_dur (0.25 s).  Below that the encoder emits a")
        p("   wake word and the decoder never looks for one.")
        if wake_on_pll:
            p(f"   Affected candidates: {', '.join(wake_on_pll)} -- their rows")
            p("   below measure the absence of the feature, not its cost.")
            p("   Re-run with --dur 0.5, or with --force-mf to put this symbol")
            p("   rate through the MF decoder.")

    p("")
    p("AIRTIME  (geometry only -- what the cut is worth)")
    p(f"  {'candidate':<15} {'resume':>7} {'resume':>8} {'frame':>8} "
      f"{'payload':>8}  {'vs today':>9}")
    p(f"  {'':<15} {'syms':>7} {'s':>8} {'s':>8} {'frac':>8}")
    base = air.get("per_frame_full", {}).get("frame_s")
    for c in cands:
        a = air[c]
        save = (f"{(1 - a['frame_s'] / base) * 100:+6.1f}%"
                if base else "     --")
        p(f"  {c:<15} {a['resume_syms']:>7d} {a['resume_s']:>8.2f} "
          f"{a['frame_s']:>8.2f} {a['efficiency']:>8.3f}  {save:>9}")

    for leg, title, cols in (
            ("wake", "COLD acquisition of frame 0 -- prices the TRAINING "
                     "chirps (leg 'wake')", "cold"),
            ("wake", "DELIVERY of resumed chunks -- prices the per-frame "
                     "MARKER (leg 'wake')", "deliver"),
            ("idle", "FALSE acquisitions per hour of idle (leg 'idle')",
             "false_per_hr")):
        for fam in ("pink", "click", "drift", "wobble"):
            lvls = sorted({k[3] for k in agg if k[0] == leg and k[2] == fam},
                          reverse=(fam == "pink"))
            if not lvls:
                continue
            unit = {"pink": "dB SNR", "click": "clicks/s",
                    "drift": "ppm", "wobble": "peak ppm"}[fam]
            _hz = (f" @ {args.wobble_hz:g}Hz" if fam == "wobble" else "")
            p("")
            p(f"{title} -- {fam}{_hz} ({unit})")
            p("  " + f"{'candidate':<15}" +
              "".join(f"{lv:>9g}" for lv in lvls))
            for c in cands:
                row = [f"  {c:<15}"]
                for lv in lvls:
                    r = agg.get((leg, c, fam, lv))
                    if r is None:
                        row.append(f"{'-':>9}")
                    elif cols in ("deliver", "cold"):
                        row.append(f"{r[cols] * 100:>8.0f}%")
                    else:
                        row.append(f"{r['false_per_hr']:>9.1f}")
                p("".join(row))
            if leg == "wake" and cols == "deliver":
                # A 0% row has two completely different causes and the fix
                # differs: nothing was ever detected (no marker, no edge) vs a
                # marker that WAS detected and then refused by a gate.  The
                # reject counts are the only thing that separates them.
                worst = {}
                for c in cands:
                    rs = [agg[(leg, c, fam, lv)] for lv in lvls
                          if (leg, c, fam, lv) in agg]
                    worst[c] = (sum(r.get("cadence_rejects", 0) for r in rs),
                                sum(r.get("pre_rejects", 0) for r in rs))
                if any(sum(v) for v in worst.values()):
                    p("  (rejects across the sweep, cadence/confidence: "
                      + ", ".join(f"{c}={a:.0f}/{b:.0f}"
                                  for c, (a, b) in worst.items()) + ")")
            if leg == "idle":
                # deaf_frac saturates at 1.0, so say when the window was too
                # short for it to mean anything rather than printing a
                # confident 100%.
                fw = max(a.get("frame_window_s", 0) for a in air.values())
                if fw and args.idle < MIN_IDLE_FRAMES * fw:
                    p(f"  (idle window {args.idle:.0f}s is only "
                      f"{args.idle / fw:.1f} frame periods -- the deaf "
                      f"fraction below saturates; use --idle "
                      f"{MIN_IDLE_FRAMES * fw:.0f} or more)")
                p("  (deaf fraction of the idle window: "
                  + ", ".join(
                      f"{c}="
                      f"{max((agg[(leg, c, fam, lv)]['deaf_frac'] for lv in lvls if (leg, c, fam, lv) in agg), default=0) * 100:.0f}%"
                      for c in cands) + " at worst level)")

    # ── the mechanism behind whatever the tables just showed ─────────────────
    # Printed unconditionally, not only when something fails: a run where the
    # wake word never fires and one where it fires perfectly look identical in
    # the delivery table if you cannot see whether the edge it keys on existed.
    p("")
    p("EDGE  (the weak->strong transition the wake gate requires)")
    p(f"  {'level':<12} {'strong frac':>12} {'sym SNR floor':>14}"
      f"   vs mf_min_snr_db")
    # Idle leg only.  The wake leg's window contains the chirps themselves at
    # ~59 dB, so its strong fraction is high by construction and says nothing;
    # the idle window is genuinely quiet, so what reads STRONG there is exactly
    # what the gap would read as.
    for fam in ("pink", "click", "drift", "wobble"):
        lvls = sorted({k[3] for k in agg if k[2] == fam and k[0] == "idle"},
                      reverse=(fam == "pink"))
        for lv in lvls:
            rs = [agg[k] for k in agg
                  if k[0] == "idle" and k[2] == fam and k[3] == lv
                  and agg[k].get("strong_frac") is not None]
            if not rs:
                continue
            sfr = sum(r["strong_frac"] for r in rs) / len(rs)
            flo = sum(r["snr_floor_db"] for r in rs) / len(rs)
            unit = {"pink": "dB", "click": "/s", "drift": "ppm",
                    "wobble": "ppm~"}[fam]
            note = ("no edge -- wake cannot fire" if sfr > 0.98 else
                    "edge intermittent" if sfr > 0.80 else "edge present")
            p(f"  {fam}={lv:g}{unit:<7} {sfr * 100:>11.0f}% "
              f"{flo:>13.1f}dB   {note}")
    _g = [v["gate_db"] for v in agg.values() if v.get("gate_db") is not None]
    _gs = f"{sum(_g) / len(_g):.1f} dB" if _g else "mf_min_snr_db"
    p(f"  A gap symbol reads 'weak' only while its matched-filter SNR is below")
    p(f"  the strong gate, {_gs} for this geometry -- derived per decode from the")
    p("  filter's own noise floor (css_decoder.measure_mf_noise_floor), not a")
    p("  constant.  Where the floor above exceeds the gate, the gap reads as")
    p("  signal, the edge never occurs, and no wake word of any length can fire.")
    p("  Override with --dec-override mf_min_snr_db=N to bisect against it.")

    p("")
    p(f"TIPOVER  (deliver >= {args.deliver_thr * 100:.0f}% of resumed chunks; "
      f"false acquisitions <= {args.false_budget:g}/hr)")
    p(f"  {'candidate':<15} {'deliver':>9} {'quiet':>9} {'usable':>9} "
      f"{'binding':>9}   {'airtime':>8}")
    p(f"  {'':<15} {'floor dB':>9} {'floor dB':>9} {'floor dB':>9}")
    out = {}
    for c in cands:
        f = floors(agg, c, args.deliver_thr, args.false_budget)
        out[c] = {**f, **air[c]}

        def fmt(v, st):
            if st == "never":
                return "    FAILS"
            if st == "none" or v is None:
                return "        -"
            return f"{'<' if st == 'below' else ''}{v:>8.0f}"
        p(f"  {c:<15} {fmt(f['deliver_floor_db'], f['deliver_status'])} "
          f"{fmt(f['quiet_floor_db'], f['quiet_status'])} "
          f"{fmt(f['usable_floor_db'], f['usable_status'])} "
          f"{str(f['binding'] or '-'):>9}   {air[c]['frame_s']:>7.1f}s")
    p("")
    p("  'deliver floor' = lowest SNR where resumed chunks still land.")
    p("  'quiet floor'   = lowest SNR where the receiver is not chasing ghosts.")
    p("  usable = the worse of the two.  'FAILS' = failed even at the cleanest")
    p("  level swept, so there is no working point in this range at all.  '<N'")
    p("  = still passing at the noisiest level swept, so the true floor is")
    p("  below N -- widen --snr to find it.")

    p("")
    p(ota_plan(out, air, cands, agg, args.gap))
    return "\n".join(lines), out


def ota_plan(tip, air, cands, agg, gap_s):
    """Turn the tipover table into what to do on the air.

    The sweep's own numbers are synthetic and their absolute values should not
    be quoted at anyone; what survives contact with a real room is the RANKING
    and the mechanism.  So this prints a shortlist and, more importantly, what
    to instrument -- a field run that logs only decoded text cannot tell a
    missed wake from a false one, and those two failures want opposite fixes.
    """
    base = tip.get("per_frame_full") or {}
    bfloor = base.get("usable_floor_db")
    out = ["OTA PLAN  (what to take to the air)"]
    # Which regime was measured is the first thing to say, because the two
    # answer different questions and the default (a gap) is the harsher one.
    # A reader who runs this cold and sees every candidate fail should know
    # immediately that they measured the suspend-gap case, not that the tool
    # is broken.
    if gap_s > 0:
        out.append(f"  regime: SUSPEND GAP ({gap_s:g}s between frames) -- one "
                   f"decode session freewheeling")
        out.append("    across the pause.  This is what wake mode needs and is "
                   "the harder case.  A live")
        out.append("    node instead decodes each transmission separately "
                   "(css_modem_loopback.py header),")
        out.append("    so for the contiguous frames production actually sends, "
                   "re-run with --gap 0.")
    else:
        out.append("  regime: CONTIGUOUS frames, one transmission -- what a "
                   "live node sends today.")

    def worse_by(c):
        """dB the candidate gives up against today's scheme.

        None means "no comparable number": either the baseline or the candidate
        has no floor inside the swept range.  A candidate that FAILED
        everywhere is not "unmeasured" -- it is worse than any measured
        candidate, so it sorts last rather than being quietly grouped with the
        ones that simply have not been pinned down yet.
        """
        if tip[c].get("usable_status") == "never":
            return float("inf")
        f = tip[c].get("usable_floor_db")
        if f is None or bfloor is None:
            return None
        return f - bfloor

    def worst_case(c):
        """(lowest delivery, highest false/hr) over EVERY family, not just pink.

        The floors are pink-only, and on pink the candidates routinely tie --
        which left the ranking recommending the most aggressive cut on a
        tiebreak of airtime alone.  The families that actually separate them are
        the other ones: at 8 clicks/s the wake words spread from 67% delivery
        and 2.7 false/hr (w4) to 14% and 62.7/hr (w1), a 25x difference in
        false-wake rate that the pink floor cannot see at all.  So the tie is
        broken on the worst case anywhere in the grid.
        """
        dl = [agg[k]["deliver"] for k in agg
              if k[0] == "wake" and k[1] == c and "deliver" in agg[k]]
        fa = [agg[k]["false_per_hr"] for k in agg
              if k[0] == "idle" and k[1] == c and "false_per_hr" in agg[k]]
        return (min(dl) if dl else None, max(fa) if fa else None)

    ranked = []
    for c in cands:
        if c == "per_frame_full" or not air.get(c):
            continue
        saving = 1 - air[c]["frame_s"] / base.get("frame_s", air[c]["frame_s"])
        if saving <= 0:
            continue
        wd, wf = worst_case(c)
        # Sort key: robustness first (floor given up, then worst-case delivery,
        # then worst-case false rate), airtime only as the last tiebreak.  A
        # candidate that saves 1% more airtime and quadruples the false-wake
        # rate is not the better recommendation.
        ranked.append((worse_by(c), -(wd if wd is not None else 0),
                       wf if wf is not None else 0, -saving, c))
    # Sort by robustness given up first, airtime saved second: a candidate that
    # holds the same floor for less airtime is strictly better, and one that
    # saves more airtime by giving up margin is a trade the operator makes, not
    # one the sweep should make for them.
    ranked.sort(key=lambda t: ((999 if t[0] is None else t[0]),) + t[1:4])

    if bfloor is not None:
        out.append(f"  today (per_frame_full): usable floor {bfloor:.0f} dB, "
                   f"{base['frame_s']:.1f}s per packet")
    elif base.get("usable_status") == "never":
        out.append("  today (per_frame_full): FAILED at every level swept -- "
                   "the baseline is broken here,")
        out.append("    so no candidate below can be read as better or worse "
                   "than production.  Fix that first.")
    for cost, negwd, wf, negsave, c in ranked[:3]:
        d = ("no working point in the swept range" if cost == float("inf") else
             "margin unmeasured" if cost is None else
             "same pink floor" if cost <= 0 else f"gives up {cost:.0f} dB")
        verb = "ruled out" if cost == float("inf") else "try"
        out.append(f"  {verb:<9} {c:<15} -{-negsave * 100:.0f}% airtime "
                   f"({air[c]['frame_s']:.1f}s), {d}")
        # The worst cell anywhere in the grid, named, because that is where
        # these candidates differ and the floors above are pink-only.
        out.append(f"            worst cell across all families: "
                   f"{-negwd * 100:.0f}% delivery, {wf:.1f} false/hr")
    if not ranked:
        out.append("  nothing to try: no candidate both saves airtime and "
                   "scored -- widen --snr or check the path banner above")

    # A candidate that never fired at all is the most misleading kind of pass:
    # zero false acquisitions AND zero deliveries reads as "quiet" in one table
    # and "broken" in the other, and only the pairing says so.
    dead = [c for c in cands if c.startswith("wake_") and all(
        agg[k].get("wakes", 0) == 0 for k in agg if k[0] == "wake" and k[1] == c)]
    if dead:
        out.append(f"  NOT usable as-is: {', '.join(dead)} -- the wake word "
                   f"never fired once in the whole sweep.")
        out.append("    Check the EDGE table: if the strong fraction is ~100% "
                   "the gap reads as signal and")
        out.append("    no wake word can fire regardless of length.  Price the "
                   "fix with --dec-override mf_min_snr_db=N.")

    out.append("  instrument on the node before trusting any of this OTA:")
    out.append("    MF_WAKE / MF_PREAMBLE / MF_PRE_CADENCE_REJECT counts against"
               " packets ACTUALLY SENT.")
    out.append("    Delivered-text rate alone cannot separate a missed wake "
               "(marker not seen) from")
    out.append("    a false one (frame window burned on noise); the fixes are "
               "opposite.")
    out.append("    Log the per-symbol SNR floor during a known-idle minute -- "
               "that is the number the")
    out.append("    edge gate is compared against, and it is the one thing here"
               " a real room will change.")
    return "\n".join(out)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes (default: cores, capped at 32)")
    ap.add_argument("--leg", choices=("wake", "idle"),
                    help="run only one leg (default: both)")
    ap.add_argument("--candidate", action="append",
                    help=f"restrict to a candidate (repeatable): "
                         f"{', '.join(CANDIDATES)}")
    ap.add_argument("--family", action="append",
                    choices=("pink", "click", "drift", "wobble", "reverb"),
                    help="restrict to an impairment family (repeatable)")
    ap.add_argument("--snr", action="append", type=float,
                    help="pink-noise SNR level in dB (repeatable; "
                         f"default {list(PINK_SNRS)})")
    ap.add_argument("--density", action="append", type=float,
                    help="click density in clicks/s (repeatable; "
                         f"default {list(CLICK_DENSITIES)})")
    ap.add_argument("--ppm", action="append", type=float,
                    help="clock offset in ppm for the drift family "
                         f"(repeatable; default {list(DRIFT_PPMS)})")
    ap.add_argument("--wobble", action="append", type=float,
                    help="peak ppm of a wobbling clock for the wobble family "
                         f"(repeatable; default {list(WOBBLE_PPMS)})")
    ap.add_argument("--wobble-hz", type=float, default=WOBBLE_HZ,
                    help=f"wobble frequency in Hz (default {WOBBLE_HZ}).  Not "
                         f"a grid axis but a whole-run setting: it is the "
                         f"control for whether a training-length result is a "
                         f"real ranking or an artifact of where the run-in "
                         f"happens to land in the wobble cycle.  If the worst "
                         f"candidate MOVES when this changes, the ranking was "
                         f"phase, not training.")
    ap.add_argument("--wobble-phase", choices=("random", "fixed"),
                    default="random",
                    help="'random' (default) redraws the wobble's start phase "
                         "per trial, so trials sample the CLOCK as well as the "
                         "noise; 'fixed' pins it at 0, which is what every run "
                         "before 2026-09-02 did -- use it only to reproduce "
                         "those numbers, not to rank candidates.")
    ap.add_argument("--dur", type=float, default=DEF_DUR,
                    help=f"symbol duration in seconds (default {DEF_DUR})")
    ap.add_argument("--gap", type=float, default=DEF_GAP_S,
                    help=f"suspend gap between chunks (default {DEF_GAP_S}s)")
    ap.add_argument("--idle", type=float, default=DEF_IDLE_S,
                    help=f"idle-leg listening window (default {DEF_IDLE_S}s)")
    ap.add_argument("--chunk-size", type=int, default=DEF_CHUNK,
                    help=f"payload bytes per frame (default {DEF_CHUNK}, the "
                         f"modem PHY's value: one packet = one frame)")
    ap.add_argument("--frames", type=int, default=DEF_FRAMES,
                    help=f"frames per run; frames-1 are resumes "
                         f"(default {DEF_FRAMES})")
    ap.add_argument("--trials", type=int, default=DEF_TRIALS,
                    help=f"noise draws per point (default {DEF_TRIALS})")
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--deliver-thr", type=float, default=0.9,
                    help="resumed-chunk delivery rate that counts as working")
    ap.add_argument("--false-budget", type=float, default=2.0,
                    help="tolerable false acquisitions per hour of idle")
    ap.add_argument("--force-mf", action="store_true",
                    help="push the MF (long-symbol) decoder below its usual "
                         "0.25s dispatch threshold, so a production-rate "
                         "--dur 0.12 run still exercises the wake gate")
    ap.add_argument("--dec-override", action="append", metavar="K=V",
                    help="css_decoder config override applied to every decode "
                         "(repeatable), e.g. mf_min_snr_db=20 -- sweeps a "
                         "proposed fix on the same grid as the default")
    ap.add_argument("--quick", action="store_true",
                    help="1 trial, 3 SNRs, pink only, 60s idle -- a smoke test")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="seconds before a single variant is abandoned")
    ap.add_argument("--out", metavar="PATH",
                    help="write the full result JSON here")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every variant as it finishes")
    args = ap.parse_args()

    if args.quick:
        args.trials = 1
        args.snr = args.snr or [18.0, 9.0, 3.0]
        args.family = args.family or ["pink"]
        args.idle = 60.0
        args.frames = min(args.frames, 4)
    message = make_message(args.frames, args.chunk_size)
    args.snr = args.snr or list(PINK_SNRS)
    args.density = args.density or list(CLICK_DENSITIES)
    args.ppm = args.ppm or list(DRIFT_PPMS)
    args.wobble = args.wobble or list(WOBBLE_PPMS)
    for c in (args.candidate or []):
        if c not in CANDIDATES:
            ap.error(f"unknown candidate {c!r}; known: {', '.join(CANDIDATES)}")

    try:
        dec_overrides = parse_overrides(args.dec_override)
    except ValueError as e:
        ap.error(str(e))
    if args.force_mf:
        # Half the symbol duration, so the dispatch is unambiguous rather than
        # sitting on the threshold.  An explicit --dec-override still wins:
        # this is a convenience, not a lock.
        dec_overrides.setdefault("mf_longsym", True)
        dec_overrides.setdefault("mf_longsym_min_dur", args.dur / 2.0)

    specs = build_specs(args, dec_overrides, message)
    # Worker count is bounded by MEMORY, not cores.  Each worker holds the
    # encoded reference, an impaired copy of it and the decoder's own
    # workspace, and the idle leg's streams are the length of the idle window;
    # a 48-worker run on a 15-minute idle window lost every worker at once and
    # then sat there, because the parent was still waiting on futures that
    # could no longer complete.  16 is comfortable at the default geometry.
    jobs = args.jobs or min(16, len(os.sched_getaffinity(0)))
    jobs = max(1, min(jobs, len(specs)))
    cands = [c for c in CANDIDATES if c in {s["candidate"] for s in specs}]

    print(f"CSS WAKE-PREAMBLE SWEEP -- {len(specs)} variants on {jobs} workers")
    print(f"  candidates : {', '.join(cands)}")
    print(f"  geometry   : symbol_dur={args.dur}s chunk={args.chunk_size}B "
          f"frames={args.frames} gap={args.gap}s idle={args.idle}s "
          f"trials={args.trials}")
    if dec_overrides:
        print(f"  decoder    : {dec_overrides}")
    print("")

    import concurrent.futures as cf
    t0 = time.time()
    results = []
    # NO max_tasks_per_child.  It is the obvious way to bound a worker's
    # memory high-water mark, css_sm_sweep.py uses it, and on this Python
    # (3.13, fork start method) it HANGS THE POOL: every worker dies at the
    # first replacement wave and the parent waits on futures that can never
    # complete.  Observed twice here at exactly workers x 8 completions --
    # 375/420 at 48 workers, 125/420 at 16 -- and reproduced in isolation with
    # 80 tasks of `return i*i`, which wedges with the option and finishes
    # instantly without it.  Worker lifetime is bounded by _ENC's LRU instead;
    # the per-task arrays are transient and freed anyway.
    #
    # The `with` block still matters for teardown -- an orphaned worker keeps
    # the inherited stdout pipe open and leaves a finished run looking hung.
    # A dead pool must not look like a slow one.  as_completed's timeout is the
    # only thing standing between "the workers were killed" and a run that
    # waits forever: ProcessPoolExecutor raises BrokenProcessPool when it
    # notices, but it does not always notice.  Budget the wall clock from the
    # work actually queued -- variants per worker, times the per-variant
    # timeout, plus slack for startup -- so an abandoned run reports what it
    # finished instead of hanging.
    wall_budget = args.timeout * (-(-len(specs) // jobs)) + 120.0
    try:
        with cf.ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(run_variant, s): s for s in specs}
            done = 0
            for fut in cf.as_completed(futs, timeout=wall_budget):
                done += 1
                try:
                    r = fut.result(timeout=args.timeout)
                except Exception as e:          # noqa: BLE001
                    r = {"spec": futs[fut], "ok": False,
                         "error": f"{type(e).__name__}: {e}"}
                results.append(r)
                if not r.get("ok"):
                    print(f"  ERR  {spec_label(r['spec']):<44} {r['error']}",
                          flush=True)
                elif args.verbose:
                    s = r["spec"]
                    detail = (f"deliver={r['deliver'] * 100:3.0f}% "
                              f"wakes={r['wakes']} pre={r['pre_locks']}"
                              if s["leg"] == "wake" else
                              f"false={r['false_acq']} "
                              f"({r['false_per_hr']:.1f}/hr) "
                              f"wake={r['false_wake']} pre={r['false_pre']}")
                    print(f"       {spec_label(s):<44} t{s['trial']} "
                          f"{detail}  [{r['elapsed']}s]", flush=True)
                if done % 25 == 0 and not args.verbose:
                    print(f"  ... {done}/{len(specs)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted -- reporting what finished")
    except (cf.TimeoutError, cf.BrokenExecutor) as e:
        # Partial results are still worth reporting -- the floors just cover
        # fewer levels -- but say loudly that the grid is incomplete, because a
        # missing level and a level that failed look the same in the tables.
        abandoned = len(specs) - len(results)
        print(f"\n*** POOL LOST ({type(e).__name__}) after {done}/{len(specs)} "
              f"-- {abandoned} variants abandoned")
        print("    Usually memory: lower --jobs or --idle and re-run.")
        print("    The tables below cover only what finished.")

    errs = [r for r in results if not r.get("ok")]
    agg = aggregate(results)
    air = {c: airtime(c, args.dur, args.gap, message, args.chunk_size)
           for c in cands}
    text, tip = report(agg, cands, args, air)
    print(text)
    print(f"\n{len(results) - len(errs)} decoded, {len(errs)} errored, "
          f"{time.time() - t0:.0f}s wall")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({
                "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "args": {k: v for k, v in vars(args).items()},
                "airtime": air,
                "tipover": tip,
                "levels": [{"leg": k[0], "candidate": k[1], "family": k[2],
                            "level": k[3], **v} for k, v in sorted(
                                agg.items(), key=lambda kv: str(kv[0]))],
                "errors": [{"spec": r["spec"], "error": r.get("error")}
                           for r in errs],
            }, fh, indent=1)
        print(f"wrote {args.out}")

    # Non-zero when a candidate could not be scored at all: that is the
    # actionable outcome (the sweep did not answer the question) as distinct
    # from a candidate that scored badly, which is an answer.
    unscored = [c for c in cands if not any(k[1] == c for k in agg)]
    if unscored:
        print(f"UNSCORED: {', '.join(unscored)}")
    return 1 if (unscored or errs) else 0


if __name__ == "__main__":
    sys.exit(main())
