/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* css_enc.h — CSS transmit chain, freestanding C89.
 *
 * The encode-side counterpart to css_mf.c: everything between a payload chunk
 * and the bit sequence that drives the chirp emitter, plus the chirp itself.
 * Port of the pure functions in tests/CSS/css_encoder.py and the frame loop
 * of its encode(), not the WAV writing, JSON sidecar, or config defaults.
 *
 * Freestanding, like the rest of tests/CSS/c: no malloc, no stdio, no
 * globals, caller-owned buffers throughout. Every routine here is O(bits)
 * with tiny constants, so unlike the matched filter there is no workspace
 * object to carve; the caller sizes its own arrays from css_enc_geometry().
 *
 * Reed-Solomon is not linked here. The chain is
 *
 *     css_enc_inner()  ->  seq | data | crc32
 *     [ rs255_encode() ]   optional, caller's call
 *     css_enc_bits()   ->  bits -> diff -> conv -> interleave
 *
 * so a TX-only build at rs_t=0 pulls in no RS code, and a build that does
 * want RS uses tests/c/rs255.c rather than a second copy here.
 *
 * C89: every declaration sits at the top of its block, before any statement
 * in that block. OpenWatcom and several embedded C89 toolchains reject a
 * mid-block declaration outright; `make c89check` enforces it with
 * -Wdeclaration-after-statement.
 */
#ifndef CSS_ENC_H
#define CSS_ENC_H

#ifdef __cplusplus
extern "C" {
#endif

/* Phase precision. The chirp phase argument reaches ~2*pi*f_max*duration
 * radians, about 113,000 at f_max=6 kHz over a 3 s symbol, so its float32
 * round-off is a real fraction of a radian by the end of a long sweep
 * (measured: up to 0.013 in sample amplitude at 3 s).
 *
 * What matters is the correlation between the transmitted chirp and the
 * decoder's template. Across every geometry this repo uses (0.25-3.0 s,
 * 44.1/48 kHz, 400-6000 and 800-4000 Hz) the float32-phase chirp correlates
 * with the float64 reference at >= 0.99999, under 0.001 dB of matched-filter
 * loss, reverified by regression/css_enc_regress.py on every run.
 *
 * float is the default. -DCSS_ENC_F64_PHASE selects double, reproducing
 * numpy's float64-phase chirp to float32 round-off, so the equivalence
 * harness can grade the port itself apart from the precision choice (the
 * same two-stage split ofdm_fft.h uses). sin() is always the C89 double
 * entry point: sinf() is C99, and the chirp is built once per run, so the
 * widening costs nothing.
 */
#ifdef CSS_ENC_F64_PHASE
typedef double css_phase;
#else
typedef float  css_phase;
#endif

/* Frame geometry, derived once from the config and then read-only.
 * Mirrors the quantities css_encoder.encode() computes and writes into the
 * info.json sidecar, under the same names, so a mismatch between the two is a
 * visible field-by-field difference rather than a silent desync. */
typedef struct {
    int chunk_size;         /* payload bytes per frame                      */
    int rs_parity;          /* rs_t * 2, or 0                               */
    int base_bits;          /* (1 + chunk_size + 4 + rs_parity) * 8         */
    int conv_code;          /* 1 = k7r12 (rate 1/2, K=7), 0 = none          */
    int diff_enc;           /* 1 = differential encode before conv          */
    int interleave_rows;    /* 0 or 1 = disabled, else the block height     */
    int conv_bits;          /* base_bits, or (base_bits + 6) * 2            */
    int frame_symbols;      /* conv_bits, padded up to a whole block        */
} css_enc_cfg;

/* Fills `g` from the four config knobs.  chunk_size >= 1, rs_t >= 0,
 * conv_code 0/1, interleave_rows 0 or >= 1.  Returns 0, or -1 on bad args.
 * frame_symbols is the number of chirps the frame body occupies and is the
 * size every bit buffer below must be able to hold. */
int css_enc_geometry(css_enc_cfg *g, int chunk_size, int rs_t,
                     int conv_code, int diff_enc, int interleave_rows);

/* CRC-32 (IEEE 802.3, reflected, poly 0xEDB88320, init/final 0xFFFFFFFF),
 * byte-identical to zlib.crc32, which the Python encoder uses.
 *
 * Bitwise rather than table-driven: the 1 KB table is a poor trade on a part
 * where flash is the scarce resource, and this runs over 1 + chunk_size
 * bytes once per frame, where eight shifts a byte is unmeasurable. */
unsigned long css_crc32(const unsigned char *buf, int n);

/* seq | data[chunk_size] | crc32(seq|data), MSB-first, into `out`.
 * `out` needs 1 + chunk_size + 4 bytes. Returns bytes written, or -1.
 * `data` shorter than chunk_size must be zero-padded by the caller first, as
 * the Python encoder pads its final short chunk: a frame with fewer chirps
 * than frame_symbols shifts the decoder's CRC window and fails every time. */
int css_enc_inner(const css_enc_cfg *g, int seq, const unsigned char *data,
                  unsigned char *out);

/* bits -> diff -> conv -> interleave, the transmit bit pipeline.
 *
 * `frame` is the inner frame AFTER any RS encoding, `nframe` its length in
 * bytes.  `bits` receives frame_symbols one-bit-per-byte values (0/1), the
 * chirp directions to emit in order.  `diff_ref` is the last preamble bit,
 * the absolute polarity reference the decoder recovers from sync.
 * Returns bits written (== g->frame_symbols), or -1 bad args, -2 cap too
 * small. */
int css_enc_bits(const css_enc_cfg *g, const unsigned char *frame, int nframe,
                 int diff_ref, unsigned char *bits, int cap,
                 unsigned char *scratch, int scap);

/* Bytes of scratch css_enc_bits needs for this geometry. */
int css_enc_scratch(const css_enc_cfg *g);

/* The individual stages, exposed so the equivalence harness can grade each
 * against its Python original separately rather than only the whole
 * pipeline. All are safe to call with out == in. Each returns the number of
 * bits written, or -1 on bad args. */
int css_bytes_to_bits(const unsigned char *bytes, int n, unsigned char *bits);
int css_conv_encode_k7(const unsigned char *bits, int n, unsigned char *out);
int css_diff_encode(const unsigned char *bits, int n, int ref,
                    unsigned char *out);
/* Interleave writes rows*ceil(n/rows) bits and needs `out` != `bits`. */
int css_interleave(const unsigned char *bits, int n, int rows,
                   unsigned char *out);

/* Linear FM sweep f0 -> f1 over n samples spanning `duration` seconds,
 * rectangular envelope. n is the caller's int(fs * duration).
 *
 * `duration`, not fs, is the third argument: the Python original builds its
 * time base as linspace(0, duration, n, endpoint=False), i.e.
 * t[i] = i * duration / n, which equals i / fs only when fs * duration is a
 * whole number. At fs=44100 with a 0.3 s symbol it is not; taking i / fs
 * instead would stretch the sweep by a fraction of a sample per symbol.
 *
 * Must stay identical to the decoder's template (css_mf_init's up_ref/dn_ref):
 * the MF correlates against the actually-transmitted waveform, so an
 * envelope or phase-convention difference between the two shows up only as
 * correlation loss. Returns 0, or -1 on bad args. */
int css_enc_chirp(float *out, int n, double f0, double f1, double duration);

#ifdef __cplusplus
}
#endif
#endif
