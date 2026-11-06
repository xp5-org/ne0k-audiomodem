/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_dx.h -- one end of a full-duplex link: link-frame codec, freestanding C89.
 *
 * Port of the data plane of modemgui's OfdmLink (pack_frame / unpack_frame,
 * _frame_audio, _decode_from) onto the C cores.  An end owns TWO independent
 * OFDM configs: the sub-band it transmits on and the sub-band it listens to.
 * Nothing here is duplex-specific beyond that -- the same codec serves a
 * half-duplex link by passing the same config twice.
 *
 *   TX:  ofdm_dx_tx_frame()   (seq, ack, data) -> samples
 *   RX:  ofdm_dx_rx_frame()   samples -> (seq, ack, data)
 *
 * LINK FRAME (wire contract, matches ofdm_link.pack_frame):
 *     seq(1) ack(1) len(1) crc16(2, big-endian) data(len)
 * crc16 covers seq, ack, len and data.  The frame is carried as however many
 * OFDM frames of `chunk_bytes` it takes (last one NUL padded); each OFDM frame
 * carries its chunk index as the modulator sequence, so chunks may arrive in
 * any order and duplicates are harmless.
 *
 * Same contract as the other cores: no malloc, no stdio, no globals, one
 * caller-owned workspace.
 */
#ifndef OFDM_DX_H
#define OFDM_DX_H
#include <stddef.h>
#include "ofdm_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OFDM_DX_HDR        5      /* seq ack len crc16 */
#define OFDM_DX_MAX_DATA   255
#define OFDM_DX_MAX_FRAME  (OFDM_DX_HDR + OFDM_DX_MAX_DATA)
#define OFDM_DX_MAX_STARTS 16     /* frame starts examined per rx call */

typedef struct ofdm_dx ofdm_dx;

typedef struct {
    ofdm_cfg    tx_cfg;           /* the band we transmit on  */
    ofdm_cfg    rx_cfg;           /* the band we listen to    */
    const int  *tx_bat;           /* tx_cfg.n_data_carriers entries; copied */
    const int  *rx_bat;           /* rx_cfg.n_data_carriers entries; copied */
} ofdm_dx_params;

typedef struct {
    int    seq, ack, len;
    int    n_chunks;              /* OFDM frames the link frame spanned     */
    int    n_found;               /* OFDM frames seen (incl. undecodable)   */
    int    n_bad;                 /* of those, how many failed RS/sequence  */
    int    errata;                /* RS symbols corrected                   */
    double evm;                   /* mean linear EVM over decoded frames    */
    int    first_start;           /* sample offset of the first frame       */
} ofdm_dx_rxinfo;

/* Bytes of caller memory.  0 if the parameters would be rejected. */
size_t ofdm_dx_workspace(const ofdm_dx_params *p);

/* Errors: -1 bad args, -2 memlen too small, -3 a sub-core failed to init,
 *         -4 geometry unusable (e.g. a frame spanning too many chunks). */
int ofdm_dx_init(ofdm_dx **dx, void *mem, size_t memlen, const ofdm_dx_params *p);

/* Largest `len` that fits one link frame without exceeding `max_chunks` OFDM
 * frames (and the 255-byte wire limit).  Mirrors OfdmLink._capacity, which
 * derives max_chunks from the airtime window. */
int ofdm_dx_capacity(const ofdm_dx *dx, int max_chunks);

/* Samples ofdm_dx_tx_frame() will write for an n-byte payload.  <0 on error. */
long ofdm_dx_tx_samples(const ofdm_dx *dx, int n);

/* Link frame -> audio at RAW modulator level (levelling is the caller's).
 * Returns samples written; -1 bad args, -2 out_cap too small, -3 encode failed. */
long ofdm_dx_tx_frame(ofdm_dx *dx, int seq, int ack, const unsigned char *data,
                      int n, double *out, long out_cap);

/* Look for one complete link frame in rx[0..n).  Returns 1 and fills the
 * outputs on success, 0 if nothing decodable, -1 on bad args.  `data` needs
 * OFDM_DX_MAX_DATA bytes.  `info` may be NULL. */
int ofdm_dx_rx_frame(ofdm_dx *dx, const double *rx, int n,
                     unsigned char *data, ofdm_dx_rxinfo *info);

/* Geometry for sizing capture buffers and timers. */
long ofdm_dx_rx_frame_samples(const ofdm_dx *dx);   /* one RX OFDM frame */
const ofdm_cfg *ofdm_dx_tx_cfg(const ofdm_dx *dx);
const ofdm_cfg *ofdm_dx_rx_cfg(const ofdm_dx *dx);

/* The link-frame CRC, exposed so a regression can pin it to the reference. */
unsigned ofdm_dx_crc16(const unsigned char *data, size_t len);

#ifdef __cplusplus
}
#endif
#endif
