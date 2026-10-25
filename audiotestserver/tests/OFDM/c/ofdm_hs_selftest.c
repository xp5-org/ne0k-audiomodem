/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */

/* ofdm_hs_selftest.c -- host-side smoke test for ofdm_hs.c.
 *
 * Drives two ofdm_hs handles (initiator + responder) against each other
 * directly, with no audio/DSP layer -- messages are pack/unpack'd through
 * ofdm_hs_pack_msg/unpack_msg (proving the wire framing round-trips) and
 * "probes" are just opaque BAT byte blobs handed straight across.
 *
 * Build: cc -std=c89 -pedantic -Wall -Wextra ofdm_hs.c ofdm_hs_selftest.c \
 *           -o /tmp/ofdm_hs_selftest && /tmp/ofdm_hs_selftest
 */
#include <stdio.h>
#include <string.h>
#include "ofdm_hs.h"

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        failures++; \
    } \
} while (0)

static const unsigned char BAT_I[] = {0x01, 0x02, 0x03, 0x02, 0x01};
static const unsigned char BAT_R[] = {0x02, 0x02, 0x01, 0x01, 0x02, 0x02};

static const unsigned char *empty_body(void);

/* Full happy-path exchange, driven purely by feeding each side's emitted
 * actions to the other -- mirrors the protocol docstring in ofdm_hs.h. */
static void test_happy_path(int skip_fin)
{
    ofdm_hs ini, resp;
    ofdm_hs_action a[OFDM_HS_MAX_ACTIONS], b[OFDM_HS_MAX_ACTIONS];
    int na, nb, i;
    double now = 0.0;

    ofdm_hs_init(&ini, OFDM_HS_ROLE_INITIATOR, 2, 60.0, 60.0, -1.0, skip_fin);
    ofdm_hs_init(&resp, OFDM_HS_ROLE_RESPONDER, 2, 60.0, 60.0, -1.0, skip_fin);

    na = ofdm_hs_start(&ini, now, a, OFDM_HS_MAX_ACTIONS);
    nb = ofdm_hs_start(&resp, now, b, OFDM_HS_MAX_ACTIONS);
    CHECK(na == 1 && a[0].kind == OFDM_HS_ACT_SEND_PROBE,
          "initiator starts by sending its probe");
    CHECK(nb == 0, "responder just starts listening");
    CHECK(ini.state == OFDM_HS_I_WAIT_PROBE, "initiator -> i_wait_probe");
    CHECK(resp.state == OFDM_HS_R_WAIT_PROBE, "responder -> r_wait_probe");

    /* I's probe reaches R; R has nothing to send yet (waits for I's probe
     * too), so R measures it and stays quiet until it ALSO transmits its
     * own probe, per the protocol (R only reports once it has sent its
     * own probe and I has (implicitly) received it). Drive R's own probe
     * send explicitly, as the DSP binding would on its own turn. */
    nb = ofdm_hs_heard_probe(&resp, BAT_I, sizeof BAT_I, now, b,
                             OFDM_HS_MAX_ACTIONS);
    CHECK(nb == 2 && b[0].kind == OFDM_HS_ACT_SEND_PROBE
          && b[1].kind == OFDM_HS_ACT_SEND_MSG
          && b[1].mtype == OFDM_HS_MSG_BAT_REPORT,
          "responder answers a heard probe with its own probe + REPORT");
    CHECK(resp.state == OFDM_HS_R_WAIT_ACK, "responder -> r_wait_ack");

    /* I hears R's probe (from b[0]) -- feeds the same BAT_R bytes I would
     * have measured. */
    na = ofdm_hs_heard_probe(&ini, BAT_R, sizeof BAT_R, now, a,
                             OFDM_HS_MAX_ACTIONS);
    CHECK(na == 0, "initiator measuring R's probe emits nothing yet");
    CHECK(ini.state == OFDM_HS_I_WAIT_BAT, "initiator -> i_wait_bat");

    /* I hears R's REPORT (b[1]): unpack it through the real wire framing. */
    {
        unsigned char wire[64];
        int wlen = ofdm_hs_pack_msg(wire, sizeof wire, b[1].mtype,
                                    b[1].proof_crc, b[1].body, b[1].body_len);
        int mtype;
        unsigned proof;
        const unsigned char *body;
        size_t body_len;
        CHECK(wlen > 0, "REPORT packs onto the wire");
        CHECK(ofdm_hs_unpack_msg(wire, (size_t)wlen, &mtype, &proof, &body,
                                 &body_len) == 0, "REPORT round-trips CRC-clean");
        na = ofdm_hs_heard_msg(&ini, mtype, proof, body, body_len, now, a,
                               OFDM_HS_MAX_ACTIONS);
    }
    CHECK(na == 1 && a[0].kind == OFDM_HS_ACT_SEND_MSG
          && a[0].mtype == OFDM_HS_MSG_BAT_ACK,
          "initiator answers REPORT with BAT_ACK");
    CHECK(ini.state == OFDM_HS_I_WAIT_GO, "initiator -> i_wait_go");

    /* R hears I's ACK -> sends GO, enters r_linger. */
    nb = ofdm_hs_heard_msg(&resp, a[0].mtype, a[0].proof_crc, a[0].body,
                           a[0].body_len, now, b, OFDM_HS_MAX_ACTIONS);
    CHECK(nb == 1 && b[0].kind == OFDM_HS_ACT_SEND_MSG
          && b[0].mtype == OFDM_HS_MSG_GO, "responder answers ACK with GO");
    CHECK(resp.state == OFDM_HS_R_LINGER, "responder -> r_linger");

    /* I hears GO with a matching proof -> done (plus FIN unless skipped). */
    na = ofdm_hs_heard_msg(&ini, b[0].mtype, b[0].proof_crc, b[0].body,
                           b[0].body_len, now, a, OFDM_HS_MAX_ACTIONS);
    if (skip_fin) {
        CHECK(na == 1 && a[0].kind == OFDM_HS_ACT_DONE,
              "skip_fin: GO finishes without a FIN action");
    } else {
        CHECK(na == 2 && a[0].kind == OFDM_HS_ACT_SEND_MSG
              && a[0].mtype == OFDM_HS_MSG_FIN && a[1].kind == OFDM_HS_ACT_DONE,
              "GO with valid proof sends FIN then finishes");
    }
    CHECK(ini.done == 1 && ini.state == OFDM_HS_DONE, "initiator reaches done");
    /* R's REPORT carries R's measurement of I's probe (BAT_I) -- that IS
     * I's tx table (my_tx_bat). I's own peer_tx_bat is what it measured of
     * R's probe directly (BAT_R), never carried on the wire at all. */
    CHECK(memcmp(a[na - 1].my_tx_bat, BAT_I, sizeof BAT_I) == 0
          && a[na - 1].my_tx_bat_len == sizeof BAT_I,
          "initiator's negotiated my_tx_bat is R's REPORT of I's probe");
    CHECK(memcmp(a[na - 1].peer_tx_bat, BAT_R, sizeof BAT_R) == 0
          && a[na - 1].peer_tx_bat_len == sizeof BAT_R,
          "initiator's negotiated peer_tx_bat is what it measured of R's probe");

    if (!skip_fin) {
        /* R hears FIN -> also done, before its linger would have expired. */
        i = ofdm_hs_heard_msg(&resp, a[0].mtype, a[0].proof_crc, a[0].body,
                              a[0].body_len, now, b, OFDM_HS_MAX_ACTIONS);
        CHECK(i == 1 && b[0].kind == OFDM_HS_ACT_DONE,
              "responder finishes on FIN");
    } else {
        /* No FIN: R's linger must still expire into success (two-generals
         * compromise -- see ofdm_hs.h / the r_linger tick case). */
        i = ofdm_hs_tick(&resp, resp.deadline + 0.01, b, OFDM_HS_MAX_ACTIONS);
        CHECK(i == 1 && b[0].kind == OFDM_HS_ACT_DONE,
              "responder's linger expiry assumes GO landed (skip_fin path)");
    }
    CHECK(resp.done == 1, "responder reaches done");
}

/* A GO with the wrong proof (peer holds a stale BAT_R) must RETRY, not
 * finish -- and the retry must eventually succeed once given the right
 * proof. */
static void test_go_proof_mismatch_retries(void)
{
    ofdm_hs ini;
    ofdm_hs_action a[OFDM_HS_MAX_ACTIONS];
    int na;

    ofdm_hs_init(&ini, OFDM_HS_ROLE_INITIATOR, 2, 60.0, 60.0, -1.0, 0);
    ofdm_hs_start(&ini, 0.0, a, OFDM_HS_MAX_ACTIONS);
    ofdm_hs_heard_probe(&ini, BAT_R, sizeof BAT_R, 0.0, a,
                       OFDM_HS_MAX_ACTIONS); /* -> i_wait_bat */
    ofdm_hs_heard_msg(&ini, OFDM_HS_MSG_BAT_REPORT, 0u, BAT_R, sizeof BAT_R,
                     0.0, a, OFDM_HS_MAX_ACTIONS); /* -> i_wait_go */

    na = ofdm_hs_heard_msg(&ini, OFDM_HS_MSG_GO, 0xDEAD, empty_body(), 0,
                          0.0, a, OFDM_HS_MAX_ACTIONS);
    CHECK(na == 1 && a[0].kind == OFDM_HS_ACT_SEND_MSG
          && a[0].mtype == OFDM_HS_MSG_RETRY,
          "GO with wrong proof triggers RETRY, not done");
    CHECK(ini.state == OFDM_HS_I_WAIT_GO, "stays in i_wait_go after a RETRY");
    CHECK(ini.done == 0, "not done after a proof mismatch");

    {
        unsigned want = ofdm_hs_crc16(BAT_R, sizeof BAT_R);
        na = ofdm_hs_heard_msg(&ini, OFDM_HS_MSG_GO, want, empty_body(), 0,
                              1.0, a, OFDM_HS_MAX_ACTIONS);
    }
    CHECK(na >= 1 && a[na - 1].kind == OFDM_HS_ACT_DONE,
          "GO with the correct proof on retry finishes");
}

/* Fixed empty-body helper so the test file needs no extra static array. */
static const unsigned char *empty_body(void)
{
    static const unsigned char e[1] = {0};
    return e;
}

/* Timeout with no reply exhausts max_retries and fails, with the fail_state
 * naming the wait state the timeout actually happened in. */
static void test_timeout_exhausts_and_fails(void)
{
    ofdm_hs ini;
    ofdm_hs_action a[OFDM_HS_MAX_ACTIONS];
    int n, i;
    double now = 0.0;

    ofdm_hs_init(&ini, OFDM_HS_ROLE_INITIATOR, 2, 10.0, 10.0, -1.0, 0);
    ofdm_hs_start(&ini, now, a, OFDM_HS_MAX_ACTIONS);
    CHECK(ini.state == OFDM_HS_I_WAIT_PROBE, "starts in i_wait_probe");

    for (i = 0; i < 2; i++) {
        now += 11.0;
        n = ofdm_hs_tick(&ini, now, a, OFDM_HS_MAX_ACTIONS);
        CHECK(n == 1 && a[0].kind == OFDM_HS_ACT_SEND_PROBE,
              "each retry re-sends the probe");
        CHECK(ini.failed == 0, "not failed until retries exhausted");
    }
    now += 11.0;
    n = ofdm_hs_tick(&ini, now, a, OFDM_HS_MAX_ACTIONS);
    CHECK(n == 1 && a[0].kind == OFDM_HS_ACT_FAIL, "final timeout fails");
    CHECK(a[0].fail_reason == OFDM_HS_FAIL_TIMEOUT, "reason is TIMEOUT");
    CHECK(a[0].fail_state == OFDM_HS_I_WAIT_PROBE,
          "fail_state names i_wait_probe, not the post-fail state");
    CHECK(ini.failed == 1 && ini.state == OFDM_HS_FAILED, "handle is failed");
}

/* hold_off must push the deadline out, not just reset it to the full
 * interval -- confirms it is additive. */
static void test_hold_off_extends_deadline(void)
{
    ofdm_hs ini;
    ofdm_hs_action a[OFDM_HS_MAX_ACTIONS];
    double before;

    ofdm_hs_init(&ini, OFDM_HS_ROLE_INITIATOR, 2, 10.0, 10.0, -1.0, 0);
    ofdm_hs_start(&ini, 0.0, a, OFDM_HS_MAX_ACTIONS);
    before = ini.deadline;
    ofdm_hs_hold_off(&ini, 5.0);
    CHECK(ini.deadline == before + 5.0, "hold_off adds to the deadline");
}

int main(void)
{
    test_happy_path(0);
    test_happy_path(1);
    test_go_proof_mismatch_retries();
    test_timeout_exhausts_and_fails();
    test_hold_off_extends_deadline();
    if (failures) {
        printf("%d check(s) FAILED\n", failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
