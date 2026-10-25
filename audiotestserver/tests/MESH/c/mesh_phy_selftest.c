/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 XP5-ORG */
/* mesh_phy_selftest.c -- host-side smoke test for mesh_phy.c.
 *
 * NOT a port of mesh_fallback_regress.py: that suite wires PhyPolicy to the
 * real ModemNode/ModemDaemon packet layer and a scripted channel; no such
 * harness exists here (see tests/MESH/c/README.md). This instead drives a
 * handful of the same scenarios directly against the state machine with no
 * packet layer, so a change to mesh_phy.c that breaks an obvious case fails
 * loudly instead of only showing up once someone wires up a real DOS build.
 *
 * Build: cc -std=c89 -pedantic -Wall -Wextra mesh_phy.c mesh_phy_selftest.c \
 *           -o /tmp/mesh_phy_selftest && /tmp/mesh_phy_selftest
 */
#include <stdio.h>
#include "mesh_phy.h"

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        failures++; \
    } \
} while (0)

/* css_then_ofdm: no autonomous re-entry after a drop, only an external
 * signal re-arms upgrade_wanted (case 6 of mesh_fallback_regress.py). */
static void test_no_autonomous_reentry(void)
{
    meshphy p;
    meshphy_action acts[MESHPHY_MAX_ACTIONS];
    int n;

    meshphy_init(&p, 1, 2, MESHPHY_CSS_THEN_OFDM, 1,
                 10.0, 3, 120.0, 2, 35.0, -1.0, 20.0, 4,
                 70.0, 18.0, 3.0, 18.0, 2, 4, 1, 0);
    CHECK(p.upgrade_wanted == 1, "initiator starts wanting the upgrade");

    /* REQ retries exhaust with no ACK -> give up wanting OFDM */
    meshphy_on_req_queued(&p, 0);
    p.req_last_tx = 10.0;
    n = meshphy_tick(&p, 10.0 + 120.0 + 1.0, 0, 0, 1, acts,
                     MESHPHY_MAX_ACTIONS);
    CHECK(n >= 1, "req timeout should retransmit or drop");

    /* drive it to the final give-up: keep failing to hear an ACK */
    {
        int i;
        double now = 10.0;
        for (i = 0; i < 6 && p.upgrade_wanted; i++) {
            n = meshphy_tick(&p, now, 0, 0, 1, acts, MESHPHY_MAX_ACTIONS);
            if (n >= 1 && acts[0].kind == MESHPHY_ACT_SEND_REQ) {
                meshphy_on_req_queued(&p, i);
                meshphy_on_tx(&p, 0, 1, now);
            }
            now += 121.0;
        }
    }
    CHECK(p.upgrade_wanted == 0, "req_unanswered should clear upgrade_wanted");

    /* time passing alone must NOT re-arm it */
    n = meshphy_tick(&p, 100000.0, 0, 0, 1, acts, MESHPHY_MAX_ACTIONS);
    CHECK(p.upgrade_wanted == 0,
          "no timed autonomous re-upgrade after giving up");
    (void)n;

    /* only an external signal (the detector) re-arms it */
    meshphy_on_detect(&p, 100000.0, 0.9, acts, MESHPHY_MAX_ACTIONS);
    CHECK(p.upgrade_wanted == 1, "detector hit re-arms upgrade_wanted");
}

/* ofdm_only: data unacked for a full retry set renegotiates, not falls back
 * (there is nothing to fall back to) -- and a bounded number of doomed
 * re-handshakes eventually gives up rather than looping forever. */
static void test_ofdm_only_bounded_giveup(void)
{
    meshphy p;
    meshphy_action acts[MESHPHY_MAX_ACTIONS];
    int n, i;
    double now = 0.0;

    meshphy_init(&p, 1, 2, MESHPHY_OFDM_ONLY, 1,
                 0.0, 3, 120.0, 2, 35.0, -1.0, 5.0, 3,
                 70.0, 18.0, 3.0, 18.0, 2, 4, 1, 0);

    for (i = 0; i < 20 && !p.gave_up; i++) {
        n = meshphy_tick(&p, now, 0, 0, 1, acts, MESHPHY_MAX_ACTIONS);
        if (n >= 1 && acts[0].kind == MESHPHY_ACT_ENTER_HS) {
            /* handshake attempted and failed every time */
            meshphy_on_hs_result(&p, 0, 0, now, acts, MESHPHY_MAX_ACTIONS);
        }
        now += 6.0;
    }
    CHECK(p.gave_up == 1, "ofdm_only must give up after max_hs_attempts");
    CHECK(p.hs_attempts == 3, "gives up exactly at max_hs_attempts, not later");
}

/* link-silence timeout must fire for a pure receiver even though it has no
 * DATA of its own pending (the desync fix, OTA 20260728). */
static void test_link_silent_receiver(void)
{
    meshphy p;
    meshphy_action acts[MESHPHY_MAX_ACTIONS];
    int n;

    meshphy_init(&p, 2, 1, MESHPHY_OFDM_ONLY, 0,
                 0.0, 3, 120.0, 2, 35.0, -1.0, 5.0, 4,
                 70.0, 18.0, 3.0, 18.0, 2, 4, 1, 0);
    meshphy_on_hs_result(&p, 1, 1, 0.0, acts, MESHPHY_MAX_ACTIONS);
    CHECK(p.link_up == 1, "handshake success brings the link up");

    /* link_silent_s defaults to (1+data_retries)*ack_wait_s = 3*35 = 105 */
    n = meshphy_tick(&p, 104.0, 0, 0, 1, acts, MESHPHY_MAX_ACTIONS);
    CHECK(n == 0, "must not renegotiate before link_silent_s elapses");
    n = meshphy_tick(&p, 106.0, 0, 0, 1, acts, MESHPHY_MAX_ACTIONS);
    CHECK(n == 1 && acts[0].kind == MESHPHY_ACT_LINK_DOWN
          && acts[0].reason == MESHPHY_REASON_LINK_SILENT,
          "silent link with no pending DATA must renegotiate");
    CHECK(p.link_up == 0, "renegotiate actually drops the link first");
}

/* a CSS packet addressed to someone else must not tear down our link
 * (OTA 20260720: to_me must gate this). */
static void test_css_rx_to_me_gate(void)
{
    meshphy p;
    meshphy_action acts[MESHPHY_MAX_ACTIONS];
    int n;

    meshphy_init(&p, 2, 1, MESHPHY_CSS_THEN_OFDM, 0,
                 10.0, 3, 120.0, 2, 35.0, -1.0, 20.0, 4,
                 70.0, 18.0, 3.0, 18.0, 2, 4, 1, 0);
    meshphy_on_hs_result(&p, 1, 1, 0.0, acts, MESHPHY_MAX_ACTIONS);
    CHECK(p.link_up == 1, "handshake success brings the link up");

    n = meshphy_on_css_rx_from_peer(&p, 5.0, 0, acts, MESHPHY_MAX_ACTIONS);
    CHECK(n == 0 && p.link_up == 1,
          "CSS traffic not addressed to us must not drop the link");

    n = meshphy_on_css_rx_from_peer(&p, 5.0, 1, acts, MESHPHY_MAX_ACTIONS);
    CHECK(n == 1 && p.link_up == 0,
          "CSS traffic addressed to us proves the peer left OFDM");
}

int main(void)
{
    test_no_autonomous_reentry();
    test_ofdm_only_bounded_giveup();
    test_link_silent_receiver();
    test_css_rx_to_me_gate();
    if (failures) {
        printf("%d check(s) FAILED\n", failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
