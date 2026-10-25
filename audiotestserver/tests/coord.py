# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""
tests/coord.py - coordination helper for test scripts.
Usage:
    from coord import Coord
    c = Coord(server_url, run_id, client_id)

    c.signal("ready")                        # POST - announce this client is ready
    c.wait_for("ready", ["client1"])         # block until client1 has signalled ready
    c.wait_for_any("ready", timeout=15)      # block until at least 1 client signals
    c.who_signalled("ready")                 # returns list of client IDs
"""

import time
import urllib.request
import urllib.error
import json as _json


class Coord:
    def __init__(self, server_url: str, run_id: str, client_id: str):
        self.base       = server_url.rstrip("/")
        self.run_id     = run_id
        self.client_id  = client_id

    def _url(self, signal: str) -> str:
        return f"{self.base}/api/coord/{self.run_id}/{signal}"

    def signal(self, signal_name: str) -> list:
        """POST this client's signal. Returns list of all clients who have signalled."""
        url  = self._url(signal_name)
        body = _json.dumps({"client_id": self.client_id}).encode()
        req  = urllib.request.Request(url, data=body,
                                      headers={"Content-Type": "application/json"},
                                      method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return _json.loads(r.read()).get("clients", [])
        except Exception as e:
            print(f"[coord] signal '{signal_name}' failed: {e}", flush=True)
            return []

    def who_signalled(self, signal_name: str) -> list:
        """GET - return list of client IDs who have posted this signal."""
        try:
            with urllib.request.urlopen(self._url(signal_name), timeout=5) as r:
                return _json.loads(r.read()).get("clients", [])
        except Exception as e:
            print(f"[coord] check '{signal_name}' failed: {e}", flush=True)
            return []

    def wait_for(self, signal_name: str, client_ids: list,
                 timeout: float = 30.0, poll: float = 0.5) -> bool:
        """Block until ALL of client_ids have signalled, or timeout. Returns True on success."""
        needed = set(client_ids)
        deadline = time.time() + timeout
        while time.time() < deadline:
            have = set(self.who_signalled(signal_name))
            if needed.issubset(have):
                return True
            time.sleep(poll)
        return False

    def wait_for_any(self, signal_name: str,
                     timeout: float = 30.0, poll: float = 0.5) -> list:
        """Block until at least one client has signalled. Returns list of signalers (may be empty on timeout)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            have = self.who_signalled(signal_name)
            if have:
                return have
            time.sleep(poll)
        return []

    def wait_for_count(self, signal_name: str, count: int,
                       timeout: float = 30.0, poll: float = 0.5) -> list:
        """Block until at least `count` clients have signalled."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            have = self.who_signalled(signal_name)
            if len(have) >= count:
                return have
            time.sleep(poll)
        return self.who_signalled(signal_name)