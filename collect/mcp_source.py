"""The trajectory source that talks to the game.

`run_lua_vehicle` is asynchronous. It answers "queued in vehicle VM(s)" and
delivers the result on a later call, so a drain is not one round trip but a
short retry loop with a budget. The budget matters: without one, a vehicle VM
that has stopped answering hangs the run instead of ending it.
"""

from __future__ import annotations

import time

from collect.decode import DecodeError, decode, decode_install
from collect.derive import add_derived
from collect.lua import (
    drain_source, install_source, uninstall_source, verdict_source,
)
from collect.source import TrajectorySource


class MCPTrajectorySource(TrajectorySource):
    def __init__(
        self,
        client,
        interval_s: float = 0.01,
        attempts: int = 8,
        wait_s: float = 0.15,
    ) -> None:
        self.client = client
        self.interval_s = interval_s
        self.attempts = attempts
        self.wait_s = wait_s
        self.dropped = 0
        self.mass_kg = 0.0
        self.capture_error: str | None = None
        self.installed = False

    def _call(self, code: str):
        return self.client.call("run_lua_vehicle", {"code": code})

    def start(self) -> None:
        """Install the sampler, and confirm it can actually read the vehicle."""
        self._call(install_source(self.interval_s))
        self.installed = True
        # The install returns its own self-test, but asynchronously. Poll for it
        # so a bad field name surfaces here rather than as an empty CSV later.
        for attempt in range(self.attempts):
            if attempt and self.wait_s:
                time.sleep(self.wait_s)
            try:
                result = decode_install(self._call(verdict_source()))
            except DecodeError as error:
                raise RuntimeError(str(error)) from error
            if result:
                self.mass_kg = float(result.get("mass") or 0.0)
                return

    def drain(self) -> list[dict]:
        code = drain_source()
        for attempt in range(self.attempts):
            if attempt and self.wait_s:
                time.sleep(self.wait_s)
            try:
                result = decode(self._call(code))
            except DecodeError as error:
                raise RuntimeError(str(error)) from error
            if result.pending:
                continue
            self.dropped += result.dropped
            self.mass_kg = result.mass_kg or self.mass_kg
            if result.error and not self.capture_error:
                self.capture_error = result.error
            return [add_derived(sample) for sample in result.samples]
        # Budget spent. An empty drain is honest: nothing arrived.
        return []

    def stop(self) -> None:
        if not self.installed:
            return
        self.installed = False
        self._call(uninstall_source())
