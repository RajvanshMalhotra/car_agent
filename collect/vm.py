"""One correct way to run Lua in the vehicle VM.

`run_lua_vehicle` is asynchronous and its replies carry **no indication of
which request they answer**. Issue two calls close together and the second
reads the first one's answer. That has now caused three separate wrong
conclusions in this project: the first probe run came back shifted by two
places, `did_it_move` reported 0.00 m/s for a car doing 9 m/s, and every
experiment in the stuck-car diagnosis printed "could not read position" while
reading somebody else's `{ok = true}`.

The fix is the same each time, so it lives in one place now: every request
carries a tag, and the caller drains the queue until its own tag comes back.
Nothing else in this project may call `run_lua_vehicle` directly.
"""

from __future__ import annotations

import itertools
import json
import time

from collect.decode import is_async_notice

#: Distinct per process. Two sessions against one game would otherwise collide.
_TAGS = itertools.count(1)

#: A tagged, pcall-wrapped chunk. The tag travels *with the answer*, which is
#: the whole point -- a reply that cannot say what it answers is not an answer.
TEMPLATE = """
local ok, value = pcall(function() %(body)s end)
if not ok then
  return jsonEncode({tag = '%(tag)s', ok = false, error = tostring(value)})
end
if value == nil then value = {} end
return jsonEncode({tag = '%(tag)s', ok = true, value = value})
"""

#: A request that does nothing, used to keep the queue turning while waiting.
DRAIN_BODY = "return 0"


class VMError(RuntimeError):
    """The vehicle VM refused, or never answered."""


class VehicleVM:
    """Tagged request/response against the vehicle's Lua VM."""

    def __init__(self, client, attempts: int = 24, wait_s: float = 0.12) -> None:
        self.client = client
        self.attempts = attempts
        self.wait_s = wait_s

    # -- transport --------------------------------------------------------

    def _payloads(self, raw) -> list[dict]:
        """Every tagged answer carried by one reply."""
        if raw is None or is_async_notice(raw):
            return []
        candidates = list(raw.values()) if isinstance(raw, dict) else [raw]
        found = []
        for value in candidates:
            if not isinstance(value, str):
                continue
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict) and "tag" in parsed:
                found.append(parsed)
        return found

    def _send(self, body: str, tag: str):
        return self.client.call(
            "run_lua_vehicle",
            {"code": TEMPLATE % {"body": body, "tag": tag}},
        )

    # -- the only entry point ---------------------------------------------

    def call(self, body: str):
        """Run `body` and return its value, waiting for *this* request's reply.

        `body` is a Lua chunk that returns something JSON-encodable.
        """
        tag = f"ca{next(_TAGS)}"
        seen: dict[str, dict] = {}

        def absorb(raw) -> None:
            for payload in self._payloads(raw):
                seen[payload["tag"]] = payload

        absorb(self._send(body, tag))
        attempt = 0
        while tag not in seen and attempt < self.attempts:
            attempt += 1
            if self.wait_s:
                time.sleep(self.wait_s)
            # Draining needs traffic: the VM answers on the next call, so a
            # request that does nothing is what makes the previous one arrive.
            absorb(self._send(DRAIN_BODY, f"drain{next(_TAGS)}"))

        if tag not in seen:
            raise VMError(f"the vehicle VM did not answer within {attempt} calls")
        answer = seen[tag]
        if not answer.get("ok"):
            raise VMError(str(answer.get("error", "the chunk failed")))
        return answer.get("value")

    def try_call(self, body: str, default=None):
        """As `call`, but hand back `default` instead of raising."""
        try:
            return self.call(body)
        except (VMError, Exception) as error:  # noqa: BLE001
            if isinstance(error, VMError):
                return default
            raise
