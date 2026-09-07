"""Minimal MCP client for BeamNG's in-engine server.

BeamNG 0.39 ships an MCP server (Options -> Advanced -> "Enable MCP server"),
served as Streamable HTTP on `http://127.0.0.1:29292/mcp`. It is first-party and
needs neither BeamNG.tech nor a third-party mod.

This is deliberately a plain RPC client and not an AI integration. **The LLM is
never in the control loop** -- the classical controller calls these tools
directly, at whatever rate it runs. Using MCP here is a transport decision, not
an agent decision.

Only the three calls we need are implemented: initialize, tools/list,
tools/call. Streamable HTTP servers may answer with either a JSON body or an SSE
stream, so both framings are handled.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
DEFAULT_ENDPOINT = "http://127.0.0.1:29292/mcp"


class MCPError(RuntimeError):
    """The MCP server was unreachable, or answered with an error."""


class MCPClient:
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 10.0,
        client_name: str = "car_agent",
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.client_name = client_name
        self.session_id: str | None = None
        self.server_info: dict[str, Any] | None = None
        self._next_id = 0

    def __repr__(self) -> str:
        return f"MCPClient({self.endpoint!r})"

    def __enter__(self) -> "MCPClient":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- transport --------------------------------------------------------

    def _post(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        body = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            # Streamable HTTP: the server picks one of these two framings.
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        request = urllib.request.Request(
            self.endpoint, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                session = response.headers.get("Mcp-Session-Id")
                if session:
                    self.session_id = session
                raw = response.read()
                if not raw:
                    return None
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type:
                    return self._parse_sse(raw.decode())
                return json.loads(raw.decode())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:300]
            raise MCPError(
                f"{self.endpoint} returned HTTP {error.code}: {detail}"
            ) from error
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise MCPError(
                f"could not reach the MCP server at {self.endpoint} ({error}). "
                "Is BeamNG running with Options > Advanced > 'Enable MCP server' on?"
            ) from error

    @staticmethod
    def _parse_sse(text: str) -> dict[str, Any] | None:
        """Pull the first JSON payload out of an SSE stream."""
        for line in text.splitlines():
            if line.startswith("data:"):
                chunk = line[len("data:"):].strip()
                if chunk:
                    return json.loads(chunk)
        return None

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._next_id += 1
        response = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": method,
                "params": params or {},
            }
        )
        if response is None:
            raise MCPError(f"{method}: empty response")
        if "error" in response:
            error = response["error"]
            raise MCPError(
                f"{method}: {error.get('message', error)} "
                f"(code {error.get('code', '?')})"
            )
        return response.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._post({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # -- MCP --------------------------------------------------------------

    def connect(self) -> dict[str, Any]:
        result = self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": self.client_name, "version": "0.1"},
            },
        )
        self.server_info = result
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = self.request("tools/list", params)
            tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a tool and return its payload, parsed as JSON where possible."""
        result = self.request(
            "tools/call", {"name": name, "arguments": arguments or {}}
        )
        texts = [
            block.get("text", "")
            for block in result.get("content", [])
            if block.get("type") == "text"
        ]
        joined = "\n".join(texts)
        if result.get("isError"):
            raise MCPError(f"{name} failed: {joined[:300]}")
        if "structuredContent" in result:
            return result["structuredContent"]
        try:
            return json.loads(joined)
        except (json.JSONDecodeError, ValueError):
            return joined

    def close(self) -> None:
        self.session_id = None
