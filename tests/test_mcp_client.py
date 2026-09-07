"""Minimal MCP client for BeamNG's in-engine server (0.39+).

BeamNG exposes Streamable HTTP MCP on 127.0.0.1:29292. The point of talking to
it from our own code rather than from an AI client is that **the LLM is never in
the control loop** -- this is a plain RPC transport that the classical
controller drives.

Tested against a real local HTTP server, not a mock: the framing (session
headers, SSE vs JSON responses) is exactly what tends to go wrong.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sim.mcp_client import MCPClient, MCPError

TOOLS = [
    {"name": "vehicle_state", "description": "read vehicle state"},
    {"name": "vehicle_control", "description": "set throttle/brake/steering"},
]


class Handler(BaseHTTPRequestHandler):
    mode = "json"
    session_id = "sess-123"
    seen: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen.append((body, dict(self.headers)))
        method = body.get("method")

        if method == "notifications/initialized":
            self.send_response(202)
            self.end_headers()
            return

        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "beamng", "version": "0.39"},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name = body["params"]["name"]
            if name == "explode":
                result = {"content": [{"type": "text", "text": "boom"}], "isError": True}
            else:
                result = {
                    "content": [
                        {"type": "text", "text": json.dumps({"speed": 12.5, "rpm": 3000})}
                    ]
                }
        else:
            payload = {"jsonrpc": "2.0", "id": body.get("id"),
                       "error": {"code": -32601, "message": "no such method"}}
            self._respond(payload)
            return

        self._respond({"jsonrpc": "2.0", "id": body.get("id"), "result": result})

    def _respond(self, payload):
        raw = json.dumps(payload).encode()
        if Handler.mode == "sse":
            raw = b"event: message\ndata: " + raw + b"\n\n"
            content_type = "text/event-stream"
        else:
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Mcp-Session-Id", Handler.session_id)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def server():
    Handler.seen = []
    Handler.mode = "json"
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/mcp"
    httpd.shutdown()
    httpd.server_close()


def test_connecting_reports_the_server_it_found(server):
    client = MCPClient(server)
    info = client.connect()
    assert info["serverInfo"]["name"] == "beamng"
    client.close()


def test_the_tool_list_comes_back(server):
    with MCPClient(server) as client:
        assert [t["name"] for t in client.list_tools()] == [
            "vehicle_state",
            "vehicle_control",
        ]


def test_a_tool_call_returns_its_parsed_json_payload(server):
    with MCPClient(server) as client:
        assert client.call("vehicle_state") == {"speed": 12.5, "rpm": 3000}


def test_tool_arguments_are_sent(server):
    with MCPClient(server) as client:
        client.call("vehicle_control", {"throttle": 0.5})
    call = [b for b, _ in Handler.seen if b.get("method") == "tools/call"][0]
    assert call["params"]["arguments"] == {"throttle": 0.5}


def test_the_session_id_is_returned_on_later_requests(server):
    # Streamable HTTP servers hand out a session on initialize and expect it
    # back; dropping it makes every subsequent call fail in confusing ways.
    with MCPClient(server) as client:
        client.list_tools()
    later = [h for b, h in Handler.seen if b.get("method") == "tools/list"][0]
    assert later.get("Mcp-Session-Id") == "sess-123"


def test_a_server_error_is_raised_with_its_message(server):
    with MCPClient(server) as client:
        with pytest.raises(MCPError, match="no such method"):
            client.request("nonsense/method")


def test_a_tool_reporting_an_error_raises(server):
    with MCPClient(server) as client:
        with pytest.raises(MCPError, match="boom"):
            client.call("explode")


def test_an_sse_framed_response_is_understood(server):
    # Streamable HTTP may answer with either JSON or an SSE stream.
    Handler.mode = "sse"
    with MCPClient(server) as client:
        assert client.call("vehicle_state") == {"speed": 12.5, "rpm": 3000}


def test_a_dead_endpoint_raises_a_clear_error():
    client = MCPClient("http://127.0.0.1:9/mcp", timeout=1.0)
    with pytest.raises(MCPError, match="could not reach"):
        client.connect()


def test_the_client_says_which_endpoint_it_is_using(server):
    assert server in repr(MCPClient(server))
