"""Security and transport contracts exercised against the installed dependencies."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import jwt
import pytest


@pytest.mark.parametrize("decoder", [jwt.decode, jwt.decode_complete])
@pytest.mark.parametrize("invalid_claims,error", [
    ({"exp": 0}, jwt.ExpiredSignatureError),
    ({"aud": "different-service"}, jwt.InvalidAudienceError),
])
def test_jwt_unverified_peek_cannot_disable_later_claim_validation(decoder, invalid_claims, error):
    """GHSA-gvp8-978c-rx2q: reusing options must not retain disabled checks."""
    key = "local-regression-test-key-with-32-bytes"
    token = jwt.encode(invalid_claims, key, algorithm="HS256")
    options = {"verify_signature": False}
    decoder(token, options=options)
    assert options == {"verify_signature": False}
    options["verify_signature"] = True
    with pytest.raises(error):
        decoder(token, key, algorithms=["HS256"], options=options, audience="this-service")
    valid = jwt.encode({"aud": "this-service", "exp": time.time() + 300}, key, algorithm="HS256")
    decoded = decoder(valid, key, algorithms=["HS256"], options=options, audience="this-service")
    claims = decoded["payload"] if decoder is jwt.decode_complete else decoded
    assert claims["aud"] == "this-service"


_SERVER = '''
import sys
from mcp.server.fastmcp import FastMCP
from starlette.responses import PlainTextResponse, RedirectResponse
from starlette.routing import Route
import uvicorn

mode, port = sys.argv[1], int(sys.argv[2])
server = FastMCP("dependency-contract", host="127.0.0.1", port=port)

@server.tool()
def echo(value: str) -> str:
    return value

if mode == "stdio":
    server.run(transport="stdio")
else:
    app = server.sse_app() if mode == "sse" else server.streamable_http_app()
    endpoint = "/sse" if mode == "sse" else "/mcp"
    async def redirect(request):
        return RedirectResponse(endpoint, status_code=307)
    app.routes.insert(0, Route("/redirect", redirect, methods=["GET", "POST", "DELETE"]))
    class CheckHeader:
        def __init__(self, app):
            self.app = app
        async def __call__(self, scope, receive, send):
            if scope["type"] == "http" and dict(scope["headers"]).get(b"authorization") != b"Bearer local-contract":
                await PlainTextResponse("missing authorization", status_code=401)(scope, receive, send)
                return
            await self.app(scope, receive, send)
    uvicorn.run(CheckHeader(app), host="127.0.0.1", port=port, log_level="error")
'''

_CLIENT = '''
import json, sys
from unchain.toolkits.mcp import MCPToolkit
options = json.loads(sys.argv[1])
with MCPToolkit(**options) as toolkit:
    assert toolkit.connected
    assert "echo" in toolkit.tools
    result = toolkit.execute("echo", {"value": "dependency-contract-ok"})
    assert "dependency-contract-ok" in json.dumps(result), result
    assert not result.get("error"), result
assert not toolkit.connected
'''


@pytest.mark.parametrize("transport,redirect", [
    ("stdio", False), ("sse", False), ("sse", True),
    ("streamable_http", False), ("streamable_http", True),
])
def test_mcp_real_transport_preserves_tool_calls_and_headers(tmp_path, transport, redirect):
    """Exercise the actual SDK/server pair without credentials or external services."""
    server_file = tmp_path / "mcp_contract_server.py"
    server_file.write_text(_SERVER)
    process = None
    with (tmp_path / "server.log").open("w+") as log:
        try:
            if transport == "stdio":
                options = {"command": sys.executable, "args": [str(server_file), "stdio", "0"]}
            else:
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                process = subprocess.Popen(
                    [sys.executable, str(server_file), transport, str(port)], stdout=log, stderr=log,
                )
                deadline = time.monotonic() + 15
                while True:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        log.seek(0)
                        pytest.fail(f"MCP fixture did not start: {log.read()}")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                            break
                    except OSError:
                        time.sleep(0.05)
                endpoint = "redirect" if redirect else ("sse" if transport == "sse" else "mcp")
                options = {
                    "transport": transport, "url": f"http://127.0.0.1:{port}/{endpoint}",
                    "headers": {"Authorization": "Bearer local-contract"},
                }
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            completed = subprocess.run(
                [sys.executable, "-c", _CLIENT, json.dumps(options)],
                env=environment, text=True, capture_output=True, timeout=40,
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
