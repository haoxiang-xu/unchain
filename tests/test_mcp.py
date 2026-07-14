"""Tests for unchain MCP client toolkit integration."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from unchain.toolkits import MCPToolkit
from unchain.tools import Toolkit
from unchain.agent import Agent, ToolsModule
from unchain.agent.builder import AgentCallContext


# ── helpers: fake MCP objects ──────────────────────────────────────────────

def _make_fake_tool(name: str, description: str, input_schema: dict):
    """Create a fake MCP Tool-like object."""
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=input_schema,
    )


def _make_fake_call_result(text: str, is_error: bool = False):
    """Create a fake MCP CallToolResult-like object."""
    content = [SimpleNamespace(text=text)]
    return SimpleNamespace(
        isError=is_error,
        content=content,
        structuredContent=None,
    )


def _make_fake_call_result_structured(data: dict, is_error: bool = False):
    """Create a fake MCP CallToolResult with structured content."""
    return SimpleNamespace(
        isError=is_error,
        content=[],
        structuredContent=data,
    )


# ── unit tests ─────────────────────────────────────────────────────────────

class TestMcpInit:
    def test_stdio_transport_inferred(self):
        m = MCPToolkit(command="echo", args=["hello"])
        assert m._transport == "stdio"
        assert not m.connected

    def test_stdio_cwd_is_stored(self):
        m = MCPToolkit(command="echo", args=["hello"], cwd="/tmp")
        assert m._cwd == "/tmp"

    def test_sse_transport_inferred(self):
        m = MCPToolkit(url="http://localhost:8080/sse")
        assert m._transport == "sse"

    def test_explicit_transport_override(self):
        m = MCPToolkit(url="http://localhost:8080/mcp", transport="streamable_http")
        assert m._transport == "streamable_http"

    def test_missing_args_raises(self):
        with pytest.raises(ValueError, match="requires either command"):
            MCPToolkit()

    def test_inherits_Toolkit(self):
        m = MCPToolkit(command="echo")
        assert isinstance(m, Toolkit)


class TestMcpToolConversion:
    def test_convert_mcp_tool_basic(self):
        m = MCPToolkit(command="echo")
        fake_tool = _make_fake_tool(
            "read_file",
            "Read a file from disk",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        )

        converted = m._convert_mcp_tool(fake_tool)
        assert converted.name == "read_file"
        assert converted.description == "Read a file from disk"
        assert len(converted.parameters) == 1
        assert converted.parameters[0].name == "path"
        assert converted.parameters[0].type_ == "string"
        assert converted.parameters[0].required is True

    def test_convert_mcp_tool_no_required(self):
        m = MCPToolkit(command="echo")
        fake_tool = _make_fake_tool(
            "search",
            "Search for text",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
            },
        )

        converted = m._convert_mcp_tool(fake_tool)
        assert len(converted.parameters) == 2
        assert all(not p.required for p in converted.parameters)

    def test_convert_mcp_tool_empty_schema(self):
        m = MCPToolkit(command="echo")
        fake_tool = _make_fake_tool("ping", "Ping the server", {})
        converted = m._convert_mcp_tool(fake_tool)
        assert converted.name == "ping"
        assert len(converted.parameters) == 0

    def test_converted_tool_delegates_execution_to_mcp_toolkit(self):
        m = MCPToolkit(command="echo")
        calls = []

        def fake_execute(function_name, arguments):
            calls.append((function_name, arguments))
            return {"ok": True, "tool": function_name}

        m.execute = fake_execute
        fake_tool = _make_fake_tool(
            "repo_search",
            "Search repository content",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        )

        converted = m._convert_mcp_tool(fake_tool)

        assert converted.execute({"query": "pupu"}) == {"ok": True, "tool": "repo_search"}
        assert calls == [("repo_search", {"query": "pupu"})]

    def test_to_json_produces_valid_tool_definitions(self):
        m = MCPToolkit(command="echo")
        fake_tool = _make_fake_tool(
            "greet",
            "Say hello",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Person name"},
                },
                "required": ["name"],
            },
        )
        converted = m._convert_mcp_tool(fake_tool)
        m.tools[converted.name] = converted

        json_list = m.to_json()
        assert len(json_list) == 1
        defn = json_list[0]
        assert defn["name"] == "greet"
        assert defn["type"] == "function"
        assert "name" in defn["parameters"]["properties"]
        assert "name" in defn["parameters"]["required"]


class TestMcpResultParsing:
    def test_parse_text_result(self):
        result = _make_fake_call_result("hello world")
        parsed = MCPToolkit(command="echo")._parse_call_result(result, "test_tool")
        assert parsed == {"result": "hello world"}

    def test_parse_json_text_result(self):
        result = _make_fake_call_result('{"count": 42}')
        parsed = MCPToolkit(command="echo")._parse_call_result(result, "test_tool")
        assert parsed == {"count": 42}

    def test_parse_structured_result(self):
        result = _make_fake_call_result_structured({"files": ["a.txt", "b.txt"]})
        parsed = MCPToolkit(command="echo")._parse_call_result(result, "test_tool")
        assert parsed == {"files": ["a.txt", "b.txt"]}

    def test_parse_error_result(self):
        result = _make_fake_call_result("file not found", is_error=True)
        parsed = MCPToolkit(command="echo")._parse_call_result(result, "test_tool")
        assert "error" in parsed
        assert "file not found" in parsed["error"]

    def test_parse_empty_error(self):
        result = SimpleNamespace(isError=True, content=[], structuredContent=None)
        parsed = MCPToolkit(command="echo")._parse_call_result(result, "test_tool")
        assert parsed["error"] == "unknown MCP error"

    # ── S0: image blocks are collected into content_blocks, not dropped ──────

    def test_parse_image_block_collected_into_content_blocks(self):
        # MCP text + image content; text keeps working, image is no longer dropped.
        result = SimpleNamespace(
            isError=False,
            content=[
                SimpleNamespace(type="text", text="shot"),
                SimpleNamespace(type="image", data="aW1n", mimeType="image/png"),
            ],
            structuredContent=None,
        )
        parsed = MCPToolkit(command="echo")._parse_call_result(result, "test_tool")
        assert parsed["result"] == "shot"
        assert parsed["content_blocks"] == [
            {"type": "image", "media_type": "image/png", "data_b64": "aW1n"}
        ]

    def test_parse_image_alongside_structured_content(self):
        result = SimpleNamespace(
            isError=False,
            content=[SimpleNamespace(type="image", data="Zm9v", mimeType="image/jpeg")],
            structuredContent={"files": ["a.txt"]},
        )
        parsed = MCPToolkit(command="echo")._parse_call_result(result, "test_tool")
        assert parsed["files"] == ["a.txt"]
        assert parsed["content_blocks"] == [
            {"type": "image", "media_type": "image/jpeg", "data_b64": "Zm9v"}
        ]

    def test_parse_image_defaults_media_type(self):
        result = SimpleNamespace(
            isError=False,
            content=[SimpleNamespace(type="image", data="aW1n", mimeType=None)],
            structuredContent=None,
        )
        parsed = MCPToolkit(command="echo")._parse_call_result(result, "test_tool")
        assert parsed["content_blocks"][0]["media_type"] == "image/png"

    def test_parse_text_only_result_has_no_content_blocks_key(self):
        # zero-migration: results without images are byte-identical to before.
        result = _make_fake_call_result("hello world")
        parsed = MCPToolkit(command="echo")._parse_call_result(result, "test_tool")
        assert parsed == {"result": "hello world"}
        assert "content_blocks" not in parsed


class TestMcpConnectDisconnect:
    """Test connect/disconnect with a fully mocked MCP session."""

    @staticmethod
    def _start_loop_thread():
        """Start an event loop in a daemon thread, return (loop, thread, stop_event)."""
        loop = asyncio.new_event_loop()
        stop_event = asyncio.Event()
        ready = threading.Event()

        async def _keep_alive():
            ready.set()
            await stop_event.wait()

        def _run():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_keep_alive())

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        ready.wait(timeout=5)
        return loop, t, stop_event

    @staticmethod
    def _stop_loop_thread(loop, t, stop_event):
        loop.call_soon_threadsafe(stop_event.set)
        t.join(timeout=5)
        loop.close()

    def test_populate_tools_from_mock(self):
        """Verify _populate_tools correctly converts MCP tools."""
        m = MCPToolkit(command="echo")

        fake_tools = [
            _make_fake_tool("tool_a", "Tool A", {
                "type": "object",
                "properties": {"x": {"type": "integer", "description": "Value"}},
                "required": ["x"],
            }),
            _make_fake_tool("tool_b", "Tool B", {}),
        ]
        list_result = SimpleNamespace(tools=fake_tools)

        async def fake_list_tools():
            return list_result

        mock_session = MagicMock()
        mock_session.list_tools = fake_list_tools

        loop, t, stop = self._start_loop_thread()
        m._loop = loop
        m._session = mock_session

        try:
            m._populate_tools()
        finally:
            self._stop_loop_thread(loop, t, stop)

        assert "tool_a" in m.tools
        assert "tool_b" in m.tools
        assert len(m.tools) == 2
        assert m.tools["tool_a"].parameters[0].name == "x"
        assert m.tools["tool_a"].parameters[0].required is True

    def test_execute_calls_mcp_server(self):
        """Verify execute() forwards to the MCP session's call_tool."""
        m = MCPToolkit(command="echo")

        call_result = _make_fake_call_result('{"status": "ok"}')
        captured_calls = []

        async def fake_call_tool(name, arguments):
            captured_calls.append((name, arguments))
            return call_result

        mock_session = MagicMock()
        mock_session.call_tool = fake_call_tool

        from unchain.tools import Tool
        m.tools["my_tool"] = Tool(name="my_tool", func=lambda: {}, parameters=[])

        loop, t, stop = self._start_loop_thread()
        m._loop = loop
        m._session = mock_session
        m._connected = True

        try:
            result = m.execute("my_tool", {"key": "value"})
        finally:
            m._connected = False
            self._stop_loop_thread(loop, t, stop)

        assert result == {"status": "ok"}
        assert len(captured_calls) == 1
        assert captured_calls[0] == ("my_tool", {"key": "value"})

    def test_execute_parses_string_arguments(self):
        """Verify execute() handles string arguments correctly."""
        m = MCPToolkit(command="echo")

        call_result = _make_fake_call_result("done")
        captured_calls = []

        async def fake_call_tool(name, arguments):
            captured_calls.append((name, arguments))
            return call_result

        mock_session = MagicMock()
        mock_session.call_tool = fake_call_tool

        from unchain.tools import Tool
        m.tools["ping"] = Tool(name="ping", func=lambda: {}, parameters=[])

        loop, t, stop = self._start_loop_thread()
        m._loop = loop
        m._session = mock_session
        m._connected = True

        try:
            m.execute("ping", '{"host": "localhost"}')
        finally:
            m._connected = False
            self._stop_loop_thread(loop, t, stop)

        assert captured_calls[0] == ("ping", {"host": "localhost"})

    def test_execute_disconnected_falls_back_to_local(self):
        """When disconnected, execute() falls back to the parent toolkit."""
        m = MCPToolkit(command="echo")
        result = m.execute("nonexistent", {})
        assert "error" in result

    def test_repr_disconnected(self):
        m = MCPToolkit(command="npx", args=["-y", "my-server"])
        assert "disconnected" in repr(m)
        assert "npx" in repr(m)

    def test_repr_url(self):
        m = MCPToolkit(url="http://localhost:3000/sse")
        assert "http://localhost:3000/sse" in repr(m)


class TestMcpWithAgent:
    """Test that MCP integrates with the current agent toolkit assembly."""

    def test_agent_can_add_mcp_Toolkit(self):
        m = MCPToolkit(command="echo")

        # Register a fake tool manually (simulating what connect() does)
        from unchain.tools import Tool
        m.tools["mcp_tool"] = Tool(
            name="mcp_tool",
            func=lambda x="": {"echoed": x},
            parameters=[],
        )

        agent = Agent(
            name="mcp_test",
            modules=(ToolsModule(tools=(m,)),),
            model_io_factory=lambda spec, context: None,
        )
        prepared = agent._prepare(AgentCallContext(mode="run", input_messages=[]))

        # Verify tools are visible through the merged view
        merged_json = prepared.toolkit.to_json()
        tool_names = [t["name"] for t in merged_json]
        assert "mcp_tool" in tool_names

    def test_agent_find_tool_in_mcp_Toolkit(self):
        m = MCPToolkit(command="echo")

        from unchain.tools import Tool
        m.tools["remote_tool"] = Tool(
            name="remote_tool",
            func=lambda: {"ok": True},
            parameters=[],
        )

        agent = Agent(
            name="mcp_test",
            modules=(ToolsModule(tools=(m,)),),
            model_io_factory=lambda spec, context: None,
        )
        prepared = agent._prepare(AgentCallContext(mode="run", input_messages=[]))
        found = prepared.toolkit.get("remote_tool")
        assert found is not None
        assert found.name == "remote_tool"


# ── live smoke test (requires an actual MCP server) ────────────────────────

def test_mcp_stdio_smoke():
    """Smoke test with a real stdio MCP server.

    Skipped unless the 'npx' command is available and the filesystem
    server package can be downloaded.  Set MCP_SMOKE=1 to enable.
    """
    import os
    import shutil

    if not os.environ.get("MCP_SMOKE"):
        pytest.skip("MCP_SMOKE not set — skipping live MCP smoke test")

    if not shutil.which("npx"):
        pytest.skip("npx not found on PATH")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a test file
        test_file = Path(tmpdir) / "hello.txt"
        test_file.write_text("hello from unchain")

        with MCPToolkit(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", tmpdir],
        ) as server:
            assert server.connected
            assert len(server.tools) > 0

            # The filesystem server should have tools like read_file, list_directory
            tool_names = list(server.tools.keys())
            assert any("read" in name.lower() or "list" in name.lower() for name in tool_names), \
                f"Expected filesystem tools, got: {tool_names}"

            # Try listing the directory
            if "list_directory" in server.tools:
                result = server.execute("list_directory", {"path": tmpdir})
                assert "error" not in result or not result.get("error")

        assert not server.connected
