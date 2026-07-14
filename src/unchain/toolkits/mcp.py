"""MCP (Model Context Protocol) client wrapper for unchain.

Exposes any MCP server as a standard ``Toolkit`` so it can be registered
with the agent via ``agent.add_toolkit()``.

Supports three transport modes:

* **stdio** – launches a local subprocess  
  ``MCPToolkit(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])``

* **sse** – connects to an SSE endpoint  
  ``MCPToolkit(url="http://localhost:8080/sse")``

* **streamable_http** – connects to a Streamable HTTP endpoint  
  ``MCPToolkit(url="http://localhost:8080/mcp", transport="streamable_http")``

Usage::

    from unchain.agent import Agent, ToolsModule
    from unchain.toolkits import MCPToolkit

    server = MCPToolkit(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
    server.connect()

    agent = Agent(name="developer", modules=(ToolsModule(tools=(server,)),))
    result = agent.run(...)

    server.disconnect()

Or as a context manager::

    with MCPToolkit(command="npx", args=[...]) as server:
        agent = Agent(name="developer", modules=(ToolsModule(tools=(server,)),))
        result = agent.run(...)
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from ..tools.decorators import tool
from ..tools.models import ToolParameter
from ..tools.tool import Tool
from ..tools.toolkit import Toolkit

# ── MCP SDK imports (lazy – only required when actually used) ─────────────
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamablehttp_client
    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MCP_AVAILABLE = False


class MCPToolkit(Toolkit):
    """Connect to an MCP server and expose its tools as an unchain ``Toolkit``.

    Parameters
    ----------
    command : str, optional
        Executable to spawn for stdio transport (e.g. ``"npx"``).
    args : list[str], optional
        Arguments for the subprocess (e.g. ``["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]``).
    env : dict[str, str], optional
        Additional environment variables for the subprocess.
    url : str, optional
        URL for SSE or Streamable HTTP transport.
    headers : dict[str, str], optional
        Extra HTTP headers for SSE / Streamable HTTP.
    transport : str, optional
        Force transport type: ``"stdio"``, ``"sse"``, or ``"streamable_http"``.
        When omitted the transport is inferred from the provided arguments
        (``command`` → stdio, ``url`` → sse).
    """

    def __init__(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        transport: str | None = None,
    ):
        super().__init__()

        if not _MCP_AVAILABLE:
            raise ImportError("mcp package is required — pip install mcp")

        # ── configuration ──────────────────────────────────────────────────
        self._command = command
        self._args = args or []
        self._env = env
        self._cwd = cwd
        self._url = url
        self._headers = headers

        if transport is not None:
            self._transport = transport
        elif command is not None:
            self._transport = "stdio"
        elif url is not None:
            self._transport = "sse"
        else:
            raise ValueError("mcp requires either command (stdio) or url (sse/streamable_http)")

        # ── runtime state ──────────────────────────────────────────────────
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._ready = threading.Event()
        self._stop_event: asyncio.Event | None = None
        self._error: BaseException | None = None
        self._connected = False

    # ── public lifecycle ───────────────────────────────────────────────────

    def connect(self) -> "MCPToolkit":
        """Connect to the MCP server, discover tools, and populate the toolkit.

        This method blocks until the session is ready and tools have been
        fetched.  It is safe to call ``connect()`` on an already-connected
        instance (it will be a no-op).

        Returns ``self`` for convenient chaining.
        """
        if self._connected:
            return self

        self._ready.clear()
        self._error = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Wait for the session to be initialised (or an error).
        self._ready.wait()
        if self._error is not None:
            self._cleanup_thread()
            raise RuntimeError(f"mcp: failed to connect — {self._error}") from self._error

        # Fetch tools into the toolkit.
        self._populate_tools()
        self._connected = True
        return self

    def disconnect(self) -> None:
        """Disconnect from the MCP server and clean up resources."""
        if not self._connected:
            return

        # Signal the session lifecycle coroutine to exit.
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)

        self._cleanup_thread()
        self._session = None
        self._connected = False
        self.tools.clear()

    @property
    def connected(self) -> bool:
        return self._connected

    # ── context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "MCPToolkit":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    # ── toolkit overrides ──────────────────────────────────────────────────

    def execute(self, function_name: str, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        """Execute a tool on the MCP server.

        Falls back to local toolkit execution if the server is disconnected.
        """
        if not self._connected or self._session is None or self._loop is None:
            return super().execute(function_name, arguments)

        # Parse string arguments.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {}
        elif arguments is None:
            arguments = {}

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._session.call_tool(function_name, arguments),
                self._loop,
            )
            result = future.result(timeout=120)
        except Exception as exc:
            return {"error": str(exc), "tool": function_name}

        return self._parse_call_result(result, function_name)

    # ── internal: background event loop ────────────────────────────────────

    def _run_loop(self) -> None:
        """Entry point for the background thread."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._session_lifecycle())
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:
                pass
            self._loop.close()

    async def _session_lifecycle(self) -> None:
        """Open the MCP transport + session and keep it alive until stopped."""
        self._stop_event = asyncio.Event()

        transport_cm = self._create_transport()
        async with transport_cm as streams:
            # stdio yields (read, write); sse also yields (read, write);
            # streamable_http yields (read, write, get_session_id)
            if len(streams) == 3:
                read_stream, write_stream, _ = streams
            else:
                read_stream, write_stream = streams

            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                self._session = session
                self._ready.set()

                # Keep the session alive until disconnect() sets the stop event.
                await self._stop_event.wait()

    def _create_transport(self):
        """Return the appropriate async context manager for the configured transport."""
        if self._transport == "stdio":
            server_params = StdioServerParameters(
                command=self._command,
                args=self._args,
                env=self._env,
                cwd=self._cwd,
            )
            return stdio_client(server_params)

        if self._transport == "sse":
            kwargs: dict[str, Any] = {"url": self._url}
            if self._headers:
                kwargs["headers"] = self._headers
            return sse_client(**kwargs)

        if self._transport == "streamable_http":
            kwargs = {"url": self._url}
            if self._headers:
                kwargs["headers"] = self._headers
            return streamablehttp_client(**kwargs)

        raise ValueError(f"mcp: unsupported transport: {self._transport}")

    def _cleanup_thread(self) -> None:
        """Wait for the background thread to finish."""
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self._loop = None

    # ── internal: tool discovery ───────────────────────────────────────────

    def _populate_tools(self) -> None:
        """Fetch tools from the MCP server and register them as ``tool`` objects."""
        if self._session is None or self._loop is None:
            return

        future = asyncio.run_coroutine_threadsafe(
            self._session.list_tools(),
            self._loop,
        )
        list_result = future.result(timeout=30)

        self.tools.clear()
        for mcp_tool in list_result.tools:
            converted = self._convert_mcp_tool(mcp_tool)
            self.tools[converted.name] = converted

    def _convert_mcp_tool(self, mcp_tool) -> Tool:
        """Convert an MCP ``Tool`` object to a unchain ``Tool``."""
        name = mcp_tool.name
        description = mcp_tool.description or ""
        input_schema = mcp_tool.inputSchema or {}

        # Build parameter list from the JSON Schema.
        parameters: list[ToolParameter] = []
        properties = input_schema.get("properties", {})
        required_set = set(input_schema.get("required", []))

        for param_name, param_schema in properties.items():
            parameters.append(
                ToolParameter(
                    name=param_name,
                    description=param_schema.get("description", f"Parameter {param_name}"),
                    type_=param_schema.get("type", "string"),
                    required=param_name in required_set,
                )
            )

        # Derive requires_confirmation from MCP tool annotations if present.
        annotations = getattr(mcp_tool, "annotations", None)
        requires_confirmation = False
        if annotations is not None:
            # MCP ToolAnnotations may flag destructive / side-effect-ful tools.
            if getattr(annotations, "destructiveHint", None) is True:
                requires_confirmation = True

        def _execute_mcp_tool(**kwargs: Any) -> dict[str, Any]:
            return self.execute(name, kwargs)

        # AgentBuilder flattens Toolkit instances into a single Toolkit by
        # registering Tool objects, so each converted Tool must delegate back
        # to this MCPToolkit instance for execution.
        return tool(
            name=name,
            description=description,
            func=_execute_mcp_tool,
            parameters=parameters,
            requires_confirmation=requires_confirmation,
        )

    # ── internal: result parsing ───────────────────────────────────────────

    @staticmethod
    def _parse_call_result(result, tool_name: str) -> dict[str, Any]:
        """Convert an MCP ``CallToolResult`` to a plain dict.

        Image content blocks are collected into the reserved ``content_blocks``
        key (S0 rich tool result) instead of being dropped. Text-only results
        are byte-identical to the pre-S0 behavior.
        """
        if result.isError:
            error_text = ""
            for block in result.content or []:
                if hasattr(block, "text"):
                    error_text += block.text
            return {"error": error_text or "unknown MCP error", "tool": tool_name}

        parsed = MCPToolkit._parse_success_payload(result)

        content_blocks = MCPToolkit._collect_content_blocks(result.content or [])
        if content_blocks:
            existing = parsed.get("content_blocks") if isinstance(parsed, dict) else None
            blocks = list(existing) if isinstance(existing, list) else []
            blocks.extend(content_blocks)
            parsed = dict(parsed)
            parsed["content_blocks"] = blocks

        return parsed

    @staticmethod
    def _parse_success_payload(result) -> dict[str, Any]:
        """Legacy text/structured payload parsing (unchanged pre-S0 behavior)."""
        # Try structured content first.
        if result.structuredContent is not None:
            return dict(result.structuredContent)

        # Fall back to text content.
        text_parts: list[str] = []
        for block in result.content or []:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        combined = "\n".join(text_parts)

        # If the text is valid JSON, parse it.
        try:
            parsed = json.loads(combined)
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        except (json.JSONDecodeError, ValueError):
            pass

        return {"result": combined}

    @staticmethod
    def _collect_content_blocks(content) -> list[dict[str, Any]]:
        """Map MCP text+image content blocks to the S0 ``content_blocks`` shape.

        Only builds blocks when an image is present (text-only results keep the
        pre-S0 byte-identical behavior). When it does, the accompanying text is
        surfaced too — otherwise "screenshot + caption" MCP results would lose
        the caption, since the model only sees ``content_blocks`` when present.
        Text blocks come before the image (Anthropic guidance: text-first aids
        localization accuracy).
        """
        text_blocks: list[dict[str, Any]] = []
        image_blocks: list[dict[str, Any]] = []
        for block in content:
            btype = getattr(block, "type", None)
            if btype == "image":
                data = getattr(block, "data", None)
                if isinstance(data, str) and data:
                    image_blocks.append({
                        "type": "image",
                        "media_type": getattr(block, "mimeType", None) or "image/png",
                        "data_b64": data,
                    })
                continue
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                text_blocks.append({"type": "text", "text": text})
        if not image_blocks:
            return []
        return text_blocks + image_blocks

    # ── repr ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if self._transport == "stdio":
            target = f"command={self._command!r} args={self._args!r} cwd={self._cwd!r}"
        else:
            target = f"url={self._url!r}"
        status = "connected" if self._connected else "disconnected"
        n_tools = len(self.tools)
        return f"MCPToolkit({target}, {status}, {n_tools} tools)"
