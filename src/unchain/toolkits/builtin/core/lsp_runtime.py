from __future__ import annotations

import hashlib
import json
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


_SYMBOL_KIND_NAMES: dict[int, str] = {
    1: "File",
    2: "Module",
    3: "Namespace",
    4: "Package",
    5: "Class",
    6: "Method",
    7: "Property",
    8: "Field",
    9: "Constructor",
    10: "Enum",
    11: "Interface",
    12: "Function",
    13: "Variable",
    14: "Constant",
    15: "String",
    16: "Number",
    17: "Boolean",
    18: "Array",
    19: "Object",
    20: "Key",
    21: "Null",
    22: "EnumMember",
    23: "Struct",
    24: "Event",
    25: "Operator",
    26: "TypeParameter",
}


@dataclass(frozen=True)
class LSPServerSpec:
    language: str
    server_name: str
    command: list[str]


@dataclass
class _OpenDocumentState:
    version: int
    content_sha1: str


@dataclass
class _LSPServerSession:
    root: Path
    language: str
    server_name: str
    command: list[str]
    process: subprocess.Popen[bytes]
    stdout_queue: queue.Queue[dict[str, Any] | None] = field(default_factory=queue.Queue)
    stderr_lines: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    next_request_id: int = 1
    initialized: bool = False
    opened_documents: dict[str, _OpenDocumentState] = field(default_factory=dict)
    reader_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None

    def is_alive(self) -> bool:
        return self.process.poll() is None


class LSPRuntimeError(RuntimeError):
    pass


class LSPRuntime:
    MAX_FILE_BYTES = 10_000_000
    REQUEST_TIMEOUT_SECONDS = 20.0
    _LANGUAGE_BY_SUFFIX: dict[str, tuple[str, str]] = {
        ".py": ("python", "python"),
        ".js": ("javascript", "javascript"),
        ".jsx": ("javascript", "javascriptreact"),
        ".mjs": ("javascript", "javascript"),
        ".cjs": ("javascript", "javascript"),
        ".ts": ("typescript", "typescript"),
        ".tsx": ("typescript", "typescriptreact"),
        ".mts": ("typescript", "typescript"),
        ".cts": ("typescript", "typescript"),
    }

    def __init__(self, workspace_roots: list[Path]):
        self.workspace_roots = [Path(root).resolve() for root in workspace_roots]
        self._sessions: dict[tuple[str, str], _LSPServerSession] = {}

    def execute(
        self,
        *,
        file_path: Path,
        operation: str,
        line: int | None = None,
        character: int | None = None,
        query: str = "",
    ) -> dict[str, Any]:
        root = self._workspace_root_for_path(file_path)
        if root is None:
            raise LSPRuntimeError("file is outside all workspace roots")

        language, language_id = self._detect_language(file_path)
        if not language:
            raise LSPRuntimeError(f"unsupported language for lsp: {file_path.suffix or '<no suffix>'}")

        spec = self._server_spec_for_language(language)
        if spec is None:
            raise LSPRuntimeError(f"no LSP server available for language: {language}")

        raw_bytes = file_path.read_bytes()
        if len(raw_bytes) > self.MAX_FILE_BYTES:
            raise LSPRuntimeError(
                f"file too large for lsp analysis ({len(raw_bytes)} bytes exceeds {self.MAX_FILE_BYTES} byte limit)"
            )
        if b"\x00" in raw_bytes[:8192]:
            raise LSPRuntimeError("binary files are not supported by lsp")
        text = raw_bytes.decode("utf-8", errors="replace")

        session = self._get_or_create_session(root=root, language=language, spec=spec)
        with session.lock:
            self._ensure_initialized_locked(session)
            self._sync_document_locked(session, file_path=file_path, language_id=language_id, text=text)
            raw_result = self._request_locked(
                session,
                method=self._method_for_operation(operation),
                params=self._params_for_operation(
                    operation=operation,
                    file_path=file_path,
                    line=line,
                    character=character,
                    query=query,
                ),
            )

        filtered_result = self._filter_result(operation=operation, result=raw_result, root=root)
        formatted, result_count, file_count = self._format_result(
            operation=operation,
            result=filtered_result,
            root=root,
        )
        return {
            "ok": True,
            "operation": operation,
            "file_path": str(file_path),
            "result": formatted,
            "result_count": result_count,
            "file_count": file_count,
            "language": language,
            "server": spec.server_name,
            "error": "",
        }

    def shutdown(self) -> None:
        for key, session in list(self._sessions.items()):
            self._shutdown_session(session)
            self._sessions.pop(key, None)

    def _workspace_root_for_path(self, path: Path) -> Path | None:
        resolved = path.resolve()
        for root in self.workspace_roots:
            try:
                resolved.relative_to(root)
                return root
            except ValueError:
                continue
        return None

    def _detect_language(self, file_path: Path) -> tuple[str | None, str | None]:
        return self._LANGUAGE_BY_SUFFIX.get(file_path.suffix.lower(), (None, None))

    def _server_spec_for_language(self, language: str) -> LSPServerSpec | None:
        if language == "python":
            pylsp = shutil.which("pylsp")
            if pylsp:
                return LSPServerSpec(language=language, server_name="pylsp", command=[pylsp])
            jedi = shutil.which("jedi-language-server")
            if jedi:
                return LSPServerSpec(language=language, server_name="jedi-language-server", command=[jedi])
            return None

        if language in {"javascript", "typescript"}:
            npx = shutil.which("npx")
            if npx:
                return LSPServerSpec(
                    language=language,
                    server_name="typescript-language-server",
                    command=[
                        npx,
                        "--yes",
                        "--quiet",
                        "-p",
                        "typescript",
                        "-p",
                        "typescript-language-server",
                        "typescript-language-server",
                        "--stdio",
                    ],
                )
            return None

        return None

    def _get_or_create_session(
        self,
        *,
        root: Path,
        language: str,
        spec: LSPServerSpec,
    ) -> _LSPServerSession:
        key = (str(root), language)
        existing = self._sessions.get(key)
        if existing is not None and existing.is_alive():
            return existing
        if existing is not None:
            self._shutdown_session(existing)
            self._sessions.pop(key, None)

        try:
            process = subprocess.Popen(
                spec.command,
                cwd=str(root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                bufsize=0,
            )
        except Exception as exc:
            raise LSPRuntimeError(
                f"failed to start lsp server '{spec.server_name}': {type(exc).__name__}: {exc}"
            ) from exc

        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise LSPRuntimeError(f"failed to attach stdio to lsp server '{spec.server_name}'")

        session = _LSPServerSession(
            root=root,
            language=language,
            server_name=spec.server_name,
            command=list(spec.command),
            process=process,
        )
        session.reader_thread = threading.Thread(
            target=self._stdout_reader,
            args=(session, process.stdout),
            daemon=True,
            name=f"lsp-reader-{language}",
        )
        session.stderr_thread = threading.Thread(
            target=self._stderr_reader,
            args=(session, process.stderr),
            daemon=True,
            name=f"lsp-stderr-{language}",
        )
        session.reader_thread.start()
        session.stderr_thread.start()
        self._sessions[key] = session
        return session

    def _stdout_reader(self, session: _LSPServerSession, pipe: Any) -> None:
        try:
            while True:
                message = self._read_protocol_message(pipe)
                if message is None:
                    break
                session.stdout_queue.put(message)
        finally:
            session.stdout_queue.put(None)

    def _stderr_reader(self, session: _LSPServerSession, pipe: Any) -> None:
        try:
            while True:
                chunk = pipe.readline()
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace").rstrip()
                if text:
                    session.stderr_lines.append(text)
                    if len(session.stderr_lines) > 50:
                        del session.stderr_lines[:-50]
        finally:
            try:
                pipe.close()
            except Exception:
                # Closing must not suppress a read failure or interruption.
                pass

    def _read_protocol_message(self, pipe: Any) -> dict[str, Any] | None:
        headers: dict[str, str] = {}
        while True:
            header_line = pipe.readline()
            if not header_line:
                return None
            if header_line in {b"\r\n", b"\n"}:
                break
            decoded = header_line.decode("utf-8", errors="replace").strip()
            if not decoded or ":" not in decoded:
                continue
            key, value = decoded.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        try:
            content_length = int(headers.get("content-length", "0"))
        except ValueError as exc:
            raise LSPRuntimeError("invalid Content-Length header from lsp server") from exc
        if content_length <= 0:
            return None
        payload = pipe.read(content_length)
        if not payload:
            return None
        return json.loads(payload.decode("utf-8", errors="replace"))

    def _ensure_initialized_locked(self, session: _LSPServerSession) -> None:
        if session.initialized:
            return
        initialize_result = self._request_locked(
            session,
            method="initialize",
            params={
                "processId": None,
                "rootUri": session.root.as_uri(),
                "capabilities": {},
                "workspaceFolders": [{"uri": session.root.as_uri(), "name": session.root.name}],
                "initializationOptions": {},
            },
        )
        if not isinstance(initialize_result, dict):
            raise LSPRuntimeError(f"initialize failed for lsp server '{session.server_name}'")
        self._notify_locked(session, method="initialized", params={})
        session.initialized = True

    def _sync_document_locked(
        self,
        session: _LSPServerSession,
        *,
        file_path: Path,
        language_id: str,
        text: str,
    ) -> None:
        uri = file_path.resolve().as_uri()
        content_sha1 = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
        existing = session.opened_documents.get(uri)
        if existing is None:
            session.opened_documents[uri] = _OpenDocumentState(version=1, content_sha1=content_sha1)
            self._notify_locked(
                session,
                method="textDocument/didOpen",
                params={
                    "textDocument": {
                        "uri": uri,
                        "languageId": language_id,
                        "version": 1,
                        "text": text,
                    }
                },
            )
            return

        if existing.content_sha1 == content_sha1:
            return

        next_version = existing.version + 1
        session.opened_documents[uri] = _OpenDocumentState(version=next_version, content_sha1=content_sha1)
        self._notify_locked(
            session,
            method="textDocument/didChange",
            params={
                "textDocument": {
                    "uri": uri,
                    "version": next_version,
                },
                "contentChanges": [{"text": text}],
            },
        )

    def _request_locked(
        self,
        session: _LSPServerSession,
        *,
        method: str,
        params: dict[str, Any],
    ) -> Any:
        if not session.is_alive():
            raise LSPRuntimeError(f"lsp server '{session.server_name}' is not running")

        request_id = session.next_request_id
        session.next_request_id += 1
        self._write_locked(
            session,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            },
        )
        while True:
            try:
                message = session.stdout_queue.get(timeout=self.REQUEST_TIMEOUT_SECONDS)
            except queue.Empty as exc:
                raise LSPRuntimeError(
                    f"timed out waiting for lsp response from '{session.server_name}' for {method}"
                ) from exc
            if message is None:
                stderr_text = "; ".join(session.stderr_lines[-3:])
                extra = f" ({stderr_text})" if stderr_text else ""
                raise LSPRuntimeError(f"lsp server '{session.server_name}' exited unexpectedly{extra}")
            if "id" not in message:
                continue
            if message.get("id") != request_id:
                continue
            if message.get("error"):
                error_obj = message["error"]
                if isinstance(error_obj, dict):
                    error_message = str(error_obj.get("message") or "unknown lsp error")
                else:
                    error_message = str(error_obj)
                raise LSPRuntimeError(f"lsp request failed for {method}: {error_message}")
            return message.get("result")

    def _notify_locked(self, session: _LSPServerSession, *, method: str, params: dict[str, Any]) -> None:
        self._write_locked(
            session,
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            },
        )

    def _write_locked(self, session: _LSPServerSession, payload: dict[str, Any]) -> None:
        if session.process.stdin is None:
            raise LSPRuntimeError(f"lsp server '{session.server_name}' stdin is unavailable")
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        session.process.stdin.write(header)
        session.process.stdin.write(body)
        session.process.stdin.flush()

    def _shutdown_session(self, session: _LSPServerSession) -> None:
        if session.is_alive():
            with session.lock:
                try:
                    if session.initialized:
                        self._request_locked(session, method="shutdown", params={})
                        self._notify_locked(session, method="exit", params={})
                except Exception:
                    # A failed protocol shutdown must still proceed to operating-system process termination.
                    pass
            try:
                session.process.terminate()
                session.process.wait(timeout=2.0)
            except Exception:
                try:
                    session.process.kill()
                    session.process.wait(timeout=1.0)
                except Exception:
                    # Process teardown is best effort; continue the bounded joins even if the process cannot be reaped.
                    pass
        if session.reader_thread is not None:
            session.reader_thread.join(timeout=0.2)
        if session.stderr_thread is not None:
            session.stderr_thread.join(timeout=0.2)

    def _method_for_operation(self, operation: str) -> str:
        if operation == "goToDefinition":
            return "textDocument/definition"
        if operation == "findReferences":
            return "textDocument/references"
        if operation == "hover":
            return "textDocument/hover"
        if operation == "documentSymbol":
            return "textDocument/documentSymbol"
        if operation == "workspaceSymbol":
            return "workspace/symbol"
        raise LSPRuntimeError(f"unsupported lsp operation: {operation}")

    def _params_for_operation(
        self,
        *,
        operation: str,
        file_path: Path,
        line: int | None,
        character: int | None,
        query: str,
    ) -> dict[str, Any]:
        uri = file_path.resolve().as_uri()
        position = None
        if line is not None and character is not None:
            position = {"line": line - 1, "character": character - 1}

        if operation == "goToDefinition":
            return {"textDocument": {"uri": uri}, "position": position}
        if operation == "findReferences":
            return {
                "textDocument": {"uri": uri},
                "position": position,
                "context": {"includeDeclaration": True},
            }
        if operation == "hover":
            return {"textDocument": {"uri": uri}, "position": position}
        if operation == "documentSymbol":
            return {"textDocument": {"uri": uri}}
        if operation == "workspaceSymbol":
            return {"query": str(query or "")}
        raise LSPRuntimeError(f"unsupported lsp operation: {operation}")

    def _filter_result(self, *, operation: str, result: Any, root: Path) -> Any:
        if operation in {"goToDefinition", "findReferences"}:
            locations = self._normalize_locations(result)
            return self._filter_locations(locations, root=root)
        if operation == "workspaceSymbol":
            symbols = result if isinstance(result, list) else []
            filtered_symbols: list[dict[str, Any]] = []
            candidate_paths = [
                self._uri_to_path(symbol.get("location", {}).get("uri"))
                for symbol in symbols
                if isinstance(symbol, dict)
            ]
            ignored_paths = self._gitignored_paths([path for path in candidate_paths if path is not None], root=root)
            for symbol in symbols:
                if not isinstance(symbol, dict):
                    continue
                location = symbol.get("location")
                if not isinstance(location, dict):
                    filtered_symbols.append(symbol)
                    continue
                file_path = self._uri_to_path(location.get("uri"))
                if file_path is None:
                    continue
                if not self._path_within_root(file_path, root):
                    continue
                if file_path in ignored_paths:
                    continue
                filtered_symbols.append(symbol)
            return filtered_symbols
        return result

    def _normalize_locations(self, result: Any) -> list[dict[str, Any]]:
        if result is None:
            return []
        items = result if isinstance(result, list) else [result]
        normalized: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if "targetUri" in item:
                normalized.append(
                    {
                        "uri": item.get("targetUri"),
                        "range": item.get("targetSelectionRange") or item.get("targetRange"),
                    }
                )
            else:
                normalized.append(item)
        return normalized

    def _filter_locations(self, locations: list[dict[str, Any]], *, root: Path) -> list[dict[str, Any]]:
        candidate_paths = [self._uri_to_path(location.get("uri")) for location in locations]
        ignored_paths = self._gitignored_paths([path for path in candidate_paths if path is not None], root=root)
        filtered: list[dict[str, Any]] = []
        for location in locations:
            file_path = self._uri_to_path(location.get("uri"))
            if file_path is None:
                continue
            if not self._path_within_root(file_path, root):
                continue
            if file_path in ignored_paths:
                continue
            filtered.append(location)
        return filtered

    def _gitignored_paths(self, paths: list[Path], *, root: Path) -> set[Path]:
        relative_paths: list[str] = []
        absolute_paths: dict[str, Path] = {}
        for path in paths:
            try:
                relative = str(path.resolve().relative_to(root))
            except ValueError:
                continue
            if not relative:
                continue
            absolute_paths[relative] = path.resolve()
            relative_paths.append(relative)
        if not relative_paths:
            return set()

        git_executable = shutil.which("git")
        if git_executable is None:
            return set()

        try:
            completed = subprocess.run(
                [git_executable, "-C", str(root), "check-ignore", "--stdin"],
                input="\n".join(relative_paths).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except Exception:
            return set()

        if completed.returncode not in {0, 1}:
            return set()

        ignored: set[Path] = set()
        for line in completed.stdout.decode("utf-8", errors="replace").splitlines():
            relative = line.strip()
            if relative and relative in absolute_paths:
                ignored.add(absolute_paths[relative])
        return ignored

    def _path_within_root(self, path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root)
            return True
        except ValueError:
            return False

    def _uri_to_path(self, uri: Any) -> Path | None:
        if not isinstance(uri, str) or not uri.strip():
            return None
        parsed = urlparse(uri)
        if parsed.scheme and parsed.scheme != "file":
            return None
        raw_path = parsed.path or uri.replace("file://", "", 1)
        if raw_path.startswith("/") and len(raw_path) > 3 and raw_path[2] == ":":
            raw_path = raw_path[1:]
        resolved = Path(unquote(raw_path)).resolve()
        return resolved

    def _format_result(self, *, operation: str, result: Any, root: Path) -> tuple[str, int, int]:
        if operation == "goToDefinition":
            return self._format_definition_result(result, root=root)
        if operation == "findReferences":
            return self._format_references_result(result, root=root)
        if operation == "hover":
            return self._format_hover_result(result)
        if operation == "documentSymbol":
            return self._format_document_symbols_result(result, root=root)
        if operation == "workspaceSymbol":
            return self._format_workspace_symbols_result(result, root=root)
        return ("Unsupported lsp operation.", 0, 0)

    def _format_definition_result(self, result: Any, *, root: Path) -> tuple[str, int, int]:
        locations = self._normalize_locations(result)
        if not locations:
            return (
                "No definition found. This may occur if the cursor is not on a symbol or the server has not indexed the file.",
                0,
                0,
            )
        if len(locations) == 1:
            return (f"Defined in {self._format_location(locations[0], root=root)}", 1, 1)
        lines = [f"Found {len(locations)} definitions:"]
        for location in locations:
            lines.append(f"  {self._format_location(location, root=root)}")
        return ("\n".join(lines), len(locations), self._count_unique_location_files(locations))

    def _format_references_result(self, result: Any, *, root: Path) -> tuple[str, int, int]:
        locations = self._normalize_locations(result)
        if not locations:
            return (
                "No references found. This may occur if the symbol has no usages or the server has not indexed the workspace.",
                0,
                0,
            )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for location in locations:
            file_label = self._format_path_from_uri(location.get("uri"), root=root)
            grouped.setdefault(file_label, []).append(location)

        lines = [f"Found {len(locations)} references across {len(grouped)} files:"]
        for file_label, file_locations in grouped.items():
            lines.append(f"\n{file_label}:")
            for location in file_locations:
                range_info = location.get("range") or {}
                start = range_info.get("start") or {}
                lines.append(
                    f"  Line {int(start.get('line', 0)) + 1}:{int(start.get('character', 0)) + 1}"
                )
        return ("\n".join(lines), len(locations), len(grouped))

    def _format_hover_result(self, result: Any) -> tuple[str, int, int]:
        if not isinstance(result, dict):
            return (
                "No hover information available. This may occur if the cursor is not on a symbol or the server is not ready.",
                0,
                0,
            )
        contents = self._extract_hover_text(result.get("contents"))
        if not contents:
            return (
                "No hover information available. This may occur if the cursor is not on a symbol or the server is not ready.",
                0,
                0,
            )
        result_range = result.get("range")
        if isinstance(result_range, dict):
            start = result_range.get("start") or {}
            return (
                f"Hover info at {int(start.get('line', 0)) + 1}:{int(start.get('character', 0)) + 1}:\n\n{contents}",
                1,
                1,
            )
        return (contents, 1, 1)

    def _format_document_symbols_result(self, result: Any, *, root: Path) -> tuple[str, int, int]:
        symbols = result if isinstance(result, list) else []
        if not symbols:
            return (
                "No symbols found in document. This may occur if the file is empty or not supported by the server.",
                0,
                0,
            )
        if isinstance(symbols[0], dict) and isinstance(symbols[0].get("location"), dict):
            formatted, result_count, file_count = self._format_workspace_symbols_result(symbols, root=root)
            return (formatted.replace("Found ", "Document symbols: ", 1), result_count, file_count or 1)

        lines = ["Document symbols:"]
        for symbol in symbols:
            lines.extend(self._format_document_symbol_node(symbol, indent=0))
        return ("\n".join(lines), self._count_document_symbols(symbols), 1)

    def _format_workspace_symbols_result(self, result: Any, *, root: Path) -> tuple[str, int, int]:
        symbols = [symbol for symbol in (result if isinstance(result, list) else []) if isinstance(symbol, dict)]
        if not symbols:
            return (
                "No symbols found in workspace. This may occur if the workspace is empty or not indexed yet.",
                0,
                0,
            )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for symbol in symbols:
            location = symbol.get("location")
            uri = location.get("uri") if isinstance(location, dict) else None
            file_label = self._format_path_from_uri(uri, root=root) if uri else "<unknown location>"
            grouped.setdefault(file_label, []).append(symbol)

        lines = [f"Found {len(symbols)} symbols in workspace:"]
        for file_label, file_symbols in grouped.items():
            lines.append(f"\n{file_label}:")
            for symbol in file_symbols:
                location = symbol.get("location") or {}
                range_info = location.get("range") or {}
                start = range_info.get("start") or {}
                kind = _SYMBOL_KIND_NAMES.get(int(symbol.get("kind", 0)), "Unknown")
                container = symbol.get("containerName")
                suffix = f" in {container}" if isinstance(container, str) and container else ""
                lines.append(
                    f"  {symbol.get('name', '<unnamed>')} ({kind}) - Line {int(start.get('line', 0)) + 1}{suffix}"
                )
        return ("\n".join(lines), len(symbols), len(grouped))

    def _format_document_symbol_node(self, symbol: Any, *, indent: int) -> list[str]:
        if not isinstance(symbol, dict):
            return []
        kind = _SYMBOL_KIND_NAMES.get(int(symbol.get("kind", 0)), "Unknown")
        symbol_range = symbol.get("range") or {}
        start = symbol_range.get("start") or {}
        detail = symbol.get("detail")
        label = f"{'  ' * indent}{symbol.get('name', '<unnamed>')} ({kind})"
        if isinstance(detail, str) and detail:
            label += f" {detail}"
        label += f" - Line {int(start.get('line', 0)) + 1}"
        lines = [label]
        children = symbol.get("children")
        if isinstance(children, list):
            for child in children:
                lines.extend(self._format_document_symbol_node(child, indent=indent + 1))
        return lines

    def _count_document_symbols(self, symbols: list[Any]) -> int:
        count = 0
        for symbol in symbols:
            if not isinstance(symbol, dict):
                continue
            count += 1
            children = symbol.get("children")
            if isinstance(children, list):
                count += self._count_document_symbols(children)
        return count

    def _count_unique_location_files(self, locations: list[dict[str, Any]]) -> int:
        paths = {str(self._uri_to_path(location.get("uri"))) for location in locations if self._uri_to_path(location.get("uri"))}
        return len(paths)

    def _format_location(self, location: dict[str, Any], *, root: Path) -> str:
        range_info = location.get("range") or {}
        start = range_info.get("start") or {}
        return (
            f"{self._format_path_from_uri(location.get('uri'), root=root)}:"
            f"{int(start.get('line', 0)) + 1}:{int(start.get('character', 0)) + 1}"
        )

    def _format_path_from_uri(self, uri: Any, *, root: Path) -> str:
        file_path = self._uri_to_path(uri)
        if file_path is None:
            return "<unknown location>"
        try:
            return str(file_path.relative_to(root))
        except ValueError:
            return str(file_path)

    def _extract_hover_text(self, contents: Any) -> str:
        if isinstance(contents, str):
            return contents
        if isinstance(contents, dict):
            if isinstance(contents.get("value"), str):
                return contents["value"]
            if isinstance(contents.get("contents"), str):
                return contents["contents"]
        if isinstance(contents, list):
            parts = [self._extract_hover_text(item) for item in contents]
            return "\n\n".join(part for part in parts if part)
        return ""


__all__ = ["LSPRuntime", "LSPRuntimeError", "LSPServerSpec"]
