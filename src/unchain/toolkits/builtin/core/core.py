from __future__ import annotations

from pathlib import Path
from typing import Any

from ...base import BuiltinToolkit
from ....input.human_input import build_ask_user_question_tool
from ....tools.models import (
    ToolConfirmationPolicy,
    ToolExecutionContext,
    ToolHistoryOptimizationContext,
    ToolPromptSpec,
)
from .coding_backend import CoreCodingBackend
from .web_backend import CoreWebBackend


class CoreToolkit(BuiltinToolkit):
    """Core builtin toolkit for coding, shell, web fetch, LSP, and structured user questions."""

    def __init__(
        self,
        *,
        workspace_root: str | Path | None = None,
        workspace_roots: list[str | Path] | None = None,
    ) -> None:
        super().__init__(workspace_root=workspace_root, workspace_roots=workspace_roots)
        self._coding_backend = CoreCodingBackend(
            workspace_roots=self.workspace_roots,
            execution_context_provider=lambda: self.current_execution_context,
        )
        self._web_backend = CoreWebBackend(
            runtime_config_provider=self._tool_runtime_config_for,
            execution_context_provider=lambda: self.current_execution_context,
        )
        self._register_tools()

    def ask_user_question(
        self,
        title: str,
        question: str,
        selection_mode: str,
        options: list[dict[str, Any]],
        allow_other: bool = False,
        other_label: str | None = None,
        other_placeholder: str | None = None,
        min_selected: int | None = None,
        max_selected: int | None = None,
    ) -> dict[str, Any]:
        """Ask the user a structured selector question and suspend the run until they respond."""
        return {"error": "ask_user_question is a reserved runtime tool and cannot be executed directly"}

    def _register_tools(self) -> None:
        self.register(build_ask_user_question_tool())
        self.register(
            self.read,
            description="Read a UTF-8 text file by absolute path with line-numbered output and optional line slicing.",
            history_arguments_optimizer=self._compact_read_args,
            history_result_optimizer=self._compact_read_result,
            output_policy="head_tail",
        )
        self.register(
            self.write,
            description="Create or fully overwrite a UTF-8 text file by absolute path. Existing files must be fully read first.",
            requires_confirmation=True,
            confirmation_resolver=self._resolve_write_confirmation,
            history_arguments_optimizer=self._compact_write_args,
            history_result_optimizer=self._compact_mutation_result,
            output_policy="default",
        )
        self.register(
            self.edit,
            description="Replace one unique string match, or all matches when requested, in an existing UTF-8 text file by absolute path.",
            requires_confirmation=True,
            confirmation_resolver=self._resolve_edit_confirmation,
            history_arguments_optimizer=self._compact_edit_args,
            history_result_optimizer=self._compact_mutation_result,
            output_policy="default",
        )
        self.register(
            self.glob,
            description="List files matching a glob pattern inside the workspace, sorted by most recently modified first.",
            history_result_optimizer=self._compact_glob_result,
            output_policy="head_tail",
        )
        self.register(
            self.grep,
            description="Search UTF-8 text files inside the workspace with regex, optional glob filters, and paginated result modes.",
            history_result_optimizer=self._compact_grep_result,
            output_policy="head_tail",
        )
        self.register(
            self.web_fetch,
            name="web_fetch",
            description=(
                "Fetch a public web page over HTTP(S). Raw mode returns the page text "
                "converted to Markdown with scripts, styles and embedded data removed, "
                "paginated by offset/max_chars; a truncated result ends with a notice and "
                "next_offset — continue with offset=next_offset before answering. Extract "
                "mode runs a runtime-configured extraction model."
            ),
            requires_confirmation=True,
            history_arguments_optimizer=self._compact_web_fetch_args,
            history_result_optimizer=self._compact_web_fetch_result,
            output_policy="head_tail",
            prompt_spec=ToolPromptSpec(
                purpose=(
                    "Read a public web page and return its text as Markdown, or extract "
                    "specific information with a configured model."
                ),
                when_to_use=(
                    "The answer depends on the current content of a specific public URL.",
                ),
                when_not_to_use=(
                    "A local file or an earlier fetch already contains the needed text.",
                ),
                examples=(
                    'web_fetch(url="https://example.com/docs")',
                    'web_fetch(url="https://example.com/docs", offset=20000)',
                ),
                advanced_tips=(
                    "A result whose text ends with a [web_fetch notice: …] line is "
                    "incomplete: call web_fetch again with offset=next_offset (repeat until "
                    "next_offset is null) before answering, or say you could not read the "
                    "whole page.",
                    "Never answer from a truncated page as if it were complete, and never "
                    "cite a URL you did not fetch.",
                    "Prefer raw.githubusercontent.com or a docs page over a rendered app "
                    "page when both exist.",
                ),
            ),
        )
        self.register(
            self.shell,
            description="Run a shell command, poll or wait for a background task, or kill it within the workspace.",
            requires_confirmation=True,
            confirmation_resolver=self._resolve_shell_confirmation,
            history_arguments_optimizer=self._compact_shell_args,
            history_result_optimizer=self._compact_shell_result,
            output_policy="head_tail",
        )
        self.register(
            self.lsp,
            description="Query a language server for definitions, references, hover text, and symbols for Python or TS/JS files.",
            history_arguments_optimizer=self._compact_lsp_args,
            history_result_optimizer=self._compact_lsp_result,
            output_policy="head_tail",
        )

    def read(self, path: str, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
        """Read a UTF-8 text file by absolute path with optional line slicing.

        Args:
            path: Absolute path to the file inside the workspace roots.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return. Omit for the full file.
        """
        return self._coding_backend.read(path=path, offset=offset, limit=limit)

    def write(self, path: str, content: str) -> dict[str, Any]:
        """Create or fully overwrite a UTF-8 text file by absolute path.

        Args:
            path: Absolute path to the file inside the workspace roots.
            content: Full replacement content for the file.
        """
        return self._coding_backend.write(path=path, content=content)

    def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        """Replace one unique string match, or all matches, in a UTF-8 text file.

        Args:
            path: Absolute path to the file inside the workspace roots.
            old_string: Existing string to replace.
            new_string: Replacement string.
            replace_all: Replace every occurrence instead of requiring a unique match.
        """
        return self._coding_backend.edit(
            path=path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
        )

    def glob(self, pattern: str, path: str | None = None) -> dict[str, Any]:
        """List files matching a glob pattern inside the workspace.

        Args:
            pattern: Glob pattern relative to the base path, for example `**/*.py`.
            path: Optional absolute base directory or file path. Defaults to the first workspace root.
        """
        return self._coding_backend.glob(pattern=pattern, path=path)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: str = "content",
        context: int = 0,
        head_limit: int = 50,
        offset: int = 0,
        case_sensitive: bool = True,
        multiline: bool = False,
    ) -> dict[str, Any]:
        """Search UTF-8 text files inside the workspace with regex.

        Args:
            pattern: Regex pattern to search for.
            path: Optional absolute base directory or file path. Defaults to the first workspace root.
            glob: Optional glob filter applied to relative file paths.
            output_mode: One of `content`, `files_with_matches`, or `count`.
            context: Number of surrounding lines to include for `content` mode.
            head_limit: Maximum number of results to return.
            offset: Pagination offset for `content` and `files_with_matches` modes.
            case_sensitive: When false, search using case-insensitive regex.
            multiline: When true, allow regex matches to span newlines.
        """
        return self._coding_backend.grep(
            pattern=pattern,
            path=path,
            glob=glob,
            output_mode=output_mode,
            context=context,
            head_limit=head_limit,
            offset=offset,
            case_sensitive=case_sensitive,
            multiline=multiline,
        )

    def web_fetch(
        self,
        url: str,
        mode: str = "raw",
        prompt: str | None = None,
        offset: int = 0,
        max_chars: int = 20000,
    ) -> dict[str, Any]:
        """Fetch a public web page as Markdown text (scripts and styles removed), or run a configured extraction model; a truncated result ends with a notice and next_offset.

        Args:
            url: Public HTTP(S) URL to fetch.
            mode: Either `raw` (page text as Markdown, paginated) or `extract` (needs a runtime-configured extraction model).
            prompt: Extraction prompt used only when `mode="extract"`.
            offset: Zero-based character offset for `raw` mode pagination. Pass the `next_offset` of a truncated result to continue reading.
            max_chars: Maximum characters to return in `raw` mode. Capped at 50,000. A truncated result ends with a notice and sets `next_offset`.
        """
        return self._web_backend.fetch(
            url=url,
            mode=mode,
            prompt=prompt,
            offset=offset,
            max_chars=max_chars,
        )

    def _tool_runtime_config_for(self, tool_name: str) -> dict[str, Any]:
        context = self.current_execution_context
        config = getattr(context, "tool_runtime_config", None)
        if not isinstance(config, dict):
            return {}
        tool_config = config.get(tool_name)
        return dict(tool_config) if isinstance(tool_config, dict) else {}

    def lsp(
        self,
        operation: str,
        file_path: str,
        line: int | None = None,
        character: int | None = None,
        query: str = "",
    ) -> dict[str, Any]:
        """Run an LSP operation for Python or TS/JS files inside the workspace.

        Args:
            operation: One of `goToDefinition`, `findReferences`, `hover`, `documentSymbol`, or `workspaceSymbol`.
            file_path: Absolute path to the file inside the workspace roots.
            line: One-based line number for cursor-based operations.
            character: One-based character offset for cursor-based operations.
            query: Optional workspace symbol query used only when `operation="workspaceSymbol"`.
        """
        return self._coding_backend.lsp(
            operation=operation,
            file_path=file_path,
            line=line,
            character=character,
            query=query,
        )

    def shell(
        self,
        action: str,
        command: str = "",
        cwd: str | None = None,
        timeout_ms: int = 120000,
        run_in_background: bool = False,
        max_output_chars: int = 20000,
        yield_time_ms: int = 300,
        task_id: str = "",
    ) -> dict[str, Any]:
        """Run, poll, wait for, or kill a shell task inside the workspace.

        Args:
            action: One of `run`, `poll`, `wait`, or `kill`.
            command: Shell command used when `action="run"`.
            cwd: Optional working directory. Relative paths resolve from the session cwd.
            timeout_ms: Maximum runtime for `run`, or maximum blocking time for `wait`. Clamped to 1s..600s.
            run_in_background: When true, start a background task and return a `task_id`.
            max_output_chars: Maximum characters returned per stream, capped at 100,000.
            yield_time_ms: Small delay before returning a background task so early output can accumulate.
            task_id: Background task id used by `poll`, `wait`, and `kill`.
        """
        return self._coding_backend.shell(
            action=action,
            command=command,
            cwd=cwd,
            timeout_ms=timeout_ms,
            run_in_background=run_in_background,
            max_output_chars=max_output_chars,
            yield_time_ms=yield_time_ms,
            task_id=task_id,
        )

    def _resolve_shell_confirmation(
        self,
        arguments: dict[str, Any],
        execution_context: ToolExecutionContext | None,
    ) -> ToolConfirmationPolicy:
        return self._coding_backend._resolve_shell_confirmation(arguments, execution_context)

    def _resolve_write_confirmation(
        self,
        arguments: dict[str, Any],
        execution_context: ToolExecutionContext | None,
    ) -> ToolConfirmationPolicy:
        return self._coding_backend._resolve_write_confirmation(arguments, execution_context)

    def _resolve_edit_confirmation(
        self,
        arguments: dict[str, Any],
        execution_context: ToolExecutionContext | None,
    ) -> ToolConfirmationPolicy:
        return self._coding_backend._resolve_edit_confirmation(arguments, execution_context)

    def shutdown(self) -> None:
        self._coding_backend.shutdown()

    def _compact_read_args(self, payload: Any, context: ToolHistoryOptimizationContext) -> Any:
        return self._coding_backend.compact_read_args(payload, context)

    def _compact_read_result(self, payload: Any, context: ToolHistoryOptimizationContext) -> Any:
        return self._coding_backend.compact_read_result(payload, context)

    def _compact_write_args(self, payload: Any, context: ToolHistoryOptimizationContext) -> Any:
        return self._coding_backend.compact_write_args(payload, context)

    def _compact_edit_args(self, payload: Any, context: ToolHistoryOptimizationContext) -> Any:
        return self._coding_backend.compact_edit_args(payload, context)

    def _compact_mutation_result(self, payload: Any, context: ToolHistoryOptimizationContext) -> Any:
        return self._coding_backend.compact_mutation_result(payload, context)

    def _compact_glob_result(self, payload: Any, context: ToolHistoryOptimizationContext) -> Any:
        return self._coding_backend.compact_glob_result(payload, context)

    def _compact_grep_result(self, payload: Any, context: ToolHistoryOptimizationContext) -> Any:
        return self._coding_backend.compact_grep_result(payload, context)

    def _compact_web_fetch_args(self, payload: Any, context: ToolHistoryOptimizationContext) -> Any:
        return self._web_backend.compact_args(payload, context)

    def _compact_web_fetch_result(self, payload: Any, context: ToolHistoryOptimizationContext) -> Any:
        return self._web_backend.compact_result(payload, context)

    def _compact_shell_args(self, payload: Any, context: ToolHistoryOptimizationContext) -> Any:
        return self._coding_backend.compact_shell_args(payload, context)

    def _compact_shell_result(self, payload: Any, context: ToolHistoryOptimizationContext) -> Any:
        return self._coding_backend.compact_shell_result(payload, context)

    def _compact_lsp_args(self, payload: Any, context: ToolHistoryOptimizationContext) -> Any:
        return self._coding_backend.compact_lsp_args(payload, context)

    def _compact_lsp_result(self, payload: Any, context: ToolHistoryOptimizationContext) -> Any:
        return self._coding_backend.compact_lsp_result(payload, context)


__all__ = ["CoreToolkit"]
