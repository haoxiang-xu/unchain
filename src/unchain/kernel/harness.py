from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from .delta import HarnessDelta
from .state import RunState


RuntimePhase = Literal[
    "bootstrap",
    "before_model",
    "on_context_limit",
    "after_model",
    "on_tool_call",
    "after_tool_batch",
    "before_commit",
    "on_suspend",
    "suspend_persist",
    "on_resume",
    "run_finalizing",
    "finalize_persist",
]


# Event keys whose values are live runtime handles or callbacks, not data.
# The kernel places these into the phase event (see KernelLoop._run_iteration):
# an MCP ``toolkit`` owns a background event loop, daemon thread and session;
# ``loop`` is the live KernelLoop; the rest are callables. None of them are
# serializable, so deep-copying them is meaningless (callables) or a hard crash
# (``cannot pickle 'async_generator'/'_queue.SimpleQueue'`` for live handles).
# They are passed through by reference — consistent with ToolContext.toolkit /
# .loop, which already read them from the un-copied ``raw_event``.
_PASSTHROUGH_EVENT_KEYS = frozenset({
    "toolkit",
    "loop",
    "callback",
    "on_tool_confirm",
    "on_human_input",
    "on_max_iterations",
    "tool_runtime_plugins",
    "emit_stream",
    "model_io",
    "execution_guard",
})


@dataclass(frozen=True)
class HarnessContext:
    state: RunState
    phase: RuntimePhase
    event: dict[str, Any] = field(default_factory=dict)

    @property
    def latest_version_id(self) -> str | None:
        return self.state.latest_version_id

    def latest_messages(self) -> list[dict[str, Any]]:
        return self.state.latest_messages()

    def view_messages(self, version_id: str | None = None) -> list[dict[str, Any]]:
        return self.state.view_messages(version_id)

    def event_payload(self) -> dict[str, Any]:
        copied: dict[str, Any] = {}
        for key, value in self.event.items():
            if key in _PASSTHROUGH_EVENT_KEYS:
                copied[key] = value
                continue
            try:
                copied[key] = copy.deepcopy(value)
            except TypeError:
                # A live/unpicklable object reached the event under an
                # unexpected key; pass it through rather than crashing the run.
                copied[key] = value
        return copied


@runtime_checkable
class RuntimeHarness(Protocol):
    name: str
    phases: tuple[RuntimePhase, ...]
    order: int

    def applies(self, context: HarnessContext) -> bool:
        ...

    def build_delta(self, context: HarnessContext) -> Any:
        ...

    def apply(self, context: HarnessContext) -> Any:
        ...


@dataclass
class BaseRuntimeHarness:
    name: str
    phases: tuple[RuntimePhase, ...]
    order: int = 100

    def applies(self, context: HarnessContext) -> bool:
        return context.phase in self.phases

    def apply(self, context: HarnessContext):
        from ..capabilities import normalize_capability_outcome

        raw_outcome = self.build_delta(context)
        if raw_outcome is None:
            return None
        return normalize_capability_outcome(
            raw_outcome,
            created_by=f"harness.{self.name}",
        )

    def build_delta(self, context: HarnessContext) -> HarnessDelta | None:
        raise NotImplementedError
