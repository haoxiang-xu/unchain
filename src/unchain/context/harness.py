from __future__ import annotations

from dataclasses import dataclass, field

from ..kernel.delta import HarnessDelta, ReplaceSpanOp
from ..kernel.harness import BaseRuntimeHarness, HarnessContext, RuntimePhase
from .runtime import ContextRuntime, _CONTEXT_EXECUTION_BINDING_AUTHORITY


@dataclass
class ContextExecutionBindingHarness(BaseRuntimeHarness):
    """Resolve an attempt-scoped durable bundle before any runtime callback."""

    name: str = "context_v2_execution_binding"
    phases: tuple[RuntimePhase, ...] = ("bootstrap",)
    order: int = -1_000
    runtime: ContextRuntime = field(repr=False, kw_only=True)
    shadow_mode: bool = field(default=False, kw_only=True)

    def build_delta(self, context: HarnessContext) -> None:
        self.runtime.bind_context(
            context,
            _binding_authority=_CONTEXT_EXECUTION_BINDING_AUTHORITY,
            _shadow_mode=self.shadow_mode,
        )
        return None


@dataclass
class ContextCompilerHarness(BaseRuntimeHarness):
    """The sole model-context mutator for an explicitly owned V2 run."""

    name: str = "context_v2_compiler"
    phases: tuple[RuntimePhase, ...] = ("before_model", "on_context_limit")
    order: int = 900
    runtime: ContextRuntime = field(repr=False, kw_only=True)

    @property
    def semantic_context_owner(self) -> str:
        return self.runtime.owner_id

    def build_delta(self, context: HarnessContext) -> HarnessDelta:
        if context.phase == "on_context_limit":
            event = self.runtime.record_provider_context_recovery(context)
            return HarnessDelta(
                created_by=f"context.{self.name}",
                base_version_id=context.latest_version_id,
                state_updates={"provider_context_recovery_ack": (
                    {"iteration": context.state.iteration, "event_id": event.event_id}
                    if event is not None else None
                )},
            )
        result = self.runtime.compile_context(context)
        result_data = result.to_dict()
        messages = result_data["messages"]
        envelope = result_data["envelope"]
        current_messages = context.latest_messages()
        ops = ()
        if messages != current_messages:
            ops = (
                ReplaceSpanOp(
                    start=0,
                    end=len(current_messages),
                    messages=messages,
                ),
            )
        return HarnessDelta(
            created_by=f"context.{self.name}",
            base_version_id=context.latest_version_id,
            ops=ops,
            state_updates={
                "context_v2": {
                    "owner_id": self.semantic_context_owner,
                    "last_build": envelope,
                    "diagnostics": result_data["diagnostics"],
                }
            },
            trace={
                "semantic_context_owner": self.semantic_context_owner,
                "context_build_id": envelope["build_id"],
                "context_build_status": envelope["status"],
            },
        )


@dataclass
class ContextShadowCompilerHarness(BaseRuntimeHarness):
    """Persist a V2 build without changing the provider-visible context."""

    name: str = "context_v2_shadow_compiler"
    phases: tuple[RuntimePhase, ...] = ("before_model",)
    order: int = 900
    runtime: ContextRuntime = field(repr=False, kw_only=True)

    def build_delta(self, context: HarnessContext) -> HarnessDelta:
        current_messages = context.latest_messages()
        result_data = self.runtime.compile_context(context).to_dict()
        compiled_messages = result_data["messages"]
        envelope = result_data["envelope"]
        would_replace_messages = compiled_messages != current_messages
        diagnostics = {
            "mode": "shadow",
            "owner_id": self.runtime.owner_id,
            "last_build": envelope,
            "diagnostics": result_data["diagnostics"],
            "observed_message_count": len(current_messages),
            "compiled_message_count": len(compiled_messages),
            "would_replace_messages": would_replace_messages,
        }
        return HarnessDelta(
            created_by=f"context.{self.name}",
            base_version_id=context.latest_version_id,
            ops=(),
            state_updates={"context_v2_shadow": diagnostics},
            trace={
                "context_shadow": True,
                "context_build_id": envelope["build_id"],
                "context_build_status": envelope["status"],
                "would_replace_messages": would_replace_messages,
            },
        )


__all__ = [
    "ContextCompilerHarness",
    "ContextExecutionBindingHarness",
    "ContextShadowCompilerHarness",
]
