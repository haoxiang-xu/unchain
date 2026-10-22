"""Optional Memory V2 module attachment for Unchain agents.

The implementation lives with the Memory subsystem so the core agent module
package does not own Memory-specific policy or types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..agent.modules.base import BaseAgentModule
from ..kernel.types import KernelRunResult
from ..runtime.module_context import (
    AgentRuntimeContext,
    ExecutionIdentity,
    ModuleGrant,
)
from .curator import RootRunCompletion
from .curator.host import MemoryAgentHostAdapter
from .toolkit import (
    MemoryToolkitRunBinding,
    NormalMemoryToolkitCapabilities,
)
from .toolkit.policy import MemoryLearningPromptHarness


MEMORY_V2_MODULE_KEY = "memory_v2"
MEMORY_CONTEXT_READ = "memory.context.read"
MEMORY_WORKSPACE_READ = "memory.workspace.read"
MEMORY_CANDIDATE_PROPOSE = "memory.candidate.propose"
MEMORY_EXECUTION_COMPLETE = "memory.execution.complete"
MEMORY_V2_CAPABILITIES = frozenset(
    {
        MEMORY_CONTEXT_READ,
        MEMORY_WORKSPACE_READ,
        MEMORY_CANDIDATE_PROPOSE,
        MEMORY_EXECUTION_COMPLETE,
    }
)
_TOOLS_BY_CAPABILITY = {
    MEMORY_CONTEXT_READ: frozenset(
        {"context_content_read", "context_checkpoint_events_read"}
    ),
    MEMORY_WORKSPACE_READ: frozenset(
        {"memory_list", "memory_search", "memory_read"}
    ),
    MEMORY_CANDIDATE_PROPOSE: frozenset({"memory_propose"}),
}


class MemoryV2ModuleError(RuntimeError):
    """Stable host attachment or terminal identity failure."""


@dataclass(frozen=True)
class MemoryAttachmentRequest:
    agent_name: str
    mode: str
    identity: ExecutionIdentity
    grant: ModuleGrant

    def __post_init__(self) -> None:
        for field_name in ("agent_name", "mode"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be text")
        if not self.agent_name:
            raise ValueError("agent_name is required")
        if not self.mode:
            raise ValueError("mode is required")
        if not isinstance(self.identity, ExecutionIdentity):
            raise TypeError("identity must be an ExecutionIdentity")
        if not isinstance(self.grant, ModuleGrant):
            raise TypeError("grant must be a ModuleGrant")
        if self.grant.module_key != MEMORY_V2_MODULE_KEY:
            raise ValueError("grant belongs to another module")
        unsupported = self.grant.capabilities.difference(MEMORY_V2_CAPABILITIES)
        if unsupported:
            raise ValueError("grant contains unsupported Memory V2 capabilities")
        if self.grant.allows(MEMORY_EXECUTION_COMPLETE) and not self.grant.authority:
            raise ValueError("completion capability requires an authority")

    @property
    def session_id(self) -> str:
        return self.identity.execution_id

    @property
    def attempt_id(self) -> str:
        return self.identity.attempt_id

    @property
    def run_id(self) -> str:
        return self.identity.run_id

    @property
    def root_run_id(self) -> str:
        return self.identity.root_run_id

    @property
    def parent_run_id(self) -> str | None:
        return self.identity.parent_run_id


class MemoryCompletionFactory(Protocol):
    def build(
        self,
        *,
        result: KernelRunResult,
    ) -> RootRunCompletion | None:
        ...


@dataclass(frozen=True)
class MemoryAttachment:
    binding: MemoryToolkitRunBinding
    capabilities: NormalMemoryToolkitCapabilities
    completion_factory: MemoryCompletionFactory | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.binding, MemoryToolkitRunBinding):
            raise TypeError("binding must be a MemoryToolkitRunBinding")
        if not isinstance(self.capabilities, NormalMemoryToolkitCapabilities):
            raise TypeError("capabilities must be NormalMemoryToolkitCapabilities")
        if self.completion_factory is not None and not callable(
            getattr(self.completion_factory, "build", None)
        ):
            raise TypeError("completion_factory must provide build(result=...)")


class MemoryAttachmentFactory(Protocol):
    binding_id: str

    def attach(
        self,
        request: MemoryAttachmentRequest,
    ) -> MemoryAttachment:
        ...


@dataclass(frozen=True)
class MemoryV2Module(BaseAgentModule):
    host: MemoryAgentHostAdapter | None = None
    attachment_factory: MemoryAttachmentFactory | None = None
    name: str = "memory_v2"

    @property
    def propagation_key(self) -> str:
        return MEMORY_V2_MODULE_KEY

    @property
    def requires_runtime_context(self) -> bool:
        return bool(self.host is not None and getattr(self.host, "enabled", False))

    def derive_for_child(self, request, configured_child):
        del request
        if configured_child is None:
            return self
        if configured_child is not self:
            raise ValueError(
                "template subagents must use the exact parent Memory V2 module"
            )
        return configured_child

    def configure(self, builder) -> None:
        host = self.host
        if host is None:
            return
        if not isinstance(host, MemoryAgentHostAdapter):
            raise TypeError("host must be a MemoryAgentHostAdapter")
        if not host.enabled:
            return
        if str(getattr(builder.call_context, "mode", "") or "") == "submit_interaction":
            return

        factory = self.attachment_factory
        if factory is None or not callable(getattr(factory, "attach", None)):
            raise MemoryV2ModuleError(
                "enabled Memory V2 module requires an attachment factory"
            )
        if str(getattr(factory, "binding_id", "")) != host.binding_id:
            raise MemoryV2ModuleError(
                "attachment factory belongs to another host binding"
            )

        runtime_context = getattr(builder.call_context, "runtime_context", None)
        if not isinstance(runtime_context, AgentRuntimeContext):
            raise MemoryV2ModuleError(
                "enabled Memory V2 attachment requires an AgentRuntimeContext"
            )
        grant = runtime_context.grant_for(MEMORY_V2_MODULE_KEY)
        if grant is None or not grant.capabilities:
            return

        request = MemoryAttachmentRequest(
            agent_name=str(builder.spec.name or "").strip(),
            mode=str(builder.call_context.mode or "").strip(),
            identity=runtime_context.identity,
            grant=grant,
        )
        attachment = factory.attach(request)
        if not isinstance(attachment, MemoryAttachment):
            raise MemoryV2ModuleError(
                "attachment factory returned an invalid attachment"
            )
        binding = attachment.binding
        if binding.binding_id != host.binding_id:
            raise MemoryV2ModuleError("attachment belongs to another host binding")
        for field_name in ("session_id", "attempt_id", "run_id"):
            requested = getattr(request, field_name)
            if requested and getattr(binding, field_name) != requested:
                raise MemoryV2ModuleError(
                    f"attachment {field_name} does not match the agent call"
                )

        toolkit = host.build_normal_toolkit(
            binding,
            attachment.capabilities,
        )
        if toolkit is None:
            raise MemoryV2ModuleError(
                "enabled Memory V2 host returned no normal toolkit"
            )
        allowed_tool_names = set()
        for capability, tool_names in _TOOLS_BY_CAPABILITY.items():
            if grant.allows(capability):
                allowed_tool_names.update(tool_names)
        toolkit.tools = {
            name: tool
            for name, tool in toolkit.tools.items()
            if name in allowed_tool_names
        }
        setattr(
            toolkit,
            "_unchain_memory_v2_tool_names",
            tuple(toolkit.tools),
        )
        callables = getattr(toolkit, "_unchain_memory_v2_callables", {})
        if isinstance(callables, dict):
            setattr(
                toolkit,
                "_unchain_memory_v2_callables",
                {
                    name: function
                    for name, function in callables.items()
                    if name in allowed_tool_names
                },
            )

        completion_factory = attachment.completion_factory
        completion_granted = grant.allows(MEMORY_EXECUTION_COMPLETE)
        if completion_factory is not None and not completion_granted:
            raise MemoryV2ModuleError(
                "attachment returned completion authority without a grant"
            )

        def enqueue_terminal(result: KernelRunResult) -> None:
            if completion_factory is None:
                return None
            completion = completion_factory.build(result=result)
            if completion is None:
                return None
            if not isinstance(completion, RootRunCompletion):
                raise MemoryV2ModuleError(
                    "completion factory returned an invalid root completion"
                )
            if (
                completion.session_id != binding.session_id
                or completion.attempt_id != binding.attempt_id
                or completion.run_id != binding.run_id
            ):
                raise MemoryV2ModuleError(
                    "root completion changed the attached run identity"
                )
            receipt = host.enqueue_root_completion(completion)
            recorder = getattr(
                completion_factory,
                "record_enqueue_result",
                None,
            )
            if callable(recorder):
                recorder(completion=completion, receipt=receipt)
            return None

        if toolkit.tools:
            builder.add_tool(toolkit)
        if (
            completion_granted
            and grant.allows(MEMORY_CANDIDATE_PROPOSE)
        ):
            builder.add_harness(MemoryLearningPromptHarness())
        if completion_granted and completion_factory is not None:
            builder.add_run_hook(enqueue_terminal)


__all__ = [
    "MemoryAttachment",
    "MemoryAttachmentFactory",
    "MemoryAttachmentRequest",
    "MemoryCompletionFactory",
    "MemoryV2Module",
    "MemoryV2ModuleError",
    "MEMORY_CANDIDATE_PROPOSE",
    "MEMORY_CONTEXT_READ",
    "MEMORY_EXECUTION_COMPLETE",
    "MEMORY_V2_CAPABILITIES",
    "MEMORY_V2_MODULE_KEY",
    "MEMORY_WORKSPACE_READ",
]
