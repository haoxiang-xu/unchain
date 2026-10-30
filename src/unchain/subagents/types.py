from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal


SubagentMode = Literal["delegate", "handoff", "worker"]
SubagentOutputMode = Literal["summary", "last_message", "full_trace"]
SubagentMemoryPolicy = Literal["ephemeral", "scoped_persistent"]


class SubagentWaitSignal(RuntimeError):
    """Propagate a durable child-graph wait without turning it into tool output."""

    def __init__(
        self,
        wait_id: str,
        request: dict[str, Any],
        *,
        parent_tool_call: dict[str, str] | None = None,
    ):
        if type(wait_id) is not str or not wait_id:
            raise ValueError("subagent wait id must be non-empty")
        if type(request) is not dict:
            raise TypeError("subagent wait request must be an object")
        self.wait_id = wait_id
        self.request = copy.deepcopy(request)
        if parent_tool_call is not None:
            if (
                type(parent_tool_call) is not dict
                or set(parent_tool_call) != {"call_id", "tool_name"}
                or any(
                    type(value) is not str or not value.strip()
                    for value in parent_tool_call.values()
                )
            ):
                raise ValueError("subagent parent tool call is invalid")
            self.parent_tool_call = {
                "call_id": parent_tool_call["call_id"].strip(),
                "tool_name": parent_tool_call["tool_name"].strip(),
            }
        else:
            self.parent_tool_call = None
        super().__init__("subagent graph is awaiting interaction")

    def for_parent_tool_call(self, tool_call) -> "SubagentWaitSignal":
        """Bind a propagated wait to the immutable parent tool invocation."""

        call_id = getattr(tool_call, "call_id", None)
        tool_name = getattr(tool_call, "name", None)
        if type(call_id) is not str or not call_id.strip() or type(tool_name) is not str or not tool_name.strip():
            raise ValueError("subagent parent tool call is invalid")
        parent_tool_call = {"call_id": call_id.strip(), "tool_name": tool_name.strip()}
        if self.parent_tool_call is not None and self.parent_tool_call != parent_tool_call:
            raise ValueError("subagent wait parent tool call changed")
        return SubagentWaitSignal(
            self.wait_id,
            self.request,
            parent_tool_call=parent_tool_call,
        )


@dataclass(frozen=True)
class SubagentTemplate:
    name: str
    description: str
    agent: Any | None = None
    allowed_modes: tuple[SubagentMode, ...] = ("delegate", "handoff", "worker")
    output_mode: SubagentOutputMode = "summary"
    memory_policy: SubagentMemoryPolicy = "ephemeral"
    parallel_safe: bool = False
    allowed_tools: tuple[str, ...] | None = None
    model: str | None = None
    module_capabilities: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        normalized: list[tuple[str, tuple[str, ...]]] = []
        seen: set[str] = set()
        for item in self.module_capabilities:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "module_capabilities must contain (module_key, capabilities) pairs"
                )
            module_key, capabilities = item
            if not isinstance(module_key, str) or not module_key.strip():
                raise ValueError("module capability key must be non-empty")
            module_key = module_key.strip()
            if module_key in seen:
                raise ValueError("module capability keys must be unique")
            seen.add(module_key)
            if isinstance(capabilities, (str, bytes, bytearray)):
                raise TypeError("module capabilities must be a string collection")
            values = tuple(capabilities)
            if any(
                not isinstance(value, str) or not value.strip()
                for value in values
            ):
                raise ValueError("module capabilities must be non-empty strings")
            normalized.append(
                (module_key, tuple(dict.fromkeys(value.strip() for value in values)))
            )
        object.__setattr__(self, "module_capabilities", tuple(normalized))

    def supports_mode(self, mode: SubagentMode) -> bool:
        return mode in self.allowed_modes

    def requested_module_capabilities(self) -> dict[str, frozenset[str]]:
        return {
            module_key: frozenset(capabilities)
            for module_key, capabilities in self.module_capabilities
        }


@dataclass(frozen=True)
class SubagentPolicy:
    max_depth: int = 6
    max_children_per_parent: int = 10
    max_total_subagents: int = 100
    max_parallel_workers: int = 4
    worker_timeout_seconds: float = 30.0
    allow_dynamic_workers: bool = False
    allow_dynamic_delegate: bool = False
    handoff_requires_template: bool = True
    max_open_threads: int = 10
    max_mailbox_messages: int = 100
    max_message_chars: int = 8000
    max_board_items: int = 200
    max_board_item_chars: int = 12000
    allow_child_to_child_messages: bool = False
    allow_broadcast_messages: bool = False
    allow_return_handoff: bool = True
    retain_completed_threads: bool = True


@dataclass
class SubagentState:
    root_agent_id: str = ""
    active_agent_id: str = ""
    active_lineage: list[str] = field(default_factory=list)
    handoff_stack: list[dict[str, Any]] = field(default_factory=list)
    lineage_counters: dict[str, int] = field(default_factory=dict)
    running_batches: dict[str, Any] = field(default_factory=dict)
    threads: dict[str, dict[str, Any]] = field(default_factory=dict)
    mailboxes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    blackboards: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    return_handoff_stack: list[dict[str, Any]] = field(default_factory=list)
    blocked_clarifications: list[dict[str, Any]] = field(default_factory=list)
    run_bundles: dict[str, dict[str, Any]] = field(default_factory=dict)
    spawn_stats: dict[str, int] = field(
        default_factory=lambda: {"delegate": 0, "handoff": 0, "worker": 0}
    )

    def copy(self) -> "SubagentState":
        return SubagentState(
            root_agent_id=self.root_agent_id,
            active_agent_id=self.active_agent_id,
            active_lineage=list(self.active_lineage),
            handoff_stack=copy.deepcopy(self.handoff_stack),
            lineage_counters=dict(self.lineage_counters),
            running_batches=copy.deepcopy(self.running_batches),
            threads=copy.deepcopy(self.threads),
            mailboxes=copy.deepcopy(self.mailboxes),
            blackboards=copy.deepcopy(self.blackboards),
            return_handoff_stack=copy.deepcopy(self.return_handoff_stack),
            blocked_clarifications=copy.deepcopy(self.blocked_clarifications),
            run_bundles=copy.deepcopy(self.run_bundles),
            spawn_stats=dict(self.spawn_stats),
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "root_agent_id": self.root_agent_id,
            "active_agent_id": self.active_agent_id,
            "active_lineage": list(self.active_lineage),
            "handoff_stack": copy.deepcopy(self.handoff_stack),
            "lineage_counters": dict(self.lineage_counters),
            "running_batches": copy.deepcopy(self.running_batches),
            "threads": copy.deepcopy(self.threads),
            "mailboxes": copy.deepcopy(self.mailboxes),
            "blackboards": copy.deepcopy(self.blackboards),
            "return_handoff_stack": copy.deepcopy(self.return_handoff_stack),
            "blocked_clarifications": copy.deepcopy(self.blocked_clarifications),
            "spawn_stats": dict(self.spawn_stats),
        }
        # Preserve the legacy empty-state wire while still carrying canonical
        # child bundles whenever accounting evidence actually exists.
        if self.run_bundles:
            value["run_bundles"] = copy.deepcopy(self.run_bundles)
        return value

    @classmethod
    def from_raw(cls, raw: Any) -> "SubagentState":
        if isinstance(raw, SubagentState):
            return raw.copy()
        if not isinstance(raw, dict):
            return cls()
        state = cls()
        state.root_agent_id = str(raw.get("root_agent_id") or "")
        state.active_agent_id = str(raw.get("active_agent_id") or "")
        active_lineage = raw.get("active_lineage")
        if isinstance(active_lineage, list):
            state.active_lineage = [str(item) for item in active_lineage if isinstance(item, str)]
        handoff_stack = raw.get("handoff_stack")
        if isinstance(handoff_stack, list):
            state.handoff_stack = [copy.deepcopy(item) for item in handoff_stack if isinstance(item, dict)]
        lineage_counters = raw.get("lineage_counters")
        if isinstance(lineage_counters, dict):
            state.lineage_counters = {
                str(key): int(value)
                for key, value in lineage_counters.items()
                if isinstance(key, str) and isinstance(value, int)
            }
        running_batches = raw.get("running_batches")
        if isinstance(running_batches, dict):
            state.running_batches = copy.deepcopy(running_batches)
        blocked = raw.get("blocked_clarifications")
        if isinstance(blocked, list):
            state.blocked_clarifications = [copy.deepcopy(item) for item in blocked if isinstance(item, dict)]
        raw_run_bundles = raw.get("run_bundles")
        if isinstance(raw_run_bundles, dict):
            from ..run_bundle_v2 import run_bundle_from_dict

            for key, value in raw_run_bundles.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                bundle = run_bundle_from_dict(value)
                if key != bundle.bundle_id:
                    raise ValueError("subagent run bundle key does not match bundle_id")
                state.run_bundles[key] = bundle.to_dict()
        threads = raw.get("threads")
        if isinstance(threads, dict):
            state.threads = {
                str(key): copy.deepcopy(value)
                for key, value in threads.items()
                if isinstance(key, str) and isinstance(value, dict)
            }
        mailboxes = raw.get("mailboxes")
        if isinstance(mailboxes, dict):
            state.mailboxes = {
                str(key): [copy.deepcopy(item) for item in value if isinstance(item, dict)]
                for key, value in mailboxes.items()
                if isinstance(key, str) and isinstance(value, list)
            }
        blackboards = raw.get("blackboards")
        if isinstance(blackboards, dict):
            state.blackboards = {
                str(key): [copy.deepcopy(item) for item in value if isinstance(item, dict)]
                for key, value in blackboards.items()
                if isinstance(key, str) and isinstance(value, list)
            }
        return_handoff_stack = raw.get("return_handoff_stack")
        if isinstance(return_handoff_stack, list):
            state.return_handoff_stack = [
                copy.deepcopy(item)
                for item in return_handoff_stack
                if isinstance(item, dict)
            ]
        spawn_stats = raw.get("spawn_stats")
        if isinstance(spawn_stats, dict):
            for mode in ("delegate", "handoff", "worker"):
                if isinstance(spawn_stats.get(mode), int):
                    state.spawn_stats[mode] = int(spawn_stats[mode])
        return state

    def merged(self, raw: Any) -> "SubagentState":
        default_spawn_stats = SubagentState().spawn_stats
        raw_spawn_stats = (
            raw.spawn_stats
            if isinstance(raw, SubagentState) and raw.spawn_stats != default_spawn_stats
            else None
        )
        if isinstance(raw, dict):
            raw_spawn_stats = raw.get("spawn_stats") if "spawn_stats" in raw else None
        update = SubagentState.from_raw(raw)
        current = self.copy()
        if update.root_agent_id:
            current.root_agent_id = update.root_agent_id
        if update.active_agent_id:
            current.active_agent_id = update.active_agent_id
        if update.active_lineage:
            current.active_lineage = list(update.active_lineage)
        if update.handoff_stack:
            current.handoff_stack = copy.deepcopy(update.handoff_stack)
        if update.lineage_counters:
            current.lineage_counters.update(update.lineage_counters)
        if update.running_batches:
            current.running_batches.update(copy.deepcopy(update.running_batches))
        if update.blocked_clarifications:
            current.blocked_clarifications.extend(copy.deepcopy(update.blocked_clarifications))
        if update.run_bundles:
            from ..run_bundle import RunBundleProtocolError
            from ..run_bundle_v2 import run_bundle_from_dict

            for bundle_id, bundle in update.run_bundles.items():
                prior = current.run_bundles.get(bundle_id)
                if prior is not None:
                    prior_bundle = run_bundle_from_dict(prior)
                    incoming_bundle = run_bundle_from_dict(bundle)
                    if prior_bundle.identity != incoming_bundle.identity:
                        raise RunBundleProtocolError(
                            "one child bundle_id changed its immutable identity"
                        )
                    if incoming_bundle.revision < prior_bundle.revision:
                        continue
                    if (
                        incoming_bundle.revision == prior_bundle.revision
                        and incoming_bundle.bundle_digest
                        != prior_bundle.bundle_digest
                    ):
                        raise RunBundleProtocolError(
                            "one child bundle revision has conflicting projections"
                        )
                current.run_bundles[bundle_id] = copy.deepcopy(bundle)
        if update.threads:
            current.threads.update(copy.deepcopy(update.threads))
        if update.mailboxes:
            for key, value in update.mailboxes.items():
                current.mailboxes.setdefault(key, []).extend(copy.deepcopy(value))
        if update.blackboards:
            for key, value in update.blackboards.items():
                current.blackboards.setdefault(key, []).extend(copy.deepcopy(value))
        if update.return_handoff_stack:
            current.return_handoff_stack.extend(copy.deepcopy(update.return_handoff_stack))
        if isinstance(raw_spawn_stats, dict):
            for key, value in raw_spawn_stats.items():
                if key not in {"delegate", "handoff", "worker"} or not isinstance(value, int):
                    continue
                current.spawn_stats[key] = int(value)
        return current


@dataclass(frozen=True)
class SubagentResult:
    mode: str
    agent_name: str
    template_name: str | None
    status: str
    output: str = ""
    summary: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    lineage: list[str] = field(default_factory=list)
    clarification_request: dict[str, Any] | None = None
    error: str = ""
    subagent_state: dict[str, Any] = field(default_factory=dict)
    run_bundle: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.run_bundle is None:
            return
        if not isinstance(self.run_bundle, dict):
            raise TypeError("run_bundle must be a run bundle object or null")
        from ..run_bundle_v2 import run_bundle_from_dict

        object.__setattr__(
            self,
            "run_bundle",
            run_bundle_from_dict(self.run_bundle).to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        model_visible_subagent_state = copy.deepcopy(self.subagent_state)
        if isinstance(model_visible_subagent_state, dict):
            # Canonical RunBundles are accounting evidence, not model context.
            # They remain available through to_record_dict() and the parent
            # SubagentState merge, but must never bloat or influence a tool
            # result sent back to the model.
            model_visible_subagent_state.pop("run_bundles", None)
        return {
            "mode": self.mode,
            "agent_name": self.agent_name,
            "template_name": self.template_name,
            "status": self.status,
            "output": self.output,
            "summary": self.summary,
            "messages": copy.deepcopy(self.messages),
            "lineage": list(self.lineage),
            "clarification_request": copy.deepcopy(self.clarification_request),
            "error": self.error,
            "subagent_state": model_visible_subagent_state,
        }

    def to_record_dict(self) -> dict[str, Any]:
        """Serialize durable internal state without changing model-visible output."""

        value = self.to_dict()
        value["subagent_state"] = copy.deepcopy(self.subagent_state)
        if self.run_bundle is not None:
            value["run_bundle"] = copy.deepcopy(self.run_bundle)
        return value

    @classmethod
    def from_record_dict(cls, value: dict[str, Any]) -> "SubagentResult":
        if not isinstance(value, dict):
            raise TypeError("subagent result record must be an object")
        result = cls(**copy.deepcopy(value))
        if result.to_record_dict() != value:
            raise ValueError("subagent result record is not canonical")
        return result
