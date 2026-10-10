"""Bridge fixed Skeleton Agent snapshots into the existing subagent module."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable

from unchain.agent.modules.subagents import SubagentModule
from unchain.subagents.types import SubagentPolicy, SubagentTemplate

from .agent_preparation import PreparedAgentPlan, prepare_agent_graph


@dataclass(frozen=True)
class SkeletonSubagentAgent:
    """Agent-compatible facade whose execution remains a full Skeleton graph."""

    name: str
    plan: PreparedAgentPlan
    run_child: Callable[..., Any]
    entry_id: str
    allowed_tools: tuple[str, ...] | None = None
    disabled_tools: tuple[str, ...] = ()
    instructions: str = ""
    modules: tuple[Any, ...] = ()
    child_manifests: tuple[dict[str, Any], ...] = ()
    invocation: dict[str, Any] | None = None
    manages_subagent_completion: bool = True
    shares_parent_execution_session: bool = False

    @property
    def spec(self):
        return SimpleNamespace(
            modules=self.modules,
            allowed_tools=self.allowed_tools,
            missing_tool_policy="warn_skip",
        )

    def clone(self, *, name=None, instructions=None, modules=None, model=None,
              allowed_tools=None, missing_tool_policy=None):
        del model, missing_tool_policy
        return replace(
            self,
            name=self.name if name is None else name,
            instructions=self.instructions if instructions is None else instructions,
            modules=self.modules if modules is None else tuple(modules),
            allowed_tools=self.allowed_tools if allowed_tools is None else tuple(allowed_tools),
        )

    def fork_for_subagent(self, *, subagent_name, mode, parent_name, lineage,
                          task, instructions, expected_output,
                          disabled_module_keys=(), model=None,
                          allowed_tools=None, missing_tool_policy=None):
        del disabled_module_keys, model, missing_tool_policy
        effective_allowed = self.allowed_tools if allowed_tools is None else tuple(allowed_tools)
        return replace(
            self,
            name=subagent_name,
            allowed_tools=effective_allowed,
            invocation={
                "mode": mode,
                "parent_name": parent_name,
                "lineage": list(lineage),
                "task": task,
                "instructions": instructions,
                "expected_output": expected_output,
            },
        )

    def run(self, messages, **kwargs):
        return self.run_child(self, deepcopy(messages), dict(kwargs))


class SkeletonSubagentRuntime:
    """Build node-scoped SubagentModule instances from a fixed run manifest."""

    def __init__(self, manifests, *, run_child, policy=None):
        if not callable(run_child):
            raise TypeError("Skeleton subagent runtime requires a child runner")
        self.run_child = run_child
        self.policy = policy or SubagentPolicy(
            max_depth=6,
            max_children_per_parent=10,
            max_total_subagents=50,
            max_parallel_workers=4,
            worker_timeout_seconds=60.0,
            allow_dynamic_workers=False,
            allow_dynamic_delegate=False,
            handoff_requires_template=True,
        )
        self.by_kernel: dict[str, tuple[SubagentTemplate, ...]] = {}
        self.by_attachment: dict[tuple[str, str], SkeletonSubagentAgent] = {}
        grouped: dict[str, list[SubagentTemplate]] = {}
        for raw in manifests:
            if type(raw) is not dict:
                raise TypeError("Subagent manifest entry must be an object")
            document = deepcopy(raw["document"])
            plan = prepare_agent_graph(document)
            allowed = raw.get("allowed_tools")
            disabled = tuple(raw.get("disabled_tools") or ())
            effective = None if allowed is None else tuple(
                tool for tool in allowed if tool not in set(disabled)
            )
            adapter = SkeletonSubagentAgent(
                name=raw["name"],
                plan=plan,
                run_child=run_child,
                entry_id=raw["entry_id"],
                allowed_tools=effective,
                disabled_tools=disabled,
                child_manifests=tuple(deepcopy(raw.get("subagents") or ())),
            )
            template = SubagentTemplate(
                name=raw["name"],
                description=raw["description"],
                agent=adapter,
                allowed_modes=tuple(raw["allowed_modes"]),
                output_mode=raw["output_mode"],
                memory_policy=raw["memory_policy"],
                parallel_safe=raw["parallel_safe"],
                allowed_tools=effective,
                model=None,
            )
            key = (raw["kernel_node_id"], raw["entry_id"])
            if key in self.by_attachment:
                raise ValueError("attached Subagent Pools contain duplicate entry ids")
            self.by_attachment[key] = adapter
            grouped.setdefault(raw["kernel_node_id"], []).append(template)
        for kernel_id, templates in grouped.items():
            names = [template.name for template in templates]
            if len(names) != len(set(names)):
                raise ValueError("attached Subagent Pools contain duplicate names")
            self.by_kernel[kernel_id] = tuple(templates)

    def modules_for_node(self, context):
        templates = self.by_kernel.get(context.node_id, ())
        return (SubagentModule(templates=templates, policy=self.policy),) if templates else ()

    @staticmethod
    def tools_for_child(adapter, resolver):
        """Apply the fixed pool entry tool policy at the child Kernel boundary."""
        allowed = adapter.allowed_tools
        disabled = frozenset(adapter.disabled_tools)

        def filtered(context):
            tools = tuple(resolver(context))
            if allowed is not None:
                allowed_names = frozenset(allowed)
                tools = tuple(tool for tool in tools if getattr(tool, "name", None) in allowed_names)
            return tuple(tool for tool in tools if getattr(tool, "name", None) not in disabled)

        return filtered

    def agent_for_attachment(self, kernel_node_id, entry_id):
        adapter = self.by_attachment.get((kernel_node_id, entry_id))
        if adapter is None:
            raise ValueError("subagent attachment is absent from the fixed manifest")
        return adapter


__all__ = ["SkeletonSubagentAgent", "SkeletonSubagentRuntime"]
