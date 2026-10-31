"""Pin the published Workflow dependencies attached to Agent Hook Pools.

This module deliberately owns admission only. Invocation, approvals, and
durable recovery are implemented at the Context V2 tool authority boundary;
accepting a reference here must never make a Hook execute on the legacy tool
path by accident.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from .agent_preparation import PreparedAgentPlan, checked_execution_plan
from .errors import SkeletonValidationError
from .hooks_config import attached_hooks
from .preparation import PreparedWorkflowPlan, prepare_workflow
from .workflow_runtime import PublishedWorkflow


def _fail(code: str, path: tuple[str | int, ...], message: str) -> None:
    raise SkeletonValidationError(code, path, message)


@dataclass(frozen=True)
class EmbeddedHookWorkflow:
    """A definition-derived run identity, never a published Store revision."""

    workflow_id: str
    revision_id: str
    document: dict[str, Any]


@dataclass(frozen=True)
class PinnedHookWorkflow:
    """One closed Hook entry plus its exact executable Workflow revision."""

    hook: dict[str, Any]
    workflow: PublishedWorkflow | EmbeddedHookWorkflow
    plan: PreparedWorkflowPlan


class SkeletonHookRuntime:
    """Resolve every attached Hook before a Kernel node may issue a model call."""

    def __init__(self, plan: PreparedAgentPlan,
                 resolver: Callable[[str, str], PublishedWorkflow]) -> None:
        if type(plan) is not PreparedAgentPlan:
            _fail("invalid_type", ("plan",), "Hook runtime requires an Agent plan")
        if not callable(resolver):
            _fail("invalid_type", ("resolver",), "Hook Workflow resolver is required")
        self.plan = checked_execution_plan(plan)
        self._by_kernel: dict[str, tuple[PinnedHookWorkflow, ...]] = {}
        snapshot = self.plan.snapshot()
        positions = {node["id"]: index for index, node in enumerate(snapshot["nodes"])}
        for kernel_id, hooks in attached_hooks(snapshot).items():
            bindings: list[PinnedHookWorkflow] = []
            hook_ids: set[str] = set()
            for hook_index, hook in enumerate(hooks):
                hook_path = (
                    "document", "nodes", positions[kernel_id], "hooks", hook_index,
                )
                hook_id = hook["id"]
                if hook_id in hook_ids:
                    _fail(
                        "duplicate_name", hook_path + ("id",),
                        "Hook id is duplicated in the attached Kernel scope",
                    )
                hook_ids.add(hook_id)
                path = hook_path + ("workflow",)
                reference = hook["workflow"]
                embedded = reference.get("kind") == "embedded"
                if embedded:
                    from .agent_preparation import execution_definition_digest
                    embedded_plan = prepare_workflow(reference["document"])
                    workflow_id = embedded_plan.document_id
                    if workflow_id == self.plan.document_id:
                        _fail("recursive_reference", path, "embedded Hook repeats its parent definition id")
                    revision_id = "embedded-sha256-" + execution_definition_digest(embedded_plan)
                    published = EmbeddedHookWorkflow(workflow_id, revision_id, embedded_plan.snapshot())
                else:
                    workflow_id = reference["workflow_id"]
                    revision_id = reference["revision_id"]
                    try:
                        published = resolver(workflow_id, revision_id)
                    except SkeletonValidationError:
                        raise
                    except Exception as exc:
                        raise SkeletonValidationError(
                            "resource_unavailable", path,
                            "published Hook Workflow revision is unavailable",
                        ) from exc
                    if type(published) is not PublishedWorkflow:
                        _fail("invalid_type", path, "Hook resolver returned an invalid Workflow revision")
                    if published.workflow_id != workflow_id or published.revision_id != revision_id:
                        _fail("identity_mismatch", path, "resolved Hook Workflow changed identity")
                if type(published.document) is not dict or published.document.get("kind") != "workflow":
                    _fail("wrong_kind", path, "Hook resolved a non-Workflow resource")
                try:
                    workflow_plan = prepare_workflow(published.document)
                    from .execution import _validate_all
                    _validate_all(workflow_plan.snapshot())
                except SkeletonValidationError as exc:
                    raise SkeletonValidationError(exc.code, path + exc.path[1:], str(exc)) from exc
                if workflow_plan.document_id != workflow_id:
                    _fail("identity_mismatch", path + ("workflow_id",), "Hook definition id differs from resource id")
                bindings.append(PinnedHookWorkflow(
                    hook=deepcopy(hook),
                    workflow=type(published)(workflow_id, revision_id, deepcopy(published.document)),
                    plan=workflow_plan,
                ))
            self._by_kernel[kernel_id] = tuple(bindings)

    def bindings_for_node(self, node_id: str) -> tuple[PinnedHookWorkflow, ...]:
        """Return detached, stable-order Hook dependencies for one Kernel node."""
        return tuple(PinnedHookWorkflow(
            hook=deepcopy(binding.hook),
            workflow=type(binding.workflow)(
                binding.workflow.workflow_id,
                binding.workflow.revision_id,
                deepcopy(binding.workflow.document),
            ),
            plan=binding.plan,
        ) for binding in self._by_kernel.get(node_id, ()))


__all__ = ["PinnedHookWorkflow", "SkeletonHookRuntime"]
