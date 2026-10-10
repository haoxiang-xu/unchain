"""Official durable Workflow Call handler with fixed revision resolution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable

from unchain.journal import AttemptRef, JournalConflictError

from .agent_preparation import execution_definition_digest
from .durable_execution import (
    DurableWorkflowRunner,
    WorkflowNodeCompleted,
    WorkflowNodeWaiting,
)
from .durable_store import WorkflowExecutionStore
from .errors import SkeletonValidationError
from .preparation import PreparedWorkflowPlan, prepare_workflow


_WAIT_SCHEMA = "unchain.skeleton_workflow_call_wait.v1"
_TERMINAL_FAILURES = {"cancelled", "failed", "outcome_unknown"}


@dataclass(frozen=True)
class PublishedWorkflow:
    workflow_id: str
    revision_id: str
    document: dict[str, Any]


def _fail(code: str, path: tuple[str | int, ...], message: str) -> None:
    raise SkeletonValidationError(code, path, message)


def _text(value: Any, path: tuple[str | int, ...], label: str) -> str:
    if type(value) is not str:
        _fail("invalid_type", path, f"{label} must be a string")
    if not value:
        _fail("invalid_value", path, f"{label} must be non-empty")
    return value


class SkeletonWorkflowCallRuntime:
    """Run one published Workflow revision as a durable child graph.

    The resolver is host-owned storage policy. It must return the exact immutable
    revision requested; this adapter never asks for or follows a latest version.
    """

    def __init__(self, store: WorkflowExecutionStore,
                 resolver: Callable[[str, str], PublishedWorkflow], *,
                 handlers: dict[str, Callable] | None = None, max_depth: int = 32):
        if type(store) is not WorkflowExecutionStore:
            _fail("invalid_type", ("store",), "WorkflowExecutionStore required")
        if not callable(resolver):
            _fail("invalid_type", ("resolver",), "Workflow resolver required")
        if handlers is not None and type(handlers) is not dict:
            _fail("invalid_type", ("handlers",), "handlers must be an object")
        if type(max_depth) is not int or isinstance(max_depth, bool) or not 1 <= max_depth <= 32:
            _fail("invalid_value", ("max_depth",), "max depth must be between 1 and 32")
        self.store = store
        self.plan = store.plan
        self.resolver = resolver
        self.handlers = dict(handlers or {})
        self.handlers.pop("workflow_call", None)
        self.max_depth = max_depth
        snapshot = self.plan.snapshot()
        self._nodes = {node["id"]: node for node in snapshot["nodes"]}
        self._positions = {node["id"]: index for index, node in enumerate(snapshot["nodes"])}
        self._resolved: dict[str, tuple[PublishedWorkflow, PreparedWorkflowPlan]] = {}
        # A runner instance is one fixed dependency snapshot. Resolve every
        # direct child before create/dispatch so a later publish or edit cannot
        # change this run between Start and Workflow Call.
        for node in snapshot["nodes"]:
            if node["type"] == "workflow_call":
                self._resolve(node["id"])

    def _path(self, node_id: str) -> tuple[str | int, ...]:
        return ("document", "nodes", self._positions[node_id], "config", "workflow")

    def _reference(self, node_id: str) -> tuple[str, str]:
        node = self._nodes.get(node_id)
        if node is None or node.get("type") != "workflow_call":
            _fail("identity_mismatch", ("node_id",), "node is not a Workflow Call")
        config = node.get("config")
        workflow = config.get("workflow") if type(config) is dict else None
        if type(workflow) is not dict or set(workflow) != {"workflow_id", "revision_id"}:
            _fail("invalid_shape", self._path(node_id), "Workflow Call reference is invalid")
        return (
            _text(workflow["workflow_id"], self._path(node_id) + ("workflow_id",), "workflow id"),
            _text(workflow["revision_id"], self._path(node_id) + ("revision_id",), "revision id"),
        )

    def _resolve(self, node_id: str) -> tuple[PublishedWorkflow, PreparedWorkflowPlan]:
        cached = self._resolved.get(node_id)
        if cached is not None:
            return cached
        workflow_id, revision_id = self._reference(node_id)
        try:
            published = self.resolver(workflow_id, revision_id)
        except SkeletonValidationError:
            raise
        except Exception as exc:
            raise SkeletonValidationError(
                "resource_unavailable", self._path(node_id),
                "published Workflow revision is unavailable",
            ) from exc
        if type(published) is not PublishedWorkflow:
            _fail("invalid_type", self._path(node_id), "resolver returned an invalid Workflow revision")
        if published.workflow_id != workflow_id or published.revision_id != revision_id:
            _fail("identity_mismatch", self._path(node_id), "resolved Workflow revision changed identity")
        if type(published.document) is not dict or published.document.get("kind") != "workflow":
            _fail("wrong_kind", self._path(node_id), "Workflow Call resolved a non-Workflow resource")
        plan = prepare_workflow(published.document)
        if plan.document_id != workflow_id:
            _fail("identity_mismatch", self._path(node_id) + ("workflow_id",), "definition id differs from resource id")
        if plan.document_id == self.plan.document_id:
            _fail("recursive_reference", self._path(node_id), "recursive Workflow Call is forbidden")
        from .execution import _validate_all
        _validate_all(plan.snapshot())
        result = (PublishedWorkflow(workflow_id, revision_id, deepcopy(published.document)), plan)
        self._resolved[node_id] = result
        return result

    def preflight_inputs(self, node_id: str, _inputs: dict[str, Any]) -> None:
        """Resolve before the parent records an external-effect intent."""
        published, plan = self._resolve(node_id)
        record = self.store.load()
        if record is None:
            _fail("missing_value", ("execution",), "parent execution is unavailable")
        frames = record.state["call_stack"]
        if len(frames) >= self.max_depth:
            _fail("limit_depth", ("execution", "call_stack"), "Workflow Call depth exceeded")
        definition_ids = {frame["definition_id"] for frame in frames}
        if published.workflow_id in definition_ids:
            _fail("recursive_reference", self._path(node_id), "recursive Workflow Call is forbidden")
        digest = execution_definition_digest(plan)
        if any(frame["definition_digest"] == digest for frame in frames):
            _fail("recursive_reference", self._path(node_id), "recursive Workflow definition is forbidden")

    def _identity(self, context, published: PublishedWorkflow, plan: PreparedWorkflowPlan):
        basis = "\x00".join((context.graph_id, context.node_invocation_id,
                              published.workflow_id, published.revision_id))
        suffix = sha256(basis.encode("utf-8", "surrogatepass")).hexdigest()
        graph_id = f"{context.graph_id}.child.{suffix}"
        attempt_id = f"workflow-call-{suffix}"
        digest = execution_definition_digest(plan)
        return graph_id, attempt_id, digest

    def _handlers_for_child(self, plan: PreparedWorkflowPlan) -> dict[str, Callable]:
        handlers: dict[str, Callable] = {}
        for kind, handler in self.handlers.items():
            clone = getattr(handler, "for_plan", None)
            handlers[kind] = clone(plan) if callable(clone) else handler
        return handlers

    def _child_runner(self, plan: PreparedWorkflowPlan, graph_id: str, attempt_id: str,
                      *, ancestor_assert_current: Callable[[], None] | None = None):
        attempt = AttemptRef(self.store.attempt.generation, attempt_id)
        child_store = WorkflowExecutionStore(
            self.store.journal, plan, graph_id=graph_id, attempt=attempt,
        )
        child_handlers = self._handlers_for_child(plan)
        child_runtime = SkeletonWorkflowCallRuntime(
            child_store, self.resolver, handlers=child_handlers, max_depth=self.max_depth,
        )
        child_handlers["workflow_call"] = child_runtime
        return DurableWorkflowRunner(
            child_store,
            child_handlers,
            ancestor_assert_current=ancestor_assert_current,
        ), child_runtime

    def _child_for_active(self, active: dict[str, Any],
                          *, ancestor_assert_current: Callable[[], None] | None = None):
        published, plan = self._resolve(active["node_id"])
        context = type("IdentityContext", (), {
            "graph_id": self.store.graph_id,
            "node_invocation_id": active["node_invocation_id"],
        })()
        graph_id, attempt_id, _digest = self._identity(context, published, plan)
        child, runtime = self._child_runner(
            plan,
            graph_id,
            attempt_id,
            ancestor_assert_current=ancestor_assert_current,
        )
        return child, runtime, published, plan

    def _child(self, context, inputs: dict[str, Any]):
        published, plan = self._resolve(context.node_id)
        graph_id, attempt_id, digest = self._identity(context, published, plan)
        parent = self.store.load()
        if (parent is None or parent.state["status"] != "running"
                or parent.state["active"] is None):
            _fail("missing_value", ("execution", "active"), "parent Workflow Call is unavailable")
        if parent.state["active"]["node_invocation_id"] != context.node_invocation_id:
            _fail("identity_mismatch", ("execution", "active"), "parent Workflow Call changed")
        frames = deepcopy(parent.state["call_stack"])
        frames[-1]["caller_node_invocation_id"] = context.node_invocation_id
        frames.append({
            "execution_id": self.store.attempt.generation.execution_id,
            "graph_invocation_id": graph_id,
            "attempt_id": attempt_id,
            "definition_id": plan.document_id,
            "definition_digest": digest,
        })
        child, child_runtime = self._child_runner(
            plan,
            graph_id,
            attempt_id,
            ancestor_assert_current=context.assert_current,
        )
        child.create(inputs, call_stack=frames)
        try:
            context.assert_current()
        except Exception:
            child.cancel()
            raise
        return child, child_runtime, published, digest

    def __call__(self, context, inputs: dict[str, Any]):
        child, _runtime, published, digest = self._child(context, inputs)
        result = child.run()
        if result.status == "completed":
            return WorkflowNodeCompleted(result.outputs)
        if result.status in _TERMINAL_FAILURES:
            _fail("child_" + result.status, ("execution", "child"), "child Workflow did not complete")
        request = {
            "schema": _WAIT_SCHEMA,
            "parent_graph_id": context.graph_id,
            "child_graph_id": result.graph_id,
            "child_attempt_id": child.store.attempt.attempt_id,
            "workflow_id": published.workflow_id,
            "revision_id": published.revision_id,
            "definition_digest": digest,
        }
        return WorkflowNodeWaiting("child", result.graph_id, request)

    def _active_child(self, parent: DurableWorkflowRunner, node_invocation_id: str):
        record = parent.store.load()
        if record is None or record.state["status"] != "waiting":
            _fail("identity_mismatch", ("execution",), "parent is not waiting")
        active = record.state["active"]
        wait = active["wait"]
        if (active["node_invocation_id"] != node_invocation_id or wait["kind"] != "child"):
            _fail("identity_mismatch", ("node_invocation_id",), "child wait is not current")
        request = parent.store.read_json(wait["request_ref"])
        if type(request) is not dict or set(request) != {
            "schema", "parent_graph_id", "child_graph_id", "child_attempt_id",
            "workflow_id", "revision_id", "definition_digest",
        } or request.get("schema") != _WAIT_SCHEMA:
            _fail("invalid_shape", ("execution", "active", "wait"), "child wait receipt is invalid")
        node_id = active["node_id"]
        published, plan = self._resolve(node_id)
        if (request["parent_graph_id"] != parent.store.graph_id
                or request["workflow_id"] != published.workflow_id
                or request["revision_id"] != published.revision_id
                or request["definition_digest"] != execution_definition_digest(plan)):
            _fail("identity_mismatch", ("execution", "active", "wait"), "child wait identity changed")
        def assert_parent_wait_current():
            current = parent.store.load()
            current_active = current.state["active"] if current is not None else None
            current_wait = current_active.get("wait") if current_active is not None else None
            if (current is None or current.state["status"] != "waiting"
                    or current.state["paused"] or current_active["node_invocation_id"] != node_invocation_id
                    or current_wait is None or current_wait["kind"] != "child"
                    or current_wait["id"] != wait["id"]):
                raise JournalConflictError("parent Workflow Call wait is no longer current")
            if parent._ancestor_assert_current is not None:
                parent._ancestor_assert_current()
        child, child_runtime = self._child_runner(
            plan, request["child_graph_id"], request["child_attempt_id"],
            ancestor_assert_current=assert_parent_wait_current,
        )
        return record, active, wait, child, child_runtime

    def recover_result(self, context):
        """Recover a deterministically identified child without replaying it."""
        active = self.store.load().state["active"]
        if active is None or active["node_invocation_id"] != context.node_invocation_id:
            _fail("identity_mismatch", ("execution", "active"), "parent Workflow Call changed")
        child, child_runtime, published, plan = self._child_for_active(
            active,
            ancestor_assert_current=context.assert_current,
        )
        if child.store.load() is None:
            return None
        result = child.inspect()
        if result.status == "running":
            result = child.recover()
        elif result.status == "waiting":
            child_record = child.store.load()
            child_active = child_record.state["active"]
            if child_active["wait"]["kind"] == "child":
                child_runtime.resume_child(
                    child, node_invocation_id=child_active["node_invocation_id"],
                )
                result = child.inspect()
        if result.status in {"ready", "result_received"}:
            result = child.run()
        if result.status == "completed":
            return WorkflowNodeCompleted(result.outputs)
        if result.status in _TERMINAL_FAILURES:
            _fail("child_" + result.status, ("execution", "child"), "child Workflow did not complete")
        graph_id, _attempt_id, digest = self._identity(context, published, plan)
        return WorkflowNodeWaiting("child", graph_id, {
            "schema": _WAIT_SCHEMA,
            "parent_graph_id": context.graph_id,
            "child_graph_id": graph_id,
            "child_attempt_id": child.store.attempt.attempt_id,
            "workflow_id": published.workflow_id,
            "revision_id": published.revision_id,
            "definition_digest": digest,
        })

    def resume_child(self, parent: DurableWorkflowRunner, *, node_invocation_id: str):
        record, active, wait, child, child_runtime = self._active_child(parent, node_invocation_id)
        result = child.inspect()
        if result.status == "running":
            result = child.recover()
        elif result.status == "waiting":
            child_record = child.store.load()
            if child_record.state["active"]["wait"]["kind"] == "child":
                child_runtime.resume_child(
                    child,
                    node_invocation_id=child_record.state["active"]["node_invocation_id"],
                )
                result = child.inspect()
        if result.status in {"ready", "result_received"}:
            result = child.run()
        if result.status == "completed":
            parent.complete_wait(wait["id"], result.outputs,
                node_invocation_id=active["node_invocation_id"])
            return parent.run()
        if result.status in _TERMINAL_FAILURES:
            error = SkeletonValidationError("child_" + result.status,
                ("execution", "child"), "child Workflow did not complete")
            return parent._result(parent._failed(record, error, unknown=result.status == "outcome_unknown"))
        return parent.inspect()

    def tool_interaction_wait(self, parent: DurableWorkflowRunner, *,
                              node_invocation_id: str) -> dict[str, Any]:
        """Return the exact deepest Tool approval under one Workflow Call wait."""
        _record, _active, _wait, child, child_runtime = self._active_child(
            parent, node_invocation_id,
        )
        child_record = child.store.load()
        child_active = child_record.state["active"] if child_record is not None else None
        child_wait = child_active.get("wait") if child_active is not None else None
        if (child_record is None or child_record.state["status"] != "waiting"
                or child_wait is None):
            _fail("identity_mismatch", ("execution", "child"),
                  "nested Workflow is not waiting")
        node_type = child.nodes[child_active["node_id"]]["type"]
        if child_wait["kind"] == "interaction" and node_type == "tool_call":
            return {
                "graph_id": child.store.graph_id,
                "attempt_id": child.store.attempt.attempt_id,
                "wait_id": child_wait["id"],
                "node_invocation_id": child_active["node_invocation_id"],
                "request": child.store.read_json(child_wait["request_ref"]),
            }
        if child_wait["kind"] == "child" and node_type == "workflow_call":
            return child_runtime.tool_interaction_wait(
                child, node_invocation_id=child_active["node_invocation_id"],
            )
        _fail("invalid_transition", ("execution", "child", "wait"),
              "nested Workflow wait is not a Tool approval")

    def resume_tool_interaction(self, parent: DurableWorkflowRunner, *,
                                node_invocation_id: str, wait_id: str,
                                response: Any, submitted_by: str = "user"):
        """Resume one deepest Tool approval, then settle each Workflow Call parent."""
        _record, _active, _wait, child, child_runtime = self._active_child(
            parent, node_invocation_id,
        )
        child_record = child.store.load()
        child_active = child_record.state["active"] if child_record is not None else None
        child_wait = child_active.get("wait") if child_active is not None else None
        if (child_record is None or child_record.state["status"] != "waiting"
                or child_wait is None):
            _fail("identity_mismatch", ("execution", "child"),
                  "nested Workflow is not waiting")
        node_type = child.nodes[child_active["node_id"]]["type"]
        if child_wait["kind"] == "interaction" and node_type == "tool_call":
            if child_wait["id"] != wait_id:
                _fail("identity_mismatch", ("wait_id",),
                      "nested Tool approval wait is not current")
            result = child.resume_tool_interaction(
                wait_id,
                response,
                node_invocation_id=child_active["node_invocation_id"],
                submitted_by=submitted_by,
            )
            if result.status in {"ready", "result_received"}:
                child.run()
        elif child_wait["kind"] == "child" and node_type == "workflow_call":
            child_runtime.resume_tool_interaction(
                child,
                node_invocation_id=child_active["node_invocation_id"],
                wait_id=wait_id,
                response=response,
                submitted_by=submitted_by,
            )
        else:
            _fail("invalid_transition", ("execution", "child", "wait"),
                  "nested Workflow wait is not a Tool approval")
        return self.resume_child(parent, node_invocation_id=node_invocation_id)

    def expire_tool_interaction(self, parent: DurableWorkflowRunner, *,
                                node_invocation_id: str, wait_id: str, now_ms: int):
        """Terminalize the exact deepest expired Tool wait and its waiting callers."""
        _record, _active, _wait, child, child_runtime = self._active_child(
            parent, node_invocation_id,
        )
        child_record = child.store.load()
        child_active = child_record.state["active"] if child_record is not None else None
        child_wait = child_active.get("wait") if child_active is not None else None
        if (child_record is None or child_record.state["status"] != "waiting"
                or child_wait is None):
            _fail("identity_mismatch", ("execution", "child"),
                  "nested Workflow is not waiting")
        node_type = child.nodes[child_active["node_id"]]["type"]
        if child_wait["kind"] == "interaction" and node_type == "tool_call":
            if child_wait["id"] != wait_id:
                _fail("identity_mismatch", ("wait_id",),
                      "nested Tool approval wait is not current")
            child.expire_tool_interaction(wait_id,
                node_invocation_id=child_active["node_invocation_id"],
                now_ms=now_ms)
        elif child_wait["kind"] == "child" and node_type == "workflow_call":
            child_runtime.expire_tool_interaction(
                child, node_invocation_id=child_active["node_invocation_id"],
                wait_id=wait_id, now_ms=now_ms,
            )
        else:
            _fail("invalid_transition", ("execution", "child", "wait"),
                  "nested Workflow wait is not a Tool approval")
        return self.resume_child(parent, node_invocation_id=node_invocation_id)

    def cancel_child(self, parent: DurableWorkflowRunner, *, node_invocation_id: str):
        record = parent.store.load()
        active = record.state["active"] if record is not None else None
        if active is None or active["node_invocation_id"] != node_invocation_id:
            _fail("identity_mismatch", ("node_invocation_id",), "child invocation is not current")
        child, _runtime, _published, _plan = self._child_for_active(active)
        if child.store.load() is None:
            return None
        try:
            return child.cancel()
        except JournalConflictError:
            # Parent cancellation can race a live descendant recording its own
            # terminal outcome after inherited authority ends. If that outcome
            # is already durable, it is safe to observe; otherwise retry once
            # against the new revision so cancellation remains authoritative.
            result = child.inspect()
            if result.status in _TERMINAL_FAILURES | {"completed"}:
                return result
            return child.cancel()


__all__ = ["PublishedWorkflow", "SkeletonWorkflowCallRuntime"]
