"""Durable exclusive Workflow execution with explicit host-supplied handlers.

This is an opt-in entry point, not a replacement for an Agent or tool executor.
Handler intents are never automatically replayed after an uncertain outcome.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from unchain.journal import JournalConflictError

from .conditions import evaluate_condition, select_case
from .durable_protocol import (
    STATE_SCHEMA, TERMINAL_STATUSES, canonical_bytes, fail, node_identity,
    validate_execution_state,
)
from .durable_store import WorkflowExecutionStore
from .errors import SkeletonValidationError
from .execution import (
    WorkflowNodeCompleted, _HANDLERS, _map_node_outputs, _mapped,
    _resolve_node_inputs, _validate_all, _value,
)
from .io import _validate_json
from .agent_preparation import checked_execution_plan, execution_definition_digest


@dataclass(frozen=True)
class WorkflowNodeWaiting:
    """A handler has durably suspended; host resolves this exact wait later."""

    kind: str
    id: str
    request: dict[str, Any]


@dataclass(frozen=True)
class DurableNodeContext:
    document_id: str
    node_id: str
    node_type: str
    graph_id: str
    node_invocation_id: str
    node_attempt_id: str
    assert_current: Callable[[], None]
    read_attachment: Callable[[str], bytes]


@dataclass(frozen=True)
class DurableWorkflowResult:
    graph_id: str
    status: str
    paused: bool
    revision: int
    outputs: dict[str, Any] | None
    completed_node_ids: tuple[str, ...]
    selected_edge_ids: tuple[str, ...]
    completion_id: str | None
    root_committed: bool
    error: dict[str, Any] | None


class DurableWorkflowRunner:
    """Advance a fixed plan by CAS, using the journal as execution authority."""

    def __init__(self, store: WorkflowExecutionStore, handlers: dict[str, Callable],
                 *, ancestor_assert_current: Callable[[], None] | None = None):
        if not isinstance(store, WorkflowExecutionStore):
            fail("invalid_type", ("store",), "WorkflowExecutionStore required")
        if type(handlers) is not dict:
            fail("invalid_type", ("handlers",), "handlers must be an object")
        if ancestor_assert_current is not None and not callable(ancestor_assert_current):
            fail("invalid_type", ("ancestor_assert_current",), "ancestor assertion must be callable")
        self.store = store
        self.plan = checked_execution_plan(store.plan)
        self.handlers = dict(handlers)
        self._ancestor_assert_current = ancestor_assert_current
        self.document = self.plan.snapshot()
        official_handlers = []
        if "tool_call" in self.handlers or "skill_read" in self.handlers:
            from .tool_runtime import SkeletonSkillRuntime, SkeletonToolRuntime
            official_handlers.extend((("tool_call", SkeletonToolRuntime), ("skill_read", SkeletonSkillRuntime)))
        if "code" in self.handlers:
            from .code_runtime import SkeletonCodeRuntime
            official_handlers.append(("code", SkeletonCodeRuntime))
        if "workflow_call" in self.handlers:
            from .workflow_runtime import SkeletonWorkflowCallRuntime
            official_handlers.append(("workflow_call", SkeletonWorkflowCallRuntime))
        for kind, official_type in official_handlers:
            handler = self.handlers.get(kind)
            if type(handler) is official_type:
                bound_plan = checked_execution_plan(handler.plan)
                if bound_plan != self.plan:
                    fail(
                        "identity_mismatch",
                        ("handlers", kind, "plan"),
                        "official handler definition differs from the execution store",
                    )
                if kind == "workflow_call" and handler.store is not self.store:
                    fail(
                        "identity_mismatch",
                        ("handlers", kind, "store"),
                        "official Workflow Call handler is bound to another execution store",
                    )
        self.configs = _validate_all(self.document, agent_graph=self.document["kind"] == "agent")
        self.handler_types = _HANDLERS | ({"agent"} if self.document["kind"] == "agent" else set())
        self.nodes = {node["id"]: node for node in self.document["nodes"]}
        self.positions = {node["id"]: i for i, node in enumerate(self.document["nodes"])}
        self.edges = {(edge.source_node_id, edge.source_port_id): edge for edge in self.plan.flow_edges}
        self.by_edge_id = {edge.id: edge for edge in self.plan.flow_edges}
        self.start_id = next(n["id"] for n in self.nodes.values() if n["type"] == "start")
        for node_id, node in self.nodes.items():
            outlets = {port for source, port in self.edges if source == node_id}
            if node["type"] == "if_else" and outlets != {"true", "false"}:
                fail("invalid_value", self._config_path(node_id), "if outlets must be true and false")
            if node["type"] == "switch_case":
                expected = {case["id"] for case in self.configs[node_id]["cases"]} | {"default"}
                if outlets != expected:
                    fail("invalid_value", self._config_path(node_id), "switch outlets do not match cases")

    def _config_path(self, node_id):
        return ("document", "nodes", self.positions[node_id], "config", "execution")

    def _runtime_path(self, node_id):
        return ("execution", "nodes", self.positions[node_id])

    def _load(self):
        record = self.store.load()
        if record is None:
            fail("missing_value", ("execution",), "execution has not been created")
        return record

    def _save(self, record, state, artifacts=()):
        return self.store.save(state, expected_revision=record.revision,
                               operation_id=uuid4().hex, artifacts=tuple(artifacts))

    def create(self, inputs: dict, *, attachments: dict | None = None, call_stack: list | None = None):
        """Snapshot inputs and optional ``name: (bytes, media_type)`` attachments."""
        _validate_json(inputs, ("inputs",))
        if type(inputs) is not dict:
            fail("invalid_type", ("inputs",), "root inputs must be an object")
        attachments = {} if attachments is None else attachments
        if type(attachments) is not dict:
            fail("invalid_type", ("attachments",), "attachments must be an object")
        root_ref, root_pending = self.store.prepare_json(inputs, key="root.inputs")
        refs = {}; pending = [root_pending]
        for name, value in attachments.items():
            if type(name) is not str or not name or type(value) is not tuple or len(value) != 2:
                fail("invalid_shape", ("attachments",), "attachment must have a name and bytes/media pair")
            ref, artifact = self.store.prepare_bytes(value[0], media_type=value[1], key="attachment." + name)
            refs[name] = ref; pending.append(artifact)
        attempt = self.store.attempt
        digest = execution_definition_digest(self.plan)
        current_frame = {"execution_id": attempt.generation.execution_id,
                         "graph_invocation_id": self.store.graph_id, "attempt_id": attempt.attempt_id,
                         "definition_id": self.plan.document_id, "definition_digest": digest}
        state = {"schema": STATE_SCHEMA, "graph_id": self.store.graph_id, "attempt": attempt.to_dict(),
                 "definition_digest": digest, "call_stack": [current_frame] if call_stack is None else deepcopy(call_stack),
                 "inputs_ref": root_ref, "attachments": refs, "completed": [], "active": None,
                 "status": "ready", "paused": False, "error": None, "root_committed": False}
        existing = self.store.load()
        if existing is not None:
            for key in ("attempt", "definition_digest", "call_stack", "inputs_ref", "attachments"):
                if canonical_bytes(existing.state[key]) != canonical_bytes(state[key]):
                    raise JournalConflictError("execution already exists with different fixed inputs")
            return self._result(existing)
        from .code_runtime import SkeletonCodeRuntime
        code_handler = self.handlers.get("code")
        pending_code = ()
        if type(code_handler) is SkeletonCodeRuntime:
            bindings, pending_code = code_handler.prepare_execution_bindings(refs, self.store.prepare_bytes)
            state["code_bindings"] = bindings
        validate_execution_state(state, self.plan)
        record = self.store.save(state, expected_revision=0, operation_id="create", artifacts=tuple(pending) + tuple(pending_code))
        return self._result(record)

    def _next(self, state):
        if not state["completed"]:
            return self.start_id
        edge_id = state["completed"][-1]["edge_id"]
        return None if edge_id is None else self.by_edge_id[edge_id].target_node_id

    def _result(self, record):
        state = record.state
        completed = state["status"] == "completed"
        outputs = self.store.read_json(state["completed"][-1]["output_ref"]) if completed else None
        if completed and type(outputs) is not dict:
            fail("invalid_type", ("execution", "end"), "End output must be an object")
        return DurableWorkflowResult(
            state["graph_id"], state["status"], state["paused"], record.revision, outputs,
            tuple(row["node_id"] for row in state["completed"]),
            tuple(row["edge_id"] for row in state["completed"] if row["edge_id"] is not None),
            state["completed"][-1]["node_invocation_id"] + ".end" if completed else None,
            state["root_committed"], deepcopy(state["error"]),
        )

    def inspect(self):
        return self._result(self._load())

    def _failed(self, record, error, *, unknown=False):
        state = deepcopy(record.state)
        state["status"] = "outcome_unknown" if unknown else "failed"
        state["paused"] = False
        if state["active"] is not None:
            state["active"].pop("resume", None)
        state["error"] = {"code": error.code, "path": list(error.path)}
        return self._save(record, state)

    def _inputs(self, node_id, state):
        exports = {row["node_id"]: self.store.read_json(row["output_ref"]) for row in state["completed"]}
        return _resolve_node_inputs(self.configs[node_id]["inputs"], exports,
                                    self._config_path(node_id), self._runtime_path(node_id))

    def _commit_node(self, record, node_id, resolved, result, port, input_ref=None, result_ref=None):
        state = deepcopy(record.state)
        kind = self.nodes[node_id]["type"]
        pending = []
        if kind == "agent":
            if (type(result) is dict and set(result) == {"schema", "code"}
                    and result["schema"] == "unchain.kernel_failure.v1"
                    and result["code"] in {"context_overflow", "unsupported_capability"}):
                fail(result["code"], self._runtime_path(node_id), "Kernel could not build a supported request")
            # The raw provider completion is already durable. Validation never
            # calls a model or discards the saved result on schema failure.
            if type(result) is not dict or set(result) != {"message", "context"} or type(result["message"]) is not str or type(result["context"]) is not dict:
                fail("invalid_shape", self._runtime_path(node_id), "Kernel result must contain message and context")
            config = self.nodes[node_id]["config"]["kernel"]
            result = deepcopy(result)
            if config["json_schema"] is not None:
                from .kernel_output import parse_kernel_output
                result.update(parse_kernel_output(result["message"], config["json_schema"]))
        if kind == "tool_call" and self.nodes[node_id]["config"]["tool"]["on_error"] == "stop":
            if type(result) is not dict or result.get("ok") is not True:
                fail("tool_failed", self._runtime_path(node_id), "Tool Call did not complete successfully")
        public_result = result
        if kind == "skill_read":
            if type(result) is dict and set(result) == {"_skeleton_error"}:
                error = result["_skeleton_error"]
                if type(error) is dict and set(error) == {"code"} and type(error["code"]) is str and error["code"]:
                    fail(error["code"], self._runtime_path(node_id), "Skill Read failed before publication")
                fail("skill_failed", self._runtime_path(node_id), "Skill Read failed before publication")
            if type(result) is not dict or set(result) != {"content", "name", "_skeleton_skill"}:
                fail("invalid_shape", self._runtime_path(node_id), "Skill Read result is invalid")
            metadata = result["_skeleton_skill"]
            if (type(result["content"]) is not str or type(result["name"]) is not str
                    or type(metadata) is not dict or set(metadata) != {"identity", "revision"}
                    or type(metadata["identity"]) is not dict or set(metadata["identity"]) != {"source", "source_id", "name"}
                    or type(metadata["revision"]) is not str):
                fail("invalid_shape", self._runtime_path(node_id), "Skill Read snapshot is invalid")
            public_result = {"content": result["content"], "name": result["name"]}
        if kind == "code" and type(result) is dict and set(result) == {"_skeleton_code"}:
            receipt = result["_skeleton_code"]
            required = {"schema", "result", "error", "code_sha256", "environment_sha256", "logs"}
            if type(receipt) is not dict or set(receipt) != required or receipt.get("schema") != "unchain.skeleton.code-receipt.v1":
                fail("invalid_shape", self._runtime_path(node_id), "Code receipt is invalid")
            hashes = (receipt["code_sha256"], receipt["environment_sha256"])
            logs = receipt["logs"]
            if (any(type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for value in hashes)
                    or type(logs) is not dict
                    or set(logs) != {"stdout", "stderr", "stdout_truncated", "stderr_truncated"}
                    or type(logs["stdout"]) is not str or type(logs["stderr"]) is not str
                    or type(logs["stdout_truncated"]) is not bool or type(logs["stderr_truncated"]) is not bool):
                fail("invalid_shape", self._runtime_path(node_id), "Code receipt provenance is invalid")
            error = receipt["error"]
            if error is not None:
                if type(error) is not dict or set(error) != {"code"} or type(error["code"]) is not str or not error["code"]:
                    fail("invalid_shape", self._runtime_path(node_id), "Code error receipt is invalid")
                if receipt["result"] is not None:
                    fail("invalid_shape", self._runtime_path(node_id), "Code error receipt cannot publish a result")
                fail(error["code"], self._runtime_path(node_id), "Code result was rejected")
            if type(receipt["result"]) is not dict:
                fail("invalid_shape", self._runtime_path(node_id), "Code result is invalid")
            public_result = receipt["result"]
        if kind in {"if_else", "switch_case"}:
            outputs = {}
        else:
            outputs = _map_node_outputs(self.configs[node_id]["outputs"], resolved, public_result,
                                        self._config_path(node_id), self._runtime_path(node_id))
        edge = None if kind == "end" else self.edges.get((node_id, port))
        if kind != "end" and edge is None:
            fail("missing_value", self._runtime_path(node_id) + ("selected_port",), "selected outlet is unavailable")
        invocation, attempt = node_identity(state, node_id)
        if input_ref is None:
            input_ref, item = self.store.prepare_json(resolved, key=invocation + ".input")
            pending.append(item)
        if result_ref is None:
            result_ref, item = self.store.prepare_json(result, key=invocation + ".raw")
            pending.append(item)
        output_ref, item = self.store.prepare_json(outputs, key=invocation + ".output")
        pending.append(item)
        state["completed"].append({"node_id": node_id, "node_invocation_id": invocation,
                                   "node_attempt_id": attempt, "input_ref": input_ref, "result_ref": result_ref,
                                   "output_ref": output_ref, "edge_id": None if edge is None else edge.id})
        state["active"] = None
        state["status"] = "completed" if kind == "end" else "ready"
        return self._save(record, state, pending)

    def _guard(self, node_id, token):
        record = self._load()
        active = record.state["active"]
        if record.state["status"] != "running" or active is None or active["node_id"] != node_id or active["owner_token"] != token:
            raise JournalConflictError("node dispatch ownership is no longer current")
        return record

    def _assert_current(self, node_id, token):
        """Verify this node, then any live parent authority it inherits."""
        record = self._guard(node_id, token)
        if self._ancestor_assert_current is not None:
            self._ancestor_assert_current()
        return record

    def _attachment(self, record, name):
        if type(name) is not str or name not in record.state["attachments"]:
            fail("missing_value", ("attachments",), "attachment is not granted to this graph")
        return self.store.read_bytes(record.state["attachments"][name])

    def _invoke(self, record, node_id, resolved):
        handler = self.handlers.get(self.nodes[node_id]["type"])
        if not callable(handler):
            fail("missing_value", self._runtime_path(node_id) + ("handler",), "handler is required")
        state = deepcopy(record.state)
        invocation, attempt = node_identity(state, node_id)
        if self.nodes[node_id]["type"] == "agent":
            from .kernel_runtime import SkeletonKernelHandler
            if type(handler) is SkeletonKernelHandler:
                handler.preflight_inputs(node_id, resolved)
        if self.nodes[node_id]["type"] == "code":
            from .code_runtime import SkeletonCodeRuntime
            if type(handler) is SkeletonCodeRuntime:
                binding = state.get("code_bindings", {}).get(node_id)
                if binding is not None:
                    handler.bind_node_execution(node_id, binding, self.store.read_bytes, attempt)
                handler.preflight_inputs(node_id, resolved, attempt)
        if self.nodes[node_id]["type"] == "workflow_call":
            from .workflow_runtime import SkeletonWorkflowCallRuntime
            if type(handler) is SkeletonWorkflowCallRuntime:
                handler.preflight_inputs(node_id, resolved)
        input_ref, pending = self.store.prepare_json(resolved, key=invocation + ".input")
        token = uuid4().hex
        state["active"] = {"node_id": node_id, "node_invocation_id": invocation, "node_attempt_id": attempt,
                           "input_ref": input_ref, "owner_token": token, "result_ref": None, "wait": None}
        state["status"] = "running"
        self._save(record, state, (pending,))
        current = self._guard(node_id, token)
        context = DurableNodeContext(self.plan.document_id, node_id, self.nodes[node_id]["type"],
                                     state["graph_id"], invocation, attempt,
                                     lambda: self._assert_current(node_id, token),
                                     lambda name: self._attachment(current, name))
        try:
            completion = handler(context, deepcopy(resolved))
        except SkeletonValidationError as error:
            current = self._guard(node_id, token)
            official_workflow_handler = False
            if self.nodes[node_id]["type"] == "workflow_call":
                from .workflow_runtime import SkeletonWorkflowCallRuntime
                official_workflow_handler = (
                    type(handler) is SkeletonWorkflowCallRuntime
                    and handler.store is self.store
                )
            official_kernel_handler = False
            if self.nodes[node_id]["type"] == "agent":
                from .kernel_runtime import SkeletonKernelHandler
                official_kernel_handler = (
                    type(handler) is SkeletonKernelHandler
                    and handler.store is self.store
                )
            if ((official_workflow_handler or official_kernel_handler)
                    and error.code in {"child_failed", "child_cancelled",
                                       "child_outcome_unknown", "hook_invalid_decision"}):
                return self._failed(
                    current, error, unknown=error.code == "child_outcome_unknown",
                )
            error = SkeletonValidationError(
                "handler_error", self._runtime_path(node_id),
                "handler outcome is unknown",
            )
            return self._failed(current, error, unknown=True)
        except Exception:
            current = self._guard(node_id, token)
            error = SkeletonValidationError("handler_error", self._runtime_path(node_id), "handler outcome is unknown")
            return self._failed(current, error, unknown=True)
        return self._seal_completion(node_id, token, invocation, completion)

    def _seal_completion(self, node_id, token, invocation, completion):
        current = self._guard(node_id, token)
        try:
            if type(completion) is WorkflowNodeWaiting:
                if type(completion.kind) is not str or completion.kind not in {"interaction", "child"} or type(completion.id) is not str or not completion.id:
                    fail("invalid_value", self._runtime_path(node_id), "invalid wait identity")
                if type(completion.request) is not dict:
                    fail("invalid_type", self._runtime_path(node_id), "wait request must be an object")
                ref, pending = self.store.prepare_json(completion.request, key=invocation + ".wait." + completion.id)
                state = deepcopy(current.state)
                state["status"] = "waiting"
                state["active"]["wait"] = {"kind": completion.kind, "id": completion.id, "request_ref": ref, "receipt_ref": None}
            else:
                if type(completion) is not WorkflowNodeCompleted or completion.selected_port != "out" or type(completion.result) is not dict:
                    fail("invalid_type", self._runtime_path(node_id) + ("result",), "handler must return an object completion on out")
                ref, pending = self.store.prepare_json(completion.result, key=invocation + ".raw")
                state = deepcopy(current.state)
                state["status"] = "result_received"
                state["active"]["result_ref"] = ref
            state["active"].pop("resume", None)
        except SkeletonValidationError as exc:
            return self._failed(current, exc, unknown=True)
        # Do not catch storage errors: a lost response may mean this already committed.
        # A cold reader determines whether the raw receipt exists before any recovery.
        for _ in range(8):
            try:
                return self._save(current, state, (pending,))
            except JournalConflictError:
                # Pause/resume can advance the revision while this handler
                # returns. Rebase only its receipt, never invoke it again.
                # Cancellation and explicit recovery revoke the token/phase.
                current = self._guard(node_id, token)
                state["paused"] = current.state["paused"]
        raise JournalConflictError("control state kept changing while sealing handler result")

    def step(self):
        record = self._load(); state = record.state
        if state["status"] not in {"ready", "result_received"} or state["paused"]:
            return self._result(record)
        node_id = self._next(state)
        kind = self.nodes[node_id]["type"]
        try:
            if state["status"] == "result_received":
                active = state["active"]
                resolved = self.store.read_json(active["input_ref"])
                result = self.store.read_json(active["result_ref"])
                record = self._commit_node(record, node_id, resolved, result, "out", active["input_ref"], active["result_ref"])
            elif kind in self.handler_types:
                resolved = self._inputs(node_id, state)
                # Exceptions after the intent use the durable running state, not this ready record.
                return self._result(self._invoke(record, node_id, resolved))
            else:
                cfg = self.configs[node_id]
                resolved = {} if kind == "start" else self._inputs(node_id, state)
                result = self.store.read_json(state["inputs_ref"]) if kind == "start" else {}
                port = "out"
                if kind == "if_else":
                    match = _mapped(lambda: evaluate_condition(cfg["condition"], resolved),
                                    {"condition": self._config_path(node_id) + ("condition",)})
                    port = "true" if match else "false"
                elif kind == "switch_case":
                    value = _value(cfg["value"], resolved, self._config_path(node_id) + ("value",))
                    port = _mapped(lambda: select_case(value, cfg["cases"]),
                                   {"cases": self._config_path(node_id) + ("cases",),
                                    "value": self._config_path(node_id) + ("value",)})
                record = self._commit_node(record, node_id, resolved, result, port)
        except SkeletonValidationError as exc:
            # Only a pre-effect error or a saved-result mapping error is safely failed.
            latest = self._load()
            if latest.revision != record.revision:
                raise
            record = self._failed(record, exc)
        return self._result(record)

    def run(self, *, max_steps: int | None = None):
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            fail("invalid_value", ("max_steps",), "max_steps must be a positive integer")
        steps = 0
        while True:
            result = self.inspect()
            if result.status not in {"ready", "result_received"} or result.paused:
                return result
            if max_steps is not None and steps >= max_steps:
                return result
            self.step(); steps += 1

    def pause(self):
        record = self._load()
        if record.state["status"] in TERMINAL_STATUSES or record.state["paused"]:
            return self._result(record)
        state = deepcopy(record.state); state["paused"] = True
        return self._result(self._save(record, state))

    def resume(self, *, max_steps=None):
        record = self._load()
        if record.state["paused"]:
            state = deepcopy(record.state); state["paused"] = False
            self._save(record, state)
        return self.run(max_steps=max_steps)

    def cancel(self):
        record = self._load()
        if record.state["status"] in TERMINAL_STATUSES:
            if record.state["status"] == "cancelled":
                self._cancel_active_child(record.state["active"])
            return self._result(record)
        active = record.state["active"]
        state = deepcopy(record.state); state["status"] = "cancelled"; state["paused"] = False
        cancelled = self._save(record, state)
        self._cancel_active_child(active)
        return self._result(cancelled)

    def _cancel_active_child(self, active):
        if active is None:
            return
        node_type = self.nodes[active["node_id"]]["type"]
        if node_type == "agent":
            from .kernel_runtime import SkeletonKernelHandler
            handler = self.handlers.get("agent")
            if type(handler) is SkeletonKernelHandler and handler.store is self.store:
                handler.cancel_active_hook(node_invocation_id=active["node_invocation_id"])
            return
        if node_type != "workflow_call":
            return
        from .workflow_runtime import SkeletonWorkflowCallRuntime
        handler = self.handlers.get("workflow_call")
        if type(handler) is SkeletonWorkflowCallRuntime:
            handler.cancel_child(self, node_invocation_id=active["node_invocation_id"])

    def recover(self):
        """Host declares the previous runner lost; never re-dispatch its intent."""
        record = self._load()
        if record.state["status"] == "cancelled":
            self._cancel_active_child(record.state["active"])
            return self._result(record)
        if record.state["status"] == "running":
            active = record.state["active"]
            if self.nodes[active["node_id"]]["type"] == "workflow_call":
                from .workflow_runtime import SkeletonWorkflowCallRuntime
                handler = self.handlers.get("workflow_call")
                if type(handler) is SkeletonWorkflowCallRuntime and handler.store is self.store:
                    context = DurableNodeContext(self.plan.document_id, active["node_id"], "workflow_call",
                        self.store.graph_id, active["node_invocation_id"], active["node_attempt_id"],
                        lambda: self._assert_current(active["node_id"], active["owner_token"]),
                        lambda name: self._attachment(record, name))
                    try:
                        completion = handler.recover_result(context)
                    except SkeletonValidationError as error:
                        return self._result(self._failed(
                            record, error, unknown=error.code == "child_outcome_unknown",
                        ))
                    if completion is not None:
                        return self._result(self._seal_completion(
                            active["node_id"], active["owner_token"],
                            active["node_invocation_id"], completion,
                        ))
            if self.nodes[active["node_id"]]["type"] == "agent":
                from .kernel_runtime import SkeletonKernelHandler
                handler = self.handlers.get("agent")
                if type(handler) is SkeletonKernelHandler and handler.store is self.store:
                    context = DurableNodeContext(self.plan.document_id, active["node_id"], "agent",
                        self.store.graph_id, active["node_invocation_id"], active["node_attempt_id"],
                        lambda: self._assert_current(active["node_id"], active["owner_token"]),
                        lambda name: self._attachment(record, name))
                    child_completion = handler.recover_child_continuation(
                        context, active.get("resume"),
                    )
                    if child_completion is not None:
                        resolved = self.store.read_json(active["input_ref"])
                        completion = handler.resume_child(
                            context, resolved, child_completion=child_completion,
                        )
                        return self._result(self._seal_completion(
                            active["node_id"], active["owner_token"],
                            active["node_invocation_id"], completion,
                        ))
                    completion = handler.recover_result(context)
                    if completion is not None:
                        return self._result(self._seal_completion(
                            active["node_id"], active["owner_token"],
                            active["node_invocation_id"], completion,
                        ))
                    resolved = self.store.read_json(active["input_ref"])
                    try:
                        completion = handler.recover_interrupted_tool_turn(context, resolved)
                    except SkeletonValidationError as error:
                        known = error.code in {
                            "child_failed", "child_cancelled", "hook_invalid_decision",
                        }
                        return self._result(self._failed(record, error, unknown=not known))
                    except Exception:
                        error = SkeletonValidationError("outcome_unknown",
                            ("execution", "active"),
                            "interrupted Tool could not be recovered safely")
                        return self._result(self._failed(record, error, unknown=True))
                    if completion is not None:
                        return self._result(self._seal_completion(
                            active["node_id"], active["owner_token"],
                            active["node_invocation_id"], completion,
                        ))
            error = SkeletonValidationError("outcome_unknown", ("execution", "active"), "interrupted handler has no durable result")
            record = self._failed(record, error, unknown=True)
        return self._result(record)

    def resume_kernel_interaction(self, wait_id, response, *, node_invocation_id, submitted_by="user"):
        """Resume the official Kernel checkpoint, never restart its model node."""
        from .kernel_runtime import SkeletonKernelHandler
        from unchain.journal.models import _required_text
        submitted_by = _required_text(submitted_by, "submitted_by", identifier=True)
        record = self._load()
        state = deepcopy(record.state)
        active = state["active"]
        if (state["status"] != "waiting" or state["paused"]
                or active["node_invocation_id"] != node_invocation_id
                or active["wait"]["id"] != wait_id or active["wait"]["kind"] != "interaction"
                or self.nodes[active["node_id"]]["type"] != "agent"):
            fail("identity_mismatch", ("wait_id",), "Kernel wait is not current")
        handler = self.handlers.get("agent")
        if type(handler) is not SkeletonKernelHandler or handler.store is not self.store:
            fail("invalid_type", ("handler",), "official bound Kernel handler required")
        node_id, invocation = active["node_id"], active["node_invocation_id"]
        token = uuid4().hex
        context = DurableNodeContext(self.plan.document_id, node_id, "agent", self.store.graph_id,
            invocation, active["node_attempt_id"], lambda: self._assert_current(node_id, token),
            lambda name: self._attachment(record, name))
        response = handler.validate_interaction(context, active["wait"], response)
        resolved = self.store.read_json(active["input_ref"])
        state["status"] = "running"
        active["owner_token"] = token
        active["wait"] = None
        self._save(record, state)
        try:
            completion = handler.resume_interaction(context, resolved, interaction_id=wait_id,
                response=response, submitted_by=submitted_by)
        except Exception:
            current = self._guard(node_id, token)
            error = SkeletonValidationError("handler_error", self._runtime_path(node_id), "resume outcome is unknown")
            return self._result(self._failed(current, error, unknown=True))
        return self._result(self._seal_completion(node_id, token, invocation, completion))

    def resume_kernel_child(self, wait_id, *, node_invocation_id, child_completion=None):
        """Re-enter one Kernel after its fixed child graph has advanced."""
        from .kernel_runtime import SkeletonKernelHandler
        record = self._load()
        state = deepcopy(record.state)
        active = state["active"]
        if (state["status"] != "waiting" or state["paused"]
                or active["node_invocation_id"] != node_invocation_id
                or active["wait"]["id"] != wait_id or active["wait"]["kind"] != "child"
                or self.nodes[active["node_id"]]["type"] != "agent"):
            fail("identity_mismatch", ("wait_id",), "Kernel child wait is not current")
        handler = self.handlers.get("agent")
        if type(handler) is not SkeletonKernelHandler or handler.store is not self.store:
            fail("invalid_type", ("handler",), "official bound Kernel handler required")
        node_id, invocation = active["node_id"], active["node_invocation_id"]
        token = uuid4().hex
        context = DurableNodeContext(self.plan.document_id, node_id, "agent", self.store.graph_id,
            invocation, active["node_attempt_id"], lambda: self._assert_current(node_id, token),
            lambda name: self._attachment(record, name))
        resolved = self.store.read_json(active["input_ref"])
        state["status"] = "running"
        active["owner_token"] = token
        active["wait"] = None
        active["resume"] = {
            "kind": "child",
            "wait_id": wait_id,
            "request_ref": deepcopy(record.state["active"]["wait"]["request_ref"]),
        }
        self._save(record, state)
        try:
            completion = handler.resume_child(
                context,
                resolved,
                child_completion=deepcopy(child_completion),
            )
        except Exception:
            current = self._guard(node_id, token)
            error = SkeletonValidationError("handler_error", self._runtime_path(node_id), "child resume outcome is unknown")
            return self._result(self._failed(current, error, unknown=True))
        return self._result(self._seal_completion(node_id, token, invocation, completion))

    def fail_kernel_child(self, wait_id, *, node_invocation_id, code="subagent_failed"):
        """Terminally fail a Kernel whose handoff child cannot complete."""
        if type(code) is not str or not code:
            fail("invalid_value", ("code",), "child failure code is invalid")
        record = self._load()
        state = record.state
        active = state["active"]
        if (state["status"] != "waiting" or state["paused"] or active is None
                or active["node_invocation_id"] != node_invocation_id
                or active["wait"] is None or active["wait"]["kind"] != "child"
                or active["wait"]["id"] != wait_id
                or self.nodes[active["node_id"]]["type"] != "agent"):
            fail("identity_mismatch", ("wait_id",), "Kernel child wait is not current")
        return self._result(self._failed(record, SkeletonValidationError(
            code,
            self._runtime_path(active["node_id"]),
            "subagent child did not complete",
        )))

    def complete_kernel_child(self, wait_id, result, *, node_invocation_id):
        """Seal a handoff child's terminal output without re-running its parent model."""
        if type(result) is not dict:
            fail("invalid_type", ("result",), "handoff child result must be an object")
        record = self._load()
        state = deepcopy(record.state)
        active = state["active"]
        if (state["status"] != "waiting" or state["paused"]
                or active["node_invocation_id"] != node_invocation_id
                or active["wait"]["id"] != wait_id or active["wait"]["kind"] != "child"
                or self.nodes[active["node_id"]]["type"] != "agent"):
            fail("identity_mismatch", ("wait_id",), "Kernel child wait is not current")
        node_id, invocation = active["node_id"], active["node_invocation_id"]
        token = uuid4().hex
        state["status"] = "running"
        active["owner_token"] = token
        active["wait"] = None
        self._save(record, state)
        return self._result(self._seal_completion(
            node_id,
            token,
            invocation,
            WorkflowNodeCompleted(deepcopy(result)),
        ))

    def replace_kernel_child_wait(self, wait_id, request, *, next_wait_id, node_invocation_id):
        """Carry a nested child's next wait without re-running this Kernel."""
        if type(next_wait_id) is not str or not next_wait_id or type(request) is not dict:
            fail("invalid_value", ("wait",), "child wait replacement is invalid")
        record = self._load()
        state = deepcopy(record.state)
        active = state["active"]
        if (state["status"] != "waiting" or state["paused"] or active is None
                or active["node_invocation_id"] != node_invocation_id
                or active["wait"] is None or active["wait"]["kind"] != "child"
                or active["wait"]["id"] != wait_id
                or self.nodes[active["node_id"]]["type"] != "agent"):
            fail("identity_mismatch", ("wait_id",), "Kernel child wait is not current")
        ref, pending = self.store.prepare_json(
            request, key=active["node_invocation_id"] + ".wait." + next_wait_id,
        )
        active["wait"] = {"kind": "child", "id": next_wait_id,
                          "request_ref": ref, "receipt_ref": None}
        return self._result(self._save(record, state, (pending,)))

    def resume_tool_interaction(self, wait_id, response, *, node_invocation_id, submitted_by="user"):
        """Persist one exact approval receipt, then invoke the official Tool executor once."""
        from .tool_runtime import SkeletonToolRuntime
        from unchain.journal.models import _required_text
        submitted_by = _required_text(submitted_by, "submitted_by", identifier=True)
        record = self._load()
        state = deepcopy(record.state)
        active = state["active"]
        if (state["status"] != "waiting" or state["paused"]
                or active["node_invocation_id"] != node_invocation_id
                or active["wait"]["id"] != wait_id or active["wait"]["kind"] != "interaction"
                or self.nodes[active["node_id"]]["type"] != "tool_call"):
            fail("identity_mismatch", ("wait_id",), "Tool wait is not current")
        handler = self.handlers.get("tool_call")
        if type(handler) is not SkeletonToolRuntime:
            fail("invalid_type", ("handler",), "official bound Tool handler required")
        node_id, invocation = active["node_id"], active["node_invocation_id"]
        context = DurableNodeContext(self.plan.document_id, node_id, "tool_call", self.store.graph_id,
            invocation, active["node_attempt_id"], lambda: None, lambda name: self._attachment(record, name))
        resolved = self.store.read_json(active["input_ref"])
        request_payload = self.store.read_json(active["wait"]["request_ref"])
        receipt_payload = (self.store.read_json(active["wait"]["receipt_ref"])
                           if active["wait"]["receipt_ref"] is not None else None)
        receipt, preparation, normalized = handler.seal_interaction(
            context, resolved, request_payload, receipt_payload, response, submitted_by=submitted_by,
        )
        if active["wait"]["receipt_ref"] is None:
            ref, pending = self.store.prepare_json(receipt, key=invocation + ".wait." + wait_id + ".receipt")
            state["active"]["wait"]["receipt_ref"] = ref
            record = self._save(record, state, (pending,))
            state = deepcopy(record.state)
            active = state["active"]
        token = uuid4().hex
        state["status"] = "running"
        active["owner_token"] = token
        active["wait"] = None
        self._save(record, state)
        context = DurableNodeContext(self.plan.document_id, node_id, "tool_call", self.store.graph_id,
            invocation, active["node_attempt_id"], lambda: self._assert_current(node_id, token),
            lambda name: self._attachment(record, name))
        try:
            completion = handler.resume_interaction(context, resolved, preparation, normalized)
        except Exception:
            current = self._guard(node_id, token)
            error = SkeletonValidationError("handler_error", self._runtime_path(node_id), "Tool resume outcome is unknown")
            return self._result(self._failed(current, error, unknown=True))
        return self._result(self._seal_completion(node_id, token, invocation, completion))

    def expire_tool_interaction(self, wait_id, *, node_invocation_id, now_ms):
        """Fail an unexecuted Tool approval after its persisted deadline."""
        from .tool_runtime import SkeletonToolRuntime
        record = self._load(); state = deepcopy(record.state); active = state["active"]
        if (state["status"] != "waiting" or type(now_ms) is not int or isinstance(now_ms, bool)
                or active["node_invocation_id"] != node_invocation_id or active["wait"]["id"] != wait_id
                or self.nodes[active["node_id"]]["type"] != "tool_call"):
            fail("identity_mismatch", ("wait_id",), "Tool wait is not current")
        handler = self.handlers.get("tool_call")
        if type(handler) is not SkeletonToolRuntime:
            fail("invalid_type", ("handler",), "official bound Tool handler required")
        if not handler.is_expired(self.store.read_json(active["wait"]["request_ref"]), now_ms):
            fail("invalid_transition", ("wait_id",), "Tool approval has not expired")
        return self._result(self._failed(record, SkeletonValidationError(
            "interaction_expired", self._runtime_path(active["node_id"]), "Tool approval expired",
        )))

    def complete_wait(self, wait_id: str, result: dict, *, node_invocation_id: str):
        """Accept one host-validated wait completion, not an approval bypass API."""
        record = self._load(); state = deepcopy(record.state)
        if (state["status"] != "waiting" or type(wait_id) is not str
                or type(node_invocation_id) is not str
                or state["active"]["node_invocation_id"] != node_invocation_id
                or state["active"]["wait"]["id"] != wait_id):
            fail("identity_mismatch", ("wait_id",), "wait is not current")
        node_type = self.nodes[state["active"]["node_id"]]["type"]
        if node_type in {"agent", "tool_call"}:
            fail("invalid_transition", ("wait_id",), node_type + " waits require official interaction resume")
        if type(result) is not dict:
            fail("invalid_type", ("result",), "wait result must be an object")
        ref, pending = self.store.prepare_json(result, key=state["active"]["node_invocation_id"] + ".raw")
        state["active"]["result_ref"] = ref; state["active"]["wait"] = None
        state["status"] = "result_received"
        return self._result(self._save(record, state, (pending,)))

    def commit_root(self):
        """Persist the End completion interface; actual chat projection is host-owned."""
        record = self._load(); state = deepcopy(record.state)
        if state["status"] != "completed" or len(state["call_stack"]) != 1:
            fail("invalid_transition", ("root_committed",), "only completed root graphs may commit")
        if not state["root_committed"]:
            state["root_committed"] = True
            record = self._save(record, state)
        return self._result(record)
