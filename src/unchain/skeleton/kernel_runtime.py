"""Compose Skeleton Kernel nodes with Unchain's official durable model loop.

Host capabilities supply the existing journal/compiler repositories and tools.
This module owns no database and never commits the root conversation.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import re

from unchain.journal import AttemptRef, EventCursor, EventRange, JournalAppendRequest, SemanticEventDraft, build_operation_ref, journal_event_sha256
from unchain.journal.models import _thaw_json

from .agent_preparation import PreparedAgentPlan, checked_execution_plan, execution_definition_digest
from .durable_execution import WorkflowNodeCompleted, WorkflowNodeWaiting
from .durable_protocol import canonical_bytes
from .errors import SkeletonValidationError

_EVENT = "skeleton.kernel_result.saved"
_SCHEMA = "unchain.skeleton_kernel_result.v1"
_CHILD_TOOL_SETTLEMENT_EVENT = "skeleton.subagent_tool_settlement"
_CHILD_TOOL_SETTLEMENT_SCHEMA = "unchain.skeleton_subagent_tool_settlement.v2"
_SUBAGENT_WAIT_SCHEMA_V1 = "pupu.skeleton_subagent_wait.v1"
_SUBAGENT_WAIT_SCHEMA_V2 = "pupu.skeleton_subagent_wait.v2"
_SUBAGENT_WAIT_FIELDS = {
    "schema", "parent_graph_id", "parent_node_invocation_id", "parent_kernel_node_id",
    "child_graph_id", "child_attempt_id", "entry_id", "definition_digest",
    "child_guard_session_id", "child_node_invocation_id", "child_wait_kind", "child_wait_id",
    "interaction_request", "mode",
}
_SUBAGENT_WAIT_V2_FIELDS = _SUBAGENT_WAIT_FIELDS | {"parent_tool_call"}
_INPUT_PROMPT_VARIABLE = re.compile(r"\{\{#input\.([^#}]+)#\}\}")


def _render_prompt(template, resolved):
    def replace(match):
        value = resolved.get(match.group(1), "")
        return value if type(value) is str else str(value)
    return _INPUT_PROMPT_VARIABLE.sub(replace, template)


def build_node_bundle(*, journal, compiler_store, view, config, parent_attempt_id,
                      artifact_sanitizer, payload_sanitizer, partial_attempt_sink, compression_service=None,
                      user_message_artifact_sanitizer=None, remote_source_decoder=None):
    """Use the deployed journal and compiler ports, with one immutable view."""
    from unchain.context import (
        ArtifactService, ContextCompileCoordinator, ContextExecutionBundle,
        ContextInputIngress, DurableHandoffRecorder, DurableToolBoundary,
        HandoffService, JournalContextRequestFactory,
    )
    from unchain.context.model_projection import ModelContextProjection
    from unchain.context.projector import CanonicalSemanticEventProjector
    from unchain.context.provider_execution import ContextProviderTurnExecutionService, official_provider_transport_target_sha256
    from unchain.journal import DurableEventSink
    from unchain.providers.durable_turn_runtime import DurableProviderTurnMode

    attempt = view.attempt
    if type(parent_attempt_id) is not str or not parent_attempt_id or parent_attempt_id == attempt.attempt_id:
        raise ValueError("Kernel node requires a distinct parent attempt")
    view.persist(journal)
    artifacts = ArtifactService(journal, sanitizer=artifact_sanitizer,
                                user_message_sanitizer=user_message_artifact_sanitizer or artifact_sanitizer)
    compiler = compiler_store.bind_execution(journal.execution_id, artifacts=artifacts)
    def scoped_payload(kind, payload):
        result = dict(payload_sanitizer(kind, payload))
        if result.get("parent_run_id", parent_attempt_id) != parent_attempt_id:
            raise ValueError("Kernel node event changed parent")
        result["parent_run_id"] = parent_attempt_id
        return result
    projector = CanonicalSemanticEventProjector(attempt=attempt, artifacts=artifacts,
                                                payload_sanitizer=scoped_payload)
    sink = DurableEventSink(journal, attempt, projector)
    handoffs = HandoffService(artifacts)
    coordinator = ContextCompileCoordinator(journal=journal, artifacts=artifacts,
        checkpoint_repository=compiler.checkpoints, build_repository=compiler.context_builds,
        partial_attempt_sink=partial_attempt_sink, model_projection=ModelContextProjection(artifacts, remote_source_decoder=remote_source_decoder),
        provider_turn_result_reader=journal.read_full_verified,
        node_memory_view=view, compression_service=compression_service)
    request_factory = JournalContextRequestFactory(attempt=attempt, journal=journal,
        artifacts=artifacts, model_window_fallback=lambda _p, _m: config["profile"]["context_window"],
        output_reserve_tokens=config["output_tokens"], node_memory_view=view)
    return ContextExecutionBundle(attempt=attempt, journal=journal, projector=projector,
        durable_event_sink=sink, coordinator=coordinator, artifacts=artifacts, handoffs=handoffs,
        ingress=ContextInputIngress(attempt=attempt, projector=projector, sink=sink),
        request_factory=request_factory,
        provider_turn_service=ContextProviderTurnExecutionService(attempt=attempt, store=journal,
            mode=DurableProviderTurnMode.ENFORCE, transport_target_sha256=official_provider_transport_target_sha256()),
        tool_boundary=DurableToolBoundary(attempt=attempt, projector=projector, sink=sink),
        handoff_recorder=DurableHandoffRecorder(attempt=attempt, handoffs=handoffs, projector=projector, sink=sink),
        partial_attempt_sink=partial_attempt_sink)


class SkeletonKernelHandler:
    """A host-supplied Agent handler; graph dispatch still owns the lease."""

    def __init__(self, store, *, bind_profile, bundle_builder, tools_for_node=None,
                 attachments_for_node=None, modules_for_node=None, callback=None,
                 execution_guard=None, durability_runtime=None, fyi_channel=None,
                 run_bundle_root_id=None, run_bundle_parent_id=None,
                 hook_workflow_resolver=None, hook_runtime=None,
                 hook_workflow_runner=None, hook_workflow_canceller=None):
        if type(store.plan) is not PreparedAgentPlan:
            raise ValueError("Kernel handler requires an Agent plan")
        self.store = store
        self.plan = checked_execution_plan(store.plan)
        self.definition_digest = execution_definition_digest(self.plan)
        self.configs = {n["id"]: n["config"]["kernel"] for n in self.plan.snapshot()["nodes"] if n["type"] == "agent"}
        from .hooks_config import attached_hooks
        has_attached_hooks = bool(attached_hooks(self.plan.snapshot()))
        if hook_runtime is not None:
            from .hook_runtime import SkeletonHookRuntime
            if type(hook_runtime) is not SkeletonHookRuntime:
                raise TypeError("Kernel Hook runtime must be an admitted SkeletonHookRuntime")
            if not has_attached_hooks:
                raise ValueError("Kernel Hook runtime requires an attached Hook Pool")
            if hook_runtime.plan.snapshot() != self.plan.snapshot():
                raise ValueError("Kernel Hook runtime changed the fixed definition")
            self.hook_runtime = hook_runtime
        elif has_attached_hooks:
            if not callable(hook_workflow_resolver):
                raise ValueError("Hook pools require a callable hook_workflow_resolver")
            from .hook_runtime import SkeletonHookRuntime
            self.hook_runtime = SkeletonHookRuntime(self.plan, hook_workflow_resolver)
        else:
            self.hook_runtime = None
        if hook_workflow_runner is not None and not callable(hook_workflow_runner):
            raise TypeError("Kernel Hook workflow runner must be callable")
        if hook_workflow_canceller is not None and not callable(hook_workflow_canceller):
            raise TypeError("Kernel Hook workflow canceller must be callable")
        self.hook_workflow_runner = hook_workflow_runner
        self.hook_workflow_canceller = hook_workflow_canceller
        self.bind_profile = bind_profile
        # Preflight every branch before the graph is started. No model calls here.
        self.bindings = {key: bind_profile(deepcopy(config)) for key, config in self.configs.items()}
        from .kernel_profile import KernelProfileBinding
        for key, binding in self.bindings.items():
            if type(binding) is not KernelProfileBinding or _thaw_json(binding.config) != self.configs[key]:
                raise ValueError("Kernel profile binding changed the fixed definition")
        self.bundle_builder = bundle_builder
        self.tools_for_node = tools_for_node or (lambda _context: ())
        self.attachments_for_node = attachments_for_node or (lambda _context, _inputs: ())
        self.modules_for_node = modules_for_node or (lambda _context: ())
        self.callback = callback
        self.execution_guard = execution_guard
        if (run_bundle_root_id is None) != (run_bundle_parent_id is None):
            raise ValueError("Kernel run bundle scope requires root and parent together")
        if run_bundle_root_id is not None and (
            type(run_bundle_root_id) is not str or not run_bundle_root_id
            or type(run_bundle_parent_id) is not str or not run_bundle_parent_id
        ):
            raise ValueError("Kernel run bundle scope must use non-empty ids")
        self.run_bundle_root_id = run_bundle_root_id
        self.run_bundle_parent_id = run_bundle_parent_id
        if durability_runtime is not None:
            from unchain.memory import KernelMemoryRuntime
            if type(durability_runtime) is not KernelMemoryRuntime:
                raise TypeError("Kernel durability requires KernelMemoryRuntime")
        self.durability_runtime = durability_runtime
        if fyi_channel is not None:
            from unchain.interaction import FyiChannel
            if type(fyi_channel) is not FyiChannel or len(self.configs) != 1:
                raise ValueError("FYI requires one Kernel node and an exact FyiChannel")
        self.fyi_channel = fyi_channel

    def cancel_active_hook(self, *, node_invocation_id):
        """Ask the host to stop the Hook child owned by this agent invocation."""
        if type(node_invocation_id) is not str or not node_invocation_id:
            raise ValueError("Kernel Hook cancellation requires a node invocation id")
        if self.hook_workflow_canceller is None:
            return None
        return self.hook_workflow_canceller(
            parent_graph_id=self.store.graph_id,
            parent_node_invocation_id=node_invocation_id,
        )

    def _view(self, attempt, config):
        from unchain.context.node_memory_view import NodeMemoryView
        record = self.store.load()
        if record is None and config["history"] != "none":
            raise ValueError("node memory view requires a created graph")
        # Private compression operations are owned by their durable intent,
        # rather than a public graph, and read no root history at all.
        current_root_attempt_id = None if record is None else record.state["call_stack"][0]["attempt_id"]
        snapshot = self.store.journal.capture_snapshot()
        candidates = [event for event in snapshot.events if event.attempt == attempt and event.event_type == "context.node_memory_view"]
        if candidates:
            if len(candidates) != 1:
                raise ValueError("duplicate node memory view")
            view = NodeMemoryView.from_dict(_thaw_json(candidates[0].payload))
            if (view.attempt != attempt or view.definition_digest != self.definition_digest
                    or view.current_root_attempt_id != current_root_attempt_id
                    or (config["history"] == "none" and view.history)):
                raise ValueError("node memory view definition changed")
            view.select(snapshot)
            return view
        history = []
        if config["history"] == "root":
            for event in snapshot.events:
                if (event.attempt.generation == attempt.generation and event.attempt != attempt
                        and event.event_type in {"message.user", "message.assistant", "final_message", "run_completed"}
                        and event.payload.get("run_id") == event.attempt.attempt_id
                        and not event.payload.get("parent_run_id")
                        and not any(event.payload.get(key) is not None for key in ("workflow_node_id", "workflow_step_index", "workflow_step_count"))):
                    history.append((event.event_id, event.store_seq, journal_event_sha256(event)))
        view = NodeMemoryView(attempt, self.definition_digest, tuple(history), current_root_attempt_id)
        view.persist(self.store.journal)
        return view

    def _identity(self, context):
        return {"schema": _SCHEMA, "graph_id": self.store.graph_id,
                "node_id": context.node_id, "node_invocation_id": context.node_invocation_id,
                "attempt": AttemptRef(self.store.attempt.generation, context.node_attempt_id).to_dict(),
                "definition_digest": self.definition_digest}

    def _runtime_session_id(self, attempt):
        """Keep a child interaction checkpoint in its descendant session."""
        if self.execution_guard is not None:
            return self.execution_guard.lease.execution_id
        return attempt.generation.execution_id

    @staticmethod
    def _request_with_parent_tool_call(request, parent_tool_call):
        """Upgrade one emitted PuPu child wait to its call-bound v2 wire form."""

        if (type(request) is not dict or set(request) != _SUBAGENT_WAIT_FIELDS
                or request.get("schema") != _SUBAGENT_WAIT_SCHEMA_V1):
            raise ValueError("subagent wait cannot bind a parent tool")
        if (type(parent_tool_call) is not dict
                or set(parent_tool_call) != {"call_id", "tool_name"}
                or any(type(value) is not str or not value.strip()
                       for value in parent_tool_call.values())):
            raise ValueError("subagent parent tool call is invalid")
        return {
            **request,
            "schema": _SUBAGENT_WAIT_SCHEMA_V2,
            "parent_tool_call": {
                "call_id": parent_tool_call["call_id"].strip(),
                "tool_name": parent_tool_call["tool_name"].strip(),
            },
        }

    def _next_provider_iteration(self, attempt):
        """Reserve the next durable provider turn for a child-resume continuation."""
        from unchain.journal.tool_catalog import TOOL_CATALOG_SNAPSHOT_EVENT_TYPE

        iterations = []
        for event in self.store.journal.capture_snapshot().events:
            if (
                event.attempt != attempt
                or event.event_type != TOOL_CATALOG_SNAPSHOT_EVENT_TYPE
            ):
                continue
            payload = _thaw_json(event.payload)
            iteration = payload.get("iteration") if type(payload) is dict else None
            if (
                isinstance(iteration, bool)
                or not isinstance(iteration, int)
                or iteration < 0
            ):
                raise ValueError("Kernel provider catalog has an invalid iteration")
            iterations.append(iteration)
        return max(iterations, default=-1) + 1

    def _run_identity(self, context):
        from unchain.run_bundle import RunIdentity
        root_run_id = self.run_bundle_root_id or self.store.attempt.attempt_id
        parent_run_id = self.run_bundle_parent_id or self.store.attempt.attempt_id
        return RunIdentity(
            execution_id=self.store.attempt.generation.execution_id,
            attempt_id=context.node_attempt_id,
            root_run_id=root_run_id,
            run_id=context.node_attempt_id,
            parent_run_id=parent_run_id,
            relation="graph_node" if self.run_bundle_parent_id is None else "subagent",
        )

    def preflight_inputs(self, node_id, resolved):
        from unchain.context.budget import estimate_context_tokens, resolve_context_budget
        config = self.configs[node_id]
        try:
            budget = resolve_context_budget(context_window_tokens=config["profile"]["context_window"],
                output_reserve_tokens=config["output_tokens"])
        except (TypeError, ValueError) as exc:
            raise SkeletonValidationError("context_overflow", ("kernel", "profile"), "model window has no input capacity") from exc
        if budget.output_reserve_tokens != config["output_tokens"]:
            raise SkeletonValidationError("context_overflow", ("kernel", "output_tokens"), "output reservation cannot be reduced")
        task = json.dumps(resolved, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        required = [{"role": "system", "content": _render_prompt(config["prompt"], resolved)},
                    {"role": "user", "content": task}]
        if estimate_context_tokens(required).total_tokens > budget.available_input_tokens:
            raise SkeletonValidationError("context_overflow", ("kernel", "inputs"), "required inputs exceed model context")

    def _request(self, context, ref):
        identity = self._identity(context)
        payload = {**identity, "result_ref": ref}
        operation_id = "skeleton.kernel-result." + sha256(canonical_bytes(identity)).hexdigest()
        from unchain.journal import ArtifactRef
        return JournalAppendRequest(event_id="event." + operation_id, event_type=_EVENT,
            attempt=AttemptRef.from_dict(identity["attempt"]),
            operation=build_operation_ref(operation_id, domain=_EVENT, payload=payload),
            payload=payload, resource_refs=(ArtifactRef.from_dict(ref).ref,))

    def recover_result(self, context):
        identity = self._identity(context)
        snapshot = self.store.journal.capture_snapshot()
        matches = [event for event in snapshot.events
                   if event.event_type == _EVENT and event.attempt.attempt_id == context.node_attempt_id]
        if matches:
            if len(matches) != 1:
                raise ValueError("Kernel result receipt is ambiguous")
            event = matches[0]
            payload = _thaw_json(event.payload)
            if set(payload) != set(identity) | {"result_ref"} or any(payload[k] != v for k, v in identity.items()):
                raise ValueError("Kernel result scope changed")
            expected = self._request(context, payload["result_ref"])
            if (event.attempt != expected.attempt or event.event_id != expected.event_id
                    or event.operation != expected.operation or event.resource_refs != expected.resource_refs):
                raise ValueError("Kernel result receipt changed")
            result = self.store.read_json(payload["result_ref"])
            failure = (type(result) is dict and set(result) == {"schema", "code"}
                       and result["schema"] == "unchain.kernel_failure.v1"
                       and result["code"] in {"context_overflow", "unsupported_capability"})
            if not failure and (type(result) is not dict or set(result) != {"message", "context"}
                    or type(result.get("message")) is not str or type(result.get("context")) is not dict):
                raise ValueError("Kernel result shape changed")
            return WorkflowNodeCompleted(result)
        waiting = self._pending_interaction(context)
        if waiting is not None:
            return waiting
        completed = self._completed_output(context, snapshot)
        if completed is None:
            return None
        return self._save_completion(context, completed)

    def recover_interrupted_tool_turn(self, context, resolved):
        """Resume a sealed Tool from its verified provider turn, without a new send."""
        from unchain.journal.provider_result import (
            provider_turn_result_event_fields, recover_provider_turn_result,
        )
        attempt = AttemptRef(self.store.attempt.generation, context.node_attempt_id)
        events = [event for event in self.store.journal.capture_snapshot().events
                  if event.attempt == attempt]
        provider_events = [event for event in events
                           if event.event_type == "provider.turn_result"]
        if not provider_events:
            return None
        candidates = []
        for event in provider_events:
            fields = provider_turn_result_event_fields(event)
            if fields.subject.attempt != attempt:
                raise ValueError("provider result changed Kernel attempt")
            candidates.append(fields)
        iteration = max(fields.subject.iteration for fields in candidates)
        candidates = [fields for fields in candidates
                      if fields.subject.iteration == iteration]
        if len(candidates) != 1:
            raise ValueError("interrupted Tool has no unique durable provider result")
        fields = candidates[0]
        turn = recover_provider_turn_result(self.store.journal,
            subject=fields.subject,
            expected_route_sha256=fields.route_sha256).envelope.to_model_turn_result()
        if not turn.tool_calls:
            return None
        unfinished_seals = [event for event in events
            if event.event_type == "tool.hook_raw.sealed"
            and not any(result.event_type in {"tool_result", "tool.result"}
                        and result.payload.get("call_id") == event.payload.get("call_id")
                        for result in events)]
        if (any(event.payload.get("iteration") != iteration for event in unfinished_seals)
                or not {event.payload.get("call_id") for event in unfinished_seals}.issubset(
                    {call.call_id for call in turn.tool_calls})):
            raise ValueError("unfinished Hook seal changed the provider call batch")
        return self._execute(context, resolved, recovery_mode=True,
            recovered_tool_turn=(iteration, turn))

    def recover_child_continuation(self, context, resume):
        """Rebuild one deferred child completion from durable receipts only.

        The marker retains the original child wait.  A handoff may have reached
        the journal before its call-bound settlement, so recovery derives the
        completion again from the child terminal state and lets the normal,
        idempotent ingress finish the missing receipt.
        """
        from unchain.context.compiler import _subagent_tool_settlement
        from unchain.context import HandoffEnvelope
        from unchain.journal.provider_wire import PROVIDER_WIRE_SNAPSHOT_EVENT_TYPE
        from unchain.journal.provider_result import PROVIDER_TURN_RESULT_EVENT_TYPE
        from unchain.journal.tool_catalog import TOOL_CATALOG_SNAPSHOT_EVENT_TYPE

        if (type(resume) is not dict or set(resume) != {"kind", "wait_id", "request_ref"}
                or resume["kind"] != "child" or type(resume["wait_id"]) is not str
                or not resume["wait_id"]):
            return None
        attempt = AttemptRef(self.store.attempt.generation, context.node_attempt_id)
        try:
            request = self.store.read_json(resume["request_ref"])
            parent_tool_call = request["parent_tool_call"]
            child_attempt_id = request["child_attempt_id"]
            if (type(request) is not dict or set(request) != _SUBAGENT_WAIT_V2_FIELDS
                    or request.get("schema") != _SUBAGENT_WAIT_SCHEMA_V2
                    or request.get("parent_graph_id") != self.store.graph_id
                    or request.get("parent_node_invocation_id") != context.node_invocation_id
                    or request.get("parent_kernel_node_id") != context.node_id
                    or any(type(request.get(key)) is not str or not request[key]
                           for key in ("child_graph_id", "child_attempt_id", "child_wait_id"))
                    or resume["wait_id"] != "subagent." + sha256(
                        (request["child_graph_id"] + "\x00" + request["child_wait_id"])
                        .encode("utf-8", "surrogatepass")
                    ).hexdigest()
                    or type(parent_tool_call) is not dict
                    or set(parent_tool_call) != {"call_id", "tool_name"}
                    or any(type(value) is not str or not value.strip()
                           for value in parent_tool_call.values())
                    or type(child_attempt_id) is not str or not child_attempt_id):
                return None
            source_attempt = AttemptRef(attempt.generation, child_attempt_id)
        except (KeyError, TypeError, ValueError):
            return None
        events = self.store.journal.capture_snapshot().events
        source_events = [event for event in events if event.attempt == source_attempt]
        if not source_events:
            return None
        terminal_event = source_events[-1]
        try:
            completion = self._completion_from_child_terminal(
                source_attempt=source_attempt, terminal_event=terminal_event,
            )
        except ValueError:
            return None
        settlements = []
        for event in events:
            if event.attempt != attempt or event.event_type != _CHILD_TOOL_SETTLEMENT_EVENT:
                continue
            try:
                payload = _thaw_json(event.payload)
                _subagent_tool_settlement(payload)
                if AttemptRef.from_dict(payload["consumer_attempt"]) != attempt:
                    return None
                settled_attempt = AttemptRef.from_dict(payload["source_attempt"])
            except (KeyError, TypeError, ValueError):
                return None
            if settled_attempt == source_attempt:
                settlements.append(event)
            elif event.store_seq > terminal_event.store_seq:
                return None
        anchor = None
        if settlements:
            if len(settlements) != 1:
                return None
            anchor = settlements[0]
        else:
            handoffs = []
            for event in events:
                if event.attempt != attempt or event.event_type != "handoff.recorded":
                    continue
                try:
                    envelope = HandoffEnvelope.from_dict(_thaw_json(event.payload)["handoff_envelope"])
                except (KeyError, TypeError, ValueError):
                    return None
                if envelope.child_attempt == source_attempt:
                    handoffs.append(event)
                elif event.store_seq > terminal_event.store_seq:
                    return None
            if handoffs:
                if len(handoffs) != 1:
                    return None
                anchor = handoffs[0]
            else:
                # The child End can commit before parent ingress writes its
                # first handoff. Older settled children are valid history;
                # receipts from another child after this terminal are not.
                # The original wait marker identifies this child exactly and
                # its terminal receipt fences later parent provider dispatch.
                anchor = terminal_event
        dispatched = {
            TOOL_CATALOG_SNAPSHOT_EVENT_TYPE,
            PROVIDER_WIRE_SNAPSHOT_EVENT_TYPE,
            PROVIDER_TURN_RESULT_EVENT_TYPE,
        }
        if any(
            event.attempt == attempt
            and event.store_seq > anchor.store_seq
            and event.event_type in dispatched
            for event in events
        ):
            return None
        return {**completion, "parent_tool_call": deepcopy(parent_tool_call)}

    def _pending_interaction(self, context, expected_wait=None):
        if self.durability_runtime is None:
            return None
        from unchain.interaction.runtime import DurableInteractionRuntime
        session = self._runtime_session_id(self.store.attempt)
        checkpoint = self.durability_runtime.load_execution_checkpoint(session)
        if not isinstance(checkpoint, dict) or checkpoint.get("status") != "awaiting_interaction":
            return None
        continuation = checkpoint.get("continuation")
        interaction_ref = checkpoint.get("interaction_ref")
        if type(continuation) is not dict or type(interaction_ref) is not dict:
            raise ValueError("Kernel interaction checkpoint is incomplete")
        attempt_id = context.node_attempt_id
        config = self.configs[context.node_id]
        binding = self.bindings[context.node_id]
        interaction_id = interaction_ref.get("interaction_id")
        if (checkpoint.get("session_id") != session
                or checkpoint.get("source_run_id") != attempt_id
                or continuation.get("session_id") != session
                or continuation.get("run_id") != attempt_id
                or continuation.get("interaction_id") != interaction_id
                or checkpoint.get("provider") != binding.transport_provider
                or continuation.get("provider") != binding.transport_provider
                or checkpoint.get("model") != config["profile"]["model"]
                or continuation.get("model") != config["profile"]["model"]):
            raise ValueError("Kernel interaction checkpoint changed node identity")
        pending = DurableInteractionRuntime(self.durability_runtime).load(
            session, interaction_id=interaction_id, require_active=True)
        request = pending.request
        subject = request.subject
        from unchain.interaction.durable import strict_json_digest
        transcript = checkpoint.get("transcript")
        if type(transcript) is not list:
            raise ValueError("Kernel interaction checkpoint has no transcript")
        prompt_manifest = [
            {"role": message.get("role"), "content": deepcopy(message.get("content"))}
            for message in transcript
            if isinstance(message, dict)
            and message.get("role") in {"system", "developer"}
        ]
        active = self.store.load().state["active"]
        if (active is None or active["node_invocation_id"] != context.node_invocation_id
                or active["node_id"] != context.node_id):
            raise ValueError("Kernel interaction active node changed")
        resolved = self.store.read_json(active["input_ref"])
        configured_prompt = _render_prompt(config["prompt"], resolved)
        if configured_prompt and (
            not prompt_manifest
            or prompt_manifest[0] != {"role": "system", "content": configured_prompt}
        ):
            raise ValueError("Kernel interaction checkpoint changed configured prompt")
        prompt_digest = strict_json_digest(prompt_manifest)
        if (pending.checkpoint_id != checkpoint.get("checkpoint_id")
                or request.request_digest != interaction_ref.get("request_digest")
                or request.session_id != session or request.source_run_id != attempt_id
                or type(subject) is not dict
                or subject.get("provider") != binding.transport_provider
                or subject.get("model") != config["profile"]["model"]
                or subject.get("prompt_digest") != prompt_digest):
            raise ValueError("Kernel interaction request changed node identity")
        request_value = request.to_dict()
        if expected_wait is not None:
            if (expected_wait.get("kind") != "interaction"
                    or expected_wait.get("id") != interaction_id
                    or canonical_bytes(self.store.read_json(expected_wait["request_ref"]))
                    != canonical_bytes(request_value)):
                raise ValueError("Kernel interaction does not match the active node")
        return WorkflowNodeWaiting("interaction", interaction_id, request_value)

    def _completed_output(self, context, snapshot):
        from unchain.journal.provider_result import (
            provider_turn_result_event_fields, recover_provider_turn_result,
        )
        from unchain.run_bundle import ProviderCallUsage
        from unchain.run_bundle_ledger import RunBundleLedger
        attempt = AttemptRef(self.store.attempt.generation, context.node_attempt_id)
        node_events = [event for event in snapshot.events if event.attempt == attempt]
        final_events = [event for event in node_events if event.event_type == "final_message"]
        terminal_events = [event for event in node_events if event.event_type == "run_completed"]
        if not final_events and not terminal_events:
            return None
        if len(final_events) != 1 or len(terminal_events) != 1:
            raise ValueError("Kernel terminal receipts are incomplete or ambiguous")
        final_event, terminal_event = final_events[0], terminal_events[0]
        final = _thaw_json(final_event.payload)
        terminal = _thaw_json(terminal_event.payload)
        parent_id = self.store.attempt.attempt_id
        if (set(final) != {"run_id", "iteration", "content", "parent_run_id"}
                or set(terminal) != {"run_id", "iteration", "status", "parent_run_id"}
                or final["run_id"] != attempt.attempt_id
                or terminal["run_id"] != attempt.attempt_id
                or final["parent_run_id"] != parent_id
                or terminal["parent_run_id"] != parent_id
                or terminal["status"] != "completed"
                or type(final["iteration"]) is not int or final["iteration"] < 0
                or terminal["iteration"] != final["iteration"]
                or type(final["content"]) is not str
                or final_event.store_seq >= terminal_event.store_seq):
            raise ValueError("Kernel terminal receipt identity changed")
        provider_events = [
            event for event in node_events
            if event.event_type == "provider.turn_result" and event.store_seq < final_event.store_seq
        ]
        candidates = []
        recovered_turns = []
        for event in provider_events:
            fields = provider_turn_result_event_fields(event)
            if (fields.subject.attempt != attempt
                    or fields.subject.iteration > final["iteration"]):
                raise ValueError("Kernel provider result changed node identity")
            result = recover_provider_turn_result(
                self.store.journal,
                subject=fields.subject,
                expected_route_sha256=fields.route_sha256,
            ).envelope.to_model_turn_result()
            recovered_turns.append((fields, event, result))
            if (fields.subject.iteration == final["iteration"]
                    and fields.visible_output):
                candidates.append((fields, event, result))
        if len(candidates) != 1:
            raise ValueError("Kernel final provider result is missing or ambiguous")
        fields, _, provider_result = candidates[0]
        assistant_contents = [
            message.get("content")
            for _, _, result in recovered_turns
            for message in result.assistant_messages
            if isinstance(message, dict)
            and message.get("role") == "assistant"
        ]
        raw_message = assistant_contents[-1] if assistant_contents else None
        semantic_message = next((
            content.strip()
            for content in reversed(assistant_contents)
            if isinstance(content, str) and content.strip()
        ), "")
        if (provider_result.tool_calls or type(provider_result.final_text) is not str
                or type(raw_message) is not str
                or semantic_message != final["content"]):
            raise ValueError("Kernel final provider result changed terminal content")
        ledger = self.store.journal
        if not isinstance(ledger, RunBundleLedger):
            raise TypeError("Kernel recovery requires the durable RunBundle ledger")
        identity = self._run_identity(context)
        receipts = ledger.load_receipts(
            root_run_id=identity.root_run_id,
            owner_run_id=attempt.attempt_id,
            attempt_id=attempt.attempt_id,
        )
        binding = self.bindings[context.node_id]
        model = self.configs[context.node_id]["profile"]["model"]
        if not receipts or any(
            receipt.identity.execution_id != attempt.generation.execution_id
            or receipt.identity.root_run_id != identity.root_run_id
            or receipt.identity.owner_run_id != attempt.attempt_id
            or receipt.identity.parent_run_id != identity.parent_run_id
            or receipt.identity.attempt_id != attempt.attempt_id
            or receipt.provider != binding.transport_provider
            or receipt.model != model
            for receipt in receipts
        ) or not any(
            receipt.status == "completed"
            and receipt.identity.iteration == fields.subject.iteration
            for receipt in receipts
        ):
            raise ValueError("Kernel provider accounting receipts changed node identity")
        config = self.configs[context.node_id]
        return {"message": raw_message, "context": {
            "provider": config["profile"]["provider"],
            "model": model,
            "profile_id": config["profile"]["id"],
            "usage": ProviderCallUsage.sum(receipt.usage for receipt in receipts).to_dict(),
        }}

    def validate_interaction(self, context, wait, response):
        """Validate the actual checkpoint before acquiring a resume intent."""
        from unchain.interaction.runtime import normalize_interaction_response
        pending = self._pending_interaction(context, expected_wait=wait)
        if pending is None:
            raise ValueError("Kernel interaction checkpoint is absent")
        request = self.store.read_json(wait["request_ref"])
        from unchain.interaction.durable import InteractionRequest
        return normalize_interaction_response(InteractionRequest.from_dict(request), response)

    def resume_interaction(self, context, resolved, *, interaction_id, response, submitted_by):
        return self._with_known_failure(context, resolved, interaction=(interaction_id, response, submitted_by))

    def resume_child(self, context, resolved, *, child_completion=None):
        """Continue the parent after a child finishes at its next provider turn."""
        return self._with_known_failure(
            context,
            resolved,
            child_resume=True,
            child_completion=child_completion,
        )

    def __call__(self, context, resolved):
        return self._with_known_failure(context, resolved)

    def _with_known_failure(self, context, resolved, *, interaction=None, child_resume=False,
                            child_completion=None):
        from unchain.context.compiler import ContextBudgetExceededError
        try:
            return self._execute(
                context,
                resolved,
                interaction=interaction,
                child_resume=child_resume,
                child_completion=child_completion,
            )
        except ContextBudgetExceededError:
            return self._save_completion(context, {"schema": "unchain.kernel_failure.v1", "code": "context_overflow"})
        except SkeletonValidationError as error:
            if error.code not in {"context_overflow", "unsupported_capability"}:
                raise
            return self._save_completion(context, {"schema": "unchain.kernel_failure.v1", "code": error.code})

    def _execute(self, context, resolved, *, interaction=None, child_resume=False,
                 child_completion=None, recovery_mode=False, recovered_tool_turn=None):
        from unchain.agent import Agent
        from unchain.agent.modules.context import ContextModule
        from unchain.agent.modules.tools import ToolsModule
        from unchain.context import ContextRuntime, DurableContextRuntimeFactory, HostResolvedCurrentInput, HostResolvedInteractionInput
        from unchain.context.attachments import normalize_host_resolved_attachments

        context.assert_current()
        self.preflight_inputs(context.node_id, resolved)
        if interaction is None and not recovery_mode:
            saved = self.recover_result(context)
            if saved is not None:
                return saved
        config = self.configs[context.node_id]
        configured_prompt = _render_prompt(config["prompt"], resolved)
        binding = self.bindings[context.node_id]
        attempt = AttemptRef(self.store.attempt.generation, context.node_attempt_id)
        attachments = normalize_host_resolved_attachments(self.attachments_for_node(context, deepcopy(resolved)))
        for attachment in attachments:
            capability = "images" if attachment.media_type.startswith("image/") else "files"
            if not config["profile"]["capabilities"][capability]:
                raise SkeletonValidationError("unsupported_capability", ("kernel", "attachments"), "Kernel profile does not support supplied attachments")
        view = self._view(attempt, config)
        bundle_arguments = dict(attempt=attempt, view=view, config=deepcopy(config))
        compression = None
        if config["overflow"] == "compress":
            from .kernel_compression import kernel_compression_service
            compression = kernel_compression_service(self, context)
            bundle_arguments["compression_service"] = compression
        bundle = self.bundle_builder(**bundle_arguments)
        if bundle.coordinator.compression_service is not compression:
            raise ValueError("Kernel compression capability changed")
        if (bundle.attempt != attempt or bundle.journal is not self.store.journal
                or bundle.request_factory.node_memory_view != view
                or bundle.request_factory.output_reserve_tokens != config["output_tokens"]
                or bundle.provider_turn_service is None):
            raise ValueError("Kernel node capabilities do not match its fixed scope")
        if child_completion is not None:
            self._persist_child_completion_input(
                context,
                bundle,
                child_completion,
            )
        task = json.dumps(resolved, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        def exact_bundle(candidate):
            if candidate != attempt:
                raise ValueError("Kernel runtime changed attempt")
            return bundle
        def current_input(_context, candidate):
            if child_completion is not None:
                return None
            if interaction is None:
                return HostResolvedCurrentInput(candidate, task, attachments=attachments)
            return HostResolvedInteractionInput(candidate, interaction[0], interaction[1], interaction[2])
        factory = DurableContextRuntimeFactory(bundle_builder=exact_bundle,
            generation_resolver=lambda _context, _execution: attempt.generation.generation_id,
            current_input_resolver=current_input,
            # A subagent may use a descendant session for its interaction
            # checkpoint while every graph node remains scoped to the root
            # journal execution that owns its artifacts and provider receipts.
            execution_id_resolver=lambda _context, _session: attempt.generation.execution_id)
        tool_preparation_policy = None
        tool_completion_policy = None
        if self.hook_runtime is not None:
            from .hooks_runtime import (
                SkeletonAfterToolHookDispatcher,
                SkeletonBeforeToolHookDispatcher,
            )

            tool_preparation_policy = SkeletonBeforeToolHookDispatcher(
                hook_runtime=self.hook_runtime,
                kernel_node_id=context.node_id,
                parent_context=context,
                parent_store=self.store,
                run_hook_workflow=self.hook_workflow_runner,
            )
            tool_completion_policy = SkeletonAfterToolHookDispatcher(
                hook_runtime=self.hook_runtime,
                kernel_node_id=context.node_id,
                parent_context=context,
                parent_store=self.store,
                run_hook_workflow=self.hook_workflow_runner,
            )
        runtime = ContextRuntime.from_factory(owner_id="skeleton." + context.node_invocation_id,
            execution_factory=factory, provider_turns_enabled=True,
            tool_output_management_active=True,
            tool_preparation_policy=tool_preparation_policy,
            tool_completion_policy=tool_completion_policy)
        modules = [ContextModule(runtime=runtime), ToolsModule(tools=tuple(self.tools_for_node(context)))]
        extra_modules = tuple(self.modules_for_node(context))
        if any(type(module) in {ContextModule, ToolsModule} for module in extra_modules):
            raise ValueError("Kernel host modules cannot replace Context or Tools")
        modules.extend(extra_modules)
        if self.fyi_channel is not None:
            from unchain.agent.modules.interaction import InteractionModule
            modules.append(InteractionModule(fyi_channel=self.fyi_channel))
        if self.durability_runtime is not None:
            from unchain.agent.modules.durability import DurabilityModule
            modules.append(DurabilityModule(runtime=self.durability_runtime))
        agent = Agent(name=context.node_id, instructions=configured_prompt,
            provider=binding.transport_provider, model=config["profile"]["model"],
            modules=tuple(modules),
            model_io_factory=lambda _spec, _call: binding.model_io)
        def callback(event):
            context.assert_current()
            if event.get("type") == "fyi_injected":
                from unchain.interaction.fyi import FyiMessage, wrap_fyi
                for message in event["messages"]:
                    wrapped = wrap_fyi(FyiMessage(**message))
                    index = 1 + int(sha256(message["message_id"].encode()).hexdigest()[:12], 16)
                    bundle.ingress.persist(HostResolvedCurrentInput(attempt,
                        wrapped["content"], message_index=index))
            if self.callback is not None:
                self.callback({**event, "node_id": context.node_id, "parent_run_id": self.store.attempt.attempt_id})
        runtime_session_id = self._runtime_session_id(attempt)
        next_provider_iteration = self._next_provider_iteration(attempt)
        initial_iteration = (recovered_tool_turn[0] if recovered_tool_turn is not None
                             else next_provider_iteration if child_resume else 0)
        # A tool approval may finish the provider turn that requested it, but
        # must not grant a fresh full iteration budget on every approval.
        max_iterations = config["max_iterations"] + (1 if interaction is not None else 0)
        identity = self._run_identity(context)
        arguments = dict(payload=binding.payload_copy(), response_format=binding.response_format,
            max_iterations=max_iterations, max_context_window_tokens=config["profile"]["context_window"],
            session_id=runtime_session_id, run_id=attempt.attempt_id, callback=callback)
        try:
            if interaction is None:
                result = agent.run(task, **arguments, _execution_guard=self.execution_guard,
                                   _run_bundle_identity=identity,
                                   _initial_iteration=initial_iteration,
                                   _recovered_tool_turn=recovered_tool_turn)
            else:
                from unchain.agent.builder import AgentCallContext
                prepared = agent._prepare(AgentCallContext(mode="resume_interaction", **arguments,
                    response=deepcopy(interaction[1]), submitted_by=interaction[2],
                    interaction_id=interaction[0], execution_guard=self.execution_guard,
                    interaction_execution_fence_overridden=True,
                    resume_max_iterations=max_iterations,
                    run_bundle_identity=identity))
                result = prepared.resume_interaction()
        except Exception as exc:
            from unchain.subagents.types import SubagentWaitSignal
            if not isinstance(exc, SubagentWaitSignal):
                raise
            request = exc.request
            if exc.parent_tool_call is not None:
                request = self._request_with_parent_tool_call(request, exc.parent_tool_call)
            return WorkflowNodeWaiting("child", exc.wait_id, request)
        if result.status == "awaiting_interaction":
            from unchain.interaction.durable import InteractionRequest
            request = InteractionRequest.from_dict(result.interaction_request)
            return WorkflowNodeWaiting("interaction", request.interaction_id, request.to_dict())
        if result.status == "awaiting_human_input":
            from unchain.tools.runtime import decode_subagent_wait_marker_details
            marker = next((decode_subagent_wait_marker_details(m.get("content"))
                           for m in reversed(result.messages)
                           if isinstance(m, dict) and m.get("role") == "assistant"), None)
            if marker is not None:
                request = marker[1]
                if marker[2] is not None:
                    request = self._request_with_parent_tool_call(request, marker[2])
                return WorkflowNodeWaiting("child", marker[0], request)
        if result.status != "completed":
            raise SkeletonValidationError("kernel_incomplete", ("kernel",), "Kernel requires a durable continuation")
        message = next((m.get("content") for m in reversed(result.messages) if m.get("role") == "assistant"), None)
        if type(message) is not str:
            raise ValueError("Kernel has no final text result")
        from unchain.tools.runtime import decode_subagent_wait_marker_details
        child_wait = decode_subagent_wait_marker_details(message)
        if child_wait is not None:
            request = child_wait[1]
            if child_wait[2] is not None:
                request = self._request_with_parent_tool_call(request, child_wait[2])
            return WorkflowNodeWaiting("child", child_wait[0], request)
        public = {"provider": config["profile"]["provider"], "model": config["profile"]["model"],
                  "profile_id": config["profile"]["id"], "usage": None}
        if result.run_bundle is not None:
            from unchain.run_bundle_v2 import run_bundle_from_dict
            parsed = run_bundle_from_dict(result.run_bundle)
            public["usage"] = parsed.aggregation.direct_usage.to_dict()
        return self._save_completion(context, {"message": message, "context": public})

    def _persist_child_completion_input(self, context, bundle, completion):
        """Project one terminal child result as the parent's next durable input."""
        completion_fields = {"schema", "attempt", "status", "outputs", "error"}
        if (type(completion) is not dict
                or (set(completion) != completion_fields
                    and set(completion) != completion_fields | {"parent_tool_call"})
                or completion.get("schema") != "unchain.skeleton_child_completion.v1"):
            raise ValueError("child completion has an invalid shape")
        parent_tool_call = completion.get("parent_tool_call")
        if parent_tool_call is not None and (
            type(parent_tool_call) is not dict
            or set(parent_tool_call) != {"call_id", "tool_name"}
            or any(type(value) is not str or not value.strip()
                   for value in parent_tool_call.values())
        ):
            raise ValueError("child completion parent tool call is invalid")
        source_attempt = AttemptRef.from_dict(completion["attempt"])
        if source_attempt.to_dict() != completion["attempt"]:
            raise ValueError("child completion attempt is non-canonical")
        if source_attempt.generation != bundle.attempt.generation or source_attempt == bundle.attempt:
            raise ValueError("child completion escaped the parent generation")
        status = completion["status"]
        if status not in {"completed", "failed", "cancelled", "outcome_unknown"}:
            raise ValueError("child completion status is invalid")
        if completion["outputs"] is not None and type(completion["outputs"]) is not dict:
            raise ValueError("child completion outputs are invalid")
        if type(completion["error"]) is not str:
            raise ValueError("child completion error is invalid")
        snapshot = self.store.journal.capture_snapshot()
        source_events = [event for event in snapshot.events if event.attempt == source_attempt]
        if not source_events:
            raise ValueError("child completion has no durable source events")
        # Parent and child graph receipts share one journal and can interleave.
        # A derived handoff range must therefore use the child's terminal
        # receipt itself, never the broad first-to-last span.
        terminal_event = source_events[-1]
        self._verify_child_terminal_completion(
            source_attempt=source_attempt,
            terminal_event=terminal_event,
            completion=completion,
        )
        source_range = EventRange(
            start=EventCursor(terminal_event.store_seq, terminal_event.event_id),
            end=EventCursor(terminal_event.store_seq, terminal_event.event_id),
        )
        from unchain.context import (
            DerivedHandoffInputIngress,
            HandoffStatus,
            HostResolvedDerivedHandoffInput,
        )
        output = deepcopy(completion["outputs"])
        summary = (
            output.get("message")
            if type(output) is dict and type(output.get("message")) is str
            else completion["error"] or f"Subagent {status}"
        )
        full_output = {
            "schema": "unchain.skeleton_child_completion.v1",
            "status": status,
            "outputs": output,
            "error": completion["error"],
            "summary": summary,
        }
        ingress = DerivedHandoffInputIngress(
            consumer_attempt=bundle.attempt,
            source_attempt=source_attempt,
            handoff_recorder=bundle.handoff_recorder,
            input_ingress=bundle.ingress,
        )
        operation_id = "skeleton.child-completion." + sha256(canonical_bytes({
            "consumer": bundle.attempt.to_dict(),
            "source": source_attempt.to_dict(),
            "node_invocation_id": context.node_invocation_id,
        })).hexdigest()
        message_index = 1 + int(operation_id.rsplit(".", 1)[-1][:12], 16)
        receipt = ingress.persist(HostResolvedDerivedHandoffInput(
            consumer_attempt=bundle.attempt,
            source_attempt=source_attempt,
            status=(HandoffStatus.COMPLETE if status == "completed" else HandoffStatus.FAILED),
            full_output=full_output,
            source_event_range=source_range,
            operation_id=operation_id,
            summary=summary,
            message_index=message_index,
        ))
        if parent_tool_call is not None:
            self._persist_child_tool_settlement(
                bundle,
                source_attempt=source_attempt,
                completion=completion,
                parent_tool_call=parent_tool_call,
                receipt=receipt,
            )

    def _verify_child_terminal_completion(self, *, source_attempt, terminal_event, completion):
        """Bind a parent handoff to the child's committed End receipt."""
        expected = self._completion_from_child_terminal(
            source_attempt=source_attempt,
            terminal_event=terminal_event,
        )
        if any(completion[key] != expected[key] for key in ("status", "outputs", "error")):
            raise ValueError("child completion contradicts its durable terminal outcome")

    def _completion_from_child_terminal(self, *, source_attempt, terminal_event):
        """Read the one canonical child completion from its terminal state."""
        from .durable_store import _EVENT_TYPE, _PAYLOAD_SCHEMA, _decode_state
        from .durable_protocol import validate_execution_state

        if terminal_event.event_type != _EVENT_TYPE:
            raise ValueError("child completion has no terminal execution receipt")
        try:
            payload = _thaw_json(terminal_event.payload)
            expected_fields = {
                "schema", "graph_key", "revision", "previous_event_id",
                "encoded_state", "state_sha256",
            }
            if set(payload) != expected_fields or payload["schema"] != _PAYLOAD_SCHEMA:
                raise ValueError("child execution receipt changed")
            state = validate_execution_state(_decode_state(
                payload["encoded_state"], payload["state_sha256"], ("child_execution",),
            ))
            state_attempt = AttemptRef.from_dict(state["attempt"])
        except (KeyError, TypeError, ValueError, SkeletonValidationError) as exc:
            raise ValueError("child execution receipt is invalid") from exc
        if state_attempt != source_attempt:
            raise ValueError("child execution receipt escaped its source attempt")
        status = state["status"]
        if status != "completed":
            if status not in {"failed", "cancelled", "outcome_unknown"}:
                raise ValueError("child has no terminal completion")
            return {
                "schema": "unchain.skeleton_child_completion.v1",
                "attempt": source_attempt.to_dict(),
                "status": status,
                "outputs": None,
                "error": "" if state["error"] is None else str(state["error"]),
            }
        if not state["completed"]:
            raise ValueError("completed child receipt has an invalid terminal outcome")
        end = state["completed"][-1]
        if end["edge_id"] is not None:
            raise ValueError("completed child has no committed End receipt")
        try:
            outputs = self.store.read_json(end["output_ref"])
        except Exception as exc:
            raise ValueError("child End output is unavailable") from exc
        return {
            "schema": "unchain.skeleton_child_completion.v1",
            "attempt": source_attempt.to_dict(),
            "status": "completed",
            "outputs": outputs,
            "error": "",
        }

    @staticmethod
    def _persist_child_tool_settlement(
        bundle,
        *,
        source_attempt,
        completion,
        parent_tool_call,
        receipt,
    ):
        """Seal the child outcome against the original parent tool call.

        The waiting result is already an immutable tool receipt.  This second,
        call-bound receipt therefore carries the terminal result instead of
        mutating or replaying the original tool execution.
        """

        full_output = SkeletonKernelHandler._verified_child_settlement_output(
            bundle,
            source_attempt=source_attempt,
            completion=completion,
            receipt=receipt,
        )

        identity = {
            "schema": _CHILD_TOOL_SETTLEMENT_SCHEMA,
            "consumer_attempt": bundle.attempt.to_dict(),
            "source_attempt": source_attempt.to_dict(),
            "call_id": parent_tool_call["call_id"],
        }
        operation_id = "skeleton.subagent-tool-settlement." + sha256(
            canonical_bytes(identity)
        ).hexdigest()
        payload = {
            **identity,
            "tool_name": parent_tool_call["tool_name"],
            "status": completion["status"],
            "outputs": deepcopy(completion["outputs"]),
            "error": completion["error"],
            "summary": full_output["summary"],
            "full_output_ref": receipt.full_output_artifact.ref.to_dict(),
            "result_bytes": receipt.full_output_artifact.byte_length,
            "result_sha256": receipt.full_output_artifact.sha256,
            "handoff_cursor": receipt.handoff_cursor.to_dict(),
        }
        draft = SemanticEventDraft(
            event_id="event." + operation_id,
            event_type=_CHILD_TOOL_SETTLEMENT_EVENT,
            attempt=bundle.attempt,
            operation_id=operation_id,
            payload=payload,
            resource_refs=(receipt.full_output_artifact.ref,),
        )
        request = draft.to_append_request()
        appended = bundle.journal.append(request=request)
        event = appended.event
        if (
            event.event_id != request.event_id
            or event.event_type != request.event_type
            or event.attempt != request.attempt
            or event.operation != request.operation
            or event.payload != request.payload
            or event.resource_refs != request.resource_refs
        ):
            raise ValueError("child tool settlement durable receipt changed")

    @staticmethod
    def _verified_child_settlement_output(bundle, *, source_attempt, completion, receipt):
        """Read the actual handoff before binding it to a parent tool result.

        The caller's completion and receipt are recovery inputs, never authority.
        A settlement may only reuse the exact handoff event already persisted in
        this execution journal.
        """

        from unchain.context import HandoffEnvelope, HandoffStatus
        from unchain.journal import ArtifactRef

        summary = (
            completion["outputs"].get("message")
            if type(completion["outputs"]) is dict
            and type(completion["outputs"].get("message")) is str
            else completion["error"] or f"Subagent {completion['status']}"
        )
        expected_output = {
            "schema": "unchain.skeleton_child_completion.v1",
            "status": completion["status"],
            "outputs": deepcopy(completion["outputs"]),
            "error": completion["error"],
            "summary": summary,
        }
        try:
            cursor = receipt.handoff_cursor
            candidates = [
                event for event in bundle.journal.capture_snapshot().events
                if event.event_type == "handoff.recorded"
                and event.store_seq == cursor.store_seq
                and event.event_id == cursor.event_id
            ]
            if len(candidates) != 1:
                raise ValueError("child settlement has no exact persisted handoff")
            handoff = candidates[0]
            envelope = HandoffEnvelope.from_dict(handoff.payload["handoff_envelope"])
            artifact = ArtifactRef.from_dict(handoff.payload["full_output_artifact"])
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("child settlement handoff receipt is invalid") from exc
        expected_status = (
            HandoffStatus.COMPLETE
            if completion["status"] == "completed"
            else HandoffStatus.FAILED
        )
        expected_refs = tuple(dict.fromkeys((envelope.full_output_ref, *envelope.artifact_refs)))
        if (
            handoff.attempt != bundle.attempt
            or envelope.child_attempt != source_attempt
            or envelope.status != expected_status
            or artifact.ref != envelope.full_output_ref
            or artifact.media_type != "application/json"
            or artifact.byte_length != envelope.byte_length
            or artifact.sha256 != envelope.sha256
            or handoff.resource_refs != expected_refs
            or receipt.full_output_artifact != artifact
        ):
            raise ValueError("child settlement contradicts its persisted handoff")
        try:
            content = bundle.artifacts.read_full(
                artifact,
                remaining_budget_bytes=artifact.byte_length,
            )
            decoded = json.loads(content.decode("utf-8"))
        except Exception as exc:
            raise ValueError("child settlement artifact is unavailable") from exc
        if canonical_bytes(decoded) != content or decoded != expected_output:
            raise ValueError("child settlement outcome contradicts its persisted artifact")
        return expected_output

    def _save_completion(self, context, output):
        ref, pending = self.store.prepare_json(output, key=context.node_invocation_id + ".kernel-result")
        context.assert_current()
        self.store.journal.append_with_artifacts(request=self._request(context, ref), artifacts=(pending,))
        return WorkflowNodeCompleted(output)
