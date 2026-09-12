from __future__ import annotations

import hashlib
import json

import pytest

from unchain.context import (
    ArtifactService,
    DurableToolApprovalState,
    DurableToolExecutionSubject,
    DurableToolRouteKind,
)
from unchain.execution import ExecutionFence
from unchain.tools.output_management import ToolOutputManager
from unchain.context.projector import (
    CanonicalSemanticEventProjector,
    SemanticEventProjectionError,
)
from unchain.context.ports import BoundArtifactRepository
from unchain.journal import (
    ArtifactRef,
    AttemptRef,
    BoundExecutionJournal,
    DurableEventSink,
    EventCursor,
    GenerationRef,
    JournalAppendResult,
    JournalEvent,
    JournalPage,
    ResourceRef,
    capture_journal_snapshot,
)


def _tool_subject() -> DurableToolExecutionSubject:
    return DurableToolExecutionSubject(
        intent_cursor=EventCursor(1, "event-intent"),
        original_arguments_sha256="1" * 64,
        effective_arguments_sha256="2" * 64,
        approval_state=DurableToolApprovalState.NOT_REQUIRED,
        approval_request_sha256="",
        approval_receipt_sha256="",
        route_kind=DurableToolRouteKind.NORMAL,
        route_manifest_sha256="3" * 64,
        terminal_handler_manifest_sha256="4" * 64,
        execution_fence=ExecutionFence("execution-1", "owner-1", 1),
    )


class _ArtifactRepository(BoundArtifactRepository):
    def __init__(self, order: list[str]) -> None:
        super().__init__("execution-1")
        self.order = order
        self.by_operation = {}
        self.content = {}

    def put(self, *, content, media_type, operation, preview=""):
        self.order.append("artifact.put")
        previous = self.by_operation.get(operation.operation_id)
        if previous is not None:
            prior_operation, artifact = previous
            if prior_operation != operation:
                raise RuntimeError("artifact operation conflict")
            return artifact
        digest = hashlib.sha256(content).hexdigest()
        artifact = ArtifactRef(
            ref=ResourceRef("artifact", f"object-{operation.operation_id}", 1),
            media_type=media_type,
            byte_length=len(content),
            sha256=digest,
            preview=preview,
        )
        self.by_operation[operation.operation_id] = (operation, artifact)
        self.content[artifact.ref.resource_id] = content
        return artifact

    def artifact_id_for(self, *, logical_kind, logical_key):
        return f"object-{logical_key}"

    def read_verified(self, *, artifact, offset=0, limit=65_536):
        return self.content[artifact.ref.resource_id][offset : offset + limit]

    def read_full_verified(self, *, artifact):
        return self.content[artifact.ref.resource_id]


class _Journal(BoundExecutionJournal):
    def __init__(self, order: list[str]) -> None:
        super().__init__("execution-1")
        self.order = order
        self.events = []
        self.operations = {}

    def append(self, *, request):
        self.order.append("journal.append")
        previous = self.operations.get(request.operation.operation_id)
        if previous is not None:
            prior_request, event = previous
            if prior_request != request:
                raise RuntimeError("journal operation conflict")
            return JournalAppendResult(
                cursor=event_cursor(event),
                event=event,
                duplicate=True,
            )
        event = JournalEvent(
            event_id=request.event_id,
            event_type=request.event_type,
            attempt=request.attempt,
            operation=request.operation,
            store_seq=len(self.events) + 1,
            payload=request.payload,
            resource_refs=request.resource_refs,
        )
        self.events.append(event)
        self.operations[request.operation.operation_id] = (request, event)
        return JournalAppendResult(
            cursor=event_cursor(event),
            event=event,
        )

    def read(self, *, after=None, limit=100):
        start = after.store_seq if after is not None else 0
        selected = tuple(self.events[start : start + limit])
        return JournalPage(events=selected, has_more=False)

    def capture_snapshot(self, *, max_events=10_000, max_bytes=32 * 1024 * 1024):
        del max_events, max_bytes
        return capture_journal_snapshot(
            execution_id=self.execution_id,
            events=tuple(self.events),
        )


def event_cursor(event):
    from unchain.journal import EventCursor

    return EventCursor(store_seq=event.store_seq, event_id=event.event_id)


def _attempt() -> AttemptRef:
    return AttemptRef(
        generation=GenerationRef("execution-1", "generation-1"),
        attempt_id="run-1",
    )


def _projector(
    order,
    *,
    content_sanitizer=lambda content, media_type: content,
    payload_sanitizer=lambda event_type, payload: payload,
):
    artifacts = ArtifactService(
        _ArtifactRepository(order),
        sanitizer=content_sanitizer,
    )
    return CanonicalSemanticEventProjector(
        attempt=_attempt(),
        artifacts=artifacts,
        payload_sanitizer=payload_sanitizer,
    )


def test_tool_result_is_sanitized_and_artifactized_before_journal_append() -> None:
    order = []

    def redact(content: bytes, media_type: str) -> bytes:
        assert media_type == "application/json"
        return content.replace(b"secret-value", b"[REDACTED]")

    journal = _Journal(order)
    sink = DurableEventSink(
        journal,
        _attempt(),
        _projector(order, content_sanitizer=redact),
    )

    subject = _tool_subject()
    result = sink(
        {
            "type": "tool_result",
            "run_id": "run-1",
            "iteration": 0,
            "tool_name": "lookup",
            "call_id": "call-1",
            "execution_subject": subject.to_dict(),
            "execution_subject_sha256": subject.sha256,
            "result": {"value": "secret-value"},
        }
    )

    assert order == ["artifact.put", "journal.append"]
    assert result is not None
    event = result.event
    assert event.event_type == "tool_result"
    assert event.payload["result"] == {"value": "[REDACTED]"}
    assert event.payload["full_output_ref"] == event.resource_refs[0].to_dict()
    assert event.payload["result_bytes"] > 0
    assert len(event.payload["result_sha256"]) == 64
    assert "secret-value" not in str(event.to_dict())


def test_explicit_tool_result_policy_uses_bound_output_manager() -> None:
    """An explicit route policy is projected by the attempt-bound manager."""

    order = []
    projector = _projector(order)
    manager = ToolOutputManager.active_default(attempt_id="run-1")
    projector.bind_tool_output_manager(manager)
    journal = _Journal(order)
    sink = DurableEventSink(journal, _attempt(), projector)
    subject = _tool_subject()

    result = sink(
        {
            "type": "tool_result",
            "run_id": "run-1",
            "iteration": 0,
            "tool_name": "lookup",
            "call_id": "call-explicit-policy",
            "tool_result_policy": "artifact_only",
            "execution_subject": subject.to_dict(),
            "execution_subject_sha256": subject.sha256,
            "result": {"large": "payload"},
        }
    )

    assert result is not None
    payload = result.event.payload
    assert payload["result"]["projection"] == "artifact_only"
    assert payload["result_projection"]["projection_policy"] == "artifact_only"
    assert payload["full_output_ref"] == result.event.resource_refs[0].to_dict()


def test_explicit_tool_result_policy_requires_attempt_bound_manager() -> None:
    projector = _projector([])

    with pytest.raises(
        SemanticEventProjectionError,
        match="attempt-bound output manager",
    ):
        projector._bound_tool_output_manager()


def test_prepared_tool_result_keeps_sealed_result_and_records_model_projection() -> None:
    order = []
    projector = _projector(order)
    projector.bind_tool_output_manager(
        ToolOutputManager.active_default(
            attempt_id="run-1",
            preview_chars=4,
            inline_chars=1,
        )
    )
    artifactization = projector.artifacts.artifactize_tool_result(
        {"value": "durable tool result"},
        operation_id="prepared-tool-result",
    )
    completion = projector.artifacts.persist_exact_json(
        {"completion": "sealed"},
        operation_id="prepared-tool-completion",
    )
    subject = _tool_subject()

    draft = projector.project_prepared_tool_result(
        {
            "type": "tool_result",
            "run_id": "run-1",
            "iteration": 0,
            "tool_name": "lookup",
            "call_id": "call-prepared-projection",
            "tool_result_policy": "artifact_only",
            "execution_subject": subject.to_dict(),
            "execution_subject_sha256": subject.sha256,
        },
        artifactization=artifactization,
        completion_artifact=completion,
    )

    assert draft.payload["result"] == artifactization.event_fields()["result"]
    assert draft.payload["model_projection"]["result"]["projection"] == "artifact_only"


def test_user_message_artifact_sanitizer_preserves_only_its_provenance_lane() -> None:
    order = []
    handle = "pvh1_" + ("a" * 64)
    marker = f'<secret-handle label="API key" handle="{handle}"/>'

    def redact_handle(content: bytes, media_type: str) -> bytes:
        assert media_type == "application/json"
        return content.replace(handle.encode("utf-8"), b"[VAULT_HANDLE]")

    repository = _ArtifactRepository(order)
    artifacts = ArtifactService(
        repository,
        sanitizer=redact_handle,
        user_message_sanitizer=lambda content, media_type: content,
    )
    projector = CanonicalSemanticEventProjector(
        attempt=_attempt(),
        artifacts=artifacts,
        payload_sanitizer=lambda event_type, payload: payload,
    )

    draft = projector.project_user_message(
        {"role": "user", "content": f"Use {marker}"}
    )
    user_artifact = repository.content[
        draft.payload["content_ref"]["id"]
    ]
    tool_result = artifacts.artifactize_tool_result(
        {"value": handle},
        operation_id="tool-result-redacts-handle",
    )

    assert draft.payload["message"]["content"] == f"Use {marker}"
    assert handle.encode("utf-8") in user_artifact
    assert handle not in str(tool_result.visible_result)
    assert "[VAULT_HANDLE]" in str(tool_result.visible_result)

    default_repository = _ArtifactRepository([])
    default_artifacts = ArtifactService(
        default_repository,
        sanitizer=redact_handle,
    )
    _artifact, default_message, _content = (
        default_artifacts.artifactize_user_message(
            {"role": "user", "content": marker},
            operation_id="user-message-default-sanitizer",
        )
    )
    assert handle not in default_message["content"]
    assert "[VAULT_HANDLE]" in default_message["content"]


def test_tool_call_is_a_stable_pre_execution_receipt() -> None:
    order = []
    projector = _projector(order)
    raw = {
        "type": "tool_call",
        "run_id": "run-1",
        "iteration": 2,
        "tool_name": "write_file",
        "call_id": "call-2",
        "arguments": {"path": "notes.md"},
    }

    first = projector(raw)
    second = projector(raw)

    assert first == second
    assert first is not None
    assert first.event_type == "tool_call"
    assert first.payload["tool_name"] == "write_file"
    assert first.payload["call_id"] == "call-2"
    assert first.payload["arguments"] == {"path": "notes.md"}
    assert first.resource_refs == ()


@pytest.mark.parametrize(
    ("raw_provider", "expected_provider"),
    [
        ("openai", "openai"),
        (" Anthropic ", "anthropic"),
        ("HYPERSPACE", "hyperspace"),
        ("ollama", "ollama"),
    ],
)
def test_tool_call_persists_a_normalized_source_provider(
    raw_provider,
    expected_provider,
) -> None:
    draft = _projector([])(
        {
            "type": "tool_call",
            "run_id": "run-1",
            "iteration": 0,
            "tool_name": "lookup",
            "call_id": "call-provider",
            "arguments": {"query": "safe"},
            "source_provider": raw_provider,
        }
    )

    assert draft is not None
    assert draft.payload["source_provider"] == expected_provider


@pytest.mark.parametrize("source_provider", ["unsupported-provider", "", 7, object()])
def test_tool_call_rejects_an_unknown_source_provider(source_provider) -> None:
    with pytest.raises(SemanticEventProjectionError, match="source provider"):
        _projector([])(
            {
                "type": "tool_call",
                "run_id": "run-1",
                "iteration": 0,
                "tool_name": "lookup",
                "call_id": "call-provider",
                "arguments": {},
                "source_provider": source_provider,
            }
        )


def test_tool_call_source_provider_is_protected_from_payload_sanitizer() -> None:
    def rewrite_provider(event_type, payload):
        assert event_type == "tool_call"
        payload["source_provider"] = "ollama"
        return payload

    with pytest.raises(
        SemanticEventProjectionError,
        match="protected source_provider",
    ):
        _projector([], payload_sanitizer=rewrite_provider)(
            {
                "type": "tool_call",
                "run_id": "run-1",
                "iteration": 0,
                "tool_name": "lookup",
                "call_id": "call-provider",
                "arguments": {},
                "source_provider": "openai",
            }
        )


@pytest.mark.parametrize(
    "raw",
    [
        {
            "type": "tool_call",
            "run_id": "run-1",
            "iteration": 0,
            "tool_name": "shell",
        },
        {
            "type": "tool_result",
            "run_id": "run-1",
            "iteration": 0,
            "call_id": "call-1",
            "result": {},
        },
        {
            "type": "tool_result",
            "run_id": "foreign-run",
            "iteration": 0,
            "tool_name": "shell",
            "call_id": "call-1",
            "result": {},
        },
    ],
)
def test_tool_boundaries_fail_closed_without_exact_identity(raw) -> None:
    with pytest.raises(SemanticEventProjectionError):
        _projector([])(raw)


def test_tool_boundary_rejects_digest_valid_but_untyped_execution_subject() -> None:
    forged_subject = {"schema": "forged.subject.v1", "value": "safe"}
    forged_digest = hashlib.sha256(
        json.dumps(
            forged_subject,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(SemanticEventProjectionError, match="schema"):
        _projector([])(
            {
                "type": "tool_result",
                "run_id": "run-1",
                "iteration": 0,
                "tool_name": "shell",
                "call_id": "call-1",
                "execution_subject": forged_subject,
                "execution_subject_sha256": forged_digest,
                "result": {"ok": True},
            }
        )


def test_user_message_projection_persists_sanitized_content_and_receipt() -> None:
    order = []
    projector = _projector(order)

    draft = projector.project_user_message(
        {"role": "user", "content": "remember this"},
        message_index=3,
    )

    assert order == ["artifact.put"]
    assert draft.event_type == "message.user"
    assert draft.payload["message"] == {
        "role": "user",
        "content": "remember this",
    }


def test_interaction_request_is_projected_with_canonical_causal_identity() -> None:
    draft = _projector([])(
        {
            "type": "interaction_requested",
            "run_id": "run-1",
            "iteration": 2,
            "interaction_request": {
                "interaction_id": "interaction-human-1",
                "kind": "human_input",
                "question": "Choose one",
            },
        }
    )

    assert draft is not None
    assert draft.event_type == "interaction.requested"
    assert draft.payload["interaction_id"] == "interaction-human-1"
    assert draft.payload["interaction_request"]["question"] == "Choose one"


def test_durable_human_input_request_uses_the_canonical_interaction_event() -> None:
    draft = _projector([])(
        {
            "type": "human_input_requested",
            "run_id": "run-1",
            "iteration": 2,
            "interaction_id": "interaction-human-1",
            "interaction_request": {
                "interaction_id": "interaction-human-1",
                "kind": "human_input",
                "question": "Choose one",
            },
            "request_id": "call-human-1",
        }
    )

    assert draft is not None
    assert draft.event_type == "interaction.requested"
    assert draft.payload["interaction_id"] == "interaction-human-1"


def test_interaction_resume_replay_is_not_a_second_tool_side_effect() -> None:
    order: list[str] = []

    draft = _projector(order)(
        {
            "type": "tool_result",
            "run_id": "run-1",
            "iteration": 2,
            "interaction_id": "interaction-human-1",
            "interaction_resume_replay": True,
            "tool_name": "ask_user_question",
            "call_id": "call-human-1",
            "result": {"selected_values": ["a"]},
        }
    )

    assert draft is None
    assert order == []


def test_interaction_resume_replay_requires_its_durable_identity() -> None:
    with pytest.raises(
        SemanticEventProjectionError,
        match="interaction resume replay identity",
    ):
        _projector([])(
            {
                "type": "tool_result",
                "run_id": "run-1",
                "iteration": 2,
                "interaction_resume_replay": True,
                "tool_name": "ask_user_question",
                "call_id": "call-human-1",
                "result": {"selected_values": ["a"]},
            }
        )


def test_interaction_resolution_is_sanitized_and_artifactized_before_append() -> None:
    order: list[str] = []

    def redact(content: bytes, media_type: str) -> bytes:
        assert media_type == "application/json"
        return content.replace(b"secret-choice", b"[REDACTED]")

    projector = _projector(order, content_sanitizer=redact)
    journal = _Journal(order)
    sink = DurableEventSink(journal, _attempt(), projector)

    draft = projector.project_interaction_resolution(
        interaction_id="interaction-human-1",
        response={"selected": "secret-choice"},
        submitted_by="ui:test",
    )
    receipt = sink.append_projected(draft)

    assert order == ["artifact.put", "journal.append"]
    assert receipt.event.event_type == "interaction.resolved"
    assert receipt.event.payload["interaction_id"] == "interaction-human-1"
    assert receipt.event.payload["submitted_by"] == "ui:test"
    assert receipt.event.payload["content_ref"] == (
        receipt.event.resource_refs[0].to_dict()
    )
    assert receipt.event.payload["content_bytes"] > 0
    assert len(receipt.event.payload["content_sha256"]) == 64
    assert "secret-choice" not in str(receipt.event.to_dict())
    artifact = ArtifactRef(
        ref=receipt.event.resource_refs[0],
        media_type="application/json",
        byte_length=receipt.event.payload["content_bytes"],
        sha256=receipt.event.payload["content_sha256"],
        preview=receipt.event.payload["preview"],
    )
    stored = projector.artifacts.read_full(
        artifact,
        remaining_budget_bytes=receipt.event.payload["content_bytes"],
    )
    assert json.loads(stored) == {
        "interaction_id": "interaction-human-1",
        "response": {"selected": "[REDACTED]"},
        "submitted_by": "ui:test",
    }
    assert draft.payload["content_ref"] == draft.resource_refs[0].to_dict()
    assert draft.payload["content_bytes"] > 0
    assert len(draft.payload["content_sha256"]) == 64


def test_ephemeral_reasoning_and_token_deltas_are_not_projected() -> None:
    projector = _projector([])

    assert projector({"type": "reasoning", "run_id": "run-1"}) is None
    assert projector({"type": "token_delta", "run_id": "run-1"}) is None
