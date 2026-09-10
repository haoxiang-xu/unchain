from __future__ import annotations

import json

import pytest

from unchain.context import (
    BoundContextTaskStateReader,
    ContextCompileRequest,
    ContextCompiler,
    ContextTaskStateReadOutcome,
    PinnedTaskState,
    SourceMessageCursor,
    resolve_context_budget,
)
from unchain.context.long_term_recall_request_factory import (
    LongTermRecallContextRequestFactory,
    LongTermRecallContextRequestFactoryError,
)
from unchain.context.task_state_request_factory import (
    TaskStateContextRequestFactory,
)
from unchain.journal import ResourceRef
from unchain.memory.long_term_recall_v2 import (
    LongTermFirstMessageRecall,
    LongTermRecallDisposition,
    LongTermRecallEnvelope,
    LongTermRecallReference,
)
from unchain.memory.workspace.models import MemoryEntryKind


class _BaseFactory:
    def __init__(self, request):
        self.request = request
        self.attempt = object()
        self.journal = object()

    def __call__(self, context):
        del context
        return self.request


class _Recall(LongTermFirstMessageRecall):
    def __init__(self, envelope, *, binding_id="binding-a"):
        self.envelope = envelope
        self._test_binding_id = binding_id
        self.queries = []

    @property
    def binding_id(self):
        return self._test_binding_id

    def recall_first_message(self, first_user_message, *, limit=5):
        self.queries.append((first_user_message, limit))
        return self.envelope


def _reference(entry_id="entry-a"):
    return LongTermRecallReference(
        entry_ref=ResourceRef("memory", entry_id, 3, "long-term-space"),
        path=f"/shared/{entry_id}.md",
        name=entry_id,
        kind=MemoryEntryKind.MARKDOWN,
        media_type="text/markdown",
        preview="A durable preference reference",
        provenance_refs=(ResourceRef("context_event", "event-a", 1),),
        score=0.98,
        matched_by=("exact_name",),
    )


def _envelope(disposition):
    if disposition is LongTermRecallDisposition.NONE:
        return LongTermRecallEnvelope(
            disposition=disposition,
            namespace="user-a",
        )
    if disposition is LongTermRecallDisposition.CURATOR_REQUIRED:
        return LongTermRecallEnvelope(
            disposition=disposition,
            namespace="user-a",
            references=(_reference(),),
            reason="semantic_key_conflict",
        )
    return LongTermRecallEnvelope(
        disposition=disposition,
        namespace="user-a",
        references=(_reference(),),
    )


def _request(**changes):
    values = {
        "case": "journal-runtime",
        "source_messages": (
            {"role": "system", "content": "current instructions"},
            {"role": "user", "content": "remember my editor preference"},
        ),
        "source_message_cursors": (
            SourceMessageCursor(1, "event-user", 7),
        ),
        "budget": resolve_context_budget(context_window_tokens=8192),
    }
    values.update(changes)
    return ContextCompileRequest(**values)


def _factory(envelope, *, base=None, sink=None):
    base = base or _BaseFactory(_request())
    return LongTermRecallContextRequestFactory(
        binding_id="binding-a",
        base_factory=base,
        recall=_Recall(envelope),
        outcome_sink=sink,
    )


def test_reference_envelope_is_untrusted_budgeted_and_before_current_user():
    original = _request()
    outcomes = []
    factory = _factory(
        _envelope(LongTermRecallDisposition.CONTEXT_REFERENCES),
        base=_BaseFactory(original),
        sink=outcomes.append,
    )

    decorated = factory(object())
    compiled = ContextCompiler().compile(decorated)
    baseline = ContextCompiler().compile(original)

    assert [message["role"] for message in decorated.source_messages] == [
        "system",
        "assistant",
        "user",
    ]
    reference = decorated.source_messages[1]
    current = decorated.source_messages[2]
    assert "MEMORY_V2_UNTRUSTED_LONG_TERM_REFERENCES" in reference["content"]
    assert "ends with this message" in reference["content"]
    payload = json.loads(reference["content"].split("\n", 2)[2])
    assert payload["trusted"] is False
    assert payload["placement"] == "context_reference"
    assert payload["references"][0]["entry_ref"]["kind"] == "memory"
    assert current == original.source_messages[1]
    assert decorated.source_message_cursors == (
        SourceMessageCursor(2, "event-user", 7),
    )
    assert compiled.messages[1] == reference
    assert compiled.messages[2] == current
    assert (
        compiled.diagnostics["before_estimated_tokens"]
        > baseline.diagnostics["before_estimated_tokens"]
    )
    assert outcomes[0].injected is True
    assert outcomes[0].reference_message_index == 1
    assert outcomes[0].current_user_message_index == 2


@pytest.mark.parametrize(
    "disposition",
    (
        LongTermRecallDisposition.NONE,
        LongTermRecallDisposition.CURATOR_REQUIRED,
    ),
)
def test_none_or_curator_required_is_typed_without_direct_injection(disposition):
    original = _request()
    factory = _factory(
        _envelope(disposition),
        base=_BaseFactory(original),
    )

    decorated, outcome = factory.decorate(object())

    assert decorated is original
    assert outcome.injected is False
    assert outcome.reference_message_index is None
    assert outcome.disposition is disposition
    assert outcome.curator_required is (
        disposition is LongTermRecallDisposition.CURATOR_REQUIRED
    )
    assert not any(
        "LONG_TERM_REFERENCES" in str(message.get("content") or "")
        for message in decorated.source_messages
    )


def test_all_explicit_source_cursors_are_shifted_without_forging_a_cursor():
    request = _request(
        source_message_cursors=(
            SourceMessageCursor(0, "event-system", 2),
            SourceMessageCursor(1, "event-user", 7),
        )
    )

    decorated = _factory(
        _envelope(LongTermRecallDisposition.CONTEXT_REFERENCES),
        base=_BaseFactory(request),
    )(object())

    assert decorated.source_message_cursors == (
        SourceMessageCursor(0, "event-system", 2),
        SourceMessageCursor(2, "event-user", 7),
    )
    assert all(
        cursor.message_index != 1
        for cursor in decorated.source_message_cursors
    )


class _TaskStateReader(BoundContextTaskStateReader):
    def __init__(self):
        super().__init__("binding-a")

    def read_for_context(self):
        return ContextTaskStateReadOutcome.from_state(
            PinnedTaskState(
                state_id="task-a",
                revision=1,
                objective="preserve composed decorators",
            )
        )


def test_task_state_factory_composes_inside_or_outside_recall_decorator():
    base = _BaseFactory(_request())
    task_inner = TaskStateContextRequestFactory(
        binding_id="binding-a",
        base_factory=base,
        reader=_TaskStateReader(),
    )
    recall_outer = _factory(
        _envelope(LongTermRecallDisposition.CONTEXT_REFERENCES),
        base=task_inner,
    )
    outer_request = recall_outer(object())

    outcomes = []
    recall_inner = _factory(
        _envelope(LongTermRecallDisposition.CONTEXT_REFERENCES),
        base=base,
        sink=outcomes.append,
    )
    task_outer = TaskStateContextRequestFactory(
        binding_id="binding-a",
        base_factory=recall_inner,
        reader=_TaskStateReader(),
    )
    inner_request = task_outer(object())

    assert outer_request.task_state is not None
    assert inner_request.task_state is not None
    assert outer_request.source_messages == inner_request.source_messages
    assert outer_request.source_message_cursors == inner_request.source_message_cursors
    assert recall_outer.attempt is base.attempt
    assert recall_outer.journal is base.journal
    assert outcomes[0].injected is True


def test_decoration_is_restart_deterministic_and_binding_scoped():
    envelope = _envelope(LongTermRecallDisposition.CONTEXT_REFERENCES)
    before, before_outcome = _factory(envelope).decorate(object())
    reopened, reopened_outcome = _factory(envelope).decorate(object())

    assert before.to_dict() == reopened.to_dict()
    assert before_outcome == reopened_outcome

    with pytest.raises(
        LongTermRecallContextRequestFactoryError,
        match="another binding",
    ):
        LongTermRecallContextRequestFactory(
            binding_id="binding-a",
            base_factory=_BaseFactory(_request()),
            recall=_Recall(envelope, binding_id="binding-b"),
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "official context-reference injection slot requires CRITICAL compiler review"
    ),
)
def test_red_semantic_projection_must_keep_recall_before_current_user():
    """RED: canonical projection currently moves cursorless recall to the tail."""

    request = _request(
        semantic_events=(
            {
                "type": "message.user",
                "event_id": "event-user",
                "store_seq": 7,
                "attempt_id": "attempt-current",
                "run_id": "attempt-current",
                "message": {
                    "role": "user",
                    "content": "remember my editor preference",
                },
            },
        ),
    )
    decorated = _factory(
        _envelope(LongTermRecallDisposition.CONTEXT_REFERENCES),
        base=_BaseFactory(request),
    )(object())

    compiled = ContextCompiler().compile(decorated)
    reference_index = next(
        index
        for index, message in enumerate(compiled.messages)
        if "MEMORY_V2_UNTRUSTED_LONG_TERM_REFERENCES"
        in str(message.get("content") or "")
    )
    current_user_index = next(
        index
        for index, message in enumerate(compiled.messages)
        if message.get("content") == "remember my editor preference"
    )

    assert reference_index < current_user_index
