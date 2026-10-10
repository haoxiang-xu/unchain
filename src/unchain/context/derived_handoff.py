from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from unchain.journal import (
    ArtifactRef,
    AttemptRef,
    EventCursor,
    EventRange,
    JournalAppendResult,
    ResourceRef,
    SemanticEventDraft,
)
from unchain.journal.models import (
    _freeze_json,
    _record_tuple,
    _required_text,
    _thaw_json,
)

from .attachments import HostResolvedAttachment
from .handoff import DurableHandoffRecorder, PersistedHandoff
from .ingress import ContextInputIngress, HostResolvedCurrentInput
from .models import HandoffEnvelope, HandoffStatus


class DerivedHandoffInputError(RuntimeError):
    """A durable predecessor handoff could not authorize one derived input."""


@dataclass(frozen=True)
class HostResolvedDerivedHandoffInput:
    """Trusted-host predecessor output with no free-form current-input field."""

    consumer_attempt: AttemptRef
    source_attempt: AttemptRef
    status: HandoffStatus
    full_output: Any
    source_event_range: EventRange
    operation_id: str
    artifact_refs: tuple[ResourceRef, ...] = ()
    summary: Any = None
    message_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.consumer_attempt, AttemptRef):
            object.__setattr__(
                self,
                "consumer_attempt",
                AttemptRef.from_dict(self.consumer_attempt),
            )
        if not isinstance(self.source_attempt, AttemptRef):
            object.__setattr__(
                self,
                "source_attempt",
                AttemptRef.from_dict(self.source_attempt),
            )
        if self.consumer_attempt == self.source_attempt:
            raise ValueError("derived handoff source and consumer must be distinct")
        if self.consumer_attempt.generation != self.source_attempt.generation:
            raise ValueError(
                "derived handoff source and consumer must share one generation"
            )
        try:
            object.__setattr__(self, "status", HandoffStatus(self.status))
        except ValueError as exc:
            raise ValueError("derived handoff status is invalid") from exc
        if not isinstance(self.source_event_range, EventRange):
            object.__setattr__(
                self,
                "source_event_range",
                EventRange.from_dict(self.source_event_range),
            )
        object.__setattr__(
            self,
            "operation_id",
            _required_text(
                self.operation_id,
                "operation_id",
                identifier=True,
            ),
        )
        refs = _record_tuple(self.artifact_refs, ResourceRef, "artifact_refs")
        if (
            len(set(refs)) != len(refs)
            or any(ref.kind != "artifact" or ref.fragment for ref in refs)
        ):
            raise ValueError(
                "artifact_refs must be distinct whole artifact references"
            )
        object.__setattr__(self, "artifact_refs", refs)
        object.__setattr__(
            self,
            "full_output",
            _freeze_json(self.full_output, path="full_output"),
        )
        if self.summary is not None:
            object.__setattr__(
                self,
                "summary",
                _freeze_json(self.summary, path="summary"),
            )
        if (type(self.message_index) is not int or self.message_index < 0):
            raise ValueError("derived handoff message_index is invalid")


@dataclass(frozen=True)
class DurableDerivedHandoffInputReceipt:
    """The exact handoff descriptor and the two ordered journal receipts."""

    envelope: HandoffEnvelope
    full_output_artifact: ArtifactRef
    handoff_cursor: EventCursor
    input_cursor: EventCursor
    handoff_duplicate: bool = False
    input_duplicate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, HandoffEnvelope):
            object.__setattr__(
                self,
                "envelope",
                HandoffEnvelope.from_dict(self.envelope),
            )
        if not isinstance(self.full_output_artifact, ArtifactRef):
            object.__setattr__(
                self,
                "full_output_artifact",
                ArtifactRef.from_dict(self.full_output_artifact),
            )
        if not isinstance(self.handoff_cursor, EventCursor):
            object.__setattr__(
                self,
                "handoff_cursor",
                EventCursor.from_dict(self.handoff_cursor),
            )
        if not isinstance(self.input_cursor, EventCursor):
            object.__setattr__(
                self,
                "input_cursor",
                EventCursor.from_dict(self.input_cursor),
            )
        if (
            self.full_output_artifact.ref != self.envelope.full_output_ref
            or self.full_output_artifact.media_type != "application/json"
            or self.full_output_artifact.byte_length != self.envelope.byte_length
            or self.full_output_artifact.sha256 != self.envelope.sha256
        ):
            raise ValueError(
                "derived handoff envelope changed its full output descriptor"
            )
        if self.handoff_cursor.store_seq >= self.input_cursor.store_seq:
            raise ValueError("derived input receipt must follow its handoff receipt")
        if not isinstance(self.handoff_duplicate, bool) or not isinstance(
            self.input_duplicate,
            bool,
        ):
            raise TypeError("duplicate flags must be booleans")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _exact_range_events(
    journal,
    *,
    source_attempt: AttemptRef,
    source_event_range: EventRange,
) -> tuple[Any, ...]:
    snapshot = journal.capture_snapshot()
    if snapshot.execution_id != source_attempt.generation.execution_id:
        raise DerivedHandoffInputError(
            "source range snapshot escaped the bound execution"
        )
    start = source_event_range.start
    end = source_event_range.end
    selected = tuple(
        event
        for event in snapshot.events
        if start.store_seq <= event.store_seq <= end.store_seq
    )
    if (
        not selected
        or selected[0].store_seq != start.store_seq
        or selected[0].event_id != start.event_id
        or selected[-1].store_seq != end.store_seq
        or selected[-1].event_id != end.event_id
        or tuple(event.store_seq for event in selected)
        != tuple(range(start.store_seq, end.store_seq + 1))
    ):
        raise DerivedHandoffInputError(
            "source event range is not one complete durable journal range"
        )
    if any(event.attempt != source_attempt for event in selected):
        raise DerivedHandoffInputError(
            "source event range contains a foreign attempt"
        )
    return selected


def _descriptor_bound_handoff_draft(
    recorder: DurableHandoffRecorder,
    persisted: PersistedHandoff,
) -> SemanticEventDraft:
    draft = recorder.projector.project_handoff_envelope(persisted.envelope)
    payload = _thaw_json(draft.payload)
    payload["full_output_artifact"] = persisted.full_output_artifact.to_dict()
    return SemanticEventDraft(
        event_id=draft.event_id,
        event_type=draft.event_type,
        attempt=draft.attempt,
        operation_id=draft.operation_id,
        payload=payload,
        resource_refs=draft.resource_refs,
    )


def _verified_handoff_append(
    appended: JournalAppendResult,
    *,
    consumer_attempt: AttemptRef,
    persisted: PersistedHandoff,
) -> None:
    event = appended.event
    try:
        envelope = HandoffEnvelope.from_dict(event.payload["handoff_envelope"])
        artifact = ArtifactRef.from_dict(event.payload["full_output_artifact"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DerivedHandoffInputError(
            "durable handoff receipt has no exact output descriptor"
        ) from exc
    expected_refs = tuple(
        dict.fromkeys(
            (
                persisted.envelope.full_output_ref,
                *persisted.envelope.artifact_refs,
            )
        )
    )
    if (
        event.attempt != consumer_attempt
        or event.event_type != "handoff.recorded"
        or appended.cursor.store_seq != event.store_seq
        or appended.cursor.event_id != event.event_id
        or envelope != persisted.envelope
        or artifact != persisted.full_output_artifact
        or event.resource_refs != expected_refs
    ):
        raise DerivedHandoffInputError(
            "durable handoff append failed exact verification"
        )


def _verified_input_append(
    appended: JournalAppendResult,
    *,
    consumer_attempt: AttemptRef,
    content: str,
    attachment: HostResolvedAttachment,
    handoff_cursor: EventCursor,
) -> None:
    event = appended.event
    message = _thaw_json(event.payload.get("message"))
    try:
        content_ref = ResourceRef.from_dict(event.payload["content_ref"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DerivedHandoffInputError(
            "derived input receipt has no canonical content reference"
        ) from exc
    if (
        event.attempt != consumer_attempt
        or event.event_type != "message.user"
        or appended.cursor.store_seq != event.store_seq
        or appended.cursor.event_id != event.event_id
        or appended.cursor.store_seq <= handoff_cursor.store_seq
        or message
        != {
            "role": "user",
            "content": content,
            "attachments": [attachment.to_dict()],
        }
        or event.resource_refs != (content_ref, attachment.artifact.ref)
    ):
        raise DerivedHandoffInputError(
            "derived input append failed exact verification"
        )


class DerivedHandoffInputIngress:
    """Turn a verified predecessor range into one consumer input receipt.

    This service is a trusted-host boundary. It accepts no arbitrary current
    user text: the model-visible message is deterministically derived from the
    exact durable handoff envelope and descriptor.
    """

    def __init__(
        self,
        *,
        consumer_attempt: AttemptRef,
        source_attempt: AttemptRef,
        handoff_recorder: DurableHandoffRecorder,
        input_ingress: ContextInputIngress,
    ) -> None:
        if not isinstance(consumer_attempt, AttemptRef):
            consumer_attempt = AttemptRef.from_dict(consumer_attempt)
        if not isinstance(source_attempt, AttemptRef):
            source_attempt = AttemptRef.from_dict(source_attempt)
        if consumer_attempt == source_attempt:
            raise DerivedHandoffInputError(
                "derived handoff source and consumer must be distinct"
            )
        if consumer_attempt.generation != source_attempt.generation:
            raise DerivedHandoffInputError(
                "derived handoff source and consumer must share one generation"
            )
        if type(handoff_recorder) is not DurableHandoffRecorder:
            raise TypeError(
                "handoff_recorder must be the official DurableHandoffRecorder"
            )
        if type(input_ingress) is not ContextInputIngress:
            raise TypeError("input_ingress must be the official ContextInputIngress")
        if (
            handoff_recorder.attempt != consumer_attempt
            or input_ingress.attempt != consumer_attempt
        ):
            raise DerivedHandoffInputError(
                "derived handoff components do not share the consumer attempt"
            )
        if (
            handoff_recorder.projector is not input_ingress.projector
            or handoff_recorder.sink is not input_ingress.sink
        ):
            raise DerivedHandoffInputError(
                "derived handoff components do not share one durable boundary"
            )
        if (
            handoff_recorder.handoffs.execution_id
            != consumer_attempt.generation.execution_id
        ):
            raise DerivedHandoffInputError(
                "derived handoff service escaped the consumer execution"
            )
        self._consumer_attempt = consumer_attempt
        self._source_attempt = source_attempt
        self._handoff_recorder = handoff_recorder
        self._input_ingress = input_ingress

    @property
    def consumer_attempt(self) -> AttemptRef:
        return self._consumer_attempt

    @property
    def source_attempt(self) -> AttemptRef:
        return self._source_attempt

    @property
    def handoff_recorder(self) -> DurableHandoffRecorder:
        return self._handoff_recorder

    @property
    def input_ingress(self) -> ContextInputIngress:
        return self._input_ingress

    def persist(
        self,
        current_input: HostResolvedDerivedHandoffInput,
    ) -> DurableDerivedHandoffInputReceipt:
        if type(current_input) is not HostResolvedDerivedHandoffInput:
            raise TypeError(
                "current_input must be an exact HostResolvedDerivedHandoffInput"
            )
        if (
            current_input.consumer_attempt != self._consumer_attempt
            or current_input.source_attempt != self._source_attempt
        ):
            raise DerivedHandoffInputError(
                "derived handoff input changed its bound attempts"
            )
        source_events = _exact_range_events(
            self._input_ingress.sink.journal,
            source_attempt=self._source_attempt,
            source_event_range=current_input.source_event_range,
        )
        source_refs = {
            ref for event in source_events for ref in event.resource_refs
        }
        if any(ref not in source_refs for ref in current_input.artifact_refs):
            raise DerivedHandoffInputError(
                "derived handoff artifact reference is outside the source range"
            )

        persisted = self._handoff_recorder.handoffs.persist_artifactized(
            child_attempt=self._source_attempt,
            status=current_input.status,
            full_output=_thaw_json(current_input.full_output),
            source_event_range=current_input.source_event_range,
            artifact_refs=current_input.artifact_refs,
            operation_id=current_input.operation_id,
            summary=(
                None
                if current_input.summary is None
                else _thaw_json(current_input.summary)
            ),
        )
        if (
            persisted.envelope.child_attempt != self._source_attempt
            or persisted.envelope.status != current_input.status
            or persisted.envelope.source_event_range
            != current_input.source_event_range
            or persisted.envelope.artifact_refs != current_input.artifact_refs
        ):
            raise DerivedHandoffInputError(
                "handoff service changed the derived input binding"
            )
        handoff_append = self._handoff_recorder.sink.append_projected(
            _descriptor_bound_handoff_draft(
                self._handoff_recorder,
                persisted,
            )
        )
        _verified_handoff_append(
            handoff_append,
            consumer_attempt=self._consumer_attempt,
            persisted=persisted,
        )

        attachment = HostResolvedAttachment(
            artifact=persisted.full_output_artifact,
            kind="handoff",
            name=f"{self._source_attempt.attempt_id}.json",
            media_type=persisted.full_output_artifact.media_type,
        )
        content = _canonical_json(
            {
                "schema": "unchain.derived_handoff_input.v1",
                "consumer_attempt": self._consumer_attempt.to_dict(),
                "source_attempt": self._source_attempt.to_dict(),
                "handoff_event": handoff_append.cursor.to_dict(),
                "handoff_envelope": persisted.envelope.to_dict(),
                "full_output_artifact": persisted.full_output_artifact.to_dict(),
            }
        )
        input_append = self._input_ingress.persist(
            HostResolvedCurrentInput(
                attempt=self._consumer_attempt,
                content=content,
                message_index=current_input.message_index,
                attachments=(attachment,),
            )
        )
        _verified_input_append(
            input_append,
            consumer_attempt=self._consumer_attempt,
            content=content,
            attachment=attachment,
            handoff_cursor=handoff_append.cursor,
        )
        return DurableDerivedHandoffInputReceipt(
            envelope=persisted.envelope,
            full_output_artifact=persisted.full_output_artifact,
            handoff_cursor=handoff_append.cursor,
            input_cursor=input_append.cursor,
            handoff_duplicate=handoff_append.duplicate,
            input_duplicate=input_append.duplicate,
        )


__all__ = [
    "DerivedHandoffInputError",
    "DerivedHandoffInputIngress",
    "DurableDerivedHandoffInputReceipt",
    "HostResolvedDerivedHandoffInput",
]
