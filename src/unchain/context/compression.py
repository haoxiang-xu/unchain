"""Durable, at-most-once context-compression receipts.

The service deliberately owns only the compression side effect.  Admission of
root history and selection of a checkpoint remain coordinator responsibilities.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from unchain.journal import (
    AttemptRef,
    BoundExecutionJournal,
    JournalAppendRequest,
    JournalEvent,
    JournalSnapshot,
    build_operation_ref,
    capture_journal_snapshot,
)
from unchain.journal.snapshot import journal_event_sha256

from .budget import estimate_context_tokens
from .checkpoints import CheckpointProjectionError, CheckpointRequest


_INTENT_SCHEMA = "unchain.context_compression_intent.v1"
_RESULT_SCHEMA = "unchain.context_compression_result.v1"
_PROJECTION_SCHEMA = "unchain.context_compression.v1"
_INTENT_EVENT = "context.compression_intent"
_RESULT_EVENT = "context.compression_result"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_SUMMARY_BYTES = 128 * 1024


class ContextCompressionError(RuntimeError):
    """A compression request or its durable receipt is invalid."""


class CompressionOutcomeUnknown(ContextCompressionError):
    """An intent exists without an exact durable result; it must not be retried."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise ContextCompressionError("compression value is not canonical JSON") from exc


def _plain(value: Any) -> Any:
    """Detach a JSON value and reject non-string mapping keys."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        detached: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContextCompressionError("source messages contain a non-string key")
            detached[key] = _plain(item)
        return detached
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    raise ContextCompressionError("source messages are not JSON")


def _source_digest(messages: tuple[dict[str, Any], ...]) -> str:
    digest = hashlib.sha256()
    for message in messages:
        encoded = _canonical_bytes(message)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _profile_digest(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("profile_digest must be a lowercase SHA-256 digest")
    return value


def _target_tokens(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 8_192:
        raise ValueError("target_tokens must be between 1 and 8192")
    return value


def _summary(value: object, *, target_tokens: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContextCompressionError("compression summary must be non-empty text")
    if len(value.encode("utf-8")) > _MAX_SUMMARY_BYTES:
        raise ContextCompressionError("compression summary exceeds the durable limit")
    if estimate_context_tokens(({"role": "user", "content": value},)).total_tokens > target_tokens:
        raise ContextCompressionError("compression summary exceeds the requested token target")
    return value


def _operation_id(
    *,
    attempt: AttemptRef,
    checkpoint: CheckpointRequest,
    target_tokens: int,
) -> str:
    identity = {
        "attempt": attempt.to_dict(),
        "checkpoint_request_id": checkpoint.request_id,
        "target_tokens": target_tokens,
    }
    return "context-compression." + hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def _intent_payload(
    *,
    attempt: AttemptRef,
    checkpoint: CheckpointRequest,
    profile_digest: str,
    target_tokens: int,
) -> dict[str, Any]:
    return {
        "schema": _INTENT_SCHEMA,
        "attempt": attempt.to_dict(),
        "checkpoint": checkpoint.to_dict(),
        "profile_digest": profile_digest,
        # The generic durable journal rejects arbitrary ``*token*`` payload
        # keys as potential plaintext credentials.  ``target_limit`` is the
        # registered integer metadata form for this non-secret limit.
        "target_limit": target_tokens,
    }


def _projection(
    *,
    operation_id: str,
    checkpoint: CheckpointRequest,
    profile_digest: str,
    target_tokens: int,
    summary: str,
) -> dict[str, Any]:
    return {
        "schema": _PROJECTION_SCHEMA,
        "operation_id": operation_id,
        "checkpoint_request_id": checkpoint.request_id,
        "profile_digest": profile_digest,
        "target_tokens": target_tokens,
        "summary": summary,
    }


def validate_compression_projection(
    value: Mapping[str, Any],
    *,
    checkpoint_request_id: str,
) -> dict[str, Any]:
    """Validate and detach the small public proof embedded in model context.

    This validates representation only.  A coordinator must still verify the
    proof against :class:`ContextCompressionService` and its journal.
    """

    expected = {
        "schema",
        "operation_id",
        "checkpoint_request_id",
        "profile_digest",
        "target_tokens",
        "summary",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CheckpointProjectionError("compression projection is invalid")
    if value.get("schema") != _PROJECTION_SCHEMA:
        raise CheckpointProjectionError("compression projection schema is invalid")
    if value.get("checkpoint_request_id") != checkpoint_request_id:
        raise CheckpointProjectionError("compression projection checkpoint is invalid")
    operation_id = value.get("operation_id")
    if (
        not isinstance(operation_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", operation_id)
        or len(operation_id) > 256
    ):
        raise CheckpointProjectionError("compression projection operation is invalid")
    try:
        profile_digest = _profile_digest(value.get("profile_digest"))
        target_tokens = _target_tokens(value.get("target_tokens"))
        summary = _summary(value.get("summary"), target_tokens=target_tokens)
    except (TypeError, ValueError, ContextCompressionError) as exc:
        raise CheckpointProjectionError("compression projection is invalid") from exc
    return {
        "schema": _PROJECTION_SCHEMA,
        "operation_id": operation_id,
        "checkpoint_request_id": checkpoint_request_id,
        "profile_digest": profile_digest,
        "target_tokens": target_tokens,
        "summary": summary,
    }


class ContextCompressionService:
    """Record an intent before invoking a single non-retryable summarizer."""

    def __init__(
        self,
        journal: BoundExecutionJournal,
        *,
        attempt: AttemptRef,
        profile_digest: str,
        summarize: Callable[..., str],
    ) -> None:
        if not isinstance(journal, BoundExecutionJournal):
            raise TypeError("journal must be a BoundExecutionJournal")
        if not isinstance(attempt, AttemptRef):
            raise TypeError("attempt must be an AttemptRef")
        if journal.execution_id != attempt.generation.execution_id:
            raise ContextCompressionError("journal scope differs from compression attempt")
        if not callable(summarize):
            raise TypeError("summarize must be callable")
        self._journal = journal
        self._attempt = attempt
        self._profile_digest = _profile_digest(profile_digest)
        self._summarize = summarize

    @property
    def journal(self) -> BoundExecutionJournal:
        return self._journal

    @property
    def attempt(self) -> AttemptRef:
        return self._attempt

    def _snapshot(self) -> JournalSnapshot:
        snapshot = self._journal.capture_snapshot()
        if not isinstance(snapshot, JournalSnapshot):
            raise ContextCompressionError("journal did not return a JournalSnapshot")
        try:
            snapshot = JournalSnapshot.from_dict(snapshot.to_dict())
        except (TypeError, ValueError) as exc:
            raise ContextCompressionError("journal snapshot is invalid") from exc
        if snapshot.execution_id != self._attempt.generation.execution_id:
            raise ContextCompressionError("journal snapshot escaped the attempt execution")
        return snapshot

    def _validate_sources(
        self,
        *,
        snapshot: JournalSnapshot,
        checkpoint: CheckpointRequest,
        messages: tuple[dict[str, Any], ...],
    ) -> None:
        if _source_digest(messages) != checkpoint.source_messages_sha256:
            raise ContextCompressionError("source messages do not match checkpoint digest")
        self._validate_checkpoint_receipts(snapshot=snapshot, checkpoint=checkpoint)

    def _validate_checkpoint_receipts(
        self,
        *,
        snapshot: JournalSnapshot,
        checkpoint: CheckpointRequest,
    ) -> None:
        """Bind checkpoint source and projection receipt digests to one snapshot."""

        by_seq = {event.store_seq: event for event in snapshot.events}
        by_id = {event.event_id: event for event in snapshot.events}
        for event_id, store_seq, event_digest in zip(
            checkpoint.source_event_ids,
            checkpoint.source_event_store_seqs,
            checkpoint.source_event_sha256s,
            strict=True,
        ):
            event = by_seq.get(store_seq)
            if (
                event is None
                or by_id.get(event_id) is not event
                or event.event_id != event_id
                or event.attempt.generation != self._attempt.generation
                or journal_event_sha256(event) != event_digest
            ):
                raise ContextCompressionError("checkpoint source receipt failed exact verification")
        for dependency in checkpoint.projection_dependencies:
            receipt = by_seq.get(dependency.receipt_cursor.store_seq)
            if (
                receipt is None
                or by_id.get(dependency.receipt_cursor.event_id) is not receipt
                or receipt.event_id != dependency.receipt_cursor.event_id
                or receipt.attempt.generation != self._attempt.generation
                or receipt.event_type != dependency.event_type
                or receipt.attempt.attempt_id != dependency.attempt_id
                or journal_event_sha256(receipt) != dependency.event_sha256
            ):
                raise ContextCompressionError("checkpoint dependency receipt failed exact verification")

    def _requests(
        self,
        *,
        checkpoint: CheckpointRequest,
        target_tokens: int,
        summary: str | None = None,
    ) -> tuple[str, JournalAppendRequest, JournalAppendRequest | None]:
        operation_id = _operation_id(
            attempt=self._attempt,
            checkpoint=checkpoint,
            target_tokens=target_tokens,
        )
        intent_payload = _intent_payload(
            attempt=self._attempt,
            checkpoint=checkpoint,
            profile_digest=self._profile_digest,
            target_tokens=target_tokens,
        )
        intent = JournalAppendRequest(
            event_id=f"event.{operation_id}.intent",
            event_type=_INTENT_EVENT,
            attempt=self._attempt,
            operation=build_operation_ref(
                f"{operation_id}.intent",
                domain=_INTENT_EVENT,
                payload=intent_payload,
            ),
            payload=intent_payload,
        )
        if summary is None:
            return operation_id, intent, None
        result_payload = {
            "schema": _RESULT_SCHEMA,
            "intent": intent_payload,
            "summary": summary,
        }
        result = JournalAppendRequest(
            event_id=f"event.{operation_id}.result",
            event_type=_RESULT_EVENT,
            attempt=self._attempt,
            operation=build_operation_ref(
                f"{operation_id}.result",
                domain=_RESULT_EVENT,
                payload=result_payload,
            ),
            payload=result_payload,
        )
        return operation_id, intent, result

    @staticmethod
    def _exact_event(event: JournalEvent, request: JournalAppendRequest) -> bool:
        return (
            event.event_id == request.event_id
            and event.event_type == request.event_type
            and event.attempt == request.attempt
            and event.operation == request.operation
            and dict(event.payload) == dict(request.payload)
            and event.resource_refs == request.resource_refs
        )

    def _find_event(self, snapshot: JournalSnapshot, event_id: str) -> JournalEvent | None:
        return next((event for event in snapshot.events if event.event_id == event_id), None)

    @staticmethod
    def _validate_pre_intent_order(
        *,
        checkpoint: CheckpointRequest,
        intent_event: JournalEvent,
        result_event: JournalEvent | None = None,
    ) -> None:
        """Ensure a receipt cannot authorize evidence written after its intent."""

        evidence_sequences = [*checkpoint.source_event_store_seqs]
        for dependency in checkpoint.projection_dependencies:
            evidence_sequences.extend(
                (
                    dependency.source_cursor.store_seq,
                    dependency.receipt_cursor.store_seq,
                )
            )
        if any(sequence >= intent_event.store_seq for sequence in evidence_sequences):
            raise ContextCompressionError(
                "checkpoint evidence must precede the compression intent"
            )
        if result_event is not None and result_event.store_seq <= intent_event.store_seq:
            raise ContextCompressionError(
                "compression result must follow the compression intent"
            )

    def _stored_result(
        self,
        *,
        snapshot: JournalSnapshot,
        checkpoint: CheckpointRequest,
        target_tokens: int,
    ) -> tuple[dict[str, Any], JournalEvent, JournalEvent] | None:
        operation_id, intent, _ = self._requests(
            checkpoint=checkpoint,
            target_tokens=target_tokens,
        )
        result_event = self._find_event(snapshot, f"event.{operation_id}.result")
        intent_event = self._find_event(snapshot, intent.event_id)
        if result_event is None:
            if intent_event is not None and not self._exact_event(intent_event, intent):
                raise ContextCompressionError("compression intent receipt changed")
            if intent_event is not None:
                self._validate_pre_intent_order(
                    checkpoint=checkpoint,
                    intent_event=intent_event,
                )
            return None
        if intent_event is None or not self._exact_event(intent_event, intent):
            raise ContextCompressionError("compression result lacks its exact intent")
        self._validate_pre_intent_order(
            checkpoint=checkpoint,
            intent_event=intent_event,
            result_event=result_event,
        )
        payload = dict(result_event.payload)
        if set(payload) != {"schema", "intent", "summary"} or payload.get("schema") != _RESULT_SCHEMA:
            raise ContextCompressionError("compression result receipt is invalid")
        if payload.get("intent") != dict(intent.payload):
            raise ContextCompressionError("compression result intent identity changed")
        summary = _summary(payload.get("summary"), target_tokens=target_tokens)
        _, _, expected_result = self._requests(
            checkpoint=checkpoint,
            target_tokens=target_tokens,
            summary=summary,
        )
        assert expected_result is not None
        if not self._exact_event(result_event, expected_result):
            raise ContextCompressionError("compression result receipt changed")
        return (
            _projection(
                operation_id=operation_id,
                checkpoint=checkpoint,
                profile_digest=self._profile_digest,
                target_tokens=target_tokens,
                summary=summary,
            ),
            intent_event,
            result_event,
        )

    def _messages(self, source_messages: object) -> tuple[dict[str, Any], ...]:
        if not isinstance(source_messages, tuple):
            raise TypeError("source_messages must be a tuple")
        detached = _plain(source_messages)
        if not isinstance(detached, list) or any(not isinstance(item, dict) for item in detached):
            raise ContextCompressionError("source_messages must be a tuple of objects")
        if len(_canonical_bytes(detached)) > _MAX_SOURCE_BYTES:
            raise ContextCompressionError("source messages exceed the durable limit")
        return tuple(detached)

    def compress(
        self,
        checkpoint: CheckpointRequest,
        source_messages: tuple[dict[str, Any], ...],
        *,
        target_tokens: int,
    ) -> dict[str, Any]:
        if not isinstance(checkpoint, CheckpointRequest) or not checkpoint.authoritative:
            raise TypeError("checkpoint must be an authoritative CheckpointRequest")
        target_tokens = _target_tokens(target_tokens)
        messages = self._messages(source_messages)
        snapshot = self._snapshot()
        self._validate_sources(snapshot=snapshot, checkpoint=checkpoint, messages=messages)
        stored = self._stored_result(
            snapshot=snapshot,
            checkpoint=checkpoint,
            target_tokens=target_tokens,
        )
        if stored is not None:
            return stored[0]

        operation_id, intent, _ = self._requests(
            checkpoint=checkpoint,
            target_tokens=target_tokens,
        )
        intent_event = self._find_event(snapshot, intent.event_id)
        if intent_event is not None:
            if not self._exact_event(intent_event, intent):
                raise ContextCompressionError("compression intent receipt changed")
            raise CompressionOutcomeUnknown("compression intent has no durable result")
        appended = self._journal.append(request=intent)
        if not self._exact_event(appended.event, intent):
            raise ContextCompressionError("journal returned a changed compression intent")
        if appended.duplicate:
            # A competing writer may have completed after its append.  It is
            # still forbidden for this caller to invoke the summarizer.
            replay = self._stored_result(
                snapshot=self._snapshot(),
                checkpoint=checkpoint,
                target_tokens=target_tokens,
            )
            if replay is not None:
                return replay[0]
            raise CompressionOutcomeUnknown("compression intent is owned by another caller")
        try:
            summary = _summary(
                self._summarize(
                    source_messages=messages,
                    target_tokens=target_tokens,
                    operation_id=operation_id,
                ),
                target_tokens=target_tokens,
            )
        except Exception as exc:
            raise CompressionOutcomeUnknown("compression intent has no durable result") from exc
        _, _, result = self._requests(
            checkpoint=checkpoint,
            target_tokens=target_tokens,
            summary=summary,
        )
        assert result is not None
        appended_result = self._journal.append(request=result)
        if not self._exact_event(appended_result.event, result):
            raise ContextCompressionError("journal returned a changed compression result")
        return _projection(
            operation_id=operation_id,
            checkpoint=checkpoint,
            profile_digest=self._profile_digest,
            target_tokens=target_tokens,
            summary=summary,
        )

    def source_snapshot(self, projection: Mapping[str, Any]) -> JournalSnapshot:
        """Return the verified pre-intent snapshot used for stable replay."""

        if not isinstance(projection, Mapping):
            raise CheckpointProjectionError("compression projection is invalid")
        checkpoint_request_id = projection.get("checkpoint_request_id")
        if not isinstance(checkpoint_request_id, str):
            raise CheckpointProjectionError("compression projection is invalid")
        public = validate_compression_projection(
            projection,
            checkpoint_request_id=checkpoint_request_id,
        )
        snapshot = self._snapshot()
        result_event = self._find_event(
            snapshot,
            f"event.{public['operation_id']}.result",
        )
        if result_event is None:
            raise ContextCompressionError("compression result receipt is unavailable")
        payload = dict(result_event.payload)
        if set(payload) != {"schema", "intent", "summary"} or payload.get("schema") != _RESULT_SCHEMA:
            raise ContextCompressionError("compression result receipt is invalid")
        raw_intent = payload.get("intent")
        if not isinstance(raw_intent, Mapping) or set(raw_intent) != {
            "schema", "attempt", "checkpoint", "profile_digest", "target_limit"
        } or raw_intent.get("schema") != _INTENT_SCHEMA:
            raise ContextCompressionError("compression result intent is invalid")
        try:
            checkpoint = CheckpointRequest.from_dict(raw_intent["checkpoint"])
            attempt = AttemptRef.from_dict(raw_intent["attempt"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ContextCompressionError("compression result intent is invalid") from exc
        if attempt != self._attempt or checkpoint.request_id != checkpoint_request_id:
            raise ContextCompressionError("compression result scope is invalid")
        target_tokens = _target_tokens(raw_intent.get("target_limit"))
        if _profile_digest(raw_intent.get("profile_digest")) != self._profile_digest:
            raise ContextCompressionError("compression result profile is invalid")
        # ``_snapshot`` has already round-tripped the complete current
        # snapshot, including its chain digest.  Checkpoint receipt hashes
        # must also bind to that full snapshot before the prefix is cut.
        self._validate_checkpoint_receipts(snapshot=snapshot, checkpoint=checkpoint)
        stored = self._stored_result(
            snapshot=snapshot,
            checkpoint=checkpoint,
            target_tokens=target_tokens,
        )
        if stored is None or stored[0] != public:
            raise ContextCompressionError("compression projection is not durably verified")
        intent_event = stored[1]
        return capture_journal_snapshot(
            execution_id=snapshot.execution_id,
            events=snapshot.events[: intent_event.store_seq - 1],
        )


__all__ = [
    "CompressionOutcomeUnknown",
    "ContextCompressionError",
    "ContextCompressionService",
    "validate_compression_projection",
]
