"""Durable, bounded recovery of an explicitly rejected provider input.

These records change an input allowance, never the model's declared window or
the canonical transcript. The compiler still owns all source/checkpoint choices.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from unchain.journal import AttemptRef, JournalAppendRequest, JournalEvent
from unchain.journal.models import _thaw_json
from unchain.journal.runtime import SemanticEventDraft

from .models import ContextCompileRequest


RECOVERY_EVENT = "context.provider_recovery"
RECOVERY_SCHEMA = "unchain.context_provider_recovery.v1"
_INPUT_TYPES = frozenset({"message.user", "interaction.resolved", "tool_result"})
_FIELDS = frozenset({
    "schema", "iteration", "provider", "model", "ordinal", "available_input_tokens",
    "source_input_event_id", "source_input_store_seq", "rejected_build_id",
    "before_message_bytes", "estimated_input_tokens", "parent_recovery_event_id",
})


class ContextLimitRecoveryError(RuntimeError):
    code = "context_limit_recovery_unavailable"


def _positive(value: Any) -> bool:
    return type(value) is int and 0 < value <= (1 << 53) - 1


def projection_bytes(messages: Sequence[Mapping[str, Any]]) -> int:
    return len(json.dumps(_thaw_json(messages), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _validate_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(payload) != _FIELDS or payload.get("schema") != RECOVERY_SCHEMA:
        raise ContextLimitRecoveryError("Context recovery record has invalid fields or version")
    if (type(payload["iteration"]) is not int or payload["iteration"] < 0
            or type(payload["ordinal"]) is not int or payload["ordinal"] not in {1, 2}
            or any(not _positive(payload[key]) for key in (
                "available_input_tokens", "source_input_store_seq", "before_message_bytes", "estimated_input_tokens"))):
        raise ContextLimitRecoveryError("Context recovery record has invalid limits")
    for key in ("provider", "model", "source_input_event_id", "rejected_build_id", "parent_recovery_event_id"):
        value = payload[key]
        if type(value) is not str or len(value) > 512 or (not value and key != "parent_recovery_event_id"):
            raise ContextLimitRecoveryError("Context recovery record has invalid identity")
    return payload


def _validate(event: JournalEvent) -> Mapping[str, Any]:
    payload = _validate_payload(event.payload)
    digest = _record_digest(event.attempt, payload["iteration"], payload["ordinal"])
    draft = SemanticEventDraft(
        event_id=f"context-recovery-{digest}", event_type=RECOVERY_EVENT, attempt=event.attempt,
        operation_id=f"context-recovery-{digest}", payload=payload,
    )
    if (event.event_type != RECOVERY_EVENT or event.event_id != draft.event_id
            or event.operation != draft.operation):
        raise ContextLimitRecoveryError("Context recovery record has invalid durable identity")
    return payload


def _record_digest(attempt: AttemptRef, iteration: int, ordinal: int) -> str:
    identity = {"attempt": attempt.to_dict(), "iteration": iteration, "ordinal": ordinal}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _history_events(*, events, attempt: AttemptRef, iteration: int, provider: str, model: str):
    if type(iteration) is not int or iteration < 0:
        raise ContextLimitRecoveryError("Context recovery iteration is invalid")
    inputs = [event for event in events if event.attempt.generation == attempt.generation
              and event.event_type in _INPUT_TYPES]
    attempt_inputs = [event for event in inputs if event.attempt == attempt]
    source = attempt_inputs[-1] if attempt_inputs else None
    records = [event for event in events if event.attempt == attempt and event.event_type == RECOVERY_EVENT]
    selected = [event for event in records if _validate(event)["iteration"] == iteration]
    if selected and source is not None and inputs[-1] != source:
        from .request_factory import _is_verified_derived_input

        generation_events = tuple(event for event in events if event.attempt.generation == attempt.generation)
        if not _is_verified_derived_input(source, generation_events=generation_events):
            raise ContextLimitRecoveryError("Context recovery has a foreign latest input")
    previous = None
    for ordinal, event in enumerate(selected, start=1):
        payload = _validate(event)
        if (payload["ordinal"] != ordinal or payload["provider"] != provider or payload["model"] != model
                or source is None
                or payload["source_input_event_id"] != source.event_id
                or payload["source_input_store_seq"] != source.store_seq
                or event.store_seq <= (previous.store_seq if previous else source.store_seq)
                or payload["parent_recovery_event_id"] != (previous.event_id if previous else "")
                or (previous and payload["available_input_tokens"] >= previous.payload["available_input_tokens"])):
            raise ContextLimitRecoveryError("Context recovery record conflicts with the current task")
        previous = event
    return previous, source


def load_provider_context_recovery(*, journal, attempt: AttemptRef, iteration: int, provider: str, model: str):
    event, _input = _history_events(events=journal.capture_snapshot().events, attempt=attempt,
                                  iteration=iteration, provider=provider, model=model)
    return event


def prepare_provider_context_recovery(*, journal, attempt: AttemptRef, iteration: int,
                                      provider: str, model: str, envelope: Mapping[str, Any], messages):
    previous, source = _history_events(events=journal.capture_snapshot().events, attempt=attempt,
                                      iteration=iteration, provider=provider, model=model)
    ordinal = previous.payload["ordinal"] + 1 if previous else 1
    if ordinal > 2:
        raise ContextLimitRecoveryError("Provider context limit still exceeded after two compaction recoveries")
    if source is None or source.attempt != attempt:
        raise ContextLimitRecoveryError("Context recovery has no current canonical input")
    estimate = envelope.get("estimated_input_tokens")
    budget = envelope.get("budget", {}).get("available_input_tokens")
    if not _positive(estimate) or not _positive(budget):
        raise ContextLimitRecoveryError("Context recovery has no measured input budget")
    if previous:
        budget = min(budget, previous.payload["available_input_tokens"])
    allowance = min(budget, estimate) * 3 // 4
    if allowance < 1:
        raise ContextLimitRecoveryError("Required context cannot be reduced further")
    payload = {
        "schema": RECOVERY_SCHEMA, "iteration": iteration, "provider": provider, "model": model,
        "ordinal": ordinal, "available_input_tokens": allowance,
        "source_input_event_id": source.event_id, "source_input_store_seq": source.store_seq,
        "rejected_build_id": envelope.get("build_id"), "before_message_bytes": projection_bytes(messages),
        "estimated_input_tokens": estimate, "parent_recovery_event_id": previous.event_id if previous else "",
    }
    _validate_payload(payload)
    digest = _record_digest(attempt, iteration, ordinal)
    draft = SemanticEventDraft(
        event_id=f"context-recovery-{digest}", event_type=RECOVERY_EVENT, attempt=attempt,
        operation_id=f"context-recovery-{digest}", payload=payload,
    )
    request = JournalAppendRequest(
        event_id=f"context-recovery-{digest}", event_type=RECOVERY_EVENT, attempt=attempt,
        operation=draft.operation, payload=payload,
    )
    # Validate the producer shape before persisting. The journal supplies store_seq.
    result = journal.append(request=request)
    event = result.event
    _validate(event)
    return event


def apply_provider_context_recovery(request: ContextCompileRequest, event: JournalEvent):
    payload = _validate(event)
    attempt = event.attempt
    if (request.execution_id != attempt.generation.execution_id
            or request.generation_id != attempt.generation.generation_id
            or request.attempt_id != attempt.attempt_id
            or request.provider != payload["provider"] or request.model != payload["model"]
            or request.budget is None):
        raise ContextLimitRecoveryError("Context recovery changed its compiler scope")
    allowance = min(request.budget.available_input_tokens, payload["available_input_tokens"])
    budget = replace(request.budget, available_input_tokens=allowance,
                     transport_margin_tokens=(request.budget.context_window_tokens
                                              - request.budget.output_reserve_tokens - allowance),
                     pressure_threshold_tokens=max(1, allowance * 9 // 10))
    build_digest = hashlib.sha256(f"{request.build_id}\x00{event.event_id}".encode()).hexdigest()
    return replace(request, budget=budget, build_id=f"context-recovery-build-{build_digest}")


def verify_provider_context_reduction(result, event: JournalEvent) -> None:
    payload = _validate(event)
    if (result.envelope is None
            or result.envelope.estimated_input_tokens >= payload["estimated_input_tokens"]
            or projection_bytes(result.messages) >= payload["before_message_bytes"]):
        raise ContextLimitRecoveryError("Required context could not be compacted; no provider retry was sent")


def recovery_occurrence_sha256(event: JournalEvent) -> str:
    _validate(event)
    return hashlib.sha256(f"provider-context-recovery\x00{event.event_id}".encode()).hexdigest()


def provider_recovery_build_trigger(request, snapshot, input_receipt):
    """Resolve a new build trigger without rebinding the original input claim.

    The coordinator calls this only after independently verifying the original
    canonical input. A recovery record is control evidence, never a user input.
    """
    if not str(request.build_id or "").startswith("context-recovery-build-"):
        return input_receipt
    for event in reversed(snapshot.events):
        if (event.event_type != RECOVERY_EVENT
                or event.attempt.generation.execution_id != request.execution_id
                or event.attempt.generation.generation_id != request.generation_id
                or event.attempt.attempt_id != request.attempt_id):
            continue
        payload = _validate(event)
        base_identity = {
            "domain": "unchain.context_build.trigger.v1", "attempt": event.attempt.to_dict(),
            "trigger": {"event_id": input_receipt.event_id, "store_seq": input_receipt.store_seq},
            "provider": request.provider, "model": request.model,
        }
        base = "context-build-" + hashlib.sha256(json.dumps(
            base_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        expected = "context-recovery-build-" + hashlib.sha256(f"{base}\x00{event.event_id}".encode()).hexdigest()
        if request.build_id != expected:
            continue
        latest, source = _history_events(events=snapshot.events, attempt=event.attempt,
            iteration=payload["iteration"], provider=request.provider, model=request.model)
        if latest != event or source != input_receipt:
            raise ContextLimitRecoveryError("Context recovery build is stale")
        rejected_build = base
        for predecessor in snapshot.events:
            if (predecessor.event_type == RECOVERY_EVENT and predecessor.attempt == event.attempt
                    and predecessor.payload.get("iteration") == payload["iteration"]):
                if predecessor.payload["rejected_build_id"] != rejected_build:
                    raise ContextLimitRecoveryError("Context recovery lost its rejected build chain")
                rejected_build = "context-recovery-build-" + hashlib.sha256(
                    f"{base}\x00{predecessor.event_id}".encode()).hexdigest()
        if (payload["source_input_event_id"] != input_receipt.event_id
                or payload["source_input_store_seq"] != input_receipt.store_seq
                or event.store_seq <= input_receipt.store_seq
                or payload["provider"] != request.provider or payload["model"] != request.model
                or request.budget is None
                or request.budget.available_input_tokens > payload["available_input_tokens"]):
            raise ContextLimitRecoveryError("Context recovery build changed its input or budget binding")
        return event
    raise ContextLimitRecoveryError("Context recovery build has no matching durable authorization")
