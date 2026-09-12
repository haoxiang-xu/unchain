from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from ..interaction.durable import (
    INTERACTION_JOURNAL_KEY,
    InteractionError,
    InteractionRequest,
    mark_interaction_applied,
    register_interaction_request,
    validate_interaction_journal,
    validate_interaction_request,
)
from ..kernel.state import RunState
from .ownership import ensure_session_delta_input


EXECUTION_CHECKPOINT_KEY = "execution_checkpoint"
EXECUTION_CHECKPOINT_DOMAIN_KEY = "execution_checkpoint_domain"
EXECUTION_CHECKPOINT_SCHEMA_VERSION = 1
_CHECKPOINT_STATUSES = frozenset(
    {"max_iterations", "awaiting_human_input", "awaiting_interaction"}
)


class ExecutionCheckpointError(RuntimeError):
    """Base error for durable execution-checkpoint failures."""

    code = "execution_checkpoint_error"


class ExecutionCheckpointIntegrityError(ExecutionCheckpointError):
    code = "execution_checkpoint_integrity_error"


class ExecutionCheckpointCompatibilityError(ExecutionCheckpointError):
    code = "execution_checkpoint_compatibility_error"


class ExecutionCheckpointResumeRequiredError(ExecutionCheckpointError):
    code = "execution_checkpoint_resume_required"


class ExecutionCheckpointReplayUnavailableError(ExecutionCheckpointError):
    code = "execution_checkpoint_replay_unavailable"


class ExecutionCheckpointPersistenceError(ExecutionCheckpointError):
    code = "execution_checkpoint_persistence_error"


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_payload_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def transcript_digest(messages: list[dict[str, Any]]) -> str:
    return stable_payload_digest(messages)


def _provider_replay_format(provider: str) -> str:
    if provider == "openai":
        return "openai.responses.v1"
    if provider in {"anthropic", "hyperspace"}:
        return "anthropic.messages.v1"
    if provider == "gemini":
        return "gemini.contents.v1"
    if provider == "ollama":
        return "ollama.chat.v1"
    return f"{provider or 'unknown'}.messages.v1"


def _fallback_replay_frame(state: RunState) -> dict[str, Any]:
    provider = str(state.provider_state.provider or "")
    reasoning_items = (
        list(state.last_model_turn.reasoning_items or [])
        if state.last_model_turn is not None
        else []
    )
    complete = not reasoning_items
    frame: dict[str, Any] = {
        "format": _provider_replay_format(provider),
        "complete": complete,
        "items": copy.deepcopy(state.transcript),
        "source": "kernel_transcript",
    }
    if not complete:
        frame["incomplete_reason"] = (
            f"{provider or 'provider'} reasoning output was not captured as an ordered, "
            "provider-ready replay frame"
        )
    return frame


def replay_frame_from_state(state: RunState) -> dict[str, Any]:
    replay_bucket = state.component_state.get("provider_replay")
    raw_frame = replay_bucket.get("frame") if isinstance(replay_bucket, dict) else None
    if isinstance(raw_frame, dict):
        frame = copy.deepcopy(raw_frame)
        frame.setdefault("format", _provider_replay_format(str(state.provider_state.provider or "")))
        frame.setdefault("complete", False)
        frame.setdefault("items", [])
        return _json_safe(frame)
    return _json_safe(_fallback_replay_frame(state))


def _human_input_request(state: RunState) -> dict[str, Any] | None:
    request = state.tool_batch_state.human_input_request
    if request is None:
        return None
    to_dict = getattr(request, "to_dict", None)
    return _json_safe(to_dict()) if callable(to_dict) else _json_safe(request)


def _checkpoint_interaction_request(
    state: RunState,
    *,
    status: str,
    run_id: str,
    session_id: str,
    base_session_revision: int | None,
) -> InteractionRequest | None:
    suspend_payload = state.suspend_state.payload
    raw_request = (
        suspend_payload.get("interaction_request")
        if isinstance(suspend_payload, dict)
        else None
    )
    if raw_request is None:
        if status == "awaiting_interaction":
            raise ExecutionCheckpointError(
                "awaiting_interaction checkpoint requires an interaction_request"
            )
        return None
    try:
        request = validate_interaction_request(raw_request)
    except InteractionError as exc:
        raise ExecutionCheckpointIntegrityError(
            f"invalid checkpoint interaction_request: {exc}"
        ) from exc
    if request.session_id != session_id:
        raise ExecutionCheckpointCompatibilityError(
            "interaction_request session_id does not match the checkpoint session"
        )
    expected_run_id = str(run_id or "kernel")
    if request.source_run_id != expected_run_id:
        raise ExecutionCheckpointCompatibilityError(
            "interaction_request source_run_id does not match the checkpoint run"
        )
    if (
        base_session_revision is not None
        and request.created_revision != base_session_revision
    ):
        raise ExecutionCheckpointCompatibilityError(
            "interaction_request created_revision does not match the checkpoint base revision"
        )
    expected_kind = {
        "awaiting_human_input": "human_input",
        "max_iterations": "max_budget",
    }.get(status)
    if expected_kind is not None and request.kind != expected_kind:
        raise ExecutionCheckpointCompatibilityError(
            f"{status} checkpoint requires an {expected_kind} interaction_request"
        )
    return request


def build_execution_checkpoint(
    state: RunState,
    *,
    status: str,
    run_id: str,
) -> dict[str, Any]:
    normalized_status = str(status or "")
    if normalized_status not in _CHECKPOINT_STATUSES:
        raise ExecutionCheckpointError(
            f"cannot create an execution checkpoint for status {normalized_status!r}"
        )
    session_id = str(state.session_state.session_id or "")
    if not session_id:
        raise ExecutionCheckpointError("execution checkpoint requires a non-empty session_id")

    base_session_revision = (
        int(state.memory_state.get("session_revision"))
        if isinstance(state.memory_state.get("session_revision"), int)
        and not isinstance(state.memory_state.get("session_revision"), bool)
        else None
    )
    interaction_request = _checkpoint_interaction_request(
        state,
        status=normalized_status,
        run_id=run_id,
        session_id=session_id,
        base_session_revision=base_session_revision,
    )

    payload = _json_safe(
        {
            "schema_version": EXECUTION_CHECKPOINT_SCHEMA_VERSION,
            "status": normalized_status,
            "session_id": session_id,
            "source_run_id": str(run_id or "kernel"),
            "provider": str(state.provider_state.provider or ""),
            "model": str(state.provider_state.model or ""),
            "iteration": int(state.iteration),
            "base_session_revision": base_session_revision,
            "transcript": copy.deepcopy(state.transcript),
            "pending_model_context": copy.deepcopy(state.next_model_input),
            "replay_frame": replay_frame_from_state(state),
            "continuation": copy.deepcopy(state.last_continuation),
            "human_input_request": _human_input_request(state),
            "previous_response_id": state.provider_state.previous_response_id,
            "use_previous_response_chain": bool(
                state.provider_state.use_previous_response_chain
            ),
            "max_context_window_tokens": int(
                state.provider_state.max_context_window_tokens or 0
            ),
            "token_state": {
                "consumed_tokens": int(state.token_state.consumed_tokens or 0),
                "input_tokens": int(state.token_state.input_tokens or 0),
                "output_tokens": int(state.token_state.output_tokens or 0),
                "cache_read_input_tokens": int(
                    state.token_state.cache_read_input_tokens or 0
                ),
                "cache_creation_input_tokens": int(
                    state.token_state.cache_creation_input_tokens or 0
                ),
                "last_turn_tokens": int(state.token_state.last_turn_tokens or 0),
                "last_turn_input_tokens": int(state.token_state.last_turn_input_tokens or 0),
                "last_turn_output_tokens": int(state.token_state.last_turn_output_tokens or 0),
                "last_turn_cache_read_input_tokens": int(
                    state.token_state.last_turn_cache_read_input_tokens or 0
                ),
                "last_turn_cache_creation_input_tokens": int(
                    state.token_state.last_turn_cache_creation_input_tokens or 0
                ),
            },
            "workspace_change_state": copy.deepcopy(state.workspace_change_state),
            **(
                {
                    "interaction_ref": {
                        "interaction_id": interaction_request.interaction_id,
                        "request_digest": interaction_request.request_digest,
                    }
                }
                if interaction_request is not None
                else {}
            ),
        }
    )
    digest = stable_payload_digest(payload)
    return {
        **payload,
        "checkpoint_id": f"checkpoint_{digest[:32]}",
        "integrity": {
            "algorithm": "sha256",
            "payload_sha256": digest,
        },
    }


def validate_execution_checkpoint(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ExecutionCheckpointIntegrityError("execution checkpoint must be a dict")
    checkpoint = _json_safe(copy.deepcopy(raw))
    if checkpoint.get("schema_version") != EXECUTION_CHECKPOINT_SCHEMA_VERSION:
        raise ExecutionCheckpointIntegrityError(
            "unsupported execution checkpoint schema_version: "
            f"{checkpoint.get('schema_version')!r}"
        )
    if checkpoint.get("status") not in _CHECKPOINT_STATUSES:
        raise ExecutionCheckpointIntegrityError(
            f"invalid execution checkpoint status: {checkpoint.get('status')!r}"
        )
    for key in ("session_id", "provider", "checkpoint_id"):
        if not isinstance(checkpoint.get(key), str) or not checkpoint.get(key):
            raise ExecutionCheckpointIntegrityError(
                f"execution checkpoint requires non-empty {key}"
            )
    if not isinstance(checkpoint.get("transcript"), list):
        raise ExecutionCheckpointIntegrityError(
            "execution checkpoint transcript must be a list"
        )
    iteration = checkpoint.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ExecutionCheckpointIntegrityError(
            "execution checkpoint iteration must be a non-negative integer"
        )
    max_context_window_tokens = checkpoint.get("max_context_window_tokens", 0)
    if (
        isinstance(max_context_window_tokens, bool)
        or not isinstance(max_context_window_tokens, int)
        or max_context_window_tokens < 0
    ):
        raise ExecutionCheckpointIntegrityError(
            "execution checkpoint max_context_window_tokens must be a non-negative integer"
        )
    token_state = checkpoint.get("token_state")
    if not isinstance(token_state, dict):
        raise ExecutionCheckpointIntegrityError(
            "execution checkpoint token_state must be a dict"
        )
    for token_key in (
        "consumed_tokens",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "last_turn_tokens",
        "last_turn_input_tokens",
        "last_turn_output_tokens",
        "last_turn_cache_read_input_tokens",
        "last_turn_cache_creation_input_tokens",
    ):
        token_value = token_state.get(token_key)
        if (
            isinstance(token_value, bool)
            or not isinstance(token_value, int)
            or token_value < 0
        ):
            raise ExecutionCheckpointIntegrityError(
                f"execution checkpoint token_state.{token_key} must be a non-negative integer"
            )
    if not isinstance(checkpoint.get("workspace_change_state"), dict):
        raise ExecutionCheckpointIntegrityError(
            "execution checkpoint workspace_change_state must be a dict"
        )
    pending_model_context = checkpoint.get("pending_model_context")
    if pending_model_context is not None and (
        not isinstance(pending_model_context, list)
        or any(not isinstance(message, dict) for message in pending_model_context)
    ):
        raise ExecutionCheckpointIntegrityError(
            "execution checkpoint pending_model_context must be a message list or null"
        )
    replay_frame = checkpoint.get("replay_frame")
    if not isinstance(replay_frame, dict):
        raise ExecutionCheckpointIntegrityError(
            "execution checkpoint replay_frame must be a dict"
        )
    if not isinstance(replay_frame.get("complete"), bool):
        raise ExecutionCheckpointIntegrityError(
            "execution checkpoint replay_frame.complete must be a bool"
        )
    if not isinstance(replay_frame.get("items"), list):
        raise ExecutionCheckpointIntegrityError(
            "execution checkpoint replay_frame.items must be a list"
        )
    if checkpoint.get("status") == "awaiting_human_input":
        if not isinstance(checkpoint.get("continuation"), dict):
            raise ExecutionCheckpointIntegrityError(
                "awaiting_human_input checkpoint requires continuation"
            )
        if not isinstance(checkpoint.get("human_input_request"), dict):
            raise ExecutionCheckpointIntegrityError(
                "awaiting_human_input checkpoint requires human_input_request"
            )
    interaction_ref = checkpoint.get("interaction_ref")
    if interaction_ref is not None:
        if not isinstance(interaction_ref, dict) or set(interaction_ref) != {
            "interaction_id",
            "request_digest",
        }:
            raise ExecutionCheckpointIntegrityError(
                "execution checkpoint interaction_ref must contain only "
                "interaction_id and request_digest"
            )
        for key in ("interaction_id", "request_digest"):
            if not isinstance(interaction_ref.get(key), str) or not interaction_ref.get(key):
                raise ExecutionCheckpointIntegrityError(
                    f"execution checkpoint interaction_ref.{key} must be a non-empty string"
                )
    if checkpoint.get("status") == "awaiting_interaction":
        if interaction_ref is None:
            raise ExecutionCheckpointIntegrityError(
                "awaiting_interaction checkpoint requires interaction_ref"
            )
        if not isinstance(checkpoint.get("continuation"), dict):
            raise ExecutionCheckpointIntegrityError(
                "awaiting_interaction checkpoint requires continuation"
            )
    base_revision = checkpoint.get("base_session_revision")
    if (
        base_revision is not None
        and (
            isinstance(base_revision, bool)
            or not isinstance(base_revision, int)
            or base_revision < 0
        )
    ):
        raise ExecutionCheckpointIntegrityError(
            "execution checkpoint base_session_revision must be a non-negative integer or null"
        )

    integrity = checkpoint.get("integrity")
    if not isinstance(integrity, dict) or integrity.get("algorithm") != "sha256":
        raise ExecutionCheckpointIntegrityError(
            "execution checkpoint integrity.algorithm must be sha256"
        )
    expected_digest = integrity.get("payload_sha256")
    payload = {
        key: copy.deepcopy(value)
        for key, value in checkpoint.items()
        if key not in {"checkpoint_id", "integrity"}
    }
    actual_digest = stable_payload_digest(payload)
    if expected_digest != actual_digest:
        raise ExecutionCheckpointIntegrityError(
            "execution checkpoint payload hash mismatch"
        )
    if checkpoint.get("checkpoint_id") != f"checkpoint_{actual_digest[:32]}":
        raise ExecutionCheckpointIntegrityError(
            "execution checkpoint id does not match its payload"
        )
    return checkpoint


def _validate_checkpoint_request_binding(
    checkpoint: dict[str, Any],
    request: InteractionRequest,
) -> None:
    interaction_ref = checkpoint.get("interaction_ref")
    if not isinstance(interaction_ref, dict):
        raise ExecutionCheckpointCompatibilityError(
            "interaction_request was supplied for a checkpoint without interaction_ref"
        )
    if (
        interaction_ref.get("interaction_id") != request.interaction_id
        or interaction_ref.get("request_digest") != request.request_digest
    ):
        raise ExecutionCheckpointCompatibilityError(
            "interaction_request does not match the checkpoint interaction_ref"
        )
    if request.session_id != checkpoint.get("session_id"):
        raise ExecutionCheckpointCompatibilityError(
            "interaction_request belongs to a different session"
        )
    if request.source_run_id != checkpoint.get("source_run_id"):
        raise ExecutionCheckpointCompatibilityError(
            "interaction_request belongs to a different source run"
        )
    base_revision = checkpoint.get("base_session_revision")
    if base_revision is not None and request.created_revision != base_revision:
        raise ExecutionCheckpointCompatibilityError(
            "interaction_request created_revision does not match the checkpoint"
        )


def register_checkpoint_interaction_request(
    session_state: dict[str, Any],
    *,
    checkpoint: dict[str, Any],
    interaction_request: Any,
) -> None:
    """Register a checkpoint-bound request inside the same session-state CAS."""

    try:
        request = validate_interaction_request(interaction_request)
        _validate_checkpoint_request_binding(checkpoint, request)
        journal = validate_interaction_journal(
            session_state.get(INTERACTION_JOURNAL_KEY)
        )
    except InteractionError as exc:
        raise ExecutionCheckpointIntegrityError(
            f"invalid durable interaction state: {exc}"
        ) from exc

    active_id = journal.get("active_id")
    if active_id is not None and active_id != request.interaction_id:
        active_entry = journal["entries"].get(active_id)
        current_raw = session_state.get(EXECUTION_CHECKPOINT_KEY)
        current = (
            validate_execution_checkpoint(current_raw)
            if isinstance(current_raw, dict)
            else None
        )
        current_ref = current.get("interaction_ref") if isinstance(current, dict) else None
        if (
            not isinstance(active_entry, dict)
            or not isinstance(current_ref, dict)
            or current_ref.get("interaction_id") != active_id
            or active_entry.get("checkpoint_id") != current.get("checkpoint_id")
        ):
            raise ExecutionCheckpointCompatibilityError(
                "active interaction is not bound to the current execution checkpoint"
            )
        receipt = active_entry.get("receipt")
        if not isinstance(receipt, dict):
            raise ExecutionCheckpointResumeRequiredError(
                "cannot replace an unanswered durable interaction"
            )
        try:
            journal = mark_interaction_applied(
                journal,
                interaction_id=active_id,
                receipt_id=str(receipt.get("receipt_id") or ""),
                applied_checkpoint_id=str(checkpoint.get("checkpoint_id") or ""),
            )
        except InteractionError as exc:
            raise ExecutionCheckpointIntegrityError(
                f"failed to apply the prior durable interaction: {exc}"
            ) from exc

    existing = journal["entries"].get(request.interaction_id)
    if isinstance(existing, dict) and existing.get("application") is not None:
        raise ExecutionCheckpointCompatibilityError(
            "cannot reactivate an already-applied durable interaction"
        )
    try:
        journal = register_interaction_request(
            journal,
            request,
            checkpoint_id=str(checkpoint.get("checkpoint_id") or ""),
        )
    except InteractionError as exc:
        raise ExecutionCheckpointCompatibilityError(
            f"failed to register checkpoint interaction: {exc}"
        ) from exc
    session_state[INTERACTION_JOURNAL_KEY] = journal


def apply_checkpoint_interaction_receipt(
    session_state: dict[str, Any],
    *,
    checkpoint: dict[str, Any],
    applied_checkpoint_id: str,
) -> bool:
    """Tombstone a receipt before atomically clearing/replacing its checkpoint."""

    interaction_ref = checkpoint.get("interaction_ref")
    if interaction_ref is None:
        return False
    try:
        journal = validate_interaction_journal(
            session_state.get(INTERACTION_JOURNAL_KEY)
        )
    except InteractionError as exc:
        raise ExecutionCheckpointIntegrityError(
            f"invalid durable interaction journal: {exc}"
        ) from exc
    interaction_id = str(interaction_ref.get("interaction_id") or "")
    entry = journal["entries"].get(interaction_id)
    if not isinstance(entry, dict):
        raise ExecutionCheckpointIntegrityError(
            "checkpoint interaction_ref is missing from the interaction journal"
        )
    request = entry.get("request")
    if (
        not isinstance(request, dict)
        or request.get("request_digest") != interaction_ref.get("request_digest")
        or request.get("session_id") != checkpoint.get("session_id")
        or entry.get("checkpoint_id") != checkpoint.get("checkpoint_id")
    ):
        raise ExecutionCheckpointIntegrityError(
            "checkpoint interaction_ref does not match its journal entry"
        )
    receipt = entry.get("receipt")
    if not isinstance(receipt, dict):
        raise ExecutionCheckpointResumeRequiredError(
            "cannot clear an interaction checkpoint before its receipt is recorded"
        )
    try:
        session_state[INTERACTION_JOURNAL_KEY] = mark_interaction_applied(
            journal,
            interaction_id=interaction_id,
            receipt_id=str(receipt.get("receipt_id") or ""),
            applied_checkpoint_id=applied_checkpoint_id,
        )
    except InteractionError as exc:
        raise ExecutionCheckpointIntegrityError(
            f"failed to apply checkpoint interaction receipt: {exc}"
        ) from exc
    return True


def ensure_checkpoint_compatible(
    checkpoint: dict[str, Any],
    *,
    session_id: str,
    provider: str | None,
    model: str | None,
) -> None:
    if checkpoint.get("session_id") != session_id:
        raise ExecutionCheckpointCompatibilityError(
            "execution checkpoint belongs to a different session"
        )
    expected_provider = str(checkpoint.get("provider") or "")
    actual_provider = str(provider or "")
    if expected_provider and actual_provider and expected_provider != actual_provider:
        raise ExecutionCheckpointCompatibilityError(
            "execution checkpoint provider mismatch: "
            f"expected {expected_provider!r}, got {actual_provider!r}"
        )
    expected_model = str(checkpoint.get("model") or "")
    actual_model = str(model or "")
    if expected_model and actual_model and expected_model != actual_model:
        raise ExecutionCheckpointCompatibilityError(
            "execution checkpoint model mismatch: "
            f"expected {expected_model!r}, got {actual_model!r}"
        )


def ensure_checkpoint_replayable(checkpoint: dict[str, Any]) -> None:
    replay_frame = checkpoint.get("replay_frame")
    if not isinstance(replay_frame, dict) or replay_frame.get("complete") is not True:
        reason = (
            str(replay_frame.get("incomplete_reason") or "")
            if isinstance(replay_frame, dict)
            else ""
        )
        suffix = f": {reason}" if reason else ""
        raise ExecutionCheckpointReplayUnavailableError(
            "execution checkpoint does not contain a complete provider replay frame"
            + suffix
        )


def restore_fresh_checkpoint_messages(
    checkpoint: dict[str, Any],
    *,
    incoming_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if (
        checkpoint.get("status") in {"awaiting_human_input", "awaiting_interaction"}
        or isinstance(checkpoint.get("interaction_ref"), dict)
    ):
        raise ExecutionCheckpointResumeRequiredError(
            "session is awaiting a durable interaction; resume the persisted "
            "continuation instead of starting a fresh run"
        )
    ensure_checkpoint_replayable(checkpoint)
    transcript = copy.deepcopy(checkpoint.get("transcript") or [])
    ensure_session_delta_input(
        history=transcript,
        incoming=incoming_messages,
    )
    return transcript


def merge_checkpoint_transcript_with_incoming(
    checkpoint_messages: list[dict[str, Any]],
    incoming_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def split(messages: list[dict[str, Any]]):
        systems: list[dict[str, Any]] = []
        conversation: list[dict[str, Any]] = []
        for message in messages:
            copied = copy.deepcopy(message)
            if copied.get("role") in {"system", "developer"}:
                systems.append(copied)
            else:
                conversation.append(copied)
        return systems, conversation

    def prompt_projection(messages: list[dict[str, Any]]):
        return [
            {
                "role": message.get("role"),
                "content": copy.deepcopy(message.get("content")),
            }
            for message in messages
        ]

    checkpoint_systems, checkpoint_conversation = split(checkpoint_messages)
    incoming_systems, incoming_conversation = split(incoming_messages)
    if incoming_systems:
        expected = prompt_projection(checkpoint_systems)
        actual = prompt_projection(incoming_systems)
        same_full_prompt = actual == expected
        same_leading_agent_instruction = bool(
            len(actual) == 1
            and expected
            and actual[0] == expected[0]
        )
        if not same_full_prompt and not same_leading_agent_instruction:
            raise ExecutionCheckpointCompatibilityError(
                "incoming system/developer instructions do not match the persisted "
                "execution checkpoint"
            )
    return [
        *copy.deepcopy(checkpoint_systems),
        *copy.deepcopy(checkpoint_conversation),
        *copy.deepcopy(incoming_conversation),
    ]


def restore_resume_checkpoint_messages(
    checkpoint: dict[str, Any],
    *,
    conversation: list[dict[str, Any]],
    continuation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    resumable_interaction = (
        isinstance(checkpoint.get("interaction_ref"), dict)
        and checkpoint.get("status")
        in {"awaiting_human_input", "awaiting_interaction", "max_iterations"}
    )
    if checkpoint.get("status") != "awaiting_human_input" and not resumable_interaction:
        raise ExecutionCheckpointCompatibilityError(
            "interaction resume requires a durable interaction checkpoint"
        )
    ensure_checkpoint_replayable(checkpoint)
    transcript = copy.deepcopy(checkpoint.get("transcript") or [])
    if _json_safe(conversation) != _json_safe(transcript):
        raise ExecutionCheckpointCompatibilityError(
            "resume conversation does not match the persisted execution checkpoint"
        )
    expected_continuation = checkpoint.get("continuation")
    if not isinstance(continuation, dict) or _json_safe(continuation) != _json_safe(
        expected_continuation
    ):
        raise ExecutionCheckpointCompatibilityError(
            "resume continuation does not match the persisted execution checkpoint"
        )
    return transcript


__all__ = [
    "EXECUTION_CHECKPOINT_DOMAIN_KEY",
    "EXECUTION_CHECKPOINT_KEY",
    "EXECUTION_CHECKPOINT_SCHEMA_VERSION",
    "ExecutionCheckpointCompatibilityError",
    "ExecutionCheckpointError",
    "ExecutionCheckpointIntegrityError",
    "ExecutionCheckpointPersistenceError",
    "ExecutionCheckpointReplayUnavailableError",
    "ExecutionCheckpointResumeRequiredError",
    "apply_checkpoint_interaction_receipt",
    "build_execution_checkpoint",
    "ensure_checkpoint_compatible",
    "ensure_checkpoint_replayable",
    "replay_frame_from_state",
    "register_checkpoint_interaction_request",
    "merge_checkpoint_transcript_with_incoming",
    "restore_fresh_checkpoint_messages",
    "restore_resume_checkpoint_messages",
    "stable_payload_digest",
    "transcript_digest",
    "validate_execution_checkpoint",
]
