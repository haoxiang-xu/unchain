from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol

from unchain.journal import (
    ArtifactRef,
    AttemptRef,
    BoundToolReceiptIndex,
    PreparedSemanticEvent,
    ResourceRef,
    SemanticEventDraft,
)
from unchain.journal.models import _required_text, _sha256
from unchain.tools.output_management import ToolOutputManager

from .artifacts import ArtifactService, ArtifactServiceError, ToolResultArtifactization
from .attachments import HostResolvedAttachment, normalize_host_resolved_attachments
from .models import HandoffEnvelope
from .ports import ContextRepositoryError


_EPHEMERAL_EVENT_TYPES = frozenset(
    {
        "reasoning",
        "reasoning_delta",
        "token_delta",
        "content_delta",
        "message_delta",
        "response_delta",
    }
)
_LIFECYCLE_EVENT_TYPES = frozenset(
    {
        "run_started",
        "iteration_started",
        "response_received",
        "iteration_completed",
        "final_message",
        "run_completed",
        "run_failed",
        "run_cancelled",
        "run_canceled",
        "run_aborted",
        "run_max_iterations",
    }
)
_INTERACTION_EVENT_TYPES = frozenset(
    {
        "interaction_requested",
        "interaction_resolved",
        "tool_confirmation_requested",
        "human_input_requested",
        "continuation_request",
        "tool_confirmed",
        "tool_denied",
    }
)
_ARTIFACT_EVENT_TYPES = frozenset({"artifact_created", "artifact_updated"})
_SUBAGENT_EVENT_TYPES = frozenset(
    {
        "subagent_spawned",
        "subagent_started",
        "subagent_completed",
        "subagent_failed",
        "subagent_cancelled",
        "subagent_canceled",
        "subagent_handoff",
        "subagent_batch_started",
        "subagent_batch_joined",
        "subagent_clarification_requested",
        "subagent_return_handoff_started",
        "subagent_return_handoff_completed",
        "agent_thread_spawned",
        "agent_thread_completed",
        "agent_thread_failed",
        "agent_thread_closed",
    }
)
_SAFE_RUNTIME_FIELDS = frozenset(
    {
        "agent_id",
        "arguments",
        "artifact",
        "artifact_id",
        "batch_id",
        "call_id",
        "child_run_id",
        "code",
        "confirmation_id",
        "content",
        "error",
        "has_tool_calls",
        "interaction_id",
        "interaction_request",
        "iteration",
        "lineage",
        "message",
        "mode",
        "model",
        "parent_id",
        "parent_run_id",
        "plan_id",
        "provider",
        "reason",
        "recoverable",
        "request_id",
        "response",
        "response_id",
        "root_agent",
        "root_run_id",
        "status",
        "subagent_id",
        "template",
        "thread_id",
        "tool_call_id",
        "tool_name",
        "workflow_node_id",
        "workflow_step_count",
        "workflow_step_index",
    }
)


class SemanticEventProjectionError(RuntimeError):
    """A raw runtime event could not become one safe canonical journal event."""


class SemanticEventProjectionMode(StrEnum):
    """Whether callbacks are authoritative execution events or observations."""

    CANONICAL = "canonical"
    SHADOW_OBSERVED = "shadow_observed"


class SemanticPayloadSanitizer(Protocol):
    def __call__(
        self,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _bounded_iteration(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticEventProjectionError("runtime event iteration is invalid")
    return value


_SOURCE_TOOL_PROVIDERS = frozenset(
    {"openai", "anthropic", "hyperspace", "ollama", "gemini"}
)


def _normalized_source_provider(event: Mapping[str, Any]) -> str | None:
    if "source_provider" not in event:
        return None
    raw = event.get("source_provider")
    if type(raw) is not str:
        raise SemanticEventProjectionError("tool call source provider is invalid")
    provider = raw.strip().casefold()
    if provider not in _SOURCE_TOOL_PROVIDERS:
        raise SemanticEventProjectionError("tool call source provider is unsupported")
    return provider


_SHADOW_OBSERVATION = {
    "schema": "unchain.shadow_observed_tool_event.v1",
    "mode": "shadow",
    "observed": True,
    "authoritative": False,
    "source": "legacy_runtime_callback",
}


class ShadowObservedToolEventAdapter:
    """Project legacy shadow callbacks without claiming execution authority."""

    def __init__(
        self,
        *,
        attempt: AttemptRef,
        journal: BoundToolReceiptIndex,
        artifacts: ArtifactService,
        payload_sanitizer: SemanticPayloadSanitizer,
    ) -> None:
        if not isinstance(attempt, AttemptRef):
            attempt = AttemptRef.from_dict(attempt)
        if not isinstance(journal, BoundToolReceiptIndex):
            raise TypeError(
                "shadow observed tool events require a durable receipt index"
            )
        if not isinstance(artifacts, ArtifactService):
            raise TypeError("artifacts must be an ArtifactService")
        execution_id = attempt.generation.execution_id
        if journal.execution_id != execution_id or artifacts.execution_id != execution_id:
            raise SemanticEventProjectionError(
                "shadow observed tool capabilities crossed the bound execution"
            )
        if not callable(payload_sanitizer):
            raise TypeError("payload_sanitizer must be callable")
        self._attempt = attempt
        self._journal = journal
        self._artifacts = artifacts
        self._payload_sanitizer = payload_sanitizer

    @property
    def attempt(self) -> AttemptRef:
        return self._attempt

    @property
    def artifacts(self) -> ArtifactService:
        return self._artifacts

    @staticmethod
    def _tool_identity(event: Mapping[str, Any]) -> tuple[str, str]:
        try:
            return (
                _required_text(
                    event.get("call_id") or event.get("tool_call_id"),
                    "call_id",
                    identifier=True,
                ),
                _required_text(
                    event.get("tool_name"),
                    "tool_name",
                    identifier=True,
                ),
            )
        except (TypeError, ValueError) as exc:
            raise SemanticEventProjectionError(
                "shadow observed tool boundary requires exact call and tool identity"
            ) from exc

    def _sanitize(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        protected: Mapping[str, Any],
    ) -> dict[str, Any]:
        sanitized = self._payload_sanitizer(
            event_type,
            copy.deepcopy(payload),
        )
        if not isinstance(sanitized, Mapping):
            raise SemanticEventProjectionError(
                "payload sanitizer must return an object"
            )
        result = copy.deepcopy(dict(sanitized))
        for key, expected in protected.items():
            if result.get(key) != expected:
                raise SemanticEventProjectionError(
                    f"payload sanitizer changed protected {key}"
                )
        return result

    def _draft(
        self,
        *,
        event_type: str,
        call_id: str,
        payload: Mapping[str, Any],
        resource_refs: tuple[ResourceRef, ...] = (),
    ) -> SemanticEventDraft:
        identity = {
            "attempt": self._attempt.to_dict(),
            "event_type": event_type,
            "discriminator": {"call_id": call_id},
        }
        digest = _stable_digest(identity)
        return SemanticEventDraft(
            event_id="event-" + digest,
            event_type=event_type,
            attempt=self._attempt,
            operation_id="operation-" + digest,
            payload=payload,
            resource_refs=resource_refs,
        )

    def project_tool_call(self, event: Mapping[str, Any]) -> SemanticEventDraft:
        call_id, tool_name = self._tool_identity(event)
        iteration = _bounded_iteration(event.get("iteration"))
        source_provider = _normalized_source_provider(event)
        protected = {
            "run_id": self._attempt.attempt_id,
            "tool_name": tool_name,
            "call_id": call_id,
            "observation": copy.deepcopy(_SHADOW_OBSERVATION),
            **(
                {"source_provider": source_provider}
                if source_provider is not None
                else {}
            ),
        }
        payload = {
            **protected,
            "iteration": iteration,
            "arguments": copy.deepcopy(event.get("arguments") or {}),
        }
        return self._draft(
            event_type="tool_call",
            call_id=call_id,
            payload=self._sanitize(
                "tool_call",
                payload,
                protected=protected,
            ),
        )

    def _require_observed_call_pair(
        self,
        *,
        call_id: str,
        tool_name: str,
        iteration: int | None,
    ) -> None:
        lookup = self._journal.lookup_tool_execution_receipts(
            attempt=self._attempt,
            call_id=call_id,
        )
        if (
            lookup.attempt != self._attempt
            or lookup.call_id != call_id
            or lookup.overflow
        ):
            raise SemanticEventProjectionError(
                "shadow observed tool receipt lookup is invalid"
            )
        calls = [event for event in lookup.events if event.event_type == "tool_call"]
        results = [event for event in lookup.events if event.event_type == "tool_result"]
        if len(calls) != 1 or len(results) > 1 or len(calls) + len(results) != len(
            lookup.events
        ):
            raise SemanticEventProjectionError(
                "shadow observed tool call pair is missing or ambiguous"
            )
        for receipt in (*calls, *results):
            if (
                receipt.payload.get("tool_name") != tool_name
                or receipt.payload.get("iteration") != iteration
                or receipt.payload.get("observation") != _SHADOW_OBSERVATION
            ):
                raise SemanticEventProjectionError(
                    "shadow observed tool call pair changed identity"
                )

    def project_tool_result(self, event: Mapping[str, Any]) -> SemanticEventDraft:
        call_id, tool_name = self._tool_identity(event)
        iteration = _bounded_iteration(event.get("iteration"))
        self._require_observed_call_pair(
            call_id=call_id,
            tool_name=tool_name,
            iteration=iteration,
        )
        identity = {
            "attempt": self._attempt.to_dict(),
            "event_type": "tool_result",
            "discriminator": {"call_id": call_id},
        }
        artifactization = self._artifacts.artifactize_tool_result(
            event.get("result"),
            operation_id="artifact.tool-result." + _stable_digest(identity),
        )
        result_fields = artifactization.event_fields()
        protected = {
            "run_id": self._attempt.attempt_id,
            "tool_name": tool_name,
            "call_id": call_id,
            "observation": copy.deepcopy(_SHADOW_OBSERVATION),
            "result": result_fields["result"],
            "full_output_ref": artifactization.full_output_ref.to_dict(),
            "result_bytes": artifactization.result_bytes,
            "result_sha256": artifactization.result_sha256,
        }
        payload = {
            **protected,
            "iteration": iteration,
            "preview": artifactization.artifact.preview,
            "preview_truncated": (
                artifactization.result_bytes
                > len(artifactization.artifact.preview.encode("utf-8"))
            ),
        }
        return self._draft(
            event_type="tool_result",
            call_id=call_id,
            payload=self._sanitize(
                "tool_result",
                payload,
                protected=protected,
            ),
            resource_refs=(artifactization.full_output_ref,),
        )


class CanonicalSemanticEventProjector:
    """Project one execution's raw callbacks into durable semantic receipts.

    Tool output bytes are sanitized and persisted by ``ArtifactService`` before
    the returned draft can be appended to the execution journal.
    """

    def __init__(
        self,
        *,
        attempt: AttemptRef,
        artifacts: ArtifactService,
        payload_sanitizer: SemanticPayloadSanitizer,
        observed_tool_adapter: ShadowObservedToolEventAdapter | None = None,
    ) -> None:
        if not isinstance(attempt, AttemptRef):
            attempt = AttemptRef.from_dict(attempt)
        if not isinstance(artifacts, ArtifactService):
            raise TypeError("artifacts must be an ArtifactService")
        if artifacts.execution_id != attempt.generation.execution_id:
            raise SemanticEventProjectionError(
                "artifact service execution does not match the bound attempt"
            )
        if not callable(payload_sanitizer):
            raise TypeError("payload_sanitizer must be callable")
        if observed_tool_adapter is not None:
            if type(observed_tool_adapter) is not ShadowObservedToolEventAdapter:
                raise TypeError(
                    "observed_tool_adapter must be a ShadowObservedToolEventAdapter"
                )
            if (
                observed_tool_adapter.attempt != attempt
                or observed_tool_adapter.artifacts is not artifacts
            ):
                raise SemanticEventProjectionError(
                    "observed tool adapter does not share the projector boundary"
                )
        self._attempt = attempt
        self._artifacts = artifacts
        self._payload_sanitizer = payload_sanitizer
        self._observed_tool_adapter = observed_tool_adapter
        self._tool_output_manager: ToolOutputManager | None = None

    @property
    def attempt(self) -> AttemptRef:
        return self._attempt

    @property
    def artifacts(self) -> ArtifactService:
        return self._artifacts

    @property
    def projection_mode(self) -> SemanticEventProjectionMode:
        if self._observed_tool_adapter is not None:
            return SemanticEventProjectionMode.SHADOW_OBSERVED
        return SemanticEventProjectionMode.CANONICAL

    def bind_tool_output_manager(self, manager: ToolOutputManager) -> None:
        """Bind the one attempt-scoped model-visible tool-output owner."""

        if type(manager) is not ToolOutputManager:
            raise TypeError("tool output manager must be the official ToolOutputManager")
        current = self._tool_output_manager
        if current is not None and current is not manager:
            raise SemanticEventProjectionError(
                "tool output manager changed the projector attempt binding"
            )
        self._tool_output_manager = manager

    @property
    def bound_tool_output_manager(self) -> ToolOutputManager | None:
        """Return the manager fixed for this attempt, if bootstrap completed."""

        return self._tool_output_manager

    def _bound_tool_output_manager(self) -> ToolOutputManager:
        manager = self._tool_output_manager
        if manager is None:
            raise SemanticEventProjectionError(
                "canonical tool-result projection requires an attempt-bound output manager"
            )
        if not manager.active:
            raise SemanticEventProjectionError(
                "canonical tool-result projection requires an active output manager"
            )
        return manager

    def _project_artifactized_tool_result(
        self,
        artifactization: ToolResultArtifactization,
        *,
        call_id: str,
        requested_policy: Any = None,
    ) -> dict[str, Any]:
        content = self._artifacts.read_full(
            artifactization.artifact,
            remaining_budget_bytes=artifactization.result_bytes,
        )
        receipt = self._bound_tool_output_manager().project(
            content,
            full_output_ref=artifactization.full_output_ref.to_dict(),
            digest=artifactization.result_sha256,
            content_bytes=artifactization.result_bytes,
            requested_policy=requested_policy,
            call_id=call_id,
        )
        return {
            "result": receipt.payload,
            "full_output_ref": artifactization.full_output_ref.to_dict(),
            "result_bytes": artifactization.result_bytes,
            "result_sha256": artifactization.result_sha256,
            "result_projection": receipt.metadata,
        }

    def __call__(self, event: Mapping[str, Any]) -> SemanticEventDraft | None:
        if not isinstance(event, Mapping):
            raise TypeError("runtime event must be an object")
        event_type = str(event.get("type") or "").strip()
        if not event_type:
            raise SemanticEventProjectionError("runtime event type is missing")
        if event_type in _EPHEMERAL_EVENT_TYPES:
            return None
        self._require_bound_run(event)
        if event_type in {"interaction_requested", "interaction.requested"} or (
            event_type == "human_input_requested"
            and isinstance(event.get("interaction_request"), Mapping)
        ):
            return self._project_interaction_requested(event)
        if event_type == "tool_call":
            if self._observed_tool_adapter is not None:
                return self._observed_tool_adapter.project_tool_call(event)
            return self._project_tool_call(event)
        if event_type == "tool.started":
            return self._project_tool_started(event)
        if event_type == "tool_result":
            if event.get("interaction_resume_replay") is True:
                self._validate_interaction_resume_replay(event)
                return None
            if self._observed_tool_adapter is not None:
                return self._observed_tool_adapter.project_tool_result(event)
            return self._project_tool_result(event)
        if event_type in (
            _LIFECYCLE_EVENT_TYPES
            | _INTERACTION_EVENT_TYPES
            | _ARTIFACT_EVENT_TYPES
            | _SUBAGENT_EVENT_TYPES
        ):
            return self._project_bounded_event(event_type, event)
        return self._project_unknown_event(event_type, event)

    def project_user_message(
        self,
        message: Mapping[str, Any],
        *,
        message_index: int = 0,
        attachments: tuple[HostResolvedAttachment, ...] = (),
    ) -> SemanticEventDraft:
        normalized_attachments = normalize_host_resolved_attachments(attachments)
        if (
            not isinstance(message, Mapping)
            or set(message) != {"role", "content"}
            or message.get("role") != "user"
            or not isinstance(message.get("content"), str)
            or (
                not str(message.get("content") or "").strip()
                and not normalized_attachments
            )
        ):
            raise SemanticEventProjectionError(
                "user message must contain canonical text or one attachment"
            )
        if (
            isinstance(message_index, bool)
            or not isinstance(message_index, int)
            or message_index < 0
        ):
            raise SemanticEventProjectionError("message_index is invalid")
        identity = self._identity(
            "message.user",
            {"message_index": message_index},
        )
        for attachment in normalized_attachments:
            try:
                self._artifacts.read_page(
                    attachment.artifact,
                    offset=0,
                    limit=1,
                )
            except (ArtifactServiceError, ContextRepositoryError) as exc:
                raise SemanticEventProjectionError(
                    "attachment is not verified by the bound artifact service"
                ) from exc
        canonical_message = dict(message)
        attachment_envelopes = [
            attachment.to_dict() for attachment in normalized_attachments
        ]
        if attachment_envelopes:
            canonical_message["attachments"] = attachment_envelopes
        artifact, sanitized_message, content = self._artifacts.artifactize_user_message(
            canonical_message,
            operation_id="artifact.user-message." + _stable_digest(identity),
            operation_binding={
                "kind": "current_user_message",
                "attempt": self._attempt.to_dict(),
                "message_index": message_index,
            },
        )
        if (
            not isinstance(sanitized_message, dict)
            or sanitized_message.get("role") != "user"
            or not isinstance(sanitized_message.get("content"), str)
            or sanitized_message != canonical_message
            or (
                not str(sanitized_message.get("content") or "").strip()
                and not attachment_envelopes
            )
        ):
            raise SemanticEventProjectionError(
                "sanitized user message is not canonical input"
            )
        payload_source = {
            "run_id": self._attempt.attempt_id,
            "message": sanitized_message,
            "content_ref": artifact.ref.to_dict(),
            "content_bytes": len(content),
            "content_sha256": artifact.sha256,
            "preview": artifact.preview,
            "preview_truncated": len(content)
            > len(artifact.preview.encode("utf-8")),
        }
        protected = {
            "run_id": self._attempt.attempt_id,
            "message": sanitized_message,
            "content_ref": artifact.ref.to_dict(),
            "content_bytes": len(content),
            "content_sha256": artifact.sha256,
        }
        attachment_refs = tuple(
            attachment.artifact.ref for attachment in normalized_attachments
        )
        if attachment_envelopes:
            payload_source["attachments"] = attachment_envelopes
            payload_source["attachment_refs"] = [
                ref.to_dict() for ref in attachment_refs
            ]
            protected["attachments"] = attachment_envelopes
            protected["attachment_refs"] = [
                ref.to_dict() for ref in attachment_refs
            ]
        payload = self._sanitize_payload(
            "message.user",
            payload_source,
            protected=protected,
        )
        return self._draft(
            event_type="message.user",
            identity=identity,
            payload=payload,
            resource_refs=(artifact.ref, *attachment_refs),
        )

    def _interaction_resolution_identity(
        self,
        interaction_id: str,
        submitted_by: str,
    ) -> tuple[str, str, dict[str, Any]]:
        try:
            normalized_interaction_id = _required_text(
                interaction_id,
                "interaction_id",
                identifier=True,
            )
            normalized_submitted_by = _required_text(
                submitted_by,
                "submitted_by",
                identifier=True,
            )
        except (TypeError, ValueError) as exc:
            raise SemanticEventProjectionError(
                "interaction resolution identity is invalid"
            ) from exc
        identity = self._identity(
            "interaction.resolved",
            {"interaction_id": normalized_interaction_id},
        )
        return normalized_interaction_id, normalized_submitted_by, identity

    def _interaction_resolution_draft(
        self,
        *,
        identity: Mapping[str, Any],
        artifact: ArtifactRef,
        decoded: Any,
        content: bytes,
        normalized_interaction_id: str,
        normalized_submitted_by: str,
    ) -> SemanticEventDraft:
        if (
            not isinstance(decoded, dict)
            or set(decoded) != {"interaction_id", "response", "submitted_by"}
            or decoded.get("interaction_id") != normalized_interaction_id
            or decoded.get("submitted_by") != normalized_submitted_by
        ):
            raise SemanticEventProjectionError(
                "sanitized interaction resolution changed its identity"
            )
        payload = self._sanitize_payload(
            "interaction.resolved",
            {
                "run_id": self._attempt.attempt_id,
                "interaction_id": normalized_interaction_id,
                "submitted_by": normalized_submitted_by,
                "content_ref": artifact.ref.to_dict(),
                "content_bytes": len(content),
                "content_sha256": artifact.sha256,
                "preview": artifact.preview,
                "preview_truncated": len(content)
                > len(artifact.preview.encode("utf-8")),
            },
            protected={
                "run_id": self._attempt.attempt_id,
                "interaction_id": normalized_interaction_id,
                "submitted_by": normalized_submitted_by,
                "content_ref": artifact.ref.to_dict(),
                "content_bytes": len(content),
                "content_sha256": artifact.sha256,
            },
        )
        return self._draft(
            event_type="interaction.resolved",
            identity=identity,
            payload=payload,
            resource_refs=(artifact.ref,),
        )

    def prepare_interaction_resolution(
        self,
        *,
        interaction_id: str,
        response: Any,
        submitted_by: str = "user",
    ) -> PreparedSemanticEvent:
        """Compute the resolution's draft and artifact claim without writing.

        The caller commits both atomically through
        :meth:`~unchain.journal.runtime.DurableEventSink.append_prepared`.
        """

        normalized_interaction_id, normalized_submitted_by, identity = (
            self._interaction_resolution_identity(interaction_id, submitted_by)
        )
        prepared = self._artifacts.prepare_json_value(
            {
                "interaction_id": normalized_interaction_id,
                "response": copy.deepcopy(response),
                "submitted_by": normalized_submitted_by,
            },
            operation_id=(
                "artifact.interaction-resolution." + _stable_digest(identity)
            ),
            operation_binding={
                "kind": "interaction_resolution",
                "attempt": self._attempt.to_dict(),
                "interaction_id": normalized_interaction_id,
            },
        )
        draft = self._interaction_resolution_draft(
            identity=identity,
            artifact=prepared.artifact,
            decoded=prepared.decoded,
            content=prepared.sanitized,
            normalized_interaction_id=normalized_interaction_id,
            normalized_submitted_by=normalized_submitted_by,
        )
        return PreparedSemanticEvent(draft=draft, artifacts=(prepared.pending,))

    def project_interaction_resolution(
        self,
        *,
        interaction_id: str,
        response: Any,
        submitted_by: str = "user",
    ) -> SemanticEventDraft:
        prepared = self.prepare_interaction_resolution(
            interaction_id=interaction_id,
            response=response,
            submitted_by=submitted_by,
        )
        for pending in prepared.artifacts:
            self._artifacts.persist_pending(pending)
        return prepared.draft

    def _project_interaction_requested(
        self,
        event: Mapping[str, Any],
    ) -> SemanticEventDraft:
        interaction_request = event.get("interaction_request")
        if not isinstance(interaction_request, Mapping):
            raise SemanticEventProjectionError("interaction request payload is missing")
        try:
            nested_interaction_id = _required_text(
                interaction_request.get("interaction_id"),
                "interaction_id",
                identifier=True,
            )
            raw_interaction_id = event.get("interaction_id")
            interaction_id = (
                nested_interaction_id
                if raw_interaction_id is None
                else _required_text(
                    raw_interaction_id,
                    "interaction_id",
                    identifier=True,
                )
            )
        except (TypeError, ValueError) as exc:
            raise SemanticEventProjectionError(
                "interaction request identity is invalid"
            ) from exc
        if interaction_id != nested_interaction_id:
            raise SemanticEventProjectionError("interaction request identity changed")
        iteration = _bounded_iteration(event.get("iteration"))
        identity = self._identity(
            "interaction.requested",
            {"interaction_id": interaction_id},
        )
        payload = {
            "run_id": self._attempt.attempt_id,
            "iteration": iteration,
            "interaction_id": interaction_id,
            "interaction_request": copy.deepcopy(dict(interaction_request)),
        }
        return self._draft(
            event_type="interaction.requested",
            identity=identity,
            payload=self._sanitize_payload(
                "interaction.requested",
                payload,
                protected={
                    "run_id": self._attempt.attempt_id,
                    "interaction_id": interaction_id,
                },
            ),
        )

    def project_handoff_envelope(
        self,
        envelope: HandoffEnvelope,
    ) -> SemanticEventDraft:
        if not isinstance(envelope, HandoffEnvelope):
            envelope = HandoffEnvelope.from_dict(envelope)
        identity = self._identity(
            "handoff.recorded",
            {
                "child_attempt": envelope.child_attempt.to_dict(),
                "source_event_range": envelope.source_event_range.to_dict(),
            },
        )
        serialized = envelope.to_dict()
        payload = {
            "run_id": self._attempt.attempt_id,
            "child_run_id": envelope.child_run_id,
            "handoff_envelope": serialized,
        }
        refs = tuple(dict.fromkeys((envelope.full_output_ref, *envelope.artifact_refs)))
        return self._draft(
            event_type="handoff.recorded",
            identity=identity,
            payload=self._sanitize_payload(
                "handoff.recorded",
                payload,
                protected=payload,
            ),
            resource_refs=refs,
        )

    def _project_tool_call(
        self,
        event: Mapping[str, Any],
    ) -> SemanticEventDraft:
        call_id, tool_name = self._tool_identity(event)
        iteration = _bounded_iteration(event.get("iteration"))
        source_provider = _normalized_source_provider(event)
        identity = self._identity(
            "tool_call",
            {"call_id": call_id},
        )
        payload = {
            "run_id": self._attempt.attempt_id,
            "iteration": iteration,
            "tool_name": tool_name,
            "call_id": call_id,
            "arguments": copy.deepcopy(event.get("arguments") or {}),
            **(
                {"source_provider": source_provider}
                if source_provider is not None
                else {}
            ),
        }
        return self._draft(
            event_type="tool_call",
            identity=identity,
            payload=self._sanitize_payload(
                "tool_call",
                payload,
                protected={
                    "run_id": self._attempt.attempt_id,
                    "tool_name": tool_name,
                    "call_id": call_id,
                    **(
                        {"source_provider": source_provider}
                        if source_provider is not None
                        else {}
                    ),
                },
            ),
        )

    def _project_tool_result(
        self,
        event: Mapping[str, Any],
    ) -> SemanticEventDraft:
        call_id, tool_name = self._tool_identity(event)
        iteration = _bounded_iteration(event.get("iteration"))
        execution_subject, execution_subject_sha256 = self._tool_execution_subject(
            event
        )
        identity = self._identity(
            "tool_result",
            {"call_id": call_id},
        )
        artifactization = self._artifacts.artifactize_tool_result(
            event.get("result"),
            operation_id="artifact.tool-result." + _stable_digest(identity),
        )
        requested_policy = (
            event.get("tool_result_policy")
            or event.get("tool_result_projection_policy")
        )
        result_fields = artifactization.event_fields()
        if requested_policy is not None:
            result_fields = self._project_artifactized_tool_result(
                artifactization,
                call_id=call_id,
                requested_policy=requested_policy,
            )
        payload = {
            "run_id": self._attempt.attempt_id,
            "iteration": iteration,
            "tool_name": tool_name,
            "call_id": call_id,
            "execution_subject": execution_subject,
            "execution_subject_sha256": execution_subject_sha256,
            **result_fields,
            "preview": artifactization.artifact.preview,
            "preview_truncated": (
                artifactization.result_bytes
                > len(artifactization.artifact.preview.encode("utf-8"))
            ),
        }
        return self._draft(
            event_type="tool_result",
            identity=identity,
            payload=self._sanitize_payload(
                "tool_result",
                payload,
                protected={
                    "run_id": self._attempt.attempt_id,
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "execution_subject": execution_subject,
                    "execution_subject_sha256": execution_subject_sha256,
                    **result_fields,
                },
            ),
            resource_refs=(artifactization.full_output_ref,),
        )

    def _validate_interaction_resume_replay(
        self,
        event: Mapping[str, Any],
    ) -> None:
        """Reject malformed replay markers before omitting their duplicate effect."""

        call_id, tool_name = self._tool_identity(event)
        try:
            interaction_id = _required_text(
                event.get("interaction_id"),
                "interaction_id",
                identifier=True,
            )
        except (TypeError, ValueError) as exc:
            raise SemanticEventProjectionError(
                "interaction resume replay identity is invalid"
            ) from exc
        if tool_name != "ask_user_question" or not call_id or not interaction_id:
            raise SemanticEventProjectionError(
                "interaction resume replay identity is invalid"
            )

    def project_prepared_tool_result(
        self,
        event: Mapping[str, Any],
        *,
        artifactization: ToolResultArtifactization,
        completion_artifact: ArtifactRef,
    ) -> SemanticEventDraft:
        """Project an already-sanitized result and completion without rewriting either."""

        if type(artifactization) is not ToolResultArtifactization:
            raise TypeError("prepared tool result requires ToolResultArtifactization")
        if not isinstance(completion_artifact, ArtifactRef):
            completion_artifact = ArtifactRef.from_dict(completion_artifact)
        if completion_artifact.ref.kind != "artifact" or (
            completion_artifact.media_type != "application/json"
        ):
            raise SemanticEventProjectionError(
                "tool completion must be a whole JSON artifact"
            )
        call_id, tool_name = self._tool_identity(event)
        iteration = _bounded_iteration(event.get("iteration"))
        execution_subject, execution_subject_sha256 = self._tool_execution_subject(
            event
        )
        identity = self._identity(
            "tool_result",
            {"call_id": call_id},
        )
        # Durable execution receipts retain the verified artifact view.  Tool
        # output reduction happens only at the model-visible projection edge;
        # changing this receipt would invalidate its sealed-result verifier.
        result_fields = artifactization.event_fields()
        model_projection: dict[str, Any] | None = None
        manager = self.bound_tool_output_manager
        if manager is not None and manager.active:
            projection_fields = self._project_artifactized_tool_result(
                artifactization,
                call_id=call_id,
                requested_policy=(
                    event.get("tool_result_policy")
                    or event.get("tool_result_projection_policy")
                ),
            )
            model_projection = {
                "result": projection_fields["result"],
                "metadata": projection_fields["result_projection"],
            }
        completion_fields = {
            "completion_ref": completion_artifact.ref.to_dict(),
            "completion_bytes": completion_artifact.byte_length,
            "completion_sha256": completion_artifact.sha256,
            "completion_preview": completion_artifact.preview,
        }
        payload = {
            "run_id": self._attempt.attempt_id,
            "iteration": iteration,
            "tool_name": tool_name,
            "call_id": call_id,
            "execution_subject": execution_subject,
            "execution_subject_sha256": execution_subject_sha256,
            **result_fields,
            "preview": artifactization.artifact.preview,
            "preview_truncated": (
                artifactization.result_bytes
                > len(artifactization.artifact.preview.encode("utf-8"))
            ),
            **completion_fields,
            **(
                {"model_projection": model_projection}
                if model_projection is not None
                else {}
            ),
        }
        protected = {
            "run_id": self._attempt.attempt_id,
            "tool_name": tool_name,
            "call_id": call_id,
            "execution_subject": execution_subject,
            "execution_subject_sha256": execution_subject_sha256,
            **result_fields,
            **completion_fields,
            **(
                {"model_projection": model_projection}
                if model_projection is not None
                else {}
            ),
        }
        return self._draft(
            event_type="tool_result",
            identity=identity,
            payload=self._sanitize_payload(
                "tool_result",
                payload,
                protected=protected,
            ),
            resource_refs=(
                artifactization.full_output_ref,
                completion_artifact.ref,
            ),
        )

    def project_sealed_tool_completion(
        self,
        event: Mapping[str, Any],
        *,
        result_artifact: ArtifactRef,
        completion_artifact: ArtifactRef,
        transition: Mapping[str, Any],
        next_state_artifact: ArtifactRef,
        handoff_refs: tuple[ResourceRef, ...] = (),
    ) -> SemanticEventDraft:
        """Project the executor-owned pre-result seal for one stateful tool."""

        call_id, tool_name = self._tool_identity(event)
        iteration = _bounded_iteration(event.get("iteration"))
        execution_subject, execution_subject_sha256 = self._tool_execution_subject(
            event
        )
        artifacts = (result_artifact, completion_artifact, next_state_artifact)
        if any(
            type(artifact) is not ArtifactRef
            or artifact.ref.kind != "artifact"
            or artifact.ref.fragment
            or artifact.media_type != "application/json"
            for artifact in artifacts
        ):
            raise SemanticEventProjectionError(
                "sealed completion requires exact whole JSON artifacts"
            )
        refs = tuple(handoff_refs)
        if any(
            type(ref) is not ResourceRef or ref.kind != "artifact" or ref.fragment
            for ref in refs
        ):
            raise SemanticEventProjectionError(
                "sealed completion handoff refs are invalid"
            )
        payload = {
            "run_id": self._attempt.attempt_id,
            "iteration": iteration,
            "tool_name": tool_name,
            "call_id": call_id,
            "execution_subject": execution_subject,
            "execution_subject_sha256": execution_subject_sha256,
            "result_artifact": result_artifact.to_dict(),
            "completion_artifact": completion_artifact.to_dict(),
            "next_state_artifact": next_state_artifact.to_dict(),
            "transition": dict(transition),
            "handoff_refs": [ref.to_dict() for ref in refs],
        }
        identity = self._identity(
            "tool.subagent_completion.sealed",
            {"call_id": call_id},
        )
        return self._draft(
            event_type="tool.subagent_completion.sealed",
            identity=identity,
            payload=self._sanitize_payload(
                "tool.subagent_completion.sealed",
                payload,
                protected=payload,
            ),
            resource_refs=(
                result_artifact.ref,
                completion_artifact.ref,
                next_state_artifact.ref,
                *refs,
            ),
        )

    def _project_tool_started(
        self,
        event: Mapping[str, Any],
    ) -> SemanticEventDraft:
        call_id, tool_name = self._tool_identity(event)
        iteration = _bounded_iteration(event.get("iteration"))
        execution_subject, execution_subject_sha256 = self._tool_execution_subject(
            event
        )
        identity = self._identity(
            "tool.started",
            {"call_id": call_id},
        )
        payload = {
            "run_id": self._attempt.attempt_id,
            "iteration": iteration,
            "tool_name": tool_name,
            "call_id": call_id,
            "execution_subject": execution_subject,
            "execution_subject_sha256": execution_subject_sha256,
        }
        return self._draft(
            event_type="tool.started",
            identity=identity,
            payload=self._sanitize_payload(
                "tool.started",
                payload,
                protected=payload,
            ),
        )

    def _project_bounded_event(
        self,
        event_type: str,
        event: Mapping[str, Any],
    ) -> SemanticEventDraft:
        iteration = _bounded_iteration(event.get("iteration"))
        discriminator = self._event_discriminator(event_type, event)
        identity = self._identity(
            event_type,
            {"iteration": iteration, "discriminator": discriminator},
        )
        payload = {
            "run_id": self._attempt.attempt_id,
            **{
                key: copy.deepcopy(value)
                for key, value in event.items()
                if key in _SAFE_RUNTIME_FIELDS and key != "run_id"
            },
        }
        return self._draft(
            event_type=event_type,
            identity=identity,
            payload=self._sanitize_payload(
                event_type,
                payload,
                protected={"run_id": self._attempt.attempt_id},
            ),
        )

    def _project_unknown_event(
        self,
        event_type: str,
        event: Mapping[str, Any],
    ) -> SemanticEventDraft:
        iteration = _bounded_iteration(event.get("iteration"))
        identity = self._identity(
            "runtime_event",
            {"raw_type": event_type, "iteration": iteration},
        )
        payload = self._sanitize_payload(
            "runtime_event",
            {
                "run_id": self._attempt.attempt_id,
                "raw_type": event_type,
                "iteration": iteration,
            },
            protected={"run_id": self._attempt.attempt_id},
        )
        return self._draft(
            event_type="runtime_event",
            identity=identity,
            payload=payload,
        )

    def _sanitize_payload(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        protected: Mapping[str, Any],
    ) -> dict[str, Any]:
        sanitized = self._payload_sanitizer(event_type, copy.deepcopy(payload))
        if not isinstance(sanitized, Mapping):
            raise SemanticEventProjectionError(
                "payload sanitizer must return an object"
            )
        result = copy.deepcopy(dict(sanitized))
        for key, expected in protected.items():
            if result.get(key) != expected:
                raise SemanticEventProjectionError(
                    f"payload sanitizer changed protected {key}"
                )
        return result

    def _require_bound_run(self, event: Mapping[str, Any]) -> None:
        run_id = str(event.get("run_id") or "").strip()
        if run_id != self._attempt.attempt_id:
            raise SemanticEventProjectionError(
                "runtime event run does not match the bound attempt"
            )

    @staticmethod
    def _tool_identity(event: Mapping[str, Any]) -> tuple[str, str]:
        try:
            call_id = _required_text(
                event.get("call_id") or event.get("tool_call_id"),
                "call_id",
                identifier=True,
            )
            tool_name = _required_text(
                event.get("tool_name"),
                "tool_name",
                identifier=True,
            )
        except (TypeError, ValueError) as exc:
            raise SemanticEventProjectionError(
                "tool boundary requires exact call and tool identity"
            ) from exc
        return call_id, tool_name

    @staticmethod
    def _tool_execution_subject(
        event: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str]:
        subject = event.get("execution_subject")
        if not isinstance(subject, Mapping):
            raise SemanticEventProjectionError("tool execution subject is missing")
        try:
            from .tool_boundary import DurableToolExecutionSubject

            typed_subject = DurableToolExecutionSubject.from_dict(subject)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise SemanticEventProjectionError(
                "tool execution subject schema is invalid"
            ) from exc
        subject_copy = typed_subject.to_dict()
        try:
            digest = _sha256(
                event.get("execution_subject_sha256"),
                "execution_subject_sha256",
            )
        except (TypeError, ValueError) as exc:
            raise SemanticEventProjectionError(
                "tool execution subject digest is invalid"
            ) from exc
        if typed_subject.sha256 != digest:
            raise SemanticEventProjectionError(
                "tool execution subject digest does not match"
            )
        return subject_copy, digest

    @staticmethod
    def _event_discriminator(
        event_type: str,
        event: Mapping[str, Any],
    ) -> str:
        for key in (
            "interaction_id",
            "confirmation_id",
            "request_id",
            "artifact_id",
            "child_run_id",
            "batch_id",
            "thread_id",
            "call_id",
        ):
            value = str(event.get(key) or "").strip()
            if value:
                return f"{key}:{value}"
        return event_type

    def _identity(
        self,
        event_type: str,
        discriminator: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "attempt": self._attempt.to_dict(),
            "event_type": event_type,
            "discriminator": dict(discriminator),
        }

    def _draft(
        self,
        *,
        event_type: str,
        identity: Mapping[str, Any],
        payload: Mapping[str, Any],
        resource_refs: tuple[ResourceRef, ...] = (),
    ) -> SemanticEventDraft:
        digest = _stable_digest(identity)
        return SemanticEventDraft(
            event_id="event-" + digest,
            event_type=event_type,
            attempt=self._attempt,
            operation_id="operation-" + digest,
            payload=payload,
            resource_refs=resource_refs,
        )


__all__ = [
    "CanonicalSemanticEventProjector",
    "SemanticEventProjectionMode",
    "SemanticEventProjectionError",
    "SemanticPayloadSanitizer",
    "ShadowObservedToolEventAdapter",
]
