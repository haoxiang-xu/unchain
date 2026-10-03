"""Pure durable CAS contract for one provider request attempt.

This request-level fence is intentionally separate from the execution lease.
Concrete persistence and transport integration belong to host adapters.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from .failure_diagnostic import ProviderFailureDiagnostic
from .uncertainty_diagnostic import ProviderUncertaintyDiagnostic

from unchain.journal.models import (
    ArtifactRef,
    AttemptRef,
    EventCursor,
    GenerationRef,
    ModelValidationError,
    OperationRef,
    _record_data,
)


MAX_PROVIDER_REQUEST_ITERATION = 2**31 - 1
MAX_PROVIDER_REQUEST_RETRY_ORDINAL = 2**31 - 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROUTES = frozenset({"primary", "openai_previous_response_fallback"})
_FAILURE_CLASSIFICATIONS = frozenset(
    {"bad_request", "non_retryable", "previous_response_fallback", "transient"}
)


class ProviderRequestLeaseError(RuntimeError):
    """Base error for the durable provider-request fence."""


class ProviderRequestLeaseConflict(ProviderRequestLeaseError):
    """A subject was already claimed or its CAS revision changed."""


class ProviderRequestLeaseIntegrityError(ProviderRequestLeaseError):
    """A durable lease record contradicted the requested state transition."""


class ProviderRequestTransitionError(ProviderRequestLeaseError):
    """A request transition is not allowed by the durable predecessor."""


class ProviderRequestStatus(StrEnum):
    STARTED = "started"
    FAILED = "failed"
    COMPLETED = "completed"


class ProviderRequestDisposition(StrEnum):
    NOT_STARTED = "not_started"
    UNCERTAIN = "uncertain"
    RETRYABLE_FAILURE = "retryable_failure"
    TERMINAL = "terminal"


def _bounded_int(
    value: object,
    field_name: str,
    *,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an exact integer")
    if not 0 <= value <= maximum:
        raise ModelValidationError(f"{field_name} must be between 0 and {maximum}")
    return value


def _positive_revision(value: object) -> int:
    if type(value) is not int:
        raise TypeError("revision must be an exact integer")
    if value < 1:
        raise ModelValidationError("revision must be positive")
    return value


def _sha256(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ModelValidationError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: object) -> str:
    try:
        content = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise ModelValidationError(
            "provider request lease must be strict canonical JSON"
        ) from exc
    return hashlib.sha256(content).hexdigest()


def _canonical_text(value: object, field_name: str, *, maximum: int = 128) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be exact text")
    normalized = unicodedata.normalize("NFC", value.strip())
    if (
        normalized != value
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise ModelValidationError(f"{field_name} must be canonical text")
    return value


def _attempt(value: object) -> AttemptRef:
    if type(value) is dict:
        return AttemptRef.from_dict(value)
    if type(value) is not AttemptRef or type(value.generation) is not GenerationRef:
        raise TypeError("attempt must be an exact AttemptRef")
    return AttemptRef.from_dict(value.to_dict())


@dataclass(frozen=True, slots=True)
class ProviderRequestSubject:
    """The complete durable uniqueness key for one provider send."""

    SCHEMA: ClassVar[str] = "unchain.provider_request_subject.v1"

    attempt: AttemptRef
    iteration: int
    envelope_sha256: str
    route: str
    retry_ordinal: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt", _attempt(self.attempt))
        object.__setattr__(
            self,
            "iteration",
            _bounded_int(
                self.iteration,
                "iteration",
                maximum=MAX_PROVIDER_REQUEST_ITERATION,
            ),
        )
        object.__setattr__(
            self,
            "envelope_sha256",
            _sha256(self.envelope_sha256, "envelope_sha256"),
        )
        route = _canonical_text(self.route, "route")
        if route not in _ROUTES:
            raise ModelValidationError("route is unsupported")
        object.__setattr__(self, "route", route)
        object.__setattr__(
            self,
            "retry_ordinal",
            _bounded_int(
                self.retry_ordinal,
                "retry_ordinal",
                maximum=MAX_PROVIDER_REQUEST_RETRY_ORDINAL,
            ),
        )

    @property
    def key(self) -> tuple[AttemptRef, int, str, str, int]:
        return (
            self.attempt,
            self.iteration,
            self.envelope_sha256,
            self.route,
            self.retry_ordinal,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "attempt": self.attempt.to_dict(),
            "iteration": self.iteration,
            "envelope_sha256": self.envelope_sha256,
            "route": self.route,
            "retry_ordinal": self.retry_ordinal,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProviderRequestSubject:
        if type(value) is not dict:
            raise TypeError("provider request subject must be an exact dict")
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset(
                {
                    "attempt",
                    "iteration",
                    "envelope_sha256",
                    "route",
                    "retry_ordinal",
                }
            ),
        )
        return cls(
            attempt=AttemptRef.from_dict(raw["attempt"]),
            iteration=raw["iteration"],
            envelope_sha256=raw["envelope_sha256"],
            route=raw["route"],
            retry_ordinal=raw["retry_ordinal"],
        )


@dataclass(frozen=True, slots=True)
class ProviderTurnResultBinding:
    """Durable result reference required by a completed provider lease."""

    SCHEMA: ClassVar[str] = "unchain.provider_turn_result_binding.v1"

    route_sha256: str
    result_sha256: str
    artifact: ArtifactRef
    cursor: EventCursor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_sha256",
            _sha256(self.route_sha256, "route_sha256"),
        )
        object.__setattr__(
            self,
            "result_sha256",
            _sha256(self.result_sha256, "result_sha256"),
        )
        artifact = self.artifact
        if type(artifact) is dict:
            artifact = ArtifactRef.from_dict(artifact)
        elif type(artifact) is ArtifactRef:
            artifact = ArtifactRef.from_dict(artifact.to_dict())
        else:
            raise TypeError("artifact must be an exact ArtifactRef")
        if (
            artifact.ref.kind != "artifact"
            or artifact.ref.fragment
            or artifact.media_type != "application/json"
            or artifact.preview
        ):
            raise ModelValidationError(
                "provider result binding requires one unpreviewed JSON artifact"
            )
        object.__setattr__(self, "artifact", artifact)
        cursor = self.cursor
        if type(cursor) is dict:
            cursor = EventCursor.from_dict(cursor)
        elif type(cursor) is EventCursor:
            cursor = EventCursor.from_dict(cursor.to_dict())
        else:
            raise TypeError("cursor must be an exact EventCursor")
        object.__setattr__(self, "cursor", cursor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "route_sha256": self.route_sha256,
            "result_sha256": self.result_sha256,
            "artifact": self.artifact.to_dict(),
            "cursor": self.cursor.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProviderTurnResultBinding:
        if type(value) is not dict:
            raise TypeError("provider result binding must be an exact dict")
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"route_sha256", "result_sha256", "artifact", "cursor"}),
        )
        return cls(
            route_sha256=raw["route_sha256"],
            result_sha256=raw["result_sha256"],
            artifact=ArtifactRef.from_dict(raw["artifact"]),
            cursor=EventCursor.from_dict(raw["cursor"]),
        )


@dataclass(frozen=True, slots=True)
class ProviderRequestLease:
    """One revisioned durable state for a unique request subject."""

    SCHEMA: ClassVar[str] = "unchain.provider_request_lease.v2"
    # Preserve both branches' closed wire shapes. Historical v3 stores HTTP v1
    # or null-status response diagnostics; v4 stores either extended HTTP
    # failure diagnostics or uncertainty observations, distinguished exactly.
    DIAGNOSTIC_SCHEMA: ClassVar[str] = "unchain.provider_request_lease.v3"
    DIAGNOSTIC_SCHEMA_V4: ClassVar[str] = "unchain.provider_request_lease.v4"
    UNCERTAINTY_SCHEMA: ClassVar[str] = "unchain.provider_request_lease.v4"

    subject: ProviderRequestSubject
    route_sha256: str
    status: ProviderRequestStatus
    revision: int
    visible_output: bool
    retryable: bool
    classification: str
    operation: OperationRef
    result_binding: ProviderTurnResultBinding | None = None
    predecessor_sha256: str | None = None
    failure_diagnostic: ProviderFailureDiagnostic | None = None
    uncertainty_diagnostic: ProviderUncertaintyDiagnostic | None = None

    def __post_init__(self) -> None:
        diagnostic = self.failure_diagnostic
        if type(diagnostic) is dict:
            diagnostic = ProviderFailureDiagnostic.from_dict(diagnostic)
        elif diagnostic is not None and type(diagnostic) is not ProviderFailureDiagnostic:
            raise TypeError("failure diagnostic must be exact or null")
        object.__setattr__(self, "failure_diagnostic", diagnostic)
        uncertainty = self.uncertainty_diagnostic
        if type(uncertainty) is dict:
            uncertainty = ProviderUncertaintyDiagnostic.from_dict(uncertainty)
        elif uncertainty is not None and type(uncertainty) is not ProviderUncertaintyDiagnostic:
            raise TypeError("uncertainty diagnostic must be exact or null")
        object.__setattr__(self, "uncertainty_diagnostic", uncertainty)
        subject = self.subject
        if type(subject) is dict:
            subject = ProviderRequestSubject.from_dict(subject)
        elif type(subject) is ProviderRequestSubject:
            subject = ProviderRequestSubject.from_dict(subject.to_dict())
        else:
            raise TypeError("subject must be an exact ProviderRequestSubject")
        object.__setattr__(self, "subject", subject)
        object.__setattr__(
            self,
            "route_sha256",
            _sha256(self.route_sha256, "route_sha256"),
        )
        try:
            status = ProviderRequestStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("status is unsupported") from exc
        object.__setattr__(self, "status", status)
        if diagnostic is not None and status is not ProviderRequestStatus.FAILED:
            raise ModelValidationError("only failed leases can contain failure diagnostics")
        if uncertainty is not None and status is not ProviderRequestStatus.STARTED:
            raise ModelValidationError("only started leases can contain uncertainty diagnostics")
        object.__setattr__(self, "revision", _positive_revision(self.revision))
        if type(self.visible_output) is not bool:
            raise TypeError("visible_output must be an exact boolean")
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be an exact boolean")
        classification = self.classification
        if classification:
            classification = _canonical_text(
                classification,
                "classification",
            )
            if classification not in _FAILURE_CLASSIFICATIONS:
                raise ModelValidationError("classification is unsupported")
        elif type(classification) is not str:
            raise TypeError("classification must be exact text")
        object.__setattr__(self, "classification", classification)
        operation = self.operation
        if type(operation) is dict:
            operation = OperationRef.from_dict(operation)
        elif type(operation) is OperationRef:
            operation = OperationRef.from_dict(operation.to_dict())
        else:
            raise TypeError("operation must be an exact OperationRef")
        object.__setattr__(self, "operation", operation)
        result_binding = self.result_binding
        if type(result_binding) is dict:
            result_binding = ProviderTurnResultBinding.from_dict(result_binding)
        elif (
            result_binding is not None
            and type(result_binding) is not ProviderTurnResultBinding
        ):
            raise TypeError(
                "result_binding must be an exact ProviderTurnResultBinding or null"
            )
        if type(result_binding) is ProviderTurnResultBinding:
            result_binding = ProviderTurnResultBinding.from_dict(
                result_binding.to_dict()
            )
        object.__setattr__(self, "result_binding", result_binding)
        predecessor_sha256 = self.predecessor_sha256
        if predecessor_sha256 is not None:
            predecessor_sha256 = _sha256(
                predecessor_sha256,
                "predecessor_sha256",
            )
        object.__setattr__(self, "predecessor_sha256", predecessor_sha256)

        if status is ProviderRequestStatus.STARTED:
            if result_binding is not None:
                raise ModelValidationError(
                    "started request lease cannot contain a result binding"
                )
            if self.visible_output or self.retryable or classification:
                raise ModelValidationError(
                    "started request lease cannot contain terminal fields"
                )
            if predecessor_sha256 is not None:
                raise ModelValidationError(
                    "started request lease cannot contain a predecessor digest"
                )
        elif status is ProviderRequestStatus.FAILED:
            if result_binding is not None:
                raise ModelValidationError(
                    "failed request lease cannot contain a result binding"
                )
            if predecessor_sha256 is not None:
                raise ModelValidationError(
                    "failed request lease cannot contain a predecessor digest"
                )
            if not classification:
                raise ModelValidationError(
                    "failed request lease requires a classification"
                )
            if classification == "previous_response_fallback" and not self.retryable:
                raise ModelValidationError(
                    "previous_response_fallback must be retryable"
                )
        else:
            if self.retryable or classification:
                raise ModelValidationError(
                    "completed request lease cannot contain failure fields"
                )
            if result_binding is None:
                raise ModelValidationError(
                    "completed request lease requires a durable result binding"
                )
            if result_binding.route_sha256 != self.route_sha256:
                raise ModelValidationError(
                    "completed request result binding route changed"
                )
            if predecessor_sha256 is None:
                raise ModelValidationError(
                    "completed request lease requires a predecessor digest"
                )

    @property
    def lease_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        record = {
            "schema": self.SCHEMA,
            "subject": self.subject.to_dict(),
            "route_sha256": self.route_sha256,
            "status": self.status.value,
            "revision": self.revision,
            "visible_output": self.visible_output,
            "retryable": self.retryable,
            "classification": self.classification,
            "operation": self.operation.to_dict(),
            "result_binding": (
                self.result_binding.to_dict() if self.result_binding else None
            ),
            "predecessor_sha256": self.predecessor_sha256,
        }
        if self.failure_diagnostic is not None:
            diagnostic = self.failure_diagnostic.to_dict()
            record["schema"] = (
                self.DIAGNOSTIC_SCHEMA_V4
                if self.failure_diagnostic.http_status is not None
                and diagnostic["schema"] == ProviderFailureDiagnostic.SCHEMA_V2
                else self.DIAGNOSTIC_SCHEMA
            )
            record["failure_diagnostic"] = diagnostic
        if self.uncertainty_diagnostic is not None:
            record["schema"] = self.UNCERTAINTY_SCHEMA
            record["uncertainty_diagnostic"] = self.uncertainty_diagnostic.to_dict()
        return record

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProviderRequestLease:
        if type(value) is not dict:
            raise TypeError("provider request lease must be an exact dict")
        fields = frozenset(
            {
                "subject",
                "route_sha256",
                "status",
                "revision",
                "visible_output",
                "retryable",
                "classification",
                "operation",
                "result_binding",
                "predecessor_sha256",
            }
        )
        schema = value.get("schema")
        if schema == cls.UNCERTAINTY_SCHEMA and "uncertainty_diagnostic" in value:
            fields = fields | {"uncertainty_diagnostic"}
            if type(value.get("uncertainty_diagnostic")) is not dict:
                raise ModelValidationError("v4 started lease requires an uncertainty diagnostic")
            if value.get("status") != ProviderRequestStatus.STARTED:
                raise ModelValidationError("uncertainty lease must remain started")
        elif schema in {cls.DIAGNOSTIC_SCHEMA, cls.DIAGNOSTIC_SCHEMA_V4}:
            fields = fields | {"failure_diagnostic"}
            diagnostic = value.get("failure_diagnostic")
            if type(diagnostic) is not dict:
                raise ModelValidationError("failure lease requires a diagnostic")
            parsed = ProviderFailureDiagnostic.from_dict(diagnostic)
            expected = (
                cls.DIAGNOSTIC_SCHEMA_V4
                if parsed.http_status is not None
                and diagnostic["schema"] == ProviderFailureDiagnostic.SCHEMA_V2
                else cls.DIAGNOSTIC_SCHEMA
            )
            if schema != expected:
                raise ModelValidationError("failure lease schema does not match its diagnostic shape")
            if value.get("status") != ProviderRequestStatus.FAILED:
                raise ModelValidationError("failure diagnostic lease must be failed")
        else:
            schema = cls.SCHEMA
        raw = _record_data(value, schema=schema, required=fields)
        return cls(**{field_name: raw[field_name] for field_name in fields})

    @classmethod
    def from_durable_dict(cls, value: dict[str, Any]) -> ProviderRequestLease:
        """Read the current schema or safely upgrade a pre-result-binding lease."""

        if type(value) is not dict:
            raise TypeError("provider request lease must be an exact dict")
        if value.get("schema") in {
            cls.SCHEMA, cls.DIAGNOSTIC_SCHEMA, cls.DIAGNOSTIC_SCHEMA_V4,
            cls.UNCERTAINTY_SCHEMA,
        }:
            return cls.from_dict(value)
        legacy_schema = "unchain.provider_request_lease.v1"
        legacy_fields = frozenset(
            {
                "subject",
                "route_sha256",
                "status",
                "revision",
                "visible_output",
                "retryable",
                "classification",
                "operation",
            }
        )
        raw = _record_data(
            value,
            schema=legacy_schema,
            required=legacy_fields,
        )
        try:
            status = ProviderRequestStatus(raw["status"])
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("status is unsupported") from exc
        if status is ProviderRequestStatus.COMPLETED:
            raise ProviderRequestLeaseIntegrityError(
                "legacy completed request lease has no durable result binding"
            )
        return cls(
            **{field_name: raw[field_name] for field_name in legacy_fields},
            result_binding=None,
            predecessor_sha256=None,
        )


class ProviderRequestLeasePort(ABC):
    """Minimal atomic durable state capability; no repository implementation."""

    @abstractmethod
    def load(
        self,
        *,
        subject: ProviderRequestSubject,
    ) -> ProviderRequestLease | None:
        """Read the current state for the exact subject key."""

    @abstractmethod
    def compare_and_swap(
        self,
        *,
        subject: ProviderRequestSubject,
        expected_revision: int,
        replacement: ProviderRequestLease,
    ) -> ProviderRequestLease:
        """Atomically replace revision zero/expected with the next revision."""


@dataclass(frozen=True, slots=True)
class ProviderRequestRecovery:
    subject: ProviderRequestSubject
    disposition: ProviderRequestDisposition
    lease: ProviderRequestLease | None = None

    @property
    def auto_resend_allowed(self) -> bool:
        return self.disposition is ProviderRequestDisposition.NOT_STARTED


class ProviderRequestLeaseCoordinator:
    """Pure state machine over an atomic durable request-lease port."""

    def __init__(self, port: ProviderRequestLeasePort) -> None:
        if not isinstance(port, ProviderRequestLeasePort):
            raise TypeError("port must be a ProviderRequestLeasePort")
        self._port = port

    def _load(self, subject: ProviderRequestSubject) -> ProviderRequestLease | None:
        record = self._port.load(subject=subject)
        if record is None:
            return None
        if type(record) is not ProviderRequestLease:
            raise ProviderRequestLeaseIntegrityError(
                "durable request lease load returned a foreign record"
            )
        if record.subject != subject:
            raise ProviderRequestLeaseIntegrityError(
                "durable request lease subject changed"
            )
        return record

    def _current(self, expected: ProviderRequestLease) -> ProviderRequestLease:
        if type(expected) is not ProviderRequestLease:
            raise TypeError("lease must be an exact ProviderRequestLease")
        current = self._load(expected.subject)
        if current is None or current != expected:
            raise ProviderRequestLeaseIntegrityError(
                "current durable request lease changed"
            )
        return current

    def _cas(
        self,
        *,
        expected_revision: int,
        replacement: ProviderRequestLease,
    ) -> ProviderRequestLease:
        stored = self._port.compare_and_swap(
            subject=replacement.subject,
            expected_revision=expected_revision,
            replacement=replacement,
        )
        if type(stored) is not ProviderRequestLease or stored != replacement:
            raise ProviderRequestLeaseIntegrityError(
                "request lease CAS returned changed status, revision, or subject"
            )
        return stored

    def _claim(
        self,
        *,
        subject: ProviderRequestSubject,
        route_sha256: str,
        operation: OperationRef,
    ) -> ProviderRequestLease:
        if self._load(subject) is not None:
            raise ProviderRequestLeaseConflict(
                "provider request subject is already claimed"
            )
        replacement = ProviderRequestLease(
            subject=subject,
            route_sha256=route_sha256,
            status=ProviderRequestStatus.STARTED,
            revision=1,
            visible_output=False,
            retryable=False,
            classification="",
            operation=operation,
        )
        return self._cas(expected_revision=0, replacement=replacement)

    def claim_initial(
        self,
        *,
        attempt: AttemptRef,
        iteration: int,
        envelope_sha256: str,
        route: str,
        route_sha256: str,
        operation: OperationRef,
    ) -> ProviderRequestLease:
        if route != "primary":
            raise ProviderRequestTransitionError(
                "initial provider request must use the primary route"
            )
        subject = ProviderRequestSubject(
            attempt=attempt,
            iteration=iteration,
            envelope_sha256=envelope_sha256,
            route=route,
            retry_ordinal=0,
        )
        return self._claim(
            subject=subject,
            route_sha256=route_sha256,
            operation=operation,
        )

    def record_failure(
        self,
        lease: ProviderRequestLease,
        *,
        classification: str,
        retryable: bool,
        visible_output: bool,
        operation: OperationRef,
        failure_diagnostic: ProviderFailureDiagnostic | None = None,
    ) -> ProviderRequestLease:
        current = self._current(lease)
        if current.status is not ProviderRequestStatus.STARTED:
            raise ProviderRequestTransitionError(
                "only a started request can record a terminal failure"
            )
        if classification == "previous_response_fallback" and (
            current.subject.route != "primary" or current.subject.retry_ordinal != 0
        ):
            raise ProviderRequestTransitionError(
                "previous_response_fallback requires primary ordinal zero"
            )
        replacement = ProviderRequestLease(
            subject=current.subject,
            route_sha256=current.route_sha256,
            status=ProviderRequestStatus.FAILED,
            revision=current.revision + 1,
            visible_output=visible_output,
            retryable=retryable,
            classification=classification,
            operation=operation,
            failure_diagnostic=failure_diagnostic,
        )
        return self._cas(
            expected_revision=current.revision,
            replacement=replacement,
        )

    def record_uncertainty(
        self,
        lease: ProviderRequestLease,
        *,
        diagnostic: ProviderUncertaintyDiagnostic,
        operation: OperationRef,
    ) -> ProviderRequestLease:
        current = self._current(lease)
        if current.status is not ProviderRequestStatus.STARTED:
            raise ProviderRequestTransitionError("only a started request can record uncertainty")
        if current.uncertainty_diagnostic is not None:
            raise ProviderRequestTransitionError("uncertainty is already recorded")
        replacement = ProviderRequestLease(
            subject=current.subject, route_sha256=current.route_sha256,
            status=ProviderRequestStatus.STARTED, revision=current.revision + 1,
            visible_output=False, retryable=False, classification="",
            operation=operation, uncertainty_diagnostic=diagnostic,
        )
        return self._cas(expected_revision=current.revision, replacement=replacement)

    def record_completed(
        self,
        lease: ProviderRequestLease,
        *,
        visible_output: bool,
        operation: OperationRef,
    ) -> ProviderRequestLease:
        del lease, visible_output, operation
        raise ProviderRequestTransitionError(
            "completed provider request requires a durable result binding"
        )

    def record_completed_result(
        self,
        lease: ProviderRequestLease,
        *,
        result_binding: ProviderTurnResultBinding,
        visible_output: bool,
        operation: OperationRef,
    ) -> ProviderRequestLease:
        if type(lease) is not ProviderRequestLease:
            raise TypeError("lease must be an exact ProviderRequestLease")
        if type(result_binding) is not ProviderTurnResultBinding:
            raise TypeError("result_binding must be an exact ProviderTurnResultBinding")
        if result_binding.route_sha256 != lease.route_sha256:
            raise ProviderRequestLeaseIntegrityError(
                "provider result binding route changed"
            )
        current = self._load(lease.subject)
        if current is None:
            raise ProviderRequestLeaseIntegrityError(
                "current durable request lease is missing"
            )
        replacement = ProviderRequestLease(
            subject=lease.subject,
            route_sha256=lease.route_sha256,
            status=ProviderRequestStatus.COMPLETED,
            revision=lease.revision + 1,
            visible_output=visible_output,
            retryable=False,
            classification="",
            operation=operation,
            result_binding=result_binding,
            predecessor_sha256=lease.lease_sha256,
        )
        if current == replacement:
            return current
        if current != lease:
            raise ProviderRequestLeaseIntegrityError(
                "current completed durable request lease changed"
            )
        if current.status is not ProviderRequestStatus.STARTED:
            raise ProviderRequestTransitionError(
                "only a started request can record completed status"
            )
        return self._cas(
            expected_revision=current.revision,
            replacement=replacement,
        )

    def claim_retry(
        self,
        previous: ProviderRequestLease,
        *,
        operation: OperationRef,
    ) -> ProviderRequestLease:
        current = self._current(previous)
        if current.status is ProviderRequestStatus.COMPLETED:
            raise ProviderRequestTransitionError(
                "completed provider request cannot retry"
            )
        if current.status is not ProviderRequestStatus.FAILED:
            raise ProviderRequestTransitionError(
                "retry requires a durable terminal failure"
            )
        if current.visible_output:
            raise ProviderRequestTransitionError(
                "request with visible output cannot retry"
            )
        if not current.retryable:
            raise ProviderRequestTransitionError(
                "retry requires a durable retryable failure"
            )
        if current.classification != "transient":
            raise ProviderRequestTransitionError(
                "only transient failure may retry; route transition is separate"
            )
        if current.subject.retry_ordinal >= MAX_PROVIDER_REQUEST_RETRY_ORDINAL:
            raise ProviderRequestTransitionError("retry ordinal is exhausted")
        subject = ProviderRequestSubject(
            attempt=current.subject.attempt,
            iteration=current.subject.iteration,
            envelope_sha256=current.subject.envelope_sha256,
            route=current.subject.route,
            retry_ordinal=current.subject.retry_ordinal + 1,
        )
        return self._claim(
            subject=subject,
            route_sha256=current.route_sha256,
            operation=operation,
        )

    def claim_openai_fallback(
        self,
        primary: ProviderRequestLease,
        *,
        fallback_route_sha256: str,
        operation: OperationRef,
    ) -> ProviderRequestLease:
        current = self._current(primary)
        if (
            current.subject.route != "primary"
            or current.subject.retry_ordinal != 0
            or current.status is not ProviderRequestStatus.FAILED
            or not current.retryable
            or current.visible_output
            or current.classification != "previous_response_fallback"
        ):
            raise ProviderRequestTransitionError(
                "OpenAI fallback route requires durable primary fallback classification"
            )
        subject = ProviderRequestSubject(
            attempt=current.subject.attempt,
            iteration=current.subject.iteration,
            envelope_sha256=current.subject.envelope_sha256,
            route="openai_previous_response_fallback",
            retry_ordinal=0,
        )
        return self._claim(
            subject=subject,
            route_sha256=fallback_route_sha256,
            operation=operation,
        )

    def recover(self, subject: ProviderRequestSubject) -> ProviderRequestRecovery:
        if type(subject) is not ProviderRequestSubject:
            raise TypeError("subject must be an exact ProviderRequestSubject")
        lease = self._load(subject)
        if lease is None:
            disposition = ProviderRequestDisposition.NOT_STARTED
        elif lease.status is ProviderRequestStatus.STARTED:
            disposition = ProviderRequestDisposition.UNCERTAIN
        elif (
            lease.status is ProviderRequestStatus.FAILED
            and lease.retryable
            and not lease.visible_output
        ):
            disposition = ProviderRequestDisposition.RETRYABLE_FAILURE
        else:
            disposition = ProviderRequestDisposition.TERMINAL
        return ProviderRequestRecovery(
            subject=subject,
            disposition=disposition,
            lease=lease,
        )


__all__ = [
    "MAX_PROVIDER_REQUEST_ITERATION",
    "MAX_PROVIDER_REQUEST_RETRY_ORDINAL",
    "ProviderRequestDisposition",
    "ProviderRequestLease",
    "ProviderRequestLeaseConflict",
    "ProviderRequestLeaseCoordinator",
    "ProviderRequestLeaseError",
    "ProviderRequestLeaseIntegrityError",
    "ProviderRequestLeasePort",
    "ProviderRequestRecovery",
    "ProviderRequestStatus",
    "ProviderRequestSubject",
    "ProviderRequestTransitionError",
    "ProviderTurnResultBinding",
]
