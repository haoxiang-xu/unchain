"""Pure data primitives for durable, idempotent runtime interactions.

This module deliberately has no dependency on tools, memory, or the kernel.
Persistence layers can store the journal inside their existing session snapshot
and use the content digests here as optimistic-concurrency identities.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Final, Literal


INTERACTION_JOURNAL_KEY: Final = "interaction_journal"
INTERACTION_SCHEMA_VERSION: Final = 1
INTERACTION_JOURNAL_SCHEMA_VERSION: Final = 1

INTERACTION_KIND_HUMAN_INPUT: Final = "human_input"
INTERACTION_KIND_TOOL_APPROVAL: Final = "tool_approval"
INTERACTION_KIND_MAX_BUDGET: Final = "max_budget"
INTERACTION_KINDS: Final = frozenset(
    {
        INTERACTION_KIND_HUMAN_INPUT,
        INTERACTION_KIND_TOOL_APPROVAL,
        INTERACTION_KIND_MAX_BUDGET,
    }
)

InteractionKind = Literal["human_input", "tool_approval", "max_budget"]


class InteractionError(RuntimeError):
    """Base error for durable interaction data failures."""

    code = "interaction_error"


class DurableInteractionSettlementPending(InteractionError):
    """An exact interaction must finish durable settlement before its terminal.

    Hosts raise this only when an interrupted callback cannot yet project its
    canonical resolution. This is not an ownership-loss signal.
    """

    code = "durable_interaction_settlement_pending"

    def __init__(self, *, execution_id: str, attempt_id: str, interaction_id: str):
        self.execution_id = _required_text(execution_id, field_name="execution_id")
        self.attempt_id = _required_text(attempt_id, field_name="attempt_id")
        self.interaction_id = _required_text(interaction_id, field_name="interaction_id")
        super().__init__("Durable interaction settlement is pending")


class InteractionIntegrityError(InteractionError):
    """Raised when persisted interaction data is malformed or tampered with."""

    code = "interaction_integrity_error"


class InteractionNotPendingError(InteractionError):
    """Raised when an operation does not target the active interaction."""

    code = "interaction_not_pending"


class InteractionReceiptConflictError(InteractionError):
    """Raised when two different answers target one immutable request."""

    code = "interaction_receipt_conflict"


class InteractionAlreadyAppliedError(InteractionError):
    """Raised when an applied receipt is replaced by a different application."""

    code = "interaction_already_applied"


# Descriptive alias for callers that prefer the subsystem-qualified name.
DurableInteractionError = InteractionError


def _strict_json_copy(value: Any, *, path: str = "$") -> Any:
    """Return a deep JSON copy, rejecting lossy or non-canonical inputs."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise InteractionIntegrityError(
                f"{path} contains an invalid Unicode string"
            ) from exc
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InteractionIntegrityError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [
            _strict_json_copy(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise InteractionIntegrityError(
                    f"{path} contains a non-string object key"
                )
            copied[key] = _strict_json_copy(item, path=f"{path}.{key}")
        return copied
    raise InteractionIntegrityError(
        f"{path} contains a non-JSON value of type {type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _strict_json_copy(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise InteractionIntegrityError("value cannot be encoded as canonical JSON") from exc


def _digest(value: Any) -> str:
    try:
        encoded = _canonical_json(value).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InteractionIntegrityError("value cannot be encoded as UTF-8") from exc
    return hashlib.sha256(encoded).hexdigest()


def strict_json_digest(value: Any) -> str:
    """Return the canonical SHA-256 identity used by durable interactions."""

    return _digest(value)


def _required_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InteractionIntegrityError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InteractionIntegrityError(
            f"{field_name} contains invalid Unicode"
        ) from exc
    return normalized


def _non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InteractionIntegrityError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _require_exact_fields(
    raw: dict[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        detail: list[str] = []
        if missing:
            detail.append(f"missing={missing}")
        if unknown:
            detail.append(f"unknown={unknown}")
        raise InteractionIntegrityError(
            f"{label} fields do not match schema: {', '.join(detail)}"
        )


def _validate_schema_version(value: Any, *, label: str) -> int:
    if value != INTERACTION_SCHEMA_VERSION or isinstance(value, bool):
        raise InteractionIntegrityError(
            f"unsupported {label} schema_version: {value!r}"
        )
    return INTERACTION_SCHEMA_VERSION


def _request_core(
    *,
    session_id: str,
    kind: str,
    source_run_id: str,
    occurrence: str,
    payload: Any,
    response_contract: Any,
    schema_digest: str,
    created_revision: int,
    subject: Any,
) -> dict[str, Any]:
    return {
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "session_id": session_id,
        "kind": kind,
        "source_run_id": source_run_id,
        "occurrence": occurrence,
        "payload": _strict_json_copy(payload, path="$.payload"),
        "response_contract": _strict_json_copy(
            response_contract,
            path="$.response_contract",
        ),
        "schema_digest": schema_digest,
        "created_revision": created_revision,
        "subject": _strict_json_copy(subject, path="$.subject"),
    }


@dataclass(frozen=True, slots=True)
class InteractionRequest:
    schema_version: int
    interaction_id: str
    session_id: str
    kind: InteractionKind
    source_run_id: str
    occurrence: str
    payload: Any
    response_contract: Any
    schema_digest: str
    request_digest: str
    created_revision: int
    subject: Any = None

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, label="interaction request")
        session_id = _required_text(self.session_id, field_name="session_id")
        source_run_id = _required_text(
            self.source_run_id,
            field_name="source_run_id",
        )
        occurrence = _required_text(self.occurrence, field_name="occurrence")
        interaction_id = _required_text(
            self.interaction_id,
            field_name="interaction_id",
        )
        if self.kind not in INTERACTION_KINDS:
            raise InteractionIntegrityError(
                f"unsupported interaction kind: {self.kind!r}"
            )
        created_revision = _non_negative_int(
            self.created_revision,
            field_name="created_revision",
        )
        payload = _strict_json_copy(self.payload, path="$.payload")
        response_contract = _strict_json_copy(
            self.response_contract,
            path="$.response_contract",
        )
        subject = _strict_json_copy(self.subject, path="$.subject")
        schema_digest = _required_text(
            self.schema_digest,
            field_name="schema_digest",
        )
        request_digest = _required_text(
            self.request_digest,
            field_name="request_digest",
        )

        expected_schema_digest = _digest(response_contract)
        if schema_digest != expected_schema_digest:
            raise InteractionIntegrityError(
                "interaction request response schema digest mismatch"
            )
        core = _request_core(
            session_id=session_id,
            kind=self.kind,
            source_run_id=source_run_id,
            occurrence=occurrence,
            payload=payload,
            response_contract=response_contract,
            schema_digest=schema_digest,
            created_revision=created_revision,
            subject=subject,
        )
        expected_request_digest = _digest(core)
        if request_digest != expected_request_digest:
            raise InteractionIntegrityError("interaction request digest mismatch")
        expected_interaction_id = f"interaction_{expected_request_digest[:32]}"
        if interaction_id != expected_interaction_id:
            raise InteractionIntegrityError("interaction_id does not match request digest")

        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "source_run_id", source_run_id)
        object.__setattr__(self, "occurrence", occurrence)
        object.__setattr__(self, "interaction_id", interaction_id)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "response_contract", response_contract)
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "created_revision", created_revision)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interaction_id": self.interaction_id,
            "session_id": self.session_id,
            "kind": self.kind,
            "source_run_id": self.source_run_id,
            "occurrence": self.occurrence,
            "payload": copy.deepcopy(self.payload),
            "response_contract": copy.deepcopy(self.response_contract),
            "schema_digest": self.schema_digest,
            "request_digest": self.request_digest,
            "created_revision": self.created_revision,
            "subject": copy.deepcopy(self.subject),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "InteractionRequest":
        if isinstance(raw, cls):
            return cls(**raw.to_dict())
        if not isinstance(raw, dict):
            raise InteractionIntegrityError("interaction request must be an object")
        copied = _strict_json_copy(raw)
        _require_exact_fields(
            copied,
            {
                "schema_version",
                "interaction_id",
                "session_id",
                "kind",
                "source_run_id",
                "occurrence",
                "payload",
                "response_contract",
                "schema_digest",
                "request_digest",
                "created_revision",
                "subject",
            },
            label="interaction request",
        )
        return cls(**copied)


def build_interaction_request(
    *,
    session_id: str,
    kind: InteractionKind,
    source_run_id: str,
    occurrence: str,
    payload: Any,
    response_contract: Any,
    created_revision: int,
    subject: Any = None,
) -> InteractionRequest:
    normalized_session_id = _required_text(session_id, field_name="session_id")
    normalized_source_run_id = _required_text(
        source_run_id,
        field_name="source_run_id",
    )
    normalized_occurrence = _required_text(occurrence, field_name="occurrence")
    if kind not in INTERACTION_KINDS:
        raise InteractionIntegrityError(f"unsupported interaction kind: {kind!r}")
    normalized_revision = _non_negative_int(
        created_revision,
        field_name="created_revision",
    )
    normalized_payload = _strict_json_copy(payload, path="$.payload")
    normalized_contract = _strict_json_copy(
        response_contract,
        path="$.response_contract",
    )
    normalized_subject = _strict_json_copy(subject, path="$.subject")
    schema_digest = _digest(normalized_contract)
    core = _request_core(
        session_id=normalized_session_id,
        kind=kind,
        source_run_id=normalized_source_run_id,
        occurrence=normalized_occurrence,
        payload=normalized_payload,
        response_contract=normalized_contract,
        schema_digest=schema_digest,
        created_revision=normalized_revision,
        subject=normalized_subject,
    )
    request_digest = _digest(core)
    return InteractionRequest(
        **core,
        interaction_id=f"interaction_{request_digest[:32]}",
        request_digest=request_digest,
    )


def _receipt_core(
    *,
    interaction_id: str,
    request_digest: str,
    response: Any,
    response_digest: str,
    submitted_by: str,
) -> dict[str, Any]:
    return {
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "interaction_id": interaction_id,
        "request_digest": request_digest,
        "response": _strict_json_copy(response, path="$.response"),
        "response_digest": response_digest,
        "submitted_by": submitted_by,
    }


@dataclass(frozen=True, slots=True)
class InteractionReceipt:
    schema_version: int
    receipt_id: str
    interaction_id: str
    request_digest: str
    response: Any
    response_digest: str
    submitted_by: str
    receipt_digest: str
    submitted_at_ms: int

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, label="interaction receipt")
        interaction_id = _required_text(
            self.interaction_id,
            field_name="interaction_id",
        )
        request_digest = _required_text(
            self.request_digest,
            field_name="request_digest",
        )
        response = _strict_json_copy(self.response, path="$.response")
        response_digest = _required_text(
            self.response_digest,
            field_name="response_digest",
        )
        submitted_by = _required_text(
            self.submitted_by,
            field_name="submitted_by",
        )
        receipt_digest = _required_text(
            self.receipt_digest,
            field_name="receipt_digest",
        )
        receipt_id = _required_text(self.receipt_id, field_name="receipt_id")
        submitted_at_ms = _non_negative_int(
            self.submitted_at_ms,
            field_name="submitted_at_ms",
        )

        expected_response_digest = _digest(response)
        if response_digest != expected_response_digest:
            raise InteractionIntegrityError("interaction response digest mismatch")
        core = _receipt_core(
            interaction_id=interaction_id,
            request_digest=request_digest,
            response=response,
            response_digest=response_digest,
            submitted_by=submitted_by,
        )
        expected_receipt_digest = _digest(core)
        if receipt_digest != expected_receipt_digest:
            raise InteractionIntegrityError("interaction receipt digest mismatch")
        if receipt_id != f"receipt_{expected_receipt_digest[:32]}":
            raise InteractionIntegrityError("receipt_id does not match receipt digest")

        object.__setattr__(self, "interaction_id", interaction_id)
        object.__setattr__(self, "request_digest", request_digest)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "submitted_by", submitted_by)
        object.__setattr__(self, "submitted_at_ms", submitted_at_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "interaction_id": self.interaction_id,
            "request_digest": self.request_digest,
            "response": copy.deepcopy(self.response),
            "response_digest": self.response_digest,
            "submitted_by": self.submitted_by,
            "receipt_digest": self.receipt_digest,
            "submitted_at_ms": self.submitted_at_ms,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Any,
        *,
        request: InteractionRequest | dict[str, Any] | None = None,
    ) -> "InteractionReceipt":
        if isinstance(raw, cls):
            receipt = cls(**raw.to_dict())
        else:
            if not isinstance(raw, dict):
                raise InteractionIntegrityError("interaction receipt must be an object")
            copied = _strict_json_copy(raw)
            _require_exact_fields(
                copied,
                {
                    "schema_version",
                    "receipt_id",
                    "interaction_id",
                    "request_digest",
                    "response",
                    "response_digest",
                    "submitted_by",
                    "receipt_digest",
                    "submitted_at_ms",
                },
                label="interaction receipt",
            )
            receipt = cls(**copied)
        if request is not None:
            bound_request = InteractionRequest.from_dict(request)
            if (
                receipt.interaction_id != bound_request.interaction_id
                or receipt.request_digest != bound_request.request_digest
            ):
                raise InteractionIntegrityError(
                    "interaction receipt does not belong to the request"
                )
        return receipt


def build_interaction_receipt(
    request: InteractionRequest | dict[str, Any],
    response: Any,
    *,
    submitted_by: str = "user",
    submitted_at_ms: int,
) -> InteractionReceipt:
    bound_request = InteractionRequest.from_dict(request)
    normalized_response = _strict_json_copy(response, path="$.response")
    normalized_submitted_at_ms = _non_negative_int(
        submitted_at_ms,
        field_name="submitted_at_ms",
    )
    normalized_submitted_by = _required_text(
        submitted_by,
        field_name="submitted_by",
    )
    response_digest = _digest(normalized_response)
    core = _receipt_core(
        interaction_id=bound_request.interaction_id,
        request_digest=bound_request.request_digest,
        response=normalized_response,
        response_digest=response_digest,
        submitted_by=normalized_submitted_by,
    )
    receipt_digest = _digest(core)
    return InteractionReceipt(
        **core,
        receipt_id=f"receipt_{receipt_digest[:32]}",
        receipt_digest=receipt_digest,
        submitted_at_ms=normalized_submitted_at_ms,
    )


def validate_interaction_request(raw: Any) -> InteractionRequest:
    return InteractionRequest.from_dict(raw)


def validate_interaction_receipt(
    raw: Any,
    *,
    request: InteractionRequest | dict[str, Any] | None = None,
) -> InteractionReceipt:
    return InteractionReceipt.from_dict(raw, request=request)


def new_interaction_journal() -> dict[str, Any]:
    return {
        "schema_version": INTERACTION_JOURNAL_SCHEMA_VERSION,
        "active_id": None,
        "entries": {},
        "order": [],
    }


def _validate_application(
    raw: Any,
    *,
    receipt: InteractionReceipt,
) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise InteractionIntegrityError("interaction application must be an object")
    application = _strict_json_copy(raw)
    _require_exact_fields(
        application,
        {"schema_version", "receipt_id", "applied_checkpoint_id"},
        label="interaction application",
    )
    _validate_schema_version(
        application.get("schema_version"),
        label="interaction application",
    )
    receipt_id = _required_text(
        application.get("receipt_id"),
        field_name="application.receipt_id",
    )
    if receipt_id != receipt.receipt_id:
        raise InteractionIntegrityError(
            "interaction application does not match its receipt"
        )
    return {
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "receipt_id": receipt_id,
        "applied_checkpoint_id": _required_text(
            application.get("applied_checkpoint_id"),
            field_name="application.applied_checkpoint_id",
        ),
    }


def validate_interaction_journal(raw: Any) -> dict[str, Any]:
    if raw is None:
        return new_interaction_journal()
    if not isinstance(raw, dict):
        raise InteractionIntegrityError("interaction journal must be an object")
    journal = _strict_json_copy(raw)
    _require_exact_fields(
        journal,
        {"schema_version", "active_id", "entries", "order"},
        label="interaction journal",
    )
    if (
        journal.get("schema_version") != INTERACTION_JOURNAL_SCHEMA_VERSION
        or isinstance(journal.get("schema_version"), bool)
    ):
        raise InteractionIntegrityError(
            "unsupported interaction journal schema_version: "
            f"{journal.get('schema_version')!r}"
        )
    entries_raw = journal.get("entries")
    order_raw = journal.get("order")
    active_id_raw = journal.get("active_id")
    if not isinstance(entries_raw, dict):
        raise InteractionIntegrityError("interaction journal entries must be an object")
    if not isinstance(order_raw, list) or any(
        not isinstance(item, str) or not item for item in order_raw
    ):
        raise InteractionIntegrityError(
            "interaction journal order must contain non-empty interaction ids"
        )
    if len(set(order_raw)) != len(order_raw):
        raise InteractionIntegrityError("interaction journal order contains duplicates")
    if active_id_raw is not None and (
        not isinstance(active_id_raw, str) or not active_id_raw
    ):
        raise InteractionIntegrityError(
            "interaction journal active_id must be a non-empty string or null"
        )

    entries: dict[str, dict[str, Any]] = {}
    pending_ids: list[str] = []
    for interaction_id, raw_entry in entries_raw.items():
        if not isinstance(interaction_id, str) or not interaction_id:
            raise InteractionIntegrityError(
                "interaction journal entry ids must be non-empty strings"
            )
        if not isinstance(raw_entry, dict):
            raise InteractionIntegrityError("interaction journal entry must be an object")
        entry = _strict_json_copy(raw_entry)
        _require_exact_fields(
            entry,
            {"request", "checkpoint_id", "receipt", "application"},
            label="interaction journal entry",
        )
        request = InteractionRequest.from_dict(entry.get("request"))
        if request.interaction_id != interaction_id:
            raise InteractionIntegrityError(
                "interaction journal entry id does not match its request"
            )
        checkpoint_id = _required_text(
            entry.get("checkpoint_id"),
            field_name="checkpoint_id",
        )
        raw_receipt = entry.get("receipt")
        receipt = (
            InteractionReceipt.from_dict(raw_receipt, request=request)
            if raw_receipt is not None
            else None
        )
        if entry.get("application") is not None and receipt is None:
            raise InteractionIntegrityError(
                "interaction application requires a receipt"
            )
        application = (
            _validate_application(entry.get("application"), receipt=receipt)
            if receipt is not None
            else None
        )
        if application is None:
            pending_ids.append(interaction_id)
        entries[interaction_id] = {
            "request": request.to_dict(),
            "checkpoint_id": checkpoint_id,
            "receipt": receipt.to_dict() if receipt is not None else None,
            "application": application,
        }

    if set(order_raw) != set(entries):
        raise InteractionIntegrityError(
            "interaction journal order must list every entry exactly once"
        )
    if active_id_raw is not None and active_id_raw not in entries:
        raise InteractionIntegrityError(
            "interaction journal active_id does not identify an entry"
        )
    expected_pending = [active_id_raw] if active_id_raw is not None else []
    if pending_ids != expected_pending:
        raise InteractionIntegrityError(
            "interaction journal active_id does not match pending entries"
        )

    return {
        "schema_version": INTERACTION_JOURNAL_SCHEMA_VERSION,
        "active_id": active_id_raw,
        "entries": entries,
        "order": list(order_raw),
    }


def register_interaction_request(
    journal: Any,
    request: InteractionRequest | dict[str, Any],
    *,
    checkpoint_id: str,
) -> dict[str, Any]:
    validated = validate_interaction_journal(journal)
    bound_request = InteractionRequest.from_dict(request)
    normalized_checkpoint_id = _required_text(
        checkpoint_id,
        field_name="checkpoint_id",
    )
    interaction_id = bound_request.interaction_id
    existing = validated["entries"].get(interaction_id)
    if existing is not None:
        if (
            existing["request"] == bound_request.to_dict()
            and existing["checkpoint_id"] == normalized_checkpoint_id
        ):
            return validated
        raise InteractionIntegrityError(
            "interaction_id is already registered with different request data"
        )

    active_id = validated.get("active_id")
    if active_id is not None and active_id != interaction_id:
        raise InteractionNotPendingError(
            f"another interaction is already pending: {active_id!r}"
        )
    validated["entries"][interaction_id] = {
        "request": bound_request.to_dict(),
        "checkpoint_id": normalized_checkpoint_id,
        "receipt": None,
        "application": None,
    }
    validated["active_id"] = interaction_id
    validated["order"].append(interaction_id)
    return validate_interaction_journal(validated)


def get_active_interaction(journal: Any) -> dict[str, Any] | None:
    validated = validate_interaction_journal(journal)
    active_id = validated.get("active_id")
    if active_id is None:
        return None
    return copy.deepcopy(validated["entries"][active_id])


def record_interaction_receipt(
    journal: Any,
    receipt: InteractionReceipt | dict[str, Any],
) -> dict[str, Any]:
    validated = validate_interaction_journal(journal)
    bound_receipt = InteractionReceipt.from_dict(receipt)
    interaction_id = bound_receipt.interaction_id
    entry = validated["entries"].get(interaction_id)
    if entry is None:
        raise InteractionNotPendingError(
            f"interaction is not pending: {interaction_id!r}"
        )
    request = InteractionRequest.from_dict(entry["request"])
    InteractionReceipt.from_dict(bound_receipt, request=request)
    existing_application = entry.get("application")
    existing_receipt_raw = entry.get("receipt")
    if existing_receipt_raw is not None:
        existing_receipt = InteractionReceipt.from_dict(
            existing_receipt_raw,
            request=request,
        )
        if existing_receipt.receipt_digest == bound_receipt.receipt_digest:
            return validated
        if existing_application is not None:
            raise InteractionAlreadyAppliedError(
                f"interaction receipt is already applied: {interaction_id!r}"
            )
        raise InteractionReceiptConflictError(
            f"interaction already has a different receipt: {interaction_id!r}"
        )
    if existing_application is not None:
        raise InteractionIntegrityError(
            "interaction application exists without a receipt"
        )
    if validated.get("active_id") != interaction_id:
        raise InteractionNotPendingError(
            f"interaction is no longer pending: {interaction_id!r}"
        )
    entry["receipt"] = bound_receipt.to_dict()
    return validate_interaction_journal(validated)


def mark_interaction_applied(
    journal: Any,
    *,
    interaction_id: str,
    receipt_id: str,
    applied_checkpoint_id: str,
) -> dict[str, Any]:
    validated = validate_interaction_journal(journal)
    normalized_interaction_id = _required_text(
        interaction_id,
        field_name="interaction_id",
    )
    normalized_receipt_id = _required_text(
        receipt_id,
        field_name="receipt_id",
    )
    normalized_checkpoint_id = _required_text(
        applied_checkpoint_id,
        field_name="applied_checkpoint_id",
    )
    entry = validated["entries"].get(normalized_interaction_id)
    if entry is None:
        raise InteractionNotPendingError(
            f"interaction is not pending: {normalized_interaction_id!r}"
        )
    application = {
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "receipt_id": normalized_receipt_id,
        "applied_checkpoint_id": normalized_checkpoint_id,
    }
    existing_application = entry.get("application")
    if existing_application is not None:
        if existing_application == application:
            return validated
        raise InteractionAlreadyAppliedError(
            f"interaction is already applied: {normalized_interaction_id!r}"
        )
    if validated.get("active_id") != normalized_interaction_id:
        raise InteractionNotPendingError(
            f"interaction is no longer pending: {normalized_interaction_id!r}"
        )
    raw_receipt = entry.get("receipt")
    if raw_receipt is None:
        raise InteractionNotPendingError(
            "interaction cannot be applied before a receipt is recorded"
        )
    receipt = InteractionReceipt.from_dict(raw_receipt, request=entry["request"])
    if receipt.receipt_id != normalized_receipt_id:
        raise InteractionIntegrityError(
            "applied receipt_id does not match the recorded receipt"
        )
    entry["application"] = application
    validated["active_id"] = None
    return validate_interaction_journal(validated)


__all__ = [
    "DurableInteractionSettlementPending",
    "DurableInteractionError",
    "INTERACTION_JOURNAL_KEY",
    "INTERACTION_JOURNAL_SCHEMA_VERSION",
    "INTERACTION_KIND_HUMAN_INPUT",
    "INTERACTION_KIND_MAX_BUDGET",
    "INTERACTION_KIND_TOOL_APPROVAL",
    "INTERACTION_KINDS",
    "INTERACTION_SCHEMA_VERSION",
    "InteractionAlreadyAppliedError",
    "InteractionError",
    "InteractionIntegrityError",
    "InteractionKind",
    "InteractionNotPendingError",
    "InteractionReceipt",
    "InteractionReceiptConflictError",
    "InteractionRequest",
    "build_interaction_receipt",
    "build_interaction_request",
    "get_active_interaction",
    "mark_interaction_applied",
    "new_interaction_journal",
    "record_interaction_receipt",
    "register_interaction_request",
    "strict_json_digest",
    "validate_interaction_journal",
    "validate_interaction_receipt",
    "validate_interaction_request",
]
