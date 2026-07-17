from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..kernel.delta import HarnessDelta
from ..kernel.harness import BaseRuntimeHarness, HarnessContext


DURABLE_TOOL_RECEIPT_LEDGER_KEY = "durable_tool_receipt_ledger"
TOOL_RECEIPT_JOURNAL_KEY = "tool_receipt_journal"
DURABLE_TOOL_RECEIPT_SCHEMA_VERSION = 1
DURABLE_TOOL_RECEIPT_LEDGER_SCHEMA_VERSION = 1


class DurableToolReceiptError(RuntimeError):
    """Base error for durable tool-result receipt failures."""

    code = "durable_tool_receipt_error"


class DurableToolReceiptIntegrityError(DurableToolReceiptError):
    code = "durable_tool_receipt_integrity_error"


class DurableToolReceiptPersistenceError(DurableToolReceiptError):
    code = "durable_tool_receipt_persistence_error"


class DurableToolReceiptAckError(DurableToolReceiptError):
    code = "durable_tool_receipt_ack_error"


@runtime_checkable
class DurableToolReceiptAcknowledger(Protocol):
    """Plugin hook for committing an externally prepared result cursor.

    Implementations must be idempotent. Returning ``False`` means that the
    plugin does not own this receipt; returning ``True`` means that the exact
    receipt/application pair has been acknowledged (or was already applied).
    """

    def acknowledge_durable_receipt(
        self,
        *,
        receipt: dict[str, Any],
        application: dict[str, Any],
    ) -> bool:
        ...


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise DurableToolReceiptIntegrityError(
            f"{label} must be JSON serializable"
        ) from exc
    return json.loads(encoded)


def _payload_digest(value: Any) -> str:
    encoded = json.dumps(
        _json_copy(value, label="receipt payload"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_non_empty_string(value: Any, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise DurableToolReceiptIntegrityError(f"{label} must be a non-empty string")
    return normalized


def _require_sha256(value: Any, *, label: str) -> str:
    normalized = _require_non_empty_string(value, label=label).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise DurableToolReceiptIntegrityError(f"{label} must be a sha256 hex digest")
    return normalized


def build_durable_tool_receipt(
    *,
    source: str,
    receipt_id: str,
    execution_id: str,
    logical_execution_id: str,
    created_by_attempt_id: str,
    tool_call_id: str,
    request_digest: str,
    result_digest: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "schema_version": DURABLE_TOOL_RECEIPT_SCHEMA_VERSION,
        "source": _require_non_empty_string(source, label="receipt source"),
        "receipt_id": _require_non_empty_string(receipt_id, label="receipt_id"),
        "execution_id": _require_non_empty_string(
            execution_id,
            label="execution_id",
        ),
        "logical_execution_id": _require_non_empty_string(
            logical_execution_id,
            label="logical_execution_id",
        ),
        "created_by_attempt_id": _require_non_empty_string(
            created_by_attempt_id,
            label="created_by_attempt_id",
        ),
        "tool_call_id": _require_non_empty_string(tool_call_id, label="tool_call_id"),
        "request_digest": _require_sha256(request_digest, label="request_digest"),
        "result_digest": _require_sha256(result_digest, label="result_digest"),
        "payload": _json_copy(payload or {}, label="receipt payload"),
    }
    return {
        **body,
        "receipt_digest": _payload_digest(body),
    }


def validate_durable_tool_receipt(raw: Any) -> dict[str, Any]:
    receipt = _json_copy(raw, label="durable tool receipt")
    expected_fields = {
        "schema_version",
        "source",
        "receipt_id",
        "execution_id",
        "logical_execution_id",
        "created_by_attempt_id",
        "tool_call_id",
        "request_digest",
        "result_digest",
        "payload",
        "receipt_digest",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_fields:
        raise DurableToolReceiptIntegrityError(
            "durable tool receipt has an invalid shape"
        )
    if receipt.get("schema_version") != DURABLE_TOOL_RECEIPT_SCHEMA_VERSION:
        raise DurableToolReceiptIntegrityError(
            "unsupported durable tool receipt schema_version"
        )
    for key in (
        "source",
        "receipt_id",
        "execution_id",
        "logical_execution_id",
        "created_by_attempt_id",
        "tool_call_id",
    ):
        receipt[key] = _require_non_empty_string(receipt.get(key), label=key)
    for key in ("request_digest", "result_digest"):
        receipt[key] = _require_sha256(receipt.get(key), label=key)
    if not isinstance(receipt.get("payload"), dict):
        raise DurableToolReceiptIntegrityError("receipt payload must be a dict")
    receipt_digest = _require_sha256(
        receipt.get("receipt_digest"),
        label="receipt_digest",
    )
    body = {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key != "receipt_digest"
    }
    if receipt_digest != _payload_digest(body):
        raise DurableToolReceiptIntegrityError("durable tool receipt digest mismatch")
    return receipt


def durable_tool_receipt_key(receipt: Any) -> str:
    validated = validate_durable_tool_receipt(receipt)
    identity = {
        "source": validated["source"],
        "execution_id": validated["execution_id"],
        "receipt_id": validated["receipt_id"],
    }
    return f"receipt_{_payload_digest(identity)[:32]}"


def empty_durable_tool_receipt_ledger() -> dict[str, Any]:
    return {
        "schema_version": DURABLE_TOOL_RECEIPT_LEDGER_SCHEMA_VERSION,
        "pending": {},
        "applications": {},
    }


def _build_application(
    receipt: dict[str, Any],
    *,
    application_id: str,
) -> dict[str, Any]:
    normalized_application_id = _require_non_empty_string(
        application_id,
        label="receipt application_id",
    )
    body = {
        "receipt": validate_durable_tool_receipt(receipt),
        "application_id": normalized_application_id,
    }
    return {
        **body,
        "application_digest": _payload_digest(body),
    }


def _durable_tool_call_identity(
    receipt: dict[str, Any],
) -> tuple[str, str, str, str]:
    """Return the provider-call identity inside one durable execution ledger."""

    return (
        receipt["source"],
        receipt["execution_id"],
        receipt["logical_execution_id"],
        receipt["tool_call_id"],
    )


def validate_durable_tool_receipt_ledger(raw: Any) -> dict[str, Any]:
    if raw is None:
        return empty_durable_tool_receipt_ledger()
    ledger = _json_copy(raw, label="durable tool receipt ledger")
    if not isinstance(ledger, dict) or set(ledger) != {
        "schema_version",
        "pending",
        "applications",
    }:
        raise DurableToolReceiptIntegrityError(
            "durable tool receipt ledger has an invalid shape"
        )
    if ledger.get("schema_version") != DURABLE_TOOL_RECEIPT_LEDGER_SCHEMA_VERSION:
        raise DurableToolReceiptIntegrityError(
            "unsupported durable tool receipt ledger schema_version"
        )
    raw_pending = ledger.get("pending")
    raw_applications = ledger.get("applications")
    if not isinstance(raw_pending, dict) or not isinstance(raw_applications, dict):
        raise DurableToolReceiptIntegrityError(
            "durable tool receipt ledger entries must be dicts"
        )

    pending: dict[str, dict[str, Any]] = {}
    applications: dict[str, dict[str, Any]] = {}
    seen_call_ids: dict[tuple[str, str, str, str], str] = {}
    for key, raw_receipt in raw_pending.items():
        if not isinstance(key, str) or not key:
            raise DurableToolReceiptIntegrityError(
                "durable tool receipt pending key must be non-empty"
            )
        receipt = validate_durable_tool_receipt(raw_receipt)
        if key != durable_tool_receipt_key(receipt):
            raise DurableToolReceiptIntegrityError(
                "durable tool receipt pending key does not match its receipt"
            )
        call_identity = _durable_tool_call_identity(receipt)
        previous = seen_call_ids.get(call_identity)
        if previous is not None and previous != key:
            raise DurableToolReceiptIntegrityError(
                "one tool call cannot own multiple pending durable receipts"
            )
        seen_call_ids[call_identity] = key
        pending[key] = receipt

    for key, raw_application in raw_applications.items():
        if not isinstance(key, str) or not key or not isinstance(raw_application, dict):
            raise DurableToolReceiptIntegrityError(
                "durable tool receipt application has an invalid shape"
            )
        if set(raw_application) != {
            "receipt",
            "application_id",
            "application_digest",
        }:
            raise DurableToolReceiptIntegrityError(
                "durable tool receipt application has an invalid shape"
            )
        receipt = validate_durable_tool_receipt(raw_application.get("receipt"))
        if key != durable_tool_receipt_key(receipt):
            raise DurableToolReceiptIntegrityError(
                "durable tool receipt application key does not match its receipt"
            )
        call_identity = _durable_tool_call_identity(receipt)
        previous = seen_call_ids.get(call_identity)
        if previous is not None and previous != key:
            raise DurableToolReceiptIntegrityError(
                "one tool call cannot own multiple durable receipts"
            )
        seen_call_ids[call_identity] = key
        application_id = _require_non_empty_string(
            raw_application.get("application_id"),
            label="receipt application_id",
        )
        expected = _build_application(receipt, application_id=application_id)
        digest = _require_sha256(
            raw_application.get("application_digest"),
            label="application_digest",
        )
        if digest != expected["application_digest"]:
            raise DurableToolReceiptIntegrityError(
                "durable tool receipt application digest mismatch"
            )
        applications[key] = expected
        if key in pending and pending[key] != receipt:
            raise DurableToolReceiptIntegrityError(
                "pending receipt conflicts with its durable application"
            )

    return {
        "schema_version": DURABLE_TOOL_RECEIPT_LEDGER_SCHEMA_VERSION,
        "pending": pending,
        "applications": applications,
    }


def merge_durable_tool_receipt_ledgers(current: Any, update: Any) -> dict[str, Any]:
    merged = validate_durable_tool_receipt_ledger(current)
    incoming = validate_durable_tool_receipt_ledger(update)
    for key, application in incoming["applications"].items():
        existing = merged["applications"].get(key)
        if existing is not None and existing != application:
            raise DurableToolReceiptIntegrityError(
                "durable tool receipt application cannot be rewritten"
            )
        merged["applications"][key] = copy.deepcopy(application)
        merged["pending"].pop(key, None)
    for key, receipt in incoming["pending"].items():
        applied = merged["applications"].get(key)
        if applied is not None:
            if applied.get("receipt") != receipt:
                raise DurableToolReceiptIntegrityError(
                    "pending receipt conflicts with its prior application"
                )
            continue
        existing = merged["pending"].get(key)
        if existing is not None and existing != receipt:
            raise DurableToolReceiptIntegrityError(
                "pending durable tool receipt cannot be rewritten"
            )
        for other_key, other in merged["pending"].items():
            if (
                other_key != key
                and _durable_tool_call_identity(other)
                == _durable_tool_call_identity(receipt)
            ):
                raise DurableToolReceiptIntegrityError(
                    "one tool call cannot own multiple pending durable receipts"
                )
        merged["pending"][key] = copy.deepcopy(receipt)
    return validate_durable_tool_receipt_ledger(merged)


def durable_tool_receipt_state_update(
    state_or_ledger: Any,
    *receipts: dict[str, Any],
) -> dict[str, Any]:
    """Build the existing ``state_updates`` payload used by runtime plugins."""

    if hasattr(state_or_ledger, "metadata"):
        metadata = getattr(state_or_ledger, "metadata", {})
        current = (
            metadata.get(DURABLE_TOOL_RECEIPT_LEDGER_KEY)
            if isinstance(metadata, dict)
            else None
        )
    else:
        current = state_or_ledger
    update = empty_durable_tool_receipt_ledger()
    for raw_receipt in receipts:
        receipt = validate_durable_tool_receipt(raw_receipt)
        update["pending"][durable_tool_receipt_key(receipt)] = receipt
    return {
        DURABLE_TOOL_RECEIPT_LEDGER_KEY: merge_durable_tool_receipt_ledgers(
            current,
            update,
        )
    }


def apply_durable_tool_receipt_ledger(
    session_state: dict[str, Any],
    *,
    ledger: Any,
    application_id: str,
) -> dict[str, Any]:
    """Tombstone pending receipts inside the caller's session-state CAS."""

    pending_ledger = validate_durable_tool_receipt_ledger(ledger)
    journal = validate_durable_tool_receipt_ledger(
        session_state.get(TOOL_RECEIPT_JOURNAL_KEY)
    )
    for key, receipt in pending_ledger["pending"].items():
        application = _build_application(receipt, application_id=application_id)
        existing = journal["applications"].get(key)
        if existing is not None and existing != application:
            raise DurableToolReceiptIntegrityError(
                "durable tool receipt is already bound to another application"
            )
        journal["applications"][key] = application
    journal["pending"] = {}
    validated = validate_durable_tool_receipt_ledger(journal)
    if validated["applications"]:
        session_state[TOOL_RECEIPT_JOURNAL_KEY] = validated
    return validated


def ensure_durable_tool_receipt_execution_id(
    ledger: Any,
    *,
    execution_id: str,
) -> dict[str, Any]:
    validated = validate_durable_tool_receipt_ledger(ledger)
    receipts = list(validated["pending"].values()) + [
        application["receipt"]
        for application in validated["applications"].values()
    ]
    if not receipts:
        return validated
    expected = _require_non_empty_string(execution_id, label="execution_id")
    if any(receipt.get("execution_id") != expected for receipt in receipts):
        raise DurableToolReceiptIntegrityError(
            "durable tool receipt belongs to a different execution_id"
        )
    return validated


def acknowledge_durable_tool_receipt_applications(
    plugins: list[Any] | tuple[Any, ...],
    session_state: dict[str, Any],
    *,
    receipt_keys: set[str] | None = None,
    execution_id: str | None = None,
) -> set[str]:
    journal = validate_durable_tool_receipt_ledger(
        session_state.get(TOOL_RECEIPT_JOURNAL_KEY)
    )
    if execution_id is not None:
        journal = ensure_durable_tool_receipt_execution_id(
            journal,
            execution_id=execution_id,
        )
    acknowledged: set[str] = set()
    selected = (
        set(journal["applications"])
        if receipt_keys is None
        else set(receipt_keys)
    )
    missing = selected - set(journal["applications"])
    if missing:
        raise DurableToolReceiptPersistenceError(
            "requested receipt application is missing from the session journal"
        )
    for key in sorted(selected):
        application = journal["applications"][key]
        receipt = application["receipt"]
        handled = False
        for plugin in plugins:
            acknowledge = getattr(plugin, "acknowledge_durable_receipt", None)
            if not callable(acknowledge):
                continue
            try:
                handled = acknowledge(
                    receipt=copy.deepcopy(receipt),
                    application=copy.deepcopy(application),
                ) is True
            except Exception as exc:
                raise DurableToolReceiptAckError(
                    f"failed to acknowledge durable tool receipt {key}: {exc}"
                ) from exc
            if handled:
                break
        if not handled:
            raise DurableToolReceiptAckError(
                f"no runtime plugin acknowledged durable tool receipt {key}"
            )
        acknowledged.add(key)
    return acknowledged


def settle_durable_tool_receipt_ledger(
    local_ledger: Any,
    session_state: dict[str, Any],
    *,
    acknowledged: set[str],
) -> dict[str, Any]:
    local = validate_durable_tool_receipt_ledger(local_ledger)
    journal = validate_durable_tool_receipt_ledger(
        session_state.get(TOOL_RECEIPT_JOURNAL_KEY)
    )
    for key in acknowledged:
        application = journal["applications"].get(key)
        if not isinstance(application, dict):
            raise DurableToolReceiptPersistenceError(
                "acknowledged receipt is missing its durable application"
            )
        pending = local["pending"].get(key)
        if pending is not None and pending != application.get("receipt"):
            raise DurableToolReceiptIntegrityError(
                "acknowledged receipt does not match the local pending result"
            )
        local["pending"].pop(key, None)
        local["applications"][key] = copy.deepcopy(application)
    return validate_durable_tool_receipt_ledger(local)


@dataclass
class DurableToolReceiptAckHarness(BaseRuntimeHarness):
    """Ack only after a session checkpoint/commit exposes a receipt tombstone."""

    plugins: tuple[Any, ...] = field(default_factory=tuple)
    name: str = "durable_tool_receipt_ack"
    phases: tuple[str, ...] = ("bootstrap", "on_resume", "before_model")
    order: int = 95

    def build_delta(self, context: HarnessContext) -> HarnessDelta | None:
        raw_local = context.state.metadata.get(DURABLE_TOOL_RECEIPT_LEDGER_KEY)
        local = validate_durable_tool_receipt_ledger(raw_local)
        session_snapshot = context.state.memory_state.get("session_snapshot")
        session_state = (
            copy.deepcopy(session_snapshot)
            if isinstance(session_snapshot, dict)
            else {}
        )
        journal = validate_durable_tool_receipt_ledger(
            session_state.get(TOOL_RECEIPT_JOURNAL_KEY)
        )
        missing = set(local["pending"]) - set(journal["applications"])
        if missing:
            raise DurableToolReceiptPersistenceError(
                "durable tool receipt reached the model boundary before its "
                "session application was persisted"
            )
        if not journal["applications"]:
            return None
        receipt_keys = {
            key
            for key, application in journal["applications"].items()
            if local["applications"].get(key) != application
            or key in local["pending"]
        }
        if not receipt_keys:
            return None
        acknowledged = acknowledge_durable_tool_receipt_applications(
            self.plugins,
            session_state,
            receipt_keys=receipt_keys,
            execution_id=str(context.state.session_state.session_id or ""),
        )
        settled = settle_durable_tool_receipt_ledger(
            local,
            session_state,
            acknowledged=acknowledged,
        )
        if settled == local:
            return None
        return HarnessDelta(
            created_by=f"harness.{self.name}",
            state_updates={DURABLE_TOOL_RECEIPT_LEDGER_KEY: settled},
            trace={
                "durable_tool_receipt_ack_count": len(acknowledged),
            },
        )


__all__ = [
    "DURABLE_TOOL_RECEIPT_LEDGER_KEY",
    "TOOL_RECEIPT_JOURNAL_KEY",
    "DurableToolReceiptAckError",
    "DurableToolReceiptAckHarness",
    "DurableToolReceiptAcknowledger",
    "DurableToolReceiptError",
    "DurableToolReceiptIntegrityError",
    "DurableToolReceiptPersistenceError",
    "acknowledge_durable_tool_receipt_applications",
    "apply_durable_tool_receipt_ledger",
    "build_durable_tool_receipt",
    "durable_tool_receipt_key",
    "durable_tool_receipt_state_update",
    "empty_durable_tool_receipt_ledger",
    "ensure_durable_tool_receipt_execution_id",
    "merge_durable_tool_receipt_ledgers",
    "settle_durable_tool_receipt_ledger",
    "validate_durable_tool_receipt",
    "validate_durable_tool_receipt_ledger",
]
