from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Protocol, runtime_checkable

from ..execution import (
    ActiveExecutionLeaseError,
    ExecutionCancellation,
    ExecutionCancelledError,
    ExecutionFence,
    ExecutionLease,
    ExecutionLeaseConflictError,
    ExecutionLeaseError,
    ExecutionLeaseExpiredError,
    ExecutionLeaseNotOwnedError,
    StaleExecutionLeaseError,
)
from ..tools.models import NormalizedToolHistoryRecord, ToolHistoryOptimizationContext
from .checkpoint_state import (
    EXECUTION_CHECKPOINT_DOMAIN_KEY,
    EXECUTION_CHECKPOINT_KEY,
    ExecutionCheckpointCompatibilityError,
    ExecutionCheckpointResumeRequiredError,
    apply_checkpoint_interaction_receipt,
    validate_execution_checkpoint,
)
from .ownership import ensure_session_delta_input
from .revision import (
    SessionRevisionConflictError,
    SessionSnapshot,
    _validate_execution_fence_target,
    load_session_snapshot,
    save_session_snapshot,
)


SummaryGenerator = Callable[[str, list[dict[str, Any]], int, str], str]
LongTermExtractor = Callable[..., dict[str, Any]]
HistoryToolResolver = Callable[[str], Any]


@runtime_checkable
class SessionStore(Protocol):
    def load(self, session_id: str) -> dict[str, Any]:
        ...

    def save(self, session_id: str, state: dict[str, Any]) -> None:
        ...


@runtime_checkable
class VectorStoreAdapter(Protocol):
    def add_texts(
        self,
        *,
        session_id: str,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        ...

    def similarity_search(
        self,
        *,
        session_id: str,
        query: str,
        k: int,
        min_score: float | None = None,
    ) -> list[str | dict[str, Any]]:
        ...


@runtime_checkable
class LongTermProfileStore(Protocol):
    def load(self, namespace: str) -> dict[str, Any]:
        ...

    def save(self, namespace: str, profile: dict[str, Any]) -> None:
        ...


@runtime_checkable
class LongTermVectorAdapter(Protocol):
    def add_texts(
        self,
        *,
        namespace: str,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        ...

    def similarity_search(
        self,
        *,
        namespace: str,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        ...


@runtime_checkable
class ContextStrategy(Protocol):
    def prepare(
        self,
        *,
        state: dict[str, Any],
        incoming: list[dict[str, Any]],
        max_context_window_tokens: int,
        model: str,
    ) -> list[dict[str, Any]]:
        ...

    def commit(
        self,
        *,
        state: dict[str, Any],
        full_conversation: list[dict[str, Any]],
    ) -> None:
        ...


class InMemorySessionStore:
    """Process-local session state store."""

    execution_lease_scope = "process_local"

    def __init__(self, *, clock_ms: Callable[[], int] | None = None):
        self._sessions: dict[str, dict[str, Any]] = {}
        self._revisions: dict[str, int] = {}
        self._execution_leases: dict[str, dict[str, Any]] = {}
        self._execution_cancellations: dict[
            str,
            dict[str, ExecutionCancellation],
        ] = {}
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._lock = RLock()

    def _now_ms(self) -> int:
        now = self._clock_ms()
        if isinstance(now, bool) or not isinstance(now, int):
            raise TypeError("clock_ms must return an integer epoch timestamp")
        return now

    @staticmethod
    def _validate_lease_request(
        execution_id: str,
        owner_id: str,
        *,
        ttl_ms: int | None = None,
        fencing_token: int | None = None,
    ) -> None:
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("execution_id must be a non-empty string")
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must be a non-empty string")
        if ttl_ms is not None and (
            isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms <= 0
        ):
            raise ValueError("ttl_ms must be a positive integer")
        if fencing_token is not None and (
            isinstance(fencing_token, bool)
            or not isinstance(fencing_token, int)
            or fencing_token <= 0
        ):
            raise ValueError("fencing_token must be a positive integer")

    @staticmethod
    def _validate_expected_revision(expected_revision: int | None) -> None:
        if expected_revision is not None and (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")

    def _lease_record_unlocked(self, execution_id: str) -> dict[str, Any]:
        return self._execution_leases.setdefault(
            execution_id,
            {
                "last_fencing_token": 0,
                "last_owner_id": "",
                "active": None,
            },
        )

    def _assert_active_lease_unlocked(
        self,
        execution_id: str,
        owner_id: str,
        fencing_token: int,
    ) -> ExecutionLease:
        cancellation = self._execution_cancellations.get(execution_id, {}).get(
            owner_id
        )
        if cancellation is not None:
            raise ExecutionCancelledError(
                cancellation.reason or "execution was cancelled",
                execution_id=execution_id,
                owner_id=owner_id,
                fencing_token=cancellation.fencing_token or fencing_token,
            )
        record = self._execution_leases.get(execution_id)
        active = record.get("active") if isinstance(record, dict) else None
        last_token = int(record.get("last_fencing_token") or 0) if record else 0
        if not isinstance(active, ExecutionLease):
            if fencing_token <= last_token:
                raise StaleExecutionLeaseError(
                    "execution lease has already been released or superseded",
                    execution_id=execution_id,
                    owner_id=owner_id,
                    fencing_token=fencing_token,
                )
            raise ExecutionLeaseNotOwnedError(
                "execution lease is not owned",
                execution_id=execution_id,
                owner_id=owner_id,
                fencing_token=fencing_token,
            )
        if fencing_token != active.fencing_token:
            error_type = (
                StaleExecutionLeaseError
                if fencing_token < active.fencing_token
                else ExecutionLeaseNotOwnedError
            )
            raise error_type(
                "execution fencing token is not current",
                execution_id=execution_id,
                owner_id=owner_id,
                fencing_token=fencing_token,
            )
        if owner_id != active.owner_id:
            raise ExecutionLeaseNotOwnedError(
                "execution lease belongs to a different owner",
                execution_id=execution_id,
                owner_id=owner_id,
                fencing_token=fencing_token,
            )
        if self._now_ms() >= active.expires_at_ms:
            raise ExecutionLeaseExpiredError(
                "execution lease has expired",
                execution_id=execution_id,
                owner_id=owner_id,
                fencing_token=fencing_token,
            )
        return active

    def _ensure_unfenced_write_allowed_unlocked(self, session_id: str) -> None:
        record = self._execution_leases.get(session_id)
        active = record.get("active") if isinstance(record, dict) else None
        if isinstance(active, ExecutionLease) and self._now_ms() < active.expires_at_ms:
            raise ActiveExecutionLeaseError(
                "unfenced session write is forbidden while an execution lease is active",
                execution_id=session_id,
                owner_id=active.owner_id,
                fencing_token=active.fencing_token,
            )

    def load(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._sessions.get(session_id, {}))

    def save(self, session_id: str, state: dict[str, Any]) -> None:
        with self._lock:
            self._ensure_unfenced_write_allowed_unlocked(session_id)
            self._sessions[session_id] = copy.deepcopy(state)
            self._revisions[session_id] = self._revisions.get(session_id, 0) + 1

    def load_with_revision(self, session_id: str) -> SessionSnapshot:
        with self._lock:
            return SessionSnapshot(
                state=copy.deepcopy(self._sessions.get(session_id, {})),
                revision=self._revisions.get(session_id, 0),
            )

    def save_if_revision(
        self,
        session_id: str,
        state: dict[str, Any],
        expected_revision: int,
    ) -> int:
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")
        with self._lock:
            self._ensure_unfenced_write_allowed_unlocked(session_id)
            actual_revision = self._revisions.get(session_id, 0)
            if actual_revision != expected_revision:
                raise SessionRevisionConflictError(
                    session_id=session_id,
                    expected_revision=expected_revision,
                    actual_revision=actual_revision,
                )
            next_revision = actual_revision + 1
            self._sessions[session_id] = copy.deepcopy(state)
            self._revisions[session_id] = next_revision
            return next_revision

    def save_if_revision_and_execution_not_cancelled(
        self,
        session_id: str,
        state: dict[str, Any],
        expected_revision: int,
        *,
        execution_id: str,
        owner_id: str,
    ) -> int:
        """CAS a detached child write only while its root attempt is live."""

        _validate_execution_fence_target(session_id, execution_id)
        self._validate_expected_revision(expected_revision)
        self._validate_lease_request(execution_id, owner_id)
        with self._lock:
            cancellation = self._execution_cancellations.get(
                execution_id,
                {},
            ).get(owner_id)
            if cancellation is not None:
                raise ExecutionCancelledError(
                    cancellation.reason or "execution was cancelled",
                    execution_id=execution_id,
                    owner_id=owner_id,
                    fencing_token=cancellation.fencing_token,
                )
            actual_revision = self._revisions.get(session_id, 0)
            if actual_revision != expected_revision:
                raise SessionRevisionConflictError(
                    session_id=session_id,
                    expected_revision=expected_revision,
                    actual_revision=actual_revision,
                )
            next_revision = actual_revision + 1
            self._sessions[session_id] = copy.deepcopy(state)
            self._revisions[session_id] = next_revision
            return next_revision

    def save_if_revision_and_execution_cancelled(
        self,
        session_id: str,
        state: dict[str, Any],
        expected_revision: int,
        *,
        execution_id: str,
        owner_id: str,
    ) -> int:
        """CAS terminal cleanup only after the exact root attempt was stopped."""

        _validate_execution_fence_target(session_id, execution_id)
        self._validate_expected_revision(expected_revision)
        self._validate_lease_request(execution_id, owner_id)
        with self._lock:
            cancellation = self._execution_cancellations.get(
                execution_id,
                {},
            ).get(owner_id)
            if cancellation is None:
                raise ExecutionLeaseError(
                    "execution cancellation is required for terminal cleanup",
                    execution_id=execution_id,
                    owner_id=owner_id,
                )
            actual_revision = self._revisions.get(session_id, 0)
            if actual_revision != expected_revision:
                raise SessionRevisionConflictError(
                    session_id=session_id,
                    expected_revision=expected_revision,
                    actual_revision=actual_revision,
                )
            next_revision = actual_revision + 1
            self._sessions[session_id] = copy.deepcopy(state)
            self._revisions[session_id] = next_revision
            return next_revision

    def acquire_lease(
        self,
        execution_id: str,
        owner_id: str,
        ttl_ms: int,
        *,
        expected_revision: int | None = None,
    ) -> ExecutionLease:
        self._validate_lease_request(execution_id, owner_id, ttl_ms=ttl_ms)
        self._validate_expected_revision(expected_revision)
        with self._lock:
            cancellation = self._execution_cancellations.get(
                execution_id,
                {},
            ).get(owner_id)
            if cancellation is not None:
                raise ExecutionCancelledError(
                    cancellation.reason or "execution was cancelled",
                    execution_id=execution_id,
                    owner_id=owner_id,
                    fencing_token=cancellation.fencing_token,
                )
            actual_revision = self._revisions.get(execution_id, 0)
            if expected_revision is not None and actual_revision != expected_revision:
                raise SessionRevisionConflictError(
                    session_id=execution_id,
                    expected_revision=expected_revision,
                    actual_revision=actual_revision,
                )
            now = self._now_ms()
            record = self._lease_record_unlocked(execution_id)
            active = record.get("active")
            if isinstance(active, ExecutionLease) and now < active.expires_at_ms:
                if active.owner_id == owner_id:
                    return active
                raise ExecutionLeaseConflictError(
                    "execution is already leased by another owner",
                    execution_id=execution_id,
                    owner_id=owner_id,
                    fencing_token=active.fencing_token,
                )
            token = int(record.get("last_fencing_token") or 0) + 1
            lease = ExecutionLease(
                execution_id=execution_id,
                owner_id=owner_id,
                fencing_token=token,
                acquired_at_ms=now,
                expires_at_ms=now + ttl_ms,
            )
            record.update(
                {
                    "last_fencing_token": token,
                    "last_owner_id": owner_id,
                    "active": lease,
                }
            )
            return lease

    def request_execution_cancel(
        self,
        execution_id: str,
        owner_id: str,
        *,
        reason: str = "",
    ) -> ExecutionCancellation:
        self._validate_lease_request(execution_id, owner_id)
        if not isinstance(reason, str):
            raise ValueError("reason must be a string")
        with self._lock:
            cancellations = self._execution_cancellations.setdefault(
                execution_id,
                {},
            )
            existing = cancellations.get(owner_id)
            if existing is not None:
                return existing
            record = self._lease_record_unlocked(execution_id)
            active = record.get("active")
            fencing_token = None
            if isinstance(active, ExecutionLease) and active.owner_id == owner_id:
                fencing_token = active.fencing_token
                record["active"] = None
            elif str(record.get("last_owner_id") or "") == owner_id:
                last_token = int(record.get("last_fencing_token") or 0)
                fencing_token = last_token or None
            cancellation = ExecutionCancellation(
                execution_id=execution_id,
                owner_id=owner_id,
                fencing_token=fencing_token,
                requested_at_ms=self._now_ms(),
                reason=reason,
            )
            cancellations[owner_id] = cancellation
            return cancellation

    def load_execution_cancellation(
        self,
        execution_id: str,
        owner_id: str,
    ) -> ExecutionCancellation | None:
        self._validate_lease_request(execution_id, owner_id)
        with self._lock:
            return self._execution_cancellations.get(execution_id, {}).get(
                owner_id
            )

    def verify_lease(
        self,
        execution_id: str,
        owner_id: str,
        fencing_token: int,
    ) -> ExecutionLease:
        self._validate_lease_request(
            execution_id,
            owner_id,
            fencing_token=fencing_token,
        )
        with self._lock:
            return self._assert_active_lease_unlocked(
                execution_id,
                owner_id,
                fencing_token,
            )

    def renew_lease(
        self,
        execution_id: str,
        owner_id: str,
        fencing_token: int,
        ttl_ms: int,
    ) -> ExecutionLease:
        self._validate_lease_request(
            execution_id,
            owner_id,
            ttl_ms=ttl_ms,
            fencing_token=fencing_token,
        )
        with self._lock:
            active = self._assert_active_lease_unlocked(
                execution_id,
                owner_id,
                fencing_token,
            )
            renewed = ExecutionLease(
                execution_id=active.execution_id,
                owner_id=active.owner_id,
                fencing_token=active.fencing_token,
                acquired_at_ms=active.acquired_at_ms,
                expires_at_ms=max(active.expires_at_ms, self._now_ms() + ttl_ms),
            )
            self._lease_record_unlocked(execution_id)["active"] = renewed
            return renewed

    def release_lease(
        self,
        execution_id: str,
        owner_id: str,
        fencing_token: int,
    ) -> None:
        self._validate_lease_request(
            execution_id,
            owner_id,
            fencing_token=fencing_token,
        )
        with self._lock:
            record = self._execution_leases.get(execution_id)
            active = record.get("active") if isinstance(record, dict) else None
            if not isinstance(active, ExecutionLease):
                if (
                    isinstance(record, dict)
                    and int(record.get("last_fencing_token") or 0) == fencing_token
                    and str(record.get("last_owner_id") or "") == owner_id
                ):
                    return
                self._assert_active_lease_unlocked(
                    execution_id,
                    owner_id,
                    fencing_token,
                )
                return
            self._assert_active_lease_unlocked(
                execution_id,
                owner_id,
                fencing_token,
            )
            record["active"] = None

    def save_if_revision_and_fence(
        self,
        session_id: str,
        state: dict[str, Any],
        expected_revision: int,
        *,
        execution_id: str,
        owner_id: str,
        fencing_token: int,
    ) -> int:
        _validate_execution_fence_target(session_id, execution_id)
        self._validate_expected_revision(expected_revision)
        self._validate_lease_request(
            execution_id,
            owner_id,
            fencing_token=fencing_token,
        )
        with self._lock:
            self._assert_active_lease_unlocked(
                execution_id,
                owner_id,
                fencing_token,
            )
            actual_revision = self._revisions.get(session_id, 0)
            if actual_revision != expected_revision:
                raise SessionRevisionConflictError(
                    session_id=session_id,
                    expected_revision=expected_revision,
                    actual_revision=actual_revision,
                )
            next_revision = actual_revision + 1
            self._sessions[session_id] = copy.deepcopy(state)
            self._revisions[session_id] = next_revision
            return next_revision


class JsonFileLongTermProfileStore:
    """Profile store backed by one JSON file per namespace."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base = Path(base_dir) if base_dir is not None else (_default_user_data_dir() / "long_term_profiles")
        self._base.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_ ." else "_" for c in namespace).strip().replace(" ", "_")
        return self._base / f"{safe or 'default'}.json"

    def load(self, namespace: str) -> dict[str, Any]:
        p = self._path(namespace)
        if not p.exists():
            return {}
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return copy.deepcopy(raw) if isinstance(raw, dict) else {}

    def save(self, namespace: str, profile: dict[str, Any]) -> None:
        p = self._path(namespace)
        p.write_text(json.dumps(profile, default=str, ensure_ascii=False), encoding="utf-8")


@dataclass
class LongTermMemoryConfig:
    profile_store: LongTermProfileStore | None = None
    vector_adapter: LongTermVectorAdapter | None = None
    extractor: LongTermExtractor | None = None
    vector_top_k: int = 4
    vector_min_score: float | None = None
    episode_top_k: int = 2
    episode_min_score: float | None = None
    playbook_top_k: int = 2
    playbook_min_score: float | None = None
    max_profile_chars: int = 1200
    max_fact_items: int = 6
    max_episode_items: int = 3
    max_playbook_items: int = 2
    extract_every_n_turns: int = 1
    embedding_model: str = "text-embedding-3-small"
    embedding_payload: dict[str, Any] | None = None
    profile_base_dir: str | Path | None = None
    qdrant_path: str | Path | None = None
    collection_prefix: str = "long_term"


@dataclass
class MemoryConfig:
    last_n_turns: int = 8
    summary_trigger_pct: float = 0.75
    summary_target_pct: float = 0.45
    max_summary_chars: int = 2400
    vector_top_k: int = 4
    vector_min_score: float | None = None
    vector_adapter: VectorStoreAdapter | None = None
    long_term: LongTermMemoryConfig | None = None
    deferred_tool_compaction_enabled: bool = True
    deferred_tool_compaction_keep_completed_turns: int = 1
    deferred_tool_compaction_max_chars: int = 1200
    deferred_tool_compaction_preview_chars: int = 160
    deferred_tool_compaction_include_tools: list[str] | None = None
    deferred_tool_compaction_hash_payloads: bool = True
    sliding_window_pct: float = 0.7
    sliding_window_max_tokens: int | None = None


@dataclass(frozen=True)
class MemoryCommitResult:
    """Per-call commit output that is safe to consume across concurrent runs."""

    commit_info: dict[str, Any]
    session_snapshot: SessionSnapshot


def _default_user_data_dir() -> Path:
    try:
        from platformdirs import user_data_dir

        return Path(user_data_dir("unchain", "unchain"))
    except Exception:
        return Path.home() / ".unchain"


def _deepcopy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [copy.deepcopy(m) for m in messages if isinstance(m, dict)]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in ("text", "input_text", "output_text"):
                text = block.get("text", "")
                if text:
                    parts.append(text if isinstance(text, str) else str(text))
                continue
            if block_type in ("image", "input_image"):
                parts.append("[image]")
                continue
            if block_type in ("pdf", "document", "input_file"):
                parts.append("[pdf]")
                continue
            parts.append(json.dumps(block, default=str, ensure_ascii=False))
        return "".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, default=str, ensure_ascii=False)
    if content is None:
        return ""
    return str(content)


def _message_to_text(message: dict[str, Any]) -> str:
    if not isinstance(message, dict):
        return ""
    if "content" in message:
        return _content_to_text(message.get("content"))
    text_parts: list[str] = []
    for key in ("arguments", "output", "result", "text"):
        if key not in message:
            continue
        value = message.get(key)
        if isinstance(value, str):
            text_parts.append(value)
        else:
            text_parts.append(json.dumps(value, default=str, ensure_ascii=False))
    if text_parts:
        return " ".join(text_parts)
    return json.dumps(message, default=str, ensure_ascii=False)


def _normalize_dialog_role(role: Any) -> str | None:
    if not isinstance(role, str):
        return None
    normalized = role.strip().lower()
    if normalized in ("user", "assistant"):
        return normalized
    return None


def _normalize_recall_messages_array(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = _normalize_dialog_role(item.get("role"))
        if role is None:
            continue
        raw_content = item.get("content")
        if raw_content is None:
            raw_content = item.get("text")
        content = _content_to_text(raw_content).strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _parse_prefixed_dialog_text(text: str) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    current_role: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_role, current_lines
        if current_role is None:
            current_lines = []
            return
        content = "\n".join(line for line in current_lines if isinstance(line, str)).strip()
        if content:
            parsed.append({"role": current_role, "content": content})
        current_role = None
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith("user:") or lowered.startswith("assistant:"):
            _flush()
            role = "user" if lowered.startswith("user:") else "assistant"
            content = line.split(":", 1)[1].strip() if ":" in line else ""
            current_role = role
            current_lines = [content] if content else []
            continue

        if current_role is not None:
            current_lines.append(raw_line)

    _flush()
    return parsed


def _normalize_recall_legacy_item(item: str | dict[str, Any]) -> list[dict[str, str]]:
    role: str | None = None
    raw_text: Any = ""
    if isinstance(item, str):
        raw_text = item
    elif isinstance(item, dict):
        raw_text = item.get("text")
        if raw_text is None:
            raw_text = item.get("content")
        role = _normalize_dialog_role(item.get("role"))
    else:
        return []

    text = _content_to_text(raw_text).strip()
    if not text:
        return []

    parsed = _parse_prefixed_dialog_text(text)
    if parsed:
        return parsed
    if role is not None:
        return [{"role": role, "content": text}]
    return [{"content": text}]


def _normalize_recalled_messages(
    recalled: list[str | dict[str, Any]],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in recalled:
        if isinstance(item, dict):
            parsed_messages = _normalize_recall_messages_array(item.get("messages"))
            if parsed_messages:
                normalized.extend(parsed_messages)
                continue
        normalized.extend(_normalize_recall_legacy_item(item))
    return normalized


def _infer_recall_roles_from_messages(
    recalled: list[dict[str, str]],
    messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    text_to_role: dict[str, str] = {}
    text_role_pairs: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        normalized_role = _normalize_dialog_role(role)
        if normalized_role is None:
            continue
        text = _message_to_text(message).strip()
        if not text:
            continue
        text_to_role[text] = normalized_role
        text_role_pairs.append((text, normalized_role))

    output: list[dict[str, str]] = []
    for item in recalled:
        normalized = dict(item)
        if "role" not in normalized:
            item_text = normalized.get("content", "")
            if not item_text:
                item_text = normalized.get("text", "")
            matched_role = text_to_role.get(item_text)
            if not matched_role and item_text:
                candidate_roles = {
                    role
                    for message_text, role in text_role_pairs
                    if item_text in message_text or message_text in item_text
                }
                if len(candidate_roles) == 1:
                    matched_role = next(iter(candidate_roles))
            if matched_role:
                normalized["role"] = matched_role
        output.append(normalized)
    return output


def _coerce_recall_messages(recalled: list[dict[str, str]]) -> list[dict[str, str]]:
    coerced: list[dict[str, str]] = []
    for item in recalled:
        content = _content_to_text(item.get("content")).strip()
        if not content:
            continue
        role = _normalize_dialog_role(item.get("role"))
        if role is None:
            role = "assistant"
        coerced.append({"role": role, "content": content})
    return coerced


def _dedupe_adjacent_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    for message in messages:
        if (
            deduped
            and deduped[-1].get("role") == message.get("role")
            and deduped[-1].get("content") == message.get("content")
        ):
            continue
        deduped.append(message)
    return deduped


def _build_turn_embedding_payload(
    turn_messages: list[tuple[int, dict[str, Any]]],
    *,
    turn_start_index: int,
    turn_end_index: int,
) -> tuple[str, dict[str, Any]] | None:
    normalized_messages: list[dict[str, str]] = []
    has_user = False
    has_assistant = False

    for _, message in turn_messages:
        role = _normalize_dialog_role(message.get("role"))
        if role is None:
            continue
        content = _message_to_text(message).strip()
        if not content:
            continue
        normalized_messages.append({"role": role, "content": content})
        if role == "user":
            has_user = True
        elif role == "assistant":
            has_assistant = True

    if not (has_user and has_assistant):
        return None

    text = "\n".join(f"{item['role']}: {item['content']}" for item in normalized_messages)
    metadata = {
        "messages": normalized_messages,
        "turn_start_index": turn_start_index,
        "turn_end_index": turn_end_index,
    }
    return text, metadata


def _collect_complete_turns_for_vector_index(
    messages: list[dict[str, Any]],
    *,
    start_index: int,
) -> tuple[list[str], list[dict[str, Any]], int, int]:
    n = len(messages)
    safe_start = max(0, min(int(start_index), n))
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    next_indexed_until = safe_start
    indexed_turn_count = 0

    i = safe_start
    while i < n:
        while i < n:
            message = messages[i]
            if isinstance(message, dict) and message.get("role") == "user":
                break
            i += 1
        next_indexed_until = i
        if i >= n:
            break

        turn_start = i
        i += 1
        while i < n:
            message = messages[i]
            if isinstance(message, dict) and message.get("role") == "user":
                break
            i += 1
        turn_end = i - 1
        is_tail = i >= n

        turn_messages: list[tuple[int, dict[str, Any]]] = []
        for idx in range(turn_start, turn_end + 1):
            msg = messages[idx]
            if isinstance(msg, dict):
                turn_messages.append((idx, msg))

        payload = _build_turn_embedding_payload(
            turn_messages,
            turn_start_index=turn_start,
            turn_end_index=turn_end,
        )
        if payload is not None:
            text, metadata = payload
            texts.append(text)
            metadatas.append(metadata)
            indexed_turn_count += 1
            next_indexed_until = turn_end + 1
            continue

        if is_tail:
            next_indexed_until = turn_start
            break

        next_indexed_until = turn_end + 1

    return texts, metadatas, next_indexed_until, indexed_turn_count


def _estimate_tokens_from_messages(messages: list[dict[str, Any]]) -> int:
    total_chars = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if isinstance(role, str):
            total_chars += len(role) + 1
        total_chars += len(_message_to_text(message))
    if total_chars <= 0:
        return 0
    return int(math.ceil(total_chars / 4.0))


def _split_turns(non_system_messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for msg in non_system_messages:
        role = msg.get("role") if isinstance(msg, dict) else None
        if role == "user" and current:
            turns.append(current)
            current = [copy.deepcopy(msg)]
            continue
        current.append(copy.deepcopy(msg))
    if current:
        turns.append(current)
    return turns


def _is_tool_result_like_message(message: dict[str, Any]) -> bool:
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, list) and content:
        if all(isinstance(block, dict) and block.get("type") == "tool_result" for block in content):
            return True
    parts = message.get("parts")
    if isinstance(parts, list) and parts:
        if all(isinstance(part, dict) and "function_response" in part for part in parts):
            return True
    return False


def _split_turns_for_deferred_compaction(non_system_messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for msg in non_system_messages:
        role = msg.get("role") if isinstance(msg, dict) else None
        if role == "user" and current and not _is_tool_result_like_message(msg):
            turns.append(current)
            current = [copy.deepcopy(msg)]
            continue
        current.append(copy.deepcopy(msg))
    if current:
        turns.append(current)
    return turns


def _flatten_turns(turns: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for turn in turns:
        flattened.extend(_deepcopy_messages(turn))
    return flattened


def _safe_json_dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)


def _count_text_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _make_text_preview(text: str, preview_chars: int) -> str:
    if len(text) <= max(1, preview_chars * 2):
        return text
    head = text[:preview_chars]
    tail = text[-preview_chars:]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n... <omitted {omitted} chars> ...\n{tail}"


def _payload_digest(value: Any) -> str:
    return hashlib.sha1(_safe_json_dumps(value).encode("utf-8", errors="replace")).hexdigest()


def _safe_json_loads(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped[0] not in "{[":
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return None


def _normalize_tool_history_payload(raw_payload: Any) -> tuple[Any, str]:
    if isinstance(raw_payload, str):
        parsed = _safe_json_loads(raw_payload)
        if isinstance(parsed, (dict, list)):
            return parsed, "json_string"
        return raw_payload, "raw_string"
    return copy.deepcopy(raw_payload), "native"


def _serialize_tool_history_payload(payload: Any, payload_format: str) -> Any:
    if payload_format == "json_string":
        return _safe_json_dumps(payload)
    if payload_format == "raw_string":
        if isinstance(payload, str):
            return payload
        return _safe_json_dumps(payload)
    return copy.deepcopy(payload)


def _shallow_preview_value(value: Any, preview_chars: int) -> Any:
    if isinstance(value, str):
        if len(value) <= preview_chars * 2:
            return value
        return {
            "type": "string",
            "chars": len(value),
            "lines": _count_text_lines(value),
            "preview": _make_text_preview(value, preview_chars),
        }
    if isinstance(value, dict):
        return {
            "type": "object",
            "keys": list(value.keys())[:8],
            "size_estimate_chars": len(_safe_json_dumps(value)),
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "length": len(value),
            "preview_items": [_shallow_preview_value(item, preview_chars) for item in value[:3]],
        }
    return copy.deepcopy(value)


def _generic_compact_tool_payload(payload: Any, context: ToolHistoryOptimizationContext) -> Any:
    if isinstance(payload, dict) and payload.get("compacted") is True:
        return copy.deepcopy(payload)

    if isinstance(payload, str):
        if len(payload) <= context.max_chars:
            return payload
        compacted = {
            "compacted": True,
            "original_type": "string",
            "chars": len(payload),
            "lines": _count_text_lines(payload),
            "preview": _make_text_preview(payload, context.preview_chars),
        }
        if context.include_hash:
            compacted["digest"] = _payload_digest(payload)
        return compacted

    if isinstance(payload, dict):
        serialized = _safe_json_dumps(payload)
        if len(serialized) <= context.max_chars:
            return copy.deepcopy(payload)
        compacted: dict[str, Any] = {
            "compacted": True,
            "original_type": "object",
            "size_estimate_chars": len(serialized),
            "keys": list(payload.keys())[:20],
            "preview": {},
        }
        preview_items = list(payload.items())[:8]
        compacted["preview"] = {
            str(key): _shallow_preview_value(value, context.preview_chars)
            for key, value in preview_items
        }
        if context.include_hash:
            compacted["digest"] = _payload_digest(payload)
        return compacted

    if isinstance(payload, list):
        serialized = _safe_json_dumps(payload)
        if len(serialized) <= context.max_chars:
            return copy.deepcopy(payload)
        compacted = {
            "compacted": True,
            "original_type": "array",
            "size_estimate_chars": len(serialized),
            "length": len(payload),
            "preview_items": [_shallow_preview_value(item, context.preview_chars) for item in payload[:5]],
        }
        if context.include_hash:
            compacted["digest"] = _payload_digest(payload)
        return compacted

    return copy.deepcopy(payload)


def _build_tool_call_name_map(messages: list[dict[str, Any]]) -> dict[str, str]:
    call_map: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue

        if message.get("type") == "function_call":
            call_id = message.get("call_id")
            name = message.get("name")
            if isinstance(call_id, str) and isinstance(name, str) and call_id and name:
                call_map[call_id] = name

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                call_id = tool_call.get("id")
                fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                name = fn.get("name")
                if isinstance(call_id, str) and isinstance(name, str) and call_id and name:
                    call_map[call_id] = name

        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue
                call_id = block.get("id")
                name = block.get("name")
                if isinstance(call_id, str) and isinstance(name, str) and call_id and name:
                    call_map[call_id] = name
    return call_map


def _normalize_tool_history_records(
    messages: list[dict[str, Any]],
    *,
    provider: str,
    call_name_map: dict[str, str],
) -> list[NormalizedToolHistoryRecord]:
    records: list[NormalizedToolHistoryRecord] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue

        if message.get("type") == "function_call":
            payload, payload_format = _normalize_tool_history_payload(message.get("arguments", "{}"))
            records.append(
                NormalizedToolHistoryRecord(
                    tool_name=str(message.get("name") or ""),
                    call_id=str(message.get("call_id") or ""),
                    kind="arguments",
                    payload=payload,
                    provider=provider,
                    message_index=message_index,
                    location_type="openai_function_call",
                    payload_format=payload_format,
                    field_name="arguments",
                )
            )

        if message.get("type") == "function_call_output":
            call_id = str(message.get("call_id") or "")
            payload, payload_format = _normalize_tool_history_payload(message.get("output", ""))
            records.append(
                NormalizedToolHistoryRecord(
                    tool_name=call_name_map.get(call_id, ""),
                    call_id=call_id,
                    kind="result",
                    payload=payload,
                    provider=provider,
                    message_index=message_index,
                    location_type="openai_function_call_output",
                    payload_format=payload_format,
                    field_name="output",
                )
            )

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for block_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                payload, payload_format = _normalize_tool_history_payload(function.get("arguments", {}))
                records.append(
                    NormalizedToolHistoryRecord(
                        tool_name=str(function.get("name") or ""),
                        call_id=str(tool_call.get("id") or ""),
                        kind="arguments",
                        payload=payload,
                        provider=provider,
                        message_index=message_index,
                        location_type="assistant_tool_calls",
                        payload_format=payload_format,
                        block_index=block_index,
                        field_name="function.arguments",
                    )
                )

        content = message.get("content")
        if isinstance(content, list):
            for block_index, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "tool_use":
                    payload, payload_format = _normalize_tool_history_payload(block.get("input", {}))
                    records.append(
                        NormalizedToolHistoryRecord(
                            tool_name=str(block.get("name") or ""),
                            call_id=str(block.get("id") or ""),
                            kind="arguments",
                            payload=payload,
                            provider=provider,
                            message_index=message_index,
                            location_type="content_block_tool_use",
                            payload_format=payload_format,
                            block_index=block_index,
                            field_name="input",
                        )
                    )
                elif block_type == "tool_result":
                    call_id = str(block.get("tool_use_id") or "")
                    payload, payload_format = _normalize_tool_history_payload(block.get("content", ""))
                    records.append(
                        NormalizedToolHistoryRecord(
                            tool_name=call_name_map.get(call_id, ""),
                            call_id=call_id,
                            kind="result",
                            payload=payload,
                            provider=provider,
                            message_index=message_index,
                            location_type="content_block_tool_result",
                            payload_format=payload_format,
                            block_index=block_index,
                            field_name="content",
                        )
                    )

        parts = message.get("parts")
        if isinstance(parts, list):
            for part_index, part in enumerate(parts):
                if not isinstance(part, dict):
                    continue
                function_call = part.get("function_call")
                if isinstance(function_call, dict):
                    payload, payload_format = _normalize_tool_history_payload(function_call.get("args", {}))
                    records.append(
                        NormalizedToolHistoryRecord(
                            tool_name=str(function_call.get("name") or ""),
                            call_id=str(function_call.get("id") or part.get("id") or ""),
                            kind="arguments",
                            payload=payload,
                            provider=provider,
                            message_index=message_index,
                            location_type="parts_function_call",
                            payload_format=payload_format,
                            part_index=part_index,
                            field_name="function_call.args",
                        )
                    )
                function_response = part.get("function_response")
                if isinstance(function_response, dict):
                    payload, payload_format = _normalize_tool_history_payload(function_response.get("response", {}))
                    records.append(
                        NormalizedToolHistoryRecord(
                            tool_name=str(function_response.get("name") or ""),
                            call_id=str(function_response.get("id") or part.get("id") or ""),
                            kind="result",
                            payload=payload,
                            provider=provider,
                            message_index=message_index,
                            location_type="parts_function_response",
                            payload_format=payload_format,
                            part_index=part_index,
                            field_name="function_response.response",
                        )
                    )

        if message.get("role") == "tool" and "content" in message:
            call_id = str(message.get("tool_call_id") or "")
            payload, payload_format = _normalize_tool_history_payload(message.get("content", ""))
            records.append(
                NormalizedToolHistoryRecord(
                    tool_name=call_name_map.get(call_id, ""),
                    call_id=call_id,
                    kind="result",
                    payload=payload,
                    provider=provider,
                    message_index=message_index,
                    location_type="role_tool_content",
                    payload_format=payload_format,
                    field_name="content",
                )
            )
    return records


def _coerce_object_field_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return copy.deepcopy(payload)
    return {"value": copy.deepcopy(payload)}


def _write_back_tool_history_payload(
    messages: list[dict[str, Any]],
    record: NormalizedToolHistoryRecord,
    payload: Any,
) -> None:
    if record.message_index < 0 or record.message_index >= len(messages):
        return
    message = messages[record.message_index]
    if not isinstance(message, dict):
        return

    serialized = _serialize_tool_history_payload(payload, record.payload_format)

    if record.location_type == "openai_function_call":
        message["arguments"] = serialized
        return
    if record.location_type == "openai_function_call_output":
        message["output"] = serialized
        return
    if record.location_type == "assistant_tool_calls":
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or record.block_index is None or record.block_index >= len(tool_calls):
            return
        raw_tool_call = tool_calls[record.block_index]
        if not isinstance(raw_tool_call, dict):
            return
        function = raw_tool_call.get("function")
        if not isinstance(function, dict):
            return
        function["arguments"] = serialized
        return
    if record.location_type == "content_block_tool_use":
        content = message.get("content")
        if not isinstance(content, list) or record.block_index is None or record.block_index >= len(content):
            return
        block = content[record.block_index]
        if not isinstance(block, dict):
            return
        block["input"] = _coerce_object_field_payload(serialized)
        return
    if record.location_type == "content_block_tool_result":
        content = message.get("content")
        if not isinstance(content, list) or record.block_index is None or record.block_index >= len(content):
            return
        block = content[record.block_index]
        if not isinstance(block, dict):
            return
        block["content"] = serialized
        return
    if record.location_type == "parts_function_call":
        parts = message.get("parts")
        if not isinstance(parts, list) or record.part_index is None or record.part_index >= len(parts):
            return
        part = parts[record.part_index]
        if not isinstance(part, dict):
            return
        function_call = part.get("function_call")
        if not isinstance(function_call, dict):
            return
        function_call["args"] = _coerce_object_field_payload(serialized)
        return
    if record.location_type == "parts_function_response":
        parts = message.get("parts")
        if not isinstance(parts, list) or record.part_index is None or record.part_index >= len(parts):
            return
        part = parts[record.part_index]
        if not isinstance(part, dict):
            return
        function_response = part.get("function_response")
        if not isinstance(function_response, dict):
            return
        function_response["response"] = _coerce_object_field_payload(serialized)
        return
    if record.location_type == "role_tool_content":
        message["content"] = serialized


def _compact_tool_history_messages(
    messages: list[dict[str, Any]],
    *,
    all_messages: list[dict[str, Any]],
    latest_messages: list[dict[str, Any]],
    provider: str,
    session_id: str,
    tool_resolver: HistoryToolResolver | None,
    config: MemoryConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    updated = _deepcopy_messages(messages)
    if not updated:
        return updated, {
            "deferred_compaction_applied": False,
            "deferred_compaction_message_count": 0,
            "deferred_compaction_record_count": 0,
            "deferred_compaction_bytes_removed_estimate": 0,
        }

    before_chars = len(_safe_json_dumps(updated))
    include_tools = {
        str(name).strip()
        for name in (config.deferred_tool_compaction_include_tools or [])
        if isinstance(name, str) and name.strip()
    }
    call_name_map = _build_tool_call_name_map(all_messages)
    records = _normalize_tool_history_records(updated, provider=provider, call_name_map=call_name_map)
    compacted_record_count = 0

    for record in records:
        if include_tools and record.tool_name not in include_tools:
            continue

        optimizer = None
        if callable(tool_resolver):
            tool_obj = tool_resolver(record.tool_name)
            if tool_obj is not None:
                optimizer = (
                    getattr(tool_obj, "history_arguments_optimizer", None)
                    if record.kind == "arguments"
                    else getattr(tool_obj, "history_result_optimizer", None)
                )

        context = ToolHistoryOptimizationContext(
            tool_name=record.tool_name,
            call_id=record.call_id,
            kind=record.kind,
            provider=provider,
            session_id=session_id,
            latest_messages=_deepcopy_messages(latest_messages),
            max_chars=max(64, int(config.deferred_tool_compaction_max_chars)),
            preview_chars=max(32, int(config.deferred_tool_compaction_preview_chars)),
            include_hash=bool(config.deferred_tool_compaction_hash_payloads),
        )

        try:
            optimized = optimizer(copy.deepcopy(record.payload), context) if callable(optimizer) else _generic_compact_tool_payload(record.payload, context)
        except ExecutionLeaseError:
            raise
        except Exception:
            optimized = _generic_compact_tool_payload(record.payload, context)

        if optimized != record.payload:
            compacted_record_count += 1
        _write_back_tool_history_payload(updated, record, optimized)

    after_chars = len(_safe_json_dumps(updated))
    return updated, {
        "deferred_compaction_applied": compacted_record_count > 0,
        "deferred_compaction_message_count": len(updated),
        "deferred_compaction_record_count": compacted_record_count,
        "deferred_compaction_bytes_removed_estimate": max(0, before_chars - after_chars),
    }


def _apply_deferred_tool_compaction(
    messages: list[dict[str, Any]],
    *,
    provider: str,
    session_id: str,
    tool_resolver: HistoryToolResolver | None,
    config: MemoryConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = _deepcopy_messages(messages)
    if not config.deferred_tool_compaction_enabled:
        return output, {
            "deferred_compaction_applied": False,
            "deferred_compaction_skip_reason": "disabled",
        }

    systems, non_system = _split_system_and_non_system(output)
    turns = _split_turns_for_deferred_compaction(non_system)
    keep_completed_turns = max(0, int(config.deferred_tool_compaction_keep_completed_turns))
    if len(turns) <= keep_completed_turns + 1:
        return output, {
            "deferred_compaction_applied": False,
            "deferred_compaction_skip_reason": "insufficient_turns",
        }

    compacted_turn_count = max(0, len(turns) - keep_completed_turns - 1)
    if compacted_turn_count <= 0:
        return output, {
            "deferred_compaction_applied": False,
            "deferred_compaction_skip_reason": "no_older_turns",
        }

    old_turns = turns[:compacted_turn_count]
    recent_turns = turns[compacted_turn_count:]
    latest_messages = systems + _flatten_turns(recent_turns)
    compacted_old_messages, stats = _compact_tool_history_messages(
        _flatten_turns(old_turns),
        all_messages=output,
        latest_messages=latest_messages,
        provider=provider,
        session_id=session_id,
        tool_resolver=tool_resolver,
        config=config,
    )
    compacted_messages = systems + compacted_old_messages + _flatten_turns(recent_turns)
    return compacted_messages, {
        **stats,
        "deferred_compaction_turns_compacted": compacted_turn_count,
    }


def _split_system_and_non_system(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    systems: list[dict[str, Any]] = []
    non_system: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            systems.append(copy.deepcopy(msg))
        else:
            non_system.append(copy.deepcopy(msg))
    return systems, non_system


def _latest_user_query(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _message_to_text(msg).strip()
            if text:
                return text
    return ""


def _merge_history_and_incoming(
    history: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    clean_history = _deepcopy_messages(history)
    clean_incoming = _deepcopy_messages(incoming)
    if not clean_history:
        return clean_incoming
    if not clean_incoming:
        return clean_history
    incoming_systems, incoming_non_system = _split_system_and_non_system(clean_incoming)
    history_systems, history_non_system = _split_system_and_non_system(clean_history)
    systems = copy.deepcopy(history_systems)
    for contribution in incoming_systems:
        if contribution not in systems:
            systems.append(copy.deepcopy(contribution))
    return systems + history_non_system + incoming_non_system


def _resolve_memory_namespace(session_id: str, memory_namespace: str | None) -> str:
    if isinstance(memory_namespace, str) and memory_namespace.strip():
        return memory_namespace.strip()
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()
    return ""


def _normalize_profile_document(profile: Any) -> dict[str, Any]:
    return copy.deepcopy(profile) if isinstance(profile, dict) else {}


# Recursive merge keeps existing nested keys unless the patch overwrites them.
def _merge_profile_documents(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(key, str) and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_profile_documents(merged[key], value)
        elif isinstance(key, str):
            merged[key] = copy.deepcopy(value)
    return merged


def _flatten_turn_messages(metadatas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for metadata in metadatas:
        raw_messages = metadata.get("messages")
        if not isinstance(raw_messages, list):
            continue
        for message in raw_messages:
            if not isinstance(message, dict):
                continue
            role = _normalize_dialog_role(message.get("role"))
            content = _content_to_text(message.get("content")).strip()
            if role is None or not content:
                continue
            messages.append({"role": role, "content": content})
    return messages


def _normalize_long_term_text(item: Any, *, max_chars: int = 2_000) -> str:
    text = _content_to_text(item).strip()
    if not text:
        return ""
    if len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def _infer_long_term_memory_type(item: dict[str, Any]) -> str:
    raw_type = str(item.get("memory_type") or item.get("type") or "").strip().lower()
    if raw_type in {"fact", "episode", "playbook"}:
        return raw_type
    if "steps" in item or "trigger" in item or "goal" in item:
        return "playbook"
    if "situation" in item or "action" in item or "outcome" in item:
        return "episode"
    return "fact"


def _normalize_long_term_fact_item(item: Any) -> dict[str, str] | None:
    if isinstance(item, str):
        text = item.strip()
        return {"memory_type": "fact", "subtype": "fact", "text": text} if text else None
    if not isinstance(item, dict):
        return None
    text = _normalize_long_term_text(item.get("text") if "text" in item else item.get("content"))
    if not text:
        return None
    raw_subtype = item.get("subtype") if "subtype" in item else item.get("kind")
    subtype = raw_subtype.strip().lower() if isinstance(raw_subtype, str) else "fact"
    if subtype not in {"fact", "decision", "project_context", "entity", "event"}:
        subtype = "fact"
    normalized = {"memory_type": "fact", "subtype": subtype, "text": text}
    for key in ("source", "created_at"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
    return normalized


def _normalize_long_term_facts(items: Any, max_items: int) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        fact = _normalize_long_term_fact_item(item)
        if fact is None:
            continue
        key = (fact["subtype"], fact["text"])
        if key in seen:
            continue
        seen.add(key)
        normalized.append(fact)
        if len(normalized) >= max(0, int(max_items)):
            break
    return normalized


def _format_episode_memory_text(item: dict[str, Any]) -> str:
    explicit_text = _normalize_long_term_text(item.get("text"))
    if explicit_text:
        return explicit_text
    parts = []
    situation = _normalize_long_term_text(item.get("situation"), max_chars=600)
    action = _normalize_long_term_text(item.get("action"), max_chars=600)
    outcome = _normalize_long_term_text(item.get("outcome"), max_chars=600)
    if situation:
        parts.append(f"Situation: {situation}")
    if action:
        parts.append(f"Action: {action}")
    if outcome:
        parts.append(f"Outcome: {outcome}")
    return "\n".join(parts).strip()


def _normalize_long_term_episode_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    text = _format_episode_memory_text(item)
    if not text:
        return None
    normalized: dict[str, Any] = {
        "memory_type": "episode",
        "text": text,
    }
    for key in ("situation", "action", "outcome", "title"):
        value = _normalize_long_term_text(item.get(key), max_chars=600)
        if value:
            normalized[key] = value
    for key in ("source", "created_at"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
    return normalized


def _normalize_long_term_episodes(items: Any, max_items: int) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        episode = _normalize_long_term_episode_item(item)
        if episode is None:
            continue
        key = episode["text"]
        if key in seen:
            continue
        seen.add(key)
        normalized.append(episode)
        if len(normalized) >= max(0, int(max_items)):
            break
    return normalized


def _format_playbook_memory_text(item: dict[str, Any]) -> str:
    explicit_text = _normalize_long_term_text(item.get("text"), max_chars=2_400)
    if explicit_text:
        return explicit_text
    parts = []
    trigger = _normalize_long_term_text(item.get("trigger"), max_chars=400)
    goal = _normalize_long_term_text(item.get("goal"), max_chars=400)
    steps = item.get("steps")
    caveats = _normalize_long_term_text(item.get("caveats"), max_chars=600)
    if trigger:
        parts.append(f"Trigger: {trigger}")
    if goal:
        parts.append(f"Goal: {goal}")
    if isinstance(steps, list):
        normalized_steps = [
            _normalize_long_term_text(step, max_chars=400)
            for step in steps
            if _normalize_long_term_text(step, max_chars=400)
        ]
        if normalized_steps:
            parts.append("Steps:")
            parts.extend(f"{idx + 1}. {step}" for idx, step in enumerate(normalized_steps[:8]))
    if caveats:
        parts.append(f"Caveats: {caveats}")
    return "\n".join(parts).strip()


def _normalize_long_term_playbook_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    text = _format_playbook_memory_text(item)
    if not text:
        return None
    normalized: dict[str, Any] = {
        "memory_type": "playbook",
        "text": text,
    }
    for key in ("trigger", "goal", "caveats", "title"):
        value = _normalize_long_term_text(item.get(key), max_chars=600)
        if value:
            normalized[key] = value
    steps = item.get("steps")
    if isinstance(steps, list):
        normalized_steps = [
            _normalize_long_term_text(step, max_chars=400)
            for step in steps
            if _normalize_long_term_text(step, max_chars=400)
        ]
        if normalized_steps:
            normalized["steps"] = normalized_steps[:8]
    for key in ("source", "created_at"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
    return normalized


def _normalize_long_term_playbooks(items: Any, max_items: int) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        playbook = _normalize_long_term_playbook_item(item)
        if playbook is None:
            continue
        key = playbook["text"]
        if key in seen:
            continue
        seen.add(key)
        normalized.append(playbook)
        if len(normalized) >= max(0, int(max_items)):
            break
    return normalized


def _normalize_long_term_search_results(
    items: Any,
    *,
    max_items: int,
    memory_type: str,
) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    filtered: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        inferred_type = _infer_long_term_memory_type(item)
        if inferred_type != memory_type:
            continue
        filtered.append(item)
    if memory_type == "fact":
        return _normalize_long_term_facts(filtered, max_items)
    if memory_type == "episode":
        return _normalize_long_term_episodes(filtered, max_items)
    if memory_type == "playbook":
        return _normalize_long_term_playbooks(filtered, max_items)
    return []


def _build_long_term_memory_payloads(
    *,
    facts: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    playbooks: list[dict[str, Any]],
    session_id: str,
    memory_namespace: str,
    turn_metadatas: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    if not facts and not episodes and not playbooks:
        return [], []
    first_turn = min((meta.get("turn_start_index", 0) for meta in turn_metadatas if isinstance(meta, dict)), default=0)
    last_turn = max((meta.get("turn_end_index", 0) for meta in turn_metadatas if isinstance(meta, dict)), default=0)
    created_at = _current_timestamp()
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for item in list(facts) + list(episodes) + list(playbooks):
        memory_type = _infer_long_term_memory_type(item) if isinstance(item, dict) else ""
        text = _normalize_long_term_text(item.get("text") if isinstance(item, dict) else "")
        if memory_type not in {"fact", "episode", "playbook"} or not text:
            continue
        texts.append(text)
        metadata: dict[str, Any] = {
            "memory_type": memory_type,
            "kind": item.get("subtype", memory_type) if isinstance(item, dict) else memory_type,
            "session_id": session_id,
            "memory_namespace": memory_namespace,
            "turn_start_index": first_turn,
            "turn_end_index": last_turn,
            "created_at": item.get("created_at", created_at) if isinstance(item, dict) else created_at,
            "source": item.get("source", "long_term_extractor") if isinstance(item, dict) else "long_term_extractor",
        }
        if isinstance(item, dict):
            for key in ("subtype", "title", "situation", "action", "outcome", "trigger", "goal", "caveats"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    metadata[key] = value.strip()
            steps = item.get("steps")
            if isinstance(steps, list):
                normalized_steps = [step for step in steps if isinstance(step, str) and step.strip()]
                if normalized_steps:
                    metadata["steps"] = normalized_steps[:8]
        metadatas.append(metadata)
    return texts, metadatas


def _profile_priority(key: str) -> tuple[int, str]:
    normalized = str(key or "").strip().lower()
    if normalized in {"hard_constraints", "constraints"}:
        return (0, normalized)
    if normalized == "preferences":
        return (1, normalized)
    if normalized == "communication_style":
        return (2, normalized)
    if normalized == "identity":
        return (3, normalized)
    return (10, normalized)


def _select_profile_for_context(profile: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    if not profile:
        return {}
    rendered = json.dumps(profile, ensure_ascii=False)
    if len(rendered) <= max_chars:
        return profile
    selected: dict[str, Any] = {}
    for key in sorted(profile.keys(), key=_profile_priority):
        candidate = {**selected, key: profile[key]}
        if len(json.dumps(candidate, ensure_ascii=False)) > max_chars:
            continue
        selected[key] = copy.deepcopy(profile[key])
    return selected or {}


def _should_recall_episode_memories(query: str) -> bool:
    normalized = query.strip().lower()
    if not normalized:
        return False
    hints = (
        "before",
        "previous",
        "last time",
        "again",
        "similar",
        "history",
        "historical",
        "remember",
        "之前",
        "以前",
        "上次",
        "类似",
        "历史",
        "记得",
    )
    return any(hint in normalized for hint in hints)


def _should_recall_playbooks(query: str) -> bool:
    normalized = query.strip().lower()
    if not normalized:
        return False
    hints = (
        "how to",
        "how do",
        "steps",
        "process",
        "procedure",
        "workflow",
        "plan",
        "debug",
        "troubleshoot",
        "fix",
        "resolve",
        "repair",
        "排查",
        "步骤",
        "流程",
        "方案",
        "修复",
        "如何",
        "怎么做",
        "调试",
        "排错",
    )
    return any(hint in normalized for hint in hints)


def _call_long_term_extractor(
    extractor: LongTermExtractor,
    *,
    previous_profile: dict[str, Any],
    extraction_messages: list[dict[str, Any]],
    long_term: LongTermMemoryConfig,
    model: str,
) -> dict[str, Any]:
    try:
        parameters = inspect.signature(extractor).parameters
    except Exception:
        parameters = {}

    if "config" in parameters:
        return extractor(
            previous_profile,
            extraction_messages,
            long_term.max_profile_chars,
            long_term.max_fact_items,
            model,
            config=long_term,
        )
    if "long_term_config" in parameters:
        return extractor(
            previous_profile,
            extraction_messages,
            long_term.max_profile_chars,
            long_term.max_fact_items,
            model,
            long_term_config=long_term,
        )
    return extractor(
        previous_profile,
        extraction_messages,
        long_term.max_profile_chars,
        long_term.max_fact_items,
        model,
    )


def _call_vector_similarity_search(
    adapter: VectorStoreAdapter,
    *,
    session_id: str,
    query: str,
    k: int,
    min_score: float | None,
) -> list[str | dict[str, Any]]:
    try:
        parameters = inspect.signature(adapter.similarity_search).parameters
    except Exception:
        parameters = {}

    kwargs: dict[str, Any] = {
        "session_id": session_id,
        "query": query,
        "k": k,
    }
    if "min_score" in parameters:
        kwargs["min_score"] = min_score
    return adapter.similarity_search(**kwargs)


def _call_long_term_similarity_search(
    adapter: LongTermVectorAdapter,
    *,
    namespace: str,
    query: str,
    k: int,
    filters: dict[str, Any] | None = None,
    min_score: float | None,
) -> list[dict[str, Any]]:
    try:
        parameters = inspect.signature(adapter.similarity_search).parameters
    except Exception:
        parameters = {}

    kwargs: dict[str, Any] = {
        "namespace": namespace,
        "query": query,
        "k": k,
        "filters": filters,
    }
    if "min_score" in parameters:
        kwargs["min_score"] = min_score
    return adapter.similarity_search(**kwargs)


def _current_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class LastNTurnsStrategy:
    def __init__(self, last_n_turns: int = 8):
        self.last_n_turns = max(0, int(last_n_turns))

    def prepare(
        self,
        *,
        state: dict[str, Any],
        incoming: list[dict[str, Any]],
        max_context_window_tokens: int,
        model: str,
    ) -> list[dict[str, Any]]:
        del max_context_window_tokens, model
        systems, non_system = _split_system_and_non_system(incoming)
        turns = _split_turns(non_system)
        kept_turns = turns[-self.last_n_turns :] if self.last_n_turns > 0 else []
        output = systems + _flatten_turns(kept_turns)

        memory_meta = state.setdefault("_memory_meta", {})
        memory_meta["last_n_turns"] = self.last_n_turns
        memory_meta["kept_turn_count"] = len(kept_turns)
        memory_meta["total_turn_count"] = len(turns)
        return output

    def commit(
        self,
        *,
        state: dict[str, Any],
        full_conversation: list[dict[str, Any]],
    ) -> None:
        """Keep no persistent state for the last-N-turns projection."""


class SummaryTokenStrategy:
    def __init__(
        self,
        *,
        summary_trigger_pct: float = 0.75,
        summary_target_pct: float = 0.45,
        max_summary_chars: int = 2400,
    ):
        self.summary_trigger_pct = float(summary_trigger_pct)
        self.summary_target_pct = float(summary_target_pct)
        self.max_summary_chars = max(256, int(max_summary_chars))

    def prepare(
        self,
        *,
        state: dict[str, Any],
        incoming: list[dict[str, Any]],
        max_context_window_tokens: int,
        model: str,
    ) -> list[dict[str, Any]]:
        output = _deepcopy_messages(incoming)
        memory_meta = state.setdefault("_memory_meta", {})
        memory_meta["summary_triggered"] = False

        if max_context_window_tokens <= 0:
            memory_meta["summary_skip_reason"] = "invalid_max_context_window"
            return output

        estimated_tokens = _estimate_tokens_from_messages(output)
        usage_pct = estimated_tokens / max_context_window_tokens
        memory_meta["summary_usage_pct"] = round(usage_pct, 4)

        if usage_pct < self.summary_trigger_pct:
            memory_meta["summary_skip_reason"] = "below_trigger_threshold"
            return output

        systems, non_system = _split_system_and_non_system(output)
        turns = _split_turns(non_system)
        if len(turns) <= 1:
            memory_meta["summary_skip_reason"] = "insufficient_turns"
            return output

        target_tokens = max(1, int(max_context_window_tokens * self.summary_target_pct))
        summary_budget_tokens = max(32, int(math.ceil(self.max_summary_chars / 4.0)))
        running_tokens = _estimate_tokens_from_messages(systems) + summary_budget_tokens

        kept_turns: list[list[dict[str, Any]]] = []
        for turn in reversed(turns):
            turn_tokens = _estimate_tokens_from_messages(turn)
            if kept_turns and running_tokens + turn_tokens > target_tokens:
                break
            kept_turns.insert(0, _deepcopy_messages(turn))
            running_tokens += turn_tokens

        if len(kept_turns) >= len(turns):
            memory_meta["summary_skip_reason"] = "no_old_turns_to_summarize"
            return output

        old_turn_count = len(turns) - len(kept_turns)
        old_messages = _flatten_turns(turns[:old_turn_count])
        previous_summary = str(state.get("summary", ""))
        summary_generator = state.get("_summary_generator")
        if not callable(summary_generator):
            memory_meta["summary_fallback_reason"] = "summary_generator_missing"
            return output

        try:
            summary_text = summary_generator(
                previous_summary,
                old_messages,
                self.max_summary_chars,
                model,
            )
        except ExecutionLeaseError:
            raise
        except Exception as exc:
            memory_meta["summary_fallback_reason"] = f"summary_generation_failed: {exc}"
            return output

        summary_text = (summary_text or "").strip()
        if not summary_text:
            memory_meta["summary_fallback_reason"] = "empty_summary"
            return output

        if len(summary_text) > self.max_summary_chars:
            summary_text = summary_text[: self.max_summary_chars].rstrip()

        state["summary"] = summary_text
        summary_message = {"role": "system", "content": f"[MEMORY SUMMARY]\n{summary_text}"}
        summarized_output = systems + [summary_message] + _flatten_turns(kept_turns)

        memory_meta["summary_triggered"] = True
        memory_meta["summary_old_turn_count"] = old_turn_count
        memory_meta["summary_kept_turn_count"] = len(kept_turns)
        memory_meta["summary_target_tokens"] = target_tokens
        return summarized_output

    def commit(
        self,
        *,
        state: dict[str, Any],
        full_conversation: list[dict[str, Any]],
    ) -> None:
        """Summary state is updated during preparation; commit has no work."""


class HybridContextStrategy:
    def __init__(
        self,
        *,
        summary_strategy: SummaryTokenStrategy,
        last_n_strategy: LastNTurnsStrategy,
        vector_top_k: int = 4,
        vector_min_score: float | None = None,
        vector_adapter: VectorStoreAdapter | None = None,
    ):
        self.summary_strategy = summary_strategy
        self.last_n_strategy = last_n_strategy
        self.vector_top_k = max(0, int(vector_top_k))
        self.vector_min_score = float(vector_min_score) if vector_min_score is not None else None
        self.vector_adapter = vector_adapter

    def prepare(
        self,
        *,
        state: dict[str, Any],
        incoming: list[dict[str, Any]],
        max_context_window_tokens: int,
        model: str,
    ) -> list[dict[str, Any]]:
        summarized = self.summary_strategy.prepare(
            state=state,
            incoming=incoming,
            max_context_window_tokens=max_context_window_tokens,
            model=model,
        )
        trimmed = self.last_n_strategy.prepare(
            state=state,
            incoming=summarized,
            max_context_window_tokens=max_context_window_tokens,
            model=model,
        )
        output = _deepcopy_messages(trimmed)

        adapter = self.vector_adapter
        session_id = str(state.get("_session_id", "") or "")
        if adapter is None or self.vector_top_k <= 0 or not session_id:
            return output

        query = _latest_user_query(output)
        if not query:
            return output

        memory_meta = state.setdefault("_memory_meta", {})
        try:
            recalled = _call_vector_similarity_search(
                adapter,
                session_id=session_id,
                query=query,
                k=self.vector_top_k,
                min_score=self.vector_min_score,
            )
        except ExecutionLeaseError:
            raise
        except Exception as exc:
            memory_meta["vector_fallback_reason"] = f"vector_search_failed: {exc}"
            return output

        normalized_recalled = _normalize_recalled_messages(recalled)
        if normalized_recalled:
            normalized_recalled = _infer_recall_roles_from_messages(normalized_recalled, incoming)
        recall_messages = _dedupe_adjacent_messages(_coerce_recall_messages(normalized_recalled))
        if not recall_messages:
            return output

        systems, non_system = _split_system_and_non_system(output)
        recall_message = {
            "role": "system",
            "content": f"[Recall messages]\n{json.dumps(recall_messages, ensure_ascii=False)}",
        }
        memory_meta["vector_recall_hit_count"] = len(recalled) if isinstance(recalled, list) else 0
        memory_meta["vector_recall_count"] = len(recall_messages)
        memory_meta["vector_recall_with_role_count"] = len(recall_messages)
        return systems + [recall_message] + non_system

    def commit(
        self,
        *,
        state: dict[str, Any],
        full_conversation: list[dict[str, Any]],
    ) -> None:
        self.summary_strategy.commit(state=state, full_conversation=full_conversation)
        self.last_n_strategy.commit(state=state, full_conversation=full_conversation)


class MemoryManager:
    def __init__(
        self,
        config: MemoryConfig | None = None,
        store: SessionStore | None = None,
        strategy: ContextStrategy | None = None,
    ):
        self.config = config or MemoryConfig()
        self.store = store or InMemorySessionStore()
        self.strategy = strategy or HybridContextStrategy(
            summary_strategy=SummaryTokenStrategy(
                summary_trigger_pct=self.config.summary_trigger_pct,
                summary_target_pct=self.config.summary_target_pct,
                max_summary_chars=self.config.max_summary_chars,
            ),
            last_n_strategy=LastNTurnsStrategy(last_n_turns=self.config.last_n_turns),
            vector_top_k=self.config.vector_top_k,
            vector_min_score=self.config.vector_min_score,
            vector_adapter=self.config.vector_adapter,
        )
        self._last_prepare_info: dict[str, Any] = {}
        self._last_commit_info: dict[str, Any] = {}

    @property
    def last_prepare_info(self) -> dict[str, Any]:
        return copy.deepcopy(self._last_prepare_info)

    @property
    def last_commit_info(self) -> dict[str, Any]:
        return copy.deepcopy(self._last_commit_info)

    def _verify_execution_fence(
        self,
        execution_fence: ExecutionFence | None,
    ) -> None:
        if execution_fence is None:
            return
        verify_lease = getattr(self.store, "verify_lease", None)
        if not callable(verify_lease):
            raise TypeError("execution fence requires a lease-aware session store")
        verify_lease(
            execution_fence.execution_id,
            execution_fence.owner_id,
            execution_fence.fencing_token,
        )

    @staticmethod
    def _normalize_top_k(candidate: Any, default: int) -> int:
        try:
            if candidate is None:
                return max(0, int(default))
            return max(0, int(candidate))
        except Exception:
            return max(0, int(default))

    def _short_term_recall_result(
        self,
        *,
        session_id: str,
        query: str,
        top_k: int | None = None,
        history_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        resolved_query = str(query or "").strip()
        resolved_top_k = self._normalize_top_k(top_k, self.config.vector_top_k)
        result = {
            "available": False,
            "session_id": session_id,
            "query": resolved_query,
            "top_k": resolved_top_k,
            "min_score": self.config.vector_min_score,
            "messages": [],
            "hit_count": 0,
            "count": 0,
            "fallback_reason": "",
        }

        adapter = self.config.vector_adapter
        if adapter is None:
            result["fallback_reason"] = "vector_adapter_missing"
            return result
        if resolved_top_k <= 0:
            result["fallback_reason"] = "top_k_disabled"
            return result
        if not session_id:
            result["fallback_reason"] = "missing_session_id"
            return result
        if not resolved_query:
            result["fallback_reason"] = "missing_query"
            return result

        if history_messages is None:
            state = self.store.load(session_id) if session_id else {}
            raw_history = state.get("messages", []) if isinstance(state, dict) else []
            if not isinstance(raw_history, list):
                raw_history = []
            history_messages = _deepcopy_messages(raw_history)
        else:
            history_messages = _deepcopy_messages(history_messages)

        try:
            recalled = _call_vector_similarity_search(
                adapter,
                session_id=session_id,
                query=resolved_query,
                k=resolved_top_k,
                min_score=self.config.vector_min_score,
            )
        except ExecutionLeaseError:
            raise
        except Exception as exc:
            result["fallback_reason"] = f"vector_search_failed: {exc}"
            return result

        normalized_recalled = _normalize_recalled_messages(recalled)
        if normalized_recalled:
            normalized_recalled = _infer_recall_roles_from_messages(normalized_recalled, history_messages)
        recall_messages = _dedupe_adjacent_messages(_coerce_recall_messages(normalized_recalled))
        result["available"] = True
        result["messages"] = recall_messages
        result["hit_count"] = len(recalled) if isinstance(recalled, list) else 0
        result["count"] = len(recall_messages)
        return result

    def _profile_recall_result(
        self,
        *,
        session_id: str,
        memory_namespace: str | None,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        long_term = self.config.long_term
        namespace = _resolve_memory_namespace(session_id, memory_namespace)
        result = {
            "available": False,
            "memory_namespace": namespace or "",
            "max_chars": self._normalize_top_k(
                max_chars,
                long_term.max_profile_chars if long_term is not None else 0,
            ),
            "profile": {},
            "key_count": 0,
            "fallback_reason": "",
        }

        if long_term is None:
            result["fallback_reason"] = "long_term_disabled"
            return result
        if not namespace:
            result["fallback_reason"] = "missing_memory_namespace"
            return result

        self.ensure_long_term_components()
        profile_store = long_term.profile_store
        if profile_store is None:
            result["fallback_reason"] = "profile_store_missing"
            return result

        try:
            profile = _normalize_profile_document(profile_store.load(namespace))
        except ExecutionLeaseError:
            raise
        except Exception as exc:
            result["fallback_reason"] = f"profile_load_failed: {exc}"
            return result

        selected_profile = _select_profile_for_context(
            profile,
            max_chars=result["max_chars"],
        )
        result["available"] = True
        result["profile"] = selected_profile
        result["key_count"] = len(selected_profile) if isinstance(selected_profile, dict) else 0
        return result

    def _long_term_recall_result(
        self,
        *,
        session_id: str,
        memory_namespace: str | None,
        query: str,
        memory_type: str,
        top_k: int | None = None,
        apply_query_hints: bool = False,
    ) -> dict[str, Any]:
        long_term = self.config.long_term
        namespace = _resolve_memory_namespace(session_id, memory_namespace)
        default_top_k = 0
        min_score = None
        if long_term is not None:
            if memory_type == "fact":
                default_top_k = long_term.vector_top_k
                min_score = long_term.vector_min_score
            elif memory_type == "episode":
                default_top_k = long_term.episode_top_k
                min_score = long_term.episode_min_score
            elif memory_type == "playbook":
                default_top_k = long_term.playbook_top_k
                min_score = long_term.playbook_min_score

        resolved_query = str(query or "").strip()
        resolved_top_k = self._normalize_top_k(top_k, default_top_k)
        result = {
            "available": False,
            "memory_namespace": namespace or "",
            "memory_type": memory_type,
            "query": resolved_query,
            "top_k": resolved_top_k,
            "min_score": min_score,
            "items": [],
            "hit_count": 0,
            "count": 0,
            "fallback_reason": "",
        }

        if long_term is None:
            result["fallback_reason"] = "long_term_disabled"
            return result
        if not namespace:
            result["fallback_reason"] = "missing_memory_namespace"
            return result

        self.ensure_long_term_components()
        if long_term.vector_adapter is None:
            result["fallback_reason"] = "long_term_vector_adapter_missing"
            return result
        if resolved_top_k <= 0:
            result["fallback_reason"] = "top_k_disabled"
            return result
        if not resolved_query:
            result["fallback_reason"] = "missing_query"
            return result

        if apply_query_hints:
            if memory_type == "episode" and not _should_recall_episode_memories(resolved_query):
                result["fallback_reason"] = "query_hints_not_matched"
                return result
            if memory_type == "playbook" and not _should_recall_playbooks(resolved_query):
                result["fallback_reason"] = "query_hints_not_matched"
                return result

        try:
            recalled = _call_long_term_similarity_search(
                long_term.vector_adapter,
                namespace=namespace,
                query=resolved_query,
                k=resolved_top_k,
                filters={"memory_type": memory_type},
                min_score=float(min_score) if min_score is not None else None,
            )
        except ExecutionLeaseError:
            raise
        except Exception as exc:
            result["fallback_reason"] = f"{memory_type}_search_failed: {exc}"
            return result

        normalized = _normalize_long_term_search_results(
            recalled,
            max_items=resolved_top_k,
            memory_type=memory_type,
        )
        result["available"] = True
        result["items"] = normalized
        result["hit_count"] = len(recalled) if isinstance(recalled, list) else 0
        result["count"] = len(normalized)
        return result

    def recall_profile(
        self,
        *,
        session_id: str,
        memory_namespace: str | None = None,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        result = self._profile_recall_result(
            session_id=session_id,
            memory_namespace=memory_namespace,
            max_chars=max_chars,
        )
        return {
            "memory_namespace": result["memory_namespace"],
            "profile": copy.deepcopy(result["profile"]),
            "key_count": result["key_count"],
            "available": result["available"],
            **({"fallback_reason": result["fallback_reason"]} if result["fallback_reason"] else {}),
        }

    def recall_memory(
        self,
        *,
        session_id: str,
        query: str,
        memory_namespace: str | None = None,
        top_k: int | None = None,
        include_short_term: bool = True,
        include_long_term: bool = True,
    ) -> dict[str, Any]:
        state = self.store.load(session_id) if session_id else {}
        raw_history = state.get("messages", []) if isinstance(state, dict) else []
        history_messages = _deepcopy_messages(raw_history) if isinstance(raw_history, list) else []
        namespace = _resolve_memory_namespace(session_id, memory_namespace) or ""

        result = {
            "session_id": session_id,
            "memory_namespace": namespace,
            "query": str(query or "").strip(),
            "top_k_override": top_k if top_k is None else self._normalize_top_k(top_k, 0),
            "short_term": {
                "messages": [],
                "count": 0,
                "hit_count": 0,
                "available": False,
            },
            "long_term": {
                "facts": [],
                "episodes": [],
                "playbooks": [],
                "counts": {"facts": 0, "episodes": 0, "playbooks": 0},
                "hit_counts": {"facts": 0, "episodes": 0, "playbooks": 0},
                "available": False,
            },
        }

        if include_short_term:
            short_term = self._short_term_recall_result(
                session_id=session_id,
                query=query,
                top_k=top_k,
                history_messages=history_messages,
            )
            result["short_term"] = {
                "messages": copy.deepcopy(short_term["messages"]),
                "count": short_term["count"],
                "hit_count": short_term["hit_count"],
                "available": short_term["available"],
                **({"fallback_reason": short_term["fallback_reason"]} if short_term["fallback_reason"] else {}),
            }

        if include_long_term:
            facts = self._long_term_recall_result(
                session_id=session_id,
                memory_namespace=memory_namespace,
                query=query,
                memory_type="fact",
                top_k=top_k,
                apply_query_hints=False,
            )
            episodes = self._long_term_recall_result(
                session_id=session_id,
                memory_namespace=memory_namespace,
                query=query,
                memory_type="episode",
                top_k=top_k,
                apply_query_hints=False,
            )
            playbooks = self._long_term_recall_result(
                session_id=session_id,
                memory_namespace=memory_namespace,
                query=query,
                memory_type="playbook",
                top_k=top_k,
                apply_query_hints=False,
            )
            long_term_result = {
                "facts": copy.deepcopy(facts["items"]),
                "episodes": copy.deepcopy(episodes["items"]),
                "playbooks": copy.deepcopy(playbooks["items"]),
                "counts": {
                    "facts": facts["count"],
                    "episodes": episodes["count"],
                    "playbooks": playbooks["count"],
                },
                "hit_counts": {
                    "facts": facts["hit_count"],
                    "episodes": episodes["hit_count"],
                    "playbooks": playbooks["hit_count"],
                },
                "available": any(
                    item["available"] for item in (facts, episodes, playbooks)
                ),
            }
            fallback_reasons = {
                "facts": facts["fallback_reason"],
                "episodes": episodes["fallback_reason"],
                "playbooks": playbooks["fallback_reason"],
            }
            if any(fallback_reasons.values()):
                long_term_result["fallback_reasons"] = {
                    key: value
                    for key, value in fallback_reasons.items()
                    if value
                }
            result["long_term"] = long_term_result

        return result

    def ensure_long_term_components(self, *, api_key_source: Any | None = None) -> None:
        long_term = self.config.long_term
        if long_term is None:
            return

        if long_term.profile_store is None:
            long_term.profile_store = JsonFileLongTermProfileStore(base_dir=long_term.profile_base_dir)

        if long_term.vector_adapter is not None:
            return

        from .qdrant import build_default_long_term_qdrant_vector_adapter

        qdrant_path = Path(long_term.qdrant_path) if long_term.qdrant_path is not None else (_default_user_data_dir() / "qdrant_long_term")
        long_term.vector_adapter = build_default_long_term_qdrant_vector_adapter(
            api_key_source=api_key_source,
            model=long_term.embedding_model,
            payload=long_term.embedding_payload,
            path=qdrant_path,
            collection_prefix=long_term.collection_prefix,
        )

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        return _estimate_tokens_from_messages(_deepcopy_messages(messages))

    def _prepare_long_term_messages(
        self,
        *,
        state: dict[str, Any],
        prepared: list[dict[str, Any]],
        session_id: str,
        memory_namespace: str | None,
        supports_tools: bool | None = None,
    ) -> list[dict[str, Any]]:
        long_term = self.config.long_term
        if long_term is None:
            return prepared

        namespace = _resolve_memory_namespace(session_id, memory_namespace)
        if not namespace:
            return prepared

        self.ensure_long_term_components()
        memory_meta = state.setdefault("_memory_meta", {})
        memory_meta["memory_namespace"] = namespace

        systems, non_system = _split_system_and_non_system(prepared)
        injected: list[dict[str, Any]] = []

        profile_result = self._profile_recall_result(
            session_id=session_id,
            memory_namespace=memory_namespace,
            max_chars=long_term.max_profile_chars,
        )
        profile_fallback_reason = str(profile_result.get("fallback_reason") or "").strip()
        if profile_fallback_reason:
            memory_meta["long_term_profile_fallback_reason"] = profile_fallback_reason
        if not bool(supports_tools):
            selected_profile = profile_result.get("profile") or {}
            if selected_profile:
                injected.append({
                    "role": "system",
                    "content": f"[MEMORY_PROFILE]\n{json.dumps(selected_profile, ensure_ascii=False)}",
                })
                memory_meta["long_term_profile_applied"] = True
                memory_meta["long_term_profile_key_count"] = int(profile_result.get("key_count") or 0)

        query = _latest_user_query(prepared)
        fact_result = self._long_term_recall_result(
            session_id=session_id,
            memory_namespace=memory_namespace,
            query=query,
            memory_type="fact",
            apply_query_hints=False,
        )
        fact_fallback_reason = str(fact_result.get("fallback_reason") or "").strip()
        if fact_fallback_reason and fact_fallback_reason != "missing_query":
            memory_meta["long_term_vector_fallback_reason"] = fact_fallback_reason
        facts = fact_result.get("items") or []
        if facts:
            injected.append({
                "role": "system",
                "content": f"[MEMORY_FACTS]\n{json.dumps(facts, ensure_ascii=False)}",
            })
            memory_meta["long_term_fact_recall_hit_count"] = int(fact_result.get("hit_count") or 0)
            memory_meta["long_term_fact_recall_count"] = int(fact_result.get("count") or 0)

        episode_result = self._long_term_recall_result(
            session_id=session_id,
            memory_namespace=memory_namespace,
            query=query,
            memory_type="episode",
            apply_query_hints=True,
        )
        episode_fallback_reason = str(episode_result.get("fallback_reason") or "").strip()
        if episode_fallback_reason and episode_fallback_reason not in {"missing_query", "query_hints_not_matched"}:
            memory_meta["long_term_episode_fallback_reason"] = episode_fallback_reason
        episodes = episode_result.get("items") or []
        if episodes:
            injected.append({
                "role": "system",
                "content": f"[MEMORY_EPISODES]\n{json.dumps(episodes, ensure_ascii=False)}",
            })
            memory_meta["long_term_episode_recall_count"] = int(episode_result.get("count") or 0)

        playbook_result = self._long_term_recall_result(
            session_id=session_id,
            memory_namespace=memory_namespace,
            query=query,
            memory_type="playbook",
            apply_query_hints=True,
        )
        playbook_fallback_reason = str(playbook_result.get("fallback_reason") or "").strip()
        if playbook_fallback_reason and playbook_fallback_reason not in {"missing_query", "query_hints_not_matched"}:
            memory_meta["long_term_playbook_fallback_reason"] = playbook_fallback_reason
        playbooks = playbook_result.get("items") or []
        if playbooks:
            injected.append({
                "role": "system",
                "content": f"[MEMORY_PLAYBOOKS]\n{json.dumps(playbooks, ensure_ascii=False)}",
            })
            memory_meta["long_term_playbook_recall_count"] = int(playbook_result.get("count") or 0)

        if not injected:
            return prepared
        return systems + injected + non_system

    def prepare_messages(
        self,
        session_id: str,
        incoming: list[dict[str, Any]],
        *,
        max_context_window_tokens: int,
        model: str,
        summary_generator: SummaryGenerator | None = None,
        memory_namespace: str | None = None,
        provider: str | None = None,
        tool_resolver: HistoryToolResolver | None = None,
        supports_tools: bool | None = None,
    ) -> list[dict[str, Any]]:
        session_snapshot = (
            load_session_snapshot(self.store, session_id)
            if session_id
            else SessionSnapshot(state={}, revision=None)
        )
        state = copy.deepcopy(session_snapshot.state)
        loaded_state = copy.deepcopy(state)
        if EXECUTION_CHECKPOINT_KEY in state:
            checkpoint = validate_execution_checkpoint(
                state.get(EXECUTION_CHECKPOINT_KEY)
            )
            raise ExecutionCheckpointResumeRequiredError(
                "session contains a durable execution checkpoint "
                f"({checkpoint.get('status')}); use Agent/KernelMemoryRuntime "
                "to restore it before preparing semantic memory"
            )
        history = state.get("messages", [])
        if not isinstance(history, list):
            history = []

        ensure_session_delta_input(history=history, incoming=incoming)
        merged = _merge_history_and_incoming(history, incoming)
        before_tokens = self.estimate_tokens(merged)

        state["_memory_meta"] = {}
        state["_session_id"] = session_id
        state["_memory_model"] = model
        if summary_generator is not None:
            state["_summary_generator"] = summary_generator

        try:
            compacted_input, compaction_meta = _apply_deferred_tool_compaction(
                merged,
                provider=str(provider or ""),
                session_id=session_id,
                tool_resolver=tool_resolver,
                config=self.config,
            )
            memory_meta = state.setdefault("_memory_meta", {})
            memory_meta.update(compaction_meta)
            prepared = self.strategy.prepare(
                state=state,
                incoming=compacted_input,
                max_context_window_tokens=max_context_window_tokens,
                model=model,
            )
            prepared = self._prepare_long_term_messages(
                state=state,
                prepared=_deepcopy_messages(prepared),
                session_id=session_id,
                memory_namespace=memory_namespace,
                supports_tools=supports_tools,
            )
        finally:
            state.pop("_summary_generator", None)
            state.pop("_memory_model", None)
            state.pop("_session_id", None)

        prepared = _deepcopy_messages(prepared)
        after_tokens = self.estimate_tokens(prepared)
        memory_meta = state.pop("_memory_meta", {})

        persisted_snapshot = session_snapshot
        if session_id and state != loaded_state:
            persisted_snapshot = save_session_snapshot(
                self.store,
                session_id,
                state,
                expected_revision=session_snapshot.revision,
            )

        self._last_prepare_info = {
            "session_id": session_id,
            "memory_namespace": _resolve_memory_namespace(session_id, memory_namespace),
            "before_estimated_tokens": before_tokens,
            "after_estimated_tokens": after_tokens,
            "session_revision": persisted_snapshot.revision,
            "session_revision_supported": persisted_snapshot.revision_supported,
            "session_consistency": persisted_snapshot.consistency,
            **(memory_meta if isinstance(memory_meta, dict) else {}),
        }
        return prepared

    def _commit_long_term_memory(
        self,
        *,
        state: dict[str, Any],
        session_id: str,
        clean_conversation: list[dict[str, Any]],
        commit_info: dict[str, Any],
        memory_namespace: str | None,
        model: str | None,
        long_term_extractor: LongTermExtractor | None,
        execution_fence: ExecutionFence | None,
    ) -> None:
        long_term = self.config.long_term
        if long_term is None:
            return

        namespace = _resolve_memory_namespace(session_id, memory_namespace)
        if not namespace:
            return

        self.ensure_long_term_components()
        commit_info["memory_namespace"] = namespace

        indexed_until_raw = state.get("long_term_indexed_until", 0)
        indexed_until = int(indexed_until_raw) if isinstance(indexed_until_raw, int) else 0
        if indexed_until < 0 or indexed_until > len(clean_conversation):
            indexed_until = 0

        _, turn_metadatas, next_indexed_until, indexed_turn_count = _collect_complete_turns_for_vector_index(
            clean_conversation,
            start_index=indexed_until,
        )
        if not turn_metadatas:
            state["long_term_indexed_until"] = next_indexed_until
            state["long_term_pending_turn_count"] = 0
            return

        pending_turn_count = max(0, indexed_turn_count)
        extract_every_n_turns = max(1, int(long_term.extract_every_n_turns or 1))
        state["long_term_pending_turn_count"] = pending_turn_count
        commit_info["long_term_pending_turn_count"] = pending_turn_count
        commit_info["long_term_extract_every_n_turns"] = extract_every_n_turns

        if pending_turn_count < extract_every_n_turns:
            commit_info["long_term_extraction_deferred"] = True
            return

        extractor = long_term_extractor or long_term.extractor
        if not callable(extractor):
            commit_info["long_term_fallback_reason"] = "long_term_extractor_missing"
            return

        profile_store = long_term.profile_store
        if profile_store is None:
            commit_info["long_term_profile_fallback_reason"] = "profile_store_missing"
            return

        try:
            previous_profile = _normalize_profile_document(profile_store.load(namespace))
        except ExecutionLeaseError:
            raise
        except Exception as exc:
            commit_info["long_term_profile_fallback_reason"] = f"profile_load_failed: {exc}"
            return
        self._verify_execution_fence(execution_fence)

        extraction_messages = _flatten_turn_messages(turn_metadatas)
        try:
            extraction = _call_long_term_extractor(
                extractor,
                previous_profile=previous_profile,
                extraction_messages=extraction_messages,
                long_term=long_term,
                model=model or "",
            )
        except ExecutionLeaseError:
            raise
        except Exception as exc:
            commit_info["long_term_extractor_fallback_reason"] = f"long_term_extraction_failed: {exc}"
            return
        self._verify_execution_fence(execution_fence)

        extraction = extraction if isinstance(extraction, dict) else {}
        profile_patch = _normalize_profile_document(extraction.get("profile_patch"))
        facts = _normalize_long_term_facts(extraction.get("facts"), long_term.max_fact_items)
        episodes = _normalize_long_term_episodes(extraction.get("episodes"), long_term.max_episode_items)
        playbooks = _normalize_long_term_playbooks(extraction.get("playbooks"), long_term.max_playbook_items)

        advance_cursor = True
        did_write = False

        if profile_patch:
            did_write = True
            merged_profile = _merge_profile_documents(previous_profile, profile_patch)
            self._verify_execution_fence(execution_fence)
            try:
                profile_store.save(namespace, merged_profile)
                commit_info["long_term_profile_updated"] = True
                commit_info["long_term_profile_key_count"] = len(merged_profile)
            except ExecutionLeaseError:
                raise
            except Exception as exc:
                commit_info["long_term_profile_fallback_reason"] = f"profile_save_failed: {exc}"
                advance_cursor = False
            self._verify_execution_fence(execution_fence)

        if (facts or episodes or playbooks) and long_term.vector_adapter is not None:
            did_write = True
            texts, metadatas = _build_long_term_memory_payloads(
                facts=facts,
                episodes=episodes,
                playbooks=playbooks,
                session_id=session_id,
                memory_namespace=namespace,
                turn_metadatas=turn_metadatas,
            )
            if texts:
                self._verify_execution_fence(execution_fence)
                try:
                    long_term.vector_adapter.add_texts(
                        namespace=namespace,
                        texts=texts,
                        metadatas=metadatas,
                    )
                    fact_count = len([meta for meta in metadatas if meta.get("memory_type") == "fact"])
                    episode_count = len([meta for meta in metadatas if meta.get("memory_type") == "episode"])
                    playbook_count = len([meta for meta in metadatas if meta.get("memory_type") == "playbook"])
                    commit_info["long_term_fact_indexed_count"] = fact_count
                    commit_info["long_term_episode_indexed_count"] = episode_count
                    commit_info["long_term_playbook_indexed_count"] = playbook_count
                    commit_info["long_term_memory_indexed_count"] = len(texts)
                except ExecutionLeaseError:
                    raise
                except Exception as exc:
                    commit_info["long_term_vector_fallback_reason"] = f"vector_index_failed: {exc}"
                    advance_cursor = False
                self._verify_execution_fence(execution_fence)

        if not did_write:
            commit_info["long_term_noop"] = True

        if advance_cursor:
            state["long_term_indexed_until"] = next_indexed_until
            state["long_term_pending_turn_count"] = 0
            commit_info["long_term_indexed_turn_count"] = indexed_turn_count

    def commit_messages(
        self,
        session_id: str,
        full_conversation: list[dict[str, Any]],
        *,
        memory_namespace: str | None = None,
        model: str | None = None,
        long_term_extractor: LongTermExtractor | None = None,
        expected_revision: int | None = None,
        summary_text: str | None = None,
        return_result: bool = False,
        clear_execution_checkpoint_id: str | None = None,
        execution_fence: ExecutionFence | None = None,
    ) -> MemoryCommitResult | None:
        if execution_fence is not None:
            _validate_execution_fence_target(
                session_id,
                execution_fence.execution_id,
            )
        self._verify_execution_fence(execution_fence)
        session_snapshot = (
            load_session_snapshot(self.store, session_id)
            if session_id
            else SessionSnapshot(state={}, revision=None)
        )
        if (
            expected_revision is not None
            and session_snapshot.revision is not None
            and session_snapshot.revision != expected_revision
        ):
            raise SessionRevisionConflictError(
                session_id=session_id,
                expected_revision=expected_revision,
                actual_revision=session_snapshot.revision,
            )
        state = copy.deepcopy(session_snapshot.state)
        if (
            EXECUTION_CHECKPOINT_KEY in state
            and not clear_execution_checkpoint_id
        ):
            checkpoint = validate_execution_checkpoint(
                state.get(EXECUTION_CHECKPOINT_KEY)
            )
            raise ExecutionCheckpointResumeRequiredError(
                "session contains a durable execution checkpoint "
                f"({checkpoint.get('status')}); semantic commit must resume or "
                "atomically clear that checkpoint"
            )
        checkpoint_cleared = False
        if clear_execution_checkpoint_id:
            current_raw = state.get(EXECUTION_CHECKPOINT_KEY)
            current_checkpoint = (
                validate_execution_checkpoint(current_raw)
                if isinstance(current_raw, dict)
                else None
            )
            current_checkpoint_id = (
                str(current_checkpoint.get("checkpoint_id") or "")
                if isinstance(current_checkpoint, dict)
                else ""
            )
            if current_checkpoint_id != clear_execution_checkpoint_id:
                raise ExecutionCheckpointCompatibilityError(
                    "refusing to commit over a different execution checkpoint"
                )
            if isinstance(current_checkpoint, dict):
                apply_checkpoint_interaction_receipt(
                    state,
                    checkpoint=current_checkpoint,
                    applied_checkpoint_id=clear_execution_checkpoint_id,
                )
            state.pop(EXECUTION_CHECKPOINT_KEY, None)
            state.pop(EXECUTION_CHECKPOINT_DOMAIN_KEY, None)
            checkpoint_cleared = True
        clean_conversation = _deepcopy_messages(full_conversation)
        state["messages"] = clean_conversation

        self.strategy.commit(state=state, full_conversation=clean_conversation)

        commit_info: dict[str, Any] = {
            "session_id": session_id,
            "memory_namespace": _resolve_memory_namespace(session_id, memory_namespace),
            "stored_message_count": len(clean_conversation),
            "execution_checkpoint_cleared": checkpoint_cleared,
        }
        if summary_text is not None:
            normalized_summary = str(summary_text or "").strip()
            state["summary"] = normalized_summary
            commit_info["summary_persisted"] = bool(normalized_summary)
            commit_info["summary_length"] = len(normalized_summary)

        adapter = self.config.vector_adapter
        if adapter is not None:
            indexed_until_raw = state.get("vector_indexed_until", 0)
            indexed_until = int(indexed_until_raw) if isinstance(indexed_until_raw, int) else 0
            if indexed_until < 0 or indexed_until > len(clean_conversation):
                indexed_until = 0
            texts, metadatas, next_indexed_until, indexed_turn_count = _collect_complete_turns_for_vector_index(
                clean_conversation,
                start_index=indexed_until,
            )

            if texts:
                self._verify_execution_fence(execution_fence)
                try:
                    adapter.add_texts(
                        session_id=session_id,
                        texts=texts,
                        metadatas=metadatas,
                    )
                    state["vector_indexed_until"] = next_indexed_until
                    commit_info["vector_indexed_count"] = len(texts)
                    commit_info["vector_indexed_turn_count"] = indexed_turn_count
                except ExecutionLeaseError:
                    raise
                except Exception as exc:
                    commit_info["vector_fallback_reason"] = f"vector_index_failed: {exc}"
                self._verify_execution_fence(execution_fence)
            else:
                state["vector_indexed_until"] = next_indexed_until

        self._commit_long_term_memory(
            state=state,
            session_id=session_id,
            clean_conversation=clean_conversation,
            commit_info=commit_info,
            memory_namespace=memory_namespace,
            model=model,
            long_term_extractor=long_term_extractor,
            execution_fence=execution_fence,
        )

        persisted_snapshot = session_snapshot
        if session_id:
            persisted_snapshot = save_session_snapshot(
                self.store,
                session_id,
                state,
                expected_revision=(
                    expected_revision
                    if expected_revision is not None
                    else session_snapshot.revision
                ),
                execution_fence=execution_fence,
            )
        else:
            persisted_snapshot = SessionSnapshot(state=copy.deepcopy(state), revision=None)
        commit_info["session_revision"] = persisted_snapshot.revision
        commit_info["session_revision_supported"] = persisted_snapshot.revision_supported
        commit_info["session_consistency"] = persisted_snapshot.consistency
        self._last_commit_info = commit_info
        result = MemoryCommitResult(
            commit_info=copy.deepcopy(commit_info),
            session_snapshot=SessionSnapshot(
                state=copy.deepcopy(persisted_snapshot.state),
                revision=persisted_snapshot.revision,
            ),
        )
        return result if return_result else None


__all__ = [
    "ContextStrategy",
    "HybridContextStrategy",
    "InMemorySessionStore",
    "JsonFileLongTermProfileStore",
    "LastNTurnsStrategy",
    "LongTermExtractor",
    "LongTermMemoryConfig",
    "MemoryCommitResult",
    "LongTermProfileStore",
    "LongTermVectorAdapter",
    "MemoryConfig",
    "MemoryManager",
    "SessionStore",
    "SummaryGenerator",
    "SummaryTokenStrategy",
    "VectorStoreAdapter",
]
