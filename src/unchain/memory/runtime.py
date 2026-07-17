from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..execution import (
    ExecutionCancellation,
    ExecutionFence,
    ExecutionLeaseError,
)
from ..interaction.durable import (
    INTERACTION_JOURNAL_KEY,
    build_interaction_receipt,
    record_interaction_receipt,
    validate_interaction_journal,
    validate_interaction_request,
)
from ..tools.receipts import (
    TOOL_RECEIPT_JOURNAL_KEY,
    apply_durable_tool_receipt_ledger,
    ensure_durable_tool_receipt_execution_id,
    validate_durable_tool_receipt_ledger,
)
from .checkpoint_state import (
    EXECUTION_CHECKPOINT_DOMAIN_KEY,
    EXECUTION_CHECKPOINT_KEY,
    ExecutionCheckpointCompatibilityError,
    ExecutionCheckpointIntegrityError,
    ExecutionCheckpointPersistenceError,
    apply_checkpoint_interaction_receipt,
    ensure_checkpoint_compatible,
    merge_checkpoint_transcript_with_incoming,
    register_checkpoint_interaction_request,
    restore_fresh_checkpoint_messages,
    restore_resume_checkpoint_messages,
    transcript_digest,
    validate_execution_checkpoint,
)
from .config import MemoryConfig
from .long_term import LongTermExtractor
from .manager import MemoryManager, SummaryGenerator
from .ownership import ensure_session_delta_input
from .revision import (
    SessionRevisionConflictError,
    SessionSnapshot,
    load_session_snapshot,
    save_session_snapshot,
)
from .stores import InMemorySessionStore, SessionStore


_EXECUTION_CHECKPOINT_DOMAIN_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class _ExecutionCheckpointDomain:
    """Exact root-attempt ownership persisted beside one checkpoint."""

    checkpoint_id: str
    execution_id: str
    owner_id: str
    fencing_token: int

    @classmethod
    def from_fence(
        cls,
        checkpoint: dict[str, Any],
        fence: ExecutionFence,
    ) -> "_ExecutionCheckpointDomain":
        return cls(
            checkpoint_id=str(checkpoint.get("checkpoint_id") or ""),
            execution_id=fence.execution_id,
            owner_id=fence.owner_id,
            fencing_token=fence.fencing_token,
        )

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        checkpoint: dict[str, Any],
        session_id: str,
    ) -> "_ExecutionCheckpointDomain | None":
        raw = state.get(EXECUTION_CHECKPOINT_DOMAIN_KEY)
        if raw is None:
            return None
        expected_fields = {
            "schema_version",
            "checkpoint_id",
            "execution_id",
            "owner_id",
            "fencing_token",
        }
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise ExecutionCheckpointIntegrityError(
                "execution checkpoint domain has an invalid shape"
            )
        if raw.get("schema_version") != _EXECUTION_CHECKPOINT_DOMAIN_SCHEMA_VERSION:
            raise ExecutionCheckpointIntegrityError(
                "unsupported execution checkpoint domain schema_version"
            )
        checkpoint_id = raw.get("checkpoint_id")
        execution_id = raw.get("execution_id")
        owner_id = raw.get("owner_id")
        fencing_token = raw.get("fencing_token")
        if (
            not isinstance(checkpoint_id, str)
            or not checkpoint_id
            or checkpoint_id != checkpoint.get("checkpoint_id")
        ):
            raise ExecutionCheckpointIntegrityError(
                "execution checkpoint domain is bound to a different checkpoint"
            )
        if not isinstance(execution_id, str) or not execution_id:
            raise ExecutionCheckpointIntegrityError(
                "execution checkpoint domain requires execution_id"
            )
        if not isinstance(owner_id, str) or not owner_id:
            raise ExecutionCheckpointIntegrityError(
                "execution checkpoint domain requires owner_id"
            )
        if (
            isinstance(fencing_token, bool)
            or not isinstance(fencing_token, int)
            or fencing_token <= 0
        ):
            raise ExecutionCheckpointIntegrityError(
                "execution checkpoint domain requires a positive fencing_token"
            )
        if not (
            session_id == execution_id
            or session_id.startswith(f"{execution_id}:")
        ):
            raise ExecutionCheckpointIntegrityError(
                "execution checkpoint domain cannot own the target session"
            )
        return cls(
            checkpoint_id=checkpoint_id,
            execution_id=execution_id,
            owner_id=owner_id,
            fencing_token=fencing_token,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _EXECUTION_CHECKPOINT_DOMAIN_SCHEMA_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "execution_id": self.execution_id,
            "owner_id": self.owner_id,
            "fencing_token": self.fencing_token,
        }


def _deepcopy_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [copy.deepcopy(message) for message in (messages or []) if isinstance(message, dict)]


def _split_system_and_non_system(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    systems: list[dict[str, Any]] = []
    non_system: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system":
            systems.append(copy.deepcopy(message))
        else:
            non_system.append(copy.deepcopy(message))
    return systems, non_system


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


def _ensure_expected_revision(
    snapshot: SessionSnapshot,
    *,
    session_id: str,
    expected_revision: int | None,
) -> None:
    if (
        expected_revision is not None
        and snapshot.revision is not None
        and snapshot.revision != expected_revision
    ):
        raise SessionRevisionConflictError(
            session_id=session_id,
            expected_revision=expected_revision,
            actual_revision=snapshot.revision,
        )


def _verify_execution_fence(store: object, fence: ExecutionFence | None) -> None:
    if fence is None:
        return
    verify_lease = getattr(store, "verify_lease", None)
    if not callable(verify_lease):
        raise TypeError("execution fence requires a lease-aware session store")
    verify_lease(
        fence.execution_id,
        fence.owner_id,
        fence.fencing_token,
    )


def _ensure_checkpoint_extends_semantic_history(
    *,
    history: list[dict[str, Any]],
    checkpoint: dict[str, Any],
) -> None:
    if not history:
        return
    checkpoint_transcript = checkpoint.get("transcript")
    if (
        not isinstance(checkpoint_transcript, list)
        or checkpoint_transcript[: len(history)] != history
    ):
        raise ExecutionCheckpointCompatibilityError(
            "stored semantic history diverges from the execution checkpoint; "
            "refusing to restore a stale checkpoint"
        )


@dataclass
class KernelMemoryRuntime:
    memory_manager: MemoryManager
    summary_generator: SummaryGenerator | None = None
    long_term_extractor: LongTermExtractor | None = None
    last_prepare_info: dict[str, Any] = field(default_factory=dict)
    last_commit_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.long_term_extractor is None:
            long_term = getattr(self.memory_manager.config, "long_term", None)
            self.long_term_extractor = getattr(long_term, "extractor", None)

    @classmethod
    def from_memory_manager(
        cls,
        memory_manager: MemoryManager,
        *,
        summary_generator: SummaryGenerator | None = None,
        long_term_extractor: LongTermExtractor | None = None,
    ) -> "KernelMemoryRuntime":
        if not isinstance(memory_manager, MemoryManager):
            raise TypeError("memory_manager must be a MemoryManager")
        return cls(
            memory_manager=memory_manager,
            summary_generator=summary_generator,
            long_term_extractor=long_term_extractor,
        )

    @classmethod
    def from_config(
        cls,
        config: MemoryConfig | None = None,
        *,
        store: SessionStore | None = None,
        summary_generator: SummaryGenerator | None = None,
        long_term_extractor: LongTermExtractor | None = None,
    ) -> "KernelMemoryRuntime":
        manager = MemoryManager(
            config=copy.deepcopy(config) if config is not None else MemoryConfig(),
            store=store if store is not None else InMemorySessionStore(),
        )
        return cls.from_memory_manager(
            manager,
            summary_generator=summary_generator,
            long_term_extractor=long_term_extractor,
        )

    @property
    def config(self) -> MemoryConfig:
        return self.memory_manager.config

    @property
    def store(self) -> SessionStore:
        return self.memory_manager.store

    def ensure_long_term_components(self) -> None:
        try:
            self.memory_manager.ensure_long_term_components()
        except ExecutionLeaseError:
            raise
        except Exception:
            return

    def load_session_state(self, session_id: str) -> dict[str, Any]:
        if not session_id:
            return {}
        return copy.deepcopy(self.load_session_snapshot(session_id).state)

    def load_session_snapshot(self, session_id: str) -> SessionSnapshot:
        if not session_id:
            return SessionSnapshot(state={}, revision=None)
        return load_session_snapshot(self.store, session_id)

    def save_session_state(
        self,
        session_id: str,
        state: dict[str, Any],
        *,
        expected_revision: int | None = None,
        execution_fence: ExecutionFence | None = None,
    ) -> SessionSnapshot:
        if not session_id:
            return SessionSnapshot(state={}, revision=None)
        current = self.load_session_snapshot(session_id)
        _ensure_expected_revision(
            current,
            session_id=session_id,
            expected_revision=expected_revision,
        )
        return save_session_snapshot(
            self.store,
            session_id,
            copy.deepcopy(state if isinstance(state, dict) else {}),
            expected_revision=(
                expected_revision
                if expected_revision is not None
                else current.revision
            ),
            execution_fence=execution_fence,
        )

    def _checkpoint_domain(
        self,
        session_id: str,
        session_snapshot: SessionSnapshot,
    ) -> tuple[dict[str, Any], _ExecutionCheckpointDomain] | None:
        checkpoint_raw = session_snapshot.state.get(EXECUTION_CHECKPOINT_KEY)
        if not isinstance(checkpoint_raw, dict):
            return None
        checkpoint = validate_execution_checkpoint(checkpoint_raw)
        domain = _ExecutionCheckpointDomain.from_state(
            session_snapshot.state,
            checkpoint=checkpoint,
            session_id=session_id,
        )
        if domain is None:
            return None
        return checkpoint, domain

    def reconcile_cancelled_execution_checkpoint(
        self,
        session_id: str,
        *,
        submitted_by: str = "runtime:cancelled_execution_tree",
    ) -> tuple[SessionSnapshot, ExecutionCancellation | None]:
        """Terminalize a wait whose exact owning attempt was cancelled.

        Hosts normally terminalize the root wait in their Stop transaction.
        This reconciliation is the crash-safe fallback for root and descendant
        checkpoints.  The persisted owner binding and cancellation-aware CAS
        ensure that cleanup can never consume a newer attempt's checkpoint.
        """

        if not session_id:
            return SessionSnapshot(state={}, revision=None), None

        last_conflict: SessionRevisionConflictError | None = None
        for _ in range(16):
            session_snapshot = self.load_session_snapshot(session_id)
            bound = self._checkpoint_domain(session_id, session_snapshot)
            if bound is None:
                return session_snapshot, None
            checkpoint, domain = bound

            load_cancellation = getattr(
                self.store,
                "load_execution_cancellation",
                None,
            )
            save_cancelled = getattr(
                self.store,
                "save_if_revision_and_execution_cancelled",
                None,
            )
            if not callable(load_cancellation) or not callable(save_cancelled):
                raise TypeError(
                    "execution checkpoint cancellation requires atomic "
                    "cancellation-aware session storage"
                )

            cancellation = load_cancellation(
                domain.execution_id,
                domain.owner_id,
            )
            if cancellation is None:
                return session_snapshot, None
            if not isinstance(cancellation, ExecutionCancellation):
                raise ExecutionLeaseError(
                    "execution cancellation store returned an invalid tombstone",
                    execution_id=domain.execution_id,
                    owner_id=domain.owner_id,
                    fencing_token=domain.fencing_token,
                )
            if (
                cancellation.execution_id != domain.execution_id
                or cancellation.owner_id != domain.owner_id
            ):
                raise ExecutionLeaseError(
                    "execution cancellation store returned a different owner",
                    execution_id=domain.execution_id,
                    owner_id=domain.owner_id,
                    fencing_token=domain.fencing_token,
                )

            state = copy.deepcopy(session_snapshot.state)
            interaction_ref = checkpoint.get("interaction_ref")
            if isinstance(interaction_ref, dict):
                journal = validate_interaction_journal(
                    state.get(INTERACTION_JOURNAL_KEY)
                )
                interaction_id = str(
                    interaction_ref.get("interaction_id") or ""
                )
                entry = journal["entries"].get(interaction_id)
                if (
                    not isinstance(entry, dict)
                    or entry.get("checkpoint_id") != checkpoint.get("checkpoint_id")
                ):
                    raise ExecutionCheckpointIntegrityError(
                        "cancelled checkpoint is missing its interaction"
                    )
                if entry.get("receipt") is None:
                    request = validate_interaction_request(entry.get("request"))
                    cancellation_receipt = build_interaction_receipt(
                        request,
                        {
                            "cancelled": True,
                            "reason": (
                                cancellation.reason or "execution_cancelled"
                            ),
                        },
                        submitted_by=submitted_by,
                        submitted_at_ms=cancellation.requested_at_ms,
                    )
                    journal = record_interaction_receipt(
                        journal,
                        cancellation_receipt,
                    )
                    state[INTERACTION_JOURNAL_KEY] = journal
                apply_checkpoint_interaction_receipt(
                    state,
                    checkpoint=checkpoint,
                    applied_checkpoint_id=(
                        f"cancelled:{checkpoint.get('checkpoint_id')}"
                    ),
                )

            state.pop(EXECUTION_CHECKPOINT_KEY, None)
            state.pop(EXECUTION_CHECKPOINT_DOMAIN_KEY, None)
            expected_revision = session_snapshot.revision
            if expected_revision is None:
                raise TypeError(
                    "execution checkpoint cancellation requires revisioned storage"
                )
            try:
                next_revision = save_cancelled(
                    session_id,
                    state,
                    expected_revision,
                    execution_id=domain.execution_id,
                    owner_id=domain.owner_id,
                )
            except SessionRevisionConflictError as exc:
                last_conflict = exc
                continue
            if (
                isinstance(next_revision, bool)
                or not isinstance(next_revision, int)
                or next_revision <= expected_revision
            ):
                raise ExecutionCheckpointPersistenceError(
                    "cancellation-aware checkpoint CAS returned an invalid revision"
                )
            persisted = SessionSnapshot(
                state=copy.deepcopy(state),
                revision=next_revision,
            )
            return persisted, cancellation

        if last_conflict is not None:
            raise last_conflict
        raise ExecutionCheckpointPersistenceError(
            "cancelled checkpoint reconciliation did not converge"
        )

    def resolve_interaction_execution_fence(
        self,
        session_id: str,
        *,
        checkpoint_id: str,
        session_snapshot: SessionSnapshot,
        execution_fence: ExecutionFence | None,
    ) -> ExecutionFence | None:
        """Validate any live fence supplied for an interaction receipt."""

        bound = self._checkpoint_domain(session_id, session_snapshot)
        if bound is None:
            return execution_fence
        checkpoint, domain = bound
        if checkpoint.get("checkpoint_id") != checkpoint_id:
            raise ExecutionCheckpointCompatibilityError(
                "interaction receipt belongs to a different checkpoint"
            )
        if domain.execution_id == session_id:
            return execution_fence
        if execution_fence is not None:
            if (
                execution_fence.execution_id != domain.execution_id
                or execution_fence.owner_id != domain.owner_id
            ):
                raise ExecutionCheckpointCompatibilityError(
                    "interaction receipt execution fence does not match its checkpoint"
                )
            return execution_fence
        return None

    def save_interaction_session_state(
        self,
        session_id: str,
        state: dict[str, Any],
        *,
        checkpoint_id: str,
        session_snapshot: SessionSnapshot,
        expected_revision: int | None,
        execution_fence: ExecutionFence | None,
    ) -> SessionSnapshot:
        """Persist a receipt without reopening a cancelled execution tree."""

        effective_fence = self.resolve_interaction_execution_fence(
            session_id,
            checkpoint_id=checkpoint_id,
            session_snapshot=session_snapshot,
            execution_fence=execution_fence,
        )
        bound = self._checkpoint_domain(session_id, session_snapshot)
        if bound is None or effective_fence is not None:
            return self.save_session_state(
                session_id,
                state,
                expected_revision=expected_revision,
                execution_fence=effective_fence,
            )

        _, domain = bound
        save_not_cancelled = getattr(
            self.store,
            "save_if_revision_and_execution_not_cancelled",
            None,
        )
        if not callable(save_not_cancelled):
            raise TypeError(
                "descendant interaction receipt requires atomic "
                "cancellation-aware session storage"
            )
        revision = (
            expected_revision
            if expected_revision is not None
            else session_snapshot.revision
        )
        if revision is None:
            raise TypeError(
                "descendant interaction receipt requires revisioned session storage"
            )
        next_revision = save_not_cancelled(
            session_id,
            copy.deepcopy(state),
            revision,
            execution_id=domain.execution_id,
            owner_id=domain.owner_id,
        )
        if (
            isinstance(next_revision, bool)
            or not isinstance(next_revision, int)
            or next_revision <= revision
        ):
            raise ExecutionCheckpointPersistenceError(
                "cancellation-aware receipt CAS returned an invalid revision"
            )
        return SessionSnapshot(
            state=copy.deepcopy(state),
            revision=next_revision,
        )

    def load_execution_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        if not session_id:
            return None
        session_snapshot, cancellation = (
            self.reconcile_cancelled_execution_checkpoint(session_id)
        )
        if cancellation is not None:
            return None
        state = session_snapshot.state
        if EXECUTION_CHECKPOINT_KEY not in state:
            return None
        return validate_execution_checkpoint(state.get(EXECUTION_CHECKPOINT_KEY))

    def save_execution_checkpoint(
        self,
        session_id: str,
        checkpoint: dict[str, Any],
        *,
        interaction_request: Any = None,
        execution_fence: ExecutionFence | None = None,
    ) -> dict[str, Any]:
        persisted, _ = self.save_execution_checkpoint_snapshot(
            session_id,
            checkpoint,
            interaction_request=interaction_request,
            execution_fence=execution_fence,
        )
        return persisted

    def save_execution_checkpoint_snapshot(
        self,
        session_id: str,
        checkpoint: dict[str, Any],
        *,
        interaction_request: Any = None,
        durable_receipt_ledger: Any = None,
        expected_revision: int | None = None,
        execution_fence: ExecutionFence | None = None,
    ) -> tuple[dict[str, Any], SessionSnapshot]:
        if not session_id:
            raise ExecutionCheckpointPersistenceError(
                "execution checkpoint persistence requires a non-empty session_id"
            )
        validated = validate_execution_checkpoint(checkpoint)
        if validated.get("session_id") != session_id:
            raise ExecutionCheckpointCompatibilityError(
                "execution checkpoint session_id does not match the target session"
            )
        has_interaction_ref = isinstance(validated.get("interaction_ref"), dict)
        if has_interaction_ref and interaction_request is None:
            raise ExecutionCheckpointCompatibilityError(
                "checkpoint interaction_ref requires its full interaction_request"
            )
        if not has_interaction_ref and interaction_request is not None:
            raise ExecutionCheckpointCompatibilityError(
                "interaction_request requires a checkpoint interaction_ref"
            )
        session_snapshot = self.load_session_snapshot(session_id)
        current_raw = session_snapshot.state.get(EXECUTION_CHECKPOINT_KEY)
        current = None
        if isinstance(current_raw, dict):
            current = validate_execution_checkpoint(current_raw)
            current_revision = session_snapshot.revision
            same_checkpoint = (
                current.get("checkpoint_id") == validated.get("checkpoint_id")
            )
            same_revision_noop = current_revision == expected_revision
            verified_retry = (
                expected_revision is not None
                and current_revision == expected_revision + 1
                and current.get("base_session_revision") == expected_revision
            )
            retry_candidate = (
                expected_revision is not None
                and same_checkpoint
                and (same_revision_noop or verified_retry)
            )
            if not retry_candidate:
                _ensure_expected_revision(
                    session_snapshot,
                    session_id=session_id,
                    expected_revision=expected_revision,
                )
        else:
            retry_candidate = False
            _ensure_expected_revision(
                session_snapshot,
                session_id=session_id,
                expected_revision=expected_revision,
            )

        state = copy.deepcopy(session_snapshot.state)
        if has_interaction_ref:
            register_checkpoint_interaction_request(
                state,
                checkpoint=validated,
                interaction_request=interaction_request,
            )
        elif (
            isinstance(current, dict)
            and isinstance(current.get("interaction_ref"), dict)
        ):
            apply_checkpoint_interaction_receipt(
                state,
                checkpoint=current,
                applied_checkpoint_id=str(validated.get("checkpoint_id") or ""),
            )
        state[EXECUTION_CHECKPOINT_KEY] = copy.deepcopy(validated)
        receipt_ledger = ensure_durable_tool_receipt_execution_id(
            durable_receipt_ledger,
            execution_id=session_id,
        )
        if receipt_ledger["pending"]:
            apply_durable_tool_receipt_ledger(
                state,
                ledger=receipt_ledger,
                application_id=str(validated.get("checkpoint_id") or ""),
            )
        if execution_fence is not None:
            cancellation_aware = all(
                callable(getattr(self.store, method, None))
                for method in (
                    "load_execution_cancellation",
                    "save_if_revision_and_execution_not_cancelled",
                    "save_if_revision_and_execution_cancelled",
                )
            )
            if session_id != execution_fence.execution_id and not cancellation_aware:
                raise TypeError(
                    "descendant execution checkpoint requires atomic "
                    "cancellation-aware session storage"
                )
            if cancellation_aware:
                next_domain = _ExecutionCheckpointDomain.from_fence(
                    validated,
                    execution_fence,
                )
                existing_domain = (
                    _ExecutionCheckpointDomain.from_state(
                        session_snapshot.state,
                        checkpoint=current,
                        session_id=session_id,
                    )
                    if retry_candidate and isinstance(current, dict)
                    else None
                )
                if (
                    existing_domain is not None
                    and existing_domain.execution_id == next_domain.execution_id
                    and existing_domain.owner_id == next_domain.owner_id
                ):
                    state[EXECUTION_CHECKPOINT_DOMAIN_KEY] = (
                        existing_domain.to_dict()
                    )
                elif (
                    retry_candidate
                    and EXECUTION_CHECKPOINT_DOMAIN_KEY
                    not in session_snapshot.state
                ):
                    state.pop(EXECUTION_CHECKPOINT_DOMAIN_KEY, None)
                else:
                    state[EXECUTION_CHECKPOINT_DOMAIN_KEY] = next_domain.to_dict()
            else:
                state.pop(EXECUTION_CHECKPOINT_DOMAIN_KEY, None)
        else:
            state.pop(EXECUTION_CHECKPOINT_DOMAIN_KEY, None)
        if retry_candidate:
            if state != session_snapshot.state:
                raise ExecutionCheckpointIntegrityError(
                    "checkpoint retry found a non-atomic interaction journal state"
                )
            if isinstance(current, dict):
                _verify_execution_fence(self.store, execution_fence)
                return current, session_snapshot
        try:
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
        except Exception as exc:
            if isinstance(exc, (SessionRevisionConflictError, ExecutionLeaseError)):
                raise
            raise ExecutionCheckpointPersistenceError(
                f"failed to persist execution checkpoint: {exc}"
            ) from exc
        try:
            verified_snapshot = self.load_session_snapshot(session_id)
            persisted_raw = verified_snapshot.state.get(EXECUTION_CHECKPOINT_KEY)
            persisted = (
                validate_execution_checkpoint(persisted_raw)
                if isinstance(persisted_raw, dict)
                else None
            )
        except Exception as exc:
            raise ExecutionCheckpointPersistenceError(
                f"persisted execution checkpoint could not be verified: {exc}"
            ) from exc
        if (
            not isinstance(persisted, dict)
            or persisted.get("checkpoint_id") != validated.get("checkpoint_id")
        ):
            if (
                persisted_snapshot.revision is not None
                and verified_snapshot.revision is not None
                and verified_snapshot.revision != persisted_snapshot.revision
            ):
                raise SessionRevisionConflictError(
                    session_id=session_id,
                    expected_revision=persisted_snapshot.revision,
                    actual_revision=verified_snapshot.revision,
                )
            raise ExecutionCheckpointPersistenceError(
                "execution checkpoint write verification failed"
            )
        if has_interaction_ref:
            try:
                request = validate_interaction_request(interaction_request)
                journal = validate_interaction_journal(
                    verified_snapshot.state.get(INTERACTION_JOURNAL_KEY)
                )
                entry = journal["entries"].get(request.interaction_id)
            except Exception as exc:
                raise ExecutionCheckpointPersistenceError(
                    f"persisted interaction journal could not be verified: {exc}"
                ) from exc
            if (
                journal.get("active_id") != request.interaction_id
                or not isinstance(entry, dict)
                or entry.get("request") != request.to_dict()
                or entry.get("checkpoint_id") != validated.get("checkpoint_id")
                or entry.get("application") is not None
            ):
                raise ExecutionCheckpointPersistenceError(
                    "interaction journal write verification failed"
                )
        if receipt_ledger["pending"]:
            try:
                receipt_journal = validate_durable_tool_receipt_ledger(
                    verified_snapshot.state.get(TOOL_RECEIPT_JOURNAL_KEY)
                )
            except Exception as exc:
                raise ExecutionCheckpointPersistenceError(
                    f"persisted tool receipt journal could not be verified: {exc}"
                ) from exc
            for receipt_key, receipt in receipt_ledger["pending"].items():
                application = receipt_journal["applications"].get(receipt_key)
                if (
                    not isinstance(application, dict)
                    or application.get("receipt") != receipt
                    or application.get("application_id")
                    != validated.get("checkpoint_id")
                ):
                    raise ExecutionCheckpointPersistenceError(
                        "tool receipt journal write verification failed"
                    )
        return persisted, persisted_snapshot

    def clear_execution_checkpoint(
        self,
        session_id: str,
        *,
        expected_checkpoint_id: str | None = None,
        execution_fence: ExecutionFence | None = None,
    ) -> bool:
        cleared, _ = self.clear_execution_checkpoint_snapshot(
            session_id,
            expected_checkpoint_id=expected_checkpoint_id,
            execution_fence=execution_fence,
        )
        return cleared

    def clear_execution_checkpoint_snapshot(
        self,
        session_id: str,
        *,
        expected_checkpoint_id: str | None = None,
        expected_revision: int | None = None,
        execution_fence: ExecutionFence | None = None,
    ) -> tuple[bool, SessionSnapshot]:
        if not session_id:
            return False, SessionSnapshot(state={}, revision=None)
        session_snapshot = self.load_session_snapshot(session_id)
        _ensure_expected_revision(
            session_snapshot,
            session_id=session_id,
            expected_revision=expected_revision,
        )
        state = copy.deepcopy(session_snapshot.state)
        if EXECUTION_CHECKPOINT_KEY not in state:
            _verify_execution_fence(self.store, execution_fence)
            return False, session_snapshot
        current = validate_execution_checkpoint(state.get(EXECUTION_CHECKPOINT_KEY))
        if (
            expected_checkpoint_id
            and current.get("checkpoint_id") != expected_checkpoint_id
        ):
            raise ExecutionCheckpointCompatibilityError(
                "refusing to clear a different execution checkpoint"
            )
        interaction_applied = apply_checkpoint_interaction_receipt(
            state,
            checkpoint=current,
            applied_checkpoint_id=str(current.get("checkpoint_id") or ""),
        )
        interaction_id = str(
            (current.get("interaction_ref") or {}).get("interaction_id") or ""
        )
        state.pop(EXECUTION_CHECKPOINT_KEY, None)
        state.pop(EXECUTION_CHECKPOINT_DOMAIN_KEY, None)
        try:
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
        except Exception as exc:
            if isinstance(exc, (SessionRevisionConflictError, ExecutionLeaseError)):
                raise
            raise ExecutionCheckpointPersistenceError(
                f"failed to clear execution checkpoint: {exc}"
            ) from exc
        verified_snapshot = self.load_session_snapshot(session_id)
        if EXECUTION_CHECKPOINT_KEY in verified_snapshot.state:
            if (
                persisted_snapshot.revision is not None
                and verified_snapshot.revision is not None
                and verified_snapshot.revision != persisted_snapshot.revision
            ):
                raise SessionRevisionConflictError(
                    session_id=session_id,
                    expected_revision=persisted_snapshot.revision,
                    actual_revision=verified_snapshot.revision,
                )
            raise ExecutionCheckpointPersistenceError(
                "execution checkpoint clear verification failed"
            )
        if interaction_applied:
            try:
                journal = validate_interaction_journal(
                    verified_snapshot.state.get(INTERACTION_JOURNAL_KEY)
                )
                entry = journal["entries"].get(interaction_id)
            except Exception as exc:
                raise ExecutionCheckpointPersistenceError(
                    f"cleared interaction journal could not be verified: {exc}"
                ) from exc
            if (
                journal.get("active_id") is not None
                or not isinstance(entry, dict)
                or not isinstance(entry.get("application"), dict)
            ):
                raise ExecutionCheckpointPersistenceError(
                    "interaction receipt application verification failed"
                )
        return True, persisted_snapshot

    def bootstrap_session(
        self,
        *,
        session_id: str,
        memory_namespace: str | None,
        incoming_messages: list[dict[str, Any]],
        resume_mode: bool,
        provider: str | None = None,
        model: str | None = None,
        resume_continuation: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str]:
        if not session_id:
            prepare_info = {
                "applied": False,
                "session_id": "",
                "memory_namespace": _resolve_memory_namespace(session_id, memory_namespace),
                "resume_mode": bool(resume_mode),
                "skip_reason": "missing_session_id",
            }
            self.last_prepare_info = copy.deepcopy(prepare_info)
            return _deepcopy_messages(incoming_messages), {}, prepare_info, ""

        session_snapshot, _ = self.reconcile_cancelled_execution_checkpoint(
            session_id
        )
        loaded_state = copy.deepcopy(session_snapshot.state)
        history = loaded_state.get("messages", [])
        if not isinstance(history, list):
            history = []
        summary_text = str(loaded_state.get("summary", "") or "").strip()
        resolved_namespace = _resolve_memory_namespace(session_id, memory_namespace)
        checkpoint = None
        if EXECUTION_CHECKPOINT_KEY in loaded_state:
            checkpoint = validate_execution_checkpoint(
                loaded_state.get(EXECUTION_CHECKPOINT_KEY)
            )
        checkpoint_restored = False
        if checkpoint is not None:
            _ensure_checkpoint_extends_semantic_history(
                history=history,
                checkpoint=checkpoint,
            )
            ensure_checkpoint_compatible(
                checkpoint,
                session_id=session_id,
                provider=provider,
                model=model,
            )
            if resume_mode:
                checkpoint_messages = restore_resume_checkpoint_messages(
                    checkpoint,
                    conversation=incoming_messages,
                    continuation=resume_continuation,
                )
                merged_messages = _deepcopy_messages(checkpoint_messages)
            else:
                checkpoint_messages = restore_fresh_checkpoint_messages(
                    checkpoint,
                    incoming_messages=incoming_messages,
                )
                merged_messages = merge_checkpoint_transcript_with_incoming(
                    checkpoint_messages,
                    incoming_messages,
                )
            checkpoint_restored = True
        elif resume_mode:
            merged_messages = _deepcopy_messages(incoming_messages)
        else:
            ensure_session_delta_input(history=history, incoming=incoming_messages)
            merged_messages = _merge_history_and_incoming(history, incoming_messages)
        self.ensure_long_term_components()
        prepare_info = {
            "applied": True,
            "session_id": session_id,
            "memory_namespace": resolved_namespace,
            "resume_mode": bool(resume_mode),
            "loaded_message_count": len(history),
            "merged_message_count": len(merged_messages),
            "summary_present": bool(summary_text),
            "vector_indexed_until": int(loaded_state.get("vector_indexed_until") or 0),
            "long_term_indexed_until": int(loaded_state.get("long_term_indexed_until") or 0),
            "long_term_pending_turn_count": int(loaded_state.get("long_term_pending_turn_count") or 0),
            "execution_checkpoint_restored": checkpoint_restored,
            "execution_checkpoint_status": (
                str(checkpoint.get("status") or "")
                if isinstance(checkpoint, dict)
                else ""
            ),
            "execution_checkpoint_id": (
                str(checkpoint.get("checkpoint_id") or "")
                if isinstance(checkpoint, dict)
                else ""
            ),
            "session_revision": session_snapshot.revision,
            "session_revision_supported": session_snapshot.revision_supported,
            "session_consistency": session_snapshot.consistency,
        }
        self.last_prepare_info = copy.deepcopy(prepare_info)
        return merged_messages, loaded_state, prepare_info, summary_text

    def recall_profile(
        self,
        *,
        session_id: str,
        memory_namespace: str | None = None,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        self.ensure_long_term_components()
        return self.memory_manager.recall_profile(
            session_id=session_id,
            memory_namespace=memory_namespace,
            max_chars=max_chars,
        )

    def recall_memory(
        self,
        *,
        session_id: str,
        memory_namespace: str | None,
        query: str,
        include_short_term: bool,
        include_long_term: bool,
    ) -> dict[str, Any]:
        self.ensure_long_term_components()
        return self.memory_manager.recall_memory(
            session_id=session_id,
            memory_namespace=memory_namespace,
            query=query,
            include_short_term=include_short_term,
            include_long_term=include_long_term,
        )

    def recall_long_term(
        self,
        *,
        session_id: str,
        memory_namespace: str | None,
        query: str,
    ) -> dict[str, Any]:
        self.ensure_long_term_components()
        facts = self.memory_manager._long_term_recall_result(
            session_id=session_id,
            memory_namespace=memory_namespace,
            query=query,
            memory_type="fact",
            apply_query_hints=False,
        )
        episodes = self.memory_manager._long_term_recall_result(
            session_id=session_id,
            memory_namespace=memory_namespace,
            query=query,
            memory_type="episode",
            apply_query_hints=True,
        )
        playbooks = self.memory_manager._long_term_recall_result(
            session_id=session_id,
            memory_namespace=memory_namespace,
            query=query,
            memory_type="playbook",
            apply_query_hints=True,
        )
        output = {
            "facts": copy.deepcopy(facts.get("items", [])),
            "episodes": copy.deepcopy(episodes.get("items", [])),
            "playbooks": copy.deepcopy(playbooks.get("items", [])),
            "counts": {
                "facts": int(facts.get("count") or 0),
                "episodes": int(episodes.get("count") or 0),
                "playbooks": int(playbooks.get("count") or 0),
            },
            "hit_counts": {
                "facts": int(facts.get("hit_count") or 0),
                "episodes": int(episodes.get("hit_count") or 0),
                "playbooks": int(playbooks.get("hit_count") or 0),
            },
            "available": any(bool(item.get("available")) for item in (facts, episodes, playbooks)),
        }
        fallback_reasons = {
            "facts": str(facts.get("fallback_reason") or "").strip(),
            "episodes": str(episodes.get("fallback_reason") or "").strip(),
            "playbooks": str(playbooks.get("fallback_reason") or "").strip(),
        }
        if any(fallback_reasons.values()):
            output["fallback_reasons"] = {
                key: value
                for key, value in fallback_reasons.items()
                if value
            }
        return output

    def commit_transcript(
        self,
        *,
        session_id: str,
        transcript: list[dict[str, Any]],
        memory_namespace: str | None,
        model: str | None,
        summary_text: str | None,
        expected_revision: int | None = None,
        expected_checkpoint_id: str | None = None,
        durable_receipt_ledger: Any = None,
        execution_fence: ExecutionFence | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not session_id:
            commit_info = {
                "applied": False,
                "session_id": "",
                "memory_namespace": _resolve_memory_namespace(session_id, memory_namespace),
                "skip_reason": "missing_session_id",
            }
            self.last_commit_info = copy.deepcopy(commit_info)
            return commit_info, {}

        self.ensure_long_term_components()
        outcome = self.memory_manager.commit_messages(
            session_id=session_id,
            full_conversation=_deepcopy_messages(transcript),
            memory_namespace=memory_namespace,
            model=model,
            long_term_extractor=self.long_term_extractor,
            expected_revision=expected_revision,
            summary_text=str(summary_text or "").strip(),
            return_result=True,
            clear_execution_checkpoint_id=expected_checkpoint_id,
            durable_receipt_ledger=durable_receipt_ledger,
            execution_fence=execution_fence,
        )
        if outcome is None:
            raise RuntimeError("memory commit did not return its persisted session snapshot")
        stored_state = copy.deepcopy(outcome.session_snapshot.state)
        commit_info = copy.deepcopy(outcome.commit_info)
        commit_info["applied"] = True
        commit_info["transcript_digest"] = transcript_digest(transcript)
        self.last_commit_info = copy.deepcopy(commit_info)
        return commit_info, stored_state

    def build_default_components(self) -> list[Any]:
        from .assembly import build_default_memory_components

        return build_default_memory_components(self)
