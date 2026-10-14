from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from unchain.journal import ResourceRef
from unchain.memory.curator.models import CuratorLeaseFence
from unchain.memory.curator.ports import BoundCuratorMutationGuard

from .models import MemoryToolContentPage, MemoryToolkitError, ReferencePurpose


MAX_RECALLED_LONG_TERM_REFS = 5


@runtime_checkable
class BoundExternalReferenceCodec(Protocol):
    binding_id: str

    def decode(self, value: str, *, purpose: ReferencePurpose) -> ResourceRef:
        ...

    def encode(self, ref: ResourceRef) -> str:
        ...


@runtime_checkable
class BoundContextMemoryCapability(Protocol):
    binding_id: str

    def authorize(
        self,
        *,
        ref: ResourceRef,
        purpose: ReferencePurpose,
    ) -> ResourceRef: ...

    def read_content(
        self,
        *,
        ref: ResourceRef,
        offset: int,
        limit: int,
    ) -> MemoryToolContentPage: ...

    def read_checkpoint_events(
        self,
        *,
        ref: ResourceRef,
        after_position: int,
        limit: int,
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class BoundMemoryReadCapability(Protocol):
    binding_id: str
    space_id: str
    space_revision: int


@runtime_checkable
class BoundCandidateProposalCapability(Protocol):
    binding_id: str


@runtime_checkable
class BoundMemoryCuratorCapability(BoundMemoryReadCapability, Protocol):
    pass


@runtime_checkable
class BoundPromotionProposalCapability(Protocol):
    binding_id: str
    target_namespace: str


@runtime_checkable
class BoundTaskStateCapability(Protocol):
    binding_id: str


@runtime_checkable
class BoundConsolidationCapability(Protocol):
    binding_id: str

    def read_candidate(
        self,
        *,
        job_id: str,
        ref: ResourceRef,
        offset: int,
        limit: int,
    ) -> MemoryToolContentPage: ...

    def apply_new(
        self,
        *,
        job_id: str,
        candidate_ref: ResourceRef,
        expected_binding_revision: int,
        expected_space_revision: int,
        mutation_guard: BoundCuratorMutationGuard,
        operation_id: str,
    ) -> Mapping[str, Any]: ...

    def propose_review(
        self,
        *,
        job_id: str,
        candidate_ref: ResourceRef,
        expected_binding_revision: int,
        target_entry_id: str,
        expected_target_revision: int,
        mode: str,
        mutation_guard: BoundCuratorMutationGuard,
        operation_id: str,
    ) -> Mapping[str, Any]: ...


_CONSOLIDATION_METHOD_ARGUMENTS = {
    "read_candidate": {
        "job_id": "job",
        "ref": object(),
        "offset": 0,
        "limit": 1,
    },
    "apply_new": {
        "job_id": "job",
        "candidate_ref": object(),
        "expected_binding_revision": 1,
        "expected_space_revision": 1,
        "mutation_guard": object(),
        "operation_id": "operation",
    },
    "propose_review": {
        "job_id": "job",
        "candidate_ref": object(),
        "expected_binding_revision": 1,
        "target_entry_id": "entry",
        "expected_target_revision": 1,
        "mode": "overwrite",
        "mutation_guard": object(),
        "operation_id": "operation",
    },
}


def _validate_consolidation_capability(value: Any) -> None:
    for method_name, arguments in _CONSOLIDATION_METHOD_ARGUMENTS.items():
        method = getattr(value, method_name, None)
        if not callable(method):
            raise MemoryToolkitError(
                f"consolidation capability {method_name} must be callable"
            )
        try:
            inspect.signature(method).bind(**arguments)
        except (TypeError, ValueError) as exc:
            raise MemoryToolkitError(
                f"consolidation capability {method_name} has an incompatible signature"
            ) from exc


def _structured_refs(
    values: Sequence[ResourceRef],
    *,
    field_name: str,
    allowed_kinds: frozenset[str],
) -> tuple[ResourceRef, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must contain structured ResourceRef values")
    normalized: list[ResourceRef] = []
    for value in values:
        if not isinstance(value, ResourceRef):
            raise TypeError(f"{field_name} must contain structured ResourceRef values")
        if value.kind not in allowed_kinds:
            raise MemoryToolkitError(
                f"{field_name} contains a disallowed reference kind"
            )
        normalized.append(value)
    return tuple(normalized)


@dataclass(frozen=True)
class NormalMemoryToolkitCapabilities:
    references: BoundExternalReferenceCodec
    context: BoundContextMemoryCapability
    chat: BoundMemoryReadCapability
    candidates: BoundCandidateProposalCapability
    proposal_source_refs_resolver: Callable[[], Sequence[ResourceRef]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    long_term: BoundMemoryReadCapability | None = None
    allowed_long_term_refs: tuple[ResourceRef, ...] = ()

    def __post_init__(self) -> None:
        if self.proposal_source_refs_resolver is not None and not callable(
            self.proposal_source_refs_resolver
        ):
            raise TypeError("proposal_source_refs_resolver must be callable")
        allowed_long_term_refs = _structured_refs(
            self.allowed_long_term_refs,
            field_name="allowed_long_term_refs",
            allowed_kinds=frozenset({"memory"}),
        )
        if len(allowed_long_term_refs) > MAX_RECALLED_LONG_TERM_REFS:
            raise MemoryToolkitError(
                "recalled long-term reference set exceeds the bundle limit"
            )
        object.__setattr__(
            self,
            "allowed_long_term_refs",
            allowed_long_term_refs,
        )
        if self.allowed_long_term_refs and self.long_term is None:
            raise MemoryToolkitError(
                "recalled long-term references require a bound long-term capability"
            )

    def resolve_proposal_source_refs(self) -> tuple[ResourceRef, ...]:
        resolver = self.proposal_source_refs_resolver
        if resolver is None:
            raise MemoryToolkitError(
                "memory proposal provenance is unavailable for this run"
            )
        refs = _structured_refs(
            resolver(),
            field_name="proposal_source_refs",
            allowed_kinds=frozenset({"context_event"}),
        )
        if not refs:
            raise MemoryToolkitError(
                "memory proposal provenance must include current-run evidence"
            )
        if len(refs) > 128:
            raise MemoryToolkitError(
                "memory proposal provenance exceeds the run evidence limit"
            )
        if len(set(refs)) != len(refs):
            raise MemoryToolkitError(
                "memory proposal provenance must not contain duplicates"
            )
        return refs


@dataclass(frozen=True)
class CuratorMemoryToolkitCapabilities:
    references: BoundExternalReferenceCodec
    context: BoundContextMemoryCapability
    chat: BoundMemoryCuratorCapability
    task_state: BoundTaskStateCapability
    promotions: BoundPromotionProposalCapability | None = None
    long_term: BoundMemoryReadCapability | None = None


@dataclass(frozen=True)
class ConsolidationMemoryToolkitCapabilities:
    binding_id: str
    references: BoundExternalReferenceCodec
    context: BoundContextMemoryCapability
    chat: BoundMemoryReadCapability
    consolidation: BoundConsolidationCapability
    job_id: str
    candidate_refs: tuple[ResourceRef, ...]
    lease_fence: CuratorLeaseFence
    mutation_guard: BoundCuratorMutationGuard
    source_refs: tuple[ResourceRef, ...] = ()
    target_space_revision: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.binding_id, str) or not self.binding_id.strip():
            raise MemoryToolkitError("a consolidation run binding must be bound")
        if not isinstance(self.job_id, str) or not self.job_id.strip():
            raise MemoryToolkitError("a consolidation job must be bound")
        if not isinstance(self.lease_fence, CuratorLeaseFence):
            raise TypeError("lease_fence must be a CuratorLeaseFence")
        if not isinstance(self.mutation_guard, BoundCuratorMutationGuard):
            raise TypeError("mutation_guard must be a BoundCuratorMutationGuard")
        if self.lease_fence.binding_id != self.binding_id:
            raise MemoryToolkitError(
                "consolidation lease fence belongs to another run binding"
            )
        if self.lease_fence.job_id != self.job_id:
            raise MemoryToolkitError("consolidation lease fence belongs to another job")
        if self.mutation_guard.fence != self.lease_fence:
            raise MemoryToolkitError(
                "consolidation mutation guard does not match the lease fence"
            )
        target_space_revision = getattr(self.chat, "space_revision", None)
        if (
            isinstance(target_space_revision, bool)
            or not isinstance(target_space_revision, int)
            or target_space_revision < 1
        ):
            raise MemoryToolkitError(
                "consolidation target space revision must be host-bound"
            )
        object.__setattr__(
            self,
            "target_space_revision",
            target_space_revision,
        )
        candidate_refs = _structured_refs(
            self.candidate_refs,
            field_name="candidate_refs",
            allowed_kinds=frozenset({"memory_candidate"}),
        )
        source_refs = _structured_refs(
            self.source_refs,
            field_name="source_refs",
            allowed_kinds=frozenset({"context_event"}),
        )
        if len(set(candidate_refs)) != len(candidate_refs):
            raise MemoryToolkitError("candidate_refs must not contain duplicates")
        if len(set(source_refs)) != len(source_refs):
            raise MemoryToolkitError("source_refs must not contain duplicates")
        object.__setattr__(
            self,
            "candidate_refs",
            candidate_refs,
        )
        object.__setattr__(
            self,
            "source_refs",
            source_refs,
        )
        if not self.candidate_refs:
            raise MemoryToolkitError(
                "a consolidation job must bind at least one frozen candidate"
            )
        if len(self.candidate_refs) > 200:
            raise MemoryToolkitError(
                "consolidation candidate set exceeds the job limit"
            )
        if len(self.source_refs) > 20_000:
            raise MemoryToolkitError("consolidation source set exceeds the job limit")


@dataclass(frozen=True)
class TaskStateMemoryToolkitCapabilities:
    references: BoundExternalReferenceCodec
    context: BoundContextMemoryCapability
    chat: BoundMemoryReadCapability
    task_state: BoundTaskStateCapability
    long_term: BoundMemoryReadCapability | None = None
    allowed_long_term_refs: tuple[ResourceRef, ...] = ()

    def __post_init__(self) -> None:
        allowed_long_term_refs = _structured_refs(
            self.allowed_long_term_refs,
            field_name="allowed_long_term_refs",
            allowed_kinds=frozenset({"memory"}),
        )
        if len(allowed_long_term_refs) > MAX_RECALLED_LONG_TERM_REFS:
            raise MemoryToolkitError(
                "recalled long-term reference set exceeds the bundle limit"
            )
        object.__setattr__(
            self,
            "allowed_long_term_refs",
            allowed_long_term_refs,
        )
        if self.allowed_long_term_refs and self.long_term is None:
            raise MemoryToolkitError(
                "recalled long-term references require a bound long-term capability"
            )


MemoryToolkitCapabilities = (
    NormalMemoryToolkitCapabilities
    | CuratorMemoryToolkitCapabilities
    | ConsolidationMemoryToolkitCapabilities
    | TaskStateMemoryToolkitCapabilities
)


def validate_capability_bindings(
    binding_id: str,
    capabilities: MemoryToolkitCapabilities,
) -> None:
    if (
        isinstance(capabilities, ConsolidationMemoryToolkitCapabilities)
        and capabilities.binding_id != binding_id
    ):
        raise MemoryToolkitError(
            "consolidation bundle belongs to another run binding"
        )
    values: list[tuple[str, Any]] = [
        ("references", capabilities.references),
        ("context", capabilities.context),
        ("chat", capabilities.chat),
    ]
    for field_name in (
        "candidates",
        "consolidation",
        "task_state",
        "promotions",
        "long_term",
    ):
        value = getattr(capabilities, field_name, None)
        if value is not None:
            values.append((field_name, value))
    for label, value in values:
        if str(getattr(value, "binding_id", "")) != binding_id:
            raise MemoryToolkitError(
                f"{label} capability belongs to another run binding"
            )
    if isinstance(capabilities, ConsolidationMemoryToolkitCapabilities):
        _validate_consolidation_capability(capabilities.consolidation)
    long_term = getattr(capabilities, "long_term", None)
    if long_term is not None and long_term.space_id == capabilities.chat.space_id:
        raise MemoryToolkitError(
            "chat and long-term capabilities must use distinct spaces"
        )


__all__ = [
    "BoundCandidateProposalCapability",
    "BoundConsolidationCapability",
    "BoundContextMemoryCapability",
    "BoundExternalReferenceCodec",
    "BoundMemoryCuratorCapability",
    "BoundMemoryReadCapability",
    "BoundPromotionProposalCapability",
    "BoundTaskStateCapability",
    "ConsolidationMemoryToolkitCapabilities",
    "CuratorMemoryToolkitCapabilities",
    "MemoryToolkitCapabilities",
    "NormalMemoryToolkitCapabilities",
    "TaskStateMemoryToolkitCapabilities",
    "validate_capability_bindings",
]
