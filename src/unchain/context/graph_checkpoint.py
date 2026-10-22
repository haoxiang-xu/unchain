from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from unchain.journal import (
    ArtifactRef,
    AttemptRef,
    BoundExecutionJournal,
    EventCursor,
    EventRange,
    JournalAppendResult,
    JournalEvent,
    JournalSnapshot,
    ResourceRef,
    SemanticEventDraft,
)
from unchain.journal.models import (
    _freeze_json,
    _record_tuple,
    _required_text,
    _sha256,
    _thaw_json,
)
from unchain.journal.interaction_resolution_compat import (
    InteractionResolutionCompatibilityError,
    interaction_resolution_compatibility_record,
    legacy_interaction_resolution_supersession_pairs,
)
from unchain.journal.graph_attempt_quiescence import (
    CANCELLED_ATTEMPT_TERMINALS,
    COMPLETED_ATTEMPT_TERMINALS,
    FAILED_ATTEMPT_TERMINALS,
    MAX_ITERATIONS_TERMINALS,
    select_attempt_terminal,
)
from unchain.journal.interaction_cycles import (
    INTERACTION_LIVENESS_EXEMPT_EVENT_TYPES,
    INTERACTION_REQUESTS,
    INTERACTION_RESOLUTIONS,
    LIVE_INTERACTION_OUTCOMES,
)

from .artifacts import ArtifactService, MAX_ARTIFACT_BYTES
from .derived_handoff import (
    DerivedHandoffInputIngress,
    DurableDerivedHandoffInputReceipt,
    HostResolvedDerivedHandoffInput,
)
from .models import HandoffStatus


_COMPLETED_TERMINALS = COMPLETED_ATTEMPT_TERMINALS
_FAILED_TERMINALS = FAILED_ATTEMPT_TERMINALS | MAX_ITERATIONS_TERMINALS
_CANCELLED_TERMINALS = CANCELLED_ATTEMPT_TERMINALS
_INTERACTION_REQUESTS = INTERACTION_REQUESTS
_INTERACTION_RESOLUTIONS = INTERACTION_RESOLUTIONS
# Events a host may append to a step attempt while the run is genuinely
# suspended.  Anything outside this set is run-execution activity: seeing it
# after an unresolved request proves the attempt never exited, i.e. the
# prompt was answered through the in-run callback and its receipt lives in
# the session-state interaction journal rather than this events journal.
_LIVENESS_EXEMPT_EVENT_TYPES = INTERACTION_LIVENESS_EXEMPT_EVENT_TYPES
# Orchestration-attempt terminals that end a graph that did not complete.
_ROOT_DEAD_TERMINALS = FAILED_ATTEMPT_TERMINALS | CANCELLED_ATTEMPT_TERMINALS


class GraphCheckpointError(RuntimeError):
    """A graph execution cannot safely advance from its durable journal."""


class GraphCheckpointConflict(GraphCheckpointError):
    """A replay changed graph topology, identity, or immutable output."""


class GraphStepDisposition(StrEnum):
    STARTED = "started"
    SKIP_COMPLETED = "skip_completed"


class GraphTerminalStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _thaw_json(_freeze_json(value, path="graph_checkpoint")),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    return f"{prefix}-{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _cursor(event: JournalEvent) -> EventCursor:
    return EventCursor(event.store_seq, event.event_id)


def _same_cursor(left: EventCursor, right: EventCursor) -> bool:
    return (
        left.store_seq == right.store_seq
        and left.event_id == right.event_id
    )


def _interaction_id(event: JournalEvent) -> str:
    raw_interaction_id = event.payload.get("interaction_id")
    if raw_interaction_id is None:
        request = event.payload.get("interaction_request")
        if isinstance(request, Mapping):
            raw_interaction_id = request.get("interaction_id")
    if raw_interaction_id is None:
        for field_name in ("confirmation_id", "request_id", "call_id"):
            candidate = event.payload.get(field_name)
            if candidate is not None:
                raw_interaction_id = candidate
                break
    try:
        return _required_text(
            raw_interaction_id,
            "interaction_id",
            identifier=True,
        )
    except (TypeError, ValueError) as error:
        raise GraphCheckpointError(
            "graph interaction identity is missing or invalid"
        ) from error


def _graph_interaction_resolution_suppressions(
    events: Sequence[JournalEvent],
) -> frozenset[int]:
    records = tuple(
        interaction_resolution_compatibility_record(
            ordinal=event.store_seq,
            event_type=event.event_type,
            interaction_id=_interaction_id(event),
            execution_id=event.attempt.generation.execution_id,
            generation_id=event.attempt.generation.generation_id,
            attempt_id=event.attempt.attempt_id,
            payload=event.payload,
            resource_refs=event.resource_refs,
        )
        for event in events
        if event.event_type in {"interaction_resolved", "interaction.resolved"}
    )
    try:
        pairs = legacy_interaction_resolution_supersession_pairs(records)
    except InteractionResolutionCompatibilityError as error:
        raise GraphCheckpointError(
            "graph interaction resolution is ambiguous"
        ) from error

    admitted_resolution_store_seqs: set[int] = set()
    for event in events:
        if event.event_type != "graph.step.resume.admitted":
            continue
        raw_cursor = event.payload.get("resolution_cursor")
        try:
            admitted_resolution_store_seqs.add(
                EventCursor.from_dict(raw_cursor).store_seq
            )
        except (AttributeError, TypeError, ValueError):
            continue
    return frozenset(
        (
            pair.canonical_ordinal
            if pair.legacy_ordinal in admitted_resolution_store_seqs
            else pair.legacy_ordinal
        )
        for pair in pairs
    )


def _interaction_aliases(event: JournalEvent) -> frozenset[str]:
    candidates: list[Any] = []

    def collect(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        for field_name in (
            "interaction_id",
            "confirmation_id",
            "request_id",
            "call_id",
            "tool_call_id",
        ):
            candidates.append(value.get(field_name))

    collect(event.payload)
    request = event.payload.get("interaction_request")
    collect(request)
    if isinstance(request, Mapping):
        collect(request.get("payload"))
        collect(request.get("subject"))
    aliases: set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            aliases.add(
                _required_text(
                    candidate,
                    "interaction_alias",
                    identifier=True,
                )
            )
        except (TypeError, ValueError):
            continue
    aliases.add(_interaction_id(event))
    return frozenset(aliases)


@dataclass(frozen=True)
class GraphStepBinding:
    index: int
    node_id: str
    attempt: AttemptRef
    source_attempt: AttemptRef
    provider: str
    model: str
    configuration_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValueError("graph step index must be a non-negative integer")
        object.__setattr__(
            self,
            "node_id",
            _required_text(self.node_id, "node_id", identifier=True),
        )
        if not isinstance(self.attempt, AttemptRef):
            object.__setattr__(self, "attempt", AttemptRef.from_dict(self.attempt))
        if not isinstance(self.source_attempt, AttemptRef):
            object.__setattr__(
                self,
                "source_attempt",
                AttemptRef.from_dict(self.source_attempt),
            )
        if self.attempt == self.source_attempt:
            raise ValueError("graph step source and consumer attempts must differ")
        if self.attempt.generation != self.source_attempt.generation:
            raise ValueError("graph step attempts must share one generation")
        object.__setattr__(
            self,
            "provider",
            _required_text(self.provider, "provider", maximum=128).casefold(),
        )
        object.__setattr__(
            self,
            "model",
            _required_text(self.model, "model", maximum=512),
        )
        object.__setattr__(
            self,
            "configuration_sha256",
            _sha256(self.configuration_sha256, "configuration_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "node_id": self.node_id,
            "attempt": self.attempt.to_dict(),
            "source_attempt": self.source_attempt.to_dict(),
            "provider": self.provider,
            "model": self.model,
            "configuration_sha256": self.configuration_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphStepBinding":
        return cls(
            index=value.get("index"),
            node_id=value.get("node_id"),
            attempt=AttemptRef.from_dict(value.get("attempt")),
            source_attempt=AttemptRef.from_dict(value.get("source_attempt")),
            provider=value.get("provider"),
            model=value.get("model"),
            configuration_sha256=value.get("configuration_sha256"),
        )


@dataclass(frozen=True)
class GraphExecutionPlan:
    orchestration_attempt: AttemptRef
    topology_sha256: str
    initial_input_cursor: EventCursor
    steps: tuple[GraphStepBinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.orchestration_attempt, AttemptRef):
            object.__setattr__(
                self,
                "orchestration_attempt",
                AttemptRef.from_dict(self.orchestration_attempt),
            )
        object.__setattr__(
            self,
            "topology_sha256",
            _sha256(self.topology_sha256, "topology_sha256"),
        )
        if not isinstance(self.initial_input_cursor, EventCursor):
            object.__setattr__(
                self,
                "initial_input_cursor",
                EventCursor.from_dict(self.initial_input_cursor),
            )
        steps = _record_tuple(self.steps, GraphStepBinding, "steps")
        if not steps:
            raise ValueError("graph plan requires at least one step")
        generation = self.orchestration_attempt.generation
        seen_attempts = {self.orchestration_attempt.attempt_id}
        seen_nodes: set[str] = set()
        predecessor = self.orchestration_attempt
        for expected_index, step in enumerate(steps):
            if step.index != expected_index:
                raise ValueError("graph step indexes must be contiguous")
            if step.attempt.generation != generation:
                raise ValueError("graph plan attempts must share one generation")
            if step.source_attempt != predecessor:
                raise ValueError("graph step source must be its immediate predecessor")
            if step.attempt.attempt_id in seen_attempts:
                raise ValueError("graph plan attempt IDs must be distinct")
            if step.node_id in seen_nodes:
                raise ValueError("graph plan node IDs must be distinct")
            seen_attempts.add(step.attempt.attempt_id)
            seen_nodes.add(step.node_id)
            predecessor = step.attempt
        object.__setattr__(self, "steps", steps)

    @property
    def execution_id(self) -> str:
        return self.orchestration_attempt.generation.execution_id

    @property
    def plan_id(self) -> str:
        return _stable_id("graph-plan", self.to_dict())

    @property
    def scope_id(self) -> str:
        return _stable_id(
            "graph-scope",
            {"orchestration_attempt": self.orchestration_attempt.to_dict()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "unchain.graph_execution_plan.v1",
            "orchestration_attempt": self.orchestration_attempt.to_dict(),
            "topology_sha256": self.topology_sha256,
            "initial_input_cursor": self.initial_input_cursor.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphExecutionPlan":
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "orchestration_attempt",
            "topology_sha256",
            "initial_input_cursor",
            "steps",
        }:
            raise ValueError("graph execution plan descriptor is not closed")
        if value.get("schema") != "unchain.graph_execution_plan.v1":
            raise ValueError("graph execution plan schema is unsupported")
        raw_steps = value.get("steps")
        if isinstance(raw_steps, (str, bytes, bytearray)) or not isinstance(
            raw_steps,
            Sequence,
        ):
            raise ValueError("graph execution plan steps must be a sequence")
        plan = cls(
            orchestration_attempt=AttemptRef.from_dict(
                value.get("orchestration_attempt")
            ),
            topology_sha256=value.get("topology_sha256"),
            initial_input_cursor=EventCursor.from_dict(
                value.get("initial_input_cursor")
            ),
            steps=tuple(GraphStepBinding.from_dict(step) for step in raw_steps),
        )
        if plan.to_dict() != _thaw_json(value):
            raise ValueError("graph execution plan descriptor is not canonical")
        return plan


@dataclass(frozen=True)
class GraphStepCompletion:
    step: GraphStepBinding
    output_artifact: ArtifactRef
    source_event_range: EventRange
    terminal_cursor: EventCursor
    checkpoint_cursor: EventCursor

    def __post_init__(self) -> None:
        if not isinstance(self.step, GraphStepBinding):
            object.__setattr__(self, "step", GraphStepBinding.from_dict(self.step))
        if not isinstance(self.output_artifact, ArtifactRef):
            object.__setattr__(
                self,
                "output_artifact",
                ArtifactRef.from_dict(self.output_artifact),
            )
        for field_name in ("source_event_range", "terminal_cursor", "checkpoint_cursor"):
            value = getattr(self, field_name)
            expected = EventRange if field_name == "source_event_range" else EventCursor
            if not isinstance(value, expected):
                object.__setattr__(
                    self,
                    field_name,
                    expected.from_dict(value),
                )
        if (
            self.source_event_range.start != self.checkpoint_cursor
            or self.source_event_range.end != self.checkpoint_cursor
        ):
            raise ValueError(
                "graph handoff source range must be its checkpoint event"
            )
        if self.terminal_cursor.store_seq >= self.checkpoint_cursor.store_seq:
            raise ValueError("graph completion checkpoint must follow its terminal")


@dataclass(frozen=True)
class GraphRecovery:
    plan: GraphExecutionPlan
    completed_steps: tuple[GraphStepCompletion, ...]
    next_step_index: int | None
    last_cursor: EventCursor
    uncertain_step_index: int | None = None
    suspended_step_index: int | None = None
    resume_ready_step_index: int | None = None
    resuming_step_index: int | None = None
    terminal_status: GraphTerminalStatus | None = None

    @property
    def is_complete(self) -> bool:
        return (
            self.terminal_status is GraphTerminalStatus.COMPLETED
            and len(self.completed_steps) == len(self.plan.steps)
        )


@dataclass(frozen=True)
class GraphStepStartReceipt:
    disposition: GraphStepDisposition
    step: GraphStepBinding
    started_cursor: EventCursor | None = None
    handoff: DurableDerivedHandoffInputReceipt | None = None
    completion: GraphStepCompletion | None = None


@dataclass(frozen=True)
class GraphStepResumeEvidence:
    graph_plan_id: str
    graph_scope_id: str
    step: GraphStepBinding
    interaction_id: str
    request_cursor: EventCursor
    resolution_cursor: EventCursor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "graph_plan_id",
            _required_text(self.graph_plan_id, "graph_plan_id", identifier=True),
        )
        object.__setattr__(
            self,
            "graph_scope_id",
            _required_text(self.graph_scope_id, "graph_scope_id", identifier=True),
        )
        if not isinstance(self.step, GraphStepBinding):
            object.__setattr__(
                self,
                "step",
                GraphStepBinding.from_dict(self.step),
            )
        object.__setattr__(
            self,
            "interaction_id",
            _required_text(self.interaction_id, "interaction_id", identifier=True),
        )
        for field_name in ("request_cursor", "resolution_cursor"):
            value = getattr(self, field_name)
            if not isinstance(value, EventCursor):
                object.__setattr__(
                    self,
                    field_name,
                    EventCursor.from_dict(value),
                )
        if self.request_cursor.store_seq >= self.resolution_cursor.store_seq:
            raise ValueError("interaction resolution must follow its request")


@dataclass(frozen=True)
class GraphStepResumeReceipt:
    graph_plan_id: str
    graph_scope_id: str
    step: GraphStepBinding
    interaction_id: str
    request_cursor: EventCursor
    resolution_cursor: EventCursor
    admitted_cursor: EventCursor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "graph_plan_id",
            _required_text(self.graph_plan_id, "graph_plan_id", identifier=True),
        )
        object.__setattr__(
            self,
            "graph_scope_id",
            _required_text(self.graph_scope_id, "graph_scope_id", identifier=True),
        )
        if not isinstance(self.step, GraphStepBinding):
            object.__setattr__(
                self,
                "step",
                GraphStepBinding.from_dict(self.step),
            )
        object.__setattr__(
            self,
            "interaction_id",
            _required_text(self.interaction_id, "interaction_id", identifier=True),
        )
        for field_name in (
            "request_cursor",
            "resolution_cursor",
            "admitted_cursor",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, EventCursor):
                object.__setattr__(
                    self,
                    field_name,
                    EventCursor.from_dict(value),
                )
        if not (
            self.request_cursor.store_seq
            < self.resolution_cursor.store_seq
            < self.admitted_cursor.store_seq
        ):
            raise ValueError(
                "graph resume must follow its interaction request and resolution"
            )


@dataclass(frozen=True)
class _GraphInteractionCycle:
    interaction_id: str
    request_cursor: EventCursor
    aliases: frozenset[str]
    resolution_cursor: EventCursor | None = None
    resume_receipt: GraphStepResumeReceipt | None = None


@dataclass(frozen=True)
class _GraphScan:
    recovery: GraphRecovery
    snapshot_events: tuple[JournalEvent, ...]
    started_events: Mapping[int, JournalEvent]
    active_interactions: Mapping[int, _GraphInteractionCycle]
    resume_receipts: Mapping[tuple[int, str], GraphStepResumeReceipt]


def _validated_graph_plan_admission(
    events: Sequence[JournalEvent],
    *,
    execution_id: str,
    generation_id: str,
    orchestration_attempt_id: str,
) -> tuple[JournalEvent, GraphExecutionPlan]:
    admissions = tuple(
        event
        for event in events
        if event.event_type == "graph.execution.admitted"
        and event.attempt.generation.execution_id == execution_id
        and event.attempt.generation.generation_id == generation_id
        and event.attempt.attempt_id == orchestration_attempt_id
    )
    if len(admissions) != 1:
        raise GraphCheckpointError("graph plan descriptor is missing or ambiguous")
    admission = admissions[0]
    if set(admission.payload) != {
        "graph_plan_id",
        "graph_scope_id",
        "plan",
    }:
        raise GraphCheckpointError("graph plan descriptor is not closed")
    try:
        plan = GraphExecutionPlan.from_dict(
            _thaw_json(admission.payload.get("plan"))
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise GraphCheckpointError("graph plan descriptor is invalid") from error
    if (
        plan.execution_id != execution_id
        or plan.orchestration_attempt != admission.attempt
        or plan.orchestration_attempt.generation.generation_id != generation_id
        or admission.payload.get("graph_plan_id") != plan.plan_id
        or admission.payload.get("graph_scope_id") != plan.scope_id
    ):
        raise GraphCheckpointError("graph plan descriptor changed identity")
    return admission, plan


def locate_graph_execution_plan(
    journal: BoundExecutionJournal,
    *,
    generation_id: str,
    orchestration_attempt_id: str,
    snapshot: JournalSnapshot | None = None,
) -> GraphExecutionPlan:
    """Rebuild one exact admitted plan without appending or persisting data."""

    if not isinstance(journal, BoundExecutionJournal):
        raise TypeError("journal must be a BoundExecutionJournal")
    generation_id = _required_text(
        generation_id,
        "generation_id",
        identifier=True,
    )
    orchestration_attempt_id = _required_text(
        orchestration_attempt_id,
        "orchestration_attempt_id",
        identifier=True,
    )
    snapshot = snapshot or journal.capture_snapshot()
    if snapshot.execution_id != journal.execution_id:
        raise GraphCheckpointError("graph journal snapshot changed execution")
    _admission, plan = _validated_graph_plan_admission(
        snapshot.events,
        execution_id=journal.execution_id,
        generation_id=generation_id,
        orchestration_attempt_id=orchestration_attempt_id,
    )
    return plan


@dataclass(frozen=True)
class GraphInteractionLineageProof:
    """Closed, read-only evidence for a currently suspended graph interaction.

    The proof binds an immutable admitted plan, its completed prefix, and one
    still-unresolved interaction in the same journal snapshot.  Consumers can
    therefore authorize a transport recovery without consulting transient
    step-resume files or reconstructing graph topology themselves.
    """

    execution_id: str
    generation_id: str
    coordinator_attempt_id: str
    graph_plan_id: str
    graph_scope_id: str
    topology_sha256: str
    source_step: GraphStepBinding
    current_step: GraphStepBinding
    interaction_id: str
    request_cursor: EventCursor
    journal_high_water: EventCursor
    completed_step_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "execution_id",
            "generation_id",
            "coordinator_attempt_id",
            "graph_plan_id",
            "graph_scope_id",
            "interaction_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(
                    getattr(self, field_name),
                    field_name,
                    identifier=True,
                ),
            )
        object.__setattr__(
            self,
            "topology_sha256",
            _sha256(self.topology_sha256, "topology_sha256"),
        )
        for field_name in ("source_step", "current_step"):
            value = getattr(self, field_name)
            if not isinstance(value, GraphStepBinding):
                object.__setattr__(
                    self,
                    field_name,
                    GraphStepBinding.from_dict(value),
                )
        for field_name in ("request_cursor", "journal_high_water"):
            value = getattr(self, field_name)
            if not isinstance(value, EventCursor):
                object.__setattr__(
                    self,
                    field_name,
                    EventCursor.from_dict(value),
                )
        indexes = tuple(self.completed_step_indexes)
        if any(type(index) is not int or index < 0 for index in indexes):
            raise ValueError("completed_step_indexes must contain non-negative integers")
        if indexes != tuple(range(self.current_step.index)):
            raise ValueError("completed step indexes must be the current-step prefix")
        object.__setattr__(self, "completed_step_indexes", indexes)
        if self.source_step.index > self.current_step.index:
            raise ValueError("source graph step cannot follow the current step")
        if (
            self.source_step.attempt.generation.execution_id != self.execution_id
            or self.current_step.attempt.generation.execution_id != self.execution_id
            or self.source_step.attempt.generation.generation_id != self.generation_id
            or self.current_step.attempt.generation.generation_id != self.generation_id
        ):
            raise ValueError("graph interaction proof escaped its execution generation")


def prove_graph_interaction_lineage(
    journal: BoundExecutionJournal,
    *,
    generation_id: str,
    coordinator_attempt_id: str,
    source_attempt_id: str,
    current_attempt_id: str,
    interaction_id: str,
    allow_resolved: bool = False,
    snapshot: JournalSnapshot | None = None,
) -> GraphInteractionLineageProof:
    """Return canonical evidence for one pending graph interaction.

    This function has no append or recovery side effect.  It rejects a source
    or current attempt outside the unique admitted plan, a non-contiguous
    completed prefix, resuming interactions, terminal graphs, and
    planned-but-not-executing future steps.

    ``allow_resolved`` is reserved for reconciliation after canonical ingress
    committed but before the host receipt was persisted.  It accepts either
    still-unresolved state or that exact resolved-and-not-resumed state; the
    caller must still replay through canonical ingress to reject a competing
    response before mutating host state.

    ``snapshot`` lets a caller that already holds an in-transaction
    :class:`JournalSnapshot` (for example an
    ``append_with_artifacts`` precondition) reuse it instead of capturing a
    second, potentially different one. It must belong to ``journal``'s exact
    execution; a snapshot from a different execution is rejected rather than
    silently mixed with this call's identity.
    """

    if not isinstance(journal, BoundExecutionJournal):
        raise TypeError("journal must be a BoundExecutionJournal")
    generation_id = _required_text(generation_id, "generation_id", identifier=True)
    coordinator_attempt_id = _required_text(
        coordinator_attempt_id,
        "coordinator_attempt_id",
        identifier=True,
    )
    source_attempt_id = _required_text(
        source_attempt_id,
        "source_attempt_id",
        identifier=True,
    )
    current_attempt_id = _required_text(
        current_attempt_id,
        "current_attempt_id",
        identifier=True,
    )
    interaction_id = _required_text(
        interaction_id,
        "interaction_id",
        identifier=True,
    )
    if snapshot is None:
        snapshot = journal.capture_snapshot()
    elif not isinstance(snapshot, JournalSnapshot):
        raise TypeError("snapshot must be a JournalSnapshot")
    elif snapshot.execution_id != journal.execution_id:
        raise GraphCheckpointError("graph journal snapshot changed execution")
    plan = locate_graph_execution_plan(
        journal,
        generation_id=generation_id,
        orchestration_attempt_id=coordinator_attempt_id,
        snapshot=snapshot,
    )
    source_steps = tuple(
        step for step in plan.steps if step.attempt.attempt_id == source_attempt_id
    )
    current_steps = tuple(
        step for step in plan.steps if step.attempt.attempt_id == current_attempt_id
    )
    if len(source_steps) != 1 or len(current_steps) != 1:
        raise GraphCheckpointError(
            "graph interaction lineage does not name one admitted graph step"
        )
    source_step = source_steps[0]
    current_step = current_steps[0]
    if source_step.index > current_step.index:
        raise GraphCheckpointError("graph interaction lineage moves backwards")

    scan = JournalGraphCheckpointRepository(journal).scan(plan, snapshot=snapshot)
    recovery = scan.recovery
    if recovery.terminal_status is not None:
        raise GraphCheckpointError("graph interaction lineage is terminal")
    active = scan.active_interactions.get(current_step.index)
    if (
        active is None
        or active.interaction_id != interaction_id
        or active.resume_receipt is not None
        or (not allow_resolved and active.resolution_cursor is not None)
    ):
        raise GraphCheckpointError(
            "graph interaction lineage has no exact unresolved interaction"
        )
    expected_step_index = (
        recovery.resume_ready_step_index
        if active.resolution_cursor is not None
        else recovery.suspended_step_index
    )
    if expected_step_index != current_step.index:
        raise GraphCheckpointError(
            "graph interaction lineage is not the expected pending step"
        )
    if len(recovery.completed_steps) != current_step.index:
        raise GraphCheckpointError(
            "graph interaction lineage completed prefix is not current"
        )
    return GraphInteractionLineageProof(
        execution_id=plan.execution_id,
        generation_id=plan.orchestration_attempt.generation.generation_id,
        coordinator_attempt_id=plan.orchestration_attempt.attempt_id,
        graph_plan_id=plan.plan_id,
        graph_scope_id=plan.scope_id,
        topology_sha256=plan.topology_sha256,
        source_step=source_step,
        current_step=current_step,
        interaction_id=interaction_id,
        request_cursor=active.request_cursor,
        journal_high_water=recovery.last_cursor,
        completed_step_indexes=tuple(
            completion.step.index for completion in recovery.completed_steps
        ),
    )


class JournalGraphCheckpointRepository:
    """Store graph control-plane checkpoints in the canonical execution journal."""

    def __init__(self, journal: BoundExecutionJournal) -> None:
        if not isinstance(journal, BoundExecutionJournal):
            raise TypeError("journal must be a BoundExecutionJournal")
        self.journal = journal

    def _append(
        self,
        *,
        plan: GraphExecutionPlan,
        attempt: AttemptRef,
        event_type: str,
        discriminator: str,
        payload: Mapping[str, Any],
        resource_refs: Sequence[ResourceRef] = (),
    ) -> JournalAppendResult:
        identity = {
            "scope_id": plan.scope_id,
            "event_type": event_type,
            "discriminator": discriminator,
        }
        draft = SemanticEventDraft(
            event_id=_stable_id("event", identity),
            event_type=event_type,
            attempt=attempt,
            operation_id=_stable_id("operation", identity),
            payload={
                "graph_plan_id": plan.plan_id,
                "graph_scope_id": plan.scope_id,
                **dict(payload),
            },
            resource_refs=tuple(resource_refs),
        )
        try:
            result = self.journal.append(request=draft.to_append_request())
        except Exception as error:
            raise GraphCheckpointConflict(
                f"graph checkpoint append conflicted: {event_type}"
            ) from error
        if result.event.attempt != attempt or result.event.event_type != event_type:
            raise GraphCheckpointError("graph checkpoint append changed its identity")
        return result

    def admit(self, plan: GraphExecutionPlan) -> JournalAppendResult:
        return self._append(
            plan=plan,
            attempt=plan.orchestration_attempt,
            event_type="graph.execution.admitted",
            discriminator="plan",
            payload={"plan": plan.to_dict()},
        )

    def start(
        self,
        plan: GraphExecutionPlan,
        step: GraphStepBinding,
        handoff: DurableDerivedHandoffInputReceipt,
    ) -> JournalAppendResult:
        return self._append(
            plan=plan,
            attempt=step.attempt,
            event_type="graph.step.started",
            discriminator=f"step-{step.index}",
            payload={
                "step": step.to_dict(),
                "handoff_cursor": handoff.handoff_cursor.to_dict(),
                "input_cursor": handoff.input_cursor.to_dict(),
            },
            resource_refs=(handoff.envelope.full_output_ref,),
        )

    def resume(
        self,
        plan: GraphExecutionPlan,
        step: GraphStepBinding,
        *,
        interaction_id: str,
        request_cursor: EventCursor,
        resolution_cursor: EventCursor,
    ) -> JournalAppendResult:
        return self._append(
            plan=plan,
            attempt=step.attempt,
            event_type="graph.step.resume.admitted",
            discriminator=(
                f"step-{step.index}-interaction-{interaction_id}"
            ),
            payload={
                "step": step.to_dict(),
                "interaction_id": interaction_id,
                "request_cursor": request_cursor.to_dict(),
                "resolution_cursor": resolution_cursor.to_dict(),
            },
        )

    def complete(
        self,
        plan: GraphExecutionPlan,
        step: GraphStepBinding,
        *,
        output_artifact: ArtifactRef,
        execution_event_range: EventRange,
        terminal_cursor: EventCursor,
    ) -> JournalAppendResult:
        return self._append(
            plan=plan,
            attempt=step.attempt,
            event_type="graph.step.completed",
            discriminator=f"step-{step.index}",
            payload={
                "step": step.to_dict(),
                "output_artifact": output_artifact.to_dict(),
                "execution_event_range": execution_event_range.to_dict(),
                "terminal_cursor": terminal_cursor.to_dict(),
            },
            resource_refs=(output_artifact.ref,),
        )

    def terminal(
        self,
        plan: GraphExecutionPlan,
        step: GraphStepBinding,
        *,
        status: GraphTerminalStatus,
        terminal_cursor: EventCursor,
    ) -> JournalAppendResult:
        if status is GraphTerminalStatus.COMPLETED:
            raise ValueError("completed graph steps require an output checkpoint")
        return self._append(
            plan=plan,
            attempt=step.attempt,
            event_type=f"graph.step.{status.value}",
            discriminator=f"step-{step.index}",
            payload={
                "step": step.to_dict(),
                "terminal_cursor": terminal_cursor.to_dict(),
            },
        )

    def finalize(
        self,
        plan: GraphExecutionPlan,
        completion: GraphStepCompletion,
    ) -> JournalAppendResult:
        return self._append(
            plan=plan,
            attempt=plan.orchestration_attempt,
            event_type="graph.execution.completed",
            discriminator="terminal",
            payload={
                "status": "completed",
                "final_step_index": completion.step.index,
                "output_artifact": completion.output_artifact.to_dict(),
                "source_event_range": completion.source_event_range.to_dict(),
            },
            resource_refs=(completion.output_artifact.ref,),
        )

    def scan(
        self,
        plan: GraphExecutionPlan,
        *,
        snapshot: JournalSnapshot | None = None,
    ) -> _GraphScan:
        snapshot = snapshot or self.journal.capture_snapshot()
        if snapshot.execution_id != plan.execution_id or snapshot.high_water is None:
            raise GraphCheckpointError("graph journal snapshot is unavailable")
        events = tuple(snapshot.events)
        suppressed_interaction_resolutions = (
            _graph_interaction_resolution_suppressions(events)
        )
        _admission, persisted_plan = _validated_graph_plan_admission(
            events,
            execution_id=plan.execution_id,
            generation_id=plan.orchestration_attempt.generation.generation_id,
            orchestration_attempt_id=plan.orchestration_attempt.attempt_id,
        )
        if persisted_plan != plan:
            raise GraphCheckpointConflict("graph plan changed after admission")

        starts: dict[int, JournalEvent] = {}
        completions: dict[int, GraphStepCompletion] = {}
        step_terminals: dict[int, GraphTerminalStatus] = {}
        resume_events: list[tuple[GraphStepBinding, JournalEvent]] = []
        graph_terminal: GraphTerminalStatus | None = None
        for event in events:
            if (
                event.attempt == plan.orchestration_attempt
                and event.event_type in _ROOT_DEAD_TERMINALS
            ):
                # The orchestration attempt ended failed or cancelled. Its
                # events carry no graph scope, so they are matched here by
                # attempt. After a completed graph this is the root's own
                # later outcome and changes nothing; before it, the graph is
                # terminal and a later completion would contradict it.
                if graph_terminal is GraphTerminalStatus.COMPLETED:
                    continue
                if graph_terminal is not None:
                    raise GraphCheckpointError("graph terminal is ambiguous")
                graph_terminal = (
                    GraphTerminalStatus.FAILED
                    if event.event_type in FAILED_ATTEMPT_TERMINALS
                    else GraphTerminalStatus.CANCELLED
                )
                continue
            if event.payload.get("graph_scope_id") != plan.scope_id:
                continue
            if event.payload.get("graph_plan_id") != plan.plan_id:
                raise GraphCheckpointConflict("graph checkpoint plan identity changed")
            if event.event_type == "graph.execution.completed":
                if graph_terminal is not None:
                    raise GraphCheckpointError("graph terminal is ambiguous")
                graph_terminal = GraphTerminalStatus.COMPLETED
                continue
            if not event.event_type.startswith("graph.step."):
                continue
            raw_step = event.payload.get("step")
            try:
                step = GraphStepBinding.from_dict(raw_step)
            except (AttributeError, TypeError, ValueError) as error:
                raise GraphCheckpointError("graph step checkpoint is corrupt") from error
            if step.index >= len(plan.steps) or step != plan.steps[step.index]:
                raise GraphCheckpointConflict("graph step binding changed")
            if event.attempt != step.attempt:
                raise GraphCheckpointError("graph checkpoint escaped its step attempt")
            if event.event_type == "graph.step.started":
                if step.index in starts:
                    raise GraphCheckpointError("graph step start is ambiguous")
                starts[step.index] = event
                continue
            if event.event_type == "graph.step.resume.admitted":
                if event.resource_refs:
                    raise GraphCheckpointError(
                        "graph resume admission cannot carry resource references"
                    )
                resume_events.append((step, event))
                continue
            if event.event_type == "graph.step.completed":
                if step.index in completions or step.index in step_terminals:
                    raise GraphCheckpointError("graph step terminal is ambiguous")
                try:
                    artifact = ArtifactRef.from_dict(event.payload["output_artifact"])
                    terminal_cursor = EventCursor.from_dict(
                        event.payload["terminal_cursor"]
                    )
                    execution_range = EventRange.from_dict(
                        event.payload["execution_event_range"]
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise GraphCheckpointError(
                        "graph completion descriptor is corrupt"
                    ) from error
                checkpoint_cursor = _cursor(event)
                if event.resource_refs != (artifact.ref,):
                    raise GraphCheckpointError(
                        "graph completion artifact reference changed"
                    )
                if execution_range.end != terminal_cursor:
                    raise GraphCheckpointError(
                        "graph execution range does not end at its terminal"
                    )
                completions[step.index] = GraphStepCompletion(
                    step=step,
                    output_artifact=artifact,
                    source_event_range=EventRange(
                        checkpoint_cursor,
                        checkpoint_cursor,
                    ),
                    terminal_cursor=terminal_cursor,
                    checkpoint_cursor=checkpoint_cursor,
                )
                continue
            terminal_name = event.event_type.removeprefix("graph.step.")
            if terminal_name in {"failed", "cancelled"}:
                if step.index in completions or step.index in step_terminals:
                    raise GraphCheckpointError("graph step terminal is ambiguous")
                step_terminals[step.index] = GraphTerminalStatus(terminal_name)

        completed_prefix: list[GraphStepCompletion] = []
        for index in range(len(plan.steps)):
            completion = completions.get(index)
            if completion is None:
                break
            if index not in starts:
                raise GraphCheckpointError("completed graph step has no durable start")
            completed_prefix.append(completion)
        if set(completions) != set(range(len(completed_prefix))):
            raise GraphCheckpointError("graph completion checkpoints are not a prefix")
        resume_event_keys = {
            (step.index, event.event_id)
            for step, event in resume_events
        }
        processed_resume_event_keys: set[tuple[int, str]] = set()
        active_interactions: dict[int, _GraphInteractionCycle] = {}
        resume_receipts: dict[
            tuple[int, str], GraphStepResumeReceipt
        ] = {}
        for index, start in starts.items():
            step = plan.steps[index]
            active: _GraphInteractionCycle | None = None
            seen_interactions: set[str] = set()
            step_admitted_interaction_ids = frozenset(
                str(resume_event.payload.get("interaction_id"))
                for resume_step_binding, resume_event in resume_events
                if resume_step_binding.index == index
            )
            step_events = tuple(
                event
                for event in events
                if event.attempt == step.attempt
                and event.store_seq >= start.store_seq
            )
            for event in step_events:
                if event.store_seq in suppressed_interaction_resolutions:
                    continue
                if (
                    active is not None
                    and active.resume_receipt is None
                    and active.interaction_id
                    not in step_admitted_interaction_ids
                    and event.event_type not in _LIVENESS_EXEMPT_EVENT_TYPES
                ):
                    # Run-execution activity while the cycle has neither a
                    # receipt nor any recorded admission proves the attempt
                    # never exited: the prompt was answered through the
                    # in-run callback (the active host boundary may journal
                    # a canonical resolution for it), so the cycle closes
                    # here and never takes a resume admission.  A genuine
                    # durable resume always appends its admission before the
                    # resumed run executes, so its cycle is pre-indexed and
                    # kept.
                    active = None
                if event.event_type in _INTERACTION_REQUESTS:
                    if active is not None and active.resume_receipt is None:
                        raise GraphCheckpointError(
                            "graph step requested a new interaction before the "
                            "previous interaction resumed"
                        )
                    interaction_id = _interaction_id(event)
                    if interaction_id in seen_interactions:
                        raise GraphCheckpointError(
                            "graph interaction request is ambiguous"
                        )
                    seen_interactions.add(interaction_id)
                    active = _GraphInteractionCycle(
                        interaction_id=interaction_id,
                        request_cursor=_cursor(event),
                        aliases=_interaction_aliases(event),
                    )
                    continue
                if event.event_type in _INTERACTION_RESOLUTIONS:
                    interaction_id = _interaction_id(event)
                    is_compatibility_outcome = (
                        event.event_type in LIVE_INTERACTION_OUTCOMES
                    )
                    if active is None:
                        if is_compatibility_outcome:
                            continue
                        raise GraphCheckpointError(
                            "graph interaction resolution has no exact request"
                        )
                    if (
                        interaction_id != active.interaction_id
                        and not (
                            is_compatibility_outcome
                            and interaction_id in active.aliases
                        )
                    ):
                        if is_compatibility_outcome:
                            continue
                        raise GraphCheckpointError(
                            "graph interaction resolution has no exact request"
                        )
                    if active.resolution_cursor is not None:
                        if (
                            is_compatibility_outcome
                            and active.resume_receipt is not None
                        ):
                            continue
                        raise GraphCheckpointError(
                            "graph interaction resolution is ambiguous"
                        )
                    if (
                        is_compatibility_outcome
                        and active.interaction_id
                        not in step_admitted_interaction_ids
                    ):
                        # A live outcome closes the cycle in place: durable
                        # resolutions arrive as ``interaction.resolved``, so
                        # an outcome without any admission means the answer
                        # came through the in-run callback and the attempt
                        # kept running.  A historical admission for this id
                        # keeps the durable reading instead, so legacy
                        # journals that admitted an outcome-resolved pause
                        # stay valid.
                        active = None
                        continue
                    active = _GraphInteractionCycle(
                        interaction_id=active.interaction_id,
                        request_cursor=active.request_cursor,
                        aliases=active.aliases,
                        resolution_cursor=_cursor(event),
                    )
                    continue
                if event.event_type != "graph.step.resume.admitted":
                    continue
                try:
                    receipt = GraphStepResumeReceipt(
                        graph_plan_id=event.payload.get("graph_plan_id"),
                        graph_scope_id=event.payload.get("graph_scope_id"),
                        step=GraphStepBinding.from_dict(event.payload.get("step")),
                        interaction_id=event.payload.get("interaction_id"),
                        request_cursor=EventCursor.from_dict(
                            event.payload.get("request_cursor")
                        ),
                        resolution_cursor=EventCursor.from_dict(
                            event.payload.get("resolution_cursor")
                        ),
                        admitted_cursor=_cursor(event),
                    )
                except (AttributeError, TypeError, ValueError) as error:
                    raise GraphCheckpointError(
                        "graph resume admission is corrupt"
                    ) from error
                if (
                    receipt.graph_plan_id != plan.plan_id
                    or receipt.graph_scope_id != plan.scope_id
                    or receipt.step != step
                ):
                    raise GraphCheckpointConflict(
                        "graph resume admission changed its plan or step"
                    )
                if (
                    active is None
                    or active.interaction_id != receipt.interaction_id
                    or active.resolution_cursor is None
                    or not _same_cursor(
                        active.request_cursor,
                        receipt.request_cursor,
                    )
                    or not _same_cursor(
                        active.resolution_cursor,
                        receipt.resolution_cursor,
                    )
                ):
                    raise GraphCheckpointConflict(
                        "graph resume admission changed its interaction evidence"
                    )
                key = (index, receipt.interaction_id)
                if key in resume_receipts:
                    raise GraphCheckpointError(
                        "graph resume admission is ambiguous"
                    )
                resume_receipts[key] = receipt
                processed_resume_event_keys.add((index, event.event_id))
                active = _GraphInteractionCycle(
                    interaction_id=active.interaction_id,
                    request_cursor=active.request_cursor,
                    aliases=active.aliases,
                    resolution_cursor=active.resolution_cursor,
                    resume_receipt=receipt,
                )
            if active is not None:
                active_interactions[index] = active
        if processed_resume_event_keys != resume_event_keys:
            raise GraphCheckpointError(
                "graph resume admission has no durable step start"
            )

        # A root failure or cancellation ends the graph, but a started step
        # that has no seal yet must still be reported so that recovery can
        # seal the evidence it already has.
        next_index = (
            None
            if graph_terminal is GraphTerminalStatus.COMPLETED
            or len(completed_prefix) == len(plan.steps)
            or step_terminals
            else len(completed_prefix)
        )
        uncertain: int | None = None
        suspended: int | None = None
        resume_ready: int | None = None
        resuming: int | None = None
        if next_index is not None and next_index in starts:
            active = active_interactions.get(next_index)
            if active is None:
                uncertain = next_index
            elif active.resolution_cursor is None:
                suspended = next_index
            elif active.resume_receipt is None:
                resume_ready = next_index
            else:
                resuming = next_index
            next_index = None
        if graph_terminal is not None:
            next_index = None

        terminal_status = graph_terminal
        if terminal_status is None and step_terminals:
            first_terminal = min(step_terminals)
            if any(index > first_terminal for index in starts):
                raise GraphCheckpointError("graph advanced after a failed step")
            terminal_status = step_terminals[first_terminal]
        return _GraphScan(
            recovery=GraphRecovery(
                plan=plan,
                completed_steps=tuple(completed_prefix),
                next_step_index=next_index,
                last_cursor=snapshot.high_water,
                uncertain_step_index=uncertain,
                suspended_step_index=suspended,
                resume_ready_step_index=resume_ready,
                resuming_step_index=resuming,
                terminal_status=terminal_status,
            ),
            snapshot_events=events,
            started_events=starts,
            active_interactions=active_interactions,
            resume_receipts=resume_receipts,
        )


class GraphCheckpointService:
    """Advance a linear graph only from exact durable predecessor evidence."""

    def __init__(
        self,
        *,
        repository: JournalGraphCheckpointRepository,
        artifacts: ArtifactService,
        derived_ingress_resolver: Callable[
            [AttemptRef, AttemptRef], DerivedHandoffInputIngress
        ],
    ) -> None:
        if not isinstance(repository, JournalGraphCheckpointRepository):
            raise TypeError("repository must be a JournalGraphCheckpointRepository")
        if not isinstance(artifacts, ArtifactService):
            raise TypeError("artifacts must be an ArtifactService")
        if artifacts.execution_id != repository.journal.execution_id:
            raise ValueError("graph artifacts and journal must share one execution")
        if not callable(derived_ingress_resolver):
            raise TypeError("derived_ingress_resolver must be callable")
        self.repository = repository
        self.artifacts = artifacts
        self._derived_ingress_resolver = derived_ingress_resolver

    def _event_at(
        self,
        events: Sequence[JournalEvent],
        cursor: EventCursor,
    ) -> JournalEvent:
        matches = tuple(
            event
            for event in events
            if event.store_seq == cursor.store_seq and event.event_id == cursor.event_id
        )
        if len(matches) != 1:
            raise GraphCheckpointError("graph cursor is absent from the journal")
        return matches[0]

    def admit(self, plan: GraphExecutionPlan) -> GraphRecovery:
        if plan.execution_id != self.repository.journal.execution_id:
            raise GraphCheckpointError("graph plan escaped the bound execution")
        snapshot = self.repository.journal.capture_snapshot()
        seed = self._event_at(snapshot.events, plan.initial_input_cursor)
        if seed.attempt != plan.orchestration_attempt or seed.event_type not in {
            "message.user",
            "interaction.resolved",
        }:
            raise GraphCheckpointError(
                "graph initial input cursor is not an orchestration input"
            )
        self.repository.admit(plan)
        return self.recover(plan)

    def _decode_artifact(self, artifact: ArtifactRef) -> Any:
        raw = self.artifacts.read_full(
            artifact,
            remaining_budget_bytes=min(MAX_ARTIFACT_BYTES, artifact.byte_length),
        )
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GraphCheckpointError("graph output artifact is not canonical JSON") from error

    def _step_execution_range(
        self,
        *,
        events: Sequence[JournalEvent],
        step: GraphStepBinding,
        terminal: JournalEvent,
    ) -> EventRange:
        starts = tuple(
            event
            for event in events
            if event.attempt == step.attempt
            and event.event_type == "graph.step.started"
            and event.store_seq < terminal.store_seq
        )
        if len(starts) != 1:
            raise GraphCheckpointError(
                "graph terminal has no unambiguous durable step start"
            )
        first = starts[0]
        interval = tuple(
            event
            for event in events
            if first.store_seq <= event.store_seq <= terminal.store_seq
        )
        if tuple(event.store_seq for event in interval) != tuple(
            range(first.store_seq, terminal.store_seq + 1)
        ):
            raise GraphCheckpointError("graph step event range has a journal gap")
        return EventRange(_cursor(first), _cursor(terminal))

    def _terminal_after_start(
        self,
        scan: _GraphScan,
        step_index: int,
    ) -> JournalEvent | None:
        start = scan.started_events.get(step_index)
        if start is None:
            return None
        step = scan.recovery.plan.steps[step_index]
        attempt_events = tuple(
            event
            for event in scan.snapshot_events
            if event.attempt == step.attempt
            and event.store_seq > start.store_seq
        )
        selection = select_attempt_terminal(attempt_events)
        if selection.ambiguous:
            raise GraphCheckpointError("graph step has ambiguous canonical terminals")
        terminal = selection.event
        return terminal if isinstance(terminal, JournalEvent) else None

    def _completed_output(
        self,
        scan: _GraphScan,
        step: GraphStepBinding,
        terminal: JournalEvent,
    ) -> Mapping[str, Any]:
        finals = tuple(
            event
            for event in scan.snapshot_events
            if event.attempt == step.attempt
            and event.store_seq < terminal.store_seq
            and event.event_type == "final_message"
        )
        if not finals or not isinstance(finals[-1].payload.get("content"), str):
            raise GraphCheckpointError(
                "completed graph step has no canonical final message"
            )
        return {
            "schema": "unchain.graph_step_output.v1",
            "status": "completed",
            "output": finals[-1].payload["content"],
        }

    def _assert_matching_completed_output(
        self,
        supplied_output: Any,
        canonical_output: Mapping[str, Any],
    ) -> None:
        normalized = _thaw_json(
            _freeze_json(supplied_output, path="graph_step_output")
        )
        if normalized in (
            canonical_output,
            canonical_output["output"],
            {"output": canonical_output["output"]},
        ):
            return
        raise GraphCheckpointConflict(
            "graph output disagrees with the canonical final message"
        )

    def _seal_completed_terminal(
        self,
        plan: GraphExecutionPlan,
        step: GraphStepBinding,
        terminal: JournalEvent,
        events: Sequence[JournalEvent],
        full_output: Any,
    ) -> GraphStepCompletion:
        frozen = _freeze_json(full_output, path="graph_step_output")
        artifact = self.artifacts.persist_exact_json(
            _thaw_json(frozen),
            operation_id=_stable_id(
                "graph-output",
                {"scope_id": plan.scope_id, "step_index": step.index},
            ),
            operation_binding={
                "graph_plan_id": plan.plan_id,
                "step": step.to_dict(),
            },
        )
        execution_range = self._step_execution_range(
            events=events,
            step=step,
            terminal=terminal,
        )
        appended = self.repository.complete(
            plan,
            step,
            output_artifact=artifact,
            execution_event_range=execution_range,
            terminal_cursor=_cursor(terminal),
        )
        return GraphStepCompletion(
            step=step,
            output_artifact=artifact,
            source_event_range=EventRange(appended.cursor, appended.cursor),
            terminal_cursor=_cursor(terminal),
            checkpoint_cursor=appended.cursor,
        )

    def recover(self, plan: GraphExecutionPlan) -> GraphRecovery:
        scan = self.repository.scan(plan)
        pending = next(
            (
                index
                for index in (
                    scan.recovery.uncertain_step_index,
                    scan.recovery.suspended_step_index,
                    scan.recovery.resume_ready_step_index,
                    scan.recovery.resuming_step_index,
                )
                if index is not None
            ),
            None,
        )
        if pending is None:
            return scan.recovery
        terminal = self._terminal_after_start(scan, pending)
        if terminal is None:
            return scan.recovery
        step = plan.steps[pending]
        terminal_cursor = _cursor(terminal)
        if terminal.event_type in _COMPLETED_TERMINALS:
            output = self._completed_output(scan, step, terminal)
            self._seal_completed_terminal(
                plan,
                step,
                terminal,
                scan.snapshot_events,
                output,
            )
        elif terminal.event_type in _FAILED_TERMINALS:
            self.repository.terminal(
                plan,
                step,
                status=GraphTerminalStatus.FAILED,
                terminal_cursor=terminal_cursor,
            )
        else:
            self.repository.terminal(
                plan,
                step,
                status=GraphTerminalStatus.CANCELLED,
                terminal_cursor=terminal_cursor,
            )
        return self.repository.scan(plan).recovery

    def read_completed_output(
        self,
        plan: GraphExecutionPlan,
        step_index: int,
    ) -> Mapping[str, Any]:
        """Read one completed step's canonical output envelope from durability."""

        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or not 0 <= step_index < len(plan.steps)
        ):
            raise ValueError("step_index is outside the graph plan")
        recovery = self.recover(plan)
        if step_index >= len(recovery.completed_steps):
            raise GraphCheckpointError("graph step has no completed output")
        scan = self.repository.scan(plan)
        completion = scan.recovery.completed_steps[step_index]
        if completion.step != plan.steps[step_index]:
            raise GraphCheckpointError("graph completed output changed step identity")
        terminal = self._event_at(scan.snapshot_events, completion.terminal_cursor)
        if (
            terminal.attempt != completion.step.attempt
            or terminal.event_type not in _COMPLETED_TERMINALS
        ):
            raise GraphCheckpointError(
                "graph completed output has no canonical terminal"
            )
        expected = self._completed_output(scan, completion.step, terminal)
        persisted = self._decode_artifact(completion.output_artifact)
        if persisted != expected:
            raise GraphCheckpointError(
                "graph completed output artifact changed its canonical envelope"
            )
        return dict(expected)

    def _source_for_step(
        self,
        plan: GraphExecutionPlan,
        recovery: GraphRecovery,
        step: GraphStepBinding,
    ) -> tuple[Any, EventRange, tuple[ResourceRef, ...]]:
        if step.index == 0:
            snapshot = self.repository.journal.capture_snapshot()
            seed = self._event_at(snapshot.events, plan.initial_input_cursor)
            return (
                {
                    "schema": "unchain.graph_input_seed.v1",
                    "input_event": seed.to_dict(),
                },
                EventRange(plan.initial_input_cursor, plan.initial_input_cursor),
                (),
            )
        predecessor = recovery.completed_steps[step.index - 1]
        if predecessor.step.attempt != step.source_attempt:
            raise GraphCheckpointError("graph predecessor completion changed identity")
        return (
            self._decode_artifact(predecessor.output_artifact),
            predecessor.source_event_range,
            (predecessor.output_artifact.ref,),
        )

    def resolved_interaction_for_step(
        self,
        plan: GraphExecutionPlan,
        step_index: int,
        *,
        interaction_id: str,
    ) -> GraphStepResumeEvidence:
        """Read exact durable resume evidence without reconstructing it in a host."""

        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or not 0 <= step_index < len(plan.steps)
        ):
            raise ValueError("step_index is outside the graph plan")
        interaction_id = _required_text(
            interaction_id,
            "interaction_id",
            identifier=True,
        )
        scan = self.repository.scan(plan)
        if scan.recovery.terminal_status is not None:
            raise GraphCheckpointError(
                "terminal graph execution has no resumable interaction"
            )
        terminal = self._terminal_after_start(scan, step_index)
        if terminal is not None:
            raise GraphCheckpointError(
                "graph step already has a canonical terminal"
            )
        if step_index not in {
            scan.recovery.resume_ready_step_index,
            scan.recovery.resuming_step_index,
        }:
            raise GraphCheckpointError(
                "graph step has no resolved resumable interaction"
            )
        active = scan.active_interactions.get(step_index)
        if active is None or active.resolution_cursor is None:
            raise GraphCheckpointError(
                "graph step has no exact resolved interaction evidence"
            )
        if active.interaction_id != interaction_id:
            raise GraphCheckpointConflict(
                "graph resume interaction identity changed"
            )
        return GraphStepResumeEvidence(
            graph_plan_id=plan.plan_id,
            graph_scope_id=plan.scope_id,
            step=plan.steps[step_index],
            interaction_id=active.interaction_id,
            request_cursor=active.request_cursor,
            resolution_cursor=active.resolution_cursor,
        )

    def resume_step(
        self,
        plan: GraphExecutionPlan,
        step_index: int,
        *,
        interaction_id: str,
        request_cursor: EventCursor,
        resolution_cursor: EventCursor,
    ) -> GraphStepResumeReceipt:
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or not 0 <= step_index < len(plan.steps)
        ):
            raise ValueError("step_index is outside the graph plan")
        interaction_id = _required_text(
            interaction_id,
            "interaction_id",
            identifier=True,
        )
        if not isinstance(request_cursor, EventCursor):
            request_cursor = EventCursor.from_dict(request_cursor)
        if not isinstance(resolution_cursor, EventCursor):
            resolution_cursor = EventCursor.from_dict(resolution_cursor)
        if request_cursor.store_seq >= resolution_cursor.store_seq:
            raise GraphCheckpointConflict(
                "graph resume resolution must follow its exact request"
            )

        scan = self.repository.scan(plan)
        key = (step_index, interaction_id)
        existing = scan.resume_receipts.get(key)
        if existing is not None:
            if (
                existing.step != plan.steps[step_index]
                or existing.graph_plan_id != plan.plan_id
                or existing.graph_scope_id != plan.scope_id
                or not _same_cursor(existing.request_cursor, request_cursor)
                or not _same_cursor(existing.resolution_cursor, resolution_cursor)
            ):
                raise GraphCheckpointConflict(
                    "graph resume replay changed its durable evidence"
                )
            return existing

        recovery = self.recover(plan)
        if recovery.terminal_status is not None:
            raise GraphCheckpointError("terminal graph execution cannot resume")
        if recovery.suspended_step_index == step_index:
            raise GraphCheckpointError(
                "graph interaction has not been durably resolved"
            )
        if recovery.resuming_step_index == step_index:
            raise GraphCheckpointError(
                "graph step already has a different resume admission"
            )
        if recovery.resume_ready_step_index != step_index:
            raise GraphCheckpointError(
                "graph step is not ready for interaction resume"
            )

        scan = self.repository.scan(plan)
        active = scan.active_interactions.get(step_index)
        if (
            active is None
            or active.interaction_id != interaction_id
            or active.resolution_cursor is None
            or active.resume_receipt is not None
            or not _same_cursor(active.request_cursor, request_cursor)
            or not _same_cursor(active.resolution_cursor, resolution_cursor)
        ):
            raise GraphCheckpointConflict(
                "graph resume input changed its exact interaction evidence"
            )
        step = plan.steps[step_index]
        appended = self.repository.resume(
            plan,
            step,
            interaction_id=interaction_id,
            request_cursor=request_cursor,
            resolution_cursor=resolution_cursor,
        )
        if appended.cursor.store_seq <= resolution_cursor.store_seq:
            raise GraphCheckpointError(
                "graph resume admission did not follow its resolution"
            )
        rescanned = self.repository.scan(plan)
        receipt = rescanned.resume_receipts.get(key)
        if (
            receipt is None
            or not _same_cursor(receipt.admitted_cursor, appended.cursor)
            or rescanned.recovery.resuming_step_index != step_index
        ):
            raise GraphCheckpointError("graph resume admission did not persist")
        return receipt

    def start_step(
        self,
        plan: GraphExecutionPlan,
        step_index: int,
    ) -> GraphStepStartReceipt:
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or not 0 <= step_index < len(plan.steps)
        ):
            raise ValueError("step_index is outside the graph plan")
        recovery = self.recover(plan)
        if step_index < len(recovery.completed_steps):
            return GraphStepStartReceipt(
                disposition=GraphStepDisposition.SKIP_COMPLETED,
                step=plan.steps[step_index],
                completion=recovery.completed_steps[step_index],
            )
        if recovery.uncertain_step_index is not None:
            raise GraphCheckpointError(
                "graph step started without a durable terminal; replay is forbidden"
            )
        if recovery.suspended_step_index is not None:
            raise GraphCheckpointError(
                "graph step is durably suspended and requires interaction resume"
            )
        if recovery.resume_ready_step_index is not None:
            raise GraphCheckpointError(
                "graph step has a resolved interaction and requires resume admission"
            )
        if recovery.resuming_step_index is not None:
            raise GraphCheckpointError(
                "graph step is already resuming; start replay is forbidden"
            )
        if recovery.terminal_status is not None:
            raise GraphCheckpointError("terminal graph execution cannot advance")
        if recovery.next_step_index != step_index:
            raise GraphCheckpointError("graph steps must advance in plan order")
        step = plan.steps[step_index]
        full_output, source_range, artifact_refs = self._source_for_step(
            plan,
            recovery,
            step,
        )
        ingress = self._derived_ingress_resolver(
            step.attempt,
            step.source_attempt,
        )
        if (
            type(ingress) is not DerivedHandoffInputIngress
            or ingress.consumer_attempt != step.attempt
            or ingress.source_attempt != step.source_attempt
        ):
            raise GraphCheckpointError("graph derived ingress changed attempt identity")
        handoff = ingress.persist(
            HostResolvedDerivedHandoffInput(
                consumer_attempt=step.attempt,
                source_attempt=step.source_attempt,
                status=HandoffStatus.COMPLETE,
                full_output=full_output,
                source_event_range=source_range,
                operation_id=_stable_id(
                    "graph-handoff",
                    {"scope_id": plan.scope_id, "step_index": step.index},
                ),
                artifact_refs=artifact_refs,
            )
        )
        started = self.repository.start(plan, step, handoff)
        if started.cursor.store_seq <= handoff.input_cursor.store_seq:
            raise GraphCheckpointError("graph step started before its derived input")
        return GraphStepStartReceipt(
            disposition=GraphStepDisposition.STARTED,
            step=step,
            started_cursor=started.cursor,
            handoff=handoff,
        )

    def complete_step(
        self,
        plan: GraphExecutionPlan,
        step_index: int,
        *,
        full_output: Any,
    ) -> GraphStepCompletion:
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or not 0 <= step_index < len(plan.steps)
        ):
            raise ValueError("step_index is outside the graph plan")
        scan = self.repository.scan(plan)
        recovery = scan.recovery
        if step_index < len(recovery.completed_steps):
            completion = recovery.completed_steps[step_index]
            expected = self._decode_artifact(completion.output_artifact)
            if not isinstance(expected, Mapping):
                raise GraphCheckpointError(
                    "graph completion artifact has an invalid schema"
                )
            self._assert_matching_completed_output(full_output, expected)
            return completion
        if step_index not in scan.started_events:
            raise GraphCheckpointError("graph step completion has no durable start")
        terminal = self._terminal_after_start(scan, step_index)
        if terminal is None or terminal.event_type not in _COMPLETED_TERMINALS:
            raise GraphCheckpointError(
                "graph step completion requires a canonical completed terminal"
            )
        expected_output = self._completed_output(
            scan,
            plan.steps[step_index],
            terminal,
        )
        self._assert_matching_completed_output(full_output, expected_output)
        return self._seal_completed_terminal(
            plan,
            plan.steps[step_index],
            terminal,
            scan.snapshot_events,
            expected_output,
        )

    def fail_step(
        self,
        plan: GraphExecutionPlan,
        step_index: int,
        *,
        status: GraphTerminalStatus,
    ) -> GraphRecovery:
        """Seal a step whose run ended with a failed or cancelled terminal.

        The run-time counterpart of the seal ``recover`` writes later: it
        requires the canonical terminal to be in the journal already, and the
        requested status to agree with it. Calling it again for an already
        sealed step changes nothing. The orchestration attempt is left as it
        is: rebase accepts a sealed dead step under an admitted-only root.
        """

        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or not 0 <= step_index < len(plan.steps)
        ):
            raise ValueError("step_index is outside the graph plan")
        if type(status) is not GraphTerminalStatus:
            raise TypeError("status must be a GraphTerminalStatus")
        if status is GraphTerminalStatus.COMPLETED:
            raise ValueError("a completed graph step is sealed by complete_step")
        scan = self.repository.scan(plan)
        step = plan.steps[step_index]
        if step_index not in scan.started_events:
            raise GraphCheckpointError("graph step failure has no durable start")
        seals = {
            event.event_type
            for event in scan.snapshot_events
            if event.attempt == step.attempt
            and event.event_type
            in {"graph.step.completed", "graph.step.failed", "graph.step.cancelled"}
        }
        if "graph.step.completed" in seals:
            raise GraphCheckpointConflict("graph step already completed")
        if f"graph.step.{status.value}" in seals:
            return scan.recovery
        if seals:
            raise GraphCheckpointConflict(
                "graph step was already sealed with a different terminal"
            )
        terminal = self._terminal_after_start(scan, step_index)
        if terminal is None or terminal.event_type in _COMPLETED_TERMINALS:
            raise GraphCheckpointError(
                "graph step failure requires a canonical failed or cancelled terminal"
            )
        expected = (
            _FAILED_TERMINALS
            if status is GraphTerminalStatus.FAILED
            else _CANCELLED_TERMINALS
        )
        if terminal.event_type not in expected:
            raise GraphCheckpointConflict(
                "graph step terminal disagrees with the requested status"
            )
        self.repository.terminal(
            plan,
            step,
            status=status,
            terminal_cursor=_cursor(terminal),
        )
        return self.repository.scan(plan).recovery

    def finalize(self, plan: GraphExecutionPlan) -> GraphRecovery:
        recovery = self.recover(plan)
        if recovery.is_complete:
            return recovery
        if recovery.terminal_status in {
            GraphTerminalStatus.FAILED,
            GraphTerminalStatus.CANCELLED,
        }:
            raise GraphCheckpointError("terminal graph execution cannot finalize")
        if len(recovery.completed_steps) != len(plan.steps):
            raise GraphCheckpointError("graph cannot finalize before every step completes")
        self.repository.finalize(plan, recovery.completed_steps[-1])
        finalized = self.repository.scan(plan).recovery
        if not finalized.is_complete:
            raise GraphCheckpointError("graph terminal did not persist")
        return finalized


__all__ = [
    "GraphCheckpointConflict",
    "GraphCheckpointError",
    "GraphCheckpointService",
    "GraphExecutionPlan",
    "GraphInteractionLineageProof",
    "GraphRecovery",
    "GraphStepBinding",
    "GraphStepCompletion",
    "GraphStepDisposition",
    "GraphStepResumeEvidence",
    "GraphStepResumeReceipt",
    "GraphStepStartReceipt",
    "GraphTerminalStatus",
    "JournalGraphCheckpointRepository",
    "locate_graph_execution_plan",
    "prove_graph_interaction_lineage",
]
