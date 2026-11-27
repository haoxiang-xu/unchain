from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..journal.models import (
    ArtifactRef,
    AttemptRef,
    ResourceRef,
    _freeze_json,
    _record_tuple,
    _required_text,
    _sha256,
    _thaw_json,
)


_SUBAGENT_SNAPSHOT_KIND = "subagent_snapshot"
_SUBAGENT_SNAPSHOT_CONTRACT = "subagent_snapshot.v1"
_SAFE_SUBAGENT_STATE_KEYS = frozenset({"subagent_state"})
_SUBAGENT_TERMINAL_HANDOFF_KIND = "subagent_terminal_handoff"
_SUBAGENT_TERMINAL_HANDOFF_CONTRACT = "subagent_terminal_handoff.v1"
_SUBAGENT_TERMINAL_HANDOFF_STATE_KEYS = frozenset(
    {
        "subagent_state",
        "transcript",
        "run_status",
        "pending_tool_calls",
        "tool_batch_state",
        "last_continuation",
        "next_model_input",
    }
)
_SUPPORTED_SUBAGENT_TRANSITION_KINDS = frozenset(
    {
        _SUBAGENT_SNAPSHOT_KIND,
        _SUBAGENT_TERMINAL_HANDOFF_KIND,
    }
)
_SUPPORTED_SUBAGENT_TRANSITION_CONTRACTS = frozenset(
    {
        _SUBAGENT_SNAPSHOT_CONTRACT,
        _SUBAGENT_TERMINAL_HANDOFF_CONTRACT,
    }
)


def _contract_error(message: str) -> Exception:
    from .tool_executor import DurableToolExecutorContractError

    return DurableToolExecutorContractError(message)


def _canonical_state(value: Any, *, field_name: str) -> Any:
    try:
        frozen = _freeze_json(value, path=field_name)
    except (TypeError, ValueError) as exc:
        raise _contract_error(f"{field_name} must be canonical JSON") from exc
    thawed = _thaw_json(frozen)
    if not isinstance(thawed, dict):
        raise _contract_error(f"{field_name} must be an object")
    return frozen


def canonical_state_sha256(value: Any) -> str:
    frozen = _freeze_json(value, path="durable_state_digest")
    content = json.dumps(
        _thaw_json(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _exact_subagent_state_snapshot(value: Any, *, field_name: str) -> dict[str, Any]:
    from ..subagents.types import SubagentState

    if type(value) is SubagentState:
        raw = value.to_dict()
    elif type(value) is dict:
        raw = value
    else:
        raise _contract_error(f"{field_name} requires an exact SubagentState snapshot")
    try:
        canonical = _thaw_json(_freeze_json(raw, path=field_name))
    except (TypeError, ValueError) as exc:
        raise _contract_error(f"{field_name} must be canonical JSON") from exc
    state = SubagentState.from_raw(canonical)
    if state.to_dict() != canonical:
        raise _contract_error(f"{field_name} is not an exact SubagentState snapshot")
    return state.to_dict()


def _default_tool_batch_state_snapshot() -> dict[str, Any]:
    return {
        "result_messages": [],
        "should_observe": False,
        "awaiting_human_input": False,
        "human_input_request": None,
        "human_input_tool_call_id": None,
        "executed_call_ids": [],
    }


def _canonical_json_value(value: Any, *, field_name: str) -> Any:
    try:
        return _thaw_json(_freeze_json(value, path=field_name))
    except (TypeError, ValueError) as exc:
        raise _contract_error(f"{field_name} must be canonical JSON") from exc


def _message_list_snapshot(
    value: Any,
    *,
    field_name: str,
    optional: bool = False,
) -> list[dict[str, Any]] | None:
    if value is None and optional:
        return None
    canonical = _canonical_json_value(value, field_name=field_name)
    if type(canonical) is not list or any(type(item) is not dict for item in canonical):
        raise _contract_error(f"{field_name} must be a message list")
    return canonical


def _pending_tool_calls_snapshot(value: Any) -> list[dict[str, Any]]:
    from ..kernel.types import ToolCall

    if type(value) is not list:
        raise _contract_error(
            "subagent terminal handoff pending tool calls must be a list"
        )
    calls: list[dict[str, Any]] = []
    for item in value:
        if type(item) is ToolCall:
            raw = {
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
            }
        elif type(item) is dict:
            raw = dict(item)
        else:
            raise _contract_error(
                "subagent terminal handoff contains an invalid tool call"
            )
        if set(raw) != {"call_id", "name", "arguments"}:
            raise _contract_error(
                "subagent terminal handoff tool call schema is invalid"
            )
        if type(raw["call_id"]) is not str or type(raw["name"]) is not str:
            raise _contract_error(
                "subagent terminal handoff tool call identity is invalid"
            )
        calls.append(
            _canonical_json_value(
                raw,
                field_name="subagent_terminal_handoff.pending_tool_calls",
            )
        )
    return calls


def _tool_batch_state_snapshot(value: Any) -> dict[str, Any]:
    from ..input.human_input import HumanInputRequest
    from ..tools.types import ToolBatchState

    if type(value) is ToolBatchState:
        raw = {
            "result_messages": value.result_messages,
            "should_observe": value.should_observe,
            "awaiting_human_input": value.awaiting_human_input,
            "human_input_request": value.human_input_request,
            "human_input_tool_call_id": value.human_input_tool_call_id,
            "executed_call_ids": value.executed_call_ids,
        }
    elif type(value) is dict:
        if value == {}:
            return _default_tool_batch_state_snapshot()
        raw = dict(value)
    else:
        raise _contract_error("subagent terminal handoff tool batch state is invalid")
    if set(raw) != set(_default_tool_batch_state_snapshot()):
        raise _contract_error(
            "subagent terminal handoff tool batch state schema is invalid"
        )
    human_input_request = raw["human_input_request"]
    if type(human_input_request) is HumanInputRequest:
        human_input_request = human_input_request.to_dict()
    elif human_input_request is not None:
        human_input_request = _canonical_json_value(
            human_input_request,
            field_name="subagent_terminal_handoff.human_input_request",
        )
        try:
            verified_request = HumanInputRequest.from_dict(
                human_input_request
            ).to_dict()
        except (TypeError, ValueError) as exc:
            raise _contract_error(
                "subagent terminal handoff human input request is invalid"
            ) from exc
        if verified_request != human_input_request:
            raise _contract_error(
                "subagent terminal handoff human input request is not exact"
            )
    tool_call_id = raw["human_input_tool_call_id"]
    if tool_call_id is not None and type(tool_call_id) is not str:
        raise _contract_error(
            "subagent terminal handoff human input call id is invalid"
        )
    if (
        type(raw["should_observe"]) is not bool
        or type(raw["awaiting_human_input"]) is not bool
    ):
        raise _contract_error("subagent terminal handoff tool batch flags are invalid")
    executed_call_ids = raw["executed_call_ids"]
    if type(executed_call_ids) is not list or any(
        type(call_id) is not str for call_id in executed_call_ids
    ):
        raise _contract_error("subagent terminal handoff executed call ids are invalid")
    return {
        "result_messages": _message_list_snapshot(
            raw["result_messages"],
            field_name="subagent_terminal_handoff.result_messages",
        ),
        "should_observe": raw["should_observe"],
        "awaiting_human_input": raw["awaiting_human_input"],
        "human_input_request": human_input_request,
        "human_input_tool_call_id": tool_call_id,
        "executed_call_ids": list(executed_call_ids),
    }


def _terminal_handoff_state_snapshot(
    value: Any,
    *,
    require_terminal: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _contract_error("subagent terminal handoff state must be an object")
    raw = dict(value)
    if set(raw) != _SUBAGENT_TERMINAL_HANDOFF_STATE_KEYS:
        raise _contract_error(
            "subagent terminal handoff requires one exact terminal state"
        )
    run_status = raw["run_status"]
    if type(run_status) is not str or not run_status:
        raise _contract_error("subagent terminal handoff run status is invalid")
    pending_tool_calls = _pending_tool_calls_snapshot(raw["pending_tool_calls"])
    tool_batch_state = _tool_batch_state_snapshot(raw["tool_batch_state"])
    last_continuation = raw["last_continuation"]
    if last_continuation is not None:
        last_continuation = _canonical_json_value(
            last_continuation,
            field_name="subagent_terminal_handoff.last_continuation",
        )
        if type(last_continuation) is not dict:
            raise _contract_error("subagent terminal handoff continuation is invalid")
    next_model_input = _message_list_snapshot(
        raw["next_model_input"],
        field_name="subagent_terminal_handoff.next_model_input",
        optional=True,
    )
    if require_terminal:
        child_wait = None
        if run_status == "awaiting_human_input" and raw["transcript"]:
            from ..tools.runtime import decode_subagent_wait_marker
            child_wait = decode_subagent_wait_marker(
                raw["transcript"][-1].get("content")
                if type(raw["transcript"][-1]) is dict else None
            )
        if run_status not in {"completed", "awaiting_human_input"} or (
            run_status == "awaiting_human_input" and child_wait is None
        ):
            raise _contract_error(
                "subagent terminal handoff run status is invalid"
            )
        if pending_tool_calls:
            raise _contract_error(
                "subagent terminal handoff cannot retain pending tool calls"
            )
        if tool_batch_state != _default_tool_batch_state_snapshot():
            raise _contract_error(
                "subagent terminal handoff tool batch state must be closed"
            )
        if last_continuation is not None:
            raise _contract_error(
                "subagent terminal handoff cannot retain a continuation"
            )
        if next_model_input is not None:
            raise _contract_error("subagent terminal handoff cannot retain model input")
    return {
        "subagent_state": _exact_subagent_state_snapshot(
            raw["subagent_state"],
            field_name="subagent_terminal_handoff.subagent_state",
        ),
        "transcript": _message_list_snapshot(
            raw["transcript"],
            field_name="subagent_terminal_handoff.transcript",
        ),
        "run_status": run_status,
        "pending_tool_calls": pending_tool_calls,
        "tool_batch_state": tool_batch_state,
        "last_continuation": last_continuation,
        "next_model_input": next_model_input,
    }


def _terminal_handoff_base_snapshot(value: Any) -> dict[str, Any]:
    return _terminal_handoff_state_snapshot(value, require_terminal=False)


def _terminal_handoff_next_snapshot(value: Any) -> dict[str, Any]:
    return _terminal_handoff_state_snapshot(value, require_terminal=True)


def load_verified_subagent_state(
    artifacts: Any,
    transition: DurableToolStateTransitionEnvelope,
):
    """Read one exact execution-scoped next-state artifact fail closed."""

    from .artifacts import ArtifactService, _bounded_utf8_preview
    from ..subagents.types import SubagentState

    if type(artifacts) is not ArtifactService:
        raise _contract_error(
            "next-state verification requires the official artifact service"
        )
    if type(transition) is not DurableToolStateTransitionEnvelope:
        raise _contract_error(
            "next-state verification requires a runtime-owned transition"
        )
    if artifacts.execution_id != transition.parent_attempt.generation.execution_id:
        raise _contract_error("next-state artifact crossed its execution scope")
    if transition.handoff_refs:
        raise _contract_error(
            "sealed handoff refs require an execution-bound artifact descriptor lookup"
        )
    artifact = transition.next_state_artifact
    content = artifacts.read_full(
        artifact,
        remaining_budget_bytes=artifact.byte_length,
    )
    expected_preview = _bounded_utf8_preview(content)
    if artifact.preview != expected_preview:
        raise _contract_error("next-state artifact preview changed")
    try:
        decoded = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _contract_error("next-state artifact is not valid UTF-8 JSON") from exc
    canonical = json.dumps(
        decoded,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical != content or not isinstance(decoded, dict):
        raise _contract_error("next-state artifact is not exact canonical JSON")
    if transition.kind == _SUBAGENT_SNAPSHOT_KIND:
        state = SubagentState.from_raw(decoded)
        if (
            json.dumps(
                state.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            != content
        ):
            raise _contract_error(
                "next-state artifact is not an exact SubagentState snapshot"
            )
        return state
    if transition.kind == _SUBAGENT_TERMINAL_HANDOFF_KIND:
        snapshot = _terminal_handoff_next_snapshot(decoded)
        if snapshot != decoded:
            raise _contract_error(
                "next-state artifact is not an exact terminal handoff snapshot"
            )
        from ..tools.types import ToolBatchState

        return {
            "subagent_state": SubagentState.from_raw(snapshot["subagent_state"]),
            "transcript": snapshot["transcript"],
            "run_status": snapshot["run_status"],
            "pending_tool_calls": [],
            "tool_batch_state": ToolBatchState(),
            "last_continuation": None,
            "next_model_input": None,
        }
    raise _contract_error("durable tool state transition kind is unsupported")


def resolve_subagent_transition_cas(
    *,
    artifacts: Any,
    transition: DurableToolStateTransitionEnvelope,
    current_state: Any,
):
    """Materialize an exact replacement under base/next/conflict CAS."""

    from ..subagents.types import SubagentState

    next_state = load_verified_subagent_state(artifacts, transition)
    if transition.kind == _SUBAGENT_SNAPSHOT_KIND:
        if type(current_state) is not SubagentState:
            raise _contract_error(
                "state transition CAS requires an exact SubagentState"
            )
        current_sha256 = canonical_state_sha256(current_state.to_dict())
        next_sha256 = canonical_state_sha256(next_state.to_dict())
        if next_sha256 != transition.next_state_sha256:
            raise _contract_error(
                "next-state artifact digest disagrees with its canonical state"
            )
        if current_sha256 == transition.base_state_sha256:
            return next_state
        if current_sha256 == next_sha256:
            return None
        raise _contract_error("state transition CAS conflict")
    if transition.kind != _SUBAGENT_TERMINAL_HANDOFF_KIND:
        raise _contract_error("durable tool state transition kind is unsupported")

    from ..kernel.state import RunState

    if type(current_state) is not RunState:
        raise _contract_error("terminal handoff CAS requires an exact RunState")
    next_snapshot = _terminal_handoff_next_snapshot(next_state)
    next_sha256 = canonical_state_sha256(next_snapshot)
    if next_sha256 != transition.next_state_sha256:
        raise _contract_error(
            "next-state artifact digest disagrees with its canonical state"
        )
    try:
        current_snapshot = _terminal_handoff_base_snapshot(
            {
                "subagent_state": current_state.subagent_state,
                "transcript": current_state.transcript,
                "run_status": current_state.run_status,
                "pending_tool_calls": current_state.pending_tool_calls,
                "tool_batch_state": current_state.tool_batch_state,
                "last_continuation": current_state.last_continuation,
                "next_model_input": current_state.next_model_input,
            }
        )
    except Exception as exc:
        raise _contract_error("state transition CAS conflict") from exc
    current_sha256 = canonical_state_sha256(current_snapshot)
    if current_sha256 == transition.base_state_sha256:
        return next_state
    if current_sha256 == next_sha256:
        return None
    raise _contract_error("state transition CAS conflict")


@dataclass(frozen=True, init=False)
class DurableToolStateTransitionDraft:
    """In-memory state change awaiting executor-owned durable sealing."""

    kind: str
    _base_state: Any = field(repr=False)
    _next_state: Any = field(repr=False)
    handoff_refs: tuple[ResourceRef, ...] = ()

    def __init__(
        self,
        *,
        kind: str,
        base_state: Any,
        next_state: Any,
        handoff_refs: Sequence[ResourceRef] = (),
    ) -> None:
        if kind not in _SUPPORTED_SUBAGENT_TRANSITION_KINDS:
            raise _contract_error("durable tool state transition kind is unsupported")
        if kind == _SUBAGENT_TERMINAL_HANDOFF_KIND:
            base_state = _terminal_handoff_base_snapshot(base_state)
            next_state = _terminal_handoff_next_snapshot(next_state)
        object.__setattr__(
            self,
            "kind",
            kind,
        )
        object.__setattr__(
            self,
            "_base_state",
            _canonical_state(
                base_state,
                field_name="durable_tool_state_transition.base_state",
            ),
        )
        object.__setattr__(
            self,
            "_next_state",
            _canonical_state(
                next_state,
                field_name="durable_tool_state_transition.next_state",
            ),
        )
        refs = _record_tuple(
            handoff_refs,
            ResourceRef,
            "handoff_refs",
        )
        if any(ref.kind != "artifact" or ref.fragment for ref in refs):
            raise _contract_error("durable tool handoff refs must be whole artifacts")
        object.__setattr__(self, "handoff_refs", refs)

    @property
    def base_state(self) -> dict[str, Any]:
        return _thaw_json(self._base_state)

    @property
    def next_state(self) -> dict[str, Any]:
        return _thaw_json(self._next_state)


@dataclass(frozen=True)
class DurableToolStateTransitionEnvelope:
    """Receipt-bound reference to one sanitized next-state snapshot."""

    SCHEMA: ClassVar[str] = "unchain.durable_tool_state_transition.v1"

    parent_attempt: AttemptRef
    call_id: str
    execution_subject_sha256: str
    operation_id: str
    kind: str
    base_state_sha256: str
    next_state_artifact: ArtifactRef
    handoff_refs: tuple[ResourceRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.parent_attempt, AttemptRef):
            object.__setattr__(
                self,
                "parent_attempt",
                AttemptRef.from_dict(self.parent_attempt),
            )
        object.__setattr__(
            self,
            "call_id",
            _required_text(self.call_id, "call_id", identifier=True),
        )
        object.__setattr__(
            self,
            "execution_subject_sha256",
            _sha256(
                self.execution_subject_sha256,
                "execution_subject_sha256",
            ),
        )
        object.__setattr__(
            self,
            "operation_id",
            _required_text(self.operation_id, "operation_id", identifier=True),
        )
        expected_operation_id = (
            "transition.subagent-completion." + self.execution_subject_sha256
        )
        if self.operation_id != expected_operation_id:
            raise _contract_error(
                "transition operation must be derived from its execution subject"
            )
        if self.kind not in _SUPPORTED_SUBAGENT_TRANSITION_KINDS:
            raise _contract_error("durable tool state transition kind is unsupported")
        object.__setattr__(
            self,
            "base_state_sha256",
            _sha256(self.base_state_sha256, "base_state_sha256"),
        )
        if not isinstance(self.next_state_artifact, ArtifactRef):
            object.__setattr__(
                self,
                "next_state_artifact",
                ArtifactRef.from_dict(self.next_state_artifact),
            )
        if (
            self.next_state_artifact.ref.kind != "artifact"
            or self.next_state_artifact.ref.fragment
            or self.next_state_artifact.media_type != "application/json"
        ):
            raise _contract_error("durable tool next-state artifact is invalid")
        refs = _record_tuple(
            self.handoff_refs,
            ResourceRef,
            "handoff_refs",
        )
        if any(ref.kind != "artifact" or ref.fragment for ref in refs):
            raise _contract_error("durable tool handoff refs must be whole artifacts")
        object.__setattr__(self, "handoff_refs", refs)

    @property
    def next_state_sha256(self) -> str:
        return self.next_state_artifact.sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "parent_attempt": self.parent_attempt.to_dict(),
            "call_id": self.call_id,
            "execution_subject_sha256": self.execution_subject_sha256,
            "operation_id": self.operation_id,
            "kind": self.kind,
            "base_state_sha256": self.base_state_sha256,
            "next_state_artifact": self.next_state_artifact.to_dict(),
            "handoff_refs": [ref.to_dict() for ref in self.handoff_refs],
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> DurableToolStateTransitionEnvelope:
        if cls is not DurableToolStateTransitionEnvelope:
            raise _contract_error(
                "durable tool state transition must be an exact object"
            )
        if not isinstance(value, Mapping):
            raise _contract_error("durable tool state transition must be an object")
        raw = dict(value)
        expected = {
            "schema",
            "parent_attempt",
            "call_id",
            "execution_subject_sha256",
            "operation_id",
            "kind",
            "base_state_sha256",
            "next_state_artifact",
            "handoff_refs",
        }
        if set(raw) != expected or raw.get("schema") != cls.SCHEMA:
            raise _contract_error("durable tool state transition schema is invalid")
        refs = raw["handoff_refs"]
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
            raise _contract_error("durable tool handoff refs are invalid")
        return cls(
            parent_attempt=AttemptRef.from_dict(raw["parent_attempt"]),
            call_id=raw["call_id"],
            execution_subject_sha256=raw["execution_subject_sha256"],
            operation_id=raw["operation_id"],
            kind=raw["kind"],
            base_state_sha256=raw["base_state_sha256"],
            next_state_artifact=ArtifactRef.from_dict(raw["next_state_artifact"]),
            handoff_refs=tuple(ResourceRef.from_dict(ref) for ref in refs),
        )


def build_declared_tool_state_transition(
    *,
    completion_contract: Any,
    state_updates: Mapping[str, Any],
    base_state: Any,
) -> DurableToolStateTransitionDraft | None:
    """Validate one plugin-declared, P0-safe state transition draft."""

    if not isinstance(completion_contract, Mapping):
        raise _contract_error(
            "tool runtime plugin state transition has no bound manifest"
        )
    contract = dict(completion_contract)
    if contract.get("schema") != "unchain.tool_completion_contract.v1":
        raise _contract_error(
            "tool runtime plugin state transition manifest is invalid"
        )

    is_variant_contract = set(contract) == {
        "schema",
        "state_transition_variants",
    }
    if is_variant_contract:
        raw_variants = contract.get("state_transition_variants")
        if type(raw_variants) is not list or len(raw_variants) != 2:
            raise _contract_error(
                "tool runtime plugin state transition variants are invalid"
            )
        variants: list[tuple[str, frozenset[str]]] = []
        for raw_variant in raw_variants:
            if type(raw_variant) is not dict or set(raw_variant) != {
                "state_transition",
                "allowed_state_keys",
            }:
                raise _contract_error(
                    "tool runtime plugin state transition variant is invalid"
                )
            variants.append(
                _validated_state_transition_variant(
                    transition_contract=raw_variant.get("state_transition"),
                    allowed_keys=raw_variant.get("allowed_state_keys"),
                )
            )
        if {
            transition_contract for transition_contract, _allowed_keys in variants
        } != _SUPPORTED_SUBAGENT_TRANSITION_CONTRACTS:
            raise _contract_error(
                "tool runtime plugin state transition variants are invalid"
            )
    elif set(contract) == {
        "schema",
        "state_transition",
        "allowed_state_keys",
    }:
        variants = [
            _validated_state_transition_variant(
                transition_contract=contract.get("state_transition"),
                allowed_keys=contract.get("allowed_state_keys"),
            )
        ]
    else:
        raise _contract_error(
            "tool runtime plugin state transition manifest is invalid"
        )

    updates = dict(state_updates)
    if not updates:
        return None
    matching_variants = [
        transition_contract
        for transition_contract, allowed_keys in variants
        if set(updates) == allowed_keys
    ]
    if len(matching_variants) != 1:
        raise _contract_error(
            "tool runtime plugin state transition requires one exact declared variant"
        )
    transition_contract = matching_variants[0]
    if transition_contract == _SUBAGENT_TERMINAL_HANDOFF_CONTRACT:
        return DurableToolStateTransitionDraft(
            kind=_SUBAGENT_TERMINAL_HANDOFF_KIND,
            base_state=_terminal_handoff_base_snapshot(base_state),
            next_state=_terminal_handoff_next_snapshot(updates),
        )
    snapshot_base_state = base_state
    if is_variant_contract:
        snapshot_base_state = _terminal_handoff_base_snapshot(base_state)[
            "subagent_state"
        ]
    raw_state = updates["subagent_state"]
    from ..subagents.types import SubagentState

    if type(raw_state) is SubagentState:
        next_state = raw_state.to_dict()
    elif type(raw_state) is dict:
        next_state = raw_state
    else:
        raise _contract_error(
            "subagent state transition requires an exact state snapshot"
        )
    if type(snapshot_base_state) is SubagentState:
        base_snapshot = snapshot_base_state.to_dict()
    elif type(snapshot_base_state) is dict:
        base_snapshot = snapshot_base_state
    else:
        raise _contract_error(
            "subagent state transition requires an exact base snapshot"
        )
    return DurableToolStateTransitionDraft(
        kind=_SUBAGENT_SNAPSHOT_KIND,
        base_state=base_snapshot,
        next_state=next_state,
    )


def _validated_state_transition_variant(
    *,
    transition_contract: Any,
    allowed_keys: Any,
) -> tuple[str, frozenset[str]]:
    if transition_contract not in _SUPPORTED_SUBAGENT_TRANSITION_CONTRACTS:
        raise _contract_error(
            "tool runtime plugin state transition contract is unsupported"
        )
    safe_state_keys = (
        _SAFE_SUBAGENT_STATE_KEYS
        if transition_contract == _SUBAGENT_SNAPSHOT_CONTRACT
        else _SUBAGENT_TERMINAL_HANDOFF_STATE_KEYS
    )
    if (
        type(allowed_keys) is not list
        or not allowed_keys
        or any(type(key) is not str for key in allowed_keys)
        or len(set(allowed_keys)) != len(allowed_keys)
        or set(allowed_keys) != safe_state_keys
    ):
        raise _contract_error(
            "tool runtime plugin state transition allowed keys are invalid"
        )
    return transition_contract, frozenset(allowed_keys)


__all__ = [
    "DurableToolStateTransitionDraft",
    "DurableToolStateTransitionEnvelope",
    "build_declared_tool_state_transition",
    "canonical_state_sha256",
    "load_verified_subagent_state",
    "resolve_subagent_transition_cas",
]
