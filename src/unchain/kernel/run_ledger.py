"""Production wiring for the immutable RunBundle provider-call ledger.

This module deliberately owns only runtime projection.  Provider adapters own
usage extraction, while :mod:`unchain.run_bundle` owns the closed public wire
contracts and the receipt-set reducer.
"""

from __future__ import annotations

import hashlib
import math
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..run_bundle import (
    ProviderCallIdentity,
    ProviderBillingDimensions,
    ProviderCallIds,
    ProviderCallReceipt,
    ProviderCallTiming,
    ProviderCallUsage,
    RunBundle,
    RunChild,
    RunBundleProtocolError,
    RunBundleReducer,
    RunDescriptor,
    RunIdentity,
    RunLifecycle,
    RunMetricError,
    RunMetricEvent,
    RunMetricEvidenceRef,
)
from ..run_bundle_v2 import CompactRunBundle, run_bundle_from_dict
if TYPE_CHECKING:
    from ..providers.turn_ownership import (
        ProviderTurnOwnership,
        ProviderTurnOwnershipFactory,
    )
    from ..run_bundle_ledger import RunBundleLedger
    from ..runtime.module_context import AgentRuntimeContext


_SUSPENDED_KERNEL_STATUSES = frozenset(
    {
        "awaiting_human_input",
        "awaiting_interaction",
        "max_iterations",
    }
)
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_PROVIDER_ROUTES = {
    "openai": "openai.responses.create",
    "anthropic": "anthropic.messages.stream",
    "hyperspace": "hyperspace.anthropic.messages.stream",
    "ollama": "ollama.chat",
}
_COMPACT_PROJECTION_EXTENSION = "unchain.runtime/compact_projection"
_COMPACT_PROJECTION_MODE = "compact_v1"
_COMPACT_PROJECTION_TRIGGER_BYTES = 1_572_864
_RUN_BUNDLE_CAPACITY_ERRORS = frozenset(
    {
        "run bundle exceeds the canonical byte limit",
        "run bundle exceeds the JSON node limit",
    }
)
_COMPACT_PROJECTION_UNSUPPORTED_ERROR = (
    "compact projection requires durable projection detail persistence"
)
_COMPACT_PROJECTION_DETAIL_TYPE_ERROR = (
    "compact projection details did not return a metric-event tuple"
)
_COMPACT_PROJECTION_DETAIL_COUNT_ERROR = (
    "compact projection details metric count does not match durable metadata"
)
_COMPACT_PROJECTION_DETAIL_DIGEST_ERROR = (
    "compact projection details digest does not match durable metadata"
)
_COMPACT_PROJECTION_EXTENSION_ERROR = "compact projection extension is invalid"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMPACT_PROJECTION_KEYS = frozenset(
    {
        "mode",
        "projection_hash",
        "metric_events_sha256",
        "projection_counts",
        "projection_status",
        "projection_truncation",
        "metric_kinds",
        "metric_outcomes",
    }
)
_COMPACT_PROJECTION_COUNT_KEYS = frozenset(
    {
        "provider_calls",
        "direct_provider_calls",
        "descendant_provider_calls",
        "metric_events",
        "children",
    }
)


def provider_call_route(provider: str) -> str:
    normalized = str(provider or "unknown").strip().lower() or "unknown"
    return _PROVIDER_ROUTES.get(normalized, f"{normalized}.model_turn")


def _provider_id_sha256(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise RunBundleProtocolError(
            "provider request digest source must be strict canonical JSON"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _projection_canonical_sha256(value: Any) -> str:
    """Return a deterministic projection digest without run-bundle size policy."""

    try:
        return _canonical_sha256(value)
    except RunBundleProtocolError as exc:
        raise RunBundleProtocolError(
            "run bundle projection key must use strict canonical JSON"
        ) from exc


def _is_run_bundle_canonical_limit_error(exc: BaseException) -> bool:
    if not isinstance(exc, RunBundleProtocolError):
        return False
    return str(exc) in _RUN_BUNDLE_CAPACITY_ERRORS


def _extract_compact_projection_extension(
    extensions: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if extensions is None:
        return None
    if _COMPACT_PROJECTION_EXTENSION not in extensions:
        return None
    raw = extensions[_COMPACT_PROJECTION_EXTENSION]
    if type(raw) is not dict or set(raw) != _COMPACT_PROJECTION_KEYS:
        raise RunBundleProtocolError(_COMPACT_PROJECTION_EXTENSION_ERROR)
    if raw.get("mode") != _COMPACT_PROJECTION_MODE:
        raise RunBundleProtocolError(_COMPACT_PROJECTION_EXTENSION_ERROR)
    for field_name in ("projection_hash", "metric_events_sha256"):
        if (
            type(raw.get(field_name)) is not str
            or _SHA256_RE.fullmatch(raw[field_name]) is None
        ):
            raise RunBundleProtocolError(_COMPACT_PROJECTION_EXTENSION_ERROR)
    projection_counts = raw.get("projection_counts")
    if (
        type(projection_counts) is not dict
        or set(projection_counts) != _COMPACT_PROJECTION_COUNT_KEYS
    ):
        raise RunBundleProtocolError(_COMPACT_PROJECTION_EXTENSION_ERROR)
    for value in projection_counts.values():
        if type(value) is not int or value < 0:
            raise RunBundleProtocolError(_COMPACT_PROJECTION_EXTENSION_ERROR)
    if (
        projection_counts["direct_provider_calls"]
        + projection_counts["descendant_provider_calls"]
        != projection_counts["provider_calls"]
    ):
        raise RunBundleProtocolError(_COMPACT_PROJECTION_EXTENSION_ERROR)
    if type(raw.get("projection_status")) is not str or not raw["projection_status"]:
        raise RunBundleProtocolError(_COMPACT_PROJECTION_EXTENSION_ERROR)
    if raw.get("projection_truncation") != ["metric_events"]:
        raise RunBundleProtocolError(_COMPACT_PROJECTION_EXTENSION_ERROR)
    for field_name in ("metric_kinds", "metric_outcomes"):
        counts = raw.get(field_name)
        if type(counts) is not dict:
            raise RunBundleProtocolError(_COMPACT_PROJECTION_EXTENSION_ERROR)
        for key, value in counts.items():
            if type(key) is not str or not key or type(value) is not int or value < 0:
                raise RunBundleProtocolError(_COMPACT_PROJECTION_EXTENSION_ERROR)
    if sum(raw["metric_kinds"].values()) != projection_counts["metric_events"]:
        raise RunBundleProtocolError(_COMPACT_PROJECTION_EXTENSION_ERROR)
    if sum(raw["metric_outcomes"].values()) != projection_counts["metric_events"]:
        raise RunBundleProtocolError(_COMPACT_PROJECTION_EXTENSION_ERROR)
    return raw


def _compact_projection_extensions(
    *,
    kernel_status: str,
    projection_hash: str,
    receipts: tuple[Any, ...],
    metric_events: tuple[Any, ...],
    children: tuple[Any, ...],
    identity: "RunIdentity",
    metric_events_sha256: str,
    projection_truncation: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build safe bounded runtime extension metadata for compact projection."""

    if (
        _SHA256_RE.fullmatch(projection_hash) is None
        or _SHA256_RE.fullmatch(metric_events_sha256) is None
    ):
        raise RunBundleProtocolError(_COMPACT_PROJECTION_EXTENSION_ERROR)
    direct_receipt_count = 0
    descendant_receipt_count = 0
    root_run_id = identity.run_id
    for receipt in receipts:
        owner_run_id = getattr(
            getattr(receipt, "identity", None),
            "owner_run_id",
            None,
        )
        if owner_run_id == root_run_id:
            direct_receipt_count += 1
        else:
            descendant_receipt_count += 1
    event_kind_counts: dict[str, int] = {}
    event_outcome_counts: dict[str, int] = {}
    for event in metric_events:
        kind = getattr(event, "kind", "unknown")
        outcome = getattr(event, "outcome", "unknown")
        event_kind_counts[kind] = event_kind_counts.get(kind, 0) + 1
        event_outcome_counts[outcome] = event_outcome_counts.get(outcome, 0) + 1
    return {
        "unchain.runtime/compact_projection": {
            "mode": _COMPACT_PROJECTION_MODE,
            "projection_hash": projection_hash,
            "metric_events_sha256": metric_events_sha256,
            "projection_counts": {
                "provider_calls": len(receipts),
                "direct_provider_calls": direct_receipt_count,
                "descendant_provider_calls": descendant_receipt_count,
                "metric_events": len(metric_events),
                "children": len(children),
            },
            "projection_status": (
                str(kernel_status or "unknown").strip().lower() or "unknown"
            ),
            "projection_truncation": list(projection_truncation),
            "metric_kinds": event_kind_counts,
            "metric_outcomes": event_outcome_counts,
        }
    }


def _compact_projection_metric_event_count(extensions: Mapping[str, Any] | None) -> int:
    extension = _extract_compact_projection_extension(extensions)
    if extension is None:
        return 0
    return extension["projection_counts"]["metric_events"]


def _ordered_metric_events(
    metric_events: tuple[RunMetricEvent, ...],
) -> tuple[RunMetricEvent, ...]:
    return tuple(sorted(metric_events, key=lambda item: item.metric_event_id))


def _compact_projection_metric_events_sha256(
    metric_events: tuple[RunMetricEvent, ...],
) -> str:
    return _projection_canonical_sha256(
        [event.to_dict() for event in _ordered_metric_events(metric_events)]
    )


def _merged_projection_inputs(
    *,
    identity: RunIdentity,
    receipts: tuple[ProviderCallReceipt, ...],
    metric_events: tuple[RunMetricEvent, ...],
    children: tuple[RunChild, ...],
    child_bundles: tuple[RunBundle | CompactRunBundle, ...],
    child_facts: Mapping[
        str,
        tuple[
            tuple[ProviderCallReceipt, ...],
            tuple[RunMetricEvent, ...],
            tuple[RunChild, ...],
        ],
    ],
) -> tuple[
    tuple[ProviderCallReceipt, ...],
    tuple[RunMetricEvent, ...],
    tuple[RunChild, ...],
]:
    receipt_by_id: dict[str, ProviderCallReceipt] = {}
    metric_event_by_id: dict[str, RunMetricEvent] = {}
    child_by_id: dict[str, RunChild] = {}

    def add_receipt(receipt: ProviderCallReceipt) -> None:
        if type(receipt) is not ProviderCallReceipt:
            raise TypeError("receipts must contain exact ProviderCallReceipt values")
        prior = receipt_by_id.get(receipt.provider_call_id)
        if prior is not None and prior.receipt_sha256 != receipt.receipt_sha256:
            raise RunBundleProtocolError(
                "one provider_call_id has conflicting immutable receipts"
            )
        receipt_by_id[receipt.provider_call_id] = receipt

    def add_metric_event(event: RunMetricEvent) -> None:
        if type(event) is not RunMetricEvent:
            raise TypeError("metric_events must contain exact RunMetricEvent values")
        prior = metric_event_by_id.get(event.metric_event_id)
        if prior is not None and prior != event:
            raise RunBundleProtocolError(
                "one metric_event_id has conflicting immutable events"
            )
        metric_event_by_id[event.metric_event_id] = event

    def add_child(child: RunChild) -> None:
        if type(child) is not RunChild:
            raise TypeError("children must contain exact RunChild values")
        prior = child_by_id.get(child.run_id)
        if prior is not None and prior != child:
            raise RunBundleProtocolError("one child run id has conflicting topology")
        child_by_id[child.run_id] = child

    for receipt in receipts:
        add_receipt(receipt)
    for event in metric_events:
        add_metric_event(event)
    for child in children:
        add_child(child)
    for bundle in child_bundles:
        if type(bundle) not in {RunBundle, CompactRunBundle}:
            raise TypeError("bundles must contain exact v1 or v2 RunBundle values")
        if (
            bundle.identity.execution_id != identity.execution_id
            or bundle.identity.root_run_id != identity.root_run_id
        ):
            raise RunBundleProtocolError(
                "child bundle identity crosses the target run boundary"
            )
        facts = child_facts.get(bundle.bundle_id)
        if facts is None:
            raise RunBundleProtocolError("child bundle durable facts are unavailable")
        child_receipts, child_events, nested_children = facts
        for receipt in child_receipts:
            add_receipt(receipt)
        for child in nested_children:
            add_child(child)
        for event in child_events:
            add_metric_event(event)
        if bundle.identity.run_id != identity.run_id:
            if bundle.identity.parent_run_id is None:
                raise RunBundleProtocolError(
                    "child bundle identity requires a parent run"
                )
            add_child(
                RunChild(
                    run_id=bundle.identity.run_id,
                    attempt_id=bundle.identity.attempt_id,
                    parent_run_id=bundle.identity.parent_run_id,
                    relation=bundle.identity.relation,
                    bundle_id=bundle.bundle_id,
                    status=bundle.lifecycle.status,
                )
            )

    return (
        tuple(receipt_by_id[call_id] for call_id in sorted(receipt_by_id)),
        tuple(
            metric_event_by_id[event_id]
            for event_id in sorted(metric_event_by_id)
        ),
        tuple(child_by_id[run_id] for run_id in sorted(child_by_id)),
    )


def _stable_json_value(value: Any, *, depth: int = 0) -> Any:
    """Return bounded deterministic request material suitable only for hashing."""

    if depth > 32:
        return {"type": "depth_limit"}
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"float": str(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _stable_json_value(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _stable_json_value(to_dict(), depth=depth + 1)
        except Exception:
            pass
    value_type = type(value)
    return {"type": f"{value_type.__module__}.{value_type.__qualname__}"}


def request_sha256(
    *,
    state: Any,
    payload: Mapping[str, Any] | None,
    toolkit: Any,
    response_format: Any,
    openai_text_format: Mapping[str, Any] | None,
    provider: str,
    model: str,
    messages: Any = None,
) -> str:
    """Hash the logical provider request without retaining prompt or payload data."""

    try:
        from ..kernel.provider_replay import tool_schema_digest

        schema_digest = tool_schema_digest(toolkit, provider)
    except Exception:
        tools = getattr(toolkit, "tools", {})
        schema_digest = _canonical_sha256(
            sorted(str(name) for name in tools) if isinstance(tools, Mapping) else []
        )
    request_messages = messages
    if request_messages is None:
        request_messages = (
            state.view_messages()
            if callable(getattr(state, "view_messages", None))
            else getattr(state, "transcript", [])
        )
    material = {
        "provider": provider,
        "model": model,
        "messages": _stable_json_value(request_messages),
        "payload": _stable_json_value(dict(payload or {})),
        "response_format": _stable_json_value(response_format),
        "openai_text_format": _stable_json_value(dict(openai_text_format or {})),
        "previous_response_id": getattr(
            getattr(state, "provider_state", None),
            "previous_response_id",
            None,
        ),
        "tool_schema_sha256": schema_digest,
    }
    return _canonical_sha256(material)


def _identity_from_runtime_context(
    runtime_context: "AgentRuntimeContext",
) -> RunIdentity:
    identity = runtime_context.identity
    return RunIdentity(
        execution_id=identity.execution_id,
        attempt_id=identity.attempt_id,
        root_run_id=identity.root_run_id,
        run_id=identity.run_id,
        parent_run_id=identity.parent_run_id,
        relation="root" if identity.parent_run_id is None else "subagent",
    )


def _same_identity_coordinates(left: RunIdentity, right: RunIdentity) -> bool:
    """Compare every identity coordinate except its topology relation."""

    return (
        left.execution_id == right.execution_id
        and left.attempt_id == right.attempt_id
        and left.root_run_id == right.root_run_id
        and left.run_id == right.run_id
        and left.parent_run_id == right.parent_run_id
    )


def _synthetic_root_identity(state: Any, *, run_id: str) -> RunIdentity:
    execution_id = str(
        getattr(getattr(state, "session_state", None), "session_id", None)
        or run_id
    )
    return RunIdentity(
        execution_id=execution_id,
        attempt_id=run_id,
        root_run_id=run_id,
        run_id=run_id,
        parent_run_id=None,
        relation="root",
    )


def child_run_identity(
    *,
    parent: RunIdentity,
    child_run_id: str,
    child_attempt_id: str,
    relation: str = "subagent",
) -> RunIdentity:
    if type(parent) is not RunIdentity:
        raise TypeError("parent must be an exact RunIdentity")
    return RunIdentity(
        execution_id=parent.execution_id,
        attempt_id=child_attempt_id,
        root_run_id=parent.root_run_id,
        run_id=child_run_id,
        parent_run_id=parent.run_id,
        relation=relation,
    )


def _lifecycle_status(kernel_status: str) -> str:
    normalized = str(kernel_status or "").strip().lower()
    if normalized in _SUSPENDED_KERNEL_STATUSES:
        return "suspended"
    if normalized in {"completed", "failed", "cancelled"}:
        return normalized
    if normalized in {"running", "idle"}:
        return "running"
    return "uncertain"


def _now_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@dataclass
class RunLedger:
    """Attempt-local, exact-once collection of immutable provider receipts."""

    identity: RunIdentity | None = None
    identity_source: str = "legacy_synthetic"
    receipts: dict[str, ProviderCallReceipt] = field(default_factory=dict)
    metric_events: dict[str, RunMetricEvent] = field(default_factory=dict)
    descriptor: RunDescriptor = field(default_factory=RunDescriptor)
    started_at: str | None = None
    continued_from_run_id: str | None = None
    continuation_verified: bool = True
    persistence: "RunBundleLedger | None" = field(default=None, repr=False)
    provider_turn_ownership: "ProviderTurnOwnership | None" = field(
        default=None,
        repr=False,
    )
    provider_turn_ownership_factory: "ProviderTurnOwnershipFactory | None" = field(
        default=None,
        repr=False,
    )
    _seed_bundles: dict[str, RunBundle] = field(default_factory=dict, repr=False)
    _seed_children: dict[str, RunChild] = field(default_factory=dict, repr=False)
    _revision: int = 0
    _last_projection_key: str | None = None
    _last_bundle: RunBundle | None = None
    _last_compact_bundle: CompactRunBundle | None = None

    def __deepcopy__(self, memo: dict[int, Any]) -> "RunLedger":
        """Copy mutable indexes without copying immutable/capability values."""

        existing = memo.get(id(self))
        if existing is not None:
            return existing
        copied = RunLedger(
            identity=self.identity,
            identity_source=self.identity_source,
            receipts=dict(self.receipts),
            metric_events=dict(self.metric_events),
            descriptor=self.descriptor,
            started_at=self.started_at,
            continued_from_run_id=self.continued_from_run_id,
            continuation_verified=self.continuation_verified,
            persistence=self.persistence,
            provider_turn_ownership=self.provider_turn_ownership,
            provider_turn_ownership_factory=self.provider_turn_ownership_factory,
            _seed_bundles=dict(self._seed_bundles),
            _seed_children=dict(self._seed_children),
            _revision=self._revision,
            _last_projection_key=self._last_projection_key,
            _last_bundle=self._last_bundle,
            _last_compact_bundle=self._last_compact_bundle,
        )
        memo[id(self)] = copied
        return copied

    def bind_provider_turn_ownership(
        self,
        ownership: "ProviderTurnOwnership | None",
    ) -> None:
        if ownership is None:
            return
        from ..providers.turn_ownership import ProviderTurnOwnership

        if type(ownership) is not ProviderTurnOwnership:
            raise TypeError(
                "ownership must be an exact ProviderTurnOwnership or null"
            )
        if self.identity is None or ownership.identity != self.identity:
            raise RunBundleProtocolError(
                "provider turn ownership does not match the active run identity"
            )
        if (
            self.provider_turn_ownership is not None
            and self.provider_turn_ownership is not ownership
        ):
            raise RunBundleProtocolError(
                "provider turn ownership changed after run initialization"
            )
        self.attach_persistence(ownership.ledger)
        self.provider_turn_ownership = ownership
        self.provider_turn_ownership_factory = ownership.factory

    def initialize(
        self,
        *,
        state: Any,
        run_id: str,
        runtime_context: "AgentRuntimeContext | None" = None,
        explicit_identity: RunIdentity | None = None,
        descriptor: RunDescriptor | None = None,
        continued_from_run_id: str | None = None,
    ) -> None:
        if self.identity is not None:
            expected = self.identity
        elif explicit_identity is not None:
            if type(explicit_identity) is not RunIdentity:
                raise TypeError("explicit run bundle identity must be an exact RunIdentity")
            expected = explicit_identity
            self.identity_source = "explicit"
        elif runtime_context is not None:
            from ..runtime.module_context import AgentRuntimeContext

            if not isinstance(runtime_context, AgentRuntimeContext):
                raise TypeError("runtime_context must be an AgentRuntimeContext")
            expected = _identity_from_runtime_context(runtime_context)
            self.identity_source = "runtime_context"
        else:
            expected = _synthetic_root_identity(state, run_id=run_id)
            self.identity_source = "legacy_synthetic"
        if expected.run_id != run_id:
            raise RunBundleProtocolError(
                "run bundle identity does not match the kernel run_id"
            )
        if runtime_context is not None:
            runtime_identity = _identity_from_runtime_context(runtime_context)
            if not _same_identity_coordinates(runtime_identity, expected):
                raise RunBundleProtocolError(
                    "explicit run bundle identity disagrees with runtime_context"
                )
        self.identity = expected
        if descriptor is not None:
            if type(descriptor) is not RunDescriptor:
                raise TypeError("descriptor must be an exact RunDescriptor")
            if self.descriptor != RunDescriptor() and self.descriptor != descriptor:
                raise RunBundleProtocolError(
                    "run descriptor changed after ledger initialization"
                )
            self.descriptor = descriptor
        if continued_from_run_id is not None:
            candidate = str(continued_from_run_id).strip()
            if not candidate:
                raise RunBundleProtocolError(
                    "continued_from_run_id must be non-empty exact text"
                )
            if (
                self.continued_from_run_id is not None
                and self.continued_from_run_id != candidate
            ):
                raise RunBundleProtocolError(
                    "continued_from_run_id changed after ledger initialization"
                )
            self.continued_from_run_id = candidate
            self.continuation_verified = False
        if self.started_at is None:
            self.started_at = _now_timestamp()
        if self.persistence is not None:
            self.attach_persistence(self.persistence)

    def _load_projection_events_from_durable_bundle(
        self,
        *,
        bundle: "RunBundle",
    ) -> tuple[RunMetricEvent, ...]:
        extension = _extract_compact_projection_extension(bundle.extensions)
        if extension is None:
            return tuple(bundle.metrics.events)
        projection_hash = extension["projection_hash"]
        expected_metric_events_sha256 = extension["metric_events_sha256"]
        expected_count = _compact_projection_metric_event_count(bundle.extensions)
        if expected_count == 0:
            if expected_metric_events_sha256 != _compact_projection_metric_events_sha256(
                ()
            ):
                raise RunBundleProtocolError(_COMPACT_PROJECTION_DETAIL_DIGEST_ERROR)
            return ()
        if self.persistence is None:
            raise RunBundleProtocolError(_COMPACT_PROJECTION_UNSUPPORTED_ERROR)
        from ..run_bundle_ledger import RunBundleProjectionDetailsLedger

        if not isinstance(self.persistence, RunBundleProjectionDetailsLedger):
            raise RunBundleProtocolError(_COMPACT_PROJECTION_UNSUPPORTED_ERROR)
        loaded = self.persistence.load_projection_details(
            bundle_id=bundle.bundle_id,
            revision=bundle.revision,
            projection_hash=projection_hash,
            metric_events_sha256=expected_metric_events_sha256,
        )
        if not isinstance(loaded, tuple):
            raise RunBundleProtocolError(_COMPACT_PROJECTION_DETAIL_TYPE_ERROR)
        if len(loaded) != expected_count:
            raise RunBundleProtocolError(_COMPACT_PROJECTION_DETAIL_COUNT_ERROR)
        if (
            _compact_projection_metric_events_sha256(loaded)
            != expected_metric_events_sha256
        ):
            raise RunBundleProtocolError(_COMPACT_PROJECTION_DETAIL_DIGEST_ERROR)
        valid_owner_run_ids = {bundle.identity.run_id}
        valid_owner_run_ids.update(child.run_id for child in bundle.children)
        for event in loaded:
            if (
                event.execution_id != bundle.identity.execution_id
                or event.root_run_id != bundle.identity.root_run_id
                or event.owner_run_id not in valid_owner_run_ids
            ):
                raise RunBundleProtocolError(
                    "compact projection details cross the bundle boundary"
                )
        return loaded

    def _load_projection_facts_from_durable_bundle(
        self,
        *,
        bundle: RunBundle | CompactRunBundle,
    ) -> tuple[
        tuple[ProviderCallReceipt, ...],
        tuple[RunMetricEvent, ...],
        tuple[RunChild, ...],
    ]:
        if type(bundle) is RunBundle:
            return (
                tuple(bundle.provider_calls),
                self._load_projection_events_from_durable_bundle(bundle=bundle),
                tuple(bundle.children),
            )
        if type(bundle) is not CompactRunBundle:
            raise TypeError("bundle must be an exact v1 or v2 RunBundle")
        if self.persistence is None:
            raise RunBundleProtocolError(
                "compact run bundle requires durable details persistence"
            )
        from ..run_bundle_ledger import RunBundleCompactDetailsLedger

        if not isinstance(self.persistence, RunBundleCompactDetailsLedger):
            raise RunBundleProtocolError(
                "compact run bundle requires durable details persistence"
            )
        receipts, events, children = self.persistence.load_compact_bundle_details(
            bundle=bundle,
        )
        valid_owner_run_ids = {bundle.identity.run_id}
        valid_owner_run_ids.update(child.run_id for child in children)
        for receipt in receipts:
            if (
                receipt.identity.execution_id != bundle.identity.execution_id
                or receipt.identity.root_run_id != bundle.identity.root_run_id
                or receipt.identity.owner_run_id not in valid_owner_run_ids
            ):
                raise RunBundleProtocolError(
                    "compact run bundle receipt crosses the bundle boundary"
                )
        for event in events:
            if (
                event.execution_id != bundle.identity.execution_id
                or event.root_run_id != bundle.identity.root_run_id
                or event.owner_run_id not in valid_owner_run_ids
            ):
                raise RunBundleProtocolError(
                    "compact run bundle metric event crosses the bundle boundary"
                )
        return receipts, events, children

    def attach_persistence(self, persistence: "RunBundleLedger") -> None:
        from ..run_bundle_ledger import RunBundleLedger

        if not isinstance(persistence, RunBundleLedger):
            raise TypeError("persistence must implement the RunBundleLedger protocol")
        if self.identity is None:
            self.persistence = persistence
            return
        if persistence.execution_id != self.identity.execution_id:
            raise RunBundleProtocolError(
                "durable run bundle ledger belongs to another execution"
            )
        if (
            self.persistence is not None
            and self.persistence is not persistence
            and self.persistence.execution_id != persistence.execution_id
        ):
            raise RunBundleProtocolError(
                "run ledger persistence changed its execution binding"
            )
        self.persistence = persistence
        for receipt in tuple(self.receipts.values()):
            durable = persistence.append_receipt(receipt)
            if durable != receipt:
                raise RunBundleProtocolError(
                    "durable provider receipt changed immutable accounting"
                )
        durable_receipts = persistence.load_receipts(
            root_run_id=self.identity.root_run_id,
            owner_run_id=self.identity.run_id,
            attempt_id=self.identity.attempt_id,
        )
        for receipt in durable_receipts:
            prior = self.receipts.get(receipt.provider_call_id)
            if prior is not None and prior.receipt_sha256 != receipt.receipt_sha256:
                raise RunBundleProtocolError(
                    "in-memory provider receipt conflicts with durable accounting"
                )
            self.receipts[receipt.provider_call_id] = receipt
        durable_v1_bundles = persistence.list_bundles(
            root_run_id=self.identity.root_run_id,
            run_id=self.identity.run_id,
            attempt_id=self.identity.attempt_id,
        )
        from ..run_bundle_ledger import RunBundleCompactDetailsLedger

        durable_compact_bundles = (
            persistence.list_compact_bundles(
                root_run_id=self.identity.root_run_id,
                run_id=self.identity.run_id,
                attempt_id=self.identity.attempt_id,
            )
            if isinstance(persistence, RunBundleCompactDetailsLedger)
            else ()
        )
        durable_bundles: tuple[RunBundle | CompactRunBundle, ...] = (
            *durable_v1_bundles,
            *durable_compact_bundles,
        )
        if len({bundle.bundle_id for bundle in durable_bundles}) > 1:
            raise RunBundleProtocolError(
                "one run identity resolved to multiple durable bundle IDs"
            )
        if durable_bundles:
            head_revision = max(bundle.revision for bundle in durable_bundles)
            durable_heads = tuple(
                bundle
                for bundle in durable_bundles
                if bundle.revision == head_revision
            )
            if len(durable_heads) != 1:
                raise RunBundleProtocolError(
                    "one run revision resolved to conflicting durable schemas"
                )
            durable = durable_heads[0]
            if durable.identity != self.identity:
                raise RunBundleProtocolError(
                    "durable bundle identity disagrees with the active run"
                )
            self._revision = max(self._revision, durable.revision)
            if type(durable) is RunBundle:
                self._seed_bundles[durable.bundle_id] = durable
                self._last_bundle = durable
            else:
                self._last_compact_bundle = durable
            self.started_at = durable.lifecycle.started_at or self.started_at
            if (
                self.continued_from_run_id is not None
                and durable.lifecycle.continued_from_run_id is not None
                and self.continued_from_run_id
                != durable.lifecycle.continued_from_run_id
            ):
                raise RunBundleProtocolError(
                    "continued_from_run_id conflicts with durable lifecycle"
                )
            self.continued_from_run_id = (
                durable.lifecycle.continued_from_run_id
                or self.continued_from_run_id
            )
            if self.descriptor == RunDescriptor():
                self.descriptor = durable.descriptor
            durable_receipts, durable_events, durable_children = (
                self._load_projection_facts_from_durable_bundle(
                    bundle=durable,
                )
            )
            for receipt in durable_receipts:
                prior_receipt = self.receipts.get(receipt.provider_call_id)
                if (
                    prior_receipt is not None
                    and prior_receipt.receipt_sha256 != receipt.receipt_sha256
                ):
                    raise RunBundleProtocolError(
                        "durable provider receipt conflicts with in-memory accounting"
                    )
                self.receipts[receipt.provider_call_id] = receipt
            for event in durable_events:
                prior_event = self.metric_events.get(event.metric_event_id)
                if prior_event is not None and prior_event != event:
                    raise RunBundleProtocolError(
                        "durable metric event conflicts with in-memory accounting"
                    )
                self.metric_events[event.metric_event_id] = event
            for child in durable_children:
                prior_child = self._seed_children.get(child.run_id)
                if prior_child is not None and prior_child != child:
                    raise RunBundleProtocolError(
                        "durable child topology conflicts with in-memory accounting"
                    )
                self._seed_children[child.run_id] = child
        from ..run_bundle_ledger import (
            RunBundleContinuationError,
            RunBundleContinuationLedger,
        )

        if isinstance(persistence, RunBundleContinuationLedger):
            predecessor = persistence.claim_continuation(
                successor=self.identity,
                requested_run_id=self.continued_from_run_id,
            )
            claimed_run_id = (
                predecessor.identity.run_id if predecessor is not None else None
            )
            if (
                self.continued_from_run_id is not None
                and claimed_run_id != self.continued_from_run_id
            ):
                raise RunBundleContinuationError(
                    "continued_from_not_claimable"
                )
            if claimed_run_id is not None:
                self.continued_from_run_id = claimed_run_id
                self.continuation_verified = True
            elif self.continued_from_run_id is None:
                self.continuation_verified = True
        elif self.continued_from_run_id is not None:
            raise RunBundleContinuationError(
                "continuation_ledger_unavailable"
            )
        else:
            self.continuation_verified = True

    def assert_continuation_verified(self) -> None:
        if self.continued_from_run_id is None or self.continuation_verified:
            return
        from ..run_bundle_ledger import RunBundleContinuationError

        raise RunBundleContinuationError("continuation_ledger_unavailable")

    def append(self, receipt: ProviderCallReceipt) -> None:
        if type(receipt) is not ProviderCallReceipt:
            raise TypeError("run ledger accepts only exact ProviderCallReceipt values")
        if self.identity is None:
            raise RunBundleProtocolError(
                "run ledger identity must be initialized before receipt capture"
            )
        call_identity = receipt.identity
        if (
            call_identity.execution_id != self.identity.execution_id
            or call_identity.attempt_id != self.identity.attempt_id
            or call_identity.root_run_id != self.identity.root_run_id
            or call_identity.owner_run_id != self.identity.run_id
            or call_identity.parent_run_id != self.identity.parent_run_id
        ):
            raise RunBundleProtocolError(
                "provider receipt does not belong to the active run identity"
            )
        prior = self.receipts.get(receipt.provider_call_id)
        if prior is not None and prior.receipt_sha256 != receipt.receipt_sha256:
            raise RunBundleProtocolError(
                "one provider_call_id has conflicting immutable receipts"
            )
        if self.persistence is not None:
            durable = self.persistence.append_receipt(receipt)
            if durable != receipt:
                raise RunBundleProtocolError(
                    "durable provider receipt changed immutable accounting"
                )
        self.receipts[receipt.provider_call_id] = receipt

    def append_metric_event(self, event: RunMetricEvent) -> None:
        if type(event) is not RunMetricEvent:
            raise TypeError("run ledger accepts only exact RunMetricEvent values")
        if self.identity is None:
            raise RunBundleProtocolError(
                "run ledger identity must be initialized before metric capture"
            )
        if (
            event.execution_id != self.identity.execution_id
            or event.attempt_id != self.identity.attempt_id
            or event.root_run_id != self.identity.root_run_id
            or event.owner_run_id != self.identity.run_id
            or event.parent_run_id != self.identity.parent_run_id
        ):
            raise RunBundleProtocolError(
                "metric event does not belong to the active run identity"
            )
        prior = self.metric_events.get(event.metric_event_id)
        if prior is not None and prior != event:
            raise RunBundleProtocolError(
                "one metric_event_id has conflicting immutable events"
            )
        self.metric_events[event.metric_event_id] = event

    def record_metric_event(
        self,
        *,
        kind: str,
        subject_id: str,
        outcome: str = "completed",
        error_category: str | None = None,
        error_code: str | None = None,
        evidence_refs: tuple[RunMetricEvidenceRef, ...] = (),
    ) -> RunMetricEvent:
        if self.identity is None:
            raise RunBundleProtocolError(
                "run ledger identity must be initialized before metric capture"
            )
        error = None
        if error_category is not None or error_code is not None:
            if error_category is None or error_code is None:
                raise RunBundleProtocolError(
                    "metric error requires both category and code"
                )
            error = RunMetricError(category=error_category, code=error_code)
        event = RunMetricEvent(
            execution_id=self.identity.execution_id,
            attempt_id=self.identity.attempt_id,
            root_run_id=self.identity.root_run_id,
            owner_run_id=self.identity.run_id,
            parent_run_id=self.identity.parent_run_id,
            kind=kind,
            subject_id=subject_id,
            outcome=outcome,
            error=error,
            evidence_refs=evidence_refs,
        )
        self.append_metric_event(event)
        return event

    def record_iteration_outcome(
        self,
        *,
        iteration: int,
        outcome: str,
        error_category: str | None = None,
        error_code: str | None = None,
    ) -> RunMetricEvent:
        """Seal one logical iteration exactly once.

        Suspended iterations are immutable uncertain facts.  If that same
        run is later resumed, the original seal remains authoritative instead
        of rewriting the event or double-counting the logical iteration.
        """

        subject_id = f"iteration:{max(0, int(iteration))}"
        for event in self.metric_events.values():
            if event.kind == "iteration" and event.subject_id == subject_id:
                return event
        return self.record_metric_event(
            kind="iteration",
            subject_id=subject_id,
            outcome=outcome,
            error_category=error_category,
            error_code=error_code,
        )

    def materialize(
        self,
        *,
        kernel_status: str,
        child_bundle_values: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> RunBundle | CompactRunBundle:
        if self.identity is None:
            raise RunBundleProtocolError("run ledger identity is not initialized")
        self.assert_continuation_verified()
        projected_children = tuple(
            run_bundle_from_dict(dict(value))
            for _key, value in sorted((child_bundle_values or {}).items())
        )
        child_bundles_by_id = {
            bundle_id: bundle
            for bundle_id, bundle in self._seed_bundles.items()
            if bundle.identity != self.identity
        }
        for bundle in projected_children:
            if bundle.identity != self.identity:
                child_bundles_by_id[bundle.bundle_id] = bundle
        child_bundles = tuple(child_bundles_by_id.values())
        lifecycle_status = _lifecycle_status(kernel_status)
        sorted_metric_events = tuple(
            sorted(self.metric_events.values(), key=lambda item: item.metric_event_id)
        )
        sorted_receipts = tuple(
            sorted(self.receipts.values(), key=lambda item: item.provider_call_id)
        )
        sorted_child_bundles = tuple(
            sorted(child_bundles, key=lambda item: item.bundle_id)
        )
        child_facts = {
            bundle.bundle_id: self._load_projection_facts_from_durable_bundle(
                bundle=bundle,
            )
            for bundle in sorted_child_bundles
        }
        (
            projection_receipts,
            projection_metric_events,
            projection_children,
        ) = _merged_projection_inputs(
            identity=self.identity,
            receipts=sorted_receipts,
            metric_events=sorted_metric_events,
            children=tuple(self._seed_children.values()),
            child_bundles=sorted_child_bundles,
            child_facts=child_facts,
        )
        completed_at = None
        if lifecycle_status != "running":
            last_durable_bundle = self._last_bundle or self._last_compact_bundle
            if (
                last_durable_bundle is not None
                and last_durable_bundle.lifecycle.status == lifecycle_status
            ):
                completed_at = last_durable_bundle.lifecycle.completed_at
            else:
                completed_at = _now_timestamp()
        projection_key = _projection_canonical_sha256(
            {
                "status": kernel_status,
                "started_at": self.started_at,
                "completed_at": completed_at,
                "continued_from_run_id": self.continued_from_run_id,
                "descriptor": self.descriptor.to_dict(),
                "metric_events": [
                    [
                        event.metric_event_id,
                        _projection_canonical_sha256(event.to_dict()),
                    ]
                    for event in projection_metric_events
                ],
                "receipts": [
                    [receipt.provider_call_id, receipt.receipt_sha256]
                    for receipt in projection_receipts
                ],
                "children": [
                    [
                        child.run_id,
                        _projection_canonical_sha256(child.to_dict()),
                    ]
                    for child in projection_children
                ],
            }
        )
        if (
            self._last_projection_key == projection_key
            and self._last_bundle is not None
        ):
            return self._last_bundle
        if (
            self._last_projection_key == projection_key
            and self._last_compact_bundle is not None
        ):
            return self._last_compact_bundle
        lifecycle = RunLifecycle(
            status=lifecycle_status,
            started_at=self.started_at,
            completed_at=completed_at,
            continued_from_run_id=self.continued_from_run_id,
        )
        last_durable_bundle = self._last_bundle or self._last_compact_bundle
        projection_extensions = dict(
            last_durable_bundle.extensions
            if last_durable_bundle is not None
            else {"unchain.runtime/identity_source": self.identity_source}
        )
        projection_extensions.pop(_COMPACT_PROJECTION_EXTENSION, None)
        projection_extensions["unchain.runtime/kernel_status"] = str(
            kernel_status or "unknown"
        )

        def compact_projection() -> CompactRunBundle:
            from ..run_bundle_ledger import RunBundleCompactDetailsLedger

            if self.persistence is None or not isinstance(
                self.persistence, RunBundleCompactDetailsLedger
            ):
                raise RunBundleProtocolError(
                    "compact run bundle requires durable details persistence"
                )
            compact_bundle, details = CompactRunBundle.from_facts(
                identity=self.identity,
                lifecycle=lifecycle,
                descriptor=self.descriptor,
                revision=(self._revision if self._revision > 0 else 1),
                receipts=projection_receipts,
                metric_events=projection_metric_events,
                children=projection_children,
                extensions=projection_extensions,
            )
            durable = self.persistence.persist_compact_bundle_with_details(
                bundle=compact_bundle,
                details=details,
            )
            if durable != compact_bundle:
                raise RunBundleProtocolError(
                    "durable compact run bundle changed its immutable projection"
                )
            self._last_compact_bundle = compact_bundle
            return compact_bundle

        def project(
            revision: int,
        ) -> tuple[RunBundle, tuple[RunMetricEvent, ...] | None]:
            candidate: RunBundle | None = None
            try:
                candidate = RunBundleReducer.reduce(
                    identity=self.identity,
                    lifecycle=lifecycle,
                    receipts=projection_receipts,
                    children=projection_children,
                    descriptor=self.descriptor,
                    metric_events=projection_metric_events,
                    revision=revision,
                    extensions=projection_extensions,
                )
            except RunBundleProtocolError as exc:
                if not _is_run_bundle_canonical_limit_error(exc):
                    raise
            if candidate is not None and (
                not projection_metric_events
                or len(candidate.canonical_bytes()) < _COMPACT_PROJECTION_TRIGGER_BYTES
            ):
                return candidate, None
            if candidate is None and not projection_metric_events:
                return compact_projection(), None
            compact_extensions = {
                **dict(projection_extensions),
                **_compact_projection_extensions(
                    kernel_status=kernel_status,
                    projection_hash=projection_key,
                    receipts=projection_receipts,
                    metric_events=projection_metric_events,
                    children=projection_children,
                    identity=self.identity,
                    metric_events_sha256=(
                        _compact_projection_metric_events_sha256(
                            projection_metric_events
                        )
                    ),
                    projection_truncation=("metric_events",),
                ),
            }
            try:
                return (
                    RunBundleReducer.reduce(
                        identity=self.identity,
                        lifecycle=lifecycle,
                        receipts=projection_receipts,
                        children=projection_children,
                        descriptor=self.descriptor,
                        metric_events=(),
                        revision=revision,
                        extensions=compact_extensions,
                    ),
                    projection_metric_events,
                )
            except RunBundleProtocolError as fallback_exc:
                if not _is_run_bundle_canonical_limit_error(fallback_exc):
                    raise
                return compact_projection(), None

        head_revision = self._revision if self._revision > 0 else 1
        candidate, candidate_metric_events = project(head_revision)
        if isinstance(candidate, CompactRunBundle):
            self._last_projection_key = projection_key
            return candidate
        if self._last_bundle is not None and candidate == self._last_bundle:
            self._last_projection_key = projection_key
            return self._last_bundle
        self._revision = 1 if self._revision == 0 else self._revision + 1
        if candidate.revision == self._revision:
            bundle = candidate
            bundle_metric_events = candidate_metric_events
        else:
            bundle, bundle_metric_events = project(self._revision)
        if isinstance(bundle, CompactRunBundle):
            self._last_projection_key = projection_key
            return bundle
        if self.persistence is None:
            if bundle_metric_events is not None:
                raise RunBundleProtocolError(_COMPACT_PROJECTION_UNSUPPORTED_ERROR)
            durable = bundle
        elif bundle_metric_events is not None:
            from ..run_bundle_ledger import RunBundleProjectionDetailsLedger

            if not isinstance(self.persistence, RunBundleProjectionDetailsLedger):
                raise RunBundleProtocolError(_COMPACT_PROJECTION_UNSUPPORTED_ERROR)
            durable = self.persistence.persist_bundle_with_projection_details(
                bundle=bundle,
                projection_hash=projection_key,
                projection_metric_events=bundle_metric_events,
            )
        else:
            durable = self.persistence.persist_bundle(bundle)
        if durable != bundle:
            raise RunBundleProtocolError(
                "durable run bundle changed its immutable projection"
            )
        self._last_projection_key = projection_key
        self._last_bundle = bundle
        return bundle


def initialize_run_ledger(
    state: Any,
    *,
    run_id: str,
    runtime_context: "AgentRuntimeContext | None" = None,
    explicit_identity: RunIdentity | None = None,
    descriptor: RunDescriptor | None = None,
    continued_from_run_id: str | None = None,
) -> RunLedger:
    ledger = getattr(state, "run_ledger", None)
    if type(ledger) is not RunLedger:
        raise TypeError("RunState.run_ledger must be an exact RunLedger")
    ledger.initialize(
        state=state,
        run_id=run_id,
        runtime_context=runtime_context,
        explicit_identity=explicit_identity,
        descriptor=descriptor,
        continued_from_run_id=continued_from_run_id,
    )
    return ledger


def attach_state_run_bundle_ledger(
    state: Any,
    persistence: "RunBundleLedger | None",
) -> None:
    if persistence is None:
        return
    ledger = getattr(state, "run_ledger", None)
    if type(ledger) is not RunLedger:
        raise TypeError("RunState.run_ledger must be an exact RunLedger")
    ledger.attach_persistence(persistence)


def build_model_attempt_receipt(
    *,
    identity: RunIdentity,
    provider: str,
    model: str,
    iteration: int,
    retry_ordinal: int,
    purpose: str,
    request_digest: str,
    route: str,
    payload: Mapping[str, Any] | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    turn: Any = None,
    status: str = "completed",
    classification: str | None = None,
) -> ProviderCallReceipt:
    """Build one immutable receipt without requiring a mutable RunState."""

    if type(identity) is not RunIdentity:
        raise TypeError("identity must be an exact RunIdentity")
    normalized_provider = str(provider or "unknown").strip().lower() or "unknown"
    normalized_model = str(model or "unknown-model").strip() or "unknown-model"
    service_tier = None
    raw_service_tier = (payload or {}).get("service_tier")
    if isinstance(raw_service_tier, str) and raw_service_tier.strip():
        candidate_service_tier = raw_service_tier.strip().lower()
        if _SLUG_RE.fullmatch(candidate_service_tier) is not None:
            service_tier = candidate_service_tier
    raw_billing_surface = (payload or {}).get("billing_surface")
    billing_surface = (
        raw_billing_surface.strip().lower()
        if isinstance(raw_billing_surface, str) and raw_billing_surface.strip()
        else None
    )
    raw_batch = (payload or {}).get("batch")
    batch = raw_batch if type(raw_batch) is bool else None
    raw_inference_geo = (payload or {}).get("inference_geo")
    inference_geo = (
        raw_inference_geo.strip().lower()
        if isinstance(raw_inference_geo, str) and raw_inference_geo.strip()
        else None
    )
    billing_dimensions = ProviderBillingDimensions(
        billing_surface=billing_surface,
        batch=batch,
        inference_geo=inference_geo,
    )
    timing = ProviderCallTiming(
        started_at=started_at,
        completed_at=completed_at,
    )
    call_identity = ProviderCallIdentity(
        execution_id=identity.execution_id,
        attempt_id=identity.attempt_id,
        root_run_id=identity.root_run_id,
        owner_run_id=identity.run_id,
        parent_run_id=identity.parent_run_id,
        iteration=max(0, int(iteration)),
        retry_ordinal=max(0, int(retry_ordinal)),
        purpose=purpose,
        request_sha256=request_digest,
        route=route,
    )
    runtime_extensions = {
        "unchain.runtime/usage_capture": (
            "provider_attempt_unobserved" if turn is None else (
                "provider_adapter"
                if getattr(turn, "provider_call_usage", None) is not None
                else "legacy_model_turn"
            )
        ),
    }
    if classification is not None:
        runtime_extensions[
            "unchain.runtime/provider_attempt_classification"
        ] = str(classification)
    if turn is None:
        receipt = ProviderCallReceipt(
            identity=call_identity,
            provider=normalized_provider,
            model=normalized_model,
            status=status,
            usage=ProviderCallUsage(source="unavailable"),
            service_tier=service_tier,
            timing=timing,
            provider_ids=ProviderCallIds(),
            billing_dimensions=billing_dimensions,
            extensions=runtime_extensions,
        )
    else:
        receipt = ProviderCallReceipt.from_model_turn_result(
            identity=call_identity,
            provider=normalized_provider,
            model=normalized_model,
            result=turn,
            status=status,
            service_tier=service_tier,
            raw_usage_sha256=getattr(
                turn,
                "provider_raw_usage_sha256",
                None,
            ),
            timing=timing,
            provider_ids=ProviderCallIds(
                request_id_sha256=getattr(
                    turn,
                    "provider_request_id_sha256",
                    None,
                ),
                response_id_sha256=(
                    getattr(turn, "provider_response_id_sha256", None)
                    or _provider_id_sha256(turn.response_id)
                ),
            ),
            billing_dimensions=billing_dimensions,
            extensions=runtime_extensions,
        )
    from ..pricing_catalog import resolve_pricing_for_receipt

    pricing = resolve_pricing_for_receipt(
        receipt,
        occurred_at=started_at,
        billing_surface=receipt.billing_dimensions.billing_surface,
        batch=receipt.billing_dimensions.batch,
        inference_geo=receipt.billing_dimensions.inference_geo,
    )
    if pricing != receipt.pricing:
        receipt = replace(
            receipt,
            pricing=pricing,
            extensions=dict(receipt.extensions),
        )
    return receipt


def record_model_turn(
    state: Any,
    turn: Any,
    *,
    iteration: int,
    retry_ordinal: int,
    purpose: str,
    request_digest: str,
    payload: Mapping[str, Any] | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    route: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> ProviderCallReceipt:
    ledger = getattr(state, "run_ledger", None)
    if type(ledger) is not RunLedger or ledger.identity is None:
        raise RunBundleProtocolError("provider turn has no initialized run ledger")
    resolved_provider = str(
        provider
        or getattr(getattr(state, "provider_state", None), "provider", None)
        or "unknown"
    ).strip().lower()
    resolved_model = str(
        model
        or getattr(getattr(state, "provider_state", None), "model", None)
        or "unknown-model"
    ).strip()
    expected_route = route or provider_call_route(resolved_provider)
    atomic_receipt = getattr(turn, "provider_call_receipt", None)
    if atomic_receipt is not None:
        if type(atomic_receipt) is not ProviderCallReceipt:
            raise TypeError(
                "ModelTurnResult.provider_call_receipt must be an exact receipt"
            )
        call_identity = atomic_receipt.identity
        if (
            atomic_receipt.provider != resolved_provider
            or atomic_receipt.model != resolved_model
            or atomic_receipt.status != "completed"
            or call_identity.execution_id != ledger.identity.execution_id
            or call_identity.attempt_id != ledger.identity.attempt_id
            or call_identity.root_run_id != ledger.identity.root_run_id
            or call_identity.owner_run_id != ledger.identity.run_id
            or call_identity.parent_run_id != ledger.identity.parent_run_id
            or call_identity.iteration != max(0, int(iteration))
            or call_identity.retry_ordinal != max(0, int(retry_ordinal))
            or call_identity.purpose != purpose
            or call_identity.request_sha256 != request_digest
            or call_identity.route != expected_route
        ):
            raise RunBundleProtocolError(
                "durable provider accounting receipt changed the live send identity"
            )
        durable_replay = ledger.receipts.get(atomic_receipt.provider_call_id)
        if durable_replay is not None:
            if durable_replay != atomic_receipt:
                raise RunBundleProtocolError(
                    "durable provider accounting receipt conflicts with memory"
                )
            return durable_replay
        ledger.append(atomic_receipt)
        return atomic_receipt
    receipt = build_model_attempt_receipt(
        identity=ledger.identity,
        provider=resolved_provider,
        model=resolved_model,
        iteration=iteration,
        retry_ordinal=retry_ordinal,
        purpose=purpose,
        request_digest=request_digest,
        route=expected_route,
        payload=payload,
        started_at=started_at,
        completed_at=completed_at,
        turn=turn,
        status=(
            "completed"
            if started_at is not None and completed_at is not None
            else "uncertain"
        ),
    )
    durable_replay = ledger.receipts.get(receipt.provider_call_id)
    if durable_replay is not None:
        return durable_replay
    ledger.append(receipt)
    return receipt


def record_unobserved_model_attempt(
    state: Any,
    *,
    iteration: int,
    retry_ordinal: int,
    purpose: str,
    request_digest: str,
    payload: Mapping[str, Any] | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    route: str | None = None,
    status: str = "uncertain",
    provider: str | None = None,
    model: str | None = None,
) -> ProviderCallReceipt:
    """Record one provider send that produced no trustworthy usage result."""

    ledger = getattr(state, "run_ledger", None)
    if type(ledger) is not RunLedger or ledger.identity is None:
        raise RunBundleProtocolError("provider attempt has no initialized run ledger")
    resolved_provider = str(
        provider
        or getattr(getattr(state, "provider_state", None), "provider", None)
        or "unknown"
    ).strip().lower()
    resolved_model = str(
        model
        or getattr(getattr(state, "provider_state", None), "model", None)
        or "unknown-model"
    ).strip()
    receipt = build_model_attempt_receipt(
        identity=ledger.identity,
        provider=resolved_provider,
        model=resolved_model,
        iteration=iteration,
        retry_ordinal=retry_ordinal,
        purpose=purpose,
        request_digest=request_digest,
        route=route or provider_call_route(resolved_provider),
        payload=payload,
        started_at=started_at,
        completed_at=completed_at,
        turn=None,
        status=status,
    )
    durable_replay = ledger.receipts.get(receipt.provider_call_id)
    if durable_replay is not None:
        return durable_replay
    ledger.append(receipt)
    return receipt


def materialize_state_bundle(state: Any, *, status: str) -> dict[str, Any]:
    ledger = getattr(state, "run_ledger", None)
    if type(ledger) is not RunLedger:
        raise TypeError("RunState.run_ledger must be an exact RunLedger")
    if ledger.identity is None:
        fallback_run_id = str(
            getattr(getattr(state, "session_state", None), "session_id", None)
            or "legacy-kernel"
        )
        ledger.initialize(state=state, run_id=fallback_run_id)
    subagent_state = getattr(state, "subagent_state", None)
    child_values = getattr(subagent_state, "run_bundles", {})
    iteration = max(0, int(getattr(state, "iteration", 0) or 0))
    if ledger.descriptor.iteration != iteration:
        ledger.descriptor = replace(ledger.descriptor, iteration=iteration)
    return ledger.materialize(
        kernel_status=status,
        child_bundle_values=(
            child_values if isinstance(child_values, Mapping) else {}
        ),
    ).to_dict()


def merge_run_bundle_values(
    bundle_values: list[dict[str, Any] | None],
    *,
    compact_details_ledger: Any = None,
) -> dict[str, Any] | None:
    bundles = tuple(
        run_bundle_from_dict(value)
        for value in bundle_values
        if isinstance(value, dict)
    )
    if not bundles:
        return None
    if len(bundles) == 1:
        return bundles[0].to_dict()
    final = bundles[-1]
    root_candidates = tuple(
        bundle
        for bundle in bundles
        if bundle.identity.parent_run_id is None
        and bundle.identity.run_id == bundle.identity.root_run_id
    )
    if len({bundle.identity for bundle in root_candidates}) != 1:
        raise RunBundleProtocolError(
            "run bundle merge requires one unambiguous root identity"
        )
    root = root_candidates[0]
    merged_lifecycle = RunLifecycle(
        status=final.lifecycle.status,
        started_at=root.lifecycle.started_at,
        completed_at=final.lifecycle.completed_at or root.lifecycle.completed_at,
        continued_from_run_id=root.lifecycle.continued_from_run_id,
    )
    next_revision = max(bundle.revision for bundle in bundles) + 1
    merged_extensions = {
        **dict(root.extensions),
        "unchain.runtime/merged_bundle_count": len(bundles),
    }
    if all(type(bundle) is RunBundle for bundle in bundles):
        try:
            merged = RunBundleReducer.reduce_bundles(
                identity=root.identity,
                lifecycle=merged_lifecycle,
                bundles=bundles,
                descriptor=root.descriptor,
                revision=next_revision,
                extensions=merged_extensions,
            )
            return merged.to_dict()
        except RunBundleProtocolError as exc:
            if not _is_run_bundle_canonical_limit_error(exc):
                raise

    from ..run_bundle_ledger import (
        RunBundleCompactDetailsLedger,
        RunBundleProjectionDetailsLedger,
    )

    if not isinstance(compact_details_ledger, RunBundleCompactDetailsLedger):
        raise RunBundleProtocolError(
            "compact run bundle merge requires durable details persistence"
        )
    receipt_by_id: dict[str, ProviderCallReceipt] = {}
    event_by_id: dict[str, RunMetricEvent] = {}
    child_by_id: dict[str, RunChild] = {}

    def add_receipt(receipt: ProviderCallReceipt) -> None:
        prior = receipt_by_id.get(receipt.provider_call_id)
        if prior is not None and prior.receipt_sha256 != receipt.receipt_sha256:
            raise RunBundleProtocolError(
                "one provider_call_id has conflicting immutable receipts"
            )
        receipt_by_id[receipt.provider_call_id] = receipt

    def add_event(event: RunMetricEvent) -> None:
        prior = event_by_id.get(event.metric_event_id)
        if prior is not None and prior != event:
            raise RunBundleProtocolError(
                "one metric_event_id has conflicting immutable events"
            )
        event_by_id[event.metric_event_id] = event

    def add_child(child: RunChild) -> None:
        prior = child_by_id.get(child.run_id)
        if prior is not None and prior != child:
            raise RunBundleProtocolError("one child run id has conflicting topology")
        child_by_id[child.run_id] = child

    for bundle in bundles:
        if (
            bundle.identity.execution_id != root.identity.execution_id
            or bundle.identity.root_run_id != root.identity.root_run_id
        ):
            raise RunBundleProtocolError(
                "child bundle identity crosses the target run boundary"
            )
        if type(bundle) is CompactRunBundle:
            receipts, events, children = (
                compact_details_ledger.load_compact_bundle_details(bundle=bundle)
            )
        else:
            receipts = tuple(bundle.provider_calls)
            children = tuple(bundle.children)
            extension = _extract_compact_projection_extension(bundle.extensions)
            if extension is None:
                events = tuple(bundle.metrics.events)
            else:
                if not isinstance(
                    compact_details_ledger,
                    RunBundleProjectionDetailsLedger,
                ):
                    raise RunBundleProtocolError(
                        _COMPACT_PROJECTION_UNSUPPORTED_ERROR
                    )
                events = compact_details_ledger.load_projection_details(
                    bundle_id=bundle.bundle_id,
                    revision=bundle.revision,
                    projection_hash=extension["projection_hash"],
                    metric_events_sha256=extension["metric_events_sha256"],
                )
        for receipt in receipts:
            add_receipt(receipt)
        for event in events:
            add_event(event)
        for child in children:
            add_child(child)
        if bundle.identity.run_id != root.identity.run_id:
            if bundle.identity.parent_run_id is None:
                raise RunBundleProtocolError(
                    "child bundle identity requires a parent run"
                )
            add_child(
                RunChild(
                    run_id=bundle.identity.run_id,
                    attempt_id=bundle.identity.attempt_id,
                    parent_run_id=bundle.identity.parent_run_id,
                    relation=bundle.identity.relation,
                    bundle_id=bundle.bundle_id,
                    status=bundle.lifecycle.status,
                )
            )
    compact, details = CompactRunBundle.from_facts(
        identity=root.identity,
        lifecycle=merged_lifecycle,
        descriptor=root.descriptor,
        revision=next_revision,
        receipts=tuple(receipt_by_id.values()),
        metric_events=tuple(event_by_id.values()),
        children=tuple(child_by_id.values()),
        extensions=merged_extensions,
    )
    durable = compact_details_ledger.persist_compact_bundle_with_details(
        bundle=compact,
        details=details,
    )
    if durable != compact:
        raise RunBundleProtocolError(
            "durable compact run bundle changed its immutable projection"
        )
    return compact.to_dict()


__all__ = [
    "RunLedger",
    "attach_state_run_bundle_ledger",
    "build_model_attempt_receipt",
    "child_run_identity",
    "initialize_run_ledger",
    "materialize_state_bundle",
    "merge_run_bundle_values",
    "provider_call_route",
    "record_model_turn",
    "record_unobserved_model_attempt",
    "request_sha256",
]
