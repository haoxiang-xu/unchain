"""Content-free Context Composition carrier and receipt projection."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from ..kernel.delta import HarnessDelta
from ..kernel.harness import BaseRuntimeHarness, HarnessContext, RuntimePhase
from ..providers.physical_send import ProviderPhysicalSendContext
from ..run_bundle import ProviderCallReceipt


_logger = logging.getLogger(__name__)


INTERNAL_CONTEXT_COMPOSITION_SCHEMA = (
    "unchain.context/internal_context_composition_v1"
)
CONTEXT_COMPOSITION_EXTENSION_KEY = "unchain.context/context_composition_v1"
CONTEXT_COMPOSITION_EXTENSION_SCHEMA = "unchain.context/context_composition_v1"
# v2 (2026-08-21) bundles two changes as one method bump: the calibrated
# divisor below, and rescaling onto the provider's own billed total when the
# heuristic overshoots it (see _scale_categories_to_target). Both change the
# numbers a receipt carries, so both must be visible in `method` for anyone
# doing archaeology across old (v1) and new (v2) receipts.
CONTEXT_COMPOSITION_METHOD = "utf8_heuristic_v2"
# Calibrated against real o200k_base tokenization (2026-08-21) across four
# distinct real-content samples — this repo's own builtin tool schemas (17
# tools, provider-shaped JSON), source file text (code + docstrings), README
# markdown, and a representative assistant reply — NOT a single content type.
# The old bytes/4 (~4.0 chars/token) assumption measured ~4.66 chars/token in
# aggregate against real tokenization, an ~17% systematic overestimate across
# every category, not a tool-schema-specific defect as first suspected: real
# tool schemas measured the LEAST biased of the four (4.86 chars/token),
# prose and markdown measured similarly (4.35-4.57). 4.5 lands close to that
# real aggregate while staying on the safe (overestimating, never
# underestimating) side by design — this indicator must never tell a user
# their context has more room than it actually does. On its own this still
# leaves most real calls landing on "estimated" (three real-traffic shapes —
# short English, long English, long CJK — all measured attributed tokens
# exceeding the provider's real total under this divisor alone); rescaling
# onto the billed total is what actually resolves that, see
# _scale_categories_to_target below.
_HEURISTIC_BYTES_PER_TOKEN = 4.5
_SOURCE_SCHEMA = "unchain.context/context_composition_source_v1"
_SOURCE_STATE_KEY = "context_composition_source_v1"
_MAX_SAFE_INTEGER = (1 << 53) - 1
_MAX_MANIFEST_BYTES = 16 * 1024
_MAX_ROUTE_CONTRIBUTIONS = 128

_SURFACE_ORDER = (
    "messages",
    "tool_schema",
    "response_schema",
    "provider_state",
)
_ROUTE_ORDER = ("primary", "openai_previous_response_fallback")
_CONTEXT_MODES = frozenset(
    {"semantic", "local_replay", "remote_continuation"}
)
_TAXONOMY = {
    "instructions": (
        "core_system",
        "agent_instructions",
        "user_rules",
        "runtime_safety",
        "recipe_workflow",
    ),
    "skills": ("catalog_metadata", "loaded_body", "expanded_invocation"),
    "tool_definitions": ("provider_schema", "prompt_guidance", "dynamic_tool"),
    "conversation": (
        "current_input",
        "user_history",
        "assistant_history",
        "summary",
    ),
    "tool_activity": ("arguments", "results", "errors_observations"),
    "memory": ("short_term_recall", "long_term_recall", "pending_memory"),
    "task_state": ("pinned_state", "pending_interaction", "plan_state"),
    "files_media": ("file_excerpt", "artifact", "image_media", "web_pdf"),
    "agent_coordination": (
        "inherited_context",
        "handoff_summary",
        "child_instructions",
        "subagent_report_roster",
    ),
    "output_contract": ("response_schema", "format_instruction"),
}
_CATEGORY_ORDER = tuple(_TAXONOMY)
_PRIVATE_HINT_KEYS = frozenset(
    {"category", "subtype", "surface", "utf8_bytes", "source_count"}
)
_MANIFEST_KEYS = frozenset(
    {"schema", "method", "context_window_tokens", "routes"}
)
_ROUTE_KEYS = frozenset(
    {
        "route_name",
        "context_mode",
        "provider_retained",
        "manifest_items",
        "wire_surfaces",
        "contributions",
    }
)


class ContextCompositionContractError(ValueError):
    """Context Composition metadata is malformed or inconsistent."""


def _safe_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if not minimum <= value <= _MAX_SAFE_INTEGER:
        raise ContextCompositionContractError(
            f"{label} must be between {minimum} and {_MAX_SAFE_INTEGER}"
        )
    return value


def _checked_sum(values: list[int], label: str) -> int:
    total = sum(values)
    if total > _MAX_SAFE_INTEGER:
        raise ContextCompositionContractError(f"{label} exceeds JSON safe integer")
    return total


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _thaw_json(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise ContextCompositionContractError(
            "context composition value must be strict canonical JSON"
        ) from exc


def validate_private_context_composition_hint(value: object) -> dict[str, Any]:
    """Validate the one privacy-preserving durable aggregate tuple."""

    if type(value) is not dict or set(value) != _PRIVATE_HINT_KEYS:
        raise ContextCompositionContractError(
            "private context composition hint fields are invalid"
        )
    if (
        value["category"] != "skills"
        or value["subtype"] != "expanded_invocation"
        or value["surface"] != "messages"
        or value["source_count"] != 1
    ):
        raise ContextCompositionContractError(
            "private context composition hint identity is invalid"
        )
    utf8_bytes = _safe_int(value["utf8_bytes"], "utf8_bytes", minimum=1)
    _safe_int(value["source_count"], "source_count", minimum=1)
    return {
        "category": "skills",
        "subtype": "expanded_invocation",
        "surface": "messages",
        "utf8_bytes": utf8_bytes,
        "source_count": 1,
    }


def _validate_contribution(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _PRIVATE_HINT_KEYS:
        raise ContextCompositionContractError(
            "context contribution fields are invalid"
        )
    category = value["category"]
    subtype = value["subtype"]
    surface = value["surface"]
    if type(category) is not str or category not in _TAXONOMY:
        raise ContextCompositionContractError("context contribution category is invalid")
    if type(subtype) is not str or subtype not in _TAXONOMY[category]:
        raise ContextCompositionContractError("context contribution subtype is invalid")
    if type(surface) is not str or surface not in _SURFACE_ORDER:
        raise ContextCompositionContractError("context contribution surface is invalid")
    return {
        "category": category,
        "subtype": subtype,
        "surface": surface,
        "utf8_bytes": _safe_int(value["utf8_bytes"], "utf8_bytes", minimum=1),
        "source_count": _safe_int(
            value["source_count"], "source_count", minimum=1
        ),
    }


def _contribution_order(value: Mapping[str, Any]) -> tuple[int, int, int]:
    category = value["category"]
    return (
        _CATEGORY_ORDER.index(category),
        _TAXONOMY[category].index(value["subtype"]),
        _SURFACE_ORDER.index(value["surface"]),
    )


def _validate_manifest(value: object) -> dict[str, Any]:
    raw = _thaw_json(value)
    if type(raw) is not dict or set(raw) != _MANIFEST_KEYS:
        raise ContextCompositionContractError(
            "internal context composition fields are invalid"
        )
    if raw["schema"] != INTERNAL_CONTEXT_COMPOSITION_SCHEMA:
        raise ContextCompositionContractError(
            "internal context composition schema is unsupported"
        )
    if raw["method"] != CONTEXT_COMPOSITION_METHOD:
        raise ContextCompositionContractError(
            "internal context composition method is unsupported"
        )
    context_window = raw["context_window_tokens"]
    if context_window is not None:
        context_window = _safe_int(
            context_window,
            "context_window_tokens",
            minimum=1,
        )
    routes = raw["routes"]
    if type(routes) is not list or not 1 <= len(routes) <= 2:
        raise ContextCompositionContractError(
            "internal context composition routes are invalid"
        )
    normalized_routes: list[dict[str, Any]] = []
    route_names: list[str] = []
    for route in routes:
        if type(route) is not dict or set(route) != _ROUTE_KEYS:
            raise ContextCompositionContractError(
                "internal context composition route fields are invalid"
            )
        route_name = route["route_name"]
        if type(route_name) is not str or route_name not in _ROUTE_ORDER:
            raise ContextCompositionContractError(
                "internal context composition route is unsupported"
            )
        if route_name in route_names:
            raise ContextCompositionContractError(
                "internal context composition routes are duplicated"
            )
        route_names.append(route_name)
        context_mode = route["context_mode"]
        if type(context_mode) is not str or context_mode not in _CONTEXT_MODES:
            raise ContextCompositionContractError(
                "internal context composition context mode is unsupported"
            )
        provider_retained = route["provider_retained"]
        if type(provider_retained) is not bool:
            raise TypeError("provider_retained must be an exact boolean")
        manifest_items = _safe_int(route["manifest_items"], "manifest_items")
        wire_surfaces = _safe_int(route["wire_surfaces"], "wire_surfaces")
        if wire_surfaces > len(_SURFACE_ORDER):
            raise ContextCompositionContractError("wire_surfaces exceeds its limit")
        contributions = route["contributions"]
        if type(contributions) is not list or len(contributions) > _MAX_ROUTE_CONTRIBUTIONS:
            raise ContextCompositionContractError(
                "internal context composition contributions are invalid"
            )
        normalized = [_validate_contribution(item) for item in contributions]
        identities = [
            (item["category"], item["subtype"], item["surface"])
            for item in normalized
        ]
        if len(identities) != len(set(identities)):
            raise ContextCompositionContractError(
                "internal context composition contributions are duplicated"
            )
        if normalized != sorted(normalized, key=_contribution_order):
            raise ContextCompositionContractError(
                "internal context composition contributions are not canonical"
            )
        represented_items = _checked_sum(
            [item["source_count"] for item in normalized],
            "represented context items",
        )
        represented_surfaces = len({item["surface"] for item in normalized})
        if manifest_items < represented_items or wire_surfaces < represented_surfaces:
            raise ContextCompositionContractError(
                "internal context composition route totals are inconsistent"
            )
        normalized_routes.append(
            {
                "route_name": route_name,
                "context_mode": context_mode,
                "provider_retained": provider_retained,
                "manifest_items": manifest_items,
                "wire_surfaces": wire_surfaces,
                "contributions": normalized,
            }
        )
    if route_names != list(_ROUTE_ORDER[: len(route_names)]):
        raise ContextCompositionContractError(
            "internal context composition routes are not canonical"
        )
    normalized_manifest = {
        "schema": INTERNAL_CONTEXT_COMPOSITION_SCHEMA,
        "method": CONTEXT_COMPOSITION_METHOD,
        "context_window_tokens": context_window,
        "routes": normalized_routes,
    }
    if len(_canonical_bytes(normalized_manifest)) > _MAX_MANIFEST_BYTES:
        raise ContextCompositionContractError(
            "internal context composition manifest exceeds 16 KiB"
        )
    return normalized_manifest


def freeze_internal_context_composition(value: object) -> Mapping[str, Any]:
    """Return one deeply immutable, strictly validated internal manifest."""

    return _freeze_json(_validate_manifest(value))


@dataclass
class ContextCompositionBootstrapHarness(BaseRuntimeHarness):
    """Publish one official content-free private contribution before a model call."""

    name: str = "context_composition_bootstrap"
    phases: tuple[RuntimePhase, ...] = ("before_model",)
    order: int = 950
    private_hint: Mapping[str, Any] = field(kw_only=True, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "private_hint",
            _freeze_json(
                validate_private_context_composition_hint(
                    _thaw_json(self.private_hint)
                )
            ),
        )

    def build_delta(self, context: HarnessContext) -> HarnessDelta:
        if type(context) is not HarnessContext:
            raise TypeError("context must be an exact HarnessContext")
        source = {
            "schema": _SOURCE_SCHEMA,
            "contributions": [_thaw_json(self.private_hint)],
        }
        return HarnessDelta(
            created_by="context.context_composition_bootstrap",
            base_version_id=context.latest_version_id,
            ops=(),
            state_updates={_SOURCE_STATE_KEY: source},
            trace={"context_composition_source": True},
        )


_MESSAGE_TYPE_SLOTS = {
    "function_call": ("tool_activity", "arguments"),
    "tool_use": ("tool_activity", "arguments"),
    "function_call_output": ("tool_activity", "results"),
    "tool_result": ("tool_activity", "results"),
}

_MESSAGE_ROLE_SLOTS = {
    "system": ("instructions", "core_system"),
    "developer": ("instructions", "core_system"),
    "assistant": ("conversation", "assistant_history"),
    "tool": ("tool_activity", "results"),
}


def _classify_message(
    message: Mapping[str, Any],
    *,
    current_input: bool,
) -> tuple[str, str] | None:
    """Map one wire message onto its taxonomy slot, or None when unknown.

    Classification is deliberately shallow — top-level ``type``/``role`` only.
    Provider payloads nest tool blocks inside message content in provider-
    specific ways, and guessing at those would attribute bytes to the wrong
    slot. Anything unrecognised stays uninstrumented and lands in the residual,
    which is honest about what is not yet attributed.
    """

    message_type = message.get("type")
    if isinstance(message_type, str) and message_type in _MESSAGE_TYPE_SLOTS:
        return _MESSAGE_TYPE_SLOTS[message_type]
    role = message.get("role")
    if not isinstance(role, str):
        return None
    if role == "system":
        # The skills harnesses render two delimited system blocks; attribute
        # them to the reserved skills slots instead of core instructions.
        content = message.get("content")
        if isinstance(content, str):
            stripped = content.lstrip()
            if stripped.startswith("<available_skills>"):
                return ("skills", "catalog_metadata")
            if stripped.startswith("<active_skills>"):
                return ("skills", "loaded_body")
    if role == "user":
        return (
            "conversation",
            "current_input" if current_input else "user_history",
        )
    return _MESSAGE_ROLE_SLOTS.get(role)


def _measured_message_contributions(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attribute the wire messages this route actually carries.

    A route's contribution identities must be unique, so messages sharing a slot
    merge into one entry holding their combined bytes and count. Bytes are the
    canonical JSON of the message — the same shape that goes on the wire — so
    the estimate tracks what the provider is really billed for.
    """

    last_user_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, Mapping) and message.get("role") == "user":
            last_user_index = index

    merged: dict[tuple[str, str], dict[str, int]] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        slot = _classify_message(message, current_input=index == last_user_index)
        if slot is None:
            continue
        bucket = merged.setdefault(slot, {"utf8_bytes": 0, "source_count": 0})
        bucket["utf8_bytes"] += len(_canonical_bytes(message))
        bucket["source_count"] += 1

    return [
        {
            "category": category,
            "subtype": subtype,
            "surface": "messages",
            "utf8_bytes": values["utf8_bytes"],
            "source_count": values["source_count"],
        }
        for (category, subtype), values in merged.items()
    ]


def _measured_tool_schema_contribution(
    tool_schemas: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Attribute the wire tool schema list this route actually carries.

    Every schema shares one taxonomy slot — bytes are the canonical JSON of
    each provider-shaped tool schema, the same shape that goes on the wire
    (providers call ``toolkit.to_provider_json(provider)`` verbatim to build
    the request), mirroring how ``_measured_message_contributions`` treats
    messages. A route with no tools attached has nothing to attribute.
    """

    if not tool_schemas:
        return None
    return {
        "category": "tool_definitions",
        "subtype": "provider_schema",
        "surface": "tool_schema",
        "utf8_bytes": _checked_sum(
            [len(_canonical_bytes(schema)) for schema in tool_schemas],
            "tool schema utf8_bytes",
        ),
        "source_count": len(tool_schemas),
    }


def _source_contributions(state: object) -> list[dict[str, Any]]:
    metadata = getattr(state, "metadata", None)
    raw = metadata.get(_SOURCE_STATE_KEY) if isinstance(metadata, dict) else None
    if raw is None:
        return []
    if (
        type(raw) is not dict
        or set(raw) != {"schema", "contributions"}
        or raw["schema"] != _SOURCE_SCHEMA
        or type(raw["contributions"]) is not list
        or not 1 <= len(raw["contributions"]) <= _MAX_ROUTE_CONTRIBUTIONS
    ):
        raise ContextCompositionContractError(
            "official context composition source facts are invalid"
        )
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source in raw["contributions"]:
        item = validate_private_context_composition_hint(source)
        identity = (item["category"], item["subtype"], item["surface"])
        current = merged.get(identity)
        if current is None:
            merged[identity] = dict(item)
            continue
        current["utf8_bytes"] = _checked_sum(
            [current["utf8_bytes"], item["utf8_bytes"]],
            "merged contribution utf8_bytes",
        )
        current["source_count"] = _checked_sum(
            [current["source_count"], item["source_count"]],
            "merged contribution source_count",
        )
    return sorted(merged.values(), key=_contribution_order)


def _assembly_messages(assembly: object, field_name: str) -> list[dict[str, Any]]:
    raw = getattr(assembly, field_name, None)
    if raw is None:
        return []
    if type(raw) is not list or any(type(item) is not dict for item in raw):
        raise ContextCompositionContractError(
            f"provider context {field_name} must be an exact message array"
        )
    return raw


def build_internal_context_composition(
    state: object,
    assembly: object,
    *,
    tool_schemas: list[dict[str, Any]] | None = None,
    response_schema_surface: str | None = None,
) -> Mapping[str, Any] | None:
    """Project official source facts into exact primary/fallback route manifests."""

    contributions = _source_contributions(state)
    # Messages alone are enough to build a manifest: attribution no longer
    # depends on the host having supplied contribution facts.
    if not contributions and not _assembly_messages(assembly, "messages"):
        return None
    mode = getattr(assembly, "mode", None)
    if mode not in _CONTEXT_MODES:
        raise ContextCompositionContractError("provider context mode is unsupported")
    if tool_schemas is not None and type(tool_schemas) is not list:
        raise TypeError("tool_schemas must be an exact list")
    tool_schema_contribution = _measured_tool_schema_contribution(tool_schemas)
    if (
        response_schema_surface is not None
        and response_schema_surface not in {"messages", "response_schema"}
    ):
        raise ContextCompositionContractError(
            "response schema surface is unsupported"
        )
    previous_response_id = getattr(assembly, "previous_response_id", None)
    provider_retained = bool(
        mode == "remote_continuation"
        and isinstance(previous_response_id, str)
        and previous_response_id
    )

    def route_contributions(
        *,
        retained: bool,
        measured: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in (*contributions, *measured):
            projected = dict(item)
            if retained:
                projected["surface"] = "provider_state"
            output.append(projected)
        return sorted(output, key=_contribution_order)

    def route_value(
        route_name: str,
        context_mode: str,
        retained: bool,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        measured = _measured_message_contributions(messages)
        projected = route_contributions(retained=retained, measured=measured)
        # Tool schemas are always sent on the wire as their own surface,
        # independent of whether the route retains prior turns server-side —
        # unlike measured messages, they must NOT go through the retained ->
        # provider_state surface override above.
        if tool_schema_contribution is not None:
            projected = sorted(
                [*projected, tool_schema_contribution], key=_contribution_order
            )
        represented_items = _checked_sum(
            [item["source_count"] for item in projected],
            "represented route items",
        )
        attributed_messages = _checked_sum(
            [item["source_count"] for item in measured],
            "attributed route messages",
        )
        known_uninstrumented_items = _checked_sum(
            [
                max(0, len(messages) - attributed_messages),
                1 if response_schema_surface is not None else 0,
                1 if retained else 0,
            ],
            "known uninstrumented route items",
        )
        surfaces = {item["surface"] for item in projected}
        if messages:
            surfaces.add("messages")
        if response_schema_surface is not None:
            surfaces.add(response_schema_surface)
        if retained:
            surfaces.add("provider_state")
        return {
            "route_name": route_name,
            "context_mode": context_mode,
            "provider_retained": retained,
            "manifest_items": _checked_sum(
                [represented_items, known_uninstrumented_items],
                "manifest_items",
            ),
            "wire_surfaces": len(surfaces),
            "contributions": projected,
        }

    primary_messages = _assembly_messages(assembly, "messages")
    routes = [
        route_value(
            "primary",
            mode,
            provider_retained,
            primary_messages,
        )
    ]
    fallback_messages = getattr(assembly, "fallback_messages", None)
    if fallback_messages is not None:
        fallback_messages = _assembly_messages(assembly, "fallback_messages")
        routes.append(
            route_value(
                "openai_previous_response_fallback",
                "local_replay",
                False,
                fallback_messages,
            )
        )
    provider_state = getattr(state, "provider_state", None)
    context_window_raw = getattr(provider_state, "max_context_window_tokens", 0)
    context_window = (
        int(context_window_raw)
        if type(context_window_raw) is int and context_window_raw > 0
        else None
    )
    return freeze_internal_context_composition(
        {
            "schema": INTERNAL_CONTEXT_COMPOSITION_SCHEMA,
            "method": CONTEXT_COMPOSITION_METHOD,
            "context_window_tokens": context_window,
            "routes": routes,
        }
    )


def _surface_present(route_request: Mapping[str, Any], surface: str) -> bool:
    if surface == "messages":
        return any(
            route_request.get(key) not in (None, "", [], {})
            for key in ("input", "messages", "system")
        )
    if surface == "tool_schema":
        return route_request.get("tools") not in (None, [], {})
    if surface == "response_schema":
        return any(
            route_request.get(key) not in (None, "", [], {})
            for key in ("text", "response_format", "output_config")
        )
    if surface == "provider_state":
        return route_request.get("previous_response_id") not in (None, "")
    return False


def _scale_categories_to_target(
    categories: list[dict[str, Any]],
    attributed_tokens: int,
    target: int,
) -> list[dict[str, Any]] | None:
    """Rescale every subtype's heuristic token estimate onto the provider's
    own billed total, preserving each subtype's relative share of the whole.

    The heuristic's absolute error is the same error shared across every
    category — this panel sells composition *proportions*, and proportions
    are what the heuristic gets closest to right, so anchoring the sum to the
    one number that is exactly true (the provider's own usage receipt) turns
    a self-contradictory total (parts summing to more than the whole the
    headline shows) into a self-consistent one, without pretending any single
    category's number is more precise than it is.

    Returns None — caller must not scale, quality stays "estimated" — when
    there are more leaf subtypes than target tokens to hand out. Every
    subtype must keep at least 1 token (the wire-attribution invariant this
    module and its PuPu consumer both enforce), so scaling below that count
    would have to invent tokens nobody measured.

    Uses largest-remainder apportionment: each subtype is guaranteed floor
    of its proportional share, plus 1 baseline token, then whatever is left
    over from integer rounding goes to the subtypes with the largest
    fractional remainder — ties broken by canonical taxonomy order — so the
    total lands on `target` exactly. This must be fully deterministic:
    enrich_provider_call_receipt independently re-derives this same value to
    verify against the one a provider call actually produced, and any
    non-deterministic tie-break would make that comparison flap.
    """

    leaves = [
        (category, subtype)
        for category in categories
        for subtype in category["subtypes"]
    ]
    leaf_count = len(leaves)
    if leaf_count == 0 or target < leaf_count:
        return None

    pool = target - leaf_count
    allocations: list[int] = []
    fractions: list[float] = []
    for _category, subtype in leaves:
        share = subtype["tokens"] / attributed_tokens * pool
        floor_share = math.floor(share)
        allocations.append(1 + floor_share)
        fractions.append(share - floor_share)

    remaining = target - sum(allocations)
    order = sorted(range(leaf_count), key=lambda i: (-fractions[i], i))
    for i in order[:remaining]:
        allocations[i] += 1

    scaled: list[dict[str, Any]] = []
    cursor = 0
    for category in categories:
        scaled_subtypes: list[dict[str, Any]] = []
        category_tokens = 0
        for subtype in category["subtypes"]:
            alloc = allocations[cursor]
            cursor += 1
            category_tokens += alloc
            scaled_subtypes.append(
                {
                    "id": subtype["id"],
                    "tokens": alloc,
                    "source_count": subtype["source_count"],
                }
            )
        scaled.append(
            {
                "id": category["id"],
                "tokens": category_tokens,
                "source_count": category["source_count"],
                "subtypes": scaled_subtypes,
            }
        )
    return scaled


def _derive_context_composition_extension(
    *,
    manifest: Mapping[str, Any],
    envelope: object,
    send_context: ProviderPhysicalSendContext,
    receipt: ProviderCallReceipt,
) -> dict[str, Any] | None:
    """Build the closed extension for the exact authoritative physical route."""

    from ..providers.wire_envelope import ProviderWireEnvelope

    if type(receipt) is not ProviderCallReceipt:
        raise TypeError("receipt must be an exact ProviderCallReceipt")
    if type(envelope) is not ProviderWireEnvelope:
        raise TypeError("envelope must be an exact ProviderWireEnvelope")
    if type(send_context) is not ProviderPhysicalSendContext:
        raise TypeError(
            "send_context must be an exact ProviderPhysicalSendContext"
        )
    normalized = _validate_manifest(manifest)
    subject = send_context.subject
    if (
        subject.attempt != envelope.attempt
        or subject.iteration != envelope.iteration
        or subject.envelope_sha256 != envelope.envelope_sha256
    ):
        raise ContextCompositionContractError(
            "provider physical context changed the authoritative envelope"
        )
    envelope_route = next(
        (route for route in envelope.routes if route.name == subject.route),
        None,
    )
    if envelope_route is None or envelope_route != send_context.route:
        raise ContextCompositionContractError(
            "provider physical context changed the authoritative route"
        )
    route_manifest = next(
        (
            route
            for route in normalized["routes"]
            if route["route_name"] == subject.route
        ),
        None,
    )
    if route_manifest is None:
        return None
    route_request = envelope_route.request_copy()
    matched = [
        item
        for item in route_manifest["contributions"]
        if _surface_present(route_request, item["surface"])
    ]
    matched_items = _checked_sum(
        [item["source_count"] for item in matched],
        "matched_items",
    )
    matched_surfaces = len({item["surface"] for item in matched})
    coverage_complete = (
        matched_items == route_manifest["manifest_items"]
        and matched_surfaces == route_manifest["wire_surfaces"]
    )

    categories: list[dict[str, Any]] = []
    attributed_tokens = 0
    for category in _CATEGORY_ORDER:
        category_items = [item for item in matched if item["category"] == category]
        if not category_items:
            continue
        subtypes: list[dict[str, Any]] = []
        for subtype in _TAXONOMY[category]:
            subtype_items = [
                item for item in category_items if item["subtype"] == subtype
            ]
            if not subtype_items:
                continue
            subtype_tokens = _checked_sum(
                [
                    max(1, math.ceil(item["utf8_bytes"] / _HEURISTIC_BYTES_PER_TOKEN))
                    for item in subtype_items
                ],
                "subtype tokens",
            )
            subtype_sources = _checked_sum(
                [item["source_count"] for item in subtype_items],
                "subtype source_count",
            )
            subtypes.append(
                {
                    "id": subtype,
                    "tokens": subtype_tokens,
                    "source_count": subtype_sources,
                }
            )
        category_tokens = _checked_sum(
            [item["tokens"] for item in subtypes],
            "category tokens",
        )
        category_sources = _checked_sum(
            [item["source_count"] for item in subtypes],
            "category source_count",
        )
        categories.append(
            {
                "id": category,
                "tokens": category_tokens,
                "source_count": category_sources,
                "subtypes": subtypes,
            }
        )
        attributed_tokens = _checked_sum(
            [attributed_tokens, category_tokens],
            "attributed_tokens",
        )

    provider_input_total = receipt.usage.input_total_tokens
    residual_tokens: int | None = None
    if not coverage_complete:
        # Scaling onto an incomplete manifest would be false precision —
        # some of what the provider billed was never even attributed to a
        # category, so there is nothing honest to anchor a proportion to.
        quality = "partial"
    elif provider_input_total is None:
        quality = "estimated"
    elif attributed_tokens <= provider_input_total:
        quality = "reconciled_estimate"
        residual_tokens = provider_input_total - attributed_tokens
    else:
        scaled_categories = _scale_categories_to_target(
            categories, attributed_tokens, provider_input_total
        )
        if scaled_categories is None:
            quality = "estimated"
        else:
            categories = scaled_categories
            attributed_tokens = provider_input_total
            residual_tokens = 0
            quality = "reconciled_estimate"
    return {
        "schema": CONTEXT_COMPOSITION_EXTENSION_SCHEMA,
        "method": CONTEXT_COMPOSITION_METHOD,
        "quality": quality,
        "context_window_tokens": normalized["context_window_tokens"],
        "wire": {
            "envelope_sha256": f"sha256:{envelope.envelope_sha256}",
            "route_name": subject.route,
            "route_sha256": f"sha256:{envelope_route.route_sha256}",
            "context_mode": route_manifest["context_mode"],
        },
        "categories": categories,
        "attributed_tokens": attributed_tokens,
        "residual_tokens": residual_tokens,
        "coverage": {
            "status": "complete" if coverage_complete else "partial",
            "manifest_items": route_manifest["manifest_items"],
            "matched_items": matched_items,
            "wire_surfaces": route_manifest["wire_surfaces"],
            "matched_surfaces": matched_surfaces,
        },
    }


def build_context_composition_extension(
    *,
    manifest: Mapping[str, Any],
    envelope: object,
    send_context: ProviderPhysicalSendContext,
    receipt: ProviderCallReceipt,
) -> dict[str, Any] | None:
    """Build the closed extension for the exact authoritative physical route."""

    return _derive_context_composition_extension(
        manifest=manifest,
        envelope=envelope,
        send_context=send_context,
        receipt=receipt,
    )


def enrich_provider_call_receipt(
    *,
    receipt: ProviderCallReceipt,
    manifest: Mapping[str, Any] | None,
    envelope: object,
    send_context: ProviderPhysicalSendContext,
) -> ProviderCallReceipt:
    """Optionally enrich one already-valid base receipt, never replacing it."""

    if type(receipt) is not ProviderCallReceipt:
        raise TypeError("receipt must be an exact ProviderCallReceipt")
    base_extensions = dict(receipt.extensions)
    base_extensions.pop(CONTEXT_COMPOSITION_EXTENSION_KEY, None)
    sanitized_receipt = (
        receipt
        if len(base_extensions) == len(receipt.extensions)
        else replace(receipt, extensions=base_extensions)
    )
    if manifest is None:
        return sanitized_receipt
    try:
        extension = build_context_composition_extension(
            manifest=manifest,
            envelope=envelope,
            send_context=send_context,
            receipt=sanitized_receipt,
        )
        expected_extension = _derive_context_composition_extension(
            manifest=manifest,
            envelope=envelope,
            send_context=send_context,
            receipt=sanitized_receipt,
        )
        if extension is not None and (
            type(extension) is not dict or extension != expected_extension
        ):
            return sanitized_receipt
        if extension is None:
            return sanitized_receipt
        return replace(
            sanitized_receipt,
            extensions={
                **dict(sanitized_receipt.extensions),
                CONTEXT_COMPOSITION_EXTENSION_KEY: extension,
            },
        )
    except Exception as exc:
        # Composition is availability-only. This try block contains only the
        # optional projection and receipt-copy step; base factory, ledger,
        # lease and CAS failures remain outside it and must still propagate.
        # Only the exception's type name is logged, never its message or any
        # argument — this receipt carries real provider content and the log
        # must stay content-free even when the projection fails.
        _logger.warning(
            "context composition receipt enrichment failed, receipt left "
            "unenriched: %s",
            type(exc).__name__,
        )
        return sanitized_receipt


__all__ = [
    "CONTEXT_COMPOSITION_EXTENSION_KEY",
    "CONTEXT_COMPOSITION_EXTENSION_SCHEMA",
    "CONTEXT_COMPOSITION_METHOD",
    "ContextCompositionBootstrapHarness",
    "ContextCompositionContractError",
    "INTERNAL_CONTEXT_COMPOSITION_SCHEMA",
    "build_context_composition_extension",
    "build_internal_context_composition",
    "enrich_provider_call_receipt",
    "freeze_internal_context_composition",
    "validate_private_context_composition_hint",
]
