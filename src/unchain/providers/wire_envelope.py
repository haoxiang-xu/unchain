"""Immutable, canonical descriptions of final provider wire requests.

The records in this module contain plain JSON data only.  They perform no
network, journal, artifact, or provider-client work; those capabilities belong
to host adapters outside this contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

from unchain.context.tool_catalog import ToolCatalogEnvelope
from unchain.journal.models import (
    AttemptRef,
    ModelValidationError,
    _bounded_int,
    _sha256,
)
from unchain.journal.resource_limits import BoundaryResourceLimitError

from .message_contract import (
    validate_anthropic_messages,
    validate_ollama_messages,
    validate_openai_input_items,
)


# Transport safety ceilings.  These do not express model context policy.
MAX_PROVIDER_WIRE_BYTES = 64 * 1024 * 1024
MAX_PROVIDER_WIRE_DEPTH = 64
MAX_PROVIDER_WIRE_NODES = 1_000_000
MAX_PROVIDER_WIRE_CONTAINER_ITEMS = 250_000
MAX_PROVIDER_WIRE_STRING_BYTES = 32 * 1024 * 1024
MAX_PROVIDER_WIRE_ROUTES = 2

MAX_PROVIDER_WIRE_BETAS = 64
MAX_PROVIDER_WIRE_BETA_BYTES = 256
MAX_PROVIDER_WIRE_BETAS_BYTES = 16 * 1024

_JSON_STRING_ESCAPE_PATTERN = re.compile(r'["\\\x00-\x1f]')
_JSON_SHORT_ESCAPES = frozenset({'"', "\\", "\b", "\t", "\n", "\f", "\r"})
_EPHEMERAL = {"type": "ephemeral"}
_ROUTE_NAMES = frozenset({"primary", "openai_previous_response_fallback"})
_PROVIDER_PROFILES = MappingProxyType(
    {
        "gemini": (
            "unchain.gemini.contents.request.v1",
            "gemini.models.generate_content_stream",
        ),
        "openai": (
            "unchain.openai.responses.request.v1",
            "openai.responses.create",
        ),
        "anthropic": (
            "unchain.anthropic.messages.request.v1",
            "anthropic.messages.stream",
        ),
        "hyperspace": (
            "unchain.hyperspace.anthropic-messages.request.v1",
            "hyperspace.anthropic.messages.stream",
        ),
        "ollama": (
            "unchain.ollama.chat.request.v1",
            "ollama.api.chat.post",
        ),
    }
)
_FORBIDDEN_REQUEST_ROOTS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "base_url",
        "client",
        "credential",
        "credentials",
        "headers",
        "password",
        "secret",
        "token",
    }
)
_FORBIDDEN_HEADER_NAMES = frozenset(
    {
        "api-key",
        "authorization",
        "cookie",
        "proxy-authorization",
        "set-cookie",
        "x-api-key",
    }
)


class ProviderWireContractError(ModelValidationError):
    """A provider wire record is invalid, inconsistent, or corrupt."""


def _canonical_bytes(value: Any, *, boundary: str) -> bytes:
    _preflight_json(value, boundary=boundary)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise ProviderWireContractError(
            f"{boundary} must be strict canonical JSON"
        ) from exc


def _canonical_sha256(value: Any, *, boundary: str) -> str:
    return hashlib.sha256(_canonical_bytes(value, boundary=boundary)).hexdigest()


def _canonical_text(value: object, field_name: str, *, maximum: int = 512) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be exact text")
    normalized = unicodedata.normalize("NFC", value.strip())
    if (
        normalized != value
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise ProviderWireContractError(f"{field_name} is not canonical text")
    return value


def _canonical_provider(value: object) -> str:
    provider = _canonical_text(value, "provider", maximum=128)
    if provider not in _PROVIDER_PROFILES:
        raise ProviderWireContractError("provider is not supported")
    return provider


def _reject_resource_limit(
    *,
    boundary: str,
    dimension: str,
    limit: int,
    observed: int,
) -> None:
    raise BoundaryResourceLimitError(
        boundary=boundary,
        dimension=dimension,
        limit=limit,
        observed=observed,
    )


def _preflight_json(value: object, *, boundary: str) -> None:
    """Validate exact JSON and all resource limits before serialization."""

    node_count = 0
    byte_count = 0
    active_containers: set[int] = set()
    stack: list[tuple[str, object, int]] = [("enter", value, 0)]

    def reject(dimension: str, limit: int, observed: int) -> None:
        _reject_resource_limit(
            boundary=boundary,
            dimension=dimension,
            limit=limit,
            observed=observed,
        )

    def add_bytes(amount: int) -> None:
        nonlocal byte_count
        byte_count += amount
        if byte_count > MAX_PROVIDER_WIRE_BYTES:
            reject("bytes", MAX_PROVIDER_WIRE_BYTES, byte_count)

    def add_string(text: str) -> None:
        character_count = len(text)
        if character_count > MAX_PROVIDER_WIRE_STRING_BYTES:
            reject(
                "string_bytes",
                MAX_PROVIDER_WIRE_STRING_BYTES,
                character_count,
            )
        raw_byte_count = 0
        encoded_byte_count = 2
        for offset in range(0, character_count, 64 * 1024):
            chunk = text[offset : offset + 64 * 1024]
            try:
                raw_chunk = chunk.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ProviderWireContractError(
                    f"{boundary} contains invalid Unicode"
                ) from exc
            raw_byte_count += len(raw_chunk)
            if raw_byte_count > MAX_PROVIDER_WIRE_STRING_BYTES:
                reject(
                    "string_bytes",
                    MAX_PROVIDER_WIRE_STRING_BYTES,
                    raw_byte_count,
                )
            encoded_byte_count += len(raw_chunk)
            for match in _JSON_STRING_ESCAPE_PATTERN.finditer(chunk):
                encoded_byte_count += 1 if match.group(0) in _JSON_SHORT_ESCAPES else 5
            if byte_count + encoded_byte_count > MAX_PROVIDER_WIRE_BYTES:
                reject(
                    "bytes",
                    MAX_PROVIDER_WIRE_BYTES,
                    byte_count + encoded_byte_count,
                )
        add_bytes(encoded_byte_count)

    while stack:
        action, item, depth = stack.pop()
        if action == "exit":
            active_containers.remove(id(item))
            continue
        if depth > MAX_PROVIDER_WIRE_DEPTH:
            reject("depth", MAX_PROVIDER_WIRE_DEPTH, depth)
        node_count += 1
        if node_count > MAX_PROVIDER_WIRE_NODES:
            reject("nodes", MAX_PROVIDER_WIRE_NODES, node_count)

        item_type = type(item)
        if item is None:
            add_bytes(4)
            continue
        if item_type is bool:
            add_bytes(4 if item else 5)
            continue
        if item_type is str:
            add_string(item)
            continue
        if item_type is int:
            try:
                add_bytes(len(str(item)))
            except ValueError as exc:
                raise ProviderWireContractError(
                    f"{boundary} contains an invalid integer"
                ) from exc
            continue
        if item_type is float:
            if not math.isfinite(item):
                raise ProviderWireContractError(
                    f"{boundary} contains a non-finite number"
                )
            add_bytes(len(repr(item)))
            continue
        if item_type not in {dict, list}:
            raise TypeError(f"{boundary} requires exact JSON value types")

        item_count = len(item)
        if item_count > MAX_PROVIDER_WIRE_CONTAINER_ITEMS:
            reject(
                "container_items",
                MAX_PROVIDER_WIRE_CONTAINER_ITEMS,
                item_count,
            )
        identity = id(item)
        if identity in active_containers:
            raise ProviderWireContractError(
                f"{boundary} contains a circular JSON value"
            )
        active_containers.add(identity)
        stack.append(("exit", item, depth))
        add_bytes(2 + max(0, item_count - 1))

        if item_type is dict:
            children: list[object] = []
            for key, child in item.items():
                if type(key) is not str:
                    raise TypeError(f"{boundary} requires exact JSON object keys")
                add_string(key)
                add_bytes(1)
                children.append(child)
            for child in reversed(children):
                stack.append(("enter", child, depth + 1))
            continue
        for child in reversed(item):
            stack.append(("enter", child, depth + 1))


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json(child) for key, child in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _strict_json_object(value: object, *, boundary: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{boundary} must be an exact JSON object")
    _preflight_json(value, boundary=boundary)
    try:
        copied = json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise ProviderWireContractError(
            f"{boundary} must be detached strict JSON"
        ) from exc
    frozen = _freeze_json(copied)
    if not isinstance(frozen, Mapping):
        raise ProviderWireContractError(f"{boundary} must remain an object")
    return frozen


def _strict_record(
    value: object,
    *,
    schema: str,
    fields: frozenset[str],
    record_name: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{record_name} must be an exact dict")
    expected = fields | {"schema"}
    if set(value) != expected:
        raise ProviderWireContractError(
            f"{record_name} must use the exact record shape"
        )
    if value.get("schema") != schema:
        raise ProviderWireContractError(f"{record_name} schema is unsupported")
    return value


def _canonical_betas(value: object, *, field_name: str) -> tuple[str, ...]:
    if type(value) not in {list, tuple}:
        raise TypeError(f"{field_name} must be an exact ordered array")
    if len(value) > MAX_PROVIDER_WIRE_BETAS:
        _reject_resource_limit(
            boundary="provider wire betas",
            dimension="items",
            limit=MAX_PROVIDER_WIRE_BETAS,
            observed=len(value),
        )
    output: list[str] = []
    seen: set[str] = set()
    for index, beta in enumerate(value):
        if type(beta) is not str:
            raise TypeError(f"{field_name}[{index}] must be exact text")
        normalized = unicodedata.normalize("NFC", beta.strip())
        if (
            normalized != beta
            or not beta
            or "," in beta
            or any(ord(character) < 32 for character in beta)
        ):
            raise ProviderWireContractError(f"{field_name}[{index}] is invalid")
        try:
            byte_length = len(beta.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ProviderWireContractError(
                f"{field_name}[{index}] contains invalid Unicode"
            ) from exc
        if byte_length > MAX_PROVIDER_WIRE_BETA_BYTES:
            _reject_resource_limit(
                boundary="provider wire betas",
                dimension="string_bytes",
                limit=MAX_PROVIDER_WIRE_BETA_BYTES,
                observed=byte_length,
            )
        if beta in seen:
            raise ProviderWireContractError(f"{field_name} contains duplicates")
        seen.add(beta)
        output.append(beta)
    encoded = _canonical_bytes(output, boundary="provider wire betas")
    if len(encoded) > MAX_PROVIDER_WIRE_BETAS_BYTES:
        _reject_resource_limit(
            boundary="provider wire betas",
            dimension="bytes",
            limit=MAX_PROVIDER_WIRE_BETAS_BYTES,
            observed=len(encoded),
        )
    return tuple(output)


def _merged_betas(base: Sequence[str], required: Sequence[str]) -> tuple[str, ...]:
    merged: list[str] = list(base)
    for beta in required:
        if beta not in merged:
            merged.append(beta)
    return tuple(merged)


def _validate_headers(
    *,
    provider: str,
    request: dict[str, Any],
    base_betas: Sequence[str],
    required_betas: Sequence[str],
) -> None:
    raw_headers = request.get("extra_headers")
    if raw_headers is None:
        headers: dict[str, Any] = {}
    elif type(raw_headers) is dict:
        headers = raw_headers
    else:
        raise TypeError("extra_headers must be an exact JSON object")

    for name, value in headers.items():
        if type(name) is not str or type(value) is not str:
            raise TypeError("provider headers must contain exact text")
        lowered = name.casefold()
        if lowered in _FORBIDDEN_HEADER_NAMES:
            raise ProviderWireContractError(
                "provider header contains credential material"
            )
        if provider not in {"anthropic", "hyperspace"} or name != "anthropic-beta":
            raise ProviderWireContractError("provider header is not allowlisted")
        if any(ord(character) < 32 for character in value):
            raise ProviderWireContractError("provider header contains control data")

    expected_betas = _merged_betas(base_betas, required_betas)
    if provider in {"anthropic", "hyperspace"}:
        if expected_betas:
            if headers != {"anthropic-beta": ",".join(expected_betas)}:
                raise ProviderWireContractError(
                    "anthropic-beta header does not match the final beta projection"
                )
        elif headers:
            raise ProviderWireContractError(
                "anthropic-beta header is present without declared betas"
            )
    elif headers:
        raise ProviderWireContractError("provider header is not allowlisted")


def _validate_ephemeral_tail(value: object, *, field_name: str) -> None:
    if type(value) is not list or not value:
        raise ProviderWireContractError(
            f"{field_name} must be a non-empty exact array for cache_control"
        )
    tail = value[-1]
    if type(tail) is not dict or tail.get("cache_control") != _EPHEMERAL:
        raise ProviderWireContractError(
            f"{field_name} tail must have ephemeral cache_control"
        )


def _validate_request(
    *,
    provider: str,
    request_model: str,
    request: dict[str, Any],
    base_betas: Sequence[str],
    required_betas: Sequence[str],
) -> None:
    for key in request:
        if key.casefold().replace("-", "_") in _FORBIDDEN_REQUEST_ROOTS:
            raise ProviderWireContractError(
                f"provider request root {key!r} is not allowed"
            )
    if request.get("model") != request_model:
        raise ProviderWireContractError("provider request model is inconsistent")
    _validate_headers(
        provider=provider,
        request=request,
        base_betas=base_betas,
        required_betas=required_betas,
    )

    if provider == "gemini":
        from google.genai import types

        if set(request) not in (
            {"model", "contents", "config"},
            {"model", "contents", "config", "tools"},
        ):
            raise ProviderWireContractError("Unknown Gemini wire field")
        if not isinstance(request["contents"], list) or not request["contents"]:
            raise ProviderWireContractError("Gemini requires contents")
        config = request["config"]
        if (
            not isinstance(config, dict)
            or "tools" in config
            or config.get("automatic_function_calling") != {"disable": True}
        ):
            raise ProviderWireContractError(
                "Gemini tools are catalog-owned and automatic execution is disabled"
            )
        types.GenerateContentConfig.model_validate(config)
        from .gemini import validate_gemini_contents

        validate_gemini_contents(request["contents"])
        for tool in request.get("tools", []):
            types.FunctionDeclaration.model_validate(tool)
        return

    if provider == "openai":
        if request.get("stream") is not True:
            raise ProviderWireContractError("OpenAI stream must be exactly true")
        if type(request.get("input")) is not list:
            raise ProviderWireContractError("OpenAI input must be an exact array")
        validate_openai_input_items(request["input"])
        return

    if provider in {"anthropic", "hyperspace"}:
        if "stream" in request:
            raise ProviderWireContractError(
                "Anthropic-family stream is transport-owned and must be absent"
            )
        messages = request.get("messages")
        if type(messages) is not list or not messages:
            raise ProviderWireContractError(
                "Anthropic-family messages must be a non-empty exact array"
            )
        validate_anthropic_messages(messages)
        max_tokens = request.get("max_tokens")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ProviderWireContractError(
                "Anthropic-family max_tokens must be a positive integer"
            )
        system = request.get("system")
        if system is not None:
            _validate_ephemeral_tail(system, field_name="system")
        last_message = messages[-1]
        if type(last_message) is not dict:
            raise ProviderWireContractError("messages tail must be an exact object")
        _validate_ephemeral_tail(
            last_message.get("content"),
            field_name="messages tail content",
        )
        return

    if request.get("stream") is not True:
        raise ProviderWireContractError("Ollama stream must be exactly true")
    if type(request.get("messages")) is not list:
        raise ProviderWireContractError("Ollama messages must be an exact array")
    validate_ollama_messages(request["messages"])
    tools = request.get("tools")
    if tools is None:
        if "tool_choice" in request:
            raise ProviderWireContractError(
                "Ollama tool_choice must be absent without tools"
            )
    else:
        if type(tools) is not list or not tools:
            raise ProviderWireContractError("Ollama tools must be a non-empty array")
        if request.get("tool_choice") != "auto":
            raise ProviderWireContractError(
                "Ollama tool_choice must be exactly auto when tools exist"
            )


@dataclass(frozen=True, slots=True)
class ProviderWireRoute:
    """One immutable provider request route and its canonical digests."""

    SCHEMA: ClassVar[str] = "unchain.provider_wire_route.v1"

    name: str
    request: Mapping[str, Any] = field(repr=False)
    request_sha256: str = ""
    tools_sha256: str = ""
    headers_sha256: str = ""
    route_sha256: str = ""

    def __post_init__(self) -> None:
        name = _canonical_text(self.name, "route name", maximum=128)
        if name not in _ROUTE_NAMES:
            raise ProviderWireContractError("route name is unsupported")
        request = _strict_json_object(
            self.request,
            boundary="provider wire request",
        )
        request_plain = _thaw_json(request)
        tools = request_plain.get("tools", [])
        headers = request_plain.get("extra_headers", {})
        if type(tools) is not list:
            raise TypeError("provider wire tools must be an exact array")
        if type(headers) is not dict:
            raise TypeError("provider wire headers must be an exact object")

        request_sha256 = _canonical_sha256(
            request_plain,
            boundary="provider wire request",
        )
        tools_sha256 = _canonical_sha256(
            tools,
            boundary="provider wire tools",
        )
        headers_sha256 = _canonical_sha256(
            headers,
            boundary="provider wire headers",
        )
        route_body = {
            "schema": self.SCHEMA,
            "name": name,
            "request": request_plain,
            "request_sha256": request_sha256,
            "tools_sha256": tools_sha256,
            "headers_sha256": headers_sha256,
        }
        route_sha256 = _canonical_sha256(
            route_body,
            boundary="provider wire route",
        )

        for field_name, supplied, expected in (
            ("request_sha256", self.request_sha256, request_sha256),
            ("tools_sha256", self.tools_sha256, tools_sha256),
            ("headers_sha256", self.headers_sha256, headers_sha256),
            ("route_sha256", self.route_sha256, route_sha256),
        ):
            if supplied:
                digest = _sha256(supplied, field_name)
                if digest != expected:
                    raise ProviderWireContractError(
                        f"{field_name} digest does not match route contents"
                    )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "request_sha256", request_sha256)
        object.__setattr__(self, "tools_sha256", tools_sha256)
        object.__setattr__(self, "headers_sha256", headers_sha256)
        object.__setattr__(self, "route_sha256", route_sha256)

    def request_copy(self) -> dict[str, Any]:
        value = _thaw_json(self.request)
        if type(value) is not dict:
            raise ProviderWireContractError("provider wire request changed")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "name": self.name,
            "request": self.request_copy(),
            "request_sha256": self.request_sha256,
            "tools_sha256": self.tools_sha256,
            "headers_sha256": self.headers_sha256,
            "route_sha256": self.route_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict(), boundary="provider wire route")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProviderWireRoute:
        fields = frozenset(
            {
                "name",
                "request",
                "request_sha256",
                "tools_sha256",
                "headers_sha256",
                "route_sha256",
            }
        )
        raw = _strict_record(
            value,
            schema=cls.SCHEMA,
            fields=fields,
            record_name="provider wire route",
        )
        return cls(**{field_name: raw[field_name] for field_name in fields})


@dataclass(frozen=True, slots=True)
class ProviderWireEnvelope:
    """Canonical final-wire routes bound to one tool catalog and attempt."""

    SCHEMA: ClassVar[str] = "unchain.provider_wire_envelope.v1"

    attempt: AttemptRef
    iteration: int
    provider: str
    configured_model: str
    request_model: str
    adapter_revision: str
    transport_kind: str
    transport_target_sha256: str
    source_request_sha256: str
    source_payload_sha256: str
    catalog_sha256: str
    prompt_sha256: str
    tool_schema_sha256: str
    required_betas: Sequence[str]
    base_anthropic_betas: Sequence[str]
    routes: Sequence[ProviderWireRoute]
    required_betas_sha256: str = ""
    envelope_sha256: str = ""

    def __post_init__(self) -> None:
        attempt = self.attempt
        if type(attempt) is dict:
            attempt = AttemptRef.from_dict(attempt)
        elif type(attempt) is AttemptRef:
            attempt = AttemptRef.from_dict(attempt.to_dict())
        else:
            raise TypeError("attempt must be an exact AttemptRef")

        iteration = _bounded_int(self.iteration, "iteration")
        if iteration > 2**31 - 1:
            raise ProviderWireContractError("iteration exceeds provider wire limit")
        provider = _canonical_provider(self.provider)
        configured_model = _canonical_text(
            self.configured_model,
            "configured_model",
        )
        request_model = _canonical_text(self.request_model, "request_model")
        adapter_revision = _canonical_text(
            self.adapter_revision,
            "adapter_revision",
        )
        transport_kind = _canonical_text(self.transport_kind, "transport_kind")
        expected_profile = _PROVIDER_PROFILES[provider]
        if (adapter_revision, transport_kind) != expected_profile:
            if adapter_revision != expected_profile[0]:
                raise ProviderWireContractError(
                    "adapter_revision does not match the provider profile"
                )
            raise ProviderWireContractError(
                "transport_kind does not match the provider profile"
            )

        digest_fields: dict[str, str] = {}
        for field_name in (
            "transport_target_sha256",
            "source_request_sha256",
            "source_payload_sha256",
            "catalog_sha256",
            "prompt_sha256",
            "tool_schema_sha256",
        ):
            digest_fields[field_name] = _sha256(
                getattr(self, field_name),
                field_name,
            )

        required_betas = _canonical_betas(
            self.required_betas,
            field_name="required_betas",
        )
        base_betas = _canonical_betas(
            self.base_anthropic_betas,
            field_name="base_anthropic_betas",
        )
        if provider not in {"anthropic", "hyperspace"} and (
            required_betas or base_betas
        ):
            raise ProviderWireContractError(
                "betas are only valid for Anthropic-family providers"
            )
        required_betas_sha256 = _canonical_sha256(
            list(required_betas),
            boundary="provider wire required betas",
        )
        if self.required_betas_sha256:
            supplied_beta_sha256 = _sha256(
                self.required_betas_sha256,
                "required_betas_sha256",
            )
            if supplied_beta_sha256 != required_betas_sha256:
                raise ProviderWireContractError(
                    "required_betas_sha256 digest does not match required_betas"
                )

        if type(self.routes) not in {list, tuple}:
            raise TypeError("routes must be an exact ordered array")
        if not self.routes or len(self.routes) > MAX_PROVIDER_WIRE_ROUTES:
            raise ProviderWireContractError(
                f"routes must contain one or at most {MAX_PROVIDER_WIRE_ROUTES} entries"
            )
        routes: list[ProviderWireRoute] = []
        for route in self.routes:
            if type(route) is ProviderWireRoute:
                routes.append(ProviderWireRoute.from_dict(route.to_dict()))
            elif type(route) is dict:
                routes.append(ProviderWireRoute.from_dict(route))
            else:
                raise TypeError("routes require exact ProviderWireRoute records")
        route_names = tuple(route.name for route in routes)
        if len(route_names) != len(set(route_names)):
            raise ProviderWireContractError("routes must have unique names")
        if route_names[0] != "primary":
            raise ProviderWireContractError("routes must start with primary")
        if provider != "openai" and route_names != ("primary",):
            raise ProviderWireContractError(
                "provider does not support the OpenAI fallback route"
            )

        request_bodies = [route.request_copy() for route in routes]
        for request in request_bodies:
            _validate_request(
                provider=provider,
                request_model=request_model,
                request=request,
                base_betas=base_betas,
                required_betas=required_betas,
            )
        if provider == "openai":
            self._validate_openai_routes(route_names, request_bodies)

        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "configured_model", configured_model)
        object.__setattr__(self, "request_model", request_model)
        object.__setattr__(self, "adapter_revision", adapter_revision)
        object.__setattr__(self, "transport_kind", transport_kind)
        for field_name, digest in digest_fields.items():
            object.__setattr__(self, field_name, digest)
        object.__setattr__(self, "required_betas", required_betas)
        object.__setattr__(self, "base_anthropic_betas", base_betas)
        object.__setattr__(self, "routes", tuple(routes))
        object.__setattr__(
            self,
            "required_betas_sha256",
            required_betas_sha256,
        )

        envelope_sha256 = _canonical_sha256(
            self._envelope_body(),
            boundary="provider wire envelope",
        )
        if self.envelope_sha256:
            supplied_envelope_sha256 = _sha256(
                self.envelope_sha256,
                "envelope_sha256",
            )
            if supplied_envelope_sha256 != envelope_sha256:
                raise ProviderWireContractError(
                    "envelope_sha256 digest does not match envelope contents"
                )
        object.__setattr__(self, "envelope_sha256", envelope_sha256)

    @staticmethod
    def _validate_openai_routes(
        route_names: tuple[str, ...],
        requests: list[dict[str, Any]],
    ) -> None:
        primary = requests[0]
        has_previous = "previous_response_id" in primary
        if has_previous:
            previous_response_id = primary.get("previous_response_id")
            if type(previous_response_id) is not str or not previous_response_id:
                raise ProviderWireContractError(
                    "previous_response_id must be non-empty exact text"
                )
            if route_names != (
                "primary",
                "openai_previous_response_fallback",
            ):
                raise ProviderWireContractError(
                    "previous_response_id requires the OpenAI fallback route"
                )
            fallback = requests[1]
            if "previous_response_id" in fallback:
                raise ProviderWireContractError(
                    "OpenAI fallback must omit previous_response_id"
                )
            fallback_input = fallback.get("input")
            if type(fallback_input) is not list or not fallback_input:
                raise ProviderWireContractError(
                    "OpenAI fallback input must be a non-empty exact array"
                )
            primary_common = dict(primary)
            fallback_common = dict(fallback)
            primary_common.pop("previous_response_id")
            primary_common.pop("input")
            fallback_common.pop("input")
            if primary_common != fallback_common:
                raise ProviderWireContractError(
                    "OpenAI fallback changed common wire fields"
                )
            return
        if route_names != ("primary",):
            raise ProviderWireContractError(
                "OpenAI fallback requires primary previous_response_id"
            )

    def _envelope_body(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "attempt": self.attempt.to_dict(),
            "iteration": self.iteration,
            "provider": self.provider,
            "configured_model": self.configured_model,
            "request_model": self.request_model,
            "adapter_revision": self.adapter_revision,
            "transport_kind": self.transport_kind,
            "transport_target_sha256": self.transport_target_sha256,
            "source_request_sha256": self.source_request_sha256,
            "source_payload_sha256": self.source_payload_sha256,
            "catalog_sha256": self.catalog_sha256,
            "prompt_sha256": self.prompt_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
            "required_betas": list(self.required_betas),
            "base_anthropic_betas": list(self.base_anthropic_betas),
            "required_betas_sha256": self.required_betas_sha256,
            "routes": [route.to_dict() for route in self.routes],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._envelope_body(),
            "envelope_sha256": self.envelope_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(
            self.to_dict(),
            boundary="provider wire envelope",
        )

    def request_copy(self, route_name: str = "primary") -> dict[str, Any]:
        name = _canonical_text(route_name, "route name", maximum=128)
        for route in self.routes:
            if route.name == name:
                return route.request_copy()
        raise ProviderWireContractError(f"provider wire route {name!r} was not found")

    def verify_against_catalog(
        self,
        catalog: ToolCatalogEnvelope,
    ) -> ProviderWireEnvelope:
        if type(catalog) is not ToolCatalogEnvelope:
            raise TypeError("catalog must be an exact ToolCatalogEnvelope")
        if catalog.attempt.to_dict() != self.attempt.to_dict():
            raise ProviderWireContractError("catalog attempt does not match envelope")
        if catalog.iteration != self.iteration:
            raise ProviderWireContractError("catalog iteration does not match envelope")
        if catalog.provider != self.provider:
            raise ProviderWireContractError("catalog provider does not match envelope")
        if catalog.model != self.configured_model:
            raise ProviderWireContractError("catalog model does not match envelope")
        if catalog.catalog_sha256 != self.catalog_sha256:
            raise ProviderWireContractError("catalog digest does not match envelope")
        if catalog.prompt_sha256 != self.prompt_sha256:
            raise ProviderWireContractError(
                "catalog prompt digest does not match envelope"
            )
        if catalog.tool_schema_sha256 != self.tool_schema_sha256:
            raise ProviderWireContractError(
                "catalog tool schema digest does not match envelope"
            )
        if catalog.required_betas_sha256 != self.required_betas_sha256:
            raise ProviderWireContractError(
                "catalog required betas digest does not match envelope"
            )

        expected_tools = [_thaw_json(schema) for schema in catalog.provider_schemas]
        if expected_tools and self.provider in {"anthropic", "hyperspace"}:
            expected_tools[-1]["cache_control"] = dict(_EPHEMERAL)
        for route in self.routes:
            request = route.request_copy()
            if expected_tools:
                if request.get("tools") != expected_tools:
                    raise ProviderWireContractError(
                        "provider wire tools do not match the catalog"
                    )
            elif "tools" in request:
                raise ProviderWireContractError(
                    "provider wire tools do not match the empty catalog"
                )
        return self

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProviderWireEnvelope:
        fields = frozenset(
            {
                "attempt",
                "iteration",
                "provider",
                "configured_model",
                "request_model",
                "adapter_revision",
                "transport_kind",
                "transport_target_sha256",
                "source_request_sha256",
                "source_payload_sha256",
                "catalog_sha256",
                "prompt_sha256",
                "tool_schema_sha256",
                "required_betas",
                "base_anthropic_betas",
                "required_betas_sha256",
                "routes",
                "envelope_sha256",
            }
        )
        raw = _strict_record(
            value,
            schema=cls.SCHEMA,
            fields=fields,
            record_name="provider wire envelope",
        )
        return cls(**{field_name: raw[field_name] for field_name in fields})


__all__ = [
    "MAX_PROVIDER_WIRE_BETA_BYTES",
    "MAX_PROVIDER_WIRE_BETAS",
    "MAX_PROVIDER_WIRE_BETAS_BYTES",
    "MAX_PROVIDER_WIRE_BYTES",
    "MAX_PROVIDER_WIRE_CONTAINER_ITEMS",
    "MAX_PROVIDER_WIRE_DEPTH",
    "MAX_PROVIDER_WIRE_NODES",
    "MAX_PROVIDER_WIRE_ROUTES",
    "MAX_PROVIDER_WIRE_STRING_BYTES",
    "ProviderWireContractError",
    "ProviderWireEnvelope",
    "ProviderWireRoute",
]
