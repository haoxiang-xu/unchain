from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar

from unchain.journal.models import (
    ArtifactRef,
    AttemptRef,
    EventCursor,
    ModelValidationError,
    _bounded_int,
    _credential_metadata_kind,
    _normalize_key,
    _record_data,
    _required_text,
    _sha256,
    _thaw_json,
    _validate_credential_metadata,
)
from unchain.journal.resource_limits import (
    BoundaryResourceLimitError,
    JsonResourceLimits,
    enforce_item_limit,
    validate_json_resource,
)
from unchain.tools.handler_registry import (
    DurableToolHandlerBinding,
    DurableToolHandlerRegistry,
    DurableToolHandlerResolution,
)


class ToolCatalogContractError(ModelValidationError):
    """A tool catalog record is incomplete, ambiguous, or corrupt."""


class ToolCatalogRouteKind(StrEnum):
    NORMAL = "normal"
    PLUGIN = "plugin"


MAX_TOOL_CATALOG_ITERATION = 2**31 - 1
MAX_TOOL_CATALOG_ENTRIES = 256
MAX_TOOL_SCHEMA_BYTES = 64 * 1024
MAX_TOOL_CATALOG_BYTES = 1024 * 1024
MAX_TOOL_CATALOG_ARTIFACT_BYTES = 2 * 1024 * 1024
MAX_TOOL_SCHEMA_CONTAINER_ITEMS = 1024
MAX_TOOL_SCHEMA_STRING_BYTES = 16 * 1024
TOOL_CATALOG_JSON_LIMITS = JsonResourceLimits(
    max_items=MAX_TOOL_CATALOG_ENTRIES,
    max_bytes=MAX_TOOL_CATALOG_BYTES,
    max_depth=32,
    max_nodes=50_000,
)
TOOL_SCHEMA_JSON_LIMITS = JsonResourceLimits(
    max_items=MAX_TOOL_SCHEMA_CONTAINER_ITEMS,
    max_bytes=MAX_TOOL_SCHEMA_BYTES,
    max_depth=TOOL_CATALOG_JSON_LIMITS.max_depth,
    max_nodes=TOOL_CATALOG_JSON_LIMITS.max_nodes,
)
SUPPORTED_TOOL_CATALOG_PROVIDERS = frozenset(
    {"openai", "anthropic", "hyperspace", "ollama", "gemini"}
)
_SCHEMA_PROPERTY_CONTAINERS = frozenset({"properties", "patternProperties"})
_SCHEMA_CHILD_KEYS = frozenset(
    {
        "additionalProperties",
        "allOf",
        "anyOf",
        "contains",
        "else",
        "if",
        "items",
        "not",
        "oneOf",
        "prefixItems",
        "propertyNames",
        "then",
        "unevaluatedProperties",
    }
)
_SECRET_SCHEMA_REVISION = 1
_SECRET_SCHEMA_KINDS = frozenset({"handle", "handle_map"})


def _secret_schema_template(kind: str) -> dict[str, Any] | None:
    if kind == "handle":
        return {
            "type": "string",
            "x-pupu-secret": True,
            "x-pupu-secret-kind": "handle",
        }
    if kind == "handle_map":
        return {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "x-pupu-secret": True,
            "x-pupu-secret-kind": "handle_map",
        }
    return None


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ToolCatalogContractError(
            "tool catalog value must be strict canonical JSON"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json_shape(value: Any, *, path: str) -> Any:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        if value_type is str and unicodedata.normalize("NFC", value) != value:
            raise ToolCatalogContractError(f"{path} must be strict canonical JSON")
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ToolCatalogContractError(f"{path} must be strict canonical JSON")
        return value
    if value_type is list:
        return [
            _strict_json_shape(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value_type is dict:
        output: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ToolCatalogContractError(f"{path} must be strict canonical JSON")
            if unicodedata.normalize("NFC", key) != key:
                raise ToolCatalogContractError(f"{path} must be strict canonical JSON")
            output[key] = _strict_json_shape(item, path=f"{path}.{key}")
        return output
    raise ToolCatalogContractError(f"{path} must be strict canonical JSON")


def _reject_schema_limit(
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


def _validate_schema_shape_limits(value: Any, *, boundary: str) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is str:
            try:
                byte_length = len(item.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise ToolCatalogContractError(
                    f"{boundary} contains invalid Unicode"
                ) from exc
            if byte_length > MAX_TOOL_SCHEMA_STRING_BYTES:
                _reject_schema_limit(
                    boundary=boundary,
                    dimension="string_bytes",
                    limit=MAX_TOOL_SCHEMA_STRING_BYTES,
                    observed=byte_length,
                )
            continue
        if type(item) is dict:
            if len(item) > MAX_TOOL_SCHEMA_CONTAINER_ITEMS:
                _reject_schema_limit(
                    boundary=boundary,
                    dimension="container_items",
                    limit=MAX_TOOL_SCHEMA_CONTAINER_ITEMS,
                    observed=len(item),
                )
            for key, child in item.items():
                stack.append(key)
                stack.append(child)
            continue
        if type(item) is list:
            if len(item) > MAX_TOOL_SCHEMA_CONTAINER_ITEMS:
                _reject_schema_limit(
                    boundary=boundary,
                    dimension="container_items",
                    limit=MAX_TOOL_SCHEMA_CONTAINER_ITEMS,
                    observed=len(item),
                )
            stack.extend(item)


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _is_provider_parameter_root(
    provider: str,
    location: tuple[str | int, ...],
) -> bool:
    if provider in {"openai", "gemini"}:
        return location == ("parameters",)
    if provider in {"anthropic", "hyperspace"}:
        return location == ("input_schema",)
    if provider == "ollama":
        return location == ("function", "parameters")
    return False


def _freeze_known_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_known_json(item) for key, item in sorted(value.items())}
        )
    if type(value) is list:
        return tuple(_freeze_known_json(item) for item in value)
    return value


def _freeze_schema_json(
    value: Any,
    *,
    path: str,
    pointer: str,
    secret_manifest: list[dict[str, Any]],
    provider: str,
    location: tuple[str | int, ...],
    in_parameter_schema: bool = False,
    allow_secret_marker: bool = False,
) -> Any:
    if type(value) is dict:
        if "x-pupu-secret" in value:
            if not allow_secret_marker:
                raise ToolCatalogContractError(
                    f"{path} secret marker is only allowed on a parameter property schema"
                )
            marker = value["x-pupu-secret"]
            if marker is not True:
                raise ToolCatalogContractError(
                    f"{path}.x-pupu-secret secret marker must be exactly true"
                )
            kind = value.get("x-pupu-secret-kind")
            template = _secret_schema_template(kind)
            if template is None:
                raise ToolCatalogContractError(
                    f"{path} secret schema kind must be handle or handle_map"
                )
            if value != template:
                raise ToolCatalogContractError(
                    f"{path} secret schema violates the fixed allowlist template"
                )
            canonical_template = _strict_json_shape(
                template,
                path=f"{path}.system_template",
            )
            secret_manifest.append(
                {
                    "path": pointer,
                    "kind": kind,
                    "revision": _SECRET_SCHEMA_REVISION,
                    "schema_sha256": _canonical_sha256(canonical_template),
                }
            )
            return _freeze_known_json(canonical_template)
        if "x-pupu-secret-kind" in value:
            raise ToolCatalogContractError(
                f"{path}.x-pupu-secret-kind requires an explicit x-pupu-secret marker"
            )

        frozen: dict[str, Any] = {}
        for key in sorted(value):
            item = value[key]
            normalized_key = _normalize_key(key)
            child_pointer = f"{pointer}/{_pointer_token(key)}"
            child_location = (*location, key)
            if key in _SCHEMA_PROPERTY_CONTAINERS:
                if type(item) is not dict:
                    raise ToolCatalogContractError(
                        f"{path}.{key} must be strict canonical JSON"
                    )
                properties: dict[str, Any] = {}
                for property_name in sorted(item):
                    property_schema = item[property_name]
                    if type(property_schema) is not dict:
                        raise ToolCatalogContractError(
                            f"{path}.{key}.{property_name} must be a schema object"
                        )
                    if (
                        in_parameter_schema
                        and _credential_metadata_kind(_normalize_key(property_name))
                        == "plaintext"
                        and "x-pupu-secret" not in property_schema
                    ):
                        raise ToolCatalogContractError(
                            f"{path}.{key}.{property_name} requires an explicit "
                            "x-pupu-secret marker"
                        )
                    properties[property_name] = _freeze_schema_json(
                        property_schema,
                        path=f"{path}.{key}.{property_name}",
                        pointer=(f"{child_pointer}/{_pointer_token(property_name)}"),
                        secret_manifest=secret_manifest,
                        provider=provider,
                        location=(*child_location, property_name),
                        in_parameter_schema=in_parameter_schema,
                        allow_secret_marker=in_parameter_schema,
                    )
                frozen[key] = MappingProxyType(properties)
                continue
            _validate_credential_metadata(
                normalized_key,
                item,
                path=f"{path}.{key}",
            )
            frozen[key] = _freeze_schema_json(
                item,
                path=f"{path}.{key}",
                pointer=child_pointer,
                secret_manifest=secret_manifest,
                provider=provider,
                location=child_location,
                in_parameter_schema=(
                    _is_provider_parameter_root(provider, child_location)
                    or (in_parameter_schema and key in _SCHEMA_CHILD_KEYS)
                ),
            )
        return MappingProxyType(frozen)
    if type(value) is list:
        return tuple(
            _freeze_schema_json(
                item,
                path=f"{path}[{index}]",
                pointer=f"{pointer}/{index}",
                secret_manifest=secret_manifest,
                provider=provider,
                location=(*location, index),
                in_parameter_schema=in_parameter_schema,
            )
            for index, item in enumerate(value)
        )
    return value


def _strict_json_copy(
    value: Any,
    *,
    path: str,
    pointer: str,
    provider: str,
) -> tuple[Any, tuple[dict[str, Any], ...]]:
    try:
        validate_json_resource(
            value,
            boundary=path,
            limits=TOOL_SCHEMA_JSON_LIMITS,
        )
    except BoundaryResourceLimitError:
        raise
    except (TypeError, ValueError) as exc:
        raise ToolCatalogContractError(f"{path} must be strict canonical JSON") from exc
    _validate_schema_shape_limits(value, boundary=path)
    shaped = _strict_json_shape(value, path=path)
    secret_manifest: list[dict[str, Any]] = []
    frozen = _freeze_schema_json(
        shaped,
        path=path,
        pointer=pointer,
        secret_manifest=secret_manifest,
        provider=provider,
        location=(),
    )
    _canonical_bytes(_thaw_json(frozen))
    return frozen, tuple(secret_manifest)


def _strict_record(
    value: Any,
    *,
    schema: str,
    required: frozenset[str],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("tool catalog record must be an exact dict")
    return _record_data(value, schema=schema, required=required)


def _canonical_text(
    value: Any,
    field_name: str,
    *,
    identifier: bool = False,
    maximum: int = 256,
) -> str:
    normalized = _required_text(
        value,
        field_name,
        identifier=identifier,
        maximum=maximum,
    )
    if normalized != value:
        raise ToolCatalogContractError(f"{field_name} is not canonical text")
    return normalized


def _canonical_provider(value: Any) -> str:
    provider = _canonical_text(value, "provider", maximum=128)
    if provider not in SUPPORTED_TOOL_CATALOG_PROVIDERS:
        raise ToolCatalogContractError(
            "provider is not supported by the tool catalog contract"
        )
    return provider


def _canonical_secret_schema_manifest(
    value: Any,
) -> tuple[Mapping[str, Any], ...]:
    if type(value) not in {list, tuple}:
        raise TypeError("secret_schema_manifest must be an ordered array")
    enforce_item_limit(
        value,
        boundary="secret schema manifest",
        limits=TOOL_CATALOG_JSON_LIMITS,
    )
    validate_json_resource(
        value,
        boundary="secret schema manifest",
        limits=TOOL_CATALOG_JSON_LIMITS,
    )
    canonical: list[Mapping[str, Any]] = []
    seen_paths: set[str] = set()
    required = {"path", "kind", "revision", "schema_sha256"}
    for index, item in enumerate(value):
        if type(item) is not dict or set(item) != required:
            raise ToolCatalogContractError(
                f"secret_schema_manifest[{index}] must use the exact manifest shape"
            )
        path = _canonical_text(
            item["path"],
            f"secret_schema_manifest[{index}].path",
            maximum=4096,
        )
        if not path.startswith("/semantic_schemas/"):
            raise ToolCatalogContractError(
                f"secret_schema_manifest[{index}].path is not a catalog JSON Pointer"
            )
        kind = item["kind"]
        if kind not in _SECRET_SCHEMA_KINDS:
            raise ToolCatalogContractError(
                f"secret_schema_manifest[{index}].kind is invalid"
            )
        revision = _bounded_int(
            item["revision"],
            f"secret_schema_manifest[{index}].revision",
            minimum=1,
        )
        if revision != _SECRET_SCHEMA_REVISION:
            raise ToolCatalogContractError(
                f"secret_schema_manifest[{index}].revision is unsupported"
            )
        if path in seen_paths:
            raise ToolCatalogContractError(
                "secret_schema_manifest contains duplicate paths"
            )
        seen_paths.add(path)
        canonical.append(
            MappingProxyType(
                {
                    "path": path,
                    "kind": kind,
                    "revision": revision,
                    "schema_sha256": _sha256(
                        item["schema_sha256"],
                        f"secret_schema_manifest[{index}].schema_sha256",
                    ),
                }
            )
        )
    if tuple(item["path"] for item in canonical) != tuple(
        sorted(item["path"] for item in canonical)
    ):
        raise ToolCatalogContractError(
            "secret_schema_manifest paths must be in canonical order"
        )
    return tuple(canonical)


def _schema_name(schema: dict[str, Any], *, provider: str) -> str:
    candidates: list[str] = []
    direct = schema.get("name")
    if isinstance(direct, str) and direct:
        candidates.append(direct)
    function = schema.get("function")
    if isinstance(function, dict):
        nested = function.get("name")
        if isinstance(nested, str) and nested:
            candidates.append(nested)
    if (
        not candidates
        and provider.casefold() == "openai"
        and schema.get("type") == "computer"
    ):
        candidates.append("computer")
    normalized = tuple(
        _canonical_text(candidate, "semantic schema tool name", identifier=True)
        for candidate in candidates
    )
    if not normalized:
        raise ToolCatalogContractError(
            "provider semantic schema requires a stable tool name"
        )
    if len(set(normalized)) != 1:
        raise ToolCatalogContractError(
            "provider semantic schema has ambiguous tool names"
        )
    return normalized[0]


@dataclass(frozen=True)
class ToolCatalogEntry:
    SCHEMA: ClassVar[str] = "unchain.tool_catalog_entry.v1"

    tool_name: str
    semantic_schema_sha256: str
    tool_descriptor_sha256: str
    handler_binding: DurableToolHandlerBinding
    route_kind: ToolCatalogRouteKind | str = ToolCatalogRouteKind.NORMAL

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tool_name",
            _canonical_text(self.tool_name, "tool_name", identifier=True),
        )
        object.__setattr__(
            self,
            "semantic_schema_sha256",
            _sha256(self.semantic_schema_sha256, "semantic_schema_sha256"),
        )
        object.__setattr__(
            self,
            "tool_descriptor_sha256",
            _sha256(self.tool_descriptor_sha256, "tool_descriptor_sha256"),
        )
        binding = self.handler_binding
        if type(binding) not in {dict, DurableToolHandlerBinding}:
            raise TypeError(
                "handler_binding must be an exact DurableToolHandlerBinding"
            )
        try:
            binding_payload = binding if type(binding) is dict else binding.to_dict()
            binding = DurableToolHandlerBinding.from_dict(binding_payload)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ToolCatalogContractError("handler binding is not canonical") from exc
        object.__setattr__(self, "handler_binding", binding)
        try:
            route_kind = ToolCatalogRouteKind(self.route_kind)
        except ValueError as exc:
            raise ToolCatalogContractError("invalid tool catalog route kind") from exc
        object.__setattr__(self, "route_kind", route_kind)
        if (
            route_kind is ToolCatalogRouteKind.PLUGIN
            and binding.route_resolver_id is None
        ):
            raise ToolCatalogContractError(
                "plugin catalog entries require a durable route resolver identity"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "tool_name": self.tool_name,
            "semantic_schema_sha256": self.semantic_schema_sha256,
            "tool_descriptor_sha256": self.tool_descriptor_sha256,
            "route_kind": self.route_kind.value,
            "handler_binding": self.handler_binding.to_dict(),
        }

    @property
    def entry_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ToolCatalogEntry:
        fields = frozenset(
            {
                "tool_name",
                "semantic_schema_sha256",
                "tool_descriptor_sha256",
                "route_kind",
                "handler_binding",
            }
        )
        raw = _strict_record(value, schema=cls.SCHEMA, required=fields)
        return cls(
            tool_name=raw["tool_name"],
            semantic_schema_sha256=raw["semantic_schema_sha256"],
            tool_descriptor_sha256=raw["tool_descriptor_sha256"],
            route_kind=raw["route_kind"],
            handler_binding=raw["handler_binding"],
        )


def build_tool_catalog_entry_from_resolution(
    *,
    registry: DurableToolHandlerRegistry,
    resolution: DurableToolHandlerResolution,
    provider: str,
    route_kind: ToolCatalogRouteKind | str = ToolCatalogRouteKind.NORMAL,
) -> tuple[dict[str, Any], ToolCatalogEntry]:
    if type(registry) is not DurableToolHandlerRegistry:
        raise TypeError("catalog entry builder requires an exact handler registry")
    verified = registry.verify_resolution(resolution)
    provider = _canonical_provider(provider)
    provider_schemas = verified.tool_descriptor.get("provider_schemas")
    if not isinstance(provider_schemas, Mapping):
        raise ToolCatalogContractError(
            "verified tool descriptor has no provider schema authority"
        )
    frozen_schema = provider_schemas.get(provider)
    if provider == "gemini":
        # Derive the new transport projection from already-verified authority.
        # Adding a fifth entry to every handler descriptor would change existing
        # providers' durable identities and invalidate their cold resumes.
        from unchain.providers.gemini_schema import sanitize_gemini_schema

        descriptor = verified.tool_descriptor
        native = (
            descriptor.get("config", {}).get("provider_native_specs", {}).get("gemini")
        )
        if isinstance(native, Mapping):
            frozen_schema = native
        else:
            neutral = _thaw_json(descriptor["provider_neutral_schema"])
            frozen_schema = {
                "name": neutral["name"],
                "description": neutral["description"],
                "parameters": sanitize_gemini_schema(neutral["parameters"]),
            }
    if not isinstance(frozen_schema, Mapping):
        raise ToolCatalogContractError(
            "verified tool descriptor does not support the selected provider"
        )
    schema = _thaw_json(frozen_schema)
    if type(schema) is not dict:
        raise ToolCatalogContractError("verified provider schema must be a JSON object")
    tool_name = _schema_name(schema, provider=provider)
    return schema, ToolCatalogEntry(
        tool_name=tool_name,
        semantic_schema_sha256=_canonical_sha256(schema),
        tool_descriptor_sha256=verified.tool_descriptor_sha256,
        handler_binding=verified.binding,
        route_kind=route_kind,
    )


def verify_tool_catalog_entry_resolution(
    entry: ToolCatalogEntry,
    *,
    registry: DurableToolHandlerRegistry,
    resolution: DurableToolHandlerResolution,
    provider: str,
) -> ToolCatalogEntry:
    if type(entry) is not ToolCatalogEntry:
        raise TypeError("descriptor verification requires an exact catalog entry")
    if type(registry) is not DurableToolHandlerRegistry:
        raise TypeError("descriptor verification requires an exact handler registry")
    verified = registry.verify_resolution(resolution)
    provider = _canonical_provider(provider)
    if entry.handler_binding.to_dict() != verified.binding.to_dict():
        raise ToolCatalogContractError(
            "catalog entry handler binding does not match verified resolution"
        )
    if entry.tool_descriptor_sha256 != verified.tool_descriptor_sha256:
        raise ToolCatalogContractError(
            "catalog entry descriptor digest does not match verified resolution"
        )
    provider_schemas = verified.tool_descriptor.get("provider_schemas")
    frozen_schema = (
        provider_schemas.get(provider)
        if isinstance(provider_schemas, Mapping)
        else None
    )
    if not isinstance(frozen_schema, Mapping):
        raise ToolCatalogContractError(
            "verified tool descriptor does not support the selected provider"
        )
    schema = _thaw_json(frozen_schema)
    if (
        type(schema) is not dict
        or entry.tool_name != _schema_name(schema, provider=provider)
        or entry.semantic_schema_sha256 != _canonical_sha256(schema)
    ):
        raise ToolCatalogContractError(
            "catalog entry semantic schema does not match verified descriptor"
        )
    return entry


@dataclass(frozen=True)
class ToolCatalogEnvelope:
    """Pre-persistence catalog binding one model turn's visible tools."""

    SCHEMA: ClassVar[str] = "unchain.tool_catalog_envelope.v1"

    attempt: AttemptRef
    iteration: int
    provider: str
    model: str
    semantic_schemas: Sequence[dict[str, Any]]
    entries: Sequence[ToolCatalogEntry]
    required_betas_sha256: str
    prompt_sha256: str
    exposure_plan_sha256: str
    secret_schema_manifest: Sequence[Mapping[str, Any]] | None = None
    tool_schema_manifest: Mapping[str, str] | None = None
    tool_schema_sha256: str = ""
    catalog_sha256: str = ""

    def __post_init__(self) -> None:
        attempt = self.attempt
        if type(attempt) is dict:
            attempt = AttemptRef.from_dict(attempt)
        elif type(attempt) is not AttemptRef:
            raise TypeError("attempt must be an exact AttemptRef")
        else:
            attempt = AttemptRef.from_dict(attempt.to_dict())
        object.__setattr__(self, "attempt", attempt)
        iteration = _bounded_int(self.iteration, "iteration")
        if iteration > MAX_TOOL_CATALOG_ITERATION:
            raise ToolCatalogContractError(
                f"iteration exceeds {MAX_TOOL_CATALOG_ITERATION}"
            )
        object.__setattr__(self, "iteration", iteration)
        provider = _canonical_provider(self.provider)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(
            self,
            "model",
            _canonical_text(self.model, "model", maximum=512),
        )

        if type(self.semantic_schemas) not in {list, tuple}:
            raise TypeError("semantic_schemas must be an ordered array")
        enforce_item_limit(
            self.semantic_schemas,
            boundary="tool catalog semantic schemas",
            limits=TOOL_CATALOG_JSON_LIMITS,
        )
        try:
            validate_json_resource(
                self.semantic_schemas,
                boundary="tool catalog semantic schemas",
                limits=TOOL_CATALOG_JSON_LIMITS,
            )
        except BoundaryResourceLimitError:
            raise
        except (TypeError, ValueError) as exc:
            raise ToolCatalogContractError(
                "semantic_schemas must be strict canonical JSON"
            ) from exc
        schemas: list[Mapping[str, Any]] = []
        discovered_secret_manifest: list[dict[str, Any]] = []
        for index, schema in enumerate(self.semantic_schemas):
            if type(schema) is not dict:
                raise ToolCatalogContractError(
                    f"semantic_schemas[{index}] must be strict canonical JSON"
                )
            copied, schema_secret_manifest = _strict_json_copy(
                schema,
                path=f"semantic_schemas[{index}]",
                pointer=f"/semantic_schemas/{index}",
                provider=provider,
            )
            if not isinstance(copied, Mapping):
                raise ToolCatalogContractError(
                    f"semantic_schemas[{index}] must be a JSON object"
                )
            schemas.append(copied)
            discovered_secret_manifest.extend(schema_secret_manifest)
        object.__setattr__(
            self,
            "semantic_schemas",
            tuple(schemas),
        )

        expected_secret_manifest = _canonical_secret_schema_manifest(
            sorted(
                discovered_secret_manifest,
                key=lambda item: item["path"],
            )
        )
        supplied_secret_manifest = self.secret_schema_manifest
        if supplied_secret_manifest is not None:
            canonical_supplied_secret_manifest = _canonical_secret_schema_manifest(
                supplied_secret_manifest
            )
            if [dict(item) for item in canonical_supplied_secret_manifest] != [
                dict(item) for item in expected_secret_manifest
            ]:
                raise ToolCatalogContractError(
                    "secret schema manifest does not match system templates"
                )
        object.__setattr__(
            self,
            "secret_schema_manifest",
            expected_secret_manifest,
        )

        if type(self.entries) not in {list, tuple}:
            raise TypeError("entries must be an ordered array")
        enforce_item_limit(
            self.entries,
            boundary="tool catalog entries",
            limits=TOOL_CATALOG_JSON_LIMITS,
        )
        entries: list[ToolCatalogEntry] = []
        for entry in self.entries:
            if type(entry) is ToolCatalogEntry:
                try:
                    entries.append(ToolCatalogEntry.from_dict(entry.to_dict()))
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ToolCatalogContractError(
                        "tool catalog entry is not canonical"
                    ) from exc
            elif type(entry) is dict:
                entries.append(ToolCatalogEntry.from_dict(entry))
            else:
                raise TypeError("entries require exact ToolCatalogEntry records")
        object.__setattr__(self, "entries", tuple(entries))

        schema_names = tuple(
            _schema_name(_thaw_json(schema), provider=provider) for schema in schemas
        )
        if len(schema_names) != len(set(schema_names)):
            raise ToolCatalogContractError(
                "provider semantic schemas contain duplicate tool names"
            )
        entry_names = tuple(entry.tool_name for entry in entries)
        if schema_names != entry_names:
            raise ToolCatalogContractError(
                "provider semantic schemas and catalog entries must have the same order"
            )

        expected_manifest = {
            name: _canonical_sha256(_thaw_json(schema))
            for name, schema in zip(schema_names, schemas, strict=True)
        }
        for entry, expected_schema_sha256 in zip(
            entries,
            expected_manifest.values(),
            strict=True,
        ):
            if entry.semantic_schema_sha256 != expected_schema_sha256:
                raise ToolCatalogContractError(
                    f"catalog entry schema digest mismatch for {entry.tool_name}"
                )

        supplied_manifest = self.tool_schema_manifest
        if supplied_manifest is None:
            canonical_manifest = expected_manifest
        else:
            if type(supplied_manifest) is not dict:
                raise TypeError("tool_schema_manifest must be an exact dict")
            enforce_item_limit(
                supplied_manifest,
                boundary="tool schema manifest",
                limits=TOOL_CATALOG_JSON_LIMITS,
            )
            validate_json_resource(
                supplied_manifest,
                boundary="tool schema manifest",
                limits=TOOL_CATALOG_JSON_LIMITS,
            )
            canonical_manifest: dict[str, str] = {}
            for name, digest in supplied_manifest.items():
                canonical_name = _canonical_text(
                    name,
                    "tool_schema_manifest name",
                    identifier=True,
                )
                canonical_manifest[canonical_name] = _sha256(
                    digest,
                    f"tool_schema_manifest[{canonical_name}]",
                )
            if canonical_manifest != expected_manifest:
                raise ToolCatalogContractError(
                    "tool schema manifest does not match semantic schemas"
                )
        canonical_manifest = dict(sorted(expected_manifest.items()))
        object.__setattr__(
            self,
            "tool_schema_manifest",
            MappingProxyType(dict(canonical_manifest)),
        )

        expected_schema_sha256 = _canonical_sha256(
            [_thaw_json(schema) for schema in schemas]
        )
        supplied_schema_sha256 = self.tool_schema_sha256
        if supplied_schema_sha256:
            supplied_schema_sha256 = _sha256(
                supplied_schema_sha256,
                "tool_schema_sha256",
            )
            if supplied_schema_sha256 != expected_schema_sha256:
                raise ToolCatalogContractError(
                    "tool schema digest does not match semantic schemas"
                )
        object.__setattr__(self, "tool_schema_sha256", expected_schema_sha256)
        for field_name in (
            "required_betas_sha256",
            "prompt_sha256",
            "exposure_plan_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), field_name),
            )

        validate_json_resource(
            self._catalog_body(),
            boundary="tool catalog envelope",
            limits=TOOL_CATALOG_JSON_LIMITS,
        )

        expected_catalog_sha256 = _canonical_sha256(self._catalog_body())
        supplied_catalog_sha256 = self.catalog_sha256
        if supplied_catalog_sha256:
            supplied_catalog_sha256 = _sha256(
                supplied_catalog_sha256,
                "catalog_sha256",
            )
            if supplied_catalog_sha256 != expected_catalog_sha256:
                raise ToolCatalogContractError(
                    "catalog digest does not match catalog contents"
                )
        object.__setattr__(self, "catalog_sha256", expected_catalog_sha256)

    @property
    def provider_schemas(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.semantic_schemas)

    def _catalog_body(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "attempt": self.attempt.to_dict(),
            "iteration": self.iteration,
            "provider": self.provider,
            "model": self.model,
            "semantic_schemas": [
                _thaw_json(schema) for schema in self.semantic_schemas
            ],
            "entries": [entry.to_dict() for entry in self.entries],
            "secret_schema_manifest": [
                dict(item) for item in self.secret_schema_manifest
            ],
            "tool_schema_manifest": dict(self.tool_schema_manifest),
            "tool_schema_sha256": self.tool_schema_sha256,
            "required_betas_sha256": self.required_betas_sha256,
            "prompt_sha256": self.prompt_sha256,
            "exposure_plan_sha256": self.exposure_plan_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._catalog_body(),
            "catalog_sha256": self.catalog_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ToolCatalogEnvelope:
        fields = frozenset(
            {
                "attempt",
                "iteration",
                "provider",
                "model",
                "semantic_schemas",
                "entries",
                "secret_schema_manifest",
                "tool_schema_manifest",
                "tool_schema_sha256",
                "required_betas_sha256",
                "prompt_sha256",
                "exposure_plan_sha256",
                "catalog_sha256",
            }
        )
        raw = _strict_record(value, schema=cls.SCHEMA, required=fields)
        return cls(**{field_name: raw[field_name] for field_name in fields})


@dataclass(frozen=True)
class ToolCatalogSnapshot:
    """Persisted catalog envelope bound to its journal event and artifact."""

    SCHEMA: ClassVar[str] = "unchain.tool_catalog_snapshot.v1"

    envelope: ToolCatalogEnvelope
    event_cursor: EventCursor
    artifact: ArtifactRef
    snapshot_sha256: str = ""

    def __post_init__(self) -> None:
        envelope = self.envelope
        if type(envelope) is dict:
            envelope = ToolCatalogEnvelope.from_dict(envelope)
        elif type(envelope) is ToolCatalogEnvelope:
            try:
                envelope = ToolCatalogEnvelope.from_dict(envelope.to_dict())
            except (AttributeError, TypeError, ValueError) as exc:
                raise ToolCatalogContractError(
                    f"tool catalog snapshot envelope is not canonical: {exc}"
                ) from exc
        else:
            raise TypeError("envelope must be an exact ToolCatalogEnvelope")
        object.__setattr__(self, "envelope", envelope)

        cursor = self.event_cursor
        if type(cursor) is dict:
            cursor = EventCursor.from_dict(cursor)
        elif type(cursor) is not EventCursor:
            raise TypeError("event_cursor must be an exact EventCursor")
        else:
            cursor = EventCursor.from_dict(cursor.to_dict())
        object.__setattr__(self, "event_cursor", cursor)

        artifact = self.artifact
        if type(artifact) is dict:
            artifact = ArtifactRef.from_dict(artifact)
        elif type(artifact) is not ArtifactRef:
            raise TypeError("artifact must be an exact ArtifactRef")
        else:
            artifact = ArtifactRef.from_dict(artifact.to_dict())
        object.__setattr__(self, "artifact", artifact)

        if artifact.ref.kind != "artifact" or artifact.ref.fragment:
            raise ToolCatalogContractError(
                "tool catalog snapshot requires a whole artifact ref"
            )
        if artifact.media_type != "application/json":
            raise ToolCatalogContractError(
                "tool catalog snapshot artifact must use application/json"
            )
        if artifact.preview:
            raise ToolCatalogContractError(
                "tool catalog snapshot artifact preview must be empty"
            )
        content = envelope.canonical_bytes()
        if len(content) > MAX_TOOL_CATALOG_ARTIFACT_BYTES:
            _reject_schema_limit(
                boundary="tool catalog artifact",
                dimension="bytes",
                limit=MAX_TOOL_CATALOG_ARTIFACT_BYTES,
                observed=len(content),
            )
        if artifact.byte_length != len(content):
            raise ToolCatalogContractError(
                "tool catalog artifact byte length does not match envelope"
            )
        if artifact.sha256 != hashlib.sha256(content).hexdigest():
            raise ToolCatalogContractError(
                "tool catalog artifact digest does not match envelope"
            )

        expected_snapshot_sha256 = _canonical_sha256(self._snapshot_body())
        supplied_snapshot_sha256 = self.snapshot_sha256
        if supplied_snapshot_sha256:
            supplied_snapshot_sha256 = _sha256(
                supplied_snapshot_sha256,
                "snapshot_sha256",
            )
            if supplied_snapshot_sha256 != expected_snapshot_sha256:
                raise ToolCatalogContractError(
                    "snapshot digest does not match catalog reference"
                )
        object.__setattr__(
            self,
            "snapshot_sha256",
            expected_snapshot_sha256,
        )

    @property
    def attempt(self) -> AttemptRef:
        return self.envelope.attempt

    @property
    def iteration(self) -> int:
        return self.envelope.iteration

    @property
    def provider(self) -> str:
        return self.envelope.provider

    @property
    def model(self) -> str:
        return self.envelope.model

    @property
    def catalog_sha256(self) -> str:
        return self.envelope.catalog_sha256

    def _snapshot_body(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "envelope": self.envelope.to_dict(),
            "event_cursor": self.event_cursor.to_dict(),
            "artifact": self.artifact.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._snapshot_body(),
            "snapshot_sha256": self.snapshot_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ToolCatalogSnapshot:
        fields = frozenset({"envelope", "event_cursor", "artifact", "snapshot_sha256"})
        raw = _strict_record(value, schema=cls.SCHEMA, required=fields)
        return cls(**{field_name: raw[field_name] for field_name in fields})


__all__ = [
    "MAX_TOOL_CATALOG_ARTIFACT_BYTES",
    "MAX_TOOL_CATALOG_BYTES",
    "MAX_TOOL_CATALOG_ENTRIES",
    "MAX_TOOL_CATALOG_ITERATION",
    "MAX_TOOL_SCHEMA_BYTES",
    "MAX_TOOL_SCHEMA_CONTAINER_ITEMS",
    "MAX_TOOL_SCHEMA_STRING_BYTES",
    "SUPPORTED_TOOL_CATALOG_PROVIDERS",
    "TOOL_CATALOG_JSON_LIMITS",
    "TOOL_SCHEMA_JSON_LIMITS",
    "ToolCatalogContractError",
    "ToolCatalogEntry",
    "ToolCatalogEnvelope",
    "ToolCatalogRouteKind",
    "ToolCatalogSnapshot",
    "build_tool_catalog_entry_from_resolution",
    "verify_tool_catalog_entry_resolution",
]
