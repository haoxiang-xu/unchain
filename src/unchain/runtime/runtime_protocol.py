"""Code-backed protocol manifest for host/runtime compatibility admission.

The manifest is derived only from this imported module.  It deliberately has
no Git, filesystem-resource, environment, or packaging-state dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar


RUNTIME_PROTOCOL_MANIFEST_SCHEMA = "unchain.runtime_protocol_manifest.v1"
RUNTIME_PROTOCOL_MANIFEST_DIGEST_DOMAIN = (
    b"unchain.runtime_protocol_manifest.v1\\u0000"
)
_RUNTIME_NAME = "unchain"
_MANIFEST_KEYS = frozenset({"manifest_digest", "protocols", "runtime", "schema"})
_PROTOCOL_KEYS = frozenset({"features", "id", "major", "minor"})
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_SAFE_INTEGER = (1 << 53) - 1


class RuntimeProtocolManifestError(ValueError):
    """The runtime protocol manifest is malformed or non-canonical."""


def _nfc_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeProtocolManifestError(f"{label} must be a non-empty string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise RuntimeProtocolManifestError(
            f"{label} must be a strict UTF-8 Unicode scalar sequence"
        ) from exc
    if unicodedata.normalize("NFC", value) != value:
        raise RuntimeProtocolManifestError(f"{label} must use NFC normalization")
    return value


def _version(value: object, *, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > _MAX_SAFE_INTEGER
    ):
        raise RuntimeProtocolManifestError(
            f"{label} must be a non-negative safe integer"
        )
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        Sequence,
    ):
        raise RuntimeProtocolManifestError(f"{label} must be an array")
    return value


def _canonical_strings(value: object, *, label: str) -> tuple[str, ...]:
    items = tuple(
        _nfc_text(item, label=f"{label} item")
        for item in _sequence(value, label=label)
    )
    if len(set(items)) != len(items):
        raise RuntimeProtocolManifestError(f"{label} must be unique")
    if items != tuple(sorted(items, key=lambda item: item.encode("utf-8"))):
        raise RuntimeProtocolManifestError(f"{label} must use canonical order")
    return items


@dataclass(frozen=True, slots=True)
class RuntimeProtocol:
    """One canonically ordered protocol version and its advertised features."""

    id: str
    major: int
    minor: int
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _nfc_text(self.id, label="protocol id"))
        object.__setattr__(self, "major", _version(self.major, label="major"))
        object.__setattr__(self, "minor", _version(self.minor, label="minor"))
        object.__setattr__(
            self,
            "features",
            _canonical_strings(self.features, label="features"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "id": self.id,
            "major": self.major,
            "minor": self.minor,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimeProtocol:
        if not isinstance(value, Mapping) or set(value) != _PROTOCOL_KEYS:
            raise RuntimeProtocolManifestError("protocol item fields are invalid")
        return cls(
            id=value["id"],
            major=value["major"],
            minor=value["minor"],
            features=tuple(_sequence(value["features"], label="features")),
        )


def _canonical_body_bytes(
    *,
    schema: str,
    runtime: str,
    protocols: tuple[RuntimeProtocol, ...],
) -> bytes:
    body = {
        "protocols": [protocol.to_dict() for protocol in protocols],
        "runtime": runtime,
        "schema": schema,
    }
    try:
        canonical = json.dumps(
            body,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - typed above
        raise RuntimeProtocolManifestError(
            "runtime protocol body is not canonical JSON"
        ) from exc
    return canonical.encode("utf-8")


def _manifest_digest(
    *,
    schema: str,
    runtime: str,
    protocols: tuple[RuntimeProtocol, ...],
) -> str:
    body = _canonical_body_bytes(
        schema=schema,
        runtime=runtime,
        protocols=protocols,
    )
    return "sha256:" + hashlib.sha256(
        RUNTIME_PROTOCOL_MANIFEST_DIGEST_DOMAIN + body
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeProtocolManifest:
    """Strict, digest-bound runtime protocol manifest."""

    SCHEMA: ClassVar[str] = RUNTIME_PROTOCOL_MANIFEST_SCHEMA

    protocols: tuple[RuntimeProtocol, ...]
    manifest_digest: str
    runtime: str = _RUNTIME_NAME
    schema: str = RUNTIME_PROTOCOL_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        schema = _nfc_text(self.schema, label="manifest schema")
        runtime = _nfc_text(self.runtime, label="runtime")
        if schema != self.SCHEMA or runtime != _RUNTIME_NAME:
            raise RuntimeProtocolManifestError(
                "runtime protocol manifest identity is invalid"
            )
        protocols = tuple(self.protocols)
        if not all(isinstance(item, RuntimeProtocol) for item in protocols):
            raise RuntimeProtocolManifestError(
                "protocols must contain RuntimeProtocol values"
            )
        protocol_ids = tuple(item.id for item in protocols)
        if len(set(protocol_ids)) != len(protocol_ids):
            raise RuntimeProtocolManifestError("protocol ids must be unique")
        if protocol_ids != tuple(
            sorted(protocol_ids, key=lambda item: item.encode("utf-8"))
        ):
            raise RuntimeProtocolManifestError(
                "protocols must use canonical order"
            )
        digest = self.manifest_digest
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise RuntimeProtocolManifestError("manifest digest is invalid")
        expected_digest = _manifest_digest(
            schema=schema,
            runtime=runtime,
            protocols=protocols,
        )
        if digest != expected_digest:
            raise RuntimeProtocolManifestError("manifest digest does not match")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "protocols", protocols)

    @classmethod
    def build(
        cls,
        protocols: Sequence[RuntimeProtocol],
    ) -> RuntimeProtocolManifest:
        normalized = tuple(protocols)
        return cls(
            protocols=normalized,
            manifest_digest=_manifest_digest(
                schema=cls.SCHEMA,
                runtime=_RUNTIME_NAME,
                protocols=normalized,
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimeProtocolManifest:
        if not isinstance(value, Mapping) or set(value) != _MANIFEST_KEYS:
            raise RuntimeProtocolManifestError("manifest fields are invalid")
        raw_protocols = _sequence(value["protocols"], label="protocols")
        protocols = tuple(RuntimeProtocol.from_dict(item) for item in raw_protocols)
        return cls(
            schema=value["schema"],
            runtime=value["runtime"],
            protocols=protocols,
            manifest_digest=value["manifest_digest"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_digest": self.manifest_digest,
            "protocols": [protocol.to_dict() for protocol in self.protocols],
            "runtime": self.runtime,
            "schema": self.schema,
        }


_PROTOCOLS = (
    RuntimeProtocol(
        id="context_memory",
        major=1,
        minor=0,
        features=(
            "artifact_handoff",
            "canonical_journal",
            "chat_deletion_sqlite_scope_closure",
            "context_compiler",
            "context_contribution_manifest_v1",
            "generation_rebase_live_interaction_cycles",
            "interaction_resolution_compat",
            "long_term_promotion",
            "memory_curator",
            "memory_toolkit",
            "memory_workspace",
            "tool_output_management_v1",
        ),
    ),
    RuntimeProtocol(
        id="durable_interaction",
        major=1,
        minor=0,
        features=(
            "cancel_pending",
            "expected_interaction_id_cas",
            "fresh_run_lineage",
            "graph_interaction_lineage_preflight_v1",
            "host_controlled_resume",
            "interaction_resolution_atomic_acceptance_v1",
        ),
    ),
    RuntimeProtocol(
        id="provider_turn_ownership",
        major=1,
        minor=0,
        features=(
            "atomic_receipt_cas",
            "auxiliary_calls",
            "enforce_mode",
            "graph_runs",
            "memory_off",
            "ollama_reasoning_preview_v1",
            "subagent_runs",
        ),
    ),
    RuntimeProtocol(
        id="run_bundle",
        major=1,
        minor=0,
        features=(
            "canonical_metrics",
            "completion_diagnostics_ref",
            "context_composition_ref_v1",
            "continuation_claim",
            "immutable_pricing_snapshot",
            "provider_call_set_union",
            "provider_call_usage_v1",
            "run_bundle_v1",
            "run_bundle_v2",
        ),
    ),
    RuntimeProtocol(
        id="skills",
        major=1,
        minor=0,
        features=(
            "active_skills_snapshot_v1",
            "catalog_v1",
            "skill_md_registry_v1",
            "skill_tool_v1",
            "toolkit_embedded_skills_v1",
            "user_invocation_v1",
        ),
    ),
)


def build_runtime_protocol_manifest() -> RuntimeProtocolManifest:
    """Build the manifest advertised by this imported Unchain runtime."""

    return RuntimeProtocolManifest.build(_PROTOCOLS)


def runtime_protocol_manifest() -> dict[str, Any]:
    """Return a fresh wire representation of the loaded runtime protocol."""

    return build_runtime_protocol_manifest().to_dict()


__all__ = (
    "RUNTIME_PROTOCOL_MANIFEST_DIGEST_DOMAIN",
    "RUNTIME_PROTOCOL_MANIFEST_SCHEMA",
    "RuntimeProtocol",
    "RuntimeProtocolManifest",
    "RuntimeProtocolManifestError",
    "build_runtime_protocol_manifest",
    "runtime_protocol_manifest",
)
