"""Fail-closed adapter for a host-verified, hash-pinned pricing catalog.

Ed25519 envelope verification intentionally happens in the PuPu offline Node
release tool.  This module accepts only its deterministic projection together
with an independently configured projection digest.  It performs no network
access and never treats an unsigned/unpinned catalog as trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
from typing import Any, Mapping
from urllib.parse import urlsplit

from .run_bundle import (
    PricingSnapshotRef,
    ProviderCallPricing,
    ProviderCallReceipt,
)


PRICING_CATALOG_SCHEMA = "pupu.pricing_catalog.v1"
VERIFIED_PRICING_PROJECTION_SCHEMA = (
    "pupu.verified_pricing_catalog_projection.v1"
)
PRICING_PROJECTION_PATH_ENV = "UNCHAIN_PRICING_CATALOG_PROJECTION_PATH"
PRICING_PROJECTION_SHA256_ENV = "UNCHAIN_PRICING_CATALOG_PROJECTION_SHA256"

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z$"
)
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_MAX_FILE_BYTES = 4 * 1024 * 1024
_RATE_KEYS = frozenset(
    {
        "input_uncached",
        "input_cache_read",
        "input_cache_write",
        "input_cache_write_5m",
        "input_cache_write_1h",
        "output",
    }
)
_OFFICIAL_SOURCE_RULES: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "openai",
        "developers.openai.com",
        ("/api/docs/pricing", "/api/docs/guides/latest-model"),
        ("/api/docs/models/",),
    ),
    (
        "anthropic",
        "platform.claude.com",
        (
            "/docs/en/about-claude/pricing",
            "/docs/en/manage-claude/usage-cost-api",
        ),
        (),
    ),
)


class PricingCatalogError(ValueError):
    """A stable fail-closed pricing catalog error."""

    def __init__(self, code: str, message: str, target: str = "pricing") -> None:
        super().__init__(f"{code}: {target}: {message}")
        self.code = code
        self.target = target


def _fail(code: str, message: str, target: str) -> None:
    raise PricingCatalogError(code, message, target)


def _strict_record(
    value: Any,
    *,
    keys: frozenset[str],
    target: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("pricing_projection_invalid", "must be an exact object", target)
    if frozenset(value) != keys:
        _fail(
            "pricing_projection_invalid",
            f"unexpected key set ({','.join(sorted(value))})",
            target,
        )
    return value


def _text(value: Any, target: str, *, maximum: int = 4096) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 for character in value)
    ):
        _fail("pricing_projection_invalid", "must be bounded NFC text", target)
    return value


def _slug(value: Any, target: str) -> str:
    observed = _text(value, target, maximum=128)
    if not _SLUG_RE.fullmatch(observed):
        _fail("pricing_projection_invalid", "must be a lowercase slug", target)
    return observed


def _digest(value: Any, target: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(
            "pricing_projection_invalid",
            "must be a lowercase SHA-256",
            target,
        )
    return value


def _timestamp(value: Any, target: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if type(value) is not str or _RFC3339_RE.fullmatch(value) is None:
        _fail(
            "pricing_projection_invalid",
            "must be an RFC3339 UTC timestamp",
            target,
        )
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _fail("pricing_projection_invalid", "timestamp is not valid", target)
    return value


def _instant(value: str) -> int:
    base = datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = base - epoch
    seconds = delta.days * 86_400 + delta.seconds
    fraction_match = re.search(r"\.(\d+)Z$", value)
    fraction = (
        int(fraction_match.group(1).ljust(9, "0"))
        if fraction_match is not None
        else 0
    )
    return seconds * 1_000_000_000 + fraction


def _normalized_timestamp(value: str) -> str:
    """Return an RFC3339 value whose lexical order is chronological."""

    fraction_match = re.search(r"\.(\d+)Z$", value)
    fraction = (
        fraction_match.group(1).ljust(9, "0")
        if fraction_match is not None
        else "000000000"
    )
    return f"{value[:19]}.{fraction}Z"


def _safe_integer(
    value: Any,
    target: str,
    *,
    nullable: bool = False,
    positive: bool = False,
) -> int | None:
    if nullable and value is None:
        return None
    if type(value) is not int or value < (1 if positive else 0) or value > 2**53 - 1:
        _fail(
            "pricing_projection_invalid",
            "must be a safe non-negative integer",
            target,
        )
    return value


def _canonical_value(value: Any, target: str = "value") -> Any:
    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        return _text(value, target, maximum=max(1, len(value)))
    if type(value) is int:
        return _safe_integer(value, target)
    if type(value) is list:
        return [
            _canonical_value(item, f"{target}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            normalized_key = _text(key, f"{target}.key", maximum=4096)
            if normalized_key in normalized:
                _fail(
                    "pricing_projection_invalid",
                    "contains duplicate keys",
                    target,
                )
            normalized[normalized_key] = _canonical_value(
                value[key], f"{target}.{normalized_key}"
            )
        return normalized
    _fail("pricing_projection_invalid", "must be strict JSON", target)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("pricing_projection_invalid", "contains duplicate keys", "file")
        result[key] = value
    return result


def _load_json_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
    except (OSError, TypeError, ValueError) as exc:
        raise PricingCatalogError(
            "pricing_projection_unavailable", "projection file cannot be read", "file"
        ) from exc
    if not raw or len(raw) > _MAX_FILE_BYTES:
        _fail(
            "pricing_projection_invalid", "projection file is empty or too large", "file"
        )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=lambda constant: _fail(
                "pricing_projection_invalid",
                f"non-finite number {constant} is prohibited",
                "file",
            ),
        )
    except PricingCatalogError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PricingCatalogError(
            "pricing_projection_invalid", "projection file is not strict JSON", "file"
        ) from exc
    if type(value) is not dict:
        _fail("pricing_projection_invalid", "projection must be an object", "file")
    return value


def _official_source_provider(source_url: str) -> str:
    try:
        parsed = urlsplit(source_url)
        parsed_port = parsed.port
    except ValueError:
        _fail(
            "pricing_source_not_official",
            "source URL is malformed",
            "catalog.source.url",
        )
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed_port is not None
        or parsed.query
        or parsed.fragment
    ):
        _fail(
            "pricing_source_not_official",
            "must be an uncredentialed HTTPS URL without port, query, or fragment",
            "catalog.source.url",
        )
    for provider, hostname, exact_paths, prefixes in _OFFICIAL_SOURCE_RULES:
        if parsed.hostname == hostname and (
            parsed.path in exact_paths
            or any(parsed.path.startswith(prefix) for prefix in prefixes)
        ):
            return provider
    _fail(
        "pricing_source_not_official",
        "URL is not in the reviewed official pricing-source allowlist",
        "catalog.source.url",
    )


@dataclass(frozen=True, slots=True)
class _CatalogEntry:
    snapshot: PricingSnapshotRef
    effective_from: int
    effective_until: int | None


@dataclass(frozen=True, slots=True)
class PricingCatalogResolver:
    """Immutable exact-identity resolver for one historical catalog projection."""

    projection_sha256: str
    catalog_payload_sha256: str
    catalog_version: str
    key_id: str
    trusted_public_key_sha256: str
    _entries: tuple[_CatalogEntry, ...]

    @classmethod
    def from_projection_file(
        cls,
        projection_path: str | os.PathLike[str],
        *,
        expected_projection_sha256: str,
    ) -> PricingCatalogResolver:
        expected_digest = _digest(
            expected_projection_sha256, "expected_projection_sha256"
        )
        projection = _validate_projection(_load_json_file(projection_path))
        if projection["projection_sha256"] != expected_digest:
            _fail(
                "pricing_projection_untrusted",
                "projection digest is not pinned",
                "expected_projection_sha256",
            )
        verification = projection["verification"]
        catalog = projection["catalog"]
        entries: list[_CatalogEntry] = []
        for entry in catalog["entries"]:
            source = catalog["sources"][entry["source_index"]]
            rates = entry["rates_nano_usd_per_million"]
            rule = entry["long_context_rule"]
            snapshot = PricingSnapshotRef(
                catalog_version=catalog["catalog_version"],
                catalog_sha256=verification["catalog_payload_sha256"],
                source_url=source["url"],
                source_sha256=source["source_digest"],
                effective_from=_normalized_timestamp(catalog["effective_from"]),
                effective_until=(
                    _normalized_timestamp(catalog["effective_to"])
                    if catalog["effective_to"] is not None
                    else None
                ),
                currency=entry["currency"],
                provider=entry["provider"],
                billing_surface=entry["billing_surface"],
                model=entry["model"],
                service_tier=entry["service_tier"],
                batch=entry["batch"],
                inference_geo=entry["inference_geo"],
                input_uncached_nano_usd_per_million=rates["input_uncached"],
                input_cache_read_nano_usd_per_million=rates["input_cache_read"],
                input_cache_write_nano_usd_per_million=rates["input_cache_write"],
                input_cache_write_5m_nano_usd_per_million=rates[
                    "input_cache_write_5m"
                ],
                input_cache_write_1h_nano_usd_per_million=rates[
                    "input_cache_write_1h"
                ],
                output_nano_usd_per_million=rates["output"],
                long_context_threshold_input_tokens=(
                    rule["threshold_input_tokens"] if rule is not None else None
                ),
                long_context_input_multiplier_ppm=(
                    rule["input_multiplier_ppm"] if rule is not None else None
                ),
                long_context_output_multiplier_ppm=(
                    rule["output_multiplier_ppm"] if rule is not None else None
                ),
            )
            entries.append(
                _CatalogEntry(
                    snapshot=snapshot,
                    effective_from=_instant(catalog["effective_from"]),
                    effective_until=(
                        _instant(catalog["effective_to"])
                        if catalog["effective_to"] is not None
                        else None
                    ),
                )
            )
        return cls(
            projection_sha256=projection["projection_sha256"],
            catalog_payload_sha256=verification["catalog_payload_sha256"],
            catalog_version=catalog["catalog_version"],
            key_id=verification["key_id"],
            trusted_public_key_sha256=verification[
                "trusted_public_key_sha256"
            ],
            _entries=tuple(entries),
        )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> PricingCatalogResolver:
        source = os.environ if environment is None else environment
        projection_path = str(source.get(PRICING_PROJECTION_PATH_ENV, "")).strip()
        projection_sha256 = str(
            source.get(PRICING_PROJECTION_SHA256_ENV, "")
        ).strip()
        if not projection_path and not projection_sha256:
            _fail(
                "pricing_catalog_unconfigured",
                "no trusted pricing projection is configured",
                "environment",
            )
        if not projection_path or not projection_sha256:
            _fail(
                "pricing_catalog_trust_configuration_incomplete",
                "projection path and digest must be configured together",
                "environment",
            )
        return _load_cached_pricing_catalog(
            str(Path(projection_path).resolve()), projection_sha256
        )

    def resolve_snapshot(
        self,
        *,
        provider: str,
        billing_surface: str,
        model: str,
        service_tier: str,
        batch: bool,
        inference_geo: str,
        occurred_at: str,
    ) -> PricingSnapshotRef:
        occurred_text = _timestamp(occurred_at, "occurred_at")
        assert occurred_text is not None
        occurred = _instant(occurred_text)
        matches = [
            entry.snapshot
            for entry in self._entries
            if entry.snapshot.provider == provider
            and entry.snapshot.billing_surface == billing_surface
            and entry.snapshot.model == model
            and entry.snapshot.service_tier == service_tier
            and entry.snapshot.batch is batch
            and entry.snapshot.inference_geo == inference_geo
            and occurred >= entry.effective_from
            and (
                entry.effective_until is None
                or occurred < entry.effective_until
            )
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            _fail(
                "pricing_identity_ambiguous",
                "multiple exact pricing identities matched",
                "catalog.entries",
            )
        identity_exists = any(
            entry.snapshot.provider == provider
            and entry.snapshot.billing_surface == billing_surface
            and entry.snapshot.model == model
            and entry.snapshot.service_tier == service_tier
            and entry.snapshot.batch is batch
            and entry.snapshot.inference_geo == inference_geo
            for entry in self._entries
        )
        raise PricingCatalogError(
            "pricing_catalog_not_effective"
            if identity_exists
            else "pricing_identity_unknown",
            "no exact effective pricing identity matched",
            "catalog.entries",
        )


def _validate_source(value: Any, index: int) -> dict[str, Any]:
    target = f"catalog.sources[{index}]"
    source = _strict_record(
        value,
        keys=frozenset(
            {"provider", "url", "retrieved_at", "source_digest", "review_note"}
        ),
        target=target,
    )
    provider = _slug(source["provider"], f"{target}.provider")
    source_url = _text(source["url"], f"{target}.url", maximum=2_048)
    if _official_source_provider(source_url) != provider:
        _fail(
            "pricing_source_not_official",
            "source provider does not match its official host",
            f"{target}.provider",
        )
    return {
        "provider": provider,
        "url": source_url,
        "retrieved_at": _timestamp(source["retrieved_at"], f"{target}.retrieved_at"),
        "source_digest": _digest(
            source["source_digest"], f"{target}.source_digest"
        ),
        "review_note": _text(source["review_note"], f"{target}.review_note"),
    }


def _validate_entry(value: Any, index: int, source_count: int) -> dict[str, Any]:
    target = f"catalog.entries[{index}]"
    entry = _strict_record(
        value,
        keys=frozenset(
            {
                "provider",
                "billing_surface",
                "model",
                "service_tier",
                "batch",
                "inference_geo",
                "currency",
                "rates_nano_usd_per_million",
                "long_context_rule",
                "source_index",
            }
        ),
        target=target,
    )
    if type(entry["batch"]) is not bool:
        _fail("pricing_projection_invalid", "must be boolean", f"{target}.batch")
    source_index = _safe_integer(entry["source_index"], f"{target}.source_index")
    assert source_index is not None
    if source_index >= source_count:
        _fail(
            "pricing_projection_invalid",
            "must reference one source",
            f"{target}.source_index",
        )
    raw_rates = _strict_record(
        entry["rates_nano_usd_per_million"],
        keys=_RATE_KEYS,
        target=f"{target}.rates_nano_usd_per_million",
    )
    rates = {
        key: _safe_integer(
            raw_rates[key],
            f"{target}.rates_nano_usd_per_million.{key}",
            nullable=True,
        )
        for key in sorted(_RATE_KEYS)
    }
    long_context_rule = entry["long_context_rule"]
    if long_context_rule is not None:
        rule = _strict_record(
            long_context_rule,
            keys=frozenset(
                {
                    "threshold_input_tokens",
                    "input_multiplier_ppm",
                    "output_multiplier_ppm",
                }
            ),
            target=f"{target}.long_context_rule",
        )
        long_context_rule = {
            "threshold_input_tokens": _safe_integer(
                rule["threshold_input_tokens"],
                f"{target}.long_context_rule.threshold_input_tokens",
                positive=True,
            ),
            "input_multiplier_ppm": _safe_integer(
                rule["input_multiplier_ppm"],
                f"{target}.long_context_rule.input_multiplier_ppm",
                positive=True,
            ),
            "output_multiplier_ppm": _safe_integer(
                rule["output_multiplier_ppm"],
                f"{target}.long_context_rule.output_multiplier_ppm",
                positive=True,
            ),
        }
    currency = _text(entry["currency"], f"{target}.currency", maximum=8)
    if currency != "USD":
        _fail("pricing_projection_invalid", "currency must be USD", f"{target}.currency")
    return {
        "provider": _slug(entry["provider"], f"{target}.provider"),
        "billing_surface": _slug(
            entry["billing_surface"], f"{target}.billing_surface"
        ),
        "model": _text(entry["model"], f"{target}.model", maximum=256),
        "service_tier": _slug(
            entry["service_tier"], f"{target}.service_tier"
        ),
        "batch": entry["batch"],
        "inference_geo": _slug(
            entry["inference_geo"], f"{target}.inference_geo"
        ),
        "currency": currency,
        "rates_nano_usd_per_million": rates,
        "long_context_rule": long_context_rule,
        "source_index": source_index,
    }


def _validate_catalog(value: Any) -> dict[str, Any]:
    catalog = _strict_record(
        value,
        keys=frozenset(
            {
                "schema",
                "catalog_version",
                "currency",
                "effective_from",
                "effective_to",
                "retrieved_at",
                "sources",
                "entries",
            }
        ),
        target="catalog",
    )
    if catalog["schema"] != PRICING_CATALOG_SCHEMA:
        _fail(
            "pricing_projection_invalid",
            f"schema must be {PRICING_CATALOG_SCHEMA}",
            "catalog.schema",
        )
    currency = _text(catalog["currency"], "catalog.currency", maximum=8)
    if currency != "USD":
        _fail("pricing_projection_invalid", "currency must be USD", "catalog.currency")
    effective_from = _timestamp(catalog["effective_from"], "catalog.effective_from")
    effective_to = _timestamp(
        catalog["effective_to"], "catalog.effective_to", nullable=True
    )
    assert effective_from is not None
    if effective_to is not None and _instant(effective_to) <= _instant(effective_from):
        _fail(
            "pricing_projection_invalid",
            "effective_to must follow effective_from",
            "catalog.effective_to",
        )
    if type(catalog["sources"]) is not list or not catalog["sources"]:
        _fail(
            "pricing_projection_invalid",
            "must contain at least one source",
            "catalog.sources",
        )
    if type(catalog["entries"]) is not list or not catalog["entries"]:
        _fail(
            "pricing_projection_invalid",
            "must contain at least one entry",
            "catalog.entries",
        )
    sources = [
        _validate_source(source, index)
        for index, source in enumerate(catalog["sources"])
    ]
    entries = [
        _validate_entry(entry, index, len(sources))
        for index, entry in enumerate(catalog["entries"])
    ]
    for index, entry in enumerate(entries):
        if entry["currency"] != currency:
            _fail(
                "pricing_projection_invalid",
                "entry currency must match catalog currency",
                f"catalog.entries[{index}].currency",
            )
        if sources[entry["source_index"]]["provider"] != entry["provider"]:
            _fail(
                "pricing_projection_invalid",
                "entry provider must match its source provider",
                f"catalog.entries[{index}].source_index",
            )
    identities = [
        (
            entry["provider"],
            entry["billing_surface"],
            entry["model"],
            entry["service_tier"],
            entry["batch"],
            entry["inference_geo"],
        )
        for entry in entries
    ]
    if len(set(identities)) != len(identities):
        _fail(
            "pricing_projection_invalid",
            "contains a duplicate price identity",
            "catalog.entries",
        )
    retrieved_at = _timestamp(catalog["retrieved_at"], "catalog.retrieved_at")
    assert retrieved_at is not None
    for index, source in enumerate(sources):
        if _instant(source["retrieved_at"]) > _instant(retrieved_at):
            _fail(
                "pricing_projection_invalid",
                "source retrieval cannot follow catalog retrieval",
                f"catalog.sources[{index}].retrieved_at",
            )
    return {
        "schema": PRICING_CATALOG_SCHEMA,
        "catalog_version": _text(
            catalog["catalog_version"], "catalog.catalog_version", maximum=128
        ),
        "currency": currency,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "retrieved_at": retrieved_at,
        "sources": sources,
        "entries": entries,
    }


def _validate_projection(value: Any) -> dict[str, Any]:
    projection = _strict_record(
        value,
        keys=frozenset(
            {"schema", "verification", "catalog", "projection_sha256"}
        ),
        target="projection",
    )
    if projection["schema"] != VERIFIED_PRICING_PROJECTION_SCHEMA:
        _fail(
            "pricing_projection_invalid",
            f"schema must be {VERIFIED_PRICING_PROJECTION_SCHEMA}",
            "projection.schema",
        )
    verification = _strict_record(
        projection["verification"],
        keys=frozenset(
            {
                "algorithm",
                "key_id",
                "trusted_public_key_sha256",
                "catalog_payload_sha256",
            }
        ),
        target="projection.verification",
    )
    if verification["algorithm"] != "ed25519":
        _fail(
            "pricing_projection_invalid",
            "algorithm must be ed25519",
            "projection.verification.algorithm",
        )
    normalized_verification = {
        "algorithm": "ed25519",
        "key_id": _text(
            verification["key_id"], "projection.verification.key_id", maximum=128
        ),
        "trusted_public_key_sha256": _digest(
            verification["trusted_public_key_sha256"],
            "projection.verification.trusted_public_key_sha256",
        ),
        "catalog_payload_sha256": _digest(
            verification["catalog_payload_sha256"],
            "projection.verification.catalog_payload_sha256",
        ),
    }
    catalog = _validate_catalog(projection["catalog"])
    if _sha256(_canonical_json(catalog)) != normalized_verification[
        "catalog_payload_sha256"
    ]:
        _fail(
            "pricing_projection_invalid",
            "catalog payload digest changed",
            "projection.catalog",
        )
    body = {
        "schema": VERIFIED_PRICING_PROJECTION_SCHEMA,
        "verification": normalized_verification,
        "catalog": catalog,
    }
    projection_sha256 = _digest(
        projection["projection_sha256"], "projection.projection_sha256"
    )
    if _sha256(_canonical_json(body)) != projection_sha256:
        _fail(
            "pricing_projection_invalid",
            "projection digest changed",
            "projection.projection_sha256",
        )
    return {**body, "projection_sha256": projection_sha256}


def load_pinned_pricing_catalog(
    projection_path: str | os.PathLike[str],
    *,
    expected_projection_sha256: str,
) -> PricingCatalogResolver:
    """Load one reviewed projection only when the caller supplies its exact pin."""

    return PricingCatalogResolver.from_projection_file(
        projection_path,
        expected_projection_sha256=expected_projection_sha256,
    )


@lru_cache(maxsize=16)
def _load_cached_pricing_catalog(
    projection_path: str,
    projection_sha256: str,
) -> PricingCatalogResolver:
    """Cache only immutable path+digest pairs used by the production adapter."""

    return PricingCatalogResolver.from_projection_file(
        projection_path,
        expected_projection_sha256=projection_sha256,
    )


def resolve_pricing_for_receipt(
    receipt: ProviderCallReceipt,
    *,
    occurred_at: str | None = None,
    resolver: PricingCatalogResolver | None = None,
    billing_surface: str | None = None,
    batch: bool | None = None,
    inference_geo: str | None = None,
    service_tier: str | None = None,
    projection_path: str | os.PathLike[str] | None = None,
    expected_projection_sha256: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> ProviderCallPricing:
    """Resolve list pricing without ever raising into an agent execution.

    The provider ledger may pass a preloaded resolver.  Otherwise both an
    explicit projection path/pin or both environment settings are required.
    Missing call time, tier, billing surface, batch mode, or inference geography
    remains unknown instead of being guessed.
    """

    try:
        if type(receipt) is not ProviderCallReceipt:
            return ProviderCallPricing.unavailable("pricing_receipt_invalid")
        if receipt.pricing.status == "estimated":
            return receipt.pricing
        if occurred_at is None:
            return ProviderCallPricing.unavailable("pricing_time_unknown")
        if type(occurred_at) is not str or _RFC3339_RE.fullmatch(occurred_at) is None:
            return ProviderCallPricing.unavailable("pricing_time_invalid")
        try:
            _instant(occurred_at)
        except ValueError:
            return ProviderCallPricing.unavailable("pricing_time_invalid")
        resolved_tier = service_tier or receipt.service_tier
        if resolved_tier is None:
            return ProviderCallPricing.unavailable("pricing_service_tier_unknown")
        if type(resolved_tier) is not str or _SLUG_RE.fullmatch(resolved_tier) is None:
            return ProviderCallPricing.unavailable("pricing_service_tier_invalid")
        if billing_surface is None:
            return ProviderCallPricing.unavailable("pricing_billing_surface_unknown")
        if (
            type(billing_surface) is not str
            or _SLUG_RE.fullmatch(billing_surface) is None
        ):
            return ProviderCallPricing.unavailable("pricing_billing_surface_invalid")
        if batch is None:
            return ProviderCallPricing.unavailable("pricing_batch_mode_unknown")
        if type(batch) is not bool:
            return ProviderCallPricing.unavailable("pricing_batch_mode_invalid")
        if inference_geo is None:
            return ProviderCallPricing.unavailable("pricing_inference_geo_unknown")
        if type(inference_geo) is not str or _SLUG_RE.fullmatch(inference_geo) is None:
            return ProviderCallPricing.unavailable("pricing_inference_geo_invalid")
        active_resolver = resolver
        if active_resolver is None:
            if projection_path is not None or expected_projection_sha256 is not None:
                if projection_path is None or expected_projection_sha256 is None:
                    return ProviderCallPricing.unavailable(
                        "pricing_catalog_trust_configuration_incomplete"
                    )
                active_resolver = PricingCatalogResolver.from_projection_file(
                    projection_path,
                    expected_projection_sha256=expected_projection_sha256,
                )
            else:
                active_resolver = PricingCatalogResolver.from_environment(environment)
        snapshot = active_resolver.resolve_snapshot(
            provider=receipt.provider,
            billing_surface=billing_surface,
            model=receipt.model,
            service_tier=resolved_tier,
            batch=batch,
            inference_geo=inference_geo,
            occurred_at=occurred_at,
        )
        return ProviderCallPricing.estimate(snapshot=snapshot, usage=receipt.usage)
    except PricingCatalogError as exc:
        return ProviderCallPricing.unavailable(exc.code)
    except Exception:
        return ProviderCallPricing.unavailable("pricing_resolver_internal_error")


__all__ = [
    "PRICING_CATALOG_SCHEMA",
    "PRICING_PROJECTION_PATH_ENV",
    "PRICING_PROJECTION_SHA256_ENV",
    "VERIFIED_PRICING_PROJECTION_SCHEMA",
    "PricingCatalogError",
    "PricingCatalogResolver",
    "load_pinned_pricing_catalog",
    "resolve_pricing_for_receipt",
]
