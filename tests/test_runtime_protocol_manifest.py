from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from pathlib import Path

import pytest


SCHEMA = "unchain.runtime_protocol_manifest.v1"
DIGEST_DOMAIN = b"unchain.runtime_protocol_manifest.v1\\u0000"


def _canonical_body_bytes(value: dict[str, object]) -> bytes:
    body = {
        "protocols": value["protocols"],
        "runtime": value["runtime"],
        "schema": value["schema"],
    }
    return json.dumps(
        body,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _resign(value: dict[str, object]) -> dict[str, object]:
    resigned = copy.deepcopy(value)
    resigned["manifest_digest"] = "sha256:" + hashlib.sha256(
        DIGEST_DOMAIN + _canonical_body_bytes(resigned)
    ).hexdigest()
    return resigned


def test_code_backed_runtime_protocol_manifest_is_closed_and_deterministic() -> None:
    from unchain.runtime import runtime_protocol as producer
    from unchain.runtime.runtime_protocol import (
        RuntimeProtocolManifest,
        build_runtime_protocol_manifest,
        runtime_protocol_manifest,
    )

    first = runtime_protocol_manifest()
    second = runtime_protocol_manifest()

    assert first == second == build_runtime_protocol_manifest().to_dict()
    assert first is not second
    assert set(first) == {"manifest_digest", "protocols", "runtime", "schema"}
    assert first["schema"] == SCHEMA
    assert first["runtime"] == "unchain"
    assert first["manifest_digest"] == "sha256:" + hashlib.sha256(
        DIGEST_DOMAIN + _canonical_body_bytes(first)
    ).hexdigest()
    assert RuntimeProtocolManifest.from_dict(first).to_dict() == first
    assert Path(producer.__file__).name == "runtime_protocol.py"


def test_runtime_protocol_manifest_advertises_every_frozen_required_feature() -> None:
    from unchain.runtime.runtime_protocol import runtime_protocol_manifest

    protocols = {
        item["id"]: set(item["features"])
        for item in runtime_protocol_manifest()["protocols"]
    }

    assert "chat_deletion_sqlite_scope_closure" in protocols["context_memory"]
    assert (
        "context_contribution_manifest_v1" in protocols["context_memory"]
    )
    assert (
        "generation_rebase_live_interaction_cycles"
        in protocols["context_memory"]
    )
    assert "tool_output_management_v1" in protocols["context_memory"]
    assert "interaction_resolution_compat" in protocols["context_memory"]
    assert "expected_interaction_id_cas" in protocols["durable_interaction"]
    assert (
        "graph_interaction_lineage_preflight_v1"
        in protocols["durable_interaction"]
    )
    assert (
        "interaction_resolution_atomic_acceptance_v1"
        in protocols["durable_interaction"]
    )
    assert "ollama_reasoning_preview_v1" in protocols["provider_turn_ownership"]
    assert {
        "canonical_metrics",
        "completion_diagnostics_ref",
        "context_composition_ref_v1",
        "continuation_claim",
        "immutable_pricing_snapshot",
        "provider_call_set_union",
        "provider_call_usage_v1",
        "run_bundle_v1",
        "run_bundle_v2",
    }.issubset(protocols["run_bundle"])


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: {**value, "unknown": True},
        lambda value: {**value, "schema": "unchain.runtime_protocol_manifest.v2"},
        lambda value: {**value, "runtime": "fork"},
        lambda value: {**value, "manifest_digest": "sha256:" + "A" * 64},
        lambda value: {
            **value,
            "protocols": list(reversed(value["protocols"])),
        },
        lambda value: {
            **value,
            "protocols": [
                {
                    **value["protocols"][0],
                    "features": list(reversed(value["protocols"][0]["features"])),
                },
                *value["protocols"][1:],
            ],
        },
        lambda value: {
            **value,
            "protocols": [value["protocols"][0], *value["protocols"]],
        },
        lambda value: {
            **value,
            "protocols": [
                {
                    **value["protocols"][0],
                    "features": [
                        value["protocols"][0]["features"][0],
                        *value["protocols"][0]["features"],
                    ],
                },
                *value["protocols"][1:],
            ],
        },
        lambda value: {
            **value,
            "protocols": [
                {**value["protocols"][0], "major": True},
                *value["protocols"][1:],
            ],
        },
        lambda value: {
            **value,
            "protocols": [
                {**value["protocols"][0], "minor": -1},
                *value["protocols"][1:],
            ],
        },
        lambda value: {
            **value,
            "protocols": [
                {**value["protocols"][0], "minor": 1 << 53},
                *value["protocols"][1:],
            ],
        },
        lambda value: {
            **value,
            "protocols": [
                {**value["protocols"][0], "unknown": True},
                *value["protocols"][1:],
            ],
        },
    ),
)
def test_strict_parser_rejects_shape_order_type_and_digest_mutations(mutation) -> None:
    from unchain.runtime.runtime_protocol import (
        RuntimeProtocolManifest,
        RuntimeProtocolManifestError,
        runtime_protocol_manifest,
    )

    mutated = mutation(runtime_protocol_manifest())

    with pytest.raises(RuntimeProtocolManifestError):
        RuntimeProtocolManifest.from_dict(mutated)


def test_strict_parser_rejects_non_nfc_strings_even_with_a_matching_digest() -> None:
    from unchain.runtime.runtime_protocol import (
        RuntimeProtocolManifest,
        RuntimeProtocolManifestError,
        runtime_protocol_manifest,
    )

    value = runtime_protocol_manifest()
    decomposed = unicodedata.normalize("NFD", "café")
    assert decomposed != unicodedata.normalize("NFC", decomposed)
    value["protocols"] = [
        *value["protocols"],
        {"features": [], "id": decomposed, "major": 1, "minor": 0},
    ]
    value = _resign(value)

    with pytest.raises(RuntimeProtocolManifestError, match="NFC"):
        RuntimeProtocolManifest.from_dict(value)


@pytest.mark.parametrize("location", ("optional_protocol", "optional_feature"))
def test_strict_parser_reports_lone_surrogate_as_manifest_error(
    location: str,
) -> None:
    from unchain.runtime.runtime_protocol import (
        RuntimeProtocolManifest,
        RuntimeProtocolManifestError,
        runtime_protocol_manifest,
    )

    value = runtime_protocol_manifest()
    if location == "optional_protocol":
        value["protocols"].append(
            {"features": [], "id": "\ud800", "major": 1, "minor": 0}
        )
    else:
        value["protocols"][0]["features"].append("\ud800")

    with pytest.raises(RuntimeProtocolManifestError, match="strict UTF-8"):
        RuntimeProtocolManifest.from_dict(value)


def test_whitespace_bearing_nfc_optional_protocol_and_feature_are_preserved() -> None:
    from unchain.runtime.runtime_protocol import (
        RuntimeProtocolManifest,
        runtime_protocol_manifest,
    )

    value = runtime_protocol_manifest()
    value["protocols"].insert(
        0,
        {
            "features": [" optional_feature "],
            "id": " optional_protocol ",
            "major": 1,
            "minor": 0,
        },
    )
    value = _resign(value)

    assert RuntimeProtocolManifest.from_dict(value).to_dict() == value


def test_unknown_canonical_protocols_and_features_round_trip() -> None:
    from unchain.runtime.runtime_protocol import (
        RuntimeProtocolManifest,
        runtime_protocol_manifest,
    )

    value = runtime_protocol_manifest()
    value["protocols"][0]["features"].append("zz_optional_feature")
    value["protocols"].append(
        {
            "features": ["optional_feature"],
            "id": "zz_optional_protocol",
            "major": 9,
            "minor": 3,
        }
    )
    value = _resign(value)

    assert RuntimeProtocolManifest.from_dict(value).to_dict() == value
