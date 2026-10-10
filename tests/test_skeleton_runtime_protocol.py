"""BC-019: the imported runtime advertises the complete Skeleton baseline."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


MANIFEST_SCHEMA = "unchain.runtime_protocol_manifest.v1"
DIGEST_DOMAIN = b"unchain.runtime_protocol_manifest.v1\\u0000"
SKELETON_FEATURES = [
    "code_python_v1",
    "draft_diagnostics_v1",
    "durable_exclusive_dag_v2",
    "end_only_root_completion_v1",
    "explicit_bindings_v1",
    "hook_phase_receipts_v1",
    "kernel_profiles_v1",
    "legacy_conversion_v1",
    "non_llm_workflow_v1",
    "pinned_workflow_calls_v1",
    "production_host_bindings_v1",
    "skill_read_v1",
    "structured_output_v1",
    "subagent_graphs_v1",
    "tool_call_v1",
]
# Frozen from the actual imported producer before the BC-019 addition. These
# are independent expected wire values, not data read from the producer parser.
BASELINE_FEATURES = {
    "context_memory": [
        "artifact_handoff", "canonical_journal",
        "chat_deletion_sqlite_scope_closure", "context_compiler",
        "context_content_paging_v1", "context_content_read_recovery_v1",
        "context_contribution_manifest_v1",
        "generation_rebase_live_interaction_cycles",
        "interaction_resolution_compat", "long_term_promotion", "memory_curator",
        "memory_toolkit", "memory_workspace", "tool_output_management_v1",
    ],
    "durable_interaction": [
        "cancel_pending", "expected_interaction_id_cas", "fresh_run_lineage",
        "graph_interaction_lineage_preflight_v1", "host_controlled_resume",
        "interaction_resolution_atomic_acceptance_v1",
    ],
    "provider_turn_ownership": [
        "atomic_receipt_cas", "auxiliary_calls", "enforce_mode", "graph_runs",
        "memory_off", "ollama_reasoning_preview_v1",
        "provider_response_outcomes_v1", "provider_uncertainty_diagnostics_v1",
        "subagent_runs",
    ],
    "run_bundle": [
        "canonical_metrics", "completion_diagnostics_ref",
        "context_composition_ref_v1", "continuation_claim",
        "immutable_pricing_snapshot", "provider_call_set_union",
        "provider_call_usage_v1", "run_bundle_v1", "run_bundle_v2",
    ],
    "skills": [
        "active_skills_snapshot_v1", "catalog_v1", "skill_md_registry_v1",
        "skill_tool_v1", "toolkit_embedded_skills_v1", "user_invocation_v1",
    ],
}


def _actual_manifest():
    from unchain.runtime.runtime_protocol import runtime_protocol_manifest

    return runtime_protocol_manifest()


def _skeleton_protocol(manifest):
    matches = [item for item in manifest["protocols"]
               if item["id"] == "skeleton_execution"]
    assert len(matches) == 1, "the actual imported runtime must advertise Skeleton"
    return matches[0]


def _expected_skeleton_protocol():
    return {
        "features": SKELETON_FEATURES,
        "id": "skeleton_execution",
        "major": 1,
        "minor": 1,
    }


def test_actual_runtime_advertises_exact_closed_skeleton_execution_v1():
    protocol = _skeleton_protocol(_actual_manifest())

    assert set(protocol) == {"features", "id", "major", "minor"}
    assert protocol == _expected_skeleton_protocol()
    assert type(protocol["major"]) is int
    assert type(protocol["minor"]) is int
    assert len(protocol["features"]) == len(set(protocol["features"])) == 15
    assert protocol["features"] == sorted(
        protocol["features"], key=lambda item: item.encode("utf-8"),
    )


@pytest.mark.parametrize("protocol_id", tuple(BASELINE_FEATURES))
def test_skeleton_addition_preserves_each_existing_protocol_exactly(protocol_id):
    matches = [item for item in _actual_manifest()["protocols"]
               if item["id"] == protocol_id]

    assert matches == [{
        "features": BASELINE_FEATURES[protocol_id],
        "id": protocol_id,
        "major": 1,
        "minor": 0,
    }]


def test_expanded_manifest_has_exact_envelope_and_canonical_protocol_order():
    manifest = _actual_manifest()

    assert set(manifest) == {"manifest_digest", "protocols", "runtime", "schema"}
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["runtime"] == "unchain"
    ids = [item["id"] for item in manifest["protocols"]]
    assert ids == sorted(
        [*BASELINE_FEATURES, "skeleton_execution"],
        key=lambda item: item.encode("utf-8"),
    )
    for item in manifest["protocols"]:
        assert set(item) == {"features", "id", "major", "minor"}
        assert type(item["features"]) is list
        assert item["features"] == sorted(
            set(item["features"]), key=lambda feature: feature.encode("utf-8"),
        )


def test_actual_manifest_digest_is_independently_recomputed_and_repeatable():
    first = _actual_manifest()
    second = _actual_manifest()
    body = {key: first[key] for key in ("protocols", "runtime", "schema")}
    canonical = json.dumps(
        body, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")

    assert first == second
    assert first is not second
    assert first["manifest_digest"] == "sha256:" + hashlib.sha256(
        DIGEST_DOMAIN + canonical,
    ).hexdigest()


def test_mutating_a_returned_skeleton_manifest_does_not_change_the_next_call():
    first = _actual_manifest()
    skeleton = _skeleton_protocol(first)
    expected_digest = first["manifest_digest"]
    skeleton["features"].clear()
    skeleton["major"] = 99
    first["manifest_digest"] = "sha256:" + "0" * 64

    fresh = _actual_manifest()
    assert _skeleton_protocol(fresh) == _expected_skeleton_protocol()
    assert fresh["manifest_digest"] == expected_digest


def test_fresh_process_exports_skeleton_metadata_without_credentials_or_network(tmp_path):
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
import json
import socket

def forbidden_network(*args, **kwargs):
    raise AssertionError('runtime metadata must not initialize network access')

socket.socket.connect = forbidden_network
socket.socket.connect_ex = forbidden_network
socket.socket.sendto = forbidden_network
socket.create_connection = forbidden_network
from unchain.runtime.runtime_protocol import runtime_protocol_manifest
print(json.dumps(runtime_protocol_manifest(), sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(source_root),
            "PYTHONIOENCODING": "utf-8",
        },
        capture_output=True, text=True, timeout=20,
    )

    assert result.returncode == 0, result.stderr
    exported = json.loads(result.stdout)
    assert _skeleton_protocol(exported) == _expected_skeleton_protocol()
    assert exported == _actual_manifest()
