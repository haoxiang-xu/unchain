import hashlib
import copy
import json
import re
from pathlib import Path


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "pupu_p0"
PARTITION_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "compiler_p0"
    / "pupu_host_partition.json"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def test_pupu_p0_manifest_digests_and_provenance_are_valid() -> None:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == "unchain.context_v2.golden.v1"
    assert manifest["exporter_version"] == 1
    assert manifest["memory_schema_version"] == 4
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["source"]["head_sha"])
    assert manifest["source"]["dirty"] is True
    assert manifest["source"]["files"] == [
        "unchain_runtime/server/memory_v2_context.py",
        "unchain_runtime/server/memory_v2_curator.py",
        "unchain_runtime/server/memory_v2_legacy_adapter.py",
        "unchain_runtime/server/memory_v2_rollout.py",
        "unchain_runtime/server/memory_v2_sanitizer.py",
        "unchain_runtime/server/memory_v2_toolkit.py",
    ]
    assert SHA256_RE.fullmatch(manifest["source"]["sha256"])

    entries = manifest["fixtures"]
    assert [entry["file"] for entry in entries] == [
        "below_pressure.json",
        "legacy_partial.json",
        "tool_and_pinned.json",
    ]
    for entry in entries:
        raw = (FIXTURE_ROOT / entry["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        fixture = json.loads(raw)
        assert list(fixture) == ["schema", "case", "source_contracts", "input", "expected"]
        assert fixture["schema"] == "unchain.context_v2.fixture.v1"
        assert fixture["case"] == entry["case"]
        assert fixture["source_contracts"]
        assert all(contract["path"].startswith("unchain_runtime/server/") for contract in fixture["source_contracts"])
        assert all(contract["symbol"] for contract in fixture["source_contracts"])
        assert "pupu://" not in json.dumps(fixture["expected"], sort_keys=True)


def test_context_compiler_public_contract_is_available() -> None:
    from unchain.context import ContextCompileRequest, ContextCompiler

    partitions = json.loads(PARTITION_PATH.read_text(encoding="utf-8"))["cases"]
    for fixture_path in sorted(FIXTURE_ROOT.glob("*.json")):
        if fixture_path.name == "manifest.json":
            continue
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        ownership = partitions[fixture["case"]]
        assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == ownership[
            "source_fixture_sha256"
        ]
        fixture_input = copy.deepcopy(fixture["input"])
        if fixture["case"] == "tool_and_pinned":
            # PuPu persisted the full tool body before invoking the P0 compiler.
            # Materialize that already-authorized portable descriptor here; the
            # Unchain compiler deliberately owns neither object storage nor URI
            # allocation.
            for store_seq, event in enumerate(
                fixture_input["semantic_events"],
                start=1,
            ):
                event.setdefault("store_seq", store_seq)
                synthetic = event.pop("synthetic_result", None)
                if synthetic is None:
                    continue
                result_body = {
                    synthetic["field"]: synthetic["fill"]
                    * int(synthetic["char_count"])
                }
                body = json.dumps(
                    result_body,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                ref = {
                    "kind": "artifact",
                    "id": event["call_id"],
                    "revision": 1,
                }
                digest = hashlib.sha256(body).hexdigest()
                event.update(
                    {
                        "result": {
                            "preview": body[:1_200].decode(
                                "utf-8", errors="replace"
                            ),
                            "full_output_ref": ref,
                            "content_bytes": len(body),
                            "content_sha256": digest,
                        },
                        "result_bytes": len(body),
                        "result_sha256": digest,
                        "full_output_ref": ref,
                    }
                )
            host_materialized = ownership.get("host_materialized_input")
            if host_materialized:
                target = next(
                    event
                    for event in fixture_input["semantic_events"]
                    if event.get("event_id")
                    == host_materialized["semantic_event_id"]
                )
                target["content_bytes"] = host_materialized["content_bytes"]
                target["content_sha256"] = host_materialized["content_sha256"]
        request = ContextCompileRequest.from_dict(fixture_input)
        result = ContextCompiler().compile(request)
        serialized = result.to_dict()
        comparable = serialized.get("projections", {}).get(
            "unchain.context_v2.comparable.v1"
        )
        if comparable is None and serialized.get("schema") == "unchain.context_v2.comparable.v1":
            comparable = serialized
        expected = copy.deepcopy(fixture["expected"])
        if fixture["case"] == "tool_and_pinned":
            expected["pending_interaction"]["request"]["store_seq"] = 4
            # Preserve the original PuPu snapshot and its provenance hash;
            # #447 changes only this host-owned model presentation label.
            assert expected["pending_interaction"].pop("trust") == "UNTRUSTED_DATA"
            expected["pending_interaction"]["evidence_kind"] = "historical_record"
        compiler_owned = ownership["compiler_owned_expected_fields"]
        if compiler_owned == "*":
            assert comparable == expected
            continue
        compiler_owned = set(compiler_owned)
        host_owned = set(ownership["host_owned_expected_fields"])
        assert comparable == {key: expected[key] for key in compiler_owned}
        assert set(expected) == compiler_owned | host_owned
        assert not compiler_owned & host_owned
