"""Reproduce the deterministic mixed native-turn matrix on a pinned runtime.

No credentials, network requests, production profile or raw incident captures
are used. Test clients feed fixed provider-shaped responses to real adapters.
Case definitions remain in the candidate test source, even for a baseline run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--case-source-root", type=Path,
                        default=Path(__file__).resolve().parents[4])
    parser.add_argument("--expect", choices=("red", "green"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime_root = args.runtime_root.resolve()
    case_root = args.case_source_root.resolve()
    sys.path.insert(0, str(runtime_root / "src"))
    sys.path.insert(0, str(case_root / "tests/context_v2"))
    import unchain
    from unchain.providers.context_assembler import ProviderContextProjectionError

    if not Path(unchain.__file__).resolve().is_relative_to(runtime_root):
        raise RuntimeError("imported runtime does not match --runtime-root")
    relative_cases = Path("tests/context_v2/test_context_provider_turn_cross_provider.py")
    cases = load_module("ticket390_candidate_cases", case_root / relative_cases)
    if runtime_root != case_root:
        baseline = load_module("ticket390_baseline_factory", runtime_root / relative_cases)
        cases._runtime = baseline._runtime
    control_path = runtime_root / "tests/context_v2/test_context_provider_turn_boundary.py"
    control = load_module("ticket390_openai_control", control_path)
    signed_path = case_root / "tests/context_v2/test_ticket390_signed_gemini.py"
    signed = load_module("ticket390_signed_gemini", signed_path)
    matrix = [
        ("anthropic_mixed_single", cases.test_durable_anthropic_mixed_text_and_tool_turn_replays_complete_semantics),
        ("hyperspace_mixed_single", cases.test_durable_hyperspace_mixed_text_and_tool_turn_replays_complete_semantics),
        ("hyperspace_mixed_four", cases.test_durable_hyperspace_four_tool_turn_replays_complete_semantics),
        ("gemini_signed_mixed", lambda path: signed.test_signed_gemini_mixed_turn_retains_visible_text(
            path, runtime_factory=cases._runtime)),
        ("openai_control", control.test_tool_bearing_turn_persists_result_and_continues_through_same_boundary),
    ]
    results = []
    for name, test in matrix:
        expected = "green" if name == "openai_control" else args.expect
        record = {"case": name, "expected": expected}
        with tempfile.TemporaryDirectory(prefix="ticket390-matrix-") as directory:
            try:
                test(Path(directory))
            except Exception as error:
                record.update(error_type=type(error).__name__, error=str(error),
                              traceback=traceback.format_exc())
                record["matched_expectation"] = (
                    expected == "red"
                    and isinstance(error, ProviderContextProjectionError)
                    and str(error) == "provider-native tool/reasoning segment was mutated ambiguously"
                )
            else:
                record["matched_expectation"] = expected == "green"
        results.append(record)
        print(name, "PASS" if record["matched_expectation"] else "FAIL",
              record.get("error", "completed"))
    report = {
        "schema": "ticket390.deterministic_replay_evidence.v1",
        "provenance": "deterministic provider-shaped SDK responses; not production captures",
        "runtime_root": str(runtime_root), "case_source_root": str(case_root),
        "runtime_head": subprocess.check_output(
            ["git", "-C", str(runtime_root), "rev-parse", "HEAD"], text=True).strip(),
        "case_source_sha256": digest(case_root / relative_cases),
        "harness_sha256": digest(Path(__file__)),
        "control_source_sha256": digest(control_path),
        "signed_gemini_source_sha256": digest(signed_path),
        "runtime_source_sha256": {
            str(relative): digest(runtime_root / relative)
            for relative in (
                Path("src/unchain/context/compiler.py"),
                Path("src/unchain/context/coordinator.py"),
                Path("src/unchain/providers/context_assembler.py"),
            )
        },
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("anthropic", "openai", "google-genai", "pytest")},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0 if all(record["matched_expectation"] for record in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
