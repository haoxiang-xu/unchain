from __future__ import annotations

import copy
import json
import random

import pytest

from unchain import Agent
from unchain.context import (
    AgentSummarizer,
    CompressionError,
    ContextCompressor,
    DirectoryCompressionStore,
    estimate_context_tokens,
)
from unchain.context.suffix_cost import SuffixTokenCost
from unchain.kernel import ModelTurnResult


def history(count=30):
    return [{"role": "developer", "content": "Preserve evidence."}] + [
        message
        for i in range(count)
        for message in (
            {"role": "user", "content": f"question {i} " + "证据😀" * 60},
            {"role": "assistant", "content": f"answer {i} " + "x" * 200},
        )
    ]


def draft(request):
    return {
        "schema": "unchain.context_summary.v1",
        **{
            section: ([{"text": "Continue the verified task.", "source_ids": [request.source_ids[0]]}]
                      if section == "goal" else [])
            for section in ("goal", "constraints", "decisions", "progress", "open_items", "next_steps")
        },
    }


def test_trim_never_calls_model_and_keeps_developer_and_latest_turn(tmp_path):
    original = history()
    before = copy.deepcopy(original)
    def forbidden(_):
        pytest.fail("trim invoked a model")
    compressor = ContextCompressor(
        mode="trim", archive_store=DirectoryCompressionStore(tmp_path), summarizer=forbidden,
    )
    result = compressor.compress(original, max_input_tokens=1700, fixed_overhead_tokens=90)
    assert result.compacted and result.mode == "trim"
    assert list(result.messages)[0] == original[0]
    assert list(result.messages)[-2:] == original[-2:]
    assert result.estimated_tokens + 90 <= 1700
    assert original == before
    checkpoint = compressor.archive_store.get(result.archive_id)
    assert checkpoint is not None
    assert list(checkpoint.messages) == original[1:1 + len(checkpoint.messages)]


def test_summary_preserves_citations_and_replays_after_cold_restart(tmp_path):
    calls = []
    def summarize(request):
        calls.append(request)
        return json.dumps(draft(request))
    source = history()
    compressor = ContextCompressor(
        mode="summary", archive_store=DirectoryCompressionStore(tmp_path),
        summarizer=summarize, summary_reserve_tokens=650, summary_key="fake-model-v1",
    )
    first = compressor.compress(source, max_input_tokens=1800)
    assert first.compacted and len(calls) == 1
    record = compressor.archive_store.get(first.archive_id)
    assert record.summary["goal"][0]["source_ids"][0] in record.source_ids
    assert "UNTRUSTED" in first.messages[1]["content"]
    cold = ContextCompressor(
        mode="summary", archive_store=DirectoryCompressionStore(tmp_path),
        summarizer=lambda _: pytest.fail("cold replay called the model"),
        summary_reserve_tokens=650, summary_key="fake-model-v1",
    )
    assert cold.compress(source, max_input_tokens=1800) == first
    assert list(first.messages)[-2:] == source[-2:]


@pytest.mark.parametrize("change", ["unknown", "citation", "oversize", "schema", "empty"])
def test_invalid_summary_never_archives_or_returns_a_trimmed_view(tmp_path, change):
    def bad(request):
        value = draft(request)
        if change == "unknown": value["extra"] = True
        if change == "citation": value["goal"][0]["source_ids"] = ["invented"]
        if change == "oversize": value["goal"][0]["text"] = "x" * 10000
        if change == "schema": value["schema"] = "v999"
        if change == "empty": value["goal"] = []
        return value
    compressor = ContextCompressor(mode="summary", archive_store=DirectoryCompressionStore(tmp_path),
                                   summarizer=bad, summary_reserve_tokens=650, summary_key="bad-fixture")
    with pytest.raises(CompressionError):
        compressor.compress(history(), max_input_tokens=1800)
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("pending", [
    {"type": "function_call", "call_id": "pending", "name": "write", "arguments": "{}"},
    {"type": "custom_tool_call", "call_id": "pending", "name": "edit", "input": "patch"},
    {"type": "mcp_approval_request", "id": "pending", "name": "send", "arguments": "{}"},
])
def test_summary_defers_unfinished_calls_without_invoking_a_model(tmp_path, pending):
    source = history() + [pending]
    compressor = ContextCompressor(mode="summary", archive_store=DirectoryCompressionStore(tmp_path),
                                   summarizer=lambda _: pytest.fail("pending call summarized"),
                                   summary_reserve_tokens=650, summary_key="pending-fixture")
    with pytest.raises(CompressionError, match="pending"):
        compressor.compress(source, max_input_tokens=1800)
    assert not list(tmp_path.glob("*.json"))


def test_parallel_cross_turn_chain_is_archived_completely_and_exactly(tmp_path):
    source = [
        {"role": "user", "content": "Find the evidence." + "x" * 2000},
        {"role": "assistant", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "read", "arguments": '{"path":"a"}'}},
            {"id": "b", "type": "function", "function": {"name": "read", "arguments": '{"path":"b"}'}},
        ], "content": ""},
        {"role": "tool", "tool_call_id": "a", "content": "exact error E42"},
        {"role": "user", "content": "Also inspect b."},
        {"role": "tool", "tool_call_id": "b", "content": "exact evidence B"},
    ] + history(10)[1:]
    compressor = ContextCompressor(mode="summary", archive_store=DirectoryCompressionStore(tmp_path),
                                   summarizer=draft, summary_reserve_tokens=650, summary_key="chain-fixture")
    result = compressor.compress(source, max_input_tokens=1800)
    archive = compressor.archive_store.get(result.archive_id)
    assert list(archive.messages)[:5] == source[:5]
    assert {item["call_id"] for item in archive.tool_chain} == {"a", "b"}
    assert all(item["call_source_ids"] and item["result_source_ids"] for item in archive.tool_chain)
    assert archive.read(archive.source_ids[2]) == source[2]
    read_tool = result.read_tool(compressor.archive_store)
    assert json.loads(read_tool(limit=4096)["text"])["tool_chain"] == list(archive.tool_chain)
    assert read_tool(archive.source_ids[2])["message"] == source[2]
    with pytest.raises(CompressionError, match="unknown"):
        read_tool("another-archive-source")


def test_store_tampering_and_write_failure_are_errors(tmp_path):
    store = DirectoryCompressionStore(tmp_path)
    compressor = ContextCompressor(mode="trim", archive_store=store)
    result = compressor.compress(history(), max_input_tokens=1700)
    path = tmp_path / f"{result.archive_id}.json"
    record = json.loads(path.read_text())
    record["messages"][0]["content"] = "tampered"
    path.write_text(json.dumps(record))
    with pytest.raises(CompressionError): store.get(result.archive_id)
    class BrokenStore:
        def get(self, _): return None
        def put(self, _): raise OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        ContextCompressor(mode="trim", archive_store=BrokenStore()).compress(history(), max_input_tokens=1700)


def test_real_agent_adapter_is_isolated_and_uses_actual_provider_messages(tmp_path):
    seen = []
    class FakeIO:
        provider = "openai"
        model = "summary-fixture"
        def fetch_turn(self, request):
            seen.append(request)
            payload = json.loads(request.messages[-1]["content"])
            summary = {"schema": "unchain.context_summary.v1", **{
                section: ([{"text": "Keep evidence.", "source_ids": [payload["sources"][0]["source_id"]]}]
                          if section == "goal" else [])
                for section in ("goal", "constraints", "decisions", "progress", "open_items", "next_steps")
            }}
            return ModelTurnResult(assistant_messages=[{"role": "assistant", "content": json.dumps(summary)}],
                                   tool_calls=[], final_text=json.dumps(summary))
    agent = Agent(name="summary-fixture", instructions="Original instructions must be replaced.",
                  model="summary-fixture", model_io_factory=lambda spec, ctx: FakeIO())
    compressor = ContextCompressor(mode="summary", archive_store=DirectoryCompressionStore(tmp_path),
                                   summarizer=AgentSummarizer(agent), summary_reserve_tokens=650)
    result = compressor.compress(history(), max_input_tokens=1800)
    assert result.compacted and len(seen) == 1
    assert "UNVERIFIED" in seen[0].messages[0]["content"]
    assert not seen[0].toolkit.tools


def test_suffix_cost_matches_full_estimator_for_unicode_json_and_media():
    rng = random.Random(20261023)
    for _ in range(40):
        messages = [rng.choice([
            {"role": "user", "content": "证据😀\\\"" * rng.randrange(10)},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": [{"type": "input_image", "image_url": "url"}]},
            {"role": "user", "content": [{"type": "pdf", "page_count": 3}]},
            {"type": "function_call", "call_id": "id", "name": "find", "arguments": {"x": [1, "é"]}},
        ]) for _ in range(rng.randrange(20))]
        counter = SuffixTokenCost(messages)
        for start in range(len(messages) + 1):
            assert counter.estimate(start) == estimate_context_tokens(messages[start:])


@pytest.mark.parametrize("mode", ["trim", "summary"])
def test_no_pressure_does_not_call_model_or_write_archive(tmp_path, mode):
    source = history(1)
    compressor = ContextCompressor(mode=mode, archive_store=DirectoryCompressionStore(tmp_path),
                                   summarizer=lambda _: pytest.fail("no-pressure summary"), summary_key="test")
    result = compressor.compress(source, max_input_tokens=10000)
    assert not result.compacted and list(result.messages) == source
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("provider_pair", [
    [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "p", "name": "read", "input": {"path": "a"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "p", "content": "exact answer"}]},
    ],
    [
        {"role": "assistant", "parts": [{"function_call": {"id": "p", "name": "read", "args": {"path": "a"}}}]},
        {"role": "user", "parts": [{"function_response": {"id": "p", "name": "read", "response": {"answer": "exact"}}}]},
    ],
    [
        {"type": "custom_tool_call", "call_id": "p", "name": "edit", "input": "exact patch"},
        {"type": "custom_tool_call_output", "call_id": "p", "output": "exact result"},
    ],
    [
        {"type": "mcp_approval_request", "id": "p", "name": "send", "arguments": "{}"},
        {"type": "mcp_approval_response", "approval_request_id": "p", "approve": False},
    ],
])
def test_provider_call_and_result_shapes_survive_archiving(tmp_path, provider_pair):
    source = [{"role": "user", "content": "Use tool." + "x" * 2000}, *provider_pair, *history(12)[1:]]
    compressor = ContextCompressor(mode="summary", archive_store=DirectoryCompressionStore(tmp_path),
                                   summarizer=draft, summary_key="providers", summary_reserve_tokens=650)
    result = compressor.compress(source, max_input_tokens=1800)
    archive = compressor.archive_store.get(result.archive_id)
    assert list(archive.messages)[1:3] == provider_pair
    assert len(archive.tool_chain) == 1
    chain = archive.tool_chain[0]
    assert archive.read(chain["call_source_ids"][0]) == provider_pair[0]
    assert archive.read(chain["result_source_ids"][0]) == provider_pair[1]


def test_summary_input_budget_and_model_failure_preserve_originals(tmp_path):
    source = history()
    original = copy.deepcopy(source)
    def unavailable(_): raise RuntimeError("model unavailable")
    compressor = ContextCompressor(mode="summary", archive_store=DirectoryCompressionStore(tmp_path),
                                   summarizer=unavailable, summary_key="failure", summary_reserve_tokens=650)
    with pytest.raises(RuntimeError, match="model unavailable"):
        compressor.compress(source, max_input_tokens=1800)
    compressor.max_summary_input_tokens = 1
    with pytest.raises(CompressionError, match="input exceeds"):
        compressor.compress(source, max_input_tokens=1800)
    assert source == original and not list(tmp_path.glob("*.json"))


def test_reduced_view_is_not_accepted_as_canonical_history(tmp_path):
    compressor = ContextCompressor(mode="trim", archive_store=DirectoryCompressionStore(tmp_path))
    first = compressor.compress(history(), max_input_tokens=1700)
    with pytest.raises(CompressionError, match="canonical"):
        compressor.compress(first.messages, max_input_tokens=1700)


def test_summary_rejects_duplicate_json_keys(tmp_path):
    def duplicate(request):
        content = json.dumps(draft(request))
        return '{"schema":"wrong",' + content[1:]
    compressor = ContextCompressor(mode="summary", archive_store=DirectoryCompressionStore(tmp_path),
                                   summarizer=duplicate, summary_key="duplicate", summary_reserve_tokens=650)
    with pytest.raises(CompressionError, match="duplicate"):
        compressor.compress(history(), max_input_tokens=1800)


def test_custom_summarizer_requires_explicit_policy_identity(tmp_path):
    with pytest.raises(ValueError, match="summary_key"):
        ContextCompressor(mode="summary", archive_store=DirectoryCompressionStore(tmp_path), summarizer=draft)


def test_current_turn_overflow_never_discards_that_turn(tmp_path):
    compressor = ContextCompressor(mode="trim", archive_store=DirectoryCompressionStore(tmp_path))
    with pytest.raises(CompressionError, match="current turn"):
        compressor.compress(history(1), max_input_tokens=50)
    assert not list(tmp_path.iterdir())


def test_read_tool_is_callable_by_a_real_agent_without_exposing_other_archives(tmp_path):
    from unchain.agent import ToolsModule
    from unchain.kernel import ToolCall
    compressor = ContextCompressor(mode="trim", archive_store=DirectoryCompressionStore(tmp_path))
    reduced = compressor.compress(history(), max_input_tokens=1700)
    archive = compressor.archive_store.get(reduced.archive_id)
    source_id = archive.source_ids[0]
    class FakeIO:
        provider = "ollama"
        model = "reader-fixture"
        calls = 0
        def fetch_turn(self, request):
            self.calls += 1
            if self.calls == 1:
                args = {"source_id": source_id}
                return ModelTurnResult(
                    assistant_messages=[{"role": "assistant", "content": "", "tool_calls": [
                        {"id": "read-1", "type": "function", "function": {
                            "name": "context_archive_read", "arguments": json.dumps(args)}}]}],
                    tool_calls=[ToolCall("read-1", "context_archive_read", args)],
                )
            content = json.loads(request.messages[-1]["content"])
            assert content["message"] == archive.read(source_id)
            return ModelTurnResult(assistant_messages=[{"role": "assistant", "content": "Evidence recovered."}],
                                   tool_calls=[], final_text="Evidence recovered.")
    io = FakeIO()
    agent = Agent(name="reader", provider="ollama", model="reader-fixture",
                  modules=(ToolsModule(tools=(reduced.read_tool(compressor.archive_store),)),),
                  model_io_factory=lambda spec, ctx: io)
    result = agent.run(list(reduced.messages), max_iterations=2)
    assert result.status == "completed" and io.calls == 2


@pytest.mark.parametrize("mode", ["trim", "summary"])
def test_reused_provider_call_names_do_not_merge_independent_exchanges(tmp_path, mode):
    source = []
    for i in range(12):
        source.extend([
            {"role": "user", "content": f"Read file {i}." + "x" * 1000},
            {"role": "assistant", "parts": [{"function_call": {"name": "read", "args": {"path": str(i)}}}]},
            {"role": "user", "parts": [{"function_response": {"name": "read", "response": {"value": str(i)}}}]},
        ])
    compressor = ContextCompressor(mode=mode, archive_store=DirectoryCompressionStore(tmp_path),
                                   summarizer=draft, summary_key="reused-name", summary_reserve_tokens=650)
    result = compressor.compress(source, max_input_tokens=1800)
    chain = compressor.archive_store.get(result.archive_id).tool_chain
    assert len(chain) > 1
    assert all(len(item["call_source_ids"]) == len(item["result_source_ids"]) == 1 for item in chain)


def test_evidence_reader_pages_large_unicode_results_without_truncating_the_archive(tmp_path):
    import hashlib
    source = [{"role": "user", "content": "证据😀" * 10000}, *history(8)[1:]]
    compressor = ContextCompressor(mode="trim", archive_store=DirectoryCompressionStore(tmp_path))
    result = compressor.compress(source, max_input_tokens=1800)
    archive = compressor.archive_store.get(result.archive_id)
    reader = result.read_tool(compressor.archive_store)
    offset, pieces = 0, []
    while True:
        page = reader(archive.source_ids[0], offset=offset, limit=997)
        assert len(page["text"]) <= 997
        pieces.append(page["text"])
        if page["next_offset"] is None:
            break
        assert page["message"] is None
        offset = page["next_offset"]
    text = "".join(pieces)
    assert json.loads(text) == source[0]
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == page["sha256"]
    with pytest.raises(CompressionError, match="page bounds"):
        reader(limit=4097)
