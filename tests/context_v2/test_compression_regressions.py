"""Reproductions from the real-model summary and evidence-recovery trial."""

from __future__ import annotations

import hashlib
import json

import pytest

from unchain.context import (
    CompressionError,
    ContextCompressor,
    DirectoryCompressionStore,
)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _history_tail():
    return [
        message
        for index in range(12)
        for message in (
            {"role": "user", "content": f"Continue task {index}. " + "x" * 900},
            {"role": "assistant", "content": f"Task {index} remains unfinished. " + "y" * 300},
        )
    ]


def _draft(source_id):
    return {
        "schema": "unchain.context_summary.v1",
        "goal": [{"text": "Complete the requested task.", "source_ids": [source_id]}],
        "constraints": [],
        "decisions": [],
        "progress": [],
        "open_items": [],
        "next_steps": [],
    }


def _summarize(tmp_path, callback):
    compressor = ContextCompressor(
        mode="summary",
        archive_store=DirectoryCompressionStore(tmp_path),
        summarizer=callback,
        summary_key="fidelity-regression-v1",
        summary_reserve_tokens=650,
    )
    result = compressor.compress(_history_tail(), max_input_tokens=1800)
    return result, compressor.archive_store.get(result.archive_id)


def test_single_enclosing_json_fence_is_accepted_without_changing_evidence(tmp_path):
    result, archive = _summarize(
        tmp_path,
        lambda request: "```json\n" + _canonical(_draft(request.source_ids[0])) + "\n```",
    )

    assert result.compacted
    assert archive.summary == _draft(archive.source_ids[0])
    assert list(archive.messages) == _history_tail()[: len(archive.messages)]


@pytest.mark.parametrize("invalid", ["prose", "multiple_objects", "unknown_id", "near_alias"])
def test_summary_format_tolerance_does_not_guess_content_or_citations(tmp_path, invalid):
    def summarize(request):
        value = _draft(request.source_ids[0])
        if invalid == "unknown_id":
            value["goal"][0]["source_ids"] = ["s999999"]
        elif invalid == "near_alias":
            value["goal"][0]["source_ids"] = ["s00"]
        text = _canonical(value)
        if invalid == "prose":
            return "Here is the summary:\n```json\n" + text + "\n```"
        if invalid == "multiple_objects":
            return "```json\n" + text + "\n" + text + "\n```"
        return text

    with pytest.raises(CompressionError):
        _summarize(tmp_path, summarize)
    assert list(tmp_path.glob("*.json")) == []


def test_model_uses_short_source_aliases_and_host_stores_canonical_citations(tmp_path):
    requests = []

    def summarize(request):
        requests.append(request)
        payload = request.to_dict()
        assert [item["source_id"] for item in payload["sources"]] == [
            f"s{index}" for index in range(len(request.messages))
        ]
        assert [item["message"] for item in payload["sources"]] == list(request.messages)
        return _draft("s0")

    _, archive = _summarize(tmp_path, summarize)

    assert len(requests) == 1
    assert archive.summary == _draft(archive.source_ids[0])
    assert all(source_id.startswith("message-") for source_id in archive.source_ids)
    assert list(archive.source_ids) == [
        f"message-{archive.to_dict()['source_sha256'][:16]}-{index}"
        for index in range(len(archive.messages))
    ]


def _exchange_history(provider):
    source = [{"role": "user", "content": "Repair the module and preserve the original test evidence."}]
    expected = {}
    first_batch = [
        ("blob-before", "read_blob", {"path": "before.log"}, "before " + "甲😀" * 6000),
        ("checks-original", "run_checks", {"suite": "initial"}, 'FAILED original: E42 "起点"'),
        ("blob-after", "read_blob", {"path": "after.log"}, "after " + "乙😀" * 6000),
    ]
    later_batch = [("checks-after-fix", "run_checks", {"suite": "after-fix"}, "PASSED after repair")]

    def append_batch(batch):
        if provider == "anthropic":
            calls = [
                {"type": "tool_use", "id": call_id, "name": name, "input": arguments}
                for call_id, name, arguments, _ in batch
            ]
            results = [
                {"type": "tool_result", "tool_use_id": call_id, "content": output}
                for call_id, _, _, output in batch
            ]
            call_index = len(source)
            source.append({"role": "assistant", "content": [{"type": "text", "text": "Inspecting."}, *calls]})
            result_index = len(source)
            source.append({"role": "user", "content": [*results, {"type": "text", "text": "Evidence collected."}]})
            for (call_id, name, _, _), call, result in zip(batch, calls, results, strict=True):
                expected[call_id] = (name, call, result, call_index, result_index)
        elif provider == "openai_chat":
            calls = [
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": _canonical(arguments)}}
                for call_id, name, arguments, _ in batch
            ]
            call_index = len(source)
            source.append({"role": "assistant", "content": "Inspecting.", "tool_calls": calls})
            for (call_id, name, _, output), call in zip(batch, calls, strict=True):
                result = {"role": "tool", "tool_call_id": call_id, "content": output}
                result_index = len(source)
                source.append(result)
                expected[call_id] = (name, call, result, call_index, result_index)
        else:
            for call_id, name, arguments, output in batch:
                call = {"type": "function_call", "call_id": call_id, "name": name, "arguments": _canonical(arguments)}
                result = {"type": "function_call_output", "call_id": call_id, "output": output}
                call_index = len(source)
                source.append(call)
                result_index = len(source)
                source.append(result)
                expected[call_id] = (name, call, result, call_index, result_index)

    append_batch(first_batch)
    source.append({"role": "user", "content": "The patch is ready; run the same tool again."})
    append_batch(later_batch)
    source.extend(_history_tail())
    return source, expected


def _archived_exchanges(tmp_path, provider):
    source, exchanges = _exchange_history(provider)
    store = DirectoryCompressionStore(tmp_path)
    result = ContextCompressor(mode="trim", archive_store=store).compress(source, max_input_tokens=1800)
    archive = store.get(result.archive_id)
    assert list(archive.messages) == source[: len(archive.messages)]
    assert len(archive.messages) > max(value[4] for value in exchanges.values())
    envelopes = {
        call_id: {
            "call_id": call_id,
            "name": name,
            "call": call,
            "result": item,
            "call_source_ids": [archive.source_ids[call_index]],
            "result_source_ids": [archive.source_ids[result_index]],
        }
        for call_id, (name, call, item, call_index, result_index) in exchanges.items()
    }
    return result.read_tool(store), archive, envelopes


@pytest.mark.parametrize("provider", ["anthropic", "openai_chat", "openai_responses"])
def test_scoped_reader_recovers_original_exchange_among_repeated_calls_and_large_siblings(tmp_path, provider):
    read, _, expected = _archived_exchanges(tmp_path, provider)

    for selector in ({"call_id": "checks-original"}, {"tool_name": "run_checks", "occurrence": 0}):
        page = read(**selector, limit=4096)
        assert page["message"] == expected["checks-original"]
        assert page["text"] == _canonical(expected["checks-original"])
        assert page["sha256"] == hashlib.sha256(page["text"].encode("utf-8")).hexdigest()
        assert page["next_offset"] is None
        assert "PASSED after repair" not in page["text"]

    later = read(tool_name="run_checks", occurrence=1, limit=4096)
    assert later["message"] == expected["checks-after-fix"]


@pytest.mark.parametrize("provider", ["anthropic", "openai_chat", "openai_responses"])
def test_exchange_pages_reconstruct_exact_original_json_and_digest(tmp_path, provider):
    read, _, expected = _archived_exchanges(tmp_path, provider)
    complete = _canonical(expected["blob-before"])
    digest = hashlib.sha256(complete.encode("utf-8")).hexdigest()
    parts = []
    offset = 0

    while True:
        page = read(call_id="blob-before", offset=offset, limit=997)
        assert page["offset"] == offset
        assert page["total_chars"] == len(complete)
        assert page["sha256"] == digest
        assert page["message"] is None
        parts.append(page["text"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]

    assert "".join(parts) == complete
    assert json.loads("".join(parts)) == expected["blob-before"]


@pytest.mark.parametrize("selector", [
    {"call_id": "checks-original", "tool_name": "run_checks"},
    {"call_id": "checks-original", "occurrence": 1},
    {"call_id": "unknown-call"},
    {"tool_name": "unknown-tool"},
    {"tool_name": "run_checks", "occurrence": 2},
    {"tool_name": "run_checks", "occurrence": -1},
    {"tool_name": "run_checks", "occurrence": True},
    {"tool_name": "run_checks", "occurrence": "0"},
    {"occurrence": 1},
])
def test_archive_reader_rejects_ambiguous_or_invalid_exchange_selectors(tmp_path, selector):
    read, _, _ = _archived_exchanges(tmp_path, "anthropic")
    with pytest.raises(CompressionError):
        read(**selector)


@pytest.mark.parametrize("selector", [{"call_id": "checks-original"}, {"tool_name": "run_checks"}])
def test_source_id_cannot_be_mixed_with_exchange_selector(tmp_path, selector):
    read, archive, _ = _archived_exchanges(tmp_path, "anthropic")
    with pytest.raises(CompressionError):
        read(source_id=archive.source_ids[0], **selector)


def test_model_tool_chain_references_the_same_short_source_aliases(tmp_path):
    source, _ = _exchange_history("anthropic")
    requests = []

    def summarize(request):
        requests.append(request)
        payload = request.to_dict()
        aliases = {source_id: f"s{index}" for index, source_id in enumerate(request.source_ids)}
        assert payload["tool_chain"] == [
            {
                "call_id": exchange["call_id"],
                "call_source_ids": [aliases[ref] for ref in exchange["call_source_ids"]],
                "result_source_ids": [aliases[ref] for ref in exchange["result_source_ids"]],
            }
            for exchange in request.tool_chain
        ]
        return _draft("s0")

    store = DirectoryCompressionStore(tmp_path)
    result = ContextCompressor(
        mode="summary", archive_store=store, summarizer=summarize,
        summary_key="alias-tool-chain-v1", summary_reserve_tokens=650,
    ).compress(source, max_input_tokens=1800)

    assert len(requests) == 1
    archive = store.get(result.archive_id)
    assert archive.summary == _draft(archive.source_ids[0])
    assert all(
        ref in archive.source_ids
        for exchange in archive.tool_chain
        for ref in exchange["call_source_ids"] + exchange["result_source_ids"]
    )
