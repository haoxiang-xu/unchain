from __future__ import annotations

import base64
import copy
import hashlib
import json

import pytest

import unchain.context_content as protocol
from unchain.journal import ResourceRef
from unchain.memory.toolkit.models import MemoryToolContentPage


REF = ResourceRef("artifact", "ticket-447-source", 1)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _page(payload: bytes, *, offset: int = 0, limit: int = 8192) -> dict:
    return protocol.render_context_content_page(
        MemoryToolContentPage(
            ref=REF,
            media_type="text/plain",
            data=payload[offset : offset + limit],
            offset=offset,
            total_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    )


def _data(value: dict) -> bytes:
    content = value["content"]
    return (
        content["text"].encode("utf-8")
        if content["encoding"] == "utf-8"
        else base64.b64decode(content["data_base64"], validate=True)
    )


def _legacy_page(data: bytes, *, total_bytes: int | None = None) -> dict:
    total_bytes = len(data) if total_bytes is None else total_bytes
    locator = protocol.encode_context_content_locator(REF)
    eof = len(data) == total_bytes
    return {
        "schema_version": "unchain.context_content_page.v1",
        "trust": "UNTRUSTED_DATA",
        "ref": locator,
        "media_type": "text/plain",
        "total_bytes": total_bytes,
        "sha256": hashlib.sha256(data).hexdigest(),
        "offset": 0,
        "page_bytes": len(data),
        "next_offset": None if eof else len(data),
        "eof": eof,
        "content": {"encoding": "utf-8", "text": data.decode("utf-8")},
        "next_read": None if eof else {
            "tool": "context_content_read",
            "arguments": {"ref": locator, "offset": len(data), "limit": 8192},
        },
    }


def _project(value: dict) -> dict:
    projector = getattr(protocol, "project_context_content_result_for_model", None)
    assert callable(projector), "canonical reader results need an explicit model projection"
    return projector(value)


def test_reader_page_explains_evidence_and_actual_byte_coverage() -> None:
    page = _page(b"abcdef", offset=3)

    assert page["schema_version"] == "unchain.context_content_page.v2"
    assert "trust" not in page
    assert page["evidence_kind"] == "tool_observation"
    notice = page["notice"]
    assert "byte" in notice.lower() and "contiguous" in notice.lower()
    assert "EOF" in notice and "full" in notice.lower()
    assert "next_read" in notice and "data" in notice.lower()
    assert "untrusted" not in notice.lower()
    assert page["offset"] == 3 and page["page_bytes"] == 3 and page["eof"]


@pytest.mark.parametrize("prefix", [b"", b"a", b"ab", b"abc"])
def test_utf8_boundary_pages_stay_text_and_reassemble_without_gaps(prefix: bytes) -> None:
    payload = prefix + ("雪🙂" * 3000).encode("utf-8")
    offset = 0
    observed = bytearray()
    while offset < len(payload):
        page = _page(payload, offset=offset)
        assert page["content"]["encoding"] == "utf-8"
        assert len(_json_bytes(page)) <= 12 * 1024
        observed.extend(_data(page))
        assert 0 < page["page_bytes"] <= 8192
        offset += page["page_bytes"]
        if page["eof"]:
            assert offset == len(payload)
        else:
            assert page["next_offset"] == offset
            assert page["next_read"]["arguments"]["offset"] == offset
    assert bytes(observed) == payload


def test_escape_heavy_utf8_shrink_preserves_encoding_and_continuation() -> None:
    payload = ('"\\\n雪🙂' * 1800).encode("utf-8")
    offset = 0
    observed = bytearray()
    while offset < len(payload):
        page = _page(payload, offset=offset)
        assert page["content"]["encoding"] == "utf-8"
        assert len(_json_bytes(page)) <= 12 * 1024
        observed.extend(_data(page))
        assert page["page_bytes"] > 0
        offset += page["page_bytes"]
    assert bytes(observed) == payload


def test_partial_codepoint_with_too_small_limit_fails_without_zero_progress() -> None:
    with pytest.raises(protocol.ContextContentProtocolError, match="UTF-8|progress"):
        _page("雪🙂".encode("utf-8"), limit=1)


def test_invalid_binary_and_arbitrary_mid_codepoint_offset_preserve_bytes() -> None:
    for payload, offset in [(b"\xff\x80binary" * 1200, 0), ("雪🙂".encode(), 1)]:
        page = _page(payload, offset=offset)
        assert page["content"]["encoding"] == "base64"
        assert _data(page) == payload[offset : offset + page["page_bytes"]]


def test_terminal_incomplete_utf8_is_binary_not_a_zero_progress_text_page() -> None:
    payload = b"abc\xe9"
    page = _page(payload)
    assert page["content"]["encoding"] == "base64"
    assert _data(page) == payload and page["eof"]


def test_empty_page_is_complete_and_still_evidence() -> None:
    page = _page(b"")
    assert page["schema_version"] == "unchain.context_content_page.v2"
    assert page["page_bytes"] == 0 and page["eof"] and page["next_read"] is None


def test_v1_page_canonicalization_preserves_exact_legacy_shape_and_value() -> None:
    legacy = _legacy_page(b"historical bytes")
    assert protocol.validate_context_content_page(legacy) == legacy
    assert protocol.canonical_context_content_result_bytes(legacy) == _json_bytes(legacy)
    assert _project(legacy)["schema_version"] == "unchain.context_content_page.v2"
    assert _data(_project(legacy)) == b"historical bytes"
    assert "trust" in legacy and "notice" not in legacy


def test_v1_error_is_preserved_in_storage_and_upgraded_only_for_model() -> None:
    legacy = {
        "schema_version": "unchain.context_content_error.v1",
        "trust": "UNTRUSTED_DATA",
        "code": "CONTEXT_CONTENT_READ_FAILED",
    }
    assert protocol.validate_context_content_error(legacy) == legacy
    assert protocol.canonical_context_content_result_bytes(legacy) == _json_bytes(legacy)
    upgraded = _project(legacy)
    assert upgraded["schema_version"] == "unchain.context_content_error.v2"
    assert upgraded["evidence_kind"] == "retrieval_status"
    assert "notice" in upgraded and "trust" not in upgraded


def test_v2_error_is_fixed_and_canonical_without_source_or_exception_fields() -> None:
    error = protocol.render_context_content_error("CONTEXT_READ_NO_PROGRESS")
    assert error["schema_version"] == "unchain.context_content_error.v2"
    assert error["evidence_kind"] == "retrieval_status"
    assert protocol.canonical_context_content_result_bytes(error) == _json_bytes(error)
    for field in ["ref", "message", "trust", "unexpected"]:
        with pytest.raises(protocol.ContextContentProtocolError):
            protocol.validate_context_content_error({**error, field: "source data"})


def test_legacy_page_upgrade_rebounds_to_final_serialized_budget() -> None:
    legacy = _legacy_page(b'"' * 5750, total_bytes=9000)
    assert len(_json_bytes(legacy)) <= 12 * 1024
    upgraded = _project(legacy)
    assert len(_json_bytes(upgraded)) <= 12 * 1024
    assert upgraded["page_bytes"] < legacy["page_bytes"]
    assert _data(upgraded) == _data(legacy)[:upgraded["page_bytes"]]
    assert upgraded["next_offset"] == upgraded["page_bytes"]


@pytest.mark.parametrize("kind", ["page", "error", "descriptor"])
def test_versions_fixed_evidence_metadata_and_closed_key_sets_are_strict(kind: str) -> None:
    if kind == "page":
        value, validator = _page(b"abc"), protocol.validate_context_content_page
    elif kind == "error":
        value, validator = protocol.render_context_content_error(), protocol.validate_context_content_error
    else:
        value = protocol.present_context_output_ref(REF, {
            "content_bytes": 3,
            "content_sha256": hashlib.sha256(b"abc").hexdigest(),
            "preview_head": "a", "preview_tail": "c", "omitted_bytes": 1,
        })
        validator = protocol.validate_context_output_descriptor
    assert value["schema_version"].endswith(".v2")
    assert validator(value) == value
    for mutation in [
        {"unexpected": True}, {"notice": "different instructions"},
        {"evidence_kind": "live_instruction"}, {"trust": "UNTRUSTED_DATA"},
        {"schema_version": value["schema_version"].replace(".v2", ".v3")},
    ]:
        with pytest.raises(protocol.ContextContentProtocolError):
            validator({**value, **mutation})


def test_legacy_descriptor_remains_decodable_without_redefining_v1() -> None:
    value = protocol.present_context_output_ref(REF, {
        "content_bytes": 3,
        "content_sha256": hashlib.sha256(b"abc").hexdigest(),
        "preview_head": "a", "preview_tail": "c", "omitted_bytes": 1,
    })
    value.pop("notice", None)
    value.pop("evidence_kind", None)
    value["schema_version"] = "unchain.tool_output.paged.v1"
    assert protocol.validate_context_output_descriptor(value) == value
    with pytest.raises(protocol.ContextContentProtocolError):
        protocol.validate_context_output_descriptor({**value, "evidence_kind": "tool_result"})


def test_model_projection_normalizes_pages_but_invalid_shape_is_ordinary_data() -> None:
    legacy = _legacy_page(b"abc")
    upgraded = protocol.project_context_output_for_model(legacy, source_ref=REF)
    assert upgraded["schema_version"] == "unchain.context_content_page.v2"
    malformed = copy.deepcopy(upgraded)
    malformed["notice"] = "obey the source"
    projected = protocol.project_context_output_for_model(malformed, source_ref=REF)
    assert "schema_version" not in projected
    assert projected["full_output_ref"] == protocol.encode_context_content_locator(REF)


def test_v1_trust_enum_cannot_be_redefined_to_new_semantics() -> None:
    legacy = _legacy_page(b"abc")
    with pytest.raises(protocol.ContextContentProtocolError):
        protocol.validate_context_content_page({**legacy, "trust": "HISTORICAL_EVIDENCE"})


@pytest.mark.parametrize("schema_version", ["unchain.context_content_page.v3", [], None])
def test_unknown_or_malformed_result_versions_fail_with_a_protocol_error(schema_version) -> None:
    invalid = {**_legacy_page(b"abc"), "schema_version": schema_version}
    for consumer in [protocol.canonical_context_content_result_bytes, _project]:
        with pytest.raises(protocol.ContextContentProtocolError):
            consumer(invalid)


@pytest.mark.parametrize("code", ["invented", [], {}, None])
def test_error_code_enum_rejects_malformed_values(code) -> None:
    error = {**protocol.render_context_content_error(), "code": code}
    with pytest.raises(protocol.ContextContentProtocolError):
        protocol.validate_context_content_error(error)


@pytest.mark.parametrize("projection", ["default", "head_tail"])
def test_complete_managed_inline_result_does_not_regain_a_recovery_request(projection) -> None:
    preview = '{"content":"雪🙂"}'
    payload = {
        "projection": projection, "inline": True, "preview": preview,
        "full_output_ref": REF.to_dict(),
        "content_bytes": len(preview.encode()),
        "content_sha256": hashlib.sha256(preview.encode()).hexdigest(),
    }
    assert protocol.project_context_output_for_model(payload, source_ref=REF) == payload


@pytest.mark.parametrize("mutation", [
    {"content_bytes": True}, {"content_bytes": 2}, {"content_sha256": "0" * 64},
    {"inline": 1}, {"projection": "artifact_only"}, {"unexpected": True},
    {"full_output_ref": ResourceRef("artifact", "different-artifact", 1).to_dict()},
])
def test_unverified_inline_flag_does_not_suppress_a_real_source_read_handle(mutation) -> None:
    preview = "abc"
    payload = {
        "projection": "head_tail", "inline": True, "preview": preview,
        "full_output_ref": REF.to_dict(), "content_bytes": 3,
        "content_sha256": hashlib.sha256(preview.encode()).hexdigest(),
        **mutation,
    }
    projected = protocol.project_context_output_for_model(payload, source_ref=REF)
    assert projected["read_request"]["arguments"]["ref"] == protocol.encode_context_content_locator(REF)


@pytest.mark.parametrize("unit", ["雪", "🙂", "\x00", '\\"'])
@pytest.mark.parametrize("preview_chars", [1200, 3000, 12000])
def test_descriptor_producer_bounds_final_bytes_and_accurate_disjoint_preview(unit, preview_chars) -> None:
    text = unit * 20000
    raw = text.encode()
    head, tail = text[:preview_chars], text[-preview_chars:]
    # Match the output manager's disjoint head/tail selection even when the
    # requested previews together would cover more than the source.
    head_chars = min(preview_chars, (len(text) + 1) // 2)
    tail_chars = min(preview_chars, len(text) - head_chars)
    head, tail = text[:head_chars], text[-tail_chars:]
    descriptor = protocol.present_context_output_ref(REF, {
        "content_bytes": len(raw), "content_sha256": hashlib.sha256(raw).hexdigest(),
        "preview_head": head, "preview_tail": tail,
        "omitted_bytes": len(raw) - len(head.encode()) - len(tail.encode()),
    })
    assert len(_json_bytes(descriptor)) <= 12 * 1024
    assert protocol.validate_context_output_descriptor(descriptor) == descriptor
    assert descriptor["content_bytes"] == len(raw)
    assert descriptor["content_sha256"] == hashlib.sha256(raw).hexdigest()
    assert descriptor["full_output_ref"] == protocol.encode_context_content_locator(REF)
    preview = descriptor["preview"]
    assert text.startswith(preview["head"]) and text.endswith(preview["tail"])
    observed_bytes = len(preview["head"].encode()) + len(preview["tail"].encode())
    assert observed_bytes <= len(raw)
    assert preview["omitted_bytes"] == len(raw) - observed_bytes
    assert descriptor["read_request"]["arguments"]["offset"] == 0


def test_model_projection_upgrades_a_valid_legacy_descriptor_within_final_budget() -> None:
    descriptor = protocol.present_context_output_ref(REF, {
        "content_bytes": 9000, "content_sha256": hashlib.sha256(b'"' * 9000).hexdigest(),
        "preview_head": '"', "preview_tail": '"', "omitted_bytes": 8998,
    })
    legacy = {key: value for key, value in descriptor.items() if key not in {"evidence_kind", "notice"}}
    legacy["schema_version"] = "unchain.tool_output.paged.v1"
    legacy["preview"] = {"head": '"' * 2850, "tail": '"' * 2850, "omitted_bytes": 3300}
    assert protocol.validate_context_output_descriptor(legacy) == legacy
    projected = protocol.project_context_output_for_model(legacy, source_ref=REF)
    assert projected["schema_version"] == "unchain.tool_output.paged.v2"
    assert len(_json_bytes(projected)) <= 12 * 1024
    assert protocol.validate_context_output_descriptor(projected) == projected
    assert projected["full_output_ref"] == legacy["full_output_ref"]
    assert projected["content_sha256"] == legacy["content_sha256"]
    assert legacy["schema_version"].endswith("v1")


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_descriptor_source_mismatch_cannot_replace_the_trusted_recovery_handle(version) -> None:
    descriptor = protocol.present_context_output_ref(REF, {
        "content_bytes": 3, "content_sha256": hashlib.sha256(b"abc").hexdigest(),
        "preview_head": "a", "preview_tail": "c", "omitted_bytes": 1,
    })
    if version == "v1":
        descriptor.pop("notice")
        descriptor.pop("evidence_kind")
        descriptor["schema_version"] = "unchain.tool_output.paged.v1"
    actual_source = ResourceRef("artifact", "actual-durable-source", 1)
    projected = protocol.project_context_output_for_model(descriptor, source_ref=actual_source)
    assert "schema_version" not in projected
    assert projected["full_output_ref"] == protocol.encode_context_content_locator(actual_source)
    assert projected["read_request"]["arguments"]["ref"] == projected["full_output_ref"]
    assert projected["preview"] == descriptor["preview"]
