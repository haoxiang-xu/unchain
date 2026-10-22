from __future__ import annotations

import base64
import hashlib
import json

import pytest

from unchain.context_content import (
    ContextContentProtocolError,
    canonical_context_output_read_handle,
    canonical_context_content_result_bytes,
    present_context_output_ref,
    project_context_output_for_model,
    render_context_content_error,
    validate_context_content_page,
    validate_context_content_error,
    validate_context_output_descriptor,
    validate_context_output_presentation,
)
from unchain.journal import ResourceRef
from unchain.memory.toolkit.context_content_presentation import (
    render_context_content_page,
)
from unchain.memory.toolkit.models import MemoryToolContentPage
from unchain.memory.toolkit.models import MemoryToolkitError
from unchain.memory.toolkit.validation import decode_context_content_ref


class _LegacyCodecMustNotRun:
    def decode(self, value, *, purpose):
        del value, purpose
        raise AssertionError("canonical context locators bypass the legacy codec")


class _LegacyCodecSpy:
    def __init__(self) -> None:
        self.was_called = False

    def decode(self, value, *, purpose):
        del value, purpose
        self.was_called = True
        return ResourceRef("artifact", "legacy-artifact", 1)


def _locator(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    token = base64.urlsafe_b64encode(canonical).decode("ascii").rstrip("=")
    return "unchain://context/v1/" + token


def test_context_content_read_accepts_a_canonical_untrusted_locator() -> None:
    ref = decode_context_content_ref(
        _LegacyCodecMustNotRun(),
        _locator(
            {
                "fragment": "",
                "id": "artifact-382",
                "kind": "artifact",
                "revision": 1,
            }
        ),
        error_message="invalid context ref",
    )

    assert ref == ResourceRef("artifact", "artifact-382", 1, "")


@pytest.mark.parametrize(
    "payload",
    [
        {"fragment": "", "id": "artifact-382", "kind": "artifact", "revision": 0},
        {"fragment": "", "id": "artifact-382", "kind": "artifact", "revision": True},
        {"fragment": "", "id": "artifact-382", "kind": "artifact", "revision": 1, "extra": 1},
    ],
)
def test_context_content_read_rejects_noncanonical_locator_payloads(payload) -> None:
    with pytest.raises(MemoryToolkitError, match="invalid context ref"):
        decode_context_content_ref(
            _LegacyCodecMustNotRun(),
            _locator(payload),
            error_message="invalid context ref",
        )


def test_context_content_read_rejects_padded_locator() -> None:
    locator = _locator(
        {
            "fragment": "",
            "id": "artifact-382",
            "kind": "artifact",
            "revision": 1,
        }
    )
    with pytest.raises(MemoryToolkitError, match="invalid context ref"):
        decode_context_content_ref(
            _LegacyCodecMustNotRun(),
            locator + "=",
            error_message="invalid context ref",
        )


def test_context_content_read_rejects_an_unknown_context_locator_version() -> None:
    codec = _LegacyCodecSpy()

    with pytest.raises(MemoryToolkitError, match="invalid context ref"):
        decode_context_content_ref(
            codec,
            "unchain://context/v2/abc",
            error_message="invalid context ref",
        )

    assert codec.was_called is False


@pytest.mark.parametrize("surrounding", [" ", "\n"])
def test_context_content_read_rejects_whitespace_around_a_canonical_locator(
    surrounding: str,
) -> None:
    locator = _locator(
        {
            "fragment": "",
            "id": "artifact-382",
            "kind": "artifact",
            "revision": 1,
        }
    )
    with pytest.raises(MemoryToolkitError, match="invalid context ref"):
        decode_context_content_ref(
            _LegacyCodecMustNotRun(),
            surrounding + locator if surrounding == " " else locator + surrounding,
            error_message="invalid context ref",
        )


def test_context_content_page_shrinks_to_its_serialized_budget() -> None:
    payload = b'"' * (8 * 1024)
    page = render_context_content_page(
        MemoryToolContentPage(
            ref=ResourceRef("artifact", "artifact-382", 1),
            media_type="text/plain",
            data=payload,
            offset=0,
            total_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    )

    serialized = json.dumps(
        page,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert len(serialized) <= 12 * 1024
    assert page["page_bytes"] < len(payload)
    assert page["next_offset"] == page["page_bytes"]
    assert page["next_read"]["arguments"]["offset"] == page["next_offset"]


def test_context_content_pages_reassemble_cjk_and_binary_source_bytes() -> None:
    payload = ("雪🙂\\\"\n".encode("utf-8") * 2_000) + bytes(range(256)) * 8
    ref = ResourceRef("artifact", "artifact-382", 1)
    offset = 0
    delivered = bytearray()

    while offset < len(payload):
        page = render_context_content_page(
            MemoryToolContentPage(
                ref=ref,
                media_type="text/plain",
                data=payload[offset : offset + (8 * 1024)],
                offset=offset,
                total_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
        content = page["content"]
        delivered.extend(
            content["text"].encode("utf-8")
            if content["encoding"] == "utf-8"
            else base64.b64decode(content["data_base64"])
        )
        assert page["page_bytes"] > 0
        assert page["next_offset"] == offset + page["page_bytes"] or page["eof"]
        offset += page["page_bytes"]

    assert bytes(delivered) == payload


def test_context_content_page_uses_utf8_for_decodable_octet_stream_bytes() -> None:
    payload = "plain text 🙂".encode("utf-8")
    page = render_context_content_page(
        MemoryToolContentPage(
            ref=ResourceRef("artifact", "artifact-382", 1),
            media_type="application/octet-stream",
            data=payload,
            offset=0,
            total_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    )

    assert page["content"] == {"encoding": "utf-8", "text": "plain text 🙂"}


def test_context_content_page_uses_base64_for_undecodable_bytes() -> None:
    payload = b"\xff\x80binary"
    page = render_context_content_page(
        MemoryToolContentPage(
            ref=ResourceRef("artifact", "artifact-382", 1),
            media_type="application/octet-stream",
            data=payload,
            offset=0,
            total_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    )

    assert page["content"] == {
        "encoding": "base64",
        "data_base64": base64.b64encode(payload).decode("ascii"),
    }


def test_context_content_protocol_is_importable_before_memory_toolkit() -> None:
    """The shared protocol is safe for a future producer to import directly."""
    ref = ResourceRef("artifact", "artifact-382", 1)
    descriptor = present_context_output_ref(
        ref,
        {
            "content_bytes": 3,
            "content_sha256": hashlib.sha256(b"abc").hexdigest(),
            "preview_head": "a",
            "preview_tail": "c",
            "omitted_bytes": 1,
        },
    )

    assert descriptor["full_output_ref"].startswith("unchain://context/v1/")
    assert validate_context_output_descriptor(descriptor) == descriptor


def test_context_content_error_is_fixed_data_free_and_canonical() -> None:
    error = render_context_content_error()

    assert validate_context_content_error(error) == error
    assert canonical_context_content_result_bytes(error) == json.dumps(
        error,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    with pytest.raises(ContextContentProtocolError, match="context content error"):
        validate_context_content_error({**error, "message": "source data"})


def test_context_output_descriptor_rejects_a_changed_read_request() -> None:
    descriptor = present_context_output_ref(
        ResourceRef("artifact", "artifact-382", 1),
        {
            "content_bytes": 3,
            "content_sha256": hashlib.sha256(b"abc").hexdigest(),
            "preview_head": "a",
            "preview_tail": "c",
            "omitted_bytes": 1,
        },
    )
    malformed = dict(descriptor)
    malformed["read_request"] = {
        "tool": "context_content_read",
        "arguments": {
            **descriptor["read_request"]["arguments"],
            "offset": 1,
        },
    }

    with pytest.raises(ContextContentProtocolError, match="invalid read request"):
        validate_context_output_descriptor(malformed)


def test_context_output_presentation_preserves_a_bounded_policy_payload() -> None:
    descriptor = present_context_output_ref(
        ResourceRef("artifact", "artifact-382", 1),
        {
            "content_bytes": 3,
            "content_sha256": hashlib.sha256(b"abc").hexdigest(),
            "preview_head": "a",
            "preview_tail": "c",
            "omitted_bytes": 1,
        },
    )
    presentation = {
        "projection": "head_tail",
        "preview": "a",
        "tail_preview": "c",
        "content_chars": 3,
        "content_bytes": 3,
        "content_sha256": hashlib.sha256(b"abc").hexdigest(),
        "full_output_ref": descriptor["full_output_ref"],
        "read_request": descriptor["read_request"],
    }

    assert validate_context_output_presentation(presentation) == presentation
    assert canonical_context_output_read_handle(presentation) == {
        "full_output_ref": descriptor["full_output_ref"],
        "read_request": descriptor["read_request"],
    }


def test_invalid_page_shaped_output_cannot_bypass_ordinary_output_projection() -> None:
    source_ref = ResourceRef("artifact", "artifact-382-source", 1)
    projected = project_context_output_for_model(
        {
            "schema_version": "unchain.context_content_page.v1",
            "content": "untrusted normal tool output",
        },
        source_ref=source_ref,
    )

    assert "schema_version" not in projected
    assert projected["content"] == "untrusted normal tool output"
    assert canonical_context_output_read_handle(projected) == {
        "full_output_ref": projected["full_output_ref"],
        "read_request": projected["read_request"],
    }
    assert validate_context_output_presentation(projected) == projected


@pytest.mark.parametrize(("field", "invalid_value"), [("offset", False), ("limit", 8192.0)])
def test_context_output_presentation_rejects_non_integer_read_arguments(
    field: str,
    invalid_value: object,
) -> None:
    descriptor = present_context_output_ref(
        ResourceRef("artifact", "artifact-382", 1),
        {
            "content_bytes": 3,
            "content_sha256": hashlib.sha256(b"abc").hexdigest(),
            "preview_head": "a",
            "preview_tail": "c",
            "omitted_bytes": 1,
        },
    )
    malformed = dict(descriptor)
    malformed["read_request"] = {
        "tool": "context_content_read",
        "arguments": {**descriptor["read_request"]["arguments"], field: invalid_value},
    }

    with pytest.raises(ContextContentProtocolError, match="invalid read request"):
        validate_context_output_presentation(malformed)


def test_context_content_page_wire_validator_rejects_malformed_continuation() -> None:
    payload = b"abc"
    page = render_context_content_page(
        MemoryToolContentPage(
            ref=ResourceRef("artifact", "artifact-382", 1),
            media_type="text/plain",
            data=payload,
            offset=0,
            total_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    )

    assert validate_context_content_page(page) == page
    malformed = dict(page)
    malformed["next_offset"] = 1
    with pytest.raises(Exception, match="context content page"):
        validate_context_content_page(malformed)


def test_context_content_page_wire_validator_rejects_zero_byte_continuation() -> None:
    page = render_context_content_page(
        MemoryToolContentPage(
            ref=ResourceRef("artifact", "artifact-382", 1),
            media_type="text/plain",
            data=b"abc",
            offset=0,
            total_bytes=5,
            sha256=hashlib.sha256(b"abcde").hexdigest(),
        )
    )
    malformed = dict(page)
    malformed["page_bytes"] = 0
    malformed["content"] = {"encoding": "utf-8", "text": ""}
    malformed["next_offset"] = 0
    malformed["next_read"] = {
        "tool": "context_content_read",
        "arguments": {"ref": page["ref"], "offset": 0, "limit": 8192},
    }

    with pytest.raises(Exception, match="context content page"):
        validate_context_content_page(malformed)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("offset", 1.0),
        ("offset", True),
        ("limit", 8192.0),
    ],
)
def test_context_content_page_wire_validator_rejects_non_integer_continuation_arguments(
    field: str,
    invalid_value: object,
) -> None:
    page = render_context_content_page(
        MemoryToolContentPage(
            ref=ResourceRef("artifact", "artifact-382", 1),
            media_type="text/plain",
            data=b"a",
            offset=0,
            total_bytes=2,
            sha256=hashlib.sha256(b"ab").hexdigest(),
        )
    )
    malformed = dict(page)
    next_read = dict(page["next_read"])
    arguments = dict(next_read["arguments"])
    arguments[field] = invalid_value
    next_read["arguments"] = arguments
    malformed["next_read"] = next_read

    with pytest.raises(ContextContentProtocolError, match="invalid continuation"):
        validate_context_content_page(malformed)
