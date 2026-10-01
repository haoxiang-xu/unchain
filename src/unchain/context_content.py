"""Versioned, host-codec-independent protocol for model-visible context bytes."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping
from typing import Any

from .journal import ResourceRef


MAX_CONTEXT_CONTENT_PAGE_READ_BYTES = 8 * 1024
MAX_CONTEXT_CONTENT_SERIALIZED_BYTES = 12 * 1024
_CONTEXT_PROTOCOL_PREFIX = "unchain://context/"
_LOCATOR_PREFIX = "unchain://context/v1/"
_LOCATOR_KEYS = frozenset({"fragment", "id", "kind", "revision"})
_CONTEXT_OUTPUT_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "projection",
        "preview",
        "content_bytes",
        "content_sha256",
        "full_output_ref",
        "read_request",
    }
)
_CONTEXT_CONTENT_ERROR = {
    "schema_version": "unchain.context_content_error.v1",
    "trust": "UNTRUSTED_DATA",
    "code": "CONTEXT_CONTENT_READ_FAILED",
}
_CONTEXT_CONTENT_ERROR_CODES = frozenset(
    {
        "CONTEXT_CONTENT_READ_FAILED", "CONTEXT_READ_NO_PROGRESS",
        "CONTEXT_READ_LIMIT_INVALID_USE_INTEGER_1_TO_8192",
        "CONTEXT_READ_OFFSET_INVALID_USE_INTEGER_0_TO_33554432",
    }
)


class ContextContentProtocolError(ValueError):
    """A malformed versioned context-content protocol value."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContextContentProtocolError("context content page has an invalid hash")
    return value


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContextContentProtocolError(
            f"context content page has an invalid {field_name}"
        )
    return value


def encode_context_content_locator(ref: ResourceRef) -> str:
    """Encode an opaque canonical locator; it carries no read authority."""
    if not isinstance(ref, ResourceRef):
        raise TypeError("context content locator requires a ResourceRef")
    payload = {
        "fragment": ref.fragment,
        "id": ref.resource_id,
        "kind": ref.kind,
        "revision": ref.revision,
    }
    encoded = base64.urlsafe_b64encode(_canonical_json(payload)).decode("ascii")
    return _LOCATOR_PREFIX + encoded.rstrip("=")


def decode_context_content_locator(
    value: Any,
    *,
    error_message: str,
) -> ResourceRef | None:
    """Decode only the canonical namespace, leaving legacy values to its host."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped.startswith(_CONTEXT_PROTOCOL_PREFIX) and value != stripped:
        raise ContextContentProtocolError(error_message)
    if value.startswith(_CONTEXT_PROTOCOL_PREFIX) and not value.startswith(
        _LOCATOR_PREFIX
    ):
        raise ContextContentProtocolError(error_message)
    if not value.startswith(_LOCATOR_PREFIX):
        return None
    if len(value) > 1024 or "\x00" in value:
        raise ContextContentProtocolError(error_message)
    token = value.removeprefix(_LOCATOR_PREFIX)
    if not token or re.fullmatch(r"[A-Za-z0-9_-]+", token) is None:
        raise ContextContentProtocolError(error_message)
    try:
        padding = "=" * (-len(token) % 4)
        decoded = base64.b64decode(token + padding, altchars=b"-_", validate=True)
        pairs: list[tuple[str, Any]] = json.loads(
            decoded.decode("utf-8"),
            object_pairs_hook=lambda items: items,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ContextContentProtocolError(error_message) from exc
    if not isinstance(pairs, list) or any(
        not isinstance(item, tuple) or len(item) != 2 for item in pairs
    ):
        raise ContextContentProtocolError(error_message)
    keys = [key for key, _value in pairs]
    if (
        any(not isinstance(key, str) for key in keys)
        or len(set(keys)) != len(keys)
        or frozenset(keys) != _LOCATOR_KEYS
    ):
        raise ContextContentProtocolError(error_message)
    payload = dict(pairs)
    if (
        not isinstance(payload["fragment"], str)
        or not isinstance(payload["id"], str)
        or not isinstance(payload["kind"], str)
        or isinstance(payload["revision"], bool)
        or not isinstance(payload["revision"], int)
        or payload["revision"] < 1
    ):
        raise ContextContentProtocolError(error_message)
    try:
        ref = ResourceRef(
            payload["kind"],
            payload["id"],
            payload["revision"],
            payload["fragment"],
        )
    except (TypeError, ValueError) as exc:
        raise ContextContentProtocolError(error_message) from exc
    if encode_context_content_locator(ref) != value:
        raise ContextContentProtocolError(error_message)
    return ref


def _page_fields(page: Any) -> tuple[ResourceRef, str, bytes, int, int, str]:
    ref = getattr(page, "ref", None)
    media_type = getattr(page, "media_type", None)
    data = getattr(page, "data", None)
    offset = getattr(page, "offset", None)
    total_bytes = getattr(page, "total_bytes", None)
    sha256 = getattr(page, "sha256", None)
    if not isinstance(ref, ResourceRef) or not isinstance(media_type, str):
        raise ContextContentProtocolError("context content capability returned an invalid page")
    if not isinstance(data, bytes):
        raise ContextContentProtocolError("context content capability returned an invalid page")
    offset = _require_nonnegative_int(offset, "offset")
    total_bytes = _require_nonnegative_int(total_bytes, "total_bytes")
    if offset > total_bytes or offset + len(data) > total_bytes:
        raise ContextContentProtocolError("context content capability returned an invalid page")
    return ref, media_type, data, offset, total_bytes, _require_sha256(sha256)


def validate_capability_content_page(
    page: Any,
    *,
    expected_ref: ResourceRef,
    expected_offset: int,
    requested_limit: int,
) -> Any:
    """Reject a capability page that cannot advance this exact context read."""
    ref, _media_type, data, offset, total_bytes, _sha256 = _page_fields(page)
    if ref != expected_ref:
        raise ContextContentProtocolError("context content page does not match requested ref")
    if offset != expected_offset or len(data) > requested_limit:
        raise ContextContentProtocolError(
            "context content page does not match requested range"
        )
    if len(data) > MAX_CONTEXT_CONTENT_PAGE_READ_BYTES:
        raise ContextContentProtocolError(
            "context content page exceeded source byte limit"
        )
    if not data and offset < total_bytes:
        raise ContextContentProtocolError(
            "context content page cannot continue without bytes"
        )
    return page


def present_context_output_ref(ref: ResourceRef, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Build the closed descriptor that a later producer can adopt unchanged."""
    expected = {
        "content_bytes",
        "content_sha256",
        "preview_head",
        "preview_tail",
        "omitted_bytes",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != expected:
        raise ContextContentProtocolError("context output descriptor metadata is invalid")
    content_bytes = _require_nonnegative_int(metadata["content_bytes"], "content_bytes")
    omitted_bytes = _require_nonnegative_int(metadata["omitted_bytes"], "omitted_bytes")
    if not isinstance(metadata["preview_head"], str) or not isinstance(
        metadata["preview_tail"], str
    ):
        raise ContextContentProtocolError("context output descriptor metadata is invalid")
    handle = context_content_read_handle(ref)
    return {
        "schema_version": "unchain.tool_output.paged.v1",
        "projection": "paged",
        "preview": {
            "head": metadata["preview_head"],
            "tail": metadata["preview_tail"],
            "omitted_bytes": omitted_bytes,
        },
        "content_bytes": content_bytes,
        "content_sha256": _require_sha256(metadata["content_sha256"]),
        **handle,
    }


def context_content_read_handle(ref: ResourceRef) -> dict[str, Any]:
    """Return the model-only locator and initial request for one durable source."""

    locator = encode_context_content_locator(ref)
    return {
        "full_output_ref": locator,
        "read_request": {
            "tool": "context_content_read",
            "arguments": {"ref": locator, "offset": 0, "limit": 8192},
        },
    }


def _validate_context_output_read_handle(
    value: Mapping[str, Any],
    *,
    error_message: str,
) -> ResourceRef:
    locator_ref = decode_context_content_locator(
        value.get("full_output_ref"),
        error_message=error_message,
    )
    if locator_ref is None:
        raise ContextContentProtocolError(error_message)
    read_request = value.get("read_request")
    if not isinstance(read_request, Mapping) or set(read_request) != {
        "tool",
        "arguments",
    }:
        raise ContextContentProtocolError(error_message)
    arguments = read_request["arguments"]
    if (
        read_request["tool"] != "context_content_read"
        or not isinstance(arguments, Mapping)
        or set(arguments) != {"ref", "offset", "limit"}
        or arguments["ref"] != value["full_output_ref"]
        or type(arguments["offset"]) is not int
        or arguments["offset"] != 0
        or type(arguments["limit"]) is not int
        or arguments["limit"] != MAX_CONTEXT_CONTENT_PAGE_READ_BYTES
    ):
        raise ContextContentProtocolError(error_message)
    return locator_ref


def canonical_context_output_read_handle(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact initial read handle only after strict validation."""

    ref = _validate_context_output_read_handle(
        value,
        error_message="context output presentation has an invalid read request",
    )
    return context_content_read_handle(ref)


def project_context_output_for_model(
    value: Mapping[str, Any],
    *,
    source_ref: ResourceRef,
) -> dict[str, Any]:
    """Copy a trusted output and replace only harness-owned retrieval fields."""

    if source_ref.kind != "artifact" or source_ref.fragment:
        raise ContextContentProtocolError("context output source is invalid")
    projected = dict(value)
    if projected.get("schema_version") == "unchain.context_content_page.v1":
        try:
            return validate_context_content_page(projected)
        except ContextContentProtocolError:
            projected.pop("schema_version", None)
    elif projected.get("schema_version") == "unchain.context_content_error.v1":
        try:
            return validate_context_content_error(projected)
        except ContextContentProtocolError:
            projected.pop("schema_version", None)
    projected.update(context_content_read_handle(source_ref))
    return projected


def validate_context_output_presentation(value: Any) -> dict[str, Any]:
    """Validate a bounded model view that carries the official read handle."""

    if not isinstance(value, Mapping):
        raise ContextContentProtocolError("context output presentation is invalid")
    canonical_context_output_read_handle(value)
    presentation = dict(value)
    if len(_canonical_json(presentation)) > MAX_CONTEXT_CONTENT_SERIALIZED_BYTES:
        raise ContextContentProtocolError(
            "context output presentation exceeds its serialized budget"
        )
    return presentation


def _content(data: bytes) -> dict[str, str]:
    try:
        return {"encoding": "utf-8", "text": data.decode("utf-8")}
    except UnicodeDecodeError:
        pass
    return {"encoding": "base64", "data_base64": base64.b64encode(data).decode("ascii")}


def _render_page(
    *,
    locator: str,
    media_type: str,
    total_bytes: int,
    sha256: str,
    offset: int,
    data: bytes,
) -> dict[str, Any]:
    next_offset = offset + len(data)
    eof = next_offset >= total_bytes
    return {
        "schema_version": "unchain.context_content_page.v1",
        "trust": "UNTRUSTED_DATA",
        "ref": locator,
        "media_type": media_type,
        "total_bytes": total_bytes,
        "sha256": sha256,
        "offset": offset,
        "page_bytes": len(data),
        "next_offset": None if eof else next_offset,
        "eof": eof,
        "content": _content(data),
        "next_read": (
            None
            if eof
            else {
                "tool": "context_content_read",
                "arguments": {"ref": locator, "offset": next_offset, "limit": 8192},
            }
        ),
    }


def validate_context_content_page(value: Any) -> dict[str, Any]:
    """Strictly validate the closed model wire representation of one page."""
    expected = {
        "schema_version",
        "trust",
        "ref",
        "media_type",
        "total_bytes",
        "sha256",
        "offset",
        "page_bytes",
        "next_offset",
        "eof",
        "content",
        "next_read",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ContextContentProtocolError("context content page has an invalid shape")
    if value["schema_version"] != "unchain.context_content_page.v1" or value[
        "trust"
    ] != "UNTRUSTED_DATA":
        raise ContextContentProtocolError("context content page has an invalid version")
    ref = value["ref"]
    if not isinstance(ref, str) or decode_context_content_locator(
        ref,
        error_message="context content page has an invalid locator",
    ) is None:
        raise ContextContentProtocolError("context content page has an invalid locator")
    if not isinstance(value["media_type"], str) or "/" not in value["media_type"]:
        raise ContextContentProtocolError("context content page has an invalid media type")
    total_bytes = _require_nonnegative_int(value["total_bytes"], "total_bytes")
    offset = _require_nonnegative_int(value["offset"], "offset")
    page_bytes = _require_nonnegative_int(value["page_bytes"], "page_bytes")
    _require_sha256(value["sha256"])
    content = value["content"]
    if not isinstance(content, Mapping) or not isinstance(content.get("encoding"), str):
        raise ContextContentProtocolError("context content page has invalid content")
    if content["encoding"] == "utf-8" and set(content) == {"encoding", "text"}:
        if not isinstance(content["text"], str):
            raise ContextContentProtocolError("context content page has invalid content")
        data = content["text"].encode("utf-8")
    elif content["encoding"] == "base64" and set(content) == {"encoding", "data_base64"}:
        if not isinstance(content["data_base64"], str):
            raise ContextContentProtocolError("context content page has invalid content")
        try:
            data = base64.b64decode(content["data_base64"], validate=True)
        except ValueError as exc:
            raise ContextContentProtocolError("context content page has invalid content") from exc
    else:
        raise ContextContentProtocolError("context content page has invalid content")
    if len(data) != page_bytes or page_bytes > MAX_CONTEXT_CONTENT_PAGE_READ_BYTES:
        raise ContextContentProtocolError("context content page has an invalid byte count")
    if offset + page_bytes > total_bytes:
        raise ContextContentProtocolError("context content page exceeds source bounds")
    eof = value["eof"]
    if not isinstance(eof, bool):
        raise ContextContentProtocolError("context content page has an invalid eof flag")
    if eof:
        if offset + page_bytes != total_bytes or value["next_offset"] is not None or value[
            "next_read"
        ] is not None:
            raise ContextContentProtocolError("context content page has an invalid eof state")
    else:
        if page_bytes == 0:
            raise ContextContentProtocolError(
                "context content page cannot continue without bytes"
            )
        next_offset = _require_nonnegative_int(value["next_offset"], "next_offset")
        expected_next_offset = offset + page_bytes
        if next_offset != expected_next_offset or next_offset >= total_bytes:
            raise ContextContentProtocolError("context content page has an invalid continuation")
        next_read = value["next_read"]
        if not isinstance(next_read, Mapping) or set(next_read) != {"tool", "arguments"}:
            raise ContextContentProtocolError("context content page has an invalid continuation")
        arguments = next_read["arguments"]
        if (
            next_read["tool"] != "context_content_read"
            or not isinstance(arguments, Mapping)
            or set(arguments) != {"ref", "offset", "limit"}
            or arguments["ref"] != ref
            or type(arguments["offset"]) is not int
            or arguments["offset"] != next_offset
            or type(arguments["limit"]) is not int
            or arguments["limit"] != 8192
        ):
            raise ContextContentProtocolError("context content page has an invalid continuation")
    if len(_canonical_json(dict(value))) > MAX_CONTEXT_CONTENT_SERIALIZED_BYTES:
        raise ContextContentProtocolError("context content page exceeds its serialized budget")
    return dict(value)


def validate_context_output_descriptor(value: Any) -> dict[str, Any]:
    """Validate the bounded source descriptor retained in fresh history."""

    if not isinstance(value, Mapping) or set(value) != _CONTEXT_OUTPUT_DESCRIPTOR_FIELDS:
        raise ContextContentProtocolError("context output descriptor has an invalid shape")
    if value["schema_version"] != "unchain.tool_output.paged.v1" or value[
        "projection"
    ] != "paged":
        raise ContextContentProtocolError("context output descriptor has an invalid version")
    preview = value["preview"]
    if not isinstance(preview, Mapping) or set(preview) != {
        "head",
        "tail",
        "omitted_bytes",
    }:
        raise ContextContentProtocolError("context output descriptor has an invalid preview")
    if not isinstance(preview["head"], str) or not isinstance(preview["tail"], str):
        raise ContextContentProtocolError("context output descriptor has an invalid preview")
    _require_nonnegative_int(preview["omitted_bytes"], "omitted bytes")
    _require_nonnegative_int(value["content_bytes"], "content bytes")
    _require_sha256(value["content_sha256"])
    _validate_context_output_read_handle(
        value,
        error_message="context output descriptor has an invalid read request",
    )
    descriptor = dict(value)
    if len(_canonical_json(descriptor)) > MAX_CONTEXT_CONTENT_SERIALIZED_BYTES:
        raise ContextContentProtocolError("context output descriptor exceeds its serialized budget")
    return descriptor


def validate_context_content_error(value: Any) -> dict[str, Any]:
    """Accept only the fixed, data-free model result for a failed reader call."""

    if (
        not isinstance(value, Mapping)
        or set(value) != set(_CONTEXT_CONTENT_ERROR)
        or value.get("schema_version") != _CONTEXT_CONTENT_ERROR["schema_version"]
        or value.get("trust") != _CONTEXT_CONTENT_ERROR["trust"]
        or value.get("code") not in _CONTEXT_CONTENT_ERROR_CODES
    ):
        raise ContextContentProtocolError("context content error has an invalid shape")
    return {**_CONTEXT_CONTENT_ERROR, "code": value["code"]}


def render_context_content_error(
    code: str = "CONTEXT_CONTENT_READ_FAILED",
) -> dict[str, Any]:
    """Return the fixed failure shape without echoing a source or exception text."""

    return validate_context_content_error({**_CONTEXT_CONTENT_ERROR, "code": code})


def canonical_context_content_page_bytes(value: Any) -> bytes:
    """Return the only byte representation accepted for a model-visible page."""

    return _canonical_json(validate_context_content_page(value))


def canonical_context_content_result_bytes(value: Any) -> bytes:
    """Return canonical bytes for either permitted reader outcome."""

    if isinstance(value, Mapping) and value.get("schema_version") == (
        "unchain.context_content_page.v1"
    ):
        return canonical_context_content_page_bytes(value)
    return _canonical_json(validate_context_content_error(value))


def render_context_content_page(page: Any) -> dict[str, Any]:
    """Render one bounded source-byte page for the official context reader."""
    ref, media_type, data, offset, total_bytes, sha256 = _page_fields(page)
    if len(data) > MAX_CONTEXT_CONTENT_PAGE_READ_BYTES:
        raise ContextContentProtocolError(
            "context content page exceeded source byte limit"
        )
    locator = encode_context_content_locator(ref)
    full = _render_page(
        locator=locator,
        media_type=media_type,
        total_bytes=total_bytes,
        sha256=sha256,
        offset=offset,
        data=data,
    )
    if len(_canonical_json(full)) <= MAX_CONTEXT_CONTENT_SERIALIZED_BYTES:
        return validate_context_content_page(full)
    empty = _render_page(
        locator=locator,
        media_type=media_type,
        total_bytes=total_bytes,
        sha256=sha256,
        offset=offset,
        data=b"",
    )
    if len(_canonical_json(empty)) > MAX_CONTEXT_CONTENT_SERIALIZED_BYTES:
        raise ContextContentProtocolError("context content page metadata exceeded its budget")
    lower, upper = 0, len(data)
    while lower < upper:
        candidate = (lower + upper + 1) // 2
        rendered = _render_page(
            locator=locator,
            media_type=media_type,
            total_bytes=total_bytes,
            sha256=sha256,
            offset=offset,
            data=data[:candidate],
        )
        if len(_canonical_json(rendered)) <= MAX_CONTEXT_CONTENT_SERIALIZED_BYTES:
            lower = candidate
        else:
            upper = candidate - 1
    return validate_context_content_page(
        _render_page(
            locator=locator,
            media_type=media_type,
            total_bytes=total_bytes,
            sha256=sha256,
            offset=offset,
            data=data[:lower],
        )
    )


__all__ = [
    "ContextContentProtocolError",
    "canonical_context_content_result_bytes",
    "canonical_context_content_page_bytes",
    "context_content_read_handle",
    "decode_context_content_locator",
    "encode_context_content_locator",
    "present_context_output_ref",
    "project_context_output_for_model",
    "render_context_content_error",
    "render_context_content_page",
    "validate_capability_content_page",
    "validate_context_content_error",
    "validate_context_content_page",
    "validate_context_output_descriptor",
]
