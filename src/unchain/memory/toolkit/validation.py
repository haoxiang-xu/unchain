from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from unchain.journal import ModelValidationError, ResourceRef
from unchain.memory.workspace import canonical_memory_link_url

from .models import (
    MAX_TASK_STATE_ITEM_CHARS,
    MAX_TASK_STATE_ITEMS,
    MAX_WRITE_BYTES,
    MemoryToolkitError,
    MemoryToolkitRunBinding,
    ReferencePurpose,
)


_PLACEHOLDER_NAMES = frozenset(
    {
        "file",
        "folder",
        "memory",
        "new",
        "new file",
        "note",
        "temp",
        "tmp",
        "untitled",
    }
)
_TASK_STATE_FIELDS = frozenset(
    {
        "objective",
        "success_criteria",
        "constraints",
        "confirmed_decisions",
        "open_questions",
        "active_plan",
        "artifact_memory_refs",
    }
)
_TASK_STATE_LIST_FIELDS = _TASK_STATE_FIELDS - {"objective"}


def canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MemoryToolkitError(
            "memory tool arguments must be canonical JSON"
        ) from exc


def bounded_integer(
    value: Any,
    field_name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MemoryToolkitError(f"{field_name} must be an integer")
    if value < minimum or value > maximum:
        raise MemoryToolkitError(
            f"{field_name} must be between {minimum} and {maximum}"
        )
    return value


def bounded_text(
    value: Any,
    field_name: str,
    *,
    maximum: int,
    required: bool = False,
) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise MemoryToolkitError(f"{field_name} must be text")
    normalized = unicodedata.normalize("NFC", value.strip())
    if required and not normalized:
        raise MemoryToolkitError(f"{field_name} is required")
    if len(normalized) > maximum or "\x00" in normalized:
        raise MemoryToolkitError(f"{field_name} is invalid")
    return normalized


def meaningful_path(value: Any) -> str:
    raw = bounded_text(value, "path", maximum=1024, required=True)
    if "\\" in raw:
        raise MemoryToolkitError("path must be a virtual POSIX path")
    segments = [segment for segment in raw.split("/") if segment]
    if not segments:
        raise MemoryToolkitError("path must name an entry")
    for segment in segments:
        if segment in {".", ".."} or any(ord(character) < 32 for character in segment):
            raise MemoryToolkitError("path contains an invalid segment")
    name = segments[-1]
    stem = name.rsplit(".", 1)[0].strip().casefold()
    if stem in _PLACEHOLDER_NAMES or len(stem) < 2:
        raise MemoryToolkitError(
            "path must use a specific, meaningful name instead of a placeholder"
        )
    return "/" + "/".join(segments)


def meaningful_description(value: Any) -> str:
    description = bounded_text(
        value,
        "description",
        maximum=8192,
        required=True,
    )
    if description.casefold() in _PLACEHOLDER_NAMES:
        raise MemoryToolkitError(
            "description must explain what the entry contains and when it is useful"
        )
    return description


def decode_external_ref(
    codec: Any,
    value: Any,
    *,
    purpose: ReferencePurpose,
    allowed_kinds: frozenset[str],
    error_message: str,
    field_name: str = "ref",
) -> ResourceRef:
    raw = bounded_text(value, field_name, maximum=1024, required=True)
    try:
        ref = codec.decode(raw, purpose=purpose)
    except Exception as exc:
        raise MemoryToolkitError(error_message) from exc
    if not isinstance(ref, ResourceRef) or ref.kind not in allowed_kinds:
        raise MemoryToolkitError(error_message)
    return ref


def encode_context_content_locator(ref: ResourceRef) -> str:
    """Compatibility facade for the shared context-content protocol."""
    from unchain.context_content import encode_context_content_locator as encode

    return encode(ref)


def decode_memory_ref(
    codec: Any,
    value: Any,
    *,
    error_message: str,
) -> ResourceRef:
    return decode_external_ref(
        codec,
        value,
        purpose=ReferencePurpose.MEMORY,
        allowed_kinds=frozenset({"memory"}),
        error_message=error_message,
    )


def decode_candidate_ref(
    codec: Any,
    value: Any,
    *,
    error_message: str,
) -> ResourceRef:
    ref = decode_external_ref(
        codec,
        value,
        purpose=ReferencePurpose.CANDIDATE,
        allowed_kinds=frozenset({"memory_candidate"}),
        error_message=error_message,
        field_name="candidate_ref",
    )
    if ref.fragment:
        raise MemoryToolkitError(error_message)
    return ref


def decode_context_content_ref(
    codec: Any,
    value: Any,
    *,
    error_message: str,
) -> ResourceRef:
    error = error_message
    from unchain.context_content import (
        ContextContentProtocolError,
        decode_context_content_locator,
    )

    try:
        ref = decode_context_content_locator(value, error_message=error)
    except ContextContentProtocolError as exc:
        raise MemoryToolkitError(str(exc)) from exc
    if ref is None:
        ref = decode_external_ref(
            codec,
            value,
            purpose=ReferencePurpose.CONTEXT_CONTENT,
            allowed_kinds=frozenset({"artifact", "checkpoint", "context_event"}),
            error_message=error,
        )
    elif ref.kind not in {"artifact", "checkpoint", "context_event"}:
        raise MemoryToolkitError(error)
    valid = (
        (ref.kind == "artifact" and not ref.fragment)
        or (
            ref.kind == "checkpoint"
            and (
                not ref.fragment
                or re.fullmatch(r"event/[1-9][0-9]*", ref.fragment) is not None
            )
        )
        or (ref.kind == "context_event" and ref.fragment == "content")
    )
    if not valid:
        raise MemoryToolkitError(error)
    return ref


def decode_checkpoint_ref(
    codec: Any,
    value: Any,
    *,
    error_message: str,
) -> ResourceRef:
    error = error_message
    ref = decode_external_ref(
        codec,
        value,
        purpose=ReferencePurpose.CHECKPOINT,
        allowed_kinds=frozenset({"checkpoint"}),
        error_message=error,
        field_name="checkpoint_ref",
    )
    if ref.fragment:
        raise MemoryToolkitError(error)
    return ref


def decode_source_ref(
    codec: Any,
    value: Any,
    *,
    field_name: str = "ref",
    error_message: str,
) -> ResourceRef:
    error = error_message
    ref = decode_external_ref(
        codec,
        value,
        purpose=ReferencePurpose.SOURCE,
        allowed_kinds=frozenset({"memory", "artifact", "context_event", "checkpoint"}),
        error_message=error,
        field_name=field_name,
    )
    if ref.kind == "artifact" and ref.fragment:
        raise MemoryToolkitError(error)
    if ref.kind == "context_event" and ref.fragment not in {"", "content"}:
        raise MemoryToolkitError(error)
    if (
        ref.kind == "checkpoint"
        and ref.fragment
        and re.fullmatch(r"event/[1-9][0-9]*", ref.fragment) is None
    ):
        raise MemoryToolkitError(error)
    return ref


def decode_source_event_refs(
    codec: Any,
    values: Sequence[str] | None,
    *,
    list_error: str,
    item_error: str,
) -> tuple[ResourceRef, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise MemoryToolkitError(list_error)
    if len(values) > 100:
        raise MemoryToolkitError("source_refs has too many items")
    refs: list[ResourceRef] = []
    for value in values:
        try:
            ref = decode_external_ref(
                codec,
                value,
                purpose=ReferencePurpose.TASK_EVENT,
                allowed_kinds=frozenset({"context_event"}),
                error_message=item_error,
            )
        except MemoryToolkitError as exc:
            raise MemoryToolkitError(item_error) from exc
        if ref.fragment:
            raise MemoryToolkitError(item_error)
        refs.append(ref)
    return tuple(refs)


def decode_write_content(
    *,
    kind: str,
    content: str,
    content_base64: str,
    mime_type: str,
    url: str,
) -> tuple[str, bytes | None, str, str]:
    public_kind = bounded_text(kind, "kind", maximum=32, required=True).lower()
    if public_kind not in {"folder", "markdown", "image", "link"}:
        raise MemoryToolkitError("kind must be folder, markdown, image, or link")
    normalized_content = content if isinstance(content, str) else None
    normalized_base64 = content_base64 if isinstance(content_base64, str) else None
    if normalized_content is None or normalized_base64 is None:
        raise MemoryToolkitError("content and content_base64 must be text")
    normalized_url = bounded_text(url, "url", maximum=8192)
    normalized_mime = bounded_text(mime_type, "mime_type", maximum=255)
    if public_kind == "folder":
        if normalized_content or normalized_base64 or normalized_url:
            raise MemoryToolkitError("folder entries cannot contain content or a URL")
        return "folder", None, "", ""
    if public_kind == "link":
        if normalized_content or normalized_base64:
            raise MemoryToolkitError("link entries cannot contain inline content")
        try:
            normalized_url = canonical_memory_link_url(normalized_url)
        except (TypeError, ModelValidationError) as exc:
            if "credential" in str(exc).casefold():
                raise MemoryToolkitError(
                    "link URLs cannot contain credentials"
                ) from exc
            raise MemoryToolkitError("link URL must use http or https") from exc
        return "link", None, "", normalized_url
    if normalized_url:
        raise MemoryToolkitError("file entries cannot contain a URL")
    if public_kind == "markdown":
        if normalized_base64:
            raise MemoryToolkitError("markdown content must not use content_base64")
        raw = normalized_content.encode("utf-8")
        resolved_mime = normalized_mime or "text/markdown"
    else:
        if normalized_content:
            raise MemoryToolkitError("image content must use content_base64")
        if not normalized_base64:
            raise MemoryToolkitError("image content_base64 is required")
        try:
            raw = base64.b64decode(normalized_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise MemoryToolkitError("image content_base64 is invalid") from exc
        resolved_mime = normalized_mime or "image/png"
        if not resolved_mime.lower().startswith("image/"):
            raise MemoryToolkitError("image mime_type must start with image/")
    if len(raw) > MAX_WRITE_BYTES:
        raise MemoryToolkitError(
            f"content exceeds the memory tool limit of {MAX_WRITE_BYTES} bytes"
        )
    return public_kind, raw, resolved_mime, ""


def normalize_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MemoryToolkitError("confidence must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0 or normalized > 1:
        raise MemoryToolkitError("confidence must be between 0 and 1")
    return normalized


def task_state_patch(
    codec: Any,
    value: Any,
    *,
    reference_error: str,
) -> tuple[dict[str, Any], tuple[ResourceRef, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise MemoryToolkitError("patch must be a non-empty task-state object")
    unknown = set(value) - _TASK_STATE_FIELDS
    if unknown:
        raise MemoryToolkitError("patch contains unsupported task-state fields")
    normalized: dict[str, Any] = {}
    referenced: list[ResourceRef] = []
    if "objective" in value:
        normalized["objective"] = bounded_text(
            value.get("objective"),
            "patch.objective",
            maximum=16_384,
        )
    for field_name in sorted(_TASK_STATE_LIST_FIELDS):
        if field_name not in value:
            continue
        items = value.get(field_name)
        if not isinstance(items, list) or len(items) > MAX_TASK_STATE_ITEMS:
            raise MemoryToolkitError(
                f"patch.{field_name} must be an array with at most "
                f"{MAX_TASK_STATE_ITEMS} items"
            )
        if field_name == "artifact_memory_refs":
            artifact_refs: list[ResourceRef] = []
            memory_refs: list[ResourceRef] = []
            for item in items:
                error = reference_error
                ref = decode_external_ref(
                    codec,
                    item,
                    purpose=ReferencePurpose.ARTIFACT_OR_MEMORY,
                    allowed_kinds=frozenset({"artifact", "memory"}),
                    error_message=error,
                )
                if ref.kind == "artifact":
                    artifact_refs.append(ref)
                else:
                    memory_refs.append(ref)
                referenced.append(ref)
            normalized["artifact_refs"] = tuple(artifact_refs)
            normalized["memory_refs"] = tuple(memory_refs)
            continue
        normalized[field_name] = tuple(
            bounded_text(
                item,
                f"patch.{field_name}",
                maximum=MAX_TASK_STATE_ITEM_CHARS,
                required=True,
            )
            for item in items
        )
    canonical_patch = {
        key: [
            item.to_dict() if isinstance(item, ResourceRef) else item for item in value
        ]
        if isinstance(value, tuple)
        else value
        for key, value in normalized.items()
    }
    if len(canonical_json(canonical_patch)) > MAX_WRITE_BYTES:
        raise MemoryToolkitError("task-state patch exceeds the memory tool limit")
    return normalized, tuple(referenced)


def mutation_id(
    binding: MemoryToolkitRunBinding,
    *,
    tool_name: str,
    payload: Mapping[str, Any],
    qualifier: str = "",
) -> str:
    def model_value(value: Any) -> Any:
        if isinstance(value, ResourceRef):
            return value.to_dict()
        if isinstance(value, Mapping):
            return {key: model_value(child) for key, child in value.items()}
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [model_value(child) for child in value]
        return value

    digest = hashlib.sha256(
        canonical_json(
            {
                "tool": tool_name,
                "binding_id": binding.binding_id,
                "session_id": binding.session_id,
                "attempt_id": binding.attempt_id,
                "run_id": binding.run_id,
                "qualifier": qualifier,
                "payload": model_value(payload),
            }
        )
    ).hexdigest()
    return f"memory-v2:{digest}"


__all__ = [
    "bounded_integer",
    "bounded_text",
    "canonical_json",
    "decode_candidate_ref",
    "decode_checkpoint_ref",
    "decode_context_content_ref",
    "decode_memory_ref",
    "decode_source_event_refs",
    "decode_source_ref",
    "decode_write_content",
    "meaningful_description",
    "meaningful_path",
    "mutation_id",
    "normalize_confidence",
    "task_state_patch",
]
