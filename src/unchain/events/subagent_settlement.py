"""Bounded display evidence for an already committed Skeleton child result.

The projection is not a tool result or an execution capability. Its stable
cursor and typed attempts identify the durable settlement it describes.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

from ..journal.models import AttemptRef, EventCursor, JournalEvent, ResourceRef, _thaw_json


SETTLEMENT_PROJECTION_SCHEMA = "unchain.skeleton_subagent_settlement_projection.v1"
MAX_SETTLEMENT_PREVIEW_BYTES = 8192
MAX_SETTLEMENT_RESULT_BYTES = 32 * 1024 * 1024
_FIELDS = {"schema", "cursor", "consumer_attempt", "source_attempt", "call_id",
           "tool_name", "status", "result"}
_RESULT_FIELDS = {"full_output_ref", "content_bytes", "content_sha256", "preview", "inline"}
_TERMINAL = {"completed", "failed", "cancelled", "outcome_unknown"}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def validate_subagent_settlement_projection(raw, *, execution_id=None):
    """Read one exact projection; reject ambiguity rather than coerce identities."""
    if (type(raw) is not dict or set(raw) != _FIELDS
            or raw.get("schema") != SETTLEMENT_PROJECTION_SCHEMA):
        raise ValueError("subagent settlement projection has an invalid shape")
    consumer = AttemptRef.from_dict(raw["consumer_attempt"])
    source = AttemptRef.from_dict(raw["source_attempt"])
    cursor = EventCursor.from_dict(raw["cursor"])
    if (consumer.to_dict() != raw["consumer_attempt"]
            or source.to_dict() != raw["source_attempt"]
            or cursor.to_dict() != raw["cursor"] or cursor.store_seq < 1
            or consumer == source or consumer.generation != source.generation
            or execution_id is not None and consumer.generation.execution_id != execution_id):
        raise ValueError("subagent settlement projection changed its identity")
    for key in ("call_id", "tool_name"):
        value = raw[key]
        if type(value) is not str or not value or value.strip() != value:
            raise ValueError("subagent settlement projection has an invalid call")
    if type(raw["status"]) is not str or raw["status"] not in _TERMINAL:
        raise ValueError("subagent settlement projection is not terminal")
    result = raw["result"]
    if type(result) is not dict or set(result) != _RESULT_FIELDS:
        raise ValueError("subagent settlement result has an invalid shape")
    ref = ResourceRef.from_dict(result["full_output_ref"])
    if (ref.to_dict() != result["full_output_ref"]
            or ref.kind != "artifact" or ref.fragment):
        raise ValueError("subagent settlement result changed its artifact")
    size, digest, preview, inline = (result[key] for key in
        ("content_bytes", "content_sha256", "preview", "inline"))
    if (type(size) is not int or not 0 < size <= MAX_SETTLEMENT_RESULT_BYTES
            or type(digest) is not str or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or type(preview) is not str or len(preview) > MAX_SETTLEMENT_PREVIEW_BYTES
            or type(inline) is not bool or inline != (size <= MAX_SETTLEMENT_PREVIEW_BYTES)):
        raise ValueError("subagent settlement result has an invalid descriptor")
    content = preview.encode("utf-8")
    if len(content) > MAX_SETTLEMENT_PREVIEW_BYTES:
        raise ValueError("subagent settlement preview exceeds its budget")
    if inline:
        if len(content) != size or sha256(content).hexdigest() != digest:
            raise ValueError("subagent settlement inline bytes changed")
        try:
            body = json.loads(preview)
            if _canonical(body) != content:
                raise ValueError("noncanonical child result")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("subagent settlement result is not canonical JSON") from exc
        if (type(body) is not dict
                or set(body) != {"schema", "status", "outputs", "error", "summary"}
                or body.get("schema") != "unchain.skeleton_child_completion.v1"
                or body.get("status") != raw["status"]
                or type(body.get("error")) is not str or type(body.get("summary")) is not str
                or (type(body.get("outputs")) is not dict if raw["status"] == "completed"
                    else body.get("outputs") is not None)):
            raise ValueError("subagent settlement result contradicts its terminal state")
    return deepcopy(raw)


def build_subagent_settlement_projection(event, *, full_output):
    """Project the exact receipt after the caller verified its handoff artifact."""
    if type(event) is not JournalEvent or event.event_type != "skeleton.subagent_tool_settlement":
        raise ValueError("subagent settlement requires its canonical Journal receipt")
    from ..context.compiler import _subagent_tool_settlement
    payload = _thaw_json(event.payload)
    _subagent_tool_settlement(payload)
    consumer = AttemptRef.from_dict(payload["consumer_attempt"])
    if event.attempt != consumer:
        raise ValueError("subagent settlement escaped its consumer attempt")
    content = _canonical(full_output)
    if (len(content) != payload["result_bytes"]
            or sha256(content).hexdigest() != payload["result_sha256"]):
        raise ValueError("subagent settlement artifact bytes changed")
    preview = content[:MAX_SETTLEMENT_PREVIEW_BYTES].decode("utf-8", errors="ignore")
    projection = {
        "schema": SETTLEMENT_PROJECTION_SCHEMA,
        "cursor": EventCursor(event.store_seq, event.event_id).to_dict(),
        "consumer_attempt": payload["consumer_attempt"],
        "source_attempt": payload["source_attempt"],
        "call_id": payload["call_id"], "tool_name": payload["tool_name"],
        "status": payload["status"],
        "result": {"full_output_ref": payload["full_output_ref"],
            "content_bytes": len(content), "content_sha256": payload["result_sha256"],
            "preview": preview, "inline": len(content) <= MAX_SETTLEMENT_PREVIEW_BYTES},
    }
    return validate_subagent_settlement_projection(
        projection, execution_id=consumer.generation.execution_id,
    )
