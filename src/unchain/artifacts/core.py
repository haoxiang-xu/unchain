from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "unchain.artifact.v1"
SUPPORTED_KINDS = frozenset({"file_diff", "plan", "markdown", "table", "kv", "log", "link"})
DEFAULT_MAX_LINES = 400
DEFAULT_MAX_BYTES = 128 * 1024
PLAN_TOOL_NAMES = frozenset({"plan_start", "plan_update", "plan_finalize"})


@dataclass(frozen=True)
class ArtifactCaps:
    max_lines: int = DEFAULT_MAX_LINES
    max_bytes: int = DEFAULT_MAX_BYTES


@dataclass(frozen=True)
class ArtifactOwner:
    run_id: str
    turn_id: str | None
    tool_name: str
    call_id: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "tool_name": self.tool_name,
            "call_id": self.call_id,
        }


def _safe_deepcopy(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        try:
            return json.loads(json.dumps(value, default=str, ensure_ascii=False))
        except Exception:
            return str(value)


def markdown(
    title: str,
    markdown: str,
    source_path: str | None = None,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    return _with_optional(
        {
            "kind": "markdown",
            "title": title,
            "markdown": markdown,
        },
        source_path=source_path,
        artifact_id=artifact_id,
    )


def table(
    title: str,
    columns: list[Any],
    rows: list[Any],
    source_path: str | None = None,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    return _with_optional(
        {
            "kind": "table",
            "title": title,
            "columns": copy.deepcopy(columns),
            "rows": copy.deepcopy(rows),
        },
        source_path=source_path,
        artifact_id=artifact_id,
    )


def kv(
    title: str,
    pairs: dict[str, Any] | list[Any],
    source_path: str | None = None,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    return _with_optional(
        {
            "kind": "kv",
            "title": title,
            "pairs": copy.deepcopy(pairs),
        },
        source_path=source_path,
        artifact_id=artifact_id,
    )


def log(
    title: str,
    text: str,
    source_path: str | None = None,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    return _with_optional(
        {
            "kind": "log",
            "title": title,
            "text": text,
        },
        source_path=source_path,
        artifact_id=artifact_id,
    )


def link(
    title: str,
    url: str | None = None,
    path: str | None = None,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "link",
        "title": title,
    }
    if isinstance(url, str) and url:
        payload["url"] = url
    if isinstance(path, str) and path:
        payload["path"] = path
    if isinstance(artifact_id, str) and artifact_id:
        payload["artifact_id"] = artifact_id
    return payload


def file_diff(
    title: str,
    files: list[dict[str, Any]],
    artifact_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "file_diff",
        "title": title,
        "files": copy.deepcopy(files),
    }
    if isinstance(artifact_id, str) and artifact_id:
        payload["artifact_id"] = artifact_id
    return payload


def _with_optional(
    payload: dict[str, Any],
    *,
    source_path: str | None = None,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    if isinstance(source_path, str) and source_path:
        payload["source_path"] = source_path
    if isinstance(artifact_id, str) and artifact_id:
        payload["artifact_id"] = artifact_id
    return payload


def canonicalize_artifacts(
    raw_artifacts: Any,
    *,
    owner: ArtifactOwner,
    existing_artifacts: list[dict[str, Any]] | None = None,
    caps: ArtifactCaps | None = None,
) -> list[dict[str, Any]]:
    descriptors = _coerce_artifact_list(raw_artifacts)
    if not descriptors:
        return []

    caps = caps or ArtifactCaps()
    revision_by_id = _revision_map(existing_artifacts)
    emitted: list[dict[str, Any]] = []
    for index, descriptor in enumerate(descriptors):
        artifact = _canonicalize_one(
            descriptor,
            owner=owner,
            index=index,
            revision_by_id=revision_by_id,
            caps=caps,
        )
        if artifact is None:
            continue
        revision_by_id[artifact["artifact_id"]] = int(artifact["revision"])
        emitted.append(artifact)
    return emitted


def extract_authored_artifacts(tool_result: Any) -> tuple[Any, list[dict[str, Any]]]:
    if not isinstance(tool_result, dict):
        return _safe_deepcopy(tool_result), []
    raw_artifacts = tool_result.get("_artifacts")
    visible = {
        key: _safe_deepcopy(value)
        for key, value in tool_result.items()
        if key != "_artifacts"
    }
    return visible, _coerce_artifact_list(raw_artifacts)


def is_failed_tool_result(tool_result: Any) -> bool:
    if not isinstance(tool_result, dict):
        return True
    if tool_result.get("denied") is True:
        return True
    if tool_result.get("ok") is False:
        return True
    if tool_result.get("error") is not None:
        return True
    return False


def upsert_artifacts(
    existing_artifacts: list[dict[str, Any]] | None,
    new_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    order: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for artifact in existing_artifacts or []:
        if not isinstance(artifact, dict):
            continue
        artifact_id = artifact.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            continue
        if artifact_id not in by_id:
            order.append(artifact_id)
        by_id[artifact_id] = copy.deepcopy(artifact)
    for artifact in new_artifacts:
        artifact_id = artifact.get("artifact_id") if isinstance(artifact, dict) else None
        if not isinstance(artifact_id, str) or not artifact_id:
            continue
        if artifact_id not in by_id:
            order.append(artifact_id)
        by_id[artifact_id] = copy.deepcopy(artifact)
    return [by_id[artifact_id] for artifact_id in order]


def artifacts_from_code_diff_policy(
    interact_config: Any,
    *,
    call_id: str,
    tool_name: str,
) -> list[dict[str, Any]]:
    files = _code_diff_files(interact_config)
    if not files:
        return []
    title = _code_diff_title(interact_config, tool_name=tool_name)
    return [
        file_diff(
            title,
            files,
            artifact_id=f"file_diff:{call_id}",
        )
    ]


def plan_artifact_from_tool_result(
    *,
    tool_name: str,
    tool_result: dict[str, Any],
) -> dict[str, Any] | None:
    if tool_name not in PLAN_TOOL_NAMES or is_failed_tool_result(tool_result):
        return None
    plan_id = _clean_string(tool_result.get("plan_id"))
    if not plan_id:
        return None
    workspace_file = tool_result.get("workspace_file")
    if not isinstance(workspace_file, dict):
        return None
    source_path = _clean_string(workspace_file.get("path"))
    if not source_path:
        return None
    try:
        markdown_text = Path(source_path).read_text(encoding="utf-8")
    except Exception:
        return None

    revision = _coerce_positive_int(tool_result.get("revision"), fallback=1)
    status = _clean_string(tool_result.get("status")) or "ready"
    title = _first_markdown_heading(markdown_text) or f"Plan {plan_id}"
    descriptor: dict[str, Any] = {
        "kind": "plan",
        "title": title,
        "artifact_id": f"plan:{plan_id}",
        "plan_id": plan_id,
        "status": status,
        "revision": revision,
        "markdown": markdown_text,
        "source_path": source_path,
    }
    relative_path = _clean_string(workspace_file.get("relative_path"))
    if relative_path:
        descriptor["source_relative_path"] = relative_path
    return descriptor


def _canonicalize_one(
    descriptor: dict[str, Any],
    *,
    owner: ArtifactOwner,
    index: int,
    revision_by_id: dict[str, int],
    caps: ArtifactCaps,
) -> dict[str, Any] | None:
    kind = _clean_string(descriptor.get("kind"))
    title = _clean_string(descriptor.get("title"))
    if kind not in SUPPORTED_KINDS or not title:
        return None

    artifact_id = _clean_string(descriptor.get("artifact_id")) or _default_artifact_id(kind, owner, index)
    snapshot = _snapshot_for(kind, descriptor, caps=caps)
    if snapshot is None:
        return None

    explicit_revision = descriptor.get("revision")
    revision = (
        _coerce_positive_int(explicit_revision, fallback=0)
        if explicit_revision is not None
        else 0
    )
    if revision <= 0:
        revision = revision_by_id.get(artifact_id, 0) + 1

    summary = _clean_string(descriptor.get("summary")) or _summary_for(kind, title, snapshot)
    status = _clean_string(descriptor.get("status")) or "ready"
    presentation = _presentation_for(descriptor, owner=owner)
    source = _source_for(descriptor, snapshot)

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "kind": kind,
        "title": title,
        "summary": summary,
        "revision": revision,
        "status": status,
        "owner": owner.to_dict(),
        "snapshot": snapshot,
        "source": source,
        "presentation": presentation,
    }
    return artifact


def _snapshot_for(
    kind: str,
    descriptor: dict[str, Any],
    *,
    caps: ArtifactCaps,
) -> dict[str, Any] | None:
    if kind == "markdown":
        text = descriptor.get("markdown")
        if not isinstance(text, str):
            return None
        snapshot = _text_snapshot(text, caps=caps)
        snapshot["markdown"] = snapshot.pop("text")
        return snapshot

    if kind == "plan":
        text = descriptor.get("markdown")
        if not isinstance(text, str):
            return None
        snapshot = _text_snapshot(text, caps=caps)
        snapshot["markdown"] = snapshot.pop("text")
        plan_id = _clean_string(descriptor.get("plan_id"))
        if plan_id:
            snapshot["plan_id"] = plan_id
        snapshot["status"] = _clean_string(descriptor.get("status")) or "ready"
        snapshot["revision"] = _coerce_positive_int(descriptor.get("revision"), fallback=1)
        snapshot["title"] = _clean_string(descriptor.get("title"))
        return snapshot

    if kind == "log":
        text = descriptor.get("text")
        if not isinstance(text, str):
            return None
        return _text_snapshot(text, caps=caps)

    if kind == "table":
        columns = descriptor.get("columns")
        rows = descriptor.get("rows")
        if not isinstance(columns, list) or not isinstance(rows, list):
            return None
        displayed_rows, meta = _truncate_rows(rows, caps=caps, header_lines=1)
        content = {"columns": columns, "rows": rows}
        return {
            "columns": copy.deepcopy(columns),
            "rows": copy.deepcopy(displayed_rows),
            "sha256": _sha256(_canonical_json(content)),
            **meta,
        }

    if kind == "kv":
        pairs = descriptor.get("pairs")
        if isinstance(pairs, dict):
            display_pairs, meta = _truncate_mapping(pairs, caps=caps)
            content = pairs
        elif isinstance(pairs, list):
            display_pairs, meta = _truncate_rows(pairs, caps=caps, header_lines=0)
            content = pairs
        else:
            return None
        return {
            "pairs": copy.deepcopy(display_pairs),
            "sha256": _sha256(_canonical_json(content)),
            **meta,
        }

    if kind == "link":
        url = _clean_string(descriptor.get("url"))
        path = _clean_string(descriptor.get("path"))
        if not url and not path:
            return None
        snapshot: dict[str, Any] = {}
        if url:
            snapshot["url"] = url
        if path:
            snapshot["path"] = path
        rendered = _canonical_json(snapshot)
        snapshot.update(_base_snapshot_meta(rendered, rendered))
        snapshot["sha256"] = _sha256(rendered)
        return snapshot

    if kind == "file_diff":
        files = descriptor.get("files")
        if not isinstance(files, list):
            return None
        normalized_files, rendered_full, rendered_display = _file_diff_snapshot_files(files, caps=caps)
        if not normalized_files:
            return None
        snapshot = _base_snapshot_meta(rendered_full, rendered_display)
        snapshot["sha256"] = _sha256(rendered_full)
        snapshot["files"] = normalized_files
        return snapshot

    return None


def _text_snapshot(text: str, *, caps: ArtifactCaps) -> dict[str, Any]:
    displayed = _truncate_text(text, caps=caps)
    snapshot = _base_snapshot_meta(text, displayed)
    snapshot["sha256"] = _sha256(text)
    snapshot["text"] = displayed
    return snapshot


def _base_snapshot_meta(full_text: str, displayed_text: str) -> dict[str, Any]:
    full_bytes = len(full_text.encode("utf-8"))
    displayed_bytes = len(displayed_text.encode("utf-8"))
    total_lines = _line_count(full_text)
    displayed_lines = _line_count(displayed_text)
    return {
        "truncated": displayed_text != full_text,
        "total_lines": total_lines,
        "displayed_lines": displayed_lines,
        "total_bytes": full_bytes,
        "displayed_bytes": displayed_bytes,
    }


def _truncate_text(text: str, *, caps: ArtifactCaps) -> str:
    max_lines = max(0, int(caps.max_lines))
    max_bytes = max(0, int(caps.max_bytes))
    lines = text.splitlines()
    if max_lines and len(lines) > max_lines:
        displayed = "\n".join(lines[:max_lines])
    elif max_lines == 0 and lines:
        displayed = ""
    else:
        displayed = text

    encoded = displayed.encode("utf-8")
    if max_bytes and len(encoded) > max_bytes:
        displayed = encoded[:max_bytes].decode("utf-8", errors="ignore")
    elif max_bytes == 0 and encoded:
        displayed = ""
    return displayed


def _file_diff_snapshot_files(
    files: list[Any],
    *,
    caps: ArtifactCaps,
) -> tuple[list[dict[str, Any]], str, str]:
    normalized: list[dict[str, Any]] = []
    full_parts: list[str] = []
    displayed_parts: list[str] = []
    remaining_lines = max(0, int(caps.max_lines))
    remaining_bytes = max(0, int(caps.max_bytes))

    for raw in files:
        if not isinstance(raw, dict):
            continue
        path = _clean_string(raw.get("path"))
        full_diff = raw.get("unified_diff_full")
        if not isinstance(full_diff, str):
            full_diff = raw.get("unified_diff")
        if not path or not isinstance(full_diff, str):
            continue

        full_parts.append(full_diff)
        if remaining_lines == 0 or remaining_bytes == 0:
            displayed_diff = ""
        else:
            displayed_diff = _truncate_text(
                full_diff,
                caps=ArtifactCaps(max_lines=remaining_lines, max_bytes=remaining_bytes),
            )
        displayed_parts.append(displayed_diff)

        remaining_lines = max(0, remaining_lines - _line_count(displayed_diff))
        remaining_bytes = max(0, remaining_bytes - len(displayed_diff.encode("utf-8")))

        entry: dict[str, Any] = {
            "path": path,
            "unified_diff": displayed_diff,
            "truncated": displayed_diff != full_diff or bool(raw.get("truncated")),
            "total_lines": _line_count(full_diff),
            "displayed_lines": _line_count(displayed_diff),
        }
        operation = _clean_string(raw.get("operation")) or _clean_string(raw.get("sub_operation"))
        if operation:
            entry["operation"] = operation
        sub_operation = _clean_string(raw.get("sub_operation"))
        if sub_operation:
            entry["sub_operation"] = sub_operation
        normalized.append(entry)

    return normalized, "".join(full_parts), "".join(displayed_parts)


def _default_artifact_id(kind: str, owner: ArtifactOwner, index: int) -> str:
    if kind == "file_diff":
        return f"file_diff:{owner.call_id}" if index == 0 else f"file_diff:{owner.call_id}:{index}"
    return f"{kind}:{owner.call_id}:{index}"


def _truncate_rows(
    rows: list[Any],
    *,
    caps: ArtifactCaps,
    header_lines: int,
) -> tuple[list[Any], dict[str, Any]]:
    max_rows = max(0, int(caps.max_lines) - max(0, int(header_lines)))
    displayed = rows[:max_rows] if len(rows) > max_rows else list(rows)
    full_text = _canonical_json(rows)
    displayed_text = _canonical_json(displayed)
    max_bytes = max(0, int(caps.max_bytes))
    while displayed and max_bytes and len(displayed_text.encode("utf-8")) > max_bytes:
        displayed = displayed[:-1]
        displayed_text = _canonical_json(displayed)
    if max_bytes == 0 and displayed_text:
        displayed = []
        displayed_text = _canonical_json(displayed)
    return copy.deepcopy(displayed), _base_snapshot_meta(full_text, displayed_text)


def _truncate_mapping(
    pairs: dict[str, Any],
    *,
    caps: ArtifactCaps,
) -> tuple[dict[str, Any], dict[str, Any]]:
    max_items = max(0, int(caps.max_lines))
    items = list(pairs.items())
    displayed = dict(items[:max_items]) if len(items) > max_items else dict(items)
    full_text = _canonical_json(pairs)
    displayed_text = _canonical_json(displayed)
    max_bytes = max(0, int(caps.max_bytes))
    while displayed and max_bytes and len(displayed_text.encode("utf-8")) > max_bytes:
        items = list(displayed.items())[:-1]
        displayed = dict(items)
        displayed_text = _canonical_json(displayed)
    if max_bytes == 0 and displayed_text:
        displayed = {}
        displayed_text = _canonical_json(displayed)
    return copy.deepcopy(displayed), _base_snapshot_meta(full_text, displayed_text)


def _coerce_artifact_list(raw_artifacts: Any) -> list[dict[str, Any]]:
    if raw_artifacts is None:
        return []
    if isinstance(raw_artifacts, dict):
        candidates = [raw_artifacts]
    elif isinstance(raw_artifacts, list):
        candidates = raw_artifacts
    else:
        return []
    copied: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        snapshot = _safe_deepcopy(item)
        if isinstance(snapshot, dict):
            copied.append(snapshot)
    return copied


def _revision_map(existing_artifacts: list[dict[str, Any]] | None) -> dict[str, int]:
    revisions: dict[str, int] = {}
    for artifact in existing_artifacts or []:
        if not isinstance(artifact, dict):
            continue
        artifact_id = artifact.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            continue
        revisions[artifact_id] = _coerce_positive_int(artifact.get("revision"), fallback=0)
    return revisions


def _presentation_for(descriptor: dict[str, Any], *, owner: ArtifactOwner) -> dict[str, Any]:
    raw = descriptor.get("presentation")
    presentation = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    surface = _clean_string(presentation.get("surface")) or "iteration_summary"
    group = _clean_string(presentation.get("group")) or _clean_string(descriptor.get("group")) or owner.tool_name
    collapsed = presentation.get("collapsed")
    return {
        "surface": surface,
        "group": group,
        "collapsed": bool(collapsed) if isinstance(collapsed, bool) else True,
    }


def _source_for(descriptor: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    source: dict[str, Any] = {}
    source_path = (
        _clean_string(descriptor.get("source_path"))
        or _clean_string(descriptor.get("path") if descriptor.get("kind") != "link" else None)
    )
    source_url = _clean_string(descriptor.get("source_url"))
    if not source_url and descriptor.get("kind") == "link":
        source_url = _clean_string(snapshot.get("url"))
    if not source_path and descriptor.get("kind") == "link":
        source_path = _clean_string(snapshot.get("path"))
    if source_path:
        source["path"] = source_path
    if source_url:
        source["url"] = source_url
    relative_path = _clean_string(descriptor.get("source_relative_path"))
    if relative_path:
        source["relative_path"] = relative_path
    return source


def _summary_for(kind: str, title: str, snapshot: dict[str, Any]) -> str:
    if kind in {"markdown", "plan"}:
        text = _clean_string(snapshot.get("markdown"))
    elif kind == "log":
        text = _clean_string(snapshot.get("text"))
    elif kind == "file_diff":
        files = snapshot.get("files")
        count = len(files) if isinstance(files, list) else 0
        return f"{count} file{'s' if count != 1 else ''} changed"
    elif kind == "link":
        text = _clean_string(snapshot.get("url")) or _clean_string(snapshot.get("path"))
    else:
        text = ""
    if not text:
        return title
    first_line = text.splitlines()[0].strip()
    if not first_line:
        return title
    return first_line[:160]


def _code_diff_files(interact_config: Any) -> list[dict[str, Any]]:
    if isinstance(interact_config, dict):
        path = _clean_string(interact_config.get("path"))
        unified_diff = interact_config.get("unified_diff")
        if not path or not isinstance(unified_diff, str):
            return []
        entry = {
            "path": path,
            "operation": _clean_string(interact_config.get("operation")) or "edit",
            "unified_diff": unified_diff,
            "truncated": bool(interact_config.get("truncated")),
            "total_lines": _coerce_nonnegative_int(interact_config.get("total_lines"), _line_count(unified_diff)),
            "displayed_lines": _coerce_nonnegative_int(
                interact_config.get("displayed_lines"),
                _line_count(unified_diff),
            ),
        }
        return [entry]
    if isinstance(interact_config, list):
        entries: list[dict[str, Any]] = []
        for item in interact_config:
            if not isinstance(item, dict):
                continue
            path = _clean_string(item.get("path"))
            unified_diff = item.get("unified_diff")
            if not path or not isinstance(unified_diff, str):
                continue
            entries.append(
                {
                    "path": path,
                    "sub_operation": _clean_string(item.get("sub_operation")) or "edit",
                    "unified_diff": unified_diff,
                    "truncated": bool(item.get("truncated")),
                    "total_lines": _coerce_nonnegative_int(item.get("total_lines"), _line_count(unified_diff)),
                    "displayed_lines": _coerce_nonnegative_int(
                        item.get("displayed_lines"),
                        _line_count(unified_diff),
                    ),
                }
            )
        return entries
    return []


def _code_diff_title(interact_config: Any, *, tool_name: str) -> str:
    if isinstance(interact_config, dict):
        title = _clean_string(interact_config.get("title"))
        if title:
            return title
    if tool_name == "git_commit":
        return "Committed staged changes"
    return "File changes"


def _first_markdown_heading(markdown_text: str) -> str:
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _clean_string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())


def _coerce_positive_int(value: Any, *, fallback: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return fallback
    return parsed if parsed > 0 else fallback


def _coerce_nonnegative_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return fallback
    return max(0, parsed)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
