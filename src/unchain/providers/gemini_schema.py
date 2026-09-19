"""Projection of JSON Schema onto Gemini's function declaration Schema subset."""
from __future__ import annotations

import copy
from typing import Any


def sanitize_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise ValueError("Gemini schema must be an object")
    root = copy.deepcopy(schema)
    allowed = {
        "type",
        "description",
        "enum",
        "nullable",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "pattern",
    }

    def project(node: Any, refs: tuple[str, ...] = ()) -> dict[str, Any]:
        if not isinstance(node, dict):
            raise ValueError("Gemini schema nodes must be objects")
        node = copy.deepcopy(node)
        ref = node.pop("$ref", None)
        if ref is not None:
            if not isinstance(ref, str) or not ref.startswith("#/") or ref in refs:
                raise ValueError(
                    "Gemini schema requires non-recursive local references"
                )
            target = root
            try:
                for key in ref[2:].split("/"):
                    target = target[key.replace("~1", "/").replace("~0", "~")]
            except (KeyError, TypeError) as exc:
                raise ValueError("Unresolved Gemini schema reference") from exc
            return project({**target, **node}, (*refs, ref))
        for union in ("anyOf", "oneOf"):
            if union in node:
                branches = node.pop(union)
                if not isinstance(branches, list) or not branches:
                    raise ValueError("Invalid Gemini schema union")
                concrete = [b for b in branches if b != {"type": "null"}]
                if len(concrete) != 1:
                    raise ValueError(
                        "Gemini schema cannot flatten a heterogeneous union"
                    )
                return project(
                    {**concrete[0], **node, "nullable": len(concrete) != len(branches)},
                    refs,
                )
        if "allOf" in node or "not" in node:
            raise ValueError("Unsupported Gemini schema composition")
        result = {k: copy.deepcopy(v) for k, v in node.items() if k in allowed}
        typ = result.get("type")
        if isinstance(typ, list):
            concrete = [t for t in typ if t != "null"]
            if len(concrete) != 1:
                raise ValueError("Gemini schema cannot flatten multiple types")
            result["type"] = concrete[0]
            result["nullable"] = "null" in typ
        if "properties" in node:
            result["properties"] = {
                k: project(v, refs) for k, v in node["properties"].items()
            }
        if "required" in node:
            required = node["required"]
            if not isinstance(required, list) or any(
                k not in result.get("properties", {}) for k in required
            ):
                raise ValueError("Gemini required fields must name declared properties")
            if required:
                result["required"] = list(required)
        if "items" in node:
            result["items"] = project(node["items"], refs)
        if node.get("format") in {"date-time", "enum"}:
            result["format"] = node["format"]
        return result

    return project(root)
