from __future__ import annotations

import hashlib
import json

from unchain.context.composition import _classify_message, _measured_message_contributions
from unchain.runtime.runtime_protocol import (
    RuntimeProtocolManifest,
    runtime_protocol_manifest,
)


def test_runtime_manifest_advertises_versioned_skills_protocol():
    manifest = runtime_protocol_manifest()
    by_id = {item["id"]: item for item in manifest["protocols"]}

    skills = by_id["skills"]
    assert (skills["major"], skills["minor"]) == (1, 0)
    assert skills["features"] == [
        "active_skills_snapshot_v1",
        "catalog_v1",
        "skill_md_registry_v1",
        "skill_tool_v1",
        "toolkit_embedded_skills_v1",
        "user_invocation_v1",
    ]
    # Canonical order and digest still hold with the extra protocol; the strict
    # parser (the same one PuPu embeds) must round-trip it unchanged.
    assert [item["id"] for item in manifest["protocols"]] == sorted(by_id)
    assert RuntimeProtocolManifest.from_dict(manifest).to_dict() == manifest


def test_composition_classifies_skills_blocks_into_reserved_slots():
    catalog = {"role": "system", "content": "<available_skills>\n# unchain generated skills catalog v1\n- `a`: b\n</available_skills>"}
    active = {"role": "system", "content": "<active_skills>\n# unchain generated active skills v1\n</active_skills>"}
    plain = {"role": "system", "content": "You are a coding assistant."}

    assert _classify_message(catalog, current_input=False) == ("skills", "catalog_metadata")
    assert _classify_message(active, current_input=False) == ("skills", "loaded_body")
    assert _classify_message(plain, current_input=False) == ("instructions", "core_system")

    contributions = _measured_message_contributions(
        [plain, catalog, active, {"role": "user", "content": "hi"}]
    )
    slots = {(item["category"], item["subtype"]) for item in contributions}
    assert ("skills", "catalog_metadata") in slots
    assert ("skills", "loaded_body") in slots
    assert ("instructions", "core_system") in slots
