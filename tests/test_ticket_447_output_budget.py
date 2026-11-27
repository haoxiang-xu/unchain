from __future__ import annotations

import hashlib
import json

import pytest

from unchain.journal import ResourceRef
from unchain.tools.output_management import ToolOutputManager, canonical_content_bytes


@pytest.mark.parametrize("policy", ["default", "head_tail"])
@pytest.mark.parametrize("text", ["plain text", "雪🙂" * 100, '\\"' * 100])
def test_complete_output_fits_final_serialized_budget_without_duplicate_body(policy, text):
    raw = json.dumps({"content": text}, ensure_ascii=False).encode()
    result = ToolOutputManager.active_default().project(
        raw, full_output_ref=ResourceRef("artifact", "budget-source", 1).to_dict(),
        digest=hashlib.sha256(raw).hexdigest(), content_bytes=len(raw),
        requested_policy=policy,
    )
    assert result.payload["inline"] is True
    assert json.loads(result.payload["preview"])["content"] == text
    assert "tail_preview" not in result.payload
    assert len(canonical_content_bytes(result.payload)) <= 16000


@pytest.mark.parametrize("policy", ["default", "head_tail"])
def test_projection_escaping_counts_toward_the_inline_budget(policy):
    raw = json.dumps({"content": '\\"' * 2300}).encode()
    assert len(raw) < 16000
    result = ToolOutputManager.active_default().project(
        raw, full_output_ref=ResourceRef("artifact", "escape-source", 1).to_dict(),
        digest=hashlib.sha256(raw).hexdigest(), content_bytes=len(raw),
        requested_policy=policy,
    )
    assert result.payload.get("inline") is not True
    assert result.payload["read_request"]["arguments"]["offset"] == 0
    assert result.payload["content_sha256"] == hashlib.sha256(raw).hexdigest()


def test_artifact_only_does_not_inline_small_sensitive_output():
    raw = b'"small"'
    result = ToolOutputManager.active_default().project(
        raw, full_output_ref=ResourceRef("artifact", "artifact-only", 1).to_dict(),
        digest=hashlib.sha256(raw).hexdigest(), content_bytes=len(raw),
        requested_policy="artifact_only",
    )
    assert result.payload["projection"] == "artifact_only"
    assert "preview" not in result.payload
