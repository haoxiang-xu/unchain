"""Developer-facing presentation policy for adjacent tool-call timeline groups."""

from __future__ import annotations

TIMELINE_MERGE_POLICIES = frozenset({"never", "no_feedback", "approved", "always"})
DEFAULT_TIMELINE_MERGE_POLICY = "approved"


def validate_timeline_merge_policy(value: object) -> str:
    if not isinstance(value, str) or value not in TIMELINE_MERGE_POLICIES:
        raise ValueError(
            "timeline_merge_policy must be one of: never, no_feedback, approved, always"
        )
    return value
