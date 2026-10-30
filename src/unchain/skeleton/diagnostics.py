"""Public draft diagnostics for Skeleton structural validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkeletonDiagnostic:
    code: str
    path: tuple[str | int, ...]
    message: str
