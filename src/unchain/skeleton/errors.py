"""Errors raised while validating Skeleton protocol values."""

from __future__ import annotations


class SkeletonValidationError(ValueError):
    """A validation error with a machine-readable code and input location."""

    def __init__(self, code: str, path: tuple[str | int, ...], message: str):
        self.code = code
        self.path = path
        super().__init__(message)
