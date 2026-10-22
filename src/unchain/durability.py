"""Dependency-free classification for durable persistence boundary failures.

Durable writes are correctness boundaries: callers may add context to an
exception, but they must never turn the failure into a retry, fallback, tool
result, or partial agent result.  This module intentionally uses only Python's
standard exception protocol so low-level persistence code and high-level
runtime code can share the same classification without import cycles.
"""

from __future__ import annotations


_DURABLE_FAILURE_MARKER = "_unchain_durable_persistence_failure"
MAX_DURABLE_FAILURE_TRAVERSAL_NODES = 1_024
MAX_DURABLE_FAILURE_TRAVERSAL_DEPTH = 32
_UNREADABLE_EXCEPTION_ATTRIBUTE = object()


class DurablePersistenceBoundaryError(RuntimeError):
    """Identity-preserving wrapper for exceptions that cannot carry a marker."""

    code = "durable_persistence_boundary_failure"

    def __init__(self, original: BaseException) -> None:
        if not isinstance(original, BaseException):
            raise TypeError("original must be an exception")
        self.original = original
        self.__suppress_context__ = True
        # Never copy original args/str/repr into diagnostics: persistence
        # failures may originate in providers whose exceptions contain secret
        # request material.
        super().__init__(self.code)


class DurablePersistenceTraversalExhaustedError(RuntimeError):
    """Safe fail-closed result when exception traversal cannot finish.

    This classifier outcome intentionally contains no source exception.  It is
    returned by :func:`find_durable_persistence_failure` when a depth/node
    budget or hostile exception attribute prevents a complete classification,
    ensuring callers preserve the caught exception instead of treating it as
    an ordinary failure.
    """

    code = "durable_persistence_traversal_exhausted"

    def __init__(self) -> None:
        self.__suppress_context__ = True
        super().__init__(self.code)


def mark_durable_persistence_failure(
    error: BaseException,
) -> BaseException:
    """Mark ``error`` in place, or return a boundary wrapper if it is immutable."""

    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception")
    if isinstance(error, DurablePersistenceBoundaryError):
        return error
    try:
        setattr(error, _DURABLE_FAILURE_MARKER, True)
        if bool(getattr(error, _DURABLE_FAILURE_MARKER, False)):
            return error
    except Exception:
        # Immutable exception objects use the boundary wrapper below instead of an attached marker.
        pass
    return DurablePersistenceBoundaryError(error)


def find_durable_persistence_failure(
    error: BaseException,
) -> BaseException | None:
    """Find a durable boundary through known wrappers, groups, and cycles."""

    if not isinstance(error, BaseException):
        return None
    pending: list[tuple[BaseException, int]] = [(error, 0)]
    queued: set[int] = {id(error)}
    visited: set[int] = set()
    visited_nodes = 0
    traversal_truncated = False
    while pending and visited_nodes < MAX_DURABLE_FAILURE_TRAVERSAL_NODES:
        current, depth = pending.pop()
        identity = id(current)
        queued.discard(identity)
        if identity in visited:
            continue
        visited.add(identity)
        visited_nodes += 1
        marker = _safe_exception_attribute(current, _DURABLE_FAILURE_MARKER)
        if isinstance(
            current,
            (
                DurablePersistenceBoundaryError,
                DurablePersistenceTraversalExhaustedError,
            ),
        ) or marker is True:
            return current
        if marker is _UNREADABLE_EXCEPTION_ATTRIBUTE:
            traversal_truncated = True

        candidates: list[BaseException] = []
        for attribute_name in (
            "original",
            "real",
            "last_error",
            "__cause__",
            "__context__",
        ):
            candidate = _safe_exception_attribute(current, attribute_name)
            if candidate is _UNREADABLE_EXCEPTION_ATTRIBUTE:
                traversal_truncated = True
            elif isinstance(candidate, BaseException):
                candidates.append(candidate)
        if isinstance(current, BaseExceptionGroup):
            group_exceptions = _safe_exception_attribute(current, "exceptions")
            if group_exceptions is _UNREADABLE_EXCEPTION_ATTRIBUTE:
                traversal_truncated = True
            elif isinstance(group_exceptions, tuple):
                group_limit = max(
                    0,
                    MAX_DURABLE_FAILURE_TRAVERSAL_NODES
                    - visited_nodes
                    - len(pending),
                )
                group_prefix, group_truncated = _safe_tuple_prefix(
                    group_exceptions,
                    group_limit,
                )
                if group_truncated:
                    traversal_truncated = True
                candidates.extend(
                    candidate
                    for candidate in group_prefix
                    if isinstance(candidate, BaseException)
                )

        unique_candidates: list[BaseException] = []
        candidate_ids: set[int] = set()
        for candidate in candidates:
            candidate_id = id(candidate)
            if (
                candidate_id in visited
                or candidate_id in queued
                or candidate_id in candidate_ids
            ):
                continue
            candidate_ids.add(candidate_id)
            unique_candidates.append(candidate)

        if depth >= MAX_DURABLE_FAILURE_TRAVERSAL_DEPTH:
            if unique_candidates:
                traversal_truncated = True
            continue

        available_slots = max(
            0,
            MAX_DURABLE_FAILURE_TRAVERSAL_NODES
            - visited_nodes
            - len(pending),
        )
        if len(unique_candidates) > available_slots:
            traversal_truncated = True
        valid_candidates = unique_candidates[:available_slots]
        queued.update(id(candidate) for candidate in valid_candidates)
        pending.extend(
            (candidate, depth + 1)
            for candidate in reversed(valid_candidates)
        )
    if pending:
        traversal_truncated = True
    return (
        DurablePersistenceTraversalExhaustedError()
        if traversal_truncated
        else None
    )


def _safe_exception_attribute(error: BaseException, name: str) -> object:
    try:
        return getattr(error, name, None)
    except BaseException:
        # Exception accessors can themselves raise; incomplete evidence must
        # fail closed rather than make a durable failure appear retryable.
        return _UNREADABLE_EXCEPTION_ATTRIBUTE


def _safe_tuple_prefix(value: tuple, limit: int) -> tuple[tuple, bool]:
    try:
        length = tuple.__len__(value)
        prefix = tuple.__getitem__(value, slice(0, max(0, limit)))
    except BaseException:
        # An interrupted inspection leaves the original exception group
        # unclassified, so preserve the fail-closed traversal result.
        return (), True
    return prefix, length > max(0, limit)


def is_durable_persistence_failure(error: BaseException) -> bool:
    """Return whether ``error`` contains a durable persistence boundary."""

    return find_durable_persistence_failure(error) is not None


__all__ = [
    "DurablePersistenceBoundaryError",
    "DurablePersistenceTraversalExhaustedError",
    "MAX_DURABLE_FAILURE_TRAVERSAL_DEPTH",
    "MAX_DURABLE_FAILURE_TRAVERSAL_NODES",
    "find_durable_persistence_failure",
    "is_durable_persistence_failure",
    "mark_durable_persistence_failure",
]
