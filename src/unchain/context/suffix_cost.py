"""Per-call sufficient statistics for the built-in deterministic token estimator."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from unchain.optimizers.common import _message_to_text

from .budget import (
    ContextTokenEstimate,
    _IMAGE_TOKEN_CHARGE,
    _PDF_PAGE_TOKEN_CHARGE,
    estimate_context_tokens,
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def _cost(message: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    normalized = _plain(message)
    role = normalized.get("role")
    chars = (len(role) + 1 if isinstance(role, str) else 0) + len(_message_to_text(normalized))
    size = len(json.dumps(normalized, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), default=str).encode("utf-8"))
    media = estimate_context_tokens((normalized,))
    return (1, chars, size, media.image_count, media.pdf_page_count)


class SuffixTokenCost:
    """Exact for estimate_context_tokens, not a BPE/custom-counter approximation.

    Store raw lengths, not rounded per-message token counts. List framing and
    max/ceil are applied once to the complete candidate, including fixed items.
    """

    def __init__(self, messages: Sequence[Mapping[str, Any]], *, fixed=()) -> None:
        self._fixed = tuple(sum(values) for values in zip(*map(_cost, fixed))) if fixed else (0,) * 5
        suffix = [(0,) * 5]
        for message in reversed(messages):
            suffix.append(tuple(a + b for a, b in zip(_cost(message), suffix[-1], strict=True)))
        self._suffix = tuple(reversed(suffix))

    def estimate(self, start: int = 0) -> ContextTokenEstimate:
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < len(self._suffix):
            raise ValueError("suffix start is out of range")
        count, chars, size, images, pages = (
            a + b for a, b in zip(self._fixed, self._suffix[start], strict=True)
        )
        framed_bytes = 2 + size + max(0, count - 1)
        return ContextTokenEstimate(
            text_tokens=max((chars + 3) // 4, (framed_bytes + 3) // 4),
            multimodal_tokens=images * _IMAGE_TOKEN_CHARGE + pages * _PDF_PAGE_TOKEN_CHARGE,
            image_count=images,
            pdf_page_count=pages,
        )
