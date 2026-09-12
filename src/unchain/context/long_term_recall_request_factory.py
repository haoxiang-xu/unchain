"""Additive first-message long-term recall projection for Context V2.

The decorator keeps the compiler as the only context budget authority.  It
adds only bounded reference metadata as a provider-neutral user message and
never injects long-term entry content or promotes recalled data to an
instruction role.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from unchain.journal.models import ModelValidationError, _required_text
from unchain.kernel.harness import HarnessContext
from unchain.memory.long_term_recall_v2 import (
    LongTermFirstMessageRecall,
    LongTermRecallDisposition,
    LongTermRecallEnvelope,
)

from .models import ContextCompileRequest, SourceMessageCursor
from .runtime import ContextRequestFactory


_REFERENCE_MARKER = "MEMORY_V2_UNTRUSTED_LONG_TERM_REFERENCES"


class LongTermRecallContextRequestFactoryError(RuntimeError):
    """A first-message recall could not safely decorate a compile request."""


@dataclass(frozen=True, slots=True)
class LongTermRecallContextOutcome:
    """Typed host action and placement metadata for one recall lookup."""

    envelope: LongTermRecallEnvelope
    injected: bool
    reference_message_index: int | None
    current_user_message_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, LongTermRecallEnvelope):
            raise TypeError("envelope must be a LongTermRecallEnvelope")
        if not isinstance(self.injected, bool):
            raise TypeError("injected must be a boolean")
        if (
            isinstance(self.current_user_message_index, bool)
            or not isinstance(self.current_user_message_index, int)
            or self.current_user_message_index < 0
        ):
            raise ValueError("current_user_message_index is invalid")
        if self.injected:
            if (
                self.envelope.disposition
                is not LongTermRecallDisposition.CONTEXT_REFERENCES
                or isinstance(self.reference_message_index, bool)
                or not isinstance(self.reference_message_index, int)
                or self.reference_message_index < 0
                or self.reference_message_index
                >= self.current_user_message_index
            ):
                raise ModelValidationError(
                    "injected recall outcome has invalid placement"
                )
        elif self.reference_message_index is not None:
            raise ModelValidationError(
                "non-injected recall outcome cannot name a reference message"
            )

    @property
    def disposition(self) -> LongTermRecallDisposition:
        return self.envelope.disposition

    @property
    def curator_required(self) -> bool:
        return (
            self.disposition
            is LongTermRecallDisposition.CURATOR_REQUIRED
        )


LongTermRecallOutcomeSink = Callable[[LongTermRecallContextOutcome], Any]


def _current_user_message(
    request: ContextCompileRequest,
) -> tuple[int, str]:
    marker = f"[{_REFERENCE_MARKER}]"
    if any(
        marker in str(message.get("content") or "")
        for message in request.source_messages
        if isinstance(message, Mapping)
    ):
        raise LongTermRecallContextRequestFactoryError(
            "base request already owns long-term recall projection"
        )
    user_messages = tuple(
        (index, message)
        for index, message in enumerate(request.source_messages)
        if isinstance(message, Mapping)
        and str(message.get("role") or "").strip().casefold() == "user"
    )
    if len(user_messages) != 1:
        raise LongTermRecallContextRequestFactoryError(
            "first-message recall requires one current user message"
        )
    index, message = user_messages[0]
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LongTermRecallContextRequestFactoryError(
            "first-message recall requires text current input"
        )
    return index, content


def _untrusted_reference_message(
    envelope: LongTermRecallEnvelope,
) -> dict[str, str]:
    from .untrusted import untrusted_context_message

    payload = envelope.to_dict()
    if (
        payload.get("trusted") is not False
        or payload.get("placement") != "context_reference"
    ):
        raise LongTermRecallContextRequestFactoryError(
            "recall envelope changed its untrusted placement"
        )
    return untrusted_context_message(_REFERENCE_MARKER, payload)


def _shifted_cursors(
    request: ContextCompileRequest,
    *,
    insertion_index: int,
) -> tuple[SourceMessageCursor, ...]:
    cursors = request.source_message_cursors
    if request.source_event_ids:
        cursors = tuple(
            SourceMessageCursor(
                message_index=index,
                event_id=event_id,
                store_seq=request.source_event_store_seqs[index],
            )
            for index, event_id in enumerate(request.source_event_ids)
        )
    return tuple(
        SourceMessageCursor(
            message_index=(
                cursor.message_index + 1
                if cursor.message_index >= insertion_index
                else cursor.message_index
            ),
            event_id=cursor.event_id,
            store_seq=cursor.store_seq,
        )
        for cursor in cursors
    )


@dataclass(frozen=True, slots=True)
class LongTermRecallContextRequestFactory:
    """Compose any Context V2 request factory with first-message recall."""

    binding_id: str
    base_factory: ContextRequestFactory
    recall: LongTermFirstMessageRecall
    outcome_sink: LongTermRecallOutcomeSink | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "binding_id",
            _required_text(
                self.binding_id,
                "binding_id",
                maximum=512,
                identifier=True,
            ),
        )
        if not callable(self.base_factory):
            raise TypeError("base_factory must be a ContextRequestFactory")
        if not isinstance(self.recall, LongTermFirstMessageRecall):
            raise TypeError("recall must be a LongTermFirstMessageRecall")
        if self.recall.binding_id != self.binding_id:
            raise LongTermRecallContextRequestFactoryError(
                "long-term recall belongs to another binding"
            )
        if self.outcome_sink is not None and not callable(self.outcome_sink):
            raise TypeError("outcome_sink must be callable or None")

    @property
    def attempt(self) -> Any:
        return self.base_factory.attempt

    @property
    def journal(self) -> Any:
        return self.base_factory.journal

    def decorate(
        self,
        context: HarnessContext,
    ) -> tuple[ContextCompileRequest, LongTermRecallContextOutcome]:
        request = self.base_factory(context)
        if not isinstance(request, ContextCompileRequest):
            raise LongTermRecallContextRequestFactoryError(
                "base factory returned an invalid context request"
            )
        user_index, first_message = _current_user_message(request)
        try:
            envelope = self.recall.recall_first_message(first_message)
        except Exception as error:
            raise LongTermRecallContextRequestFactoryError(
                "long-term first-message recall failed closed"
            ) from error
        if not isinstance(envelope, LongTermRecallEnvelope):
            raise LongTermRecallContextRequestFactoryError(
                "long-term recall returned an invalid envelope"
            )

        if (
            envelope.disposition
            is not LongTermRecallDisposition.CONTEXT_REFERENCES
        ):
            return request, LongTermRecallContextOutcome(
                envelope=envelope,
                injected=False,
                reference_message_index=None,
                current_user_message_index=user_index,
            )

        source_messages = list(request.source_messages)
        source_messages.insert(
            user_index,
            _untrusted_reference_message(envelope),
        )
        decorated = replace(
            request,
            source_messages=tuple(source_messages),
            source_event_ids=(),
            source_event_store_seqs=(),
            source_message_cursors=_shifted_cursors(
                request,
                insertion_index=user_index,
            ),
        )
        return decorated, LongTermRecallContextOutcome(
            envelope=envelope,
            injected=True,
            reference_message_index=user_index,
            current_user_message_index=user_index + 1,
        )

    def __call__(self, context: HarnessContext) -> ContextCompileRequest:
        request, outcome = self.decorate(context)
        if self.outcome_sink is not None:
            self.outcome_sink(outcome)
        return request


__all__ = [
    "LongTermRecallContextOutcome",
    "LongTermRecallContextRequestFactory",
    "LongTermRecallContextRequestFactoryError",
    "LongTermRecallOutcomeSink",
]
