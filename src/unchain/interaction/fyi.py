from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FyiMessage:
    text: str
    origin: str = "user"
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class FyiChannel:
    """Thread-safe mid-run message channel.

    The caller keeps a reference and may ``post`` from any thread while the
    agent loop drains pending messages at each ``before_model`` boundary.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[FyiMessage] = []

    def post(self, text: str, *, origin: str = "user") -> str:
        message = FyiMessage(text=text, origin=origin)
        with self._lock:
            self._pending.append(message)
        return message.message_id

    def drain(self) -> list[FyiMessage]:
        with self._lock:
            drained = self._pending
            self._pending = []
        return drained

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)
