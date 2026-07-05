from __future__ import annotations

import threading
from collections import deque
from typing import Any

_SIDE_SYSTEM_TEMPLATE = (
    "You are a side assistant. A main agent is currently working on this task:\n"
    "{task}\n\n"
    "Progress so far (from the live event stream):\n"
    "{digest}\n\n"
    "The user is asking a quick side question while the main agent keeps "
    "working. Answer briefly and directly based on the progress above. Do not "
    "start doing the task yourself."
)


class ProgressDigest:
    """Bounded, thread-safe collector turning raw run events into a short summary.

    Attach as an extra callback alongside the main one; it never raises.
    """

    def __init__(self, max_entries: int = 20) -> None:
        self._lock = threading.Lock()
        self._entries: deque[str] = deque(maxlen=max_entries)
        self._iterations = 0

    def __call__(self, event: dict[str, Any]) -> None:
        try:
            etype = str(event.get("type") or event.get("event") or "")
            if not etype:
                return
            with self._lock:
                if etype == "iteration_started":
                    self._iterations += 1
                elif "tool" in etype:
                    tool = event.get("tool_name") or event.get("tool") or event.get("name") or ""
                    if tool:
                        self._entries.append(f"tool: {tool}")
                elif etype in ("response_received", "final_message"):
                    text = str(event.get("content") or event.get("final_text") or "")[:120]
                    if text:
                        self._entries.append(f"assistant: {text}")
        except Exception:  # noqa: BLE001 - digest must never break the main run
            return

    def summary(self) -> str:
        with self._lock:
            lines = [f"iterations: {self._iterations}"]
            if self._entries:
                lines.append("recent: " + "; ".join(self._entries))
        return "\n".join(lines)


def build_btw_prompt(original_task: str, digest_summary: str, question: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": _SIDE_SYSTEM_TEMPLATE.format(task=original_task, digest=digest_summary),
        },
        {"role": "user", "content": question},
    ]
