from __future__ import annotations

from dataclasses import dataclass

from ..tools.receipts import DURABLE_TOOL_RECEIPT_LEDGER_KEY
from .base import BaseMemoryHarness, MemoryContext
from .effects import build_memory_delta, memory_commit_update, memory_state_update


@dataclass
class MemoryCommitHarness(BaseMemoryHarness):
    name: str = "memory_commit"
    phases: tuple[str, ...] = ("before_commit",)
    order: int = 10

    def build_memory_delta(self, context: MemoryContext) -> HarnessDelta | None:
        summary_bucket = context.state.optimizer_state.get("llm_summary", {})
        summary_text = ""
        if isinstance(summary_bucket, dict):
            summary_text = str(summary_bucket.get("summary", "") or "").strip()
        commit_info, stored_state = context.runtime.commit_transcript(
            session_id=context.session_id,
            transcript=context.state.transcript,
            memory_namespace=context.memory_namespace,
            model=context.model,
            summary_text=summary_text,
            expected_revision=context.state.memory_state.get("session_revision"),
            expected_checkpoint_id=(
                str(context.state.memory_state.get("execution_checkpoint_id") or "")
                or None
            ),
            durable_receipt_ledger=context.state.metadata.get(
                DURABLE_TOOL_RECEIPT_LEDGER_KEY
            ),
            execution_fence=context.execution_fence,
        )
        memory_state = {
            "loaded": bool(context.session_id),
            "resume_mode": False,
            "session_id": context.session_id,
            "memory_namespace": commit_info.get("memory_namespace", context.memory_namespace),
            "session_snapshot": stored_state,
            "vector_indexed_until": int(stored_state.get("vector_indexed_until") or 0),
            "long_term_indexed_until": int(stored_state.get("long_term_indexed_until") or 0),
            "long_term_pending_turn_count": int(stored_state.get("long_term_pending_turn_count") or 0),
            "summary": str(stored_state.get("summary", "") or ""),
            "session_revision": commit_info.get("session_revision"),
            "session_revision_supported": bool(
                commit_info.get("session_revision_supported", False)
            ),
            "session_consistency": str(
                commit_info.get("session_consistency") or "best_effort"
            ),
        }
        if commit_info.get("execution_checkpoint_cleared"):
            memory_state.update(
                {
                    "execution_checkpoint_restored": False,
                    "execution_checkpoint_status": "",
                    "execution_checkpoint_id": "",
                }
            )
        return build_memory_delta(
            created_by=self.created_by,
            state_updates={
                **memory_state_update(memory_state),
                **memory_commit_update(commit_info),
            },
            trace={
                "stored_message_count": int(commit_info.get("stored_message_count") or 0),
                "summary_persisted": bool(commit_info.get("summary_persisted")),
            },
        )
