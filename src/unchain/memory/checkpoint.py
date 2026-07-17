from __future__ import annotations

import copy
from dataclasses import dataclass

from ..tools.receipts import (
    DURABLE_TOOL_RECEIPT_LEDGER_KEY,
    validate_durable_tool_receipt_ledger,
)
from .base import BaseMemoryHarness, MemoryContext
from .checkpoint_state import (
    build_execution_checkpoint,
    transcript_digest,
)
from .effects import (
    build_memory_delta,
    memory_commit_update,
    memory_state_update,
)


@dataclass
class ExecutionCheckpointHarness(BaseMemoryHarness):
    """Persist resumable execution state separately from semantic memory."""

    durable_barrier = True
    name: str = "memory_execution_checkpoint"
    phases: tuple[str, ...] = (
        "before_model",
        "suspend_persist",
        "finalize_persist",
    )
    order: int = 90

    def build_memory_delta(self, context: MemoryContext):
        status = str(context.event.get("status") or context.state.run_status or "")
        receipt_ledger = validate_durable_tool_receipt_ledger(
            context.state.metadata.get(DURABLE_TOOL_RECEIPT_LEDGER_KEY)
        )
        if context.phase == "before_model" and not receipt_ledger["pending"]:
            return None
        if not context.session_id:
            return build_memory_delta(
                created_by=self.created_by,
                trace={
                    "execution_checkpoint_applied": False,
                    "execution_checkpoint_skip_reason": "missing_session_id",
                    "status": status,
                },
            )

        if context.phase == "before_model":
            return self._persist(
                context,
                status="tool_result_ready",
                receipt_ledger=receipt_ledger,
            )
        if context.phase == "suspend_persist":
            suspend_status = str(context.event.get("status") or "awaiting_human_input")
            if suspend_status not in {
                "awaiting_human_input",
                "awaiting_interaction",
                "max_iterations",
            }:
                return None
            return self._persist(
                context,
                status=suspend_status,
                receipt_ledger=receipt_ledger,
            )
        if context.phase != "finalize_persist":
            return None
        if status == "max_iterations":
            return self._persist(
                context,
                status="max_iterations",
                receipt_ledger=receipt_ledger,
            )
        if status == "completed":
            return self._complete(context, receipt_ledger=receipt_ledger)
        if receipt_ledger["pending"]:
            return self._persist(
                context,
                status="tool_result_ready",
                receipt_ledger=receipt_ledger,
            )
        return None

    def _persist(
        self,
        context: MemoryContext,
        *,
        status: str,
        receipt_ledger: dict,
    ):
        checkpoint = build_execution_checkpoint(
            context.state,
            status=status,
            run_id=context.run_id,
        )
        suspend_payload = context.state.suspend_state.payload
        interaction_request = (
            suspend_payload.get("interaction_request")
            if isinstance(suspend_payload, dict)
            else None
        )
        persisted, session_snapshot = context.runtime.save_execution_checkpoint_snapshot(
            context.session_id,
            checkpoint,
            interaction_request=interaction_request,
            durable_receipt_ledger=receipt_ledger,
            expected_revision=context.state.memory_state.get("session_revision"),
            execution_fence=context.execution_fence,
        )
        replay_frame = persisted.get("replay_frame")
        replay_complete = bool(
            isinstance(replay_frame, dict) and replay_frame.get("complete") is True
        )
        return build_memory_delta(
            created_by=self.created_by,
            state_updates=memory_state_update(
                {
                    "execution_checkpoint_restored": False,
                    "execution_checkpoint_status": status,
                    "execution_checkpoint_id": str(
                        persisted.get("checkpoint_id") or ""
                    ),
                    "session_snapshot": copy.deepcopy(session_snapshot.state),
                    "session_revision": session_snapshot.revision,
                    "session_revision_supported": session_snapshot.revision_supported,
                    "session_consistency": session_snapshot.consistency,
                }
            ),
            trace={
                "execution_checkpoint_applied": True,
                "execution_checkpoint_action": "persisted",
                "execution_checkpoint_status": status,
                "execution_checkpoint_id": persisted.get("checkpoint_id"),
                "execution_checkpoint_replay_complete": replay_complete,
                "execution_checkpoint_message_count": len(
                    persisted.get("transcript") or []
                ),
                "durable_tool_receipt_applied_count": len(
                    receipt_ledger["pending"]
                ),
            },
        )

    def _complete(
        self,
        context: MemoryContext,
        *,
        receipt_ledger: dict,
    ):
        current_digest = transcript_digest(context.state.transcript)
        committed_digest = str(
            context.state.memory_commit_info.get("transcript_digest") or ""
        )
        commit_info = None
        expected_revision = context.state.memory_state.get("session_revision")
        expected_checkpoint_id = str(
            context.state.memory_state.get("execution_checkpoint_id") or ""
        )
        if committed_digest != current_digest or receipt_ledger["pending"]:
            summary_bucket = context.state.optimizer_state.get("llm_summary", {})
            summary_text = (
                str(summary_bucket.get("summary") or "").strip()
                if isinstance(summary_bucket, dict)
                else ""
            )
            commit_info, _ = context.runtime.commit_transcript(
                session_id=context.session_id,
                transcript=context.state.transcript,
                memory_namespace=context.memory_namespace,
                model=context.model,
                summary_text=summary_text,
                expected_revision=expected_revision,
                expected_checkpoint_id=expected_checkpoint_id or None,
                durable_receipt_ledger=receipt_ledger,
                execution_fence=context.execution_fence,
            )
            expected_revision = commit_info.get("session_revision")

        cleared, session_snapshot = context.runtime.clear_execution_checkpoint_snapshot(
            context.session_id,
            expected_checkpoint_id=expected_checkpoint_id or None,
            expected_revision=expected_revision,
            execution_fence=context.execution_fence,
        )

        memory_state = {
            "execution_checkpoint_restored": False,
            "execution_checkpoint_status": "",
            "execution_checkpoint_id": "",
            "session_revision": session_snapshot.revision,
            "session_revision_supported": session_snapshot.revision_supported,
            "session_consistency": session_snapshot.consistency,
        }
        final_state = copy.deepcopy(session_snapshot.state)
        memory_state.update(
            {
                "session_snapshot": final_state,
                "summary": str(final_state.get("summary") or ""),
            }
        )
        state_updates = memory_state_update(memory_state)
        if isinstance(commit_info, dict):
            state_updates.update(memory_commit_update(commit_info))
        atomically_cleared = bool(
            isinstance(commit_info, dict)
            and commit_info.get("execution_checkpoint_cleared")
        )

        return build_memory_delta(
            created_by=self.created_by,
            state_updates=state_updates,
            trace={
                "execution_checkpoint_applied": True,
                "execution_checkpoint_action": (
                    "cleared" if cleared or atomically_cleared else "absent"
                ),
                "execution_checkpoint_cleared": cleared or atomically_cleared,
                "semantic_commit_repaired": isinstance(commit_info, dict),
                "transcript_digest": current_digest,
            },
        )


__all__ = ["ExecutionCheckpointHarness"]
