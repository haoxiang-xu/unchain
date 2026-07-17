from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...jobs.plugin import DurableShellJobPlugin
from ...tools.receipts import (
    DurableToolReceiptAckHarness,
    acknowledge_durable_tool_receipt_applications,
)
from .base import BaseAgentModule

if TYPE_CHECKING:
    from ...jobs.process import ProcessJobSupervisor


@dataclass(frozen=True)
class JobsModule(BaseAgentModule):
    """Attach durable shell-job routing without replacing the shell tool."""

    supervisor: ProcessJobSupervisor
    name: str = field(default="jobs", init=False)

    def configure(self, builder) -> None:
        plugin = DurableShellJobPlugin(supervisor=self.supervisor)
        builder.add_tool_runtime_plugin(plugin)
        builder.add_harness(
            DurableToolReceiptAckHarness(plugins=(plugin,))
        )

        def _ack_persisted_receipts(result):
            runtime = builder.memory_runtime
            session_id = str(builder.call_context.session_id or "")
            if runtime is not None and session_id:
                snapshot = runtime.load_session_snapshot(session_id)
                acknowledge_durable_tool_receipt_applications(
                    [plugin],
                    snapshot.state,
                    execution_id=session_id,
                )
            return result

        builder.add_run_hook(_ack_persisted_receipts)


__all__ = ["JobsModule"]
