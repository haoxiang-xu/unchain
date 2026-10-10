"""Provider-free helpers for the Skeleton protocol."""

from .errors import SkeletonValidationError
from .conditions import evaluate_condition, select_case
from .diagnostics import SkeletonDiagnostic
from .io import map_outputs, resolve_inputs
from .legacy import convert_legacy_definition, materialize_legacy_agent
from .preparation import PreparedFlowEdge, PreparedWorkflowPlan, prepare_workflow
from .recovery import RecoveryEnvelopeError, build_workflow_recovery_envelope, validate_workflow_recovery_envelope, workflow_definition_digest
from .execution import WorkflowExecutionError, WorkflowNodeCompleted, WorkflowNodeContext, WorkflowRunResult, run_workflow
from .schema import validate_skeleton_document
from .validation import diagnose_skeleton_draft, validate_skeleton_for_execution

__all__ = [
    "PreparedAgentPlan", "prepare_agent_graph", "execution_definition_digest",
    "DurableWorkflowRunner", "DurableWorkflowResult", "WorkflowNodeWaiting",
    "WorkflowExecutionStore", "SkeletonKernelHandler", "build_node_bundle",
    "KernelProfileBinding", "bind_kernel_profile",
    "SkeletonToolRuntime", "SkeletonSkillRuntime",
    "SkeletonCodeRuntime", "PythonEnvironmentProfile", "CodeSourceSnapshot", "FileCodeSourceResolver",
    "PublishedWorkflow", "SkeletonWorkflowCallRuntime",
    "attached_subagents", "validate_subagent_pool_config",
    "attached_hooks", "validate_hooks_pool_config", "PinnedHookWorkflow", "SkeletonHookRuntime",
    "apply_after_decision", "apply_before_decision",
    "SkeletonSubagentAgent", "SkeletonSubagentRuntime",
    "SkeletonValidationError",
    "SkeletonDiagnostic",
    "convert_legacy_definition", "materialize_legacy_agent",
    "PreparedFlowEdge",
    "PreparedWorkflowPlan",
    "prepare_workflow",
    "RecoveryEnvelopeError",
    "build_workflow_recovery_envelope",
    "validate_workflow_recovery_envelope",
    "workflow_definition_digest",
    "WorkflowExecutionError",
    "WorkflowNodeCompleted",
    "WorkflowNodeContext",
    "WorkflowRunResult",
    "run_workflow",
    "diagnose_skeleton_draft",
    "evaluate_condition",
    "map_outputs",
    "resolve_inputs",
    "select_case",
    "validate_skeleton_document",
    "validate_skeleton_for_execution",
]


# Keep the basic Workflow loader/executor provider- and persistence-free.
# Durable and model capabilities are loaded only when explicitly imported.
_LAZY_EXPORTS = {
    "PreparedAgentPlan": "agent_preparation", "prepare_agent_graph": "agent_preparation",
    "execution_definition_digest": "agent_preparation",
    "DurableWorkflowRunner": "durable_execution", "DurableWorkflowResult": "durable_execution",
    "WorkflowNodeWaiting": "durable_execution", "WorkflowExecutionStore": "durable_store",
    "SkeletonKernelHandler": "kernel_runtime", "build_node_bundle": "kernel_runtime",
    "KernelProfileBinding": "kernel_profile", "bind_kernel_profile": "kernel_profile",
    "SkeletonToolRuntime": "tool_runtime", "SkeletonSkillRuntime": "tool_runtime",
    "SkeletonCodeRuntime": "code_runtime", "PythonEnvironmentProfile": "code_runtime",
    "CodeSourceSnapshot": "code_runtime", "FileCodeSourceResolver": "code_runtime",
    "PublishedWorkflow": "workflow_runtime", "SkeletonWorkflowCallRuntime": "workflow_runtime",
    "attached_subagents": "subagent_config", "validate_subagent_pool_config": "subagent_config",
    "attached_hooks": "hooks_config", "validate_hooks_pool_config": "hooks_config",
    "PinnedHookWorkflow": "hook_runtime", "SkeletonHookRuntime": "hook_runtime",
    "apply_after_decision": "hooks_runtime", "apply_before_decision": "hooks_runtime",
    "SkeletonSubagentAgent": "subagent_runtime", "SkeletonSubagentRuntime": "subagent_runtime",
}


def __getattr__(name):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    from importlib import import_module
    value = getattr(import_module("." + module, __name__), name)
    globals()[name] = value
    return value
