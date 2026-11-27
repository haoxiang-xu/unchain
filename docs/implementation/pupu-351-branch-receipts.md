# #351 durable branch selection receipts

The #351 close audit found that selection was durable but the untaken outlets
were not explicitly recorded as skipped. This follow-up preserves the existing
exclusive traversal and adds no node-skipping propagation.

## BC-351-B01 — branch evaluation to its Journal-backed raw result

Producer: `DurableWorkflowRunner.step` after successful strict condition/case
evaluation. Consumer: the existing atomic execution store and read-only trace
readers. Representation: canonical JSON in the branch node's existing
`result_ref`, committed in the same state revision as the selected `edge_id`.
No HTTP envelope, execution-state schema, published node output or endpoint
changes. The runtime manifest stays unchanged because this is additional
internal result content in an already opaque result artifact slot.

New receipt schema `unchain.skeleton.branch_selection.v1` has exactly
`schema`, `node_id`, `selected_edge_id`, `skipped_edges`; each skipped entry has
exactly `edge_id`, `status` (always `skipped`). Only other flow outlets of the
same successfully evaluated branch are listed, in the fixed plan's edge order.
No inferred node status is stored. Errors before selection write no receipt.
Older raw results (including `{}`) remain readable; recovery follows the already
committed edge and never depends on this additive observation. Unknown future
receipt shapes must not be interpreted as this version by a reader.

## SEQ-351-B01 and acceptance

Identity: execution/generation/graph attempt, node invocation, and immutable plan.
Create -> successful branch commit -> reconstruct store -> finish/replay.
AC-351-B01: true/false and case/default have the exact selected edge and only
untaken sibling edges; the rejoin/End still executes once. The referenced result
artifact must be in the actual Journal commit's resource set. Cold reconstruction
keeps its bytes, and completed replay appends nothing. Tests must first fail on
the old empty raw result. A failing comparison must not create a selection
receipt. Reuse one newly built wheel for Host acceptance after this source change;
old source-only evidence is not qualification of that new pair. No provider or
API call is needed for these deterministic branch tests.
