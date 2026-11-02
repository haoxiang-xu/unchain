# PuPu #356: deliver the frozen child-settlement producer

This independent branch starts at dev e0d2eb5e. It applies only the five-path
25918480..2bf67b21 cold-child producer delta, leaving the original checkpoint,
its tests, artifacts, H-4 fixes and the later denial/Hook changes untouched.

BC-356-CLOSE-002 is VERSIONED/CLOSED: the canonical, call-bound Journal receipt
is the authority for `unchain.skeleton_subagent_settlement_projection.v1`.
Typed consumer/source attempts, generation, call/tool, cursor and bounded
artifact bytes must match. The projection is display evidence, never a tool
result or authority to resume, dispatch or fetch content. The original waiting
tool result is not replaced. Unknown fields and contradictory content fail.

SEQ-356-CLOSE-002: verify the committed child completion and call settlement,
then emit one `subagent_tool_settled` callback; normalize it to an existing V4
child terminal envelope with the strict projection. No model/tool/Code effect
is repeated. A root terminal alone is insufficient evidence for a child card.

AC-356-CLOSE-002: exact terminal kind and identity, all malformed controls,
large-output descriptor, real Host child reconstruction, cancelled-child Stop
producer, renderer persistence and current approval regression. Keep source,
installed-wheel, binary HTTP, packaged App and real-service results separate.
Live provider verification remains Release #208's explicitly accepted boundary.

The existing wheel 17b04aea lacks this module and is prior evidence only for
this path. Build and verify one replacement wheel and the corresponding Host
sidecar before deployment/acceptance. Do not relabel the old pair as passing.
Evidence is recorded in PuPu #356 and its direct close implementation plan.
