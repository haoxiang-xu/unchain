# Ticket 63 — public context compaction

Ticket: https://github.com/haoxiang-xu/unchain/issues/63
Branch: codex/ticket-63-compaction-modes
Base: dev 654017b793513df6cd9687a4096913bf1344e561
Workspace: /Users/red/Desktop/GITRepo/unchain-63

## Scope and functional flow

Deliver the public Python trim/summary API and the existing compiler's suffix-search
optimization. Full canonical input -> pressure detection -> legal completed prefix ->
optional isolated summary inference -> strict output/citation validation -> durable
original archive -> budget-valid reduced view -> scoped original evidence recovery.
Keep the original workspaces untouched. PuPu #457 UI and refinement are HOLD by the
owner's latest direction; no PuPu implementation, default selection, UI, rollout or
packaged-pair acceptance is included here.

The previous prototype is unaccepted input, preserved as a patch under the external
artifact directory. The prior real-model trials are retained as failures. Use the
repository's own .venv/pytest. The parent implements because evidence authority,
format admission and chronological recovery require strong judgment. A bounded
independent TDD agent created only regression tests: 20 failed, 4 passed against the
untouched prototype. No worker is authorized to make production or merge decisions.

## Decisions and impact

- Two explicit library modes: trim does not call a model; summary clones the supplied
  Agent with no modules/tools and explicit handoff instructions.
- The caller retains canonical history. Never summarize a previous reduced view.
  Reject known legacy tool-preview envelopes before even a no-pressure return;
  require rehydration from original journal/artifacts. Arbitrary upstream data
  loss cannot be detected or repaired by this API.
- Use short model-only aliases s0, s1, ...; the host maps exact valid aliases back to
  immutable canonical IDs. No fuzzy correction of references.
- Accept one enclosing JSON fence, but no prose, multiple objects, duplicate keys,
  unknown fields/schema or unknown citations.
- Distinguish user instructions, tool observations, assistant claims and approval
  outcomes. Do not infer new permission requirements from log relevance labels or
  assistant plans; the latest live user message remains authoritative.
  Mixed provider messages have separate block authority labels. Built-in summary
  payload accepts generation settings only; a custom callback owns other controls.
- Original per-exchange retrieval accepts either call_id, or tool_name + zero-based
  chronological occurrence. It returns exact provider call/result items and their
  enclosing source IDs, without forcing an Agent to scan unrelated large results.
- Retain existing checkpoint authority; only binary-search monotone suffix cost and
  revalidate actual chronological final candidates with the original estimator.

Index: unchain-63 at base 654017b. _reduce upstream impact LOW: direct _compile_core;
upstream compile/_compile_for_coordinator and coordinator._compile_pass. New APIs
return UNKNOWN because absent; text search confirmed no existing uses. Index flow
inventory is capped and cannot prove absent paths unused. Renew impact before any
additional existing-symbol changes; complete graph change detection before commits.
The initial incremental index incorrectly inflated SummaryRequest impact to CRITICAL
(warned before editing). A full rebuild with --no-parse-cache and the same walk depth
resolved it to LOW; source-kind/to_dict have direct summary-consumer callers. Public
compress/read/constructor dynamic entrypoints remain UNKNOWN; text checks find only
the new tests/docs, so they retain full regression and independent-review coverage.

## Boundary contracts

**BC-001 — model summary.** Producer SummaryRequest from a frozen prefix -> isolated
Agent/provider -> strict host validator. CLOSED summary v1 keys: schema, goal,
constraints, decisions, progress, open_items, next_steps; each non-schema value is an
array of {text, source_ids}. Model sources use sN and host-provided source authority;
archive IDs stay content/position-bound. A canonical-ID callback remains supported.
Only exact known refs are admitted. Unknown keys/schema/refs, ambiguous outer text,
empty goal and oversize output fail before a reduced view is returned. Policy identity
binds prompt and summarizer key; no silent fallback to trim. AC-001/002.

**BC-002 — persistence and reader.** Compressor -> archive store -> cold reader,
CLOSED archive v1 with schema, archive_id, policy_sha256, source_sha256, source_ids,
messages, tool_chain, summary, content_sha256. Original JSON values and digests must
match. Directory storage atomically installs content, refuses conflicting/tampered
receipts and caches an identical source/policy operation. The bound reader cannot
leave this archive; source_id is mutually exclusive with exchange selectors.
call_id is exclusive with tool_name/nonzero occurrence. Name matches are ordered by
original call appearance; unknown, negative, boolean, noninteger or out-of-range
selectors fail. Read-page v1 exact keys remain schema, archive_id, source_id, offset,
total_chars, next_offset, sha256, text, message; Unicode offsets and UTF-8 digest.
A selected exchange payload is CLOSED {call_id,name,call,result,call_source_ids,
result_source_ids}. The call/result are exact provider-level raw items. AC-003/004.

**BC-003 — compiler.** Suffix statistics -> existing ContextCompiler._reduce ->
existing coordinator/checkpoint consumer. No durable V2/provider wire/manifest
schema change is needed for this standalone API; no automatic runtime activation.
Preserve pinned input, legal exchange boundaries, exact checkpoint ref pricing and
final-candidate checks; statistics combine before rounding. AC-005.

## State sequences and acceptance mapping

**SEQ-001:** identity source digest + policy digest. Full canonical history -> pressure
-> legal prefix -> isolated inference -> validate -> archive installation -> view ->
read -> repeat/cold reopen. Identical operation replays without another model call;
changed policy creates a different receipt. Failed validation/storage cannot return a
destructive view. Repeated task tools are an Agent task error, not archive replay.

**SEQ-002:** pending tool/approval exchange under pressure -> summary defers with zero
model calls/writes -> response or approval denial arrives -> safe compression can run.
Cross-turn/parallel exchange boundaries and current user turn remain atomic.

- AC-001: public mode selection, no-pressure no-op and trim zero-model behavior;
  narrow fence compatibility and strict malformed-output negatives.
- AC-002: actual tool-free Agent -> short aliases -> canonical citations; long/near
  aliases rejected, source kinds and grounding reviewed in real-model artifacts.
- AC-003: exact original messages/digests/links, per-call/per-name chronological reads
  for Anthropic/OpenAI Chat/Responses, Unicode pages, scope and selector negatives.
- AC-004: pending calls/approvals, identical replay and cold reopen, tampering/storage
  failure, policy isolation and no recursive summary-of-summary.
- AC-005: suffix estimator equality plus existing Context V2 golden/SQLite/checkpoint
  tests and differential/performance evidence for the same final output behavior.
- AC-006: fixed wheel + real model, two controlled cases x three consecutive summary
  calls, owner corrections, actual task artifacts and independent task checks; exact
  earliest code/error/hash/call recovery. Preserve all failed attempts. Independently
  check canonical bytes/links and semantic grounding, not only task success.

Library calls, repeated calls, pending exchanges and cold reopen are applicable.
PuPu first/second chat, graph/subagent/interaction/sidecar restart and deployment
artifact-pair paths are NOT_RUN and explicitly deferred to held #457; this library
change does not activate a new runtime owner or deploy a PuPu candidate. Tests of the
existing compiler paths remain required for regression coverage.

## Verification and delivery

Independent checkpoint review found mixed-message authority promotion and native
payload instruction override. Twelve new negatives failed against that candidate;
both were fixed and the focused suite now passes 71 tests. The first full suite
passed 4,724 tests (16 skipped, 5 xfailed); run it again after these checkpoint fixes.

Run focused tests, then the complete library suite; build a single wheel and retain
its SHA256 and imported module path for live regression evidence. Review all summary
responses and actual saved artifacts. Archive evidence outside the clone and remotely
before cleanup. The owner explicitly authorized PR/merge to dev and closure after
verification. Do not bypass branch protection. Do not close PuPu #457 or its Release.

Final evidence: `tests/evals/fixtures/compaction-63/README.md`, including immutable
wheel 6088d92b59bbd9f7a4d26ea55b3ea86b7a462771270cabe71b9444c5c285206f,
4,737 full-suite passes, 71 focused passes, two cases x three real summaries,
six independently checked original archives, completed tasks and exact original
code/error/hash/call-ID recovery. Semantic review passes with documented minor
qualifications, not perfectly faithful prose. Earlier failures remain preserved.
GitNexus raw full output (rather than the CLI's 15-symbol display) was inspected:
142/142 changed symbols, all 9 processes, no partial/truncated/error signals.
Aggregate HIGH risk was reported before further editing; renew after evidence files.
