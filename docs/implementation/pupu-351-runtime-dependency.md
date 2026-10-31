# Skeleton runtime dependency for PuPu #351

This branch integrates the previously reviewed local Skeleton foundation
`09eec580` with remote dev `d2aa98f5`, solely to make PuPu #351's delivery
dependency recoverable and reviewable. It does not close PuPu #356 or authorize
new features, live providers or release rollout.

The frozen source/checkpoints remain unchanged. All edits are in the independent
`unchain-351` clone. Current dependency security floors are retained and
`jsonschema>=4.22,<5` is added for the existing Skeleton validation contract.

## Merge decisions

- Keep both current explicit history compression and the durable Skeleton
  compression service. They were independently added under the same module
  name and have different private helpers. Current `context/compression.py`
  retains its complete existing implementation. The frozen durable service is
  copied byte-for-byte into `context/durable_compression.py` and re-exported at
  its original public import path. No public symbol or receipt shape is renamed.
- Retain current post-terminal memory-audit allowances and the Skeleton
  open/quiescent-attempt checks together during generation rebase.
- Retain both subagent completion seals and Hook raw-result seals in the
  Journal, with the existing distinction between raw and final tool results.
- Preserve current provider/dependency security upgrades; do not replace remote
  dev with the old frozen tree.

## Boundary, sequence and evidence

BC-351-D01 and SEQ-351-D01 are shared with PuPu's
`docs/implementation/ticket-351-delivery-dependencies.md`. Existing CLOSED
manifest, Journal, checkpoint and tool-execution envelopes remain authoritative.
The actual imported runtime manifest, not this revision, controls compatibility.

Source regression: 443 tests pass across 13 explicitly named files covering both
compression APIs, generation rebase, strict branch/schema behavior, execution
and recovery. A source-origin receipt confirms all imported Core modules came
from this delivery clone. This is not installed-wheel, desktop or provider proof.

Pre-edit graph: 117 file/symbol targets; one CRITICAL file and eight HIGH symbols
disclosed, UNKNOWNs source-corroborated. Graph process discovery has caps and
is not exhaustive. Change analysis must be run before committing.

Build one new wheel after source integration, reuse it for applicable PuPu
consumer checks, and record its hash/manifest/source equality in #351 and the
dependency PR. Source tests using the available offline test environment do not
waive the newer dependency floors. Missing offline packages are a qualification
limit, not permission to lower those floors or fetch packages.

## Full-CI integration follow-up

The first collected full run on `6a1cb594` finished with 5514 passed, five
failed, 17 skipped and five xfailed. The failures are retained, not waived.

- The Skeleton addition changed bundle identity lookup to the execution factory.
  A runtime without a factory must raise the existing bundle-unavailable domain
  error before dereferencing it. The authority harness then reaches its ordinary
  permission gate; real factory-based identity resolution remains unchanged.
- The real-venv Code test now creates its own standard-library venv without pip
  and verifies that the worker actually runs inside it. CI's base interpreter is
  no longer assumed to be a venv.
- The compression check compares original message bodies with the final wire;
  the words "historical evidence" also legitimately occur in current guidance.
- The independent protocol baseline is recaptured from remote dev `d2aa98f5`'s
  actual module. It preserves all newer context features instead of freezing the
  older local checkpoint's feature set.

The unchanged failing authority tests and updated portable/compatibility checks
pass 18 focused cases. A broader 12-file source run passes 200 tests; the two
macOS ps-dependent worker-reaping cases remain BLOCKED by the offline OS policy,
and the optional bundled-interpreter case is skipped. The two worker cases passed
in the preceding Linux full CI and remain in the fresh full-CI run. No test is
deleted, weakened into a skip, or counted passing when blocked.

Pre-edit impact for the bundle method is HIGH: 21 upstream impacts, 15 direct
callers, including compilation, graph binding and tool-authority paths. This is
reported without waiver. Raw evidence is kept separately under
`~/.codex/artifacts/pupu-351/ci-02-20261030/`; the prior stage seal is unchanged.
Because runtime source changed, build and reuse a new wheel for the next exact
Host pairing; do not reuse the preceding wheel's qualification.
