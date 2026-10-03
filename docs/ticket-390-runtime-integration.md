# #386 / #390 runtime compatibility

PuPu PR #395 combines the host changes from #386 and #390. A wheel containing
only either runtime branch is insufficient. The release artifact must include
the native result-reader and uncertainty capabilities from #390, plus the
interruptible retry wait and extended HTTP diagnostics from #386.

## Durable format compatibility (BC-390-15, AC-CI-01)

Both branches assigned existing version identifiers to different closed shapes.
Compatibility is decided by the complete key set, HTTP/null type, and lease
status; unknown fields, crossed statuses and mixed diagnostics are rejected.
Existing values are neither renumbered nor rewritten on read, so canonical bytes,
lease SHA-256 and predecessor identities remain unchanged.

| Stored shape | Meaning |
| --- | --- |
| Diagnostic v1, four keys, integer HTTP status | Original HTTP failure |
| Diagnostic v2, four keys, null HTTP status | Explicit Gemini response outcome from #390 |
| Diagnostic v2, six keys, integer HTTP status | Extended HTTP failure from #386; includes provider status and catalog-validated replacement model |
| Lease v3, FAILED, failure diagnostic | Original HTTP or explicit response failure |
| Lease v4, FAILED, six-key HTTP failure diagnostic | Extended HTTP failure from #386 |
| Lease v4, STARTED, uncertainty diagnostic | Unknown outcome from #390; never authorizes another send |

Real SQLite fixtures cover both prior branches, exact bytes and hashes after
reopen, and malformed/hybrid negatives in
`tests/context_v2/test_provider_diagnostic_branch_compatibility.py`.

## Retry and error projection (SEQ-390-13, AC-CI-02)

The normal kernel installs one retry-wait hook. It emits a start and subsequent
heartbeats, remains interruptible, and stops before claiming or sending the next
request. Direct execution-service callers without a hook retain #390's ordinal
notice for compatibility. The normalizer explicitly distinguishes these formats;
malformed ordinal markers cannot fall through into the countdown projection.

Gemini 503 requires an actual rejected HTTP response, with matching SDK error
code. An error inside HTTP 200 or without enclosing HTTP evidence is uncertain
and is never replayed. Gemini 503 remains capped at two retries; other retry-safe
failures retain their configured budget. Cold recovery of older #386 retry
records applies the cap to new sends while preserving already recorded outcomes.

Explicit response failure outcomes remain terminal and have no invented HTTP
status. Uncertainty is projected through closed reason/phase diagnostics; raw
provider text and arbitrary exception class names are excluded. The legacy safe
HTTP diagnostic constructor remains supported, but free-text uncertainty details
are rejected. Tests inherited from #386 use these stricter #390 observations,
retaining send counts, cancellation, persistence and no-resend assertions.

## Exact artifact qualification (AC-CI-03)

Build the reconciled clean source once, then reuse the same wheel for runtime,
sidecar and boundary-contract tests. Runtime admission remains governed by the
strict imported manifest, not Git revision. The revision is immutable release
provenance only. PuPu's QA defaults must point to this combined runtime, rather
than an unmerged sibling revision. Detailed immutable artifact identities and
full-suite results are recorded in PuPu's #390 CI acceptance report.
