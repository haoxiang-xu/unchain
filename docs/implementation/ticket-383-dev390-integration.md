# Integrate current dev / #390 into #383

Status: MERGE CANDIDATE CHECKPOINT, validation pending. 2026-10-03 UTC.

## Authorized scope

Merge current dev into this bugfix branch and push fast-forward checkpoints. Preserve shared history. Do not merge this PR into dev/main, force push, deploy, restart an owner profile, modify real databases, or call live paid providers.

Initial remote head: 73b11eb7db996fcd303555d448c4a145246f6760
Confirmed dev: 358b96d723daa0d2882158985c8245c7f8c7fb23
#390 is merged in PuPu PR395 and Unchain PR48. PuPu dev is f689b9fa9732fc4dc3ae527eea96e56c82dc1873; Unchain dev is 358b96d723daa0d2882158985c8245c7f8c7fb23.

## Integration plan

1. Read branch implementation/evidence and repository AGENTS and cross-boundary rules. Map three-way conflicts without discarding either intent.
2. Run complete graph impact before substantive edits. Resolve each semantic conflict using Sol, preserving the owner-accepted #390 behavior and #383 invariants.
3. Obtain independent GPT-6.1 Sol review of final candidate and tests. Repeat affected checks after corrections.
4. Publish small safe remote checkpoints and verify remote SHA/tree and restoration. Final merge must have this branch and confirmed dev as parents; no rebase.
5. Build an immutable Unchain artifact once, reuse its exact bytes for applicable PuPu contract and backend tests. Keep all five workflow default pins consistent per bugfix line.
6. Run applicable aggregate frontend, Electron, runtime/backend, boundary gates and build checks. Publish final tested state; monitor exact-head CI to terminal status or a verified blocker.

## Contract and state tracking

BC-390-I01: Unchain imported artifact to PuPu host/Electron/artifact consumers. VERSIONED; strict exported protocol manifest admission, provenance and digest fences remain unchanged. No Git SHA may become capability admission. AC-390-I01: validate the actual installed wheel, strict manifest positive/negative checks, Context V2 and RunBundle gates against the same immutable wheel.

BC-390-I02: historical diagnostics and leases across persisted state to runtime. CLOSED exact key sets, types, HTTP/null/status associations, canonical bytes, hashes and predecessor identities. AC-390-I02: historical diagnostic-v2/lease-v4 matrices plus unknown/hybrid/crossed state negatives; unknown outcomes cannot authorize resend.

SEQ-390-I01: first/second message, first/second interaction, retry/durable resume/cold restart and normal/graph/subagent paths as applicable to inherited #383/#384 plans. Preserve approval gating, no-resend and Stop behavior. AC-390-I03: combined artifact-pair regression suites and focused policy/interrupted chain tests. Inapplicable cells require a reason; unrun cells stay NOT_RUN.

## Pending evidence

### Three-way conflict and semantic map

- Parents: plan checkpoint `0dcbf9c92a58f40de7c39d84fe35a40ff91fa8fc` and confirmed dev `358b96d723daa0d2882158985c8245c7f8c7fb23`. Merge base: `1ec49ddfc28d3b42ba035debada5e3db759dad1b`.
- The automatic three-way merge is text-clean, with no unresolved entries. No incoming source or test was discarded and no new production behavior was invented.
- Shared file overlap is `events/normalizer.py`: #383 tool/interaction presentation policy coexists with #390 closed Gemini retry-ordinal projection. Presentation metadata remains outside the durable interaction request and provider schema.
- Context replay semantic overlap: #383 recovers historic tool policy or exact legacy omission; #390 verifies the current native batch against durable provider result and wire snapshot. Verification compares call identity/name/arguments and preserves existing receipt/hash/no-resend gates.
- Historical diagnostics-v2 and leases-v4 compatibility is imported byte-for-byte from merged dev, including exact key sets, HTTP/null and lease status distinctions. Unknown outcomes do not authorize resend.
- Relative to dev, the production/test diff is exactly the existing #383 policy work, plus this integration record. macOS-only unpublished `fadde9f3` is unavailable and is not reconstructed.

### Graph preflight

GitNexus 1.6.12 indexed the premerge checkpoint in an isolated index (21,528 nodes, 50,706 edges, 818 sampled flows). Before checkout edits, upstream impact was run for all 60 incoming changed/added/removed production callables. Fourteen new callables have no premerge node; their existing caller boundaries were included. The refreshed merged index contains 21,834 nodes and 51,449 edges. Complete staged detect-changes succeeded: 46 changed files, 388 symbols and 19 affected sampled execution flows, aggregate risk CRITICAL; the structured result has no error or partial flag.

The aggregate semantic risk is CRITICAL. Journal ephemeral classification reaches three direct append/project callers and six sampled durable flows (result persistence, tool authorization, subagent result and sealed completion). HIGH graph findings also include wire-authority recovery, Anthropic native translation, historical lease parsing and the replaced retry delay. The integration therefore requires the inherited negative matrices, full runtime suite and exact installed wheel tests, not merely a conflict-free merge.

Indexing succeeded with a 512 MiB DB pool, one parser worker and FTS disabled after the default analyzer exited early. Static Python dynamic/callable-value edges and process sampling are incomplete; the graph does not prove absent runtime paths safe.

Final artifact identity, test results, independent review, remote restore proof and exact-head CI remain PENDING. Prior branch acceptance is historical, not acceptance of this merged pair. Live/frozen/platform-specific acceptance is NOT_RUN until demonstrated.
