# Integrate current dev / #390 into #383

Status: RUNTIME QUALIFIED; host-pair gates and exact-head CI tracked separately. 2026-10-03 UTC.

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

## Integration evidence

### Three-way conflict and semantic map

- Parents: plan checkpoint `0dcbf9c92a58f40de7c39d84fe35a40ff91fa8fc` and confirmed dev `358b96d723daa0d2882158985c8245c7f8c7fb23`. Merge base: `1ec49ddfc28d3b42ba035debada5e3db759dad1b`.
- The automatic three-way merge is text-clean, with no unresolved entries. No incoming source or test was discarded and no new production behavior was invented.
- Shared file overlap is `events/normalizer.py`: #383 tool/interaction presentation policy coexists with #390 closed Gemini retry-ordinal projection. Presentation metadata remains outside the durable interaction request and provider schema.
- Context replay semantic overlap: #383 recovers historic tool policy or exact legacy omission; #390 verifies the current native batch against durable provider result and wire snapshot. Verification compares call identity/name/arguments and preserves existing receipt/hash/no-resend gates.
- Historical diagnostics-v2 and leases-v4 compatibility is imported byte-for-byte from merged dev, including exact key sets, HTTP/null and lease status distinctions. Unknown outcomes do not authorize resend.
- Relative to dev, the production/test diff is the existing #383 policy work plus the bounded replay correction recorded below. macOS-only unpublished `fadde9f3` is unavailable and is not reconstructed; merged dev already contains its specific 13-code closed legacy RPC diagnostic behavior.

### Graph preflight

GitNexus 1.6.12 indexed the premerge checkpoint in an isolated index (21,528 nodes, 50,706 edges, 818 sampled flows). Before checkout edits, upstream impact was run for all 60 incoming changed/added/removed production callables. Fourteen new callables have no premerge node; their existing caller boundaries were included. The refreshed merged index contains 21,834 nodes and 51,449 edges. Complete staged detect-changes succeeded: 46 changed files, 388 symbols and 19 affected sampled execution flows, aggregate risk CRITICAL; the structured result has no error or partial flag.

The aggregate semantic risk is CRITICAL. Journal ephemeral classification reaches three direct append/project callers and six sampled durable flows (result persistence, tool authorization, subagent result and sealed completion). HIGH graph findings also include wire-authority recovery, Anthropic native translation, historical lease parsing and the replaced retry delay. The integration therefore requires the inherited negative matrices, full runtime suite and exact installed wheel tests, not merely a conflict-free merge.

Indexing succeeded with a 512 MiB DB pool, one parser worker and FTS disabled after the default analyzer exited early. Static Python dynamic/callable-value edges and process sampling are incomplete; the graph does not prove absent runtime paths safe.

Final corrected runtime artifact identity, test results and independent re-review are recorded below. Both remote checkpoints were restored and verified independently. Host-pair and exact-head CI evidence is tracked by the paired PuPu report. Live/frozen/platform-specific acceptance is NOT_RUN until demonstrated.


## Independent review correction: recursively frozen array arguments

Independent Sol review found an inherited #383 compatibility defect, not a #390 regression. `ContextToolAuthorityHarness` compared frozen journal arguments directly with live JSON arguments. Journal arrays are recursively stored as tuples, so a logically identical array call missed original policy recovery. A toolkit policy change then caused a journal operation conflict on resume.

RED on merge checkpoint `4a0add1936e0369e16d5286289b280456eb913ca`: the original omission and all four declared policies were tested against scalar, array, nested-array and object-with-array arguments. All 15 array-bearing cases failed; the five scalar controls passed. Resumed arguments exactly matched the original live call, so these failures were not changed execution subjects.

The bounded correction applies the existing journal `_thaw_json` helper only to the stored arguments used for presentation-policy comparison. Its existing local import was moved to module scope; the journal models module imports only standard-library dependencies and creates no context/tool/provider import cycle. No durable intent, interaction request, approval subject, schema, digest, lease or no-resend gate is modified.

GREEN: 73 focused tests pass. The 20-state successful matrix checks one invocation, unchanged original journal record and unchanged full durable request/digest reloaded from the interaction store after resume. Five changed-array subjects remain rejected before invocation. Existing unknown-call, wrong-runtime, different-context/attempt, forged receipt and one-shot permit negatives remain intact and pass.

Pre-edit upstream impact of `build_delta` reports UNKNOWN/lower-bound because five dynamic receiver call sites are unresolved; direct dispatch is manually traced through `BaseRuntimeHarness.__call__`. Test helper impact is HIGH (18 direct callers) and MEDIUM (11 direct callers); default scalar behavior is preserved. Complete staged graph detection succeeded for the correction: 3 files, 13 symbols and 0 sampled execution flows; risk LOW. The full structured result contains no error or partial flag.

### Historical artifact and infrastructure evidence

The superseded merge-only wheel was built once from clean `4a0add1` source: SHA-256 `d1735fac3affe1a00e402e78c85d110f7d939193c2ea014ebe747c428e248e54`. Installed focused qualification passed 198 tests. Old full source/installed runs reported 4,215/4,213 passes with one/three setup failures; those aggregate runs are not recorded as PASS. All three distinct setup failures subsequently pass unchanged: a clean `/dev/shm` pytest temporary root avoids sandbox-injected ancestor `.git` markers, and root-anchored archive extraction preserves nested eval fixture `src` directories while keeping the runtime package imported from the actual wheel. Missing SOCKS support during Gemini SDK initialization was resolved by installing HTTPX's optional `socksio`; all six real-SDK mock 503 tests passed thereafter. No live provider request was made.

The corrected immutable source and once-built final wheel are qualified below. Host consumers must reuse those exact bytes and pin that source revision. Live-provider, actual owner-profile restart, actual macOS database recovery and owner acceptance remain NOT_RUN; Electron and host-pair gates are recorded separately, never inferred from this runtime suite.

The 12 excluded live-provider cases are the streamed-text, structured-output, core-read-roundtrip and repeated-long-context-cache scenarios for Gemini, OpenAI and Anthropic. They remain NOT_RUN to avoid real network/provider effects.


## Final immutable runtime qualification

- Confirmed dev parent: `358b96d723daa0d2882158985c8245c7f8c7fb23`.
- Three-way merge checkpoint: `4a0add1936e0369e16d5286289b280456eb913ca`, parents `0dcbf9c92a58f40de7c39d84fe35a40ff91fa8fc` and confirmed dev; tree `cc13548266726f53502075196fafaf9797ff4183`.
- Final source checkpoint and required #383 host pin: `a7fa15d685b1130bf5678c1e493a51d2551185cf`, parent merge checkpoint; tree `e9fcc714c0756a8db67b4b9811b2829161e3bc30`. Publication and independent bare remote restoration verified both trees/parents. Subsequent evidence-only commits do not change this artifact provenance or host pin. #384 continues to use clean dev `358b96d` without #383 policy work.
- Artifact: `unchain-0.2.0-py3-none-any.whl`, built once from a clean git archive of final source. Wheel SHA-256 `a88028bbb7d286d4792e49b5945f80e33d71203579298ddb053190cca1aced62`; source archive SHA-256 `3366f0216d54a10944a2dad385f1f130ca9ef479f9280a9551a8cf577871e992`.
- Imported manifest digest: `sha256:b80cde70e35f93c21ed069990f26817d353e4fd1c9d3ff109cfda933f5295672`. Capability admission still depends on strict imported protocol shape/features/digest, not the Git pin.
- Byte verification: all 346 package entries match clean immutable source and installed wheel; all 351 immutable distribution entries match installation. Pip's rewritten `RECORD` is excluded from byte equality. The same unchanged wheel is supplied to host consumers.

### Terminal test evidence

- Corrected focused policy/approval/permit regression: 73 passed, including 20 original-policy/omission and argument-shape cases plus five changed-array rejection cases.
- Full source runtime suite: 4,239 passed, 3 skipped, 12 live-provider cases deselected, 5 existing expected failures; zero failures, 117.16 seconds.
- Full installed-wheel runtime suite: identical counts, zero failures, 126.70 seconds. A separate immutable test mirror retains nested eval fixtures and points only its top-level `src` to the actual installed wheel, including cold subprocess imports.
- Independent Sol final review: 237 critical source tests and 355 critical installed-wheel tests pass; 346 packaged files match immutable source/install; all five independently reproduced nested-array policy states execute once. No other source defect found in the combined #390/#383 paths.
- Closed historical v1 RPC smoke: all 13 merged-dev codes round-trip as exact four-key v1 diagnostics. `UNAVAILABLE`/HTTP 503 survives a real temporary SQLite reopen with exact canonical lease bytes and SHA-256 `60a9138df7b0006adcb001b69c60de24716be28b32b572901002420a5e6adcb8`. Seven unknown/hybrid diagnostic and three crossed lease cases are rejected; a terminal historical lease cannot authorize resend. The existing diagnostics-v2/leases-v4 matrices remain part of both full suites. This synthetic fixture is not the inaccessible owner's macOS database record.
- Clean-process import, compileall and whitespace checks pass. The final wheel digest is unchanged after every qualification run.

### Explicit remaining coverage limits

The three skips are Windows named-mutex acceptance, Windows destination-handle replacement and the opt-in live MCP stdio smoke. The five inherited strict XFAILs are four workspace virtual-path guard cases and the long-term recall/current-user ordering case. They are retained, not relabeled as passes. Twelve credentialed provider scenarios are deselected as listed above. Two SDK warnings arise from synthetic finish-reason negatives.

The final runtime source and installed artifact are qualified. Host-pair/frontend/Electron/CI acceptance, frozen desktop packaging and deployed/real-profile behavior require their own evidence. No profile restart, real database modification, live paid provider call, deployment, force push or merge of PR47 into dev/main was performed.

Independent review deliverable: [scoped Sol runtime review](ticket-383-dev390-independent-review.md). Structured identities and terminal evidence: [runtime qualification record](ticket-383-dev390-runtime-evidence.json). Standalone runtime CI is NOT_RUN: `.github/workflows/ci.yml` accepts pull requests only with base `dev`/`main` and pushes only to `dev`/`main`, with no `workflow_dispatch`. PR47 still targets `codex/ticket-386-gemini-5xx-catalog`; GitHub returned zero runs for both source `a7fa15d` and merge `4a0add1`. The base retarget was outside the authorized scope and is deferred; no retry or base mutation is performed. The old stacked PR view also includes already-merged #390; source review compares current dev to the candidate. Host QA independently tests pinned `a7fa15d`, but it does not substitute for standalone runtime CI proof.
