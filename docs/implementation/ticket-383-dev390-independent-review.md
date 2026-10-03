# Independent Sol review: #383 runtime integration

Verdict: **PASS for deterministic runtime integration**, 2026-10-03 UTC.

## Reviewed immutable identity

- Source: `a7fa15d685b1130bf5678c1e493a51d2551185cf`
- Source tree: `e9fcc714c0756a8db67b4b9811b2829161e3bc30`
- Wheel: `unchain-0.2.0-py3-none-any.whl`
- Wheel SHA-256: `a88028bbb7d286d4792e49b5945f80e33d71203579298ddb053190cca1aced62`
- Imported manifest: `sha256:b80cde70e35f93c21ed069990f26817d353e4fd1c9d3ff109cfda933f5295672`

All 346 packaged files, including 335 Python files, were independently compared with archived pinned source and the installed distribution. There were zero differences. The imported installed manifest matched the recorded complete manifest.

## Blocking finding and correction

The original merge `4a0add1` and wheel `d1735fac…248e54` were rejected. The #383 presentation-policy recovery compared frozen journal arguments directly with live JSON arguments. Recursively frozen arrays are tuples, while live arrays are lists. This caused an exact replay call to miss its historical policy/omission and re-emit a different live policy, triggering an operation conflict before tool execution.

The defect was independently reproduced against both immutable original source and its installed wheel for original omission, never, no_feedback, approved and always, with zero effects. Resumed live arguments matched the original live call exactly.

The correction thaws stored JSON only for presentation-policy comparison. No durable intent, approval request, execution subject, hash, schema or no-resend fence changes. The existing fallback helper import moves to module scope; its standard-library-only dependency creates no circular context import. The final 20 positive policy/shape cases and five changed-argument negatives check that the journal record and the durable request reloaded after resume remain unchanged. Changed arguments still fail before any effect.

## Independent checks

- Twelve critical source suites: 237 passed
- Seventeen critical installed-wheel suites: 355 passed; installed import binding asserted before collection
- Nested object/array/bool/null/integer probe: all five historical policy/omission states pass against source and wheel, exactly one effect each
- Temporary SQLite historical v1/HTTP503/UNAVAILABLE diagnostic and v3 failure lease: exact canonical bytes/hash after reopen; six schema/type/hybrid negatives rejected
- Both full-suite terminal logs independently inspected: 4,239 passed, 3 skipped, 12 live-provider cases deselected, 5 inherited strict XFAILs, zero failures for each
- Additional installed historical smoke: 13 RPC v1 round-trips, seven diagnostic negatives, three lease negatives, SQLite byte/hash preservation and terminal no-resend

## Reviewed boundaries

No further substantive defect was found in policy/replay authorization, closed historical diagnostics-v2/leases-v4, native result/text projection, provider-wire provenance, HTTP200/SSE/STARTED uncertainty and no-resend, bounded Gemini retry, interruptible wait or Stop paths. Presentation policy remains outside provider schemas and durable approval requests. Native verification remains bound to durable identities, names and arguments.

## Coverage limits

This is scoped runtime acceptance. It does not establish exact host-pair acceptance, native Windows behavior, frozen desktop packaging, paid live-provider acceptance, actual macOS database/profile recovery, owner acceptance or remote CI. The two Windows checks and opt-in live MCP are skipped; twelve live-provider scenarios are deselected; four virtual-path and one recall-ordering strict XFAILs are retained. The original rejected artifact is not accepted retroactively.

Reproduction commands, environment versions, terminal counts and artifact/log digests are recorded in the companion integration report and `ticket-383-dev390-runtime-evidence.json`. The immutable source commit contains the complete regression fixtures.
