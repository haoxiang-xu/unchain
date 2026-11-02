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

The loaded producer declares `subagent_tool_settlement_projection_v1` in
Skeleton protocol v1.5. The Host independently requires that manifest feature
and minor; neither a module-presence check nor a source/Git allowlist substitutes
for compatibility admission. Other protocols retain their exact wire values.


BC-356-CLOSE-004 / SEQ-356-CLOSE-004 extends original M3/AC007/008/011/014:
Kernel profile gains optional closed auth_mode, bound into immutable definition
and profile digests. Protocol-specific modes: Responses=bearer;
Anthropic=x-api-key|bearer; Gemini=api-key; Ollama=none|bearer. Known native
brands retain their existing mode; absence preserves legacy profiles verbatim.
Unsupported header/custom authentication fails explicitly. Authenticated SDK
construction pins one explicit Anthropic credential argument and explicitly omits
the unselected HTTP header. Endpoints and max_retries remain pinned. Real SDK
HTTPX no-socket tests cover both modes, hostile environment auth and byte-exact
header identity. Minor6 advertises kernel_profile_auth_v1; old minors remain
valid for their old contracts but do not authorize the updated Host.

91745f90 full CI found one stale Hooks test asserting minor==4 (5664 passed,
1 failed,17 skipped,5 xfailed). Keep that failure. That compatibility test now
requires minor>=4 plus its exact features; the dedicated protocol test still
asserts the complete current minor6 manifest exactly. No runtime contract was
weakened to satisfy an obsolete version pin.


The a95444c full CI found 3 real-SDK authentication failures (5678 passed,
17 skipped,5 xfailed). Its newer SDK rejects simultaneously passing api_key
and auth_token even when one is empty. Correct by selecting exactly one non-null
argument and explicitly omitting the alternate HTTP header. Older SDKs may read
an alternate ambient credential into the SDK object, but Omit prevents that
header from reaching the wire; the tests poison both environment variables.
Do not claim older SDKs never read the alternate environment variable. CI now
runs these bounded protocol/auth checks before the unchanged full suite.

Process exception: the existing auth test gained its explicit single-credential
assertion before a dedicated impact query on that test symbol. The production
factory had pre-edit impact LOW. Record the test-only sequencing error openly;
subsequent impact/text corroboration is not a retroactive pre-edit receipt.

157ab8e CI failed its fast check: 3 SDK tests injected httpx.Client into
anthropic2.4.0, whose public client requires httpx2. Product timeout handling
already uses the SDK public Timeout, preserving Hyperspace read600/connect10/
write30/pool10. Test transport now follows DefaultHttpxClient base classes and
passes that actual adapter timeout; it does not guess an SDK version or alter
production admission. Old SDK local control16/16 passes with hostile ambient
credentials. A first test-only attempt used Timeout.__module__, but re-exported
class metadata is shared/mutable across SDKs; that 3-failure log is retained.
A misconstructed narrow-test argument list ran no tests and is retained too.
Fresh CI must pass on the installed newer SDK before acceptance.

3e97c0f CI passed the SDK transport seam, but found two real wire failures:
Anthropic2.4 sends api_key via Authorization, while these retained profiles
explicitly require X-Api-Key. The pinned factory now writes the selected header
as well as omitting the alternate header, independent of SDK defaults. No SDK
pin, credential fallback or product-wide provider change. Both the real-SDK
wire controls and the exact factory-options test retain the selected mode;
local old-SDK regression74/74. The new SDK full CI remains required.
Primary source corroboration: https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/src/anthropic/_client.py
(auth_headers/_api_key_auth). CI's installed SDK request is the actual evidence.
