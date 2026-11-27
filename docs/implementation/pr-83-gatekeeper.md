# PR #83 gatekeeper follow-up

The owner selected **retain Intel Mac support and hold both merges** on
2026-11-27. Existing published v0.2.0 artifacts are frozen; these fixes apply to
the source follow-up only.

- **BC-001:** the lazy Skeleton public API remains provider/persistence-free
  during basic import. TYPE_CHECKING declarations expose the existing names to
  static consumers without eager imports or suppression of missing exports.
- **AC-001:** a fresh-process test checks actual source origins and runtime
  manifest, imports every public export with networking forbidden, verifies
  public-name identity and rejects an unknown attribute.
- **BC-002:** ContextCompileCoordinator and provider_recovery_build_trigger use
  the same node-memory-view digest as JournalContextRequestFactory. A recovery
  authorization remains bound to that exact node view; the second immutable
  journal projection preserves its filtering. The legacy no-view identity is
  unchanged, so no journal schema migration is introduced.
- **SEQ-001 / AC-002:** compile an isolated node, request provider-context
  recovery, rebuild and cold replay. Excluded root history and event identities
  must remain absent. The original implementation failed closed because its
  authorization identity omitted the view digest; this is not evidence of
  pre-fix exfiltration. Fixing only that digest exposed the missing second-pass
  filter in the regression, so both propagation points are corrected together.
- **AC-003:** requirements.txt pins Tornado 6.6.1, outside the vulnerable ranges
  of the four newer header/body/redirect advisories. This legacy notebook
  requirement is not the installed library's pyproject runtime closure.
- **AC-004:** PuPu must consume one retained wheel from the selected fixed Core
  source and verify actual consuming imports/bytes and runtime manifest before
  a dependent runtime PASS. Source tests are not package qualification.
- **BC-003 / AC-005:** imported Code result schemas are data, independently of
  the selected Python source. Use an explicit empty referencing registry so
  result validation cannot automatically fetch HTTP or file references in the
  host. Offline regressions capture the original attempted reads and require
  zero reads after the fix. Local definitions and document-local relative-ID
  references still validate; invalid Code results remain failed and replay does
  not rerun their subprocess.

## Outstanding compatibility/security blocker

cryptography 48.0.1 remains deliberately selected for Intel Mac compatibility.
GHSA-g6cj-pr64-35w5, GHSA-jwv3-5hgf-82ww and GHSA-m2h6-j472-rp4c are not
fixed or waived. Upstream fixes require versions that removed Intel Mac support.
Other Dependabot entries must be compared against the PR's current lockfiles,
not assumed active from the older default branch alone. Do not merge or close a
ticket as completed while this compatibility resolution remains outstanding.
