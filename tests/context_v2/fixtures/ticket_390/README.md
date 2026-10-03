# Ticket 390 deterministic native replay evidence

These fixtures are constructed SDK response shapes, not recorded production
captures. Inputs contain synthetic text, IDs, arguments and signature bytes.
Their source is the reported mixed text/tool symptom and the real adapters'
normalized representation. Network calls and credentials are not needed.

The retained harness imports the selected runtime's real SQLite store, journal,
coordinator, compiler, adapters and assembler. It runs the candidate's identical
scenario definitions on both runtimes. For baseline execution it substitutes
only the baseline's context factory, which lacks the candidate reader argument.
The OpenAI control uses the selected runtime's existing boundary test.

Checked-in fixture definitions:

- `tests/context_v2/test_context_provider_turn_cross_provider.py`: Anthropic and
  Hyperspace mixed single-call and four-call SDK responses; unsigned Gemini.
- `tests/context_v2/test_ticket390_signed_gemini.py`: mixed Gemini text and a
  function call carrying a synthetic thought signature.
- `tests/context_v2/test_ticket390_durable_negative_matrix.py`: durable negative
  receipt matrix and assembler replay-format negative.

The unsigned Gemini shape loses visible text on the baseline without raising
the opaque-replay error. The signed Gemini shape reproduces the exact error.
The baseline harness uses the signed shape; pytest also covers unsigned text.

## Repeat the baseline/candidate comparison

From the candidate Unchain root, set `TICKET390_PYTHON` to the environment with
the same provider SDK versions used for both runs. The recorded environment is
`/Users/red/Desktop/GITRepo/pupu-acceptance-369-371/runtime-venv-369-370/bin/python`.

```sh
TICKET390_PYTHON=/Users/red/Desktop/GITRepo/pupu-acceptance-369-371/runtime-venv-369-370/bin/python
TICKET390_BASE=$(mktemp -d /private/tmp/ticket390-base.XXXXXX)
git worktree add --detach "$TICKET390_BASE/repo" da5d55b8bf50a51bd81974ada99ba8a888f969d5
"$TICKET390_PYTHON" tests/context_v2/fixtures/ticket_390/reproduce_native_replay.py \
  --runtime-root "$TICKET390_BASE/repo" --expect red --output /private/tmp/ticket390-baseline.json
"$TICKET390_PYTHON" tests/context_v2/fixtures/ticket_390/reproduce_native_replay.py \
  --runtime-root "$PWD" --expect green --output /private/tmp/ticket390-candidate.json
git worktree remove "$TICKET390_BASE/repo"
rmdir "$TICKET390_BASE"
```

Both commands must exit zero. Baseline: all four affected scenarios fail with
the exact ambiguous-mutation error, while OpenAI succeeds. Candidate: all five
succeed. JSON output preserves full error traces, the pinned runtime HEAD,
Python and SDK versions, and SHA-256 of the harness, scenario files and runtime
compiler/coordinator/assembler source. These hashes identify source evidence;
they do not claim deployed wheel or runtime-manifest acceptance.

For the receipt repair, pytest's late-result negative failed before the repair
with `DID NOT RAISE ContextCompileCoordinatorError`; all 12 new boundary
negatives passed afterward. SQL mutations occur only in pytest's temporary
database; updated digests deliberately reach the strict semantic consumers.
Each corruption test asserts injection happened and no second provider send.
