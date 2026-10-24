# Choose local trimming or an agent summary

`unchain.context.ContextCompressor` is an explicit Python-library API. It creates
a model-input view without overwriting canonical conversation history. The
existing Context V2 compiler also uses binary suffix search under pressure;
its journal/checkpoint protocol is unchanged.

## Local trimming: no extra model request

```python
from unchain.context import ContextCompressor, DirectoryCompressionStore

store = DirectoryCompressionStore("./context-archives")
compressor = ContextCompressor(mode="trim", archive_store=store)
view = compressor.compress(
    full_history,                 # canonical originals from your journal
    max_input_tokens=16_000,       # input allocation, after output reservation
    fixed_overhead_tokens=700,    # tools/schema/transport overhead, counted once
)
# Send list(view.messages) to your Agent instead of the full input.
```

The search retains the largest safe recent suffix. System/developer messages
and the latest user turn remain intact. A closed tool call/result exchange is
never split, including parallel and cross-turn exchanges. Unfinished exchanges
are retained. If mandatory context cannot fit, compression raises rather than
dropping it. Original tool names, arguments, outputs, citations and errors stay
in the archive.

## Agent summary: one model run for a new archived prefix

```python
from unchain import Agent
from unchain.context import AgentSummarizer, ContextCompressor

summary_agent = Agent(
    name="summarizer", provider="openai", model=your_model, api_key=your_key,
)
compressor = ContextCompressor(
    mode="summary",
    archive_store=store,
    summarizer=AgentSummarizer(
        summary_agent,
        payload={"max_output_tokens": 2_000},  # provider-native generation limit
    ),
    summary_reserve_tokens=2_400,  # summary + archive marker, within input budget
    max_summary_input_tokens=64_000,
)
view = compressor.compress(full_history, max_input_tokens=16_000)
```

`AgentSummarizer` uses a fresh clone with the published Unchain summary prompt,
no modules/tools, no previous response id, and a one-iteration run. Its provider
and model are the supplied Agent's. The default replay policy key includes that
provider/model and the optional generation settings. The built-in adapter accepts
output limits, sampling, stop, penalty and reasoning/thinking settings; it rejects
payload controls that replace instructions, inputs, state, model or tools.
For other provider controls, use a custom summarizer. For a custom summarizer
callable, pass an explicit `summary_key` identifying its model/prompt/settings.
The callable receives `SummaryRequest` and returns the documented JSON object
or JSON text. One enclosing JSON code fence is accepted; surrounding prose,
multiple objects, duplicate keys and unknown references are rejected. Change the
key when those settings change.

The prompt adapts the principles in OpenAI's public
[session-memory example](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory):
chronology, contradictions, exact error strings, uncertainty and tool outcomes.
It is not the unpublished Responses service's internal compaction prompt.

The six summary sections are `goal`, `constraints`, `decisions`, `progress`,
`open_items`, and `next_steps`. Every entry must have `text` and `source_ids`
pointing into the exact archived prefix. `request.to_dict()` supplies short IDs
`s0`, `s1`, etc., and labels source authority, including separate block labels for
mixed user/tool messages. The host maps exact short references back to canonical
archive IDs; it never guesses or repairs a mistyped ID. A callback may also cite
`request.source_ids` directly. The host validates the closed schema,
citations and final budget. The summary is carried as untrusted historical
observations, never as a new system instruction. References establish where a
claim points; they do not prove that the model interpreted it correctly.

If any call or approval is unfinished, summary is deferred with
`CompressionError` when pressure requires compression. Complete the pending
exchange before retrying, or explicitly choose local trim. Summary/model/input
budget/storage failures never silently fall back to trimming. Original tool
outputs are not truncated to make the summarizer request fit; an oversized
summary input raises explicitly. Configure a suitable summarizer window.

## Recover the evidence with a model-visible tool

```python
from unchain.agent import ToolsModule

reader = view.read_tool(store)  # only when view.archive_id is not None
main = Agent(
    name="worker", provider=your_provider, model=your_model, api_key=your_key,
    modules=(ToolsModule(tools=(reader,)),),
)
result = main.run(list(view.messages))
```

`context_archive_read()` reads the JSON catalog of exact source ids and the
call/result reference table, including a `tool_catalog` with chronological occurrences.
`context_archive_read(source_id="...")` reads that original JSON message.
For exact exchanges, use `context_archive_read(call_id="...")` or
`context_archive_read(tool_name="run_checks", occurrence=0)` for the first call,
`occurrence=1` for the next. The returned envelope contains the exact provider
call and result items, with their source IDs. This avoids confusing an original
failure with a later rerun or scanning unrelated parallel outputs. Source and
exchange selectors cannot be combined; ambiguous call IDs fail explicitly.
Results are bounded
Unicode-text pages: follow `next_offset` using `offset` to read more, then verify
the complete UTF-8 text against `sha256`. A small complete source also appears
as a parsed `message`; large sources remain fully preserved in the archive.
The tool is bound to this one archive; it does
not expose the directory or arbitrary files. The application may also use
`store.get(view.archive_id).read(source_id)` directly.

Both modes save a validated archive before returning a reduced view. A repeated
source/policy reuses the same receipt, including after a process restart, without
rerunning the summarizer. Archive data has a schema and content/source digests;
corruption is an error. Keep the archive directory for as long as its views are
in use. Store it with the same access controls as your original chat journal.

Always compress the canonical full history, not `view.messages` plus new turns.
The API rejects its own reduced markers as input, preventing recursive loss of
original evidence. Append real messages/results to your journal and build the
next view from that journal. If another harness already shortened a tool result,
rehydrate its original from the journal/artifacts first. Known Unchain tool-result
preview envelopes are rejected even without compression pressure; arbitrary
caller-side truncation cannot be detected. The compressor cannot recover data
discarded before it received the input. In particular, do not assume a previous
Agent's returned model view is always an original journal.
The archive is a retrieval copy, not a replacement
for the host's canonical journal.

This API does not automatically change a host's `ContextModule` or PuPu memory
mode. Use it at the application's history-to-Agent boundary. Hosts with durable
Context V2 authority continue to use its coordinator and consumption proofs;
do not stack this view over an already compiled V2 view.
