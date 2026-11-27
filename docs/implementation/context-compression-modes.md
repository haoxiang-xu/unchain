# Context compression prior art

Implementation, contracts, acceptance mapping and delivery evidence are maintained
in [ticket-63.md](ticket-63.md). This file records the design's prior-art sources;
PuPu adoption and UI remain deferred under #457.

- OpenAI's public [session-memory Cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory)
  describes task-specific summaries: temporal ordering, contradictions, uncertainty,
  exact errors, tool outcomes and next steps. This implementation adapts those
  principles; it does not claim to use the unpublished Responses compaction prompt.
  Complete originals and call/result links are retained outside the model.
- [OpenAI Agents #5363](https://github.com/openai/openai-agents-python/pull/5363)
  motivates deferring summaries while tools or approval exchanges are incomplete.
- [LangChain #41250](https://github.com/langchain-ai/langchain/issues/41250) and
  [#41259](https://github.com/langchain-ai/langchain/issues/41259) motivate counting
  fixed overhead once and checking the actual chronological final candidate.
  Only the monotone bare suffix cost is binary-searched; checkpoint pricing and
  legal tool boundaries are still validated separately.

Initial live-model failures additionally motivated short exact source aliases,
source/block authority labels, an explicit prompt against invented permission
requirements, and chronological per-exchange retrieval. Citation validation proves
reference membership, not semantic grounding; real-model evidence must be reviewed.
