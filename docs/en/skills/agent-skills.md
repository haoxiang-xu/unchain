# Agent Skills

Canonical English skill chapter for the `agent-skills` topic.

## Role and boundaries

This chapter covers `SkillsModule`: discovering `SKILL.md`-shaped instruction files and toolkit-embedded skill descriptors, rendering a name+description catalog, exposing a `skill` activation tool, resolving `/name` explicit invocation from the user's own words, and projecting every activation into a durable `<active_skills>` system block that survives compaction, resume, retry/replay, and cold restart.

A skill is a reusable block of task-specific instructions, addressed by a short kebab-case name and a one-line description. Skills are discovered, not registered by hand: drop a `SKILL.md` file in a known directory, or declare `[[skills]]` in a toolkit manifest, and `SkillsModule` finds it the next time an agent is configured.

**Cross-harness compatibility.** The `<name>/SKILL.md` directory shape — a YAML frontmatter block with `name` and `description`, followed by a Markdown instruction body — is the same convention used by Claude Code, Codex, and DeepSeek Harness skill directories. A skill folder written for one of those harnesses can be dropped into `.unchain/skills/<name>/` unchanged and discovered by Unchain: the frontmatter contract is a bounded, compatible subset (see "SKILL.md frontmatter" below), and unrecognized frontmatter keys are preserved as inert metadata rather than rejected.

## Dependency view

- `unchain.skills.frontmatter` — dependency-free safe-YAML-subset parser (`parse_skill_file`); no knowledge of the registry or the kernel.
- `unchain.skills.models` — the shared vocabulary: `SkillIdentity`, `SkillSummary`, `LoadedSkill`, `ActiveSkill`, `SkillDiagnostic`, `SkillInventory`, plus name/description validation and revision hashing. No I/O.
- `unchain.skills.registry` (`SkillRegistry`, `SkillsConfig`) — scans the ranked filesystem roots and the runtime `Toolkit.skills` list, merges them, and resolves conflicts. Depends on `frontmatter` and `models`.
- `unchain.skills.rendering` — turns `models.py` records into the `<available_skills>` / `<active_skills>` wire text and back (strict parser). No knowledge of the kernel or the registry.
- `unchain.skills.activation` (`ActiveSkillSet`, `ActivationQueue`) — the durable activation state machine: parses the persisted block, merges new activations, re-renders.
- `unchain.skills.harness` (`SkillCatalogHarness`, `SkillActivationHarness`) — `RuntimeHarness` implementations that commit the two blocks through `KernelLoop.dispatch_phase`, reusing the same delimited-system-block pattern as `ToolPromptHarness`.
- `unchain.skills.tools` (`build_skill_tool`) — builds the model-facing `skill` tool from a registry + queue.
- `unchain.agent.modules.skills` (`SkillsModule`) — the `BaseAgentModule` that wires all of the above into an `AgentBuilder` at configure time.
- `unchain.tools.models.SkillDescriptor` and `unchain.tools.toolkit.Toolkit(skills=...)` — the toolkit-embedded skill carrier, parsed from `[[skills]]` manifest tables by `unchain.tools.registry.ToolkitRegistry` or constructed directly.
- `unchain.context.composition` and `unchain.runtime.runtime_protocol` — attribute the two rendered blocks in Context Composition telemetry and advertise the `skills` runtime protocol.

## Core objects

- `SkillsConfig` — discovery configuration (roots, reserved commands, tool name, description budget).
- `SkillRegistry` — stateless-per-call scanner: `list()`, `get(name)`, `resolve(token)`.
- `SkillIdentity` / `SkillSummary` / `LoadedSkill` / `ActiveSkill` / `SkillDiagnostic` / `SkillInventory` — the canonical records (`unchain.skills.models`).
- `SkillDescriptor` — one skill embedded in a toolkit (`unchain.tools.models`).
- `ActivationQueue` / `ActiveSkillSet` — the activation-state machinery (`unchain.skills.activation`).
- `SkillCatalogHarness` / `SkillActivationHarness` — the two `before_model` / `after_tool_batch` harnesses.
- `SkillsModule` — the agent module.

## Execution and state flow

- `SkillsModule.configure(builder)` builds one `SkillRegistry`, lists the inventory once, and registers the `skill` tool only if at least one discovered skill is model-invocable at that moment (decided once per run — see D7 below).
- Every `before_model` step, `SkillCatalogHarness` re-scans the registry and re-renders `<available_skills>` from the live inventory (stateless — like the `<tools>` block, it is not persisted).
- The model calls `skill(name=...)`, or the user writes `/name` in their own message; either path ends up as one pending activation.
- `SkillActivationHarness` (order 262, after `SkillCatalogHarness`'s order 260) parses the existing `<active_skills>` block, merges pending activations, and re-renders it. The block is inserted into both the working messages and `state.transcript`, so it survives checkpoint/resume.
- On `after_tool_batch`, activations queued by the `skill` tool during this iteration are drained and committed, so the tool's result and the activated body land in the same iteration.

## Configuration surface

See "SkillsConfig" below for the full field table. In short: which roots to scan (`include_project_dirs`, `extra_dirs`, `include_user_dirs`, `project_root`, `home`), the catalog description truncation budget, the tool's name, and the set of `/name` tokens that must never resolve to a skill.

## Extension points

- Drop `SKILL.md` (or flat `<name>.md`) files under `.unchain/skills/` or `.agents/skills/` — no code change needed.
- Point `extra_dirs` at any other directory (e.g. a synced `.claude/skills` or `.codex/skills` checkout) to make it discoverable without implicit scanning.
- Declare `[[skills]]` in a toolkit's `toolkit.toml` manifest to ship a skill with a toolkit.
- Construct `SkillDescriptor` instances directly and pass them to `Toolkit(skills=(...))` for programmatic sources.

## Common gotchas

- The `skill` tool's presence is decided once, at configure time, from the inventory snapshot — a skill added to disk mid-run will still appear in the catalog (re-rendered every step) but won't retroactively add the tool if none existed at configure time.
- Filesystem `SKILL.md`/`<name>.md` skills never carry a `tools` list or aliases — those fields only exist on toolkit-embedded `SkillDescriptor` skills.
- `/name` tokens that don't resolve to any known skill are left as ordinary literal text; nothing strips or rewrites the user's message.
- The prompt-cache prefix changes on every catalog or activation change, same as any other harness that edits leading system messages.

## Related class references

- [Tool System API](../api/tools.md) — `SkillDescriptor`, `Toolkit`.
- [Runtime API](../api/runtime.md) — `RuntimeProtocol` manifest.
- [Tool System Patterns](tool-system-patterns.md) — the wider tool/toolkit model this feature builds on.

## Source entry points

- `src/unchain/skills/frontmatter.py`
- `src/unchain/skills/models.py`
- `src/unchain/skills/registry.py`
- `src/unchain/skills/rendering.py`
- `src/unchain/skills/activation.py`
- `src/unchain/skills/harness.py`
- `src/unchain/skills/tools.py`
- `src/unchain/agent/modules/skills.py`
- `src/unchain/tools/models.py` (`SkillDescriptor`)
- `src/unchain/context/composition.py`
- `src/unchain/runtime/runtime_protocol.py`

## Detailed reference

The sections below cover the exact wire formats, configuration fields, and lifecycle guarantees. Every rendered example is reproduced verbatim from what the corresponding function actually emits.

## SKILL.md frontmatter

A skill file is a YAML-frontmatter Markdown file:

```
---
name: release-notes
description: Draft release notes from the merged PRs since the last tag.
---
1. Collect merged PRs since the last release tag with `git log`.
2. Group entries by type: feat, fix, docs, chore.
3. Write one Markdown bullet per PR, linking the PR number.
```

`name` is required and must match `^[a-z0-9]+(?:-[a-z0-9]+)*$` (lowercase kebab-case, 1–64 characters) — and it must exactly equal the directory name (for `<name>/SKILL.md`) or the file stem (for flat `<name>.md`), or the candidate is dropped with an `invalid` diagnostic. `description` is required, non-empty, and at most 1024 characters (persisted verbatim; only the rendered catalog line collapses whitespace and truncates it). Two optional policy keys, `disable-model-invocation` and `user-invocable`, control whether the skill can be activated by the model (`skill` tool) and/or by the user (`/name`) respectively. Any other frontmatter key is preserved verbatim as inert metadata (available through `SkillRegistry.get(name).metadata`, never through the catalog).

The parser (`unchain.skills.frontmatter.parse_skill_file`, decision D1) is a bounded, dependency-free safe subset of YAML — deliberately not a general parser:

**Accepted:**
- `key: value` lines at column 0, with nested mappings and `- item` block lists distinguished purely by indentation (captured as plain `dict`/`list` of `str`, no special typing).
- Plain, `'single'`, and `"double"` quoted scalars; double-quoted scalars support `\"`, `\\`, `\n`, `\t` escapes.
- Inline `# comment` text outside of quotes.
- Block scalars `|`, `|-`, `|+` (literal) and `>`, `>-`, `>+` (folded — lines joined with spaces, a blank line becomes a newline), with the three chomping variants (`clip`/default, `strip`, `keep`).
- Flow sequences `[a, "b", 'c']` (each item scalar-parsed the same way).
- An empty value (`key:` with nothing after it and nothing indented under it) becomes `""`.
- `true`/`false`/`yes`/`no`/`on`/`off` stay as plain strings in `fields` — boolean coercion (`coerce_bool`) is only applied later, and only to the two policy keys.
- Duplicate keys: rejected with `SkillParseError` only for the four keys `name`, `description`, `disable-model-invocation`, `user-invocable`; every other duplicate key lets the last occurrence win.

**Rejected** (all raise `SkillParseError`):
- No leading `---` line, or an unterminated frontmatter fence (no closing `---`).
- Tab characters used for indentation.
- YAML tags (`!...`), anchors (`&x`), and aliases (`*x`).
- A top-level line that isn't `key: value` (or a continuation of a block scalar/nested block).
- Flow mappings (`{a: 1}`) — flow sequences are supported, flow mappings are not.

The output is `ParsedSkillFile(fields: dict[str, object], body: str)`, where `body` is everything after the closing `---` fence with leading/trailing blank lines stripped (the body's own `---` lines, e.g. inside a fenced code block, are never mistaken for a second frontmatter delimiter, because only the first `---` line after the opening one closes the fence).

## Discovery roots and ranking

`SkillRegistry.roots()` returns a fixed, ranked list of filesystem roots plus one virtual root for toolkit-embedded skills. Lower rank always wins on a name collision:

| Rank | Source label | Location | Controlled by |
|---|---|---|---|
| 100 | `project-unchain` | `<project_root>/.unchain/skills` | `include_project_dirs` |
| 200 | `project-agents` | `<project_root>/.agents/skills` | `include_project_dirs` |
| 300 | `custom` | each entry of `extra_dirs`, in the order given | `extra_dirs` |
| 400 | `user-unchain` | `<home>/.unchain/skills` | `include_user_dirs` |
| 500 | `user-agents` | `<home>/.agents/skills` | `include_user_dirs` |
| 600 | `toolkit` (or the descriptor's own `source`) | `Toolkit.skills` on the runtime toolkit | always scanned |

`include_project_dirs=False` skips ranks 100 and 200 together (e.g. PuPu running with no workspace). `include_user_dirs=False` skips ranks 400 and 500. There is **no implicit scanning of `.claude/skills` or `.codex/skills`** — point `extra_dirs` at them explicitly if you want Unchain to discover skills written for those harnesses.

`project_root` resolution: if `SkillsConfig.project_root` is set explicitly, it is used as-is with no `.git` walk. If it is `None`, `resolve_project_root(Path.cwd())` walks up from the current working directory to the nearest ancestor containing a `.git` entry, falling back to the cwd itself if none is found.

Within one root, only **one directory level** is scanned (no recursive `**/SKILL.md`): each direct child is either a subdirectory containing a `SKILL.md` file (skill name = directory name, base directory = that subdirectory) or a flat `<name>.md` file sitting directly in the root (skill name = file stem, base directory = the **root directory itself**, not a per-skill directory). A bare `SKILL.md` file sitting directly in a root (not inside a subdirectory) is ignored, and so are dotfiles/dot-directories. Entries are deduplicated by resolved real path — a root that resolves to an already-scanned real path (e.g. two configured roots or a symlink pointing back into an already-scanned tree) is skipped silently.

**Winner selection.** When two or more discovered skills share the same canonical name, the winner is the one with the lowest `(rank, identity.key)` tuple — lowest rank first, then lexicographically lowest `source:source_id:name` identity key as a deterministic tiebreaker. Every non-winning candidate gets a `shadowed` diagnostic naming the winning source.

**Aliases** exist only on toolkit-embedded `SkillDescriptor` skills (filesystem `SKILL.md`/`<name>.md` skills always have `aliases=()` — there is no frontmatter key for them). An alias that equals any winning canonical name is dropped with an `alias_shadowed` diagnostic; an alias claimed by two different winning identities is dropped with an `alias_ambiguous` diagnostic.

**Reserved commands** (`SkillsConfig.reserved_commands`) apply only to `/name` resolution: a skill whose name collides with a reserved command still appears in the catalog and can still be loaded through the `skill` tool, but `/reserved-name` in user text will never resolve to it (a `reserved` diagnostic is emitted at listing time).

## Source identity and revision

Every discovered skill has a `SkillIdentity(source, source_id, name)`. Its `key` property is `f"{source}:{source_id}:{name}"` — the durable, source-qualified identity used everywhere a skill needs to be told apart from a same-named skill in a different location (the registry never merges descriptors by name alone).

- Filesystem source: `source` is the root label from the table above (`project-unchain`, `project-agents`, `custom`, `user-unchain`, `user-agents`); `source_id` is the resolved absolute path to the `SKILL.md` (or `<name>.md`) file.
- Toolkit source: `source` defaults to `"toolkit"` (overridable on `SkillDescriptor`); `source_id` is the toolkit's id for manifest-declared `[[skills]]` (set by `ToolkitRegistry`'s manifest loader), or whatever the caller passes to `SkillDescriptor(source_id=...)` for a directly constructed `Toolkit(skills=...)` (defaults to `""` if not supplied — pass a distinct `source_id` for each programmatic source to avoid identity collisions).

**Revision** (`compute_skill_revision`) is `"sha256:" + sha256(canonical_json({"body", "tools", "model_invocable", "user_invocable"})).hexdigest()` — a 64-character hex digest prefixed with `sha256:`. It is deliberately independent of `name`, `description`, `source`, and filesystem path: only the parts of a skill that actually change activated behavior affect its revision. `compute_inventory_revision` similarly hashes the sorted `(identity.key, revision)` pairs of every winning skill into one order-independent inventory revision.

For filesystem skills, `tools` is always `()` — the frontmatter parser has no `tools:` mapping to the toolkit tool-list concept; any `tools:` key in a `SKILL.md` file's frontmatter is preserved only as inert metadata, not as the `tools` used in the revision or in `<skill_content tools="...">`. Only toolkit-embedded `SkillDescriptor.tools` populates that field.

## SkillsConfig

```python
from unchain.skills import SkillsConfig
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `project_root` | `str \| Path \| None` | `None` | Explicit project root; `None` resolves via nearest `.git` ancestor of `cwd()`. |
| `include_project_dirs` | `bool` | `True` | Scan ranks 100/200 (`.unchain/skills`, `.agents/skills` under the project root). |
| `extra_dirs` | `tuple[str \| Path, ...]` | `()` | Additional roots, rank 300, scanned in the order given. |
| `include_user_dirs` | `bool` | `True` | Scan ranks 400/500 (`.unchain/skills`, `.agents/skills` under `home`). |
| `catalog_description_max_length` | `int` | `500` | Max characters per catalog description line before truncation with `…`. Must be `>= 3`. |
| `tool_name` | `str` | `"skill"` | Name of the model-facing activation tool. Must be a non-empty string. |
| `reserved_commands` | `tuple[str, ...]` | `()` | Tokens that `/name` must never resolve, even if a skill has that name. |
| `home` | `str \| Path \| None` | `None` | Override for the user home directory used by ranks 400/500; `None` uses `Path.home()`. |
| `extra_skills` | `tuple[SkillDescriptor, ...]` | `()` | Programmatic skills without a toolkit (for example a host's installed skill packs). They join the registry at rank 600 alongside toolkit-embedded skills, each keeping its own `source` / `source_id` identity. |

`reserved_commands` is normalized on construction: each entry is stripped, has any leading `/` removed, lowercased, and empty results are dropped — so `SkillsConfig(reserved_commands=("/Foo", "bar", "//baz"))` stores `("foo", "bar", "baz")`. `extra_dirs` entries are coerced to `Path`. `SkillsConfig.coerce(value)` accepts `None` (defaults), an existing `SkillsConfig`, or a plain `dict` of these fields — this is what `SkillsModule(config=...)` uses internally, so `SkillsModule({"tool_name": "load_skill"})` works.

## The `<available_skills>` catalog block

`SkillCatalogHarness` (order 260, phase `before_model`) renders one `<available_skills>` system message every step from the live registry inventory — it is **not** persisted to the transcript, the same way a `<tools>`-style prompt block is re-derived each step rather than stored. Only model-invocable skills appear, sorted by name. If none are model-invocable, no block is rendered (and any previously-inserted block is removed).

Rendered example for one skill (`release-notes`, `tool_name="skill"`):

```
<available_skills>
# unchain generated skills catalog v1
A skill is a reusable set of task-specific instructions. The following skills are available in this session:

- `release-notes`: Draft release notes from the merged PRs since the last tag.

If the user names a skill, or the task clearly matches a skill's description, call the `skill` tool with the exact skill name before taking task actions. Load all applicable skills, then follow their full instructions. This catalog contains summaries only; do not infer or follow a skill's instructions until it has been loaded.
Skills a user invokes directly, and skills you load, appear in the <active_skills> system block; follow that block and do not load a skill that is already listed there.
</available_skills>
```

## The `skill` tool

`SkillsModule` registers the `skill` tool (`build_skill_tool`) only when the discovery inventory has at least one model-invocable skill **at configure time** — a per-run, one-shot decision (D7). The tool is built with `always_load=True`, so it stays present on the wire even when a `ToolDiscoveryRuntime`/`ToolkitCatalogRuntime` lazy-exposure mode is in effect for everything else.

Calling `skill(name="release-notes")` validates the name, resolves it through the registry (rejecting a valid-looking but unknown or non-model-invocable name), loads the full body, and queues the activation — it never returns the body itself. The queued activation is committed by `SkillActivationHarness` on the next `after_tool_batch`, so the tool result and the projected `<active_skills>` entry land in the same iteration. The tool result is a no-body envelope:

```
<skill_loaded name="release-notes" revision="sha256:f6ade60e95853dfb585e71ed497d5c3b2abc0e9495ff57141fc72abcfd7449a0" status="activated">
Skill "release-notes" is now active. Its full instructions are in the <active_skills> system block; follow them for the rest of this run.
</skill_loaded>
```

`status` is one of:

| Status | Meaning | Sentence |
|---|---|---|
| `activated` | First activation of this identity in this run. | `Skill "{name}" is now active. Its full instructions are in the <active_skills> system block; follow them for the rest of this run.` |
| `already_active` | Same identity, same revision, already active. | `Skill "{name}" was already active at this revision; its instructions are already in the <active_skills> system block.` |
| `superseded` | Same identity, different (newer) revision. | `Skill "{name}" was re-activated at a new revision; the <active_skills> system block now holds the updated instructions.` |

Errors are returned as plain `Error: ...` text, e.g. `Error: invalid skill name "Nope"`, `Error: unknown skill "missing"`, `Error: skill "x" is not available for model invocation`.

## `/name` explicit invocation

Any real user message can activate a skill directly, with no tool call: whitespace-delimited tokens matching `/([A-Za-z0-9_-]+)` (a leading `/`, then letters, digits, `_`, or `-`) are found anywhere in the text, in textual order, deduplicated case-insensitively. Each token is resolved case-insensitively against canonical skill names first, then aliases; `reserved_commands` are skipped before resolution and never activate anything; an unrecognized token is left as ordinary literal text — **the user's message is never modified**, stripped, or rewritten in any way.

"Real user message" means the latest transcript entry with `role: "user"` whose content is plain text (or text content blocks) and is not itself a synthetic skill message (`<skill_content...>` / `<skill_loaded...>`) — tool-result-only user turns are skipped when looking for the latest real text.

If the resolved skill has `user-invocable: false`, no activation happens and a `user_invocation_denied` diagnostic is recorded instead (the skill can still be model-invocable, or vice versa — the two policy flags are independent). Otherwise the skill is loaded and activated with provenance `user:<ordinal>:<digest>`, where `<ordinal>` is the number of real user messages in the durable transcript at that point and `<digest>` is the first 16 hex characters of the SHA-256 of the exact user message text. The same `<ordinal>:<digest>` value is written into the block header as `turn=…`: it records the last real user turn whose `/name` tokens were resolved. On every later step the harness first compares the current turn identity with that marker — a tool loop, retry, interaction resume or cold restart re-processes the *same* turn, so the marker matches and the harness never touches the registry or the filesystem again (even if a higher-ranked source or a new revision appeared meanwhile). Only a genuinely new turn (higher ordinal, or different text) is resolved live; repeating byte-identical text in a later turn is therefore a new explicit activation that supersedes a changed revision, and re-invoking an unchanged revision just refreshes the entry's provenance.

## The durable `<active_skills>` block

`SkillActivationHarness` (order 262, phases `before_model` and `after_tool_batch`) is the one part of the skills system whose state must outlive compaction and process restarts, so it is projected as its own delimited system message and — critically — written into `state.transcript` directly (not just the working messages), because `ContextCompilerHarness` rebuilds the whole message list every step and execution checkpoints persist `state.transcript`, not ad hoc model-context edits.

Rendered example, one activated skill:

```
<active_skills>
# unchain generated active skills v1 turn=1:0123456789abcdef
These skills were activated in this run. Follow their instructions; they stay in force until the run ends.

<skill_content name="release-notes" source="project-unchain" source_id="/repo/.unchain/skills/release-notes/SKILL.md" revision="sha256:f6ade60e95853dfb585e71ed497d5c3b2abc0e9495ff57141fc72abcfd7449a0" activation="user:7e4fca1cce098c11" tools="">
<skill_resources>
Base directory for this skill: /repo/.unchain/skills/release-notes
Resolve relative paths mentioned by this skill against the base directory before using them. Load referenced resources only as needed.
</skill_resources>

<skill_instructions>
1. Collect merged PRs since the last release tag with `git log`.
2. Group entries by type: feat, fix, docs, chore.
3. Write one Markdown bullet per PR, linking the PR number.
</skill_instructions>
</skill_content>
</active_skills>
```

Attribute values are XML-escaped (`&`, `"`, `<`, `>`); `activation` is `user:<ordinal>:<digest>` for `/name` invocation, or `tool:<call_id>` for a `skill` tool call once its call id is known (falling back to the literal `tool` if it cannot be matched). A skill body containing the literal string `</skill_instructions>` or `</skill_content>` is rejected at activation time with a `body_delimiter` diagnostic, rather than being persisted in a form that would break the strict parser. A skill with no base directory (a toolkit skill with `base_dir=None`) gets `<skill_resources>This skill has no resource directory.</skill_resources>` instead of the `Base directory for this skill: ...` line.

**Why this survives everything:**

- It is a `role: "system"` message. Context V2's `ContextCompilerHarness` copies every source system message across each compile and only budget-compacts non-system turns; `SlidingWindow`/`LastN` optimizers likewise only ever touch the non-system portion. A system-role block is therefore outside the turn-compaction path by construction, in both context modes.
- It is written into `state.transcript`, which is what execution checkpoints persist and what `merge_checkpoint_transcript_with_incoming` restores on resume — so suspend/resume, retry, replay, and cold restart (rebuilding purely from a persisted checkpoint dict) all see the same block without Unchain re-reading a single skill source file.
- Only the *first* `<active_skills>` block found among the messages is treated as authoritative on each pass (`ActiveSkillSet.from_messages`); a duplicate would be collapsed into one on the next render.
- There is exactly one entry per identity (`source:source_id:name`), keyed in a dict — re-activating the same identity at the same revision is a no-op (`already_active`); re-activating it at a different revision (e.g. after the skill file was edited and the user explicitly types `/name` again) replaces that entry **in place** (its position in the block is unchanged) and reports `superseded`. Editing or deleting the underlying source file between turns changes nothing about an already-active entry — the block is never re-read against the filesystem, only re-parsed and re-merged with new activations.
- A brand-new run with no checkpoint to resume from starts with no `<active_skills>` block at all (empty set) — activation state never leaks across unrelated runs.
- The block's header carries an explicit version (`# unchain generated active skills v1`). If a persisted block's version is anything other than `v1`, parsing fails closed with `SkillActivationStateError` rather than silently ignoring or reinterpreting it — a corrupted or future-format snapshot stops the run instead of being guessed at.

The runtime also mirrors a summarized view into `state.component_bucket("skills")` as `{"version": 1, "active": {...}, "diagnostics": [...]}` for introspection; this mirror is always rebuilt from the rendered block, never the other way around — the block itself is the only durable authority.

## Toolkit-embedded skills

A toolkit manifest (`toolkit.toml`) can ship one or more skills directly via `[[skills]]` tables. From `src/unchain/toolkits/builtin/plan/toolkit.toml`:

```toml
[[skills]]
name = "plan"
title = "Plan First"
description = "Draft a step-by-step plan and wait for confirmation before executing."
body = "Before doing any work: draft a step-by-step plan using the planning tools ({tools}), present it to me, and wait for my explicit confirmation before executing."
tools = ["plan_start", "plan_update", "plan_read", "plan_finalize", "plan_list"]
```

Recognized keys: `name`, `description`, and `body` are required; `tools` (a list of tool names that must already be declared in the same manifest's `[[tools]]`), `disable-model-invocation`, `user-invocable`, and `aliases` are optional. `name` and every entry in `aliases` must be kebab-case, at most 64 characters. Any other key — including the PuPu-only `title` shown above, or a `phase` key from an older manifest — is simply ignored by the loader; `plan/toolkit.toml` keeps `title` for PuPu's own display purposes and no longer declares `phase`.

`{tools}` is substituted into `body` with the declared tool names, each backtick-quoted and comma-separated, **every time the registry scans** (`SkillRegistry.list()`), not once at manifest-parse time. For the `plan` skill above, the rendered body is:

```
Before doing any work: draft a step-by-step plan using the planning tools (`plan_start`, `plan_update`, `plan_read`, `plan_finalize`, `plan_list`), present it to me, and wait for my explicit confirmation before executing.
```

`ToolkitRegistry` parses each `[[skills]]` table into a `SkillDescriptor(source="toolkit", source_id=<toolkit id>, base_dir=<toolkit root path>)` and attaches the resulting tuple to `ToolkitDescriptor.skills`; `ToolkitRegistry.instantiate_toolkit()` copies that tuple onto the instantiated `Toolkit.skills`. `AgentBuilder.add_tool(toolkit)` (decision D10) appends every incoming `toolkit.skills` descriptor onto `builder.toolkit.skills` — it does not deduplicate; the skills registry's rank/identity-key resolution is what decides a winner if two sources declare the same name.

Skills can also be constructed programmatically and attached to any `Toolkit`, without a manifest:

```python
from unchain.tools import SkillDescriptor, Toolkit

toolkit = Toolkit(
    skills=(
        SkillDescriptor(
            name="echo-helper",
            description="Echo any text back verbatim.",
            body="When asked to echo something, call `echo` with the exact text.",
            tools=("echo",),
            source="toolkit",
            source_id="echo-toolkit",
        ),
    ),
)
```

Pass an explicit, distinct `source_id` for each programmatically-built source — the default is `""`, and two descriptors that share `source`, `source_id`, and `name` are indistinguishable identities.

## Diagnostics

`SkillDiagnostic(kind, name, source, source_id, message)` is a non-fatal note about discovery or invocation; `SKILL_DIAGNOSTIC_KINDS` in `unchain.skills.models` names ten allowed kinds, of which the registry and activation harness currently emit eight:

| Kind | Emitted when |
|---|---|
| `invalid` | Frontmatter fails to parse, the name doesn't match the file/directory, the name or description is invalid, or a boolean policy field can't be coerced. |
| `shadowed` | A discovered skill lost a name collision to a lower-rank (or lexicographically-earlier) winner. |
| `alias_shadowed` | An alias equals an existing winning skill's canonical name and was dropped. |
| `alias_ambiguous` | An alias is claimed by two different winning identities and was dropped. |
| `reserved` | A winning skill's name collides with a `reserved_commands` entry (`/name` will never resolve it). |
| `inaccessible` | The `SKILL.md`/`<name>.md` file couldn't be resolved, read, or decoded as UTF-8; or a skill's body couldn't be re-read at activation time. |
| `body_delimiter` | An activation was rejected because the body contains a literal `</skill_instructions>` or `</skill_content>`. |
| `user_invocation_denied` | A `/name` token resolved to a skill with `user-invocable: false`. |

Two kinds, `unknown_skill` and `model_invocation_denied`, are declared as valid `SkillDiagnostic` kinds but are not currently emitted anywhere in the registry, activation, or tool code — they are reserved for future use.

## Context Composition attribution

`context/composition.py::_classify_message` gives the two rendered blocks their own reserved taxonomy slots under the `"skills"` category: a system message whose content starts with `<available_skills>` classifies as `("skills", "catalog_metadata")`, and one starting with `<active_skills>` classifies as `("skills", "loaded_body")`. Every other message keeps its previous classification unchanged. (The `"skills"` category also defines a third subtype, `expanded_invocation`, used by an unrelated privacy-preserving context-composition hint — it is not produced by either skills harness.)

## Runtime protocol `skills` 1.0

`unchain.runtime.runtime_protocol` advertises a versioned `skills` entry in the canonically-ordered runtime protocol manifest:

```python
RuntimeProtocol(
    id="skills",
    major=1,
    minor=0,
    features=(
        "active_skills_snapshot_v1",
        "catalog_v1",
        "skill_md_registry_v1",
        "skill_tool_v1",
        "toolkit_embedded_skills_v1",
        "user_invocation_v1",
    ),
)
```

`unchain.skills` also exports `SKILLS_PROTOCOL_ID = "skills"`, `SKILLS_CONTRACT_VERSION = 1` (the descriptor/inventory/resolution shape), and `ACTIVE_SKILLS_SNAPSHOT_VERSION = 1` (the `<active_skills>` block format, matching the block's own `v1` header).

## Complete usage example

```python
from unchain import Agent
from unchain.agent import ToolsModule, SkillsModule
from unchain.skills import SkillsConfig
from unchain.toolkits import CoreToolkit

agent = Agent(
    name="assistant",
    provider="openai",
    model="gpt-5",
    modules=(
        ToolsModule(tools=(CoreToolkit(workspace_root="."),)),
        SkillsModule(
            SkillsConfig(
                project_root=".",
                reserved_commands=("btw", "fyi", "queue"),
            )
        ),
    ),
)

result = agent.run("Please /release-notes for the v2.3 tag.")
```

`SkillsConfig(project_root=".")` scans `./.unchain/skills` and `./.agents/skills` (ranks 100/200) plus the user's `~/.unchain/skills` and `~/.agents/skills` (ranks 400/500, since `include_user_dirs` defaults to `True`); `reserved_commands=("btw", "fyi", "queue")` means a skill accidentally named one of those never activates through `/name` (it can still be listed and loaded via the `skill` tool). The `/release-notes` token in the example message resolves and activates the skill from the user's own turn, with no model round-trip required first.

## Known limitations

- The prompt-cache prefix changes on every catalog or activation change — the same trade-off any other harness that edits leading system messages makes (e.g. `ToolPromptHarness`).
- The `skill` tool's presence is decided per run at configure time (D7): a skill that becomes model-invocable mid-run (through a file edit or a toolkit change) will not retroactively add the tool to an already-configured agent.
- Only one directory level per root is scanned — there is no recursive `**/SKILL.md` discovery, matching the reference (`dsh`) harness's behavior.
- Turn identity for `/name` activation counts real user messages in the durable transcript. A mid-run transcript rewrite that removes earlier user messages (for example a summarising compaction) can lower that count; a later turn is still detected when its text differs, but a byte-identical repeat sent right after such a rewrite may be treated as a replay of the previous turn (a missed re-activation, never a wrong one).

## Related Skills

- [Tool System Patterns](tool-system-patterns.md) — Tool/Toolkit definitions this feature builds on
- [Architecture Overview](architecture-overview.md) — Where `SkillsModule` fits in the agent build pipeline
- [Runtime Engine](runtime-engine.md) — `KernelLoop`, `RuntimeHarness` phases, and checkpoint/resume mechanics
