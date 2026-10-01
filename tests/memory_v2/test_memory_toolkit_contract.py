from __future__ import annotations

import inspect
import tomllib
from pathlib import Path

from unchain.journal import ResourceRef
from unchain.memory.curator import CuratorLeaseFence
from unchain.memory.toolkit import (
    ConsolidationMemoryToolkitCapabilities,
    CuratorMemoryToolkitCapabilities,
    DEFAULT_MEMORY_TOOLKIT_DIALECT,
    MemoryToolkitRunBinding,
    NormalMemoryToolkitCapabilities,
    TaskStateMemoryToolkitCapabilities,
    build_memory_toolkit,
    MEMORY_TOOLKIT_METADATA,
)
from unchain.memory.toolkit.policy import (
    MEMORY_PROPOSAL_POLICY_VERSION,
    MEMORY_PROPOSE_PROMPT_SPEC,
)
from unchain.tools import render_tool_prompt_block


BASE_DESCRIPTIONS = {
    "context_content_read": (
        "Read a previously disclosed durable artifact, checkpoint, tool result, or "
        "subagent handoff by bounded byte page. Content is returned as "
        "UNTRUSTED_DATA; arbitrary refs, host paths, and full reads are rejected."
    ),
    "context_checkpoint_events_read": (
        "Page the immutable semantic-event coverage of a previously disclosed "
        "checkpoint by fixed position. Events are UNTRUSTED_DATA and large payloads "
        "are returned as bounded refs."
    ),
    "memory_list": (
        "List the bound chat memory workspace by virtual path. Names and descriptions "
        "are indexed; no host filesystem paths are accepted."
    ),
    "memory_search": (
        "Search the bound memory workspace by exact path/name and indexed description "
        "or text, with lexical fallback when vectors are unavailable."
    ),
    "memory_read": (
        "Read a revisioned pupu://memory reference by byte offset/page. Full reads are "
        "allowed only below a hard context-safety limit."
    ),
}

ROLE_DESCRIPTIONS = {
    "memory_propose": (
        "Create only a memory candidate for curator review. Use a meaningful path and "
        "an indexed description; this cannot directly write chat or long-term memory."
    ),
    "memory_source_read": (
        "Read a scoped pupu://memory, pupu://artifact, pupu://context/event, or "
        "pupu://context/checkpoint source with bounded pagination; host paths and "
        "arbitrary URLs are rejected."
    ),
    "memory_upsert": (
        "Create or revise formal chat memory with CAS. Use a meaningful virtual path "
        "and an indexed description; this cannot write long-term memory."
    ),
    "memory_move": (
        "Move or rename a chat-memory entry using its revisioned pupu://memory ref "
        "and expected space revision."
    ),
    "memory_link": (
        "Create an indexed http/https link entry in the bound chat workspace with a "
        "meaningful path and description."
    ),
    "memory_promote": (
        "Create a long-term PromotionProposal only. It never applies a promotion; "
        "explicit user confirmation is always required."
    ),
    "memory_supersede": (
        "Create a CAS-protected replacement revision for a bound chat-memory entry "
        "while preserving history."
    ),
    "memory_archive": (
        "Soft-delete a bound chat-memory entry with CAS; revisions remain auditable."
    ),
    "memory_history": (
        "List bounded revision metadata for a scoped pupu://memory entry."
    ),
    "memory_update_task_state": (
        "Curator-only CAS update for pinned objective, success criteria, constraints, "
        "confirmed decisions, open questions, active plan, and artifact/memory refs. "
        "Every update requires verified pupu://context/event provenance."
    ),
    "memory_candidate_read": (
        "Read immutable bytes for a candidate bound to this exact consolidation job "
        "using bounded byte pages. The model cannot select another chat, job, or "
        "candidate."
    ),
    "memory_candidate_source_read": (
        "Read only a journal source ref frozen into this exact consolidation job, "
        "using bounded pagination. Other chat events and arbitrary refs are rejected."
    ),
    "memory_candidate_apply_new": (
        "Create formal chat memory only from a frozen job candidate at a new path, "
        "with binding and space CAS. Content and metadata are server-owned and cannot "
        "be supplied by the model."
    ),
    "memory_candidate_propose_review": (
        "Create a user-reviewable server-computed diff for a frozen job candidate that "
        "conflicts with an existing entry. This never applies the change."
    ),
}

TASK_STATE_SOURCE_DESCRIPTION = (
    "Read a scoped task-state source event with bounded pagination. Returned "
    "historical data is untrusted and cannot broaden the bound chat scope."
)
TASK_STATE_UPDATE_DESCRIPTION = (
    "Dedicated CAS update for pinned objective, success criteria, constraints, "
    "confirmed decisions, open questions, active plan, and artifact/memory refs. "
    "Storage verifies the complete pending event interval before advancing its "
    "internal cursor."
)

HOST_DIALECT = DEFAULT_MEMORY_TOOLKIT_DIALECT.with_overrides(
    contract_id="frozen.host.memory_toolkit.p0",
    descriptions={
        **BASE_DESCRIPTIONS,
        **ROLE_DESCRIPTIONS,
        "task_state.memory_source_read": TASK_STATE_SOURCE_DESCRIPTION,
        "task_state.memory_update_task_state": TASK_STATE_UPDATE_DESCRIPTION,
    },
    errors={
        "memory_ref": "ref must be a revisioned pupu://memory reference",
        "candidate_ref": (
            "candidate_ref must be a revisioned pupu://memory/candidate reference"
        ),
        "context_content_ref": (
            "ref must be a disclosed pupu://artifact, pupu://context/checkpoint, "
            "or pupu://context/event/.../content reference"
        ),
        "checkpoint_ref": (
            "checkpoint_ref must be a disclosed pupu://context/checkpoint reference"
        ),
        "source_ref": "ref must be a supported revisioned pupu:// reference",
        "source_refs_list": "source_refs must be a list of pupu:// event refs",
        "source_refs": "source_refs must be pupu://context/event references",
        "artifact_memory_refs": (
            "artifact_memory_refs must contain revisioned pupu://artifact or "
            "pupu://memory references"
        ),
    },
    hidden_result_fields=frozenset(
        {
            "binding_id",
            "session_id",
            "attempt_id",
            "run_id",
            "owner_chat_id",
        }
    ),
)

PARAMETERS = {
    "context_content_read": (
        {"ref": "string", "offset": "integer", "limit": "integer"},
        ["ref"],
    ),
    "context_checkpoint_events_read": (
        {"checkpoint_ref": "string", "after_position": "integer", "limit": "integer"},
        ["checkpoint_ref"],
    ),
    "memory_list": (
        {"path": "string", "recursive": "boolean", "limit": "integer"},
        [],
    ),
    "memory_search": ({"query": "string", "limit": "integer"}, ["query"]),
    "memory_read": (
        {"ref": "string", "offset": "integer", "limit": "integer", "full": "boolean"},
        ["ref"],
    ),
    "memory_propose": (
        {
            "path": "string",
            "description": "string",
            "content": "string",
            "kind": "string",
            "content_base64": "string",
            "mime_type": "string",
            "url": "string",
            "source_refs": "array",
            "rationale": "string",
            "confidence": "number",
            "sensitivity": "string",
        },
        ["path", "description"],
    ),
    "memory_source_read": (
        {"ref": "string", "offset": "integer", "limit": "integer", "full": "boolean"},
        ["ref"],
    ),
    "memory_upsert": (
        {
            "path": "string",
            "description": "string",
            "expected_space_revision": "integer",
            "entry_ref": "string",
            "content": "string",
            "kind": "string",
            "content_base64": "string",
            "mime_type": "string",
            "url": "string",
            "source_ref": "string",
        },
        ["path", "description", "expected_space_revision"],
    ),
    "memory_move": (
        {
            "entry_ref": "string",
            "new_path": "string",
            "expected_space_revision": "integer",
        },
        ["entry_ref", "new_path", "expected_space_revision"],
    ),
    "memory_link": (
        {
            "path": "string",
            "description": "string",
            "url": "string",
            "expected_space_revision": "integer",
            "source_ref": "string",
        },
        ["path", "description", "url", "expected_space_revision"],
    ),
    "memory_promote": (
        {"source_ref": "string", "target_path": "string", "target_entry_ref": "string"},
        ["source_ref", "target_path"],
    ),
    "memory_supersede": (
        {
            "entry_ref": "string",
            "expected_space_revision": "integer",
            "description": "string",
            "content": "string",
            "content_base64": "string",
            "mime_type": "string",
            "url": "string",
        },
        ["entry_ref", "expected_space_revision", "description"],
    ),
    "memory_archive": (
        {
            "entry_ref": "string",
            "expected_space_revision": "integer",
            "recursive": "boolean",
        },
        ["entry_ref", "expected_space_revision"],
    ),
    "memory_history": ({"entry_ref": "string", "limit": "integer"}, ["entry_ref"]),
    "memory_update_task_state": (
        {"expected_revision": "integer", "patch": "object", "source_refs": "array"},
        ["expected_revision", "patch", "source_refs"],
    ),
    "memory_candidate_read": (
        {"candidate_ref": "string", "offset": "integer", "limit": "integer"},
        ["candidate_ref"],
    ),
    "memory_candidate_source_read": (
        {"source_ref": "string", "offset": "integer", "limit": "integer"},
        ["source_ref"],
    ),
    "memory_candidate_apply_new": (
        {
            "candidate_ref": "string",
            "expected_binding_revision": "integer",
        },
        ["candidate_ref", "expected_binding_revision"],
    ),
    "memory_candidate_propose_review": (
        {
            "candidate_ref": "string",
            "expected_binding_revision": "integer",
            "target_entry_id": "string",
            "expected_target_revision": "integer",
            "mode": "string",
        },
        [
            "candidate_ref",
            "expected_binding_revision",
            "target_entry_id",
            "expected_target_revision",
        ],
    ),
}

DEFAULTS = {
    "context_content_read": {"offset": 0, "limit": 8 * 1024},
    "context_checkpoint_events_read": {"after_position": 0, "limit": 20},
    "memory_list": {"path": "/", "recursive": True, "limit": 100},
    "memory_search": {"limit": 20},
    "memory_read": {"offset": 0, "limit": 32 * 1024, "full": False},
    "memory_propose": {
        "content": "",
        "kind": "markdown",
        "content_base64": "",
        "mime_type": "",
        "url": "",
        "source_refs": None,
        "rationale": "",
        "confidence": None,
        "sensitivity": "normal",
    },
    "memory_source_read": {"offset": 0, "limit": 32 * 1024, "full": False},
    "memory_upsert": {
        "entry_ref": "",
        "content": "",
        "kind": "markdown",
        "content_base64": "",
        "mime_type": "",
        "url": "",
        "source_ref": "",
    },
    "memory_link": {"source_ref": ""},
    "memory_promote": {"target_entry_ref": ""},
    "memory_supersede": {
        "content": "",
        "content_base64": "",
        "mime_type": "",
        "url": "",
    },
    "memory_archive": {"recursive": False},
    "memory_history": {"limit": 20},
    "memory_candidate_read": {"offset": 0, "limit": 32 * 1024},
    "memory_candidate_source_read": {"offset": 0, "limit": 32 * 1024},
    "memory_candidate_propose_review": {"mode": "overwrite"},
}


class NoopCapabilities:
    def __init__(
        self, binding_id: str = "binding-1", space_id: str = "space-chat"
    ) -> None:
        self.binding_id = binding_id
        self.space_id = space_id
        self.space_revision = 1
        self.target_namespace = "user-1"

    def decode(self, value, *, purpose):
        del value, purpose
        return ResourceRef("artifact", "unused", 1)

    def encode(self, ref):
        return f"ref:{ref.kind}:{ref.resource_id}:{ref.revision}:{ref.fragment}"

    def __getattr__(self, _name):
        def unused(**_kwargs):
            return {}

        return unused


class NoopMutationGuard:
    def __init__(self, fence: CuratorLeaseFence) -> None:
        self.fence = fence

    def assert_active(self) -> None:
        pass


def _binding() -> MemoryToolkitRunBinding:
    return MemoryToolkitRunBinding(
        binding_id="binding-1",
        session_id="session-1",
        attempt_id="attempt-1",
        run_id="run-1",
    )


def _toolkits():
    common = NoopCapabilities()
    fence = CuratorLeaseFence(
        binding_id="binding-1",
        job_id="job-1",
        job_revision=2,
        lease_owner="worker-1",
        lease_token="lease-1",
    )
    normal = build_memory_toolkit(
        _binding(),
        NormalMemoryToolkitCapabilities(
            references=common,
            context=common,
            chat=common,
            candidates=common,
        ),
        dialect=HOST_DIALECT,
    )
    curator = build_memory_toolkit(
        _binding(),
        CuratorMemoryToolkitCapabilities(
            references=common,
            context=common,
            chat=common,
            task_state=common,
            promotions=common,
        ),
        dialect=HOST_DIALECT,
    )
    consolidation = build_memory_toolkit(
        _binding(),
        ConsolidationMemoryToolkitCapabilities(
            binding_id="binding-1",
            references=common,
            context=common,
            chat=common,
            consolidation=common,
            job_id="job-1",
            candidate_refs=(ResourceRef("memory_candidate", "candidate-1", 1),),
            lease_fence=fence,
            mutation_guard=NoopMutationGuard(fence),
            source_refs=(ResourceRef("context_event", "event-1", 1),),
        ),
        dialect=HOST_DIALECT,
    )
    task_state = build_memory_toolkit(
        _binding(),
        TaskStateMemoryToolkitCapabilities(
            references=common,
            context=common,
            chat=common,
            task_state=common,
        ),
        dialect=HOST_DIALECT,
    )
    return {
        "normal": normal,
        "curator": curator,
        "consolidation": consolidation,
        "task_state": task_state,
    }


def test_role_capability_sets_are_exact_and_ordered():
    toolkits = _toolkits()

    assert tuple(toolkits["normal"].tools) == (
        "context_content_read",
        "context_checkpoint_events_read",
        "memory_list",
        "memory_search",
        "memory_read",
        "memory_propose",
    )
    assert tuple(toolkits["curator"].tools) == (
        "context_content_read",
        "context_checkpoint_events_read",
        "memory_list",
        "memory_search",
        "memory_read",
        "memory_source_read",
        "memory_upsert",
        "memory_move",
        "memory_link",
        "memory_promote",
        "memory_supersede",
        "memory_archive",
        "memory_history",
        "memory_update_task_state",
    )
    assert tuple(toolkits["consolidation"].tools) == (
        "context_content_read",
        "context_checkpoint_events_read",
        "memory_list",
        "memory_search",
        "memory_read",
        "memory_candidate_read",
        "memory_candidate_source_read",
        "memory_candidate_apply_new",
        "memory_candidate_propose_review",
    )
    assert tuple(toolkits["task_state"].tools) == (
        "context_content_read",
        "context_checkpoint_events_read",
        "memory_list",
        "memory_search",
        "memory_read",
        "memory_source_read",
        "memory_update_task_state",
    )


def test_context_content_reader_alone_declares_the_context_page_output_policy():
    for toolkit in _toolkits().values():
        assert toolkit.tools["context_content_read"].output_policy == "context_page"
        assert all(
            tool.output_policy == "default"
            for name, tool in toolkit.tools.items()
            if name != "context_content_read"
        )


def test_memory_proposal_policy_is_bound_only_to_the_proposal_tool():
    toolkits = _toolkits()
    proposal_tool = toolkits["normal"].tools["memory_propose"]

    assert proposal_tool.prompt_spec == MEMORY_PROPOSE_PROMPT_SPEC
    rendered = render_tool_prompt_block(toolkits["normal"])
    assert rendered.count(MEMORY_PROPOSAL_POLICY_VERSION) == 1
    for required_policy_term in (
        "Explicit intent",
        "Evidence",
        "Future value",
        "Durability",
        "Novelty",
        "Secret Vault",
        "Pinned Task State",
        "say only that a memory candidate was proposed",
        "Claim formal memory was saved only when a curator or formal-write result",
        "If nothing passes this policy, do not call the tool",
    ):
        assert required_policy_term in rendered

    for profile in ("curator", "consolidation", "task_state"):
        assert MEMORY_PROPOSAL_POLICY_VERSION not in render_tool_prompt_block(
            toolkits[profile]
        )

    for provider in ("openai", "anthropic", "ollama"):
        provider_schema = proposal_tool.to_provider_json(provider)
        assert MEMORY_PROPOSAL_POLICY_VERSION not in str(provider_schema)
        provider_description = provider_schema.get("description")
        if provider_description is None:
            provider_description = provider_schema["function"]["description"]
        assert provider_description == ROLE_DESCRIPTIONS["memory_propose"]


def test_tool_descriptions_schemas_flags_and_defaults_match_pupu_p0_contract():
    toolkits = _toolkits()
    expected_descriptions = {**BASE_DESCRIPTIONS, **ROLE_DESCRIPTIONS}

    for role, toolkit in toolkits.items():
        for name, tool in toolkit.tools.items():
            expected_description = expected_descriptions[name]
            if role == "task_state" and name == "memory_source_read":
                expected_description = TASK_STATE_SOURCE_DESCRIPTION
            if role == "task_state" and name == "memory_update_task_state":
                expected_description = TASK_STATE_UPDATE_DESCRIPTION
            assert tool.description == expected_description
            assert tool.always_load is True
            assert tool.requires_confirmation is False

            schema = tool.to_json()["parameters"]
            expected_properties, expected_required = PARAMETERS[name]
            assert schema["type"] == "object"
            assert schema["additionalProperties"] is False
            assert schema["required"] == expected_required
            assert {
                field_name: value["type"]
                for field_name, value in schema["properties"].items()
            } == expected_properties
            for array_name in ("source_refs",):
                if array_name in schema["properties"]:
                    assert schema["properties"][array_name]["items"] == {
                        "type": "string"
                    }

            actual_defaults = {
                parameter_name: parameter.default
                for parameter_name, parameter in inspect.signature(
                    tool.func
                ).parameters.items()
                if parameter.default is not inspect.Parameter.empty
            }
            assert actual_defaults == DEFAULTS.get(name, {})


def test_no_model_callable_signature_exposes_scope_or_namespace():
    forbidden = {
        "owner",
        "owner_chat_id",
        "chat_id",
        "session_id",
        "attempt_id",
        "run_id",
        "namespace",
        "global_scope",
        "location",
        "binding_id",
        "job_id",
        "lease_fence",
        "mutation_guard",
    }

    for toolkit in _toolkits().values():
        for tool in toolkit.tools.values():
            assert forbidden.isdisjoint(inspect.signature(tool.func).parameters)


def test_complete_model_tool_json_contract_is_frozen():
    toolkits = _toolkits()
    expected_descriptions = {**BASE_DESCRIPTIONS, **ROLE_DESCRIPTIONS}
    special_property_descriptions = {
        ("context_content_read", "ref"): "Previously disclosed content ref",
        ("context_content_read", "offset"): "Byte offset: integer 0..33554432, default 0",
        ("context_content_read", "limit"): "Page bytes: integer 1..8192, default 8192",
        ("memory_list", "path"): (
            "Virtual folder path; never use a host filesystem path."
        ),
        ("memory_list", "recursive"): "Include descendants when true.",
        ("memory_list", "limit"): ("Maximum number of entries returned (up to 200)."),
    }

    for role, toolkit in toolkits.items():
        for name, tool in toolkit.tools.items():
            description = expected_descriptions[name]
            if role == "task_state" and name == "memory_source_read":
                description = TASK_STATE_SOURCE_DESCRIPTION
            if role == "task_state" and name == "memory_update_task_state":
                description = TASK_STATE_UPDATE_DESCRIPTION
            parameter_types, required = PARAMETERS[name]
            properties = {}
            for parameter_name, parameter_type in parameter_types.items():
                property_schema = {
                    "type": parameter_type,
                    "description": special_property_descriptions.get(
                        (name, parameter_name),
                        f"Argument {parameter_name}",
                    ),
                }
                if parameter_type == "array":
                    property_schema["items"] = {"type": "string"}
                properties[parameter_name] = property_schema

            assert tool.to_json() == {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            }


def test_system_toolkit_metadata_and_readme_are_packaged_without_public_manifest():
    repository_root = Path(__file__).resolve().parents[2]
    package_root = repository_root / "src" / "unchain" / "memory" / "toolkit"
    configuration = tomllib.loads(
        (repository_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    package_data = configuration["tool"]["setuptools"]["package-data"]["unchain"]

    assert MEMORY_TOOLKIT_METADATA.package_id == "memory_v2_system"
    assert MEMORY_TOOLKIT_METADATA.public_registry is False
    assert MEMORY_TOOLKIT_METADATA.capability_profiles == (
        "agent_read_propose",
        "workspace_curator",
        "consolidation_curator",
        "task_state_curator",
    )
    assert "memory/toolkit/README.md" in package_data
    assert (package_root / "README.md").is_file()
    assert not (package_root / "toolkit.toml").exists()


def test_host_dialect_is_immutable_and_does_not_mutate_the_generic_default():
    assert HOST_DIALECT.contract_id == "frozen.host.memory_toolkit.p0"
    assert DEFAULT_MEMORY_TOOLKIT_DIALECT.contract_id == "unchain.memory_toolkit.v1"
    assert "pupu://" not in DEFAULT_MEMORY_TOOLKIT_DIALECT.description("memory_read")

    try:
        HOST_DIALECT.descriptions["memory_read"] = "changed"
    except TypeError:
        pass
    else:  # pragma: no cover - documents the immutable contract boundary
        raise AssertionError("host dialect descriptions must be immutable")
