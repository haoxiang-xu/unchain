from __future__ import annotations

import hashlib

import pytest

from unchain.journal import ResourceRef
from unchain.memory.toolkit import (
    CuratorMemoryToolkitCapabilities,
    MemoryToolContentPage,
    MemoryToolkitError,
    NormalMemoryToolkitCapabilities,
    TaskStateMemoryToolkitCapabilities,
    build_memory_toolkit,
)

from .test_memory_toolkit_security import (
    FakeCandidates,
    FakeChat,
    FakeCodec,
    FakeContext,
    binding,
    invoke,
)
from .test_memory_toolkit_contract import HOST_DIALECT
from .test_memory_toolkit_service_capabilities import workspace_capability


class FakeCurator(FakeChat):
    def __init__(self, **options) -> None:
        super().__init__(**options)
        self.calls = []
        self.operations = {}
        self.entries = {
            "entry-1": {
                "entry_id": "entry-1",
                "space_id": self.space_id,
                "path": "/decisions/model.md",
                "name": "model.md",
                "kind": "markdown",
                "description": "Model choice and the reason behind it",
                "mime_type": "text/markdown",
                "revision": 3,
            }
        }

    def get_entry(self, *, ref):
        self.calls.append(("get_entry", ref))
        return {**self.entries[ref.resource_id], "revision": ref.revision}

    def upsert(self, *, request):
        self.calls.append(("upsert", request))
        payload = (
            request.path,
            request.description,
            request.expected_space_revision,
            request.entry_ref,
            request.kind,
            request.content,
            request.media_type,
            request.url,
            request.source_refs,
        )
        previous = self.operations.get(request.operation_id)
        if previous is not None and previous[0] != payload:
            raise RuntimeError("operation payload changed")
        if previous is not None:
            return previous[1]
        entry_id = (
            request.entry_ref.resource_id if request.entry_ref else "entry-created"
        )
        revision = request.entry_ref.revision + 1 if request.entry_ref else 1
        result = {
            "entry_ref": ResourceRef("memory", entry_id, revision, self.space_id),
            "path": request.path,
            "kind": request.kind,
            "space_revision": request.expected_space_revision + 1,
        }
        self.operations[request.operation_id] = (payload, result)
        return result

    def move(self, *, ref, new_path, expected_space_revision, operation_id):
        self.calls.append(("move", ref, operation_id))
        return {
            "entry_ref": ResourceRef(
                "memory", ref.resource_id, ref.revision + 1, self.space_id
            ),
            "path": new_path,
            "space_revision": expected_space_revision + 1,
        }

    def archive(self, *, ref, expected_space_revision, recursive, operation_id):
        self.calls.append(("archive", ref, recursive, operation_id))
        return {
            "entry_ref": ResourceRef(
                "memory", ref.resource_id, ref.revision + 1, self.space_id
            ),
            "deleted": True,
            "space_revision": expected_space_revision + 1,
        }

    def history(self, *, ref, limit):
        self.calls.append(("history", ref, limit))
        return tuple(
            {
                "entry_ref": ResourceRef(
                    "memory", ref.resource_id, revision, self.space_id
                ),
                "revision": revision,
            }
            for revision in range(ref.revision, max(0, ref.revision - limit), -1)
        )


class FakeLongTerm(FakeChat):
    def __init__(self) -> None:
        super().__init__(space_id="space-long", payload=b"long-term memory")

    def list_entries(self, **_arguments):
        return {
            "entries": [
                {
                    "entry_ref": ResourceRef("memory", "long-1", 1, self.space_id),
                    "path": "/preferences/provider.md",
                    "name": "provider.md",
                    "description": "Preferred provider",
                }
            ],
            "truncated": False,
        }

    def search_entries(self, **arguments):
        return {
            "query": arguments["query"],
            "backend": "fts5",
            "vector_status": "ready",
            "results": self.list_entries()["entries"],
        }

    def get_entry(self, *, ref):
        if ref.fragment != self.space_id:
            raise RuntimeError("foreign")
        return {
            "entry_id": ref.resource_id,
            "space_id": self.space_id,
            "path": "/preferences/provider.md",
            "kind": "markdown",
            "revision": ref.revision,
        }


class FakePromotions:
    def __init__(self) -> None:
        self.binding_id = "binding-1"
        self.target_namespace = "user-1"
        self.calls = []
        self.decide_calls = 0

    def propose(self, **arguments):
        self.calls.append(arguments)
        return {
            "owner_chat_id": "must-not-leak",
            "promotion_id": "promotion-1",
            "status": "pending",
        }

    def decide(self, **_arguments):
        self.decide_calls += 1
        raise AssertionError("toolkit cannot decide promotions")


class FakeTaskState:
    def __init__(self) -> None:
        self.binding_id = "binding-1"
        self.calls = []
        self.operations = {}

    def update(self, *, request):
        self.calls.append(request)
        payload = (request.expected_revision, request.patch, request.source_refs)
        previous = self.operations.get(request.operation_id)
        if previous is not None and previous[0] != payload:
            raise RuntimeError("operation payload changed")
        if previous is not None:
            return previous[1]
        result = {
            **request.patch,
            "revision": request.expected_revision + 1,
            "source_event_refs": request.source_refs,
        }
        self.operations[request.operation_id] = (payload, result)
        return result


def curator_toolkit(*, long_term=None, promotions=None, task_state=None):
    codec = FakeCodec()
    context = FakeContext()
    chat = FakeCurator()
    task_state = task_state or FakeTaskState()
    toolkit = build_memory_toolkit(
        binding(),
        CuratorMemoryToolkitCapabilities(
            references=codec,
            context=context,
            chat=chat,
            task_state=task_state,
            promotions=promotions,
            long_term=long_term,
        ),
        dialect=HOST_DIALECT,
    )
    return toolkit, chat, context, task_state, codec


def test_curator_upsert_passes_structured_refs_and_replays_same_operation():
    toolkit, chat, context, _, _ = curator_toolkit()
    event = ResourceRef("context_event", "event-2", 1)
    context.source_refs.add(event)
    arguments = {
        "path": "/constraints/runtime.md",
        "description": "Runtime constraints that must remain true during implementation",
        "expected_space_revision": 1,
        "content": "No silent provider fallback.",
        "source_ref": "pupu://context/event/event-2",
    }

    first = invoke(toolkit, "memory_upsert", **arguments)
    second = invoke(toolkit, "memory_upsert", **arguments)

    assert first == second
    requests = [value for name, value in chat.calls if name == "upsert"]
    assert len(requests) == 2
    assert requests[0].source_refs == (event,)
    assert requests[0].entry_ref is None
    assert requests[0].operation_id == requests[1].operation_id
    assert first["entry_ref"] == "pupu://memory/space-chat/entry-created@1"

    invoke(toolkit, "memory_upsert", **{**arguments, "content": "changed"})
    requests = [value for name, value in chat.calls if name == "upsert"]
    assert requests[-1].operation_id != requests[0].operation_id


def test_curator_update_move_supersede_link_and_archive_are_chat_scoped():
    toolkit, chat, context, _, _ = curator_toolkit()
    event = ResourceRef("context_event", "event-3", 1)
    context.source_refs.add(event)
    entry_ref = "pupu://memory/space-chat/entry-1@3"

    updated = invoke(
        toolkit,
        "memory_upsert",
        path="/decisions/model.md",
        description="Updated model choice and the reason behind it",
        expected_space_revision=1,
        entry_ref=entry_ref,
        content="Use the selected model.",
        source_ref="pupu://context/event/event-3",
    )
    assert updated["entry_ref"] == "pupu://memory/space-chat/entry-1@4"
    request = [value for name, value in chat.calls if name == "upsert"][-1]
    assert request.entry_ref == ResourceRef("memory", "entry-1", 3, "space-chat")

    moved = invoke(
        toolkit,
        "memory_move",
        entry_ref=entry_ref,
        new_path="/decisions/provider.md",
        expected_space_revision=2,
    )
    assert moved["path"] == "/decisions/provider.md"

    linked = invoke(
        toolkit,
        "memory_link",
        path="/references/provider-docs.link",
        description="Provider documentation used during implementation",
        url="https://example.com/docs",
        expected_space_revision=3,
        source_ref="pupu://context/event/event-3",
    )
    assert linked["kind"] == "link"

    superseded = invoke(
        toolkit,
        "memory_supersede",
        entry_ref=entry_ref,
        expected_space_revision=4,
        description="Replacement model choice and current rationale",
        content="Use the replacement model.",
    )
    assert superseded["entry_ref"] == "pupu://memory/space-chat/entry-1@4"

    archived = invoke(
        toolkit,
        "memory_archive",
        entry_ref=entry_ref,
        expected_space_revision=5,
    )
    assert archived["deleted"] is True

    for call in chat.calls:
        if call[0] in {"move", "archive"}:
            assert call[1].fragment == "space-chat"

    with pytest.raises(
        MemoryToolkitError,
        match="^memory ref is outside this toolkit's bound scope$",
    ):
        invoke(
            toolkit,
            "memory_move",
            entry_ref="pupu://memory/space-foreign/entry-1@3",
            new_path="/decisions/provider.md",
            expected_space_revision=6,
        )


def test_real_workspace_entries_are_projected_for_all_curator_entry_results():
    chat, _, _, _ = workspace_capability()
    codec = FakeCodec()
    toolkit = build_memory_toolkit(
        binding(),
        CuratorMemoryToolkitCapabilities(
            references=codec,
            context=FakeContext(),
            chat=chat,
            task_state=FakeTaskState(),
            promotions=None,
            long_term=None,
        ),
        dialect=HOST_DIALECT,
    )

    def assert_public_entry(value, expected_ref):
        assert value["entry_ref"] == expected_ref
        assert "content_ref" not in value
        assert "schema" not in value

    created = invoke(
        toolkit,
        "memory_upsert",
        path="/facts/release-window.md",
        description="Release timing used when planning the rollout",
        expected_space_revision=1,
        content="Ship after the final verification.",
    )
    ref_1 = "pupu://memory/space-chat/" + created["entry_id"] + "@1"
    assert_public_entry(created, ref_1)

    listing = invoke(toolkit, "memory_list")
    assert_public_entry(listing["entries"][0], ref_1)
    search = invoke(toolkit, "memory_search", query="release")
    assert_public_entry(search["results"][0]["entry"], ref_1)

    moved = invoke(
        toolkit,
        "memory_move",
        entry_ref=ref_1,
        new_path="/facts/release-date.md",
        expected_space_revision=2,
    )
    ref_2 = "pupu://memory/space-chat/" + created["entry_id"] + "@2"
    assert_public_entry(moved, ref_2)

    superseded = invoke(
        toolkit,
        "memory_supersede",
        entry_ref=ref_2,
        expected_space_revision=3,
        description="Updated release timing after final verification",
        content="Ship after the final verification passes.",
    )
    ref_3 = "pupu://memory/space-chat/" + created["entry_id"] + "@3"
    assert_public_entry(superseded, ref_3)

    archived = invoke(
        toolkit,
        "memory_archive",
        entry_ref=ref_3,
        expected_space_revision=4,
    )
    ref_4 = "pupu://memory/space-chat/" + created["entry_id"] + "@4"
    assert_public_entry(archived, ref_4)

    history = invoke(toolkit, "memory_history", entry_ref=ref_4, limit=4)
    assert [revision["entry_ref"] for revision in history["revisions"]] == [
        ref_4,
        ref_3,
        ref_2,
        ref_1,
    ]
    for revision in history["revisions"]:
        assert "content_ref" not in revision
        assert "schema" not in revision


@pytest.mark.parametrize(
    ("entry", "message"),
    (
        (
            {
                "entry_ref": ResourceRef(
                    "memory", "entry-foreign", 1, "space-foreign"
                )
            },
            "outside this toolkit's bound scope",
        ),
        (
            {
                "entry_ref": ResourceRef(
                    "memory", "entry-canonical", 1, "space-chat"
                ),
                "entry_id": "entry-divergent",
            },
            "diverges from its canonical entry ref",
        ),
    ),
)
def test_entry_projection_rejects_foreign_or_divergent_capability_results(
    entry,
    message,
):
    toolkit, chat, _, _, _ = curator_toolkit()
    chat.list_entries = lambda **_arguments: {
        "entries": [entry],
        "truncated": False,
    }

    with pytest.raises(MemoryToolkitError, match=message):
        invoke(toolkit, "memory_list")


@pytest.mark.parametrize(
    ("tool_name", "arguments", "capability_method"),
    (
        (
            "memory_upsert",
            {
                "path": "/decisions/model.md",
                "description": "Updated model choice and rationale",
                "expected_space_revision": 1,
                "entry_ref": "pupu://memory/space-chat/entry-1@3",
                "content": "Use the selected model.",
            },
            "upsert",
        ),
        (
            "memory_move",
            {
                "entry_ref": "pupu://memory/space-chat/entry-1@3",
                "new_path": "/decisions/provider.md",
                "expected_space_revision": 1,
            },
            "move",
        ),
        (
            "memory_supersede",
            {
                "entry_ref": "pupu://memory/space-chat/entry-1@3",
                "expected_space_revision": 1,
                "description": "Replacement model choice and rationale",
                "content": "Use the replacement model.",
            },
            "upsert",
        ),
        (
            "memory_archive",
            {
                "entry_ref": "pupu://memory/space-chat/entry-1@3",
                "expected_space_revision": 1,
            },
            "archive",
        ),
    ),
)
def test_entry_mutations_reject_results_for_a_different_entry(
    tool_name,
    arguments,
    capability_method,
):
    toolkit, chat, _, _, _ = curator_toolkit()
    setattr(
        chat,
        capability_method,
        lambda **_arguments: {
            "entry_ref": ResourceRef(
                "memory", "different-entry", 4, "space-chat"
            )
        },
    )

    with pytest.raises(
        MemoryToolkitError,
        match="^memory entry result does not match the requested entry$",
    ):
        invoke(toolkit, tool_name, **arguments)


@pytest.mark.parametrize("revision", (2, 3, 7))
def test_entry_mutations_reject_an_unexpected_result_revision(revision):
    toolkit, chat, _, _, _ = curator_toolkit()
    chat.move = lambda **_arguments: {
        "entry_ref": ResourceRef("memory", "entry-1", revision, "space-chat")
    }

    with pytest.raises(
        MemoryToolkitError,
        match="^memory entry result has unexpected revision$",
    ):
        invoke(
            toolkit,
            "memory_move",
            entry_ref="pupu://memory/space-chat/entry-1@3",
            new_path="/decisions/provider.md",
            expected_space_revision=1,
        )


def test_entry_creation_requires_the_first_revision():
    toolkit, chat, _, _, _ = curator_toolkit()
    chat.upsert = lambda **_arguments: {
        "entry_ref": ResourceRef(
            "memory", "entry-created", 2, "space-chat"
        )
    }

    with pytest.raises(
        MemoryToolkitError,
        match="^memory entry result has unexpected revision$",
    ):
        invoke(
            toolkit,
            "memory_upsert",
            path="/facts/release-window.md",
            description="Release timing used when planning the rollout",
            expected_space_revision=1,
            content="Ship after verification.",
        )


@pytest.mark.parametrize(
    ("revisions", "limit", "message"),
    (
        (
            (
                {
                    "entry_ref": ResourceRef(
                        "memory", "different-entry", 3, "space-chat"
                    )
                },
            ),
            2,
            "memory entry result does not match the requested entry",
        ),
        (
            tuple(
                {
                    "entry_ref": ResourceRef(
                        "memory", "entry-1", revision, "space-chat"
                    )
                }
                for revision in (3, 2, 1)
            ),
            2,
            "memory history exceeded the requested limit",
        ),
        (
            tuple(
                {
                    "entry_ref": ResourceRef(
                        "memory", "entry-1", revision, "space-chat"
                    )
                }
                for revision in (2, 3)
            ),
            2,
            "memory history is not in revision order",
        ),
    ),
)
def test_memory_history_rejects_uncorrelated_or_unbounded_results(
    revisions,
    limit,
    message,
):
    toolkit, chat, _, _, _ = curator_toolkit()
    chat.history = lambda **_arguments: revisions

    with pytest.raises(MemoryToolkitError, match=f"^{message}$"):
        invoke(
            toolkit,
            "memory_history",
            entry_ref="pupu://memory/space-chat/entry-1@3",
            limit=limit,
        )


def test_upsert_cannot_change_path_or_kind_and_link_validation_is_stable():
    toolkit, _, _, _, _ = curator_toolkit()
    entry_ref = "pupu://memory/space-chat/entry-1@3"

    with pytest.raises(
        MemoryToolkitError,
        match="^use memory_move to change an entry path$",
    ):
        invoke(
            toolkit,
            "memory_upsert",
            path="/decisions/other.md",
            description="Another path for an existing decision",
            expected_space_revision=1,
            entry_ref=entry_ref,
            content="content",
        )
    with pytest.raises(
        MemoryToolkitError,
        match="^memory_upsert cannot change an entry kind$",
    ):
        invoke(
            toolkit,
            "memory_upsert",
            path="/decisions/model.md",
            description="Another representation of an existing decision",
            expected_space_revision=1,
            entry_ref=entry_ref,
            kind="link",
            url="https://example.com",
        )
    with pytest.raises(
        MemoryToolkitError,
        match="^link URL must use http or https$",
    ):
        invoke(
            toolkit,
            "memory_link",
            path="/references/docs.link",
            description="Documentation used to implement the provider",
            url="file:///etc/passwd",
            expected_space_revision=1,
        )


def test_promotion_remains_a_proposal_and_target_is_bound_long_term():
    long_term = FakeLongTerm()
    promotions = FakePromotions()
    toolkit, _, _, _, _ = curator_toolkit(
        long_term=long_term,
        promotions=promotions,
    )

    result = invoke(
        toolkit,
        "memory_promote",
        source_ref="pupu://memory/space-chat/entry-1@3",
        target_path="/preferences/model-provider.md",
        target_entry_ref="pupu://memory/space-long/long-1@1",
    )

    assert result["status"] == "pending"
    assert result["requires_user_confirmation"] is True
    assert promotions.decide_calls == 0
    call = promotions.calls[0]
    assert call["source_ref"] == ResourceRef("memory", "entry-1", 3, "space-chat")
    assert call["target_entry_ref"] == ResourceRef("memory", "long-1", 1, "space-long")

    with pytest.raises(
        MemoryToolkitError,
        match="^target_entry_ref must be in bound long-term memory$",
    ):
        invoke(
            toolkit,
            "memory_promote",
            source_ref="pupu://memory/space-chat/entry-1@3",
            target_path="/preferences/model-provider.md",
            target_entry_ref="pupu://memory/space-chat/entry-1@3",
        )

    unavailable, *_ = curator_toolkit()
    with pytest.raises(
        MemoryToolkitError,
        match="^a server-bound long-term namespace is required$",
    ):
        invoke(
            unavailable,
            "memory_promote",
            source_ref="pupu://memory/space-chat/entry-1@3",
            target_path="/preferences/model-provider.md",
        )


def test_curator_list_search_read_and_history_merge_only_bound_spaces():
    long_term = FakeLongTerm()
    toolkit, chat, _, _, _ = curator_toolkit(long_term=long_term)
    chat.list_entries = lambda **_arguments: {
        "entries": [
            {
                "entry_ref": ResourceRef("memory", "chat-1", 1, "space-chat"),
                "path": "/decisions/model.md",
            }
        ],
        "truncated": False,
    }
    chat.search_entries = lambda **arguments: {
        "query": arguments["query"],
        "backend": "fts5",
        "vector_status": "degraded",
        "results": chat.list_entries()["entries"],
    }

    listing = invoke(toolkit, "memory_list")
    assert [entry["scope_kind"] for entry in listing["entries"]] == [
        "chat",
        "long_term",
    ]
    assert [space["space_id"] for space in listing["spaces"]] == [
        "space-chat",
        "space-long",
    ]

    search = invoke(toolkit, "memory_search", query="provider")
    assert [entry["scope_kind"] for entry in search["results"]] == [
        "chat",
        "long_term",
    ]

    long_page = invoke(
        toolkit,
        "memory_source_read",
        ref="pupu://memory/space-long/long-1@1",
        limit=64,
    )
    assert long_page["text"] == "long-term memory"
    assert long_term.read_refs[-1] == ResourceRef("memory", "long-1", 1, "space-long")

    history = invoke(
        toolkit,
        "memory_history",
        entry_ref="pupu://memory/space-chat/entry-1@3",
        limit=2,
    )
    assert [revision["revision"] for revision in history["revisions"]] == [3, 2]
    assert history["truncated"] is True
    assert history["next_revision"] == 1


def test_normal_long_term_read_is_limited_to_predecoded_recalled_refs():
    codec = FakeCodec()
    context = FakeContext()
    allowed = ResourceRef("memory", "long-1", 1, "space-long")
    long_term = FakeLongTerm()
    toolkit = build_memory_toolkit(
        binding(),
        NormalMemoryToolkitCapabilities(
            references=codec,
            context=context,
            chat=FakeChat(),
            candidates=FakeCandidates(),
            long_term=long_term,
            allowed_long_term_refs=(allowed,),
        ),
        dialect=HOST_DIALECT,
    )

    assert (
        invoke(
            toolkit,
            "memory_read",
            ref="pupu://memory/space-long/long-1@1",
            limit=64,
        )["text"]
        == "long-term memory"
    )
    with pytest.raises(
        MemoryToolkitError,
        match="^memory ref is outside this toolkit's bound scope$",
    ):
        invoke(
            toolkit,
            "memory_read",
            ref="pupu://memory/space-long/long-2@1",
            limit=64,
        )


def test_task_state_curator_converts_external_refs_before_idempotent_capability_update():
    codec = FakeCodec()
    context = FakeContext()
    task_state = FakeTaskState()
    event = ResourceRef("context_event", "event-9", 1)
    context.source_refs.update({event, ResourceRef("artifact", "artifact-1", 1)})
    toolkit = build_memory_toolkit(
        binding(),
        TaskStateMemoryToolkitCapabilities(
            references=codec,
            context=context,
            chat=FakeChat(),
            task_state=task_state,
        ),
        dialect=HOST_DIALECT,
    )
    patch = {
        "objective": "Ship Memory V2 P0 safely",
        "success_criteria": ["Tool results survive restart"],
        "constraints": ["Never store secret plaintext"],
        "confirmed_decisions": ["Use one canonical journal"],
        "open_questions": ["When should canary advance?"],
        "active_plan": ["Run the recovery matrix"],
        "artifact_memory_refs": [
            "pupu://artifact/artifact-1@1",
            "pupu://memory/space-chat/entry-1@1",
        ],
    }

    first = invoke(
        toolkit,
        "memory_update_task_state",
        expected_revision=3,
        patch=patch,
        source_refs=["pupu://context/event/event-9"],
    )
    second = invoke(
        toolkit,
        "memory_update_task_state",
        expected_revision=3,
        patch=patch,
        source_refs=["pupu://context/event/event-9"],
    )

    assert first == second
    request = task_state.calls[0]
    assert request.patch["artifact_refs"] == (ResourceRef("artifact", "artifact-1", 1),)
    assert request.patch["memory_refs"] == (
        ResourceRef("memory", "entry-1", 1, "space-chat"),
    )
    assert request.source_refs == (event,)
    assert request.operation_id == task_state.calls[1].operation_id
    assert first["artifact_refs"] == ["pupu://artifact/artifact-1@1"]

    with pytest.raises(
        MemoryToolkitError,
        match="^source_refs must include at least one journal event reference$",
    ):
        invoke(
            toolkit,
            "memory_update_task_state",
            expected_revision=4,
            patch={"constraints": ["source required"]},
            source_refs=[],
        )
    with pytest.raises(
        MemoryToolkitError,
        match=(
            "^artifact_memory_refs must contain revisioned pupu://artifact or "
            "pupu://memory references$"
        ),
    ):
        invoke(
            toolkit,
            "memory_update_task_state",
            expected_revision=4,
            patch={"artifact_memory_refs": ["pupu://context/event/event-9"]},
            source_refs=["pupu://context/event/event-9"],
        )


def test_context_source_read_requires_bound_authorization():
    toolkit, _, context, _, _ = curator_toolkit()
    artifact = ResourceRef("artifact", "artifact-1", 1)
    context.source_refs.add(artifact)

    page = invoke(
        toolkit,
        "memory_source_read",
        ref="pupu://artifact/artifact-1@1",
        limit=64,
    )
    assert page["text"] == "hello context"
    assert context.calls[-1] == ("read_content", artifact)

    with pytest.raises(
        MemoryToolkitError,
        match="^source ref is outside this toolkit's bound scope$",
    ):
        invoke(
            toolkit,
            "memory_source_read",
            ref="pupu://artifact/foreign@1",
        )
