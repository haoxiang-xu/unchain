from __future__ import annotations

import base64
import hashlib
import re

import pytest

from unchain.journal import ResourceRef
from unchain.memory.curator import CuratorLeaseFence, FenceBoundConsolidationToolkit
from unchain.memory.toolkit import (
    ConsolidationMemoryToolkitCapabilities,
    MemoryToolContentPage,
    MemoryToolkitError,
    MemoryToolkitRunBinding,
    NormalMemoryToolkitCapabilities,
    ReferencePurpose,
    build_memory_toolkit,
)

from .test_memory_toolkit_contract import HOST_DIALECT
from unchain.memory.toolkit.validation import (
    encode_context_content_locator,
    mutation_id,
)


MEMORY_RE = re.compile(
    r"^pupu://memory/([A-Za-z0-9._:-]+)/([A-Za-z0-9._:-]+)@([1-9][0-9]*)$"
)
CANDIDATE_RE = re.compile(r"^pupu://memory/candidate/([A-Za-z0-9._:-]+)@([1-9][0-9]*)$")
ARTIFACT_RE = re.compile(r"^pupu://artifact/([A-Za-z0-9._:-]+)@([1-9][0-9]*)$")
EVENT_RE = re.compile(r"^pupu://context/event/([A-Za-z0-9._:-]+)(?:/(content))?$")
CHECKPOINT_RE = re.compile(
    r"^pupu://context/checkpoint/([A-Za-z0-9._:-]+)(?:/event/([1-9][0-9]*))?$"
)


class FakeCodec:
    def __init__(self, binding_id: str = "binding-1") -> None:
        self.binding_id = binding_id
        self.decoded: list[tuple[str, ReferencePurpose, ResourceRef]] = []

    def decode(self, value: str, *, purpose: ReferencePurpose) -> ResourceRef:
        match = CANDIDATE_RE.fullmatch(value)
        if match:
            ref = ResourceRef("memory_candidate", match.group(1), int(match.group(2)))
        else:
            match = MEMORY_RE.fullmatch(value)
            if match:
                ref = ResourceRef(
                    "memory", match.group(2), int(match.group(3)), match.group(1)
                )
            else:
                match = ARTIFACT_RE.fullmatch(value)
                if match:
                    ref = ResourceRef("artifact", match.group(1), int(match.group(2)))
                else:
                    match = EVENT_RE.fullmatch(value)
                    if match:
                        ref = ResourceRef(
                            "context_event", match.group(1), 1, match.group(2) or ""
                        )
                    else:
                        match = CHECKPOINT_RE.fullmatch(value)
                        if match:
                            fragment = (
                                f"event/{match.group(2)}" if match.group(2) else ""
                            )
                            ref = ResourceRef("checkpoint", match.group(1), 1, fragment)
                        else:
                            raise ValueError("PuPu reference is invalid or unsupported")
        self.decoded.append((value, purpose, ref))
        return ref

    def encode(self, ref: ResourceRef) -> str:
        if ref.kind == "memory":
            return f"pupu://memory/{ref.fragment}/{ref.resource_id}@{ref.revision}"
        if ref.kind == "memory_candidate":
            return f"pupu://memory/candidate/{ref.resource_id}@{ref.revision}"
        if ref.kind == "artifact":
            return f"pupu://artifact/{ref.resource_id}@{ref.revision}"
        if ref.kind == "context_event":
            suffix = "/content" if ref.fragment == "content" else ""
            return f"pupu://context/event/{ref.resource_id}{suffix}"
        if ref.kind == "checkpoint":
            suffix = f"/{ref.fragment}" if ref.fragment else ""
            return f"pupu://context/checkpoint/{ref.resource_id}{suffix}"
        raise ValueError("unsupported test ref")


class FakeContext:
    def __init__(self, binding_id: str = "binding-1") -> None:
        self.binding_id = binding_id
        self.disclosed: set[ResourceRef] = set()
        self.source_refs: set[ResourceRef] = set()
        self.calls: list[tuple[str, object]] = []
        self.payload = b"hello context"
        self.total_bytes = len(self.payload)

    def authorize(self, *, ref: ResourceRef, purpose: ReferencePurpose) -> ResourceRef:
        self.calls.append(("authorize", (ref, purpose)))
        allowed = (
            self.disclosed
            if purpose
            in {
                ReferencePurpose.CONTEXT_CONTENT,
                ReferencePurpose.CHECKPOINT,
            }
            else self.source_refs
        )
        if ref not in allowed:
            raise PermissionError("foreign")
        return ref

    def read_content(
        self, *, ref: ResourceRef, offset: int, limit: int
    ) -> MemoryToolContentPage:
        self.calls.append(("read_content", ref))
        data = self.payload[offset : offset + limit]
        return MemoryToolContentPage(
            ref=ref,
            media_type="text/plain",
            data=data,
            offset=offset,
            total_bytes=self.total_bytes,
            sha256=hashlib.sha256(self.payload).hexdigest(),
        )

    def read_checkpoint_events(
        self, *, ref: ResourceRef, after_position: int, limit: int
    ):
        self.calls.append(("read_checkpoint_events", ref))
        return {
            "owner_chat_id": "must-not-leak",
            "checkpoint_ref": ref,
            "coverage": {"ceiling_position": 2},
            "after_position": after_position,
            "next_after_position": after_position + 1,
            "has_more": True,
            "events": [{"position": after_position + 1}],
        }


class FakeChat:
    def __init__(
        self,
        binding_id: str = "binding-1",
        space_id: str = "space-chat",
        payload: bytes = b"hello memory",
    ) -> None:
        self.binding_id = binding_id
        self.space_id = space_id
        self.space_revision = 1
        self.payload = payload
        self.read_refs: list[ResourceRef] = []

    def list_entries(self, **_arguments):
        return {"entries": [], "truncated": False}

    def search_entries(self, **arguments):
        return {
            "query": arguments["query"],
            "backend": "fts5",
            "vector_status": "degraded",
            "results": [],
        }

    def read_content(
        self, *, ref: ResourceRef, offset: int, limit: int
    ) -> MemoryToolContentPage:
        self.read_refs.append(ref)
        return MemoryToolContentPage(
            ref=ref,
            media_type="text/markdown",
            data=self.payload[offset : offset + limit],
            offset=offset,
            total_bytes=len(self.payload),
            sha256=hashlib.sha256(self.payload).hexdigest(),
        )


class FakeCandidates:
    def __init__(self, binding_id: str = "binding-1") -> None:
        self.binding_id = binding_id
        self.calls = []
        self.operations = {}

    def propose(self, *, request):
        self.calls.append(request)
        previous = self.operations.get(request.operation_id)
        payload = (
            request.path,
            request.description,
            request.kind,
            request.content,
            request.media_type,
            request.url,
            request.source_refs,
            request.rationale,
            request.confidence,
            request.sensitivity,
        )
        if previous is not None and previous != payload:
            raise RuntimeError("operation payload changed")
        self.operations[request.operation_id] = payload
        return {
            "owner_chat_id": "must-not-leak",
            "candidate_ref": ResourceRef("memory_candidate", "candidate-1", 1),
            "status": "pending",
        }


class FakeConsolidation:
    def __init__(self, binding_id: str = "binding-1") -> None:
        self.binding_id = binding_id
        self.calls = []
        self.payload = b"candidate content"

    def read_candidate(self, *, job_id, ref, offset, limit):
        self.calls.append(("read_candidate", job_id, ref))
        return MemoryToolContentPage(
            ref=ref,
            media_type="text/markdown",
            data=self.payload[offset : offset + limit],
            offset=offset,
            total_bytes=len(self.payload),
            sha256=hashlib.sha256(self.payload).hexdigest(),
        )

    def apply_new(self, **arguments):
        self.calls.append(("apply_new", arguments))
        return {
            "outcome": "applied",
            "candidate_ref": arguments["candidate_ref"],
            "lease_fence": arguments.get("mutation_guard").fence,
            "mutation_guard": arguments.get("mutation_guard"),
        }

    def propose_review(self, **arguments):
        self.calls.append(("propose_review", arguments))
        return {
            "outcome": "awaiting_user",
            "candidate_ref": arguments["candidate_ref"],
            "lease_fence": arguments.get("mutation_guard").fence,
            "mutation_guard": arguments.get("mutation_guard"),
        }


class BindingOnlyConsolidation:
    binding_id = "binding-1"


class MissingReviewConsolidation(FakeConsolidation):
    propose_review = None


class NonCallableReadConsolidation(FakeConsolidation):
    read_candidate = "not-callable"


class MissingMutationGuardConsolidation(FakeConsolidation):
    def apply_new(
        self,
        *,
        job_id,
        candidate_ref,
        expected_binding_revision,
        expected_space_revision,
        operation_id,
    ):
        del (
            job_id,
            candidate_ref,
            expected_binding_revision,
            expected_space_revision,
            operation_id,
        )
        return {"outcome": "applied"}


class MissingReviewMutationGuardConsolidation(FakeConsolidation):
    def propose_review(
        self,
        *,
        job_id,
        candidate_ref,
        expected_binding_revision,
        target_entry_id,
        expected_target_revision,
        mode,
        operation_id,
    ):
        del (
            job_id,
            candidate_ref,
            expected_binding_revision,
            target_entry_id,
            expected_target_revision,
            mode,
            operation_id,
        )
        return {"outcome": "awaiting_user"}


class FakeMutationGuard:
    def __init__(self, fence: CuratorLeaseFence) -> None:
        self.fence = fence

    def assert_active(self) -> None:
        pass


def curator_fence(
    *,
    binding_id: str = "binding-1",
    job_id: str = "job-1",
    job_revision: int = 2,
) -> CuratorLeaseFence:
    return CuratorLeaseFence(
        binding_id=binding_id,
        job_id=job_id,
        job_revision=job_revision,
        lease_owner="worker-1",
        lease_token="lease-1",
    )


def consolidation_bundle(consolidation) -> ConsolidationMemoryToolkitCapabilities:
    fence = curator_fence()
    return ConsolidationMemoryToolkitCapabilities(
        binding_id="binding-1",
        references=FakeCodec(),
        context=FakeContext(),
        chat=FakeChat(),
        consolidation=consolidation,
        job_id="job-1",
        candidate_refs=(ResourceRef("memory_candidate", "candidate-1", 1),),
        lease_fence=fence,
        mutation_guard=FakeMutationGuard(fence),
    )


def binding(binding_id: str = "binding-1") -> MemoryToolkitRunBinding:
    return MemoryToolkitRunBinding(
        binding_id=binding_id,
        session_id="session-1",
        attempt_id="attempt-1",
        run_id="run-1",
    )


def normal_toolkit(*, chat=None, context=None, candidates=None, codec=None):
    chat = chat or FakeChat()
    context = context or FakeContext()
    candidates = candidates or FakeCandidates()
    codec = codec or FakeCodec()
    toolkit = build_memory_toolkit(
        binding(),
        NormalMemoryToolkitCapabilities(
            references=codec,
            context=context,
            chat=chat,
            candidates=candidates,
        ),
        dialect=HOST_DIALECT,
    )
    return toolkit, chat, context, candidates, codec


def invoke(toolkit, name, **arguments):
    return toolkit.tools[name].func(**arguments)


def test_context_reads_require_disclosure_and_pass_only_structured_refs_to_capability():
    context = FakeContext()
    artifact = ResourceRef("artifact", "artifact-1", 1)
    checkpoint = ResourceRef("checkpoint", "checkpoint-1", 1)
    context.disclosed.update({artifact, checkpoint})
    toolkit, _, _, _, codec = normal_toolkit(context=context)

    page = invoke(
        toolkit,
        "context_content_read",
        ref="pupu://artifact/artifact-1@1",
        offset=2,
        limit=7,
    )
    assert page["schema_version"] == "unchain.context_content_page.v1"
    assert page["trust"] == "UNTRUSTED_DATA"
    assert page["content"] == {
        "encoding": "utf-8",
        "text": "llo con",
    }
    assert page["sha256"] == hashlib.sha256(context.payload).hexdigest()
    assert page["page_bytes"] == 7
    assert page["next_read"]["tool"] == "context_content_read"
    assert codec.decoded[-1][1] is ReferencePurpose.CONTEXT_CONTENT
    assert context.calls[-1] == ("read_content", artifact)

    context.calls.clear()
    codec.decoded.clear()
    canonical_page = invoke(
        toolkit,
        "context_content_read",
        ref=encode_context_content_locator(artifact),
        offset=0,
        limit=7,
    )
    assert canonical_page["ref"].startswith("unchain://context/v1/")
    assert codec.decoded == []
    assert context.calls == [
        ("authorize", (artifact, ReferencePurpose.CONTEXT_CONTENT)),
        ("read_content", artifact),
    ]

    context.payload = b"x" * 20_000
    context.total_bytes = len(context.payload)
    context.calls.clear()
    default_page = invoke(
        toolkit,
        "context_content_read",
        ref=encode_context_content_locator(artifact),
    )
    assert default_page["page_bytes"] == 8 * 1024
    assert default_page["next_offset"] == 8 * 1024
    assert context.calls == [
        ("authorize", (artifact, ReferencePurpose.CONTEXT_CONTENT)),
        ("read_content", artifact),
    ]

    checkpoint_page = invoke(
        toolkit,
        "context_checkpoint_events_read",
        checkpoint_ref="pupu://context/checkpoint/checkpoint-1",
        limit=1,
    )
    assert checkpoint_page["schema_version"] == "context_checkpoint_events.v1"
    assert checkpoint_page["trust"] == "UNTRUSTED_DATA"
    assert "owner_chat_id" not in checkpoint_page
    assert checkpoint_page["checkpoint_ref"] == "pupu://context/checkpoint/checkpoint-1"

    with pytest.raises(
        MemoryToolkitError,
        match="^content ref was not disclosed to this agent context$",
    ):
        invoke(
            toolkit,
            "context_content_read",
            ref="pupu://artifact/not-disclosed@1",
        )


@pytest.mark.parametrize(
    ("returned_offset", "returned_bytes", "message"),
    [
        (0, 100, "requested range"),
        (50, 101, "requested range"),
        (50, 0, "cannot continue"),
    ],
)
def test_context_content_read_rejects_invalid_capability_pages(
    returned_offset: int,
    returned_bytes: int,
    message: str,
) -> None:
    context = FakeContext()
    artifact = ResourceRef("artifact", "artifact-1", 1)
    context.disclosed.add(artifact)
    context.payload = b"x" * 20_000
    context.total_bytes = len(context.payload)

    def invalid_read(*, ref, offset, limit):
        del offset, limit
        return MemoryToolContentPage(
            ref=ref,
            media_type="text/plain",
            data=b"x" * returned_bytes,
            offset=returned_offset,
            total_bytes=context.total_bytes,
            sha256=hashlib.sha256(context.payload).hexdigest(),
        )

    context.read_content = invalid_read
    toolkit, _, _, _, _ = normal_toolkit(context=context)

    with pytest.raises(MemoryToolkitError, match=message):
        invoke(
            toolkit,
            "context_content_read",
            ref=encode_context_content_locator(artifact),
            offset=50,
            limit=100,
        )


def test_context_content_read_stops_only_the_third_identical_nonprogress_read():
    context = FakeContext()
    artifact = ResourceRef("artifact", "artifact-1", 1)
    context.disclosed.add(artifact)
    locator = encode_context_content_locator(artifact)
    original_read = context.read_content

    def invalid_read(*, ref, offset, limit):
        del limit
        context.calls.append(("read_content", ref))
        return MemoryToolContentPage(
            ref=ref,
            media_type="text/plain",
            data=b"",
            offset=offset,
            total_bytes=2,
            sha256=hashlib.sha256(b"xx").hexdigest(),
        )

    context.read_content = invalid_read
    toolkit, _, _, _, _ = normal_toolkit(context=context)
    arguments = {"ref": locator, "offset": 0, "limit": 1}

    for _ in range(2):
        with pytest.raises(MemoryToolkitError, match="cannot continue"):
            invoke(toolkit, "context_content_read", **arguments)
    assert [name for name, _value in context.calls].count("read_content") == 2

    with pytest.raises(MemoryToolkitError, match="^CONTEXT_READ_NO_PROGRESS$"):
        invoke(toolkit, "context_content_read", **arguments)
    assert [name for name, _value in context.calls].count("read_content") == 2

    with pytest.raises(MemoryToolkitError, match="cannot continue"):
        invoke(
            toolkit,
            "context_content_read",
            **{**arguments, "offset": 1},
        )
    assert [name for name, _value in context.calls].count("read_content") == 3

    second_toolkit, _, _, _, _ = normal_toolkit(context=context)
    with pytest.raises(MemoryToolkitError, match="cannot continue"):
        invoke(second_toolkit, "context_content_read", **arguments)
    assert [name for name, _value in context.calls].count("read_content") == 4

    context.read_content = original_read
    result = invoke(second_toolkit, "context_content_read", **arguments)
    assert result["offset"] == 0

    context.read_content = invalid_read
    for _ in range(2):
        with pytest.raises(MemoryToolkitError, match="cannot continue"):
            invoke(second_toolkit, "context_content_read", **arguments)
    with pytest.raises(MemoryToolkitError, match="^CONTEXT_READ_NO_PROGRESS$"):
        invoke(second_toolkit, "context_content_read", **arguments)


def test_context_content_read_success_clears_another_request_failure_streak():
    context = FakeContext()
    failed = ResourceRef("artifact", "failed-artifact", 1)
    successful = ResourceRef("artifact", "successful-artifact", 1)
    context.disclosed.add(successful)
    toolkit, _, _, _, _ = normal_toolkit(context=context)
    failed_arguments = {
        "ref": encode_context_content_locator(failed),
        "offset": 0,
        "limit": 1,
    }

    for _ in range(2):
        with pytest.raises(MemoryToolkitError, match="disclosed"):
            invoke(toolkit, "context_content_read", **failed_arguments)

    assert invoke(
        toolkit,
        "context_content_read",
        ref=encode_context_content_locator(successful),
        offset=0,
        limit=1,
    )["page_bytes"] == 1

    context.disclosed.add(failed)
    assert invoke(toolkit, "context_content_read", **failed_arguments)["page_bytes"] == 1


@pytest.mark.parametrize(
    "changed_request",
    (
        {"ref": "other-artifact", "offset": 0, "limit": 1},
        {"ref": "failed-artifact", "offset": 1, "limit": 1},
        {"ref": "failed-artifact", "offset": 0, "limit": 2},
    ),
)
def test_context_content_read_changed_request_clears_failure_streak(
    changed_request: dict[str, object],
) -> None:
    context = FakeContext()
    failed = ResourceRef("artifact", "failed-artifact", 1)
    toolkit, _, _, _, _ = normal_toolkit(context=context)
    failed_arguments = {
        "ref": encode_context_content_locator(failed),
        "offset": 0,
        "limit": 1,
    }

    for _ in range(2):
        with pytest.raises(MemoryToolkitError, match="disclosed"):
            invoke(toolkit, "context_content_read", **failed_arguments)

    altered = dict(changed_request)
    altered["ref"] = encode_context_content_locator(
        ResourceRef("artifact", str(altered["ref"]), 1)
    )
    with pytest.raises(MemoryToolkitError, match="disclosed"):
        invoke(toolkit, "context_content_read", **altered)
    with pytest.raises(MemoryToolkitError, match="disclosed"):
        invoke(toolkit, "context_content_read", **failed_arguments)


def test_context_content_read_does_not_suppress_retryable_failures():
    class RetryableReadError(MemoryToolkitError):
        retryable = True

    context = FakeContext()
    artifact = ResourceRef("artifact", "artifact-1", 1)
    context.disclosed.add(artifact)

    def retryable_read(*, ref, offset, limit):
        del offset, limit
        context.calls.append(("read_content", ref))
        raise RetryableReadError("temporary context content failure")

    context.read_content = retryable_read
    toolkit, _, _, _, _ = normal_toolkit(context=context)
    arguments = {
        "ref": encode_context_content_locator(artifact),
        "offset": 0,
        "limit": 1,
    }

    for _ in range(3):
        with pytest.raises(MemoryToolkitError, match="temporary context content failure"):
            invoke(toolkit, "context_content_read", **arguments)
    assert [name for name, _value in context.calls].count("read_content") == 3


def test_context_content_read_does_not_suppress_retryable_authorization_failures():
    class RetryableAuthorizationError(MemoryToolkitError):
        retryable = True

    class RetryableAuthorizationContext(FakeContext):
        def __init__(self) -> None:
            super().__init__()
            self.authorization_attempts = 0

        def authorize(self, *, ref: ResourceRef, purpose: ReferencePurpose) -> ResourceRef:
            self.authorization_attempts += 1
            if self.authorization_attempts <= 2:
                raise RetryableAuthorizationError("temporary authorization failure")
            return super().authorize(ref=ref, purpose=purpose)

    context = RetryableAuthorizationContext()
    artifact = ResourceRef("artifact", "artifact-1", 1)
    context.disclosed.add(artifact)
    toolkit, _, _, _, _ = normal_toolkit(context=context)
    arguments = {"ref": encode_context_content_locator(artifact)}

    for _ in range(2):
        with pytest.raises(MemoryToolkitError, match="disclosed"):
            invoke(toolkit, "context_content_read", **arguments)

    result = invoke(toolkit, "context_content_read", **arguments)

    assert context.authorization_attempts == 3
    assert result["content"]["text"] == "hello context"


def test_checkpoint_derived_content_reuses_base_checkpoint_disclosure():
    context = FakeContext()
    checkpoint = ResourceRef("checkpoint", "checkpoint-1", 1)
    context.disclosed.add(checkpoint)
    toolkit, _, _, _, _ = normal_toolkit(context=context)

    result = invoke(
        toolkit,
        "context_content_read",
        ref="pupu://context/checkpoint/checkpoint-1/event/7",
        limit=4,
    )

    assert result["ref"].startswith("unchain://context/v1/")
    authorization = next(value for name, value in context.calls if name == "authorize")
    assert authorization[0] == checkpoint
    assert context.calls[-1][1] == ResourceRef(
        "checkpoint", "checkpoint-1", 1, "event/7"
    )


def test_memory_read_rejects_cross_scope_before_capability_and_bounds_full_reads():
    oversized = b"x" * (128 * 1024 + 1)
    chat = FakeChat(payload=oversized)
    toolkit, _, _, _, _ = normal_toolkit(chat=chat)

    with pytest.raises(
        MemoryToolkitError,
        match="^memory ref is outside this toolkit's bound scope$",
    ):
        invoke(
            toolkit,
            "memory_read",
            ref="pupu://memory/space-foreign/entry-1@1",
        )
    assert chat.read_refs == []

    refused = invoke(
        toolkit,
        "memory_read",
        ref="pupu://memory/space-chat/entry-1@1",
        full=True,
    )
    assert refused["full_read_allowed"] is False
    assert refused["max_full_read_bytes"] == 128 * 1024
    assert len(chat.read_refs) == 1


def test_normal_candidate_proposal_is_structured_idempotent_and_never_formal_write():
    candidates = FakeCandidates()
    toolkit, _, context, _, _ = normal_toolkit(candidates=candidates)
    event = ResourceRef("context_event", "event-1", 1)
    context.source_refs.add(event)
    arguments = {
        "path": "/decisions/provider-selection.md",
        "description": "Chosen provider and the constraints that justify the choice",
        "content": "# Decision\nUse the selected provider.",
        "source_refs": ["pupu://context/event/event-1"],
        "rationale": "Needed in later implementation turns",
        "confidence": 0.9,
    }

    first = invoke(toolkit, "memory_propose", **arguments)
    second = invoke(toolkit, "memory_propose", **arguments)

    assert first == second
    assert first["candidate_ref"] == "pupu://memory/candidate/candidate-1@1"
    assert "owner_chat_id" not in first
    assert candidates.calls[0].source_refs == (event,)
    assert all(isinstance(ref, ResourceRef) for ref in candidates.calls[0].source_refs)
    assert candidates.calls[0].operation_id == candidates.calls[1].operation_id

    invoke(toolkit, "memory_propose", **{**arguments, "content": "changed"})
    assert candidates.calls[2].operation_id != candidates.calls[0].operation_id


def test_consolidation_tools_accept_only_frozen_structured_candidate_and_source_refs():
    codec = FakeCodec()
    context = FakeContext()
    consolidation = FakeConsolidation()
    candidate = ResourceRef("memory_candidate", "candidate-1", 1)
    source = ResourceRef("context_event", "event-1", 1)
    fence = curator_fence()
    mutation_guard = FakeMutationGuard(fence)
    chat = FakeChat()
    chat.space_revision = 3
    capabilities = ConsolidationMemoryToolkitCapabilities(
        binding_id="binding-1",
        references=codec,
        context=context,
        chat=chat,
        consolidation=consolidation,
        job_id="job-1",
        candidate_refs=(candidate,),
        lease_fence=fence,
        mutation_guard=mutation_guard,
        source_refs=(source,),
    )
    toolkit = build_memory_toolkit(
        binding(),
        capabilities,
        dialect=HOST_DIALECT,
    )
    chat.space_revision = 99

    candidate_page = invoke(
        toolkit,
        "memory_candidate_read",
        candidate_ref="pupu://memory/candidate/candidate-1@1",
    )
    assert candidate_page["text"] == "candidate content"
    assert consolidation.calls[-1] == ("read_candidate", "job-1", candidate)

    context.source_refs.add(source)
    source_page = invoke(
        toolkit,
        "memory_candidate_source_read",
        source_ref="pupu://context/event/event-1",
    )
    assert source_page["text"] == "hello context"

    applied = invoke(
        toolkit,
        "memory_candidate_apply_new",
        candidate_ref="pupu://memory/candidate/candidate-1@1",
        expected_binding_revision=2,
    )
    assert applied["outcome"] == "applied"
    assert "lease_fence" not in applied
    assert "mutation_guard" not in applied
    assert consolidation.calls[-1][1]["candidate_ref"] == candidate
    assert consolidation.calls[-1][1]["expected_space_revision"] == 3
    assert consolidation.calls[-1][1]["mutation_guard"] is mutation_guard

    with pytest.raises(TypeError, match="expected_space_revision"):
        invoke(
            toolkit,
            "memory_candidate_apply_new",
            candidate_ref="pupu://memory/candidate/candidate-1@1",
            expected_binding_revision=2,
            expected_space_revision=99,
        )

    proposed = invoke(
        toolkit,
        "memory_candidate_propose_review",
        candidate_ref="pupu://memory/candidate/candidate-1@1",
        expected_binding_revision=2,
        target_entry_id="entry-1",
        expected_target_revision=3,
    )
    assert proposed["outcome"] == "awaiting_user"
    assert "lease_fence" not in proposed
    assert "mutation_guard" not in proposed
    assert consolidation.calls[-1][1]["candidate_ref"] == candidate
    assert consolidation.calls[-1][1]["mutation_guard"] is mutation_guard

    with pytest.raises(
        MemoryToolkitError,
        match="^candidate_ref is outside this curator job's frozen candidate set$",
    ):
        invoke(
            toolkit,
            "memory_candidate_read",
            candidate_ref="pupu://memory/candidate/candidate-2@1",
        )
    with pytest.raises(
        MemoryToolkitError,
        match="^source_ref is outside this curator job's frozen provenance set$",
    ):
        invoke(
            toolkit,
            "memory_candidate_source_read",
            source_ref="pupu://context/event/event-2",
        )


def test_binding_mismatch_and_invalid_external_references_fail_closed():
    with pytest.raises(
        MemoryToolkitError,
        match="^context capability belongs to another run binding$",
    ):
        build_memory_toolkit(
            binding(),
            NormalMemoryToolkitCapabilities(
                references=FakeCodec(),
                context=FakeContext("binding-foreign"),
                chat=FakeChat(),
                candidates=FakeCandidates(),
            ),
        )

    toolkit, _, _, _, _ = normal_toolkit()
    with pytest.raises(
        MemoryToolkitError,
        match="^ref must be a revisioned pupu://memory reference$",
    ):
        invoke(toolkit, "memory_read", ref="/host/path/secrets.md")
    with pytest.raises(
        MemoryToolkitError,
        match="^limit must be between 1 and 8192$",
    ):
        invoke(
            toolkit,
            "context_content_read",
            ref="pupu://artifact/artifact-1@1",
            limit=32 * 1024 + 1,
        )


@pytest.mark.parametrize(
    ("field_name", "foreign_capability"),
    (
        ("references", FakeCodec("binding-foreign")),
        ("context", FakeContext("binding-foreign")),
        ("chat", FakeChat("binding-foreign")),
        ("consolidation", FakeConsolidation("binding-foreign")),
    ),
)
def test_consolidation_dependency_binding_mismatch_fails_before_toolkit_exposure(
    field_name,
    foreign_capability,
):
    fence = curator_fence()
    arguments = {
        "binding_id": "binding-1",
        "references": FakeCodec(),
        "context": FakeContext(),
        "chat": FakeChat(),
        "consolidation": FakeConsolidation(),
        "job_id": "job-1",
        "candidate_refs": (ResourceRef("memory_candidate", "candidate-1", 1),),
        "lease_fence": fence,
        "mutation_guard": FakeMutationGuard(fence),
    }
    arguments[field_name] = foreign_capability

    with pytest.raises(
        MemoryToolkitError,
        match=rf"^{field_name} capability belongs to another run binding$",
    ):
        build_memory_toolkit(
            binding(),
            ConsolidationMemoryToolkitCapabilities(**arguments),
        )


def test_consolidation_bundle_binding_mismatch_fails_before_toolkit_exposure():
    fence = curator_fence(binding_id="binding-foreign")
    capabilities = ConsolidationMemoryToolkitCapabilities(
        binding_id="binding-foreign",
        references=FakeCodec(),
        context=FakeContext(),
        chat=FakeChat(),
        consolidation=FakeConsolidation(),
        job_id="job-1",
        candidate_refs=(ResourceRef("memory_candidate", "candidate-1", 1),),
        lease_fence=fence,
        mutation_guard=FakeMutationGuard(fence),
    )

    with pytest.raises(
        MemoryToolkitError,
        match="^consolidation bundle belongs to another run binding$",
    ):
        build_memory_toolkit(binding(), capabilities)


def test_consolidation_capability_with_only_a_binding_fails_before_toolkit_exposure():
    with pytest.raises(
        MemoryToolkitError,
        match="^consolidation capability read_candidate must be callable$",
    ):
        build_memory_toolkit(
            binding(),
            consolidation_bundle(BindingOnlyConsolidation()),
        )


def test_consolidation_capability_with_a_missing_method_fails_before_toolkit_exposure():
    with pytest.raises(
        MemoryToolkitError,
        match="^consolidation capability propose_review must be callable$",
    ):
        build_memory_toolkit(
            binding(),
            consolidation_bundle(MissingReviewConsolidation()),
        )


def test_consolidation_capability_with_a_noncallable_method_fails_before_toolkit_exposure():
    with pytest.raises(
        MemoryToolkitError,
        match="^consolidation capability read_candidate must be callable$",
    ):
        build_memory_toolkit(
            binding(),
            consolidation_bundle(NonCallableReadConsolidation()),
        )


@pytest.mark.parametrize(
    ("method_name", "consolidation"),
    (
        ("apply_new", MissingMutationGuardConsolidation()),
        ("propose_review", MissingReviewMutationGuardConsolidation()),
    ),
)
def test_consolidation_capability_missing_mutation_guard_fails_before_toolkit_exposure(
    method_name,
    consolidation,
):
    with pytest.raises(
        MemoryToolkitError,
        match=(
            rf"^consolidation capability {method_name} has an incompatible signature$"
        ),
    ):
        build_memory_toolkit(
            binding(),
            consolidation_bundle(consolidation),
        )


def test_consolidation_bundle_rejects_mismatched_fence_job_and_guard():
    candidate_refs = (
        ResourceRef("memory_candidate", "candidate-2", 1),
        ResourceRef("memory_candidate", "candidate-1", 1),
    )
    fence = curator_fence()
    common = {
        "binding_id": "binding-1",
        "references": FakeCodec(),
        "context": FakeContext(),
        "chat": FakeChat(),
        "consolidation": FakeConsolidation(),
        "job_id": "job-1",
        "candidate_refs": candidate_refs,
    }

    with pytest.raises(
        MemoryToolkitError,
        match="^consolidation lease fence belongs to another run binding$",
    ):
        ConsolidationMemoryToolkitCapabilities(
            **common,
            lease_fence=curator_fence(binding_id="binding-foreign"),
            mutation_guard=FakeMutationGuard(
                curator_fence(binding_id="binding-foreign")
            ),
        )

    with pytest.raises(
        MemoryToolkitError,
        match="^consolidation lease fence belongs to another job$",
    ):
        ConsolidationMemoryToolkitCapabilities(
            **common,
            lease_fence=curator_fence(job_id="job-foreign"),
            mutation_guard=FakeMutationGuard(curator_fence(job_id="job-foreign")),
        )

    with pytest.raises(
        MemoryToolkitError,
        match="^consolidation mutation guard does not match the lease fence$",
    ):
        ConsolidationMemoryToolkitCapabilities(
            **common,
            lease_fence=fence,
            mutation_guard=FakeMutationGuard(curator_fence(job_revision=3)),
        )

    capabilities = ConsolidationMemoryToolkitCapabilities(
        **common,
        lease_fence=fence,
        mutation_guard=FakeMutationGuard(fence),
    )
    assert isinstance(capabilities, FenceBoundConsolidationToolkit)
    assert capabilities.candidate_refs == candidate_refs


def test_binary_content_is_returned_as_base64_untrusted_data():
    payload = b"\x89PNG\r\n\x1a\n"
    context = FakeContext()
    context.payload = payload
    context.total_bytes = len(payload)
    artifact = ResourceRef("artifact", "image-1", 1)
    context.disclosed.add(artifact)

    def binary_read(*, ref, offset, limit):
        return MemoryToolContentPage(
            ref=ref,
            media_type="image/png",
            data=payload[offset : offset + limit],
            offset=offset,
            total_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    context.read_content = binary_read
    toolkit, _, _, _, _ = normal_toolkit(context=context)

    page = invoke(
        toolkit,
        "context_content_read",
        ref="pupu://artifact/image-1@1",
    )
    assert page["content"] == {
        "encoding": "base64",
        "data_base64": base64.b64encode(payload).decode("ascii"),
    }
    assert page["page_bytes"] == len(payload)


def test_normal_bundle_bounds_recalled_long_term_refs_before_model_exposure():
    allowed = tuple(
        ResourceRef("memory", f"entry-{index}", 1, "space-long") for index in range(6)
    )

    with pytest.raises(
        MemoryToolkitError,
        match="^recalled long-term reference set exceeds the bundle limit$",
    ):
        NormalMemoryToolkitCapabilities(
            references=FakeCodec(),
            context=FakeContext(),
            chat=FakeChat(),
            candidates=FakeCandidates(),
            long_term=FakeChat(space_id="space-long"),
            allowed_long_term_refs=allowed,
        )


def test_consolidation_bundle_rejects_duplicate_frozen_refs():
    candidate = ResourceRef("memory_candidate", "candidate-1", 1)
    source = ResourceRef("context_event", "event-1", 1)
    fence = curator_fence()
    mutation_guard = FakeMutationGuard(fence)

    with pytest.raises(
        MemoryToolkitError,
        match="^candidate_refs must not contain duplicates$",
    ):
        ConsolidationMemoryToolkitCapabilities(
            binding_id="binding-1",
            references=FakeCodec(),
            context=FakeContext(),
            chat=FakeChat(),
            consolidation=FakeConsolidation(),
            job_id="job-1",
            candidate_refs=(candidate, candidate),
            lease_fence=fence,
            mutation_guard=mutation_guard,
            source_refs=(source,),
        )

    with pytest.raises(
        MemoryToolkitError,
        match="^source_refs must not contain duplicates$",
    ):
        ConsolidationMemoryToolkitCapabilities(
            binding_id="binding-1",
            references=FakeCodec(),
            context=FakeContext(),
            chat=FakeChat(),
            consolidation=FakeConsolidation(),
            job_id="job-1",
            candidate_refs=(candidate,),
            lease_fence=fence,
            mutation_guard=mutation_guard,
            source_refs=(source, source),
        )


def test_mutation_identity_is_deterministic_and_bound_to_every_execution_coordinate():
    operation = mutation_id(
        binding(),
        tool_name="memory_propose",
        payload={
            "path": "/facts/release.md",
            "refs": (ResourceRef("context_event", "event-1", 1),),
        },
        qualifier="job-1",
    )
    replay = mutation_id(
        binding(),
        tool_name="memory_propose",
        payload={
            "refs": (ResourceRef("context_event", "event-1", 1),),
            "path": "/facts/release.md",
        },
        qualifier="job-1",
    )

    assert replay == operation
    assert operation.startswith("memory-v2:")
    variants = (
        mutation_id(
            MemoryToolkitRunBinding("binding-2", "session-1", "attempt-1", "run-1"),
            tool_name="memory_propose",
            payload={
                "path": "/facts/release.md",
                "refs": (ResourceRef("context_event", "event-1", 1),),
            },
            qualifier="job-1",
        ),
        mutation_id(
            MemoryToolkitRunBinding("binding-1", "session-2", "attempt-1", "run-1"),
            tool_name="memory_propose",
            payload={
                "path": "/facts/release.md",
                "refs": (ResourceRef("context_event", "event-1", 1),),
            },
            qualifier="job-1",
        ),
        mutation_id(
            binding(),
            tool_name="memory_upsert",
            payload={
                "path": "/facts/release.md",
                "refs": (ResourceRef("context_event", "event-1", 1),),
            },
            qualifier="job-1",
        ),
        mutation_id(
            binding(),
            tool_name="memory_propose",
            payload={
                "path": "/facts/changed.md",
                "refs": (ResourceRef("context_event", "event-1", 1),),
            },
            qualifier="job-1",
        ),
        mutation_id(
            binding(),
            tool_name="memory_propose",
            payload={
                "path": "/facts/release.md",
                "refs": (ResourceRef("context_event", "event-1", 1),),
            },
            qualifier="job-2",
        ),
    )
    assert len({operation, *variants}) == 1 + len(variants)


def test_read_boundaries_reject_capability_pages_larger_than_the_requested_limit():
    class OversizedChat(FakeChat):
        def list_entries(self, **arguments):
            return {
                "entries": [{} for _ in range(arguments["limit"] + 1)],
                "truncated": False,
            }

        def search_entries(self, **arguments):
            return {
                "query": arguments["query"],
                "backend": "hybrid",
                "vector_status": "ready",
                "results": [{} for _ in range(arguments["limit"] + 1)],
            }

    toolkit, _, _, _, _ = normal_toolkit(chat=OversizedChat())
    with pytest.raises(
        MemoryToolkitError,
        match="^memory listing exceeded the requested limit$",
    ):
        invoke(toolkit, "memory_list", limit=1)
    with pytest.raises(
        MemoryToolkitError,
        match="^memory search exceeded the requested limit$",
    ):
        invoke(toolkit, "memory_search", query="release", limit=1)

    context = FakeContext()
    checkpoint = ResourceRef("checkpoint", "checkpoint-1", 1)
    context.disclosed.add(checkpoint)

    def oversized_checkpoint_page(*, ref, after_position, limit):
        del ref, after_position
        return {
            "events": [{} for _ in range(limit + 1)],
            "has_more": False,
        }

    context.read_checkpoint_events = oversized_checkpoint_page
    toolkit, _, _, _, _ = normal_toolkit(context=context)
    with pytest.raises(
        MemoryToolkitError,
        match="^checkpoint event page exceeded the requested limit$",
    ):
        invoke(
            toolkit,
            "context_checkpoint_events_read",
            checkpoint_ref="pupu://context/checkpoint/checkpoint-1",
            limit=1,
        )


@pytest.mark.parametrize(
    "unsafe_url",
    (
        "https://user:password@example.test/docs",
        "https://example.test/docs?token=plain-secret",
        "https://example.test/docs#access_token=plain-secret",
        "https://example.test/token/abcdefghijklmnop",
        "https://example.test/%74%6f%6b%65%6e/abcdefghijklmnop",
        "https://example.test/docs?%EF%BD%94%EF%BD%8F%EF%BD%8B%EF%BD%85%EF%BD%8E=value",
    ),
)
def test_link_candidates_reuse_workspace_credential_url_guard_before_persistence(
    unsafe_url,
):
    candidates = FakeCandidates()
    toolkit, _, _, _, _ = normal_toolkit(candidates=candidates)

    with pytest.raises(
        MemoryToolkitError,
        match="^link URLs cannot contain credentials$",
    ):
        invoke(
            toolkit,
            "memory_propose",
            path="/references/provider-docs.link",
            description="Provider documentation used for the implementation",
            kind="link",
            url=unsafe_url,
        )
    assert candidates.calls == []


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Architecture-owner gate: meaningful_path has CRITICAL GitNexus impact; "
        "replace it with the shared workspace path validator after approval"
    ),
)
@pytest.mark.parametrize(
    "unsafe_path",
    (
        "/notes//secret.md",
        "/%2e%2e/secret.md",
        "/Users/red/secret.md",
        "/C:/Users/red/secret.md",
    ),
)
def test_candidate_paths_share_the_workspace_virtual_path_guard(unsafe_path):
    candidates = FakeCandidates()
    toolkit, _, _, _, _ = normal_toolkit(candidates=candidates)

    with pytest.raises(MemoryToolkitError):
        invoke(
            toolkit,
            "memory_propose",
            path=unsafe_path,
            description="Candidate that must stay inside the virtual workspace",
            content="memory",
        )
    assert candidates.calls == []
