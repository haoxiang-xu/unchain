from __future__ import annotations

import copy
import hashlib
import json
import pickle
from dataclasses import replace
from types import MappingProxyType

import pytest

from unchain.context.tool_catalog import ToolCatalogSnapshot
from unchain.journal.models import (
    ArtifactRef,
    AttemptRef,
    EventCursor,
    GenerationRef,
    JournalAppendRequest,
    JournalAppendResult,
    JournalEvent,
    JournalPage,
    OperationRef,
    ResourceRef,
    ToolExecutionReceiptLookup,
)
from unchain.journal.resource_limits import BoundaryResourceLimitError
from unchain.tools.handler_registry import (
    DurableToolHandlerBinding,
    DurableToolHandlerRegistry,
    ToolHandlerRegistryError,
    tool_config_sha256,
)
from unchain.tools.tool import Tool


ZERO_SHA = "0" * 64
ONE_SHA = "1" * 64
ATTEMPT = AttemptRef(
    GenerationRef("execution-prepared-1", "generation-prepared-1"),
    "attempt-prepared-1",
)


class _FakeModelIO:
    def __init__(
        self,
        *,
        provider: str = "openai",
        model: str = "frontier-model",
    ) -> None:
        self.provider = provider
        self.model = model

    def fetch_turn(self, request):  # pragma: no cover - contract never calls it
        raise AssertionError(request)


def _handler(query: str = "") -> dict[str, str]:
    return {"query": query}


def _registered_resolution(
    *,
    registry: DurableToolHandlerRegistry | None = None,
    name: str = "search",
    provider_native_specs: dict | None = None,
    required_betas: dict | None = None,
    handler=None,
):
    owner = registry or DurableToolHandlerRegistry()
    resolved_handler = handler or _handler
    tool = Tool(
        name=name,
        description=f"Run {name}",
        func=resolved_handler,
        provider_native_specs=provider_native_specs,
        required_betas=required_betas,
    )
    binding = DurableToolHandlerBinding(
        handler_id=f"host.{name}",
        revision=1,
        config_sha256=tool_config_sha256(tool),
        kind="stable",
    )
    return owner, owner.register(
        binding,
        tool=tool,
        handler=resolved_handler,
    )


def _draft(
    *,
    module,
    model_io: object | None = None,
    registry: DurableToolHandlerRegistry | None = None,
    resolutions: list | tuple | None = None,
    supports_tools: bool = True,
    request_payload: dict | None = None,
):
    if registry is None or resolutions is None:
        registry, resolution = _registered_resolution()
        resolutions = [resolution]
    return module._build_provider_turn_draft(
        model_io=model_io or _FakeModelIO(),
        registry=registry,
        resolutions=resolutions,
        attempt=ATTEMPT,
        iteration=7,
        supports_tools=supports_tools,
        request_payload=request_payload
        or {"messages": [{"role": "user", "content": "hi"}]},
        prompt_sha256=ZERO_SHA,
        exposure_plan_sha256=ONE_SHA,
    )


def _snapshot_for_draft(draft) -> ToolCatalogSnapshot:
    content = draft.catalog.canonical_bytes()
    artifact = ArtifactRef(
        ref=ResourceRef("artifact", "prepared-tool-catalog-1", 1),
        media_type="application/json",
        byte_length=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        preview="",
    )
    return ToolCatalogSnapshot(
        envelope=draft.catalog,
        event_cursor=EventCursor(11, "event-prepared-tool-catalog-1"),
        artifact=artifact,
    )


def _prepared(*, module, draft=None, model_io=None):
    resolved_draft = draft or _draft(module=module, model_io=model_io)
    authority = module._issue_persisted_tool_catalog_authority(
        _snapshot_for_draft(resolved_draft)
    )
    return resolved_draft, module._issue_prepared_provider_turn(
        draft=resolved_draft,
        catalog_authority=authority,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _frozen_data_tool(*, module, name: str, schema: dict):
    entry = module.ToolCatalogEntry(
        tool_name=name,
        semantic_schema_sha256=_canonical_sha256(schema),
        tool_descriptor_sha256="d" * 64,
        handler_binding=DurableToolHandlerBinding(
            handler_id=f"host.{name}",
            revision=1,
            config_sha256="c" * 64,
            kind="stable",
        ),
    )
    return module.FrozenProviderTool(
        provider="openai",
        name=name,
        semantic_schema=schema,
        catalog_entry=entry,
        required_betas=(),
    )


def _stored_object_graph(root: object) -> tuple[object, ...]:
    seen: set[int] = set()
    found: list[object] = []
    stack = [root]
    while stack:
        value = stack.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        found.append(value)
        if value is None or type(value) in {bool, int, float, str, bytes}:
            continue
        if isinstance(value, MappingProxyType):
            stack.extend(value.keys())
            stack.extend(value.values())
            continue
        if type(value) is dict:
            stack.extend(value.keys())
            stack.extend(value.values())
            continue
        if type(value) in {list, tuple, set, frozenset}:
            stack.extend(value)
            continue
        try:
            attributes = vars(value)
        except TypeError:
            attributes = {}
        stack.extend(attributes.values())
        for value_type in type(value).__mro__:
            slots = value_type.__dict__.get("__slots__", ())
            if type(slots) is str:
                slots = (slots,)
            for slot in slots:
                if slot in {"__dict__", "__weakref__"}:
                    continue
                attribute_name = slot
                if slot.startswith("__") and not slot.endswith("__"):
                    attribute_name = f"_{value_type.__name__.lstrip('_')}{slot}"
                try:
                    stack.append(object.__getattribute__(value, attribute_name))
                except AttributeError:
                    pass
        if len(seen) > 10_000:
            raise AssertionError("public toolkit object graph exceeded probe bound")
    return tuple(found)


def _recovered_authority_for_draft(draft):
    from unchain.journal import tool_catalog as journal_catalog

    content = draft.catalog.canonical_bytes()
    artifact = ArtifactRef(
        ref=ResourceRef("artifact", "recovered-provider-catalog-1", 1),
        media_type="application/json",
        byte_length=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        preview="",
    )
    event = JournalEvent(
        event_id="event-recovered-provider-catalog-1",
        event_type=journal_catalog.TOOL_CATALOG_SNAPSHOT_EVENT_TYPE,
        attempt=draft.attempt,
        operation=OperationRef("operation-recovered-provider-catalog-1", "a" * 64),
        store_seq=19,
        payload={
            "iteration": draft.iteration,
            "catalog_sha256": draft.catalog.catalog_sha256,
            "catalog_artifact": artifact.to_dict(),
        },
        resource_refs=(artifact.ref,),
    )

    class _CatalogIndex(journal_catalog.BoundToolCatalogIndex):
        def __init__(self) -> None:
            super().__init__(draft.attempt.generation.execution_id)

        def append(self, *, request: JournalAppendRequest) -> JournalAppendResult:
            raise AssertionError(request)

        def read(
            self,
            *,
            after: EventCursor | None = None,
            limit: int = 100,
        ) -> JournalPage:
            raise AssertionError((after, limit))

        def capture_snapshot(self, *, max_events=10_000, max_bytes=32 * 1024 * 1024):
            raise AssertionError((max_events, max_bytes))

        def lookup_tool_execution_receipts(
            self,
            *,
            attempt: AttemptRef,
            call_id: str,
        ) -> ToolExecutionReceiptLookup:
            raise AssertionError((attempt, call_id))

        def lookup_tool_catalog_receipts(self, *, attempt, iteration):
            return journal_catalog.ToolCatalogReceiptLookup(
                attempt=attempt,
                iteration=iteration,
                events=(event,),
            )

    return journal_catalog.recover_tool_catalog_authority(
        _CatalogIndex(),
        attempt=draft.attempt,
        iteration=draft.iteration,
        expected_catalog_sha256=draft.catalog.catalog_sha256,
        expected_catalog_artifact=artifact,
    )


def test_frozen_provider_toolkit_is_built_from_verified_resolutions() -> None:
    from unchain.providers import prepared_turn as module

    registry, resolution = _registered_resolution(
        required_betas={"anthropic": ["computer-use-2025-11-24"]}
    )
    draft = _draft(
        module=module,
        model_io=_FakeModelIO(provider="anthropic"),
        registry=registry,
        resolutions=[resolution],
    )
    toolkit = draft.toolkit

    assert type(toolkit) is module.FrozenProviderToolkit
    assert toolkit.provider == "anthropic"
    assert toolkit.supports_tools is True
    assert tuple(tool.name for tool in toolkit.tools) == ("search",)
    assert toolkit.required_betas("anthropic") == ["computer-use-2025-11-24"]
    assert toolkit.required_betas("openai") == []
    schemas = toolkit.to_provider_json("anthropic")
    assert schemas[0]["name"] == "search"
    assert toolkit.tool_schema_manifest == {"search": _canonical_sha256(schemas[0])}
    assert toolkit.tool_schema_sha256 == _canonical_sha256(schemas)
    assert toolkit.required_betas_sha256 == _canonical_sha256(
        ["computer-use-2025-11-24"]
    )
    assert len(toolkit.toolkit_sha256) == 64
    assert draft.catalog.catalog_sha256
    assert draft.catalog.required_betas_sha256 == toolkit.required_betas_sha256
    assert draft.catalog.tool_schema_sha256 == toolkit.tool_schema_sha256

    schemas[0]["description"] = "caller mutation"
    assert toolkit.to_provider_json("anthropic")[0]["description"] == "Run search"
    with pytest.raises(TypeError):
        toolkit.tools[0].semantic_schema["name"] = "mutation"


def test_public_frozen_toolkit_cannot_reach_or_invoke_live_tool_authority() -> None:
    from unchain.providers import prepared_turn as module

    calls: list[str] = []

    def handler(query: str = "") -> dict[str, str]:
        calls.append(query)
        return {"query": query}

    registry, resolution = _registered_resolution(handler=handler)
    draft = _draft(
        module=module,
        registry=registry,
        resolutions=[resolution],
    )
    _draft_value, prepared = _prepared(module=module, draft=draft)

    toolkit = prepared.toolkit
    reachable = _stored_object_graph((prepared, draft, toolkit))
    leaked_resolutions = [
        value
        for value in reachable
        if type(value).__name__ == "DurableToolHandlerResolution"
    ]
    for leaked in leaked_resolutions:
        leaked.handler("public-probe")

    assert any(
        type(value).__name__ == "_PreparedProviderTurnRecord" for value in reachable
    )
    assert not hasattr(toolkit, "_registry")
    assert not hasattr(toolkit, "_resolutions")
    assert all(type(value) is not DurableToolHandlerRegistry for value in reachable)
    assert all(type(value) is not Tool for value in reachable)
    assert all(
        type(value).__name__ != "DurableToolHandlerResolution" for value in reachable
    )
    assert registry not in reachable
    assert resolution not in reachable
    assert resolution.tool not in reachable
    assert resolution.handler not in reachable
    assert leaked_resolutions == []
    assert calls == []


def test_private_draft_authority_seals_exact_registry_and_resolution_ids() -> None:
    from unchain.providers import prepared_turn as module

    registry, resolution = _registered_resolution()
    draft = _draft(
        module=module,
        registry=registry,
        resolutions=[resolution],
    )
    seal = module._PROVIDER_TURN_DRAFT_ISSUER._records[id(draft)]

    assert seal.registry is registry
    assert seal.registry_id == id(registry)
    assert seal.resolutions == (resolution,)
    assert seal.resolution_ids == (id(resolution),)

    _foreign_registry, foreign_resolution = _registered_resolution()
    object.__setattr__(seal, "resolutions", (foreign_resolution,))
    authority = module._issue_persisted_tool_catalog_authority(
        _snapshot_for_draft(draft)
    )
    with pytest.raises((TypeError, ValueError), match="draft.*authority|changed"):
        module._issue_prepared_provider_turn(
            draft=draft,
            catalog_authority=authority,
        )


def test_draft_authority_rejects_coherent_foreign_handler_substitution() -> None:
    from unchain.providers import prepared_turn as module

    registry, resolution = _registered_resolution()
    draft = _draft(
        module=module,
        registry=registry,
        resolutions=[resolution],
    )
    evil_calls: list[str] = []

    def evil_handler(query: str = "") -> dict[str, str]:
        evil_calls.append(query)
        return {"query": query}

    foreign_registry, foreign_resolution = _registered_resolution(
        handler=evil_handler,
    )
    assert foreign_resolution.binding.to_dict() == resolution.binding.to_dict()
    assert (
        foreign_resolution.tool_descriptor_sha256 == resolution.tool_descriptor_sha256
    )
    assert foreign_registry.verify_resolution(foreign_resolution) is foreign_resolution

    seal = module._PROVIDER_TURN_DRAFT_ISSUER._records[id(draft)]
    object.__setattr__(seal, "registry", foreign_registry)
    object.__setattr__(seal, "resolutions", (foreign_resolution,))
    object.__setattr__(seal, "registry_id", id(foreign_registry))
    object.__setattr__(seal, "resolution_ids", (id(foreign_resolution),))
    object.__setattr__(seal, "tool_ids", (id(foreign_resolution.tool),))
    object.__setattr__(seal, "handler_ids", (id(foreign_resolution.handler),))

    with pytest.raises(
        module.PreparedProviderTurnError,
        match="authority anchor|issuer authority",
    ):
        module._verify_draft(draft)
    assert evil_calls == []


def test_draft_authority_exposes_no_oracle_for_resigning_a_tampered_seal() -> None:
    from unchain.providers import prepared_turn as module

    registry, resolution = _registered_resolution()
    draft = _draft(
        module=module,
        registry=registry,
        resolutions=[resolution],
    )
    issuer = module._PROVIDER_TURN_DRAFT_ISSUER
    seal = issuer._records[id(draft)]
    foreign_registry, foreign_resolution = _registered_resolution()
    object.__setattr__(seal, "registry", foreign_registry)
    object.__setattr__(seal, "resolutions", (foreign_resolution,))
    object.__setattr__(seal, "registry_id", id(foreign_registry))
    object.__setattr__(seal, "resolution_ids", (id(foreign_resolution),))
    object.__setattr__(seal, "tool_ids", (id(foreign_resolution.tool),))
    object.__setattr__(seal, "handler_ids", (id(foreign_resolution.handler),))

    assert not hasattr(issuer, "_authority_mac")
    assert "authority_mac" not in module._ProviderTurnDraftSeal.__slots__
    assert all(
        not (type(value) is bytes and len(value) == 32)
        for value in vars(issuer).values()
    )
    assert not hasattr(
        issuer,
        "_ProviderTurnDraftIssuer__register_authority_anchor",
    )
    with pytest.raises(module.PreparedProviderTurnError, match="authority anchor"):
        module._verify_draft(draft)


def test_draft_authority_rejects_map_seal_identity_replacement() -> None:
    from unchain.providers import prepared_turn as module

    draft = _draft(module=module)
    issuer = module._PROVIDER_TURN_DRAFT_ISSUER
    seal = issuer._records[id(draft)]
    replacement = replace(seal)
    assert replacement is not seal
    issuer._records[id(draft)] = replacement
    assert not hasattr(
        issuer,
        "_ProviderTurnDraftIssuer__register_authority_anchor",
    )

    with pytest.raises(
        module.PreparedProviderTurnError,
        match="authority anchor|issuer authority",
    ):
        module._verify_draft(draft)


def test_draft_authority_cannot_be_reanchored_by_another_issuer() -> None:
    from unchain.providers import prepared_turn as module

    draft = _draft(module=module)
    owner = module._PROVIDER_TURN_DRAFT_ISSUER
    seal = owner._records[id(draft)]
    foreign = module._ProviderTurnDraftIssuer(max_records=8)

    assert not hasattr(
        foreign,
        "_ProviderTurnDraftIssuer__register_authority_anchor",
    )
    foreign._records[id(draft)] = seal
    with pytest.raises(
        module.PreparedProviderTurnError,
        match="authority anchor|issuer authority",
    ):
        foreign.verify(draft)


@pytest.mark.parametrize(
    "provider",
    [" OpenAI", "openai ", "OPENAI", "unsupported-provider", ""],
)
def test_provider_name_uses_an_exact_allowlist(provider: str) -> None:
    from unchain.providers import prepared_turn as module

    with pytest.raises((TypeError, ValueError), match="provider"):
        _draft(module=module, model_io=_FakeModelIO(provider=provider))


def test_supports_tools_false_freezes_an_empty_provider_catalog() -> None:
    from unchain.providers import prepared_turn as module

    draft = _draft(module=module, supports_tools=False)

    assert draft.toolkit.supports_tools is False
    assert draft.toolkit.tools == ()
    assert draft.toolkit.to_provider_json("openai") == []
    assert draft.toolkit.required_betas("openai") == []
    assert draft.catalog.semantic_schemas == ()
    assert draft.catalog.entries == ()


def test_frozen_toolkit_rejects_foreign_registry_resolution() -> None:
    from unchain.providers import prepared_turn as module

    owner, resolution = _registered_resolution()
    foreign = DurableToolHandlerRegistry()

    with pytest.raises(ToolHandlerRegistryError, match="authority"):
        _draft(module=module, registry=foreign, resolutions=[resolution])
    assert owner.verify_resolution(resolution) is resolution


def test_tool_mutation_before_or_after_freeze_fails_closed() -> None:
    from unchain.providers import prepared_turn as module

    registry, resolution = _registered_resolution()
    resolution.tool.description = "mutated before freeze"
    with pytest.raises(ToolHandlerRegistryError, match="changed"):
        _draft(module=module, registry=registry, resolutions=[resolution])

    registry, resolution = _registered_resolution()
    draft = _draft(module=module, registry=registry, resolutions=[resolution])
    resolution.tool.description = "mutated after freeze"
    authority = module._issue_persisted_tool_catalog_authority(
        _snapshot_for_draft(draft)
    )
    with pytest.raises(ToolHandlerRegistryError, match="changed"):
        module._issue_prepared_provider_turn(
            draft=draft,
            catalog_authority=authority,
        )


def test_frozen_tool_data_rejects_cycles_before_canonical_hashing() -> None:
    from unchain.providers import prepared_turn as module

    circular: dict = {"name": "search"}
    circular["parameters"] = circular

    with pytest.raises((TypeError, ValueError), match="circular"):
        module.FrozenProviderTool(
            provider="openai",
            name="search",
            semantic_schema=circular,
            catalog_entry=None,
            required_betas=(),
        )


@pytest.mark.parametrize(
    ("schema", "dimension"),
    [
        ({"name": "search", "description": "x" * (16 * 1024 + 1)}, "string_bytes"),
        ({"name": "search", "payload": [None] * 1025}, "container_items"),
    ],
)
def test_frozen_tool_data_enforces_schema_shape_limits(
    schema: dict,
    dimension: str,
) -> None:
    from unchain.providers import prepared_turn as module

    with pytest.raises(BoundaryResourceLimitError) as caught:
        module.FrozenProviderTool(
            provider="openai",
            name="search",
            semantic_schema=schema,
            catalog_entry=None,
            required_betas=(),
        )
    assert caught.value.dimension == dimension


def test_frozen_tool_data_enforces_depth_and_per_schema_bytes() -> None:
    from unchain.providers import prepared_turn as module

    nested: dict = {"type": "string"}
    for _index in range(40):
        nested = {"items": nested}
    with pytest.raises(BoundaryResourceLimitError) as depth_error:
        module.FrozenProviderTool(
            provider="openai",
            name="search",
            semantic_schema={"name": "search", "parameters": nested},
            catalog_entry=None,
            required_betas=(),
        )
    assert depth_error.value.dimension == "depth"

    oversized = {
        "name": "search",
        "metadata": ["x" * 15_000 for _index in range(5)],
    }
    with pytest.raises(BoundaryResourceLimitError) as bytes_error:
        module.FrozenProviderTool(
            provider="openai",
            name="search",
            semantic_schema=oversized,
            catalog_entry=None,
            required_betas=(),
        )
    assert bytes_error.value.dimension == "bytes"


def test_frozen_provider_tool_never_accepts_a_missing_catalog_entry() -> None:
    from unchain.providers import prepared_turn as module

    with pytest.raises(TypeError, match="catalog_entry"):
        module.FrozenProviderTool(
            provider="openai",
            name="search",
            semantic_schema={"name": "search", "parameters": {}},
            catalog_entry=None,
            required_betas=(),
        )


def test_frozen_toolkit_enforces_tool_count_before_duplicate_names() -> None:
    from unchain.providers import prepared_turn as module

    registry, resolution = _registered_resolution()
    draft = _draft(module=module, registry=registry, resolutions=[resolution])
    tool = draft.toolkit.tools[0]

    with pytest.raises(BoundaryResourceLimitError) as caught:
        module.FrozenProviderToolkit(
            provider="openai",
            supports_tools=True,
            tools=(tool,) * 257,
        )
    assert caught.value.dimension == "items"


def test_frozen_toolkit_rejects_duplicate_stable_names() -> None:
    from unchain.providers import prepared_turn as module

    tool = _frozen_data_tool(
        module=module,
        name="search",
        schema={"name": "search", "parameters": {}},
    )
    with pytest.raises(ValueError, match="duplicate stable names"):
        module.FrozenProviderToolkit(
            provider="openai",
            supports_tools=True,
            tools=(tool, tool),
        )


def test_frozen_toolkit_enforces_aggregate_catalog_bytes() -> None:
    from unchain.providers import prepared_turn as module

    tools = tuple(
        _frozen_data_tool(
            module=module,
            name=f"tool_{index}",
            schema={
                "name": f"tool_{index}",
                "metadata": ["x" * 14_000 for _item in range(4)],
            },
        )
        for index in range(19)
    )
    with pytest.raises(BoundaryResourceLimitError) as caught:
        module.FrozenProviderToolkit(
            provider="openai",
            supports_tools=True,
            tools=tools,
        )
    assert caught.value.dimension == "bytes"


def test_frozen_toolkit_enforces_aggregate_node_limit() -> None:
    from unchain.providers import prepared_turn as module

    tools = tuple(
        _frozen_data_tool(
            module=module,
            name=f"node_tool_{index}",
            schema={
                "name": f"node_tool_{index}",
                "metadata": {
                    f"field_{field_index}": False for field_index in range(205)
                },
            },
        )
        for index in range(245)
    )
    with pytest.raises(BoundaryResourceLimitError) as caught:
        module.FrozenProviderToolkit(
            provider="openai",
            supports_tools=True,
            tools=tools,
        )
    assert caught.value.dimension == "nodes"


@pytest.mark.parametrize(
    ("betas", "dimension"),
    [
        (tuple(f"beta-{index}" for index in range(65)), "items"),
        (("x" * 257,), "string_bytes"),
        (tuple(("x" * 252) + f"{index:04d}" for index in range(64)), "bytes"),
    ],
)
def test_required_betas_have_independent_resource_limits(
    betas: tuple[str, ...],
    dimension: str,
) -> None:
    from unchain.providers import prepared_turn as module

    registry, resolution = _registered_resolution()
    draft = _draft(module=module, registry=registry, resolutions=[resolution])
    base = draft.toolkit.tools[0]

    with pytest.raises(BoundaryResourceLimitError) as caught:
        module.FrozenProviderTool(
            provider="openai",
            name=base.name,
            semantic_schema=base.to_provider_json(),
            catalog_entry=base.catalog_entry,
            required_betas=betas,
        )
    assert caught.value.dimension == dimension


def test_schema_and_beta_text_must_already_be_nfc() -> None:
    from unchain.providers import prepared_turn as module

    registry, resolution = _registered_resolution()
    draft = _draft(module=module, registry=registry, resolutions=[resolution])
    base = draft.toolkit.tools[0]

    bad_schema = base.to_provider_json()
    bad_schema["description"] = "Cafe\u0301"
    with pytest.raises(ValueError, match="canonical JSON"):
        module.FrozenProviderTool(
            provider="openai",
            name=base.name,
            semantic_schema=bad_schema,
            catalog_entry=base.catalog_entry,
            required_betas=(),
        )
    with pytest.raises(ValueError, match="NFC"):
        module.FrozenProviderTool(
            provider="openai",
            name=base.name,
            semantic_schema=base.to_provider_json(),
            catalog_entry=base.catalog_entry,
            required_betas=("Cafe\u0301",),
        )


def test_prepared_turn_is_an_identity_bound_nonserializable_capability() -> None:
    from unchain.providers import prepared_turn as module

    model_io = _FakeModelIO()
    draft, prepared = _prepared(module=module, model_io=model_io)

    assert type(prepared) is module.PreparedProviderTurn
    assert prepared.provider == "openai"
    assert prepared.model == "frontier-model"
    assert prepared.attempt == ATTEMPT
    assert prepared.iteration == 7
    assert prepared.toolkit is draft.toolkit
    assert prepared.catalog_sha256 == draft.catalog.catalog_sha256
    assert (
        module.verify_prepared_provider_turn(
            prepared,
            model_io=model_io,
            attempt=ATTEMPT,
            iteration=7,
        )
        is prepared
    )
    assert (
        module._consume_prepared_provider_turn(
            prepared,
            model_io=model_io,
            attempt=ATTEMPT,
            iteration=7,
        )
        is draft
    )
    assert not hasattr(prepared, "request_payload")
    assert not hasattr(prepared, "draft")
    assert not hasattr(prepared, "catalog_authority")
    assert not hasattr(prepared, "__dict__")

    with pytest.raises(TypeError, match="issuer"):
        module.PreparedProviderTurn()
    with pytest.raises(TypeError, match="subclass"):
        type("PreparedSubclass", (module.PreparedProviderTurn,), {})
    for copier in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="copied|serialized"):
            copier(prepared)


def test_private_draft_requires_exact_issuer_identity() -> None:
    from unchain.providers import prepared_turn as module

    draft = _draft(module=module)
    authority = module._issue_persisted_tool_catalog_authority(
        _snapshot_for_draft(draft)
    )
    manual = module._ProviderTurnDraft(
        model_io=draft.model_io,
        provider=draft.provider,
        model=draft.model,
        attempt=draft.attempt,
        iteration=draft.iteration,
        toolkit=draft.toolkit,
        catalog=draft.catalog,
        _request_payload=draft._request_payload,
        _request_payload_sha256=draft._request_payload_sha256,
    )
    copied = copy.copy(draft)

    for forged in (manual, copied):
        with pytest.raises((TypeError, ValueError), match="draft.*issuer|authority"):
            module._issue_prepared_provider_turn(
                draft=forged,
                catalog_authority=authority,
            )


def test_model_io_provider_and_model_cannot_drift_after_draft_freeze() -> None:
    from unchain.providers import prepared_turn as module

    model_io = _FakeModelIO()
    draft = _draft(module=module, model_io=model_io)
    authority = module._issue_persisted_tool_catalog_authority(
        _snapshot_for_draft(draft)
    )

    model_io.provider = "anthropic"
    with pytest.raises((TypeError, ValueError), match="draft.*changed"):
        module._issue_prepared_provider_turn(
            draft=draft,
            catalog_authority=authority,
        )
    model_io.provider = "openai"
    model_io.model = "different-model"
    with pytest.raises((TypeError, ValueError), match="draft.*changed"):
        module._issue_prepared_provider_turn(
            draft=draft,
            catalog_authority=authority,
        )


def test_model_name_must_be_exact_canonical_text() -> None:
    from unchain.providers import prepared_turn as module

    for model in (" frontier-model", "frontier-model ", "Cafe\u0301"):
        with pytest.raises((TypeError, ValueError), match="model"):
            _draft(module=module, model_io=_FakeModelIO(model=model))


def test_prepared_turn_rejects_manual_duck_foreign_consumer_and_subjects() -> None:
    from unchain.providers import prepared_turn as module

    model_io = _FakeModelIO()
    _draft_value, prepared = _prepared(module=module, model_io=model_io)
    manual = object.__new__(module.PreparedProviderTurn)

    for forged in (
        manual,
        {
            "provider": prepared.provider,
            "attempt": prepared.attempt,
            "iteration": prepared.iteration,
        },
    ):
        with pytest.raises((TypeError, ValueError), match="authority|issuer|exact"):
            module.verify_prepared_provider_turn(
                forged,
                model_io=model_io,
                attempt=ATTEMPT,
                iteration=7,
            )

    with pytest.raises((TypeError, ValueError), match="model_io|consumer"):
        module.verify_prepared_provider_turn(
            prepared,
            model_io=_FakeModelIO(),
            attempt=ATTEMPT,
            iteration=7,
        )
    with pytest.raises((TypeError, ValueError), match="attempt"):
        module.verify_prepared_provider_turn(
            prepared,
            model_io=model_io,
            attempt=replace(ATTEMPT, attempt_id="attempt-foreign"),
            iteration=7,
        )
    with pytest.raises((TypeError, ValueError), match="iteration"):
        module.verify_prepared_provider_turn(
            prepared,
            model_io=model_io,
            attempt=ATTEMPT,
            iteration=8,
        )


def test_prepared_turn_detects_record_and_draft_replacement() -> None:
    from unchain.providers import prepared_turn as module

    model_io = _FakeModelIO()
    draft, prepared = _prepared(module=module, model_io=model_io)
    other_draft = _draft(module=module, model_io=model_io)

    object.__setattr__(
        prepared,
        "_PreparedProviderTurn__issued_record",
        object(),
    )
    with pytest.raises((TypeError, ValueError), match="authority|record"):
        module.verify_prepared_provider_turn(
            prepared,
            model_io=model_io,
            attempt=ATTEMPT,
            iteration=7,
        )

    _draft_value, prepared = _prepared(module=module, draft=draft)
    object.__setattr__(draft, "toolkit", other_draft.toolkit)
    with pytest.raises((TypeError, ValueError), match="draft|changed"):
        module.verify_prepared_provider_turn(
            prepared,
            model_io=model_io,
            attempt=ATTEMPT,
            iteration=7,
        )


def test_prepared_turn_issuer_has_a_bounded_weak_registry() -> None:
    from unchain.providers import prepared_turn as module

    draft = _draft(module=module)
    authority = module._issue_persisted_tool_catalog_authority(
        _snapshot_for_draft(draft)
    )
    issuer = module._PreparedProviderTurnIssuer(max_records=1)
    first = issuer.issue(draft=draft, catalog_authority=authority)

    with pytest.raises(RuntimeError, match="capacity"):
        issuer.issue(draft=draft, catalog_authority=authority)
    del first
    second = issuer.issue(draft=draft, catalog_authority=authority)
    assert type(second) is module.PreparedProviderTurn


def test_fresh_catalog_authority_is_bound_to_exact_snapshot_and_digest() -> None:
    from unchain.providers import prepared_turn as module

    draft = _draft(module=module)
    snapshot = _snapshot_for_draft(draft)
    authority = module._issue_persisted_tool_catalog_authority(snapshot)

    assert type(authority) is module.PersistedToolCatalogAuthority
    assert authority.attempt == ATTEMPT
    assert authority.iteration == 7
    assert authority.catalog_sha256 == draft.catalog.catalog_sha256
    assert authority.catalog_artifact == snapshot.artifact
    assert module.verify_persisted_tool_catalog_authority(authority) is authority
    with pytest.raises(TypeError, match="persisted|issuer"):
        module.PersistedToolCatalogAuthority()
    for copier in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="copied|serialized"):
            copier(authority)


def test_prepared_turn_accepts_an_authentic_recovered_catalog_authority() -> None:
    from unchain.providers import prepared_turn as module

    model_io = _FakeModelIO()
    draft = _draft(module=module, model_io=model_io)
    recovered = _recovered_authority_for_draft(draft)

    prepared = module._issue_prepared_provider_turn(
        draft=draft,
        catalog_authority=recovered,
    )

    assert (
        module.verify_prepared_provider_turn(
            prepared,
            model_io=model_io,
            attempt=draft.attempt,
            iteration=draft.iteration,
        )
        is prepared
    )


def test_recovered_receipt_cannot_authorize_a_noncanonical_catalog_body() -> None:
    from unchain.providers import prepared_turn as module

    draft = _draft(module=module)
    object.__setattr__(draft.catalog.entries[0], "tool_name", "forged_name")
    recovered = _recovered_authority_for_draft(draft)

    with pytest.raises((TypeError, ValueError), match="catalog|draft"):
        module._issue_prepared_provider_turn(
            draft=draft,
            catalog_authority=recovered,
        )


def test_prepared_turn_rejects_catalog_authority_for_another_draft() -> None:
    from unchain.providers import prepared_turn as module

    first = _draft(module=module)
    second = _draft(
        module=module,
        request_payload={"messages": [{"role": "user", "content": "different"}]},
    )
    object.__setattr__(second.catalog, "catalog_sha256", "f" * 64)
    authority = module._issue_persisted_tool_catalog_authority(
        _snapshot_for_draft(first)
    )

    with pytest.raises((TypeError, ValueError), match="catalog|draft"):
        module._issue_prepared_provider_turn(
            draft=second,
            catalog_authority=authority,
        )


def test_request_payload_is_detached_and_only_available_to_private_consumer() -> None:
    from unchain.providers import prepared_turn as module

    payload = {"messages": [{"role": "user", "content": "original"}]}
    model_io = _FakeModelIO()
    draft = _draft(module=module, model_io=model_io, request_payload=payload)
    payload["messages"][0]["content"] = "mutated"
    _draft_value, prepared = _prepared(module=module, draft=draft)

    opened = module._consume_prepared_provider_turn(
        prepared,
        model_io=model_io,
        attempt=ATTEMPT,
        iteration=7,
    )
    assert opened._request_payload_copy()["messages"][0]["content"] == "original"
    first = opened._request_payload_copy()
    first["messages"][0]["content"] = "caller mutation"
    assert opened._request_payload_copy()["messages"][0]["content"] == "original"
    assert isinstance(opened._request_payload, MappingProxyType)


def test_request_payload_uses_transport_safety_not_context_budget_limits() -> None:
    from unchain.providers import prepared_turn as module

    assert module.MAX_PROVIDER_REQUEST_PAYLOAD_BYTES == 64 * 1024 * 1024
    assert module.MAX_PROVIDER_REQUEST_PAYLOAD_DEPTH == 64
    assert module.MAX_PROVIDER_REQUEST_PAYLOAD_NODES == 1_000_000
    assert module.MAX_PROVIDER_REQUEST_PAYLOAD_CONTAINER_ITEMS == 250_000
    assert module.MAX_PROVIDER_REQUEST_PAYLOAD_STRING_BYTES == 32 * 1024 * 1024


@pytest.mark.parametrize(
    ("constant", "limit", "payload", "dimension"),
    [
        (
            "MAX_PROVIDER_REQUEST_PAYLOAD_DEPTH",
            3,
            {"value": [[[[None]]]]},
            "depth",
        ),
        (
            "MAX_PROVIDER_REQUEST_PAYLOAD_NODES",
            5,
            {"values": [0, 1, 2, 3]},
            "nodes",
        ),
        (
            "MAX_PROVIDER_REQUEST_PAYLOAD_BYTES",
            24,
            {"value": "01234567890123456789"},
            "bytes",
        ),
        (
            "MAX_PROVIDER_REQUEST_PAYLOAD_CONTAINER_ITEMS",
            2,
            {"values": [0, 1, 2]},
            "container_items",
        ),
        (
            "MAX_PROVIDER_REQUEST_PAYLOAD_STRING_BYTES",
            4,
            {"value": "12345"},
            "string_bytes",
        ),
    ],
)
def test_request_payload_iterative_preflight_rejects_each_resource_dimension(
    monkeypatch,
    constant: str,
    limit: int,
    payload: dict,
    dimension: str,
) -> None:
    from unchain.providers import prepared_turn as module

    monkeypatch.setattr(module, constant, limit, raising=False)
    issuer = module._ProviderTurnDraftIssuer(max_records=8)
    monkeypatch.setattr(module, "_PROVIDER_TURN_DRAFT_ISSUER", issuer)

    with pytest.raises(BoundaryResourceLimitError) as caught:
        _draft(module=module, request_payload=payload)

    assert caught.value.boundary == "provider request payload"
    assert caught.value.dimension == dimension
    assert issuer._records == {}


def test_request_payload_preflight_runs_before_json_serialization(monkeypatch) -> None:
    from unchain.providers import prepared_turn as module

    monkeypatch.setattr(
        module,
        "MAX_PROVIDER_REQUEST_PAYLOAD_CONTAINER_ITEMS",
        1,
        raising=False,
    )

    def forbidden_json_dump(*args, **kwargs):
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(module.json, "dumps", forbidden_json_dump)
    with pytest.raises(BoundaryResourceLimitError) as caught:
        module._strict_private_payload({"values": [0, 1]})
    assert caught.value.dimension == "container_items"


def test_request_payload_preflight_rejects_cycles_before_issuer() -> None:
    from unchain.providers import prepared_turn as module

    payload: dict = {"messages": []}
    payload["cycle"] = payload
    issuer = module._ProviderTurnDraftIssuer(max_records=8)
    original_issuer = module._PROVIDER_TURN_DRAFT_ISSUER
    module._PROVIDER_TURN_DRAFT_ISSUER = issuer
    try:
        with pytest.raises(
            module.PreparedProviderTurnError,
            match="circular request payload",
        ):
            _draft(module=module, request_payload=payload)
    finally:
        module._PROVIDER_TURN_DRAFT_ISSUER = original_issuer
    assert issuer._records == {}


@pytest.mark.parametrize(
    "non_json",
    [
        {"value": (1, 2)},
        {"value": type("ListSubclass", (list,), {})([1, 2])},
        type("DictSubclass", (dict,), {})({"value": 1}),
        {"value": type("IntSubclass", (int,), {})(1)},
    ],
)
def test_request_payload_requires_exact_json_container_and_scalar_types(
    non_json: object,
) -> None:
    from unchain.providers import prepared_turn as module

    with pytest.raises(TypeError, match="exact JSON"):
        module._strict_private_payload(non_json)
