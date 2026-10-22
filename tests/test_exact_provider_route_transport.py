from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from unchain.context.tool_catalog import ToolCatalogEnvelope
from unchain.journal.models import AttemptRef, GenerationRef
from unchain.providers.exact_route_transport import OpenAIExactRouteTransport
from unchain.providers.anthropic import AnthropicModelIO
from unchain.providers.hyperspace import HyperspaceModelIO
from unchain.providers.ollama import OllamaModelIO
from unchain.providers.openai import OpenAIModelIO
from unchain.providers.wire_envelope import ProviderWireEnvelope, ProviderWireRoute


ATTEMPT = AttemptRef(
    GenerationRef("execution-exact-provider", "generation-exact-provider"),
    "attempt-exact-provider",
)


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _catalog(provider: str, model: str) -> ToolCatalogEnvelope:
    return ToolCatalogEnvelope(
        attempt=ATTEMPT,
        iteration=2,
        provider=provider,
        model=model,
        semantic_schemas=[],
        entries=[],
        required_betas_sha256=_json_sha256([]),
        prompt_sha256="0" * 64,
        exposure_plan_sha256="1" * 64,
    )


def _envelope(
    *,
    provider: str,
    model: str,
    request: dict,
) -> tuple[ToolCatalogEnvelope, ProviderWireEnvelope]:
    profiles = {
        "openai": (
            "unchain.openai.responses.request.v1",
            "openai.responses.create",
        ),
        "anthropic": (
            "unchain.anthropic.messages.request.v1",
            "anthropic.messages.stream",
        ),
        "hyperspace": (
            "unchain.hyperspace.anthropic-messages.request.v1",
            "hyperspace.anthropic.messages.stream",
        ),
        "ollama": (
            "unchain.ollama.chat.request.v1",
            "ollama.api.chat.post",
        ),
        "gemini": (
            "unchain.gemini.contents.request.v1",
            "gemini.models.generate_content_stream",
        ),
    }
    catalog = _catalog(provider, model)
    adapter_revision, transport_kind = profiles[provider]
    envelope = ProviderWireEnvelope(
        attempt=ATTEMPT,
        iteration=2,
        provider=provider,
        configured_model=model,
        request_model=model,
        adapter_revision=adapter_revision,
        transport_kind=transport_kind,
        transport_target_sha256="2" * 64,
        source_request_sha256="3" * 64,
        source_payload_sha256="4" * 64,
        catalog_sha256=catalog.catalog_sha256,
        prompt_sha256=catalog.prompt_sha256,
        tool_schema_sha256=catalog.tool_schema_sha256,
        required_betas=(),
        base_anthropic_betas=(),
        routes=(ProviderWireRoute(name="primary", request=request),),
    )
    return catalog, envelope


class _OpenAIStream:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        yield SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                id="response-exact-openai",
                output=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "exact openai"}],
                    }
                ],
                usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            ),
        )


def test_openai_exact_transport_sends_the_persisted_route_once_with_sdk_retry_off():
    request = {
        "model": "gpt-4.1",
        "input": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
        "stream": True,
    }
    catalog, envelope = _envelope(
        provider="openai",
        model="gpt-4.1",
        request=request,
    )
    init_calls: list[dict] = []
    send_calls: list[dict] = []

    class _Responses:
        def create(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            return _OpenAIStream()

    class _Client:
        responses = _Responses()

    def client_factory(**kwargs):
        init_calls.append(copy.deepcopy(kwargs))
        return _Client()

    model_io = OpenAIModelIO(
        model="gpt-4.1",
        api_key="secret",
        client_factory=client_factory,
        default_payloads={},
        model_capabilities={},
    )
    transport = OpenAIExactRouteTransport(model_io=model_io, catalog=catalog)

    result = transport.send(
        envelope=envelope,
        route=envelope.routes[0],
        retry_ordinal=0,
    )

    assert init_calls == [{"api_key": "secret", "max_retries": 0}]
    assert send_calls == [envelope.routes[0].request_copy()]
    assert result.final_text == "exact openai"
    assert result.provider_replay_frame["items"][:1] == request["input"]


def test_exact_transport_buffers_provider_callbacks_until_explicit_release():
    request = {
        "model": "gpt-4.1",
        "input": [{"role": "user", "content": "hello"}],
        "stream": True,
    }
    catalog, envelope = _envelope(
        provider="openai",
        model="gpt-4.1",
        request=request,
    )
    external_events: list[dict] = []

    class _StreamingResponse(_OpenAIStream):
        def __iter__(self):
            yield SimpleNamespace(type="response.output_text.delta", delta="exact ")
            yield from super().__iter__()

    class _Responses:
        def create(self, **kwargs):
            assert kwargs == envelope.routes[0].request_copy()
            return _StreamingResponse()

    class _Client:
        responses = _Responses()

    transport = OpenAIExactRouteTransport(
        model_io=OpenAIModelIO(
            model="gpt-4.1",
            api_key="secret",
            client_factory=lambda **_kwargs: _Client(),
            default_payloads={},
            model_capabilities={},
        ),
        catalog=catalog,
        callback=external_events.append,
        run_id="exact-openai-run",
        emit_stream=True,
    )

    transport.send(
        envelope=envelope,
        route=envelope.routes[0],
        retry_ordinal=0,
    )

    assert external_events == []
    assert [event["type"] for event in transport.buffered_events] == [
        "request_messages",
        "token_delta",
    ]
    transport.release_buffered_events()
    assert [event["type"] for event in external_events] == [
        "request_messages",
        "token_delta",
    ]
    assert transport.buffered_events == ()


def test_anthropic_exact_transport_sends_native_route_once_with_sdk_retry_off():
    from unchain.providers.exact_route_transport import AnthropicExactRouteTransport

    request = {
        "model": "claude-3-7-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        "max_tokens": 32,
    }
    catalog, envelope = _envelope(
        provider="anthropic",
        model="claude-3-7-sonnet",
        request=request,
    )
    init_calls: list[dict] = []
    send_calls: list[dict] = []

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage={"input_tokens": 2, "output_tokens": 0}
                ),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="exact anthropic"),
            )
            yield SimpleNamespace(
                type="message_delta",
                usage={"input_tokens": 2, "output_tokens": 2},
            )

    class _Messages:
        def stream(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            return _Stream()

    class _Client:
        messages = _Messages()

    def client_factory(**kwargs):
        init_calls.append(copy.deepcopy(kwargs))
        return _Client()

    model_io = AnthropicModelIO(
        model="claude-3-7-sonnet",
        api_key="secret",
        client_factory=client_factory,
        default_payloads={},
        model_capabilities={},
    )
    transport = AnthropicExactRouteTransport(model_io=model_io, catalog=catalog)

    result = transport.send(
        envelope=envelope,
        route=envelope.routes[0],
        retry_ordinal=0,
    )

    assert init_calls == [
        {
            "api_key": "secret",
            "timeout": model_io._ANTHROPIC_TIMEOUT,
            "max_retries": 0,
        }
    ]
    assert send_calls == [envelope.routes[0].request_copy()]
    assert result.final_text == "exact anthropic"
    assert result.provider_replay_frame["items"][:1] == request["messages"]


def test_hyperspace_exact_transport_reuses_anthropic_protocol_once():
    from unchain.providers.exact_route_transport import HyperspaceExactRouteTransport

    request = {
        "model": "hyperspace-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        "max_tokens": 16,
    }
    catalog, envelope = _envelope(
        provider="hyperspace",
        model="hyperspace-model",
        request=request,
    )
    init_calls: list[dict] = []
    send_calls: list[dict] = []

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="exact hyperspace"),
            )

    class _Messages:
        def stream(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            return _Stream()

    class _Client:
        messages = _Messages()

    def client_factory(**kwargs):
        init_calls.append(copy.deepcopy(kwargs))
        return _Client()

    model_io = HyperspaceModelIO(
        model="hyperspace-model",
        api_key="secret",
        client_factory=client_factory,
        default_payloads={},
        model_capabilities={},
    )
    transport = HyperspaceExactRouteTransport(model_io=model_io, catalog=catalog)

    result = transport.send(
        envelope=envelope,
        route=envelope.routes[0],
        retry_ordinal=0,
    )

    assert init_calls == [
        {
            "api_key": "secret",
            "timeout": model_io._ANTHROPIC_TIMEOUT,
            "max_retries": 0,
        }
    ]
    assert send_calls == [envelope.routes[0].request_copy()]
    assert result.final_text == "exact hyperspace"
    assert result.provider_replay_frame["format"] == "anthropic.messages.v1"


def test_ollama_exact_transport_posts_the_persisted_body_once():
    from unchain.providers.exact_route_transport import OllamaExactRouteTransport

    request = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": "hello"}],
        "options": {"temperature": 0.25},
        "stream": True,
    }
    catalog, envelope = _envelope(
        provider="ollama",
        model="qwen3",
        request=request,
    )
    send_calls: list[dict] = []

    class _Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield json.dumps(
                {
                    "message": {"role": "assistant", "content": "exact ollama"},
                    "prompt_eval_count": 2,
                    "eval_count": 2,
                    "done": True,
                }
            )

        def read(self):
            return b""

    def stream_factory(method, url, **kwargs):
        send_calls.append(
            {
                "method": method,
                "url": url,
                **copy.deepcopy(kwargs),
            }
        )
        return _Response()

    model_io = OllamaModelIO(
        model="qwen3",
        base_url="http://ollama.test",
        stream_factory=stream_factory,
        default_payloads={},
        model_capabilities={},
    )
    transport = OllamaExactRouteTransport(model_io=model_io, catalog=catalog)

    result = transport.send(
        envelope=envelope,
        route=envelope.routes[0],
        retry_ordinal=0,
    )

    assert send_calls == [
        {
            "method": "POST",
            "url": "http://ollama.test/api/chat",
            "json": envelope.routes[0].request_copy(),
            "timeout": None,
        }
    ]
    assert result.final_text == "exact ollama"
    assert result.provider_replay_frame["items"][:1] == request["messages"]


def test_openai_exact_transport_classifies_fallback_without_sending_it():
    from unchain.providers.durable_turn_runtime import (
        ExactProviderRouteFailure,
        ExactProviderRouteFailureKind,
    )

    catalog = _catalog("openai", "gpt-4.1")
    primary = {
        "model": "gpt-4.1",
        "input": [{"role": "user", "content": "delta"}],
        "previous_response_id": "response-missing",
        "stream": True,
    }
    fallback = {
        "model": "gpt-4.1",
        "input": [{"role": "user", "content": "complete replay"}],
        "stream": True,
    }
    envelope = ProviderWireEnvelope(
        attempt=ATTEMPT,
        iteration=2,
        provider="openai",
        configured_model="gpt-4.1",
        request_model="gpt-4.1",
        adapter_revision="unchain.openai.responses.request.v1",
        transport_kind="openai.responses.create",
        transport_target_sha256="2" * 64,
        source_request_sha256="3" * 64,
        source_payload_sha256="4" * 64,
        catalog_sha256=catalog.catalog_sha256,
        prompt_sha256=catalog.prompt_sha256,
        tool_schema_sha256=catalog.tool_schema_sha256,
        required_betas=(),
        base_anthropic_betas=(),
        routes=(
            ProviderWireRoute(name="primary", request=primary),
            ProviderWireRoute(
                name="openai_previous_response_fallback",
                request=fallback,
            ),
        ),
    )
    send_calls: list[dict] = []

    class _PreviousResponseError(RuntimeError):
        status_code = 404

    class _Responses:
        def create(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            raise _PreviousResponseError("previous_response_id not_found")

    class _Client:
        responses = _Responses()

    transport = OpenAIExactRouteTransport(
        model_io=OpenAIModelIO(
            model="gpt-4.1",
            api_key="secret",
            client_factory=lambda **_kwargs: _Client(),
            default_payloads={},
            model_capabilities={},
        ),
        catalog=catalog,
    )

    with pytest.raises(ExactProviderRouteFailure) as caught:
        transport.send(
            envelope=envelope,
            route=envelope.routes[0],
            retry_ordinal=0,
        )

    assert caught.value.kind is ExactProviderRouteFailureKind.PREVIOUS_RESPONSE_FALLBACK
    assert send_calls == [primary]
    assert transport.buffered_events == ()


def test_anthropic_exact_transport_classifies_explicit_rate_limit_as_retry_safe():
    from unchain.providers.durable_turn_runtime import (
        ExactProviderRouteFailure,
        ExactProviderRouteFailureKind,
    )
    from unchain.providers.exact_route_transport import AnthropicExactRouteTransport

    request = {
        "model": "claude-3-7-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        "max_tokens": 32,
    }
    catalog, envelope = _envelope(
        provider="anthropic",
        model="claude-3-7-sonnet",
        request=request,
    )
    sends = []

    class _RateLimitError(RuntimeError):
        status_code = 429

    class _Messages:
        def stream(self, **kwargs):
            sends.append(copy.deepcopy(kwargs))
            raise _RateLimitError("rate limited")

    class _Client:
        messages = _Messages()

    transport = AnthropicExactRouteTransport(
        model_io=AnthropicModelIO(
            model="claude-3-7-sonnet",
            api_key="secret",
            client_factory=lambda **_kwargs: _Client(),
            default_payloads={},
            model_capabilities={},
        ),
        catalog=catalog,
    )

    with pytest.raises(ExactProviderRouteFailure) as caught:
        transport.send(
            envelope=envelope,
            route=envelope.routes[0],
            retry_ordinal=0,
        )

    assert caught.value.kind is ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE
    assert sends == [request]
    assert transport.buffered_events == ()


def test_openai_exact_transport_classifies_explicit_rate_limit_as_retry_safe():
    from unchain.providers.durable_turn_runtime import (
        ExactProviderRouteFailure,
        ExactProviderRouteFailureKind,
    )

    request = {
        "model": "gpt-4.1",
        "input": [{"role": "user", "content": "hello"}],
        "stream": True,
    }
    catalog, envelope = _envelope(
        provider="openai",
        model="gpt-4.1",
        request=request,
    )
    sends = []

    class _RateLimitError(RuntimeError):
        status_code = 429

    class _Responses:
        def create(self, **kwargs):
            sends.append(copy.deepcopy(kwargs))
            raise _RateLimitError("rate limited")

    class _Client:
        responses = _Responses()

    transport = OpenAIExactRouteTransport(
        model_io=OpenAIModelIO(
            model="gpt-4.1",
            api_key="secret",
            client_factory=lambda **_kwargs: _Client(),
            default_payloads={},
            model_capabilities={},
        ),
        catalog=catalog,
    )

    with pytest.raises(ExactProviderRouteFailure) as caught:
        transport.send(
            envelope=envelope,
            route=envelope.routes[0],
            retry_ordinal=0,
        )

    assert caught.value.kind is ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE
    assert sends == [request]


def test_anthropic_exact_transport_classifies_explicit_bad_request_as_terminal():
    from unchain.providers.durable_turn_runtime import (
        ExactProviderRouteFailure,
        ExactProviderRouteFailureKind,
    )
    from unchain.providers.exact_route_transport import AnthropicExactRouteTransport

    request = {
        "model": "claude-3-7-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        "max_tokens": 32,
    }
    catalog, envelope = _envelope(
        provider="anthropic",
        model="claude-3-7-sonnet",
        request=request,
    )

    class _BadRequestError(RuntimeError):
        status_code = 400

    class _Messages:
        def stream(self, **_kwargs):
            raise _BadRequestError("bad request")

    class _Client:
        messages = _Messages()

    transport = AnthropicExactRouteTransport(
        model_io=AnthropicModelIO(
            model="claude-3-7-sonnet",
            api_key="secret",
            client_factory=lambda **_kwargs: _Client(),
            default_payloads={},
            model_capabilities={},
        ),
        catalog=catalog,
    )

    with pytest.raises(ExactProviderRouteFailure) as caught:
        transport.send(
            envelope=envelope,
            route=envelope.routes[0],
            retry_ordinal=0,
        )

    assert caught.value.kind is ExactProviderRouteFailureKind.TERMINAL


@pytest.mark.parametrize('status,expected', [
    (400, 'terminal'), (401, 'terminal'), (404, 'terminal'), (429, 'transient_retry_safe'),
    (502, 'transient_retry_safe'), (503, 'transient_retry_safe'),
    # 500 and 504 may have been processed by the provider, so they stay unclassified.
    (500, None), (504, None),
])
def test_google_sdk_http_failures_have_exact_retry_classification(status, expected):
    import httpx
    from google.genai.errors import APIError
    from unchain.providers.exact_route_transport import _classified_failure_kind

    error = APIError(
        status, {'error': {'code': status, 'message': 'private'}},
        httpx.Response(status, request=httpx.Request('POST', 'https://example.invalid/gemini')),
    )
    kind = _classified_failure_kind(error)
    assert (kind.value if kind else None) == expected


def test_google_503_requires_the_actual_http_response_to_be_rejected():
    import httpx
    from google.genai.errors import APIError
    from unchain.providers.durable_turn_runtime import ExactProviderRouteFailureKind
    from unchain.providers.exact_route_transport import _classified_failure_kind

    body = {"error": {"code": 503, "message": "provider content is private"}}
    request = httpx.Request("POST", "https://example.test/stream")
    rejected = APIError(503, body, httpx.Response(503, request=request))
    streamed_error = APIError(503, body, httpx.Response(200, request=request))
    missing_response = APIError(503, body)
    mismatched_code = APIError(500, body, httpx.Response(503, request=request))
    spoofed_status = RuntimeError("not an SDK response")
    spoofed_status.status_code = 503

    assert (
        _classified_failure_kind(rejected)
        is ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE
    )
    assert _classified_failure_kind(streamed_error) is None
    assert _classified_failure_kind(missing_response) is None
    assert _classified_failure_kind(mismatched_code) is None
    assert _classified_failure_kind(spoofed_status) is None


@pytest.mark.parametrize("provider_code", [429, 502, 503, 529])
@pytest.mark.parametrize("http_status", [None, 200])
def test_google_numeric_error_codes_without_rejected_http_response_are_not_retry_evidence(
    provider_code, http_status,
):
    import httpx
    from google.genai.errors import APIError
    from unchain.providers.exact_route_transport import _classified_failure_kind

    response = (
        None if http_status is None else
        httpx.Response(http_status, request=httpx.Request("POST", "https://example.invalid/gemini"))
    )
    error = APIError(provider_code, {"error": {"code": provider_code, "message": "PRIVATE"}}, response)
    assert _classified_failure_kind(error) is None


def test_arbitrary_numeric_error_code_is_not_http_evidence():
    from unchain.providers.exact_route_transport import _classified_failure_kind

    error = RuntimeError('private')
    error.code = 429
    assert _classified_failure_kind(error) is None


@pytest.mark.parametrize(
    ("status", "provider_status", "expected"),
    [
        (503, "UNAVAILABLE", "TRANSIENT_RETRY_SAFE"),
        (502, "UNAVAILABLE", "TRANSIENT_RETRY_SAFE"),
        (429, "RESOURCE_EXHAUSTED", "TRANSIENT_RETRY_SAFE"),
        (404, "NOT_FOUND", "TERMINAL"),
    ],
)
def test_gemini_exact_transport_classifies_real_sdk_http_failures(
    status, provider_status, expected
):
    import httpx
    from google.genai import errors

    from unchain.providers.durable_turn_runtime import (
        ExactProviderRouteFailure,
        ExactProviderRouteFailureKind,
    )
    from unchain.providers.exact_route_transport import GeminiExactRouteTransport
    from unchain.providers.gemini import GeminiModelIO

    request = {
        "model": "gemini-3.8-flash",
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        "config": {
            "automatic_function_calling": {"disable": True},
            "max_output_tokens": 64,
        },
    }
    catalog, envelope = _envelope(
        provider="gemini", model="gemini-3.8-flash", request=request
    )
    sends = []
    error_type = errors.ServerError if status >= 500 else errors.ClientError

    def factory(**_kwargs):
        def send(**kwargs):
            sends.append(kwargs["model"])
            raise error_type(
                status,
                {"error": {"code": status, "message": "private", "status": provider_status}},
                httpx.Response(
                    status,
                    request=httpx.Request("POST", "https://example.invalid/gemini"),
                ),
            )

        return SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=send), close=lambda: None
        )

    transport = GeminiExactRouteTransport(
        model_io=GeminiModelIO(
            model="gemini-3.8-flash", api_key="secret", client_factory=factory
        ),
        catalog=catalog,
    )

    with pytest.raises(ExactProviderRouteFailure) as caught:
        transport.send(envelope=envelope, route=envelope.routes[0], retry_ordinal=0)

    assert caught.value.kind is ExactProviderRouteFailureKind[expected]
    assert sends == ["gemini-3.8-flash"]
