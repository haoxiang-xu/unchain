from __future__ import annotations

import base64
import copy
import json

import pytest

import unchain.context  # noqa: F401 - initialize the public context/provider graph
from unchain.kernel import ModelTurnRequest
from unchain.providers import OllamaModelIO, OpenAIModelIO
from unchain.providers.message_contract import ProviderMessageContractError
from unchain.providers.native import _translate_content_blocks_for_anthropic
from unchain.providers.wire_preparer import _normalize_openai_messages


def _canonical_media_messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect these"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(b"image").decode("ascii"),
                    },
                },
                {
                    "type": "pdf",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "filename": "report.pdf",
                        "data": base64.b64encode(b"pdf").decode("ascii"),
                    },
                },
            ],
        }
    ]


@pytest.mark.parametrize(
    "normalize",
    [
        lambda messages: OpenAIModelIO(
            model="gpt-test",
            api_key="test-key",
            default_payloads={},
            model_capabilities={},
        )._normalize_input_messages(messages),
        _normalize_openai_messages,
    ],
)
def test_openai_native_and_exact_wire_reject_unknown_message_fields(normalize) -> None:
    with pytest.raises(
        ProviderMessageContractError,
        match=r"input\[0\].*attachments|messages\[0\].*attachments",
    ):
        normalize(
            [
                {
                    "role": "user",
                    "content": "hello",
                    "attachments": [{"kind": "image"}],
                }
            ]
        )


def test_openai_projects_canonical_media_to_exact_input_blocks() -> None:
    normalized = _normalize_openai_messages(_canonical_media_messages())

    assert normalized == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "inspect these"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,aW1hZ2U=",
                },
                {
                    "type": "input_file",
                    "filename": "report.pdf",
                    "file_data": "data:application/pdf;base64,cGRm",
                },
            ],
        }
    ]


def test_anthropic_rejects_unknown_message_fields_after_projection() -> None:
    messages = [
        {
            "role": "user",
            "content": "hello",
            "attachments": [{"kind": "image"}],
        }
    ]

    with pytest.raises(
        ProviderMessageContractError,
        match=r"messages\[0\].*attachments",
    ):
        _translate_content_blocks_for_anthropic(messages)


def test_anthropic_projects_canonical_media_to_native_blocks() -> None:
    messages = _canonical_media_messages()
    _translate_content_blocks_for_anthropic(messages)

    assert [block["type"] for block in messages[0]["content"]] == [
        "text",
        "image",
        "document",
    ]
    assert messages[0]["content"][1]["source"]["type"] == "base64"
    assert messages[0]["content"][2]["source"]["type"] == "base64"


@pytest.mark.parametrize("parsed_output", [None, {"schema": {"answer": "ready"}}])
def test_anthropic_legacy_sdk_text_helper_is_removed_only_from_outbound_copy(
    parsed_output,
) -> None:
    stored_messages = [
        {"role": "user", "content": "continue"},
        {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "I checked the result.",
                "citations": None,
                "parsed_output": parsed_output,
            }],
        },
    ]
    outbound_messages = copy.deepcopy(stored_messages)

    _translate_content_blocks_for_anthropic(outbound_messages)

    assert stored_messages[1]["content"][0]["parsed_output"] == parsed_output
    assert outbound_messages[1]["content"] == [{
        "type": "text",
        "text": "I checked the result.",
        "citations": None,
    }]


def test_anthropic_legacy_sdk_helper_does_not_open_text_schema() -> None:
    messages = [
        {"role": "user", "content": "continue"},
        {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "I checked the result.",
                "parsed_output": None,
                "extension": {"untrusted": True},
            }],
        },
    ]

    with pytest.raises(ProviderMessageContractError, match="unknown fields: extension"):
        _translate_content_blocks_for_anthropic(messages)


def test_ollama_projects_base64_image_and_rejects_pdf() -> None:
    messages = _canonical_media_messages()
    image_only = [
        {
            "role": "user",
            "content": messages[0]["content"][:2],
        }
    ]

    from unchain.providers.native import _translate_content_blocks_for_ollama

    _translate_content_blocks_for_ollama(image_only)
    assert image_only == [
        {
            "role": "user",
            "content": "inspect these",
            "images": ["aW1hZ2U="],
        }
    ]

    with pytest.raises(
        ProviderMessageContractError,
        match="unsupported content block type: pdf",
    ):
        _translate_content_blocks_for_ollama(messages)


def test_ollama_replay_frame_uses_the_exact_projected_request_messages() -> None:
    captured: dict = {}

    class _Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            yield json.dumps(
                {
                    "message": {"content": "done"},
                    "done": True,
                }
            )

    def stream_factory(method, url, **kwargs):
        del method, url
        captured.update(kwargs)
        return _Response()

    result = OllamaModelIO(
        model="vision-test",
        stream_factory=stream_factory,
    ).fetch_turn(
        ModelTurnRequest(
            messages=[
                {
                    "role": "user",
                    "content": _canonical_media_messages()[0]["content"][:2],
                }
            ]
        )
    )

    expected = {
        "role": "user",
        "content": "inspect these",
        "images": ["aW1hZ2U="],
    }
    assert captured["json"]["messages"] == [expected]
    assert result.provider_replay_frame["items"][0] == expected
