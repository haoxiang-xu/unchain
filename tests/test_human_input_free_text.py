import json

import pytest

from unchain.input import (
    HUMAN_INPUT_OTHER_VALUE,
    HumanInputOption,
    HumanInputRequest,
    HumanInputResponse,
)
from unchain.tools import render_tool_prompt_block
from unchain.toolkits import CoreToolkit


def _arguments(**overrides):
    arguments = {
        "title": "Input needed",
        "question": "What folder should I use?",
        "selection_mode": "single",
        "options": [],
        "allow_other": True,
        "other_label": "Folder path",
        "other_placeholder": "/path/to/folder",
    }
    arguments.update(overrides)
    return arguments


@pytest.mark.parametrize("selection_mode", ["single", "multiple"])
def test_admits_empty_options_as_free_text_for_both_selection_modes(selection_mode):
    request = HumanInputRequest.from_tool_arguments(
        _arguments(selection_mode=selection_mode), request_id="req-free-text"
    )

    assert request.kind == "selector"
    assert request.options == []
    assert request.allow_other is True
    assert request.min_selected == 1
    assert request.max_selected == 1
    assert request.allowed_values() == {HUMAN_INPUT_OTHER_VALUE}


def test_admits_free_text_from_json_string():
    request = HumanInputRequest.from_tool_arguments(
        json.dumps(_arguments()), request_id="req-json"
    )

    assert request.other_label == "Folder path"
    assert request.other_placeholder == "/path/to/folder"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ([], "must be a dict or JSON string"),
        ("[]", "must be an object"),
        ('"text"', "must be an object"),
        ("null", "must be an object"),
        (_arguments(allow_other=False), "options must be a non-empty array"),
        (_arguments(options={}), "options must be an array"),
        (_arguments(allow_other="true"), "allow_other must be a boolean"),
        (_arguments(selection_mode=1), "selection_mode must be 'single' or 'multiple'"),
        (_arguments(unknown_field=True), "unknown argument field"),
    ],
)
def test_rejects_malformed_or_unknown_admission_arguments(arguments, message):
    with pytest.raises(ValueError, match=message):
        HumanInputRequest.from_tool_arguments(arguments, request_id="req-invalid")


def test_rejects_unknown_option_fields():
    with pytest.raises(ValueError, match="unknown option field"):
        HumanInputRequest.from_tool_arguments(
            _arguments(options=[{"label": "A", "value": "a", "unexpected": 1}]),
            request_id="req-invalid-option",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_selected": 2},
        {"max_selected": 0},
        {"max_selected": 2},
        {"min_selected": 1, "max_selected": 0},
    ],
)
def test_free_text_keeps_single_available_selection_constraints(overrides):
    with pytest.raises(ValueError):
        HumanInputRequest.from_tool_arguments(
            _arguments(**overrides), request_id="req-invalid-count"
        )


def test_rejects_duplicate_and_reserved_option_values():
    with pytest.raises(ValueError, match="duplicate option value"):
        HumanInputRequest.from_tool_arguments(
            _arguments(
                options=[
                    {"label": "A", "value": "same"},
                    {"label": "B", "value": "same"},
                ],
                allow_other=False,
            ),
            request_id="req-duplicate",
        )

    with pytest.raises(ValueError, match="reserved value"):
        HumanInputOption.from_raw({"label": "Other", "value": HUMAN_INPUT_OTHER_VALUE})


def test_free_text_request_roundtrips_through_dict_and_from_dict():
    request = HumanInputRequest.from_tool_arguments(
        _arguments(), request_id="req-roundtrip"
    )

    encoded = request.to_dict()
    restored = HumanInputRequest.from_dict(encoded)

    assert restored.to_dict() == encoded
    assert restored.kind == "selector"
    assert restored.options == []


def test_bounded_selector_and_other_response_remain_supported():
    request = HumanInputRequest.from_tool_arguments(
        _arguments(
            question="Which approach should I use?",
            options=[{"label": "Existing", "value": "existing"}],
            allow_other=True,
        ),
        request_id="req-bounded",
    )

    assert request.allowed_values() == {"existing", HUMAN_INPUT_OTHER_VALUE}
    assert HumanInputResponse.from_raw(
        {"request_id": "req-bounded", "selected_values": ["existing"]},
        request=request,
    ).to_tool_result() == {
        "submitted": True,
        "selected_values": ["existing"],
        "other_text": None,
    }


def test_free_text_response_requires_nonempty_other_text_and_preserves_payload():
    request = HumanInputRequest.from_tool_arguments(
        _arguments(), request_id="req-response"
    )

    with pytest.raises(ValueError, match="other_text is required"):
        HumanInputResponse.from_raw(
            {"request_id": request.request_id, "selected_values": [HUMAN_INPUT_OTHER_VALUE]},
            request=request,
        )

    response = HumanInputResponse.from_raw(
        {
            "request_id": request.request_id,
            "selected_values": [HUMAN_INPUT_OTHER_VALUE],
            "other_text": "  /Users/red/project  ",
        },
        request=request,
    )

    assert response.to_dict() == {
        "request_id": "req-response",
        "selected_values": [HUMAN_INPUT_OTHER_VALUE],
        "other_text": "/Users/red/project",
    }
    assert response.to_tool_result() == {
        "submitted": True,
        "selected_values": [HUMAN_INPUT_OTHER_VALUE],
        "other_text": "/Users/red/project",
    }


def test_free_text_response_rejects_wrong_id_unsupported_and_duplicate_values():
    request = HumanInputRequest.from_tool_arguments(
        _arguments(), request_id="req-response-errors"
    )

    invalid_responses = [
        {"request_id": "wrong", "selected_values": [HUMAN_INPUT_OTHER_VALUE], "other_text": "x"},
        {"request_id": request.request_id, "selected_values": ["path"], "other_text": "x"},
        {
            "request_id": request.request_id,
            "selected_values": [HUMAN_INPUT_OTHER_VALUE, HUMAN_INPUT_OTHER_VALUE],
            "other_text": "x",
        },
    ]
    for raw in invalid_responses:
        with pytest.raises(ValueError):
            HumanInputResponse.from_raw(raw, request=request)


def test_prompt_guidance_explains_free_text_shape():
    tk = CoreToolkit(workspace_root=".")
    tool_json = next(item for item in tk.to_json() if item["name"] == "ask_user_question")
    prompt_block = render_tool_prompt_block(tk)

    assert "options=[]" in tool_json["description"]
    assert "allow_other=true" in tool_json["description"]
    assert "folder path" in tool_json["description"].lower()
    assert "before making downstream feature-scope decisions" in tool_json["description"]
    assert "Never guess a local or OS-specific path" in tool_json["description"]
    assert "custom_path or enter_text options" in tool_json["description"]
    assert "empty array only with allow_other=true" in tool_json["parameters"]["properties"]["options"]["description"]
    assert "options=[]" in prompt_block
    assert "allow_other=true" in prompt_block
    assert 'selection_mode="single"' in prompt_block
    assert "other_placeholder=\"/path/to/folder\"" in prompt_block
    assert "blocking path or name directly" in prompt_block
    assert "invent custom_path/enter_text options" in prompt_block
    assert "folder path" in prompt_block.lower()
