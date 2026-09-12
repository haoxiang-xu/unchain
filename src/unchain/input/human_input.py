from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Literal

from ..tools.decorators import tool
from ..tools.models import _escape_control_chars_inside_json_strings, ToolParameter, ToolPromptSpec
from ..tools.tool import Tool

ASK_USER_QUESTION_TOOL_NAME = "ask_user_question"
HUMAN_INPUT_KIND_SELECTOR = "selector"
HUMAN_INPUT_OTHER_VALUE = "__other__"
_HUMAN_INPUT_ARGUMENT_KEYS = frozenset(
    {
        "title",
        "question",
        "selection_mode",
        "options",
        "allow_other",
        "other_label",
        "other_placeholder",
        "min_selected",
        "max_selected",
    }
)
_HUMAN_INPUT_OPTION_KEYS = frozenset({"label", "value", "description"})
ASK_USER_QUESTION_TOOL_DESCRIPTION = (
    "Ask the user a structured selector question and suspend the run until they respond. "
    "Use this when you need a concrete decision, clarification, or preference that would materially change "
    "the implementation or final result. Ask for the smallest concrete answer that unblocks the work. "
    "Do not ask meta-questions such as what to discuss first. "
    "The question must let the user directly choose an answer, and the options must be concrete candidate answers, "
    "not categories, dimensions, or discussion topics. "
    "For an unbounded answer such as a folder path or name, ask for that blocking value directly before making "
    "downstream feature-scope decisions; use options=[] with allow_other=true, selection_mode=\"single\", and "
    "a meaningful other_label such as \"Folder path\". Never guess a local or OS-specific path, and never "
    "invent custom_path or enter_text options. "
    "When several reasonable paths exist, ask the user instead of silently guessing."
)
ASK_USER_QUESTION_TOOL_PROMPT_SPEC = ToolPromptSpec(
    purpose=(
        "Ask the user for a concrete decision, clarification, or preference before continuing work that "
        "depends on that choice."
    ),
    when_to_use=(
        "Multiple plausible approaches or interpretations would materially change the result.",
        "You need the user's preference on product direction, UX behavior, technical approach, scope, or tradeoffs.",
        "A specific decision blocks implementation and should come from the user rather than a guess.",
    ),
    when_not_to_use=(
        "You can discover the answer from the repo, runtime state, or earlier user messages.",
        "You only want permission to continue, plan approval, or a meta-discussion about what to talk about next.",
        "Your options are categories, dimensions, or brainstorming topics instead of direct candidate answers.",
    ),
    examples=(
        'ask_user_question(title="Auth strategy", question="Which auth approach should I implement?", selection_mode="single", options=[{"label": "Session cookies", "value": "session", "description": "Server-managed sessions with cookies."}, {"label": "JWT", "value": "jwt", "description": "Stateless tokens for API clients."}])',
        'ask_user_question(title="Platform support", question="Which platforms should this first version support?", selection_mode="multiple", options=[{"label": "Desktop web", "value": "desktop_web"}, {"label": "Mobile web", "value": "mobile_web"}, {"label": "Native mobile", "value": "native_mobile"}], max_selected=2)',
        'ask_user_question(title="Project folder", question="What folder path should I use?", selection_mode="single", options=[], allow_other=true, other_label="Folder path", other_placeholder="/path/to/folder")',
    ),
    advanced_tips=(
        "Prefer 2-4 strong options and make each option a concrete answer the user can directly pick.",
        "If one option is the best default, put it first and say so in the option label or description.",
        'For free text such as a folder path or name, use options=[] with allow_other=true, '
        'selection_mode="single", and a meaningful other_label such as "Folder path".',
        "Ask for the blocking path or name directly before deciding downstream feature scope; never guess a local or OS-specific path or invent custom_path/enter_text options.",
        "Use this tool whenever a direct user answer is needed, including an unbounded folder path or name; use assistant text for discussion that does not require an answer.",
    ),
)


def is_human_input_tool_name(name: Any) -> bool:
    return name == ASK_USER_QUESTION_TOOL_NAME


def _parse_tool_arguments(arguments: dict[str, Any] | str | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return copy.deepcopy(arguments)
    if isinstance(arguments, str):
        raw = arguments.strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            repaired = _escape_control_chars_inside_json_strings(raw)
            return json.loads(repaired)
    raise ValueError("human input tool arguments must be a dict or JSON string")


def _clean_required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _clean_optional_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("optional text fields must be strings when provided")
    return value


def _clean_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer when provided")
    return value


@dataclass
class HumanInputOption:
    label: str
    value: str
    description: str = ""

    @classmethod
    def from_raw(cls, raw: Any) -> "HumanInputOption":
        if not isinstance(raw, dict):
            raise ValueError("each selector option must be an object")
        unknown_keys = set(raw) - _HUMAN_INPUT_OPTION_KEYS
        if unknown_keys:
            raise ValueError(f"unknown option field(s): {sorted(unknown_keys)!r}")

        label = _clean_required_text(raw.get("label"), "option.label")
        value = _clean_required_text(raw.get("value"), "option.value")
        description = _clean_optional_text(raw.get("description", ""))

        if value == HUMAN_INPUT_OTHER_VALUE:
            raise ValueError(f"option.value cannot use reserved value '{HUMAN_INPUT_OTHER_VALUE}'")

        return cls(label=label, value=value, description=description)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "value": self.value,
            "description": self.description,
        }


@dataclass
class HumanInputRequest:
    request_id: str
    kind: Literal["selector"]
    title: str
    question: str
    selection_mode: Literal["single", "multiple"]
    options: list[HumanInputOption]
    allow_other: bool = False
    other_label: str = "Other"
    other_placeholder: str = ""
    min_selected: int | None = None
    max_selected: int | None = None

    @classmethod
    def from_tool_arguments(
        cls,
        arguments: dict[str, Any] | str | None,
        *,
        request_id: str,
    ) -> "HumanInputRequest":
        raw = _parse_tool_arguments(arguments)
        if not isinstance(raw, dict):
            raise ValueError("human input tool arguments must be an object")
        unknown_keys = set(raw) - _HUMAN_INPUT_ARGUMENT_KEYS
        if unknown_keys:
            raise ValueError(f"unknown argument field(s): {sorted(unknown_keys)!r}")

        title = _clean_required_text(raw.get("title"), "title")
        question = _clean_required_text(raw.get("question"), "question")
        selection_mode = raw.get("selection_mode")
        if not isinstance(selection_mode, str) or selection_mode not in {"single", "multiple"}:
            raise ValueError("selection_mode must be 'single' or 'multiple'")

        raw_allow_other = raw.get("allow_other", False)
        if not isinstance(raw_allow_other, bool):
            raise ValueError("allow_other must be a boolean")
        allow_other = raw_allow_other

        raw_options = raw.get("options")
        if not isinstance(raw_options, list):
            raise ValueError("options must be an array")
        if not raw_options and not allow_other:
            raise ValueError("options must be a non-empty array unless allow_other is true")

        options = [HumanInputOption.from_raw(item) for item in raw_options]
        seen_values: set[str] = set()
        for option in options:
            if option.value in seen_values:
                raise ValueError(f"duplicate option value: {option.value}")
            seen_values.add(option.value)

        other_label = _clean_optional_text(raw.get("other_label", "Other")) or "Other"
        other_placeholder = _clean_optional_text(raw.get("other_placeholder", ""))
        min_selected = _clean_optional_int(raw.get("min_selected"), "min_selected")
        max_selected = _clean_optional_int(raw.get("max_selected"), "max_selected")

        if selection_mode == "single":
            if min_selected is None:
                min_selected = 1
            if max_selected is None:
                max_selected = 1
            if min_selected not in {0, 1}:
                raise ValueError("single selection mode only supports min_selected of 0 or 1")
            if max_selected != 1:
                raise ValueError("single selection mode requires max_selected to be 1")
        else:
            allowed_total = len(options) + (1 if allow_other else 0)
            if min_selected is None:
                min_selected = 1
            if max_selected is None:
                max_selected = allowed_total

        if min_selected is None or max_selected is None:
            raise ValueError("min_selected and max_selected could not be resolved")
        if min_selected < 0:
            raise ValueError("min_selected must be >= 0")
        if max_selected < 1:
            raise ValueError("max_selected must be >= 1")
        if min_selected > max_selected:
            raise ValueError("min_selected cannot be greater than max_selected")

        allowed_total = len(options) + (1 if allow_other else 0)
        if max_selected > allowed_total:
            raise ValueError("max_selected cannot exceed the total number of available selections")

        return cls(
            request_id=request_id,
            kind=HUMAN_INPUT_KIND_SELECTOR,
            title=title,
            question=question,
            selection_mode=selection_mode,
            options=options,
            allow_other=allow_other,
            other_label=other_label,
            other_placeholder=other_placeholder,
            min_selected=min_selected,
            max_selected=max_selected,
        )

    @classmethod
    def from_dict(cls, raw: Any) -> "HumanInputRequest":
        if isinstance(raw, HumanInputRequest):
            return raw
        if not isinstance(raw, dict):
            raise ValueError("human input request must be an object")
        request_id = _clean_required_text(raw.get("request_id"), "request_id")
        kind = raw.get("kind")
        if kind != HUMAN_INPUT_KIND_SELECTOR:
            raise ValueError("human input request kind must be 'selector'")
        payload = {
            "title": raw.get("title"),
            "question": raw.get("question"),
            "selection_mode": raw.get("selection_mode"),
            "options": raw.get("options"),
            "allow_other": raw.get("allow_other", False),
            "other_label": raw.get("other_label", "Other"),
            "other_placeholder": raw.get("other_placeholder", ""),
            "min_selected": raw.get("min_selected"),
            "max_selected": raw.get("max_selected"),
        }
        return cls.from_tool_arguments(payload, request_id=request_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "kind": self.kind,
            "title": self.title,
            "question": self.question,
            "selection_mode": self.selection_mode,
            "options": [option.to_dict() for option in self.options],
            "allow_other": self.allow_other,
            "other_label": self.other_label,
            "other_placeholder": self.other_placeholder,
            "min_selected": self.min_selected,
            "max_selected": self.max_selected,
        }

    def allowed_values(self) -> set[str]:
        values = {option.value for option in self.options}
        if self.allow_other:
            values.add(HUMAN_INPUT_OTHER_VALUE)
        return values


@dataclass
class HumanInputResponse:
    request_id: str
    selected_values: list[str]
    other_text: str | None = None

    @classmethod
    def from_raw(
        cls,
        raw: Any,
        *,
        request: HumanInputRequest,
    ) -> "HumanInputResponse":
        if isinstance(raw, HumanInputResponse):
            raw = raw.to_dict()
        if not isinstance(raw, dict):
            raise ValueError("human input response must be an object")

        request_id = _clean_required_text(raw.get("request_id"), "request_id")
        if request_id != request.request_id:
            raise ValueError("human input response request_id does not match the pending request")

        selected_values = raw.get("selected_values")
        if not isinstance(selected_values, list):
            raise ValueError("selected_values must be an array")
        normalized_values: list[str] = []
        seen_values: set[str] = set()
        for value in selected_values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("selected_values must contain non-empty strings")
            normalized = value.strip()
            if normalized in seen_values:
                raise ValueError(f"duplicate selected value: {normalized}")
            seen_values.add(normalized)
            normalized_values.append(normalized)

        count = len(normalized_values)
        min_selected = request.min_selected or 0
        max_selected = request.max_selected or 0
        if count < min_selected:
            raise ValueError(f"selected_values must contain at least {min_selected} item(s)")
        if max_selected and count > max_selected:
            raise ValueError(f"selected_values cannot contain more than {max_selected} item(s)")

        allowed_values = request.allowed_values()
        invalid_values = [value for value in normalized_values if value not in allowed_values]
        if invalid_values:
            raise ValueError(f"selected_values contains unsupported option(s): {invalid_values}")

        raw_other_text = raw.get("other_text")
        other_text: str | None = None
        if raw_other_text is not None:
            if not isinstance(raw_other_text, str):
                raise ValueError("other_text must be a string when provided")
            other_text = raw_other_text.strip() or None

        selected_other = HUMAN_INPUT_OTHER_VALUE in normalized_values
        if selected_other:
            if not request.allow_other:
                raise ValueError("other selection is not allowed for this request")
            if not other_text:
                raise ValueError("other_text is required when '__other__' is selected")
        elif other_text:
            raise ValueError("other_text can only be provided when '__other__' is selected")

        return cls(
            request_id=request_id,
            selected_values=normalized_values,
            other_text=other_text,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "selected_values": list(self.selected_values),
            "other_text": self.other_text,
        }

    def to_tool_result(self) -> dict[str, Any]:
        return {
            "submitted": True,
            "selected_values": list(self.selected_values),
            "other_text": self.other_text,
        }


def build_ask_user_question_tool() -> Tool:
    option_item_schema = {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "User-facing label for a concrete answer the user can directly choose.",
            },
            "value": {"type": "string", "description": "Stable option value returned to the assistant."},
            "description": {
                "type": "string",
                "description": "Optional helper text clarifying what selecting this concrete answer means.",
            },
        },
        "required": ["label", "value"],
        "additionalProperties": False,
    }

    return tool(
        name=ASK_USER_QUESTION_TOOL_NAME,
        description=ASK_USER_QUESTION_TOOL_DESCRIPTION,
        func=lambda **_: {"error": "ask_user_question is a reserved runtime tool and cannot be executed directly"},
        prompt_spec=ASK_USER_QUESTION_TOOL_PROMPT_SPEC,
        parameters=[
            ToolParameter(
                name="title",
                description="Short title naming the concrete decision being made.",
                type_="string",
                required=True,
            ),
            ToolParameter(
                name="question",
                description=(
                    "Concrete user-facing question for the decision you need now. "
                    "Ask for a direct answer, not which dimensions, topics, or categories to explore. "
                    "For an unbounded value such as a folder path or name, ask directly for that value."
                ),
                type_="string",
                required=True,
            ),
            ToolParameter(
                name="selection_mode",
                description="Selection mode. Must be 'single' or 'multiple'.",
                type_="string",
                required=True,
            ),
            ToolParameter(
                name="options",
                description=(
                    "Concrete candidate answers shown to the user. "
                    "Do not use categories, placeholders, or meta options such as 'platform', 'tech', or 'scope'. "
                    "For free text, use an empty array only with allow_other=true and set a meaningful other_label."
                ),
                type_="array",
                required=True,
                items=option_item_schema,
            ),
            ToolParameter(
                name="allow_other",
                description=(
                    "Whether to allow a freeform concrete answer when the listed options are insufficient. "
                    "For an unbounded answer, set this true with options=[] and selection_mode=\"single\"."
                ),
                type_="boolean",
                required=False,
            ),
            ToolParameter(
                name="other_label",
                description="Meaningful input label for the freeform answer when allow_other is true, such as 'Folder path'.",
                type_="string",
                required=False,
            ),
            ToolParameter(
                name="other_placeholder",
                description="Placeholder for the freeform input shown when Other is selected.",
                type_="string",
                required=False,
            ),
            ToolParameter(
                name="min_selected",
                description="Minimum number of selections required.",
                type_="integer",
                required=False,
            ),
            ToolParameter(
                name="max_selected",
                description="Maximum number of selections allowed.",
                type_="integer",
                required=False,
            ),
        ],
    )

__all__ = [
    "ASK_USER_QUESTION_TOOL_NAME",
    "HUMAN_INPUT_KIND_SELECTOR",
    "HUMAN_INPUT_OTHER_VALUE",
    "HumanInputOption",
    "HumanInputRequest",
    "HumanInputResponse",
    "is_human_input_tool_name",
    "build_ask_user_question_tool",
]
