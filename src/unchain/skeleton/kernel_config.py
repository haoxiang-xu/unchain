"""Provider-free admission of the fixed configuration of a Kernel Loop node."""
from __future__ import annotations

from copy import deepcopy
from urllib.parse import urlsplit

from .errors import SkeletonValidationError
from .io import _closed_object, _validate_json


def validate_kernel_config(value, path=("kernel",)):
    _validate_json(value, path)
    config = _closed_object(value, path, {
        "schema_version", "profile", "prompt", "max_iterations", "output_tokens",
        "history", "overflow", "json_schema",
    })
    def reject(code, tail):
        raise SkeletonValidationError(code, path + tail, "invalid Kernel Loop configuration")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        reject("unknown_variant", ("schema_version",))
    profile = _closed_object(config["profile"], path + ("profile",), {
        "id", "provider", "model", "protocol", "endpoint", "credential_ref",
        "context_window", "output_limit", "capabilities",
    })
    for key in ("id", "provider", "model", "protocol", "endpoint", "credential_ref"):
        if type(profile[key]) is not str or not profile[key].strip() or profile[key] != profile[key].strip():
            reject("invalid_value", ("profile", key))
    if profile["protocol"] not in {"openai-responses", "anthropic", "gemini", "ollama"}:
        reject("unsupported_capability", ("profile", "protocol"))
    try:
        endpoint = urlsplit(profile["endpoint"])
        if (endpoint.scheme not in {"https", "http"} or not endpoint.hostname
                or endpoint.username is not None or endpoint.password is not None
                or endpoint.query or endpoint.fragment):
            reject("invalid_value", ("profile", "endpoint"))
        endpoint.port
    except ValueError:
        reject("invalid_value", ("profile", "endpoint"))
    for key in ("context_window", "output_limit"):
        if type(profile[key]) is not int or profile[key] < 1:
            reject("invalid_value", ("profile", key))
    caps = _closed_object(profile["capabilities"], path + ("profile", "capabilities"), {"json_schema", "images", "files"})
    if any(type(flag) is not bool for flag in caps.values()):
        reject("invalid_type", ("profile", "capabilities"))
    for key in ("max_iterations", "output_tokens"):
        if type(config[key]) is not int or config[key] < 1:
            reject("invalid_value", (key,))
    if config["output_tokens"] > profile["output_limit"] or config["output_tokens"] >= profile["context_window"]:
        reject("invalid_value", ("output_tokens",))
    if type(config["prompt"]) is not str:
        reject("invalid_type", ("prompt",))
    if config["history"] not in ("root", "none"):
        reject("unknown_variant", ("history",))
    if config["overflow"] not in ("error", "compress"):
        reject("unknown_variant", ("overflow",))
    if config["json_schema"] is not None:
        if not caps["json_schema"]:
            reject("unsupported_capability", ("json_schema",))
        from .kernel_output import validate_output_schema
        validate_output_schema(config["json_schema"])
    return deepcopy(config)
