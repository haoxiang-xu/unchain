"""A fixed Kernel must preserve the selected endpoint's authentication mode."""
from copy import deepcopy
import json

import anthropic
import httpx
import pytest

from unchain.skeleton.errors import SkeletonValidationError
from unchain.skeleton.kernel_config import validate_kernel_config
from unchain.skeleton.kernel_profile import bind_kernel_profile
from tests.test_skeleton_kernel_profile import _config


@pytest.mark.parametrize("auth_mode", [None, "bearer", "x-api-key"])
def test_real_anthropic_sdk_sends_only_the_fixed_authentication_mode(auth_mode, monkeypatch):
    config = _config(provider="custom.route", protocol="anthropic")
    if auth_mode is not None:
        config["profile"]["auth_mode"] = auth_mode
    selected_auth = auth_mode or "x-api-key"
    config["profile"]["endpoint"] = "https://profile.example"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "foreign-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "foreign-auth-token")
    requests = []

    def respond(request):
        requests.append(request)
        assert request.url == "https://profile.example/v1/messages"
        assert request.headers.get("authorization") == ("Bearer sentinel-fixed" if selected_auth == "bearer" else None)
        assert request.headers.get("x-api-key") == ("sentinel-fixed" if selected_auth == "x-api-key" else None)
        assert json.loads(request.content)["model"] == "model-a"
        return httpx.Response(200, json={"id": "msg_offline", "type": "message", "role": "assistant",
            "model": "model-a", "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})

    made = []
    def factory(**kwargs):
        made.append(deepcopy(kwargs))
        assert sum(kwargs.get(key) is not None for key in ("api_key", "auth_token")) == 1
        return anthropic.Anthropic(**kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    binding = bind_kernel_profile(config,
        profile_resolver=lambda _id: deepcopy(config["profile"]),
        credential_resolver=lambda _ref: "sentinel-fixed",
        client_factories={"hyperspace": factory})
    with binding.model_io._client_factory(api_key=binding.model_io.api_key) as sdk:
        assert sdk.messages.create(model="model-a", max_tokens=4,
            messages=[{"role": "user", "content": "local only"}]).content[0].text == "ok"
    assert len(requests) == 1
    assert made[0]["max_retries"] == 0
    assert made[0]["base_url"] == config["profile"]["endpoint"]
    # One explicit argument on new SDKs; Omit keeps the alternate header out
    # even when an older SDK reads the alternate environment credential.
    assert made[0]["api_key"] == (None if selected_auth == "bearer" else "sentinel-fixed")
    assert made[0]["auth_token"] == ("sentinel-fixed" if selected_auth == "bearer" else None)
    assert "sentinel-fixed" not in repr(binding)


@pytest.mark.parametrize("provider,protocol,auth_mode", [
    ("openai", "openai-responses", "x-api-key"),
    ("anthropic", "anthropic", "bearer"),
    ("gemini", "gemini", "bearer"),
    ("ollama", "ollama", "x-api-key"),
    ("ollama", "ollama", "bearer"),
    ("custom.route", "anthropic", None),
    ("custom.route", "anthropic", False),
    ("custom.route", "anthropic", " bearer"),
    ("custom.route", "anthropic", "header"),
])
def test_invalid_authentication_fails_before_profile_or_credential_resolution(provider, protocol, auth_mode):
    config = _config(provider=provider, protocol=protocol)
    config["profile"]["auth_mode"] = auth_mode
    calls = []
    with pytest.raises(SkeletonValidationError) as caught:
        bind_kernel_profile(config,
            profile_resolver=lambda _id: calls.append("profile"),
            credential_resolver=lambda _ref: calls.append("secret"))
    assert caught.value.path[-1] == "auth_mode"
    assert calls == []


def test_legacy_profile_is_not_rewritten_and_explicit_auth_changes_its_identity():
    config = _config(provider="custom.route", protocol="anthropic")
    assert validate_kernel_config(config) == config
    bindings = []
    for mode in [None, "x-api-key", "bearer"]:
        value = deepcopy(config)
        if mode is not None:
            value["profile"]["auth_mode"] = mode
        bindings.append(bind_kernel_profile(value,
            profile_resolver=lambda _id: deepcopy(value["profile"]),
            credential_resolver=lambda _ref: "sentinel-fixed"))
    assert len({binding.profile_digest for binding in bindings}) == 3


@pytest.mark.parametrize("mode,credential", [("bearer", None), ("none", "unexpected-key")])
def test_custom_ollama_authentication_never_silently_changes_mode(mode, credential):
    config = _config(provider="custom.local", protocol="ollama")
    config["profile"]["auth_mode"] = mode
    with pytest.raises(ValueError):
        bind_kernel_profile(config, profile_resolver=lambda _id: deepcopy(config["profile"]),
            credential_resolver=lambda _ref: credential)


def test_changed_authentication_in_resolver_is_rejected_before_reading_a_key():
    config = _config(provider="custom.route", protocol="anthropic")
    config["profile"]["auth_mode"] = "bearer"
    changed = deepcopy(config["profile"])
    changed["auth_mode"] = "x-api-key"
    reads = []
    with pytest.raises(ValueError, match="does not exactly match"):
        bind_kernel_profile(config, profile_resolver=lambda _id: changed,
            credential_resolver=lambda _ref: reads.append(_ref))
    assert reads == []
