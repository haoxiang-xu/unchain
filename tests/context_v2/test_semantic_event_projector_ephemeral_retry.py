"""BC-386-5: provider_retry is presentation and never reaches the journal."""

from __future__ import annotations

from unchain.context import projector as projector_module


def test_provider_retry_is_ephemeral():
    assert "provider_retry" in projector_module._EPHEMERAL_EVENT_TYPES
