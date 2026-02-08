from __future__ import annotations

from kimi_cli.app import _fallback_model_and_provider_from_env


def test_fallback_model_and_provider_defaults_to_kimi(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_NAME", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_MAX_CONTEXT_SIZE", raising=False)

    model, provider = _fallback_model_and_provider_from_env()

    assert model.provider == ""
    assert model.model == ""
    assert model.max_context_size == 100_000
    assert provider.type == "kimi"
    assert provider.base_url == ""
    assert provider.api_key.get_secret_value() == ""


def test_fallback_model_and_provider_uses_openai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.example/v1")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_MODEL_MAX_CONTEXT_SIZE", "222222")

    model, provider = _fallback_model_and_provider_from_env()

    assert model.provider == "env:openai"
    assert model.model == "gpt-4.1-mini"
    assert model.max_context_size == 222222
    assert provider.type == "openai_responses"
    assert provider.base_url == "https://api.openai.example/v1"
    assert provider.api_key.get_secret_value() == "sk-test-openai"
