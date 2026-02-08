from __future__ import annotations

from pydantic import SecretStr

from kimi_cli.config import LLMModel, LLMProvider, get_default_config
from kimi_cli.ui.shell.slash import _is_openai_chat_model, _upsert_openai_models


def test_is_openai_chat_model_filters_non_chat_models() -> None:
    assert _is_openai_chat_model("gpt-4.1")
    assert _is_openai_chat_model("o3-mini")
    assert _is_openai_chat_model("chatgpt-4o-latest")
    assert not _is_openai_chat_model("text-embedding-3-large")
    assert not _is_openai_chat_model("whisper-1")


def test_upsert_openai_models_replaces_existing_openai_entries() -> None:
    config = get_default_config()
    config.providers["openai"] = LLMProvider(
        type="openai_responses",
        base_url="https://old.example/v1",
        api_key=SecretStr("old-key"),
    )
    config.models["openai/old"] = LLMModel(
        provider="openai",
        model="gpt-old",
        max_context_size=100000,
    )
    config.models["kimi/model"] = LLMModel(
        provider="kimi",
        model="kimi-k2.5",
        max_context_size=262144,
    )

    _upsert_openai_models(
        config,
        model_ids=["gpt-4.1-mini", "o3-mini"],
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        max_context_size=128000,
    )

    assert config.providers["openai"].base_url == "https://api.openai.com/v1"
    assert config.providers["openai"].api_key.get_secret_value() == "sk-test"
    assert "openai/old" not in config.models
    assert "openai/gpt-4.1-mini" in config.models
    assert "openai/o3-mini" in config.models
    assert "kimi/model" in config.models
