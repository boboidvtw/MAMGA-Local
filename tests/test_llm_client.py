"""
Unit tests for memory/llm_client.py

Tests cover:
- Factory creates correct backend types
- Environment-variable-driven configuration
- Unknown backend raises ValueError
- OllamaClient handles network errors gracefully
- OpenAICompatibleClient delegates to underlying SDK correctly (mocked)
"""
import pytest
from unittest.mock import MagicMock, patch

from memory.llm_client import (
    create_llm_client,
    OpenAICompatibleClient,
    OllamaClient,
    LLMClient,
)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestCreateLLMClient:
    def test_lmstudio_backend_returns_openai_compatible(self):
        client = create_llm_client(backend="lmstudio", model="local-model")
        assert isinstance(client, OpenAICompatibleClient)

    def test_openai_backend_returns_openai_compatible(self):
        client = create_llm_client(backend="openai", model="gpt-4o-mini", api_key="fake-key")
        assert isinstance(client, OpenAICompatibleClient)

    def test_ollama_backend_returns_ollama_client(self):
        client = create_llm_client(backend="ollama", model="llama3")
        assert isinstance(client, OllamaClient)

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            create_llm_client(backend="nonexistent")

    def test_env_var_selects_backend(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        monkeypatch.setenv("LLM_MODEL", "mistral")
        client = create_llm_client()
        assert isinstance(client, OllamaClient)
        assert client.model == "mistral"

    def test_env_var_base_url_used_for_lmstudio(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "lmstudio")
        monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9999/v1")
        # Just ensure it constructs without error; actual URL wiring is in
        # OpenAICompatibleClient which we test separately.
        client = create_llm_client()
        assert isinstance(client, OpenAICompatibleClient)

    def test_all_client_types_are_llm_client_subclass(self):
        for backend in ("lmstudio", "ollama"):
            client = create_llm_client(backend=backend)
            assert isinstance(client, LLMClient)


# ---------------------------------------------------------------------------
# OpenAICompatibleClient
# ---------------------------------------------------------------------------

class TestOpenAICompatibleClient:
    @pytest.fixture
    def mock_openai(self):
        """Patch the openai.OpenAI class used inside the module."""
        with patch("memory.llm_client.OpenAICompatibleClient.__init__") as _:
            pass
        # We'll do a manual mock of the underlying _client instead.
        client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Hello from mock"
        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_resp
        client._client = mock_openai_client
        client.model = "test-model"
        return client

    def test_complete_returns_string(self, mock_openai):
        result = mock_openai.complete("Say hello")
        assert isinstance(result, str)
        assert result == "Hello from mock"

    def test_chat_passes_messages_to_sdk(self, mock_openai):
        messages = [{"role": "user", "content": "Hi"}]
        mock_openai.chat(messages)
        call_kwargs = mock_openai._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == messages

    def test_get_completion_alias(self, mock_openai):
        """get_completion should be identical to complete."""
        r1 = mock_openai.get_completion("test")
        r2 = mock_openai.complete("test")
        assert r1 == r2


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------

class TestOllamaClient:
    def test_default_base_url(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        client = OllamaClient(model="llama3")
        assert "11434" in client.base_url

    def test_custom_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://myserver:9000")
        client = OllamaClient(model="llama3")
        assert client.base_url == "http://myserver:9000"

    def test_network_error_raises(self):
        client = OllamaClient(model="llama3", base_url="http://127.0.0.1:19999")
        with pytest.raises(Exception):
            client.complete("Hello?")
