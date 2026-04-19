"""
LLM Client Abstraction Layer
----------------------------
Provides a unified interface for multiple LLM backends:
  - openai   : OpenAI cloud API
  - lmstudio : LM Studio local server (OpenAI-compatible, port 1234)
  - llamacpp : llama.cpp llama-server (OpenAI-compatible, port 8080)
  - ollama   : Ollama local inference

All configuration is driven by environment variables so no
endpoint or key needs to be hardcoded anywhere in the codebase.

Usage:
    from memory.llm_client import create_llm_client

    client = create_llm_client()            # reads env vars
    reply  = client.complete("Hello!")
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Minimal interface every backend must implement."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return a text completion for *prompt*."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return a reply given an OpenAI-style *messages* list."""

    # Convenience wrapper kept for backwards compatibility
    def get_completion(self, prompt: str, **kwargs) -> str:
        return self.complete(prompt, **kwargs)


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible (LM Studio, vLLM, …) backend
# ---------------------------------------------------------------------------

class OpenAICompatibleClient(LLMClient):
    """
    Works with any server that speaks the OpenAI REST API:
      - OpenAI cloud  (base_url=None / default)
      - LM Studio     (base_url="http://localhost:1234/v1")
      - vLLM, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            ) from exc

        self.model = model
        resolved_key = api_key or os.getenv("OPENAI_API_KEY") or "no-key"
        kwargs: Dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    # ------------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        try:
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.error("OpenAI-compatible call failed: %s", exc)
            raise


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaClient(LLMClient):
    """
    Calls a locally running Ollama server.
    Default endpoint: http://localhost:11434
    Override via LLM_BASE_URL env var.
    """

    def __init__(self, model: str, base_url: Optional[str] = None) -> None:
        self.model = model
        self.base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or "http://localhost:11434"
        ).rstrip("/")

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        import json
        import urllib.request

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
        ).encode()

        url = f"{self.base_url}/api/chat"
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
            return data.get("message", {}).get("content", "")
        except Exception as exc:
            logger.error("Ollama call failed: %s", exc)
            raise


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_client(
    backend: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMClient:
    """
    Build an LLMClient from arguments or environment variables.

    Priority (highest → lowest):
      1. Explicit keyword arguments passed to this function
      2. Environment variables

    Environment variables:
      LLM_BACKEND   : "openai" | "lmstudio" | "llamacpp" | "ollama"  (default: "lmstudio")
      LLM_MODEL     : model name  (default varies by backend)
      LLM_BASE_URL  : override endpoint URL
      OPENAI_API_KEY: API key for openai backend

    Examples::

        # Use LM Studio running on the default port
        client = create_llm_client()

        # Use llama.cpp llama-server running on port 8080
        client = create_llm_client(backend="llamacpp", model="local-model")

        # Use OpenAI cloud
        client = create_llm_client(backend="openai", model="gpt-4o-mini")

        # Use Ollama with a custom model
        client = create_llm_client(backend="ollama", model="llama3")
    """
    resolved_backend = (backend or os.getenv("LLM_BACKEND") or "lmstudio").lower()
    resolved_base_url = base_url or os.getenv("LLM_BASE_URL")

    _defaults: Dict[str, str] = {
        "openai":    "gpt-4o-mini",
        "lmstudio":  "local-model",
        "llamacpp":  "local-model",
        "ollama":    "llama3",
    }
    resolved_model = model or os.getenv("LLM_MODEL") or _defaults.get(resolved_backend, "local-model")

    logger.info("Creating LLM client: backend=%s  model=%s", resolved_backend, resolved_model)

    if resolved_backend == "openai":
        return OpenAICompatibleClient(
            model=resolved_model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=resolved_base_url,  # None → use OpenAI cloud default
        )

    if resolved_backend == "lmstudio":
        url = resolved_base_url or "http://localhost:1234/v1"
        return OpenAICompatibleClient(
            model=resolved_model,
            api_key="lm-studio",  # LM Studio ignores the key
            base_url=url,
        )

    if resolved_backend == "llamacpp":
        # llama-server exposes an OpenAI-compatible API on port 8080 by default.
        # Start with: llama-server -m /path/to/model.gguf --port 8080
        url = resolved_base_url or "http://localhost:8080/v1"
        return OpenAICompatibleClient(
            model=resolved_model,
            api_key="no-key",  # llama-server does not require auth by default
            base_url=url,
        )

    if resolved_backend == "ollama":
        return OllamaClient(model=resolved_model, base_url=resolved_base_url)

    raise ValueError(
        f"Unknown LLM backend: '{resolved_backend}'. "
        "Choose from: openai, lmstudio, llamacpp, ollama"
    )
