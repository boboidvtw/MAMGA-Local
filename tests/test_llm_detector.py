"""
Unit tests for utils.llm_detector.

Focus: the logic that can be tested without touching the network. Live probing
is covered by a single end-to-end smoke test at the bottom which is skipped
unless `MAMGA_LIVE_PROBE=1` is set.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any
from unittest.mock import patch

import pytest

# Ensure the repo root is importable when pytest is invoked from tests/.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from utils import llm_detector as d  # noqa: E402


# --------------------------------------------------------------------------- #
# _extract_models                                                             #
# --------------------------------------------------------------------------- #


class TestExtractModels:
    def test_openai_compat(self):
        data = {"data": [{"id": "qwen3-9b"}, {"id": "gemma-4-e4b-it"}]}
        assert d._extract_models(data) == ("qwen3-9b", "gemma-4-e4b-it")

    def test_ollama_native(self):
        data = {"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5:7b"}]}
        assert d._extract_models(data) == ("llama3.1:8b", "qwen2.5:7b")

    def test_empty(self):
        assert d._extract_models({}) == ()
        assert d._extract_models({"data": []}) == ()

    def test_skips_malformed_entries(self):
        data = {"data": [{"id": "ok"}, {"not_id": "x"}, None]}
        assert d._extract_models(data) == ("ok",)


# --------------------------------------------------------------------------- #
# _fingerprint                                                                #
# --------------------------------------------------------------------------- #


class TestFingerprint:
    def test_lmstudio_owned_by(self):
        data = {"data": [{"id": "x", "owned_by": "organization_owner"}]}
        assert d._fingerprint(data, {}) == d.LLMPlatform.LM_STUDIO

    def test_ollama_server_header(self):
        assert d._fingerprint({}, {"server": "ollama/0.1.0"}) == d.LLMPlatform.OLLAMA

    def test_ollama_library_owner(self):
        data = {"data": [{"id": "x", "owned_by": "library"}]}
        assert d._fingerprint(data, {}) == d.LLMPlatform.OLLAMA

    def test_vllm_owner(self):
        data = {"data": [{"id": "x", "owned_by": "vllm"}]}
        assert d._fingerprint(data, {}) == d.LLMPlatform.VLLM

    def test_unknown_fallback(self):
        assert d._fingerprint({"data": []}, {}) == d.LLMPlatform.UNKNOWN


# --------------------------------------------------------------------------- #
# _score_model_name                                                           #
# --------------------------------------------------------------------------- #


class TestScoreModelName:
    @pytest.mark.parametrize(
        "name,expect_positive",
        [
            ("qwen3-9b", True),
            ("qwen3.5-14b-instruct", True),
            ("gemma-4-e4b-it", True),
            ("llama-3.1-8b-instruct", True),
            ("mistral-nemo-12b", True),
            ("phi-4", True),
            ("deepseek-v3", True),
        ],
    )
    def test_preferred_models_score_positive(self, name: str, expect_positive: bool):
        assert (d._score_model_name(name) > 0) is expect_positive

    @pytest.mark.parametrize(
        "name",
        [
            "text-embedding-nomic-embed-text-v1.5",
            "bge-large-en",
            "all-MiniLM-L6-v2",
            "rerank-multilingual-v3",
            "whisper-large-v3",
            "stable-diffusion-xl",
            "flux-dev",
        ],
    )
    def test_blocklist(self, name: str):
        assert d._score_model_name(name) == -1

    def test_size_sweet_spot_preferred(self):
        small = d._score_model_name("qwen3-3b")     # <7 -> penalty
        mid = d._score_model_name("qwen3-9b")       # 7..32 -> bonus
        huge = d._score_model_name("qwen3-72b")     # >70 -> penalty
        assert mid > small
        assert mid > huge

    def test_newer_family_wins(self):
        assert d._score_model_name("qwen3-9b") > d._score_model_name("qwen2.5-9b")


# --------------------------------------------------------------------------- #
# pick_best_model                                                             #
# --------------------------------------------------------------------------- #


def _ep(models: tuple[str, ...]) -> d.DetectedEndpoint:
    return d.DetectedEndpoint(
        platform=d.LLMPlatform.LM_STUDIO,
        base_url="http://localhost:1234/v1",
        models=models,
        raw_host="http://localhost:1234",
        latency_ms=5,
    )


class TestPickBestModel:
    def test_preferred_exact_match(self):
        ep = _ep(("qwen3-9b", "gemma-4-e4b-it"))
        assert d.pick_best_model(ep, preferred="gemma-4-e4b-it") == "gemma-4-e4b-it"

    def test_preferred_substring(self):
        ep = _ep(("qwen3-9b-instruct-q4", "gemma-4-e4b-it"))
        assert d.pick_best_model(ep, preferred="qwen3") == "qwen3-9b-instruct-q4"

    def test_prefers_qwen3_over_gemma3(self):
        ep = _ep(("gemma-3-4b-it", "qwen3-9b", "phi-4"))
        assert d.pick_best_model(ep) == "qwen3-9b"

    def test_skips_embeddings(self):
        ep = _ep(("text-embedding-nomic-embed", "qwen3-9b"))
        assert d.pick_best_model(ep) == "qwen3-9b"

    def test_all_embeddings_returns_none(self):
        ep = _ep(("bge-large-en", "text-embedding-3-small"))
        assert d.pick_best_model(ep) is None

    def test_empty_endpoint(self):
        assert d.pick_best_model(_ep(())) is None

    def test_unknown_model_falls_back_to_first_allowed(self):
        ep = _ep(("bge-large-en", "mystery-model-v1"))
        assert d.pick_best_model(ep) == "mystery-model-v1"


# --------------------------------------------------------------------------- #
# _platform_from_env                                                          #
# --------------------------------------------------------------------------- #


class TestPlatformFromEnv:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("lmstudio", d.LLMPlatform.LM_STUDIO),
            ("lm-studio", d.LLMPlatform.LM_STUDIO),
            ("LM_STUDIO", d.LLMPlatform.LM_STUDIO),
            ("ollama", d.LLMPlatform.OLLAMA),
            ("llama.cpp", d.LLMPlatform.LLAMA_CPP),
            ("vLLM", d.LLMPlatform.VLLM),
            ("koboldcpp", d.LLMPlatform.KOBOLD),
        ],
    )
    def test_aliases(self, raw: str, expected: d.LLMPlatform):
        with patch.dict(os.environ, {"LLM_BACKEND": raw}, clear=False):
            assert d._platform_from_env() == expected

    def test_auto_returns_none(self):
        with patch.dict(os.environ, {"LLM_BACKEND": "auto"}, clear=False):
            assert d._platform_from_env() is None

    def test_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            assert d._platform_from_env() is None


# --------------------------------------------------------------------------- #
# detect_platforms (mocked HTTP)                                              #
# --------------------------------------------------------------------------- #


def _fake_response(status: int, payload: Any, headers: dict | None = None):
    resp = types.SimpleNamespace()
    resp.status_code = status
    resp.headers = headers or {}
    resp.json = lambda: payload
    return resp


class TestDetectPlatforms:
    def test_picks_up_lmstudio(self):
        def fake_get(url: str, timeout: float):
            if "1234" in url:
                return _fake_response(
                    200,
                    {"data": [{"id": "qwen3-9b", "owned_by": "organization_owner"}]},
                )
            raise __import__("requests").exceptions.ConnectionError("refused")

        with patch.dict(os.environ, {}, clear=True), \
             patch("utils.llm_detector.requests.get", side_effect=fake_get):
            eps = d.detect_platforms(timeout=0.1)

        assert len(eps) == 1
        assert eps[0].platform == d.LLMPlatform.LM_STUDIO
        assert "qwen3-9b" in eps[0].models

    def test_picks_up_multiple_platforms(self):
        def fake_get(url: str, timeout: float):
            if "1234" in url:
                return _fake_response(
                    200, {"data": [{"id": "a", "owned_by": "organization_owner"}]}
                )
            if "11434" in url:
                return _fake_response(
                    200, {"data": [{"id": "llama3:8b", "owned_by": "library"}]}
                )
            raise __import__("requests").exceptions.ConnectionError("refused")

        with patch.dict(os.environ, {}, clear=True), \
             patch("utils.llm_detector.requests.get", side_effect=fake_get):
            eps = d.detect_platforms(timeout=0.1)

        platforms = [e.platform for e in eps]
        assert d.LLMPlatform.LM_STUDIO in platforms
        assert d.LLMPlatform.OLLAMA in platforms
        # Priority ordering: LM Studio before Ollama (from _CANDIDATES).
        assert platforms.index(d.LLMPlatform.LM_STUDIO) < platforms.index(d.LLMPlatform.OLLAMA)

    def test_nothing_reachable_returns_empty(self):
        def fake_get(url: str, timeout: float):
            raise __import__("requests").exceptions.ConnectionError("refused")

        with patch.dict(os.environ, {}, clear=True), \
             patch("utils.llm_detector.requests.get", side_effect=fake_get):
            assert d.detect_platforms(timeout=0.1) == []


# --------------------------------------------------------------------------- #
# auto_configure                                                              #
# --------------------------------------------------------------------------- #


class TestAutoConfigure:
    def test_manual_override_skips_selection(self):
        env = {
            "LLM_BASE_URL": "http://example.com:9999/v1",
            "LLM_MODEL": "my-custom-model",
            "OPENAI_API_KEY": "sk-fake",
        }
        with patch.dict(os.environ, env, clear=True), \
             patch("utils.llm_detector.requests.get",
                   side_effect=__import__("requests").exceptions.ConnectionError("refused")):
            cfg = d.auto_configure(timeout=0.1)

        assert cfg.base_url == env["LLM_BASE_URL"]
        assert cfg.model == env["LLM_MODEL"]
        assert cfg.api_key == "sk-fake"

    def test_nothing_found_raises(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("utils.llm_detector.requests.get",
                   side_effect=__import__("requests").exceptions.ConnectionError("refused")):
            with pytest.raises(RuntimeError, match="No local LLM server"):
                d.auto_configure(timeout=0.1)

    def test_selects_preferred_backend(self):
        def fake_get(url: str, timeout: float):
            if "1234" in url:
                return _fake_response(
                    200, {"data": [{"id": "gemma-4-e4b-it", "owned_by": "organization_owner"}]}
                )
            if "11434" in url:
                return _fake_response(
                    200, {"data": [{"id": "qwen3:14b", "owned_by": "library"}]}
                )
            raise __import__("requests").exceptions.ConnectionError("refused")

        with patch.dict(os.environ, {"LLM_BACKEND": "ollama"}, clear=True), \
             patch("utils.llm_detector.requests.get", side_effect=fake_get):
            cfg = d.auto_configure(timeout=0.1)

        assert cfg.platform == d.LLMPlatform.OLLAMA
        assert cfg.model == "qwen3:14b"

    def test_all_models_blocklisted_raises(self):
        def fake_get(url: str, timeout: float):
            if "1234" in url:
                return _fake_response(
                    200,
                    {"data": [
                        {"id": "bge-large-en", "owned_by": "organization_owner"},
                        {"id": "text-embedding-nomic", "owned_by": "organization_owner"},
                    ]},
                )
            raise __import__("requests").exceptions.ConnectionError("refused")

        with patch.dict(os.environ, {}, clear=True), \
             patch("utils.llm_detector.requests.get", side_effect=fake_get):
            with pytest.raises(RuntimeError, match="no usable"):
                d.auto_configure(timeout=0.1)


# --------------------------------------------------------------------------- #
# Optional live probe                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    os.environ.get("MAMGA_LIVE_PROBE") != "1",
    reason="Set MAMGA_LIVE_PROBE=1 to run against your real local LLM servers",
)
def test_live_detection_smoke():
    cfg = d.auto_configure(timeout=2.0)
    assert cfg.base_url.startswith("http")
    assert cfg.model
    assert cfg.endpoint is not None
    # LM Studio's /v1/models can be slow on a cold call; the probe timeout
    # guards the upper bound, we just want a non-negative measurement here.
    assert cfg.endpoint.latency_ms >= 0
