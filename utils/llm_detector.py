"""
Local LLM Platform Auto-Detection for MAMGA-Local.

Probes common local LLM server endpoints (LM Studio, Ollama, llama.cpp server,
vLLM, text-generation-webui, LocalAI, KoboldCpp, Jan, GPT4All, Cortex), identifies
running platforms via response fingerprints, and picks a suitable model for MAMGA's
memory-extraction workload.

Design goals:
- Zero required configuration for the common path (one running local LLM).
- Deterministic priority order with env-var overrides.
- Fast parallel probing with strict per-endpoint timeouts.
- No third-party deps beyond stdlib + `requests` (already in the project).

Typical usage:
    from utils.llm_detector import auto_configure
    cfg = auto_configure()                 # returns LLMConfig
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    response = client.chat.completions.create(model=cfg.model, ...)

Env overrides (checked before probing):
    LLM_BASE_URL      full OpenAI-compat base URL, e.g. http://localhost:1234/v1
    LLM_MODEL         exact model id to use
    LLM_BACKEND       one of: lmstudio|ollama|llamacpp|vllm|tgw|localai|kobold|jan|auto
    OPENAI_API_KEY    API key (defaults to a placeholder for local servers)

Author: MAMGA-Local contributors. Date: 2026-04-11.
"""

from __future__ import annotations

import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable

try:
    import requests
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "llm_detector requires `requests`. Install via `pip install requests`."
    ) from e

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Types                                                                       #
# --------------------------------------------------------------------------- #


class LLMPlatform(str, Enum):
    LM_STUDIO = "lmstudio"
    OLLAMA = "ollama"
    LLAMA_CPP = "llamacpp"
    VLLM = "vllm"
    TGW = "tgw"  # text-generation-webui
    LOCALAI = "localai"
    KOBOLD = "kobold"  # KoboldCpp
    JAN = "jan"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DetectedEndpoint:
    """A reachable local LLM server."""

    platform: LLMPlatform
    base_url: str          # OpenAI-compatible base, e.g. http://localhost:1234/v1
    models: tuple[str, ...]
    raw_host: str          # e.g. http://localhost:1234 (no /v1 suffix)
    latency_ms: int        # how fast it responded to the probe
    notes: str = ""

    @property
    def alive(self) -> bool:
        return bool(self.models)


@dataclass(frozen=True)
class LLMConfig:
    """Resolved configuration ready to hand to an OpenAI-compatible client."""

    platform: LLMPlatform
    base_url: str
    model: str
    api_key: str
    endpoint: DetectedEndpoint | None = None
    candidates: tuple[DetectedEndpoint, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Model preference tables                                                     #
# --------------------------------------------------------------------------- #

# Ordered regex list — earlier patterns outrank later ones.
# Tuned for MAMGA's extraction workload: needs JSON fidelity, reasonable size,
# and modern instruction-tuning. Very small (<=3B) models are deprioritized.
EXTRACTION_PREFERENCES: tuple[tuple[str, int], ...] = (
    # name_regex, base_score (higher = better)
    (r"qwen[._-]?3\.?5[._-]?(?P<size>\d+)\s*b", 100),
    (r"qwen[._-]?3[._-]?(?P<size>\d+)\s*b", 95),
    (r"qwen[._-]?2\.5[._-]?(?P<size>\d+)\s*b", 90),
    (r"gemma[._-]?4[._-]?(?P<size>\d+)?\s*b?", 88),
    (r"gemma[._-]?3[._-]?(?P<size>\d+)?\s*b?", 80),
    (r"minimax", 85),
    (r"llama[._-]?3\.[123][._-]?(?P<size>\d+)\s*b", 82),
    (r"mistral[._-]?(nemo|small|medium)", 78),
    (r"mixtral[._-]?(?P<size>\d+x\d+)\s*b?", 76),
    (r"phi[._-]?[34][._-]?(?P<size>\d+)?\s*b?", 70),
    (r"deepseek[._-]?(v3|v2\.5|chat|coder)", 72),
    (r"yi[._-]?(?P<size>\d+)\s*b", 65),
    (r"command[._-]?r", 68),
)

# Models to actively avoid for extraction (embeddings, rerankers, vision-only).
EXTRACTION_BLOCKLIST: tuple[str, ...] = (
    r"embed",
    r"embedding",
    r"bge[._-]",
    r"rerank",
    r"whisper",
    r"clip",
    r"siglip",
    r"dall[._-]?e",
    r"stable[._-]?diffusion",
    r"sd[._-]?xl",
    r"flux",
    # Sentence-transformer encoders (MiniLM, MPNet, E5, GTE, Nomic, etc.)
    # — they don't match "embed" literally but are never chat-capable.
    r"mini\s*lm",
    r"mpnet",
    r"e5[._-](small|base|large|mistral)",
    r"gte[._-]",
    r"nomic[._-](embed|bert)",
    r"instructor[._-](base|large|xl)",
)


def _score_model_name(name: str) -> int:
    """
    Score a model id for MAMGA's extraction task. Higher is better.
    Returns -1 if the model is blocklisted (embeddings, etc.).
    """
    lowered = name.lower()

    for pattern in EXTRACTION_BLOCKLIST:
        if re.search(pattern, lowered):
            return -1

    best = 0
    for pattern, base in EXTRACTION_PREFERENCES:
        match = re.search(pattern, lowered)
        if not match:
            continue
        score = base
        # Size heuristic: prefer 7B–32B. Extract numeric group if present.
        try:
            raw_size = match.groupdict().get("size")
            if raw_size and raw_size.isdigit():
                size = int(raw_size)
                if 7 <= size <= 32:
                    score += 10
                elif size < 7:
                    score -= 5
                elif size > 70:
                    score -= 8  # too slow for per-turn extraction
        except (AttributeError, ValueError):
            pass
        best = max(best, score)

    return best


def pick_best_model(
    endpoint: DetectedEndpoint,
    *,
    task: str = "extraction",
    preferred: str | None = None,
) -> str | None:
    """
    Select a suitable model from the endpoint's model list.

    Args:
        endpoint: a reachable DetectedEndpoint.
        task: reserved for future task-specific preferences. Currently only
            "extraction" is tuned.
        preferred: if provided and present in endpoint.models, returned verbatim
            (supports both exact and case-insensitive substring match).

    Returns:
        The chosen model id, or None if no non-blocklisted model exists.
    """
    if not endpoint.models:
        return None

    if preferred:
        pref_l = preferred.lower()
        for m in endpoint.models:
            if m == preferred or m.lower() == pref_l:
                return m
        for m in endpoint.models:
            if pref_l in m.lower():
                return m

    if task != "extraction":
        logger.debug("Unknown task %r, falling back to extraction scoring", task)

    scored: list[tuple[int, int, str]] = []
    for idx, name in enumerate(endpoint.models):
        score = _score_model_name(name)
        if score < 0:
            continue
        # idx tiebreaker keeps listing order stable
        scored.append((score, -idx, name))

    if not scored:
        # Nothing scored positively — return first non-blocklisted model.
        for name in endpoint.models:
            if _score_model_name(name) >= 0:
                return name
        return None

    scored.sort(reverse=True)
    return scored[0][2]


# --------------------------------------------------------------------------- #
# Platform probes                                                             #
# --------------------------------------------------------------------------- #

# Candidate table: (platform, probe URL, derived base_url, raw_host).
# Probe URLs are OpenAI-compatible `/v1/models` unless the platform requires
# its native endpoint. Order matters for reporting — the first alive entry
# at a given priority wins.
_CANDIDATES: tuple[tuple[LLMPlatform, str, str, str], ...] = (
    (LLMPlatform.LM_STUDIO, "http://localhost:1234/v1/models", "http://localhost:1234/v1", "http://localhost:1234"),
    (LLMPlatform.OLLAMA,    "http://localhost:11434/v1/models", "http://localhost:11434/v1", "http://localhost:11434"),
    (LLMPlatform.LLAMA_CPP, "http://localhost:8080/v1/models", "http://localhost:8080/v1", "http://localhost:8080"),
    (LLMPlatform.VLLM,      "http://localhost:8000/v1/models", "http://localhost:8000/v1", "http://localhost:8000"),
    (LLMPlatform.TGW,       "http://localhost:5000/v1/models", "http://localhost:5000/v1", "http://localhost:5000"),
    (LLMPlatform.LOCALAI,   "http://localhost:8081/v1/models", "http://localhost:8081/v1", "http://localhost:8081"),
    (LLMPlatform.KOBOLD,    "http://localhost:5001/v1/models", "http://localhost:5001/v1", "http://localhost:5001"),
    (LLMPlatform.JAN,       "http://localhost:1337/v1/models", "http://localhost:1337/v1", "http://localhost:1337"),
)


def _fingerprint(data: dict, headers: dict) -> LLMPlatform:
    """
    Guess the platform from /v1/models JSON + HTTP headers.
    Used only to refine ambiguous port collisions.
    """
    server_hdr = headers.get("server", "").lower()
    if "ollama" in server_hdr:
        return LLMPlatform.OLLAMA
    if "llamacpp" in server_hdr or "llama.cpp" in server_hdr:
        return LLMPlatform.LLAMA_CPP
    if "uvicorn" in server_hdr and "vllm" in str(data).lower():
        return LLMPlatform.VLLM

    items = data.get("data") or data.get("models") or []
    if items and isinstance(items, list):
        first = items[0]
        if isinstance(first, dict):
            owned = str(first.get("owned_by", "")).lower()
            if owned == "organization_owner":
                return LLMPlatform.LM_STUDIO
            if owned == "library":
                return LLMPlatform.OLLAMA
            if owned == "vllm":
                return LLMPlatform.VLLM
            if owned == "localai":
                return LLMPlatform.LOCALAI
    return LLMPlatform.UNKNOWN


def _extract_models(data: dict) -> tuple[str, ...]:
    """Extract model ids from either OpenAI-compat or native response shapes."""
    # OpenAI-compat: {"data": [{"id": "..."}]}
    items = data.get("data")
    if isinstance(items, list):
        ids = tuple(str(x.get("id")) for x in items if isinstance(x, dict) and x.get("id"))
        if ids:
            return ids
    # Ollama native: {"models": [{"name": "..."}]}
    items = data.get("models")
    if isinstance(items, list):
        ids = tuple(
            str(x.get("name") or x.get("model") or "")
            for x in items
            if isinstance(x, dict)
        )
        return tuple(i for i in ids if i)
    return ()


def _probe_one(
    platform: LLMPlatform,
    probe_url: str,
    base_url: str,
    raw_host: str,
    timeout: float,
) -> DetectedEndpoint | None:
    """Probe a single candidate. Returns None on any failure."""
    import time

    start = time.monotonic()
    try:
        resp = requests.get(probe_url, timeout=timeout)
    except requests.RequestException as e:
        logger.debug("probe %s failed: %s", probe_url, e)
        return None

    latency_ms = int((time.monotonic() - start) * 1000)

    if resp.status_code != 200:
        logger.debug("probe %s returned %s", probe_url, resp.status_code)
        return None

    try:
        data = resp.json()
    except ValueError:
        logger.debug("probe %s returned non-JSON", probe_url)
        return None

    models = _extract_models(data)
    if not models:
        logger.debug("probe %s: no models listed", probe_url)
        return None

    # Refine platform if the response contradicts the candidate guess (port collisions).
    refined = _fingerprint(data, {k.lower(): v for k, v in resp.headers.items()})
    if refined != LLMPlatform.UNKNOWN and refined != platform:
        logger.info("port collision: %s responded on %s port", refined.value, platform.value)
        platform = refined

    return DetectedEndpoint(
        platform=platform,
        base_url=base_url,
        models=models,
        raw_host=raw_host,
        latency_ms=latency_ms,
        notes=f"{len(models)} model(s)",
    )


def detect_platforms(
    *,
    timeout: float = 1.5,
    extra_candidates: Iterable[tuple[LLMPlatform, str, str, str]] = (),
) -> list[DetectedEndpoint]:
    """
    Probe all known local LLM platforms concurrently. Respects the
    `LLM_BASE_URL` env var by prepending it as a high-priority candidate.

    Returns:
        List of reachable DetectedEndpoint objects, ordered by platform priority
        (LM Studio first, then Ollama, etc., mirroring _CANDIDATES order).
    """
    candidates: list[tuple[LLMPlatform, str, str, str]] = []

    override = os.environ.get("LLM_BASE_URL")
    if override:
        base = override.rstrip("/")
        host = base[:-3] if base.endswith("/v1") else base
        probe = base if base.endswith("/models") else f"{base}/models"
        platform = _platform_from_env() or LLMPlatform.UNKNOWN
        candidates.append((platform, probe, base, host))

    candidates.extend(_CANDIDATES)
    candidates.extend(extra_candidates)

    # Deduplicate by base_url while preserving order.
    seen: set[str] = set()
    deduped = []
    for c in candidates:
        if c[2] in seen:
            continue
        seen.add(c[2])
        deduped.append(c)

    results: list[tuple[int, DetectedEndpoint]] = []
    with ThreadPoolExecutor(max_workers=min(8, len(deduped))) as pool:
        futures = {
            pool.submit(_probe_one, *c, timeout): idx
            for idx, c in enumerate(deduped)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            ep = fut.result()
            if ep is not None:
                results.append((idx, ep))

    results.sort(key=lambda t: t[0])
    return [ep for _, ep in results]


def _platform_from_env() -> LLMPlatform | None:
    """Parse LLM_BACKEND env var into a platform enum, if set."""
    raw = os.environ.get("LLM_BACKEND", "").strip().lower()
    if not raw or raw == "auto":
        return None
    alias = {
        "lmstudio": LLMPlatform.LM_STUDIO,
        "lm_studio": LLMPlatform.LM_STUDIO,
        "lm-studio": LLMPlatform.LM_STUDIO,
        "ollama": LLMPlatform.OLLAMA,
        "llamacpp": LLMPlatform.LLAMA_CPP,
        "llama.cpp": LLMPlatform.LLAMA_CPP,
        "llama_cpp": LLMPlatform.LLAMA_CPP,
        "vllm": LLMPlatform.VLLM,
        "tgw": LLMPlatform.TGW,
        "textgen": LLMPlatform.TGW,
        "localai": LLMPlatform.LOCALAI,
        "kobold": LLMPlatform.KOBOLD,
        "koboldcpp": LLMPlatform.KOBOLD,
        "jan": LLMPlatform.JAN,
    }
    return alias.get(raw)


# --------------------------------------------------------------------------- #
# Public high-level API                                                       #
# --------------------------------------------------------------------------- #


def auto_configure(
    *,
    task: str = "extraction",
    preferred_model: str | None = None,
    preferred_platform: LLMPlatform | None = None,
    timeout: float = 1.5,
) -> LLMConfig:
    """
    One-shot configuration: detect platforms, pick the best one, pick a model.

    Resolution order:
        1. `LLM_BASE_URL` + `LLM_MODEL` env (full manual override — no probing).
        2. `LLM_BACKEND` env preference (prefer that platform if alive).
        3. `preferred_platform` argument.
        4. First reachable platform in _CANDIDATES order.

    Raises:
        RuntimeError if nothing is reachable.
    """
    # 1. Full manual override
    manual_url = os.environ.get("LLM_BASE_URL")
    manual_model = preferred_model or os.environ.get("LLM_MODEL")
    api_key = os.environ.get("OPENAI_API_KEY") or "local"

    # Probe regardless, so we can emit a helpful candidates list even under manual mode.
    endpoints = detect_platforms(timeout=timeout)

    if manual_url and manual_model:
        platform = _platform_from_env() or LLMPlatform.UNKNOWN
        # If the manual URL matches a probed endpoint, attach it for richer info.
        matching = next((e for e in endpoints if e.base_url.rstrip("/") == manual_url.rstrip("/")), None)
        if matching:
            platform = matching.platform
        logger.info("Using manual LLM_BASE_URL=%s model=%s", manual_url, manual_model)
        return LLMConfig(
            platform=platform,
            base_url=manual_url,
            model=manual_model,
            api_key=api_key,
            endpoint=matching,
            candidates=tuple(endpoints),
        )

    if not endpoints:
        raise RuntimeError(
            "No local LLM server detected. Start LM Studio, Ollama, llama.cpp, "
            "vLLM, text-generation-webui, LocalAI, KoboldCpp, or Jan — or set "
            "LLM_BASE_URL + LLM_MODEL env vars."
        )

    # 2/3. Apply platform preference. Explicit `preferred_platform` argument
    # wins over `LLM_BACKEND` env — CLI flags must override shell/.env config
    # per POSIX convention (env provides defaults, args override).
    pref_platform = preferred_platform or _platform_from_env()
    chosen: DetectedEndpoint | None = None
    if pref_platform:
        chosen = next((e for e in endpoints if e.platform == pref_platform), None)
        if chosen is None:
            logger.warning(
                "Preferred platform %s not reachable; falling back to %s",
                pref_platform.value,
                endpoints[0].platform.value,
            )

    # 4. First reachable.
    if chosen is None:
        chosen = endpoints[0]

    model = pick_best_model(chosen, task=task, preferred=manual_model)
    if model is None:
        raise RuntimeError(
            f"Platform {chosen.platform.value} is reachable at {chosen.base_url} "
            "but has no usable (non-embedding) models loaded."
        )

    logger.info(
        "Auto-configured: platform=%s base_url=%s model=%s (chosen from %d)",
        chosen.platform.value,
        chosen.base_url,
        model,
        len(chosen.models),
    )

    return LLMConfig(
        platform=chosen.platform,
        base_url=chosen.base_url,
        model=model,
        api_key=api_key,
        endpoint=chosen,
        candidates=tuple(endpoints),
    )


def describe(config: LLMConfig) -> str:
    """Pretty multi-line summary of a resolved LLMConfig for CLI output."""
    lines = [
        "=" * 62,
        "  MAMGA-Local :: LLM Auto-Detection",
        "=" * 62,
        f"  Platform   : {config.platform.value}",
        f"  Base URL   : {config.base_url}",
        f"  Model      : {config.model}",
        f"  API key    : {'<env>' if config.api_key not in ('local','lm-studio','') else config.api_key}",
    ]
    if config.endpoint is not None:
        lines.append(f"  Latency    : {config.endpoint.latency_ms} ms")
        lines.append(f"  Models     : {len(config.endpoint.models)} available")
    if config.candidates:
        lines.append("")
        lines.append("  Detected platforms:")
        for ep in config.candidates:
            marker = "*" if ep.base_url == config.base_url else " "
            lines.append(
                f"   {marker} {ep.platform.value:<9} {ep.base_url:<34} "
                f"({len(ep.models)} models, {ep.latency_ms} ms)"
            )
    lines.append("=" * 62)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI entry point                                                             #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Detect local LLM platforms for MAMGA-Local")
    parser.add_argument("--timeout", type=float, default=1.5, help="per-probe timeout (seconds)")
    parser.add_argument("--task", default="extraction", help="task tag for model scoring")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    try:
        cfg = auto_configure(task=args.task, timeout=args.timeout)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        import json
        payload = {
            "platform": cfg.platform.value,
            "base_url": cfg.base_url,
            "model": cfg.model,
            "api_key_source": "env" if os.environ.get("OPENAI_API_KEY") else "default",
            "candidates": [
                {
                    "platform": ep.platform.value,
                    "base_url": ep.base_url,
                    "models": list(ep.models),
                    "latency_ms": ep.latency_ms,
                }
                for ep in cfg.candidates
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(describe(cfg))
