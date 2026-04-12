# Changelog

All notable changes to MAMGA-Local are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- **LLM auto-detection extracted to standalone package.** `utils/llm_detector.py`
  and `tests/test_llm_detector.py` have been moved to a separate, reusable
  project: [boboidvtw/local-llm-detector](https://github.com/boboidvtw/local-llm-detector)
  (MIT, published as `local-llm-detector` on PyPI — pending).
  MAMGA-Local now consumes it as a pinned git dependency
  (`local-llm-detector @ git+https://github.com/boboidvtw/local-llm-detector.git@v0.1.0`).
  `main.py` imports were updated from `utils.llm_detector` to `llm_detector`.
  Runtime behaviour is unchanged — `python main.py detect` produces identical output.

### Removed
- `utils/llm_detector.py` (moved upstream to `local-llm-detector` v0.1.0).
- `tests/test_llm_detector.py` (moved upstream to `local-llm-detector` v0.1.0).

## [Phase 1-3] - 2026-04

### Added
- Temporal Resonance Graph (TRG) memory layer with 4 orthogonal graphs
  (temporal, semantic, causal, entity).
- LoCoMo dataset loader and benchmark harness.
- Memory builder, query engine, and evaluator.
- Local LLM auto-detection for LM Studio, Ollama, llama.cpp, vLLM,
  text-generation-webui, LocalAI, KoboldCpp, and Jan (later extracted).
- `main.py` CLI with `detect`, `build`, `query`, `test` subcommands.
