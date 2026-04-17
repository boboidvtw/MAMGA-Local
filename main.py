#!/usr/bin/env python3
"""
MAMGA-Local :: main CLI

Rewritten 2026-04-11 to:
  1. Use MemoryBuilder / QueryEngine with the correct (current) signatures.
  2. Auto-detect a running local LLM platform (LM Studio / Ollama / llama.cpp /
     vLLM / TGW / LocalAI / KoboldCpp / Jan) via the `local-llm-detector` package
     (https://github.com/boboidvtw/local-llm-detector).
  3. Work out of the box with a LoCoMo-format dataset (data/locomo10.json).

Modes
-----
    detect   - probe local LLM servers and print what was found (no build).
    build    - build TRG memory from a LoCoMo JSON and persist to cache.
    query    - load cached memory and answer a single question.
    test     - run the benchmark harness on one LoCoMo sample.

Env precedence
--------------
    LLM_BASE_URL + LLM_MODEL   --> used verbatim, no probing
    LLM_BACKEND=<name>         --> prefer that platform if reachable
    (nothing)                  --> probe all known ports, pick best model
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Silence noisy deps before heavy imports.
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Project-internal imports (deferred heavy ones are inside command handlers).
from llm_detector import LLMConfig, LLMPlatform, auto_configure, describe

logger = logging.getLogger("mamga.main")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_llm(args: argparse.Namespace) -> LLMConfig:
    """
    Resolve the LLM configuration and push it into env vars so downstream
    code (which reads os.environ) picks it up uniformly.
    """
    preferred_platform: LLMPlatform | None = None
    if args.backend and args.backend != "auto":
        try:
            preferred_platform = LLMPlatform(args.backend)
        except ValueError:
            logger.warning("Unknown --backend %r, falling back to auto", args.backend)

    cfg = auto_configure(
        task="extraction",
        preferred_model=args.model if args.model != "auto" else None,
        preferred_platform=preferred_platform,
        timeout=args.probe_timeout,
    )

    # Export for downstream libs (OpenAIController, etc.) that read env directly.
    os.environ["LLM_BASE_URL"] = cfg.base_url
    os.environ["LLM_MODEL"] = cfg.model
    os.environ["OPENAI_API_KEY"] = cfg.api_key  # required by the openai SDK
    # Some callers still look at OPENAI_BASE_URL; set it too for compatibility.
    os.environ["OPENAI_BASE_URL"] = cfg.base_url

    return cfg


def _load_locomo_samples(path: Path):
    from load_dataset import load_locomo_dataset

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Expecting a LoCoMo-format JSON (see data/locomo10.json)."
        )
    return load_locomo_dataset(path)


def _truncate_sample(sample, max_turns: int | None):
    """Cap total turns across sessions for quick smoke runs."""
    if not max_turns or max_turns <= 0:
        return sample

    remaining = max_turns
    kept: dict = {}
    for sid in sorted(sample.conversation.sessions.keys()):
        if remaining <= 0:
            break
        session = sample.conversation.sessions[sid]
        turns = list(session.turns)[:remaining]
        session.turns = turns
        kept[sid] = session
        remaining -= len(turns)
    sample.conversation.sessions = kept
    return sample


# --------------------------------------------------------------------------- #
# Command handlers                                                            #
# --------------------------------------------------------------------------- #


def cmd_detect(args: argparse.Namespace) -> int:
    """Probe and print. Never imports heavy ML deps."""
    try:
        cfg = _resolve_llm(args)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(describe(cfg))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    cfg = _resolve_llm(args)
    print(describe(cfg))

    from memory.memory_builder import MemoryBuilder

    samples = _load_locomo_samples(Path(args.input))
    if args.sample >= len(samples):
        print(f"ERROR: --sample {args.sample} out of range (have {len(samples)})", file=sys.stderr)
        return 2

    sample = _truncate_sample(samples[args.sample], args.max_turns)

    cache_dir = Path(args.cache_dir) / f"sample{args.sample}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    builder = MemoryBuilder(
        cache_dir=str(cache_dir),
        llm_model=cfg.model,
        use_episodes=args.use_episodes,
        embedding_model=args.embedding_model,
    )

    logger.info("Building memory (cache_dir=%s)", cache_dir)
    stats = builder.build_memory(sample)
    logger.info("Build stats: %s", stats)
    print(f"\nMemory built. Cache: {cache_dir}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    if not args.question:
        print("ERROR: --question is required for query mode", file=sys.stderr)
        return 2

    cfg = _resolve_llm(args)
    print(describe(cfg))

    from memory.memory_builder import MemoryBuilder
    from memory.query_engine import QueryEngine

    samples = _load_locomo_samples(Path(args.input))
    sample = _truncate_sample(samples[args.sample], args.max_turns)

    cache_dir = Path(args.cache_dir) / f"sample{args.sample}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    builder = MemoryBuilder(
        cache_dir=str(cache_dir),
        llm_model=cfg.model,
        use_episodes=args.use_episodes,
        embedding_model=args.embedding_model,
    )

    # Try to reuse an existing cache before rebuilding from scratch.
    loaded = False
    graph_file = cache_dir / "graph.json"
    if graph_file.exists() and not args.rebuild:
        try:
            builder.trg.load_from_file(str(graph_file))
            loaded = True
            logger.info("Loaded existing memory from %s", graph_file)
        except Exception as e:
            logger.warning("Failed to load cache (%s); rebuilding", e)

    if not loaded:
        builder.build_memory(sample)

    qe = QueryEngine(
        trg_memory=builder.trg,
        node_index=getattr(builder, "node_index", {}),
        entity_session_map=getattr(builder, "entity_session_map", None),
        entity_dia_map=getattr(builder, "entity_dia_map", None),
        llm_controller=builder.llm_controller,
    )

    context, answer = qe.query(args.question, top_k=args.top_k)
    print(f"\nQ: {args.question}")
    print(f"A: {answer}")
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """
    Thin wrapper over the existing test_fixed_memory.py harness so everything
    runs through the auto-detected LLM config.
    """
    cfg = _resolve_llm(args)
    print(describe(cfg))

    import test_fixed_memory  # noqa: WPS433 (intentional late import)

    # Synthesize the argparse.Namespace expected by test_fixed_memory.main().
    forwarded = argparse.Namespace(
        dataset=args.input,
        sample=[args.sample],
        max_questions=args.max_questions,
        cache_dir=str(Path(args.cache_dir) / "harness"),
        rebuild=args.rebuild,
        model=cfg.model,
        embedding_model=args.embedding_model,
        use_episodes=args.use_episodes,
        score_only=False,
        input_results=None,
        skip_category_5=False,
        category_to_test=args.categories,
        no_parallel=True,
        n_workers=1,
        best_of_n=1,
        best_of_n_method="llm_judge",
        ablation=None,
    )

    # test_fixed_memory.main() uses sys.argv -> override before calling.
    orig_argv = sys.argv[:]
    sys.argv = ["test_fixed_memory.py"]
    try:
        return test_fixed_memory.main() or 0
    finally:
        sys.argv = orig_argv


# --------------------------------------------------------------------------- #
# Argument parser                                                             #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mamga",
        description="MAMGA-Local CLI with auto-detected local LLM backends.",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")

    # LLM / platform options shared by all commands.
    llm = parser.add_argument_group("llm")
    llm.add_argument(
        "--backend",
        default=os.environ.get("LLM_BACKEND", "auto"),
        choices=[p.value for p in LLMPlatform] + ["auto"],
        help="preferred platform (default: auto-detect)",
    )
    llm.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL", "auto"),
        help='model id, or "auto" to let the detector pick (default: auto)',
    )
    llm.add_argument(
        "--probe-timeout",
        type=float,
        default=1.5,
        help="per-endpoint probe timeout in seconds (default: 1.5)",
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # ---- detect ---------------------------------------------------------- #
    sub.add_parser("detect", help="probe local LLM servers and exit")

    # ---- shared dataset args --------------------------------------------- #
    def _add_dataset_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--input", default="data/locomo10.json", help="LoCoMo JSON path")
        p.add_argument("--sample", type=int, default=0, help="sample index (default 0)")
        p.add_argument("--cache-dir", default="./mamga_cache", help="cache directory")
        p.add_argument("--embedding-model", default="minilm", choices=["minilm", "openai"])
        p.add_argument("--use-episodes", action="store_true")
        p.add_argument(
            "--max-turns",
            type=int,
            default=0,
            help="truncate conversation to N turns (0 = full sample)",
        )

    p_build = sub.add_parser("build", help="build TRG memory from a LoCoMo sample")
    _add_dataset_args(p_build)

    p_query = sub.add_parser("query", help="answer a question against a cached memory")
    _add_dataset_args(p_query)
    p_query.add_argument("--question", required=True)
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--rebuild", action="store_true", help="force rebuild before querying")

    p_test = sub.add_parser("test", help="run the benchmark harness on one sample")
    _add_dataset_args(p_test)
    p_test.add_argument("--max-questions", type=int, default=10)
    p_test.add_argument("--categories", default="4", help="comma-separated category ids")
    p_test.add_argument("--rebuild", action="store_true")

    return parser


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    handlers = {
        "detect": cmd_detect,
        "build": cmd_build,
        "query": cmd_query,
        "test": cmd_test,
    }
    return handlers[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
