"""Command-line flags shared by the search runners for the LLM operator."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cache import ResponseCache
from .client import BACKENDS, make_client
from .operator import LLMProposalOperator


def add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("LLM proposal operator")
    group.add_argument(
        "--llm-rate", type=float, default=0.0,
        help="Share of evolution children proposed by the LLM (default 0: off).",
    )
    group.add_argument("--llm-backend", choices=("none", *BACKENDS), default="none")
    group.add_argument("--llm-model", help="Model name; each backend has a default.")
    group.add_argument("--llm-base-url", help="Endpoint for --llm-backend openai-compatible.")
    group.add_argument(
        "--llm-cache", type=Path, default=Path("data/processed/llm/cache.jsonl"),
        help="Answers are reused from here, so reruns of a seed are free.",
    )
    group.add_argument(
        "--llm-failures", type=Path, default=Path("data/processed/llm/failures.jsonl"),
        help="Rejected proposals are appended here with the reason.",
    )


def build_llm_operator(
    args: argparse.Namespace, max_depth: int, horizon: int
) -> LLMProposalOperator | None:
    """Return the operator, or None when ``--llm-rate`` is 0."""

    if args.llm_rate <= 0:
        return None
    if args.llm_backend == "none":
        raise SystemExit("--llm-rate > 0 needs --llm-backend")
    client = make_client(args.llm_backend, args.llm_model, args.llm_base_url)
    return LLMProposalOperator(
        client,
        cache=ResponseCache(args.llm_cache),
        max_depth=max_depth,
        horizon=horizon,
        failures_path=args.llm_failures,
    )


def llm_config_fields(args: argparse.Namespace, operator: LLMProposalOperator | None) -> dict:
    """EvolutionConfig keyword arguments describing the LLM settings."""

    if operator is None:
        return {}
    return {
        "llm_rate": args.llm_rate,
        "llm_backend": args.llm_backend,
        "llm_model": operator.client.model,
    }
