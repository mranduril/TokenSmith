#!/usr/bin/env python3
"""
Score a single TokenSmith chat log against one benchmark entry.

Usage:
  python scripts/score_log_against_benchmark.py logs/chat_x.json --benchmark-id lossy_decomposition
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from tests.metrics.keyword_match import KeywordMatchMetric
from tests.metrics.semantic import SemanticSimilarityMetric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one TokenSmith log against a benchmark entry.")
    parser.add_argument("log_path", help="Path to the chat log JSON file")
    parser.add_argument("--benchmark-id", required=True, help="Benchmark id from tests/benchmarks.yaml")
    parser.add_argument(
        "--benchmarks-file",
        default="tests/benchmarks.yaml",
        help="Path to benchmarks.yaml",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_benchmark(benchmarks_file: str | Path, benchmark_id: str) -> dict[str, Any]:
    with open(benchmarks_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    for benchmark in data.get("benchmarks", []):
        if benchmark.get("id") == benchmark_id:
            return benchmark
    raise ValueError(f"Benchmark id '{benchmark_id}' not found in {benchmarks_file}")


def extract_question(data: dict[str, Any]) -> str:
    return (data.get("query") or data.get("question") or "").strip()


def extract_answer(data: dict[str, Any]) -> str:
    candidates = [
        data.get("full_response"),
        data.get("answer"),
        data.get("retrieved_answer"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise ValueError("Could not find an answer field in the log")


def bleu_similarity(expected: str, answer: str) -> float:
    reference = expected.split()
    hypothesis = answer.split()
    if not reference or not hypothesis:
        return 0.0
    return float(
        sentence_bleu(
            [reference],
            hypothesis,
            smoothing_function=SmoothingFunction().method1,
        )
    )


def text_similarity(expected: str, answer: str) -> float:
    return float(SequenceMatcher(None, expected, answer).ratio())


def main() -> None:
    args = parse_args()

    log_data = load_json(args.log_path)
    benchmark = load_benchmark(args.benchmarks_file, args.benchmark_id)

    question = extract_question(log_data)
    answer = extract_answer(log_data)
    expected = (benchmark.get("expected_answer") or "").strip()
    keywords = list(benchmark.get("keywords") or [])
    threshold = benchmark.get("similarity_threshold")

    semantic_metric = SemanticSimilarityMetric()
    keyword_metric = KeywordMatchMetric()

    semantic_score = semantic_metric.calculate(answer, expected) if semantic_metric.is_available() else 0.0
    bleu_score = bleu_similarity(expected, answer)
    text_score = text_similarity(expected, answer)
    keyword_score = keyword_metric.calculate(answer, "", keywords)

    print("Log:")
    print(f"  path:               {args.log_path}")
    print(f"  question:           {question or '(missing)'}")
    print()

    print("Benchmark:")
    print(f"  id:                 {benchmark.get('id')}")
    print(f"  question:           {benchmark.get('question')}")
    print(f"  threshold:          {threshold}")
    print(f"  keywords:           {', '.join(keywords) if keywords else '(none)'}")
    print()

    if question and benchmark.get("question") and question != benchmark.get("question"):
        print("Warning:")
        print("  log question does not exactly match the benchmark question")
        print()

    print("Scores:")
    print(f"  semantic_similarity:{semantic_score:.4f}")
    print(f"  bleu_similarity:    {bleu_score:.4f}")
    print(f"  text_similarity:    {text_score:.4f}")
    print(f"  keyword_score:      {keyword_score:.4f}")
    print()

    print("Expected Answer Preview:")
    print(expected[:800].replace("\n", " "))
    print()
    print("Log Answer Preview:")
    print(answer[:800].replace("\n", " "))


if __name__ == "__main__":
    main()
