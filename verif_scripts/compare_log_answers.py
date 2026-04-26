#!/usr/bin/env python3
"""
Compare answers stored in two TokenSmith chat logs.

Reports:
- semantic similarity between the two answers
- BLEU score between the two answers
- plain text similarity (difflib ratio)
- keyword coverage for each answer if keywords are provided explicitly or
  found from tests/benchmarks.yaml by matching the question text
- optional benchmark-aware scoring for each log answer against the expected
  answer in tests/benchmarks.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import yaml
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from tests.metrics.keyword_match import KeywordMatchMetric
from tests.metrics.semantic import SemanticSimilarityMetric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare answers from two TokenSmith log files.")
    parser.add_argument("log_a", help="Path to the first log file")
    parser.add_argument("log_b", help="Path to the second log file")
    parser.add_argument(
        "--benchmarks-file",
        default="tests/benchmarks.yaml",
        help="Benchmark YAML used to auto-load keywords by matching question text",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Optional explicit keywords to use for keyword matching",
    )
    parser.add_argument(
        "--benchmark-id",
        default=None,
        help="Optional benchmark id to use instead of matching by question text",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def load_benchmarks(benchmarks_file: str | Path) -> list[dict[str, Any]]:
    benchmark_path = Path(benchmarks_file)
    if not benchmark_path.exists():
        return []

    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("benchmarks", []))


def find_benchmark(
    benchmarks: list[dict[str, Any]],
    question: str,
    benchmark_id: str | None = None,
) -> dict[str, Any] | None:
    if benchmark_id:
        for benchmark in benchmarks:
            if benchmark.get("id") == benchmark_id:
                return benchmark
        return None

    for benchmark in benchmarks:
        if (benchmark.get("question") or "").strip() == question:
            return benchmark
    return None


def bleu_similarity(answer_a: str, answer_b: str) -> float:
    reference = answer_a.split()
    hypothesis = answer_b.split()
    if not reference or not hypothesis:
        return 0.0
    return float(
        sentence_bleu(
            [reference],
            hypothesis,
            smoothing_function=SmoothingFunction().method1,
        )
    )


def text_similarity(answer_a: str, answer_b: str) -> float:
    return float(SequenceMatcher(None, answer_a, answer_b).ratio())


def print_keywords(keywords: Iterable[str]) -> str:
    values = list(keywords)
    return ", ".join(values) if values else "(none)"


def score_against_expected(
    answer: str,
    expected_answer: str,
    keywords: list[str],
    semantic_metric: SemanticSimilarityMetric,
    keyword_metric: KeywordMatchMetric,
) -> dict[str, float]:
    semantic_score = semantic_metric.calculate(answer, expected_answer) if semantic_metric.is_available() else 0.0
    bleu_score = bleu_similarity(expected_answer, answer)
    text_score = text_similarity(expected_answer, answer)
    keyword_score = keyword_metric.calculate(answer, "", keywords)
    return {
        "semantic_similarity": semantic_score,
        "bleu_similarity": bleu_score,
        "text_similarity": text_score,
        "keyword_score": keyword_score,
    }


def main() -> None:
    args = parse_args()

    log_a = load_json(args.log_a)
    log_b = load_json(args.log_b)

    question_a = extract_question(log_a)
    question_b = extract_question(log_b)
    answer_a = extract_answer(log_a)
    answer_b = extract_answer(log_b)

    if question_a != question_b:
        print("Warning: questions differ between the two logs.")
        print(f"  A: {question_a}")
        print(f"  B: {question_b}")

    question = question_a or question_b
    benchmarks = load_benchmarks(args.benchmarks_file)
    matched_benchmark = find_benchmark(benchmarks, question, args.benchmark_id)
    benchmark_keywords = list(matched_benchmark.get("keywords") or []) if matched_benchmark else []
    expected_answer = (matched_benchmark.get("expected_answer") or "").strip() if matched_benchmark else ""
    benchmark_label = matched_benchmark.get("id") if matched_benchmark else None

    keywords = list(args.keywords) if args.keywords else benchmark_keywords

    semantic_metric = SemanticSimilarityMetric()
    keyword_metric = KeywordMatchMetric()

    semantic_score = semantic_metric.calculate(answer_a, answer_b) if semantic_metric.is_available() else 0.0
    bleu_score = bleu_similarity(answer_a, answer_b)
    text_score = text_similarity(answer_a, answer_b)
    keyword_score_a = keyword_metric.calculate(answer_a, "", keywords)
    keyword_score_b = keyword_metric.calculate(answer_b, "", keywords)

    print("Question:")
    print(question or "(missing)")
    print()

    if matched_benchmark:
        print("Matched Benchmark:")
        print(f"  id:                 {benchmark_label}")
        print(f"  threshold:          {matched_benchmark.get('similarity_threshold')}")
        print()
    elif args.benchmark_id:
        print(f"Warning: benchmark id '{args.benchmark_id}' was not found in {args.benchmarks_file}")
        print()

    print("Logs:")
    print(f"  A: {args.log_a}")
    print(f"  B: {args.log_b}")
    print()

    print("Answer-To-Answer Metrics:")
    print(f"  semantic_similarity: {semantic_score:.4f}")
    print(f"  bleu_similarity:     {bleu_score:.4f}")
    print(f"  text_similarity:     {text_score:.4f}")
    print()

    print("Keyword Coverage:")
    print(f"  keywords:            {print_keywords(keywords)}")
    print(f"  log_a_keyword_score: {keyword_score_a:.4f}")
    print(f"  log_b_keyword_score: {keyword_score_b:.4f}")
    print()

    if expected_answer:
        scores_a = score_against_expected(answer_a, expected_answer, keywords, semantic_metric, keyword_metric)
        scores_b = score_against_expected(answer_b, expected_answer, keywords, semantic_metric, keyword_metric)

        print("Benchmark-Referenced Scores:")
        print("  Against expected answer from benchmarks.yaml")
        print(f"  log_a_semantic:     {scores_a['semantic_similarity']:.4f}")
        print(f"  log_a_bleu:         {scores_a['bleu_similarity']:.4f}")
        print(f"  log_a_text:         {scores_a['text_similarity']:.4f}")
        print(f"  log_a_keyword:      {scores_a['keyword_score']:.4f}")
        print(f"  log_b_semantic:     {scores_b['semantic_similarity']:.4f}")
        print(f"  log_b_bleu:         {scores_b['bleu_similarity']:.4f}")
        print(f"  log_b_text:         {scores_b['text_similarity']:.4f}")
        print(f"  log_b_keyword:      {scores_b['keyword_score']:.4f}")
        print()

        print("Expected Answer Preview:")
        print(expected_answer[:600].replace("\n", " "))
        print()

    print("Answer A Preview:")
    print(answer_a[:600].replace("\n", " "))
    print()
    print("Answer B Preview:")
    print(answer_b[:600].replace("\n", " "))


if __name__ == "__main__":
    main()
