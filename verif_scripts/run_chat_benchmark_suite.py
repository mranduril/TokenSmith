#!/usr/bin/env python3
"""
Run benchmark questions through the interactive chat pipeline and score each
generated log against its benchmark entry.

By default this mirrors `make run-chat-exp`:
  conda run --no-capture-output -n tokensmith python -m src.main chat --experimental_chunking

For each benchmark:
  1. feeds the question into chat
  2. exits chat
  3. finds the newly generated log file
  4. scores the answer in that log against benchmarks.yaml
  5. writes a scored JSON file to the output directory using the same basename
     as the generated log
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
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


@dataclass
class BenchmarkEntry:
    benchmark_id: str
    question: str
    expected_answer: str
    keywords: list[str]
    threshold: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark questions through TokenSmith chat and score the logs.")
    parser.add_argument(
        "--benchmarks-file",
        default="tests/benchmarks.yaml",
        help="Path to benchmarks.yaml relative to the repo root",
    )
    parser.add_argument(
        "--benchmark-ids",
        default=None,
        help="Comma-separated benchmark ids to run; defaults to all",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/results/chat_benchmark_runs",
        help="Directory where scored outputs are written",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory where chat logs are written",
    )
    parser.add_argument(
        "--conda-env",
        default="tokensmith",
        help="Conda environment used to run chat",
    )
    parser.add_argument(
        "--chat-mode",
        choices=["experimental", "standard"],
        default="experimental",
        help="Whether to mirror run-chat-exp or run-chat",
    )
    parser.add_argument(
        "--index-prefix",
        default=None,
        help="Optional --index_prefix override passed to chat",
    )
    return parser.parse_args()


def load_benchmarks(path: Path, selected_ids: set[str] | None) -> list[BenchmarkEntry]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    entries: list[BenchmarkEntry] = []
    for benchmark in data.get("benchmarks", []):
        benchmark_id = benchmark.get("id")
        if not benchmark_id:
            continue
        if selected_ids and benchmark_id not in selected_ids:
            continue
        entries.append(
            BenchmarkEntry(
                benchmark_id=benchmark_id,
                question=(benchmark.get("question") or "").strip(),
                expected_answer=(benchmark.get("expected_answer") or "").strip(),
                keywords=list(benchmark.get("keywords") or []),
                threshold=benchmark.get("similarity_threshold"),
            )
        )
    return entries


def extract_answer(log_data: dict[str, Any]) -> str:
    for key in ("full_response", "answer", "retrieved_answer"):
        value = log_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("Could not find an answer field in the generated log")


def extract_question(log_data: dict[str, Any]) -> str:
    return (log_data.get("query") or log_data.get("question") or "").strip()


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


def find_newest_log(log_dir: Path, previous_logs: set[Path]) -> Path:
    current_logs = {path.resolve() for path in log_dir.glob("chat_*.json")}
    new_logs = sorted(current_logs - previous_logs, key=lambda p: p.stat().st_mtime, reverse=True)
    if new_logs:
        return new_logs[0]

    fallback = sorted(current_logs, key=lambda p: p.stat().st_mtime, reverse=True)
    if fallback:
        return fallback[0]
    raise FileNotFoundError(f"No chat logs found in {log_dir}")


def run_chat_question(
    benchmark: BenchmarkEntry,
    log_dir: Path,
    conda_env: str,
    chat_mode: str,
    index_prefix: str | None,
) -> tuple[Path, str]:
    previous_logs = {path.resolve() for path in log_dir.glob("chat_*.json")}

    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        "-m",
        "src.main",
        "chat",
    ]
    if chat_mode == "experimental":
        cmd.append("--experimental_chunking")
    if index_prefix:
        cmd.extend(["--index_prefix", index_prefix])

    question_input = f"{benchmark.question}\nexit\n"
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        input=question_input,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Chat command failed for '{benchmark.benchmark_id}' with code {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    newest_log = find_newest_log(log_dir, previous_logs)
    combined_output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    return newest_log, combined_output.strip()


def score_answer(
    answer: str,
    benchmark: BenchmarkEntry,
    semantic_metric: SemanticSimilarityMetric,
    keyword_metric: KeywordMatchMetric,
) -> dict[str, float]:
    semantic_score = semantic_metric.calculate(answer, benchmark.expected_answer) if semantic_metric.is_available() else 0.0
    bleu_score = bleu_similarity(benchmark.expected_answer, answer)
    text_score = text_similarity(benchmark.expected_answer, answer)
    keyword_score = keyword_metric.calculate(answer, "", benchmark.keywords)
    return {
        "semantic_similarity": semantic_score,
        "bleu_similarity": bleu_score,
        "text_similarity": text_score,
        "keyword_score": keyword_score,
    }


def write_result(
    output_dir: Path,
    log_path: Path,
    payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / log_path.name, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    benchmarks_file = (PROJECT_ROOT / args.benchmarks_file).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    log_dir = (PROJECT_ROOT / args.log_dir).resolve()
    selected_ids = (
        {item.strip() for item in args.benchmark_ids.split(",") if item.strip()}
        if args.benchmark_ids
        else None
    )

    benchmarks = load_benchmarks(benchmarks_file, selected_ids)
    if not benchmarks:
        raise ValueError("No benchmarks selected to run")

    semantic_metric = SemanticSimilarityMetric()
    keyword_metric = KeywordMatchMetric()

    summary: list[dict[str, Any]] = []
    copied_logs_dir = output_dir / "raw_logs"
    copied_logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(benchmarks)} benchmark(s) in {args.chat_mode} chat mode...")
    for benchmark in benchmarks:
        print(f"\n--- {benchmark.benchmark_id}: {benchmark.question}")
        log_path, chat_output = run_chat_question(
            benchmark=benchmark,
            log_dir=log_dir,
            conda_env=args.conda_env,
            chat_mode=args.chat_mode,
            index_prefix=args.index_prefix,
        )

        with open(log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        answer = extract_answer(log_data)
        log_question = extract_question(log_data)
        scores = score_answer(answer, benchmark, semantic_metric, keyword_metric)
        passed = (
            scores["semantic_similarity"] >= benchmark.threshold
            if benchmark.threshold is not None
            else None
        )

        result_payload = {
            "timestamp": datetime.now().isoformat(),
            "benchmark": {
                "id": benchmark.benchmark_id,
                "question": benchmark.question,
                "expected_answer": benchmark.expected_answer,
                "keywords": benchmark.keywords,
                "threshold": benchmark.threshold,
            },
            "source_log": str(log_path),
            "log_question": log_question,
            "scores": scores,
            "passed_semantic_threshold": passed,
            "answer_preview": answer[:1200],
            "chat_output_preview": chat_output[:2000],
        }
        write_result(output_dir, log_path, result_payload)
        shutil.copy2(log_path, copied_logs_dir / log_path.name)

        summary.append(
            {
                "benchmark_id": benchmark.benchmark_id,
                "log_file": log_path.name,
                "scores": scores,
                "passed_semantic_threshold": passed,
            }
        )
        print(
            f"Saved {log_path.name} -> semantic={scores['semantic_similarity']:.4f}, "
            f"bleu={scores['bleu_similarity']:.4f}, text={scores['text_similarity']:.4f}, "
            f"keyword={scores['keyword_score']:.4f}"
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nWrote scored outputs to {output_dir}")


if __name__ == "__main__":
    main()
