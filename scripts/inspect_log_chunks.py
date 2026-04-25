#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the actual retrieved chunks referenced by a TokenSmith chat log."
    )
    parser.add_argument("log_path", help="Path to a chat_*.json log file")
    parser.add_argument(
        "--artifacts-dir",
        default="index/sections",
        help="Directory containing chunk/meta artifacts",
    )
    parser.add_argument(
        "--index-prefix",
        default="textbook_index",
        help="Artifact prefix, e.g. textbook_index or exp_textbook_index",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1200,
        help="Number of characters of each chunk to print",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_path = Path(args.log_path)
    artifacts_dir = Path(args.artifacts_dir)

    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    top_idxs = log.get("top_idxs") or log.get("retrieval_top_idxs")
    if not top_idxs:
        raise ValueError("Log does not contain top_idxs or retrieval_top_idxs")

    with open(artifacts_dir / f"{args.index_prefix}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open(artifacts_dir / f"{args.index_prefix}_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    print(f"Log: {log_path}")
    print(f"Query: {log.get('query', '')}")
    print(f"Index prefix: {args.index_prefix}")
    print(f"Retrieved IDs: {top_idxs}")

    for rank, idx in enumerate(top_idxs, 1):
        idx = int(idx)
        print(f"\n=== rank {rank} | idx {idx} ===")
        if 0 <= idx < len(meta):
            item = meta[idx]
            print("pages:", item.get("page_numbers"))
            print("section:", item.get("section"))
            print("section_path:", item.get("section_path"))
        else:
            print("warning: idx out of bounds for metadata")

        if 0 <= idx < len(chunks):
            print(chunks[idx][: args.preview_chars])
        else:
            print("warning: idx out of bounds for chunks")


if __name__ == "__main__":
    main()
