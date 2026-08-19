#!/usr/bin/env python3
"""Validate and summarize an msModelSpec prepared Arrow dataset."""

from __future__ import annotations

import argparse
import statistics

from datasets import load_from_disk


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[min(len(values) - 1, round((len(values) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    args = parser.parse_args()

    dataset = load_from_disk(args.prepared_data)
    if len(dataset) == 0:
        raise RuntimeError("Prepared dataset is empty")

    lengths = []
    assistant_tokens = []
    for index, row in enumerate(dataset):
        token_ids = row["input_ids"]
        loss_mask = row["loss_mask"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if hasattr(loss_mask, "tolist"):
            loss_mask = loss_mask.tolist()
        if len(token_ids) != len(loss_mask):
            raise ValueError(f"sample {index}: input_ids/loss_mask length mismatch")
        if len(token_ids) > args.seq_length:
            raise ValueError(f"sample {index}: length {len(token_ids)} > {args.seq_length}")
        lengths.append(len(token_ids))
        assistant_tokens.append(int(sum(loss_mask)))

    print(f"samples={len(dataset)} columns={dataset.column_names}")
    print(
        "seq_len "
        f"min={min(lengths)} p50={percentile(lengths, 0.50)} "
        f"p95={percentile(lengths, 0.95)} max={max(lengths)} "
        f"mean={statistics.mean(lengths):.1f}"
    )
    print(
        "assistant_tokens "
        f"total={sum(assistant_tokens)} min={min(assistant_tokens)} "
        f"p50={percentile(assistant_tokens, 0.50)} "
        f"max={max(assistant_tokens)}"
    )
    print(f"at_seq_limit={sum(length == args.seq_length for length in lengths)}")


if __name__ == "__main__":
    main()

