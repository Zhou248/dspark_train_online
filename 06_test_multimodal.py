#!/usr/bin/env python3
"""Send one ALLaVA prepared sample to the trained DSpark endpoint."""

from __future__ import annotations

import argparse

import openai
from datasets import load_from_disk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    dataset = load_from_disk(args.prepared_data)
    row = dataset[args.index]
    messages = row.get("messages")
    if not messages:
        raise ValueError(f"Prepared sample {args.index} has no multimodal messages")

    client = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY")
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=0,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()

