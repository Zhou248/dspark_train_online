#!/usr/bin/env python3
"""Extract DSpark loss metrics from a tee log and write CSV/PNG outputs."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PAIR_RE = re.compile(
    r"['\"](?P<key>[^'\"]*loss[^'\"]*)['\"]\s*:\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
STEP_RE = re.compile(
    r"['\"]global_step['\"]\s*:\s*(?P<step>\d+)", re.IGNORECASE
)
EPOCH_RE = re.compile(r"['\"]epoch['\"]\s*:\s*(?P<epoch>\d+)")


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            output.update(flatten(child, name))
    else:
        output[prefix] = value
    return output


def mapping_fragments(text: str):
    """Yield probable one-line or Rich-wrapped metric dictionaries."""
    pending = ""
    depth = 0
    for raw_line in text.splitlines():
        line = ANSI_RE.sub("", raw_line).strip()
        if not pending:
            start = line.find("{")
            if start < 0 or not any(
                marker in line.lower() for marker in ("loss", "train", "val")
            ):
                continue
            pending = line[start:]
            depth = pending.count("{") - pending.count("}")
        else:
            pending += " " + line
            depth += line.count("{") - line.count("}")
        if depth <= 0:
            yield pending
            pending = ""
            depth = 0


def parse_log(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    fallback_step = 0
    for fragment in mapping_fragments(text):
        payload = None
        try:
            payload = ast.literal_eval(fragment[: fragment.rfind("}") + 1])
        except (SyntaxError, ValueError):
            pass

        if isinstance(payload, dict):
            flat = flatten(payload)
            step = int(flat.get("global_step", fallback_step))
            epoch = int(flat.get("epoch", -1))
            losses = {
                key: float(value)
                for key, value in flat.items()
                if "loss" in key.lower()
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            }
        else:
            step_match = STEP_RE.search(fragment)
            epoch_match = EPOCH_RE.search(fragment)
            step = int(step_match.group("step")) if step_match else fallback_step
            epoch = int(epoch_match.group("epoch")) if epoch_match else -1
            losses = {
                match.group("key").replace(".", "/"): float(match.group("value"))
                for match in PAIR_RE.finditer(fragment)
            }

        split = (
            "val"
            if any(key.startswith(("val/", "validation/")) for key in losses)
            else "train"
        )
        for metric, value in losses.items():
            rows.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "split": split,
                    "metric": metric,
                    "value": value,
                }
            )
        if losses:
            fallback_step = max(fallback_step + 1, step + 1)
    return rows


def moving_average(values: list[float], window: int) -> list[float]:
    queue: deque[float] = deque()
    total = 0.0
    result = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.popleft()
        result.append(total / len(queue))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="train_online_*.log produced by tee")
    parser.add_argument("--output", type=Path, help="PNG path; defaults beside log")
    parser.add_argument("--csv", type=Path, help="CSV path; defaults beside PNG")
    parser.add_argument(
        "--smooth", type=int, default=20, help="train moving-average window"
    )
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be >= 1")

    rows = parse_log(args.log)
    if not rows:
        raise SystemExit(
            "No finite loss metrics found. Check that this is a successful training log."
        )
    output = args.output or args.log.with_name(f"{args.log.stem}_loss.png")
    csv_path = args.csv or output.with_suffix(".csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("step", "epoch", "split", "metric", "value")
        )
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            f"CSV written to {csv_path}, but PNG requires: pip install matplotlib"
        ) from exc

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["metric"]].append(row)
    fig, axis = plt.subplots(figsize=(12, 7), dpi=140)
    for metric, points in sorted(grouped.items()):
        points.sort(key=lambda item: item["step"])
        steps = [item["step"] for item in points]
        values = [item["value"] for item in points]
        is_val = metric.startswith(("val/", "validation/"))
        if is_val or len(values) < 3:
            axis.plot(steps, values, marker="o", linewidth=1.8, label=metric)
        else:
            axis.plot(steps, values, alpha=0.18, linewidth=0.8)
            axis.plot(
                steps,
                moving_average(values, args.smooth),
                linewidth=1.8,
                label=f"{metric} (MA{args.smooth})",
            )
    axis.set(title="DSpark training loss", xlabel="Global step", ylabel="Loss")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    print(f"Parsed {len(rows)} loss points")
    print(f"CSV: {csv_path}")
    print(f"PNG: {output}")


if __name__ == "__main__":
    main()
