#!/usr/bin/env python3
"""Patch legacy vLLM-Ascend DSpark reduced-vocabulary Markov bias handling."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

LEGACY = "            logits[:, idx].add_(logits_bias)"
MARKER = "# DSpark reduced-vocab compatibility: scatter bias into target vocab."
REPLACEMENT = """            # DSpark reduced-vocab compatibility: scatter bias into target vocab.
            if logits[:, idx].shape[-1] != logits_bias.shape[-1]:
                d2t_offset = getattr(
                    self.model, "draft_id_to_target_id", None
                )
                if d2t_offset is None:
                    d2t_offset = getattr(self.model, "d2t", None)
                if d2t_offset is None:
                    raise RuntimeError(
                        "DSpark logits/Markov-bias vocab mismatch but the "
                        "checkpoint has no draft_id_to_target_id mapping: "
                        f"{logits[:, idx].shape[-1]} vs "
                        f"{logits_bias.shape[-1]}"
                    )
                if d2t_offset.numel() != logits_bias.shape[-1]:
                    raise RuntimeError(
                        "DSpark d2t mapping size does not match Markov bias: "
                        f"{d2t_offset.numel()} vs {logits_bias.shape[-1]}"
                    )
                draft_ids = torch.arange(
                    d2t_offset.numel(),
                    device=d2t_offset.device,
                    dtype=d2t_offset.dtype,
                )
                target_ids = draft_ids + d2t_offset
                if (
                    target_ids.min().item() < 0
                    or target_ids.max().item() >= logits[:, idx].shape[-1]
                ):
                    raise RuntimeError(
                        "DSpark d2t mapping contains an out-of-range target id"
                    )
                expanded_bias = torch.full_like(
                    logits[:, idx], float("-inf")
                )
                expanded_bias.index_copy_(
                    -1, target_ids.to(torch.long), logits_bias
                )
                logits[:, idx].add_(expanded_bias)
            else:
                logits[:, idx].add_(logits_bias)"""


def discover_file() -> Path:
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(
            "Cannot import vllm_ascend. Activate the vLLM-Ascend environment "
            "or pass --file explicitly."
        )
    root = Path(next(iter(spec.submodule_search_locations)))
    return root / "spec_decode" / "llm_base_proposer.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        help="Explicit llm_base_proposer.py path; defaults to imported package",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report whether the compatibility patch is present",
    )
    args = parser.parse_args()
    path = (args.file or discover_file()).resolve()
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"Already patched: {path}")
        return
    matches = text.count(LEGACY)
    if args.check:
        raise SystemExit(
            f"Patch missing: {path} (legacy match count={matches})"
        )
    if matches != 1:
        raise SystemExit(
            "Refusing to patch because the expected legacy line did not occur "
            f"exactly once: {path} (count={matches}). The installed version "
            "needs a version-specific review."
        )

    patched_text = text.replace(LEGACY, REPLACEMENT)
    compile(patched_text, str(path), "exec")

    backup = path.with_suffix(path.suffix + ".bak_dspark_vocab")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(patched_text, encoding="utf-8")
    print(f"Patched: {path}")
    print(f"Backup:  {backup}")
    print("Restart every vLLM process before testing.")


if __name__ == "__main__":
    main()
