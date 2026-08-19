#!/usr/bin/env python3
"""Build the text-only 1-D Qwen3 decoder config used by DSpark."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoConfig, Qwen3Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--sliding-window", type=int, default=2048)
    args = parser.parse_args()

    root = AutoConfig.from_pretrained(args.target_model, trust_remote_code=True)
    text = getattr(root, "text_config", root)
    rope_parameters = getattr(text, "rope_parameters", {}) or {}
    rope_theta = float(
        getattr(text, "rope_theta", None)
        or rope_parameters.get("rope_theta", 10_000_000.0)
    )
    hidden_size = int(text.hidden_size)
    config = Qwen3Config(
        vocab_size=int(text.vocab_size),
        hidden_size=hidden_size,
        intermediate_size=int(getattr(text, "intermediate_size", 3 * hidden_size)),
        num_hidden_layers=args.num_layers,
        num_attention_heads=int(text.num_attention_heads),
        num_key_value_heads=int(text.num_key_value_heads),
        head_dim=int(getattr(text, "head_dim", hidden_size // text.num_attention_heads)),
        hidden_act=getattr(text, "hidden_act", "silu"),
        max_position_embeddings=int(text.max_position_embeddings),
        initializer_range=float(getattr(text, "initializer_range", 0.02)),
        rms_norm_eps=float(getattr(text, "rms_norm_eps", 1e-6)),
        attention_bias=bool(getattr(text, "attention_bias", False)),
        attention_dropout=float(getattr(text, "attention_dropout", 0.0)),
        tie_word_embeddings=False,
        use_cache=True,
        sliding_window=args.sliding_window,
        use_sliding_window=True,
        layer_types=["sliding_attention"] * args.num_layers,
        # The multimodal target uses MRoPE, but the DSpark draft is a text-only
        # Qwen3 decoder. Do not copy mrope_section/partial_rotary_factor.
        rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output)
    print(f"saved draft config: {output / 'config.json'}")


if __name__ == "__main__":
    main()

