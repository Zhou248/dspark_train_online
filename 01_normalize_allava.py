#!/usr/bin/env python3
"""Stream ALLaVA-4V JSON into msModelSpec multimodal conversations JSONL."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_json_array(path: Path, chunk_size: int = 4 * 1024 * 1024) -> Iterator[dict]:
    """Read a large top-level JSON array without loading the full file."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buffer = ""
        started = False
        eof = False
        while True:
            if not eof and len(buffer) < chunk_size:
                chunk = stream.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            buffer = buffer.lstrip()
            if not started:
                if not buffer:
                    if eof:
                        raise ValueError(f"Empty JSON file: {path}")
                    continue
                if buffer[0] != "[":
                    raise ValueError(f"Expected a top-level JSON array in {path}")
                buffer = buffer[1:]
                started = True

            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return

            try:
                item, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = stream.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue

            if not isinstance(item, dict):
                raise TypeError(f"Expected each ALLaVA row to be an object, got {type(item)}")
            yield item
            buffer = buffer[offset:]


def media_values(item: dict[str, Any]) -> list[str]:
    value = item.get("images", item.get("image", item.get("image_path", [])))
    if not value:
        return []
    if not isinstance(value, list):
        value = [value]
    paths = []
    for entry in value:
        if isinstance(entry, str):
            paths.append(entry)
        elif isinstance(entry, dict):
            candidate = entry.get("path") or entry.get("file_name") or entry.get("image")
            if candidate:
                paths.append(str(candidate))
    return paths


def resolve_media(source_dir: Path, values: list[str]) -> list[Path]:
    resolved = []
    for value in values:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = source_dir / path
        resolved.append(path.resolve())
    return resolved


def normalize_part(part: Any, source_dir: Path) -> dict[str, Any]:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return {"type": "text", "text": str(part)}

    kind = part.get("type", "text")
    if kind == "text":
        return {"type": "text", "text": str(part.get("text", part.get("value", "")))}
    if kind in {"image", "video", "audio"}:
        value = part.get("path") or part.get("url")
        if not value:
            raise ValueError(f"Media part has no path/url: {part}")
        if part.get("url") and not part.get("path"):
            return {"type": kind, "url": str(value)}
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = source_dir / path
        return {"type": kind, "path": str(path.resolve())}
    raise ValueError(f"Unsupported content type: {kind}")


def normalize_row(item: dict[str, Any], source_dir: Path) -> dict[str, Any]:
    raw_conversation = item.get("conversations", item.get("messages"))
    if not isinstance(raw_conversation, list) or not raw_conversation:
        raise ValueError("Missing non-empty conversations/messages")

    top_level_media = resolve_media(source_dir, media_values(item))
    media_injected = False
    conversations = []
    for turn in raw_conversation:
        role = str(turn.get("from", turn.get("role", "")))
        raw_value = turn.get("value", turn.get("content", ""))
        is_user = role in {"human", "user"}

        if isinstance(raw_value, list):
            parts = [normalize_part(part, source_dir) for part in raw_value]
            if any(part["type"] == "image" for part in parts):
                media_injected = True
            elif is_user and not media_injected and top_level_media:
                parts = [
                    *(
                        {"type": "image", "path": str(path)}
                        for path in top_level_media
                    ),
                    *parts,
                ]
                media_injected = True
        else:
            text = str(raw_value).replace("<image>", "").strip()
            parts = []
            if is_user and not media_injected and top_level_media:
                parts.extend({"type": "image", "path": str(path)} for path in top_level_media)
                media_injected = True
            parts.append({"type": "text", "text": text})

        conversations.append({"from": role, "value": parts})

    if top_level_media and not media_injected:
        raise ValueError("Row declares an image but has no user turn to attach it to")
    return {"conversations": conversations}


def row_media_paths(row: dict[str, Any]) -> Iterator[Path]:
    for turn in row["conversations"]:
        for part in turn["value"]:
            if part.get("type") in {"image", "video", "audio"} and part.get("path"):
                yield Path(part["path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-invalid", action="store_true")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    media_count = 0
    with output.open("w", encoding="utf-8") as destination:
        for source_index, item in enumerate(iter_json_array(source)):
            if args.max_samples is not None and written >= args.max_samples:
                break
            try:
                row = normalize_row(item, source.parent)
                paths = list(row_media_paths(row))
                missing = [str(path) for path in paths if not path.exists()]
                if missing:
                    raise FileNotFoundError(f"Missing media: {missing[:3]}")
                if not paths:
                    raise ValueError("No media found in multimodal ALLaVA row")
            except Exception as exc:  # noqa: BLE001 - optionally skip bad source rows
                if not args.skip_invalid:
                    raise RuntimeError(f"Invalid source row {source_index}: {exc}") from exc
                skipped += 1
                print(f"SKIP source_index={source_index}: {type(exc).__name__}: {exc}")
                continue

            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            media_count += len(paths)
            if written % 1000 == 0:
                print(f"written={written} skipped={skipped} media={media_count}")

    if written == 0:
        raise RuntimeError("No valid ALLaVA samples were written")
    print(f"DONE output={output} written={written} skipped={skipped} media={media_count}")


if __name__ == "__main__":
    main()
