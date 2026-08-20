#!/usr/bin/env python3
"""Stream ALLaVA-4V JSON into msModelSpec multimodal conversations JSONL."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


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


def resize_image(
    source: Path,
    destination: Path,
    max_pixels: int,
    max_side: int,
) -> tuple[Path, bool, tuple[int, int], tuple[int, int]]:
    """Copy an oversized image to a bounded RGB JPEG without touching source."""
    with Image.open(source) as raw_image:
        metadata_invalid = False
        try:
            image = ImageOps.exif_transpose(raw_image)
        except (OSError, SyntaxError, UnicodeError, ValueError) as exc:
            # Some ALLaVA images have valid pixels but malformed EXIF/XMP.
            # Preserve the sample by dropping metadata and writing a clean copy.
            metadata_invalid = True
            warnings.warn(
                f"Ignoring corrupt EXIF/XMP metadata in {source}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            image = raw_image.copy()
        original_size = image.size
        width, height = original_size
        scale = min(
            1.0,
            math.sqrt(max_pixels / max(width * height, 1)),
            max_side / max(width, height, 1),
        )
        if scale >= 1.0 and not metadata_invalid:
            return source, False, original_size, original_size

        if scale < 1.0:
            target_size = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        else:
            target_size = original_size
        if image.mode != "RGB":
            if "A" in image.getbands():
                background = Image.new("RGB", image.size, "white")
                background.paste(image, mask=image.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="JPEG", quality=95, optimize=True)
        return destination, scale < 1.0, original_size, target_size


def bound_row_images(
    row: dict[str, Any],
    output_dir: Path,
    source_index: int,
    max_pixels: int,
    max_side: int,
) -> tuple[int, list[str]]:
    """Rewrite oversized local image parts to bounded copies."""
    resized = 0
    changes = []
    image_number = 0
    for turn in row["conversations"]:
        for part in turn["value"]:
            if part.get("type") != "image" or not part.get("path"):
                continue
            source = Path(part["path"])
            destination = output_dir / f"sample_{source_index:08d}_{image_number}.jpg"
            bounded, changed, before, after = resize_image(
                source, destination, max_pixels, max_side
            )
            part["path"] = str(bounded)
            if changed:
                resized += 1
                changes.append(f"{before[0]}x{before[1]}->{after[0]}x{after[1]}")
            image_number += 1
    return resized, changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--media-output-dir", required=True)
    parser.add_argument("--max-image-pixels", type=int, default=1_048_576)
    parser.add_argument("--max-image-side", type=int, default=2048)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    media_output_dir = Path(args.media_output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    media_output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    media_count = 0
    resized_count = 0
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
                resized, changes = bound_row_images(
                    row,
                    media_output_dir,
                    source_index,
                    args.max_image_pixels,
                    args.max_image_side,
                )
                resized_count += resized
                if changes and resized_count <= 20:
                    print(f"RESIZE source_index={source_index}: {', '.join(changes)}")
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
                print(
                    f"written={written} skipped={skipped} media={media_count} "
                    f"resized={resized_count}"
                )

    if written == 0:
        raise RuntimeError("No valid ALLaVA samples were written")
    print(
        f"DONE output={output} written={written} skipped={skipped} "
        f"media={media_count} resized={resized_count}"
    )


if __name__ == "__main__":
    main()
