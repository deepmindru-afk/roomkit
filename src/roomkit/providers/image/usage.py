"""Normalize observed counters without manufacturing missing measurements."""

from __future__ import annotations

from typing import Any


def openai_image_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    result: dict[str, int] = {}
    total = getattr(usage, "input_tokens", None)
    details = getattr(usage, "input_tokens_details", None)
    image = getattr(details, "image_tokens", None)
    text = getattr(details, "text_tokens", None)
    if image is not None:
        result["input_image_tokens"] = int(image)
    if text is not None:
        result["input_tokens"] = int(text)
    elif total is not None and image is not None:
        result["input_tokens"] = max(int(total) - int(image), 0)
    elif total is not None:
        result["unclassified_input_tokens"] = int(total)
    output = getattr(usage, "output_tokens", None)
    if output is not None:
        result.update(output_tokens=0, output_image_tokens=int(output))
    return result


def gemini_image_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    result: dict[str, int] = {}
    for direction in ("input", "output"):
        total = getattr(usage, f"total_{direction}_tokens", None)
        breakdown = getattr(usage, f"{direction}_tokens_by_modality", None)
        if breakdown is not None:
            counts: dict[str, int] = {}
            for entry in breakdown:
                if getattr(entry, "tokens", None) is not None:
                    modality = str(getattr(entry, "modality", "")).lower()
                    counts[modality] = counts.get(modality, 0) + int(entry.tokens)
            image = counts.get("image", 0)
            result[f"{direction}_image_tokens"] = image
            if total is not None:
                result[f"{direction}_tokens"] = max(int(total) - image, 0)
            elif "text" in counts:
                result[f"{direction}_tokens"] = counts["text"]
        elif total is not None:
            result[f"unclassified_{direction}_tokens"] = int(total)
    thoughts = getattr(usage, "total_thought_tokens", None)
    if thoughts is not None:
        result["output_tokens"] = result.get("output_tokens", 0) + int(thoughts)
    cached = getattr(usage, "total_cached_tokens", None)
    if cached:
        # The input modality totals include these tokens. Without their modality
        # split, a consumer cannot derive disjoint cached/uncached image rates.
        result["unclassified_cached_tokens"] = int(cached)
    return result
