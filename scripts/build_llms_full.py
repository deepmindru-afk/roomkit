"""Assemble docs/c7/ topic pages into llms-full.txt.

Reads topic pages in order, strips h1 headers, prepends llms.txt header,
joins with ``---`` separators, and writes llms-full.txt.

Run with:
    uv run python scripts/build_llms_full.py
"""

from __future__ import annotations

import re
from pathlib import Path

# Ordered list of topic pages — determines section order in llms-full.txt
PAGES = [
    "overview.md",
    "installation.md",
    "quickstart.md",
    "architecture.md",
    "rooms-and-channels.md",
    "hooks.md",
    "ai-channels.md",
    "ollama-provider.md",
    "acp.md",
    "image-generation.md",
    "voice-channels.md",
    "voice-pipeline.md",
    "realtime-voice.md",
    "video.md",
    "conference.md",
    "orchestration.md",
    "transport-providers.md",
    "identity-and-realtime.md",
    "resilience.md",
    "storage.md",
    "testing.md",
    "api-reference.md",
]

HEADER = """\
# RoomKit

> RoomKit is a pure async Python library for building multi-channel conversation
> systems. It provides room-based abstractions for managing conversations across
> SMS, Email, Voice, WebSocket, AI, and other channels with pluggable storage,
> identity resolution, hooks, and realtime events.
> Python 3.12+, Pydantic 2.x, fully typed, no required dependency beyond Pydantic and PyYAML.

"""


def strip_h1(content: str) -> str:
    """Remove leading h1 header (first ``# Title`` line) from a page."""
    return re.sub(r"^# .+\n+", "", content, count=1)


REPO_ROOT = Path(__file__).resolve().parent.parent


def assemble(repo_root: Path = REPO_ROOT) -> str:
    """Return llms-full.txt as the docs/c7 topic pages assemble it."""
    c7_dir = repo_root / "docs" / "c7"
    sections: list[str] = []
    for page in PAGES:
        path = c7_dir / page
        if not path.exists():
            msg = f"Missing topic page: {path}"
            raise FileNotFoundError(msg)
        content = path.read_text(encoding="utf-8").strip()
        sections.append(strip_h1(content))

    return HEADER + "\n---\n\n".join(sections) + "\n"


def main() -> None:
    output_path = REPO_ROOT / "llms-full.txt"
    assembled = assemble()
    output_path.write_text(assembled, encoding="utf-8")
    line_count = assembled.count("\n") + 1
    print(f"Wrote {output_path} ({line_count} lines)")


if __name__ == "__main__":
    main()
